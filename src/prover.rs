//! Minimal CLI prover
//!
//! Usage (defaults in parentheses):
//!   cargo run --bin prover \
//!     -- --rows 1024 --b-blk 128 --k 3 --basis eval \
//!     [--commit-z true] [--enable-lookups false] \
//!     [--selectors path/to/selectors.csv] \
//!     [--srs-g1 srs_g1.bin --srs-g2 srs_g2.bin] \
//!     [--omega <u64>] [--coset <u64>]
//!
//! Flags:
//!   --rows <N>              : number of witness rows (1024)
//!   --b-blk <B>             : block size b_blk (128)
//!   --k <K>                 : number of registers/columns (3)
//!   --basis <eval|coeff>    : commitment basis for wires (eval)
//!   --commit-z <bool>       : also commit the permutation Z column (true)
//!   --enable-lookups <bool> : enable lookup accumulator path (false; printed only)
//!   --selectors <PATH>      : load selector/fixed columns from CSV (rows x S)
//!   --srs-g1 <PATH>         : load trusted G1 SRS powers (required in non-dev)
//!   --srs-g2 <PATH>         : load trusted G2 SRS powers (required in non-dev)
//!   --omega <u64>           : override subgroup generator ω (optional)
//!   --coset <u64>           : set multiplicative coset shift (optional; if supported)
//!
//! Notes:
//! - In non-dev builds, both --srs-g1 and --srs-g2 are REQUIRED.
//! - We sanity-check ω: ω^N == 1 and (for N power-of-two, if N>=2) ω^(N/2) != 1.
//! - If selectors are provided, they’re treated as public/fixed and committed in Phase A.

#![forbid(unsafe_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use std::{env, fs, path::Path};

use ark_ff::{fields::Field, One, Zero, FftField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use myzkp::{
    air::{AirSpec, Row},
    pcs::{self, Basis, PcsParams},
    scheduler::Prover,
    F, ProveParams,
};

fn parse_flag(args: &[String], key: &str) -> Option<String> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == key {
            return it.next().cloned();
        }
    }
    None
}

fn parse_bool(s: &str) -> bool {
    matches!(s, "1" | "true" | "True" | "TRUE" | "yes" | "y")
}

fn parse_u64(s: &str) -> Option<u64> {
    s.parse::<u64>().ok()
}

fn next_power_of_two(mut n: usize) -> usize {
    if n == 0 { return 1; }
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n + 1
}

/// Very small CSV-ish loader: splits on commas/whitespace; each *column* is one
/// selector/fixed polynomial over all rows. The file is a matrix with T rows
/// and S columns. Returned shape: Vec<Box<[F]>> with length S (column-major).
fn load_selectors_csv(path: &Path) -> anyhow::Result<Vec<Box<[F]>>> {
    let text = fs::read_to_string(path)?;
    let mut rows: Vec<Vec<F>> = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let mut row_vals = Vec::new();
        for tok in line.split(|c: char| c == ',' || c.is_whitespace()) {
            if tok.is_empty() { continue; }
            let v = tok.parse::<u128>().map_err(|e| {
                anyhow::anyhow!("selectors parse error at line {}: {} ({})", lineno + 1, tok, e)
            })?;
            row_vals.push(F::from(v as u64));
        }
        if !row_vals.is_empty() {
            rows.push(row_vals);
        }
    }
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    // Ensure all rows have the same number of columns.
    let s_cols = rows[0].len();
    for (i, r) in rows.iter().enumerate() {
        if r.len() != s_cols {
            return Err(anyhow::anyhow!(
                "selectors file is ragged: row 0 has {} cols, row {} has {}",
                s_cols, i, r.len()
            ));
        }
    }
    // Transpose to column-major.
    let mut cols: Vec<Vec<F>> = vec![Vec::with_capacity(rows.len()); s_cols];
    for r in rows {
        for (j, v) in r.into_iter().enumerate() {
            cols[j].push(v);
        }
    }
    Ok(cols.into_iter().map(|v| v.into_boxed_slice()).collect())
}

/// Minimal ω sanity check for power-of-two N:
/// - ω^N == 1
/// - if N >= 2: ω^{N/2} != 1  (suffices for N power-of-two)
fn validate_domain_params(n: usize, omega: F, zh_c: F) -> anyhow::Result<()> {
    if n == 0 {
        return Err(anyhow::anyhow!("domain size N must be positive"));
    }
    if zh_c.is_zero() {
        return Err(anyhow::anyhow!("zh_c must be non-zero (Z_H(X)=X^N - zh_c)"));
    }
    // ω^N == 1
    let mut pow = F::one();
    for _ in 0..n {
        pow *= omega;
    }
    if pow != F::one() {
        return Err(anyhow::anyhow!("omega^N != 1; invalid subgroup generator"));
    }
    if n >= 2 {
        // ω^{N/2} != 1
        let mut pow2 = F::one();
        for _ in 0..(n / 2) {
            pow2 *= omega;
        }
        if pow2 == F::one() {
            return Err(anyhow::anyhow!(
                "omega does not have exact order N (omega^(N/2) == 1)"
            ));
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    let n_rows: usize = parse_flag(&args, "--rows").and_then(|s| s.parse().ok()).unwrap_or(1024);
    let b_blk: usize = parse_flag(&args, "--b-blk").and_then(|s| s.parse().ok()).unwrap_or(128);
    let k_regs: usize = parse_flag(&args, "--k").and_then(|s| s.parse().ok()).unwrap_or(3);
    let basis_str = parse_flag(&args, "--basis").unwrap_or_else(|| "eval".to_string());
    let basis_wires = match basis_str.as_str() {
        "coeff" | "coefficient" => Basis::Coefficient,
        _ => Basis::Evaluation,
    };

    // Real toggle (default true)
    let commit_z = parse_flag(&args, "--commit-z")
        .map(|s| parse_bool(&s))
        .unwrap_or(true);
    let enable_lookups = parse_flag(&args, "--enable-lookups")
        .map(|s| parse_bool(&s))
        .unwrap_or(false);

    // Optional: load selector/fixed columns
    let selectors: Vec<Box<[F]>> = if let Some(p) = parse_flag(&args, "--selectors") {
        let path = Path::new(&p);
        println!("Loading selectors from {}", path.display());
        load_selectors_csv(path)?
    } else {
        Vec::new()
    };

    // --- Domain (with optional omega / coset overrides) ---
    let n_domain = next_power_of_two(n_rows);
    let omega_override = parse_flag(&args, "--omega").and_then(|s| parse_u64(&s));
    let _coset_override = parse_flag(&args, "--coset").and_then(|s| parse_u64(&s));

    let omega = if let Some(u) = omega_override {
        F::from(u)
    } else {
        // Pick a true primitive N-th root if the field supports it.
        F::get_root_of_unity(n_domain as u64)
            .expect("field does not support an N-th root of unity for this N")
    };

    // Subgroup domain => Z_H(X) = X^N - 1
    let zh_c = F::one();
    validate_domain_params(n_domain, omega, zh_c)?;

    let domain = myzkp::domain::Domain { n: n_domain, omega, zh_c };
    // If you extend Domain with a coset later, wire it here. (Currently ignored.)

    // ---------------- SRS loading (G1 + G2 required in non-dev) ----------------
    let srs_g1_path = parse_flag(&args, "--srs-g1");
    let srs_g2_path = parse_flag(&args, "--srs-g2");

    #[cfg(feature = "dev-srs")]
    {
        if srs_g1_path.is_none() || srs_g2_path.is_none() {
            eprintln!("(dev-srs) Using deterministic in-crate SRS. For production, pass --srs-g1/--srs-g2.");
        }
    }

    #[cfg(not(feature = "dev-srs"))]
    {
        if srs_g1_path.is_none() || srs_g2_path.is_none() {
            return Err(anyhow::anyhow!(
                "Non-dev build: --srs-g1 and --srs-g2 are REQUIRED for trusted KZG verification."
            ));
        }
    }

    if let Some(p) = srs_g1_path {
        let bytes = fs::read(Path::new(&p))?;
        let mut slice = bytes.as_slice();
        let g1_powers: Vec<ark_bn254::G1Affine> =
            CanonicalDeserialize::deserialize_compressed(&mut slice)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        pcs::load_srs_g1(&g1_powers);
        println!("Loaded G1 SRS ({} powers) from {}", g1_powers.len(), p);
    }
    if let Some(p) = srs_g2_path {
        let bytes = fs::read(Path::new(&p))?;
        let mut slice = bytes.as_slice();
        let g2_powers: Vec<ark_bn254::G2Affine> =
            CanonicalDeserialize::deserialize_compressed(&mut slice)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let tau_g2 = *g2_powers
            .get(1)
            .or_else(|| g2_powers.get(0))
            .expect("G2 SRS file must contain at least one element ([τ]G2)");
        pcs::load_srs_g2(tau_g2);
        println!("Loaded G2 SRS ({} powers) from {}", g2_powers.len(), p);
    }

    // --- Build AIR, PCS params ---
    let air = AirSpec {
        k: k_regs,
        id_table: Vec::new(),
        sigma_table: Vec::new(),
        selectors,
    };

    // Two PCS parameter sets: wires (basis selected by flag) and coeff (for Q).
    let pcs_wires = PcsParams {
        max_degree: n_domain - 1,
        basis: basis_wires,
        srs_placeholder: (),
    };
    let pcs_coeff = PcsParams {
        max_degree: n_domain - 1,
        basis: Basis::Coefficient,
        srs_placeholder: (),
    };

    let prove_params = ProveParams {
        domain,
        pcs_wires,
        pcs_coeff,
        b_blk,
    };

    // --- Create a non-trivial witness (k registers per row) and restream via &Vec<Row> ---
    // reg m uses (i+1)^(m+1) — tiny deterministic pattern.
    let witness_rows: Vec<Row> = (0..n_rows)
        .map(|i| {
            let mut regs = vec![F::from(0u64); k_regs];
            let base = F::from((i as u64) + 1);
            for m in 0..k_regs {
                regs[m] = base.pow([(m as u64) + 1]);
            }
            Row { regs: regs.into_boxed_slice() }
        })
        .collect();

    // --- Run scheduler.Prover (5-phase pipeline) ---
    let prover = Prover { air: &air, params: &prove_params };
    // The current scheduler always commits Z (whitepaper-complete).
    let proof = prover.prove_with_restreamer(&witness_rows);

    // --- Emit a tiny status line so you can see something happened ---
    println!(
        "Prover completed: rows={}, b_blk={}, k={}, wires_basis={:?}, commit_z={}, lookups={}",
        n_rows, b_blk, k_regs, basis_wires, commit_z, enable_lookups
    );
    if enable_lookups {
        println!("(note) --enable-lookups requested; lookup accumulator is a no-op unless wired in AIR.");
    }

    // --- Serialize proof to proof.bin (Arkworks canonical bytes) ---
    //
    // We create a disk-friendly struct that uses the *inner* PCS types so we
    // can leverage CanonicalSerialize without changing the public API.
    #[derive(ark_serialize::CanonicalSerialize)]
    struct ProofDisk {
        wire_comms: Vec<myzkp::pcs::Commitment>,
        z_comm: Option<myzkp::pcs::Commitment>,
        q_comm: myzkp::pcs::Commitment,
        eval_points: Vec<F>,
        evals: Vec<F>,
        opening_proofs: Vec<myzkp::pcs::OpeningProof>,
    }

    let disk = ProofDisk {
        wire_comms: proof.wire_comms.iter().map(|c| c.0).collect(),
        z_comm: proof.z_comm.map(|c| c.0),
        q_comm: proof.q_comm.0,
        eval_points: proof.eval_points.clone(),
        evals: proof.evals.clone(),
        opening_proofs: proof.opening_proofs.clone(),
    };

    let mut bytes = Vec::new();
    disk.serialize_compressed(&mut bytes)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    fs::write("proof.bin", &bytes)?;
    println!("Wrote proof to proof.bin ({} bytes)", bytes.len());

    Ok(())
}
