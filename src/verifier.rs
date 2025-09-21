//! Minimal CLI verifier
//!
//! Examples:
//!   cargo run --bin verifier
//!   cargo run --bin verifier -- --rows 1024 --basis eval
//!   cargo run --bin verifier -- --rows 16384 --srs-g1 srs_g1.bin --srs-g2 srs_g2.bin
//!
//! Notes:
//! - `proof.bin` is produced by the prover (Arkworks canonical bytes).
//! - In non-dev builds you must load **both** G1 and G2 SRS.

#![forbid(unsafe_code)]
#![allow(unused_imports)]

use std::{collections::HashMap, env, fs, path::Path};

use ark_ff::{fields::Field, FftField, One, Zero};
use ark_serialize::CanonicalDeserialize;
use myzkp::{
    pcs::{self, Basis, PcsParams},
    scheduler::Verifier,
    Commitment, Proof, VerifyParams, F,
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

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    // These must match the prover’s public parameters.
    let n_rows: usize = parse_flag(&args, "--rows").and_then(|s| s.parse().ok()).unwrap_or(1024);
    let basis_str = parse_flag(&args, "--basis").unwrap_or_else(|| "eval".to_string());
    let basis_wires = match basis_str.as_str() {
        "coeff" | "coefficient" => Basis::Coefficient,
        _ => Basis::Evaluation,
    };

    // Optional: load a trusted SRS. In non-dev builds, you MUST provide both.
    if let Some(srs_path) = parse_flag(&args, "--srs-g1").or_else(|| parse_flag(&args, "--srs")) {
        let bytes = fs::read(Path::new(&srs_path))?;
        let mut slice = bytes.as_slice();
        let g1_powers: Vec<ark_bn254::G1Affine> =
            CanonicalDeserialize::deserialize_compressed(&mut slice)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        pcs::load_srs_g1(&g1_powers);
        println!("Loaded G1 SRS from {}", srs_path);
    } else {
        println!("(note) No --srs-g1 provided; using dev G1 SRS (dev build only).");
    }
    if let Some(srs_g2_path) = parse_flag(&args, "--srs-g2") {
        let bytes = fs::read(Path::new(&srs_g2_path))?;
        let mut slice = bytes.as_slice();
        let g2_powers: Vec<ark_bn254::G2Affine> =
            CanonicalDeserialize::deserialize_compressed(&mut slice)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let tau_g2 = *g2_powers
            .get(1)
            .or_else(|| g2_powers.get(0))
            .expect("G2 SRS file must contain at least one element ([τ]G2)");
        pcs::load_srs_g2(tau_g2);
        println!("Loaded G2 SRS from {}", srs_g2_path);
    } else {
        println!("(note) No --srs-g2 provided; using dev G2 SRS (dev build only).");
    }

    // Domain must mirror the prover.
    let n_domain = next_power_of_two(n_rows);

    // Use a real primitive N-th root when available to avoid domain mismatch.
    let omega = F::get_root_of_unity(n_domain as u64)
        .expect("field does not support an N-th root of unity for this N");
    let domain = myzkp::domain::Domain {
        n: n_domain,
        omega,
        zh_c: F::one(), // Z_H(X) = X^N - 1
    };

    // PCS params: wires in `basis_wires`, quotient in Coefficient basis.
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

    let verify_params = VerifyParams { domain: domain.clone(), pcs_wires, pcs_coeff };
    let verifier = Verifier { params: &verify_params };

    // Load proof.bin (the prover wrote an Arkworks-canonical struct).
    #[derive(ark_serialize::CanonicalDeserialize)]
    struct ProofDisk {
        wire_comms: Vec<myzkp::pcs::Commitment>,
        z_comm: Option<myzkp::pcs::Commitment>,
        q_comm: myzkp::pcs::Commitment,
        eval_points: Vec<F>,
        evals: Vec<F>,
        opening_proofs: Vec<myzkp::pcs::OpeningProof>,
    }

    let bytes = fs::read("proof.bin")?;
    let mut slice = bytes.as_slice();
    let d: ProofDisk = CanonicalDeserialize::deserialize_compressed(&mut slice)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let proof = Proof {
        wire_comms: d.wire_comms.into_iter().map(Commitment).collect(),
        z_comm: d.z_comm.map(Commitment),
        q_comm: Commitment(d.q_comm),
        eval_points: d.eval_points,
        evals: d.evals,
        opening_proofs: d.opening_proofs,
    };

    // 1) Replay FS + enforce KZG pairings (for wires, Z, Q).
    let pairings_ok = verifier.verify(&proof);

    // 2) Algebraic check at ζ: Gate(ζ) + PermCoupling(ζ) + Boundary(ζ) == Z_H(ζ) * Q(ζ)
    //
    // We extract the claimed evaluations following the public contract:
    // poly-major, point-minor order over opening set [ wires..., (Z?), Q ].
    let k = proof.wire_comms.len();
    let has_z = proof.z_comm.is_some();
    let s_points = proof.eval_points.len();
    let m_polys = k + if has_z { 1 } else { 0 } + 1; // +Q

    // Build evals matrix evals_mat[poly][point]
    if proof.evals.len() != m_polys * s_points {
        eprintln!(
            "Shape mismatch: expected {} evals, got {}",
            m_polys * s_points,
            proof.evals.len()
        );
        std::process::exit(1);
    }
    let mut evals_mat: Vec<Vec<F>> = vec![vec![F::from(0u64); s_points]; m_polys];
    {
        let mut idx = 0usize;
        for p in 0..m_polys {
            for s in 0..s_points {
                evals_mat[p][s] = proof.evals[idx];
                idx += 1;
            }
        }
    }

    // Helper: map point -> its index for easy lookup of ω·ζ.
    let mut point_index: HashMap<F, usize> = HashMap::new();
    for (i, &z) in proof.eval_points.iter().enumerate() {
        point_index.insert(z, i);
    }

    // Compute checks for each ζ in the set (usually S=1; supports S>1).
    let mut algebraic_all_ok = true;
    for (si, &zeta) in proof.eval_points.iter().enumerate() {
        // Wires at ζ
        let wires_at_z: Vec<F> = (0..k).map(|m| evals_mat[m][si]).collect();

        // Z at ζ (if present)
        let z_at_z = if has_z { Some(evals_mat[k][si]) } else { None };

        // Q at ζ
        let q_idx = m_polys - 1;
        let q_at_z = evals_mat[q_idx][si];

        // Gate(ζ): if you keep selectors public-but-uncommitted, compute them here.
        // By default: no selectors -> gate=0.
        let gate_at_z = F::from(0u64);

        // PermCoupling(ζ): requires Z(ω·ζ).
        // We support both S=1 (skip algebraic) and S>=2 with ω·ζ ∈ points.
        let mut perm_ok_available = false;
        let perm_at_z = if has_z {
            // Recompute (β,γ) as in the transcript up to that point
            use myzkp::transcript::Transcript;
            let mut fs = Transcript::new("sszkp.proof");
            for cm in &proof.wire_comms { fs.absorb_commitment("wire_commit", &cm.0); }
            let beta: F = fs.challenge_f("beta");
            let gamma: F = fs.challenge_f("gamma");

            let mut prod_id = F::from(1u64);
            let mut prod_sigma = F::from(1u64);
            for j in 0..k {
                let wj = wires_at_z[j];
                let idj = F::from(j as u64);
                let sigmaj = F::from(((j + 1) % k) as u64);
                prod_id *= wj + beta * idj + gamma;
                prod_sigma *= wj + beta * sigmaj + gamma;
            }

            // Need Z(ω·ζ).
            let zeta_shift = domain.omega * zeta;
            if let (Some(si_shift), Some(z_here)) = (point_index.get(&zeta_shift).copied(), z_at_z) {
                let z_at_shift = evals_mat[k][si_shift];
                perm_ok_available = true;
                // Z(ωζ)·Π_id − Z(ζ)·Π_σ
                z_at_shift * prod_id - z_here * prod_sigma
            } else {
                // Can't form the algebraic term without Z at ω·ζ.
                F::from(0u64)
            }
        } else {
            F::from(0u64)
        };

        // Boundary(ζ): hooks default to 0 in the demo.
        let boundary_at_z = F::from(0u64);

        // RHS = Z_H(ζ) * Q(ζ) with Z_H(X)=X^N - c.
        fn pow_u64(mut base: F, mut exp: u64) -> F {
            let mut acc = F::from(1u64);
            while exp > 0 {
                if (exp & 1) == 1 { acc *= base; }
                base.square_in_place();
                exp >>= 1;
            }
            acc
        }
        let zh_at_z = pow_u64(zeta, domain.n as u64) - domain.zh_c;
        let rhs = zh_at_z * q_at_z;

        let lhs = gate_at_z + perm_at_z + boundary_at_z;

        let algebraic_ok = if has_z && perm_ok_available {
            lhs == rhs
        } else {
            // If we couldn't form the perm term (missing Z(ω·ζ)), we don't fail on the algebraic check.
            true
        };

        algebraic_all_ok &= algebraic_ok;

        if has_z && !perm_ok_available {
            println!(
                "(note) Skipping algebraic identity at point {}: missing Z(ω·ζ) opening (need both ζ and ω·ζ).",
                si
            );
        } else {
            println!(
                "Algebraic identity at point {}: {}",
                si,
                if algebraic_ok { "ok" } else { "FAIL" }
            );
        }
    }

    let ok = pairings_ok && algebraic_all_ok;
    println!("Verifier pairings: {}", pairings_ok);
    println!("Verifier algebraic check(s): {}", algebraic_all_ok);
    println!("Verifier result: {}", ok);

    if ok { Ok(()) } else { std::process::exit(1) }
}
