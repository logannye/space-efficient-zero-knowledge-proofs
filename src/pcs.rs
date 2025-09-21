//! Polynomial Commitment Scheme (PCS) — KZG on BN254
//!
//! This module provides a **linear PCS interface** with a streaming-friendly
//! **Aggregator** that never materializes whole polynomials. It implements a
//! practical KZG-style commitment path over BN254 for the **commit** side,
//! with on-the-fly MSM aggregation that preserves the **aggregate-only**
//! Fiat–Shamir discipline from the whitepaper (Lemma 2 / Corollary 1).
//!
//! What’s new here vs the previous version:
//! - **Real KZG openings for wires/Z too.** Use
//!   [`open_eval_stream_at_points`] to accept **eval-basis** streams and
//!   internally restream **coefficients** (per-block IFFT) to build the witness
//!   `W(X) = (f(X) - f(ζ))/(X - ζ)`.
//! - **Always enforce pairings.** Add a **G2 SRS loader** (`load_srs_g2`) and
//!   require it in non-`dev-srs` builds (panic if missing).
//! - Keep [`open_at_points_with_coeffs`] as the canonical path when you can
//!   stream **coefficients** directly (e.g., for `Q`).
//!
//! Dev-mode behavior (`--features dev-srs`):
//! - Uses a deterministic τ to synthesize both G1 and G2 powers if not loaded.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use ark_bn254::{Bn254, Fr as ScalarField, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, Group};
use ark_ff::{Field, One, PrimeField, UniformRand, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rand::{rngs::StdRng, SeedableRng};
use std::sync::{Mutex, OnceLock};

use crate::{domain, F, G1};

/// Which basis the PCS expects when **committing**.
///
/// The commit path (`Aggregator`) validates the expected discipline and
/// performs the correct conversion (e.g., IFFT per block when fed evaluations).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// The caller provides **evaluation** slices.
    Evaluation,
    /// The caller provides **coefficient** slices.
    Coefficient,
}

/// Public parameters for the polynomial commitment scheme.
#[derive(Debug, Clone)]
pub struct PcsParams {
    /// Maximum supported degree **inclusive** for the committed polynomial.
    pub max_degree: usize,
    /// Which basis discipline callers must follow for the *commit* side.
    pub basis: Basis,
    /// Placeholder to keep the public API stable.
    pub srs_placeholder: (),
}

/// PCS commitment newtype (keeps API stable if backend changes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Commitment(pub G1);

/// KZG opening proof at a single point.
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct OpeningProof {
    /// Evaluation point `ζ` used to build the witness `W`.
    pub zeta: F,
    /// Claimed evaluation `f(ζ)` (the remainder from synthetic division).
    pub value: F,
    /// Commitment to the witness polynomial `W(X) = (f(X) - f(ζ)) / (X - ζ)`.
    pub witness_comm: Commitment,
}

// ----------------------- Internal SRS: G1 -----------------------

#[derive(Debug)]
struct SrsG1 {
    powers: Vec<G1Affine>,
    #[cfg(feature = "dev-srs")]
    tau: ScalarField,
}

impl SrsG1 {
    #[cfg(feature = "dev-srs")]
    fn new_dev() -> Self {
        let mut rng = StdRng::from_seed([42u8; 32]);
        let tau = ScalarField::rand(&mut rng);
        let mut s = SrsG1 { powers: Vec::new(), tau };
        s.ensure_len(1);
        s
    }

    fn ensure_len(&mut self, new_len: usize) {
        if self.powers.len() >= new_len {
            return;
        }
        #[cfg(feature = "dev-srs")]
        {
            let gen = G1Projective::generator();
            let current = self.powers.len();
            for idx in current..new_len {
                let gi = gen.mul_bigint(self.tau.pow([idx as u64]).into_bigint());
                self.powers.push(gi.into_affine());
            }
        }
        #[cfg(not(feature = "dev-srs"))]
        {
            assert!(
                self.powers.len() >= new_len,
                "G1 SRS insufficient; call load_srs_g1 with at least {} elements",
                new_len
            );
        }
    }

    #[inline]
    fn get_power(&self, idx: usize) -> G1Affine {
        self.powers[idx]
    }
}

fn srs_g1() -> &'static Mutex<SrsG1> {
    static SRS: OnceLock<Mutex<SrsG1>> = OnceLock::new();
    #[cfg(feature = "dev-srs")]
    {
        SRS.get_or_init(|| Mutex::new(SrsG1::new_dev()))
    }
    #[cfg(not(feature = "dev-srs"))]
    {
        SRS.get_or_init(|| Mutex::new(SrsG1 { powers: Vec::new() }))
    }
}

/// Load a trusted **G1** SRS (powers `[τ^0]G1, …, [τ^d]G1`) and return a template.
pub fn load_srs_g1(powers: &[G1Affine]) -> PcsParams {
    let mut guard = srs_g1().lock().expect("SRS mutex poisoned");
    guard.powers.clear();
    guard.powers.extend_from_slice(powers);
    drop(guard);

    PcsParams {
        max_degree: powers
            .len()
            .checked_sub(1)
            .expect("SRS must contain at least one power"),
        basis: Basis::Coefficient,
        srs_placeholder: (),
    }
}

// ----------------------- Internal SRS: G2 -----------------------

#[derive(Debug, Clone)]
struct SrsG2 {
    /// [τ]G2 — needed for verification (right-hand pairing key).
    tau_g2: Option<G2Affine>,
}

impl SrsG2 {
    #[cfg(feature = "dev-srs")]
    fn new_dev() -> Self {
        let tau = srs_g1().lock().expect("SRS mutex poisoned").tau;
        let g2_gen = <Bn254 as Pairing>::G2::generator();
        let tau_g2 = (G2Projective::from(g2_gen) * tau).into_affine();
        Self { tau_g2: Some(tau_g2) }
    }

    #[cfg(not(feature = "dev-srs"))]
    fn new_prod() -> Self {
        Self { tau_g2: None }
    }
}

fn srs_g2() -> &'static Mutex<SrsG2> {
    static SRS2: OnceLock<Mutex<SrsG2>> = OnceLock::new();
    #[cfg(feature = "dev-srs")]
    {
        SRS2.get_or_init(|| Mutex::new(SrsG2::new_dev()))
    }
    #[cfg(not(feature = "dev-srs"))]
    {
        SRS2.get_or_init(|| Mutex::new(SrsG2::new_prod()))
    }
}

/// Load **G2** SRS element `[τ]G2` for verification.
pub fn load_srs_g2(tau_g2: G2Affine) {
    let mut guard = srs_g2().lock().expect("SRS mutex poisoned");
    guard.tau_g2 = Some(tau_g2);
}

// ----------------------- Aggregator -----------------------

/// Linear aggregator over block slices.
pub struct Aggregator<'a> {
    pub(crate) pcs: &'a PcsParams,
    pub(crate) poly_id: &'static str,
    acc: G1Projective,
    cursor: usize,
}

impl<'a> Aggregator<'a> {
    /// Create a new aggregator for a single polynomial.
    pub fn new(pcs: &'a PcsParams, poly_id: &'static str) -> Self {
        Self { pcs, poly_id, acc: G1Projective::zero(), cursor: 0 }
    }

    /// Add a **domain-aligned evaluation slice** (Lagrange basis).
    pub fn add_block_evals(&mut self, d: &crate::domain::Domain, slice: &[F]) {
        assert!(
            matches!(self.pcs.basis, Basis::Evaluation),
            "add_block_evals called but PCS basis is {:?}",
            self.pcs.basis
        );
        let coeffs = domain::ifft_block_evals_to_coeffs(d, slice);
        self.add_block_coeffs_inner(&coeffs);
    }

    /// Add a **coefficient slice** (monomial basis).
    pub fn add_block_coeffs(&mut self, slice: &[F]) {
        assert!(
            matches!(self.pcs.basis, Basis::Coefficient),
            "add_block_coeffs called but PCS basis is {:?}",
            self.pcs.basis
        );
        self.add_block_coeffs_inner(slice);
    }

    /// Finalize and return the **aggregate** commitment.
    pub fn finalize(self) -> Commitment {
        Commitment(self.acc.into_affine())
    }

    fn add_block_coeffs_inner(&mut self, coeffs: &[F]) {
        assert!(
            self.cursor + coeffs.len() <= self.pcs.max_degree + 1,
            "coefficient stream exceeds max_degree: cursor={}, len={}, max_degree={}",
            self.cursor,
            coeffs.len(),
            self.pcs.max_degree
        );

        {
            let mut guard = srs_g1().lock().expect("SRS mutex poisoned");
            guard.ensure_len(self.cursor + coeffs.len());
        }

        let guard = srs_g1().lock().expect("SRS mutex poisoned");
        for (i, c) in coeffs.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            let base = guard.get_power(self.cursor + i);
            // explicit bigint mult to match MSM everywhere
            let term = base.into_group().mul_bigint(c.into_bigint());
            self.acc += term;
        }
        drop(guard);

        self.cursor += coeffs.len();
    }
}

// ----------------------- Openings -----------------------

/// Legacy opening API to keep older call sites compiling.
pub fn open_at_points(
    _pcs: &PcsParams,
    commitments: &[Commitment],
    stream_eval: impl Fn(usize, F) -> F,
    points: &[F],
) -> Vec<OpeningProof> {
    let mut proofs = Vec::with_capacity(commitments.len().saturating_mul(points.len()));
    for (pi, _c) in commitments.iter().enumerate() {
        for &zeta in points {
            let val = stream_eval(pi, zeta);
            proofs.push(OpeningProof {
                zeta,
                value: val,
                witness_comm: Commitment(G1Affine::identity()),
            });
        }
    }
    proofs
}

/// Real KZG openings where the caller streams **coefficients** for the witness.
pub fn open_at_points_with_coeffs(
    pcs_for_poly: &PcsParams,
    commitments: &[Commitment],
    _stream_eval: impl Fn(usize, F) -> F,
    mut stream_coeff_hi_to_lo: impl FnMut(usize, &mut dyn FnMut(Vec<F>)),
    points: &[F],
) -> Vec<OpeningProof> {
    let mut proofs = Vec::with_capacity(commitments.len().saturating_mul(points.len()));
    let pcs_witness = PcsParams { basis: Basis::Coefficient, ..pcs_for_poly.clone() };

    for (pi, _c) in commitments.iter().enumerate() {
        for &zeta in points {
            let mut w_high_to_low: Vec<F> = Vec::new();
            let mut w_next = F::zero(); // conceptually b_d = 0

            let mut sink = |block: Vec<F>| {
                for &a_i in &block {
                    let b_i_minus_1 = a_i + zeta * w_next;
                    w_high_to_low.push(b_i_minus_1);
                    w_next = b_i_minus_1;
                }
            };
            stream_coeff_hi_to_lo(pi, &mut sink);

            let f_at_z = w_high_to_low.pop().unwrap_or(F::zero());

            // Commit to W: aggregator expects **low→high**.
            w_high_to_low.reverse();
            let mut agg_w = Aggregator::new(&pcs_witness, "witness");
            const CHUNK: usize = 1 << 12;
            for chunk in w_high_to_low.chunks(CHUNK) {
                agg_w.add_block_coeffs(chunk);
            }
            let w_comm = agg_w.finalize();

            proofs.push(OpeningProof { zeta, value: f_at_z, witness_comm: w_comm });
        }
    }

    proofs
}

/// Wrapper that accepts **evaluation-basis** streams and internally converts
/// them to **coefficient-basis**, then delegates to the coeff path.
pub fn open_eval_stream_at_points(
    pcs_for_poly: &PcsParams,
    commitments: &[Commitment],
    domain: &crate::domain::Domain,
    mut stream_evals: impl FnMut(usize, &mut dyn FnMut(Vec<F>)),
    points: &[F],
) -> Vec<OpeningProof> {
    let mut as_coeff_hi_to_lo = |idx: usize, sink: &mut dyn FnMut(Vec<F>)| {
        let mut coeff_blocks: Vec<Vec<F>> = Vec::new();
        let mut collect = |eval_block: Vec<F>| {
            let coeffs = domain::ifft_block_evals_to_coeffs(domain, &eval_block);
            coeff_blocks.push(coeffs);
        };
        stream_evals(idx, &mut collect);

        for block in coeff_blocks.into_iter().rev() {
            let mut tmp: Vec<F> = block.into_iter().rev().collect();
            sink(tmp.split_off(0))
        }
    };

    open_at_points_with_coeffs(
        pcs_for_poly,
        commitments,
        |_i, _z| F::zero(),
        &mut as_coeff_hi_to_lo,
        points,
    )
}

// ----------------------- Verification -----------------------

/// Verify a batch of openings via pairings. Returns `true` on success.
///
/// We check multiplicatively:
///   e(C, G2) · e(−f(ζ)·G1, G2) · e(−W, [τ]G2 − ζ·G2) == 1
///
/// Also enforces shape and **point** agreement:
///   - `claimed_evals[i] == proofs[i].value`
///   - `proofs[i].zeta == points[j]` for the matching `(poly, point)` slot.
pub fn verify_openings(
    _pcs: &PcsParams,
    commitments: &[Commitment],
    points: &[F],
    claimed_evals: &[F],
    proofs: &[OpeningProof],
) -> bool {
    let expected = commitments.len().saturating_mul(points.len());
    if proofs.len() != expected || claimed_evals.len() != expected {
        return false;
    }

    // Fetch [1]_G1 and [τ]_G2.
    let g1_gen = {
        let guard = srs_g1().lock().expect("SRS G1 mutex poisoned");
        guard.get_power(0) // = generator
    };
    let g2_gen = <Bn254 as Pairing>::G2::generator().into_affine();
    let g2_tau = {
        let guard = srs_g2().lock().expect("SRS G2 mutex poisoned");
        match guard.tau_g2 {
            Some(t) => t,
            None => panic!("G2 SRS not loaded. Call pcs::load_srs_g2([τ]G2) before verifying."),
        }
    };

    let mut a_all: Vec<<Bn254 as Pairing>::G1Prepared> = Vec::with_capacity(expected * 3);
    let mut b_all: Vec<<Bn254 as Pairing>::G2Prepared> = Vec::with_capacity(expected * 3);

    // Build pairings in poly-major / point-minor order.
    let mut idx = 0usize;
    for (p, cmt) in commitments.iter().enumerate() {
        let c_aff = cmt.0;
        for (j, &pt) in points.iter().enumerate() {
            let pr = &proofs[idx];
            let val = claimed_evals[idx];

            // Sanity: eval value and point must match the proof.
            if pr.value != val {
                return false;
            }
            if pr.zeta != pt {
                return false;
            }

            // e(C, G2)
            a_all.push(<Bn254 as Pairing>::G1Prepared::from(c_aff));
            b_all.push(<Bn254 as Pairing>::G2Prepared::from(g2_gen));

            // e(−f(ζ)·G1, G2)
            let minus_f_g1 = (-g1_gen.into_group().mul_bigint(val.into_bigint())).into_affine();
            a_all.push(<Bn254 as Pairing>::G1Prepared::from(minus_f_g1));
            b_all.push(<Bn254 as Pairing>::G2Prepared::from(g2_gen));

            // e(−W, [τ]G2 − ζ·G2)
            let right_g2 =
                (g2_tau.into_group() - g2_gen.into_group().mul_bigint(pt.into_bigint())).into_affine();
            let minus_w = (-pr.witness_comm.0).into_group().into_affine();
            a_all.push(<Bn254 as Pairing>::G1Prepared::from(minus_w));
            b_all.push(<Bn254 as Pairing>::G2Prepared::from(right_g2));

            idx += 1;
        }
    }

    if a_all.is_empty() {
        return true;
    }

    let mlo = <Bn254 as Pairing>::multi_miller_loop(a_all, b_all);
    if let Some(fe) = <Bn254 as Pairing>::final_exponentiation(mlo) {
        return fe.0.is_one();
    }
    false
}