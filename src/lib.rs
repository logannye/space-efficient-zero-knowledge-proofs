//! Public surface & core types
//!
//! This file defines the crate’s public API and foundational type aliases.
//! It also exposes the concrete `Proof` object shape produced by the prover
//! and consumed by the verifier, along with re-exports of the orchestrators
//! implemented in `scheduler.rs`.
//!
//! Whitepaper constraints reflected here:
//! - `SECURITY_LAMBDA` is fixed while keeping λ = Θ(log T) implicit (we do not
//!   bind it to T anywhere).
//! - `Proof` explicitly carries only **aggregate commitments** (wires, optional
//!   permutation accumulator `Z`, quotient `Q`) and the Fiat–Shamir challenges’
//!   evaluation points and claimed evaluations, plus PCS opening proofs.
//!
//! ## Transcript & Ordering (Public Contract)
//!
//! The verifier **replays** the Fiat–Shamir transcript in the following order:
//!
//! 1. (Optional) **Selector/fixed** commitments, if the AIR provides them.
//! 2. **Wire** commitments, one per register, in register order `m = 0..k-1`.
//! 3. Sample challenges `(β, γ)`.
//! 4. (Optional) **Permutation accumulator** commitment `Z` (if committed).
//! 5. Sample challenge `α`.
//! 6. **Quotient** commitment `Q` (coefficient basis).
//! 7. Sample evaluation point set (e.g., `ζ`, possibly more points).
//!
//! The **opening set** is assembled in the same order as (2), (4?), (6):
//! ```text
//!   [ C_wire[0], C_wire[1], …, C_wire[k-1], (C_Z?), C_Q ]
//! ```
//!
//! The **claimed evaluations** `Proof::evals` are flattened in **poly-major
//! order** *within* the same opening-set order, and **point-major inside each
//! polynomial**. If `S` points are sampled and `M = k (+1 if Z) + 1`
//! polynomials are opened, the layout is:
//!
//! ```text
//!   evals = [
//!     f_0(ζ_0), f_0(ζ_1), …, f_0(ζ_{S-1}),
//!     f_1(ζ_0), f_1(ζ_1), …, f_1(ζ_{S-1}),
//!     …,
//!     f_{M-1}(ζ_0), …, f_{M-1}(ζ_{S-1}),
//!   ]
//! ```
//!
//! The `opening_proofs` vector matches this **exact** order, one proof object
//! per entry of `evals`.

#![forbid(unsafe_code)]
#![deny(missing_docs, rust_2018_idioms)]

/// Domain & transforms (vanishing polynomial X^N - c, barycentric, NTT/IFFT).
pub mod domain;
/// Polynomial commitment scheme interface and linear aggregator (KZG by default).
pub mod pcs;
/// Fiat–Shamir transcript (domain-separated hashing, hash→field).
pub mod transcript;
/// AIR template & block evaluator (local transitions / locals tuple).
pub mod air;
/// Permutation & lookup accumulators (multiplicative, time-ordered).
pub mod perm_lookup;
/// Streaming/blocking utilities and O(b_blk) workspace.
pub mod stream;
/// Quotient builder (blocked IFFT + X^N - c coefficient recurrence).
pub mod quotient;
/// Streaming polynomial evaluation (barycentric / Horner).
pub mod opening;
/// Five-phase scheduler orchestrating A–E with aggregate-only FS discipline.
pub mod scheduler;

/// Re-export the real orchestrators implemented in `scheduler.rs`.
pub use scheduler::{Prover, Verifier};

/// Scalar field used across the crate (BN254 by default).
pub type F = ark_bn254::Fr;

/// G1 affine group element used for commitments (KZG default).
pub type G1 = ark_bn254::G1Affine;

/// Security parameter λ. In the manuscript, λ = Θ(log T) is implicit;
/// we **do not** hardwire T here.
pub const SECURITY_LAMBDA: usize = 128;

/// Parameters required by the prover.
///
/// - `domain`: evaluation domain (size `N`, generator `ω`, and `Z_H(X)=X^N-c` constant).
/// - `pcs_wires`: PCS parameters for **wire polynomials** (usually Evaluation basis).
/// - `pcs_coeff`: PCS parameters for **coefficient-basis polys** like `Q`.
/// - `b_blk`: block size `b_blk` (choose `≈ √T` to realize sublinear space).
#[derive(Clone, Debug)]
pub struct ProveParams {
    /// Evaluation/coset domain and vanishing polynomial descriptor.
    pub domain: crate::domain::Domain,
    /// PCS parameters used for wires (basis discipline must match scheduler).
    pub pcs_wires: crate::pcs::PcsParams,
    /// PCS parameters used for coefficient-basis commitments (e.g., Q).
    pub pcs_coeff: crate::pcs::PcsParams,
    /// Block size used by the streaming prover (`b_blk ≈ √T`).
    pub b_blk: usize,
}

/// Parameters required by the verifier.
///
/// These must match the prover’s public parameters (same domain/SRS/bases).
#[derive(Clone, Debug)]
pub struct VerifyParams {
    /// Evaluation/coset domain and vanishing polynomial descriptor.
    pub domain: crate::domain::Domain,
    /// PCS parameters used for wires (basis must match prover).
    pub pcs_wires: crate::pcs::PcsParams,
    /// PCS parameters used for coefficient-basis commitments (e.g., Q).
    pub pcs_coeff: crate::pcs::PcsParams,
}

/// Wrapper around the PCS commitment type.
///
/// This newtype keeps the crate’s external API stable if the PCS backend changes.
#[derive(Clone, Copy, Debug)]
pub struct Commitment(pub crate::pcs::Commitment);

/// The SSZKP proof object.
///
/// This carries exactly what the verifier needs to reconstruct the Fiat–Shamir
/// transcript and check PCS openings:
/// - Aggregated per-polynomial commitments (wires `C_wire[m]`, optional `C_Z`,
///   and the quotient `C_Q`).
/// - Evaluation points (e.g., `ζ`, possibly more).
/// - Claimed evaluations and opening proofs (ordering documented above).
#[derive(Clone, Debug)]
pub struct Proof {
    /// Per-register wire commitments (aggregated across blocks; order `m = 0..k-1`).
    ///
    /// These are absorbed into the transcript **in order** before sampling `(β, γ)`.
    pub wire_comms: Vec<Commitment>,

    /// Optional permutation accumulator commitment `Z` (if committed by the scheme).
    ///
    /// If present, it is absorbed **after** sampling `(β, γ)` and **before** sampling `α`.
    pub z_comm: Option<Commitment>,

    /// Quotient commitment `Q` (coefficient-basis).
    ///
    /// This is absorbed **after** sampling `α` and **before** sampling the evaluation points.
    pub q_comm: Commitment,

    /// Evaluation points sampled via FS (e.g., `[ζ, …]`).
    ///
    /// The prover and verifier derive these *after* absorbing `Q`, using the same transcript state.
    pub eval_points: Vec<F>,

    /// Claimed evaluations flattened in **poly-major, point-minor** order
    /// matching the opening set `[ C_wire[0..k-1], (C_Z?), C_Q ]`.
    ///
    /// For `S = eval_points.len()` and `M = k (+1 if Z) + 1`, the layout is:
    /// `[
    ///   f_0(ζ_0)..f_0(ζ_{S-1}),
    ///   f_1(ζ_0)..f_1(ζ_{S-1}),
    ///   …,
    ///   f_{M-1}(ζ_0)..f_{M-1}(ζ_{S-1})
    /// ]`.
    pub evals: Vec<F>,

    /// PCS opening proofs corresponding 1-to-1 with `evals` in the **same order**.
    pub opening_proofs: Vec<crate::pcs::OpeningProof>,
}
