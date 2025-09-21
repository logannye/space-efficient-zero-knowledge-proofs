//! Permutation & Lookup Accumulators (feature-gated lookups)
//!
//! This module implements the **multiplicative accumulators** used by the
//! permutation (Plonk-style) and (optionally) lookup arguments. The scheduler
//! must process blocks in **strictly increasing time order** so the products
//! factor correctly by block.
//!
//! Baseline behavior (no features):
//! - Permutation accumulator `Z` is fully implemented.
//! - Lookup path is a **no-op** (kept simple for the default build).
//!
//! Feature `lookups`:
//! - Enables a concrete φ_L(i) for a **lookup accumulator** `Z_L`.
//! - Adds a generic helper to compute a compressed multiplicand from
//!   caller-provided (LHS, RHS) row slices (useful if your AIR exposes
//!   explicit lookup wiring).
//!
//! NOTE: Commitment of `Z_L`, inclusion in the residual, and a verifier
//! check are deliberately **not** wired here. They’ll be added in the
//! scheduler/verifier under the same feature to keep the baseline clean.

#![forbid(unsafe_code)]
#![allow(unused_variables)]

use ark_ff::{Field, One, Zero};
use crate::F;

/// Plonk-style permutation accumulator `Z`.
///
/// In the protocol, `Z(i+1) = Z(i) * φ_perm(i)` where `φ_perm(i)` is the
/// per-row multiplicand derived from the local tuple and challenges `(β, γ)`.
#[derive(Debug, Clone, Copy)]
pub struct PermAcc {
    /// Current accumulator value (start at 1).
    pub z: F,
}

impl PermAcc {
    /// Initialize the permutation accumulator to 1.
    pub fn new() -> Self {
        Self { z: F::one() }
    }
}

/// Compute the per-row permutation multiplicand φ_perm for a given row.
///
/// φ_perm(i) = Π_c (w_c + β·id_c + γ) / Π_c (w_c + β·σ_c + γ)
#[inline]
fn phi_perm_row(row: &crate::air::Locals, beta: F, gamma: F) -> F {
    debug_assert_eq!(row.w_row.len(), row.id_row.len());
    debug_assert_eq!(row.w_row.len(), row.sigma_row.len());

    let mut num = F::one();
    let mut den = F::one();

    // k is small (fixed-column AIR), so this loop is constant-time w.r.t. T.
    for ((&w, &id), &sig) in row
        .w_row
        .iter()
        .zip(row.id_row.iter())
        .zip(row.sigma_row.iter())
    {
        num *= w + beta * id + gamma;
        den *= w + beta * sig + gamma;
    }

    // In valid traces with random (β,γ), denominators are nonzero with overwhelming probability.
    match den.inverse() {
        Some(inv) => num * inv,
        None => F::zero(),
    }
}

/// Absorb one **block** of rows into the permutation accumulator.
///
/// Time-order: callers *must* process blocks in strictly increasing `t`.
pub fn absorb_block_perm(
    acc: &mut PermAcc,
    locals: &[crate::air::Locals],
    beta: F,
    gamma: F,
) {
    for row in locals {
        let phi = phi_perm_row(row, beta, gamma);
        acc.z *= phi;
    }
}

/// Optionally **emit the committed Z-column** values for this block.
///
/// Given the start value and block `locals`, return the vector `z_vals` where
/// `z_vals[i]` equals `Z` **after** processing the i-th row of the block.
pub fn emit_z_column_block(
    start: F,
    locals: &[crate::air::Locals],
    beta: F,
    gamma: F,
) -> Vec<F> {
    let mut out = Vec::with_capacity(locals.len());
    let mut z = start;
    for row in locals {
        let phi = phi_perm_row(row, beta, gamma);
        z *= phi;
        out.push(z);
    }
    out
}

// --------------------------- Lookup (optional, feature-gated) ---------------------------

/// Lookup accumulator `Z_L` (optional, scheme-dependent).
///
/// For standard lookup arguments, the accumulator evolves multiplicatively by
/// a per-row factor φ_L(i) derived from `Locals` and challenges `(β, γ)`.
#[derive(Debug, Clone, Copy)]
pub struct LookupAcc {
    /// Current lookup accumulator value (start at 1).
    pub z: F,
}

impl LookupAcc {
    /// Initialize the lookup accumulator to 1.
    pub fn new() -> Self {
        Self { z: F::one() }
    }
}

/// A **generic compressed multiplicand** builder for lookup-style accumulators.
///
/// Given *left* and *right* slices for a row (caller-defined), compress them
/// with `(β, γ)` as Π_j ( left_j + β·right_j + γ ). This is useful if your AIR
/// exposes explicit lookup wiring and you want to form a ratio across multiple
/// calls (e.g., multiply with witness terms, divide with table terms).
#[inline]
pub fn phi_lookup_compress(left: &[F], right: &[F], beta: F, gamma: F) -> F {
    debug_assert_eq!(left.len(), right.len());
    let mut acc = F::one();
    for (&l, &r) in left.iter().zip(right.iter()) {
        acc *= l + beta * r + gamma;
    }
    acc
}

/// Convenience helper to build a **fractional** lookup multiplicand:
///   φ_L = Π (LHS_j + β·RHS_j + γ)  /  Π (LHS'_j + β·RHS'_j + γ)
///
/// Callers can feed the denominator terms in a second pass by multiplying with
/// `phi_lookup_compress(...)^{-1}` if desired, or pre-invert here.
#[inline]
pub fn phi_lookup_fraction(
    lhs: &[F],
    rhs: &[F],
    lhs_den: &[F],
    rhs_den: &[F],
    beta: F,
    gamma: F,
) -> F {
    let num = phi_lookup_compress(lhs, rhs, beta, gamma);
    let den = phi_lookup_compress(lhs_den, rhs_den, beta, gamma);
    match den.inverse() {
        Some(inv) => num * inv,
        None => F::zero(),
    }
}

/// Feature-gated **demo** φ_L(i) using the row’s `selectors_row` as a cheap wiring:
///
/// When the `lookups` feature is **enabled**, we interpret `selectors_row` as:
/// - first `k` entries: an auxiliary "table projection" `t_j`
/// - next  `k` entries (if present): an auxiliary "right projection" `r_j`
///
/// and compress as Π_j ( w_j + β·t_j + γ ) / Π_j ( w_j + β·r_j + γ ) if both
/// halves exist; otherwise we fall back to Π_j ( w_j + β·t_j + γ ) (no ratio).
///
/// This is only a **demo** shape to exercise the flow without extending `Locals`.
#[cfg(feature = "lookups")]
#[inline]
fn phi_lookup_row(row: &crate::air::Locals, beta: F, gamma: F) -> F {
    let k = row.w_row.len();
    let s = &row.selectors_row;
    if s.is_empty() {
        return F::one();
    }
    let t_len = core::cmp::min(k, s.len());
    let lhs = &row.w_row[..t_len];
    let rhs_table = &s[..t_len];

    // If we have 2k selectors, use the second half as the "right" projection for the denominator.
    if s.len() >= 2 * t_len {
        let rhs_den = &s[t_len..(2 * t_len)];
        return phi_lookup_fraction(lhs, rhs_table, lhs, rhs_den, beta, gamma);
    }

    // Otherwise, do a single compressed product with the table projection.
    phi_lookup_compress(lhs, rhs_table, beta, gamma)
}

/// With the feature **OFF**, φ_L(i) is a strict no-op (1).
#[cfg(not(feature = "lookups"))]
#[inline]
fn phi_lookup_row(_row: &crate::air::Locals, _beta: F, _gamma: F) -> F {
    F::one()
}

/// Absorb one **block** into the lookup accumulator.
///
/// Baseline: no-op unless the `lookups` feature is enabled and `phi_lookup_row`
/// returns a nontrivial factor. Time-order still matters if you commit Z_L.
pub fn absorb_block_lookup(
    acc: &mut LookupAcc,
    locals: &[crate::air::Locals],
) {
    #[allow(unused_variables)]
    for row in locals {
        #[cfg(feature = "lookups")]
        {
            // Default to β=0, γ=0 for the plain no-op path; real scheduling uses the
            // `*_with_challenges` variant below.
            let phi = phi_lookup_row(row, F::zero(), F::zero());
            acc.z *= phi;
        }
        #[cfg(not(feature = "lookups"))]
        {
            let _ = row; // silence unused
        }
    }
}

/// Absorb one **block** into the lookup accumulator using `(β, γ)`.
///
/// Use this variant from the scheduler when you *do* want real lookups
/// (behind the `lookups` feature).
pub fn absorb_block_lookup_with_challenges(
    acc: &mut LookupAcc,
    locals: &[crate::air::Locals],
    beta: F,
    gamma: F,
) {
    for row in locals {
        let phi = phi_lookup_row(row, beta, gamma);
        acc.z *= phi;
    }
}

/// Optionally **emit a committed lookup Z_L column** for this block.
///
/// Mirrors `emit_z_column_block` for permutation, but uses the lookup
/// multiplicand φ_L(i). Useful if your protocol commits the lookup
/// accumulator column. Time-order still must be increasing.
pub fn emit_lookup_column_block(
    start: F,
    locals: &[crate::air::Locals],
    beta: F,
    gamma: F,
) -> Vec<F> {
    let mut out = Vec::with_capacity(locals.len());
    let mut z = start;
    for row in locals {
        let phi = phi_lookup_row(row, beta, gamma);
        z *= phi;
        out.push(z);
    }
    out
}
