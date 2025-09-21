//! Streaming polynomial evaluation helpers
//!
//! This module provides a minimal, **streaming** interface to evaluate a
//! committed polynomial at a challenge point `ζ` using exactly **one pass**
//! over slices supplied by the caller. It supports two slice types:
//!   - `PolySlice::Evals(Vec<F>)`   : contiguous evaluations on a domain slice,
//!   - `PolySlice::Coeffs(Vec<F>)`  : contiguous monomial coefficients.
//!
//! Whitepaper constraints honored here:
//! - Exactly one pass per polynomial per point (no whole vectors in memory).
//! - When using evaluation slices, callers should provide domain-aligned
//!   chunks in **global order**; this function uses **barycentric** evaluation
//!   with the vanishing-polynomial derivative identity for `Z_H(X)=X^N - c`.
//! - When using coefficient slices, this function uses **Horner** evaluation,
//!   assuming coefficients are streamed **from highest degree down to 0**.
//!
//! Mixing slice variants within a single call is a caller bug. In debug builds
//! we panic; in release we ignore mismatched slices to keep the prover robust.

#![forbid(unsafe_code)]

use ark_ff::Zero;

use crate::F;

/// Streaming slices used to evaluate a polynomial without materializing it.
pub enum PolySlice {
    /// Contiguous evaluations over a domain slice (e.g., a block H_t).
    /// Must be provided in **global increasing** index order across calls.
    Evals(Vec<F>),
    /// Contiguous monomial coefficients for a degree range.
    /// Must be provided **from highest degree down to 0** across calls,
    /// and within each slice.
    Coeffs(Vec<F>),
}

/// Evaluate a polynomial at `zeta` by **streaming** slices produced by `next_slice`.
///
/// - If the provider yields `PolySlice::Evals` slices, this function performs a
///   **single-pass barycentric** evaluation over the domain `H` using
///   `domain::eval_stream_barycentric`.
/// - If the provider yields `PolySlice::Coeffs` slices, this function performs a
///   **single-pass Horner** scheme over the coefficients (high → low).
///
/// The provider controls which mode is used by choosing which slice variant(s)
/// to yield; do not mix variants for the same call.
///
/// Returns `f(ζ)`.
pub fn eval_poly_at(
    domain: &crate::domain::Domain,
    zeta: F,
    mut next_slice: impl FnMut() -> Option<PolySlice>,
) -> F {
    // Peek at the first slice to determine the mode.
    let first = match next_slice() {
        None => return F::zero(),
        Some(s) => s,
    };

    match first {
        PolySlice::Evals(first_block) => {
            // --------- Barycentric path over evaluation slices ---------
            //
            // We use the domain’s precomputed implicit barycentric weights and
            // stream the evaluations via an iterator that lazily pulls blocks
            // from the provider (exactly one pass).
            let w = crate::domain::bary_weights(domain);

            // Build a lazy iterator over all eval values across blocks.
            // We keep a small cursor into the current block and fetch a new
            // block from `next_slice` when exhausted.
            let mut cur_block: Option<(Vec<F>, usize)> = Some((first_block, 0));

            let iter = std::iter::from_fn(|| loop {
                if let Some((ref block, ref mut idx)) = cur_block {
                    if *idx < block.len() {
                        let y = block[*idx];
                        *idx += 1;
                        return Some(y);
                    } else {
                        cur_block = None; // exhaust; fetch next
                        continue;
                    }
                }
                match next_slice() {
                    Some(PolySlice::Evals(v)) => {
                        cur_block = Some((v, 0));
                        continue;
                    }
                    Some(PolySlice::Coeffs(_)) => {
                        // Mismatched variant within the same call.
                        // Debug panic to reveal caller bug; ignore in release.
                        #[cfg(debug_assertions)]
                        panic!("Mixed PolySlice variants (Evals + Coeffs) in eval_poly_at");
                        #[cfg(not(debug_assertions))]
                        return None;
                    }
                    None => return None,
                }
            });

            crate::domain::eval_stream_barycentric(domain, iter, zeta, &w)
        }

        PolySlice::Coeffs(first_block) => {
            // --------- Horner path over coefficient slices (high → low) ---------
            //
            // Caller must stream coefficients from highest degree down to 0
            // across all slices and within each slice.
            let mut acc = F::zero();

            // Fold first block
            for &a in &first_block {
                acc = acc * zeta + a;
            }

            // Fold remaining blocks; ignore Evals if provider mixes by mistake.
            while let Some(slc) = next_slice() {
                match slc {
                    PolySlice::Coeffs(v) => {
                        for &a in &v {
                            acc = acc * zeta + a;
                        }
                    }
                    PolySlice::Evals(_) => {
                        #[cfg(debug_assertions)]
                        panic!("Mixed PolySlice variants (Coeffs + Evals) in eval_poly_at");
                        // In release, ignore mismatched slices.
                    }
                }
            }
            acc
        }
    }
}
