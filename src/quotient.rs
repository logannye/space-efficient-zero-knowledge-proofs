//! Streaming Quotient Builder
//!
//! Algebra unchanged: for `Z_H(X)=X^N - c`, residuals and quotient satisfy
//! `R = Z_H * Q`, hence per-coefficient `q_i = -r_i / c`.
//!
//! IMPORTANT: We build `Q` by performing a **single global IFFT** of length `N`.
//! If the residual stream has only `T < N` values, we **pad with zeros**.

#![forbid(unsafe_code)]

use ark_ff::{Field, Zero};
use crate::{domain, pcs, F};

/// Build the quotient polynomial `Q` and return its PCS commitment.
///
/// Inputs:
/// - `domain`: evaluation domain `H` with `Z_H(X) = X^N - zh_c`.
/// - `pcs`: PCS params (must be configured for **coefficient basis** for `Q`).
/// - `(_alpha, _beta, _gamma)`: FS challenges (plumbed-through for schedule parity).
/// - `stream_r_rows`: iterator over `R(ω^i)` for `i = 0..N-1` in domain order.
pub fn build_and_commit_quotient<'a>(
    domain: &domain::Domain,
    pcs: &'a pcs::PcsParams,
    _alpha: F,
    _beta: F,
    _gamma: F,
    stream_r_rows: impl Iterator<Item = F>,
) -> pcs::Commitment {
    let n = domain.n;
    let c = domain.zh_c;
    let inv_c = c
        .inverse()
        .expect("domain.zh_c must be nonzero (division by X^N - c)");

    // 1) Collect residual evaluations and **pad to N with zeros**.
    let mut evals_r: Vec<F> = stream_r_rows.collect();
    if evals_r.len() < n {
        evals_r.resize(n, F::zero());
    }

    // 2) Single global IFFT → r_i coefficients (low→high).
    let r_coeffs_lo_to_hi = domain::ifft_block_evals_to_coeffs(domain, &evals_r);

    // 3) Convert to q_i = -r_i / c and feed to PCS in chunks.
    let mut agg = pcs::Aggregator::new(pcs, "Q");
    const CHUNK: usize = 1 << 12;

    let mut buf: Vec<F> = Vec::with_capacity(CHUNK);
    for &ri in &r_coeffs_lo_to_hi {
        buf.push(-ri * inv_c);
        if buf.len() == CHUNK {
            agg.add_block_coeffs(&buf);
            buf.clear();
        }
    }
    if !buf.is_empty() {
        agg.add_block_coeffs(&buf);
    }

    agg.finalize()
}
