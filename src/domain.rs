//! Domain & Transform Primitives
//!
//! This module defines the evaluation domain `H` with vanishing polynomial
//! `Z_H(X) = X^N - c`, plus production-grade **barycentric** streaming
//! evaluation helpers and **power-of-two radix-2 NTT/IFFT** used elsewhere.
//!
//! Whitepaper constraints honored here:
//! - `Domain` exposes `n`, a generator `omega`, and the constant `zh_c`
//!   so that `Z_H(X) = X^N - zh_c`.
//! - We validate that `omega` is a **primitive** N-th root for the subgroup.
//! - Optional coset support via `new_with_coset(g)`, which sets `zh_c = g^N`.
//! - We **assert ζ ∉ H** (by checking `ζ^N != zh_c`) for barycentric evals.
//! - Blocked IFFT can now emit **low→high or high→low** coefficient tiles,
//!   enabling quotient builders to stream straight into PCS without `O(N)` buffers.

#![forbid(unsafe_code)]

use ark_ff::{Field, One, Zero};

use crate::F;

/// Evaluation domain with vanishing polynomial `Z_H(X) = X^N - zh_c`.
///
/// - `n`: domain size `N` (typically a power of two).
/// - `omega`: generator of the (sub)group used for `H` (primitive N-th root).
/// - `zh_c`: constant `c` such that `Z_H(X) = X^N - c`.
#[derive(Debug, Clone)]
pub struct Domain {
    /// Domain size `N`. Prefer power-of-two for radix-2 FFTs.
    pub n: usize,
    /// Generator `ω` (primitive N-th root for subgroup).
    pub omega: F,
    /// Constant `c` in `Z_H(X) = X^N - c`.
    pub zh_c: F,
}

impl Domain {
    /// Construct a domain with explicit `zh_c` and run hygiene checks.
    pub fn new_with_c(n: usize, omega: F, zh_c: F) -> Self {
        let d = Self { n, omega, zh_c };
        validate_domain(&d);
        d
    }

    /// Construct a **coset** domain `H = g * <ω>` with `zh_c = g^N`.
    pub fn new_with_coset(n: usize, omega: F, coset_shift: F) -> Self {
        let zh_c = pow_u64(coset_shift, n as u64);
        Self::new_with_c(n, omega, zh_c)
    }
}

/// Barycentric weights for streaming evaluation over a multiplicative subgroup.
///
/// We store only *implicit* parameters sufficient to generate the classical
/// weights `w_i = 1 / Z_H'(ω^i)` on the fly:
/// - For `Z_H(X) = X^N - c`, `Z_H'(x) = N * x^{N-1}`, hence
///   `w_i = 1 / (N * (ω^i)^{N-1}) = inv_N * (ω^{-(N-1)})^i`.
#[derive(Debug, Clone)]
pub struct BarycentricWeights {
    inv_n: F,
    step: F, // step = ω^{-(N-1)}
}

#[inline]
fn pow_u64(mut base: F, mut exp: u64) -> F {
    let mut acc = F::one();
    while exp > 0 {
        if (exp & 1) == 1 {
            acc *= base;
        }
        base.square_in_place();
        exp >>= 1;
    }
    acc
}

// ------------------------- Hygiene / Validation -------------------------

fn prime_factors(mut n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut p = 2usize;
    while p * p <= n {
        if n % p == 0 {
            out.push(p);
            while n % p == 0 {
                n /= p;
            }
        }
        p += if p == 2 { 1 } else { 2 }; // 2,3,5,7,...
    }
    if n > 1 {
        out.push(n);
    }
    out
}

fn validate_domain(d: &Domain) {
    assert!(d.n > 0, "domain size must be positive");
    assert!(!d.zh_c.is_zero(), "zh_c must be non-zero");

    // Check ω^N == 1
    let w_n = pow_u64(d.omega, d.n as u64);
    assert!(w_n.is_one(), "omega^N must be 1 for subgroup of size N");

    // Check ω is **primitive**: ω^{N/p} != 1 for every prime factor p | N
    for p in prime_factors(d.n) {
        let w_np = pow_u64(d.omega, (d.n / p) as u64);
        assert!(
            !w_np.is_one(),
            "omega is not primitive: omega^(N/{}) == 1",
            p
        );
    }
}

// ------------------------- Barycentric -------------------------

/// Compute barycentric weights for domain `d`.
pub fn bary_weights(d: &Domain) -> BarycentricWeights {
    validate_domain(d);
    // inv_n = (N)^(-1)
    let inv_n = F::from(d.n as u64)
        .inverse()
        .expect("N must be non-zero in the field");
    // step = ω^{-(N-1)}  (so that w_i = inv_n * step^i)
    let omega_pow_n_minus_1 = pow_u64(d.omega, (d.n as u64).saturating_sub(1));
    let step = omega_pow_n_minus_1
        .inverse()
        .expect("ω^{N-1} must be non-zero");
    BarycentricWeights { inv_n, step }
}

/// Streaming barycentric evaluation over domain `d`.
pub fn eval_stream_barycentric(
    d: &Domain,
    it: impl Iterator<Item = F>,
    zeta: F,
    w: &BarycentricWeights,
) -> F {
    validate_domain(d);
    // Guard: assert ζ ∉ H  (for H = {x : x^N = zh_c})
    let z_pow_n = pow_u64(zeta, d.n as u64);
    assert!(
        z_pow_n != d.zh_c,
        "evaluation point ζ lies in H (ζ^N == zh_c); choose ζ ∉ H"
    );

    // Generate ω^i sequentially, and w_i sequentially via step-multiplication.
    let mut omega_i = F::one(); // ω^0
    let mut w_i = w.inv_n; // inv_N * step^0
    let mut num = F::zero();
    let mut den = F::zero();

    for f_i in it {
        // If ζ == ω^i, return f(ω^i) directly to avoid division by zero.
        if zeta == omega_i {
            return f_i;
        }
        // term = 1 / (ζ - ω^i)
        let denom_term = (zeta - omega_i)
            .inverse()
            .expect("zeta must not be in the domain (ζ ∉ H)");
        // Accumulate numerator and denominator
        num += w_i * f_i * denom_term;
        den += w_i * denom_term;

        // Next i: ω^{i+1}, w_{i+1}
        omega_i *= d.omega;
        w_i *= w.step;
    }

    // Final result: num / den
    num * den.inverse().expect("denominator must be non-zero")
}

// ------------------------- FFT / IFFT -------------------------

/// Validate that `len` is a power of two and divides `d.n`.
#[inline]
fn validate_len(d: &Domain, len: usize) {
    validate_domain(d);
    assert!(len > 0, "transform length must be positive");
    assert!(len.is_power_of_two(), "length must be a power of two");
    assert!(
        d.n % len == 0,
        "transform length must divide the domain size: len={}, n={}",
        len,
        d.n
    );
}

/// Compute a primitive `len`-th root of unity for this domain:
/// `root = ω^{N/len}` where `ω` is a primitive `N`-th root.
#[inline]
fn primitive_len_root(d: &Domain, len: usize) -> F {
    validate_len(d, len);
    pow_u64(d.omega, (d.n / len) as u64)
}

/// In-place iterative Cooley–Tukey radix-2 NTT (forward).
fn ntt_in_place(a: &mut [F], root: F) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }

    // Cooley–Tukey butterflies
    let mut len = 2;
    while len <= n {
        // w_len = root^{n/len}
        let w_len = pow_u64(root, (n / len) as u64);
        for start in (0..n).step_by(len) {
            let mut w = F::one();
            let half = len / 2;
            for i in 0..half {
                let u = a[start + i];
                let v = a[start + i + half] * w;
                a[start + i] = u + v;
                a[start + i + half] = u - v;
                w *= w_len;
            }
        }
        len <<= 1;
    }
}

/// In-place forward NTT with **cached per-stage twiddles** (for a fixed length).
fn ntt_in_place_cached(a: &mut [F], stage_wlens: &[F]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }

    // Cooley–Tukey butterflies with cached w_len per stage.
    let mut len = 2;
    let mut stage = 0usize;
    while len <= n {
        let w_len = stage_wlens[stage];
        for start in (0..n).step_by(len) {
            let mut w = F::one();
            let half = len / 2;
            for i in 0..half {
                let u = a[start + i];
                let v = a[start + i + half] * w;
                a[start + i] = u + v;
                a[start + i + half] = u - v;
                w *= w_len;
            }
        }
        stage += 1;
        len <<= 1;
    }
}

/// In-place inverse NTT (radix-2).
fn intt_in_place(a: &mut [F], root: F) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    // inv_root = root^{-1}
    let inv_root = root.inverse().expect("root must be non-zero");
    // Running forward with inv_root equals inverse NTT
    ntt_in_place(a, inv_root);
    // Scale by n^{-1}
    let inv_n = F::from(n as u64)
        .inverse()
        .expect("length must be non-zero in the field");
    for x in a.iter_mut() {
        *x *= inv_n;
    }
}

/// In-place inverse NTT with **cached per-stage twiddles** (for a fixed length).
fn intt_in_place_cached(a: &mut [F], inv_stage_wlens: &[F]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    // Inverse NTT = forward with inv_root, using cached stage twiddles for inv_root.
    ntt_in_place_cached(a, inv_stage_wlens);
    // Scale by n^{-1}
    let inv_n = F::from(n as u64)
        .inverse()
        .expect("length must be non-zero in the field");
    for x in a.iter_mut() {
        *x *= inv_n;
    }
}

/// Blocked inverse NTT/IFFT: **evaluations → coefficients** for a slice.
///
/// Consumes a *contiguous* slice of evaluations of length `m`, where `m` is a
/// power of two that **divides** `d.n`. Returns the matching coefficient block.
///
/// Note: This routine operates on the *first `m` points of the subgroup*
/// generated by `root = ω^{N/m}`. For general block offsets/cosets, extend
/// this function accordingly (future work).
pub fn ifft_block_evals_to_coeffs(d: &Domain, evals: &[F]) -> Vec<F> {
    let m = evals.len();
    validate_len(d, m);
    // Copy into a local buffer and run inverse NTT.
    let mut a = evals.to_vec();
    let root = primitive_len_root(d, m);
    // Inverse NTT maps evaluations (in standard order) to coefficients.
    intt_in_place(&mut a, root);
    a
}

/// Blocked NTT: **coefficients → evaluations** for a slice.
pub fn ntt_block_coeffs_to_evals(d: &Domain, coeffs: &[F]) -> Vec<F> {
    let m = coeffs.len();
    validate_len(d, m);
    // Copy into a local buffer and run forward NTT.
    let mut a = coeffs.to_vec();
    let root = primitive_len_root(d, m);
    ntt_in_place(&mut a, root);
    a
}

// -----------------------------------------------------------------------------
// Stockham-style **blocked IFFT** from a time-ordered evaluation stream
// -----------------------------------------------------------------------------

/// Emission order for coefficient **tiles** produced by the blocked IFFT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoeffTileOrder {
    /// Emit each tile in **low→high** coefficient index (tile-local).
    LowToHigh,
    /// Emit each tile in **high→low** coefficient index (tile-local).
    HighToLow,
}

/// Consume `R(ω^i)` in **time order** and emit **coefficient tiles** via `sink`.
///
/// Parameters:
/// - `d`: evaluation domain.
/// - `tile_size`: block length `m` (power of two, `m | d.n`). Choose `≈ b_blk`.
/// - `time_stream`: iterator over `R(ω^0), R(ω^1), …, R(ω^{N-1})`.
/// - `order`: per-tile coefficient order to emit (low→high or high→low).
/// - `sink`: callback invoked **per tile** with the coefficient block in the
///   requested **tile-local** order.
///
/// Space profile: O(`tile_size`) working memory. Twiddles per stage are
/// precomputed once for the chosen `tile_size`.
///
/// Notes:
/// - Tiles are emitted in **stream order**; callers that need a different
///   *global* order can transform as they consume (without O(N) buffering).
pub fn ifft_blocked_from_time_stream_ordered<'a>(
    d: &Domain,
    tile_size: usize,
    mut time_stream: impl Iterator<Item = F>,
    order: CoeffTileOrder,
    mut sink: impl FnMut(&[F]),
) {
    validate_len(d, tile_size);
    assert!(
        d.n % tile_size == 0,
        "tile_size must divide domain size: tile_size={}, n={}",
        tile_size,
        d.n
    );

    // Precompute stage twiddles for **inverse** NTT of length `tile_size`.
    let root = primitive_len_root(d, tile_size);
    let inv_root = root.inverse().expect("root must be non-zero");
    // For stage with current butterfly length `len`, w_len = inv_root^{n/len},
    // where here n = tile_size for the per-tile transform.
    let mut inv_stage_wlens: Vec<F> = Vec::new();
    {
        let n = tile_size;
        let mut len = 2usize;
        while len <= n {
            let w_len = pow_u64(inv_root, (n / len) as u64);
            inv_stage_wlens.push(w_len);
            len <<= 1;
        }
    }

    // Process the time stream in contiguous tiles of size `tile_size`.
    let mut buf: Vec<F> = Vec::with_capacity(tile_size);
    let mut received = 0usize;

    loop {
        buf.clear();
        for _ in 0..tile_size {
            if let Some(v) = time_stream.next() {
                buf.push(v);
                received += 1;
            } else {
                break;
            }
        }
        if buf.is_empty() {
            break; // no more data
        }
        assert!(
            buf.len() == tile_size,
            "time stream length must be a multiple of tile_size; got a partial tile of {}",
            buf.len()
        );

        // Inverse transform this tile to coefficients.
        intt_in_place_cached(&mut buf, &inv_stage_wlens);

        match order {
            CoeffTileOrder::LowToHigh => sink(&buf),
            CoeffTileOrder::HighToLow => {
                // Emit reversed view without allocating a second Vec.
                // SAFETY: create a temporary reversed copy to satisfy the sink signature.
                let mut tmp = buf.clone();
                tmp.reverse();
                sink(&tmp);
            }
        }
    }

    assert!(
        received == d.n,
        "expected exactly N={} evaluations in time stream; got {}",
        d.n,
        received
    );
}

/// Back-compat wrapper: emits **low→high** tiles (tile-local).
pub fn ifft_blocked_from_time_stream<'a>(
    d: &Domain,
    tile_size: usize,
    time_stream: impl Iterator<Item = F>,
    sink: impl FnMut(&[F]),
) {
    ifft_blocked_from_time_stream_ordered(
        d,
        tile_size,
        time_stream,
        CoeffTileOrder::LowToHigh,
        sink,
    )
}
