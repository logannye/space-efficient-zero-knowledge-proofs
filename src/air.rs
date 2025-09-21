//! AIR template & block evaluator
//!
//! This module encodes the local transition semantics needed by the streaming
//! prover. It exposes a *block evaluator* that consumes only the **previous
//! block’s boundary** (bounded read-degree) and a streaming iterator of rows,
//! and emits exactly the artifacts later phases need:
//!   - the target register’s values on this block (`reg_m_vals`),
//!   - the per-row local tuple used by permutation/lookup accumulators (`locals`),
//!   - the end-of-block boundary vector (`boundary_out`).
//!
//! Whitepaper constraints honored here:
//! - The evaluator never receives or materializes the full trace.
//! - Only the prior block boundary is required (first-order, bounded read-degree).
//! - The emitted artifacts are precisely those consumed by permutation/lookup and
//!   by wire commitments in the streaming pipeline.
//!
//! Plonkish-style fixed-column model:
//! - We lock a compact “locals” view with `w_row` (the k register values),
//!   `id_row` and `sigma_row` (the per-row permutation wiring), plus optional
//!   **selector/fixed** columns used by gate residuals in Phase D.
//! - By default, we synthesize a **cyclic σ** within the row (a 1-step rotation)
//!   over the identity `id = [0, 1, …, k-1]`. This is sufficient to exercise
//!   the multiplicative permutation accumulator flow while keeping the AIR
//!   simple. Integrators can plug real `id/σ` and selector tables later
//!   without changing the scheduler or accumulators.

#![forbid(unsafe_code)]

use crate::stream::{BlockIdx, RegIdx, Restreamer};
use crate::F;
use ark_ff::{Field, One, Zero};

/// AIR template (fixed-column model).
#[derive(Debug, Clone)]
pub struct AirSpec {
    /// Number of registers/columns (fixed-column AIR).
    pub k: usize,
    /// Optional identity table per column per row.
    pub id_table: Vec<Box<[F]>>,
    /// Optional σ-table per column per row.
    pub sigma_table: Vec<Box<[F]>>,
    /// Optional **selector/fixed** columns used by gate constraints.
    pub selectors: Vec<Box<[F]>>,
}

impl AirSpec {
    /// Construct a minimal AIR with `k` registers and **synthesized** id/σ, no selectors.
    pub fn with_cyclic_sigma(k: usize) -> Self {
        Self { k, id_table: Vec::new(), sigma_table: Vec::new(), selectors: Vec::new() }
    }

    /// Construct an AIR with `k`, optional explicit `id/σ`, and `selectors`.
    pub fn with_tables(
        k: usize,
        id_table: Vec<Box<[F]>>,
        sigma_table: Vec<Box<[F]>>,
        selectors: Vec<Box<[F]>>,
    ) -> Self {
        Self { k, id_table, sigma_table, selectors }
    }

    /// Produce `(id_row, sigma_row)` for the current row.
    fn make_id_sigma_row(&self, row_ctr: usize) -> (Box<[F]>, Box<[F]>) {
        if self.id_table.is_empty() || self.sigma_table.is_empty() {
            // Synthesize identity & cyclic σ within the row (size k).
            let mut id = Vec::with_capacity(self.k);
            let mut sigma = Vec::with_capacity(self.k);
            for j in 0..self.k {
                id.push(F::from(j as u64));
                sigma.push(F::from(((j + 1) % self.k) as u64));
            }
            return (id.into_boxed_slice(), sigma.into_boxed_slice());
        }

        // Use provided tables. If a column has insufficient length, wrap.
        let mut id = Vec::with_capacity(self.k);
        let mut sigma = Vec::with_capacity(self.k);
        for col in 0..self.k {
            let id_col = &self.id_table[col];
            let sigma_col = &self.sigma_table[col];
            let id_val = if id_col.is_empty() { F::from(col as u64) } else { id_col[row_ctr % id_col.len()] };
            let sigma_val =
                if sigma_col.is_empty() { F::from(((col + 1) % self.k) as u64) } else { sigma_col[row_ctr % sigma_col.len()] };
            id.push(id_val);
            sigma.push(sigma_val);
        }
        (id.into_boxed_slice(), sigma.into_boxed_slice())
    }

    /// Produce the **selectors/fixed** row for the current row index.
    fn make_selectors_row(&self, row_ctr: usize) -> Box<[F]> {
        if self.selectors.is_empty() {
            return Box::from([]);
        }
        let mut s = Vec::with_capacity(self.selectors.len());
        for col in &self.selectors {
            if col.is_empty() {
                s.push(F::zero());
            } else {
                s.push(col[row_ctr % col.len()]);
            }
        }
        s.into_boxed_slice()
    }
}

/// One row of the execution trace (k registers).
#[derive(Debug, Clone)]
pub struct Row {
    /// Register values for this row; length must be exactly `k`.
    pub regs: Box<[F]>,
}

/// Row-local tuple needed by global accumulators (perm/lookups) **and gates**.
#[derive(Debug, Clone)]
pub struct Locals {
    /// The k register values at this row.
    pub w_row: Box<[F]>,
    /// The identity projection for this row (k elements).
    pub id_row: Box<[F]>,
    /// The σ projection for this row (k elements).
    pub sigma_row: Box<[F]>,
    /// Selector/fixed columns for this row (may be empty).
    pub selectors_row: Box<[F]>,
}

/// Result of evaluating one block for a target register `m`.
#[derive(Debug, Clone)]
pub struct BlockResult {
    /// Values of register `m` on the rows of this block (in order).
    pub reg_m_vals: Vec<F>,
    /// Row-local tuples for this block; one entry per row.
    pub locals: Vec<Locals>,
    /// End-of-block boundary vector (k registers) to seed the next block.
    pub boundary_out: Box<[F]>,
}

/// Evaluate one time block for target register `m`.
pub fn eval_block(
    air: &AirSpec,
    m: RegIdx,
    _t: BlockIdx,
    boundary_in: &[F],
    iter_rows: impl Iterator<Item = Row>,
) -> BlockResult {
    assert_eq!(
        boundary_in.len(),
        air.k,
        "boundary vector must have k={} registers",
        air.k
    );
    assert!(m.0 < air.k, "target register m={} out of range (k={})", m.0, air.k);

    let mut reg_m_vals: Vec<F> = Vec::new();
    let mut locals: Vec<Locals> = Vec::new();
    // Start with the incoming boundary; update to the last row.
    let mut boundary_out: Box<[F]> = boundary_in.to_vec().into_boxed_slice();

    // Stream the rows.
    let mut row_ctr = 0usize;
    for row in iter_rows {
        assert_eq!(row.regs.len(), air.k, "row.regs length must be k={}, got {}", air.k, row.regs.len());

        // Emit the target register’s value for this row.
        reg_m_vals.push(row.regs[m.0]);

        // Build locals for permutation/lookup accumulators and gate residuals.
        let (id_row, sigma_row) = air.make_id_sigma_row(row_ctr);
        let selectors_row = air.make_selectors_row(row_ctr);
        locals.push(Locals { w_row: row.regs.clone(), id_row, sigma_row, selectors_row });

        // Update boundary_out to this row’s final state.
        boundary_out = row.regs;
        row_ctr += 1;
    }

    BlockResult { reg_m_vals, locals, boundary_out }
}

// ============================================================================
// Residual builder (Phase D)
// ============================================================================

/// Fiat–Shamir challenges used in the residual.
/// (α may weight / batch gate vs. perm vs. boundary parts.)
#[derive(Copy, Clone, Debug)]
pub struct ResidualCfg {
    /// Gate batching coefficient. Multiplies the selector-weighted gate residuals.
    pub alpha: F,
    /// Permutation challenge β used in the (w + β·id/σ + γ) factors.
    pub beta: F,
    /// Permutation challenge γ used in the (w + β·id/σ + γ) factors.
    pub gamma: F,
}

/// Compute Π_j (w_j + β·id_j + γ) and Π_j (w_j + β·σ_j + γ).
#[inline]
fn prod_id_sigma(air: &AirSpec, locals: &Locals, beta: F, gamma: F) -> (F, F) {
    let w = &locals.w_row;
    let mut prod_id = F::one();
    let mut prod_sigma = F::one();
    for j in 0..air.k {
        prod_id *= w[j] + beta * locals.id_row[j] + gamma;
        prod_sigma *= w[j] + beta * locals.sigma_row[j] + gamma;
    }
    (prod_id, prod_sigma)
}

/// Compute the **full permutation-coupled per-row residual**.
///
/// R_i = α·Gate_i
///       + ( Z_{i+1} · Π_j (w_j + β·id_j + γ)  −  Z_i · Π_j (w_j + β·σ_j + γ) )
///       + Boundary_i
///
/// `is_first_row` enables the Z(0)=1 hook via `Boundary_i += (Z_i - 1)`.
/// `is_last_row` enables the final-row hook via `Boundary_i += (Z_{i+1} - 1)`.
pub fn residual_row(
    air: &AirSpec,
    locals: &Locals,
    cfg: &ResidualCfg,
    z_i: F,
    z_ip1: F,
    is_first_row: bool,
    is_last_row: bool,
) -> F {
    // ----- Gate part (selector-weighted; demo gates kept as-is) -----
    let w = &locals.w_row;
    let s = &locals.selectors_row;
    // q_add: s[0], q_mul: s[1] (if available)
    let gate_add = if s.len() >= 1 && air.k >= 3 { s[0] * (w[0] + w[1] - w[2]) } else { F::zero() };
    let gate_mul = if s.len() >= 2 && air.k >= 3 { s[1] * (w[0] * w[1] - w[2]) } else { F::zero() };
    let gate_part = cfg.alpha * (gate_add + gate_mul);

    // ----- Permutation-coupled term -----
    let (prod_id, prod_sigma) = prod_id_sigma(air, locals, cfg.beta, cfg.gamma);
    let perm_coupled = z_ip1 * prod_id - z_i * prod_sigma;

    // ----- Boundary hooks -----
    // Enforce Z(0) = 1 at the first row (before multiplying by φ_perm),
    // and Z(T) = 1 at the last step (after multiplying by φ_perm).
    let mut boundary_part = F::zero();
    if is_first_row {
        boundary_part += z_i - F::one();
    }
    if is_last_row {
        boundary_part += z_ip1 - F::one();
    }

    gate_part + perm_coupled + boundary_part
}

// ============================================================================
// Residual stream over the full domain (Phase D input to quotient)
// ============================================================================

/// Build a **streamed** iterator of residual evaluations `R(ω^i)` in row order,
/// maintaining the permutation accumulator `Z` **on the fly** to avoid a
/// second pass.
///
/// For each row i:
///   φ_perm(i) = Π(w+β·id+γ) / Π(w+β·σ+γ)
///   Z_{i+1} = Z_i · φ_perm(i)
///   R_i as in `residual_row` above (with boundary hooks).
pub fn residual_stream<'a>(
    air: &'a AirSpec,
    cfg: ResidualCfg,
    rs: &'a impl Restreamer<Item = Row>,
    b_blk: usize,
) -> impl Iterator<Item = F> + 'a {
    let t_rows = rs.len_rows();

    // Global Z accumulator and row index (captured/mutated inside the closure).
    let mut z_cur = F::one();
    let mut global_idx = 0usize;

    (0..crate::stream::block_count(t_rows, b_blk)).flat_map(move |t| {
        let (s, e) = crate::stream::block_bounds(crate::stream::BlockIdx(t), t_rows, b_blk);
        let it = rs.stream_rows(s, e);
        let boundary_seed = vec![F::zero(); air.k].into_boxed_slice();
        let br = eval_block(air, RegIdx(0), BlockIdx(t), &boundary_seed, it);

        // Map each row to its residual while updating Z in lockstep.
        br.locals.into_iter().map(move |loc| {
            // Products for this row (reuse to compute φ_perm and residual).
            let (prod_id, prod_sigma) = prod_id_sigma(air, &loc, cfg.beta, cfg.gamma);

            // φ_perm = prod_id / prod_sigma
            let phi = match prod_sigma.inverse() {
                Some(inv) => prod_id * inv,
                None => F::zero(), // degenerate; makes failures clear downstream in debug setups
            };

            let z_next = z_cur * phi;

            let is_first = global_idx == 0;
            let is_last = global_idx + 1 == t_rows;

            let r_i = residual_row(air, &loc, &cfg, z_cur, z_next, is_first, is_last);

            // Advance Z and the global row index.
            z_cur = z_next;
            global_idx += 1;

            r_i
        })
    })
}
