//! Streaming & Block Buffers
//!
//! This module centralizes the *streaming discipline* required by the
//! sublinear-space prover: at most **O(b_blk)** live data per worker.
//! It provides:
//!   - light-weight block/row/register indices used across the crate,
//!   - a canonical block partitioner over `T` rows,
//!   - simple traversal adapters (layered-BFS or small-stack DFS),
//!   - a reusable per-block workspace `BlockWs` with pre-allocated buffers,
//!   - tiny utilities to help enforce monotone block order and validate bounds,
//!   - a **Restreamer** trait that lets the scheduler obtain *fresh* row
//!     iterators per phase without buffering the entire witness again.
//!
//! Whitepaper constraints honored here:
//! - Callers should only keep one `BlockWs` live per worker, ensuring the
//!   working set remains O(b_blk).
//! - Partitioning preserves row order within each block; schedulers must
//!   process permutation blocks in **increasing** time order.

#![forbid(unsafe_code)]

use crate::F;

/// Index of a time block `t ∈ {0..B-1}`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockIdx(pub usize);
impl BlockIdx {
    /// Access the underlying index.
    #[inline]
    pub fn as_usize(self) -> usize {
        self.0
    }
}

/// Index of a row in the global trace `i ∈ {0..T-1}`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RowIdx(pub usize);
impl RowIdx {
    /// Access the underlying index.
    #[inline]
    pub fn as_usize(self) -> usize {
        self.0
    }
}

/// Index of a register/column `m ∈ {0..k-1}`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegIdx(pub usize);
impl RegIdx {
    /// Access the underlying index.
    #[inline]
    pub fn as_usize(self) -> usize {
        self.0
    }
}

/// Compute the number of blocks `B` for `n_rows = T` and block size `b_blk`.
#[inline]
pub fn block_count(n_rows: usize, b_blk: usize) -> usize {
    assert!(b_blk > 0, "b_blk must be positive");
    (n_rows + b_blk - 1) / b_blk
}

/// Get the half-open row bounds `[start, end)` for block `t`.
#[inline]
pub fn block_bounds(t: BlockIdx, n_rows: usize, b_blk: usize) -> (RowIdx, RowIdx) {
    let b_cnt = block_count(n_rows, b_blk);
    assert!(
        t.0 < b_cnt,
        "block index {} out of range (B={})",
        t.0,
        b_cnt
    );
    let start = t.0 * b_blk;
    let end = ((t.0 + 1) * b_blk).min(n_rows);
    (RowIdx(start), RowIdx(end))
}

/// Length of a block slice `[start, end)`.
#[inline]
pub fn block_len(start: RowIdx, end: RowIdx) -> usize {
    debug_assert!(end.0 >= start.0, "invalid block bounds");
    end.0 - start.0
}

/// Partition `n_rows = T` into contiguous time blocks of size `b_blk`
/// (except possibly the final shorter block).
///
/// Returns an iterator yielding `(BlockIdx, RowIdx(start), RowIdx(end))`
/// triples, where the half-open interval `[start, end)` is the block slice.
pub fn blocks(
    n_rows: usize,
    b_blk: usize,
) -> impl Iterator<Item = (BlockIdx, RowIdx, RowIdx)> {
    let b_cnt = block_count(n_rows, b_blk);
    (0..b_cnt).map(move |t| {
        let (s, e) = block_bounds(BlockIdx(t), n_rows, b_blk);
        (BlockIdx(t), s, e)
    })
}

/// Suggested traversal strategies for block processing.
///
/// - `LayeredBfs`: layer-by-layer (increasing `t`), natural for streaming.
/// - `DfsSmallStack`: depth-first adapter that keeps a tiny stack; the
///   scheduler may interleave registers but must still respect causality.
pub enum Traversal {
    /// Layer-by-layer traversal in increasing time (recommended for streaming).
    LayeredBfs,
    /// Depth-first traversal with a tiny bounded stack (still respects time order).
    DfsSmallStack,
}

/// Return an iterator over block indices according to a traversal policy.
///
/// For Phase 7 we expose a simple increasing sequence. The scheduler can
/// switch between BFS/DFS policies while preserving **increasing time**
/// for permutation/lookup accumulators.
pub fn traverse_blocks(_t: Traversal, b_cnt: usize) -> impl Iterator<Item = BlockIdx> {
    (0..b_cnt).map(BlockIdx)
}

/// A tiny guard to help enforce **strictly increasing** block order.
///
/// Usage:
/// ```ignore
/// let mut guard = MonotoneBlockGuard::new();
/// for (t, s, e) in blocks(T, b_blk) {
///     guard.observe(t); // panics if not strictly increasing
///     // ... process block t ...
/// }
/// ```
pub struct MonotoneBlockGuard {
    prev: Option<BlockIdx>,
}
impl MonotoneBlockGuard {
    /// Create a new guard.
    #[inline]
    pub fn new() -> Self {
        Self { prev: None }
    }
    /// Observe `t` and assert that it is strictly increasing.
    #[inline]
    pub fn observe(&mut self, t: BlockIdx) {
        if let Some(p) = self.prev {
            assert!(
                t.0 > p.0,
                "block indices must be processed in strictly increasing order (got {}, prev {})",
                t.0,
                p.0
            );
        }
        self.prev = Some(t);
    }
}

/// Per-block workspace with preallocated buffers.
///
/// Reuse one `BlockWs` per worker to keep the peak memory at O(b_blk).
pub struct BlockWs {
    /// Buffer for the target register’s values within the block.
    pub reg_vals: Vec<F>,
    /// Buffer for the block’s row-local tuples (perm/lookup inputs).
    pub locals: Vec<crate::air::Locals>,
    /// Scratch used by PCS/MSM or small transforms within the block.
    pub msm_tmp: Vec<F>,
}

impl BlockWs {
    /// Create a new workspace with capacity `cap = b_blk` for all buffers.
    pub fn new(cap: usize) -> Self {
        Self {
            reg_vals: Vec::with_capacity(cap),
            locals: Vec::with_capacity(cap),
            msm_tmp: Vec::with_capacity(cap),
        }
    }

    /// Clear buffers between blocks without freeing capacity.
    #[inline]
    pub fn reset(&mut self) {
        self.reg_vals.clear();
        self.locals.clear();
        self.msm_tmp.clear();
    }

    /// Ensure capacities are at least `cap` (useful if `b_blk` changes).
    pub fn ensure_cap(&mut self, cap: usize) {
        if self.reg_vals.capacity() < cap {
            self.reg_vals.reserve(cap - self.reg_vals.capacity());
        }
        if self.locals.capacity() < cap {
            self.locals.reserve(cap - self.locals.capacity());
        }
        if self.msm_tmp.capacity() < cap {
            self.msm_tmp.reserve(cap - self.msm_tmp.capacity());
        }
    }

    /// Assert that current live memory is bounded by `O(b_blk)` (debug aid).
    ///
    /// This is not a hard guarantee (Rust can over-allocate), but it helps
    /// catch accidental growth patterns during development.
    #[inline]
    pub fn debug_assert_o_bblk(&self, b_blk: usize) {
        debug_assert!(
            self.reg_vals.capacity() <= 2 * b_blk
                && self.locals.capacity() <= 2 * b_blk
                && self.msm_tmp.capacity() <= 2 * b_blk,
            "BlockWs buffers look larger than expected: reg_vals cap={}, locals cap={}, msm_tmp cap={}, b_blk={}",
            self.reg_vals.capacity(),
            self.locals.capacity(),
            self.msm_tmp.capacity(),
            b_blk
        );
    }
}

// ============================================================================
// Restreaming API — *read again* without buffering more state
// ============================================================================

/// A source that can **re-stream** rows between `[start, end)` on demand.
///
/// This is the canonical way schedulers perform multiple passes (wires,
/// permutation, quotient, openings) while keeping only **O(b_blk)** live data.
pub trait Restreamer {
    /// The item (row) type produced by this restreamer.
    type Item;

    /// Total number of rows `T` available from this source.
    fn len_rows(&self) -> usize;

    /// Produce a fresh iterator over rows in the half-open range `[start, end)`.
    ///
    /// Implementations may clone, re-read from disk, or produce a cursor over an
    /// in-memory slice — the scheduler doesn’t care, as long as items arrive
    /// in **time order**.
    fn stream_rows(
        &self,
        start: RowIdx,
        end: RowIdx,
    ) -> Box<dyn Iterator<Item = Self::Item> + '_>;
}

/// Trivial restreamer over an in-memory `Vec<Row>`.
///
/// This lets the current buffered witness plug in unchanged, while allowing the
/// scheduler to move to a trait-based interface (and later swap a file-backed
/// or network-backed restreamer without touching the orchestrator).
impl Restreamer for Vec<crate::air::Row> {
    type Item = crate::air::Row;

    #[inline]
    fn len_rows(&self) -> usize {
        self.len()
    }

    #[inline]
    fn stream_rows(
        &self,
        start: RowIdx,
        end: RowIdx,
    ) -> Box<dyn Iterator<Item = Self::Item> + '_> {
        let s = start.as_usize();
        let e = end.as_usize();
        assert!(s <= e && e <= self.len(), "restream range out of bounds");
        Box::new(self[s..e].iter().cloned())
    }
}

// ---- Notes for integrators ----
//
// * The scheduler should hold exactly one `BlockWs` per worker/thread and call
//   `reset()` after each block to maintain the O(b_blk) live-memory invariant.
//
// * Permutation/lookup passes must iterate blocks in strictly increasing
//   `(BlockIdx.0)` order to respect the whitepaper’s causality constraints.
//   You can use `MonotoneBlockGuard` to enforce this at runtime.
//
// * If you add parallelism, shard by disjoint registers/blocks such that each
//   worker’s peak live memory remains O(b_blk); avoid naive fan-out that would
//   multiply workspace by the number of threads.
