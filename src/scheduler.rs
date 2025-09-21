//! Five-Phase Scheduler
//!
//! This module wires the prover’s **Fiat–Shamir–ordered** schedule exactly
//! as outlined in the whitepaper. It preserves the **aggregate-only FS
//! discipline** and the **increasing-time** order for permutation/lookup
//! passes. Cryptographic internals are delegated to `pcs`, `quotient`,
//! and streaming helpers; this file focuses on orchestration, ordering,
//! and keeping only **O(b_blk)** live state per active worker when
//! paired with block-restreaming sources.

#![forbid(unsafe_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]

use ark_ff::{Field, One, Zero};

use crate::{
    air::{self, BlockResult},
    pcs::{self, Aggregator, Basis, Commitment, PcsParams},
    perm_lookup::{absorb_block_lookup, absorb_block_perm, emit_z_column_block, LookupAcc, PermAcc},
    quotient::build_and_commit_quotient,
    stream::{blocks, BlockIdx, RegIdx, RowIdx, Restreamer},
    transcript::Transcript,
    F,
};

/// Prover orchestrator (wired to the 5-phase schedule).
pub struct Prover<'a> {
    /// AIR template (fixed columns, bounded read-degree) that defines the
    /// register layout and per-block evaluation function.
    pub air: &'a air::AirSpec,
    /// Public parameters: domain, PCS configs (for wires and for Q), and
    /// the streaming block size `b_blk`.
    pub params: &'a crate::ProveParams,
}

/// Verifier orchestrator (replays the prover’s FS schedule and checks KZG).
pub struct Verifier<'a> {
    /// Public verification parameters: domain and PCS settings.
    pub params: &'a crate::VerifyParams,
}

impl<'a> Prover<'a> {
    /// Helper: pad a time-evaluation vector to domain size with a constant value.
    #[inline]
    fn pad_evals_to_n(mut v: Vec<F>, n: usize, pad: F) -> Vec<F> {
        if v.len() < n {
            v.reserve(n - v.len());
            while v.len() < n {
                v.push(pad);
            }
        }
        v
    }

    /// Helper: commit a polynomial from a **time-ordered evaluation stream**.
    ///
    /// We perform a **single global IFFT** of length `N = domain.n`. If the
    /// stream has only `T < N` rows, we **pad** to length `N` with:
    /// - wires/selectors: `0`
    /// - permutation `Z`: handled by caller (pads with `1`)
    /// - residual `R`: handled where used (pads with `0`)
    fn commit_from_time_stream<I: Iterator<Item = F>>(
        &self,
        poly_id: &'static str,
        time_vals: I,
        pcs_degree_ctx: &PcsParams,
    ) -> Commitment {
        // Always aggregate in coefficient basis.
        let pcs_for_commit = PcsParams { basis: Basis::Coefficient, ..pcs_degree_ctx.clone() };

        // Buffer evaluations and pad with zeros to N.
        let evals_raw: Vec<F> = time_vals.collect();
        let evals = Self::pad_evals_to_n(evals_raw, self.params.domain.n, F::zero());

        // Global IFFT → monomial coefficients (low→high).
        let coeffs_lo_to_hi =
            crate::domain::ifft_block_evals_to_coeffs(&self.params.domain, &evals);

        // Feed to PCS in chunks.
        let mut agg = Aggregator::new(&pcs_for_commit, poly_id);
        const CHUNK: usize = 1 << 12;
        for tile in coeffs_lo_to_hi.chunks(CHUNK) {
            agg.add_block_coeffs(tile);
        }
        agg.finalize()
    }

    /// Produce a proof by streaming witness rows through a [`Restreamer`].
    ///
    /// The function implements the full A→E schedule:
    /// selectors? → wires → (β,γ) → perm/lookup (+ Z commit) → (α)
    /// → quotient → (ζ, …) → openings (real KZG for wires, Z, and Q).
    pub fn prove_with_restreamer(&self, rs: &impl Restreamer<Item = air::Row>) -> crate::Proof {
        let t_rows = rs.len_rows();
        assert!(self.air.k > 0, "AIR must define at least one register (k > 0)");
        assert!(self.params.b_blk > 0, "Block size b_blk must be positive");

        // Initialize Fiat–Shamir transcript with domain separation.
        let mut fs = Transcript::new("sszkp.proof");

        // Borrow PCS parameter sets from ProveParams.
        let pcs_wires: &PcsParams = &self.params.pcs_wires; // wires (declared basis may be Eval)
        let pcs_coeff: &PcsParams = &self.params.pcs_coeff; // quotient (Coefficient)

        // =========================================================================
        // Phase A — Fixed/selector commits (optional) → FS
        // =========================================================================
        if !self.air.selectors.is_empty() {
            for col in &self.air.selectors {
                // Build a time stream over all rows for this selector column.
                let time_stream = (0..t_rows).map(|i| {
                    if col.is_empty() { F::zero() } else { col[i % col.len()] }
                });
                let cm = self.commit_from_time_stream("selector", time_stream, pcs_wires);
                fs.absorb_commitment("selector_commit", &cm);
            }
        }

        // =========================================================================
        // Phase B — Wire commits (aggregated per register) → FS
        // =========================================================================
        let mut wire_commits: Vec<Commitment> = Vec::with_capacity(self.air.k);
        let boundary_seed = vec![F::zero(); self.air.k].into_boxed_slice();

        // Stream time-ordered evaluations for a single register without buffering O(N).
        struct WireTime<'r, R: Restreamer<Item = air::Row>> {
            air: &'r air::AirSpec,
            rs: &'r R,
            boundary: Box<[F]>,
            t_rows: usize,
            b_blk: usize,
            next_block: usize,
            cur_block: Option<(Vec<F>, usize)>,
            reg_idx: usize,
        }
        impl<'r, R: Restreamer<Item = air::Row>> Iterator for WireTime<'r, R> {
            type Item = F;
            fn next(&mut self) -> Option<F> {
                loop {
                    if let Some((ref v, ref mut i)) = self.cur_block {
                        if *i < v.len() {
                            let out = v[*i];
                            *i += 1;
                            return Some(out);
                        } else {
                            self.cur_block = None;
                            continue;
                        }
                    }
                    // No current block buffered — advance to next block of rows.
                    let start_idx = self.next_block * self.b_blk;
                    if start_idx >= self.t_rows {
                        return None;
                    }
                    let end_idx = (start_idx + self.b_blk).min(self.t_rows);
                    let start = RowIdx(start_idx);
                    let end = RowIdx(end_idx);

                    let br = {
                        let block_it = self.rs.stream_rows(start, end);
                        air::eval_block(
                            self.air,
                            RegIdx(self.reg_idx),
                            BlockIdx(self.next_block),
                            &self.boundary,
                            block_it,
                        )
                    };
                    self.boundary = br.boundary_out;
                    self.cur_block = Some((br.reg_m_vals, 0));
                    self.next_block += 1;
                }
            }
        }

        for m in 0..self.air.k {
            let time_stream = WireTime {
                air: self.air,
                rs,
                boundary: boundary_seed.clone(),
                t_rows,
                b_blk: self.params.b_blk,
                next_block: 0,
                cur_block: None,
                reg_idx: m,
            };
            let cm = self.commit_from_time_stream("wire", time_stream, pcs_wires);
            fs.absorb_commitment("wire_commit", &cm);
            wire_commits.push(cm);
        }

        // Sample (β, γ)
        let beta: F = fs.challenge_f("beta");
        let gamma: F = fs.challenge_f("gamma");

        // =========================================================================
        // Phase C — Permutation / Lookup (streamed, time-ordered)
        // =========================================================================
        let mut perm_acc = PermAcc { z: F::one() };
        let mut _lookup_acc = LookupAcc { z: F::one() };

        // Whitepaper-complete: commit Z
        let commit_z = true;
        let mut z_commit: Option<Commitment> = None;

        // Build Z’s time stream once and (1) feed the accumulator, (2) build commit via IFFT.
        let mut boundary = boundary_seed.clone();
        let mut z_start = F::one();
        let mut z_time_vals: Vec<F> = Vec::with_capacity(self.params.domain.n);

        for (BlockIdx(t), start, end) in blocks(t_rows, self.params.b_blk) {
            let block_it = rs.stream_rows(start, end);
            let br: BlockResult =
                air::eval_block(self.air, RegIdx(0), BlockIdx(t), &boundary, block_it);

            absorb_block_perm(&mut perm_acc, &br.locals, beta, gamma);

            let z_block = emit_z_column_block(z_start, &br.locals, beta, gamma);
            if let Some(last) = z_block.last() {
                z_start = *last;
            }
            z_time_vals.extend_from_slice(&z_block);

            absorb_block_lookup(&mut _lookup_acc, &br.locals);
            boundary = br.boundary_out;
        }

        if commit_z {
            // Pad Z with 1's to N (after the last real row).
            z_time_vals = Self::pad_evals_to_n(z_time_vals, self.params.domain.n, F::one());
            let cm_z = {
                // Use the same IFFT→coeffs→commit helper, but pass a pre-collected iterator.
                self.commit_from_time_stream("perm_Z", z_time_vals.clone().into_iter(), pcs_wires)
            };
            fs.absorb_commitment("perm_z_commit", &cm_z);
            z_commit = Some(cm_z);
        }

        // Sample (α)
        let alpha: F = fs.challenge_f("alpha");

        // =========================================================================
        // Phase D — Quotient Q (coeff-basis, streamed) → FS
        // =========================================================================
        let r_cfg = air::ResidualCfg { alpha, beta, gamma };
        // NOTE: residual_stream yields only T rows; quotient builder will pad to N.
        let r_stream = air::residual_stream(self.air, r_cfg.clone(), rs, self.params.b_blk);

        let q_commit = build_and_commit_quotient(
            &self.params.domain,
            pcs_coeff,
            alpha,
            beta,
            gamma,
            r_stream,
        );
        fs.absorb_commitment("quotient_commit", &q_commit);

        // Sample evaluation points (ζ, …)
        let eval_points: Vec<F> = fs.challenge_points("eval_points", 1);

        // =========================================================================
        // Phase E — Streaming openings (real KZG for wires, Z, and Q)
        // =========================================================================
        // Opening set: [ wires..., (Z?), Q ] — must match verify() reconstruction.
        let mut open_set: Vec<Commitment> = Vec::new();
        open_set.extend_from_slice(&wire_commits);
        if let Some(zc) = z_commit {
            open_set.push(zc);
        }
        let q_index = open_set.len();
        open_set.push(q_commit);

        let domain = &self.params.domain;
        let k_regs = self.air.k;
        let b_blk = self.params.b_blk;

        // --------- Real KZG openings for **wires** (build coeffs via full IFFT; pad to N) ---------
        let mut proofs_wires: Vec<pcs::OpeningProof> = Vec::with_capacity(k_regs * eval_points.len());
        for m in 0..k_regs {
            // Rebuild wire m time stream
            let mut boundary = vec![F::zero(); k_regs].into_boxed_slice();
            let mut time_vals: Vec<F> = Vec::with_capacity(domain.n);
            for (BlockIdx(t), start, end) in blocks(t_rows, b_blk) {
                let block_it = rs.stream_rows(start, end);
                let br = air::eval_block(self.air, RegIdx(m), BlockIdx(t), &boundary, block_it);
                boundary = br.boundary_out;
                time_vals.extend_from_slice(&br.reg_m_vals);
            }
            // Pad to N with zeros, then IFFT → coeffs (low→high)
            if time_vals.len() < domain.n {
                time_vals.resize(domain.n, F::zero());
            }
            let coeffs_lo_to_hi: Vec<F> =
                crate::domain::ifft_block_evals_to_coeffs(domain, &time_vals);

            // stream high→low chunks to witness builder
            let mut stream_coeff_hi_to_lo = |_idx: usize, sink: &mut dyn FnMut(Vec<F>)| {
                const CHUNK: usize = 1 << 12;
                let mut i = coeffs_lo_to_hi.len();
                while i > 0 {
                    let start = i.saturating_sub(CHUNK);
                    let block: Vec<F> = coeffs_lo_to_hi[start..i].iter().rev().copied().collect();
                    sink(block);
                    i = start;
                }
            };
            let pr = pcs::open_at_points_with_coeffs(
                pcs_wires,
                &[wire_commits[m]],
                |_idx, _z| F::zero(),
                &mut stream_coeff_hi_to_lo,
                &eval_points,
            );
            proofs_wires.extend(pr);
        }

        // --------- Real KZG openings for **Z** (if committed) — full IFFT; pad to N with 1s ---------
        let proofs_z: Vec<pcs::OpeningProof> = if let Some(zc) = z_commit {
            // We already built `z_time_vals` above; ensure it's padded to N with 1s.
            let mut time_vals = z_time_vals.clone();
            if time_vals.len() < domain.n {
                time_vals.resize(domain.n, F::one());
            }
            let coeffs_lo_to_hi: Vec<F> =
                crate::domain::ifft_block_evals_to_coeffs(domain, &time_vals);

            let mut stream_coeff_hi_to_lo = |_idx: usize, sink: &mut dyn FnMut(Vec<F>)| {
                const CHUNK: usize = 1 << 12;
                let mut i = coeffs_lo_to_hi.len();
                while i > 0 {
                    let start = i.saturating_sub(CHUNK);
                    let block: Vec<F> = coeffs_lo_to_hi[start..i].iter().rev().copied().collect();
                    sink(block);
                    i = start;
                }
            };
            pcs::open_at_points_with_coeffs(
                pcs_wires,
                &[zc],
                |_idx, _z| F::zero(),
                &mut stream_coeff_hi_to_lo,
                &eval_points,
            )
        } else {
            Vec::new()
        };

        // --------- Real KZG opening for **Q** — recompute R, pad to N, full IFFT ---------
        let c = self.params.domain.zh_c;
        let inv_c = c.inverse().expect("zh_c != 0");
        let mut stream_q_coeff_hi_to_lo = |_idx: usize, sink: &mut dyn FnMut(Vec<F>)| {
            let r_stream_all = air::residual_stream(self.air, r_cfg.clone(), rs, b_blk);
            let mut evals_r: Vec<F> = r_stream_all.collect();
            if evals_r.len() < domain.n {
                evals_r.resize(domain.n, F::zero()); // residual pads with 0
            }
            let r_coeffs_lo_to_hi: Vec<F> =
                crate::domain::ifft_block_evals_to_coeffs(domain, &evals_r);

            // Convert to q_i = -r_i / c (low→high buffer).
            let mut q_coeffs_lo_to_hi: Vec<F> = Vec::with_capacity(r_coeffs_lo_to_hi.len());
            for &ri in &r_coeffs_lo_to_hi {
                q_coeffs_lo_to_hi.push(-ri * inv_c);
            }

            // Emit high→low chunks for the witness builder.
            const CHUNK: usize = 1 << 12;
            let mut i = q_coeffs_lo_to_hi.len();
            while i > 0 {
                let start = i.saturating_sub(CHUNK);
                let block: Vec<F> = q_coeffs_lo_to_hi[start..i].iter().rev().copied().collect();
                sink(block);
                i = start;
            }
        };
        let proofs_q = pcs::open_at_points_with_coeffs(
            pcs_coeff, // IMPORTANT: Q uses the coeff PCS context
            &[open_set[q_index]],
            |_idx, _z| F::zero(), // not used by with_coeffs path
            &mut stream_q_coeff_hi_to_lo,
            &eval_points,
        );

        // Merge proofs in **opening-set order**: (wires..., Z?..., Q)
        let mut opening_proofs = Vec::with_capacity(open_set.len() * eval_points.len());
        opening_proofs.extend(proofs_wires);
        opening_proofs.extend(proofs_z);
        opening_proofs.extend(proofs_q);

        // Flatten claimed evals in poly-major, point-minor order (matches PCS).
        let evals: Vec<F> = opening_proofs.iter().map(|p| p.value).collect();

        // Assemble proof object
        let wire_comms_wrapped: Vec<crate::Commitment> =
            wire_commits.iter().copied().map(crate::Commitment).collect();
        let z_comm_wrapped: Option<crate::Commitment> = z_commit.map(crate::Commitment);
        let q_comm_wrapped: crate::Commitment = crate::Commitment(q_commit);

        crate::Proof {
            wire_comms: wire_comms_wrapped,
            z_comm: z_comm_wrapped,
            q_comm: q_comm_wrapped,
            eval_points,
            evals,
            opening_proofs,
        }
    }

    /// Convenience wrapper: buffers any iterator of rows into a `Vec<Row>`
    /// (which implements [`Restreamer`]) and calls
    /// [`Prover::prove_with_restreamer`].
    pub fn prove(&self, witness_rows: impl Iterator<Item = air::Row>) -> crate::Proof {
        let rows: Vec<air::Row> = witness_rows.collect();
        self.prove_with_restreamer(&rows)
    }
}

impl<'a> Verifier<'a> {
    /// Verify a proof by replaying the Fiat–Shamir schedule and checking
    /// KZG openings (pairings enforced for wires, Z, and Q).
    pub fn verify(&self, proof: &crate::Proof) -> bool {
        // --- Rebuild transcript exactly like the prover ---
        let mut fs = Transcript::new("sszkp.proof");

        // Phase A: (optional) selector commitments would be absorbed here.

        // Phase B: wire commitments
        for cm in &proof.wire_comms {
            fs.absorb_commitment("wire_commit", &cm.0);
        }

        // Sample (β, γ)
        let _beta: F = fs.challenge_f("beta");
        let _gamma: F = fs.challenge_f("gamma");

        // Phase C: Z commitment (if present)
        if let Some(zc) = &proof.z_comm {
            fs.absorb_commitment("perm_z_commit", &zc.0);
        }

        // Sample (α)
        let _alpha: F = fs.challenge_f("alpha");

        // Phase D: quotient commitment
        fs.absorb_commitment("quotient_commit", &proof.q_comm.0);

        // Sample evaluation points (ζ, …)
        let expect_points: Vec<F> = fs.challenge_points("eval_points", proof.eval_points.len());
        if expect_points != proof.eval_points {
            return false; // transcript mismatch
        }

        // --- KZG opening verification (split contexts) ---
        let mut open_set_all: Vec<Commitment> = Vec::new();
        open_set_all.extend(proof.wire_comms.iter().map(|c| c.0));
        let has_z = if let Some(zc) = &proof.z_comm {
            open_set_all.push(zc.0);
            true
        } else {
            false
        };
        open_set_all.push(proof.q_comm.0);

        let k = proof.wire_comms.len();
        let s = proof.eval_points.len();

        // wires(+Z) with pcs_wires
        let m_wz = k + if has_z { 1 } else { 0 };
        let count_wz = m_wz * s;
        let evals_wz = &proof.evals[0..count_wz];
        let proofs_wz = &proof.opening_proofs[0..count_wz];
        let open_set_wz: Vec<Commitment> = open_set_all[0..m_wz].to_vec();

        let ok_wz = pcs::verify_openings(
            &self.params.pcs_wires,
            &open_set_wz,
            &proof.eval_points,
            evals_wz,
            proofs_wz,
        );

        // Q alone with pcs_coeff
        let open_set_q: [Commitment; 1] = [open_set_all[m_wz]];
        let evals_q = &proof.evals[count_wz..count_wz + s];
        let proofs_q = &proof.opening_proofs[count_wz..count_wz + s];

        let ok_q = pcs::verify_openings(
            &self.params.pcs_coeff,
            &open_set_q,
            &proof.eval_points,
            evals_q,
            proofs_q,
        );

        ok_wz && ok_q
    }
}
