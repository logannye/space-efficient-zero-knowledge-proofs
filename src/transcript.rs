//! Fiat–Shamir transcript with domain separation
//!
//! This module implements a deterministic, domain-separated Fiat–Shamir
//! transcript. Challenges are derived by hashing the running state with
//! a per-draw counter and reducing 64 bytes of BLAKE3 XOF output modulo
//! the BN254 scalar field (`Fr`) in a constant-time-ish manner using
//! arkworks’ `from_le_bytes_mod_order`.
//!
//! Whitepaper constraints honored here:
//! - Commitments are absorbed using **compressed G1** encoding.
//! - Byte ordering for aux bytes is explicit and length-delimited to pin
//!   transcript identity.
//! - Challenge derivation uses a **fixed DST** and a **monotone counter**,
//!   ensuring stable bindings across implementations.

#![forbid(unsafe_code)]

use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use blake3::Hasher;
use std::io::Read;

use crate::{pcs, F};

/// Fiat–Shamir transcript with domain separation.
///
/// Internals include a running hash (BLAKE3), a construction label (DST),
/// and a monotone counter to derive multiple challenges in sequence.
pub struct Transcript {
    /// Domain-separation string for this transcript instance.
    label: &'static str,
    /// Running hash state (BLAKE3).
    hasher: Hasher,
    /// Monotone counter for challenge derivations.
    ctr: u64,
}

impl Transcript {
    /// Create a new transcript with a domain-separation `label`.
    pub fn new(label: &'static str) -> Self {
        let mut hasher = Hasher::new();
        // Domain separation preamble: fixed prefix + label.
        hasher.update(b"SSZKP.transcript.v1");
        hasher.update(label.as_bytes());
        Self {
            label,
            hasher,
            ctr: 0,
        }
    }

    /// Absorb a PCS commitment using **compressed G1** encoding.
    ///
    /// The `label` here is an additional domain separator for this item
    /// (e.g., "wire_commitments", "perm_commitment", "quotient_commitment").
    pub fn absorb_commitment(&mut self, label: &'static str, c: &pcs::Commitment) {
        let mut bytes = Vec::with_capacity(48); // BN254 compressed G1 is ~48 bytes
        c.0.serialize_compressed(&mut bytes)
            .expect("G1 serialization should not fail");
        self.absorb_bytes(label, &bytes);
    }

    /// Absorb arbitrary bytes with an item label (length-delimited).
    ///
    /// The caller must ensure the bytes conform to the baseline encoding
    /// (e.g., big-endian scalars) to preserve transcript identity.
    pub fn absorb_bytes(&mut self, label: &'static str, bytes: &[u8]) {
        // Item preamble: tag + label + length + data.
        self.hasher.update(b"item:");
        self.hasher.update(label.as_bytes());
        self.hasher.update(b":len:");
        self.hasher.update(&(bytes.len() as u64).to_be_bytes());
        self.hasher.update(b":data:");
        self.hasher.update(bytes);
    }

    /// Derive a single field challenge `F` from the current transcript state.
    ///
    /// Uses a fixed per-draw DST and a monotone counter. Implemented via:
    ///  1) Clone running state
    ///  2) Absorb `challenge_f` label + draw counter
    ///  3) BLAKE3 XOF → 64 bytes
    ///  4) Reduce with `Fr::from_le_bytes_mod_order`
    pub fn challenge_f(&mut self, label: &'static str) -> F {
        let out = hash_to_field(&self.hasher, self.label, label, self.ctr, 1);
        self.ctr = self.ctr.wrapping_add(1);
        out[0]
    }

    /// Derive `k` field challenges (e.g., evaluation points `(ζ, …)`).
    ///
    /// Each element is derived from the same running state plus a per-draw
    /// counter to ensure unique bindings. We produce `k` distinct field
    /// elements in one shot using a single XOF stream per call.
    pub fn challenge_points(&mut self, label: &'static str, k: usize) -> Vec<F> {
        let out = hash_to_field(&self.hasher, self.label, label, self.ctr, k);
        self.ctr = self.ctr.wrapping_add(1);
        out
    }
}

// ------------------------ Internals ------------------------

/// Derive `k` field elements from (a clone of) `base` using a fixed DST.
///
/// We do **not** mutate the running hasher; we clone it to bind to the
/// current transcript state, then append per-call DST: construction label,
/// per-call label, and the monotone draw counter.
///
/// We then use BLAKE3 XOF to generate `k * 64` bytes and reduce each 64-byte
/// chunk modulo `Fr` (little-endian) via `from_le_bytes_mod_order`.
fn hash_to_field(
    base: &Hasher,
    tlabel: &'static str,
    label: &'static str,
    ctr: u64,
    k: usize,
) -> Vec<F> {
    // Clone running state to preserve transcript state for future absorbs.
    let mut h = base.clone();
    // Challenge DST
    h.update(b"challenge:");
    h.update(b"SSZKP.v1");
    h.update(b":tlabel:");
    h.update(tlabel.as_bytes());
    h.update(b":label:");
    h.update(label.as_bytes());
    h.update(b":ctr:");
    h.update(&ctr.to_be_bytes());

    // XOF → k * 64 bytes
    let mut xof = h.finalize_xof();
    let mut out = Vec::with_capacity(k);
    let mut buf = [0u8; 64];
    for _ in 0..k {
        let _ = xof.read(&mut buf);
        // Reduce 64 bytes → field using arkworks canonical reduction.
        // Use little-endian reduction (constant-time-ish in arkworks).
        let f = F::from_le_bytes_mod_order(&buf);
        out.push(f);
    }
    out
}
