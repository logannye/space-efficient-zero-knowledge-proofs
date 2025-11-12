# SSZKP — Sublinear-Space Zero-Knowledge Proofs (Rust, KZG/BN254)

A reference implementation of the **sublinear-space ZKP** prover/Verifier described in our whitepaper: "Zero-knowledge Proofs in Sublinear Space" (https://arxiv.org/abs/2509.05326).
It realizes a **streaming prover that uses only *O(√T)* memory** over a trace of length *T*, while producing standard **KZG commitments** (BN254) for wires, the permutation accumulator `Z`, and the quotient `Q`. The design keeps **aggregate-only Fiat–Shamir** and never materializes whole polynomials.

---

## Why this matters

Traditional zk proving pipelines routinely buffer whole polynomials, forcing *O(T)* memory and large intermediate states. This repository demonstrates a practical alternative:

* **Sublinear space:** the active working set stays **O(√T)** using *blocked IFFTs* and streaming accumulators.
* **Production-style commitments:** standard **KZG** commitments/openings (pairing-checked) over **BN254**.
* **No full-poly buffers:** wires, `Z`, and `Q` are built and opened **without** holding entire vectors.
* **Deterministic dev SRS:** easy to run locally; switch to trusted SRS files for production.

If you’re building scalable zk systems, this repo shows how to restructure your pipeline around **streaming** and **aggregate-only FS** without giving up familiar cryptographic backends.

---

## Features at a glance

* **PCS:** KZG over BN254 with a **linear** interface and a streaming **Aggregator**.
* **Two commitment bases:** commit from **evaluation** (domain-aligned) or **coefficient** slices.
* **Openings:** real KZG openings for wires/`Z`/**`Q`**, with consistent witness construction.
* **Domain & transforms:** radix-2 blocked IFFT/NTT, **barycentric** eval for streaming points.
* **AIR & residuals:** small fixed-column AIR and permutation-coupled residual stream.
* **Scheduler:** five-phase A→E pipeline, aggregate-only Fiat–Shamir, strictly increasing time order.
* **CLI tools:** `prover` and `verifier` plus an end-to-end script.
* **Space profile:** peak memory **≈ O(b\_blk)** with `b_blk ≈ √T`.

---

## How it works (one-screen version)

1. **Phase A:** (Optional) commit selectors/fixed columns.
2. **Phase B:** **Wires** — stream a register’s evaluations block-by-block → **blocked IFFT** → feed **coeff tiles (low→high)** into PCS Aggregator.
3. **Phase C:** **Permutation accumulator `Z`** — stream locals, update `Z` on the fly and **emit the `Z` column in time order**, then commit via the same blocked IFFT path.
4. **Phase D:** **Quotient `Q`** — stream residual `R(ω^i)` and convert to `Q` coefficients online using `Z_H(X)=X^N−c` (**no full-poly buffers**).
5. **Phase E:** **Openings** — produce **real KZG** openings for wires, `Z`, and `Q` (witness `W = (f−f(ζ))/(X−ζ)`) and verify via pairings.

All Fiat–Shamir challenges are replayed by the verifier; pairing checks are always enforced. In dev builds, SRS is deterministic; in production, provide trusted SRS files.

---

## Quick start

### Prerequisites

* Rust (stable toolchain)
* No external SRS required for dev runs (deterministic in-crate SRS)

### Build & test (dev SRS)

```bash
# Clone, then:
cargo build --quiet --bins --features dev-srs

# End-to-end script runs three scenarios + a tamper test
scripts/test_sszkp.sh
```

Expected output (abridged):

```
✔ build succeeded
✔ verification OK for eval-basis wires, b_blk=128, rows=1024
✔ tampered proof correctly rejected
✔ verification OK for coeff-basis wires, b_blk=64, rows=1536
✔ verification OK for eval-basis wires, b_blk=256, rows=2048
==> All tests passed 
```

### Run the prover manually

```bash
cargo run --features dev-srs --bin prover -- \
  --rows 1024 --b-blk 128 --k 3 --basis eval
# writes proof.bin
```

### Run the verifier manually

```bash
cargo run --features dev-srs --bin verifier -- --rows 1024 --basis eval
# reads proof.bin and verifies
```

---

## Production SRS (trusted, non-dev)

In non-dev builds you must provide both **G1** and **G2** SRS files.

Prover:

```bash
cargo run --bin prover -- \
  --rows 1024 --b-blk 128 --k 3 --basis eval \
  --srs-g1 srs_g1.bin --srs-g2 srs_g2.bin
```

Verifier:

```bash
cargo run --bin verifier -- --rows 1024 --basis eval \
  --srs-g1 srs_g1.bin --srs-g2 srs_g2.bin
```

> **Format:** the SRS files are Arkworks-serialized vectors of affine powers.
> **G1:** `[τ^0]G1 … [τ^d]G1` (we use the degree bound you load).
> **G2:** a vector containing at least `[τ]G2` (we read element 1 or 0).

---

## Configuration knobs

* `--rows <T>`: total rows in the trace (domain size rounds up to power of two).
* `--b-blk <B>`: block size; pick **≈ √T** to achieve the sublinear memory bound.
* `--k <K>`: number of registers (columns) in the AIR.
* `--basis <eval|coeff>`: commitment basis for **wires** (Q is always **coeff**).
* `--selectors <FILE>`: optional selectors/fixed columns CSV (rows × S).
* `--omega <u64>`: override ω (power-of-two order must hold).
* `--coset <u64>`: reserved; current domain uses subgroup (`Z_H(X)=X^N−1`).

---

## Repository layout

* `src/pcs.rs` — KZG PCS (BN254), streaming **Aggregator**, real openings, pairings.
* `src/domain.rs` — domain `H`, barycentric weights, blocked NTT/IFFT.
* `src/air.rs` — tiny fixed-column AIR + residual stream + permutation coupling.
* `src/perm_lookup.rs` — permutation accumulator `Z` (lookups optional).
* `src/quotient.rs` — streaming quotient builder (**R→Q** tilewise).
* `src/scheduler.rs` — 5-phase orchestrator (aggregate-only FS, O(√T) space).
* `src/opening.rs` — streaming polynomial evaluation helpers (eval/coeff mode).
* `src/transcript.rs` — domain-separated FS transcript (BLAKE3→field).
* `src/stream.rs` — block partitioning + restreaming interfaces.
* `bin/prover.rs`, `bin/verifier.rs` — CLIs.
* `scripts/test_sszkp.sh` — end-to-end tests + tamper test.

---

## Using this in your own pipeline

* Treat the `Restreamer` trait as the integration seam: implement it to feed rows from your own storage (disk, network, GPU), all while keeping **O(b\_blk)** memory.
* Keep your **permutation/lookup** logic time-ordered; the accumulator state must evolve monotonically in `t`.
* When committing from **evaluations**, ensure your blocks align to the domain and use the provided **blocked IFFT** helpers to produce coeff tiles.
* For openings, prefer the **coeff-stream** path; the code adapts eval-streams internally when needed.

---

## Correctness & security notes

* Pairing checks are always enforced; the verifier **replays** the FS transcript and checks KZG equalities for wires, `Z`, and `Q`.
* The **tamper test** flips one byte in `proof.bin`—verification must fail.
* Dev SRS exists only for convenience; do not use dev mode in production.
* Algebraic identity at ζ (gate + perm coupling + boundary = `Z_H(ζ)·Q(ζ)`) is implemented; by default, selectors are optional and gates are minimal.

---

## Limitations & roadmap

* AIR is a compact demo; plug in your real selector/table wiring as needed.
* Lookup accumulator is feature-gated and intentionally minimal (demo path).
* Only BN254/KZG is shipped; adding Pallas/BLS12-381 is straightforward in this architecture.

---

## Getting help

* File issues for bugs or suggestions.
* PRs welcome—especially alternative domains, SRS loaders, or integration examples.

---

## Acknowledgments

This codebase follows the **aggregate-only Fiat–Shamir** and **streaming** discipline described in the whitepaper and demonstrates that **production-style commitments and sublinear space can coexist** in a practical Rust implementation.

