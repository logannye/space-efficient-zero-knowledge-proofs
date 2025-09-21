#!/usr/bin/env bash
# scripts/test_sszkp.sh
# End-to-end tests for the sublinear-space ZK system (whitepaper-complete).
# Uses the dev SRS so no external SRS files are needed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

banner() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
pass()   { printf "\033[1;32m✔ %s\033[0m\n" "$*"; }
fail()   { printf "\033[1;31m✘ %s\033[0m\n" "$*"; }

run_case() {
  local name="$1"; shift
  local prover_args=("$@")

  banner "Case: $name"
  rm -f proof.bin

  # Prover: streams witness, commits wires + Z, builds Q, opens at ζ
  cargo run --quiet --features dev-srs --bin prover -- "${prover_args[@]}"

  if [[ ! -f proof.bin ]]; then
    fail "prover did not write proof.bin"
    exit 1
  fi

  # Verifier: replays FS, checks KZG pairings on wires/Z/Q and algebra at ζ
  set +e
  cargo run --quiet --features dev-srs --bin verifier -- --rows "$(echo "${prover_args[@]}" | sed -n 's/.*--rows \([0-9]\+\).*/\1/p')" \
    --basis "$(echo "${prover_args[@]}" | sed -n 's/.*--basis \([a-zA-Z]\+\).*/\1/p')" >/dev/null
  local rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    fail "verification FAILED for ${name}"
    exit 1
  else
    pass "verification OK for ${name}"
  fi
}

tamper_should_fail() {
  banner "Tamper test: corrupt one byte of proof.bin (should FAIL)"
  [[ -f proof.bin ]] || { fail "no proof.bin to tamper with"; exit 1; }

  # Flip a byte near the middle of the file
  local size
  size=$(wc -c < proof.bin)
  local off=$((size/2))
  # xor with 0x01 at offset
  printf '\x01' | dd of=proof.bin bs=1 seek="$off" count=1 conv=notrunc >/dev/null 2>&1

  set +e
  cargo run --quiet --features dev-srs --bin verifier -- --rows "${ROWS}" --basis "${BASIS}" >/dev/null
  local rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    fail "tampered proof unexpectedly VERIFIED"
    exit 1
  else
    pass "tampered proof correctly rejected"
  fi
}

banner "Build (dev-srs)"
cargo build --quiet --bins --features dev-srs
pass "build succeeded"

# ---------- Scenarios ----------
# We vary rows, block sizes (b-blk), and wire basis to hit the streaming paths.
# All runs: Z committed, lookups off (default), selectors absent.

# Case 1: eval-basis wires, moderate block size
ROWS=1024
BASIS=eval
run_case "eval-basis wires, b_blk=128, rows=1024" \
  --rows "$ROWS" --b-blk 128 --k 3 --basis "$BASIS"

tamper_should_fail

# Case 2: coeff-basis wires, smaller block size (stress streaming)
ROWS=1536
BASIS=coeff
run_case "coeff-basis wires, b_blk=64, rows=1536" \
  --rows "$ROWS" --b-blk 64 --k 3 --basis "$BASIS"

# Case 3: eval-basis wires, larger block size (near-domain)
ROWS=2048
BASIS=eval
run_case "eval-basis wires, b_blk=256, rows=2048" \
  --rows "$ROWS" --b-blk 256 --k 4 --basis "$BASIS"

banner "All tests passed 🎉"
