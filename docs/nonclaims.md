# Non-Claims

This document explicitly states what `reservoir` does NOT establish.

## What this repository does NOT claim

### 1. Performance

This is a **correctness-oriented research codebase**, not a performance result.
The exact-arithmetic buffer using Python integers and `fractions.Fraction` is
orders of magnitude slower than float-based implementations. This is expected
and intentional. No throughput, latency, or scalability claims are made.

### 2. Large-Scale RL Training Quality

We do not claim that using exact PER improves or changes RL training quality
at scale. The buffer correctness proofs apply to the sampling semantics only.
Whether exact sampling meaningfully affects agent performance in practice is
a separate empirical question not addressed here.

### 3. Distributed Buffers

This is a single-process, single-machine buffer. No claims about distributed
experience replay, multi-actor systems, or network-replicated buffers are made.

### 4. Security Boundary

The keyed BLAKE2b draw is **deterministic replay identity**, not a cryptographic
or anti-tamper security boundary. The attestation chain is a consistency
verification tool, not a tamper-proof audit log. A sufficiently privileged
adversary with filesystem access can forge attestation records by computing valid
BLAKE2b digests. We make no claims about security against such adversaries.

### 5. α-Exponentiation Semantics

The declared priority for transition i is proportional to the **binary64 value**
of `p_i^α`, not the true real value `p_i^α`. The float64-once integerization
boundary is declared in `docs/design.md`. The resulting distribution is exactly
the declared distribution (T1), but it differs from a hypothetical exact-real
distribution. This gap is bounded by the ULP error of float64 `pow()`, which
is at most 1 ULP ≈ 2^(-52) relative error on the exponentiated value.

### 6. Crash Durability on Non-APFS Filesystems

Crash-atomicity evidence is established on **macOS/APFS with `F_FULLFSYNC`**.
Power-loss durability on Linux (where `fsync()` semantics may differ by kernel
and filesystem), Windows, or other filesystems is **not tested** and not claimed.
The `os.fsync()` fallback on Linux is provided for correctness of the logical
protocol, not as evidence of physical durability on those platforms.

### 7. β Exponentiation for IS Weights

The IS weight computation uses the same float64-once boundary as α-exponentiation.
The declared IS weights are exact Fraction normalizations of float64 evaluations
of `(N·P(i))^(-β)`. The gap from the true real IS weights is bounded by 1 ULP
of float64 `pow()`.

### 8. TLA+ Model Scope

The TLA+ model (`spec/ReplayLifecycle.tla`) uses a finite scope (capacity 2,
2-value priority set, ≤3 operations). It establishes the safety properties
within that finite scope only. It does not constitute a proof for all possible
buffer sizes, priority values, or operation sequences. It is a falsification tool:
if the model checker finds a counterexample in the small scope, the protocol is wrong.
If it finds no counterexample, that is evidence (not proof) of correctness.
