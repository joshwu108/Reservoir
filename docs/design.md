# Reservoir Design Document

## 1. Exact Arithmetic Boundary

All decision paths (priority comparison, tree traversal, sampling, IS weight computation)
use Python integers and `fractions.Fraction` exclusively. Floats enter exactly once:
as raw TD-error inputs. They are immediately converted to exact rationals via
`fractions.Fraction(float_value)`, which is exact for every finite binary64 value
(binary64 is a rational number with denominator a power of 2).

Infinities and NaNs are rejected loudly with `ValueError`.

## 2. α-Exponentiation Design Decision

### The Problem

Schaul et al. 2016 PER defines priority `p_i = |δ_i| + ε` and sampling probability
`P(i) = p_i^α / Σ_k p_k^α`. For the canonical α = 0.6, `p_i^α` is irrational for
almost all rational `p_i`. "Exact arithmetic" requires a defined boundary.

### Three Options

#### Option (a): Rational α with Exact Integer Roots

Restrict α to rationals p/q with small denominator (e.g., 3/5 for α≈0.6).
Compute `p_i^(3/5)` as: integer-floor of `(p_i^3)^(1/5)` via Newton iteration
with an exactness certificate.

**Correctness trade-offs:**
- The declared distribution uses floor semantics, which is not the same as the
  Schaul et al. float distribution — different semantics, not merely a more exact
  version of the same thing.
- Exactness certificate is non-trivial: must verify `x^5 ≤ p_i^3 < (x+1)^5`.
- Restricted α values only (α must be a rational with small denominator).
- Computation is O(log p_i) per priority, potentially slow for large priorities.
- The declared distribution *is* exactly implementable, so T1 holds.

**When appropriate:** Research on exact floor-semantics PER; α restricted to Q.

#### Option (b): Float64-Once Integerization (RECOMMENDED)

Compute `p_i^α` in float64 once, then exactly integerize the binary64 result.
Every downstream operation (tree sums, sampling, IS weights) is exact integer
or Fraction arithmetic. The declared distribution is "proportional to the
binary64 value of p_i^α", not "proportional to the true real p_i^α".

**Correctness trade-offs:**
- The exact/float distinction lives entirely at one declared boundary.
- Downstream is provably exact: tree arithmetic is integer, sampling is integer.
- The priority values stored are exactly the binary64 bit-patterns of p_i^α,
  interpreted as integers (via `float.hex()` mantissa extraction or
  `struct.pack`). This is documented as the declared semantics.
- Matches the spirit of honest-boundary style: floats enter once, immediately
  frozen to an integer, and everything downstream is exact.
- For T2 (float divergence campaign), the float baseline's p_i^α values
  are used directly in the float tree, while our tree uses the same
  binary64-integerized values — apples-to-apples except for tree arithmetic.

**Integerization method:** Scale the float64 result by `2^52` (the mantissa bits)
to get an exact positive integer, or more simply: use `fractions.Fraction(float_val)`
to get the exact rational, then multiply numerator/denominator to extract an integer
priority that preserves relative ordering. In practice: priority integer =
`int(p_i^α * SCALE)` where SCALE is a fixed power of 2 chosen so the smallest
non-epsilon priority rounds to ≥ 1. Equivalently, use the bit-representation of
the float64 as a 53-bit mantissa integer (with implicit leading 1) scaled by the
appropriate power of 2 from the exponent field — this gives an exact integer
representing the binary64 value exactly.

**This is our chosen approach (Option b).**

#### Option (c): Interval Arithmetic with Certified Tie-Refinement (Stretch)

Compute `p_i^α` with outward-rounding interval arithmetic (using `mpmath` or a
custom interval type). If the sampling decision is ambiguous (two candidates
straddle the draw boundary), narrow the interval until the comparison is
unambiguous.

**Correctness trade-offs:**
- Termination: tie-refinement may not terminate for adversarially chosen priorities.
- Complexity: each sampling step may require arbitrary-precision computation.
- Semantics: closer to "true real distribution" but with unbounded latency.
- Genuinely correct in the sense that the *real* probability ratios are used.

**When appropriate:** When true-real-semantics are required and worst-case latency
is acceptable. Not chosen here because the unbounded computation cost conflicts with
the buffer's reliability goals.

### Chosen Implementation: Option (b)

We implement Option (b). The integerization procedure:

```python
import struct, math

PRIORITY_SCALE_BITS = 52  # mantissa bits in binary64
PRIORITY_SCALE = 1 << PRIORITY_SCALE_BITS  # = 2^52

def float_to_priority_int(x: float, alpha: float) -> int:
    """
    Compute x^alpha in float64, then exactly integerize the binary64 result.

    The declared priority integer is: round(x^alpha * 2^52) where x^alpha
    is computed in IEEE 754 double precision. This is exact for the
    binary64 result: every finite float f has exact integer representation
    int(f * 2^52) when f is in [0, 2^11).

    For larger values, we extract the exact integer mantissa using struct.
    """
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"Priority must be finite non-negative, got {x}")
    if x == 0.0:
        return 0
    powered = x ** alpha
    if not math.isfinite(powered):
        raise ValueError(f"x^alpha overflow: x={x}, alpha={alpha}")
    # Extract exact integer: Fraction(float) is exact, scale up by 2^52
    # to clear the denominator for normalized floats in [1, 2).
    from fractions import Fraction
    frac = Fraction(powered)
    # Scale to integer: multiply by 2^52, take floor
    scaled = frac * PRIORITY_SCALE
    return int(scaled)  # exact — Fraction.__int__ truncates
```

The exact conversion is documented here. The declared distribution for the
integerized sum-tree is proportional to `int(p_i^α * 2^52)`.

## 3. Durability Protocol

### POSIX Filesystem Crash Atomicity

The durable buffer uses a write-ahead log (WAL) with the following commit protocol:

1. Write intent record to `intent.json.tmp`
2. fsync `intent.json.tmp` (F_FULLFSYNC on Darwin)
3. Write segment files with data
4. fsync each segment file (F_FULLFSYNC on Darwin)
5. `rename(intent.json.tmp, intent.json)` — atomic commit point
6. fsync parent directory (F_FULLFSYNC on Darwin)
7. Delete or supersede old intent record

**macOS/APFS note:** `fsync()` on macOS does not guarantee data reached durable
storage (it only flushes to the kernel buffer cache). We use `fcntl(fd, F_FULLFSYNC)`
on Darwin (detected via `sys.platform == 'darwin'`), falling back to `os.fsync()`
on Linux. This is documented in `docs/nonclaims.md`.

### Recovery

On startup, the buffer checks:
1. If no `intent.json` exists: buffer is in a consistent pre-operation state.
2. If `intent.json` exists and all referenced segments exist and are complete:
   apply the operation (post-commit recovery).
3. If `intent.json` exists but segments are incomplete or missing:
   discard intent, restore pre-operation state.

The recovery is conservative: if any ambiguity exists, fall back to pre-state.

## 4. Attestation Schema

### MutationRecord

Every buffer mutation (insert, update, evict) appends a `MutationRecord`:

```json
{
  "digest": "<hex BLAKE2b of canonical JSON of this record minus digest field>",
  "op": "insert" | "update" | "evict",
  "index": <int>,
  "old_priority_int": "<int as string>",
  "new_priority_int": "<int as string>",
  "op_counter": <int>,
  "prev_digest": "<hex or 'genesis'>"
}
```

### SampleAttestation

Every sampled batch appends a `SampleAttestation`:

```json
{
  "digest": "<hex BLAKE2b>",
  "op_counter": <int>,
  "root_total": "<int as string>",
  "samples": [
    {
      "leaf_index": <int>,
      "draw_int": "<int as string>",
      "prob_num": "<int as string>",
      "prob_den": "<int as string>",
      "is_weight_num": "<int as string>",
      "is_weight_den": "<int as string>",
      "rejection_count": <int>
    }
  ],
  "prev_digest": "<hex or chain head digest>"
}
```

Canonical serialization: `json.dumps(record, sort_keys=True, separators=(',', ':'))`.
Integers are encoded as strings (to avoid JSON integer precision limits).
Digests are BLAKE2b-256 of the canonical UTF-8 encoding.

## 5. Keyed Draw

The deterministic draw is keyed by `(seed, buffer_id, op_counter)`. The draw
produces a uniform integer in `[0, N)` using:

1. Key = BLAKE2b-256(`seed || buffer_id || op_counter`, person=b'reservoir')
2. Generate blocks of 256 bits by BLAKE2b-256(key || block_counter)
3. Interpret blocks as big-endian integers
4. Rejection-sample: return first integer in block that is < N when taken mod 2^256
   (using standard rejection to get uniform distribution — reject values in the
   last partial block to avoid modulo bias)

The uniformity argument: for any N ≤ 2^256, the rejection rate is < 1/2^256 per
256-bit block except for the last incomplete group, where we reject the remainder.
Expected number of hash evaluations is < 2.

## 6. IS Weight Computation

The importance-sampling weight is:

```
w_i = (N · P(i))^(-β) / max_j w_j
```

where `P(i) = priority_int_i / root_total` (exact rational).

We compute this as an exact `Fraction`:

```python
from fractions import Fraction

def is_weight(N: int, priority_int: int, root_total: int, beta: float,
              max_priority_int: int) -> Fraction:
    # P(i) = priority_int / root_total  (exact)
    # w_i_unnorm = (N * P(i))^(-beta) — but beta may be non-integer
    # We use the same float64-once strategy: compute (N * P(i))^(-beta) in float64,
    # then represent as Fraction.
    p_i = Fraction(priority_int, root_total)
    n_p_i = N * p_i  # exact Fraction
    # Float-once boundary for (-beta) exponentiation
    w_unnorm_float = float(n_p_i) ** (-beta)
    w_unnorm = Fraction(w_unnorm_float)
    # max weight corresponds to min priority (max IS weight = min priority case)
    p_max = Fraction(max_priority_int, root_total)
    n_p_max = N * p_max
    w_max_float = float(n_p_max) ** (-beta)
    w_max = Fraction(w_max_float)
    return Fraction(w_unnorm, w_max)  # exact rational normalization
```

The float-once boundary for IS weights is declared here. All arithmetic after
the float64 evaluation of `(N·P(i))^(-β)` is exact Fraction arithmetic.

## 7. Age-Decayed Priorities in 64 Bits (Phase 1 design spike)

Status: primitive shipped in `src/reservoir/decay.py` with tests in
`tests/test_decay.py`. Not yet wired into any buffer, the attestation log, the
checker, or the C extension. Nothing in this section is a training-quality or
throughput claim.

### 7.1 Declared semantics

Integer parameters, fixed at construction (`DecayParams`):

| Symbol | Parameter | Meaning |
|---|---|---|
| h | `half_life` | model versions per halving of the sampling weight, 1 ≤ h ≤ 1024 |
| A | `max_policy_age` | an entry written at version t is live at version v iff v − t ≤ A |
| N | `capacity` | maximum number of leaves |
| P | `priority_bits` | base priority satisfies 0 ≤ q < 2^P (default 32) |
| Q | `priority_frac_bits` | fixed-point fraction bits of q (default 16) |
| F | `table_frac_bits` | fraction bits of the decay table (default 31) |
| R | `rebase_slack` | extra epochs of shift headroom (default 0) |

1. **Quantization — the only float boundary.** A finite raw priority x ≥ 0
   (the binary64 value of `p^α`, as in §2) becomes `q = 0` if `x = 0` and
   `q = max(1, floor(x · 2^Q))` if `x > 0`, computed exactly through
   `Fraction(x)`. The lower bound keeps every positive priority sampleable;
   only an exact zero is excluded. `q ≥ 2^P` is a `ValueError`; there is no
   upper clamp and no wrap.
2. **Decay table.** `T[k] = floor(2^(k/h) · 2^F)` for k in [0, h), where
   `2^(k/h)` is the true real value. `T[k]` is the unique integer with
   `T[k]^h ≤ 2^(k + F·h) < (T[k] + 1)^h`.
3. **Weight.** With epoch `E = t // h` and phase `k = t mod h`, the absolute
   weight of an entry is `W = floor(q · T[k] / 2^F) · 2^E`. The product is
   rounded by floor, once, before the epoch shift.
4. **Distribution.** Over live entries, `P(i) = W_i / Σ_j W_j` exactly.
5. **Stored leaf.** `leaf = floor(q · T[k] / 2^F) << (E − B)` for a base epoch
   B with `0 ≤ E − B ≤ S`, where `S = max_shift = ceil(A / h) + R`.

Consequences that are tested, not assumed: two entries with equal q whose
versions differ by m·h have weights in ratio exactly 2^m; a positive q never
yields a zero weight; W is non-decreasing in t for fixed q.

What this is **not**: it is not `p · exp(−Δ/τ)` evaluated in the reals. It is
base-2 decay with an integer half-life and floor rounding at two declared
points. FreshPER's τ maps to `h ≈ τ · ln 2` in the priority domain; because
FreshPER applies α after the decay, the half-life of the *sampling weight* is
`τ · ln 2 / α`. h is an integer, so τ is matched only approximately.

### 7.2 Bit budget (Question 1)

```
tree:     P + 1 + S + ceil(log2(N)) ≤ 64        S = ceil(A / h) + R
product:  P + F + 1 ≤ 64
```

Proof sketch. `T[k] < 2^(F+1)` and `q ≤ 2^P − 1` give
`floor(q·T[k]/2^F) < 2^(P+1)`, so every leaf is `< 2^(P+1+S)`. Any internal
node is a sum of at most N' leaves, where N' is the tree's power-of-two
capacity and `log2(N') = ceil(log2(N))`. The "+1" is the cost of the phase
within an epoch. The bound is within one bit of tight.

Both inequalities are checked in `DecayParams.__post_init__`; a violation is a
`ValueError` naming each term. The table bits F do not appear in the tree
budget: that is the purpose of flooring the product (see 7.5, alternative 2).

Supported ranges at the default P = 32 (`S + ceil(log2 N) ≤ 31`):

| capacity | max half-lives of spread (S) |
|---|---|
| 2^10 | 21 |
| 2^16 (covers 50K) | 15 |
| 2^20 | 11 |
| 2^24 | 7 |

At P = 24 each row gains 8. An entry 15 half-lives old has weight 2^−15 of a
fresh entry with the same base priority.

### 7.3 Rebase cost (Question 3)

A rebase by d epochs divides every live leaf by 2^d. Because every live leaf
is a multiple of 2^d, every internal node is too, so a rebase is one right
shift over the whole node array with no re-propagation. It is O(N) in touched
memory. Measured here with numpy on 2^21 uint64 nodes (a 2^20-leaf tree):
0.3 ms on the development machine — one measurement, not a benchmark.

Frequency: once every `(R + 1) · h` model versions. It is independent of the
number of inserts, updates and samples.

Avoiding the O(N) pass is possible with one sub-tree per epoch and an
(S+1)-entry top level weighted by `2^(E−B)`: a rebase then drops the oldest
sub-tree in O(1). It needs the same bit budget, more memory or a position
indirection, and a second traversal level. Not chosen: the flat shift is
cheap, rare, and keeps the tree identical to the existing one.

Ordering constraint: expired entries must be evicted (leaf set to 0) **before**
the shift. `rebase_priority` refuses a shift that would drop a set bit, but
divisibility is necessary, not sufficient, evidence of liveness; eviction is
decided by version.

### 7.4 Remaining questions

**Age on update (Question 4).** Default: an update keeps the original
version and changes only q. `version_after_update(...,
reset_age_on_update=True)` re-stamps instead. Source: arXiv 2604.16918 was
read through an automated summary of its HTML version, not line by line. It
defines t_i as "the global training step when trajectory i was collected" and
Δ_i = t − t_i, and states that base priorities may be recomputed after a
training step. No sentence about resetting t_i was found. The default follows
the definition; the paper does not state it explicitly.

**Attestation (Question 5) — proposal, not implemented.**

- New first record `decay_config`: `half_life`, `max_policy_age`, `capacity`,
  `priority_bits`, `priority_frac_bits`, `table_frac_bits`, `rebase_slack`,
  `reset_age_on_update`. The checker rebuilds `T` itself from `half_life` and
  `table_frac_bits` with integer arithmetic; the table is not shipped.
- `insert` / `update` records gain `base_priority_int` (q), `entry_version`
  (t) and `base_epoch` (B). The checker recomputes the leaf and requires it to
  equal `new_priority_int`.
- New record `advance_version`: `old_version`, `new_version`. The checker
  requires every subsequent `evict` with reason `"stale"` to satisfy
  `new_version − t > A`, and requires that no live entry is missing one.
- New record `rebase`: `old_base_epoch`, `new_base_epoch`,
  `root_total_before`, `root_total_after`. The checker shifts every replayed
  leaf, fails if any dropped bit is non-zero, and compares totals. One record
  per rebase, not N mutation records.
- `sample` records are unchanged.

Files that must change when this is wired in (not changed by this spike):
`attest.py` (`append_mutation` rejects unknown ops and has no fields for
q/t/B), `checker/verify.py` (unknown op is a `CheckerError`), and
`sumtree.c`/`sumtree.h` (nodes are `double`; a `uint64_t` variant is needed).
`ExactMinTree.INFINITY = 2^2048` must become `UINT64_MAX` in a fixed-width
min-tree.

**Quantization loss (Question 6).** The 2^52 scheme keeps every mantissa
bit for x ≥ 1 and floors below 2^−52. The default 16.16 format has absolute
step 2^−16 and range [0, 65536):

| raw priority x | q | relative step 1/q |
|---|---|---|
| 1e-6 | 1 (raised from 0) | 100% |
| 2.5e-4 (= (1e-6)^0.6) | 16 | 6.3% |
| 1e-3 | 65 | 1.5% |
| 1e-2 | 655 | 0.15% |
| 1.0 | 65 536 | 1.5e-5 |
| 1000 | 65 536 000 | 1.5e-8 |

So 32 bits lose about 36 bits of relative resolution at x = 1. For priorities
in roughly [1e-2, 6e4] the per-entry probability error is below 0.2%. It does
matter in two cases: (a) 0 < x < 2^−16 is raised to q = 1, so the entry stays
sampleable but all such priorities are indistinguishable and over-weighted
relative to their raw value — with α = 1 and ε = 1e-6 a zero-error entry hits
this; (b) x ≥ 65536 is rejected. Both are moved by Q (e.g. Q = 24 gives step 6e-8, range
[0, 256)). The product floor in 7.1(3) has absolute error below one unit of
the same grid, so it is the same order as the quantization error. Whether any
of this affects training is not measured and not claimed.

### 7.5 Chosen approach and rejected alternatives

Chosen: inflate-new-entries with exact epoch shifts, as proposed, with two
changes to the proposal.

**Change 1 — integer-root table instead of float64.** The proposal computed
`2^(k/h)` in float64. `pow` is not required to be correctly rounded and libm
implementations differ, so a checker on another machine could derive a
different table. §2 tolerates float `pow` because the result is recorded in
the log and never recomputed; a decay table would have to be either shipped
in the log or recomputed. The integer h-th root removes the dependency. On
the development machine the float64 table equals the integer table for
h ∈ {3, 347, 1000} at F = 31; that is an observation on one platform, not a
guarantee. Cost: table construction is 0.23 s at h = 1024 and about 10 s at
h = 4096, hence `MAX_HALF_LIFE = 1024`. Larger half-lives need a coarser
version unit.

**Change 2 — floor the product.** Rejected alternative: keep the full product
`q · T[k]`. It needs no rounding, but its budget is
`P + F + 1 + S + ceil(log2 N) ≤ 64`. At P = 32, F = 15 that leaves
`S + ceil(log2 N) ≤ 16`: a 2^16 buffer gets S = 0. For equal leaf width the
floored product is at least as close to the real-valued weight.

Other rejected alternatives:

- *Decay old entries in place* (multiply every leaf each version): O(N) per
  version and rounding error accumulates per step.
- *Lazy decay at sample time with stale internal nodes*: internal sums no
  longer equal the sum of current leaf weights, so `P(i) = leaf_i / total`
  is false.
- *Unbounded Python integers, never rebase*: exact, but leaf width grows by
  one bit per half-life without bound; cannot move to C.
- *Base-e table*: whole decay periods are no longer shifts, so rebasing is a
  multiplication with rounding and is not exact.

### 7.6 Verdict

Exact 64-bit age decay is viable. With 32-bit priorities it supports, for
example, 2^16 entries with 15 half-lives of spread or 2^20 entries with 11.
A FreshPER-scale configuration (h = 347, 50K entries, A = 15·h) uses exactly
64 bits.

Costs: priorities are 16.16 fixed point by default instead of 2^52-scaled;
the decay rate is base-2 with integer half-life ≤ 1024; one O(N) shift every
(R+1)·h versions; the decayed weight is floored once.

Implications for the rollout buffer: it must store (q, t) per entry, track the
current version and base epoch, evict stale entries before rebasing, and call
`pending_rebase_shift` before writing at a new version. The min-tree used for
IS-weight normalisation needs the same shift on rebase.

Implications for the C backend: the leaf is `((uint64_t)q * T[k]) >> F << s`
with q and T[k] both fitting in uint32 at the defaults; the tree needs a
`uint64_t` node type. The draw must also be produced as an integer below the
root total; that path is not examined here.

Interaction with the known DurableBuffer issue: a rebase rewrites every leaf,
so under the current full-state serialization it costs the same as any other
operation, but it must be a single durable operation (eviction plus shift
under one intent). An incremental WAL would need a `rebase` entry rather than
N leaf writes.

Not established by this spike: behaviour of variable-length trajectories,
IS weights under decay, any end-to-end buffer, and any C implementation.
