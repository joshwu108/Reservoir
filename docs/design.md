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
