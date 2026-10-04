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

Status: primitive in `src/reservoir/decay.py`, wired into
`DecayedPriorityTree` (`decayed_tree.py`), `RolloutBuffer`
(`rollout_buffer.py`), the attestation log (§7.4, now implemented), the
independent checker (`checker/decay_replay.py`) and the durable buffer
(`durable_rollout.py`). Not in the C extension. Nothing in this section is a
training-quality or throughput claim.

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

**Attestation (Question 5) — implemented.** The records below are written by
`rollout_attest.py` and replayed by `checker/decay_replay.py`, which
re-derives the decay table and every leaf with its own integer arithmetic.
The checker also enforces the order `advance_version → evict(stale)* →
rebase? → writes`, rejects a rebase that is not due or that is skipped, and
verifies that capacity evictions were necessary (the inserts that follow
consume exactly the freed slots). 38 forged variants are rejected in the
mutation campaign.

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

Still to change for a C backend: `sumtree.c`/`sumtree.h` (nodes are
`double`; a `uint64_t` variant is needed) and `ExactMinTree.INFINITY =
2^2048`, which must become `UINT64_MAX` in a fixed-width min-tree. The
Python path keeps the big-integer sentinel.

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

## 8. RolloutBuffer

`RolloutBuffer` (`src/reservoir/rollout_buffer.py`) is the LLM-RL replay
buffer built on §7. Its rules, each tested in `tests/test_rollout_buffer.py`:

| Rule | Choice | Why |
|---|---|---|
| Leaf granularity | One sum-tree leaf per rollout; the `RolloutGroup` is shared metadata | Priorities and updates are per rollout; group statistics (mean reward, pass rate) are what strategies read |
| Priority pipeline | `score → score ** alpha (float, once) → quantize (16.16) → inflate (§7)` | Two declared float boundaries, then exact. `alpha` defaults to 1.0 for rollouts |
| Versions | `current_version` only moves forward; `add_group` and `sample` advance it. A group may arrive below the current version if not expired | Asynchronous rollout workers |
| Staleness | Entries older than `max_policy_age` are evicted on every advance, before any rebase | §7.3 ordering; evict-then-shift keeps the rebase exact |
| Capacity | Freed slots reused lowest-index first; when short, the lowest-version entries are evicted (ties: insertion order). A group never displaces entries newer than itself: `add_group` raises | A late group must not push out fresher data |
| Age on update | Kept, unless `reset_age_on_update=True` | §7.4 |
| Sampling | With replacement; `batch_size` may exceed the live count. Draws keyed on a per-rollout counter, so no two draws share a key whatever batch sizes are used | `ExactPERBuffer`'s `batch * batch_size + k` keys collide when batch sizes vary |
| IS weights | `(N·P(i))^-β / (N·P_min)^-β`, `P(i)` from the decayed leaf, min over positive leaves | Same formula as §6; zero-weight entries are live but excluded from the minimum |
| Errors | All validation before any mutation; a bad group, score or index leaves the buffer unchanged | Lets the durable wrapper treat an exception as "nothing happened" |

`DurableRolloutBuffer` (`durable_rollout.py`) commits each operation through
the §3 protocol with full-state snapshots (`state_dict`/`load_state_dict`),
including the attestation log; 70 SIGKILL tests in the crash campaign hit
every cut point during an `add_group` that evicts and rebases.

## 9. TRL Integration

`reservoir.integrations.trl` (`ReservoirReplay`, `ReservoirGRPOTrainer`)
connects `RolloutBuffer` to TRL's `GRPOTrainer`. Tested against TRL 1.13.0;
`_trl_compat.require_trl` refuses a TRL whose `GRPOTrainer` lacks any member
the adapter relies on and warns on an untested version.

### 9.1 Why not `GRPOWithReplayBufferTrainer`

TRL's experimental `GRPOWithReplayBufferTrainer` was removed from TRL's main
branch on 2026-09-10 (PR #7132); 1.13.0 is the last release that ships it. At
that version its constructor has no `replay_buffer` argument, and the trainer
passes 1-D per-sample tensors to replay helpers that expect
`(num_groups, num_generations)`, so `group_std_rewards.max(dim=0).values > 0`
collapses to a scalar: only group 0 of each batch is ever buffered, with a
single float as its "advantages", and replacement happens only when the whole
batch has zero variance (issue #6804). That defect lives in the trainer, not
in its buffer object, so a buffer swap cannot fix it. Under the default
configuration the trainer also computes no behavior logprobs, which `Rollout`
requires. The adapter therefore attaches to the stable `GRPOTrainer` instead.

### 9.2 Attachment point

`GRPOTrainer._prepare_inputs` calls `_generate_and_score_completions` once
per generation step and feeds the dict it returns, shuffled and split into
micro-batches, to `_compute_loss`. `ReservoirReplayMixin` overrides that one
method: it calls the original and passes the result to
`ReservoirReplay.mix(output, trainer)` in train mode. No TRL method body is
copied; the adapter reads and rewrites only the keys the loss consumes:

| key | shape, meaning |
|---|---|
| `prompt_ids`, `prompt_mask` | `(B, Lp)` long, left-padded (suffix mask) |
| `completion_ids`, `completion_mask` | `(B, Lc)` long, right-padded prefix mask; all zeros for a masked truncated completion |
| `advantages` | `(B,)` float32, group-centred; a zero-variance group is exactly `0.0` in every row |
| `old_per_token_logps` | `(B, Lc)` float, optional |
| `ref_per_token_logps` | `(B, Lc)` float, present iff `beta != 0` |
| `num_items_in_batch` | 0-d tensor, loss normaliser for the dapo/cispo/vespo losses |

Rows `[g·G, (g+1)·G)` are the `G = num_generations` completions of prompt
`g`; the shuffle happens after the hook. `ReservoirReplayCallback` raises at
the end of the first optimizer step if the hook never ran, so a TRL rename of
the overridden method cannot silently disable replay.

### 9.3 Field mapping

| TRL (row `r` of group `g`) | Reservoir |
|---|---|
| `completion_ids[r][:n]`, `n` = prefix length of `completion_mask[r]` | `Rollout.tokens` |
| `old_per_token_logps[r][:n]` (TRL's, or computed by the adapter) | `Rollout.logprobs` |
| `advantages[r]` | `Rollout.reward` |
| unpadded `prompt_ids[r]`, `r`, `global_step` | `Rollout.metadata["prompt_ids"]`, `["row"]`, `["global_step"]` |
| `ref_per_token_logps[r][:n]` when present | `Rollout.metadata["ref_logprobs"]` |
| rows of group `g` with a non-empty mask | one `RolloutGroup`; `prompt_id` = BLAKE2b of the prompt ids |
| `trainer.state.global_step` | `model_version` of `add_group`, `current_version` of `sample` |

`Rollout.reward` holds TRL's advantage rather than the raw reward because
the raw reward is not part of the loss contract and the advantage is the
value the loss consumes on replay. The default priority is therefore
`StoredAdvantagePriority` (`|reward| + ε`), the group-relative magnitude TRL
has already centred; `AdvantagePriority` would re-centre on the rows that
survived truncation.

### 9.4 One generation step

1. Refuse what the adapter does not handle: `global_step` below the buffer's
   version, more than one process, `loss_type == "vespo"` with `beta > 0`,
   or any of the tool, vLLM or vision keys.
2. Behavior logprobs: use `old_per_token_logps` if TRL computed it (it does
   so only when generation and optimizer steps are misaligned, or under vLLM
   importance correction); otherwise one no-grad forward through
   `trainer._get_per_token_logps_and_entropies`. Values in `(0, 10⁻³]` are
   clamped to 0 and counted; larger or non-finite values raise naming the
   row.
3. Store: `add_group(prompt_id, global_step, rollouts)` for every live group.
   Dead groups (all advantages `0.0`) and rows with an empty mask are
   skipped and counted.
4. Advance the buffer to `global_step` (stale eviction, rebase if due).
5. Replay: with `d` dead rows and a non-empty buffer,
   `sample(d, current_version=global_step)`; each sampled rollout is written
   row-wise into a dead row (prompt right-aligned, completion left-aligned,
   masks rebuilt, logprobs and ref logprobs filled), the batch is padded
   first if a replayed sequence is longer than the current width, and
   `num_items_in_batch` is recomputed from the final mask. A row in one dead
   slot may come from any prompt; the loss does not need group contiguity.
6. A step with nothing to replay returns the dict TRL produced, unchanged
   (the extra forward of step 2 may still have run; nonclaims §14).

Every function in `_trl_rows.py` returns new tensors; TRL's own tensors are
reused across micro-steps and are never modified in place.

### 9.5 Importance weights

TRL's loss has no per-sample weight slot. For every supported loss type the
per-token loss is positively homogeneous of degree 1 in the advantage:
`−min(ρA, clip(ρ)A) = −w·min(ρA', clip(ρ)A')` with `A = w·A'` and `w > 0`,
and likewise for the cispo, sapo, bnpo, dr_grpo, dapo and luspo variants. So
multiplying a replayed row's advantage by its IS weight `w ∈ (0, 1]` is
exactly the IS-weighted policy-gradient term; the KL term stays unweighted,
as in classic PER. `vespo` feeds the advantage into a non-linear gamma
weight and is refused unless `beta = 0`. `float(w)` and the multiply are
the declared float boundary, as in §6.

### 9.6 What the log records

Per generation step the log gains one `insert` per stored row (with
`entry_version = global_step`), one `advance_version` when the step moved
and the `evict`/`rebase` records that follow from it, and one `sample`
record per step that replayed rows. Generation, reward functions and TRL's
shuffle are outside the log. Draws are keyed BLAKE2b values and never touch
the torch RNG, so the adapter does not perturb the rest of the run the way
`torch.multinomial` in TRL's buffer did.

### 9.7 TRL quirks the design works around

- The removed trainer's `update_with_replay_buffer` also ran on eval
  batches, crashed on `for item in None` when its buffer was empty, trimmed
  left-padded prompts from the wrong side, and raised `TypeError` on equal
  heap scores. The adapter is train-only, treats an empty buffer as "leave
  dead groups alone", slices by mask side, and has no heap.
- `old_per_token_logps` is absent under the default configuration; a
  missing value becomes `per_token_logps.detach()` in the loss, i.e. ratio
  1 and no clipping for replayed rows. The adapter supplies it whenever rows
  are replayed.
- `mask_truncated_completions=True` zeroes a row's whole mask; such rows
  cannot form a `Rollout` and are skipped.

## 10. Content Commitment, Transcript and Diff

Version 0.5.0 binds every buffer slot to the training example it holds, so
a verified log can answer *which examples* were sampled rather than only
*which slots*.

### 10.1 Content digest

```
preimage = canonical JSON of {"prompt_id": <str>, "reward": <reward.hex()>, "tokens": [<int>, ...]}
           (sorted keys, separators (",", ":"), ensure_ascii=True, UTF-8)
content_digest = BLAKE2b-256(preimage, person=b"rollout-content\x00").hex()
```

`float.hex()` makes the reward exact and gives one spelling per value in
every Python, so `-0.0` and `0.0` have different digests. Behavior logprobs
and metadata are excluded: they describe the generating model and the
caller, not the example. The library computes the digest in
`reservoir.rollout.content_digest_of`; the checker recomputes it in
`checker/content.py` from this definition with the standard library, and
the two are cross-checked by known-answer vectors (one with a non-ASCII
prompt id) and a property test.

### 10.2 Schema additions

`insert` records gain two optional fields (absent, not null, when unused):

| field | value | rule |
|---|---|---|
| `content_digest` | 64 lowercase hex characters | `insert` only; a log has it on every insert or on none |
| `source` | printable string, 1–256 characters, not whitespace-only | requires `content_digest`; `RolloutGroup.source`, set by `add_group(..., source=)` |

`update` and `evict` records are unchanged: they refer to a slot whose
example the replay already knows. Records without the fields serialise
byte-for-byte as before, so legacy logs verify unchanged.

### 10.3 Manifest

The manifest (`reservoir.rollout_manifest`) is a separate JSON-lines file,
one line per insert, keyed by the same `(op_counter, index)`:

```json
{"op_counter": 12, "index": 50, "content_digest": "<hex>", "prompt_id": "a3f…",
 "source": "gsm8k", "tokens": [1, 2, 3], "reward_hex": "0x1.0000000000000p+0", "entry_version": 7}
```

It is the opening of the digests, not a second chain: the log already
commits to every line. `RolloutBuffer(manifest=path)` requires `attest`.
The durable buffer keeps the manifest lines in its committed state and
rewrites the file from that state on reopen, exactly as it does for the
attestation file, and refuses to reopen a directory with a different
manifest setting than it was saved with. `RolloutAttester.prepare_inserts`
computes digests and manifest lines before the first tree write of
`add_group`, so nothing on the insert path can fail after the buffer has
started mutating; `restore` validates every manifest line (types, digest
recomputation, canonical `reward_hex`) and cross-checks
`(op_counter, index, content_digest, source, entry_version)` against the
log before replacing anything.

### 10.4 What the checker verifies

`checker/content.py` adds to `verify_chain`:

- field well-formedness and the all-or-none rule for `content_digest`;
- slot tracking through inserts and evicts, so every draw resolves to the
  example its slot held (a draw of a slot with no committed example is an
  error);
- with `--manifest`: every line's digest recomputes from its own prompt,
  tokens and reward; the lines, in order, are exactly the log's
  content-bearing inserts with the same digest, source and entry version;
  a log with no inserts accepts only an empty manifest; a log whose inserts
  carry no digests accepts none.

`verify_chain` returns a `VerifiedLog` whose `content` state (insert
history, resolved samples) is what the two tools below build on.

### 10.5 Transcript

`python -m checker.transcript` derives, from the log alone:

- **exposure** per content digest: `times_sampled`, the exact sum of
  importance weights, first and last `op_counter`, every copy with the
  record that evicted it;
- **mixture** per source: inserted examples, sampled rows, share of sampled
  rows; and per *window*, the stretch of records between two
  `advance_version` records (keyed by position, so a repeated version
  cannot double-count);
- **quota** verdicts (`--quota source=N`, exit 2 on violation) and **find**
  (`--find <digest>`, every sample record and batch position).

`--quota` and `--find` refuse a log without content digests; the report key
`(none)` denotes untagged examples and a real source spelled that way is
refused.

### 10.6 Diff

`python -m checker.diff` verifies two logs and classifies their first
differing record: `identical`, `config` (`decay_config` differs), `data`
(an insert or update differs, or the operation sequences diverge: the
stored examples, scores or group sizes differed upstream), `schedule`
(`advance_version`), `sampler` (two same-sized `sample` records on an
identical prefix: different seeds or buffer ids, which are not in the log),
`internal` (an evict or rebase differs on identical state, or a rebase
appears where the other log has a different record; deterministic, must
never happen), `truncated` (one is a prefix of the other). When the
operations differ, `decay_config`, `advance_version` and `rebase` take
precedence in that order; two `sample` records of different sizes are
`data`, and two with identical draws and slots but different weights are
`config` (a different `beta`).

### 10.7 Limit

The log commits; the manifest opens. A chain-consistent change to an
insert's `content_digest` or `source` is invisible without the manifest.
The mutation campaign measures this (`content_limit` in
`results/mutation_campaign_report.json`): 3 such forgeries survive the
log-only check and all 3 are rejected with the manifest. The campaign fails
if that measurement ever changes.
