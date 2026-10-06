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

### Command log (rollout buffer, version 0.5.0)

`DurableRolloutBuffer` no longer writes a full state per operation. Every
`RolloutBuffer` operation is a deterministic function of state and inputs
(exact priorities, keyed draws, derived attestation and manifest records),
so the wrapper applies the operation in memory, appends `{"epoch", "seq",
"op", "args", "digest"}` to `wal.jsonl` with a full fsync (and an fsync of
the directory when the file is created), and returns. Once `compact_every`
commands have accumulated, the next operation first writes a snapshot
(`state.json`, through the protocol above, carrying the `seq` and `epoch`
it includes) and resets the log; a compaction failure therefore surfaces
before that operation is applied and never after a result was returned.

Recovery loads the snapshot, reads the log, cuts a torn tail (an
incomplete or digest-mismatching *last* line; the same damage before the
end, or a sequence gap, is corruption and refuses to open), skips lines the
snapshot already includes and lines of another epoch, replays the rest onto
a throwaway buffer to validate it, and only then rebuilds the live buffer
and its attestation and manifest files. A command that raised was never
logged, and a command whose append failed is cut off the log, so a failed
operation is undone by rebuilding from disk. Witness and telemetry
commands name the sample they were issued against (`sample_op`); the
buffer rebuilds that batch only while every sampled slot still holds the
rollout it held at sample time (insert stamps and versions are recorded
with the sample), so a replay can never bind a witness to a later
occupant.

A snapshot (compaction, the first write of a fresh directory, a restore)
has no operation to apply, so it goes through `durably_snapshot`: the
intent carries no pre-state (the committed `state.json` is the
pre-state), the state is serialised once into the fsynced segment, and
after the commit rename the segment becomes `state.json` by rename
rather than a second write. The cut points are those of `durably_apply`
plus two of its own around that rename (`after_snapshot_rename`,
`after_snapshot_dir_fsync`), and the crash campaign arms all of them;
recovery keeps the old `state.json` for an intent without a readable
segment and installs the segment for an intent with one.
`checkpoint(tag)` compacts and copies the snapshot under
`checkpoints/<tag>/` with fsyncs of the file and its directory, and
writes an optional `binding.json` beside it the same way: the TRL and
verl adapters record there the digest of the trainer's own checkpoint
directory (`checkpoint-N`, `global_step_N`: every regular file's relative
path, size and bytes under BLAKE2b-256), and `resume_from_checkpoint`
refuses to rewind when the directory the model restarts from digests
differently or cannot be found (`check_model_binding`; every rank is
waited for before the owner digests, and `ReservoirReplay.model_checkpoint`
names a relocated directory); a buffer checkpoint taken without a trainer
directory records no digest and is not checked, and a re-taken tag drops
its old binding before the new state lands, so a crash leaves it unbound
rather than wrongly bound.
`restore_checkpoint(tag)` writes the checkpoint state as a new snapshot
with a fresh epoch, then resets the log; lines of the abandoned timeline
left by a crash between the two are ignored on replay whatever their
`seq`. Cut points: `mid_wal_write`, `after_wal_write`, `after_wal_fsync`,
the snapshot protocol's seven, `after_snapshot_before_wal_reset` and
`after_restore_before_wal_reset`.

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

A decayed buffer adds `base_priority_int`, `entry_version` and `base_epoch`
(§7) and, on an evict, `reason`: `stale`, `capacity`, `explicit`, `drift`
or `quarantine`. A `quarantine` evict also carries `predicate` (the text
of the predicate that selected the entry, at most 1024 printable
characters) and `note` (the operator's reason, at most 256); see §10.10.

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

`DurableRolloutBuffer` (`durable_rollout.py`) appends each operation's inputs
to a command log and writes a full-state snapshot (`state_dict`/
`load_state_dict`, including the attestation log and manifest) through the §3
protocol every `compact_every` commands; see "Command log" in §3. The crash
campaign kills the process at every log and snapshot cut point during an
`add_group` that evicts and rebases and during a `sample`.

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
   version, `loss_type == "vespo"` with `beta > 0`, or any of the tool, vLLM
   or vision keys (more than one process is routed through §9.8).
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

### 9.8 More than one process

Under `accelerate` with `N > 1` processes, `_generate_and_score_completions`
runs on every rank over that rank's prompts; rewards and advantages are
gathered, computed on the whole generation batch, and sliced back with
`advantages[process_slice]`, where the slice of rank `k` is rows
`[k·n, (k+1)·n)` of the global batch. So each rank sees a contiguous,
rank-ordered slice; a prompt group of `G` rows straddles two ranks whenever
`n` is not a multiple of `G`; and `num_items_in_batch` is already the
global mask sum (`accelerator.gather(loss_mask.sum()).sum()`), which the
loss divides by `num_processes`.

`reservoir.integrations._trl_distributed` keeps §9.4 unchanged and adds a
transport around it:

1. **Ownership.** Rank 0 owns the buffer and is the only log writer. A
   `ReservoirReplay` constructed under a launcher that sets `RANK` or
   `LOCAL_RANK` to a non-zero value builds no buffer (its `is_owner` is
   False and `.buffer` raises); `ReservoirGRPOTrainer` calls
   `attach(accelerator)` at construction, which fixes the rank from the
   process group. A log, manifest or directory is not opened until the rank
   is known; in-memory state built before that is closed and dropped on a
   non-owner. `attach` is itself a collective under more than one process:
   if any rank but 0 has already opened file-backed state (a launcher that
   sets no `RANK`, since `LOCAL_RANK=0` alone decides nothing, and a `.buffer`
   touched before the trainer was built), every rank raises together and
   nothing is attached. `bind_checkpoint` and
   `resume_from_checkpoint` are no-ops off the owner.
2. **Behavior logprobs** are computed per rank for its own rows (one
   no-grad forward, balanced across devices), then every rank's slice, with
   its `global_step`, is all-gathered as CPU tensors (`gather_object`; every
   rank receives every slice and holds them until the step returns).
   A rank whose local work fails contributes its error instead of a batch.
3. **Rank 0 runs §9.4 on the global batch**: shards are padded to common
   widths (`pad_batch`, prompts on the left, completions on the right) and
   concatenated in rank order, so groups are whole and the insert order in
   the log is the global row order. `old_per_token_logps` is present, so no
   second forward runs. The witness, telemetry, gate and rescoring see the
   global batch.
4. **The result is broadcast** (`broadcast_object`) and each rank takes
   back its rows, padded to the global width when replay widened the batch,
   together with the recomputed global `num_items_in_batch`. When rank 0
   had nothing to replay, every rank returns the dict TRL produced, as in
   one process; a step whose every draw was declined returns the batch with
   behavior logprobs attached in both paths, and the witness digest is over
   that batch.
5. **Every rank performs exactly one gather and one broadcast per hook
   call** (plus the one gather of `attach` on the first call). A failure before the gather on any rank (a refused key, a bad
   logprob, an out-of-memory forward) travels in that rank's shard; rank 0
   turns it, or its own failure (a version regression, ranks disagreeing on
   `global_step`, a gather that returned fewer shards than the world size),
   into the broadcast value; every rank raises after the broadcast, the
   failing rank with its own exception and the others naming it. No rank
   is left waiting in a collective.

`Communicator` is the four-member interface this needs (`num_processes`,
`process_index`, `gather_object`, `broadcast_object`). A real `Accelerator`
is wrapped over `accelerate.utils.gather_object` / `broadcast_object_list`;
the tests drive the same code with a fake accelerator of two and four ranks
on threads that rendezvous on a barrier, with a parity check against the
single-process adapter on the concatenated batch. On a real process group
(two A10Gs under `accelerate launch`, `benchmarks/modal/trl_replay_distributed.py`)
the gathered shards arrive as CPU tensors and rank 0 moves the assembled
global batch to its own device before the hook, because the telemetry
forward feeds it to the model; the fake-world tests had not caught that
(nonclaims §21).

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

### 10.7 Declared draw configuration

`decay_config` optionally records `seed`, `buffer_id` (decimal strings)
and `alpha`, `beta` (`float.hex()` strings), all four or none. With them
the checker re-derives every `draw_int` from the keyed BLAKE2b definition
of §5 (`checker/draw.py`), keyed on the running count of rollouts drawn,
and every importance weight from the §6 formula evaluated on the replayed
tree. Without them a draw moved inside its leaf's range, or a reweighted
sample, passes the range and reduced-form checks; the mutation campaign
measures this (`draw_limit`). The weight recomputation evaluates `pow` in
float64 exactly as the library does, so a verifier whose libm rounds `pow`
differently from the producer's would report a mismatch; nonclaims §7
declares that boundary. Two runs that differ only in their seed now differ
at record 0 (`config`), so `checker.diff`'s `sampler` class is reachable
only for logs without the configuration or by a defect.

### 10.8 Batch witness

After an adapter has placed a sampled batch into training-batch rows it
writes one `batch` record (`RolloutBuffer.witness_batch`):

```json
{"op": "batch", "step": "<trainer step>", "sample_op_counter": <op_counter of the sample record>,
 "batch_rows": <rows in the training batch>,
 "replaced": [{"row": <int>, "draw": <position in the sample record>, "content_digest": "<hex>"}, ...],
 "tensor_digest": "<BLAKE2b-256 of canonical JSON of the final prompt ids, completion ids and advantages>"}
```

An optional `declined` list names draw positions the adapter refused to
place (see the drift gate below); placed and declined draws together must
cover the sample exactly.

The witness is the adapter's declaration. Before writing it the adapter
re-reads every replaced row and refuses to continue unless it holds
exactly the sampled rollout's prompt, tokens, behavior logprobs and
weighted advantage, so a padding or masking mistake in the row writer
fails loudly at the first replayed step. The checker then proves the
declaration consistent with the sample record: it names the latest sample,
every draw is placed or declined exactly once, every row is inside the
batch and replaced at most once, each row's digest equals the digest the
draw resolved to, sample `op_counter` values increase, and no sample is
witnessed twice. `reservoir-transcript --explain STEP ROW` answers why a
row holds what it holds. The tensor digest (prompt ids, masks, completion
ids, advantages and behavior logprobs, the per-token logprobs zeroed outside
the completion mask, in a fixed-width encoding) is a
commitment the log cannot open; a holder of the batch can. The `format`
field of `decay_config` is `"2"` for logs that may carry witnesses and
telemetry and `"3"` for logs that may also carry quarantine evictions
(§10.10); the checker reads formats 1 to 3 and refuses each record kind in
a log whose format predates it.

### 10.9 Telemetry and the drift gate

One `telemetry` record per adapter step carries the integer counters
(`batch_rows`, `replaced_rows`, `declined_rows`, `dead_groups`,
`near_dead_groups`) and, for a replayed step, the exact effective sample
size `(Σw)²/Σw²` of the sampled batch's importance weights and the maximum
and sum of its rows' ages in model versions; the checker recomputes the
latter two from the sample record and the live slots and cross-checks
`replaced + declined` against the number of draws. Float measurements the
log cannot recompute (the per-sequence log-ratio between stored behavior
logprobs and the current policy, mean and max of the absolute value) are
written in `float.hex()` form and listed under `reported`. The same
numbers go to TRL's metrics under `reservoir/`.

The log-ratio costs one extra no-grad forward over the replayed rows per
step whenever telemetry is on, in whatever mode the trainer's model is in.
The checker also cross-checks the telemetry counters against the batch
witness of the same sample (step, batch size, placed and declined counts)
and requires the sampled slots to still hold the drawn examples when the
telemetry record is written.

The drift gate is off by default. With `max_log_ratio` set, a sampled row
whose absolute sequence log-ratio exceeds it, or is not finite, is
declined: the dead row it would have filled stays dead, the witness lists
the draw under `declined`, the entry is evicted with reason `drift` after
the telemetry record (a slot drawn more than once is evicted only if every
draw of it was declined), and more than `max_declines_per_step` declines in
one step raise. A decline is never silent.

### 10.10 Quarantine and blast radius

Incident response has two halves: remove what a bad reward function or a
leaked prompt set put into the buffer, and find out what it reached.

`RolloutBuffer.quarantine(predicate, reason, predicate_text=None)` runs
`predicate(rollout, group)` over a copy of every live entry before anything
is mutated (a predicate that raises, or returns anything but a `bool`,
leaves the buffer unchanged, and one that writes into its arguments
changes nothing the buffer stores, so the live state and the replayed log
agree), then evicts each match with reason `quarantine`. A buffer that
continues a format-1 or format-2 log refuses, before mutating, because
that log's checker would reject the record.
Each such `evict` record carries `predicate`, the caller's text or the
predicate's source line collapsed to one line, and `note`, the `reason`
argument. Nothing is written when nothing matches. The durable buffer
evaluates the predicate against the committed state and logs a
`quarantine` command holding the selected positions and both texts, never
the callable, so recovery replays the same evictions; the crash campaign
cuts inside it (`rollout_quarantine`).

The checker requires both texts on a `quarantine` evict, refuses them on
any other record, refuses the reason in a log of format 1 or 2, and
otherwise treats the record as the evict it is (a live slot, leaf set to
zero, no pending stale evictions). It resolves the slot to the example it
held, so `reservoir-transcript --blast-radius <digest>` can answer the
question an incident asks: every insert of the example, every
training-batch row that held it (from the batch witnesses) with its step,
the sorted steps touched, how often it was drawn and how many of those
draws have no witness (rows and steps are then a lower bound, and the
entry says so), and the quarantine records that removed it.
With the manifest the radius covers every committed example of the same
prompt; without it only the exact digest is followed, and the entry says
so. A log without witnesses lists no rows and says so. The earliest step
in the radius is the checkpoint a run has to be rolled back to.

### 10.11 Reward provenance

GRPO rewards are usually a weighted sum of several functions (a verifier,
a format check, a judge). The example's `reward` in the content digest is
the advantage the loss consumed; which function produced it is lost. The
TRL row conversion accepts `reward_names` and a `(B, F)` tensor of
per-function values and stores `{name: value}` under the reserved
metadata key `rewards` (`rollout_manifest.REWARDS_KEY`); the buffer
validates it before any insert and writes it as the optional `rewards`
field of the manifest line. The field is numeric only (printable names to
finite JSON numbers; a `NaN`, TRL's "this function abstained", is left
out; text is refused) and outside the content digest, so the log does not
commit to it: it is reported, like the telemetry log-ratios, and the
checker verifies only its shape. The transcript shows it next to each
example, which is what "verifier high, judge low" queries need after the
fact. Wiring the two arguments into `ReservoirReplay._ingest` is a
two-line change in the adapter that reads `trainer.reward_func_names` and
the per-function rewards TRL computes.

### 10.12 Limit

The log commits; the manifest opens. A chain-consistent change to an
insert's `content_digest` or `source` is invisible without the manifest.
The mutation campaign measures this (`content_limit` in
`results/mutation_campaign_report.json`): 3 such forgeries survive the
log-only check and all 3 are rejected with the manifest. The campaign fails
if that measurement ever changes.

The log commits to neither a quarantine record's texts nor the manifest's
`rewards`. A chain-consistent change to the predicate text or the note, a
changed reward value and a dropped `rewards` field all pass the checker;
the campaign measures these four (`provenance_limit`) and fails if the
measurement changes. They are the operator's and the adapter's statements,
recorded so an auditor can read them, checked for shape only.

## 11. Offline Replay

`reservoir_checker.replay` (console script `reservoir-replay-offline`)
is the first consumer of the witness and the manifest together. Its
input is a verified log and the manifest that opens it; its output is
JSON lines a third party can train on or audit without the trainer, the
model or the library: one header, then one line per batch witness.

Resolution goes witness → draw → slot → insert → manifest line. The
witness binds a row to a draw position of the latest sample record; the
sample record binds that draw to a slot and carries the exact
probability and importance weight; the slot resolves to the insert that
was live at the sample record (the latest insert into that slot before
it, found by bisection over the insert history by slot); and the insert
is the manifest line at the same position in the history, which
`check_manifest` has already proved to be its opening. A reused slot
therefore resolves to the example the draw saw, not to the slot's later
occupant. The checker has already proved that the witnessed digest equals
the draw's; the replay checks once more that the manifest line it
reached carries that digest, so a lookup bug cannot emit a wrong example
silently.

Each row carries the opened example (prompt id, tokens, reward as a
float and as the canonical `float.hex()` the digest was computed over,
source, entry version, and `rewards` when the manifest line has it), the
draw's weight and probability as floats and as exact
numerator/denominator strings, and the slot. The batch line carries the
step, the sample it was built from, the batch size, the tensor digest,
the declined draws, the rows the witness does not bind (`fresh_rows`),
and `generated`: every example inserted before the witness at an entry
version equal to the step, unordered, which is what the adapter stored
from that step's own generation when it stamps the entry version with
the trainer step, as the TRL adapter does; the log does not verify that
mapping. The header names the format, the record and example counts,
the head digest, the witnessed steps and the telemetry steps that have
no witness. Output lines are canonical JSON (sorted keys, no spaces), so
two replays of one run are byte-identical and `diff` is meaningful.

What it refuses: a log below format 2 (batch witnesses arrived with
format 2, so an older log has nothing to reconstruct), a log without a
manifest, and anything `verify_chain` rejects. The mutation campaign
category `replay_manifest` tampers the manifest with the log untouched:
16 tamperings (a changed token, reward, prompt, source, version, slot,
operation counter, order or count, a recomputed digest, a non-canonical
reward spelling) must break the replay, because the manifest no longer
opens the log's commitments, and are counted as rejected. Beside the
category, as a measurement and not as rejections, 3 tamperings of the
per-reward-function values (changed or dropped on a replayed row,
changed on a generated example) must run, differ from the untampered
replay, and differ only in `rewards` fields: the log does not commit to
those values (§10.12), so a single manifest cannot be told from a
tampered one, and the replay reports what it states. The campaign fails
if a tampering lands on the other side of that line. The limits are in
`docs/nonclaims.md` §23.

## 12. verl Integration

`reservoir.integrations.verl` (`ReservoirReplay`, `ReservoirRayPPOTrainer`)
connects `RolloutBuffer` to verl's `DataProto`-based `RayPPOTrainer` under
GRPO. Tested against verl 0.9.1; `_verl_compat.require_verl` refuses a verl
whose `RayPPOTrainer`, `DataProto` or `core_algos` lacks a member the adapter
relies on and warns on an untested version.

### 12.1 Which trainer

verl 0.9.1 ships two training loops. The default (`trainer.use_v1=true`) is
`verl.trainer.ppo.v1.PPOTrainer`, which keeps trajectories in TransferQueue,
a key-value store, and passes `KVBatchMeta` handles between stages; the
older `RayPPOTrainer` (`trainer.use_v1=false`, `main_ppo_v0.py`) holds the
whole training batch as a `DataProto` on the Ray driver and is marked
deprecated. The adapter targets the `DataProto` trainer: its batch is a
padded tensor dict the row writer of §9 maps onto directly, the same shape
every verl release from 0.4 to 0.9 used, and the driver already owns the
whole batch, so the rank-ownership machinery of §9.8 is unnecessary. A V1
adapter would replace rows inside TransferQueue (clear the dead trajectory
keys, put stored rows under new keys with verl's tags, return a new
`KVBatchMeta`); verl's own DAPO dynamic sampling (`algorithm.filter_groups`)
does the eviction half of that in `v1/replay_buffer.py` and regenerates
from fresh prompts rather than from a store. That is the follow-up, not
this adapter.

### 12.2 Attachment point

`RayPPOTrainer.fit` generates through the agent loop, computes
`old_log_probs` with `_compute_old_log_prob(batch)`, computes advantages on
the driver (`compute_advantage`, which for GRPO calls
`core_algos.compute_grpo_outcome_advantage`) and then calls
`self._update_actor(batch)`, which converts the `DataProto` to a no-padding
TensorDict and ships it to the actor workers. `ReservoirReplayMixin`
overrides `_update_actor`: it hands the batch to `ReservoirReplay.mix` and
passes the result to the original. It also wraps `_compute_old_log_prob`
(the hook-ran check runs there, once per step before the update),
`_save_checkpoint` and `_load_checkpoint` (checkpoint binding) and `fit`
(final hook-ran check). No verl method body is copied. The dead-group
criterion is exact: `compute_grpo_outcome_advantage` computes
`(score - mean) / (std + eps)` per uid and an all-equal group has a zero
numerator, so every row of such a group is `0.0` in every response token.

### 12.3 Field mapping

| verl (row `r` of uid `u`) | Reservoir |
|---|---|
| `responses[r][:n]`, `n` = prefix length of `response_mask[r]` | `Rollout.tokens` |
| `old_log_probs[r][:n]` (verl's, computed on the current policy before the update) | `Rollout.logprobs` |
| `advantages[r][0]` (constant over the response tokens under GRPO) | `Rollout.reward` |
| `prompts[r]` under the prompt columns of `attention_mask`, `r`, `global_steps`, `u`, `token_level_scores[r].sum()` | `metadata["prompt_ids"]`, `["row"]`, `["global_step"]`, `["uid"]`, `["score"]` |
| `ref_log_prob[r][:n]` when present | `metadata["ref_logprobs"]` |
| rows sharing `non_tensor_batch["uid"]` with a non-empty mask | one `RolloutGroup`; `prompt_id` = BLAKE2b of the prompt ids |
| `trainer.global_steps` (1 during the first step) | `model_version` of `add_group`, `current_version` of `sample` |

Rows of one uid need not be contiguous: `balance_batch` reorders the batch
by length before the update, so groups are found by uid, not by position.

A replayed row rewrites `prompts` (right-aligned), `responses`
(left-aligned), `input_ids` and `attention_mask` (the concatenations),
`response_mask`, `position_ids` (verl's rule: cumsum of the mask over the
prompt, then last prompt position plus one, two, ... over the response),
`old_log_probs`, `ref_log_prob` when present, `advantages` and `returns`
(the weighted advantage broadcast over the response tokens; under GRPO
`returns` is `advantages`), and `token_level_scores`, `token_level_rewards`
and `rm_scores` (the stored score at the last response token). `dummy_tensor`,
the dataset's `(B, 1)` placeholder, passes through. `rollout_log_probs`,
which the rollout attaches by default (`calculate_log_probs: True`), is
dropped from the mixed batch when `algorithm.rollout_correction` is off,
the only case in which the actor does not read it; with any of
`bypass_mode`, `rollout_is` or `rollout_rs` set the batch is refused. Any
other per-row tensor (`values`, `rollout_is_weights`, `routed_experts`,
`teacher_*`, `sum_pi_squared`) is refused by name. The mixed `DataProto`
carries `uid` as its only non-tensor column (a replayed row takes the uid
of the group it came from; the actor reads none of the others, and the
generated prompt's `data_source`, `reward_model` and `extra_info` would be
wrong for a replayed row) and the original `meta_info` with
`global_token_num` recomputed from the final `attention_mask`.

### 12.4 One training step

1. Refuse what the adapter does not handle: no `uid`, an unsupported
   per-row tensor or multimodal column, `algorithm.adv_estimator` other
   than `grpo`, rollout correction, a covariance loss mode (`clip_cov`,
   `kl_cov`) with `beta > 0`, `global_steps` below the buffer's version,
   KL-in-reward (`token_level_rewards != token_level_scores`), 3-D position
   ids, a response mask that is not a prefix mask (multi-turn tool output
   masked in the middle), advantages that vary over a row's tokens.
2. Store: `add_group(prompt_id, global_steps, rollouts)` for every live
   uid. Dead groups and rows with an empty response are skipped and
   counted.
3. Advance the buffer to `global_steps`.
4. Replay: with `d` dead rows and a non-empty buffer,
   `sample(d, current_version=global_steps)`; the rows are rewritten as in
   §12.3, the batch padded first if a replayed sequence is longer than the
   current width, every replaced row re-read against its rollout, the
   batch witness written with the tensor digest of §12.5, and telemetry
   measured and recorded (§10.9), the log-ratios from one
   `_compute_old_log_prob` call over the replayed rows. The telemetry goes
   into the actor output's `meta_info["metrics"]` under `reservoir/*`,
   which `fit` reduces and logs next to `actor/*`.
5. A step with nothing to replay hands the original `_update_actor` the
   `DataProto` `fit` built.

Importance weights fold into the advantage as in §9.5: verl's `vanilla`,
`gspo`, `gpg` and `geo_mean` policy losses are positively homogeneous in
the advantage; `clip_cov` and `kl_cov` select tokens by covariance with the
advantage and are refused unless `beta = 0`.

### 12.5 Tensor digest

BLAKE2b-256 (personalisation `verl-batch`) over `input_ids`,
`attention_mask`, `position_ids`, `response_mask`, `advantages`,
`old_log_probs` and `ref_log_prob` when present, as `name|shape|kind`
followed by the values in fixed-width little-endian encoding (int64 or
float64), with per-token tensors zeroed outside `response_mask` and
`position_ids` zeroed outside `attention_mask`, so the digest depends on
what the model and the loss consume and the padded shape only.

### 12.6 Checkpoints

verl saves `<default_local_dir>/global_step_<n>/` at `global_steps == n`
and on resume sets `global_steps = n` from the directory name before the
first new step (`n + 1`). `_save_checkpoint` snapshots a durable buffer as
`step-<n>` and prunes buffer checkpoints to the `global_step_*` directories
the trainer kept; `_load_checkpoint` at `n > 0` rewinds the buffer to
`step-<n>` and refuses if that snapshot is missing or if
`global_step_<n>` no longer digests as it did when the buffer checkpoint
was taken (§3 above). The hook-ran check
counts steps trained by this process (`global_steps - n`), so a resumed run
is not failed for the steps the checkpoint already contained.
