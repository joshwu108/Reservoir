# reservoir

**Certified Exact Prioritized Experience Replay with Crash Atomicity and Sampling Attestation**

A correctness-oriented research codebase for the experience/data side of off-policy reinforcement learning. The subject is Prioritized Experience Replay (PER).

---

## T2 NEGATIVE RESULT (read first)

**Thesis T2 is DEAD (falsified).** The preregistered search (docs/preregistration.md) found **zero decision-relevant divergences** between the exact and float sum-tree idioms in 135 workloads across all grid cells. The total-variation distance between exact and float-implied distributions was ≤ 8.67×10⁻¹⁹ in every cell — well below the preregistered kill threshold of 2⁻⁴⁰ ≈ 9.09×10⁻¹³.

This is a **negative result**, reported prominently as required. The kill thresholds were frozen before any data was collected (see `docs/preregistration.md`). See `results/divergence_campaign_report.json` for the full report. Note: the campaign was run with a reduced grid (5 workloads per cell, capacities up to 16384) due to pure-Python exact-tree throughput constraints; the full 10,000-workload preregistered grid remains to be run with an optimized exact implementation.

---

## Scope

`reservoir` implements four correctness claims about Prioritized Experience Replay:

| Thesis | Status | Evidence |
|--------|--------|---------|
| **T1 (exactness):** Proportional PER can be implemented with zero floats on any decision path | **Alive** | `src/reservoir/`, `tests/` |
| **T2 (float divergence):** Float idioms produce decision-relevant divergences from the exact reference | **DEAD** (falsified) | `results/divergence_campaign_report.json` |
| **T3 (crash atomicity):** Durable buffer is failure-atomic under SIGKILL | **Alive** | 70/70 crash tests, 0 torn states |
| **T4 (attestation):** Independent checker verifies sampled batches; ≥60 forgeries rejected at 100% | **Alive** | 63/63 forgeries rejected |

---

## Architecture

```
reservoir/
├── src/reservoir/
│   ├── rational.py      # Exact float→integer conversion (float64-once boundary)
│   ├── sumtree.py       # Exact integer sum-tree (insert, update, prefix-locate)
│   ├── draw.py          # Keyed BLAKE2b uniform-integer draw (no random/numpy/torch RNG)
│   ├── buffer.py        # In-memory exact PER: insert/update/evict/sample, exact IS weights
│   ├── durable.py       # WAL-based durable buffer: F_FULLFSYNC, atomic rename, recovery
│   └── attest.py        # Hash-chained MutationRecord + SampleAttestation
├── checker/
│   └── verify.py        # Independent verifier (imports NOTHING from src/reservoir)
├── campaigns/
│   ├── float_baselines.py  # Faithful float reimplementations (OpenAI, labml idioms)
│   ├── divergence.py       # T2 float-divergence search
│   ├── crash.py            # T3 kill-9 crash campaign
│   └── mutation.py         # T4 forgery campaign
├── spec/
│   └── ReplayLifecycle.tla # TLA+ safety model (44,611 states, no errors found)
├── demo/
│   └── tiny_dqn.py      # 5-state chain MDP DQN: reward=25, bitwise-identical runs
├── docs/
│   ├── design.md         # α-exponentiation decision, durability protocol, attestation schema
│   ├── preregistration.md # Frozen T2 search protocol + kill thresholds (BEFORE campaigns)
│   └── nonclaims.md      # Explicit non-claims
└── results/
    ├── mutation_campaign_report.json   # T4: 63/63 rejected
    ├── crash_campaign_report.json      # T3: 70/70 pass, 0 torn states
    └── divergence_campaign_report.json # T2: DEAD (kill rule fired)
```

---

## Key Design Decisions

### Float64-Once Boundary (α-exponentiation)

Priority `p_i = |δ_i| + ε` is raised to power α using float64 (the only allowed float operation on decision paths). The binary64 result is immediately integerized:

```python
priority_int = int(Fraction(p_i ** alpha) * 2**52)
```

All downstream arithmetic (tree sums, sampling, IS weights) uses Python integers and `fractions.Fraction`. See `docs/design.md` §2 for the full design rationale and three options considered.

### Deterministic Keyed Draw

Sampling uses BLAKE2b-256 rejection sampling keyed by `(seed, buffer_id, op_counter)`. No `random`, `numpy.random`, or `torch` RNG on decision paths. This is deterministic replay identity, **not** a cryptographic security guarantee.

### Crash Atomicity (macOS/APFS)

The durable buffer uses write-ahead intent records with `fcntl(fd, F_FULLFSYNC)` on Darwin (falling back to `os.fsync()` on Linux) and atomic `rename()` as the commit point, followed by parent-directory fsync. See `docs/nonclaims.md` §6 for the scope of the durability claim.

### Independent Checker

`checker/verify.py` imports nothing from `src/reservoir` (CI-enforced via AST import check). It re-implements the sum-tree from scratch and verifies every attestation record from first principles.

---

## Non-Claims (Summary)

Full list: `docs/nonclaims.md`

- **Not a performance result.** Pure Python is orders of magnitude slower than float implementations.
- **Not a large-scale RL training quality result.** Sampling correctness ≠ training quality.
- **Not a distributed buffer.**
- **Not a security boundary.** The BLAKE2b draw is replay identity, not tamper-proof.
- **α-exponentiation uses the float64-once boundary.** The distribution is proportional to `binary64(p_i^α)`, not the true real `p_i^α`.
- **Crash durability evidence is on macOS/APFS with F_FULLFSYNC only.**
- **TLA+ model covers a small finite scope** (capacity 2, 2-value priorities, ≤3 ops). See `docs/nonclaims.md` §8.

---

## Running

### Setup

```bash
# Install uv
brew install uv

# Install dependencies
uv sync
```

### Run All Tests

```bash
make check
# or
uv run pytest tests/ -v
```

### Run Campaigns

```bash
# T4: Mutation forgery campaign (63/63 rejected)
uv run python -m campaigns.mutation

# T3: Crash atomicity campaign (70/70 pass)
uv run python -m campaigns.crash

# T2: Float-divergence campaign (T2 DEAD)
uv run python -m campaigns.divergence
```

### TLA+ Model Check

```bash
bash spec/check.sh
# No errors found, 44,611 states explored
```

### End-to-End Demo

```bash
uv run python -m demo.tiny_dqn
# Expected output:
#   Run A: Total reward: 25.0
#   Run B: Total reward: 25.0
#   Q(right) > Q(left) for all states 0-3 ✓
#   Attestation VERIFIED ✓
#   Bitwise-identical: YES ✓
```

---

## Campaign Results Summary

### T3 Crash Atomicity (70/70 pass)

| Operations | Cut Points | Seeds | Total | Torn States |
|-----------|-----------|-------|-------|-------------|
| insert, update | 7 cut points | 5 | 70 | **0** |

All 70 crash-recovery tests recovered to exactly the pre-operation or post-operation state.

### T4 Mutation Campaign (63/63 rejected)

| Category | Mutants | Rejected |
|----------|---------|---------|
| Digest bit-flips | 16 | 16 |
| Off-by-one draw integers | 10 | 10 |
| Swapped sampled indices | 4 | 4 |
| Probability not in reduced form | 12 | 12 |
| Deleted mutation records | 9 | 9 |
| Reordered records | 5 | 5 |
| Stale suffix replay | 7 | 7 |
| **Total** | **63** | **63** |

### T2 Float Divergence (DEAD)

| Grid Cells | Workloads | Divergences | Max TV Distance |
|-----------|-----------|-------------|----------------|
| 27 | 135 | **0** | 8.67×10⁻¹⁹ |

Kill threshold: 2⁻⁴⁰ ≈ 9.09×10⁻¹³. **Kill rule fired.**

---

## License

Apache-2.0
