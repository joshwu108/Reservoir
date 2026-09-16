# reservoir

**Prioritized Experience Replay you can actually verify — exact by default, fast when it matters.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-270%20passing-brightgreen.svg)](tests/)
[![TLA+ verified](https://img.shields.io/badge/TLA%2B-44%2C611%20states%2C%20no%20errors-success.svg)](spec/ReplayLifecycle.tla)

Most Prioritized Experience Replay (PER) implementations trust that float sum-trees, `np.random.choice`, and priority updates all stay consistent under concurrent inserts, crashes, and floating-point drift — and never check. `reservoir` doesn't trust it: it *proves* it, with an integer-exact reference implementation, an independent verifier, a crash-injection test harness, and a TLA+ model — and then ships a fast numpy/torch/C-backed buffer that's cross-checked against that reference so you don't have to give up performance to get it.

---

## Table of Contents

- [Why reservoir?](#why-reservoir)
- [Install](#install)
- [Quick Start](#quick-start)
- [Correctness Claims](#correctness-claims)
- [Architecture](#architecture)
- [Performance](#performance)
- [Non-Claims](#non-claims)
- [Testing & Campaigns](#testing--campaigns)
- [Contributing](#contributing)
- [License](#license)

---

## Why reservoir?

- **Exact by construction.** `ExactPERBuffer` performs zero floating-point arithmetic on any sampling decision path — priorities, tree sums, and importance-sampling weights are all exact Python integers / `fractions.Fraction`. The one unavoidable float operation (`p^alpha`) is isolated at a documented float64-once boundary. See [`docs/design.md`](docs/design.md).
- **Fast when you need it.** `FastPERBuffer` auto-selects a C-accelerated sum-tree (falling back to a vectorized numpy/torch implementation) for real training workloads, and can run side-by-side with an exact "audit" buffer that flags any divergence (`AuditedPERBuffer`).
- **Crash-safe.** The durable buffer (`durable.py`) uses write-ahead intent records, `F_FULLFSYNC`/`fsync`, and atomic rename as the commit point. Verified with 70/70 SIGKILL crash-injection tests and zero torn states.
- **Attested, not just tested.** Every sampled batch can carry a hash-chained cryptographic attestation record. An independent checker (`checker/verify.py`, which imports nothing from `reservoir` itself) re-derives the sum-tree from scratch and rejects forged batches — 63/63 forgery attempts caught in the mutation campaign.
- **Model-checked.** The replay lifecycle (insert/update/sample/evict/crash-recover) is specified in TLA+ and exhaustively checked: 44,611 states, no invariant violations.
- **Honest about its limits.** Every claim above has a matching non-claim in [`docs/nonclaims.md`](docs/nonclaims.md) — including a preregistered, falsified hypothesis (see below). We report negative results the same way we report positive ones.

If you just want a drop-in prioritized replay buffer for a Gymnasium environment, jump to [Quick Start](#quick-start). If you're evaluating whether your RL research needs bit-exact replay guarantees, start with [Correctness Claims](#correctness-claims).

---

## Install

```bash
git clone https://github.com/joshwu108/Reservoir.git
cd Reservoir

# Recommended: uv
brew install uv
uv sync

# Or plain pip (editable install)
pip install -e .

# Optional: Atari benchmark suite (stable-baselines3, ale-py, autorom)
pip install -e ".[atari]"
```

The C-accelerated sum-tree extension builds automatically as part of the package. If it fails to build for your platform, `reservoir` transparently falls back to the pure-Python/numpy implementation — check which backend you got with:

```bash
python -c "import reservoir; print(reservoir.backend)"  # "c" or "python"
```

---

## Quick Start

### Drop-in replay buffer for a Gymnasium env

```python
import gymnasium as gym
from reservoir.fast_buffer import FastPERBuffer
from reservoir.gym_wrapper import GymCollector
from reservoir.nstep import NStepBuffer
from reservoir.audit import AuditedPERBuffer

env = gym.make("CartPole-v1")

# 1. Create buffer from env (auto-detects obs/action shapes)
buf = FastPERBuffer.from_env(env, capacity=100_000, alpha=0.6, beta=0.4, device="cpu")

# 2. Optional: wrap with n-step returns (Rainbow DQN)
buf = NStepBuffer(buf, n=3, gamma=0.99)

# 3. Optional: wrap with a correctness audit layer (cross-checks against the exact buffer)
buf = AuditedPERBuffer(buf, audit_capacity=512, audit_interval=1000)

# 4. Collect experience
collector = GymCollector(env, buf.buffer.buffer, policy=lambda obs: env.action_space.sample())
episodes = collector.step(1000)

# 5. Train
batch = buf.sample(256)
# batch.states, batch.actions, batch.rewards, batch.next_states,
# batch.dones, batch.is_weights — all torch tensors, GPU-ready

# 6. Update priorities after the TD update
buf.update_priorities(batch.indices, td_errors)
buf.anneal_beta(step, total_steps)

print(buf.audit_report())
```

### Exact, attested buffer (for correctness-critical work)

```python
from reservoir.buffer import ExactPERBuffer

buf = ExactPERBuffer(capacity=1024, alpha=0.6, beta=0.4, seed=42)
buf.insert(experience, priority=1.0)
batch, attestation = buf.sample(batch_size=32, return_attestation=True)

# Independently verify the batch without trusting reservoir's own code
from checker.verify import verify_attestation
assert verify_attestation(attestation, buf.mutation_log())
```

Hindsight Experience Replay is available too — see `reservoir.her.HERBuffer`.

---

## Correctness Claims

`reservoir` was built around four specific, falsifiable claims:

| Thesis | Claim | Status | Evidence |
|--------|-------|--------|---------|
| **T1** | Proportional PER can be implemented with zero floats on any decision path | ✅ **Alive** | [`src/reservoir/`](src/reservoir/), [`tests/`](tests/) (270 tests) |
| **T2** | Float sum-tree idioms produce decision-relevant divergences from the exact reference | ❌ **Dead (falsified)** | [`results/divergence_campaign_report.json`](results/divergence_campaign_report.json) |
| **T3** | The durable buffer is failure-atomic under `SIGKILL` | ✅ **Alive** | 70/70 crash tests, 0 torn states |
| **T4** | An independent checker can verify sampled batches and reject forgeries | ✅ **Alive** | 63/63 forgeries rejected |

**T2 is a negative result, reported prominently on purpose.** The preregistered search ([`docs/preregistration.md`](docs/preregistration.md), thresholds frozen *before* any data was collected) found **zero decision-relevant divergences** between exact and float sum-tree idioms across 135 workloads. Max total-variation distance: 8.67×10⁻¹⁹, against a kill threshold of 2⁻⁴⁰ ≈ 9.09×10⁻¹³. The campaign used a reduced grid (5 workloads/cell, capacity ≤ 16384) due to pure-Python throughput; the full 10,000-workload grid is future work with an optimized exact implementation.

<details>
<summary>Campaign detail tables</summary>

**T3 — Crash Atomicity**

| Operations | Cut Points | Seeds | Total | Torn States |
|-----------|-----------|-------|-------|-------------|
| insert, update | 7 | 5 | 70 | **0** |

**T4 — Mutation / Forgery Detection**

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

**T2 — Float Divergence**

| Grid Cells | Workloads | Divergences | Max TV Distance |
|-----------|-----------|-------------|----------------|
| 27 | 135 | **0** | 8.67×10⁻¹⁹ |

</details>

---

## Architecture

```
reservoir/
├── src/reservoir/
│   ├── rational.py      # Exact float→integer conversion (float64-once boundary)
│   ├── sumtree.py        # Exact integer sum-tree (insert, update, prefix-locate)
│   ├── draw.py            # Keyed BLAKE2b uniform-integer draw (deterministic, not RNG)
│   ├── buffer.py           # ExactPERBuffer: insert/update/evict/sample, exact IS weights
│   ├── durable.py           # WAL durable buffer: F_FULLFSYNC, atomic rename, recovery
│   ├── attest.py             # Hash-chained MutationRecord + SampleAttestation
│   ├── fast_buffer.py         # FastPERBuffer: numpy/torch, vectorized tree, GPU-ready
│   ├── c_buffer.py             # C-accelerated FastPERBuffer backend
│   ├── csrc/                    # C sum-tree extension (sumtreemodule.c)
│   ├── gym_wrapper.py           # Gymnasium integration: from_env(), GymCollector
│   ├── nstep.py                  # NStepBuffer — n-step returns (Rainbow DQN)
│   ├── her.py                     # HERBuffer — Hindsight Experience Replay
│   └── audit.py                    # AuditedPERBuffer — exact shadow-buffer cross-check
├── checker/verify.py       # Independent verifier (imports nothing from src/reservoir)
├── campaigns/               # float_baselines.py, divergence.py, crash.py, mutation.py
├── benchmarks/                # Atari DQN benchmark suite (57 games, c/python/uniform)
├── spec/ReplayLifecycle.tla     # TLA+ safety model (44,611 states, no errors)
├── demo/tiny_dqn.py               # 5-state chain MDP: bitwise-identical, attested runs
├── docs/design.md, preregistration.md, nonclaims.md
└── results/                         # Campaign reports (JSON)
```

### Key design decisions

- **Float64-once boundary.** `priority_int = int(Fraction(p_i ** alpha) * 2**52)` — the only float op on the decision path, immediately integerized. All downstream arithmetic uses Python `int`/`Fraction`. Details and alternatives considered: [`docs/design.md`](docs/design.md) §2.
- **Deterministic keyed draw.** Sampling uses BLAKE2b-256 rejection sampling keyed by `(seed, buffer_id, op_counter)` — no `random`, `numpy.random`, or `torch` RNG on decision paths. This gives deterministic replay identity, **not** a cryptographic security guarantee.
- **Crash atomicity (POSIX).** Write-ahead intent records, `fcntl(fd, F_FULLFSYNC)` on Darwin (falls back to `os.fsync()` on Linux), atomic `rename()` as the commit point, then parent-directory fsync.
- **Independent checker.** `checker/verify.py` re-implements the sum-tree from scratch and never imports `src/reservoir` (enforced via an AST-based import check), so it can't inherit a bug from the code it's checking.

---

## Performance

`FastPERBuffer` (C backend) vs. Stable-Baselines3's uniform `ReplayBuffer`:

| | reservoir `FastPERBuffer` | SB3 `ReplayBuffer` (uniform) |
|--|--|--|
| Insert | 5μs | 2μs |
| Sample (batch=256, cap=100K) | 0.12ms | 0.03ms |
| Importance-sampling weights | ✓ | ✗ |
| Priority updates | ✓ | ✗ |
| GPU tensors | ✓ | ✓ |
| Correctness audit against an exact reference | ✓ (unique to reservoir) | ✗ |

The remaining gap vs. uniform sampling is inherent: PER requires an O(log N) tree traversal per sample where uniform sampling is O(1). An Atari benchmark harness (57 games, C/Python/uniform backends) is included in [`benchmarks/`](benchmarks/) if you want to reproduce or extend this comparison at scale.

---

## Non-Claims

Full list with rationale: [`docs/nonclaims.md`](docs/nonclaims.md).

- Not a performance result — pure-Python paths are orders of magnitude slower than float implementations; use `FastPERBuffer` for training.
- Not a large-scale RL training-quality result — sampling correctness ≠ training quality.
- Not a distributed buffer, and not a security boundary (the BLAKE2b draw is replay identity, not tamper-proofing).
- α-exponentiation uses the float64-once boundary — the sampled distribution is proportional to `binary64(p_i^α)`, not the exact real `p_i^α`.
- Crash-durability evidence is on macOS/APFS with `F_FULLFSYNC` only.
- The TLA+ model covers a small finite scope (capacity 2, 2-value priorities, ≤3 ops) — see §8.

---

## Testing & Campaigns

```bash
# Full test suite (270 tests)
make check
# or
uv run pytest tests/ -v

# T4: mutation/forgery campaign (63/63 rejected)
uv run python -m campaigns.mutation

# T3: crash atomicity campaign (70/70 pass)
uv run python -m campaigns.crash

# T2: float-divergence campaign (T2 DEAD)
uv run python -m campaigns.divergence

# TLA+ model check (44,611 states, no errors)
bash spec/check.sh

# End-to-end demo (bitwise-identical, attested DQN run)
uv run python -m demo.tiny_dqn
```

---

## Contributing

Contributions are welcome — this project benefits from more eyes on the correctness claims as much as from new features.

1. Fork the repo and create a feature branch.
2. Run `uv sync` to install dependencies, then `make check` to confirm the existing suite passes before you start.
3. If you touch anything on a sampling decision path (`rational.py`, `sumtree.py`, `draw.py`, `buffer.py`, `attest.py`), add or update tests in `tests/` — new floats on a decision path should fail review, not just tests.
4. If you touch `checker/verify.py`, remember it must never import from `src/reservoir` (this is CI-enforced via an AST check) — it exists specifically to not share bugs with the code it verifies.
5. Open a PR describing what changed and why, and which of the correctness campaigns (if any) you re-ran.

Bug reports and questions are welcome via [GitHub Issues](https://github.com/joshwu108/Reservoir/issues).

---

## License

[Apache-2.0](LICENSE)
