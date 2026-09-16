# reservoir

**Fast, auditable priority sampling for any PyTorch training loop.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-337%20passing-brightgreen.svg)](tests/)
[![TLA+ verified](https://img.shields.io/badge/TLA%2B-44%2C611%20states%2C%20no%20errors-success.svg)](spec/ReplayLifecycle.tla)

Reservoir is a prioritized replay buffer library for PyTorch. It samples the most
informative examples more often — whether those examples are Atari frames in an RL
agent or training pairs in an LLM fine-tuning run. It comes with a C extension for
speed, a crash-safe durable mode, a cryptographic audit chain for correctness
verification, a preference noise detector that catches mislabeled RLHF pairs before
they corrupt your reward model, and a catastrophic forgetting monitor that alerts
you during fine-tuning before your eval suite fails.

```bash
pip install reservoir
```

---

## What it does

- **Prioritized sampling** — sample transitions/examples proportional to their
  importance (TD error for RL, training loss for supervised learning), with
  importance-sampling weight correction to prevent bias
- **Fast** — C-backed sum-tree, vectorized numpy fallback, returns `torch.Tensor`
  directly; auto-selects best available backend at import time
- **Auditable** — every sampled batch carries a hash-chained attestation record;
  an independent verifier re-derives the sum-tree from scratch to catch any divergence
- **Crash-safe** — write-ahead log with `F_FULLFSYNC` and atomic rename; verified
  with 70 SIGKILL crash tests, zero torn states
- **Catastrophic forgetting monitor** — attach `ForgettingMonitor` to any HuggingFace
  Trainer; get per-group forgetting alerts during training, not after; optionally
  replay the most-forgotten anchors back into training via PER
- **Drop-in** — same API whether you use the C backend, numpy fallback, or exact
  reference implementation

---

## Quick start — RL replay buffer

```python
from reservoir import FastPERBuffer

buf = FastPERBuffer(
    capacity=100_000,
    obs_shape=(84, 84, 4),  # Atari frame stack
    alpha=0.6,               # prioritization strength
    beta=0.4,                # IS correction (anneal to 1.0)
    device="cuda",
)

print(reservoir.backend)  # "c" if C extension built, "python" otherwise

# Add a transition
buf.add(obs, action, reward, next_obs, done)

# Sample — returns torch tensors on your device
batch = buf.sample(batch_size=32)
# batch.states, batch.actions, batch.rewards, batch.next_states,
# batch.dones, batch.is_weights, batch.indices

# After computing TD errors, update priorities
buf.update_priorities(batch.indices, td_errors)
buf.anneal_beta(step, total_steps)  # linearly anneals beta → 1.0
```

---

## Quick start — LLM / supervised fine-tuning

The core insight of PER applies to any gradient-based training: sample the examples
your model finds most surprising (high loss) more often, correct for the sampling
bias with IS weights. Uniform sampling wastes steps on examples the model already knows.

```python
import torch
from reservoir import FastPERBuffer

# One slot per training example; obs_shape holds the tokenized input
buf = FastPERBuffer(
    capacity=len(dataset),
    obs_shape=(seq_len,),
    action_dim=1,
    alpha=0.6,
    beta=0.4,
    device="cuda",
)

# Seed the buffer — initial priority = max (every example seen at least once)
for i, example in enumerate(dataset):
    buf.add(
        state=example["input_ids"],
        action=0,
        reward=0.0,
        next_state=example["input_ids"],
        done=False,
    )

# Training loop
for step in range(total_steps):
    batch = buf.sample(batch_size=32)

    # Forward pass — compute per-example loss
    logits = model(batch.states)
    loss_per_example = F.cross_entropy(logits, targets, reduction="none")

    # IS-weighted loss — prevents over-fitting to hard examples
    loss = (batch.is_weights * loss_per_example).mean()
    loss.backward()
    optimizer.step()

    # Update priorities with current per-example loss
    buf.update_priorities(batch.indices, loss_per_example.detach().cpu().numpy())
    buf.anneal_beta(step, total_steps)
```

---

## Catastrophic forgetting monitor

When you fine-tune on new data, the model forgets what it previously knew. Teams
typically detect this after training by running eval suites — by then the compute is
wasted and a retrain is needed.

`reservoir-anchor` gives you an early warning **during** training. You hold a small
`AnchorSet` of examples representing prior knowledge, run them through the model
every N steps, and get per-group forgetting alerts the moment anchor loss rises
significantly above baseline.

The PER connection is direct: priority = how much the model has forgotten a specific
anchor (relative loss increase from baseline). `ReplayScheduler` samples the
most-forgotten anchors proportionally for replay, with IS weight correction.

```bash
pip install "reservoir[anchor]"
```

```python
from reservoir import AnchorSet, ForgettingMonitor
from transformers import Trainer, TrainingArguments

# Build anchor set from examples the model must not forget
anchor_set = AnchorSet.from_dataset(
    prior_knowledge_dataset,
    n=500,
    tags="legal-QA",          # group label for per-group alerts
)

monitor = ForgettingMonitor(
    anchor_sets=[anchor_set],
    eval_every_n_steps=500,    # evaluate anchors every 500 steps
    alert_threshold=0.5,       # alert when mean loss rose > 50% above baseline
    auto_replay=False,         # True to inject forgotten anchors back into training
    verbose=True,
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(...),
    train_dataset=fine_tune_dataset,
    callbacks=[monitor],
)
trainer.train()

# After training: get the full report
report = monitor.get_report()
print(f"Alerts fired: {report.n_alerts}")
for tag, group in report.groups.items():
    print(f"  {tag}: final_forgetting_score={group.final_forgetting_score:.3f}")

report.to_json("forgetting_report.json")
report.to_html("forgetting_report.html")
report.plot("forgetting_plot.png")   # requires matplotlib
```

Enable automatic replay of the most-forgotten anchors:

```python
monitor = ForgettingMonitor(
    anchor_sets=[anchor_set],
    eval_every_n_steps=500,
    alert_threshold=0.5,
    auto_replay=True,          # inject forgotten anchors back into training
    replay_ratio=0.1,          # 10% of each batch is replayed anchors
)
```

Multiple anchor groups — track forgetting separately per domain:

```python
legal_anchors   = AnchorSet.from_dataset(legal_dataset,   n=200, tags="legal")
medical_anchors = AnchorSet.from_dataset(medical_dataset, n=200, tags="medical")
code_anchors    = AnchorSet.from_dataset(code_dataset,    n=100, tags="code")

monitor = ForgettingMonitor(
    anchor_sets=[legal_anchors, medical_anchors, code_anchors],
    alert_threshold=0.5,
)
```

---

## Preference noise detection (RLHF)

`reservoir-prefcheck` wraps TRL's `RewardTrainer` to catch mislabeled preference
pairs before they poison your reward model. The key insight: when a pair is
mislabeled, the reward model's loss on it stays high or rises over training because
the model keeps predicting the correct answer and getting penalized.

```bash
pip install "reservoir[prefcheck]"
```

```python
from reservoir import PreferenceNoiseDetector

detector = PreferenceNoiseDetector(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,   # HuggingFace Dataset with "chosen"/"rejected" fields
    mode="audit",            # "audit" (uniform) or "accelerated" (PER sampling)
)
detector.train()

report = detector.get_report()
print(report.summary())
# {'n_total': 10000, 'n_flipped': 83, 'pct_flipped': 0.83,
#  'n_ambiguous': 214, 'pct_ambiguous': 2.14, 'n_clean': 9703, ...}

# Top suspect pairs — highest confidence mislabeling
for r in report.flipped[:10]:
    print(r.example_idx, r.confidence, r.features.slope)

# Export
report.to_json("noise_report.json")
report.to_csv("noise_report.csv")
report.to_html("noise_report.html")
```

Three loss-trajectory buckets:

| Label | Trajectory shape | Meaning |
|-------|-----------------|---------|
| `FLIPPED` | Loss flat or rising (slope > 0.01, high end-loss) | Label is probably wrong — re-annotate or remove |
| `AMBIGUOUS` | Loss oscillates (high variance, near-zero slope) | Genuine annotator disagreement — get a second opinion |
| `CLEAN` | Loss decays normally | Fine |

---

## Wrappers

```python
from reservoir.nstep import NStepBuffer
from reservoir.her import HERBuffer
from reservoir.audit import AuditedPERBuffer

# n-step returns for Rainbow DQN
buf = NStepBuffer(buf, n=3, gamma=0.99)

# Hindsight Experience Replay for goal-conditioned tasks
buf = HERBuffer(buf, goal_strategy="future", k=4)

# Shadow exact buffer — alerts if C/numpy sampling diverges from reference
buf = AuditedPERBuffer(buf, audit_capacity=512, audit_interval=1000)
print(buf.audit_report())
```

---

## Install

```bash
# Standard install (builds C extension automatically)
pip install reservoir

# With Atari benchmark suite
pip install "reservoir[atari]"

# With catastrophic forgetting monitor
pip install "reservoir[anchor]"

# With preference noise detector (requires TRL)
pip install "reservoir[prefcheck]"

# Check which backend is active
python -c "import reservoir; print(reservoir.backend)"  # "c" or "python"
```

A C compiler is required to build the C extension. If none is available, `reservoir`
falls back to the pure-Python/numpy implementation transparently. To opt into
CPU-specific tuning:

```bash
CFLAGS="-O3 -march=native" pip install reservoir
```

---

## Performance

C-backed `FastPERBuffer` vs. Stable-Baselines3 uniform `ReplayBuffer`:

| | reservoir (C backend) | SB3 ReplayBuffer (uniform) |
|--|--|--|
| Insert | 5μs | 2μs |
| Sample (batch=256, cap=100K) | 0.12ms | 0.03ms |
| Importance-sampling weights | ✓ | ✗ |
| Priority updates | ✓ | ✗ |
| GPU tensors | ✓ | ✓ |
| Correctness audit | ✓ | ✗ |

The sampling gap vs. uniform is inherent: PER is O(log N) tree traversal vs. O(1)
for uniform. The C extension closes the gap vs. pure-Python significantly.

---

## Correctness guarantees

Reservoir was built around four falsifiable claims. Three are proven; one is a
prominently reported negative result.

| | Claim | Result |
|--|--|--|
| **T1** | PER can be implemented with zero floats on any decision path | ✅ Alive — 337 tests |
| **T2** | Float sum-trees produce decision-relevant divergences from the exact reference | ❌ Dead — falsified |
| **T3** | Durable buffer is failure-atomic under SIGKILL | ✅ Alive — 70/70 crash tests |
| **T4** | Independent checker verifies batches and rejects forgeries | ✅ Alive — 63/63 rejected |

**T2 is a negative result, reported on purpose.** A pre-registered search (thresholds
frozen before data collection, see [`docs/preregistration.md`](docs/preregistration.md))
found zero decision-relevant divergences between exact and float sum-trees across 135
workloads. Max total-variation distance: 8.67×10⁻¹⁹, well below the kill threshold of
2⁻⁴⁰. Float PER is accurate enough in practice. Reservoir's exact implementation
remains the reference for verifying this on new workloads.

**T5 — Preference noise detection.** `PreferenceNoiseDetector` correctly recovers
100% precision and recall on synthetic FLIPPED pairs (rising-loss trajectories).
AMBIGUOUS detection precision is 100%; recall is bounded by the 75th-percentile
variance threshold. See `benchmarks/prefcheck/synthetic_noise.py`.

<details>
<summary>Campaign detail tables</summary>

**T3 — Crash Atomicity (70/70 pass, 0 torn states)**

| Operations | Cut Points | Seeds | Total | Torn States |
|-----------|-----------|-------|-------|-------------|
| insert, update | 7 | 5 | 70 | **0** |

**T4 — Forgery Detection (63/63 rejected)**

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

**T2 — Float Divergence (T2 dead)**

| Grid Cells | Workloads | Divergences | Max TV Distance |
|-----------|-----------|-------------|----------------|
| 27 | 135 | **0** | 8.67×10⁻¹⁹ |

</details>

---

## Architecture

```
src/reservoir/
  fast_buffer.py         FastPERBuffer — numpy/torch, vectorized tree, GPU-ready
  c_buffer.py            CFastPERBuffer — C-backed sum-tree, same API as FastPERBuffer
  csrc/                  C extension: sumtree.c, sumtreemodule.c → reservoir._sumtree
  buffer.py              ExactPERBuffer — exact integer arithmetic, reference implementation
  durable.py             WAL durable buffer — F_FULLFSYNC, atomic rename, crash recovery
  attest.py              Hash-chained MutationRecord + SampleAttestation
  draw.py                BLAKE2b-256 keyed draw — deterministic, no RNG on decision paths
  rational.py            float64-once boundary — p^alpha integerized via Fraction
  nstep.py               NStepBuffer — n-step return wrapper
  her.py                 HERBuffer — Hindsight Experience Replay
  audit.py               AuditedPERBuffer — shadow exact buffer cross-check
  gym_wrapper.py         GymCollector — Gymnasium environment integration
  anchor_set.py          AnchorSet — tracks per-example forgetting severity  [anchor]
  forgetting_monitor.py  ForgettingMonitor TrainerCallback — real-time alerts [anchor]
  replay_scheduler.py    ReplayScheduler — PER-prioritized anchor replay      [anchor]
  dataset_buffer.py      DatasetBuffer — prioritized sampler for HF datasets  [prefcheck]
  prefcheck.py           PreferenceNoiseDetector — RLHF label noise detection [prefcheck]
  report.py              PreferenceQualityReport — noise results and export   [prefcheck]

checker/verify.py        Independent verifier — zero imports from src/reservoir
benchmarks/              57-game Atari DQN benchmark suite
spec/                    TLA+ safety model — 44,611 states, no invariant violations
```

---

## Running tests and campaigns

```bash
# All 337 tests
uv run pytest tests/ -v

# Forgery detection campaign (T4)
uv run python -m campaigns.mutation

# Crash atomicity campaign (T3)
uv run python -m campaigns.crash

# Float divergence campaign (T2)
uv run python -m campaigns.divergence

# TLA+ model check
bash spec/check.sh

# End-to-end DQN demo — bitwise-identical attested runs
uv run python -m demo.tiny_dqn
```

---

## License

[Apache-2.0](LICENSE)
