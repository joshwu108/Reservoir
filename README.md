# reservoir

**Exact, reproducible, auditable replay for LLM reinforcement learning.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

> **Status:** this README describes the target interface. Not everything below is
> implemented yet.

Generating rollouts is the most expensive part of GRPO-style training, and the
standard recipe uses each rollout once and throws it away. Reservoir is a replay
library that lets you keep them: store rollouts, re-sample them by priority, and
get a record of exactly what was sampled and why.

It is built around three properties:

1. **Exact** — priorities are integers and the sum-tree is integer arithmetic.
   Floats enter at one declared boundary.
2. **Reproducible and verifiable** — draws are keyed BLAKE2b hashes, not RNG
   calls. Every sampled batch is written to a hash-chained attestation log that
   an independent checker can replay.
3. **Crash-atomic** — a write-ahead log with full fsync. A killed trial does not
   lose its rollout history.

```bash
pip install reservoir
```

---

## Quick start — replay for GRPO

```python
from reservoir import RolloutBuffer, Rollout
from reservoir.priorities import AdvantagePriority

buf = RolloutBuffer(
    capacity=50_000,
    priority=AdvantagePriority(),
    half_life=4,            # priority halves every 4 model versions
    max_policy_age=16,      # evict rollouts older than 16 versions
    seed=0,
)

# After each generation step, store the group of rollouts for a prompt
buf.add_group(
    prompt_id="gsm8k-0412",
    model_version=step,
    rollouts=[
        Rollout(tokens=ids, logprobs=lp, reward=r)
        for ids, lp, r in zip(completions, behavior_logprobs, rewards)
    ],
)

# Mix replayed rollouts into the next update
batch = buf.sample(batch_size=64, current_version=step)
# batch.rollouts       variable-length entries
# batch.logprobs       behavior logprobs, for importance correction
# batch.model_versions the policy version that produced each rollout
# batch.is_weights     importance-sampling weights
# batch.indices

buf.update_priorities(batch.indices, new_advantages)
```

Entries are whole rollouts, not fixed-shape transitions. Each one carries the
behavior logprobs and the model version that produced it, which is what
off-policy correction and staleness handling need.

---

## Priority strategies

Priority functions are pluggable. Each one maps a rollout or a prompt to a
non-negative score; Reservoir handles integerization, decay and sampling.

```python
from reservoir.priorities import (
    AdvantagePriority,     # |advantage| of the rollout
    PassRateTargeting,     # favor prompts near a target pass rate
    PassRateVariance,      # favor prompts with uncertain outcomes
)

buf = RolloutBuffer(
    capacity=50_000,
    priority=PassRateTargeting(target=0.5, width=0.15),
)
```

Write your own by implementing one method:

```python
from reservoir.priorities import PriorityStrategy

class RewardGap(PriorityStrategy):
    def score(self, rollout, group) -> float:
        return abs(rollout.reward - group.mean_reward)
```

**Age decay** is exact. Priorities decay by half-life in model versions, and
the decayed distribution is the declared distribution the checker verifies.
See [`docs/design.md`](docs/design.md).

**Prompt-level sampling** is available through `DatasetBuffer`, for choosing
which prompts to generate rollouts for in the first place:

```python
from reservoir import DatasetBuffer
from reservoir.priorities import PassRateVariance

prompts = DatasetBuffer(dataset, priority=PassRateVariance())
next_prompts = prompts.sample_indices(batch_size=128)
```

---

## Reproducible, verifiable runs

Deterministic inference engines make the compute reproducible. Reservoir does
the same for the data: which entry was sampled, when, and under what priority.

```python
buf = RolloutBuffer(capacity=50_000, seed=0, attest="run-01/attest.jsonl")
```

Every insert, priority update, eviction and sampled batch is appended to the
log. Anyone with the log can verify it, without your code or your model:

```bash
python -m checker.verify run-01/attest.jsonl
```

The checker shares no code with the library. It rebuilds the sum-tree from the
mutation records and confirms that every sampled index follows from the
recorded draw.

Two runs with the same seed and the same inputs produce identical sampling
transcripts. Paired with a deterministic inference engine, that gives a
bitwise-reproducible training run with a sampling record a third party can
check.

The same log supports data-mixture and quota reporting, and sampling records
for unlearning audits. It is a consistency-verification tool, not a security
boundary; see [`docs/nonclaims.md`](docs/nonclaims.md).

---

## Durable buffers

```python
buf = RolloutBuffer(capacity=50_000, directory="run-01/buffer")
```

Every operation is committed through a write-ahead log. If the process is
killed, reopening the same directory recovers the last committed state, with
no torn entries.

---

## Integrations

### TRL

```bash
pip install "reservoir[trl]"
```

```python
from reservoir.integrations.trl import ReservoirReplay

trainer = GRPOWithReplayBufferTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    replay_buffer=ReservoirReplay(capacity=50_000, half_life=4),
)
trainer.train()
```

### verl

```bash
pip install reservoir-verl
```

A trajectory store and prioritized sampler plugin with WAL durability, so
rollout history survives a failed trial.

### Classic RL

`FastPERBuffer` is a C-backed PER buffer for transition-based RL, with n-step
and HER wrappers. Adapters are provided for Stable-Baselines3 and TorchRL.

```python
from reservoir import FastPERBuffer

buf = FastPERBuffer(capacity=100_000, obs_shape=(84, 84, 4), alpha=0.6, beta=0.4)
buf.add(obs, action, reward, next_obs, done)
batch = buf.sample(batch_size=32)
buf.update_priorities(batch.indices, td_errors)
```

---

## Fine-tuning tools

### Preference noise detection

Finds likely mislabeled pairs in RLHF preference data from their loss
trajectories during reward-model training.

```bash
pip install "reservoir[prefcheck]"
```

```python
from reservoir import PreferenceNoiseDetector

detector = PreferenceNoiseDetector(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,   # "chosen" / "rejected" fields
)
detector.train()

report = detector.get_report()
for r in report.flipped[:10]:
    print(r.example_idx, r.confidence)
report.to_html("noise_report.html")
```

| Label | Meaning |
|-------|---------|
| `FLIPPED` | Label is probably wrong — re-annotate or remove |
| `AMBIGUOUS` | Genuine annotator disagreement — get a second opinion |
| `CLEAN` | Fine |

### Forgetting monitor

Measures forgetting during fine-tuning on a held set of anchor examples, and
can replay the most-forgotten anchors back into training.

```bash
pip install "reservoir[anchor]"
```

```python
from reservoir import AnchorSet, ForgettingMonitor

anchors = AnchorSet.from_dataset(prior_knowledge_dataset, n=500, tags="legal-QA")

monitor = ForgettingMonitor(
    anchor_sets=[anchors],
    metrics=["loss", "kl_to_base"],
    eval_every_n_steps=500,
    alert_threshold=0.5,
    auto_replay=True,
    replay_ratio=0.1,
)

trainer = Trainer(model=model, args=args, train_dataset=data, callbacks=[monitor])
trainer.train()

report = monitor.get_report()
report.transitions   # per-example correct→wrong and wrong→correct counts
report.to_html("forgetting_report.html")
```

---

## Guarantees

| Claim | Evidence |
|-------|----------|
| Sampling is deterministic and reproducible under a keyed draw | Property tests against a brute-force reference |
| The durable buffer is failure-atomic under SIGKILL | 70/70 crash tests, zero torn states |
| The independent checker rejects forged logs | 63/63 mutants rejected |
| The lifecycle protocol is safe within a finite scope | TLA+ model, 44,611 states |

Reservoir reports negative results. A pre-registered search for
decision-relevant divergence between float and exact sum-trees found none
([`docs/preregistration.md`](docs/preregistration.md)); float PER is accurate
enough in practice. Exactness is for reproducibility and verification, not
for training quality.

What Reservoir does not claim is listed in
[`docs/nonclaims.md`](docs/nonclaims.md).

---

## Install

```bash
pip install reservoir                 # core
pip install "reservoir[trl]"          # TRL integration
pip install "reservoir[prefcheck]"    # preference noise detector
pip install "reservoir[anchor]"       # forgetting monitor
pip install "reservoir[atari]"        # Atari benchmark suite

python -c "import reservoir; print(reservoir.backend)"  # "c" or "python"
```

A C compiler is needed for the C extension. Without one, Reservoir falls back
to the numpy implementation.

---

## Development

```bash
uv run pytest tests/ -v               # tests
uv run python -m campaigns.mutation   # forgery detection campaign
uv run python -m campaigns.crash      # crash atomicity campaign
uv run python -m campaigns.divergence # float divergence campaign
bash spec/check.sh                    # TLA+ model check
make check                            # tests + checker import isolation
```

---

## License

[Apache-2.0](LICENSE)
