# reservoir

**Exact, reproducible, auditable replay for LLM reinforcement learning.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

> **Status:** the rollout buffer, priority strategies, age decay, attestation
> with content commitments, the independent checker with its transcript and
> diff tools, the durable buffer, the TRL integration and the reproducibility
> demo below are implemented and tested. The verl integration and the
> Stable-Baselines3 / TorchRL adapters are not yet.

Generating rollouts is the most expensive part of GRPO-style training, and the
standard recipe uses each rollout once and throws it away. Reservoir is a replay
library that lets you keep them: store rollouts, re-sample them by priority, and
get a record of exactly what was sampled and why.

It is built around three properties:

1. **Exact** — priorities are integers and the sum-tree is integer arithmetic.
   Floats enter at one declared boundary.
2. **Reproducible and verifiable** — draws are keyed BLAKE2b hashes, not RNG
   calls. Every stored example (by content digest and source) and every sampled
   batch is written to a hash-chained attestation log that an independent
   checker can replay, report on, and compare between runs.
3. **Crash-atomic** — a write-ahead log with full fsync. A killed trial does not
   lose its rollout history.

```bash
pip install reservoir-replay          # import name: reservoir
```

The distribution is `reservoir-replay` (the name `reservoir` on PyPI belongs
to an unrelated package); the import name is `reservoir`. The rollout buffer,
the attestation log and the checker are pure Python with no required
dependencies; numpy and torch are pulled in by the extras that use them.

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

Write your own by implementing one method, and optionally a second that
re-scores a rollout when it is replayed (the TRL adapter calls it with what
is known at that moment: the weighted advantage, the importance weight, the
log-ratio to the current policy, the age in steps):

```python
from reservoir.priorities import PriorityStrategy, ReplaySignal

class RewardGap(PriorityStrategy):
    def score(self, rollout, group) -> float:
        return abs(rollout.reward - group.mean_reward)

    def rescore(self, rollout, group, signal: ReplaySignal) -> float | None:
        return None if signal.log_ratio is None else abs(rollout.reward) / (1 + abs(signal.log_ratio))
```

Rescoring is a hook, not a recommendation: the library ships no rescoring
strategy of its own, and every rescoring is an `update` record in the log.

**Age decay** is exact. Priorities decay by half-life in model versions, and
the decayed distribution is the declared distribution the checker verifies.
The decay is base-2 with an integer half-life, not `exp(-age/tau)`; see
[`docs/design.md`](docs/design.md) §7 for the arithmetic and §8 for the
buffer's rules on eviction, versions and updates.

**Prompt-level sampling** is available through `DatasetBuffer`, for choosing
which prompts to generate rollouts for in the first place:

```python
from reservoir import DatasetBuffer
from reservoir.priorities import PassRateVariance

prompts = DatasetBuffer(dataset, priority=PassRateVariance())
next_prompts = prompts.sample_indices(batch_size=128)
# after generating a group for prompt i:
prompts.update_group(i, model_version=step, rollouts=group_rollouts)
```

---

## Reproducible, verifiable runs

Deterministic inference engines make the compute reproducible. Reservoir does
the same for the data: which example was sampled, when, and under what
probability.

```python
buf = RolloutBuffer(capacity=50_000, seed=0,
                    attest="run-01/attest.jsonl", manifest="run-01/manifest.jsonl")
buf.add_group(prompt_id, model_version=step, rollouts=rollouts, source="gsm8k")
```

Every insert (with the content digest of prompt, completion and reward, and
the source tag), priority update, eviction and sampled batch is appended to
the log; the manifest holds the opening of every digest. Anyone with the files
can verify them, without your code or your model:

```bash
python -m checker.verify run-01/attest.jsonl --manifest run-01/manifest.jsonl
# e.g. OK: 451 records verified; 296 examples committed; manifest opens 296 of them
```

The checker shares no code with the library. It rebuilds the sum-tree from
the mutation records, recomputes every decayed leaf and every content digest,
and confirms that every sampled index follows from the recorded draw and
names a committed example.

Two runs with the same seed and the same inputs produce byte-identical logs.
The demo runs the TRL integration three times on CPU and compares:

```bash
uv run python -m demo.reproducible_grpo        # output abridged
# run a: seed=42 buffer_seed=0 records=102 replaced_rows=8 head=f81a9b4b6aeb1335…
# run b: seed=42 buffer_seed=0 records=102 replaced_rows=8 head=f81a9b4b6aeb1335…
# run c: seed=43 buffer_seed=0 records=102 replaced_rows=8 head=eb6853bf6f813344…
# a vs b (same seeds): identical
# a vs c (different data seed): first difference at record 1 (data) ... differing fields: content_digest
```

When two logs differ, `python -m checker.diff a.jsonl c.jsonl` says at which
record and why: `data` (the stored examples differed upstream; every draw
before that point was identical), `schedule`, or `config` (different buffer
parameters, including the seed, which the log records). Logs and the report are committed under
`benchmarks/modal/results/repro_cpu_12steps_seed42/` and
`results/reproducible_grpo_report.json`; the same triplet on a T4 with HF
generation, with and without deterministic kernels, is under
`benchmarks/modal/results/repro_hf_t4_*` (verdict IDENTICAL in both). The
write-up is [`docs/reproducible-training.md`](docs/reproducible-training.md).

---

## Auditable training data

The transcript tool turns a verified log into the answers an audit asks for:

```bash
python -m checker.transcript run-01/attest.jsonl --manifest run-01/manifest.jsonl --by source
python -m checker.transcript run-01/attest.jsonl --quota scraped=0 --quota licensed=50000   # exit 2 if exceeded
python -m checker.transcript run-01/attest.jsonl --find <content digest>                     # every time it was sampled
python -m checker.transcript run-01/attest.jsonl --manifest run-01/manifest.jsonl --blast-radius <content digest>  # what it reached
```

Exposure per example (times sampled, exact importance-weight sum), mixture
per source overall and per model version, quota verdicts, and the sample
records where a given example appears. Every number is derived from the log;
the manifest only adds the readable example next to its digest.

Reservoir implements a subset of the data-commitment and
public-replayable-sampler components of Verifiable Fine-Tuning (arXiv
2510.16830): content digests and self-declared source tags opened by a
manifest, and keyed, publicly replayable draws under a hash chain. It makes
no claim about the update step. A transcript is evidence a team can use to
defend per-source statements about what the replay buffer inserted and
replayed during post-training; fresh rollouts used directly in a step, and
data outside the buffer, are not in the log, and the transcript is not an EU
AI Act training-content summary. The log commits and the manifest opens: a
chain-consistent change to a digest or source is invisible without the
manifest, and the mutation campaign measures exactly that. See
[`docs/nonclaims.md`](docs/nonclaims.md) §15–§19.

---

## Durable buffers

```python
from reservoir import DurableRolloutBuffer

buf = DurableRolloutBuffer("run-01/buffer", capacity=50_000, half_life=4,
                           max_policy_age=16, seed=0,
                           attest="run-01/attest.jsonl", manifest="run-01/manifest.jsonl")
```

Every operation is committed through a write-ahead log before it returns,
including the stale evictions and rebase an operation may trigger. If the
process is killed, reopening the same directory with the same parameters
recovers the last committed state: the same live rollouts, the same next
draw, and the same attestation chain and manifest, with no torn entries. Each
operation appends one command to a write-ahead log and the buffer snapshots
every 256 commands, so per-operation cost does not grow with history: in
`results/durable_overhead.json` (3000 groups of 8 rollouts, 128 tokens each,
Apple silicon) an `add_group` took 4.5 ms in the first tenth of the run and
5.7 ms in the last, against 0.4 and 1.0 ms in memory. The snapshot does grow,
because the attestation log is part of the state: compaction took 0.6 s at the
median and 0.9 s by the end of that run, 3.4 ms per operation amortised, and
reopening replayed in 5.3 s.
`buf.checkpoint("step-100")` and `buf.restore_checkpoint("step-100")` bind the
buffer to a training checkpoint; the TRL adapter does this on every save
(pruning buffer checkpoints to the ones the trainer kept) and rewinds on
resume, so a restarted run continues the chain from the checkpoint rather
than from wherever the crash happened. Resume at a non-zero step without a
matching buffer checkpoint is an error, not a fresh buffer.

---

## Integrations

### TRL

```bash
pip install "reservoir-replay[trl]"      # pins trl==1.13.0
```

```python
from trl import GRPOConfig
from reservoir.integrations.trl import ReservoirGRPOTrainer, ReservoirReplay

trainer = ReservoirGRPOTrainer(
    model=model,
    args=GRPOConfig(...),
    train_dataset=dataset,
    reward_funcs=[reward],
    replay_buffer=ReservoirReplay(capacity=50_000, half_life=4, max_policy_age=16,
                                  seed=0, attest="run-01/attest.jsonl",
                                  manifest="run-01/manifest.jsonl", source="gsm8k"),
)
trainer.train()
```

`ReservoirGRPOTrainer` is `GRPOTrainer` plus one override: after each
generation step it stores every non-empty completion of a prompt whose rewards varied,
fills the rows of prompts whose rewards were all equal (which contribute no
gradient) with rollouts replayed from the buffer, checks the written rows
against the sampled rollouts, and writes a batch witness naming which row
holds which draw, which the checker proves consistent with the sample record
(`reservoir-transcript --explain STEP ROW` says why a given row is in the
batch). Every step it logs replay health under `reservoir/` in TRL's metrics
(replay fraction, dead and near-dead groups, effective sample size and
staleness of the replayed rows, log-ratio between stored and current
logprobs) and into the attestation log, where the checker recomputes the
effective sample size and the staleness. An optional drift gate (`max_log_ratio=`) declines rows whose
log-ratio is too large; declines are recorded, never silent. After an incident,
`buffer.quarantine(predicate, reason)` evicts every matching entry with a record
that carries the predicate text and the reason, and
`reservoir-transcript --blast-radius <digest>` lists every step and training-batch
row the example (or, with the manifest, its prompt) reached. The manifest can carry
each example's per-reward-function values, numeric only and outside the digest. Replayed rows carry their
behavior logprobs, so the loss applies a real off-policy ratio, and their
advantages are multiplied by the importance-sampling weight. Versions,
half-life and `max_policy_age` are counted in optimizer steps. Every insertion
(with its content digest and the `source` tag), draw and eviction goes to the
attestation log; `python -m checker.verify` checks it,
`python -m checker.transcript` reports on it, and `reservoir-replay-offline`
rebuilds every witnessed batch as content (tokens, rewards, weights, sources)
from the log and manifest alone, for a data-side audit or rerun with no generation. A step with nothing to replay returns the batch TRL produced
unchanged (see [`docs/nonclaims.md`](docs/nonclaims.md) §14 for the one
RNG caveat).

Scope: text-only, TRL 1.13.0, one or more training processes (rank 0 owns
the buffer and the single log writer; run on two GPUs through
`accelerate launch`, see [`docs/nonclaims.md`](docs/nonclaims.md) §21).
The adapter refuses batches
with tool masks, vLLM importance-sampling ratios or vision inputs, and fails
with a clear message if the installed TRL lacks the trainer members it relies
on. It does not target TRL's experimental `GRPOWithReplayBufferTrainer`: TRL
removed that trainer after 1.13.0, and in 1.13.0 the trainer fed its buffer
only the first group of each batch. See [`docs/design.md`](docs/design.md)
§9 for the interface mapping.

### verl

Not yet available. verl's disaggregated generation and update suit a
rank-0 buffer; the adapter is planned after the multi-process TRL adapter.

### Classic RL

`FastPERBuffer` is a C-backed PER buffer for transition-based RL, with n-step
and HER wrappers. No Stable-Baselines3 or TorchRL adapter exists yet; the
wrappers are tested but have no consumer in this repository.

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
pip install "reservoir-replay[prefcheck]"
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
pip install "reservoir-replay[anchor]"
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
| Sampling is deterministic and reproducible under a keyed draw | Property tests against a brute-force reference; the log records seed, buffer id and `beta`, and the checker recomputes every draw and importance weight from them |
| The durable buffers are failure-atomic under SIGKILL | 275/275 crash tests across both buffers: every child was killed at its armed cut point (the campaign fails a row otherwise), 210 recovered the pre-state and 65 the post-state (an unsynced write that SIGKILL leaves in the page cache recovers to the post-state; a power loss there would give the pre-state, which is also legal), zero torn; the rollout cases cross the command log and the snapshot protocol in one operation, crash mid-rebase, crash inside a quarantine of a whole group, and crash inside a checkpoint restore |
| The independent checker rejects forged logs | 221/221 mutants rejected: 38 age-decay protocol forgeries, 29 content-commitment forgeries (3 of them chain-consistent, invisible without the manifest), 21 draw and weight forgeries (2 of them invisible to a log that does not record its seed and `beta`), 14 batch-witness forgeries, 18 telemetry forgeries, 22 quarantine and reward-provenance forgeries (the campaign also records that a quarantine record's texts and the manifest's reward values are not committed: 4 such changes pass, by design) and 16 manifest tamperings against the offline replay (the campaign also measures that 3 tamperings of the reward-provenance values, which the log does not commit to, run and change only those fields of the output) |
| A witnessed batch can be rebuilt as content from the log and manifest alone | `reservoir-replay-offline` emits every witnessed batch as JSON lines (rows, draws, exact importance weights, tokens, rewards, sources) importing nothing from the library; on the committed CPU and T4 reproducibility runs, runs a and b replay byte-identically and run c differs; fresh rows are listed by index only (`docs/nonclaims.md` §23) |
| Two runs with the same inputs give one transcript | CPU demo: runs a and b byte-identical (102 records), run c differs at record 1, classified `data` |
| Attestation is cheap relative to generation | about 3× the no-attestation insert cost in memory, 6× with a manifest file; the checker verifies 10k records in 0.3 s |
| The lifecycle protocol is safe within a finite scope | TLA+ model, 44,611 states. A no-parent-fsync variant (`spec/NoParentFsync.tla`) is written to show why the directory fsync exists; `bash spec/check.sh --with-counterexample` runs it in CI and fails unless TLC reports the violation. It has not been run on a developer machine yet |

Reservoir reports negative results. A pre-registered search for
decision-relevant divergence between float and exact sum-trees
([`docs/preregistration.md`](docs/preregistration.md)) has so far been run on
1% of its grid (5,400 of 270,000 workload programs) and found none, with
total-variation distances below 2^-40; the preregistered verdict needs the
full grid, which has not been run, and the committed run before 2026-10-04
deviated from the protocol ([`docs/preregistration-deviations.md`](docs/preregistration-deviations.md)).
Exactness is for reproducibility and verification, not for training quality.

What Reservoir does not claim is listed in
[`docs/nonclaims.md`](docs/nonclaims.md).

---

## Install

```bash
pip install reservoir-replay                 # rollout buffer, attestation, checker (no dependencies)
pip install "reservoir-replay[classic]"      # transition buffers, C extension, wrappers (numpy, torch)
pip install "reservoir-replay[trl]"          # TRL integration
pip install "reservoir-replay[prefcheck]"    # preference noise detector
pip install "reservoir-replay[anchor]"       # forgetting monitor
pip install "reservoir-replay[atari]"        # Atari benchmark suite

reservoir-verify run-01/attest.jsonl --manifest run-01/manifest.jsonl   # also: reservoir-transcript, reservoir-diff, reservoir-replay-offline
python -c "import reservoir; print(reservoir.backend)"                 # "c" or "python" (needs [classic])
```

The checker ships in the wheel as `reservoir_checker` with the four
console scripts above; `python -m checker.verify` and friends keep working
from a checkout. A C compiler is needed for the C extension; without one,
the classic buffer falls back to the numpy implementation.

---

## Development

```bash
uv run pytest tests/ -v                       # tests
uv run python -m campaigns.mutation           # forgery detection campaign
uv run python -m campaigns.crash              # crash atomicity campaign
uv run python -m campaigns.divergence         # float divergence campaign
uv run python -m demo.reproducible_grpo       # two runs, one transcript (make demo-repro)
uv run python -m benchmarks.attestation_overhead   # what attestation costs (make bench-attest)
bash spec/check.sh                            # TLA+ model check
make check                                    # tests + checker import isolation
```

---

## License

[Apache-2.0](LICENSE)
