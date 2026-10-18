# reservoir

**Exact, reproducible, auditable replay for LLM reinforcement learning.**

```diff
-from trl import GRPOTrainer
+from reservoir.integrations.trl import ReservoirGRPOTrainer, ReservoirReplay
-trainer = GRPOTrainer(model=model, args=args, train_dataset=dataset, reward_funcs=reward_funcs)
+trainer = ReservoirGRPOTrainer(model=model, args=args, train_dataset=dataset, reward_funcs=reward_funcs, replay_buffer=ReservoirReplay())
```

Install with `pip install "reservoir-replay[trl]"`. The trainer writes an
attestation log and manifest to `args.output_dir/reservoir/`; after training,
run the `reservoir-verify` command printed in its summary.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

> **Status:** the rollout buffer, priority strategies, age decay, attestation
> with content commitments, the independent checker with its transcript and
> diff tools, the durable buffer, the TRL integration and the reproducibility
> demo below are implemented and tested, as is the verl integration for
> verl's `DataProto` trainer. The Stable-Baselines3 / TorchRL adapters are
> not yet.

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
pip install "reservoir-replay[trl]"
reservoir doctor
```

The distribution is `reservoir-replay` (the name `reservoir` on PyPI belongs
to an unrelated package); the import name is `reservoir`. The rollout buffer,
the attestation log and the checker are pure Python with no required
dependencies; numpy and torch are pulled in by the extras that use them.

---

## Quick start — replay for GRPO

The two changed lines above are enough for a stock TRL script. For a
schedule-based starting point, choose a preset and inspect its choices:

```python
replay, choices = ReservoirReplay.for_grpo(
    num_generations=args.num_generations,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.per_device_train_batch_size,
)
print(choices)
trainer = ReservoirGRPOTrainer(
    model=model, args=args, train_dataset=dataset, reward_funcs=reward_funcs,
    replay_buffer=replay,
)
trainer.train()
```

`profile="conservative"` applies a strict drift gate, `"async"` allows a
wider drift window, and `"off"` disables the gate. The preset returns the
replay object and a dictionary explaining capacity, half-life, age and gate
choices. The raw constructor remains available for exact control.

For a custom generation loop, use the underlying buffer directly:

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

Rescoring is a hook, not a recommendation: `DriftAwarePriority` ships as a
reference implementation of it (the base score divided by `1 + scale *
|log_ratio|` when the adapter measured a log-ratio), documented as an example
and not as something that improves training, and every rescoring is an
`update` record in the log.

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
# run a: seed=42 buffer_seed=0 records=116 replaced_rows=8 head=78f14c388f625c71…
# run b: seed=42 buffer_seed=0 records=116 replaced_rows=8 head=78f14c388f625c71…
# run c: seed=43 buffer_seed=0 records=116 replaced_rows=8 head=0a705cdd155159a2…
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
`benchmarks/modal/results/repro_hf_t4_*` (verdict IDENTICAL; the two
variants' logs are byte-identical to each other, so this is one
observation). The
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
Apple silicon, the 0.6.0 working tree, medians of 5 repeats on a machine
running other work) an `add_group` took 5.1 ms in the first tenth of the run and 5.5 ms in the last, against 0.4 and 0.8 ms in memory
(each figure's range over the repeats is in the file). The snapshot does grow,
because the attestation log is part of the state: compaction took 0.4 s at the median and 0.6 s by the end of that run (1.5 ms per operation amortised at the median, 2.2 ms at the end), and reopening replayed in 5.3 s.
`buf.checkpoint("step-100")` and `buf.restore_checkpoint("step-100")` bind the
buffer to a training checkpoint; the TRL adapter does this on every save
(pruning buffer checkpoints to the ones the trainer kept) and rewinds on
resume, so a restarted run continues the chain from the checkpoint rather
than from wherever the crash happened. Each buffer checkpoint records the
digest of the trainer's checkpoint directory it was saved with, and a resume
against a directory that digests differently is refused. Resume at a non-zero
step without a matching buffer checkpoint is an error, not a fresh buffer.

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
row the example (or, with the manifest, its prompt) reached. The manifest carries
each example's per-reward-function values, numeric only and outside the digest; the
TRL adapter takes them from the trainer's own reward computation (the verl adapter
records the summed score only). The committed GPU manifests predate this wiring. Replayed rows carry their
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

```bash
pip install "reservoir-replay[verl]" "verl[vllm]==0.9.1"   # verl 0.9.1; a rollout engine from verl's own extras
```

```python
from reservoir.integrations.verl import ReservoirRayPPOTrainer, ReservoirReplay

trainer = ReservoirRayPPOTrainer(
    config=config, tokenizer=tokenizer, role_worker_mapping=..., resource_pool_manager=...,
    train_dataset=..., val_dataset=..., collate_fn=..., train_sampler=...,
    replay_buffer=ReservoirReplay(capacity=50_000, half_life=4, max_policy_age=16, seed=0,
                                  attest="run-01/attest.jsonl", manifest="run-01/manifest.jsonl",
                                  directory="run-01/buffer", source="gsm8k"),
)
trainer.init_workers()
trainer.fit()
```

`ReservoirRayPPOTrainer` is verl's `RayPPOTrainer` plus one override: the
batch `fit` hands to `_update_actor` goes through the same steps as the TRL
adapter (store the responses of prompts whose rewards varied, fill the rows
of prompts whose rewards were all equal with replayed rollouts, check the
written rows, write the batch witness and the telemetry, apply the optional
drift gate), with the replay health logged under `reservoir/*` next to
verl's `actor/*` metrics. Groups are found by verl's `uid`, so
`balance_batch` reordering is fine. The buffer is bound to verl's
`global_step_N` checkpoints: a save snapshots it, a resume rewinds it.
`benchmarks/modal/verl_replay_real.py` runs it on one T4 with verl 0.9.1,
vLLM 0.24.0 and a 135M model; the committed record
(`benchmarks/modal/results/verl_replay_smollm2-135m-instruct_12steps_seed42.*`)
shows 14 dead groups replaced with 56 rows over 12 steps, buffer snapshots at
every trainer checkpoint, and a log the checker accepts with its manifest.

Scope: verl 0.9.1's `DataProto` trainer (`trainer.use_v1=false`), which verl
marks deprecated in favour of its TransferQueue-based V1 loop; `algorithm.adv_estimator=grpo`;
text models; single-turn responses; one GPU verified. The adapter refuses by
name a critic, rollout correction, KL-in-reward, multimodal inputs and any
per-row tensor it does not know how to replace, and fails with a clear
message if the installed verl lacks the members it relies on. See
[`docs/design.md`](docs/design.md) §12 and
[`docs/nonclaims.md`](docs/nonclaims.md) §24.

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

## Guarantees

| Claim | Evidence |
|-------|----------|
| Sampling is deterministic and reproducible under a keyed draw | Property tests against a brute-force reference; the log records seed, buffer id and `beta`, and the checker recomputes every draw and importance weight from them |

| The durable buffers are failure-atomic under SIGKILL | 330/330 crash tests on macOS/APFS across both buffers (`results/crash_campaign_report.json`): every child was killed at its armed cut point (the campaign fails a row otherwise), each recovered either the pre-state (250) or the post-state (80) (an unsynced write that SIGKILL leaves in the page cache recovers to the post-state; a power loss there would give the pre-state, which is also legal), zero torn; the rollout cases cross the command log and the snapshot protocol in one operation, crash mid-rebase, crash inside a quarantine of a whole group, crash inside a checkpoint restore, and crash in the two windows of the single-write snapshot install. SIGKILL only: no power loss, and no claim for network or FUSE-backed filesystems (`docs/nonclaims.md` §6) |
| The independent checker rejects forged logs | 279/279 mutants rejected in 18 categories (`results/mutation_campaign_report.json`): chain, draw-arithmetic and record-order forgeries; age-decay protocol forgeries; content commitments with and without the manifest; draws and weights; batch witnesses; telemetry; quarantine and reward provenance; manifest tampering against the offline replay; staleness-policy decisions; malformed fields (`schema`, 13); declared minimum entry ages (`entry_age`, 7). CI runs the campaign on every push to `main` and `release/**` and on pull requests. The campaign checks that each mutant is rejected, not the reason. It also measures what the checker is not expected to catch: 3 chain-consistent changes to a digest or source pass without the manifest and are rejected with an unedited one; 2 draw forgeries pass in a log that does not record its seed and `beta`; 4 changes to quarantine texts and reward values pass, because the log does not commit to them; 3 manifest edits of reward values change only those fields of the offline replay; 2 staleness-policy parameter changes that flip no decision pass. A log re-chained after editing, or a log and manifest edited together, is consistency evidence only (`docs/nonclaims.md` §4) |

| A witnessed batch can be rebuilt as content from the log and manifest alone | `reservoir-replay-offline` emits every witnessed batch as JSON lines (rows, draws, exact importance weights, tokens, rewards, sources) importing nothing from the library; on the committed CPU and T4 reproducibility runs, runs a and b replay byte-identically and run c differs; fresh rows are listed by index only (`docs/nonclaims.md` §23) |
| Two runs with the same inputs give one transcript | CPU demo: runs a and b byte-identical (116 records, one head digest), run c differs at record 1, classified `data` (`results/reproducible_grpo_report.json`). One T4 run with HF generation: a and b byte-identical, c differs as `data`, verdict `IDENTICAL`; the report records `cuda:0` and torch 2.14.0+cu130, not the GPU model or driver. The triplet was run twice, with PyTorch defaults and with deterministic kernels, and the two variants' logs and manifests are byte-identical to each other, so they are one observation of a reproducible configuration, not two, and say nothing about whether the deterministic settings were needed (`benchmarks/modal/results/repro_hf_t4_*`) |
| The adapters survive a real trainer's call path | TRL 1.13.0 on two A10G GPUs (the driver reports them as NVIDIA A10) under `accelerate launch` (40 steps, 14 dead groups replaced with 56 rows, 1019 records, manifest opens all 584 examples) and verl 0.9.1 on one Tesla T4 (12 steps, 14 dead groups replaced with 56 rows, 445 records, manifest opens all 328 examples, a buffer checkpoint at each of the 3 trainer checkpoints); the test suite re-verifies both logs, the verl log with its manifest and the TRL log without it (`tests/test_trl_results.py`, `tests/test_verl_results.py`); one run each, tiny models, no training-quality claim (`docs/nonclaims.md` §13) |
| The staleness sweep harness runs every arm and verifies its logs | CPU smoke committed in `benchmarks/modal/results/sweep_tiny_cpu_smoke/` (tiny Qwen2 test model, HF generation, coin-flip reward): 9 arms (the 8 of the main grid on seeds 42 and 43, the `reservoir_age128_on` stress arm on seed 42), 12 steps, 17 runs; the 11 Reservoir logs verify with their manifests, every draw declares `min_entry_age` 1, no replayed row came from its own step, and `tests/test_staleness_sweep_results.py` re-verifies the logs and rebuilds the report from the records. The GPU sweep (Qwen2.5-0.5B-Instruct, GSM8K, 300 steps × 8 arms × 3 seeds plus the stress run on one A10G with colocated vLLM) has not been run. The smoke shows that the harness and the logs work; it makes no training-quality claim (`docs/staleness.md`) |
| What attestation and checking cost, on one machine | `results/attestation_overhead.json` (0.6.0 working tree; Apple silicon, Python 3.12, load average about 9; medians of 5 interleaved repeats after a warm-up, collector paused, ranges in the file): `add_group` costs 4.2× the unattested insert with an in-memory log, 4.1× with a log file and 8.8× with a manifest as well; in steady state a draw costs 14.9 µs unattested and 15.1 to 15.8 µs attested. An unattested buffer computes a group's content digests on the first sample that touches it rather than on insert, so over the first 200 batches its draws average 37.0 µs. The checker verifies 10k records with the manifest in 0.44 s. Pure-Python buffer; generation time is not measured |
| The snapshot protocol is safe within a finite scope | `spec/ReplayLifecycle.tla` models the seven-stage intent, segment and rename snapshot protocol only: no command log, epochs, compaction, checkpoints, restore or attestation files. CI runs TLC on it (`bash spec/check.sh`, `.github/workflows/ci.yml`); no TLC output or state count is committed, and TLC has not been run on a developer machine. `spec/NoParentFsync.tla` reaches its violation through an action that sets the state to a torn one directly; its recovery action is the correct one, so it shows that the invariant detects a torn state, not that omitting the directory fsync produces one |

Reservoir reports negative results. A pre-registered search for
decision-relevant divergence between float and exact sum-trees
([`docs/preregistration.md`](docs/preregistration.md)) has so far been run on
2% of its grid (5,400 of 270,000 workload programs) and found none, with
total-variation distances below 2^-40; the preregistered verdict needs the
full grid, which has not been run. The committed run before 2026-10-04
deviated from the protocol, the current code still departs from it in two
disclosed ways, and the grid as designed expects about 5.4e-05
divergences even if run in full, so it cannot be expected to find one
([`docs/preregistration-deviations.md`](docs/preregistration-deviations.md)).
Exactness is for reproducibility and verification, not for training quality.

What Reservoir does not claim is listed in
[`docs/nonclaims.md`](docs/nonclaims.md).

---

## Install

```bash
pip install "reservoir-replay[trl]"
reservoir doctor
```

For verl use `pip install "reservoir-replay[verl]"` (Python < 3.13). The
dependency-free rollout buffer and checker use `pip install reservoir-replay`.
For classic transition replay use `[classic]`; the Atari benchmark uses
`[atari]`.

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
