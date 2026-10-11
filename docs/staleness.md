# The staleness sweep

How far behind the current policy may a replayed rollout be before
replaying it stops helping, and does a staleness policy recover what is
lost? This note describes the experiment, how to run and verify it, and
what is committed so far. Everything a run claims is in its committed
record; no run record is committed yet, so this note makes no claim about
any run.

## Status

| Stage | Observed |
|---|---|
| Harness and report builder | `tests/test_staleness_sweep.py` runs the arm planning, the policy switch, the per-run summaries, the report and its refusals on fabricated records. A CPU smoke of the harness (tiny Qwen2 test model, HF generation, coin-flip reward) was run on 2026-10-06; its records were not committed, so the repository holds no evidence of it and `tests/test_staleness_sweep_results.py` skips |
| 16-step GPU engine smoke | not yet run |
| 0.5B sweep (7 arms × 3 seeds × 300 steps) | not yet run |
| 1.5B sweep | not yet run; needs an A100-80GB (float32 master weights plus Adam do not fit a 24 GB A10G) |

The README's guarantees row for this experiment states exactly what the
most recent stage observed and is updated when a stage lands.

## Design

**Trainer.** TRL 1.13.0 `GRPOTrainer` with vLLM 0.28.0 in colocate mode
on one A10G, the configuration Martingale's runner uses (server mode hangs
in the NCCL weight handshake on Modal; `benchmarks/modal/nccl_probe.py`).
No batch invariance: vLLM's deterministic kernels have no backward and
would break the colocated trainer. The trainer holds the model in float32;
vLLM serves bfloat16 and takes the synced weights each step.
`vllm_importance_sampling_correction=False`, so the loss never reads
vLLM's sampling logprobs and the adapter drops the key
(`docs/nonclaims.md` §14); the stale-engine check below reads it instead.

**Task.** GSM8K train split (`openai/gsm8k`, `main`), conversational
prompts with a system line asking for step-by-step reasoning and a final
`#### <number>` line. One reward: 1.0 when the number after the last
marker (or, failing a marker, the last number in the completion) equals
the gold answer, else 0.0 (`benchmarks/staleness/gsm8k.py`). At the end
of every run the trained model answers `--eval-size` (default 200) test
problems greedily with HF `generate`, scored by the same function.

**Batch.** 8 generations per prompt, micro-batch 8, 4 accumulation steps:
4 prompts, 32 completions, one generation per optimizer step
(`num_iterations=1`), up to 256 completion tokens, temperature 1.0,
learning rate 1e-6, no KL (`beta=0`), TRL's default `dapo` loss with group
reward scaling, 300 optimizer steps, seeds 42, 43, 44.

**Arms.**

| Arm | Trainer | `max_policy_age` | Policy |
|---|---|---|---|
| `grpo` | stock `GRPOTrainer` | – | – |
| `reservoir_age{8,32,128}_off` | `ReservoirGRPOTrainer` | 8, 32, 128 | off |
| `reservoir_age{8,32,128}_on` | `ReservoirGRPOTrainer` | 8, 32, 128 | on |

Every Reservoir arm uses capacity 4096, `half_life = max_policy_age / 2`
(a row at the age bound has decayed to a quarter of its priority),
`beta=0.4`, telemetry on, the run seed as the buffer seed, an attestation
log and a manifest. "Policy on" is selected by `--policy-on` through one
function, `benchmarks.staleness.report.replay_kwargs`. `staleness` passes
a `StalenessPolicy` with the `conservative` preset's sampling-design ESS
floor (0.5) and group-mass cap (0.25) only: no policy age bound (the age
axis stays the buffer's `max_policy_age`) and no legacy gate. `gate`
(the harness default) is the 0.6.0 drift gate, `max_log_ratio=2.0` with
uncapped declines: a replayed row whose absolute log-ratio exceeds 2 is
declined, evicted with reason `drift`, and listed in the step's batch
witness. The TRL adapter now measures that log-ratio as the mean over
completion tokens of current minus stored logprob; when 2.0 was chosen it
was the sum, so the gate arms are much looser than they were designed to
be. Pass `--policy-on staleness` for the policy the library ships.

**Stale-engine check.** Martingale's first colocate run caught TRL's
in-process vLLM serving the initial weights for some generation steps
while the trainer trained. A sweep over staleness cannot trust its axis
if the engine is silently stale, so every Reservoir arm compares, at
each step, vLLM's sampling logprobs with the trainer's own forward over
the same tokens, which `ReservoirReplay` already runs to obtain behavior
logprobs: zero extra cost. The rule is Martingale's confident
disagreement (engine probability at least 0.5, trainer more than 2 nats
lower, which bf16-vs-fp32 numerics cannot produce); a step with more than
1% such tokens is `stale`, and the run record and the report list the
stale steps (`benchmarks/staleness/engine_probe.py`). A vLLM run whose
probe did not see sampling logprobs at every hook call fails instead of
returning a record, and a run with any stale step is marked `suspect` in
the report: it stays listed with its numbers but is left out of the
per-arm means and counted in the totals. The plain arm is not probed: that
would add a forward pass the stock trainer does not run.

## What every run writes

Under `benchmarks/modal/results/sweep_<model>_<steps>steps/`:

- `<arm>_seed<seed>.json`: the configuration and library versions, TRL's
  per-step log with the `reservoir/*` telemetry (replay fraction, dead and
  near-dead groups, replaced and declined rows, the exact sampling-design
  ESS and its fraction of the replayed rows, staleness, log-ratio
  statistics),
  per-step wall clock, the adapter's counters, the greedy evaluation, the
  stale-engine check, the container's wall clock and the dollars at
  Modal's list price (A10 $1.10/h, A100-80GB $2.50/h, 2026-10-06), and for
  Reservoir arms the log's record count, head digest and the checker's
  verdict.
- `<arm>_seed<seed>.attest.jsonl` and `.manifest.jsonl` for Reservoir
  arms: the hash-chained sampling transcript (every insert with its
  content digest, every draw with its exact probability and importance
  weight, every eviction, every batch witness, every telemetry record)
  and the opening of every digest. `python -m checker.verify <log>
  --manifest <manifest>` runs before the record is written; a rejected log
  is written for inspection and fails the sweep.

`results/staleness_sweep_report.json` aggregates the records per arm
(mean and sample standard deviation over seeds of the held-out accuracy,
the last-quarter train reward, the dead-group rate, replaced and declined
rows, the mean sampling-design ESS fraction and log-ratio, wall clock and dollars, stale
engine steps) and keeps every per-step series;
`results/staleness_sweep_ess_vs_reward.png` plots the reward curves per
arm and sampling-design ESS fraction against reward at every replayed step. Rebuilding the
report (`--report-only`) runs the checker again on every log and manifest
next to the records and compares the log's last digest with the head the
run reported, refuses a Reservoir run without a log or with a rejected
one, and refuses records that do not share one configuration and one seed
set per arm; arms that are absent are listed under `missing_arms`.
`tests/test_staleness_sweep_results.py` rebuilds the report from the
committed records and compares it with the committed one. Dollars count
the time inside the run function at list price, so they are a lower bound
on what Modal bills (image pull and container start are not included).

## Running it

```bash
# CPU smoke, no Modal account: tiny model, HF generation, coin-flip reward (seconds)
python benchmarks/modal/staleness_sweep.py --local --max-steps 12 --seeds 42,43

# the plan and the cost estimate, nothing launched
modal run benchmarks/modal/staleness_sweep.py --dry-run

# 16-step engine smoke on the GPU before the sweep proper (about $0.50)
modal run benchmarks/modal/staleness_sweep.py --max-steps 16 --seeds 42 \
    --arms grpo,reservoir_age8_on,reservoir_age128_on --out benchmarks/modal/results/sweep_0.5b_16steps_smoke

# the 0.5B sweep: 21 containers in parallel
modal run benchmarks/modal/staleness_sweep.py

# rebuild the report and the figure from a results directory
python benchmarks/modal/staleness_sweep.py --report-only benchmarks/modal/results/sweep_0.5b_300steps
```

The estimate printed by `--dry-run` uses `--seconds-per-step` (prior
12 s on an A10G, to be replaced by the 16-step smoke's measurement) plus
300 s of container overhead per run: at the prior, 21 runs are 22.8
GPU-hours, about $25. The 1.5B model (`--model 1.5b`) runs on an
A100-80GB at $2.50/h; expect roughly three times the per-step time. Each
job runs in its own container (`single_use_containers`), so no trainer or
colocated engine outlives its run. On the A10G the budget is tight: about
7 GB for vLLM at 0.3 utilisation, 8 GB for the float32 weights, gradients
and Adam state, and the float32 logits of 8 rows × 256 tokens × 152k
vocabulary; if the 16-step smoke shows peak memory near 22 GB, use
micro-batch 4 with 8 accumulation steps (`run_arm`'s
`per_device_train_batch_size` and `gradient_accumulation_steps`).

## Verifying a run

```bash
python -m checker.verify benchmarks/modal/results/sweep_0.5b_300steps/reservoir_age32_on_seed42.attest.jsonl \
    --manifest benchmarks/modal/results/sweep_0.5b_300steps/reservoir_age32_on_seed42.manifest.jsonl
python -m checker.transcript <log> --manifest <manifest>      # exposure per example, declines per step
```

The checker recomputes every draw, importance weight, decayed priority,
batch witness and the exact sampling-design ESS and staleness of every
telemetry record, recomputes the log-ratio statistics from the per-draw
log-ratios the record carries, and replays every staleness-policy
decision. The log-ratios themselves are measurements it cannot redo (the
checker has no model). See `docs/reproducible-training.md` for what the transcript
does and does not establish.

## What this does not claim

- Nothing about training quality until the GPU sweep has run; the CPU
  smoke only shows that the harness and the logs work.
- The drift gate is a bias-for-variance trade, not an unbiased estimator;
  Martingale's exact bench shows per-token clipping is biased, and the
  sequence-level gate declines rows rather than reweighting them. The
  sweep measures the trade; it does not justify it.
- The stale-engine check can show that the engine disagreed with the
  trainer; it cannot say which weights the engine held.
- One model family, one task, one reward, 300 steps. Nothing is said
  about other models, longer runs, asynchronous generation or `num_iterations > 1`.
- Dollars are Modal list prices at the time of the run, not what was billed.
