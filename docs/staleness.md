# The staleness sweep

How far behind the current policy may a replayed rollout be before
replaying it stops helping, does a staleness policy recover what is lost,
and does replay beat the cheap baselines that also deal with dead groups?
This note describes the experiment, how to run and verify it, and what is
committed so far. Everything a run claims is in its committed record.

## Status

| Stage | Observed |
|---|---|
| Harness and report builder | `tests/test_staleness_sweep.py`, `test_staleness_arms.py`, `test_staleness_controls.py` and `test_staleness_stats.py` run the arm plan, the control trainers' pieces, the per-step diagnostics, the evaluation statistics, the report and its refusals on fabricated inputs |
| CPU smoke | committed: `benchmarks/modal/results/sweep_tiny_cpu_smoke/` (9 arms, the 8 of the main grid on seeds 42 and 43 and the stress arm on seed 42, 12 steps, 17 runs; all 11 Reservoir logs verify with their manifests, every draw declares `min_entry_age` 1, no replayed row came from its own step, nothing was declined or rescaled, so the `_on` arms trained exactly as the `_off` arms). Tiny Qwen2 test model, HF generation, coin-flip reward, so it shows the harness and the logs work and nothing about training. `tests/test_staleness_sweep_results.py` re-verifies every log and rebuilds the report |
| 16-step GPU engine smoke | not yet run |
| 0.5B sweep (8 arms × 3 seeds + 1 stress run × 300 steps) | not yet run |
| 1.5B sweep | not yet run; needs an A100-80GB (float32 master weights plus Adam do not fit a 24 GB A10G) |

The README's guarantees row for this experiment states exactly what the
most recent stage observed and is updated when a stage lands.

## Design

**Trainer.** TRL 1.13.0 `GRPOTrainer` with vLLM 0.28.0 in colocate mode
on one A10G, the configuration Martingale's runner uses (server mode hangs
in the NCCL weight handshake on Modal; `benchmarks/modal/nccl_probe.py`).
No batch invariance: vLLM's deterministic kernels have no backward and
would break the colocated trainer. The trainer keeps float32 master weights
and runs `bf16=True` autocast, so its forward matches the bfloat16 engine
it syncs to every step: Martingale's first colocate run showed that a pure
float32 trainer against a bfloat16 engine leaves the engine sampling from a
policy that moves only when an update crosses a bfloat16 step, which is a
staleness of its own on every arm. The CPU smoke runs float32 without
autocast. `vllm_importance_sampling_correction=False`, so the loss never
reads vLLM's sampling logprobs and the adapter drops the key
(`docs/nonclaims.md` §14); the stale-engine check below reads it instead.

**Task.** GSM8K train split (`openai/gsm8k`, `main`), conversational
prompts with a system line asking for step-by-step reasoning and a final
`#### <number>` line. One reward: 1.0 when the number after the last
marker (or, failing a marker, the last number in the completion) equals
the gold answer, else 0.0 (`benchmarks/staleness/gsm8k.py`).

**Batch.** 8 generations per prompt, micro-batch 8, 4 accumulation steps:
4 prompts, 32 completions, one generation per optimizer step
(`num_iterations=1`), up to 256 completion tokens, temperature 1.0,
learning rate 1e-6, no KL penalty (`beta=0`), TRL's default `dapo` loss
with group reward scaling, 300 optimizer steps, seeds 42, 43, 44. Every
arm draws its step-`t` prompts from TRL's seeded sampler, so all arms see
the same prompt stream (`grpo_filter`'s extra prompts come from a separate
stream; see below).

**Arms** (`benchmarks/staleness/arms.py`).

| Arm | Trainer | What it isolates |
|---|---|---|
| `grpo` | stock `GRPOTrainer` | the framework default: dead groups kept at zero advantage |
| `grpo_filter` | `GRPOTrainer` + DAPO dynamic sampling | dead groups dropped, fresh prompts generated until the batch holds 4 live groups (at most 4 generation rounds; a shortfall is filled with the first round's dead groups and logged). Extra prompts come from a seeded stream over the training set separate from the main one, so step `t`'s first round is the same prompts every arm sees. The strongest cheap baseline and the control for the step-size effect of replacement under the `dapo` loss; reported at matched effective batch (per step, like every arm) and at matched total rollouts (below) |
| `wreinforce` | `GRPOTrainer`, advantage only | fixed-reference advantage (Zhu et al. 2025): +0.1 for a correct completion, −1 for a wrong one, no group baseline, no buffer |
| `reservoir_age{8,32}_off` | `ReservoirGRPOTrainer` | cross-prompt replacement of dead rows, `max_policy_age` 8 or 32, policy off |
| `reservoir_age{8,32}_on` | `ReservoirGRPOTrainer` | the same with the policy on (`--policy-on`, default `staleness`) |
| `reservoir_naive_age32` | `ReservoirGRPOTrainer` | uniform priority and no age decay (half-life 1024, the library's maximum: a row at age 32 keeps 97.9% of its weight), policy off. The literature predicts this is worse than `grpo`; if it is not, the harness is not exercising replay |
| `reservoir_age128_on` | `ReservoirGRPOTrainer` | stress arm, first seed only: nothing in the literature trains well at that age |

Every Reservoir arm uses capacity 4096, `half_life = max_policy_age / 2`
(a row at the age bound has decayed to a quarter of its priority) except
the naive arm, `beta=0.4`, `min_entry_age=1` (a replayed row is never one
inserted at its own step), telemetry on, the run seed as the buffer seed,
an attestation log and a manifest. "Policy on" is selected by
`--policy-on`. `staleness` (the default) passes a `StalenessPolicy` with
the `conservative` preset's sampling-design ESS floor (0.5) and group-mass
cap (0.25) only: no policy age bound (the age axis stays the buffer's
`max_policy_age`) and no legacy gate. `gate` is the 0.6.0 drift gate,
`max_log_ratio=2.0` with uncapped declines; the TRL adapter now measures
that log-ratio as a per-token mean, so 2.0 is far looser than when it was
chosen and the gate rarely fires. On the CPU smoke the sampling-design ESS
fraction is 0.97 to 1.0 and nothing is declined or rescaled, so at lr 1e-6
the `_on` arms may train exactly like the `_off` arms; the per-step
`reservoir/declined_*` and `rescaled_rows` metrics show whether the policy
ever acted.

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
stale steps (`benchmarks/staleness/engine_probe.py`). The control arms
apply the same rule in their loss forward (whose weights are the ones the
step sampled with) to the fresh rows, logged as
`sweep/engine_confident_fraction`, at no extra forward either. A vLLM run
whose probe did not see sampling logprobs at every hook call is returned
with `engine_check.failure` set, so its log is kept. A run with a failed
probe or any stale step is marked `suspect` in the report, with its
reasons: it stays listed with its numbers but is left out of the per-arm
means and counted in the totals.

**Per-step diagnostics** (every arm; `benchmarks/staleness/diagnostics.py`,
logged in TRL's per-step log under `sweep/`):

- dead-group rate of the generated groups, split all-wrong and all-correct
  (`grpo_filter` counts every generation round), completions generated, and
  the mean reward of the step's first generation round, the prompts every
  arm sees. The report's train-reward series is this first-round reward:
  `grpo_filter`'s TRL `reward` also averages its extra rounds, which run
  only after dead-heavy first rounds; `grpo_filter` also logs its rounds and
  any groups left unfilled after the last round;
- fraction of trained rows that are replayed, fraction of replayed rows
  inserted at the same step (the run raises if it is ever non-zero), and
  replayed-row age mean, nearest-rank p90 and max;
- clip fraction of TRL's clipped objective, split fresh vs replayed rows and
  by advantage sign, measured on the loss's own forward. With
  `num_iterations=1` the control arms' fresh rows have no stored logprobs,
  so their ratio is exactly 1; a Reservoir arm attaches behavior logprobs to
  every row on the steps it replays, so there a fresh row's ratio is 1 only
  up to the agreement of two forwards. The fresh clip fraction is therefore
  not comparable across arms; on the GPU it checks that the adapter's
  forward runs under the same autocast as the loss (it should be about 0);
- KL to the initial model (k3 estimator on the fresh rows' tokens; the
  replayed tokens were not sampled from the current policy), from a frozen
  float32 copy of the initial model run under the same autocast;
- `lm_head` gradient norm before clipping and its z-score against the
  previous 20 steps (Qwen2.5 ties `lm_head` to the embedding, so the norm
  covers both uses of the matrix);
- and from TRL and the adapter: policy entropy, mean completion length,
  sampling-design ESS, replaced and declined rows, log-ratio statistics.

**Evaluation** (`benchmarks/staleness/evaluation.py`, `math500.py`): the
GSM8K test split, all 1,319 problems, greedy pass@1 at the end; avg@8 at
temperature 1.0 on the same problems every 50 steps; MATH500 (500
problems, up to 512 tokens, last `\boxed{}` answer) greedy at the end. On
the GPU the colocated vLLM engine answers after the trainer's weights are
pushed to it; on the CPU HF `generate` answers inside a forked RNG. Every
arm and seed answers the same items and per-item scores are kept. The
MATH500 check is exact match after a narrow LaTeX normalization (or exact
rational equality), stricter than Math-Verify, so it can only undercount,
equally for every arm.

**Report** (`benchmarks/staleness/report.py`, `stats.py`): per arm, mean ±
sample standard deviation over seeds of every summary field; a percentile
bootstrap CI (2,000 resamples, seeds and items resampled jointly) of the
final GSM8K and MATH500 accuracy, and of each arm's paired difference from
`grpo` on the seeds both ran; area under the train-reward curve (mean
reward per step); steps until the 10-step rolling train reward first
reaches 95% of `grpo`'s final reward (with the count of runs that got
there, and a censored mean that counts a run that never did at its full
length); and every run cut at the median non-suspect `grpo` run's total
generated completions (matched total rollouts: `grpo_filter` generates
more per step), with the step reached, the train reward and the mean
reward per step up to it, and the last avg@8 before it with its step. The
final greedy GSM8K and MATH500 evaluations are not rollout-matched: every
arm trains for 300 steps. With three seeds, resampling seeds makes the
bootstrap CI somewhat narrow (a mean of three resampled with replacement
has (k−1)/k of the variance); read it with the per-seed spread next to it.

## What every run writes

Under `benchmarks/modal/results/sweep_<model>_<steps>steps/`:

- `<arm>_seed<seed>.json`: the configuration and library versions, TRL's
  per-step log with the `reservoir/*` telemetry and the `sweep/*`
  diagnostics, per-step wall clock, peak GPU memory, the adapter's
  counters, every evaluation with its per-item scores, the stale-engine
  check, the container's wall clock and the dollars at Modal's list price
  (A10 $1.10/h, A100-80GB $2.50/h, 2026-10-06), and for Reservoir arms the
  log's record count, head digest and the checker's verdict.
- `<arm>_seed<seed>.attest.jsonl` and `.manifest.jsonl` for Reservoir
  arms: the hash-chained sampling transcript (every insert with its
  content digest, every draw with its exact probability, importance
  weight and entry version, every eviction, every batch witness, every
  telemetry record) and the opening of every digest.
  `python -m checker.verify <log> --manifest <manifest>` runs before the
  record is written; a rejected log is written for inspection and fails
  the sweep.
- `report.json` and `ess_vs_reward.png`, built from the records; the GPU
  sweep's report is also written to `results/staleness_sweep_report.json`.
- `<arm>_seed<seed>.failed.json` only when a returned record could not be
  written (the report never reads it). An evaluation that raises is stored
  as `{"error": ...}` in the record instead of losing the trained run.

Rebuilding the report (`--report-only`) runs the checker again on every log
and manifest next to the records, compares the log's last digest with the
head the run reported, and refuses: a Reservoir run without a log or with
a rejected one; a log whose sample records do not all declare
`min_entry_age` 1; a run that replayed a row from its own step; records
that differ in configuration (steps, batch, loss, reward, temperature,
precision, vLLM settings, evaluation sizes, library versions); an arm whose seeds differ in policy, gate, trainer, priority or
buffer settings; main arms that do not share one seed set; a stress arm
outside it. Arms that are absent are listed under `missing_arms`.
`tests/test_staleness_sweep_results.py` rebuilds every committed report and
compares it with the committed one. Dollars count the time inside the run
function at list price, so they are a lower bound on what Modal bills
(image pull and container start are not included).

## Running it

```bash
# CPU smoke, no Modal account: tiny model, HF generation, coin-flip reward (a few minutes)
python benchmarks/modal/staleness_sweep.py --local --max-steps 12 --seeds 42,43 \
    --out-dir benchmarks/modal/results/sweep_tiny_cpu_smoke

# the plan and the cost estimate, nothing launched (no Modal account needed)
python benchmarks/modal/staleness_sweep.py --dry-run
modal run benchmarks/modal/staleness_sweep.py --dry-run      # the same text, through Modal

# 16-step engine smoke on the GPU before the sweep proper
caffeinate -i modal run benchmarks/modal/staleness_sweep.py --max-steps 16 --seeds 42 \
    --arms grpo,grpo_filter,reservoir_age8_on,reservoir_age128_on --eval-every 8 \
    --out benchmarks/modal/results/sweep_0.5b_16steps_smoke

# the 0.5B sweep: 25 containers in parallel
caffeinate -i modal run benchmarks/modal/staleness_sweep.py

# rebuild the report and the figure from a results directory (run from the repo root)
python benchmarks/modal/staleness_sweep.py --report-only benchmarks/modal/results/sweep_0.5b_300steps \
    --report-path results/staleness_sweep_report.json --figure-path results/staleness_sweep_ess_vs_reward.png
```

The estimate uses `--seconds-per-step` (prior 13 s on an A10G: 12 s before
the KL reference forward; `grpo_filter` at 1.35×), 300 s of container
overhead per run, and `--seconds-per-eval-completion` (prior 0.02 s
through vLLM) times the completions one run's evaluations generate
(6 × 1,319 × 8 avg@8 + 1,319 + 500 = 65,131 at the defaults). At the
priors the 25 runs are 39.4 GPU-hours, about $43 (94 minutes per run on
average, of which about 22 are evaluation); the 16-step smoke above is
about $1.20, most of it its two full-size avg@8 evaluations. Replace both priors
with the 16-step smoke's measurements (its records carry per-step seconds,
peak memory and each evaluation's seconds). `--periodic-eval-size N`
evaluates avg@8 on the first N shuffled test items instead of all 1,319
and is the cheapest lever on cost. The 1.5B model (`--model 1.5b`) runs on
an A100-80GB at $2.50/h; expect roughly three times the per-step time.
Each job runs in its own container (`single_use_containers`), so no
trainer or colocated engine outlives its run. On the A10G the budget is
tight: about 7 GB for vLLM at 0.3 utilisation, 8 GB for the float32
weights, gradients and Adam state, 2 GB for the frozen float32 reference copy, and
the logits of 8 rows × 256 tokens × 152k vocabulary; if the 16-step smoke
shows peak memory near 22 GB, use micro-batch 4 with 8 accumulation steps
(`run_arm`'s `per_device_train_batch_size` and
`gradient_accumulation_steps`). Keep the laptop awake for the whole
`modal run` (`caffeinate -i`): the records are written locally as the runs
return.

**Before the sweep, from the 16-step smoke:** the probe is `available` with
no skipped steps and no failure, its per-step mean absolute engine-vs-trainer
difference is at the bf16 floor (about 0.03 nats) in the Reservoir arms and
`sweep/engine_mean_abs_diff` is too in the controls, there are no stale
steps, the fresh-row clip fraction of the Reservoir arms is about 0 on
replayed steps, every evaluation finished without an `error` (MATH500's
long prompts included: the engine's `max_model_len` is 1536), and peak
memory leaves headroom. Then replace the two priors and re-run the dry run.

## Verifying a run

```bash
python -m checker.verify benchmarks/modal/results/sweep_0.5b_300steps/reservoir_age32_on_seed42.attest.jsonl \
    --manifest benchmarks/modal/results/sweep_0.5b_300steps/reservoir_age32_on_seed42.manifest.jsonl
python -m checker.transcript <log> --manifest <manifest>      # exposure per example, declines per step
```

The checker recomputes every draw, importance weight, decayed priority,
entry age against the declared floor, batch witness and the exact
sampling-design ESS and staleness of every telemetry record, recomputes the
log-ratio statistics from the per-draw log-ratios the record carries, and
replays every staleness-policy decision. The log-ratios themselves are
measurements it cannot redo (the checker has no model), and nothing in the
log covers the control arms or the `sweep/*` diagnostics, which are
reported values. See `docs/reproducible-training.md` for what the
transcript does and does not establish.

## What this does not claim

- Nothing about training quality until the GPU sweep has run; the CPU
  smoke only shows that the harness and the logs work.
- The sweep is powered for mechanism questions (replayed fraction, dead
  groups, age, clipping, acceleration), not for endpoint accuracy: the
  literature puts replay's endpoint effect at 0.5B within the 1 to 2 point
  seed-to-seed spread.
- The drift gate is a bias-for-variance trade, not an unbiased estimator;
  Martingale's exact bench shows per-token clipping is biased, and the
  sequence-level gate declines rows rather than reweighting them. The
  sweep measures the trade; it does not justify it.
- The stale-engine check can show that the engine disagreed with the
  trainer; it cannot say which weights the engine held.
- `grpo_filter` and `wreinforce` are reimplementations inside TRL, not the
  DAPO or W-REINFORCE authors' code.
- One model family, one task, one reward, 300 steps. Nothing is said
  about other models, longer runs, asynchronous generation or `num_iterations > 1`.
- Dollars are Modal list prices at the time of the run, not what was billed.
