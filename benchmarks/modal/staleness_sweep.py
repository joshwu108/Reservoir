"""benchmarks/modal/staleness_sweep.py

The staleness sweep: how far behind the policy may a replayed rollout be before replaying it
hurts, does the staleness policy recover the loss, and does replay beat the cheap baselines?
``docs/staleness.md`` is the full description.

Setup
-----
- TRL 1.13.0 GRPO with vLLM 0.28.0 in **colocate** mode on one A10G, as
  Martingale's runner does (server mode hangs in the NCCL weight handshake
  on Modal; ``benchmarks/modal/nccl_probe.py``). No batch invariance: its
  kernel override has no backward and would break the colocated trainer.
- ``Qwen/Qwen2.5-0.5B-Instruct``, float32 master weights under ``bf16=True``
  autocast against the bfloat16 engine; ``--model 1.5b`` selects the 1.5B
  on an A100-80GB.
- GSM8K train split, conversational prompts, one exact-match reward
  (``benchmarks/staleness/gsm8k.py``).
- 8 generations per prompt, 4 prompts per optimizer step (micro-batch 8,
  4 accumulation steps), 300 steps, 3 seeds.

Arms (``benchmarks/staleness/arms.py``)::

    grpo                       plain GRPOTrainer
    grpo_filter                DAPO dynamic sampling (control_arms.DynamicSamplingMixin)
    wreinforce                 fixed-reference advantage +0.1 / -1 (control_arms.WReinforceMixin)
    reservoir_age{8,32}_off    ReservoirGRPOTrainer, max_policy_age=N, no policy
    reservoir_age{8,32}_on     the same with the policy on (--policy-on, default staleness)
    reservoir_naive_age32      uniform priority, no age decay, policy off
    reservoir_age128_on        stress arm, first seed only

Every arm's trainer starts with ``diagnostics.SweepDiagnosticsMixin`` (per-step
``sweep/*`` metrics). Evaluation: GSM8K test greedy pass@1 and MATH500 at the
end, avg@8 at temperature 1.0 every 50 steps (``benchmarks/staleness/evaluation.py``).

What every run writes
---------------------
``<arm>_seed<seed>.json`` (config, versions, TRL's per-step log with the
``reservoir/*`` telemetry and ``sweep/*`` diagnostics, per-step wall clock,
peak memory, every evaluation with per-item scores, the stale-engine check
and the dollars at Modal's list price), and for the Reservoir arms
``.attest.jsonl`` and ``.manifest.jsonl``, verified by ``python -m
checker.verify`` before the record is written (a rejected log is written for
inspection and the sweep fails). ``report.json`` and ``ess_vs_reward.png``
next to the records (and, for the GPU sweep, ``results/``) are built by
``write_report``.

Usage
-----
    # CPU smoke, no Modal account: tiny model, HF generation, coin-flip reward
    python benchmarks/modal/staleness_sweep.py --local --max-steps 12 --seeds 42,43
    # the plan and the cost estimate, nothing launched (also: modal run ... --dry-run)
    python benchmarks/modal/staleness_sweep.py --dry-run
    # 16-step engine smoke on the GPU before the sweep proper
    modal run benchmarks/modal/staleness_sweep.py --max-steps 16 --seeds 42 \
        --arms grpo,grpo_filter,reservoir_age8_on,reservoir_age128_on --eval-every 8
    # the sweep
    modal run benchmarks/modal/staleness_sweep.py
    # rebuild the report and figure from a results directory
    python benchmarks/modal/staleness_sweep.py --report-only benchmarks/modal/results/sweep_0.5b_300steps
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import modal


def _add_repo_to_path() -> None:
    """Run as a script, the checkout root is not on ``sys.path``; in the container it is ``/root``."""
    here = Path(__file__).resolve()
    root = here.parents[2] if len(here.parents) > 2 else Path("/root")
    if (root / "benchmarks").is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


_add_repo_to_path()

from benchmarks.modal.trl_replay_real import RESULTS_DIR, hf_cache, repo_root, with_sources, write_results
from benchmarks.staleness.arms import (
    DEFAULT_MAX_LOG_RATIO,
    ArmSpec,
    half_life_for_spec,
    plan_jobs,
    priority_kwargs,
    replay_kwargs,
)
from benchmarks.staleness.report import (
    GRPO_FILTER_STEP_FACTOR,
    build_report,
    estimate_jobs_cost,
    format_table,
    plot_report,
)

VLLM_VERSION = os.environ.get("RESERVOIR_VLLM_VERSION", "0.28.0")
_REPO = repo_root()
_repo_src = str(_REPO / "src") if (_REPO / "src").is_dir() else "/reservoir_src"
REPORT_PATH = _REPO / "results" / "staleness_sweep_report.json"
FIGURE_PATH = _REPO / "results" / "staleness_sweep_ess_vs_reward.png"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    gpu: Optional[str]
    price_per_hour: float   # Modal list price, 2026-10-06 (modal.com/pricing): A10 $0.000306/s, A100-80GB $0.000694/s


MODELS: dict[str, ModelSpec] = {
    "tiny": ModelSpec("tiny", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", None, 0.0),
    "0.5b": ModelSpec("0.5b", "Qwen/Qwen2.5-0.5B-Instruct", "A10G", 1.10),
    "1.5b": ModelSpec("1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "A100-80GB", 2.50),
}
DEFAULT_SEEDS = (42, 43, 44)
# Prior for the dry-run estimate until the 16-step smoke measures it: ~32 completions of up to 256
# tokens from vLLM plus four fp32 micro-batches and one behavior-logprob forward per step on an A10G.
SECONDS_PER_STEP_PRIOR = 13.0       # 12 s before the KL-to-init reference forward every arm now runs
SECONDS_PER_EVAL_COMPLETION_PRIOR = 0.02   # vLLM, ~200 tokens per GSM8K completion at ~10k tokens/s on an A10G
CONTAINER_OVERHEAD_SECONDS = 300.0   # image start, model download from the volume, vLLM engine start
GSM8K_TEST_SIZE = 1319
MATH500_SIZE = 500
EVAL_SEED = 0                        # the held-out subset is the same for every seed and arm

vllm_image = with_sources(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}")
    .pip_install("trl==1.13.0", "transformers==5.17.0", "datasets==5.0.1", "accelerate==1.15.0", "numpy==2.2.6")
    .env({"HF_HOME": "/hf_cache"})
)
app = modal.App("reservoir-staleness-sweep")


# ---------------------------------------------------------------------------
# One arm, one seed (plain Python; CPU or GPU)
# ---------------------------------------------------------------------------

def trainer_class_for(spec: ArmSpec) -> type:
    """The arm's trainer: ``SweepDiagnosticsMixin`` first, then the arm's change, then TRL's ``GRPOTrainer``."""
    from trl import GRPOTrainer

    from benchmarks.staleness.control_arms import DynamicSamplingMixin, WReinforceMixin
    from benchmarks.staleness.diagnostics import SweepDiagnosticsMixin

    if spec.trainer == "reservoir":
        from reservoir.integrations.trl import ReservoirGRPOTrainer

        bases: tuple = (ReservoirGRPOTrainer,)
    elif spec.trainer == "grpo_filter":
        bases = (DynamicSamplingMixin, GRPOTrainer)
    elif spec.trainer == "wreinforce":
        bases = (WReinforceMixin, GRPOTrainer)
    elif spec.trainer == "grpo":
        bases = (GRPOTrainer,)
    else:
        raise ValueError(f"unknown trainer {spec.trainer!r}")
    return type(f"Sweep{spec.trainer.title().replace('_', '')}Trainer", (SweepDiagnosticsMixin, *bases), {})


def _frozen_copy(model_id: str, like: Any) -> Any:
    """The initial model, frozen, in the trainer's dtype and on its device (the KL-to-init reference)."""
    from transformers import AutoModelForCausalLM

    first = next(like.parameters())
    ref = AutoModelForCausalLM.from_pretrained(model_id, dtype=first.dtype).to(first.device)
    ref.eval().requires_grad_(False)
    return ref


def _final_evals(trainer: Any, datasets: dict, reward_fn: Any, *, max_new_tokens: int, math_max_new_tokens: int,
                 batch_size: int) -> dict:
    from benchmarks.staleness import math500
    from benchmarks.staleness.evaluation import evaluate, guarded

    out = {}
    for name, dataset in datasets.items():
        if dataset is None:
            continue
        is_math = name == "math500"
        score = math500.math_reward if is_math and reward_fn.__name__ == "gsm8k_reward" else reward_fn
        out[name] = guarded(lambda: evaluate(
            trainer, dataset, score, n=1, temperature=0.0,
            max_new_tokens=math_max_new_tokens if is_math else max_new_tokens, seed=EVAL_SEED,
            batch_size=batch_size, step=int(trainer.state.global_step)), step=int(trainer.state.global_step))
    return out


def _eval_set(build: Any, size: Optional[int]) -> Any:
    """``size`` None: the whole split; 0: skipped."""
    return None if size == 0 else build(size)


def run_arm(
    spec: ArmSpec,
    *,
    seed: int,
    max_steps: int,
    model_key: str = "0.5b",
    reward: str = "gsm8k",
    use_cpu: bool = False,
    work_dir: str = "/tmp/reservoir_sweep",
    eval_size: Optional[int] = None,
    periodic_eval_size: Optional[int] = None,
    math500_size: Optional[int] = None,
    eval_every: int = 50,
    eval_samples: int = 8,
    math_max_new_tokens: int = 512,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    num_generations: int = 8,
    max_completion_length: int = 256,
    learning_rate: float = 1e-6,
    capacity: int = 4096,
    train_size: Optional[int] = None,
    vllm_gpu_memory_utilization: float = 0.3,
    vllm_max_model_length: int = 1536,   # MATH500 prompts (some carry Asymptote code) plus 512 answer tokens
) -> dict:
    """Train one arm for ``max_steps`` and return its run record (log and manifest text included).

    Evaluation sizes are item counts drawn with ``EVAL_SEED`` (the same items
    for every arm and seed); ``None`` is the whole split, 0 skips it.
    """
    import torch
    import transformers
    import trl
    from transformers import TrainerCallback
    from trl import GRPOConfig

    from benchmarks.staleness import gsm8k, math500
    from benchmarks.staleness.control_arms import MAX_GENERATION_ROUNDS, OversampleStream
    from benchmarks.staleness.diagnostics import make_lm_head_monitor
    from benchmarks.staleness.engine_probe import ProbedReplay, assert_probe_complete
    from benchmarks.staleness.evaluation import fit_prompts, make_periodic_eval

    started = time.time()
    model = MODELS[model_key]
    use_vllm = not use_cpu and model_key != "tiny"
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    reward_fn = gsm8k.REWARDS[reward]
    train_dataset = gsm8k.build_dataset("train", train_size, seed)
    datasets = {
        "gsm8k_test": _eval_set(lambda n: gsm8k.build_dataset("test", n, EVAL_SEED), eval_size),
        "math500": _eval_set(lambda n: math500.build_dataset(n, EVAL_SEED), math500_size),
    }
    periodic_dataset = _eval_set(lambda n: gsm8k.build_dataset("test", n, EVAL_SEED), periodic_eval_size)

    vllm_config = {
        "use_vllm": True, "vllm_mode": "colocate", "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
        "vllm_max_model_length": vllm_max_model_length, "vllm_importance_sampling_correction": False,
    } if use_vllm else {}
    args = GRPOConfig(
        output_dir=str(work / "trainer"),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        learning_rate=learning_rate,
        beta=0.0,
        temperature=1.0,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=seed,
        use_cpu=use_cpu,
        bf16=use_vllm,   # fp32 master weights under bf16 autocast against the bf16 engine; plain fp32 on CPU
        fp16=False,
        **vllm_config,
    )

    steps: list[dict] = []

    class StepTimer(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            self._t0 = time.time()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            steps.append({"step": state.global_step, "seconds": time.time() - self._t0})
            return control

    attest_path = work / "attest.jsonl"
    manifest_path = work / "manifest.jsonl"
    replay: Optional[ProbedReplay] = None
    trainer_class = trainer_class_for(spec)
    common = dict(model=model.model_id, reward_funcs=[reward_fn], args=args, train_dataset=train_dataset,
                  callbacks=[StepTimer()])
    if spec.replay:
        attest_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        replay = ProbedReplay(
            capacity=capacity, half_life=half_life_for_spec(spec), max_policy_age=spec.max_policy_age,
            seed=seed, attest=str(attest_path), manifest=str(manifest_path), source=f"{reward}-train",
            **replay_kwargs(spec), **priority_kwargs(spec),
        )
        trainer = trainer_class(replay_buffer=replay, **common)
    else:
        trainer = trainer_class(**common)
    if spec.trainer == "grpo_filter":
        trainer.sweep_oversample = OversampleStream(train_dataset, seed, num_generations)
    trainer.sweep_ref_model = _frozen_copy(model.model_id, trainer.model)
    dropped_prompts: dict[str, int] = {}
    if use_vllm:   # the engine refuses a call whose longest prompt plus its budget exceeds max_model_len
        budgets = {"gsm8k_test": max_completion_length, "math500": math_max_new_tokens}
        for name, budget in budgets.items():
            if datasets[name] is not None:
                datasets[name], dropped_prompts[name] = fit_prompts(
                    datasets[name], trainer.processing_class, vllm_max_model_length - budget)
        if periodic_dataset is not None:
            periodic_dataset, dropped_prompts["gsm8k_avg_at_k"] = fit_prompts(
                periodic_dataset, trainer.processing_class, vllm_max_model_length - max_completion_length)
    trainer.add_callback(make_lm_head_monitor(trainer))
    periodic: list[dict] = []
    if periodic_dataset is not None and eval_every > 0:
        trainer.add_callback(make_periodic_eval(trainer, periodic_dataset, reward_fn, every=eval_every, n=eval_samples,
                                                max_new_tokens=max_completion_length, seed=EVAL_SEED,
                                                batch_size=per_device_train_batch_size, sink=periodic))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_started = time.time()
    trainer.train()
    wall_clock = time.time() - train_started
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    engine_check = {"available": False, "steps": [], "stale_steps": []}
    if replay is not None:
        replay.close()
        engine_check = replay.engine_check()
        if use_vllm:
            # A colocated engine that served stale weights would corrupt the staleness axis. A run that could
            # not be probed at every step is still returned, so its log is kept, but marked failed: the report
            # refuses it rather than reporting it as an ordinary result.
            try:
                assert_probe_complete(engine_check, replay.stats["hook_calls"])
            except RuntimeError as exc:
                engine_check = {**engine_check, "failure": str(exc)}

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    evals = _final_evals(trainer, datasets, reward_fn, max_new_tokens=max_completion_length,
                         math_max_new_tokens=math_max_new_tokens, batch_size=per_device_train_batch_size)
    evals["gsm8k_avg_at_k"] = periodic
    final = evals.get("gsm8k_test")
    record = {
        "arm": spec.name, "replay": spec.replay, "policy": spec.policy, "max_policy_age": spec.max_policy_age,
        "max_log_ratio": spec.max_log_ratio, "trainer": spec.trainer, "priority": spec.priority, "stress": spec.stress,
        "seed": seed,
        "model": model.model_id, "model_key": model_key, "gpu": "cpu" if use_cpu else model.gpu,
        "gpu_name": None if use_cpu or not torch.cuda.is_available() else torch.cuda.get_device_name(0),
        "price_per_hour": 0.0 if use_cpu else model.price_per_hour,
        "device": str(trainer.model.device),
        "config": {
            "max_steps": max_steps, "seed": seed, "reward": reward,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps, "num_generations": num_generations,
            "prompts_per_step": per_device_train_batch_size * gradient_accumulation_steps // num_generations,
            "max_completion_length": max_completion_length, "learning_rate": learning_rate,
            "beta": 0.0, "temperature": 1.0, "loss_type": args.loss_type, "scale_rewards": args.scale_rewards,
            "train_size": train_size or len(train_dataset),
            "eval_size": len(datasets["gsm8k_test"]) if datasets["gsm8k_test"] is not None else 0,
            "periodic_eval_size": len(periodic_dataset) if periodic_dataset is not None else 0,
            "math500_size": len(datasets["math500"]) if datasets["math500"] is not None else 0,
            "eval_every": eval_every, "eval_samples": eval_samples, "math_max_new_tokens": math_max_new_tokens,
            "eval_prompts_dropped_over_length": dropped_prompts,
            "max_generation_rounds": MAX_GENERATION_ROUNDS if spec.trainer == "grpo_filter" else None,
            "capacity": capacity if spec.replay else None,
            "half_life": half_life_for_spec(spec) if spec.replay else None,
            "beta_is": replay.beta if replay is not None else None,
            "vllm": {"version": VLLM_VERSION, **vllm_config} if use_vllm else None,
            "vllm_seed": 0 if use_vllm else None,  # TRL colocate seeds LLM from process_index // tp
            "bf16": args.bf16, "fp16": args.fp16, "dtype": str(next(unwrapped.parameters()).dtype),
        },
        "versions": {"trl": trl.__version__, "transformers": transformers.__version__, "torch": torch.__version__},
        "wall_clock_seconds": wall_clock,
        "peak_memory_gb": peak_memory_gb,
        "steps": steps,
        "log_history": trainer.state.log_history,
        "totals": dict(replay.stats) if replay is not None else None,
        "final_buffer_size": replay.buffer.size if replay is not None else None,
        "eval": {k: v for k, v in final.items() if k != "per_item"} if final is not None and "error" not in final else None,
        "evals": evals,
        "engine_check": engine_check,
        "attestation": None,
    }
    if replay is not None:
        log = replay.buffer.attestation_log
        record["attestation"] = {
            "records": len(log.records), "head_digest": log.head_digest,
            "text": attest_path.read_text(), "manifest_text": manifest_path.read_text(),
        }
    record["container_seconds"] = time.time() - started
    return record


def _remote(job: dict) -> dict:
    sys.path.insert(0, "/reservoir_src")
    spec = ArmSpec(**job.pop("spec"))
    return run_arm(spec, work_dir=f"/tmp/sweep_{spec.name}_seed{job['seed']}", **job)


# single_use_containers: one job per container, so no trainer, colocated vLLM engine or process group outlives its run.
@app.function(gpu="A10G", image=vllm_image, volumes={"/hf_cache": hf_cache}, timeout=4 * 3600, single_use_containers=True)
def run_arm_a10g(job: dict) -> dict:
    return _remote(job)


@app.function(gpu="A100-80GB", image=vllm_image, volumes={"/hf_cache": hf_cache}, timeout=4 * 3600, single_use_containers=True)
def run_arm_a100(job: dict) -> dict:
    return _remote(job)


REMOTE = {"A10G": run_arm_a10g, "A100-80GB": run_arm_a100}


# ---------------------------------------------------------------------------
# Writing, reporting
# ---------------------------------------------------------------------------

def label_for(record: dict) -> str:
    return f"{record['arm']}_seed{record['seed']}"


def sanitize(value: Any) -> Any:
    """Replace non-finite floats with ``None`` so the record is standard JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def write_run(record: dict, out_dir: Path) -> Path:
    """Write the run's files; a Reservoir arm's log is verified by the checker first (raises if rejected)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = sanitize(record)
    if clean.get("attestation") is None:
        from benchmarks.provenance import gpu_of, stamped

        path = out_dir / f"{label_for(clean)}.json"
        path.write_text(json.dumps(stamped(clean, gpu=gpu_of(clean)), indent=2))
        return path
    try:
        return write_results(clean, label_for(clean), out_dir)
    finally:
        # The caller's record learns the checker's verdict too (write_results sets it even when it raises).
        record["attestation"]["checker"] = clean["attestation"].get("checker")


def _dump_failed(record: dict, out_dir: Path) -> None:
    """Keep a record that could not be written as ``<label>.failed.json`` (outside the report's glob of runs)."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{label_for(record)}.failed.json").write_text(json.dumps(sanitize(record), indent=2, default=str))
    except Exception as exc:  # the original failure is already reported; say this one too
        print(f"could not dump the failed record of {label_for(record)}: {exc!r}")


def write_report(run_dir: Path, policy_on: Optional[str] = None, max_log_ratio: Optional[float] = None,
                 copies: tuple[Path, ...] = (), figure_path: Optional[Path] = None) -> dict:
    """Build the report from ``run_dir`` and write it to ``run_dir/report.json`` and to every path in ``copies``."""
    run_dir = Path(run_dir)
    report = build_report(run_dir, policy_on=policy_on, max_log_ratio=max_log_ratio)
    figure = figure_path or run_dir / "ess_vs_reward.png"
    report["figure"] = str(figure)
    for path in (run_dir / "report.json", *copies):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
    plot_report(report, figure)
    print(format_table(report))
    print(f"wrote {', '.join(str(p) for p in (run_dir / 'report.json', *copies))} and {figure}")
    return report


def _size(value: int) -> Optional[int]:
    """Command-line evaluation size: -1 is the whole split, 0 skips, N is N items."""
    return None if value < 0 else value


def _eval_completions(max_steps: int, eval_every: int, eval_samples: int, sizes: dict[str, int]) -> int:
    """Completions one run's evaluations generate (full split sizes for -1)."""
    full = {"eval": GSM8K_TEST_SIZE, "periodic": GSM8K_TEST_SIZE, "math500": MATH500_SIZE}
    n = {k: full[k] if v < 0 else v for k, v in sizes.items()}
    periodic = (max_steps // eval_every) * n["periodic"] * eval_samples if eval_every > 0 else 0
    return periodic + n["eval"] + n["math500"]


def describe_plan(*, max_steps: int, seeds: str, arms: str, model: str, policy_on: str, max_log_ratio: float,
                  sizes: dict[str, int], eval_every: int, eval_samples: int, seconds_per_step: float,
                  seconds_per_eval_completion: float, out: Optional[str]) -> tuple[list, Path, dict]:
    """Plan the jobs, print them and the cost estimate; the same text for ``modal run --dry-run`` and the local one."""
    spec = MODELS[model]
    if spec.gpu not in REMOTE:
        raise SystemExit(f"--model {model} has no GPU function ({spec.gpu!r}); choose from {[k for k, m in MODELS.items() if m.gpu in REMOTE]}")
    jobs = plan_jobs(arms, seeds, policy_on, max_log_ratio)
    completions = _eval_completions(max_steps, eval_every, eval_samples, sizes)
    estimate = estimate_jobs_cost([arm for arm, _ in jobs], max_steps=max_steps, seconds_per_step=seconds_per_step,
                                  overhead_seconds=CONTAINER_OVERHEAD_SECONDS, price_per_hour=spec.price_per_hour,
                                  eval_seconds=completions * seconds_per_eval_completion)
    out_dir = Path(out) if out else RESULTS_DIR / f"sweep_{model}_{max_steps}steps"
    gate = f" (max_log_ratio={max_log_ratio})" if policy_on == "gate" else ""
    print(f"{len(jobs)} runs on {spec.gpu} ({spec.model_id}), {max_steps} steps each, policy on = {policy_on}{gate},"
          f" results -> {out_dir}")
    for arm, seed in jobs:
        print(f"  {arm.name:<24} seed {seed}{'  (stress arm, first seed only)' if arm.stress else ''}")
    print(f"estimate at {seconds_per_step:.1f} s/step (x{GRPO_FILTER_STEP_FACTOR} for grpo_filter) + "
          f"{CONTAINER_OVERHEAD_SECONDS:.0f} s overhead + {completions} eval completions at "
          f"{seconds_per_eval_completion:.3f} s: {estimate['per_run_minutes']:.0f} min/run on average, "
          f"{estimate['gpu_hours']:.1f} GPU-hours, ${estimate['dollars']:.2f} at ${spec.price_per_hour:.2f}/h "
          "(Modal list price)")
    return jobs, out_dir, {**estimate, "eval_completions_per_run": completions}


@app.local_entrypoint()
def main(
    max_steps: int = 300,
    seeds: str = ",".join(str(s) for s in DEFAULT_SEEDS),
    arms: str = "all",
    model: str = "0.5b",
    policy_on: str = "staleness",
    max_log_ratio: float = DEFAULT_MAX_LOG_RATIO,
    eval_size: int = -1,
    periodic_eval_size: int = -1,
    math500_size: int = -1,
    eval_every: int = 50,
    eval_samples: int = 8,
    seconds_per_step: float = SECONDS_PER_STEP_PRIOR,
    seconds_per_eval_completion: float = SECONDS_PER_EVAL_COMPLETION_PRIOR,
    dry_run: bool = False,
    out: Optional[str] = None,
):
    spec = MODELS[model]
    sizes = {"eval": eval_size, "periodic": periodic_eval_size, "math500": math500_size}
    jobs, out_dir, _ = describe_plan(max_steps=max_steps, seeds=seeds, arms=arms, model=model, policy_on=policy_on,
                                     max_log_ratio=max_log_ratio, sizes=sizes, eval_every=eval_every,
                                     eval_samples=eval_samples, seconds_per_step=seconds_per_step,
                                     seconds_per_eval_completion=seconds_per_eval_completion, out=out)
    if dry_run:
        return
    payloads = [{"spec": arm.to_dict(), "seed": seed, "max_steps": max_steps, "model_key": model,
                 "eval_size": _size(eval_size), "periodic_eval_size": _size(periodic_eval_size),
                 "math500_size": _size(math500_size), "eval_every": eval_every, "eval_samples": eval_samples}
                for arm, seed in jobs]
    failures: list[str] = []
    for (arm, seed), result in zip(jobs, REMOTE[spec.gpu].map(payloads, return_exceptions=True)):
        if isinstance(result, Exception):
            failures.append(f"{arm.name} seed {seed}: {result!r}")
            print(f"FAILED {arm.name} seed {seed}: {result!r}")
            continue
        try:
            path = write_run(result, out_dir)
        except Exception as exc:  # a rejected log or a write error must not lose the other finished runs
            failures.append(f"{arm.name} seed {seed}: {exc!r}")
            print(f"NOT WRITTEN {arm.name} seed {seed}: {exc!r}")
            _dump_failed(result, out_dir)
            continue
        print(f"wrote {path}: {_run_line(result)}")
    if failures:
        raise SystemExit("some runs failed; the report was not built:\n  " + "\n  ".join(failures))
    write_report(out_dir, policy_on=policy_on, max_log_ratio=max_log_ratio, copies=(REPORT_PATH,),
                 figure_path=FIGURE_PATH)


def _run_line(record: dict) -> str:
    totals = record["totals"] or {}
    acc = (record.get("eval") or {}).get("accuracy")
    memory = record.get("peak_memory_gb")
    return (f"steps {len(record['steps'])}, eval acc {'-' if acc is None else format(acc, '.3f')}, "
            f"replaced {totals.get('replaced_rows', 0)}, declined {totals.get('declined_rows', 0)}, "
            f"stale engine steps {len(record['engine_check']['stale_steps'])}, "
            f"peak memory {'-' if memory is None else format(memory, '.1f') + ' GB'}, "
            f"{record['container_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU smoke run, offline report or the GPU plan, without a Modal account")
    parser.add_argument("--local", action="store_true", help="run the arms here on CPU with the tiny model")
    parser.add_argument("--report-only", type=Path, metavar="RUN_DIR", help="rebuild the report and figure from RUN_DIR")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the GPU sweep's plan and cost estimate at the Modal entrypoint's defaults")
    parser.add_argument("--max-steps", type=int, default=None, help="local: default 3; dry run: default 300")
    parser.add_argument("--seeds", default=None, help="local: default 42; dry run: default 42,43,44")
    parser.add_argument("--arms", default="all")
    parser.add_argument("--policy-on", default=None, choices=("gate", "staleness"),
                        help="local runs: the _on arms' policy (default staleness); report-only: read from the records")
    parser.add_argument("--max-log-ratio", type=float, default=None, help="local runs: the gate (default 2.0)")
    parser.add_argument("--reward", default="even_length", choices=("even_length", "gsm8k"))
    parser.add_argument("--eval-every", type=int, default=None, help="avg@k every N steps (local default 6, GPU 50)")
    parser.add_argument("--eval-samples", type=int, default=None, help="k of avg@k (local default 2, GPU 8)")
    parser.add_argument("--eval-size", type=int, default=None,
                        help="items per evaluation, -1 the whole split (local default 4, GPU -1)")
    parser.add_argument("--seconds-per-step", type=float, default=SECONDS_PER_STEP_PRIOR, help="dry run only")
    parser.add_argument("--seconds-per-eval-completion", type=float, default=SECONDS_PER_EVAL_COMPLETION_PRIOR,
                        help="dry run only")
    parser.add_argument("--out-dir", type=Path, default=None, help="where to write (default: a temporary directory)")
    parser.add_argument("--report-path", type=Path, default=None,
                        help="also write the report here (it is always written to <run dir>/report.json)")
    parser.add_argument("--figure-path", type=Path, default=None, help="figure PNG (default: <run dir>/ess_vs_reward.png)")
    cli = parser.parse_args()
    if sum([cli.local, cli.report_only is not None, cli.dry_run]) != 1:
        parser.error("exactly one of --local, --report-only or --dry-run is required")
    sys.path.insert(0, _repo_src)
    import tempfile

    policy_on = cli.policy_on or "staleness"
    max_log_ratio = cli.max_log_ratio if cli.max_log_ratio is not None else DEFAULT_MAX_LOG_RATIO
    copies = (cli.report_path,) if cli.report_path else ()
    if cli.dry_run:
        size = -1 if cli.eval_size is None else cli.eval_size
        describe_plan(max_steps=cli.max_steps or 300, seeds=cli.seeds or ",".join(map(str, DEFAULT_SEEDS)),
                      arms=cli.arms, model="0.5b", policy_on=policy_on, max_log_ratio=max_log_ratio,
                      sizes={"eval": size, "periodic": size, "math500": size},
                      eval_every=50 if cli.eval_every is None else cli.eval_every,
                      eval_samples=cli.eval_samples or 8, seconds_per_step=cli.seconds_per_step,
                      seconds_per_eval_completion=cli.seconds_per_eval_completion, out=None)
        sys.exit(0)
    if cli.report_only is not None:
        write_report(cli.report_only, policy_on=cli.policy_on, max_log_ratio=cli.max_log_ratio, copies=copies,
                     figure_path=cli.figure_path)
        sys.exit(0)
    out_dir = cli.out_dir or Path(tempfile.mkdtemp(prefix="reservoir_staleness_sweep_"))
    work_root = Path(tempfile.mkdtemp(prefix="reservoir_staleness_work_"))   # trainer output stays out of out_dir
    size = _size(4 if cli.eval_size is None else cli.eval_size)
    for arm, seed in plan_jobs(cli.arms, cli.seeds or "42", policy_on, max_log_ratio):
        record = run_arm(arm, seed=seed, max_steps=cli.max_steps or 3, model_key="tiny", reward=cli.reward,
                         use_cpu=True, work_dir=str(work_root / f"{arm.name}_seed{seed}"), eval_size=size,
                         periodic_eval_size=size, math500_size=size,
                         eval_every=6 if cli.eval_every is None else cli.eval_every,
                         eval_samples=cli.eval_samples or 2, math_max_new_tokens=16, per_device_train_batch_size=8,
                         gradient_accumulation_steps=1, num_generations=4, max_completion_length=16, train_size=64)
        path = write_run(record, out_dir)
        print(f"wrote {path}: {_run_line(record)}")
    write_report(out_dir, policy_on=policy_on, max_log_ratio=max_log_ratio, copies=copies, figure_path=cli.figure_path)
