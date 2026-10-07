"""benchmarks/modal/staleness_sweep.py

The staleness sweep (planning/2026-10-06-staleness-controller.md, D3):
how far behind the policy may a replayed rollout be before replaying it
hurts, and does the staleness policy recover the loss?

Setup
-----
- TRL 1.13.0 GRPO with vLLM 0.28.0 in **colocate** mode on one A10G, as
  Martingale's runner does (server mode hangs in the NCCL weight handshake
  on Modal; ``benchmarks/modal/nccl_probe.py``). No batch invariance: its
  kernel override has no backward and would break the colocated trainer.
- ``Qwen/Qwen2.5-0.5B-Instruct`` (trainer in float32, vLLM in bfloat16);
  ``--model 1.5b`` selects the 1.5B on an A100-80GB.
- GSM8K train split, conversational prompts, one exact-match reward
  (``benchmarks/staleness/gsm8k.py``); greedy held-out accuracy on
  ``--eval-size`` test problems at the end of every run.
- 8 generations per prompt, 4 prompts per optimizer step (micro-batch 8,
  4 accumulation steps), 300 steps, 3 seeds.

Arms (``benchmarks/staleness/report.arm_specs``)::

    grpo                       plain GRPOTrainer, no replay
    reservoir_age{8,32,128}_off  ReservoirGRPOTrainer, max_policy_age=N, no policy
    reservoir_age{8,32,128}_on   the same with the policy on

"Policy on" is the 0.6.0 drift gate (``max_log_ratio``, default 2.0,
uncapped declines) until S1's ``StalenessPolicy`` lands; ``--policy-on
staleness`` is the one flag that switches, through
``benchmarks.staleness.report.replay_kwargs``.

What every run writes
---------------------
``<arm>_seed<seed>.json`` (config, versions, TRL's per-step log with the
``reservoir/*`` telemetry, per-step wall clock, the greedy evaluation, the
stale-engine check and the dollars at Modal's list price), and for the
Reservoir arms ``.attest.jsonl`` and ``.manifest.jsonl``, verified by
``python -m checker.verify`` before the record is written (a rejected log
is written for inspection and the sweep fails). ``results/
staleness_sweep_report.json`` and ``results/staleness_sweep_ess_vs_reward.png``
are built from the records by ``--report-only``.

Stale-engine check
------------------
Every Reservoir arm compares vLLM's sampling logprobs with the trainer's
forward at each step (``benchmarks/staleness/engine_probe.py``); a run
whose engine served stale weights is flagged in its record and in the
report. The plain arm is not probed: a probe there would add a forward
pass the stock trainer does not run.

Usage
-----
    # CPU smoke, no Modal account: tiny model, HF generation, coin-flip reward
    python benchmarks/modal/staleness_sweep.py --local --max-steps 3 --seeds 42
    # the plan and the cost estimate, nothing launched
    modal run benchmarks/modal/staleness_sweep.py --dry-run
    # 16-step engine smoke on the GPU before the sweep proper
    modal run benchmarks/modal/staleness_sweep.py --max-steps 16 --seeds 42 --arms grpo,reservoir_age8_on
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
from benchmarks.staleness.report import (
    DEFAULT_MAX_LOG_RATIO,
    ArmSpec,
    arm_by_name,
    arm_specs,
    build_report,
    estimate_cost,
    format_table,
    half_life_for,
    plot_report,
    replay_kwargs,
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
SECONDS_PER_STEP_PRIOR = 12.0
CONTAINER_OVERHEAD_SECONDS = 300.0   # image start, model download from the volume, vLLM engine start, eval
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

def run_arm(
    spec: ArmSpec,
    *,
    seed: int,
    max_steps: int,
    model_key: str = "0.5b",
    reward: str = "gsm8k",
    use_cpu: bool = False,
    work_dir: str = "/tmp/reservoir_sweep",
    eval_size: int = 200,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    num_generations: int = 8,
    max_completion_length: int = 256,
    learning_rate: float = 1e-6,
    capacity: int = 4096,
    train_size: Optional[int] = None,
    vllm_gpu_memory_utilization: float = 0.3,
    vllm_max_model_length: int = 768,
) -> dict:
    """Train one arm for ``max_steps`` and return its run record (log and manifest text included)."""
    import torch
    import transformers
    import trl
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from benchmarks.staleness import gsm8k
    from benchmarks.staleness.engine_probe import ProbedReplay, assert_probe_complete
    from reservoir.integrations.trl import ReservoirGRPOTrainer

    started = time.time()
    model = MODELS[model_key]
    use_vllm = not use_cpu and model_key != "tiny"
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    reward_fn = gsm8k.REWARDS[reward]
    train_dataset = gsm8k.build_dataset("train", train_size, seed)
    eval_dataset = gsm8k.build_dataset("test", eval_size, EVAL_SEED)

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
        bf16=False,
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
    if spec.replay:
        attest_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        replay = ProbedReplay(
            capacity=capacity, half_life=half_life_for(spec.max_policy_age), max_policy_age=spec.max_policy_age,
            seed=seed, attest=str(attest_path), manifest=str(manifest_path), source=f"{reward}-train",
            **replay_kwargs(spec),
        )
        trainer = ReservoirGRPOTrainer(model=model.model_id, reward_funcs=[reward_fn], args=args,
                                       train_dataset=train_dataset, replay_buffer=replay, callbacks=[StepTimer()])
    else:
        trainer = GRPOTrainer(model=model.model_id, reward_funcs=[reward_fn], args=args,
                              train_dataset=train_dataset, callbacks=[StepTimer()])
    train_started = time.time()
    trainer.train()
    wall_clock = time.time() - train_started
    if replay is not None:
        replay.close()
        if use_vllm:
            # A colocated engine that served stale weights would corrupt the staleness axis; a run that
            # could not be probed at every step is refused rather than reported as an ordinary result.
            assert_probe_complete(replay.engine_check(), replay.stats["hook_calls"])

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    evaluation = gsm8k.evaluate_greedy(unwrapped, trainer.processing_class, eval_dataset,
                                       max_new_tokens=max_completion_length,
                                       batch_size=per_device_train_batch_size, reward_fn=reward_fn)
    record = {
        "arm": spec.name, "replay": spec.replay, "policy": spec.policy, "max_policy_age": spec.max_policy_age,
        "max_log_ratio": spec.max_log_ratio, "seed": seed,
        "model": model.model_id, "model_key": model_key, "gpu": "cpu" if use_cpu else model.gpu,
        "price_per_hour": 0.0 if use_cpu else model.price_per_hour,
        "device": str(trainer.model.device),
        "config": {
            "max_steps": max_steps, "seed": seed, "reward": reward,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps, "num_generations": num_generations,
            "prompts_per_step": per_device_train_batch_size * gradient_accumulation_steps // num_generations,
            "max_completion_length": max_completion_length, "learning_rate": learning_rate,
            "beta": 0.0, "temperature": 1.0, "loss_type": args.loss_type, "scale_rewards": args.scale_rewards,
            "train_size": train_size or len(train_dataset), "eval_size": len(eval_dataset),
            "capacity": capacity if spec.replay else None,
            "half_life": half_life_for(spec.max_policy_age) if spec.replay else None,
            "beta_is": replay.beta if replay is not None else None,
            "vllm": {"version": VLLM_VERSION, **vllm_config} if use_vllm else None,
            "bf16": args.bf16, "fp16": args.fp16, "dtype": str(next(unwrapped.parameters()).dtype),
        },
        "versions": {"trl": trl.__version__, "transformers": transformers.__version__, "torch": torch.__version__},
        "wall_clock_seconds": wall_clock,
        "steps": steps,
        "log_history": trainer.state.log_history,
        "totals": dict(replay.stats) if replay is not None else None,
        "final_buffer_size": replay.buffer.size if replay is not None else None,
        "eval": evaluation,
        "engine_check": replay.engine_check() if replay is not None else {"available": False, "steps": [], "stale_steps": []},
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
        path = out_dir / f"{label_for(clean)}.json"
        path.write_text(json.dumps(clean, indent=2))
        return path
    try:
        return write_results(clean, label_for(clean), out_dir)
    finally:
        # The caller's record learns the checker's verdict too (write_results sets it even when it raises).
        record["attestation"]["checker"] = clean["attestation"].get("checker")


def write_report(run_dir: Path, policy_on: Optional[str] = None, max_log_ratio: Optional[float] = None,
                 report_path: Path = REPORT_PATH, figure_path: Path = FIGURE_PATH) -> dict:
    report = build_report(run_dir, policy_on=policy_on, max_log_ratio=max_log_ratio)
    report["figure"] = str(figure_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    plot_report(report, figure_path)
    print(format_table(report))
    print(f"wrote {report_path} and {figure_path}")
    return report


def _parse_list(text: str, cast=str) -> list:
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def plan_jobs(arms: str, seeds: str, policy_on: str, max_log_ratio: float) -> list[tuple[ArmSpec, int]]:
    specs = arm_specs(policy_on=policy_on, max_log_ratio=max_log_ratio)
    chosen = specs if arms == "all" else [arm_by_name(specs, name) for name in _parse_list(arms)]
    return [(spec, seed) for spec in chosen for seed in _parse_list(seeds, int)]


@app.local_entrypoint()
def main(
    max_steps: int = 300,
    seeds: str = ",".join(str(s) for s in DEFAULT_SEEDS),
    arms: str = "all",
    model: str = "0.5b",
    policy_on: str = "gate",
    max_log_ratio: float = DEFAULT_MAX_LOG_RATIO,
    eval_size: int = 200,
    seconds_per_step: float = SECONDS_PER_STEP_PRIOR,
    dry_run: bool = False,
    out: Optional[str] = None,
):
    spec = MODELS[model]
    if spec.gpu not in REMOTE:
        raise SystemExit(f"--model {model} has no GPU function ({spec.gpu!r}); choose from {[k for k, m in MODELS.items() if m.gpu in REMOTE]}")
    jobs = plan_jobs(arms, seeds, policy_on, max_log_ratio)
    estimate = estimate_cost(n_runs=len(jobs), max_steps=max_steps, seconds_per_step=seconds_per_step,
                             overhead_seconds=CONTAINER_OVERHEAD_SECONDS, price_per_hour=spec.price_per_hour)
    out_dir = Path(out) if out else RESULTS_DIR / f"sweep_{model}_{max_steps}steps"
    print(f"{len(jobs)} runs on {spec.gpu} ({spec.model_id}), {max_steps} steps each, policy on = {policy_on}"
          f" (max_log_ratio={max_log_ratio}), results -> {out_dir}")
    for arm, seed in jobs:
        print(f"  {arm.name:<22} seed {seed}")
    print(f"estimate at {seconds_per_step:.1f} s/step + {CONTAINER_OVERHEAD_SECONDS:.0f} s overhead: "
          f"{estimate['per_run_minutes']:.0f} min/run, {estimate['gpu_hours']:.1f} GPU-hours, "
          f"${estimate['dollars']:.2f} at ${spec.price_per_hour:.2f}/h (Modal list price)")
    if dry_run:
        return
    payloads = [{"spec": arm.to_dict(), "seed": seed, "max_steps": max_steps, "model_key": model,
                 "eval_size": eval_size} for arm, seed in jobs]
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
            continue
        totals = result["totals"] or {}
        print(f"wrote {path}: eval acc {result['eval']['accuracy']:.3f}, replaced {totals.get('replaced_rows', 0)}, "
              f"declined {totals.get('declined_rows', 0)}, stale engine steps {len(result['engine_check']['stale_steps'])}, "
              f"{result['container_seconds'] / 60:.1f} min")
    if failures:
        raise SystemExit("some runs failed; the report was not built:\n  " + "\n  ".join(failures))
    write_report(out_dir, policy_on=policy_on, max_log_ratio=max_log_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU smoke run or offline report, without a Modal account")
    parser.add_argument("--local", action="store_true", help="run the arms here on CPU with the tiny model")
    parser.add_argument("--report-only", type=Path, metavar="RUN_DIR", help="rebuild the report and figure from RUN_DIR")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--arms", default="all")
    parser.add_argument("--policy-on", default=None, choices=("gate", "staleness"),
                        help="local runs: the _on arms' policy (default gate); report-only: read from the records")
    parser.add_argument("--max-log-ratio", type=float, default=None, help="local runs: the gate (default 2.0)")
    parser.add_argument("--reward", default="even_length", choices=("even_length", "gsm8k"))
    parser.add_argument("--out-dir", type=Path, default=None, help="where to write (default: a temporary directory)")
    parser.add_argument("--report-path", type=Path, default=None, help="report JSON (default: <out-dir>/report.json)")
    parser.add_argument("--figure-path", type=Path, default=None, help="figure PNG (default: <out-dir>/ess_vs_reward.png)")
    cli = parser.parse_args()
    if not cli.local and cli.report_only is None:
        parser.error("one of --local or --report-only is required")
    sys.path.insert(0, _repo_src)
    import tempfile

    if cli.report_only is not None:
        run_dir = cli.report_only
        write_report(run_dir, policy_on=cli.policy_on, max_log_ratio=cli.max_log_ratio,
                     report_path=cli.report_path or REPORT_PATH, figure_path=cli.figure_path or FIGURE_PATH)
        sys.exit(0)
    out_dir = cli.out_dir or Path(tempfile.mkdtemp(prefix="reservoir_staleness_sweep_"))
    policy_on = cli.policy_on or "gate"
    max_log_ratio = cli.max_log_ratio if cli.max_log_ratio is not None else DEFAULT_MAX_LOG_RATIO
    for arm, seed in plan_jobs(cli.arms, cli.seeds, policy_on, max_log_ratio):
        record = run_arm(arm, seed=seed, max_steps=cli.max_steps, model_key="tiny", reward=cli.reward, use_cpu=True,
                         work_dir=str(out_dir / "work" / f"{arm.name}_seed{seed}"), eval_size=4,
                         per_device_train_batch_size=8, gradient_accumulation_steps=1, num_generations=4,
                         max_completion_length=16, train_size=64)
        path = write_run(record, out_dir)
        totals = record["totals"] or {}
        print(f"wrote {path}: steps {len(record['steps'])}, replaced {totals.get('replaced_rows', 0)}, "
              f"declined {totals.get('declined_rows', 0)}, eval acc {record['eval']['accuracy']:.2f}, "
              f"{record['container_seconds']:.1f} s")
    write_report(out_dir, policy_on=policy_on, max_log_ratio=max_log_ratio,
                 report_path=cli.report_path or out_dir / "report.json",
                 figure_path=cli.figure_path or out_dir / "ess_vs_reward.png")
