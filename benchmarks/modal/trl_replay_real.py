"""benchmarks/modal/trl_replay_real.py

Integration check for ``reservoir.integrations.trl``: a short GRPO run on a
tiny model with ``ReservoirGRPOTrainer`` replaying dead groups from a
``ReservoirReplay`` buffer, producing an attestation log that the
independent checker (``python -m checker.verify``) accepts.

This is an integration check, not a benchmark: the model is a test-size
Qwen2 that cannot learn the task, and no training-quality claim is made
for replay (``docs/nonclaims.md``). What the run establishes is that the
adapter survives TRL's real call path, that dead groups are replaced, and
that the resulting log verifies.

Reward: 1.0 if the completion text has an even number of characters,
else 0.0. It is deterministic, close to a coin flip for a near-random
model, and so gives groups with variance most of the time (which fill
the buffer) and all-equal "dead" groups about one time in eight with four
generations (which trigger replay). A reward the model cannot earn at
all, such as "contains a digit", makes every group dead and nothing is
ever stored.

Usage
-----
    modal run benchmarks/modal/trl_replay_real.py                     # T4, 40 steps
    modal run benchmarks/modal/trl_replay_real.py --max-steps 20
    python benchmarks/modal/trl_replay_real.py --local --max-steps 3   # CPU smoke run, no Modal account
    python benchmarks/modal/trl_replay_real.py --local --max-steps 40 --out-dir benchmarks/modal/results

Results (JSON plus the attestation log, which the entrypoint verifies with
the checker before writing) land in benchmarks/modal/results/ for Modal
runs and in a temporary directory for ``--local`` runs unless ``--out-dir``
says otherwise, so smoke runs do not land next to committed evidence.

Precision is fixed at float32 (``bf16=False, fp16=False``): TRL defaults to
bf16, which a T4 does not support natively, and the model is small enough
that float32 costs nothing. Cost estimate, not yet measured: T4 @ $0.59/hr
x ~10 min, about $0.10.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Modal image: the versions the local test suite runs against
# ---------------------------------------------------------------------------

_DEPS = [
    "torch==2.14.0",
    "transformers==5.17.0",
    "trl==1.13.0",
    "datasets==5.0.1",
    "accelerate==1.15.0",
    "numpy==2.2.6",
]

MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
DATASET_ID = "trl-internal-testing/zen"
DATASET_CONFIG = "standard_prompt_only"
RESULTS_DIR = Path(__file__).parent / "results"

try:
    _repo_src = str(Path(__file__).parents[2] / "src")
except IndexError:
    _repo_src = "."  # inside the container the image already holds the sources

hf_cache = modal.Volume.from_name("reservoir-hf-cache", create_if_missing=True)

# Build steps first, the local source mount last: Modal refuses a build step
# after an add_local_* layer, so scripts that derive a variant (an extra env
# var, say) start from ``base_image`` and add the mount themselves.
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*_DEPS)
    .env({"HF_HOME": "/hf_cache"})
)
image = base_image.add_local_dir(_repo_src, remote_path="/reservoir_src")

app = modal.App("reservoir-trl-replay-real")


# ---------------------------------------------------------------------------
# The run itself (plain Python; runs locally or inside the container)
# ---------------------------------------------------------------------------

def even_length_reward(completions, **kwargs) -> list[float]:
    """1.0 when the completion text has an even number of characters, else 0.0."""
    return [1.0 if len(text) % 2 == 0 else 0.0 for text in completions]


def run_grpo(
    *,
    max_steps: int = 40,
    seed: int = 42,
    per_device_train_batch_size: int = 8,
    num_generations: int = 4,
    max_completion_length: int = 16,
    capacity: int = 256,
    half_life: int = 8,
    max_policy_age: int = 24,
    buffer_seed: int = 0,
    attest_path: str = "/tmp/reservoir_trl_attest.jsonl",
    manifest_path: Optional[str] = None,
    source: Optional[str] = DATASET_ID.rsplit("/", 1)[-1],
    output_dir: str = "/tmp/reservoir_trl_out",
    use_cpu: bool = False,
    extra_config: Optional[dict] = None,
    model_id: str = MODEL_ID,
    torch_deterministic: bool = False,
) -> dict:
    """Train for ``max_steps`` with replay and return the run record (log and manifest included).

    ``source`` is written on every stored row's insert record; ``manifest_path``
    adds the manifest file so the checker can open every content digest.
    ``extra_config`` is merged into ``GRPOConfig`` (the GPU reproducibility
    runs use it for vLLM settings); ``model_id`` overrides the tiny test
    model (vLLM cannot serve it: its attention head size is 2).
    ``torch_deterministic`` asks PyTorch for deterministic kernels
    (``use_deterministic_algorithms`` with ``warn_only``, cuDNN
    deterministic, no autotuning). ``CUBLAS_WORKSPACE_CONFIG`` must already
    be set in the environment before CUDA initialises for that to be
    complete; the GPU runner sets it on the image.
    """
    import torch

    if torch_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    from datasets import load_dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig

    from reservoir.integrations.trl import ReservoirGRPOTrainer, ReservoirReplay

    Path(attest_path).unlink(missing_ok=True)
    if manifest_path is not None:
        Path(manifest_path).unlink(missing_ok=True)
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")
    args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        learning_rate=1e-5,
        beta=0.0,
        scale_rewards="group",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=seed,
        use_cpu=use_cpu,
        bf16=False,
        fp16=False,
        **(extra_config or {}),
    )
    replay = ReservoirReplay(
        capacity=capacity, half_life=half_life, max_policy_age=max_policy_age,
        seed=buffer_seed, attest=attest_path, manifest=manifest_path, source=source,
    )

    steps: list[dict] = []

    class StepRecorder(TrainerCallback):
        """Snapshot the adapter's counters after every optimizer step."""

        def on_step_end(self, args, state, control, **kwargs):
            steps.append({
                "global_step": state.global_step,
                "buffer_size": replay.buffer.size,
                "buffer_version": replay.buffer.current_version,
                **replay.stats,
            })
            return control

    trainer = ReservoirGRPOTrainer(
        model=model_id,
        reward_funcs=[even_length_reward],
        args=args,
        train_dataset=dataset,
        replay_buffer=replay,
        callbacks=[StepRecorder()],
    )
    started = time.time()
    trainer.train()
    wall_clock = time.time() - started
    replay.close()

    import transformers
    import trl

    log_text = Path(attest_path).read_text()
    manifest_text = Path(manifest_path).read_text() if manifest_path is not None else None
    log = replay.buffer.attestation_log
    return {
        "model": model_id,
        "dataset": f"{DATASET_ID}/{DATASET_CONFIG}",
        "config": {
            "max_steps": max_steps, "seed": seed, "per_device_train_batch_size": per_device_train_batch_size,
            "num_generations": num_generations, "max_completion_length": max_completion_length,
            "capacity": capacity, "half_life": half_life, "max_policy_age": max_policy_age,
            "buffer_seed": buffer_seed, "beta_is": replay.beta, "source": source,
            "device": "cpu" if use_cpu else str(trainer.model.device), "extra_config": extra_config or {},
            "torch_deterministic": torch_deterministic,
            "bf16": args.bf16, "fp16": args.fp16, "gradient_checkpointing": args.gradient_checkpointing,
        },
        "versions": {"trl": trl.__version__, "transformers": transformers.__version__, "torch": torch.__version__},
        "wall_clock_seconds": wall_clock,
        "totals": dict(replay.stats),
        "final_buffer_size": replay.buffer.size,
        "n_rebases": replay.buffer.n_rebases,
        "steps": steps,
        # TRL computes its reward statistics before the hook replaces dead
        # rows, so "reward", "reward_std" and "frac_reward_zero_std" describe
        # the generated batch; "loss" and "grad_norm" reflect the replayed one.
        "log_history": trainer.state.log_history,
        "attestation": {"records": len(log.records), "head_digest": log.head_digest, "text": log_text,
                        "manifest_text": manifest_text},
    }


@app.function(gpu="T4", image=image, volumes={"/hf_cache": hf_cache}, timeout=3600)
def run_grpo_remote(**kwargs) -> dict:
    sys.path.insert(0, "/reservoir_src")
    return run_grpo(**kwargs)


# ---------------------------------------------------------------------------
# Writing results and verifying the log
# ---------------------------------------------------------------------------

def write_results(results: dict, label: str, out_dir: Path = RESULTS_DIR) -> Path:
    """Write <label>.json, <label>.attest.jsonl and, if present, <label>.manifest.jsonl;
    verify the log (with the manifest) through the checker first."""
    out_dir.mkdir(parents=True, exist_ok=True)
    attest = out_dir / f"{label}.attest.jsonl"
    attest.write_text(results["attestation"].pop("text"))
    command = [sys.executable, "-m", "checker.verify", str(attest)]
    manifest_text = results["attestation"].pop("manifest_text", None)
    if manifest_text is not None:
        manifest = out_dir / f"{label}.manifest.jsonl"
        manifest.write_text(manifest_text)
        command += ["--manifest", str(manifest)]
    verify = subprocess.run(command, capture_output=True, text=True, cwd=str(Path(__file__).parents[2]))
    results["attestation"]["checker"] = {
        "command": "python -m " + " ".join(command[2:]),
        "returncode": verify.returncode,
        "output": (verify.stdout + verify.stderr).strip(),
    }
    out = out_dir / f"{label}.json"
    out.write_text(json.dumps(results, indent=2))
    return out


def label_for(results: dict) -> str:
    cfg = results["config"]
    return f"trl_replay_tiny-qwen2_{cfg['max_steps']}steps_seed{cfg['seed']}"


def report(results: dict, path: Path) -> None:
    totals = results["totals"]
    print(f"wrote {path}")
    print(f"steps={results['config']['max_steps']} hook_calls={totals['hook_calls']} "
          f"dead_groups={totals['dead_groups']} replaced_rows={totals['replaced_rows']} "
          f"ingested_rows={totals['ingested_rows']} final_buffer_size={results['final_buffer_size']}")
    checker = results["attestation"]["checker"]
    print(f"checker: returncode={checker['returncode']} {checker['output']}")


@app.local_entrypoint()
def main(max_steps: int = 40, seed: int = 42):
    results = run_grpo_remote.remote(max_steps=max_steps, seed=seed)
    report(results, write_results(results, label_for(results)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU smoke run without a Modal account")
    parser.add_argument("--local", action="store_true", required=True)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where to write results (default: a temporary directory)")
    cli = parser.parse_args()
    sys.path.insert(0, _repo_src)
    import tempfile

    out_dir = cli.out_dir or Path(tempfile.mkdtemp(prefix="reservoir_trl_replay_"))
    local = run_grpo(max_steps=cli.max_steps, seed=cli.seed, use_cpu=True,
                     attest_path=str(out_dir / "attest.tmp.jsonl"),
                     manifest_path=str(out_dir / "manifest.tmp.jsonl"))
    (out_dir / "attest.tmp.jsonl").unlink(missing_ok=True)
    (out_dir / "manifest.tmp.jsonl").unlink(missing_ok=True)
    report(local, write_results(local, label_for(local) + "_cpu", out_dir))
