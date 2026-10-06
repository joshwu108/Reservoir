"""benchmarks/modal/trl_replay_distributed.py

The TRL adapter's multi-process path on a real process group: two
training processes on two GPUs, launched with
``accelerate launch --num_processes 2 --multi_gpu`` (plain NCCL through
``torch.distributed``; TRL's default DDP path, no vLLM and no TRL server
mode, which is parked on an NCCL weight-sync hang recorded in
``docs/reproducible-training.md``). Each process runs ``run_grpo`` from
the tier-1 script through ``trl_replay_driver.py``; rank 0 owns the
``ReservoirReplay`` buffer and writes the attestation log and the
manifest once, the other rank holds no buffer, and the resulting log is
verified locally by the independent checker together with the manifest.

Like the tier-1 run this is an integration check, not a benchmark: the
model is a test-size Qwen2 that cannot learn the task and no training
quality claim is made. What the run establishes is that the gather,
the owner-side hook over the global batch and the broadcast back work on
a real NCCL process group, with one log writer, and that the log
verifies.

Usage
-----
    uv run modal run benchmarks/modal/trl_replay_distributed.py                  # A10G:2, 40 steps
    uv run modal run benchmarks/modal/trl_replay_distributed.py --max-steps 20
    uv run python benchmarks/modal/trl_replay_distributed.py --local --max-steps 3   # 2 CPU processes (gloo), no Modal account
    uv run python benchmarks/modal/trl_replay_distributed.py --local --max-steps 40 --out-dir benchmarks/modal/results

Results (``<label>.json``, ``<label>.attest.jsonl``, ``<label>.manifest.jsonl``)
land in benchmarks/modal/results/ for Modal runs and in a temporary
directory for ``--local`` runs unless ``--out-dir`` says otherwise. The
``--local`` run uses ``torchrun`` on the gloo backend because
``accelerate launch --cpu`` starts a single process; the collectives the
adapter calls are the same ones.

Cost estimate, not yet measured: A10G x2 @ about $2.20/hr x ~10 min, about
$0.40.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import modal

# ``modal run`` puts the checkout on sys.path and the container has /root
# there; a plain ``python benchmarks/modal/...py --local`` has only this
# directory, so add the checkout root for the ``benchmarks`` import.
_HERE = Path(__file__).resolve()
if len(_HERE.parents) > 2 and str(_HERE.parents[2]) not in sys.path:
    sys.path.insert(0, str(_HERE.parents[2]))

from benchmarks.modal.trl_replay_real import (  # noqa: E402
    RESULTS_DIR, hf_cache, image, label_for, repo_root, report, write_results,
)

GPU = "A10G:2"
NUM_PROCESSES = 2
LAUNCH_TIMEOUT_S = 1800
"""A hung collective must not sit until Modal's own timeout; the launcher output is reported instead."""
TAIL_CHARS = 6000

_REPO = repo_root()
_SRC = str(_REPO / "src") if (_REPO / "src").is_dir() else "/reservoir_src"
DRIVER = _REPO / "benchmarks" / "modal" / "trl_replay_driver.py"

app = modal.App("reservoir-trl-replay-distributed")


# ---------------------------------------------------------------------------
# Launching the driver (plain Python; runs locally or inside the container)
# ---------------------------------------------------------------------------

def launch_command(driver_args: list[str], *, cpu: bool) -> list[str]:
    """``accelerate launch --num_processes 2 --multi_gpu`` on GPUs; ``torchrun`` on CPUs (gloo)."""
    if cpu:
        return [sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc_per_node={NUM_PROCESSES}", str(DRIVER), "--cpu", *driver_args]
    return [sys.executable, "-m", "accelerate.commands.launch", f"--num_processes={NUM_PROCESSES}",
            "--multi_gpu", "--mixed_precision=no", str(DRIVER), *driver_args]


def _tail(text: str | bytes | None) -> str:
    if isinstance(text, bytes):          # TimeoutExpired carries bytes even with text=True
        text = text.decode("utf-8", errors="replace")
    return (text or "")[-TAIL_CHARS:]


def launch(*, max_steps: int, seed: int, cpu: bool, work_dir: Path) -> dict:
    """Run the driver under the launcher and return rank 0's record with the launcher's output attached."""
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "record.json"
    out.unlink(missing_ok=True)
    driver_args = [
        "--out", str(out), "--max-steps", str(max_steps), "--seed", str(seed),
        "--attest", str(work_dir / "attest.jsonl"), "--manifest", str(work_dir / "manifest.jsonl"),
        "--output-dir", str(work_dir / "trainer_out"),
    ]
    command = launch_command(driver_args, cpu=cpu)
    python_path = os.pathsep.join(p for p in (str(_REPO), _SRC, os.environ.get("PYTHONPATH", "")) if p)
    env = {**os.environ, "PYTHONPATH": python_path, "NCCL_DEBUG": "WARN", "TOKENIZERS_PARALLELISM": "false"}
    started = time.time()
    try:
        proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"the launcher did not finish within {LAUNCH_TIMEOUT_S}s (a hung collective?)\n"
            f"--- stdout tail ---\n{_tail(exc.stdout)}\n--- stderr tail ---\n{_tail(exc.stderr)}"
        ) from exc
    launcher = {
        "command": shlex.join(command),
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 1),
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
    }
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"launcher exited {proc.returncode}, record {'written' if out.exists() else 'missing'}\n"
            f"--- stdout tail ---\n{launcher['stdout_tail']}\n--- stderr tail ---\n{launcher['stderr_tail']}"
        )
    record = json.loads(out.read_text())
    record["launcher"] = launcher
    return record


@app.function(gpu=GPU, image=image, volumes={"/hf_cache": hf_cache}, timeout=3600)
def run_distributed_remote(max_steps: int = 40, seed: int = 42) -> dict:
    sys.path.insert(0, "/reservoir_src")
    return launch(max_steps=max_steps, seed=seed, cpu=False, work_dir=Path("/tmp/reservoir_trl_distributed"))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def label_for_distributed(record: dict) -> str:
    device = "cpu" if record["config"]["device"] == "cpu" else "a10g"
    return f"{label_for(record)}_{device}-x{record['distributed']['world_size']}"


def report_distributed(record: dict) -> None:
    dist = record["distributed"]
    print(f"world_size={dist['world_size']} backend={dist['backend']} {dist['distributed_type']} "
          f"hook_calls_agree={dist['hook_calls_agree']} launcher_seconds={record['launcher']['seconds']}")
    for rank in dist["ranks"]:
        print(f"  rank {rank['rank']}: owner={rank['is_owner']} device={rank['device']} ({rank['device_name']}) "
              f"hook_calls={rank['hook_calls']} logprob_forwards={rank['logprob_forwards']}")


@app.local_entrypoint()
def main(max_steps: int = 40, seed: int = 42):
    results = run_distributed_remote.remote(max_steps=max_steps, seed=seed)
    report(results, write_results(results, label_for_distributed(results)))
    report_distributed(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="two CPU processes on gloo, without a Modal account")
    parser.add_argument("--local", action="store_true", required=True)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where to write results (default: a temporary directory)")
    cli = parser.parse_args()
    out_dir = cli.out_dir or Path(tempfile.mkdtemp(prefix="reservoir_trl_distributed_"))
    with tempfile.TemporaryDirectory(prefix="reservoir_trl_distributed_work_") as work:
        local = launch(max_steps=cli.max_steps, seed=cli.seed, cpu=True, work_dir=Path(work))
    report(local, write_results(local, label_for_distributed(local), out_dir))
    report_distributed(local)
