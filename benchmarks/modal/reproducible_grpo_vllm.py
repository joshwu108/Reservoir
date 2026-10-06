"""benchmarks/modal/reproducible_grpo_vllm.py

The reproducibility demo with vLLM generation in batch-invariant mode
(``VLLM_BATCH_INVARIANT=1``). Same three runs as ``reproducible_grpo_real.py``
(**a**, **b** same seeds; **c** a different data seed), same verification
and the same report shape; the expected verdict is ``IDENTICAL``.

Why server mode, and why one container per run
-----------------------------------------------
vLLM's batch-invariant mode is inference-only: at start-up it replaces
PyTorch's CUDA ``linear``, ``mm``, ``addmm``, ``bmm`` and ``log_softmax``
kernels process-wide with its own deterministic ones, and those have no
backward. In TRL's ``colocate`` mode the trainer shares that process, so its
first backward pass fails with ``NotImplementedError: Could not run
'aten::linear_backward' with arguments from the 'CUDA' backend`` (observed
2026-10-05, vLLM 0.28.0, torch 2.14). So this tier uses TRL's ``server``
mode: vLLM runs as a separate process on its own GPU with the override, the
trainer runs on the other GPU with stock PyTorch, and the trainer pushes
its weights to the server over NCCL before each generation. NCCL needs the
two ranks on different devices, hence ``A10G:2``.

Each of the three runs gets its own container (they run in parallel), so
no server state, NCCL group or CUDA context carries over from one run to
the next; a and b being identical across two machines is a stronger test
than within one process.

What is different from the HF tier
----------------------------------
- **Model.** ``Qwen/Qwen2.5-0.5B-Instruct``. TRL's tiny test model has an
  attention head size of 2, which no GPU attention kernel serves; vLLM
  needs a real architecture. The trainer holds it in float32; the server
  serves it in bfloat16 and casts the synced weights.
- **GPU.** Two A10Gs (compute capability 8.6; batch invariance needs 8.0 or
  higher, a T4 is 7.5). GPU 0 trains, GPU 1 serves.
- **TRL settings.** ``use_vllm=True, vllm_mode="server",
  vllm_importance_sampling_correction=False``. TRL attaches vLLM's
  sampling logprobs to every batch; with the correction off the loss never
  reads them and ``ReservoirReplay`` drops the key instead of refusing the
  batch (``docs/nonclaims.md`` §14).
- **Image.** vLLM is installed first and chooses its own torch; TRL 1.13.0
  accepts vLLM 0.19.1 through 0.28.0 (``trl/import_utils.py``).
  ``RESERVOIR_VLLM_VERSION`` sets the pin; the default 0.28.0 co-installs
  with ``transformers==5.17.0`` (confirmed by the colocate attempt, which
  built the image and reached training).

Results land under
``benchmarks/modal/results/repro_vllm_a10g-x2_<steps>steps_seed<seed>/``.

Usage
-----
    modal run benchmarks/modal/reproducible_grpo_vllm.py
    RESERVOIR_VLLM_VERSION=0.27.0 modal run benchmarks/modal/reproducible_grpo_vllm.py
    modal run benchmarks/modal/reproducible_grpo_vllm.py --max-steps 40

Cost: three containers with two A10Gs each at about $2.20/h, each a few
minutes of server start-up plus a short training run; a few dollars.
"""

from __future__ import annotations

import faulthandler
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

from benchmarks.modal.reproducible_grpo_real import RESULTS_DIR, print_report, write_triplet
from benchmarks.modal.trl_replay_real import hf_cache, with_sources

VLLM_VERSION = os.environ.get("RESERVOIR_VLLM_VERSION", "0.28.0")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GPU = "A10G:2"
GPU_LABEL = "a10g-x2"
TRAINER_DEVICE, SERVER_DEVICE = "0", "1"
SERVER_HOST, SERVER_PORT, GROUP_PORT = "127.0.0.1", 8000, 51216
SERVER_LOG = Path("/tmp/vllm_server.log")
SERVER_START_TIMEOUT_S = 900.0
# A training run of twelve steps on this model takes well under this; if the
# trainer is still inside run_grpo after it, the process dumps every thread's
# stack to stderr and exits, so a hang costs minutes, not the function timeout.
RUN_DEADLINE_S = 900

# NCCL settings for BOTH the trainer and the server. They are the two ranks
# of the weight-transfer group; the first attempt (2026-10-05) hung in NCCL's
# warm-up all-reduce with the defaults. benchmarks/modal/nccl_probe.py finds
# a transport that works between two single-GPU-masked processes here.
NCCL_ENV: dict[str, str] = {
    "NCCL_DEBUG": "WARN",
}

# Merged into GRPOConfig for the trainer process.
SERVER_CONFIG = {
    "use_vllm": True,
    "vllm_mode": "server",
    "vllm_server_base_url": f"http://{SERVER_HOST}:{SERVER_PORT}",
    "vllm_group_port": GROUP_PORT,
    "vllm_server_timeout": 120.0,
    "vllm_importance_sampling_correction": False,
}

# vLLM first, so it pins the torch build it was compiled against; then the
# training stack at the versions the local test suite runs with. The
# batch-invariant switch is NOT on the image: it must reach only the server
# process (see the module docstring).
vllm_image = with_sources(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}")
    .pip_install("trl==1.13.0", "transformers==5.17.0", "datasets==5.0.1", "accelerate==1.15.0", "numpy==2.2.6")
    .env({"HF_HOME": "/hf_cache"})
)

app = modal.App("reservoir-reproducible-grpo-vllm")


def server_command() -> list[str]:
    """``vllm serve`` the way ``trl vllm-serve`` launches it (TRL 1.13 ``build_command``).

    The weight-transfer engine is what the trainer pushes weights through;
    processed logprobs and the lifted logprob cap are what TRL's client
    expects even though this run never reads them.
    """
    return [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL_ID,
        "--host", SERVER_HOST, "--port", str(SERVER_PORT),
        "--gpu-memory-utilization", "0.5", "--max-model-len", "1024", "--dtype", "bfloat16",
        "--enforce-eager",
        "--weight-transfer-config", json.dumps({"backend": "nccl"}),
        "--logprobs-mode", "processed_logprobs", "--max-logprobs", "-1",
        "--uvicorn-log-level", "warning",
    ]


def server_env() -> dict[str, str]:
    """The server's environment: its own GPU, batch invariance, and vLLM's dev-mode weight endpoints."""
    return {
        **os.environ,
        **NCCL_ENV,
        "CUDA_VISIBLE_DEVICES": SERVER_DEVICE,
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }


def start_server(log_path: Path) -> subprocess.Popen:
    log = open(log_path, "wb")
    return subprocess.Popen(server_command(), env=server_env(), stdout=log, stderr=subprocess.STDOUT)


def _health(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_for_server(proc: subprocess.Popen, log_path: Path, base_url: str, timeout: float) -> float:
    """Block until ``/health`` answers; raise with the log's tail if the server dies or the timeout passes."""
    started = time.time()
    while time.time() - started < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited with {proc.returncode} before becoming healthy:\n{_tail(log_path)}")
        if _health(f"{base_url}/health"):
            return time.time() - started
        time.sleep(2.0)
    raise RuntimeError(f"vLLM server not healthy after {timeout:.0f}s:\n{_tail(log_path)}")


def _tail(log_path: Path, lines: int = 60) -> str:
    try:
        return "\n".join(log_path.read_text(errors="replace").splitlines()[-lines:])
    except OSError as exc:
        return f"(no server log: {exc})"


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


@app.function(gpu=GPU, image=vllm_image, volumes={"/hf_cache": hf_cache}, timeout=2400)
def run_one(name: str, data_seed: int, buffer_seed: int, max_steps: int) -> dict:
    """One training run next to a fresh batch-invariant vLLM server; returns the run record."""
    os.environ["CUDA_VISIBLE_DEVICES"] = TRAINER_DEVICE   # before anything in this process touches CUDA
    os.environ.update(NCCL_ENV)
    sys.path.insert(0, "/reservoir_src")
    from benchmarks.modal.trl_replay_real import run_grpo

    proc = start_server(SERVER_LOG)
    try:
        server_seconds = wait_for_server(proc, SERVER_LOG, SERVER_CONFIG["vllm_server_base_url"], SERVER_START_TIMEOUT_S)
        started = time.time()
        faulthandler.dump_traceback_later(RUN_DEADLINE_S, exit=True)
        result = run_grpo(
            max_steps=max_steps, seed=data_seed, buffer_seed=buffer_seed,
            attest_path=f"/tmp/{name}.attest.jsonl", manifest_path=f"/tmp/{name}.manifest.jsonl",
            output_dir=f"/tmp/{name}_trainer", extra_config=SERVER_CONFIG, model_id=MODEL_ID,
        )
        faulthandler.cancel_dump_traceback_later()
        result["wall_clock_seconds"] = round(time.time() - started, 2)
        result["seed"], result["buffer_seed"] = data_seed, buffer_seed
        result["vllm_server"] = {
            "version": VLLM_VERSION, "mode": "server", "batch_invariant": True,
            "command": server_command()[1:], "start_seconds": round(server_seconds, 1), "nccl_env": NCCL_ENV,
            "log_tail": _tail(SERVER_LOG, 20),
        }
        return result
    finally:
        stop_server(proc)


@app.local_entrypoint()
def main(max_steps: int = 12, seed: int = 42, buffer_seed: int = 0):
    runs = [("a", seed), ("b", seed), ("c", seed + 1)]
    records = list(run_one.starmap([(name, data_seed, buffer_seed, max_steps) for name, data_seed in runs]))
    results = {name: record for (name, _), record in zip(runs, records)}
    out_dir = RESULTS_DIR / f"repro_vllm_{GPU_LABEL}_{max_steps}steps_seed{seed}"
    report = write_triplet(results, out_dir, f"vllm-{VLLM_VERSION}-batch-invariant-server")
    report["vllm_server"] = {k: v for k, v in results["a"]["vllm_server"].items() if k != "log_tail"}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print_report(report, out_dir)
