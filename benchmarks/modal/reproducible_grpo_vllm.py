"""benchmarks/modal/reproducible_grpo_vllm.py

The reproducibility demo with vLLM generation in TRL's colocate mode and
vLLM's batch-invariant kernels (``VLLM_BATCH_INVARIANT=1``). Same three
runs as ``reproducible_grpo_real.py`` (**a**, **b** same seeds; **c** a
different data seed), same verification and the same report shape; the
expected verdict is ``IDENTICAL``.

What is different from the HF tier, and why
-------------------------------------------
- **Model.** ``Qwen/Qwen2.5-0.5B-Instruct``. TRL's tiny test model has an
  attention head size of 2, which no GPU attention kernel serves; vLLM
  needs a real architecture. The 0.5B model trains in float32 with
  gradient checkpointing next to a vLLM instance holding 30% of an A10G.
- **GPU.** ``A10G`` (compute capability 8.6). Batch invariance needs 8.0
  or higher; a T4 is 7.5.
- **TRL settings.** ``use_vllm=True, vllm_mode="colocate",
  vllm_importance_sampling_correction=False``. TRL attaches vLLM's
  sampling logprobs to every batch; with the correction off the loss never
  reads them and ``ReservoirReplay`` drops the key instead of refusing the
  batch (``docs/nonclaims.md`` §14).
- **Image.** vLLM is installed first and chooses its own torch; TRL 1.13.0
  accepts vLLM 0.19.1 through 0.28.0 (``trl/import_utils.py``), so the pin
  must lie in that range. ``RESERVOIR_VLLM_VERSION`` sets it; the default
  is the newest version TRL 1.13.0 accepts. Whether ``transformers==5.17.0``
  co-installs with that vLLM is not known in advance: this script is the
  time-boxed dependency search the Phase 4 plan calls for. If the image
  does not resolve, the HF tier stands on its own and the blocker is
  recorded.

Results land under
``benchmarks/modal/results/repro_vllm_a10g_<steps>steps_seed<seed>/``.

Usage
-----
    modal run benchmarks/modal/reproducible_grpo_vllm.py
    RESERVOIR_VLLM_VERSION=0.27.0 modal run benchmarks/modal/reproducible_grpo_vllm.py
    modal run benchmarks/modal/reproducible_grpo_vllm.py --max-steps 40

Cost estimate, not yet measured: the image build (vLLM is large) plus three
short training runs on an A10G at $1.10/h; about $1 to $3 if the image
resolves on the first try.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

from benchmarks.modal.reproducible_grpo_real import RESULTS_DIR, print_report, run_triplet, write_triplet
from benchmarks.modal.trl_replay_real import _repo_src, hf_cache

VLLM_VERSION = os.environ.get("RESERVOIR_VLLM_VERSION", "0.28.0")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GPU = "A10G"

VLLM_CONFIG = {
    "use_vllm": True,
    "vllm_mode": "colocate",
    "vllm_importance_sampling_correction": False,
    "vllm_gpu_memory_utilization": 0.3,
}

# vLLM first, so it pins the torch build it was compiled against; then the
# training stack at the versions the local test suite runs with.
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}")
    .pip_install("trl==1.13.0", "transformers==5.17.0", "datasets==5.0.1", "accelerate==1.15.0", "numpy==2.2.6")
    .env({"HF_HOME": "/hf_cache", "VLLM_BATCH_INVARIANT": "1"})
    .add_local_dir(_repo_src, remote_path="/reservoir_src")
)

app = modal.App("reservoir-reproducible-grpo-vllm")


@app.function(gpu=GPU, image=vllm_image, volumes={"/hf_cache": hf_cache}, timeout=3600)
def run_vllm(max_steps: int = 12, seed: int = 42, buffer_seed: int = 0) -> dict:
    return run_triplet(max_steps, seed, buffer_seed, extra_config=VLLM_CONFIG, model_id=MODEL_ID)


@app.local_entrypoint()
def main(max_steps: int = 12, seed: int = 42, buffer_seed: int = 0):
    results = run_vllm.remote(max_steps=max_steps, seed=seed, buffer_seed=buffer_seed)
    out_dir = RESULTS_DIR / f"repro_vllm_{GPU.lower()}_{max_steps}steps_seed{seed}"
    print_report(write_triplet(results, out_dir, f"vllm-{VLLM_VERSION}-batch-invariant"), out_dir)
