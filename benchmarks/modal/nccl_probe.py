"""benchmarks/modal/nccl_probe.py

Which NCCL transport works between two processes on two GPUs in one Modal
container, each process seeing one GPU through ``CUDA_VISIBLE_DEVICES``?
That is the layout TRL's vLLM server mode needs (trainer on GPU 0, server
on GPU 1, weights pushed over NCCL), and on 2026-10-05 the trainer hung in
NCCL's warm-up all-reduce with the defaults. This probe runs on the vLLM
tier's image so it tests the same torch and NCCL build, and tries each
environment in turn with a hard deadline, so a hang is reported as such
instead of waiting for the function timeout.

Usage::

    modal run benchmarks/modal/nccl_probe.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time

import modal

from benchmarks.modal.reproducible_grpo_vllm import GPU, hf_cache, vllm_image

app = modal.App("reservoir-nccl-probe")

CANDIDATES = [
    ("defaults", {}),
    ("p2p_off", {"NCCL_P2P_DISABLE": "1"}),
    ("p2p_shm_off", {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}),
    ("p2p_shm_off_cumem_off", {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1", "NCCL_CUMEM_ENABLE": "0"}),
]
DEADLINE_S = 120

WORKER = textwrap.dedent(
    """
    import datetime, os, sys, torch, torch.distributed as dist
    rank, port = int(sys.argv[1]), sys.argv[2]
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=60), device_id=torch.device("cuda", 0))
    x = torch.ones(1 << 20, device="cuda") * (rank + 1)
    dist.all_reduce(x)
    torch.cuda.synchronize()
    assert float(x[0]) == 3.0, float(x[0])
    dist.destroy_process_group()
    print(f"rank {rank}: all_reduce ok on {torch.cuda.get_device_name(0)}")
    """
)


def try_env(label: str, extra: dict[str, str], port: int) -> dict:
    procs = []
    logs = []
    for rank in range(2):
        env = {**os.environ, **extra, "CUDA_VISIBLE_DEVICES": str(rank), "NCCL_DEBUG": "WARN"}
        log = open(f"/tmp/probe_{label}_{rank}.log", "wb")
        logs.append(log.name)
        procs.append(subprocess.Popen([sys.executable, "-c", WORKER, str(rank), str(port)],
                                      env=env, stdout=log, stderr=subprocess.STDOUT))
    started = time.time()
    while time.time() - started < DEADLINE_S and any(p.poll() is None for p in procs):
        time.sleep(1)
    hung = [p.poll() is None for p in procs]
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait()
    tails = [open(l).read()[-1500:] for l in logs]
    ok = not any(hung) and all(p.returncode == 0 for p in procs)
    return {"label": label, "env": extra, "ok": ok, "hung": hung,
            "returncodes": [p.returncode for p in procs], "seconds": round(time.time() - started, 1),
            "logs": tails}


@app.function(gpu=GPU, image=vllm_image, volumes={"/hf_cache": hf_cache}, timeout=1200)
def probe() -> list[dict]:
    results = []
    for k, (label, extra) in enumerate(CANDIDATES):
        result = try_env(label, extra, 29500 + k)
        results.append(result)
        print(f"{label}: {'OK' if result['ok'] else 'FAILED'} in {result['seconds']}s hung={result['hung']}")
        if result["ok"]:
            break
    return results


@app.local_entrypoint()
def main():
    results = probe.remote()
    for r in results:
        print(f"== {r['label']} env={r['env']} ok={r['ok']} hung={r['hung']} rc={r['returncodes']} {r['seconds']}s")
        for rank, tail in enumerate(r["logs"]):
            print(f"-- rank {rank} log tail:\n{tail.strip()[-800:]}")
    working = next((r for r in results if r["ok"]), None)
    print("WORKING ENV:", json.dumps(working["env"]) if working else "none of the candidates")
