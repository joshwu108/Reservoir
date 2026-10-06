"""benchmarks/modal/nccl_probe.py

Two probes for the layout TRL's vLLM server mode needs on Modal: trainer
on GPU 0, vLLM server on GPU 1, each process masked to its GPU with
``CUDA_VISIBLE_DEVICES``, weights pushed over NCCL. On 2026-10-05 the real
run hung in NCCL's warm-up all-reduce inside TRL's ``init_communicator``.

1. ``plain``: two bare processes, ``torch.distributed`` NCCL all-reduce.
   Passed with the defaults in 5 s (2026-10-06), so the GPU pair and the
   NCCL build are fine.
2. ``handshake``: the exact TRL path. Start ``vllm serve`` the way the
   tier does, then run ``VLLMClient.init_communicator`` in a separate
   trainer process, with ``NCCL_DEBUG=INFO`` on both sides and a hard
   deadline, trying each candidate environment in turn. The server log and
   the trainer log come back so the stall can be located.

Both probes run on the vLLM tier's image so they test the same torch,
NCCL and vLLM build.

Usage::

    modal run benchmarks/modal/nccl_probe.py                                   # both probes, transport round
    modal run benchmarks/modal/nccl_probe.py --which handshake --round-name identity
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import modal

import benchmarks.modal.reproducible_grpo_vllm as tier
from benchmarks.modal.reproducible_grpo_vllm import (
    GPU, GROUP_PORT, SERVER_CONFIG, SERVER_DEVICE, TRAINER_DEVICE, hf_cache, server_command, start_server,
    stop_server, vllm_image, wait_for_server,
)

app = modal.App("reservoir-nccl-probe")

CANDIDATES = {
    # Round 1 (2026-10-06): every one of these hung in the handshake; the
    # trainer's NCCL log said "nRanks 2 nNodes 2 localRanks 1", i.e. NCCL took
    # the two processes for two machines and went over the network path.
    "transports": [
        ("defaults", {}),
        ("p2p_off", {"NCCL_P2P_DISABLE": "1"}),
        ("p2p_shm_off", {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}),
        ("p2p_shm_off_cumem_off", {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1", "NCCL_CUMEM_ENABLE": "0"}),
    ],
    # Round 2: make NCCL see one host (NCCL_HOSTID overrides the hostname +
    # boot-id hash), or pin the socket interface it uses for the network path.
    "identity": [
        ("hostid", {"NCCL_HOSTID": "modal-shared-host"}),
        ("ifname", {"NCCL_SOCKET_IFNAME": "eth0", "NCCL_IB_DISABLE": "1"}),
        ("hostid_ifname", {"NCCL_HOSTID": "modal-shared-host", "NCCL_SOCKET_IFNAME": "eth0", "NCCL_IB_DISABLE": "1"}),
    ],
    # Round 3: same as before but NCCL writes its own per-process debug files
    # (stdout is block-buffered and lost when a hung process is killed), so
    # both ranks' init paths are visible. P2P off forces the SHM transport the
    # bare probe used.
    "files": [
        ("defaults", {}),
        ("hostid_p2p_off", {"NCCL_HOSTID": "modal-shared-host", "NCCL_P2P_DISABLE": "1"}),
    ],
    # Round 4: round 3 showed the trainer completing ncclCommInitRank (one
    # host, two local ranks, SHM transport) and then waiting for the worker's
    # SHM connection. vLLM forces NCCL_CUMEM_ENABLE=0 in its worker; the
    # trainer runs with the default. Ranks must agree on it, so try both ways.
    "cumem": [
        ("cumem_off", {"NCCL_CUMEM_ENABLE": "0"}),
        ("cumem_on", {"NCCL_CUMEM_ENABLE": "1"}),
    ],
}
NCCL_FILE_PATTERN = "/tmp/nccl_debug_%h_%p.log"
DEADLINE_S = 150
DEBUG_ENV = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET,P2P,SHM,GRAPH,BOOTSTRAP",
             "NCCL_DEBUG_FILE": NCCL_FILE_PATTERN}

PLAIN_WORKER = textwrap.dedent(
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

HANDSHAKE_TRAINER = textwrap.dedent(
    """
    import sys, time, torch
    from trl.generation.vllm_client import VLLMClient
    base_url, group_port = sys.argv[1], int(sys.argv[2])
    t = time.time()
    client = VLLMClient(base_url=base_url, group_port=group_port, connection_timeout=60)
    print("server reachable, world size", client.get_world_size(), flush=True)
    client.init_communicator(device=0)
    print(f"init_communicator ok in {time.time() - t:.1f}s", flush=True)
    client.update_named_param("probe", torch.ones(4, device="cuda"))
    print("one broadcast ok", flush=True)
    client.close_communicator()
    print("handshake complete", flush=True)
    """
)


def _wait(procs, deadline: float) -> list[bool]:
    started = time.time()
    while time.time() - started < deadline and any(p.poll() is None for p in procs):
        time.sleep(1)
    hung = [p.poll() is None for p in procs]
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait()
    return hung


def _nccl_lines(path: str, cap: int = 40_000) -> str:
    """The NCCL, error and warning lines of a log, whole, up to ``cap`` characters."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError as exc:
        return f"(no log: {exc})"
    keep = [l for l in text.splitlines()
            if any(k in l for k in ("NCCL", "nccl", "Error", "error", "WARN", "Traceback", "ok", "complete"))]
    return "\n".join(keep)[-cap:]


def _nccl_environ_of_processes() -> str:
    """NCCL_* and CUDA_VISIBLE_DEVICES of every live process, to compare the two ranks' settings."""
    out = []
    for proc_dir in sorted(Path("/proc").glob("[0-9]*")):
        try:
            cmd = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")[:90]
            env = (proc_dir / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        keys = sorted(e.decode(errors="replace") for e in env
                      if e.startswith(b"NCCL_") or e.startswith(b"CUDA_VISIBLE") or e.startswith(b"VLLM_"))
        if cmd.strip():
            out.append(f"{proc_dir.name}: {cmd}\n    " + " ".join(keys))
    return "\n".join(out)


def _tail(path: str, n: int = 2500) -> str:
    try:
        return Path(path).read_text(errors="replace")[-n:]
    except OSError as exc:
        return f"(no log: {exc})"


def plain(label: str, extra: dict[str, str], port: int) -> dict:
    procs, logs = [], []
    for rank in range(2):
        env = {**os.environ, **extra, **DEBUG_ENV, "CUDA_VISIBLE_DEVICES": str(rank)}
        log = open(f"/tmp/plain_{label}_{rank}.log", "wb")
        logs.append(log.name)
        procs.append(subprocess.Popen([sys.executable, "-c", PLAIN_WORKER, str(rank), str(port)],
                                      env=env, stdout=log, stderr=subprocess.STDOUT))
    started = time.time()
    hung = _wait(procs, DEADLINE_S)
    out = {"rank0": _nccl_lines(logs[0]), "rank1": _nccl_lines(logs[1])}
    for f in sorted(Path("/tmp").glob("nccl_debug_*.log")):
        out[f"nccl_file:{f.name}"] = _nccl_lines(str(f), cap=60_000)
        f.unlink()
    return {"probe": "plain", "label": label, "env": extra, "hung": hung,
            "ok": not any(hung) and all(p.returncode == 0 for p in procs),
            "returncodes": [p.returncode for p in procs], "seconds": round(time.time() - started, 1),
            "logs": out}


def handshake(label: str, extra: dict[str, str]) -> dict:
    """Start the tier's vLLM server with ``extra`` NCCL settings, then run TRL's init_communicator against it."""
    server_log = Path(f"/tmp/handshake_{label}_server.log")
    os.environ.update(extra)           # start_server copies os.environ; the trainer below gets it too
    os.environ.update(DEBUG_ENV)
    tier.NCCL_ENV.clear()              # the tier's own NCCL_DEBUG=WARN would override the probe's INFO in the server
    proc = start_server(server_log)
    started = time.time()
    try:
        wait_for_server(proc, server_log, SERVER_CONFIG["vllm_server_base_url"], 900.0)
        server_up = round(time.time() - started, 1)
        live_env = _nccl_environ_of_processes()
        env = {**os.environ, **extra, **DEBUG_ENV, "CUDA_VISIBLE_DEVICES": TRAINER_DEVICE}
        trainer_log = f"/tmp/handshake_{label}_trainer.log"
        trainer = subprocess.Popen(
            [sys.executable, "-c", HANDSHAKE_TRAINER, SERVER_CONFIG["vllm_server_base_url"], str(GROUP_PORT)],
            env=env, stdout=open(trainer_log, "wb"), stderr=subprocess.STDOUT,
        )
        t0 = time.time()
        hung = _wait([trainer], DEADLINE_S)
        logs = {"trainer": _nccl_lines(trainer_log), "server": _nccl_lines(str(server_log))}
        for f in sorted(Path("/tmp").glob("nccl_debug_*.log")):
            logs[f"nccl_file:{f.name}"] = _nccl_lines(str(f), cap=60_000)
            f.unlink()
        logs["identity"] = subprocess.run(
            ["bash", "-c", "echo hostname=$(hostname); echo boot_id=$(cat /proc/sys/kernel/random/boot_id); "
                           "echo shm=$(df -h /dev/shm | tail -1)"],
            capture_output=True, text=True).stdout
        logs["process_nccl_env"] = live_env
        return {"probe": "handshake", "label": label, "env": extra, "hung": hung,
                "ok": not hung[0] and trainer.returncode == 0, "returncodes": [trainer.returncode],
                "server_up_seconds": server_up, "seconds": round(time.time() - t0, 1), "logs": logs}
    finally:
        stop_server(proc)
        for k in {**extra, **DEBUG_ENV}:
            os.environ.pop(k, None)


@app.function(gpu=GPU, image=vllm_image, volumes={"/hf_cache": hf_cache}, timeout=2400)
def probe(which: str = "both", round_name: str = "transports") -> list[dict]:
    results = []
    candidates = CANDIDATES[round_name]
    if which in ("both", "plain"):
        for k, (label, extra) in enumerate(candidates):
            r = plain(label, extra, 29500 + k)
            results.append(r)
            print(f"plain/{label}: {'OK' if r['ok'] else 'FAILED'} {r['seconds']}s hung={r['hung']}", flush=True)
            if r["ok"]:
                break
    if which in ("both", "handshake"):
        for label, extra in candidates:
            r = handshake(label, extra)
            results.append(r)
            print(f"handshake/{label}: {'OK' if r['ok'] else 'FAILED'} {r['seconds']}s hung={r['hung']}", flush=True)
            if r["ok"]:
                break
    return results


@app.local_entrypoint()
def main(which: str = "both", round_name: str = "transports"):
    results = probe.remote(which, round_name)
    out = Path(__file__).parent / "results" / f"nccl_probe_{round_name}.json"
    out.write_text(json.dumps(results, indent=2))
    for r in results:
        print(f"== {r['probe']}/{r['label']} env={r['env']} ok={r['ok']} hung={r['hung']} rc={r['returncodes']} {r['seconds']}s")
        for name, tail in r["logs"].items():
            print(f"-- {name} log (last 1800 chars):\n{tail.strip()[-1800:]}\n")
    working = [r for r in results if r["probe"] == "handshake" and r["ok"]]
    print("WORKING HANDSHAKE ENV:", json.dumps(working[0]["env"]) if working else "none of the candidates")
    print(f"wrote {out}")
