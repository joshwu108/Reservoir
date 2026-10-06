"""
benchmarks/durable_overhead.py — What durability costs, and that it stays flat.

Drives ``DurableRolloutBuffer`` for a long run and reports the wall clock
of ``add_group`` and ``sample`` in the first and the last tenth of the
run, the bytes the command log grows by per operation, and the cost of a
compaction, so a reader can see that per-operation cost does not grow with
history (the command log) while the snapshot does (the state, including
the attestation records and manifest lines). Also reports the in-memory
``RolloutBuffer`` on the same workload for the baseline.

Usage::

    uv run python -m benchmarks.durable_overhead                       # writes results/durable_overhead.json
    uv run python -m benchmarks.durable_overhead --groups 200 --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from reservoir import __version__
from reservoir.attest import AttestationLog
from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "results" / "durable_overhead.json"


def _rollouts(group_size: int, n_tokens: int, version: int) -> list[Rollout]:
    return [
        Rollout(tokens=[(version * 131 + k * 17 + t) % 50_000 for t in range(n_tokens)],
                logprobs=[-0.25] * n_tokens, reward=float(k % 3) / 2)
        for k in range(group_size)
    ]


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def _measure(buf, fn, durable: bool) -> tuple[float, bool]:
    """Time one operation; report whether it ended in a compaction (the pending count fell)."""
    before = buf.pending_commands if durable else 0
    t = _timed(fn)
    return t, durable and buf.pending_commands < before


def run(buf, *, groups: int, group_size: int, n_tokens: int, batch_size: int, durable: bool) -> dict:
    """One run; returns per-operation timings for the first and last tenth and log growth."""
    adds: list[float] = []
    samples: list[float] = []
    wal_growth: list[int] = []
    compactions: list[float] = []
    for g in range(groups):
        version = g // 10
        rollouts = _rollouts(group_size, n_tokens, version)
        before_bytes = buf.wal_bytes if durable else 0
        t, compacted = _measure(buf, lambda: buf.add_group(f"p{g}", version, rollouts, source="bench"), durable)
        (compactions if compacted else adds).append(t)
        if durable and not compacted:
            wal_growth.append(buf.wal_bytes - before_bytes)
        t, compacted = _measure(buf, lambda: buf.sample(batch_size, current_version=version), durable)
        (compactions if compacted else samples).append(t)
    tenth = max(1, len(adds) // 10)
    return {
        "add_group_ms_first_tenth": 1e3 * statistics.median(adds[:tenth]),
        "add_group_ms_last_tenth": 1e3 * statistics.median(adds[-tenth:]),
        "sample_ms_first_tenth": 1e3 * statistics.median(samples[:tenth]),
        "sample_ms_last_tenth": 1e3 * statistics.median(samples[-tenth:]),
        "wal_bytes_per_add_group": statistics.median(wal_growth) if wal_growth else 0,
        "compactions": len(compactions),
        "compaction_ms_median": 1e3 * statistics.median(compactions) if compactions else None,
        "compaction_ms_last": 1e3 * compactions[-1] if compactions else None,
    }


def run_benchmark(*, groups: int, group_size: int, n_tokens: int, batch_size: int, capacity: int,
                  compact_every: int, workdir: Optional[Path] = None) -> dict:
    workdir = workdir or Path(tempfile.mkdtemp(prefix="reservoir_durable_bench_"))
    kwargs = dict(capacity=capacity, half_life=8, max_policy_age=64, seed=0)
    memory = RolloutBuffer(attest=AttestationLog(), **kwargs)
    in_memory = run(memory, groups=groups, group_size=group_size, n_tokens=n_tokens, batch_size=batch_size, durable=False)
    durable = DurableRolloutBuffer(workdir / "buf", attest=workdir / "attest.jsonl", manifest=workdir / "manifest.jsonl",
                                   compact_every=compact_every, **kwargs)
    on_disk = run(durable, groups=groups, group_size=group_size, n_tokens=n_tokens, batch_size=batch_size, durable=True)
    on_disk["snapshot_bytes"] = (workdir / "buf" / "state.json").stat().st_size
    reopen_seconds = _timed(lambda: DurableRolloutBuffer(
        workdir / "buf", attest=workdir / "attest.jsonl", manifest=workdir / "manifest.jsonl",
        compact_every=compact_every, **kwargs).close())
    durable.close()
    return {
        "benchmark": "durable_overhead",
        "reservoir_version": __version__,
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()}",
        "workload": {"groups": groups, "group_size": group_size, "tokens_per_rollout": n_tokens,
                     "batch_size": batch_size, "capacity": capacity, "compact_every": compact_every,
                     "version_advance_every_groups": 10},
        "in_memory_attested": in_memory,
        "durable": on_disk,
        "reopen_seconds": reopen_seconds,
        "note": "cost of durability only; durability itself is evidenced by the crash campaign, not here",
    }


def render(report: dict) -> str:
    m, d = report["in_memory_attested"], report["durable"]
    return "\n".join([
        f"durable overhead ({report['machine']}, Python {report['python']}, "
        f"{report['workload']['groups']} groups, compact every {report['workload']['compact_every']})",
        f"  in-memory add_group  first/last tenth: {m['add_group_ms_first_tenth']:.2f} / {m['add_group_ms_last_tenth']:.2f} ms",
        f"  durable   add_group  first/last tenth: {d['add_group_ms_first_tenth']:.2f} / {d['add_group_ms_last_tenth']:.2f} ms"
        f"  ({d['wal_bytes_per_add_group']:.0f} B appended per add)",
        f"  in-memory sample     first/last tenth: {m['sample_ms_first_tenth']:.2f} / {m['sample_ms_last_tenth']:.2f} ms",
        f"  durable   sample     first/last tenth: {d['sample_ms_first_tenth']:.2f} / {d['sample_ms_last_tenth']:.2f} ms",
        f"  compactions: {d['compactions']}, median {d['compaction_ms_median']} ms, last {d['compaction_ms_last']} ms, "
        f"snapshot {d['snapshot_bytes']} B; reopen {report['reopen_seconds']:.2f} s",
    ])


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure what durability costs and that it stays flat.")
    parser.add_argument("--groups", type=int, default=3000)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--compact-every", type=int, default=256)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    report = run_benchmark(groups=args.groups, group_size=args.group_size, n_tokens=args.tokens,
                           batch_size=args.batch_size, capacity=args.capacity, compact_every=args.compact_every)
    print(render(report))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
