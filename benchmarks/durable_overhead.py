"""
benchmarks/durable_overhead.py — What durability costs, and that it stays flat.

Drives ``DurableRolloutBuffer`` for a long run and reports the wall clock
of ``add_group`` and ``sample`` in the first and the last tenth of the
run, the bytes the command log grows by per operation, and the cost of a
compaction, so a reader can see that per-operation cost does not grow with
history (the command log) while the snapshot does (the state, including
the attestation records and manifest lines). Also reports the in-memory
``RolloutBuffer`` on the same workload for the baseline.

Method. A short discarded warm-up run of each buffer, then ``--repeats``
runs in fresh directories, each buffer alternating with the other. The
garbage collector is collected before and disabled during each timed
run. Every figure is the median over the repeats of that run's figure,
with the range of the timings under ``spread`` and the machine's load
average before and after the runs.

Usage::

    uv run python -m benchmarks.durable_overhead                       # writes results/durable_overhead.json
    uv run python -m benchmarks.durable_overhead --groups 200 --repeats 1 --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from benchmarks.provenance import stamped
from reservoir.attest import AttestationLog
from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "results" / "durable_overhead.json"
WARMUP_GROUPS = 50


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


def _paused_gc(fn):
    gc.collect()
    gc.disable()
    try:
        return fn()
    finally:
        gc.enable()


def _one_repeat(workdir: Path, *, groups: int, group_size: int, n_tokens: int, batch_size: int, capacity: int,
                compact_every: int) -> tuple[dict, dict, float]:
    kwargs = dict(capacity=capacity, half_life=8, max_policy_age=64, seed=0)
    workload = dict(groups=groups, group_size=group_size, n_tokens=n_tokens, batch_size=batch_size)
    memory = RolloutBuffer(attest=AttestationLog(), **kwargs)
    in_memory = _paused_gc(lambda: run(memory, durable=False, **workload))
    files = dict(attest=workdir / "attest.jsonl", manifest=workdir / "manifest.jsonl", compact_every=compact_every)
    durable = DurableRolloutBuffer(workdir / "buf", **files, **kwargs)
    on_disk = _paused_gc(lambda: run(durable, durable=True, **workload))
    durable.close()
    on_disk["snapshot_bytes"] = (workdir / "buf" / "state.json").stat().st_size
    reopen_seconds = _paused_gc(lambda: _timed(lambda: DurableRolloutBuffer(workdir / "buf", **files, **kwargs).close()))
    return in_memory, on_disk, reopen_seconds


def _median(rows: list[dict]) -> dict:
    """Per key, the median over the repeats (counts and byte sizes stay integers: the lower median),
    and for the timings their range over the repeats under ``spread``."""
    out: dict = {}
    spread: dict = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        if any(v is None for v in values):
            out[key] = None
        elif all(isinstance(v, int) for v in values):
            out[key] = statistics.median_low(values)
        else:
            out[key] = statistics.median(values)
        if "_ms_" in key and out[key] is not None:
            spread[key] = [min(values), max(values)]
    if spread:
        out["spread"] = spread
    return out


def _load_average() -> Optional[list[float]]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def run_benchmark(*, groups: int, group_size: int, n_tokens: int, batch_size: int, capacity: int,
                  compact_every: int, repeats: int = 1, workdir: Optional[Path] = None) -> dict:
    workdir = workdir or Path(tempfile.mkdtemp(prefix="reservoir_durable_bench_"))
    load_before = _load_average()
    shape = dict(group_size=group_size, n_tokens=n_tokens, batch_size=batch_size, capacity=capacity,
                 compact_every=compact_every)
    _one_repeat(workdir / "warmup", groups=min(groups, WARMUP_GROUPS), **shape)
    runs = [_one_repeat(workdir / f"repeat{k}", groups=groups, **shape) for k in range(repeats)]
    return stamped({
        "benchmark": "durable_overhead",
        "workload": {"groups": groups, "group_size": group_size, "tokens_per_rollout": n_tokens,
                     "batch_size": batch_size, "capacity": capacity, "compact_every": compact_every,
                     "version_advance_every_groups": 10, "repeats": repeats,
                     "warmup_groups": min(groups, WARMUP_GROUPS),
                     "gc": "collected before and disabled during each timed run"},
        "load_average_1_5_15_min": {"before": load_before, "after": _load_average()},
        "in_memory_attested": _median([m for m, _, _ in runs]),
        "durable": _median([d for _, d, _ in runs]),
        "reopen_seconds": statistics.median(r for _, _, r in runs),
        "reopen_seconds_spread": [min(r for _, _, r in runs), max(r for _, _, r in runs)],
        "note": "cost of durability only; durability itself is evidenced by the crash campaign, not here",
    })


def render(report: dict) -> str:
    m, d, where = report["in_memory_attested"], report["durable"], report["provenance"]
    return "\n".join([
        f"durable overhead ({where['machine']}, Python {where['python']}, reservoir {where['reservoir_version']} "
        f"at {where['commit']}; {report['workload']['groups']} groups, compact every "
        f"{report['workload']['compact_every']}, medians of {report['workload']['repeats']})",
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
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    report = run_benchmark(groups=args.groups, group_size=args.group_size, n_tokens=args.tokens,
                           batch_size=args.batch_size, capacity=args.capacity, compact_every=args.compact_every,
                           repeats=args.repeats)
    print(render(report))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
