"""
benchmarks/attestation_overhead.py — What attestation costs, and what checking costs.

Measures ``RolloutBuffer`` throughput for ``add_group`` and ``sample`` under
four configurations:

1. ``none``: no attestation.
2. ``memory``: attestation into an in-memory ``AttestationLog``.
3. ``file``: attestation mirrored to a JSON-lines file (flushed per record).
4. ``file+manifest``: the file plus the manifest of every insert.

and then the cost of the other side: ``python -m checker.verify`` (with the
manifest) and ``python -m checker.transcript`` on the log that run 4
produced, as seconds per 10k records. Log and manifest sizes are reported
as bytes per record and per rollout.

These numbers describe the cost of reproducibility and verification; they
say nothing about training quality (docs/nonclaims.md). The buffer is pure
Python with big-integer arithmetic, so the absolute throughput is modest
by design; the useful reading is the *ratio* between configurations.

Usage::

    uv run python -m benchmarks.attestation_overhead                 # default scale, writes results/attestation_overhead.json
    uv run python -m benchmarks.attestation_overhead --groups 200 --samples 20 --out /tmp/x.json   # a quick check

Method. One discarded warm-up pass per configuration, then ``--repeats``
passes with the configurations interleaved (none, memory, file,
file+manifest, none, ...) so drift on the machine lands on every
configuration alike. The garbage collector is collected before and
disabled during each timed loop. Every call is timed on its own: the
report gives the mean per rollout or per draw (the whole-pass cost) and
the median call, each as the median over the repeats, with the range
over the repeats under ``spread`` and the machine's load average before
and after: on a shared machine the ratios move between runs, and the
report says by how much.

Where a content digest is paid. A sampled batch carries the content
digest of every row. An attested buffer computes a group's digests when
it inserts the group, so that cost is in its ``add`` column; an
unattested buffer computes them on the first sample that touches the
group, so the same cost is in its ``sample`` column. That is why the
unattested sample mean can exceed the attested one. A second run of
``--samples`` calls on the same buffer, after every sampled group has
been digested, is reported as ``sample_us_per_draw_steady``.

Run from the repository root with the ``reservoir`` package installed.
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
from typing import Callable, Optional

from benchmarks.provenance import stamped
from checker.transcript import build_transcript
from checker.verify import verify_json_lines
from reservoir.attest import AttestationLog
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer

REPO = Path(__file__).resolve().parents[1]

CONFIGS = ("none", "memory", "file", "file+manifest")
DEFAULT_OUT = REPO / "results" / "attestation_overhead.json"


def _rollouts(group_size: int, n_tokens: int, version: int) -> list[Rollout]:
    """Deterministic synthetic rollouts; rewards alternate so every group is live."""
    return [
        Rollout(tokens=[(version * 131 + k * 17 + t) % 50_000 for t in range(n_tokens)],
                logprobs=[-0.25] * n_tokens, reward=float(k % 3) / 2)
        for k in range(group_size)
    ]


def _buffer(config: str, workdir: Path, capacity: int) -> RolloutBuffer:
    kwargs = dict(capacity=capacity, half_life=8, max_policy_age=64, seed=0)
    if config == "none":
        return RolloutBuffer(**kwargs)
    if config == "memory":
        return RolloutBuffer(attest=AttestationLog(), **kwargs)
    attest = workdir / f"{config}.attest.jsonl"
    attest.unlink(missing_ok=True)
    if config == "file":
        return RolloutBuffer(attest=attest, **kwargs)
    manifest = workdir / f"{config}.manifest.jsonl"
    manifest.unlink(missing_ok=True)
    return RolloutBuffer(attest=attest, manifest=manifest, **kwargs)


def _timed(fn: Callable[[], None]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def _timed_calls(calls: list[Callable[[], None]]) -> list[float]:
    """Wall clock of each call, with the collector collected first and paused throughout."""
    gc.collect()
    gc.disable()
    try:
        return [_timed(call) for call in calls]
    finally:
        gc.enable()


def run_config(config: str, workdir: Path, *, groups: int, group_size: int, n_tokens: int,
               samples: int, batch_size: int, capacity: int) -> dict:
    """One pass: ``groups`` add_group calls (a version advance every 10), ``samples`` draws, then ``samples`` more."""
    buf = _buffer(config, workdir, capacity)
    batches = [(f"p{g}", g // 10, _rollouts(group_size, n_tokens, g // 10)) for g in range(groups)]
    adds = _timed_calls([lambda b=b: buf.add_group(b[0], b[1], b[2], source="bench") for b in batches])
    version = groups // 10
    draws = _timed_calls([lambda: buf.sample(batch_size, current_version=version)] * samples)
    result = {
        "config": config,
        "add_group_us_per_rollout": 1e6 * sum(adds) / (groups * group_size),
        "add_group_us_per_rollout_median_call": 1e6 * statistics.median(adds) / group_size,
        "sample_us_per_draw": 1e6 * sum(draws) / (samples * batch_size),
        "sample_us_per_draw_median_call": 1e6 * statistics.median(draws) / batch_size,
        "records": len(buf.attestation_log.records) if buf.attestation_log is not None else 0,
    }
    if config in ("file", "file+manifest"):
        attest = workdir / f"{config}.attest.jsonl"
        result["log_bytes"] = attest.stat().st_size
        result["log_bytes_per_record"] = attest.stat().st_size / max(result["records"], 1)
        result["log_bytes_per_rollout_inserted"] = attest.stat().st_size / (groups * group_size)
    if config == "file+manifest":
        manifest = workdir / f"{config}.manifest.jsonl"
        result["manifest_bytes"] = manifest.stat().st_size
        result["manifest_bytes_per_rollout_inserted"] = manifest.stat().st_size / (groups * group_size)
    # Sizes and counts above are of the log as the two timed phases left it; the steady pass adds records.
    steady = _timed_calls([lambda: buf.sample(batch_size, current_version=version)] * samples)
    result["sample_us_per_draw_steady"] = 1e6 * sum(steady) / (samples * batch_size)
    buf.close()
    return result


def check_costs(workdir: Path) -> dict:
    """Checker and transcript wall clock on the file+manifest log (steady pass included), per 10k records."""
    attest = (workdir / "file+manifest.attest.jsonl").read_text()
    manifest = (workdir / "file+manifest.manifest.jsonl").read_text()
    records = sum(1 for line in attest.splitlines() if line)
    verified = verify_json_lines(attest, manifest=manifest)          # warm-up, and the input of the transcript
    build_transcript(verified)
    verify_seconds, = _timed_calls([lambda: verify_json_lines(attest, manifest=manifest)])
    transcript_seconds, = _timed_calls([lambda: build_transcript(verified)])
    scale = 10_000 / max(records, 1)
    return {
        "records": records,
        "verify_with_manifest_seconds_per_10k_records": verify_seconds * scale,
        "transcript_seconds_per_10k_records": transcript_seconds * scale,
    }


def _median(rows: list[dict]) -> dict:
    """Per key, the median over the repeats, and for the timings their range under ``spread``."""
    out: dict = {}
    spread: dict = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        out[key] = values[0] if isinstance(values[0], str) else statistics.median(values)
        if "_us_" in key or key.endswith("_seconds_per_10k_records"):
            spread[key] = [min(values), max(values)]
    if spread:
        out["spread"] = spread
    return out


def _load_average() -> Optional[list[float]]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def run_benchmark(*, groups: int, group_size: int, n_tokens: int, samples: int, batch_size: int,
                  capacity: int, repeats: int, workdir: Optional[Path] = None) -> dict:
    workdir = workdir or Path(tempfile.mkdtemp(prefix="reservoir_bench_"))
    load_before = _load_average()
    workload = dict(groups=groups, group_size=group_size, n_tokens=n_tokens, samples=samples,
                    batch_size=batch_size, capacity=capacity)
    for config in CONFIGS:                                      # warm-up, discarded
        run_config(config, workdir, **workload)
    rows: dict[str, list[dict]] = {config: [] for config in CONFIGS}
    checks = []
    for _ in range(repeats):
        for config in CONFIGS:
            rows[config].append(run_config(config, workdir, **workload))
        checks.append(check_costs(workdir))                     # on the file+manifest log just written
    return stamped({
        "benchmark": "attestation_overhead",
        "workload": {"groups": groups, "group_size": group_size, "tokens_per_rollout": n_tokens,
                     "samples": samples, "batch_size": batch_size, "capacity": capacity,
                     "version_advance_every_groups": 10, "repeats": repeats, "warmup_passes": 1,
                     "gc": "collected before and disabled during each timed loop",
                     "order": "configurations interleaved within each repeat"},
        "load_average_1_5_15_min": {"before": load_before, "after": _load_average()},
        "configs": {config: _median(rows[config]) for config in CONFIGS},
        "checker": _median(checks),
        "note": ("cost of attestation and verification only; no training-quality measurement. An unattested "
                 "buffer computes a group's content digests on the first sample that touches it, an attested "
                 "one on insert, so that cost sits in different columns"),
    })


def render(report: dict) -> str:
    where = report["provenance"]
    lines = [f"attestation overhead (medians of {report['workload']['repeats']} interleaved repeats after a warm-up, "
             f"load {report['load_average_1_5_15_min']['before']}; "
             f"{where['machine']}, Python {where['python']}, reservoir {where['reservoir_version']} at {where['commit']})"]
    base = report["configs"]["none"]
    for name, row in report["configs"].items():
        ratio_add = row["add_group_us_per_rollout"] / base["add_group_us_per_rollout"]
        ratio_sample = row["sample_us_per_draw"] / base["sample_us_per_draw"]
        extra = ""
        if "log_bytes_per_record" in row:
            extra = f", log {row['log_bytes_per_record']:.0f} B/record"
        if "manifest_bytes_per_rollout_inserted" in row:
            extra += f", manifest {row['manifest_bytes_per_rollout_inserted']:.0f} B/rollout"
        lines.append(f"  {name:14s} add {row['add_group_us_per_rollout']:8.1f} us/rollout ({ratio_add:4.2f}x)"
                     f"  sample {row['sample_us_per_draw']:8.1f} us/draw ({ratio_sample:4.2f}x),"
                     f" steady {row['sample_us_per_draw_steady']:6.1f}{extra}")
    c = report["checker"]
    lines.append(f"  checker: verify+manifest {c['verify_with_manifest_seconds_per_10k_records']:.2f} s / 10k records, "
                 f"transcript {c['transcript_seconds_per_10k_records']:.2f} s / 10k records")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure what attestation and verification cost.")
    parser.add_argument("--groups", type=int, default=2000)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    report = run_benchmark(groups=args.groups, group_size=args.group_size, n_tokens=args.tokens,
                           samples=args.samples, batch_size=args.batch_size, capacity=args.capacity,
                           repeats=args.repeats)
    print(render(report))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
