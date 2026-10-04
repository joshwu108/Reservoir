"""
demo/reproducible_grpo.py — Two GRPO runs, one sampling transcript.

Runs the Phase 2 GRPO integration (``benchmarks/modal/trl_replay_real.py``:
``ReservoirGRPOTrainer`` on TRL's tiny Qwen2 test model, a rule-based
reward, replay of dead groups from a ``ReservoirReplay`` buffer) three
times on CPU:

- **a** and **b**: the same data seed and the same buffer seed. Their
  attestation logs and manifests must be byte-identical: the log is the
  fingerprint of everything the buffer stored, drew and evicted, and with
  deterministic generation (HF ``generate`` on CPU under a fixed seed, on
  this tiny model) the whole pipeline is reproducible.
- **c**: a different data seed. Its log must differ from **a**'s, and
  ``checker.diff`` must locate the first difference on an *insert* record
  and classify it ``data``: the generated completions differed upstream;
  Reservoir's draws did not.

Every log is verified with its manifest by the independent checker, the
transcript of run **a** reports exposure per source, and the whole
comparison is written to ``results/reproducible_grpo_report.json``. The
three logs and manifests themselves land under
``benchmarks/modal/results/repro_cpu_<steps>steps_seed<seed>/{a,b,c}/`` so
they can be committed as evidence and re-verified by the test suite.

Usage::

    uv run python -m demo.reproducible_grpo                    # 12 steps, ~1 min on a laptop
    uv run python -m demo.reproducible_grpo --max-steps 40
    uv run python -m demo.reproducible_grpo --out-dir /tmp/repro --no-report

What this does not show: GPU determinism. HF ``generate`` on a GPU is not
guaranteed to be bitwise reproducible; ``benchmarks/modal/reproducible_grpo_real.py``
runs the same comparison on a GPU. "PASS" means exactly: a and b have
byte-identical logs and manifests, and a and c first differ on an insert
record classified ``data``. No training-quality claim is made
(docs/nonclaims.md).

Run from the repository root with ``python -m demo.reproducible_grpo`` (the
``reservoir`` package must be installed, for example with ``uv sync``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from checker.diff import diff_logs, render_text as render_diff
from checker.transcript import build_transcript
from checker.verify import verify_json_lines

REPO = Path(__file__).resolve().parents[1]

REPORT_PATH = REPO / "results" / "reproducible_grpo_report.json"
EVIDENCE_DIR = REPO / "benchmarks" / "modal" / "results"


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in _resolve(path).read_text().splitlines() if line]


def _resolve(path: Path) -> Path:
    """Run records store repo-relative paths when they can; resolve them back."""
    return path if path.is_absolute() else REPO / path


def run_once(out_dir: Path, *, max_steps: int, seed: int, buffer_seed: int) -> dict:
    """One CPU GRPO run with replay; returns paths, head digest and totals.

    The log is verified with its manifest before this returns, so a run
    whose log the checker rejects fails here rather than in the comparison.
    """
    from benchmarks.modal.trl_replay_real import run_grpo

    import tempfile

    out_dir.mkdir(parents=True, exist_ok=True)
    attest, manifest = out_dir / "attest.jsonl", out_dir / "manifest.jsonl"
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="reservoir_trainer_") as trainer_dir:
        result = run_grpo(
            max_steps=max_steps, seed=seed, buffer_seed=buffer_seed, use_cpu=True,
            attest_path=str(attest), manifest_path=str(manifest), output_dir=trainer_dir,
        )
    verified = verify_json_lines(attest.read_text(), manifest=manifest.read_text())
    return {
        "attest": str(attest.relative_to(REPO)) if attest.is_relative_to(REPO) else str(attest),
        "manifest": str(manifest.relative_to(REPO)) if manifest.is_relative_to(REPO) else str(manifest),
        "seed": seed,
        "buffer_seed": buffer_seed,
        "head_digest": result["attestation"]["head_digest"],
        "records": result["attestation"]["records"],
        "examples_committed": len(verified.content.history),
        "totals": result["totals"],
        "versions": result["versions"],
        "wall_clock_seconds": round(time.time() - started, 2),
    }


def compare(runs: dict[str, dict]) -> dict:
    """Pairwise diffs: a vs b must be identical, a vs c must differ in the data."""
    logs = {name: _records(Path(run["attest"])) for name, run in runs.items()}
    return {
        "a_vs_b": diff_logs(logs["a"], logs["b"]),
        "a_vs_c": diff_logs(logs["a"], logs["c"]),
    }


def _same_bytes(run_a: dict, run_b: dict) -> bool:
    return all(
        _resolve(Path(run_a[key])).read_bytes() == _resolve(Path(run_b[key])).read_bytes()
        for key in ("attest", "manifest")
    )


def build_report(runs: dict[str, dict], diffs: dict[str, dict], max_steps: int) -> dict:
    transcript = build_transcript(verify_json_lines(
        _resolve(Path(runs["a"]["attest"])).read_text(), manifest=_resolve(Path(runs["a"]["manifest"])).read_text(),
    ))
    identical = diffs["a_vs_b"]["identical"] and _same_bytes(runs["a"], runs["b"])
    first = diffs["a_vs_c"]["first_difference"]
    data_diverged = (
        diffs["a_vs_c"]["class"] == "data"
        and first is not None
        and _records(Path(runs["a"]["attest"]))[first]["op"] == "insert"
    )
    return {
        "demo": "reproducible_grpo",
        "device": "cpu",
        "max_steps": max_steps,
        "versions": runs["a"]["versions"],
        "runs": runs,
        "same_seed_identical": identical,
        "different_data_seed_class": diffs["a_vs_c"]["class"],
        "diffs": diffs,
        "transcript_a": {
            "sampled_rows": transcript["sampled_rows"],
            "sources": transcript["sources"],
            "examples_committed": len(transcript["content"]),
        },
        "verdict": "PASS" if identical and data_diverged else "FAIL",
        "verdict_means": "a and b have byte-identical logs and manifests; a and c first differ on an "
                         "insert record that checker.diff classifies as data",
        "commands": {
            "verify": "python -m checker.verify <run>/attest.jsonl --manifest <run>/manifest.jsonl",
            "diff": "python -m checker.diff a/attest.jsonl b/attest.jsonl",
            "transcript": "python -m checker.transcript a/attest.jsonl --manifest a/manifest.jsonl --by source",
        },
    }


def print_report(report: dict) -> None:
    for name, run in report["runs"].items():
        print(f"run {name}: seed={run['seed']} buffer_seed={run['buffer_seed']} records={run['records']} "
              f"replaced_rows={run['totals']['replaced_rows']} head={run['head_digest'][:16]}… "
              f"({run['wall_clock_seconds']}s)")
    print()
    print("a vs b (same seeds):")
    print("  " + render_diff(report["diffs"]["a_vs_b"]).replace("\n", "\n  "))
    print("a vs c (different data seed):")
    print("  " + render_diff(report["diffs"]["a_vs_c"]).replace("\n", "\n  "))
    print()
    sources = ", ".join(f"{k}: {v['sampled']} rows" for k, v in report["transcript_a"]["sources"].items())
    print(f"transcript of a: {report['transcript_a']['sampled_rows']} sampled rows by source ({sources})")
    print(f"verdict: {report['verdict']} ({report['verdict_means']})")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42, help="data seed of runs a and b; c uses seed + 1")
    parser.add_argument("--buffer-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where the three runs go (default: benchmarks/modal/results/repro_cpu_<steps>steps_seed<seed>)")
    parser.add_argument("--no-report", action="store_true", help="do not write results/reproducible_grpo_report.json")
    args = parser.parse_args(argv)

    out_dir = args.out_dir or EVIDENCE_DIR / f"repro_cpu_{args.max_steps}steps_seed{args.seed}"
    runs = {
        "a": run_once(out_dir / "a", max_steps=args.max_steps, seed=args.seed, buffer_seed=args.buffer_seed),
        "b": run_once(out_dir / "b", max_steps=args.max_steps, seed=args.seed, buffer_seed=args.buffer_seed),
        "c": run_once(out_dir / "c", max_steps=args.max_steps, seed=args.seed + 1, buffer_seed=args.buffer_seed),
    }
    report = build_report(runs, compare(runs), args.max_steps)
    print_report(report)
    if not args.no_report:
        REPORT_PATH.parent.mkdir(exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2))
        print(f"wrote {REPORT_PATH}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
