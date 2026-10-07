"""Committed staleness-sweep runs must keep verifying and must match their report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from checker.verify import verify_json_lines

RESULTS = Path(__file__).parents[1] / "benchmarks" / "modal" / "results"
LOGS = sorted(RESULTS.glob("sweep_*/*.attest.jsonl"))
REPORT = Path(__file__).parents[1] / "results" / "staleness_sweep_report.json"


@pytest.mark.parametrize("log", LOGS, ids=[f"{p.parent.name}/{p.name}" for p in LOGS])
def test_committed_sweep_log_verifies_with_its_manifest(log: Path):
    manifest = log.with_name(log.name.replace(".attest.jsonl", ".manifest.jsonl"))
    record = json.loads(log.with_name(log.name.replace(".attest.jsonl", ".json")).read_text())
    result = verify_json_lines(log.read_text(), manifest=manifest.read_text())
    records = [json.loads(line) for line in log.read_text().splitlines() if line]
    assert record["attestation"]["records"] == len(records)
    assert result.content.manifest_matched > 0
    assert record["totals"]["hook_calls"] == record["config"]["max_steps"]


@pytest.mark.skipif(not REPORT.exists(), reason="no sweep report committed")
def test_committed_report_matches_a_rebuild_from_the_committed_records():
    from benchmarks.staleness.report import build_report

    committed = json.loads(REPORT.read_text())
    assert committed["experiment"] == "staleness_sweep"
    rebuilt = build_report(Path(committed["run_dir"]))      # re-runs the checker on every committed log
    assert sorted(rebuilt["arms"]) == sorted(committed["arms"])
    for name, arm in rebuilt["arms"].items():
        assert [r["seed"] for r in arm["runs"]] == [r["seed"] for r in committed["arms"][name]["runs"]]
        assert arm["suspect_runs"] == committed["arms"][name]["suspect_runs"]
    assert rebuilt["totals"]["runs"] == committed["totals"]["runs"] == len(LOGS) + sum(
        1 for a in committed["arms"].values() if not a["replay"]) * len(committed["seeds"])
    assert rebuilt["verified_logs"] == len(LOGS)
