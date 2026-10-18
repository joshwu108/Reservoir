"""Committed staleness-sweep runs must keep verifying and must match their report.

Every ``benchmarks/modal/results/sweep_*/`` directory holds run records, the
Reservoir arms' logs and manifests, and the ``report.json`` built from them
(the CPU smoke's lives there; the GPU sweep's report is also copied to
``results/staleness_sweep_report.json``). Each log is re-verified, and each
report is rebuilt from its directory (which re-runs the checker on every
log) and compared with the committed one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from checker.verify import verify_json_lines

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "benchmarks" / "modal" / "results"
LOGS = sorted(RESULTS.glob("sweep_*/*.attest.jsonl"))
REPORTS = sorted(RESULTS.glob("sweep_*/report.json"))
TOP_REPORT = ROOT / "results" / "staleness_sweep_report.json"


@pytest.mark.parametrize("log", LOGS, ids=[f"{p.parent.name}/{p.name}" for p in LOGS])
def test_committed_sweep_log_verifies_with_its_manifest(log: Path):
    manifest = log.with_name(log.name.replace(".attest.jsonl", ".manifest.jsonl"))
    record = json.loads(log.with_name(log.name.replace(".attest.jsonl", ".json")).read_text())
    result = verify_json_lines(log.read_text(), manifest=manifest.read_text())
    records = [json.loads(line) for line in log.read_text().splitlines() if line]
    assert record["attestation"]["records"] == len(records)
    assert result.content.manifest_matched > 0
    assert record["totals"]["hook_calls"] == record["config"]["max_steps"]
    samples = [r for r in records if r["op"] == "sample"]
    assert samples and {r.get("min_entry_age") for r in samples} == {"1"}, "every draw must exclude same-step rows"


@pytest.mark.parametrize("path", REPORTS, ids=[p.parent.name for p in REPORTS])
def test_committed_report_matches_a_rebuild_from_its_directory(path: Path):
    from benchmarks.staleness.report import build_report

    committed = json.loads(path.read_text())
    assert committed["experiment"] == "staleness_sweep"
    rebuilt = build_report(path.parent)      # re-runs the checker on every committed log
    assert sorted(rebuilt["arms"]) == sorted(committed["arms"])
    for name, arm in rebuilt["arms"].items():
        old = committed["arms"][name]
        assert [r["seed"] for r in arm["runs"]] == [r["seed"] for r in old["runs"]]
        assert arm["suspect_runs"] == old["suspect_runs"]
        for field in ("eval_accuracy", "train_reward_last_quarter", "replaced_rows", "reward_auc"):
            assert arm["summary"][field] == old["summary"][field], (name, field)
        assert arm["eval_ci"] == old["eval_ci"] and arm["eval_diff_vs_grpo"] == old["eval_diff_vs_grpo"]
    runs = sorted(p for p in path.parent.glob("*.json") if p.name != "report.json")
    assert rebuilt["totals"]["runs"] == committed["totals"]["runs"] == len(runs)
    assert rebuilt["verified_logs"] == len(list(path.parent.glob("*.attest.jsonl")))
    assert all(p.exists() for p in path.parent.glob("*.attest.jsonl"))


@pytest.mark.skipif(not TOP_REPORT.exists(), reason="no GPU sweep report committed")
def test_top_level_report_is_a_copy_of_a_committed_sweep_report():
    top = json.loads(TOP_REPORT.read_text())
    matches = [p for p in REPORTS if json.loads(p.read_text())["run_dir"] == top["run_dir"]]
    assert matches, f"results/staleness_sweep_report.json names {top['run_dir']}, which has no committed report.json"
    assert top == json.loads(matches[0].read_text())
