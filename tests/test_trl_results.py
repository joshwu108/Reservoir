"""The attestation logs committed under benchmarks/modal/results must keep verifying.

Each ``trl_replay_*.attest.jsonl`` was produced by a real ``GRPOTrainer``
run through ``ReservoirGRPOTrainer``; the matching JSON records the run.
The independent checker is run on every committed log so a change to the
log format or the checker cannot silently invalidate the published evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from checker.verify import verify_json_lines

RESULTS = Path(__file__).parents[1] / "benchmarks" / "modal" / "results"
LOGS = sorted(RESULTS.glob("trl_replay_*.attest.jsonl"))


@pytest.mark.parametrize("log", LOGS, ids=[p.name for p in LOGS])
def test_committed_trl_replay_log_verifies_and_shows_replay(log: Path):
    record_path = log.with_name(log.name.replace(".attest.jsonl", ".json"))
    assert record_path.exists(), f"{log.name} has no matching run record"
    text = log.read_text()

    verify_json_lines(text)

    records = [json.loads(line) for line in text.splitlines() if line]
    record = json.loads(record_path.read_text())
    totals, config = record["totals"], record["config"]
    assert record["attestation"]["records"] == len(records)
    assert totals["hook_calls"] == config["max_steps"]
    assert totals["replaced_rows"] > 0, "the committed run never replayed anything"
    assert totals["replaced_rows"] == totals["dead_groups"] * config["num_generations"]
    assert sum(1 for r in records if r["op"] == "sample") > 0
    assert sum(1 for r in records if r["op"] == "insert") == totals["ingested_rows"]


def test_at_least_one_log_is_committed():
    assert LOGS, "no trl_replay_*.attest.jsonl under benchmarks/modal/results"


REPRO_DIRS = sorted(p for p in RESULTS.glob("repro_*") if p.is_dir())
REPRO_REPORT = Path(__file__).parents[1] / "results" / "reproducible_grpo_report.json"


@pytest.mark.parametrize("run_dir", REPRO_DIRS, ids=[p.name for p in REPRO_DIRS])
def test_committed_reproducibility_runs_verify_and_agree_with_their_report(run_dir: Path):
    """Each repro_* directory holds runs a, b, c with a log and a manifest each; a and b
    must be byte-identical, c must differ, and every log must open its manifest."""
    runs = {name: run_dir / name for name in ("a", "b", "c")}
    for path in runs.values():
        attest, manifest = path / "attest.jsonl", path / "manifest.jsonl"
        assert attest.exists() and manifest.exists(), f"{path} is missing its log or manifest"
        result = verify_json_lines(attest.read_text(), manifest=manifest.read_text())
        assert result.content.manifest_matched > 0
    assert (runs["a"] / "attest.jsonl").read_bytes() == (runs["b"] / "attest.jsonl").read_bytes()
    assert (runs["a"] / "manifest.jsonl").read_bytes() == (runs["b"] / "manifest.jsonl").read_bytes()
    assert (runs["a"] / "attest.jsonl").read_bytes() != (runs["c"] / "attest.jsonl").read_bytes()


def test_reproducibility_report_matches_committed_logs():
    if not REPRO_REPORT.exists():
        pytest.skip("results/reproducible_grpo_report.json not present")
    report = json.loads(REPRO_REPORT.read_text())
    assert report["verdict"] == "PASS" and report["same_seed_identical"] is True
    assert report["different_data_seed_class"] == "data"
    repo = Path(__file__).parents[1]
    for name, run in report["runs"].items():
        attest = repo / run["attest"]
        assert attest.exists(), f"run {name} log {run['attest']} is not committed"
        records = [json.loads(l) for l in attest.read_text().splitlines() if l]
        assert len(records) == run["records"]
        assert records[-1]["digest"] == run["head_digest"]
