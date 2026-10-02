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
