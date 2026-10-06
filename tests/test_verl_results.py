"""The verl run records committed under benchmarks/modal/results must keep verifying.

Each ``verl_replay_*.attest.jsonl`` was produced by a real verl
``RayPPOTrainer`` run through ``ReservoirRayPPOTrainer``; the matching JSON
records the run and the ``.manifest.jsonl`` opens every content digest. The
independent checker is run on every committed log with its manifest so a
change to the log format or the checker cannot silently invalidate the
published evidence, and the record's own counters are checked against the
log and against each other.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from checker.verify import verify_json_lines

RESULTS = Path(__file__).parents[1] / "benchmarks" / "modal" / "results"
LOGS = sorted(RESULTS.glob("verl_replay_*.attest.jsonl"))


@pytest.mark.parametrize("log", LOGS, ids=[p.name for p in LOGS])
def test_committed_verl_replay_log_verifies_and_shows_replay(log: Path):
    record_path = log.with_name(log.name.replace(".attest.jsonl", ".json"))
    manifest_path = log.with_name(log.name.replace(".attest.jsonl", ".manifest.jsonl"))
    assert record_path.exists() and manifest_path.exists(), f"{log.name} lacks its record or manifest"
    text = log.read_text()

    result = verify_json_lines(text, manifest=manifest_path.read_text())

    records = [json.loads(line) for line in text.splitlines() if line]
    record = json.loads(record_path.read_text())
    totals, config = record["totals"], record["config"]
    assert record["attestation"]["records"] == len(records)
    assert record["attestation"]["checker"]["returncode"] == 0
    assert totals["hook_calls"] == config["max_steps"]
    assert totals["replaced_rows"] > 0, "the committed run never replayed anything"
    assert totals["replaced_rows"] == totals["dead_groups"] * config["n"]
    assert totals["declined_rows"] == 0 and totals["skipped_rows"] == 0
    assert sum(1 for r in records if r["op"] == "insert") == totals["ingested_rows"] == result.content.manifest_matched
    witnesses = [r for r in records if r["op"] == "batch"]
    assert len(witnesses) == sum(1 for s in record["steps"] if s["rows_rebuilt"])
    assert sum(len(w["replaced"]) for w in witnesses) == totals["replaced_rows"]
    assert all(w["batch_rows"] == config["train_batch_size"] * config["n"] for w in witnesses)
    # The buffer was snapshotted at every trainer checkpoint, under the same step numbers.
    trainer_steps = sorted(int(name.rsplit("_", 1)[1]) for name in record["trainer_checkpoints"])
    assert trainer_steps and sorted(int(tag.split("-")[1]) for tag in record["buffer_checkpoints"]) == trainer_steps
    assert config["use_v1"] is False and record["versions"]["verl"] == "0.9.1"
