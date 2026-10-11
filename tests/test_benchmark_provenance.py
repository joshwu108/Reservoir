"""Every result JSON written under benchmarks/ carries version, commit, date and GPU name.

``benchmarks.provenance`` builds the block; these tests pin its fields,
the GPU name lookup across the record shapes the writers produce, the
committed overhead results, and (statically) that every file under
``benchmarks/`` that writes JSON stamps it, with the one documented
exception.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from benchmarks import provenance as prov

REPO = Path(__file__).resolve().parents[1]
FIELDS = {"reservoir_version", "commit", "changed_files", "commit_dirty", "date", "python", "machine", "gpu"}

# Writes the per-rank record the launcher reads back inside the container; the result file is written by
# trl_replay_real.write_results, which stamps it.
NOT_A_RESULT_WRITER = {"benchmarks/modal/trl_replay_driver.py"}


def test_provenance_has_every_field_and_a_utc_date():
    block = prov.provenance(gpu="NVIDIA A10")
    assert set(block) == FIELDS
    assert block["gpu"] == "NVIDIA A10"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", block["date"])
    from reservoir import __version__
    assert block["reservoir_version"] == __version__


def test_commit_is_head_in_a_checkout_and_none_without_git(monkeypatch):
    assert re.fullmatch(r"[0-9a-f]{40}", prov.git_commit() or "")
    monkeypatch.setattr(prov.shutil, "which", lambda name: None)
    assert prov.git_commit() is None and prov.git_changed_files() is None and prov.local_gpu_name() is None
    block = prov.provenance()
    assert block["commit"] is None and block["commit_dirty"] is None and block["gpu"] is None


def test_stamped_puts_provenance_first_and_replaces_an_old_block():
    report = {"provenance": {"stale": True}, "benchmark": "x", "value": 1}
    out = prov.stamped(report, gpu="T4")
    assert list(out)[0] == "provenance" and out["provenance"]["gpu"] == "T4"
    assert out["benchmark"] == "x" and "stale" not in out["provenance"]
    assert report["provenance"] == {"stale": True}                       # the input is not mutated


@pytest.mark.parametrize("record, expected", [
    ({"gpu_name": "NVIDIA A10G"}, "NVIDIA A10G"),
    ({"config": {"gpu_name": "Tesla T4"}}, "Tesla T4"),
    ({"versions": {"gpu": "Tesla T4"}}, "Tesla T4"),
    ({"distributed": {"ranks": [{"device_name": "NVIDIA A10"}, {"device_name": "NVIDIA A10"}]}}, "NVIDIA A10"),
    ({"config": {"gpu_name": None}, "distributed": {"ranks": [{"device_name": "cpu"}]}}, None),
    ({}, None),
])
def test_gpu_of_reads_every_record_shape(record, expected):
    assert prov.gpu_of(record) == expected


@pytest.mark.parametrize("name", ["attestation_overhead.json", "durable_overhead.json", "divergence_power.json"])
def test_committed_results_carry_provenance(name):
    block = json.loads((REPO / "results" / name).read_text())["provenance"]
    assert set(block) == FIELDS
    assert re.fullmatch(r"[0-9a-f]{40}", block["commit"])
    assert "gpu" in block


def test_every_json_writer_under_benchmarks_stamps_its_output():
    unstamped = []
    for path in sorted((REPO / "benchmarks").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        text = path.read_text()
        writes = re.search(r"\.write_text\(json\.dumps\(|json\.dump\(", text)
        if writes and rel not in NOT_A_RESULT_WRITER and "stamped(" not in text:
            unstamped.append(rel)
    assert unstamped == []
