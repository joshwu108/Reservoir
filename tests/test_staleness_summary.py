"""Tests for the committed sweep summary (``benchmarks/staleness/summary.py``)."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.staleness import summary


def _record(arm: str, seed: int, items: list[float]) -> dict:
    return {"arm": arm, "seed": seed, "evals": {"gsm8k_avg_at_k": [
        {"step": 150, "accuracy": 0.0, "per_item": [0.0] * len(items)},
        {"step": 300, "accuracy": sum(items) / len(items), "per_item": items}]}}


def test_paired_avg_at_k_takes_the_final_evaluation_per_seed_in_points():
    records = [_record("grpo", 42, [0.5, 0.5]), _record("grpo", 43, [0.25, 0.25]),
               _record("reservoir_age8_off", 42, [0.75, 0.5]), _record("reservoir_age8_off", 43, [0.25, 0.5])]

    out = summary.paired_avg_at_k(records, [("reservoir_age8_off", "grpo")], n_boot=200)

    comparison = out["reservoir_age8_off vs grpo"]
    assert comparison["per_seed_points"] == {42: 12.5, 43: 12.5}
    assert comparison["mean_points"] == 12.5
    assert comparison["seeds_above"] == 2 and comparison["n_seeds"] == 2
    assert comparison["lo_points"] <= 12.5 <= comparison["hi_points"]


def test_paired_avg_at_k_skips_a_comparison_without_shared_seeds():
    records = [_record("grpo", 42, [0.5]), _record("reservoir_age128_on", 44, [0.5])]

    assert summary.paired_avg_at_k(records, [("reservoir_age128_on", "grpo")]) == {}


def test_slim_report_drops_the_per_step_series_and_keeps_the_rest():
    report = {"arms": {"grpo": {"summary": {}}}, "per_step": {"grpo": {"42": {"reward": [0.1] * 300}}}, "seeds": [42]}

    slim = summary.slim_report(report)

    assert "per_step" not in slim and slim["arms"] == report["arms"] and slim["seeds"] == [42]
    assert "per_step" in report


def test_build_summary_reads_the_run_dir(tmp_path: Path):
    (tmp_path / "report.json").write_text(json.dumps({"arms": {}, "per_step": {}}))
    for rec in [_record("grpo", 42, [0.5]), _record("reservoir_age8_off", 42, [1.0])]:
        (tmp_path / f"{rec['arm']}_seed{rec['seed']}.json").write_text(json.dumps(rec))

    out = summary.build_summary(tmp_path, n_boot=100)

    assert out["paired_avg_at_k"]["reservoir_age8_off vs grpo"]["per_seed_points"] == {42: 50.0}
    assert out["avg_at_k_final"]["grpo"] == {42: 0.5}
