"""The attestation-overhead benchmark runs at a tiny scale and writes a well-formed report."""

from __future__ import annotations

import json

from benchmarks.attestation_overhead import CONFIGS, main, run_benchmark


def test_tiny_benchmark_report_shape(tmp_path):
    report = run_benchmark(groups=12, group_size=2, n_tokens=4, samples=3, batch_size=4,
                           capacity=16, repeats=1, workdir=tmp_path)
    assert set(report["configs"]) == set(CONFIGS)
    for row in report["configs"].values():
        assert row["add_group_us_per_rollout"] > 0 and row["sample_us_per_draw"] > 0
    assert report["configs"]["none"]["records"] == 0
    assert report["configs"]["file"]["records"] == report["configs"]["memory"]["records"] > 0
    assert report["configs"]["file+manifest"]["manifest_bytes"] > 0
    assert report["checker"]["records"] == report["configs"]["file+manifest"]["records"]
    assert report["checker"]["verify_with_manifest_seconds_per_10k_records"] > 0


def test_cli_writes_json(tmp_path):
    out = tmp_path / "bench.json"
    assert main(["--groups", "12", "--group-size", "2", "--tokens", "4", "--samples", "2",
                 "--batch-size", "3", "--capacity", "16", "--repeats", "1", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["benchmark"] == "attestation_overhead"
    assert "training" in report["note"]
