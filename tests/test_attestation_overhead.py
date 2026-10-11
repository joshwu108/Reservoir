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
    # The checker reads the log after the steady pass, which adds one sample record per call.
    assert report["checker"]["records"] == report["configs"]["file+manifest"]["records"] + 3
    assert report["checker"]["verify_with_manifest_seconds_per_10k_records"] > 0
    for row in report["configs"].values():
        assert row["sample_us_per_draw_steady"] > 0 and row["sample_us_per_draw_median_call"] > 0
        low, high = row["spread"]["add_group_us_per_rollout"]
        assert low <= row["add_group_us_per_rollout"] <= high
    assert report["workload"]["warmup_passes"] == 1
    assert list(report)[0] == "provenance" and report["provenance"]["reservoir_version"]
    assert set(report["load_average_1_5_15_min"]) == {"before", "after"}


def test_cli_writes_json(tmp_path):
    out = tmp_path / "bench.json"
    assert main(["--groups", "12", "--group-size", "2", "--tokens", "4", "--samples", "2",
                 "--batch-size", "3", "--capacity", "16", "--repeats", "1", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["benchmark"] == "attestation_overhead"
    assert "training" in report["note"]
