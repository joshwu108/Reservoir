"""Tests for reservoir.report.PreferenceQualityReport."""

import json
import os
import tempfile

import pytest

from reservoir.report import ExampleReport, NoiseLabel, PreferenceQualityReport
from reservoir.trajectory import TrajectoryFeatures


def _make_features(
    idx: int,
    slope: float,
    variance: float,
    mean_loss: float,
    n_obs: int = 10,
) -> TrajectoryFeatures:
    return TrajectoryFeatures(
        example_idx=idx,
        n_observations=n_obs,
        mean_loss_last_k=mean_loss,
        slope=slope,
        variance=variance,
        first_correct_step=None,
        loss_history=[mean_loss] * n_obs,
    )


def _make_report_with_examples():
    """Create a report with a mix of FLIPPED, AMBIGUOUS, and CLEAN examples."""
    # Use enough examples so percentile calculations are meaningful
    features = {}
    # FLIPPED: high slope AND high mean_loss
    features[0] = _make_features(0, slope=0.15, variance=0.01, mean_loss=0.9)
    features[1] = _make_features(1, slope=0.20, variance=0.01, mean_loss=0.85)
    # AMBIGUOUS: high variance AND near-zero slope
    features[2] = _make_features(2, slope=0.0, variance=0.5, mean_loss=0.5)
    features[3] = _make_features(3, slope=0.005, variance=0.6, mean_loss=0.5)
    # CLEAN: low slope + low mean_loss
    features[4] = _make_features(4, slope=-0.05, variance=0.01, mean_loss=0.1)
    features[5] = _make_features(5, slope=0.001, variance=0.01, mean_loss=0.1)
    return PreferenceQualityReport(features)


def test_flipped_bucket_assigned_correctly():
    report = _make_report_with_examples()
    flipped_idxs = {r.example_idx for r in report.flipped}
    assert 0 in flipped_idxs
    assert 1 in flipped_idxs


def test_ambiguous_bucket_assigned_correctly():
    report = _make_report_with_examples()
    ambiguous_idxs = {r.example_idx for r in report.ambiguous}
    assert 2 in ambiguous_idxs or 3 in ambiguous_idxs


def test_clean_bucket_assigned_correctly():
    report = _make_report_with_examples()
    clean_idxs = {r.example_idx for r in report.clean}
    assert 4 in clean_idxs or 5 in clean_idxs


def test_flipped_sorted_by_confidence_desc():
    report = _make_report_with_examples()
    flipped = report.flipped
    if len(flipped) >= 2:
        for i in range(len(flipped) - 1):
            assert flipped[i].confidence >= flipped[i + 1].confidence


def test_summary_percentages_sum_to_100():
    report = _make_report_with_examples()
    s = report.summary()
    total_pct = s["pct_flipped"] + s["pct_ambiguous"] + s["pct_clean"]
    assert abs(total_pct - 100.0) < 0.01, f"Percentages sum to {total_pct}"


def test_summary_counts_match():
    report = _make_report_with_examples()
    s = report.summary()
    assert s["n_total"] == 6
    assert s["n_flipped"] + s["n_ambiguous"] + s["n_clean"] == 6


def test_to_json_is_valid_json():
    report = _make_report_with_examples()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        report.to_json(path)
        with open(path) as f:
            data = json.load(f)
        assert "summary" in data
        assert "examples" in data
    finally:
        os.unlink(path)


def test_to_csv_has_correct_columns():
    report = _make_report_with_examples()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    try:
        report.to_csv(path)
        with open(path) as f:
            header = f.readline().strip()
        cols = [c.strip() for c in header.split(",")]
        for col in ["idx", "label", "confidence", "slope", "variance",
                    "mean_loss_last_k", "n_observations", "first_correct_step"]:
            assert col in cols, f"Missing column: {col}"
    finally:
        os.unlink(path)


def test_to_html_contains_html_tag():
    report = _make_report_with_examples()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        path = f.name
    try:
        report.to_html(path)
        with open(path) as f:
            content = f.read()
        assert "<html" in content.lower()
    finally:
        os.unlink(path)


def test_to_html_contains_summary():
    report = _make_report_with_examples()
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        path = f.name
    try:
        report.to_html(path)
        with open(path) as f:
            content = f.read()
        # Should contain summary table info
        assert "flipped" in content.lower() or "FLIPPED" in content
        assert "ambiguous" in content.lower() or "AMBIGUOUS" in content
    finally:
        os.unlink(path)


def test_empty_report_handles_gracefully():
    report = PreferenceQualityReport({})
    assert report.flipped == []
    assert report.ambiguous == []
    assert report.clean == []
    s = report.summary()
    assert s["n_total"] == 0
    assert s["n_flipped"] == 0
    assert s["n_ambiguous"] == 0
    assert s["n_clean"] == 0
