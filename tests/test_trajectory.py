"""Tests for reservoir.trajectory.TrajectoryLogger and TrajectoryFeatures."""

import pytest

from reservoir.trajectory import TrajectoryFeatures, TrajectoryLogger


def test_log_records_entry():
    logger = TrajectoryLogger(n_examples=5)
    logger.log(0, 1, 0.9)
    logger.log(0, 2, 0.8)
    logger.finalize(total_steps=10)
    feat = logger.get_features(0)
    assert feat is not None
    assert feat.n_observations == 2
    assert feat.loss_history == [0.9, 0.8]


def test_finalize_computes_slope_positive():
    logger = TrajectoryLogger(n_examples=5)
    # Increasing losses -> positive slope
    for step in range(5):
        logger.log(0, step, float(step) * 0.2)
    logger.finalize(total_steps=10)
    feat = logger.get_features(0)
    assert feat.slope > 0, f"Expected positive slope, got {feat.slope}"


def test_finalize_computes_slope_negative():
    logger = TrajectoryLogger(n_examples=5)
    # Decreasing losses -> negative slope
    for step in range(5):
        logger.log(0, step, 1.0 - float(step) * 0.2)
    logger.finalize(total_steps=10)
    feat = logger.get_features(0)
    assert feat.slope < 0, f"Expected negative slope, got {feat.slope}"


def test_finalize_variance_zero_for_constant():
    logger = TrajectoryLogger(n_examples=5)
    for step in range(5):
        logger.log(0, step, 0.5)
    logger.finalize(total_steps=10)
    feat = logger.get_features(0)
    assert feat.variance == pytest.approx(0.0, abs=1e-12)


def test_finalize_mean_last_k_uses_window():
    logger = TrajectoryLogger(n_examples=5, window_frac=0.5)
    # Steps 0-9, losses: 1.0 for first 5, 0.0 for last 5
    for step in range(5):
        logger.log(0, step, 1.0)
    for step in range(5, 10):
        logger.log(0, step, 0.0)
    logger.finalize(total_steps=10)
    feat = logger.get_features(0)
    # window cutoff = 10 * (1 - 0.5) = 5 -> steps >= 5 are in window (0.0 losses)
    assert feat.mean_loss_last_k == pytest.approx(0.0, abs=1e-9)


def test_first_correct_step_found():
    logger = TrajectoryLogger(n_examples=5, correct_threshold=0.5)
    logger.log(0, 0, 0.8)
    logger.log(0, 1, 0.8)
    logger.log(0, 2, 0.3)
    logger.finalize(total_steps=10)
    feat = logger.get_features(0)
    assert feat.first_correct_step == 2


def test_first_correct_step_none_when_never_correct():
    logger = TrajectoryLogger(n_examples=5, correct_threshold=0.5)
    logger.log(0, 0, 0.9)
    logger.log(0, 1, 0.8)
    logger.finalize(total_steps=10)
    feat = logger.get_features(0)
    assert feat.first_correct_step is None


def test_get_features_returns_none_for_unseen():
    logger = TrajectoryLogger(n_examples=5)
    logger.log(0, 0, 0.5)
    logger.finalize(total_steps=10)
    assert logger.get_features(3) is None


def test_finalize_required_before_get_all():
    logger = TrajectoryLogger(n_examples=5)
    logger.log(0, 0, 0.5)
    with pytest.raises(RuntimeError):
        logger.get_all_features()


def test_n_observations_counts_correctly():
    logger = TrajectoryLogger(n_examples=5)
    for step in range(7):
        logger.log(2, step, 0.5)
    logger.finalize(total_steps=20)
    feat = logger.get_features(2)
    assert feat.n_observations == 7


def test_summary_stats():
    logger = TrajectoryLogger(n_examples=5)
    logger.log(0, 0, 0.5)
    logger.log(1, 0, 0.5)
    logger.finalize(total_steps=10)
    stats = logger.summary_stats()
    assert stats["n_logged"] == 2
    assert stats["n_never_seen"] == 3
    assert stats["mean_observations"] == pytest.approx(1.0)


# --- residual_variance (detrended oscillation measure) ---

def test_residual_variance_zero_for_perfect_linear_decline():
    logger = TrajectoryLogger(n_examples=5)
    # Perfectly linear: variance is large but residual_variance ~ 0
    for step in range(6):
        logger.log(0, step, 1.0 - 0.15 * step)
    logger.finalize(total_steps=10)
    feat = logger.get_features(0)
    assert feat.variance > 0.05
    assert feat.residual_variance == pytest.approx(0.0, abs=1e-12)


def test_residual_variance_equals_variance_for_trendless_oscillation():
    logger = TrajectoryLogger(n_examples=5)
    # Oscillation around 0.5 with no trend: residual variance ~ full variance
    losses = [0.9, 0.1, 0.9, 0.1, 0.9, 0.1]
    for step, loss in enumerate(losses):
        logger.log(0, step, loss)
    logger.finalize(total_steps=10)
    feat = logger.get_features(0)
    assert feat.residual_variance == pytest.approx(feat.variance, rel=0.15)
    assert feat.residual_variance > 0.1


def test_residual_variance_zero_when_too_few_points():
    logger = TrajectoryLogger(n_examples=5)
    logger.log(0, 0, 0.9)
    logger.log(0, 1, 0.1)
    logger.finalize(total_steps=10)
    feat = logger.get_features(0)
    # 2 points always fit a line exactly
    assert feat.residual_variance == pytest.approx(0.0, abs=1e-12)


def test_residual_variance_separates_oscillator_from_converger():
    logger = TrajectoryLogger(n_examples=5)
    # Example 0: clean converger (linear decline)
    for step in range(6):
        logger.log(0, step, 1.0 - 0.15 * step)
    # Example 1: flipped-style oscillator (same overall variance scale)
    for step, loss in enumerate([0.8, 0.2, 0.9, 0.1, 0.7, 0.3]):
        logger.log(1, step, loss)
    logger.finalize(total_steps=10)
    assert logger.get_features(1).residual_variance > 10 * logger.get_features(0).residual_variance
