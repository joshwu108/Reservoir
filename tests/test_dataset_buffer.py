"""Tests for reservoir.dataset_buffer.DatasetBuffer."""

import random
import warnings

import numpy as np
import pytest

from reservoir.dataset_buffer import DatasetBuffer


def _make_dataset(n: int) -> list[dict]:
    return [{"text": f"example_{i}", "label": i % 2} for i in range(n)]


def test_audit_mode_len():
    ds = _make_dataset(20)
    buf = DatasetBuffer(ds)
    assert len(buf) == 20


def test_audit_mode_getitem_has_index():
    ds = _make_dataset(10)
    buf = DatasetBuffer(ds)
    item = buf[3]
    assert "__index__" in item
    assert item["__index__"] == 3
    assert item["text"] == "example_3"


def test_audit_mode_update_priority_stores_value():
    ds = _make_dataset(10)
    buf = DatasetBuffer(ds)
    buf.update_priority(2, 0.8)
    prios = buf.priorities
    assert prios[2] > prios[0]  # updated > default epsilon


def test_audit_mode_sample_indices_returns_correct_count():
    ds = _make_dataset(50)
    buf = DatasetBuffer(ds)
    indices = buf.sample_indices(8)
    assert len(indices) == 8
    assert all(0 <= i < 50 for i in indices)


def test_audit_mode_is_weights_all_ones():
    ds = _make_dataset(20)
    buf = DatasetBuffer(ds, mode="audit")
    indices = buf.sample_indices(5)
    weights = buf.get_is_weights(indices)
    assert len(weights) == 5
    assert all(w == 1.0 for w in weights)


def test_accelerated_mode_high_priority_sampled_more():
    random.seed(42)
    np.random.seed(42)
    ds = _make_dataset(10)
    # Use high priority_cap so the 1000x ratio is not capped away
    buf = DatasetBuffer(ds, mode="accelerated", priority_cap=1000.0)
    # Give index 0 a very high priority
    for i in range(10):
        buf.update_priority(i, 0.001)
    buf.update_priority(0, 1.0)  # 1000x others

    counts = [0] * 10
    for _ in range(1000):
        idxs = buf.sample_indices(1)
        counts[idxs[0]] += 1

    # index 0 should appear significantly more (>= 40% of samples)
    assert counts[0] >= 400, f"Expected index 0 to dominate, counts={counts}"


def test_priority_cap_limits_max_priority():
    ds = _make_dataset(20)
    buf = DatasetBuffer(ds, mode="accelerated", priority_cap=2.0)
    # Set a moderate baseline
    for i in range(20):
        buf.update_priority(i, 0.1)
    # Now try to set an extreme priority
    buf.update_priority(0, 1e9)
    prios = buf.priorities
    median_val = float(np.median(prios[prios > 0]))
    cap = 2.0 * median_val
    assert prios[0] <= cap + 1e-9, f"Priority {prios[0]} exceeds cap {cap}"


def test_priorities_property_length():
    ds = _make_dataset(15)
    buf = DatasetBuffer(ds)
    prios = buf.priorities
    assert len(prios) == 15


def test_large_dataset_warns():
    ds = _make_dataset(100_001)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        DatasetBuffer(ds)
        assert any("DatasetBuffer" in str(warning.message) for warning in w), \
            "Expected a DatasetBuffer warning for large dataset"


def test_size_property():
    ds = _make_dataset(7)
    buf = DatasetBuffer(ds)
    assert buf.size == 7


# ---------------------------------------------------------------------------
# Prompt-level priority strategies (Phase 1, Task 7)
# ---------------------------------------------------------------------------

from reservoir.priorities import (  # noqa: E402
    AdvantagePriority,
    PassRateTargeting,
    PassRateVariance,
    PromptPriority,
)
from reservoir.rollout import Rollout  # noqa: E402


def _rollouts(rewards: list[float]) -> list[Rollout]:
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in rewards]


def test_strategy_defaults_mode_to_accelerated():
    buf = DatasetBuffer(_make_dataset(4), priority=PassRateVariance())
    assert buf.mode == "accelerated"
    assert isinstance(buf.priority, PromptPriority)


def test_no_strategy_keeps_audit_default():
    buf = DatasetBuffer(_make_dataset(4))
    assert buf.mode == "audit"
    assert buf.priority is None


def test_explicit_mode_wins_over_strategy_default():
    buf = DatasetBuffer(_make_dataset(4), priority=PassRateVariance(), mode="audit")
    assert buf.mode == "audit"


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        DatasetBuffer(_make_dataset(4), mode="turbo")


def test_rollout_level_strategy_rejected():
    with pytest.raises(TypeError, match="PromptPriority"):
        DatasetBuffer(_make_dataset(4), priority=AdvantagePriority())


def test_update_group_sets_priority_to_the_strategy_score():
    strategy = PassRateVariance(epsilon=1e-6)
    buf = DatasetBuffer(_make_dataset(4), priority=strategy, priority_cap=1e9)
    buf.update_group(2, model_version=5, rollouts=_rollouts([1.0, 0.0, 1.0, 0.0]))
    assert buf.priorities[2] == pytest.approx(0.25 + 1e-6)


def test_update_group_honours_custom_success_predicate():
    buf = DatasetBuffer(_make_dataset(2), priority=PassRateTargeting(epsilon=0.0), priority_cap=1e9)
    buf.update_group(0, 0, _rollouts([0.9, 0.4]), is_success=lambda r: r.reward >= 0.5)
    assert buf.priorities[0] == pytest.approx(1.0)  # pass rate 0.5 hits the default target


def test_update_group_requires_a_strategy():
    buf = DatasetBuffer(_make_dataset(2))
    with pytest.raises(ValueError, match="priority"):
        buf.update_group(0, 0, _rollouts([1.0]))


def test_update_group_validates_index_and_rollouts():
    buf = DatasetBuffer(_make_dataset(2), priority=PassRateVariance())
    with pytest.raises(IndexError):
        buf.update_group(5, 0, _rollouts([1.0]))
    with pytest.raises(ValueError, match="index"):
        buf.update_group(1.5, 0, _rollouts([1.0]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rollouts"):
        buf.update_group(0, 0, [])


def test_update_group_respects_priority_cap():
    class Huge(PromptPriority):
        def score_prompt(self, group) -> float:
            return 1e6

    buf = DatasetBuffer(_make_dataset(3), priority=Huge(), priority_cap=2.0)
    buf.update_group(0, 0, _rollouts([1.0]))
    buf.update_group(1, 0, _rollouts([1.0]))
    # cap = 2 * median of positive raw priorities, so nothing can run away
    assert buf.priorities[1] <= 2.0 * np.median(buf.priorities[buf.priorities > 0]) + 1e-9


def test_update_priority_still_works_with_a_strategy():
    buf = DatasetBuffer(_make_dataset(2), priority=PassRateVariance())
    buf.update_priority(1, 0.5)
    assert buf.priorities[1] == pytest.approx(0.5 + 1e-6)


def test_cap_ignores_examples_never_updated():
    # 100 prompts at the epsilon placeholder must not cap the first real score.
    buf = DatasetBuffer(_make_dataset(100), mode="accelerated")
    buf.update_priority(0, 1.0)
    assert buf.priorities[0] == pytest.approx(1.0 + 1e-6)
    buf.update_priority(1, 1.0)
    buf.update_priority(2, 1000.0)  # capped against the median of the two updated
    assert buf.priorities[2] == pytest.approx(10.0 * (1.0 + 1e-6))


def test_score_below_epsilon_is_floored_and_tree_agrees():
    buf = DatasetBuffer(_make_dataset(2), priority=PassRateVariance(epsilon=0.0))
    buf.update_group(0, 0, _rollouts([1.0, 1.0]))  # score 0
    buf.update_group(1, 0, _rollouts([1.0, 0.0]))  # score 0.25
    assert buf.priorities[0] == pytest.approx(1e-6)
    leaf = lambda i: float(buf._per._tree[buf._per._tree_capacity - 1 + i])  # noqa: E731
    assert leaf(0) < leaf(1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.5", None, True])
def test_update_priority_rejects_non_finite_loss(bad):
    buf = DatasetBuffer(_make_dataset(2))
    with pytest.raises(ValueError, match="loss"):
        buf.update_priority(0, bad)


def test_positional_alpha_beta_still_work():
    buf = DatasetBuffer(_make_dataset(2), 0.6, 0.4)
    assert buf.priority is None


def test_bad_strategy_score_rejected_and_nothing_changes():
    class Bad(PromptPriority):
        def score_prompt(self, group) -> float:
            return float("nan")

    buf = DatasetBuffer(_make_dataset(2), priority=Bad())
    before = buf.priorities.copy()
    with pytest.raises(ValueError, match="Bad"):
        buf.update_group(0, 0, _rollouts([1.0]))
    assert np.array_equal(buf.priorities, before)


def test_pass_rate_variance_samples_uncertain_prompts_more():
    random.seed(7)
    np.random.seed(7)
    buf = DatasetBuffer(_make_dataset(3), priority=PassRateVariance(epsilon=1e-6), priority_cap=1e9)
    buf.update_group(0, 0, _rollouts([1.0, 1.0, 1.0, 1.0]))  # all pass  -> epsilon
    buf.update_group(1, 0, _rollouts([0.0, 0.0, 0.0, 0.0]))  # all fail  -> epsilon
    buf.update_group(2, 0, _rollouts([1.0, 0.0, 1.0, 0.0]))  # 50% pass -> 0.25
    counts = [0, 0, 0]
    for _ in range(2000):
        counts[buf.sample_indices(1)[0]] += 1
    assert counts[2] > 1800, counts
    assert counts[0] < 100 and counts[1] < 100, counts
