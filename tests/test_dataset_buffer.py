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
