"""Tests for reservoir.anchor_set — written before implementation (TDD RED)."""

import pytest
import random
from reservoir.anchor_set import AnchorSet, AnchorExample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_examples(n=5):
    return [{"text": f"example {i}", "label": i % 2} for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_construction_from_list():
    examples = make_examples(5)
    anchor_set = AnchorSet(examples)
    assert len(anchor_set) == 5


def test_construction_with_tags():
    examples = make_examples(3)
    anchor_set = AnchorSet(examples, tags=["legal", "medical", "legal"])
    groups = anchor_set.groups()
    assert "legal" in groups
    assert "medical" in groups
    assert len(groups["legal"]) == 2
    assert len(groups["medical"]) == 1


def test_len():
    examples = make_examples(7)
    anchor_set = AnchorSet(examples)
    assert len(anchor_set) == 7


def test_getitem():
    examples = make_examples(4)
    anchor_set = AnchorSet(examples)
    item = anchor_set[0]
    assert isinstance(item, AnchorExample)
    assert item.idx == 0
    assert item.data == examples[0]


def test_groups_by_tag():
    examples = make_examples(6)
    tags = ["A", "B", "A", "C", "B", "A"]
    anchor_set = AnchorSet(examples, tags=tags)
    groups = anchor_set.groups()
    assert set(groups.keys()) == {"A", "B", "C"}
    assert len(groups["A"]) == 3
    assert len(groups["B"]) == 2
    assert len(groups["C"]) == 1


def test_snapshot_baseline_stores_losses():
    examples = make_examples(3)
    anchor_set = AnchorSet(examples)
    losses = {0: 1.0, 1: 2.0, 2: 0.5}
    anchor_set.snapshot_baseline(losses)
    for i, anchor in enumerate(anchor_set):
        assert anchor.baseline_loss == losses[i]


def test_update_current_losses_computes_priority():
    """baseline=0.5, current=1.0 -> priority = (1.0 - 0.5) / (0.5 + 1e-8) ≈ 1.0"""
    examples = make_examples(2)
    anchor_set = AnchorSet(examples)
    anchor_set.snapshot_baseline({0: 0.5, 1: 0.5})
    anchor_set.update_current_losses({0: 1.0, 1: 0.5})
    assert abs(anchor_set[0].priority - 1.0) < 1e-6
    assert anchor_set[1].priority == pytest.approx(0.0, abs=1e-6)


def test_priority_zero_when_no_forgetting():
    """baseline=0.5, current=0.4 -> priority = max(0, ...) = 0 (clamped)"""
    examples = make_examples(1)
    anchor_set = AnchorSet(examples)
    anchor_set.snapshot_baseline({0: 0.5})
    anchor_set.update_current_losses({0: 0.4})
    assert anchor_set[0].priority == 0.0


def test_forgetting_scores_by_group():
    examples = make_examples(4)
    tags = ["good", "good", "bad", "bad"]
    anchor_set = AnchorSet(examples, tags=tags)
    anchor_set.snapshot_baseline({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})
    # good group: no forgetting; bad group: loss doubled
    anchor_set.update_current_losses({0: 1.0, 1: 1.0, 2: 2.0, 3: 2.0})
    scores = anchor_set.forgetting_scores()
    assert scores["good"] == pytest.approx(0.0, abs=1e-6)
    assert scores["bad"] == pytest.approx(1.0, abs=1e-6)


def test_most_forgotten_sorted_by_priority():
    examples = make_examples(5)
    anchor_set = AnchorSet(examples)
    baseline = {i: 1.0 for i in range(5)}
    anchor_set.snapshot_baseline(baseline)
    # Priorities: 0->0, 1->0.5, 2->2.0, 3->1.0, 4->0.2
    anchor_set.update_current_losses({0: 1.0, 1: 1.5, 2: 3.0, 3: 2.0, 4: 1.2})
    top3 = anchor_set.most_forgotten(k=3)
    assert len(top3) == 3
    # Highest priority first: idx=2 (priority≈2.0), idx=3 (≈1.0), idx=1 (≈0.5)
    assert top3[0].idx == 2
    assert top3[1].idx == 3
    assert top3[2].idx == 1


def test_from_dataset_classmethod():
    examples = make_examples(20)
    anchor_set = AnchorSet.from_dataset(examples, n=10, tags="test")
    assert len(anchor_set) == 10
    for anchor in anchor_set:
        assert anchor.tag == "test"


def test_subsampling_with_n():
    random.seed(42)
    examples = make_examples(100)
    anchor_set = AnchorSet(examples, n=20)
    assert len(anchor_set) == 20


def test_iteration():
    examples = make_examples(3)
    anchor_set = AnchorSet(examples)
    items = list(anchor_set)
    assert len(items) == 3
    assert all(isinstance(item, AnchorExample) for item in items)
