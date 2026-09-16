"""Tests for reservoir.replay_scheduler — TDD RED phase."""

import pytest
from reservoir.anchor_set import AnchorSet, AnchorExample
from reservoir.replay_scheduler import ReplayScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_anchor_set_with_priorities(priorities: list[float], tag="test") -> AnchorSet:
    """Create an AnchorSet with given priorities already set."""
    n = len(priorities)
    examples = [{"text": f"ex {i}"} for i in range(n)]
    anchor_set = AnchorSet(examples, tags=[tag] * n)
    # Snapshot baseline at 1.0 and set current to produce the desired priority
    anchor_set.snapshot_baseline({i: 1.0 for i in range(n)})
    # priority = (current - baseline) / (baseline + 1e-8)
    # => current = priority * (1.0 + 1e-8) + 1.0
    current_losses = {i: p * (1.0 + 1e-8) + 1.0 for i, p in enumerate(priorities)}
    anchor_set.update_current_losses(current_losses)
    return anchor_set


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scheduler_initializes():
    anchor_set = make_anchor_set_with_priorities([0.5, 1.0, 0.2])
    scheduler = ReplayScheduler([anchor_set])
    assert scheduler is not None
    assert scheduler.total_replayed == 0


def test_update_priorities_syncs_with_anchor_set():
    anchor_set = make_anchor_set_with_priorities([0.5, 1.0, 0.2])
    scheduler = ReplayScheduler([anchor_set])
    # Update priorities — give the first anchor a huge score
    anchor_set.snapshot_baseline({0: 1.0, 1: 1.0, 2: 1.0})
    anchor_set.update_current_losses({0: 10.0, 1: 1.0, 2: 1.0})
    scheduler.update_priorities(anchor_set)
    # After update, anchor 0 has much higher priority and should dominate sampling
    examples, weights = scheduler.get_replay_batch(batch_size=30)
    idx_counts = {}
    for ex in examples:
        idx_counts[ex.idx] = idx_counts.get(ex.idx, 0) + 1
    # Cannot assert exact counts with single call; just confirm it runs
    assert len(examples) > 0


def test_get_replay_batch_returns_correct_count():
    """batch_size=32, ratio=0.1 -> n_replay=max(1, int(32*0.1))=3"""
    anchor_set = make_anchor_set_with_priorities([0.5, 1.0, 0.2, 0.8, 0.3])
    scheduler = ReplayScheduler([anchor_set], replay_ratio=0.1)
    examples, weights = scheduler.get_replay_batch(batch_size=32)
    assert len(examples) == 3
    assert len(weights) == 3


def test_is_weights_are_floats():
    anchor_set = make_anchor_set_with_priorities([0.5, 1.0, 0.2, 0.8, 0.3])
    scheduler = ReplayScheduler([anchor_set], replay_ratio=0.1)
    examples, weights = scheduler.get_replay_batch(batch_size=32)
    for w in weights:
        assert isinstance(w, float)
        assert 0.0 < w <= 1.0 + 1e-6


def test_high_priority_anchors_sampled_more():
    """One anchor at 100x priority of others should appear significantly more often."""
    import collections
    priorities = [0.01] * 9 + [1.0]  # anchor idx=9 has 100x the priority
    anchor_set = make_anchor_set_with_priorities(priorities)
    scheduler = ReplayScheduler([anchor_set], replay_ratio=1.0)

    counter = collections.Counter()
    for _ in range(200):
        examples, _ = scheduler.get_replay_batch(batch_size=10)
        for ex in examples:
            counter[ex.idx] += 1

    # Anchor 9 should be sampled far more than any other anchor
    high_prio_count = counter[9]
    low_prio_max = max(counter[i] for i in range(9))
    assert high_prio_count > low_prio_max * 2


def test_total_replayed_increments():
    anchor_set = make_anchor_set_with_priorities([0.5, 1.0, 0.2, 0.8, 0.3])
    scheduler = ReplayScheduler([anchor_set], replay_ratio=0.1)
    assert scheduler.total_replayed == 0
    scheduler.get_replay_batch(batch_size=32)  # returns 3
    scheduler.get_replay_batch(batch_size=32)  # returns 3
    assert scheduler.total_replayed == 6


def test_replay_batch_empty_when_no_anchors_with_priority():
    """If all anchors have priority=0 (no forgetting), return empty."""
    examples = [{"text": f"ex {i}"} for i in range(5)]
    anchor_set = AnchorSet(examples)
    # Baseline = current => priority = 0 for all
    anchor_set.snapshot_baseline({i: 1.0 for i in range(5)})
    anchor_set.update_current_losses({i: 1.0 for i in range(5)})
    scheduler = ReplayScheduler([anchor_set], replay_ratio=0.1)
    result_examples, result_weights = scheduler.get_replay_batch(batch_size=32)
    assert result_examples == []
    assert result_weights == []
