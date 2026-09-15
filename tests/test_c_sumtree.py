"""Tests for reservoir._sumtree C extension (SumTree and MinTree).

All tests import from reservoir._sumtree, which requires the C extension to be
built first. Run `pip install -e .` before running these tests.
"""
import math
import pytest

_sumtree = pytest.importorskip(
    "reservoir._sumtree",
    reason="C extension not built — run `pip install -e .` first",
)
SumTree = _sumtree.SumTree
MinTree = _sumtree.MinTree


# ---------------------------------------------------------------------------
# SumTree: basic construction
# ---------------------------------------------------------------------------

class TestSumTreeConstruction:
    def test_capacity_rounded_to_power_of_two(self):
        t = SumTree(5)
        assert t.tree_capacity == 8

    def test_exact_power_of_two_unchanged(self):
        t = SumTree(8)
        assert t.tree_capacity == 8

    def test_capacity_1(self):
        t = SumTree(1)
        assert t.tree_capacity == 1

    def test_initial_total_is_zero(self):
        t = SumTree(4)
        assert t.total == 0.0

    def test_invalid_capacity(self):
        with pytest.raises((ValueError, OverflowError)):
            SumTree(0)


# ---------------------------------------------------------------------------
# SumTree: update and total
# ---------------------------------------------------------------------------

class TestSumTreeUpdate:
    def test_single_update_sets_total(self):
        t = SumTree(4)
        t.update(0, 3.0)
        assert t.total == pytest.approx(3.0)

    def test_multiple_updates_sum(self):
        t = SumTree(4)
        t.update(0, 1.0)
        t.update(1, 2.0)
        t.update(2, 3.0)
        t.update(3, 4.0)
        assert t.total == pytest.approx(10.0)

    def test_update_overwrites_previous(self):
        t = SumTree(4)
        t.update(0, 5.0)
        t.update(0, 2.0)
        assert t.total == pytest.approx(2.0)

    def test_get_returns_leaf_value(self):
        t = SumTree(4)
        t.update(2, 7.5)
        assert t.get(2) == pytest.approx(7.5)

    def test_update_out_of_range_raises(self):
        t = SumTree(4)
        with pytest.raises(IndexError):
            t.update(4, 1.0)

    def test_update_negative_raises(self):
        t = SumTree(4)
        with pytest.raises(ValueError):
            t.update(0, -1.0)

    def test_update_nan_raises(self):
        t = SumTree(4)
        with pytest.raises(ValueError):
            t.update(0, float("nan"))

    def test_update_inf_raises(self):
        t = SumTree(4)
        with pytest.raises(ValueError):
            t.update(0, float("inf"))

    def test_update_zero_allowed(self):
        t = SumTree(4)
        t.update(0, 0.0)
        assert t.total == 0.0


# ---------------------------------------------------------------------------
# SumTree: sample_batch
# ---------------------------------------------------------------------------

class TestSumTreeSampleBatch:
    def setup_method(self):
        self.t = SumTree(4)
        # priorities [1, 2, 3, 4], total = 10
        for i, p in enumerate([1.0, 2.0, 3.0, 4.0]):
            self.t.update(i, p)

    def test_sample_batch_returns_list(self):
        result = self.t.sample_batch([0.5])
        assert isinstance(result, list)

    def test_sample_batch_correct_length(self):
        result = self.t.sample_batch([1.0, 5.0, 9.5])
        assert len(result) == 3

    def test_sample_batch_boundary_start(self):
        # draw 0.0 -> position 0 (priority 1.0 covers [0, 1))
        result = self.t.sample_batch([0.0])
        assert result[0] == 0

    def test_sample_batch_boundary_end(self):
        # draw 9.99 -> position 3 (priority 4.0 covers [6, 10))
        result = self.t.sample_batch([9.99])
        assert result[0] == 3

    def test_sample_batch_proportional(self):
        # position 1 has priority 2.0, covers [1.0, 3.0)
        result = self.t.sample_batch([2.0])
        assert result[0] == 1

    def test_sample_batch_empty_tree_raises(self):
        t = SumTree(4)
        with pytest.raises(RuntimeError):
            t.sample_batch([0.5])

    def test_sample_batch_out_of_range_raises(self):
        with pytest.raises(ValueError):
            self.t.sample_batch([10.0])  # total is exactly 10.0

    def test_sample_batch_negative_raises(self):
        with pytest.raises(ValueError):
            self.t.sample_batch([-0.1])

    def test_sample_batch_capacity_1(self):
        t = SumTree(1)
        t.update(0, 5.0)
        assert t.sample_batch([2.5]) == [0]


# ---------------------------------------------------------------------------
# MinTree: basic behavior
# ---------------------------------------------------------------------------

class TestMinTree:
    def test_initial_minimum_is_infinity(self):
        t = MinTree(4)
        assert math.isinf(t.minimum) and t.minimum > 0

    def test_update_single_sets_minimum(self):
        t = MinTree(4)
        t.update(0, 3.0)
        assert t.minimum == pytest.approx(3.0)

    def test_minimum_tracks_smallest(self):
        t = MinTree(4)
        t.update(0, 5.0)
        t.update(1, 2.0)
        t.update(2, 8.0)
        assert t.minimum == pytest.approx(2.0)

    def test_update_replaces_minimum(self):
        t = MinTree(4)
        t.update(0, 2.0)
        t.update(1, 5.0)
        t.update(0, 7.0)  # overwrite min slot
        assert t.minimum == pytest.approx(5.0)

    def test_get_returns_leaf_value(self):
        t = MinTree(4)
        t.update(1, 4.5)
        assert t.get(1) == pytest.approx(4.5)

    def test_tree_capacity_power_of_two(self):
        t = MinTree(6)
        assert t.tree_capacity == 8

    def test_update_out_of_range_raises(self):
        t = MinTree(4)
        with pytest.raises(IndexError):
            t.update(4, 1.0)

    def test_update_negative_raises(self):
        t = MinTree(4)
        with pytest.raises(ValueError):
            t.update(0, -1.0)

    def test_empty_slot_acts_as_infinity_in_min(self):
        # Only slot 1 set - minimum should be that value, not 0
        t = MinTree(4)
        t.update(1, 3.0)
        assert t.minimum == pytest.approx(3.0)
