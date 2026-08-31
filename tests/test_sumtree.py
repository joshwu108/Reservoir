"""
Tests for reservoir.sumtree — exact integer sum-tree and min-tree.

All arithmetic in these tests is integer-based; no floats.
"""

import pytest

from reservoir.sumtree import ExactMinTree, ExactSumTree, _next_power_of_two


class TestNextPowerOfTwo:
    def test_one(self):
        assert _next_power_of_two(1) == 1

    def test_two(self):
        assert _next_power_of_two(2) == 2

    def test_three_rounds_up(self):
        assert _next_power_of_two(3) == 4

    def test_four(self):
        assert _next_power_of_two(4) == 4

    def test_five_rounds_up(self):
        assert _next_power_of_two(5) == 8

    def test_powers_of_two_unchanged(self):
        for k in range(1, 17):
            assert _next_power_of_two(1 << k) == (1 << k)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            _next_power_of_two(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            _next_power_of_two(-1)


class TestExactSumTreeBasics:
    def test_capacity_rounded_up(self):
        tree = ExactSumTree(5)
        assert tree.capacity == 8

    def test_capacity_power_of_two_unchanged(self):
        tree = ExactSumTree(8)
        assert tree.capacity == 8

    def test_initial_total_is_zero(self):
        tree = ExactSumTree(4)
        assert tree.total == 0

    def test_initial_leaves_all_zero(self):
        tree = ExactSumTree(4)
        for pos, val in tree.leaves():
            assert val == 0

    def test_update_single_leaf(self):
        tree = ExactSumTree(4)
        tree.update(0, 10)
        assert tree.total == 10
        assert tree.get(0) == 10

    def test_update_multiple_leaves(self):
        tree = ExactSumTree(4)
        tree.update(0, 10)
        tree.update(1, 20)
        tree.update(2, 30)
        tree.update(3, 40)
        assert tree.total == 100

    def test_update_returns_old_value(self):
        tree = ExactSumTree(4)
        tree.update(0, 10)
        old = tree.update(0, 99)
        assert old == 10
        assert tree.total == 99

    def test_invariant_after_updates(self):
        tree = ExactSumTree(8)
        for i in range(8):
            tree.update(i, (i + 1) * 100)
        assert tree.verify_invariant()

    def test_large_integers(self):
        """Must handle arbitrarily large Python integers exactly."""
        tree = ExactSumTree(4)
        big = 10**50
        tree.update(0, big)
        tree.update(1, big * 2)
        assert tree.total == big * 3
        assert tree.verify_invariant()

    def test_update_out_of_range_raises(self):
        tree = ExactSumTree(4)
        with pytest.raises(ValueError):
            tree.update(4, 10)

    def test_update_negative_raises(self):
        tree = ExactSumTree(4)
        with pytest.raises(ValueError):
            tree.update(0, -1)

    def test_get_out_of_range_raises(self):
        tree = ExactSumTree(4)
        with pytest.raises(ValueError):
            tree.get(4)


class TestExactSumTreePrefixLocate:
    """prefix_sum_locate: deterministic walk using integer comparisons only."""

    def test_single_element(self):
        tree = ExactSumTree(4)
        tree.update(0, 100)
        # Any target in [0, 100) should return position 0
        assert tree.prefix_sum_locate(0) == 0
        assert tree.prefix_sum_locate(50) == 0
        assert tree.prefix_sum_locate(99) == 0

    def test_two_equal_elements(self):
        tree = ExactSumTree(2)
        tree.update(0, 50)
        tree.update(1, 50)
        # [0, 50) -> position 0; [50, 100) -> position 1
        assert tree.prefix_sum_locate(0) == 0
        assert tree.prefix_sum_locate(49) == 0
        assert tree.prefix_sum_locate(50) == 1
        assert tree.prefix_sum_locate(99) == 1

    def test_unequal_elements(self):
        tree = ExactSumTree(4)
        tree.update(0, 10)
        tree.update(1, 30)
        tree.update(2, 20)
        tree.update(3, 40)
        # Prefix sums: [0,10), [10,40), [40,60), [60,100)
        assert tree.prefix_sum_locate(0) == 0
        assert tree.prefix_sum_locate(9) == 0
        assert tree.prefix_sum_locate(10) == 1
        assert tree.prefix_sum_locate(39) == 1
        assert tree.prefix_sum_locate(40) == 2
        assert tree.prefix_sum_locate(59) == 2
        assert tree.prefix_sum_locate(60) == 3
        assert tree.prefix_sum_locate(99) == 3

    def test_empty_tree_raises(self):
        tree = ExactSumTree(4)
        with pytest.raises(ValueError, match="empty"):
            tree.prefix_sum_locate(0)

    def test_target_too_large_raises(self):
        tree = ExactSumTree(4)
        tree.update(0, 10)
        with pytest.raises(ValueError):
            tree.prefix_sum_locate(10)

    def test_target_negative_raises(self):
        tree = ExactSumTree(4)
        tree.update(0, 10)
        with pytest.raises(ValueError):
            tree.prefix_sum_locate(-1)

    def test_exhaustive_coverage_capacity8(self):
        """Every integer target in [0, total) maps to exactly one leaf.
        The union of intervals covers [0, total) exactly (no gaps, no overlaps).
        """
        tree = ExactSumTree(8)
        priorities = [10, 30, 5, 20, 15, 25, 8, 12]
        for i, p in enumerate(priorities):
            tree.update(i, p)

        total = sum(priorities)
        assert tree.total == total

        # Count how many times each position is selected
        counts = [0] * 8
        for target in range(total):
            pos = tree.prefix_sum_locate(target)
            counts[pos] += 1

        # Each position should be selected exactly priority[pos] times
        for i, (p, c) in enumerate(zip(priorities, counts)):
            assert p == c, f"Position {i}: priority={p}, count={c}"

    def test_large_capacity(self):
        """Verify correctness for capacity=1024."""
        capacity = 1024
        tree = ExactSumTree(capacity)
        for i in range(capacity):
            tree.update(i, i + 1)  # priorities 1, 2, ..., 1024

        assert tree.total == capacity * (capacity + 1) // 2
        assert tree.verify_invariant()
        # Spot check: last leaf (priority=1024) should cover target in
        # [total - 1024, total - 1]
        last_start = tree.total - 1024
        assert tree.prefix_sum_locate(last_start) == 1023
        assert tree.prefix_sum_locate(tree.total - 1) == 1023

    def test_only_one_nonzero_leaf(self):
        """With only one non-zero leaf, all targets must return that leaf."""
        tree = ExactSumTree(8)
        tree.update(5, 999)
        for target in range(999):
            assert tree.prefix_sum_locate(target) == 5

    def test_update_then_locate(self):
        """After updating a priority, locate reflects the new distribution."""
        tree = ExactSumTree(4)
        tree.update(0, 100)
        tree.update(1, 100)
        # Equal split: [0,100)->0, [100,200)->1
        assert tree.prefix_sum_locate(99) == 0
        assert tree.prefix_sum_locate(100) == 1

        # Increase position 1's priority dramatically
        tree.update(1, 900)
        # Now [0,100)->0, [100,1000)->1
        assert tree.prefix_sum_locate(99) == 0
        assert tree.prefix_sum_locate(100) == 1
        assert tree.prefix_sum_locate(999) == 1


class TestExactMinTree:
    def test_initial_minimum_is_infinity(self):
        mt = ExactMinTree(4)
        assert mt.minimum == ExactMinTree.INFINITY

    def test_single_update(self):
        mt = ExactMinTree(4)
        mt.update(0, 50)
        assert mt.minimum == 50

    def test_minimum_updates_correctly(self):
        mt = ExactMinTree(4)
        mt.update(0, 50)
        mt.update(1, 10)
        mt.update(2, 30)
        assert mt.minimum == 10

    def test_remove_minimum_updates(self):
        mt = ExactMinTree(4)
        mt.update(0, 50)
        mt.update(1, 10)
        mt.remove(1)
        assert mt.minimum == 50

    def test_invariant_holds(self):
        mt = ExactMinTree(8)
        vals = [50, 10, 80, 30, 5, 20, 70, 40]
        for i, v in enumerate(vals):
            mt.update(i, v)
        assert mt.verify_invariant()
        assert mt.minimum == min(vals)

    def test_large_integer_minimum(self):
        # Use 10^50 which is < INFINITY (2^2048 ≈ 3.2×10^616)
        mt = ExactMinTree(4)
        big = 10**50
        mt.update(0, big)
        mt.update(1, big - 1)
        assert mt.minimum == big - 1

    def test_update_negative_raises(self):
        mt = ExactMinTree(4)
        with pytest.raises(ValueError):
            mt.update(0, -1)
