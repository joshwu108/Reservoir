"""
reservoir.sumtree — Exact integer sum-tree for prioritized experience replay.

The sum-tree is a binary heap stored as a flat array where:
  - Leaves are at indices [capacity-1, 2*capacity-1)
  - Internal node i has children 2i+1 and 2i+2
  - Internal node i stores the SUM of its subtree's leaf values
  - Node 0 is the root (total sum)

All values are Python integers — no floats anywhere in this module.
Tree invariant: internal_node[i] == sum of all leaf values in its subtree.

The tree supports a parallel min-tree (same structure, tracks minimums)
for computing the maximum IS weight (minimum priority transition).

Capacity must be a power of 2 for clean indexing; this module enforces it.
"""

from __future__ import annotations

import math
from typing import Iterator


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n <= 0:
        raise ValueError(f"Capacity must be positive, got {n}")
    if n == 1:
        return 1
    return 1 << (n - 1).bit_length()


class ExactSumTree:
    """Exact integer sum-tree with power-of-two capacity.

    The tree stores integer priorities at leaves and maintains their prefix
    sums in internal nodes. All arithmetic is exact integer arithmetic.

    Parameters
    ----------
    capacity : int
        Maximum number of elements. Will be rounded up to next power of 2.

    Attributes
    ----------
    capacity : int
        Actual capacity (power of 2 >= requested capacity).
    size : int
        Number of currently occupied slots.
    _tree : list[int]
        Flat tree array of length 2*capacity. Index 0 is root.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity: int = _next_power_of_two(capacity)
        self.size: int = 0
        # tree[0] = root (sum of all)
        # tree[capacity-1 .. 2*capacity-2] = leaves
        self._tree: list[int] = [0] * (2 * self.capacity)

    @property
    def total(self) -> int:
        """Total sum of all leaf priorities (the root value)."""
        return self._tree[0]

    def _leaf_index(self, position: int) -> int:
        """Convert a 0-based logical position to a tree array index."""
        return self.capacity - 1 + position

    def _propagate_up(self, tree_idx: int) -> None:
        """Propagate changes up from tree_idx to the root."""
        parent = (tree_idx - 1) >> 1  # (tree_idx - 1) // 2
        while tree_idx > 0:
            left = 2 * parent + 1
            right = 2 * parent + 2
            self._tree[parent] = self._tree[left] + self._tree[right]
            tree_idx = parent
            parent = (tree_idx - 1) >> 1

    def update(self, position: int, priority_int: int) -> int:
        """Update the priority at a logical position.

        Parameters
        ----------
        position : int
            0-based position in [0, capacity).
        priority_int : int
            New priority, must be a non-negative integer.

        Returns
        -------
        int
            The old priority value that was replaced.

        Raises
        ------
        ValueError
            If position is out of range or priority_int is negative.
        """
        if not (0 <= position < self.capacity):
            raise ValueError(
                f"Position {position} out of range [0, {self.capacity})"
            )
        if priority_int < 0:
            raise ValueError(f"Priority must be non-negative, got {priority_int}")

        tree_idx = self._leaf_index(position)
        old_value = self._tree[tree_idx]
        self._tree[tree_idx] = priority_int
        self._propagate_up(tree_idx)
        return old_value

    def get(self, position: int) -> int:
        """Return the priority at a logical position."""
        if not (0 <= position < self.capacity):
            raise ValueError(
                f"Position {position} out of range [0, {self.capacity})"
            )
        return self._tree[self._leaf_index(position)]

    def prefix_sum_locate(self, target: int) -> int:
        """Find the leaf position whose prefix sum first exceeds target.

        Uses a deterministic walk from root to leaf using integer comparisons.
        Returns the 0-based logical position of the leaf such that:
            sum(leaves[0..pos-1]) <= target < sum(leaves[0..pos])

        This is a pure integer operation — no floats.

        Parameters
        ----------
        target : int
            An integer in [0, total). Must be < total.

        Returns
        -------
        int
            0-based leaf position.

        Raises
        ------
        ValueError
            If total == 0 or target >= total.
        """
        if self.total == 0:
            raise ValueError("Cannot sample from empty tree (total == 0)")
        if target < 0 or target >= self.total:
            raise ValueError(
                f"Target {target} out of range [0, {self.total})"
            )

        idx = 0  # start at root
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = 2 * idx + 2
            if target < self._tree[left]:
                idx = left
            else:
                target -= self._tree[left]
                idx = right

        return idx - (self.capacity - 1)

    def verify_invariant(self) -> bool:
        """Recompute every internal node from leaves and check consistency.

        Returns True if the tree is consistent, raises AssertionError otherwise.
        """
        # Walk bottom-up from internal nodes
        for i in range(self.capacity - 2, -1, -1):
            left = 2 * i + 1
            right = 2 * i + 2
            expected = self._tree[left] + self._tree[right]
            actual = self._tree[i]
            if expected != actual:
                raise AssertionError(
                    f"Tree invariant violated at node {i}: "
                    f"expected {expected}, got {actual} "
                    f"(left={self._tree[left]}, right={self._tree[right]})"
                )
        return True

    def __len__(self) -> int:
        return self.size

    def leaves(self) -> Iterator[tuple[int, int]]:
        """Iterate over (position, priority) for all leaves."""
        for pos in range(self.capacity):
            yield pos, self._tree[self._leaf_index(pos)]


class ExactMinTree:
    """Exact integer min-tree (same structure as ExactSumTree, tracks minimums).

    Used to find the maximum IS weight (minimum priority transition).
    Stores INFINITY as sentinel for empty leaves.
    """

    # Sentinel for empty leaves: must exceed any priority integer produced by
    # float_to_priority_int. max float64 ≈ 1.8e308, integerized * 2^52 ≈ 2^1075.
    # 2^2048 is safely larger than any realistic priority.
    INFINITY: int = (1 << 2048)

    def __init__(self, capacity: int) -> None:
        self.capacity: int = _next_power_of_two(capacity)
        self._tree: list[int] = [self.INFINITY] * (2 * self.capacity)

    @property
    def minimum(self) -> int:
        """Minimum priority across all leaves. Returns INFINITY if all empty."""
        return self._tree[0]

    def _leaf_index(self, position: int) -> int:
        return self.capacity - 1 + position

    def _propagate_up(self, tree_idx: int) -> None:
        parent = (tree_idx - 1) >> 1
        while tree_idx > 0:
            left = 2 * parent + 1
            right = 2 * parent + 2
            self._tree[parent] = min(self._tree[left], self._tree[right])
            tree_idx = parent
            parent = (tree_idx - 1) >> 1

    def update(self, position: int, priority_int: int) -> None:
        """Update priority at position."""
        if not (0 <= position < self.capacity):
            raise ValueError(f"Position {position} out of range")
        if priority_int < 0:
            raise ValueError(f"Priority must be non-negative, got {priority_int}")

        tree_idx = self._leaf_index(position)
        self._tree[tree_idx] = priority_int
        self._propagate_up(tree_idx)

    def remove(self, position: int) -> None:
        """Reset position to INFINITY (slot becoming empty)."""
        if not (0 <= position < self.capacity):
            raise ValueError(f"Position {position} out of range")
        tree_idx = self._leaf_index(position)
        self._tree[tree_idx] = self.INFINITY
        self._propagate_up(tree_idx)

    def get(self, position: int) -> int:
        return self._tree[self._leaf_index(position)]

    def verify_invariant(self) -> bool:
        """Recompute every internal node from leaves and check consistency."""
        for i in range(self.capacity - 2, -1, -1):
            left = 2 * i + 1
            right = 2 * i + 2
            expected = min(self._tree[left], self._tree[right])
            actual = self._tree[i]
            if expected != actual:
                raise AssertionError(
                    f"MinTree invariant violated at node {i}: "
                    f"expected {expected}, got {actual}"
                )
        return True
