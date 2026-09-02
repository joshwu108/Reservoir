"""
campaigns.float_baselines — Faithful float sum-tree reimplementations.

Two classes, each mirroring a named public idiom:

1. OpenAISegmentTree
   Mirrors: openai/baselines/deepq/replay_buffer.py (2019)
   URL: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
   Idiom: recursive/iterative float64 segment tree, single flat array.
   Node values: float64 sums.
   Update: propagate from leaf to root, recomputing each parent as sum of children.
   Sample: tree search from root using float comparison.

2. LabmlArraySumTree
   Mirrors: labml-ai tutorial implementation (2020-2021)
   URL: https://nn.labml.ai/rl/dqn/replay_buffer.html
   Also mirrors: numerous tutorial implementations (stable-baselines, RLlib)
   Idiom: flat float64 array tree, size=2*capacity, insert at next slot.
   Node values: float64 sums.
   Update: leaf update + propagate to root.
   Sample: recursive descent, subtracting left child sum.

Both classes use float64 throughout. Alpha exponentiation is applied
in float64 at priority insertion time (p_i^alpha, float64).
"""

from __future__ import annotations

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Idiom 1: OpenAI Baselines-style Segment Tree
# ---------------------------------------------------------------------------

class OpenAISegmentTree:
    """
    Float64 segment tree mirroring openai/baselines deepq/replay_buffer.py.

    Citation: OpenAI Baselines, SegmentTree class, 2017-2019.
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Implementation choices (matching the public idiom):
      - Single flat float64 array, 1-indexed (index 0 unused)
      - Capacity must be a power of 2
      - Root at index 1; node i's children at 2*i, 2*i+1
      - Update: walk from leaf to root, recompute parent as left+right
      - Sum query: prefix sum via tree descent
      - Sample: tree descent with float comparison to find leaf
    """

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        assert capacity > 0 and (capacity & (capacity - 1)) == 0, (
            "Capacity must be a positive power of 2"
        )
        self.capacity = capacity
        self.alpha = alpha
        # 1-indexed array: indices [1, 2*capacity)
        # Index 1 = root; leaves at [capacity, 2*capacity)
        self._tree = [0.0] * (2 * capacity)
        self._max_priority: float = 1.0

    def _update(self, leaf_idx: int, priority_alpha: float) -> None:
        """Update leaf at tree index leaf_idx and propagate to root."""
        self._tree[leaf_idx] = priority_alpha
        idx = leaf_idx >> 1  # parent
        while idx >= 1:
            self._tree[idx] = self._tree[2 * idx] + self._tree[2 * idx + 1]
            idx >>= 1

    def update(self, position: int, priority: float) -> None:
        """Update priority at 0-indexed position."""
        p_alpha = priority ** self.alpha
        self._max_priority = max(self._max_priority, priority)
        leaf_idx = self.capacity + position
        self._update(leaf_idx, p_alpha)

    def get_leaf(self, position: int) -> float:
        """Return float64 priority^alpha at position."""
        return self._tree[self.capacity + position]

    @property
    def total(self) -> float:
        """Total sum (root value)."""
        return self._tree[1]

    def sample(self, value: float) -> int:
        """Find leaf whose prefix sum first exceeds value.

        Parameters
        ----------
        value : float
            Target in [0, total). Uses float64 comparison.

        Returns
        -------
        int
            0-indexed position.
        """
        idx = 1  # root
        while idx < self.capacity:
            left = 2 * idx
            if self._tree[left] > value:
                idx = left
            else:
                value -= self._tree[left]
                idx = left + 1
        return idx - self.capacity


# ---------------------------------------------------------------------------
# Idiom 2: labml/tutorial-style Array Sum Tree
# ---------------------------------------------------------------------------

class LabmlArraySumTree:
    """
    Float64 array sum-tree mirroring labml-ai and many tutorial implementations.

    Citation: labml.ai DQN tutorial, Sadeep Jayasumana et al., 2020-2021.
    https://nn.labml.ai/rl/dqn/replay_buffer.html
    Also mirrors: stable-baselines3 PrioritizedReplayBuffer, RLlib similar implementations.

    Implementation choices (matching the public idiom):
      - Flat float64 array of size 2*capacity, 0-indexed
      - Leaves at [capacity, 2*capacity); root at 0
      - Node i stores sum of its subtree
      - Leaf i (0-indexed) at tree index: capacity - 1 + i
      - Update: propagate from leaf to parent
      - Sample: descent from root, subtracting left child

    Note: this idiom uses 0-indexed root (unlike OpenAI's 1-indexed).
    Structural choice matches the "double-size array" pattern common in
    competitive programming implementations adapted for PER.
    """

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        assert capacity > 0 and (capacity & (capacity - 1)) == 0
        self.capacity = capacity
        self.alpha = alpha
        self._tree = [0.0] * (2 * capacity)
        self._max_priority: float = 1.0

    def update(self, position: int, priority: float) -> None:
        """Update priority at 0-indexed position."""
        p_alpha = priority ** self.alpha
        self._max_priority = max(self._max_priority, priority)
        idx = self.capacity - 1 + position  # leaf index (0-indexed tree)
        self._tree[idx] = p_alpha
        # Propagate up
        parent = (idx - 1) >> 1
        while idx > 0:
            left = 2 * parent + 1
            right = 2 * parent + 2
            self._tree[parent] = self._tree[left] + self._tree[right]
            idx = parent
            parent = (idx - 1) >> 1

    def get_leaf(self, position: int) -> float:
        return self._tree[self.capacity - 1 + position]

    @property
    def total(self) -> float:
        return self._tree[0]

    def sample(self, value: float) -> int:
        """Find leaf by descending from root, subtracting left child."""
        idx = 0  # root
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            if self._tree[left] > value:
                idx = left
            else:
                value -= self._tree[left]
                idx = 2 * idx + 2
        return idx - (self.capacity - 1)
