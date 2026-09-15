"""reservoir.c_buffer — C-tree-backed Prioritized Experience Replay buffer.

Drop-in replacement for FastPERBuffer. Uses reservoir._sumtree.SumTree and
MinTree for O(log N) priority operations in C. All numpy/torch data storage
is unchanged from FastPERBuffer.

Requires the C extension to be built:
    pip install -e .
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Optional

from reservoir._sumtree import SumTree, MinTree
from reservoir.fast_buffer import FastBatch


class CFastPERBuffer:
    """Prioritized Experience Replay buffer backed by a C sum-tree.

    Public API is identical to FastPERBuffer. Only the internal tree
    implementation differs (C vs numpy).

    Parameters
    ----------
    capacity : int
        Maximum number of transitions.
    obs_shape : tuple
        Shape of a single observation.
    action_dim : int
        Number of action dimensions (1 for discrete).
    alpha : float
        Priority exponent. 0 = uniform, 1 = full prioritization.
    beta : float
        IS correction exponent. Anneal from 0.4 to 1.0 over training.
    epsilon : float
        Minimum priority offset to prevent zero priorities.
    device : str
        Torch device for returned tensors.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_dim: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        device: str = "cpu",
    ) -> None:
        self.capacity   = capacity
        self.obs_shape  = obs_shape
        self.alpha      = alpha
        self.beta       = beta
        self.epsilon    = epsilon
        self.device     = device

        self._size = 0
        self._ptr  = 0  # circular write pointer
        self._max_priority: float = 1.0

        # Pre-allocated numpy arrays (same layout as FastPERBuffer)
        self._states      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards     = np.zeros((capacity,), dtype=np.float32)
        self._dones       = np.zeros((capacity,), dtype=np.float32)

        # C-backed trees (mirror FastPERBuffer's internal _tree / _min_tree)
        self._sum_tree = SumTree(capacity)
        self._min_tree = MinTree(capacity)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    @property
    def total_priority(self) -> float:
        return self._sum_tree.total

    # ------------------------------------------------------------------
    # Tree operations (mirrors FastPERBuffer._tree_update)
    # ------------------------------------------------------------------

    def _tree_update(self, pos: int, priority: float) -> None:
        """Update tree at pos with pre-exponentiated priority."""
        p_alpha = float(priority ** self.alpha)
        self._sum_tree.update(pos, p_alpha)
        self._min_tree.update(pos, p_alpha)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        state: np.ndarray,
        action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """Add a transition. O(log N)."""
        if priority is None:
            priority = self._max_priority
        else:
            self._max_priority = max(self._max_priority, priority)
            priority = abs(priority) + self.epsilon

        pos = self._ptr
        self._states[pos]      = state
        self._next_states[pos] = next_state
        self._actions[pos]     = action
        self._rewards[pos]     = reward
        self._dones[pos]       = float(done)
        self._tree_update(pos, priority)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        """Update priorities after learning. Call after each training step."""
        for idx, err in zip(indices, td_errors):
            priority = float(abs(err)) + self.epsilon
            self._max_priority = max(self._max_priority, priority)
            self._tree_update(int(idx), priority)

    def sample(self, batch_size: int) -> FastBatch:
        """Sample a batch using stratified PER sampling.

        Mirrors FastPERBuffer.sample() exactly: stratified draws, IS weights,
        same FastBatch return type.
        """
        assert self._size >= batch_size, (
            f"Buffer has {self._size} transitions, need {batch_size}"
        )

        total       = self._sum_tree.total
        min_p_alpha = self._min_tree.minimum
        segment     = total / batch_size

        # Stratified draws - one per segment (matches FastPERBuffer exactly)
        offsets = np.random.uniform(0, segment, size=batch_size)
        values  = (offsets + segment * np.arange(batch_size)).astype(np.float64)

        # C tree walk - returns list of int positions
        indices = np.array(self._sum_tree.sample_batch(values.tolist()), dtype=np.int64)
        priorities = np.array(
            [self._sum_tree.get(int(i)) for i in indices], dtype=np.float64
        )

        # IS weights (same formula as FastPERBuffer)
        n = self._size
        probs      = np.maximum(priorities / total, 1e-10)
        max_weight = (n * min_p_alpha / total) ** (-self.beta) if min_p_alpha > 0 else 1.0
        weights    = ((n * probs) ** (-self.beta) / max_weight).clip(0.0, 1.0).astype(np.float32)

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(self.device)

        return FastBatch(
            states      = _t(self._states[indices]),
            actions     = _t(self._actions[indices]).squeeze(-1).long()
                          if self._actions.shape[1] == 1
                          else _t(self._actions[indices]),
            rewards     = _t(self._rewards[indices]),
            next_states = _t(self._next_states[indices]),
            dones       = _t(self._dones[indices]),
            is_weights  = _t(weights),
            indices     = indices,
        )

    def anneal_beta(
        self, step: int, total_steps: int, beta_end: float = 1.0
    ) -> None:
        """Linearly anneal beta from initial value to beta_end."""
        self.beta = min(
            beta_end,
            self.beta + (beta_end - self.beta) * step / total_steps,
        )
