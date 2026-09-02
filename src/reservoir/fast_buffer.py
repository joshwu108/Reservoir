"""
reservoir.fast_buffer — Fast numpy-based PER buffer for real model training.

This is the production buffer. It uses float32/float64 numpy arrays and
returns torch tensors directly — compatible with any PyTorch training loop.

The exact buffer (buffer.py) serves as a reference and auditor, not a
training tool. Use this module for actual training.

Drop-in interface compatible with stable-baselines3 and CleanRL conventions.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import NamedTuple, Optional


@dataclass
class FastBatch:
    """Batch returned by FastPERBuffer.sample(). All fields are torch tensors."""
    states: torch.Tensor          # (batch, *obs_shape)
    actions: torch.Tensor         # (batch,)
    rewards: torch.Tensor         # (batch,)
    next_states: torch.Tensor     # (batch, *obs_shape)
    dones: torch.Tensor           # (batch,)
    is_weights: torch.Tensor      # (batch,) normalized to [0, 1]
    indices: np.ndarray           # (batch,) int — for priority updates


class FastPERBuffer:
    """Prioritized Experience Replay buffer backed by numpy arrays.

    Compatible with PyTorch training loops. Designed to not bottleneck
    GPU training.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions.
    obs_shape : tuple
        Shape of a single observation (e.g. (84, 84, 4) for Atari, (8,) for LunarLander).
    action_dim : int
        Number of action dimensions (1 for discrete, n for continuous).
    alpha : float
        Priority exponent. 0 = uniform, 1 = full prioritization.
    beta : float
        IS correction exponent. Anneal from 0.4 to 1.0 over training.
    epsilon : float
        Minimum priority offset to prevent zero priorities.
    device : str
        Torch device for returned tensors ('cpu' or 'cuda').
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
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.device = device

        self._size = 0
        self._ptr = 0  # Next write position

        # Pre-allocated numpy arrays for O(1) inserts
        self._states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)

        # Priority sum-tree (float32)
        # Tree size = 2 * next_power_of_2(capacity)
        self._tree_capacity = 1
        while self._tree_capacity < capacity:
            self._tree_capacity <<= 1
        self._tree = np.zeros(2 * self._tree_capacity, dtype=np.float64)
        self._min_tree = np.full(2 * self._tree_capacity, np.inf, dtype=np.float64)

        # Default max priority for new transitions
        self._max_priority: float = 1.0

    # ------------------------------------------------------------------
    # Sum-tree operations
    # ------------------------------------------------------------------

    def _tree_update(self, pos: int, priority: float) -> None:
        """Update priority at leaf position and propagate to root."""
        p_alpha = float(priority ** self.alpha)
        idx = self._tree_capacity - 1 + pos
        self._tree[idx] = p_alpha
        self._min_tree[idx] = p_alpha
        parent = (idx - 1) >> 1
        while idx > 0:
            l, r = 2 * parent + 1, 2 * parent + 2
            self._tree[parent] = self._tree[l] + self._tree[r]
            self._min_tree[parent] = min(self._min_tree[l], self._min_tree[r])
            idx = parent
            parent = (idx - 1) >> 1

    def _tree_sample(self, value: float) -> int:
        """Find leaf by descending from root. Returns 0-indexed position."""
        idx = 0
        while idx < self._tree_capacity - 1:
            l = 2 * idx + 1
            if self._tree[l] > value:
                idx = l
            else:
                value -= self._tree[l]
                idx = 2 * idx + 2
        return idx - (self._tree_capacity - 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    @property
    def total_priority(self) -> float:
        return float(self._tree[0])

    def add(
        self,
        state: np.ndarray,
        action: int | float | np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """Add a transition. O(log N).

        Parameters
        ----------
        state, next_state : np.ndarray
            Observations matching obs_shape.
        action : int or float or array
            Action taken.
        reward : float
            Reward received.
        done : bool
            Episode termination flag.
        priority : float or None
            Raw priority (before alpha exponentiation). If None, uses max seen.
        """
        if priority is None:
            priority = self._max_priority
        else:
            self._max_priority = max(self._max_priority, priority)
            priority = abs(priority) + self.epsilon

        pos = self._ptr
        self._states[pos] = state
        self._next_states[pos] = next_state
        self._actions[pos] = action
        self._rewards[pos] = reward
        self._dones[pos] = float(done)
        self._tree_update(pos, priority)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities after learning. Call after each training step.

        Parameters
        ----------
        indices : np.ndarray
            Buffer positions (from FastBatch.indices).
        td_errors : np.ndarray
            Absolute TD errors for each transition.
        """
        for idx, err in zip(indices, td_errors):
            priority = float(abs(err)) + self.epsilon
            self._max_priority = max(self._max_priority, priority)
            self._tree_update(int(idx), priority)

    def sample(self, batch_size: int) -> FastBatch:
        """Sample a batch. O(batch_size * log N).

        Uses stratified sampling: divides [0, total) into batch_size
        equal segments and samples one value per segment. This gives
        better coverage than i.i.d. sampling.

        Parameters
        ----------
        batch_size : int

        Returns
        -------
        FastBatch
            All tensors on self.device.
        """
        assert self._size >= batch_size, (
            f"Buffer has {self._size} transitions, need {batch_size}"
        )

        total = self.total_priority
        segment = total / batch_size
        min_priority = float(self._min_tree[0])

        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        for k in range(batch_size):
            lo = segment * k
            hi = segment * (k + 1)
            value = np.random.uniform(lo, hi)
            pos = self._tree_sample(value)
            pos = np.clip(pos, 0, self.capacity - 1)
            indices[k] = pos
            leaf_idx = self._tree_capacity - 1 + pos
            priorities[k] = self._tree[leaf_idx]

        # IS weights: w_i = (N * P(i))^(-beta) / max_w
        n = self._size
        probs = priorities / total
        # max weight comes from min priority
        max_weight = (n * min_priority / total) ** (-self.beta)
        weights = (n * probs) ** (-self.beta) / max_weight
        weights = np.clip(weights, 0.0, 1.0).astype(np.float32)

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr).to(self.device)

        return FastBatch(
            states=_t(self._states[indices]),
            actions=_t(self._actions[indices]).squeeze(-1).long()
            if self._actions.shape[1] == 1
            else _t(self._actions[indices]),
            rewards=_t(self._rewards[indices]),
            next_states=_t(self._next_states[indices]),
            dones=_t(self._dones[indices]),
            is_weights=_t(weights),
            indices=indices,
        )

    def anneal_beta(self, step: int, total_steps: int, beta_end: float = 1.0) -> None:
        """Linearly anneal beta from initial value to beta_end."""
        self.beta = min(
            beta_end,
            self.beta + (beta_end - self.beta) * step / total_steps,
        )
