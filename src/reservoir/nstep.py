"""
reservoir.nstep — N-step return wrapper for FastPERBuffer.

Wraps FastPERBuffer to compute n-step returns before storing transitions.

The n-step return is:
    G_t^(n) = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n * V(s_{t+n})

For training, we store (s_t, a_t, G_t^(n), s_{t+n}, done_{t+n}) with
done=True if any step in the n-step window terminated the episode.

Episode boundaries: if step k (0 ≤ k < n) is terminal, the n-step return
is truncated at that point: G_t^(k+1) = r_t + ... + γ^k * r_{t+k}.

Usage
-----
    buf = FastPERBuffer(capacity=100_000, obs_shape=(8,))
    nstep_buf = NStepBuffer(buf, n=3, gamma=0.99)

    # Replace buf.add(...) with nstep_buf.add(...)
    nstep_buf.add(obs, action, reward, next_obs, done)

    # Sample from the underlying buffer as usual
    batch = nstep_buf.sample(256)
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from reservoir.fast_buffer import FastPERBuffer, FastBatch


class NStepBuffer:
    """N-step return wrapper around FastPERBuffer.

    Parameters
    ----------
    buffer : FastPERBuffer
        The underlying replay buffer transitions are stored in.
    n : int
        Number of steps for n-step return. n=1 is equivalent to 1-step TD.
    gamma : float
        Discount factor.
    """

    def __init__(self, buffer: FastPERBuffer, n: int = 3, gamma: float = 0.99) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")

        self.buffer = buffer
        self.n = n
        self.gamma = gamma

        # Pending transitions: each entry is (obs, action, reward, next_obs, done)
        self._pending: deque = deque(maxlen=n)
        self._gamma_powers = np.array([gamma ** k for k in range(n)], dtype=np.float64)

    def add(
        self,
        obs: np.ndarray,
        action,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """Add a transition. Flushes to the underlying buffer once n steps
        are accumulated or an episode ends.

        Parameters
        ----------
        obs, next_obs : np.ndarray
            Observations.
        action : int or np.ndarray
            Action taken.
        reward : float
            Immediate reward.
        done : bool
            Whether this step ended the episode.
        priority : float or None
            If provided, passed through to the underlying buffer.
        """
        self._pending.append((obs, action, reward, next_obs, done))

        # We can flush a transition once we have n steps in the queue
        if len(self._pending) == self.n:
            self._flush_oldest(priority)

        # If the episode ends, flush all remaining pending transitions
        if done:
            while len(self._pending) > 0:
                self._flush_oldest(priority)

    def _flush_oldest(self, priority: Optional[float]) -> None:
        """Flush the oldest pending transition with its n-step return."""
        if not self._pending:
            return

        # s_t, a_t are from the oldest entry
        s_t, a_t, _, _, _ = self._pending[0]

        # Accumulate n-step return from index 0..len(pending)-1
        n_actual = len(self._pending)
        rewards = np.array([self._pending[k][2] for k in range(n_actual)], dtype=np.float64)
        dones = [self._pending[k][4] for k in range(n_actual)]

        # Find first terminal step
        first_done = n_actual  # default: no terminal within window
        for k, d in enumerate(dones):
            if d:
                first_done = k + 1
                break

        # G = sum_{k=0}^{first_done-1} gamma^k * r_k
        g = float(np.dot(self._gamma_powers[:first_done], rewards[:first_done]))

        # Bootstrap: if window didn't terminate, add V(s_{t+n})
        # The agent's value function is not available here — the caller uses
        # the next_obs from the last step, and the training code handles the
        # bootstrapping. We flag how many steps were actually used.
        last_idx = first_done - 1
        s_tn = self._pending[last_idx][3]   # next_obs after last real step
        done_tn = dones[last_idx]

        self.buffer.add(s_t, a_t, g, s_tn, done_tn, priority)
        self._pending.popleft()

    def flush_all(self) -> None:
        """Flush all remaining pending transitions. Call at episode end."""
        while self._pending:
            self._flush_oldest(None)

    def sample(self, batch_size: int) -> FastBatch:
        """Sample from the underlying buffer."""
        return self.buffer.sample(batch_size)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        self.buffer.update_priorities(indices, td_errors)

    def anneal_beta(self, step: int, total_steps: int, beta_end: float = 1.0) -> None:
        self.buffer.anneal_beta(step, total_steps, beta_end)

    @property
    def size(self) -> int:
        return self.buffer.size

    @property
    def gamma_n(self) -> float:
        """γ^n — used by training code for n-step bootstrapping: target += γ^n * V(s_{t+n})."""
        return self.gamma ** self.n
