"""
reservoir.gym_wrapper — Gymnasium integration for FastPERBuffer.

Provides:
  - FastPERBuffer.from_env(env, ...) classmethod
  - GymCollector: handles the env-step → buffer-add loop
  - Support for Box and Discrete obs/action spaces
  - Optional observation normalization (running mean/std)
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Any, Callable, Optional, Tuple

from reservoir.fast_buffer import FastPERBuffer, FastBatch


def _obs_shape_from_space(space) -> tuple:
    """Extract observation shape from a gymnasium space."""
    import gymnasium as gym
    if isinstance(space, gym.spaces.Box):
        return space.shape
    elif isinstance(space, gym.spaces.Discrete):
        return (int(space.n),)  # one-hot encoded
    elif isinstance(space, gym.spaces.Dict):
        # Concatenate all subspaces into a flat vector
        total = sum(
            int(np.prod(s.shape)) if isinstance(s, gym.spaces.Box) else int(s.n)
            for s in space.spaces.values()
        )
        return (total,)
    else:
        raise ValueError(f"Unsupported observation space: {type(space)}")


def _action_dim_from_space(space) -> tuple[int, bool]:
    """Return (action_dim, is_discrete) from a gymnasium action space."""
    import gymnasium as gym
    if isinstance(space, gym.spaces.Discrete):
        return 1, True
    elif isinstance(space, gym.spaces.Box):
        return int(np.prod(space.shape)), False
    else:
        raise ValueError(f"Unsupported action space: {type(space)}")


def _flatten_obs(obs, space) -> np.ndarray:
    """Flatten an observation from a gymnasium space to a 1D numpy array."""
    import gymnasium as gym
    if isinstance(space, gym.spaces.Box):
        return np.asarray(obs, dtype=np.float32).flatten()
    elif isinstance(space, gym.spaces.Discrete):
        one_hot = np.zeros(int(space.n), dtype=np.float32)
        one_hot[int(obs)] = 1.0
        return one_hot
    elif isinstance(space, gym.spaces.Dict):
        parts = []
        for key, subspace in space.spaces.items():
            parts.append(_flatten_obs(obs[key], subspace))
        return np.concatenate(parts).astype(np.float32)
    return np.asarray(obs, dtype=np.float32).flatten()


class RunningNormalizer:
    """Online running mean/variance normalizer (Welford's algorithm)."""

    def __init__(self, shape: tuple, clip: float = 10.0) -> None:
        self.shape = shape
        self.clip = clip
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += (x - self.mean) * delta  # M2

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.count < 2:
            return x.astype(np.float32)
        std = np.sqrt(self.var / (self.count - 1) + 1e-8)
        return np.clip((x - self.mean) / std, -self.clip, self.clip).astype(np.float32)

    def normalize_batch(self, x: np.ndarray) -> np.ndarray:
        if self.count < 2:
            return x.astype(np.float32)
        std = np.sqrt(self.var / (self.count - 1) + 1e-8)
        return np.clip((x - self.mean) / std, -self.clip, self.clip).astype(np.float32)


# ---------------------------------------------------------------------------
# FastPERBuffer.from_env classmethod (monkey-patched in)
# ---------------------------------------------------------------------------

def _from_env(
    cls,
    env,
    capacity: int,
    alpha: float = 0.6,
    beta: float = 0.4,
    epsilon: float = 1e-6,
    device: str = "cpu",
    normalize_obs: bool = False,
) -> "FastPERBuffer":
    """Create a FastPERBuffer sized for the given Gymnasium environment.

    Parameters
    ----------
    env : gymnasium.Env
        The environment (used to read observation/action space shapes).
    capacity : int
        Replay buffer capacity.
    alpha, beta, epsilon : float
        PER hyperparameters.
    device : str
        Torch device for sampled tensors.
    normalize_obs : bool
        If True, attach a RunningNormalizer for online obs normalization.

    Returns
    -------
    FastPERBuffer
        Configured for the environment's spaces.
    """
    obs_shape = _obs_shape_from_space(env.observation_space)
    action_dim, is_discrete = _action_dim_from_space(env.action_space)

    buf = cls(
        capacity=capacity,
        obs_shape=obs_shape,
        action_dim=action_dim,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        device=device,
    )
    buf._is_discrete_action = is_discrete
    buf._obs_space = env.observation_space
    buf._act_space = env.action_space

    if normalize_obs:
        buf._normalizer = RunningNormalizer(obs_shape)
    else:
        buf._normalizer = None

    return buf


FastPERBuffer.from_env = classmethod(_from_env)


def _add_env_step(
    self: FastPERBuffer,
    obs,
    action,
    reward: float,
    next_obs,
    done: bool,
    priority: Optional[float] = None,
) -> None:
    """Convenience: flatten obs, add to buffer. Handles normalization."""
    obs_flat = _flatten_obs(obs, self._obs_space)
    next_obs_flat = _flatten_obs(next_obs, self._obs_space)

    if self._normalizer is not None:
        self._normalizer.update(obs_flat)
        obs_flat = self._normalizer.normalize(obs_flat)
        next_obs_flat = self._normalizer.normalize(next_obs_flat)

    act = np.array([action], dtype=np.float32) if self._is_discrete_action \
        else np.asarray(action, dtype=np.float32)
    self.add(obs_flat, act, reward, next_obs_flat, done, priority)


FastPERBuffer.add_env_step = _add_env_step


# ---------------------------------------------------------------------------
# GymCollector: manages the env-step → buffer loop
# ---------------------------------------------------------------------------

class GymCollector:
    """Collects experience from a Gymnasium environment into a FastPERBuffer.

    Handles:
      - Episode resets
      - Observation flattening and normalization
      - Discrete and continuous action spaces

    Parameters
    ----------
    env : gymnasium.Env
        The environment to collect from.
    buf : FastPERBuffer
        Buffer created with FastPERBuffer.from_env(env, ...).
    policy : callable(obs_tensor) -> action
        Policy function. Receives a (1, *obs_shape) float32 tensor,
        returns an action (int for discrete, array for continuous).
    device : str
        Device for the observation tensor passed to policy.
    """

    def __init__(
        self,
        env,
        buf: FastPERBuffer,
        policy: Callable,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.buf = buf
        self.policy = policy
        self.device = device

        self._obs = None
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._total_steps = 0
        self._total_episodes = 0
        self._reset()

    def _reset(self) -> None:
        obs, _ = self.env.reset()
        self._obs = _flatten_obs(obs, self.buf._obs_space)
        if self.buf._normalizer is not None:
            self.buf._normalizer.update(self._obs)
            self._obs = self.buf._normalizer.normalize(self._obs)
        self._episode_reward = 0.0
        self._episode_steps = 0

    def step(self, n_steps: int = 1) -> list[dict]:
        """Collect n_steps transitions into the buffer.

        Returns a list of episode_info dicts for any completed episodes.
        """
        completed_episodes = []

        for _ in range(n_steps):
            obs_t = torch.from_numpy(self._obs).unsqueeze(0).to(self.device)
            action = self.policy(obs_t)

            # Convert action to env-compatible format
            if self.buf._is_discrete_action:
                env_action = int(action) if not hasattr(action, 'item') else action.item()
            else:
                env_action = np.asarray(action, dtype=np.float32)

            next_obs_raw, reward, terminated, truncated, info = self.env.step(env_action)
            done = terminated or truncated

            next_obs = _flatten_obs(next_obs_raw, self.buf._obs_space)
            if self.buf._normalizer is not None:
                next_obs = self.buf._normalizer.normalize(next_obs)

            act_arr = np.array([env_action], dtype=np.float32) if self.buf._is_discrete_action \
                else np.asarray(env_action, dtype=np.float32)
            self.buf.add(self._obs, act_arr, float(reward), next_obs, done)

            self._episode_reward += float(reward)
            self._episode_steps += 1
            self._total_steps += 1

            if done:
                completed_episodes.append({
                    "reward": self._episode_reward,
                    "length": self._episode_steps,
                    "episode": self._total_episodes,
                })
                self._total_episodes += 1
                self._reset()
            else:
                self._obs = next_obs

        return completed_episodes

    @property
    def total_steps(self) -> int:
        return self._total_steps
