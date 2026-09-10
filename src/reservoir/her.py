"""
reservoir.her — Hindsight Experience Replay (HER) for goal-conditioned envs.

Reference: Andrychowicz et al. 2017, "Hindsight Experience Replay"
https://arxiv.org/abs/1707.01495

HER works with goal-conditioned environments where observations include
both the current state and the desired goal. When an episode fails to
achieve the goal, HER relabels some transitions with achieved goals
(states visited during the episode) as if they were the desired goal.

Supported relabeling strategies:
  - "final"   : use the final achieved state as the new goal
  - "future"  : use a random achieved state from AFTER the current step
  - "episode" : use a random achieved state from anywhere in the episode

Usage
-----
    # Environment must expose:
    #   obs = {"observation": ..., "achieved_goal": ..., "desired_goal": ...}
    # and have a compute_reward(achieved_goal, desired_goal, info) method.

    buf = FastPERBuffer.from_env(env, capacity=100_000)
    her = HERBuffer(buf, env, strategy="future", k=4)

    # Collect episodes
    obs, _ = env.reset()
    episode = []
    for step in range(max_steps):
        action = agent.act(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        episode.append((obs, action, reward, next_obs, done or truncated, info))
        obs = next_obs
        if done or truncated:
            her.store_episode(episode)
            episode = []
            obs, _ = env.reset()

    batch = her.sample(256)
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional, Tuple

from reservoir.fast_buffer import FastPERBuffer, FastBatch


# A transition as collected from a goal-conditioned env
EpisodeTransition = Tuple[dict, object, float, dict, bool, dict]


class HERBuffer:
    """Hindsight Experience Replay wrapper.

    Works with goal-conditioned environments whose observations are dicts
    with keys: "observation", "achieved_goal", "desired_goal".

    Parameters
    ----------
    buffer : FastPERBuffer
        Underlying replay buffer. Should be created with obs_shape matching
        the concatenated (observation + goal) size.
    env : gymnasium.Env
        The goal-conditioned environment (provides compute_reward).
    strategy : str
        Relabeling strategy: "final", "future", or "episode".
    k : int
        Number of HER relabeled transitions per real transition.
    compute_reward : callable or None
        Function (achieved_goal, desired_goal, info) -> float.
        If None, uses env.compute_reward.
    """

    def __init__(
        self,
        buffer: FastPERBuffer,
        env,
        strategy: str = "future",
        k: int = 4,
        compute_reward: Optional[Callable] = None,
    ) -> None:
        if strategy not in ("final", "future", "episode"):
            raise ValueError(f"strategy must be 'final', 'future', or 'episode', got {strategy!r}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.buffer = buffer
        self.env = env
        self.strategy = strategy
        self.k = k
        self._compute_reward = compute_reward or env.compute_reward

    def _obs_to_array(self, obs: dict) -> np.ndarray:
        """Concatenate observation + desired_goal into a flat array."""
        obs_part = np.asarray(obs["observation"], dtype=np.float32).flatten()
        goal_part = np.asarray(obs["desired_goal"], dtype=np.float32).flatten()
        return np.concatenate([obs_part, goal_part])

    def _obs_with_goal(self, obs: dict, goal: np.ndarray) -> np.ndarray:
        """Replace obs["desired_goal"] with a new goal and flatten."""
        obs_part = np.asarray(obs["observation"], dtype=np.float32).flatten()
        return np.concatenate([obs_part, goal.flatten()])

    def store_episode(self, episode: List[EpisodeTransition]) -> None:
        """Store one full episode, including HER relabeled transitions.

        Parameters
        ----------
        episode : list of (obs, action, reward, next_obs, done, info)
            The collected episode. obs and next_obs are dicts with keys
            "observation", "achieved_goal", "desired_goal".
        """
        if not episode:
            return

        T = len(episode)
        achieved_goals = [
            np.asarray(transition[3]["achieved_goal"], dtype=np.float32)
            for transition in episode
        ]  # next_obs["achieved_goal"] for each step

        # Store real transitions
        for t, (obs, action, reward, next_obs, done, info) in enumerate(episode):
            obs_arr = self._obs_to_array(obs)
            next_arr = self._obs_to_array(next_obs)
            act = np.array([action], dtype=np.float32) if np.isscalar(action) \
                else np.asarray(action, dtype=np.float32)
            self.buffer.add(obs_arr, act, reward, next_arr, done)

        # Store HER relabeled transitions
        for t, (obs, action, reward, next_obs, done, info) in enumerate(episode):
            for _ in range(self.k):
                # Sample a new goal using the chosen strategy
                new_goal = self._sample_goal(achieved_goals, t, T)

                # Recompute reward with the new goal
                achieved = np.asarray(next_obs["achieved_goal"], dtype=np.float32)
                her_reward = float(self._compute_reward(achieved, new_goal, info))

                obs_her = self._obs_with_goal(obs, new_goal)
                next_her = self._obs_with_goal(next_obs, new_goal)
                act = np.array([action], dtype=np.float32) if np.isscalar(action) \
                    else np.asarray(action, dtype=np.float32)

                # Episode is "done" if the relabeled goal was achieved
                her_done = her_reward > -0.5  # reward=0 means achieved, -1 means not

                self.buffer.add(obs_her, act, her_reward, next_her, her_done)

    def _sample_goal(
        self,
        achieved_goals: List[np.ndarray],
        t: int,
        T: int,
    ) -> np.ndarray:
        """Sample a new goal for HER relabeling."""
        if self.strategy == "final":
            return achieved_goals[-1]
        elif self.strategy == "future":
            # Random step AFTER t (or t itself if t is last)
            future_t = np.random.randint(t, T)
            return achieved_goals[future_t]
        else:  # episode
            episode_t = np.random.randint(0, T)
            return achieved_goals[episode_t]

    def sample(self, batch_size: int) -> FastBatch:
        return self.buffer.sample(batch_size)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        self.buffer.update_priorities(indices, td_errors)

    def anneal_beta(self, step: int, total_steps: int, beta_end: float = 1.0) -> None:
        self.buffer.anneal_beta(step, total_steps, beta_end)

    @property
    def size(self) -> int:
        return self.buffer.size


def obs_shape_for_her_env(env) -> tuple:
    """Compute obs_shape for a goal-conditioned env with Dict observation space.

    The flat obs = concatenate(observation, desired_goal).
    """
    import gymnasium as gym
    obs_space = env.observation_space
    assert isinstance(obs_space, gym.spaces.Dict), \
        "HER requires a Dict observation space with 'observation' and 'desired_goal' keys"
    obs_size = int(np.prod(obs_space["observation"].shape))
    goal_size = int(np.prod(obs_space["desired_goal"].shape))
    return (obs_size + goal_size,)
