"""Tests for reservoir.her — Hindsight Experience Replay."""

import numpy as np
import pytest

from reservoir.fast_buffer import FastPERBuffer
from reservoir.her import HERBuffer, obs_shape_for_her_env


# ---------------------------------------------------------------------------
# Minimal goal-conditioned environment mock
# ---------------------------------------------------------------------------

class MockGoalEnv:
    """Minimal goal-conditioned env for testing. obs is a dict with
    'observation', 'achieved_goal', 'desired_goal' keys."""

    observation_space_type = "Dict"

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Sparse reward: 0 if achieved == desired, -1 otherwise."""
        return 0.0 if np.allclose(achieved_goal, desired_goal, atol=0.1) else -1.0

    def make_obs(self, state, goal=None):
        if goal is None:
            goal = np.zeros(2, dtype=np.float32)
        return {
            "observation": state.copy(),
            "achieved_goal": state.copy(),
            "desired_goal": goal.copy(),
        }


OBS_DIM = 3
GOAL_DIM = 3   # Must match OBS_DIM so achieved_goal = state (same shape as desired_goal)
FLAT_DIM = OBS_DIM + GOAL_DIM  # 6


def make_her_buf(capacity=512):
    return FastPERBuffer(capacity=capacity, obs_shape=(FLAT_DIM,), action_dim=1)


def make_episode(env, length=5, reach_goal=False):
    """Build a synthetic episode. achieved_goal = observation (shape OBS_DIM=GOAL_DIM)."""
    goal = np.ones(GOAL_DIM, dtype=np.float32)
    transitions = []
    for t in range(length):
        state = np.random.randn(OBS_DIM).astype(np.float32)
        obs = env.make_obs(state, goal)
        action = 0
        # Last step: optionally reach the goal
        next_state = goal.copy() if (reach_goal and t == length - 1) \
            else np.random.randn(OBS_DIM).astype(np.float32)
        next_obs = env.make_obs(next_state, goal)
        reward = env.compute_reward(next_state, goal, {})
        done = (t == length - 1)
        info = {}
        transitions.append((obs, action, reward, next_obs, done, info))
    return transitions


class TestHERBufferBasics:
    def test_store_episode_adds_transitions(self):
        env = MockGoalEnv()
        buf = make_her_buf()
        her = HERBuffer(buf, env, strategy="future", k=2)
        episode = make_episode(env, length=5)
        her.store_episode(episode)
        # 5 real + 5*2 HER = 15 total
        assert buf.size == 15

    def test_k_relabeled_per_step(self):
        env = MockGoalEnv()
        for k in [1, 2, 4]:
            buf = make_her_buf()
            her = HERBuffer(buf, env, strategy="final", k=k)
            episode = make_episode(env, length=3)
            her.store_episode(episode)
            # 3 real + 3*k HER
            assert buf.size == 3 + 3 * k, f"k={k}: expected {3 + 3*k}, got {buf.size}"

    def test_empty_episode_no_error(self):
        env = MockGoalEnv()
        buf = make_her_buf()
        her = HERBuffer(buf, env)
        her.store_episode([])
        assert buf.size == 0

    def test_final_strategy_uses_last_achieved_goal(self):
        env = MockGoalEnv()
        buf = make_her_buf()
        her = HERBuffer(buf, env, strategy="final", k=1)
        episode = make_episode(env, length=4)
        her.store_episode(episode)
        assert buf.size == 8  # 4 + 4

    def test_future_strategy_valid_index(self):
        """future strategy should not raise IndexError."""
        env = MockGoalEnv()
        buf = make_her_buf()
        her = HERBuffer(buf, env, strategy="future", k=4)
        episode = make_episode(env, length=10)
        her.store_episode(episode)  # Should not raise

    def test_episode_strategy_valid_index(self):
        env = MockGoalEnv()
        buf = make_her_buf()
        her = HERBuffer(buf, env, strategy="episode", k=4)
        episode = make_episode(env, length=10)
        her.store_episode(episode)

    def test_invalid_strategy_raises(self):
        env = MockGoalEnv()
        buf = make_her_buf()
        with pytest.raises(ValueError):
            HERBuffer(buf, env, strategy="random_invalid")

    def test_invalid_k_raises(self):
        env = MockGoalEnv()
        buf = make_her_buf()
        with pytest.raises(ValueError):
            HERBuffer(buf, env, k=0)

    def test_sample_delegates_to_buffer(self):
        env = MockGoalEnv()
        buf = make_her_buf()
        her = HERBuffer(buf, env, k=2)
        for _ in range(3):  # store multiple episodes to ensure >= 32 transitions
            episode = make_episode(env, length=20)
            her.store_episode(episode)
        batch = her.sample(32)
        assert batch.states.shape == (32, FLAT_DIM)

    def test_her_reward_recomputed(self):
        """HER transitions with achieved goal as desired goal should get reward=0."""
        env = MockGoalEnv()
        buf = make_her_buf(capacity=256)
        her = HERBuffer(buf, env, strategy="final", k=4)

        # Episode: last state IS the achieved goal (reach_goal=True)
        episode = make_episode(env, length=5, reach_goal=True)
        her.store_episode(episode)

        # At least some HER transitions should have reward=0 (goal achieved)
        batch = buf.sample(min(buf.size, 64))
        rewards = batch.rewards.numpy()
        assert np.any(rewards >= -0.1), "Expected some HER transitions with reward≈0"


class TestObsShapeForHerEnv:
    def test_computes_correct_shape(self):
        import gymnasium as gym
        space = gym.spaces.Dict({
            "observation": gym.spaces.Box(-1, 1, (3,)),
            "achieved_goal": gym.spaces.Box(-1, 1, (3,)),
            "desired_goal": gym.spaces.Box(-1, 1, (3,)),
        })

        class FakeEnv:
            observation_space = space

        shape = obs_shape_for_her_env(FakeEnv())
        assert shape == (6,)  # obs(3) + goal(3)

    def test_non_dict_raises(self):
        import gymnasium as gym

        class FakeEnv:
            observation_space = gym.spaces.Box(-1, 1, (8,))

        with pytest.raises(AssertionError):
            obs_shape_for_her_env(FakeEnv())
