"""Tests for reservoir.gym_wrapper — Gymnasium integration."""

import numpy as np
import pytest
import gymnasium as gym

from reservoir.fast_buffer import FastPERBuffer
from reservoir.gym_wrapper import (
    GymCollector,
    RunningNormalizer,
    _flatten_obs,
    _obs_shape_from_space,
    _action_dim_from_space,
)
# Trigger from_env patch
import reservoir.gym_wrapper  # noqa


class TestObsShapeFromSpace:
    def test_box_space(self):
        space = gym.spaces.Box(low=-1, high=1, shape=(8,))
        assert _obs_shape_from_space(space) == (8,)

    def test_box_2d(self):
        space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
        assert _obs_shape_from_space(space) == (84, 84, 4)

    def test_discrete_space(self):
        space = gym.spaces.Discrete(5)
        assert _obs_shape_from_space(space) == (5,)

    def test_unsupported_raises(self):
        space = gym.spaces.MultiBinary(4)
        with pytest.raises(ValueError):
            _obs_shape_from_space(space)


class TestActionDimFromSpace:
    def test_discrete(self):
        space = gym.spaces.Discrete(4)
        dim, is_disc = _action_dim_from_space(space)
        assert dim == 1
        assert is_disc is True

    def test_box_continuous(self):
        space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        dim, is_disc = _action_dim_from_space(space)
        assert dim == 3
        assert is_disc is False


class TestFlattenObs:
    def test_box_passthrough(self):
        space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        result = _flatten_obs(obs, space)
        np.testing.assert_array_equal(result, obs.astype(np.float32))

    def test_discrete_one_hot(self):
        space = gym.spaces.Discrete(4)
        result = _flatten_obs(2, space)
        expected = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_dict_concatenates(self):
        space = gym.spaces.Dict({
            "obs": gym.spaces.Box(low=-1, high=1, shape=(3,)),
            "goal": gym.spaces.Box(low=-1, high=1, shape=(2,)),
        })
        obs = {"obs": np.ones(3), "goal": np.zeros(2)}
        result = _flatten_obs(obs, space)
        assert result.shape == (5,)


class TestFromEnv:
    def test_creates_buffer_with_correct_shape(self):
        env = gym.make("CartPole-v1")
        buf = FastPERBuffer.from_env(env, capacity=1000)
        assert buf.obs_shape == env.observation_space.shape
        assert buf.capacity == 1000
        env.close()

    def test_add_env_step_works(self):
        env = gym.make("CartPole-v1")
        buf = FastPERBuffer.from_env(env, capacity=1000)
        obs, _ = env.reset()
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        buf.add_env_step(obs, action, reward, next_obs, done or truncated)
        assert buf.size == 1
        env.close()

    def test_normalize_obs(self):
        env = gym.make("CartPole-v1")
        buf = FastPERBuffer.from_env(env, capacity=1000, normalize_obs=True)
        assert buf._normalizer is not None
        obs, _ = env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            next_obs, reward, done, _, _ = env.step(action)
            buf.add_env_step(obs, action, reward, next_obs, done)
            obs = next_obs
            if done:
                obs, _ = env.reset()
        assert buf.size > 0
        env.close()


class TestRunningNormalizer:
    def test_normalizes_after_warmup(self):
        norm = RunningNormalizer(shape=(4,))
        for _ in range(100):
            norm.update(np.random.randn(4))
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = norm.normalize(x)
        assert result.shape == (4,)
        assert result.dtype == np.float32

    def test_clips_extreme_values(self):
        norm = RunningNormalizer(shape=(1,), clip=5.0)
        for _ in range(50):
            norm.update(np.array([0.0]))
        result = norm.normalize(np.array([1000.0]))
        assert abs(result[0]) <= 5.0

    def test_returns_input_before_warmup(self):
        norm = RunningNormalizer(shape=(2,))
        x = np.array([3.0, 4.0])
        result = norm.normalize(x)
        np.testing.assert_array_almost_equal(result, x.astype(np.float32))


class TestGymCollector:
    def test_step_collects_transitions(self):
        env = gym.make("CartPole-v1")
        buf = FastPERBuffer.from_env(env, capacity=1000)

        def random_policy(obs_t):
            return env.action_space.sample()

        collector = GymCollector(env, buf, random_policy)
        collector.step(50)
        assert buf.size == 50
        env.close()

    def test_episode_info_on_done(self):
        env = gym.make("CartPole-v1")
        buf = FastPERBuffer.from_env(env, capacity=1000)

        def random_policy(obs_t):
            return env.action_space.sample()

        collector = GymCollector(env, buf, random_policy)
        episodes = collector.step(500)
        # CartPole episodes are short, should complete at least one
        assert len(episodes) >= 1
        assert "reward" in episodes[0]
        assert "length" in episodes[0]
        env.close()

    def test_total_steps_tracked(self):
        env = gym.make("CartPole-v1")
        buf = FastPERBuffer.from_env(env, capacity=1000)
        collector = GymCollector(env, buf, lambda obs: env.action_space.sample())
        collector.step(100)
        assert collector.total_steps == 100
        env.close()
