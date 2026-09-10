"""Tests for reservoir.nstep — N-step return wrapper."""

import numpy as np
import pytest

from reservoir.fast_buffer import FastPERBuffer
from reservoir.nstep import NStepBuffer

OBS_SHAPE = (4,)


def make_buf(capacity=256):
    return FastPERBuffer(capacity, OBS_SHAPE, alpha=0.6, beta=0.4)


def make_obs():
    return np.random.randn(*OBS_SHAPE).astype(np.float32)


class TestNStepBufferBasics:
    def test_n1_is_passthrough(self):
        """n=1 should behave identically to direct buffer.add()."""
        buf = make_buf()
        nb = NStepBuffer(buf, n=1, gamma=0.99)
        obs = make_obs()
        for _ in range(10):
            nb.add(obs, 0, 1.0, obs, False)
        assert buf.size == 10

    def test_n3_delays_storage(self):
        """With n=3, first transition is stored only after 3 adds."""
        buf = make_buf()
        nb = NStepBuffer(buf, n=3, gamma=0.99)
        obs = make_obs()
        nb.add(obs, 0, 1.0, obs, False)
        assert buf.size == 0
        nb.add(obs, 0, 1.0, obs, False)
        assert buf.size == 0
        nb.add(obs, 0, 1.0, obs, False)
        assert buf.size == 1

    def test_n_step_return_correct(self):
        """3-step return = r0 + γ*r1 + γ²*r2."""
        buf = make_buf()
        nb = NStepBuffer(buf, n=3, gamma=0.5)
        obs = make_obs()
        # Rewards: 1, 2, 4 → G = 1 + 0.5*2 + 0.25*4 = 1 + 1 + 1 = 3
        nb.add(obs, 0, 1.0, obs, False)
        nb.add(obs, 0, 2.0, obs, False)
        nb.add(obs, 0, 4.0, obs, False)
        assert buf.size == 1
        batch = buf.sample(1)
        assert abs(float(batch.rewards[0]) - 3.0) < 1e-5

    def test_episode_boundary_truncates_return(self):
        """If done=True at step k, return is truncated at k."""
        buf = make_buf()
        nb = NStepBuffer(buf, n=4, gamma=0.99)
        obs = make_obs()
        # Step 0: r=1, done=False
        # Step 1: r=2, done=True (episode ends)
        # → G = 1 + 0.99*2 = 2.98; transition stored immediately
        nb.add(obs, 0, 1.0, obs, False)
        nb.add(obs, 0, 2.0, obs, True)  # done
        assert buf.size == 2  # Both flushed at episode end

    def test_flush_all_on_done(self):
        """At episode end, all pending transitions are flushed."""
        buf = make_buf()
        nb = NStepBuffer(buf, n=5, gamma=0.99)
        obs = make_obs()
        for _ in range(3):
            nb.add(obs, 0, 1.0, obs, False)
        nb.add(obs, 0, 1.0, obs, True)  # episode end
        # All 4 transitions should be stored
        assert buf.size == 4

    def test_invalid_n_raises(self):
        buf = make_buf()
        with pytest.raises(ValueError):
            NStepBuffer(buf, n=0)

    def test_invalid_gamma_raises(self):
        buf = make_buf()
        with pytest.raises(ValueError):
            NStepBuffer(buf, n=3, gamma=1.5)

    def test_sample_delegates_to_buffer(self):
        buf = make_buf()
        nb = NStepBuffer(buf, n=2, gamma=0.99)
        obs = make_obs()
        for _ in range(100):
            nb.add(obs, 0, 1.0, obs, False)
        batch = nb.sample(32)
        assert batch.states.shape == (32, *OBS_SHAPE)

    def test_gamma_n_property(self):
        buf = make_buf()
        nb = NStepBuffer(buf, n=3, gamma=0.99)
        assert abs(nb.gamma_n - 0.99 ** 3) < 1e-9

    def test_multi_episode_accumulation(self):
        """Multiple episodes stored correctly."""
        buf = make_buf()
        nb = NStepBuffer(buf, n=3, gamma=0.99)
        obs = make_obs()
        # Episode 1: 5 steps
        for i in range(4):
            nb.add(obs, 0, 1.0, obs, False)
        nb.add(obs, 0, 1.0, obs, True)
        ep1_size = buf.size

        # Episode 2: 3 steps
        for i in range(2):
            nb.add(obs, 0, 1.0, obs, False)
        nb.add(obs, 0, 1.0, obs, True)

        assert buf.size > ep1_size

    def test_update_priorities_delegates(self):
        buf = make_buf()
        nb = NStepBuffer(buf, n=2, gamma=0.99)
        obs = make_obs()
        for _ in range(100):
            nb.add(obs, 0, 1.0, obs, False)
        batch = nb.sample(32)
        td_errors = np.random.rand(32)
        nb.update_priorities(batch.indices, td_errors)  # Should not raise


class TestNStepReturnMath:
    def test_discount_accumulates_correctly(self):
        """Verify the first-transition's n-step return G = Σ γ^k r_k.

        The first transition's return is the full n-step sum (no truncation).
        We read it from the buffer directly via the sum-tree leaf, since
        sample() uses priority-weighted random selection.
        """
        obs = make_obs()
        for n, gamma in [(2, 0.9), (3, 0.5), (5, 1.0)]:
            buf2 = make_buf()
            nb = NStepBuffer(buf2, n=n, gamma=gamma)
            rewards = list(range(1, n + 1))
            for k, r in enumerate(rewards):
                nb.add(obs, 0, float(r), obs, k == n - 1)

            # The first transition stored has G = Σ γ^k r_k for k in 0..n-1
            expected = sum(gamma ** k * r for k, r in enumerate(rewards[:n]))

            # Read all stored rewards and check the expected value is among them
            stored_rewards = buf2._rewards[:buf2.size]
            assert any(abs(r - expected) < 1e-4 for r in stored_rewards), (
                f"n={n}, gamma={gamma}: expected G={expected:.4f} not found in "
                f"stored rewards {stored_rewards.tolist()}"
            )
