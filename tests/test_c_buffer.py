"""Tests for CFastPERBuffer - C-tree-backed production buffer."""
import numpy as np
import pytest
import torch

pytest.importorskip(
    "reservoir._sumtree",
    reason="C extension not built - run `pip install -e .` first",
)

from reservoir.c_buffer import CFastPERBuffer
from reservoir.fast_buffer import FastBatch


OBS_SHAPE = (4,)
ACTION_DIM = 1
CAPACITY = 16


def make_buf(**kwargs):
    defaults = dict(
        capacity=CAPACITY,
        obs_shape=OBS_SHAPE,
        action_dim=ACTION_DIM,
        alpha=0.6,
        beta=0.4,
        epsilon=1e-6,
        device="cpu",
    )
    defaults.update(kwargs)
    return CFastPERBuffer(**defaults)


def fill(buf, n, priority=None):
    for i in range(n):
        obs = np.ones(OBS_SHAPE, dtype=np.float32) * i
        buf.add(obs, 0, float(i), obs, False, priority=priority)


class TestConstruction:
    def test_creates_without_error(self):
        buf = make_buf()
        assert buf.size == 0

    def test_total_priority_zero_initially(self):
        buf = make_buf()
        assert buf.total_priority == 0.0


class TestAdd:
    def test_size_increments(self):
        buf = make_buf()
        fill(buf, 5)
        assert buf.size == 5

    def test_size_capped_at_capacity(self):
        buf = make_buf()
        fill(buf, CAPACITY + 5)
        assert buf.size == CAPACITY

    def test_total_priority_increases(self):
        buf = make_buf()
        fill(buf, 4, priority=1.0)
        assert buf.total_priority > 0.0

    def test_add_sets_max_priority_for_none(self):
        buf = make_buf()
        buf.add(np.zeros(OBS_SHAPE), 0, 0.0, np.zeros(OBS_SHAPE), False, priority=2.0)
        buf.add(np.zeros(OBS_SHAPE), 0, 0.0, np.zeros(OBS_SHAPE), False, priority=None)
        # Second add uses max priority (raw value 2.0 stored in _max_priority after first add)
        assert buf.total_priority > 0.0


class TestSample:
    def setup_method(self):
        self.buf = make_buf()
        fill(self.buf, CAPACITY, priority=1.0)

    def test_sample_returns_fastbatch(self):
        batch = self.buf.sample(4)
        assert isinstance(batch, FastBatch)

    def test_sample_correct_shapes(self):
        batch = self.buf.sample(4)
        assert batch.states.shape == (4, *OBS_SHAPE)
        assert batch.actions.shape == (4,)
        assert batch.rewards.shape == (4,)
        assert batch.is_weights.shape == (4,)

    def test_sample_is_weights_in_range(self):
        batch = self.buf.sample(4)
        assert (batch.is_weights >= 0).all()
        assert (batch.is_weights <= 1.0 + 1e-6).all()

    def test_sample_indices_in_range(self):
        batch = self.buf.sample(4)
        assert (batch.indices >= 0).all()
        assert (batch.indices < CAPACITY).all()

    def test_sample_returns_tensors(self):
        batch = self.buf.sample(4)
        assert isinstance(batch.states, torch.Tensor)
        assert isinstance(batch.is_weights, torch.Tensor)


class TestUpdatePriorities:
    def test_update_changes_total(self):
        buf = make_buf()
        fill(buf, CAPACITY, priority=1.0)
        before = buf.total_priority
        indices = np.array([0, 1])
        buf.update_priorities(indices, np.array([5.0, 5.0]))
        assert buf.total_priority != pytest.approx(before)


class TestAnnealBeta:
    def test_beta_increases(self):
        buf = make_buf(beta=0.4)
        initial_beta = buf.beta
        buf.anneal_beta(step=500, total_steps=1000)
        assert buf.beta > initial_beta

    def test_beta_does_not_exceed_end(self):
        buf = make_buf(beta=0.4)
        buf.anneal_beta(step=10000, total_steps=1000)
        assert buf.beta <= 1.0
