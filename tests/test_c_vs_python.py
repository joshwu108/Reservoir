"""Property-based parity: CFastPERBuffer must produce identical results to FastPERBuffer.

Uses Hypothesis to generate random sequences of add/update_priorities/sample calls.
Both buffers receive the same inputs and the same numpy random seed; outputs must match.
"""
import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

pytest.importorskip(
    "reservoir._sumtree",
    reason="C extension not built",
)

from reservoir.c_buffer import CFastPERBuffer
from reservoir.fast_buffer import FastPERBuffer

OBS_SHAPE  = (4,)
ACTION_DIM = 1
CAPACITY   = 32
ALPHA      = 0.6
BETA       = 0.4
EPSILON    = 1e-6


def make_both():
    """Create a CFastPERBuffer and FastPERBuffer with identical config."""
    kwargs = dict(
        capacity=CAPACITY,
        obs_shape=OBS_SHAPE,
        action_dim=ACTION_DIM,
        alpha=ALPHA,
        beta=BETA,
        epsilon=EPSILON,
        device="cpu",
    )
    return CFastPERBuffer(**kwargs), FastPERBuffer(**kwargs)


def add_same(c_buf, py_buf, n_transitions, priorities):
    """Add n transitions with given float32 priorities to both buffers."""
    for i, p in zip(range(n_transitions), priorities):
        obs = np.ones(OBS_SHAPE, dtype=np.float32) * (i % 10)
        for buf in (c_buf, py_buf):
            buf.add(obs, 0, float(i % 5), obs, False, priority=float(p))


@given(
    priorities=st.lists(
        st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=CAPACITY,
        max_size=CAPACITY,
    ),
    batch_size=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50, deadline=5000)
def test_sample_identical(priorities, batch_size, seed):
    """CFastPERBuffer.sample() must return the same indices and IS weights as FastPERBuffer."""
    c_buf, py_buf = make_both()
    add_same(c_buf, py_buf, CAPACITY, priorities)

    assume(c_buf.size >= batch_size and py_buf.size >= batch_size)

    np.random.seed(seed)
    c_batch = c_buf.sample(batch_size)

    np.random.seed(seed)
    py_batch = py_buf.sample(batch_size)

    np.testing.assert_array_equal(
        c_batch.indices, py_batch.indices,
        err_msg="Sampled indices differ between C and Python buffers",
    )
    np.testing.assert_allclose(
        c_batch.is_weights.numpy(), py_batch.is_weights.numpy(),
        rtol=1e-5, atol=1e-6,
        err_msg="IS weights differ between C and Python buffers",
    )


@given(
    priorities=st.lists(
        st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=CAPACITY,
        max_size=CAPACITY,
    ),
)
@settings(max_examples=30, deadline=3000)
def test_total_priority_identical(priorities):
    """Both buffers must report identical total_priority after same inserts."""
    c_buf, py_buf = make_both()
    add_same(c_buf, py_buf, CAPACITY, priorities)
    assert c_buf.total_priority == pytest.approx(py_buf.total_priority, rel=1e-9)


@given(
    priorities=st.lists(
        st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=CAPACITY,
        max_size=CAPACITY,
    ),
    new_td_errors=st.lists(
        st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=4, max_size=4,
    ),
)
@settings(max_examples=30, deadline=3000)
def test_update_priorities_identical(priorities, new_td_errors):
    """update_priorities must produce identical total_priority in both buffers."""
    c_buf, py_buf = make_both()
    add_same(c_buf, py_buf, CAPACITY, priorities)

    indices = np.array([0, 1, 2, 3])
    td = np.array(new_td_errors, dtype=np.float64)
    c_buf.update_priorities(indices, td)
    py_buf.update_priorities(indices, td)

    assert c_buf.total_priority == pytest.approx(py_buf.total_priority, rel=1e-9)
