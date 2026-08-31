"""
Tests for reservoir.buffer — in-memory exact PER buffer.

Key properties:
1. insert/update operations update sum-tree and min-tree correctly.
2. sample() produces draws proportional to priorities.
3. IS weights are normalized to [0, 1] with maximum weight = 1.
4. For tiny buffers (N≤8), exhaustive probability counting verifies exact distribution.
5. Determinism: same seed/counter -> same samples.
6. Buffer wraps correctly at capacity.
"""

from fractions import Fraction
from typing import Any

import pytest

from reservoir.buffer import ExactPERBuffer, SampledBatch, Transition
from reservoir.sumtree import ExactMinTree


def make_transition(i: int) -> Transition:
    return Transition(state=i, action=0, reward=float(i), next_state=i + 1, done=False)


class TestTransition:
    def test_basic_construction(self):
        t = make_transition(0)
        assert t.state == 0
        assert t.action == 0
        assert t.reward == 0.0
        assert not t.done


class TestBufferBasics:
    def test_initial_size_is_zero(self):
        buf = ExactPERBuffer(capacity=8)
        assert buf.size == 0

    def test_capacity_rounded_up(self):
        buf = ExactPERBuffer(capacity=5)
        assert buf.capacity == 8

    def test_insert_increments_size(self):
        buf = ExactPERBuffer(capacity=8)
        buf.insert(make_transition(0), td_error=1.0)
        assert buf.size == 1
        buf.insert(make_transition(1), td_error=2.0)
        assert buf.size == 2

    def test_insert_returns_position(self):
        buf = ExactPERBuffer(capacity=4)
        pos0 = buf.insert(make_transition(0))
        pos1 = buf.insert(make_transition(1))
        pos2 = buf.insert(make_transition(2))
        assert pos0 == 0
        assert pos1 == 1
        assert pos2 == 2

    def test_circular_wrap_at_capacity(self):
        buf = ExactPERBuffer(capacity=4)
        for i in range(4):
            buf.insert(make_transition(i))
        assert buf.size == 4
        # One more insert wraps to position 0
        pos = buf.insert(make_transition(99))
        assert pos == 0
        assert buf.size == 4  # Size stays at capacity

    def test_insert_with_td_error(self):
        buf = ExactPERBuffer(capacity=4, alpha=1.0)
        pos = buf.insert(make_transition(0), td_error=1.0)
        assert buf._sum_tree.total > 0

    def test_insert_without_td_error_uses_max_priority(self):
        buf = ExactPERBuffer(capacity=4, alpha=1.0)
        # First insert with explicit priority
        buf.insert(make_transition(0), td_error=10.0)
        # Second insert without — should get max priority
        initial_max = buf._max_prio
        buf.insert(make_transition(1))  # no td_error
        p1 = buf._sum_tree.get(1)
        assert p1 == initial_max

    def test_update_priority(self):
        buf = ExactPERBuffer(capacity=4, alpha=1.0)
        buf.insert(make_transition(0), td_error=1.0)
        old_total = buf._sum_tree.total
        buf.update_priority(0, td_error=5.0)
        assert buf._sum_tree.total > old_total

    def test_verify_trees_after_operations(self):
        buf = ExactPERBuffer(capacity=8, alpha=0.6, beta=0.4)
        for i in range(8):
            buf.insert(make_transition(i), td_error=float(i + 1))
        assert buf.verify_trees()


class TestBufferSampling:
    def test_sample_returns_correct_types(self):
        buf = ExactPERBuffer(capacity=4, seed=0)
        for i in range(4):
            buf.insert(make_transition(i), td_error=float(i + 1))
        batch = buf.sample(2)
        assert isinstance(batch, SampledBatch)
        assert len(batch.indices) == 2
        assert len(batch.transitions) == 2
        assert len(batch.is_weights) == 2
        assert all(isinstance(w, Fraction) for w in batch.is_weights)

    def test_sample_indices_in_range(self):
        buf = ExactPERBuffer(capacity=8, seed=0)
        for i in range(8):
            buf.insert(make_transition(i), td_error=1.0)
        batch = buf.sample(4)
        for idx in batch.indices:
            assert 0 <= idx < buf.capacity

    def test_is_weights_in_0_1(self):
        buf = ExactPERBuffer(capacity=8, seed=0, beta=0.6)
        for i in range(8):
            buf.insert(make_transition(i), td_error=float(i + 1))
        batch = buf.sample(4)
        for w in batch.is_weights:
            assert Fraction(0) <= w <= Fraction(1), f"IS weight out of range: {w}"

    def test_max_is_weight_equals_one(self):
        """The maximum IS weight should be exactly 1 (corresponds to min priority).

        We verify this directly by computing the IS weight for a transition
        that has the minimum priority, rather than relying on sampling.
        """
        buf = ExactPERBuffer(capacity=4, seed=0, beta=0.6)
        buf.insert(make_transition(0), td_error=0.001)  # Low priority -> high IS weight
        buf.insert(make_transition(1), td_error=100.0)
        buf.insert(make_transition(2), td_error=0.001)
        buf.insert(make_transition(3), td_error=100.0)

        root_total = buf._sum_tree.total
        min_prio = buf._min_tree.minimum
        n = buf.size

        # IS weight for the minimum-priority transition should be exactly 1
        w_min = buf._compute_is_weight(min_prio, root_total, min_prio, n)
        assert w_min == Fraction(1), f"IS weight for min-priority should be 1, got {w_min}"

        # IS weight for a higher-priority transition should be < 1
        high_prio = buf._sum_tree.get(1)  # td_error=100.0 -> high priority
        w_high = buf._compute_is_weight(high_prio, root_total, min_prio, n)
        assert w_high < Fraction(1), f"IS weight for high-priority should be <1, got {w_high}"
        assert w_high > Fraction(0), f"IS weight should be positive"

    def test_deterministic_same_seed(self):
        """Same seed and state -> same samples."""
        buf1 = ExactPERBuffer(capacity=8, seed=42, buffer_id=0)
        buf2 = ExactPERBuffer(capacity=8, seed=42, buffer_id=0)
        for i in range(8):
            t = make_transition(i)
            buf1.insert(t, td_error=float(i + 1))
            buf2.insert(t, td_error=float(i + 1))

        b1 = buf1.sample(4)
        b2 = buf2.sample(4)
        assert b1.indices == b2.indices
        assert b1.draw_integers == b2.draw_integers
        assert b1.is_weights == b2.is_weights

    def test_different_seeds_differ(self):
        """Different seeds should produce different samples."""
        buf1 = ExactPERBuffer(capacity=8, seed=0, buffer_id=0)
        buf2 = ExactPERBuffer(capacity=8, seed=1, buffer_id=0)
        for i in range(8):
            t = make_transition(i)
            buf1.insert(t, td_error=1.0)
            buf2.insert(t, td_error=1.0)

        b1 = buf1.sample(4)
        b2 = buf2.sample(4)
        # Very likely to differ with different seeds
        assert b1.indices != b2.indices or b1.draw_integers != b2.draw_integers

    def test_sample_too_few_raises(self):
        buf = ExactPERBuffer(capacity=8, seed=0)
        buf.insert(make_transition(0), td_error=1.0)
        with pytest.raises(RuntimeError):
            buf.sample(2)

    def test_sample_zero_batch_raises(self):
        buf = ExactPERBuffer(capacity=4, seed=0)
        buf.insert(make_transition(0), td_error=1.0)
        with pytest.raises(ValueError):
            buf.sample(0)


class TestExhaustiveDistribution:
    """Exhaustive probability counting for tiny buffers (N <= 8).

    For a buffer with N transitions and integer priorities p_0, ..., p_{N-1},
    the probability of sampling transition i is p_i / sum(p_j).

    The key property: for draw_int in [0, total), prefix_sum_locate(draw_int)
    maps exactly p_i draw integers to position i. This is a pure integer
    identity — no statistics involved.

    NOTE: float_to_priority_int produces priorities ~2^52 (enormous).
    We CANNOT iterate over range(total) when total ~ N * 2^52 ~ 10^16.
    Instead, we:
    1. Inject small integer priorities directly into the sum-tree to test
       the counting property (these are the same tests as test_sumtree.py
       but now at the buffer API level with alpha=1.0 and carefully chosen TDs).
    2. Verify the theoretical probability formula as exact Fractions.
    """

    def _inject_small_priorities(self, buf: ExactPERBuffer, small_prios: list[int]) -> None:
        """Directly set leaf values in the sum-tree and min-tree for testing.

        This bypasses the float integerization to allow small integer priorities.
        """
        for i, p in enumerate(small_prios):
            buf._sum_tree.update(i, p)
            buf._min_tree.update(i, p)
            buf._transitions[i] = make_transition(i)
        buf._size = len(small_prios)
        buf._max_prio = max(small_prios)

    def test_exact_probability_match_n2_small_ints(self):
        """2-element buffer with small integer priorities: exact count check."""
        buf = ExactPERBuffer(capacity=2, alpha=1.0, beta=0.0, seed=0)
        small_prios = [10, 30]
        self._inject_small_priorities(buf, small_prios)

        total = buf._sum_tree.total
        assert total == 40

        counts = [0, 0]
        for draw_int in range(total):
            pos = buf._sum_tree.prefix_sum_locate(draw_int)
            counts[pos] += 1

        assert counts == small_prios, f"Expected {small_prios}, got {counts}"

    def test_exact_probability_match_n4_small_ints(self):
        """4-element buffer with small integer priorities."""
        buf = ExactPERBuffer(capacity=4, alpha=1.0, beta=0.0, seed=0)
        small_prios = [5, 15, 10, 20]
        self._inject_small_priorities(buf, small_prios)

        total = buf._sum_tree.total
        counts = [0] * 4
        for draw_int in range(total):
            pos = buf._sum_tree.prefix_sum_locate(draw_int)
            counts[pos] += 1

        assert counts == small_prios

    def test_exact_probability_match_n8_small_ints(self):
        """8-element buffer with small integer priorities."""
        buf = ExactPERBuffer(capacity=8, alpha=1.0, beta=0.0, seed=0)
        small_prios = [10, 30, 5, 20, 15, 25, 8, 12]  # Total = 125
        self._inject_small_priorities(buf, small_prios)

        total = buf._sum_tree.total
        assert total == sum(small_prios)

        counts = [0] * 8
        for draw_int in range(total):
            pos = buf._sum_tree.prefix_sum_locate(draw_int)
            counts[pos] += 1

        assert counts == small_prios

    def test_exact_probability_as_fractions(self):
        """Verify that priority_int / root_total equals the declared sampling probability."""
        buf = ExactPERBuffer(capacity=4, alpha=1.0, beta=0.0, seed=0)
        small_prios = [1, 2, 3, 4]  # Total = 10
        self._inject_small_priorities(buf, small_prios)

        total = buf._sum_tree.total
        assert total == 10

        # Declared probability for each position is exactly priority/total
        for i, p in enumerate(small_prios):
            declared_prob = Fraction(p, total)
            # Count draw integers mapping to position i
            count = sum(1 for d in range(total) if buf._sum_tree.prefix_sum_locate(d) == i)
            empirical_prob = Fraction(count, total)
            assert empirical_prob == declared_prob, (
                f"Position {i}: declared={declared_prob}, empirical={empirical_prob}"
            )

    def test_is_weight_formula_exact(self):
        """Verify IS weight formula matches manual computation."""
        buf = ExactPERBuffer(capacity=4, alpha=1.0, beta=1.0, seed=0)

        # Insert 4 transitions
        for i in range(4):
            buf.insert(make_transition(i), td_error=float(i + 1))

        batch = buf.sample(4)
        root_total = batch.root_total
        min_prio = batch.min_priority_int
        n = 4

        for idx, prio, w in zip(batch.indices, batch.priorities, batch.is_weights):
            # Manual IS weight computation (float64-once, then Fraction ratio)
            n_p_i = (n * prio) / root_total
            n_p_min = (n * min_prio) / root_total
            expected_w = Fraction(n_p_i ** (-1.0)) / Fraction(n_p_min ** (-1.0))
            assert w == expected_w, (
                f"IS weight mismatch at idx={idx}: got {w}, expected {expected_w}"
            )


class TestBufferEdgeCases:
    def test_all_equal_priorities(self):
        """All transitions have equal priority -> uniform distribution.

        Verify the tree structure is symmetric, not via full enumeration
        (priority ints are ~2^52, so total is too large to enumerate).
        """
        buf = ExactPERBuffer(capacity=4, alpha=1.0, beta=0.0, seed=0)
        for i in range(4):
            buf.insert(make_transition(i), td_error=1.0)

        p0 = buf._sum_tree.get(0)
        total = buf._sum_tree.total
        # All priorities equal -> total = 4 * p0
        assert total == 4 * p0
        # All leaves equal
        for i in range(4):
            assert buf._sum_tree.get(i) == p0
        # Tree invariant holds
        assert buf._sum_tree.verify_invariant()
        # Prefix-sum: each position covers exactly p0 draw integers
        # Boundary: position i starts at i * p0
        for i in range(4):
            # The first draw integer for position i is i * p0
            start = i * p0
            assert buf._sum_tree.prefix_sum_locate(start) == i
            if start > 0:
                # One before start maps to previous position
                assert buf._sum_tree.prefix_sum_locate(start - 1) == i - 1

    def test_single_element_buffer(self):
        buf = ExactPERBuffer(capacity=1, seed=0)
        buf.insert(make_transition(0), td_error=5.0)
        batch = buf.sample(1)
        assert batch.indices[0] == 0
        assert batch.is_weights[0] == Fraction(1)

    def test_alpha_zero_rejected(self):
        with pytest.raises(ValueError):
            ExactPERBuffer(capacity=4, alpha=0.0)

    def test_negative_alpha_rejected(self):
        with pytest.raises(ValueError):
            ExactPERBuffer(capacity=4, alpha=-0.6)
