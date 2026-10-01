"""Tests for reservoir.decayed_tree — the sum-tree with exact age decay.

``DecayedPriorityTree`` is the only object that holds (q, t) per leaf, the
base epoch and the current version. Everything it does must agree with
the pure functions in ``reservoir.decay`` recomputed from scratch, which
is what most of these tests check: after any sequence of writes, evicts
and advances, every live leaf equals ``inflated_priority(q, t, base_epoch)``
and the tree total equals the sum of the leaves.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import pytest
from hypothesis import given, settings, strategies as st

from reservoir.decay import (
    DecayParams,
    canonical_base_epoch,
    inflated_priority,
    is_expired,
    max_tree_total,
)
from reservoir.decayed_tree import (
    AdvanceResult,
    DecayedPriorityTree,
    RebaseEvent,
    WriteEvent,
)
from reservoir.sumtree import ExactMinTree


def params(**kw) -> DecayParams:
    defaults = dict(half_life=4, max_policy_age=16, capacity=8)
    defaults.update(kw)
    return DecayParams(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_fresh_tree_state(self) -> None:
        t = DecayedPriorityTree(params())
        assert t.capacity == 8
        assert t.base_epoch == 0
        assert t.current_version == 0
        assert t.total == 0
        assert t.minimum == ExactMinTree.INFINITY
        assert t.live_count == 0
        assert t.live_positions() == ()
        assert t.verify_invariant()

    @pytest.mark.parametrize("cap", [1, 2, 3, 50_000, 1 << 16])
    def test_capacity_rounds_up_within_params_bit_budget(self, cap: int) -> None:
        p = params(capacity=cap)
        t = DecayedPriorityTree(p)
        assert t.capacity >= cap
        assert t.capacity & (t.capacity - 1) == 0  # power of two
        # The bit budget in DecayParams already accounts for the rounding.
        assert (t.capacity - 1).bit_length() <= p.capacity_bits

    def test_rejects_non_params(self) -> None:
        with pytest.raises(TypeError):
            DecayedPriorityTree({"half_life": 4})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------

class TestWrite:
    def test_leaf_equals_inflated_priority(self) -> None:
        p = params()
        t = DecayedPriorityTree(p)
        t.advance(5)
        ev = t.write(3, q=1000, entry_version=5)
        expected = inflated_priority(1000, 5, 0, p)
        assert t.leaf(3) == expected
        assert t.minimum == expected
        assert t.total == expected
        assert t.entry(3) == (1000, 5)
        assert t.live_positions() == (3,)

    def test_write_event_fields(self) -> None:
        p = params()
        t = DecayedPriorityTree(p)
        ev = t.write(0, q=7, entry_version=0)
        assert isinstance(ev, WriteEvent)
        assert ev.op == "insert"
        assert ev.position == 0
        assert ev.old_leaf == 0
        assert ev.new_leaf == inflated_priority(7, 0, 0, p)
        assert ev.base_priority_int == 7
        assert ev.entry_version == 0
        assert ev.base_epoch == 0
        assert ev.reason is None
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.position = 1  # type: ignore[misc]

    def test_second_write_to_live_slot_is_update(self) -> None:
        t = DecayedPriorityTree(params())
        t.write(0, q=7, entry_version=0)
        first_leaf = t.leaf(0)
        ev = t.write(0, q=9, entry_version=0)
        assert ev.op == "update"
        assert ev.old_leaf == first_leaf
        assert t.live_count == 1

    def test_zero_q_entry_is_live_but_not_the_minimum(self) -> None:
        t = DecayedPriorityTree(params())
        t.write(0, q=5, entry_version=0)
        t.write(1, q=0, entry_version=0)
        assert t.leaf(1) == 0
        assert t.live_count == 2
        assert 1 in t.live_positions()
        # A zero leaf can never be sampled, so it must not drive IS-weight normalisation.
        assert t.minimum == t.leaf(0)

    def test_updating_to_zero_q_removes_from_min_tree(self) -> None:
        t = DecayedPriorityTree(params())
        t.write(0, q=5, entry_version=0)
        t.write(1, q=3, entry_version=0)
        t.write(1, q=0, entry_version=0)
        assert t.minimum == t.leaf(0)

    def test_rejects_entry_newer_than_current_version(self) -> None:
        t = DecayedPriorityTree(params())
        with pytest.raises(ValueError, match="newer"):
            t.write(0, q=1, entry_version=1)

    def test_rejects_expired_entry(self) -> None:
        t = DecayedPriorityTree(params(max_policy_age=2))
        t.advance(10)
        with pytest.raises(ValueError, match="expired"):
            t.write(0, q=1, entry_version=7)
        t.write(0, q=1, entry_version=8)  # exactly max_policy_age old is live

    @pytest.mark.parametrize("pos", [-1, 8, 100])
    def test_rejects_position_out_of_range(self, pos: int) -> None:
        t = DecayedPriorityTree(params())
        with pytest.raises(ValueError, match="osition"):
            t.write(pos, q=1, entry_version=0)

    def test_rejects_q_out_of_range(self) -> None:
        p = params()
        t = DecayedPriorityTree(p)
        with pytest.raises(ValueError):
            t.write(0, q=1 << p.priority_bits, entry_version=0)
        with pytest.raises(ValueError):
            t.write(0, q=-1, entry_version=0)

    def test_failed_write_leaves_tree_unchanged(self) -> None:
        t = DecayedPriorityTree(params())
        t.write(0, q=5, entry_version=0)
        before = (t.total, t.minimum, t.live_positions(), t.entry(0))
        with pytest.raises(ValueError):
            t.write(0, q=-1, entry_version=0)
        assert (t.total, t.minimum, t.live_positions(), t.entry(0)) == before


# ---------------------------------------------------------------------------
# evict
# ---------------------------------------------------------------------------

class TestEvict:
    def test_evict_zeroes_leaf_and_forgets_entry(self) -> None:
        t = DecayedPriorityTree(params())
        t.write(0, q=5, entry_version=0)
        t.write(1, q=3, entry_version=0)
        leaf0 = t.leaf(0)
        ev = t.evict(0, reason="explicit")
        assert ev.op == "evict"
        assert ev.reason == "explicit"
        assert ev.old_leaf == leaf0
        assert ev.new_leaf == 0
        assert ev.base_priority_int == 5
        assert ev.entry_version == 0
        assert t.leaf(0) == 0
        assert t.live_positions() == (1,)
        assert t.minimum == t.leaf(1)
        assert t.total == t.leaf(1)

    def test_evicting_last_entry_resets_minimum_to_infinity(self) -> None:
        t = DecayedPriorityTree(params())
        t.write(0, q=5, entry_version=0)
        t.evict(0, reason="explicit")
        assert t.minimum == ExactMinTree.INFINITY
        assert t.total == 0

    def test_evict_non_live_position_raises(self) -> None:
        t = DecayedPriorityTree(params())
        with pytest.raises(ValueError, match="live"):
            t.evict(0, reason="explicit")

    def test_evict_bad_reason_raises(self) -> None:
        t = DecayedPriorityTree(params())
        t.write(0, q=5, entry_version=0)
        with pytest.raises(ValueError, match="reason"):
            t.evict(0, reason="because")

    def test_entry_of_non_live_position_raises(self) -> None:
        t = DecayedPriorityTree(params())
        with pytest.raises(ValueError, match="live"):
            t.entry(0)


# ---------------------------------------------------------------------------
# advance: staleness eviction and rebase
# ---------------------------------------------------------------------------

class TestAdvance:
    def test_monotone(self) -> None:
        t = DecayedPriorityTree(params())
        t.advance(5)
        with pytest.raises(ValueError, match="monoton"):
            t.advance(4)
        res = t.advance(5)  # same version is a no-op
        assert res.evicted == () and res.rebase is None

    def test_result_fields(self) -> None:
        t = DecayedPriorityTree(params())
        res = t.advance(3)
        assert isinstance(res, AdvanceResult)
        assert res.old_version == 0
        assert res.new_version == 3
        assert t.current_version == 3

    def test_evicts_exactly_the_expired_entries(self) -> None:
        p = params(max_policy_age=3)
        t = DecayedPriorityTree(p)
        t.advance(10)
        t.write(0, q=1, entry_version=7)   # age 3 at v=10 → live, expires at 11
        t.write(1, q=1, entry_version=8)
        t.write(2, q=1, entry_version=10)
        res = t.advance(11)
        assert [e.position for e in res.evicted] == [0]
        assert res.evicted[0].reason == "stale"
        assert t.live_positions() == (1, 2)
        res = t.advance(13)
        assert [e.position for e in res.evicted] == [1]
        assert t.live_positions() == (2,)

    def test_evictions_are_in_position_order(self) -> None:
        p = params(max_policy_age=0)
        t = DecayedPriorityTree(p)
        for pos in (5, 2, 7, 0):
            t.write(pos, q=1, entry_version=0)
        res = t.advance(1)
        assert [e.position for e in res.evicted] == [0, 2, 5, 7]
        assert t.live_count == 0

    def test_no_rebase_while_shift_fits(self) -> None:
        p = params(half_life=4, max_policy_age=16)  # max_shift = 4
        t = DecayedPriorityTree(p)
        t.write(0, q=1, entry_version=0)
        res = t.advance(4 * p.max_shift + 3)  # newest epoch == max_shift
        assert res.rebase is None
        assert t.base_epoch == 0

    def test_rebase_when_shift_exceeded(self) -> None:
        p = params(half_life=4, max_policy_age=16)  # max_shift = 4
        t = DecayedPriorityTree(p)
        v = 4 * p.max_shift + 4  # first version in epoch max_shift + 1
        res = t.advance(v)
        assert res.rebase is not None
        assert isinstance(res.rebase, RebaseEvent)
        assert res.rebase.old_base_epoch == 0
        assert res.rebase.new_base_epoch == canonical_base_epoch(v, p)
        assert t.base_epoch == res.rebase.new_base_epoch

    def test_rebase_shifts_every_live_leaf_and_total(self) -> None:
        p = params(half_life=2, max_policy_age=4)  # max_shift = 2
        t = DecayedPriorityTree(p)
        t.advance(4)
        t.write(0, q=100, entry_version=4)
        t.write(1, q=200, entry_version=3)
        t.write(2, q=0, entry_version=4)
        before = {pos: t.leaf(pos) for pos in range(t.capacity)}
        total_before = t.total
        res = t.advance(7)  # epoch 3 > max_shift 2 → rebase; nothing expires (age <= 4)
        assert res.evicted == ()
        assert res.rebase is not None
        shift = res.rebase.new_base_epoch - res.rebase.old_base_epoch
        assert shift >= 1
        for pos in range(t.capacity):
            assert t.leaf(pos) == before[pos] >> shift
            assert (before[pos] >> shift) << shift == before[pos]  # no bits lost
        assert res.rebase.root_total_before == total_before
        assert res.rebase.root_total_after == t.total == total_before >> shift
        assert res.rebase.shifted == tuple(
            (pos, before[pos], before[pos] >> shift) for pos in (0, 1, 2)
        )
        assert t.verify_invariant()
        # Leaves still equal a fresh derivation at the new base.
        for pos in (0, 1, 2):
            q, ver = t.entry(pos)
            assert t.leaf(pos) == inflated_priority(q, ver, t.base_epoch, p)

    def test_rebase_preserves_min_tree_including_infinity_sentinels(self) -> None:
        p = params(half_life=2, max_policy_age=4)
        t = DecayedPriorityTree(p)
        t.advance(4)
        t.write(0, q=100, entry_version=4)
        t.write(1, q=300, entry_version=4)
        min_before = t.minimum
        res = t.advance(7)
        shift = res.rebase.new_base_epoch - res.rebase.old_base_epoch
        assert t.minimum == min_before >> shift
        t.evict(0, reason="explicit")
        t.evict(1, reason="explicit")
        assert t.minimum == ExactMinTree.INFINITY  # sentinel untouched by the shift

    def test_stale_entries_are_evicted_before_the_rebase(self) -> None:
        # An entry too old for the new base must leave before the shift; if it
        # were shifted, rebase_priority would raise on its dropped bits.
        p = params(half_life=1, max_policy_age=1)  # max_shift = 1
        t = DecayedPriorityTree(p)
        t.write(0, q=3, entry_version=0)
        t.advance(1)
        t.write(1, q=3, entry_version=1)
        res = t.advance(3)  # entry 0 (age 3) and entry 1 (age 2) both expire; rebase follows
        assert [e.position for e in res.evicted] == [0, 1]
        assert res.rebase is not None
        assert t.total == 0
        assert t.verify_invariant()

    def test_long_idle_jump_with_no_survivors_does_not_fail(self) -> None:
        # A jump of >= 64 epochs is only possible when nothing survives, and
        # shifting all-zero arrays by any amount must be a clean no-op.
        p = params(half_life=4, max_policy_age=16)
        t = DecayedPriorityTree(p)
        t.write(0, q=5, entry_version=0)
        t.write(3, q=9, entry_version=0)
        res = t.advance(1000)
        assert [e.position for e in res.evicted] == [0, 3]
        assert res.rebase is not None and res.rebase.shifted == ()
        assert res.rebase.root_total_before == 0 == res.rebase.root_total_after
        assert t.base_epoch == canonical_base_epoch(1000, p)
        assert t.current_version == 1000
        assert t.total == 0 and t.minimum == ExactMinTree.INFINITY
        assert t.verify_invariant()
        # The tree keeps working afterwards.
        t.write(1, q=5, entry_version=1000)
        assert t.leaf(1) == inflated_priority(5, 1000, t.base_epoch, p)
        t.advance(5000)
        assert t.live_count == 0 and t.total == 0

    def test_rebase_changes_no_sampling_decision(self) -> None:
        p = params(half_life=2, max_policy_age=6)  # max_shift = 3
        t = DecayedPriorityTree(p)
        t.advance(6)
        for pos, q in enumerate([5, 0, 17, 9, 1]):
            t.write(pos, q=q, entry_version=6 - pos)
        total_before = t.total
        located_before = [t.prefix_sum_locate(d) for d in range(total_before)]
        probs_before = [Fraction(t.leaf(pos), total_before) for pos in range(t.capacity)]
        res = t.advance(9)  # epoch 4 > 3 → rebase; oldest entry (v=2, age 7) expires
        assert res.rebase is not None
        shift = res.rebase.new_base_epoch - res.rebase.old_base_epoch
        evicted = {e.position for e in res.evicted}
        assert evicted == {4}
        for d in range(total_before):
            pos = located_before[d]
            if pos in evicted:
                continue
            assert t.prefix_sum_locate(d >> shift) == pos or (d >> shift) >= t.total
        # Surviving entries keep their relative probabilities exactly.
        survivors = [pos for pos in range(t.capacity) if pos not in evicted and t.leaf(pos) > 0]
        for a in survivors:
            for b in survivors:
                assert Fraction(t.leaf(a), t.leaf(b)) == probs_before[a] / probs_before[b]


# ---------------------------------------------------------------------------
# Properties: the tree always agrees with decay.py recomputed from scratch
# ---------------------------------------------------------------------------

@st.composite
def small_params(draw: st.DrawFn) -> DecayParams:
    """Small DecayParams that always satisfy the 64-bit budget.

    The budget is ``priority_bits + 1 + max_shift + capacity_bits <= 64``
    with ``max_shift = ceil(max_policy_age / half_life) + rebase_slack``.
    ``max_policy_age`` is drawn last and bounded so the inequality holds,
    instead of letting DecayParams reject the draw.
    """
    half_life = draw(st.integers(1, 6))
    capacity = draw(st.integers(1, 8))
    rebase_slack = draw(st.integers(0, 2))
    capacity_bits = (capacity - 1).bit_length()
    max_epochs = 64 - 32 - 1 - capacity_bits - rebase_slack  # headroom for ceil(a / h)
    max_policy_age = draw(st.integers(0, min(30, max_epochs * half_life)))
    return DecayParams(
        half_life=half_life,
        max_policy_age=max_policy_age,
        capacity=capacity,
        rebase_slack=rebase_slack,
    )


def _check_against_reference(t: DecayedPriorityTree, p: DecayParams, ref: dict) -> None:
    """ref: {position: (q, t)} — recompute every leaf and the totals from it."""
    expected_leaves = {
        pos: inflated_priority(q, v, t.base_epoch, p) for pos, (q, v) in ref.items()
    }
    for pos in range(t.capacity):
        assert t.leaf(pos) == expected_leaves.get(pos, 0)
    assert t.total == sum(expected_leaves.values())
    positive = [leaf for leaf in expected_leaves.values() if leaf > 0]
    assert t.minimum == (min(positive) if positive else ExactMinTree.INFINITY)
    assert t.live_positions() == tuple(sorted(ref))
    assert t.live_count == len(ref)
    assert t.base_epoch == canonical_base_epoch(t.current_version, p) or (
        t.current_version // p.half_life - t.base_epoch <= p.max_shift
    )
    assert t.total <= max_tree_total(p)
    assert t.total < 1 << 64
    assert t.verify_invariant()


@given(p=small_params(), data=st.data())
@settings(max_examples=200, deadline=None)
def test_random_operation_sequences_match_reference(p: DecayParams, data: st.DataObject) -> None:
    t = DecayedPriorityTree(p)
    ref: dict[int, tuple[int, int]] = {}
    max_q = (1 << p.priority_bits) - 1
    n_ops = data.draw(st.integers(1, 40))
    for _ in range(n_ops):
        kind = data.draw(st.sampled_from(["write", "write", "advance", "evict"]))
        if kind == "write":
            pos = data.draw(st.integers(0, t.capacity - 1))
            q = data.draw(st.integers(0, max_q))
            oldest = max(0, t.current_version - p.max_policy_age)
            v = data.draw(st.integers(oldest, t.current_version))
            ev = t.write(pos, q=q, entry_version=v)
            assert ev.op == ("update" if pos in ref else "insert")
            ref[pos] = (q, v)
        elif kind == "advance":
            new_v = t.current_version + data.draw(st.integers(0, 3 * p.half_life + 2))
            res = t.advance(new_v)
            expired = sorted(pos for pos, (q, v) in ref.items() if is_expired(v, new_v, p))
            assert [e.position for e in res.evicted] == expired
            for pos in expired:
                del ref[pos]
        else:
            if not ref:
                continue
            pos = data.draw(st.sampled_from(sorted(ref)))
            t.evict(pos, reason="explicit")
            del ref[pos]
        _check_against_reference(t, p, ref)


@given(p=small_params(), data=st.data())
@settings(max_examples=100, deadline=None)
def test_equal_q_entries_h_versions_apart_differ_by_exactly_a_factor_of_two(
    p: DecayParams, data: st.DataObject
) -> None:
    if p.capacity < 2 or p.max_policy_age < p.half_life:
        return
    t = DecayedPriorityTree(p)
    m = data.draw(st.integers(1, p.max_policy_age // p.half_life))
    base_v = data.draw(st.integers(0, 50))
    t.advance(base_v + m * p.half_life)
    q = data.draw(st.integers(1, (1 << p.priority_bits) - 1))
    t.write(0, q=q, entry_version=base_v)
    t.write(1, q=q, entry_version=base_v + m * p.half_life)
    assert t.leaf(1) == t.leaf(0) << m
    # And it survives rebases that happen on further advances.
    t.advance(t.current_version + data.draw(st.integers(0, 4 * p.half_life)))
    if 0 in t.live_positions() and 1 in t.live_positions():
        assert t.leaf(1) == t.leaf(0) << m
