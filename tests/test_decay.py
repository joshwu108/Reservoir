"""
Tests for reservoir.decay — exact age-decayed priorities in uint64.

The reference model in this file is deliberately independent of the
implementation's tree representation. It computes the DECLARED absolute
weight of every entry with fractions.Fraction,

    W_i = floor(q_i * T[t_i mod h] / 2^F) * 2^(t_i // h)

where T[k] is characterised (not recomputed) by the integer certificate
T[k]^h <= 2^(k + F*h) < (T[k] + 1)^h, and samples by brute-force prefix sums.
"""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from reservoir import decay
from reservoir.decay import (
    DEFAULT_PRIORITY_BITS,
    DEFAULT_TABLE_FRAC_BITS,
    INTRA_EPOCH_BITS,
    MAX_HALF_LIFE,
    UINT64_BITS,
    DecayParams,
    canonical_base_epoch,
    decay_table,
    inflated_priority,
    is_expired,
    max_leaf_value,
    max_tree_total,
    pending_rebase_shift,
    quantize_priority,
    rebase_priorities,
    rebase_priority,
    version_after_update,
)
from reservoir.sumtree import ExactSumTree

UINT64_LIMIT = 1 << 64


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------

def reference_absolute_weight(q: int, version: int, params: DecayParams) -> Fraction:
    """Declared weight relative to epoch 0 (unbounded, base-independent)."""
    table = decay_table(params.half_life, params.table_frac_bits)
    epoch, phase = divmod(version, params.half_life)
    mantissa = (q * table[phase]) // (1 << params.table_frac_bits)
    return Fraction(mantissa) * Fraction(2) ** epoch


def reference_locate(weights: list[Fraction], draw: Fraction) -> int:
    """First index whose cumulative probability exceeds draw in [0, 1)."""
    total = sum(weights, Fraction(0))
    running = Fraction(0)
    for index, weight in enumerate(weights):
        running += weight
        if draw < running / total:
            return index
    raise AssertionError("draw outside [0, 1)")


def fill_tree(capacity: int, leaves: list[int]) -> ExactSumTree:
    tree = ExactSumTree(capacity)
    for position, value in enumerate(leaves):
        assert 0 <= value < UINT64_LIMIT
        tree.update(position, value)
    assert tree.total < UINT64_LIMIT
    return tree


def boundary_draws(leaves: list[int]) -> list[int]:
    """Every prefix-sum boundary and its predecessor: the hard cases."""
    total = sum(leaves)
    draws = {0, total - 1}
    running = 0
    for value in leaves:
        running += value
        draws.update(d for d in (running - 1, running) if 0 <= d < total)
    return sorted(draws)


@st.composite
def params_and_entries(draw: st.DrawFn) -> tuple[DecayParams, int, list[tuple[int, int]]]:
    """Valid params, a current version, and live (q, version) entries."""
    half_life = draw(st.integers(1, 24))
    capacity = draw(st.integers(1, 16))
    max_policy_age = draw(st.integers(0, half_life * 20))
    rebase_slack = draw(st.integers(0, 3))
    params = DecayParams(
        half_life=half_life,
        max_policy_age=max_policy_age,
        capacity=capacity,
        rebase_slack=rebase_slack,
    )
    current_version = draw(st.integers(0, 5000))
    oldest = max(0, current_version - max_policy_age)
    entries = draw(
        st.lists(
            st.tuples(
                st.integers(0, (1 << params.priority_bits) - 1),
                st.integers(oldest, current_version),
            ),
            min_size=1,
            max_size=capacity,
        )
    )
    return params, current_version, entries


@st.composite
def rebase_case(draw: st.DrawFn) -> tuple[DecayParams, int, list[tuple[int, int]], int]:
    """Valid params, current version, live entries, and a version advance.

    Constructed rather than filtered, so Hypothesis never discards draws:

    - one entry has a positive base priority and is written recently
      enough to survive the advance, so the rebased tree has a non-zero
      total;
    - ``max_policy_age`` is at least 1, since with 0 any advance expires
      every entry;
    - entries are shuffled so the survivor is not always leaf 0.
    """
    half_life = draw(st.integers(1, 24))
    capacity = draw(st.integers(1, 16))
    max_policy_age = draw(st.integers(1, half_life * 20))
    rebase_slack = draw(st.integers(0, 3))
    params = DecayParams(
        half_life=half_life,
        max_policy_age=max_policy_age,
        capacity=capacity,
        rebase_slack=rebase_slack,
    )
    current_version = draw(st.integers(0, 5000))
    oldest = max(0, current_version - max_policy_age)
    advance = draw(st.integers(1, max_policy_age))
    # An entry stays live while current_version - entry_version <= max_policy_age.
    # The survivor is written at most (max_policy_age - advance) versions ago,
    # so after advancing by `advance` its age is at most max_policy_age.
    survivor_version = draw(
        st.integers(
            max(oldest, current_version - (max_policy_age - advance)),
            current_version,
        )
    )
    max_q = (1 << params.priority_bits) - 1
    survivor = (draw(st.integers(1, max_q)), survivor_version)
    others = draw(
        st.lists(
            st.tuples(st.integers(0, max_q), st.integers(oldest, current_version)),
            min_size=0,
            max_size=capacity - 1,
        )
    )
    # Shuffle so zero-weight and expired entries can land at leaf 0 too;
    # prefix_sum_locate must skip leading zero-weight leaves correctly.
    entries = draw(st.permutations([survivor] + others))
    return params, current_version, entries, advance


# ---------------------------------------------------------------------------
# Decay table
# ---------------------------------------------------------------------------

class TestDecayTable:
    @pytest.mark.parametrize("half_life", [1, 2, 3, 7, 64, 347])
    @pytest.mark.parametrize("frac_bits", [1, 8, 16, 31])
    def test_entries_satisfy_integer_root_certificate(
        self, half_life: int, frac_bits: int
    ) -> None:
        table = decay_table(half_life, frac_bits)

        assert len(table) == half_life
        for k, value in enumerate(table):
            target = 1 << (k + frac_bits * half_life)
            assert value**half_life <= target < (value + 1) ** half_life

    def test_first_entry_is_exactly_one(self) -> None:
        assert decay_table(5, 31)[0] == 1 << 31

    def test_entries_lie_in_one_octave_and_increase(self) -> None:
        table = decay_table(100, DEFAULT_TABLE_FRAC_BITS)

        assert all(a < b for a, b in zip(table, table[1:]))
        assert table[0] == 1 << DEFAULT_TABLE_FRAC_BITS
        assert table[-1] < 1 << (DEFAULT_TABLE_FRAC_BITS + INTRA_EPOCH_BITS)

    def test_table_is_immutable_tuple_of_ints(self) -> None:
        table = decay_table(4, 16)

        assert isinstance(table, tuple)
        assert all(type(v) is int for v in table)

    @pytest.mark.parametrize("half_life", [0, -1, MAX_HALF_LIFE + 1])
    def test_bad_half_life_raises(self, half_life: int) -> None:
        with pytest.raises(ValueError):
            decay_table(half_life, 16)

    @pytest.mark.parametrize("half_life", [1.5, "3", True, None])
    def test_non_integer_half_life_raises(self, half_life: object) -> None:
        with pytest.raises(ValueError):
            decay_table(half_life, 16)  # type: ignore[arg-type]

    @pytest.mark.parametrize("frac_bits", [0, -1, 64])
    def test_bad_frac_bits_raises(self, frac_bits: int) -> None:
        with pytest.raises(ValueError):
            decay_table(4, frac_bits)


# ---------------------------------------------------------------------------
# Parameter validation and the bit budget
# ---------------------------------------------------------------------------

class TestBitBudget:
    def test_budget_boundary_is_accepted(self) -> None:
        # 32 + 1 + 15 + 16 == 64
        params = DecayParams(half_life=1, max_policy_age=15, capacity=1 << 16)

        assert params.max_shift == 15
        assert params.capacity_bits == 16
        assert params.tree_bits == UINT64_BITS

    def test_one_bit_over_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="bit budget"):
            DecayParams(half_life=1, max_policy_age=16, capacity=1 << 16)

    def test_capacity_over_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="bit budget"):
            DecayParams(half_life=1, max_policy_age=15, capacity=(1 << 16) + 1)

    def test_rebase_slack_counts_against_budget(self) -> None:
        with pytest.raises(ValueError, match="bit budget"):
            DecayParams(
                half_life=1, max_policy_age=15, capacity=1 << 16, rebase_slack=1
            )

    def test_intermediate_product_over_64_bits_raises(self) -> None:
        with pytest.raises(ValueError, match="product"):
            DecayParams(
                half_life=4,
                max_policy_age=4,
                capacity=4,
                priority_bits=40,
                priority_frac_bits=16,
                table_frac_bits=31,
            )

    def test_max_shift_is_ceiling_of_age_over_half_life(self) -> None:
        assert DecayParams(half_life=4, max_policy_age=5, capacity=2).max_shift == 2
        assert DecayParams(half_life=4, max_policy_age=8, capacity=2).max_shift == 2
        assert DecayParams(half_life=4, max_policy_age=0, capacity=2).max_shift == 0

    def test_freshper_scale_configuration_fits(self) -> None:
        # tau=500 steps -> half-life 347; 50K trajectories; 15 half-lives.
        params = DecayParams(half_life=347, max_policy_age=347 * 15, capacity=50_000)

        assert params.tree_bits == 64

    @pytest.mark.parametrize(
        "overrides",
        [
            {"half_life": 0},
            {"half_life": -3},
            {"half_life": MAX_HALF_LIFE + 1},
            {"half_life": 2.0},
            {"half_life": True},
            {"max_policy_age": -1},
            {"max_policy_age": 1.5},
            {"capacity": 0},
            {"capacity": -4},
            {"priority_bits": 0},
            {"priority_frac_bits": -1},
            {"priority_frac_bits": 33},
            {"table_frac_bits": 0},
            {"rebase_slack": -1},
        ],
    )
    def test_invalid_parameters_raise(self, overrides: dict) -> None:
        kwargs = {"half_life": 4, "max_policy_age": 8, "capacity": 8, **overrides}

        with pytest.raises(ValueError):
            DecayParams(**kwargs)

    def test_params_are_immutable(self) -> None:
        params = DecayParams(half_life=4, max_policy_age=8, capacity=8)

        with pytest.raises(dataclasses.FrozenInstanceError):
            params.half_life = 5  # type: ignore[misc]

    @pytest.mark.parametrize("priority_bits", [8, 16, 24, 32, 40])
    @pytest.mark.parametrize("capacity_bits", [0, 1, 10, 16, 20, 23])
    @pytest.mark.parametrize("half_life", [1, 3, 347])
    def test_worst_case_fits_uint64_on_the_budget_boundary(
        self, priority_bits: int, capacity_bits: int, half_life: int
    ) -> None:
        # Largest shift the budget allows for this (P, L): the tightest case.
        max_shift = UINT64_BITS - priority_bits - INTRA_EPOCH_BITS - capacity_bits
        params = DecayParams(
            half_life=half_life,
            max_policy_age=max_shift * half_life,
            capacity=1 << capacity_bits,
            priority_bits=priority_bits,
            priority_frac_bits=0,
            table_frac_bits=min(31, UINT64_BITS - priority_bits - INTRA_EPOCH_BITS),
        )
        q_max = (1 << priority_bits) - 1
        base_epoch = 7
        newest = (base_epoch + max_shift) * half_life + (half_life - 1)

        leaf = inflated_priority(q_max, newest, base_epoch, params)

        assert params.tree_bits == UINT64_BITS
        assert leaf == max_leaf_value(params)
        assert leaf < UINT64_LIMIT
        assert params.capacity * leaf == max_tree_total(params)
        assert max_tree_total(params) < UINT64_LIMIT

    def test_worst_case_full_tree_every_node_below_2_64(self) -> None:
        params = DecayParams(half_life=3, max_policy_age=21 * 3, capacity=1 << 10)
        assert params.tree_bits == UINT64_BITS
        newest = (5 + params.max_shift) * 3 + 2
        leaf = inflated_priority((1 << 32) - 1, newest, 5, params)

        tree = fill_tree(params.capacity, [leaf] * params.capacity)

        assert tree.verify_invariant()
        assert all(0 <= node < UINT64_LIMIT for node in tree._tree)
        assert tree.total == max_tree_total(params)

    @given(params_and_entries())
    @settings(max_examples=200, deadline=None)
    def test_random_valid_params_never_exceed_uint64(
        self, case: tuple[DecayParams, int, list[tuple[int, int]]]
    ) -> None:
        params, _, _ = case

        assert max_leaf_value(params) < UINT64_LIMIT
        assert max_tree_total(params) < UINT64_LIMIT


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

class TestQuantizePriority:
    PARAMS = DecayParams(half_life=4, max_policy_age=8, capacity=8)

    def test_floor_of_fixed_point_value(self) -> None:
        assert quantize_priority(1.0, self.PARAMS) == 1 << 16
        assert quantize_priority(0.0, self.PARAMS) == 0
        assert quantize_priority(1.5, self.PARAMS) == 3 << 15

    @given(st.floats(min_value=0.0, max_value=65535.0, allow_nan=False))
    def test_matches_exact_rational_floor(self, x: float) -> None:
        floored = math.floor(Fraction(x) * (1 << 16))
        expected = max(1, floored) if x > 0 else 0

        assert quantize_priority(x, self.PARAMS) == expected

    @pytest.mark.parametrize("x", [2.0**-17, 1e-6, 5e-324])
    def test_positive_below_resolution_quantizes_to_one(self, x: float) -> None:
        assert quantize_priority(x, self.PARAMS) == 1

    @pytest.mark.parametrize("x", [0.0, -0.0, 0])
    def test_exact_zero_stays_zero(self, x: float) -> None:
        assert quantize_priority(x, self.PARAMS) == 0

    @given(st.floats(min_value=0.0, max_value=65535.0, allow_nan=False, exclude_min=True))
    def test_positive_priority_is_always_sampleable(self, x: float) -> None:
        q = quantize_priority(x, self.PARAMS)

        for version in range(8, 12):
            assert inflated_priority(q, version, 2, self.PARAMS) >= 1

    def test_largest_representable_value(self) -> None:
        largest = Fraction((1 << 32) - 1, 1 << 16)

        assert quantize_priority(float(largest), self.PARAMS) == (1 << 32) - 1

    @pytest.mark.parametrize("x", [65536.0, 1e300])
    def test_out_of_range_raises_instead_of_wrapping(self, x: float) -> None:
        with pytest.raises(ValueError, match="range"):
            quantize_priority(x, self.PARAMS)

    @pytest.mark.parametrize("x", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_raises(self, x: float) -> None:
        with pytest.raises(ValueError):
            quantize_priority(x, self.PARAMS)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            quantize_priority(-0.5, self.PARAMS)


# ---------------------------------------------------------------------------
# Inflated priority
# ---------------------------------------------------------------------------

class TestInflatedPriority:
    PARAMS = DecayParams(half_life=5, max_policy_age=40, capacity=16)

    @given(
        q=st.integers(0, (1 << 32) - 1),
        version=st.integers(0, 10_000),
        halvings=st.integers(1, 8),
    )
    def test_whole_half_life_ratio_is_exactly_two(
        self, q: int, version: int, halvings: int
    ) -> None:
        base_epoch = version // 5
        older = inflated_priority(q, version, base_epoch, self.PARAMS)
        newer = inflated_priority(q, version + 5 * halvings, base_epoch, self.PARAMS)

        assert newer == older << halvings
        if older:
            assert Fraction(newer, older) == 2**halvings

    def test_epoch_aligned_entry_at_base_is_the_base_priority(self) -> None:
        assert inflated_priority(12345, 35, 7, self.PARAMS) == 12345

    def test_product_is_floored(self) -> None:
        table = decay_table(5, 31)

        assert inflated_priority(3, 36, 7, self.PARAMS) == (3 * table[1]) >> 31
        assert inflated_priority(1, 39, 7, self.PARAMS) == 1

    def test_positive_priority_never_becomes_zero(self) -> None:
        for phase in range(5):
            assert inflated_priority(1, 35 + phase, 7, self.PARAMS) >= 1

    @given(params_and_entries())
    @settings(max_examples=200, deadline=None)
    def test_result_is_a_uint64_matching_the_declared_weight(
        self, case: tuple[DecayParams, int, list[tuple[int, int]]]
    ) -> None:
        params, current_version, entries = case
        base_epoch = canonical_base_epoch(current_version, params)

        for q, version in entries:
            value = inflated_priority(q, version, base_epoch, params)

            assert type(value) is int
            assert 0 <= value <= max_leaf_value(params) < UINT64_LIMIT
            assert value * Fraction(2) ** base_epoch == reference_absolute_weight(
                q, version, params
            )

    def test_entry_older_than_base_raises(self) -> None:
        with pytest.raises(ValueError, match="base"):
            inflated_priority(1, 34, 7, self.PARAMS)

    def test_entry_beyond_shift_budget_raises(self) -> None:
        beyond = (7 + self.PARAMS.max_shift + 1) * 5

        with pytest.raises(ValueError, match="shift"):
            inflated_priority(1, beyond, 7, self.PARAMS)

    @pytest.mark.parametrize("q", [-1, 1 << 32])
    def test_base_priority_out_of_range_raises(self, q: int) -> None:
        with pytest.raises(ValueError):
            inflated_priority(q, 35, 7, self.PARAMS)

    @pytest.mark.parametrize("q", [1.0, True, "1"])
    def test_non_integer_base_priority_raises(self, q: object) -> None:
        with pytest.raises(ValueError):
            inflated_priority(q, 35, 7, self.PARAMS)  # type: ignore[arg-type]

    def test_negative_version_raises(self) -> None:
        with pytest.raises(ValueError):
            inflated_priority(1, -1, 0, self.PARAMS)

    def test_negative_base_epoch_raises(self) -> None:
        with pytest.raises(ValueError):
            inflated_priority(1, 0, -1, self.PARAMS)


# ---------------------------------------------------------------------------
# Sampling equivalence against the Fraction reference
# ---------------------------------------------------------------------------

class TestSamplingMatchesReference:
    @given(case=params_and_entries(), data=st.data())
    @settings(max_examples=300, deadline=None)
    def test_tree_selects_same_leaf_as_fraction_reference(
        self,
        case: tuple[DecayParams, int, list[tuple[int, int]]],
        data: st.DataObject,
    ) -> None:
        params, current_version, entries = case
        base_epoch = canonical_base_epoch(current_version, params)
        leaves = [inflated_priority(q, v, base_epoch, params) for q, v in entries]
        assume(sum(leaves) > 0)
        tree = fill_tree(params.capacity, leaves)
        reference = [reference_absolute_weight(q, v, params) for q, v in entries]
        random_draws = data.draw(
            st.lists(st.integers(0, tree.total - 1), min_size=1, max_size=20)
        )

        for draw_int in boundary_draws(leaves) + random_draws:
            expected = reference_locate(reference, Fraction(draw_int, tree.total))

            assert tree.prefix_sum_locate(draw_int) == expected

    @given(case=params_and_entries())
    @settings(max_examples=100, deadline=None)
    def test_probabilities_equal_declared_distribution(
        self, case: tuple[DecayParams, int, list[tuple[int, int]]]
    ) -> None:
        params, current_version, entries = case
        base_epoch = canonical_base_epoch(current_version, params)
        leaves = [inflated_priority(q, v, base_epoch, params) for q, v in entries]
        assume(sum(leaves) > 0)
        reference = [reference_absolute_weight(q, v, params) for q, v in entries]
        reference_total = sum(reference, Fraction(0))

        for leaf, weight in zip(leaves, reference):
            assert Fraction(leaf, sum(leaves)) == weight / reference_total

    def test_end_to_end_from_float_priorities(self) -> None:
        params = DecayParams(half_life=347, max_policy_age=347 * 15, capacity=8)
        raw = [0.013, 1.0, 0.5, 7.25, 1e-3, 0.0, 42.0, 0.999]
        versions = [5200, 5205, 4000, 1000, 347, 2000, 5205, 346 * 3]
        current_version = 5205
        base_epoch = canonical_base_epoch(current_version, params)
        quantized = [quantize_priority(x, params) for x in raw]
        leaves = [
            inflated_priority(q, v, base_epoch, params)
            for q, v in zip(quantized, versions)
        ]
        tree = fill_tree(params.capacity, leaves)
        reference = [
            reference_absolute_weight(q, v, params)
            for q, v in zip(quantized, versions)
        ]

        for draw_int in boundary_draws(leaves):
            expected = reference_locate(reference, Fraction(draw_int, tree.total))
            assert tree.prefix_sum_locate(draw_int) == expected


# ---------------------------------------------------------------------------
# Expiry, base epoch and rebasing
# ---------------------------------------------------------------------------

class TestExpiryAndBase:
    PARAMS = DecayParams(half_life=4, max_policy_age=10, capacity=8)

    def test_expiry_boundary(self) -> None:
        assert not is_expired(90, 100, self.PARAMS)
        assert is_expired(89, 100, self.PARAMS)
        assert not is_expired(100, 100, self.PARAMS)

    def test_entry_from_the_future_raises(self) -> None:
        with pytest.raises(ValueError):
            is_expired(101, 100, self.PARAMS)

    def test_canonical_base_epoch(self) -> None:
        assert canonical_base_epoch(0, self.PARAMS) == 0
        assert canonical_base_epoch(9, self.PARAMS) == 0
        assert canonical_base_epoch(100, self.PARAMS) == 22
        assert canonical_base_epoch(102, self.PARAMS) == 23

    @given(params_and_entries())
    @settings(max_examples=200, deadline=None)
    def test_every_live_entry_is_representable_at_canonical_base(
        self, case: tuple[DecayParams, int, list[tuple[int, int]]]
    ) -> None:
        params, current_version, _ = case
        base_epoch = canonical_base_epoch(current_version, params)
        oldest_live = max(0, current_version - params.max_policy_age)

        for version in (oldest_live, current_version):
            assert not is_expired(version, current_version, params)
            shift = version // params.half_life - base_epoch
            assert 0 <= shift <= params.max_shift - params.rebase_slack

    @pytest.mark.parametrize("bad", [-1, 1.0, True])
    def test_invalid_versions_raise(self, bad: object) -> None:
        with pytest.raises(ValueError):
            canonical_base_epoch(bad, self.PARAMS)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            is_expired(bad, 100, self.PARAMS)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            pending_rebase_shift(bad, 0, self.PARAMS)  # type: ignore[arg-type]


class TestPendingRebaseShift:
    def test_no_rebase_while_shift_budget_holds(self) -> None:
        params = DecayParams(half_life=4, max_policy_age=8, capacity=8)

        assert pending_rebase_shift(11, 0, params) == 0

    def test_rebase_when_newest_epoch_exceeds_budget(self) -> None:
        params = DecayParams(half_life=4, max_policy_age=8, capacity=8)

        assert pending_rebase_shift(12, 0, params) == 1

    def test_slack_defers_and_batches_rebases(self) -> None:
        params = DecayParams(
            half_life=4, max_policy_age=8, capacity=8, rebase_slack=3
        )

        assert pending_rebase_shift(23, 0, params) == 0
        assert pending_rebase_shift(24, 0, params) == 4

    def test_rebase_period_is_slack_plus_one_epochs(self) -> None:
        params = DecayParams(
            half_life=6, max_policy_age=13, capacity=8, rebase_slack=2
        )
        base_epoch = 0
        rebase_versions = []

        for version in range(0, 6 * 40):
            shift = pending_rebase_shift(version, base_epoch, params)
            if shift:
                rebase_versions.append(version)
                base_epoch += shift
            assert version // 6 - base_epoch <= params.max_shift

        gaps = {b - a for a, b in zip(rebase_versions, rebase_versions[1:])}
        assert gaps == {6 * (params.rebase_slack + 1)}

    def test_base_ahead_of_current_version_raises(self) -> None:
        params = DecayParams(half_life=4, max_policy_age=8, capacity=8)

        with pytest.raises(ValueError):
            pending_rebase_shift(3, 1, params)


class TestRebase:
    def test_rebase_is_exact_right_shift(self) -> None:
        assert rebase_priority(40, 3) == 5
        assert rebase_priority(0, 9) == 0
        assert rebase_priority(7, 0) == 7

    def test_rebase_that_would_lose_bits_raises(self) -> None:
        with pytest.raises(ValueError, match="lose"):
            rebase_priority(41, 3)

    @pytest.mark.parametrize("value,shift", [(-1, 1), (1 << 64, 1), (8, -1), (8, 64)])
    def test_invalid_arguments_raise(self, value: int, shift: int) -> None:
        with pytest.raises(ValueError):
            rebase_priority(value, shift)

    def test_inputs_are_not_mutated(self) -> None:
        original = [8, 16, 0, 24]
        snapshot = list(original)

        result = rebase_priorities(original, 3)

        assert original == snapshot
        assert result == (1, 2, 0, 3)
        assert isinstance(result, tuple)

    def test_batch_rebase_is_all_or_nothing(self) -> None:
        with pytest.raises(ValueError, match="lose"):
            rebase_priorities([8, 16, 3], 3)

    @given(case=rebase_case())
    @settings(max_examples=300, deadline=None)
    def test_rebase_loses_no_bits_and_changes_no_sampling_decision(
        self,
        case: tuple[DecayParams, int, list[tuple[int, int]], int],
    ) -> None:
        params, current_version, entries, advance = case
        old_base = canonical_base_epoch(current_version, params)
        later_version = current_version + advance
        new_base = canonical_base_epoch(later_version, params)
        shift = new_base - old_base
        # Staleness-bounded eviction happens BEFORE the rebase.
        survivors = [
            (q, v) for q, v in entries if not is_expired(v, later_version, params)
        ]
        old_leaves = [inflated_priority(q, v, old_base, params) for q, v in survivors]
        # rebase_case() guarantees a live survivor with positive weight. If a
        # future edit to the strategy breaks that, fail here with a clear
        # message rather than inside the tree code below.
        assert survivors and sum(old_leaves) > 0

        new_leaves = rebase_priorities(old_leaves, shift)

        # No bits lost: the shift is invertible and equals a fresh derivation.
        assert [leaf << shift for leaf in new_leaves] == old_leaves
        assert list(new_leaves) == [
            inflated_priority(q, v, new_base, params) for q, v in survivors
        ]
        # Every node of the tree (not only leaves) is the shifted old node.
        old_tree = fill_tree(params.capacity, old_leaves)
        new_tree = fill_tree(params.capacity, list(new_leaves))
        assert new_tree._tree == [node >> shift for node in old_tree._tree]
        assert [node << shift for node in new_tree._tree] == old_tree._tree
        # No sampling decision changes.
        for draw_int in boundary_draws(old_leaves):
            assert old_tree.prefix_sum_locate(draw_int) == new_tree.prefix_sum_locate(
                draw_int >> shift
            )
        for leaf_old, leaf_new in zip(old_leaves, new_leaves):
            assert Fraction(leaf_old, old_tree.total) == Fraction(
                leaf_new, new_tree.total
            )


# ---------------------------------------------------------------------------
# Age on update
# ---------------------------------------------------------------------------

class TestVersionAfterUpdate:
    def test_default_keeps_original_version(self) -> None:
        assert version_after_update(10, 50) == 10

    def test_reset_uses_current_version(self) -> None:
        assert version_after_update(10, 50, reset_age_on_update=True) == 50

    def test_entry_newer_than_current_raises(self) -> None:
        with pytest.raises(ValueError):
            version_after_update(51, 50)

    @pytest.mark.parametrize("bad", [-1, 2.5, True])
    def test_invalid_versions_raise(self, bad: object) -> None:
        with pytest.raises(ValueError):
            version_after_update(bad, 50)  # type: ignore[arg-type]


def test_module_exposes_named_bit_width_constants() -> None:
    assert decay.UINT64_BITS == 64
    assert decay.DEFAULT_PRIORITY_BITS == 32
    assert decay.DEFAULT_PRIORITY_FRAC_BITS == 16
    assert decay.DEFAULT_TABLE_FRAC_BITS == 31
    assert decay.INTRA_EPOCH_BITS == 1
    assert DEFAULT_PRIORITY_BITS + DEFAULT_TABLE_FRAC_BITS + INTRA_EPOCH_BITS <= 64
