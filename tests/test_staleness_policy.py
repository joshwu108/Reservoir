"""Tests for ``reservoir.integrations._trl_staleness``: the policy that admits, declines or reweights replayed rows.

The policy is a pure function of per-row inputs the log records (sequence
log-ratio, exact importance weight, age, training row and group). Its four
stages run in a fixed order: age bound, exact ESS floor, per-group
importance-mass cap, legacy ``max_log_ratio``. The checker replays the same
function from the record (``reservoir_checker.staleness``), so the two
implementations are cross-checked here on random inputs.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
from hypothesis import given, settings, strategies as st

from reservoir.integrations._trl_staleness import (
    DECLINE_REASONS,
    EXP_DIGITS,
    LOG_RATIO_CLAMP,
    PRESETS,
    RowDecision,
    StalenessPolicy,
    decide,
    exact_exp,
)
from reservoir_checker import staleness as checker_staleness

W = [Fraction(1), Fraction(1, 2), Fraction(1, 4), Fraction(1, 8)]


def run(policy, ratios, weights=None, ages=None, rows=None, groups=None, group_size=2):
    n = len(ratios)
    weights = weights if weights is not None else [Fraction(1)] * n
    ages = ages if ages is not None else [0] * n
    rows = rows if rows is not None else list(range(n))
    groups = groups if groups is not None else [r // group_size for r in rows]
    return decide(policy, ratios=ratios, is_weights=weights, ages=ages, rows=rows, groups=groups)


# ---------------------------------------------------------------------------
# Construction, presets, record form
# ---------------------------------------------------------------------------

class TestPolicy:
    def test_off_is_inactive_and_declines_nothing(self):
        off = StalenessPolicy()
        assert not off.active and off == StalenessPolicy.preset("off") == PRESETS["off"]
        decisions = run(off, [float("nan"), 50.0], ages=[10, 10])
        assert decisions == ()

    def test_presets(self):
        c = StalenessPolicy.preset("conservative")
        assert (c.max_age, c.ess_floor, c.mass_cap, c.max_log_ratio) == (16, 0.5, 0.25, None) and c.active
        a = StalenessPolicy.preset("async")
        assert (a.max_age, a.ess_floor, a.mass_cap, a.max_log_ratio) == (64, 0.3, 0.5, None)
        with pytest.raises(ValueError, match="preset"):
            StalenessPolicy.preset("aggressive")

    @pytest.mark.parametrize("kwargs", [
        dict(max_age=-1), dict(max_age=True), dict(max_age=2.0),
        dict(ess_floor=0.0), dict(ess_floor=1.5), dict(ess_floor=float("nan")),
        dict(mass_cap=-0.1), dict(mass_cap=float("inf")),
        dict(max_log_ratio=0.0), dict(max_log_ratio=-1.0),
        dict(max_declines_per_step=-1), dict(max_declines_per_step=1.5),
    ])
    def test_bad_parameters_raise(self, kwargs):
        with pytest.raises(ValueError):
            StalenessPolicy(**kwargs)

    def test_legacy_gate_alone_is_an_active_policy(self):
        p = StalenessPolicy(max_log_ratio=1.0, max_declines_per_step=3)
        assert p.active and p.legacy_only

    def test_record_round_trip_uses_hex_floats(self):
        p = StalenessPolicy(max_age=5, ess_floor=0.3, mass_cap=0.5, max_log_ratio=2.5, max_declines_per_step=4)
        record = p.to_record(group_size=8)
        assert record == {
            "max_age": 5, "ess_floor": (0.3).hex(), "mass_cap": (0.5).hex(), "max_log_ratio": (2.5).hex(),
            "max_declines_per_step": 4, "group_size": 8,
        }
        assert StalenessPolicy.from_record(record) == p
        assert StalenessPolicy().to_record(group_size=None)["ess_floor"] is None


# ---------------------------------------------------------------------------
# exact_exp: the one transcendental, made deterministic
# ---------------------------------------------------------------------------

class TestExactExp:
    def test_matches_float_exp_to_many_digits(self):
        for r in (0.0, 0.1, -3.7, 12.5, -40.0):
            assert abs(float(exact_exp(r)) - math.exp(r)) <= 1e-13 * math.exp(r)
        assert exact_exp(0.0) == 1

    def test_is_a_fraction_with_bounded_size(self):
        assert isinstance(exact_exp(700.0), Fraction)
        assert exact_exp(1e9) == exact_exp(LOG_RATIO_CLAMP) and exact_exp(-1e9) == exact_exp(-LOG_RATIO_CLAMP)
        assert len(str(exact_exp(LOG_RATIO_CLAMP).numerator)) < 320

    def test_checker_mirror_agrees_bit_for_bit(self):
        assert checker_staleness.EXP_DIGITS == EXP_DIGITS and checker_staleness.LOG_RATIO_CLAMP == LOG_RATIO_CLAMP
        for r in (0.0, 1e-300, 0.3, -2.25, 123.456, 699.9, -699.9):
            assert checker_staleness.exact_exp(r) == exact_exp(r)

    def test_non_finite_is_a_value_error(self):
        with pytest.raises(ValueError):
            exact_exp(float("nan"))


# ---------------------------------------------------------------------------
# The four stages, in order
# ---------------------------------------------------------------------------

class TestStages:
    def test_non_finite_ratio_is_declined_first_as_drift(self):
        d = run(StalenessPolicy(max_age=100), [float("nan"), float("inf"), 0.5])
        assert [x.reason for x in d] == ["drift", "drift", None]
        assert all(x.scale == 1 for x in d)

    def test_age_bound_is_strict(self):
        d = run(StalenessPolicy(max_age=3), [0.0, 0.0, 0.0], ages=[3, 4, 0])
        assert [x.reason for x in d] == [None, "age", None]

    def test_ess_floor_declines_largest_ratio_first_ties_by_index(self):
        # Weights 1, 1/2, 1/4, 1/8: ESS/n = (15/8)^2 / (1+1/4+1/16+1/64) / 4 = 0.66; a floor of 0.9 needs
        # declines. The first victim is the largest |r| (draw 2, |r|=2.0), then the tie at |r|=1.0 goes to
        # the lower draw index (0 before 3).
        d = run(StalenessPolicy(ess_floor=0.9), [1.0, 0.0, -2.0, 1.0], weights=W)
        reasons = [x.reason for x in d]
        assert reasons[2] == "ess" and reasons[0] == "ess"
        kept = [k for k, x in enumerate(d) if x.reason is None]
        ws = [W[k] for k in kept]
        ess = sum(ws) ** 2 / sum(w * w for w in ws)
        assert ess >= Fraction(9, 10) * len(kept)

    def test_ess_floor_stops_at_one_row(self):
        d = run(StalenessPolicy(ess_floor=1.0), [0.0, 0.0], weights=[Fraction(1), Fraction(1, 2)])
        assert [x.reason for x in d] == ["ess", None]       # tie on |r| -> draw 0 goes first

    def test_ess_floor_satisfied_declines_nothing(self):
        d = run(StalenessPolicy(ess_floor=0.5), [5.0, -5.0], weights=[Fraction(1), Fraction(1)])
        assert all(x.reason is None for x in d)

    def test_ess_ignores_rows_declined_by_age(self):
        d = run(StalenessPolicy(max_age=1, ess_floor=1.0), [0.0, 9.0, 0.0],
                weights=[Fraction(1), Fraction(1, 100), Fraction(1)], ages=[0, 5, 0])
        assert [x.reason for x in d] == [None, "age", None]

    def test_mass_cap_rescales_a_group_by_an_exact_fraction(self):
        # Two groups of two rows. Group 0 has ratios 0 and ln(8): mass = 1*1 + 1*8 = 9 > (1+0.25)*2 = 2.5.
        r = math.log(8.0)
        d = run(StalenessPolicy(mass_cap=0.25), [0.0, r, 0.0, 0.0])
        mass = exact_exp(0.0) + exact_exp(r)
        expected = Fraction(5, 2) / mass
        assert d[0].scale == d[1].scale == expected and d[2].scale == d[3].scale == 1
        assert all(x.reason is None for x in d)
        assert d[0].group == d[1].group == 0 and d[2].group == 1

    def test_mass_cap_weights_rows_by_their_importance_weight(self):
        r = math.log(8.0)
        d = run(StalenessPolicy(mass_cap=0.0), [r, 0.0], weights=[Fraction(1, 8), Fraction(1)], group_size=2)
        # mass = 1/8 * 8 + 1 * 1 = 2, cap = (1+0) * 2 = 2: not exceeded.
        assert d[0].scale == 1 and d[1].scale == 1

    def test_mass_cap_counts_only_rows_still_kept(self):
        r = math.log(8.0)
        d = run(StalenessPolicy(max_age=0, mass_cap=0.25), [r, 0.0, 0.0], ages=[1, 0, 0], group_size=3)
        assert d[0].reason == "age" and d[1].scale == 1 and d[2].scale == 1   # mass 2 <= 1.25 * 2

    def test_legacy_gate_runs_last(self):
        # The mass cap sees the drifted row (ratios 0 and ln 8); the legacy gate then declines it.
        r = math.log(8.0)
        d = run(StalenessPolicy(mass_cap=0.25, max_log_ratio=1.0), [0.0, r])
        assert d[1].reason == "drift" and d[1].scale == 1
        assert d[0].reason is None and d[0].scale == Fraction(5, 2) / (1 + exact_exp(r))

    def test_legacy_gate_alone_matches_the_old_rule(self):
        d = run(StalenessPolicy(max_log_ratio=1.0), [0.5, -1.5, 1.0])
        assert [x.reason for x in d] == [None, "drift", None]

    def test_too_many_declines_raise(self):
        with pytest.raises(RuntimeError, match="would decline 2"):
            run(StalenessPolicy(max_log_ratio=0.1, max_declines_per_step=1), [1.0, 1.0])
        assert len(run(StalenessPolicy(max_log_ratio=0.1, max_declines_per_step=2), [1.0, 1.0])) == 2

    def test_decisions_carry_draw_row_group(self):
        d = run(StalenessPolicy(max_age=1), [0.0, 0.0], rows=[6, 7], groups=[3, 3])
        assert d == (RowDecision(0, 6, 3, None, Fraction(1)), RowDecision(1, 7, 3, None, Fraction(1)))

    def test_input_validation(self):
        p = StalenessPolicy(max_age=1)
        with pytest.raises(ValueError, match="length"):
            decide(p, ratios=[0.0], is_weights=[Fraction(1)], ages=[0, 0], rows=[0], groups=[0])
        with pytest.raises(ValueError, match="distinct"):
            decide(p, ratios=[0.0, 0.0], is_weights=[Fraction(1)] * 2, ages=[0, 0], rows=[1, 1], groups=[0, 0])
        with pytest.raises(ValueError, match="age"):
            decide(p, ratios=[0.0], is_weights=[Fraction(1)], ages=[-1], rows=[0], groups=[0])
        assert set(DECLINE_REASONS) == {"drift", "age", "ess"}


# ---------------------------------------------------------------------------
# Adapter and checker implementations agree
# ---------------------------------------------------------------------------

finite = st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False)
ratio = st.one_of(finite, st.just(float("nan")), st.just(float("inf")), st.just(-0.0))
weight = st.fractions(min_value=Fraction(1, 1000), max_value=Fraction(1), max_denominator=4096)


@settings(max_examples=300, deadline=None)
@given(
    rows=st.lists(st.tuples(ratio, weight, st.integers(0, 30)), min_size=1, max_size=8),
    max_age=st.one_of(st.none(), st.integers(0, 30)),
    ess_floor=st.one_of(st.none(), st.floats(0.05, 1.0)),
    mass_cap=st.one_of(st.none(), st.floats(0.0, 2.0)),
    max_log_ratio=st.one_of(st.none(), st.floats(0.1, 40.0)),
    group_size=st.integers(1, 4),
)
def test_checker_replays_every_decision(rows, max_age, ess_floor, mass_cap, max_log_ratio, group_size):
    policy = StalenessPolicy(max_age=max_age, ess_floor=ess_floor, mass_cap=mass_cap, max_log_ratio=max_log_ratio)
    ratios = [r for r, _, _ in rows]
    weights = [w for _, w, _ in rows]
    ages = [a for _, _, a in rows]
    positions = list(range(len(rows)))
    groups = [p // group_size for p in positions]
    ours = decide(policy, ratios=ratios, is_weights=weights, ages=ages, rows=positions, groups=groups)
    if not policy.active:
        assert ours == ()
        with pytest.raises(checker_staleness.CheckerError, match="no stage"):
            checker_staleness.parse_policy(policy.to_record(group_size=group_size), "test")
        return
    theirs = checker_staleness.replay_decisions(
        checker_staleness.parse_policy(policy.to_record(group_size=group_size), "test"),
        ratios=ratios, is_weights=weights, ages=ages, rows=positions, groups=groups,
    )
    assert [(d.draw, d.row, d.group, d.reason, d.scale) for d in ours] == list(theirs)
