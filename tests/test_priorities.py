"""Tests for reservoir.priorities — pluggable priority strategies.

A strategy maps a rollout (or a whole prompt group) to a non-negative
finite score. The buffer turns that score into an exact integer priority;
the strategy itself is the only place floating-point judgement lives.
The tests cover the three built-ins' formulas, the validation wrapper
that guards the buffer against bad scores, and the two base classes a
user subclasses.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from hypothesis import given, strategies as st

from reservoir.priorities import (
    AdvantagePriority,
    PassRateTargeting,
    PassRateVariance,
    PriorityStrategy,
    PromptPriority,
    validated_prompt_score,
    validated_score,
)
from reservoir.rollout import MAX_ABS_REWARD, Rollout, RolloutGroup


def make_group(rewards: list[float], version: int = 0) -> RolloutGroup:
    return RolloutGroup(
        prompt_id="p",
        model_version=version,
        rollouts=[Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in rewards],
    )


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class TestBaseClasses:
    def test_priority_strategy_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            PriorityStrategy()  # type: ignore[abstract]

    def test_prompt_priority_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            PromptPriority()  # type: ignore[abstract]

    def test_readme_custom_strategy_works(self) -> None:
        class RewardGap(PriorityStrategy):
            def score(self, rollout, group) -> float:
                return abs(rollout.reward - group.mean_reward)

        g = make_group([1.0, 0.0])
        assert RewardGap().score(g.rollouts[0], g) == pytest.approx(0.5)
        assert validated_score(RewardGap(), g.rollouts[0], g) == pytest.approx(0.5)

    def test_custom_prompt_priority_works(self) -> None:
        class GroupSize(PromptPriority):
            def score_prompt(self, group) -> float:
                return float(group.size)

        g = make_group([1.0, 0.0, 0.5])
        assert validated_prompt_score(GroupSize(), g) == 3.0

    @pytest.mark.parametrize("cls", [PassRateTargeting, PassRateVariance])
    def test_pass_rate_strategies_implement_both_interfaces(self, cls: type) -> None:
        s = cls()
        assert isinstance(s, PriorityStrategy)
        assert isinstance(s, PromptPriority)

    def test_advantage_priority_is_rollout_level_only(self) -> None:
        s = AdvantagePriority()
        assert isinstance(s, PriorityStrategy)
        assert not isinstance(s, PromptPriority)


# ---------------------------------------------------------------------------
# validated_score — the boundary the buffer relies on
# ---------------------------------------------------------------------------

class _Returns(PriorityStrategy):
    def __init__(self, value: object) -> None:
        self.value = value

    def score(self, rollout, group) -> float:
        return self.value  # type: ignore[return-value]


class _PromptReturns(PromptPriority):
    def __init__(self, value: object) -> None:
        self.value = value

    def score_prompt(self, group) -> float:
        return self.value  # type: ignore[return-value]


class TestValidatedScore:
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -1.0, -1e-300])
    def test_rejects_non_finite_or_negative(self, bad: float) -> None:
        g = make_group([1.0])
        with pytest.raises(ValueError, match="_Returns"):
            validated_score(_Returns(bad), g.rollouts[0], g)

    @pytest.mark.parametrize("bad", [None, "1", True, [1.0]])
    def test_rejects_non_numeric(self, bad: object) -> None:
        g = make_group([1.0])
        with pytest.raises(ValueError, match="_Returns"):
            validated_score(_Returns(bad), g.rollouts[0], g)

    def test_zero_is_allowed(self) -> None:
        g = make_group([1.0])
        assert validated_score(_Returns(0.0), g.rollouts[0], g) == 0.0

    def test_int_score_becomes_python_float(self) -> None:
        g = make_group([1.0])
        out = validated_score(_Returns(3), g.rollouts[0], g)
        assert out == 3.0 and type(out) is float

    def test_numpy_scalar_score_becomes_python_float(self) -> None:
        np = pytest.importorskip("numpy")
        g = make_group([1.0])
        out = validated_score(_Returns(np.float32(2.5)), g.rollouts[0], g)
        assert out == 2.5 and type(out) is float

    def test_huge_int_score_is_a_value_error_naming_the_strategy(self) -> None:
        g = make_group([1.0])
        with pytest.raises(ValueError, match="_Returns"):
            validated_score(_Returns(10**400), g.rollouts[0], g)

    def test_rejects_wrong_rollout_or_group_type(self) -> None:
        g = make_group([1.0])
        with pytest.raises(TypeError, match="rollout"):
            validated_score(AdvantagePriority(), "not a rollout", g)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="group"):
            validated_score(AdvantagePriority(), g.rollouts[0], None)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="group"):
            validated_prompt_score(PassRateVariance(), g.rollouts[0])  # type: ignore[arg-type]

    def test_prompt_variant_has_the_same_checks(self) -> None:
        g = make_group([1.0])
        with pytest.raises(ValueError, match="_PromptReturns"):
            validated_prompt_score(_PromptReturns(-0.5), g)
        assert validated_prompt_score(_PromptReturns(2), g) == 2.0

    def test_rejects_wrong_strategy_type(self) -> None:
        g = make_group([1.0])
        with pytest.raises(TypeError, match="PriorityStrategy"):
            validated_score(object(), g.rollouts[0], g)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="PromptPriority"):
            validated_prompt_score(AdvantagePriority(), g)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Epsilon handling shared by every built-in
# ---------------------------------------------------------------------------

class TestEpsilon:
    @pytest.mark.parametrize("cls", [AdvantagePriority, PassRateTargeting, PassRateVariance])
    @pytest.mark.parametrize("bad", [-1e-9, math.nan, math.inf, "0", None, True, 10**400])
    def test_bad_epsilon_rejected(self, cls: type, bad: object) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            cls(epsilon=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("cls", [AdvantagePriority, PassRateTargeting, PassRateVariance])
    def test_zero_epsilon_allowed(self, cls: type) -> None:
        assert cls(epsilon=0.0).epsilon == 0.0

    @pytest.mark.parametrize("cls", [AdvantagePriority, PassRateTargeting, PassRateVariance])
    def test_default_epsilon_is_small_and_positive(self, cls: type) -> None:
        assert 0 < cls().epsilon <= 1e-3

    @pytest.mark.parametrize("cls", [AdvantagePriority, PassRateTargeting, PassRateVariance])
    def test_parameters_are_immutable_after_construction(self, cls: type) -> None:
        s = cls()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.epsilon = 0.5  # type: ignore[misc]

    @pytest.mark.parametrize("cls", [AdvantagePriority, PassRateTargeting, PassRateVariance])
    def test_equal_parameters_compare_equal(self, cls: type) -> None:
        assert cls() == cls()
        assert cls(epsilon=1e-3) != cls()


# ---------------------------------------------------------------------------
# AdvantagePriority
# ---------------------------------------------------------------------------

class TestAdvantagePriority:
    def test_abs_advantage_plus_epsilon(self) -> None:
        g = make_group([1.0, 0.0, 0.5])
        s = AdvantagePriority(epsilon=1e-6)
        assert s.score(g.rollouts[0], g) == pytest.approx(0.5 + 1e-6)
        assert s.score(g.rollouts[1], g) == pytest.approx(0.5 + 1e-6)
        assert s.score(g.rollouts[2], g) == pytest.approx(1e-6)

    def test_identical_rewards_give_epsilon_for_all(self) -> None:
        g = make_group([1.0, 1.0, 1.0])
        s = AdvantagePriority(epsilon=1e-6)
        assert all(s.score(r, g) == pytest.approx(1e-6) for r in g.rollouts)

    def test_stays_finite_at_the_reward_magnitude_limit(self) -> None:
        # MAX_ABS_REWARD exists so this cannot overflow; check the strategy honours it.
        g = make_group([MAX_ABS_REWARD, -MAX_ABS_REWARD])
        for r in g.rollouts:
            score = validated_score(AdvantagePriority(), r, g)
            assert math.isfinite(score) and score == MAX_ABS_REWARD + 1e-6

    def test_uses_the_group_passed_in_not_a_cached_one(self) -> None:
        # The score depends on the group's mean, not on anything cached in the rollout.
        g_low = make_group([1.0, 0.0])
        g_high = make_group([1.0, 1.0, 1.0, 0.0])
        r = g_low.rollouts[0]
        s = AdvantagePriority(epsilon=0.0)
        assert s.score(r, g_low) == pytest.approx(0.5)
        assert s.score(r, g_high) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# PassRateTargeting
# ---------------------------------------------------------------------------

class TestPassRateTargeting:
    def test_peak_at_target_is_one_plus_epsilon(self) -> None:
        g = make_group([1.0, 0.0])  # pass rate 0.5
        s = PassRateTargeting(target=0.5, width=0.15, epsilon=1e-6)
        assert s.score_prompt(g) == pytest.approx(1.0 + 1e-6)
        assert s.score(g.rollouts[0], g) == pytest.approx(1.0 + 1e-6)

    def test_gaussian_falloff(self) -> None:
        g = make_group([1.0, 1.0, 1.0, 0.0])  # pass rate 0.75
        s = PassRateTargeting(target=0.5, width=0.25, epsilon=0.0)
        # One width away from target: exp(-1/2).
        assert s.score_prompt(g) == pytest.approx(math.exp(-0.5))

    def test_same_score_for_every_rollout_in_group(self) -> None:
        g = make_group([1.0, 0.0, 0.0])
        s = PassRateTargeting()
        scores = {s.score(r, g) for r in g.rollouts}
        assert len(scores) == 1
        assert scores.pop() == s.score_prompt(g)

    def test_all_pass_and_all_fail_are_far_from_default_target(self) -> None:
        s = PassRateTargeting(epsilon=0.0)
        mid = s.score_prompt(make_group([1.0, 0.0]))
        assert s.score_prompt(make_group([1.0, 1.0])) < mid / 100
        assert s.score_prompt(make_group([0.0, 0.0])) < mid / 100

    @pytest.mark.parametrize("bad", [-0.1, 1.1, math.nan, "0.5", None, True, 10**400])
    def test_bad_target_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="target"):
            PassRateTargeting(target=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0.0, -0.1, math.nan, math.inf, "0.1", None, True, 10**400])
    def test_bad_width_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="width"):
            PassRateTargeting(width=bad)  # type: ignore[arg-type]

    def test_boundary_targets_allowed(self) -> None:
        assert PassRateTargeting(target=0.0).target == 0.0
        assert PassRateTargeting(target=1.0).target == 1.0

    def test_custom_success_predicate_is_respected(self) -> None:
        g = RolloutGroup(
            prompt_id="p", model_version=0,
            rollouts=[Rollout([1], [-0.1], 0.9), Rollout([1], [-0.1], 0.4)],
            is_success=lambda r: r.reward >= 0.5,
        )
        assert g.pass_rate == 0.5
        assert PassRateTargeting(target=0.5, epsilon=0.0).score_prompt(g) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PassRateVariance
# ---------------------------------------------------------------------------

class TestPassRateVariance:
    def test_formula(self) -> None:
        g = make_group([1.0, 1.0, 1.0, 0.0])  # 0.75
        assert PassRateVariance(epsilon=1e-6).score_prompt(g) == pytest.approx(0.75 * 0.25 + 1e-6)

    def test_all_pass_or_all_fail_score_epsilon(self) -> None:
        s = PassRateVariance(epsilon=1e-6)
        assert s.score_prompt(make_group([1.0, 1.0])) == pytest.approx(1e-6)
        assert s.score_prompt(make_group([0.0, 0.0])) == pytest.approx(1e-6)

    def test_maximum_at_half(self) -> None:
        s = PassRateVariance(epsilon=0.0)
        assert s.score_prompt(make_group([1.0, 0.0])) == pytest.approx(0.25)

    def test_same_score_for_every_rollout_in_group(self) -> None:
        g = make_group([1.0, 0.0, 0.0, 1.0])
        s = PassRateVariance()
        assert {s.score(r, g) for r in g.rollouts} == {s.score_prompt(g)}


# ---------------------------------------------------------------------------
# Property: every built-in is a valid strategy for every valid group
# ---------------------------------------------------------------------------

group_rewards = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=16,
)


@given(rewards=group_rewards, epsilon=st.floats(0.0, 1e-3, allow_nan=False))
def test_builtins_return_finite_scores_at_least_epsilon(
    rewards: list[float], epsilon: float
) -> None:
    g = make_group(rewards)
    for s in (
        AdvantagePriority(epsilon=epsilon),
        PassRateTargeting(epsilon=epsilon),
        PassRateVariance(epsilon=epsilon),
    ):
        for r in g.rollouts:
            score = validated_score(s, r, g)
            assert math.isfinite(score)
            assert score >= epsilon
    for s in (PassRateTargeting(epsilon=epsilon), PassRateVariance(epsilon=epsilon)):
        score = validated_prompt_score(s, g)
        assert math.isfinite(score)
        assert score >= epsilon
        assert score <= 1.0 + epsilon
