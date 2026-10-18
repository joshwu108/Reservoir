"""Seed × item bootstrap, reward AUC, steps-to-target and matched-rollout summaries (``benchmarks/staleness/stats.py``)."""

from __future__ import annotations

import pytest

from benchmarks.staleness import math500, stats

# --- bootstrap ---------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    items = {42: [1, 0, 1, 1, 0, 1, 0, 1] * 10, 43: [1, 1, 0, 1, 0, 0, 1, 1] * 10}
    ci = stats.bootstrap_ci(items, n_boot=500)
    assert ci["mean"] == pytest.approx(sum(map(sum, items.values())) / 160)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert ci == stats.bootstrap_ci(items, n_boot=500)
    assert (ci["n_seeds"], ci["n_items"], ci["n_boot"]) == (2, 80, 500)


def test_bootstrap_ci_of_a_constant_is_a_point():
    ci = stats.bootstrap_ci({1: [1.0] * 5, 2: [1.0] * 5}, n_boot=50)
    assert ci["lo"] == ci["hi"] == ci["mean"] == 1.0


def test_bootstrap_refuses_ragged_items():
    with pytest.raises(ValueError, match="items"):
        stats.bootstrap_ci({1: [1, 0], 2: [1]}, n_boot=10)


def test_paired_difference_uses_common_seeds_and_the_same_items():
    base = {42: [0, 0, 1, 1], 43: [0, 1, 0, 1], 44: [1, 1, 1, 1]}
    arm = {42: [1, 0, 1, 1], 43: [1, 1, 0, 1]}
    diff = stats.paired_bootstrap_diff(arm, base, n_boot=400)
    assert diff["seeds"] == [42, 43]
    assert diff["mean"] == pytest.approx(0.25)
    assert diff["lo"] <= 0.25 <= diff["hi"]
    assert stats.paired_bootstrap_diff({7: [1]}, base, n_boot=10) is None


# --- curves ---------------------------------------------------------------------------


def test_reward_auc_is_the_mean_reward_over_logged_steps():
    assert stats.reward_auc([0.0, 0.5, 1.0, None]) == pytest.approx(0.5)
    assert stats.reward_auc([]) is None


def test_steps_to_target_on_the_rolling_curve():
    steps = [1, 2, 3, 4, 5]
    rewards = [0.0, 0.2, 0.9, 0.9, 0.9]
    assert stats.steps_to_target(steps, rewards, target=0.6, window=1) == 3
    assert stats.steps_to_target(steps, rewards, target=0.6, window=2) == 4     # rolling: 0, .1, .55, .9
    assert stats.steps_to_target(steps, rewards, target=2.0, window=1) is None


def test_matched_rollouts_cuts_each_run_at_the_budget():
    series = {"step": [1, 2, 3, 4], "reward": [0.0, 0.4, 0.8, 1.0], "rollouts": [64, 64, 64, 64]}
    evals = [{"step": 2, "accuracy": 0.3}, {"step": 4, "accuracy": 0.5}]
    cut = stats.at_rollout_budget(series, evals, budget=130)
    assert cut["step"] == 2 and cut["rollouts"] == 128
    assert cut["reward_auc"] == pytest.approx(0.2)
    assert cut["avg_at_k"] == 0.3
    whole = stats.at_rollout_budget({**series, "rollouts": [32] * 4}, evals, budget=128)
    assert whole["step"] == 4 and whole["avg_at_k"] == 0.5
    assert stats.at_rollout_budget({**series, "rollouts": [None] * 4}, evals, budget=128) is None


# --- MATH500 grading ----------------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    (r"so the answer is \boxed{\frac{1}{2}}.", r"\frac{1}{2}"),
    (r"\boxed{1} then \boxed{\left( 3, \frac{\pi}{2} \right)}", r"\left( 3, \frac{\pi}{2} \right)"),
    ("#### 42", "42"),
    ("the result is 17", "17"),
    ("nothing", None),
])
def test_math500_extracts_the_last_boxed_answer(text, expected):
    assert math500.extract_answer(text) == expected


@pytest.mark.parametrize("predicted, gold", [
    (r"\dfrac{1}{2}", r"\frac{1}{2}"),
    ("0.5", r"\frac12"),
    (r"\left( 3, \frac{\pi}{2} \right)", r"\left( 3, \frac{\pi}{2} \right)"),
    (r"(3,\frac{\pi}{2})", r"\left( 3, \frac{\pi}{2} \right)"),
    (r"10\%", "10"),
    (r"90^\circ", "90"),
    (r"\text{(C)}", "(C)"),
    ("x = 5", "5"),
    ("1,000", "1000"),
])
def test_math500_equivalent_answers_match(predicted, gold):
    assert math500.answers_match(predicted, gold)


@pytest.mark.parametrize("predicted, gold", [("3", "4"), (r"\frac{1}{3}", "0.33"), (None, "1"), ("", "1")])
def test_math500_different_answers_do_not_match(predicted, gold):
    assert not math500.answers_match(predicted, gold)


def test_math500_reward_scores_completions():
    assert math500.math_reward([r"\boxed{4}", r"\boxed{5}"], answer=["4", "4"]) == [1.0, 0.0]
