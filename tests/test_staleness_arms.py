"""The sweep's arm grid: control arms, the naive replay arm and the single-seed stress arm."""

from __future__ import annotations

import pytest

from benchmarks.staleness import arms
from benchmarks.staleness.arms import ArmSpec, arm_specs, plan_jobs

MAIN = ["grpo", "grpo_filter", "wreinforce", "reservoir_age8_off", "reservoir_age32_off",
        "reservoir_age8_on", "reservoir_age32_on", "reservoir_naive_age32"]


def test_main_grid_then_one_stress_arm():
    specs = arm_specs(policy_on="staleness")
    assert [s.name for s in specs] == [*MAIN, "reservoir_age128_on"]
    assert [s.name for s in specs if s.stress] == ["reservoir_age128_on"]


def test_control_arms_have_no_buffer_and_their_own_trainer():
    specs = {s.name: s for s in arm_specs()}
    assert (specs["grpo"].trainer, specs["grpo_filter"].trainer, specs["wreinforce"].trainer) == (
        "grpo", "grpo_filter", "wreinforce")
    for name in ("grpo", "grpo_filter", "wreinforce"):
        assert not specs[name].replay and specs[name].max_policy_age is None and specs[name].policy == "off"
    assert all(s.trainer == "reservoir" for s in specs.values() if s.replay)


def test_naive_arm_is_uniform_priority_without_age_decay_and_policy_off():
    from reservoir.decay import MAX_HALF_LIFE

    naive = arms.arm_by_name(arm_specs(policy_on="staleness"), "reservoir_naive_age32")
    assert naive.replay and naive.priority == "uniform" and naive.policy == "off" and naive.max_policy_age == 32
    assert arms.half_life_for_spec(naive) == MAX_HALF_LIFE
    assert arms.half_life_for_spec(arms.arm_by_name(arm_specs(), "reservoir_age32_off")) == 16


def test_on_arms_carry_the_requested_policy():
    on = [s for s in arm_specs(policy_on="gate", max_log_ratio=2.0) if s.name.endswith("_on")]
    assert [s.max_policy_age for s in on] == [8, 32, 128]
    assert all(s.policy == "gate" and s.max_log_ratio == 2.0 for s in on)
    on = [s for s in arm_specs(policy_on="staleness") if s.name.endswith("_on")]
    assert all(s.policy == "staleness" and s.max_log_ratio is None for s in on)


def test_uniform_priority_scores_every_rollout_the_same_and_fingerprints():
    from reservoir.rollout import Rollout, RolloutGroup
    from reservoir.rollout_snapshot import _strategy_fingerprint

    group = RolloutGroup(prompt_id="p", model_version=0,
                         rollouts=[Rollout(tokens=[1], logprobs=[-0.1], reward=r) for r in (-2.0, 0.0, 3.0)])
    priority = arms.UniformPriority()
    assert {priority.score(r, group) for r in group.rollouts} == {1.0}
    assert _strategy_fingerprint(priority) == {"type": "UniformPriority"}


def test_replay_priority_kwargs():
    specs = {s.name: s for s in arm_specs()}
    assert arms.priority_kwargs(specs["reservoir_age8_off"]) == {}
    assert isinstance(arms.priority_kwargs(specs["reservoir_naive_age32"])["priority"], arms.UniformPriority)


def test_stress_arm_runs_on_the_first_seed_only():
    jobs = plan_jobs("all", "42,43,44", "staleness", 2.0)
    by_arm: dict[str, list[int]] = {}
    for spec, seed in jobs:
        by_arm.setdefault(spec.name, []).append(seed)
    assert by_arm["reservoir_age128_on"] == [42]
    assert all(by_arm[name] == [42, 43, 44] for name in MAIN)
    assert len(jobs) == 8 * 3 + 1


def test_named_arms_and_unknown_arm():
    jobs = plan_jobs("grpo,reservoir_age128_on", "42,43", "gate", 2.0)
    assert [(s.name, seed) for s, seed in jobs] == [("grpo", 42), ("grpo", 43), ("reservoir_age128_on", 42)]
    with pytest.raises(KeyError, match="no arm"):
        plan_jobs("nope", "42", "gate", 2.0)


def test_arm_spec_round_trips_through_a_job_payload():
    spec = arms.arm_by_name(arm_specs(), "reservoir_naive_age32")
    assert ArmSpec(**spec.to_dict()) == spec
