"""Tests for the staleness sweep harness (``benchmarks/modal/staleness_sweep.py``).

The pure parts (reward extraction, arm specs, cost estimate, stale-engine
statistics, report aggregation and the figure) are tested on synthetic
inputs; one end-to-end CPU smoke run with TRL's tiny model checks that
every arm trains, writes a record and, for the Reservoir arms, a log the
checker accepts. The GPU sweep itself is the Modal entrypoint's job.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks.staleness import gsm8k, report
from benchmarks.staleness.report import ArmSpec, arm_specs, estimate_cost, replay_kwargs

# --- GSM8K reward ------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("Natalia sold 48/2 = 24 clips.\n#### 72", "72"),
    ("so the answer is $1,234.", "1234"),
    ("#### 3.50", "3.50"),
    ("#### -7", "-7"),
    ("no digits here", None),
    ("first 5 then 6 and finally 7", "7"),
    ("3-4 of them", "4"),
    ("it is -4", "-4"),
])
def test_extract_answer_takes_the_marked_or_last_number(text, expected):
    assert gsm8k.extract_answer(text) == expected


def test_answers_match_numerically():
    assert gsm8k.answers_match("3.50", "3.5")
    assert gsm8k.answers_match("72", "72")
    assert not gsm8k.answers_match("72", "27")
    assert not gsm8k.answers_match(None, "72")


def test_gsm8k_reward_scores_conversational_and_plain_completions():
    completions = [[{"role": "assistant", "content": "x\n#### 72"}], "wrong 1", [{"role": "assistant", "content": "#### 8.0"}]]
    assert gsm8k.gsm8k_reward(completions, answer=["72", "72", "8"]) == [1.0, 0.0, 1.0]


def test_gold_answer_is_the_text_after_the_marker():
    assert gsm8k.gold_answer("a = <<1+1=2>>2\n#### 2") == "2"
    assert gsm8k.gold_answer("#### 1,000") == "1000"


def test_even_length_reward_is_a_coin_flip_on_text_length():
    assert gsm8k.even_length_reward(["ab", "abc", [{"role": "assistant", "content": "abcd"}]]) == [1.0, 0.0, 1.0]


# --- arms --------------------------------------------------------------------


def test_arm_specs_are_plain_plus_three_ages_off_and_on():
    specs = arm_specs(policy_on="gate", max_log_ratio=2.0)
    names = [s.name for s in specs]
    assert names == ["grpo", "reservoir_age8_off", "reservoir_age32_off", "reservoir_age128_off",
                     "reservoir_age8_on", "reservoir_age32_on", "reservoir_age128_on"]
    plain = specs[0]
    assert not plain.replay and plain.max_policy_age is None and plain.policy == "off"
    on = [s for s in specs if s.name.endswith("_on")]
    assert all(s.policy == "gate" and s.max_log_ratio == 2.0 for s in on)
    assert [s.max_policy_age for s in on] == [8, 32, 128]
    assert all(s.policy == "off" for s in specs[1:4])


def test_arm_lookup_by_name_and_unknown_name():
    specs = arm_specs()
    assert report.arm_by_name(specs, "reservoir_age32_on").max_policy_age == 32
    with pytest.raises(KeyError, match="no arm"):
        report.arm_by_name(specs, "nope")


def test_replay_kwargs_off_and_gate():
    assert replay_kwargs(ArmSpec("x", True, 8, "off", None)) == {}
    assert replay_kwargs(ArmSpec("x", True, 8, "gate", 2.0)) == {"max_log_ratio": 2.0, "max_declines_per_step": None}
    with pytest.raises(ValueError, match="policy"):
        replay_kwargs(ArmSpec("x", True, 8, "bogus", None))


def test_policy_kwarg_name_prefers_the_explicit_name_and_is_none_until_the_adapter_has_one():
    assert report.policy_kwarg_name(["capacity", "policy", "staleness_policy"]) == "staleness_policy"
    assert report.policy_kwarg_name(["capacity", "policy"]) == "policy"
    assert report.policy_kwarg_name(["capacity", "max_log_ratio"]) is None


def test_replay_kwargs_staleness_passes_the_policy_or_refuses_before_the_adapter_takes_it():
    import inspect

    from reservoir.integrations.trl import ReservoirReplay

    spec = ArmSpec("x", True, 8, "staleness", None)
    name = report.policy_kwarg_name(list(inspect.signature(ReservoirReplay.__init__).parameters))
    if name is None:
        with pytest.raises(NotImplementedError, match="replay_kwargs"):
            replay_kwargs(spec)
        return
    kwargs = replay_kwargs(spec)
    policy = kwargs[name]
    assert policy.ess_floor == report.STALENESS_ESS_FLOOR and policy.mass_cap == report.STALENESS_MASS_CAP
    assert policy.max_age is None and policy.max_log_ratio is None, "the age axis is the buffer's, not the policy's"


def test_staleness_policy_is_floor_and_cap_only():
    pytest.importorskip("reservoir.integrations._trl_staleness")
    policy = report.staleness_policy()
    assert (policy.ess_floor, policy.mass_cap, policy.max_age, policy.max_log_ratio) == (0.5, 0.25, None, None)


def test_half_life_follows_the_age_bound():
    assert report.half_life_for(8) == 4
    assert report.half_life_for(32) == 16
    assert report.half_life_for(1) == 1


# --- cost --------------------------------------------------------------------


def test_estimate_cost_is_runs_times_steps_plus_overhead_at_the_hourly_rate():
    est = estimate_cost(n_runs=21, max_steps=300, seconds_per_step=12.0, overhead_seconds=300.0, price_per_hour=1.10)
    assert est["per_run_minutes"] == pytest.approx((300 * 12 + 300) / 60)
    assert est["gpu_hours"] == pytest.approx(21 * (300 * 12 + 300) / 3600)
    assert est["dollars"] == pytest.approx(est["gpu_hours"] * 1.10)


def test_dollars_for_a_run_uses_container_seconds():
    assert report.dollars_for(3600.0, 1.10) == pytest.approx(1.10)


# --- stale-engine probe --------------------------------------------------------


def test_probe_stats_counts_confident_disagreements_under_the_mask():
    torch = pytest.importorskip("torch")
    from benchmarks.staleness.engine_probe import probe_stats

    sampling = torch.tensor([[-0.1, -0.2, -5.0], [-0.3, -0.1, -0.1]])
    trainer = torch.tensor([[-0.1, -3.0, -5.0], [-0.3, -0.1, -9.0]])  # (0,1): engine sure, trainer 2.8 lower -> stale
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]]).bool()                 # (1,2) is masked out
    stats = probe_stats(sampling, trainer, mask)
    assert stats["tokens"] == 5
    assert stats["confident_disagreements"] == 1
    assert stats["confident_fraction"] == pytest.approx(0.2)
    assert stats["max_abs_diff"] == pytest.approx(2.8)
    assert stats["mean_abs_diff"] == pytest.approx(2.8 / 5)
    assert stats["stale"] is True


def test_probe_stats_ignores_non_finite_and_empty_masks():
    torch = pytest.importorskip("torch")
    from benchmarks.staleness.engine_probe import probe_stats

    sampling = torch.tensor([[float("nan"), -0.2]])
    trainer = torch.tensor([[-0.1, -0.2]])
    stats = probe_stats(sampling, trainer, torch.tensor([[1, 1]]).bool())
    assert stats["tokens"] == 1 and stats["stale"] is False
    empty = probe_stats(sampling, trainer, torch.tensor([[0, 0]]).bool())
    assert empty["tokens"] == 0 and empty["mean_abs_diff"] is None and empty["stale"] is False


# --- report ------------------------------------------------------------------


def _fake_record(arm: str, seed: int, replay: bool, steps: int = 4, stale: bool = False) -> dict:
    log_history = []
    for step in range(1, steps + 1):
        entry = {"step": step, "reward": 0.1 * step, "frac_reward_zero_std": 0.5 if step % 2 else 0.0}
        if replay:
            entry.update({"reservoir/replaced_rows": 4.0, "reservoir/declined_rows": 1.0 if step == 2 else 0.0,
                          "reservoir/ess_fraction": 0.8, "reservoir/log_ratio_mean_abs": 0.3,
                          "reservoir/dead_groups": 1.0})
        log_history.append(entry)
    log_history.append({"train_runtime": 10.0, "step": steps})
    return {
        "arm": arm, "seed": seed, "replay": replay, "policy": "gate" if arm.endswith("_on") else "off",
        "max_policy_age": 8 if replay else None, "model": "tiny", "gpu": "cpu",
        "config": {"max_steps": steps, "num_generations": 4}, "versions": {"trl": "1.13.0"},
        "wall_clock_seconds": 10.0, "container_seconds": 12.0, "price_per_hour": 1.10,
        "steps": [{"step": s, "seconds": 2.0} for s in range(1, steps + 1)],
        "log_history": log_history,
        "totals": {"replaced_rows": 4 * steps, "declined_rows": 1, "dead_groups": steps, "hook_calls": steps} if replay else None,
        "eval": {"accuracy": 0.25, "n": 4},
        "engine_check": {"available": True, "stale_steps": [2] if stale else [], "steps": []},
        "attestation": {"records": 10, "head_digest": "ab", "checker": {"returncode": 0, "output": "OK"}} if replay else None,
    }


def test_per_step_series_reads_trl_and_reservoir_metrics():
    series = report.per_step_series(_fake_record("reservoir_age8_on", 42, True)["log_history"])
    assert series["step"] == [1, 2, 3, 4]
    assert series["reward"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert series["dead_group_rate"] == [0.5, 0.0, 0.5, 0.0]
    assert series["declined_rows"] == [0.0, 1.0, 0.0, 0.0]
    assert series["ess_fraction"] == [0.8] * 4
    plain = report.per_step_series(_fake_record("grpo", 42, False)["log_history"])
    assert plain["ess_fraction"] == [None] * 4 and plain["replaced_rows"] == [None] * 4


def test_summarize_run_reports_reward_tail_dead_rate_rows_ess_and_dollars():
    summary = report.summarize_run(_fake_record("reservoir_age8_on", 42, True, steps=8, stale=True))
    assert summary["seed"] == 42
    assert summary["train_reward_last_quarter"] == pytest.approx((0.7 + 0.8) / 2)
    assert summary["dead_group_rate"] == pytest.approx(0.25)
    assert summary["replaced_rows"] == 32 and summary["declined_rows"] == 1
    assert summary["ess_fraction_mean"] == pytest.approx(0.8)
    assert summary["eval_accuracy"] == 0.25
    assert summary["dollars"] == pytest.approx(12.0 / 3600 * 1.10)
    assert summary["engine_stale_steps"] == 1
    assert summary["checker_returncode"] == 0
    assert summary["suspect"] is True
    plain = report.summarize_run(_fake_record("grpo", 1, False))
    assert plain["replaced_rows"] is None and plain["ess_fraction_mean"] is None and plain["checker_returncode"] is None
    assert plain["suspect"] is False


def test_build_report_groups_runs_by_arm_with_seed_means(tmp_path: Path):
    for arm, replay in (("grpo", False), ("reservoir_age8_on", True)):
        for seed in (42, 43):
            (tmp_path / f"{arm}_seed{seed}.json").write_text(json.dumps(_fake_record(arm, seed, replay)))
    out = report.build_report(tmp_path, policy_on="gate", max_log_ratio=2.0, reverify=False)
    assert out["experiment"] == "staleness_sweep"
    assert sorted(out["arms"]) == ["grpo", "reservoir_age8_on"]
    arm = out["arms"]["reservoir_age8_on"]
    assert [r["seed"] for r in arm["runs"]] == [42, 43]
    assert arm["summary"]["eval_accuracy"]["mean"] == pytest.approx(0.25)
    assert arm["summary"]["eval_accuracy"]["std"] == pytest.approx(0.0)
    assert arm["summary"]["replaced_rows"]["mean"] == 16
    assert out["totals"]["runs"] == 4
    assert out["totals"]["dollars"] == pytest.approx(4 * 12.0 / 3600 * 1.10)
    assert out["per_step"]["reservoir_age8_on"]["42"]["ess_fraction"] == [0.8] * 4
    assert out["verified_logs"] == 2 and out["reverified"] is False
    assert out["missing_arms"] == ["reservoir_age8_off", "reservoir_age32_off", "reservoir_age128_off",
                                   "reservoir_age32_on", "reservoir_age128_on"]
    assert out["totals"]["suspect_runs"] == 0


def test_build_report_lists_suspect_runs_and_leaves_them_out_of_the_means(tmp_path: Path):
    (tmp_path / "a_seed42.json").write_text(json.dumps(_fake_record("reservoir_age8_on", 42, True)))
    (tmp_path / "a_seed43.json").write_text(json.dumps({**_fake_record("reservoir_age8_on", 43, True, stale=True),
                                                        "eval": {"accuracy": 0.9, "n": 4}}))
    out = report.build_report(tmp_path, reverify=False)
    arm = out["arms"]["reservoir_age8_on"]
    assert arm["suspect_runs"] == [43] and out["totals"]["suspect_runs"] == 1
    assert arm["summary"]["eval_accuracy"] == {"mean": 0.25, "std": 0.0, "n": 1}
    assert len(arm["runs"]) == 2


def test_build_report_refuses_a_reservoir_run_without_a_log_or_with_mixed_configs(tmp_path: Path):
    rec = _fake_record("reservoir_age8_on", 42, True)
    rec["attestation"] = None
    (tmp_path / "x_seed42.json").write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="without an attestation log"):
        report.build_report(tmp_path, reverify=False)
    (tmp_path / "x_seed42.json").write_text(json.dumps(_fake_record("reservoir_age8_on", 42, True)))
    (tmp_path / "y_seed42.json").write_text(json.dumps(_fake_record("grpo", 42, False, steps=6)))
    with pytest.raises(RuntimeError, match="one configuration"):
        report.build_report(tmp_path, reverify=False)
    (tmp_path / "y_seed42.json").write_text(json.dumps(_fake_record("grpo", 43, False)))
    with pytest.raises(RuntimeError, match="seed set"):
        report.build_report(tmp_path, reverify=False)


def test_build_report_reverify_needs_the_log_files(tmp_path: Path):
    (tmp_path / "reservoir_age8_on_seed42.json").write_text(json.dumps(_fake_record("reservoir_age8_on", 42, True)))
    with pytest.raises(FileNotFoundError, match="attest"):
        report.build_report(tmp_path)


def test_build_report_refuses_a_rejected_log(tmp_path: Path):
    rec = _fake_record("reservoir_age8_on", 42, True)
    rec["attestation"]["checker"]["returncode"] = 1
    (tmp_path / "reservoir_age8_on_seed42.json").write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="rejected"):
        report.build_report(tmp_path, reverify=False)


def test_mean_std_handles_missing_values():
    assert report.mean_std([1.0, None, 3.0]) == {"mean": 2.0, "std": pytest.approx(math.sqrt(2)), "n": 2}
    assert report.mean_std([None]) == {"mean": None, "std": None, "n": 0}


def test_plot_report_writes_a_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    for arm, replay in (("grpo", False), ("reservoir_age8_off", True), ("reservoir_age8_on", True)):
        (tmp_path / f"{arm}_seed42.json").write_text(json.dumps(_fake_record(arm, 42, replay)))
    out = report.build_report(tmp_path, reverify=False)
    png = tmp_path / "fig.png"
    report.plot_report(out, png)
    assert png.exists() and png.stat().st_size > 1000


def test_probed_replay_records_every_step_from_either_logprob_source(monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from benchmarks.staleness.engine_probe import ProbedReplay, assert_probe_complete
    from reservoir.integrations import trl as trl_module

    monkeypatch.setattr(trl_module.ReservoirReplay, "mix", lambda self, output, trainer: output)
    monkeypatch.setattr(trl_module.ReservoirReplay, "behavior_logprobs", lambda self, output, trainer: output["_fwd"])
    replay = ProbedReplay(capacity=4)
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=3))
    sampling = torch.tensor([[-0.1, -0.2]])
    mask = torch.tensor([[1, 1]])
    # TRL computed old_per_token_logps: compared in mix, trainer 2.8 nats below a confident engine -> stale
    replay.mix({"sampling_per_token_logps": sampling, "old_per_token_logps": torch.tensor([[-0.1, -3.0]]),
                "completion_mask": mask}, trainer)
    # no old logprobs: compared when the adapter's behavior forward runs
    trainer.state.global_step = 4
    out = {"sampling_per_token_logps": sampling, "completion_mask": mask, "_fwd": torch.tensor([[-0.1, -0.2]])}
    replay.mix(out, trainer)
    replay.behavior_logprobs(out, trainer)
    # a shape mismatch is recorded as skipped, not as clean
    trainer.state.global_step = 5
    out = {"sampling_per_token_logps": torch.zeros(1, 3), "completion_mask": mask, "_fwd": torch.zeros(1, 2)}
    replay.mix(out, trainer)
    replay.behavior_logprobs(out, trainer)
    check = replay.engine_check()
    assert check["available"] and [s["step"] for s in check["steps"]] == [3, 4, 5]
    assert check["stale_steps"] == [3] and check["skipped_steps"] == [5]
    with pytest.raises(RuntimeError, match="skipped"):
        assert_probe_complete(check, hook_calls=3)
    with pytest.raises(RuntimeError, match="no sampling logprobs"):
        assert_probe_complete({"available": False, "steps": [], "skipped_steps": []}, hook_calls=3)
    with pytest.raises(RuntimeError, match="hook calls"):
        assert_probe_complete({"available": True, "steps": [{}, {}], "skipped_steps": []}, hook_calls=3)
    assert_probe_complete({"available": True, "steps": [{}, {}, {}], "skipped_steps": []}, hook_calls=3)


# --- CPU end-to-end smoke --------------------------------------------------------


@pytest.mark.slow
def test_cpu_smoke_runs_a_plain_and_a_reservoir_arm_and_the_logs_verify(tmp_path: Path):
    pytest.importorskip("trl")
    pytest.importorskip("modal")
    from benchmarks.modal import staleness_sweep as sweep

    specs = arm_specs(policy_on="gate", max_log_ratio=2.0)
    records = {}
    for name in ("grpo", "reservoir_age8_on"):
        spec = report.arm_by_name(specs, name)
        rec = sweep.run_arm(spec, seed=42, max_steps=2, model_key="tiny", reward="even_length", use_cpu=True,
                            work_dir=str(tmp_path / name), eval_size=2, per_device_train_batch_size=8,
                            num_generations=4, max_completion_length=8, gradient_accumulation_steps=1)
        records[name] = rec
        sweep.write_run(rec, tmp_path)
    assert records["grpo"]["attestation"] is None and records["grpo"]["totals"] is None
    res = records["reservoir_age8_on"]
    assert res["totals"]["hook_calls"] == 2
    assert res["attestation"]["checker"]["returncode"] == 0, res["attestation"]["checker"]["output"]
    assert (tmp_path / "reservoir_age8_on_seed42.attest.jsonl").exists()
    assert (tmp_path / "reservoir_age8_on_seed42.manifest.jsonl").exists()
    assert len(res["steps"]) == 2 and all(s["seconds"] > 0 for s in res["steps"])
    assert res["eval"]["n"] == 2 and 0.0 <= res["eval"]["accuracy"] <= 1.0
    assert res["engine_check"]["available"] is False  # HF generation: nothing to probe
    out = report.build_report(tmp_path)   # reverify=True: the checker runs again on the written files
    assert sorted(out["arms"]) == ["grpo", "reservoir_age8_on"]
    assert out["verified_logs"] == 1 and out["reverified"] is True
    # a tampered head is caught on rebuild
    rec_path = tmp_path / "reservoir_age8_on_seed42.json"
    rec = json.loads(rec_path.read_text())
    rec["attestation"]["head_digest"] = "00" * 32
    rec_path.write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="head"):
        report.build_report(tmp_path)
