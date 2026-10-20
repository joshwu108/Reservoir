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


def test_replay_kwargs_staleness_passes_a_policy_the_adapter_accepts():
    # The adapter's __init__ is wrapped by _trl_presets as (capacity, *args, **kwargs), so a signature probe
    # cannot see staleness_policy; constructing the replay is the check that the keyword is taken.
    from reservoir.integrations.trl import ReservoirReplay

    kwargs = replay_kwargs(ArmSpec("x", True, 8, "staleness", None))
    policy = kwargs["staleness_policy"]
    assert policy.ess_floor == report.STALENESS_ESS_FLOOR and policy.mass_cap == report.STALENESS_MASS_CAP
    assert policy.max_age == 4 and policy.max_log_ratio is None   # half the buffer's max_policy_age of 8
    replay = ReservoirReplay(capacity=8, **kwargs)
    assert replay.policy == policy
    replay.close()


def test_staleness_policy_binds_below_each_buffer_age():
    pytest.importorskip("reservoir.integrations._trl_staleness")
    from reservoir.integrations._trl_staleness import PRESET_RULES

    conservative = PRESET_RULES["conservative"]
    on_arms = [spec for spec in report.arm_specs() if spec.name.endswith("_on")]
    bounds = {}
    for spec in on_arms:
        policy = report.staleness_policy(spec.max_policy_age)
        assert (policy.ess_floor, policy.mass_cap) == (conservative["ess_floor"], conservative["mass_cap"])
        assert policy.max_log_ratio is None
        # The bound binds (it is below the buffer's age) but scales with it, so the _on arms stay distinct.
        assert 0 < policy.max_age < spec.max_policy_age
        bounds[spec.name] = policy.max_age
    assert bounds == {"reservoir_age8_on": 4, "reservoir_age32_on": 16, "reservoir_age128_on": 64}


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
                          "reservoir/sampling_design_ess_fraction": 0.8, "reservoir/log_ratio_mean_abs": 0.3,
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


def test_per_step_series_reads_the_pre_rename_ess_metric():
    history = [{"step": 1, "reward": 0.0, "reservoir/ess_fraction": 0.7},
               {"step": 2, "reward": 0.0, "reservoir/sampling_design_ess_fraction": 0.9}]
    assert report.per_step_series(history)["ess_fraction"] == [0.7, 0.9]


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
    assert out["missing_arms"] == ["grpo_filter", "wreinforce", "reservoir_age8_off", "reservoir_age32_off",
                                   "reservoir_age32_on", "reservoir_naive_age32", "reservoir_age128_on"]
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
    assert res["config"]["vllm_seed"] is None
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


# --- control arms, diagnostics and evaluation in the report -----------------------------------


def _full_record(arm: str, seed: int, *, replay: bool = False, trainer: str = "grpo", stress: bool = False,
                 rollouts: int = 32, per_item=(1, 0, 1, 0), steps: int = 8) -> dict:
    rec = _fake_record(arm, seed, replay, steps=steps)
    for entry in rec["log_history"]:
        if "reward" in entry:
            entry.update({"sweep/rollouts_generated": float(rollouts), "sweep/dead_all_wrong_rate": 0.25,
                          "sweep/dead_all_correct_rate": 0.25, "sweep/kl_to_init": 0.01, "entropy": 0.5,
                          "completions/mean_length": 100.0, "sweep/lm_head_grad_z": 0.1})
            if replay:
                entry.update({"sweep/replayed_row_fraction": 0.25, "sweep/same_step_fraction": 0.0,
                              "sweep/replayed_age_p90": 3.0, "sweep/clip_fraction/replayed_pos": 0.1})
    rec.update({"trainer": trainer, "stress": stress, "priority": "advantage",
                "evals": {"gsm8k_test": {"accuracy": sum(per_item) / len(per_item), "per_item": list(per_item)},
                          "math500": {"accuracy": 0.5, "per_item": [1, 0]},
                          "gsm8k_avg_at_k": [{"step": 4, "accuracy": 0.3}, {"step": 8, "accuracy": 0.4}]}})
    rec["eval"] = {"accuracy": rec["evals"]["gsm8k_test"]["accuracy"], "n": len(per_item)}
    return rec


def _write(tmp_path: Path, rec: dict) -> None:
    (tmp_path / f"{rec['arm']}_seed{rec['seed']}.json").write_text(json.dumps(rec))


def test_per_step_series_reads_the_sweep_diagnostics():
    series = report.per_step_series(_full_record("reservoir_age8_off", 42, replay=True)["log_history"])
    assert series["rollouts"] == [32.0] * 8
    assert series["dead_all_wrong"] == [0.25] * 8 and series["kl_to_init"] == [0.01] * 8
    assert series["replayed_age_p90"] == [3.0] * 8 and series["same_step_fraction"] == [0.0] * 8
    assert series["clip_replayed_pos"] == [0.1] * 8 and series["entropy"] == [0.5] * 8


def test_report_adds_bootstrap_cis_differences_auc_and_rollout_matched_cuts(tmp_path: Path):
    for seed in (42, 43):
        _write(tmp_path, _full_record("grpo", seed, per_item=(1, 0, 1, 0)))
        _write(tmp_path, _full_record("grpo_filter", seed, trainer="grpo_filter", rollouts=64, per_item=(1, 1, 1, 0)))
    _write(tmp_path, _full_record("reservoir_age128_on", 42, replay=True, trainer="reservoir", stress=True))
    out = report.build_report(tmp_path, reverify=False)
    grpo, filt = out["arms"]["grpo"], out["arms"]["grpo_filter"]
    assert grpo["eval_ci"]["mean"] == pytest.approx(0.5) and grpo["eval_ci"]["n_seeds"] == 2
    assert filt["eval_diff_vs_grpo"]["mean"] == pytest.approx(0.25)
    assert grpo["eval_diff_vs_grpo"] is None
    assert filt["math500_ci"]["mean"] == pytest.approx(0.5)
    assert filt["summary"]["reward_auc"]["mean"] == pytest.approx(sum(0.1 * s for s in range(1, 9)) / 8)
    assert out["rollout_budget"] == 8 * 32
    assert filt["summary"]["steps_at_rollout_budget"]["mean"] == 4
    assert grpo["summary"]["steps_at_rollout_budget"]["mean"] == 8
    assert filt["summary"]["avg_at_k_at_rollout_budget"]["mean"] == pytest.approx(0.3)
    target = 0.95 * grpo["summary"]["train_reward_last_quarter"]["mean"]
    assert out["steps_to_95pct_grpo_target"] == pytest.approx(target)
    # a 10-step rolling mean over 8 steps of 0.1..0.8 peaks at 0.45, below the 0.71 target
    assert filt["reached_95pct_grpo"] == 0 and filt["summary"]["steps_to_95pct_grpo"]["n"] == 0
    stress = out["arms"]["reservoir_age128_on"]
    assert stress["stress"] is True and [r["seed"] for r in stress["runs"]] == [42]


def test_report_refuses_a_stress_seed_outside_the_main_seed_set(tmp_path: Path):
    _write(tmp_path, _full_record("grpo", 42))
    _write(tmp_path, _full_record("reservoir_age128_on", 99, replay=True, trainer="reservoir", stress=True))
    with pytest.raises(RuntimeError, match="stress"):
        report.build_report(tmp_path, reverify=False)


def test_estimate_jobs_cost_weights_grpo_filter_and_adds_eval_time():
    specs = {s.name: s for s in arm_specs()}
    est = report.estimate_jobs_cost([specs["grpo"], specs["grpo_filter"]], max_steps=100, seconds_per_step=10.0,
                                    overhead_seconds=60.0, price_per_hour=3.6, eval_seconds=40.0)
    per_run = [100 * 10.0 + 60 + 40, 100 * 10.0 * report.GRPO_FILTER_STEP_FACTOR + 60 + 40]
    assert est["runs"] == 2
    assert est["gpu_hours"] == pytest.approx(sum(per_run) / 3600)
    assert est["dollars"] == pytest.approx(sum(per_run) / 3600 * 3.6)
    assert est["per_run_minutes"] == pytest.approx(sum(per_run) / 2 / 60)


def test_report_marks_a_failed_probe_suspect_and_refuses_same_step_replay_and_mixed_settings(tmp_path: Path):
    failed = _full_record("reservoir_age8_off", 42, replay=True, trainer="reservoir")
    failed["engine_check"] = {"available": False, "steps": [], "stale_steps": [], "failure": "no sampling logprobs"}
    _write(tmp_path, failed)
    run = report.build_report(tmp_path, reverify=False)["arms"]["reservoir_age8_off"]["runs"][0]
    assert run["suspect"] and run["suspect_reasons"] == ["probe failed: no sampling logprobs"]
    same = _full_record("reservoir_age8_off", 42, replay=True, trainer="reservoir")
    same["log_history"][2]["sweep/same_step_fraction"] = 0.5
    _write(tmp_path, same)
    with pytest.raises(RuntimeError, match="own step"):
        report.build_report(tmp_path, reverify=False)
    _write(tmp_path, _full_record("reservoir_age8_off", 42, replay=True, trainer="reservoir"))
    mixed = _full_record("reservoir_age8_off", 43, replay=True, trainer="reservoir")
    mixed["policy"] = "staleness"
    _write(tmp_path, mixed)
    with pytest.raises(RuntimeError, match="mix settings"):
        report.build_report(tmp_path, reverify=False)


def test_report_refuses_records_that_differ_in_precision(tmp_path: Path):
    a, b = _full_record("grpo", 42), _full_record("wreinforce", 42, trainer="wreinforce")
    a["config"]["bf16"], b["config"]["bf16"] = True, False
    _write(tmp_path, a)
    _write(tmp_path, b)
    with pytest.raises(RuntimeError, match="precision"):
        report.build_report(tmp_path, reverify=False)


def test_reverify_refuses_a_log_without_the_same_step_floor(tmp_path: Path):
    from reservoir.rollout import Rollout
    from reservoir.rollout_buffer import RolloutBuffer

    stem = tmp_path / "reservoir_age8_off_seed42"
    buf = RolloutBuffer(capacity=8, half_life=4, max_policy_age=8, seed=1, min_entry_age=0,
                        attest=str(stem) + ".attest.jsonl", manifest=str(stem) + ".manifest.jsonl")
    buf.add_group("p", 0, [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in (1.0, 0.0)])
    buf.sample(1, current_version=0)
    head = buf.attestation_log.head_digest
    buf.close()
    with pytest.raises(RuntimeError, match="min_entry_age"):
        report.reverify_log(stem.with_suffix(".json"), head)


def test_control_arm_probe_marks_stale_steps_and_first_round_reward_is_the_reward_series(tmp_path: Path):
    rec = _full_record("grpo_filter", 42, trainer="grpo_filter")
    rec["engine_check"] = {"available": False, "steps": [], "stale_steps": []}
    steps = [e for e in rec["log_history"] if "reward" in e]
    for e in steps:
        e["sweep/engine_confident_fraction"] = 0.0
        e["sweep/first_round_reward"] = 0.9
        e["sweep/filter_unfilled_groups"] = 1.0
    steps[3]["sweep/engine_confident_fraction"] = 0.05
    summary = report.summarize_run(rec)
    assert summary["engine_stale_steps"] == 1 and summary["suspect"]
    assert summary["train_reward_last_quarter"] == pytest.approx(0.9)
    assert summary["filter_unfilled_groups_total"] == 8
    series = report.per_step_series(rec["log_history"])
    assert series["reward_all_rounds"][0] == pytest.approx(0.1)


def test_steps_to_target_is_censored_at_the_run_length(tmp_path: Path):
    for seed in (42, 43):
        _write(tmp_path, _full_record("grpo", seed))
    run = report.build_report(tmp_path, reverify=False)["arms"]["grpo"]["runs"][0]
    assert run["steps_to_95pct_grpo"] is None and run["steps_to_95pct_grpo_censored"] == 8


def test_failed_evaluations_are_listed_and_an_unprobed_vllm_run_is_suspect(tmp_path: Path):
    rec = _full_record("grpo", 42)
    rec["evals"]["math500"] = {"step": 8, "error": "ValueError: prompt too long", "seconds": 1.0}
    rec["evals"]["gsm8k_avg_at_k"].append({"step": 12, "error": "RuntimeError: boom", "seconds": 0.5})
    rec["config"]["vllm"] = {"version": "0.28.0"}
    rec["engine_check"] = {"available": False, "steps": [], "stale_steps": []}
    _write(tmp_path, rec)
    out = report.build_report(tmp_path, reverify=False)
    run = out["arms"]["grpo"]["runs"][0]
    assert run["eval_errors"] == ["math500: ValueError: prompt too long", "gsm8k_avg_at_k step 12: RuntimeError: boom"]
    assert out["arms"]["grpo"]["eval_error_runs"] == [42]
    assert run["suspect_reasons"] == ["vLLM run was never engine-probed"]
    assert "evaluation failed: grpo seed 42: math500" in report.format_table(out)


def test_fit_prompts_drops_only_the_over_length_items():
    datasets = pytest.importorskip("datasets")
    from benchmarks.staleness.evaluation import fit_prompts

    class Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return messages[-1]["content"]

        def __call__(self, text, add_special_tokens):
            return {"input_ids": text.split()}

    data = datasets.Dataset.from_dict({"prompt": [[{"role": "user", "content": "a b"}],
                                                  [{"role": "user", "content": "a b c d e"}]], "answer": ["1", "2"]})
    kept, dropped = fit_prompts(data, Tok(), max_prompt_tokens=3)
    assert dropped == 1 and kept["answer"] == ["1"]
    same, none = fit_prompts(data, Tok(), max_prompt_tokens=10)
    assert none == 0 and same is data


def test_sweep_entrypoint_puts_sources_on_path_before_benchmark_imports(tmp_path, monkeypatch):
    # Modal imports the entrypoint before any function body runs, and benchmarks.staleness
    # imports reservoir at module level: the sources must be on sys.path at import time.
    from benchmarks.modal import staleness_sweep

    checkout = tmp_path / "checkout"
    (checkout / "src").mkdir(parents=True)
    (checkout / "benchmarks" / "modal").mkdir(parents=True)
    monkeypatch.setattr("sys.path", [])
    staleness_sweep._add_repo_to_path(checkout / "benchmarks" / "modal" / "staleness_sweep.py")
    import sys

    assert str(checkout / "src") in sys.path
    assert str(checkout) in sys.path


def test_gpu_defaults_leave_room_for_a_full_gsm8k_solution():
    # At 256 completion tokens half of the 16-step GPU smoke's completions were truncated (completions/clipped_ratio
    # 0.50), so the reward mostly taught "finish sooner". 512 tokens, with micro-batch 4 x 8 accumulation steps to
    # keep the logits' memory at the smoke's 8 x 256 level and 32 completions (4 prompts x 8) per optimizer step.
    import inspect

    from benchmarks.modal import staleness_sweep

    defaults = {k: v.default for k, v in inspect.signature(staleness_sweep.run_arm).parameters.items()}
    assert defaults["max_completion_length"] == 512
    assert (defaults["per_device_train_batch_size"], defaults["gradient_accumulation_steps"]) == (4, 8)
    assert defaults["per_device_train_batch_size"] * defaults["gradient_accumulation_steps"] == 32
    assert defaults["vllm_max_model_length"] >= 512 + 512   # MATH500 prompts plus their 512-token budget
