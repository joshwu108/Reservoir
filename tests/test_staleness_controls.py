"""Pure parts of the sweep's control arms and per-step diagnostics (``benchmarks/staleness``)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from benchmarks.staleness import control_arms, diagnostics  # noqa: E402

# --- W-REINFORCE -------------------------------------------------------------------


def test_wreinforce_advantage_is_a_fixed_reference_per_row():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, float("nan")])
    adv = control_arms.wreinforce_advantages(rewards)
    assert adv.tolist() == pytest.approx([0.1, -1.0, 0.1, -1.0, 0.0])


def test_wreinforce_does_not_depend_on_the_rest_of_the_group():
    alone = control_arms.wreinforce_advantages(torch.tensor([1.0, 1.0, 1.0, 1.0]))
    mixed = control_arms.wreinforce_advantages(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert alone[0] == mixed[0] == pytest.approx(0.1)


# --- dynamic sampling --------------------------------------------------------------


def test_live_groups_are_those_with_unequal_rewards():
    rewards = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert control_arms.live_groups(rewards, 4) == [0]
    assert control_arms.live_groups(torch.tensor([0.5, 0.5, 0.5, 0.5000001]), 4) == [0]


def _output(prompt_len: int, completion_len: int, rows: int, start: int = 0, logps: bool = True) -> dict:
    out = {
        "prompt_ids": torch.arange(start, start + rows * prompt_len).view(rows, prompt_len),
        "prompt_mask": torch.ones(rows, prompt_len, dtype=torch.long),
        "completion_ids": torch.arange(start, start + rows * completion_len).view(rows, completion_len),
        "completion_mask": torch.ones(rows, completion_len, dtype=torch.long),
        "advantages": torch.arange(rows, dtype=torch.float32) + start,
        "num_items_in_batch": torch.tensor(rows * completion_len),
    }
    if logps:
        out["old_per_token_logps"] = -torch.ones(rows, completion_len)
    return out


def test_concat_outputs_pads_prompts_left_and_completions_right_and_recounts_items():
    a, b = _output(3, 2, 2), _output(5, 4, 2, start=100)
    out = control_arms.concat_outputs([a, b], pad_id=-7)
    assert out["prompt_ids"].shape == (4, 5) and out["completion_ids"].shape == (4, 4)
    assert out["prompt_ids"][0].tolist() == [-7, -7, 0, 1, 2]
    assert out["prompt_mask"][0].tolist() == [0, 0, 1, 1, 1]
    assert out["completion_ids"][0].tolist() == [0, 1, -7, -7]
    assert out["completion_mask"][0].tolist() == [1, 1, 0, 0]
    assert out["old_per_token_logps"][0].tolist() == [-1.0, -1.0, 0.0, 0.0]
    assert out["advantages"].tolist() == [0.0, 1.0, 100.0, 101.0]
    assert int(out["num_items_in_batch"]) == 2 * 2 + 2 * 4


def test_take_rows_and_unknown_keys_refused():
    out = control_arms.take_rows(_output(3, 2, 4), [1, 3])
    assert out["advantages"].tolist() == [1.0, 3.0] and out["prompt_ids"].shape == (2, 3)
    with pytest.raises(ValueError, match="pixel_values"):
        control_arms.concat_outputs([{**_output(3, 2, 2), "pixel_values": torch.zeros(2)}], pad_id=0)
    with pytest.raises(ValueError, match="old_per_token_logps"):
        control_arms.concat_outputs([_output(3, 2, 2), _output(3, 2, 2, logps=False)], pad_id=0)


def test_fill_batch_keeps_live_groups_first_then_pads_with_primary_dead_groups():
    plan = control_arms.fill_plan(needed=3, live_by_round=[[1], [0, 2], [1]], primary_groups=3)
    assert plan.taken == [(0, 1), (1, 0), (1, 2)] and plan.unfilled == 0
    short = control_arms.fill_plan(needed=3, live_by_round=[[2], []], primary_groups=3)
    assert short.taken == [(0, 2), (0, 0), (0, 1)] and short.unfilled == 2


def test_oversample_prompts_are_seeded_and_repeated_per_generation():
    dataset = [{"prompt": f"p{i}"} for i in range(50)]
    a = control_arms.OversampleStream(dataset, seed=42, num_generations=3)
    b = control_arms.OversampleStream(dataset, seed=42, num_generations=3)
    first = a.next_inputs(2)
    assert first == b.next_inputs(2) and len(first) == 6
    assert first[0] == first[1] == first[2] and first[3] == first[4] == first[5]
    assert control_arms.OversampleStream(dataset, seed=43, num_generations=3).next_inputs(2) != first


# --- diagnostics ---------------------------------------------------------------------


def test_group_outcomes_split_dead_groups_by_value():
    rewards = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0]
    assert diagnostics.group_outcomes(rewards, 4) == {"groups": 3, "dead": 2, "dead_all_wrong": 1, "dead_all_correct": 1}


def test_age_stats_nearest_rank_p90():
    stats = diagnostics.age_stats(list(range(1, 11)))
    assert stats == {"mean": 5.5, "p90": 9, "max": 10}
    assert diagnostics.age_stats([]) == {"mean": None, "p90": None, "max": None}


def test_clip_fractions_split_fresh_replayed_and_by_advantage_sign():
    coef = torch.tensor([[1.5, 1.0], [0.5, 1.0], [1.5, 1.5], [0.5, 1.0]])
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])
    mask = torch.tensor([[1, 1], [1, 1], [1, 0], [1, 1]])
    replayed = torch.tensor([False, False, True, True])
    out = diagnostics.clip_fractions(coef, advantages, mask, replayed, eps_low=0.2, eps_high=0.28)
    assert out == {"fresh_pos": 0.5, "fresh_neg": 0.5, "replayed_pos": 1.0, "replayed_neg": 0.5}
    none = diagnostics.clip_fractions(coef[:2], advantages[:2], mask[:2], replayed[:2], eps_low=0.2, eps_high=0.28)
    assert none["replayed_pos"] is None and none["replayed_neg"] is None


def test_kl_k3_is_zero_at_init_and_positive_after_drift_on_fresh_rows_only():
    logp = torch.log(torch.tensor([[0.5, 0.25], [0.5, 0.5]]))
    mask = torch.ones(2, 2)
    fresh = torch.tensor([True, False])
    assert diagnostics.kl_k3(logp, logp, mask, fresh) == pytest.approx(0.0)
    ref = torch.log(torch.tensor([[0.25, 0.25], [0.01, 0.01]]))
    kl = diagnostics.kl_k3(logp, ref, mask, fresh)
    expected = (math.exp(math.log(0.5)) - math.log(0.5) - 1) / 2   # one token at ratio 0.5, one at 1
    assert kl == pytest.approx(expected)
    assert diagnostics.kl_k3(logp, ref, mask, torch.tensor([False, False])) is None


def test_running_z_needs_history_and_uses_the_trailing_window():
    z = diagnostics.RunningZ(window=4, min_history=3)
    assert [z.push(x) for x in (1.0, 2.0, 3.0)] == [None, None, None]
    assert z.push(2.0) == pytest.approx(0.0)
    window = (2.0, 3.0, 2.0)            # the trailing four were 1, 2, 3, 2; window 4 keeps all four
    mean = (1.0 + 2.0 + 3.0 + 2.0) / 4
    std = math.sqrt(sum((v - mean) ** 2 for v in (1.0, *window)) / 3)
    assert z.push(10.0) == pytest.approx((10.0 - mean) / std)
    flat = diagnostics.RunningZ(window=4, min_history=2)
    flat.push(1.0), flat.push(1.0)
    assert flat.push(5.0) is None, "zero spread: no z-score"


def test_running_z_value_against_previous_window():
    z = diagnostics.RunningZ(window=3, min_history=3)
    for x in (1.0, 2.0, 3.0):
        z.push(x)
    assert z.push(4.0) == pytest.approx((4.0 - 2.0) / 1.0)
    assert z.push(0.0) == pytest.approx((0.0 - 3.0) / 1.0)


def test_a_group_with_an_unscorable_reward_is_dead_for_the_filter():
    assert control_arms.live_groups(torch.tensor([1.0, float("nan"), 0.0, 1.0]), 2) == [1]


def test_kl_ignores_overflow_on_masked_positions():
    logp = torch.tensor([[-1.0, -1000.0]])
    ref = torch.tensor([[-1.0, 0.0]])
    assert diagnostics.kl_k3(logp, ref, torch.tensor([[1, 0]]), torch.tensor([True])) == pytest.approx(0.0)


def test_guarded_evaluation_records_the_error_instead_of_raising():
    from benchmarks.staleness.evaluation import guarded

    def boom() -> dict:
        raise ValueError("engine gone")

    out = guarded(boom, step=7)
    assert out["step"] == 7 and out["error"] == "ValueError: engine gone" and out["seconds"] >= 0
    assert guarded(lambda: {"accuracy": 0.5})["accuracy"] == 0.5


def test_local_dry_run_prints_the_gpu_plan_and_estimate():
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip("modal")
    root = Path(__file__).parents[1]
    out = subprocess.run([sys.executable, "benchmarks/modal/staleness_sweep.py", "--dry-run"], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    assert out.startswith("25 runs on A10G")
    stress = [line.split() for line in out.splitlines() if "reservoir_age128_on" in line]
    assert stress == [["reservoir_age128_on", "seed", "42", "(stress", "arm,", "first", "seed", "only)"]]
    assert "65131 eval completions" in out
