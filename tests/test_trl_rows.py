"""Tests for ``reservoir.integrations._trl_rows``: the tensor <-> Rollout boundary.

TRL hands the adapter a dict of padded batch tensors; the adapter stores
individual completions as ``Rollout`` objects and later writes sampled
rollouts back into rows of such a dict. These tests pin that boundary:
mask handling (prefix masks for completions, suffix masks for left-padded
prompts), which rows are stored, what each stored field equals when
sliced by hand, the logprob sanity rules, and that padding and row
writes return new tensors instead of mutating TRL's.
"""

from __future__ import annotations

import pytest
import torch

from reservoir.integrations._trl_rows import (
    MAX_POSITIVE_LOGPROB,
    GroupSpec,
    clamp_logprobs,
    pad_batch,
    prefix_mask_lengths,
    prompt_id_of,
    rows_to_groups,
    suffix_mask_lengths,
    write_rows,
)
from reservoir.rollout import Rollout

PAD = 0


def f32(values):
    """Round a nested list of floats to float32, the dtype TRL's tensors use.

    Stored logprobs and advantages round-trip float32 -> Python float ->
    float32 exactly; the expectations must be float32 values too.
    """
    return torch.tensor(values, dtype=torch.float32).tolist()


def make_output(
    prompts: list[list[int]],
    completions: list[list[int]],
    advantages: list[float],
    *,
    old_logps: list[list[float]] | None = None,
    ref_logps: list[list[float]] | None = None,
) -> dict:
    """Build a TRL-shaped output dict: left-padded prompts, right-padded completions."""
    lp = max(len(p) for p in prompts)
    lc = max(len(c) for c in completions)
    prompt_ids = torch.full((len(prompts), lp), PAD, dtype=torch.long)
    prompt_mask = torch.zeros((len(prompts), lp), dtype=torch.long)
    completion_ids = torch.full((len(completions), lc), PAD, dtype=torch.long)
    completion_mask = torch.zeros((len(completions), lc), dtype=torch.long)
    for r, (p, c) in enumerate(zip(prompts, completions)):
        if p:
            prompt_ids[r, lp - len(p):] = torch.tensor(p)
            prompt_mask[r, lp - len(p):] = 1
        if c:
            completion_ids[r, : len(c)] = torch.tensor(c)
            completion_mask[r, : len(c)] = 1
    out = {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "num_items_in_batch": completion_mask.sum(),
    }
    for key, rows in (("old_per_token_logps", old_logps), ("ref_per_token_logps", ref_logps)):
        if rows is not None:
            t = torch.zeros((len(rows), lc), dtype=torch.float32)
            for r, row in enumerate(rows):
                t[r, : len(row)] = torch.tensor(row)
            out[key] = t
    return out


# ---------------------------------------------------------------------------
# masks, ids, logprobs
# ---------------------------------------------------------------------------

def test_prefix_mask_lengths_counts_leading_ones():
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 1, 1]])
    assert prefix_mask_lengths(mask) == (2, 1, 0, 3)


def test_prefix_mask_lengths_rejects_a_one_after_a_zero():
    with pytest.raises(ValueError, match="row 1"):
        prefix_mask_lengths(torch.tensor([[1, 1, 0], [1, 0, 1]]))


def test_suffix_mask_lengths_counts_trailing_ones():
    mask = torch.tensor([[0, 1, 1], [0, 0, 1], [1, 1, 1]])
    assert suffix_mask_lengths(mask) == (2, 1, 3)


def test_suffix_mask_lengths_rejects_a_zero_after_a_one():
    with pytest.raises(ValueError, match="row 0"):
        suffix_mask_lengths(torch.tensor([[1, 0, 1]]))


def test_prompt_id_is_deterministic_and_content_addressed():
    assert prompt_id_of([5, 6, 7]) == prompt_id_of((5, 6, 7))
    assert prompt_id_of([5, 6, 7]) != prompt_id_of([5, 6, 8])
    assert len(prompt_id_of([1])) == 16


def test_clamp_logprobs_zeroes_tiny_positives_and_counts_them():
    values, clamped = clamp_logprobs([-0.5, 5e-4, 0.0, MAX_POSITIVE_LOGPROB], row=3)
    assert values == (-0.5, 0.0, 0.0, 0.0)
    assert clamped == 2


@pytest.mark.parametrize("bad", [0.01, float("nan"), float("inf"), -float("inf")])
def test_clamp_logprobs_rejects_large_positive_or_non_finite(bad):
    with pytest.raises(ValueError, match="row 7"):
        clamp_logprobs([-0.1, bad], row=7)


# ---------------------------------------------------------------------------
# rows_to_groups
# ---------------------------------------------------------------------------

def test_rows_to_groups_slices_each_live_row_by_its_masks():
    out = make_output(
        prompts=[[9, 8], [9, 8], [7], [7]],
        completions=[[1, 2, 3], [4], [5, 6], [5, 6]],
        advantages=[0.5, -0.5, 0.0, 0.0],  # group 1 is dead
    )
    logps = torch.tensor([[-0.1, -0.2, -0.3], [-0.4, 0.0, 0.0], [-9.0, -9.0, 0.0], [-9.0, -9.0, 0.0]])

    result = rows_to_groups(out, num_generations=2, step=3, logprobs=logps)

    assert result.dead_groups == 1
    assert result.clamped_logprobs == 0
    assert len(result.groups) == 1
    group = result.groups[0]
    assert isinstance(group, GroupSpec)
    assert group.rows == (0, 1)
    assert group.prompt_id == prompt_id_of([9, 8])
    first, second = group.rollouts
    assert first.tokens == (1, 2, 3) and first.reward == 0.5
    assert list(first.logprobs) == f32([-0.1, -0.2, -0.3])
    assert second.tokens == (4,) and second.reward == -0.5
    assert list(second.logprobs) == f32([-0.4])
    assert first.metadata["prompt_ids"] == [9, 8]
    assert first.metadata["row"] == 0 and second.metadata["row"] == 1
    assert first.metadata["global_step"] == 3
    assert "ref_logprobs" not in first.metadata


def test_rows_to_groups_stores_ref_logprobs_when_present():
    out = make_output(
        prompts=[[1], [1]], completions=[[2, 3], [4]], advantages=[1.0, -1.0],
        ref_logps=[[-1.5, -2.5], [-3.5]],
    )
    logps = torch.full((2, 2), -0.25)

    result = rows_to_groups(out, num_generations=2, step=0, logprobs=logps)

    a, b = result.groups[0].rollouts
    assert a.metadata["ref_logprobs"] == f32([-1.5, -2.5])
    assert b.metadata["ref_logprobs"] == f32([-3.5])


def test_rows_to_groups_skips_empty_rows_and_fully_masked_groups():
    out = make_output(
        prompts=[[1], [1], [2], [2]],
        completions=[[3], [], [], []],   # row 1 truncated; group 1 fully masked
        advantages=[1.0, -1.0, 1.0, -1.0],
    )
    logps = torch.full((4, 1), -0.5)

    result = rows_to_groups(out, num_generations=2, step=0, logprobs=logps)

    assert result.dead_groups == 0
    assert result.skipped_rows == 3
    assert [g.rows for g in result.groups] == [(0,)]


def test_rows_to_groups_counts_clamped_logprobs():
    out = make_output(prompts=[[1], [1]], completions=[[2], [3]], advantages=[1.0, -1.0])
    logps = torch.tensor([[2e-4], [-0.3]])

    result = rows_to_groups(out, num_generations=2, step=0, logprobs=logps)

    assert result.clamped_logprobs == 1
    assert result.groups[0].rollouts[0].logprobs == (0.0,)


def test_rows_to_groups_rejects_batch_not_divisible_by_num_generations():
    out = make_output(prompts=[[1]] * 3, completions=[[2]] * 3, advantages=[1.0, -1.0, 0.0])
    with pytest.raises(ValueError, match="num_generations"):
        rows_to_groups(out, num_generations=2, step=0, logprobs=torch.zeros(3, 1))


def test_rows_to_groups_rejects_differing_prompts_within_a_group():
    out = make_output(prompts=[[1], [2]], completions=[[3], [4]], advantages=[1.0, -1.0])
    with pytest.raises(ValueError, match="group 0"):
        rows_to_groups(out, num_generations=2, step=0, logprobs=torch.full((2, 1), -1.0))


def test_rows_to_groups_rejects_logprobs_of_the_wrong_shape():
    out = make_output(prompts=[[1], [1]], completions=[[3], [4]], advantages=[1.0, -1.0])
    with pytest.raises(ValueError, match="logprobs"):
        rows_to_groups(out, num_generations=2, step=0, logprobs=torch.zeros(2, 5))


# ---------------------------------------------------------------------------
# pad_batch / write_rows
# ---------------------------------------------------------------------------

def test_pad_batch_pads_prompts_left_and_completions_right_without_mutating():
    out = make_output(
        prompts=[[1, 2]], completions=[[3]], advantages=[1.0],
        old_logps=[[-0.5]], ref_logps=[[-0.7]],
    )
    snapshot = {k: v.clone() for k, v in out.items()}

    padded = pad_batch(out, target_prompt_len=4, target_completion_len=3, pad_token_id=PAD)

    assert padded["prompt_ids"].tolist() == [[PAD, PAD, 1, 2]]
    assert padded["prompt_mask"].tolist() == [[0, 0, 1, 1]]
    assert padded["completion_ids"].tolist() == [[3, PAD, PAD]]
    assert padded["completion_mask"].tolist() == [[1, 0, 0]]
    assert padded["old_per_token_logps"].tolist() == f32([[-0.5, 0.0, 0.0]])
    assert padded["ref_per_token_logps"].tolist() == f32([[-0.7, 0.0, 0.0]])
    assert torch.equal(padded["advantages"], out["advantages"])
    for key, before in snapshot.items():
        assert torch.equal(out[key], before), key


def test_pad_batch_is_identity_when_targets_match():
    out = make_output(prompts=[[1, 2]], completions=[[3, 4]], advantages=[1.0])
    padded = pad_batch(out, target_prompt_len=2, target_completion_len=2, pad_token_id=PAD)
    assert padded is out


def test_pad_batch_refuses_to_shrink():
    out = make_output(prompts=[[1, 2]], completions=[[3, 4]], advantages=[1.0])
    with pytest.raises(ValueError):
        pad_batch(out, target_prompt_len=1, target_completion_len=2, pad_token_id=PAD)


def _rollout(tokens, logprobs, reward, prompt, ref=None) -> Rollout:
    meta = {"prompt_ids": list(prompt), "row": 0, "global_step": 0}
    if ref is not None:
        meta["ref_logprobs"] = list(ref)
    return Rollout(tokens=tokens, logprobs=logprobs, reward=reward, metadata=meta)


def test_write_rows_replaces_rows_and_grows_the_batch_when_needed():
    out = make_output(
        prompts=[[1, 2], [1, 2], [5], [5]],
        completions=[[3], [4], [6, 7], [6, 7]],
        advantages=[1.0, -1.0, 0.0, 0.0],
        old_logps=[[-0.1], [-0.2], [-0.3, -0.4], [-0.5, -0.6]],
    )
    snapshot = {k: v.clone() for k, v in out.items()}
    replacements = [
        _rollout([8, 9, 10], [-1.0, -2.0, -3.0], 0.75, prompt=[11, 12, 13]),  # longer than batch
        _rollout([14], [-4.0], -0.25, prompt=[15]),                           # shorter
    ]

    new = write_rows(out, rows=(2, 3), rollouts=replacements, advantages=(0.6, -0.2), pad_token_id=PAD)

    assert new["prompt_ids"].tolist() == [[PAD, 1, 2], [PAD, 1, 2], [11, 12, 13], [PAD, PAD, 15]]
    assert new["prompt_mask"].tolist() == [[0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]]
    assert new["completion_ids"].tolist() == [[3, PAD, PAD], [4, PAD, PAD], [8, 9, 10], [14, PAD, PAD]]
    assert new["completion_mask"].tolist() == [[1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0]]
    assert new["old_per_token_logps"].tolist() == f32([
        [-0.1, 0.0, 0.0], [-0.2, 0.0, 0.0], [-1.0, -2.0, -3.0], [-4.0, 0.0, 0.0],
    ])
    assert new["advantages"].tolist() == f32([1.0, -1.0, 0.6, -0.2])
    assert new["advantages"].dtype == torch.float32
    assert "ref_per_token_logps" not in new
    for key, before in snapshot.items():
        assert torch.equal(out[key], before), key


def test_write_rows_fills_ref_logprobs_when_the_batch_has_them():
    out = make_output(
        prompts=[[1], [1]], completions=[[2], [3]], advantages=[1.0, -1.0],
        old_logps=[[-0.1], [-0.2]], ref_logps=[[-0.5], [-0.6]],
    )
    replacement = _rollout([4, 5], [-1.0, -2.0], 0.5, prompt=[1], ref=[-7.0, -8.0])

    new = write_rows(out, rows=(1,), rollouts=[replacement], advantages=(0.5,), pad_token_id=PAD)

    assert new["ref_per_token_logps"].tolist() == f32([[-0.5, 0.0], [-7.0, -8.0]])


@pytest.mark.parametrize("batch_has_ref, rollout_has_ref", [(True, False), (False, True)])
def test_write_rows_rejects_ref_logprob_mismatch(batch_has_ref, rollout_has_ref):
    out = make_output(
        prompts=[[1]], completions=[[2]], advantages=[1.0], old_logps=[[-0.1]],
        ref_logps=[[-0.5]] if batch_has_ref else None,
    )
    replacement = _rollout([4], [-1.0], 0.5, prompt=[1], ref=[-7.0] if rollout_has_ref else None)
    with pytest.raises(ValueError, match="ref_per_token_logps"):
        write_rows(out, rows=(0,), rollouts=[replacement], advantages=(0.5,), pad_token_id=PAD)


def test_write_rows_requires_old_logprobs_in_the_batch():
    out = make_output(prompts=[[1]], completions=[[2]], advantages=[1.0])
    replacement = _rollout([4], [-1.0], 0.5, prompt=[1])
    with pytest.raises(ValueError, match="old_per_token_logps"):
        write_rows(out, rows=(0,), rollouts=[replacement], advantages=(0.5,), pad_token_id=PAD)


def test_write_rows_rejects_misaligned_arguments():
    out = make_output(prompts=[[1]], completions=[[2]], advantages=[1.0], old_logps=[[-0.1]])
    replacement = _rollout([4], [-1.0], 0.5, prompt=[1])
    with pytest.raises(ValueError):
        write_rows(out, rows=(0,), rollouts=[replacement, replacement], advantages=(0.5,), pad_token_id=PAD)


# ---------------------------------------------------------------------------
# Edge cases and round trips
# ---------------------------------------------------------------------------

def test_rows_to_groups_reports_dead_rows():
    out = make_output(
        prompts=[[1], [1], [2], [2], [3], [3]],
        completions=[[4], [5], [6], [7], [8], [9]],
        advantages=[0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
    )
    result = rows_to_groups(out, num_generations=2, step=0, logprobs=torch.full((6, 1), -1.0))
    assert result.dead_rows == (0, 1, 4, 5)
    assert result.dead_groups == 2


def test_rows_to_groups_rejects_a_non_finite_advantage_by_row():
    out = make_output(prompts=[[1], [1]], completions=[[2], [3]], advantages=[float("nan"), 1.0])
    with pytest.raises(ValueError, match="row 0"):
        rows_to_groups(out, num_generations=2, step=0, logprobs=torch.full((2, 1), -1.0))


def test_rows_to_groups_rejects_a_non_finite_ref_logprob_by_row():
    out = make_output(
        prompts=[[1], [1]], completions=[[2], [3]], advantages=[1.0, -1.0],
        ref_logps=[[-1.0], [float("inf")]],
    )
    with pytest.raises(ValueError, match="row 1"):
        rows_to_groups(out, num_generations=2, step=0, logprobs=torch.full((2, 1), -1.0))


def test_rows_to_groups_checks_prompts_even_on_skipped_rows():
    out = make_output(prompts=[[1], [2]], completions=[[3], []], advantages=[1.0, -1.0])
    with pytest.raises(ValueError, match="group 0"):
        rows_to_groups(out, num_generations=2, step=0, logprobs=torch.full((2, 1), -1.0))


def test_rows_to_groups_rejects_malformed_masks():
    out = make_output(prompts=[[1], [1]], completions=[[2, 3], [4, 5]], advantages=[1.0, -1.0])
    bad = dict(out, completion_mask=torch.tensor([[1, 1], [0, 1]]))
    with pytest.raises(ValueError, match="completion_mask row 1"):
        rows_to_groups(bad, num_generations=2, step=0, logprobs=torch.full((2, 2), -1.0))
    wide = make_output(prompts=[[1, 2], [1, 2]], completions=[[2], [4]], advantages=[1.0, -1.0])
    bad = dict(wide, prompt_mask=torch.tensor([[1, 0], [1, 1]]))
    with pytest.raises(ValueError, match="prompt_mask row 0"):
        rows_to_groups(bad, num_generations=2, step=0, logprobs=torch.full((2, 1), -1.0))


def test_rows_to_groups_accepts_bool_masks_and_an_empty_prompt():
    out = make_output(prompts=[[], []], completions=[[2], [3]], advantages=[1.0, -1.0])
    out["prompt_mask"] = out["prompt_mask"].bool()
    out["completion_mask"] = out["completion_mask"].bool()

    result = rows_to_groups(out, num_generations=2, step=0, logprobs=torch.full((2, 1), -1.0))

    assert result.groups[0].rollouts[0].metadata["prompt_ids"] == []


def test_metadata_survives_a_json_round_trip():
    import json

    out = make_output(
        prompts=[[1, 2], [1, 2]], completions=[[3], [4]], advantages=[1.0, -1.0],
        ref_logps=[[-0.5], [-0.25]],
    )
    result = rows_to_groups(out, num_generations=2, step=5, logprobs=torch.full((2, 1), -1.0))

    for rollout in result.groups[0].rollouts:
        meta = dict(rollout.metadata)
        assert json.loads(json.dumps(meta)) == meta


def test_write_rows_does_not_mutate_inputs_when_the_batch_does_not_grow():
    out = make_output(
        prompts=[[1, 2], [1, 2]], completions=[[3, 4], [5, 6]], advantages=[1.0, -1.0],
        old_logps=[[-0.1, -0.2], [-0.3, -0.4]],
    )
    snapshot = {k: v.clone() for k, v in out.items()}
    replacement = _rollout([7], [-9.0], 0.5, prompt=[8])

    new = write_rows(out, rows=(1,), rollouts=[replacement], advantages=(0.5,), pad_token_id=PAD)

    for key, before in snapshot.items():
        assert torch.equal(out[key], before), key
    for key in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "advantages", "old_per_token_logps"):
        assert new[key].data_ptr() != out[key].data_ptr(), key
    assert new["completion_ids"][1].tolist() == [7, PAD]
    assert new["prompt_ids"][1].tolist() == [PAD, 8]


def test_write_rows_handles_full_width_and_empty_prompt_rollouts():
    out = make_output(prompts=[[1, 2]], completions=[[3, 4]], advantages=[1.0], old_logps=[[-0.1, -0.2]])
    full = _rollout([5, 6], [-1.0, -2.0], 0.5, prompt=[])

    new = write_rows(out, rows=(0,), rollouts=[full], advantages=(0.5,), pad_token_id=PAD)

    assert new["prompt_ids"].tolist() == [[PAD, PAD]] and new["prompt_mask"].tolist() == [[0, 0]]
    assert new["completion_ids"].tolist() == [[5, 6]] and new["completion_mask"].tolist() == [[1, 1]]


def test_write_rows_recomputes_num_items_in_batch():
    out = make_output(prompts=[[1], [1]], completions=[[2], [3]], advantages=[0.0, 0.0], old_logps=[[-0.1], [-0.2]])
    assert out["num_items_in_batch"].item() == 2
    replacement = _rollout([4, 5, 6], [-1.0, -2.0, -3.0], 0.5, prompt=[1])

    new = write_rows(out, rows=(0,), rollouts=[replacement], advantages=(0.5,), pad_token_id=PAD)

    assert new["num_items_in_batch"].item() == 4
    assert new["num_items_in_batch"].dtype == out["num_items_in_batch"].dtype


@pytest.mark.parametrize("rows", [(-1,), (5,), (0, 0)])
def test_write_rows_rejects_bad_row_indices(rows):
    out = make_output(prompts=[[1], [1]], completions=[[2], [3]], advantages=[1.0, -1.0], old_logps=[[-0.1], [-0.2]])
    replacement = _rollout([4], [-1.0], 0.5, prompt=[1])
    with pytest.raises(ValueError, match="row"):
        write_rows(out, rows=rows, rollouts=[replacement] * len(rows), advantages=(0.5,) * len(rows), pad_token_id=PAD)


def test_write_rows_rejects_unknown_per_row_tensors():
    out = make_output(prompts=[[1]], completions=[[2]], advantages=[1.0], old_logps=[[-0.1]])
    out["sampling_per_token_logps"] = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="sampling_per_token_logps"):
        write_rows(out, rows=(0,), rollouts=[_rollout([4], [-1.0], 0.5, prompt=[1])], advantages=(0.5,), pad_token_id=PAD)


def test_rows_to_groups_then_write_rows_reproduces_the_batch():
    out = make_output(
        prompts=[[1, 2], [1, 2], [3], [3]],
        completions=[[4, 5, 6], [7], [8, 9], [8]],
        advantages=[1.0, -1.0, 0.5, -0.5],
        old_logps=[[-0.1, -0.2, -0.3], [-0.4], [-0.5, -0.6], [-0.7]],
        ref_logps=[[-1.1, -1.2, -1.3], [-1.4], [-1.5, -1.6], [-1.7]],
    )
    result = rows_to_groups(out, num_generations=2, step=0, logprobs=out["old_per_token_logps"])
    rows = [r for g in result.groups for r in g.rows]
    rollouts = [ro for g in result.groups for ro in g.rollouts]

    new = write_rows(out, rows=rows, rollouts=rollouts, advantages=[ro.reward for ro in rollouts], pad_token_id=PAD)

    for key in out:
        assert torch.equal(new[key], out[key]), key



def test_near_dead_groups_are_kept_and_counted():
    from reservoir.integrations._trl_rows import NEAR_DEAD_ADVANTAGE, rows_to_groups

    out = make_output(
        prompts=[[1], [1], [2], [2], [3], [3]],
        completions=[[4], [5], [6], [7], [8], [9]],
        advantages=[1.0, -1.0, 5e-4, -5e-4, 0.0, 0.0],
    )
    logps = torch.full_like(out["completion_ids"], -0.5, dtype=torch.float32)
    conv = rows_to_groups(out, num_generations=2, step=0, logprobs=logps)
    assert conv.dead_groups == 1 and conv.dead_rows == (4, 5)
    assert conv.near_dead_groups == 1
    assert len(conv.groups) == 2            # the near-dead group is stored like any live group
    assert 5e-4 < NEAR_DEAD_ADVANTAGE


# ---------------------------------------------------------------------------
# reward provenance
# ---------------------------------------------------------------------------

def test_rows_to_groups_attaches_per_reward_function_values_to_metadata():
    out = make_output(
        prompts=[[9], [9], [7], [7]],
        completions=[[1, 2], [3], [5], [6]],
        advantages=[0.5, -0.5, 0.25, -0.25],
    )
    per_func = torch.tensor([[1.0, 0.2], [0.0, 0.8], [1.0, float("nan")], [0.0, 0.1]])
    conversion = rows_to_groups(
        out, num_generations=2, step=4, logprobs=torch.zeros(4, 2),
        reward_names=["verifier", "judge"], rewards_per_func=per_func,
    )
    rollouts = [r for g in conversion.groups for r in g.rollouts]
    assert rollouts[0].metadata["rewards"] == {"verifier": 1.0, "judge": f32([0.2])[0]}
    assert rollouts[1].metadata["rewards"] == {"verifier": 0.0, "judge": f32([0.8])[0]}
    assert rollouts[2].metadata["rewards"] == {"verifier": 1.0}       # NaN: the judge abstained on this row
    assert rollouts[3].metadata["rewards"] == {"verifier": 0.0, "judge": f32([0.1])[0]}
    assert all(isinstance(v, float) for r in rollouts for v in r.metadata["rewards"].values())


def test_rows_to_groups_without_provenance_adds_no_rewards_key():
    out = make_output(prompts=[[9], [9]], completions=[[1], [2]], advantages=[0.5, -0.5])
    conversion = rows_to_groups(out, num_generations=2, step=0, logprobs=torch.zeros(2, 1))
    assert all("rewards" not in r.metadata for g in conversion.groups for r in g.rollouts)


def test_rows_to_groups_writes_an_empty_mapping_when_every_function_abstained():
    # {} says "provenance recorded, every function abstained"; an absent key says "no provenance".
    out = make_output(prompts=[[9], [9]], completions=[[1], [2]], advantages=[0.5, -0.5])
    per_func = torch.tensor([[float("nan")], [1.0]])
    conversion = rows_to_groups(out, num_generations=2, step=0, logprobs=torch.zeros(2, 1),
                                reward_names=["judge"], rewards_per_func=per_func)
    rollouts = conversion.groups[0].rollouts
    assert rollouts[0].metadata["rewards"] == {} and rollouts[1].metadata["rewards"] == {"judge": 1.0}


@pytest.mark.parametrize("name", ["a\tb", "x" * 257, "   "])
def test_rows_to_groups_rejects_names_the_manifest_would_refuse(name):
    out = make_output(prompts=[[9], [9]], completions=[[1], [2]], advantages=[0.5, -0.5])
    with pytest.raises(ValueError, match="reward_names"):
        rows_to_groups(out, num_generations=2, step=0, logprobs=torch.zeros(2, 1),
                       reward_names=[name], rewards_per_func=torch.zeros(2, 1))


@pytest.mark.parametrize("names, shape, message", [
    (["a"], (2, 2), "reward_names"),
    (["a", "a"], (2, 2), "distinct"),
    (["a", ""], (2, 2), "non-empty"),
    (["a", "b"], (3, 2), "rows"),
    (["a", "b"], (2,), "shape"),
])
def test_rows_to_groups_rejects_misaligned_provenance(names, shape, message):
    out = make_output(prompts=[[9], [9]], completions=[[1], [2]], advantages=[0.5, -0.5])
    with pytest.raises(ValueError, match=message):
        rows_to_groups(out, num_generations=2, step=0, logprobs=torch.zeros(2, 1),
                       reward_names=names, rewards_per_func=torch.zeros(*shape))


def test_rows_to_groups_requires_names_and_values_together():
    out = make_output(prompts=[[9], [9]], completions=[[1], [2]], advantages=[0.5, -0.5])
    with pytest.raises(ValueError, match="together"):
        rows_to_groups(out, num_generations=2, step=0, logprobs=torch.zeros(2, 1), reward_names=["a"])


def test_rows_to_groups_rejects_an_infinite_reward_by_row():
    out = make_output(prompts=[[9], [9]], completions=[[1], [2]], advantages=[0.5, -0.5])
    per_func = torch.tensor([[1.0], [float("inf")]])
    with pytest.raises(ValueError, match="row 1"):
        rows_to_groups(out, num_generations=2, step=0, logprobs=torch.zeros(2, 1),
                       reward_names=["judge"], rewards_per_func=per_func)


def test_reward_metadata_survives_json_and_the_manifest():
    import json
    from reservoir.attest import AttestationLog
    from reservoir.rollout_buffer import RolloutBuffer
    from reservoir.rollout_manifest import ManifestWriter

    out = make_output(prompts=[[9], [9]], completions=[[1], [2]], advantages=[0.5, -0.5])
    conversion = rows_to_groups(out, num_generations=2, step=0, logprobs=torch.zeros(2, 1),
                                reward_names=["verifier"], rewards_per_func=torch.tensor([[1.0], [0.0]]))
    manifest = ManifestWriter()
    buf = RolloutBuffer(capacity=4, attest=AttestationLog(), manifest=manifest)
    spec = conversion.groups[0]
    buf.add_group(spec.prompt_id, 0, spec.rollouts, source="s")
    lines = manifest.records
    assert [l["rewards"] for l in lines] == [{"verifier": 1.0}, {"verifier": 0.0}]
    assert json.loads(json.dumps(lines)) == lines
