"""Tests for ``reservoir.integrations._verl_rows``: the DataProto tensor <-> Rollout boundary.

verl hands the adapter a TensorDict of padded batch tensors keyed by uid;
these tests pin that boundary: the layout checks (input_ids as prompts
followed by responses, response_mask as the tail of attention_mask, 2-D
position ids, outcome-level advantages), grouping by non-contiguous uid,
what each stored field equals when sliced by hand, and that writes
rebuild every dependent tensor (input_ids, attention_mask, position_ids,
returns, token-level scores) without touching the input tensors.
"""

from __future__ import annotations

import copy

import pytest
import torch

from reservoir.integrations._verl_rows import (
    REQUIRED_KEYS,
    check_layout,
    global_token_num,
    pad_batch,
    rows_to_groups,
    tensor_digest,
    verify_written_rows,
    write_rows,
)
from reservoir.rollout import Rollout

PAD = 0


def f32(values):
    return torch.tensor(values, dtype=torch.float32).tolist()


def verl_position_ids(attention_mask: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """verl's rule: cumsum-clip over the prompt, then last position + 1.. over the response."""
    prompt = torch.clip(torch.cumsum(attention_mask[:, :prompt_len], dim=-1) - 1, min=0)
    lr = attention_mask.size(1) - prompt_len
    response = prompt[:, -1:] + torch.arange(1, lr + 1)
    return torch.cat([prompt, response], dim=1)


def make_batch(
    prompts: list[list[int]],
    responses: list[list[int]],
    advantages: list[float],
    *,
    logps: list[list[float]] | None = None,
    ref_logps: list[list[float]] | None = None,
    scores: list[float] | None = None,
    uids: list[str] | None = None,
) -> tuple[dict, list[str]]:
    """Build a verl-shaped batch (left-padded prompts, right-padded responses) and its uids."""
    lp = max(len(p) for p in prompts)
    lr = max(len(c) for c in responses)
    b = len(prompts)
    prompt_ids = torch.full((b, lp), PAD, dtype=torch.long)
    prompt_mask = torch.zeros((b, lp), dtype=torch.long)
    response_ids = torch.full((b, lr), PAD, dtype=torch.long)
    response_mask = torch.zeros((b, lr), dtype=torch.long)
    for r, (p, c) in enumerate(zip(prompts, responses)):
        if p:
            prompt_ids[r, lp - len(p):] = torch.tensor(p)
            prompt_mask[r, lp - len(p):] = 1
        if c:
            response_ids[r, : len(c)] = torch.tensor(c)
            response_mask[r, : len(c)] = 1
    attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
    adv = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1) * response_mask
    scores_t = torch.zeros((b, lr), dtype=torch.float32)
    for r, c in enumerate(responses):
        if c:
            scores_t[r, len(c) - 1] = (scores or [1.0] * b)[r]

    def per_token(rows):
        t = torch.zeros((b, lr), dtype=torch.float32)
        for r, row in enumerate(rows):
            t[r, : len(row)] = torch.tensor(row)
        return t

    batch = {
        "prompts": prompt_ids,
        "responses": response_ids,
        "input_ids": torch.cat([prompt_ids, response_ids], dim=1),
        "attention_mask": attention_mask,
        "position_ids": verl_position_ids(attention_mask, lp),
        "response_mask": response_mask,
        "old_log_probs": per_token(logps) if logps is not None else torch.full((b, lr), -0.5) * response_mask,
        "advantages": adv,
        "returns": adv.clone(),
        "token_level_scores": scores_t,
        "token_level_rewards": scores_t.clone(),
    }
    if ref_logps is not None:
        batch["ref_log_prob"] = per_token(ref_logps)
    return batch, (uids or [f"u{r // 2}" for r in range(b)])


def two_groups(**kw):
    """Live group u0 (rows 0, 1) and dead group u1 (rows 2, 3)."""
    return make_batch(
        prompts=[[20, 21], [20, 21], [22], [22]],
        responses=[[3], [4, 4], [5, 5], [5]],
        advantages=[0.25, -0.25, 0.0, 0.0],
        logps=[[-0.1], [-0.2, -0.3], [-0.4, -0.5], [-0.6]],
        scores=[1.0, 0.0, 1.0, 1.0],
        **kw,
    )


# ---------------------------------------------------------------------------
# Layout checks
# ---------------------------------------------------------------------------

def test_check_layout_accepts_a_verl_shaped_batch():
    batch, _ = two_groups()
    layout = check_layout(batch)
    assert (layout.rows, layout.prompt_len, layout.response_len) == (4, 2, 2)


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_a_missing_key_is_named(key):
    batch, _ = two_groups()
    del batch[key]
    with pytest.raises(ValueError, match=key):
        check_layout(batch)


def test_input_ids_must_be_prompts_then_responses():
    batch, _ = two_groups()
    batch["input_ids"] = batch["input_ids"].clone()
    batch["input_ids"][0, -1] += 1
    with pytest.raises(ValueError, match="prompts followed by responses"):
        check_layout(batch)


def test_attention_mask_tail_must_equal_response_mask():
    batch, _ = two_groups()
    batch["attention_mask"] = batch["attention_mask"].clone()
    batch["attention_mask"][0, -1] = 1
    with pytest.raises(ValueError, match="response_mask"):
        check_layout(batch)


def test_three_dimensional_position_ids_are_refused():
    batch, _ = two_groups()
    batch["position_ids"] = batch["position_ids"].unsqueeze(1).expand(-1, 3, -1)
    with pytest.raises(NotImplementedError, match="position_ids"):
        check_layout(batch)


def test_kl_in_reward_is_refused():
    batch, _ = two_groups()
    batch["token_level_rewards"] = batch["token_level_scores"] - 0.01
    with pytest.raises(NotImplementedError, match="use_kl_in_reward"):
        check_layout(batch)


def test_per_token_advantages_are_refused():
    batch, uids = two_groups()
    batch["advantages"] = batch["advantages"].clone()
    batch["advantages"][1, 1] = 0.5
    with pytest.raises(NotImplementedError, match="vary across"):
        rows_to_groups(batch, uids, step=0)


def test_a_multi_turn_loss_mask_is_not_a_prefix_mask():
    batch, uids = two_groups()
    for key in ("response_mask", "attention_mask"):
        batch[key] = batch[key].clone()
    batch["response_mask"][1] = torch.tensor([0, 1])
    batch["attention_mask"][1, -2:] = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="response_mask row 1 is not a prefix"):
        rows_to_groups(batch, uids, step=0)


# ---------------------------------------------------------------------------
# Batch -> rollouts
# ---------------------------------------------------------------------------

def test_rows_to_groups_stores_live_rows_and_returns_dead_rows():
    batch, uids = two_groups()
    conv = rows_to_groups(batch, uids, step=3)

    assert conv.dead_rows == (2, 3) and conv.dead_groups == 1
    assert conv.skipped_rows == 0 and conv.near_dead_groups == 0
    (group,) = conv.groups
    assert group.rows == (0, 1)
    r0, r1 = group.rollouts
    assert r0.tokens == (3,) and list(r0.logprobs) == f32([-0.1]) and r0.reward == 0.25
    assert r1.tokens == (4, 4) and list(r1.logprobs) == f32([-0.2, -0.3]) and r1.reward == -0.25
    assert r0.metadata == {"prompt_ids": [20, 21], "row": 0, "global_step": 3, "uid": "u0", "score": 1.0}
    assert r1.metadata["score"] == 0.0


def test_groups_follow_uids_not_row_order():
    # balance_batch reorders rows by length; the uid is what names a group.
    batch, _ = make_batch(
        prompts=[[1], [2], [1], [2]], responses=[[5], [6], [7], [8]], advantages=[1.0, 0.0, -1.0, 0.0],
    )
    conv = rows_to_groups(batch, ["a", "b", "a", "b"], step=0)
    assert [g.rows for g in conv.groups] == [(0, 2)]
    assert conv.dead_rows == (1, 3)


def test_rows_of_one_uid_must_share_a_prompt():
    batch, _ = make_batch(prompts=[[1], [2]], responses=[[5], [6]], advantages=[1.0, -1.0])
    with pytest.raises(ValueError, match="do not share one prompt"):
        rows_to_groups(batch, ["a", "a"], step=0)


def test_uid_count_must_match_the_batch():
    batch, uids = two_groups()
    with pytest.raises(ValueError, match="uids"):
        rows_to_groups(batch, uids[:-1], step=0)


def test_empty_responses_are_skipped_in_live_groups_and_count_as_zero_in_dead_ones():
    batch, _ = make_batch(
        prompts=[[1], [1], [2], [2]], responses=[[5], [], [], []], advantages=[1.0, 0.0, 0.0, 0.0],
    )
    conv = rows_to_groups(batch, ["a", "a", "b", "b"], step=0)
    assert conv.skipped_rows == 1 and len(conv.groups[0].rollouts) == 1
    assert conv.dead_rows == (2, 3) and conv.dead_groups == 1


def test_near_dead_groups_are_counted_but_stored():
    batch, uids = make_batch(prompts=[[1], [1]], responses=[[5], [6]], advantages=[1e-5, -1e-5])
    conv = rows_to_groups(batch, uids, step=0)
    assert conv.near_dead_groups == 1 and len(conv.groups) == 1 and conv.dead_rows == ()


def grpo_advantages(scores: list[float]) -> list[float]:
    """verl's compute_grpo_outcome_advantage for one group, in float32 as verl runs it."""
    t = torch.tensor(scores, dtype=torch.float32)
    return ((t - t.mean()) / (t.std() + 1e-6)).tolist()


def test_equal_non_dyadic_scores_leave_a_residue_that_is_counted_as_near_dead():
    # 0.3 six times: verl's float32 mean carries a rounding residue, the std is of that order and
    # the quotient is a few hundredths. verl trains on those rows, so the adapter stores the group
    # (the criterion is zero gradient, not equal scores) and counts it.
    advs = grpo_advantages([0.3] * 6)
    assert any(a != 0.0 for a in advs) and max(abs(a) for a in advs) > 1e-3
    batch, uids = make_batch(
        prompts=[[1]] * 6, responses=[[2], [3], [4], [5], [6], [7]], advantages=advs, scores=[0.3] * 6, uids=["g"] * 6,
    )
    conv = rows_to_groups(batch, uids, step=0)
    assert conv.dead_groups == 0 and conv.near_dead_groups == 1 and len(conv.groups[0].rollouts) == 6
    # Exactly representable equal scores are exactly dead.
    advs = grpo_advantages([1.0] * 6)
    assert advs == [0.0] * 6
    batch, uids = make_batch(prompts=[[1]] * 6, responses=[[2]] * 6, advantages=advs, scores=[1.0] * 6, uids=["g"] * 6)
    assert rows_to_groups(batch, uids, step=0).dead_groups == 1


def test_an_unknown_per_row_tensor_is_refused_by_the_layout_check():
    batch, uids = two_groups()
    batch["values"] = torch.zeros(4, 2)
    with pytest.raises(ValueError, match="values"):
        check_layout(batch)
    with pytest.raises(ValueError, match="values"):
        rows_to_groups(batch, uids, step=0)


def test_ref_logprobs_are_stored_when_present():
    batch, uids = two_groups(ref_logps=[[-1.0], [-2.0, -2.1], [-3.0, -3.1], [-4.0]])
    conv = rows_to_groups(batch, uids, step=0)
    assert conv.groups[0].rollouts[1].metadata["ref_logprobs"] == f32([-2.0, -2.1])


def test_non_finite_values_name_the_row():
    batch, uids = two_groups()
    batch["old_log_probs"] = batch["old_log_probs"].clone()
    batch["old_log_probs"][1, 0] = float("nan")
    with pytest.raises(ValueError, match="row 1"):
        rows_to_groups(batch, uids, step=0)


# ---------------------------------------------------------------------------
# Rollouts -> batch
# ---------------------------------------------------------------------------

def stored(tokens, logprobs, prompt, score=1.0, ref=None, reward=0.5) -> Rollout:
    metadata = {"prompt_ids": prompt, "row": 0, "global_step": 0, "uid": "x", "score": score}
    if ref is not None:
        metadata["ref_logprobs"] = ref
    return Rollout(tokens=tokens, logprobs=logprobs, reward=reward, metadata=metadata)


def test_write_rows_rebuilds_every_dependent_tensor():
    batch, uids = two_groups()
    pristine = copy.deepcopy(batch)
    rollouts = [stored([7, 8], [-0.7, -0.8], [30, 31], score=0.0), stored([9], [-0.9], [32])]

    new = write_rows(batch, [2, 3], rollouts, [0.5, -0.5], PAD)

    for key in pristine:
        assert torch.equal(batch[key], pristine[key]), key
    assert new["prompts"][2].tolist() == [30, 31] and new["prompts"][3].tolist() == [PAD, 32]
    assert new["responses"][2].tolist() == [7, 8] and new["responses"][3].tolist() == [9, PAD]
    assert new["response_mask"][2:].tolist() == [[1, 1], [1, 0]]
    assert new["attention_mask"][2:].tolist() == [[1, 1, 1, 1], [0, 1, 1, 0]]
    assert torch.equal(new["input_ids"], torch.cat([new["prompts"], new["responses"]], dim=1))
    assert torch.equal(new["position_ids"], verl_position_ids(new["attention_mask"], 2))
    assert new["old_log_probs"][2:].tolist() == f32([[-0.7, -0.8], [-0.9, 0.0]])
    assert new["advantages"][2:].tolist() == [[0.5, 0.5], [-0.5, 0.0]]
    assert torch.equal(new["returns"], new["advantages"])
    assert new["token_level_scores"][2:].tolist() == [[0.0, 0.0], [1.0, 0.0]]
    assert torch.equal(new["token_level_rewards"], new["token_level_scores"])
    for key in ("prompts", "responses", "advantages"):  # fresh rows untouched
        assert torch.equal(new[key][:2], batch[key][:2])
    verify_written_rows(new, [2, 3], rollouts, [0.5, -0.5])
    check_layout(new)


def test_write_rows_pads_the_batch_when_a_stored_sequence_is_longer():
    batch, uids = two_groups()
    rollouts = [stored([7, 8, 9], [-0.7, -0.8, -0.9], [30, 31, 32, 33]), stored([9], [-0.9], [32])]

    new = write_rows(batch, [2, 3], rollouts, [0.5, -0.5], PAD)

    assert new["prompts"].shape == (4, 4) and new["responses"].shape == (4, 3)
    assert new["input_ids"].shape == (4, 7) and new["position_ids"].shape == (4, 7)
    # Fresh rows keep their content: prompts moved right, responses stay left.
    assert new["prompts"][0].tolist() == [PAD, PAD, 20, 21] and new["responses"][1].tolist() == [4, 4, PAD]
    assert torch.equal(new["position_ids"], verl_position_ids(new["attention_mask"], 4))
    assert new["advantages"][1].tolist() == [-0.25, -0.25, 0.0]
    verify_written_rows(new, [2, 3], rollouts, [0.5, -0.5])
    check_layout(new)


def test_pad_batch_is_a_no_op_at_the_current_widths_and_refuses_to_shrink():
    batch, _ = two_groups()
    same = pad_batch(batch, 2, 2, PAD)
    assert all(same[k] is batch[k] for k in batch)
    with pytest.raises(ValueError, match="shrink"):
        pad_batch(batch, 1, 2, PAD)


def test_write_rows_refuses_unknown_per_row_tensors():
    batch, _ = two_groups()
    batch["values"] = torch.zeros(4, 2)
    with pytest.raises(ValueError, match="values"):
        write_rows(batch, [2], [stored([7], [-0.7], [30])], [0.5], PAD)


def test_write_rows_requires_ref_logprobs_to_match_the_batch():
    batch, _ = two_groups(ref_logps=[[-1.0], [-2.0, -2.1], [-3.0, -3.1], [-4.0]])
    with pytest.raises(ValueError, match="KL setting"):
        write_rows(batch, [2], [stored([7], [-0.7], [30])], [0.5], PAD)
    new = write_rows(batch, [2], [stored([7], [-0.7], [30], ref=[-9.0])], [0.5], PAD)
    assert new["ref_log_prob"][2].tolist() == [-9.0, 0.0]


def test_write_rows_validates_rows_and_lengths():
    batch, _ = two_groups()
    r = stored([7], [-0.7], [30])
    with pytest.raises(ValueError, match="same length"):
        write_rows(batch, [2, 3], [r], [0.5], PAD)
    with pytest.raises(ValueError, match="outside"):
        write_rows(batch, [4], [r], [0.5], PAD)
    with pytest.raises(ValueError, match="duplicates"):
        write_rows(batch, [2, 2], [r, r], [0.5, 0.5], PAD)


@pytest.mark.parametrize("corrupt, message", [
    (lambda t: t["responses"].__setitem__((2, 1), 99), "response ids"),
    (lambda t: t["prompts"].__setitem__((2, 0), 99), "prompt ids"),
    (lambda t: t["old_log_probs"].__setitem__((2, 0), -0.1), "behavior logprobs"),
    (lambda t: t["advantages"].__setitem__((2, 1), 0.0), "advantages"),
    (lambda t: t["returns"].__setitem__((2, 0), 0.0), "returns"),
    (lambda t: t["position_ids"].__setitem__((2, 3), 7), "position_ids"),
    (lambda t: t["token_level_scores"].__setitem__((2, 0), 1.0), "token_level_scores"),
    (lambda t: t["rm_scores"].__setitem__((2, 1), 0.5), "rm_scores"),
    (lambda t: t["ref_log_prob"].__setitem__((2, 0), -1.0), "reference logprobs"),
])
def test_verify_written_rows_catches_a_corruption_in_every_rewritten_tensor(corrupt, message):
    batch, _ = two_groups(ref_logps=[[-1.0], [-2.0, -2.1], [-3.0, -3.1], [-4.0]])
    batch["rm_scores"] = batch["token_level_scores"].clone()
    rollouts = [stored([7, 8], [-0.7, -0.8], [30, 31], score=0.0, ref=[-9.0, -9.5])]
    new = write_rows(batch, [2], rollouts, [0.5], PAD)
    verify_written_rows(new, [2], rollouts, [0.5])
    corrupt(new)
    with pytest.raises(ValueError, match=message):
        verify_written_rows(new, [2], rollouts, [0.5])


def test_rm_scores_are_rewritten_with_the_other_score_tensors_and_must_match():
    batch, _ = two_groups()
    batch["rm_scores"] = batch["token_level_scores"].clone()
    new = write_rows(batch, [2], [stored([7, 8], [-0.7, -0.8], [30, 31], score=0.25)], [0.5], PAD)
    assert new["rm_scores"][2].tolist() == [0.0, 0.25] and torch.equal(new["rm_scores"], new["token_level_scores"])
    batch["rm_scores"][0, 0] = 9.0
    with pytest.raises(ValueError, match="rm_scores"):
        check_layout(batch)


def test_dummy_tensor_passes_through_unchanged():
    batch, _ = two_groups()
    batch["dummy_tensor"] = torch.zeros(4, 1, dtype=torch.uint8)
    new = write_rows(batch, [2], [stored([7], [-0.7], [30])], [0.5], PAD)
    assert new["dummy_tensor"] is batch["dummy_tensor"]


# ---------------------------------------------------------------------------
# Digest and meta
# ---------------------------------------------------------------------------

def test_tensor_digest_depends_on_consumed_values_only():
    a, _ = two_groups()
    b = copy.deepcopy(a)
    assert tensor_digest(a) == tensor_digest(b)
    b["old_log_probs"][0, 1] = -7.0  # outside the response mask of row 0
    b["position_ids"][2, 0] = 5      # a left-pad position of row 2
    assert tensor_digest(a) == tensor_digest(b)
    b["responses"][0, 0] += 1
    b["input_ids"] = torch.cat([b["prompts"], b["responses"]], dim=1)
    assert tensor_digest(a) != tensor_digest(b)
    c = copy.deepcopy(a)
    c["advantages"] = c["advantages"].to(torch.float64)
    assert tensor_digest(a) == tensor_digest(c)


def test_global_token_num_counts_valid_tokens_per_row():
    batch, _ = two_groups()
    assert global_token_num(batch["attention_mask"]) == [3, 4, 3, 2]
