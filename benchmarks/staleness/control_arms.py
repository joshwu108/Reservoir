"""The sweep's two control trainers: DAPO dynamic sampling and W-REINFORCE.

Both are ``GRPOTrainer`` mixins that change one thing and keep TRL's
generation, reward, loss and logging.

``DynamicSamplingMixin`` (arm ``grpo_filter``) is DAPO's dynamic sampling
(Yu et al. 2025): after TRL scores a generation batch, groups whose rewards
are all equal are dropped and fresh prompts are generated and scored until
the batch holds as many live groups as it started with, or
``MAX_GENERATION_ROUNDS`` rounds have run. The first round is the same
prompts every other arm sees at that step (TRL's seeded sampler); extra
prompts come from a separate seeded stream over the same training set, so
the main stream stays aligned across arms. A batch still short after the
last round is filled with the first round's dead groups (zero advantage, as
in plain GRPO) and the shortfall is logged. Every extra round generates a
full batch of prompts and the surplus live groups are discarded, so the
arm's rollout count is an upper bound on what DAPO would spend. A group
with an unscorable (NaN) reward is treated as dead. Each round's rows were scored
against their own group, so concatenating them leaves every advantage as
TRL computed it; ``num_items_in_batch`` is recounted from the final mask.

``WReinforceMixin`` (arm ``wreinforce``) replaces the advantage with a fixed
reference (Zhu et al. 2025, "The Surprising Effectiveness of Negative
Reinforcement"): +0.1 for a correct completion, -1 for a wrong one, no group
baseline and no buffer. Only the ``advantages`` entry of TRL's output is
replaced.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

MAX_GENERATION_ROUNDS = 4
"""Generation rounds per step for ``grpo_filter``, the first included: at most 4x the plain arm's rollouts."""
WREINFORCE_POSITIVE = 0.1
WREINFORCE_NEGATIVE = -1.0
CORRECT_THRESHOLD = 0.5
"""A completion is correct when its summed reward is at least this (both sweep rewards are 0 or 1)."""

LEFT_PADDED = {"prompt_ids": "pad", "prompt_mask": 0}
RIGHT_PADDED = {"completion_ids": "pad", "completion_mask": 0, "old_per_token_logps": 0.0,
                "sampling_per_token_logps": 0.0, "ref_per_token_logps": 0.0}
PER_ROW = {"advantages"}
RECOUNTED = {"num_items_in_batch"}


def wreinforce_advantages(rewards: Any, positive: float = WREINFORCE_POSITIVE, negative: float = WREINFORCE_NEGATIVE,
                          threshold: float = CORRECT_THRESHOLD) -> Any:
    """Per-row W-REINFORCE advantage; an unscorable (NaN) row gets 0, as TRL gives it."""
    import torch

    adv = torch.where(rewards >= threshold, torch.full_like(rewards, positive), torch.full_like(rewards, negative))
    return torch.where(torch.isnan(rewards), torch.zeros_like(rewards), adv)


def live_groups(rewards: Any, group_size: int) -> list[int]:
    """Indices of the groups whose rewards are not all exactly equal (DAPO keeps these)."""
    import torch

    grouped = rewards.view(-1, group_size)
    return [g for g in range(grouped.size(0))
            if bool(torch.isfinite(grouped[g]).all()) and not bool((grouped[g] == grouped[g][0]).all())]


def _check_keys(output: dict) -> None:
    unknown = set(output) - set(LEFT_PADDED) - set(RIGHT_PADDED) - PER_ROW - RECOUNTED
    if unknown:
        raise ValueError(f"dynamic sampling cannot merge output keys {sorted(unknown)}")


def take_rows(output: dict, rows: Sequence[int]) -> dict:
    """The output restricted to ``rows`` (``num_items_in_batch`` is left for ``concat_outputs`` to recount)."""
    _check_keys(output)
    index = list(rows)
    return {k: (v if k in RECOUNTED else v[index]) for k, v in output.items()}


def _pad(tensor: Any, width: int, value: Any, left: bool) -> Any:
    import torch.nn.functional as F

    missing = width - tensor.size(1)
    if missing == 0:
        return tensor
    return F.pad(tensor, (missing, 0) if left else (0, missing), value=value)


def concat_outputs(outputs: Sequence[dict], pad_id: int) -> dict:
    """Stack TRL generation outputs scored in separate calls into one batch.

    Prompts are left-padded and completions right-padded to the widest
    member; every output must carry the same keys.
    """
    import torch

    for out in outputs:
        _check_keys(out)
    keys = set(outputs[0])
    for out in outputs[1:]:
        if set(out) != keys:
            raise ValueError(f"outputs differ in keys: {sorted(keys ^ set(out))}")
    merged: dict = {}
    for key in keys - RECOUNTED:
        parts = [out[key] for out in outputs]
        if key in PER_ROW:
            merged[key] = torch.cat(parts)
            continue
        left = key in LEFT_PADDED
        fill = (LEFT_PADDED if left else RIGHT_PADDED)[key]
        width = max(p.size(1) for p in parts)
        merged[key] = torch.cat([_pad(p, width, pad_id if fill == "pad" else fill, left) for p in parts])
    merged["num_items_in_batch"] = merged["completion_mask"].sum()
    return merged


@dataclass(frozen=True)
class FillPlan:
    taken: list[tuple[int, int]]     # (round, group) in batch order
    unfilled: int                    # groups filled from the first round's dead groups


def fill_plan(needed: int, live_by_round: Sequence[Sequence[int]], primary_groups: int) -> FillPlan:
    """Live groups in generation order up to ``needed``; any shortfall from round 0's dead groups."""
    taken = [(r, g) for r, groups in enumerate(live_by_round) for g in groups][:needed]
    short = needed - len(taken)
    if short:
        used = {g for r, g in taken if r == 0}
        taken += [(0, g) for g in range(primary_groups) if g not in used][:short]
    return FillPlan(taken, short)


class OversampleStream:
    """Extra prompts for dynamic sampling: a seeded draw over the training set, each repeated per generation."""

    def __init__(self, dataset: Any, seed: int, num_generations: int) -> None:
        self._dataset = dataset
        self._rng = random.Random(f"oversample-{seed}")
        self._g = num_generations

    def next_inputs(self, prompts: int) -> list:
        rows = [self._dataset[self._rng.randrange(len(self._dataset))] for _ in range(prompts)]
        return [row for row in rows for _ in range(self._g)]


def _summed_rewards(trainer: Any, rewards_per_func: Any) -> Any:
    import torch

    weights = trainer.reward_weights.to(rewards_per_func.device)
    rewards = (rewards_per_func * weights.unsqueeze(0)).nansum(dim=1)
    return torch.where(torch.isnan(rewards_per_func).all(dim=1), torch.full_like(rewards, float("nan")), rewards)


class _RewardTap:
    """Keeps the summed reward of the last ``_calculate_rewards`` call (single process: the whole batch)."""

    _last_rewards: Any = None

    def _calculate_rewards(self, *args, **kwargs):  # type: ignore[override]
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)  # type: ignore[misc]
        self._last_rewards = _summed_rewards(self, rewards_per_func)
        return rewards_per_func


class WReinforceMixin(_RewardTap):
    """Arm ``wreinforce``: TRL's output with the advantage replaced by the fixed reference."""

    def _generate_and_score_completions(self, inputs):  # type: ignore[override]
        output = super()._generate_and_score_completions(inputs)  # type: ignore[misc]
        if not self.model.training:  # type: ignore[attr-defined]
            return output
        if int(self.accelerator.num_processes) != 1:  # type: ignore[attr-defined]
            raise RuntimeError("the wreinforce arm is single-process")
        return {**output, "advantages": wreinforce_advantages(self._last_rewards).to(output["advantages"].dtype)}


class DynamicSamplingMixin(_RewardTap):
    """Arm ``grpo_filter``: DAPO dynamic sampling at a fixed effective batch."""

    sweep_oversample: OversampleStream

    def _generate_and_score_completions(self, inputs):  # type: ignore[override]
        generate = super()._generate_and_score_completions  # type: ignore[misc]
        if not self.model.training:  # type: ignore[attr-defined]
            return generate(inputs)
        if int(self.accelerator.num_processes) != 1:  # type: ignore[attr-defined]
            raise RuntimeError("the grpo_filter arm is single-process")
        g = int(self.num_generations)  # type: ignore[attr-defined]
        needed = len(inputs) // g
        outputs, live = [], []
        for round_ in range(MAX_GENERATION_ROUNDS):
            batch = inputs if round_ == 0 else self.sweep_oversample.next_inputs(needed)
            outputs.append(generate(batch))
            live.append(live_groups(self._last_rewards, g))
            if sum(map(len, live)) >= needed:
                break
        plan = fill_plan(needed, live, needed)
        parts = [take_rows(outputs[r], range(grp * g, (grp + 1) * g)) for r, grp in plan.taken]
        merged = concat_outputs(parts, pad_id=self._tokenizer.pad_token_id)  # type: ignore[attr-defined]
        metrics = self._metrics["train"]  # type: ignore[attr-defined]
        metrics["sweep/filter_rounds"].append(float(len(outputs)))
        metrics["sweep/filter_unfilled_groups"].append(float(plan.unfilled))
        return merged
