"""Per-step diagnostics logged by every arm of the staleness sweep.

``SweepDiagnosticsMixin`` sits first in each arm's trainer class, so it sees
the batch exactly as the loss will (after replay, filtering or the
W-REINFORCE advantage), and appends to TRL's per-step metrics under
``sweep/``:

- ``dead_rate``, ``dead_all_wrong_rate``, ``dead_all_correct_rate``: groups
  generated this step whose rewards are all equal, split by the value they
  share (``grpo_filter`` counts every generation round);
- ``rollouts_generated``: completions generated this step;
- ``first_round_reward``: mean reward of the step's first generation round,
  the prompts every arm sees at that step (``grpo_filter``'s TRL ``reward``
  also averages its extra rounds, which run only after dead-heavy first
  rounds, so the report reads this series instead);
- ``replayed_row_fraction``: trained rows that are replayed rows;
- ``same_step_fraction``: replayed rows inserted at this step. The adapter's
  ``min_entry_age=1`` makes it 0; any other value raises and ends the run;
- ``replayed_age_mean``, ``_p90`` (nearest rank), ``_max``: ``step - entry
  version`` of the replayed rows;
- ``trained_completion_length``: mean completion tokens of the trained rows;
- ``clip_fraction/{fresh,replayed}_{pos,neg}``: tokens TRL's clipped
  objective clips, among the fresh or replayed rows with positive or
  negative advantage, measured on the loss's own forward. With
  ``num_iterations=1`` the control arms' fresh rows have no stored
  logprobs, so their ratio is exactly 1 and their clip fraction 0. A
  Reservoir arm attaches behavior logprobs to every row on the steps it
  replays, so there a fresh row's ratio is 1 only up to the numerics of two
  forwards; ``fresh_*`` is therefore not comparable across arms, and on
  the GPU it measures whether the adapter's forward and the loss's agree;
- ``engine_confident_fraction`` and ``engine_mean_abs_diff`` (control arms
  on vLLM): the stale-engine check of ``engine_probe.probe_stats`` on the
  fresh rows, comparing vLLM's sampling logprobs with the loss forward,
  whose weights are the ones the step sampled with. Reservoir arms are
  probed by ``ProbedReplay`` instead;
- ``kl_to_init``: the k3 estimate of KL(policy || initial model) on the
  fresh rows' completion tokens (replayed tokens were not sampled from the
  current policy, so the estimator does not apply to them), from the loss
  forward and one no-grad forward of a frozen copy of the initial model per
  generation step, in every arm;
- ``lm_head_grad_norm`` and ``lm_head_grad_z``: the accumulated output-
  embedding gradient norm before clipping and its z-score against the
  previous ``Z_WINDOW`` steps (``make_lm_head_monitor``).

TRL already logs the policy entropy (``entropy``) and the generated
completion length (``completions/mean_length``); the adapter logs the
sampling-design ESS (``reservoir/sampling_design_ess``).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Optional, Sequence

CORRECT_THRESHOLD = 0.5
Z_WINDOW = 20
Z_MIN_HISTORY = 5
PREFIX = "sweep/"


# --- pure statistics ---------------------------------------------------------------


def group_outcomes(rewards: Sequence[float], group_size: int, correct_threshold: float = CORRECT_THRESHOLD) -> dict:
    """Dead groups (all rewards exactly equal) among ``rewards``, split by the shared value."""
    groups = len(rewards) // group_size
    wrong = correct = 0
    for g in range(groups):
        values = [float(v) for v in rewards[g * group_size:(g + 1) * group_size]]
        if all(math.isfinite(v) for v in values) and len(set(values)) == 1:
            if values[0] >= correct_threshold:
                correct += 1
            else:
                wrong += 1
    return {"groups": groups, "dead": wrong + correct, "dead_all_wrong": wrong, "dead_all_correct": correct}


def age_stats(ages: Sequence[int]) -> dict:
    """Mean, nearest-rank 90th percentile and maximum; ``None`` for an empty list."""
    if not ages:
        return {"mean": None, "p90": None, "max": None}
    ordered = sorted(ages)
    return {"mean": sum(ordered) / len(ordered), "p90": ordered[math.ceil(0.9 * len(ordered)) - 1], "max": ordered[-1]}


def clip_fractions(coef: Any, advantages: Any, mask: Any, replayed: Any, *, eps_low: float,
                   eps_high: float) -> dict[str, Optional[float]]:
    """Fraction of completion tokens TRL's clipped objective clips, by row origin and advantage sign.

    The rule is TRL's: a token is clipped when its ratio is below
    ``1 - eps_low`` with a negative advantage or above ``1 + eps_high`` with
    a positive one. ``coef`` is ``(B, T)`` (or ``(B, 1)`` at sequence level).
    """
    adv = advantages.view(-1, 1)
    clipped = ((coef < 1 - eps_low) & (adv < 0)) | ((coef > 1 + eps_high) & (adv > 0))
    clipped = clipped.expand_as(mask).float() * mask
    tokens = mask.float()
    out: dict[str, Optional[float]] = {}
    for origin, rows in (("fresh", ~replayed), ("replayed", replayed)):
        for sign, selected in (("pos", advantages > 0), ("neg", advantages < 0)):
            pick = rows & selected
            count = float(tokens[pick].sum())
            out[f"{origin}_{sign}"] = float(clipped[pick].sum()) / count if count else None
    return out


def kl_k3(logp: Any, ref_logp: Any, mask: Any, rows: Any) -> Optional[float]:
    """Mean over the selected rows' completion tokens of ``exp(ref - logp) - (ref - logp) - 1``."""
    tokens = mask[rows].float()
    count = float(tokens.sum())
    if not count:
        return None
    import torch

    delta = (ref_logp[rows] - logp[rows]).double()
    k3 = torch.where(tokens.bool(), delta.exp() - delta - 1, torch.zeros_like(delta))
    return float(k3.sum()) / count


class RunningZ:
    """z-score of each new value against the previous ``window`` values (sample std)."""

    def __init__(self, window: int = Z_WINDOW, min_history: int = Z_MIN_HISTORY) -> None:
        self.history: deque[float] = deque(maxlen=window)
        self._min = min_history

    def push(self, value: float) -> Optional[float]:
        past = list(self.history)
        self.history.append(value)
        if len(past) < self._min:
            return None
        mean = sum(past) / len(past)
        std = math.sqrt(sum((v - mean) ** 2 for v in past) / (len(past) - 1))
        return (value - mean) / std if std > 0 else None


# --- the trainer mixin -----------------------------------------------------------------


def _log(trainer: Any, name: str, value: Optional[float]) -> None:
    if value is not None:
        trainer._metrics["train"][PREFIX + name].append(float(value))


class SweepDiagnosticsMixin:
    """Logs the per-step diagnostics above; first in the base list of every arm's trainer.

    ``sweep_ref_model`` is the frozen initial model (``None`` skips the KL);
    ``replay_buffer``, when the trainer has one, must be a
    ``ProbedReplay`` so the replayed rows of the step are known.
    """

    sweep_ref_model: Any = None
    _sweep_capturing = False
    _sweep_captured: Any = None

    def _calculate_rewards(self, *args, **kwargs):  # type: ignore[override]
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)  # type: ignore[misc]
        if self.model.training:  # type: ignore[attr-defined]
            weights = self.reward_weights.to(rewards_per_func.device)  # type: ignore[attr-defined]
            summed = (rewards_per_func * weights.unsqueeze(0)).nansum(dim=1)
            summed[rewards_per_func.isnan().all(dim=1)] = float("nan")
            if not self._sweep_step_rewards:
                finite = [v for v in summed.tolist() if math.isfinite(v)]
                _log(self, "first_round_reward", sum(finite) / len(finite) if finite else None)
            self._sweep_step_rewards.extend(summed.tolist())
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):  # type: ignore[override]
        if not self.model.training:  # type: ignore[attr-defined]
            return super()._generate_and_score_completions(inputs)  # type: ignore[misc]
        self._sweep_step_rewards: list[float] = []
        output = super()._generate_and_score_completions(inputs)  # type: ignore[misc]
        return self._sweep_after_generation(output)

    def _sweep_after_generation(self, output: dict) -> dict:
        import torch

        step = int(self.state.global_step)  # type: ignore[attr-defined]
        rows = output["advantages"].size(0)
        outcome = group_outcomes(self._sweep_step_rewards, int(self.num_generations))  # type: ignore[attr-defined]
        if outcome["groups"]:
            _log(self, "dead_rate", outcome["dead"] / outcome["groups"])
            _log(self, "dead_all_wrong_rate", outcome["dead_all_wrong"] / outcome["groups"])
            _log(self, "dead_all_correct_rate", outcome["dead_all_correct"] / outcome["groups"])
        _log(self, "rollouts_generated", len(self._sweep_step_rewards))
        replayed_rows, versions = getattr(getattr(self, "replay_buffer", None), "last_replayed", ((), ()))
        _log(self, "replayed_row_fraction", len(replayed_rows) / rows)
        if replayed_rows:
            same = sum(1 for v in versions if v >= step)
            _log(self, "same_step_fraction", same / len(replayed_rows))
            if same:
                raise RuntimeError(f"step {step}: {same} of {len(replayed_rows)} replayed rows were inserted at this "
                                   "step; the run is refused (min_entry_age must keep them out)")
            for name, value in age_stats([step - v for v in versions]).items():
                _log(self, f"replayed_age_{name}", value)
        mask = output["completion_mask"]
        _log(self, "trained_completion_length", float(mask.sum()) / rows)
        flags = torch.zeros(rows, dtype=torch.bool, device=mask.device)
        flags[list(replayed_rows)] = True
        extra = {"sweep_replayed": flags}
        if self.sweep_ref_model is not None:
            extra["sweep_ref_logps"] = self._sweep_ref_logps(output)
        return {**output, **extra}

    def _sweep_ref_logps(self, output: dict) -> Any:
        import torch

        input_ids = torch.cat([output["prompt_ids"], output["completion_ids"]], dim=1)
        attention_mask = torch.cat([output["prompt_mask"], output["completion_mask"]], dim=1)
        # Under the accelerator's autocast, as the loss forward it is compared with runs.
        with torch.no_grad(), self.accelerator.autocast():  # type: ignore[attr-defined]
            logps, _, _ = super()._get_per_token_logps_and_entropies(  # type: ignore[misc]
                self.sweep_ref_model, input_ids, attention_mask, output["completion_ids"].size(1),
                batch_size=self.args.per_device_train_batch_size,  # type: ignore[attr-defined]
            )
        return logps

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):  # type: ignore[override]
        result = super()._get_per_token_logps_and_entropies(*args, **kwargs)  # type: ignore[misc]
        if self._sweep_capturing and self._sweep_captured is None:
            self._sweep_captured = result[0].detach()
        return result

    def _compute_loss(self, model, inputs):  # type: ignore[override]
        self._sweep_captured, self._sweep_capturing = None, True
        try:
            loss = super()._compute_loss(model, inputs)  # type: ignore[misc]
        finally:
            self._sweep_capturing = False
        if self.model.training and self._sweep_captured is not None and "sweep_replayed" in inputs:  # type: ignore[attr-defined]
            self._sweep_loss_metrics(inputs, self._sweep_captured)
        self._sweep_captured = None
        return loss

    def _sweep_loss_metrics(self, inputs: dict, logp: Any) -> None:
        import torch

        mask = inputs["completion_mask"]
        old = inputs.get("old_per_token_logps")
        log_ratio = logp - (logp if old is None else old)
        if getattr(self, "importance_sampling_level", "token") == "sequence":
            log_ratio = ((log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).unsqueeze(-1)
        replayed = inputs["sweep_replayed"]
        fractions = clip_fractions(torch.exp(log_ratio), inputs["advantages"], mask, replayed,
                                   eps_low=self.epsilon_low, eps_high=self.epsilon_high)  # type: ignore[attr-defined]
        for name, value in fractions.items():
            _log(self, f"clip_fraction/{name}", value)
        ref = inputs.get("sweep_ref_logps")
        if ref is not None:
            _log(self, "kl_to_init", kl_k3(logp, ref, mask, ~replayed))
        sampling = inputs.get("sampling_per_token_logps")
        if sampling is not None and getattr(self, "replay_buffer", None) is None:
            from benchmarks.staleness.engine_probe import probe_stats

            fresh = (~replayed).view(-1, 1).expand_as(mask)
            stats = probe_stats(sampling, logp, mask.bool() & fresh)
            _log(self, "engine_confident_fraction", stats["confident_fraction"])
            _log(self, "engine_mean_abs_diff", stats["mean_abs_diff"])


def make_lm_head_monitor(trainer: Any, window: int = Z_WINDOW, min_history: int = Z_MIN_HISTORY) -> Any:
    """A ``TrainerCallback`` logging the output-embedding gradient norm before clipping, and its z-score.

    A post-accumulate hook on the weight keeps the norm of the gradient
    accumulated so far; the last value before the optimizer step is the
    step's full gradient, read in ``on_pre_optimizer_step`` (which TRL runs
    after clipping, hence the hook). Qwen2.5-0.5B ties the embedding and
    ``lm_head``, so the norm covers both uses of the matrix.
    """
    from transformers import TrainerCallback

    class LmHeadGradMonitor(TrainerCallback):
        def __init__(self) -> None:
            self._z = RunningZ(window, min_history)
            self._last: Optional[float] = None
            self._handle = None

        def _hook(self, param) -> None:
            self._last = float(param.grad.detach().float().norm())

        def on_train_begin(self, args, state, control, **kwargs):
            weight = trainer.model.get_output_embeddings().weight
            self._handle = weight.register_post_accumulate_grad_hook(self._hook)
            return control

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if self._last is not None:
                _log(trainer, "lm_head_grad_norm", self._last)
                _log(trainer, "lm_head_grad_z", self._z.push(self._last))
                self._last = None
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if self._handle is not None:
                self._handle.remove()
            return control

    return LmHeadGradMonitor()
