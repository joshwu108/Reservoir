"""
reservoir.integrations._trl_telemetry — Replay health per generation step, and the drift gate.

What an engineer watching a replayed GRPO run wants to see, every step:

- how much of the batch was replayed (``reservoir/replay_fraction``), how
  many groups were dead or near-dead;
- the effective sample size of the replayed rows (``reservoir/ess``, from
  the exact importance weights) and its fraction of the replayed count;
- how stale the replayed rows are (``reservoir/staleness_mean`` and
  ``_max``, in optimizer steps behind);
- how far the stored behavior logprobs have drifted from the current
  policy on the replayed tokens (``reservoir/log_ratio_mean_abs`` and
  ``_max_abs``, per-sequence sums of ``current - behavior``). This needs
  one no-grad forward over the replayed rows only.

The numbers go to TRL's metrics (so wandb and TensorBoard show them) and
into a ``telemetry`` record of the attestation log, where the checker
recomputes the ESS and the staleness and carries the log-ratios as
reported values.

The drift gate is off by default. With ``max_log_ratio`` set, a replayed
row whose absolute sequence log-ratio exceeds it (or is not finite) is
declined: the dead row it would have filled stays dead, the buffer entry is
evicted with reason ``"drift"`` unless another draw of the same slot was
kept, and the batch witness lists the declined draw. Nothing about a
decline is silent, and more than ``max_declines_per_step`` declines in one
step raise instead of continuing, because a gate that quietly drops the
only informative rows would hide a collapsed reward signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

from reservoir.rollout_buffer import RolloutBatch

METRIC_PREFIX = "reservoir/"


@dataclass(frozen=True)
class StepTelemetry:
    """Everything measured about one generation step."""

    batch_rows: int
    replaced_rows: int
    declined_rows: int
    dead_groups: int
    near_dead_groups: int
    ess: Optional[float] = None
    staleness_mean: Optional[float] = None
    staleness_max: Optional[int] = None
    log_ratio_mean_abs: Optional[float] = None
    log_ratio_max_abs: Optional[float] = None

    def counts(self) -> dict:
        return {
            "batch_rows": self.batch_rows, "replaced_rows": self.replaced_rows,
            "declined_rows": self.declined_rows, "dead_groups": self.dead_groups,
            "near_dead_groups": self.near_dead_groups,
        }

    def reported(self) -> dict:
        """The float measurements the checker can only carry."""
        out = {}
        if self.log_ratio_mean_abs is not None:
            out["log_ratio_mean_abs"] = self.log_ratio_mean_abs
        if self.log_ratio_max_abs is not None:
            out["log_ratio_max_abs"] = self.log_ratio_max_abs
        return out

    def metrics(self) -> dict[str, float]:
        """Flat ``reservoir/*`` metrics for the trainer's logger."""
        out: dict[str, float] = {
            "replay_fraction": self.replaced_rows / self.batch_rows if self.batch_rows else 0.0,
            "replaced_rows": float(self.replaced_rows),
            "declined_rows": float(self.declined_rows),
            "dead_groups": float(self.dead_groups),
            "near_dead_groups": float(self.near_dead_groups),
        }
        for name in ("ess", "staleness_mean", "staleness_max", "log_ratio_mean_abs", "log_ratio_max_abs"):
            value = getattr(self, name)
            if value is not None:
                out[name] = float(value)
        if self.ess is not None and self.replaced_rows:
            out["ess_fraction"] = self.ess / self.replaced_rows
        return {METRIC_PREFIX + k: v for k, v in out.items()}


def sequence_log_ratios(output: dict, rows: list[int], trainer: Any) -> list[float]:
    """``sum over masked tokens of (current - behavior)`` for each of ``rows``; one no-grad forward."""
    if not rows:
        return []
    idx = torch.tensor(rows, device=output["completion_ids"].device)
    input_ids = torch.cat([output["prompt_ids"][idx], output["completion_ids"][idx]], dim=1)
    attention_mask = torch.cat([output["prompt_mask"][idx], output["completion_mask"][idx]], dim=1)
    logits_to_keep = output["completion_ids"].size(1)
    with torch.no_grad():
        current, _, _ = trainer._get_per_token_logps_and_entropies(
            trainer.model, input_ids, attention_mask, logits_to_keep,
            batch_size=trainer.args.per_device_train_batch_size,
        )
    behavior = output["old_per_token_logps"][idx]
    mask = output["completion_mask"][idx].bool()
    # where(), not multiply: a NaN at a masked position would otherwise poison the whole row.
    diff = current.to(behavior.dtype) - behavior
    ratios = torch.where(mask, diff, torch.zeros_like(diff)).sum(dim=1)
    return [float(x) for x in ratios.detach().cpu().tolist()]


def choose_declines(ratios: list[float], max_log_ratio: Optional[float], max_declines: Optional[int]) -> list[int]:
    """Draw positions whose |log-ratio| exceeds the gate; raises if too many would be declined."""
    if max_log_ratio is None:
        return []
    # A non-finite ratio (NaN or inf from a degenerate forward) is the worst
    # case and is declined; it must never slip through a comparison with NaN.
    declined = [k for k, r in enumerate(ratios) if not math.isfinite(r) or abs(r) > max_log_ratio]
    if max_declines is not None and len(declined) > max_declines:
        raise RuntimeError(
            f"the drift gate would decline {len(declined)} of {len(ratios)} replayed rows this step, above "
            f"max_declines_per_step={max_declines}; the buffer may have gone stale or the policy moved far"
        )
    return declined


def summarize(batch: Optional[RolloutBatch], current_version: int, ratios: list[float], kept: list[int],
              batch_rows: int, dead_groups: int, near_dead_groups: int, declined: int) -> StepTelemetry:
    """Fold a step's measurements into a ``StepTelemetry``.

    The effective sample size, the staleness and the log-ratio statistics
    describe the whole sampled batch, declined draws included: that is
    what the sampler produced and what the checker recomputes from the
    sample record. ``replaced_rows`` counts the draws that were placed.
    """
    if batch is None:
        return StepTelemetry(batch_rows, 0, declined, dead_groups, near_dead_groups)
    weights = [float(w) for w in batch.is_weights]
    ess = sum(weights) ** 2 / sum(w * w for w in weights)
    ages = [current_version - v for v in batch.model_versions]
    # Statistics over the finite ratios only; a non-finite one is declined by
    # the gate (when on) and would make the record unrepresentable.
    magnitudes = [abs(r) for r in ratios if math.isfinite(r)]
    return StepTelemetry(
        batch_rows=batch_rows, replaced_rows=len(kept), declined_rows=declined,
        dead_groups=dead_groups, near_dead_groups=near_dead_groups,
        ess=ess, staleness_mean=sum(ages) / len(ages), staleness_max=max(ages),
        log_ratio_mean_abs=(sum(magnitudes) / len(magnitudes)) if magnitudes else None,
        log_ratio_max_abs=max(magnitudes) if magnitudes else None,
    )


def log_metrics(trainer: Any, telemetry: StepTelemetry) -> None:
    """Append the step's metrics to TRL's per-mode metric lists, if the trainer has them."""
    metrics = getattr(trainer, "_metrics", None)
    if not isinstance(metrics, dict) or "train" not in metrics:
        return
    for name, value in telemetry.metrics().items():
        metrics["train"][name].append(value)
