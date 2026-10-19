"""Replayed-row provenance for a token flight recorder, without importing one.

A flight recorder such as Martingale (``martingale.integrations.trl.
MartingaleRecorder``) records every fresh sequence a GRPO step generates
and gives each batch row an id in the ``martingale_row_id`` column. When
``ReservoirReplay`` writes a replayed rollout into a dead row, that row no
longer holds the sequence the recorder saw; this module tells the
recorder what is there instead, so its loss path scores the row as a
replay of an older generation rather than mis-attributing it.

Compose the trainers so the recorder sees the fresh batch first::

    class Trainer(ReservoirReplayMixin, MartingaleGRPOMixin, GRPOTrainer):
        ...

and pass ``ReservoirReplay(replay_recorder=flight)`` (or set
``trainer.flight_recorder``, which is where Martingale's trainer keeps it).
For every kept draw ``ReservoirReplay._replay`` then calls::

    recorder.register_replayed(
        trainer, prompt_ids=..., completion_ids=..., behavior_step=...,
        behavior_logprobs=..., draw_id="<sample op_counter>/<draw>",
        content_digest=..., is_weight=Fraction, rescale=Fraction,
        buffer_age=int, advantage=float) -> int

and writes the returned id (``-1`` when the recorder cannot identify the
origin, which its loss path skips) into the row. ``is_weight`` is the
draw's exact importance weight, normalised by the batch maximum as
``RolloutBatch.is_weights`` is; ``rescale`` is the staleness policy's exact
group-mass scale (``Fraction(1)`` when the policy is off or did not bind),
so ``is_weight * rescale`` is the factor the row's advantage was
multiplied by. ``advantage`` is the float written to the row. The call is
made after the step's witness, telemetry and evictions are logged, so a
recorder that raises leaves a complete attestation log behind.

Nothing here imports Martingale: any object with ``register_replayed`` is
accepted. Only the owner rank replays, so under more than one process the
owner's recorder registers every row and the ids travel with the
scattered slices (Martingale's row ids name their rank).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import torch

from reservoir.integrations._trl_staleness import RowDecision
from reservoir.rollout_buffer import RolloutBatch

ROW_ID_KEY = "martingale_row_id"
"""The per-row id column a flight recorder adds to TRL's generation output."""


@runtime_checkable
class ReplayRecorder(Protocol):
    """What ``ReservoirReplay`` needs from a flight recorder."""

    def register_replayed(self, trainer: Any, *, draw_id: str, content_digest: str, is_weight: Any,
                          rescale: Any = 1, advantage: Optional[float] = None, prompt_ids: Any = None,
                          completion_ids: Any = None, behavior_step: Optional[int] = None,
                          behavior_logprobs: Any = None, buffer_age: Optional[int] = None) -> int: ...


def resolve_recorder(explicit: Optional[ReplayRecorder], trainer: Any) -> Optional[ReplayRecorder]:
    """The recorder given to ``ReservoirReplay``, else ``trainer.flight_recorder``, else None."""
    recorder = explicit if explicit is not None else getattr(trainer, "flight_recorder", None)
    if recorder is not None and not callable(getattr(recorder, "register_replayed", None)):
        raise TypeError(f"replay recorder {type(recorder).__name__} has no register_replayed method")
    return recorder


def draw_scales(decisions: Sequence[RowDecision], n: int) -> list[Fraction]:
    """The staleness policy's exact scale for each of ``n`` draws; all 1 when the policy is off."""
    if not decisions:
        return [Fraction(1)] * n
    scales = [Fraction(1)] * n
    for d in decisions:
        scales[d.draw] = d.scale
    return scales


def register_replayed_rows(recorder: ReplayRecorder, trainer: Any, output: dict, batch: RolloutBatch,
                           rows: Sequence[int], kept: Sequence[int], decisions: Sequence[RowDecision],
                           weighted: Sequence[float], step: int) -> dict:
    """Register every kept draw with ``recorder``; return ``output`` with their row ids written.

    ``rows[i]`` is the batch row draw ``kept[i]`` filled and ``weighted[i]``
    the advantage written there. Refuses a batch without the recorder's
    ``martingale_row_id`` column: the ids would have nowhere to go and the
    recorder would score the replayed rows as the fresh ones they replaced.
    """
    if ROW_ID_KEY not in output:
        raise ValueError(
            f"a replay recorder is attached but the batch has no {ROW_ID_KEY!r} column; compose the "
            "recorder's trainer mixin after ReservoirReplayMixin so it records the fresh batch first"
        )
    if not (len(rows) == len(kept) == len(weighted)):
        raise ValueError("register_replayed_rows: rows, kept and weighted must have the same length")
    scales = draw_scales(decisions, len(batch.indices))
    ids = torch.as_tensor(output[ROW_ID_KEY]).clone()
    for row, k, advantage in zip(rows, kept, weighted):
        rollout = batch.rollouts[k]
        ids[row] = int(recorder.register_replayed(
            trainer,
            prompt_ids=list(rollout.metadata["prompt_ids"]),
            completion_ids=list(rollout.tokens),
            behavior_step=int(batch.model_versions[k]),
            behavior_logprobs=list(rollout.logprobs),
            draw_id=f"{batch.op_counter}/{k}",
            content_digest=batch.content_digests[k],
            is_weight=batch.is_weights[k],
            rescale=scales[k],
            buffer_age=int(step) - int(batch.model_versions[k]),
            advantage=float(advantage),
        ))
    return {**output, ROW_ID_KEY: ids}
