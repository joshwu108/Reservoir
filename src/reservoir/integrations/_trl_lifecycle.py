"""
reservoir.integrations._trl_lifecycle — Which process owns the buffer, and binding it to checkpoints.

Two things a trainer does outside the hook: decide which rank holds the
buffer and the log (``env_rank``: what the launcher said before any code
ran), and tie a durable buffer to the trainer's own checkpoints so a resumed
run rewinds the buffer to the step the model restarts from
(``bind_checkpoint``, ``resume_from_checkpoint``). All of it is a no-op on a
rank that does not own the buffer.

Rank from the environment: ``RANK`` is the global rank torchrun and
``accelerate launch`` set. ``LOCAL_RANK`` alone identifies only the rank
within one node, so a non-zero ``LOCAL_RANK`` means "not rank 0" but a zero
one, without ``RANK``, decides nothing (on a second node it is not the
owner); the rank is then fixed by ``ReservoirReplay.attach`` from the
process group before any file is touched.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Optional

from reservoir.durable_rollout import DurableRolloutBuffer

RANK_ENV_VARS: Final[tuple[str, ...]] = ("RANK", "LOCAL_RANK")
"""Set by torchrun / accelerate launch before user code runs; a non-zero value means "not rank 0"."""


def env_rank() -> Optional[int]:
    """The launcher's rank for this process, or None when no launcher set one."""
    values = {}
    for name in RANK_ENV_VARS:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            values[name] = int(raw)
        except ValueError as exc:
            raise RuntimeError(f"{name}={raw!r} in the environment is not an integer") from exc
    if "RANK" in values:
        return values["RANK"]
    local = values.get("LOCAL_RANK")
    return local if local else None


def checkpoint_tag(global_step: int) -> str:
    return f"step-{int(global_step)}"


def trainer_checkpoint_steps(output_dir) -> Optional[set[int]]:
    """Steps of the ``checkpoint-N`` directories under ``output_dir``, or None if it cannot be listed."""
    if output_dir is None:
        return None
    root = Path(output_dir)
    if not root.is_dir():
        return None
    steps = set()
    for entry in root.iterdir():
        name = entry.name
        if entry.is_dir() and name.startswith("checkpoint-") and name[len("checkpoint-"):].isdigit():
            steps.add(int(name[len("checkpoint-"):]))
    return steps


def bind_checkpoint(replay: ReservoirReplay, global_step: int, output_dir=None) -> None:
    """When the trainer saves a checkpoint, snapshot the durable buffer under the same step.

    With ``output_dir`` the buffer's checkpoints are pruned to the steps the
    trainer still has (it rotates its own under ``save_total_limit`` before
    this is called), so the buffer directory does not grow without bound.
    A buffer without a directory has nothing to bind; the call is a no-op,
    as it is on every rank but the owner.
    """
    if not replay.is_owner or not isinstance(replay.buffer, DurableRolloutBuffer):
        return
    replay.buffer.checkpoint(checkpoint_tag(global_step))
    steps = trainer_checkpoint_steps(output_dir)
    if steps is not None:
        replay.buffer.prune_checkpoints({checkpoint_tag(n) for n in steps | {global_step}})


def resume_from_checkpoint(replay: ReservoirReplay, global_step: int) -> None:
    """When training (re)starts at ``global_step > 0``, rewind the buffer to that step's snapshot.

    Without the rewind the buffer would carry every operation the crashed
    run logged after the checkpoint while the model restarts before them.
    A durable buffer with no snapshot for the step is an error, so a resume
    cannot silently continue from the wrong point; a non-durable buffer
    cannot resume and raises if asked to. Off the owner rank there is no
    buffer to rewind and the call is a no-op.
    """
    if global_step <= 0 or not replay.is_owner:
        return
    tag = checkpoint_tag(global_step)
    if not isinstance(replay.buffer, DurableRolloutBuffer):
        raise RuntimeError(
            f"resuming at step {global_step} needs a durable buffer: pass ReservoirReplay(directory=...)"
        )
    if tag not in replay.buffer.checkpoints():
        raise RuntimeError(
            f"resuming at step {global_step} but the buffer has no checkpoint {tag!r} "
            f"(available: {replay.buffer.checkpoints()}); the buffer directory does not belong to this run"
        )
    replay.buffer.restore_checkpoint(tag)


__all__ = [
    "RANK_ENV_VARS",
    "bind_checkpoint",
    "checkpoint_tag",
    "env_rank",
    "resume_from_checkpoint",
    "trainer_checkpoint_steps",
]
