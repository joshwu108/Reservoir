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

import hashlib
import os
import warnings
from pathlib import Path
from typing import Final, Optional

from reservoir.durable_rollout import DurableRolloutBuffer

RANK_ENV_VARS: Final[tuple[str, ...]] = ("RANK", "LOCAL_RANK")
"""Set by torchrun / accelerate launch before user code runs; a non-zero value means "not rank 0"."""

MODEL_CHECKPOINT_KEY: Final[str] = "model_checkpoint"
"""Key of the model-checkpoint digest in a buffer checkpoint's binding."""
_DIGEST_PERSON: Final[bytes] = b"ckpt-dir\x00\x00\x00\x00\x00\x00\x00\x00"
_READ_CHUNK: Final[int] = 1 << 20
TRAINER_CHECKPOINT_PREFIX: Final[str] = "checkpoint-"
"""The HF Trainer's checkpoint directory name under ``output_dir``."""


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
    prefix = TRAINER_CHECKPOINT_PREFIX
    for entry in root.iterdir():
        name = entry.name
        if entry.is_dir() and name.startswith(prefix) and name[len(prefix):].isdigit():
            steps.add(int(name[len(prefix):]))
    return steps


def trainer_checkpoint_dir(output_dir, global_step: int) -> Optional[Path]:
    """``output_dir/checkpoint-N`` when ``output_dir`` is known, else None (it need not exist)."""
    if output_dir is None:
        return None
    return Path(output_dir) / f"{TRAINER_CHECKPOINT_PREFIX}{int(global_step)}"


# -- the model checkpoint's digest -----------------------------------------------

def directory_digest(path) -> dict:
    """BLAKE2b-256 over every regular file under ``path``: relative path, size and bytes, in path order.

    Symbolic links to files are read through; symbolic links to
    directories, broken links and anything that is not a regular file are
    skipped without notice. Returns ``{"digest", "files", "bytes"}``. Two
    directories with the same files, names and contents digest the same
    wherever they sit; a changed byte, name or file count changes the
    digest. The cost is one read of the directory (optimizer state
    included), so for a large model checkpoint it is the cost of copying
    it once, paid synchronously on the owner rank at every save and
    resume. A file that cannot be read raises ``OSError`` naming it.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    h = hashlib.blake2b(digest_size=32, person=_DIGEST_PERSON)
    files = total = 0
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = file.relative_to(root).as_posix()
        try:
            size = file.stat().st_size
            with open(file, "rb") as f:
                h.update(relative.encode("utf-8") + b"\0" + str(size).encode("ascii") + b"\0")
                while chunk := f.read(_READ_CHUNK):
                    h.update(chunk)
                    total += len(chunk)
        except OSError as exc:
            raise OSError(f"cannot digest {file} inside the model checkpoint {root}: {exc}") from exc
        h.update(b"\0")
        files += 1
    return {"digest": h.hexdigest(), "files": files, "bytes": total}


def model_binding(model_checkpoint) -> Optional[dict]:
    """The binding to record with a buffer checkpoint: the model checkpoint's digest, or None without one."""
    if model_checkpoint is None or not Path(model_checkpoint).is_dir():
        return None
    digest = directory_digest(model_checkpoint)
    return {MODEL_CHECKPOINT_KEY: {"name": Path(model_checkpoint).name, **digest}}


def check_model_binding(binding: Optional[dict], model_checkpoint, global_step: int) -> None:
    """Refuse a resume whose model checkpoint is not the one the buffer checkpoint was bound to.

    A buffer checkpoint recorded without a model digest (no trainer
    checkpoint directory existed when it was taken, or an older buffer)
    carries nothing to check. One that recorded a digest is honoured only
    against a directory whose digest equals it; a missing directory cannot
    be verified and is refused too.
    """
    if not binding or MODEL_CHECKPOINT_KEY not in binding:
        return
    recorded = binding[MODEL_CHECKPOINT_KEY]
    bound_to = f"model checkpoint {recorded.get('name')!r} (digest {str(recorded.get('digest'))[:16]}…)"
    if model_checkpoint is None:
        raise RuntimeError(
            f"resuming at step {global_step}: the buffer checkpoint is bound to {bound_to} but no directory to "
            "check was given; set ReservoirReplay.model_checkpoint to the directory the model restarts from"
        )
    if not Path(model_checkpoint).is_dir():
        raise RuntimeError(
            f"resuming at step {global_step}: the buffer checkpoint is bound to {bound_to} but {model_checkpoint} "
            "is not a directory; set ReservoirReplay.model_checkpoint to the directory the model restarts from"
        )
    current = directory_digest(model_checkpoint)
    if current["digest"] != recorded.get("digest"):
        raise RuntimeError(
            f"resuming at step {global_step}: model checkpoint {Path(model_checkpoint)} ({current['files']} files, "
            f"{current['bytes']} bytes, digest {current['digest'][:16]}…) is not the one the buffer checkpoint was "
            f"bound to ({recorded.get('files')} files, {recorded.get('bytes')} bytes, digest "
            f"{str(recorded.get('digest'))[:16]}…); the buffer would replay against a different model"
        )


def _wait_for_all_ranks(replay: ReservoirReplay) -> None:
    """Every rank reaches this before the owner reads the checkpoint directory.

    Under more than one process each rank writes its own files into the
    trainer's checkpoint (RNG state, sharded optimizer or model state), and
    the trainer's ``on_save`` carries no barrier of its own; the all-gather
    is used as one so the owner's digest covers a complete directory.
    """
    comm = getattr(replay, "_comm", None)
    if comm is not None and comm.num_processes > 1:
        comm.gather_object(None)


def resume_model_checkpoint(replay: ReservoirReplay, args, global_step: int) -> Optional[Path]:
    """The directory the model restarts from, for the resume check.

    In order: ``replay.model_checkpoint`` when the user set it (a copied or
    relocated checkpoint), the trainer argument ``resume_from_checkpoint``
    when it names a directory, else ``output_dir/checkpoint-N``.
    """
    override = getattr(replay, "model_checkpoint", None)
    if override is not None:
        return Path(override)
    named = getattr(args, "resume_from_checkpoint", None)
    if isinstance(named, (str, os.PathLike)) and Path(named).is_dir():
        return Path(named)
    return trainer_checkpoint_dir(getattr(args, "output_dir", None), global_step)


def bind_checkpoint(replay: ReservoirReplay, global_step: int, output_dir=None) -> None:
    """When the trainer saves a checkpoint, snapshot the durable buffer under the same step.

    With ``output_dir`` the digest of ``output_dir/checkpoint-N`` (the
    model checkpoint the trainer has just written; every rank is waited for
    first) is recorded in the buffer checkpoint, so a resume can refuse a
    different model, and the
    buffer's checkpoints are pruned to the steps the trainer still has (it
    rotates its own under ``save_total_limit`` before this is called), so
    the buffer directory does not grow without bound. A buffer without a
    directory has nothing to bind; the call is a no-op, as it is on every
    rank but the owner.
    """
    _wait_for_all_ranks(replay)
    if not replay.is_owner or not isinstance(replay.buffer, DurableRolloutBuffer):
        return
    model_checkpoint = trainer_checkpoint_dir(output_dir, global_step)
    binding = model_binding(model_checkpoint)
    if binding is None and model_checkpoint is not None:
        warnings.warn(
            f"no model checkpoint at {model_checkpoint} when the buffer checkpoint {checkpoint_tag(global_step)!r} "
            "was taken; a resume at this step will not be checked against the model",
            RuntimeWarning, stacklevel=2,
        )
    replay.buffer.checkpoint(checkpoint_tag(global_step), binding=binding)
    steps = trainer_checkpoint_steps(output_dir)
    if steps is not None:
        replay.buffer.prune_checkpoints({checkpoint_tag(n) for n in steps | {global_step}})


def resume_from_checkpoint(replay: ReservoirReplay, global_step: int, model_checkpoint=None) -> None:
    """When training (re)starts at ``global_step > 0``, rewind the buffer to that step's snapshot.

    Without the rewind the buffer would carry every operation the crashed
    run logged after the checkpoint while the model restarts before them.
    A durable buffer with no snapshot for the step is an error, so a resume
    cannot silently continue from the wrong point; a non-durable buffer
    cannot resume and raises if asked to. ``model_checkpoint`` is the
    directory the model restarts from; when the buffer checkpoint recorded
    a model digest, it must be that directory's (``check_model_binding``).
    Off the owner rank there is no buffer to rewind and the call is a no-op.
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
    check_model_binding(replay.buffer.checkpoint_binding(tag), model_checkpoint, global_step)
    replay.buffer.restore_checkpoint(tag)


__all__ = [
    "MODEL_CHECKPOINT_KEY",
    "RANK_ENV_VARS",
    "TRAINER_CHECKPOINT_PREFIX",
    "bind_checkpoint",
    "check_model_binding",
    "checkpoint_tag",
    "directory_digest",
    "env_rank",
    "model_binding",
    "resume_from_checkpoint",
    "resume_model_checkpoint",
    "trainer_checkpoint_dir",
    "trainer_checkpoint_steps",
]
