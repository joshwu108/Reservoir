"""
reservoir.integrations._trl_distributed — Replay with more than one training process.

TRL's ``GRPOTrainer`` gives each process a contiguous, rank-ordered slice
of every generation batch: process ``k`` holds rows
``[k * n, (k + 1) * n)`` of the global batch, where the global batch has
the ``[g * G, (g + 1) * G)`` prompt-group layout the adapter relies on. A
group may therefore straddle two ranks, and only the concatenation of all
slices in rank order has whole groups.

The arrangement here:

- **Rank 0 owns the buffer and the only log writer.** The other ranks hold a
  ``ReservoirReplay`` whose buffer is never built; they open no file, write
  no record, and ``bind_checkpoint``/``resume_from_checkpoint`` are no-ops
  for them.
- **Every rank computes behavior logprobs for its own rows** (one no-grad
  forward, balanced across devices), then the slices are gathered to rank 0.
- **Rank 0 runs the ordinary single-process hook on the global batch**, after
  moving the gathered CPU shards to its own device (the hook's telemetry
  forward feeds the batch to the model): store
  live groups, advance, replay dead rows, gate, witness, measure. Nothing in
  that path knows it is distributed.
- **The result is broadcast and sliced back**: each rank receives the rows it
  contributed, padded to the global width when replay widened the batch,
  and the recomputed ``num_items_in_batch`` (TRL's value is the global sum,
  so the value computed on the global batch is the one every rank needs).

Collectives are one all-gather (``gather_object``: every rank receives
every slice as CPU tensors and holds them until the step returns)
followed by one ``broadcast_object`` per generation step, plus one gather
in ``attach`` on the first call. Both run on every rank exactly once per
hook call whatever fails where: a rank that
fails before the gather contributes its error instead of a batch, rank 0
broadcasts any failure instead of a result, and every rank raises after
the broadcast rather than hanging in a collective another rank never
joined.

``Communicator`` is the small interface the module needs. A real
``accelerate.Accelerator`` is wrapped by ``AcceleratorCommunicator`` on top
of ``accelerate.utils.gather_object`` / ``broadcast_object_list``; an
object that already provides the four members (the tests' fake) is used
as is. A step whose ranks disagree on ``global_step`` is refused on all of
them.
"""

from __future__ import annotations

from typing import Any, Final, Optional, Protocol, runtime_checkable

import torch

from reservoir.integrations._trl_rows import pad_batch

PADDED_KEYS: Final[tuple[str, ...]] = (
    "prompt_ids", "prompt_mask", "completion_ids", "completion_mask",
    "old_per_token_logps", "ref_per_token_logps",
)
"""Per-row tensors whose width may differ between ranks; ``pad_batch`` aligns them."""

OWNER_RANK: Final[int] = 0


@runtime_checkable
class Communicator(Protocol):
    """What the distributed hook needs from the launcher: a world size, a rank and two collectives."""

    num_processes: int
    process_index: int

    def gather_object(self, obj: Any) -> list[Any]:
        """All-gather a picklable object; every rank receives the list in rank order."""

    def broadcast_object(self, obj: Any) -> Any:
        """Broadcast a picklable object from rank 0; every rank receives rank 0's value."""


class AcceleratorCommunicator:
    """``Communicator`` over an ``accelerate.Accelerator``'s process group."""

    def __init__(self, accelerator: Any) -> None:
        self.accelerator = accelerator
        self.num_processes = int(accelerator.num_processes)
        index = getattr(accelerator, "process_index", None)
        if index is None:
            if self.num_processes > 1:
                raise RuntimeError(
                    f"the accelerator reports {self.num_processes} processes but no process_index; "
                    "the adapter cannot tell which rank owns the buffer"
                )
            index = OWNER_RANK
        self.process_index = int(index)

    def gather_object(self, obj: Any) -> list[Any]:
        from accelerate.utils import gather_object

        return list(gather_object([obj]))

    def broadcast_object(self, obj: Any) -> Any:
        from accelerate.utils import broadcast_object_list

        return broadcast_object_list([obj], from_process=OWNER_RANK)[0]


def communicator_for(accelerator: Any) -> Communicator:
    """The accelerator itself when it already speaks ``Communicator``, else a wrapper around accelerate."""
    if isinstance(accelerator, Communicator):
        return accelerator
    return AcceleratorCommunicator(accelerator)


# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------

def _row_count(shard: dict) -> int:
    return int(shard["advantages"].size(0))


def _is_row_tensor(value: Any, rows: int) -> bool:
    return isinstance(value, torch.Tensor) and value.dim() >= 1 and value.size(0) == rows


def to_cpu(batch: dict) -> dict:
    """A copy of ``batch`` with every tensor detached on the CPU, ready to pickle."""
    return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def shard_sizes(shards: list[dict]) -> tuple[int, ...]:
    return tuple(_row_count(s) for s in shards)


def concat_shards(shards: list[dict], pad_token_id: int) -> dict:
    """The global batch: shards padded to common widths and concatenated in rank order.

    Entries that are not per-row tensors (``num_items_in_batch`` and any
    scalar) are taken from rank 0's shard. A per-row tensor outside
    ``PADDED_KEYS`` whose width differs between ranks cannot be aligned and
    is a ``ValueError`` naming the key.
    """
    target_p = max(s["prompt_ids"].size(1) for s in shards)
    target_c = max(s["completion_ids"].size(1) for s in shards)
    padded = [pad_batch(s, target_p, target_c, pad_token_id) for s in shards]
    out = dict(padded[0])
    rows0 = _row_count(padded[0])
    for key, first in padded[0].items():
        if isinstance(first, (list, tuple)) and len(first) == rows0:
            raise ValueError(
                f"batch entry {key!r} is a per-row list; the adapter gathers tensors only and would "
                "otherwise hand every rank rank 0's copy"
            )
        if not _is_row_tensor(first, rows0):
            continue
        parts = [p[key] for p in padded]
        shapes = {tuple(t.shape[1:]) for t in parts}
        if len(shapes) != 1:
            raise ValueError(
                f"batch entry {key!r} has different widths on different ranks ({sorted(shapes)}) and "
                "the adapter does not know how to pad it"
            )
        out[key] = torch.cat(parts, dim=0)
    return out


def slice_shard(batch: dict, rank: int, sizes: tuple[int, ...]) -> dict:
    """Rank ``rank``'s rows of a global ``batch`` whose shards had ``sizes`` rows."""
    start = sum(sizes[:rank])
    stop = start + sizes[rank]
    rows = sum(sizes)
    return {
        k: v[start:stop] if _is_row_tensor(v, rows) else v
        for k, v in batch.items()
    }


# ---------------------------------------------------------------------------
# The distributed hook
# ---------------------------------------------------------------------------

class _OwnerFailure(Exception):
    """Carried from rank 0 to the other ranks through the broadcast."""


def _owner_mix(replay: Any, shards: list[dict], steps: list[int], trainer: Any, step: int,
               device: torch.device):
    """Rank 0's half: assemble, run the single-process hook, return the CPU payload (or None).

    The gathered shards are CPU tensors; the hook runs on ``device`` (rank
    0's own batch device) because its telemetry forward and the drift gate
    feed the batch to the model, which lives there.
    """
    if any(s != step for s in steps):
        raise RuntimeError(
            f"ranks disagree on global_step: {steps}; the trainer state is not in sync across processes"
        )
    global_batch = to_device(concat_shards(shards, trainer._tokenizer.pad_token_id), device)
    new = replay.mix_local(global_batch, trainer)
    # A step that sampled returns a rewritten dict even if every draw was declined (the one
    # process path does too, and the witness digest is over it); only a step with nothing
    # to replay hands back the dict it was given.
    return None if replay.last_replay is None and new is global_batch else to_cpu(new)


def _local_shard(replay: Any, output: dict, trainer: Any, step: int) -> tuple[dict, dict, Optional[BaseException]]:
    """This rank's contribution to the gather: its prepared batch, or the error that stopped it."""
    rank = getattr(getattr(trainer, "accelerator", None), "process_index", "?")
    try:
        prepared = replay.prepare_local(output, trainer)
        local = prepared
        if "old_per_token_logps" not in local:
            local = {**local, "old_per_token_logps": replay.behavior_logprobs(local, trainer)}
        shard = {"step": step, "batch": to_cpu(local)}
    except BaseException as exc:  # noqa: BLE001 - the rank must still join the collectives
        return output, {"step": step, "error": f"rank {rank}: {type(exc).__name__}: {exc}"}, exc
    return prepared, shard, None


def _owner_payload(replay: Any, shards: list[dict], trainer: Any, step: int, comm: Communicator,
                   device: torch.device) -> Any:
    """Rank 0's work between the gather and the broadcast; never returns without a broadcastable value."""
    errors = [s["error"] for s in shards if "error" in s]
    if errors:
        raise RuntimeError("replay stopped before the gather on " + "; ".join(errors))
    if len(shards) != comm.num_processes:
        raise RuntimeError(
            f"gathered {len(shards)} shards from a world of {comm.num_processes} processes; "
            "the process group is not initialised or the accelerator is not the launcher's"
        )
    return _owner_mix(replay, [s["batch"] for s in shards], [s["step"] for s in shards], trainer, step, device)


def mix_distributed(replay: Any, output: dict, trainer: Any, comm: Communicator) -> dict:
    """Run one generation step of replay across ``comm.num_processes`` ranks; see the module docstring.

    Returns ``output`` itself on every rank when rank 0 had nothing to
    replay (minus a dropped vLLM key, as in one process), otherwise this
    rank's slice of the rewritten global batch. Every rank makes exactly
    one gather and one broadcast, whatever fails where: a
    rank that failed before the gather contributes its error instead of a
    batch, rank 0 turns any such error (or its own) into the broadcast
    value, and every rank raises after the broadcast.
    """
    step = int(trainer.state.global_step)
    prepared, shard, local_error = _local_shard(replay, output, trainer, step)
    shards = comm.gather_object(shard)

    payload: Any = None
    if comm.process_index == OWNER_RANK:
        try:
            payload = _owner_payload(replay, shards, trainer, step, comm, output["completion_ids"].device)
        except BaseException as exc:  # noqa: BLE001 - every rank must leave the collective
            comm.broadcast_object({"error": f"{type(exc).__name__}: {exc}"})
            raise (local_error if local_error is not None else exc)
    payload = comm.broadcast_object(payload)
    if local_error is not None:
        raise local_error
    if isinstance(payload, dict) and "error" in payload:
        raise RuntimeError(f"replay failed on rank 0: {payload['error']}")
    if comm.process_index != OWNER_RANK:
        replay.stats["hook_calls"] += 1   # the owner counted its own in mix_local
    if payload is None:
        return prepared
    sizes = shard_sizes([s["batch"] for s in shards])
    return to_device(slice_shard(payload, comm.process_index, sizes), output["completion_ids"].device)


__all__ = [
    "AcceleratorCommunicator",
    "Communicator",
    "OWNER_RANK",
    "PADDED_KEYS",
    "communicator_for",
    "concat_shards",
    "mix_distributed",
    "shard_sizes",
    "slice_shard",
    "to_cpu",
    "to_device",
]
