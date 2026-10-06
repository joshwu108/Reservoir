"""
reservoir.rollout_wal — A command log for the durable rollout buffer.

Every ``RolloutBuffer`` operation is a deterministic function of the
buffer's state and the operation's inputs: priorities are exact integers,
draws are keyed hashes, and the attestation records and manifest lines
are derived from the same inputs. So the durable buffer need not write its
whole state after every operation. It appends one line per operation to a
write-ahead log (the operation's name and inputs), and recovery replays
those lines onto the last compact snapshot, which reproduces the exact
state, the same attestation chain and the same manifest.

Line format: canonical JSON ``{"epoch": e, "seq": n, "op": name, "args":
{...}, "digest": blake2b(...)}`` followed by a newline. ``seq`` increases by
one per line within an epoch; ``digest`` is BLAKE2b-256 of the canonical
JSON of the line without the digest. A crash can leave a torn last line;
``read_commands`` stops at a last line that is incomplete, malformed or
whose digest does not match, and the caller truncates the file there. The
same damage before the end, or a sequence gap, is refused as corruption. A
snapshot records the ``seq`` of the last command it includes, so lines at
or below it (left by a crash between writing a snapshot and resetting the
log) are skipped. Restoring a checkpoint starts a new ``epoch``, recorded
in the snapshot and in every later line, so lines of the abandoned
timeline (left by a crash between the restore snapshot and the log reset)
are ignored rather than replayed, whatever their ``seq``.

Appending is: write the line, flush, full fsync. The operation has already
been applied in memory and validated by then; the caller returns the
result to its user only after the fsync, so a result that was observed is
always recoverable, and a crash before the fsync recovers to the state
before the operation, which no one observed. Instrumented cut points
(``RESERVOIR_CUT_POINT``): ``after_wal_write`` (line written, not synced),
``mid_wal_write`` (half a line), ``after_wal_fsync``.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, Optional

from reservoir.durable import _cut_byte_offset, _full_fsync, _kill_self, _should_cut
from reservoir.rollout import Rollout

WAL_FILE = "wal.jsonl"
_PERSON = b"reservoir-wal\x00\x00\x00"


def _digest(line: dict) -> str:
    body = {k: v for k, v in line.items() if k != "digest"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(canonical, digest_size=32, person=_PERSON).hexdigest()


def encode_rollouts(rollouts) -> list[dict]:
    """Rollouts as JSON (metadata must round-trip; the durable buffer checks that first)."""
    return [{"tokens": list(r.tokens), "logprobs": list(r.logprobs), "reward": r.reward,
             "metadata": dict(r.metadata)} for r in rollouts]


def decode_rollouts(raw: list[dict]) -> list[Rollout]:
    return [Rollout(tokens=r["tokens"], logprobs=r["logprobs"], reward=r["reward"],
                    metadata=r.get("metadata") or None) for r in raw]


def _fsync_directory(directory: Path) -> None:
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        _full_fsync(fd)
    finally:
        os.close(fd)


def _parse_line(line: bytes) -> Optional[dict]:
    """The command on a complete line, or None if it is malformed or its digest does not match."""
    if not line.strip():
        return None
    try:
        parsed = json.loads(line)
    except ValueError:
        return None
    if not isinstance(parsed, dict) or parsed.get("digest") != _digest(parsed):
        return None
    seq = parsed.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool):
        return None
    return parsed


class CommandLog:
    """Append-only command log in ``directory / wal.jsonl``."""

    def __init__(self, directory: Path) -> None:
        self.path = Path(directory) / WAL_FILE

    def append(self, seq: int, op: str, args: dict[str, Any], epoch: int = 0) -> None:
        """Durably append one command; returns after the fsync.

        The file's directory entry is fsynced when the file is created, so
        a power loss cannot drop a log whose first commands were already
        reported durable.
        """
        line = {"epoch": epoch, "seq": seq, "op": op, "args": args}
        line["digest"] = _digest(line)
        raw = (json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        created = not self.path.exists()
        if _should_cut("mid_wal_write"):
            offset = _cut_byte_offset()
            with open(self.path, "ab") as f:
                f.write(raw[:len(raw) // 2 if offset is None else offset])
                f.flush()
            _kill_self()
        with open(self.path, "ab") as f:
            f.write(raw)
            f.flush()
            if _should_cut("after_wal_write"):
                _kill_self()
            _full_fsync(f.fileno())
        if created:
            _fsync_directory(self.path.parent)
        if _should_cut("after_wal_fsync"):
            _kill_self()

    def read_commands(self, after_seq: int, epoch: int = 0) -> tuple[list[dict], int]:
        """Commands of ``epoch`` with ``seq > after_seq`` in order, and the byte offset where valid content ends.

        A crash can only damage the end of the file, so an incomplete,
        malformed or digest-mismatching *last* line is a torn tail and
        everything before it is returned. The same damage followed by more
        content, or a well-formed line whose ``seq`` does not follow the
        previous one of its epoch, is not something a crash produces and
        raises ``ValueError`` rather than silently dropping commands. Lines
        of another epoch (an abandoned timeline) and lines at or below
        ``after_seq`` (already in the snapshot) are skipped.
        """
        if not self.path.exists():
            return [], 0
        raw = self.path.read_bytes()
        commands: list[dict] = []
        valid_end = 0
        expected: Optional[int] = None
        while valid_end < len(raw):
            newline = raw.find(b"\n", valid_end)
            parsed = _parse_line(raw[valid_end:newline]) if newline >= 0 else None
            if parsed is None:
                if newline >= 0 and newline + 1 < len(raw):
                    raise ValueError(f"{self.path} has a damaged line at byte {valid_end} with content after it")
                if newline >= 0:
                    warnings.warn(f"{self.path}: discarding a complete last line whose digest does not match",
                                  RuntimeWarning, stacklevel=2)
                break  # torn tail: the last line is incomplete or damaged
            valid_end = newline + 1
            if parsed.get("epoch", 0) != epoch:
                continue
            seq = parsed["seq"]
            if expected is not None and seq != expected:
                raise ValueError(f"{self.path} has seq {seq} where {expected} was expected")
            expected = seq + 1
            if seq > after_seq:
                commands.append(parsed)
        return commands, valid_end

    def truncate(self, length: int) -> None:
        """Cut a torn tail off the file and fsync it."""
        if self.path.exists() and self.path.stat().st_size > length:
            with open(self.path, "r+b") as f:
                f.truncate(length)
                f.flush()
                _full_fsync(f.fileno())

    def reset(self) -> None:
        """Replace the log with an empty one, atomically (write, fsync, rename, fsync dir)."""
        tmp = self.path.with_suffix(".jsonl.tmp")
        with open(tmp, "wb") as f:
            f.flush()
            _full_fsync(f.fileno())
        os.rename(str(tmp), str(self.path))
        _fsync_directory(self.path.parent)

    def size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0


def apply_command(buf, command: dict) -> None:
    """Replay one command on ``buf`` (a ``RolloutBuffer``)."""
    op, args = command["op"], command["args"]
    if op == "add_group":
        buf.add_group(args["prompt_id"], args["model_version"], decode_rollouts(args["rollouts"]),
                      source=args.get("source"))
    elif op == "sample":
        buf.sample(args["batch_size"], args.get("current_version"))
    elif op == "update_priorities":
        buf.update_priorities(args["indices"], args["raw_scores"])
    elif op == "advance":
        buf.advance(args["current_version"])
    elif op == "evict":
        buf.evict(args["position"], args["reason"])
    elif op == "quarantine":
        buf.quarantine_positions(args["positions"], args["predicate"], args["note"])
    elif op == "witness_batch":
        buf.witness_batch(_sampled_batch(buf, args), args["step"], args["batch_rows"], args["rows"],
                          args["tensor_digest"], args.get("declined", ()))
    elif op == "record_telemetry":
        sample = _sampled_batch(buf, args) if args["with_sample"] else None
        buf.record_telemetry(args["step"], args["counts"], sample, args.get("reported") or {})
    else:
        raise ValueError(f"unknown command {op!r} in the write-ahead log")


def _sampled_batch(buf, args: dict):
    """The batch a witness or telemetry command was issued against, or a ValueError.

    The command records the sample's ``op_counter``; the buffer rebuilds
    its last batch only while every sampled slot still holds the rollout
    it held at sample time. Anything else means the log does not describe
    the run that was observed.
    """
    batch = buf.last_batch
    if batch is None or batch.op_counter != args["sample_op"]:
        raise ValueError(
            f"command refers to sample {args['sample_op']} but the buffer's last sampled batch is "
            f"{'gone' if batch is None else batch.op_counter}"
        )
    return batch
