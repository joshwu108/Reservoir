"""
reservoir.durable_rollout — Crash-atomic RolloutBuffer backed by a directory.

``DurableRolloutBuffer`` wraps ``RolloutBuffer`` so that every operation
is durable before it returns and a crash at any point leaves exactly the
state before or after the operation, never a mix. It does so with a
command log (``rollout_wal.py``): each operation is applied in memory,
then its name and inputs are appended to ``wal.jsonl`` and fsynced, and
only then is the result returned. Every operation of the buffer is a
deterministic function of state and inputs, so replaying the log onto the
last snapshot reproduces the state, the attestation chain and the
manifest exactly. Once ``compact_every`` commands (default 256) have
accumulated, the next operation first writes a full snapshot through the
intent/segment/rename protocol of ``durable.py`` (``durably_snapshot``,
which serialises the state once) and resets the log, so
the per-operation cost does not grow with the run's history (the snapshot
does, see below).

Usage::

    buf = DurableRolloutBuffer("run-01/buffer", capacity=50_000, half_life=4,
                               max_policy_age=16, seed=0, attest="run-01/attest.jsonl")
    buf.add_group(...)            # durable before it returns
    batch = buf.sample(64, current_version=step)
    # ... process dies ...
    buf = DurableRolloutBuffer("run-01/buffer", capacity=50_000, half_life=4,
                               max_policy_age=16, seed=0, attest="run-01/attest.jsonl")
    # same live entries, same counters, same next draw, same attestation chain

Checkpoints: ``checkpoint(tag)`` compacts and copies the snapshot under
``checkpoints/<tag>/``; ``restore_checkpoint(tag)`` rewinds the directory
to it. The TRL adapter calls both so that a run resumed from a trainer
checkpoint continues the buffer, the draw counter and the chain from the
same point rather than from wherever the buffer had got to before the
crash.

Costs and limits
----------------
- A command costs one append and one fsync; a compaction costs a full
  snapshot, whose size grows with the attestation log and the manifest
  (they are part of the state). ``compact_every`` trades the two off.
- The attestation and manifest files are written as each operation runs,
  before its command is fsynced, and are rewritten from the recovered
  state on reopen. So a reader of those files during a crash window may
  see a record of an operation that recovery then retracts; after reopen
  the files and the buffer agree. ``attest`` and ``manifest`` must be
  paths or None; in-memory targets cannot be recovered into.
- Construction parameters must match the saved state; a mismatch is a
  ``ValueError`` rather than a silent reinterpretation. A corrupt
  snapshot is an error, never a fresh start. Both are checked on a
  throwaway in-memory buffer before any file is opened for writing.
- An operation that raises, or whose command cannot be written, is
  undone in memory by rebuilding the buffer from disk (a partial line is
  cut first), so memory and disk never diverge. A compaction failure
  surfaces on the operation that attempted it, before that operation is
  applied; the commands already logged are unaffected.
- A checkpoint is a snapshot copied aside with an fsync of the file and
  its directory; ``restore_checkpoint`` starts a new log epoch so that a
  crash between the restore snapshot and the log reset cannot replay the
  abandoned timeline. The crash campaign covers the log, snapshot and
  restore cut points.
- Custom ``is_success`` predicates are not supported (a callable cannot
  be saved); rollout metadata must be JSON-serialisable.
- A directory written by the pre-0.5 full-snapshot protocol opens; its
  state becomes the first snapshot of the command log.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

from reservoir.attest import AttestationLog
from reservoir.decayed_tree import AdvanceResult
from reservoir.durable import CorruptStateError, _full_fsync, _kill_self, _should_cut, durably_snapshot, recover_state
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBatch, RolloutBuffer
from reservoir.rollout_manifest import ManifestWriter
from reservoir.rollout_quarantine import (
    MAX_NOTE_LENGTH, MAX_PREDICATE_LENGTH, QuarantinePredicate, quarantine_text, require_quarantine_format,
    select_positions, validate_text,
)
from reservoir.rollout_snapshot import _require_json_round_trip
from reservoir.rollout_wal import CommandLog, _fsync_directory, apply_command, encode_rollouts

_LOAD_ERRORS = (KeyError, TypeError, ValueError, IndexError)
DEFAULT_COMPACT_EVERY = 256
BINDING_FILE = "binding.json"
"""Optional caller data beside a checkpoint's state (see ``DurableRolloutBuffer.checkpoint``)."""
CHECKPOINT_DIR = "checkpoints"


def _strict_position(value: object) -> int:
    """A slot number given as a plain int; bools and floats are refused rather than coerced."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"slot positions must be plain ints, got {value!r}")
    return value


class DurableRolloutBuffer:
    """``RolloutBuffer`` whose every operation is crash-atomic on disk.

    Parameters
    ----------
    directory : str | Path
        Where the snapshot, the command log and checkpoints live. Created
        if missing.
    attest : str | Path | None
        Attestation file path, or None. See the module docstring.
    manifest : str | Path | None
        Manifest file path, or None. Requires ``attest``.
    compact_every : int
        Commands between snapshots. Default 256.
    **buffer_kwargs
        Everything ``RolloutBuffer`` accepts except ``attest``,
        ``manifest`` and ``attest_overwrite``.

    Raises
    ------
    ValueError
        A saved state exists but was written with different parameters,
        or is corrupt, or carries an attestation log while ``attest`` is
        None (or the reverse); or the command log cannot be replayed.
    """

    def __init__(
        self,
        directory: Union[str, Path],
        attest: Union[str, Path, None] = None,
        manifest: Union[str, Path, None] = None,
        compact_every: int = DEFAULT_COMPACT_EVERY,
        **buffer_kwargs: Any,
    ) -> None:
        if attest is not None and not isinstance(attest, (str, Path)):
            raise TypeError(
                "DurableRolloutBuffer attest must be a file path or None; an in-memory "
                "AttestationLog cannot be recovered after a crash"
            )
        if manifest is not None and not isinstance(manifest, (str, Path)):
            raise TypeError(
                "DurableRolloutBuffer manifest must be a file path or None; an in-memory "
                "ManifestWriter cannot be recovered after a crash"
            )
        if manifest is not None and attest is None:
            raise ValueError("manifest requires attestation: pass attest=<path> as well")
        for forbidden in ("attest", "manifest", "attest_overwrite"):
            if forbidden in buffer_kwargs:
                raise TypeError(f"{forbidden} is managed by DurableRolloutBuffer")
        if isinstance(compact_every, bool) or not isinstance(compact_every, int) or compact_every < 1:
            raise ValueError(f"compact_every must be a positive int, got {compact_every!r}")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._buffer_kwargs = dict(buffer_kwargs)
        self._attest = Path(attest) if attest is not None else None
        self._manifest = Path(manifest) if manifest is not None else None
        self.compact_every = compact_every
        self._epoch = 0
        self._restored_tag: Optional[str] = None   # kept by prune_checkpoints
        self._wal = CommandLog(self.directory)
        self._seq = 0                  # seq of the last durable command
        self._snapshot_seq = 0         # seq the snapshot on disk includes
        self._buf = self._open()

    # -- opening and recovery ------------------------------------------------

    def _open(self) -> RolloutBuffer:
        """Recover the snapshot, replay the command log, return the live buffer."""
        try:
            snapshot = recover_state(self.directory, strict=True)
        except CorruptStateError as exc:
            raise ValueError(f"saved state in {self.directory} is corrupt: {exc}") from exc
        if snapshot is None:
            buf = self._new_buffer()
            self._snapshot_seq = self._seq = 0
            # Persist the empty state (and the decay_config record) so the
            # log and state agree even if the process dies before the first
            # operation.
            durably_snapshot(self.directory, "open", self._snapshot_of(buf, 0))
            return buf
        state, self._snapshot_seq, self._epoch = self._unwrap(snapshot)
        probe = self._validate_state(state)
        # Replay onto the throwaway buffer first: a log that cannot be
        # replayed is refused before the attestation file is rewritten.
        self._replay_log(probe)
        buf = self._buffer_from_state(state)
        self._seq = self._replay_log(buf)
        return buf

    @staticmethod
    def _unwrap(snapshot: dict) -> tuple[dict, int, int]:
        """The buffer state, command seq and log epoch of a snapshot; a pre-0.5 full-state file is seq 0, epoch 0."""
        if "buffer" in snapshot and "wal_seq" in snapshot:
            seq, epoch = snapshot["wal_seq"], snapshot.get("wal_epoch", 0)
            for name, value in (("wal_seq", seq), ("wal_epoch", epoch)):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"snapshot {name} must be a non-negative int, got {value!r}")
            return snapshot["buffer"], seq, epoch
        return snapshot, 0, 0

    def _snapshot_of(self, buf: RolloutBuffer, seq: int, epoch: Optional[int] = None) -> dict:
        return {"buffer": buf.state_dict(), "wal_seq": seq,
                "wal_epoch": self._epoch if epoch is None else epoch}

    def _replay_log(self, buf: RolloutBuffer) -> int:
        """Apply the commands after the snapshot; cut a torn tail; return the last seq."""
        commands, valid_end = self._wal.read_commands(after_seq=self._snapshot_seq, epoch=self._epoch)
        self._wal.truncate(valid_end)
        seq = self._snapshot_seq
        for command in commands:
            if command["seq"] != seq + 1:
                raise ValueError(
                    f"command log in {self.directory} skips from seq {seq} to {command['seq']}"
                )
            try:
                apply_command(buf, command)
            except Exception as exc:
                raise ValueError(
                    f"command log in {self.directory} could not be replayed at seq {command['seq']}: {exc}"
                ) from exc
            seq = command["seq"]
        return seq

    def _new_buffer(self) -> RolloutBuffer:
        """An empty buffer with this wrapper's parameters, writing to the attestation file."""
        return RolloutBuffer(
            attest=self._attest, manifest=self._manifest, attest_overwrite=True, **self._buffer_kwargs
        )

    def _validate_state(self, state: dict) -> RolloutBuffer:
        """Load ``state`` into a throwaway in-memory buffer so a bad snapshot or
        mismatched parameters are rejected before the attestation file is opened
        for writing (which would truncate it); returns that buffer."""
        if (state.get("attestation") is None) != (self._attest is None):
            raise ValueError(
                "attestation setting differs from the saved state: "
                + ("the state has a log, pass attest=<path>" if self._attest is None
                   else "the state has no log, pass attest=None")
            )
        if (state.get("manifest") is None) != (self._manifest is None):
            raise ValueError(
                "manifest setting differs from the saved state: "
                + ("the state has a manifest, pass manifest=<path>" if self._manifest is None
                   else "the state has no manifest, pass manifest=None")
            )
        probe = RolloutBuffer(
            attest=AttestationLog(), manifest=ManifestWriter() if self._manifest is not None else None,
            **self._buffer_kwargs,
        )
        try:
            probe.load_state_dict(state)
        except _LOAD_ERRORS as exc:
            raise ValueError(
                f"saved state in {self.directory} could not be loaded: {exc}"
            ) from exc
        return probe

    def _buffer_from_state(self, state: dict) -> RolloutBuffer:
        """A new attesting buffer holding ``state``; the mirror files are rewritten from it."""
        buf = self._new_buffer()
        try:
            buf.load_state_dict(state)
        except _LOAD_ERRORS:
            buf.close()
            raise
        return buf

    def _rebuild(self) -> None:
        """Undo a failed operation in memory by rebuilding from disk (it was never logged)."""
        self._buf.close()
        self._buf = self._open()

    # -- the commit protocol -----------------------------------------------

    def _commit(self, op: str, args: dict, fn) -> Any:
        """Compact if due, apply ``fn`` in memory, log the command durably, return the result.

        A raise inside ``fn`` or while writing the command rebuilds memory
        from disk and propagates; neither leaves memory ahead of the log.
        """
        if self.pending_commands >= self.compact_every:
            self.compact()
        logged_bytes = self._wal.size_bytes()
        try:
            result = fn()
        except BaseException:
            self._rebuild()
            raise
        try:
            self._wal.append(self._seq + 1, op, args, self._epoch)
        except BaseException as exc:
            self._retract(logged_bytes, exc)
            self._rebuild()
            raise
        self._seq += 1
        return result

    def _retract(self, logged_bytes: int, cause: BaseException) -> None:
        """Cut a command whose append failed back off the log.

        If even that fails the log may hold a readable but unsynced command,
        so the buffer is closed rather than left to disagree with disk.
        """
        try:
            self._wal.truncate(logged_bytes)
        except Exception as cut_exc:
            self._buf.close()
            raise RuntimeError(
                f"the command log could not be retracted after a failed append ({cause!r}); "
                "the buffer is closed and must be reopened"
            ) from cut_exc

    def compact(self) -> None:
        """Write a snapshot of the current state and reset the command log.

        The snapshot goes through the intent/segment/rename protocol
        (``durably_snapshot``: the state is serialised once and the fsynced
        segment becomes ``state.json`` by rename), so a crash leaves either
        the old snapshot (plus the full log) or the new one; the log is reset
        only after the new snapshot is committed, and commands the snapshot
        already includes are skipped on replay.
        """
        seq = self._seq
        durably_snapshot(self.directory, "compact", self._snapshot_of(self._buf, seq))
        if _should_cut("after_snapshot_before_wal_reset"):
            _kill_self()
        self._wal.reset()
        self._snapshot_seq = seq

    @property
    def pending_commands(self) -> int:
        """Commands logged since the last snapshot."""
        return self._seq - self._snapshot_seq

    @property
    def wal_bytes(self) -> int:
        """Size of the command log on disk."""
        return self._wal.size_bytes()

    # -- checkpoints -----------------------------------------------------------

    @staticmethod
    def _check_tag(tag: str) -> str:
        if not isinstance(tag, str) or not tag or "/" in tag or "\\" in tag or tag in (".", ".."):
            raise ValueError(f"checkpoint tag must be a plain name, got {tag!r}")
        return tag

    def checkpoint(self, tag: str, binding: Optional[dict] = None) -> Path:
        """Compact, then copy the snapshot to ``checkpoints/<tag>/``; returns that directory.

        The copy is fsynced and renamed into place and its directory is
        fsynced, so a checkpoint that exists after a power loss is whole.
        ``binding`` is a JSON object the caller ties to this checkpoint (an
        adapter records the digest of the model checkpoint it was taken
        with); it is written as ``binding.json`` beside the state, the same
        way, and read back by ``checkpoint_binding``. A checkpoint taken
        without one has no ``binding.json``. When a tag is re-taken, its
        old binding is removed before the new state is installed, so a
        crash inside this call can leave the checkpoint unbound (resume
        then runs unchecked) but never bound to the wrong digest.
        """
        if binding is not None and not isinstance(binding, dict):
            raise TypeError(f"binding must be a dict or None, got {type(binding).__name__}")
        target = self.directory / CHECKPOINT_DIR / self._check_tag(tag)
        self.compact()
        target.mkdir(parents=True, exist_ok=True)
        if (target / BINDING_FILE).exists():
            self._write_binding(target, None)
            _fsync_directory(target)
        tmp = target / "state.json.tmp"
        with open(self.directory / "state.json", "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            _full_fsync(dst.fileno())
        tmp.replace(target / "state.json")
        self._write_binding(target, binding)
        _fsync_directory(target)
        return target

    @staticmethod
    def _write_binding(target: Path, binding: Optional[dict]) -> None:
        """Install or remove ``binding.json`` (fsynced, renamed into place); the caller fsyncs the directory."""
        path = target / BINDING_FILE
        if binding is None:
            if path.exists():
                path.unlink()
            return
        data = json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
        tmp = target / f"{BINDING_FILE}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            _full_fsync(f.fileno())
        tmp.replace(path)

    def checkpoint_binding(self, tag: str) -> Optional[dict]:
        """The ``binding`` recorded with ``checkpoints/<tag>``, or None when it has none."""
        target = self.directory / CHECKPOINT_DIR / self._check_tag(tag)
        if not (target / "state.json").exists():
            raise FileNotFoundError(f"no checkpoint {tag!r} under {self.directory / CHECKPOINT_DIR}")
        path = target / BINDING_FILE
        if not path.exists():
            return None
        try:
            binding = json.loads(path.read_bytes())
        except (OSError, ValueError) as exc:
            raise ValueError(f"checkpoint {tag!r} has an unreadable {BINDING_FILE}: {exc}") from exc
        if not isinstance(binding, dict):
            raise ValueError(f"checkpoint {tag!r}: {BINDING_FILE} must hold a JSON object")
        return binding

    def restore_checkpoint(self, tag: str) -> None:
        """Rewind to ``checkpoints/<tag>``: later commands are discarded, the chain resumes there.

        The restore snapshot carries a new log epoch; lines of the abandoned
        timeline are ignored on replay even if the reset that follows the
        snapshot never ran.
        """
        source = self.directory / CHECKPOINT_DIR / self._check_tag(tag) / "state.json"
        if not source.exists():
            raise FileNotFoundError(f"no checkpoint {tag!r} under {self.directory / CHECKPOINT_DIR}")
        state, seq, saved_epoch = self._unwrap(json.loads(source.read_bytes()))
        self._validate_state(state)
        epoch = max(self._epoch, saved_epoch) + 1
        durably_snapshot(self.directory, "restore", {"buffer": state, "wal_seq": seq, "wal_epoch": epoch})
        if _should_cut("after_restore_before_wal_reset"):
            _kill_self()
        self._buf.close()
        self._wal.reset()
        self._snapshot_seq = self._seq = seq
        self._epoch = epoch
        self._restored_tag = tag
        self._buf = self._buffer_from_state(state)

    def prune_checkpoints(self, keep: Iterable[str]) -> list[str]:
        """Delete every checkpoint whose tag is not in ``keep``; returns the deleted tags.

        The checkpoint this buffer last restored from is always kept: it is
        the one a resumed run would need again if it crashes before its next
        save, whatever the trainer's directory says.
        """
        keep_set = {self._check_tag(t) for t in keep}
        if self._restored_tag is not None:
            keep_set.add(self._restored_tag)
        deleted = [tag for tag in self.checkpoints() if tag not in keep_set]
        for tag in deleted:
            shutil.rmtree(self.directory / CHECKPOINT_DIR / tag)
        return deleted

    def checkpoints(self) -> list[str]:
        """Checkpoint tags present, sorted."""
        root = self.directory / CHECKPOINT_DIR
        return sorted(p.name for p in root.iterdir() if (p / "state.json").exists()) if root.exists() else []

    # -- durable operations ------------------------------------------------

    def add_group(
        self, prompt_id: str, model_version: int, rollouts: Sequence[Rollout],
        source: Optional[str] = None,
    ) -> tuple[int, ...]:
        """Durably store a prompt group. No ``is_success``: predicates cannot be saved."""
        rollouts = list(rollouts)
        for k, r in enumerate(rollouts):
            _require_json_round_trip(dict(getattr(r, "metadata", {})), f"rollout {k} of prompt {prompt_id!r}")
        args = {"prompt_id": prompt_id, "model_version": model_version,
                "rollouts": encode_rollouts(rollouts), "source": source}
        return self._commit("add_group", args,
                            lambda: self._buf.add_group(prompt_id, model_version, rollouts, source=source))

    def sample(self, batch_size: int, current_version: Optional[int] = None) -> RolloutBatch:
        """Durably sample: the draw counter and any stale evictions are committed."""
        args = {"batch_size": batch_size, "current_version": current_version}
        return self._commit("sample", args, lambda: self._buf.sample(batch_size, current_version))

    def update_priorities(self, indices: Sequence[int], raw_scores: Sequence[float]) -> None:
        """Durably re-score live entries; see ``RolloutBuffer.update_priorities``."""
        idx, scores = [int(i) for i in indices], [float(s) for s in raw_scores]
        self._commit("update_priorities", {"indices": idx, "raw_scores": scores},
                     lambda: self._buf.update_priorities(idx, scores))

    def advance(self, current_version: int) -> AdvanceResult:
        """Durably move to a newer version, committing any evictions and rebase."""
        return self._commit("advance", {"current_version": current_version},
                            lambda: self._buf.advance(current_version))

    def witness_batch(self, batch: RolloutBatch, step: int, batch_rows: int, rows: Sequence[int],
                      tensor_digest: str, declined: Sequence[int] = ()) -> None:
        """Durably record the batch witness; see ``RolloutBuffer.witness_batch``."""
        args = {"step": step, "batch_rows": batch_rows, "rows": [int(r) for r in rows],
                "tensor_digest": tensor_digest, "declined": [int(d) for d in declined],
                "sample_op": int(batch.op_counter)}
        self._commit("witness_batch", args,
                     lambda: self._buf.witness_batch(batch, step, batch_rows, rows, tensor_digest, declined))

    def evict(self, position: int, reason: str = "explicit") -> None:
        """Durably remove a live entry with a reason; see ``RolloutBuffer.evict``."""
        self._commit("evict", {"position": int(position), "reason": reason},
                     lambda: self._buf.evict(position, reason))

    def quarantine(self, predicate: QuarantinePredicate, reason: str,
                   predicate_text: Optional[str] = None) -> tuple[int, ...]:
        """Durably quarantine every matching entry; see ``RolloutBuffer.quarantine``.

        The predicate is evaluated against the committed state first; the
        command logs the selected positions and both texts, never the
        callable, so recovery replays the same evictions. No command is
        logged when nothing matches.
        """
        text = quarantine_text(predicate, predicate_text)
        validate_text(reason, "quarantine reason", MAX_NOTE_LENGTH)
        require_quarantine_format(self._buf)
        positions = select_positions(self._buf, predicate)
        self.quarantine_positions(positions, text, reason)
        return positions

    def quarantine_positions(self, positions: Sequence[int], predicate_text: str, reason: str) -> None:
        """Durably quarantine the given slots; see ``RolloutBuffer.quarantine_positions``.

        Texts and format are validated before anything is logged; an
        empty list logs nothing.
        """
        validate_text(predicate_text, "predicate text", MAX_PREDICATE_LENGTH)
        validate_text(reason, "quarantine reason", MAX_NOTE_LENGTH)
        require_quarantine_format(self._buf)
        slots = [_strict_position(p) for p in positions]
        if not slots:
            return
        self._commit("quarantine", {"positions": slots, "predicate": predicate_text, "note": reason},
                     lambda: self._buf.quarantine_positions(slots, predicate_text, reason))

    def record_telemetry(self, step: int, counts: dict, sample=None, reported=None) -> None:
        """Durably write a telemetry record; see ``RolloutBuffer.record_telemetry``."""
        args = {"step": step, "counts": dict(counts), "with_sample": sample is not None,
                "sample_op": int(sample.op_counter) if sample is not None else None,
                "reported": dict(reported or {})}
        self._commit("record_telemetry", args,
                     lambda: self._buf.record_telemetry(step, counts, sample, reported))

    # -- read-only passthroughs --------------------------------------------
    # Each of these reads the wrapped buffer and touches nothing on disk;
    # they mean exactly what the same-named member of RolloutBuffer means.

    @property
    def buffer(self) -> RolloutBuffer:
        """The wrapped in-memory buffer. Mutating it directly bypasses durability."""
        return self._buf

    @property
    def capacity(self) -> int:
        """Slot count (power of two)."""
        return self._buf.capacity

    @property
    def size(self) -> int:
        """Number of live rollouts."""
        return self._buf.size

    def __len__(self) -> int:
        return self._buf.size

    @property
    def total(self) -> int:
        """Sum of decayed leaves, the sampling denominator."""
        return self._buf.total

    @property
    def current_version(self) -> int:
        """Latest committed model version."""
        return self._buf.current_version

    @property
    def base_epoch(self) -> int:
        """Epoch the leaves are shifted relative to."""
        return self._buf.base_epoch

    @property
    def n_rebases(self) -> int:
        """Rebases performed so far."""
        return self._buf.n_rebases

    @property
    def params(self):
        """The validated decay configuration."""
        return self._buf.params

    @property
    def priority(self):
        """The priority strategy the wrapped buffer scores with."""
        return self._buf.priority

    @property
    def attestation_log(self):
        """The in-memory attestation log, or None."""
        return self._buf.attestation_log

    @property
    def has_manifest(self) -> bool:
        """True when a manifest file is kept next to the attestation log."""
        return self._buf.has_manifest

    @property
    def manifest_records(self) -> list[dict]:
        """Manifest lines of the committed state (a copy)."""
        return self._buf.manifest_records

    @property
    def last_batch(self) -> Optional[RolloutBatch]:
        """The most recent sampled batch, if its slots are all still live."""
        return self._buf.last_batch

    def live_positions(self) -> tuple[int, ...]:
        """Slots holding a rollout, ascending."""
        return self._buf.live_positions()

    def entry(self, position: int):
        """``(rollout, group)`` at a live slot."""
        return self._buf.entry(position)

    def leaf(self, position: int) -> int:
        """Decayed leaf at a slot, 0 if empty."""
        return self._buf.leaf(position)

    def base_priority(self, position: int) -> int:
        """Fixed-point q of a live slot."""
        return self._buf.base_priority(position)

    def entry_version(self, position: int) -> int:
        """Version a live slot is aged from."""
        return self._buf.entry_version(position)

    def verify_trees(self) -> bool:
        """Recompute both trees' internal nodes; AssertionError on mismatch."""
        return self._buf.verify_trees()

    def state_dict(self) -> dict:
        """The current buffer state (the snapshot plus the replayed commands)."""
        return self._buf.state_dict()

    def close(self) -> None:
        """Close the attestation file. State is already committed; nothing is flushed here."""
        self._buf.close()

    def __enter__(self) -> "DurableRolloutBuffer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"DurableRolloutBuffer({str(self.directory)!r}, {self._buf!r}, pending={self.pending_commands})"
