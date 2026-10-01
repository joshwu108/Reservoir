"""
reservoir.durable — Write-ahead durable buffer over POSIX filesystem.

Crash-atomicity: a SIGKILL at any instrumented cut leaves the buffer
in exactly the pre-operation or post-operation state. Never a torn hybrid.

Commit protocol:
  1. Write intent record to intent.json.tmp (describes the pending operation)
  2. F_FULLFSYNC intent.json.tmp (Darwin) or fsync (Linux)
  3. Write segment files (data content)
  4. F_FULLFSYNC each segment file
  5. rename(intent.json.tmp, intent.json)  ← atomic commit point
  6. F_FULLFSYNC parent directory

Recovery:
  - No intent.json: pre-commit state, nothing to do.
  - intent.json + complete segments: apply operation (post-commit recovery).
  - intent.json + incomplete/missing segments: discard intent, restore pre-state.

macOS/APFS: fsync() does not guarantee durability. Use F_FULLFSYNC via fcntl.
Linux: fall back to os.fsync(). See docs/nonclaims.md §6.

Named cut points (environment variable CUT_POINT):
  - after_intent_write
  - after_intent_fsync
  - mid_segment_write
  - after_segment_fsync
  - before_rename
  - after_rename_before_dir_fsync
  - after_dir_fsync
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

# Cut-point injection for crash testing
_CUT_POINT: Optional[str] = os.environ.get("RESERVOIR_CUT_POINT")
_CUT_BYTE_OFFSET: Optional[int] = (
    int(os.environ.get("RESERVOIR_CUT_BYTE_OFFSET", "0"))
    if os.environ.get("RESERVOIR_CUT_BYTE_OFFSET")
    else None
)


def _should_cut(cut_name: str) -> bool:
    return _CUT_POINT == cut_name


def _full_fsync(fd: int) -> None:
    """Full fsync: F_FULLFSYNC on Darwin, os.fsync elsewhere."""
    if sys.platform == "darwin":
        try:
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        except (AttributeError, OSError):
            os.fsync(fd)
    else:
        os.fsync(fd)


def _fsync_file(path: Path) -> None:
    """Open a file and full-fsync it."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        _full_fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    """Full-fsync a directory (for rename commit visibility)."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        _full_fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes, parent: Path) -> None:
    """Write data to path atomically using tmp + rename, with full fsync.

    This is used for segment files and the intent record.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        _full_fsync(f.fileno())
    os.rename(str(tmp), str(path))
    _fsync_dir(parent)


# ---------------------------------------------------------------------------
# Intent record
# ---------------------------------------------------------------------------

_INTENT_FILE = "intent.json"
_INTENT_TMP = "intent.json.tmp"


def _write_intent(directory: Path, intent: dict) -> None:
    """Write the intent record to intent.json.tmp with full fsync.

    Instrumented cut points: after_intent_write, after_intent_fsync.
    """
    intent_tmp = directory / _INTENT_TMP
    data = json.dumps(intent, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with open(intent_tmp, "wb") as f:
        f.write(data)
        f.flush()
        if _should_cut("after_intent_write"):
            _kill_self()
        _full_fsync(f.fileno())
        if _should_cut("after_intent_fsync"):
            _kill_self()


def _commit_intent(directory: Path) -> None:
    """Rename intent.json.tmp to intent.json and fsync parent.

    Instrumented cut points: before_rename, after_rename_before_dir_fsync,
    after_dir_fsync.
    """
    intent_tmp = directory / _INTENT_TMP
    intent_file = directory / _INTENT_FILE

    if _should_cut("before_rename"):
        _kill_self()

    os.rename(str(intent_tmp), str(intent_file))

    if _should_cut("after_rename_before_dir_fsync"):
        _kill_self()

    _fsync_dir(directory)

    if _should_cut("after_dir_fsync"):
        _kill_self()


def _clear_intent(directory: Path) -> None:
    """Remove the intent record after successful recovery or completion."""
    intent = directory / _INTENT_FILE
    tmp = directory / _INTENT_TMP
    if intent.exists():
        intent.unlink()
    if tmp.exists():
        tmp.unlink()


def _kill_self() -> None:
    """Send SIGKILL to self — used only in crash testing subprocesses."""
    import signal
    os.kill(os.getpid(), signal.SIGKILL)


# ---------------------------------------------------------------------------
# Segment files
# ---------------------------------------------------------------------------

def _segment_path(directory: Path, index: int) -> Path:
    return directory / f"seg_{index:08d}.json"


def _write_segment(directory: Path, index: int, data: dict) -> None:
    """Write a segment file with full fsync.

    Instrumented cut points: mid_segment_write (truncated write),
    after_segment_fsync.
    """
    seg_path = _segment_path(directory, index)
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")

    if _should_cut("mid_segment_write") and _CUT_BYTE_OFFSET is not None:
        # Write only _CUT_BYTE_OFFSET bytes, then kill (simulates torn write)
        truncated = raw[:_CUT_BYTE_OFFSET]
        with open(seg_path, "wb") as f:
            f.write(truncated)
            f.flush()
        _kill_self()

    with open(seg_path, "wb") as f:
        f.write(raw)
        f.flush()
        _full_fsync(f.fileno())

    if _should_cut("after_segment_fsync"):
        _kill_self()


def _read_segment(directory: Path, index: int) -> Optional[dict]:
    """Read a segment file. Returns None if missing or corrupt."""
    seg_path = _segment_path(directory, index)
    if not seg_path.exists():
        return None
    try:
        return json.loads(seg_path.read_bytes())
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Buffer state serialization
# ---------------------------------------------------------------------------

def _serialize_state(
    size: int,
    write_pos: int,
    op_counter: int,
    max_prio: int,
    priorities: list[int],
    transitions: list[Optional[dict]],
) -> dict:
    return {
        "size": size,
        "write_pos": write_pos,
        "op_counter": op_counter,
        "max_prio": str(max_prio),
        "priorities": [str(p) for p in priorities],
        "transitions": transitions,
    }


def _deserialize_state(raw: dict) -> tuple:
    size = raw["size"]
    write_pos = raw["write_pos"]
    op_counter = raw["op_counter"]
    max_prio = int(raw["max_prio"])
    priorities = [int(p) for p in raw["priorities"]]
    transitions = raw["transitions"]
    return size, write_pos, op_counter, max_prio, priorities, transitions


# ---------------------------------------------------------------------------
# Generic protocol: any buffer that can snapshot itself to a JSON dict can
# use these two functions. DurableBuffer (classic PER) and
# DurableRolloutBuffer (rollout replay) both do.
# ---------------------------------------------------------------------------

_STATE_FILE = "state.json"


def _save_state_file(directory: Path, state: dict) -> None:
    """Atomically replace state.json (tmp file, fsync, rename, directory fsync)."""
    data = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    _atomic_write(directory / _STATE_FILE, data, directory)


class CorruptStateError(ValueError):
    """``state.json`` exists but cannot be read as a state dict."""


def _read_state_file(directory: Path, strict: bool) -> Optional[dict]:
    """``state.json`` as a dict, None if absent; unreadable is None or, if strict, an error."""
    state_file = directory / _STATE_FILE
    if not state_file.exists():
        return None
    try:
        state = json.loads(state_file.read_bytes())
    except (json.JSONDecodeError, OSError) as exc:
        if strict:
            raise CorruptStateError(f"{state_file} is unreadable: {exc}") from exc
        return None
    if not isinstance(state, dict):
        if strict:
            raise CorruptStateError(f"{state_file} does not hold a state object")
        return None
    return state


def recover_state(directory: Path, strict: bool = False) -> Optional[dict]:
    """Return the committed state for ``directory``, or None if it is fresh.

    With ``strict=True`` an unreadable or non-dict ``state.json`` raises
    ``CorruptStateError`` instead of being treated as a fresh directory,
    so data is never silently discarded. ``DurableBuffer`` keeps the lenient
    legacy behaviour; ``DurableRolloutBuffer`` is strict.

    Resolves whatever a crash left behind:

    - no ``intent.json``: nothing was in flight; return ``state.json`` if it
      exists and parses, else None (fresh or unreadable).
    - ``intent.json`` plus a complete ``seg_0``: the operation committed;
      return the post-state.
    - ``intent.json`` without a readable segment: the operation did not
      commit; return the pre-state the intent carried.

    In every case the intent and segment files are removed and the chosen
    state is written back to ``state.json`` so a second recovery is a
    no-op. The caller decides what to do with an unreadable state.
    """
    intent_file = directory / _INTENT_FILE
    intent_tmp = directory / _INTENT_TMP

    if not intent_file.exists():
        if intent_tmp.exists():
            intent_tmp.unlink()
        return _read_state_file(directory, strict)

    try:
        intent = json.loads(intent_file.read_bytes())
    except (json.JSONDecodeError, OSError):
        _clear_intent(directory)  # corrupt intent: nothing committed
        return recover_state(directory)

    post_state = _read_segment(directory, 0)
    state = post_state if post_state is not None else intent.get("pre_state")
    # state.json first, then the intent: a crash between the two leaves the
    # intent in place and the next recovery simply redoes this (idempotent).
    # The other order could expose the post-state once and then lose it.
    if state is not None:
        _save_state_file(directory, state)
    _clear_intent(directory)
    seg_path = _segment_path(directory, 0)
    if seg_path.exists():
        seg_path.unlink()
    return state


def durably_apply(
    directory: Path, operation_name: str, state_fn, apply_fn, on_abort=None
) -> Any:
    """Run ``apply_fn`` so that a crash leaves exactly the pre- or post-state.

    ``state_fn`` must return the buffer's full state as a JSON-serialisable
    dict; it is called once before and once after ``apply_fn``. Protocol:

      1. write intent (operation name + pre-state) to intent.json.tmp, fsync
      2. apply the operation in memory
      3. write the post-state to seg_0, fsync
      4. rename intent.json.tmp -> intent.json  (the atomic commit point)
      5. fsync the directory
      6. write state.json, then remove the intent and the segment

    If anything raises before the commit point (``apply_fn`` itself, or
    writing the segment), the intent is removed, ``on_abort(pre_state)``
    is called so the caller can roll its memory back to the pre-state, and
    the error propagates. An operation that raises *after* mutating (for
    example a sample that evicts stale entries and then finds the buffer
    empty) is therefore undone in memory too, and memory never diverges
    from disk.
    """
    pre_state = state_fn()
    _write_intent(directory, {"op": operation_name, "pre_state": pre_state})
    try:
        result = apply_fn()
        post_state = state_fn()
        _write_segment(directory, 0, post_state)
        _commit_intent(directory)
    except BaseException:
        _clear_intent(directory)
        seg_path = _segment_path(directory, 0)
        if seg_path.exists():
            seg_path.unlink()
        if on_abort is not None:
            on_abort(pre_state)
        raise

    _save_state_file(directory, post_state)
    _clear_intent(directory)
    seg_path = _segment_path(directory, 0)
    if seg_path.exists():
        seg_path.unlink()
    return result


# ---------------------------------------------------------------------------
# DurableBuffer
# ---------------------------------------------------------------------------

class DurableBuffer:
    """Durable, crash-atomic replay buffer backed by a POSIX directory.

    Wraps ExactPERBuffer with write-ahead logging and atomic commit.

    Parameters
    ----------
    directory : Path or str
        Directory to store intent and segment files.
    capacity : int
        Buffer capacity.
    alpha, beta, epsilon, seed, buffer_id : float/int
        Passed through to ExactPERBuffer.
    """

    def __init__(
        self,
        directory: "str | Path",
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        seed: int = 0,
        buffer_id: int = 0,
    ) -> None:
        from reservoir.buffer import ExactPERBuffer
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._capacity = capacity
        self._alpha = alpha
        self._beta = beta
        self._epsilon = epsilon
        self._seed = seed
        self._buffer_id = buffer_id

        # Recover or initialize
        self._buf = self._recover_or_init()

    def _make_fresh_buffer(self):
        from reservoir.buffer import ExactPERBuffer
        return ExactPERBuffer(
            capacity=self._capacity,
            alpha=self._alpha,
            beta=self._beta,
            epsilon=self._epsilon,
            seed=self._seed,
            buffer_id=self._buffer_id,
        )

    def _state_dict(self) -> dict:
        """Serialize current buffer state."""
        buf = self._buf
        priorities = [buf._sum_tree.get(i) for i in range(buf.capacity)]
        transitions = [
            _transition_to_dict(t) if t is not None else None
            for t in buf._transitions
        ]
        return _serialize_state(
            size=buf._size,
            write_pos=buf._write_pos,
            op_counter=buf._op_counter,
            max_prio=buf._max_prio,
            priorities=priorities,
            transitions=transitions,
        )

    def _apply_state(self, state: dict) -> None:
        """Restore buffer from a state dict."""
        size, write_pos, op_counter, max_prio, priorities, transitions = (
            _deserialize_state(state)
        )
        buf = self._make_fresh_buffer()
        buf._size = size
        buf._write_pos = write_pos
        buf._op_counter = op_counter
        buf._max_prio = max_prio
        for i, (p, t_dict) in enumerate(zip(priorities, transitions)):
            buf._sum_tree.update(i, p)
            buf._min_tree.update(i, p if p > 0 else buf._min_tree.INFINITY)
            if t_dict is not None:
                buf._transitions[i] = _dict_to_transition(t_dict)
        self._buf = buf

    def _recover_or_init(self):
        """Recover from any on-disk state or initialize fresh.

        An unreadable or structurally invalid state starts fresh, matching
        the original behaviour of this buffer.
        """
        state = recover_state(self.directory)
        self._buf = self._make_fresh_buffer()
        if state is not None:
            try:
                self._apply_state(state)
            except (KeyError, ValueError, TypeError):
                self._buf = self._make_fresh_buffer()
        return self._buf

    def _save_state(self) -> None:
        """Atomically save current state to state.json."""
        _save_state_file(self.directory, self._state_dict())

    def _durably_apply(self, operation_name: str, apply_fn) -> Any:
        """Execute an operation with crash-atomic durability (see ``durably_apply``)."""
        return durably_apply(self.directory, operation_name, self._state_dict, apply_fn)

    def insert(self, transition, td_error: Optional[float] = None) -> int:
        """Durably insert a transition."""
        def _apply():
            return self._buf.insert(transition, td_error)
        return self._durably_apply("insert", _apply)

    def update_priority(self, position: int, td_error: float) -> None:
        """Durably update a priority."""
        def _apply():
            self._buf.update_priority(position, td_error)
        self._durably_apply("update", _apply)

    def sample(self, batch_size: int):
        """Sample from buffer (read-only — no durability needed)."""
        return self._buf.sample(batch_size)

    @property
    def size(self) -> int:
        return self._buf.size

    @property
    def capacity(self) -> int:
        return self._buf.capacity


def _transition_to_dict(t) -> dict:
    from reservoir.buffer import Transition
    return {
        "state": t.state,
        "action": t.action,
        "reward": float(t.reward),
        "next_state": t.next_state,
        "done": bool(t.done),
    }


def _dict_to_transition(d: dict):
    from reservoir.buffer import Transition
    return Transition(
        state=d["state"],
        action=d["action"],
        reward=d["reward"],
        next_state=d["next_state"],
        done=d["done"],
    )
