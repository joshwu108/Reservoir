"""
reservoir.durable_rollout — Crash-atomic RolloutBuffer backed by a directory.

``DurableRolloutBuffer`` wraps ``RolloutBuffer`` so that every operation
is committed through the write-ahead protocol in ``durable.py``: an
intent record with the pre-state, the operation, a segment with the
post-state, then an atomic rename and directory fsync. If the process is
killed at any point, reopening the same directory recovers exactly the
state before or after the operation, never a mix. Stale evictions and a
rebase triggered inside ``add_group`` or ``sample`` are part of that one
operation, so a half-applied rebase cannot survive a crash.

Usage::

    buf = DurableRolloutBuffer("run-01/buffer", capacity=50_000, half_life=4,
                               max_policy_age=16, seed=0, attest="run-01/attest.jsonl")
    buf.add_group(...)            # committed before it returns
    batch = buf.sample(64, current_version=step)   # also committed: it advances counters
    # ... process dies ...
    buf = DurableRolloutBuffer("run-01/buffer", capacity=50_000, half_life=4,
                               max_policy_age=16, seed=0, attest="run-01/attest.jsonl")
    # same live entries, same counters, same next draw, same attestation chain

Costs and limits
----------------
- Each operation serialises the whole buffer (full-state snapshots, as in
  ``DurableBuffer``), so cost grows with buffer size. ``sample`` is an
  operation too: it advances the draw counter and may evict stale
  entries. An incremental write-ahead log is future work.
- The attestation log is part of the saved state. When ``attest`` is a
  path, the file is rewritten from the recovered state on reopen, so the
  file and the buffer can never disagree. ``attest`` must be a path or
  None; an in-memory ``AttestationLog`` cannot be recovered into.
- Construction parameters must match the saved state; a mismatch is a
  ``ValueError`` rather than a silent reinterpretation of the snapshot. A
  corrupt ``state.json`` is also an error, never a fresh start. Both are
  checked on a throwaway in-memory buffer first, so a failed reopen never
  touches the attestation file.
- An operation that raises is undone in memory as well as on disk, even
  if it had already changed the buffer (a ``sample`` that evicts stale
  entries and then finds nothing to draw, for example), so the two never
  diverge.
- Custom ``is_success`` predicates are not supported: a callable cannot
  be saved, and silently reverting to the default would change pass
  rates. Rollout metadata must be JSON-serialisable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

from reservoir.attest import AttestationLog
from reservoir.decayed_tree import AdvanceResult
from reservoir.durable import CorruptStateError, durably_apply, recover_state
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBatch, RolloutBuffer
from reservoir.rollout_snapshot import _require_json_round_trip

_LOAD_ERRORS = (KeyError, TypeError, ValueError, IndexError)


class DurableRolloutBuffer:
    """``RolloutBuffer`` whose every operation is crash-atomic on disk.

    Parameters
    ----------
    directory : str | Path
        Where state, intent and segment files live. Created if missing.
    attest : str | Path | None
        Attestation file path, or None. See the module docstring.
    **buffer_kwargs
        Everything ``RolloutBuffer`` accepts except ``attest`` and
        ``attest_overwrite``: ``capacity``, ``priority``, ``half_life``,
        ``max_policy_age``, ``alpha``, ``beta``, ``seed``, ``buffer_id``,
        ``reset_age_on_update``, ``priority_bits``, ``priority_frac_bits``,
        ``rebase_slack``.

    Raises
    ------
    ValueError
        A saved state exists but was written with different parameters,
        or is corrupt, or carries an attestation log while ``attest`` is
        None (or the reverse).
    """

    def __init__(
        self,
        directory: Union[str, Path],
        attest: Union[str, Path, None] = None,
        **buffer_kwargs: Any,
    ) -> None:
        if attest is not None and not isinstance(attest, (str, Path)):
            raise TypeError(
                "DurableRolloutBuffer attest must be a file path or None; an in-memory "
                "AttestationLog cannot be recovered after a crash"
            )
        for forbidden in ("attest", "attest_overwrite"):
            if forbidden in buffer_kwargs:
                raise TypeError(f"{forbidden} is managed by DurableRolloutBuffer")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._buffer_kwargs = dict(buffer_kwargs)
        self._attest = Path(attest) if attest is not None else None
        self._buf = self._open()

    def _open(self) -> RolloutBuffer:
        """Construct a fresh buffer and load the committed state, if any."""
        try:
            state = recover_state(self.directory, strict=True)
        except CorruptStateError as exc:
            raise ValueError(f"saved state in {self.directory} is corrupt: {exc}") from exc
        if state is None:
            buf = self._new_buffer()
            if buf.attestation_log is not None:
                # Persist the decay_config record so the log and state agree
                # even if the process dies before the first operation.
                durably_apply(self.directory, "open", buf.state_dict, lambda: None)
            return buf
        self._validate_state(state)
        return self._buffer_from_state(state)

    def _new_buffer(self) -> RolloutBuffer:
        """An empty buffer with this wrapper's parameters, writing to the attestation file."""
        return RolloutBuffer(attest=self._attest, attest_overwrite=True, **self._buffer_kwargs)

    def _validate_state(self, state: dict) -> None:
        """Load ``state`` into a throwaway in-memory buffer so a bad snapshot or
        mismatched parameters are rejected before the attestation file is opened
        for writing (which would truncate it)."""
        if (state.get("attestation") is None) != (self._attest is None):
            raise ValueError(
                "attestation setting differs from the saved state: "
                + ("the state has a log, pass attest=<path>" if self._attest is None
                   else "the state has no log, pass attest=None")
            )
        probe = RolloutBuffer(attest=AttestationLog(), **self._buffer_kwargs)
        try:
            probe.load_state_dict(state)
        except _LOAD_ERRORS as exc:
            raise ValueError(
                f"saved state in {self.directory} could not be loaded: {exc}"
            ) from exc

    def _buffer_from_state(self, state: dict) -> RolloutBuffer:
        """A new attesting buffer holding ``state``; the mirror file is rewritten from it."""
        buf = self._new_buffer()
        try:
            buf.load_state_dict(state)
        except _LOAD_ERRORS:
            buf.close()
            raise
        return buf

    def _rollback(self, pre_state: dict) -> None:
        """Undo an aborted operation in memory by reloading the pre-state."""
        self._buf.close()
        self._buf = self._buffer_from_state(pre_state)

    def _apply(self, name: str, fn) -> Any:
        """Run ``fn`` under the commit protocol; roll memory back if it aborts."""
        return durably_apply(self.directory, name, self._buf.state_dict, fn, on_abort=self._rollback)

    # -- durable operations ------------------------------------------------

    def add_group(
        self, prompt_id: str, model_version: int, rollouts: Sequence[Rollout]
    ) -> tuple[int, ...]:
        """Durably store a prompt group. No ``is_success``: predicates cannot be saved."""
        for k, r in enumerate(rollouts):
            _require_json_round_trip(dict(getattr(r, "metadata", {})), f"rollout {k} of prompt {prompt_id!r}")
        return self._apply("add_group", lambda: self._buf.add_group(prompt_id, model_version, rollouts))

    def sample(self, batch_size: int, current_version: Optional[int] = None) -> RolloutBatch:
        """Durably sample: the draw counter and any stale evictions are committed."""
        return self._apply("sample", lambda: self._buf.sample(batch_size, current_version))

    def update_priorities(self, indices: Sequence[int], raw_scores: Sequence[float]) -> None:
        """Durably re-score live entries; see ``RolloutBuffer.update_priorities``."""
        self._apply("update_priorities", lambda: self._buf.update_priorities(indices, raw_scores))

    def advance(self, current_version: int) -> AdvanceResult:
        """Durably move to a newer version, committing any evictions and rebase."""
        return self._apply("advance", lambda: self._buf.advance(current_version))

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
    def attestation_log(self):
        """The in-memory attestation log, or None."""
        return self._buf.attestation_log

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
        """The current snapshot, identical to what the last commit wrote to disk."""
        return self._buf.state_dict()

    def close(self) -> None:
        """Close the attestation file. State is already committed; nothing is flushed here."""
        self._buf.close()

    def __enter__(self) -> "DurableRolloutBuffer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"DurableRolloutBuffer({str(self.directory)!r}, {self._buf!r})"
