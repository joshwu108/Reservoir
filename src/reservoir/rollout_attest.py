"""
reservoir.rollout_attest — Writes RolloutBuffer events to an attestation log.

The buffer produces frozen events (``WriteEvent``, ``AdvanceResult``) and
sampled batches; this module turns them into the hash-chained records of
``reservoir.attest`` and, optionally, streams each record to a JSON-lines
file as it is produced so a crash still leaves a verifiable prefix.

What gets written today
-----------------------
Only record types the independent checker (``checker/verify.py``) already
understands: ``insert`` / ``update`` / ``evict`` mutations carrying the
old and new leaf values, and one ``sample`` record per batch. A rebase
shifts every leaf at once; it is written as one ``update`` record per
live leaf, which is verbose but lets the checker replay it unchanged.

What is planned
---------------
A ``decay_config`` record at the start of the log, an ``advance_version``
record per version change, a single ``rebase`` record in place of the
per-leaf updates, and ``(q, entry_version, base_epoch)`` on every
mutation so the checker can recompute each leaf from the decay formula
instead of trusting the recorded value. The events this module receives
already carry all of that; only the record formats and the checker need
to change.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import IO, Optional, Union

from reservoir.attest import AttestationLog, make_sample_entry
from reservoir.decayed_tree import AdvanceResult, WriteEvent

AttestTarget = Union[AttestationLog, str, Path, None]


class RolloutAttester:
    """Append buffer events to an ``AttestationLog``; optionally mirror to a file.

    Parameters
    ----------
    target : AttestationLog | str | Path | None
        An existing log to append to, or a file path. With a path, a fresh
        ``AttestationLog`` is created and every record is also written to
        the file (one canonical JSON line each, flushed immediately). The
        file must not already exist: a chain starts at "genesis", so
        appending a second run to an old file would produce a log the
        checker rejects. ``FileExistsError`` otherwise.
        ``None`` disables attestation; every ``record_*`` call is a no-op.
    """

    def __init__(self, target: AttestTarget) -> None:
        self._log: Optional[AttestationLog] = None
        self._file: Optional[IO[str]] = None
        if target is None:
            return
        if isinstance(target, AttestationLog):
            self._log = target
        elif isinstance(target, (str, Path)):
            self._log = AttestationLog()
            self._file = open(Path(target), "x", encoding="utf-8")
        else:
            raise TypeError(
                "attest must be an AttestationLog, a path, or None; "
                f"got {type(target).__name__}"
            )

    @property
    def enabled(self) -> bool:
        return self._log is not None

    @property
    def log(self) -> Optional[AttestationLog]:
        return self._log

    def record_write(self, event: WriteEvent, op_counter: int) -> None:
        """One mutation record for an insert, update or evict."""
        if self._log is None:
            return
        record = self._log.append_mutation(
            op=event.op,
            index=event.position,
            old_priority_int=event.old_leaf,
            new_priority_int=event.new_leaf,
            op_counter=op_counter,
        )
        self._emit(record)

    def record_advance(self, result: AdvanceResult, op_counter: int) -> None:
        """Record what ``advance`` did: stale evictions first, then the rebase.

        The rebase is written as one ``update`` per live leaf (old value ->
        shifted value) in position order. The checker then sees an ordinary
        sequence of leaf changes and its replayed total matches the tree.
        """
        if self._log is None:
            return
        for event in result.evicted:
            self.record_write(event, op_counter)
        if result.rebase is not None:
            for position, old_leaf, new_leaf in result.rebase.shifted:
                record = self._log.append_mutation(
                    op="update",
                    index=position,
                    old_priority_int=old_leaf,
                    new_priority_int=new_leaf,
                    op_counter=op_counter,
                )
                self._emit(record)

    def record_sample(
        self,
        op_counter: int,
        root_total: int,
        indices: tuple[int, ...],
        draw_integers: tuple[int, ...],
        priorities: tuple[int, ...],
        is_weights: tuple[Fraction, ...],
    ) -> None:
        """One sample record covering the whole batch."""
        if self._log is None:
            return
        entries = [
            make_sample_entry(
                leaf_index=indices[k],
                draw_int=draw_integers[k],
                priority_int=priorities[k],
                root_total=root_total,
                is_weight=is_weights[k],
            )
            for k in range(len(indices))
        ]
        record = self._log.append_sample(
            op_counter=op_counter, root_total=root_total, samples=entries
        )
        self._emit(record)

    def _emit(self, record: dict) -> None:
        """Mirror one record to the file as a canonical JSON line (same form as the log)."""
        if self._file is None:
            return
        self._file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the mirror file, if any. Safe to call more than once."""
        if self._file is not None:
            self._file.close()
            self._file = None
