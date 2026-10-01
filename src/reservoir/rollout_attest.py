"""
reservoir.rollout_attest — Writes RolloutBuffer events to an attestation log.

The buffer produces frozen events (``WriteEvent``, ``AdvanceResult``) and
sampled batches; this module turns them into the hash-chained records of
``reservoir.attest`` and, optionally, streams each record to a JSON-lines
file as it is produced. If the process dies, the file holds every record
written so far; the checker verifies such a prefix with
``--allow-truncated``, because the cut may fall between an
``advance_version`` and the evictions it requires.

Record order the checker relies on
----------------------------------
1. ``decay_config`` — first record, written at construction.
2. For each version change: ``advance_version``, then one ``evict`` with
   reason ``"stale"`` per expired entry, then at most one ``rebase``.
3. ``insert`` / ``update`` / ``evict`` mutations, each carrying the decay
   inputs ``(base_priority_int, entry_version, base_epoch)`` so the
   checker recomputes the leaf from the decay formula rather than
   trusting ``new_priority_int``.
4. One ``sample`` record per batch, unchanged from the classic buffer.

The buffer never emits a record out of this order; the tree's
``AdvanceResult`` already lists evictions before the rebase.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import IO, Optional, Union

from reservoir.attest import AttestationLog, make_sample_entry
from reservoir.decay import DecayParams
from reservoir.decayed_tree import AdvanceResult, WriteEvent

AttestTarget = Union[AttestationLog, str, Path, None]


class RolloutAttester:
    """Append buffer events to an ``AttestationLog``; optionally mirror to a file.

    Parameters
    ----------
    params : DecayParams
        Written as the ``decay_config`` record when attestation is on.
    reset_age_on_update : bool
        Also part of ``decay_config``; the checker uses it to validate the
        version carried by ``update`` records.
    target : AttestationLog | str | Path | None
        An existing log to append to, or a file path. With a path, a fresh
        ``AttestationLog`` is created and every record is also written to
        the file (one canonical JSON line each, flushed immediately). The
        file must not already exist: a chain starts at "genesis", so
        appending a second run to an old file would produce a log the
        checker rejects. ``FileExistsError`` otherwise.
        ``None`` disables attestation; every ``record_*`` call is a no-op.
    """

    def __init__(
        self, target: AttestTarget, params: DecayParams, reset_age_on_update: bool
    ) -> None:
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
        self._emit(self._log.append_decay_config(
            half_life=params.half_life,
            max_policy_age=params.max_policy_age,
            capacity=params.capacity,
            priority_bits=params.priority_bits,
            priority_frac_bits=params.priority_frac_bits,
            table_frac_bits=params.table_frac_bits,
            rebase_slack=params.rebase_slack,
            reset_age_on_update=reset_age_on_update,
        ))

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
            base_priority_int=event.base_priority_int,
            entry_version=event.entry_version,
            base_epoch=event.base_epoch,
            reason=event.reason,
        )
        self._emit(record)

    def record_advance(self, result: AdvanceResult, op_counter: int) -> None:
        """Record what ``advance`` did: version change, stale evictions, rebase.

        Nothing is written when the version did not change and nothing
        happened, so repeated ``sample()`` calls at the same version do not
        bloat the log.
        """
        if self._log is None:
            return
        if result.new_version != result.old_version:
            self._emit(self._log.append_advance_version(
                old_version=result.old_version,
                new_version=result.new_version,
                op_counter=op_counter,
            ))
        for event in result.evicted:
            self.record_write(event, op_counter)
        if result.rebase is not None:
            self._emit(self._log.append_rebase(
                old_base_epoch=result.rebase.old_base_epoch,
                new_base_epoch=result.rebase.new_base_epoch,
                root_total_before=result.rebase.root_total_before,
                root_total_after=result.rebase.root_total_after,
                op_counter=op_counter,
            ))

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
