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
   trusting ``new_priority_int``. An ``insert`` also carries the
   ``content_digest`` of the stored example and its ``source`` tag when
   the group has one (``record_insert``).
4. One ``sample`` record per batch, unchanged from the classic buffer.

The buffer never emits a record out of this order; the tree's
``AdvanceResult`` already lists evictions before the rebase.

Manifest
--------
With ``manifest=`` every insert also writes the opening of its digest
(prompt id, tokens, reward, source) to a manifest; see
``rollout_manifest.py``. The manifest requires attestation to be on,
because without the log there is nothing for it to open. The buffer calls
``prepare_inserts`` before it mutates anything, so digest and manifest
line construction (the only computation on this path) cannot fail
half-way through a group; ``record_insert`` then only appends the log
record and writes the line. A crash between those two leaves a
commitment without an opening, which the checker reports when given the
manifest, and the durable buffer repairs both files from its committed
state through ``restore``.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import IO, NamedTuple, Optional, Union

from reservoir.attest import AttestationLog, make_sample_entry
from reservoir.decay import DecayParams
from reservoir.decayed_tree import AdvanceResult, WriteEvent
from reservoir.rollout import RolloutGroup
from reservoir.rollout_manifest import ManifestWriter, manifest_record, validate_manifest_records

AttestTarget = Union[AttestationLog, str, Path, None]
ManifestTarget = Union[ManifestWriter, str, Path, None]


class DrawConfig(NamedTuple):
    """What the checker needs to recompute every draw and importance weight."""

    seed: int
    buffer_id: int
    alpha: float
    beta: float


class PreparedInsert(NamedTuple):
    """What ``record_insert`` needs for one rollout, computed before any mutation."""

    content_digest: str
    source: Optional[str]
    manifest_line: Optional[dict]   # without ``index``; filled in at record time


def _canonical_line(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"


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
    overwrite : bool
        Open the attestation and manifest files with ``"w"`` instead of
        ``"x"``. The durable buffer uses this because it rewrites both
        files from the recovered state, which is the source of truth.
    manifest : ManifestWriter | str | Path | None
        Also write the opening of every insert's content digest, to this
        file or into this in-memory writer. Requires ``target``;
        ``ValueError`` otherwise.
    draw : DrawConfig, optional
        The buffer's seed, buffer id, alpha and beta, written into
        ``decay_config`` so the checker can recompute every draw and
        importance weight. ``RolloutBuffer`` always passes it.

    If construction fails after a file was opened, the file is closed
    again before the error propagates.
    """

    def __init__(
        self,
        target: AttestTarget,
        params: DecayParams,
        reset_age_on_update: bool,
        overwrite: bool = False,
        manifest: ManifestTarget = None,
        draw: Optional[DrawConfig] = None,
    ) -> None:
        self._log: Optional[AttestationLog] = None
        self._file: Optional[IO[str]] = None
        self._manifest: Optional[ManifestWriter] = None
        if target is None:
            if manifest is not None:
                raise ValueError("manifest requires attestation: pass attest=<path or log> as well")
            return
        if not isinstance(target, (AttestationLog, str, Path)):
            raise TypeError(
                "attest must be an AttestationLog, a path, or None; "
                f"got {type(target).__name__}"
            )
        if manifest is not None and not isinstance(manifest, (ManifestWriter, str, Path)):
            raise TypeError(
                f"manifest must be a ManifestWriter, a path, or None; got {type(manifest).__name__}"
            )
        try:
            self._open(target, overwrite, manifest)
            self._emit(self._log.append_decay_config(  # type: ignore[union-attr]
                half_life=params.half_life,
                max_policy_age=params.max_policy_age,
                capacity=params.capacity,
                priority_bits=params.priority_bits,
                priority_frac_bits=params.priority_frac_bits,
                table_frac_bits=params.table_frac_bits,
                rebase_slack=params.rebase_slack,
                reset_age_on_update=reset_age_on_update,
                **(draw._asdict() if draw is not None else {}),
            ))
        except BaseException:
            self.close()
            raise

    def _open(self, target: Union[AttestationLog, str, Path], overwrite: bool, manifest: ManifestTarget) -> None:
        """Open the log (and its file) first, the manifest last, so a refused
        log path never leaves a freshly created or truncated manifest behind."""
        if isinstance(target, AttestationLog):
            self._log = target
        else:
            self._log = AttestationLog()
            # "x" refuses an existing file so two runs never share one chain.
            self._file = open(Path(target), "w" if overwrite else "x", encoding="utf-8")
        if isinstance(manifest, ManifestWriter):
            self._manifest = manifest
        elif manifest is not None:
            self._manifest = ManifestWriter(manifest, overwrite=overwrite)

    @property
    def enabled(self) -> bool:
        """False when the buffer was built with ``attest=None``; every record call is then a no-op."""
        return self._log is not None

    @property
    def log(self) -> Optional[AttestationLog]:
        """The in-memory log being appended to, or None when disabled."""
        return self._log

    @property
    def has_manifest(self) -> bool:
        """True when inserts are also written to a manifest."""
        return self._manifest is not None

    @property
    def manifest_records(self) -> list[dict]:
        """Manifest lines written so far (a copy); empty without a manifest."""
        return self._manifest.records if self._manifest is not None else []

    def record_write(self, event: WriteEvent, op_counter: int) -> None:
        """One mutation record for an update or evict (or an insert without content)."""
        log = self._log
        if log is None:
            return
        self._emit(self._mutation(log, event, op_counter))

    def prepare_inserts(self, group: RolloutGroup, op_counter: int) -> tuple[PreparedInsert, ...]:
        """Digest and manifest line for every rollout of ``group``, computed up front.

        Returns an empty tuple when attestation is off. Call this before the
        first tree write of ``add_group`` so nothing on the insert path can
        raise after the buffer has started mutating.
        """
        if self._log is None:
            return ()
        digests = group.content_digests
        lines: list[Optional[dict]] = [None] * group.size
        if self._manifest is not None:
            lines = [
                manifest_record(
                    op_counter=op_counter, index=0, prompt_id=group.prompt_id, source=group.source,
                    tokens=r.tokens, reward=r.reward, entry_version=group.model_version,
                )
                for r in group.rollouts
            ]
        return tuple(PreparedInsert(d, group.source, line) for d, line in zip(digests, lines))

    def record_insert(self, event: WriteEvent, op_counter: int, prepared: PreparedInsert) -> None:
        """One insert record carrying the content digest and source; plus its manifest line."""
        log = self._log
        if log is None:
            return
        self._emit(self._mutation(
            log, event, op_counter, content_digest=prepared.content_digest, source=prepared.source
        ))
        if self._manifest is not None and prepared.manifest_line is not None:
            self._manifest.write({**prepared.manifest_line, "index": event.position})

    @staticmethod
    def _mutation(
        log: AttestationLog,
        event: WriteEvent,
        op_counter: int,
        content_digest: Optional[str] = None,
        source: Optional[str] = None,
    ) -> dict:
        return log.append_mutation(
            op=event.op,
            index=event.position,
            old_priority_int=event.old_leaf,
            new_priority_int=event.new_leaf,
            op_counter=op_counter,
            base_priority_int=event.base_priority_int,
            entry_version=event.entry_version,
            base_epoch=event.base_epoch,
            reason=event.reason,
            content_digest=content_digest,
            source=source,
        )

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

    def restore(self, records: list[dict], manifest_records: Optional[list[dict]] = None) -> None:
        """Replace the log (and manifest) with recovered records and rewrite the files.

        Everything is validated before anything is replaced: the chain
        itself (``AttestationLog.restore``), every manifest record
        (``validate_manifest_records``), and that the manifest has exactly
        one line, with the same digest, for every insert record that
        carries a ``content_digest``. A failure leaves log, manifest and
        files unchanged.

        With attestation off, a non-empty ``records`` or
        ``manifest_records`` is an error: the saved state came from a
        buffer that was attesting, and silently dropping its log would
        break the chain for later records.
        """
        manifest_records = manifest_records or []
        if self._log is None:
            if records or manifest_records:
                raise ValueError(
                    "saved state carries an attestation log; reopen with attest=<path>"
                )
            return
        if self._manifest is None:
            if manifest_records:
                raise ValueError("saved state carries a manifest; reopen with manifest=<path>")
            validated: list[dict] = []
        else:
            validated = validate_manifest_records(manifest_records)
            _require_manifest_matches_log(records, validated)
        self._log.restore(records)      # validates the whole chain before replacing anything
        if self._file is not None:
            self._file.seek(0)
            self._file.truncate()
            for record in records:
                self._file.write(_canonical_line(record))
            self._file.flush()
        if self._manifest is not None:
            self._manifest.restore(validated)

    def _emit(self, record: dict) -> None:
        """Mirror one record to the file as a canonical JSON line (same form as the log)."""
        if self._file is None:
            return
        self._file.write(_canonical_line(record))
        self._file.flush()

    def close(self) -> None:
        """Close the mirror and manifest files, if any. Safe to call more than once."""
        try:
            if self._file is not None:
                self._file.close()
                self._file = None
        finally:
            if self._manifest is not None:
                self._manifest.close()


def _require_manifest_matches_log(records: list[dict], manifest: list[dict]) -> None:
    """The manifest must open exactly the content-bearing inserts of the log, in order,
    with the same digest, source and entry version."""
    expected = [
        (r.get("op_counter"), r.get("index"), r.get("content_digest"), r.get("source"),
         int(r["entry_version"]) if "entry_version" in r else None)
        for r in records
        if isinstance(r, dict) and r.get("op") == "insert" and "content_digest" in r
    ]
    actual = [
        (m["op_counter"], m["index"], m["content_digest"], m["source"], m["entry_version"])
        for m in manifest
    ]
    if len(expected) != len(actual):
        raise ValueError(
            f"saved manifest does not match the saved log: {len(actual)} manifest lines for "
            f"{len(expected)} insert records with content digests"
        )
    for i, (want, got) in enumerate(zip(expected, actual)):
        if want[:4] != got[:4] or (want[4] is not None and want[4] != got[4]):
            raise ValueError(f"saved manifest line {i} does not match the saved log's insert record")
