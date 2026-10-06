"""
checker.content — Content commitments: which example each buffer slot held.

This module imports nothing from the rest of the reservoir package. It re-derives the content
digest from its definition (docs/design.md §10) with the standard library,
so agreement with the library is evidence and not tautology.

What a content-bearing log claims
---------------------------------
Every ``insert`` record carries ``content_digest``, the BLAKE2b-256 digest
(personalisation ``b"rollout-content\\x00"``) of canonical JSON::

    {"prompt_id": <str>, "reward": <float.hex() of the reward>, "tokens": [<int>, ...]}

serialised with sorted keys, no spaces, ASCII escapes for non-ASCII
characters, UTF-8. Optionally it also carries ``source``, the caller's tag
for where the prompt came from. ``update`` and ``evict`` records carry
neither: they refer to a slot whose example the replay already knows. An
``evict`` with reason ``"quarantine"`` carries two texts, ``predicate`` and
``note``; this module checks their shape (non-empty, printable, bounded)
and resolves the evicted slot to its example (``quarantines``) so the
transcript can report what a quarantined example reached. The texts
themselves are the operator's statement; nothing in the log verifies them.

A log either has digests on every insert or on none. Mixing is rejected,
because a verifier could not then say what a sample of an undigested slot
was a sample of.

What the replay tracks
----------------------
``ContentState`` follows inserts and evicts to know, at every sample
record, which example each sampled slot held (``samples``), and keeps the
full insert history (``history``). ``checker.transcript`` turns those into
exposure counts, per-source mixtures and quota verdicts.

The manifest
------------
A manifest (``rollout_manifest.py`` in the library) is one JSON line per
insert with the opening of its digest: ``op_counter``, ``index``,
``content_digest``, ``prompt_id``, ``source``, ``tokens``, ``reward_hex``,
``entry_version``, plus an optional ``rewards`` object (per-reward-function
values, printable names to finite numbers; numeric only and outside the
digest, so reported rather than verified). ``check_manifest`` requires that the lines, in order,
are exactly the log's content-bearing inserts, that each line's digest
recomputes from its own prompt, tokens and reward, and that it equals the
digest the log committed to. Without a manifest the log is still fully
verified; the digests are then commitments whose openings the verifier
does not have.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

from reservoir_checker.decay_replay import CheckerError

_PERSON = b"rollout-content\x00"
_DIGEST_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")
_MAX_SOURCE_LENGTH = 256   # mirrors the library's documented bound; not imported
_MAX_PREDICATE_LENGTH = 1024   # mirrors reservoir.rollout_quarantine.MAX_PREDICATE_LENGTH; not imported

MANIFEST_KEYS = (
    "op_counter", "index", "content_digest", "prompt_id", "source",
    "tokens", "reward_hex", "entry_version",
)
MANIFEST_OPTIONAL_KEYS = ("rewards",)


# ---------------------------------------------------------------------------
# The digest, from the definition
# ---------------------------------------------------------------------------

def _require_prompt_id(prompt_id: object, where: str) -> str:
    if not isinstance(prompt_id, str) or not prompt_id:
        raise CheckerError(f"{where}: prompt_id must be a non-empty string, got {prompt_id!r}")
    return prompt_id


def _require_tokens(tokens: object, where: str) -> list[int]:
    if not isinstance(tokens, list) or not tokens:
        raise CheckerError(f"{where}: tokens must be a non-empty list of integers")
    for t in tokens:
        if isinstance(t, bool) or not isinstance(t, int) or t < 0:
            raise CheckerError(f"{where}: tokens must be non-negative integers, got {t!r}")
    return tokens


def _require_reward_hex(reward_hex: object, where: str) -> str:
    """The reward in ``float.hex()`` spelling. Only that exact spelling is
    accepted so two verifiers cannot disagree on the preimage bytes."""
    if not isinstance(reward_hex, str):
        raise CheckerError(f"{where}: reward_hex must be a string, got {reward_hex!r}")
    try:
        value = float.fromhex(reward_hex)
    except (ValueError, OverflowError) as exc:
        raise CheckerError(f"{where}: reward_hex is not a hexadecimal float: {exc}") from exc
    if value != value or value in (float("inf"), float("-inf")) or value.hex() != reward_hex:
        raise CheckerError(f"{where}: reward_hex must be the canonical float.hex() spelling of a finite value")
    return reward_hex


def content_digest(prompt_id: object, tokens: object, reward_hex: object, where: str = "content") -> str:
    """Recompute a content digest from its three inputs; CheckerError on malformed input."""
    canonical = json.dumps(
        {
            "prompt_id": _require_prompt_id(prompt_id, where),
            "reward": _require_reward_hex(reward_hex, where),
            "tokens": _require_tokens(tokens, where),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.blake2b(canonical, digest_size=32, person=_PERSON).hexdigest()


# ---------------------------------------------------------------------------
# Record fields
# ---------------------------------------------------------------------------

def _int_field(record: dict, name: str, where: str) -> int:
    """A non-negative JSON integer field (not bool, not float) or a decimal digit string.

    Anything else is a CheckerError. Floats are refused rather than
    truncated: ``0.7`` and ``0`` must not verify as the same history.
    """
    value = record.get(name)
    if isinstance(value, bool) or value is None or isinstance(value, float):
        raise CheckerError(f"{where}: {name} must be a non-negative integer, got {value!r}")
    if isinstance(value, str) and not (value.isascii() and value.isdigit()):
        raise CheckerError(f"{where}: {name} must be a non-negative integer, got {value!r}")
    if not isinstance(value, (int, str)):
        raise CheckerError(f"{where}: {name} must be a non-negative integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CheckerError(f"{where}: {name} must be a non-negative integer, got {value!r}") from exc
    if parsed < 0:
        raise CheckerError(f"{where}: {name} must be a non-negative integer, got {value!r}")
    return parsed


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == _DIGEST_LENGTH and set(value) <= _HEX_DIGITS


def _is_source(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _MAX_SOURCE_LENGTH
        and value.isprintable()
        and bool(value.strip())
    )


def _is_text(value: object, limit: int) -> bool:
    return isinstance(value, str) and 0 < len(value) <= limit and value.isprintable() and bool(value.strip())


def quarantine_fields(record: dict, idx: int) -> Optional[tuple[str, str]]:
    """Validate and return ``(predicate, note)`` of a quarantine evict, or None for any other record.

    Both texts are required with reason ``"quarantine"`` and forbidden
    otherwise; each must be a non-empty printable string within its bound.
    """
    has_text = "predicate" in record or "note" in record
    if record.get("reason") != "quarantine":
        if has_text:
            raise CheckerError(f"Record {idx}: predicate and note are only valid on an evict with reason 'quarantine'")
        return None
    predicate, note = record.get("predicate"), record.get("note")
    if not _is_text(predicate, _MAX_PREDICATE_LENGTH):
        raise CheckerError(
            f"Record {idx}: a quarantine evict needs predicate, a printable string of 1..{_MAX_PREDICATE_LENGTH} "
            f"characters, got {predicate!r}"
        )
    if not _is_text(note, _MAX_SOURCE_LENGTH):
        raise CheckerError(
            f"Record {idx}: a quarantine evict needs note, a printable string of 1..{_MAX_SOURCE_LENGTH} "
            f"characters, got {note!r}"
        )
    return predicate, note  # type: ignore[return-value]


def _require_rewards(value: object, where: str) -> None:
    """The optional ``rewards`` of a manifest line: printable names to finite JSON numbers (may be empty)."""
    if not isinstance(value, dict):
        raise CheckerError(f"{where}: rewards must be an object of reward function name to number")
    for name, number in value.items():
        if not _is_text(name, _MAX_SOURCE_LENGTH):
            raise CheckerError(f"{where}: rewards names must be printable strings of 1..{_MAX_SOURCE_LENGTH} characters")
        try:
            as_float = float(number) if not isinstance(number, bool) and isinstance(number, (int, float)) else None
        except OverflowError:
            as_float = None
        if as_float is None or as_float != as_float or as_float in (float("inf"), float("-inf")):
            raise CheckerError(f"{where}: rewards[{name!r}] must be a finite number, got {number!r}")


def content_fields(record: dict, idx: int) -> tuple[Optional[str], Optional[str]]:
    """Validate and return ``(content_digest, source)`` of a mutation record, or ``(None, None)``."""
    digest = record.get("content_digest")
    source = record.get("source")
    if "content_digest" not in record:
        if "source" in record:
            raise CheckerError(f"Record {idx}: source requires content_digest")
        return None, None
    if record.get("op") != "insert":
        raise CheckerError(f"Record {idx}: content_digest is only valid on insert records")
    if not _is_digest(digest):
        raise CheckerError(
            f"Record {idx}: content_digest must be {_DIGEST_LENGTH} lowercase hex characters, got {digest!r}"
        )
    if "source" in record and not _is_source(source):
        raise CheckerError(
            f"Record {idx}: source must be a printable string of 1..{_MAX_SOURCE_LENGTH} characters, got {source!r}"
        )
    return digest, source


# ---------------------------------------------------------------------------
# Replay state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommittedInsert:
    """One insert record's commitment, in log order."""

    record_index: int
    op_counter: int
    index: int
    content_digest: str
    source: Optional[str]
    entry_version: Optional[int]    # None in a log without decay fields


@dataclass(frozen=True)
class ResolvedSample:
    """One drawn rollout, resolved to the example its slot held at that moment."""

    record_index: int
    position_in_batch: int
    op_counter: int
    leaf_index: int
    content_digest: Optional[str]   # None in a log without content digests
    source: Optional[str]
    probability: Fraction
    is_weight: Fraction


@dataclass(frozen=True)
class WitnessedRow:
    """One training-batch row the witness says holds a replayed draw."""

    record_index: int
    step: int
    sample_op_counter: int
    row: int
    draw: int
    content_digest: str


@dataclass(frozen=True)
class QuarantinedSlot:
    """One quarantine evict, resolved to the example its slot held."""

    record_index: int
    op_counter: int
    index: int
    content_digest: Optional[str]   # None in a log without content digests
    source: Optional[str]
    predicate: str
    note: str


@dataclass
class ContentState:
    """Which example each slot holds, the insert history, every resolved sample,
    the batch witnesses that bind draws to training-batch rows, and the
    quarantine evictions resolved to their examples.

    ``has_content`` is ``None`` until the first insert decides whether this
    log carries digests; after that every insert must agree.
    """

    has_content: Optional[bool] = None
    slots: dict[int, tuple[str, Optional[str]]] = field(default_factory=dict)
    history: list[CommittedInsert] = field(default_factory=list)
    samples: list[ResolvedSample] = field(default_factory=list)
    manifest_matched: int = 0
    witnesses: list[dict] = field(default_factory=list)          # one entry per batch record
    witnessed_rows: list[WitnessedRow] = field(default_factory=list)
    telemetry: list = field(default_factory=list)                 # TelemetryPoint per telemetry record
    quarantines: list[QuarantinedSlot] = field(default_factory=list)
    _samples_by_op: dict[int, list[ResolvedSample]] = field(default_factory=dict)
    _witnessed_ops: set[int] = field(default_factory=set)
    last_sample_op: Optional[int] = None    # op_counter of the latest sample record; they must increase

    def on_mutation(self, record: dict, idx: int) -> None:
        """Called after verify.py accepted the mutation's tree effect (so ``index`` is valid)."""
        digest, source = content_fields(record, idx)
        texts = quarantine_fields(record, idx)
        op = record["op"]
        where = f"Record {idx}"
        if op == "insert":
            self._note_insert_style(digest is not None, idx)
            if digest is not None:
                has_version = "entry_version" in record
                self.on_insert(
                    idx, pos=record["index"], digest=digest, source=source,
                    op_counter=_int_field(record, "op_counter", where),
                    entry_version=_int_field(record, "entry_version", where) if has_version else None,
                )
        elif op == "evict":
            if texts is not None:
                held = self.slots.get(record["index"], (None, None))
                self.quarantines.append(QuarantinedSlot(
                    idx, _int_field(record, "op_counter", where), record["index"], held[0], held[1], *texts,
                ))
            self.on_evict(idx, pos=record["index"])

    def _note_insert_style(self, with_digest: bool, idx: int) -> None:
        if self.has_content is None:
            self.has_content = with_digest
        elif self.has_content != with_digest:
            raise CheckerError(
                f"Record {idx}: insert {'without' if self.has_content else 'with'} content_digest "
                f"in a log whose other inserts {'carry one' if self.has_content else 'do not'}"
            )

    def on_insert(
        self, idx: int, *, pos: int, digest: str, source: Optional[str], op_counter: int,
        entry_version: Optional[int],
    ) -> None:
        if self.has_content is None:
            self.has_content = True
        if pos in self.slots:
            raise CheckerError(f"Record {idx}: insert into slot {pos} that still holds a committed example")
        self.slots[pos] = (digest, source)
        self.history.append(CommittedInsert(idx, op_counter, pos, digest, source, entry_version))

    def on_evict(self, idx: int, *, pos: int) -> None:
        if self.has_content and pos not in self.slots:
            raise CheckerError(f"Record {idx}: evict of slot {pos} that holds no committed example")
        self.slots.pop(pos, None)

    def on_sample(self, idx: int, record: dict) -> None:
        """Resolve every drawn slot to its committed example.

        ``verify.py`` has already checked that every draw maps to its
        ``leaf_index`` and that the probability and weight fields are
        well-formed reduced fractions; this reads the same fields strictly.
        """
        where = f"Record {idx}"
        op_counter = _int_field(record, "op_counter", where)
        if self.last_sample_op is not None and op_counter <= self.last_sample_op:
            raise CheckerError(
                f"{where}: sample op_counter {op_counter} does not increase past the previous sample's "
                f"{self.last_sample_op}"
            )
        self.last_sample_op = op_counter
        samples = record.get("samples")
        if not isinstance(samples, list):
            raise CheckerError(f"{where}: samples must be a list")
        for k, s in enumerate(samples):
            at = f"{where}, sample {k}"
            if not isinstance(s, dict):
                raise CheckerError(f"{at}: not an object")
            pos = _int_field(s, "leaf_index", at)
            digest, source = None, None
            if self.has_content:
                if pos not in self.slots:
                    raise CheckerError(f"{at}: slot {pos} holds no committed example")
                digest, source = self.slots[pos]
            resolved = ResolvedSample(
                record_index=idx, position_in_batch=k, op_counter=op_counter, leaf_index=pos,
                content_digest=digest, source=source,
                probability=Fraction(_int_field(s, "prob_num", at), _int_field(s, "prob_den", at)),
                is_weight=Fraction(_int_field(s, "is_weight_num", at), _int_field(s, "is_weight_den", at)),
            )
            self.samples.append(resolved)
            self._samples_by_op.setdefault(op_counter, []).append(resolved)

    def on_batch(self, idx: int, record: dict) -> None:
        """A batch witness: every replaced row must hold exactly the example its draw selected.

        The referenced sample record must be the latest one (a witness
        describes the batch just built) and be witnessed at most once;
        placed and declined draws must cover its draws exactly, every row
        must be inside the batch and used once, and each row's content
        digest must equal the digest the draw resolved to.
        """
        where = f"Record {idx}"
        if not self.has_content:
            raise CheckerError(f"{where}: a batch witness needs content digests on the log's inserts")
        op = _strict_int(record, "sample_op_counter", where)
        if op != self.last_sample_op:
            raise CheckerError(f"{where}: batch witness names sample {op} but the latest sample is {self.last_sample_op}")
        if op in self._witnessed_ops:
            raise CheckerError(f"{where}: sample {op} already has a batch witness")
        draws = self._samples_by_op[op]
        step = _int_field(record, "step", where)
        if not isinstance(record.get("step"), str):
            raise CheckerError(f"{where}: step must be a decimal string")
        batch_rows = _strict_int(record, "batch_rows", where)
        if batch_rows > MAX_BATCH_ROWS:
            raise CheckerError(f"{where}: batch_rows {batch_rows} exceeds the checker's limit of {MAX_BATCH_ROWS}")
        replaced, declined = _witness_lists(record, where, len(draws), op)
        if not _is_digest(record.get("tensor_digest")):
            raise CheckerError(f"{where}: tensor_digest must be {_DIGEST_LENGTH} lowercase hex characters")
        rows_seen: set[int] = set()
        draws_seen: set[int] = set(declined)
        for k, entry in enumerate(replaced):
            row, draw = self._check_witness_entry(entry, f"{where}, replaced {k}", batch_rows, draws, rows_seen, draws_seen)
            self.witnessed_rows.append(WitnessedRow(idx, step, op, row, draw, draws[draw].content_digest))
        self._witnessed_ops.add(op)
        self.witnesses.append({"record_index": idx, "step": step, "sample_op_counter": op,
                               "batch_rows": batch_rows, "replaced": len(replaced), "declined": list(declined),
                               "tensor_digest": record["tensor_digest"]})

    @staticmethod
    def _check_witness_entry(entry: object, at: str, batch_rows: int, draws: list, rows_seen: set,
                             draws_seen: set) -> tuple[int, int]:
        if not isinstance(entry, dict):
            raise CheckerError(f"{at}: not an object")
        row = _strict_int(entry, "row", at)
        draw = _strict_int(entry, "draw", at)
        if row >= batch_rows:
            raise CheckerError(f"{at}: row {row} is outside a batch of {batch_rows} rows")
        if draw >= len(draws):
            raise CheckerError(f"{at}: draw {draw} does not exist in this sample ({len(draws)} draws)")
        if row in rows_seen:
            raise CheckerError(f"{at}: row {row} is replaced twice")
        if draw in draws_seen:
            raise CheckerError(f"{at}: draw {draw} is placed twice or both placed and declined")
        rows_seen.add(row)
        draws_seen.add(draw)
        if entry.get("content_digest") != draws[draw].content_digest:
            raise CheckerError(
                f"{at}: row {row} claims example {str(entry.get('content_digest'))[:16]}… but draw {draw} selected "
                f"{str(draws[draw].content_digest)[:16]}…"
            )
        return row, draw

    # -- manifest ----------------------------------------------------------

    def check_manifest(self, manifest: list[dict]) -> None:
        """The manifest must open exactly this log's commitments, in order.

        A log with no inserts at all accepts an empty manifest (a buffer
        closed before its first group); a log whose inserts carry no
        digests accepts none, because there is nothing a manifest line
        could be the opening of.
        """
        if self.has_content is None:
            if manifest:
                raise CheckerError(f"manifest has {len(manifest)} lines but the log has no inserts")
            return
        if not self.has_content:
            raise CheckerError("the log has no content digests; a manifest cannot be checked against it")
        lines = [_validated_line(line, i) for i, line in enumerate(manifest)]
        if len(lines) != len(self.history):
            raise CheckerError(
                f"manifest has {len(lines)} lines for {len(self.history)} insert records with content digests"
            )
        for i, (line, insert) in enumerate(zip(lines, self.history)):
            _match_line(line, insert, i)
        self.manifest_matched = len(lines)


def _validated_line(line: object, i: int) -> dict:
    """Field-level checks of one manifest line, plus its own digest recomputation."""
    where = f"manifest line {i}"
    required, optional = set(MANIFEST_KEYS), set(MANIFEST_OPTIONAL_KEYS)
    if not isinstance(line, dict) or not required <= set(line) <= required | optional:
        raise CheckerError(
            f"{where}: must be an object with exactly the manifest keys {MANIFEST_KEYS} "
            f"(optionally {MANIFEST_OPTIONAL_KEYS})"
        )
    if "rewards" in line:
        _require_rewards(line["rewards"], where)
    for name in ("op_counter", "index", "entry_version"):
        value = line[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CheckerError(f"{where}: {name} must be a non-negative integer, got {value!r}")
    if line["source"] is not None and not _is_source(line["source"]):
        raise CheckerError(f"{where}: source must be null or a printable string, got {line['source']!r}")
    if not _is_digest(line["content_digest"]):
        raise CheckerError(f"{where}: content_digest must be {_DIGEST_LENGTH} lowercase hex characters")
    recomputed = content_digest(line["prompt_id"], line["tokens"], line["reward_hex"], where)
    if recomputed != line["content_digest"]:
        raise CheckerError(f"{where}: content_digest does not recompute from its prompt, tokens and reward")
    return line


def _match_line(line: dict, insert: CommittedInsert, i: int) -> None:
    where = f"manifest line {i}"
    if (line["op_counter"], line["index"]) != (insert.op_counter, insert.index):
        raise CheckerError(
            f"{where}: (op_counter, index) = ({line['op_counter']}, {line['index']}) but insert record "
            f"{insert.record_index} is ({insert.op_counter}, {insert.index})"
        )
    if line["content_digest"] != insert.content_digest:
        raise CheckerError(f"{where}: content_digest disagrees with insert record {insert.record_index}")
    if line["source"] != insert.source:
        raise CheckerError(
            f"{where}: source {line['source']!r} disagrees with the log's {insert.source!r}"
        )
    if insert.entry_version is not None and line["entry_version"] != insert.entry_version:
        raise CheckerError(
            f"{where}: entry_version {line['entry_version']} disagrees with the log's {insert.entry_version}"
        )


MAX_BATCH_ROWS = 1 << 20
"""Largest training batch a witness may declare; bounds the memory ``replay`` and ``transcript`` spend on one record."""


def _strict_int(record: dict, name: str, where: str) -> int:
    """A JSON integer (not bool, not a string): the encoding batch records use for small counts."""
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckerError(f"{where}: {name} must be a non-negative JSON integer, got {value!r}")
    return value


def _witness_lists(record: dict, where: str, n_draws: int, op: int) -> tuple[list, list[int]]:
    """The ``replaced`` and ``declined`` lists of a witness, checked to cover the draws exactly."""
    replaced = record.get("replaced")
    if not isinstance(replaced, list):
        raise CheckerError(f"{where}: replaced must be a list")
    declined = record.get("declined", [])
    if not isinstance(declined, list) or any(isinstance(d, bool) or not isinstance(d, int) for d in declined):
        raise CheckerError(f"{where}: declined must be a list of draw positions")
    if any(not 0 <= d < n_draws for d in declined) or len(set(declined)) != len(declined):
        raise CheckerError(f"{where}: declined draws must be distinct positions of sample {op}")
    if len(replaced) + len(declined) != n_draws:
        raise CheckerError(
            f"{where}: witness places {len(replaced)} rows and declines {len(declined)} draws but sample "
            f"{op} drew {n_draws} rollouts"
        )
    return replaced, declined


def load_manifest(text: str) -> list[dict]:
    """Parse manifest JSON lines; every non-empty line must be a JSON object.

    Lines are split on ``\n`` only (``str.splitlines`` would also split on
    U+2028 and other separators that may legitimately occur inside a JSON
    string). Errors name the line as a text editor numbers it, from 1.
    """
    lines: list[dict] = []
    for lineno, raw in enumerate(text.split("\n"), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except (ValueError, RecursionError) as exc:
            raise CheckerError(f"manifest file line {lineno}: not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise CheckerError(f"manifest file line {lineno}: not a JSON object")
        lines.append(value)
    return lines
