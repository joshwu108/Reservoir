"""
reservoir_checker.replay — Offline replay: every witnessed training batch as content.

This module imports nothing from the rest of the reservoir package. It
builds on ``verify_chain``, which establishes that the log is a consistent
record of one buffer's life and that the manifest opens exactly its
content commitments, and emits what the buffer put into each training
batch as data a third party can train on or audit, with no access to the
trainer, the model or the library that wrote the log.

What it emits
-------------
JSON lines. The first is a header::

    {"kind": "header", "tool": "reservoir-replay-offline", "format": 2,
     "records": N, "examples": M, "witnesses": W, "head_digest": "...",
     "steps_witnessed": [...], "steps_without_witness": [...]}

then one line per batch witness, in log order::

    {"kind": "batch", "step": S, "sample_op_counter": K, "record_index": I,
     "batch_rows": B, "tensor_digest": "...",
     "rows": [{"row": r, "draw": d, "slot": s, "is_weight": w, "is_weight_exact": [num, den],
               "probability": p, "probability_exact": [num, den],
               "manifest_line": L, "content_digest": "...", "prompt_id": "...", "source": "...",
               "tokens": [...], "reward": x, "reward_hex": "...", "entry_version": v,
               "rewards": {...}}, ...],
     "declined": [draw positions the adapter refused],
     "fresh_rows": [rows the witness does not bind],
     "generated": [examples inserted at entry version S, same fields as a row's example]}

``rows`` are the replaced rows: each is the witness entry for that row,
the draw it names resolved to the slot it drew and the example that slot
held at the moment of the draw (the latest insert into that slot before
the sample record), opened by the manifest line of that insert. The
importance weight and the probability are the sample record's exact
fractions, given as floats and as ``[numerator, denominator]`` decimal
strings. ``rewards`` is present only when the manifest line carries the
per-reward-function values.

``fresh_rows`` are the rows the witness does not bind to a draw. The log
does not record which example a fresh row held (docs/nonclaims.md); what
it does record is every example inserted before the witness at an entry
version equal to the step, listed unordered under ``generated``. That
these are the step's own generation rests on the adapter stamping the
entry version with the trainer step, as the TRL adapter does; the log
does not verify that mapping. Two witnesses of one step list the same
examples. A step with no witness (nothing was replayed) has no batch
line; its number is in ``steps_without_witness`` when the log has a
telemetry record for it.

What it refuses
---------------
A log below format 2 (batch witnesses arrived with format 2), a log with
no manifest, and anything ``verify_chain`` rejects, including a manifest
that does not open the log. A tampered manifest therefore either breaks
the replay (a changed token, reward, prompt, source, order or count fails
verification) or changes its output (the per-reward-function values are
outside the content digest and are reported as the manifest states them);
the mutation campaign measures both.

Command line::

    reservoir-replay-offline run-01/attest.jsonl --manifest run-01/manifest.jsonl > replay.jsonl
    reservoir-replay-offline run-01/attest.jsonl --manifest run-01/manifest.jsonl --out replay.jsonl

Exit status 0 with the lines written, 1 with ``FAIL: <reason>`` on stderr
and nothing written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from bisect import bisect_left
from fractions import Fraction
from typing import Optional

from reservoir_checker.content import CommittedInsert, ResolvedSample, WitnessedRow, load_manifest
from reservoir_checker.decay_replay import CheckerError
from reservoir_checker.verify import VerifiedLog, verify_chain

TOOL = "reservoir-replay-offline"
MIN_FORMAT = 2
"""The log format that introduced batch witnesses; a replay has nothing to
reconstruct from an older log."""

_EXAMPLE_FIELDS = ("content_digest", "prompt_id", "source", "tokens", "reward_hex", "entry_version")


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def log_format(records: list[dict]) -> int:
    """The format a log declares: its ``decay_config`` ``format``, 1 when it has none."""
    if not records or not isinstance(records[0], dict) or records[0].get("op") != "decay_config":
        return 1
    value = records[0].get("format", "1")
    if not isinstance(value, str) or not (value.isascii() and value.isdigit()):
        raise CheckerError(f"Record 0: decay_config format must be a decimal string, got {value!r}")
    return int(value)


def require_replayable(records: list[dict], manifest: Optional[list[dict]]) -> int:
    """Refuse what the replay cannot work from; return the log's format."""
    if manifest is None:
        raise CheckerError("offline replay needs the manifest: the log commits to each example, the manifest opens it")
    fmt = log_format(records)
    if fmt < MIN_FORMAT:
        raise CheckerError(
            f"offline replay needs a format-{MIN_FORMAT} log or later (batch witnesses); this log is format {fmt}"
        )
    return fmt


# ---------------------------------------------------------------------------
# Resolving witnesses to manifest lines
# ---------------------------------------------------------------------------

class _Inserts:
    """The insert history indexed by slot, so a draw resolves to the insert live at its record."""

    def __init__(self, history: list[CommittedInsert]) -> None:
        by_slot: dict[int, list[tuple[int, int]]] = {}
        for position, insert in enumerate(history):
            by_slot.setdefault(insert.index, []).append((insert.record_index, position))
        self._by_slot = by_slot
        self._by_version: dict[int, list[tuple[int, int]]] = {}
        for position, insert in enumerate(history):
            if insert.entry_version is not None:
                self._by_version.setdefault(insert.entry_version, []).append((insert.record_index, position))

    def live_at(self, slot: int, record_index: int) -> int:
        """Manifest line (history position) of the insert into ``slot`` live at ``record_index``."""
        entries = self._by_slot.get(slot, [])
        k = bisect_left(entries, (record_index, -1))
        if k == 0:
            raise CheckerError(f"Record {record_index}: slot {slot} held no committed example before this record")
        return entries[k - 1][1]

    def inserted_at_version(self, version: int, before: int) -> list[int]:
        """Manifest lines of the inserts at ``version`` whose record precedes ``before``."""
        return [position for record_index, position in self._by_version.get(version, []) if record_index < before]


def _example(line: dict, position: int) -> dict:
    """The opened example of one manifest line, as the output spells it."""
    out = {"manifest_line": position, **{name: line[name] for name in _EXAMPLE_FIELDS}}
    out["reward"] = float.fromhex(line["reward_hex"])
    if "rewards" in line:
        out["rewards"] = dict(line["rewards"])
    return out


def _fraction_fields(name: str, value: Fraction) -> dict:
    return {name: float(value), f"{name}_exact": [str(value.numerator), str(value.denominator)]}


def _row(hit: WitnessedRow, draw: ResolvedSample, inserts: _Inserts, manifest: list[dict]) -> dict:
    position = inserts.live_at(draw.leaf_index, draw.record_index)
    line = manifest[position]
    if line["content_digest"] != hit.content_digest:
        raise CheckerError(   # verify_chain proved the witness matches the draw; this guards the lookup itself
            f"Record {hit.record_index}: row {hit.row} resolved to manifest line {position}, whose digest is not "
            f"the witnessed {hit.content_digest[:16]}…"
        )
    return {
        "row": hit.row, "draw": hit.draw, "slot": draw.leaf_index,
        **_fraction_fields("is_weight", draw.is_weight),
        **_fraction_fields("probability", draw.probability),
        **_example(line, position),
    }


def _batch(witness: dict, draws: dict[int, ResolvedSample], hits: list[WitnessedRow], inserts: _Inserts,
           manifest: list[dict]) -> dict:
    """One output line: ``draws`` are the witnessed sample's draws by position, ``hits`` its witnessed rows."""
    at = witness["record_index"]
    rows = [_row(hit, draws[hit.draw], inserts, manifest) for hit in hits]
    bound = {r["row"] for r in rows}
    return {
        "kind": "batch", "step": witness["step"], "sample_op_counter": witness["sample_op_counter"],
        "record_index": at, "batch_rows": witness["batch_rows"],
        "tensor_digest": witness["tensor_digest"], "rows": rows, "declined": list(witness["declined"]),
        "fresh_rows": [r for r in range(witness["batch_rows"]) if r not in bound],
        "generated": [_example(manifest[p], p) for p in inserts.inserted_at_version(witness["step"], at)],
    }


def _header(verified: VerifiedLog, fmt: int, manifest: list[dict], batches: list[dict]) -> dict:
    records = verified.records
    witnessed = [b["step"] for b in batches]
    witnessed_set = set(witnessed)
    telemetry_steps = sorted({point.step for point in verified.content.telemetry})
    return {
        "kind": "header", "tool": TOOL, "format": fmt, "records": len(records), "examples": len(manifest),
        "witnesses": len(batches), "head_digest": records[-1]["digest"] if records else "genesis",
        "steps_witnessed": witnessed, "steps_without_witness": [s for s in telemetry_steps if s not in witnessed_set],
    }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def replay_batches(verified: VerifiedLog, manifest: list[dict]) -> list[dict]:
    """One dict per batch witness of a log ``verify_chain`` accepted with this manifest."""
    content = verified.content
    inserts = _Inserts(content.history)
    draws_by_op: dict[int, dict[int, ResolvedSample]] = {}
    for draw in content.samples:
        draws_by_op.setdefault(draw.op_counter, {})[draw.position_in_batch] = draw
    hits_by_record: dict[int, list[WitnessedRow]] = {}
    for hit in content.witnessed_rows:
        hits_by_record.setdefault(hit.record_index, []).append(hit)
    return [_batch(w, draws_by_op.get(w["sample_op_counter"], {}), hits_by_record.get(w["record_index"], []),
                   inserts, manifest) for w in content.witnesses]


def replay(records: list[dict], manifest: Optional[list[dict]]) -> list[dict]:
    """Verify the log against its manifest and return the header followed by every batch.

    Raises ``CheckerError`` for a log below ``MIN_FORMAT``, a missing
    manifest, or anything the checker rejects.
    """
    fmt = require_replayable(records, manifest)
    verified = verify_chain(records, manifest=manifest)
    batches = replay_batches(verified, manifest)
    return [_header(verified, fmt, manifest, batches), *batches]


def replay_json_lines(data: str, manifest: str) -> list[dict]:
    """``replay`` over the two files' texts."""
    records = [json.loads(line) for line in data.split("\n") if line.strip()]
    return replay(records, load_manifest(manifest))


def render(lines: list[dict]) -> str:
    """The output as JSON lines: canonical (sorted keys, no spaces), one per line, newline-terminated."""
    return "".join(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n" for line in lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="Reconstruct every witnessed training batch of an attestation log as content, as JSON lines.",
    )
    parser.add_argument("path", help="JSON-lines attestation log (format 2 or later)")
    parser.add_argument("--manifest", default=None, help="manifest JSON lines; required, it opens the log's commitments")
    parser.add_argument("--out", default=None, metavar="PATH", help="write the lines here instead of stdout")
    return parser


def _write_atomically(path: str, text: str) -> None:
    """Write through a sibling temporary file so a failed write leaves no partial output."""
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point: exit 0 with the replay written, 1 with ``FAIL:`` on stderr."""
    args = _parser().parse_args(argv)
    try:
        if args.manifest is None:
            raise CheckerError("offline replay needs --manifest: the log commits to each example, the manifest opens it")
        with open(args.path, encoding="utf-8") as f:
            data = f.read()
        with open(args.manifest, encoding="utf-8") as f:
            manifest_text = f.read()
        lines = replay_json_lines(data, manifest_text)
        text = render(lines)
        if args.out is not None:
            _write_atomically(args.out, text)
        else:
            sys.stdout.write(text)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError,
            RecursionError, CheckerError) as exc:
        # Everything a malformed file can raise is reported as a failure, never a traceback.
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    header = lines[0]
    summary = (f"OK: {header['witnesses']} witnessed batches from {header['records']} records and "
               f"{header['examples']} examples")
    print(summary, file=sys.stderr if args.out is None else sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
