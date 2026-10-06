"""
checker.transcript — What a verified attestation log says happened.

This module imports nothing from the rest of the reservoir package. It builds on
``checker.verify.verify_chain``, which has already established that the
log is a consistent record of one buffer's life, and turns that record
into the answers an auditor asks for:

- **Exposure.** For every committed example (``content_digest``): how many
  times it was drawn, the exact sum of its importance weights, when it
  was first and last drawn, and every insert of it with the record where
  that copy was evicted (if it was).
- **Mixture.** For every ``source`` tag: how many examples were inserted,
  how many sampled rows came from it, and its share of all sampled rows;
  the same per version window (the stretch of the log between two
  ``advance_version`` records, i.e. one policy version).
- **Quota.** ``--quota source=N`` fails (exit 2) if more than ``N`` sampled
  rows came from that source over the whole log.
- **Find.** ``--find <digest>`` lists every sample record and batch
  position where that example appears.
- **Explain.** ``--explain STEP ROW`` says why row ROW of the training
  batch built at trainer step STEP holds what it holds: fresh (generated
  this step) or replayed from a given draw, with the example's digest,
  its source, the weight it was replayed with and the record that stored
  it. Needs batch witnesses in the log.
- **Blast radius.** ``--blast-radius <digest>`` lists every insert of the
  example, every training-batch row that held it (from the batch
  witnesses) with its step, the steps touched, how often it was drawn
  and how many of those draws have no witness (rows and steps are then a
  lower bound), and every quarantine record that removed it. With a
  manifest the radius widens to every example of the same prompt. The
  question it answers after an incident: what did this example reach, and
  back to which step must a run go.

Every statement is derived from the log alone; the manifest, when given,
only adds the human-readable example (prompt id, tokens, reward) next to
its digest (and its per-reward-function values when the manifest carries
them). A log without content digests supports the per-slot view only, and
the report says so.

Command line::

    python -m checker.transcript run-01/attest.jsonl
    python -m checker.transcript run-01/attest.jsonl --manifest run-01/manifest.jsonl --by source
    python -m checker.transcript run-01/attest.jsonl --quota gsm8k=5000 --quota synthetic=0
    python -m checker.transcript run-01/attest.jsonl --json report.json \
        --find 2a837616faed26935aa920b0fd1357cdd528667c469e6682e554e3b85079578f

Exit status: 0 verified (and every quota met), 2 a quota was exceeded,
1 the log did not verify, an argument was malformed, or ``--quota`` /
``--find`` / ``--blast-radius`` were asked of a log that has no content
digests.

The input to ``build_transcript`` must be a ``VerifiedLog``: the walk
reads record fields without re-validating them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from typing import Optional

from reservoir_checker.content import load_manifest
from reservoir_checker.decay_replay import CheckerError
from reservoir_checker.verify import VerifiedLog, verify_chain

NO_SOURCE = "(none)"
"""Report key for examples whose group carried no source tag. A real source
tag with this spelling is refused so the two can never be merged."""

_HEX = frozenset("0123456789abcdef")


# ---------------------------------------------------------------------------
# Building the report
# ---------------------------------------------------------------------------

def _source_key(source: Optional[str]) -> str:
    return source if source is not None else NO_SOURCE


def _walk(records: list[dict]) -> tuple[list[int], list[dict], dict[int, Optional[int]]]:
    """One pass over the records for what the content state does not keep.

    Returns the window index of every record, the version windows
    (``version``, ``from_record``, ``to_record``; a new window opens at
    every ``advance_version`` record, even one that repeats the version)
    and, for every insert record index, the index of the record that
    evicted that copy (or None).
    """
    window_at: list[int] = []
    windows: list[dict] = [{"version": 0, "from_record": 0, "to_record": None}]
    occupant: dict[int, int] = {}            # slot -> record index of the insert living there
    evicted_at: dict[int, Optional[int]] = {}
    for i, r in enumerate(records):
        op = r.get("op")
        if op == "advance_version":
            windows[-1]["to_record"] = i
            windows.append({"version": int(r["new_version"]), "from_record": i, "to_record": None})
        elif op == "insert":
            occupant[r["index"]] = i
            evicted_at[i] = None
        elif op == "evict" and r["index"] in occupant:
            evicted_at[occupant.pop(r["index"])] = i
        window_at.append(len(windows) - 1)
    windows[-1]["to_record"] = len(records)
    return window_at, windows, evicted_at


def _exposure(verified: VerifiedLog, evicted_at: dict[int, Optional[int]]) -> dict[str, dict]:
    """Per-example entry: inserts, sample count, exact weight sum, first/last draw."""
    content: dict[str, dict] = {}
    for ins in verified.content.history:
        entry = content.setdefault(ins.content_digest, {
            "content_digest": ins.content_digest, "source": ins.source, "times_sampled": 0,
            "sum_is_weight": Fraction(0), "first_op_counter": None, "last_op_counter": None, "inserts": [],
        })
        entry["inserts"].append({
            "record_index": ins.record_index, "op_counter": ins.op_counter, "index": ins.index,
            "entry_version": ins.entry_version, "evicted_at_record": evicted_at.get(ins.record_index),
        })
    for s in verified.content.samples:
        entry = content[s.content_digest]  # type: ignore[index]
        entry["times_sampled"] += 1
        entry["sum_is_weight"] += s.is_weight
        if entry["first_op_counter"] is None:
            entry["first_op_counter"] = s.op_counter
        entry["last_op_counter"] = s.op_counter
    for entry in content.values():
        total: Fraction = entry["sum_is_weight"]
        entry["sum_is_weight"] = {"num": str(total.numerator), "den": str(total.denominator), "float": float(total)}
    return content


def _mixture(verified: VerifiedLog, window_at: list[int], windows: list[dict]) -> tuple[dict, list[dict]]:
    """Per-source totals and per-window counts (new window dicts; the input is not changed)."""
    inserted: Counter = Counter(_source_key(i.source) for i in verified.content.history)
    sampled: Counter = Counter(_source_key(s.source) for s in verified.content.samples)
    total = sum(sampled.values())
    sources = {
        key: {"inserted": inserted.get(key, 0), "sampled": sampled.get(key, 0),
              "share": (sampled.get(key, 0) / total) if total else 0.0}
        for key in sorted(set(inserted) | set(sampled))
    }
    per_window: dict[int, Counter] = defaultdict(Counter)
    for s in verified.content.samples:
        per_window[window_at[s.record_index]][_source_key(s.source)] += 1
    with_counts = [
        {**w, "sampled_by_source": dict(sorted(per_window.get(k, Counter()).items())),
         "sampled": sum(per_window.get(k, Counter()).values())}
        for k, w in enumerate(windows)
    ]
    return sources, with_counts


def _quotas(sources: dict, quotas: dict[str, int]) -> list[dict]:
    """One verdict per quota; ``present`` is False when the log never saw that source."""
    return [
        {"source": name, "limit": limit, "sampled": sources.get(name, {}).get("sampled", 0),
         "present": name in sources, "ok": sources.get(name, {}).get("sampled", 0) <= limit}
        for name, limit in quotas.items()
    ]


def _require_digests(values: list[str], flag: str = "--find") -> None:
    for value in values:
        if not (isinstance(value, str) and len(value) == 64 and set(value) <= _HEX):
            raise CheckerError(f"{flag} expects a 64-character lowercase hex content digest, got {value!r}")


def _find(verified: VerifiedLog, digests: list[str]) -> dict[str, list[dict]]:
    hits: dict[str, list[dict]] = {d: [] for d in digests}
    wanted = set(digests)
    for s in verified.content.samples:
        if s.content_digest in wanted:
            hits[s.content_digest].append({  # type: ignore[index]
                "record_index": s.record_index, "position_in_batch": s.position_in_batch,
                "op_counter": s.op_counter, "leaf_index": s.leaf_index, "is_weight": float(s.is_weight),
            })
    return hits


def explain_row(verified: VerifiedLog, step: int, row: int, sample_op_counter: Optional[int] = None) -> dict:
    """Why a training-batch row holds what it holds, from the batch witness of ``step``.

    A step with several witnesses (several generation batches per
    optimizer step) needs ``sample_op_counter`` to say which one.
    """
    witnesses = [w for w in verified.content.witnesses if w["step"] == step]
    if sample_op_counter is not None:
        witnesses = [w for w in witnesses if w["sample_op_counter"] == sample_op_counter]
    if not witnesses:
        raise CheckerError(f"no batch witness for step {step}; the adapter did not record one")
    if len(witnesses) > 1:
        raise CheckerError(
            f"step {step} has {len(witnesses)} batch witnesses (samples "
            f"{[w['sample_op_counter'] for w in witnesses]}); pass --explain-sample to pick one"
        )
    witness = witnesses[0]
    if not 0 <= row < witness["batch_rows"]:
        raise CheckerError(f"row {row} is outside the batch of {witness['batch_rows']} rows at step {step}")
    hit = next((w for w in verified.content.witnessed_rows
                if w.step == step and w.row == row and w.sample_op_counter == witness["sample_op_counter"]), None)
    if hit is None:
        return {"step": step, "row": row, "kind": "fresh", "sample_op_counter": witness["sample_op_counter"]}
    sample = next(s for s in verified.content.samples
                  if s.op_counter == hit.sample_op_counter and s.position_in_batch == hit.draw)
    inserts = [i for i in verified.content.history if i.content_digest == hit.content_digest]
    return {
        "step": step, "row": row, "kind": "replayed", "sample_op_counter": hit.sample_op_counter,
        "draw": hit.draw, "content_digest": hit.content_digest, "source": sample.source,
        "slot": sample.leaf_index, "is_weight": float(sample.is_weight),
        "probability": float(sample.probability),
        "stored_at_records": [i.record_index for i in inserts],
        "entry_versions": [i.entry_version for i in inserts],
    }


_NOT_COMMITTED = "this digest was never committed in the log"


def blast_radius(verified: VerifiedLog, digests: list[str], manifest: Optional[list[dict]] = None) -> dict[str, dict]:
    """What each example reached: inserts, witnessed rows and steps, quarantine records.

    With ``manifest`` the radius covers every committed example of the
    same prompt; without it only the exact digest is followed. Rows come
    from the batch witnesses, so a log without them lists none and the
    entry says so.
    """
    by_digest = {line["content_digest"]: line for line in manifest} if manifest else {}
    committed = {i.content_digest for i in verified.content.history}
    evicted_at = _walk(verified.records)[2]
    report: dict[str, dict] = {}
    for digest in digests:
        if digest not in committed:
            report[digest] = {"found": False, "prompt_id": None, "examples": [], "inserts": [], "rows": [],
                              "steps": [], "times_sampled": 0, "unwitnessed_draws": 0, "quarantined": [],
                              "note": _NOT_COMMITTED}
            continue
        line = by_digest.get(digest)
        prompt_id = line["prompt_id"] if line is not None else None
        examples = [digest] if line is None else sorted(
            d for d, l in by_digest.items() if l["prompt_id"] == prompt_id and d in committed)
        report[digest] = _radius_entry(verified, prompt_id, examples, evicted_at, manifest is not None)
    return report


def _radius_entry(verified: VerifiedLog, prompt_id: Optional[str], examples: list[str],
                  evicted_at: dict[int, Optional[int]], with_manifest: bool) -> dict:
    wanted = set(examples)
    witnessed_ops = {w["sample_op_counter"] for w in verified.content.witnesses}
    declined_at = {(w["sample_op_counter"], k) for w in verified.content.witnesses for k in w["declined"]}
    draws = [s for s in verified.content.samples if s.content_digest in wanted]
    rows = sorted(
        ({"step": w.step, "row": w.row, "sample_op_counter": w.sample_op_counter, "draw": w.draw,
          "content_digest": w.content_digest, "record_index": w.record_index}
         for w in verified.content.witnessed_rows if w.content_digest in wanted),
        key=lambda r: (r["step"], r["row"]),
    )
    entry: dict = {
        "found": True, "prompt_id": prompt_id, "examples": examples,
        "inserts": [
            {"record_index": i.record_index, "op_counter": i.op_counter, "index": i.index,
             "entry_version": i.entry_version, "content_digest": i.content_digest,
             "evicted_at_record": evicted_at.get(i.record_index)}
            for i in verified.content.history if i.content_digest in wanted
        ],
        "rows": rows,
        "steps": sorted({r["step"] for r in rows}),
        "times_sampled": len(draws),
        "unwitnessed_draws": sum(1 for s in draws if s.op_counter not in witnessed_ops),
        # Drawn, then declined by the drift gate: never reached a training batch, so not in ``rows``.
        "declined_draws": sum(1 for s in draws if (s.op_counter, s.position_in_batch) in declined_at),
        "quarantined": [
            {"record_index": q.record_index, "index": q.index, "content_digest": q.content_digest,
             "predicate": q.predicate, "note": q.note}
            for q in verified.content.quarantines if q.content_digest in wanted
        ],
    }
    notes = []
    if not verified.content.witnesses:
        notes.append("the log has no batch witnesses, so no training-batch rows can be listed")
    elif entry["unwitnessed_draws"]:
        notes.append(f"{entry['unwitnessed_draws']} draw(s) of these examples have no batch witness, so rows and "
                     "steps are a lower bound")
    if not with_manifest:
        notes.append("without a manifest only this exact example is followed, not the prompt's other examples")
    if notes:
        entry["note"] = "; ".join(notes)
    return entry


def build_transcript(
    verified: VerifiedLog,
    quotas: Optional[dict[str, int]] = None,
    find: Optional[list[str]] = None,
    manifest: Optional[list[dict]] = None,
    explain: Optional[tuple[int, int]] = None,
    explain_sample: Optional[int] = None,
    blast: Optional[list[str]] = None,
) -> dict:
    """The full report as a JSON-serialisable dict. See the module docstring for its parts.

    Raises ``CheckerError`` when ``quotas``, ``find`` or ``blast`` are asked
    of a log without content digests (a silent pass there would be a
    false assurance), when a ``find`` or ``blast`` value is not a digest,
    or when a real source tag collides with the report's key for untagged
    examples.
    """
    records = verified.records
    window_at, windows, evicted_at = _walk(records)
    report: dict = {
        "records": len(records),
        "sampled_rows": len(verified.content.samples),
        "has_content": bool(verified.content.has_content),
        "manifest_matched": verified.content.manifest_matched,
    }
    if find is not None:
        _require_digests(find)
    if blast is not None:
        _require_digests(blast, "--blast-radius")
    if not verified.content.has_content:
        if quotas is not None or find is not None or blast is not None:
            raise CheckerError("--quota, --find and --blast-radius need content digests; this log has none")
        slots: Counter = Counter(str(s.leaf_index) for s in verified.content.samples)
        report["slots"] = dict(sorted(slots.items(), key=lambda kv: int(kv[0])))
        report["note"] = "the log has no content digests; only per-slot exposure is available"
        return report
    if any(i.source == NO_SOURCE for i in verified.content.history):
        raise CheckerError(f"a source tag spelled {NO_SOURCE!r} would be confused with untagged examples")
    content = _exposure(verified, evicted_at)
    if manifest:
        by_digest = {line["content_digest"]: line for line in manifest}
        for digest, entry in content.items():
            line = by_digest.get(digest)
            if line is not None:
                entry["example"] = {k: line[k] for k in ("prompt_id", "tokens", "reward_hex")}
                if "rewards" in line:
                    entry["example"]["rewards"] = line["rewards"]
    sources, windows = _mixture(verified, window_at, windows)
    report["content"] = content
    report["sources"] = sources
    report["windows"] = windows
    if quotas is not None:
        report["quotas"] = _quotas(sources, quotas)
        report["quota_ok"] = all(q["ok"] for q in report["quotas"])
    if find is not None:
        report["find"] = _find(verified, find)
    report["batch_witnesses"] = len(verified.content.witnesses)
    report["telemetry"] = [
        {"record_index": t.record_index, "step": t.step, **t.counts,
         "ess": float(t.ess) if t.ess is not None else None,
         "staleness_max": t.staleness_max,
         "staleness_mean": float(t.staleness_mean) if t.staleness_mean is not None else None,
         "reported": t.reported}
        for t in verified.content.telemetry
    ]
    if explain is not None:
        report["explain"] = explain_row(verified, *explain, sample_op_counter=explain_sample)
    if blast is not None:
        report["blast_radius"] = blast_radius(verified, blast, manifest)
    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_text(report: dict, by: str = "all") -> str:
    """Human-readable rendering; ``by`` selects ``source``, ``content``, ``slot`` or ``all``."""
    lines = [f"{report['records']} records verified; {report['sampled_rows']} sampled rows"]
    if report.get("manifest_matched"):
        lines.append(f"manifest opens {report['manifest_matched']} committed examples")
    if not report["has_content"]:
        lines.append(report["note"])
        if by in ("slot", "all"):
            lines += [f"  slot {slot}: {n} draws" for slot, n in report["slots"].items()]
        else:
            lines.append(f"(no per-{by} view for a log without content digests)")
        return "\n".join(lines)
    if by == "slot":
        lines.append("(the per-slot view is for logs without content digests; use --by content)")
    if by in ("source", "all"):
        lines.append("by source (inserted examples, sampled rows, share of sampled rows):")
        for name, s in report["sources"].items():
            lines.append(f"  {name}: inserted {s['inserted']}, sampled {s['sampled']}, share {s['share']:.3f}")
        windows = [w for w in report["windows"] if w["sampled"]]
        lines.append(f"versions with replay: {len(windows)} of {len(report['windows'])}")
    if by in ("content", "all"):
        lines.append(f"by example ({len(report['content'])} committed):")
        ranked = sorted(report["content"].values(), key=lambda e: (-e["times_sampled"], e["content_digest"]))
        for e in ranked[:20]:
            tag = f" [{e['source']}]" if e["source"] is not None else ""
            lines.append(f"  {e['content_digest'][:16]}…{tag}: sampled {e['times_sampled']}x, "
                         f"weight {e['sum_is_weight']['float']:.4f}, copies {len(e['inserts'])}")
        if len(ranked) > 20:
            lines.append(f"  … {len(ranked) - 20} more")
    for q in report.get("quotas", []):
        verdict = "ok" if q["ok"] else "EXCEEDED"
        absent = "" if q["present"] else "; this source never appears in the log"
        lines.append(f"quota {q['source']} <= {q['limit']}: sampled {q['sampled']} ({verdict}{absent})")
    if "explain" in report:
        e = report["explain"]
        if e["kind"] == "fresh":
            lines.append(f"step {e['step']} row {e['row']}: fresh (generated this step)")
        else:
            lines.append(f"step {e['step']} row {e['row']}: replayed from draw {e['draw']} of sample "
                         f"{e['sample_op_counter']}, example {e['content_digest'][:16]}… "
                         f"[{e['source']}] slot {e['slot']}, weight {e['is_weight']:.4f}, "
                         f"stored at record(s) {e['stored_at_records']}")
    for digest, hits in report.get("find", {}).items():
        lines.append(f"find {digest[:16]}…: {len(hits)} occurrence(s)")
        lines += [f"  record {h['record_index']} position {h['position_in_batch']} op_counter {h['op_counter']}"
                  for h in hits]
    for digest, entry in report.get("blast_radius", {}).items():
        lines += _render_radius(digest, entry)
    return "\n".join(lines)


def _render_radius(digest: str, b: dict) -> list[str]:
    if not b["found"]:
        return [f"blast radius {digest[:16]}…: {b['note']}"]
    prompt = f" prompt {b['prompt_id']!r}," if b["prompt_id"] is not None else ""
    lines = [f"blast radius {digest[:16]}…:{prompt} {len(b['examples'])} example(s), {len(b['inserts'])} insert(s), "
             f"sampled {b['times_sampled']}x ({b['unwitnessed_draws']} unwitnessed), {len(b['rows'])} witnessed "
             f"row(s) over steps {b['steps']}, quarantined {len(b['quarantined'])}x"]
    lines += [f"  step {r['step']} row {r['row']}: sample {r['sample_op_counter']} draw {r['draw']}, "
              f"example {r['content_digest'][:16]}…" for r in b["rows"]]
    lines += [f"  quarantined at record {q['record_index']} (slot {q['index']}): {q['note']} [{q['predicate']}]"
              for q in b["quarantined"]]
    if b.get("note"):
        lines.append(f"  note: {b['note']}")
    return lines


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _parse_quotas(texts: list[str]) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for text in texts:
        name, sep, limit = text.partition("=")
        if not sep or not name or not (limit.isascii() and limit.isdigit()):
            raise ValueError(f"--quota expects source=N with N a non-negative integer, got {text!r}")
        if name in quotas:
            raise ValueError(f"--quota given twice for {name!r}")
        quotas[name] = int(limit)
    return quotas


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on a usage error; here 2 means "quota exceeded", so usage errors exit 1."""

    def error(self, message: str) -> None:  # type: ignore[override]
        print(f"FAIL: {message}", file=sys.stderr)
        sys.exit(1)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        description="Verify a reservoir attestation log and report what was sampled from where.",
    )
    parser.add_argument("path", help="JSON-lines attestation log")
    parser.add_argument("--manifest", default=None, help="manifest JSON lines (adds examples to the report)")
    parser.add_argument("--capacity", type=int, default=None, help="capacity for a log without decay_config")
    parser.add_argument("--allow-truncated", action="store_true", help="accept a log cut off mid-advance")
    parser.add_argument("--by", choices=("source", "content", "slot", "all"), default="all")
    parser.add_argument("--quota", action="append", default=[], metavar="SOURCE=N",
                        help="fail (exit 2) if more than N sampled rows came from SOURCE; repeatable")
    parser.add_argument("--find", action="append", default=[], metavar="DIGEST",
                        help="list every occurrence of this content digest; repeatable")
    parser.add_argument("--explain", nargs=2, type=int, default=None, metavar=("STEP", "ROW"),
                        help="why this training-batch row holds what it holds (needs batch witnesses)")
    parser.add_argument("--explain-sample", type=int, default=None, metavar="OP_COUNTER",
                        help="with --explain: which sample's witness, when a step has several")
    parser.add_argument("--blast-radius", action="append", default=[], metavar="DIGEST",
                        help="every insert, witnessed training-batch row, step and quarantine of this example "
                             "(with --manifest: of every example of its prompt); repeatable")
    parser.add_argument("--json", default=None, metavar="PATH", help="also write the full report as JSON")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point. See the module docstring for exit codes."""
    args = _parser().parse_args(argv)
    try:
        quotas = _parse_quotas(args.quota) if args.quota else None
        with open(args.path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        manifest = None
        if args.manifest is not None:
            with open(args.manifest, encoding="utf-8") as f:
                manifest = load_manifest(f.read())
        verified = verify_chain(records, args.capacity, args.allow_truncated, manifest=manifest)
        report = build_transcript(verified, quotas=quotas, find=args.find or None, manifest=manifest,
                                  explain=tuple(args.explain) if args.explain else None,
                                  explain_sample=args.explain_sample, blast=args.blast_radius or None)
        if args.json is not None:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError,
            RecursionError, MemoryError, CheckerError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(render_text(report, args.by))
    return 0 if report.get("quota_ok", True) else 2


if __name__ == "__main__":
    sys.exit(main())
