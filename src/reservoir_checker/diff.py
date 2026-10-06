"""
checker.diff — Where and why two attestation logs diverge.

This module imports nothing from the rest of the reservoir package.

Two runs of the same training script with the same seeds should produce
byte-identical attestation logs: the log is a fingerprint of everything
the replay buffer stored, drew and evicted. When two logs differ, the
interesting question is not *that* they differ but *what kind of record*
differs first, because every record type has a different cause:

================  =====================================================
class             meaning
================  =====================================================
``identical``     same records, same head digest
``config``        the ``decay_config`` records differ: different buffer
                  parameters
``data``          the first differing record is an insert or update, or
                  the two logs perform different operations at that point
                  (one inserts where the other samples or evicts): the
                  *stored examples, their scores or the group sizes*
                  differed upstream (generation, rewards, which groups
                  were dead). Reservoir's draws were identical up to here
``schedule``      the first difference is an ``advance_version``: the
                  runs moved through versions on a different cadence
``sampler``       both records are ``sample`` records of the same size
                  on an identical prefix, so the buffer state was
                  identical and the keyed draws differed. A 0.5.0 log
                  records seed, buffer id and beta in ``decay_config``,
                  so two runs with different seeds differ at record 0
                  (``config``) and this class is reachable only for logs
                  without that configuration, or by a Reservoir defect.
                  Two samples with the same draws and slots but different
                  weights are ``config`` (a different ``beta``); two
                  samples of different sizes are ``data`` (a different
                  number of dead rows)
``internal``      an ``evict`` or ``rebase`` differs on identical state.
                  Which entries expire, which are the oldest, and when a
                  rebase is due are deterministic functions of the
                  preceding records, so this must never happen
``truncated``     one log is a strict prefix of the other
================  =====================================================

Both logs are fully verified first; a log that does not verify is an
error, not a difference. Two logs without ``decay_config`` records do not
record their capacity, so a capacity difference between them surfaces as
``data`` at the first record it affects.

Command line::

    python -m checker.diff run-a/attest.jsonl run-b/attest.jsonl [--json diff.json]

Exit status: 0 identical, 3 different, 1 a log did not verify or could
not be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from reservoir_checker.decay_replay import CheckerError
from reservoir_checker.verify import verify_chain

_IGNORED_FIELDS = frozenset({"digest"})
_CLASS_DETAIL = {
    "identical": "the logs are identical",
    "config": "the buffers were constructed with different parameters",
    "data": "the stored examples, their scores or the group sizes differed upstream of the buffer "
            "(generation, rewards, dead groups); every draw before this point was identical",
    "schedule": "the runs advanced through model versions on a different cadence",
    "sampler": "the keyed draws differed on identical buffer state: for a log without a recorded seed, the "
               "runs used different seeds or buffer ids; for a log that records them, this is a Reservoir defect",
    "internal": "an evict or rebase differs on identical state; both are deterministic functions of "
                "the preceding records and this must never happen",
    "truncated": "one log is a prefix of the other",
}


def _head(records: list[dict]) -> str:
    return records[-1]["digest"] if records else "genesis"


def _classify(a: dict, b: dict) -> str:
    """The class of the first differing record pair; see the module docstring."""
    op_a, op_b = a.get("op"), b.get("op")
    if op_a != op_b:
        ops = {op_a, op_b}
        if "decay_config" in ops:
            return "config"
        if "advance_version" in ops:
            return "schedule"
        if "rebase" in ops:
            return "internal"
        return "data"
    if op_a == "sample":
        return _classify_samples(a, b)
    return {
        "decay_config": "config",
        "insert": "data", "update": "data",
        "evict": "internal", "rebase": "internal",
        "advance_version": "schedule",
    }.get(op_a, "data")


def _classify_samples(a: dict, b: dict) -> str:
    sa, sb = a.get("samples", []), b.get("samples", [])
    if len(sa) != len(sb):
        return "data"
    slots_a = [s.get("leaf_index") for s in sa]
    slots_b = [s.get("leaf_index") for s in sb]
    draws_a = [s.get("draw_int") for s in sa]
    draws_b = [s.get("draw_int") for s in sb]
    if slots_a == slots_b and draws_a == draws_b:
        return "config"        # same draws, same slots, different weights: beta differs
    return "sampler"


def _differing_fields(a: dict, b: dict) -> list[str]:
    keys = (set(a) | set(b)) - _IGNORED_FIELDS
    return sorted(k for k in keys if a.get(k) != b.get(k))


def diff_logs(
    a: list[dict],
    b: list[dict],
    capacity: Optional[int] = None,
    allow_truncated: bool = False,
) -> dict:
    """Verify both logs, then locate and classify their first difference.

    Raises ``CheckerError`` if either log does not verify.
    """
    verify_chain(a, capacity, allow_truncated)
    verify_chain(b, capacity, allow_truncated)
    result: dict = {
        "records_a": len(a), "records_b": len(b),
        "head_a": _head(a), "head_b": _head(b),
        "identical": False, "first_difference": None, "class": "identical",
        "ops": None, "differing_fields": [], "record_a": None, "record_b": None,
    }
    common = min(len(a), len(b))
    first = next((i for i in range(common) if a[i] != b[i]), None)
    if first is None:
        if len(a) == len(b):
            result["identical"] = True
        else:
            longer = a if len(a) > common else b
            result["class"] = "truncated"
            result["first_difference"] = common
            result["ops"] = [a[common].get("op") if len(a) > common else None,
                             b[common].get("op") if len(b) > common else None]
            result["record_a" if longer is a else "record_b"] = longer[common]
    else:
        ra, rb = a[first], b[first]
        result.update(first_difference=first, ops=[ra.get("op"), rb.get("op")],
                      differing_fields=_differing_fields(ra, rb), record_a=ra, record_b=rb)
        result["class"] = _classify(ra, rb)
    result["detail"] = _CLASS_DETAIL[result["class"]]
    return result


def render_text(result: dict) -> str:
    """Human-readable rendering of a ``diff_logs`` result."""
    lines = [
        f"a: {result['records_a']} records, head {result['head_a'][:16]}…",
        f"b: {result['records_b']} records, head {result['head_b'][:16]}…",
    ]
    if result["identical"]:
        lines.append("identical")
        return "\n".join(lines)
    lines.append(f"first difference at record {result['first_difference']} ({result['class']}): {result['detail']}")
    if result["ops"]:
        lines.append(f"  ops: a={result['ops'][0]} b={result['ops'][1]}")
    if result["differing_fields"]:
        lines.append(f"  differing fields: {', '.join(result['differing_fields'])}")
    return "\n".join(lines)


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point. See the module docstring for exit codes."""
    parser = argparse.ArgumentParser(
        description="Verify two reservoir attestation logs and classify their first difference.",
    )
    parser.add_argument("log_a")
    parser.add_argument("log_b")
    parser.add_argument("--capacity", type=int, default=None, help="capacity for logs without decay_config")
    parser.add_argument("--allow-truncated", action="store_true", help="accept logs cut off mid-advance")
    parser.add_argument("--json", default=None, metavar="PATH", help="also write the result as JSON")
    args = parser.parse_args(argv)
    try:
        result = diff_logs(_load(args.log_a), _load(args.log_b), args.capacity, args.allow_truncated)
        if args.json is not None:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError,
            RecursionError, CheckerError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(render_text(result))
    return 0 if result["identical"] else 3


if __name__ == "__main__":
    sys.exit(main())
