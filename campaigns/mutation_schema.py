"""
campaigns.mutation_schema — Fields no record type defines, and degenerate records.

Each record type has a fixed set of fields (``reservoir_checker.record_keys``).
A field the checker does not read would be provenance nobody validated, so
one extra field on each op, a provenance field (``source``,
``content_digest``) on a ``sample``, an extra field inside a draw or a
witness row, and a sample that draws nothing are each re-chained into an
otherwise honest log. Every one must be rejected.
"""

from __future__ import annotations

import copy

from campaigns.mutation_batch import build_witnessed_chain
from campaigns.mutation_content import build_content_chain
from campaigns.mutation_reasons import rejection, tally
from reservoir_checker.verify import verify_chain

Mutant = tuple[list[dict], str]


def _rechain(records: list[dict], start: int) -> list[dict]:
    from reservoir.attest import _digest_record

    out = copy.deepcopy(records)
    prev = out[start - 1]["digest"] if start > 0 else "genesis"
    for rec in out[start:]:
        rec["prev_digest"] = prev
        rec["digest"] = _digest_record(rec)
        prev = rec["digest"]
    return out


def _edited(records: list[dict], idx: int, edit) -> list[dict]:
    m = copy.deepcopy(records)
    edit(m[idx])
    return _rechain(m, idx)


def _first(records: list[dict], op: str) -> int:
    return next(i for i, r in enumerate(records) if r["op"] == op)


def schema_mutants(content: list[dict], witnessed: list[dict]) -> list[Mutant]:
    mutants: list[Mutant] = []
    for op in ("decay_config", "insert", "update", "evict", "advance_version", "rebase", "sample"):
        mutants.append((_edited(content, _first(content, op), lambda r: r.update(foo="1")), f"extra_field_on_{op}"))
    b = _first(witnessed, "batch")
    s = _first(content, "sample")
    mutants += [
        (_edited(witnessed, b, lambda r: r.update(foo="1")), "extra_field_on_batch"),
        (_edited(content, s, lambda r: r.update(source="licensed")), "source_on_a_sample"),
        (_edited(content, s, lambda r: r.update(content_digest="ab" * 32)), "content_digest_on_a_sample"),
        (_edited(content, s, lambda r: r["samples"][0].update(source="licensed")), "extra_field_in_a_draw"),
        (_edited(witnessed, b, lambda r: r["replaced"][0].update(note="x")), "extra_field_in_a_witness_row"),
        (_edited(content, s, lambda r: r.update(samples=[])), "sample_draws_nothing"),
    ]
    return mutants


def run_schema_category(results: dict) -> None:
    """Append the ``schema`` category to ``results``."""
    content, _ = build_content_chain()
    mutants = schema_mutants(content, build_witnessed_chain())
    tally(results, "schema", [(description, rejection(lambda r=records: verify_chain(r)))
                              for records, description in mutants])
