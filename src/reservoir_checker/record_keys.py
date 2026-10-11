"""
reservoir_checker.record_keys — The fields each record type may carry.

This module imports nothing from the rest of the reservoir package. The
sets mirror the fields ``reservoir.attest`` writes for each ``op``; they are
written out here, not imported. A field outside its record's set is
provenance the checker never reads, so a reader of the raw log would see a
claim nothing validated: such a record is rejected. Only unknown fields are
refused here; missing and malformed ones are reported by the checks that
read them. ``telemetry`` records have their own rule (extra fields must be
listed under ``reported``; ``reservoir_checker.telemetry``).
"""

from __future__ import annotations

from reservoir_checker.decay_replay import CheckerError

_LINK = frozenset({"op", "prev_digest", "digest"})
_MUTATION = _LINK | {"index", "old_priority_int", "new_priority_int", "op_counter"}
_MUTATION_OPTIONAL = frozenset({
    "base_priority_int", "entry_version", "base_epoch", "reason",   # decay fields
    "predicate", "note",                                            # quarantine evicts
    "content_digest", "source",                                     # content commitments (inserts)
})
_CONFIG = _LINK | {
    "half_life", "max_policy_age", "capacity", "priority_bits", "priority_frac_bits", "table_frac_bits",
    "rebase_slack", "reset_age_on_update",
}

RECORD_KEYS: dict[str, tuple[frozenset, frozenset]] = {
    # op: (fields the producer always writes, fields it writes in some logs)
    "decay_config": (_CONFIG, frozenset({"format", "seed", "buffer_id", "alpha", "beta"})),
    "insert": (_MUTATION, _MUTATION_OPTIONAL),
    "update": (_MUTATION, _MUTATION_OPTIONAL),
    "evict": (_MUTATION, _MUTATION_OPTIONAL),
    "advance_version": (_LINK | {"old_version", "new_version", "op_counter"}, frozenset()),
    "rebase": (_LINK | {"old_base_epoch", "new_base_epoch", "root_total_before", "root_total_after", "op_counter"},
               frozenset()),
    "sample": (_LINK | {"op_counter", "root_total", "samples"}, frozenset({"min_entry_age"})),
    "batch": (_LINK | {"step", "sample_op_counter", "batch_rows", "replaced", "tensor_digest"}, frozenset({"declined"})),
}

DRAW_KEYS = frozenset({
    "leaf_index", "draw_int", "prob_num", "prob_den", "is_weight_num", "is_weight_den", "rejection_count",
    "entry_version",
})
WITNESS_ROW_KEYS = frozenset({"row", "draw", "content_digest"})


def check_record_keys(record: dict, idx: int) -> None:
    """Reject a record of a known op that carries a field its op does not define."""
    keys = RECORD_KEYS.get(record.get("op"))  # type: ignore[arg-type]
    if keys is None:
        return
    required, optional = keys
    check_keys(record, required | optional, f"Record {idx}: {record['op']} record")


def check_keys(obj: object, allowed: frozenset, where: str) -> None:
    """Reject an object with a key outside ``allowed``; non-objects are left to the field checks."""
    if not isinstance(obj, dict):
        return
    extra = set(obj) - allowed
    if extra:
        raise CheckerError(f"{where} has unknown field(s) {sorted(extra, key=str)}")
