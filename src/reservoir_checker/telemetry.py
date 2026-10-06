"""
reservoir_checker.telemetry — Replay-health records, recomputed where the log allows.

A ``telemetry`` record is one adapter step's replay health. Its integer
counters (``batch_rows``, ``replaced_rows``, ``declined_rows``,
``dead_groups``, ``near_dead_groups``) are carried: the log cannot see the
training batch. For a replayed step the record names the sample and
carries the exact effective sample size ``(sum w)^2 / sum w^2`` of that
sample's importance weights and the maximum and sum of the rows' ages in
model versions; both are recomputed here from the sample record and the
replayed buffer state and must match. Any further float fields are
measurements the log cannot recompute (log-ratios between stored behavior
logprobs and the current policy); the record lists their names under
``reported`` and this module checks only that they are canonical finite
``float.hex()`` strings.

Counters are cross-checked against the sample record
(``replaced_rows + declined_rows`` equals the number of draws) and, when
the sample has a batch witness, against it (same step, same batch size,
same placed and declined counts). The ages are taken from the live slots,
which must still hold the examples the sample drew.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from reservoir_checker.content import ContentState, _int_field
from reservoir_checker.decay_replay import CheckerError

COUNTERS = ("batch_rows", "replaced_rows", "declined_rows", "dead_groups", "near_dead_groups")
# Mirrors reservoir.attest.TELEMETRY_RESERVED; not imported, by design.
RESERVED = frozenset(COUNTERS) | {"op", "step", "prev_digest", "digest", "sample_op_counter", "ess_num",
                                  "ess_den", "staleness_max", "staleness_sum", "reported"}


@dataclass(frozen=True)
class TelemetryPoint:
    """One verified telemetry record, with the recomputed quantities as plain numbers."""

    record_index: int
    step: int
    counts: dict
    sample_op_counter: Optional[int]
    ess: Optional[Fraction]
    staleness_max: Optional[int]
    staleness_mean: Optional[Fraction]
    reported: dict


def _hex_float(record: dict, name: str, where: str) -> float:
    value = record.get(name)
    if not isinstance(value, str):
        raise CheckerError(f"{where}: {name} must be a float.hex() string, got {value!r}")
    try:
        parsed = float.fromhex(value)
    except (ValueError, OverflowError) as exc:
        raise CheckerError(f"{where}: {name} is not a hexadecimal float: {exc}") from exc
    if parsed != parsed or parsed in (float("inf"), float("-inf")) or parsed.hex() != value:
        raise CheckerError(f"{where}: {name} must be the canonical float.hex() spelling of a finite value")
    return parsed


def verify_telemetry(record: dict, idx: int, content: ContentState, versions: dict[int, int],
                     current_version: int) -> TelemetryPoint:
    """Check one telemetry record against the sample it names; return the verified point.

    ``versions`` maps live slot -> entry version (the replayed decay
    state) and ``current_version`` is the buffer's version, so the ages of
    the replayed rows can be recomputed.
    """
    where = f"Record {idx}"
    step = _int_field(record, "step", where)
    counts = {name: _int_field(record, name, where) for name in COUNTERS}
    if counts["replaced_rows"] + counts["declined_rows"] > counts["batch_rows"]:
        raise CheckerError(f"{where}: more replaced and declined rows than the batch has")
    reported_names = record.get("reported")
    if not isinstance(reported_names, list) or any(not isinstance(n, str) for n in reported_names):
        raise CheckerError(f"{where}: reported must be a list of field names")
    if reported_names != sorted(set(reported_names)) or any(n in RESERVED for n in reported_names):
        raise CheckerError(f"{where}: reported must be sorted, unique and free of reserved names")
    extra = set(record) - RESERVED
    if extra != set(reported_names):
        raise CheckerError(f"{where}: reported {sorted(reported_names)} does not match the extra fields {sorted(extra)}")
    reported = {name: _hex_float(record, name, where) for name in reported_names}

    if "sample_op_counter" not in record:
        if counts["replaced_rows"] or counts["declined_rows"]:
            raise CheckerError(f"{where}: rows were replaced or declined but no sample is named")
        return TelemetryPoint(idx, step, counts, None, None, None, None, reported)

    op = _int_field(record, "sample_op_counter", where)
    draws = content._samples_by_op.get(op)
    if draws is None or op != content.last_sample_op:
        raise CheckerError(f"{where}: telemetry names sample {op}, which is not the latest sample record")
    if counts["replaced_rows"] + counts["declined_rows"] != len(draws):
        raise CheckerError(
            f"{where}: replaced {counts['replaced_rows']} + declined {counts['declined_rows']} rows but sample "
            f"{op} drew {len(draws)} rollouts"
        )
    total = sum((d.is_weight for d in draws), Fraction(0))
    squares = sum((d.is_weight * d.is_weight for d in draws), Fraction(0))
    ess = total * total / squares
    declared = Fraction(_int_field(record, "ess_num", where), _int_field(record, "ess_den", where))
    if declared != ess:
        raise CheckerError(f"{where}: ess {declared} but the sample's weights give {ess}")
    for d in draws:
        if d.leaf_index not in versions or content.slots.get(d.leaf_index, (None,))[0] != d.content_digest:
            raise CheckerError(
                f"{where}: slot {d.leaf_index} no longer holds the example sample {op} drew; telemetry must "
                "precede evictions or refills of its rows"
            )
    ages = [current_version - versions[d.leaf_index] for d in draws]
    witness = next((w for w in content.witnesses if w["sample_op_counter"] == op), None)
    if witness is not None and (
        witness["step"] != step or witness["batch_rows"] != counts["batch_rows"]
        or witness["replaced"] != counts["replaced_rows"] or len(witness["declined"]) != counts["declined_rows"]
    ):
        raise CheckerError(f"{where}: telemetry counters disagree with the batch witness of sample {op}")
    if _int_field(record, "staleness_max", where) != max(ages) or _int_field(record, "staleness_sum", where) != sum(ages):
        raise CheckerError(f"{where}: staleness max/sum {record['staleness_max']}/{record['staleness_sum']} but "
                           f"the replayed rows give {max(ages)}/{sum(ages)}")
    return TelemetryPoint(idx, step, counts, op, ess, max(ages), Fraction(sum(ages), len(ages)), reported)
