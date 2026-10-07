"""
reservoir_checker.telemetry — Replay-health records, recomputed where the log allows.

A ``telemetry`` record is one adapter step's replay health. Its integer
counters (``batch_rows``, ``replaced_rows``, ``declined_rows``,
``dead_groups``, ``near_dead_groups``) are carried: the log cannot see the
training batch. For a replayed step the record names the sample and
carries the exact effective sample size ``(sum w)^2 / sum w^2`` of that
sample's importance weights and the maximum and sum of the rows' ages in
model versions; both are recomputed here from the sample record and the
replayed buffer state and must match.

Float fields listed under ``reported`` are measurements (the log-ratio
statistics between stored behavior logprobs and the current policy) that
must be canonical finite ``float.hex()`` strings. A record that also carries
``log_ratios``, one hex float per draw, has those statistics recomputed
from it; one that carries a staleness ``policy`` and its per-draw
``decisions`` has every decision replayed from the policy, the log-ratios,
the sample's weights and the live slots' ages
(``reservoir_checker.staleness``), the kept rows matched against the batch
witness and the declines against the counters. A record with log-ratios
but no policy may decline nothing (the 0.6.0 gate always wrote its
threshold; a new log without a policy has no gate).

Counters are cross-checked against the sample record
(``replaced_rows + declined_rows`` equals the number of draws) and, when
the sample has a batch witness, against it (same step, same batch size,
same placed and declined counts). The ages are taken from the live slots,
which must still hold the examples the sample drew.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from reservoir_checker.content import ContentState, _int_field
from reservoir_checker.decay_replay import CheckerError
from reservoir_checker.staleness import Decision, Policy, hex_float, parse_decisions, parse_policy, verify_decisions

COUNTERS = ("batch_rows", "replaced_rows", "declined_rows", "dead_groups", "near_dead_groups")
# Mirrors reservoir.attest.TELEMETRY_RESERVED; not imported, by design.
RESERVED = frozenset(COUNTERS) | {"op", "step", "prev_digest", "digest", "sample_op_counter", "ess_num",
                                  "ess_den", "staleness_max", "staleness_sum", "reported",
                                  "log_ratios", "policy", "decisions"}
STALENESS_FIELDS = ("log_ratios", "policy", "decisions")
MEAN_ABS, MAX_ABS = "log_ratio_mean_abs", "log_ratio_max_abs"


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
    log_ratios: tuple[float, ...] = ()              # one per draw when the record carries them
    policy: Optional[Policy] = None                 # the staleness policy when one was recorded
    decisions: tuple[Decision, ...] = ()            # one per draw when a policy was recorded


def _hex_float(record: dict, name: str, where: str) -> float:
    return hex_float(record.get(name), f"{where}: {name}")


def _log_ratios(record: dict, n_draws: int, where: str) -> tuple[float, ...]:
    values = record.get("log_ratios")
    if not isinstance(values, list) or len(values) != n_draws:
        raise CheckerError(f"{where}: log_ratios must list one hex float per draw ({n_draws})")
    return tuple(hex_float(v, f"{where}: log_ratios[{k}]", allow_non_finite=True) for k, v in enumerate(values))


def _check_reported_statistics(ratios: tuple[float, ...], reported: dict, where: str) -> None:
    """The adapter's mean and max of the finite |log-ratio|s, in the same float arithmetic."""
    magnitudes = [abs(r) for r in ratios if math.isfinite(r)]
    expected = {MEAN_ABS: sum(magnitudes) / len(magnitudes), MAX_ABS: max(magnitudes)} if magnitudes else {}
    for name in (MEAN_ABS, MAX_ABS):
        if (name in reported) != (name in expected):
            raise CheckerError(f"{where}: {name} must be reported exactly when a finite log-ratio exists")
        if name in expected and reported[name] != expected[name]:
            raise CheckerError(f"{where}: reported {name} {reported[name]!r} but the log-ratios give {expected[name]!r}")


def _check_staleness(record: dict, where: str, draws: list, ages: list[int], counts: dict, witness: Optional[dict],
                     reported: dict, content: ContentState) -> tuple[tuple[float, ...], Optional[Policy], tuple[Decision, ...]]:
    """Verify the optional ``log_ratios``, ``policy`` and ``decisions`` of a replayed step.

    The fields are optional per log, not per record: once one replayed
    step has carried them, every later replayed step must, with the same
    policy, so a tamperer cannot drop them from one record to escape the
    checks. A log may still begin without them (the 0.6.0 shape).
    """
    present = [name for name in STALENESS_FIELDS if name in record]
    if not present:
        if content.staleness_log_ratios_seen:
            raise CheckerError(f"{where}: an earlier replayed step carried log_ratios; a later one cannot leave them out")
        return (), None, ()
    content.staleness_log_ratios_seen = True
    if "log_ratios" not in present:
        raise CheckerError(f"{where}: policy and decisions require log_ratios")
    ratios = _log_ratios(record, len(draws), where)
    _check_reported_statistics(ratios, reported, where)
    if ("policy" in present) != ("decisions" in present):
        raise CheckerError(f"{where}: policy and decisions are recorded together or not at all")
    if "policy" not in present:
        if content.staleness_policy is not None:
            raise CheckerError(f"{where}: an earlier replayed step carried a staleness policy; a later one cannot leave it out")
        if counts["declined_rows"]:
            raise CheckerError(f"{where}: {counts['declined_rows']} rows declined but no policy is recorded")
        return ratios, None, ()
    policy = parse_policy(record["policy"], where)
    if content.staleness_policy is None:
        content.staleness_policy = dict(record["policy"])
    elif content.staleness_policy != record["policy"]:
        raise CheckerError(f"{where}: the staleness policy differs from the one recorded earlier in this log; "
                           "a policy is fixed for a run")
    decisions = parse_decisions(record["decisions"], len(draws), where)
    if any(d.row >= counts["batch_rows"] for d in decisions):
        raise CheckerError(f"{where}: a decision names a row outside the batch of {counts['batch_rows']} rows")
    verify_decisions(policy, decisions, ratios=ratios, is_weights=[d.is_weight for d in draws], ages=ages, where=where)
    kept = [d for d in decisions if d.reason is None]
    if len(kept) != counts["replaced_rows"] or len(decisions) - len(kept) != counts["declined_rows"]:
        raise CheckerError(f"{where}: decisions keep {len(kept)} and decline {len(decisions) - len(kept)} draws but the "
                           f"counters say {counts['replaced_rows']} replaced, {counts['declined_rows']} declined")
    if witness is not None:
        placed = sorted((e["row"], e["draw"]) for e in witness["replaced"])
        if placed != sorted((d.row, d.draw) for d in kept) or sorted(witness["declined"]) != [d.draw for d in decisions if d.reason]:
            raise CheckerError(f"{where}: the decisions disagree with the batch witness about which draws fill which rows")
    return ratios, policy, tuple(decisions)


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
        if any(name in record for name in STALENESS_FIELDS):
            raise CheckerError(f"{where}: log_ratios, policy and decisions need the sample they describe")
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
    ratios, policy, decisions = _check_staleness(record, where, draws, ages, counts, _witness_rows(content, op), reported,
                                                 content)
    return TelemetryPoint(idx, step, counts, op, ess, max(ages), Fraction(sum(ages), len(ages)), reported,
                          ratios, policy, decisions)


def _witness_rows(content: ContentState, op: int) -> Optional[dict]:
    """The witness of sample ``op`` with its row/draw pairs and declined draws, or None without one."""
    witness = next((w for w in content.witnesses if w["sample_op_counter"] == op), None)
    if witness is None:
        return None
    rows = [{"row": w.row, "draw": w.draw} for w in content.witnessed_rows if w.sample_op_counter == op]
    return {"replaced": rows, "declined": list(witness["declined"])}
