"""
reservoir.attest — Hash-chained attestation for the exact PER buffer.

Every buffer mutation appends a MutationRecord.
Every batch sample appends a SampleAttestation.
Both are chained via BLAKE2b digests into a single log.

Schema (canonical JSON — sorted keys, no spaces):
  MutationRecord:
    digest, op, index, old_priority_int, new_priority_int, op_counter, prev_digest
    Optional, present together when the buffer uses age decay:
      base_priority_int, entry_version, base_epoch
    Optional on "evict" only: reason ("stale" | "capacity" | "explicit" | "drift" | "quarantine")
    Only with reason "quarantine", both required: predicate (the text of the
      predicate that selected the entry) and note (the operator's reason)
    Optional on "insert" only: content_digest (64 hex chars, BLAKE2b-256 of
      the stored example, see rollout.py) and, only together with it,
      source (the caller's tag for where the prompt came from)

  SampleAttestation:
    digest, op_counter, root_total, samples[...], prev_digest

  Age-decay records (only in logs of a decayed buffer):
    decay_config    — must be the first record; the parameters the checker
                      needs to recompute every leaf (half_life, max_policy_age,
                      capacity, bit widths, rebase_slack, reset_age_on_update)
                      and, optionally, the draw configuration (seed, buffer_id,
                      alpha, beta) that lets it recompute every draw and weight
    advance_version — old_version, new_version; expired entries must be
                      evicted (reason "stale") before any other record
    rebase          — old_base_epoch, new_base_epoch, root_total_before,
                      root_total_after; one record for the whole shift

Optional fields are absent, not null, when unused, so a log written
without decay is byte-for-byte identical to one written before the
extension existed.

Integers are encoded as strings to avoid JSON precision limits; booleans
stay booleans. Digests are hex-encoded BLAKE2b-256 of canonical UTF-8.

See docs/design.md §4 and §7.4 for the full schema.
"""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from typing import Optional, Sequence

# Domain separator for attestation hashing
_PERSON = b"attest\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"  # 16 bytes
_GENESIS = "genesis"

_MUTATION_OPS = ("insert", "update", "evict")
_EVICT_REASONS = ("stale", "capacity", "explicit", "drift", "age", "quarantine")
_DECAY_FIELDS = ("base_priority_int", "entry_version", "base_epoch")
_CONTENT_DIGEST_LENGTH = 64   # hex characters of a BLAKE2b-256 digest
LOG_FORMAT = "3"              # schema version written into decay_config; "2" added batch witnesses and telemetry, "3" quarantine evictions
TELEMETRY_COUNTERS = ("batch_rows", "replaced_rows", "declined_rows", "dead_groups", "near_dead_groups")
# Field names a telemetry record owns; a reported measurement may not reuse one. The checker
# (reservoir_checker.telemetry) keeps the same list; the two must not drift apart.
TELEMETRY_RESERVED = frozenset(TELEMETRY_COUNTERS) | {
    "op", "step", "prev_digest", "digest", "sample_op_counter", "ess_num", "ess_den",
    "staleness_max", "staleness_sum", "reported", "log_ratios", "policy", "decisions",
}
# The staleness policy block and one decision per draw (reservoir.integrations._trl_staleness); the
# checker (reservoir_checker.staleness) keeps the same field lists.
TELEMETRY_POLICY_FIELDS = ("max_age", "ess_floor", "mass_cap", "max_log_ratio", "max_declines_per_step", "group_size")
TELEMETRY_DECISION_FIELDS = ("draw", "row", "group", "reason", "scale_num", "scale_den")
TELEMETRY_DECLINE_REASONS = ("drift", "age", "ess")
_MAX_SOURCE_LENGTH = 256      # same bound as reservoir.rollout.MAX_SOURCE_LENGTH
_MAX_PREDICATE_LENGTH = 1024  # same bound as reservoir.rollout_quarantine.MAX_PREDICATE_LENGTH


def _require_int(value: object, name: str, minimum: int = 0) -> int:
    """A true int (not bool) >= minimum, else ValueError naming the field."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _canonical_json(obj: dict) -> bytes:
    """Serialize a dict to canonical JSON bytes (sorted keys, no spaces, UTF-8)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _blake2b_digest(data: bytes) -> str:
    """BLAKE2b-256 digest of data, hex-encoded."""
    h = hashlib.blake2b(data, digest_size=32, person=_PERSON)
    return h.hexdigest()


def _staleness_fields(
    has_sample: bool, log_ratios: Optional[Sequence[float]], policy: Optional[dict], decisions: Optional[Sequence[dict]],
) -> dict:
    """Validate and serialize the optional ``log_ratios``, ``policy`` and ``decisions`` fields of a telemetry record."""
    if log_ratios is None:
        if policy is not None or decisions is not None:
            raise ValueError("telemetry: policy and decisions require log_ratios")
        return {}
    if not has_sample:
        raise ValueError("telemetry: log_ratios need a sample (a step that replayed rows)")
    out: dict = {"log_ratios": []}
    for k, value in enumerate(log_ratios):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"telemetry: log_ratios[{k}] must be a float, got {value!r}")
        out["log_ratios"].append(float(value).hex())
    if (policy is None) != (decisions is None):
        raise ValueError("telemetry: policy and decisions are written together or not at all")
    if policy is None:
        return out
    if not isinstance(policy, dict) or set(policy) != set(TELEMETRY_POLICY_FIELDS):
        raise ValueError(f"telemetry: policy must be a dict with exactly the fields {list(TELEMETRY_POLICY_FIELDS)}")
    for name in ("max_age", "max_declines_per_step", "group_size"):
        if policy[name] is not None:
            _require_int(policy[name], f"policy.{name}", 1 if name == "group_size" else 0)
    for name in ("ess_floor", "mass_cap", "max_log_ratio"):
        value = policy[name]
        if value is not None and (not isinstance(value, str) or float.fromhex(value).hex() != value
                                  or not math.isfinite(float.fromhex(value))):
            raise ValueError(f"telemetry: policy.{name} must be a canonical finite float.hex() string or None")
    if not isinstance(decisions, (list, tuple)) or len(decisions) != len(out["log_ratios"]):
        raise ValueError("telemetry: decisions must list one entry per draw")
    out["policy"] = dict(policy)
    out["decisions"] = []
    for k, entry in enumerate(decisions):
        if not isinstance(entry, dict) or set(entry) != set(TELEMETRY_DECISION_FIELDS):
            raise ValueError(f"telemetry: decisions[{k}] must have exactly the fields {list(TELEMETRY_DECISION_FIELDS)}")
        if entry["draw"] != k:
            raise ValueError(f"telemetry: decisions[{k}] is for draw {entry['draw']!r}; decisions are in draw order")
        _require_int(entry["row"], f"decisions[{k}].row", 0)
        _require_int(entry["group"], f"decisions[{k}].group", 0)
        if entry["reason"] is not None and entry["reason"] not in TELEMETRY_DECLINE_REASONS:
            raise ValueError(f"telemetry: decisions[{k}].reason must be None or one of {TELEMETRY_DECLINE_REASONS}")
        num, den = entry["scale_num"], entry["scale_den"]
        if not (isinstance(num, str) and isinstance(den, str) and num.isdigit() and den.isdigit() and int(den) > 0):
            raise ValueError(f"telemetry: decisions[{k}] scale must be decimal integer strings with a positive denominator")
        scale = Fraction(int(num), int(den))
        if (scale.numerator, scale.denominator) != (int(num), int(den)) or not 0 < scale <= 1:
            raise ValueError(f"telemetry: decisions[{k}] scale {num}/{den} must be a reduced fraction in (0, 1]")
        out["decisions"].append(dict(entry))
    return out


def _digest_record(record: dict, exclude_key: str = "digest") -> str:
    """Compute digest of a record, excluding the digest field itself."""
    record_without_digest = {k: v for k, v in record.items() if k != exclude_key}
    return _blake2b_digest(_canonical_json(record_without_digest))


class AttestationLog:
    """Mutable append-only attestation log.

    Maintains a chain of MutationRecord and SampleAttestation entries,
    each linked to the previous via its digest.

    The log can be serialized and replayed by checker/verify.py.
    """

    def __init__(self) -> None:
        self._records: list[dict] = []
        self._head_digest: str = _GENESIS

    @property
    def head_digest(self) -> str:
        """Digest of the most recent record (or 'genesis' if empty)."""
        return self._head_digest

    @property
    def records(self) -> list[dict]:
        """All records in insertion order."""
        return list(self._records)

    def append_mutation(
        self,
        op: str,
        index: int,
        old_priority_int: int,
        new_priority_int: int,
        op_counter: int,
        *,
        base_priority_int: Optional[int] = None,
        entry_version: Optional[int] = None,
        base_epoch: Optional[int] = None,
        reason: Optional[str] = None,
        content_digest: Optional[str] = None,
        source: Optional[str] = None,
        predicate: Optional[str] = None,
        note: Optional[str] = None,
    ) -> dict:
        """Append a MutationRecord and return it.

        Parameters
        ----------
        op : str
            Operation type: "insert", "update", or "evict".
        index : int
            Buffer position affected.
        old_priority_int : int
            Priority integer before this operation.
        new_priority_int : int
            Priority integer after this operation.
        op_counter : int
            Buffer's monotone operation counter at this point.
        base_priority_int, entry_version, base_epoch : int, keyword-only
            The decay inputs (q, t, B) from which ``new_priority_int`` is
            derived. Given all together or not at all. With them, the
            checker recomputes the leaf instead of trusting it.
        reason : str, keyword-only
            Why an entry was evicted; "evict" only, and only together with
            the decay fields.
        content_digest : str, keyword-only
            64 lowercase hex characters identifying the stored example;
            "insert" only. Lets a verifier say which example a slot held.
        source : str, keyword-only
            The caller's provenance tag for the example; requires
            ``content_digest``.
        predicate, note : str, keyword-only
            The text of the predicate that selected an entry for
            quarantine and the operator's reason; both required with, and
            only valid with, ``reason="quarantine"``.

        Returns
        -------
        dict
            The complete MutationRecord with digest.
        """
        if op not in _MUTATION_OPS:
            raise ValueError(f"Unknown op: {op!r}")

        record = {
            "op": op,
            "index": index,
            "old_priority_int": str(old_priority_int),
            "new_priority_int": str(new_priority_int),
            "op_counter": op_counter,
            "prev_digest": self._head_digest,
        }
        record.update(
            _decay_fields(op, base_priority_int, entry_version, base_epoch, reason)
        )
        record.update(_quarantine_fields(reason, predicate, note))
        record.update(_content_fields(op, content_digest, source))
        return self._commit(record)

    def append_decay_config(
        self,
        half_life: int,
        max_policy_age: int,
        capacity: int,
        priority_bits: int,
        priority_frac_bits: int,
        table_frac_bits: int,
        rebase_slack: int,
        reset_age_on_update: bool,
        *,
        seed: Optional[int] = None,
        buffer_id: Optional[int] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> dict:
        """Append the decay configuration. Must be the very first record.

        The checker derives the decay table from ``half_life`` and
        ``table_frac_bits`` itself; the table is never shipped. With
        ``seed``, ``buffer_id``, ``alpha`` and ``beta`` (all four or none)
        the checker can also recompute every keyed draw and every
        importance weight; the floats are written in ``float.hex()`` form.
        """
        if self._records:
            raise ValueError("decay_config must be the first record of the log")
        if not isinstance(reset_age_on_update, bool):
            raise ValueError(
                f"reset_age_on_update must be a bool, got {reset_age_on_update!r}"
            )
        record = {
            "op": "decay_config",
            "format": LOG_FORMAT,
            "half_life": str(_require_int(half_life, "half_life", 1)),
            "max_policy_age": str(_require_int(max_policy_age, "max_policy_age", 0)),
            "capacity": str(_require_int(capacity, "capacity", 1)),
            "priority_bits": str(_require_int(priority_bits, "priority_bits", 1)),
            "priority_frac_bits": str(_require_int(priority_frac_bits, "priority_frac_bits", 0)),
            "table_frac_bits": str(_require_int(table_frac_bits, "table_frac_bits", 1)),
            "rebase_slack": str(_require_int(rebase_slack, "rebase_slack", 0)),
            "reset_age_on_update": reset_age_on_update,
            "prev_digest": self._head_digest,
        }
        record.update(_draw_fields(seed, buffer_id, alpha, beta))
        return self._commit(record)

    def append_advance_version(self, old_version: int, new_version: int, op_counter: int) -> dict:
        """Append an advance_version record: the buffer moved from old to new version."""
        _require_int(old_version, "old_version", 0)
        _require_int(new_version, "new_version", 0)
        if new_version < old_version:
            raise ValueError(
                f"new_version ({new_version}) must be >= old_version ({old_version})"
            )
        record = {
            "op": "advance_version",
            "old_version": str(old_version),
            "new_version": str(new_version),
            "op_counter": op_counter,
            "prev_digest": self._head_digest,
        }
        return self._commit(record)

    def append_rebase(
        self,
        old_base_epoch: int,
        new_base_epoch: int,
        root_total_before: int,
        root_total_after: int,
        op_counter: int,
    ) -> dict:
        """Append a rebase record: every leaf was right-shifted by the epoch difference."""
        _require_int(old_base_epoch, "old_base_epoch", 0)
        _require_int(new_base_epoch, "new_base_epoch", 0)
        if new_base_epoch <= old_base_epoch:
            raise ValueError(
                f"new_base_epoch ({new_base_epoch}) must be > old_base_epoch ({old_base_epoch})"
            )
        record = {
            "op": "rebase",
            "old_base_epoch": str(old_base_epoch),
            "new_base_epoch": str(new_base_epoch),
            "root_total_before": str(_require_int(root_total_before, "root_total_before", 0)),
            "root_total_after": str(_require_int(root_total_after, "root_total_after", 0)),
            "op_counter": op_counter,
            "prev_digest": self._head_digest,
        }
        return self._commit(record)

    def _commit(self, record: dict) -> dict:
        """Digest the record, link it to the chain head, and store it."""
        digest = _digest_record(record)
        record["digest"] = digest
        self._records.append(record)
        self._head_digest = digest
        return record

    def append_sample(
        self,
        op_counter: int,
        root_total: int,
        samples: list[dict],
    ) -> dict:
        """Append a SampleAttestation and return it.

        Parameters
        ----------
        op_counter : int
            Buffer's op_counter at the time of sampling.
        root_total : int
            The exact tree total used for sampling.
        samples : list[dict]
            Per-sample dicts with keys:
              leaf_index, draw_int, prob_num, prob_den,
              is_weight_num, is_weight_den, rejection_count

        Returns
        -------
        dict
            The complete SampleAttestation with digest.
        """
        # Encode integer fields in samples as strings
        encoded_samples = []
        for s in samples:
            encoded_samples.append({
                "leaf_index": s["leaf_index"],
                "draw_int": str(s["draw_int"]),
                "prob_num": str(s["prob_num"]),
                "prob_den": str(s["prob_den"]),
                "is_weight_num": str(s["is_weight_num"]),
                "is_weight_den": str(s["is_weight_den"]),
                "rejection_count": s["rejection_count"],
            })

        record = {
            "op": "sample",
            "op_counter": op_counter,
            "root_total": str(root_total),
            "samples": encoded_samples,
            "prev_digest": self._head_digest,
        }
        digest = _digest_record(record)
        record["digest"] = digest

        self._records.append(record)
        self._head_digest = digest
        return record

    def append_batch(
        self,
        step: int,
        sample_op_counter: int,
        batch_rows: int,
        replaced: list[tuple[int, int, str]],
        tensor_digest: str,
        declined: Optional[list[int]] = None,
    ) -> dict:
        """Append a BatchWitness: which training-batch rows hold which draws.

        Parameters
        ----------
        step : int
            The trainer's step (model version) the batch was built at.
        sample_op_counter : int
            ``op_counter`` of the sample record whose draws filled the rows.
        batch_rows : int
            Number of rows in the training batch.
        replaced : list of (row, draw, content_digest)
            Row ``row`` now holds draw ``draw`` (position in the sample
            record) whose example has ``content_digest``. Rows and draws
            must be distinct.
        tensor_digest : str
            64 hex characters committing to the final tensors.
        declined : list of int, optional
            Draw positions the adapter refused to place (see evict reason
            ``"drift"``); together with the placed draws they must cover
            the sample record exactly. Absent when empty.
        """
        _require_int(step, "step", 0)
        _require_int(sample_op_counter, "sample_op_counter", 0)
        _require_int(batch_rows, "batch_rows", 1)
        if not _is_hex_digest(tensor_digest):
            raise ValueError(f"tensor_digest must be {_CONTENT_DIGEST_LENGTH} lowercase hex characters")
        rows = [r for r, _, _ in replaced]
        draws = [d for _, d, _ in replaced]
        declined = list(declined or [])
        if len(set(rows)) != len(rows) or len(set(draws + declined)) != len(draws) + len(declined):
            raise ValueError("replaced rows, placed draws and declined draws must be distinct")
        for d in declined:
            _require_int(d, "declined draw", 0)
        entries = []
        for row, draw, digest in replaced:
            _require_int(row, "row", 0)
            _require_int(draw, "draw", 0)
            if row >= batch_rows:
                raise ValueError(f"row {row} is outside a batch of {batch_rows} rows")
            if not _is_hex_digest(digest):
                raise ValueError(f"content_digest of row {row} must be {_CONTENT_DIGEST_LENGTH} lowercase hex characters")
            entries.append({"row": row, "draw": draw, "content_digest": digest})
        record = {
            "op": "batch",
            "step": str(step),
            "sample_op_counter": sample_op_counter,
            "batch_rows": batch_rows,
            "replaced": entries,
            "tensor_digest": tensor_digest,
            "prev_digest": self._head_digest,
        }
        if declined:
            record["declined"] = declined
        return self._commit(record)

    def append_telemetry(
        self,
        step: int,
        counts: dict,
        sample_op_counter: Optional[int],
        ess: Optional[Fraction],
        staleness_max: Optional[int],
        staleness_sum: Optional[int],
        reported: dict,
        log_ratios: Optional[Sequence[float]] = None,
        policy: Optional[dict] = None,
        decisions: Optional[Sequence[dict]] = None,
    ) -> dict:
        """Append a telemetry record: integer counters, exact ESS and staleness, carried floats.

        ``counts`` must hold non-negative ints for ``batch_rows``,
        ``replaced_rows``, ``declined_rows``, ``dead_groups`` and
        ``near_dead_groups``. With ``sample_op_counter`` the exact values
        are required and the checker recomputes them; ``reported`` maps
        names to finite floats, written in ``float.hex()`` form and listed
        under ``"reported"`` so a reader can tell carried from verified.

        Format-3 additive fields, all optional: ``log_ratios`` is one
        float per draw of the sample (any float, non-finite included),
        written as ``float.hex()`` so the checker can recompute the reported
        statistics and replay the policy exactly; ``policy`` is the
        staleness policy block and ``decisions`` one entry per draw, both
        already in record form (``StalenessPolicy.to_record``,
        ``RowDecision.to_record``), and both require ``log_ratios``.
        """
        _require_int(step, "step", 0)
        names = TELEMETRY_COUNTERS
        if set(counts) != set(names):
            raise ValueError(f"telemetry counts must be exactly {names}, got {sorted(counts)}")
        record: dict = {"op": "telemetry", "step": str(step), "prev_digest": self._head_digest}
        record.update({name: _require_int(counts[name], name, 0) for name in names})
        if sample_op_counter is not None:
            if ess is None or staleness_max is None or staleness_sum is None:
                raise ValueError("telemetry for a replayed step needs ess, staleness_max and staleness_sum")
            if not isinstance(ess, Fraction) or ess <= 0:
                raise ValueError(f"ess must be a positive Fraction, got {ess!r}")
            record["sample_op_counter"] = _require_int(sample_op_counter, "sample_op_counter", 0)
            record["ess_num"], record["ess_den"] = str(ess.numerator), str(ess.denominator)
            record["staleness_max"] = str(_require_int(staleness_max, "staleness_max", 0))
            record["staleness_sum"] = str(_require_int(staleness_sum, "staleness_sum", 0))
        for name, value in reported.items():
            if not isinstance(name, str) or not name or name in TELEMETRY_RESERVED:
                raise ValueError(f"reported name {name!r} is empty or a reserved telemetry field")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"reported {name} must be a finite float, got {value!r}")
            record[name] = float(value).hex()
        record["reported"] = sorted(reported)
        record.update(_staleness_fields(sample_op_counter is not None, log_ratios, policy, decisions))
        return self._commit(record)

    def restore(self, records: list[dict]) -> None:
        """Replace the log's contents with ``records``, re-verifying every link.

        Used when a buffer is recovered from a saved state. Each record's
        digest must match its content and chain to the one before, so a
        corrupt snapshot cannot resurrect a log the checker would reject.
        Raises ``ValueError`` and leaves the log unchanged otherwise.
        """
        head = _GENESIS
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"record {i} is not a dict")
            if record.get("prev_digest") != head:
                raise ValueError(f"record {i} does not chain to the previous record")
            if _digest_record(record) != record.get("digest"):
                raise ValueError(f"record {i} has a digest that does not match its content")
            head = record["digest"]
        self._records = [dict(r) for r in records]
        self._head_digest = head

    def to_json_lines(self) -> str:
        """Serialize all records as newline-separated canonical JSON."""
        lines = []
        for record in self._records:
            lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return "\n".join(lines)

    @classmethod
    def from_json_lines(cls, data: str) -> "AttestationLog":
        """Reconstruct an AttestationLog from serialized JSON lines."""
        log = cls()
        for line in data.strip().split("\n"):
            if not line:
                continue
            record = json.loads(line)
            log._records.append(record)
            log._head_digest = record["digest"]
        return log


def _decay_fields(
    op: str,
    base_priority_int: Optional[int],
    entry_version: Optional[int],
    base_epoch: Optional[int],
    reason: Optional[str],
) -> dict:
    """Validate the optional decay fields of a mutation and return those to add.

    Returns an empty dict for a legacy record so its serialisation (and
    digest) is unchanged.
    """
    given = {
        "base_priority_int": base_priority_int,
        "entry_version": entry_version,
        "base_epoch": base_epoch,
    }
    present = [name for name, value in given.items() if value is not None]
    if not present:
        if reason is not None:
            raise ValueError("reason requires the decay fields base_priority_int, entry_version, base_epoch")
        return {}
    for name in _DECAY_FIELDS:
        if given[name] is None:
            raise ValueError(f"{name} is required when any decay field is given")
        _require_int(given[name], name, 0)
    fields = {name: str(given[name]) for name in _DECAY_FIELDS}
    if reason is not None:
        if op != "evict":
            raise ValueError(f"reason is only valid on evict records, not {op!r}")
        if reason not in _EVICT_REASONS:
            raise ValueError(f"reason must be one of {_EVICT_REASONS}, got {reason!r}")
        fields["reason"] = reason
    return fields


def _require_text(value: object, name: str, limit: int) -> str:
    """A non-empty printable string of at most ``limit`` characters."""
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError(f"{name} must be a non-empty string of at most {limit} characters, got {value!r}")
    if not value.isprintable() or not value.strip():
        raise ValueError(f"{name} must be printable and not only whitespace, got {value!r}")
    return value


def _quarantine_fields(reason: Optional[str], predicate: Optional[str], note: Optional[str]) -> dict:
    """The two texts of a quarantine evict; empty for every other record."""
    if reason != "quarantine":
        if predicate is not None or note is not None:
            raise ValueError("predicate and note are only valid with reason 'quarantine'")
        return {}
    if predicate is None:
        raise ValueError("a quarantine evict needs the predicate text")
    if note is None:
        raise ValueError("a quarantine evict needs a note (the reason for the quarantine)")
    return {
        "predicate": _require_text(predicate, "predicate", _MAX_PREDICATE_LENGTH),
        "note": _require_text(note, "note", _MAX_SOURCE_LENGTH),
    }


def _draw_fields(
    seed: Optional[int], buffer_id: Optional[int], alpha: Optional[float], beta: Optional[float]
) -> dict:
    """The optional draw configuration of a decay_config record: all four or none."""
    given = {"seed": seed, "buffer_id": buffer_id, "alpha": alpha, "beta": beta}
    present = [name for name, value in given.items() if value is not None]
    if not present:
        return {}
    if len(present) != len(given):
        raise ValueError(f"seed, buffer_id, alpha and beta must be given together, got {present}")
    for name in ("seed", "buffer_id"):
        value = _require_int(given[name], name, 0)
        if value >= 1 << 64:
            raise ValueError(f"{name} must fit in 64 bits, got {value}")
    for name, minimum_exclusive in (("alpha", True), ("beta", False)):
        value = given[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{name} must be a finite real number, got {value!r}")
        if (minimum_exclusive and value <= 0) or (not minimum_exclusive and value < 0):
            raise ValueError(f"{name} must be {'> 0' if minimum_exclusive else '>= 0'}, got {value!r}")
    return {
        "seed": str(seed), "buffer_id": str(buffer_id),
        "alpha": float(alpha).hex(), "beta": float(beta).hex(),  # type: ignore[arg-type]
    }


def _is_hex_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _CONTENT_DIGEST_LENGTH
        and all(c in "0123456789abcdef" for c in value)
    )


def _content_fields(op: str, content_digest: Optional[str], source: Optional[str]) -> dict:
    """Validate the optional content fields of a mutation and return those to add.

    Empty for a record without them, so legacy serialisation is unchanged.
    """
    if content_digest is None:
        if source is not None:
            raise ValueError("source requires content_digest")
        return {}
    if op != "insert":
        raise ValueError(f"content_digest is only valid on insert records, not {op!r}")
    if not _is_hex_digest(content_digest):
        raise ValueError(
            f"content_digest must be {_CONTENT_DIGEST_LENGTH} lowercase hex characters, "
            f"got {content_digest!r}"
        )
    fields = {"content_digest": content_digest}
    if source is not None:
        if (
            not isinstance(source, str)
            or not source.strip()
            or len(source) > _MAX_SOURCE_LENGTH
            or not source.isprintable()
        ):
            raise ValueError(
                f"source must be a printable str of 1..{_MAX_SOURCE_LENGTH} characters, got {source!r}"
            )
        fields["source"] = source
    return fields


def make_sample_entry(
    leaf_index: int,
    draw_int: int,
    priority_int: int,
    root_total: int,
    is_weight: Fraction,
    rejection_count: int = 0,
) -> dict:
    """Build a sample dict for a single draw, suitable for append_sample().

    Computes exact probability: prob = priority_int / root_total (Fraction),
    stored as reduced numerator/denominator.

    Parameters
    ----------
    leaf_index : int
        The sampled buffer position.
    draw_int : int
        The draw integer used in prefix_sum_locate.
    priority_int : int
        The integerized priority at leaf_index.
    root_total : int
        The tree total at sample time.
    is_weight : Fraction
        The exact IS weight (normalized).
    rejection_count : int
        Number of hash blocks rejected before acceptance.

    Returns
    -------
    dict
        Sample entry with exact probability and IS weight as integers.
    """
    prob = Fraction(priority_int, root_total)  # Already in lowest terms via GCD
    return {
        "leaf_index": leaf_index,
        "draw_int": draw_int,
        "prob_num": prob.numerator,
        "prob_den": prob.denominator,
        "is_weight_num": is_weight.numerator,
        "is_weight_den": is_weight.denominator,
        "rejection_count": rejection_count,
    }
