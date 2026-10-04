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
    Optional on "evict" only: reason ("stale" | "capacity" | "explicit")
    Optional on "insert" only: content_digest (64 hex chars, BLAKE2b-256 of
      the stored example, see rollout.py) and, only together with it,
      source (the caller's tag for where the prompt came from)

  SampleAttestation:
    digest, op_counter, root_total, samples[...], prev_digest

  Age-decay records (only in logs of a decayed buffer):
    decay_config    — must be the first record; the parameters the checker
                      needs to recompute every leaf (half_life, max_policy_age,
                      capacity, bit widths, rebase_slack, reset_age_on_update)
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
from fractions import Fraction
from typing import Optional

# Domain separator for attestation hashing
_PERSON = b"attest\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"  # 16 bytes
_GENESIS = "genesis"

_MUTATION_OPS = ("insert", "update", "evict")
_EVICT_REASONS = ("stale", "capacity", "explicit")
_DECAY_FIELDS = ("base_priority_int", "entry_version", "base_epoch")
_CONTENT_DIGEST_LENGTH = 64   # hex characters of a BLAKE2b-256 digest
_MAX_SOURCE_LENGTH = 256      # same bound as reservoir.rollout.MAX_SOURCE_LENGTH


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
    ) -> dict:
        """Append the decay configuration. Must be the very first record.

        The checker derives the decay table from ``half_life`` and
        ``table_frac_bits`` itself; the table is never shipped.
        """
        if self._records:
            raise ValueError("decay_config must be the first record of the log")
        if not isinstance(reset_age_on_update, bool):
            raise ValueError(
                f"reset_age_on_update must be a bool, got {reset_age_on_update!r}"
            )
        record = {
            "op": "decay_config",
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
    if (
        not isinstance(content_digest, str)
        or len(content_digest) != _CONTENT_DIGEST_LENGTH
        or any(c not in "0123456789abcdef" for c in content_digest)
    ):
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
