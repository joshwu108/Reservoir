"""
reservoir.attest — Hash-chained attestation for the exact PER buffer.

Every buffer mutation appends a MutationRecord.
Every batch sample appends a SampleAttestation.
Both are chained via BLAKE2b digests into a single log.

Schema (canonical JSON — sorted keys, no spaces):
  MutationRecord:
    digest, op, index, old_priority_int, new_priority_int, op_counter, prev_digest

  SampleAttestation:
    digest, op_counter, root_total, samples[...], prev_digest

Integers are encoded as strings to avoid JSON precision limits.
Digests are hex-encoded BLAKE2b-256 of canonical UTF-8 encoding.

See docs/design.md §4 for the full schema.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Optional

# Domain separator for attestation hashing
_PERSON = b"attest\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"  # 16 bytes
_GENESIS = "genesis"


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

        Returns
        -------
        dict
            The complete MutationRecord with digest.
        """
        if op not in ("insert", "update", "evict"):
            raise ValueError(f"Unknown op: {op!r}")

        record = {
            "op": op,
            "index": index,
            "old_priority_int": str(old_priority_int),
            "new_priority_int": str(new_priority_int),
            "op_counter": op_counter,
            "prev_digest": self._head_digest,
        }
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
