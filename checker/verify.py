"""
checker.verify — Independent attestation chain verifier.

This module imports NOTHING from src/reservoir. It re-derives all
computation from first principles using only Python stdlib.

What it verifies:
1. Each record's digest matches its content (excluding the digest field).
2. Each record's prev_digest matches the previous record's digest.
3. Mutation records consistently reconstruct the sum-tree state.
4. Sample records' draw integers are consistent with the declared probabilities:
   - draw_int in [sum(p_0..p_{i-1}), sum(p_0..p_i)) corresponds to leaf_index i.
5. IS weight numerators/denominators are in reduced form (GCD == 1).
6. Probability num/den pairs are in reduced form.
7. Sample root_total matches the reconstructed tree's total at that point.

Any inconsistency raises CheckerError.
"""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from math import gcd


# Domain separator — must match attest.py exactly
_PERSON = b"attest\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
_GENESIS = "genesis"
_HASH_BYTES = 32


class CheckerError(Exception):
    """Raised when any attestation check fails."""


def _canonical_json(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _blake2b_digest(data: bytes) -> str:
    h = hashlib.blake2b(data, digest_size=_HASH_BYTES, person=_PERSON)
    return h.hexdigest()


def _digest_record(record: dict, exclude_key: str = "digest") -> str:
    record_without = {k: v for k, v in record.items() if k != exclude_key}
    return _blake2b_digest(_canonical_json(record_without))


# ---------------------------------------------------------------------------
# Pure-integer sum-tree (re-implemented, does not import from src/)
# ---------------------------------------------------------------------------

def _next_power_of_two(n: int) -> int:
    if n <= 0:
        raise ValueError(f"Capacity must be positive, got {n}")
    if n == 1:
        return 1
    return 1 << (n - 1).bit_length()


class _SumTree:
    """Minimal exact sum-tree for replay during verification."""

    def __init__(self, capacity: int) -> None:
        self.capacity = _next_power_of_two(capacity)
        self._tree = [0] * (2 * self.capacity)

    @property
    def total(self) -> int:
        return self._tree[0]

    def _leaf_idx(self, pos: int) -> int:
        return self.capacity - 1 + pos

    def _propagate(self, idx: int) -> None:
        p = (idx - 1) >> 1
        while idx > 0:
            l, r = 2 * p + 1, 2 * p + 2
            self._tree[p] = self._tree[l] + self._tree[r]
            idx, p = p, (p - 1) >> 1

    def update(self, pos: int, val: int) -> int:
        idx = self._leaf_idx(pos)
        old = self._tree[idx]
        self._tree[idx] = val
        self._propagate(idx)
        return old

    def get(self, pos: int) -> int:
        return self._tree[self._leaf_idx(pos)]

    def prefix_sum_locate(self, target: int) -> int:
        if self.total == 0 or target < 0 or target >= self.total:
            raise CheckerError(
                f"prefix_sum_locate: target {target} invalid (total={self.total})"
            )
        idx = 0
        while idx < self.capacity - 1:
            l = 2 * idx + 1
            if target < self._tree[l]:
                idx = l
            else:
                target -= self._tree[l]
                idx = 2 * idx + 2
        return idx - (self.capacity - 1)


# ---------------------------------------------------------------------------
# Verification entry point
# ---------------------------------------------------------------------------

def verify_chain(records: list[dict], capacity: int) -> None:
    """Verify the entire attestation chain from a list of records.

    Parameters
    ----------
    records : list[dict]
        Attestation records in chain order (MutationRecord or SampleAttestation).
    capacity : int
        Buffer capacity (used to initialize the replay sum-tree).

    Raises
    ------
    CheckerError
        If any check fails. The message describes what failed.
    """
    if not records:
        return  # Empty chain is valid

    tree = _SumTree(capacity)
    # priorities[pos] = current integerized priority at position pos
    priorities: dict[int, int] = {}

    prev_digest = _GENESIS

    for record_idx, record in enumerate(records):
        # 1. Check digest integrity
        expected_digest = _digest_record(record)
        actual_digest = record.get("digest", "")
        if expected_digest != actual_digest:
            raise CheckerError(
                f"Record {record_idx}: digest mismatch. "
                f"Expected {expected_digest!r}, got {actual_digest!r}"
            )

        # 2. Check chain linkage
        record_prev = record.get("prev_digest", "")
        if record_prev != prev_digest:
            raise CheckerError(
                f"Record {record_idx}: prev_digest mismatch. "
                f"Expected {prev_digest!r}, got {record_prev!r}"
            )

        op = record.get("op")

        if op in ("insert", "update", "evict"):
            _verify_mutation(record, record_idx, tree, priorities)

        elif op == "sample":
            _verify_sample(record, record_idx, tree, priorities)

        else:
            raise CheckerError(f"Record {record_idx}: unknown op {op!r}")

        prev_digest = actual_digest


def _verify_mutation(
    record: dict,
    idx: int,
    tree: _SumTree,
    priorities: dict[int, int],
) -> None:
    """Verify a MutationRecord and update the replayed tree state."""
    try:
        pos = record["index"]
        old_p = int(record["old_priority_int"])
        new_p = int(record["new_priority_int"])
        op = record["op"]
    except (KeyError, ValueError) as e:
        raise CheckerError(f"Record {idx}: malformed mutation record: {e}") from e

    if old_p < 0 or new_p < 0:
        raise CheckerError(
            f"Record {idx}: negative priority: old={old_p}, new={new_p}"
        )

    # The replayed tree's current value at pos should match old_priority_int
    current_in_tree = tree.get(pos)
    if current_in_tree != old_p:
        raise CheckerError(
            f"Record {idx} ({op} at pos={pos}): "
            f"old_priority_int={old_p} but tree has {current_in_tree}"
        )

    # Apply the update
    tree.update(pos, new_p)
    priorities[pos] = new_p


def _verify_sample(
    record: dict,
    idx: int,
    tree: _SumTree,
    priorities: dict[int, int],
) -> None:
    """Verify a SampleAttestation against the replayed tree state."""
    try:
        declared_root_total = int(record["root_total"])
        samples = record["samples"]
    except (KeyError, ValueError) as e:
        raise CheckerError(f"Record {idx}: malformed sample record: {e}") from e

    # 1. root_total must match replayed tree
    actual_total = tree.total
    if declared_root_total != actual_total:
        raise CheckerError(
            f"Record {idx}: root_total={declared_root_total} "
            f"but replayed tree total={actual_total}"
        )

    for k, s in enumerate(samples):
        try:
            leaf_index = s["leaf_index"]
            draw_int = int(s["draw_int"])
            prob_num = int(s["prob_num"])
            prob_den = int(s["prob_den"])
            is_w_num = int(s["is_weight_num"])
            is_w_den = int(s["is_weight_den"])
        except (KeyError, ValueError) as e:
            raise CheckerError(
                f"Record {idx}, sample {k}: malformed entry: {e}"
            ) from e

        # 2. draw_int must be in [0, root_total)
        if not (0 <= draw_int < declared_root_total):
            raise CheckerError(
                f"Record {idx}, sample {k}: draw_int={draw_int} "
                f"not in [0, {declared_root_total})"
            )

        # 3. prefix_sum_locate(draw_int) must equal leaf_index
        located = tree.prefix_sum_locate(draw_int)
        if located != leaf_index:
            raise CheckerError(
                f"Record {idx}, sample {k}: draw_int={draw_int} maps to "
                f"position {located}, but declared leaf_index={leaf_index}"
            )

        # 4. declared probability = priority_int / root_total
        priority_int = tree.get(leaf_index)
        declared_prob = Fraction(prob_num, prob_den)
        actual_prob = Fraction(priority_int, declared_root_total)
        if declared_prob != actual_prob:
            raise CheckerError(
                f"Record {idx}, sample {k}: "
                f"declared prob={prob_num}/{prob_den}, "
                f"but actual prob={priority_int}/{declared_root_total} "
                f"(reduced: {actual_prob})"
            )

        # 5. prob_num/prob_den must be in reduced form
        if gcd(abs(prob_num), abs(prob_den)) != 1:
            raise CheckerError(
                f"Record {idx}, sample {k}: prob {prob_num}/{prob_den} "
                f"is not in reduced form"
            )

        # 6. is_weight_num/is_weight_den must be in reduced form
        if is_w_den == 0:
            raise CheckerError(f"Record {idx}, sample {k}: is_weight denominator is 0")
        if gcd(abs(is_w_num), abs(is_w_den)) != 1:
            raise CheckerError(
                f"Record {idx}, sample {k}: IS weight {is_w_num}/{is_w_den} "
                f"is not in reduced form"
            )

        # 7. IS weight must be in (0, 1] — negative or >1 weights are invalid
        is_w = Fraction(is_w_num, is_w_den)
        if not (Fraction(0) < is_w <= Fraction(1)):
            raise CheckerError(
                f"Record {idx}, sample {k}: IS weight {is_w} not in (0, 1]"
            )


def verify_json_lines(data: str, capacity: int) -> None:
    """Verify an attestation chain from newline-separated JSON.

    Parameters
    ----------
    data : str
        Newline-separated JSON records (as produced by AttestationLog.to_json_lines).
    capacity : int
        Buffer capacity for tree replay.

    Raises
    ------
    CheckerError
        If any check fails.
    """
    records = []
    for line in data.strip().split("\n"):
        if not line:
            continue
        records.append(json.loads(line))
    verify_chain(records, capacity)
