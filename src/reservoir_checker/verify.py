"""
checker.verify — Independent attestation chain verifier.

This module imports nothing from the rest of the reservoir package. It re-derives all
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
8. For a log that starts with a decay_config record: every mutation's leaf
   equals the age-decay formula applied to its recorded (q, version, base
   epoch), expired entries are evicted right after each advance_version,
   and every rebase is exact and lands on the canonical base epoch
   (see checker/decay_replay.py).
9. When decay_config records the buffer's seed, buffer_id, alpha and beta:
   every draw_int equals the keyed BLAKE2b draw for its position in the
   run (checker/draw.py) and every IS weight equals the declared formula
   evaluated on the replayed tree, so a draw moved within its leaf's range
   or a reweighted sample is rejected too.
10. Batch witnesses (``batch`` records): each replaced training-batch row
   names a draw of the latest sample record and carries the content digest
   the draw resolved to; rows and draws are distinct; a sample is
   witnessed at most once (checker/content.py).
11. Telemetry records: the effective sample size and the staleness of a
   replayed step are recomputed from the sample record and the replayed
   state; counters are cross-checked against the draws; carried float
   measurements are listed under "reported" (reservoir_checker/telemetry.py).
12. Content commitments: insert records carry a well-formed content_digest
   (all of them or none) and an optional source; every sampled slot holds
   a committed example; with ``--manifest``, the manifest opens exactly the
   log's commitments (see checker/content.py).

Any inconsistency raises CheckerError. ``verify_chain`` returns a
``VerifiedLog`` whose ``content`` state (insert history, resolved samples)
is what ``checker.transcript`` and ``checker.diff`` build on.

Command line::

    python -m checker.verify run-01/attest.jsonl              # decayed log
    python -m checker.verify run-01/attest.jsonl --manifest run-01/manifest.jsonl
    python -m checker.verify legacy.jsonl --capacity 1024     # log without decay_config
    python -m checker.verify crashed.jsonl --allow-truncated  # prefix cut mid-advance

A log that was being streamed when the process died may stop between an
``advance_version`` and the evictions it requires. Every record present
is still checked; ``--allow-truncated`` only waives the end-of-log rule
that nothing may be left pending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from typing import Optional

from reservoir_checker.content import ContentState, load_manifest, quarantine_fields
from reservoir_checker.decay_replay import CheckerError, DecayState, has_decay_fields, parse_config
from reservoir_checker.draw import draw_uniform_below
from reservoir_checker.telemetry import verify_telemetry

# Domain separator — must match attest.py exactly
_PERSON = b"attest\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
_GENESIS = "genesis"
_HASH_BYTES = 32


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

    def shift_all(self, shift: int, idx: int) -> None:
        """Right-shift every node by ``shift`` bits; a dropped set bit is an error.

        Valid because every live leaf is a multiple of 2^shift after the
        stale evictions, so every internal sum is too.
        """
        mask = (1 << shift) - 1
        for i, node in enumerate(self._tree):
            if node & mask:
                raise CheckerError(
                    f"Record {idx}: rebase by {shift} would drop set bits of node {i} ({node})"
                )
        self._tree = [node >> shift for node in self._tree]

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

@dataclass
class VerifiedLog:
    """What a successful ``verify_chain`` establishes, for tools that build on it."""

    records: list[dict]
    capacity: Optional[int]
    content: ContentState = field(default_factory=ContentState)


def verify_chain(
    records: list[dict],
    capacity: Optional[int] = None,
    allow_truncated: bool = False,
    manifest: Optional[list[dict]] = None,
) -> VerifiedLog:
    """Verify the entire attestation chain from a list of records.

    Parameters
    ----------
    records : list[dict]
        Attestation records in chain order.
    capacity : int, optional
        Buffer capacity for the replay sum-tree. Required for a log without
        a leading ``decay_config`` record; if given for a log that has one,
        it must agree with the recorded capacity.
    allow_truncated : bool
        Accept a log that ends with stale evictions or a rebase still
        pending, as a crashed writer can leave. Every record present is
        still fully checked.
    manifest : list[dict], optional
        Parsed manifest lines (``checker.content.load_manifest``). When
        given, they must open exactly the log's content commitments.

    Raises
    ------
    CheckerError
        If any check fails, including a manifest that does not open the
        log's commitments. The message describes what failed.
    """
    content = ContentState()
    if not records:
        if manifest is not None:
            content.check_manifest(manifest)   # an empty manifest is the only one that opens nothing
        return VerifiedLog([], capacity, content)
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise CheckerError(f"Record {i}: not a JSON object")

    decay = _decay_state(records[0], capacity)
    if decay is not None:
        capacity = decay.cfg["capacity"]
    if capacity is None:
        raise CheckerError("capacity is required: the log has no decay_config record")

    tree = _SumTree(capacity)
    if decay is not None:
        decay.cfg["tree_capacity"] = tree.capacity  # power-of-two slot count
    # priorities[pos] = current integerized priority at position pos
    priorities: dict[int, int] = {}

    prev_digest = _GENESIS
    for record_idx, record in enumerate(records):
        prev_digest = _verify_link(record, record_idx, prev_digest)
        _verify_record(record, record_idx, tree, priorities, decay, content)

    if decay is not None:
        decay.close_capacity_run(len(records))
        if not allow_truncated:
            decay.require_no_pending(len(records), "end of log")
    if manifest is not None:
        content.check_manifest(manifest)
    return VerifiedLog(records, capacity, content)


def _verify_link(record: dict, idx: int, prev_digest: str) -> str:
    """Check the record's digest and its link to the previous record; return its digest."""
    expected_digest = _digest_record(record)
    actual_digest = record.get("digest", "")
    if expected_digest != actual_digest:
        raise CheckerError(
            f"Record {idx}: digest mismatch. "
            f"Expected {expected_digest!r}, got {actual_digest!r}"
        )
    record_prev = record.get("prev_digest", "")
    if record_prev != prev_digest:
        raise CheckerError(
            f"Record {idx}: prev_digest mismatch. "
            f"Expected {prev_digest!r}, got {record_prev!r}"
        )
    return actual_digest


def _verify_record(
    record: dict,
    idx: int,
    tree: _SumTree,
    priorities: dict[int, int],
    decay: Optional[DecayState],
    content: ContentState,
) -> None:
    """Semantic checks for one record (digest and chain linkage already passed)."""
    op = record.get("op")
    if decay is not None:
        decay.note_record(op, record.get("reason"), idx)

    if op == "decay_config":
        if idx != 0:
            raise CheckerError(f"Record {idx}: decay_config must be the first record")

    elif op in ("insert", "update", "evict"):
        _verify_mutation(record, idx, tree, priorities)
        if quarantine_fields(record, idx) is not None:
            _require_format(decay, idx, 3, "quarantine evictions")
        if has_decay_fields(record):
            _require_decay(decay, idx, op)
            _guard_pending(decay, record, idx)
            decay.on_mutation(record, idx, tree)  # type: ignore[union-attr]
        elif decay is not None:
            raise CheckerError(f"Record {idx}: {op} without decay fields in a decayed log")
        elif "reason" in record:
            raise CheckerError(f"Record {idx}: reason requires the decay fields")
        content.on_mutation(record, idx)

    elif op == "sample":
        if decay is not None:
            decay.require_no_pending(idx, "sample")
        _verify_sample(record, idx, tree, priorities)
        if decay is not None:
            _verify_declared_draws(record, idx, tree, decay)
        content.on_sample(idx, record)

    elif op == "batch":
        _require_format(decay, idx, 2, "batch witnesses")
        decay.require_no_pending(idx, op)  # type: ignore[union-attr]
        content.on_batch(idx, record)

    elif op == "telemetry":
        _require_format(decay, idx, 2, "telemetry records")
        decay.require_no_pending(idx, op)  # type: ignore[union-attr]
        versions = {pos: version for pos, (_, version) in decay.entries.items()}  # type: ignore[union-attr]
        content.telemetry.append(verify_telemetry(record, idx, content, versions, decay.current_version))  # type: ignore[union-attr]

    elif op == "advance_version":
        _require_decay(decay, idx, op)
        decay.require_no_pending(idx, op)  # type: ignore[union-attr]
        decay.on_advance(record, idx)  # type: ignore[union-attr]

    elif op == "rebase":
        _require_decay(decay, idx, op)
        decay.require_ready_for_rebase(idx)  # type: ignore[union-attr]
        decay.on_rebase(record, idx, tree)  # type: ignore[union-attr]

    else:
        raise CheckerError(f"Record {idx}: unknown op {op!r}")


def _decay_state(first: dict, capacity: Optional[int]) -> Optional[DecayState]:
    """Build the decay replay state if the log opens with decay_config."""
    if first.get("op") != "decay_config":
        return None
    cfg = parse_config(first, 0)
    if capacity is not None and capacity != cfg["capacity"]:
        raise CheckerError(
            f"capacity argument {capacity} disagrees with decay_config capacity {cfg['capacity']}"
        )
    return DecayState(cfg)


def _require_decay(decay: Optional[DecayState], idx: int, op: str) -> None:
    """Decay-only records and fields are errors in a log that has no decay_config."""
    if decay is None:
        raise CheckerError(
            f"Record {idx}: {op} requires a decay_config record at the start of the log"
        )


def _require_format(decay: Optional[DecayState], idx: int, minimum: int, what: str) -> None:
    """Records introduced by a later log format are forgeries in an older one."""
    _require_decay(decay, idx, what)
    if int(decay.cfg["format"]) < minimum:  # type: ignore[union-attr]
        raise CheckerError(
            f"Record {idx}: {what} need a format-{minimum} log or later (decay_config format \"{minimum}\")"
        )


def _guard_pending(decay: Optional[DecayState], record: dict, idx: int) -> None:
    """While evictions or a rebase are pending, only stale evicts of pending slots may appear."""
    if decay is None or not (decay.pending_stale or decay.pending_rebase):
        return
    is_stale_evict = record["op"] == "evict" and record.get("reason") == "stale"
    if not is_stale_evict or record["index"] not in decay.pending_stale:
        decay.require_no_pending(idx, f"{record['op']} at position {record['index']}")


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
    except (KeyError, ValueError, TypeError) as e:
        raise CheckerError(f"Record {idx}: malformed mutation record: {e}") from e

    # index is a JSON integer. A negative value would address an internal
    # node of the flat tree array, so the range check is a security check.
    if type(pos) is not int or not (0 <= pos < tree.capacity):
        raise CheckerError(
            f"Record {idx}: index must be an int in [0, {tree.capacity}), got {pos!r}"
        )

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


def _verify_declared_draws(record: dict, idx: int, tree: _SumTree, decay: DecayState) -> None:
    """With seed and beta in the config, recompute each draw and importance weight exactly.

    The weight formula is the library's, evaluated in the same order:
    ``w = (n * p / total) ** -beta`` normalised by the weight of the
    smallest positive leaf, with ``n`` the live count, each power taken
    in float64 once and the quotient exact. A verifier whose libm rounds
    ``pow`` differently from the producer's would see a mismatch here;
    docs/nonclaims.md §7 declares that boundary.
    """
    samples = record["samples"]
    if not decay.cfg.get("has_draw_config"):
        decay.draw_counter += len(samples)
        return
    root_total = int(record["root_total"])
    n = len(decay.entries)
    positive = [tree.get(pos) for pos in decay.entries if tree.get(pos) > 0]
    min_leaf = min(positive) if positive else 0
    beta = decay.cfg["beta"]
    for k, s in enumerate(samples):
        counter = decay.draw_counter + k
        expected = draw_uniform_below(root_total, decay.cfg["seed"], decay.cfg["buffer_id"], counter)
        if int(s["draw_int"]) != expected:
            raise CheckerError(
                f"Record {idx}, sample {k}: draw_int {s['draw_int']} is not the keyed draw for "
                f"counter {counter} (expected {expected})"
            )
        if min_leaf > 0:
            leaf = tree.get(s["leaf_index"])
            w = _declared_weight(n, leaf, min_leaf, root_total, beta, idx, k)
            declared = Fraction(int(s["is_weight_num"]), int(s["is_weight_den"]))
            if declared != w:
                raise CheckerError(
                    f"Record {idx}, sample {k}: IS weight {declared} is not the declared formula's value {w}"
                )
    decay.draw_counter += len(samples)


def _declared_weight(n: int, leaf: int, min_leaf: int, root_total: int, beta: float, idx: int, k: int) -> Fraction:
    """The library's IS-weight formula; a config whose beta overflows or underflows it is a forgery."""
    try:
        w_i = (n * leaf / root_total) ** (-beta)
        w_max = (n * min_leaf / root_total) ** (-beta)
        if w_max == 0.0 or w_i != w_i or w_max != w_max or w_i in (float("inf"), float("-inf")):
            raise ArithmeticError("weight not finite or normaliser zero")
        return Fraction(w_i) / Fraction(w_max)
    except (ArithmeticError, ValueError) as exc:
        raise CheckerError(
            f"Record {idx}, sample {k}: the declared beta makes the IS weight unrepresentable ({exc})"
        ) from exc


def verify_json_lines(
    data: str,
    capacity: Optional[int] = None,
    allow_truncated: bool = False,
    manifest: Optional[str] = None,
) -> VerifiedLog:
    """Verify an attestation chain from newline-separated JSON.

    Parameters
    ----------
    data : str
        Newline-separated JSON records (as produced by AttestationLog.to_json_lines).
    capacity, allow_truncated
        See ``verify_chain``.
    manifest : str, optional
        The manifest file's text (JSON lines).

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
    lines = load_manifest(manifest) if manifest is not None else None
    return verify_chain(records, capacity, allow_truncated, manifest=lines)


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point: exit 0 if the log verifies, 1 otherwise."""
    parser = argparse.ArgumentParser(
        description="Independently verify a reservoir attestation log.",
    )
    parser.add_argument("path", help="JSON-lines attestation log")
    parser.add_argument(
        "--capacity", type=int, default=None,
        help="buffer capacity; required for logs without a decay_config record",
    )
    parser.add_argument(
        "--allow-truncated", action="store_true",
        help="accept a log cut off mid-advance (a crashed writer); records present are still checked",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="manifest JSON lines; must open exactly the log's content commitments",
    )
    args = parser.parse_args(argv)
    try:
        with open(args.path, encoding="utf-8") as f:
            data = f.read()
        manifest_text = None
        if args.manifest is not None:
            with open(args.manifest, encoding="utf-8") as f:
                manifest_text = f.read()
        n_records = sum(1 for line in data.strip().split("\n") if line)
        result = verify_json_lines(data, args.capacity, args.allow_truncated, manifest_text)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, ArithmeticError,
            RecursionError, CheckerError) as exc:
        # Everything a malformed file can raise is reported as a failure, never a traceback.
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    summary = f"OK: {n_records} records verified"
    if result.content.has_content:
        summary += f"; {len(result.content.history)} examples committed"
    if args.manifest is not None:
        summary += f"; manifest opens {result.content.manifest_matched} of them"
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
