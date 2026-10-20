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
12. Entry ages: a sample that declares ``min_entry_age`` draws only slots
   at least that old, and every draw's ``entry_version`` is the slot's
   replayed version (reservoir_checker/entry_age.py).
13. Content commitments: insert records carry a well-formed content_digest
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

from reservoir_checker.content import MAX_SAMPLE_DRAWS, ContentState, load_manifest, quarantine_fields
from reservoir_checker.decay_replay import CheckerError, DecayState, has_decay_fields, parse_config
from reservoir_checker.draw import draw_uniform_below
from reservoir_checker.entry_age import has_entry_age_fields, verify_entry_ages
from reservoir_checker.record_keys import DRAW_KEYS, check_keys, check_record_keys
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
    sample_versions: list[tuple[int, int]] = []   # (op_counter, buffer version) of each sample, format 4
    for record_idx, record in enumerate(records):
        prev_digest = _verify_link(record, record_idx, prev_digest)
        try:
            _verify_record(record, record_idx, tree, priorities, decay, content)
        except CheckerError:
            raise
        except Exception as exc:  # a malformed field the per-record checks did not anticipate
            raise CheckerError(f"Record {record_idx}: malformed record ({type(exc).__name__}: {exc})") from exc
        if decay is not None and decay.cfg["format"] == "4" and record.get("op") == "sample":
            sample_versions.append((content.last_sample_op, decay.current_version))

    if decay is not None:
        decay.close_capacity_run(len(records))
        if not allow_truncated:
            decay.require_no_pending(len(records), "end of log")
        _require_format4_witnesses(sample_versions, content, allow_truncated)
    if manifest is not None:
        content.check_manifest(manifest)
    return VerifiedLog(records, capacity, content)


def _require_format4_witnesses(sample_versions: list[tuple[int, int]], content: ContentState,
                               allow_truncated: bool) -> None:
    """Every format-4 sample has a batch witness and telemetry, unless it was abandoned.

    A sample is abandoned when the writer died before witnessing it and the
    resumed trainer sampled the same step again: the next sample record is
    at the same buffer version or earlier. The last sample may lack them
    only in a log read with ``allow_truncated``.
    """
    for i, (op, version) in enumerate(sample_versions):
        if op in content.witness_by_op and op in content.telemetered_samples:
            continue
        if i + 1 < len(sample_versions):
            if sample_versions[i + 1][1] <= version:
                continue
        elif allow_truncated:
            continue
        missing = "a batch witness" if op not in content.witness_by_op else "telemetry"
        raise CheckerError(f"format-4 sample {op} at version {version} requires {missing}")


def _strict_int_value(obj: object, name: str, idx: int) -> int:
    """A JSON integer (not bool, not float) or a decimal digit string, else CheckerError.

    ``int()`` alone would accept ``True``, ``1.5`` and ``" 1_0 "``; two
    verifiers must not disagree about what a field says.
    """
    if not isinstance(obj, dict):
        raise CheckerError(f"Record {idx}: expected an object with {name!r}, got {type(obj).__name__}")
    value = obj.get(name)
    if isinstance(value, bool) or isinstance(value, float) or value is None:
        raise CheckerError(f"Record {idx}: {name} must be an integer, got {value!r}")
    if isinstance(value, str) and not (value.isascii() and value.isdigit()):
        raise CheckerError(f"Record {idx}: {name} must be an integer, got {value!r}")
    if not isinstance(value, (int, str)):
        raise CheckerError(f"Record {idx}: {name} must be an integer, got {value!r}")
    try:
        return int(value)
    except (ValueError, OverflowError) as exc:
        raise CheckerError(f"Record {idx}: {name} must be an integer, got {value!r}") from exc


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
    check_record_keys(record, idx)
    if decay is not None:
        decay.note_record(op, record.get("reason"), idx)

    if op == "decay_config":
        if idx != 0:
            raise CheckerError(f"Record {idx}: decay_config must be the first record")

    elif op in ("insert", "update", "evict"):
        _verify_mutation_record(record, idx, tree, priorities, decay, content)

    elif op == "sample":
        _verify_sample_record(record, idx, tree, priorities, decay, content)

    elif op == "batch":
        _require_format(decay, idx, 2, "batch witnesses")
        decay.require_no_pending(idx, op)  # type: ignore[union-attr]
        content.on_batch(idx, record)
        _require_step_is_version(content.witnesses[-1]["step"], decay, idx, "batch witness")

    elif op == "telemetry":
        _require_format(decay, idx, 2, "telemetry records")
        decay.require_no_pending(idx, op)  # type: ignore[union-attr]
        point = verify_telemetry(record, idx, content, decay.entries, decay.current_version)  # type: ignore[union-attr]
        _require_step_is_version(point.step, decay, idx, "telemetry")
        content.telemetry.append(point)

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


def _verify_mutation_record(record: dict, idx: int, tree: _SumTree, priorities: dict[int, int],
                            decay: Optional[DecayState], content: ContentState) -> None:
    """An insert, update or evict: tree effect, decay replay, quarantine texts, content commitments."""
    op = record["op"]
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


def _verify_sample_record(record: dict, idx: int, tree: _SumTree, priorities: dict[int, int],
                          decay: Optional[DecayState], content: ContentState) -> None:
    """A sample: draws against the replayed tree, keyed draws and weights, entry ages, resolved content."""
    if decay is not None:
        decay.require_no_pending(idx, "sample")
    _verify_sample(record, idx, tree, priorities)
    if decay is not None:
        _verify_declared_draws(record, idx, tree, decay)
        if decay.cfg["format"] == "4" and record.get("min_entry_age") != str(decay.cfg["min_entry_age"]):
            raise CheckerError(f"Record {idx}: min_entry_age differs from the format-4 decay_config")
        verify_entry_ages(record, idx, decay)
    elif has_entry_age_fields(record):
        raise CheckerError(f"Record {idx}: min_entry_age and entry_version need a log with decay fields")
    content.on_sample(idx, record)


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


def _require_step_is_version(step: int, decay: Optional[DecayState], idx: int, what: str) -> None:
    """A witness or telemetry ``step`` is the model version the buffer was at, not a free label."""
    if step != decay.current_version:  # type: ignore[union-attr]
        raise CheckerError(
            f"Record {idx}: {what} step {step} is not the buffer version {decay.current_version}"  # type: ignore[union-attr]
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
        old_p = _strict_int_value(record, "old_priority_int", idx)
        new_p = _strict_int_value(record, "new_priority_int", idx)
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
        declared_root_total = _strict_int_value(record, "root_total", idx)
        samples = record["samples"]
    except (KeyError, ValueError) as e:
        raise CheckerError(f"Record {idx}: malformed sample record: {e}") from e
    if not isinstance(samples, list):
        raise CheckerError(f"Record {idx}: samples must be a list")
    if not samples:
        raise CheckerError(f"Record {idx}: the sample record draws nothing; the buffer never writes an empty sample")
    if len(samples) > MAX_SAMPLE_DRAWS:
        raise CheckerError(f"Record {idx}: {len(samples)} draws exceeds the checker's limit of {MAX_SAMPLE_DRAWS}")

    # 1. root_total must match replayed tree
    actual_total = tree.total
    if declared_root_total != actual_total:
        raise CheckerError(
            f"Record {idx}: root_total={declared_root_total} "
            f"but replayed tree total={actual_total}"
        )

    for k, s in enumerate(samples):
        _verify_draw(s, k, idx, tree, declared_root_total)


def _verify_draw(s: dict, k: int, idx: int, tree: _SumTree, declared_root_total: int) -> None:
    """One draw of a sample record: in range, located at its leaf, exact reduced probability and weight."""
    at = f"Record {idx}, sample {k}"
    check_keys(s, DRAW_KEYS, at)
    try:
        leaf_index = s["leaf_index"]
        draw_int = _strict_int_value(s, "draw_int", idx)
        prob_num = _strict_int_value(s, "prob_num", idx)
        prob_den = _strict_int_value(s, "prob_den", idx)
        is_w_num = _strict_int_value(s, "is_weight_num", idx)
        is_w_den = _strict_int_value(s, "is_weight_den", idx)
    except (KeyError, ValueError) as e:
        raise CheckerError(f"{at}: malformed entry: {e}") from e

    # draw_int must be in [0, root_total) and locate the declared leaf
    if not (0 <= draw_int < declared_root_total):
        raise CheckerError(f"{at}: draw_int={draw_int} not in [0, {declared_root_total})")
    located = tree.prefix_sum_locate(draw_int)
    if located != leaf_index:
        raise CheckerError(f"{at}: draw_int={draw_int} maps to position {located}, but declared leaf_index={leaf_index}")

    # declared probability = priority_int / root_total, in reduced form
    priority_int = tree.get(leaf_index)
    actual_prob = Fraction(priority_int, declared_root_total)
    if Fraction(prob_num, prob_den) != actual_prob:
        raise CheckerError(f"{at}: declared prob={prob_num}/{prob_den}, but actual prob={priority_int}/"
                           f"{declared_root_total} (reduced: {actual_prob})")
    if gcd(abs(prob_num), abs(prob_den)) != 1:
        raise CheckerError(f"{at}: prob {prob_num}/{prob_den} is not in reduced form")

    # the IS weight is a reduced fraction in (0, 1]
    if is_w_den == 0:
        raise CheckerError(f"{at}: is_weight denominator is 0")
    if gcd(abs(is_w_num), abs(is_w_den)) != 1:
        raise CheckerError(f"{at}: IS weight {is_w_num}/{is_w_den} is not in reduced form")
    is_w = Fraction(is_w_num, is_w_den)
    if not (Fraction(0) < is_w <= Fraction(1)):
        raise CheckerError(f"{at}: IS weight {is_w} not in (0, 1]")


def _verify_declared_draws(record: dict, idx: int, tree: _SumTree, decay: DecayState) -> None:
    """With seed and beta in the config, recompute each draw and importance weight exactly.

    The weight formula is the library's, evaluated in the same order:
    ``w = (n * p / total) ** -beta`` normalised by the weight of the
    smallest positive leaf, with ``n`` the live count, each power taken
    in float64 once and the quotient exact. The producer's C library may
    round each ``pow`` one ulp away from this one's (a glibc log checked on
    macOS), so each power is accepted within one ulp of the checker's;
    docs/nonclaims.md §7 declares that boundary.
    """
    samples = record["samples"]
    if not decay.cfg.get("has_draw_config"):
        decay.draw_counter += len(samples)
        return
    root_total = _strict_int_value(record, "root_total", idx)
    n = len(decay.entries)
    positive = [tree.get(pos) for pos in decay.entries if tree.get(pos) > 0]
    min_leaf = min(positive) if positive else 0
    beta = decay.cfg["beta"]
    for k, s in enumerate(samples):
        counter = decay.draw_counter + k
        expected = draw_uniform_below(root_total, decay.cfg["seed"], decay.cfg["buffer_id"], counter)
        if _strict_int_value(s, "draw_int", idx) != expected:
            raise CheckerError(
                f"Record {idx}, sample {k}: draw_int {s['draw_int']} is not the keyed draw for "
                f"counter {counter} (expected {expected})"
            )
        if min_leaf > 0:
            leaf = tree.get(s["leaf_index"])
            declared = Fraction(_strict_int_value(s, "is_weight_num", idx), _strict_int_value(s, "is_weight_den", idx))
            if not _declared_weight_matches(declared, n, leaf, min_leaf, root_total, beta, idx, k):
                w = _declared_weight(n, leaf, min_leaf, root_total, beta, idx, k)
                raise CheckerError(
                    f"Record {idx}, sample {k}: IS weight {declared} is not the declared formula's value {w}"
                    " (within one ulp of each power)"
                )
    decay.draw_counter += len(samples)


def _declared_powers(n: int, leaf: int, min_leaf: int, root_total: int, beta: float, idx: int,
                     k: int) -> tuple[float, float]:
    """``(w_i, w_max)`` of the library's IS-weight formula; a config whose beta overflows or underflows it is a forgery."""
    try:
        w_i = (n * leaf / root_total) ** (-beta)
        w_max = (n * min_leaf / root_total) ** (-beta)
        if w_max == 0.0 or w_i != w_i or w_max != w_max or w_i in (float("inf"), float("-inf")):
            raise ArithmeticError("weight not finite or normaliser zero")
        return w_i, w_max
    except (ArithmeticError, ValueError) as exc:
        raise CheckerError(
            f"Record {idx}, sample {k}: the declared beta makes the IS weight unrepresentable ({exc})"
        ) from exc


def _declared_weight(n: int, leaf: int, min_leaf: int, root_total: int, beta: float, idx: int, k: int) -> Fraction:
    """The library's IS-weight formula with this platform's ``pow``."""
    w_i, w_max = _declared_powers(n, leaf, min_leaf, root_total, beta, idx, k)
    return Fraction(w_i) / Fraction(w_max)


def _one_ulp(x: float) -> tuple[float, float, float]:
    return x, math.nextafter(x, math.inf), math.nextafter(x, 0.0)


def _declared_weight_matches(declared: Fraction, n: int, leaf: int, min_leaf: int, root_total: int, beta: float,
                             idx: int, k: int) -> bool:
    """``declared`` is the formula's value with each ``pow`` within one ulp of this platform's (nonclaims §7)."""
    w_i, w_max = _declared_powers(n, leaf, min_leaf, root_total, beta, idx, k)
    return any(declared == Fraction(a) / Fraction(b)
               for a in _one_ulp(w_i) for b in _one_ulp(w_max) if b > 0.0 and math.isfinite(a))


def parse_json_lines(data: str) -> list:
    """One JSON value per non-empty line; a line the parser refuses is a CheckerError naming it (from 1)."""
    records = []
    for lineno, line in enumerate(data.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except (ValueError, RecursionError) as exc:   # includes integer literals past the digit limit
            raise CheckerError(f"log line {lineno}: not valid JSON: {exc}") from exc
    return records


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
    records = parse_json_lines(data)
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
            RecursionError, MemoryError, CheckerError) as exc:
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
