"""
Tests for reservoir.attest and checker.verify — attestation and independent verification.

TDD: tests define the required behavior of both modules.
The checker tests confirm that checker/verify.py imports nothing from src/reservoir.
"""

import ast
import copy
import json
from fractions import Fraction
from math import gcd
from pathlib import Path

import pytest

from reservoir.attest import (
    AttestationLog,
    _canonical_json,
    _blake2b_digest,
    _digest_record,
    make_sample_entry,
)
from checker.verify import CheckerError, verify_chain, verify_json_lines


# ---------------------------------------------------------------------------
# CI-enforceable import check for checker/verify.py
# ---------------------------------------------------------------------------

class TestCheckerImportIsolation:
    def test_checker_does_not_import_src_reservoir(self):
        """checker/verify.py must not import from src/reservoir."""
        checker_path = Path(__file__).parent.parent / "checker" / "verify.py"
        source = checker_path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "reservoir" not in alias.name, (
                        f"checker/verify.py imports reservoir: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "reservoir" not in module, (
                    f"checker/verify.py imports from reservoir: {module}"
                )


# ---------------------------------------------------------------------------
# AttestationLog tests
# ---------------------------------------------------------------------------

class TestAttestationLog:
    def test_empty_log(self):
        log = AttestationLog()
        assert log.records == []
        assert log.head_digest == "genesis"

    def test_append_mutation_insert(self):
        log = AttestationLog()
        rec = log.append_mutation("insert", 0, 0, 100, 1)
        assert rec["op"] == "insert"
        assert rec["index"] == 0
        assert rec["old_priority_int"] == "0"
        assert rec["new_priority_int"] == "100"
        assert rec["op_counter"] == 1
        assert rec["prev_digest"] == "genesis"
        assert "digest" in rec
        assert len(rec["digest"]) == 64  # 32 bytes hex

    def test_chain_linkage(self):
        log = AttestationLog()
        r1 = log.append_mutation("insert", 0, 0, 100, 1)
        r2 = log.append_mutation("insert", 1, 0, 200, 2)
        assert r2["prev_digest"] == r1["digest"]
        assert log.head_digest == r2["digest"]

    def test_mutation_record_digest_is_deterministic(self):
        log1 = AttestationLog()
        log2 = AttestationLog()
        r1 = log1.append_mutation("insert", 0, 0, 100, 1)
        r2 = log2.append_mutation("insert", 0, 0, 100, 1)
        assert r1["digest"] == r2["digest"]

    def test_mutation_record_changes_when_content_changes(self):
        log1 = AttestationLog()
        log2 = AttestationLog()
        r1 = log1.append_mutation("insert", 0, 0, 100, 1)
        r2 = log2.append_mutation("insert", 0, 0, 101, 1)  # different priority
        assert r1["digest"] != r2["digest"]

    def test_invalid_op_raises(self):
        log = AttestationLog()
        with pytest.raises(ValueError, match="Unknown op"):
            log.append_mutation("delete", 0, 0, 100, 1)

    def test_append_sample(self):
        log = AttestationLog()
        samples = [
            {
                "leaf_index": 2,
                "draw_int": 50,
                "prob_num": 1,
                "prob_den": 10,
                "is_weight_num": 1,
                "is_weight_den": 1,
                "rejection_count": 0,
            }
        ]
        rec = log.append_sample(op_counter=5, root_total=100, samples=samples)
        assert rec["op"] == "sample"
        assert rec["root_total"] == "100"
        assert len(rec["samples"]) == 1
        assert rec["samples"][0]["leaf_index"] == 2
        assert rec["samples"][0]["draw_int"] == "50"

    def test_serialization_round_trip(self):
        log = AttestationLog()
        log.append_mutation("insert", 0, 0, 100, 1)
        log.append_mutation("update", 0, 100, 200, 2)
        original_records = log.records

        json_lines = log.to_json_lines()
        restored = AttestationLog.from_json_lines(json_lines)

        assert len(restored.records) == len(original_records)
        assert restored.head_digest == log.head_digest

    def test_integer_fields_stored_as_strings(self):
        """Large integers in JSON must be stored as strings."""
        log = AttestationLog()
        big_int = 10**50
        rec = log.append_mutation("insert", 0, 0, big_int, 1)
        assert rec["new_priority_int"] == str(big_int)
        # The JSON serialization must not lose precision
        j = json.dumps(rec, sort_keys=True, separators=(",", ":"))
        restored = json.loads(j)
        assert int(restored["new_priority_int"]) == big_int

    def test_records_length(self):
        log = AttestationLog()
        for i in range(5):
            log.append_mutation("insert", i, 0, i + 1, i + 1)
        assert len(log.records) == 5


class TestMakeSampleEntry:
    def test_basic(self):
        w = Fraction(3, 4)
        entry = make_sample_entry(
            leaf_index=2, draw_int=50, priority_int=10, root_total=100, is_weight=w
        )
        assert entry["leaf_index"] == 2
        assert entry["draw_int"] == 50
        # prob = 10/100 = 1/10
        assert entry["prob_num"] == 1
        assert entry["prob_den"] == 10
        assert entry["is_weight_num"] == 3
        assert entry["is_weight_den"] == 4

    def test_prob_in_reduced_form(self):
        entry = make_sample_entry(6, 10, 6, 12, Fraction(1, 1))
        # 6/12 = 1/2
        assert entry["prob_num"] == 1
        assert entry["prob_den"] == 2
        assert gcd(entry["prob_num"], entry["prob_den"]) == 1

    def test_is_weight_in_reduced_form(self):
        entry = make_sample_entry(0, 0, 1, 10, Fraction(6, 4))
        # Fraction(6,4) -> reduced to Fraction(3,2) automatically
        assert entry["is_weight_num"] == 3
        assert entry["is_weight_den"] == 2


# ---------------------------------------------------------------------------
# checker/verify.py tests
# ---------------------------------------------------------------------------

def _build_simple_chain(capacity: int = 4) -> list[dict]:
    """Build a minimal valid attestation chain."""
    log = AttestationLog()
    # Insert 4 priorities: [10, 20, 30, 40]
    priorities = [10, 20, 30, 40]
    for i, p in enumerate(priorities):
        log.append_mutation("insert", i, 0, p, i + 1)
    # Sample: draw_int=15, tree=[10,20,30,40], total=100
    # prefix_sum: [0,10)->0, [10,30)->1, [30,60)->2, [60,100)->3
    # draw_int=15 -> position 1 (priority=20, prob=20/100=1/5)
    entry = make_sample_entry(1, 15, 20, 100, Fraction(1, 1))
    log.append_sample(op_counter=5, root_total=100, samples=[entry])
    return log.records


class TestVerifyChain:
    def test_valid_chain_passes(self):
        records = _build_simple_chain()
        verify_chain(records, capacity=4)  # Should not raise

    def test_empty_chain_passes(self):
        verify_chain([], capacity=4)

    def test_digest_flip_detected(self):
        records = _build_simple_chain()
        mutant = copy.deepcopy(records)
        orig = mutant[0]["digest"]
        mutant[0]["digest"] = "0" * 64  # Wrong digest
        with pytest.raises(CheckerError, match="digest"):
            verify_chain(mutant, capacity=4)

    def test_wrong_prev_digest_detected(self):
        records = _build_simple_chain()
        mutant = copy.deepcopy(records)
        mutant[1]["prev_digest"] = "0" * 64
        # Now record 1's digest is wrong too (computed before mutation)
        # But we need to re-digest record 1 for it to have a valid digest
        from reservoir.attest import _digest_record
        mutant[1]["digest"] = _digest_record(mutant[1])
        with pytest.raises(CheckerError):
            verify_chain(mutant, capacity=4)

    def test_wrong_old_priority_detected(self):
        """If old_priority_int doesn't match tree state, checker rejects."""
        records = _build_simple_chain()
        mutant = copy.deepcopy(records)
        # Change old_priority_int for record 1 (insert at pos 1)
        # Record 0 sets pos 0 to 10; record 1 should have old=0 for pos 1
        mutant[1]["old_priority_int"] = "99"  # Wrong
        # Recompute digest for this record to make it "self-consistent"
        from reservoir.attest import _digest_record
        mutant[1]["digest"] = _digest_record(mutant[1])
        # Recompute downstream
        for i in range(2, len(mutant)):
            mutant[i]["prev_digest"] = mutant[i - 1]["digest"]
            mutant[i]["digest"] = _digest_record(mutant[i])
        with pytest.raises(CheckerError):
            verify_chain(mutant, capacity=4)

    def test_wrong_draw_int_detected(self):
        """Off-by-one draw_int detected."""
        records = _build_simple_chain()
        mutant = copy.deepcopy(records)
        sample_idx = next(i for i, r in enumerate(mutant) if r.get("op") == "sample")
        # Increase draw_int by 1 (15 -> 16, which also maps to pos 1 but is wrong IF
        # the checker re-verifies it. Let's use 9 which maps to pos 0, not 1)
        mutant[sample_idx]["samples"][0]["draw_int"] = "9"
        from reservoir.attest import _digest_record
        mutant[sample_idx]["digest"] = _digest_record(mutant[sample_idx])
        for i in range(sample_idx + 1, len(mutant)):
            mutant[i]["prev_digest"] = mutant[i - 1]["digest"]
            mutant[i]["digest"] = _digest_record(mutant[i])
        with pytest.raises(CheckerError, match="maps to"):
            verify_chain(mutant, capacity=4)

    def test_wrong_root_total_detected(self):
        records = _build_simple_chain()
        mutant = copy.deepcopy(records)
        sample_idx = next(i for i, r in enumerate(mutant) if r.get("op") == "sample")
        mutant[sample_idx]["root_total"] = "999"  # Wrong
        from reservoir.attest import _digest_record
        mutant[sample_idx]["digest"] = _digest_record(mutant[sample_idx])
        for i in range(sample_idx + 1, len(mutant)):
            mutant[i]["prev_digest"] = mutant[i - 1]["digest"]
            mutant[i]["digest"] = _digest_record(mutant[i])
        with pytest.raises(CheckerError, match="root_total"):
            verify_chain(mutant, capacity=4)

    def test_prob_not_reduced_detected(self):
        """Probability not in reduced form is detected."""
        records = _build_simple_chain()
        mutant = copy.deepcopy(records)
        sample_idx = next(i for i, r in enumerate(mutant) if r.get("op") == "sample")
        # 1/5 -> 2/10 (same value, not reduced)
        mutant[sample_idx]["samples"][0]["prob_num"] = "2"
        mutant[sample_idx]["samples"][0]["prob_den"] = "10"
        from reservoir.attest import _digest_record
        mutant[sample_idx]["digest"] = _digest_record(mutant[sample_idx])
        for i in range(sample_idx + 1, len(mutant)):
            mutant[i]["prev_digest"] = mutant[i - 1]["digest"]
            mutant[i]["digest"] = _digest_record(mutant[i])
        with pytest.raises(CheckerError, match="reduced"):
            verify_chain(mutant, capacity=4)

    def test_is_weight_not_reduced_detected(self):
        """IS weight not in reduced form is detected."""
        records = _build_simple_chain()
        mutant = copy.deepcopy(records)
        sample_idx = next(i for i, r in enumerate(mutant) if r.get("op") == "sample")
        # IS weight 1/1 -> 2/2
        mutant[sample_idx]["samples"][0]["is_weight_num"] = "2"
        mutant[sample_idx]["samples"][0]["is_weight_den"] = "2"
        from reservoir.attest import _digest_record
        mutant[sample_idx]["digest"] = _digest_record(mutant[sample_idx])
        for i in range(sample_idx + 1, len(mutant)):
            mutant[i]["prev_digest"] = mutant[i - 1]["digest"]
            mutant[i]["digest"] = _digest_record(mutant[i])
        with pytest.raises(CheckerError, match="reduced"):
            verify_chain(mutant, capacity=4)

    def test_deleted_record_detected(self):
        """Removing a mutation record breaks the chain."""
        records = _build_simple_chain()
        mutant = copy.deepcopy(records)
        # Delete record 1 (insert at pos 1)
        del mutant[1]
        # Recompute chain
        from reservoir.attest import _digest_record
        for i in range(1, len(mutant)):
            mutant[i]["prev_digest"] = mutant[i - 1]["digest"]
            mutant[i]["digest"] = _digest_record(mutant[i])
        # Now the sample sees wrong tree state (pos 1 has 0 instead of 20)
        with pytest.raises(CheckerError):
            verify_chain(mutant, capacity=4)

    def test_reordered_records_detected(self):
        """Swapping an insert and a subsequent update of the SAME position is detected.

        The update record has old_priority_int = value_after_insert.
        If we swap them (update first, then insert), the checker sees:
          - update at pos 0: old=0 (correct, tree starts at 0), new=20
          - insert at pos 0: old=10, but tree actually has 20 -> CheckerError
        """
        from reservoir.attest import _digest_record
        log = AttestationLog()
        # insert at pos 0: 0 -> 10
        log.append_mutation("insert", 0, 0, 10, 1)
        # update at pos 0: 10 -> 20 (depends on previous insert)
        log.append_mutation("update", 0, 10, 20, 2)
        records = log.records

        mutant = copy.deepcopy(records)
        # Swap records 0 and 1: update now comes before insert
        mutant[0], mutant[1] = mutant[1], mutant[0]
        # Recompute chain digests from scratch
        for i in range(len(mutant)):
            prev = mutant[i - 1]["digest"] if i > 0 else "genesis"
            mutant[i]["prev_digest"] = prev
            mutant[i]["digest"] = _digest_record(mutant[i])
        # Now: record 0 is "update at pos 0, old=10, new=20"
        # But tree starts with 0 at pos 0, not 10 -> should fail
        with pytest.raises(CheckerError):
            verify_chain(mutant, capacity=4)


class TestVerifyJsonLines:
    def test_round_trip_valid(self):
        log = AttestationLog()
        log.append_mutation("insert", 0, 0, 10, 1)
        log.append_mutation("insert", 1, 0, 20, 2)
        entry = make_sample_entry(1, 15, 20, 30, Fraction(1, 1))
        log.append_sample(2, 30, [entry])

        json_lines = log.to_json_lines()
        verify_json_lines(json_lines, capacity=2)  # Should not raise

    def test_corrupted_json_lines_rejected(self):
        log = AttestationLog()
        log.append_mutation("insert", 0, 0, 10, 1)
        lines = log.to_json_lines().split("\n")
        # Corrupt the digest
        record = json.loads(lines[0])
        record["digest"] = "bad" * 21 + "bad"
        lines[0] = json.dumps(record)
        with pytest.raises(CheckerError):
            verify_json_lines("\n".join(lines), capacity=4)
