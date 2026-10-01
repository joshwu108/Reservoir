"""Tests for the checker's replay of age-decayed logs.

The checker (``checker/``) imports nothing from ``src/reservoir``. For a
decayed log it must rebuild the decay table from the definition with its
own integer arithmetic, recompute every leaf from the recorded
``(q, entry_version, base_epoch)``, and enforce the lifecycle protocol:
expired entries are evicted right after an ``advance_version``, a
``rebase`` is exact and lands on the canonical base epoch, and sample
records agree with the replayed, rebased tree.

Acceptance tests use real logs from ``RolloutBuffer``. Rejection tests
take a real log, apply one semantic edit, re-chain the digests so only
the semantic check can catch it, and expect ``CheckerError``.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from checker.decay_replay import canonical_base_epoch as checker_canonical_base_epoch
from checker.decay_replay import decay_table as checker_decay_table
from checker.decay_replay import inflated_priority as checker_inflated_priority
from checker.verify import CheckerError, verify_chain, verify_json_lines
from reservoir.attest import AttestationLog, _digest_record
from reservoir.buffer import ExactPERBuffer, Transition
from reservoir.decay import DecayParams, decay_table, inflated_priority
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer


def rollouts(rewards: list[float]) -> list[Rollout]:
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in rewards]


def rechain(records: list[dict], start: int = 0) -> list[dict]:
    """Recompute prev_digest/digest from ``start`` on, as a careful forger would."""
    out = copy.deepcopy(records)
    prev = out[start - 1]["digest"] if start > 0 else "genesis"
    for rec in out[start:]:
        rec["prev_digest"] = prev
        rec["digest"] = _digest_record(rec)
        prev = rec["digest"]
    return out


def to_lines(records: list[dict]) -> str:
    return "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in records)


def build_decayed_log(seed: int = 11) -> tuple[list[dict], RolloutBuffer]:
    """A run with inserts, updates, stale and capacity evictions, and rebases."""
    log = AttestationLog()
    buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=2, seed=seed, attest=log)
    for v in range(10):
        buf.add_group(f"g{v}", v, rollouts([1.0, 0.0, 0.5]))
        batch = buf.sample(4, current_version=v)
        buf.update_priorities(batch.indices[:2], [0.3, 0.6])
    assert buf.n_rebases >= 1
    return [dict(r) for r in log.records], buf


def first_index(records: list[dict], op: str, **match) -> int:
    for i, r in enumerate(records):
        if r["op"] == op and all(r.get(k) == v for k, v in match.items()):
            return i
    raise AssertionError(f"no {op} record matching {match}")


# ---------------------------------------------------------------------------
# Independence: the checker re-derives the decay math on its own
# ---------------------------------------------------------------------------

class TestIndependentDerivation:
    @pytest.mark.parametrize("half_life", [1, 2, 3, 7, 24, 347, 1024])
    def test_decay_table_matches_library(self, half_life: int) -> None:
        assert checker_decay_table(half_life, 31) == decay_table(half_life, 31)

    def test_inflated_priority_matches_library(self) -> None:
        p = DecayParams(half_life=3, max_policy_age=9, capacity=16)
        cfg = dict(half_life=3, priority_bits=32, table_frac_bits=31)
        for q in (0, 1, 5, 65536, (1 << 32) - 1):
            for t in range(0, 12):
                for base in range(0, 2):
                    if t // 3 < base:
                        continue  # entry older than the base epoch: both sides reject it
                    assert checker_inflated_priority(q, t, base, cfg) == inflated_priority(q, t, base, p)

    def test_canonical_base_epoch_matches_library(self) -> None:
        from reservoir.decay import canonical_base_epoch

        p = DecayParams(half_life=4, max_policy_age=10, capacity=8)
        cfg = dict(half_life=4, max_policy_age=10)
        for v in range(0, 60):
            assert checker_canonical_base_epoch(v, cfg) == canonical_base_epoch(v, p)


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------

class TestAccepts:
    def test_real_decayed_log(self) -> None:
        records, buf = build_decayed_log()
        verify_chain(records, buf.capacity)
        verify_chain(records)  # capacity comes from decay_config
        verify_json_lines(to_lines(records))

    def test_capacity_argument_must_agree_with_config(self) -> None:
        records, buf = build_decayed_log()
        with pytest.raises(CheckerError, match="capacity"):
            verify_chain(records, buf.capacity * 2)

    def test_legacy_log_still_verifies(self) -> None:
        buf = ExactPERBuffer(capacity=4, alpha=1.0, beta=1.0, seed=1)
        log = AttestationLog()
        for i in range(4):
            pos = buf.insert(Transition(i, 0, 0.0, i, False), td_error=float(i + 1))
            log.append_mutation("insert", pos, 0, buf._sum_tree.get(pos), buf._op_counter)
        verify_chain(log.records, buf.capacity)

    def test_legacy_log_without_config_needs_capacity(self) -> None:
        log = AttestationLog()
        log.append_mutation("insert", 0, 0, 5, 1)
        with pytest.raises(CheckerError, match="capacity"):
            verify_chain(log.records)

    def test_empty_decayed_buffer_log(self) -> None:
        log = AttestationLog()
        RolloutBuffer(capacity=8, attest=log)
        verify_chain(log.records)


# ---------------------------------------------------------------------------
# Rejections: one semantic edit each, digests re-chained
# ---------------------------------------------------------------------------

class TestRejects:
    def test_decay_config_not_first(self) -> None:
        records, _ = build_decayed_log()
        moved = records[1:2] + records[0:1] + records[2:]
        with pytest.raises(CheckerError, match="decay_config"):
            verify_chain(rechain(moved))

    def test_decay_fields_without_config(self) -> None:
        records, buf = build_decayed_log()
        without = rechain(records[1:])
        with pytest.raises(CheckerError, match="decay_config"):
            verify_chain(without, buf.capacity)

    def test_leaf_not_equal_to_recomputed(self) -> None:
        records, _ = build_decayed_log()
        i = first_index(records, "insert")
        records[i]["new_priority_int"] = str(int(records[i]["new_priority_int"]) + 1)
        with pytest.raises(CheckerError, match="recomputed"):
            verify_chain(rechain(records, i))

    def test_q_changed_but_leaf_kept(self) -> None:
        records, _ = build_decayed_log()
        i = first_index(records, "insert")
        records[i]["base_priority_int"] = str(int(records[i]["base_priority_int"]) + 1)
        with pytest.raises(CheckerError, match="recomputed"):
            verify_chain(rechain(records, i))

    def test_wrong_base_epoch(self) -> None:
        records, _ = build_decayed_log()
        i = first_index(records, "insert")
        records[i]["base_epoch"] = str(int(records[i]["base_epoch"]) + 1)
        with pytest.raises(CheckerError, match="base_epoch"):
            verify_chain(rechain(records, i))

    def test_entry_newer_than_current_version(self) -> None:
        records, _ = build_decayed_log()
        i = first_index(records, "insert")
        records[i]["entry_version"] = str(int(records[i]["entry_version"]) + 1)
        with pytest.raises(CheckerError, match="current version"):
            verify_chain(rechain(records, i))

    def test_insert_of_expired_entry(self) -> None:
        log = AttestationLog()
        buf = RolloutBuffer(capacity=8, max_policy_age=2, attest=log)
        buf.add_group("a", 5, rollouts([1.0, 0.0]))
        records = [dict(r) for r in log.records]
        i = first_index(records, "insert")
        records[i]["entry_version"] = "1"  # age 4 at version 5
        with pytest.raises(CheckerError, match="expired"):
            verify_chain(rechain(records, i))

    def test_stale_evict_of_a_live_entry(self) -> None:
        records, _ = build_decayed_log()
        # Take a capacity eviction and claim it was stale: the entry is not expired.
        i = first_index(records, "evict", reason="capacity")
        records[i]["reason"] = "stale"
        with pytest.raises(CheckerError, match="stale"):
            verify_chain(rechain(records, i))

    def test_expired_entry_not_evicted_after_advance(self) -> None:
        records, _ = build_decayed_log()
        i = first_index(records, "evict", reason="stale")
        del records[i]
        with pytest.raises(CheckerError, match="expired"):
            verify_chain(rechain(records, i))

    def test_rebase_before_stale_evictions(self) -> None:
        records, _ = build_decayed_log()
        r = first_index(records, "rebase")
        assert records[r - 1]["op"] == "evict" and records[r - 1]["reason"] == "stale"
        records[r - 1], records[r] = records[r], records[r - 1]
        with pytest.raises(CheckerError):
            verify_chain(rechain(records, r - 1))

    def test_rebase_to_non_canonical_epoch(self) -> None:
        records, _ = build_decayed_log()
        r = first_index(records, "rebase")
        records[r]["new_base_epoch"] = str(int(records[r]["new_base_epoch"]) + 1)
        with pytest.raises(CheckerError, match="canonical"):
            verify_chain(rechain(records, r))

    def test_rebase_total_mismatch(self) -> None:
        records, _ = build_decayed_log()
        r = first_index(records, "rebase")
        records[r]["root_total_after"] = str(int(records[r]["root_total_after"]) + 1)
        with pytest.raises(CheckerError, match="root_total_after"):
            verify_chain(rechain(records, r))

    def test_rebase_that_would_drop_bits(self) -> None:
        # Forge an advance that is not due, so a leaf with a set low bit gets shifted.
        log = AttestationLog()
        buf = RolloutBuffer(capacity=8, half_life=2, max_policy_age=4, attest=log)
        buf.add_group("a", 0, rollouts([1.0, 0.0]))
        records = [dict(r) for r in log.records]
        n = buf._op_counter
        records.append({"op": "advance_version", "old_version": "0", "new_version": "1", "op_counter": n})
        records.append({
            "op": "rebase", "old_base_epoch": "0", "new_base_epoch": "1",
            "root_total_before": str(buf.total), "root_total_after": str(buf.total >> 1),
            "op_counter": n,
        })
        with pytest.raises(CheckerError):
            verify_chain(rechain(records, len(records) - 2))

    def test_update_of_empty_slot(self) -> None:
        records, _ = build_decayed_log()
        i = first_index(records, "update")
        rec = records[i]
        # Point the update at a slot that holds no entry at that moment: an evicted one.
        evicted_before = [
            r["index"] for r in records[:i] if r["op"] == "evict"
        ]
        live_before = {r["index"] for r in records[:i] if r["op"] == "insert"} - set(evicted_before)
        empty = next(p for p in range(8) if p not in live_before)
        rec["index"] = empty
        rec["old_priority_int"] = "0"
        with pytest.raises(CheckerError, match="live"):
            verify_chain(rechain(records, i))

    def test_update_version_disagrees_with_reset_flag(self) -> None:
        records, _ = build_decayed_log()  # reset_age_on_update=False: version must be unchanged
        i = first_index(records, "update")
        records[i]["entry_version"] = str(int(records[i]["entry_version"]) + 1)
        with pytest.raises(CheckerError):
            verify_chain(rechain(records, i))

    def test_advance_version_backwards(self) -> None:
        records, _ = build_decayed_log()
        advances = [i for i, r in enumerate(records) if r["op"] == "advance_version"]
        i = advances[1]
        records[i]["new_version"] = str(int(records[i]["old_version"]) - 1)
        with pytest.raises(CheckerError, match="advance_version"):
            verify_chain(rechain(records, i))

    def test_advance_version_old_must_match_replay(self) -> None:
        records, _ = build_decayed_log()
        i = first_index(records, "advance_version")
        records[i]["old_version"] = str(int(records[i]["old_version"]) + 1)
        records[i]["new_version"] = str(int(records[i]["new_version"]) + 1)
        with pytest.raises(CheckerError, match="advance_version"):
            verify_chain(rechain(records, i))

    def test_sample_root_total_after_rebase(self) -> None:
        records, _ = build_decayed_log()
        r = first_index(records, "rebase")
        s = next(i for i in range(r, len(records)) if records[i]["op"] == "sample")
        records[s]["root_total"] = str(int(records[s]["root_total"]) << 1)
        with pytest.raises(CheckerError, match="root_total"):
            verify_chain(rechain(records, s))

    def test_unknown_op_still_rejected(self) -> None:
        records, _ = build_decayed_log()
        records[3]["op"] = "mystery"
        with pytest.raises(CheckerError, match="unknown op"):
            verify_chain(rechain(records, 3))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

class TestCommandLine:
    def run(self, path: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "checker.verify", str(path), *args],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent,
        )

    def test_valid_log_exits_zero(self, tmp_path: Path) -> None:
        records, _ = build_decayed_log()
        path = tmp_path / "attest.jsonl"
        path.write_text(to_lines(records) + "\n")
        result = self.run(path)
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_tampered_log_exits_nonzero(self, tmp_path: Path) -> None:
        records, _ = build_decayed_log()
        i = first_index(records, "insert")
        records[i]["new_priority_int"] = "1"
        path = tmp_path / "attest.jsonl"
        path.write_text(to_lines(rechain(records, i)) + "\n")
        result = self.run(path)
        assert result.returncode != 0
        assert "recomputed" in result.stderr

    def test_legacy_log_needs_capacity_flag(self, tmp_path: Path) -> None:
        log = AttestationLog()
        log.append_mutation("insert", 0, 0, 5, 1)
        path = tmp_path / "legacy.jsonl"
        path.write_text(log.to_json_lines() + "\n")
        assert self.run(path).returncode != 0
        assert self.run(path, "--capacity", "4").returncode == 0
