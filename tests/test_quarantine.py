"""Tests for ``RolloutBuffer.quarantine`` and its durable form.

Quarantine is incident response: an operator names a predicate over the
stored examples and a reason, and every matching entry is evicted with a
record that says so. The record carries the predicate text and the note,
the predicate runs over every live entry before anything is mutated, and
the durable buffer logs the evicted positions and both texts (never the
callable) so recovery replays the same records.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reservoir.attest import AttestationLog
from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer
from reservoir.rollout_quarantine import MAX_NOTE_LENGTH, MAX_PREDICATE_LENGTH, describe_predicate
from reservoir.rollout_wal import apply_command
from reservoir_checker.verify import verify_chain

KW = dict(capacity=8, half_life=1, max_policy_age=4, seed=3)


def rollouts(values, **meta):
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r, metadata=meta or None) for r in values]


def filled(**kw) -> RolloutBuffer:
    buf = RolloutBuffer(attest=AttestationLog(), **dict(KW, **kw))
    buf.add_group("clean", 0, rollouts([1.0, 0.0]), source="a")
    buf.add_group("hacked", 1, rollouts([1.0, 1.0, 0.5]), source="judge-v3")
    buf.add_group("other", 1, rollouts([0.25]), source="a")
    return buf


def quarantine_records(buf) -> list[dict]:
    return [r for r in buf.attestation_log.records if r["op"] == "evict" and r.get("reason") == "quarantine"]


class TestBufferQuarantine:
    def test_evicts_every_match_and_returns_their_positions(self) -> None:
        buf = filled()
        before = buf.size
        positions = buf.quarantine(lambda r, g: g.source == "judge-v3", "reward hack in judge v3")
        assert len(positions) == 3
        assert buf.size == before - 3
        assert all(p not in buf.live_positions() for p in positions)
        assert positions == tuple(sorted(positions))
        assert {buf.entry(p)[1].prompt_id for p in buf.live_positions()} == {"clean", "other"}

    def test_records_carry_reason_predicate_and_note(self) -> None:
        buf = filled()
        buf.quarantine(lambda r, g: g.source == "judge-v3", "reward hack in judge v3",
                       predicate_text="group.source == 'judge-v3'")
        records = quarantine_records(buf)
        assert len(records) == 3
        for rec in records:
            assert rec["reason"] == "quarantine"
            assert rec["predicate"] == "group.source == 'judge-v3'"
            assert rec["note"] == "reward hack in judge v3"
            assert rec["new_priority_int"] == "0"
        verify_chain([dict(r) for r in buf.attestation_log.records])

    def test_predicate_text_is_derived_from_the_source_when_not_given(self) -> None:
        buf = filled()
        buf.quarantine(lambda r, g: r.reward > 0.9, "suspicious perfect scores")
        texts = {rec["predicate"] for rec in quarantine_records(buf)}
        assert len(texts) == 1
        (text,) = texts
        assert "r.reward > 0.9" in text
        assert "\n" not in text

    def test_describe_predicate_falls_back_to_the_name(self) -> None:
        assert "len" in describe_predicate(len)

    def test_predicate_sees_rollout_and_group(self) -> None:
        buf = filled()
        seen = []
        buf.quarantine(lambda r, g: seen.append((r.reward, g.prompt_id)) or False, "look")
        assert len(seen) == buf.size and buf.size == 6
        assert quarantine_records(buf) == []

    def test_nothing_matched_writes_nothing(self) -> None:
        buf = filled()
        n = len(buf.attestation_log.records)
        assert buf.quarantine(lambda r, g: False, "nothing") == ()
        assert len(buf.attestation_log.records) == n

    def test_predicate_runs_before_any_eviction(self) -> None:
        buf = filled()
        calls = []

        def predicate(r, g):
            calls.append(len(buf))
            return g.prompt_id == "hacked"

        buf.quarantine(predicate, "hack")
        assert set(calls) == {6}

    def test_raising_predicate_leaves_the_buffer_unchanged(self) -> None:
        buf = filled()
        before = (buf.size, len(buf.attestation_log.records))

        def predicate(r, g):
            if g.prompt_id == "other":
                raise RuntimeError("boom")
            return True

        with pytest.raises(RuntimeError, match="boom"):
            buf.quarantine(predicate, "hack")
        assert (buf.size, len(buf.attestation_log.records)) == before

    def test_non_bool_predicate_result_is_an_error(self) -> None:
        buf = filled()
        with pytest.raises(TypeError, match="bool"):
            buf.quarantine(lambda r, g: r, "hack")
        assert buf.size == 6

    @pytest.mark.parametrize("reason", ["", "a\nb", "x" * (MAX_NOTE_LENGTH + 1), 3, None])
    def test_bad_reason_is_rejected_before_any_change(self, reason) -> None:
        buf = filled()
        with pytest.raises(ValueError, match="reason"):
            buf.quarantine(lambda r, g: True, reason)
        assert buf.size == 6

    @pytest.mark.parametrize("text", ["", "a\tb", "x" * (MAX_PREDICATE_LENGTH + 1), 7])
    def test_bad_predicate_text_is_rejected(self, text) -> None:
        buf = filled()
        with pytest.raises(ValueError, match="predicate"):
            buf.quarantine(lambda r, g: True, "hack", predicate_text=text)
        assert buf.size == 6

    def test_quarantine_positions_rejects_an_empty_slot_before_evicting_any(self) -> None:
        buf = filled()
        free = next(p for p in range(buf.capacity) if p not in buf.live_positions())
        with pytest.raises(ValueError, match="no live entry"):
            buf.quarantine_positions([0, free], "p", "n")
        assert buf.size == 6

    def test_evict_still_refuses_the_quarantine_reason(self) -> None:
        buf = filled()
        with pytest.raises(ValueError, match="quarantine"):
            buf.evict(0, "quarantine")

    def test_quarantined_sampled_slot_drops_last_batch(self) -> None:
        buf = filled()
        batch = buf.sample(2, current_version=1)
        buf.quarantine(lambda r, g: True, "all")
        assert buf.last_batch is None and batch.op_counter == 1

    def test_snapshot_round_trip_keeps_the_records(self) -> None:
        buf = filled()
        buf.quarantine(lambda r, g: g.prompt_id == "hacked", "hack", predicate_text="hacked")
        state = buf.state_dict()
        fresh = RolloutBuffer(attest=AttestationLog(), **KW)
        fresh.load_state_dict(state)
        assert fresh.attestation_log.records == buf.attestation_log.records
        assert fresh.size == 3


def rechained(records: list[dict]) -> list[dict]:
    from reservoir.attest import _digest_record
    out, prev = [], "genesis"
    for rec in records:
        rec = dict(rec, prev_digest=prev)
        rec["digest"] = _digest_record(rec)
        out.append(rec)
        prev = rec["digest"]
    return out


class TestFormatAndIsolation:
    @pytest.mark.parametrize("fmt", ["1", "2"])
    def test_buffer_restored_from_an_older_log_refuses_to_quarantine(self, fmt) -> None:
        buf = filled()
        state = buf.state_dict()
        state["attestation"][0]["format"] = fmt
        state["attestation"] = rechained(state["attestation"])
        old = RolloutBuffer(attest=AttestationLog(), **KW)
        old.load_state_dict(state)
        with pytest.raises(ValueError, match=f"format-{fmt} log"):
            old.quarantine(lambda r, g: True, "hack")
        assert old.size == 6 and not quarantine_records(old)
        old.evict(0)                                                      # other evictions still work

    def test_durable_buffer_from_an_older_log_refuses_before_logging(self, tmp_path) -> None:
        buf = open_buf(tmp_path)
        fill_durable(buf)
        state = buf.state_dict()
        buf.close()
        state["attestation"][0]["format"] = "2"
        state["attestation"] = rechained(state["attestation"])
        snapshot = json.loads((tmp_path / "buf" / "state.json").read_text())
        snapshot["buffer"] = state
        (tmp_path / "buf" / "state.json").write_text(json.dumps(snapshot))
        (tmp_path / "buf" / "wal.jsonl").write_bytes(b"")
        again = open_buf(tmp_path)
        pending = again.pending_commands
        with pytest.raises(ValueError, match="format-2 log"):
            again.quarantine(lambda r, g: True, "hack")
        assert again.pending_commands == pending and again.size == 6
        again.close()

    def test_predicate_cannot_change_what_the_buffer_stores(self) -> None:
        buf = RolloutBuffer(attest=AttestationLog(), **KW)
        buf.add_group("p", 0, rollouts([1.0, 0.0], prompt_ids=[1, 2], rewards={"judge": 1.0}), source="a")
        before = buf.state_dict()

        def predicate(r, g):
            r.metadata["prompt_ids"].append(99)
            r.metadata["rewards"]["judge"] = 0.0
            return False

        assert buf.quarantine(predicate, "probe") == ()
        assert buf.state_dict() == before

    def test_predicate_sees_equal_values(self) -> None:
        buf = filled()
        seen = []
        buf.quarantine(lambda r, g: seen.append((r.tokens, r.reward, g.prompt_id, g.source, g.model_version)) or False,
                       "look")
        assert sorted(seen) == sorted((buf.entry(p)[0].tokens, buf.entry(p)[0].reward, buf.entry(p)[1].prompt_id,
                                       buf.entry(p)[1].source, buf.entry(p)[1].model_version)
                                      for p in buf.live_positions())

    def test_derived_text_replaces_non_printable_characters(self, monkeypatch) -> None:
        import inspect
        monkeypatch.setattr(inspect, "getsource", lambda f: "lambda r, g:\n    g.source == 'zero\u200bwidth'")
        text = describe_predicate(lambda r, g: True)
        assert text == "lambda r, g: g.source == 'zero?width'"

    def test_quarantine_works_without_attestation(self) -> None:
        buf = RolloutBuffer(**KW)
        buf.add_group("p", 0, rollouts([1.0, 0.0]), source="a")
        assert len(buf.quarantine(lambda r, g: True, "all")) == 2 and buf.size == 0

    def test_durable_buffer_without_attestation_quarantines_and_reopens(self, tmp_path) -> None:
        buf = DurableRolloutBuffer(tmp_path / "plain", compact_every=1, **KW)
        buf.add_group("p", 0, rollouts([1.0, 0.0]), source="a")
        buf.add_group("q", 1, rollouts([0.5]), source="a")
        buf.quarantine(lambda r, g: g.prompt_id == "p", "hack")
        state = buf.state_dict()
        buf.close()
        again = DurableRolloutBuffer(tmp_path / "plain", compact_every=1, **KW)
        assert again.state_dict() == state and again.size == 1
        again.close()

    def test_malformed_rewards_are_ignored_without_a_manifest(self) -> None:
        buf = RolloutBuffer(attest=AttestationLog(), **KW)
        buf.add_group("p", 0, rollouts([1.0], rewards="free text"), source="a")
        assert buf.size == 1


class TestAttestFields:
    def test_quarantine_record_needs_both_texts(self) -> None:
        log = AttestationLog()
        with pytest.raises(ValueError, match="predicate"):
            log.append_mutation("evict", 0, 5, 0, 1, base_priority_int=5, entry_version=0, base_epoch=0,
                                reason="quarantine", note="n")
        with pytest.raises(ValueError, match="note"):
            log.append_mutation("evict", 0, 5, 0, 1, base_priority_int=5, entry_version=0, base_epoch=0,
                                reason="quarantine", predicate="p")

    @pytest.mark.parametrize("reason", ["stale", "capacity", "explicit", "drift"])
    def test_texts_only_with_the_quarantine_reason(self, reason) -> None:
        log = AttestationLog()
        with pytest.raises(ValueError, match="quarantine"):
            log.append_mutation("evict", 0, 5, 0, 1, base_priority_int=5, entry_version=0, base_epoch=0,
                                reason=reason, predicate="p", note="n")

    def test_log_format_is_three(self) -> None:
        buf = RolloutBuffer(attest=AttestationLog(), **KW)
        assert buf.attestation_log.records[0]["format"] == "3"


def open_buf(tmp_path: Path, **kw) -> DurableRolloutBuffer:
    return DurableRolloutBuffer(tmp_path / "buf", attest=tmp_path / "attest.jsonl",
                                manifest=tmp_path / "manifest.jsonl", **dict(KW, **kw))


def fill_durable(buf) -> None:
    buf.add_group("clean", 0, rollouts([1.0, 0.0]), source="a")
    buf.add_group("hacked", 1, rollouts([1.0, 1.0, 0.5]), source="judge-v3")
    buf.add_group("other", 1, rollouts([0.25]), source="a")


class TestDurableQuarantine:
    def test_command_logs_positions_and_texts_not_the_callable(self, tmp_path) -> None:
        buf = open_buf(tmp_path)
        fill_durable(buf)
        positions = buf.quarantine(lambda r, g: g.source == "judge-v3", "reward hack", predicate_text="judge-v3")
        buf.close()
        lines = [json.loads(l) for l in (tmp_path / "buf" / "wal.jsonl").read_text().splitlines()]
        cmd = next(l for l in lines if l["op"] == "quarantine")
        assert cmd["args"] == {"positions": list(positions), "predicate": "judge-v3", "note": "reward hack"}

    def test_reopen_replays_the_quarantine(self, tmp_path) -> None:
        buf = open_buf(tmp_path)
        fill_durable(buf)
        buf.quarantine(lambda r, g: g.source == "judge-v3", "reward hack")
        records = buf.attestation_log.records
        live = buf.live_positions()
        buf.close()
        again = open_buf(tmp_path)
        assert again.attestation_log.records == records
        assert again.live_positions() == live
        assert len([r for r in records if r.get("reason") == "quarantine"]) == 3
        again.close()
        verify_chain([json.loads(l) for l in (tmp_path / "attest.jsonl").read_text().splitlines()])

    def test_compaction_after_quarantine_keeps_the_state(self, tmp_path) -> None:
        buf = open_buf(tmp_path, compact_every=1)
        fill_durable(buf)
        buf.quarantine(lambda r, g: g.prompt_id == "hacked", "hack")
        buf.add_group("after", 2, rollouts([0.5]), source="a")
        state = buf.state_dict()
        buf.close()
        again = open_buf(tmp_path, compact_every=1)
        assert again.state_dict() == state
        again.close()

    def test_nothing_matched_logs_no_command(self, tmp_path) -> None:
        buf = open_buf(tmp_path)
        fill_durable(buf)
        pending = buf.pending_commands
        assert buf.quarantine(lambda r, g: False, "nothing") == ()
        assert buf.pending_commands == pending
        buf.close()

    def test_raising_predicate_changes_nothing_on_disk_or_in_memory(self, tmp_path) -> None:
        buf = open_buf(tmp_path)
        fill_durable(buf)
        pending, size = buf.pending_commands, buf.size

        def predicate(r, g):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            buf.quarantine(predicate, "hack")
        assert (buf.pending_commands, buf.size) == (pending, size)
        buf.close()

    def test_rewards_survive_compaction_reopen_and_restore(self, tmp_path) -> None:
        buf = open_buf(tmp_path, compact_every=1)
        buf.add_group("p", 0, rollouts([1.0, 0.0], rewards={"verifier": 1.0, "judge": 0.25}), source="a")
        buf.checkpoint("c")
        buf.add_group("q", 1, rollouts([0.5], rewards={}), source="a")
        buf.quarantine(lambda r, g: g.prompt_id == "q", "hack")
        lines = buf.manifest_records
        buf.close()
        again = open_buf(tmp_path, compact_every=1)
        assert again.manifest_records == lines
        assert [l.get("rewards") for l in lines] == [{"verifier": 1.0, "judge": 0.25}] * 2 + [{}]
        again.restore_checkpoint("c")
        assert again.manifest_records == lines[:2] and again.size == 2
        again.close()
        verify_chain([json.loads(l) for l in (tmp_path / "attest.jsonl").read_text().splitlines()],
                     manifest=[json.loads(l) for l in (tmp_path / "manifest.jsonl").read_text().splitlines()])

    def test_apply_command_replays_a_quarantine(self) -> None:
        buf = filled()
        apply_command(buf, {"op": "quarantine", "args": {"positions": [2, 3, 4], "predicate": "p", "note": "n"}})
        assert buf.size == 3
        assert [r["index"] for r in quarantine_records(buf)] == [2, 3, 4]
