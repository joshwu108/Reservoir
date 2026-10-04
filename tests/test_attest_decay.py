"""Tests for the age-decay extension of the attestation schema.

Three new record types (``decay_config``, ``advance_version``, ``rebase``)
and three optional fields on mutation records (``base_priority_int``,
``entry_version``, ``base_epoch``, plus ``reason`` on evicts) let an
independent checker recompute every decayed leaf from the decay formula
instead of trusting the recorded value. The extension must be backward
compatible: a log written without any of it is byte-for-byte what it was
before, so the existing mutation campaign still applies.

The second half checks that ``RolloutBuffer`` emits these records in the
order the checker expects: ``advance_version``, stale ``evict`` records,
an optional single ``rebase``, then the writes.
"""

from __future__ import annotations

import json

import pytest

from reservoir import attest as attest_module
from reservoir.attest import AttestationLog, _digest_record
from reservoir.decay import DecayParams
from reservoir.decayed_tree import DecayedPriorityTree
from reservoir.rollout import MAX_SOURCE_LENGTH, Rollout, RolloutGroup, content_digest_of
from reservoir.rollout_attest import RolloutAttester
from reservoir.rollout_buffer import RolloutBuffer
from reservoir.rollout_manifest import ManifestWriter

# Digests of two records appended by the pre-extension code. If the
# extension changed how a legacy record is serialised, these would move.
LEGACY_INSERT_DIGEST = "fc5d1765f8466d14728bbfc075b4f01d7cb23f35fea058a6d0ee0e5707512995"
LEGACY_EVICT_DIGEST = "bd445e3b4febf697377cc36c094289af458a4e4aa913c6e056e07b2d7131261d"


def rollouts(rewards: list[float]) -> list[Rollout]:
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in rewards]


def config_kwargs(**overrides) -> dict:
    base = dict(
        half_life=4, max_policy_age=16, capacity=8, priority_bits=32,
        priority_frac_bits=16, table_frac_bits=31, rebase_slack=0,
        reset_age_on_update=False,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_legacy_mutation_digests_are_unchanged(self) -> None:
        log = AttestationLog()
        assert log.append_mutation("insert", 3, 0, 4096, 7)["digest"] == LEGACY_INSERT_DIGEST
        assert log.append_mutation("evict", 3, 4096, 0, 9)["digest"] == LEGACY_EVICT_DIGEST

    def test_legacy_mutation_has_no_decay_keys(self) -> None:
        rec = AttestationLog().append_mutation("insert", 0, 0, 5, 1)
        assert set(rec) == {
            "op", "index", "old_priority_int", "new_priority_int", "op_counter",
            "prev_digest", "digest",
        }


# ---------------------------------------------------------------------------
# decay_config
# ---------------------------------------------------------------------------

class TestDecayConfig:
    def test_first_record_fields(self) -> None:
        log = AttestationLog()
        rec = log.append_decay_config(**config_kwargs())
        assert rec["op"] == "decay_config"
        assert rec["half_life"] == "4"
        assert rec["max_policy_age"] == "16"
        assert rec["capacity"] == "8"
        assert rec["priority_bits"] == "32"
        assert rec["priority_frac_bits"] == "16"
        assert rec["table_frac_bits"] == "31"
        assert rec["rebase_slack"] == "0"
        assert rec["reset_age_on_update"] is False
        assert rec["prev_digest"] == "genesis"
        assert rec["digest"] == _digest_record(rec)
        assert log.head_digest == rec["digest"]

    def test_must_be_first(self) -> None:
        log = AttestationLog()
        log.append_mutation("insert", 0, 0, 5, 1)
        with pytest.raises(ValueError, match="first"):
            log.append_decay_config(**config_kwargs())

    @pytest.mark.parametrize("field", ["half_life", "capacity", "priority_bits"])
    def test_rejects_bad_ints(self, field: str) -> None:
        for bad in (0, -1, 1.5, "4", True):
            with pytest.raises(ValueError, match=field):
                AttestationLog().append_decay_config(**config_kwargs(**{field: bad}))

    def test_rejects_non_bool_reset_flag(self) -> None:
        with pytest.raises(ValueError, match="reset_age_on_update"):
            AttestationLog().append_decay_config(**config_kwargs(reset_age_on_update=1))


# ---------------------------------------------------------------------------
# Mutation records with decay fields
# ---------------------------------------------------------------------------

class TestDecayMutationFields:
    def test_insert_with_decay_fields(self) -> None:
        log = AttestationLog()
        rec = log.append_mutation(
            "insert", 2, 0, 1 << 20, 5,
            base_priority_int=1000, entry_version=7, base_epoch=1,
        )
        assert rec["base_priority_int"] == "1000"
        assert rec["entry_version"] == "7"
        assert rec["base_epoch"] == "1"
        assert "reason" not in rec
        assert rec["digest"] == _digest_record(rec)

    def test_evict_with_reason(self) -> None:
        rec = AttestationLog().append_mutation(
            "evict", 2, 55, 0, 5,
            base_priority_int=1000, entry_version=7, base_epoch=1, reason="stale",
        )
        assert rec["reason"] == "stale"

    def test_all_three_decay_fields_or_none(self) -> None:
        log = AttestationLog()
        with pytest.raises(ValueError, match="base_priority_int"):
            log.append_mutation("insert", 0, 0, 5, 1, entry_version=1, base_epoch=0)
        with pytest.raises(ValueError, match="entry_version"):
            log.append_mutation("insert", 0, 0, 5, 1, base_priority_int=1, base_epoch=0)
        with pytest.raises(ValueError, match="base_epoch"):
            log.append_mutation("insert", 0, 0, 5, 1, base_priority_int=1, entry_version=1)

    def test_reason_only_on_evict(self) -> None:
        log = AttestationLog()
        with pytest.raises(ValueError, match="reason"):
            log.append_mutation(
                "insert", 0, 0, 5, 1,
                base_priority_int=1, entry_version=1, base_epoch=0, reason="stale",
            )

    def test_reason_must_be_known(self) -> None:
        with pytest.raises(ValueError, match="reason"):
            AttestationLog().append_mutation(
                "evict", 0, 5, 0, 1,
                base_priority_int=1, entry_version=1, base_epoch=0, reason="bored",
            )

    def test_evict_without_reason_is_legacy_and_allowed(self) -> None:
        rec = AttestationLog().append_mutation("evict", 0, 5, 0, 1)
        assert "reason" not in rec

    @pytest.mark.parametrize("field", ["base_priority_int", "entry_version", "base_epoch"])
    def test_negative_decay_field_rejected(self, field: str) -> None:
        kwargs = dict(base_priority_int=1, entry_version=1, base_epoch=0)
        kwargs[field] = -1
        with pytest.raises(ValueError, match=field):
            AttestationLog().append_mutation("insert", 0, 0, 5, 1, **kwargs)


# ---------------------------------------------------------------------------
# advance_version and rebase
# ---------------------------------------------------------------------------

class TestAdvanceAndRebase:
    def test_advance_version_record(self) -> None:
        log = AttestationLog()
        rec = log.append_advance_version(old_version=3, new_version=9, op_counter=2)
        assert rec["op"] == "advance_version"
        assert rec["old_version"] == "3"
        assert rec["new_version"] == "9"
        assert rec["op_counter"] == 2
        assert rec["digest"] == _digest_record(rec)

    def test_advance_version_must_not_go_backwards(self) -> None:
        with pytest.raises(ValueError, match="new_version"):
            AttestationLog().append_advance_version(old_version=5, new_version=4, op_counter=1)

    def test_rebase_record(self) -> None:
        rec = AttestationLog().append_rebase(
            old_base_epoch=2, new_base_epoch=5, root_total_before=800,
            root_total_after=100, op_counter=4,
        )
        assert rec["op"] == "rebase"
        assert rec["old_base_epoch"] == "2"
        assert rec["new_base_epoch"] == "5"
        assert rec["root_total_before"] == "800"
        assert rec["root_total_after"] == "100"
        assert rec["digest"] == _digest_record(rec)

    def test_rebase_must_advance_the_base(self) -> None:
        with pytest.raises(ValueError, match="new_base_epoch"):
            AttestationLog().append_rebase(2, 2, 8, 8, 1)

    def test_round_trip_and_chain(self) -> None:
        log = AttestationLog()
        log.append_decay_config(**config_kwargs())
        log.append_advance_version(0, 3, 0)
        log.append_mutation("insert", 0, 0, 10, 0, base_priority_int=5, entry_version=3, base_epoch=0)
        log.append_rebase(0, 1, 10, 5, 0)
        log.append_mutation("evict", 0, 5, 0, 0, base_priority_int=5, entry_version=3, base_epoch=1, reason="explicit")
        text = log.to_json_lines()
        again = AttestationLog.from_json_lines(text)
        assert again.records == log.records
        assert again.head_digest == log.head_digest
        prev = "genesis"
        for rec in again.records:
            assert rec["prev_digest"] == prev
            assert rec["digest"] == _digest_record(rec)
            prev = rec["digest"]
        for line in text.split("\n"):
            json.loads(line)


# ---------------------------------------------------------------------------
# RolloutBuffer emits the new schema in the required order
# ---------------------------------------------------------------------------

class TestBufferWiring:
    def test_decay_config_is_the_first_record_and_matches_params(self) -> None:
        log = AttestationLog()
        buf = RolloutBuffer(capacity=8, half_life=3, max_policy_age=7, rebase_slack=1,
                            reset_age_on_update=True, attest=log)
        rec = log.records[0]
        assert rec["op"] == "decay_config"
        assert int(rec["half_life"]) == buf.params.half_life == 3
        assert int(rec["max_policy_age"]) == 7
        assert int(rec["capacity"]) == buf.params.capacity == 8
        assert int(rec["priority_bits"]) == buf.params.priority_bits
        assert int(rec["priority_frac_bits"]) == buf.params.priority_frac_bits
        assert int(rec["table_frac_bits"]) == buf.params.table_frac_bits
        assert int(rec["rebase_slack"]) == 1
        assert rec["reset_age_on_update"] is True

    def test_add_group_emits_advance_then_inserts_with_decay_fields(self) -> None:
        log = AttestationLog()
        buf = RolloutBuffer(capacity=8, attest=log)
        buf.add_group("p", 3, rollouts([1.0, 0.0]))
        ops = [r["op"] for r in log.records]
        assert ops == ["decay_config", "advance_version", "insert", "insert"]
        adv = log.records[1]
        assert (adv["old_version"], adv["new_version"]) == ("0", "3")
        for rec, pos in zip(log.records[2:], (0, 1)):
            assert rec["index"] == pos
            assert int(rec["base_priority_int"]) == buf.base_priority(pos)
            assert int(rec["entry_version"]) == 3
            assert int(rec["base_epoch"]) == buf.base_epoch
            assert int(rec["new_priority_int"]) == buf.leaf(pos)

    def test_no_advance_record_when_version_unchanged(self) -> None:
        log = AttestationLog()
        buf = RolloutBuffer(capacity=8, attest=log)
        buf.add_group("a", 2, rollouts([1.0]))
        buf.add_group("b", 2, rollouts([1.0]))
        buf.sample(1)
        assert [r["op"] for r in log.records].count("advance_version") == 1

    def test_stale_evictions_carry_reason_and_follow_advance(self) -> None:
        log = AttestationLog()
        buf = RolloutBuffer(capacity=8, max_policy_age=2, attest=log)
        buf.add_group("old", 0, rollouts([1.0, 0.0]))
        buf.advance(3)  # age 3 > max_policy_age 2: both entries expire
        ops = [r["op"] for r in log.records]
        assert ops == ["decay_config", "insert", "insert", "advance_version", "evict", "evict"]
        for rec in log.records[4:6]:
            assert rec["reason"] == "stale"
            assert rec["new_priority_int"] == "0"
            assert int(rec["entry_version"]) == 0

    def test_capacity_eviction_reason(self) -> None:
        log = AttestationLog()
        buf = RolloutBuffer(capacity=2, max_policy_age=100, attest=log)
        buf.add_group("a", 0, rollouts([1.0, 0.0]))
        buf.add_group("b", 1, rollouts([1.0]))
        evicts = [r for r in log.records if r["op"] == "evict"]
        assert len(evicts) == 1 and evicts[0]["reason"] == "capacity"

    def test_rebase_is_one_record_between_evictions_and_writes(self) -> None:
        log = AttestationLog()
        buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=2, attest=log)  # max_shift 2
        buf.add_group("g0", 0, rollouts([1.0, 0.0]))
        buf.add_group("g3", 3, rollouts([1.0, 0.0]))  # epoch 3 > 2: evict g0 (age 3), then rebase
        ops = [r["op"] for r in log.records]
        assert ops == [
            "decay_config", "insert", "insert",
            "advance_version", "evict", "evict", "rebase", "insert", "insert",
        ]
        reb = log.records[6]
        assert int(reb["old_base_epoch"]) == 0
        assert int(reb["new_base_epoch"]) == buf.base_epoch
        assert int(reb["root_total_before"]) == 0 == int(reb["root_total_after"])
        assert buf.n_rebases == 1

    def test_rebase_totals_match_the_shift(self) -> None:
        log = AttestationLog()
        buf = RolloutBuffer(capacity=8, half_life=2, max_policy_age=4, attest=log)  # max_shift 2
        buf.add_group("a", 4, rollouts([1.0, 0.0]))
        total_before = buf.total
        buf.advance(7)  # epoch 3 > 2: rebase, nothing expires
        reb = [r for r in log.records if r["op"] == "rebase"]
        assert len(reb) == 1
        shift = int(reb[0]["new_base_epoch"]) - int(reb[0]["old_base_epoch"])
        assert int(reb[0]["root_total_before"]) == total_before
        assert int(reb[0]["root_total_after"]) == total_before >> shift == buf.total

    def test_update_records_carry_version_per_reset_flag(self) -> None:
        for reset in (False, True):
            log = AttestationLog()
            buf = RolloutBuffer(capacity=8, reset_age_on_update=reset, attest=log)
            buf.add_group("p", 0, rollouts([1.0, 0.0]))
            buf.sample(1, current_version=5)
            buf.update_priorities([0], [0.75])
            upd = [r for r in log.records if r["op"] == "update"]
            assert len(upd) == 1
            assert int(upd[0]["entry_version"]) == (5 if reset else 0)
            assert int(upd[0]["base_priority_int"]) == buf.base_priority(0)


# ---------------------------------------------------------------------------
# Content fields on insert records, and the manifest the attester mirrors
# ---------------------------------------------------------------------------

DIGEST = "ab" * 32


class TestContentMutationFields:
    def test_insert_with_content_fields(self) -> None:
        log = AttestationLog()
        rec = log.append_mutation("insert", 0, 0, 5, 0, base_priority_int=5, entry_version=0,
                                  base_epoch=0, content_digest=DIGEST, source="gsm8k")
        assert rec["content_digest"] == DIGEST
        assert rec["source"] == "gsm8k"

    def test_omitted_fields_are_absent_and_bytes_unchanged(self) -> None:
        with_kwargs = AttestationLog().append_mutation(
            "insert", 0, 0, 5, 0, base_priority_int=5, entry_version=0, base_epoch=0,
            content_digest=None, source=None)
        without = AttestationLog().append_mutation(
            "insert", 0, 0, 5, 0, base_priority_int=5, entry_version=0, base_epoch=0)
        assert "content_digest" not in with_kwargs and "source" not in with_kwargs
        assert with_kwargs["digest"] == without["digest"]

    def test_content_digest_without_source(self) -> None:
        rec = AttestationLog().append_mutation("insert", 0, 0, 5, 0, content_digest=DIGEST)
        assert rec["content_digest"] == DIGEST and "source" not in rec

    @pytest.mark.parametrize("op", ["update", "evict"])
    def test_content_fields_only_on_insert(self, op: str) -> None:
        with pytest.raises(ValueError, match="insert"):
            AttestationLog().append_mutation(op, 0, 5, 6, 0, content_digest=DIGEST)

    def test_source_requires_digest(self) -> None:
        with pytest.raises(ValueError, match="content_digest"):
            AttestationLog().append_mutation("insert", 0, 0, 5, 0, source="s")

    @pytest.mark.parametrize("bad", ["AB" * 32, "ab" * 31, "zz" * 32, 7, ""])
    def test_malformed_digest_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="content_digest"):
            AttestationLog().append_mutation("insert", 0, 0, 5, 0, content_digest=bad)

    @pytest.mark.parametrize("bad", ["", "a\nb", "x" * 257, "   ", "a\u200bb", 3])
    def test_malformed_source_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="source"):
            AttestationLog().append_mutation("insert", 0, 0, 5, 0, content_digest=DIGEST, source=bad)

    def test_source_boundary_accepted(self) -> None:
        rec = AttestationLog().append_mutation("insert", 0, 0, 5, 0, content_digest=DIGEST, source="x" * 256)
        assert len(rec["source"]) == 256

    def test_source_bound_matches_rollout_module(self) -> None:
        # attest.py imports nothing from the rest of the package, so the
        # bound is duplicated; this pins the two copies together.
        assert attest_module._MAX_SOURCE_LENGTH == MAX_SOURCE_LENGTH
        for text in ("ok", "x" * MAX_SOURCE_LENGTH, "é"):
            RolloutGroup(prompt_id="p", model_version=0, rollouts=rollouts([1.0]), source=text)
            AttestationLog().append_mutation("insert", 0, 0, 5, 0, content_digest=DIGEST, source=text)
        for text in ("", " ", "x" * (MAX_SOURCE_LENGTH + 1), "a\tb"):
            with pytest.raises(ValueError):
                RolloutGroup(prompt_id="p", model_version=0, rollouts=rollouts([1.0]), source=text)
            with pytest.raises(ValueError):
                AttestationLog().append_mutation("insert", 0, 0, 5, 0, content_digest=DIGEST, source=text)


def _params() -> DecayParams:
    return DecayParams(half_life=4, max_policy_age=16, capacity=8, priority_bits=32,
                       priority_frac_bits=16, rebase_slack=0)


def _group(source=None) -> RolloutGroup:
    return RolloutGroup(prompt_id="p", model_version=0, rollouts=rollouts([1.0, 0.0]), source=source)


def _insert(att: RolloutAttester, event, op_counter: int, group: RolloutGroup, member: int) -> None:
    """What RolloutBuffer.add_group does per rollout: prepare up front, then record."""
    prepared = att.prepare_inserts(group, op_counter)
    if prepared:
        att.record_insert(event, op_counter, prepared[member])


class TestAttesterManifest:
    def test_manifest_requires_attestation(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="manifest"):
            RolloutAttester(None, _params(), False, manifest=tmp_path / "m.jsonl")

    def test_record_insert_writes_log_fields_and_manifest_line(self, tmp_path) -> None:
        att = RolloutAttester(tmp_path / "a.jsonl", _params(), False, manifest=tmp_path / "m.jsonl")
        group = _group(source="zen")
        tree = DecayedPriorityTree(_params())
        event = tree.write(3, 1234, 0)
        _insert(att, event, 2, group, 1)
        att.close()
        rec = att.log.records[-1]
        assert rec["op"] == "insert" and rec["index"] == 3
        assert rec["content_digest"] == group.content_digests[1] == content_digest_of("p", group.rollouts[1].tokens, 0.0)
        assert rec["source"] == "zen"
        line = json.loads((tmp_path / "m.jsonl").read_text().splitlines()[0])
        assert line["op_counter"] == 2 and line["index"] == 3
        assert line["content_digest"] == rec["content_digest"]
        assert line["prompt_id"] == "p" and line["source"] == "zen"
        assert line["tokens"] == list(group.rollouts[1].tokens)
        assert line["reward_hex"] == (0.0).hex()
        assert line["entry_version"] == 0
        assert att.manifest_records == [line]

    def test_prepare_inserts_is_empty_when_disabled_and_carries_lines_otherwise(self, tmp_path) -> None:
        assert RolloutAttester(None, _params(), False).prepare_inserts(_group("s"), 3) == ()
        plain = RolloutAttester(AttestationLog(), _params(), False).prepare_inserts(_group("s"), 3)
        assert [p.manifest_line for p in plain] == [None, None]
        assert [p.source for p in plain] == ["s", "s"]
        att = RolloutAttester(tmp_path / "a.jsonl", _params(), False, manifest=tmp_path / "m.jsonl")
        with_lines = att.prepare_inserts(_group("s"), 3)
        att.close()
        assert [p.manifest_line["op_counter"] for p in with_lines] == [3, 3]
        assert with_lines[1].manifest_line["reward_hex"] == (0.0).hex()
        assert [p.content_digest for p in with_lines] == list(_group("s").content_digests)

    def test_record_insert_without_manifest_still_writes_log_fields(self) -> None:
        att = RolloutAttester(AttestationLog(), _params(), False)
        event = DecayedPriorityTree(_params()).write(0, 10, 0)
        _insert(att, event, 0, _group(), 0)
        rec = att.log.records[-1]
        assert "content_digest" in rec and "source" not in rec
        assert att.manifest_records == []

    def test_record_write_for_update_and_evict_adds_no_manifest_line(self, tmp_path) -> None:
        att = RolloutAttester(tmp_path / "a.jsonl", _params(), False, manifest=tmp_path / "m.jsonl")
        tree = DecayedPriorityTree(_params())
        _insert(att, tree.write(0, 10, 0), 0, _group(), 0)
        att.record_write(tree.write(0, 20, 0), 0)
        att.record_write(tree.evict(0, "explicit"), 0)
        att.close()
        assert len((tmp_path / "m.jsonl").read_text().splitlines()) == 1
        assert "content_digest" not in att.log.records[-1]

    def test_disabled_attester_ignores_everything(self) -> None:
        att = RolloutAttester(None, _params(), False)
        _insert(att, DecayedPriorityTree(_params()).write(0, 10, 0), 0, _group(), 0)
        assert att.log is None and att.manifest_records == []

    def test_restore_rewrites_both_files(self, tmp_path) -> None:
        att = RolloutAttester(tmp_path / "a.jsonl", _params(), False, manifest=tmp_path / "m.jsonl")
        tree = DecayedPriorityTree(_params())
        _insert(att, tree.write(0, 10, 0), 0, _group(), 0)
        _insert(att, tree.write(1, 10, 0), 0, _group(), 1)
        log_records, manifest_records = att.log.records[:2], att.manifest_records[:1]
        att.restore(log_records, manifest_records)
        att.close()
        assert att.log.records == log_records
        assert att.manifest_records == manifest_records
        assert len((tmp_path / "a.jsonl").read_text().splitlines()) == 2
        assert len((tmp_path / "m.jsonl").read_text().splitlines()) == 1

    def test_restore_rejects_manifest_records_without_manifest(self, tmp_path) -> None:
        att = RolloutAttester(tmp_path / "a.jsonl", _params(), False)
        with pytest.raises(ValueError, match="manifest"):
            att.restore(att.log.records, [{"op_counter": 0}])
        att.close()

    def test_restore_with_bad_manifest_leaves_everything_unchanged(self, tmp_path) -> None:
        att = RolloutAttester(tmp_path / "a.jsonl", _params(), False, manifest=tmp_path / "m.jsonl")
        tree = DecayedPriorityTree(_params())
        _insert(att, tree.write(0, 10, 0), 0, _group(), 0)
        log_before, manifest_before = att.log.records, att.manifest_records
        files_before = ((tmp_path / "a.jsonl").read_text(), (tmp_path / "m.jsonl").read_text())
        tampered = [dict(manifest_before[0], tokens=[9, 9])]
        with pytest.raises(ValueError, match="content_digest"):
            att.restore(log_before[:1], tampered)
        with pytest.raises(ValueError, match="does not match"):
            att.restore(log_before[:1], manifest_before)   # one line, zero content inserts
        with pytest.raises(ValueError, match="does not match"):
            att.restore(log_before, [])                    # one content insert, zero lines
        att.close()
        assert att.log.records == log_before and att.manifest_records == manifest_before
        assert ((tmp_path / "a.jsonl").read_text(), (tmp_path / "m.jsonl").read_text()) == files_before

    def test_restore_validates_before_touching_the_log(self, tmp_path) -> None:
        att = RolloutAttester(tmp_path / "a.jsonl", _params(), False, manifest=tmp_path / "m.jsonl")
        before = att.log.records
        with pytest.raises(ValueError, match="manifest record 0"):
            att.restore(before, [{"op_counter": 0}])
        assert att.log.records == before
        att.close()

    def test_failed_construction_leaves_no_manifest_behind(self, tmp_path) -> None:
        existing = tmp_path / "a.jsonl"
        existing.write_text("")
        with pytest.raises(FileExistsError):
            RolloutAttester(existing, _params(), False, manifest=tmp_path / "m.jsonl")
        assert not (tmp_path / "m.jsonl").exists()
        with pytest.raises(TypeError, match="attest"):
            RolloutAttester(object(), _params(), False, manifest=tmp_path / "m2.jsonl")  # type: ignore[arg-type]
        assert not (tmp_path / "m2.jsonl").exists()
        with pytest.raises(TypeError, match="manifest"):
            RolloutAttester(tmp_path / "b.jsonl", _params(), False, manifest=object())  # type: ignore[arg-type]
        assert not (tmp_path / "b.jsonl").exists()

    def test_close_closes_both_files(self, tmp_path) -> None:
        att = RolloutAttester(tmp_path / "a.jsonl", _params(), False, manifest=tmp_path / "m.jsonl")
        att.close()
        assert att._file is None and att._manifest._file is None
        att.close()

    def test_in_memory_manifest_writer(self) -> None:
        writer = ManifestWriter()
        att = RolloutAttester(AttestationLog(), _params(), False, manifest=writer)
        _insert(att, DecayedPriorityTree(_params()).write(0, 10, 0), 0, _group("s"), 0)
        assert att.has_manifest and writer.records == att.manifest_records
        assert writer.records[0]["source"] == "s"
