"""Tests for reservoir.rollout_manifest — the data behind each insert record's content digest.

The attestation log commits to every stored rollout through a content
digest; the manifest is the opening of that commitment, one JSON line per
insert, so an independent checker can recompute the digests. These tests
pin the line format, the file discipline (never append to an old file),
and the restore path the durable buffer relies on.
"""

from __future__ import annotations

import json

import pytest

from reservoir.rollout import content_digest_of
from reservoir.rollout_manifest import MANIFEST_KEYS, ManifestWriter, manifest_record, validate_manifest_records


def _record(**overrides) -> dict:
    base = dict(op_counter=3, index=7, prompt_id="p1", source="gsm8k", tokens=(1, 2, 3),
                reward=1.0, entry_version=5)
    base.update(overrides)
    return manifest_record(**base)


class TestManifestRecord:
    def test_keys_and_values(self) -> None:
        rec = _record()
        assert set(rec) == set(MANIFEST_KEYS)
        assert rec["op_counter"] == 3 and rec["index"] == 7
        assert rec["prompt_id"] == "p1" and rec["source"] == "gsm8k"
        assert rec["tokens"] == [1, 2, 3]
        assert rec["reward_hex"] == (1.0).hex()
        assert rec["entry_version"] == 5
        assert rec["content_digest"] == content_digest_of("p1", (1, 2, 3), 1.0)

    def test_source_absent_is_null(self) -> None:
        assert _record(source=None)["source"] is None

    def test_is_json_round_trippable(self) -> None:
        rec = _record()
        assert json.loads(json.dumps(rec)) == rec


class TestManifestWriter:
    def test_refuses_existing_file(self, tmp_path) -> None:
        path = tmp_path / "m.jsonl"
        path.write_text("")
        with pytest.raises(FileExistsError):
            ManifestWriter(path)

    def test_overwrite_flag_truncates(self, tmp_path) -> None:
        path = tmp_path / "m.jsonl"
        path.write_text("stale\n")
        ManifestWriter(path, overwrite=True).close()
        assert path.read_text() == ""

    def test_writes_one_canonical_line_per_record(self, tmp_path) -> None:
        path = tmp_path / "m.jsonl"
        w = ManifestWriter(path)
        a = w.write(_record(op_counter=0, index=0))
        b = w.write(_record(op_counter=0, index=1, reward=0.0))
        w.close()
        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert lines[0] == json.dumps(a, sort_keys=True, separators=(",", ":"))
        assert json.loads(lines[1]) == b
        assert w.records == [a, b]

    def test_records_is_a_copy(self, tmp_path) -> None:
        w = ManifestWriter(tmp_path / "m.jsonl")
        w.write(_record())
        w.records.clear()
        assert len(w.records) == 1
        w.close()

    def test_restore_rewrites_file_and_memory(self, tmp_path) -> None:
        path = tmp_path / "m.jsonl"
        w = ManifestWriter(path)
        w.write(_record(op_counter=9))
        recovered = [_record(op_counter=0), _record(op_counter=1, index=8)]
        w.restore(recovered)
        w.close()
        assert w.records == recovered
        assert [json.loads(l) for l in path.read_text().splitlines()] == recovered

    def test_restore_rejects_malformed_records_and_keeps_state(self, tmp_path) -> None:
        path = tmp_path / "m.jsonl"
        w = ManifestWriter(path)
        kept = w.write(_record())
        with pytest.raises(ValueError, match="manifest record 0"):
            w.restore([{"op_counter": 0}])
        assert w.records == [kept]
        assert json.loads(path.read_text().splitlines()[0]) == kept
        w.close()

    def test_in_memory_when_no_path(self) -> None:
        w = ManifestWriter()
        w.write(_record())
        assert len(w.records) == 1
        w.restore([_record(op_counter=5)])
        assert w.records[0]["op_counter"] == 5
        w.close()

    def test_write_stores_a_copy(self) -> None:
        w = ManifestWriter()
        rec = _record()
        w.write(rec)
        rec["tokens"].append(99)
        assert w.records[0]["tokens"] == [1, 2, 3]


class TestValidateManifestRecords:
    def test_accepts_well_formed(self) -> None:
        recs = [_record(), _record(op_counter=1, source=None)]
        assert validate_manifest_records(recs) == recs

    @pytest.mark.parametrize("edit, message", [
        (lambda r: r.update(tokens=[9]), "content_digest"),
        (lambda r: r.update(reward_hex=(2.0).hex()), "content_digest"),
        (lambda r: r.update(prompt_id="other"), "content_digest"),
        (lambda r: r.update(content_digest="00" * 32), "content_digest"),
        (lambda r: r.update(op_counter=-1), "op_counter"),
        (lambda r: r.update(index="7"), "index"),
        (lambda r: r.update(entry_version=True), "entry_version"),
        (lambda r: r.update(source=3), "source"),
        (lambda r: r.update(reward_hex=1.0), "reward_hex"),
        (lambda r: r.update(reward_hex="not hex"), "manifest record 0"),
        (lambda r: r.pop("tokens"), "manifest keys"),
        (lambda r: r.update(extra=1), "manifest keys"),
    ])
    def test_rejects_each_corruption(self, edit, message) -> None:
        rec = _record()
        edit(rec)
        with pytest.raises(ValueError, match=message):
            validate_manifest_records([rec])

    def test_rejects_non_list(self) -> None:
        with pytest.raises(ValueError, match="list"):
            validate_manifest_records({"a": 1})

    def test_close_is_idempotent(self, tmp_path) -> None:
        w = ManifestWriter(tmp_path / "m.jsonl")
        w.close()
        w.close()


class TestRewardProvenance:
    def test_rewards_are_written_when_given_and_absent_otherwise(self) -> None:
        rec = _record(rewards={"verifier": 1.0, "judge": 0})
        assert rec["rewards"] == {"verifier": 1.0, "judge": 0.0}
        assert all(isinstance(v, float) for v in rec["rewards"].values())
        assert set(rec) == set(MANIFEST_KEYS) | {"rewards"}
        assert "rewards" not in _record()
        assert "rewards" not in _record(rewards=None)

    def test_rewards_do_not_change_the_digest(self) -> None:
        assert _record(rewards={"judge": 0.5})["content_digest"] == _record()["content_digest"]

    def test_empty_rewards_mean_every_function_abstained(self) -> None:
        assert _record(rewards={})["rewards"] == {}

    @pytest.mark.parametrize("rewards", [
        [], {"judge": "high"}, {"judge": True}, {"judge": float("nan")}, {"judge": float("inf")},
        {"": 1.0}, {"a\nb": 1.0}, {3: 1.0}, {"x" * 257: 1.0}, {"judge": [1.0]},
    ])
    def test_malformed_rewards_are_rejected(self, rewards) -> None:
        with pytest.raises(ValueError, match="rewards"):
            _record(rewards=rewards)

    def test_validate_accepts_and_rejects_rewards(self) -> None:
        good = _record(rewards={"judge": 0.5})
        assert validate_manifest_records([good]) == [good]
        bad = dict(good, rewards={"judge": "0.5"})
        with pytest.raises(ValueError, match="rewards"):
            validate_manifest_records([bad])

    def test_buffer_reads_rewards_from_rollout_metadata(self) -> None:
        from reservoir.attest import AttestationLog
        from reservoir.rollout import Rollout
        from reservoir.rollout_buffer import RolloutBuffer

        manifest = ManifestWriter()
        buf = RolloutBuffer(capacity=4, attest=AttestationLog(), manifest=manifest)
        buf.add_group("p", 0, [
            Rollout(tokens=[1], logprobs=[-0.1], reward=1.0, metadata={"rewards": {"verifier": 1.0}}),
            Rollout(tokens=[2], logprobs=[-0.1], reward=0.0),
        ])
        first, second = manifest.records
        assert first["rewards"] == {"verifier": 1.0} and "rewards" not in second
        assert first["content_digest"] == content_digest_of("p", [1], 1.0)

    def test_malformed_metadata_rewards_fail_before_any_insert(self) -> None:
        from reservoir.attest import AttestationLog
        from reservoir.rollout import Rollout
        from reservoir.rollout_buffer import RolloutBuffer

        manifest = ManifestWriter()
        buf = RolloutBuffer(capacity=4, attest=AttestationLog(), manifest=manifest)
        with pytest.raises(ValueError, match="rewards"):
            buf.add_group("p", 0, [
                Rollout(tokens=[1], logprobs=[-0.1], reward=1.0),
                Rollout(tokens=[2], logprobs=[-0.1], reward=0.0, metadata={"rewards": {"judge": "high"}}),
            ])
        assert buf.size == 0 and manifest.records == []
