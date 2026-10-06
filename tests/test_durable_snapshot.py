"""The snapshot-only commit path: one serialisation per snapshot, same cut points, same recovery.

``durably_apply`` serialises the full state twice (the pre-state into the
intent, the post-state into the segment) and then writes it a third time
as ``state.json``. A compaction, a checkpoint restore and the first write
of a fresh directory have no operation to apply, so ``durably_snapshot``
writes the state once: the fsynced segment becomes ``state.json`` by
rename, and the intent carries no pre-state because the committed
``state.json`` already is the pre-state. Recovery must land on exactly the
old or the new snapshot from every window of that protocol.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import reservoir.durable as durable_module
from reservoir.durable import durably_snapshot, recover_state
from reservoir.durable_rollout import DurableRolloutBuffer
from tests.test_durable_wal import drive, open_buf, rollouts


def _state_file(directory: Path) -> dict:
    return json.loads((directory / "state.json").read_bytes())


class TestOneWrite:
    def test_compaction_serialises_the_state_once_and_never_through_durably_apply(self, tmp_path, monkeypatch):
        buf = open_buf(tmp_path, compact_every=1000)
        drive(buf, range(2))
        intents, segments, state_writes = [], [], []
        real_intent, real_segment = durable_module._write_intent, durable_module._write_segment
        monkeypatch.setattr(durable_module, "_write_intent", lambda d, i: (intents.append(i), real_intent(d, i)))
        monkeypatch.setattr(durable_module, "_write_segment", lambda d, k, s: (segments.append(s), real_segment(d, k, s)))
        monkeypatch.setattr(durable_module, "_save_state_file", lambda d, s: state_writes.append(s))
        import reservoir.durable_rollout as rollout_module
        assert not hasattr(rollout_module, "durably_apply")      # the buffer no longer uses the two-write path
        expected = buf.buffer.state_dict()
        buf.compact()
        assert len(segments) == 1 and len(intents) == 1 and state_writes == []
        assert intents[0] == {"op": "compact", "snapshot": True}
        assert segments[0]["buffer"] == expected and segments[0]["wal_seq"] == buf.pending_commands + buf._snapshot_seq
        assert _state_file(tmp_path / "buf") == segments[0]
        assert not list((tmp_path / "buf").glob("seg_*")) and not (tmp_path / "buf" / "intent.json").exists()
        buf.close()
        again = open_buf(tmp_path, compact_every=1000)
        assert again.buffer.state_dict() == expected
        again.close()

    def test_restore_and_fresh_open_use_the_snapshot_path_too(self, tmp_path, monkeypatch):
        names = []
        real = durable_module.durably_snapshot
        monkeypatch.setattr("reservoir.durable_rollout.durably_snapshot",
                            lambda d, name, state: (names.append(name), real(d, name, state))[1])
        buf = open_buf(tmp_path, compact_every=1000)
        drive(buf, range(1))
        buf.checkpoint("a")
        drive(buf, range(1, 2))
        buf.restore_checkpoint("a")
        assert names == ["open", "compact", "restore"]
        buf.close()


class TestRecoveryWindows:
    """Each window of the protocol, laid down by hand, must recover to the old or the new state."""

    @pytest.fixture
    def states(self, tmp_path):
        buf = open_buf(tmp_path, compact_every=1000)
        drive(buf, range(1))
        buf.compact()
        old = _state_file(tmp_path / "buf")
        drive(buf, range(1, 3))
        new = {"buffer": buf.buffer.state_dict(), "wal_seq": buf._seq, "wal_epoch": buf._epoch}
        buf.close()
        return tmp_path / "buf", old, new

    @staticmethod
    def _canonical(state: dict) -> bytes:
        return json.dumps(state, sort_keys=True, separators=(",", ":")).encode()

    def test_intent_without_a_segment_keeps_the_old_snapshot(self, states):
        directory, old, new = states
        (directory / "intent.json").write_bytes(b'{"op":"compact","snapshot":true}')
        assert recover_state(directory, strict=True) == old
        assert not (directory / "intent.json").exists()

    def test_intent_with_a_torn_segment_keeps_the_old_snapshot(self, states):
        directory, old, new = states
        (directory / "intent.json").write_bytes(b'{"op":"compact","snapshot":true}')
        (directory / "seg_00000000.json").write_bytes(self._canonical(new)[:40])
        assert recover_state(directory, strict=True) == old
        assert not (directory / "seg_00000000.json").exists()

    def test_intent_with_a_complete_segment_lands_on_the_new_snapshot(self, states):
        directory, old, new = states
        (directory / "intent.json").write_bytes(b'{"op":"compact","snapshot":true}')
        (directory / "seg_00000000.json").write_bytes(self._canonical(new))
        assert recover_state(directory, strict=True) == new
        assert _state_file(directory) == new

    def test_intent_left_after_the_rename_keeps_the_new_snapshot(self, states):
        directory, old, new = states
        (directory / "state.json").write_bytes(self._canonical(new))
        (directory / "intent.json").write_bytes(b'{"op":"compact","snapshot":true}')
        assert recover_state(directory, strict=True) == new

    def test_the_buffer_reopens_from_either_window_without_a_fresh_start(self, states):
        directory, old, new = states
        (directory / "intent.json").write_bytes(b'{"op":"compact","snapshot":true}')
        again = open_buf(directory.parent, compact_every=1000)
        assert again.buffer.state_dict() == new["buffer"]      # old snapshot plus the replayed log
        again.close()

    def test_a_fresh_directory_that_died_before_its_first_snapshot_is_fresh(self, tmp_path):
        directory = tmp_path / "buf"
        directory.mkdir()
        (directory / "intent.json").write_bytes(b'{"op":"open","snapshot":true}')
        assert recover_state(directory, strict=True) is None
        buf = DurableRolloutBuffer(directory, capacity=8, half_life=1, max_policy_age=2, seed=3)
        assert buf.size == 0
        buf.close()

    def test_a_corrupt_intent_is_still_strict_about_the_state_file(self, states):
        directory, old, new = states
        (directory / "intent.json").write_bytes(b"{not json")
        (directory / "state.json").write_bytes(b"{torn")
        with pytest.raises(durable_module.CorruptStateError):
            recover_state(directory, strict=True)


class TestDurablySnapshot:
    def test_a_failed_segment_write_leaves_no_intent_and_the_old_state(self, tmp_path, monkeypatch):
        directory = tmp_path / "d"
        directory.mkdir()
        durably_snapshot(directory, "first", {"v": 1})
        assert _state_file(directory) == {"v": 1}
        monkeypatch.setattr(durable_module, "_write_segment",
                            lambda *a: (_ for _ in ()).throw(OSError("no space")))
        with pytest.raises(OSError):
            durably_snapshot(directory, "second", {"v": 2})
        assert _state_file(directory) == {"v": 1}
        assert sorted(p.name for p in directory.iterdir()) == ["state.json"]

    def test_state_file_is_the_canonical_segment_bytes(self, tmp_path):
        directory = tmp_path / "d"
        directory.mkdir()
        durably_snapshot(directory, "x", {"b": [1, 2], "a": "é"})
        assert (directory / "state.json").read_bytes() == b'{"a":"\\u00e9","b":[1,2]}'
