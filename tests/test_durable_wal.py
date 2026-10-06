"""Tests for the durable buffer's command log.

Each operation is appended to ``wal.jsonl`` after it is applied in memory
and before its result is returned; recovery replays the log onto the last
snapshot and must reproduce the state, the attestation chain and the
manifest exactly. Snapshots are taken every ``compact_every`` commands,
checkpoints copy a snapshot aside and can be restored, a torn tail is cut,
a failed operation is rebuilt from disk, and per-operation cost does not
grow with history.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import reservoir.rollout_wal as wal_module

from reservoir.attest import AttestationLog
from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer
from reservoir.rollout_wal import CommandLog, WAL_FILE
from reservoir_checker.verify import verify_json_lines

KW = dict(capacity=8, half_life=1, max_policy_age=2, seed=3)


def rollouts(rewards, **meta):
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r, metadata=meta) for r in rewards]


def drive(buf, versions, witness: bool = True):
    for v in versions:
        buf.add_group(f"g{v}", v, rollouts([1.0, 0.0, 0.5]), source="s")
        batch = buf.sample(2, current_version=v)
        if witness:
            buf.witness_batch(batch, step=v, batch_rows=4, rows=[2, 3], tensor_digest="ab" * 32)
            buf.record_telemetry(v, dict(batch_rows=4, replaced_rows=2, declined_rows=0, dead_groups=1,
                                         near_dead_groups=0), batch, {"log_ratio_mean_abs": 0.1})
        buf.update_priorities(batch.indices[:1], [0.4])


def open_buf(tmp_path: Path, **kw) -> DurableRolloutBuffer:
    params = dict(KW, **kw)
    return DurableRolloutBuffer(tmp_path / "buf", attest=tmp_path / "attest.jsonl",
                                manifest=tmp_path / "manifest.jsonl", **params)


class TestReplay:
    def test_reopen_replays_the_log_onto_the_snapshot(self, tmp_path):
        buf = open_buf(tmp_path, compact_every=1000)
        drive(buf, range(7))
        assert buf.pending_commands > 0 and buf.n_rebases >= 1
        state, head, manifest = buf.state_dict(), buf.attestation_log.head_digest, buf.manifest_records
        twin = RolloutBuffer(attest=AttestationLog(), **KW)
        twin.load_state_dict({**state, "manifest": None})
        buf.close()

        again = open_buf(tmp_path, compact_every=1000)
        assert again.state_dict() == state
        assert again.attestation_log.head_digest == head and again.manifest_records == manifest
        assert again.sample(3, current_version=7).draw_integers == twin.sample(3, current_version=7).draw_integers
        again.close()
        verify_json_lines((tmp_path / "attest.jsonl").read_text(), manifest=(tmp_path / "manifest.jsonl").read_text())

    def test_compaction_resets_the_log_and_keeps_the_state(self, tmp_path):
        buf = open_buf(tmp_path, compact_every=5)
        drive(buf, range(6))
        assert buf.pending_commands <= 5                      # compaction runs before the next command
        snapshot = json.loads((tmp_path / "buf" / "state.json").read_text())
        assert snapshot["wal_seq"] >= 5 and snapshot["buffer"]["op_counter"] >= 1
        state = buf.state_dict()
        buf.close()
        again = open_buf(tmp_path, compact_every=5)
        assert again.state_dict() == state
        again.close()

    def test_torn_tail_is_cut_and_the_rest_replayed(self, tmp_path):
        buf = open_buf(tmp_path, compact_every=1000)
        drive(buf, range(3))
        buf.add_group("extra", 3, rollouts([1.0]), source="s")
        state_after = buf.state_dict()
        buf.close()
        wal = tmp_path / "buf" / WAL_FILE
        raw = wal.read_bytes()
        wal.write_bytes(raw + b'{"seq": 99, "op": "advance", "args": {"current_version": 9}')   # no newline, no digest
        again = open_buf(tmp_path, compact_every=1000)
        assert again.state_dict() == state_after
        assert wal.read_bytes() == raw
        again.close()

    def test_command_with_bad_digest_stops_the_replay(self, tmp_path):
        buf = open_buf(tmp_path, compact_every=1000)
        drive(buf, range(2))
        buf.close()
        wal = tmp_path / "buf" / WAL_FILE
        lines = wal.read_text().splitlines()
        tampered = json.loads(lines[-1]); tampered["args"]["raw_scores"] = [0.9]
        lines[-1] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
        wal.write_text("\n".join(lines) + "\n")
        with pytest.warns(RuntimeWarning, match="discarding a complete last line"):
            again = open_buf(tmp_path, compact_every=1000)
        assert again.pending_commands == len(lines) - 1          # the tampered last line was dropped
        assert len(wal.read_text().splitlines()) == len(lines) - 1
        again.close()

    def test_legacy_full_state_snapshot_opens(self, tmp_path):
        plain = RolloutBuffer(attest=tmp_path / "attest.jsonl", manifest=tmp_path / "manifest.jsonl", **KW)
        drive(plain, range(3), witness=False)
        legacy = plain.state_dict()
        plain.close()
        (tmp_path / "buf").mkdir()
        (tmp_path / "buf" / "state.json").write_text(json.dumps(legacy))
        buf = open_buf(tmp_path)
        assert buf.state_dict() == legacy and buf.pending_commands == 0
        buf.add_group("after", 3, rollouts([1.0]), source="s")
        assert buf.pending_commands == 1
        buf.close()

    def test_failed_operation_is_rebuilt_from_disk(self, tmp_path):
        buf = open_buf(tmp_path)
        drive(buf, range(2))
        before = buf.state_dict()
        with pytest.raises(ValueError):
            buf.add_group("bad", 1, rollouts([1.0]), source="a\nb")
        assert buf.state_dict() == before and buf.verify_trees()
        with pytest.raises(RuntimeError):
            buf.sample(1, current_version=50)           # advances, evicts everything, then cannot draw
        assert buf.state_dict() == before
        buf.close()

    def test_gap_in_the_log_is_an_error(self, tmp_path):
        buf = open_buf(tmp_path, compact_every=1000)
        drive(buf, range(2))
        buf.close()
        wal = tmp_path / "buf" / WAL_FILE
        lines = wal.read_text().splitlines()
        wal.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
        with pytest.raises(ValueError, match="was expected"):
            open_buf(tmp_path, compact_every=1000)

    def test_damage_before_the_end_is_an_error_not_a_torn_tail(self, tmp_path):
        buf = open_buf(tmp_path, compact_every=1000)
        drive(buf, range(2))
        buf.close()
        wal = tmp_path / "buf" / WAL_FILE
        lines = wal.read_text().splitlines()
        lines[1] = lines[1][:-5] + '"}'
        wal.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="damaged line"):
            open_buf(tmp_path, compact_every=1000)
        assert wal.read_text().splitlines() == lines      # nothing was truncated


class TestFailures:
    def test_failed_append_rebuilds_memory_and_cuts_the_partial_line(self, tmp_path, monkeypatch):
        buf = open_buf(tmp_path, compact_every=1000)
        drive(buf, range(2))
        before, head = buf.state_dict(), buf.attestation_log.head_digest
        logged = buf.wal_bytes

        real_fsync, failures = wal_module._full_fsync, []

        def failing_once(fd):
            if not failures:
                failures.append(fd)
                raise OSError("disk full")
            real_fsync(fd)
        monkeypatch.setattr(wal_module, "_full_fsync", failing_once)
        with pytest.raises(OSError, match="disk full"):
            buf.add_group("lost", 2, rollouts([1.0]), source="s")
        monkeypatch.undo()
        assert buf.state_dict() == before and buf.attestation_log.head_digest == head
        assert buf.wal_bytes == logged
        buf.add_group("kept", 2, rollouts([1.0]), source="s")
        state = buf.state_dict()
        buf.close()
        again = open_buf(tmp_path, compact_every=1000)
        assert again.state_dict() == state
        again.close()

    def test_unretractable_append_failure_closes_the_buffer(self, tmp_path, monkeypatch):
        buf = open_buf(tmp_path, compact_every=1000)
        drive(buf, range(1))
        monkeypatch.setattr(wal_module, "_full_fsync", lambda fd: (_ for _ in ()).throw(OSError("dead disk")))
        with pytest.raises(RuntimeError, match="could not be retracted"):
            buf.add_group("lost", 1, rollouts([1.0]), source="s")

    def test_compaction_failure_surfaces_before_the_next_operation_is_applied(self, tmp_path, monkeypatch):
        buf = open_buf(tmp_path, compact_every=5)
        drive(buf, range(1))                                     # 5 commands: compaction is due
        assert buf.pending_commands == 5
        before = buf.state_dict()
        import reservoir.durable_rollout as durable_module
        monkeypatch.setattr(durable_module, "durably_snapshot", lambda *a, **k: (_ for _ in ()).throw(OSError("no space")))
        with pytest.raises(OSError, match="no space"):
            buf.add_group("x", 1, rollouts([1.0]), source="s")
        monkeypatch.undo()
        assert buf.state_dict() == before                        # the operation was never applied
        buf.add_group("x", 1, rollouts([1.0]), source="s")       # compaction then the op, both fine
        assert buf.pending_commands == 1
        buf.close()


class TestCheckpoints:
    def test_checkpoint_and_restore_rewind_the_chain(self, tmp_path):
        buf = open_buf(tmp_path, compact_every=1000)
        drive(buf, range(3))
        target = buf.checkpoint("step-3")
        assert (target / "state.json").exists() and buf.checkpoints() == ["step-3"]
        head_at_3, state_at_3 = buf.attestation_log.head_digest, buf.state_dict()
        drive(buf, range(3, 6))
        assert buf.attestation_log.head_digest != head_at_3
        buf.restore_checkpoint("step-3")
        assert buf.state_dict() == state_at_3 and buf.attestation_log.head_digest == head_at_3
        assert (tmp_path / "attest.jsonl").read_text().count("\n") == len(buf.attestation_log.records)
        # Training continues from the checkpoint: the same next draw as a fresh replay would give.
        drive(buf, range(3, 4))
        buf.close()
        verify_json_lines((tmp_path / "attest.jsonl").read_text(), manifest=(tmp_path / "manifest.jsonl").read_text())

    def test_crash_between_restore_snapshot_and_log_reset_does_not_replay_the_abandoned_timeline(self, tmp_path, monkeypatch):
        buf = open_buf(tmp_path, compact_every=1000)
        drive(buf, range(2))
        buf.checkpoint("a")
        state_a = buf.state_dict()
        drive(buf, range(2, 5))                                  # abandoned later
        import reservoir.rollout_wal as wal

        def no_reset(self):
            raise KeyboardInterrupt                              # process dies right here
        monkeypatch.setattr(wal.CommandLog, "reset", no_reset)
        with pytest.raises(KeyboardInterrupt):
            buf.restore_checkpoint("a")
        monkeypatch.undo()
        again = open_buf(tmp_path, compact_every=1000)
        assert again.state_dict() == state_a and again.pending_commands == 0
        drive(again, range(2, 3))                                # new epoch appends after the stale lines
        state = again.state_dict()
        again.close()
        third = open_buf(tmp_path, compact_every=1000)
        assert third.state_dict() == state
        third.close()

    def test_restore_rejects_tags_that_leave_the_directory(self, tmp_path):
        buf = open_buf(tmp_path)
        with pytest.raises(ValueError):
            buf.restore_checkpoint("../state")
        buf.close()

    def test_restore_survives_reopen_and_rejects_unknown_tags(self, tmp_path):
        buf = open_buf(tmp_path)
        drive(buf, range(2))
        buf.checkpoint("a")
        drive(buf, range(2, 4))
        buf.restore_checkpoint("a")
        state = buf.state_dict()
        buf.close()
        again = open_buf(tmp_path)
        assert again.state_dict() == state
        with pytest.raises(FileNotFoundError):
            again.restore_checkpoint("nope")
        with pytest.raises(ValueError):
            again.checkpoint("../escape")
        again.close()


class TestCost:
    def test_log_bytes_per_operation_do_not_grow_with_history(self, tmp_path):
        buf = open_buf(tmp_path, compact_every=10_000, max_policy_age=8, half_life=4, capacity=64)
        sizes = []
        for v in range(60):
            before = buf.wal_bytes
            buf.add_group(f"g{v}", v // 8, rollouts([1.0, 0.0, 0.5]), source="s")
            sizes.append(buf.wal_bytes - before)
        early, late = sum(sizes[:10]) / 10, sum(sizes[-10:]) / 10
        assert late <= early * 1.1      # the command, not the state, is what gets written
        buf.close()


class TestCommandLogUnit:
    def test_read_skips_commands_the_snapshot_includes(self, tmp_path):
        log = CommandLog(tmp_path)
        for seq in range(1, 5):
            log.append(seq, "advance", {"current_version": seq})
        commands, end = log.read_commands(after_seq=2)
        assert [c["seq"] for c in commands] == [3, 4] and end == log.size_bytes()
        log.reset()
        assert log.read_commands(after_seq=0) == ([], 0)

    def test_lines_of_another_epoch_are_ignored_whatever_their_seq(self, tmp_path):
        log = CommandLog(tmp_path)
        for seq in range(1, 4):
            log.append(seq, "advance", {"current_version": seq}, epoch=0)
        log.append(1, "advance", {"current_version": 9}, epoch=1)
        commands, end = log.read_commands(after_seq=0, epoch=1)
        assert [c["seq"] for c in commands] == [1] and end == log.size_bytes()
        assert [c["seq"] for c in log.read_commands(after_seq=0, epoch=0)[0]] == [1, 2, 3]


class TestWitnessReplayBinding:
    def test_last_batch_is_none_once_a_sampled_slot_is_refilled(self):
        buf = RolloutBuffer(attest=AttestationLog(), **KW)
        buf.add_group("g", 0, rollouts([1.0]), source="s")
        batch = buf.sample(1, current_version=0)
        assert buf.last_batch is not None and buf.last_batch.op_counter == batch.op_counter
        buf.evict(batch.indices[0])
        buf.add_group("h", 0, rollouts([1.0]), source="s")
        assert buf.last_batch is None

    def test_last_batch_keeps_sample_time_versions(self):
        buf = RolloutBuffer(attest=AttestationLog(), **KW)
        buf.add_group("g", 0, rollouts([1.0]), source="s")
        batch = buf.sample(1, current_version=0)
        buf.advance(1)
        assert buf.last_batch.model_versions == batch.model_versions

    def test_replayed_witness_must_name_the_live_sample(self):
        from reservoir.rollout_wal import apply_command
        buf = RolloutBuffer(attest=AttestationLog(), **KW)
        buf.add_group("g", 0, rollouts([1.0]), source="s")
        batch = buf.sample(1, current_version=0)
        bad = {"op": "witness_batch", "args": {"step": 0, "batch_rows": 1, "rows": [0], "tensor_digest": "ab" * 32,
                                               "declined": [], "sample_op": batch.op_counter + 1}}
        with pytest.raises(ValueError, match="refers to sample"):
            apply_command(buf, bad)
        buf.evict(batch.indices[0])
        gone = {"op": "record_telemetry", "args": {"step": 0, "counts": {}, "with_sample": True,
                                                   "sample_op": batch.op_counter, "reported": {}}}
        with pytest.raises(ValueError, match="gone"):
            apply_command(buf, gone)


class TestCheckpointBinding:
    def test_binding_is_stored_beside_the_checkpoint_and_pruned_with_it(self, tmp_path):
        buf = open_buf(tmp_path, compact_every=1000)
        drive(buf, range(1))
        buf.checkpoint("plain")
        buf.checkpoint("bound", binding={"model_checkpoint": {"name": "checkpoint-1", "digest": "ab" * 32}})
        assert buf.checkpoint_binding("plain") is None
        assert buf.checkpoint_binding("bound") == {"model_checkpoint": {"name": "checkpoint-1", "digest": "ab" * 32}}
        assert (tmp_path / "buf" / "checkpoints" / "bound" / "binding.json").exists()
        buf.checkpoint("bound")                                        # re-taken without a binding: it is dropped
        assert buf.checkpoint_binding("bound") is None
        with pytest.raises(TypeError):
            buf.checkpoint("x", binding=["not", "a", "dict"])
        with pytest.raises(FileNotFoundError):
            buf.checkpoint_binding("missing")
        assert buf.prune_checkpoints({"plain"}) == ["bound"]
        buf.close()

    def test_an_unreadable_binding_is_an_error_not_none(self, tmp_path):
        buf = open_buf(tmp_path, compact_every=1000)
        drive(buf, range(1))
        buf.checkpoint("a", binding={"k": 1})
        (tmp_path / "buf" / "checkpoints" / "a" / "binding.json").write_bytes(b"[1]")
        with pytest.raises(ValueError, match="JSON object"):
            buf.checkpoint_binding("a")
        (tmp_path / "buf" / "checkpoints" / "a" / "binding.json").write_bytes(b"{nope")
        with pytest.raises(ValueError, match="unreadable"):
            buf.checkpoint_binding("a")
        buf.close()

    def test_a_retaken_tag_never_keeps_the_old_binding(self, tmp_path, monkeypatch):
        buf = open_buf(tmp_path, compact_every=1000)
        drive(buf, range(1))
        buf.checkpoint("a", binding={"v": 1})
        target = tmp_path / "buf" / "checkpoints" / "a"
        order = []
        real_copy = buf._write_binding

        def record(target_dir, binding):
            order.append(("binding", binding, (target_dir / "state.json.tmp").exists()))
            real_copy(target_dir, binding)

        monkeypatch.setattr(buf, "_write_binding", record)
        buf.checkpoint("a", binding={"v": 2})
        # The stale binding is removed before the new state is staged; the new one is written after it.
        assert order == [("binding", None, False), ("binding", {"v": 2}, False)]
        assert buf.checkpoint_binding("a") == {"v": 2}
        monkeypatch.undo()
        # A crash after the state was replaced but before the new binding: the checkpoint is unbound.
        monkeypatch.setattr(buf, "_write_binding",
                            lambda t, b: real_copy(t, b) if b is None else (_ for _ in ()).throw(OSError("crash")))
        with pytest.raises(OSError):
            buf.checkpoint("a", binding={"v": 3})
        assert buf.checkpoint_binding("a") is None and (target / "state.json").exists()
        buf.close()
