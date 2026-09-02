"""
Tests for reservoir.durable — crash-atomic durable buffer.

Tests basic durability (insert/update/recover) and the crash campaign.
The full crash campaign (T3) is verified via campaigns/crash.py.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from reservoir.buffer import Transition
from reservoir.durable import DurableBuffer


def _make_transition(i: int) -> Transition:
    return Transition(state=i, action=0, reward=float(i), next_state=i + 1, done=False)


class TestDurableBufferBasics:
    def test_create_and_insert(self, tmp_path):
        buf = DurableBuffer(tmp_path, capacity=4, seed=0)
        buf.insert(_make_transition(0), td_error=1.0)
        assert buf.size == 1

    def test_state_persists_across_reopen(self, tmp_path):
        buf = DurableBuffer(tmp_path, capacity=4, seed=0)
        buf.insert(_make_transition(0), td_error=1.0)
        buf.insert(_make_transition(1), td_error=2.0)

        # Reopen
        buf2 = DurableBuffer(tmp_path, capacity=4, seed=0)
        assert buf2.size == 2

    def test_priorities_persist(self, tmp_path):
        buf = DurableBuffer(tmp_path, capacity=4, seed=0)
        buf.insert(_make_transition(0), td_error=1.0)
        p_before = buf._buf._sum_tree.get(0)

        buf2 = DurableBuffer(tmp_path, capacity=4, seed=0)
        p_after = buf2._buf._sum_tree.get(0)
        assert p_before == p_after

    def test_update_priority_persists(self, tmp_path):
        buf = DurableBuffer(tmp_path, capacity=4, seed=0)
        buf.insert(_make_transition(0), td_error=1.0)
        buf.update_priority(0, td_error=9.9)
        p_before = buf._buf._sum_tree.get(0)

        buf2 = DurableBuffer(tmp_path, capacity=4, seed=0)
        p_after = buf2._buf._sum_tree.get(0)
        assert p_before == p_after

    def test_sample_after_reopen(self, tmp_path):
        buf = DurableBuffer(tmp_path, capacity=4, seed=0)
        for i in range(4):
            buf.insert(_make_transition(i), td_error=float(i + 1))

        buf2 = DurableBuffer(tmp_path, capacity=4, seed=0)
        batch = buf2.sample(2)
        assert len(batch.indices) == 2

    def test_no_intent_on_clean_state(self, tmp_path):
        buf = DurableBuffer(tmp_path, capacity=4, seed=0)
        buf.insert(_make_transition(0), td_error=1.0)
        # After insert, intent file should be cleaned up
        assert not (tmp_path / "intent.json").exists()
        assert not (tmp_path / "intent.json.tmp").exists()

    def test_full_capacity_wrap(self, tmp_path):
        buf = DurableBuffer(tmp_path, capacity=4, seed=0)
        for i in range(8):  # Twice capacity
            buf.insert(_make_transition(i), td_error=float(i + 1))
        assert buf.size == 4  # Stays at capacity

        buf2 = DurableBuffer(tmp_path, capacity=4, seed=0)
        assert buf2.size == 4


class TestCrashRecovery:
    """Verify crash recovery logic: pre-state and post-state recovery."""

    def test_recovery_from_incomplete_intent(self, tmp_path):
        """If intent.json.tmp exists but no seg_0.json, recover to pre-state."""
        # Set up a buffer with one transition
        buf = DurableBuffer(tmp_path, capacity=4, seed=0)
        buf.insert(_make_transition(0), td_error=1.0)
        pre_size = buf.size

        # Manually create intent.json.tmp without completing the operation
        intent = {"op": "insert", "pre_state": None}
        # Write a pre_state that's the current state
        state_file = tmp_path / "state.json"
        if state_file.exists():
            current_state = json.loads(state_file.read_bytes())
        else:
            current_state = None

        intent["pre_state"] = current_state
        (tmp_path / "intent.json.tmp").write_text(
            json.dumps(intent, sort_keys=True)
        )

        # Recovery: no seg_0.json exists, so should restore pre_state
        buf2 = DurableBuffer(tmp_path, capacity=4, seed=0)
        # Size should match pre_state
        if current_state:
            assert buf2.size == current_state.get("size", pre_size)
        else:
            assert buf2.size == pre_size

    def test_recovery_from_committed_intent(self, tmp_path):
        """If intent.json + seg_0.json exist, recover to post-state."""
        # Set up buffer
        buf = DurableBuffer(tmp_path, capacity=4, seed=0)
        buf.insert(_make_transition(0), td_error=1.0)

        # Simulate a committed-but-not-cleaned-up state:
        # manually create intent.json + seg_0.json
        from src.reservoir.durable import _segment_path, _serialize_state

        post_size = 2
        post_state = _serialize_state(
            size=post_size,
            write_pos=2,
            op_counter=5,
            max_prio=buf._buf._max_prio,
            priorities=[buf._buf._sum_tree.get(i) for i in range(4)],
            transitions=[
                {"state": 0, "action": 0, "reward": 0.0, "next_state": 1, "done": False},
                {"state": 1, "action": 0, "reward": 1.0, "next_state": 2, "done": False},
                None,
                None,
            ],
        )

        # Write intent.json and seg_0.json
        intent = {"op": "insert", "pre_state": None}
        (tmp_path / "intent.json").write_text(json.dumps(intent))
        seg_path = _segment_path(tmp_path, 0)
        seg_path.write_text(json.dumps(post_state, sort_keys=True))

        # Recovery: should apply post_state
        buf2 = DurableBuffer(tmp_path, capacity=4, seed=0)
        assert buf2.size == post_size

    def test_no_torn_state_after_recovery(self, tmp_path):
        """After recovery, the buffer's tree invariants hold."""
        buf = DurableBuffer(tmp_path, capacity=4, seed=0)
        for i in range(4):
            buf.insert(_make_transition(i), td_error=float(i + 1))

        buf2 = DurableBuffer(tmp_path, capacity=4, seed=0)
        assert buf2._buf.verify_trees()


class TestFsyncDiscipline:
    def test_no_intent_file_after_clean_insert(self, tmp_path):
        """After a successful insert, no intent files should remain."""
        buf = DurableBuffer(tmp_path, capacity=4, seed=0)
        buf.insert(_make_transition(0), td_error=1.0)

        assert not (tmp_path / "intent.json").exists()
        assert not (tmp_path / "intent.json.tmp").exists()
        assert not (tmp_path / "seg_00000000.json").exists()

    def test_state_file_written_after_insert(self, tmp_path):
        """state.json should exist after each operation."""
        buf = DurableBuffer(tmp_path, capacity=4, seed=0)
        buf.insert(_make_transition(0), td_error=1.0)

        assert (tmp_path / "state.json").exists()
        state = json.loads((tmp_path / "state.json").read_bytes())
        assert state["size"] == 1
