"""Regression tests for the 0.6.0 review round.

One test per finding, named after it: a non-canonical integer must not
verify as the history it truncates to; a hostile ``decay_config`` or
``batch_rows`` must be refused before anything is allocated; malformed
fields must surface as ``CheckerError`` from the library, not as raw
Python errors; the crash-test kill switch needs an explicit opt-in; the
checkpoint a run resumed from is never pruned; the drift gate rejects
non-finite thresholds; per-reward-function values reach the manifest from
the trainer's own reward computation; owner-only failures reach every
rank; sharded strategies are refused; metadata is copied deeply.
"""

from __future__ import annotations

import copy
import json
import math
import warnings

import pytest
import torch

from reservoir import durable as durable_module
from reservoir.attest import AttestationLog
from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.integrations._trl_distributed import owner_step
from reservoir.integrations.trl import ReservoirReplay, _validated_gate, refuse_sharded
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer
from reservoir_checker.content import MAX_BATCH_ROWS, _int_field
from reservoir_checker.decay_replay import MAX_CAPACITY
from reservoir_checker.replay import replay as offline_replay
from reservoir_checker.verify import CheckerError, verify_chain
from tests.test_trl_replay import FakeTrainer, live_batch, mixed_batch, replay
from tests.test_trl_telemetry import MetricTrainer

KW = dict(capacity=8, half_life=1, max_policy_age=2, seed=3)


def rollouts(rewards):
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in rewards]


def rechain(records: list[dict]) -> list[dict]:
    """Recompute digests after editing a record, as a forger would."""
    from reservoir_checker.verify import _GENESIS, _digest_record

    prev = _GENESIS
    out = []
    for r in records:
        r = dict(r)
        r["prev_digest"] = prev
        r.pop("digest", None)
        r["digest"] = _digest_record(r)
        prev = r["digest"]
        out.append(r)
    return out


def small_log() -> list[dict]:
    buf = RolloutBuffer(attest=AttestationLog(), **KW)
    buf.add_group("g0", 0, rollouts([1.0, 0.0, 0.5]), source="s")
    buf.sample(2, current_version=0)
    return [dict(r) for r in buf.attestation_log.records]


# ---------------------------------------------------------------------------
# checker
# ---------------------------------------------------------------------------

class TestCheckerStrictness:
    def test_float_counter_is_refused_not_truncated(self):
        with pytest.raises(CheckerError, match="non-negative integer"):
            _int_field({"op_counter": 0.7}, "op_counter", "here")
        with pytest.raises(CheckerError):
            _int_field({"op_counter": -5}, "op_counter", "here")
        assert _int_field({"op_counter": "12"}, "op_counter", "here") == 12

    def test_float_op_counter_in_a_rechained_log_is_rejected(self):
        records = small_log()
        insert = next(i for i, r in enumerate(records) if r["op"] == "insert")
        records[insert]["op_counter"] = 0.0
        with pytest.raises(CheckerError):
            verify_chain(rechain(records))

    @pytest.mark.parametrize("edit", [
        {"root_total": None}, {"samples": 5}, {"samples": [[1]]}, {"root_total": [1]},
    ])
    def test_malformed_sample_fields_are_checker_errors(self, edit):
        records = small_log()
        sample = next(i for i, r in enumerate(records) if r["op"] == "sample")
        records[sample].update(edit)
        with pytest.raises(CheckerError):
            verify_chain(rechain(records))

    def test_truthy_non_int_priority_is_rejected(self):
        records = small_log()
        insert = next(i for i, r in enumerate(records) if r["op"] == "insert")
        records[insert]["new_priority_int"] = True
        with pytest.raises(CheckerError, match="must be an integer"):
            verify_chain(rechain(records))

    def test_capacity_beyond_the_limit_is_refused_before_allocation(self):
        records = small_log()
        records[0]["capacity"] = str(MAX_CAPACITY * 2)
        with pytest.raises(CheckerError, match="exceeds the checker's limit"):
            verify_chain(rechain(records))

    def test_batch_rows_beyond_the_limit_is_refused(self):
        r = replay(attest=AttestationLog())
        trainer = FakeTrainer(r, [live_batch(), mixed_batch()])
        trainer.generate(step=1)
        trainer.generate(step=2)
        records = [dict(x) for x in r.buffer.attestation_log.records]
        batch = next(i for i, x in enumerate(records) if x["op"] == "batch")
        records[batch]["batch_rows"] = MAX_BATCH_ROWS + 1
        with pytest.raises(CheckerError, match="batch_rows"):
            verify_chain(rechain(records))

    def test_offline_replay_refuses_a_non_list_manifest(self):
        with pytest.raises(CheckerError, match="manifest must be a list"):
            offline_replay(small_log(), 5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# durability
# ---------------------------------------------------------------------------

class TestDurability:
    def test_cut_points_need_the_crash_test_opt_in(self, monkeypatch):
        monkeypatch.setenv("RESERVOIR_CUT_POINT", "after_wal_fsync")
        monkeypatch.delenv("RESERVOIR_CRASH_TEST", raising=False)
        assert durable_module._cut_point() is None and not durable_module._should_cut("after_wal_fsync")
        monkeypatch.setenv("RESERVOIR_CRASH_TEST", "1")
        assert durable_module._should_cut("after_wal_fsync")
        monkeypatch.setenv("RESERVOIR_CUT_BYTE_OFFSET", "-3")
        with pytest.raises(ValueError, match="non-negative"):
            durable_module._cut_byte_offset()

    def test_prune_keeps_the_checkpoint_the_buffer_restored_from(self, tmp_path):
        buf = DurableRolloutBuffer(tmp_path / "buf", attest=tmp_path / "a.jsonl", **KW)
        buf.add_group("g", 0, rollouts([1.0]), source="s")
        buf.checkpoint("step-1")
        buf.add_group("h", 1, rollouts([1.0]), source="s")
        buf.checkpoint("step-2")
        buf.restore_checkpoint("step-1")
        assert buf.prune_checkpoints({"step-2"}) == []          # step-1 survives: it is what we resumed from
        assert buf.checkpoints() == ["step-1", "step-2"]
        buf.close()

    def test_durable_quarantine_refuses_coerced_positions(self, tmp_path):
        buf = DurableRolloutBuffer(tmp_path / "buf", attest=tmp_path / "a.jsonl", **KW)
        buf.add_group("g", 0, rollouts([1.0]), source="s")
        with pytest.raises(ValueError, match="plain ints"):
            buf.quarantine_positions([1.0], "p", "r")
        with pytest.raises(ValueError, match="plain ints"):
            buf.quarantine_positions([True], "p", "r")
        buf.close()


# ---------------------------------------------------------------------------
# buffer
# ---------------------------------------------------------------------------

class TestBuffer:
    def test_metadata_is_copied_deeply(self):
        rewards = {"len": 1.0}
        r = Rollout(tokens=[1], logprobs=[-0.1], reward=1.0, metadata={"rewards": rewards})
        rewards["len"] = 2.0
        assert r.metadata["rewards"]["len"] == 1.0

    def test_quarantine_without_a_log_warns(self):
        buf = RolloutBuffer(**KW)
        buf.add_group("g", 0, rollouts([1.0]), source="s")
        with pytest.warns(RuntimeWarning, match="no record of the predicate"):
            buf.quarantine(lambda rollout, group: True, "r", predicate_text="all")
        assert buf.size == 0


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------

class TestAdapter:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, 0, -1.0])
    def test_gate_threshold_must_be_finite_positive_and_not_bool(self, bad):
        with pytest.raises(ValueError, match="max_log_ratio"):
            _validated_gate(bad)

    def test_decline_cap_is_validated(self):
        with pytest.raises(ValueError, match="max_declines_per_step"):
            replay(attest=AttestationLog(), max_log_ratio=1.0, max_declines_per_step=-1)
        with pytest.raises(ValueError, match="max_declines_per_step"):
            replay(attest=AttestationLog(), max_log_ratio=1.0, max_declines_per_step=True)

    def test_per_function_rewards_reach_the_manifest_through_the_trainer(self, tmp_path):
        r = replay(attest=tmp_path / "a.jsonl", manifest=tmp_path / "m.jsonl")
        trainer = FakeTrainer(r, [live_batch()])
        batch_rows = int(live_batch()["advantages"].size(0))
        r.note_rewards_per_func(torch.arange(batch_rows * 2, dtype=torch.float32).view(batch_rows, 2), ["len", "fmt"])
        trainer.generate(step=1)
        lines = [json.loads(l) for l in (tmp_path / "m.jsonl").read_text().splitlines()]
        assert lines and all("rewards" in l for l in lines)
        assert lines[0]["rewards"] == {"len": 0.0, "fmt": 1.0}
        assert r._pending_rewards is None                       # consumed, never reused for the next batch
        r.close()

    def test_mismatched_reward_rows_fail_closed(self):
        r = replay(attest=AttestationLog())
        trainer = FakeTrainer(r, [live_batch()])
        r.note_rewards_per_func(torch.zeros(3, 1), ["len"])
        with pytest.raises(ValueError, match="reward provenance"):
            trainer.generate(step=1)

    def test_unusable_reward_tensors_record_no_provenance(self):
        r = replay(attest=AttestationLog())
        r.note_rewards_per_func(torch.zeros(4), ["len"])        # 1-D: ignored
        assert r._pending_rewards is None
        r.note_rewards_per_func(torch.zeros(4, 2), ["len"])     # names do not match columns: ignored
        assert r._pending_rewards is None

    def test_sharded_strategies_are_refused(self):
        class Accel:
            num_processes = 2
            process_index = 0
            distributed_type = "DEEPSPEED"
        with pytest.raises(RuntimeError, match="DEEPSPEED"):
            refuse_sharded(Accel(), 2)
        Accel.distributed_type = "MULTI_GPU"
        refuse_sharded(Accel(), 2)                               # DDP is fine
        Accel.distributed_type = "DEEPSPEED"
        refuse_sharded(Accel(), 1)                               # one process never shards

    def test_owner_step_reports_the_owner_s_failure_on_every_rank(self):
        class Comm:
            num_processes = 2
            process_index = 0
            sent = None

            def broadcast_object(self, obj):
                Comm.sent = obj
                return obj

            def gather_object(self, obj):
                return [obj, obj]

        owner = Comm()
        with pytest.raises(ValueError, match="boom"):
            owner_step(owner, True, lambda: (_ for _ in ()).throw(ValueError("boom")))
        assert Comm.sent == "ValueError: boom"

        class Other(Comm):
            process_index = 1

            def broadcast_object(self, obj):
                return "ValueError: boom"                       # what rank 0 sent
        with pytest.raises(RuntimeError, match="rank 0 failed.*boom"):
            owner_step(Other(), False, lambda: pytest.fail("non-owner must not run the step"))
        assert owner_step(None, True, lambda: 7) == 7

    def test_masked_nan_logprobs_do_not_poison_the_log_ratio(self):
        from reservoir.integrations._trl_telemetry import sequence_log_ratios

        output = mixed_batch(old_logps=[[-0.1], [-0.2, -0.3], [-0.4, -0.5], [-0.6]])
        width = output["old_per_token_logps"].size(1)
        masked = (output["completion_mask"] == 0).nonzero()
        assert len(masked), "the fixture needs a padded position"
        row, col = masked[0].tolist()
        output["old_per_token_logps"][row, col] = float("nan")  # a NaN under the mask
        trainer = MetricTrainer(replay(attest=AttestationLog()), [output])
        ratios = sequence_log_ratios(output, list(range(int(output["advantages"].size(0)))), trainer)
        assert len(ratios) == int(output["advantages"].size(0)) and width > 0
        assert all(math.isfinite(x) for x in ratios)
