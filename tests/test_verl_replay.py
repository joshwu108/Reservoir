"""Tests for ``reservoir.integrations.verl``: the replay adapter for verl's RayPPOTrainer.

Nothing here imports verl. ``FakeTrainer`` combines the adapter's
``ReservoirReplayMixin`` with a stand-in base class shaped like the
``DataProto`` trainer: ``fit`` drives scripted batches through
``_update_actor`` the way verl does (``global_steps`` is 1 during the first
step and is incremented after each), ``_compute_old_log_prob`` returns a
``DataProto`` of constant logprobs, ``_save_checkpoint`` writes
``global_step_N`` directories and ``_load_checkpoint`` reads the latest one
back. ``FakeDataProto`` has the four members the adapter uses
(``batch``, ``non_tensor_batch``, ``meta_info``, ``from_dict``). That is the
interface verl presents to the override, so the tests exercise the real
mixin end to end.

Parity tests compare what the actor receives with a plain
``RolloutBuffer`` driven by the same ``add_group``/``sample`` sequence with
the same seed: the adapter must add no sampling logic of its own.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from checker.verify import verify_json_lines
from reservoir.attest import AttestationLog
from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.integrations._verl_rows import rows_to_groups, tensor_digest
from reservoir.integrations.trl import StoredAdvantagePriority
from reservoir.integrations.verl import (
    ReservoirReplay,
    ReservoirReplayMixin,
    append_metrics,
    assert_hook_ran,
    bind_checkpoint,
    config_value,
    trainer_checkpoint_steps,
)
from reservoir.rollout_buffer import RolloutBuffer
from tests.test_verl_rows import PAD, f32, make_batch, verl_position_ids

LOGP = -0.5  # value the fake old-logprob forward returns for every token


class FakeDataProto:
    """The four members of ``verl.protocol.DataProto`` the adapter touches."""

    def __init__(self, batch: dict, non_tensor_batch: dict, meta_info: dict) -> None:
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    @classmethod
    def from_dict(cls, tensors=None, non_tensors=None, meta_info=None):
        tensors = dict(tensors or {})
        sizes = {t.size(0) for t in tensors.values()}
        assert len(sizes) <= 1, "from_dict requires one batch size"
        non_tensors = {k: (v if isinstance(v, np.ndarray) else np.array(v, dtype=object)) for k, v in (non_tensors or {}).items()}
        return cls(tensors, non_tensors, dict(meta_info or {}))

    def __len__(self) -> int:
        return next(iter(self.batch.values())).size(0)


def proto(batch: dict, uids: list[str], **meta) -> FakeDataProto:
    meta_info = {"global_token_num": batch["attention_mask"].sum(dim=-1).tolist(), "temperature": 1.0, **meta}
    return FakeDataProto(batch, {"uid": np.array(uids, dtype=object), "data_source": np.array(["ds"] * len(uids), dtype=object)}, meta_info)


def live_batch(prompt_base: int = 10, **kwargs) -> FakeDataProto:
    """Two live groups of 2 rows: advantages (+1, -1) and (+0.5, -0.5)."""
    batch, uids = make_batch(
        prompts=[[prompt_base, 1], [prompt_base, 1], [prompt_base + 1], [prompt_base + 1]],
        responses=[[3, 4, 5], [6], [7, 8], [9, 9, 9]],
        advantages=[1.0, -1.0, 0.5, -0.5],
        **kwargs,
    )
    return proto(batch, uids)


def mixed_batch(**kwargs) -> FakeDataProto:
    """One live group (rows 0-1) and one dead group (rows 2-3)."""
    batch, uids = make_batch(
        prompts=[[20, 21], [20, 21], [22], [22]],
        responses=[[3], [4, 4], [5, 5], [5]],
        advantages=[0.25, -0.25, 0.0, 0.0],
        logps=[[-0.1], [-0.2, -0.3], [-0.4, -0.5], [-0.6]],
        **kwargs,
    )
    return proto(batch, uids)


def dead_batch() -> FakeDataProto:
    batch, uids = make_batch(prompts=[[1], [1]], responses=[[2], [3]], advantages=[0.0, 0.0])
    return proto(batch, uids)


def fake_config(**overrides) -> SimpleNamespace:
    values = {
        "algorithm.adv_estimator": "grpo",
        "algorithm.rollout_correction.rollout_is": None,
        "algorithm.rollout_correction.rollout_rs": None,
        "algorithm.rollout_correction.bypass_mode": False,
        "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
        "trainer.default_local_dir": None,
    }
    values.update(overrides)
    root: dict = {}
    for path, value in values.items():
        node = root
        *parents, leaf = path.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value

    def build(d):
        return SimpleNamespace(**{k: build(v) if isinstance(v, dict) else v for k, v in d.items()})

    return build(root)


class FakeBase:
    """Stands in for ``RayPPOTrainer``: the four methods the mixin overrides, shaped like verl's."""

    def __init__(self, batches: list[FakeDataProto], config: SimpleNamespace, ckpt_dir=None) -> None:
        self._batches = list(batches)
        self.config = config
        if ckpt_dir is not None:
            self.config.trainer.default_local_dir = str(ckpt_dir)
        self.global_steps = 0
        self.tokenizer = SimpleNamespace(pad_token_id=PAD)
        self.actor_inputs: list[FakeDataProto] = []
        self.logprob_calls: list[FakeDataProto] = []
        self.saved: list[int] = []

    def _update_actor(self, batch):
        self.actor_inputs.append(batch)
        return FakeDataProto({}, {}, {"metrics": {"actor/pg_loss": 0.1}})

    def _compute_old_log_prob(self, batch):
        self.logprob_calls.append(batch)
        n, lr = batch.batch["response_mask"].shape
        out = FakeDataProto({"old_log_probs": torch.full((n, lr), LOGP), "entropys": torch.zeros(n, lr)}, {}, {})
        return out, 0.0

    def _save_checkpoint(self):
        root = self.config.trainer.default_local_dir
        if root is not None:
            import pathlib

            (pathlib.Path(root) / f"global_step_{self.global_steps}").mkdir(parents=True, exist_ok=True)
        self.saved.append(self.global_steps)

    def _load_checkpoint(self):
        steps = trainer_checkpoint_steps(self.config.trainer.default_local_dir)
        if not steps:
            return 0
        self.global_steps = max(steps)

    def fit(self, save_freq: int = 0):
        """verl's loop shape: load, start at step 1, update, maybe save, increment."""
        self.global_steps = 0
        self._load_checkpoint()
        self.global_steps += 1
        outputs = []
        while self._batches:
            batch = self._batches.pop(0)
            self._compute_old_log_prob(batch)
            outputs.append(self._update_actor(batch))
            if save_freq and self.global_steps % save_freq == 0:
                self._save_checkpoint()
            self.global_steps += 1
        return outputs


class FakeTrainer(ReservoirReplayMixin, FakeBase):
    """The adapter's mixin over ``FakeBase``."""

    def __init__(self, replay: ReservoirReplay, batches: list[FakeDataProto], *, config=None, ckpt_dir=None) -> None:
        super().__init__(batches, config or fake_config(), ckpt_dir)
        self.replay_buffer = replay

    def step(self, batch: FakeDataProto, step: int) -> FakeDataProto:
        """One training step at ``global_steps == step``; returns what the actor received."""
        self.global_steps = step
        self._update_actor(batch)
        return self.actor_inputs[-1]


def replay(**kwargs) -> ReservoirReplay:
    defaults = dict(capacity=16, half_life=4, max_policy_age=8, seed=0)
    defaults.update(kwargs)
    return ReservoirReplay(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_defaults_wrap_an_in_memory_rollout_buffer():
    r = replay()
    assert isinstance(r.buffer, RolloutBuffer)
    assert isinstance(r.buffer.priority, StoredAdvantagePriority)
    assert r.buffer.params.half_life == 4 and r.buffer.params.max_policy_age == 8
    assert r.is_owner and r.last_replay is None and r.last_telemetry is None
    assert all(v == 0 for v in r.stats.values())


def test_directory_makes_the_buffer_durable_and_manifest_requires_attest(tmp_path):
    r = replay(directory=tmp_path / "buf", attest=tmp_path / "attest.jsonl")
    assert isinstance(r.buffer, DurableRolloutBuffer)
    r.close()
    with pytest.raises(ValueError, match="manifest"):
        replay(manifest=tmp_path / "m.jsonl")


def test_gate_threshold_is_validated():
    with pytest.raises(ValueError, match="max_log_ratio"):
        replay(max_log_ratio=-1.0)


def test_config_value_reads_namespaces_and_mappings():
    cfg = fake_config()
    assert config_value(cfg, "algorithm.adv_estimator") == "grpo"
    assert config_value(cfg, "algorithm.missing.deeper", "d") == "d"
    assert config_value({"a": {"b": 1}}, "a.b") == 1
    assert config_value(None, "a.b", 7) == 7


# ---------------------------------------------------------------------------
# Ingest and the seam
# ---------------------------------------------------------------------------

def test_update_actor_stores_live_rows_and_replays_the_dead_group():
    r = replay()
    batch = mixed_batch()
    trainer = FakeTrainer(r, [])

    out = trainer.step(batch, step=3)

    assert out is not batch
    assert r.buffer.size == 2 and r.buffer.current_version == 3
    stored = {r.buffer.entry(p)[0].metadata["row"]: r.buffer.entry(p) for p in r.buffer.live_positions()}
    rollout0, group0 = stored[0]
    assert rollout0.tokens == (3,) and list(rollout0.logprobs) == f32([-0.1]) and rollout0.reward == 0.25
    assert rollout0.metadata["uid"] == "u0" and rollout0.metadata["global_step"] == 3
    assert group0.model_version == 3 and group0.size == 2
    assert r.stats["hook_calls"] == 1 and r.stats["ingested_rows"] == 2 and r.stats["dead_groups"] == 1
    assert r.stats["replaced_rows"] == 2 and r.stats["telemetry_forwards"] == 1
    # The telemetry forward saw exactly the two replayed rows, with the keys the converter asserts.
    (call,) = trainer.logprob_calls
    assert len(call) == 2 and {"input_ids", "attention_mask", "response_mask", "position_ids"} <= set(call.batch)
    assert call.meta_info["temperature"] == 1.0


def test_a_batch_without_dead_groups_goes_to_the_actor_unchanged():
    r = replay()
    batch = live_batch()
    trainer = FakeTrainer(r, [])
    out = trainer.step(batch, step=1)
    assert out is batch
    assert r.buffer.size == 4 and r.stats["replaced_rows"] == 0 and trainer.logprob_calls == []
    assert r.last_telemetry is not None and r.last_telemetry.replaced_rows == 0


def test_dead_groups_stay_dead_while_the_buffer_is_empty():
    r = replay()
    batch = dead_batch()
    out = FakeTrainer(r, []).step(batch, step=1)
    assert out is batch and r.buffer.size == 0 and r.stats["dead_groups"] == 1 and r.last_replay is None


def test_a_step_below_the_buffer_version_is_refused_before_anything_is_stored():
    r = replay()
    trainer = FakeTrainer(r, [])
    trainer.step(live_batch(), step=5)
    stored = r.buffer.size
    with pytest.raises(ValueError, match="global_steps 3 is below"):
        trainer.step(mixed_batch(), step=3)
    assert r.buffer.size == stored and r.stats["hook_calls"] == 1


def test_a_buffer_that_went_fully_stale_is_advanced_then_left_alone():
    r = replay(max_policy_age=2)
    trainer = FakeTrainer(r, [])
    trainer.step(live_batch(), step=1)
    dead = dead_batch()
    assert trainer.step(dead, step=10) is dead
    assert r.buffer.size == 0 and r.buffer.current_version == 10


def test_reservoir_metrics_ride_on_the_actor_output():
    r = replay()
    trainer = FakeTrainer(r, [])
    trainer.step(live_batch(), step=1)
    trainer.step(mixed_batch(), step=2)
    metrics = trainer._update_actor(mixed_batch()).meta_info["metrics"]
    assert metrics["actor/pg_loss"] == 0.1
    assert metrics["reservoir/replaced_rows"] == 2.0 and metrics["reservoir/replay_fraction"] == 0.5
    assert metrics["reservoir/dead_groups"] == 1.0 and "reservoir/ess" in metrics
    assert metrics["reservoir/log_ratio_mean_abs"] >= 0.0


def test_append_metrics_is_a_no_op_without_telemetry_and_an_error_without_a_metrics_dict():
    append_metrics(None, None)
    r = replay()
    FakeTrainer(r, []).step(live_batch(), step=1)
    assert r.last_telemetry is not None
    with pytest.raises(RuntimeError, match="meta_info\\['metrics'\\]"):
        append_metrics(FakeDataProto({}, {}, {"metrics": "not a dict"}), r.last_telemetry)
    with pytest.raises(RuntimeError):
        append_metrics(FakeDataProto({}, {}, {}), r.last_telemetry)


def test_the_telemetry_forward_is_padded_to_the_worker_groups_world_size():
    r = replay()
    trainer = FakeTrainer(r, [])
    trainer.actor_rollout_wg = SimpleNamespace(world_size=4)
    trainer.step(live_batch(), step=1)
    trainer.step(mixed_batch(), step=2)                     # 2 dead rows -> padded to 4
    (call,) = trainer.logprob_calls
    assert len(call) == 4 and call.meta_info["global_token_num"] == call.batch["attention_mask"].sum(dim=-1).tolist()
    assert torch.equal(call.batch["responses"][2], call.batch["responses"][0])   # the pad repeats the first row
    assert r.last_telemetry.replaced_rows == 2 and r.last_telemetry.log_ratio_mean_abs is not None


def test_the_telemetry_forward_is_padded_to_the_log_prob_micro_batch_size():
    from reservoir.integrations.verl import logprob_batch_multiple

    cfg = fake_config(**{"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 8,
                         "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": False})
    r = replay()
    trainer = FakeTrainer(r, [], config=cfg)
    trainer.actor_rollout_wg = SimpleNamespace(world_size=2)
    assert logprob_batch_multiple(trainer) == 16
    trainer.step(live_batch(), step=1)
    trainer.step(mixed_batch(), step=2)                     # 2 dead rows -> padded to 16
    (call,) = trainer.logprob_calls
    assert len(call) == 16 and r.last_telemetry.replaced_rows == 2
    cfg.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz = True
    assert logprob_batch_multiple(trainer) == 2             # dynamic batching: only the world size matters
    assert logprob_batch_multiple(FakeTrainer(replay(), [])) == 1


def test_a_renamed_guard_setting_is_an_error_not_off():
    from reservoir.integrations.verl import config_setting

    cfg = fake_config()
    del cfg.algorithm.rollout_correction.__dict__["rollout_is"]
    with pytest.raises(ValueError, match="rollout_correction.rollout_is"):
        FakeTrainer(replay(), [], config=cfg).step(live_batch(), step=1)
    # A verl without the node at all (an older release) means the feature is absent.
    cfg = fake_config()
    del cfg.algorithm.__dict__["rollout_correction"]
    FakeTrainer(replay(), [], config=cfg).step(live_batch(), step=1)
    assert config_setting(cfg, "actor_rollout_ref.actor.policy_loss", "loss_mode", absent_node="vanilla") == "vanilla"


def test_an_unknown_per_row_tensor_is_refused_before_anything_is_stored():
    r = replay()
    batch = live_batch()
    batch.batch["some_new_verl_key"] = torch.zeros(4, 2)
    with pytest.raises(ValueError, match="some_new_verl_key"):
        FakeTrainer(r, []).step(batch, step=1)
    assert r.buffer.size == 0 and r.stats["hook_calls"] == 0


def test_telemetry_off_skips_the_forward_and_the_record():
    r = replay(telemetry=False, attest=AttestationLog())
    trainer = FakeTrainer(r, [])
    trainer.step(live_batch(), step=1)
    trainer.step(mixed_batch(), step=2)
    assert trainer.logprob_calls == [] and r.last_telemetry is None
    assert all(rec["op"] != "telemetry" for rec in r.buffer.attestation_log.records)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["values", "rollout_is_weights", "routed_experts", "sum_pi_squared"])
def test_unsupported_batch_keys_are_refused_by_name(key):
    batch = live_batch()
    batch.batch[key] = torch.zeros(4, 2)
    with pytest.raises(NotImplementedError, match=key):
        FakeTrainer(replay(), []).step(batch, step=1)


def test_multimodal_inputs_are_refused_only_when_a_row_carries_some():
    # verl's agent loop attaches the column to every batch; text rows hold {} (or None).
    batch = live_batch()
    batch.non_tensor_batch["multi_modal_inputs"] = np.array([{}, None, {}, {}], dtype=object)
    r = replay()
    FakeTrainer(r, []).step(batch, step=1)
    assert r.stats["hook_calls"] == 1
    batch = live_batch()
    batch.non_tensor_batch["multi_modal_inputs"] = np.array([{}, {"pixel_values": object()}, {}, {}], dtype=object)
    with pytest.raises(NotImplementedError, match="multi_modal_inputs"):
        FakeTrainer(replay(), []).step(batch, step=1)


def test_missing_uid_is_refused():
    batch = live_batch()
    del batch.non_tensor_batch["uid"]
    with pytest.raises(ValueError, match="uid"):
        FakeTrainer(replay(), []).step(batch, step=1)


def test_non_grpo_estimators_are_refused():
    cfg = fake_config(**{"algorithm.adv_estimator": "gae"})
    with pytest.raises(NotImplementedError, match="adv_estimator='gae'"):
        FakeTrainer(replay(), [], config=cfg).step(live_batch(), step=1)


@pytest.mark.parametrize("key", ["rollout_is", "bypass_mode"])
def test_rollout_correction_is_refused(key):
    cfg = fake_config(**{f"algorithm.rollout_correction.{key}": "token" if key == "rollout_is" else True})
    with pytest.raises(NotImplementedError, match=key):
        FakeTrainer(replay(), [], config=cfg).step(live_batch(), step=1)


def test_covariance_loss_modes_need_beta_zero():
    cfg = fake_config(**{"actor_rollout_ref.actor.policy_loss.loss_mode": "clip_cov"})
    with pytest.raises(ValueError, match="clip_cov"):
        FakeTrainer(replay(beta=0.4), [], config=cfg).step(live_batch(), step=1)
    FakeTrainer(replay(beta=0.0), [], config=cfg).step(live_batch(), step=1)


def test_a_batch_that_is_not_a_dataproto_is_refused():
    with pytest.raises(TypeError, match="from_dict"):
        FakeTrainer(replay(), []).step(SimpleNamespace(batch={}, non_tensor_batch={}), step=1)


def test_missing_pad_token_is_refused_only_when_rows_are_written():
    r = replay()
    trainer = FakeTrainer(r, [])
    trainer.tokenizer = SimpleNamespace(pad_token_id=None)
    trainer.step(live_batch(), step=1)
    with pytest.raises(ValueError, match="pad_token_id"):
        trainer.step(mixed_batch(), step=2)


# ---------------------------------------------------------------------------
# Replay: what the actor receives
# ---------------------------------------------------------------------------

def reference_run(seed: int, batches: list[tuple[int, FakeDataProto]], **kwargs):
    """Drive a bare RolloutBuffer the way the adapter is supposed to; return it and the last sample."""
    defaults = dict(capacity=16, half_life=4, max_policy_age=8, seed=seed, priority=StoredAdvantagePriority())
    defaults.update(kwargs)
    buf = RolloutBuffer(**defaults)
    sampled = None
    for step, data in batches:
        conv = rows_to_groups(data.batch, list(data.non_tensor_batch["uid"]), step=step)
        for group in conv.groups:
            buf.add_group(group.prompt_id, step, group.rollouts)
        if conv.dead_rows:
            sampled = buf.sample(len(conv.dead_rows), current_version=step)
    return buf, sampled


def test_replayed_rows_match_a_bare_rollout_buffer_with_the_same_seed():
    first, second = live_batch(), mixed_batch()
    r = replay(seed=7)
    trainer = FakeTrainer(r, [])
    trainer.step(copy.deepcopy(first), step=1)

    out = trainer.step(copy.deepcopy(second), step=2)

    _, expected = reference_run(7, [(1, first), (2, second)])
    assert r.last_replay.indices == expected.indices and r.last_replay.is_weights == expected.is_weights
    for k, row in enumerate((2, 3)):
        rollout = expected.rollouts[k]
        n = len(rollout)
        assert out.batch["responses"][row, :n].tolist() == list(rollout.tokens)
        assert out.batch["response_mask"][row].tolist() == [1] * n + [0] * (out.batch["responses"].size(1) - n)
        assert out.batch["old_log_probs"][row, :n].tolist() == list(rollout.logprobs)
        lp = len(rollout.metadata["prompt_ids"])
        assert out.batch["prompts"][row, -lp:].tolist() == rollout.metadata["prompt_ids"]
        expected_adv = f32([rollout.reward * float(expected.is_weights[k])])[0]
        assert out.batch["advantages"][row, :n].tolist() == [expected_adv] * n
        assert out.non_tensor_batch["uid"][row] == rollout.metadata["uid"]
    assert r.stats["replaced_rows"] == 2


def test_the_mixed_dataproto_is_consistent_and_leaves_verls_tensors_alone():
    second = mixed_batch()
    pristine = copy.deepcopy(second.batch)
    second.batch["rollout_log_probs"] = torch.zeros(4, 2)
    trainer = FakeTrainer(replay(), [])
    trainer.step(live_batch(), step=1)

    out = trainer.step(second, step=2)

    for key in pristine:
        assert torch.equal(second.batch[key], pristine[key]), key
    assert "rollout_log_probs" not in out.batch and trainer.replay_buffer.stats["dropped_rollout_logprobs"] == 1
    assert set(out.non_tensor_batch) == {"uid"} and list(out.non_tensor_batch["uid"][:2]) == ["u0", "u0"]
    assert out.meta_info["temperature"] == 1.0
    assert out.meta_info["global_token_num"] == out.batch["attention_mask"].sum(dim=-1).tolist()
    assert torch.equal(out.batch["input_ids"], torch.cat([out.batch["prompts"], out.batch["responses"]], dim=1))
    assert torch.equal(out.batch["position_ids"], verl_position_ids(out.batch["attention_mask"], out.batch["prompts"].size(1)))
    assert torch.equal(out.batch["returns"], out.batch["advantages"])
    assert torch.equal(out.batch["prompts"][:2], second.batch["prompts"][:2])


def test_groups_are_found_by_uid_when_rows_are_interleaved():
    # balance_batch reorders rows: dead rows 1 and 3 belong to one uid.
    batch, _ = make_batch(prompts=[[1], [2], [1], [2]], responses=[[5], [6], [7], [8]], advantages=[1.0, 0.0, -1.0, 0.0])
    data = proto(batch, ["a", "b", "a", "b"])
    r = replay()
    trainer = FakeTrainer(r, [])
    trainer.step(live_batch(), step=1)
    out = trainer.step(data, step=2)
    assert r.stats["dead_groups"] == 1 and r.stats["replaced_rows"] == 2
    width = data.batch["responses"].size(1)   # replay may have widened the batch
    for live_row in (0, 2):
        assert torch.equal(out.batch["responses"][live_row, :width], data.batch["responses"][live_row])
        assert torch.equal(out.batch["advantages"][live_row, :width], data.batch["advantages"][live_row])
    for dead_row in (1, 3):
        assert not torch.equal(out.batch["advantages"][dead_row, :width], data.batch["advantages"][dead_row])


def test_replay_grows_the_batch_when_a_stored_sequence_is_longer():
    long_batch, uids = make_batch(prompts=[[1, 2, 3], [1, 2, 3]], responses=[[4, 5, 6, 7], [8, 9, 10, 11]], advantages=[1.0, -1.0])
    trainer = FakeTrainer(replay(), [])
    trainer.step(proto(long_batch, uids), step=1)
    out = trainer.step(dead_batch(), step=2)
    assert out.batch["prompts"].size(1) == 3 and out.batch["responses"].size(1) == 4
    assert out.batch["input_ids"].size(1) == 7


def test_beta_zero_replays_the_stored_advantage_unscaled():
    r = replay(beta=0.0)
    trainer = FakeTrainer(r, [])
    trainer.step(live_batch(), step=1)
    out = trainer.step(mixed_batch(), step=2)
    assert all(w == Fraction(1) for w in r.last_replay.is_weights)
    assert [out.batch["advantages"][row, 0].item() for row in (2, 3)] == [ro.reward for ro in r.last_replay.rollouts]


def test_importance_weights_scale_the_replayed_advantage():
    r = replay(beta=1.0)
    trainer = FakeTrainer(r, [])
    trainer.step(live_batch(), step=1)
    out = trainer.step(mixed_batch(), step=2)
    weights = r.last_replay.is_weights
    assert any(w != 1 for w in weights)
    expected = f32([ro.reward * float(w) for ro, w in zip(r.last_replay.rollouts, weights)])
    assert [out.batch["advantages"][row, 0].item() for row in (2, 3)] == expected


def test_ref_logprobs_travel_with_replayed_rows():
    first = live_batch(ref_logps=[[-1.0, -1.1, -1.2], [-2.0], [-3.0, -3.1], [-4.0, -4.1, -4.2]])
    second = mixed_batch(ref_logps=[[-5.0], [-6.0, -6.1], [-7.0, -7.1], [-8.0]])
    r = replay()
    trainer = FakeTrainer(r, [])
    trainer.step(first, step=1)
    out = trainer.step(second, step=2)
    for k, row in enumerate((2, 3)):
        rollout = r.last_replay.rollouts[k]
        assert out.batch["ref_log_prob"][row, :len(rollout)].tolist() == rollout.metadata["ref_logprobs"]


# ---------------------------------------------------------------------------
# Drift gate
# ---------------------------------------------------------------------------

class DriftingTrainer(FakeTrainer):
    """The current policy has moved far from the stored logprobs on every replayed row."""

    def _compute_old_log_prob(self, batch):
        out, mfu = super()._compute_old_log_prob(batch)
        out.batch["old_log_probs"] = out.batch["old_log_probs"] - 5.0
        return out, mfu


def test_the_drift_gate_declines_and_evicts_drifted_rows():
    r = replay(max_log_ratio=1.0, attest=AttestationLog())
    trainer = DriftingTrainer(r, [])
    trainer.step(live_batch(), step=1)
    dead = mixed_batch()
    out = trainer.step(dead, step=2)
    assert r.stats["declined_rows"] == 2 and r.stats["replaced_rows"] == 0
    assert out.batch["advantages"][2:].tolist() == dead.batch["advantages"][2:].tolist()  # still dead
    witness = [rec for rec in r.buffer.attestation_log.records if rec["op"] == "batch"]
    assert len(witness) == 1 and witness[0]["replaced"] == [] and witness[0]["declined"] == [0, 1]
    drift = [rec for rec in r.buffer.attestation_log.records if rec.get("op") == "evict" and rec.get("reason") == "drift"]
    assert len(drift) >= 1


def test_too_many_declines_raise():
    r = replay(max_log_ratio=1.0, max_declines_per_step=1)
    trainer = DriftingTrainer(r, [])
    trainer.step(live_batch(), step=1)
    with pytest.raises(RuntimeError, match="max_declines_per_step"):
        trainer.step(mixed_batch(), step=2)


# ---------------------------------------------------------------------------
# Determinism and attestation
# ---------------------------------------------------------------------------

def scripted_batches() -> list[FakeDataProto]:
    return [live_batch(10), mixed_batch(), live_batch(30), mixed_batch(), mixed_batch()]


def run_scripted(seed: int, log: AttestationLog, **kwargs) -> list[FakeDataProto]:
    r = replay(seed=seed, attest=log, **kwargs)
    trainer = FakeTrainer(r, scripted_batches())
    trainer.fit()
    return trainer.actor_inputs


def test_same_seed_and_inputs_give_identical_logs_and_batches():
    log_a, log_b = AttestationLog(), AttestationLog()
    outs_a = run_scripted(3, log_a)
    outs_b = run_scripted(3, log_b)
    assert log_a.to_json_lines() == log_b.to_json_lines()
    for a, b in zip(outs_a, outs_b):
        assert tensor_digest(a.batch) == tensor_digest(b.batch)


def test_a_different_seed_changes_the_replay():
    log_a, log_b = AttestationLog(), AttestationLog()
    run_scripted(3, log_a)
    run_scripted(4, log_b)
    assert log_a.to_json_lines() != log_b.to_json_lines()


def test_attestation_log_from_a_scripted_run_passes_the_independent_checker(tmp_path):
    attest, manifest = tmp_path / "attest.jsonl", tmp_path / "manifest.jsonl"
    r = replay(seed=1, max_policy_age=3, half_life=1, attest=attest, manifest=manifest, source="zen")
    trainer = FakeTrainer(r, scripted_batches() * 2)
    trainer.fit()
    r.close()
    assert r.buffer.n_rebases >= 1 and r.stats["replaced_rows"] > 0

    result = verify_json_lines(attest.read_text(), manifest=manifest.read_text())
    assert result.content.manifest_matched == r.stats["ingested_rows"]
    assert len(result.content.witnesses) >= 1
    completed = subprocess.run(
        [sys.executable, "-m", "checker.verify", str(attest), "--manifest", str(manifest)], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_replay_writes_a_batch_witness_that_matches_the_rows(tmp_path):
    r = replay(seed=7, attest=tmp_path / "attest.jsonl")
    trainer = FakeTrainer(r, [])
    trainer.step(live_batch(), step=1)
    out = trainer.step(mixed_batch(), step=2)
    r.close()
    records = [json.loads(line) for line in (tmp_path / "attest.jsonl").read_text().splitlines()]
    (w,) = [rec for rec in records if rec["op"] == "batch"]
    assert w["step"] == "2" and w["batch_rows"] == 4
    assert [e["row"] for e in w["replaced"]] == [2, 3]
    assert w["tensor_digest"] == tensor_digest(out.batch)
    telemetry = [rec for rec in records if rec["op"] == "telemetry"]
    assert len(telemetry) == 2 and telemetry[-1]["replaced_rows"] == 2 and telemetry[0]["replaced_rows"] == 0


# ---------------------------------------------------------------------------
# Hook-ran guard and checkpoint binding through the mixin
# ---------------------------------------------------------------------------

def test_assert_hook_ran_fires_from_the_second_step_of_this_process():
    r = replay()
    assert_hook_ran(r, 1)
    assert_hook_ran(r, 41, resumed_from=40)   # resumed at 40: step 41 is this process's first
    with pytest.raises(RuntimeError, match="never ran"):
        assert_hook_ran(r, 2)
    with pytest.raises(RuntimeError, match="never ran"):
        assert_hook_ran(r, 42, resumed_from=40)


class BypassingBase(FakeBase):
    """A trainer whose loop no longer calls ``_update_actor``: the override is dead."""

    def fit(self, save_freq: int = 0):
        self.global_steps = 1
        for batch in self._batches:
            self._compute_old_log_prob(batch)
            self.global_steps += 1


class BypassingTrainer(ReservoirReplayMixin, BypassingBase):
    def __init__(self, replay, batches):
        super().__init__(batches, fake_config())
        self.replay_buffer = replay


def test_a_loop_that_bypasses_update_actor_fails_on_the_second_step():
    with pytest.raises(RuntimeError, match="never ran"):
        BypassingTrainer(replay(), [live_batch(), live_batch()]).fit()


def test_save_binds_the_durable_buffer_and_load_rewinds_it(tmp_path):
    kwargs = dict(directory=tmp_path / "buf", attest=tmp_path / "attest.jsonl")
    r = replay(**kwargs)
    trainer = FakeTrainer(r, [live_batch(), mixed_batch(), live_batch(20), mixed_batch()], ckpt_dir=tmp_path / "ckpt")
    trainer.fit(save_freq=2)
    assert trainer.saved == [2, 4]
    assert r.buffer.checkpoints() == ["step-2", "step-4"]
    head_at_4 = r.buffer.attestation_log.head_digest
    # The run went on past the last save, then crashed.
    trainer.step(live_batch(40), step=5)
    assert r.buffer.attestation_log.head_digest != head_at_4
    r.close()

    again = replay(**kwargs)
    resumed = FakeTrainer(again, [mixed_batch()], ckpt_dir=tmp_path / "ckpt")
    (tmp_path / "ckpt" / "global_step_2").rename(tmp_path / "ckpt" / "gone")   # the trainer rotated it away
    resumed.fit()                                                               # loads global_step_4, trains step 5

    assert resumed.global_steps == 6 and again.buffer.current_version == 5
    assert again.stats["hook_calls"] == 1
    again.close()


def test_resume_at_a_step_without_a_buffer_checkpoint_is_an_error(tmp_path):
    r = replay(directory=tmp_path / "buf", attest=tmp_path / "attest.jsonl")
    (tmp_path / "ckpt" / "global_step_7").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no checkpoint"):
        FakeTrainer(r, [live_batch()], ckpt_dir=tmp_path / "ckpt").fit()
    r.close()


def test_bind_checkpoint_prunes_to_the_trainers_surviving_directories(tmp_path):
    r = replay(directory=tmp_path / "buf", attest=tmp_path / "attest.jsonl")
    trainer = FakeTrainer(r, [])
    out = tmp_path / "out"
    for step in (1, 2, 3):
        trainer.step(live_batch(step * 10), step=step)
        (out / f"global_step_{step}").mkdir(parents=True)
        bind_checkpoint(r, step, out)
    (out / "global_step_1").rename(out / "gone-1")
    trainer.step(live_batch(40), step=4)
    bind_checkpoint(r, 4, out)
    assert r.buffer.checkpoints() == ["step-2", "step-3", "step-4"]
    assert trainer_checkpoint_steps(out) == {2, 3}
    assert trainer_checkpoint_steps(None) is None and trainer_checkpoint_steps(tmp_path / "missing") is None
    bind_checkpoint(replay(), 9)  # in-memory buffer: nothing to bind
    r.close()


def test_resume_against_a_changed_global_step_directory_is_refused(tmp_path):
    """The buffer checkpoint records the digest of ``global_step_<n>``; a different model is refused."""
    kwargs = dict(directory=tmp_path / "buf", attest=tmp_path / "attest.jsonl")
    r = replay(**kwargs)
    ckpt = tmp_path / "ckpt"
    FakeTrainer(r, [live_batch(), mixed_batch()], ckpt_dir=ckpt).fit(save_freq=2)   # saves global_step_2
    recorded = r.buffer.checkpoint_binding("step-2")["model_checkpoint"]
    assert recorded["name"] == "global_step_2" and recorded["files"] == 0
    r.close()

    (ckpt / "global_step_2" / "actor.pt").write_bytes(b"other weights")
    again = replay(**kwargs)
    with pytest.raises(RuntimeError, match="not the one the buffer checkpoint was bound to"):
        FakeTrainer(again, [mixed_batch()], ckpt_dir=ckpt).fit()
    again.close()

    (ckpt / "global_step_2" / "actor.pt").unlink()
    third = replay(**kwargs)
    FakeTrainer(third, [mixed_batch()], ckpt_dir=ckpt).fit()          # the directory as saved resumes
    assert third.buffer.current_version == 3
    third.close()
