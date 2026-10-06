"""Tests for ``reservoir.integrations.trl``: the replay adapter for TRL's GRPOTrainer.

Nothing here imports TRL. ``FakeTrainer`` combines the adapter's
``ReservoirReplayMixin`` with a stand-in base class whose
``_generate_and_score_completions`` returns canned output dicts shaped
exactly like TRL's, and exposes the trainer members the adapter reads
(``num_generations``, ``_tokenizer.pad_token_id``, ``state.global_step``,
``model.training``, ``accelerator``, ``args``, ``loss_type``,
``_get_per_token_logps_and_entropies``). That is the interface TRL
presents to the hook, so the tests exercise the real override end to end.

Parity tests compare what the trainer receives with a plain
``RolloutBuffer`` driven by the same ``add_group``/``sample`` sequence
with the same seed: the adapter must add no sampling logic of its own.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import sys
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from checker.verify import verify_json_lines
from reservoir.attest import AttestationLog
from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.integrations._trl_rows import rows_to_groups
from reservoir.integrations.trl import (
    ReservoirReplay,
    ReservoirReplayMixin,
    StoredAdvantagePriority,
)
from reservoir.priorities import PriorityStrategy
from reservoir.rollout_buffer import RolloutBuffer
from tests.test_trl_rows import PAD, f32, make_output

G = 2  # num_generations in every test batch
LOGP = -0.5  # value the fake forward returns for every token


class FakeBase:
    """Stands in for ``GRPOTrainer``: returns canned generation outputs in order."""

    def __init__(self, outputs: list[dict]) -> None:
        self._outputs = list(outputs)

    def _generate_and_score_completions(self, inputs):
        return self._outputs.pop(0)


class FakeTrainer(ReservoirReplayMixin, FakeBase):
    """The adapter's mixin over ``FakeBase``, with the trainer members the hook reads."""

    def __init__(
        self,
        replay: ReservoirReplay,
        outputs: list[dict],
        *,
        training: bool = True,
        step: int = 0,
        loss_type: str = "grpo",
        num_processes: int = 1,
        vllm_importance_sampling_correction: bool = False,
        off_policy_mask_threshold=None,
    ) -> None:
        super().__init__(outputs)
        self.replay_buffer = replay
        self.vllm_importance_sampling_correction = vllm_importance_sampling_correction
        self.off_policy_mask_threshold = off_policy_mask_threshold
        self.num_generations = G
        self._tokenizer = SimpleNamespace(pad_token_id=PAD)
        self.state = SimpleNamespace(global_step=step)
        self.model = SimpleNamespace(training=training)
        self.accelerator = SimpleNamespace(num_processes=num_processes, gather=lambda t: t)
        self.args = SimpleNamespace(per_device_train_batch_size=4)
        self.loss_type = loss_type
        self.logps_calls: list[dict] = []

    def _get_per_token_logps_and_entropies(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, **kwargs
    ):
        self.logps_calls.append({"logits_to_keep": logits_to_keep, "batch_size": batch_size})
        logps = torch.full((input_ids.size(0), logits_to_keep), LOGP)
        return logps, None, None

    def generate(self, step: int) -> dict:
        """Run one generation step at ``global_step == step`` through the hook."""
        self.state.global_step = step
        return self._generate_and_score_completions({})


def live_batch(prompt_base: int = 10, **kwargs) -> dict:
    """Two live groups of G=2 rows: advantages (+1, -1) and (+0.5, -0.5)."""
    return make_output(
        prompts=[[prompt_base, 1], [prompt_base, 1], [prompt_base + 1], [prompt_base + 1]],
        completions=[[3, 4, 5], [6], [7, 8], [9, 9, 9]],
        advantages=[1.0, -1.0, 0.5, -0.5],
        **kwargs,
    )


def mixed_batch(**kwargs) -> dict:
    """One live group (rows 0-1) and one dead group (rows 2-3)."""
    return make_output(
        prompts=[[20, 21], [20, 21], [22], [22]],
        completions=[[3], [4, 4], [5, 5], [5]],
        advantages=[0.25, -0.25, 0.0, 0.0],
        **kwargs,
    )


def replay(**kwargs) -> ReservoirReplay:
    defaults = dict(capacity=16, half_life=4, max_policy_age=8, seed=0)
    defaults.update(kwargs)
    return ReservoirReplay(**defaults)


# ---------------------------------------------------------------------------
# Construction and the default priority
# ---------------------------------------------------------------------------

def test_stored_advantage_priority_scores_absolute_reward_plus_epsilon():
    from reservoir.rollout import Rollout, RolloutGroup

    strategy = StoredAdvantagePriority(epsilon=1e-3)
    group = RolloutGroup("p", 0, [Rollout([1], [-0.1], -0.75), Rollout([2], [-0.1], 0.25)])

    assert isinstance(strategy, PriorityStrategy)
    assert strategy.score(group.rollouts[0], group) == pytest.approx(0.751)
    assert strategy.score(group.rollouts[1], group) == pytest.approx(0.251)
    with pytest.raises(dataclasses.FrozenInstanceError):
        strategy.epsilon = 0.5


def test_defaults_wrap_an_in_memory_rollout_buffer():
    r = replay()

    assert isinstance(r.buffer, RolloutBuffer)
    assert isinstance(r.buffer.priority, StoredAdvantagePriority)
    assert r.buffer.params.half_life == 4 and r.buffer.params.max_policy_age == 8
    assert r.last_replay is None
    assert set(r.stats) >= {
        "hook_calls", "ingested_rows", "ingested_groups", "dead_groups", "skipped_rows",
        "clamped_logprobs", "replaced_rows", "logprob_forwards",
    }
    assert all(v == 0 for v in r.stats.values())


def test_directory_makes_the_buffer_durable(tmp_path):
    r = replay(directory=tmp_path / "buf", attest=tmp_path / "attest.jsonl")
    assert isinstance(r.buffer, DurableRolloutBuffer)
    r.close()


def test_beta_and_priority_are_forwarded():
    r = replay(beta=0.0, priority=StoredAdvantagePriority(epsilon=0.0))
    assert r.buffer.beta == 0.0
    assert r.buffer.priority.epsilon == 0.0


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def test_hook_stores_every_live_row_at_the_global_step():
    r = replay()
    batch = mixed_batch(old_logps=[[-0.1], [-0.2, -0.3], [-0.4, -0.5], [-0.6]])
    trainer = FakeTrainer(r, [batch])

    out = trainer.generate(step=3)

    # The live group is stored first and is already eligible, so the dead
    # group of the same step is replayed from it.
    assert out is not batch
    assert r.buffer.size == 2
    assert r.buffer.current_version == 3
    stored = {r.buffer.entry(p)[0].metadata["row"]: r.buffer.entry(p) for p in r.buffer.live_positions()}
    rollout0, group0 = stored[0]
    assert rollout0.tokens == (3,) and list(rollout0.logprobs) == f32([-0.1]) and rollout0.reward == 0.25
    assert stored[1][0].tokens == (4, 4) and list(stored[1][0].logprobs) == f32([-0.2, -0.3])
    assert group0.model_version == 3 and group0.size == 2
    assert all(r.buffer.entry_version(p) == 3 for p in r.buffer.live_positions())
    assert r.stats["hook_calls"] == 1
    assert r.stats["ingested_rows"] == 2 and r.stats["ingested_groups"] == 1
    assert r.stats["dead_groups"] == 1
    # TRL supplied old_per_token_logps, so no behavior-logprob forward ran; the one
    # forward the fake saw measures the replayed rows' drift for telemetry.
    assert r.stats["logprob_forwards"] == 0 and r.stats["telemetry_forwards"] == 1
    assert len(trainer.logps_calls) == 1


def test_hook_computes_behavior_logprobs_once_when_trl_omits_them():
    r = replay()
    trainer = FakeTrainer(r, [live_batch()])

    out = trainer.generate(step=0)

    assert trainer.logps_calls == [{"logits_to_keep": 3, "batch_size": 4}]
    assert r.stats["logprob_forwards"] == 1
    assert "old_per_token_logps" not in out  # no replay happened: TRL's dict is untouched
    for p in r.buffer.live_positions():
        rollout = r.buffer.entry(p)[0]
        assert all(lp == LOGP for lp in rollout.logprobs)


def test_hook_is_a_no_op_in_eval_mode():
    r = replay()
    batch = live_batch()
    trainer = FakeTrainer(r, [batch], training=False)

    assert trainer.generate(step=5) is batch
    assert r.buffer.size == 0 and r.stats["hook_calls"] == 0


@pytest.mark.parametrize("key", ["tool_mask", "pixel_values", "importance_sampling_ratio"])
def test_unsupported_output_keys_are_refused_by_name(key):
    batch = live_batch()
    batch[key] = torch.zeros(1)
    trainer = FakeTrainer(replay(), [batch])

    with pytest.raises(NotImplementedError, match=key):
        trainer.generate(step=0)


def test_vllm_sampling_logprobs_are_dropped_when_trl_would_not_use_them():
    # TRL attaches them to every vLLM batch; with the importance correction
    # and off-policy masking off, the loss never reads them.
    r = replay()
    out = mixed_batch()
    out["sampling_per_token_logps"] = torch.full_like(out["completion_ids"], -0.3, dtype=torch.float32)
    trainer = FakeTrainer(r, [out])
    result = trainer.generate(step=1)
    assert "sampling_per_token_logps" not in result
    assert r.stats["dropped_sampling_logprobs"] == 1
    assert r.stats["ingested_rows"] == 2


@pytest.mark.parametrize("kwargs", [
    dict(vllm_importance_sampling_correction=True),
    dict(off_policy_mask_threshold=0.5),
])
def test_vllm_sampling_logprobs_are_refused_when_trl_would_use_them(kwargs):
    r = replay()
    out = mixed_batch()
    out["sampling_per_token_logps"] = torch.zeros_like(out["completion_ids"], dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="sampling_per_token_logps"):
        FakeTrainer(r, [out], **kwargs).generate(step=1)
    assert r.stats["dropped_sampling_logprobs"] == 0


def test_multi_process_is_refused():
    trainer = FakeTrainer(replay(), [live_batch()], num_processes=2)
    with pytest.raises(NotImplementedError, match="process"):
        trainer.generate(step=0)


def test_vespo_with_importance_weights_is_refused():
    trainer = FakeTrainer(replay(beta=0.4), [live_batch()], loss_type="vespo")
    with pytest.raises(ValueError, match="vespo"):
        trainer.generate(step=0)


def test_vespo_without_importance_weights_is_allowed():
    trainer = FakeTrainer(replay(beta=0.0), [live_batch()], loss_type="vespo")
    trainer.generate(step=0)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def reference_run(seed: int, batches: list[tuple[int, dict]], **kwargs) -> tuple[RolloutBuffer, object]:
    """Drive a bare RolloutBuffer the way the adapter is supposed to; return it and the last batch."""
    defaults = dict(capacity=16, half_life=4, max_policy_age=8, seed=seed, priority=StoredAdvantagePriority())
    defaults.update(kwargs)
    buf = RolloutBuffer(**defaults)
    sampled = None
    for step, output in batches:
        logps = output.get("old_per_token_logps")
        if logps is None:
            logps = torch.full_like(output["completion_ids"], LOGP, dtype=torch.float32)
        conv = rows_to_groups(output, num_generations=G, step=step, logprobs=logps)
        for group in conv.groups:
            buf.add_group(group.prompt_id, step, group.rollouts)
        if conv.dead_rows:
            sampled = buf.sample(len(conv.dead_rows), current_version=step)
    return buf, sampled


def test_replayed_rows_match_a_bare_rollout_buffer_with_the_same_seed():
    first, second = live_batch(), mixed_batch()
    r = replay(seed=7)
    trainer = FakeTrainer(r, [copy.deepcopy(first), copy.deepcopy(second)])
    trainer.generate(step=0)

    out = trainer.generate(step=1)

    _, expected = reference_run(7, [(0, first), (1, second)])
    assert r.last_replay is not None
    assert r.last_replay.indices == expected.indices
    assert r.last_replay.is_weights == expected.is_weights
    for k, row in enumerate((2, 3)):
        rollout = expected.rollouts[k]
        n = len(rollout)
        assert out["completion_ids"][row, :n].tolist() == list(rollout.tokens)
        assert out["completion_mask"][row].tolist() == [1] * n + [0] * (out["completion_ids"].size(1) - n)
        assert out["old_per_token_logps"][row, :n].tolist() == list(rollout.logprobs)
        lp = len(rollout.metadata["prompt_ids"])
        assert out["prompt_ids"][row, -lp:].tolist() == rollout.metadata["prompt_ids"]
        expected_adv = f32([rollout.reward * float(expected.is_weights[k])])[0]
        assert out["advantages"][row].item() == expected_adv
    assert r.stats["replaced_rows"] == 2


def test_replay_leaves_fresh_rows_alone_and_fixes_the_loss_denominator():
    second = mixed_batch()
    pristine = mixed_batch()
    trainer = FakeTrainer(replay(), [live_batch(), second])
    trainer.generate(step=0)

    out = trainer.generate(step=1)

    assert out is not second
    width = second["completion_ids"].size(1)
    assert torch.equal(out["prompt_ids"][:2], second["prompt_ids"][:2])
    assert torch.equal(out["prompt_mask"][:2], second["prompt_mask"][:2])
    assert torch.equal(out["completion_ids"][:2, :width], second["completion_ids"][:2])
    assert torch.equal(out["completion_mask"][:2, :width], second["completion_mask"][:2])
    assert torch.equal(out["advantages"][:2], second["advantages"][:2])
    for key in pristine:  # TRL's own tensors were not modified in place
        assert torch.equal(second[key], pristine[key]), key
    assert out["old_per_token_logps"].shape == out["completion_ids"].shape
    assert out["old_per_token_logps"][:2, :width].tolist() == [[LOGP] * width] * 2
    assert out["num_items_in_batch"].item() == out["completion_mask"].sum().item()


def test_dead_groups_stay_dead_while_the_buffer_is_empty():
    r = replay()
    dead = make_output(prompts=[[1], [1]], completions=[[2], [3]], advantages=[0.0, 0.0])
    trainer = FakeTrainer(r, [dead])

    out = trainer.generate(step=0)

    assert out is dead
    assert r.last_replay is None and r.stats["dead_groups"] == 1 and r.buffer.size == 0
    assert r.buffer.current_version == 0


def test_a_buffer_that_went_fully_stale_is_advanced_then_left_alone():
    r = replay(max_policy_age=2)
    dead = make_output(prompts=[[1], [1]], completions=[[2], [3]], advantages=[0.0, 0.0])
    trainer = FakeTrainer(r, [live_batch(), dead])
    trainer.generate(step=0)
    assert r.buffer.size == 4

    out = trainer.generate(step=10)

    assert out is dead
    assert r.buffer.size == 0 and r.buffer.current_version == 10
    assert r.last_replay is None


def test_a_step_below_the_buffer_version_is_refused_before_anything_is_stored():
    r = replay()
    trainer = FakeTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=5)
    stored = r.buffer.size

    with pytest.raises(ValueError, match="global_step 3 is below"):
        trainer.generate(step=3)

    assert r.buffer.size == stored and r.stats["hook_calls"] == 1


def test_an_all_live_batch_still_advances_the_version():
    r = replay()
    trainer = FakeTrainer(r, [live_batch(), live_batch(50)])
    trainer.generate(step=0)
    trainer.generate(step=4)
    assert r.buffer.current_version == 4


def test_ref_logprobs_travel_with_replayed_rows():
    first = live_batch(ref_logps=[[-1.0, -1.1, -1.2], [-2.0], [-3.0, -3.1], [-4.0, -4.1, -4.2]])
    second = mixed_batch(ref_logps=[[-5.0], [-6.0, -6.1], [-7.0, -7.1], [-8.0]])
    r = replay()
    trainer = FakeTrainer(r, [first, second])
    trainer.generate(step=0)

    out = trainer.generate(step=1)

    for k, row in enumerate((2, 3)):
        rollout = r.last_replay.rollouts[k]
        n = len(rollout)
        assert out["ref_per_token_logps"][row, :n].tolist() == rollout.metadata["ref_logprobs"]


def test_durable_buffer_through_the_hook_survives_reopen(tmp_path):
    kwargs = dict(capacity=16, half_life=4, max_policy_age=8, seed=0)
    r = ReservoirReplay(directory=tmp_path / "buf", attest=tmp_path / "attest.jsonl", **kwargs)
    trainer = FakeTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=0)
    out = trainer.generate(step=1)
    live = r.buffer.live_positions()
    head = r.buffer.attestation_log.head_digest
    r.close()

    reopened = ReservoirReplay(directory=tmp_path / "buf", attest=tmp_path / "attest.jsonl", **kwargs)

    assert reopened.buffer.live_positions() == live
    assert reopened.buffer.attestation_log.head_digest == head
    assert out["advantages"].shape == (4,)
    reopened.close()


def test_replay_grows_the_batch_when_a_stored_completion_is_longer():
    long_batch = make_output(
        prompts=[[1, 2, 3], [1, 2, 3]], completions=[[4, 5, 6, 7], [8, 9, 10, 11]], advantages=[1.0, -1.0],
    )
    short_dead = make_output(prompts=[[9], [9]], completions=[[2], [3]], advantages=[0.0, 0.0])
    trainer = FakeTrainer(replay(), [long_batch, short_dead])
    trainer.generate(step=0)

    out = trainer.generate(step=1)

    assert out["prompt_ids"].size(1) == 3 and out["completion_ids"].size(1) == 4


def test_beta_zero_replays_the_stored_advantage_unscaled():
    r = replay(beta=0.0)
    trainer = FakeTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=0)

    out = trainer.generate(step=1)

    assert all(w == Fraction(1) for w in r.last_replay.is_weights)
    assert out["advantages"][2:].tolist() == [ro.reward for ro in r.last_replay.rollouts]


def test_importance_weights_with_beta_one_give_a_hand_computed_advantage():
    """With beta=1 and priority=|advantage|, w_i = leaf_min / leaf_i, so every
    replayed advantage has magnitude min|advantage| and keeps its sign. Both
    batches arrive at the same step so no age decay enters the leaves."""
    r = replay(beta=1.0, priority=StoredAdvantagePriority(epsilon=0.0))
    trainer = FakeTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=0)

    out = trainer.generate(step=0)

    rewards = [ro.reward for ro in r.last_replay.rollouts]
    assert set(abs(x) for x in rewards) <= {1.0, 0.5, 0.25}
    assert out["advantages"][2:].tolist() == [0.25 if x > 0 else -0.25 for x in rewards]


def test_importance_weights_scale_the_replayed_advantage():
    r = replay(beta=1.0)
    trainer = FakeTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=0)

    out = trainer.generate(step=1)

    weights = r.last_replay.is_weights
    assert any(w != 1 for w in weights)
    expected = f32([ro.reward * float(w) for ro, w in zip(r.last_replay.rollouts, weights)])
    assert out["advantages"][2:].tolist() == expected


# ---------------------------------------------------------------------------
# Determinism, staleness, attestation
# ---------------------------------------------------------------------------

def scripted_outputs() -> list[dict]:
    return [live_batch(10), mixed_batch(), live_batch(30), mixed_batch(), mixed_batch()]


def run_scripted(seed: int, log: AttestationLog, **kwargs) -> list[dict]:
    r = replay(seed=seed, attest=log, **kwargs)
    trainer = FakeTrainer(r, scripted_outputs())
    return [trainer.generate(step=s) for s in range(5)]


def test_same_seed_and_inputs_give_identical_logs_and_batches():
    log_a, log_b = AttestationLog(), AttestationLog()

    outs_a = run_scripted(3, log_a)
    outs_b = run_scripted(3, log_b)

    assert log_a.to_json_lines() == log_b.to_json_lines()
    for a, b in zip(outs_a, outs_b):
        for key in a:
            assert torch.equal(a[key], b[key]), key


def test_a_different_seed_changes_the_replay():
    log_a, log_b = AttestationLog(), AttestationLog()
    run_scripted(3, log_a)
    run_scripted(4, log_b)
    assert log_a.to_json_lines() != log_b.to_json_lines()


def test_entries_older_than_max_policy_age_are_evicted_by_a_later_hook():
    log = AttestationLog()
    r = replay(max_policy_age=2, attest=log)
    trainer = FakeTrainer(r, [live_batch(), live_batch(50)])
    trainer.generate(step=0)
    assert r.buffer.size == 4

    trainer.generate(step=3)

    assert r.buffer.size == 4  # 4 stale evicted, 4 new stored
    assert all(r.buffer.entry_version(p) == 3 for p in r.buffer.live_positions())
    stale = [rec for rec in log.records if rec.get("op") == "evict" and rec.get("reason") == "stale"]
    assert len(stale) == 4


def test_attestation_log_from_a_scripted_run_passes_the_independent_checker(tmp_path):
    path = tmp_path / "attest.jsonl"
    r = replay(seed=1, max_policy_age=3, half_life=1, attest=path)
    trainer = FakeTrainer(r, scripted_outputs() * 2)
    for step in range(10):
        trainer.generate(step=step)
    r.close()
    assert r.buffer.n_rebases >= 1

    verify_json_lines(path.read_text())
    result = subprocess.run(
        [sys.executable, "-m", "checker.verify", str(path)], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


# ---------------------------------------------------------------------------
# Content commitment through the adapter: source tag and manifest
# ---------------------------------------------------------------------------

def test_source_and_manifest_pass_through_to_the_buffer(tmp_path):
    attest, manifest = tmp_path / "attest.jsonl", tmp_path / "manifest.jsonl"
    r = replay(attest=attest, manifest=manifest, source="zen")
    trainer = FakeTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=1)
    trainer.generate(step=2)
    r.close()

    records = [json.loads(l) for l in attest.read_text().splitlines()]
    inserts = [rec for rec in records if rec["op"] == "insert"]
    assert len(inserts) == r.stats["ingested_rows"] == 6
    assert all(rec["source"] == "zen" and len(rec["content_digest"]) == 64 for rec in inserts)
    assert r.source == "zen" and r.buffer.has_manifest
    result = verify_json_lines(attest.read_text(), manifest=manifest.read_text())
    assert result.content.manifest_matched == 6
    for pos in r.buffer.live_positions():
        assert r.buffer.entry(pos)[1].source == "zen"


def test_source_defaults_to_none_and_manifest_requires_attest(tmp_path):
    r = replay(attest=AttestationLog())
    FakeTrainer(r, [live_batch()]).generate(step=1)
    assert r.source is None
    assert all("source" not in rec for rec in r.buffer.attestation_log.records)
    with pytest.raises(ValueError, match="manifest"):
        replay(manifest=tmp_path / "m.jsonl")


def test_durable_replay_keeps_the_manifest(tmp_path):
    attest, manifest = tmp_path / "attest.jsonl", tmp_path / "manifest.jsonl"
    r = replay(directory=tmp_path / "buf", attest=attest, manifest=manifest, source="s")
    FakeTrainer(r, [live_batch()]).generate(step=1)
    lines = r.buffer.manifest_records
    r.close()
    again = replay(directory=tmp_path / "buf", attest=attest, manifest=manifest, source="s")
    assert isinstance(again.buffer, DurableRolloutBuffer)
    assert again.buffer.manifest_records == lines and len(lines) == 4
    again.close()



# ---------------------------------------------------------------------------
# Batch witness: the log names which rows hold which draws
# ---------------------------------------------------------------------------

def test_replay_writes_a_batch_witness_that_matches_the_rows(tmp_path):
    from reservoir.integrations.trl import tensor_digest

    first, second = live_batch(), mixed_batch()
    r = replay(seed=7, attest=tmp_path / "attest.jsonl", manifest=tmp_path / "manifest.jsonl")
    trainer = FakeTrainer(r, [copy.deepcopy(first), copy.deepcopy(second)])
    trainer.generate(step=0)
    out = trainer.generate(step=1)
    r.close()

    records = [json.loads(l) for l in (tmp_path / "attest.jsonl").read_text().splitlines()]
    witnesses = [rec for rec in records if rec["op"] == "batch"]
    assert len(witnesses) == 1
    w = witnesses[0]
    assert records.index(w) > max(i for i, rec in enumerate(records) if rec["op"] == "sample")
    assert w["step"] == "1" and w["batch_rows"] == out["advantages"].size(0)
    assert [e["row"] for e in w["replaced"]] == [2, 3]
    assert [e["draw"] for e in w["replaced"]] == [0, 1]
    assert [e["content_digest"] for e in w["replaced"]] == list(r.last_replay.content_digests)
    assert w["tensor_digest"] == tensor_digest(out)
    result = verify_json_lines((tmp_path / "attest.jsonl").read_text(), manifest=(tmp_path / "manifest.jsonl").read_text())
    assert len(result.content.witnesses) == 1
    # The witness proves the rows hold the sampled examples: check against the batch itself.
    for e, rollout in zip(w["replaced"], r.last_replay.rollouts):
        n = len(rollout)
        assert out["completion_ids"][e["row"], :n].tolist() == list(rollout.tokens)


def test_no_witness_when_nothing_was_replayed():
    r = replay(attest=AttestationLog())
    FakeTrainer(r, [live_batch()]).generate(step=0)
    assert all(rec["op"] != "batch" for rec in r.buffer.attestation_log.records)


def test_tensor_digest_is_a_function_of_the_tensors_only():
    from reservoir.integrations.trl import tensor_digest

    a = live_batch()
    b = copy.deepcopy(a)
    assert tensor_digest(a) == tensor_digest(b)
    b["completion_ids"][0, 0] += 1
    assert tensor_digest(a) != tensor_digest(b)
    c = copy.deepcopy(a)
    c["advantages"] = c["advantages"].to(torch.float64)
    assert tensor_digest(a) == tensor_digest(c)
