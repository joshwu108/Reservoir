"""Tests for the adapter's replay-health telemetry and drift gate.

Every generation step the adapter measures replay fraction, dead and
near-dead groups, the effective sample size and staleness of the replayed
rows and the log-ratio between stored behavior logprobs and the current
policy; it logs them under ``reservoir/`` in TRL's metrics and writes a
``telemetry`` record the checker recomputes where it can. With
``max_log_ratio`` set, rows that drifted too far are declined: the dead
row stays dead, the entry is evicted with reason ``drift``, the witness
lists the declined draw, and too many declines raise.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from fractions import Fraction

import pytest
import torch

from reservoir.attest import AttestationLog
from reservoir.integrations._trl_telemetry import METRIC_PREFIX, StepTelemetry, choose_declines, summarize
from reservoir_checker.verify import verify_chain
from tests.test_trl_replay import FakeTrainer, LOGP, live_batch, mixed_batch, replay


class MetricTrainer(FakeTrainer):
    """FakeTrainer with TRL's metric lists and a current policy whose logprobs can be set.

    ``current_logp`` is a float for every row, or a callable from a row's
    completion ids to its per-token logprob (so rows can drift differently).
    """

    def __init__(self, *args, current_logp=LOGP, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.current_logp = current_logp

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, **kw):
        self.logps_calls.append({"logits_to_keep": logits_to_keep, "batch_size": batch_size, "rows": input_ids.size(0)})
        if callable(self.current_logp):
            completions = input_ids[:, -logits_to_keep:]
            values = torch.tensor([self.current_logp(row.tolist()) for row in completions])
            return values[:, None].expand(-1, logits_to_keep).clone(), None, None
        return torch.full((input_ids.size(0), logits_to_keep), self.current_logp), None, None


def test_metrics_and_record_for_a_replayed_step():
    r = replay(attest=AttestationLog())
    trainer = MetricTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=1)
    trainer.generate(step=2)

    point = r.last_telemetry
    assert point.batch_rows == 4 and point.replaced_rows == 2 and point.declined_rows == 0
    assert point.dead_groups == 1 and point.near_dead_groups == 0
    assert point.ess == pytest.approx(sum(float(w) for w in r.last_replay.is_weights) ** 2
                                      / sum(float(w) ** 2 for w in r.last_replay.is_weights))
    assert point.staleness_max == 1 and point.staleness_mean == 1.0      # stored at step 1, replayed at step 2
    assert point.log_ratio_mean_abs == 0.0 and point.log_ratio_max_abs == 0.0   # current policy == behavior
    metrics = trainer._metrics["train"]
    assert metrics[METRIC_PREFIX + "replay_fraction"] == [0.0, 0.5]
    assert metrics[METRIC_PREFIX + "ess"][-1] == pytest.approx(point.ess)
    assert set(k for k in metrics if k.startswith(METRIC_PREFIX)) >= {
        METRIC_PREFIX + n for n in ("replay_fraction", "replaced_rows", "declined_rows", "dead_groups",
                                     "near_dead_groups", "ess", "ess_fraction", "staleness_mean",
                                     "staleness_max", "log_ratio_mean_abs", "log_ratio_max_abs")}
    assert r.stats["telemetry_forwards"] == 1
    records = r.buffer.attestation_log.records
    telemetry = [rec for rec in records if rec["op"] == "telemetry"]
    assert len(telemetry) == 2
    assert telemetry[0]["replaced_rows"] == 0 and "sample_op_counter" not in telemetry[0]
    assert telemetry[1]["sample_op_counter"] == 1 and telemetry[1]["reported"] == ["log_ratio_max_abs", "log_ratio_mean_abs"]
    assert Fraction(int(telemetry[1]["ess_num"]), int(telemetry[1]["ess_den"])) == pytest.approx(point.ess)
    result = verify_chain(records)
    assert [t.step for t in result.content.telemetry] == [1, 2]
    assert float(result.content.telemetry[1].ess) == pytest.approx(point.ess)


def test_log_ratio_measures_policy_drift():
    r = replay(attest=AttestationLog())
    trainer = MetricTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=1)                 # stored with behavior logprobs LOGP
    trainer.current_logp = LOGP - 0.25       # the policy moved before the next step
    trainer.generate(step=2)
    # Each replayed row sums (current - behavior) over its tokens: -0.25 per token.
    lengths = [len(rollout) for rollout in r.last_replay.rollouts]
    expected = [0.25 * n for n in lengths]
    assert r.last_telemetry.log_ratio_max_abs == pytest.approx(max(expected))
    assert r.last_telemetry.log_ratio_mean_abs == pytest.approx(sum(expected) / len(expected))


def test_telemetry_off_skips_the_forward_and_the_record():
    r = replay(attest=AttestationLog(), telemetry=False)
    trainer = MetricTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=1)
    trainer.generate(step=2)
    assert r.stats["telemetry_forwards"] == 0 and r.last_telemetry is None
    assert all(rec["op"] != "telemetry" for rec in r.buffer.attestation_log.records)
    assert not any(k.startswith(METRIC_PREFIX) for k in trainer._metrics["train"])


def test_trainer_without_metric_lists_is_fine():
    r = replay(attest=AttestationLog())
    trainer = FakeTrainer(r, [live_batch(), mixed_batch()])    # no _metrics attribute
    trainer.generate(step=1)
    trainer.generate(step=2)
    assert r.last_telemetry is not None


def test_drift_gate_declines_evicts_and_witnesses():
    r = replay(attest=AttestationLog(), max_log_ratio=0.1)
    trainer = MetricTrainer(r, [live_batch(), copy.deepcopy(mixed_batch())])
    trainer.generate(step=1)
    trainer.current_logp = LOGP - 1.0
    size_before = r.buffer.size
    out = trainer.generate(step=2)
    # Every replayed row drifted by >= 1.0 nat, so both are declined: the dead rows stay dead.
    assert r.stats["declined_rows"] == 2 and r.stats["replaced_rows"] == 0
    assert out["advantages"][2].item() == 0.0 and out["advantages"][3].item() == 0.0
    records = r.buffer.attestation_log.records
    drift = [rec for rec in records if rec["op"] == "evict" and rec.get("reason") == "drift"]
    # One eviction per distinct declined slot; the step also stored the batch's live group (2 rows).
    assert sorted(rec["index"] for rec in drift) == sorted(set(r.last_replay.indices))
    assert r.buffer.size == size_before + 2 - len(drift)
    witness = next(rec for rec in records if rec["op"] == "batch")
    assert witness["replaced"] == [] and witness["declined"] == [0, 1]
    telemetry = [rec for rec in records if rec["op"] == "telemetry"][-1]
    assert telemetry["declined_rows"] == 2 and telemetry["replaced_rows"] == 0 and "ess_num" in telemetry
    # Telemetry describes the sampled batch and precedes the drift evictions in the log.
    assert records.index(telemetry) < records.index(drift[0])
    verify_chain(records)


def test_gate_keeps_rows_within_threshold():
    r = replay(attest=AttestationLog(), max_log_ratio=100.0)
    trainer = MetricTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=1)
    trainer.current_logp = LOGP - 1.0
    trainer.generate(step=2)
    assert r.stats["declined_rows"] == 0 and r.stats["replaced_rows"] == 2
    verify_chain(r.buffer.attestation_log.records)


def test_too_many_declines_raise_instead_of_continuing():
    r = replay(attest=AttestationLog(), max_log_ratio=0.1, max_declines_per_step=1)
    trainer = MetricTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=1)
    trainer.current_logp = LOGP - 1.0
    with pytest.raises(RuntimeError, match="would decline 2"):
        trainer.generate(step=2)


def test_gate_parameters_are_validated():
    with pytest.raises(ValueError, match="max_log_ratio"):
        replay(max_log_ratio=-1.0)
    assert choose_declines([0.5, 2.0], None, None) == []
    assert choose_declines([0.5, 2.0], 1.0, None) == [1]
    with pytest.raises(RuntimeError):
        choose_declines([5.0, 5.0], 1.0, 1)


def test_summarize_without_replay():
    point = summarize(None, 3, [], [], 8, 2, 1, 0)
    assert point == StepTelemetry(8, 0, 0, 2, 1)
    assert point.metrics()[METRIC_PREFIX + "replay_fraction"] == 0.0 and point.reported() == {}



def test_partial_decline_keeps_some_rows_and_declines_others():
    # Rows whose first completion token is odd drift by 2 nats, the others not at all.
    # The seed is searched so the two draws fall on different sides of the gate.
    for seed in range(40):
        r = replay(attest=AttestationLog(), max_log_ratio=0.5, seed=seed)
        trainer = MetricTrainer(r, [live_batch(), copy.deepcopy(mixed_batch())])
        trainer.generate(step=1)
        trainer.current_logp = lambda ids: LOGP - (2.0 if ids[0] % 2 else 0.0)
        out = trainer.generate(step=2)
        batch = r.last_replay
        drifted = [k for k, rollout in enumerate(batch.rollouts) if rollout.tokens[0] % 2]
        kept = [k for k in range(len(batch.rollouts)) if k not in drifted]
        if drifted and kept:
            break
    else:
        raise AssertionError("no seed in range drew one drifting and one steady row")
    records = r.buffer.attestation_log.records
    witness = next(rec for rec in records if rec["op"] == "batch")
    assert sorted(witness["declined"]) == drifted and len(witness["replaced"]) == len(kept)
    # Declined rows keep TRL's dead row: zero advantage, no replayed tokens.
    dead_rows = (2, 3)
    for k in drifted:
        assert out["advantages"][dead_rows[k]].item() == 0.0
    for k in kept:
        n = len(batch.rollouts[k])
        assert out["completion_ids"][dead_rows[k], :n].tolist() == list(batch.rollouts[k].tokens)
    assert r.stats["declined_rows"] == len(drifted) and r.stats["replaced_rows"] == len(kept)
    telemetry = [rec for rec in records if rec["op"] == "telemetry"][-1]
    assert telemetry["declined_rows"] == len(drifted) and telemetry["replaced_rows"] == len(kept)
    verify_chain(records)


@pytest.mark.parametrize("bad", [float("nan"), float("-inf")])
def test_non_finite_log_ratios_are_declined_not_ignored(bad):
    # TRL supplies the behavior logprobs of the second batch, so the fake's forward
    # serves only the drift measurement and can return a degenerate value.
    r = replay(attest=AttestationLog(), max_log_ratio=1.0)
    second = mixed_batch(old_logps=[[-0.1], [-0.2, -0.3], [-0.4, -0.5], [-0.6]])
    trainer = MetricTrainer(r, [live_batch(), second])
    trainer.generate(step=1)
    trainer.current_logp = bad
    trainer.generate(step=2)
    assert r.stats["declined_rows"] == 2 and r.stats["replaced_rows"] == 0
    assert r.last_telemetry.log_ratio_max_abs is None      # nothing finite to report
    verify_chain(r.buffer.attestation_log.records)
    assert choose_declines([float("nan"), 0.1], 1.0, None) == [0]
