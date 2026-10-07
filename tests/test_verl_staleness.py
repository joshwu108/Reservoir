"""Tests for the staleness policy in the verl adapter's replay path.

verl's rows of one prompt share a ``uid`` and may be reordered by the
trainer, so the adapter declares each decision's group (uids in order of
first appearance among the dead rows) and records ``group_size`` as null;
everything else mirrors the TRL adapter and is verified by the same
checker.
"""

from __future__ import annotations

import pytest

from reservoir.attest import AttestationLog
from reservoir.integrations._trl_staleness import StalenessPolicy
from reservoir_checker.verify import verify_chain
from tests.test_verl_replay import DriftingTrainer, FakeTrainer, live_batch, mixed_batch, replay


class RisingTrainer(FakeTrainer):
    """The current policy assigns every replayed token 1 nat more than the stored logprob."""

    def _compute_old_log_prob(self, batch):
        out, mfu = super()._compute_old_log_prob(batch)
        out.batch["old_log_probs"] = out.batch["old_log_probs"] + 1.0
        return out, mfu


def test_mass_cap_declares_groups_by_uid_and_verifies():
    r = replay(staleness_policy=StalenessPolicy(mass_cap=0.25), attest=AttestationLog())
    trainer = RisingTrainer(r, [])
    trainer.step(live_batch(), step=1)
    dead = mixed_batch()
    out = trainer.step(dead, step=2)
    assert r.stats["rescaled_rows"] == 2 and r.stats["replaced_rows"] == 2
    telemetry = [rec for rec in r.buffer.attestation_log.records if rec["op"] == "telemetry"][-1]
    assert telemetry["policy"]["group_size"] is None
    assert [d["group"] for d in telemetry["decisions"]] == [0, 0] and [d["row"] for d in telemetry["decisions"]] == [2, 3]
    assert telemetry["decisions"][0]["scale_den"] != "1"
    batch = r.last_replay
    for k, row in enumerate((2, 3)):
        expected = batch.rollouts[k].reward * float(batch.is_weights[k] * r.last_telemetry.decisions[k].scale)
        written = out.batch["advantages"][row][out.batch["response_mask"][row].bool()].tolist()   # per token in verl
        assert written and all(v == pytest.approx(expected) for v in written)
    verify_chain(r.buffer.attestation_log.records)


def test_age_bound_declines_and_evicts_with_reason_age():
    r = replay(staleness_policy=StalenessPolicy(max_age=0), attest=AttestationLog())
    trainer = FakeTrainer(r, [])
    trainer.step(live_batch(), step=1)
    dead = mixed_batch()
    out = trainer.step(dead, step=2)
    assert r.stats["declined_rows"] == 2 and r.stats["replaced_rows"] == 0
    assert out.batch["advantages"][2:].tolist() == dead.batch["advantages"][2:].tolist()
    records = r.buffer.attestation_log.records
    assert any(rec["op"] == "evict" and rec.get("reason") == "age" for rec in records)
    verify_chain(records)


def test_async_preset_with_negative_drift_keeps_rows_and_verifies():
    r = replay(staleness_policy="async", attest=AttestationLog())
    trainer = DriftingTrainer(r, [])
    trainer.step(live_batch(), step=1)
    trainer.step(mixed_batch(), step=2)
    assert r.stats["replaced_rows"] == 2 and r.stats["rescaled_rows"] == 0
    telemetry = [rec for rec in r.buffer.attestation_log.records if rec["op"] == "telemetry"][-1]
    assert telemetry["policy"]["ess_floor"] == (0.3).hex() and all(d["reason"] is None for d in telemetry["decisions"])
    verify_chain(r.buffer.attestation_log.records)


def test_policy_stages_need_telemetry():
    with pytest.raises(ValueError, match="telemetry=True"):
        replay(staleness_policy="conservative", telemetry=False)
