"""Tests for the staleness policy in the TRL adapter's replay path.

``ReservoirReplay(staleness_policy=...)`` declines replayed rows by age,
ESS floor or the legacy log-ratio gate and rescales a group's advantages
under the mass cap. Every decision reaches the batch (the written
advantage is ``reward * float(w * scale)``, the dead slot of a declined
draw stays dead), the batch witness digests what was written, the
telemetry record carries the policy, the log-ratios and the decisions,
and the checker verifies all of it.
"""

from __future__ import annotations

import copy
from fractions import Fraction

import pytest
import torch

from reservoir.attest import AttestationLog
from reservoir.integrations._trl_staleness import StalenessPolicy
from reservoir.integrations._trl_telemetry import METRIC_PREFIX
from reservoir.integrations.trl import ReservoirReplay
from reservoir_checker.verify import verify_chain
from tests.test_trl_replay import LOGP, live_batch, mixed_batch, replay
from tests.test_trl_telemetry import MetricTrainer

DEAD_ROWS = (2, 3)   # the dead group of mixed_batch; G = 2 so both are group 1


def records_of(r):
    return r.buffer.attestation_log.records


def telemetry_of(r):
    return [rec for rec in records_of(r) if rec["op"] == "telemetry"][-1]


def drifted_run(policy, drift: float, seed: int = 0):
    """Store live_batch at step 1, move the policy by ``drift`` nats per token, replay into mixed_batch at step 2."""
    r = replay(attest=AttestationLog(), staleness_policy=policy, seed=seed)
    trainer = MetricTrainer(r, [live_batch(), copy.deepcopy(mixed_batch())])
    trainer.generate(step=1)
    trainer.current_logp = LOGP + drift
    out = trainer.generate(step=2)
    return r, trainer, out


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_presets_by_name_and_policy_object():
    assert replay(staleness_policy="conservative").policy == StalenessPolicy.preset("conservative")
    assert replay(staleness_policy="off").policy == StalenessPolicy() and not replay().policy.active
    custom = StalenessPolicy(ess_floor=0.5, mass_cap=0.25)
    assert replay(staleness_policy=custom).policy == custom
    with pytest.raises(ValueError, match="preset"):
        replay(staleness_policy="strict")
    with pytest.raises(TypeError, match="staleness_policy"):
        replay(staleness_policy=0.5)


def test_legacy_keywords_fill_the_policys_last_stage_or_conflict():
    r = replay(max_log_ratio=1.5, max_declines_per_step=2)
    assert r.policy == StalenessPolicy(max_log_ratio=1.5, max_declines_per_step=2)
    assert r.max_log_ratio == 1.5 and r.max_declines_per_step == 2 and r.policy.legacy_only
    merged = replay(staleness_policy="conservative", max_log_ratio=3.0)
    assert merged.policy.max_log_ratio == 3.0 and merged.policy.ess_floor == 0.5
    with pytest.raises(ValueError, match="one place"):
        replay(staleness_policy=StalenessPolicy(max_log_ratio=1.0), max_log_ratio=2.0)


def test_policy_stages_need_telemetry_but_the_legacy_gate_does_not():
    with pytest.raises(ValueError, match="telemetry=True"):
        replay(staleness_policy="conservative", telemetry=False)
    assert replay(max_log_ratio=1.0, telemetry=False).policy.legacy_only


# ---------------------------------------------------------------------------
# Stages through the hook
# ---------------------------------------------------------------------------

def test_age_bound_declines_evicts_with_reason_age_and_verifies():
    r, trainer, out = drifted_run(StalenessPolicy(max_age=0), drift=0.0)   # rows stored at step 1 are age 1 at step 2
    assert r.stats["declined_rows"] == 2 and r.stats["replaced_rows"] == 0
    assert out["advantages"][2].item() == 0.0 and out["advantages"][3].item() == 0.0
    records = records_of(r)
    evictions = [rec for rec in records if rec["op"] == "evict" and rec.get("reason") == "age"]
    assert sorted(rec["index"] for rec in evictions) == sorted(set(r.last_replay.indices))
    assert not any(rec.get("reason") == "drift" for rec in records)
    witness = next(rec for rec in records if rec["op"] == "batch")
    assert witness["replaced"] == [] and witness["declined"] == [0, 1]
    telemetry = telemetry_of(r)
    assert telemetry["policy"]["max_age"] == 0 and telemetry["policy"]["group_size"] == 2
    assert [d["reason"] for d in telemetry["decisions"]] == ["age", "age"]
    assert [d["row"] for d in telemetry["decisions"]] == list(DEAD_ROWS) and {d["group"] for d in telemetry["decisions"]} == {1}
    assert trainer._metrics["train"][METRIC_PREFIX + "declined_age"][-1] == 2.0
    verify_chain(records)


def test_mass_cap_rescales_the_group_and_the_witness_digests_the_rescaled_advantages():
    policy = StalenessPolicy(mass_cap=0.25)
    r, trainer, out = drifted_run(policy, drift=0.5)        # every replayed row has exp(0.5 n) > 1: the group is over mass
    batch = r.last_replay
    point = r.last_telemetry
    assert r.stats["rescaled_rows"] == 2 and r.stats["declined_rows"] == 0 and r.stats["replaced_rows"] == 2
    scales = [d.scale for d in point.decisions]
    assert scales[0] == scales[1] < 1 and isinstance(scales[0], Fraction)
    for k, row in enumerate(DEAD_ROWS):
        expected = batch.rollouts[k].reward * float(batch.is_weights[k] * scales[k])
        assert out["advantages"][row].item() == pytest.approx(expected, rel=0, abs=0) or \
            out["advantages"][row].item() == torch.tensor(expected, dtype=out["advantages"].dtype).item()
    metrics = trainer._metrics["train"]
    assert metrics[METRIC_PREFIX + "rescaled_rows"][-1] == 2.0
    assert metrics[METRIC_PREFIX + "mass_scale_min"][-1] == pytest.approx(float(scales[0]))
    telemetry = telemetry_of(r)
    assert telemetry["decisions"][0]["scale_den"] != "1" and telemetry["policy"]["mass_cap"] == (0.25).hex()
    verify_chain(records_of(r))
    # The same draws without the policy write larger advantages, so the witness's tensor digest differs.
    plain, _, plain_out = drifted_run(StalenessPolicy(), drift=0.5)
    assert plain.last_replay.indices == batch.indices
    assert abs(plain_out["advantages"][2].item()) > abs(out["advantages"][2].item())
    digest = next(rec for rec in records_of(r) if rec["op"] == "batch")["tensor_digest"]
    plain_digest = next(rec for rec in records_of(plain) if rec["op"] == "batch")["tensor_digest"]
    assert digest != plain_digest
    assert "policy" not in telemetry_of(plain) and "log_ratios" in telemetry_of(plain)


def test_ess_floor_declines_without_evicting():
    for seed in range(40):
        r, trainer, out = drifted_run(StalenessPolicy(ess_floor=1.0), drift=0.0, seed=seed)
        batch = r.last_replay
        if batch.is_weights[0] != batch.is_weights[1]:
            break
    else:
        raise AssertionError("no seed drew two rows with different importance weights")
    assert r.stats["declined_rows"] == 1 and r.stats["replaced_rows"] == 1
    records = records_of(r)
    assert not any(rec["op"] == "evict" for rec in records)       # an ess decline keeps the entry
    telemetry = telemetry_of(r)
    reasons = [d["reason"] for d in telemetry["decisions"]]
    assert sorted(reasons, key=str) == ["ess", "None"] or reasons.count("ess") == 1
    declined = reasons.index("ess")
    assert out["advantages"][DEAD_ROWS[declined]].item() == 0.0
    assert trainer._metrics["train"][METRIC_PREFIX + "declined_ess"][-1] == 1.0
    verify_chain(records)


def test_legacy_gate_now_records_its_policy_and_decisions():
    r, _, _ = drifted_run(StalenessPolicy(max_log_ratio=0.1), drift=-1.0)
    telemetry = telemetry_of(r)
    assert telemetry["policy"]["max_log_ratio"] == (0.1).hex() and telemetry["policy"]["ess_floor"] is None
    assert [d["reason"] for d in telemetry["decisions"]] == ["drift", "drift"]
    assert all(rec.get("reason") == "drift" for rec in records_of(r) if rec["op"] == "evict")
    verify_chain(records_of(r))


def test_policy_off_records_log_ratios_and_keeps_non_finite_rows():
    r = replay(attest=AttestationLog())
    second = mixed_batch(old_logps=[[-0.1], [-0.2, -0.3], [-0.4, -0.5], [-0.6]])
    trainer = MetricTrainer(r, [live_batch(), second])
    trainer.generate(step=1)
    trainer.current_logp = float("nan")
    trainer.generate(step=2)
    telemetry = telemetry_of(r)
    assert telemetry["log_ratios"] == ["nan", "nan"] and "policy" not in telemetry and "reported" in telemetry
    assert r.stats["replaced_rows"] == 2 and r.stats["declined_rows"] == 0
    verify_chain(records_of(r))


def test_too_many_declines_raise_before_anything_is_written():
    r = replay(attest=AttestationLog(), staleness_policy=StalenessPolicy(max_age=0, max_declines_per_step=1))
    trainer = MetricTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=1)
    before = len(records_of(r))
    with pytest.raises(RuntimeError, match="max_declines_per_step=1"):
        trainer.generate(step=2)
    # The sample record was written (the draw happened); no witness or telemetry followed it.
    assert all(rec["op"] not in ("batch", "telemetry") for rec in records_of(r)[before:])


def test_conservative_preset_end_to_end_verifies():
    r = replay(attest=AttestationLog(), staleness_policy="conservative")
    trainer = MetricTrainer(r, [live_batch(), mixed_batch(), live_batch(30), mixed_batch(), mixed_batch()])
    for step in range(1, 6):
        trainer.generate(step=step)
    result = verify_chain(records_of(r))
    points = [p for p in result.content.telemetry if p.policy is not None]
    assert points and all(len(p.decisions) == len(p.log_ratios) for p in points)
    assert r.stats["replaced_rows"] + r.stats["declined_rows"] == 2 * 3
