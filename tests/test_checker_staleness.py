"""Tests for recording and verifying staleness decisions in telemetry records.

The adapter writes, per replayed step, the policy, every draw's log-ratio
as a hex float and one decision per draw; the checker replays the policy
from those inputs plus the sample record's weights and the live slots'
ages, cross-checks the decisions against the batch witness and the
counters, and recomputes the reported log-ratio statistics. Every forgery
below must be rejected; the pre-policy record shape must still verify.
"""

from __future__ import annotations

import copy
import math
from fractions import Fraction

import pytest

from reservoir.attest import AttestationLog, TELEMETRY_RESERVED, _digest_record
from reservoir.integrations._trl_staleness import StalenessPolicy, decide
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer
from reservoir_checker.verify import CheckerError, verify_chain

G = 2  # rows per training group in these runs


def rechain(records, start=0):
    out = copy.deepcopy(records)
    prev = out[start - 1]["digest"] if start > 0 else "genesis"
    for rec in out[start:]:
        rec["prev_digest"] = prev
        rec["digest"] = _digest_record(rec)
        prev = rec["digest"]
    return out


def edited(records, idx, edit):
    m = copy.deepcopy(records)
    edit(m[idx])
    return rechain(m, idx)


def rollouts(rewards, base=1):
    return [Rollout(tokens=[base, i + 2], logprobs=[-0.1, -0.2], reward=r) for i, r in enumerate(rewards)]


def stats(ratios):
    finite = [abs(r) for r in ratios if math.isfinite(r)]
    return {"log_ratio_mean_abs": sum(finite) / len(finite), "log_ratio_max_abs": max(finite)} if finite else {}


DEFAULT_POLICY = StalenessPolicy(max_age=1, ess_floor=0.1, mass_cap=0.25)
# Draw 2 is non-finite (declined as drift every step); both groups carry more mass than fresh rows would, so
# every step has a decline and a rescale to forge.
DEFAULT_RATIOS = [0.0, math.log(8.0), float("nan"), 3.0]


def policy_run(policy=DEFAULT_POLICY, ratios=None, group_size=G, steps=3):
    """A buffer run that replays rows under ``policy`` with scripted log-ratios; returns (records, buffer)."""
    buf = RolloutBuffer(capacity=16, half_life=2, max_policy_age=8, seed=3, attest=AttestationLog())
    ratios = ratios if ratios is not None else DEFAULT_RATIOS
    for step in range(steps):
        # Moderate rewards keep the importance weights near 1/2; a zero reward would pin every other
        # weight near zero and make the mass cap vacuous.
        buf.add_group(f"g{step}", step, rollouts([1.0, 0.8, 0.5], base=step + 1), source="s")
        if step > buf.current_version:
            buf.advance(step)
        batch = buf.sample(len(ratios), current_version=step)
        ages = [buf.current_version - v for v in batch.model_versions]
        rows = list(range(4, 4 + len(ratios)))
        decisions = decide(policy, ratios=ratios, is_weights=batch.is_weights, ages=ages, rows=rows,
                           groups=[r // group_size for r in rows])
        kept = [d for d in decisions if d.reason is None]
        declined = [d.draw for d in decisions if d.reason is not None]
        buf.witness_batch(batch, step=step, batch_rows=8, rows=[d.row for d in kept], tensor_digest="ab" * 32,
                          declined=declined)
        counts = dict(batch_rows=8, replaced_rows=len(kept), declined_rows=len(declined), dead_groups=2,
                      near_dead_groups=0)
        buf.record_telemetry(step, counts, batch, stats(ratios), log_ratios=ratios,
                             policy=policy.to_record(group_size), decisions=[d.to_record() for d in decisions])
        kept_slots = {batch.indices[k.draw] for k in kept}
        for slot in sorted({batch.indices[d.draw] for d in decisions if d.reason in ("drift", "age")} - kept_slots):
            buf.evict(slot, next(d.reason for d in decisions if batch.indices[d.draw] == slot and d.reason in ("drift", "age")))
    return [dict(r) for r in buf.attestation_log.records], buf


def telemetry_indices(records):
    return [i for i, r in enumerate(records) if r["op"] == "telemetry"]


# ---------------------------------------------------------------------------
# Record shape and acceptance
# ---------------------------------------------------------------------------

class TestRecord:
    def test_fields_are_additive_and_reserved(self):
        assert {"log_ratios", "policy", "decisions"} <= TELEMETRY_RESERVED
        records, _ = policy_run()
        rec = records[telemetry_indices(records)[0]]
        assert rec["log_ratios"] == [(0.0).hex(), math.log(8.0).hex(), "nan", (3.0).hex()]
        assert rec["policy"]["ess_floor"] == (0.1).hex() and rec["policy"]["group_size"] == G
        assert [d["draw"] for d in rec["decisions"]] == [0, 1, 2, 3]
        assert set(rec["decisions"][0]) == {"draw", "row", "group", "reason", "scale_num", "scale_den"}
        assert rec["reported"] == ["log_ratio_max_abs", "log_ratio_mean_abs"]

    def test_verifies_and_exposes_decisions(self):
        records, _ = policy_run()
        result = verify_chain(records)
        points = result.content.telemetry
        assert len(points) == 3 and points[0].policy is not None
        assert len(points[0].decisions) == 4 and len(points[0].log_ratios) == 4
        assert points[0].log_ratios[1] == math.log(8.0) and math.isnan(points[0].log_ratios[2])
        # Both groups are mass-capped (ratios 0 and ln 8; 3.0 alone); the decisions carry the exact scales.
        scales = {d.draw: d.scale for d in points[0].decisions if d.reason is None}
        assert all(s < 1 for s in scales.values()) and points[0].decisions[2].reason == "drift"

    def test_non_finite_ratios_are_recorded_and_declined(self):
        records, _ = policy_run(ratios=[float("nan"), 0.0, float("-inf"), 0.0])
        rec = records[telemetry_indices(records)[0]]
        assert rec["log_ratios"][0] == "nan" and rec["log_ratios"][2] == "-inf"
        assert [d["reason"] for d in rec["decisions"]][0::2] == ["drift", "drift"]
        verify_chain(records)

    def test_age_evictions_verify(self):
        records, _ = policy_run(policy=StalenessPolicy(max_age=0), steps=3)
        assert any(r["op"] == "evict" and r.get("reason") == "age" for r in records)
        verify_chain(records)

    def test_legacy_record_shape_without_policy_still_verifies(self):
        # The 0.6.0 shape: reported stats, no log_ratios, declines from the old gate.
        buf = RolloutBuffer(capacity=8, attest=AttestationLog())
        buf.add_group("g", 0, rollouts([1.0, 0.0]), source="s")
        batch = buf.sample(2)
        buf.witness_batch(batch, step=0, batch_rows=4, rows=[3], tensor_digest="ab" * 32, declined=[0])
        buf.record_telemetry(0, dict(batch_rows=4, replaced_rows=1, declined_rows=1, dead_groups=1, near_dead_groups=0),
                             batch, {"log_ratio_mean_abs": 0.5, "log_ratio_max_abs": 0.75})
        buf.evict(batch.indices[0], "drift")
        point = verify_chain(buf.attestation_log.records).content.telemetry[0]
        assert point.policy is None and point.decisions == () and point.log_ratios == ()

    def test_log_ratios_without_policy_verify_the_reported_stats(self):
        buf = RolloutBuffer(capacity=8, attest=AttestationLog())
        buf.add_group("g", 0, rollouts([1.0, 0.0]), source="s")
        batch = buf.sample(2)
        buf.witness_batch(batch, step=0, batch_rows=4, rows=[2, 3], tensor_digest="ab" * 32)
        counts = dict(batch_rows=4, replaced_rows=2, declined_rows=0, dead_groups=1, near_dead_groups=0)
        buf.record_telemetry(0, counts, batch, stats([0.25, -1.0]), log_ratios=[0.25, -1.0])
        records = [dict(r) for r in buf.attestation_log.records]
        verify_chain(records)
        i = telemetry_indices(records)[0]
        with pytest.raises(CheckerError, match="log_ratio_mean_abs"):
            verify_chain(edited(records, i, lambda r: r.update(log_ratio_mean_abs=(0.5).hex())))
        with pytest.raises(CheckerError, match="log_ratio_max_abs"):
            verify_chain(edited(records, i, lambda r: r.update(log_ratio_max_abs=(2.0).hex())))
        with pytest.raises(CheckerError, match="log_ratio_max_abs"):
            verify_chain(edited(records, i, lambda r: (r.pop("log_ratio_max_abs"), r.update(reported=["log_ratio_mean_abs"]))))

    def test_writer_validates_inputs(self):
        buf = RolloutBuffer(capacity=8, attest=AttestationLog())
        buf.add_group("g", 0, rollouts([1.0, 0.0]), source="s")
        batch = buf.sample(2)
        counts = dict(batch_rows=4, replaced_rows=2, declined_rows=0, dead_groups=1, near_dead_groups=0)
        policy = StalenessPolicy(max_age=3).to_record(G)
        decisions = [d.to_record() for d in decide(StalenessPolicy(max_age=3), ratios=[0.0, 0.0],
                                                   is_weights=batch.is_weights, ages=[0, 0], rows=[2, 3], groups=[1, 1])]
        with pytest.raises(ValueError, match="one log-ratio per draw"):
            buf.record_telemetry(0, counts, batch, {}, log_ratios=[0.0])
        with pytest.raises(ValueError, match="log_ratios"):
            buf.record_telemetry(0, counts, batch, {}, policy=policy, decisions=decisions)
        with pytest.raises(ValueError, match="decisions"):
            buf.record_telemetry(0, counts, batch, {}, log_ratios=[0.0, 0.0], policy=policy)
        with pytest.raises(ValueError, match="sample"):
            buf.record_telemetry(0, dict(counts, replaced_rows=0), None, {}, log_ratios=[0.0, 0.0])
        with pytest.raises(ValueError, match="policy"):
            buf.record_telemetry(0, counts, batch, {}, log_ratios=[0.0, 0.0], policy={"max_age": 3}, decisions=decisions)
        with pytest.raises(ValueError, match="evict reason"):
            buf.evict(batch.indices[0], "ess")


# ---------------------------------------------------------------------------
# Forgeries
# ---------------------------------------------------------------------------

class TestRejects:
    @pytest.fixture
    def run(self):
        records, _ = policy_run()
        return records, telemetry_indices(records)[0]

    def test_decision_reason_flipped(self, run):
        records, i = run
        kept = next(k for k, d in enumerate(records[i]["decisions"]) if d["reason"] is None)
        with pytest.raises(CheckerError, match="declared age but the policy gives kept"):
            verify_chain(edited(records, i, lambda r: r["decisions"][kept].update(reason="age", scale_num="1", scale_den="1")))
        with pytest.raises(CheckerError, match="declined draw carries scale 1"):
            verify_chain(edited(records, i, lambda r: r["decisions"][kept].update(reason="age")))

    def test_decline_made_a_keep(self, run):
        records, i = run
        rec = records[i]
        declined = next(k for k, d in enumerate(rec["decisions"]) if d["reason"] is not None)
        # Making the draw kept also has to be reflected in the witness and counters to get past them.
        def edit(r):
            r["decisions"][declined]["reason"] = None
        with pytest.raises(CheckerError):
            verify_chain(edited(records, i, edit))

    def test_scale_changed(self, run):
        records, i = run
        scaled = next(k for k, d in enumerate(records[i]["decisions"]) if d["scale_den"] != "1")
        with pytest.raises(CheckerError, match="declared scale"):
            verify_chain(edited(records, i, lambda r: r["decisions"][scaled].update(scale_num="1", scale_den="1")))

    def test_scale_not_reduced(self, run):
        records, i = run
        scaled = next(k for k, d in enumerate(records[i]["decisions"]) if d["scale_den"] != "1")
        def edit(r):
            d = r["decisions"][scaled]
            d["scale_num"], d["scale_den"] = str(int(d["scale_num"]) * 2), str(int(d["scale_den"]) * 2)
        with pytest.raises(CheckerError, match="reduced"):
            verify_chain(edited(records, i, edit))

    def test_log_ratio_changed(self, run):
        records, i = run
        # Lowering the drifted ratio changes the mass cap's scale; the recorded decisions no longer follow.
        with pytest.raises(CheckerError):
            verify_chain(edited(records, i, lambda r: r["log_ratios"].__setitem__(1, (0.5).hex())))

    def test_log_ratio_not_canonical(self, run):
        records, i = run
        with pytest.raises(CheckerError, match="canonical"):
            verify_chain(edited(records, i, lambda r: r["log_ratios"].__setitem__(0, "0x0p+0")))

    def test_policy_parameter_changed(self, run):
        records, i = run
        with pytest.raises(CheckerError):
            verify_chain(edited(records, i, lambda r: r["policy"].update(mass_cap=(10.0).hex())))
        with pytest.raises(CheckerError, match="policy"):
            verify_chain(edited(records, i, lambda r: r["policy"].update(bonus=1)))
        with pytest.raises(CheckerError, match="no stage"):
            verify_chain(edited(records, i, lambda r: r["policy"].update(max_age=None, ess_floor=None, mass_cap=None)))

    def test_group_disagrees_with_group_size(self, run):
        records, i = run
        with pytest.raises(CheckerError, match="group"):
            verify_chain(edited(records, i, lambda r: r["decisions"][0].update(group=9)))

    def test_rows_disagree_with_witness(self, run):
        # Swapping the rows of two kept draws of one group leaves every policy decision intact (same group,
        # same mass) but the witness says which row holds which draw.
        records, i = run
        a, b = [k for k, d in enumerate(records[i]["decisions"]) if d["reason"] is None and d["group"] == 2]
        def edit(r):
            r["decisions"][a]["row"], r["decisions"][b]["row"] = r["decisions"][b]["row"], r["decisions"][a]["row"]
        with pytest.raises(CheckerError, match="witness"):
            verify_chain(edited(records, i, edit))

    def test_row_outside_the_batch(self, run):
        records, i = run
        with pytest.raises(CheckerError, match="outside the batch"):
            verify_chain(edited(records, i, lambda r: (r["decisions"][0].update(row=40, group=20))))

    def test_decisions_missing_or_short(self, run):
        records, i = run
        with pytest.raises(CheckerError, match="decisions"):
            verify_chain(edited(records, i, lambda r: r.pop("decisions")))
        with pytest.raises(CheckerError, match="one entry per draw"):
            verify_chain(edited(records, i, lambda r: r["decisions"].pop()))

    def test_log_ratios_missing_with_policy(self, run):
        records, i = run
        with pytest.raises(CheckerError, match="log_ratios"):
            verify_chain(edited(records, i, lambda r: r.pop("log_ratios")))

    def test_policy_dropped_but_declines_kept(self, run):
        records, i = run
        with pytest.raises(CheckerError, match="declined"):
            verify_chain(edited(records, i, lambda r: (r.pop("policy"), r.pop("decisions"))))

    def test_declined_counter_disagrees(self, run):
        records, i = run
        with pytest.raises(CheckerError):
            verify_chain(edited(records, i, lambda r: r.update(declined_rows=r["declined_rows"] + 1,
                                                                replaced_rows=r["replaced_rows"] - 1)))

    def test_declines_exceed_the_policy_cap(self, run):
        records, i = run
        with pytest.raises(CheckerError, match="max_declines_per_step"):
            verify_chain(edited(records, i, lambda r: r["policy"].update(max_declines_per_step=0)))

    def test_policy_on_a_step_without_a_sample(self):
        buf = RolloutBuffer(capacity=8, attest=AttestationLog())
        buf.add_group("g", 0, rollouts([1.0, 0.0]), source="s")
        buf.record_telemetry(0, dict(batch_rows=4, replaced_rows=0, declined_rows=0, dead_groups=0, near_dead_groups=0))
        records = [dict(r) for r in buf.attestation_log.records]
        i = telemetry_indices(records)[0]
        with pytest.raises(CheckerError, match="sample"):
            verify_chain(edited(records, i, lambda r: r.update(log_ratios=[])))

    def test_reported_field_cannot_reuse_a_policy_name(self):
        buf = RolloutBuffer(capacity=8, attest=AttestationLog())
        buf.add_group("g", 0, rollouts([1.0, 0.0]), source="s")
        batch = buf.sample(2)
        with pytest.raises(ValueError, match="reserved"):
            buf.record_telemetry(0, dict(batch_rows=4, replaced_rows=2, declined_rows=0, dead_groups=1,
                                         near_dead_groups=0), batch, {"decisions": 1.0})

    def test_age_evict_is_a_recorded_reason(self):
        records, _ = policy_run(policy=StalenessPolicy(max_age=0))
        i = next(k for k, r in enumerate(records) if r["op"] == "evict" and r.get("reason") == "age")
        with pytest.raises(CheckerError, match="reason"):
            verify_chain(edited(records, i, lambda r: r.update(reason="ess")))
