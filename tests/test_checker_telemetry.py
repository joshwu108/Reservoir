"""Tests for the checker's handling of telemetry records.

The effective sample size and the staleness of a replayed step are
functions of the sample record and the replayed buffer state, so the
checker recomputes them and rejects a record that disagrees; the counters
are cross-checked against the draws; carried float measurements must be
canonical and listed under ``reported``.
"""

from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from reservoir.attest import AttestationLog, _digest_record
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer
from reservoir.rollout_telemetry import exact_telemetry
from reservoir_checker.verify import CheckerError, verify_chain


def rollouts(rewards):
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in rewards]


def rechain(records, start=0):
    out = copy.deepcopy(records)
    prev = out[start - 1]["digest"] if start > 0 else "genesis"
    for rec in out[start:]:
        rec["prev_digest"] = prev
        rec["digest"] = _digest_record(rec)
        prev = rec["digest"]
    return out


COUNTS = dict(batch_rows=8, replaced_rows=3, declined_rows=0, dead_groups=1, near_dead_groups=0)


def telemetry_run(steps: int = 4):
    buf = RolloutBuffer(capacity=8, half_life=2, max_policy_age=6, seed=2, attest=AttestationLog())
    for step in range(steps):
        buf.add_group(f"g{step}", step, rollouts([1.0, 0.0, 0.5]), source="s")
        batch = buf.sample(3, current_version=step)
        buf.witness_batch(batch, step=step, batch_rows=8, rows=[5, 6, 7], tensor_digest="ab" * 32)
        buf.record_telemetry(step, COUNTS, batch, {"log_ratio_mean_abs": 0.125, "log_ratio_max_abs": 0.5})
    buf.advance(steps)   # the adapters advance to the step before they write its telemetry
    buf.record_telemetry(steps, dict(COUNTS, replaced_rows=0), None, {})
    return [dict(r) for r in buf.attestation_log.records], buf


def telemetry_indices(records):
    return [i for i, r in enumerate(records) if r["op"] == "telemetry"]


class TestExactTelemetry:
    def test_ess_and_staleness(self):
        t = exact_telemetry([Fraction(1), Fraction(1, 2), Fraction(1, 4)], [0, 3, 1])
        assert t.ess == Fraction(7, 4) ** 2 / Fraction(21, 16)
        assert t.staleness_max == 3 and t.staleness_sum == 4
        with pytest.raises(ValueError):
            exact_telemetry([], [])
        with pytest.raises(ValueError):
            exact_telemetry([Fraction(1)], [-1])


class TestAccepts:
    def test_run_verifies_and_exposes_points(self):
        records, _ = telemetry_run()
        result = verify_chain(records)
        points = result.content.telemetry
        assert [p.step for p in points] == [0, 1, 2, 3, 4]
        assert points[4].sample_op_counter is None and points[4].ess is None
        assert points[3].staleness_max >= 0 and points[3].reported == {"log_ratio_mean_abs": 0.125, "log_ratio_max_abs": 0.5}

    def test_record_shape(self):
        records, _ = telemetry_run(steps=1)
        rec = records[telemetry_indices(records)[0]]
        assert rec["step"] == "0" and rec["sample_op_counter"] == 1 and rec["batch_rows"] == 8
        assert rec["reported"] == ["log_ratio_max_abs", "log_ratio_mean_abs"]
        assert rec["log_ratio_mean_abs"] == (0.125).hex()


class TestRejects:
    def _edit(self, records, idx, edit):
        m = copy.deepcopy(records)
        edit(m[idx])
        return rechain(m, idx)

    def test_ess_forged(self):
        records, _ = telemetry_run()
        i = telemetry_indices(records)[0]
        with pytest.raises(CheckerError, match="ess"):
            verify_chain(self._edit(records, i, lambda r: r.update(ess_num=str(int(r["ess_num"]) + 1))))

    def test_staleness_forged(self):
        records, _ = telemetry_run()
        i = telemetry_indices(records)[2]
        with pytest.raises(CheckerError, match="staleness"):
            verify_chain(self._edit(records, i, lambda r: r.update(staleness_max=str(int(r["staleness_max"]) + 1))))

    def test_counts_disagree_with_draws(self):
        records, _ = telemetry_run()
        i = telemetry_indices(records)[0]
        with pytest.raises(CheckerError, match="drew 3 rollouts"):
            verify_chain(self._edit(records, i, lambda r: r.update(replaced_rows=2)))

    def test_rows_exceed_batch(self):
        records, _ = telemetry_run()
        i = telemetry_indices(records)[0]
        with pytest.raises(CheckerError, match="more replaced"):
            verify_chain(self._edit(records, i, lambda r: r.update(batch_rows=2)))

    def test_reported_list_must_match_extra_fields(self):
        records, _ = telemetry_run()
        i = telemetry_indices(records)[0]
        with pytest.raises(CheckerError, match="reported"):
            verify_chain(self._edit(records, i, lambda r: r.update(mystery=(1.0).hex())))
        with pytest.raises(CheckerError, match="canonical"):
            verify_chain(self._edit(records, i, lambda r: r.update(log_ratio_mean_abs="0x1p-3")))

    def test_names_a_non_latest_sample(self):
        records, _ = telemetry_run()
        i = telemetry_indices(records)[2]
        with pytest.raises(CheckerError, match="latest sample"):
            verify_chain(self._edit(records, i, lambda r: r.update(sample_op_counter=1)))

    def test_replayed_rows_without_a_sample(self):
        records, _ = telemetry_run()
        i = telemetry_indices(records)[-1]
        with pytest.raises(CheckerError, match="no sample is named"):
            verify_chain(self._edit(records, i, lambda r: r.update(replaced_rows=1)))

    def test_format_1_log_rejects_telemetry(self):
        records, _ = telemetry_run(steps=1)
        records[0]["format"] = "1"
        with pytest.raises(CheckerError, match="format-2"):
            verify_chain(rechain(records))

    def test_telemetry_must_precede_evictions_of_its_rows(self):
        # Staleness is recomputed from live slots: the adapter writes telemetry
        # before it evicts declined draws, and the reverse order is rejected.
        def run(evict_first: bool):
            buf = RolloutBuffer(capacity=8, attest=AttestationLog())
            buf.add_group("g", 0, rollouts([1.0, 0.0]), source="s")
            batch = buf.sample(2)
            buf.witness_batch(batch, step=0, batch_rows=4, rows=[3], tensor_digest="ab" * 32, declined=[0])
            counts = dict(COUNTS, batch_rows=4, replaced_rows=1, declined_rows=1)
            if evict_first:
                buf.evict(batch.indices[0], "drift")
                buf.record_telemetry(0, counts, batch, {})
            else:
                buf.record_telemetry(0, counts, batch, {})
                buf.evict(batch.indices[0], "drift")
            return buf.attestation_log.records

        assert len(verify_chain(run(evict_first=False)).content.telemetry) == 1
        with pytest.raises(CheckerError, match="no longer holds the example"):
            verify_chain(run(evict_first=True))


def swapped(records, first):
    """Swap records ``first`` and ``first + 1`` and re-chain from there."""
    m = copy.deepcopy(records)
    m[first], m[first + 1] = m[first + 1], m[first]
    return rechain(m, first)


class TestStepAndOrder:
    """``step`` is the buffer version the batch was built at, and a witness precedes its telemetry."""

    def _last_pair(self, records):
        b = max(i for i, r in enumerate(records) if r["op"] == "batch")
        assert records[b + 1]["op"] == "telemetry"
        return b, b + 1

    def test_witness_step_forged(self):
        records, _ = telemetry_run()
        b, _ = self._last_pair(records)
        m = copy.deepcopy(records)
        m[b]["step"] = "99"
        with pytest.raises(CheckerError, match="buffer version"):
            verify_chain(rechain(m, b))

    def test_witness_and_telemetry_step_forged_together(self):
        records, _ = telemetry_run()
        b, t = self._last_pair(records)
        m = copy.deepcopy(records)
        m[b]["step"] = m[t]["step"] = "99"
        with pytest.raises(CheckerError, match="buffer version"):
            verify_chain(rechain(m, b))

    def test_telemetry_without_sample_step_forged(self):
        records, _ = telemetry_run()
        t = telemetry_indices(records)[-1]
        m = copy.deepcopy(records)
        m[t]["step"] = str(int(m[t]["step"]) + 1)
        with pytest.raises(CheckerError, match="buffer version"):
            verify_chain(rechain(m, t))

    def test_telemetry_before_its_witness_is_rejected(self):
        records, _ = telemetry_run()
        b, _ = self._last_pair(records)
        with pytest.raises(CheckerError, match="has no batch witness"):
            verify_chain(swapped(records, b))

    def test_telemetry_before_the_first_witness_is_rejected(self):
        records, _ = telemetry_run()
        b = next(i for i, r in enumerate(records) if r["op"] == "batch")
        with pytest.raises(CheckerError, match="came without one"):
            verify_chain(swapped(records, b))

    def test_swap_and_rechain_with_disagreeing_decisions(self):
        # The forgery: the witness places a kept draw in a row the decisions do not name, and the
        # telemetry record is moved ahead of the witness so no cross-check sees both.
        from campaigns.mutation_staleness import build_staleness_chain

        records = build_staleness_chain()
        t = max(i for i, r in enumerate(records) if r["op"] == "telemetry")
        b = t - 1
        assert records[b]["op"] == "batch" and records[b]["replaced"]
        m = copy.deepcopy(records)
        m[b]["replaced"][0]["row"] = 0   # rows 4.. are the dead rows the decisions name
        with pytest.raises(CheckerError, match="disagree with the batch witness"):
            verify_chain(rechain(m, b))
        with pytest.raises(CheckerError, match="has no batch witness"):
            verify_chain(swapped(m, b))

    def test_telemetry_without_any_witness_still_verifies(self):
        # A log that never carries witnesses keeps verifying: the rule binds logs that do.
        buf = RolloutBuffer(capacity=8, attest=AttestationLog())
        buf.add_group("g", 0, rollouts([1.0, 0.0]), source="s")
        batch = buf.sample(2)
        buf.record_telemetry(0, dict(COUNTS, batch_rows=4, replaced_rows=2), batch, {})
        assert len(verify_chain(buf.attestation_log.records).content.telemetry) == 1

    def test_witnessed_log_cannot_drop_a_later_witness(self):
        records, _ = telemetry_run()
        b, _ = self._last_pair(records)
        m = copy.deepcopy(records)
        del m[b]
        with pytest.raises(CheckerError, match="has no batch witness"):
            verify_chain(rechain(m, b))

    def test_witnessed_step_cannot_hide_behind_sample_less_telemetry(self):
        # Rewriting a replayed step's telemetry as a step without replay would drop its ESS and staleness.
        records, _ = telemetry_run()
        _, t = self._last_pair(records)
        m = copy.deepcopy(records)
        for name in ("sample_op_counter", "ess_num", "ess_den", "staleness_max", "staleness_sum"):
            m[t].pop(name)
        m[t]["replaced_rows"] = 0
        with pytest.raises(CheckerError, match="must name the witnessed sample"):
            verify_chain(rechain(m, t))


class TestContentRequirement:
    def test_replayed_step_telemetry_needs_content_digests(self):
        records, _ = telemetry_run()
        stripped = [dict(r) for r in records if r["op"] != "batch"]
        for r in stripped:
            r.pop("content_digest", None)
            r.pop("source", None)
        with pytest.raises(CheckerError, match="telemetry of a replayed step needs content digests"):
            verify_chain(rechain(stripped))
