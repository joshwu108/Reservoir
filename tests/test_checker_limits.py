"""Resource bounds of the checker on hostile or very long logs.

A one-line ``sample`` record must not make the checker allocate per draw
without limit, verification time must grow linearly with the number of
witnessed steps, and integer strings past Python's digit limit must fail
as ``CheckerError`` like every other malformed field.
"""

from __future__ import annotations

import copy
import time
import tracemalloc

import pytest

from reservoir.attest import AttestationLog, _digest_record
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer
from reservoir_checker import verify as verify_module
from reservoir_checker.content import ContentState
from reservoir_checker.decay_replay import CheckerError, parse_config
from reservoir_checker.replay import log_format
from reservoir_checker.replay import replay_json_lines
from reservoir_checker.verify import MAX_SAMPLE_DRAWS, verify_chain, verify_json_lines

COUNTS = dict(batch_rows=4, replaced_rows=1, declined_rows=0, dead_groups=1, near_dead_groups=0)


def rechain(records, start=0):
    out = copy.deepcopy(records)
    prev = out[start - 1]["digest"] if start > 0 else "genesis"
    for rec in out[start:]:
        rec["prev_digest"] = prev
        rec["digest"] = _digest_record(rec)
        prev = rec["digest"]
    return out


def small_log():
    buf = RolloutBuffer(capacity=8, seed=1, attest=AttestationLog())
    buf.add_group("g", 0, [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in (1.0, 0.5)], source="s")
    buf.sample(2)
    return [dict(r) for r in buf.attestation_log.records]


def witnessed_log(steps: int):
    """One group, one sample, one witness and one telemetry record per step."""
    buf = RolloutBuffer(capacity=8, half_life=4, max_policy_age=3, seed=1, attest=AttestationLog())
    for step in range(steps):
        buf.add_group(f"g{step}", step, [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r)
                                         for r in (1.0, 0.5)], source="s")
        batch = buf.sample(1, current_version=step)
        buf.witness_batch(batch, step=step, batch_rows=4, rows=[3], tensor_digest="ab" * 32)
        buf.record_telemetry(step, COUNTS, batch, {})
    return [dict(r) for r in buf.attestation_log.records]


class TestSampleDrawCap:
    def test_over_cap_sample_raises_before_allocating_per_draw(self):
        records = small_log()
        s = next(i for i, r in enumerate(records) if r["op"] == "sample")
        record = dict(records[s], samples=[records[s]["samples"][0]] * (MAX_SAMPLE_DRAWS + 1))
        tree = verify_module._SumTree(8)
        tracemalloc.start()
        try:
            with pytest.raises(CheckerError, match="exceeds the checker's limit"):
                verify_module._verify_sample(record, s, tree, {})
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 64 * 1024, f"{peak} bytes traced before the cap fired"

    def test_cap_fires_in_verify_chain(self, monkeypatch):
        monkeypatch.setattr(verify_module, "MAX_SAMPLE_DRAWS", 1)
        with pytest.raises(CheckerError, match="exceeds the checker's limit of 1"):
            verify_chain(small_log())

    def test_at_cap_is_accepted(self, monkeypatch):
        monkeypatch.setattr(verify_module, "MAX_SAMPLE_DRAWS", 2)
        verify_chain(small_log())

    def test_content_state_refuses_an_over_cap_sample_on_its_own(self, monkeypatch):
        records = small_log()
        s = next(i for i, r in enumerate(records) if r["op"] == "sample")
        monkeypatch.setattr(verify_module, "MAX_SAMPLE_DRAWS", 1)
        from reservoir_checker import content as content_module
        monkeypatch.setattr(content_module, "MAX_SAMPLE_DRAWS", 1)
        with pytest.raises(CheckerError, match="exceeds the checker's limit"):
            ContentState().on_sample(s, records[s])


class TestLinearTime:
    def test_verify_time_is_linear_in_witnessed_steps(self):
        full = witnessed_log(8000)
        cuts = {}
        for n in (2000, 4000, 8000):
            # A cut right after a telemetry record is a complete log of n steps.
            telemetry = [i for i, r in enumerate(full) if r["op"] == "telemetry"]
            cuts[n] = full[: telemetry[n - 1] + 1]
        timings = {}
        for n, records in cuts.items():
            best = float("inf")
            for _ in range(5):
                start = time.process_time()
                verify_chain(records)
                best = min(best, time.process_time() - start)
            timings[n] = best
        # Linear growth gives a ratio near 2 per doubling; the quadratic lookup gave about 3.
        assert timings[4000] / timings[2000] < 2.8, timings
        assert timings[8000] / timings[2000] < 5.6, timings


class TestBigIntegers:
    def _config(self, **overrides):
        buf = RolloutBuffer(capacity=8, seed=1, attest=AttestationLog())
        return dict(buf.attestation_log.records[0], **overrides)

    def test_five_thousand_digit_capacity_raises_checker_error(self):
        record = self._config(capacity="1" * 5000)
        with pytest.raises(CheckerError, match="capacity"):
            parse_config(record, 0)
        with pytest.raises(CheckerError, match="capacity"):
            verify_chain(rechain([record]))

    def test_five_thousand_digit_advance_version_raises_checker_error(self):
        records = small_log()
        records.append({"op": "advance_version", "old_version": "0", "new_version": "9" * 5000,
                        "op_counter": 99})
        with pytest.raises(CheckerError, match="new_version"):
            verify_chain(rechain(records, len(records) - 1))

    def test_five_thousand_digit_format_raises_checker_error(self):
        with pytest.raises(CheckerError, match="format"):
            log_format([self._config(format="1" * 5000)])

    def test_five_thousand_digit_json_literal_raises_checker_error(self):
        line = '{"op": "decay_config", "capacity": ' + "1" * 5000 + "}"
        with pytest.raises(CheckerError, match="line 1: not valid JSON"):
            verify_json_lines(line)
        with pytest.raises(CheckerError, match="line 1: not valid JSON"):
            replay_json_lines(line, "")
