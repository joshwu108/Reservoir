"""Tests for the declared minimum entry age of ``RolloutBuffer.sample``.

Draws are keyed over the whole tree, so ``min_entry_age`` is honoured by
refusing to draw while any live entry is younger: such a sample is empty
and writes nothing. Every sample record carries ``min_entry_age`` and every
drawn slot its ``entry_version``, so the checker can confirm each draw was
at least that many versions old.
"""

from __future__ import annotations

import pytest

from checker.verify import verify_chain
from reservoir.attest import AttestationLog
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer


def rollouts(rewards: list[float]) -> list[Rollout]:
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in rewards]


def make_buffer(**kw) -> RolloutBuffer:
    defaults = dict(capacity=16, half_life=4, max_policy_age=8, seed=0, attest=AttestationLog())
    defaults.update(kw)
    return RolloutBuffer(**defaults)


def sample_records(buf: RolloutBuffer) -> list[dict]:
    return [r for r in buf.attestation_log.records if r["op"] == "sample"]


class TestValidation:
    def test_default_is_zero(self) -> None:
        assert make_buffer().min_entry_age == 0

    @pytest.mark.parametrize("bad", [-1, 1.0, True, "1", None])
    def test_must_be_a_non_negative_int(self, bad) -> None:
        with pytest.raises(ValueError, match="min_entry_age"):
            make_buffer(min_entry_age=bad)

    def test_above_max_policy_age_is_refused(self) -> None:
        with pytest.raises(ValueError, match="min_entry_age"):
            make_buffer(max_policy_age=4, min_entry_age=5)
        assert make_buffer(max_policy_age=4, min_entry_age=4).min_entry_age == 4


class TestEmptySample:
    def test_only_current_step_groups_yield_an_empty_sample(self) -> None:
        buf = make_buffer(min_entry_age=1)
        buf.add_group("p", 3, rollouts([1.0, 0.0]))
        before = list(buf.attestation_log.records)
        batch = buf.sample(4, current_version=3)
        assert batch.indices == () and batch.rollouts == () and batch.draw_integers == ()
        assert batch.op_counter == -1
        assert buf.attestation_log.records == before          # nothing written
        assert buf._op_counter == 0 and buf._draw_counter == 0

    def test_a_younger_entry_next_to_older_ones_empties_the_sample(self) -> None:
        buf = make_buffer(min_entry_age=1)
        buf.add_group("old", 1, rollouts([1.0, 0.0]))
        buf.add_group("new", 2, rollouts([1.0, 0.0]))
        assert buf.sample(4, current_version=2).indices == ()

    def test_empty_sample_cannot_be_witnessed(self) -> None:
        buf = make_buffer(min_entry_age=1)
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        batch = buf.sample(2, current_version=0)
        with pytest.raises(ValueError, match="latest sample"):
            buf.witness_batch(batch, step=0, batch_rows=2, rows=[], tensor_digest="0" * 64)

    def test_an_empty_sample_clears_last_batch(self) -> None:
        buf = make_buffer(min_entry_age=1)
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        assert buf.sample(2, current_version=1).indices
        buf.add_group("q", 1, rollouts([1.0, 0.0]))
        assert buf.sample(2, current_version=1).indices == ()
        assert buf.last_batch is None

    def test_empty_buffer_still_raises(self) -> None:
        with pytest.raises(RuntimeError, match="no live"):
            make_buffer(min_entry_age=1).sample(1, current_version=0)


class TestRecordedFields:
    def test_sample_record_carries_min_entry_age_and_entry_versions(self) -> None:
        buf = make_buffer(min_entry_age=1)
        buf.add_group("a", 0, rollouts([1.0, 0.0, 0.5]))
        buf.add_group("b", 1, rollouts([0.2, 0.9]))
        batch = buf.sample(6, current_version=2)              # every entry is at least one version old
        assert len(batch.indices) == 6
        (record,) = sample_records(buf)
        assert record["min_entry_age"] == "1"
        assert [s["entry_version"] for s in record["samples"]] == [str(v) for v in batch.model_versions]
        assert all(2 - v >= 1 for v in batch.model_versions)
        verify_chain(buf.attestation_log.records)

    def test_default_zero_is_recorded_and_same_step_draws_allowed(self) -> None:
        buf = make_buffer()
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        batch = buf.sample(3, current_version=0)
        (record,) = sample_records(buf)
        assert record["min_entry_age"] == "0"
        assert [s["entry_version"] for s in record["samples"]] == ["0", "0", "0"]
        assert len(batch.indices) == 3
        verify_chain(buf.attestation_log.records)

    def test_draws_are_unchanged_by_the_new_fields(self) -> None:
        def draws(min_age: int) -> tuple[int, ...]:
            buf = make_buffer(min_entry_age=min_age)
            buf.add_group("p", 0, rollouts([1.0, 0.0, 0.25]))
            return buf.sample(5, current_version=1).draw_integers

        assert draws(0) == draws(1)

    def test_sampling_resumes_once_young_entries_age(self) -> None:
        buf = make_buffer(min_entry_age=1)
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        assert buf.sample(2, current_version=0).indices == ()
        assert len(buf.sample(2, current_version=1).indices) == 2
        (record,) = sample_records(buf)
        assert record["op_counter"] == 1                     # the empty call consumed no counter
        verify_chain(buf.attestation_log.records)
