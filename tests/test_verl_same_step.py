"""The verl adapter never replays a rollout stored at the step it is training.

Same rule as the TRL adapter: dead rows are replaced before the step's live
groups are stored and the buffer is built with ``min_entry_age=1``.
"""

from __future__ import annotations

from checker.verify import verify_chain
from reservoir.attest import AttestationLog
from tests.test_verl_replay import FakeTrainer, live_batch, mixed_batch, replay


def samples(log: AttestationLog) -> list[dict]:
    return [r for r in log.records if r["op"] == "sample"]


def test_the_adapter_builds_its_buffer_with_a_floor_of_one():
    assert replay().buffer.min_entry_age == 1


def test_first_step_dead_rows_stay_dead_although_live_rows_were_stored():
    r = replay(attest=AttestationLog())
    batch = mixed_batch()
    out = FakeTrainer(r, []).step(batch, step=1)

    assert out is batch and r.last_replay is None and r.stats["replaced_rows"] == 0
    assert r.buffer.size == 2
    assert samples(r.buffer.attestation_log) == []


def test_replayed_rows_come_only_from_earlier_steps():
    log = AttestationLog()
    r = replay(attest=log)
    trainer = FakeTrainer(r, [])
    trainer.step(live_batch(), step=1)
    trainer.step(mixed_batch(), step=2)
    trainer.step(mixed_batch(), step=3)

    witnesses = [x for x in log.records if x["op"] == "batch"]
    assert [w["step"] for w in witnesses] == ["2", "3"]
    for witness, record in zip(witnesses, samples(log)):
        step = int(witness["step"])
        assert record["min_entry_age"] == "1"
        assert all(int(s["entry_version"]) <= step - 1 for s in record["samples"])
    verify_chain(log.records)


def test_a_second_update_at_the_same_step_leaves_its_dead_rows_dead():
    r = replay(attest=AttestationLog())
    trainer = FakeTrainer(r, [])
    trainer.step(live_batch(), step=1)
    trainer.step(mixed_batch(), step=2)
    replaced = r.stats["replaced_rows"]

    batch = mixed_batch()
    out = trainer.step(batch, step=2)

    assert out is batch and r.stats["replaced_rows"] == replaced
    assert r.stats["age_floor_skips"] == 1


def test_min_entry_age_zero_is_recorded():
    log = AttestationLog()
    r = replay(attest=log, min_entry_age=0)
    trainer = FakeTrainer(r, [])
    trainer.step(live_batch(), step=1)
    trainer.step(mixed_batch(), step=1)

    (record,) = samples(log)
    assert record["min_entry_age"] == "0"
    assert {s["entry_version"] for s in record["samples"]} == {"1"}


def test_a_floor_above_one_is_refused_because_every_sample_would_be_empty():
    import pytest

    with pytest.raises(ValueError, match="min_entry_age"):
        replay(min_entry_age=2)
