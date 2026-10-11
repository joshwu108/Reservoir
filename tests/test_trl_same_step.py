"""The TRL adapter never replays a rollout stored at the step it is training.

The hook samples for the dead rows before it stores the step's live groups,
and the buffer is built with ``min_entry_age=1``, so a replayed row is never
a copy of a row already in the batch. Every sample record declares the
floor and every drawn slot its ``entry_version``.
"""

from __future__ import annotations

import copy

from checker.verify import verify_chain
from reservoir.attest import AttestationLog
from tests.test_trl_replay import FakeTrainer, live_batch, mixed_batch, replay


def inserted_at(log: AttestationLog, step: int) -> set[str]:
    return {r["content_digest"] for r in log.records
            if r["op"] == "insert" and int(r["entry_version"]) == step}


def samples(log: AttestationLog) -> list[dict]:
    return [r for r in log.records if r["op"] == "sample"]


def test_the_adapter_builds_its_buffer_with_a_floor_of_one():
    assert replay().buffer.min_entry_age == 1


def test_first_step_dead_rows_stay_dead_although_live_rows_were_stored():
    r = replay(attest=AttestationLog())
    batch = mixed_batch()
    out = FakeTrainer(r, [batch]).generate(step=0)

    assert r.last_replay is None and r.stats["replaced_rows"] == 0
    assert r.buffer.size == 2                       # the live group is stored for later steps
    assert torch_equal(out["advantages"], batch["advantages"])
    assert samples(r.buffer.attestation_log) == []


def test_replayed_rows_come_only_from_earlier_steps():
    log = AttestationLog()
    r = replay(attest=log)
    trainer = FakeTrainer(r, [live_batch(), mixed_batch(), mixed_batch()])
    for step in range(3):
        trainer.generate(step=step)

    witnesses = [x for x in log.records if x["op"] == "batch"]
    assert [w["step"] for w in witnesses] == ["1", "2"]
    for witness, record in zip(witnesses, samples(log)):
        step = int(witness["step"])
        assert record["min_entry_age"] == "1"
        assert all(int(s["entry_version"]) <= step - 1 for s in record["samples"])
        replayed = {row["content_digest"] for row in witness["replaced"]}
        assert replayed.isdisjoint(inserted_at(log, step))
    verify_chain(log.records)


def test_a_second_generation_at_the_same_step_leaves_its_dead_rows_dead():
    r = replay(attest=AttestationLog())
    trainer = FakeTrainer(r, [live_batch(), mixed_batch(), mixed_batch()])
    trainer.generate(step=0)
    trainer.generate(step=1)
    replaced = r.stats["replaced_rows"]

    trainer.generate(step=1)                        # the step-1 live group is now in the buffer

    assert r.last_replay is None and r.stats["replaced_rows"] == replaced
    assert r.stats["age_floor_skips"] == 1


def test_min_entry_age_zero_replays_an_earlier_generation_of_the_same_step_and_records_it():
    log = AttestationLog()
    r = replay(attest=log, min_entry_age=0)
    trainer = FakeTrainer(r, [mixed_batch(), copy.deepcopy(mixed_batch())])
    trainer.generate(step=0)                        # empty buffer: nothing to draw yet
    trainer.generate(step=0)

    (record,) = samples(log)
    assert record["min_entry_age"] == "0"
    assert [s["entry_version"] for s in record["samples"]] == ["0", "0"]
    assert r.stats["replaced_rows"] == 2


def torch_equal(a, b) -> bool:
    import torch

    return torch.equal(a, b)


def test_a_floor_above_one_is_refused_because_every_sample_would_be_empty():
    import pytest

    for bad in (2, -1, True):
        with pytest.raises(ValueError, match="min_entry_age"):
            replay(min_entry_age=bad)


def test_the_hook_reads_trls_aggregation_mode_for_the_dead_group_test():
    import torch

    from tests.test_trl_rows import make_output

    def batch():
        return make_output(prompts=[[5]] * 2, completions=[[1], [2]], advantages=[-0.5, -0.5])

    for mode, dead in (("sum_then_normalize", 1), ("normalize_then_sum", 0)):
        r = replay()
        trainer = FakeTrainer(r, [batch()])
        trainer.multi_objective_aggregation = mode
        r.note_rewards_per_func(torch.tensor([[1.0], [1.0]]), ["acc"])
        trainer.generate(step=0)
        assert r.stats["dead_groups"] == dead, mode
