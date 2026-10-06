"""Tests for binding the durable buffer to the trainer's checkpoints.

``on_save`` snapshots the buffer under the trainer's step; ``on_train_begin``
at a non-zero step rewinds the buffer to that snapshot, so a resumed run
continues the draw counter and the attestation chain from the checkpoint
rather than from wherever the crashed run had got to.
"""

from __future__ import annotations

import pytest

from reservoir.integrations.trl import (
    bind_checkpoint, checkpoint_tag, resume_from_checkpoint, trainer_checkpoint_steps,
)
from tests.test_trl_replay import FakeTrainer, live_batch, mixed_batch, replay


def durable(tmp_path, **kw):
    return replay(directory=tmp_path / "buf", attest=tmp_path / "attest.jsonl", **kw)


def test_save_then_resume_rewinds_to_the_checkpoint(tmp_path):
    r = durable(tmp_path)
    trainer = FakeTrainer(r, [live_batch(), mixed_batch(), live_batch(20), mixed_batch()])
    trainer.generate(step=1)
    trainer.generate(step=2)
    bind_checkpoint(r, 2)
    head_at_2, state_at_2 = r.buffer.attestation_log.head_digest, r.buffer.state_dict()
    trainer.generate(step=3)
    trainer.generate(step=4)          # the crashed run went two steps past the checkpoint
    assert r.buffer.attestation_log.head_digest != head_at_2
    r.close()

    again = durable(tmp_path)
    assert again.buffer.attestation_log.head_digest != head_at_2   # reopened as the crashed run left it
    resume_from_checkpoint(again, 2)
    assert again.buffer.state_dict() == state_at_2
    assert again.buffer.attestation_log.head_digest == head_at_2
    assert again.buffer.checkpoints() == [checkpoint_tag(2)]
    again.close()


def test_resume_at_step_zero_is_a_no_op_and_unknown_steps_fail(tmp_path):
    r = durable(tmp_path)
    resume_from_checkpoint(r, 0)
    with pytest.raises(RuntimeError, match="no checkpoint"):
        resume_from_checkpoint(r, 7)
    r.close()


def test_non_durable_buffer_cannot_resume_but_can_save():
    r = replay()
    bind_checkpoint(r, 3)             # nothing to bind; no error
    resume_from_checkpoint(r, 0)
    with pytest.raises(RuntimeError, match="durable"):
        resume_from_checkpoint(r, 3)


def test_save_prunes_buffer_checkpoints_to_the_trainer_s_surviving_ones(tmp_path):
    r = durable(tmp_path)
    trainer = FakeTrainer(r, [live_batch(), mixed_batch(), live_batch(20), mixed_batch()])
    out = tmp_path / "out"
    for step in (1, 2, 3):
        trainer.generate(step=step)
        (out / f"checkpoint-{step}").mkdir(parents=True)
        bind_checkpoint(r, step, out)
    assert r.buffer.checkpoints() == ["step-1", "step-2", "step-3"]
    (out / "checkpoint-1").rename(out / "gone-1")           # the trainer rotated it away
    trainer.generate(step=4)
    bind_checkpoint(r, 4, out)
    assert r.buffer.checkpoints() == ["step-2", "step-3", "step-4"]
    assert trainer_checkpoint_steps(out) == {2, 3}
    assert trainer_checkpoint_steps(None) is None and trainer_checkpoint_steps(tmp_path / "missing") is None
    bind_checkpoint(r, 5, tmp_path / "missing")             # nothing to prune against: keep everything
    assert r.buffer.checkpoints() == ["step-2", "step-3", "step-4", "step-5"]
    r.close()
