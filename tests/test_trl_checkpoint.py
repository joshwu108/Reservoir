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


# ---------------------------------------------------------------------------
# The model checkpoint's digest travels with the buffer checkpoint
# ---------------------------------------------------------------------------

from reservoir.integrations._trl_lifecycle import (  # noqa: E402
    MODEL_CHECKPOINT_KEY, check_model_binding, directory_digest, model_binding, trainer_checkpoint_dir,
)


def write_model_checkpoint(root, step: int, weights: bytes = b"w" * 100) -> "Path":
    from pathlib import Path

    directory = Path(root) / f"checkpoint-{step}"
    (directory / "sub").mkdir(parents=True, exist_ok=True)
    (directory / "model.safetensors").write_bytes(weights)
    (directory / "trainer_state.json").write_text('{"global_step": %d}' % step)
    (directory / "sub" / "rng.pth").write_bytes(b"\x00\x01")
    return directory


class TestDirectoryDigest:
    def test_depends_on_names_sizes_and_bytes_but_not_on_location(self, tmp_path):
        a = write_model_checkpoint(tmp_path / "x", 1)
        b = write_model_checkpoint(tmp_path / "y", 1)
        assert directory_digest(a) == directory_digest(b)
        assert directory_digest(a)["files"] == 3 and directory_digest(a)["bytes"] == 100 + 18 + 2
        (b / "model.safetensors").write_bytes(b"w" * 99 + b"v")
        assert directory_digest(a)["digest"] != directory_digest(b)["digest"]
        c = write_model_checkpoint(tmp_path / "z", 1)
        (c / "sub" / "rng.pth").rename(c / "sub" / "rng2.pth")
        assert directory_digest(a)["digest"] != directory_digest(c)["digest"]
        d = write_model_checkpoint(tmp_path / "w", 1)
        (d / "extra.bin").write_bytes(b"")
        assert directory_digest(a)["digest"] != directory_digest(d)["digest"] and directory_digest(d)["files"] == 4

    def test_missing_directory_is_an_error_and_a_file_is_not_a_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            directory_digest(tmp_path / "missing")
        assert model_binding(tmp_path / "missing") is None and model_binding(None) is None
        assert trainer_checkpoint_dir(None, 3) is None
        assert trainer_checkpoint_dir(tmp_path, 3) == tmp_path / "checkpoint-3"


class TestBoundResume:
    def test_bind_records_the_digest_and_an_unchanged_checkpoint_resumes(self, tmp_path):
        r = durable(tmp_path)
        trainer = FakeTrainer(r, [live_batch(), mixed_batch(), live_batch(20)])
        out = tmp_path / "out"
        trainer.generate(step=1)
        trainer.generate(step=2)
        model_dir = write_model_checkpoint(out, 2)
        bind_checkpoint(r, 2, out)
        binding = r.buffer.checkpoint_binding("step-2")
        assert binding == {MODEL_CHECKPOINT_KEY: {"name": "checkpoint-2", **directory_digest(model_dir)}}
        assert (tmp_path / "buf" / "checkpoints" / "step-2" / "binding.json").exists()
        state_at_2 = r.buffer.state_dict()
        trainer.generate(step=3)
        r.close()

        again = durable(tmp_path)
        resume_from_checkpoint(again, 2, model_dir)
        assert again.buffer.state_dict() == state_at_2
        again.close()

    def test_a_changed_model_checkpoint_is_refused(self, tmp_path):
        r = durable(tmp_path)
        trainer = FakeTrainer(r, [live_batch(), mixed_batch()])
        out = tmp_path / "out"
        trainer.generate(step=1)
        model_dir = write_model_checkpoint(out, 1)
        bind_checkpoint(r, 1, out)
        head = r.buffer.attestation_log.head_digest
        (model_dir / "model.safetensors").write_bytes(b"v" * 100)       # same size, other weights
        with pytest.raises(RuntimeError, match="not the one the buffer checkpoint was bound to"):
            resume_from_checkpoint(r, 1, model_dir)
        assert r.buffer.attestation_log.head_digest == head                # nothing was rewound
        r.close()

    def test_a_missing_model_checkpoint_cannot_be_verified_and_is_refused(self, tmp_path):
        r = durable(tmp_path)
        trainer = FakeTrainer(r, [live_batch(), mixed_batch()])
        out = tmp_path / "out"
        trainer.generate(step=1)
        write_model_checkpoint(out, 1)
        bind_checkpoint(r, 1, out)
        with pytest.raises(RuntimeError, match="ReservoirReplay.model_checkpoint"):
            resume_from_checkpoint(r, 1, None)
        with pytest.raises(RuntimeError, match="ReservoirReplay.model_checkpoint"):
            resume_from_checkpoint(r, 1, tmp_path / "elsewhere" / "checkpoint-1")
        r.close()

    def test_a_buffer_checkpoint_without_a_binding_resumes_unchecked(self, tmp_path):
        r = durable(tmp_path)
        trainer = FakeTrainer(r, [live_batch(), mixed_batch()])
        trainer.generate(step=1)
        bind_checkpoint(r, 1, tmp_path / "out")           # no checkpoint-1 directory exists yet
        assert r.buffer.checkpoint_binding("step-1") is None
        resume_from_checkpoint(r, 1, tmp_path / "out" / "checkpoint-1")
        resume_from_checkpoint(r, 1, None)
        r.close()

    def test_check_model_binding_alone(self, tmp_path):
        model_dir = write_model_checkpoint(tmp_path, 4)
        binding = model_binding(model_dir)
        check_model_binding(None, model_dir, 4)
        check_model_binding({}, None, 4)
        check_model_binding(binding, model_dir, 4)
        with pytest.raises(RuntimeError, match="ReservoirReplay.model_checkpoint"):
            check_model_binding(binding, None, 4)
        (model_dir / "trainer_state.json").write_text("{}")
        with pytest.raises(RuntimeError, match="not the one"):
            check_model_binding(binding, model_dir, 4)

    def test_the_callback_passes_the_trainer_checkpoint_directory(self, tmp_path, monkeypatch):
        """``on_train_begin`` resolves ``output_dir/checkpoint-N`` for the resume check."""
        from types import SimpleNamespace

        import reservoir.integrations.trl as trl_module

        pytest.importorskip("trl")
        seen = []
        monkeypatch.setattr(trl_module, "resume_from_checkpoint",
                            lambda replay, step, model=None: seen.append((step, model)))
        callback_cls = trl_module.build_trainer_class().replay_callback_class
        args = SimpleNamespace(output_dir=str(tmp_path / "out"))
        callback_cls(replay()).on_train_begin(args, SimpleNamespace(global_step=7), None)
        assert seen == [(7, tmp_path / "out" / "checkpoint-7")]

    def test_binding_is_absent_with_a_warning_when_the_trainer_directory_is_missing_at_save(self, tmp_path):
        r = durable(tmp_path)
        FakeTrainer(r, [live_batch()]).generate(step=1)
        with pytest.warns(RuntimeWarning, match="will not be checked"):
            bind_checkpoint(r, 1, tmp_path / "out")                 # out/checkpoint-1 does not exist
        assert r.buffer.checkpoint_binding("step-1") is None
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            bind_checkpoint(r, 1)                                   # no output_dir at all: nothing to warn about
        r.close()

    def test_resume_directory_resolution_order(self, tmp_path):
        from types import SimpleNamespace

        from reservoir.integrations._trl_lifecycle import resume_model_checkpoint

        r = replay()
        out = tmp_path / "out"
        named = tmp_path / "elsewhere" / "checkpoint-4"
        named.mkdir(parents=True)
        args = SimpleNamespace(output_dir=str(out), resume_from_checkpoint=None)
        assert resume_model_checkpoint(r, args, 4) == out / "checkpoint-4"
        args.resume_from_checkpoint = str(named)
        assert resume_model_checkpoint(r, args, 4) == named
        args.resume_from_checkpoint = True                          # the trainer's "latest" flag, not a path
        assert resume_model_checkpoint(r, args, 4) == out / "checkpoint-4"
        r.model_checkpoint = tmp_path / "copied"
        assert resume_model_checkpoint(r, args, 4) == tmp_path / "copied"

    def test_a_relocated_checkpoint_resumes_through_the_override(self, tmp_path):
        import shutil

        r = durable(tmp_path)
        trainer = FakeTrainer(r, [live_batch(), mixed_batch()])
        trainer.generate(step=1)
        write_model_checkpoint(tmp_path / "out", 1)
        bind_checkpoint(r, 1, tmp_path / "out")
        shutil.move(str(tmp_path / "out" / "checkpoint-1"), str(tmp_path / "moved"))
        with pytest.raises(RuntimeError, match="is not a directory"):
            resume_from_checkpoint(r, 1, tmp_path / "out" / "checkpoint-1")
        resume_from_checkpoint(r, 1, tmp_path / "moved")
        r.close()
