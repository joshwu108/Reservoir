"""Tests for ``ReservoirGRPOTrainer`` and its hook-ran callback.

``assert_hook_ran`` is tested without TRL. The trainer class itself is
built from the real ``trl.GRPOTrainer`` (skipped when TRL is absent),
but no model is loaded: ``GRPOTrainer.__init__`` is replaced by a stub
that records its arguments, which is enough to check that the subclass
forwards everything, installs the replay buffer and registers the
callback. Running a real trainer is the Modal benchmark's job.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reservoir.integrations.trl import (
    ReservoirReplay,
    ReservoirReplayMixin,
    assert_hook_ran,
    build_trainer_class,
)


def test_assert_hook_ran_raises_only_after_a_step_without_hook_calls():
    r = ReservoirReplay(capacity=4)

    assert_hook_ran(r, global_step=0)
    with pytest.raises(RuntimeError, match="hook"):
        assert_hook_ran(r, global_step=1)
    r.stats["hook_calls"] = 1
    assert_hook_ran(r, global_step=1)


@pytest.fixture
def trainer_class():
    pytest.importorskip("trl")
    return build_trainer_class()


def test_trainer_class_is_a_grpo_trainer_with_the_override_first(trainer_class):
    from trl import GRPOTrainer

    assert issubclass(trainer_class, GRPOTrainer)
    mro = trainer_class.__mro__
    assert mro.index(ReservoirReplayMixin) < mro.index(GRPOTrainer)
    assert (
        trainer_class._generate_and_score_completions
        is ReservoirReplayMixin._generate_and_score_completions
    )


def test_module_attribute_and_import_resolve_to_the_same_class(trainer_class):
    from reservoir.integrations import trl as module
    from reservoir.integrations.trl import ReservoirGRPOTrainer

    assert module.ReservoirGRPOTrainer is trainer_class
    assert ReservoirGRPOTrainer is trainer_class


def test_trainer_class_and_callback_are_picklable_by_reference(trainer_class):
    import pickle

    assert pickle.loads(pickle.dumps(trainer_class)) is trainer_class
    assert pickle.loads(pickle.dumps(trainer_class.Callback)) is trainer_class.Callback


def test_star_import_does_not_need_the_trainer_class():
    from reservoir.integrations import trl as module

    assert "ReservoirGRPOTrainer" not in module.__all__
    assert "ReservoirGRPOTrainer" in dir(module)


def test_init_rejects_a_non_reservoir_buffer_before_loading_anything(trainer_class):
    with pytest.raises(TypeError, match="ReservoirReplay"):
        trainer_class(model="unused", replay_buffer=object())


def test_init_forwards_arguments_installs_the_buffer_and_registers_the_callback(trainer_class, monkeypatch):
    from trl import GRPOTrainer

    received: dict = {}

    def fake_init(self, *args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs

    added: list = []
    monkeypatch.setattr(GRPOTrainer, "__init__", fake_init)
    monkeypatch.setattr(trainer_class, "add_callback", lambda self, cb: added.append(cb))
    replay = ReservoirReplay(capacity=4)

    trainer = trainer_class("model-id", args=None, replay_buffer=replay, train_dataset=[])

    assert received == {"args": ("model-id",), "kwargs": {"args": None, "train_dataset": []}}
    assert trainer.replay_buffer is replay
    assert len(added) == 1 and isinstance(added[0], trainer_class.Callback)
    assert added[0].replay is replay


def test_callback_fails_the_run_when_the_hook_never_ran(trainer_class):
    from transformers import TrainerCallback

    callback = trainer_class.Callback(ReservoirReplay(capacity=4))
    control = object()

    assert isinstance(callback, TrainerCallback)
    assert callback.on_step_end(None, SimpleNamespace(global_step=0), control) is control
    with pytest.raises(RuntimeError):
        callback.on_step_end(None, SimpleNamespace(global_step=1), control)
