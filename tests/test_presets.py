from __future__ import annotations

from types import SimpleNamespace

import pytest

from reservoir.integrations.trl import ReservoirReplay, build_trainer_class


def test_grpo_presets_derive_valid_schedule_and_explain_choices():
    for profile in ("conservative", "async", "off"):
        replay, choices = ReservoirReplay.for_grpo(8, 2000, 2, profile=profile)
        assert replay.buffer.capacity >= 8
        assert replay.buffer.params.half_life > 0
        assert replay.buffer.params.max_policy_age <= 2000
        assert set(choices) == {"capacity", "half_life", "max_policy_age",
                                "max_log_ratio", "max_declines_per_step"}
    assert ReservoirReplay.for_grpo(8, 2000, 2, "off")[0].max_log_ratio is None
    assert ReservoirReplay.for_grpo(8, 2000, 2, "async")[0].max_log_ratio == 10.0


@pytest.mark.parametrize("args", [(0, 10, 1), (8, 0, 1), (8, 10, True)])
def test_grpo_preset_rejects_invalid_schedule(args):
    with pytest.raises(ValueError):
        ReservoirReplay.for_grpo(*args)


def test_trainer_attach_defaults_and_end_summary(tmp_path, monkeypatch, capsys):
    pytest.importorskip("trl")
    from trl import GRPOTrainer

    trainer_class = build_trainer_class()
    monkeypatch.setattr(GRPOTrainer, "__init__", lambda self, *a, **kw: None)
    monkeypatch.setattr(trainer_class, "add_callback", lambda self, cb: setattr(self, "callback", cb))
    replay, _ = ReservoirReplay.for_grpo(4, 10, 1)
    trainer = trainer_class("model", args=SimpleNamespace(output_dir=tmp_path), replay_buffer=replay)
    replay.attach(SimpleNamespace(num_processes=1, process_index=0))
    target = tmp_path / "reservoir"
    assert (target / "attest.jsonl").exists()
    assert (target / "manifest.jsonl").exists()
    replay.stats["replaced_rows"] = 2
    replay._generated_rows = 8
    control = object()
    assert trainer.callback.on_train_end(None, None, control) is control
    summary = capsys.readouterr().out
    assert "Rows replayed: 2" in summary
    assert "25.0%" in summary
    assert f"reservoir-verify {target / 'attest.jsonl'}" in summary


def test_explicit_log_path_survives_trainer_attach(tmp_path, monkeypatch):
    pytest.importorskip("trl")
    from trl import GRPOTrainer

    trainer_class = build_trainer_class()
    monkeypatch.setattr(GRPOTrainer, "__init__", lambda self, *a, **kw: None)
    monkeypatch.setattr(trainer_class, "add_callback", lambda self, cb: None)
    log = tmp_path / "custom.jsonl"
    replay = ReservoirReplay(capacity=4, attest=log)
    trainer_class("model", args=SimpleNamespace(output_dir=tmp_path), replay_buffer=replay)
    replay.attach(SimpleNamespace(num_processes=1, process_index=0))
    assert replay._buffer_kwargs["attest"] == log
    assert replay._buffer_kwargs["manifest"] is None
    replay.close()


def test_zero_config_replay_derives_capacity_at_attach(tmp_path, monkeypatch):
    pytest.importorskip("trl")
    from trl import GRPOTrainer

    trainer_class = build_trainer_class()
    monkeypatch.setattr(GRPOTrainer, "__init__", lambda self, *a, **kw: None)
    monkeypatch.setattr(trainer_class, "add_callback", lambda self, cb: None)
    replay = ReservoirReplay()
    trainer_class("model", args=SimpleNamespace(output_dir=tmp_path,
                  num_generations=8, max_steps=100, per_device_train_batch_size=2), replay_buffer=replay)
    replay.attach(SimpleNamespace(num_processes=1, process_index=0))
    assert replay.buffer.capacity >= 8
    assert replay.buffer.params.max_policy_age <= 100
    assert (tmp_path / "reservoir" / "attest.jsonl").exists()
    replay.close()


def test_explicit_none_disables_default_logging(tmp_path):
    replay = ReservoirReplay(capacity=4, attest=None)
    replay._trainer_output_dir = tmp_path
    replay.attach(SimpleNamespace(num_processes=1, process_index=0))
    assert replay._buffer_kwargs["attest"] is None
    assert not (tmp_path / "reservoir").exists()
