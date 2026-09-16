"""Tests for reservoir.forgetting_monitor — TDD RED phase."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from reservoir.anchor_set import AnchorSet
from reservoir.forgetting_monitor import ForgettingMonitor, ForgettingAlert, ForgettingReport


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_anchor_set(n=4, baseline=0.5):
    """Create an AnchorSet with baseline already snapshotted."""
    examples = [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]} for _ in range(n)]
    a = AnchorSet(examples, tags=["test"] * n)
    a.snapshot_baseline({i: baseline for i in range(n)})
    return a


class MockOutput:
    """Simulates model output with a controllable scalar loss."""
    def __init__(self, loss_value):
        import torch
        self.loss = torch.tensor(loss_value, requires_grad=True)
        self.logits = torch.zeros(1, 10)


class MockModel:
    """Minimal mock that returns a fixed loss."""

    def __init__(self, loss_value=0.5):
        self.loss_value = loss_value
        self.config = MagicMock()
        self.config.model_type = "gpt2"
        self.config.num_labels = 1
        self._train_mode = True

    def __call__(self, **kwargs):
        return MockOutput(self.loss_value)

    def eval(self):
        self._train_mode = False
        return self

    def train(self):
        self._train_mode = True
        return self

    def parameters(self):
        import torch
        return iter([torch.zeros(1)])


def make_trainer_state(global_step=0):
    state = MagicMock()
    state.global_step = global_step
    return state


def make_trainer_args():
    args = MagicMock()
    args.device = "cpu"
    return args


def make_trainer_control():
    return MagicMock()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_monitor_initializes():
    anchor_set = make_anchor_set()
    monitor = ForgettingMonitor([anchor_set])
    assert monitor is not None
    assert len(monitor.alerts) == 0


def test_on_train_begin_calls_baseline_snapshot(capsys):
    anchor_set = make_anchor_set(n=4, baseline=0.0)  # baselines not yet set
    monitor = ForgettingMonitor([anchor_set])
    model = MockModel(loss_value=0.5)
    monitor.on_train_begin(
        make_trainer_args(),
        make_trainer_state(),
        make_trainer_control(),
        model=model,
    )
    # Baseline should now be set on all anchors
    for anchor in anchor_set:
        assert anchor.baseline_loss > 0.0 or anchor.baseline_loss == pytest.approx(0.5)


def test_no_alert_below_threshold():
    """forgetting_score = 0.1, threshold = 0.5 -> no alert"""
    anchor_set = make_anchor_set(n=2, baseline=1.0)
    monitor = ForgettingMonitor([anchor_set], alert_threshold=0.5, eval_every_n_steps=1)

    model = MockModel(loss_value=1.1)  # current ≈ baseline + 10%, score ≈ 0.1
    state = make_trainer_state(global_step=1)
    monitor.on_step_end(
        make_trainer_args(), state, make_trainer_control(), model=model
    )
    assert len(monitor.alerts) == 0


def test_alert_fires_above_threshold():
    """forgetting_score = 0.8, threshold = 0.5 -> alert fired"""
    anchor_set = make_anchor_set(n=2, baseline=1.0)
    monitor = ForgettingMonitor([anchor_set], alert_threshold=0.5, eval_every_n_steps=1)

    model = MockModel(loss_value=1.8)  # loss rose 80%, score ≈ 0.8
    state = make_trainer_state(global_step=1)
    monitor.on_step_end(
        make_trainer_args(), state, make_trainer_control(), model=model
    )
    assert len(monitor.alerts) == 1


def test_alert_message_contains_tag():
    anchor_set = AnchorSet(
        [{"input_ids": [1, 2], "attention_mask": [1, 1]} for _ in range(2)],
        tags="my-domain",
    )
    anchor_set.snapshot_baseline({0: 1.0, 1: 1.0})
    monitor = ForgettingMonitor([anchor_set], alert_threshold=0.3, eval_every_n_steps=1)

    model = MockModel(loss_value=2.0)
    monitor.on_step_end(
        make_trainer_args(),
        make_trainer_state(global_step=1),
        make_trainer_control(),
        model=model,
    )
    assert len(monitor.alerts) == 1
    assert "my-domain" in monitor.alerts[0].message


def test_forgetting_history_records_per_step():
    anchor_set = make_anchor_set(n=2, baseline=1.0)
    monitor = ForgettingMonitor([anchor_set], eval_every_n_steps=1)

    model = MockModel(loss_value=1.5)
    for step in [1, 2, 3]:
        monitor.on_step_end(
            make_trainer_args(),
            make_trainer_state(global_step=step),
            make_trainer_control(),
            model=model,
        )

    history = monitor.forgetting_history()
    assert "test" in history
    assert len(history["test"]) == 3
    step_vals = [s for s, _ in history["test"]]
    assert step_vals == [1, 2, 3]


def test_get_report_returns_forgetting_report():
    anchor_set = make_anchor_set(n=3, baseline=1.0)
    monitor = ForgettingMonitor([anchor_set], eval_every_n_steps=1)
    model = MockModel(loss_value=1.5)
    state = make_trainer_state(global_step=1)
    state.max_steps = 10
    monitor.on_step_end(
        make_trainer_args(), state, make_trainer_control(), model=model
    )
    report = monitor.get_report()
    assert isinstance(report, ForgettingReport)
    assert report.n_anchors == 3
    assert "test" in report.groups


def test_verbose_false_suppresses_output(capsys):
    anchor_set = make_anchor_set(n=2, baseline=1.0)
    monitor = ForgettingMonitor([anchor_set], eval_every_n_steps=1, verbose=False)
    model = MockModel(loss_value=1.5)
    monitor.on_step_end(
        make_trainer_args(),
        make_trainer_state(global_step=1),
        make_trainer_control(),
        model=model,
    )
    captured = capsys.readouterr()
    assert captured.out == ""


def test_auto_replay_false_by_default():
    anchor_set = make_anchor_set()
    monitor = ForgettingMonitor([anchor_set])
    assert monitor.auto_replay is False


def test_on_train_end_prints_summary(capsys):
    anchor_set = make_anchor_set(n=2, baseline=1.0)
    monitor = ForgettingMonitor([anchor_set], eval_every_n_steps=1)
    model = MockModel(loss_value=1.2)
    monitor.on_step_end(
        make_trainer_args(),
        make_trainer_state(global_step=1),
        make_trainer_control(),
        model=model,
    )
    monitor.on_train_end(
        make_trainer_args(),
        make_trainer_state(global_step=10),
        make_trainer_control(),
    )
    captured = capsys.readouterr()
    assert "forgetting" in captured.out.lower() or "summary" in captured.out.lower()
