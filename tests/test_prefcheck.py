"""Tests for reservoir.prefcheck.PreferenceNoiseDetector.

All tests run without GPU and without actual TRL training via mocking.
"""

import unittest.mock as mock
from unittest.mock import MagicMock, patch

import pytest

from reservoir.report import PreferenceQualityReport


def _make_minimal_dataset(n: int = 10) -> list[dict]:
    return [
        {"chosen": f"good_{i}", "rejected": f"bad_{i}", "__index__": i}
        for i in range(n)
    ]


@pytest.fixture
def mock_trl(monkeypatch):
    """Patch trl so PreferenceNoiseDetector can be used without GPU.

    Patches _InstrumentedRewardTrainer (the class train() instantiates) and
    RewardConfig so no real training runs.
    """
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train = MagicMock(return_value=None)
    mock_trainer_class = MagicMock(return_value=mock_trainer_instance)

    mock_reward_config = MagicMock()

    with patch("reservoir.prefcheck._InstrumentedRewardTrainer", mock_trainer_class), \
         patch("reservoir.prefcheck.RewardConfig", mock_reward_config):
        yield mock_trainer_class, mock_reward_config, mock_trainer_instance


def test_prefcheck_raises_if_get_report_before_train():
    from reservoir.prefcheck import PreferenceNoiseDetector
    model = MagicMock()
    tokenizer = MagicMock()
    ds = _make_minimal_dataset()
    detector = PreferenceNoiseDetector(model=model, tokenizer=tokenizer, train_dataset=ds)
    with pytest.raises(RuntimeError):
        detector.get_report()


def test_prefcheck_initializes_in_audit_mode():
    from reservoir.prefcheck import PreferenceNoiseDetector
    model = MagicMock()
    tokenizer = MagicMock()
    ds = _make_minimal_dataset()
    detector = PreferenceNoiseDetector(model=model, tokenizer=tokenizer,
                                       train_dataset=ds, mode="audit")
    assert detector.mode == "audit"


def test_prefcheck_initializes_in_accelerated_mode():
    from reservoir.prefcheck import PreferenceNoiseDetector
    model = MagicMock()
    tokenizer = MagicMock()
    ds = _make_minimal_dataset()
    detector = PreferenceNoiseDetector(model=model, tokenizer=tokenizer,
                                       train_dataset=ds, mode="accelerated")
    assert detector.mode == "accelerated"


def test_prefcheck_train_calls_trainer(mock_trl):
    from reservoir.prefcheck import PreferenceNoiseDetector
    mock_trainer_class, mock_reward_config, mock_trainer_instance = mock_trl

    model = MagicMock()
    tokenizer = MagicMock()
    ds = _make_minimal_dataset()
    detector = PreferenceNoiseDetector(model=model, tokenizer=tokenizer, train_dataset=ds)
    detector.train()
    mock_trainer_instance.train.assert_called_once()


def test_prefcheck_get_report_returns_quality_report(mock_trl):
    from reservoir.prefcheck import PreferenceNoiseDetector
    mock_trainer_class, mock_reward_config, mock_trainer_instance = mock_trl

    model = MagicMock()
    tokenizer = MagicMock()
    ds = _make_minimal_dataset()
    detector = PreferenceNoiseDetector(model=model, tokenizer=tokenizer, train_dataset=ds)
    detector.train()
    report = detector.get_report()
    assert isinstance(report, PreferenceQualityReport)
