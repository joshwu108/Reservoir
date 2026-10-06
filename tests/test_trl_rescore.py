"""Tests for the priority-refresh hook: ``PriorityStrategy.rescore`` at replay time.

The adapter asks the buffer's strategy for a new priority of each placed
rollout, once per distinct slot, with a ``ReplaySignal`` of what it knows
at that moment, and writes any answer as an ``update`` record whose new
base priority is the value after ``alpha`` and quantisation. The default
keeps priorities fixed, declined draws are skipped, and a bad answer is
refused like a bad score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from reservoir.attest import AttestationLog
from reservoir.decay import quantize_priority
from reservoir.priorities import PriorityStrategy, ReplaySignal, validated_rescore
from reservoir.rollout import Rollout, RolloutGroup
from reservoir_checker.verify import verify_chain
from tests.test_trl_replay import FakeTrainer, LOGP, live_batch, mixed_batch, replay
from tests.test_trl_telemetry import MetricTrainer


@dataclass(frozen=True)
class DriftAware(PriorityStrategy):
    """Priority = |advantage| shrunk by how far the rollout has drifted; records every call."""

    calls: list = field(default_factory=list, compare=False, hash=False)

    def score(self, rollout: Rollout, group: RolloutGroup) -> float:
        return abs(rollout.reward) + 1e-6

    def rescore(self, rollout, group, signal: ReplaySignal):
        value = abs(rollout.reward) / (1.0 + abs(signal.log_ratio or 0.0)) + 1e-6
        self.calls.append((rollout, signal, value))
        return value


@dataclass(frozen=True)
class LopSided(DriftAware):
    """All priority on the one rollout with reward 1.0, so every draw lands on its slot."""

    def score(self, rollout: Rollout, group: RolloutGroup) -> float:
        return 1.0 if rollout.reward == 1.0 else 0.0


@dataclass(frozen=True)
class DriftAwareQuiet(PriorityStrategy):
    """DriftAware without the call recorder, so it can be fingerprinted in a snapshot."""

    def score(self, rollout, group):
        return abs(rollout.reward) + 1e-6

    def rescore(self, rollout, group, signal):
        return abs(rollout.reward) / (1.0 + abs(signal.log_ratio or 0.0)) + 1e-6


@dataclass(frozen=True)
class Keep(PriorityStrategy):
    def score(self, rollout, group):
        return abs(rollout.reward) + 1e-6


@dataclass(frozen=True)
class Broken(PriorityStrategy):
    def score(self, rollout, group):
        return abs(rollout.reward) + 1e-6

    def rescore(self, rollout, group, signal):
        return -1.0


def updates(records):
    return [r for r in records if r["op"] == "update"]


def test_rescore_writes_the_strategy_value_after_alpha_and_quantisation():
    strategy = DriftAware()
    r = replay(attest=AttestationLog(), priority=strategy)
    trainer = MetricTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=1)
    strategy.calls.clear()
    trainer.current_logp = LOGP - 0.5
    trainer.generate(step=2)

    batch = r.last_replay
    first_placement = {}
    for k, slot in enumerate(batch.indices):
        first_placement.setdefault(slot, k)
    assert len(strategy.calls) == len(first_placement) == r.stats["rescored_rows"]
    ups = updates(r.buffer.attestation_log.records)
    assert {u["index"] for u in ups} == set(first_placement)
    for (rollout, signal, value), (slot, k) in zip(strategy.calls, first_placement.items()):
        assert rollout is batch.rollouts[k]
        assert signal.log_ratio is not None and signal.log_ratio == pytest.approx(-0.5 * len(rollout))
        assert signal.age == r.buffer.current_version - batch.model_versions[k]
        assert signal.step == 2 and signal.is_weight == float(batch.is_weights[k])
        assert r.buffer.base_priority(slot) == quantize_priority(value ** r.buffer.alpha, r.buffer.params)
    verify_chain(r.buffer.attestation_log.records)


def test_a_slot_drawn_twice_is_rescored_once_from_its_first_placement():
    strategy = LopSided()
    r = replay(attest=AttestationLog(), priority=strategy)
    trainer = MetricTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=1)                      # no dead group: nothing replayed, nothing rescored
    assert updates(r.buffer.attestation_log.records) == []
    strategy.calls.clear()
    trainer.generate(step=2)
    batch = r.last_replay
    assert len(set(batch.indices)) == 1 < len(batch.indices)
    ups = updates(r.buffer.attestation_log.records)
    assert len(ups) == len(set(batch.indices)) == r.stats["rescored_rows"] == 1
    assert len(strategy.calls) == 1
    rollout, signal, _ = strategy.calls[0]
    assert rollout is batch.rollouts[0]           # the first placement's signal
    assert signal.advantage == batch.rollouts[0].reward * float(batch.is_weights[0])


def test_default_strategy_keeps_priorities():
    r = replay(attest=AttestationLog())
    trainer = FakeTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=1)
    trainer.generate(step=2)
    assert updates(r.buffer.attestation_log.records) == [] and r.stats["rescored_rows"] == 0


def test_explicit_none_keeps_priorities():
    r = replay(attest=AttestationLog(), priority=Keep())
    trainer = FakeTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=1)
    trainer.generate(step=2)
    assert updates(r.buffer.attestation_log.records) == []


def test_bad_rescore_is_refused_like_a_bad_score():
    r = replay(attest=AttestationLog(), priority=Broken())
    trainer = FakeTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=1)
    with pytest.raises(ValueError, match="Broken"):
        trainer.generate(step=2)


def test_declined_draws_are_not_rescored():
    strategy = DriftAware()
    r = replay(attest=AttestationLog(), priority=strategy, max_log_ratio=0.1)
    trainer = MetricTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=1)
    strategy.calls.clear()
    trainer.current_logp = LOGP - 1.0
    trainer.generate(step=2)
    assert r.stats["declined_rows"] == 2 and r.stats["rescored_rows"] == 0 and strategy.calls == []
    assert updates(r.buffer.attestation_log.records) == []


def test_non_finite_log_ratio_is_passed_as_none():
    strategy = DriftAware()
    r = replay(attest=AttestationLog(), priority=strategy)
    second = mixed_batch(old_logps=[[-0.1], [-0.2, -0.3], [-0.4, -0.5], [-0.6]])
    trainer = MetricTrainer(r, [live_batch(), second])
    trainer.generate(step=1)
    strategy.calls.clear()
    trainer.current_logp = float("nan")
    trainer.generate(step=2)
    assert strategy.calls and all(signal.log_ratio is None for _, signal, _ in strategy.calls)


def test_rescore_through_the_durable_buffer(tmp_path):
    strategy = DriftAwareQuiet()
    r = replay(directory=tmp_path / "buf", attest=tmp_path / "a.jsonl", priority=strategy)
    trainer = MetricTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=1)
    trainer.generate(step=2)
    assert r.stats["rescored_rows"] >= 1
    records = r.buffer.attestation_log.records
    assert updates(records)
    r.close()
    again = replay(directory=tmp_path / "buf", attest=tmp_path / "a.jsonl", priority=DriftAwareQuiet())
    assert again.buffer.attestation_log.records == records
    again.close()


def test_reset_age_on_update_log_verifies():
    r = replay(attest=AttestationLog(), priority=DriftAware(), reset_age_on_update=True)
    trainer = MetricTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(step=1)
    trainer.generate(step=2)
    assert updates(r.buffer.attestation_log.records)
    verify_chain(r.buffer.attestation_log.records)


def test_validated_rescore_type_checks():
    strategy = DriftAware()
    rollout = Rollout([1], [-0.1], 1.0)
    group = RolloutGroup("p", 0, [rollout])
    signal = ReplaySignal(advantage=0.5, is_weight=1.0, log_ratio=None, age=0, step=0)
    assert validated_rescore(strategy, rollout, group, signal) == pytest.approx(1.0 + 1e-6)
    assert validated_rescore(Keep(), rollout, group, signal) is None
    with pytest.raises(TypeError, match="ReplaySignal"):
        validated_rescore(strategy, rollout, group, (0.5, 1.0, None, 0, 0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validated_rescore(Broken(), rollout, group, signal)
