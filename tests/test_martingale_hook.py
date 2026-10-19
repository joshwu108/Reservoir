"""The replayed-row hook for a flight recorder, against a fake recorder.

``ReservoirReplay(replay_recorder=...)`` (or ``trainer.flight_recorder``)
receives, for every kept draw, the exact batch-normalised importance
weight and the staleness policy's exact rescale, and the ids it returns
are written into the batch's ``martingale_row_id`` column. Martingale is
never imported; the joint run against the real package is recorded in
``docs/staleness.md``.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from reservoir.attest import AttestationLog
from reservoir.integrations._trl_staleness import StalenessPolicy
from reservoir.integrations.martingale import ROW_ID_KEY, ReplayRecorder, draw_scales
from reservoir_checker.verify import verify_chain
from tests.test_trl_replay import LOGP, live_batch, mixed_batch, replay
from tests.test_trl_telemetry import MetricTrainer

DEAD_ROWS = (2, 3)   # the dead group of mixed_batch
FRESH_IDS = (40, 41, 42, 43)


class FakeRecorder:
    """Duck-typed ``register_replayed``: records every call and hands out ids from 100."""

    def __init__(self, unknown: int = -1) -> None:
        self.calls: list[dict] = []
        self.unknown = unknown

    def register_replayed(self, trainer, **kwargs) -> int:
        self.calls.append(kwargs)
        if len(self.calls) - 1 == self.unknown:
            return -1
        return 100 + len(self.calls) - 1


def with_ids(output: dict) -> dict:
    return {**output, ROW_ID_KEY: torch.tensor(FRESH_IDS)}


def run(policy=None, recorder=None, drift: float = 0.5, ids: bool = True, on_trainer=None):
    """Store live_batch at step 1, drift the policy, replay into mixed_batch at step 2."""
    r = replay(attest=AttestationLog(), staleness_policy=policy, replay_recorder=recorder)
    second = copy.deepcopy(mixed_batch())
    trainer = MetricTrainer(r, [live_batch(), with_ids(second) if ids else second])
    if on_trainer is not None:
        trainer.flight_recorder = on_trainer
    trainer.generate(step=1)
    trainer.current_logp = LOGP + drift
    return r, trainer, trainer.generate(step=2)


def test_exact_weight_and_rescale_reach_the_recorder_and_ids_reach_the_rows():
    recorder = FakeRecorder(unknown=1)
    r, _, out = run(StalenessPolicy(mass_cap=0.2), recorder)
    batch, point = r.last_replay, r.last_telemetry
    assert r.stats["rescaled_rows"] == 2 and len(recorder.calls) == 2
    for k, (call, row) in enumerate(zip(recorder.calls, DEAD_ROWS)):
        scale = point.decisions[k].scale
        assert isinstance(call["is_weight"], Fraction) and call["is_weight"] == batch.is_weights[k]
        assert isinstance(call["rescale"], Fraction) and call["rescale"] == scale < 1
        assert call["advantage"] == batch.rollouts[k].reward * float(batch.is_weights[k] * scale)
        assert call["draw_id"] == f"{batch.op_counter}/{k}"
        assert call["content_digest"] == batch.content_digests[k]
        assert call["behavior_step"] == 1 and call["buffer_age"] == 1
        assert call["completion_ids"] == list(batch.rollouts[k].tokens)
        assert call["prompt_ids"] == batch.rollouts[k].metadata["prompt_ids"]
        assert call["behavior_logprobs"] == list(batch.rollouts[k].logprobs)
    assert out[ROW_ID_KEY].tolist() == [40, 41, 100, -1]
    verify_chain(r.buffer.attestation_log.records)


def test_policy_off_passes_a_rescale_of_one():
    recorder = FakeRecorder()
    r, _, out = run(None, recorder)
    assert [c["rescale"] for c in recorder.calls] == [Fraction(1), Fraction(1)]
    assert [c["is_weight"] for c in recorder.calls] == list(r.last_replay.is_weights)
    assert out[ROW_ID_KEY].tolist()[2:] == [100, 101]


def test_declined_draws_are_not_registered():
    recorder = FakeRecorder()
    r, _, out = run(StalenessPolicy(max_age=0), recorder)
    assert r.stats["declined_rows"] == 2 and recorder.calls == []
    assert out[ROW_ID_KEY].tolist() == list(FRESH_IDS)


def test_trainer_flight_recorder_is_used_when_none_is_given():
    recorder = FakeRecorder()
    run(None, None, on_trainer=recorder)
    assert len(recorder.calls) == 2


def test_no_recorder_leaves_the_id_column_untouched():
    r, _, out = run(None, None)
    assert r.stats["replaced_rows"] == 2 and out[ROW_ID_KEY].tolist() == list(FRESH_IDS)


def test_missing_id_column_is_refused_after_the_log_is_complete():
    recorder = FakeRecorder()
    with pytest.raises(ValueError, match="martingale_row_id"):
        run(None, recorder, ids=False)


def test_a_recorder_without_register_replayed_is_refused():
    with pytest.raises(TypeError, match="register_replayed"):
        run(None, None, on_trainer=SimpleNamespace())


def test_fake_satisfies_the_protocol_and_scales_default_to_one():
    assert isinstance(FakeRecorder(), ReplayRecorder)
    assert draw_scales((), 3) == [Fraction(1)] * 3


def test_reservoir_never_imports_martingale():
    code = ("import sys, reservoir.integrations.trl, reservoir.integrations.martingale; "
            "assert not any(m == 'martingale' or m.startswith('martingale.') for m in sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True)
