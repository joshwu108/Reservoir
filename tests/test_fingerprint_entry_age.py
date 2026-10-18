"""``min_entry_age`` is a construction parameter a snapshot must be loaded with.

Two buffers that differ only in it draw different rows from the same state
(one may replay a slot inserted this step, the other may not), so a
snapshot taken under one must not restore under the other.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reservoir.attest import AttestationLog
from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer

KW = dict(capacity=8, half_life=1, max_policy_age=2, seed=3)


def _rollouts(rewards: list[float]) -> list[Rollout]:
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in rewards]


def test_buffers_differing_only_in_min_entry_age_have_different_fingerprints() -> None:
    a = RolloutBuffer(**KW, min_entry_age=0)
    b = RolloutBuffer(**KW, min_entry_age=1)
    assert a._fingerprint() != b._fingerprint()
    assert a._fingerprint()["min_entry_age"] == 0
    assert b._fingerprint()["min_entry_age"] == 1


def test_restore_across_min_entry_age_is_refused() -> None:
    source = RolloutBuffer(**KW, min_entry_age=0)
    source.add_group("p", 0, _rollouts([1.0, 0.0]))
    state = source.state_dict()
    with pytest.raises(ValueError, match="different parameters"):
        RolloutBuffer(**KW, min_entry_age=1).load_state_dict(state)
    same = RolloutBuffer(**KW, min_entry_age=0)
    same.load_state_dict(state)
    assert same.size == 2


def test_durable_reopen_across_min_entry_age_is_refused(tmp_path: Path) -> None:
    buf = DurableRolloutBuffer(tmp_path, **KW, min_entry_age=1)
    buf.add_group("p", 0, _rollouts([1.0, 0.0]))
    buf.close()
    with pytest.raises(ValueError, match="different parameters"):
        DurableRolloutBuffer(tmp_path, **KW, min_entry_age=0)


def test_restore_checkpoint_across_min_entry_age_is_refused_and_keeps_the_live_state(tmp_path: Path) -> None:
    buf = DurableRolloutBuffer(tmp_path / "a", **KW, min_entry_age=1)
    buf.add_group("p", 0, _rollouts([1.0, 0.0]))
    buf.checkpoint("step-1")
    buf.close()
    other = DurableRolloutBuffer(tmp_path / "b", **KW, min_entry_age=0)
    other.add_group("q", 0, _rollouts([0.5]))
    (tmp_path / "b" / "checkpoints").mkdir(exist_ok=True)
    import shutil
    shutil.copytree(tmp_path / "a" / "checkpoints" / "step-1", tmp_path / "b" / "checkpoints" / "step-1")
    with pytest.raises(ValueError, match="different parameters"):
        other.restore_checkpoint("step-1")
    assert other.size == 1


# --- snapshots written before min_entry_age was fingerprinted -------------------------------


def _legacy(state: dict) -> dict:
    return {**state, "fingerprint": {k: v for k, v in state["fingerprint"].items() if k != "min_entry_age"}}


def _sampled_state(min_entry_age: int, attest: bool) -> dict:
    buf = RolloutBuffer(**KW, min_entry_age=min_entry_age, attest=AttestationLog() if attest else None)
    buf.add_group("p", 0, _rollouts([1.0, 0.0]))
    buf.sample(1, current_version=1)
    return buf.state_dict()


@pytest.mark.parametrize("floor", [0, 1])
def test_legacy_snapshot_takes_the_floor_its_log_declared(floor: int) -> None:
    legacy = _legacy(_sampled_state(floor, attest=True))
    RolloutBuffer(**KW, min_entry_age=floor, attest=AttestationLog()).load_state_dict(legacy)
    with pytest.raises(ValueError, match="different parameters"):
        RolloutBuffer(**KW, min_entry_age=1 - floor, attest=AttestationLog()).load_state_dict(legacy)


def test_legacy_snapshot_whose_samples_declare_no_floor_is_refused() -> None:
    state = _sampled_state(0, attest=True)
    for record in state["attestation"]:
        record.pop("min_entry_age", None)          # the 0.6.0 sample record shape
    with pytest.raises(ValueError, match="would not verify"):
        RolloutBuffer(**KW, min_entry_age=0, attest=AttestationLog()).load_state_dict(_legacy(state))


def test_legacy_unattested_snapshot_that_sampled_is_refused() -> None:
    legacy = _legacy(_sampled_state(1, attest=False))
    for floor in (0, 1):
        with pytest.raises(ValueError, match="floor it drew under is unknown"):
            RolloutBuffer(**KW, min_entry_age=floor).load_state_dict(legacy)


@pytest.mark.parametrize("attest", [False, True])
def test_legacy_snapshot_that_never_sampled_loads_under_any_floor(attest: bool) -> None:
    source = RolloutBuffer(**KW, attest=AttestationLog() if attest else None)
    source.add_group("p", 0, _rollouts([1.0, 0.0]))
    legacy = _legacy(source.state_dict())
    for floor in (0, 1):
        again = RolloutBuffer(**KW, min_entry_age=floor, attest=AttestationLog() if attest else None)
        again.load_state_dict(legacy)
        assert again.size == 2


def test_legacy_on_disk_snapshot_reopens_under_its_declared_floor_only(tmp_path: Path) -> None:
    attest = tmp_path / "a.jsonl"
    buf = DurableRolloutBuffer(tmp_path / "buf", attest=attest, **KW, min_entry_age=1)
    buf.add_group("p", 0, _rollouts([1.0, 0.0]))
    buf.sample(1, current_version=1)
    buf.compact()
    buf.close()
    path = tmp_path / "buf" / "state.json"
    snapshot = json.loads(path.read_text())
    del snapshot["digest"]                           # written by reservoir < 0.7.0
    del snapshot["buffer"]["fingerprint"]["min_entry_age"]
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="different parameters"), pytest.warns(RuntimeWarning, match="no digest"):
        DurableRolloutBuffer(tmp_path / "buf", attest=attest, **KW, min_entry_age=0)
    with pytest.warns(RuntimeWarning, match="no digest"):
        again = DurableRolloutBuffer(tmp_path / "buf", attest=attest, **KW, min_entry_age=1)
    assert again.size == 2
    again.close()
