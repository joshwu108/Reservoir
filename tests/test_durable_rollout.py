"""Tests for reservoir.durable_rollout — crash-atomic RolloutBuffer.

Three groups: snapshot round-trips on the in-memory buffer, reopen
recovery of the durable wrapper (state, counters, next draw, attestation
chain), and SIGKILL crash tests at every instrumented cut point during
an operation that forces stale evictions and a rebase. A crash test
passes only if the recovered state equals the pre-state or the
post-state exactly; anything else is a torn state.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from checker.verify import verify_json_lines
from reservoir.attest import AttestationLog
from reservoir.decay import inflated_priority
from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.priorities import PassRateVariance
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer

KW = dict(capacity=8, half_life=1, max_policy_age=2, seed=3)  # max_shift 2: rebases often


def rollouts(rewards: list[float], **meta) -> list[Rollout]:
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r, metadata=meta) for r in rewards]


def drive(buf, versions: range) -> None:
    for v in versions:
        buf.add_group(f"g{v}", v, rollouts([1.0, 0.0, 0.5]))
        batch = buf.sample(2, current_version=v)
        buf.update_priorities(batch.indices, [0.4, 0.7])


# ---------------------------------------------------------------------------
# state_dict / load_state_dict on the in-memory buffer
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_round_trip_is_exact(self) -> None:
        log = AttestationLog()
        a = RolloutBuffer(attest=log, **KW)
        drive(a, range(6))
        state = json.loads(json.dumps(a.state_dict()))  # must survive JSON
        b = RolloutBuffer(attest=AttestationLog(), **KW)
        b.load_state_dict(state)
        assert b.state_dict() == a.state_dict()
        assert b.live_positions() == a.live_positions()
        for pos in a.live_positions():
            assert b.leaf(pos) == a.leaf(pos)
            assert b.entry(pos)[0] == a.entry(pos)[0]
            assert b.entry(pos)[1] is not a.entry(pos)[1]  # rebuilt, not shared
            assert b.entry(pos)[1] == a.entry(pos)[1]
        assert b.attestation_log.head_digest == log.head_digest
        assert a.sample(4).draw_integers == b.sample(4).draw_integers
        assert b.verify_trees()

    def test_group_shared_by_its_rollouts_after_load(self) -> None:
        a = RolloutBuffer(**KW)
        a.add_group("p", 0, rollouts([1.0, 0.0, 0.5]))
        b = RolloutBuffer(**KW)
        b.load_state_dict(a.state_dict())
        groups = {id(b.entry(pos)[1]) for pos in b.live_positions()}
        assert len(groups) == 1

    def test_evicted_members_still_count_in_group_stats(self) -> None:
        a = RolloutBuffer(capacity=4, max_policy_age=100)
        a.add_group("p", 0, rollouts([1.0, 0.0, 0.0]))
        a.add_group("q", 1, rollouts([1.0, 1.0]))  # evicts one member of p
        b = RolloutBuffer(capacity=4, max_policy_age=100)
        b.load_state_dict(a.state_dict())
        p_pos = next(pos for pos in b.live_positions() if b.entry(pos)[1].prompt_id == "p")
        assert b.entry(p_pos)[1].size == 3
        assert b.entry(p_pos)[1].mean_reward == pytest.approx(1 / 3)

    def test_requires_fresh_buffer(self) -> None:
        a = RolloutBuffer(**KW)
        a.add_group("p", 0, rollouts([1.0]))
        with pytest.raises(ValueError, match="fresh"):
            a.load_state_dict(a.state_dict())

    def test_parameter_mismatch_rejected(self) -> None:
        a = RolloutBuffer(**KW)
        a.add_group("p", 0, rollouts([1.0]))
        b = RolloutBuffer(**{**KW, "half_life": 2})
        with pytest.raises(ValueError, match="different parameters"):
            b.load_state_dict(a.state_dict())

    def test_bad_format_rejected(self) -> None:
        with pytest.raises(ValueError, match="format"):
            RolloutBuffer(**KW).load_state_dict({"format": 99})

    def test_custom_predicate_cannot_be_saved(self) -> None:
        a = RolloutBuffer(**KW)
        a.add_group("p", 0, rollouts([1.0]), is_success=lambda r: True)
        with pytest.raises(ValueError, match="is_success"):
            a.state_dict()

    def test_unserialisable_metadata_rejected(self) -> None:
        a = RolloutBuffer(**KW)
        a.add_group("p", 0, rollouts([1.0], handle=object()))
        with pytest.raises(ValueError, match="metadata"):
            a.state_dict()

    def test_corrupt_entry_rejected(self) -> None:
        a = RolloutBuffer(**KW)
        a.add_group("p", 0, rollouts([1.0]))
        state = a.state_dict()
        state["slots"][0]["version"] = 50  # newer than current_version 0
        with pytest.raises(ValueError):
            RolloutBuffer(**KW).load_state_dict(state)

    @pytest.mark.parametrize("edit", [
        lambda s: s["slots"][0].__setitem__("group", -1),
        lambda s: s["slots"][0].__setitem__("group", 5),
        lambda s: s["slots"][0].__setitem__("member", 7),
        lambda s: s["slots"][0].__setitem__("member", 1.0),
        lambda s: s["slots"][0].__setitem__("inserted", 0),
        lambda s: s["slots"][0].__setitem__("q", "1.5"),
        lambda s: s["slots"].__setitem__(1, dict(s["slots"][0])),           # duplicate rollout
        lambda s: s["slots"].pop(),                                        # wrong slot count
        lambda s: s.__setitem__("draw_counter", -1),
        lambda s: s.__setitem__("insert_seq", 0),                          # below a slot's counter
        lambda s: s.__setitem__("attestation", "not a list"),
    ])
    def test_malformed_slots_and_counters_rejected(self, edit) -> None:
        a = RolloutBuffer(**KW)
        a.add_group("p", 0, rollouts([1.0, 0.0]))
        state = a.state_dict()
        edit(state)
        with pytest.raises(ValueError):
            RolloutBuffer(**KW).load_state_dict(state)

    def test_attestation_mismatch_rejected(self) -> None:
        a = RolloutBuffer(attest=AttestationLog(), **KW)
        a.add_group("p", 0, rollouts([1.0]))
        with pytest.raises(ValueError, match="attestation"):
            RolloutBuffer(**KW).load_state_dict(a.state_dict())


# ---------------------------------------------------------------------------
# Durable wrapper: reopen recovers everything
# ---------------------------------------------------------------------------

class TestReopen:
    def test_fresh_directory_is_empty(self, tmp_path: Path) -> None:
        buf = DurableRolloutBuffer(tmp_path, **KW)
        assert buf.size == 0 and buf.current_version == 0

    def test_state_counters_and_next_draw_survive_reopen(self, tmp_path: Path) -> None:
        attest = tmp_path / "attest.jsonl"
        first = DurableRolloutBuffer(tmp_path / "buf", attest=attest, **KW)
        drive(first, range(7))
        assert first.n_rebases >= 1
        snapshot = first.state_dict()
        head = first.attestation_log.head_digest
        twin = RolloutBuffer(attest=AttestationLog(), **KW)
        twin.load_state_dict(snapshot)
        first.close()

        second = DurableRolloutBuffer(tmp_path / "buf", attest=attest, **KW)
        assert second.state_dict() == snapshot
        assert second.attestation_log.head_digest == head
        assert second.sample(4, current_version=7).draw_integers == twin.sample(4, current_version=7).draw_integers
        assert second.verify_trees()
        for pos in second.live_positions():
            q, t = second.base_priority(pos), second.entry_version(pos)
            assert second.leaf(pos) == inflated_priority(q, t, second.base_epoch, second.params)
        second.close()
        verify_json_lines(attest.read_text())

    def test_attestation_file_rewritten_from_state(self, tmp_path: Path) -> None:
        attest = tmp_path / "attest.jsonl"
        buf = DurableRolloutBuffer(tmp_path / "buf", attest=attest, **KW)
        drive(buf, range(3))
        expected = buf.attestation_log.to_json_lines()
        buf.close()
        attest.write_text("garbage that a crash might leave\n")
        again = DurableRolloutBuffer(tmp_path / "buf", attest=attest, **KW)
        assert again.attestation_log.to_json_lines() == expected
        assert attest.read_text().strip() == expected
        again.close()

    def test_config_record_persisted_before_first_operation(self, tmp_path: Path) -> None:
        attest = tmp_path / "attest.jsonl"
        DurableRolloutBuffer(tmp_path / "buf", attest=attest, **KW).close()
        again = DurableRolloutBuffer(tmp_path / "buf", attest=attest, **KW)
        assert [r["op"] for r in again.attestation_log.records] == ["decay_config"]
        again.close()

    def test_parameter_mismatch_on_reopen(self, tmp_path: Path) -> None:
        buf = DurableRolloutBuffer(tmp_path, **KW)
        buf.add_group("p", 0, rollouts([1.0]))
        with pytest.raises(ValueError, match="different parameters"):
            DurableRolloutBuffer(tmp_path, **{**KW, "max_policy_age": 5})

    def test_attest_setting_must_match_saved_state(self, tmp_path: Path) -> None:
        buf = DurableRolloutBuffer(tmp_path / "buf", attest=tmp_path / "a.jsonl", **KW)
        buf.add_group("p", 0, rollouts([1.0]))
        buf.close()
        with pytest.raises(ValueError, match="attest"):
            DurableRolloutBuffer(tmp_path / "buf", **KW)
        plain = DurableRolloutBuffer(tmp_path / "plain", **KW)
        plain.add_group("p", 0, rollouts([1.0]))
        with pytest.raises(ValueError, match="attest"):
            DurableRolloutBuffer(tmp_path / "plain", attest=tmp_path / "b.jsonl", **KW)

    def test_in_memory_log_and_managed_kwargs_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="path"):
            DurableRolloutBuffer(tmp_path, attest=AttestationLog(), **KW)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="attest_overwrite"):
            DurableRolloutBuffer(tmp_path, attest_overwrite=True, **KW)

    def test_failed_operation_leaves_no_intent_and_state_unchanged(self, tmp_path: Path) -> None:
        buf = DurableRolloutBuffer(tmp_path, **KW)
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        before = buf.state_dict()
        with pytest.raises(ValueError):
            buf.update_priorities([0, 7], [1.0, 1.0])  # slot 7 is empty
        with pytest.raises(ValueError, match="metadata"):
            buf.add_group("q", 0, rollouts([1.0], handle=object()))
        assert buf.state_dict() == before
        assert not (tmp_path / "intent.json").exists()
        assert not (tmp_path / "intent.json.tmp").exists()
        again = DurableRolloutBuffer(tmp_path, **KW)
        assert again.state_dict() == before

    def test_operation_that_raises_after_mutating_is_rolled_back(self, tmp_path: Path) -> None:
        attest = tmp_path / "a.jsonl"
        buf = DurableRolloutBuffer(tmp_path / "buf", attest=attest, **KW)
        buf.add_group("p", 1, rollouts([1.0, 0.0]))
        before = buf.state_dict()
        # Advancing to 100 evicts everything and rebases, then the sample fails.
        with pytest.raises(RuntimeError, match="no live"):
            buf.sample(1, current_version=100)
        assert buf.state_dict() == before
        assert buf.current_version == 1 and buf.size == 2
        assert attest.read_text().strip() == buf.attestation_log.to_json_lines()
        buf.close()
        assert DurableRolloutBuffer(tmp_path / "buf", attest=attest, **KW).state_dict() == before

    def test_corrupt_state_file_is_an_error_not_a_fresh_start(self, tmp_path: Path) -> None:
        attest = tmp_path / "a.jsonl"
        buf = DurableRolloutBuffer(tmp_path / "buf", attest=attest, **KW)
        buf.add_group("p", 0, rollouts([1.0]))
        buf.close()
        log_before = attest.read_text()
        (tmp_path / "buf" / "state.json").write_text("{not json")
        with pytest.raises(ValueError, match="corrupt"):
            DurableRolloutBuffer(tmp_path / "buf", attest=attest, **KW)
        assert attest.read_text() == log_before  # nothing was overwritten
        (tmp_path / "buf" / "state.json").write_text("[1, 2]")
        with pytest.raises(ValueError, match="corrupt"):
            DurableRolloutBuffer(tmp_path / "buf", attest=attest, **KW)

    def test_failed_reopen_does_not_touch_attestation_file(self, tmp_path: Path) -> None:
        attest = tmp_path / "a.jsonl"
        buf = DurableRolloutBuffer(tmp_path / "buf", attest=attest, **KW)
        drive(buf, range(3))
        buf.close()
        log_before = attest.read_text()
        with pytest.raises(ValueError, match="different parameters"):
            DurableRolloutBuffer(tmp_path / "buf", attest=attest, **{**KW, "seed": 99})
        assert attest.read_text() == log_before

    def test_strategy_parameters_are_part_of_the_fingerprint(self, tmp_path: Path) -> None:
        from reservoir.priorities import AdvantagePriority

        buf = DurableRolloutBuffer(tmp_path, priority=AdvantagePriority(epsilon=0.1), **KW)
        buf.add_group("p", 0, rollouts([1.0]))
        with pytest.raises(ValueError, match="different parameters"):
            DurableRolloutBuffer(tmp_path, priority=AdvantagePriority(epsilon=0.5), **KW)

    def test_metadata_that_json_would_alter_is_rejected(self, tmp_path: Path) -> None:
        buf = DurableRolloutBuffer(tmp_path, **KW)
        with pytest.raises(ValueError, match="alter"):
            buf.add_group("p", 0, rollouts([1.0], pair=(1, 2)))
        with pytest.raises(ValueError, match="alter"):
            buf.add_group("p", 0, rollouts([1.0], table={1: "a"}))
        assert buf.size == 0

    def test_strategy_round_trips_by_name(self, tmp_path: Path) -> None:
        kw = {**KW, "priority": PassRateVariance()}
        buf = DurableRolloutBuffer(tmp_path, **kw)
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        again = DurableRolloutBuffer(tmp_path, **kw)
        assert again.size == 2


# ---------------------------------------------------------------------------
# Crash tests: SIGKILL at each cut point during an operation with a rebase
# ---------------------------------------------------------------------------

CUT_POINTS = [
    "after_intent_write",
    "after_intent_fsync",
    "mid_segment_write",
    "after_segment_fsync",
    "before_rename",
    "after_rename_before_dir_fsync",
    "after_dir_fsync",
]

SETUP = textwrap.dedent(
    """
    import json, sys
    from reservoir.durable_rollout import DurableRolloutBuffer
    from reservoir.rollout import Rollout
    KW = dict(capacity=8, half_life=1, max_policy_age=2, seed=3)
    def rollouts(rs):
        return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in rs]
    """
)

# Pre-state: two groups at versions 0 and 1. The crashing op adds a group at
# version 4, which expires both (ages 4 and 3 > 2) and forces a rebase
# (epoch 4 > max_shift 2) before the inserts: evictions + rebase + writes
# must land together or not at all.
PREPARE = SETUP + textwrap.dedent(
    """
    buf = DurableRolloutBuffer(sys.argv[1], attest=sys.argv[2], **KW)
    buf.add_group("g0", 0, rollouts([1.0, 0.0, 0.5]))
    buf.add_group("g1", 1, rollouts([0.0, 1.0]))
    buf.close()
    """
)
CRASHING_OP = SETUP + textwrap.dedent(
    """
    buf = DurableRolloutBuffer(sys.argv[1], attest=sys.argv[2], **KW)
    buf.add_group("g4", 4, rollouts([1.0, 0.0]))
    buf.close()
    """
)
DUMP = SETUP + textwrap.dedent(
    """
    buf = DurableRolloutBuffer(sys.argv[1], attest=sys.argv[2], **KW)
    assert buf.verify_trees()
    print(json.dumps(buf.state_dict(), sort_keys=True))
    buf.close()
    """
)


def run_script(script: str, directory: Path, attest: Path, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("RESERVOIR_CUT")}
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", script, str(directory), str(attest)],
        capture_output=True, text=True, env=env, cwd=Path(__file__).parent.parent,
    )


def dump_state(directory: Path, attest: Path) -> dict:
    result = run_script(DUMP, directory, attest)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict, dict]:
    """Pre-state directory plus the pre and post oracles, built once per module.

    Each subprocess pays the library's import cost, so the pre-state is
    prepared once and copied for every cut point instead of rebuilt.
    """
    root = tmp_path_factory.mktemp("crash")
    buf_dir, attest = root / "pre", root / "pre.jsonl"
    assert run_script(PREPARE, buf_dir, attest).returncode == 0
    pre = dump_state(buf_dir, attest)
    oracle_dir, oracle_attest = root / "oracle", root / "oracle.jsonl"
    shutil.copytree(buf_dir, oracle_dir)
    shutil.copy(attest, oracle_attest)
    assert run_script(CRASHING_OP, oracle_dir, oracle_attest).returncode == 0
    post = dump_state(oracle_dir, oracle_attest)
    assert pre != post
    return root, pre, post


@pytest.mark.parametrize("cut", CUT_POINTS)
def test_sigkill_during_rebasing_add_group_leaves_pre_or_post_state(
    prepared: tuple[Path, dict, dict], cut: str
) -> None:
    root, pre, post = prepared
    buf_dir, attest = root / f"buf_{cut}", root / f"attest_{cut}.jsonl"
    shutil.copytree(root / "pre", buf_dir)
    shutil.copy(root / "pre.jsonl", attest)

    crashed = run_script(
        CRASHING_OP, buf_dir, attest,
        {"RESERVOIR_CUT_POINT": cut, "RESERVOIR_CUT_BYTE_OFFSET": "40"},
    )
    assert crashed.returncode != 0, f"child was not killed at cut {cut}"

    recovered = dump_state(buf_dir, attest)
    assert recovered in (pre, post), f"torn state after SIGKILL at {cut}"
    # And the recovered log verifies independently either way.
    verify_json_lines(attest.read_text())
