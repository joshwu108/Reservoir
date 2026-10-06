"""
campaigns.crash — T3 kill-9 crash atomicity campaign.

For each (operation type × cut point × seed), spawns a child subprocess
that arms a cut point and runs one operation; the durability protocol
SIGKILLs the process at that cut. The campaign then recovers the buffer
and compares the recovered logical state against the oracle pre-state and
post-state. Exactly one match is required (never a torn hybrid), and the
cut must actually have fired: a child that exits normally proves nothing
about crash atomicity and is recorded as a failure, not a pass.

Operation types:
  - insert, update            (classic DurableBuffer)
  - rollout_add_group,        (DurableRolloutBuffer; the add forces stale
    rollout_update,            evictions and a rebase inside one operation;
    rollout_quarantine,        the quarantine evicts a whole group with a
    rollout_restore            reasoned record; the restore rewinds to a
                               checkpoint)
Cut points:
  - after_intent_write
  - after_intent_fsync
  - mid_segment_write
  - after_segment_fsync
  - before_rename
  - after_rename_before_dir_fsync
  - after_dir_fsync

Run with: python -m campaigns.crash
"""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import sys
import tempfile
from pathlib import Path

# Set spawn start method explicitly (required by spec)
multiprocessing.set_start_method("spawn", force=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from reservoir.buffer import ExactPERBuffer, Transition
from reservoir.durable import DurableBuffer
from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.rollout import Rollout


# ---------------------------------------------------------------------------
# Subprocess worker: performs one operation then is killed at the cut point
# ---------------------------------------------------------------------------

def _worker_insert(directory: str, seed: int, cut_point: str, cut_byte: int) -> None:
    """Child process: insert a transition, cut at cut_point."""
    env_vars = {
        "RESERVOIR_CUT_POINT": cut_point,
        "RESERVOIR_CUT_BYTE_OFFSET": str(cut_byte),
    }
    for k, v in env_vars.items():
        os.environ[k] = v

    buf = DurableBuffer(directory, capacity=4, seed=seed)
    t = Transition(state=42, action=1, reward=3.14, next_state=43, done=False)
    try:
        buf.insert(t, td_error=5.0)
    except (SystemExit, Exception):
        pass
    sys.exit(0)


def _worker_update(directory: str, seed: int, cut_point: str, position: int, cut_byte: int) -> None:
    """Child process: update a priority, cut at cut_point."""
    os.environ["RESERVOIR_CUT_POINT"] = cut_point
    os.environ["RESERVOIR_CUT_BYTE_OFFSET"] = str(cut_byte)  # the mid-segment cut needs it
    buf = DurableBuffer(directory, capacity=4, seed=seed)
    try:
        buf.update_priority(position, td_error=9.9)
    except (SystemExit, Exception):
        pass
    sys.exit(0)


# ---------------------------------------------------------------------------
# State snapshot for comparison
# ---------------------------------------------------------------------------

def _get_state_snapshot(directory: str, capacity: int, seed: int) -> dict:
    """Recover and return the logical state of the buffer."""
    # Clear cut env vars for recovery
    for k in ["RESERVOIR_CUT_POINT", "RESERVOIR_CUT_BYTE_OFFSET"]:
        os.environ.pop(k, None)

    buf = DurableBuffer(directory, capacity=capacity, seed=seed)
    priorities = [buf._buf._sum_tree.get(i) for i in range(buf.capacity)]
    transitions = [
        buf._buf._transitions[i] is not None
        for i in range(buf.capacity)
    ]
    return {
        "size": buf.size,
        "write_pos": buf._buf._write_pos,
        "priorities": priorities,
        "transitions": transitions,
    }


# ---------------------------------------------------------------------------
# Single crash test
# ---------------------------------------------------------------------------

def run_crash_test(
    op_type: str,
    cut_point: str,
    seed: int,
    base_tmpdir: str,
) -> dict:
    """Run one crash test: spawn child, kill at cut, recover, verify.

    Returns a result dict.
    """
    test_dir = os.path.join(base_tmpdir, f"{op_type}_{cut_point}_{seed}")
    os.makedirs(test_dir, exist_ok=True)

    capacity = 4

    # --- Build pre-state: a buffer with some existing state ---
    for k in ["RESERVOIR_CUT_POINT", "RESERVOIR_CUT_BYTE_OFFSET"]:
        os.environ.pop(k, None)

    buf = DurableBuffer(test_dir, capacity=capacity, seed=seed)
    # Pre-populate with 2 transitions
    for i in range(2):
        t = Transition(state=i, action=0, reward=float(i), next_state=i + 1, done=False)
        buf.insert(t, td_error=float(i + 1))

    # Record pre-state oracle
    pre_state = _get_state_snapshot(test_dir, capacity, seed)

    # --- Compute post-state oracle (apply op to a fresh copy) ---
    # To get the post-state, we apply the operation without cut, on a fresh dir
    oracle_dir = test_dir + "_oracle"
    os.makedirs(oracle_dir, exist_ok=True)
    oracle_buf = DurableBuffer(oracle_dir, capacity=capacity, seed=seed)
    for i in range(2):
        t = Transition(state=i, action=0, reward=float(i), next_state=i + 1, done=False)
        oracle_buf.insert(t, td_error=float(i + 1))

    if op_type == "insert":
        t = Transition(state=42, action=1, reward=3.14, next_state=43, done=False)
        oracle_buf.insert(t, td_error=5.0)
    elif op_type == "update":
        oracle_buf.update_priority(0, td_error=9.9)

    post_state = _get_state_snapshot(oracle_dir, capacity, seed)

    # --- Spawn child; the protocol kills it at the armed cut point ---
    cut_byte = 8  # small byte offset for mid_segment_write
    ctx = multiprocessing.get_context("spawn")
    if op_type == "insert":
        p = ctx.Process(
            target=_worker_insert,
            args=(test_dir, seed, cut_point, cut_byte),
        )
    else:  # update
        p = ctx.Process(
            target=_worker_update,
            args=(test_dir, seed, cut_point, 0, cut_byte),
        )

    p.start()
    cut_fired = _wait_for_cut(p)

    # --- Recover and compare ---
    recovered_state = _get_state_snapshot(test_dir, capacity, seed)
    return _verdict(op_type, cut_point, seed, recovered_state, pre_state, post_state, cut_fired)


def _wait_for_cut(p, timeout: float = 120.0) -> bool:
    """Wait for the child; True iff the durability protocol killed it with SIGKILL.

    The child imports torch (seconds) before it reaches the operation, so the
    wait is generous. A child still alive after the timeout is killed and
    counted as "cut never fired".
    """
    p.join(timeout=timeout)
    if p.is_alive():
        os.kill(p.pid, signal.SIGKILL)
        p.join(timeout=5)
        return False
    return p.exitcode == -signal.SIGKILL


def _verdict(op_type, cut_point, seed, recovered, pre_state, post_state, cut_fired: bool) -> dict:
    """Pass iff the cut fired and the recovered state is exactly the pre- or post-state."""
    matches_pre = recovered == pre_state
    matches_post = recovered == post_state
    torn = not matches_pre and not matches_post
    return {
        "op": op_type,
        "cut": cut_point,
        "seed": seed,
        "cut_fired": cut_fired,
        "matches_pre": matches_pre,
        "matches_post": matches_post,
        "torn": torn,
        "passed": cut_fired and (matches_pre or matches_post),
    }


# ---------------------------------------------------------------------------
# Rollout buffer variant
# ---------------------------------------------------------------------------

_ROLLOUT_KW = dict(capacity=8, half_life=1, max_policy_age=2, compact_every=1)

# The rollout buffer logs each command and, with compact_every=1, snapshots
# right after, so its crashing operation passes the log cuts and then the
# snapshot protocol's cuts. The classic buffer has only the latter.
ROLLOUT_CUT_POINTS = [
    "mid_wal_write", "after_wal_write", "after_wal_fsync",
    "after_intent_write", "after_intent_fsync", "mid_segment_write", "after_segment_fsync",
    "before_rename", "after_rename_before_dir_fsync", "after_snapshot_before_wal_reset", "after_dir_fsync",
]
# Restoring a checkpoint writes a snapshot (the protocol's cuts) and then
# resets the log; no command is appended, so the WAL cuts cannot fire there.
RESTORE_CUT_POINTS = [
    "after_intent_write", "after_intent_fsync", "mid_segment_write", "after_segment_fsync",
    "before_rename", "after_rename_before_dir_fsync", "after_dir_fsync", "after_restore_before_wal_reset",
]


def _cuts_for(op_type: str) -> list[str]:
    if op_type == "rollout_restore":
        return RESTORE_CUT_POINTS
    return ROLLOUT_CUT_POINTS if op_type.startswith("rollout_") else CUT_POINTS


def _rollouts(rewards):
    """Minimal two-token rollouts with the given rewards."""
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in rewards]


def _rollout_prepare(directory: str, seed: int) -> None:
    """Two groups at versions 0 and 1; the add at version 4 will expire both and rebase."""
    buf = DurableRolloutBuffer(directory, seed=seed, **_ROLLOUT_KW)
    buf.add_group("g0", 0, _rollouts([1.0, 0.0, 0.5]))
    buf.checkpoint("a")                      # what rollout_restore rewinds to
    buf.add_group("g1", 1, _rollouts([0.0, 1.0]))
    buf.sample(2, current_version=1)
    buf.close()


def _rollout_apply(directory: str, seed: int, op_type: str) -> None:
    """The operation under test: a group add that evicts and rebases, a quarantine, a restore, or a priority update."""
    buf = DurableRolloutBuffer(directory, seed=seed, **_ROLLOUT_KW)
    if op_type == "rollout_add_group":
        buf.add_group("g4", 4, _rollouts([1.0, 0.0]))
    elif op_type == "rollout_quarantine":
        buf.quarantine(lambda r, g: g.prompt_id == "g1", "campaign: g1 is quarantined")
    elif op_type == "rollout_restore":
        buf.restore_checkpoint("a")
    else:
        buf.update_priorities([0, 1], [0.9, 0.1])
    buf.close()


def _rollout_worker(directory: str, seed: int, op_type: str, cut_point: str, cut_byte: int) -> None:
    """Child process body: arm the cut point, then run the operation until SIGKILL."""
    os.environ["RESERVOIR_CUT_POINT"] = cut_point
    os.environ["RESERVOIR_CUT_BYTE_OFFSET"] = str(cut_byte)
    _rollout_apply(directory, seed, op_type)


def _rollout_snapshot(directory: str, seed: int) -> dict:
    """Recover the directory with cut points disarmed and return its full state."""
    for k in ["RESERVOIR_CUT_POINT", "RESERVOIR_CUT_BYTE_OFFSET"]:
        os.environ.pop(k, None)
    buf = DurableRolloutBuffer(directory, seed=seed, **_ROLLOUT_KW)
    assert buf.verify_trees()
    state = buf.state_dict()
    buf.close()
    return state


def run_rollout_crash_test(op_type: str, cut_point: str, seed: int, base_tmpdir: str) -> dict:
    """Same shape as run_crash_test, for DurableRolloutBuffer."""
    test_dir = os.path.join(base_tmpdir, f"{op_type}_{cut_point}_{seed}")
    oracle_dir = test_dir + "_oracle"
    for k in ["RESERVOIR_CUT_POINT", "RESERVOIR_CUT_BYTE_OFFSET"]:
        os.environ.pop(k, None)

    _rollout_prepare(test_dir, seed)
    pre_state = _rollout_snapshot(test_dir, seed)
    _rollout_prepare(oracle_dir, seed)
    _rollout_apply(oracle_dir, seed, op_type)
    post_state = _rollout_snapshot(oracle_dir, seed)

    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_rollout_worker, args=(test_dir, seed, op_type, cut_point, 40))
    p.start()
    cut_fired = _wait_for_cut(p)

    recovered = _rollout_snapshot(test_dir, seed)
    return _verdict(op_type, cut_point, seed, recovered, pre_state, post_state, cut_fired)


# ---------------------------------------------------------------------------
# Full campaign
# ---------------------------------------------------------------------------

OP_TYPES = ["insert", "update", "rollout_add_group", "rollout_update", "rollout_quarantine", "rollout_restore"]

CUT_POINTS = [
    "after_intent_write",
    "after_intent_fsync",
    "mid_segment_write",
    "after_segment_fsync",
    "before_rename",
    "after_rename_before_dir_fsync",
    "after_dir_fsync",
]

SEEDS = [0, 1, 2, 3, 4]


def run_campaign() -> dict:
    """Run the full T3 crash campaign.

    Returns campaign results dict.
    """
    results = []
    total = 0
    passed = 0
    torn = 0
    fired = 0

    with tempfile.TemporaryDirectory(prefix="reservoir_crash_") as tmpdir:
        for op in OP_TYPES:
            for cut in _cuts_for(op):
                for seed in SEEDS:
                    runner = run_rollout_crash_test if op.startswith("rollout_") else run_crash_test
                    result = runner(op, cut, seed, tmpdir)
                    results.append(result)
                    total += 1
                    if result["passed"]:
                        passed += 1
                    if result["torn"]:
                        torn += 1
                    if result["cut_fired"]:
                        fired += 1

    return {
        "total": total,
        "passed": passed,
        "torn": torn,
        "cut_fired": fired,
        "cut_never_fired": total - fired,
        "failed": total - passed,
        "results": results,
        # Evidence only if every child was killed at its cut and none recovered torn.
        "pass": torn == 0 and fired == total and passed == total,
    }


def main() -> None:
    print("=" * 70)
    print("T3 Crash Atomicity Campaign")
    print("=" * 70)
    print()
    print(f"Operations: {OP_TYPES}")
    print(f"Cut points: {len(CUT_POINTS)}")
    print(f"Seeds: {SEEDS}")
    print(f"Total tests: {sum(len(_cuts_for(op)) for op in OP_TYPES) * len(SEEDS)}")
    print()

    campaign = run_campaign()

    print(f"Total: {campaign['total']}")
    print(f"Cut fired (child SIGKILLed at its cut point): {campaign['cut_fired']}")
    print(f"Cut never fired (child completed; proves nothing): {campaign['cut_never_fired']}")
    print(f"Passed: {campaign['passed']}")
    print(f"Torn (bugs!): {campaign['torn']}")
    print()
    if campaign["cut_never_fired"]:
        print("CUTS THAT NEVER FIRED:")
        for r in campaign["results"]:
            if not r["cut_fired"]:
                print(f"  NOT FIRED: op={r['op']}, cut={r['cut']}, seed={r['seed']}")

    if campaign["torn"] > 0:
        print("TORN STATES DETECTED:")
        for r in campaign["results"]:
            if r["torn"]:
                print(f"  TORN: op={r['op']}, cut={r['cut']}, seed={r['seed']}")

    status = "PASS" if campaign["pass"] else "FAIL"
    print(f"RESULT: {status}")

    Path("results").mkdir(exist_ok=True)
    Path("results/crash_campaign_report.json").write_text(
        json.dumps(campaign, indent=2, default=str)
    )
    print()
    print("Report written to results/crash_campaign_report.json")


if __name__ == "__main__":
    main()
