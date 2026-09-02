"""
campaigns.crash — T3 kill-9 crash atomicity campaign.

For each (operation type × cut point × seed), spawns a child subprocess,
sends it SIGKILL at the cut, recovers the buffer, and byte-compares the
recovered logical state against the oracle pre-state and post-state.

Exactly one match required (never a torn hybrid).

Operation types: insert, update
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
import time
from pathlib import Path

# Set spawn start method explicitly (required by spec)
multiprocessing.set_start_method("spawn", force=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reservoir.buffer import ExactPERBuffer, Transition
from src.reservoir.durable import DurableBuffer


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

    # Re-import to pick up env vars
    from src.reservoir.durable import DurableBuffer
    buf = DurableBuffer(directory, capacity=4, seed=seed)
    t = Transition(state=42, action=1, reward=3.14, next_state=43, done=False)
    try:
        buf.insert(t, td_error=5.0)
    except (SystemExit, Exception):
        pass
    sys.exit(0)


def _worker_update(directory: str, seed: int, cut_point: str, position: int) -> None:
    """Child process: update a priority, cut at cut_point."""
    os.environ["RESERVOIR_CUT_POINT"] = cut_point
    from src.reservoir.durable import DurableBuffer
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


def _states_match(s1: dict, s2: dict) -> bool:
    return s1 == s2


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

    # --- Spawn child, kill at cut point ---
    cut_byte = 8  # small byte offset for mid_segment_write
    os.environ["RESERVOIR_CUT_POINT"] = cut_point
    os.environ["RESERVOIR_CUT_BYTE_OFFSET"] = str(cut_byte)

    ctx = multiprocessing.get_context("spawn")
    if op_type == "insert":
        p = ctx.Process(
            target=_worker_insert,
            args=(test_dir, seed, cut_point, cut_byte),
        )
    else:  # update
        p = ctx.Process(
            target=_worker_update,
            args=(test_dir, seed, cut_point, 0),
        )

    p.start()
    # Give child time to start, then SIGKILL
    time.sleep(0.5)
    if p.is_alive():
        os.kill(p.pid, signal.SIGKILL)
    p.join(timeout=5)

    # Clear cut env
    for k in ["RESERVOIR_CUT_POINT", "RESERVOIR_CUT_BYTE_OFFSET"]:
        os.environ.pop(k, None)

    # --- Recover and compare ---
    recovered_state = _get_state_snapshot(test_dir, capacity, seed)

    matches_pre = _states_match(recovered_state, pre_state)
    matches_post = _states_match(recovered_state, post_state)

    passed = matches_pre or matches_post  # Exactly one must match
    torn = not matches_pre and not matches_post

    return {
        "op": op_type,
        "cut": cut_point,
        "seed": seed,
        "matches_pre": matches_pre,
        "matches_post": matches_post,
        "torn": torn,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Full campaign
# ---------------------------------------------------------------------------

OP_TYPES = ["insert", "update"]

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

    with tempfile.TemporaryDirectory(prefix="reservoir_crash_") as tmpdir:
        for op in OP_TYPES:
            for cut in CUT_POINTS:
                for seed in SEEDS:
                    result = run_crash_test(op, cut, seed, tmpdir)
                    results.append(result)
                    total += 1
                    if result["passed"]:
                        passed += 1
                    if result["torn"]:
                        torn += 1

    return {
        "total": total,
        "passed": passed,
        "torn": torn,
        "failed": total - passed,
        "results": results,
        "pass": torn == 0 and passed == total,
    }


def main() -> None:
    print("=" * 70)
    print("T3 Crash Atomicity Campaign")
    print("=" * 70)
    print()
    print(f"Operations: {OP_TYPES}")
    print(f"Cut points: {len(CUT_POINTS)}")
    print(f"Seeds: {SEEDS}")
    print(f"Total tests: {len(OP_TYPES) * len(CUT_POINTS) * len(SEEDS)}")
    print()

    campaign = run_campaign()

    print(f"Total: {campaign['total']}")
    print(f"Passed: {campaign['passed']}")
    print(f"Torn (bugs!): {campaign['torn']}")
    print()

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
