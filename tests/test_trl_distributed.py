"""Tests for ``reservoir.integrations._trl_distributed``: replay under more than one process.

TRL gives every process the rank-ordered slice of one generation batch
(``advantages[process_slice]``), so prompt groups may straddle ranks and
only the concatenation of all slices has the ``[g*G, (g+1)*G)`` layout the
adapter relies on. Rank 0 owns the buffer and the single log writer; every
rank computes behavior logprobs for its own rows, the slices are gathered
to rank 0, rank 0 runs the ordinary single-process hook on the global
batch, and the result is broadcast and sliced back.

``FakeAccelerator`` simulates N processes in one interpreter: each rank's
trainer runs on its own thread and the collectives rendezvous on a barrier,
so a rank that skipped a collective would deadlock the test (the barrier
timeout turns that into a failure). Nothing here imports accelerate.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from reservoir.integrations import _trl_distributed as dist
from reservoir.integrations._trl_lifecycle import env_rank
from reservoir.integrations.trl import ReservoirReplay, bind_checkpoint, resume_from_checkpoint
from reservoir_checker.verify import verify_json_lines
from tests.test_trl_replay import FakeTrainer, live_batch, mixed_batch, replay
from tests.test_trl_rows import PAD, make_output

TIMEOUT = 20.0


class FakeAccelerator:
    """One rank's view of a world of ``num_processes`` threads sharing a ``World``."""

    def __init__(self, world: "World", rank: int) -> None:
        self.world = world
        self.process_index = rank
        self.num_processes = world.size
        self.is_main_process = rank == 0

    def gather_object(self, obj):
        return self.world.exchange(self.process_index, obj)

    def broadcast_object(self, obj):
        return self.world.exchange(self.process_index, obj)[0]


class World:
    def __init__(self, size: int) -> None:
        self.size = size
        self.slots: list = [None] * size
        self.barrier = threading.Barrier(size, timeout=TIMEOUT)
        self.collectives = 0

    def exchange(self, rank: int, obj) -> list:
        self.slots[rank] = obj
        self.barrier.wait()
        got = list(self.slots)
        if rank == 0:
            self.collectives += 1
        self.barrier.wait()
        return got

    def accelerators(self) -> list[FakeAccelerator]:
        return [FakeAccelerator(self, r) for r in range(self.size)]


def run_ranks(fns):
    """Run one callable per rank concurrently; re-raise the first failure."""
    with ThreadPoolExecutor(max_workers=len(fns)) as pool:
        futures = [pool.submit(fn) for fn in fns]
        return [f.result(timeout=TIMEOUT) for f in futures]


def shard(output: dict, rank: int, world: int) -> dict:
    """Rank ``rank``'s slice of ``output``, trimmed to its own widths as TRL pads per process."""
    rows = output["advantages"].size(0)
    per = rows // world
    sl = slice(rank * per, (rank + 1) * per)
    out = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.size(0) == rows:
            out[key] = value[sl].clone()
        else:
            out[key] = value
    cw = max(int(out["completion_mask"].sum(dim=1).max()), 1)
    for key in ("completion_ids", "completion_mask", "old_per_token_logps", "ref_per_token_logps"):
        if key in out:
            out[key] = out[key][:, :cw].clone()
    pw = max(int(out["prompt_mask"].sum(dim=1).max()), 1)
    for key in ("prompt_ids", "prompt_mask"):
        out[key] = out[key][:, -pw:].clone()
    return out


def global_batch(prompt_base: int = 10) -> dict:
    """Four groups of G=2: three live, one dead (rows 6-7)."""
    return make_output(
        prompts=[[prompt_base, 1], [prompt_base, 1], [prompt_base + 1], [prompt_base + 1],
                 [prompt_base + 2, 5, 6], [prompt_base + 2, 5, 6], [prompt_base + 3], [prompt_base + 3]],
        completions=[[3, 4, 5], [6], [7, 8], [9, 9, 9], [1, 2, 3, 4], [2], [5, 5], [5]],
        advantages=[1.0, -1.0, 0.5, -0.5, 0.75, -0.75, 0.0, 0.0],
    )


def wide_live_batch() -> dict:
    """Two live groups of G=4."""
    return make_output(
        prompts=[[40, 1]] * 4 + [[41]] * 4,
        completions=[[3, 4, 5], [6], [7, 8], [9, 9, 9], [1, 2], [2], [5, 5, 5, 5], [5]],
        advantages=[1.0, -1.0, 0.5, -0.5, 0.75, -0.75, 0.25, -0.25],
    )


def wide_group_batch() -> dict:
    """Two groups of G=4: rows 0-3 live, rows 4-7 dead. On four ranks every group straddles two."""
    return make_output(
        prompts=[[30, 1]] * 4 + [[31, 2, 3]] * 4,
        completions=[[3, 4, 5], [6], [7, 8], [9, 9, 9], [1, 2, 3, 4], [2], [5, 5], [5]],
        advantages=[1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0],
    )


def dead_batch() -> dict:
    """Two dead groups of G=2: every row is replaced from the buffer, none is stored."""
    return make_output(
        prompts=[[50], [50], [51, 2], [51, 2]],
        completions=[[1], [2, 2], [3], [4, 4, 4]],
        advantages=[0.0, 0.0, 0.0, 0.0],
    )


def replay_on_rank(rank: int, **kwargs) -> ReservoirReplay:
    """Construct a ``ReservoirReplay`` the way a launcher would: with ``RANK`` set in the environment."""
    previous = os.environ.get("RANK")
    os.environ["RANK"] = str(rank)
    try:
        return replay(**kwargs)
    finally:
        if previous is None:
            del os.environ["RANK"]
        else:
            os.environ["RANK"] = previous


def build_world(world_size: int, batches: list[dict], *, num_generations: int = 2, launcher_env: bool = True,
                **replay_kwargs):
    """One replay, one trainer and one accelerator per rank, each trainer fed its shards of ``batches``."""
    world = World(world_size)
    make = (lambda r: replay_on_rank(r, **replay_kwargs)) if launcher_env else (lambda r: replay(**replay_kwargs))
    replays = [make(r) for r in range(world_size)]
    shards = [[shard(b, r, world_size) for b in batches] for r in range(world_size)]
    trainers = [FakeTrainer(replays[r], list(shards[r]), num_processes=world_size) for r in range(world_size)]
    for t, a in zip(trainers, world.accelerators()):
        t.accelerator = a
        t.num_generations = num_generations
    return replays, trainers, shards, world


def distributed_run(world_size: int, batches: list[dict], steps: list[int], **kwargs):
    """Drive ``len(steps)`` generation steps on ``world_size`` fake ranks.

    Returns ``(replays, outputs, trainers, shards, world)``; ``outputs[i]``
    is the list of what each rank's hook returned at step ``steps[i]``.
    """
    replays, trainers, shards, world = build_world(world_size, batches, **kwargs)
    outputs = [run_ranks([lambda t=t, s=step: t.generate(s) for t in trainers]) for step in steps]
    return replays, outputs, trainers, shards, world


def single_run(batches: list[dict], steps: list[int], *, num_generations: int = 2, **replay_kwargs):
    r = replay(**replay_kwargs)
    trainer = FakeTrainer(r, [dict(b) for b in batches])
    trainer.num_generations = num_generations
    return r, [trainer.generate(s) for s in steps]


def rows_of(output: dict) -> list[dict]:
    """Per-row view of a batch for comparison independent of padding width."""
    n = output["advantages"].size(0)
    out = []
    for r in range(n):
        pm, cm = output["prompt_mask"][r].bool(), output["completion_mask"][r].bool()
        row = {
            "prompt": output["prompt_ids"][r][pm].tolist(),
            "completion": output["completion_ids"][r][cm].tolist(),
            "advantage": float(output["advantages"][r]),
        }
        if "old_per_token_logps" in output:
            row["logps"] = output["old_per_token_logps"][r][cm].tolist()
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Parity with the single-process adapter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("world_size", [2, 4])
def test_replayed_slices_match_the_single_process_adapter_on_the_concatenated_batch(world_size):
    batches = [live_batch(), global_batch()]
    single, expected = single_run(batches, [1, 2], seed=7)
    replays, got, _, _, world = distributed_run(world_size, batches, [1, 2], seed=7)

    assert torch.equal(torch.cat([o["advantages"] for o in got[1]]), expected[1]["advantages"])
    assert [row for o in got[1] for row in rows_of(o)] == rows_of(expected[1])
    owner = replays[0]
    assert owner.stats == single.stats
    assert owner.last_replay.indices == single.last_replay.indices
    assert owner.last_telemetry == single.last_telemetry
    assert owner.buffer.attestation_log is None
    assert world.collectives == 1 + 2 * 2  # the attach check, then one gather and one broadcast per step


@pytest.mark.parametrize("world_size", [2, 4])
def test_a_dead_group_split_across_ranks_is_replaced_whole(world_size):
    # G=4 over 8 rows: with four ranks (two rows each) both groups straddle two ranks,
    # and the dead group is rows 4-7, so its replacements land on two different ranks.
    batches = [wide_live_batch(), wide_group_batch()]
    single, expected = single_run(batches, [1, 2], num_generations=4)
    replays, got, _, _, _ = distributed_run(world_size, batches, [1, 2], num_generations=4)

    owner = replays[0]
    assert owner.stats["dead_groups"] == 1 and owner.stats["replaced_rows"] == 4
    assert owner.stats["ingested_groups"] == single.stats["ingested_groups"] == 2 + 1
    merged = [row for o in got[1] for row in rows_of(o)]
    assert merged == rows_of(expected[1])
    assert all(row["advantage"] != 0.0 for row in merged[4:8])
    assert [o["advantages"].size(0) for o in got[1]] == [8 // world_size] * world_size


def test_every_rank_gets_the_same_global_loss_normaliser():
    _, got, _, _, _ = distributed_run(2, [live_batch(), global_batch()], [1, 2])
    values = [int(o["num_items_in_batch"]) for o in got[1]]
    total = sum(int(o["completion_mask"].sum()) for o in got[1])
    assert values == [total, total]


def test_a_step_without_replay_returns_each_rank_the_dict_it_was_given():
    _, got, _, shards, _ = distributed_run(2, [live_batch()], [1])
    for rank, out in enumerate(got[0]):
        assert out is shards[rank][0]
        assert "old_per_token_logps" not in out


def test_all_declined_matches_the_single_process_path_including_the_witness_digest():
    from reservoir.attest import AttestationLog
    from tests.test_trl_telemetry import LOGP, MetricTrainer

    def witness(r):
        return next(rec for rec in r.buffer.attestation_log.records if rec["op"] == "batch")

    world = World(2)
    replays = [replay_on_rank(r, max_log_ratio=0.1, attest=AttestationLog()) for r in range(2)]
    shards = [[shard(live_batch(), r, 2), shard(dead_batch(), r, 2)] for r in range(2)]
    trainers = [MetricTrainer(replays[r], list(shards[r]), num_processes=2) for r in range(2)]
    for t, a in zip(trainers, world.accelerators()):
        t.accelerator = a
    run_ranks([lambda t=t: t.generate(1) for t in trainers])
    for t in trainers:
        t.current_logp = LOGP - 1.0          # every stored row has drifted by one nat per token
    got = run_ranks([lambda t=t: t.generate(2) for t in trainers])
    assert replays[0].stats["declined_rows"] == 4 and replays[0].stats["replaced_rows"] == 0

    single = replay(max_log_ratio=0.1, attest=AttestationLog())
    trainer = MetricTrainer(single, [live_batch(), dead_batch()])
    trainer.generate(1)
    trainer.current_logp = LOGP - 1.0
    expected = trainer.generate(2)
    assert single.stats["declined_rows"] == 4 and single.stats["replaced_rows"] == 0

    # Both paths hand back the batch with behavior logprobs attached and the dead rows still dead,
    # and the witness digests the same tensors.
    assert "old_per_token_logps" in expected and all("old_per_token_logps" in o for o in got)
    assert [row for o in got for row in rows_of(o)] == rows_of(expected)
    assert witness(replays[0])["tensor_digest"] == witness(single)["tensor_digest"]
    assert witness(single)["replaced"] == [] and len(witness(single)["declined"]) == 4


def test_rank_widths_differ_and_the_result_is_padded_to_the_widest_input():
    _, got, _, shards, _ = distributed_run(2, [live_batch(), global_batch()], [1, 2])
    inputs = [s[1] for s in shards]
    assert len({s["completion_ids"].size(1) for s in inputs}) == 2
    assert len({s["prompt_ids"].size(1) for s in inputs}) == 2
    assert {o["completion_ids"].size(1) for o in got[1]} == {max(s["completion_ids"].size(1) for s in inputs)}
    assert {o["prompt_ids"].size(1) for o in got[1]} == {max(s["prompt_ids"].size(1) for s in inputs)}


# ---------------------------------------------------------------------------
# Ownership: one buffer, one log writer
# ---------------------------------------------------------------------------

def test_only_rank_zero_opens_the_log_and_other_ranks_have_no_buffer(tmp_path):
    attest = tmp_path / "attest.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    replays, _, _, _, _ = distributed_run(2, [live_batch(), global_batch()], [1, 2],
                                          attest=attest, manifest=manifest)
    owner, other = replays
    assert owner.is_owner and not other.is_owner
    assert other.stats["hook_calls"] == 2 and other.stats["replaced_rows"] == 0
    with pytest.raises(RuntimeError, match="rank 1"):
        other.buffer
    other.close()  # nothing to close; no error
    owner.close()
    result = verify_json_lines(attest.read_text(), manifest=manifest.read_text())
    assert len(result.records) > 0 and result.content.manifest_matched == 4 + 6
    inserts = [json.loads(line) for line in attest.read_text().splitlines() if '"insert"' in line]
    assert len(inserts) == 4 + 6


def attach_all(replays, world):
    accels = world.accelerators()
    run_ranks([lambda r=r: replays[r].attach(accels[r]) for r in range(world.size)])


def test_file_backed_state_waits_for_the_rank_and_in_memory_state_is_dropped_on_a_non_owner(tmp_path):
    attest = tmp_path / "attest.jsonl"
    world = World(2)
    replays = [replay(attest=attest) for _ in range(2)]   # no launcher env, one shared file target
    assert not attest.exists()                              # nobody opened it at construction
    attach_all(replays, world)
    owner, other = replays
    assert owner.is_owner and other.rank == 1 and not other.is_owner
    assert attest.exists() and owner._buffer is not None    # rank 0 opened it at attach
    assert other._buffer is None
    with pytest.raises(RuntimeError, match="rank 1"):
        other.buffer
    owner.close()

    world = World(2)
    replays = [replay() for _ in range(2)]                  # in-memory: built eagerly on both
    assert all(r._buffer is not None for r in replays)
    attach_all(replays, world)
    assert replays[0]._buffer is not None and replays[1]._buffer is None


def test_attach_refuses_on_every_rank_when_a_non_owner_already_opened_a_file(tmp_path):
    world = World(2)
    replays = [replay(attest=tmp_path / "attest.jsonl") for _ in range(2)]   # no launcher env
    replays[1].buffer                                                        # rank 1 touched the file first
    accels = world.accelerators()

    def attach(rank):
        with pytest.raises(RuntimeError, match=r"rank\(s\) \[1\]"):
            replays[rank].attach(accels[rank])
    run_ranks([lambda: attach(0), lambda: attach(1)])
    assert replays[0]._buffer is None                                       # rank 0 never opened it
    assert replays[1]._buffer is None                                       # the offender's is closed
    assert all(r._comm is None and r.rank is None for r in replays)         # nothing was attached


def test_env_rank_semantics(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert env_rank() is None
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert env_rank() is None                   # local rank 0 on some node: undecided until attach
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert env_rank() == 3                      # a non-zero local rank is never rank 0
    monkeypatch.setenv("RANK", "0")
    assert env_rank() == 0                      # RANK wins
    monkeypatch.setenv("RANK", "x")
    with pytest.raises(RuntimeError, match="RANK"):
        env_rank()


def test_env_rank_blocks_buffer_construction_before_attach(monkeypatch, tmp_path):
    monkeypatch.setenv("RANK", "1")
    r = replay(attest=tmp_path / "attest.jsonl")
    with pytest.raises(RuntimeError, match="RANK"):
        r.buffer
    assert not (tmp_path / "attest.jsonl").exists()
    monkeypatch.setenv("RANK", "0")
    assert replay().buffer is not None


def test_checkpoint_binding_is_a_no_op_off_the_owner_and_works_on_it(tmp_path):
    world = World(2)
    replays = [replay(directory=tmp_path / "buf", attest=tmp_path / "attest.jsonl") for _ in range(2)]
    attach_all(replays, world)
    other = replays[1]
    bind_checkpoint(other, 3, tmp_path)
    resume_from_checkpoint(other, 3)
    assert other._buffer is None and not other.is_owner
    assert replays[0].buffer.checkpoints() == []            # rank 0's buffer saw nothing from rank 1
    replays[0].close()

    replays, _, _, _, _ = distributed_run(
        2, [live_batch(), global_batch()], [1, 2], directory=tmp_path / "owner-buf", attest=tmp_path / "owner.jsonl",
    )
    owner = replays[0]
    bind_checkpoint(owner, 2)
    assert owner.buffer.checkpoints() == ["step-2"]
    owner.close()


# ---------------------------------------------------------------------------
# Failure handling: no rank is left waiting
# ---------------------------------------------------------------------------

def test_an_owner_failure_is_reported_on_every_rank():
    replays, trainers, _, _ = build_world(2, [live_batch(), global_batch()])
    run_ranks([lambda t=t: t.generate(1) for t in trainers])
    replays[0].buffer.advance(5)  # rank 0's buffer is now ahead of global_step 2

    def owner():
        with pytest.raises(ValueError, match="global_step 2 is below"):
            trainers[0].generate(2)

    def other():
        with pytest.raises(RuntimeError, match="replay failed on rank 0: ValueError"):
            trainers[1].generate(2)
    run_ranks([owner, other])


def test_a_failure_on_a_non_owner_before_the_gather_is_reported_on_every_rank():
    bad = live_batch()
    bad["tool_mask"] = torch.ones_like(bad["completion_mask"])
    replays, trainers, _, _ = build_world(2, [live_batch()])
    trainers[1]._outputs = [shard(bad, 1, 2)]

    def owner():
        with pytest.raises(RuntimeError, match="before the gather on rank 1: NotImplementedError"):
            trainers[0].generate(1)

    def other():
        with pytest.raises(NotImplementedError, match="tool_mask"):
            trainers[1].generate(1)
    run_ranks([owner, other])
    assert replays[0].stats["hook_calls"] == 0 and replays[1].stats["hook_calls"] == 0


def test_ranks_disagreeing_on_the_step_raise_everywhere():
    _, trainers, _, _ = build_world(2, [live_batch()])

    def run(t, step):
        with pytest.raises(RuntimeError, match="global_step"):
            t.generate(step)
    run_ranks([lambda: run(trainers[0], 1), lambda: run(trainers[1], 2)])


# ---------------------------------------------------------------------------
# Shard assembly helpers and the communicator seam
# ---------------------------------------------------------------------------

def test_concat_and_slice_round_trip_with_unequal_widths():
    full = global_batch()
    full["old_per_token_logps"] = torch.full_like(full["completion_ids"], -0.5, dtype=torch.float32)
    shards = [shard(full, r, 4) for r in range(4)]
    joined = dist.concat_shards(shards, PAD)
    assert joined["prompt_ids"].size(0) == 8
    assert rows_of(joined) == rows_of(full)
    sizes = dist.shard_sizes(shards)
    for r in range(4):
        assert rows_of(dist.slice_shard(joined, r, sizes)) == rows_of(shards[r])
    assert joined["num_items_in_batch"] is shards[0]["num_items_in_batch"]


def test_concat_refuses_what_it_cannot_align():
    shards = [shard(global_batch(), r, 2) for r in range(2)]
    shards[0]["extra"] = torch.zeros(4, 3)
    shards[1]["extra"] = torch.zeros(4, 5)
    with pytest.raises(ValueError, match="extra"):
        dist.concat_shards(shards, PAD)
    shards = [shard(global_batch(), r, 2) for r in range(2)]
    shards[0]["texts"] = ["a"] * 4
    shards[1]["texts"] = ["b"] * 4
    with pytest.raises(ValueError, match="texts"):
        dist.concat_shards(shards, PAD)


def test_tensor_digest_ignores_logprob_padding_but_not_masked_values():
    from reservoir.integrations.trl import tensor_digest

    batch = global_batch()
    batch["old_per_token_logps"] = torch.full_like(batch["completion_ids"], -0.5, dtype=torch.float32)
    reference = tensor_digest(batch)
    garbage = dict(batch, old_per_token_logps=batch["old_per_token_logps"].clone())
    garbage["old_per_token_logps"][batch["completion_mask"] == 0] = 7.0
    assert tensor_digest(garbage) == reference
    changed = dict(batch, old_per_token_logps=batch["old_per_token_logps"].clone())
    changed["old_per_token_logps"][0, 0] = -0.25
    assert tensor_digest(changed) != reference


def test_communicator_for_wraps_a_real_accelerator_and_passes_a_fake_through():
    fake = World(2).accelerators()[1]
    assert dist.communicator_for(fake) is fake
    plain = type("Accel", (), {"num_processes": 2, "process_index": 0})()
    comm = dist.communicator_for(plain)
    assert isinstance(comm, dist.AcceleratorCommunicator)
    assert comm.num_processes == 2 and comm.process_index == 0
    single = type("Accel", (), {"num_processes": 1})()
    assert dist.communicator_for(single).process_index == 0
    with pytest.raises(RuntimeError, match="process_index"):
        dist.communicator_for(type("Accel", (), {"num_processes": 2})())


def test_a_gather_that_returns_too_few_shards_is_refused():
    class HalfWorld:
        num_processes = 2
        process_index = 0

        def gather_object(self, obj):
            return [obj]          # a process group that was never initialised gathers only itself

        def broadcast_object(self, obj):
            return obj

    r = replay()
    trainer = FakeTrainer(r, [live_batch()], num_processes=2)
    trainer.accelerator = HalfWorld()
    with pytest.raises(RuntimeError, match="gathered 1 shards"):
        trainer.generate(1)


def test_single_process_accelerator_without_distributed_attributes_still_works():
    r = replay()
    trainer = FakeTrainer(r, [live_batch(), mixed_batch()])
    trainer.generate(1)
    out = trainer.generate(2)
    assert r.stats["replaced_rows"] == 2 and out["advantages"].size(0) == 4
