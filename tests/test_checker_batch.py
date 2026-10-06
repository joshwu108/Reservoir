"""Tests for batch witnesses: the record that proves a training-batch row holds the example its draw selected.

``RolloutBuffer.witness_batch`` writes a ``batch`` record after a sampled
batch has been placed into training-batch rows. The checker requires that
the record names the latest sample, that every placed draw exists and is
used once, that every row is inside the batch and used once, and that each
row's content digest equals the digest the draw resolved to; the transcript
then explains any row of any witnessed step.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from reservoir.attest import AttestationLog, _digest_record
from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer
from reservoir_checker.transcript import build_transcript, explain_row, main as transcript_main
from reservoir_checker.verify import CheckerError, verify_chain

DIGEST = "ab" * 32


def rollouts(rewards: list[float]) -> list[Rollout]:
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in rewards]


def rechain(records: list[dict], start: int = 0) -> list[dict]:
    out = copy.deepcopy(records)
    prev = out[start - 1]["digest"] if start > 0 else "genesis"
    for rec in out[start:]:
        rec["prev_digest"] = prev
        rec["digest"] = _digest_record(rec)
        prev = rec["digest"]
    return out


def witnessed_run(steps: int = 4, batch_rows: int = 6) -> tuple[list[dict], RolloutBuffer]:
    """Each step: store a group, sample two rollouts, witness them into rows 4 and 5."""
    buf = RolloutBuffer(capacity=8, half_life=2, max_policy_age=4, seed=1, attest=AttestationLog())
    for step in range(steps):
        buf.add_group(f"g{step}", step, rollouts([1.0, 0.0, 0.5]), source="s")
        batch = buf.sample(2, current_version=step)
        buf.witness_batch(batch, step=step, batch_rows=batch_rows, rows=[4, 5], tensor_digest=DIGEST)
    return [dict(r) for r in buf.attestation_log.records], buf


def batch_indices(records: list[dict]) -> list[int]:
    return [i for i, r in enumerate(records) if r["op"] == "batch"]


def witness_with_distinct_examples(records: list[dict]) -> int:
    """Index of a batch record whose two draws selected different examples (swapping them is detectable)."""
    for i in batch_indices(records):
        digests = [e["content_digest"] for e in records[i]["replaced"]]
        if len(set(digests)) == len(digests):
            return i
    raise AssertionError("every witness drew the same example twice; change the seed")


class TestBufferSide:
    def test_batch_carries_op_counter_and_digests(self) -> None:
        buf = RolloutBuffer(capacity=4, attest=AttestationLog())
        buf.add_group("p", 0, rollouts([1.0, 0.0]), source="s")
        batch = buf.sample(3)
        assert batch.op_counter == 1
        assert len(batch.content_digests) == 3
        for k, idx in enumerate(batch.indices):
            rollout, group = buf.entry(idx)
            assert batch.content_digests[k] == group.content_digests[group.rollouts.index(rollout)]

    def test_witness_record_shape(self) -> None:
        records, _ = witnessed_run(steps=1)
        rec = records[batch_indices(records)[0]]
        assert rec["step"] == "0" and rec["sample_op_counter"] == 1 and rec["batch_rows"] == 6
        assert [e["row"] for e in rec["replaced"]] == [4, 5]
        assert [e["draw"] for e in rec["replaced"]] == [0, 1]
        assert rec["tensor_digest"] == DIGEST
        assert records[0]["format"] == "3"

    def test_only_the_latest_batch_once(self) -> None:
        buf = RolloutBuffer(capacity=4, attest=AttestationLog())
        buf.add_group("p", 0, rollouts([1.0, 0.0]), source="s")
        first = buf.sample(1)
        second = buf.sample(1)
        with pytest.raises(ValueError, match="latest sample"):
            buf.witness_batch(first, step=0, batch_rows=2, rows=[0], tensor_digest=DIGEST)
        buf.witness_batch(second, step=0, batch_rows=2, rows=[0], tensor_digest=DIGEST)
        with pytest.raises(ValueError, match="already witnessed"):
            buf.witness_batch(second, step=0, batch_rows=2, rows=[1], tensor_digest=DIGEST)
        with pytest.raises(ValueError, match="rows for"):
            buf.witness_batch(buf.sample(2), step=0, batch_rows=2, rows=[0], tensor_digest=DIGEST)

    def test_witness_is_a_no_op_without_attestation_but_still_validates(self) -> None:
        buf = RolloutBuffer(capacity=4)
        buf.add_group("p", 0, rollouts([1.0]))
        batch = buf.sample(1)
        with pytest.raises(ValueError, match="rows for"):
            buf.witness_batch(batch, step=0, batch_rows=1, rows=[0, 0], tensor_digest=DIGEST)
        buf.witness_batch(batch, step=0, batch_rows=1, rows=[0], tensor_digest=DIGEST)
        with pytest.raises(ValueError, match="already witnessed"):
            buf.witness_batch(batch, step=0, batch_rows=1, rows=[0], tensor_digest=DIGEST)

    def test_malformed_witness_rejected_before_writing(self) -> None:
        buf = RolloutBuffer(capacity=4, attest=AttestationLog())
        buf.add_group("p", 0, rollouts([1.0, 0.0]), source="s")
        batch = buf.sample(2)
        n = len(buf.attestation_log.records)
        with pytest.raises(ValueError, match="distinct"):
            buf.witness_batch(batch, step=0, batch_rows=4, rows=[1, 1], tensor_digest=DIGEST)
        with pytest.raises(ValueError, match="outside"):
            buf.witness_batch(batch, step=0, batch_rows=2, rows=[0, 5], tensor_digest=DIGEST)
        with pytest.raises(ValueError, match="tensor_digest"):
            buf.witness_batch(batch, step=0, batch_rows=4, rows=[0, 1], tensor_digest="zz")
        assert len(buf.attestation_log.records) == n

    def test_durable_witness_survives_reopen(self, tmp_path: Path) -> None:
        kw = dict(capacity=4, half_life=2, max_policy_age=4, seed=0)
        buf = DurableRolloutBuffer(tmp_path / "buf", attest=tmp_path / "a.jsonl", **kw)
        buf.add_group("p", 0, rollouts([1.0, 0.0]), source="s")
        batch = buf.sample(1)
        buf.witness_batch(batch, step=0, batch_rows=2, rows=[1], tensor_digest=DIGEST)
        records = buf.attestation_log.records
        buf.close()
        again = DurableRolloutBuffer(tmp_path / "buf", attest=tmp_path / "a.jsonl", **kw)
        assert again.attestation_log.records == records
        with pytest.raises(ValueError, match="already witnessed"):
            again.witness_batch(batch, step=0, batch_rows=2, rows=[0], tensor_digest=DIGEST)
        again.close()


class TestCheckerAccepts:
    def test_witnessed_run_verifies(self) -> None:
        records, _ = witnessed_run()
        result = verify_chain(records)
        assert len(result.content.witnesses) == 4
        assert len(result.content.witnessed_rows) == 8
        assert all(w.row in (4, 5) for w in result.content.witnessed_rows)

    def test_witnessed_rows_resolve_to_the_draws(self) -> None:
        records, _ = witnessed_run()
        result = verify_chain(records)
        by_op = {}
        for s in result.content.samples:
            by_op.setdefault(s.op_counter, {})[s.position_in_batch] = s.content_digest
        for w in result.content.witnessed_rows:
            assert w.content_digest == by_op[w.sample_op_counter][w.draw]

    def test_log_without_witnesses_still_verifies(self) -> None:
        buf = RolloutBuffer(capacity=4, attest=AttestationLog())
        buf.add_group("p", 0, rollouts([1.0]), source="s")
        buf.sample(1)
        assert verify_chain(buf.attestation_log.records).content.witnesses == []

    def test_committed_logs_still_verify(self) -> None:
        results = Path(__file__).parents[1] / "benchmarks" / "modal" / "results"
        logs = sorted(results.glob("trl_replay_*.attest.jsonl")) + sorted(results.glob("repro_*/*/attest.jsonl"))
        assert logs
        for log in logs:
            verify_chain([json.loads(l) for l in log.read_text().splitlines() if l])


class TestCheckerRejects:
    def _edit(self, records, idx, edit):
        m = copy.deepcopy(records)
        edit(m[idx])
        return rechain(m, idx)

    def test_wrong_draw_for_a_row(self) -> None:
        records, _ = witnessed_run()
        i = witness_with_distinct_examples(records)
        # swap which draw each row claims but keep the digests: row 4 now claims draw 1's slot with draw 0's digest
        def swap(r):
            r["replaced"][0]["draw"], r["replaced"][1]["draw"] = 1, 0
        with pytest.raises(CheckerError, match="claims example"):
            verify_chain(self._edit(records, i, swap))

    def test_digest_changed(self) -> None:
        records, _ = witnessed_run()
        i = batch_indices(records)[0]
        with pytest.raises(CheckerError, match="claims example"):
            verify_chain(self._edit(records, i, lambda r: r["replaced"][0].update(content_digest="cd" * 32)))

    def test_draw_used_twice(self) -> None:
        records, _ = witnessed_run()
        i = batch_indices(records)[0]
        def dup(r):
            r["replaced"][1]["draw"] = 0
            r["replaced"][1]["content_digest"] = r["replaced"][0]["content_digest"]
        with pytest.raises(CheckerError, match="placed twice"):
            verify_chain(self._edit(records, i, dup))

    def test_row_replaced_twice(self) -> None:
        records, _ = witnessed_run()
        i = batch_indices(records)[0]
        with pytest.raises(CheckerError, match="replaced twice"):
            verify_chain(self._edit(records, i, lambda r: r["replaced"][1].update(row=4)))

    def test_row_outside_batch(self) -> None:
        records, _ = witnessed_run()
        i = batch_indices(records)[0]
        with pytest.raises(CheckerError, match="outside a batch"):
            verify_chain(self._edit(records, i, lambda r: r["replaced"][1].update(row=6)))

    def test_missing_draw(self) -> None:
        records, _ = witnessed_run()
        i = batch_indices(records)[0]
        with pytest.raises(CheckerError, match="places 1 rows and declines 0 draws but sample"):
            verify_chain(self._edit(records, i, lambda r: r["replaced"].pop()))

    def test_nonexistent_draw_index(self) -> None:
        records, _ = witnessed_run()
        i = batch_indices(records)[0]
        with pytest.raises(CheckerError, match="does not exist"):
            verify_chain(self._edit(records, i, lambda r: r["replaced"][1].update(draw=5)))

    def test_wrong_sample_op_counter(self) -> None:
        records, _ = witnessed_run()
        i = batch_indices(records)[1]
        with pytest.raises(CheckerError, match="latest sample"):
            verify_chain(self._edit(records, i, lambda r: r.update(sample_op_counter=1)))
        with pytest.raises(CheckerError, match="latest sample"):
            verify_chain(self._edit(records, i, lambda r: r.update(sample_op_counter=99)))

    def test_duplicate_witness(self) -> None:
        records, _ = witnessed_run()
        i = batch_indices(records)[0]
        dup = copy.deepcopy(records)
        dup.insert(i + 1, copy.deepcopy(records[i]))
        with pytest.raises(CheckerError, match="already has a batch witness"):
            verify_chain(rechain(dup, i + 1))

    def test_bad_tensor_digest_and_bad_step(self) -> None:
        records, _ = witnessed_run()
        i = batch_indices(records)[0]
        with pytest.raises(CheckerError, match="tensor_digest"):
            verify_chain(self._edit(records, i, lambda r: r.update(tensor_digest="xyz")))
        with pytest.raises(CheckerError, match="step"):
            verify_chain(self._edit(records, i, lambda r: r.update(step="-1")))

    def test_witness_on_a_log_without_content_digests(self) -> None:
        records, _ = witnessed_run(steps=1)
        for r in records:
            r.pop("content_digest", None)
            r.pop("source", None)
        with pytest.raises(CheckerError, match="needs content digests"):
            verify_chain(rechain(records))

    def test_unknown_format_rejected(self) -> None:
        records, _ = witnessed_run(steps=1)
        records[0]["format"] = "9"
        with pytest.raises(CheckerError, match="unknown log format"):
            verify_chain(rechain(records))

    def test_witness_in_a_format_1_log_rejected(self) -> None:
        records, _ = witnessed_run(steps=1)
        records[0]["format"] = "1"
        with pytest.raises(CheckerError, match="format-2"):
            verify_chain(rechain(records))

    def test_two_samples_with_the_same_op_counter_rejected(self) -> None:
        records, _ = witnessed_run(steps=2)
        samples = [i for i, r in enumerate(records) if r["op"] == "sample"]
        dup = copy.deepcopy(records)
        dup[samples[1]]["op_counter"] = dup[samples[0]]["op_counter"]
        with pytest.raises(CheckerError, match="does not increase"):
            verify_chain(rechain(dup, samples[1]))

    def test_witness_for_slots_evicted_afterwards_is_accepted(self) -> None:
        # The witness describes the batch at sample time; later evictions of its slots are fine.
        buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=1, seed=0, attest=AttestationLog())
        buf.add_group("g0", 0, rollouts([1.0, 0.0]), source="s")
        batch = buf.sample(2)
        buf.witness_batch(batch, step=0, batch_rows=4, rows=[2, 3], tensor_digest=DIGEST)
        buf.add_group("g3", 3, rollouts([1.0]), source="s")       # expires g0's slots
        result = verify_chain(buf.attestation_log.records)
        assert len(result.content.witnessed_rows) == 2

    def test_declined_draws_must_partition_the_sample(self) -> None:
        buf = RolloutBuffer(capacity=8, attest=AttestationLog())
        buf.add_group("g0", 0, rollouts([1.0, 0.0, 0.5]), source="s")
        batch = buf.sample(3)
        buf.witness_batch(batch, step=0, batch_rows=4, rows=[1, 2], tensor_digest=DIGEST, declined=[1])
        records = [dict(r) for r in buf.attestation_log.records]
        result = verify_chain(records)
        assert result.content.witnesses[0]["declined"] == [1] and result.content.witnesses[0]["replaced"] == 2
        i = batch_indices(records)[0]
        with pytest.raises(CheckerError, match="placed twice or both"):
            verify_chain(self._edit(records, i, lambda r: r["replaced"][0].update(draw=1)))
        with pytest.raises(CheckerError, match="distinct positions"):
            verify_chain(self._edit(records, i, lambda r: r.update(declined=[1, 1])))

    def test_witness_while_evictions_pending_rejected(self) -> None:
        # A witness may not sit between an advance_version and the stale evicts it requires.
        buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=1, seed=0, attest=AttestationLog())
        buf.add_group("g0", 0, rollouts([1.0, 0.0]), source="s")
        batch = buf.sample(1)
        buf.witness_batch(batch, step=0, batch_rows=2, rows=[0], tensor_digest=DIGEST)
        buf.add_group("g3", 3, rollouts([1.0]), source="s")      # advance + stale evicts
        records = [dict(r) for r in buf.attestation_log.records]
        adv = next(i for i, r in enumerate(records) if r["op"] == "advance_version")
        witness = records[batch_indices(records)[0]]
        moved = copy.deepcopy(records)
        moved.insert(adv + 1, copy.deepcopy(witness))
        with pytest.raises(CheckerError, match="have not been evicted|already has a batch witness"):
            verify_chain(rechain(moved, adv + 1))


class TestExplain:
    def test_explain_replayed_and_fresh_rows(self) -> None:
        records, _ = witnessed_run()
        verified = verify_chain(records)
        replayed = explain_row(verified, step=2, row=4)
        assert replayed["kind"] == "replayed" and replayed["draw"] == 0 and replayed["source"] == "s"
        assert 0 < replayed["is_weight"] <= 1 and replayed["stored_at_records"]
        assert records[replayed["stored_at_records"][0]]["content_digest"] == replayed["content_digest"]
        fresh = explain_row(verified, step=2, row=1)
        assert fresh["kind"] == "fresh"
        # A step with two witnesses is ambiguous without a sample selector.
        buf = RolloutBuffer(capacity=8, attest=AttestationLog())
        buf.add_group("g", 0, rollouts([1.0, 0.0]), source="s")
        for _ in range(2):
            b = buf.sample(1)
            buf.witness_batch(b, step=0, batch_rows=2, rows=[1], tensor_digest=DIGEST)
        twice = verify_chain(buf.attestation_log.records)
        with pytest.raises(CheckerError, match="2 batch witnesses"):
            explain_row(twice, step=0, row=1)
        assert explain_row(twice, step=0, row=1, sample_op_counter=2)["draw"] == 0
        with pytest.raises(CheckerError, match="outside the batch"):
            explain_row(verified, step=2, row=6)
        with pytest.raises(CheckerError, match="no batch witness"):
            explain_row(verified, step=9, row=0)

    def test_transcript_cli_explain(self, tmp_path: Path) -> None:
        records, _ = witnessed_run()
        log = tmp_path / "a.jsonl"
        log.write_text("\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in records) + "\n")
        out = tmp_path / "report.json"
        assert transcript_main([str(log), "--explain", "1", "5", "--json", str(out)]) == 0
        report = json.loads(out.read_text())
        assert report["batch_witnesses"] == 4 and report["explain"]["kind"] == "replayed"
        assert transcript_main([str(log), "--explain", "7", "0"]) == 1
