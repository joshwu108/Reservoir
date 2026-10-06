"""Checker tests for quarantine records and reward provenance in the manifest.

A quarantine eviction is an ``evict`` record with reason ``"quarantine"``
that must carry the predicate text and a note; nothing else may carry
them, and a log written before the format that introduced them cannot
contain one. A manifest line may carry ``rewards``, the per-reward-function
values of the example, numeric only and outside the content digest; the
checker validates their shape and the transcript shows them.
"""

from __future__ import annotations

import copy
import json

import pytest

from reservoir.attest import AttestationLog, _digest_record
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer
from reservoir.rollout_manifest import ManifestWriter
from reservoir_checker.content import ContentState, load_manifest
from reservoir_checker.verify import CheckerError, verify_chain


def rollouts(values, **meta):
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r, metadata=meta or None) for r in values]


def run(with_rewards: bool = True) -> tuple[list[dict], list[dict]]:
    """A log with inserts, a sample, a witness and a quarantine; plus its manifest."""
    manifest = ManifestWriter()
    buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=4, seed=2, attest=AttestationLog(), manifest=manifest)
    meta = {"rewards": {"verifier": 1.0, "judge": 0.25}} if with_rewards else {}
    buf.add_group("clean", 0, rollouts([1.0, 0.0], **meta), source="a")
    buf.add_group("hacked", 1, rollouts([1.0, 0.5], **meta), source="judge-v3")
    batch = buf.sample(2, current_version=1)
    buf.witness_batch(batch, step=1, batch_rows=4, rows=[2, 3], tensor_digest="ab" * 32)
    buf.quarantine(lambda r, g: g.source == "judge-v3", "reward hack", predicate_text="source == judge-v3")
    return [dict(r) for r in buf.attestation_log.records], manifest.records


def rechain(records: list[dict], start: int) -> list[dict]:
    out = copy.deepcopy(records)
    prev = out[start - 1]["digest"] if start > 0 else "genesis"
    for rec in out[start:]:
        rec["prev_digest"] = prev
        rec["digest"] = _digest_record(rec)
        prev = rec["digest"]
    return out


def edited(records: list[dict], idx: int, edit) -> list[dict]:
    m = copy.deepcopy(records)
    edit(m[idx])
    return rechain(m, idx)


def quarantine_index(records: list[dict]) -> int:
    return next(i for i, r in enumerate(records) if r.get("reason") == "quarantine")


class TestQuarantineRecords:
    def test_accepted_and_resolved_to_the_example(self) -> None:
        records, manifest = run()
        verified = verify_chain(records, manifest=manifest)
        quarantined = verified.content.quarantines
        assert len(quarantined) == 2
        hacked = {line["content_digest"] for line in manifest if line["prompt_id"] == "hacked"}
        assert {q.content_digest for q in quarantined} == hacked
        assert all(q.predicate == "source == judge-v3" and q.note == "reward hack" for q in quarantined)
        assert all(q.source == "judge-v3" for q in quarantined)

    @pytest.mark.parametrize("edit, message", [
        (lambda r: r.pop("predicate"), "predicate"),
        (lambda r: r.pop("note"), "note"),
        (lambda r: r.update(predicate=""), "predicate"),
        (lambda r: r.update(predicate=5), "predicate"),
        (lambda r: r.update(predicate="x" * 1025), "predicate"),
        (lambda r: r.update(note="a\nb"), "note"),
        (lambda r: r.update(note="x" * 257), "note"),
        (lambda r: r.update(reason="quarantined"), "reason"),
    ])
    def test_malformed_quarantine_record_is_rejected(self, edit, message) -> None:
        records, _ = run()
        with pytest.raises(CheckerError, match=message):
            verify_chain(edited(records, quarantine_index(records), edit))

    @pytest.mark.parametrize("reason", ["stale", "capacity", "drift", "explicit"])
    def test_texts_on_another_evict_reason_are_rejected(self, reason) -> None:
        records, _ = run()
        i = quarantine_index(records)
        with pytest.raises(CheckerError, match="quarantine"):
            verify_chain(edited(records, i, lambda r: r.update(reason=reason)))

    def test_texts_on_an_insert_are_rejected(self) -> None:
        records, _ = run()
        i = next(k for k, r in enumerate(records) if r["op"] == "insert")
        with pytest.raises(CheckerError, match="quarantine"):
            verify_chain(edited(records, i, lambda r: r.update(predicate="p", note="n")))

    def test_quarantine_in_a_format_2_log_is_rejected(self) -> None:
        records, _ = run()
        with pytest.raises(CheckerError, match="format-3"):
            verify_chain(edited(records, 0, lambda r: r.update(format="2")))

    def test_format_2_logs_still_verify_without_quarantine(self) -> None:
        buf = RolloutBuffer(capacity=4, attest=AttestationLog())
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        buf.sample(1)
        records = [dict(r) for r in buf.attestation_log.records]
        verify_chain(edited(records, 0, lambda r: r.update(format="2")))
        verify_chain(edited(records, 0, lambda r: r.update(format="1")))

    def test_unknown_format_names_the_range(self) -> None:
        records, _ = run()
        with pytest.raises(CheckerError, match="formats 1 to 3"):
            verify_chain(edited(records, 0, lambda r: r.update(format="4")))

    def test_quarantine_while_stale_evictions_are_pending_is_rejected(self) -> None:
        buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=1, seed=2, attest=AttestationLog())
        buf.add_group("old", 0, rollouts([1.0, 0.0]), source="a")
        buf.add_group("new", 3, rollouts([1.0, 0.5]), source="a")      # advance evicts "old" as stale
        buf.quarantine(lambda r, g: g.prompt_id == "new", "hack", predicate_text="new")
        records = [dict(r) for r in buf.attestation_log.records]
        adv = next(k for k, r in enumerate(records) if r["op"] == "advance_version")
        assert records[adv + 1]["reason"] == "stale"
        moved = copy.deepcopy(records)
        moved.insert(adv + 1, copy.deepcopy(records[quarantine_index(records)]))
        with pytest.raises(CheckerError, match="have not been evicted"):
            verify_chain(rechain(moved, adv + 1))

    def test_quarantine_of_a_slot_without_an_example_is_rejected(self) -> None:
        records, _ = run()
        i = quarantine_index(records)
        with pytest.raises(CheckerError):
            verify_chain(edited(records, i, lambda r: r.update(index=7)))

    def test_content_state_without_digests_keeps_quarantines_unresolved(self) -> None:
        state = ContentState()
        state.has_content = False
        state.on_mutation({"op": "evict", "index": 0, "reason": "quarantine", "predicate": "p", "note": "n",
                           "op_counter": 1}, 3)
        assert len(state.quarantines) == 1 and state.quarantines[0].content_digest is None


class TestManifestRewards:
    def test_manifest_with_rewards_verifies_and_is_read_back(self) -> None:
        records, manifest = run()
        assert all(line["rewards"] == {"verifier": 1.0, "judge": 0.25} for line in manifest)
        verified = verify_chain(records, manifest=load_manifest("\n".join(json.dumps(l) for l in manifest)))
        assert verified.content.manifest_matched == len(manifest)
        empty = copy.deepcopy(manifest)
        empty[0]["rewards"] = {}                 # every function abstained; distinct from no provenance
        verify_chain(records, manifest=empty)

    def test_manifest_without_rewards_still_verifies(self) -> None:
        records, manifest = run(with_rewards=False)
        assert all("rewards" not in line for line in manifest)
        verify_chain(records, manifest=manifest)

    @pytest.mark.parametrize("rewards, message", [
        ({"judge": "high"}, "rewards"),
        ({"judge": True}, "rewards"),
        ({"judge": float("nan")}, "rewards"),
        ({"judge": float("inf")}, "rewards"),
        ({"": 1.0}, "rewards"),
        ({"a\nb": 1.0}, "rewards"),
        ([1.0], "rewards"),
        ({"judge": [1.0]}, "rewards"),
        ({"judge": 10 ** 400}, "rewards"),
    ])
    def test_malformed_rewards_are_rejected(self, rewards, message) -> None:
        records, manifest = run()
        bad = copy.deepcopy(manifest)
        bad[0]["rewards"] = rewards
        with pytest.raises(CheckerError, match=message):
            verify_chain(records, manifest=bad)

    def test_text_field_beside_rewards_is_rejected(self) -> None:
        records, manifest = run()
        bad = copy.deepcopy(manifest)
        bad[0]["judge_rationale"] = "looked fine"
        with pytest.raises(CheckerError, match="manifest keys"):
            verify_chain(records, manifest=bad)

    def test_rewards_are_outside_the_content_digest(self) -> None:
        records, manifest = run()
        changed = copy.deepcopy(manifest)
        changed[0]["rewards"] = {"verifier": 0.0}
        verify_chain(records, manifest=changed)   # reported, not committed; see nonclaims
        dropped = copy.deepcopy(manifest)
        del dropped[0]["rewards"]
        verify_chain(records, manifest=dropped)
