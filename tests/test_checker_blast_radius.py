"""Tests for ``reservoir-transcript --blast-radius``.

Given a content digest, the blast radius is every insert of the example,
every training-batch row that held it (from the batch witnesses), every
step those rows belong to, and every quarantine record that removed it.
With a manifest the radius widens to every example of the same prompt.
Every number is checked against a brute-force walk over the records.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from reservoir.attest import AttestationLog
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer
from reservoir.rollout_manifest import ManifestWriter
from reservoir_checker.content import load_manifest
from reservoir_checker.decay_replay import CheckerError
from reservoir_checker.transcript import blast_radius, build_transcript, main, render_text
from reservoir_checker.verify import verify_chain

REPO = Path(__file__).parents[1]


def rollouts(rewards):
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in rewards]


def run(tmp_path: Path, steps: int = 8) -> tuple[list[dict], list[dict], Path, Path]:
    """Two prompts per step, replayed and witnessed each step, with one quarantine at the end."""
    attest, manifest = tmp_path / "attest.jsonl", tmp_path / "manifest.jsonl"
    buf = RolloutBuffer(capacity=16, half_life=2, max_policy_age=8, seed=7, attest=attest, manifest=manifest)
    for v in range(steps):
        buf.add_group(f"p{v}", v, rollouts([1.0, 0.0, 0.5]), source="s")
        buf.add_group("shared", v, rollouts([0.75 + v]), source="s")
        batch = buf.sample(3, current_version=v)
        buf.witness_batch(batch, step=v, batch_rows=8, rows=[5, 6, 7], tensor_digest="ab" * 32)
    buf.quarantine(lambda r, g: g.prompt_id == "shared", "shared prompt leaked", predicate_text="prompt == shared")
    buf.close()
    records = [json.loads(l) for l in attest.read_text().splitlines()]
    return records, load_manifest(manifest.read_text()), attest, manifest


def brute_force(records: list[dict], digests: set[str]) -> dict:
    """Rows, steps and quarantines for a set of digests, from the raw records."""
    slot: dict[int, str] = {}
    by_op: dict[int, list[str]] = {}
    rows, quarantined, inserts = [], [], []
    for i, r in enumerate(records):
        if r["op"] == "insert":
            slot[r["index"]] = r["content_digest"]
            if r["content_digest"] in digests:
                inserts.append(i)
        elif r["op"] == "evict":
            if r.get("reason") == "quarantine" and slot[r["index"]] in digests:
                quarantined.append(i)
            slot.pop(r["index"], None)
        elif r["op"] == "sample":
            by_op[r["op_counter"]] = [slot[s["leaf_index"]] for s in r["samples"]]
        elif r["op"] == "batch":
            for e in r["replaced"]:
                if by_op[r["sample_op_counter"]][e["draw"]] in digests:
                    rows.append((int(r["step"]), e["row"]))
    return {"rows": sorted(rows), "steps": sorted({s for s, _ in rows}), "quarantined": quarantined,
            "inserts": inserts}


class TestBlastRadius:
    def test_exact_digest_matches_brute_force(self, tmp_path) -> None:
        records, manifest, _, _ = run(tmp_path)
        verified = verify_chain(records)
        digest = next(s.content_digest for s in verified.content.samples
                      if manifest[[l["content_digest"] for l in manifest].index(s.content_digest)]["prompt_id"] != "shared")
        report = blast_radius(verified, [digest])
        entry = report[digest]
        expected = brute_force(records, {digest})
        assert entry["found"] and entry["examples"] == [digest]
        assert [(r["step"], r["row"]) for r in entry["rows"]] == expected["rows"]
        assert entry["steps"] == expected["steps"]
        assert [i["record_index"] for i in entry["inserts"]] == expected["inserts"]
        assert entry["quarantined"] == []
        assert entry["prompt_id"] is None
        assert entry["times_sampled"] == sum(1 for s in verified.content.samples if s.content_digest == digest) > 0
        assert entry["unwitnessed_draws"] == 0 and "lower bound" not in entry.get("note", "")

    def test_unwitnessed_draws_make_the_steps_a_lower_bound(self) -> None:
        buf = RolloutBuffer(capacity=8, half_life=2, max_policy_age=8, seed=1, attest=AttestationLog())
        buf.add_group("p", 0, rollouts([1.0]), source="s")
        buf.sample(2, current_version=0)                                   # drawn, never witnessed
        batch = buf.sample(1, current_version=1)
        buf.witness_batch(batch, step=1, batch_rows=4, rows=[3], tensor_digest="ab" * 32)
        records = [dict(r) for r in buf.attestation_log.records]
        digest = records[1]["content_digest"]
        entry = blast_radius(verify_chain(records), [digest])[digest]
        assert entry["times_sampled"] == 3 and entry["unwitnessed_draws"] == 2
        assert entry["steps"] == [1] and "lower bound" in entry["note"]

    def test_manifest_widens_to_the_prompt_and_lists_quarantines(self, tmp_path) -> None:
        records, manifest, _, _ = run(tmp_path)
        shared = [l["content_digest"] for l in manifest if l["prompt_id"] == "shared"]
        assert len(shared) >= 2
        report = blast_radius(verify_chain(records, manifest=manifest), [shared[0]], manifest=manifest)
        entry = report[shared[0]]
        expected = brute_force(records, set(shared))
        assert entry["prompt_id"] == "shared" and sorted(entry["examples"]) == sorted(set(shared))
        assert [(r["step"], r["row"]) for r in entry["rows"]] == expected["rows"]
        assert entry["steps"] == expected["steps"]
        assert [q["record_index"] for q in entry["quarantined"]] == expected["quarantined"]
        assert all(q["predicate"] == "prompt == shared" and q["note"] == "shared prompt leaked" for q in entry["quarantined"])
        assert {r["content_digest"] for r in entry["rows"]} <= set(shared)
        assert entry["rows"] == sorted(entry["rows"], key=lambda r: (r["step"], r["row"]))

    def test_rows_name_their_witness_and_draw(self, tmp_path) -> None:
        records, manifest, _, _ = run(tmp_path)
        verified = verify_chain(records, manifest=manifest)
        digest = verified.content.witnessed_rows[0].content_digest
        entry = blast_radius(verified, [digest])[digest]
        hit = entry["rows"][0]
        witness = records[hit["record_index"]]
        assert witness["op"] == "batch" and witness["sample_op_counter"] == hit["sample_op_counter"]
        assert any(e["row"] == hit["row"] and e["draw"] == hit["draw"] for e in witness["replaced"])

    def test_unknown_digest_is_reported_not_found(self, tmp_path) -> None:
        records, _, _, _ = run(tmp_path)
        entry = blast_radius(verify_chain(records), ["0" * 64])["0" * 64]
        assert entry == {"found": False, "prompt_id": None, "examples": [], "inserts": [], "rows": [],
                         "steps": [], "times_sampled": 0, "unwitnessed_draws": 0, "quarantined": [],
                         "note": "this digest was never committed in the log"}

    def test_log_without_witnesses_says_so(self) -> None:
        buf = RolloutBuffer(capacity=4, attest=AttestationLog())
        buf.add_group("p", 0, rollouts([1.0, 0.0]), source="s")
        buf.sample(2)
        records = [dict(r) for r in buf.attestation_log.records]
        digest = records[1]["content_digest"]
        entry = blast_radius(verify_chain(records), [digest])[digest]
        assert entry["found"] and entry["rows"] == [] and "witness" in entry["note"]

    @pytest.mark.parametrize("bad", ["xyz", "0" * 63, "G" * 64])
    def test_rejects_non_digests(self, tmp_path, bad) -> None:
        records, _, _, _ = run(tmp_path)
        with pytest.raises(CheckerError, match="--blast-radius"):
            build_transcript(verify_chain(records), blast=[bad])

    def test_needs_content_digests(self) -> None:
        from fractions import Fraction
        from reservoir.attest import make_sample_entry
        log = AttestationLog()
        log.append_mutation("insert", 0, 0, 5, 1)
        log.append_sample(1, 5, [make_sample_entry(leaf_index=0, draw_int=3, priority_int=5, root_total=5,
                                                   is_weight=Fraction(1))])
        with pytest.raises(CheckerError, match="content digests"):
            build_transcript(verify_chain(log.records, capacity=4), blast=["0" * 64])

    def test_report_and_text_and_cli(self, tmp_path) -> None:
        records, manifest, attest, manifest_path = run(tmp_path)
        shared = next(l["content_digest"] for l in manifest if l["prompt_id"] == "shared")
        report = build_transcript(verify_chain(records, manifest=manifest), blast=[shared], manifest=manifest)
        text = render_text(report)
        assert f"blast radius {shared[:16]}" in text and "quarantined" in text
        out = tmp_path / "report.json"
        code = main([str(attest), "--manifest", str(manifest_path), "--blast-radius", shared, "--json", str(out)])
        assert code == 0
        assert json.loads(out.read_text())["blast_radius"][shared]["prompt_id"] == "shared"
        proc = subprocess.run(
            [sys.executable, "-m", "checker.transcript", str(attest), "--blast-radius", shared],
            capture_output=True, text=True, cwd=REPO,
        )
        assert proc.returncode == 0, proc.stderr
        assert "blast radius" in proc.stdout

    def test_example_shows_rewards_from_the_manifest(self, tmp_path) -> None:
        manifest = ManifestWriter()
        buf = RolloutBuffer(capacity=4, attest=AttestationLog(), manifest=manifest)
        buf.add_group("p", 0, [Rollout(tokens=[1], logprobs=[-0.1], reward=1.0,
                                       metadata={"rewards": {"verifier": 1.0, "judge": 0.0}})], source="s")
        buf.sample(1)
        records = [dict(r) for r in buf.attestation_log.records]
        report = build_transcript(verify_chain(records, manifest=manifest.records), manifest=manifest.records)
        (entry,) = report["content"].values()
        assert entry["example"]["rewards"] == {"verifier": 1.0, "judge": 0.0}
