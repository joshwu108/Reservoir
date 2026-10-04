"""Tests for the checker's handling of content commitments.

Every insert record of a 0.5.0 log carries the ``content_digest`` of the
stored example and, optionally, its ``source``. A manifest file holds the
opening of each digest. The checker (``checker/``, which imports nothing
from ``src/reservoir``) must re-derive the digest from the definition with
the standard library, track which example each slot holds across inserts
and evictions so sample records resolve to examples, and, given the
manifest, confirm it opens exactly the log's commitments.

Acceptance tests use real logs from ``RolloutBuffer``. Rejection tests take
a real log or manifest, apply one edit, re-chain the digests where needed
so only the semantic check can catch it, and expect ``CheckerError``.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from checker.content import ContentState, content_digest as checker_content_digest
from checker.content import load_manifest
from checker.verify import CheckerError, VerifiedLog, verify_chain, verify_json_lines
from reservoir.attest import AttestationLog, _digest_record
from reservoir.rollout import Rollout, content_digest_of
from reservoir.rollout_buffer import RolloutBuffer

RESULTS = Path(__file__).parents[1] / "benchmarks" / "modal" / "results"


def rollouts(rewards: list[float], n_tokens: int = 2) -> list[Rollout]:
    return [Rollout(tokens=list(range(1, n_tokens + 1)), logprobs=[-0.1] * n_tokens, reward=r) for r in rewards]


def rechain(records: list[dict], start: int = 0) -> list[dict]:
    """Recompute prev_digest/digest from ``start`` on, as a careful forger would."""
    out = copy.deepcopy(records)
    prev = out[start - 1]["digest"] if start > 0 else "genesis"
    for rec in out[start:]:
        rec["prev_digest"] = prev
        rec["digest"] = _digest_record(rec)
        prev = rec["digest"]
    return out


def to_lines(records: list[dict]) -> str:
    return "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in records)


def build_log(tmp_path: Path, seed: int = 5) -> tuple[list[dict], list[dict], RolloutBuffer]:
    """A run with three sources, stale and capacity evictions, slot reuse and a rebase."""
    buf = RolloutBuffer(
        capacity=8, half_life=1, max_policy_age=2, seed=seed,
        attest=tmp_path / "attest.jsonl", manifest=tmp_path / "manifest.jsonl",
    )
    sources = ["a", "b", None]
    for v in range(10):
        buf.add_group(f"g{v}", v, rollouts([1.0, 0.0, 0.5]), source=sources[v % 3])
        batch = buf.sample(4, current_version=v)
        buf.update_priorities(batch.indices[:1], [0.7])
    buf.close()
    assert buf.n_rebases >= 1
    records = [json.loads(l) for l in (tmp_path / "attest.jsonl").read_text().splitlines()]
    manifest = [json.loads(l) for l in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    return records, manifest, buf


def first_index(records: list[dict], op: str, **match) -> int:
    for i, r in enumerate(records):
        if r["op"] == op and all(r.get(k) == v for k, v in match.items()):
            return i
    raise AssertionError(f"no {op} record matching {match}")


# ---------------------------------------------------------------------------
# Independence: the checker re-derives the digest on its own
# ---------------------------------------------------------------------------

class TestIndependentDerivation:
    @settings(max_examples=200)
    @given(
        prompt_id=st.text(min_size=1, max_size=30),
        tokens=st.lists(st.integers(min_value=0, max_value=1 << 40), min_size=1, max_size=16),
        reward=st.floats(allow_nan=False, allow_infinity=False, width=64, min_value=-1e9, max_value=1e9),
    )
    def test_matches_library(self, prompt_id, tokens, reward) -> None:
        assert checker_content_digest(prompt_id, tokens, reward.hex()) == content_digest_of(prompt_id, tokens, reward)

    def test_known_answer_vectors(self) -> None:
        # Same vectors as tests/test_rollout.py; pinned on both sides.
        assert checker_content_digest("p", [1, 2, 3], (0.5).hex()) == \
            "2a837616faed26935aa920b0fd1357cdd528667c469e6682e554e3b85079578f"
        assert checker_content_digest("é\U0001F600", [0], (-1.0).hex()) == \
            "724b08aaababdc38a1730a9e9167652c6c66ee419e90193be6e509b812ebd15c"

    def test_non_canonical_reward_hex_is_rejected(self) -> None:
        # "0x1p+0" is a valid float literal but not float.hex()'s spelling of 1.0.
        with pytest.raises(CheckerError, match="reward_hex"):
            checker_content_digest("p", [1], "0x1p+0")


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------

class TestAccepts:
    def test_log_with_content_fields_verifies(self, tmp_path) -> None:
        records, _, _ = build_log(tmp_path)
        result = verify_chain(records)
        assert isinstance(result, VerifiedLog)
        assert result.content.has_content

    def test_log_with_manifest_verifies(self, tmp_path) -> None:
        records, manifest, _ = build_log(tmp_path)
        result = verify_chain(records, manifest=manifest)
        assert result.content.manifest_matched == len(manifest) == \
            sum(1 for r in records if r["op"] == "insert")

    def test_content_state_tracks_live_slots(self, tmp_path) -> None:
        records, _, buf = build_log(tmp_path)
        result = verify_chain(records)
        live = {pos: (group.content_digests[group.rollouts.index(rollout)], group.source)
                for pos in buf.live_positions() for rollout, group in [buf.entry(pos)]}
        assert result.content.slots == live

    def test_samples_resolve_to_examples(self, tmp_path) -> None:
        records, _, _ = build_log(tmp_path)
        result = verify_chain(records)
        resolved = result.content.samples
        assert len(resolved) == sum(len(r["samples"]) for r in records if r["op"] == "sample")
        # Independent resolution: walk the records tracking slot occupancy.
        occupant: dict[int, tuple[str, str | None]] = {}
        expected = []
        for i, r in enumerate(records):
            if r["op"] == "insert":
                occupant[r["index"]] = (r["content_digest"], r.get("source"))
            elif r["op"] == "evict":
                del occupant[r["index"]]
            elif r["op"] == "sample":
                expected += [(i, k, *occupant[s["leaf_index"]]) for k, s in enumerate(r["samples"])]
        assert [(s.record_index, s.position_in_batch, s.content_digest, s.source) for s in resolved] == expected

    def test_exact_per_buffer_log_is_legacy(self) -> None:
        from reservoir.attest import make_sample_entry
        from reservoir.buffer import ExactPERBuffer, Transition
        log = AttestationLog()
        buf = ExactPERBuffer(capacity=4, alpha=1.0, beta=1.0, seed=1)
        for i in range(3):
            pos = buf.insert(Transition(i, 0, 1.0, i + 1, False), td_error=float(i + 1))
            log.append_mutation("insert", pos, 0, buf._sum_tree.get(pos), buf._op_counter)
        batch = buf.sample(2)
        log.append_sample(buf._op_counter, batch.root_total, [
            make_sample_entry(batch.indices[k], batch.draw_integers[k], batch.priorities[k],
                              batch.root_total, batch.is_weights[k]) for k in range(2)])
        result = verify_chain(log.records, capacity=4)
        assert result.content.has_content is False
        assert [s.content_digest for s in result.content.samples] == [None, None]
        with pytest.raises(CheckerError, match="no content digests"):
            verify_chain(log.records, capacity=4, manifest=[])

    def test_legacy_log_without_content_verifies(self) -> None:
        log = AttestationLog()
        buf = RolloutBuffer(capacity=4, attest=log)
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        records = [dict(r) for r in log.records]
        for r in records:
            r.pop("content_digest", None)
        records = rechain(records)
        result = verify_chain(records)
        assert not result.content.has_content and result.content.slots == {}

    def test_committed_phase2_log_still_verifies(self) -> None:
        logs = sorted(RESULTS.glob("trl_replay_*.attest.jsonl"))
        assert logs
        for log in logs:
            verify_json_lines(log.read_text())

    def test_verify_json_lines_with_manifest_text(self, tmp_path) -> None:
        records, _, _ = build_log(tmp_path)
        verify_json_lines(to_lines(records), manifest=(tmp_path / "manifest.jsonl").read_text())

    def test_cli_with_manifest(self, tmp_path) -> None:
        records, manifest, _ = build_log(tmp_path)
        proc = subprocess.run(
            [sys.executable, "-m", "checker.verify", str(tmp_path / "attest.jsonl"),
             "--manifest", str(tmp_path / "manifest.jsonl")],
            capture_output=True, text=True, cwd=Path(__file__).parents[1],
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == (
            f"OK: {len(records)} records verified; {len(manifest)} examples committed; "
            f"manifest opens {len(manifest)} of them"
        )

    def test_cli_rejects_bad_manifest(self, tmp_path) -> None:
        build_log(tmp_path)
        (tmp_path / "manifest.jsonl").write_text("")
        proc = subprocess.run(
            [sys.executable, "-m", "checker.verify", str(tmp_path / "attest.jsonl"),
             "--manifest", str(tmp_path / "manifest.jsonl")],
            capture_output=True, text=True, cwd=Path(__file__).parents[1],
        )
        assert proc.returncode == 1 and "FAIL: manifest has 0 lines" in proc.stderr


# ---------------------------------------------------------------------------
# Rejections: content fields
# ---------------------------------------------------------------------------

class TestRejectsContentFields:
    @pytest.mark.parametrize("bad", ["AB" * 32, "ab" * 31, "zz" * 32, 7, ""])
    def test_malformed_digest(self, tmp_path, bad) -> None:
        records, _, _ = build_log(tmp_path)
        i = first_index(records, "insert")
        records[i]["content_digest"] = bad
        with pytest.raises(CheckerError, match="content_digest"):
            verify_chain(rechain(records, i))

    @pytest.mark.parametrize("bad", ["", "a\nb", "x" * 257, 3, "   "])
    def test_malformed_source(self, tmp_path, bad) -> None:
        records, _, _ = build_log(tmp_path)
        i = first_index(records, "insert")
        records[i]["source"] = bad
        with pytest.raises(CheckerError, match="source"):
            verify_chain(rechain(records, i))

    def test_source_without_digest(self, tmp_path) -> None:
        records, _, _ = build_log(tmp_path)
        i = first_index(records, "insert", source="a")
        del records[i]["content_digest"]
        with pytest.raises(CheckerError, match="content_digest"):
            verify_chain(rechain(records, i))

    @pytest.mark.parametrize("op", ["update", "evict"])
    def test_content_fields_on_non_insert(self, tmp_path, op) -> None:
        records, _, _ = build_log(tmp_path)
        i = first_index(records, op)
        records[i]["content_digest"] = "ab" * 32
        with pytest.raises(CheckerError, match="insert"):
            verify_chain(rechain(records, i))

    def test_mixed_inserts_with_and_without_digests(self, tmp_path) -> None:
        records, _, _ = build_log(tmp_path)
        i = first_index(records, "insert") + 1
        assert records[i]["op"] == "insert"
        del records[i]["content_digest"]
        records[i].pop("source", None)
        with pytest.raises(CheckerError, match="content_digest"):
            verify_chain(rechain(records, i))


# ---------------------------------------------------------------------------
# Rejections: manifest
# ---------------------------------------------------------------------------

class TestRejectsManifest:
    def test_missing_line(self, tmp_path) -> None:
        records, manifest, _ = build_log(tmp_path)
        with pytest.raises(CheckerError, match=f"manifest has {len(manifest) - 1} lines for {len(manifest)}"):
            verify_chain(records, manifest=manifest[:-1])

    def test_extra_line(self, tmp_path) -> None:
        records, manifest, _ = build_log(tmp_path)
        extra = dict(manifest[-1], op_counter=999)
        with pytest.raises(CheckerError, match=f"manifest has {len(manifest) + 1} lines"):
            verify_chain(records, manifest=manifest + [extra])

    def test_duplicate_line(self, tmp_path) -> None:
        records, manifest, _ = build_log(tmp_path)
        with pytest.raises(CheckerError, match="manifest has"):
            verify_chain(records, manifest=manifest + [manifest[0]])

    def test_tokens_changed(self, tmp_path) -> None:
        records, manifest, _ = build_log(tmp_path)
        manifest[3]["tokens"] = [9, 9]
        with pytest.raises(CheckerError, match="manifest line 3"):
            verify_chain(records, manifest=manifest)

    def test_reward_changed(self, tmp_path) -> None:
        records, manifest, _ = build_log(tmp_path)
        manifest[0]["reward_hex"] = (2.0).hex()
        with pytest.raises(CheckerError, match="manifest line 0"):
            verify_chain(records, manifest=manifest)

    def test_prompt_changed(self, tmp_path) -> None:
        records, manifest, _ = build_log(tmp_path)
        manifest[0]["prompt_id"] = "other"
        with pytest.raises(CheckerError, match="manifest line 0"):
            verify_chain(records, manifest=manifest)

    def test_digest_disagrees_with_log(self, tmp_path) -> None:
        # A self-consistent line (its own digest recomputes) for a different example.
        records, manifest, _ = build_log(tmp_path)
        line = manifest[0]
        line["tokens"] = [7, 7]
        line["content_digest"] = content_digest_of(line["prompt_id"], [7, 7], float.fromhex(line["reward_hex"]))
        with pytest.raises(CheckerError, match="manifest line 0"):
            verify_chain(records, manifest=manifest)

    def test_source_disagrees_with_log(self, tmp_path) -> None:
        records, manifest, _ = build_log(tmp_path)
        manifest[0]["source"] = "elsewhere"
        with pytest.raises(CheckerError, match="source 'elsewhere' disagrees"):
            verify_chain(records, manifest=manifest)

    def test_null_source_when_log_has_one(self, tmp_path) -> None:
        records, manifest, _ = build_log(tmp_path)
        assert records[first_index(records, "insert")]["source"] == "a" and manifest[0]["source"] == "a"
        manifest[0]["source"] = None
        with pytest.raises(CheckerError, match="source None disagrees"):
            verify_chain(records, manifest=manifest)

    def test_null_source_when_log_has_none_passes(self, tmp_path) -> None:
        records, manifest, _ = build_log(tmp_path)
        i = first_index(records, "insert", op_counter=2)   # third group: source None
        assert "source" not in records[i]
        line = next(m for m in manifest if (m["op_counter"], m["index"]) == (2, records[i]["index"]))
        assert line["source"] is None
        verify_chain(records, manifest=manifest)

    def test_op_counter_or_index_mismatch(self, tmp_path) -> None:
        records, manifest, _ = build_log(tmp_path)
        manifest[0]["op_counter"] = 5
        with pytest.raises(CheckerError, match=r"\(op_counter, index\)"):
            verify_chain(records, manifest=manifest)

    def test_entry_version_disagrees_with_log(self, tmp_path) -> None:
        records, manifest, _ = build_log(tmp_path)
        manifest[0]["entry_version"] += 1
        with pytest.raises(CheckerError, match="entry_version"):
            verify_chain(records, manifest=manifest)

    def test_reordered_lines(self, tmp_path) -> None:
        records, manifest, _ = build_log(tmp_path)
        manifest[0], manifest[1] = manifest[1], manifest[0]
        with pytest.raises(CheckerError, match=r"manifest line 0: \(op_counter, index\)"):
            verify_chain(records, manifest=manifest)

    @pytest.mark.parametrize("edit, message", [
        (lambda m: m.pop("tokens"), "exactly the manifest keys"),
        (lambda m: m.update(extra=1), "exactly the manifest keys"),
        (lambda m: m.update(index="0"), "index must be a non-negative integer"),
        (lambda m: m.update(source=5), "source must be null or a printable string"),
        (lambda m: m.update(reward_hex="0x1p+0"), "canonical float.hex"),
        (lambda m: m.update(tokens=[True]), "tokens must be non-negative integers"),
        (lambda m: m.update(tokens=[-1]), "tokens must be non-negative integers"),
        (lambda m: m.update(prompt_id=""), "prompt_id must be a non-empty string"),
    ])
    def test_malformed_line(self, tmp_path, edit, message) -> None:
        records, manifest, _ = build_log(tmp_path)
        edit(manifest[0])
        with pytest.raises(CheckerError, match=message):
            verify_chain(records, manifest=manifest)

    def test_manifest_for_legacy_log(self, tmp_path) -> None:
        records, manifest, _ = build_log(tmp_path)
        for r in records:
            r.pop("content_digest", None)
            r.pop("source", None)
        with pytest.raises(CheckerError, match="no content digests"):
            verify_chain(rechain(records), manifest=manifest)

    def test_load_manifest_rejects_bad_json_with_file_line_numbers(self) -> None:
        with pytest.raises(CheckerError, match="manifest file line 3"):
            load_manifest('{"a": 1}\n\nnot json\n')
        with pytest.raises(CheckerError, match="manifest file line 1"):
            load_manifest("[1, 2]\n")

    def test_load_manifest_keeps_unicode_separators_inside_strings(self) -> None:
        line = json.dumps({"prompt_id": "a\u2028b"}, ensure_ascii=False)
        assert load_manifest(line + "\n") == [{"prompt_id": "a\u2028b"}]


class TestContentStateUnit:
    def test_evict_frees_slot_and_insert_reuses_it(self) -> None:
        state = ContentState()
        state.on_insert(0, pos=3, digest="ab" * 32, source="a", op_counter=0, entry_version=0)
        state.on_evict(1, pos=3)
        assert state.slots == {}
        state.on_insert(2, pos=3, digest="cd" * 32, source=None, op_counter=1, entry_version=1)
        assert state.slots == {3: ("cd" * 32, None)}
        assert [h.content_digest for h in state.history] == ["ab" * 32, "cd" * 32]

    def test_sample_of_slot_without_content_in_a_content_log_is_an_error(self) -> None:
        state = ContentState()
        state.on_insert(0, pos=0, digest="ab" * 32, source=None, op_counter=0, entry_version=0)
        sample = {"leaf_index": 5, "prob_num": "1", "prob_den": "2", "is_weight_num": "1", "is_weight_den": "1"}
        with pytest.raises(CheckerError, match="no committed example"):
            state.on_sample(1, {"op_counter": 1, "samples": [sample]})

    def test_insert_into_occupied_slot_and_evict_of_empty_slot_are_errors(self) -> None:
        state = ContentState()
        state.on_insert(0, pos=0, digest="ab" * 32, source=None, op_counter=0, entry_version=0)
        with pytest.raises(CheckerError, match="still holds"):
            state.on_insert(1, pos=0, digest="cd" * 32, source=None, op_counter=0, entry_version=0)
        with pytest.raises(CheckerError, match="no committed example"):
            state.on_evict(2, pos=4)

    def test_empty_manifest_for_log_without_inserts(self, tmp_path) -> None:
        buf = RolloutBuffer(capacity=4, attest=tmp_path / "a.jsonl", manifest=tmp_path / "m.jsonl")
        buf.close()
        records = [json.loads(l) for l in (tmp_path / "a.jsonl").read_text().splitlines()]
        assert (tmp_path / "m.jsonl").read_text() == ""
        assert verify_chain(records, manifest=[]).content.manifest_matched == 0
        assert verify_chain([], manifest=[]).content.manifest_matched == 0
        with pytest.raises(CheckerError, match="no inserts"):
            verify_chain(records, manifest=[{"op_counter": 0}])

    @pytest.mark.parametrize("op", ["insert", "sample"])
    def test_missing_op_counter_is_a_checker_error(self, tmp_path, op) -> None:
        records, _, _ = build_log(tmp_path)
        i = first_index(records, op)
        del records[i]["op_counter"]
        with pytest.raises(CheckerError, match="op_counter"):
            verify_chain(rechain(records, i))

    def test_undecayed_content_insert_has_no_entry_version(self) -> None:
        # entry_version is a decay field, so a log without decay_config never has
        # one on a content insert; the history records None for it.
        log = AttestationLog()
        log.append_mutation("insert", 0, 0, 5, 0, content_digest="ab" * 32)
        result = verify_chain(log.records, capacity=4)
        assert result.content.history[0].entry_version is None
