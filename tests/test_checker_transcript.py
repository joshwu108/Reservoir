"""Tests for checker.transcript — what a verified log says happened.

The transcript answers the audit questions the attestation log exists
for: which examples were sampled and how often (exposure), what share of
sampled rows came from each source (mixture), whether a source stayed
under a limit (quota), and where a given example appears (find). Every
number is checked here against a brute-force walk over the records that
shares no code with the transcript module.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import pytest

from checker.decay_replay import CheckerError
from checker.transcript import build_transcript, main, render_text
from checker.verify import verify_chain, verify_json_lines
from reservoir.attest import AttestationLog, _digest_record
from reservoir.buffer import ExactPERBuffer, Transition
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer

RESULTS = Path(__file__).parents[1] / "benchmarks" / "modal" / "results"
REPO = Path(__file__).parents[1]


def rollouts(rewards: list[float], n_tokens: int = 2) -> list[Rollout]:
    return [Rollout(tokens=list(range(1, n_tokens + 1)), logprobs=[-0.1] * n_tokens, reward=r) for r in rewards]


def build(tmp_path: Path, seed: int = 5, steps: int = 10) -> tuple[list[dict], Path, Path]:
    """Three sources, stale and capacity evictions, slot reuse, one rebase; returns records and paths."""
    attest, manifest = tmp_path / "attest.jsonl", tmp_path / "manifest.jsonl"
    buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=2, seed=seed, attest=attest, manifest=manifest)
    sources = ["a", "b", None]
    for v in range(steps):
        buf.add_group(f"g{v}", v, rollouts([1.0, 0.0, 0.5]), source=sources[v % 3])
        buf.sample(4, current_version=v)
    buf.close()
    assert buf.n_rebases >= 1
    return [json.loads(l) for l in attest.read_text().splitlines()], attest, manifest


def brute_force(records: list[dict]) -> dict:
    """Independent walk: per-digest sample counts and IS-weight sums, per-source counts, per-version counts."""
    slot: dict[int, tuple[str, str | None]] = {}
    times: Counter = Counter()
    weight: dict[str, Fraction] = defaultdict(Fraction)
    by_source: Counter = Counter()
    by_version: dict[int, Counter] = defaultdict(Counter)
    version = 0
    for r in records:
        if r["op"] == "insert":
            slot[r["index"]] = (r["content_digest"], r.get("source"))
        elif r["op"] == "evict":
            slot.pop(r["index"], None)
        elif r["op"] == "advance_version":
            version = int(r["new_version"])
        elif r["op"] == "sample":
            for s in r["samples"]:
                digest, source = slot[s["leaf_index"]]
                times[digest] += 1
                weight[digest] += Fraction(int(s["is_weight_num"]), int(s["is_weight_den"]))
                by_source[source] += 1
                by_version[version][source] += 1
    return {"times": times, "weight": weight, "by_source": by_source, "by_version": by_version}


class TestExposure:
    def test_times_sampled_and_weight_sums_match_brute_force(self, tmp_path) -> None:
        records, _, _ = build(tmp_path)
        report = build_transcript(verify_chain(records))
        expected = brute_force(records)
        assert {d: e["times_sampled"] for d, e in report["content"].items() if e["times_sampled"]} == dict(expected["times"])
        for digest, total in expected["weight"].items():
            entry = report["content"][digest]["sum_is_weight"]
            assert Fraction(int(entry["num"]), int(entry["den"])) == total
            assert entry["float"] == pytest.approx(float(total))

    def test_every_committed_example_is_listed_even_if_never_sampled(self, tmp_path) -> None:
        records, _, _ = build(tmp_path)
        report = build_transcript(verify_chain(records))
        inserted = {r["content_digest"] for r in records if r["op"] == "insert"}
        assert set(report["content"]) == inserted

    def test_insert_history_and_eviction_per_example(self, tmp_path) -> None:
        records, _, _ = build(tmp_path)
        report = build_transcript(verify_chain(records))
        n_inserts = sum(len(e["inserts"]) for e in report["content"].values())
        assert n_inserts == sum(1 for r in records if r["op"] == "insert")
        evicted = [i for e in report["content"].values() for i in e["inserts"] if i["evicted_at_record"] is not None]
        assert len(evicted) == sum(1 for r in records if r["op"] == "evict")
        reasons = set()
        for e in report["content"].values():
            for i in e["inserts"]:
                ins = records[i["record_index"]]
                assert ins["op"] == "insert" and ins["content_digest"] == e["content_digest"]
                if i["evicted_at_record"] is not None:
                    ev = records[i["evicted_at_record"]]
                    assert ev["op"] == "evict" and ev["index"] == ins["index"]
                    assert i["evicted_at_record"] > i["record_index"]
                    between = records[i["record_index"] + 1:i["evicted_at_record"]]
                    assert not any(r["op"] == "insert" and r["index"] == ins["index"] for r in between)
                    reasons.add(ev["reason"])
        assert reasons >= {"stale", "capacity"}

    def test_reused_slot_counts_under_the_right_example(self, tmp_path) -> None:
        # After an eviction a slot is refilled with a different example; samples
        # before and after the refill must land on the two different digests.
        records, _, _ = build(tmp_path)
        report = build_transcript(verify_chain(records))
        by_slot: dict[int, list[tuple[int, str]]] = {}
        for e in report["content"].values():
            for i in e["inserts"]:
                by_slot.setdefault(i["index"], []).append((i["record_index"], e["content_digest"]))
        reused = {slot: sorted(v) for slot, v in by_slot.items() if len({d for _, d in v}) > 1}
        assert reused, "the run never refilled a slot with a different example"
        slot, copies = next(iter(reused.items()))
        finds = build_transcript(verify_chain(records), find=[d for _, d in copies])["find"]
        bounds = copies + [(len(records), None)]
        checked = 0
        for (start, digest), (end, _) in zip(bounds, bounds[1:]):
            for i in range(start + 1, end):
                if records[i]["op"] != "sample":
                    continue
                for s in records[i]["samples"]:
                    if s["leaf_index"] == slot:
                        assert any(h["record_index"] == i for h in finds[digest]), (slot, i, digest)
                        checked += 1
        assert checked > 0

    def test_resolution_agrees_with_the_buffer_itself(self, tmp_path) -> None:
        # A second oracle that replays nothing: the buffer's own sample() return
        # values, compared with the manifest example behind each resolved digest.
        attest, manifest = tmp_path / "attest.jsonl", tmp_path / "manifest.jsonl"
        buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=2, seed=3, attest=attest, manifest=manifest)
        drawn: list[tuple[str, tuple[int, ...]]] = []
        for v in range(8):
            buf.add_group(f"g{v}", v, rollouts([1.0, 0.0, 0.5], n_tokens=v + 1), source="s")
            batch = buf.sample(3, current_version=v)
            drawn += [(g.prompt_id, r.tokens) for r, g in zip(batch.rollouts, batch.groups)]
        buf.close()
        lines = [json.loads(l) for l in manifest.read_text().splitlines()]
        result = verify_json_lines(attest.read_text(), manifest=manifest.read_text())
        example = {l["content_digest"]: (l["prompt_id"], tuple(l["tokens"])) for l in lines}
        assert [example[s.content_digest] for s in result.content.samples] == drawn


class TestMixture:
    def test_source_shares(self, tmp_path) -> None:
        records, _, _ = build(tmp_path)
        report = build_transcript(verify_chain(records))
        expected = brute_force(records)["by_source"]
        total = sum(expected.values())
        for source, count in expected.items():
            key = source if source is not None else "(none)"
            assert report["sources"][key]["sampled"] == count
            assert report["sources"][key]["share"] == pytest.approx(count / total)
        assert sum(s["share"] for s in report["sources"].values()) == pytest.approx(1.0)
        assert sum(s["inserted"] for s in report["sources"].values()) == sum(1 for r in records if r["op"] == "insert")

    def test_per_version_windows(self, tmp_path) -> None:
        records, _, _ = build(tmp_path)
        report = build_transcript(verify_chain(records))
        expected = brute_force(records)["by_version"]
        got = {w["version"]: {k if k != "(none)" else None: v for k, v in w["sampled_by_source"].items()}
               for w in report["windows"] if w["sampled"]}
        assert got == {v: dict(c) for v, c in expected.items()}
        windows = report["windows"]
        assert windows[0]["from_record"] == 0 and windows[-1]["to_record"] == len(records)
        assert all(a["to_record"] == b["from_record"] for a, b in zip(windows, windows[1:]))
        assert all(records[w["from_record"]]["op"] == "advance_version" for w in windows[1:])
        assert sum(w["sampled"] for w in windows) == report["sampled_rows"]

    def test_log_without_advances_has_one_window(self) -> None:
        log = AttestationLog()
        buf = RolloutBuffer(capacity=4, attest=log)
        buf.add_group("p", 0, rollouts([1.0, 0.0]), source="s")
        buf.sample(3)
        report = build_transcript(verify_chain(log.records))
        assert [(w["version"], w["from_record"], w["to_record"], w["sampled"]) for w in report["windows"]] == \
            [(0, 0, len(log.records), 3)]

    def test_repeated_version_opens_a_new_window_without_double_counting(self) -> None:
        # The checker accepts an advance to the same version; the transcript must
        # key windows by position, not by version number.
        log = AttestationLog()
        buf = RolloutBuffer(capacity=4, attest=log)
        buf.add_group("p", 0, rollouts([1.0, 0.0]), source="s")
        buf.sample(2)
        records = [dict(r) for r in log.records]
        n = len(records)
        records.append({"op": "advance_version", "old_version": "0", "new_version": "0", "op_counter": 1,
                        "prev_digest": records[-1]["digest"]})
        records[-1]["digest"] = _digest_record(records[-1])
        buf2 = RolloutBuffer(capacity=4, attest=AttestationLog())
        buf2.add_group("p", 0, rollouts([1.0, 0.0]), source="s")
        buf2.sample(2)
        sample2 = dict(buf2.sample(2) and buf2.attestation_log.records[-1])
        sample2["prev_digest"] = records[-1]["digest"]
        sample2["digest"] = _digest_record(sample2)
        records.append(sample2)
        report = build_transcript(verify_chain(records))
        assert [w["version"] for w in report["windows"]] == [0, 0]
        assert [w["sampled"] for w in report["windows"]] == [2, 2]
        assert sum(w["sampled"] for w in report["windows"]) == report["sampled_rows"] == 4


class TestQuota:
    def test_quota_verdicts(self, tmp_path) -> None:
        records, _, _ = build(tmp_path)
        count_a = brute_force(records)["by_source"]["a"]
        report = build_transcript(verify_chain(records), quotas={"a": count_a, "b": 0, "zzz": 5})
        verdicts = {q["source"]: q for q in report["quotas"]}
        assert verdicts["a"]["ok"] and verdicts["a"]["sampled"] == count_a
        assert not verdicts["b"]["ok"]
        assert verdicts["zzz"]["ok"] and verdicts["zzz"]["sampled"] == 0
        assert report["quota_ok"] is False

    def test_cli_exit_codes(self, tmp_path) -> None:
        records, attest, manifest = build(tmp_path)
        count_a = brute_force(records)["by_source"]["a"]
        assert main([str(attest), "--manifest", str(manifest), "--quota", f"a={count_a}"]) == 0
        assert main([str(attest), "--quota", f"a={count_a - 1}"]) == 2
        tampered = [dict(r) for r in records]
        tampered[2]["index"] = 7                      # breaks the chain: a CheckerError, not a JSON error
        bad = tmp_path / "bad.jsonl"
        bad.write_text("\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in tampered) + "\n")
        assert main([str(bad)]) == 1
        truncated = tmp_path / "cut.jsonl"
        truncated.write_text(attest.read_text()[:-20] + "\n")   # invalid JSON
        assert main([str(truncated)]) == 1

    @pytest.mark.parametrize("arg", ["a", "a=x", "=3", "a=-1", "a=²"])
    def test_malformed_quota_argument(self, tmp_path, arg) -> None:
        _, attest, _ = build(tmp_path)
        assert main([str(attest), "--quota", arg]) == 1

    def test_duplicate_quota_and_usage_errors_exit_1(self, tmp_path, capsys) -> None:
        _, attest, _ = build(tmp_path)
        assert main([str(attest), "--quota", "a=1", "--quota", "a=5"]) == 1
        assert "twice" in capsys.readouterr().err
        with pytest.raises(SystemExit) as exc:
            main([str(attest), "--capacity", "abc"])
        assert exc.value.code == 1
        with pytest.raises(SystemExit) as exc:
            main([str(attest), "--by", "nothing"])
        assert exc.value.code == 1

    def test_quota_on_absent_source_is_noted(self, tmp_path) -> None:
        records, _, _ = build(tmp_path)
        report = build_transcript(verify_chain(records), quotas={"typo": 3})
        assert report["quotas"][0]["present"] is False and report["quotas"][0]["ok"]
        assert "never appears" in render_text(report)


class TestFind:
    def test_find_lists_every_occurrence(self, tmp_path) -> None:
        records, _, _ = build(tmp_path)
        expected = brute_force(records)["times"]
        digest = max(expected, key=expected.get)
        report = build_transcript(verify_chain(records), find=[digest, "00" * 32])
        hits = report["find"][digest]
        assert len(hits) == expected[digest]
        for h in hits:
            rec = records[h["record_index"]]
            assert rec["op"] == "sample"
            assert rec["samples"][h["position_in_batch"]]["leaf_index"] == h["leaf_index"]
            assert h["op_counter"] == rec["op_counter"]
        assert report["find"]["00" * 32] == []

    @pytest.mark.parametrize("bad", ["nothex", "AB" * 32, "ab" * 31, "2a8376…578f"])
    def test_find_rejects_non_digests(self, tmp_path, bad) -> None:
        records, attest, _ = build(tmp_path)
        with pytest.raises(CheckerError, match="--find expects"):
            build_transcript(verify_chain(records), find=[bad])
        assert main([str(attest), "--find", bad]) == 1

    def test_find_with_manifest_shows_the_example(self, tmp_path) -> None:
        records, attest, manifest = build(tmp_path)
        digest = records[1]["content_digest"]
        result = verify_json_lines(attest.read_text(), manifest=manifest.read_text())
        report = build_transcript(result, find=[digest], manifest=[json.loads(l) for l in manifest.read_text().splitlines()])
        assert report["content"][digest]["example"]["prompt_id"] == "g0"
        assert report["content"][digest]["example"]["tokens"] == [1, 2]


class TestLegacyAndText:
    def test_legacy_log_reports_slots_only(self) -> None:
        log = AttestationLog()
        buf = ExactPERBuffer(capacity=4, alpha=1.0, beta=1.0, seed=1)
        for i in range(4):
            pos = buf.insert(Transition(i, 0, 1.0, i + 1, False), td_error=float(i + 1))
            log.append_mutation("insert", pos, 0, buf._sum_tree.get(pos), buf._op_counter)
        batch = buf.sample(3)
        from reservoir.attest import make_sample_entry
        log.append_sample(buf._op_counter, batch.root_total, [
            make_sample_entry(batch.indices[k], batch.draw_integers[k], batch.priorities[k], batch.root_total, batch.is_weights[k])
            for k in range(3)
        ])
        verified = verify_chain(log.records, capacity=4)
        report = build_transcript(verified)
        assert report["has_content"] is False
        assert sum(report["slots"].values()) == 3
        assert "content" not in report and "sources" not in report
        text = render_text(report)
        assert "no content digests" in text
        assert "no per-source view" in render_text(report, by="source")
        # Asking a question the log cannot answer is an error, never a silent pass.
        with pytest.raises(CheckerError, match="need content digests"):
            build_transcript(verified, quotas={"x": 0})
        with pytest.raises(CheckerError, match="need content digests"):
            build_transcript(verified, find=["ab" * 32])

    def test_source_spelled_like_the_untagged_key_is_refused(self) -> None:
        log = AttestationLog()
        buf = RolloutBuffer(capacity=4, attest=log)
        buf.add_group("p", 0, rollouts([1.0]), source="(none)")
        with pytest.raises(CheckerError, match="untagged"):
            build_transcript(verify_chain(log.records))

    def test_by_slot_on_a_content_log_says_so(self, tmp_path) -> None:
        records, _, _ = build(tmp_path)
        assert "per-slot view is for logs without" in render_text(build_transcript(verify_chain(records)), by="slot")

    def test_render_text_mentions_key_numbers(self, tmp_path) -> None:
        records, _, _ = build(tmp_path)
        report = build_transcript(verify_chain(records), quotas={"a": 1000})
        text = render_text(report)
        assert f"{len(records)} records" in text
        assert "40 sampled rows" in text
        assert "a" in text and "quota" in text.lower()

    def test_json_output_round_trips(self, tmp_path) -> None:
        records, attest, manifest = build(tmp_path)
        out = tmp_path / "report.json"
        assert main([str(attest), "--manifest", str(manifest), "--json", str(out)]) == 0
        report = json.loads(out.read_text())
        assert report["records"] == len(records)
        assert report["manifest_matched"] == sum(1 for r in records if r["op"] == "insert")

    def test_cli_subprocess(self, tmp_path) -> None:
        _, attest, manifest = build(tmp_path)
        proc = subprocess.run(
            [sys.executable, "-m", "checker.transcript", str(attest), "--manifest", str(manifest), "--by", "source"],
            capture_output=True, text=True, cwd=REPO,
        )
        assert proc.returncode == 0, proc.stderr
        assert "source" in proc.stdout


COMMITTED = sorted(RESULTS.glob("trl_replay_*.attest.jsonl"))


class TestCommittedLogs:
    def test_committed_logs_exist(self) -> None:
        assert COMMITTED, "no trl_replay_*.attest.jsonl under benchmarks/modal/results"

    @pytest.mark.parametrize("log", COMMITTED, ids=lambda p: p.name)
    def test_transcript_of_committed_log(self, log: Path) -> None:
        # The adapter samples exactly one rollout per dead row, so the sampled
        # row count equals the run record's replaced_rows.
        record = json.loads(log.with_name(log.name.replace(".attest.jsonl", ".json")).read_text())
        manifest = log.with_name(log.name.replace(".attest.jsonl", ".manifest.jsonl"))
        verified = verify_json_lines(log.read_text(), manifest=manifest.read_text() if manifest.exists() else None)
        report = build_transcript(verified)
        assert report["sampled_rows"] == record["totals"]["replaced_rows"]
        assert report["has_content"] == manifest.exists()
