"""Offline replay: every witnessed training batch, as content, from the log and manifest alone.

``reservoir_checker.replay`` turns a verified attestation log and its
manifest into JSON lines: one header, then one line per batch witness
naming every replaced row with its draw, slot, importance weight,
probability and the example (prompt id, tokens, reward, source) the
manifest opens for it. It imports nothing from ``reservoir``; the tests
here build fixtures with the library and check the replay against the
raw records, and run it over the committed reproducibility artifacts.
"""

from __future__ import annotations

import copy
import io
import json
from fractions import Fraction
from pathlib import Path

import pytest

from reservoir.attest import AttestationLog, _digest_record
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer
from reservoir.rollout_manifest import ManifestWriter
from reservoir_checker.content import content_digest
from reservoir_checker.replay import MIN_FORMAT, main, render, replay, replay_json_lines
from reservoir_checker.verify import CheckerError

REPO = Path(__file__).parents[1]
RESULTS = REPO / "benchmarks" / "modal" / "results"
REPRO_DIRS = sorted(p for p in RESULTS.glob("repro_*") if p.is_dir())
RUNS = [(d, sub) for d in REPRO_DIRS for sub in ("a", "b", "c")]


def _load(directory: Path) -> tuple[list[dict], list[dict]]:
    records = [json.loads(l) for l in (directory / "attest.jsonl").read_text().split("\n") if l.strip()]
    manifest = [json.loads(l) for l in (directory / "manifest.jsonl").read_text().split("\n") if l.strip()]
    return records, manifest


def _witness_records(records: list[dict]) -> list[dict]:
    return [r for r in records if r["op"] == "batch"]


def _sample_record(records: list[dict], op_counter: int) -> dict:
    return next(r for r in records if r["op"] == "sample" and int(r["op_counter"]) == op_counter)


def rollouts(values, **meta):
    return [Rollout(tokens=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3], reward=r, metadata=meta or None) for r in values]


def witnessed_run(seed: int = 2, versions: int = 4, capacity: int = 8, with_rewards: bool = True):
    """A buffer run with a manifest and one witnessed sample per version."""
    manifest = ManifestWriter()
    buf = RolloutBuffer(capacity=capacity, half_life=1, max_policy_age=2, seed=seed, attest=AttestationLog(),
                        manifest=manifest)
    meta = {"rewards": {"verifier": 1.0, "judge": 0.25}} if with_rewards else {}
    for v in range(versions):
        buf.add_group(f"g{v}", v, rollouts([1.0, 0.0, 0.5], **meta), source="s")
        batch = buf.sample(2, current_version=v)
        buf.witness_batch(batch, step=v, batch_rows=6, rows=[4, 5], tensor_digest="ab" * 32)
    return [dict(r) for r in buf.attestation_log.records], [dict(l) for l in manifest.records]


def rechain(records: list[dict], start: int) -> list[dict]:
    out = copy.deepcopy(records)
    prev = out[start - 1]["digest"] if start > 0 else "genesis"
    for rec in out[start:]:
        rec["prev_digest"] = prev
        rec["digest"] = _digest_record(rec)
        prev = rec["digest"]
    return out


def edited_manifest(manifest: list[dict], line: int, edit) -> list[dict]:
    m = copy.deepcopy(manifest)
    edit(m[line])
    return m


# ---------------------------------------------------------------------------
# The committed reproducibility artifacts
# ---------------------------------------------------------------------------

@pytest.fixture(params=RUNS, ids=[f"{d.name}/{s}" for d, s in RUNS])
def committed(request) -> tuple[list[dict], list[dict], list[dict]]:
    directory, sub = request.param
    records, manifest = _load(directory / sub)
    return records, manifest, replay(records, manifest)


def test_the_committed_directories_exist():
    assert len(REPRO_DIRS) == 3 and len(RUNS) == 9


def test_committed_run_replays_every_witness(committed):
    records, manifest, lines = committed
    header, batches = lines[0], lines[1:]
    assert header["kind"] == "header" and header["tool"] == "reservoir-replay-offline"
    assert header["format"] == 2 and header["records"] == len(records)
    assert header["examples"] == len(manifest)
    assert header["witnesses"] == len(batches) == len(_witness_records(records)) > 0
    assert header["head_digest"] == records[-1]["digest"]
    assert header["steps_witnessed"] == [b["step"] for b in batches]
    telemetry_steps = {int(r["step"]) for r in records if r["op"] == "telemetry"}
    assert set(header["steps_without_witness"]) == telemetry_steps - set(header["steps_witnessed"])
    assert all(b["kind"] == "batch" for b in batches)


def test_rows_are_the_witness_opened_by_the_manifest(committed):
    records, manifest, lines = committed
    by_digest = {m["content_digest"]: m for m in manifest}
    for batch, witness in zip(lines[1:], _witness_records(records)):
        assert batch["step"] == int(witness["step"])
        assert batch["sample_op_counter"] == witness["sample_op_counter"]
        assert batch["batch_rows"] == witness["batch_rows"]
        assert batch["tensor_digest"] == witness["tensor_digest"]
        assert batch["declined"] == witness.get("declined", [])
        assert len(batch["rows"]) == len(witness["replaced"])
        replaced = {e["row"]: e for e in witness["replaced"]}
        for row in batch["rows"]:
            entry = replaced[row["row"]]
            assert row["draw"] == entry["draw"] and row["content_digest"] == entry["content_digest"]
            line = manifest[row["manifest_line"]]
            assert line["content_digest"] == row["content_digest"] == by_digest[row["content_digest"]]["content_digest"]
            assert row["tokens"] == line["tokens"] and row["prompt_id"] == line["prompt_id"]
            assert row["reward_hex"] == line["reward_hex"] and row["reward"] == float.fromhex(line["reward_hex"])
            assert row["source"] == line["source"] and row["entry_version"] == line["entry_version"]
            assert content_digest(row["prompt_id"], row["tokens"], row["reward_hex"]) == row["content_digest"]
            assert "rewards" not in row    # the committed manifests predate reward provenance
        assert batch["fresh_rows"] == sorted(set(range(witness["batch_rows"])) - set(replaced))


def test_weights_and_slots_come_from_the_sample_record(committed):
    records, _, lines = committed
    for batch in lines[1:]:
        sample = _sample_record(records, batch["sample_op_counter"])
        for row in batch["rows"]:
            draw = sample["samples"][row["draw"]]
            assert row["slot"] == draw["leaf_index"]
            assert Fraction(*map(int, row["is_weight_exact"])) == Fraction(int(draw["is_weight_num"]), int(draw["is_weight_den"]))
            assert Fraction(*map(int, row["probability_exact"])) == Fraction(int(draw["prob_num"]), int(draw["prob_den"]))
            assert row["is_weight"] == float(Fraction(int(draw["is_weight_num"]), int(draw["is_weight_den"])))
            assert row["probability"] == float(Fraction(int(draw["prob_num"]), int(draw["prob_den"])))
            assert 0 < row["is_weight"] <= 1


def test_each_row_resolves_to_the_insert_live_at_the_draw(committed):
    records, manifest, lines = committed
    for batch in lines[1:]:
        sample_at = next(i for i, r in enumerate(records)
                         if r["op"] == "sample" and int(r["op_counter"]) == batch["sample_op_counter"])
        for row in batch["rows"]:
            live = max(i for i, r in enumerate(records)
                       if i < sample_at and r["op"] == "insert" and r["index"] == row["slot"])
            insert = records[live]
            line = manifest[row["manifest_line"]]
            assert (line["op_counter"], line["index"]) == (insert["op_counter"], insert["index"])
            assert line["content_digest"] == insert["content_digest"]


def test_generated_examples_are_the_steps_inserts(committed):
    records, manifest, lines = committed
    for batch in lines[1:]:
        expected = [r["content_digest"] for r in records
                    if r["op"] == "insert" and int(r["entry_version"]) == batch["step"]]
        assert [g["content_digest"] for g in batch["generated"]] == expected
        for g in batch["generated"]:
            line = manifest[g["manifest_line"]]
            assert g["tokens"] == line["tokens"] and g["reward_hex"] == line["reward_hex"]
            assert g["entry_version"] == batch["step"] and set(g) == set(batch["rows"][0]) - {
                "row", "draw", "slot", "is_weight", "is_weight_exact", "probability", "probability_exact"}


@pytest.mark.parametrize("directory", REPRO_DIRS, ids=[d.name for d in REPRO_DIRS])
def test_identical_runs_replay_identically_and_a_different_seed_does_not(directory: Path):
    outputs = {sub: render(replay(*_load(directory / sub))) for sub in ("a", "b", "c")}
    assert outputs["a"] == outputs["b"]
    assert outputs["a"] != outputs["c"]
    assert outputs["a"].endswith("\n") and all(json.loads(l) for l in outputs["a"].split("\n") if l)


def test_rendered_lines_are_canonical_and_one_per_record(committed):
    _, _, lines = committed
    text = render(lines)
    assert text.count("\n") == len(lines)
    rendered = text.split("\n")[:-1]
    assert [json.loads(l) for l in rendered] == lines
    assert rendered == [json.dumps(l, sort_keys=True, separators=(",", ":")) for l in lines]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

class TestRefusals:
    def test_without_a_manifest(self):
        records, _ = witnessed_run()
        with pytest.raises(CheckerError, match="manifest"):
            replay(records, None)

    def test_format_below_the_minimum(self):
        records, manifest = witnessed_run()
        assert MIN_FORMAT == 2
        old = rechain([{**records[0], "format": "1"}] + records[1:], 0)
        with pytest.raises(CheckerError, match=r"format-2 log or later.*format 1"):
            replay(old, manifest)

    def test_a_log_without_decay_config_is_format_1(self):
        records, manifest = witnessed_run()
        headless = rechain(records[1:], 0)
        with pytest.raises(CheckerError, match=r"format-2 log or later.*format 1"):
            replay(headless, manifest)

    def test_a_format_value_that_is_not_a_number(self):
        records, manifest = witnessed_run()
        bad = rechain([{**records[0], "format": "two"}] + records[1:], 0)
        with pytest.raises(CheckerError, match="format"):
            replay(bad, manifest)

    def test_a_log_that_does_not_verify(self):
        records, manifest = witnessed_run()
        broken = copy.deepcopy(records)
        broken[3]["source"] = "forged"
        with pytest.raises(CheckerError, match="digest mismatch"):
            replay(broken, manifest)

    def test_an_empty_log(self):
        with pytest.raises(CheckerError, match="format"):
            replay([], [])

    def test_a_manifest_with_a_changed_token(self):
        records, manifest = witnessed_run()
        with pytest.raises(CheckerError, match="content_digest"):
            replay(records, edited_manifest(manifest, 0, lambda l: l["tokens"].append(9)))

    def test_a_manifest_missing_a_line(self):
        records, manifest = witnessed_run()
        with pytest.raises(CheckerError, match="lines"):
            replay(records, manifest[:-1])


# ---------------------------------------------------------------------------
# Content of the output on library-built runs
# ---------------------------------------------------------------------------

class TestOutput:
    def test_rewards_are_carried_from_the_manifest(self):
        records, manifest = witnessed_run()
        lines = replay(records, manifest)
        assert len(lines) == 1 + 4
        for batch in lines[1:]:
            assert batch["batch_rows"] == 6 and [r["row"] for r in batch["rows"]] == [4, 5]
            assert batch["fresh_rows"] == [0, 1, 2, 3] and batch["declined"] == []
            for row in batch["rows"]:
                assert row["rewards"] == {"verifier": 1.0, "judge": 0.25}
                assert row["source"] == "s" and row["tokens"] == [1, 2, 3]
                assert row["reward"] in (1.0, 0.0, 0.5)

    def test_a_changed_reward_value_changes_the_output_of_the_row_it_opens(self):
        records, manifest = witnessed_run()
        base = replay(records, manifest)
        target = base[1]["rows"][0]["manifest_line"]
        changed = replay(records, edited_manifest(manifest, target, lambda l: l.update(rewards={"verifier": 0.0})))
        assert changed != base
        rows = [r for b in changed[1:] for r in b["rows"] if r["manifest_line"] == target]
        assert rows and all(r["rewards"] == {"verifier": 0.0} for r in rows)
        untouched = [r for b in changed[1:] for r in b["rows"] if r["manifest_line"] != target]
        assert all(r["rewards"] == {"verifier": 1.0, "judge": 0.25} for r in untouched)

    def test_no_witnesses_gives_a_header_only(self):
        manifest = ManifestWriter()
        buf = RolloutBuffer(capacity=4, half_life=1, max_policy_age=2, seed=1, attest=AttestationLog(),
                            manifest=manifest)
        buf.add_group("g", 0, rollouts([1.0, 0.0]), source="s")
        buf.sample(1, current_version=0)
        lines = replay([dict(r) for r in buf.attestation_log.records], manifest.records)
        assert len(lines) == 1 and lines[0]["witnesses"] == 0 and lines[0]["examples"] == 2
        assert lines[0]["steps_witnessed"] == [] and lines[0]["steps_without_witness"] == []

    def test_a_reused_slot_resolves_to_the_example_the_draw_saw(self):
        records, manifest = witnessed_run(seed=3, versions=8, capacity=4)
        inserts_by_slot: dict[int, int] = {}
        reused_before_a_witness = False
        for i, r in enumerate(records):
            if r["op"] == "insert":
                inserts_by_slot[r["index"]] = inserts_by_slot.get(r["index"], 0) + 1
            if r["op"] == "batch" and any(n > 1 for n in inserts_by_slot.values()):
                reused_before_a_witness = True
        assert reused_before_a_witness, "the fixture must reuse a slot before a later witness"
        first_insert_of_slot = {}
        for r in records:
            if r["op"] == "insert":
                first_insert_of_slot.setdefault(r["index"], (r["op_counter"], r["index"]))
        resolved_to_a_later_occupant = False
        for batch in replay(records, manifest)[1:]:
            sample_at = next(i for i, r in enumerate(records)
                             if r["op"] == "sample" and int(r["op_counter"]) == batch["sample_op_counter"])
            for row in batch["rows"]:
                live = max(i for i, r in enumerate(records)
                           if i < sample_at and r["op"] == "insert" and r["index"] == row["slot"])
                line = manifest[row["manifest_line"]]
                assert (line["op_counter"], line["index"]) == (records[live]["op_counter"], records[live]["index"])
                assert row["content_digest"] == records[live]["content_digest"]
                if (line["op_counter"], line["index"]) != first_insert_of_slot[row["slot"]]:
                    resolved_to_a_later_occupant = True
        assert resolved_to_a_later_occupant, "no witnessed row drew a reused slot; change the seed"

    def test_generated_excludes_inserts_after_the_witness_and_is_shared_by_a_steps_witnesses(self):
        manifest = ManifestWriter()
        buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=2, seed=6, attest=AttestationLog(),
                            manifest=manifest)
        buf.add_group("early", 0, rollouts([1.0, 0.0]), source="s")
        first = buf.sample(1, current_version=0)
        buf.witness_batch(first, step=0, batch_rows=4, rows=[3], tensor_digest="ab" * 32)
        buf.add_group("late", 0, rollouts([1.0, 0.0]), source="s")      # same entry version, after the witness
        second = buf.sample(1, current_version=0)
        buf.witness_batch(second, step=0, batch_rows=4, rows=[2], tensor_digest="cd" * 32)
        lines = replay([dict(r) for r in buf.attestation_log.records], manifest.records)
        assert lines[0]["witnesses"] == 2 and lines[0]["steps_witnessed"] == [0, 0]
        assert [g["prompt_id"] for g in lines[1]["generated"]] == ["early", "early"]
        assert [g["prompt_id"] for g in lines[2]["generated"]] == ["early", "early", "late", "late"]
        assert lines[1]["sample_op_counter"] != lines[2]["sample_op_counter"]

    def test_steps_with_telemetry_but_no_witness_are_listed(self):
        records, manifest = witnessed_run()
        manifest_w = ManifestWriter()
        buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=2, seed=2, attest=AttestationLog(),
                            manifest=manifest_w)
        zero = {"batch_rows": 6, "replaced_rows": 0, "declined_rows": 0, "dead_groups": 0, "near_dead_groups": 0}
        buf.add_group("g0", 0, rollouts([1.0, 0.0]), source="s")
        buf.record_telemetry(0, zero)
        buf.add_group("g1", 1, rollouts([1.0, 0.0]), source="s")
        batch = buf.sample(1, current_version=1)
        buf.witness_batch(batch, step=1, batch_rows=6, rows=[5], tensor_digest="ab" * 32)
        lines = replay([dict(r) for r in buf.attestation_log.records], manifest_w.records)
        assert lines[0]["steps_witnessed"] == [1] and lines[0]["steps_without_witness"] == [0]
        assert lines[0]["format"] == 3    # the library writes the current format; the floor is 2

    def test_declined_draws_are_listed_and_not_rows(self):
        manifest = ManifestWriter()
        buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=2, seed=4, attest=AttestationLog(),
                            manifest=manifest)
        buf.add_group("g", 0, rollouts([1.0, 0.0, 0.5]), source="s")
        batch = buf.sample(3, current_version=0)
        buf.witness_batch(batch, step=0, batch_rows=4, rows=[1, 2], tensor_digest="ab" * 32, declined=[1])
        lines = replay([dict(r) for r in buf.attestation_log.records], manifest.records)
        assert lines[1]["declined"] == [1] and [r["draw"] for r in lines[1]["rows"]] == [0, 2]
        assert lines[1]["fresh_rows"] == [0, 3]

    def test_json_lines_entry_point_matches_the_record_api(self):
        records, manifest = witnessed_run()
        text = "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
        manifest_text = "\n".join(json.dumps(l, sort_keys=True) for l in manifest) + "\n"
        assert replay_json_lines(text, manifest_text) == replay(records, manifest)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

class TestCommandLine:
    @pytest.fixture
    def files(self, tmp_path: Path) -> tuple[Path, Path, list[dict], list[dict]]:
        records, manifest = witnessed_run()
        log = tmp_path / "attest.jsonl"
        man = tmp_path / "manifest.jsonl"
        log.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n")
        man.write_text("\n".join(json.dumps(l, sort_keys=True) for l in manifest) + "\n")
        return log, man, records, manifest

    def test_writes_json_lines_to_a_file(self, files, tmp_path, capsys):
        log, man, records, manifest = files
        out = tmp_path / "replay.jsonl"
        assert main([str(log), "--manifest", str(man), "--out", str(out)]) == 0
        assert out.read_text() == render(replay(records, manifest))
        captured = capsys.readouterr()
        assert "4 witnessed batches" in captured.out and "12 examples" in captured.out and captured.err == ""

    def test_writes_to_stdout_by_default(self, files, capsys):
        log, man, records, manifest = files
        assert main([str(log), "--manifest", str(man)]) == 0
        captured = capsys.readouterr()
        assert captured.out == render(replay(records, manifest))
        assert "4 witnessed batches" in captured.err

    def test_refuses_without_manifest(self, files, capsys):
        log, _, _, _ = files
        assert main([str(log)]) == 1
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err.startswith("FAIL:") and "manifest" in captured.err

    def test_reports_a_tampered_manifest_as_failure(self, files, tmp_path, capsys):
        log, man, _, manifest = files
        tampered = edited_manifest(manifest, 0, lambda l: l.update(prompt_id="other"))
        man.write_text("\n".join(json.dumps(l) for l in tampered) + "\n")
        out = tmp_path / "replay.jsonl"
        assert main([str(log), "--manifest", str(man), "--out", str(out)]) == 1
        assert not out.exists()
        assert capsys.readouterr().err.startswith("FAIL:")

    def test_reports_an_old_format_as_failure(self, files, tmp_path, capsys):
        log, man, records, _ = files
        old = rechain([{**records[0], "format": "1"}] + records[1:], 0)
        log.write_text("\n".join(json.dumps(r, sort_keys=True) for r in old) + "\n")
        assert main([str(log), "--manifest", str(man)]) == 1
        assert "format-2 log or later" in capsys.readouterr().err

    def test_malformed_files_fail_without_a_traceback(self, files, tmp_path, capsys):
        log, man, _, _ = files
        man.write_text("{not json\n")
        assert main([str(log), "--manifest", str(man)]) == 1
        assert capsys.readouterr().err.startswith("FAIL:")
        assert main([str(tmp_path / "missing.jsonl"), "--manifest", str(man)]) == 1
        assert capsys.readouterr().err.startswith("FAIL:")

    def test_runs_the_committed_cpu_artifact_end_to_end(self, tmp_path):
        directory = RESULTS / "repro_cpu_12steps_seed42" / "a"
        out = tmp_path / "replay.jsonl"
        assert main([str(directory / "attest.jsonl"), "--manifest", str(directory / "manifest.jsonl"),
                     "--out", str(out)]) == 0
        lines = [json.loads(l) for l in out.read_text().split("\n") if l]
        assert lines == replay(*_load(directory))
