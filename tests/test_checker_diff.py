"""Tests for checker.diff — where and why two attestation logs diverge.

Two runs of the same training script should produce the same log. When
they do not, the diff must say at which record they part and what kind of
record it is: a different stored example or score (``data``), a different
step cadence (``schedule``), different buffer parameters (``config``), a
different draw on identical state (``sampler``: a different seed, or a
bug), one log being a prefix of the other (``truncated``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from checker.diff import _classify, diff_logs, main
from checker.verify import CheckerError
from reservoir.attest import AttestationLog
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer

REPO = Path(__file__).parents[1]


def rollouts(rewards: list[float]) -> list[Rollout]:
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in rewards]


def run(seed: int = 0, rewards=(1.0, 0.0, 0.5), versions=range(6), source="a", **kw) -> list[dict]:
    """A small run; every parameter is a knob for one kind of divergence."""
    params = dict(capacity=8, half_life=1, max_policy_age=2, seed=seed, attest=AttestationLog())
    params.update(kw)
    buf = RolloutBuffer(**params)
    for v in versions:
        buf.add_group(f"g{v}", v, rollouts(list(rewards)), source=source)
        buf.sample(2, current_version=v)
    return buf.attestation_log.records


def to_lines(records: list[dict]) -> str:
    return "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in records) + "\n"


class TestClassification:
    def test_identical(self) -> None:
        a, b = run(), run()
        result = diff_logs(a, b)
        assert result["identical"] is True and result["class"] == "identical"
        assert result["first_difference"] is None
        assert result["head_a"] == result["head_b"]

    def test_data_when_a_stored_example_differs(self) -> None:
        a, b = run(), run(rewards=(2.0, 0.0, 0.5))
        result = diff_logs(a, b)
        assert result["class"] == "data"
        idx = result["first_difference"]
        assert a[idx]["op"] == b[idx]["op"] == "insert"
        assert "content_digest" in result["differing_fields"]
        assert a[:idx] == b[:idx]

    def test_data_when_only_a_score_differs(self) -> None:
        # Same examples, different group mean: the digest is the same, the priority is not.
        a, b = run(), run(rewards=(1.0, 0.0, 0.25))
        result = diff_logs(a, b)
        assert result["class"] == "data"
        assert "content_digest" not in result["differing_fields"]
        assert "base_priority_int" in result["differing_fields"]

    def test_data_when_only_the_source_differs(self) -> None:
        a, b = run(source="a"), run(source="b")
        result = diff_logs(a, b)
        assert result["class"] == "data" and result["differing_fields"] == ["source"]

    def test_schedule_when_versions_differ(self) -> None:
        a, b = run(versions=range(6)), run(versions=[0, 1, 2, 4, 5, 6])
        result = diff_logs(a, b)
        assert result["class"] == "schedule"
        assert "advance_version" in (a[result["first_difference"]]["op"], b[result["first_difference"]]["op"])

    def test_config_when_parameters_differ(self) -> None:
        a, b = run(), run(half_life=2)
        result = diff_logs(a, b)
        assert result["class"] == "config" and result["first_difference"] == 0

    def test_config_when_only_the_seed_differs(self) -> None:
        # The seed is recorded in decay_config, so two seeds differ at record 0.
        a, b = run(seed=0), run(seed=1)
        result = diff_logs(a, b)
        assert result["class"] == "config" and result["first_difference"] == 0
        assert result["differing_fields"] == ["seed"]

    def test_sampler_when_seeds_differ_in_logs_without_a_recorded_seed(self) -> None:
        # Strip the draw configuration from both: the logs then share every
        # record up to the first sample, which differs.
        from reservoir.attest import _digest_record

        def strip(records):
            out = [dict(r) for r in records]
            for name in ("seed", "buffer_id", "alpha", "beta"):
                out[0].pop(name, None)
            prev = "genesis"
            for rec in out:
                rec["prev_digest"] = prev
                rec["digest"] = _digest_record(rec)
                prev = rec["digest"]
            return out

        a, b = strip(run(seed=0)), strip(run(seed=1))
        result = diff_logs(a, b)
        assert result["class"] == "sampler"
        assert a[result["first_difference"]]["op"] == "sample"
        assert "seed" in result["detail"]

    def test_truncated_when_one_is_a_prefix(self) -> None:
        a = run()
        samples = [i for i, r in enumerate(a) if r["op"] == "sample"]
        cut = samples[-2] + 1          # a proper prefix ending on a sample (nothing pending)
        result = diff_logs(a, a[:cut])
        assert result["class"] == "truncated"
        assert result["first_difference"] == cut
        assert result["records_a"] == len(a) and result["records_b"] == cut

    def test_data_when_op_sequences_differ(self) -> None:
        # One run samples where the other inserts: the sequence of buffer
        # operations differs, which upstream data (not the sampler) decides.
        a = run()
        buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=2, seed=0, attest=AttestationLog())
        buf.add_group("g0", 0, rollouts([1.0, 0.0, 0.5]), source="a")
        buf.add_group("h0", 0, rollouts([1.0]), source="a")
        result = diff_logs(a, buf.attestation_log.records)
        assert result["class"] == "data"
        assert result["ops"] == ["sample", "insert"]

    def test_config_when_a_decayed_log_meets_a_legacy_log(self) -> None:
        legacy = AttestationLog()
        legacy.append_mutation("insert", 0, 0, 5, 0)
        result = diff_logs(run(), legacy.records, capacity=8)
        assert result["class"] == "config" and result["first_difference"] == 0

    def test_empty_logs(self) -> None:
        assert diff_logs([], [])["identical"]
        result = diff_logs([], run())
        assert result["class"] == "truncated" and result["ops"] == [None, "decay_config"]
        assert result["record_b"]["op"] == "decay_config" and result["record_a"] is None

    @pytest.mark.parametrize("a, b, expected", [
        ({"op": "evict", "index": 1}, {"op": "evict", "index": 2}, "internal"),
        ({"op": "rebase"}, {"op": "insert"}, "internal"),
        ({"op": "evict"}, {"op": "insert"}, "data"),
        ({"op": "sample", "samples": [{}]}, {"op": "sample", "samples": [{}, {}]}, "data"),
        ({"op": "sample", "samples": [{"leaf_index": 1, "draw_int": "5", "is_weight_num": "1"}]},
         {"op": "sample", "samples": [{"leaf_index": 1, "draw_int": "5", "is_weight_num": "2"}]}, "config"),
        ({"op": "sample", "samples": [{"leaf_index": 1, "draw_int": "5"}]},
         {"op": "sample", "samples": [{"leaf_index": 2, "draw_int": "9"}]}, "sampler"),
        ({"op": "update"}, {"op": "update"}, "data"),
    ])
    def test_classify_table(self, a, b, expected) -> None:
        assert _classify(a, b) == expected

    def test_both_logs_must_verify(self) -> None:
        a = run()
        broken = [dict(r) for r in a]
        broken[3]["index"] = 7
        with pytest.raises(CheckerError):
            diff_logs(a, broken)


class TestCli:
    def test_exit_codes(self, tmp_path) -> None:
        a, b, c = run(), run(), run(rewards=(0.0, 1.0, 0.5))
        pa, pb, pc = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
        pa.write_text(to_lines(a)); pb.write_text(to_lines(b)); pc.write_text(to_lines(c))
        assert main([str(pa), str(pb)]) == 0
        assert main([str(pa), str(pc)]) == 3
        pc.write_text(to_lines(c)[:-40])
        assert main([str(pa), str(pc)]) == 1

    def test_checker_error_exits_1(self, tmp_path) -> None:
        a = run()
        broken = [dict(r) for r in a]
        broken[3]["index"] = 7
        pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        pa.write_text(to_lines(a)); pb.write_text(to_lines(broken))
        assert main([str(pa), str(pb)]) == 1

    def test_allow_truncated_flag(self, tmp_path) -> None:
        a = run()
        cut = next(i for i, r in enumerate(a) if r["op"] == "advance_version"
                   and a[i + 1]["op"] == "evict" and a[i + 1]["reason"] == "stale") + 1   # evictions pending
        pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        pa.write_text(to_lines(a)); pb.write_text(to_lines(a[:cut]))
        assert main([str(pa), str(pb)]) == 1
        assert main([str(pa), str(pb), "--allow-truncated"]) == 3

    def test_json_output(self, tmp_path) -> None:
        a, c = run(), run(rewards=(0.0, 1.0, 0.5))
        pa, pc, out = tmp_path / "a.jsonl", tmp_path / "c.jsonl", tmp_path / "diff.json"
        pa.write_text(to_lines(a)); pc.write_text(to_lines(c))
        assert main([str(pa), str(pc), "--json", str(out)]) == 3
        report = json.loads(out.read_text())
        assert report["class"] == "data" and report["record_a"]["op"] == "insert"

    def test_subprocess_identical(self, tmp_path) -> None:
        a = run()
        pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        pa.write_text(to_lines(a)); pb.write_text(to_lines(a))
        proc = subprocess.run([sys.executable, "-m", "checker.diff", str(pa), str(pb)],
                              capture_output=True, text=True, cwd=REPO)
        assert proc.returncode == 0, proc.stderr
        assert "identical" in proc.stdout
