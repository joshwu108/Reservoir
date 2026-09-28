"""Tests for the signal-separation debug stats in benchmarks/modal/prefcheck_real.py.

The benchmark script lives outside the package tree (Modal mounts it as the
entrypoint file), so we load it by file path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).parents[1] / "benchmarks" / "modal" / "prefcheck_real.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("prefcheck_real", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSeparationStats:
    def test_perfect_separation_gives_auroc_one(self, bench):
        pos = [10.0, 11.0, 12.0]   # e.g. flipped pairs, high loss
        neg = [1.0, 2.0, 3.0]      # clean pairs, low loss
        stats = bench.separation_stats(pos, neg)
        assert stats["auroc"] == pytest.approx(1.0)

    def test_identical_distributions_give_auroc_half(self, bench):
        pos = [5.0, 5.0, 5.0]
        neg = [5.0, 5.0, 5.0]
        stats = bench.separation_stats(pos, neg)
        assert stats["auroc"] == pytest.approx(0.5)

    def test_reversed_separation_gives_auroc_zero(self, bench):
        # Memorization regime: flipped pairs end up with LOWER loss than clean
        pos = [0.001, 0.002]
        neg = [1.0, 2.0]
        stats = bench.separation_stats(pos, neg)
        assert stats["auroc"] == pytest.approx(0.0)

    def test_auroc_with_ties_across_groups(self, bench):
        pos = [1.0, 2.0]
        neg = [1.0, 0.0]
        # Pairwise: (1,1)=0.5, (1,0)=1, (2,1)=1, (2,0)=1 → 3.5/4
        stats = bench.separation_stats(pos, neg)
        assert stats["auroc"] == pytest.approx(3.5 / 4.0)

    def test_distribution_stats_present_for_both_groups(self, bench):
        pos = [4.0, 6.0]
        neg = [1.0, 3.0]
        stats = bench.separation_stats(pos, neg)
        assert stats["pos"]["mean"] == pytest.approx(5.0)
        assert stats["neg"]["mean"] == pytest.approx(2.0)
        assert stats["pos"]["median"] == pytest.approx(5.0)
        assert stats["neg"]["p25"] == pytest.approx(1.5)
        assert stats["pos"]["p75"] == pytest.approx(5.5)
        assert stats["pos"]["n"] == 2
        assert stats["neg"]["n"] == 2

    def test_empty_group_returns_none_auroc(self, bench):
        stats = bench.separation_stats([], [1.0, 2.0])
        assert stats["auroc"] is None
        assert stats["pos"]["n"] == 0

    def test_accepts_numpy_arrays(self, bench):
        stats = bench.separation_stats(np.array([2.0, 3.0]), np.array([0.0, 1.0]))
        assert stats["auroc"] == pytest.approx(1.0)


class TestDedupByPrompt:
    def test_keeps_first_occurrence_only(self, bench):
        prompts = ["a", "b", "a", "c", "b"]
        assert bench.dedup_by_prompt(prompts) == [0, 1, 3]

    def test_no_duplicates_keeps_everything(self, bench):
        assert bench.dedup_by_prompt(["x", "y", "z"]) == [0, 1, 2]

    def test_empty_list(self, bench):
        assert bench.dedup_by_prompt([]) == []


class TestEncodePairConsistent:
    @pytest.fixture(scope="class")
    def tok(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def test_short_inputs_fit_completely(self, bench, tok):
        c_ids, r_ids = bench.encode_pair_consistent(tok, "hello world", "yes", "no", 64)
        assert len(c_ids) <= 64 and len(r_ids) <= 64
        # Responses must survive: decode and check
        assert "yes" in tok.decode(c_ids)
        assert "no" in tok.decode(r_ids)

    def test_long_prompt_keeps_both_full_responses(self, bench, tok):
        prompt = "word " * 500  # far beyond max_len
        c_ids, r_ids = bench.encode_pair_consistent(
            tok, prompt, "the good answer", "a bad reply", 64
        )
        assert len(c_ids) <= 64 and len(r_ids) <= 64
        assert "the good answer" in tok.decode(c_ids)
        assert "a bad reply" in tok.decode(r_ids)

    def test_prompt_context_identical_across_pair(self, bench, tok):
        prompt = "word " * 500
        c_ids, r_ids = bench.encode_pair_consistent(tok, prompt, "aaa bbb", "ccc", 64)
        c_resp = tok("aaa bbb", add_special_tokens=False)["input_ids"]
        r_resp = tok("ccc", add_special_tokens=False)["input_ids"]
        # Strip [CLS], response tokens, [SEP]: what remains is the prompt tail
        c_prompt_part = c_ids[1:len(c_ids) - len(c_resp) - 1]
        r_prompt_part = r_ids[1:len(r_ids) - len(r_resp) - 1]
        assert c_prompt_part == r_prompt_part

    def test_identical_responses_give_identical_encodings(self, bench, tok):
        c_ids, r_ids = bench.encode_pair_consistent(tok, "prompt", "same", "same", 64)
        assert c_ids == r_ids

    def test_overlong_response_is_truncated_to_fit(self, bench, tok):
        resp = "token " * 500
        c_ids, r_ids = bench.encode_pair_consistent(tok, "p", resp, "short", 64)
        assert len(c_ids) <= 64 and len(r_ids) <= 64


class TestPadBatch:
    def test_pads_to_longest_and_masks(self, bench):
        ids, mask = bench.pad_batch([[5, 6, 7], [8]], pad_id=0)
        assert ids.shape == (2, 3)
        assert ids.tolist() == [[5, 6, 7], [8, 0, 0]]
        assert mask.tolist() == [[1, 1, 1], [1, 0, 0]]

    def test_equal_lengths_no_padding(self, bench):
        ids, mask = bench.pad_batch([[1, 2], [3, 4]], pad_id=0)
        assert ids.tolist() == [[1, 2], [3, 4]]
        assert mask.tolist() == [[1, 1], [1, 1]]
