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
