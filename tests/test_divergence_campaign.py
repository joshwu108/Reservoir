"""Tests pinning the T2 divergence campaign's protocol to what docs/preregistration.md
and docs/preregistration-deviations.md say it is.

The grid, the workload priorities, the draw mapping, the total-variation
distance and the verdict rule follow the preregistration. The two
disclosed deviations are pinned as they are, so a change to either shows
up here and in the deviations file together: updates per workload are
``min(10 * r, capacity)``, all before the 100 draws, and a workload's seed
ignores the update ratio. The power estimate (``cdf_mismatch_mass``) is
checked against a tree whose divergence region is known.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from campaigns import divergence as dv
from campaigns.float_baselines import LabmlArraySumTree, OpenAISegmentTree
from src.reservoir.sumtree import ExactSumTree

REPO = Path(__file__).resolve().parents[1]


def test_grid_is_the_preregistered_one():
    assert dv.CAPACITIES == [1024, 16384, 131072]
    assert dv.MAGNITUDE_SPANS == [1e3, 1e8, 1e12]
    assert dv.UPDATE_SAMPLE_RATIOS == [(1, 1), (10, 1), (100, 1)]
    assert dv.PREREGISTERED_WORKLOADS_PER_CELL == 10_000
    assert dv.TV_SAMPLE_COUNT == 100
    assert dv.TV_KILL_THRESHOLD == 2.0 ** -40
    assert dv.SPAN_LEVELS == 8


@pytest.mark.parametrize("span", [1e3, 1e8, 1e12])
def test_priorities_cycle_eight_levels_from_base_to_base_times_span(span):
    p = dv.generate_priorities(64, span, seed=7)
    base = p[0]
    assert 1.0 <= base <= 10.0
    assert p[:8] == p[8:16] == p[56:64]                          # i mod 8
    ratios = [p[i + 1] / p[i] for i in range(7)]
    assert all(r == pytest.approx(span ** (1 / 7)) for r in ratios)
    assert p[7] == pytest.approx(base * span)


def test_deviation_updates_are_ten_per_ratio_unit_all_before_the_draws():
    # Labelled 1:1, 10:1, 100:1; run as 10, 100, 1000 updates before 100 draws: 0.1, 1 and 10 per draw.
    per_draw = [dv.updates_per_workload(r, 1 << 17) / dv.TV_SAMPLE_COUNT for r in dv.UPDATE_SAMPLE_RATIOS]
    assert per_draw == [0.1, 1.0, 10.0]
    assert dv.updates_per_workload((100, 1), 512) == 512           # capped by capacity


def test_deviation_updates_happen_before_any_draw(monkeypatch):
    events = []
    real_update = ExactSumTree.update
    real_locate = ExactSumTree.prefix_sum_locate
    monkeypatch.setattr(ExactSumTree, "update", lambda self, i, p: (events.append("u"), real_update(self, i, p))[1])
    monkeypatch.setattr(ExactSumTree, "prefix_sum_locate", lambda self, d: (events.append("d"), real_locate(self, d))[1])
    dv.check_divergence(1024, 1e3, (10, 1), workload_seed=5)
    assert events.count("d") == dv.TV_SAMPLE_COUNT
    assert events.count("u") == 1024 + 100                         # inserts, then 10 * 10 updates
    assert "u" not in events[events.index("d"):]


def test_deviation_seed_ignores_the_update_ratio():
    # The three ratio cells of a (capacity, span) pair start from the same priorities.
    seeds = {dv.workload_seed(1024, 1e8, w) for w in range(3)}
    assert len(seeds) == 3
    assert dv.workload_seed(1024, 1e8, 0) == dv._keyed_hash(1024 + int(1e8), 0)
    a = dv._build_trees(1024, 1e8, (1, 1), dv.workload_seed(1024, 1e8, 0))[0]
    b = dv._build_trees(1024, 1e8, (100, 1), dv.workload_seed(1024, 1e8, 0))[0]
    unchanged = [i for i in range(1024) if a.get(i) == b.get(i)]
    assert len(unchanged) > 1024 - 1000                            # only the 1000 - 10 extra updates differ


def test_draw_mapping_is_exact_rational_scaling_rounded_once():
    total, float_total = 3 * 2**80 + 1, 12345.678
    for d in (0, 1, total // 3, total - 1):
        assert dv._map_draw_to_float(d, total, float_total) == float(Fraction(d, total) * Fraction(float_total))


def test_total_variation_is_half_the_l1_distance_over_all_positions():
    exact = ExactSumTree(4)
    for i, v in enumerate([1, 1, 1, 1]):
        exact.update(i, v)
    flt = OpenAISegmentTree(4, alpha=1.0)
    for i, v in enumerate([1.5, 0.5, 1.0, 1.0]):
        flt.update(i, v)
    assert dv._total_variation(exact, exact.total, flt, 4) == pytest.approx(0.5 * (0.125 + 0.125))


def test_cdf_mismatch_mass_is_the_share_of_draws_that_diverge():
    # Exact leaves equal; float leaves [1.5, 0.5, 1, 1]: the boundary after leaf 0 moves from 1/4 to 3/8,
    # so draws in [1/4, 3/8) land in leaf 1 exactly and in leaf 0 in float: one eighth of all draws.
    unit = 1 << 20
    exact = ExactSumTree(4)
    for i in range(4):
        exact.update(i, unit)
    openai, labml = OpenAISegmentTree(4, alpha=1.0), LabmlArraySumTree(4, alpha=1.0)
    for tree in (openai, labml):
        for i, v in enumerate([1.5, 0.5, 1.0, 1.0]):
            tree.update(i, v)
    assert dv.cdf_mismatch_mass(exact, exact.total, openai, 4) == 0.125
    step = 1024
    draws = range(0, exact.total, step)
    diverged = sum(dv._compare_draw(0, d, exact.total, exact.prefix_sum_locate(d), openai, labml) is not None
                   for d in draws)
    assert diverged / len(draws) == 0.125


def test_cdf_mismatch_mass_is_zero_when_the_trees_agree():
    exact = ExactSumTree(8)
    flt = OpenAISegmentTree(8, alpha=1.0)
    for i in range(8):
        exact.update(i, 1 << (i + 3))
        flt.update(i, float(1 << (i + 3)))
    assert dv.cdf_mismatch_mass(exact, exact.total, flt, 8) == 0.0


def test_verdict_needs_the_full_grid():
    reduced = {"kill_conditions_met_in_run": True, "kill_rule_fired": False, "total_divergences": 0,
               "tv_max_openai": 0.0, "tv_max_labml": 0.0}
    assert dv._verdict(reduced) == "NOT_FALSIFIED_IN_REDUCED_RUN"
    assert dv._verdict({**reduced, "kill_rule_fired": True}) == "DEAD"
    assert dv._verdict({**reduced, "kill_conditions_met_in_run": False}) == "ALIVE"


def test_a_reduced_run_declares_no_verdict(monkeypatch):
    monkeypatch.setattr(dv, "CAPACITIES", [64])
    monkeypatch.setattr(dv, "MAGNITUDE_SPANS", [1e3])
    monkeypatch.setattr(dv, "UPDATE_SAMPLE_RATIOS", [(1, 1)])
    results = dv.run_campaign(workloads_per_cell=2, jobs=1)
    assert results["total_workloads"] == 2 and results["idiom_evaluations"] == 4
    assert results["full_grid"] is False and results["kill_rule_fired"] is False
    assert results["total_divergences"] == 0 and results["kill_conditions_met_in_run"] is True


def test_committed_report_matches_the_documented_reduced_run():
    report = json.loads((REPO / "results" / "divergence_campaign_report.json").read_text())
    assert report["workloads_per_cell"] == 200 and report["total_workloads"] == 5_400
    assert report["preregistered_total_workloads"] == 270_000 and report["full_grid"] is False
    assert report["thesis_T2"] == "NOT_FALSIFIED_IN_REDUCED_RUN" and report["total_divergences"] == 0
    assert max(report["tv_max_openai"], report["tv_max_labml"]) < report["tv_kill_threshold"] == 2.0 ** -40
    assert len(report["cells"]) == 27


def test_committed_power_report_backs_the_deviations_file():
    power = json.loads((REPO / "results" / "divergence_power.json").read_text())
    assert power["provenance"]["commit"]
    expected = power["expected_divergences"]
    # The full preregistered grid expects far fewer than one divergence, for both idioms.
    assert all(v < 1e-3 for v in expected["preregistered_grid"].values())
    assert all(v < 1e-5 for v in expected["committed_run_200_per_cell"].values())
    by_cap = power["mismatch_per_draw_max_by_capacity"]
    assert set(by_cap) == {"1024", "16384", "131072"}
    text = (REPO / "docs" / "preregistration-deviations.md").read_text()
    assert f"{expected['preregistered_grid']['openai']:.1e}" in text
    assert f"{expected['committed_run_200_per_cell']['openai']:.1e}" in text
    for cap in by_cap:
        assert f"{by_cap[cap]['openai']:.1e}" in text
    assert math.isclose(expected["preregistered_grid"]["openai"],
                        expected["committed_run_200_per_cell"]["openai"] * 10_000 / 200)
