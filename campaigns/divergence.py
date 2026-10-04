"""
campaigns.divergence — Preregistered T2 float-divergence campaign.

Compares exact and float sum-tree implementations across the preregistered
search grid (defined in docs/preregistration.md).

Finds decision-relevant divergences (same draw integer -> different index)
and computes total-variation distance between exact and float distributions.

Run with: python -m campaigns.divergence

The comparison is apples-to-apples:
  - Exact buffer: prefix_sum_locate(draw_int) with integer tree
  - Float buffer: sample(float_draw) where float_draw is the rational scaling
    of draw_int into [0, float_total):
      float_draw = float(Fraction(draw_int, exact_total) * Fraction(float_total))

Protocol, as preregistered: capacities 2^10, 2^14, 2^17; magnitude spans
10^3, 10^8, 10^12; update:sample ratios 1:1, 10:1, 100:1; 10,000 workloads
per cell: 270,000 workload programs, each run against both float idioms,
which the preregistration counts as 540,000; 100 draws per workload checked for a
decision-relevant divergence; total-variation distance between the exact
distribution and each float tree's implied distribution over all N
positions (the distribution does not change between the 100 draws of one
workload, so it is computed once per workload; the per-cell value is the
maximum over its workloads).

Workload programs follow the preregistered structure: ``capacity``
priorities ``p_i = base * magnitude_factor ** (i mod 8)`` with ``base`` in
[1, 10] and ``magnitude_factor`` chosen so the eight levels span the
cell's magnitude range, then ``10 * updates_per_sample`` keyed priority
updates, then 100 draws, each exactly uniform on ``[0, exact_total)``
through the library's keyed rejection-sampling draw (a 64-bit hash reduced
modulo a 70- to 90-bit total, as the campaign did before 2026-10-04, would
only ever reach the first leaves). A divergence, if one is found, is
written to ``results/divergence_reproducers/`` as the full workload
parameters; the preregistered shrinking step is not implemented and would
be run by hand on such a reproducer.

``--workloads-per-cell`` runs a disclosed fraction of the grid. The report
records the count run, and the preregistered kill rule is declared fired
only when the full 10,000 per cell were run; any smaller run says "not
falsified in N of 540,000 workloads" instead. See
docs/preregistration-deviations.md for the history of this campaign.

Usage::

    python -m campaigns.divergence                              # full grid, ~28 CPU-hours
    python -m campaigns.divergence --workloads-per-cell 200     # 1% sample, ~20 min on 8 cores
    python -m campaigns.divergence --workloads-per-cell 5 --jobs 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaigns.float_baselines import LabmlArraySumTree, OpenAISegmentTree
from src.reservoir.draw import draw_uniform_below
from src.reservoir.rational import float_to_priority_int
from src.reservoir.sumtree import ExactSumTree

# ---------------------------------------------------------------------------
# Search grid (frozen in preregistration.md)
# ---------------------------------------------------------------------------

CAPACITIES = [1 << 10, 1 << 14, 1 << 17]
MAGNITUDE_SPANS = [1e3, 1e8, 1e12]
UPDATE_SAMPLE_RATIOS = [(1, 1), (10, 1), (100, 1)]
PREREGISTERED_WORKLOADS_PER_CELL = 10_000
ALPHA = 0.6

# Draws per workload checked for a decision-relevant divergence.
TV_SAMPLE_COUNT = 100
TV_KILL_THRESHOLD = 2 ** (-40)


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------

def _keyed_hash(seed: int, i: int) -> int:
    """Deterministic hash for workload generation."""
    msg = seed.to_bytes(8, "big") + i.to_bytes(8, "big")
    return int.from_bytes(hashlib.blake2b(msg, digest_size=8).digest(), "big")


SPAN_LEVELS = 8
"""Preregistered: priorities cycle through eight magnitude levels."""


def generate_priorities(capacity: int, magnitude_span: float, seed: int) -> list[float]:
    """The preregistered workload: ``p_i = base * magnitude_factor ** (i mod 8)``.

    ``base`` is drawn from [1, 10] by the keyed hash; ``magnitude_factor`` is
    ``magnitude_span ** (1 / 7)`` so the eight levels run from ``base`` to
    ``base * magnitude_span``.
    """
    base = 1.0 + 9.0 * (_keyed_hash(seed, 0) % 1_000_000) / 1_000_000
    factor = magnitude_span ** (1.0 / (SPAN_LEVELS - 1))
    return [base * factor ** (i % SPAN_LEVELS) for i in range(capacity)]


# ---------------------------------------------------------------------------
# Apples-to-apples draw mapping
# ---------------------------------------------------------------------------

def _map_draw_to_float(draw_int: int, exact_total: int, float_total: float) -> float:
    """Map an exact integer draw to the float tree's scale.

    Exact draw: draw_int in [0, exact_total)
    Float draw: proportional value in [0, float_total)

    float_draw = float(Fraction(draw_int, exact_total) * Fraction(float_total))
    as preregistered: the scaling is exact and rounds once, at the end.
    """
    return float(Fraction(draw_int, exact_total) * Fraction(float_total))


# ---------------------------------------------------------------------------
# Single divergence check
# ---------------------------------------------------------------------------

def check_divergence(
    capacity: int,
    magnitude_span: float,
    update_sample_ratio: tuple[int, int],
    workload_seed: int,
    alpha: float = ALPHA,
) -> dict:
    """Run one workload program and check for decision-relevant divergences.

    Returns a dict with:
      - diverged: bool
      - n_samples: int
      - divergences: list of (sample_idx, exact_pos, openai_pos, labml_pos)
      - tv_openai: float (TV distance, OpenAI baseline)
      - tv_labml: float (TV distance, labml baseline)
    """
    updates_per_sample, _ = update_sample_ratio

    # Initialize trees
    exact_tree = ExactSumTree(capacity)
    openai_tree = OpenAISegmentTree(capacity, alpha=alpha)
    labml_tree = LabmlArraySumTree(capacity, alpha=alpha)

    # Insert initial priorities
    priorities_raw = generate_priorities(capacity, magnitude_span, seed=workload_seed)
    for i, p in enumerate(priorities_raw):
        p_int = float_to_priority_int(p, alpha)
        exact_tree.update(i, p_int)
        openai_tree.update(i, p)
        labml_tree.update(i, p)

    # Do some updates
    n_updates = min(updates_per_sample * 10, capacity)
    for j in range(n_updates):
        pos = _keyed_hash(workload_seed, 10000 + j) % capacity
        h = _keyed_hash(workload_seed, 20000 + j)
        new_p = priorities_raw[pos] * (0.5 + h % 100 / 50.0)
        p_int = float_to_priority_int(new_p, alpha)
        exact_tree.update(pos, p_int)
        openai_tree.update(pos, new_p)
        labml_tree.update(pos, new_p)

    divergences = []
    exact_total = exact_tree.total
    if exact_total == 0:
        return {"diverged": False, "n_samples": 0, "divergences": [], "tv_openai": 0.0, "tv_labml": 0.0}

    # Total-variation distance over all positions, once: nothing mutates
    # between the draws below, so every draw sees the same distribution.
    tv_openai_max = _total_variation(exact_tree, exact_total, openai_tree, capacity)
    tv_labml_max = _total_variation(exact_tree, exact_total, labml_tree, capacity)

    # Draws: a decision-relevant divergence is a different index for the same draw.
    n_samples = TV_SAMPLE_COUNT
    for k in range(n_samples):
        # Exactly uniform on [0, exact_total): the library's keyed draw, keyed
        # by the workload seed and the draw index.
        draw_int = draw_uniform_below(exact_total, seed=workload_seed % (1 << 64), buffer_id=0, op_counter=k)
        exact_pos = exact_tree.prefix_sum_locate(draw_int)
        divergence = _compare_draw(k, draw_int, exact_total, exact_pos, openai_tree, labml_tree)
        if divergence is not None:
            divergences.append(divergence)

    return {
        "diverged": len(divergences) > 0,
        "n_samples": n_samples,
        "divergences": divergences,
        "tv_openai": tv_openai_max,
        "tv_labml": tv_labml_max,
    }


# ---------------------------------------------------------------------------
# Full campaign
# ---------------------------------------------------------------------------

def _compare_draw(k: int, draw_int: int, exact_total: int, exact_pos: int, openai_tree, labml_tree) -> dict | None:
    """Map one exact draw onto each float tree and compare the located leaves.

    The rational mapping rounds once; the result can equal ``float_total``
    exactly, which the float trees' ``sample`` does not accept, so it is
    clamped to just below the total. That clamp is a no-op for every other
    value.
    """
    openai_float = _map_draw_to_float(draw_int, exact_total, openai_tree.total)
    labml_float = _map_draw_to_float(draw_int, exact_total, labml_tree.total)
    openai_float = max(0.0, min(openai_float, openai_tree.total - 1e-15))
    labml_float = max(0.0, min(labml_float, labml_tree.total - 1e-15))
    openai_pos = openai_tree.sample(openai_float)
    labml_pos = labml_tree.sample(labml_float)
    if openai_pos == exact_pos and labml_pos == exact_pos:
        return None
    return {"sample_idx": k, "draw_int": draw_int, "exact_pos": exact_pos,
            "openai_pos": openai_pos, "labml_pos": labml_pos}


def _total_variation(exact_tree: ExactSumTree, exact_total: int, float_tree, capacity: int) -> float:
    """``0.5 * sum_i |p_exact_i - p_float_i|`` over every position.

    The exact probabilities are rationals; the float tree's implied
    probabilities are ``leaf_i / total`` in float64. The difference is
    reported in float64, which resolves the 2^-40 threshold with 20 bits
    to spare.
    """
    float_total = float_tree.total
    if float_total <= 0:
        return 0.0
    total = 0.0
    for i in range(capacity):
        p_exact = Fraction(exact_tree.get(i), exact_total)
        p_float = float_tree.get_leaf(i) / float_total
        total += abs(float(p_exact - Fraction(p_float)))
    return 0.5 * total


def _write_reproducer(cap: int, mag_span: float, update_ratio, seed: int, divergences: list[dict]) -> None:
    """Record a divergent workload's parameters (preregistered §Minimal Reproducer, without shrinking)."""
    out = Path("results/divergence_reproducers")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"cap{cap}_span{int(mag_span)}_ratio{update_ratio[0]}_{update_ratio[1]}_seed{seed}.json").write_text(
        json.dumps({"capacity": cap, "magnitude_span": mag_span, "update_ratio": list(update_ratio),
                    "workload_seed": seed, "divergences": divergences, "shrunk": False}, indent=2)
    )


def _run_cell(args: tuple) -> dict:
    """One grid cell: ``workloads_per_cell`` workloads, run in a worker process."""
    cap, mag_span, update_ratio, workloads_per_cell = args
    cell = {
        "capacity": cap, "magnitude_span": mag_span, "update_ratio": list(update_ratio),
        "n_workloads": workloads_per_cell, "divergences": 0,
        "tv_max_openai": 0.0, "tv_max_labml": 0.0, "examples": [],
    }
    for w in range(workloads_per_cell):
        seed = _keyed_hash(cap + int(mag_span), w)
        r = check_divergence(capacity=cap, magnitude_span=mag_span, update_sample_ratio=update_ratio,
                             workload_seed=seed)
        if r["diverged"]:
            cell["divergences"] += len(r["divergences"])
            if len(cell["examples"]) < 3:
                cell["examples"].append(r["divergences"][0])
            _write_reproducer(cap, mag_span, update_ratio, seed, r["divergences"])
        cell["tv_max_openai"] = max(cell["tv_max_openai"], r["tv_openai"])
        cell["tv_max_labml"] = max(cell["tv_max_labml"], r["tv_labml"])
    return cell


def run_campaign(workloads_per_cell: int = PREREGISTERED_WORKLOADS_PER_CELL, jobs: int = 1) -> dict:
    """Run the T2 grid with ``workloads_per_cell`` workloads in every cell, ``jobs`` processes."""
    cells_spec = [(cap, span, ratio, workloads_per_cell)
                  for cap in CAPACITIES for span in MAGNITUDE_SPANS for ratio in UPDATE_SAMPLE_RATIOS]
    if jobs > 1:
        with multiprocessing.get_context("spawn").Pool(jobs) as pool:
            cells = pool.map(_run_cell, cells_spec)
    else:
        cells = [_run_cell(spec) for spec in cells_spec]

    results = {
        "workloads_per_cell": workloads_per_cell,
        "preregistered_workloads_per_cell": PREREGISTERED_WORKLOADS_PER_CELL,
        "total_workloads": sum(c["n_workloads"] for c in cells),
        "preregistered_total_workloads": PREREGISTERED_WORKLOADS_PER_CELL * len(cells_spec),
        "idiom_evaluations": 2 * sum(c["n_workloads"] for c in cells),
        "preregistered_idiom_evaluations": 2 * PREREGISTERED_WORKLOADS_PER_CELL * len(cells_spec),
        "total_divergences": sum(c["divergences"] for c in cells),
        "tv_max_openai": max(c["tv_max_openai"] for c in cells),
        "tv_max_labml": max(c["tv_max_labml"] for c in cells),
        "cells": cells,
    }
    no_divergences = results["total_divergences"] == 0
    tv_below = results["tv_max_openai"] < TV_KILL_THRESHOLD and results["tv_max_labml"] < TV_KILL_THRESHOLD
    results["kill_conditions_met_in_run"] = no_divergences and tv_below
    results["full_grid"] = workloads_per_cell >= PREREGISTERED_WORKLOADS_PER_CELL
    # The preregistered verdict exists only for the preregistered grid.
    results["kill_rule_fired"] = results["kill_conditions_met_in_run"] and results["full_grid"]
    return results


def _verdict(results: dict) -> str:
    if results["kill_rule_fired"]:
        return "DEAD"
    if results["kill_conditions_met_in_run"]:
        return "NOT_FALSIFIED_IN_REDUCED_RUN"
    return "ALIVE"


def _print_results(results: dict) -> None:
    print(f"Total workloads run: {results['total_workloads']}")
    print(f"Total divergences:   {results['total_divergences']}")
    print(f"Max TV (OpenAI):     {results['tv_max_openai']:.2e}")
    print(f"Max TV (labml):      {results['tv_max_labml']:.2e}")
    print(f"TV kill threshold:   {TV_KILL_THRESHOLD:.2e} (= 2^-40)")
    print()
    verdict = _verdict(results)
    if verdict == "DEAD":
        print("KILL RULE FIRED: T2 is DEAD (thesis falsified on the full preregistered grid)")
        print("  Zero divergences AND TV distances below 2^-40 in all cells")
    elif verdict == "NOT_FALSIFIED_IN_REDUCED_RUN":
        print(f"T2 NOT FALSIFIED in {results['total_workloads']} of {results['preregistered_total_workloads']} "
              f"preregistered workload programs ({results['idiom_evaluations']} of "
              f"{results['preregistered_idiom_evaluations']} idiom evaluations): zero divergences and TV below "
              "2^-40, but the preregistered kill rule needs the full grid, so no verdict is declared")
    else:
        print("T2 is ALIVE")
        if results["total_divergences"] > 0:
            print(f"  Decision-relevant divergences found: {results['total_divergences']}; "
                  "see results/divergence_reproducers/")
        if results["tv_max_openai"] >= TV_KILL_THRESHOLD or results["tv_max_labml"] >= TV_KILL_THRESHOLD:
            print("  TV distance >= kill threshold in some cell")


def _report(results: dict) -> dict:
    return {
        "campaign": "T2_divergence",
        "preregistration": "docs/preregistration.md",
        "deviations": "docs/preregistration-deviations.md",
        "workloads_per_cell": results["workloads_per_cell"],
        "preregistered_workloads_per_cell": results["preregistered_workloads_per_cell"],
        "total_workloads": results["total_workloads"],
        "preregistered_total_workloads": results["preregistered_total_workloads"],
        "idiom_evaluations": results["idiom_evaluations"],
        "preregistered_idiom_evaluations": results["preregistered_idiom_evaluations"],
        "full_grid": results["full_grid"],
        "total_divergences": results["total_divergences"],
        "tv_max_openai": results["tv_max_openai"],
        "tv_max_labml": results["tv_max_labml"],
        "tv_kill_threshold": TV_KILL_THRESHOLD,
        "kill_conditions_met_in_run": results["kill_conditions_met_in_run"],
        "kill_rule_fired": results["kill_rule_fired"],
        "thesis_T2": _verdict(results),
        "cells": results["cells"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preregistered T2 float-divergence campaign.")
    parser.add_argument("--workloads-per-cell", type=int, default=PREREGISTERED_WORKLOADS_PER_CELL,
                        help="workloads per grid cell; the preregistered value is 10,000")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="worker processes (cells run in parallel)")
    args = parser.parse_args(argv)

    n_cells = len(CAPACITIES) * len(MAGNITUDE_SPANS) * len(UPDATE_SAMPLE_RATIOS)
    print("=" * 70)
    print("T2 Float-Divergence Campaign")
    print("=" * 70)
    print("Preregistration: docs/preregistration.md")
    print(f"Workloads per cell: {args.workloads_per_cell} (preregistered: {PREREGISTERED_WORKLOADS_PER_CELL})")
    print(f"Grid: {len(CAPACITIES)} capacities × {len(MAGNITUDE_SPANS)} spans × {len(UPDATE_SAMPLE_RATIOS)} ratios")
    print(f"Total workloads: {n_cells * args.workloads_per_cell} of {n_cells * PREREGISTERED_WORKLOADS_PER_CELL}")
    print()

    results = run_campaign(args.workloads_per_cell, args.jobs)
    _print_results(results)
    Path("results").mkdir(exist_ok=True)
    Path("results/divergence_campaign_report.json").write_text(json.dumps(_report(results), indent=2, default=str))
    print()
    print("Report written to results/divergence_campaign_report.json")


if __name__ == "__main__":
    main()
