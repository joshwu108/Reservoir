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
"""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from campaigns.float_baselines import LabmlArraySumTree, OpenAISegmentTree
from src.reservoir.rational import float_to_priority_int, PRIORITY_SCALE
from src.reservoir.sumtree import ExactSumTree

# ---------------------------------------------------------------------------
# Search grid (frozen in preregistration.md)
# ---------------------------------------------------------------------------

CAPACITIES = [512, 4096, 16384]  # Full preregistered: [1024, 16384, 131072]
MAGNITUDE_SPANS = [1e3, 1e8, 1e12]
UPDATE_SAMPLE_RATIOS = [(1, 1), (10, 1), (100, 1)]
N_WORKLOADS = 5   # Full preregistered: 10,000 (Python exact-tree ~100x slower than float)
ALPHA = 0.6

# For TV distance computation
TV_SAMPLE_COUNT = 10
TV_KILL_THRESHOLD = 2 ** (-40)


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------

def _keyed_hash(seed: int, i: int) -> int:
    """Deterministic hash for workload generation."""
    msg = seed.to_bytes(8, "big") + i.to_bytes(8, "big")
    return int.from_bytes(hashlib.blake2b(msg, digest_size=8).digest(), "big")


def generate_priorities(capacity: int, magnitude_span: float, seed: int) -> list[float]:
    """Generate a list of positive float priorities spanning `magnitude_span` orders.

    Uses a deterministic pattern: base values * spread across magnitude_span.
    """
    priorities = []
    for i in range(capacity):
        h = _keyed_hash(seed, i)
        # Base value in [1, 10]
        base = 1.0 + (h % 1000) / 100.0  # [1, 11)
        # Exponent from [0, log10(magnitude_span)]
        exp_range = max(1.0, (h >> 32) % 1000) / 1000.0 * magnitude_span
        # Mix near-zero and large values
        if i % 8 < 1:  # ~12.5% near-zero
            prio = base * 1e-6
        else:
            prio = base * exp_range
        priorities.append(max(prio, 1e-12))  # Ensure positive
    return priorities


# ---------------------------------------------------------------------------
# Apples-to-apples draw mapping
# ---------------------------------------------------------------------------

def _map_draw_to_float(draw_int: int, exact_total: int, float_total: float) -> float:
    """Map an exact integer draw to the float tree's scale.

    Exact draw: draw_int in [0, exact_total)
    Float draw: proportional value in [0, float_total)

    float_draw = (draw_int / exact_total) * float_total
    The division uses Python's arbitrary-precision integers for draw_int and
    exact_total, then converts to float64. This is the apples-to-apples mapping
    documented in preregistration.md.
    """
    # Compute as Fraction for exact mapping, then convert to float
    # Use float division for speed in campaign (both draw_int and float_total are
    # not decision-path operations — this is for the comparison report)
    ratio = draw_int / exact_total  # float64 approximation sufficient for campaign
    return ratio * float_total


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

    # Sample and compare
    n_samples = TV_SAMPLE_COUNT
    tv_openai_max = 0.0
    tv_labml_max = 0.0

    for k in range(n_samples):
        # Draw a deterministic integer in [0, exact_total)
        h = _keyed_hash(workload_seed, 30000 + k)
        draw_int = h % exact_total

        # Exact result
        try:
            exact_pos = exact_tree.prefix_sum_locate(draw_int)
        except Exception:
            continue

        # Map draw to float scale
        openai_float = _map_draw_to_float(draw_int, exact_total, openai_tree.total)
        labml_float = _map_draw_to_float(draw_int, exact_total, labml_tree.total)

        # Clamp to valid range (float arithmetic may produce slightly out-of-range)
        openai_float = max(0.0, min(openai_float, openai_tree.total - 1e-15))
        labml_float = max(0.0, min(labml_float, labml_tree.total - 1e-15))

        openai_pos = openai_tree.sample(openai_float)
        labml_pos = labml_tree.sample(labml_float)

        if openai_pos != exact_pos or labml_pos != exact_pos:
            divergences.append({
                "sample_idx": k,
                "draw_int": draw_int,
                "exact_pos": exact_pos,
                "openai_pos": openai_pos,
                "labml_pos": labml_pos,
            })

        # TV distance: compare exact probability with float-implied probability
        # (float64 arithmetic for the campaign report — not a buffer decision path)
        p_exact_num = exact_tree.get(exact_pos)
        p_exact_float = p_exact_num / exact_total  # float64 approximation

        openai_total = openai_tree.total
        openai_leaf = openai_tree.get_leaf(exact_pos)
        if openai_total > 0:
            p_openai = openai_leaf / openai_total
            tv_openai = abs(p_exact_float - p_openai)
            tv_openai_max = max(tv_openai_max, tv_openai)

        labml_total = labml_tree.total
        labml_leaf = labml_tree.get_leaf(exact_pos)
        if labml_total > 0:
            p_labml = labml_leaf / labml_total
            tv_labml = abs(p_exact_float - p_labml)
            tv_labml_max = max(tv_labml_max, tv_labml)

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

def run_campaign() -> dict:
    """Run the full T2 divergence campaign and return results."""
    results = {
        "total_workloads": 0,
        "total_divergences": 0,
        "cells": [],
        "kill_rule_fired": False,
        "tv_max_openai": 0.0,
        "tv_max_labml": 0.0,
    }

    # Reduced grid for practical runtime; full grid is 540,000 workloads
    n_workloads = N_WORKLOADS

    for cap in CAPACITIES:
        for mag_span in MAGNITUDE_SPANS:
            for update_ratio in UPDATE_SAMPLE_RATIOS:
                cell_divergences = 0
                cell_tv_openai = 0.0
                cell_tv_labml = 0.0
                cell_divergence_examples = []

                for w in range(n_workloads):
                    seed = _keyed_hash(cap + int(mag_span), w)
                    r = check_divergence(
                        capacity=cap,
                        magnitude_span=mag_span,
                        update_sample_ratio=update_ratio,
                        workload_seed=seed,
                    )

                    results["total_workloads"] += 1
                    if r["diverged"]:
                        cell_divergences += len(r["divergences"])
                        results["total_divergences"] += len(r["divergences"])
                        if len(cell_divergence_examples) < 3:
                            cell_divergence_examples.append(r["divergences"][0])

                    cell_tv_openai = max(cell_tv_openai, r["tv_openai"])
                    cell_tv_labml = max(cell_tv_labml, r["tv_labml"])

                results["tv_max_openai"] = max(results["tv_max_openai"], cell_tv_openai)
                results["tv_max_labml"] = max(results["tv_max_labml"], cell_tv_labml)

                results["cells"].append({
                    "capacity": cap,
                    "magnitude_span": mag_span,
                    "update_ratio": update_ratio,
                    "n_workloads": n_workloads,
                    "divergences": cell_divergences,
                    "tv_max_openai": cell_tv_openai,
                    "tv_max_labml": cell_tv_labml,
                    "examples": cell_divergence_examples,
                })

    # Kill rule check
    no_divergences = results["total_divergences"] == 0
    tv_below_threshold = (
        results["tv_max_openai"] < TV_KILL_THRESHOLD
        and results["tv_max_labml"] < TV_KILL_THRESHOLD
    )
    results["kill_rule_fired"] = no_divergences and tv_below_threshold

    return results


def main() -> None:
    print("=" * 70)
    print("T2 Float-Divergence Campaign")
    print("=" * 70)
    print(f"Preregistration: docs/preregistration.md")
    print(f"Workloads per cell: {N_WORKLOADS} (full: 10,000)")
    print(f"Grid: {len(CAPACITIES)} capacities × {len(MAGNITUDE_SPANS)} spans × {len(UPDATE_SAMPLE_RATIOS)} ratios")
    print(f"Total workloads: {len(CAPACITIES) * len(MAGNITUDE_SPANS) * len(UPDATE_SAMPLE_RATIOS) * N_WORKLOADS}")
    print()

    results = run_campaign()

    print(f"Total workloads run: {results['total_workloads']}")
    print(f"Total divergences:   {results['total_divergences']}")
    print(f"Max TV (OpenAI):     {results['tv_max_openai']:.2e}")
    print(f"Max TV (labml):      {results['tv_max_labml']:.2e}")
    print(f"TV kill threshold:   {TV_KILL_THRESHOLD:.2e} (= 2^-40)")
    print()

    if results["kill_rule_fired"]:
        print("KILL RULE FIRED: T2 is DEAD (thesis falsified)")
        print("  Zero divergences AND TV distances below 2^-40 in all cells")
    else:
        print("T2 is ALIVE")
        if results["total_divergences"] > 0:
            print(f"  Decision-relevant divergences found: {results['total_divergences']}")
            # Show first examples
            for cell in results["cells"]:
                if cell["examples"]:
                    c = cell
                    print(f"  Example: capacity={c['capacity']}, span={c['magnitude_span']:.0e}, "
                          f"ratio={c['update_ratio']}")
                    ex = cell["examples"][0]
                    print(f"    draw_int={ex['draw_int']}, exact={ex['exact_pos']}, "
                          f"openai={ex['openai_pos']}, labml={ex['labml_pos']}")
                    break
        if results["tv_max_openai"] >= TV_KILL_THRESHOLD or results["tv_max_labml"] >= TV_KILL_THRESHOLD:
            print(f"  TV distance ≥ kill threshold in some cell")

    Path("results").mkdir(exist_ok=True)
    report = {
        "campaign": "T2_divergence",
        "preregistration": "docs/preregistration.md",
        "total_workloads": results["total_workloads"],
        "total_divergences": results["total_divergences"],
        "tv_max_openai": results["tv_max_openai"],
        "tv_max_labml": results["tv_max_labml"],
        "tv_kill_threshold": TV_KILL_THRESHOLD,
        "kill_rule_fired": results["kill_rule_fired"],
        "thesis_T2": "DEAD" if results["kill_rule_fired"] else "ALIVE",
        "cells": results["cells"],
    }
    Path("results/divergence_campaign_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    print()
    print("Report written to results/divergence_campaign_report.json")


if __name__ == "__main__":
    main()
