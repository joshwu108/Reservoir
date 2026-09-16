"""benchmarks/prefcheck/synthetic_noise.py

Synthetic noise benchmark for reservoir-prefcheck.

Generates a dataset of preference pairs with known ground-truth noise labels
(FLIPPED, AMBIGUOUS, CLEAN) and evaluates how accurately PreferenceQualityReport
recovers them by simulating loss trajectories.

Usage
-----
    uv run python -m benchmarks.prefcheck.synthetic_noise

Output
------
Precision/recall/F1 for FLIPPED and AMBIGUOUS recovery, plus a summary table.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np

from reservoir.report import NoiseLabel, PreferenceQualityReport
from reservoir.trajectory import TrajectoryLogger


@dataclass
class SyntheticExample:
    """A synthetic preference pair with a known ground-truth noise type."""
    idx: int
    true_label: Literal["flipped", "ambiguous", "clean"]


def _simulate_trajectory(
    true_label: str,
    n_steps: int = 100,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, float]]:
    """Simulate a loss trajectory for a known label type."""
    if rng is None:
        rng = np.random.default_rng()

    if true_label == "flipped":
        # Loss flat or rising — model can't learn (label is wrong).
        # Slope must exceed 0.01/step to satisfy the FLIPPED rule.
        base = rng.uniform(0.6, 0.85)
        slope = rng.uniform(0.012, 0.025)  # > 0.01 threshold
        noise = rng.normal(0, 0.04, n_steps)
        losses = [float(np.clip(base + slope * t + noise[t], 0.01, 2.5)) for t in range(n_steps)]

    elif true_label == "ambiguous":
        # Loss oscillates with near-zero slope and high variance.
        # Slope must stay in [-0.01, 0.01] range.
        # Use high amplitude so variance exceeds the 75th percentile across
        # all examples (including FLIPPED ones which have variance from trend).
        base = rng.uniform(0.45, 0.65)
        amplitude = rng.uniform(0.45, 0.65)  # large amplitude -> variance > flipped
        freq = rng.uniform(0.06, 0.12)
        noise = rng.normal(0, 0.03, n_steps)
        # Zero-mean sin oscillation: slope ≈ 0
        losses = [
            float(np.clip(base + amplitude * np.sin(2 * np.pi * freq * t) + noise[t], 0.01, 2.0))
            for t in range(n_steps)
        ]

    else:  # clean
        # Loss decays quickly — model learns correctly.
        # Low variance, negative slope, low end-of-training loss.
        start = rng.uniform(0.8, 1.2)
        end = rng.uniform(0.03, 0.18)
        tau = rng.uniform(15, 40)
        noise = rng.normal(0, 0.025, n_steps)
        losses = [
            float(np.clip(end + (start - end) * np.exp(-t / tau) + noise[t], 0.0, 2.0))
            for t in range(n_steps)
        ]

    return [(t, l) for t, l in enumerate(losses)]


def generate_dataset(
    n_flipped: int = 50,
    n_ambiguous: int = 50,
    n_clean: int = 150,
    n_steps: int = 100,
    seed: int = 42,
) -> tuple[list[SyntheticExample], dict[int, list[tuple[int, float]]]]:
    """Generate synthetic dataset with known labels and simulated trajectories."""
    rng = np.random.default_rng(seed)
    examples: list[SyntheticExample] = []
    trajectories: dict[int, list[tuple[int, float]]] = {}
    idx = 0

    for _ in range(n_flipped):
        examples.append(SyntheticExample(idx=idx, true_label="flipped"))
        trajectories[idx] = _simulate_trajectory("flipped", n_steps, rng)
        idx += 1

    for _ in range(n_ambiguous):
        examples.append(SyntheticExample(idx=idx, true_label="ambiguous"))
        trajectories[idx] = _simulate_trajectory("ambiguous", n_steps, rng)
        idx += 1

    for _ in range(n_clean):
        examples.append(SyntheticExample(idx=idx, true_label="clean"))
        trajectories[idx] = _simulate_trajectory("clean", n_steps, rng)
        idx += 1

    return examples, trajectories


def _precision_recall_f1(
    predicted: set[int], true_positive_set: set[int]
) -> tuple[float, float, float]:
    tp = len(predicted & true_positive_set)
    fp = len(predicted - true_positive_set)
    fn = len(true_positive_set - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def run_benchmark(
    n_flipped: int = 50,
    n_ambiguous: int = 50,
    n_clean: int = 150,
    n_steps: int = 100,
    seed: int = 42,
) -> dict:
    """Run synthetic noise benchmark and return metrics."""
    examples, trajectories = generate_dataset(
        n_flipped=n_flipped,
        n_ambiguous=n_ambiguous,
        n_clean=n_clean,
        n_steps=n_steps,
        seed=seed,
    )
    n_total = len(examples)

    # Build TrajectoryLogger and feed simulated trajectories
    logger = TrajectoryLogger(n_examples=n_total)
    for idx, obs_list in trajectories.items():
        for step, loss in obs_list:
            logger.log(idx, step, loss)
    logger.finalize(total_steps=n_steps)

    features = logger.get_all_features()
    report = PreferenceQualityReport(features)

    # Ground truth sets
    true_flipped = {e.idx for e in examples if e.true_label == "flipped"}
    true_ambiguous = {e.idx for e in examples if e.true_label == "ambiguous"}
    true_clean = {e.idx for e in examples if e.true_label == "clean"}

    # Predicted sets
    pred_flipped = {r.example_idx for r in report.flipped}
    pred_ambiguous = {r.example_idx for r in report.ambiguous}
    pred_clean = {r.example_idx for r in report.clean}

    fp_p, fp_r, fp_f1 = _precision_recall_f1(pred_flipped, true_flipped)
    amb_p, amb_r, amb_f1 = _precision_recall_f1(pred_ambiguous, true_ambiguous)
    cl_p, cl_r, cl_f1 = _precision_recall_f1(pred_clean, true_clean)

    return {
        "n_total": n_total,
        "n_flipped": n_flipped,
        "n_ambiguous": n_ambiguous,
        "n_clean": n_clean,
        "flipped": {"precision": fp_p, "recall": fp_r, "f1": fp_f1},
        "ambiguous": {"precision": amb_p, "recall": amb_r, "f1": amb_f1},
        "clean": {"precision": cl_p, "recall": cl_r, "f1": cl_f1},
        "summary": report.summary(),
    }


def print_results(results: dict) -> None:
    print(f"\n{'='*60}")
    print("reservoir-prefcheck: Synthetic Noise Benchmark")
    print(f"{'='*60}")
    print(f"Dataset: {results['n_total']} examples")
    print(f"  FLIPPED:   {results['n_flipped']}")
    print(f"  AMBIGUOUS: {results['n_ambiguous']}")
    print(f"  CLEAN:     {results['n_clean']}")
    print()
    print(f"{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)
    for label in ("flipped", "ambiguous", "clean"):
        m = results[label]
        print(f"{label.upper():<12} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")
    print()
    s = results["summary"]
    print("Report summary:")
    print(f"  FLIPPED:   {s['n_flipped']:3d} ({s['pct_flipped']:.1f}%)")
    print(f"  AMBIGUOUS: {s['n_ambiguous']:3d} ({s['pct_ambiguous']:.1f}%)")
    print(f"  CLEAN:     {s['n_clean']:3d} ({s['pct_clean']:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    results = run_benchmark()
    print_results(results)
