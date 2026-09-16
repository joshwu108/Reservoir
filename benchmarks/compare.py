"""Load benchmark results and print a comparison table + save a PNG chart.

Usage:
    python -m benchmarks.compare
    python -m benchmarks.compare --results-dir benchmarks/results/
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(results_dir: Path) -> list[dict]:
    results = []
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results


def mean_final_reward(result: dict, n: int = 100) -> float:
    rewards = result.get("episode_rewards", [])
    if not rewards:
        return float("nan")
    return float(np.mean(rewards[-n:]))


def print_table(data: dict[str, dict[str, float]]) -> None:
    buffer_types = sorted({bt for game_d in data.values() for bt in game_d})
    header = f"{'Game':<40} " + "  ".join(f"{bt:>10}" for bt in buffer_types)
    print(header)
    print("-" * len(header))
    for game in sorted(data):
        row = f"{game:<40} "
        row += "  ".join(
            f"{data[game].get(bt, float('nan')):>10.1f}" for bt in buffer_types
        )
        print(row)


def save_png(data: dict, buffer_types: list[str], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping PNG. pip install reservoir[atari]")
        return

    games = sorted(data)
    x = np.arange(len(games))
    width = 0.8 / len(buffer_types)

    fig, ax = plt.subplots(figsize=(max(12, len(games) * 0.4), 6))
    for i, bt in enumerate(buffer_types):
        values = [data[g].get(bt, 0.0) for g in games]
        ax.bar(x + i * width, values, width, label=bt)

    ax.set_xticks(x + width * (len(buffer_types) - 1) / 2)
    ax.set_xticklabels([g.replace("NoFrameskip-v4", "") for g in games],
                       rotation=90, fontsize=7)
    ax.set_ylabel("Mean episode reward (last 100 eps)")
    ax.set_title("PER Buffer Comparison — Atari 57")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).parent / "results")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    # Aggregate: game -> buffer_type -> mean final reward (averaged over seeds)
    raw: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        raw[r["game"]][r["buffer_type"]].append(mean_final_reward(r))

    data = {
        game: {bt: float(np.mean(vals)) for bt, vals in game_d.items()}
        for game, game_d in raw.items()
    }

    print_table(data)

    buffer_types = sorted({bt for game_d in data.values() for bt in game_d})
    save_png(data, buffer_types, args.results_dir / "comparison.png")


if __name__ == "__main__":
    main()
