"""Resumable Atari benchmark runner.

Usage:
    python -m benchmarks.run_atari --buffer-type c --games BreakoutNoFrameskip-v4 --seeds 1
    python -m benchmarks.run_atari --buffer-type c --all-57 --seeds 1 2 3
    python -m benchmarks.run_atari --buffer-type uniform --all-57 --seeds 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


CONFIGS = Path(__file__).parent / "configs"
RESULTS = Path(__file__).parent / "results"
GAMES_FILE = CONFIGS / "games.txt"


def load_games() -> list[str]:
    return [line.strip() for line in GAMES_FILE.read_text().splitlines() if line.strip()]


def result_path(game: str, buffer_type: str, seed: int) -> Path:
    return RESULTS / f"{game}_{buffer_type}_seed{seed}.json"


def run_one(game: str, buffer_type: str, seed: int, extra_args: list[str]) -> None:
    out_path = result_path(game, buffer_type, seed)
    if out_path.exists():
        print(f"[SKIP] {game} / {buffer_type} / seed={seed} — result exists")
        return

    print(f"[RUN]  {game} / {buffer_type} / seed={seed}")
    RESULTS.mkdir(exist_ok=True)

    cmd = [
        sys.executable, "-m", "benchmarks.atari_dqn",
        "--env-id", game,
        "--buffer-type", buffer_type,
        "--seed", str(seed),
        "--output", str(out_path),
    ] + extra_args

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[FAIL] {game} / {buffer_type} / seed={seed} — exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Atari PER benchmarks")
    parser.add_argument("--buffer-type", choices=["c", "python", "uniform"], required=True)
    parser.add_argument("--games", nargs="+", default=None)
    parser.add_argument("--all-57", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--total-steps", type=int, default=50_000_000)
    args = parser.parse_args()

    if args.all_57:
        games = load_games()
    elif args.games:
        games = args.games
    else:
        parser.error("Specify --games or --all-57")

    extra = ["--total-timesteps", str(args.total_steps)]

    for game in games:
        for seed in args.seeds:
            run_one(game, args.buffer_type, seed, extra)

    print(f"\nDone. Results in {RESULTS}/")


if __name__ == "__main__":
    main()
