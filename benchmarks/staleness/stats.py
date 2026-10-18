"""Uncertainty and curve summaries for the sweep report.

- ``bootstrap_ci``: percentile CI of an arm's accuracy, resampling seeds and
  evaluation items jointly (both with replacement), so it carries
  seed-to-seed and item-to-item variation.
- ``paired_bootstrap_diff``: the same for ``arm - base`` on the seeds both
  ran and the same resampled items (every arm answers the same items).
- ``reward_auc``: mean train reward over logged steps (area under the curve
  per step, so runs of equal length compare directly).
- ``steps_to_target``: first step whose rolling-mean reward reaches a target
  (the report uses 95% of the GRPO arm's final reward).
- ``at_rollout_budget``: a run cut at the last step whose cumulative
  generated completions fit a budget (the report's budget is the plain GRPO
  run's total), for comparing ``grpo_filter`` at matched total rollouts.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence

N_BOOT = 2000
ALPHA = 0.05
BOOT_SEED = 0
ROLLING_WINDOW = 10


def _finite(values: Sequence) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(v)]


def _matrix(items_by_seed: dict) -> tuple[list, list[list[float]]]:
    seeds = sorted(items_by_seed)
    rows = [[float(v) for v in items_by_seed[s]] for s in seeds]
    if len({len(r) for r in rows}) > 1:
        raise ValueError(f"every seed must score the same items; got item counts {[len(r) for r in rows]}")
    return seeds, rows


def _percentiles(samples: list[float], alpha: float) -> tuple[float, float]:
    ordered = sorted(samples)
    lo = ordered[int(math.floor(alpha / 2 * (len(ordered) - 1)))]
    hi = ordered[int(math.ceil((1 - alpha / 2) * (len(ordered) - 1)))]
    return lo, hi


def _resampled_means(rows: list[list[float]], n_boot: int, rng_seed: int) -> list[float]:
    """Bootstrap means over (seed, item) with both resampled; numpy when present."""
    k, n = len(rows), len(rows[0])
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy ships with every sweep environment
        rng = random.Random(rng_seed)
        out = []
        for _ in range(n_boot):
            seeds = [rng.randrange(k) for _ in range(k)]
            items = [rng.randrange(n) for _ in range(n)]
            out.append(sum(rows[s][i] for s in seeds for i in items) / (k * n))
        return out
    rng = np.random.default_rng(rng_seed)
    matrix = np.asarray(rows, dtype=np.float64)
    out = []
    for _ in range(n_boot):
        seeds = rng.integers(0, k, size=k)
        items = rng.integers(0, n, size=n)
        out.append(float(matrix[np.ix_(seeds, items)].mean()))
    return out


def bootstrap_ci(items_by_seed: dict, n_boot: int = N_BOOT, alpha: float = ALPHA, rng_seed: int = BOOT_SEED) -> dict:
    """Point estimate and ``1 - alpha`` percentile CI of the mean over seeds × items."""
    seeds, rows = _matrix(items_by_seed)
    mean = sum(map(sum, rows)) / (len(rows) * len(rows[0]))
    lo, hi = _percentiles(_resampled_means(rows, n_boot, rng_seed), alpha)
    return {"mean": mean, "lo": lo, "hi": hi, "alpha": alpha, "n_seeds": len(seeds), "n_items": len(rows[0]),
            "n_boot": n_boot}


def paired_bootstrap_diff(arm: dict, base: dict, n_boot: int = N_BOOT, alpha: float = ALPHA,
                          rng_seed: int = BOOT_SEED) -> Optional[dict]:
    """CI of ``mean(arm) - mean(base)`` on the seeds both have; ``None`` when they share none."""
    common = sorted(set(arm) & set(base))
    if not common:
        return None
    diff = {s: [a - b for a, b in zip(arm[s], base[s], strict=True)] for s in common}
    out = bootstrap_ci(diff, n_boot, alpha, rng_seed)
    return {**out, "seeds": common}


def rolling(values: Sequence, window: int) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for i in range(len(values)):
        chunk = _finite(values[max(0, i - window + 1):i + 1])
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def reward_auc(rewards: Sequence) -> Optional[float]:
    present = _finite(rewards)
    return sum(present) / len(present) if present else None


def steps_to_target(steps: Sequence[int], rewards: Sequence, target: float,
                    window: int = ROLLING_WINDOW) -> Optional[int]:
    for step, value in zip(steps, rolling(rewards, window)):
        if value is not None and value >= target:
            return int(step)
    return None


def at_rollout_budget(series: dict, evals: Sequence[dict], budget: float) -> Optional[dict]:
    """The run up to its last step within ``budget`` generated completions; ``None`` if it never logged them."""
    counts = series.get("rollouts") or []
    if not counts or any(c is None for c in counts):
        return None
    total, last = 0.0, None
    for i, count in enumerate(counts):
        if total + count > budget:
            break
        total += count
        last = i
    if last is None:
        return None
    step = int(series["step"][last])
    within = [e for e in evals if e.get("step") is not None and e["step"] <= step and "accuracy" in e]
    rewards = series["reward"][:last + 1]
    tail = rewards[-max(1, len(rewards) // 4):]
    return {"step": step, "rollouts": total, "reward_auc": reward_auc(rewards), "train_reward_last_quarter": reward_auc(tail),
            "avg_at_k": within[-1]["accuracy"] if within else None, "avg_at_k_step": within[-1]["step"] if within else None}
