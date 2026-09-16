"""reservoir.trajectory — Per-example loss trajectory recording and feature extraction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryFeatures:
    """Extracted features from a single example's loss trajectory."""

    example_idx: int
    n_observations: int
    mean_loss_last_k: float
    slope: float
    variance: float
    first_correct_step: int | None
    loss_history: list[float]


class TrajectoryLogger:
    """Records per-example loss trajectory across training steps.

    Parameters
    ----------
    n_examples : int — number of examples in dataset
    window_frac : float — fraction of total_steps for mean_loss_last_k. Default 0.2.
    correct_threshold : float — loss below this = model correct. Default 0.5.
    warn_threshold : int — warn if n_examples exceeds this. Default 100_000.
    """

    def __init__(
        self,
        n_examples: int,
        window_frac: float = 0.2,
        correct_threshold: float = 0.5,
        warn_threshold: int = 100_000,
    ) -> None:
        if n_examples > warn_threshold:
            warnings.warn(
                f"TrajectoryLogger: tracking trajectories for {n_examples} examples."
            )
        self._n_examples = n_examples
        self._window_frac = window_frac
        self._correct_threshold = correct_threshold

        # {example_idx: [(step, loss), ...]}
        self._records: dict[int, list[tuple[int, float]]] = {}
        self._features: dict[int, TrajectoryFeatures] = {}
        self._finalized = False

    def log(self, example_idx: int, step: int, loss: float) -> None:
        """Record (step, loss) for this example."""
        if example_idx not in self._records:
            self._records[example_idx] = []
        self._records[example_idx].append((step, loss))

    def finalize(self, total_steps: int) -> None:
        """Compute TrajectoryFeatures for all logged examples."""
        cutoff_step = total_steps * (1.0 - self._window_frac)
        self._features = {}

        for idx, obs_list in self._records.items():
            steps = [o[0] for o in obs_list]
            losses = [o[1] for o in obs_list]
            n = len(obs_list)

            loss_arr = np.array(losses, dtype=np.float64)

            # Slope
            if n >= 2:
                slope = float(np.polyfit(steps, losses, deg=1)[0])
            else:
                slope = 0.0

            # Variance
            if n >= 2:
                variance = float(np.var(loss_arr))
            else:
                variance = 0.0

            # mean_loss_last_k
            window_obs = [l for s, l in obs_list if s >= cutoff_step]
            if window_obs:
                mean_loss_last_k = float(np.mean(window_obs))
            else:
                mean_loss_last_k = float(losses[-1]) if losses else 0.0

            # first_correct_step
            first_correct_step: int | None = None
            for s, l in obs_list:
                if l < self._correct_threshold:
                    first_correct_step = s
                    break

            self._features[idx] = TrajectoryFeatures(
                example_idx=idx,
                n_observations=n,
                mean_loss_last_k=mean_loss_last_k,
                slope=slope,
                variance=variance,
                first_correct_step=first_correct_step,
                loss_history=losses,
            )

        self._finalized = True

    def get_features(self, example_idx: int) -> TrajectoryFeatures | None:
        """Return features for one example. None if never logged."""
        return self._features.get(example_idx, None)

    def get_all_features(self) -> dict[int, TrajectoryFeatures]:
        """Return all computed features. Raises RuntimeError if called before finalize()."""
        if not self._finalized:
            raise RuntimeError(
                "TrajectoryLogger.finalize() must be called before get_all_features()."
            )
        return dict(self._features)

    def summary_stats(self) -> dict:
        """Return summary statistics about logging coverage."""
        n_logged = len(self._records)
        n_never_seen = self._n_examples - n_logged
        if n_logged > 0:
            mean_observations = float(
                np.mean([len(v) for v in self._records.values()])
            )
        else:
            mean_observations = 0.0
        return {
            "n_logged": n_logged,
            "n_never_seen": n_never_seen,
            "mean_observations": mean_observations,
        }
