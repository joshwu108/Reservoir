"""reservoir.dataset_buffer — Prioritized sampler for supervised/RLHF datasets.

Tracks per-example loss across training and optionally samples proportional
to loss (accelerated mode) or uniformly (audit mode).
"""

from __future__ import annotations

import random
import warnings
from typing import Any

import numpy as np

from reservoir.fast_buffer import FastPERBuffer


class DatasetBuffer:
    """Prioritized sampler for supervised/RLHF training datasets.

    Tracks per-example loss across training and optionally samples
    proportional to loss (accelerated mode) or uniformly (audit mode).

    Parameters
    ----------
    dataset : any indexable dataset (HuggingFace Dataset, list, etc.)
    alpha : float — PER exponent. Default 0.6.
    beta : float — IS correction exponent. Default 0.4.
    epsilon : float — minimum priority. Default 1e-6.
    mode : "audit" | "accelerated"
        audit: uniform sampling, priorities tracked passively (default, safe)
        accelerated: prioritized sampling active
    priority_cap : float — in accelerated mode, cap priority at this multiple
        of current median to prevent never-converging examples dominating.
        Default 10.0.
    """

    def __init__(
        self,
        dataset: Any,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        mode: str = "audit",
        priority_cap: float = 10.0,
    ) -> None:
        n = len(dataset)
        if n > 100_000:
            warnings.warn(
                f"DatasetBuffer: tracking priorities for {n} examples in memory. "
                "Consider chunking for very large datasets."
            )

        self._dataset = dataset
        self._n = n
        self._mode = mode
        self._alpha = alpha
        self._beta = beta
        self._epsilon = epsilon
        self._priority_cap = priority_cap

        # Raw priorities (before alpha exponentiation), for cap computation
        self._raw_priorities = np.full(n, epsilon, dtype=np.float64)

        # Priority tree via FastPERBuffer
        # obs_shape=(1,), action_dim=1 as specified; actual data comes from dataset
        self._per = FastPERBuffer(
            capacity=n,
            obs_shape=(1,),
            action_dim=1,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
        )

        # Pre-populate with n dummy transitions so positions 0..n-1 are filled
        dummy_obs = np.zeros((1,), dtype=np.float32)
        for _ in range(n):
            self._per.add(dummy_obs, 0, 0.0, dummy_obs, False)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        item = dict(self._dataset[idx])
        item["__index__"] = idx
        return item

    def update_priority(self, idx: int, loss: float) -> None:
        """Update this example's priority based on its training loss."""
        raw = abs(loss) + self._epsilon

        if self._mode == "accelerated":
            pos_priorities = self._raw_priorities[self._raw_priorities > 0]
            if len(pos_priorities) > 0:
                cap = self._priority_cap * float(np.median(pos_priorities))
                raw = min(raw, cap)

        self._raw_priorities[idx] = raw
        # update_priorities takes td_errors; internally applies abs(err) + epsilon
        # Pass (raw - epsilon) so the stored priority = abs(raw - epsilon) + epsilon = raw
        td_err = raw - self._epsilon
        self._per.update_priorities(
            np.array([idx], dtype=np.int64),
            np.array([td_err], dtype=np.float64),
        )

    def sample_indices(self, batch_size: int) -> list[int]:
        """Sample batch_size example indices.

        audit mode: uniform random sampling
        accelerated mode: proportional to priority (PER)
        """
        if self._mode == "audit":
            return random.sample(range(self._n), batch_size)
        else:
            batch = self._per.sample(batch_size)
            return list(batch.indices.tolist())

    def get_is_weights(self, indices: list[int]) -> list[float]:
        """Return importance-sampling weights for the given indices.

        audit mode: all 1.0 (uniform sampling needs no correction)
        accelerated mode: IS weights normalized to [0, 1]
        """
        if self._mode == "audit":
            return [1.0] * len(indices)

        total = self._per.total_priority
        min_p = float(self._per._min_tree[0])
        n = self._n

        if total <= 0:
            return [1.0] * len(indices)

        max_weight = (n * min_p / total) ** (-self._beta) if min_p > 0 else 1.0

        weights = []
        for idx in indices:
            leaf_pos = self._per._tree_capacity - 1 + idx
            p_alpha = float(self._per._tree[leaf_pos])
            prob = max(p_alpha / total, 1e-10)
            w = (n * prob) ** (-self._beta) / max_weight
            weights.append(float(np.clip(w, 0.0, 1.0)))
        return weights

    @property
    def priorities(self) -> np.ndarray:
        """Current raw priorities for all examples."""
        return self._raw_priorities.copy()

    @property
    def size(self) -> int:
        """Number of examples in the dataset."""
        return self._n
