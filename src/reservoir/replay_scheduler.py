"""
reservoir.replay_scheduler — Prioritized replay of forgotten anchor examples.

Uses numpy-based priority sampling with IS weight correction (same math as
FastPERBuffer's sum-tree but adapted for generic anchor dict storage).

See DECISIONS_B.md for rationale on IS weight approximation.
"""

from __future__ import annotations

import numpy as np
from reservoir.anchor_set import AnchorSet, AnchorExample


class ReplayScheduler:
    """
    Schedules prioritized replay of forgotten anchor examples.

    Samples anchors proportional to their forgetting severity (priority),
    with IS weight correction to keep the gradient approximately unbiased.

    Parameters
    ----------
    anchor_sets : list[AnchorSet]
    replay_ratio : float — fraction of each batch to replace. Default 0.1.
    alpha : float — PER exponent. Default 0.6.
    beta : float — IS correction exponent. Default 0.4.
    """

    def __init__(
        self,
        anchor_sets: list[AnchorSet],
        replay_ratio: float = 0.1,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> None:
        self.anchor_sets = anchor_sets
        self.replay_ratio = replay_ratio
        self.alpha = alpha
        self.beta = beta

        self._total_replayed = 0

        # Collect all anchors and initialize priority arrays
        self._anchors: list[AnchorExample] = []
        for anchor_set in anchor_sets:
            self._anchors.extend(anchor_set)

        n = len(self._anchors)
        self._priorities = np.array(
            [a.priority for a in self._anchors], dtype=np.float64
        )
        # Track which anchor_set each anchor belongs to for priority sync
        self._anchor_set_map: dict[int, AnchorSet] = {}
        for anchor_set in anchor_sets:
            for anchor in anchor_set:
                self._anchor_set_map[id(anchor)] = anchor_set

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_replayed(self) -> int:
        return self._total_replayed

    def update_priorities(self, anchor_set: AnchorSet) -> None:
        """Sync PER buffer priorities from anchor_set priorities."""
        for anchor in anchor_set:
            # Find this anchor in our flat list by object identity
            for i, a in enumerate(self._anchors):
                if a is anchor:
                    self._priorities[i] = anchor.priority
                    break

    def get_replay_batch(
        self, batch_size: int
    ) -> tuple[list[AnchorExample], list[float]]:
        """
        Sample anchors proportional to forgetting priority.

        Returns
        -------
        (anchor_examples, is_weights)
            anchor_examples: list of AnchorExample to replay
            is_weights: IS correction weights (floats in (0, 1])
            Both are empty lists if total priority is zero.
        """
        n_replay = max(1, int(batch_size * self.replay_ratio))
        n = len(self._anchors)

        if n == 0:
            return [], []

        # Check if any anchor has non-zero priority
        total_priority = float(self._priorities.sum())
        if total_priority <= 0.0:
            return [], []

        # Apply alpha exponent to priorities
        priorities_alpha = self._priorities ** self.alpha
        total_alpha = float(priorities_alpha.sum())

        if total_alpha <= 0.0:
            return [], []

        probs = priorities_alpha / total_alpha

        # Sample with replacement proportional to priority
        n_sample = min(n_replay, n)
        sampled_indices = np.random.choice(n, size=n_sample, replace=True, p=probs)

        # Compute IS weights: w_i = (1 / (n * p_i))^beta, normalized by max
        sampled_probs = probs[sampled_indices]
        weights_raw = (n * sampled_probs) ** (-self.beta)
        # Normalize by max weight so weights in (0, 1]
        max_weight = float(weights_raw.max()) if len(weights_raw) > 0 else 1.0
        if max_weight <= 0.0:
            max_weight = 1.0
        is_weights = (weights_raw / max_weight).clip(0.0, 1.0).tolist()

        examples = [self._anchors[i] for i in sampled_indices]
        self._total_replayed += len(examples)

        return examples, is_weights
