"""reservoir.dataset_buffer — Prioritized sampler over a dataset of prompts or examples.

Two ways to drive it:

- **Loss-driven** (supervised / RLHF fine-tuning): call ``update_priority(idx,
  loss)`` after each step and examples with higher loss are sampled more.
- **Rollout-driven** (GRPO-style RL): construct with a ``PromptPriority``
  strategy such as ``PassRateVariance`` and call ``update_group(idx,
  model_version, rollouts)`` after generating a group for prompt ``idx``.
  The strategy scores the whole group (its pass rate, say) and that score
  becomes the prompt's priority, so the next batch of prompts to generate
  rollouts for favours the informative ones.

``mode="audit"`` samples uniformly and only records priorities, which is
the safe default when no strategy is given; ``mode="accelerated"`` samples
proportionally. A strategy switches the default to accelerated.

This buffer is backed by the float ``FastPERBuffer`` and sampling uses
numpy's global RNG; it is a convenience sampler, not part of the exact,
attested path (that is ``RolloutBuffer``).
"""

from __future__ import annotations

import math
import random
import warnings
from typing import Any, Callable, Optional, Sequence

import numpy as np

from reservoir.fast_buffer import FastPERBuffer
from reservoir.priorities import PromptPriority, validated_prompt_score
from reservoir.rollout import Rollout, RolloutGroup, default_is_success

_MODES = ("audit", "accelerated")


class DatasetBuffer:
    """Prioritized sampler for supervised/RLHF training datasets.

    Tracks per-example loss across training and optionally samples
    proportional to loss (accelerated mode) or uniformly (audit mode).

    Parameters
    ----------
    dataset : any indexable dataset (HuggingFace Dataset, list, etc.)
    priority : PromptPriority, optional
        Scores a prompt from the rollout group it produced; enables
        ``update_group``. Rollout-level strategies are rejected.
    alpha : float — PER exponent. Default 0.6.
    beta : float — IS correction exponent. Default 0.4.
    epsilon : float — minimum priority. Default 1e-6.
    mode : "audit" | "accelerated" | None
        audit: uniform sampling, priorities tracked passively.
        accelerated: prioritized sampling active.
        None (default): "accelerated" if a strategy is given, else "audit".
    priority_cap : float — in accelerated mode, cap priority at this multiple
        of the median priority of the examples updated so far, so one
        never-converging example cannot dominate. Default 10.0. Examples
        that have never been updated do not count towards the median.
    """

    def __init__(
        self,
        dataset: Any,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        mode: Optional[str] = None,
        priority_cap: float = 10.0,
        priority: Optional[PromptPriority] = None,
    ) -> None:
        n = len(dataset)
        if n > 100_000:
            warnings.warn(
                f"DatasetBuffer: tracking priorities for {n} examples in memory. "
                "Consider chunking for very large datasets."
            )
        if priority is not None and not isinstance(priority, PromptPriority):
            raise TypeError(
                f"priority must be a PromptPriority (prompt-level strategy), "
                f"got {type(priority).__name__}"
            )
        if mode is None:
            mode = "accelerated" if priority is not None else "audit"
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")

        self._dataset = dataset
        self._n = n
        self._mode = mode
        self._priority = priority
        self._alpha = alpha
        self._beta = beta
        self._epsilon = epsilon
        self._priority_cap = priority_cap

        # Raw priorities (before alpha exponentiation), for cap computation.
        # Untouched examples sit at epsilon; _updated marks the ones a caller
        # has actually scored so the cap's median ignores the rest.
        self._raw_priorities = np.full(n, epsilon, dtype=np.float64)
        self._updated = np.zeros(n, dtype=bool)

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

    @property
    def mode(self) -> str:
        """"audit" (uniform sampling) or "accelerated" (prioritized sampling)."""
        return self._mode

    @property
    def priority(self) -> Optional[PromptPriority]:
        """The prompt-level strategy, or None when loss-driven."""
        return self._priority

    def update_priority(self, idx: int, loss: float) -> None:
        """Update this example's priority from its training loss: |loss| + epsilon.

        Raises
        ------
        IndexError
            If ``idx`` is out of range.
        ValueError
            If ``idx`` is not an integer or ``loss`` is not finite.
        """
        self._require_index(idx)
        if isinstance(loss, bool) or not isinstance(loss, (int, float, np.floating)) or not math.isfinite(loss):
            raise ValueError(f"loss must be a finite number, got {loss!r}")
        self._apply_raw_priority(idx, abs(float(loss)) + self._epsilon)

    def update_group(
        self,
        idx: int,
        model_version: int,
        rollouts: Sequence[Rollout],
        is_success: Optional[Callable[[Rollout], bool]] = None,
    ) -> None:
        """Update prompt ``idx`` from the rollout group just generated for it.

        Builds a ``RolloutGroup`` (prompt_id is the dataset index as a
        string), scores it with the strategy, and stores the score as the
        prompt's raw priority. Two adjustments can apply: a score below the
        buffer's ``epsilon`` is raised to it (the underlying tree needs a
        positive priority), and in accelerated mode the score is capped at
        ``priority_cap`` times the median of the updated examples.
        Validation happens before the priority array is touched, so a bad
        group or score changes nothing.

        Raises
        ------
        IndexError
            If ``idx`` is out of range.
        ValueError
            No strategy was given, the index is not an integer, the group
            is malformed, or the strategy returned an invalid score.
        """
        if self._priority is None:
            raise ValueError("update_group needs a priority strategy; construct with priority=...")
        self._require_index(idx)
        group = RolloutGroup(
            prompt_id=str(idx),
            model_version=model_version,
            rollouts=rollouts,
            is_success=is_success if is_success is not None else default_is_success,
        )
        score = validated_prompt_score(self._priority, group)
        self._apply_raw_priority(idx, score)

    def _require_index(self, idx: int) -> None:
        """ValueError for a non-integer index, IndexError for one out of range."""
        if isinstance(idx, bool) or not isinstance(idx, (int, np.integer)):
            raise ValueError(f"index must be an int, got {idx!r}")
        if not (0 <= idx < self._n):
            raise IndexError(f"index {idx} out of range [0, {self._n})")

    def _apply_raw_priority(self, idx: int, raw: float) -> None:
        """Store a raw priority (floored at epsilon, capped in accelerated mode) and push it to the tree.

        The float tree stores ``|td_error| + epsilon``, so a priority below
        epsilon cannot be represented; flooring keeps ``priorities`` and the
        tree in agreement. The cap's median is taken over examples that
        have been updated before this one, so the first updates are never
        capped against the epsilon placeholders of untouched examples.
        """
        raw = max(raw, self._epsilon)
        if self._mode == "accelerated" and self._updated.any():
            cap = self._priority_cap * float(np.median(self._raw_priorities[self._updated]))
            raw = min(raw, cap)

        self._raw_priorities[idx] = raw
        self._updated[idx] = True
        # The tree stores abs(td_err) + epsilon; pass raw - epsilon (>= 0 after
        # the floor) so the stored priority is exactly raw.
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
