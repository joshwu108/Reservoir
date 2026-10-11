"""Pieces both replay adapters (TRL and verl) share.

``StoredAdvantagePriority`` is the default priority of both, the two
validators check the legacy drift-gate arguments, and ``rescore_placed``
asks the buffer's strategy for new priorities of the rows a step placed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from reservoir.priorities import DEFAULT_EPSILON, PriorityStrategy, ReplaySignal, _require_epsilon, validated_rescore
from reservoir.rollout import Rollout, RolloutGroup
from reservoir.rollout_buffer import RolloutBatch


@dataclass(frozen=True)
class StoredAdvantagePriority(PriorityStrategy):
    """``|reward| + epsilon`` where ``reward`` holds TRL's advantage.

    TRL centres advantages within each group before the adapter sees
    them, so ``|advantage|`` is already the group-relative magnitude
    that advantage-prioritized replay samples on. ``AdvantagePriority``
    would re-centre on the mean of the rows that survived truncation,
    which is not the same quantity.
    """

    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        object.__setattr__(self, "epsilon", _require_epsilon(self.epsilon))

    def score(self, rollout: Rollout, group: RolloutGroup) -> float:
        return abs(rollout.reward) + self.epsilon


def _validated_gate(max_log_ratio: Optional[float]) -> Optional[float]:
    """The drift gate threshold as a finite positive float, or None when off; anything else is a ValueError."""
    if max_log_ratio is None:
        return None
    if (isinstance(max_log_ratio, bool) or not isinstance(max_log_ratio, (int, float))
            or not math.isfinite(max_log_ratio) or max_log_ratio <= 0):
        raise ValueError(f"max_log_ratio must be a finite positive number or None, got {max_log_ratio!r}")
    return float(max_log_ratio)


def _validated_decline_cap(max_declines_per_step: Optional[int]) -> Optional[int]:
    """The per-step decline cap as a non-negative int, or None when uncapped."""
    if max_declines_per_step is None:
        return None
    if isinstance(max_declines_per_step, bool) or not isinstance(max_declines_per_step, int) or max_declines_per_step < 0:
        raise ValueError(f"max_declines_per_step must be a non-negative int or None, got {max_declines_per_step!r}")
    return max_declines_per_step


def _validated_floor(min_entry_age: object) -> int:
    """The adapters' ``min_entry_age``: 0 or 1.

    An adapter stores every step's live groups, so at step ``s`` the
    previous step's groups are one version old; with a floor of 2 or more
    some live entry is always too young and every sample would be empty.
    """
    if isinstance(min_entry_age, bool) or min_entry_age not in (0, 1):
        raise ValueError(
            f"min_entry_age must be 0 or 1 for an adapter that stores every step, got {min_entry_age!r}"
        )
    return int(min_entry_age)  # type: ignore[call-overload]


def rescore_placed(buffer: Any, batch: RolloutBatch, kept: Sequence[int], weighted: Sequence[float],
                   ratios: Sequence[float], step: int) -> int:
    """Ask the strategy for new priorities of the placed rollouts; write them as updates.

    ``weighted[i]`` is the advantage written for draw ``kept[i]``
    (importance weight and any policy scale applied). A slot drawn more
    than once is rescored once, from its first placement. Declined draws
    are not rescored (an entry declined for age or drift is gone).
    Returns the number of slots rescored.
    """
    indices: list[int] = []
    scores: list[float] = []
    seen: set[int] = set()
    for position, k in enumerate(kept):
        slot = batch.indices[k]
        if slot in seen:
            continue
        seen.add(slot)
        ratio = ratios[k] if ratios else None
        signal = ReplaySignal(
            advantage=weighted[position], is_weight=float(batch.is_weights[k]),
            log_ratio=ratio if ratio is not None and math.isfinite(ratio) else None,
            age=buffer.current_version - batch.model_versions[k], step=step,
        )
        value = validated_rescore(buffer.priority, batch.rollouts[k], batch.groups[k], signal)
        if value is not None:
            indices.append(slot)
            scores.append(value)
    if indices:
        buffer.update_priorities(indices, scores)
    return len(indices)
