"""
reservoir.rollout_args — Argument checks for ``RolloutBuffer`` construction.

Each returns the validated value or raises ``ValueError`` (``TypeError``
for a priority that is not a strategy) before the buffer builds anything.
"""

from __future__ import annotations

import math
from typing import Optional

from reservoir.priorities import AdvantagePriority, PriorityStrategy


def validate_min_entry_age(value: object, max_policy_age: int) -> int:
    """A non-negative int no larger than ``max_policy_age`` (an older entry is already evicted)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"min_entry_age must be a non-negative int, got {value!r}")
    if value > max_policy_age:
        raise ValueError(f"min_entry_age {value} exceeds max_policy_age {max_policy_age}; nothing could be drawn")
    return value


def validate_priority(priority: Optional[PriorityStrategy]) -> PriorityStrategy:
    """Default to AdvantagePriority; anything that is not a strategy is a TypeError."""
    if priority is None:
        return AdvantagePriority()
    if not isinstance(priority, PriorityStrategy):
        raise TypeError(f"priority must be a PriorityStrategy, got {type(priority).__name__}")
    return priority


def validate_exponent(value: object, name: str, minimum_exclusive: Optional[float],
                      maximum: Optional[float] = None) -> float:
    """Validate alpha or beta.

    Both must be finite real numbers (bool excluded). With
    ``minimum_exclusive=0.0`` the value must be strictly positive (alpha:
    ``raw ** 0`` would flatten every priority to 1); with ``None`` it only
    has to be non-negative (beta: 0 means no importance correction).
    ``maximum`` bounds it from above (beta: 1 is full correction, and a
    larger exponent can overflow the weight of a wide priority spread).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if minimum_exclusive is not None and value <= minimum_exclusive:
        raise ValueError(f"{name} must be > {minimum_exclusive}, got {value!r}")
    if minimum_exclusive is None and value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value!r}")
    return float(value)
