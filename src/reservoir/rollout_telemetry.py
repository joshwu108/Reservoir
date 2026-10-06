"""
reservoir.rollout_telemetry — Exact replay-health quantities for the log.

Two numbers about a replayed batch are functions of what the log already
records, so the checker can recompute them and the buffer writes them as
exact rationals:

- **Effective sample size** of the replayed rows, ``(sum w)^2 / sum w^2``
  over the batch's importance weights. It falls when a few rows carry most
  of the weight, which is how off-policy replay degrades silently.
- **Staleness** of the replayed rows, in model versions behind the current
  one: the maximum and the sum (the mean is ``sum / n``).

The adapter adds measurements the checker cannot recompute (the
log-ratios between stored behavior logprobs and the current policy); the
telemetry record lists those under ``reported`` so a reader knows which
numbers are verified and which are carried.
"""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple, Sequence


class ExactTelemetry(NamedTuple):
    """The verifiable part of one telemetry record."""

    ess: Fraction            # effective sample size of the replayed rows
    staleness_max: int       # oldest replayed row, in versions behind
    staleness_sum: int       # sum of ages; divide by the row count for the mean


def exact_telemetry(is_weights: Sequence[Fraction], ages: Sequence[int]) -> ExactTelemetry:
    """ESS and staleness from a batch's weights and the ages of its rows."""
    if not is_weights or len(is_weights) != len(ages):
        raise ValueError("telemetry needs one age per weight and at least one row")
    total = sum(is_weights, Fraction(0))
    squares = sum((w * w for w in is_weights), Fraction(0))
    if squares == 0:
        raise ValueError("telemetry: every importance weight is zero")
    if any(a < 0 for a in ages):
        raise ValueError("telemetry: an age is negative")
    return ExactTelemetry(ess=total * total / squares, staleness_max=max(ages), staleness_sum=sum(ages))
