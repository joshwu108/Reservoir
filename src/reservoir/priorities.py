"""
reservoir.priorities — Pluggable priority strategies for rollout replay.

What a strategy does
--------------------
A strategy answers one question: "how much should this entry be sampled,
relative to the others?" It returns a single non-negative finite float.
The buffer turns that float into an exact integer priority and everything
downstream is integer arithmetic::

    strategy.score(...)  ->  raw ** alpha  ->  quantize to 16.16 fixed point
        ->  age decay  ->  exact sum-tree  ->  P(i) = weight_i / total

So a strategy is the one place a user writes floating-point judgement;
it never sees the tree and cannot break its exactness.

Using a built-in::

    from reservoir.priorities import AdvantagePriority, PassRateVariance

    buf = RolloutBuffer(capacity=50_000, priority=AdvantagePriority())
    prompts = DatasetBuffer(dataset, priority=PassRateVariance())

Two levels
----------
Two different things get sampled, so there are two base classes:

- ``PriorityStrategy.score(rollout, group)`` scores one rollout inside the
  group it was generated in. ``RolloutBuffer`` uses it to choose which
  stored rollouts to replay.
- ``PromptPriority.score_prompt(group)`` scores a whole prompt from the
  group it produced. ``DatasetBuffer`` uses it to choose which prompts to
  generate new rollouts for.

The methods have different names so one class can implement both.
``PassRateTargeting`` and ``PassRateVariance`` do, because a pass-rate
score is a property of the group and applies equally to every rollout in
it. ``AdvantagePriority`` is rollout-level only.

Built-ins
---------
- ``AdvantagePriority``: |reward - mean_reward| + epsilon. Rollouts whose
  outcome differed most from their group carry the most signal
  (advantage-prioritized replay, arXiv 2606.04560).
- ``PassRateTargeting``: Gaussian bump centred on a target pass rate.
  ExGRPO (arXiv 2510.02245) treats prompts near 50% pass rate as the most
  informative.
- ``PassRateVariance``: pass_rate * (1 - pass_rate), the variance of a
  Bernoulli outcome. DAPO's dynamic sampling (arXiv 2503.14476) drops
  all-pass and all-fail prompts; this is the soft version.

All three are frozen dataclasses: parameters are validated once at
construction, cannot be changed afterwards, and two instances with the
same parameters compare equal. Each adds a small ``epsilon`` so a zero
score stays sampleable, mirroring the ``|TD error| + epsilon`` convention
of classic PER; pass ``epsilon=0.0`` to let zero-score entries drop out.

Writing your own
----------------
Subclass and implement one method::

    class RewardGap(PriorityStrategy):
        def score(self, rollout, group) -> float:
            return abs(rollout.reward - group.mean_reward)

``rollout`` is a ``Rollout`` and ``group`` the ``RolloutGroup`` it belongs
to; ``group.mean_reward``, ``group.pass_rate`` and ``group.advantages``
are typical inputs. Return any finite float >= 0.

Validation contract
-------------------
Buffers never call ``score`` or ``score_prompt`` directly. They call
``validated_score(strategy, rollout, group)`` or
``validated_prompt_score(strategy, group)``, which check the argument
types (``TypeError``) and then reject NaN, infinity, negatives and
non-numbers in the result (``ValueError`` naming the strategy class). A
buggy strategy therefore fails loudly at the boundary instead of writing
a bad weight into the tree.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final

# Shared with rollout.py so both modules map the same invalid inputs (NaN,
# inf, bool, integers too large for a float) to the same ValueError.
from reservoir.rollout import Rollout, RolloutGroup, _require_finite_float

DEFAULT_EPSILON: Final[float] = 1e-6
DEFAULT_PASS_RATE_TARGET: Final[float] = 0.5
DEFAULT_PASS_RATE_WIDTH: Final[float] = 0.15


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require_epsilon(epsilon: object) -> float:
    """Epsilon must be finite and >= 0."""
    value = _require_finite_float(epsilon, "epsilon")
    if value < 0:
        raise ValueError(f"epsilon must be >= 0, got {value!r}")
    return value


def _require_group(group: object) -> None:
    """Raise TypeError unless ``group`` is a RolloutGroup.

    Checked before calling into user code so a wrong argument fails here
    with a clear message instead of as an AttributeError inside ``score``.
    """
    if not isinstance(group, RolloutGroup):
        raise TypeError(f"group must be a RolloutGroup, got {type(group).__name__}")


def _check_score(score: object, strategy: object) -> float:
    """Return score as a Python float if it is finite and >= 0, else raise.

    The error names the strategy class so a misbehaving user strategy is
    easy to find.
    """
    name = type(strategy).__name__
    try:
        value = _require_finite_float(score, "score")
    except ValueError as exc:
        raise ValueError(f"{name} returned a non-finite or non-numeric score: {score!r}") from exc
    if value < 0:
        raise ValueError(f"{name} returned a negative score: {value!r}")
    return value


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class PriorityStrategy(ABC):
    """Scores one rollout within its group. Subclass and implement ``score``."""

    @abstractmethod
    def score(self, rollout: Rollout, group: RolloutGroup) -> float:
        """Return a non-negative finite priority for ``rollout``.

        ``group`` is the group the rollout was generated in; use it for
        group-relative quantities such as ``group.mean_reward``.
        """


class PromptPriority(ABC):
    """Scores a whole prompt from its rollout group. Implement ``score_prompt``."""

    @abstractmethod
    def score_prompt(self, group: RolloutGroup) -> float:
        """Return a non-negative finite priority for the prompt that produced ``group``."""


def validated_score(strategy: PriorityStrategy, rollout: Rollout, group: RolloutGroup) -> float:
    """Call ``strategy.score`` and validate the result. The buffer's entry point.

    Raises
    ------
    TypeError
        If ``strategy``, ``rollout`` or ``group`` is not of the expected type.
    ValueError
        If the score is not a finite, non-negative real number. The message
        names the strategy class.
    """
    if not isinstance(strategy, PriorityStrategy):
        raise TypeError(
            f"strategy must be a PriorityStrategy, got {type(strategy).__name__}"
        )
    if not isinstance(rollout, Rollout):
        raise TypeError(f"rollout must be a Rollout, got {type(rollout).__name__}")
    _require_group(group)
    return _check_score(strategy.score(rollout, group), strategy)


def validated_prompt_score(strategy: PromptPriority, group: RolloutGroup) -> float:
    """Call ``strategy.score_prompt`` and validate the result.

    Same checks as ``validated_score``: TypeError for a wrong ``strategy``
    or ``group`` type, ValueError for a bad score.
    """
    if not isinstance(strategy, PromptPriority):
        raise TypeError(
            f"strategy must be a PromptPriority, got {type(strategy).__name__}"
        )
    _require_group(group)
    return _check_score(strategy.score_prompt(group), strategy)


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------

# The built-ins are frozen dataclasses: parameters are validated once in
# __post_init__ and cannot be changed afterwards, so a strategy stored in a
# buffer (and recorded in its attestation log) keeps meaning the same thing.
# Validated values are stored with object.__setattr__, the standard way to
# normalise fields of a frozen dataclass (see rollout.py for the same pattern).


@dataclass(frozen=True)
class AdvantagePriority(PriorityStrategy):
    """|reward - mean_reward| + epsilon.

    Rollouts that did much better or much worse than their group carry
    the most gradient signal, so they are replayed more. A group whose
    rollouts all got the same reward has zero advantage everywhere and
    scores ``epsilon`` for each rollout.
    """

    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        object.__setattr__(self, "epsilon", _require_epsilon(self.epsilon))

    def score(self, rollout: Rollout, group: RolloutGroup) -> float:
        """How far this rollout's reward sits from its group's mean, plus epsilon."""
        return abs(rollout.reward - group.mean_reward) + self.epsilon


@dataclass(frozen=True)
class PassRateTargeting(PriorityStrategy, PromptPriority):
    """Gaussian bump around a target pass rate, plus epsilon.

    ``score = exp(-((pass_rate - target) / width)^2 / 2) + epsilon``

    Peaks at 1 + epsilon when the group's pass rate equals ``target``
    (reachable only if ``target`` is a multiple of ``1 / group.size``)
    and falls off like a normal curve with standard deviation ``width``.
    With the defaults (target 0.5, width 0.15) an all-pass or all-fail
    prompt scores about 4e-3: still sampleable, but rarely.

    The score is a property of the group, so every rollout in the group
    gets the same value and the class also works as a ``PromptPriority``.

    Parameters
    ----------
    target : float in [0, 1]
    width : float > 0
    epsilon : float >= 0
    """

    target: float = DEFAULT_PASS_RATE_TARGET
    width: float = DEFAULT_PASS_RATE_WIDTH
    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        target = _require_finite_float(self.target, "target")
        if not (0.0 <= target <= 1.0):
            raise ValueError(f"target must be in [0, 1], got {target!r}")
        width = _require_finite_float(self.width, "width")
        if width <= 0.0:
            raise ValueError(f"width must be > 0, got {width!r}")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "epsilon", _require_epsilon(self.epsilon))

    def score_prompt(self, group: RolloutGroup) -> float:
        """Gaussian bump: 1 at the target pass rate, falling off with ``width``."""
        z = (group.pass_rate - self.target) / self.width
        return math.exp(-0.5 * z * z) + self.epsilon

    def score(self, rollout: Rollout, group: RolloutGroup) -> float:
        """Same value for every rollout in the group; the score is group-level."""
        return self.score_prompt(group)


@dataclass(frozen=True)
class PassRateVariance(PriorityStrategy, PromptPriority):
    """pass_rate * (1 - pass_rate) + epsilon.

    The variance of a Bernoulli outcome with the group's pass rate. It is
    largest (0.25) at a 50% pass rate and zero when every rollout passed
    or every rollout failed, which is exactly the "no learning signal"
    case DAPO's dynamic sampling filters out.

    Group-level, so it also works as a ``PromptPriority``.
    """

    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        object.__setattr__(self, "epsilon", _require_epsilon(self.epsilon))

    def score_prompt(self, group: RolloutGroup) -> float:
        """Bernoulli variance of the pass rate: 0.25 at 50%, 0 at all-pass or all-fail."""
        p = group.pass_rate
        return p * (1.0 - p) + self.epsilon

    def score(self, rollout: Rollout, group: RolloutGroup) -> float:
        """Same value for every rollout in the group; the score is group-level."""
        return self.score_prompt(group)


__all__ = [
    "AdvantagePriority",
    "DEFAULT_EPSILON",
    "PassRateTargeting",
    "PassRateVariance",
    "PriorityStrategy",
    "PromptPriority",
    "validated_prompt_score",
    "validated_score",
]
