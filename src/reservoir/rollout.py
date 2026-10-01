"""
reservoir.rollout — Value types for LLM-RL replay entries.

In GRPO-style training the model generates several completions
("rollouts") for one prompt, each completion gets a reward, and the
update favours the completions that scored above the group average.
These two types describe that data so it can be stored in a replay
buffer and sampled again later:

- ``Rollout``: one completion. Holds the token ids, the log-probability
  the generating policy assigned to each token ("behavior logprobs"), and
  the scalar reward. The logprobs are kept because a rollout replayed
  under a *later* policy needs an importance ratio
  ``new_logprob / behavior_logprob`` to correct for being off-policy.
- ``RolloutGroup``: every rollout generated for one prompt at one model
  version, plus the group statistics (mean reward, pass rate, advantages)
  that priority strategies read.

Both are frozen dataclasses. All validation happens once, in
``__post_init__``, because construction is where external data (tokenizer
output, inference-engine logprobs, reward-model scores) enters the
library. Inputs are copied into tuples, so a caller mutating its own
lists afterwards cannot change a stored entry.

Example::

    group = RolloutGroup(
        prompt_id="gsm8k-0412",
        model_version=step,
        rollouts=[
            Rollout(tokens=ids, logprobs=lps, reward=r)
            for ids, lps, r in zip(completions, behavior_logprobs, rewards)
        ],
    )
    group.pass_rate      # fraction of rollouts with reward > 0
    group.advantages     # reward - mean_reward, per rollout

The replay buffer stores one sum-tree leaf per ``Rollout``; the
``RolloutGroup`` is shared metadata referenced by its rollouts.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence, Set, Sized
from dataclasses import dataclass, field
from functools import cached_property
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Final

_EMPTY_METADATA: Mapping[str, Any] = MappingProxyType({})

# Largest |reward| a Rollout accepts, about 3.3e150.
#
# Group statistics are computed in float64. Without a bound, two finite
# rewards near the float64 maximum (1.8e308) would overflow to infinity
# when subtracted or squared, and the error would surface lazily on first
# access to ``mean_reward`` or ``reward_std`` instead of at construction.
# With |reward| <= 2^500, a sum or difference of two rewards is at most
# 2^501 and its square at most 2^1002, both below the float64 limit of
# 2^1024. Real rewards (0/1 correctness, small shaped scores) are nowhere
# near this; the bound exists only so the statistics can never overflow.
MAX_ABS_REWARD: Final[float] = float(1 << 500)


def _is_real(value: object) -> bool:
    """True for any real scalar (Python int/float, numpy scalars), not bool."""
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_integral(value: object) -> bool:
    """True for any integer scalar (Python int, numpy ints), not bool."""
    return isinstance(value, Integral) and not isinstance(value, bool)


def _require_finite_float(value: object, name: str) -> float:
    """Convert a real scalar to a finite Python float, or raise ValueError.

    Rejects non-numeric values, bool, NaN, +/-inf, and Python ints too
    large to convert to a float (``float(10**400)`` raises OverflowError,
    which is translated to the same ValueError as every other bad input).
    """
    if not _is_real(value):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    try:
        as_float = float(value)  # type: ignore[arg-type]
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite real number, got {value!r}") from exc
    if not math.isfinite(as_float):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    return as_float


def _is_scalar_sequence(value: object) -> bool:
    """True for list/tuple/1-D numpy array style containers of scalars.

    Any sized iterable qualifies, so numpy arrays and torch tensors pass
    even though they are not ``collections.abc.Sequence``. Strings, bytes,
    mappings and sets are refused on purpose: iterating them would
    silently yield characters, keys, or elements in arbitrary order.
    Element types are checked separately by the caller.
    """
    if isinstance(value, (str, bytes, bytearray, Mapping, Set)):
        return False
    return isinstance(value, Iterable) and isinstance(value, Sized)


def _validate_tokens(tokens: object) -> tuple[int, ...]:
    """Return tokens as a non-empty tuple of non-negative Python ints.

    numpy integer scalars are accepted and converted with ``int()`` so the
    stored tuple never holds foreign numeric types.
    """
    if not _is_scalar_sequence(tokens):
        raise ValueError(f"tokens must be a sequence of ints, got {type(tokens).__name__}")
    out = tuple(tokens)  # type: ignore[call-overload]
    if not out:
        raise ValueError("tokens must be non-empty")
    for i, t in enumerate(out):
        if not _is_integral(t) or t < 0:
            raise ValueError(f"tokens[{i}] must be a non-negative int, got {t!r}")
    return tuple(int(t) for t in out)


def _validate_logprobs(logprobs: object, n_tokens: int) -> tuple[float, ...]:
    """Return logprobs as a tuple of finite Python floats, one per token.

    A log-probability is log(p) with p in (0, 1], so every value must be
    <= 0. Exactly 0.0 is allowed (p == 1, e.g. a forced token). A positive
    value is a data error upstream and would silently corrupt importance
    ratios, so it is rejected here.
    """
    if not _is_scalar_sequence(logprobs):
        raise ValueError(
            f"logprobs must be a sequence of floats, got {type(logprobs).__name__}"
        )
    raw = tuple(logprobs)  # type: ignore[call-overload]
    if len(raw) != n_tokens:
        raise ValueError(
            f"logprobs must have one entry per token: {len(raw)} logprobs "
            f"for {n_tokens} tokens"
        )
    out = tuple(_require_finite_float(lp, f"logprobs[{i}]") for i, lp in enumerate(raw))
    for i, lp in enumerate(out):
        if lp > 0.0:
            raise ValueError(f"logprobs[{i}] must be <= 0 (a log-probability), got {lp!r}")
    return out


def _validate_reward(reward: object) -> float:
    """Return the reward as a finite float with |reward| <= MAX_ABS_REWARD."""
    value = _require_finite_float(reward, "reward")
    if abs(value) > MAX_ABS_REWARD:
        raise ValueError(
            f"reward magnitude must be <= {MAX_ABS_REWARD:.3g}, got {value!r}"
        )
    return value


def _validate_metadata(metadata: object) -> Mapping[str, Any]:
    """Return metadata as a read-only mapping (shallow copy), or the shared empty one."""
    if metadata is None:
        return _EMPTY_METADATA
    if not isinstance(metadata, Mapping):
        raise ValueError(f"metadata must be a mapping, got {type(metadata).__name__}")
    return MappingProxyType(dict(metadata))


@dataclass(frozen=True)
class Rollout:
    """One completion with its behavior logprobs and reward.

    Parameters
    ----------
    tokens : sequence of int
        Generated token ids. Non-empty, each >= 0. Lists, tuples and 1-D
        numpy arrays are accepted; values are stored as Python ints.
    logprobs : sequence of float
        Log-probability of each token under the policy that generated it,
        same length as ``tokens``, every value finite and <= 0. Needed to
        compute the importance ratio when this rollout is replayed under a
        later policy.
    reward : float
        Finite scalar reward with |reward| <= ``MAX_ABS_REWARD``.
    metadata : mapping, optional
        Caller-defined extras (source dataset, sample index, ...). Stored
        as a read-only shallow copy: the mapping itself cannot be changed,
        but mutable values inside it still alias the caller's objects.
        Ignored by equality and hashing.

    Raises
    ------
    ValueError
        On any malformed input; the message names the field and, for
        sequences, the offending index.

    Example
    -------
    >>> r = Rollout(tokens=[15, 42, 7], logprobs=[-0.1, -2.3, -0.4], reward=1.0)
    >>> len(r)
    3
    """

    tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    reward: float
    metadata: Mapping[str, Any] = field(default=_EMPTY_METADATA, compare=False, hash=False)

    def __post_init__(self) -> None:
        tokens = _validate_tokens(self.tokens)
        logprobs = _validate_logprobs(self.logprobs, len(tokens))
        reward = _validate_reward(self.reward)
        metadata = _validate_metadata(self.metadata)
        # A frozen dataclass blocks ``self.x = ...`` even inside __post_init__,
        # so the validated, normalized copies are stored via object.__setattr__.
        # This is the standard pattern and the only place it is used.
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "logprobs", logprobs)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "metadata", metadata)

    def __len__(self) -> int:
        """Number of tokens."""
        return len(self.tokens)

    def __repr__(self) -> str:
        return f"Rollout(n_tokens={len(self.tokens)}, reward={self.reward!r})"


def default_is_success(rollout: Rollout) -> bool:
    """Default success test for ``RolloutGroup.pass_rate``: reward > 0.

    Correct for verifiable 0/1 rewards. For shaped or negative-baseline
    rewards pass your own predicate to ``RolloutGroup(is_success=...)``.
    """
    return rollout.reward > 0


@dataclass(frozen=True)
class RolloutGroup:
    """All rollouts generated for one prompt at one model version.

    Parameters
    ----------
    prompt_id : str
        Non-empty identifier of the prompt (dataset key, hash, ...).
    model_version : int
        Policy version, usually the training step, that generated the
        group. Must be >= 0. The replay buffer uses it to age entries.
    rollouts : sequence of Rollout
        Non-empty. Stored as a tuple.
    is_success : callable, optional
        ``Rollout -> bool`` used by ``pass_rate`` and ``n_success``.
        Default: ``reward > 0``. Ignored by equality and hashing, so two
        groups with the same data compare equal whatever predicate they
        carry.

    Statistics
    ----------
    Each statistic is computed on first access and cached for the life of
    the instance (``functools.cached_property`` works on frozen dataclasses
    because it writes to ``__dict__`` directly).

    ``advantages`` are *group-relative*: ``reward - mean_reward`` for each
    rollout, with no division by the standard deviation. A strategy that
    wants a magnitude applies ``abs`` itself.

    Example
    -------
    >>> g = RolloutGroup(
    ...     prompt_id="p1", model_version=3,
    ...     rollouts=[Rollout([1], [-0.1], 1.0), Rollout([2], [-0.2], 0.0)],
    ... )
    >>> g.pass_rate, g.mean_reward, g.advantages
    (0.5, 0.5, (0.5, -0.5))
    """

    prompt_id: str
    model_version: int
    rollouts: tuple[Rollout, ...]
    is_success: Callable[[Rollout], bool] = field(
        default=default_is_success, compare=False, hash=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ValueError(f"prompt_id must be a non-empty str, got {self.prompt_id!r}")
        if isinstance(self.model_version, bool) or not isinstance(self.model_version, int):
            raise ValueError(f"model_version must be an int, got {self.model_version!r}")
        if self.model_version < 0:
            raise ValueError(f"model_version must be >= 0, got {self.model_version}")
        if not isinstance(self.rollouts, Sequence) or isinstance(self.rollouts, (str, bytes)):
            raise ValueError(
                f"rollouts must be a sequence of Rollout, got {type(self.rollouts).__name__}"
            )
        rollouts = tuple(self.rollouts)
        if not rollouts:
            raise ValueError("rollouts must be non-empty")
        for i, r in enumerate(rollouts):
            if not isinstance(r, Rollout):
                raise ValueError(f"rollouts[{i}] must be a Rollout, got {type(r).__name__}")
        if not callable(self.is_success):
            raise ValueError(f"is_success must be callable, got {self.is_success!r}")
        # Frozen dataclass: see the note in Rollout.__post_init__.
        object.__setattr__(self, "rollouts", rollouts)

    def __len__(self) -> int:
        """Number of rollouts."""
        return len(self.rollouts)

    @property
    def size(self) -> int:
        """Number of rollouts in the group (same as ``len(group)``)."""
        return len(self.rollouts)

    @cached_property
    def rewards(self) -> tuple[float, ...]:
        """Rewards of the rollouts, in order."""
        return tuple(r.reward for r in self.rollouts)

    @cached_property
    def mean_reward(self) -> float:
        """Arithmetic mean of the rewards (``math.fsum`` for accuracy)."""
        return math.fsum(self.rewards) / self.size

    @cached_property
    def reward_std(self) -> float:
        """Population standard deviation of the rewards (divides by N, not N-1).

        Computed as ``hypot(deviations) / sqrt(N)``. ``math.hypot`` returns
        ``sqrt(sum(d**2))`` while rescaling internally, so it cannot
        overflow even when a single ``d**2`` would. Combined with
        ``MAX_ABS_REWARD`` this keeps the result finite for any input.
        """
        return math.hypot(*self.advantages) / math.sqrt(self.size)

    @cached_property
    def advantages(self) -> tuple[float, ...]:
        """``reward - mean_reward`` per rollout. Not divided by the std."""
        mean = self.mean_reward
        return tuple(r - mean for r in self.rewards)

    @cached_property
    def n_success(self) -> int:
        """Number of rollouts for which ``is_success`` returns True."""
        return sum(1 for r in self.rollouts if self.is_success(r))

    @cached_property
    def pass_rate(self) -> float:
        """``n_success / size``: fraction of successful rollouts, in [0, 1]."""
        return self.n_success / self.size

    def __repr__(self) -> str:
        return (
            f"RolloutGroup(prompt_id={self.prompt_id!r}, "
            f"model_version={self.model_version}, size={self.size})"
        )
