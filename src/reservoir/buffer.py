"""
reservoir.buffer — In-memory exact Prioritized Experience Replay buffer.

Implements the full Schaul et al. 2016 PER semantics:
    p_i = |δ_i| + ε
    P(i) = p_i^α / Σ_k p_k^α
    IS weight w_i = (N · P(i))^(-β) / max_j w_j

All priority arithmetic is exact integer or Fraction arithmetic.
The float64-once boundary is at p_i^α (see rational.py and docs/design.md).
IS weight computation also uses the float64-once boundary for (-β) exponentiation.

Randomness: keyed BLAKE2b via draw.py. No random/numpy/torch RNG.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, NamedTuple, Optional

from reservoir.draw import draw_uniform_below
from reservoir.rational import DEFAULT_EPSILON, float_to_priority_int, PRIORITY_SCALE
from reservoir.sumtree import ExactMinTree, ExactSumTree


class Transition:
    """A single experience transition stored in the buffer."""

    __slots__ = ("state", "action", "reward", "next_state", "done")

    def __init__(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def __repr__(self) -> str:
        return (
            f"Transition(state={self.state!r}, action={self.action!r}, "
            f"reward={self.reward!r}, done={self.done!r})"
        )


class SampledBatch(NamedTuple):
    """Result of a batch sample from the buffer."""

    indices: list[int]           # Logical positions in the buffer
    transitions: list[Transition]
    is_weights: list[Fraction]   # Exact IS weights, normalized to [0,1]
    priorities: list[int]        # Integerized priorities for each sampled transition
    draw_integers: list[int]     # The exact draw integers used for each sample
    root_total: int              # Tree total at sample time
    min_priority_int: int        # Minimum priority at sample time


class ExactPERBuffer:
    """Exact Prioritized Experience Replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions. Rounded up to next power of 2.
    alpha : float
        Priority exponent (Schaul et al. 2016). Typically 0.6.
    beta : float
        IS correction exponent. Typically 0.4 -> 1.0 over training.
    epsilon : float
        Priority offset (|δ| + ε). Prevents zero priorities.
    seed : int
        Base seed for deterministic draws.
    buffer_id : int
        Unique buffer ID (used in key derivation).
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = DEFAULT_EPSILON,
        seed: int = 0,
        buffer_id: int = 0,
    ) -> None:
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"alpha must be finite and positive, got {alpha!r}")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError(f"beta must be finite and non-negative, got {beta!r}")
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError(f"epsilon must be finite and non-negative, got {epsilon!r}")

        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.seed = seed
        self.buffer_id = buffer_id

        self._sum_tree = ExactSumTree(capacity)
        self._min_tree = ExactMinTree(capacity)
        self.capacity: int = self._sum_tree.capacity

        self._transitions: list[Optional[Transition]] = [None] * self.capacity
        self._size: int = 0
        self._write_pos: int = 0   # Next write position (circular)
        self._op_counter: int = 0  # Monotone counter for draw keying

        # Maximum priority seen so far; used to assign new transitions.
        # Initialized to priority of epsilon so empty buffers work correctly.
        self._max_prio: int = float_to_priority_int(self.epsilon, self.alpha)

    @property
    def size(self) -> int:
        """Number of transitions currently in the buffer."""
        return self._size

    def _next_op_counter(self) -> int:
        self._op_counter += 1
        return self._op_counter

    def _td_to_priority_int(self, td_error: float) -> int:
        """Convert TD-error to integerized priority."""
        raw = abs(td_error) + self.epsilon
        return float_to_priority_int(raw, self.alpha)

    def insert(
        self,
        transition: Transition,
        td_error: Optional[float] = None,
    ) -> int:
        """Insert a transition into the buffer.

        Parameters
        ----------
        transition : Transition
            The experience transition to store.
        td_error : float or None
            If provided, sets the priority to integerize(|td_error| + epsilon).
            If None, assigns the maximum current priority so new transitions
            are guaranteed to be sampled at least once.

        Returns
        -------
        int
            The logical position where the transition was stored.
        """
        if td_error is not None:
            if not math.isfinite(td_error):
                raise ValueError(f"td_error must be finite, got {td_error!r}")
            priority_int = self._td_to_priority_int(td_error)
        else:
            priority_int = self._max_prio

        pos = self._write_pos
        self._transitions[pos] = transition
        self._sum_tree.update(pos, priority_int)
        self._min_tree.update(pos, priority_int)

        if priority_int > self._max_prio:
            self._max_prio = priority_int

        if self._size < self.capacity:
            self._size += 1

        self._write_pos = (self._write_pos + 1) % self.capacity
        return pos

    def update_priority(self, position: int, td_error: float) -> None:
        """Update the priority of an existing transition.

        Parameters
        ----------
        position : int
            The logical position (as returned by insert or in SampledBatch.indices).
        td_error : float
            New TD-error for this transition.
        """
        if not math.isfinite(td_error):
            raise ValueError(f"td_error must be finite, got {td_error!r}")
        if not (0 <= position < self.capacity):
            raise ValueError(f"Position {position} out of range")

        priority_int = self._td_to_priority_int(td_error)
        self._sum_tree.update(position, priority_int)
        self._min_tree.update(position, priority_int)

        if priority_int > self._max_prio:
            self._max_prio = priority_int

    def _compute_is_weight(
        self,
        priority_int: int,
        root_total: int,
        min_priority_int: int,
        n: int,
    ) -> Fraction:
        """Compute the normalized IS weight for one transition.

        w_i (unnormalized) = (N * P(i))^(-β)
                           = (N * priority_int / root_total)^(-β)

        Normalization: divide by max weight = (N * P_min)^(-β)
                                           = (N * min_priority / root_total)^(-β)

        Both exponentiation calls are the float64-once boundary.
        The ratio is exact Fraction arithmetic.
        """
        # N * P(i) and N * P_min as floats (float64-once for division)
        n_p_i_float = (n * priority_int) / root_total
        n_p_min_float = (n * min_priority_int) / root_total

        # (-beta) exponentiation: float64-once boundary
        w_i_float = n_p_i_float ** (-self.beta)
        w_max_float = n_p_min_float ** (-self.beta)

        # Convert to exact Fractions, take ratio
        w_i_frac = Fraction(w_i_float)
        w_max_frac = Fraction(w_max_float)

        return Fraction(w_i_frac, w_max_frac)

    def sample(self, batch_size: int) -> SampledBatch:
        """Sample a batch of transitions proportional to their priorities.

        Each sample uses a unique op_counter to ensure distinct draws.
        The op_counter for sample k in batch with batch_op=B is:
            sample_op = B * batch_size + k

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        SampledBatch

        Raises
        ------
        RuntimeError
            If buffer has fewer transitions than batch_size, or total is zero.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self._size < batch_size:
            raise RuntimeError(
                f"Buffer has {self._size} transitions, "
                f"cannot sample batch of {batch_size}"
            )

        root_total = self._sum_tree.total
        if root_total == 0:
            raise RuntimeError("Cannot sample: total priority is zero")

        min_priority_int = self._min_tree.minimum
        if min_priority_int >= ExactMinTree.INFINITY:
            raise RuntimeError("Cannot sample: no valid priorities in min-tree")

        batch_op = self._next_op_counter()
        n = self._size

        indices: list[int] = []
        transitions: list[Transition] = []
        is_weights: list[Fraction] = []
        priorities: list[int] = []
        draw_integers: list[int] = []

        for k in range(batch_size):
            sample_op = batch_op * batch_size + k

            draw_int = draw_uniform_below(
                root_total,
                seed=self.seed,
                buffer_id=self.buffer_id,
                op_counter=sample_op,
            )
            draw_integers.append(draw_int)

            pos = self._sum_tree.prefix_sum_locate(draw_int)
            priority_int = self._sum_tree.get(pos)

            w = self._compute_is_weight(priority_int, root_total, min_priority_int, n)

            indices.append(pos)
            transitions.append(self._transitions[pos])
            is_weights.append(w)
            priorities.append(priority_int)

        return SampledBatch(
            indices=indices,
            transitions=transitions,
            is_weights=is_weights,
            priorities=priorities,
            draw_integers=draw_integers,
            root_total=root_total,
            min_priority_int=min_priority_int,
        )

    def verify_trees(self) -> bool:
        """Verify the sum-tree and min-tree invariants."""
        self._sum_tree.verify_invariant()
        self._min_tree.verify_invariant()
        return True

    def __repr__(self) -> str:
        return (
            f"ExactPERBuffer(capacity={self.capacity}, size={self._size}, "
            f"alpha={self.alpha}, beta={self.beta})"
        )
