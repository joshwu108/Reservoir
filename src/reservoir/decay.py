"""
reservoir.decay — Exact age-decayed priorities that fit in unsigned 64-bit.

Pure functions only: no state, no RNG, no mutation of inputs. See
docs/design.md §7 for the derivation, rejected alternatives and the
attestation proposal.

DECLARED SEMANTICS
------------------
Parameters (all integers, fixed at construction of DecayParams):
    h = half_life          model versions per halving of the sampling weight
    A = max_policy_age     an entry is live iff current_version - t <= A
    P = priority_bits      base priorities satisfy 0 <= q < 2^P
    Q = priority_frac_bits fixed-point fraction bits of q
    F = table_frac_bits    fraction bits of the decay table

1. Quantization (the only float boundary). A finite raw priority x >= 0
   (already alpha-exponentiated, see rational.py) becomes

       q = 0                       if x == 0
       q = max(1, floor(x * 2^Q))  if x > 0, via fractions.Fraction

   so every positive priority is sampleable and only an exact zero is
   excluded. q >= 2^P is a ValueError, never a wrap or a clamp.

2. Decay table. For k in [0, h):

       T[k] = floor(2^(k/h) * 2^F)

   with 2^(k/h) the TRUE REAL value. T[k] is the unique integer with
   T[k]^h <= 2^(k + F*h) < (T[k] + 1)^h, computed in integer arithmetic.
   No float and no libm call is involved, so any machine derives the
   same table.

3. Weight. An entry with base priority q written at model version t has
   epoch E = t // h and phase k = t mod h. Its absolute weight is

       W = floor(q * T[k] / 2^F) * 2^E

   The product is rounded by FLOOR, once, before the epoch shift.

4. Distribution. Over the live entries, P(i) = W_i / sum_j W_j, exactly.
   Age decay is obtained by inflating newer entries instead of shrinking
   older ones: W_i / W_j depends only on (q, t) of the two entries, so the
   current version never touches a stored weight.

5. Stored value. The tree stores W relative to a base epoch B:

       leaf = floor(q * T[k] / 2^F) << (E - B),   0 <= E - B <= max_shift

   Advancing B by d (a rebase) is an exact right shift by d of every leaf
   and therefore of every internal node.

BIT BUDGET
----------
    tree:     P + INTRA_EPOCH_BITS + max_shift + ceil(log2(capacity)) <= 64
    product:  P + F + INTRA_EPOCH_BITS <= 64
    max_shift = ceil(A / h) + rebase_slack

Parameters that violate either inequality raise ValueError when DecayParams
is constructed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Final, Sequence

# Width of every tree node and of the intermediate product q * T[k].
UINT64_BITS: Final[int] = 64
UINT64_LIMIT: Final[int] = 1 << UINT64_BITS

# Base priority q is a P-bit unsigned fixed-point number with Q fraction bits.
DEFAULT_PRIORITY_BITS: Final[int] = 32
DEFAULT_PRIORITY_FRAC_BITS: Final[int] = 16

# Smallest q a positive raw priority can quantize to; keeps it sampleable.
MIN_POSITIVE_PRIORITY: Final[int] = 1

# T[k] has F fraction bits; with F = 31 every T[k] fits in uint32.
DEFAULT_TABLE_FRAC_BITS: Final[int] = 31

# T[k] / 2^F lies in [1, 2): the phase within an epoch costs one bit.
INTRA_EPOCH_BITS: Final[int] = 1

# Table construction is O(h) big-integer powers with O(F*h)-bit operands.
MAX_HALF_LIFE: Final[int] = 1024


def _require_int(value: object, name: str, minimum: int) -> int:
    """Return value if it is a true int >= minimum, else raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_uint64(value: int, name: str) -> int:
    """Invariant check: value must fit in uint64."""
    if not (0 <= value < UINT64_LIMIT):
        raise AssertionError(f"{name}={value} does not fit in uint64")
    return value


@dataclass(frozen=True)
class DecayParams:
    """Validated decay configuration. Construction fails on any bad value.

    Parameters
    ----------
    half_life : int
        Model versions per halving of the sampling weight, in [1, MAX_HALF_LIFE].
    max_policy_age : int
        Maximum age, in model versions, of a live entry. Must be >= 0.
    capacity : int
        Maximum number of leaves. Must be >= 1.
    priority_bits, priority_frac_bits, table_frac_bits : int
        P, Q and F of the module docstring.
    rebase_slack : int
        Extra epochs of shift headroom. A rebase is needed once every
        (rebase_slack + 1) * half_life versions; each unit costs one bit.

    Raises
    ------
    ValueError
        On a non-integer or out-of-range field, or a bit-budget violation.
    """

    half_life: int
    max_policy_age: int
    capacity: int
    priority_bits: int = DEFAULT_PRIORITY_BITS
    priority_frac_bits: int = DEFAULT_PRIORITY_FRAC_BITS
    table_frac_bits: int = DEFAULT_TABLE_FRAC_BITS
    rebase_slack: int = 0

    def __post_init__(self) -> None:
        _require_int(self.half_life, "half_life", 1)
        if self.half_life > MAX_HALF_LIFE:
            raise ValueError(
                f"half_life must be <= {MAX_HALF_LIFE}, got {self.half_life}; "
                f"use a coarser model-version unit"
            )
        _require_int(self.max_policy_age, "max_policy_age", 0)
        _require_int(self.capacity, "capacity", 1)
        _require_int(self.priority_bits, "priority_bits", 1)
        _require_int(self.priority_frac_bits, "priority_frac_bits", 0)
        _require_int(self.table_frac_bits, "table_frac_bits", 1)
        _require_int(self.rebase_slack, "rebase_slack", 0)
        if self.priority_frac_bits > self.priority_bits:
            raise ValueError(
                f"priority_frac_bits ({self.priority_frac_bits}) must be <= "
                f"priority_bits ({self.priority_bits})"
            )
        if self.product_bits > UINT64_BITS:
            raise ValueError(
                f"intermediate product needs {self.product_bits} bits > "
                f"{UINT64_BITS}: priority_bits ({self.priority_bits}) + "
                f"table_frac_bits ({self.table_frac_bits}) + {INTRA_EPOCH_BITS}"
            )
        if self.tree_bits > UINT64_BITS:
            raise ValueError(
                f"bit budget exceeded: priority_bits ({self.priority_bits}) + "
                f"{INTRA_EPOCH_BITS} + max_shift ({self.max_shift}) + "
                f"capacity_bits ({self.capacity_bits}) = {self.tree_bits} > "
                f"{UINT64_BITS}. Reduce max_policy_age / half_life, capacity, "
                f"rebase_slack or priority_bits."
            )

    @property
    def max_shift(self) -> int:
        """Largest epoch shift of a stored leaf: ceil(A / h) + rebase_slack."""
        return -(-self.max_policy_age // self.half_life) + self.rebase_slack

    @property
    def capacity_bits(self) -> int:
        """ceil(log2(capacity)); also covers the tree's power-of-two rounding."""
        return (self.capacity - 1).bit_length()

    @property
    def product_bits(self) -> int:
        """Bits of the intermediate product q * T[k]."""
        return self.priority_bits + self.table_frac_bits + INTRA_EPOCH_BITS

    @property
    def tree_bits(self) -> int:
        """Bits sufficient for the root of a full worst-case tree."""
        return (
            self.priority_bits
            + INTRA_EPOCH_BITS
            + self.max_shift
            + self.capacity_bits
        )


def _floor_root_of_power_of_two(exponent: int, degree: int, hint: int) -> int:
    """Return floor(2^(exponent/degree)), correcting hint by exact comparison.

    The hint affects only the number of iterations, never the result.
    """
    target = 1 << exponent
    candidate = hint
    while candidate**degree > target:
        candidate -= 1
    while (candidate + 1) ** degree <= target:
        candidate += 1
    return candidate


def _bisect_root_of_power_of_two(exponent: int, degree: int, low: int, high: int) -> int:
    """Return floor(2^(exponent/degree)), known to lie in [low, high)."""
    target = 1 << exponent
    while high - low > 1:
        middle = (low + high) // 2
        if middle**degree <= target:
            low = middle
        else:
            high = middle
    return low


@lru_cache(maxsize=64)
def _decay_table_cached(half_life: int, table_frac_bits: int) -> tuple[int, ...]:
    one = 1 << table_frac_bits
    if half_life == 1:
        return (one,)
    scale_exponent = table_frac_bits * half_life
    step = _bisect_root_of_power_of_two(
        1 + scale_exponent, half_life, one, one << INTRA_EPOCH_BITS
    )
    entries = [one, step]
    for k in range(2, half_life):
        hint = (entries[-1] * step) >> table_frac_bits
        entries.append(
            _floor_root_of_power_of_two(k + scale_exponent, half_life, hint)
        )
    return tuple(entries)


def decay_table(half_life: int, table_frac_bits: int) -> tuple[int, ...]:
    """Return (T[0], ..., T[h-1]) with T[k] = floor(2^(k/h) * 2^F).

    Computed entirely in integer arithmetic; T[k] is the floor of the true
    real value, certified by T[k]^h <= 2^(k + F*h) < (T[k] + 1)^h.

    Raises
    ------
    ValueError
        If half_life is not an int in [1, MAX_HALF_LIFE] or table_frac_bits
        is not an int in [1, 64 - INTRA_EPOCH_BITS].
    """
    _require_int(half_life, "half_life", 1)
    if half_life > MAX_HALF_LIFE:
        raise ValueError(f"half_life must be <= {MAX_HALF_LIFE}, got {half_life}")
    _require_int(table_frac_bits, "table_frac_bits", 1)
    if table_frac_bits > UINT64_BITS - INTRA_EPOCH_BITS:
        raise ValueError(
            f"table_frac_bits must be <= {UINT64_BITS - INTRA_EPOCH_BITS}, "
            f"got {table_frac_bits}"
        )
    return _decay_table_cached(half_life, table_frac_bits)


def quantize_priority(x: float, params: DecayParams) -> int:
    """Quantize a raw priority to q. The float boundary.

    q = 0 if x == 0, else max(MIN_POSITIVE_PRIORITY, floor(x * 2^Q)), so a
    positive priority is always sampleable. Exact zero stays zero.

    Raises
    ------
    ValueError
        If x is not a finite non-negative number, or q >= 2^priority_bits.
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"Raw priority must be a real number, got {x!r}")
    if not math.isfinite(x):
        raise ValueError(f"Raw priority must be finite, got {x!r}")
    if x < 0:
        raise ValueError(f"Raw priority must be non-negative, got {x!r}")
    if x == 0:
        return 0
    quantized = max(
        MIN_POSITIVE_PRIORITY,
        math.floor(Fraction(x) * (1 << params.priority_frac_bits)),
    )
    if quantized >= 1 << params.priority_bits:
        integer_bits = params.priority_bits - params.priority_frac_bits
        raise ValueError(
            f"Raw priority {x!r} is outside the representable range "
            f"[0, 2^{integer_bits}) of {integer_bits}.{params.priority_frac_bits} "
            f"fixed point"
        )
    return quantized


def inflated_priority(
    base_priority_int: int,
    entry_version: int,
    base_epoch: int,
    params: DecayParams,
) -> int:
    """Return the tree leaf floor(q * T[k] / 2^F) << (E - base_epoch).

    Raises
    ------
    ValueError
        If q is outside [0, 2^P), a version or epoch is negative, the entry
        is older than the base epoch (it must be evicted, not stored), or
        its shift exceeds params.max_shift (a rebase is due first).
    """
    _require_int(base_priority_int, "base_priority_int", 0)
    if base_priority_int >= 1 << params.priority_bits:
        raise ValueError(
            f"base_priority_int must be < 2^{params.priority_bits}, "
            f"got {base_priority_int}"
        )
    _require_int(entry_version, "entry_version", 0)
    _require_int(base_epoch, "base_epoch", 0)

    epoch, phase = divmod(entry_version, params.half_life)
    shift = epoch - base_epoch
    if shift < 0:
        raise ValueError(
            f"entry_version {entry_version} (epoch {epoch}) is older than "
            f"base epoch {base_epoch}; the entry must be evicted"
        )
    if shift > params.max_shift:
        raise ValueError(
            f"epoch shift {shift} exceeds max_shift {params.max_shift}; "
            f"rebase before writing version {entry_version}"
        )

    table = decay_table(params.half_life, params.table_frac_bits)
    product = _require_uint64(base_priority_int * table[phase], "product")
    mantissa = product >> params.table_frac_bits
    return _require_uint64(mantissa << shift, "leaf")


def max_leaf_value(params: DecayParams) -> int:
    """Largest leaf any valid input can produce under params."""
    table = decay_table(params.half_life, params.table_frac_bits)
    largest_base = (1 << params.priority_bits) - 1
    mantissa = (largest_base * table[-1]) >> params.table_frac_bits
    return _require_uint64(mantissa << params.max_shift, "max leaf")


def max_tree_total(params: DecayParams) -> int:
    """Largest root value: a full tree of max_leaf_value leaves."""
    return _require_uint64(params.capacity * max_leaf_value(params), "max total")


def is_expired(entry_version: int, current_version: int, params: DecayParams) -> bool:
    """True iff the entry's age exceeds max_policy_age and it must be evicted.

    Raises
    ------
    ValueError
        If a version is negative or entry_version > current_version.
    """
    _require_int(entry_version, "entry_version", 0)
    _require_int(current_version, "current_version", 0)
    if entry_version > current_version:
        raise ValueError(
            f"entry_version {entry_version} is newer than "
            f"current_version {current_version}"
        )
    return current_version - entry_version > params.max_policy_age


def canonical_base_epoch(current_version: int, params: DecayParams) -> int:
    """Largest base epoch at which every live entry is representable.

    Equals max(0, current_version - max_policy_age) // half_life.
    """
    _require_int(current_version, "current_version", 0)
    oldest_live = max(0, current_version - params.max_policy_age)
    return oldest_live // params.half_life


def pending_rebase_shift(
    current_version: int, base_epoch: int, params: DecayParams
) -> int:
    """Epochs the base must advance before writing at current_version.

    Returns 0 while the newest epoch still fits in max_shift. Otherwise
    returns the advance to the canonical base epoch. Expired entries must
    be evicted before the returned shift is applied.

    Raises
    ------
    ValueError
        If an argument is negative or base_epoch is ahead of current_version.
    """
    _require_int(current_version, "current_version", 0)
    _require_int(base_epoch, "base_epoch", 0)
    newest_epoch = current_version // params.half_life
    if base_epoch > newest_epoch:
        raise ValueError(
            f"base_epoch {base_epoch} is ahead of current_version "
            f"{current_version} (epoch {newest_epoch})"
        )
    if newest_epoch - base_epoch <= params.max_shift:
        return 0
    return canonical_base_epoch(current_version, params) - base_epoch


def rebase_priority(inflated: int, shift: int) -> int:
    """Return inflated >> shift, refusing any shift that drops a set bit.

    Divisibility is necessary for an exact rebase but does not prove the
    entry is live: eviction is decided by version (is_expired), not here.

    Raises
    ------
    ValueError
        If inflated is not a uint64, shift is not in [0, 64), or
        inflated is not a multiple of 2^shift.
    """
    _require_int(inflated, "inflated", 0)
    if inflated >= UINT64_LIMIT:
        raise ValueError(f"inflated must fit in uint64, got {inflated}")
    _require_int(shift, "shift", 0)
    if shift >= UINT64_BITS:
        raise ValueError(f"shift must be < {UINT64_BITS}, got {shift}")
    if inflated & ((1 << shift) - 1):
        raise ValueError(
            f"rebase by {shift} would lose bits of {inflated}; "
            f"expired entries must be evicted before rebasing"
        )
    return inflated >> shift


def rebase_priorities(inflated: Sequence[int], shift: int) -> tuple[int, ...]:
    """Rebase every value; all-or-nothing. The input is not modified."""
    return tuple(rebase_priority(value, shift) for value in inflated)


def version_after_update(
    entry_version: int,
    current_version: int,
    reset_age_on_update: bool = False,
) -> int:
    """Version an entry carries after a priority update.

    Default keeps the original version: age is measured from collection,
    and an update changes only the base priority. With
    reset_age_on_update=True the entry is re-stamped at current_version.

    Raises
    ------
    ValueError
        If a version is negative or entry_version > current_version.
    """
    _require_int(entry_version, "entry_version", 0)
    _require_int(current_version, "current_version", 0)
    if entry_version > current_version:
        raise ValueError(
            f"entry_version {entry_version} is newer than "
            f"current_version {current_version}"
        )
    return current_version if reset_age_on_update else entry_version
