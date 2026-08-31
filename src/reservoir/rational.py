"""
reservoir.rational — Exact float→integer conversion and priority integerization.

Design: Option (b) from docs/design.md.
  - p_i^alpha is computed in float64 once
  - The binary64 result is exactly integerized via fractions.Fraction
  - All downstream arithmetic uses Python integers and fractions.Fraction

The declared priority integer for a transition with raw priority x is:
    int(Fraction(x**alpha) * 2**52)

This is the DECLARED semantics of this buffer. See docs/design.md §2.
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction
from typing import Final

# Scale factor: 2^52 = mantissa bits of binary64 normalized floats.
# Every finite float in [1, 2) has denominator exactly 2^52 as a Fraction,
# so multiplying by PRIORITY_SCALE yields an exact integer for that range.
# For other exponent ranges, we compensate via the exponent; the Fraction
# conversion is always exact.
PRIORITY_SCALE_BITS: Final[int] = 52
PRIORITY_SCALE: Final[int] = 1 << PRIORITY_SCALE_BITS  # 4503599627370496

# Minimum epsilon added to raw |δ| before exponentiation (Schaul et al. 2016)
DEFAULT_EPSILON: Final[float] = 1e-6


def float_to_exact(x: float) -> Fraction:
    """Convert a finite float to its exact rational value.

    Every finite binary64 is a rational number: m * 2^e where m is a
    53-bit integer and e is an integer exponent. This conversion is exact.

    Raises ValueError for inf or nan.
    """
    if not math.isfinite(x):
        raise ValueError(f"Cannot convert non-finite float to exact rational: {x!r}")
    return Fraction(x)  # Python's Fraction(float) is exact for finite floats


def _extract_float_int(x: float) -> int:
    """Return the exact integer value of x * 2^52, for x a non-negative float.

    For normalized floats, this equals mantissa * 2^(exp-52) (possibly negative
    exponent). The result is computed via Fraction to avoid integer overflow or
    rounding.

    This is an internal helper for float_to_priority_int.
    """
    assert math.isfinite(x) and x >= 0, f"Expected non-negative finite float, got {x}"
    frac = Fraction(x)
    scaled = frac * PRIORITY_SCALE
    # scaled is a Fraction with denominator that is a power of 2.
    # Converting to int truncates. For x = m * 2^e (normalized float):
    #   frac = m * 2^(e - 52) (normalized, e >= 1)
    # or frac = m * 2^(e - 52) for subnormals too.
    # Multiplication by 2^52 may not always yield an integer if e < 0,
    # so we use floor to be precise about the declared semantics.
    return int(scaled)  # truncates toward zero; x >= 0 so this is floor


def float_to_priority_int(x: float, alpha: float) -> int:
    """Compute the integerized priority for raw priority x with exponent alpha.

    The exact integer priority is: floor(x^alpha * 2^52), where x^alpha is
    computed in IEEE 754 double precision.

    This is the float64-once boundary: x^alpha is computed in float64,
    then immediately frozen to an integer. All subsequent arithmetic is exact.

    Parameters
    ----------
    x : float
        Raw priority, must be finite and non-negative. Typically |δ_i| + ε.
    alpha : float
        PER exponent, typically 0.6. Must be finite and positive.

    Returns
    -------
    int
        Exact integer priority. Zero if x == 0.0.

    Raises
    ------
    ValueError
        If x is inf, nan, or negative; or if x^alpha overflows/underflows.
    """
    if not math.isfinite(x):
        raise ValueError(f"Raw priority must be finite, got {x!r}")
    if x < 0:
        raise ValueError(f"Raw priority must be non-negative, got {x!r}")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"Alpha must be finite and positive, got {alpha!r}")

    if x == 0.0:
        return 0

    powered = x**alpha  # float64 multiplication — the ONE allowed float op
    if not math.isfinite(powered):
        raise ValueError(
            f"x^alpha overflowed or underflowed: x={x!r}, alpha={alpha!r}, "
            f"result={powered!r}"
        )
    if powered == 0.0:
        return 0

    return _extract_float_int(powered)


def td_error_to_priority(td_error: float, alpha: float,
                          epsilon: float = DEFAULT_EPSILON) -> int:
    """Convert a TD-error to an integerized priority.

    Implements the Schaul et al. 2016 formula: p_i = |δ_i| + ε.
    Then applies float_to_priority_int(p_i, alpha).

    Parameters
    ----------
    td_error : float
        TD-error from the learning agent. May be negative or positive.
    alpha : float
        PER exponent.
    epsilon : float
        Additive offset to avoid zero priorities.

    Returns
    -------
    int
        Integerized priority.

    Raises
    ------
    ValueError
        If td_error or epsilon is non-finite.
    """
    if not math.isfinite(td_error):
        raise ValueError(f"TD-error must be finite, got {td_error!r}")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"Epsilon must be finite and non-negative, got {epsilon!r}")

    raw_priority = abs(td_error) + epsilon
    return float_to_priority_int(raw_priority, alpha)


def priority_int_to_fraction(priority_int: int) -> Fraction:
    """Convert an integerized priority back to its declared exact rational value.

    The declared value is priority_int / 2^52.

    This is used for IS weight computation and probability calculations.
    """
    if priority_int < 0:
        raise ValueError(f"Priority integer must be non-negative, got {priority_int}")
    return Fraction(priority_int, PRIORITY_SCALE)
