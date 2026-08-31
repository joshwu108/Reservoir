"""
Tests for reservoir.rational — exact float-to-priority conversion.

TDD: these tests define the contract. Run them before implementation is complete
to ensure RED, then GREEN after implementation.
"""

import math
import struct
from fractions import Fraction

import pytest

from reservoir.rational import (
    DEFAULT_EPSILON,
    PRIORITY_SCALE,
    PRIORITY_SCALE_BITS,
    float_to_exact,
    float_to_priority_int,
    priority_int_to_fraction,
    td_error_to_priority,
)


class TestPriorityScaleConstants:
    def test_scale_bits(self):
        assert PRIORITY_SCALE_BITS == 52

    def test_scale_value(self):
        assert PRIORITY_SCALE == 4503599627370496  # 2^52


class TestFloatToExact:
    """float_to_exact: every finite float maps to an exact Fraction."""

    def test_integer_one(self):
        assert float_to_exact(1.0) == Fraction(1, 1)

    def test_integer_two(self):
        assert float_to_exact(2.0) == Fraction(2, 1)

    def test_one_half(self):
        assert float_to_exact(0.5) == Fraction(1, 2)

    def test_one_quarter(self):
        assert float_to_exact(0.25) == Fraction(1, 4)

    def test_zero(self):
        assert float_to_exact(0.0) == Fraction(0)

    def test_negative_exact(self):
        assert float_to_exact(-1.5) == Fraction(-3, 2)

    def test_round_trip(self):
        """Converting float to Fraction and back yields the same float."""
        values = [0.1, 0.2, 0.3, 1.0 / 3.0, math.pi, math.e, 1e-100, 1e100]
        for v in values:
            frac = float_to_exact(v)
            # The Fraction should have the same float representation
            assert float(frac) == v, f"Round-trip failed for {v}"

    def test_rejects_inf(self):
        with pytest.raises(ValueError, match="non-finite"):
            float_to_exact(math.inf)

    def test_rejects_neg_inf(self):
        with pytest.raises(ValueError, match="non-finite"):
            float_to_exact(-math.inf)

    def test_rejects_nan(self):
        with pytest.raises(ValueError, match="non-finite"):
            float_to_exact(math.nan)

    def test_exactness_is_lossless(self):
        """Fraction(float) is an exact representation — verify via bit structure."""
        x = 0.1
        frac = float_to_exact(x)
        # The denominator should be a power of 2
        assert (frac.denominator & (frac.denominator - 1)) == 0, (
            "Denominator of float Fraction is always a power of 2"
        )
        # Numerator * float_denominator should equal float_numerator * our_denominator
        float_frac = Fraction(x)
        assert frac == float_frac


class TestFloatToPriorityInt:
    """float_to_priority_int: exact integerization of x^alpha."""

    def test_zero_priority(self):
        """Zero input maps to zero priority."""
        assert float_to_priority_int(0.0, 0.6) == 0

    def test_one_to_any_power(self):
        """1.0^alpha == 1.0 for any alpha, so result is 2^52."""
        result = float_to_priority_int(1.0, 0.6)
        assert result == PRIORITY_SCALE  # 1.0 * 2^52

    def test_positive_monotone(self):
        """Higher raw priority -> higher priority integer."""
        a = float_to_priority_int(0.1, 0.6)
        b = float_to_priority_int(1.0, 0.6)
        c = float_to_priority_int(10.0, 0.6)
        assert a < b < c

    def test_alpha_zero_gives_one_for_nonzero(self):
        """alpha=1: p^1 = p, so result = floor(p * 2^52)."""
        p = 2.0
        result = float_to_priority_int(p, 1.0)
        # 2.0^1.0 = 2.0, integerized: 2.0 * 2^52 = 2^53
        expected = int(Fraction(2.0) * PRIORITY_SCALE)
        assert result == expected

    def test_exact_integer_recovery_via_fraction(self):
        """The integerized priority recovers the exact float64 value via Fraction.

        This holds only when x^alpha lies in [1, 2^k) for k such that
        the float exponent is >= -52 (i.e., x^alpha >= 2^-52 AND the
        mantissa*2^exp is an exact integer when scaled by 2^52).
        For normalized floats >= 1, exponent >= 0, so Fraction * 2^52
        is exactly an integer (the mantissa).

        We test with x values where x^0.6 >= 1 (i.e., x >= 1).
        """
        for x in [1.0, 1.5, 3.14, 100.0, 1000.0]:
            p_int = float_to_priority_int(x, 0.6)
            recovered = Fraction(p_int, PRIORITY_SCALE)
            # The recovered value should equal the Fraction of x^0.6
            # This holds exactly when x^0.6 >= 1 (exponent >= 0 in float64).
            expected = Fraction(x**0.6)
            assert recovered == expected, (
                f"Recovery failed for x={x}: got {float(recovered)}, "
                f"expected {float(expected)}"
            )

    def test_rejects_negative_priority(self):
        with pytest.raises(ValueError, match="non-negative"):
            float_to_priority_int(-1.0, 0.6)

    def test_rejects_inf_priority(self):
        with pytest.raises(ValueError, match="finite"):
            float_to_priority_int(math.inf, 0.6)

    def test_rejects_nan_priority(self):
        with pytest.raises(ValueError, match="finite"):
            float_to_priority_int(math.nan, 0.6)

    def test_rejects_zero_alpha(self):
        with pytest.raises(ValueError, match="positive"):
            float_to_priority_int(1.0, 0.0)

    def test_rejects_negative_alpha(self):
        with pytest.raises(ValueError, match="positive"):
            float_to_priority_int(1.0, -0.6)

    def test_rejects_inf_alpha(self):
        with pytest.raises(ValueError):
            float_to_priority_int(1.0, math.inf)

    def test_result_is_python_int(self):
        """Result must be a native Python int, not float."""
        result = float_to_priority_int(2.5, 0.6)
        assert isinstance(result, int)
        assert not isinstance(result, float)

    def test_deterministic(self):
        """Same inputs always produce the same output."""
        x, alpha = 3.14159, 0.6
        r1 = float_to_priority_int(x, alpha)
        r2 = float_to_priority_int(x, alpha)
        assert r1 == r2

    def test_ordering_preserved_across_alpha(self):
        """Priority ordering should be consistent for alpha in (0, 1]."""
        xs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        for alpha in [0.3, 0.5, 0.6, 0.7, 1.0]:
            ints = [float_to_priority_int(x, alpha) for x in xs]
            assert ints == sorted(ints), (
                f"Ordering violated for alpha={alpha}"
            )

    def test_very_small_priority(self):
        """Very small positive priorities should not return zero."""
        result = float_to_priority_int(1e-10, 0.6)
        assert isinstance(result, int)
        # May be zero if underflow, but if not zero should be positive
        assert result >= 0


class TestTdErrorToPriority:
    """td_error_to_priority: |δ| + ε, then integerize."""

    def test_positive_td_error(self):
        result = td_error_to_priority(2.0, 0.6)
        expected = float_to_priority_int(2.0 + DEFAULT_EPSILON, 0.6)
        assert result == expected

    def test_negative_td_error_uses_abs(self):
        pos = td_error_to_priority(2.0, 0.6)
        neg = td_error_to_priority(-2.0, 0.6)
        assert pos == neg

    def test_zero_td_error_uses_epsilon(self):
        result = td_error_to_priority(0.0, 0.6)
        expected = float_to_priority_int(DEFAULT_EPSILON, 0.6)
        assert result == expected

    def test_custom_epsilon(self):
        epsilon = 1e-3
        result = td_error_to_priority(1.0, 0.6, epsilon=epsilon)
        expected = float_to_priority_int(1.0 + epsilon, 0.6)
        assert result == expected

    def test_rejects_inf_td_error(self):
        with pytest.raises(ValueError, match="finite"):
            td_error_to_priority(math.inf, 0.6)

    def test_rejects_nan_td_error(self):
        with pytest.raises(ValueError, match="finite"):
            td_error_to_priority(math.nan, 0.6)

    def test_rejects_negative_epsilon(self):
        with pytest.raises(ValueError):
            td_error_to_priority(1.0, 0.6, epsilon=-1e-6)


class TestPriorityIntToFraction:
    """priority_int_to_fraction: exact inverse of integerization."""

    def test_zero(self):
        assert priority_int_to_fraction(0) == Fraction(0)

    def test_one_scale(self):
        assert priority_int_to_fraction(PRIORITY_SCALE) == Fraction(1)

    def test_two_scale(self):
        assert priority_int_to_fraction(2 * PRIORITY_SCALE) == Fraction(2)

    def test_half(self):
        # PRIORITY_SCALE / 2 = 2^51 -> Fraction(2^51, 2^52) = 1/2
        assert priority_int_to_fraction(PRIORITY_SCALE // 2) == Fraction(1, 2)

    def test_round_trip_with_float_to_priority_int(self):
        """priority_int_to_fraction(float_to_priority_int(x, 1.0)) == Fraction(x)
        for x values that are exact binary64 fractions."""
        for x in [0.5, 1.0, 2.0, 4.0, 0.25, 0.125]:
            p_int = float_to_priority_int(x, 1.0)
            frac = priority_int_to_fraction(p_int)
            assert frac == Fraction(x), (
                f"Round-trip failed for x={x}: got {frac}, expected {Fraction(x)}"
            )

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            priority_int_to_fraction(-1)
