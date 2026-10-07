"""
reservoir_checker.staleness — Replays a telemetry record's staleness decisions from its declared inputs.

A telemetry record written under a ``StalenessPolicy`` carries the policy,
every draw's sequence log-ratio as a ``float.hex()`` string and one
decision per draw (kept with an exact scale, or declined with a reason).
This module recomputes the decisions from the policy, the log-ratios, the
sample record's exact importance weights and the ages of the live slots,
in the adapter's fixed order (non-finite ratio, age bound, ESS floor,
group-mass cap, legacy gate), and ``reservoir_checker.telemetry`` rejects
the record if any decision differs.

It mirrors ``reservoir.integrations._trl_staleness`` and imports nothing
from ``reservoir``, by design: the two must agree, and a bug shared by
both would be invisible, so ``tests/test_staleness_policy.py`` cross-checks
them on random inputs. ``exact_exp`` uses the ``decimal`` module at a fixed
precision, which is correctly rounded and so platform-independent.
"""

from __future__ import annotations

import decimal
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence

from reservoir_checker.decay_replay import CheckerError

EXP_DIGITS = 30
LOG_RATIO_CLAMP = 700.0
_EXP_CONTEXT = decimal.Context(prec=EXP_DIGITS, Emax=1000, Emin=-1000, rounding=decimal.ROUND_HALF_EVEN,
                               traps=[decimal.InvalidOperation, decimal.Overflow, decimal.DivisionByZero])
MAX_INTEGER_DIGITS = 4000
_CANONICAL_INTEGER = re.compile(r"0|[1-9][0-9]{0,%d}" % (MAX_INTEGER_DIGITS - 1))
DECLINE_REASONS = ("drift", "age", "ess")
POLICY_FIELDS = ("max_age", "ess_floor", "mass_cap", "max_log_ratio", "max_declines_per_step", "group_size")
DECISION_FIELDS = ("draw", "row", "group", "reason", "scale_num", "scale_den")


@dataclass(frozen=True)
class Policy:
    """The declared policy, parsed."""

    max_age: Optional[int]
    ess_floor: Optional[float]
    mass_cap: Optional[float]
    max_log_ratio: Optional[float]
    max_declines_per_step: Optional[int]
    group_size: Optional[int]


@dataclass(frozen=True)
class Decision:
    draw: int
    row: int
    group: int
    reason: Optional[str]
    scale: Fraction


def exact_exp(r: float) -> Fraction:
    """Same arithmetic as the adapter: clamp, correctly rounded decimal ``exp`` in a fixed context, exact fraction."""
    if not math.isfinite(r):
        raise CheckerError(f"exact_exp needs a finite log-ratio, got {r!r}")
    r = min(max(float(r), -LOG_RATIO_CLAMP), LOG_RATIO_CLAMP)
    return Fraction(_EXP_CONTEXT.exp(decimal.Decimal(r)))


def canonical_int(value: object, where: str) -> int:
    """A decimal integer string with no sign, no leading zero and at most ``MAX_INTEGER_DIGITS`` digits."""
    if not isinstance(value, str) or _CANONICAL_INTEGER.fullmatch(value) is None:
        raise CheckerError(f"{where} must be a canonical decimal integer string of at most {MAX_INTEGER_DIGITS} digits")
    return int(value)


def hex_float(value: object, where: str, allow_non_finite: bool = False) -> float:
    """Parse a canonical ``float.hex()`` string; non-finite values only where allowed."""
    if not isinstance(value, str):
        raise CheckerError(f"{where} must be a float.hex() string, got {value!r}")
    try:
        parsed = float.fromhex(value)
    except (ValueError, OverflowError) as exc:
        raise CheckerError(f"{where} is not a hexadecimal float: {exc}") from exc
    if parsed.hex() != value:
        raise CheckerError(f"{where} must be the canonical float.hex() spelling")
    if not allow_non_finite and not math.isfinite(parsed):
        raise CheckerError(f"{where} must be finite")
    return parsed


def _optional_int(value: object, where: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CheckerError(f"{where} must be a non-negative int or null, got {value!r}")
    return value


def _optional_hex(record: dict, name: str, where: str, low: float, low_inclusive: bool, high: Optional[float]) -> Optional[float]:
    value = record.get(name)
    if value is None:
        return None
    parsed = hex_float(value, f"{where}: policy.{name}")
    if parsed < low or (parsed == low and not low_inclusive) or (high is not None and parsed > high):
        raise CheckerError(f"{where}: policy.{name}={parsed!r} is out of range")
    return parsed


def parse_policy(record: object, where: str) -> Policy:
    """Validate the ``policy`` field; every key is required (``null`` for an unused stage)."""
    if not isinstance(record, dict) or set(record) != set(POLICY_FIELDS):
        raise CheckerError(f"{where}: policy must be an object with exactly the fields {list(POLICY_FIELDS)}")
    policy = Policy(
        max_age=_optional_int(record["max_age"], f"{where}: policy.max_age"),
        ess_floor=_optional_hex(record, "ess_floor", where, 0.0, False, 1.0),
        mass_cap=_optional_hex(record, "mass_cap", where, 0.0, True, None),
        max_log_ratio=_optional_hex(record, "max_log_ratio", where, 0.0, False, None),
        max_declines_per_step=_optional_int(record["max_declines_per_step"], f"{where}: policy.max_declines_per_step"),
        group_size=_optional_int(record["group_size"], f"{where}: policy.group_size"),
    )
    if policy.group_size == 0:
        raise CheckerError(f"{where}: policy.group_size must be positive or null")
    if all(v is None for v in (policy.max_age, policy.ess_floor, policy.mass_cap, policy.max_log_ratio)):
        raise CheckerError(f"{where}: policy has no stage configured; an inactive policy is not recorded")
    return policy


def parse_decisions(value: object, n_draws: int, where: str) -> list[Decision]:
    """Validate the ``decisions`` list: one per draw, in draw order, well-formed and reduced."""
    if not isinstance(value, list) or len(value) != n_draws:
        raise CheckerError(f"{where}: decisions must list one entry per draw ({n_draws})")
    out: list[Decision] = []
    for k, entry in enumerate(value):
        w = f"{where}, decision {k}"
        if not isinstance(entry, dict) or set(entry) != set(DECISION_FIELDS):
            raise CheckerError(f"{w}: must be an object with exactly the fields {list(DECISION_FIELDS)}")
        if entry["draw"] != k or isinstance(entry["draw"], bool):
            raise CheckerError(f"{w}: draw must be {k}")
        row = _optional_int(entry["row"], f"{w}: row")
        group = _optional_int(entry["group"], f"{w}: group")
        if row is None or group is None:
            raise CheckerError(f"{w}: row and group are required")
        reason = entry["reason"]
        if reason is not None and reason not in DECLINE_REASONS:
            raise CheckerError(f"{w}: reason must be null or one of {DECLINE_REASONS}, got {reason!r}")
        num, den = canonical_int(entry["scale_num"], f"{w}: scale_num"), canonical_int(entry["scale_den"], f"{w}: scale_den")
        scale = Fraction(num, den) if den else None
        if scale is None or scale <= 0 or scale > 1 or (scale.numerator, scale.denominator) != (num, den):
            raise CheckerError(f"{w}: scale must be a reduced fraction in (0, 1], got {num}/{den}")
        if reason is not None and scale != 1:
            raise CheckerError(f"{w}: a declined draw carries scale 1")
        out.append(Decision(k, row, group, reason, scale))
    rows = [d.row for d in out]
    if len(set(rows)) != len(rows):
        raise CheckerError(f"{where}: decisions name a row twice")
    return out


def replay_decisions(
    policy: Policy, *, ratios: Sequence[float], is_weights: Sequence[Fraction], ages: Sequence[int],
    rows: Sequence[int], groups: Sequence[int],
) -> list[tuple[int, int, int, Optional[str], Fraction]]:
    """The adapter's four stages, from the declared inputs; ``(draw, row, group, reason, scale)`` per draw.

    Raises ``CheckerError`` as soon as the declines exceed the policy's
    ``max_declines_per_step`` (the adapter would have raised), so a hostile
    record cannot make the exact ESS loop run long.
    """
    n = len(ratios)
    cap = policy.max_declines_per_step
    reasons: list[Optional[str]] = [None if math.isfinite(r) else "drift" for r in ratios]
    if policy.max_age is not None:
        for k in range(n):
            if reasons[k] is None and ages[k] > policy.max_age:
                reasons[k] = "age"
    declined = sum(1 for r in reasons if r is not None)
    if cap is not None and declined > cap:
        raise CheckerError(f"{declined} declines exceed the policy's max_declines_per_step={cap}; the adapter would have raised")
    if policy.ess_floor is not None:
        floor = Fraction(policy.ess_floor)
        kept = [k for k in range(n) if reasons[k] is None]
        total = sum((is_weights[k] for k in kept), Fraction(0))
        squares = sum((is_weights[k] * is_weights[k] for k in kept), Fraction(0))
        while len(kept) > 1 and total * total < floor * len(kept) * squares:
            victim = max(kept, key=lambda k: (abs(ratios[k]), -k))
            reasons[victim] = "ess"
            kept.remove(victim)
            total -= is_weights[victim]
            squares -= is_weights[victim] * is_weights[victim]
            declined += 1
            if cap is not None and declined > cap:
                raise CheckerError(f"{declined} declines exceed the policy's max_declines_per_step={cap}; the adapter would have raised")
    scales = [Fraction(1)] * n
    if policy.mass_cap is not None:
        cap_per_row = 1 + Fraction(policy.mass_cap)
        members: dict[int, list[int]] = {}
        for k in range(n):
            if reasons[k] is None:
                members.setdefault(groups[k], []).append(k)
        for ks in members.values():
            mass = sum((is_weights[k] * exact_exp(ratios[k]) for k in ks), Fraction(0))
            cap = cap_per_row * len(ks)
            if mass > cap:
                for k in ks:
                    scales[k] = cap / mass
    if policy.max_log_ratio is not None:
        for k in range(n):
            if reasons[k] is None and abs(ratios[k]) > policy.max_log_ratio:
                reasons[k] = "drift"
                scales[k] = Fraction(1)
    return [(k, int(rows[k]), int(groups[k]), reasons[k], scales[k]) for k in range(n)]


def verify_decisions(
    policy: Policy, declared: list[Decision], *, ratios: Sequence[float], is_weights: Sequence[Fraction],
    ages: Sequence[int], where: str,
) -> None:
    """Replay the policy over the declared rows and groups; raise on the first decision that differs."""
    if policy.group_size is not None:
        for d in declared:
            if d.group != d.row // policy.group_size:
                raise CheckerError(f"{where}, decision {d.draw}: row {d.row} is in group {d.row // policy.group_size} "
                                   f"of size {policy.group_size}, not {d.group}")
    try:
        expected = replay_decisions(
            policy, ratios=ratios, is_weights=is_weights, ages=ages,
            rows=[d.row for d in declared], groups=[d.group for d in declared],
        )
    except CheckerError as exc:
        raise CheckerError(f"{where}: {exc}") from None
    for d, (_, _, _, reason, scale) in zip(declared, expected):
        if d.reason != reason:
            raise CheckerError(f"{where}, decision {d.draw}: declared {d.reason or 'kept'} but the policy gives "
                               f"{reason or 'kept'}")
        if d.scale != scale:
            raise CheckerError(f"{where}, decision {d.draw}: declared scale {d.scale} but the policy gives {scale}")
    declined = sum(1 for d in declared if d.reason is not None)
    if policy.max_declines_per_step is not None and declined > policy.max_declines_per_step:
        raise CheckerError(f"{where}: {declined} declines exceed the policy's max_declines_per_step="
                           f"{policy.max_declines_per_step}; the adapter would have raised")


__all__ = [
    "MAX_INTEGER_DIGITS",
    "canonical_int",
    "DECISION_FIELDS",
    "DECLINE_REASONS",
    "EXP_DIGITS",
    "LOG_RATIO_CLAMP",
    "POLICY_FIELDS",
    "Decision",
    "Policy",
    "exact_exp",
    "hex_float",
    "parse_decisions",
    "parse_policy",
    "replay_decisions",
    "verify_decisions",
]
