"""
reservoir_checker.entry_age — The declared minimum age of the slots a sample draws.

This module imports nothing from the rest of the reservoir package. A
``sample`` record may declare ``min_entry_age`` (a decimal string) and carry
on every draw the ``entry_version`` of the slot it drew. Then every draw's
version must be the one the decay replay holds for that slot, and the slot
must be at least ``min_entry_age`` versions old at the buffer's current
version, so a replayed row cannot be one the same step just stored. The
minimum is fixed for a run: declared from the first sample on, with the
same value on every sample. A draw may carry ``entry_version`` without a
declared minimum; it is still checked against the replay. Logs written
before the fields existed carry neither and are unaffected.
"""

from __future__ import annotations

from typing import Optional

from reservoir_checker.decay_replay import CheckerError, DecayState


def _decimal(value: object, where: str) -> int:
    canonical = isinstance(value, str) and value.isascii() and value.isdigit() and (value == "0" or value[0] != "0")
    if not canonical:
        raise CheckerError(f"{where} must be a non-negative integer encoded as a decimal string, got {value!r}")
    try:
        return int(value)
    except ValueError as exc:   # past the interpreter's digit limit
        raise CheckerError(f"{where} has {len(value)} digits") from exc


def has_entry_age_fields(record: dict) -> bool:
    samples = record.get("samples")
    drawn = isinstance(samples, list) and any(isinstance(s, dict) and "entry_version" in s for s in samples)
    return "min_entry_age" in record or drawn


def _declared_minimum(record: dict, idx: int, decay: DecayState) -> Optional[int]:
    """The record's minimum, held to the run's: present on every sample once declared, never changed."""
    if "min_entry_age" not in record:
        if decay.min_entry_age is not None:
            raise CheckerError(f"Record {idx}: an earlier sample declared min_entry_age {decay.min_entry_age}; "
                               "a later one cannot leave it out")
        decay.undeclared_samples += 1
        return None
    minimum = _decimal(record["min_entry_age"], f"Record {idx}: min_entry_age")
    if decay.min_entry_age is None and decay.undeclared_samples:
        raise CheckerError(f"Record {idx}: min_entry_age first declared after {decay.undeclared_samples} sample(s) "
                           "without it; a run declares its minimum from its first sample")
    if decay.min_entry_age is None:
        decay.min_entry_age = minimum
    elif minimum != decay.min_entry_age:
        raise CheckerError(f"Record {idx}: min_entry_age {minimum} differs from the {decay.min_entry_age} declared "
                           "earlier; the minimum is fixed for a run")
    return minimum


def verify_entry_ages(record: dict, idx: int, decay: DecayState) -> None:
    """Check each draw's declared slot version against the replay and against the declared minimum."""
    minimum = _declared_minimum(record, idx, decay)
    for k, draw in enumerate(record["samples"]):
        at = f"Record {idx}, sample {k}"
        if "entry_version" not in draw:
            if minimum is not None:
                raise CheckerError(f"{at}: a sample that declares min_entry_age needs entry_version on every draw")
            continue
        version = _decimal(draw["entry_version"], f"{at}: entry_version")
        live = decay.entries.get(draw["leaf_index"])
        if live is None or live[1] != version:
            held = "no live entry" if live is None else f"version {live[1]}"
            raise CheckerError(f"{at}: entry_version {version} but the replay holds {held} at slot {draw['leaf_index']}")
        age = decay.current_version - version
        if minimum is not None and age < minimum:
            raise CheckerError(f"{at}: draws slot {draw['leaf_index']} of version {version}, age {age} at version "
                               f"{decay.current_version}, below the declared min_entry_age {minimum}")
