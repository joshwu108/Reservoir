"""
reservoir.rollout_quarantine — Reasoned bulk eviction for incident response.

When a reward function turns out to have been gamed, or a prompt set
leaked, an operator wants every affected example out of the buffer and a
record that says why::

    buf.quarantine(lambda rollout, group: group.source == "judge-v3",
                   reason="judge v3 rewarded the empty answer")

The predicate runs over every live ``(rollout, group)`` before anything is
mutated (a predicate that raises leaves the buffer unchanged) and must
return a ``bool``; every match is then evicted with reason
``"quarantine"``. The predicate sees copies, so it cannot change what the
buffer stores (a replayed log must reproduce the live state). Each such
``evict`` record carries two texts: the ``predicate`` (the caller's
description, or the predicate's source when none is given: the whole
statement containing a lambda, collapsed to one line with non-printable
characters replaced, so pass ``predicate_text`` for a log that will be
published) and the ``note`` (the ``reason`` argument). A quarantine needs
a log of format 3 or later; a buffer restored from an older log refuses
it before mutating anything, because that log's checker would reject the
record. The
checker validates their shape and the transcript reports them, so
``reservoir-transcript --blast-radius <digest>`` can say what a quarantined
example reached before it was removed. Neither text is something the log
can verify: they are the operator's statement, recorded.

The durable buffer logs the evicted positions and both texts, never the
callable, so recovery replays the same records without the predicate.
"""

from __future__ import annotations

import warnings

import copy
import inspect
from typing import Callable, Final, Optional, Sequence

from reservoir.rollout import Rollout, RolloutGroup

MAX_PREDICATE_LENGTH: Final[int] = 1024
"""Longest predicate text an evict record carries; ``reservoir.attest`` keeps the same bound."""

MAX_NOTE_LENGTH: Final[int] = 256
"""Longest quarantine note; the same bound as a ``source`` tag."""

QUARANTINE_FORMAT: Final[int] = 3
"""The first attestation log format whose checker accepts quarantine records."""

QuarantinePredicate = Callable[[Rollout, RolloutGroup], bool]


def describe_predicate(predicate: object) -> str:
    """One line of text naming ``predicate``: its source when available, else its name.

    Whitespace is collapsed and any other non-printable character becomes
    ``?``, so the result always passes ``validate_text``.
    """
    name = getattr(predicate, "__qualname__", None) or repr(predicate)
    try:
        text = inspect.getsource(predicate)  # type: ignore[arg-type]
    except (OSError, TypeError):
        text = name
    text = "".join(c if c.isprintable() else "?" for c in " ".join(text.split()))
    return (text.strip() or name)[:MAX_PREDICATE_LENGTH]


def validate_text(value: object, name: str, limit: int) -> str:
    """A non-empty printable string of at most ``limit`` characters, else ``ValueError``."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    if len(value) > limit:
        raise ValueError(f"{name} must be at most {limit} characters, got {len(value)}")
    if not value.isprintable() or not value.strip():
        raise ValueError(f"{name} must be printable and not only whitespace, got {value!r}")
    return value


def quarantine_text(predicate: object, predicate_text: Optional[str]) -> str:
    """The validated predicate text: the caller's, or one derived from the predicate."""
    text = describe_predicate(predicate) if predicate_text is None else predicate_text
    return validate_text(text, "predicate text", MAX_PREDICATE_LENGTH)


def _copy_group(group: RolloutGroup) -> RolloutGroup:
    """A group whose rollouts share nothing mutable with the stored ones (metadata is deep-copied)."""
    return RolloutGroup(
        prompt_id=group.prompt_id, model_version=group.model_version, source=group.source,
        is_success=group.is_success,
        rollouts=[Rollout(tokens=r.tokens, logprobs=r.logprobs, reward=r.reward,
                          metadata=copy.deepcopy(dict(r.metadata)) or None) for r in group.rollouts],
    )


def select_positions(buf, predicate: QuarantinePredicate) -> tuple[int, ...]:
    """Live slots of ``buf`` whose ``(rollout, group)`` the predicate selects, ascending.

    Evaluates every live entry before returning, so a predicate that
    raises or returns something other than a ``bool`` fails here, before
    any eviction. The predicate receives copies (one per group), so it
    cannot alter the stored rollouts' metadata.
    """
    if not callable(predicate):
        raise TypeError(f"quarantine predicate must be callable, got {type(predicate).__name__}")
    matched: list[int] = []
    copies: dict[int, RolloutGroup] = {}
    for position in buf.live_positions():
        rollout, group = buf.entry(position)
        if id(group) not in copies:
            copies[id(group)] = _copy_group(group)
        safe_group = copies[id(group)]
        safe_rollout = safe_group.rollouts[next(k for k, m in enumerate(group.rollouts) if m is rollout)]
        verdict = predicate(safe_rollout, safe_group)
        if not isinstance(verdict, bool):
            raise TypeError(
                f"quarantine predicate must return a bool, got {type(verdict).__name__} for slot {position}"
            )
        if verdict:
            matched.append(position)
    return tuple(matched)


def require_quarantine_format(buf) -> None:
    """A quarantine record needs log format 3 or later; refuse it in a buffer restored from an older log.

    A buffer with no attestation log at all writes no record anywhere, so
    the reason and predicate text are lost; that is allowed but warned.
    """
    fmt = buf._attester.log_format
    if fmt is None:
        warnings.warn(
            "quarantine on a buffer without an attestation log keeps no record of the predicate or reason; "
            "pass attest=<path> to the buffer if the incident must be auditable",
            RuntimeWarning, stacklevel=3,
        )
        return
    if fmt < QUARANTINE_FORMAT:
        raise ValueError(
            f"quarantine needs an attestation log of format {QUARANTINE_FORMAT} or later; this buffer continues a "
            f"format-{fmt} log, whose checker would reject the record"
        )


def run_quarantine(buf, predicate: QuarantinePredicate, reason: str, predicate_text: Optional[str]) -> tuple[int, ...]:
    """Validate, select, evict: the body of ``RolloutBuffer.quarantine``."""
    text = quarantine_text(predicate, predicate_text)
    validate_text(reason, "quarantine reason", MAX_NOTE_LENGTH)
    require_quarantine_format(buf)
    positions = select_positions(buf, predicate)
    apply_quarantine(buf, positions, text, reason)
    return positions


def apply_quarantine(buf, positions: Sequence[int], predicate_text: str, note: str) -> None:
    """Evict ``positions`` of ``buf`` with reason ``"quarantine"`` and the two texts.

    Everything is validated before the first eviction: both texts, the
    log format, and that every position holds a live entry. Positions are
    evicted in ascending order, once each. A failing log write mid-way
    leaves the earlier slots evicted (as ``evict`` does); the durable
    buffer rebuilds from disk in that case.
    """
    text = validate_text(predicate_text, "predicate text", MAX_PREDICATE_LENGTH)
    why = validate_text(note, "quarantine reason", MAX_NOTE_LENGTH)
    require_quarantine_format(buf)
    slots = sorted({buf._to_index(p) for p in positions})
    for pos in slots:
        if pos not in buf._tree.entries:
            raise ValueError(f"quarantine: slot {pos} holds no live entry")
    for pos in slots:
        event = buf._tree.evict(pos, "quarantine")
        buf._release_slot(pos)
        buf._attester.record_write(event, buf._op_counter, quarantine=(text, why))


__all__ = [
    "MAX_NOTE_LENGTH",
    "MAX_PREDICATE_LENGTH",
    "QuarantinePredicate",
    "apply_quarantine",
    "describe_predicate",
    "quarantine_text",
    "require_quarantine_format",
    "run_quarantine",
    "select_positions",
    "validate_text",
]
