"""
checker.draw — The keyed draw, re-derived from its definition.

This module imports NOTHING from src/reservoir. It reimplements the
deterministic uniform draw of ``reservoir.draw`` (docs/design.md §5) with
the standard library so that, when a log's ``decay_config`` records the
buffer's ``seed`` and ``buffer_id``, the checker can recompute every
``draw_int`` instead of only checking that it lands in the recorded
leaf's range. A forged draw that stays inside the right segment is then
rejected too, and two logs that differ only in their seed differ in their
first record rather than in their first sample.

Definition
----------
::

    key      = BLAKE2b-256(seed_8 || 0xff || buffer_id_8 || 0xff || counter_8, person=b"reservoir" + 7 zero bytes)
    block_j  = BLAKE2b-256(key || j_8, same person), as a 256-bit big-endian integer
    draw     = block_j mod n for the first j with block_j < floor(2^256 / n) * n

where ``*_8`` is the 8-byte big-endian encoding and ``counter`` is the
buffer's per-rollout draw counter: the number of rollouts drawn before
this one over the life of the buffer (``sum(len(samples))`` of all
earlier ``sample`` records, plus the position within the batch).
Rejection sampling makes the result exactly uniform on ``[0, n)``.
"""

from __future__ import annotations

import hashlib

from checker.decay_replay import CheckerError

_PERSON = b"reservoir" + b"\x00" * 7
_SEP = b"\xff"
_HASH_BYTES = 32
_MAX_N = 1 << 256
_MAX_BLOCKS = 64   # rejection probability per block is < 2^-186 for any n <= 2^70; this never binds


def _u64(value: int, name: str) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool) or not (0 <= value < 1 << 64):
        raise CheckerError(f"{name} must be an integer in [0, 2^64), got {value!r}")
    return value.to_bytes(8, "big")


def draw_uniform_below(n: int, seed: int, buffer_id: int, counter: int) -> int:
    """The buffer's draw for ``(seed, buffer_id, counter)`` below ``n``; CheckerError on bad input."""
    if not isinstance(n, int) or isinstance(n, bool) or not (1 <= n <= _MAX_N):
        raise CheckerError(f"draw bound must be an integer in [1, 2^256], got {n!r}")
    if n == 1:
        return 0
    key = hashlib.blake2b(
        _u64(seed, "seed") + _SEP + _u64(buffer_id, "buffer_id") + _SEP + _u64(counter, "draw counter"),
        digest_size=_HASH_BYTES, person=_PERSON,
    ).digest()
    threshold = (_MAX_N // n) * n
    for j in range(_MAX_BLOCKS):
        block = int.from_bytes(
            hashlib.blake2b(key + j.to_bytes(8, "big"), digest_size=_HASH_BYTES, person=_PERSON).digest(), "big"
        )
        if block < threshold:
            return block % n
    raise CheckerError("rejection sampling did not terminate; this cannot happen for a sane bound")
