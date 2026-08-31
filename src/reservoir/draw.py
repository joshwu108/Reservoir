"""
reservoir.draw — Keyed BLAKE2b deterministic uniform-integer draw.

Produces a uniform integer in [0, N) keyed by (seed, buffer_id, op_counter).
No random, numpy.random, or torch RNG is used anywhere in this module.

## Uniformity Argument

We use the standard rejection-sampling construction for uniform integers:

1. Generate 256-bit blocks via BLAKE2b-256(key || block_counter).
2. Interpret each block as a 256-bit big-endian unsigned integer V.
3. Let K = floor(2^256 / N). Accept V if V < K*N, return V mod N.
4. Reject V if V >= K*N (at most 1/N fraction), try next block.

This gives exactly uniform distribution over [0, N) by the standard argument:
- The accepted values form K complete copies of [0, N), so every residue
  class mod N appears exactly K times in [0, K*N).
- The rejection rate per block is (2^256 mod N) / 2^256 < N / 2^256 < 2^-186
  for N <= 2^70 (typical buffer sizes).
- Expected number of hash evaluations is < 1 + N/2^256 ≈ 1.

For N up to 2^256 this construction is correct; larger N is not supported.

## Key Derivation

Key = BLAKE2b-256(b"res" || seed_bytes || sep || buffer_id_bytes || sep || counter_bytes)
where sep = b"\\x00" and all integers are encoded as big-endian 8-byte values.

The 'person' parameter of BLAKE2b is set to b'reservoir' (padded to 16 bytes)
to domain-separate this usage from any other BLAKE2b usage.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Final

# Maximum supported N: we draw 256-bit blocks.
# For N > 2^256, we'd need multiple blocks — not supported.
MAX_N_BITS: Final[int] = 256
MAX_N: Final[int] = 1 << MAX_N_BITS

# BLAKE2b output size in bytes
HASH_BYTES: Final[int] = 32  # 256 bits

# Domain separator for block index
_SEP: Final[bytes] = b"\xff"

_PERSON: Final[bytes] = b"reservoir\x00\x00\x00\x00\x00\x00\x00"  # 16 bytes


def _derive_key(seed: int, buffer_id: int, op_counter: int) -> bytes:
    """Derive a 32-byte key from (seed, buffer_id, op_counter).

    Parameters
    ----------
    seed, buffer_id, op_counter : int
        Must be non-negative integers that fit in 8 bytes.
    """
    # Encode as big-endian 8-byte unsigned integers
    seed_b = seed.to_bytes(8, "big")
    buf_b = buffer_id.to_bytes(8, "big")
    ctr_b = op_counter.to_bytes(8, "big")
    msg = seed_b + _SEP + buf_b + _SEP + ctr_b
    h = hashlib.blake2b(msg, digest_size=HASH_BYTES, person=_PERSON)
    return h.digest()


def _blake2b_block(key: bytes, block_index: int) -> int:
    """Generate one 256-bit block from key and block_index.

    Returns the block as a non-negative Python integer (big-endian).
    """
    block_b = block_index.to_bytes(8, "big")
    msg = key + block_b
    h = hashlib.blake2b(msg, digest_size=HASH_BYTES, person=_PERSON)
    return int.from_bytes(h.digest(), "big")


def draw_uniform_below(n: int, seed: int, buffer_id: int, op_counter: int) -> int:
    """Draw a uniform integer in [0, n) using rejection sampling.

    Parameters
    ----------
    n : int
        Upper bound (exclusive). Must satisfy 1 <= n <= 2^256.
    seed : int
        Global seed. Non-negative, fits in 8 bytes.
    buffer_id : int
        Unique buffer identifier. Non-negative, fits in 8 bytes.
    op_counter : int
        Monotonically increasing operation counter. Non-negative, fits in 8 bytes.

    Returns
    -------
    int
        A uniform integer in [0, n).

    Notes
    -----
    The draw is deterministic: identical (n, seed, buffer_id, op_counter) always
    produce the same result. This is deterministic replay identity, NOT a
    cryptographic security guarantee. See docs/nonclaims.md.
    """
    if not (1 <= n <= MAX_N):
        raise ValueError(
            f"n must be in [1, 2^256], got {n!r}"
        )
    if n == 1:
        return 0

    # Threshold for acceptance: accept V if V < threshold
    # threshold = floor(2^256 / n) * n
    # All V in [0, threshold) are uniform mod n.
    k = MAX_N // n  # floor(2^256 / n)
    threshold = k * n  # = floor(2^256 / n) * n

    key = _derive_key(seed, buffer_id, op_counter)
    block_idx = 0
    while True:
        v = _blake2b_block(key, block_idx)
        block_idx += 1
        if v < threshold:
            return v % n
        # Reject: try next block. Expected iterations < 2.


def draw_uniform_below_explicit(
    n: int, key: bytes, block_offset: int = 0
) -> tuple[int, int]:
    """Low-level draw using an explicit key (for testing).

    Returns (result, blocks_consumed) where blocks_consumed is the number
    of hash evaluations needed.

    Parameters
    ----------
    n : int
        Upper bound (exclusive). Must satisfy 1 <= n <= 2^256.
    key : bytes
        32-byte key.
    block_offset : int
        Starting block index.

    Returns
    -------
    tuple[int, int]
        (drawn_value, blocks_consumed)
    """
    if len(key) != HASH_BYTES:
        raise ValueError(f"Key must be {HASH_BYTES} bytes, got {len(key)}")
    if not (1 <= n <= MAX_N):
        raise ValueError(f"n must be in [1, 2^256]")

    if n == 1:
        return 0, 0

    k = MAX_N // n
    threshold = k * n

    blocks_consumed = 0
    block_idx = block_offset
    while True:
        v = _blake2b_block(key, block_idx)
        block_idx += 1
        blocks_consumed += 1
        if v < threshold:
            return v % n, blocks_consumed
