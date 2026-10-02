"""
reservoir.decayed_tree — Exact sum-tree with age-decayed priorities.

``DecayedPriorityTree`` is the piece that joins the pure functions in
``decay.py`` to the integer trees in ``sumtree.py``. It is the only object
in the library that holds, per leaf, the base priority ``q`` and the
model version ``t`` the entry was written at, together with the tree's
base epoch and current version. The rollout buffer sits on top of it and
knows nothing about epochs or shifts.

How age decay works here (full derivation in docs/design.md §7)
----------------------------------------------------------------
Older entries are not shrunk; newer entries are *inflated*. An entry's
stored leaf is::

    leaf = floor(q * T[t mod h] / 2^F) << (t // h - base_epoch)

so an entry written one half-life (``h`` versions) later has exactly
twice the weight of an equal-``q`` entry, and the current version never
touches a stored leaf. The shift grows with time, so every so often the
tree *rebases*: it right-shifts every node by ``d`` and adds ``d`` to the
base epoch. That is exact because stale entries are evicted first and
every surviving leaf is a multiple of ``2^d``.

Lifecycle a caller follows
--------------------------
1. ``advance(current_version)`` before writing or sampling at a new
   version. It evicts expired entries (``age > max_policy_age``) and, if
   the newest epoch no longer fits the bit budget, rebases. Both are
   reported in the returned ``AdvanceResult`` so the caller can log them.
2. ``write(position, q, entry_version)`` to insert or update a leaf.
3. ``evict(position, reason)`` to drop an entry early (capacity pressure,
   explicit removal).
4. ``prefix_sum_locate(draw)`` / ``leaf`` / ``total`` / ``minimum`` for
   sampling and IS-weight normalisation.

Every mutating method returns a frozen event describing exactly what
changed (old leaf, new leaf, q, t, base epoch) — the data the attestation
log records and the checker replays.

Mutability
----------
The tree arrays are mutated in place; that is this class's declared job,
as with ``ExactSumTree``. Everything it hands out (events, tuples of
positions) is immutable, and no method mutates its arguments.

Zero-weight entries
-------------------
An entry with ``q == 0`` is live (it occupies a slot and expires like
any other) but has leaf 0, so it can never be drawn. It is kept out of
the min-tree: the minimum is used to normalise importance-sampling
weights, and a zero minimum would make every weight infinite.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Mapping, Optional

from reservoir.decay import (
    DecayParams,
    inflated_priority,
    is_expired,
    pending_rebase_shift,
    rebase_priority,
)
from reservoir.sumtree import ExactMinTree, ExactSumTree

# Why an entry left the tree. Recorded in the event (and later the log):
#   "stale"     its age exceeded max_policy_age during advance()
#   "capacity"  the buffer was full and this was the oldest entry
#   "explicit"  the caller asked for it
EvictReason = Literal["stale", "capacity", "explicit"]
EVICT_REASONS: Final[frozenset[str]] = frozenset({"stale", "capacity", "explicit"})

WriteOp = Literal["insert", "update", "evict"]


@dataclass(frozen=True)
class WriteEvent:
    """One leaf changed. ``op`` is "insert" (slot was empty), "update" or "evict".

    For an evict, ``new_leaf`` is 0 and ``base_priority_int`` /
    ``entry_version`` describe the entry that was removed.
    """

    op: WriteOp
    position: int
    old_leaf: int
    new_leaf: int
    base_priority_int: int
    entry_version: int
    base_epoch: int
    reason: Optional[EvictReason] = None


@dataclass(frozen=True)
class RebaseEvent:
    """The base epoch advanced and every node was right-shifted.

    ``shifted`` lists ``(position, old_leaf, new_leaf)`` for every live
    position, in position order. Rebases are rare (once per
    ``(rebase_slack + 1) * half_life`` versions), so carrying the full
    list is cheap and lets a logger record per-leaf changes if it wants.
    """

    old_base_epoch: int
    new_base_epoch: int
    root_total_before: int
    root_total_after: int
    shifted: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class AdvanceResult:
    """What ``advance`` did: evictions first (position order), then an optional rebase."""

    old_version: int
    new_version: int
    evicted: tuple[WriteEvent, ...]
    rebase: Optional[RebaseEvent]


class DecayedPriorityTree:
    """Sum-tree plus min-tree whose leaves are exact age-decayed priorities.

    Parameters
    ----------
    params : DecayParams
        Validated decay configuration. The tree's capacity is
        ``params.capacity`` rounded up to a power of two; the bit budget
        in ``DecayParams`` already accounts for that rounding.
    """

    def __init__(self, params: DecayParams) -> None:
        if not isinstance(params, DecayParams):
            raise TypeError(f"params must be a DecayParams, got {type(params).__name__}")
        self._params = params
        self._sum = ExactSumTree(params.capacity)
        self._min = ExactMinTree(params.capacity)
        # (q, t) per live position. Absence means the slot is empty.
        self._entries: dict[int, tuple[int, int]] = {}
        self._base_epoch = 0
        self._current_version = 0
        assert (self._sum.capacity - 1).bit_length() <= params.capacity_bits

    # -- read-only state ---------------------------------------------------

    @property
    def params(self) -> DecayParams:
        """The decay configuration this tree was built with."""
        return self._params

    @property
    def capacity(self) -> int:
        """Number of leaves (a power of two)."""
        return self._sum.capacity

    @property
    def base_epoch(self) -> int:
        """Epoch the stored leaves are shifted relative to; advances on each rebase."""
        return self._base_epoch

    @property
    def current_version(self) -> int:
        """Latest version passed to ``advance``; entries are aged against it."""
        return self._current_version

    @property
    def total(self) -> int:
        """Sum of every leaf: the denominator of each sampling probability."""
        return self._sum.total

    @property
    def minimum(self) -> int:
        """Smallest positive leaf, or ``ExactMinTree.INFINITY`` if there is none."""
        return self._min.minimum

    @property
    def live_count(self) -> int:
        """Number of positions holding an entry (zero-weight entries included)."""
        return len(self._entries)

    def live_positions(self) -> tuple[int, ...]:
        """Positions that hold an entry, ascending."""
        return tuple(sorted(self._entries))

    @property
    def entries(self) -> Mapping[int, tuple[int, int]]:
        """Read-only view of ``{position: (q, entry_version)}`` for every live entry.

        A view, not a copy, so iterating it is cheap; it cannot be mutated.
        """
        return MappingProxyType(self._entries)

    def is_live(self, position: int) -> bool:
        """True if ``position`` holds an entry."""
        return position in self._entries

    def entry(self, position: int) -> tuple[int, int]:
        """``(q, entry_version)`` of a live position."""
        self._require_live(position)
        return self._entries[position]

    def leaf(self, position: int) -> int:
        """Stored leaf at ``position`` (0 for an empty slot)."""
        return self._sum.get(position)

    def prefix_sum_locate(self, target: int) -> int:
        """Leaf position whose cumulative weight range contains ``target``."""
        return self._sum.prefix_sum_locate(target)

    def verify_invariant(self) -> bool:
        """Recompute both trees bottom-up and check every internal node."""
        self._sum.verify_invariant()
        self._min.verify_invariant()
        return True

    # -- mutations ---------------------------------------------------------

    def write(self, position: int, q: int, entry_version: int) -> WriteEvent:
        """Insert or update the entry at ``position``.

        Raises
        ------
        ValueError
            If the position is out of range, ``entry_version`` is newer than
            the current version or already expired, or ``q`` is outside
            ``[0, 2^priority_bits)``. Nothing is changed on error.
        """
        self._require_position(position)
        if isinstance(entry_version, bool) or not isinstance(entry_version, int):
            raise ValueError(f"entry_version must be an int, got {entry_version!r}")
        if entry_version > self._current_version:
            raise ValueError(
                f"entry_version {entry_version} is newer than current version "
                f"{self._current_version}; call advance() first"
            )
        if is_expired(entry_version, self._current_version, self._params):
            raise ValueError(
                f"entry_version {entry_version} is expired at version "
                f"{self._current_version} (max_policy_age={self._params.max_policy_age})"
            )
        # Validates q and the epoch shift; raises before any state changes.
        new_leaf = inflated_priority(q, entry_version, self._base_epoch, self._params)

        op: WriteOp = "update" if position in self._entries else "insert"
        old_leaf = self._set_leaf(position, new_leaf)
        self._entries[position] = (q, entry_version)
        return WriteEvent(
            op=op,
            position=position,
            old_leaf=old_leaf,
            new_leaf=new_leaf,
            base_priority_int=q,
            entry_version=entry_version,
            base_epoch=self._base_epoch,
        )

    def evict(self, position: int, reason: EvictReason) -> WriteEvent:
        """Remove the entry at ``position``; its leaf becomes 0."""
        if reason not in EVICT_REASONS:
            raise ValueError(f"reason must be one of {sorted(EVICT_REASONS)}, got {reason!r}")
        self._require_live(position)
        q, entry_version = self._entries.pop(position)
        old_leaf = self._set_leaf(position, 0)
        return WriteEvent(
            op="evict",
            position=position,
            old_leaf=old_leaf,
            new_leaf=0,
            base_priority_int=q,
            entry_version=entry_version,
            base_epoch=self._base_epoch,
            reason=reason,
        )

    def advance(self, current_version: int) -> AdvanceResult:
        """Move to ``current_version``: evict expired entries, then rebase if due.

        The order matters. An expired entry may be older than the new base
        epoch, so shifting it would drop set bits; evicting it first keeps
        the rebase exact. The returned result lists the evictions before
        the rebase, and the attestation log records them in that order.

        Raises
        ------
        ValueError
            If ``current_version`` is lower than the current version.
        """
        if isinstance(current_version, bool) or not isinstance(current_version, int):
            raise ValueError(f"current_version must be an int, got {current_version!r}")
        old_version = self._current_version
        if current_version < old_version:
            raise ValueError(
                f"current_version must be monotone: {current_version} < {old_version}"
            )

        expired = sorted(
            position
            for position, (_, version) in self._entries.items()
            if is_expired(version, current_version, self._params)
        )
        evicted = tuple(self.evict(position, "stale") for position in expired)

        shift = pending_rebase_shift(current_version, self._base_epoch, self._params)
        rebase = self._rebase(shift) if shift > 0 else None
        # Committed last so a failure above (which would be a library bug,
        # not a user error) does not leave the version ahead of the tree.
        self._current_version = current_version
        return AdvanceResult(
            old_version=old_version,
            new_version=current_version,
            evicted=evicted,
            rebase=rebase,
        )

    def restore(
        self,
        base_epoch: int,
        current_version: int,
        entries: Mapping[int, tuple[int, int]],
    ) -> None:
        """Rebuild a fresh tree from saved ``(q, entry_version)`` pairs.

        Used by the durable buffer on recovery. The tree must be empty.
        Every entry is validated the way ``write`` validates it (live at
        ``current_version``, shift within budget), so a corrupt snapshot
        fails here rather than producing a tree that disagrees with the
        decay formula.
        """
        if self._entries or self._current_version or self._base_epoch:
            raise ValueError("restore() requires a fresh tree")
        if isinstance(base_epoch, bool) or not isinstance(base_epoch, int) or base_epoch < 0:
            raise ValueError(f"base_epoch must be a non-negative int, got {base_epoch!r}")
        if isinstance(current_version, bool) or not isinstance(current_version, int) or current_version < 0:
            raise ValueError(f"current_version must be a non-negative int, got {current_version!r}")
        self._base_epoch = base_epoch
        self._current_version = current_version
        try:
            for position, (q, version) in entries.items():
                self.write(position, q, version)
        except ValueError:
            self._entries.clear()
            self._sum = ExactSumTree(self._params.capacity)
            self._min = ExactMinTree(self._params.capacity)
            self._base_epoch = 0
            self._current_version = 0
            raise

    # -- internals ---------------------------------------------------------

    def _require_position(self, position: int) -> None:
        """ValueError unless ``position`` is an int within the tree's capacity."""
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError(f"position must be an int, got {position!r}")
        if not (0 <= position < self.capacity):
            raise ValueError(f"position {position} out of range [0, {self.capacity})")

    def _require_live(self, position: int) -> None:
        """ValueError unless ``position`` is in range and holds an entry."""
        self._require_position(position)
        if position not in self._entries:
            raise ValueError(f"position {position} holds no live entry")

    def _set_leaf(self, position: int, new_leaf: int) -> int:
        """Write a leaf into both trees; return the old sum-tree leaf."""
        old_leaf = self._sum.update(position, new_leaf)
        if new_leaf > 0:
            self._min.update(position, new_leaf)
        else:
            self._min.remove(position)
        return old_leaf

    def _rebase(self, shift: int) -> RebaseEvent:
        """Right-shift every node of both trees by ``shift`` epochs.

        Every live leaf is a multiple of ``2^shift`` (the caller evicted
        stale entries first), hence so is every internal sum, so shifting
        the whole array keeps the sum-tree invariant without re-propagating.
        ``rebase_priority`` checks the divisibility of each node and raises
        if a bit would be lost, which would indicate an internal error.

        The min-tree is monotone under a right shift, so its internal
        nodes stay consistent too. INFINITY sentinels are left alone.
        """
        total_before = self._sum.total
        shifted = tuple(
            (position, self._sum.get(position), self._sum.get(position) >> shift)
            for position in self.live_positions()
        )
        if self._entries:
            # A live entry's shift from the old base is at most max_shift < 64,
            # and it survived eviction only if its epoch >= the new base, so
            # shift <= max_shift here and rebase_priority's range check holds.
            self._sum._tree = [rebase_priority(node, shift) for node in self._sum._tree]
            infinity = ExactMinTree.INFINITY
            self._min._tree = [
                node if node == infinity else node >> shift for node in self._min._tree
            ]
        else:
            # Nothing live: every sum node is 0 and every min node is INFINITY,
            # so the shift (which may be >= 64 after a long idle period) is
            # a no-op on the arrays. Only the base epoch moves.
            assert total_before == 0
        old_base = self._base_epoch
        self._base_epoch = old_base + shift
        return RebaseEvent(
            old_base_epoch=old_base,
            new_base_epoch=self._base_epoch,
            root_total_before=total_before,
            root_total_after=self._sum.total,
            shifted=shifted,
        )

    def __repr__(self) -> str:
        return (
            f"DecayedPriorityTree(capacity={self.capacity}, live={self.live_count}, "
            f"version={self._current_version}, base_epoch={self._base_epoch})"
        )
