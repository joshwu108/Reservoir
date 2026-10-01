"""
checker.decay_replay — Replays the age-decay protocol of an attestation log.

This module imports NOTHING from src/reservoir. It re-derives the decay
arithmetic from the definitions in docs/design.md §7 using only the
standard library, so that agreement between the library and the checker
is evidence and not tautology.

What a decayed log claims
-------------------------
A ``decay_config`` record fixes integer parameters. Every mutation then
carries ``(base_priority_int q, entry_version t, base_epoch B)`` and
claims its leaf is::

    leaf = floor(q * T[t mod h] / 2^F) << (t // h - B)

where ``T[k] = floor(2^(k/h) * 2^F)`` is the decay table. The checker
recomputes ``T`` and the leaf itself and rejects any record whose
``new_priority_int`` disagrees.

The lifecycle it enforces
-------------------------
- ``advance_version`` raises the current version. Every live entry whose
  age now exceeds ``max_policy_age`` becomes *pending stale* and must be
  evicted (reason ``"stale"``) before any other record. If the newest
  epoch no longer fits the shift budget, a rebase becomes *pending* too.
- ``rebase`` is allowed only when pending, only after the stale
  evictions, must land on the canonical base epoch for the current
  version, must shift every replayed leaf without dropping a set bit,
  and its recorded totals must match. A rebase that is not due, or a
  due rebase that never comes, is rejected.
- A ``capacity`` evict must remove the oldest live entry, and a run of
  them must have been necessary: the inserts that follow the run must
  use exactly the slots that were free before it plus the ones it freed.
  (The buffer evicts ``group_size - free_slots`` entries, then inserts
  ``group_size``; stale evictions may have freed some slots first, so the
  buffer need not be full when a capacity evict happens.)
- An ``update`` keeps the entry's version unless the config says
  ``reset_age_on_update``, in which case it carries the current version.

Bounds
------
The config is checked against the same bit budget the library enforces
(§7.2): ``priority_bits + table_frac_bits + 1 <= 64`` for the product
and ``priority_bits + 1 + max_shift + ceil(log2 capacity) <= 64`` for the
tree, with ``max_shift = ceil(max_policy_age / half_life) + rebase_slack``.
``half_life`` is capped at 1024 because building the table is O(h * F)
big-integer work; without the cap a hostile config could stall the
checker.

``DecayState`` holds the replay state; ``verify.py`` calls its ``on_*``
methods as it walks the chain.
"""

from __future__ import annotations

from typing import Optional

# Mirrors the library's documented limit (design.md §7.5). Not imported.
MAX_HALF_LIFE = 1024
UINT64_BITS = 64


class CheckerError(Exception):
    """Raised when any attestation check fails. Re-exported by verify.py."""


_CONFIG_INT_FIELDS = (
    "half_life", "max_policy_age", "capacity", "priority_bits",
    "priority_frac_bits", "table_frac_bits", "rebase_slack",
)
_DECAY_FIELDS = ("base_priority_int", "entry_version", "base_epoch")
_EVICT_REASONS = ("stale", "capacity", "explicit")


# ---------------------------------------------------------------------------
# Pure arithmetic, from the definitions
# ---------------------------------------------------------------------------

def decay_table(half_life: int, table_frac_bits: int) -> tuple[int, ...]:
    """T[k] = floor(2^(k/h) * 2^F) for k in [0, h), by bisection on the definition.

    T[k] is the unique integer in [2^F, 2^(F+1)) with
    T[k]^h <= 2^(k + F*h) < (T[k] + 1)^h. Each entry is found by binary
    search over that interval, checking the inequality with exact integer
    powers. Deliberately not the library's hinted search: a different
    algorithm reaching the same numbers is the point.
    """
    one = 1 << table_frac_bits
    table = []
    for k in range(half_life):
        target = 1 << (k + table_frac_bits * half_life)
        low, high = one, one << 1  # answer is in [low, high)
        while high - low > 1:
            mid = (low + high) // 2
            if mid ** half_life <= target:
                low = mid
            else:
                high = mid
        table.append(low)
    return tuple(table)


def inflated_priority(q: int, entry_version: int, base_epoch: int, cfg: dict) -> int:
    """The leaf a decayed entry must have. Raises CheckerError on an impossible input.

    Impossible means: q outside its bit width, an entry older than the base
    epoch, a shift beyond ``max_shift`` (the library would have rebased
    first), or a leaf that does not fit in 64 bits.
    """
    h = cfg["half_life"]
    if not (0 <= q < (1 << cfg["priority_bits"])):
        raise CheckerError(f"base_priority_int {q} outside [0, 2^{cfg['priority_bits']})")
    epoch, phase = divmod(entry_version, h)
    shift = epoch - base_epoch
    if shift < 0:
        raise CheckerError(
            f"entry_version {entry_version} (epoch {epoch}) is older than base_epoch {base_epoch}"
        )
    if shift > cfg["max_shift"]:
        raise CheckerError(
            f"entry_version {entry_version} needs shift {shift} > max_shift {cfg['max_shift']}; "
            f"a rebase was due first"
        )
    table = _table_for(cfg)
    leaf = ((q * table[phase]) >> cfg["table_frac_bits"]) << shift
    if leaf >= 1 << UINT64_BITS:
        raise CheckerError(f"recomputed leaf {leaf} does not fit in 64 bits")
    return leaf


def canonical_base_epoch(current_version: int, cfg: dict) -> int:
    """max(0, current_version - max_policy_age) // half_life."""
    oldest_live = max(0, current_version - cfg["max_policy_age"])
    return oldest_live // cfg["half_life"]


def is_expired(entry_version: int, current_version: int, cfg: dict) -> bool:
    return current_version - entry_version > cfg["max_policy_age"]


def rebase_is_due(current_version: int, base_epoch: int, cfg: dict) -> bool:
    """True when the newest epoch no longer fits the shift budget from ``base_epoch``."""
    newest_epoch = current_version // cfg["half_life"]
    return newest_epoch - base_epoch > cfg["max_shift"]


_TABLE_CACHE: dict[tuple[int, int], tuple[int, ...]] = {}


def _table_for(cfg: dict) -> tuple[int, ...]:
    key = (cfg["half_life"], cfg["table_frac_bits"])
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = decay_table(*key)
    return _TABLE_CACHE[key]


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------

def _int_field(record: dict, name: str, idx: int) -> int:
    """Parse a non-negative integer stored as a decimal string.

    The library writes integers as ``str(int)`` to avoid JSON precision
    limits. Only that exact form is accepted: a JSON number, a boolean,
    a sign, whitespace or underscores would all be a forgery or a bug,
    even though ``int()`` would happily parse some of them.
    """
    value = record.get(name)
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise CheckerError(
            f"Record {idx}: {name} must be a non-negative integer encoded as a decimal "
            f"string, got {value!r}"
        )
    return int(value)


def parse_config(record: dict, idx: int) -> dict:
    """Validate a decay_config record and return its parameters as a dict.

    Adds the derived ``max_shift`` and ``capacity_bits``. Enforces the
    bit budget so a hostile config cannot make the replay accept leaves
    the library could never store, and caps ``half_life`` so the table
    build stays cheap.
    """
    cfg = {name: _int_field(record, name, idx) for name in _CONFIG_INT_FIELDS}
    if cfg["half_life"] < 1 or cfg["capacity"] < 1:
        raise CheckerError(f"Record {idx}: half_life and capacity must be >= 1")
    if cfg["half_life"] > MAX_HALF_LIFE:
        raise CheckerError(f"Record {idx}: half_life {cfg['half_life']} exceeds {MAX_HALF_LIFE}")
    if cfg["priority_bits"] < 1 or cfg["table_frac_bits"] < 1:
        raise CheckerError(f"Record {idx}: priority_bits and table_frac_bits must be >= 1")
    if cfg["priority_frac_bits"] > cfg["priority_bits"]:
        raise CheckerError(f"Record {idx}: priority_frac_bits exceeds priority_bits")
    reset = record.get("reset_age_on_update")
    if not isinstance(reset, bool):
        raise CheckerError(f"Record {idx}: reset_age_on_update must be a bool, got {reset!r}")
    cfg["reset_age_on_update"] = reset

    h, age, slack = cfg["half_life"], cfg["max_policy_age"], cfg["rebase_slack"]
    cfg["max_shift"] = -(-age // h) + slack  # ceil(age / h) + slack
    cfg["capacity_bits"] = (cfg["capacity"] - 1).bit_length()
    product_bits = cfg["priority_bits"] + cfg["table_frac_bits"] + 1
    tree_bits = cfg["priority_bits"] + 1 + cfg["max_shift"] + cfg["capacity_bits"]
    if product_bits > UINT64_BITS or tree_bits > UINT64_BITS:
        raise CheckerError(
            f"Record {idx}: decay_config violates the 64-bit budget "
            f"(product {product_bits} bits, tree {tree_bits} bits)"
        )
    return cfg


def has_decay_fields(record: dict) -> bool:
    return any(name in record for name in _DECAY_FIELDS)


# ---------------------------------------------------------------------------
# Replay state
# ---------------------------------------------------------------------------

class DecayState:
    """Everything the checker tracks for a decayed log, beyond the sum-tree.

    Attributes
    ----------
    entries : dict
        ``{position: (q, entry_version)}`` for every live entry.
    pending_stale : set
        Positions that expired at the last ``advance_version`` and have
        not been evicted yet. While non-empty, only their stale evicts
        are allowed.
    pending_rebase : bool
        A rebase became due at the last ``advance_version`` and has not
        happened yet. While set, only stale evicts and the rebase itself
        are allowed.
    capacity_phase : str
        Tracks a capacity-eviction run: ``""`` (none), ``"evicting"``
        (one or more capacity evicts seen, no insert yet) or
        ``"inserting"`` (the group's inserts are arriving). When the run
        ends, ``inserts_after_run`` must equal ``free_at_run_start +
        evicts_in_run``; otherwise the evictions were not necessary and
        the log does not describe what the buffer does.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.current_version = 0
        self.base_epoch = 0
        self.entries: dict[int, tuple[int, int]] = {}
        self.pending_stale: set[int] = set()
        self.pending_rebase = False
        self.capacity_phase = ""
        self.free_at_run_start = 0
        self.evicts_in_run = 0
        self.inserts_after_run = 0

    # -- protocol guards ---------------------------------------------------

    def require_no_pending(self, idx: int, op: str) -> None:
        """Nothing but stale evicts and a due rebase may happen while they are pending."""
        if self.pending_stale:
            raise CheckerError(
                f"Record {idx}: {op} while expired entries at positions "
                f"{sorted(self.pending_stale)} have not been evicted"
            )
        if self.pending_rebase:
            raise CheckerError(
                f"Record {idx}: {op} while a rebase is due (version {self.current_version}, "
                f"base_epoch {self.base_epoch}) and has not been recorded"
            )

    def require_ready_for_rebase(self, idx: int) -> None:
        if self.pending_stale:
            raise CheckerError(
                f"Record {idx}: rebase before expired entries at positions "
                f"{sorted(self.pending_stale)} were evicted"
            )
        if not self.pending_rebase:
            raise CheckerError(
                f"Record {idx}: rebase is not due at version {self.current_version} "
                f"with base_epoch {self.base_epoch}"
            )

    def note_record(self, op: str, reason: Optional[str], idx: int) -> None:
        """Advance the capacity-run state machine; see ``capacity_phase``.

        Called for every record before it is verified. A capacity evict
        starts or extends a run; inserts after a run are counted; any
        other record closes the run and triggers the necessity check.
        """
        is_capacity_evict = op == "evict" and reason == "capacity"
        if is_capacity_evict:
            if self.capacity_phase != "evicting":
                self.close_capacity_run(idx)
                self.capacity_phase = "evicting"
                self.free_at_run_start = self.cfg["tree_capacity"] - len(self.entries)
                self.evicts_in_run = 0
                self.inserts_after_run = 0
            self.evicts_in_run += 1
        elif op == "insert" and self.capacity_phase:
            self.capacity_phase = "inserting"
            self.inserts_after_run += 1
        else:
            self.close_capacity_run(idx)

    def close_capacity_run(self, idx: int) -> None:
        """End a capacity run and check that its evictions were necessary."""
        if not self.capacity_phase:
            return
        expected = self.free_at_run_start + self.evicts_in_run
        if self.capacity_phase == "evicting" or self.inserts_after_run != expected:
            raise CheckerError(
                f"Record {idx}: {self.evicts_in_run} capacity eviction(s) with "
                f"{self.free_at_run_start} free slot(s) were followed by "
                f"{self.inserts_after_run} insert(s); the buffer would insert exactly {expected}"
            )
        self.capacity_phase = ""

    # -- record handlers ---------------------------------------------------

    def on_advance(self, record: dict, idx: int) -> None:
        old = _int_field(record, "old_version", idx)
        new = _int_field(record, "new_version", idx)
        if old != self.current_version:
            raise CheckerError(
                f"Record {idx}: advance_version old_version={old} but replay is at "
                f"{self.current_version}"
            )
        if new < old:
            raise CheckerError(f"Record {idx}: advance_version goes backwards {old} -> {new}")
        self.current_version = new
        self.pending_stale = {
            pos for pos, (_, t) in self.entries.items() if is_expired(t, new, self.cfg)
        }
        self.pending_rebase = rebase_is_due(new, self.base_epoch, self.cfg)

    def on_rebase(self, record: dict, idx: int, tree) -> None:
        old = _int_field(record, "old_base_epoch", idx)
        new = _int_field(record, "new_base_epoch", idx)
        before = _int_field(record, "root_total_before", idx)
        after = _int_field(record, "root_total_after", idx)
        if old != self.base_epoch:
            raise CheckerError(
                f"Record {idx}: rebase old_base_epoch={old} but replay base_epoch is {self.base_epoch}"
            )
        canonical = canonical_base_epoch(self.current_version, self.cfg)
        if new != canonical or new <= old:
            raise CheckerError(
                f"Record {idx}: rebase new_base_epoch={new} is not the canonical base epoch "
                f"{canonical} for version {self.current_version} (old was {old})"
            )
        if tree.total != before:
            raise CheckerError(
                f"Record {idx}: rebase root_total_before={before} but replayed total is {tree.total}"
            )
        tree.shift_all(new - old, idx)
        if tree.total != after:
            raise CheckerError(
                f"Record {idx}: rebase root_total_after={after} but shifted total is {tree.total}"
            )
        self.base_epoch = new
        self.pending_rebase = False

    def on_mutation(self, record: dict, idx: int, tree) -> None:
        """Checks beyond the legacy old/new consistency, which verify.py already did."""
        op = record["op"]
        pos = record["index"]
        q = _int_field(record, "base_priority_int", idx)
        t = _int_field(record, "entry_version", idx)
        base = _int_field(record, "base_epoch", idx)
        new_leaf = _int_field(record, "new_priority_int", idx)
        if base != self.base_epoch:
            raise CheckerError(
                f"Record {idx}: base_epoch={base} but replay base_epoch is {self.base_epoch}"
            )
        if t > self.current_version:
            raise CheckerError(
                f"Record {idx}: entry_version {t} is newer than current version "
                f"{self.current_version}"
            )
        if op == "evict":
            self._on_evict(record, idx, pos, q, t, new_leaf, tree)
            return
        if "reason" in record:
            raise CheckerError(f"Record {idx}: reason is only valid on evict records")
        if is_expired(t, self.current_version, self.cfg):
            raise CheckerError(
                f"Record {idx}: {op} of an expired entry (version {t} at {self.current_version})"
            )
        expected = inflated_priority(q, t, base, self.cfg)
        if new_leaf != expected:
            raise CheckerError(
                f"Record {idx}: new_priority_int={new_leaf} but recomputed leaf is {expected} "
                f"(q={q}, entry_version={t}, base_epoch={base})"
            )
        if op == "insert":
            if pos in self.entries:
                raise CheckerError(f"Record {idx}: insert into position {pos} that holds a live entry")
        else:  # update
            if pos not in self.entries:
                raise CheckerError(f"Record {idx}: update of position {pos} with no live entry")
            self._check_update_version(idx, pos, t)
        self.entries[pos] = (q, t)

    def _check_update_version(self, idx: int, pos: int, t: int) -> None:
        stored_t = self.entries[pos][1]
        if self.cfg["reset_age_on_update"]:
            if t != self.current_version:
                raise CheckerError(
                    f"Record {idx}: update must re-stamp to current version "
                    f"{self.current_version} (reset_age_on_update), got {t}"
                )
        elif t != stored_t:
            raise CheckerError(
                f"Record {idx}: update must keep entry_version {stored_t}, got {t}"
            )

    def _on_evict(
        self, record: dict, idx: int, pos: int, q: int, t: int, new_leaf: int, tree
    ) -> None:
        if pos not in self.entries:
            raise CheckerError(f"Record {idx}: evict of position {pos} with no live entry")
        if new_leaf != 0:
            raise CheckerError(f"Record {idx}: evict must set the leaf to 0, got {new_leaf}")
        if self.entries[pos] != (q, t):
            raise CheckerError(
                f"Record {idx}: evict records (q={q}, entry_version={t}) but the live entry "
                f"is {self.entries[pos]}"
            )
        reason = record.get("reason")
        if reason not in _EVICT_REASONS:
            raise CheckerError(f"Record {idx}: evict reason must be one of {_EVICT_REASONS}, got {reason!r}")
        if reason == "stale":
            if not is_expired(t, self.current_version, self.cfg):
                raise CheckerError(
                    f"Record {idx}: stale evict of a live entry (version {t} at {self.current_version})"
                )
        else:
            self.require_no_pending(idx, f"{reason} evict")
        if reason == "capacity":
            oldest = min(version for _, version in self.entries.values())
            if t != oldest:
                raise CheckerError(
                    f"Record {idx}: capacity evict removed version {t} but the oldest live "
                    f"entry is version {oldest}"
                )
        del self.entries[pos]
        self.pending_stale.discard(pos)
