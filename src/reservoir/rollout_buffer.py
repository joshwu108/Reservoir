"""
reservoir.rollout_buffer — Prioritized replay buffer for LLM-RL rollouts.

    buf = RolloutBuffer(capacity=50_000, priority=AdvantagePriority(),
                        half_life=4, max_policy_age=16, seed=0)
    buf.add_group(prompt_id, model_version=step, rollouts=[...])
    batch = buf.sample(batch_size=64, current_version=step)
    buf.update_priorities(batch.indices, new_scores)

Entries are whole rollouts (variable-length token sequences with their
behavior logprobs), grouped by the prompt and model version that produced
them. One sum-tree leaf per rollout; the ``RolloutGroup`` is shared
metadata that the priority strategy reads.

How a priority is computed
--------------------------
::

    raw  = priority.score(rollout, group)         # user float, validated >= 0
    x    = raw ** alpha                           # float, once (alpha=1 by default)
    q    = quantize_priority(x)                   # 16.16 fixed point, exact from here
    leaf = inflated_priority(q, model_version)    # exact age decay, see decayed_tree.py

The sampling distribution is ``P(i) = leaf_i / total`` exactly, and that
is what the attestation log records and the checker verifies.

Versions and staleness
----------------------
``current_version`` only moves forward. ``add_group`` and ``sample`` both
advance it when given a newer version; advancing evicts every entry older
than ``max_policy_age`` versions and rebases the tree when needed. A group
may arrive with a version *below* the current one (asynchronous rollout
workers) as long as it is not already expired.

Capacity
--------
Freed slots are reused lowest-index first. When the buffer is full, the
entries with the lowest model version are evicted, ties broken by
insertion order (oldest first). A group is never admitted by evicting
entries *newer* than itself: if a late-arriving group would have to
displace fresher data, ``add_group`` raises and the buffer is unchanged.
A single group larger than the capacity is an error.

Call structure
--------------
``sample(batch_size, current_version)``:
    advance(current_version)            -> DecayedPriorityTree.advance
        evict expired entries             (reason "stale")
        rebase if the shift budget is hit (right-shift every node)
    draw_uniform_below(total, ...)      -> draw.py, one keyed draw per rollout
    tree.prefix_sum_locate(draw)        -> sumtree.py, walk root to leaf
    _is_weight(leaf, total, minimum, n) -> the only float on this path
    attester.record_sample(...)         -> attest.py append_sample

``add_group(prompt_id, model_version, rollouts)``:
    RolloutGroup(...)                   -> rollout.py validates every field
    _check_and_score_group              -> priorities.validated_score, then
                                           score ** alpha, decay.quantize_priority
                                           (nothing mutated yet; errors stop here)
    advance(model_version)              -> as above, if the version is newer
    _allocate(n)                        -> free-list heap, then oldest-version evicts
    tree.write(pos, q, version)         -> decay.inflated_priority gives the leaf
    attester.record_write(event)        -> attest.py append_mutation

``update_priorities(indices, scores)``: validate all, then tree.write per index
with decay.version_after_update deciding whether the entry's age resets.

Determinism
-----------
Draws are keyed BLAKE2b hashes of ``(seed, buffer_id, draw_counter)``
where ``draw_counter`` increases by one per sampled rollout for the
lifetime of the buffer, so no two draws share a key regardless of batch
size. Two buffers with the same seed and the same sequence of calls
produce identical batches and identical logs. The seed, buffer id, alpha
and beta are written into the log's ``decay_config`` record, so the
independent checker recomputes every draw and every importance weight
rather than only checking that each draw lands in the recorded leaf.
"""

from __future__ import annotations

import heapq
import math
import operator
from fractions import Fraction
from typing import Any, Callable, NamedTuple, Optional, Sequence

from reservoir.decay import DecayParams, is_expired, quantize_priority, version_after_update
from reservoir.decayed_tree import AdvanceResult, DecayedPriorityTree
from reservoir.draw import draw_uniform_below
from reservoir.priorities import AdvantagePriority, PriorityStrategy, validated_score
from reservoir.rollout import Rollout, RolloutGroup, default_is_success
from reservoir.rollout_attest import AttestTarget, DrawConfig, ManifestTarget, RolloutAttester
from reservoir.rollout_quarantine import QuarantinePredicate, apply_quarantine, run_quarantine
from reservoir.rollout_snapshot import buffer_fingerprint, buffer_state_dict, load_buffer_state
from reservoir.rollout_telemetry import exact_telemetry
from reservoir.sumtree import ExactMinTree


class RolloutBatch(NamedTuple):
    """Result of ``RolloutBuffer.sample``. All sequences are aligned by position ``k``."""

    indices: tuple[int, ...]              # buffer slots; pass back to update_priorities
    rollouts: tuple[Rollout, ...]
    groups: tuple[RolloutGroup, ...]      # the group each rollout came from
    logprobs: tuple[tuple[float, ...], ...]   # behavior logprobs, for importance correction
    model_versions: tuple[int, ...]       # policy version that produced each rollout
    rewards: tuple[float, ...]
    is_weights: tuple[Fraction, ...]      # exact IS weights normalised to (0, 1]
    priorities: tuple[int, ...]           # the decayed leaf of each sampled slot
    draw_integers: tuple[int, ...]        # the keyed draws, for the attestation log
    root_total: int                       # tree total at sample time
    min_priority_int: int                 # smallest positive leaf at sample time
    op_counter: int = -1                  # the sample record's op_counter; names this batch to a witness
    content_digests: tuple[str, ...] = () # content digest of each sampled rollout

    @property
    def float_is_weights(self) -> tuple[float, ...]:
        """``is_weights`` as floats, for feeding straight into a loss."""
        return tuple(float(w) for w in self.is_weights)


def _validate_priority(priority: Optional[PriorityStrategy]) -> PriorityStrategy:
    """Default to AdvantagePriority; anything that is not a strategy is a TypeError."""
    if priority is None:
        return AdvantagePriority()
    if not isinstance(priority, PriorityStrategy):
        raise TypeError(f"priority must be a PriorityStrategy, got {type(priority).__name__}")
    return priority


def _validate_exponent(value: object, name: str, minimum_exclusive: Optional[float]) -> float:
    """Validate alpha or beta.

    Both must be finite real numbers (bool excluded). With
    ``minimum_exclusive=0.0`` the value must be strictly positive (alpha:
    ``raw ** 0`` would flatten every priority to 1); with ``None`` it only
    has to be non-negative (beta: 0 means no importance correction).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if minimum_exclusive is not None and value <= minimum_exclusive:
        raise ValueError(f"{name} must be > {minimum_exclusive}, got {value!r}")
    if minimum_exclusive is None and value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return float(value)


class RolloutBuffer:
    """Exact, age-decayed prioritized replay over rollout groups.

    Parameters
    ----------
    capacity : int
        Maximum number of rollouts. Rounded up to a power of two.
    priority : PriorityStrategy
        Scores each rollout. Default ``AdvantagePriority()``.
    half_life : int
        Model versions per halving of sampling weight.
    max_policy_age : int
        Entries older than this many versions are evicted.
    alpha : float
        Exponent applied to the raw score before quantization. Default 1.0
        (advantage magnitudes are used as-is in the replay-for-GRPO
        papers); classic PER uses 0.6.
    beta : float
        Importance-sampling exponent. Default 0.4.
    seed, buffer_id : int
        Key material for the deterministic draw.
    attest : AttestationLog | path | None
        Where to record mutations and samples. See ``rollout_attest``.
        Every insert record carries the ``content_digest`` of the stored
        example (``rollout.content_digest_of``) and the group's ``source``.
    manifest : ManifestWriter | path | None
        Also write the opening of every insert's digest (prompt id,
        tokens, reward, source), one JSON line per insert, so the checker
        can confirm the log commits to exactly these examples. Requires
        ``attest``. See ``rollout_manifest``.
    reset_age_on_update : bool
        If True, ``update_priorities`` re-stamps an entry at the current
        version; by default age is measured from collection.
    priority_bits, priority_frac_bits, rebase_slack : int
        Passed to ``DecayParams``; see ``decay.py``.
    attest_overwrite : bool
        Replace an existing attestation file instead of refusing it. Only
        the durable buffer sets this; it rewrites the file from recovered
        state, so the file is never the source of truth there.

    Snapshots
    ---------
    ``state_dict()`` returns the complete state as a JSON-serialisable
    dict and ``load_state_dict()`` rebuilds a fresh buffer from one. The
    durable buffer uses them; they are also a plain way to checkpoint.
    """

    STATE_FORMAT = 1

    def __init__(
        self,
        capacity: int,
        priority: Optional[PriorityStrategy] = None,
        half_life: int = 4,
        max_policy_age: int = 16,
        alpha: float = 1.0,
        beta: float = 0.4,
        seed: int = 0,
        buffer_id: int = 0,
        attest: AttestTarget = None,
        reset_age_on_update: bool = False,
        priority_bits: int = 32,
        priority_frac_bits: int = 16,
        rebase_slack: int = 0,
        attest_overwrite: bool = False,
        manifest: ManifestTarget = None,
    ) -> None:
        self.priority = _validate_priority(priority)
        self.alpha = _validate_exponent(alpha, "alpha", minimum_exclusive=0.0)
        self.beta = _validate_exponent(beta, "beta", minimum_exclusive=None)
        self.seed = seed
        self.buffer_id = buffer_id
        self.reset_age_on_update = bool(reset_age_on_update)

        self._params = DecayParams(
            half_life=half_life,
            max_policy_age=max_policy_age,
            capacity=capacity,
            priority_bits=priority_bits,
            priority_frac_bits=priority_frac_bits,
            rebase_slack=rebase_slack,
        )
        self._tree = DecayedPriorityTree(self._params)
        n = self._tree.capacity
        self._rollouts: list[Optional[Rollout]] = [None] * n
        self._groups: list[Optional[RolloutGroup]] = [None] * n
        self._inserted: list[int] = [0] * n       # insertion sequence number per slot
        self._free: list[int] = list(range(n))    # min-heap of empty slots
        self._insert_seq = 0
        self._op_counter = 0                      # bumped once per sampled batch; logged
        self._witnessed = -1                      # op_counter of the last batch a witness was written for
        self._last_sample: Optional[dict] = None  # what rebuilds ``last_batch`` after a snapshot
        self._draw_counter = 0                    # bumped once per sampled rollout; keys draws
        self._n_rebases = 0
        self._attester = RolloutAttester(
            attest, self._params, self.reset_age_on_update, overwrite=attest_overwrite,
            manifest=manifest,
            draw=DrawConfig(seed=self.seed, buffer_id=self.buffer_id, alpha=self.alpha, beta=self.beta),
        )

    # -- read-only state ---------------------------------------------------

    @property
    def params(self) -> DecayParams:
        """The validated decay configuration (half-life, max age, bit widths)."""
        return self._params

    @property
    def capacity(self) -> int:
        """Number of slots: the requested capacity rounded up to a power of two."""
        return self._tree.capacity

    @property
    def size(self) -> int:
        """Number of live rollouts."""
        return self._tree.live_count

    def __len__(self) -> int:
        return self.size

    @property
    def total(self) -> int:
        """Sum of all decayed leaves; the denominator of every sampling probability."""
        return self._tree.total

    @property
    def current_version(self) -> int:
        """Latest model version the buffer has been advanced to. Never decreases."""
        return self._tree.current_version

    @property
    def base_epoch(self) -> int:
        """Epoch the stored leaves are shifted relative to; grows on each rebase."""
        return self._tree.base_epoch

    @property
    def n_rebases(self) -> int:
        """How many times the tree has rebased so far (useful in tests and diagnostics)."""
        return self._n_rebases

    @property
    def attestation_log(self) -> Optional[AttestationLog]:
        """The in-memory ``AttestationLog`` being appended to, or None if attestation is off."""
        return self._attester.log

    @property
    def has_manifest(self) -> bool:
        """True when a manifest records the opening of every insert's content digest."""
        return self._attester.has_manifest

    @property
    def manifest_records(self) -> list[dict]:
        """Manifest lines written so far (a copy); empty without a manifest."""
        return self._attester.manifest_records

    def live_positions(self) -> tuple[int, ...]:
        """Slots currently holding a rollout, ascending."""
        return self._tree.live_positions()

    def entry(self, position: int) -> tuple[Rollout, RolloutGroup]:
        """``(rollout, group)`` stored at a live position."""
        self._tree.entry(position)  # raises if not live
        return self._rollouts[position], self._groups[position]  # type: ignore[return-value]

    def base_priority(self, position: int) -> int:
        """The fixed-point ``q`` of a live position."""
        return self._tree.entry(position)[0]

    def entry_version(self, position: int) -> int:
        """The model version a live position is aged from."""
        return self._tree.entry(position)[1]

    def leaf(self, position: int) -> int:
        """The decayed leaf at a position (0 if empty)."""
        return self._tree.leaf(position)

    def verify_trees(self) -> bool:
        """Recompute every internal node of both trees; raises AssertionError on a mismatch."""
        return self._tree.verify_invariant()

    # -- lifecycle ---------------------------------------------------------

    def advance(self, current_version: int) -> AdvanceResult:
        """Move the buffer to ``current_version``, evicting stale entries and rebasing if due.

        Called automatically by ``add_group`` and ``sample`` when they are
        given a newer version; call it directly to expire entries without
        sampling. Evicted slots go back on the free list and everything
        the tree did is appended to the attestation log. Returns the
        tree's ``AdvanceResult``. Raises ``ValueError`` on a lower version.
        """
        result = self._tree.advance(current_version)
        for event in result.evicted:
            self._release_slot(event.position)
        if result.rebase is not None:
            self._n_rebases += 1
        self._attester.record_advance(result, self._op_counter)
        return result

    def add_group(
        self,
        prompt_id: str,
        model_version: int,
        rollouts: Sequence[Rollout],
        is_success: Optional[Callable[[Rollout], bool]] = None,
        source: Optional[str] = None,
    ) -> tuple[int, ...]:
        """Store every rollout of one prompt group; return their slots.

        ``source`` tags where the prompt came from (``RolloutGroup.source``)
        and is written on each insert record. Validation and scoring
        happen before anything is mutated, so a bad rollout or a
        misbehaving strategy leaves the buffer unchanged.

        Raises
        ------
        ValueError
            Malformed inputs, a group larger than the capacity, a version
            already expired, a full buffer whose oldest entries are newer
            than this group, a score the strategy's contract forbids, or a
            score outside the fixed-point range (the message names the prompt).
        """
        group = RolloutGroup(
            prompt_id=prompt_id,
            model_version=model_version,
            rollouts=rollouts,
            is_success=is_success if is_success is not None else default_is_success,
            source=source,
        )
        qs = self._check_and_score_group(group)
        prepared = self._attester.prepare_inserts(group, self._op_counter)

        if model_version > self.current_version:
            self.advance(model_version)
        positions = self._allocate(group.size)
        for member, (position, q) in enumerate(zip(positions, qs)):
            event = self._tree.write(position, q, model_version)
            self._rollouts[position] = group.rollouts[member]
            self._groups[position] = group
            self._insert_seq += 1
            self._inserted[position] = self._insert_seq
            if prepared:
                self._attester.record_insert(event, self._op_counter, prepared[member])
        return positions

    def _check_and_score_group(self, group: RolloutGroup) -> list[int]:
        """Everything that can fail in add_group, before any state changes."""
        if group.size > self.capacity:
            raise ValueError(
                f"group of {group.size} rollouts exceeds capacity {self.capacity}"
            )
        current = max(self.current_version, group.model_version)
        if is_expired(group.model_version, current, self._params):
            raise ValueError(
                f"model_version {group.model_version} is already expired at version "
                f"{current} (max_policy_age={self._params.max_policy_age})"
            )
        self._check_can_admit(group)
        context = f"prompt {group.prompt_id!r}"
        return [
            self._score_to_q(validated_score(self.priority, r, group), context)
            for r in group.rollouts
        ]

    def _check_can_admit(self, group: RolloutGroup) -> None:
        """Refuse to evict entries newer than the incoming group.

        Runs before any mutation. If the group's version is newer than the
        buffer's, ``advance`` will expire some entries first, so those are
        counted as free. What remains is the number of capacity evictions
        the group would force; if the newest of those victims is newer
        than the group, admitting it would throw away fresher data.
        """
        will_free = 0
        if group.model_version > self.current_version:
            will_free = sum(
                1 for _, version in self._tree.entries.values()
                if is_expired(version, group.model_version, self._params)
            )
        needed = group.size - len(self._free) - will_free
        if needed <= 0:
            return
        victims = self._oldest_live(needed)
        newest_victim_version = self._tree.entry(victims[-1])[1]
        if newest_victim_version > group.model_version:
            raise ValueError(
                f"buffer is full and admitting prompt {group.prompt_id!r} at version "
                f"{group.model_version} would evict entries from version "
                f"{newest_victim_version}; a group never displaces newer data"
            )

    def sample(self, batch_size: int, current_version: Optional[int] = None) -> RolloutBatch:
        """Draw ``batch_size`` rollouts with replacement, proportional to decayed priority.

        Sampling is with replacement, so ``batch_size`` may exceed the
        number of live entries; a small buffer can still fill a large
        batch. The only requirement is at least one entry with positive
        weight.

        Raises
        ------
        ValueError
            ``batch_size < 1`` or a non-monotone ``current_version``.
        RuntimeError
            No live entries, or every live entry has zero weight.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
        if current_version is not None:
            self.advance(current_version)

        n = self.size
        if n == 0:
            raise RuntimeError("cannot sample: buffer has no live rollouts")
        root_total = self.total
        if root_total == 0:
            raise RuntimeError("cannot sample: total priority is zero")
        min_priority = self._tree.minimum
        if min_priority >= ExactMinTree.INFINITY:
            raise RuntimeError("cannot sample: no positive priorities")

        # Two counters: op_counter numbers batches and is what the log
        # records; draw_counter numbers individual draws and keys them.
        self._op_counter += 1
        batch_op = self._op_counter
        draws, indices = self._draw_indices(batch_size, root_total)
        batch = self._build_batch(indices, draws, root_total, min_priority, n)
        self._attester.record_sample(
            batch_op, root_total, batch.indices, batch.draw_integers,
            batch.priorities, batch.is_weights,
        )
        self._last_sample = {"indices": list(indices), "draws": [str(d) for d in draws],
                             "root_total": str(root_total), "min_priority": str(min_priority),
                             "n": n, "priorities": [str(p) for p in batch.priorities],
                             "op_counter": batch_op, "versions": list(batch.model_versions),
                             "inserted": [self._inserted[i] for i in indices]}
        return batch

    @property
    def last_batch(self) -> Optional[RolloutBatch]:
        """The most recent ``sample`` result, rebuilt from the buffer's own records.

        Survives a snapshot (the durable command log replays witnesses and
        telemetry against it). None before the first sample, or if a slot
        of that batch has since been evicted or refilled: each slot's insert
        stamp is checked against the one recorded at sample time, so a
        rebuilt batch is the sampled batch, never a later occupant.
        """
        raw = self._last_sample
        if raw is None:
            return None
        indices = tuple(raw["indices"])
        if any(self._rollouts[i] is None or self._inserted[i] != stamp
               for i, stamp in zip(indices, raw["inserted"])):
            return None
        priorities = tuple(int(p) for p in raw["priorities"])
        root_total, min_priority, n = int(raw["root_total"]), int(raw["min_priority"]), raw["n"]
        rollouts = tuple(self._rollouts[i] for i in indices)
        groups = tuple(self._groups[i] for i in indices)
        return RolloutBatch(
            indices=indices, rollouts=rollouts, groups=groups,  # type: ignore[arg-type]
            logprobs=tuple(r.logprobs for r in rollouts),  # type: ignore[union-attr]
            model_versions=tuple(raw["versions"]),
            rewards=tuple(r.reward for r in rollouts),  # type: ignore[union-attr]
            is_weights=tuple(self._is_weight(p, root_total, min_priority, n) for p in priorities),
            priorities=priorities, draw_integers=tuple(int(d) for d in raw["draws"]),
            root_total=root_total, min_priority_int=min_priority, op_counter=raw["op_counter"],
            content_digests=tuple(
                g.content_digests[next(k for k, m in enumerate(g.rollouts) if m is r)]  # type: ignore[union-attr]
                for r, g in zip(rollouts, groups)),
        )

    def _draw_indices(
        self, batch_size: int, root_total: int
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Keyed draws for one batch; each sampled rollout consumes one draw counter value."""
        first = self._draw_counter
        self._draw_counter += batch_size
        draws = tuple(
            draw_uniform_below(
                root_total, seed=self.seed, buffer_id=self.buffer_id, op_counter=first + k,
            )
            for k in range(batch_size)
        )
        indices = tuple(self._tree.prefix_sum_locate(d) for d in draws)
        return draws, indices

    def _build_batch(
        self,
        indices: tuple[int, ...],
        draws: tuple[int, ...],
        root_total: int,
        min_priority: int,
        n: int,
    ) -> RolloutBatch:
        """Gather the stored data for the sampled slots into an immutable batch."""
        rollouts = tuple(self._rollouts[i] for i in indices)
        groups = tuple(self._groups[i] for i in indices)
        priorities = tuple(self._tree.leaf(i) for i in indices)
        digests = tuple(
            g.content_digests[next(k for k, member in enumerate(g.rollouts) if member is r)]  # type: ignore[union-attr]
            for r, g in zip(rollouts, groups)
        )
        return RolloutBatch(
            indices=indices,
            rollouts=rollouts,  # type: ignore[arg-type]
            groups=groups,  # type: ignore[arg-type]
            logprobs=tuple(r.logprobs for r in rollouts),  # type: ignore[union-attr]
            model_versions=tuple(self._tree.entry(i)[1] for i in indices),
            rewards=tuple(r.reward for r in rollouts),  # type: ignore[union-attr]
            is_weights=tuple(self._is_weight(p, root_total, min_priority, n) for p in priorities),
            priorities=priorities,
            draw_integers=draws,
            root_total=root_total,
            min_priority_int=min_priority,
            op_counter=self._op_counter,
            content_digests=digests,
        )

    def witness_batch(
        self, batch: RolloutBatch, step: int, batch_rows: int, rows: Sequence[int], tensor_digest: str,
        declined: Sequence[int] = (),
    ) -> None:
        """Record which training-batch rows now hold which draws of ``batch``.

        ``rows[k]`` is the row that received draw ``placed[k]``, where
        ``placed`` is every draw position of the batch not in ``declined``
        (draws an adapter refused to train on; see ``evict`` with reason
        ``"drift"``). The record names the sample record
        (``batch.op_counter``), every placed draw with the content digest
        of the rollout it delivered, the declined draws, and
        ``tensor_digest``, the caller's commitment to the final tensors.
        The checker then proves each row holds the example its draw
        selected. Only the most recent batch can be witnessed, and only
        once; a no-op when attestation is off.
        """
        if batch.op_counter != self._op_counter or batch.op_counter < 1:
            raise ValueError(
                f"witness_batch: batch op_counter {batch.op_counter} is not the buffer's latest sample "
                f"({self._op_counter}); only the most recent batch can be witnessed"
            )
        if self._witnessed == batch.op_counter:
            raise ValueError(f"witness_batch: sample {batch.op_counter} was already witnessed")
        declined_set = {int(d) for d in declined}
        if len(declined_set) != len(declined) or any(not 0 <= d < len(batch.rollouts) for d in declined_set):
            raise ValueError(f"witness_batch: declined draws must be distinct positions of the batch, got {list(declined)}")
        placed = [k for k in range(len(batch.rollouts)) if k not in declined_set]
        if len(rows) != len(placed):
            raise ValueError(f"witness_batch: {len(rows)} rows for {len(placed)} placed draws")
        if not self._attester.enabled:
            self._witnessed = batch.op_counter
            return
        replaced = [(int(r), k, batch.content_digests[k]) for r, k in zip(rows, placed)]
        self._attester.record_batch(step, batch.op_counter, batch_rows, replaced, tensor_digest,
                                    declined=sorted(declined_set))
        self._witnessed = batch.op_counter

    def evict(self, position: int, reason: str = "explicit") -> None:
        """Remove the live entry at ``position`` with a recorded reason.

        ``"explicit"`` is the caller's decision; ``"drift"`` is an adapter
        declining an entry whose behavior logprobs drifted too far from
        the current policy. ``"stale"`` and ``"capacity"`` are the buffer's
        own reasons and cannot be given here; ``"quarantine"`` is written
        by ``quarantine``, which records why.
        """
        if reason not in ("explicit", "drift"):
            raise ValueError(f"evict reason must be 'explicit' or 'drift' (quarantine() for 'quarantine'), got {reason!r}")
        pos = self._to_index(position)
        if pos not in self._tree.entries:
            raise ValueError(f"evict: slot {pos} holds no live entry")
        event = self._tree.evict(pos, reason)  # type: ignore[arg-type]
        self._release_slot(pos)
        self._attester.record_write(event, self._op_counter)

    def quarantine(self, predicate: QuarantinePredicate, reason: str,
                   predicate_text: Optional[str] = None) -> tuple[int, ...]:
        """Evict every live entry ``predicate(rollout, group)`` selects; return their slots.

        Incident response. The predicate runs over copies of every live
        entry before anything changes; each match is then evicted with
        reason ``"quarantine"`` and a record carrying ``predicate_text``
        (derived from the predicate's source when not given) and
        ``reason``, the operator's note. Nothing is written when nothing
        matches; a buffer continuing a pre-format-3 log refuses. See
        ``rollout_quarantine``.
        """
        return run_quarantine(self, predicate, reason, predicate_text)

    def quarantine_positions(self, positions: Sequence[int], predicate_text: str, reason: str) -> None:
        """Quarantine the given live slots with the given texts; what the durable command log replays."""
        apply_quarantine(self, positions, predicate_text, reason)

    def record_telemetry(self, step: int, counts: dict, sample: Optional[RolloutBatch] = None,
                         reported: Optional[dict] = None) -> None:
        """Write a ``telemetry`` record for ``step``; see ``rollout_telemetry``.

        ``counts`` holds the integer counters the adapter observed
        (``batch_rows``, ``replaced_rows``, ``declined_rows``,
        ``dead_groups``, ``near_dead_groups``); ``sample`` is the step's
        batch when rows were replayed, from which the exact effective
        sample size and staleness are derived here and re-derived by the
        checker; ``reported`` holds float measurements the checker can only
        carry (the log-ratio statistics). A no-op when attestation is off.
        """
        if not self._attester.enabled:
            return
        exact = None
        if sample is not None:
            if sample.op_counter != self._op_counter:
                raise ValueError("record_telemetry: the sample is not the buffer's latest batch")
            exact = exact_telemetry(sample.is_weights, [self.current_version - v for v in sample.model_versions])
        self._attester.record_telemetry(step, counts, sample.op_counter if sample else None, exact, reported or {})

    def update_priorities(self, indices: Sequence[int], raw_scores: Sequence[float]) -> None:
        """Re-score live entries. All inputs are validated before any write.

        The entry keeps the version it was collected at, so its age (and
        decay) is unchanged, unless the buffer was built with
        ``reset_age_on_update=True``, in which case it is re-stamped at
        the current version.

        Raises
        ------
        ValueError
            Length mismatch, a non-live index, or a bad score.
        """
        idx = [self._to_index(i) for i in indices]
        scores = list(raw_scores)
        if len(idx) != len(scores):
            raise ValueError(
                f"indices and raw_scores differ in length: {len(idx)} vs {len(scores)}"
            )
        for position in idx:
            self._tree.entry(position)  # raises "no live entry"
        qs = [self._score_to_q(score, f"index {position}") for position, score in zip(idx, scores)]
        current = self.current_version
        for position, q in zip(idx, qs):
            old_version = self._tree.entry(position)[1]
            version = version_after_update(old_version, current, self.reset_age_on_update)
            event = self._tree.write(position, q, version)
            self._attester.record_write(event, self._op_counter)

    def close(self) -> None:
        """Close the attestation file, if one was opened."""
        self._attester.close()

    # -- snapshots ---------------------------------------------------------

    def _fingerprint(self) -> dict:
        """The construction parameters a snapshot must be loaded with."""
        return buffer_fingerprint(self)

    def state_dict(self) -> dict:
        """Complete buffer state as a JSON-serialisable dict; see ``rollout_snapshot.buffer_state_dict``."""
        return buffer_state_dict(self)

    def load_state_dict(self, state: dict) -> None:
        """Rebuild this (fresh) buffer from a ``state_dict()``; see ``rollout_snapshot.load_buffer_state``."""
        load_buffer_state(self, state)

    def __enter__(self) -> "RolloutBuffer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _score_to_q(self, raw: float, context: str) -> int:
        """raw -> raw**alpha (float, once) -> fixed-point q. Errors name ``context``."""
        try:
            score = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{context}: score must be a real number, got {raw!r}") from exc
        if not math.isfinite(score) or score < 0:
            raise ValueError(f"{context}: score must be finite and >= 0, got {raw!r}")
        x = score if self.alpha == 1.0 else score ** self.alpha
        try:
            return quantize_priority(x, self._params)
        except ValueError as exc:
            raise ValueError(f"{context}: {exc}") from exc

    def _is_weight(self, priority: int, root_total: int, min_priority: int, n: int) -> Fraction:
        """Normalised importance-sampling weight for one sampled slot.

        ``w_i = (N * P(i)) ** -beta``, divided by the largest possible
        weight (the one the smallest positive leaf would get) so every
        weight lies in (0, 1]. The two ``** -beta`` calls are the declared
        float boundary; the division is an exact Fraction. Same formula as
        ``ExactPERBuffer``.
        """
        w_i = Fraction((n * priority / root_total) ** (-self.beta))
        w_max = Fraction((n * min_priority / root_total) ** (-self.beta))
        return Fraction(w_i, w_max)

    def _oldest_live(self, count: int) -> list[int]:
        """The ``count`` live positions with the lowest (version, insertion order).

        These are the capacity-eviction victims, oldest first. One pass
        over the live entries per call; the buffer does not keep a
        separate age-ordered structure.
        """
        entries = self._tree.entries
        return heapq.nsmallest(
            count,
            entries,
            key=lambda p: (entries[p][1], self._inserted[p]),
        )

    def _allocate(self, count: int) -> tuple[int, ...]:
        """Return ``count`` empty slots, evicting the oldest entries if needed."""
        shortfall = count - len(self._free)
        if shortfall > 0:
            for victim in self._oldest_live(shortfall):
                event = self._tree.evict(victim, "capacity")
                self._release_slot(victim)
                self._attester.record_write(event, self._op_counter)
        return tuple(heapq.heappop(self._free) for _ in range(count))

    def _release_slot(self, position: int) -> None:
        """Forget the stored rollout and group and put the slot back on the free heap."""
        self._rollouts[position] = None
        self._groups[position] = None
        heapq.heappush(self._free, position)

    @staticmethod
    def _to_index(value: object) -> int:
        """Accept true integers only (int, numpy ints); floats and strings are errors."""
        if isinstance(value, bool):
            raise ValueError(f"index must be an int, got {value!r}")
        try:
            return operator.index(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"index must be an int, got {value!r}") from exc

    def __repr__(self) -> str:
        return (
            f"RolloutBuffer(capacity={self.capacity}, size={self.size}, "
            f"version={self.current_version}, priority={self.priority!r})"
        )
