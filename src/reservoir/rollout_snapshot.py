"""
reservoir.rollout_snapshot — Serialise and validate RolloutBuffer snapshots.

``RolloutBuffer.state_dict()`` and ``load_state_dict()`` delegate to
``buffer_state_dict`` and ``load_buffer_state`` here; the per-group and
per-slot helpers below do the validation. Everything that enters from a snapshot
is treated as untrusted input: a snapshot is a file on disk and may have
been corrupted or edited, so every index, counter and reference is
checked before the buffer is rebuilt from it.

Nothing here is specific to durability; a snapshot is also a plain
checkpoint. ``durable_rollout.py`` adds the crash-atomic commit protocol
around it.
"""

from __future__ import annotations

import dataclasses
import json

import heapq
from typing import Optional

from reservoir.priorities import PriorityStrategy
from reservoir.rollout import Rollout, RolloutGroup, default_is_success


def _strategy_fingerprint(strategy: PriorityStrategy) -> dict:
    """Class name plus parameters (for dataclass strategies), so a snapshot
    saved with ``AdvantagePriority(epsilon=0.1)`` is not loaded with 0.5."""
    fingerprint: dict = {"type": type(strategy).__name__}
    if dataclasses.is_dataclass(strategy):
        fingerprint.update(dataclasses.asdict(strategy))
    return fingerprint


def _snapshot_int(container: dict, name: str, minimum: int = 0) -> int:
    """A true non-negative int from a snapshot (strings allowed for big ints)."""
    value = container.get(name)
    if isinstance(value, str) and value.isascii() and value.isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"snapshot field {name} must be an int >= {minimum}, got {value!r}")
    return value


def _require_list(state: dict, name: str) -> list:
    """The snapshot field ``name`` as a list, or ValueError."""
    value = state.get(name)
    if not isinstance(value, list):
        raise ValueError(f"snapshot field {name} must be a list")
    return value


def _validate_slots(slots: list, capacity: int, groups: list[RolloutGroup]) -> dict[int, dict]:
    """Check every slot reference and return ``{position: slot}`` for live slots.

    Rejects a wrong slot count, out-of-range or negative group/member
    indices, two slots claiming the same rollout, and duplicate insertion
    counters, all of which would load into a buffer that behaves
    differently from the one that was saved.
    """
    if len(slots) != capacity:
        raise ValueError(f"snapshot has {len(slots)} slots for capacity {capacity}")
    live: dict[int, dict] = {}
    seen_members: set[tuple[int, int]] = set()
    seen_inserted: set[int] = set()
    for position, raw in enumerate(slots):
        if raw is None:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"slot {position} must be an object or null")
        group = _snapshot_int(raw, "group")
        if group >= len(groups):
            raise ValueError(f"slot {position} references group {group} of {len(groups)}")
        member = _snapshot_int(raw, "member")
        if member >= groups[group].size:
            raise ValueError(f"slot {position} references member {member} of group {group}")
        if (group, member) in seen_members:
            raise ValueError(f"slot {position} duplicates rollout ({group}, {member})")
        seen_members.add((group, member))
        inserted = _snapshot_int(raw, "inserted", minimum=1)
        if inserted in seen_inserted:
            raise ValueError(f"slot {position} duplicates insertion counter {inserted}")
        seen_inserted.add(inserted)
        live[position] = {
            "group": group, "member": member, "inserted": inserted,
            "q": _snapshot_int(raw, "q"), "version": _snapshot_int(raw, "version"),
        }
    return live


def _require_json_round_trip(metadata: dict, where: str) -> None:
    """Metadata must come back from JSON unchanged (tuples and non-str keys do not)."""
    try:
        restored = json.loads(json.dumps(metadata))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} has metadata that is not JSON-serialisable: {exc}") from exc
    if restored != metadata:
        raise ValueError(
            f"{where} has metadata that JSON would alter (tuples become lists, "
            f"non-string keys become strings): {metadata!r}"
        )


def _group_to_dict(group: RolloutGroup) -> dict:
    """Serialise a group with all its rollouts; rejects what cannot be saved faithfully."""
    if group.is_success is not default_is_success:
        raise ValueError(
            f"group {group.prompt_id!r} uses a custom is_success predicate, which cannot "
            f"be saved; snapshots support the default predicate only"
        )
    rollouts = []
    for k, r in enumerate(group.rollouts):
        metadata = dict(r.metadata)
        _require_json_round_trip(metadata, f"rollout {k} of group {group.prompt_id!r}")
        rollouts.append({
            "tokens": list(r.tokens), "logprobs": list(r.logprobs),
            "reward": r.reward, "metadata": metadata,
        })
    return {
        "prompt_id": group.prompt_id, "model_version": group.model_version,
        "source": group.source, "rollouts": rollouts,
    }


def _group_from_dict(data: dict) -> RolloutGroup:
    """Rebuild a group; Rollout and RolloutGroup re-validate every field.

    ``source`` is optional so snapshots written before it existed still load.
    """
    return RolloutGroup(
        prompt_id=data["prompt_id"],
        model_version=int(data["model_version"]),
        source=data.get("source"),
        rollouts=[
            Rollout(tokens=r["tokens"], logprobs=r["logprobs"], reward=r["reward"],
                    metadata=r.get("metadata") or None)
            for r in data["rollouts"]
        ],
    )


# ---------------------------------------------------------------------------
# Whole-buffer snapshots
# ---------------------------------------------------------------------------

def buffer_fingerprint(buf) -> dict:
    """The construction parameters a snapshot must be loaded with."""
    p = buf._params
    return {
        "half_life": p.half_life, "max_policy_age": p.max_policy_age,
        "capacity": p.capacity, "priority_bits": p.priority_bits,
        "priority_frac_bits": p.priority_frac_bits, "table_frac_bits": p.table_frac_bits,
        "rebase_slack": p.rebase_slack, "alpha": buf.alpha, "beta": buf.beta,
        "seed": buf.seed, "buffer_id": buf.buffer_id,
        "reset_age_on_update": buf.reset_age_on_update,
        "priority": _strategy_fingerprint(buf.priority),
    }

def buffer_state_dict(buf) -> dict:
    """Complete buffer state as a JSON-serialisable dict.

    Groups are stored once each with all their rollouts (including any
    already evicted from the buffer, since group statistics depend on
    them) and their ``source``; live slots reference a group and a
    member index. The attestation records and manifest lines are part
    of the state, so a restored buffer continues the same chain. Integers
    that may exceed 2^53 are stored as strings. Rollout metadata must
    be JSON-serialisable and groups must use the default success
    predicate, because a callable cannot be saved; both are checked
    here with a clear error.
    """
    groups: list[RolloutGroup] = []
    group_index: dict[int, int] = {}
    slots: list[Optional[dict]] = []
    for position in range(buf.capacity):
        group = buf._groups[position]
        if group is None:
            slots.append(None)
            continue
        if id(group) not in group_index:
            group_index[id(group)] = len(groups)
            groups.append(group)
        rollout = buf._rollouts[position]
        member = next(k for k, r in enumerate(group.rollouts) if r is rollout)
        q, version = buf._tree.entry(position)
        slots.append({
            "group": group_index[id(group)], "member": member,
            "q": str(q), "version": version, "inserted": buf._inserted[position],
        })
    log = buf._attester.log
    return {
        "format": buf.STATE_FORMAT,
        "fingerprint": buf._fingerprint(),
        "current_version": buf.current_version,
        "base_epoch": buf.base_epoch,
        "op_counter": buf._op_counter,
        "witnessed": buf._witnessed,
        "last_sample": buf._last_sample,
        "draw_counter": buf._draw_counter,
        "insert_seq": buf._insert_seq,
        "n_rebases": buf._n_rebases,
        "groups": [_group_to_dict(g) for g in groups],
        "slots": slots,
        "attestation": log.records if log is not None else None,
        "manifest": buf._attester.manifest_records if buf._attester.has_manifest else None,
    }

LAST_SAMPLE_KEYS = frozenset(
    {"indices", "draws", "root_total", "min_priority", "n", "priorities", "op_counter", "versions", "inserted"}
)


def _validate_last_sample(last_sample):
    """The ``last_sample`` record of a snapshot, or None; anything else is a ValueError."""
    if last_sample is None:
        return None
    if not isinstance(last_sample, dict) or not LAST_SAMPLE_KEYS <= set(last_sample):
        raise ValueError("snapshot field last_sample must be null or the record sample() writes")
    n = len(last_sample["indices"])
    if any(len(last_sample[k]) != n for k in ("draws", "priorities", "versions", "inserted")):
        raise ValueError("snapshot field last_sample has lists of different lengths")
    return last_sample


def _restore_slots(buf, slots: dict, groups: list, counters: dict) -> None:
    """Fill the trees, slot arrays, free list and counters from validated snapshot parts."""
    entries = {pos: (slot["q"], slot["version"]) for pos, slot in slots.items()}
    buf._tree.restore(counters["base_epoch"], counters["current_version"], entries)
    for position, slot in slots.items():
        group = groups[slot["group"]]
        buf._groups[position] = group
        buf._rollouts[position] = group.rollouts[slot["member"]]
        buf._inserted[position] = slot["inserted"]
    buf._free = [p for p in range(buf.capacity) if p not in entries]
    heapq.heapify(buf._free)
    buf._op_counter = counters["op_counter"]
    buf._draw_counter = counters["draw_counter"]
    buf._insert_seq = counters["insert_seq"]
    buf._n_rebases = counters["n_rebases"]


def _validate_logs(buf, state: dict) -> tuple[list, list]:
    """The attestation records and manifest lines of a snapshot, checked against this buffer's settings."""
    witnessed = state.get("witnessed", -1)
    if isinstance(witnessed, bool) or not isinstance(witnessed, int) or witnessed < -1:
        raise ValueError(f"snapshot field witnessed must be an int >= -1, got {witnessed!r}")
    buf._witnessed = witnessed
    records = state.get("attestation")
    if records is not None and not isinstance(records, list):
        raise ValueError("snapshot attestation must be a list of records or null")
    manifest = state.get("manifest")
    if manifest is not None and not isinstance(manifest, list):
        raise ValueError("snapshot manifest must be a list of records or null")
    has_digests = any(
        isinstance(r, dict) and r.get("op") == "insert" and "content_digest" in r for r in records or []
    )
    if manifest is None and buf._attester.has_manifest and has_digests:
        raise ValueError("saved state has no manifest but this buffer keeps one; reopen with manifest=None")
    if manifest is not None and not buf._attester.has_manifest:
        raise ValueError("saved state carries a manifest; reopen with manifest=<path>")
    return records or [], manifest or []


def load_buffer_state(buf, state: dict) -> None:
    """Rebuild a fresh ``buf`` from a ``state_dict()``.

    Raises
    ------
    ValueError
        If the buffer is not fresh, the snapshot format or construction
        parameters do not match, or any value fails validation. A
        failed load leaves the buffer unusable; construct a new one.
    """
    if buf.size or buf.current_version or buf._op_counter or buf._insert_seq:
        raise ValueError("load_state_dict() requires a freshly constructed buffer")
    if not isinstance(state, dict) or state.get("format") != buf.STATE_FORMAT:
        raise ValueError(f"unsupported snapshot format: {state.get('format') if isinstance(state, dict) else state!r}")
    if state.get("fingerprint") != buf._fingerprint():
        raise ValueError(
            "snapshot was written by a buffer with different parameters: "
            f"{state.get('fingerprint')} vs {buf._fingerprint()}"
        )
    groups = [_group_from_dict(g) for g in _require_list(state, "groups")]
    slots = _validate_slots(_require_list(state, "slots"), buf.capacity, groups)
    counters = {name: _snapshot_int(state, name) for name in
                ("base_epoch", "current_version", "op_counter", "draw_counter",
                 "insert_seq", "n_rebases")}
    inserted_values = [slot["inserted"] for slot in slots.values()]
    if inserted_values and max(inserted_values) > counters["insert_seq"]:
        raise ValueError("a slot's insertion counter exceeds insert_seq")

    _restore_slots(buf, slots, groups, counters)
    buf._last_sample = _validate_last_sample(state.get("last_sample"))
    records, manifest = _validate_logs(buf, state)
    buf._attester.restore(records or [], manifest or [])
