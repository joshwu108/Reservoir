"""
reservoir.rollout_snapshot — Serialise and validate RolloutBuffer snapshots.

``RolloutBuffer.state_dict()`` and ``load_state_dict()`` delegate the
per-group and per-slot work here. Everything that enters from a snapshot
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
    return {"prompt_id": group.prompt_id, "model_version": group.model_version, "rollouts": rollouts}


def _group_from_dict(data: dict) -> RolloutGroup:
    """Rebuild a group; Rollout and RolloutGroup re-validate every field."""
    return RolloutGroup(
        prompt_id=data["prompt_id"],
        model_version=int(data["model_version"]),
        rollouts=[
            Rollout(tokens=r["tokens"], logprobs=r["logprobs"], reward=r["reward"],
                    metadata=r.get("metadata") or None)
            for r in data["rollouts"]
        ],
    )
