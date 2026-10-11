"""
campaigns.mutation_decay — Age-decay protocol forgeries.

Built from a RolloutBuffer log (decay_config, advance_version, rebase and
the decay fields on every mutation). Every mutant is re-chained so only
the semantic replay in ``reservoir_checker.decay_replay`` can catch it: a
leaf that is not the decay formula's value, a wrong epoch or version, an
expired entry never evicted, a rebase that is early, late, missing or
non-canonical, and malformed or hostile configuration fields.
"""

from __future__ import annotations

import copy

from reservoir_checker.verify import verify_chain


def _recompute_digests_from(records: list[dict], start_idx: int) -> list[dict]:
    """Re-chain ``records[start_idx:]`` so only the semantic checks can catch the edit."""
    from reservoir.attest import _digest_record

    result = copy.deepcopy(records)
    prev_digest = result[start_idx - 1]["digest"] if start_idx > 0 else "genesis"
    for rec in result[start_idx:]:
        rec["prev_digest"] = prev_digest
        rec["digest"] = _digest_record(rec)
        prev_digest = rec["digest"]
    return result


def _build_decay_chain(seed: int = 11) -> list[dict]:
    """A RolloutBuffer run with inserts, updates, stale/capacity evictions and rebases."""
    from reservoir.attest import AttestationLog as _BufferLog
    from reservoir.rollout import Rollout
    from reservoir.rollout_buffer import RolloutBuffer

    log = _BufferLog()
    buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=2, seed=seed, attest=log)
    for v in range(10):
        buf.add_group(
            f"g{v}", v,
            [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in (1.0, 0.0, 0.5)],
        )
        batch = buf.sample(4, current_version=v)
        buf.update_priorities(batch.indices[:2], [0.3, 0.6])
    assert buf.n_rebases >= 1
    records = [dict(r) for r in log.records]
    verify_chain(records)  # baseline must be valid
    return records


def _first(records: list[dict], op: str, **match) -> int:
    """Index of the first record with this op whose fields equal ``match``."""
    for i, r in enumerate(records):
        if r["op"] == op and all(r.get(k) == v for k, v in match.items()):
            return i
    raise RuntimeError(f"baseline has no {op} record matching {match}")


def _restamped(records: list[dict], idx: int) -> list[dict]:
    """An update re-stamped one version newer with the leaf recomputed for it, in a log that keeps versions."""
    from reservoir_checker.decay_replay import inflated_priority, parse_config

    mutant = copy.deepcopy(records)
    rec = mutant[idx]
    rec["entry_version"] = str(int(rec["entry_version"]) + 1)
    leaf = inflated_priority(int(rec["base_priority_int"]), int(rec["entry_version"]), int(rec["base_epoch"]),
                             parse_config(records[0], 0))
    rec["new_priority_int"] = str(leaf)
    return _recompute_digests_from(mutant, idx)


def _version_at(records: list[dict], idx: int) -> int:
    """The buffer version in force at record ``idx`` (the last advance before it)."""
    return max((int(r["new_version"]) for r in records[:idx] if r["op"] == "advance_version"), default=0)


def _bump(records: list[dict], idx: int, field: str, delta: int = 1) -> list[dict]:
    """Copy records, add ``delta`` to an integer-as-string field, re-chain from idx."""
    mutant = copy.deepcopy(records)
    mutant[idx][field] = str(int(mutant[idx][field]) + delta)
    return _recompute_digests_from(mutant, idx)


def decay_mutants() -> list[tuple[list[dict], str]]:
    """One mutant per rule the decay replay enforces, plus a few variants."""
    base = _build_decay_chain()
    return _decay_protocol_mutants(base) + _decay_hardening_mutants(base)


def _decay_protocol_mutants(base: list[dict]) -> list[tuple[list[dict], str]]:
    """Forgeries of the decay bookkeeping: config, leaves, versions, staleness, rebase."""
    mutants: list[tuple[list[dict], str]] = []

    # config placement / presence
    moved = base[1:2] + base[0:1] + base[2:]
    mutants.append((_recompute_digests_from(moved, 0), "decay_config_not_first"))
    mutants.append((_recompute_digests_from(copy.deepcopy(base[1:]), 0), "decay_config_removed"))
    cfg_tamper = copy.deepcopy(base)
    cfg_tamper[0]["half_life"] = "2"
    mutants.append((_recompute_digests_from(cfg_tamper, 0), "decay_config_half_life_changed"))

    # leaf must equal the recomputed decay formula
    for n, i in enumerate([j for j, r in enumerate(base) if r["op"] == "insert"][:3]):
        mutants.append((_bump(base, i, "new_priority_int"), f"insert_leaf_plus_one_{n}"))
        mutants.append((_bump(base, i, "base_priority_int"), f"insert_q_plus_one_{n}"))
    u = _first(base, "update")
    mutants.append((_bump(base, u, "new_priority_int", -1), "update_leaf_minus_one"))
    mutants.append((_bump(base, u, "base_priority_int", 7), "update_q_plus_seven"))

    # epoch / version bookkeeping
    i = _first(base, "insert")
    mutants.append((_bump(base, i, "base_epoch"), "insert_wrong_base_epoch"))
    mutants.append((_bump(base, i, "entry_version"), "insert_version_newer_than_current"))
    aged = next(j for j, r in enumerate(base)
                if r["op"] == "update" and int(r["entry_version"]) < _version_at(base, j))
    mutants.append((_restamped(base, aged), "update_version_changed_without_reset"))
    advances = [j for j, r in enumerate(base) if r["op"] == "advance_version"]
    backwards = copy.deepcopy(base)
    backwards[advances[1]]["new_version"] = str(int(backwards[advances[1]]["old_version"]) - 1)
    mutants.append((_recompute_digests_from(backwards, advances[1]), "advance_version_backwards"))
    adv = copy.deepcopy(base)
    adv[advances[1]]["old_version"] = str(int(adv[advances[1]]["old_version"]) + 1)
    adv[advances[1]]["new_version"] = str(int(adv[advances[1]]["new_version"]) + 1)
    mutants.append((_recompute_digests_from(adv, advances[1]), "advance_version_old_mismatch"))

    return mutants + _decay_lifecycle_mutants(base)


def _decay_lifecycle_mutants(base: list[dict]) -> list[tuple[list[dict], str]]:
    """Forgeries of the staleness and rebase protocol."""
    mutants: list[tuple[list[dict], str]] = []

    # staleness protocol
    s = _first(base, "evict", reason="stale")
    deleted = copy.deepcopy(base)
    deleted.pop(s)
    mutants.append((_recompute_digests_from(deleted, s), "expired_entry_never_evicted"))
    relabel = copy.deepcopy(base)
    relabel[s]["reason"] = "capacity"
    mutants.append((_recompute_digests_from(relabel, s), "stale_evict_relabelled_capacity"))
    c = _first(base, "evict", reason="capacity")
    relabel2 = copy.deepcopy(base)
    relabel2[c]["reason"] = "stale"
    mutants.append((_recompute_digests_from(relabel2, c), "live_entry_evicted_as_stale"))
    noreason = copy.deepcopy(base)
    del noreason[c]["reason"]
    mutants.append((_recompute_digests_from(noreason, c), "evict_reason_missing"))

    # rebase
    r = _first(base, "rebase")
    swapped = copy.deepcopy(base)
    swapped[r - 1], swapped[r] = swapped[r], swapped[r - 1]
    mutants.append((_recompute_digests_from(swapped, r - 1), "rebase_before_last_stale_evict"))
    mutants.append((_bump(base, r, "new_base_epoch"), "rebase_non_canonical_epoch"))
    mutants.append((_bump(base, r, "root_total_after"), "rebase_total_after_wrong"))
    mutants.append((_bump(base, r, "root_total_before"), "rebase_total_before_wrong"))
    removed = copy.deepcopy(base)
    removed.pop(r)
    mutants.append((_recompute_digests_from(removed, r), "rebase_record_removed"))

    # sample after a rebase must use the shifted tree
    smp = next(j for j in range(r, len(base)) if base[j]["op"] == "sample")
    mutants.append((_bump(base, smp, "root_total", 1), "sample_total_ignores_rebase"))
    return mutants


def _decay_hardening_mutants(base: list[dict]) -> list[tuple[list[dict], str]]:
    """Malformed-but-digested inputs: index range, no-op rebase, field encoding,
    a stray reason, and configs that would stall or overflow the replay."""
    mutants: list[tuple[list[dict], str]] = []
    i = _first(base, "insert")
    r = _first(base, "rebase")
    for bad_index in (-1, 8, 10**6):
        m = copy.deepcopy(base)
        m[i]["index"] = bad_index
        mutants.append((_recompute_digests_from(m, i), f"insert_index_{bad_index}"))
    noop = copy.deepcopy(base)
    noop[r]["new_base_epoch"] = noop[r]["old_base_epoch"]
    mutants.append((_recompute_digests_from(noop, r), "rebase_noop"))
    for bad_value in (5, 5.9, True, "-1", " 7"):
        m = copy.deepcopy(base)
        m[i]["entry_version"] = bad_value
        mutants.append((_recompute_digests_from(m, i), f"entry_version_encoded_as_{bad_value!r}"))
    stray = copy.deepcopy(base)
    stray[i]["reason"] = "explicit"
    mutants.append((_recompute_digests_from(stray, i), "reason_on_insert"))
    hostile = copy.deepcopy(base)
    hostile[0]["half_life"] = "200000"
    mutants.append((_recompute_digests_from(hostile, 0), "config_half_life_hostile"))
    budget = copy.deepcopy(base)
    budget[0]["max_policy_age"] = "100"
    mutants.append((_recompute_digests_from(budget, 0), "config_bit_budget_violated"))
    return mutants
