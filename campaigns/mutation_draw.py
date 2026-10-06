"""
campaigns.mutation_draw — Forgeries of draws and importance weights.

Extends ``campaigns.mutation`` with the forgeries that only a log carrying
the buffer's ``seed``, ``buffer_id``, ``alpha`` and ``beta`` in its
``decay_config`` lets the checker catch:

- a ``draw_int`` moved to another value inside the same leaf's range, so
  the range check (``prefix_sum_locate(draw) == leaf_index``) still holds;
- an importance weight replaced by another reduced fraction in ``(0, 1]``;
- the recorded seed, buffer id or beta changed;
- a whole ``sample`` record deleted, which keeps every remaining draw in
  range but shifts the draw counter later draws are keyed on;
- a partial or malformed draw configuration.

The campaign also measures, as a documented limit, how many of the first
two survive when the same log is stripped of its draw configuration: those
are the forgeries the log written before version 0.5.0 could not see. The statement is
checked, so the report can never claim what was not measured.
"""

from __future__ import annotations

import copy
from fractions import Fraction

from checker.verify import CheckerError, verify_chain

Mutant = tuple[list[dict], str]


def build_seeded_chain(seed: int = 5, buffer_id: int = 9) -> list[dict]:
    """A RolloutBuffer run whose decay_config carries the draw configuration."""
    from reservoir.attest import AttestationLog
    from reservoir.rollout import Rollout
    from reservoir.rollout_buffer import RolloutBuffer

    log = AttestationLog()
    buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=2, seed=seed, buffer_id=buffer_id, attest=log)
    for v in range(8):
        buf.add_group(f"g{v}", v, [Rollout(tokens=[v + 1, 2], logprobs=[-0.1, -0.2], reward=r)
                                   for r in (1.0, 0.0, 0.5)], source="s")
        buf.sample(4, current_version=v)
    records = [dict(r) for r in log.records]
    verify_chain(records)
    return records


def _rechain(records: list[dict], start: int) -> list[dict]:
    from reservoir.attest import _digest_record

    out = copy.deepcopy(records)
    prev = out[start - 1]["digest"] if start > 0 else "genesis"
    for rec in out[start:]:
        rec["prev_digest"] = prev
        rec["digest"] = _digest_record(rec)
        prev = rec["digest"]
    return out


def _samples(records: list[dict]) -> list[int]:
    return [i for i, r in enumerate(records) if r["op"] == "sample"]


def _moved_draw(records: list[dict], idx: int, k: int) -> list[dict]:
    """Move draw k of sample record idx by one, staying inside [0, root_total)."""
    m = copy.deepcopy(records)
    s = m[idx]["samples"][k]
    draw, total = int(s["draw_int"]), int(m[idx]["root_total"])
    s["draw_int"] = str(draw + 1 if draw + 1 < total else draw - 1)
    return _rechain(m, idx)


def _reweighted(records: list[dict], idx: int, k: int) -> list[dict]:
    m = copy.deepcopy(records)
    s = m[idx]["samples"][k]
    w = Fraction(int(s["is_weight_num"]), int(s["is_weight_den"]))
    forged = w / 2 if w > Fraction(1, 2) else (w * 2 if w * 2 <= 1 else w / 3)
    s["is_weight_num"], s["is_weight_den"] = str(forged.numerator), str(forged.denominator)
    return _rechain(m, idx)


def _config_edit(records: list[dict], **fields) -> list[dict]:
    m = copy.deepcopy(records)
    for name, value in fields.items():
        if value is None:
            m[0].pop(name, None)
        else:
            m[0][name] = value
    return _rechain(m, 0)


def draw_mutants(base: list[dict]) -> list[Mutant]:
    idxs = _samples(base)
    mutants: list[Mutant] = []
    for idx in idxs[:3]:
        for k in range(min(2, len(base[idx]["samples"]))):
            mutants.append((_moved_draw(base, idx, k), f"draw_moved_record{idx}_sample{k}"))
            mutants.append((_reweighted(base, idx, k), f"is_weight_replaced_record{idx}_sample{k}"))
    mutants += [
        (_config_edit(base, seed=str(int(base[0]["seed"]) + 1)), "config_seed_changed"),
        (_config_edit(base, buffer_id=str(int(base[0]["buffer_id"]) + 1)), "config_buffer_id_changed"),
        (_config_edit(base, beta=(0.9).hex()), "config_beta_changed"),
        (_config_edit(base, seed=None), "config_seed_missing"),
        (_config_edit(base, alpha="1.0"), "config_alpha_not_hex"),
        (_config_edit(base, beta=(-0.1).hex()), "config_beta_negative"),
        (_config_edit(base, beta=(2000.0).hex()), "config_beta_overflows_weights"),
        (_config_edit(base, beta=(1e308).hex()), "config_beta_huge"),
    ]
    deleted = copy.deepcopy(base)
    del deleted[idxs[0]]
    mutants.append((_rechain(deleted, idxs[0]), "sample_record_deleted"))
    return mutants


def limit_mutants(base: list[dict]) -> list[Mutant]:
    """The two forgeries whose detection needs the draw configuration."""
    idx = _samples(base)[0]
    return [(_moved_draw(base, idx, 0), "draw_moved_within_leaf"), (_reweighted(base, idx, 0), "is_weight_replaced")]


def _without_draw_config(records: list[dict]) -> list[dict]:
    return _config_edit(records, seed=None, buffer_id=None, alpha=None, beta=None)


def rejected(records: list[dict]) -> bool:
    try:
        verify_chain(records)
    except CheckerError:
        return True
    return False


def run_draw_category(results: dict) -> None:
    """Append the ``draw`` category and the ``draw_limit`` measurement to ``results``."""
    base = build_seeded_chain()
    mutants = draw_mutants(base)
    count = 0
    for records, description in mutants:
        results["total_mutants"] += 1
        if rejected(records):
            results["rejected"] += 1
            count += 1
        else:
            results["survived"].append(description)
    results["details"].append(("draw", len(mutants), count))

    # Measured limit of a log without the draw configuration: a draw moved
    # inside its leaf, or a reweighted sample, can survive the range and
    # reduced-form checks. Both must be rejected once the configuration is
    # present, or the campaign fails.
    limits = limit_mutants(base)
    with_config = [name for records, name in limits if rejected(records)]
    without = [name for records, name in limits if not rejected(_without_draw_config(records))]
    if len(with_config) != len(limits):
        results["survived"].append("draw_limit_not_rejected_with_config")
    results["draw_limit"] = {
        "mutants": [name for _, name in limits],
        "survive_without_draw_config": without,
        "rejected_with_draw_config": with_config,
        "statement": "a draw moved inside its leaf's range, or a reweighted sample, is invisible to a log "
                     "that does not record seed, buffer_id and beta; a format-3 log records them and both are rejected",
    }
