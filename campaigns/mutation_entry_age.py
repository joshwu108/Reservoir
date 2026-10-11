"""
campaigns.mutation_entry_age — Forgeries of the declared minimum entry age.

A ``sample`` record may declare ``min_entry_age`` and carry, on every draw,
the ``entry_version`` of the slot it drew; the checker then requires each
drawn slot to be at least that many versions old at the buffer's version
and each declared version to be the slot's replayed one. The library does
not write these fields yet, so the honest baseline here is synthetic:
``with_entry_age`` annotates a library log with the fields exactly as the
replay says they must read, and re-chains it. Mutants then break one claim:
a draw of a slot inserted at the current step, a declared version older
than the slot's, a draw without its version, the minimum dropped from or
changed on a later sample, the minimum in a log without decay fields.
Every one must be rejected.
"""

from __future__ import annotations

import copy

from campaigns.mutation_reasons import rejection, tally
from reservoir_checker.verify import verify_chain

Mutant = tuple[list[dict], str]


def _rechain(records: list[dict], start: int) -> list[dict]:
    from reservoir.attest import _digest_record

    out = copy.deepcopy(records)
    prev = out[start - 1]["digest"] if start > 0 else "genesis"
    for rec in out[start:]:
        rec["prev_digest"] = prev
        rec["digest"] = _digest_record(rec)
        prev = rec["digest"]
    return out


def with_entry_age(records: list[dict], min_entry_age: int) -> list[dict]:
    """``records`` with ``min_entry_age`` on every sample and each draw's slot version, re-chained."""
    out = copy.deepcopy(records)
    versions: dict[int, str] = {}
    for rec in out:
        if rec["op"] in ("insert", "update"):
            versions[rec["index"]] = rec["entry_version"]
        elif rec["op"] == "evict":
            versions.pop(rec["index"], None)
        elif rec["op"] == "sample":
            rec["min_entry_age"] = str(min_entry_age)
            for draw in rec["samples"]:
                draw["entry_version"] = versions[draw["leaf_index"]]
    return _rechain(out, 0)


def _run(sample_ahead: bool, seed: int = 6) -> list[dict]:
    """Groups stored at version v, then sampled at v + 1 (``sample_ahead``) or at v itself."""
    from reservoir.attest import AttestationLog
    from reservoir.rollout import Rollout
    from reservoir.rollout_buffer import RolloutBuffer

    buf = RolloutBuffer(capacity=16, half_life=2, max_policy_age=6, seed=seed, attest=AttestationLog())
    for v in range(4):
        buf.add_group(f"g{v}", v, [Rollout(tokens=[v + 1, i + 2], logprobs=[-0.1, -0.2], reward=r)
                                   for i, r in enumerate((1.0, 0.0, 0.5))], source="s")
        buf.sample(3, current_version=v + 1 if sample_ahead else v)
    return [dict(r) for r in buf.attestation_log.records]


def build_entry_age_chain() -> list[dict]:
    """Every draw is of a slot at least one version old: an honest log for ``min_entry_age`` 1."""
    records = with_entry_age(_run(sample_ahead=True), 1)
    verify_chain(records)
    return records


def build_same_step_chain() -> list[dict]:
    """Sampling right after storing the step's group, declared with ``min_entry_age`` 1."""
    records = with_entry_age(_run(sample_ahead=False), 1)
    if not any(d["entry_version"] == _version_at(records, i) for i, r in enumerate(records) if r["op"] == "sample"
               for d in r["samples"]):
        raise RuntimeError("no draw landed on a slot of the current step; change the seed")
    return records


def _version_at(records: list[dict], idx: int) -> str:
    return str(max((int(r["new_version"]) for r in records[:idx] if r["op"] == "advance_version"), default=0))


def _edited(records: list[dict], idx: int, edit) -> list[dict]:
    m = copy.deepcopy(records)
    edit(m[idx])
    return _rechain(m, idx)


def entry_age_mutants(base: list[dict], same_step: list[dict]) -> list[Mutant]:
    samples = [i for i, r in enumerate(base) if r["op"] == "sample"]
    s, later = samples[0], samples[-1]
    older = next(k for k, d in enumerate(base[later]["samples"]) if int(d["entry_version"]) > 0)

    def legacy(records: list[dict]) -> list[dict]:
        m = copy.deepcopy(records[1:])
        for r in m:
            for name in ("base_priority_int", "entry_version", "base_epoch", "reason", "content_digest", "source"):
                r.pop(name, None)
        return [r for r in _rechain(m, 0) if r["op"] in ("insert", "update", "evict", "sample")]

    return [
        (same_step, "sample_draws_same_step_slot"),
        (_edited(base, later, lambda r: r["samples"][older].update(entry_version="0")), "sample_draw_entry_version_forged"),
        (_edited(base, s, lambda r: r["samples"][0].pop("entry_version")), "sample_draw_entry_version_missing"),
        (_edited(base, later, lambda r: r.pop("min_entry_age")), "min_entry_age_dropped_from_a_later_sample"),
        (_edited(base, later, lambda r: r.update(min_entry_age="0")), "min_entry_age_changed_between_samples"),
        (_edited(base, s, lambda r: r.update(min_entry_age="-1")), "min_entry_age_malformed"),
        (_rechain(legacy(base), 0), "min_entry_age_in_a_log_without_decay_fields"),
    ]


def run_entry_age_category(results: dict) -> None:
    """Append the ``entry_age`` category to ``results``."""
    mutants = entry_age_mutants(build_entry_age_chain(), build_same_step_chain())
    tally(results, "entry_age", [(description, rejection(lambda r=records: verify_chain(r, 16)))
                                 for records, description in mutants])
