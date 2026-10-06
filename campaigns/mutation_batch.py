"""
campaigns.mutation_batch — Forgeries of batch witnesses.

A ``batch`` record claims which training-batch row holds which draw of a
sample record and the content digest of the example in it. These mutants
re-chain the log and change one claim: a row pointed at the wrong draw, a
digest swapped, a draw placed twice, a row replaced twice, a row outside
the batch, a nonexistent draw, a witness for the wrong or a nonexistent
sample, a duplicated witness, a witness placed between an advance and its
pending evictions. Every one must be rejected by the checker.
"""

from __future__ import annotations

import copy

from reservoir_checker.verify import CheckerError, verify_chain

Mutant = tuple[list[dict], str]


def build_witnessed_chain(seed: int = 3) -> list[dict]:
    from reservoir.attest import AttestationLog
    from reservoir.rollout import Rollout
    from reservoir.rollout_buffer import RolloutBuffer

    buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=2, seed=seed, attest=AttestationLog())
    for v in range(6):
        buf.add_group(f"g{v}", v, [Rollout(tokens=[v + 1, 2], logprobs=[-0.1, -0.2], reward=r)
                                   for r in (1.0, 0.0, 0.5)], source="s")
        batch = buf.sample(2, current_version=v)
        buf.witness_batch(batch, step=v, batch_rows=8, rows=[6, 7], tensor_digest="ab" * 32)
    records = [dict(r) for r in buf.attestation_log.records]
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


def _edited(records: list[dict], idx: int, edit) -> list[dict]:
    m = copy.deepcopy(records)
    edit(m[idx])
    return _rechain(m, idx)


def batch_mutants(base: list[dict]) -> list[Mutant]:
    idxs = [i for i, r in enumerate(base) if r["op"] == "batch"]
    # The swap mutant needs a witness whose two draws selected different examples.
    distinct = [i for i in idxs if len({e["content_digest"] for e in base[i]["replaced"]}) == 2]
    if not distinct:
        raise RuntimeError("no witness in the baseline drew two different examples; change the seed")
    i, j = distinct[0], idxs[-1]

    def swap_draws(r):
        r["replaced"][0]["draw"], r["replaced"][1]["draw"] = 1, 0

    def dup_draw(r):
        r["replaced"][1]["draw"] = 0
        r["replaced"][1]["content_digest"] = r["replaced"][0]["content_digest"]

    mutants: list[Mutant] = [
        (_edited(base, i, swap_draws), "witness_rows_point_at_each_others_draws"),
        (_edited(base, i, lambda r: r["replaced"][0].update(content_digest="cd" * 32)), "witness_digest_changed"),
        (_edited(base, i, dup_draw), "witness_draw_placed_twice"),
        (_edited(base, i, lambda r: r["replaced"][1].update(row=6)), "witness_row_replaced_twice"),
        (_edited(base, i, lambda r: r["replaced"][1].update(row=8)), "witness_row_outside_batch"),
        (_edited(base, i, lambda r: r["replaced"][1].update(draw=7)), "witness_nonexistent_draw"),
        (_edited(base, i, lambda r: r["replaced"].pop()), "witness_missing_a_draw"),
        (_edited(base, j, lambda r: r.update(sample_op_counter=base[idxs[0]]["sample_op_counter"])),
         "witness_names_an_older_sample"),
        (_edited(base, j, lambda r: r.update(sample_op_counter=99)), "witness_names_a_nonexistent_sample"),
        (_edited(base, i, lambda r: r.update(tensor_digest="not-hex")), "witness_tensor_digest_malformed"),
        (_edited(base, i, lambda r: r.update(batch_rows=0)), "witness_batch_rows_zero"),
    ]
    dup = copy.deepcopy(base)
    dup.insert(i + 1, copy.deepcopy(base[i]))
    mutants.append((_rechain(dup, i + 1), "witness_duplicated"))
    adv = next(k for k, r in enumerate(base) if r["op"] == "advance_version" and base[k + 1]["op"] == "evict")
    moved = copy.deepcopy(base)
    moved.insert(adv + 1, copy.deepcopy(base[idxs[0]]))
    mutants.append((_rechain(moved, adv + 1), "witness_between_advance_and_pending_evicts"))
    stripped = copy.deepcopy(base)
    for r in stripped:
        r.pop("content_digest", None)
        r.pop("source", None)
    mutants.append((_rechain(stripped, 0), "witness_in_a_log_without_content_digests"))
    return mutants


def rejected(records: list[dict]) -> bool:
    try:
        verify_chain(records)
    except CheckerError:
        return True
    return False


def run_batch_category(results: dict) -> None:
    """Append the ``batch`` category to ``results``."""
    mutants = batch_mutants(build_witnessed_chain())
    count = 0
    for records, description in mutants:
        results["total_mutants"] += 1
        if rejected(records):
            results["rejected"] += 1
            count += 1
        else:
            results["survived"].append(description)
    results["details"].append(("batch", len(mutants), count))
