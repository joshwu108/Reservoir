"""
campaigns.mutation_telemetry — Forgeries of telemetry records.

A ``telemetry`` record carries counters and, for a replayed step, the exact
effective sample size and staleness of the sampled rows, which the checker
recomputes from the sample record and the replayed state. These mutants
re-chain the log and change one claim: the ESS, the staleness, a counter
that no longer matches the draws, a reported float that is not listed, a
telemetry record for the wrong sample or in a format-1 log. Every one must
be rejected.
"""

from __future__ import annotations

import copy

from reservoir_checker.verify import CheckerError, verify_chain

Mutant = tuple[list[dict], str]
COUNTS = dict(batch_rows=8, replaced_rows=3, declined_rows=0, dead_groups=1, near_dead_groups=0)


def build_telemetry_chain(seed: int = 4) -> list[dict]:
    from reservoir.attest import AttestationLog
    from reservoir.rollout import Rollout
    from reservoir.rollout_buffer import RolloutBuffer

    buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=3, seed=seed, attest=AttestationLog())
    for v in range(5):
        buf.add_group(f"g{v}", v, [Rollout(tokens=[v + 1, 2], logprobs=[-0.1, -0.2], reward=r)
                                   for r in (1.0, 0.0, 0.5)], source="s")
        batch = buf.sample(3, current_version=v)
        buf.witness_batch(batch, step=v, batch_rows=8, rows=[5, 6, 7], tensor_digest="ab" * 32)
        buf.record_telemetry(v, COUNTS, batch, {"log_ratio_mean_abs": 0.25})
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


def telemetry_mutants(base: list[dict]) -> list[Mutant]:
    idxs = [i for i, r in enumerate(base) if r["op"] == "telemetry"]
    i, j = idxs[0], idxs[-1]
    mutants: list[Mutant] = [
        (_edited(base, i, lambda r: r.update(ess_num=str(int(r["ess_num"]) + 1))), "telemetry_ess_changed"),
        (_edited(base, i, lambda r: r.update(ess_den=str(int(r["ess_den"]) * 2))), "telemetry_ess_denominator_changed"),
        (_edited(base, j, lambda r: r.update(staleness_max=str(int(r["staleness_max"]) + 1))), "telemetry_staleness_max_changed"),
        (_edited(base, j, lambda r: r.update(staleness_sum=str(int(r["staleness_sum"]) + 1))), "telemetry_staleness_sum_changed"),
        (_edited(base, i, lambda r: r.update(replaced_rows=2)), "telemetry_replaced_rows_disagree_with_draws"),
        (_edited(base, i, lambda r: r.update(declined_rows=1)), "telemetry_declined_rows_disagree_with_draws"),
        (_edited(base, i, lambda r: r.update(replaced_rows=2, declined_rows=1)), "telemetry_counters_disagree_with_witness"),
        (_edited(base, i, lambda r: r.update(batch_rows=9)), "telemetry_batch_rows_disagree_with_witness"),
        (_edited(base, i, lambda r: r.update(step=str(int(r["step"]) + 1))), "telemetry_step_disagrees_with_witness"),
        (_edited(base, i, lambda r: r.update(reported=["log_ratio_mean_abs", "log_ratio_mean_abs"])), "telemetry_reported_duplicated"),
        (_edited(base, i, lambda r: r.update(batch_rows=1)), "telemetry_batch_rows_too_small"),
        (_edited(base, j, lambda r: r.update(sample_op_counter=base[i]["sample_op_counter"])), "telemetry_names_an_older_sample"),
        (_edited(base, i, lambda r: r.update(extra=(1.0).hex())), "telemetry_unlisted_reported_field"),
        (_edited(base, i, lambda r: r.update(log_ratio_mean_abs="0x1p-2")), "telemetry_reported_not_canonical"),
        (_edited(base, i, lambda r: r.update(log_ratio_mean_abs=float("nan").hex())), "telemetry_reported_nan"),
        (_edited(base, i, lambda r: r.update(reported=[])), "telemetry_reported_list_dropped"),
        (_edited(base, 0, lambda r: r.update(format="1")), "telemetry_in_format_1_log"),
    ]
    stripped = copy.deepcopy(base)
    for r in stripped:
        r.pop("content_digest", None)
        r.pop("source", None)
    mutants.append((_rechain(stripped, 0), "telemetry_in_a_log_without_content_digests"))
    return mutants


def rejected(records: list[dict]) -> bool:
    try:
        verify_chain(records)
    except CheckerError:
        return True
    return False


def run_telemetry_category(results: dict) -> None:
    """Append the ``telemetry`` category to ``results``."""
    mutants = telemetry_mutants(build_telemetry_chain())
    count = 0
    for records, description in mutants:
        results["total_mutants"] += 1
        if rejected(records):
            results["rejected"] += 1
            count += 1
        else:
            results["survived"].append(description)
    results["details"].append(("telemetry", len(mutants), count))
