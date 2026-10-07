"""
campaigns.mutation_staleness — Forgeries of staleness decisions in telemetry records.

A telemetry record written under a ``StalenessPolicy`` carries the policy,
every draw's log-ratio as a hex float and one decision per draw. The
checker replays the policy from the recorded inputs, the sample's exact
weights and the live slots' ages, and cross-checks the decisions against
the batch witness and the counters. These mutants re-chain the log and
change one claim: a decision's reason or scale, a log-ratio, a policy
parameter, a group label, a row, the reported statistics, or the presence
of the fields themselves. Every one must be rejected.

One limit is measured rather than counted: a policy parameter changed
consistently on every record of the log so that no recorded decision
differs (a decline cap that never bound, an ESS floor looser than one that
never declined) passes, because the checker verifies that the decisions
follow the declared policy, not which policy the adapter intended. The
report states it (``staleness_limit``) and the campaign fails if the
measurement changes. A parameter changed on one record only is rejected,
since the policy must be the same on every record of a log.
"""

from __future__ import annotations

import copy
import math

from reservoir_checker.verify import CheckerError, verify_chain

Mutant = tuple[list[dict], str]
GROUP_SIZE = 2
RATIOS = [0.0, math.log(8.0), float("nan"), 3.0]


def build_staleness_chain(seed: int = 3) -> list[dict]:
    from reservoir.attest import AttestationLog
    from reservoir.integrations._trl_staleness import StalenessPolicy, decide, evictions_for
    from reservoir.rollout import Rollout
    from reservoir.rollout_buffer import RolloutBuffer

    policy = StalenessPolicy(max_age=1, ess_floor=0.1, mass_cap=0.25)
    buf = RolloutBuffer(capacity=16, half_life=2, max_policy_age=8, seed=seed, attest=AttestationLog())
    for step in range(3):
        buf.add_group(f"g{step}", step, [Rollout(tokens=[step + 1, i + 2], logprobs=[-0.1, -0.2], reward=r)
                                         for i, r in enumerate((1.0, 0.8, 0.5))], source="s")
        if step > buf.current_version:
            buf.advance(step)
        batch = buf.sample(len(RATIOS), current_version=step)
        ages = [buf.current_version - v for v in batch.model_versions]
        rows = list(range(4, 4 + len(RATIOS)))
        decisions = decide(policy, ratios=RATIOS, is_weights=batch.is_weights, ages=ages, rows=rows,
                           groups=[r // GROUP_SIZE for r in rows])
        kept = [d for d in decisions if d.kept]
        buf.witness_batch(batch, step=step, batch_rows=8, rows=[d.row for d in kept], tensor_digest="ab" * 32,
                          declined=[d.draw for d in decisions if not d.kept])
        finite = [abs(r) for r in RATIOS if math.isfinite(r)]
        buf.record_telemetry(
            step, dict(batch_rows=8, replaced_rows=len(kept), declined_rows=len(decisions) - len(kept), dead_groups=2,
                       near_dead_groups=0),
            batch, {"log_ratio_mean_abs": sum(finite) / len(finite), "log_ratio_max_abs": max(finite)},
            log_ratios=RATIOS, policy=policy.to_record(GROUP_SIZE), decisions=[d.to_record() for d in decisions],
        )
        for slot, reason in evictions_for(batch.indices, decisions):
            buf.evict(slot, reason)
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


def staleness_mutants(base: list[dict]) -> list[Mutant]:
    idxs = [i for i, r in enumerate(base) if r["op"] == "telemetry"]
    i = idxs[0]
    rec = base[i]
    # Parameter forgeries need a record where the change flips a decision: step 1 has rows of age 1 (an age
    # bound of 0 declines them) and kept draws with unequal weights (a floor of 1 declines one); step 2 has an
    # age decline that a looser bound would keep.
    j, last = idxs[1], idxs[-1]
    assert any(d["reason"] == "age" for d in base[last]["decisions"]), "the baseline must decline a row by age"
    kept = [k for k, d in enumerate(rec["decisions"]) if d["reason"] is None]
    declined = [k for k, d in enumerate(rec["decisions"]) if d["reason"] is not None]
    scaled = [k for k, d in enumerate(rec["decisions"]) if d["scale_den"] != "1"]
    same_group = [k for k in kept if rec["decisions"][k]["group"] == rec["decisions"][kept[0]]["group"]]
    assert kept and declined and scaled and len(same_group) >= 2, "the baseline must exercise every stage"
    age_evict = next(k for k, r in enumerate(base) if r["op"] == "evict" and r.get("reason") == "age")

    def decision(k, **changes):
        return lambda r: r["decisions"][k].update(changes)

    def swap_rows(r):
        a, b = same_group[0], same_group[1]
        r["decisions"][a]["row"], r["decisions"][b]["row"] = r["decisions"][b]["row"], r["decisions"][a]["row"]

    def double_scale(r):
        d = r["decisions"][scaled[0]]
        d["scale_num"], d["scale_den"] = str(int(d["scale_num"]) * 2), str(int(d["scale_den"]) * 2)

    def swap_order(r):
        r["decisions"][0], r["decisions"][1] = r["decisions"][1], r["decisions"][0]

    def inactive(r):
        r["policy"].update(max_age=None, ess_floor=None, mass_cap=None, max_log_ratio=None)

    mutants: list[Mutant] = [
        (_edited(base, i, decision(kept[0], reason="age", scale_num="1", scale_den="1")), "staleness_kept_row_marked_declined"),
        (_edited(base, i, decision(declined[0], reason=None)), "staleness_declined_row_marked_kept"),
        (_edited(base, i, decision(declined[0], reason="ess")), "staleness_decline_reason_changed"),
        (_edited(base, i, decision(scaled[0], scale_num="1", scale_den="1")), "staleness_scale_removed"),
        (_edited(base, i, double_scale), "staleness_scale_not_reduced"),
        (_edited(base, i, decision(scaled[0], scale_num="3", scale_den="2")), "staleness_scale_above_one"),
        (_edited(base, i, decision(kept[0], reason="drift")), "staleness_declined_row_keeps_a_scale"),
        (_edited(base, i, lambda r: r["log_ratios"].__setitem__(1, (0.5).hex())), "staleness_log_ratio_changed"),
        (_edited(base, i, lambda r: r["log_ratios"].__setitem__(0, "0x0p+0")), "staleness_log_ratio_not_canonical"),
        (_edited(base, i, lambda r: r["log_ratios"].pop()), "staleness_log_ratio_list_short"),
        (_edited(base, i, lambda r: r["policy"].update(mass_cap=(10.0).hex())), "staleness_policy_mass_cap_changed"),
        (_edited(base, j, lambda r: r["policy"].update(ess_floor=(1.0).hex())), "staleness_policy_ess_floor_raised"),
        (_edited(base, j, lambda r: r["policy"].update(max_age=0)), "staleness_policy_max_age_tightened"),
        (_edited(base, last, lambda r: r["policy"].update(max_age=100)), "staleness_policy_max_age_loosened_past_a_decline"),
        (_edited(base, i, lambda r: r["policy"].update(max_declines_per_step=0)), "staleness_policy_decline_cap_violated"),
        (_edited(base, i, lambda r: r["policy"].update(group_size=1)), "staleness_policy_group_size_changed"),
        (_edited(base, i, lambda r: r["policy"].update(extra=1)), "staleness_policy_extra_field"),
        (_edited(base, i, inactive), "staleness_policy_inactive_but_recorded"),
        (_edited(base, i, lambda r: (r.pop("policy"), r.pop("decisions"))), "staleness_policy_dropped_declines_kept"),
        (_edited(base, i, lambda r: r.pop("decisions")), "staleness_decisions_dropped"),
        (_edited(base, i, lambda r: r.pop("log_ratios")), "staleness_log_ratios_dropped"),
        (_edited(base, i, decision(kept[0], group=9)), "staleness_group_label_changed"),
        (_edited(base, i, swap_rows), "staleness_rows_swapped_against_witness"),
        (_edited(base, i, decision(kept[0], row=40, group=20)), "staleness_row_outside_batch"),
        (_edited(base, i, swap_order), "staleness_decisions_out_of_draw_order"),
        (_edited(base, i, lambda r: r.update(log_ratio_mean_abs=(0.5).hex())), "staleness_reported_mean_disagrees_with_ratios"),
        (_edited(base, i, lambda r: r.update(log_ratio_max_abs=(2.0).hex())), "staleness_reported_max_disagrees_with_ratios"),
        (_edited(base, i, lambda r: r.update(declined_rows=r["declined_rows"] + 1, replaced_rows=r["replaced_rows"] - 1)),
         "staleness_counters_disagree_with_decisions"),
        (_edited(base, age_evict, lambda r: r.update(reason="ess")), "staleness_evict_reason_ess_is_not_a_reason"),
        (_edited(base, j, lambda r: [r.pop(k) for k in ("log_ratios", "policy", "decisions")]), "staleness_fields_dropped_from_a_later_record"),
        (_edited(base, j, lambda r: (r.pop("policy"), r.pop("decisions"))), "staleness_policy_dropped_from_a_later_record"),
        (_edited(base, j, lambda r: r["policy"].update(max_age=100)), "staleness_policy_changed_between_records"),
        (_edited(base, i, decision(scaled[0], scale_num="\u00b2")), "staleness_scale_unicode_digit"),
        (_edited(base, i, lambda r: r["decisions"][scaled[0]].update(scale_den="0" + r["decisions"][scaled[0]]["scale_den"])),
         "staleness_scale_leading_zero"),
    ]
    return mutants


def _edited_everywhere(records: list[dict], edit) -> list[dict]:
    """``edit`` applied to every telemetry record, re-chained from the first."""
    m = copy.deepcopy(records)
    idxs = [k for k, r in enumerate(m) if r["op"] == "telemetry"]
    for k in idxs:
        edit(m[k])
    return _rechain(m, idxs[0])


def consistent_parameter_changes(base: list[dict]) -> list[Mutant]:
    """Policy changes on every record that flip no decision: measured, expected to pass (see the module docstring)."""
    return [
        (_edited_everywhere(base, lambda r: r["policy"].update(max_declines_per_step=100)),
         "staleness_policy_decline_cap_added_never_binding"),
        (_edited_everywhere(base, lambda r: r["policy"].update(ess_floor=(0.05).hex())),
         "staleness_policy_ess_floor_loosened_nothing_declined_for_ess"),
    ]


def rejected(records: list[dict]) -> bool:
    try:
        verify_chain(records)
    except CheckerError:
        return True
    return False


def run_staleness_category(results: dict) -> None:
    """Append the ``staleness`` category to ``results`` and measure ``staleness_limit``."""
    base = build_staleness_chain()
    mutants = staleness_mutants(base)
    count = 0
    for records, description in mutants:
        results["total_mutants"] += 1
        if rejected(records):
            results["rejected"] += 1
            count += 1
        else:
            results["survived"].append(description)
    results["details"].append(("staleness", len(mutants), count))
    consistent = consistent_parameter_changes(base)
    passing = [description for records, description in consistent if not rejected(records)]
    results["staleness_limit"] = {
        "mutants": [description for _, description in consistent],
        "pass": passing,
        "statement": (
            "a policy parameter changed so that no recorded decision differs passes: the checker verifies that the "
            "decisions follow the declared policy from the recorded inputs, not which policy the adapter intended"
        ),
    }
    if len(passing) != len(consistent):
        results["survived"].append(
            f"staleness_limit_changed: expected {len(consistent)} consistent parameter changes to pass, {len(passing)} did"
        )
