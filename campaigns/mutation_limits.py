"""
campaigns.mutation_limits — Chain-consistent forgeries the checker cannot see, measured.

The chain is unkeyed: whoever can rewrite the log can re-chain it, and
whoever can rewrite the manifest too can keep the two in step. The
forgeries below change only what the log records as a statement rather
than as a quantity the checker can recompute, so each one verifies:

- ``source`` swapped between two inserts in the log and the manifest
  together, which flips a per-source quota verdict;
- a drift decline hidden by recording a finite log-ratio (0) where the
  adapter measured a non-finite one (the log-ratios are the adapter's
  measurements), which also removes the drift evict and adds the row to
  the witness;
- an ``evict`` with reason ``"drift"`` of a slot no decision declined;
- in a log without the draw configuration, ``beta`` and with it every
  importance weight and effective sample size rewritten consistently;
- the content digests (and their manifest openings) of two inserts swapped,
  with the witness rows re-pointed, which moves exposure from one example
  to the other;
- every batch witness dropped from a log that carried them.

They are measured, not counted: each must verify. If one is ever rejected
the statement in the report would be false, so the campaign fails instead.
"""

from __future__ import annotations

import copy
from typing import Optional

from campaigns.mutation_reasons import rejection
from reservoir_checker.verify import verify_chain

Limit = tuple[list[dict], Optional[list[dict]], str]   # records, manifest (None: log only), description


def _rechain(records: list[dict], start: int) -> list[dict]:
    from reservoir.attest import _digest_record

    out = copy.deepcopy(records)
    prev = out[start - 1]["digest"] if start > 0 else "genesis"
    for rec in out[start:]:
        rec["prev_digest"] = prev
        rec["digest"] = _digest_record(rec)
        prev = rec["digest"]
    return out


def _line_of(manifest: list[dict], record: dict) -> int:
    return next(k for k, line in enumerate(manifest)
                if (line["op_counter"], line["index"]) == (record["op_counter"], record["index"]))


def source_swapped_with_manifest() -> Limit:
    from campaigns.mutation_content import build_content_chain

    records, manifest = build_content_chain()
    i = next(k for k, r in enumerate(records) if r.get("source") == "licensed")
    j = next(k for k, r in enumerate(records) if r.get("source") == "scraped")
    m, lines = copy.deepcopy(records), copy.deepcopy(manifest)
    m[i]["source"], m[j]["source"] = "scraped", "licensed"
    lines[_line_of(manifest, records[i])]["source"] = "scraped"
    lines[_line_of(manifest, records[j])]["source"] = "licensed"
    return _rechain(m, min(i, j)), lines, "source_swapped_in_log_and_manifest"


def decline_hidden_by_log_ratio() -> Limit:
    from campaigns.mutation_staleness import RATIOS, build_staleness_chain

    honest = build_staleness_chain()
    hidden = build_staleness_chain(last_step_ratios=[RATIOS[0], RATIOS[1], 0.0, RATIOS[3]])
    drifts = [sum(1 for r in log if r.get("reason") == "drift") for log in (honest, hidden)]
    if drifts[1] >= drifts[0]:
        raise RuntimeError("the recorded log-ratio hid no drift decline; the limit would measure nothing")
    return hidden, None, "drift_decline_hidden_by_a_recorded_log_ratio"


def _witnessed_buffer(beta: float = 0.4):
    from reservoir.attest import AttestationLog
    from reservoir.rollout import Rollout
    from reservoir.rollout_buffer import RolloutBuffer

    buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=3, seed=4, beta=beta, attest=AttestationLog())
    counts = dict(batch_rows=8, replaced_rows=2, declined_rows=0, dead_groups=1, near_dead_groups=0)
    for v in range(4):
        buf.add_group(f"g{v}", v, [Rollout(tokens=[v + 1, 2], logprobs=[-0.1, -0.2], reward=r)
                                   for r in (1.0, 0.0, 0.5)], source="s")
        batch = buf.sample(2, current_version=v)
        buf.witness_batch(batch, step=v, batch_rows=8, rows=[6, 7], tensor_digest="ab" * 32)
        buf.record_telemetry(v, counts, batch, {})
    return buf, batch


def stray_drift_evict() -> Limit:
    buf, batch = _witnessed_buffer()
    slot = sorted(set(buf.live_positions()) - set(batch.indices))[0]
    buf.evict(slot, "drift")
    return [dict(r) for r in buf.attestation_log.records], None, "drift_evict_of_a_slot_never_declined"


def _without_draw_config(records: list[dict]) -> list[dict]:
    m = copy.deepcopy(records)
    for name in ("seed", "buffer_id", "alpha", "beta"):
        m[0].pop(name, None)
    return _rechain(m, 0)


def beta_rewritten_without_draw_config() -> Limit:
    honest = _without_draw_config([dict(r) for r in _witnessed_buffer(beta=0.4)[0].attestation_log.records])
    forged = _without_draw_config([dict(r) for r in _witnessed_buffer(beta=1.0)[0].attestation_log.records])
    verify_chain(honest)
    if forged == honest:
        raise RuntimeError("a different beta changed no weight; the limit would measure nothing")
    return forged, None, "beta_and_every_weight_rewritten_without_draw_config"


def digests_swapped_with_manifest() -> Limit:
    from campaigns.mutation_provenance import build_provenance_run

    records, manifest = build_provenance_run()
    inserts = [k for k, r in enumerate(records) if r["op"] == "insert"]
    i, j = inserts[0], inserts[3]
    a, b = records[i]["content_digest"], records[j]["content_digest"]
    swap = {a: b, b: a}
    m, lines = copy.deepcopy(records), copy.deepcopy(manifest)
    m[i]["content_digest"], m[j]["content_digest"] = b, a
    for rec in m:
        for row in rec.get("replaced", []) if rec["op"] == "batch" else []:
            row["content_digest"] = swap.get(row["content_digest"], row["content_digest"])
    li, lj = _line_of(manifest, records[i]), _line_of(manifest, records[j])
    for name in ("content_digest", "prompt_id", "tokens", "reward_hex", "rewards"):
        lines[li][name], lines[lj][name] = manifest[lj][name], manifest[li][name]
    return _rechain(m, i), lines, "content_digests_swapped_with_openings_and_witness_rows"


def witnesses_dropped() -> Limit:
    records = [dict(r) for r in _witnessed_buffer()[0].attestation_log.records]
    first = next(k for k, r in enumerate(records) if r["op"] == "batch")
    kept = [r for r in records if r["op"] != "batch"]
    return _rechain(kept, first), None, "every_batch_witness_dropped"


def limit_mutants() -> list[Limit]:
    return [source_swapped_with_manifest(), decline_hidden_by_log_ratio(), stray_drift_evict(),
            beta_rewritten_without_draw_config(), digests_swapped_with_manifest(), witnesses_dropped()]


def run_consistency_limits(results: dict) -> None:
    """Record ``results["consistency_limit"]``; a limit mutant that is rejected fails the campaign."""
    limits = limit_mutants()
    passing = [name for records, manifest, name in limits
               if rejection(lambda r=records, m=manifest: verify_chain(r, manifest=m)) is None]
    if len(passing) != len(limits):
        results["survived"].append("consistency_limit_measurement_does_not_match_its_statement")
    results["consistency_limit"] = {
        "mutants": [name for _, _, name in limits],
        "pass": passing,
        "statement": "a re-chained log (with its manifest edited in step) that changes only what the log records "
                     "as a statement, such as sources, log-ratios, drift evictions, beta without the draw "
                     "configuration, which digest an insert committed to, or whether witnesses were written, "
                     "verifies: the chain is unkeyed and these are not quantities the checker can recompute",
    }
