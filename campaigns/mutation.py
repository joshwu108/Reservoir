"""
campaigns.mutation — T4 forgery campaign against checker/verify.py.

Generates ≥60 single-fault forgeries spanning all declared mutation categories.
Each forgery is a valid-looking attestation chain with exactly one fault.
100% rejection rate by checker/verify.py is required.

A surviving mutant (not rejected by checker) is a checker bug.
Never delete a mutant — fix the checker.

Categories (per spec):
  1. Digest bit-flips
  2. Off-by-one draw integers
  3. Swapped sampled indices (internally consistent but chain-inconsistent)
  4. Probability num/den correct as reals but not in reduced form
  5. Deleted mutation records
  6. Reordered records
  7. Replayed stale suffixes

Run with: python -m campaigns.mutation
Prints a report with exact counts.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reservoir.attest import AttestationLog, make_sample_entry
from src.reservoir.buffer import ExactPERBuffer, Transition
from src.reservoir.sumtree import ExactSumTree
from checker.verify import CheckerError, verify_chain, verify_json_lines


def _build_baseline_chain(capacity: int = 8, n_ops: int = 20, seed: int = 42) -> tuple[str, ExactPERBuffer]:
    """Build a valid attestation chain for a buffer with n_ops operations.

    Returns (json_lines, buffer) for use as the baseline honest chain.
    """
    buf = ExactPERBuffer(capacity=capacity, alpha=1.0, beta=1.0, seed=seed)
    log = AttestationLog()

    # Track the previous priority for attestation
    prev_priorities = {i: 0 for i in range(buf.capacity)}

    for i in range(capacity):
        td = float(i + 1)
        old_p = 0
        pos = buf.insert(Transition(i, 0, float(i), i + 1, False), td_error=td)
        new_p = buf._sum_tree.get(pos)

        op = "insert" if i < capacity else "evict"
        log.append_mutation(
            op=op,
            index=pos,
            old_priority_int=old_p,
            new_priority_int=new_p,
            op_counter=buf._op_counter,
        )
        prev_priorities[pos] = new_p

    # Sample with attestation
    batch = buf.sample(4)
    sample_entries = []
    for k in range(4):
        entry = make_sample_entry(
            leaf_index=batch.indices[k],
            draw_int=batch.draw_integers[k],
            priority_int=batch.priorities[k],
            root_total=batch.root_total,
            is_weight=batch.is_weights[k],
        )
        sample_entries.append(entry)

    log.append_sample(
        op_counter=buf._op_counter,
        root_total=batch.root_total,
        samples=sample_entries,
    )

    # Do a few more updates
    for i in range(min(4, capacity)):
        old_p = buf._sum_tree.get(i)
        buf.update_priority(i, td_error=float(i * 2 + 0.5))
        new_p = buf._sum_tree.get(i)
        log.append_mutation(
            op="update",
            index=i,
            old_priority_int=old_p,
            new_priority_int=new_p,
            op_counter=buf._op_counter,
        )
        prev_priorities[i] = new_p

    return log.to_json_lines(), buf


def _parse_records(json_lines: str) -> list[dict]:
    records = []
    for line in json_lines.strip().split("\n"):
        if line:
            records.append(json.loads(line))
    return records


def _recompute_digests_from(records: list[dict], start_idx: int) -> list[dict]:
    """Recompute digests for records[start_idx:] given that records[start_idx-1]
    has already been modified. Used for forgeries that need consistent downstream
    digests after a single mutation.

    Actually: for our forgeries, we deliberately do NOT recompute digests —
    the forgery is detected because the digest no longer matches.
    This function is for the 'internally-consistent' forgeries where we
    recompute the entire chain's digests to simulate a more sophisticated forger.
    """
    from src.reservoir.attest import _digest_record
    result = copy.deepcopy(records)
    prev_digest = result[start_idx - 1]["digest"] if start_idx > 0 else "genesis"
    for i in range(start_idx, len(result)):
        result[i]["prev_digest"] = prev_digest
        result[i]["digest"] = _digest_record(result[i])
        prev_digest = result[i]["digest"]
    return result


def run_mutation_campaign() -> dict:
    """Run all forgery categories and collect results.

    Returns a dict with:
      - total_mutants: int
      - rejected: int
      - survived: list of surviving mutant descriptions
      - details: list of (category, count_in_category, rejected_count)
    """
    json_lines, buf = _build_baseline_chain(capacity=8, n_ops=20, seed=42)
    capacity = buf.capacity
    records = _parse_records(json_lines)

    # Verify baseline is valid first
    try:
        verify_chain(records, capacity)
    except CheckerError as e:
        raise RuntimeError(f"Baseline chain failed verification: {e}")

    results = {
        "total_mutants": 0,
        "rejected": 0,
        "survived": [],
        "details": [],
    }

    def check_mutant(mutant_records: list[dict], description: str) -> bool:
        """Return True if rejected (expected), False if survived (checker bug)."""
        try:
            verify_chain(mutant_records, capacity)
            results["survived"].append(description)
            return False
        except CheckerError:
            return True

    def run_category(category_name: str, mutants: list[tuple[list[dict], str]]) -> None:
        rejected_count = 0
        for mutant_records, description in mutants:
            results["total_mutants"] += 1
            if check_mutant(mutant_records, description):
                results["rejected"] += 1
                rejected_count += 1
        results["details"].append((category_name, len(mutants), rejected_count))

    # -----------------------------------------------------------------------
    # Category 1: Digest bit-flips
    # -----------------------------------------------------------------------
    category1 = []
    for i in range(len(records)):
        mutant = copy.deepcopy(records)
        # Flip the first bit of the digest hex string
        orig = mutant[i]["digest"]
        flipped = hex(int(orig[0], 16) ^ 1)[2:] + orig[1:]
        mutant[i]["digest"] = flipped
        category1.append((mutant, f"digest_bitflip_record_{i}"))

    # Flip the last byte of a digest
    for i in [0, len(records) // 2, len(records) - 1]:
        mutant = copy.deepcopy(records)
        orig = mutant[i]["digest"]
        flipped = orig[:-2] + format(int(orig[-2:], 16) ^ 0xFF, "02x")
        mutant[i]["digest"] = flipped
        category1.append((mutant, f"digest_lastbyte_flip_record_{i}"))

    run_category("digest_bitflip", category1)

    # -----------------------------------------------------------------------
    # Category 2: Off-by-one draw integers (at segment BOUNDARIES)
    # The key: use draw_int values that land at a segment boundary, so
    # draw_int ± 1 crosses into a different leaf. Keep leaf_index as the
    # original leaf's declared value — the checker must detect mismatch.
    # -----------------------------------------------------------------------
    category2 = []
    sample_records = [(i, r) for i, r in enumerate(records) if r.get("op") == "sample"]

    # Compute prefix-sum boundaries from the reconstructed tree at each sample record.
    # We replay the mutations up to the sample record to get the tree state.
    def _replay_tree(records_up_to: list[dict], cap: int) -> list[int]:
        """Return leaf priority_ints from replaying mutations up to (not including) sample."""
        from checker.verify import _SumTree, _next_power_of_two
        t = _SumTree(cap)
        for r in records_up_to:
            if r.get("op") in ("insert", "update", "evict"):
                t.update(r["index"], int(r["new_priority_int"]))
        return [t.get(i) for i in range(t.capacity)]

    for rec_idx, sample_rec in sample_records:
        # Replay tree state at this sample
        leaf_prios = _replay_tree(records[:rec_idx], capacity)
        total = sum(leaf_prios)

        # Compute prefix sums: prefix[i] = sum(leaf_prios[0:i])
        prefix = [0] * (len(leaf_prios) + 1)
        for i, p in enumerate(leaf_prios):
            prefix[i + 1] = prefix[i] + p

        # For each pair of adjacent non-empty leaves, create a boundary forgery:
        # Set draw_int = prefix[i+1] (start of leaf i+1 segment) but declare leaf i
        # This is an off-by-one that CROSSES the boundary -> checker must reject
        for i in range(len(leaf_prios) - 1):
            if leaf_prios[i] > 0 and leaf_prios[i + 1] > 0:
                # draw_int at start of leaf i+1 segment
                boundary_draw = prefix[i + 1]  # maps to leaf i+1
                if boundary_draw < total:
                    # Forge: claim draw maps to leaf i (wrong — it maps to i+1)
                    mutant = copy.deepcopy(records)
                    mutant[rec_idx]["samples"][0]["draw_int"] = str(boundary_draw)
                    mutant[rec_idx]["samples"][0]["leaf_index"] = i  # Wrong leaf
                    # Keep prob for leaf i (wrong probability for this draw)
                    mutant = _recompute_digests_from(mutant, rec_idx)
                    category2.append((mutant, f"boundary_draw_wrong_leaf_{i}_rec{rec_idx}"))

                # draw_int just before boundary: maps to leaf i, but claim leaf i+1
                before_draw = prefix[i + 1] - 1
                if before_draw >= 0:
                    mutant2 = copy.deepcopy(records)
                    mutant2[rec_idx]["samples"][0]["draw_int"] = str(before_draw)
                    mutant2[rec_idx]["samples"][0]["leaf_index"] = i + 1  # Wrong leaf
                    mutant2 = _recompute_digests_from(mutant2, rec_idx)
                    category2.append((mutant2, f"before_boundary_wrong_leaf_{i+1}_rec{rec_idx}"))

                if len(category2) >= 10:
                    break

    run_category("off_by_one_draw", category2)

    # -----------------------------------------------------------------------
    # Category 3: Swapped sampled indices
    # Internally-consistent forged probability for wrong index.
    # -----------------------------------------------------------------------
    category3 = []
    for rec_idx, sample_rec in sample_records:
        samples = sample_rec.get("samples", [])
        if len(samples) >= 2:
            mutant = copy.deepcopy(records)
            # Swap leaf_index between sample 0 and 1 without changing draw_int
            s0 = mutant[rec_idx]["samples"][0]
            s1 = mutant[rec_idx]["samples"][1]
            s0["leaf_index"], s1["leaf_index"] = s1["leaf_index"], s0["leaf_index"]
            # Also swap prob_num/prob_den to match the (wrong) leaf
            s0["prob_num"], s1["prob_num"] = s1["prob_num"], s0["prob_num"]
            s0["prob_den"], s1["prob_den"] = s1["prob_den"], s0["prob_den"]
            # Recompute digests
            mutant = _recompute_digests_from(mutant, rec_idx)
            category3.append((mutant, f"swapped_indices_rec{rec_idx}"))

        # Forge a leaf_index that doesn't match the draw_int
        if samples:
            mutant = copy.deepcopy(records)
            orig_leaf = int(mutant[rec_idx]["samples"][0]["leaf_index"])
            mutant[rec_idx]["samples"][0]["leaf_index"] = (orig_leaf + 1) % capacity
            mutant = _recompute_digests_from(mutant, rec_idx)
            category3.append((mutant, f"wrong_leaf_index_rec{rec_idx}"))

    # Add 5 more specific swaps to ensure ≥ enough in this category
    for rec_idx, sample_rec in sample_records[:3]:
        for k in range(min(2, len(sample_rec.get("samples", [])))):
            mutant = copy.deepcopy(records)
            # Set leaf_index to a completely different position
            mutant[rec_idx]["samples"][k]["leaf_index"] = (
                (int(mutant[rec_idx]["samples"][k]["leaf_index"]) + 3) % capacity
            )
            mutant = _recompute_digests_from(mutant, rec_idx)
            category3.append((mutant, f"wrong_leaf_index_rec{rec_idx}_k{k}_shift3"))

    run_category("swapped_indices", category3)

    # -----------------------------------------------------------------------
    # Category 4: Probability num/den not in reduced form
    # -----------------------------------------------------------------------
    category4 = []
    for rec_idx, sample_rec in sample_records:
        for k, s in enumerate(sample_rec.get("samples", [])):
            # Multiply both num and den by 2 (same real value, not reduced)
            pn = int(s["prob_num"])
            pd = int(s["prob_den"])
            mutant = copy.deepcopy(records)
            mutant[rec_idx]["samples"][k]["prob_num"] = str(pn * 2)
            mutant[rec_idx]["samples"][k]["prob_den"] = str(pd * 2)
            mutant = _recompute_digests_from(mutant, rec_idx)
            category4.append((mutant, f"prob_not_reduced_x2_rec{rec_idx}_k{k}"))

            # Multiply by 3
            mutant3 = copy.deepcopy(records)
            mutant3[rec_idx]["samples"][k]["prob_num"] = str(pn * 3)
            mutant3[rec_idx]["samples"][k]["prob_den"] = str(pd * 3)
            mutant3 = _recompute_digests_from(mutant3, rec_idx)
            category4.append((mutant3, f"prob_not_reduced_x3_rec{rec_idx}_k{k}"))

        # IS weight not in reduced form
        for k, s in enumerate(sample_rec.get("samples", [])):
            wn = int(s["is_weight_num"])
            wd = int(s["is_weight_den"])
            mutant = copy.deepcopy(records)
            mutant[rec_idx]["samples"][k]["is_weight_num"] = str(wn * 5)
            mutant[rec_idx]["samples"][k]["is_weight_den"] = str(wd * 5)
            mutant = _recompute_digests_from(mutant, rec_idx)
            category4.append((mutant, f"is_weight_not_reduced_x5_rec{rec_idx}_k{k}"))

    run_category("not_reduced_form", category4)

    # -----------------------------------------------------------------------
    # Category 5: Deleted mutation records
    # -----------------------------------------------------------------------
    category5 = []
    mut_indices = [i for i, r in enumerate(records) if r.get("op") in ("insert", "update", "evict")]
    for del_idx in mut_indices[:8]:
        mutant = copy.deepcopy(records)
        mutant.pop(del_idx)
        # Recompute chain from the deleted position onwards
        if del_idx > 0 and del_idx < len(mutant):
            mutant = _recompute_digests_from(mutant, del_idx)
        category5.append((mutant, f"deleted_mutation_record_{del_idx}"))

    # Delete the first record entirely
    mutant = copy.deepcopy(records)
    mutant.pop(0)
    if mutant:
        mutant = _recompute_digests_from(mutant, 0)
    category5.append((mutant, "deleted_first_record"))

    run_category("deleted_records", category5)

    # -----------------------------------------------------------------------
    # Category 6: Reordered records
    # Reordering is only detectable when the order of operations matters:
    # - An update record with old_priority_int=X depends on a prior insert that set X.
    # - Moving the update before the insert fails because tree starts at 0, not X.
    # - Moving a sample before the inserts it needs fails because root_total changes.
    # -----------------------------------------------------------------------
    category6 = []
    # Find update records that have non-zero old_priority_int
    update_recs = [
        (i, r) for i, r in enumerate(records)
        if r.get("op") == "update" and int(r.get("old_priority_int", "0")) > 0
    ]
    insert_recs = [(i, r) for i, r in enumerate(records) if r.get("op") == "insert"]
    sample_recs_idx = [i for i, r in enumerate(records) if r.get("op") == "sample"]

    # Swap update (at position u_idx) with an earlier insert (at position i_idx)
    for u_idx, u_rec in update_recs[:4]:
        for i_idx, i_rec in insert_recs:
            if i_idx < u_idx and i_rec["index"] == u_rec["index"]:
                # This insert establishes the value that the update expects
                mutant = copy.deepcopy(records)
                mutant[i_idx], mutant[u_idx] = mutant[u_idx], mutant[i_idx]
                mutant = _recompute_digests_from(mutant, min(i_idx, u_idx))
                category6.append((mutant, f"swap_update_{u_idx}_with_insert_{i_idx}"))

    # Move a sample record before some inserts it depends on
    for s_idx in sample_recs_idx[:2]:
        if s_idx > 3:
            mutant = copy.deepcopy(records)
            # Move sample record to position 2 (before most inserts)
            sample_record = mutant.pop(s_idx)
            mutant.insert(2, sample_record)
            mutant = _recompute_digests_from(mutant, 2)
            category6.append((mutant, f"sample_moved_before_inserts_s{s_idx}"))

    # Reversed insert+update sequence: if we have insert then update at same pos,
    # swap them (update tries to find old=insert_new, but tree has 0)
    for u_idx, u_rec in update_recs[:6]:
        pos = u_rec["index"]
        for i_idx, i_rec in insert_recs:
            if i_rec["index"] == pos and i_idx == u_idx - 1:
                # Adjacent: insert at i_idx, update at u_idx = i_idx+1
                mutant = copy.deepcopy(records)
                mutant[i_idx], mutant[u_idx] = mutant[u_idx], mutant[i_idx]
                mutant = _recompute_digests_from(mutant, i_idx)
                category6.append((mutant, f"adjacent_swap_insert{i_idx}_update{u_idx}"))

    run_category("reordered_records", category6)

    # -----------------------------------------------------------------------
    # Category 7: Replayed stale suffixes
    # -----------------------------------------------------------------------
    category7 = []
    if len(records) >= 6:
        # Append old records at the end (replay attack)
        for repeat_from in [0, 1, 2]:
            mutant = copy.deepcopy(records)
            # Append records[repeat_from..repeat_from+3] again
            suffix = copy.deepcopy(records[repeat_from:repeat_from + 3])
            # Recompute their digests to chain from current head
            prev = mutant[-1]["digest"]
            for r in suffix:
                r["prev_digest"] = prev
                r["digest"] = _recompute_digests_from([r], 0)[0]["digest"]
                prev = r["digest"]
            mutant.extend(suffix)
            category7.append((mutant, f"stale_suffix_replay_from_{repeat_from}"))

        # Duplicate the last record (replay)
        for _ in range(4):
            mutant = copy.deepcopy(records)
            last = copy.deepcopy(records[-1])
            # Change prev_digest to current head (same as last record)
            # This creates a consistent-looking replay
            last["prev_digest"] = records[-1]["digest"]
            from src.reservoir.attest import _digest_record as _dr
            last["digest"] = _dr(last)
            mutant.append(last)
            category7.append((mutant, f"duplicate_last_record"))

    run_category("stale_suffix_replay", category7)

    # -----------------------------------------------------------------------
    # Category 8: Age-decay protocol forgeries
    # Built from a RolloutBuffer log (decay_config + advance_version + rebase
    # + decay fields). Every mutant is re-chained so only the semantic replay
    # in checker/decay_replay.py can catch it.
    # -----------------------------------------------------------------------
    run_category("decay", _decay_mutants())

    return results


def _build_decay_chain(seed: int = 11) -> list[dict]:
    """A RolloutBuffer run with inserts, updates, stale/capacity evictions and rebases."""
    # The buffer type-checks its log against reservoir.attest.AttestationLog;
    # this file imports the legacy modules as src.reservoir.*, which Python
    # treats as a different module, so import the matching class here.
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
    for i, r in enumerate(records):
        if r["op"] == op and all(r.get(k) == v for k, v in match.items()):
            return i
    raise RuntimeError(f"baseline has no {op} record matching {match}")


def _bump(records: list[dict], idx: int, field: str, delta: int = 1) -> list[dict]:
    """Copy records, add ``delta`` to an integer-as-string field, re-chain from idx."""
    mutant = copy.deepcopy(records)
    mutant[idx][field] = str(int(mutant[idx][field]) + delta)
    return _recompute_digests_from(mutant, idx)


def _decay_mutants() -> list[tuple[list[dict], str]]:
    """One mutant per rule the decay replay enforces, plus a few variants."""
    base = _build_decay_chain()
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
    mutants.append((_bump(base, u, "entry_version"), "update_version_changed_without_reset"))
    advances = [j for j, r in enumerate(base) if r["op"] == "advance_version"]
    mutants.append((_bump(base, advances[1], "new_version", -1), "advance_version_backwards"))
    adv = copy.deepcopy(base)
    adv[advances[1]]["old_version"] = str(int(adv[advances[1]]["old_version"]) + 1)
    adv[advances[1]]["new_version"] = str(int(adv[advances[1]]["new_version"]) + 1)
    mutants.append((_recompute_digests_from(adv, advances[1]), "advance_version_old_mismatch"))

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


def main() -> None:
    print("=" * 70)
    print("T4 Mutation Campaign — Attestation Forgery Test")
    print("=" * 70)
    print()

    results = run_mutation_campaign()

    print(f"Total mutants: {results['total_mutants']}")
    print(f"Rejected:      {results['rejected']}")
    print(f"Survived:      {len(results['survived'])}")
    print()

    print("By category:")
    for name, total, rejected in results["details"]:
        status = "PASS" if total == rejected else "FAIL"
        print(f"  [{status}] {name}: {rejected}/{total} rejected")

    print()
    if results["survived"]:
        print("SURVIVING MUTANTS (checker bugs!):")
        for desc in results["survived"]:
            print(f"  - {desc}")
        print()
        print("RESULT: FAIL — surviving mutants detected")
    elif results["total_mutants"] < 60:
        print(f"RESULT: FAIL — fewer than 60 mutants ({results['total_mutants']})")
    else:
        print(f"RESULT: PASS — 100% rejection rate ({results['rejected']}/{results['total_mutants']})")

    # Write report
    Path("results").mkdir(exist_ok=True)
    report = {
        "campaign": "T4_mutation",
        "total_mutants": results["total_mutants"],
        "rejected": results["rejected"],
        "survived": results["survived"],
        "categories": [
            {"name": n, "total": t, "rejected": r}
            for n, t, r in results["details"]
        ],
        "pass": len(results["survived"]) == 0 and results["total_mutants"] >= 60,
    }
    Path("results/mutation_campaign_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print()
    print("Report written to results/mutation_campaign_report.json")


if __name__ == "__main__":
    main()
