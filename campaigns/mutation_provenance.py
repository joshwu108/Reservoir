"""
campaigns.mutation_provenance — Forgeries of quarantine records and reward provenance.

An ``evict`` with reason ``"quarantine"`` must carry the predicate text and
the operator's note, nothing else may carry them, and a log of a format
that predates quarantine cannot contain one. A manifest line may carry
``rewards``, per-reward-function values that are numeric only. These
mutants re-chain the log (or edit the manifest) and change one claim; every
one must be rejected.

The documented limit: neither text, nor the manifest's reward values, is
something the log commits to. A chain-consistent change to the predicate
text or the note, a changed reward value, or a dropped ``rewards`` field all
pass the checker. They are measured here and must survive; the campaign
fails if that measurement changes, so the report cannot claim what was not
measured.
"""

from __future__ import annotations

import copy
from typing import Optional

from reservoir_checker.verify import CheckerError, verify_chain

Mutant = tuple[list[dict], Optional[list[dict]], str]   # records, manifest (None: log only), description


def build_provenance_run(seed: int = 5) -> tuple[list[dict], list[dict]]:
    """Inserts with per-reward-function values, witnessed samples, a capacity evict, one quarantine."""
    from reservoir.attest import AttestationLog
    from reservoir.rollout import Rollout
    from reservoir.rollout_buffer import RolloutBuffer
    from reservoir.rollout_manifest import ManifestWriter

    manifest = ManifestWriter()
    buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=3, seed=seed, attest=AttestationLog(),
                        manifest=manifest)
    for v in range(4):
        buf.add_group(f"g{v}", v, [
            Rollout(tokens=[v + 1, 2], logprobs=[-0.1, -0.2], reward=r,
                    metadata={"rewards": {"verifier": r, "judge": 0.5}})
            for r in (1.0, 0.0, 0.5)
        ], source="s")
        batch = buf.sample(2, current_version=v)
        buf.witness_batch(batch, step=v, batch_rows=8, rows=[6, 7], tensor_digest="ab" * 32)
    buf.quarantine(lambda r, g: g.prompt_id == "g3", "judge rewarded empty answers", predicate_text="prompt == g3")
    records = [dict(r) for r in buf.attestation_log.records]
    lines = manifest.records
    verify_chain(records, manifest=lines)
    if not any(r.get("reason") == "capacity" for r in records):
        raise RuntimeError("the baseline has no capacity evict; change the seed or the sizes")
    return records, lines


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


def _manifest_edited(manifest: list[dict], idx: int, edit) -> list[dict]:
    m = copy.deepcopy(manifest)
    edit(m[idx])
    return m


def log_mutants(base: list[dict]) -> list[Mutant]:
    """Quarantine-record forgeries; checked from the log alone."""
    quarantines = [i for i, r in enumerate(base) if r.get("reason") == "quarantine"]
    q, q2 = quarantines[0], quarantines[1]
    capacity = next(i for i, r in enumerate(base) if r.get("reason") == "capacity")
    insert = next(i for i, r in enumerate(base) if r["op"] == "insert")
    texts = {"predicate": "prompt == g3", "note": "forged"}
    mutants: list[Mutant] = [
        (_edited(base, q, lambda r: r.pop("predicate")), None, "quarantine_without_predicate"),
        (_edited(base, q, lambda r: r.pop("note")), None, "quarantine_without_note"),
        (_edited(base, q, lambda r: r.update(predicate="")), None, "quarantine_predicate_empty"),
        (_edited(base, q, lambda r: r.update(predicate=5)), None, "quarantine_predicate_not_text"),
        (_edited(base, q, lambda r: r.update(predicate="x" * 1025)), None, "quarantine_predicate_too_long"),
        (_edited(base, q, lambda r: r.update(note="line one\nline two")), None, "quarantine_note_multiline"),
        (_edited(base, q, lambda r: r.update(note="x" * 257)), None, "quarantine_note_too_long"),
        (_edited(base, q, lambda r: r.update(reason="quarantined")), None, "quarantine_reason_misspelled"),
        (_edited(base, q, lambda r: r.update(reason="explicit")), None, "quarantine_texts_on_an_explicit_evict"),
        (_edited(base, capacity, lambda r: r.update(**texts)), None, "quarantine_texts_on_a_capacity_evict"),
        (_edited(base, insert, lambda r: r.update(**texts)), None, "quarantine_texts_on_an_insert"),
        (_edited(base, 0, lambda r: r.update(format="2")), None, "quarantine_in_a_format_2_log"),
        (_edited(base, q2, lambda r: r.update(index=base[q]["index"])), None, "quarantine_of_an_empty_slot"),
        (_edited(base, q, lambda r: r.update(new_priority_int=r["old_priority_int"])), None,
         "quarantine_keeps_the_leaf"),
    ]
    dup = copy.deepcopy(base)
    dup.insert(q + 1, copy.deepcopy(base[q]))
    mutants.append((_rechain(dup, q + 1), None, "quarantine_duplicated"))
    adv = next(k for k, r in enumerate(base) if r["op"] == "advance_version" and base[k + 1]["op"] == "evict")
    moved = copy.deepcopy(base)
    moved.insert(adv + 1, copy.deepcopy(base[q]))
    mutants.append((_rechain(moved, adv + 1), None, "quarantine_between_advance_and_pending_evicts"))
    return mutants


def manifest_mutants(base: list[dict], manifest: list[dict]) -> list[Mutant]:
    """Reward-provenance forgeries in the manifest; the log is untouched."""
    edits = [
        (lambda l: l.update(rewards={"verifier": 1.0, "judge": "high"}), "manifest_rewards_text_value"),
        (lambda l: l.update(rewards={"verifier": True}), "manifest_rewards_bool_value"),
        (lambda l: l.update(rewards={"verifier": float("nan")}), "manifest_rewards_nan"),
        (lambda l: l.update(rewards={"": 1.0}), "manifest_rewards_empty_name"),
        (lambda l: l.update(rewards=[1.0, 0.5]), "manifest_rewards_not_an_object"),
        (lambda l: l.update(judge_rationale="the answer looked fine"), "manifest_text_field_beside_rewards"),
    ]
    return [(base, _manifest_edited(manifest, 0, edit), name) for edit, name in edits]


def limit_mutants(base: list[dict], manifest: list[dict]) -> list[Mutant]:
    """Changes the log does not commit to; they must survive, and that is the documented limit."""
    q = next(i for i, r in enumerate(base) if r.get("reason") == "quarantine")
    return [
        (_edited(base, q, lambda r: r.update(predicate="prompt == g0")), manifest,
         "quarantine_predicate_text_changed_chain_consistent"),
        (_edited(base, q, lambda r: r.update(note="routine cleanup")), manifest,
         "quarantine_note_changed_chain_consistent"),
        (base, _manifest_edited(manifest, 0, lambda l: l.update(rewards={"verifier": 0.0, "judge": 0.0})),
         "manifest_rewards_value_changed"),
        (base, _manifest_edited(manifest, 0, lambda l: l.pop("rewards")), "manifest_rewards_dropped"),
    ]


def rejected(records: list[dict], manifest: Optional[list[dict]]) -> bool:
    try:
        verify_chain(records, manifest=manifest)
    except CheckerError:
        return True
    return False


def run_provenance_category(results: dict) -> None:
    """Append the ``provenance`` category and ``results["provenance_limit"]``.

    The limit mutants are not counted as survivors: their survival is the
    measurement. If one of them is rejected, the statement below would be
    false, so the campaign fails instead of reporting it.
    """
    base, manifest = build_provenance_run()
    mutants = log_mutants(base) + manifest_mutants(base, manifest)
    count = 0
    for records, lines, description in mutants:
        results["total_mutants"] += 1
        if rejected(records, lines):
            results["rejected"] += 1
            count += 1
        else:
            results["survived"].append(description)
    results["details"].append(("provenance", len(mutants), count))

    limits = limit_mutants(base, manifest)
    survive = [name for records, lines, name in limits if not rejected(records, lines)]
    if len(survive) != len(limits):
        results["survived"].append("provenance_limit_measurement_does_not_match_its_statement")
    results["provenance_limit"] = {
        "survive": survive,
        "statement": "the log does not commit to a quarantine record's predicate text or note, nor to the "
                     "manifest's per-reward-function values; they are the operator's and the adapter's "
                     "statements, recorded and checked for shape only",
    }
