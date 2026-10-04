"""
campaigns.mutation_content — Content-commitment forgeries for the mutation campaign.

Extends ``campaigns.mutation`` with forgeries of the fields that bind a
buffer slot to a training example: the ``content_digest`` and ``source``
on insert records, and the manifest that opens those digests. Every
mutant is re-chained so the hash chain is intact and only the semantic
checks in ``checker/content.py`` can catch it.

Two categories, because the checker has two levels of knowledge:

``content``
    Forgeries the checker catches from the log alone: malformed digests
    or sources, content fields on the wrong record type, a log that mixes
    digested and undigested inserts, a sample of a slot that holds no
    committed example.

``content_manifest``
    Forgeries that need the manifest. Most tamper with the manifest
    (changed tokens, reward, prompt, source or entry version; a missing,
    extra, duplicated or reordered line); the checker rejects every one
    when given the manifest. Three tamper with the *log* in a
    chain-consistent way (swap one insert's digest for another valid
    digest, change or drop its source): the log alone cannot reveal that,
    because the log commits to whatever digest was written. These are the
    mutants listed under ``undetectable_without_manifest`` in the
    campaign report. They are the documented limit of the log, not
    checker defects: a published log proves what was committed; the
    manifest proves what the commitments were commitments *to*. The
    campaign fails if that measurement ever changes, so the statement in
    the report always matches what was measured.

``run_content_categories`` is called by ``campaigns.mutation`` and
appends to its results.
"""

from __future__ import annotations

import copy
import json
from typing import Callable, Optional

from checker.verify import CheckerError, verify_chain

Mutant = tuple[list[dict], Optional[list[dict]], str]
"""(records, manifest or None, description). A None manifest means "check from the log alone"."""


def build_content_chain(seed: int = 7) -> tuple[list[dict], list[dict]]:
    """A RolloutBuffer run with three sources, evictions, slot reuse and a rebase, plus its manifest."""
    from reservoir.attest import AttestationLog
    from reservoir.rollout import Rollout
    from reservoir.rollout_buffer import RolloutBuffer
    from reservoir.rollout_manifest import ManifestWriter

    log, manifest = AttestationLog(), ManifestWriter()
    buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=2, seed=seed, attest=log, manifest=manifest)
    sources = ["licensed", "scraped", None]
    for v in range(10):
        buf.add_group(
            f"g{v}", v,
            [Rollout(tokens=[v + 1, 2], logprobs=[-0.1, -0.2], reward=r) for r in (1.0, 0.0, 0.5)],
            source=sources[v % 3],
        )
        batch = buf.sample(4, current_version=v)
        buf.update_priorities(batch.indices[:1], [0.7])
    assert buf.n_rebases >= 1
    records, lines = [dict(r) for r in log.records], manifest.records
    verify_chain(records, manifest=lines)  # baseline must be valid
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


def _first(records: list[dict], op: str, **match) -> int:
    for i, r in enumerate(records):
        if r["op"] == op and all(r.get(k) == v for k, v in match.items()):
            return i
    raise RuntimeError(f"baseline has no {op} record matching {match}")


def _edited(records: list[dict], idx: int, edit: Callable[[dict], None]) -> list[dict]:
    mutant = copy.deepcopy(records)
    edit(mutant[idx])
    return _rechain(mutant, idx)


def log_only_mutants(base: list[dict]) -> list[Mutant]:
    """Forgeries of the content fields that the log alone exposes."""
    i = _first(base, "insert", source="licensed")
    u = _first(base, "update")
    e = _first(base, "evict")
    mutants: list[Mutant] = []
    for bad in ("AB" * 32, "ab" * 31, "zz" * 32, 7, ""):
        mutants.append((_edited(base, i, lambda r, b=bad: r.update(content_digest=b)), None,
                        f"insert_digest_malformed_{bad!r:.12}"))
    for bad in ("", "a\nb", "x" * 257, "   ", 3):
        mutants.append((_edited(base, i, lambda r, b=bad: r.update(source=b)), None,
                        f"insert_source_malformed_{bad!r:.12}"))
    mutants.append((_edited(base, i, lambda r: r.pop("content_digest")), None, "insert_source_without_digest"))
    mutants.append((_edited(base, e, lambda r: r.update(content_digest="ab" * 32)), None, "evict_with_content_digest"))
    mutants.append((_edited(base, u, lambda r: r.update(content_digest="ab" * 32)), None, "update_with_content_digest"))
    second = i + 1
    assert base[second]["op"] == "insert"
    mutants.append((_edited(base, second, lambda r: (r.pop("content_digest"), r.pop("source", None))), None,
                    "insert_without_digest_in_content_log"))
    return mutants


def manifest_mutants(base: list[dict], manifest: list[dict]) -> list[Mutant]:
    """Tampered manifests; every one is rejected when the checker is given the manifest."""
    from reservoir.rollout import content_digest_of

    def m(edit: Callable[[list[dict]], None], name: str) -> Mutant:
        lines = copy.deepcopy(manifest)
        edit(lines)
        return base, lines, name

    consistent = copy.deepcopy(manifest[0])
    consistent["tokens"] = [9, 9]
    consistent["content_digest"] = content_digest_of(consistent["prompt_id"], [9, 9],
                                                     float.fromhex(consistent["reward_hex"]))
    return [
        m(lambda L: L[3].update(tokens=[9, 9]), "manifest_tokens_changed"),
        m(lambda L: L[0].update(reward_hex=(2.0).hex()), "manifest_reward_changed"),
        m(lambda L: L[0].update(prompt_id="other"), "manifest_prompt_changed"),
        m(lambda L: L[0].update(source="elsewhere"), "manifest_source_changed"),
        m(lambda L: L[0].update(source=None), "manifest_source_dropped"),
        m(lambda L: L[0].update(entry_version=L[0]["entry_version"] + 1), "manifest_entry_version_changed"),
        m(lambda L: L.pop(), "manifest_line_missing"),
        m(lambda L: L.append(dict(L[-1], op_counter=999)), "manifest_line_extra"),
        m(lambda L: L.append(copy.deepcopy(L[0])), "manifest_line_duplicated"),
        m(lambda L: L.__setitem__(slice(0, 2), [L[1], L[0]]), "manifest_lines_reordered"),
        m(lambda L: L.__setitem__(0, consistent), "manifest_line_self_consistent_but_different_example"),
        m(lambda L: L[0].update(op_counter=L[0]["op_counter"] + 1), "manifest_op_counter_changed"),
    ]


def chain_consistent_log_mutants(base: list[dict], manifest: list[dict]) -> list[Mutant]:
    """Log forgeries that are valid logs: only the manifest can expose them."""
    i = _first(base, "insert", source="licensed")
    j = _first(base, "insert", source="scraped")
    other_digest = base[j]["content_digest"]
    return [
        (_edited(base, i, lambda r: r.update(content_digest=other_digest)), manifest,
         "insert_digest_swapped_chain_consistent"),
        (_edited(base, i, lambda r: r.update(source="scraped")), manifest,
         "insert_source_changed_chain_consistent"),
        (_edited(base, i, lambda r: r.pop("source")), manifest,
         "insert_source_dropped_chain_consistent"),
    ]


def rejected(records: list[dict], manifest: Optional[list[dict]]) -> bool:
    """True if the checker rejects the mutant (with the manifest when given)."""
    try:
        verify_chain(records, manifest=manifest)
    except CheckerError:
        return True
    return False


def run_content_categories(results: dict) -> None:
    """Run both categories and append to ``results`` the way ``campaigns.mutation`` does.

    ``results["details"]`` gains ``("content", n, rejected)`` and
    ``("content_manifest", n, rejected)``; ``results["content_limit"]``
    records how many chain-consistent log forgeries the log alone cannot
    reveal (expected to equal the number of such mutants) and that every
    one of them is caught with the manifest.
    """
    base, manifest = build_content_chain()
    log_only = log_only_mutants(base)
    limit_mutants = chain_consistent_log_mutants(base, manifest)
    with_manifest = manifest_mutants(base, manifest) + limit_mutants

    def run(name: str, mutants: list[Mutant]) -> None:
        count = 0
        for records, lines, description in mutants:
            results["total_mutants"] += 1
            if rejected(records, lines):
                results["rejected"] += 1
                count += 1
            else:
                results["survived"].append(description)
        results["details"].append((name, len(mutants), count))

    run("content", log_only)
    run("content_manifest", with_manifest)

    # The documented limit: the same chain-consistent log forgeries, checked
    # from the log alone, survive. Measured here, never counted as survivors.
    # The statement below is only true if every one of them survives without
    # the manifest and every one is rejected with it; anything else fails the
    # campaign so the report can never claim what was not measured.
    undetectable = [name for records, _, name in limit_mutants if not rejected(records, None)]
    detected = [name for records, lines, name in limit_mutants if rejected(records, lines)]
    if len(undetectable) != len(limit_mutants) or len(detected) != len(limit_mutants):
        results["survived"].append("content_limit_measurement_does_not_match_its_statement")
    results["content_limit"] = {
        "undetectable_without_manifest": len(undetectable),
        "detected_with_manifest": len(detected),
        "mutants": undetectable,
        "statement": "a chain-consistent change to an insert's content_digest or source is not "
                     "detectable from the log alone; the log commits, the manifest opens",
    }


if __name__ == "__main__":
    out: dict = {"total_mutants": 0, "rejected": 0, "survived": [], "details": []}
    run_content_categories(out)
    print(json.dumps(out, indent=2))
