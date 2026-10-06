"""
campaigns.mutation_replay — Tamperings of the manifest against the offline replay.

``reservoir_checker.replay`` reconstructs every witnessed training batch
as content from the log and the manifest. The log is left untouched here
(its forgeries are the other categories); the manifest is edited one
claim at a time and the replay is run.

The category ``replay_manifest`` counts the tamperings the replay must
refuse: a changed token, reward, prompt, source, version, slot, operation
counter, order or count, a recomputed digest, a non-canonical reward
spelling. Each breaks the replay because the manifest no longer opens the
log's commitments, and each is counted as rejected only then.

Beside the category, as a measurement and not as rejections, three
tamperings must *change* the output without breaking it: a changed or
dropped per-reward-function value. The log does not commit to those
values (the ``provenance`` category's documented limit), so a single
manifest cannot be told from a tampered one; only an untampered baseline
shows the difference, and only in the ``rewards`` fields. The campaign
fails if a tampering lands on the other side of that line or if the
difference reaches any other field.
"""

from __future__ import annotations

import copy

from reservoir_checker.replay import render, replay
from reservoir_checker.verify import CheckerError, verify_chain

Mutant = tuple[list[dict], str]   # manifest, description


def build_replayed_run(seed: int = 5) -> tuple[list[dict], list[dict]]:
    """Inserts with per-reward-function values, one witnessed sample per version, capacity evicts."""
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
    records = [dict(r) for r in buf.attestation_log.records]
    lines = [dict(l) for l in manifest.records]
    verify_chain(records, manifest=lines)
    return records, lines


def _edited(manifest: list[dict], idx: int, edit) -> list[dict]:
    m = copy.deepcopy(manifest)
    edit(m[idx])
    return m


def _non_canonical_spelling(reward_hex: str) -> str:
    """The same value with one more mantissa digit; ``float.hex()`` never spells it so."""
    mantissa, exponent = reward_hex.split("p")
    spelled = f"{mantissa}0p{exponent}"
    if float.fromhex(spelled) != float.fromhex(reward_hex) or spelled == reward_hex:
        raise RuntimeError(f"could not respell {reward_hex!r}")
    return spelled


def breaking_mutants(manifest: list[dict], replayed: int) -> list[Mutant]:
    """Tamperings the replay must refuse; ``replayed`` is the line that opens a replayed row."""
    from reservoir.rollout_manifest import manifest_record   # only to forge a self-consistent digest

    def recomputed_digest(line: dict) -> None:
        line["tokens"] = line["tokens"] + [7]
        line["content_digest"] = manifest_record(
            op_counter=line["op_counter"], index=line["index"], prompt_id=line["prompt_id"],
            tokens=line["tokens"], reward=float.fromhex(line["reward_hex"]), source=line["source"],
            entry_version=line["entry_version"],
        )["content_digest"]

    r = replayed
    if manifest[r]["tokens"][0] != 1 or r + 1 >= len(manifest):
        # The bool mutant must keep the value (True == 1) so it is a type claim only, and the
        # swap needs a successor line.
        raise RuntimeError("the replayed line must start with token 1 and have a successor; change the seed")
    edits = [
        (lambda l: l["tokens"].__setitem__(0, l["tokens"][0] + 1), "manifest_token_changed"),
        (lambda l: l["tokens"].append(9), "manifest_token_appended"),
        (lambda l: l.update(tokens=[True] + l["tokens"][1:]), "manifest_token_bool"),
        (lambda l: l.update(reward_hex=(float.fromhex(l["reward_hex"]) + 1.0).hex()), "manifest_reward_changed"),
        (lambda l: l.update(reward_hex=_non_canonical_spelling(l["reward_hex"])), "manifest_reward_non_canonical_spelling"),
        (lambda l: l.update(prompt_id="other"), "manifest_prompt_changed"),
        (lambda l: l.update(source="synthetic"), "manifest_source_changed"),
        (lambda l: l.update(source=None), "manifest_source_dropped"),
        (recomputed_digest, "manifest_tokens_changed_digest_recomputed"),
        (lambda l: l.update(entry_version=l["entry_version"] + 1), "manifest_entry_version_changed"),
        (lambda l: l.update(index=(l["index"] + 1) % 8), "manifest_slot_changed"),
        (lambda l: l.update(op_counter=l["op_counter"] + 1), "manifest_op_counter_changed"),
    ]
    mutants: list[Mutant] = [(_edited(manifest, r, edit), name) for edit, name in edits]
    dropped = copy.deepcopy(manifest)
    del dropped[r]
    mutants.append((dropped, "manifest_line_dropped"))
    duplicated = copy.deepcopy(manifest)
    duplicated.insert(r, copy.deepcopy(manifest[r]))
    mutants.append((duplicated, "manifest_line_duplicated"))
    swapped = copy.deepcopy(manifest)
    swapped[r], swapped[r + 1] = swapped[r + 1], swapped[r]
    mutants.append((swapped, "manifest_lines_swapped"))
    appended = copy.deepcopy(manifest) + [copy.deepcopy(manifest[r])]
    mutants.append((appended, "manifest_line_appended"))
    return mutants


def changing_mutants(manifest: list[dict], replayed: int, generated: int) -> list[Mutant]:
    """Tamperings the replay must report, not refuse: the values the log does not commit to."""
    zeroed = {"verifier": 0.0, "judge": 0.0}
    return [
        (_edited(manifest, replayed, lambda l: l.update(rewards=zeroed)), "manifest_rewards_changed_on_replayed_row"),
        (_edited(manifest, replayed, lambda l: l.pop("rewards")), "manifest_rewards_dropped_on_replayed_row"),
        (_edited(manifest, generated, lambda l: l.update(rewards=zeroed)), "manifest_rewards_changed_on_generated_example"),
    ]


def broke(records: list[dict], manifest: list[dict]) -> bool:
    try:
        replay(records, manifest)
    except CheckerError:
        return True
    return False


def differing_paths(a: object, b: object, path: tuple = ()) -> list[tuple]:
    """Key paths at which two JSON values differ (a list index is a path element too)."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b), key=str)
        return [p for k in keys for p in differing_paths(a.get(k), b.get(k), path + (k,))]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in differing_paths(x, y, path + (i,))]
    return [] if a == b else [path]


def changed_only_rewards(records: list[dict], manifest: list[dict], base: list[dict]) -> bool:
    """The replay runs, differs from ``base``, and every difference is inside a ``rewards`` field."""
    try:
        lines = replay(records, manifest)
    except CheckerError:
        return False
    paths = differing_paths(base, lines)
    return bool(paths) and all("rewards" in p for p in paths)


def _replayed_and_generated_lines(base: list[dict], n_lines: int) -> tuple[int, int]:
    """A manifest line that opens a replayed row, and one no row holds."""
    held = [r["manifest_line"] for b in base[1:] for r in b["rows"]]
    if not held:
        raise RuntimeError("no witness in the baseline replayed a row; change the seed")
    generated = next((i for i in range(n_lines) if i not in set(held)), None)
    if generated is None:
        raise RuntimeError("every manifest line is replayed; change the sizes")
    return held[0], generated


def run_replay_category(results: dict) -> None:
    """Append the ``replay_manifest`` category and the ``results["replay_manifest"]`` measurement.

    Only the breaking tamperings are counted as mutants and rejections.
    The changing ones are the measurement: each must run, differ from the
    untampered replay, and differ only in ``rewards``. If one is refused,
    unchanged, or changes anything else, the statement below would be
    false, so the campaign fails instead of reporting it.
    """
    records, manifest = build_replayed_run()
    base = replay(records, manifest)
    replayed, generated = _replayed_and_generated_lines(base, len(manifest))
    mutants = breaking_mutants(manifest, replayed)
    count = 0
    for lines, description in mutants:
        results["total_mutants"] += 1
        if broke(records, lines):
            results["rejected"] += 1
            count += 1
        else:
            results["survived"].append(description)
    results["details"].append(("replay_manifest", len(mutants), count))

    changing = changing_mutants(manifest, replayed, generated)
    changed = [name for lines, name in changing if changed_only_rewards(records, lines, base)]
    if len(changed) != len(changing):
        results["survived"].append("replay_manifest_measurement_does_not_match_its_statement")
    results["replay_manifest"] = {
        "broke": [name for _, name in mutants], "changed": changed,
        "statement": "a tampered manifest either fails to open the log's commitments, so the replay refuses it, "
                     "or changes a per-reward-function value, which the log does not commit to; the replay then "
                     "runs and reports the manifest's value, and nothing else in its output moves",
    }
