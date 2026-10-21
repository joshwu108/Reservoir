"""The README's evidence matches the committed results it cites.

In the style of ``test_defaults_match_readme``: each number the README's
guarantees table and its durable and divergence paragraphs state is read
from the committed JSON (or log) and looked up in the README text, so a
re-run that moves a number fails here until the README follows. Counts
another branch is changing are written as ``[pending: ...]`` placeholders;
those rows check everything else and the count once the placeholder is
gone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text()
RESULTS = REPO / "results"
MODAL = REPO / "benchmarks" / "modal" / "results"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def row(claim_prefix: str) -> str:
    rows = [line for line in README.splitlines() if line.startswith("| " + claim_prefix)]
    assert len(rows) == 1, f"expected one guarantees row starting {claim_prefix!r}, found {len(rows)}"
    return rows[0]


def count(value) -> int:
    """The campaign report gives some measurements as counts and some as the lists of mutant names."""
    return len(value) if isinstance(value, list) else value


def test_mutation_row_matches_the_campaign_report():
    text = row("The independent checker rejects forged logs")
    report = load(RESULTS / "mutation_campaign_report.json")
    if "[pending: mutant total]" not in text:
        assert f"{report['total_mutants']}/{report['total_mutants']} mutants rejected" in text
    assert report["rejected"] == report["total_mutants"] and report["survived"] == []
    assert f"in {len(report['categories'])} categories" in text
    assert f"{count(report['content_limit']['undetectable_without_manifest'])} chain-consistent changes" in text
    assert count(report["content_limit"]["detected_with_manifest"]) == count(report["content_limit"]["mutants"])
    assert f"{count(report['draw_limit']['survive_without_draw_config'])} draw forgeries pass" in text
    assert f"{count(report['provenance_limit']['survive'])} changes to quarantine texts" in text
    assert f"{count(report['replay_manifest']['changed'])} manifest edits of reward values" in text
    assert f"{len(report['staleness_limit']['pass'])} staleness-policy parameter changes" in text


def test_crash_row_matches_the_campaign_report():
    text = row("The durable buffers are failure-atomic under SIGKILL")
    report = load(RESULTS / "crash_campaign_report.json")
    assert report["torn"] == 0 and report["passed"] == report["total"] and "zero torn" in text
    pre = sum(case["matches_pre"] for case in report["results"])
    post = sum(case["matches_post"] for case in report["results"])
    assert pre + post == report["total"] and f"the pre-state ({pre}) or the post-state ({post})" in text
    if "[pending: crash-campaign count]" not in text:
        assert f"{report['passed']}/{report['total']} crash tests" in text


def test_two_runs_row_matches_the_cpu_and_t4_reports():
    text = row("Two runs with the same inputs give one transcript")
    cpu = load(RESULTS / "reproducible_grpo_report.json")
    assert cpu["same_seed_identical"] and cpu["different_data_seed_class"] == "data"
    assert f"({cpu['runs']['a']['records']} records, one head digest)" in text
    assert f"differs at record {cpu['diffs']['a_vs_c']['first_difference']}" in text
    plain, deterministic = (MODAL / f"repro_hf_t4_{v}_12steps_seed42" for v in ("plain", "deterministic"))
    for variant in (plain, deterministic):
        report = load(variant / "report.json")
        assert report["verdict"] == "IDENTICAL" and "verdict `IDENTICAL`" in text
        assert f"`{report['device']}`" in text and f"torch {report['versions']['torch']}" in text
    # The README calls the two variants one observation: they must be byte-identical to each other.
    for run in ("a", "b", "c"):
        for name in ("attest.jsonl", "manifest.jsonl"):
            assert (plain / run / name).read_bytes() == (deterministic / run / name).read_bytes()
    assert "one observation" in text


def test_adapter_row_matches_the_trl_and_verl_records():
    text = row("The adapters survive a real trainer's call path")
    trl = load(MODAL / "trl_replay_tiny-qwen2_40steps_seed42_a10g-x2.json")
    verl = load(MODAL / "verl_replay_smollm2-135m-instruct_12steps_seed42.json")
    names = {rank["device_name"] for rank in trl["distributed"]["ranks"]}
    assert names == {"NVIDIA A10"} and len(trl["distributed"]["ranks"]) == 2 and "reports them as NVIDIA A10" in text
    assert f"TRL {trl['versions']['trl']}" in text and f"verl {verl['versions']['verl']}" in text
    assert f"one {verl['versions']['gpu']}" in text
    for record, steps in ((trl, 40), (verl, 12)):
        totals = record["totals"]
        assert record["config"]["max_steps"] == steps
        assert (f"{steps} steps, {totals['dead_groups']} dead groups replaced with {totals['replaced_rows']} rows, "
                f"{record['attestation']['records']} records, manifest opens all {totals['ingested_rows']} examples") in text
        assert record["attestation"]["checker"]["returncode"] == 0
    assert len(verl["buffer_checkpoints"]) == len(verl["trainer_checkpoints"])
    assert f"each of the {len(verl['trainer_checkpoints'])} trainer checkpoints" in text


def test_sweep_row_matches_the_committed_cpu_smoke():
    text = row("The staleness sweep harness")
    smoke = MODAL / "sweep_tiny_cpu_smoke"
    report = load(smoke / "report.json")
    arms = report["arms"]
    main = [a for a in arms.values() if not a["stress"]]
    stress = [name for name, a in arms.items() if a["stress"]]
    assert f"{len(arms)} arms (the {len(main)} of the main grid on seeds {' and '.join(map(str, report['seeds']))}" in text
    assert all(f"`{name}` stress arm on seed {arms[name]['runs'][0]['seed']}" in text for name in stress)
    assert f"{report['max_steps']} steps, {report['totals']['runs']} runs" in text
    assert f"the {report['verified_logs']} Reservoir logs verify" in text
    assert report["totals"]["suspect_runs"] == 0 and report["missing_arms"] == []
    assert "This is a mechanism run" in text


def test_sweep_results_match_the_committed_gpu_summary():
    text = row("The staleness sweep harness")
    summary = load(RESULTS / "staleness_sweep_summary.json")
    totals = summary["totals"]
    assert totals["suspect_runs"] == 0 and summary["missing_arms"] == []
    assert f"{totals['runs']} runs, 0 suspect, the {summary['verified_logs']} Reservoir logs verify" in text
    section = README[README.index("## Staleness sweep results"):README.index("## Guarantees")]
    for name, comparison in summary["paired_avg_at_k"].items():
        arm, base = name.split(" vs ")
        stated = (f"{comparison['mean_points']:+.2f} [{comparison['lo_points']:+.2f}, {comparison['hi_points']:+.2f}]")
        if base == "grpo":
            assert stated + f", {comparison['seeds_above']}/{comparison['n_seeds']} seeds above" in section, name
            mean = sum(summary["avg_at_k_final"][arm].values()) / len(summary["avg_at_k_final"][arm])
            assert f"| {len(summary['avg_at_k_final'][arm])} | {mean:.3f} |" in section, name
        elif name == "reservoir_age32_on vs reservoir_age32_off":
            assert stated in section and f"above it on {comparison['seeds_above']}/{comparison['n_seeds']} seeds" in section


def _ratio(configs: dict, name: str) -> str:
    return f"{configs[name]['add_group_us_per_rollout'] / configs['none']['add_group_us_per_rollout']:.1f}×"


def test_attestation_row_matches_the_overhead_benchmark():
    text = row("What attestation and checking cost")
    report = load(RESULTS / "attestation_overhead.json")
    configs, where = report["configs"], report["provenance"]
    assert where["reservoir_version"] == "0.6.0" and "0.6.0 working tree" in text
    assert f"Python {where['python'].rsplit('.', 1)[0]}" in text
    assert f"medians of {report['workload']['repeats']} interleaved repeats" in text
    load_before = report["load_average_1_5_15_min"]["before"][0]
    assert f"load average about {round(load_before)}" in text
    assert (f"{_ratio(configs, 'memory')} the unattested insert with an in-memory log, "
            f"{_ratio(configs, 'file')} with a log file and {_ratio(configs, 'file+manifest')} with a manifest") in text
    steady = {name: configs[name]["sample_us_per_draw_steady"] for name in configs}
    attested = sorted(steady[name] for name in ("memory", "file", "file+manifest"))
    assert f"{steady['none']:.1f} µs unattested and {attested[0]:.1f} to {attested[-1]:.1f} µs attested" in text
    assert f"over the first {report['workload']['samples']} batches its draws average {configs['none']['sample_us_per_draw']:.1f} µs" in text
    assert f"in {report['checker']['verify_with_manifest_seconds_per_10k_records']:.2f} s" in text


def test_durable_paragraph_matches_the_durable_benchmark():
    report = load(RESULTS / "durable_overhead.json")
    d, m, w = report["durable"], report["in_memory_attested"], report["workload"]
    assert report["provenance"]["reservoir_version"] == "0.6.0"
    assert f"({w['groups']} groups of {w['group_size']} rollouts, {w['tokens_per_rollout']} tokens each" in README
    assert (f"an `add_group` took {d['add_group_ms_first_tenth']:.1f} ms in the first tenth of the run and "
            f"{d['add_group_ms_last_tenth']:.1f} ms in the last, against {m['add_group_ms_first_tenth']:.1f} and "
            f"{m['add_group_ms_last_tenth']:.1f} ms in memory") in README
    median, last = d["compaction_ms_median"], d["compaction_ms_last"]
    assert f"compaction took {median / 1e3:.1f} s at the median and {last / 1e3:.1f} s by the end" in README
    assert (f"({median / w['compact_every']:.1f} ms per operation amortised at the median, "
            f"{last / w['compact_every']:.1f} ms at the end)") in README
    assert f"reopening replayed in {report['reopen_seconds']:.1f} s" in README
    assert "0.5.0 code" not in README


def test_divergence_paragraph_matches_the_campaign_and_power_reports():
    report = load(RESULTS / "divergence_campaign_report.json")
    power = load(RESULTS / "divergence_power.json")
    run, grid = report["total_workloads"], report["preregistered_total_workloads"]
    assert f"{run / grid:.0%} of its grid ({run:,} of {grid:,} workload programs)" in README
    assert report["total_divergences"] == 0
    assert max(report["tv_max_openai"], report["tv_max_labml"]) < 2.0 ** -40 and "below 2^-40" in README
    expected = max(power["expected_divergences"]["preregistered_grid"].values())
    assert f"about {expected:.1e}" in README


def _without_comments(tla: str) -> str:
    tla = re.sub(r"\(\*.*?\*\)", "", tla, flags=re.S)
    return re.sub(r"\\\*[^\n]*", "", tla)


def _definition(spec: str, name: str) -> str:
    body = re.search(rf"^{name} ==\n(.*?)(?=^\S)", spec, flags=re.S | re.M)
    assert body, name
    return " ".join(_without_comments(body.group(1)).split())


def test_tla_row_describes_what_the_counterexample_does():
    text = row("The snapshot protocol is safe within a finite scope")
    spec = (REPO / "spec" / "ReplayLifecycle.tla").read_text()
    # The variant's recovery is the correct one, comments aside, as the README says ...
    assert _definition(spec, "Recover") == _definition(spec, "RecoverNoParentFsync")
    assert "its recovery action is the correct one" in text
    # ... and the violation comes from an action that assigns a torn state.
    assert "current_state' = torn" in _definition(spec, "FilesystemRevertNoParentFsync")
    assert "sets the state to a torn one directly" in text
    assert "spec/check.sh" in (REPO / ".github" / "workflows" / "ci.yml").read_text()


@pytest.mark.parametrize("phrase", ["221/221", "315/315", "about 3× the no-attestation insert", "Attestation is cheap"])
def test_superseded_figures_are_gone(phrase):
    assert phrase not in README
