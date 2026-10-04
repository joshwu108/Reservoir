"""benchmarks/modal/reproducible_grpo_real.py

The reproducibility demo (``demo/reproducible_grpo.py``) on a GPU, in two
tiers. Each tier runs the Phase 2 GRPO integration three times, exactly as
the CPU demo does: **a** and **b** with the same seeds, **c** with a
different data seed, then verifies every log with its manifest and runs
``checker.diff`` on the pairs.

Tier A: ``hf``
    HF ``generate`` on a T4, the Phase 2 image. HF generation on a GPU is
    not guaranteed to be bitwise reproducible run to run, so the point of
    this tier is to *measure* it: either a and b are identical (the GRPO
    run is bitwise reproducible on this hardware) or ``checker.diff``
    locates the first insert where the generated data differed and shows
    that Reservoir's draws were identical up to it. Both outcomes are
    reported as what they are.

Tier B: ``vllm``
    vLLM in TRL's colocate mode with ``VLLM_BATCH_INVARIANT=1`` (vLLM's
    batch-invariant kernels, which need compute capability 8.0 or higher,
    so an A10G or L4, not a T4) and ``vllm_importance_sampling_correction=False``
    (TRL only adds the vLLM importance-sampling keys the adapter refuses
    when that correction is on). The expected outcome is identical logs.
    The vLLM release that co-installs with ``trl==1.13.0`` and
    ``transformers==5.17.0`` is not known in advance: ``RESERVOIR_VLLM_VERSION``
    sets the pin for the image and the default below is a starting point
    for that dependency search, not a tested combination. If no image
    resolves, this tier is reported as blocked, with the error, and Tier A
    stands on its own.

Results land under ``benchmarks/modal/results/repro_<tier>_<gpu>_<steps>steps_seed<seed>/``
as ``{a,b,c}/attest.jsonl``, ``{a,b,c}/manifest.jsonl`` and ``report.json``
in the same shape as the CPU demo's report, and ``tests/test_trl_results.py``
re-verifies every committed ``repro_*`` directory.

Usage
-----
    modal run benchmarks/modal/reproducible_grpo_real.py --tier hf                    # T4, 12 steps x 3 runs
    modal run benchmarks/modal/reproducible_grpo_real.py --tier hf --max-steps 40
    RESERVOIR_VLLM_VERSION=0.11.0 modal run benchmarks/modal/reproducible_grpo_real.py --tier vllm

Cost estimate, not yet measured: Tier A about 3 x 10 min on a T4 at $0.59/h
(about $0.30); Tier B image builds plus 3 x 10 min on an A10G at $1.10/h
(about $1 to $3 if the image resolves).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

from benchmarks.modal.trl_replay_real import _DEPS, _repo_src, hf_cache, image, run_grpo

RESULTS_DIR = Path(__file__).parent / "results"
VLLM_VERSION = os.environ.get("RESERVOIR_VLLM_VERSION", "0.11.0")
VLLM_GPU = "A10G"
HF_GPU = "T4"

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*_DEPS, f"vllm=={VLLM_VERSION}")
    .env({"HF_HOME": "/hf_cache", "VLLM_BATCH_INVARIANT": "1"})
    .add_local_dir(_repo_src, remote_path="/reservoir_src")
)

app = modal.App("reservoir-reproducible-grpo")

VLLM_CONFIG = {
    "use_vllm": True,
    "vllm_mode": "colocate",
    "vllm_importance_sampling_correction": False,
    "vllm_gpu_memory_utilization": 0.3,
}


def run_triplet(max_steps: int, seed: int, buffer_seed: int, extra_config: dict | None = None) -> dict:
    """Runs a, b (same seeds) and c (data seed + 1); returns the three run records."""
    sys.path.insert(0, "/reservoir_src")
    out: dict = {}
    for name, data_seed in (("a", seed), ("b", seed), ("c", seed + 1)):
        started = time.time()
        result = run_grpo(
            max_steps=max_steps, seed=data_seed, buffer_seed=buffer_seed,
            attest_path=f"/tmp/{name}.attest.jsonl", manifest_path=f"/tmp/{name}.manifest.jsonl",
            output_dir=f"/tmp/{name}_trainer", extra_config=extra_config,
        )
        result["wall_clock_seconds"] = round(time.time() - started, 2)
        result["seed"], result["buffer_seed"] = data_seed, buffer_seed
        out[name] = result
    return out


@app.function(gpu=HF_GPU, image=image, volumes={"/hf_cache": hf_cache}, timeout=3600)
def run_hf(max_steps: int = 12, seed: int = 42, buffer_seed: int = 0) -> dict:
    return run_triplet(max_steps, seed, buffer_seed)


@app.function(gpu=VLLM_GPU, image=vllm_image, volumes={"/hf_cache": hf_cache}, timeout=3600)
def run_vllm(max_steps: int = 12, seed: int = 42, buffer_seed: int = 0) -> dict:
    return run_triplet(max_steps, seed, buffer_seed, extra_config=VLLM_CONFIG)


def write_triplet(results: dict, out_dir: Path) -> dict:
    """Write the three logs and manifests, verify, diff, and return the report."""
    from checker.diff import diff_logs
    from checker.transcript import build_transcript
    from checker.verify import verify_json_lines

    runs: dict = {}
    for name, result in results.items():
        run_dir = out_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        attest, manifest = run_dir / "attest.jsonl", run_dir / "manifest.jsonl"
        attest.write_text(result["attestation"]["text"])
        manifest.write_text(result["attestation"]["manifest_text"])
        verified = verify_json_lines(attest.read_text(), manifest=manifest.read_text())
        runs[name] = {
            "attest": str(attest.relative_to(RESULTS_DIR.parents[1])),
            "manifest": str(manifest.relative_to(RESULTS_DIR.parents[1])),
            "seed": result["seed"], "buffer_seed": result["buffer_seed"],
            "head_digest": result["attestation"]["head_digest"], "records": result["attestation"]["records"],
            "examples_committed": len(verified.content.history), "totals": result["totals"],
            "versions": result["versions"], "device": result["config"]["device"],
            "wall_clock_seconds": result["wall_clock_seconds"],
        }
    logs = {n: [json.loads(l) for l in (out_dir / n / "attest.jsonl").read_text().splitlines() if l] for n in runs}
    diffs = {"a_vs_b": diff_logs(logs["a"], logs["b"]), "a_vs_c": diff_logs(logs["a"], logs["c"])}
    transcript = build_transcript(verify_json_lines(
        (out_dir / "a" / "attest.jsonl").read_text(), manifest=(out_dir / "a" / "manifest.jsonl").read_text()))
    same = diffs["a_vs_b"]["identical"] and all(
        (out_dir / "a" / f).read_bytes() == (out_dir / "b" / f).read_bytes() for f in ("attest.jsonl", "manifest.jsonl"))
    first = diffs["a_vs_c"]["first_difference"]
    data = diffs["a_vs_c"]["class"] == "data" and first is not None and logs["a"][first]["op"] == "insert"
    report = {
        "demo": "reproducible_grpo", "device": runs["a"]["device"], "versions": runs["a"]["versions"],
        "runs": runs, "same_seed_identical": same, "different_data_seed_class": diffs["a_vs_c"]["class"],
        "diffs": diffs,
        "transcript_a": {"sampled_rows": transcript["sampled_rows"], "sources": transcript["sources"],
                         "examples_committed": len(transcript["content"])},
        # On a GPU, a and b differing is a finding about the generation engine,
        # not a failure of the comparison: the report says which it was.
        "verdict": "IDENTICAL" if same and data else ("GENERATION_NONDETERMINISTIC" if data else "UNEXPECTED"),
        "verdict_means": {
            "IDENTICAL": "a and b have byte-identical logs and manifests; a and c first differ on an insert "
                         "record classified data",
            "GENERATION_NONDETERMINISTIC": "a and b differ although seeds match; checker.diff locates the "
                                           "first insert where the generated data differed (class data); "
                                           "Reservoir's draws were identical up to it",
            "UNEXPECTED": "a difference not attributable to the generated data; inspect diffs",
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main(tier: str = "hf", max_steps: int = 12, seed: int = 42, buffer_seed: int = 0):
    if tier == "hf":
        results = run_hf.remote(max_steps=max_steps, seed=seed, buffer_seed=buffer_seed)
        gpu = HF_GPU.lower()
    elif tier == "vllm":
        results = run_vllm.remote(max_steps=max_steps, seed=seed, buffer_seed=buffer_seed)
        gpu = VLLM_GPU.lower()
    else:
        raise SystemExit(f"--tier must be hf or vllm, got {tier!r}")
    out_dir = RESULTS_DIR / f"repro_{tier}_{gpu}_{max_steps}steps_seed{seed}"
    report = write_triplet(results, out_dir)
    for name, run in report["runs"].items():
        print(f"run {name}: seed={run['seed']} records={run['records']} head={run['head_digest'][:16]}… "
              f"({run['wall_clock_seconds']}s on {run['device']})")
    print(f"a vs b: {report['diffs']['a_vs_b']['class']}; a vs c: {report['diffs']['a_vs_c']['class']} "
          f"at record {report['diffs']['a_vs_c']['first_difference']}")
    print(f"verdict: {report['verdict']} ({report['verdict_means'][report['verdict']]})")
    print(f"wrote {out_dir / 'report.json'}")
