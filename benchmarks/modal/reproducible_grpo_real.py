"""benchmarks/modal/reproducible_grpo_real.py

The reproducibility demo (``demo/reproducible_grpo.py``) on a GPU with HF
``generate``: the same three runs as the CPU demo (**a** and **b** with the
same seeds, **c** with a different data seed), on a T4, with the Phase 2
image and model. Every log is verified with its manifest and the pairs are
compared with ``checker.diff``.

HF generation on a GPU is not guaranteed to be bitwise reproducible run to
run, so one invocation runs the triplet twice, in two containers:

- ``plain``: PyTorch defaults, which is how the Phase 2 integration ran.
- ``deterministic``: ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` on the image and
  ``torch.use_deterministic_algorithms(True, warn_only=True)`` plus
  deterministic cuDNN in the process. The two differ in which kernels
  PyTorch may pick, so they are kept in separate containers (the cuBLAS
  setting is read when CUDA initialises).

Each variant ends with one of three verdicts, and all are reported as
what they are: ``IDENTICAL`` (a and b byte-identical, a and c first
differ on an insert classified ``data``), ``GENERATION_NONDETERMINISTIC``
(a and b differ although seeds match; ``checker.diff`` locates the first
insert where the generated data differed and shows that every draw before
it was identical), or ``UNEXPECTED``.

The vLLM batch-invariant tier is a separate script,
``benchmarks/modal/reproducible_grpo_vllm.py``, because it needs a
different image and model and must not be built when only this tier runs.

Results land under
``benchmarks/modal/results/repro_hf_t4_<variant>_<steps>steps_seed<seed>/``
as ``{a,b,c}/attest.jsonl``, ``{a,b,c}/manifest.jsonl`` and
``report.json``; ``tests/test_trl_results.py`` re-verifies every committed
``repro_*`` directory.

Usage
-----
    modal run benchmarks/modal/reproducible_grpo_real.py                  # 12 steps x 3 runs x 2 variants
    modal run benchmarks/modal/reproducible_grpo_real.py --max-steps 40
    modal run benchmarks/modal/reproducible_grpo_real.py --variants plain

Cost estimate, not yet measured: the tiny model trains 40 steps in about
ten seconds on CPU, so each container is dominated by start-up and the
first model download (about two to four minutes on a T4 at $0.59/h);
expect well under $0.50 for both variants.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

from benchmarks.modal.trl_replay_real import _repo_src, base_image, hf_cache, image, run_grpo

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "benchmarks" / "modal" / "results"
GPU = "T4"

deterministic_image = (
    base_image.env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    .add_local_dir(_repo_src, remote_path="/reservoir_src")
)

app = modal.App("reservoir-reproducible-grpo-hf")


def run_triplet(max_steps: int, seed: int, buffer_seed: int, *, torch_deterministic: bool = False,
                extra_config: dict | None = None, model_id: str | None = None) -> dict:
    """Runs a, b (same seeds) and c (data seed + 1) in this process; returns the three run records."""
    sys.path.insert(0, "/reservoir_src")
    out: dict = {}
    for name, data_seed in (("a", seed), ("b", seed), ("c", seed + 1)):
        started = time.time()
        kwargs = dict(
            max_steps=max_steps, seed=data_seed, buffer_seed=buffer_seed,
            attest_path=f"/tmp/{name}.attest.jsonl", manifest_path=f"/tmp/{name}.manifest.jsonl",
            output_dir=f"/tmp/{name}_trainer", extra_config=extra_config,
            torch_deterministic=torch_deterministic,
        )
        if model_id is not None:
            kwargs["model_id"] = model_id
        result = run_grpo(**kwargs)
        result["wall_clock_seconds"] = round(time.time() - started, 2)
        result["seed"], result["buffer_seed"] = data_seed, buffer_seed
        out[name] = result
    return out


@app.function(gpu=GPU, image=image, volumes={"/hf_cache": hf_cache}, timeout=3600)
def run_plain(max_steps: int = 12, seed: int = 42, buffer_seed: int = 0) -> dict:
    return run_triplet(max_steps, seed, buffer_seed)


@app.function(gpu=GPU, image=deterministic_image, volumes={"/hf_cache": hf_cache}, timeout=3600)
def run_deterministic(max_steps: int = 12, seed: int = 42, buffer_seed: int = 0) -> dict:
    return run_triplet(max_steps, seed, buffer_seed, torch_deterministic=True)


def write_triplet(results: dict, out_dir: Path, label: str) -> dict:
    """Write the three logs and manifests, verify, diff, and return (and write) the report."""
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
            "attest": str(attest.relative_to(REPO)),
            "manifest": str(manifest.relative_to(REPO)),
            "seed": result["seed"], "buffer_seed": result["buffer_seed"], "model": result["model"],
            "head_digest": result["attestation"]["head_digest"], "records": result["attestation"]["records"],
            "examples_committed": len(verified.content.history), "totals": result["totals"],
            "versions": result["versions"], "device": result["config"]["device"],
            "torch_deterministic": result["config"]["torch_deterministic"],
            "extra_config": result["config"]["extra_config"],
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
        "demo": "reproducible_grpo", "variant": label, "device": runs["a"]["device"],
        "model": runs["a"]["model"], "versions": runs["a"]["versions"],
        "runs": runs, "same_seed_identical": same, "different_data_seed_class": diffs["a_vs_c"]["class"],
        "diffs": diffs,
        "transcript_a": {"sampled_rows": transcript["sampled_rows"], "sources": transcript["sources"],
                         "examples_committed": len(transcript["content"])},
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


def print_report(report: dict, out_dir: Path) -> None:
    print(f"== {report['variant']} on {report['device']} ({report['model']}) ==")
    for name, run in report["runs"].items():
        print(f"run {name}: seed={run['seed']} records={run['records']} replaced_rows={run['totals']['replaced_rows']} "
              f"head={run['head_digest'][:16]}… ({run['wall_clock_seconds']}s)")
    ab, ac = report["diffs"]["a_vs_b"], report["diffs"]["a_vs_c"]
    print(f"a vs b: {ab['class']}" + (f" at record {ab['first_difference']}" if not ab["identical"] else ""))
    print(f"a vs c: {ac['class']} at record {ac['first_difference']}")
    print(f"verdict: {report['verdict']} ({report['verdict_means'][report['verdict']]})")
    print(f"wrote {out_dir / 'report.json'}")


@app.local_entrypoint()
def main(max_steps: int = 12, seed: int = 42, buffer_seed: int = 0, variants: str = "plain,deterministic"):
    wanted = [v.strip() for v in variants.split(",") if v.strip()]
    functions = {"plain": run_plain, "deterministic": run_deterministic}
    unknown = [v for v in wanted if v not in functions]
    if unknown:
        raise SystemExit(f"--variants must be from {sorted(functions)}, got {unknown}")
    for variant in wanted:
        results = functions[variant].remote(max_steps=max_steps, seed=seed, buffer_seed=buffer_seed)
        out_dir = RESULTS_DIR / f"repro_hf_{GPU.lower()}_{variant}_{max_steps}steps_seed{seed}"
        print_report(write_triplet(results, out_dir, variant), out_dir)
