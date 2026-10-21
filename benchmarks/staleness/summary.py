"""benchmarks/staleness/summary.py — the committed summary and figures of a GPU sweep.

A GPU sweep directory is about 260 MB (each Reservoir run's manifest alone is
about 10 MB), too large to commit. The raw records, logs and manifests stay on
the ``reservoir-sweep-results`` Modal volume and in the local run directory;
this module writes what is committed:

- ``slim_report``: ``report.json`` without its per-step series (2.4 MB of the
  4.8 MB), which the figures below summarise.
- ``paired_avg_at_k``: each comparison's difference in final avg@k (GSM8K test,
  every item), per seed and as a paired bootstrap over seeds and items
  (``stats.paired_bootstrap_diff``), in accuracy points. avg@k is the
  steadiest evaluation the sweep has; the report's own ``eval_diff_vs_grpo``
  is on greedy accuracy.
- ``plot_summary``: the paired-difference bar chart, the avg@k curves and the
  replay age against drift scatter.

    python -m benchmarks.staleness.summary benchmarks/modal/results/sweep_0.5b_300steps \\
        --out results/staleness_sweep_summary.json --figures results
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Optional

from benchmarks.provenance import stamped
from benchmarks.staleness.stats import BOOT_SEED, N_BOOT, paired_bootstrap_diff

AVG_AT_K = "gsm8k_avg_at_k"
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("grpo_filter", "grpo"),
    ("reservoir_age8_off", "grpo"),
    ("reservoir_age8_on", "grpo"),
    ("reservoir_age32_off", "grpo"),
    ("reservoir_age32_on", "grpo"),
    ("reservoir_naive_age32", "grpo"),
    ("reservoir_age128_on", "grpo"),
    ("wreinforce", "grpo"),
    ("reservoir_age32_on", "reservoir_age32_off"),
    ("reservoir_age8_on", "reservoir_age8_off"),
)
LABELS = {
    "grpo_filter": "grpo_filter (DAPO filter)\n1.9x rollouts", "reservoir_age8_off": "reservoir age 8",
    "reservoir_age8_on": "reservoir age 8 + policy", "reservoir_age32_on": "reservoir age 32 + policy",
    "reservoir_age128_on": "reservoir age 128 + policy\n(1 seed)", "reservoir_naive_age32": "naive replay age 32",
    "reservoir_age32_off": "reservoir age 32, no policy", "wreinforce": "wreinforce", "grpo": "grpo",
}
COLORS = {
    "grpo": "black", "grpo_filter": "#888888", "wreinforce": "#888888",
    "reservoir_age8_off": "#1f6fb4", "reservoir_age8_on": "#1f6fb4",
    "reservoir_age32_on": "#2a9d5c", "reservoir_age128_on": "#2a9d5c",
    "reservoir_naive_age32": "#d9822b", "reservoir_age32_off": "#d9822b",
}


def run_records(run_dir: Path) -> list[dict]:
    """Every ``<arm>_seed<seed>.json`` run record in ``run_dir``."""
    return [json.loads(p.read_text()) for p in sorted(Path(run_dir).glob("*_seed*.json"))
            if not p.name.endswith(".failed.json")]


def _final_avg_at_k(record: dict) -> Optional[dict]:
    evals = (record.get("evals") or {}).get(AVG_AT_K) or []
    return evals[-1] if evals and evals[-1].get("per_item") else None


def final_avg_at_k(records: list[dict]) -> dict[str, dict[int, float]]:
    """Each arm's final avg@k accuracy by seed."""
    out: dict[str, dict[int, float]] = {}
    for record in records:
        final = _final_avg_at_k(record)
        if final is not None:
            out.setdefault(record["arm"], {})[record["seed"]] = final["accuracy"]
    return out


def paired_avg_at_k(records: list[dict], comparisons=COMPARISONS, n_boot: int = N_BOOT,
                    rng_seed: int = BOOT_SEED) -> dict:
    """``arm - base`` in final avg@k points, per shared seed and as a paired bootstrap CI."""
    items: dict[str, dict[int, list[float]]] = {}
    for record in records:
        final = _final_avg_at_k(record)
        if final is not None:
            items.setdefault(record["arm"], {})[record["seed"]] = final["per_item"]
    out = {}
    for arm, base in comparisons:
        if arm not in items or base not in items:
            continue
        diff = paired_bootstrap_diff(items[arm], items[base], n_boot=n_boot, rng_seed=rng_seed)
        if diff is None:
            continue
        per_seed = {s: round(100 * (statistics.fmean(items[arm][s]) - statistics.fmean(items[base][s])), 2)
                    for s in diff["seeds"]}
        out[f"{arm} vs {base}"] = {
            "per_seed_points": per_seed, "mean_points": round(100 * diff["mean"], 2),
            "lo_points": round(100 * diff["lo"], 2), "hi_points": round(100 * diff["hi"], 2),
            "seeds_above": sum(1 for v in per_seed.values() if v > 0), "n_seeds": len(per_seed),
            "alpha": diff["alpha"], "n_boot": diff["n_boot"], "n_items": diff["n_items"],
        }
    return out


def slim_report(report: dict) -> dict:
    """``report`` without the per-step series."""
    return {k: v for k, v in report.items() if k != "per_step"}


def build_summary(run_dir: Path, n_boot: int = N_BOOT) -> dict:
    run_dir = Path(run_dir)
    records = run_records(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    return {**slim_report(report), "avg_at_k_final": final_avg_at_k(records),
            "paired_avg_at_k": paired_avg_at_k(records, n_boot=n_boot)}


def plot_summary(run_dir: Path, out_dir: Path) -> list[Path]:
    """The three README figures; returns their paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = run_records(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paired = paired_avg_at_k(records, [(a, "grpo") for a, b in COMPARISONS if b == "grpo"], n_boot=200)
    paths = [out_dir / "staleness_sweep_paired_delta.png", out_dir / "staleness_sweep_avg_at_k.png",
             out_dir / "staleness_sweep_age_vs_drift.png"]

    arms = [k.split(" vs ")[0] for k in paired]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, arm in enumerate(arms):
        c = paired[f"{arm} vs grpo"]
        ax.barh(i, c["mean_points"], color=COLORS[arm], alpha=0.85)
        ax.scatter(list(c["per_seed_points"].values()), [i] * c["n_seeds"], color="black", s=14, zorder=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([LABELS[a] for a in arms], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("GSM8K avg@8 at step 300: points vs grpo on the same seed (bar = mean, dots = seeds)", fontsize=9)
    ax.set_title("Qwen2.5-0.5B GRPO on GSM8K, 300 steps, 3 seeds\n"
                 "fresh or policy-controlled replay helps; stale uncontrolled replay hurts", fontsize=10)
    fig.tight_layout()
    fig.savefig(paths[0], dpi=150)
    plt.close(fig)

    by_arm: dict[str, list[list[tuple[int, float]]]] = {}
    for record in records:
        evals = (record.get("evals") or {}).get(AVG_AT_K) or []
        by_arm.setdefault(record["arm"], []).append([(e["step"], e["accuracy"]) for e in evals])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for arm in ("grpo", "grpo_filter", "reservoir_age8_off", "reservoir_age32_on", "reservoir_age32_off"):
        curves = by_arm.get(arm)
        if not curves:
            continue
        steps = [s for s, _ in curves[0]]
        means = [statistics.fmean(c[k][1] for c in curves) for k in range(len(steps))]
        ax.plot(steps, means, color=COLORS[arm], ls="--" if arm == "grpo_filter" else "-", marker="o", ms=4,
                label=LABELS[arm].replace("\n", " "))
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("GSM8K avg@8 (1,319 test items, mean of seeds)")
    ax.set_title("Held-out accuracy during training", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths[1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    seen = set()
    for record in records:
        arm = record["arm"]
        steps = [e for e in record.get("log_history", []) if "reservoir/staleness_mean" in e]
        if not arm.startswith("reservoir") or not steps:
            continue
        age = statistics.fmean(e["reservoir/staleness_mean"] for e in steps)
        drift = statistics.fmean(e["reservoir/log_ratio_mean_abs"] for e in steps if "reservoir/log_ratio_mean_abs" in e)
        ax.scatter([age], [drift], color=COLORS[arm], marker="o" if arm.endswith("_on") else "x", s=40,
                   label=None if arm in seen else LABELS[arm].replace("\n", " "))
        seen.add(arm)
    ax.set_xlabel("mean age of replayed rows (optimizer steps)")
    ax.set_ylabel("mean |log ratio|, current vs behavior policy")
    ax.set_title("Older replay has drifted further from the current policy", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths[2], dpi=150)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="the summary JSON")
    parser.add_argument("--figures", type=Path, default=None, help="directory for the three PNGs")
    cli = parser.parse_args()
    summary = build_summary(cli.run_dir)
    # The sweep's own provenance (where the runs ran) stays as report_provenance; the stamp is this summary's.
    summary = {"report_provenance": summary.get("provenance"), **{k: v for k, v in summary.items() if k != "provenance"}}
    cli.out.write_text(json.dumps(stamped(summary, gpu=summary.get("gpu")), indent=2) + "\n")
    print(f"wrote {cli.out}")
    if cli.figures is not None:
        for path in plot_summary(cli.run_dir, cli.figures):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
