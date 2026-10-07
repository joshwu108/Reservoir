"""Arms, cost estimate, per-run summaries, the sweep report and its figure.

Everything here is plain Python over the run records that
``benchmarks/modal/staleness_sweep.run_arm`` returns (and the JSON files
``write_run`` writes), so the report can be rebuilt offline from a results
directory and tested on synthetic records. Rebuilding re-runs the
independent checker on every committed log (``reverify``), so a report
never rests on a verdict copied from a JSON file.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

AGES: tuple[int, ...] = (8, 32, 128)
POLICIES: tuple[str, ...] = ("off", "gate", "staleness")
DEFAULT_MAX_LOG_RATIO = 2.0
"""Legacy drift gate used as the "policy on" arm until S1's StalenessPolicy lands: a replayed row whose
absolute sequence log-ratio (sum over completion tokens of current minus stored logprob) exceeds this is
declined. e^2 is a 7.4x sequence-probability ratio either way; at age 8 it rarely fires, at age 128 it should."""

PER_STEP_KEYS: dict[str, str] = {
    "reward": "reward",
    "dead_group_rate": "frac_reward_zero_std",
    "replaced_rows": "reservoir/replaced_rows",
    "declined_rows": "reservoir/declined_rows",
    "ess_fraction": "reservoir/ess_fraction",
    "log_ratio_mean_abs": "reservoir/log_ratio_mean_abs",
    "staleness_mean": "reservoir/staleness_mean",
}
SUMMARY_FIELDS: tuple[str, ...] = (
    "train_reward_last_quarter", "eval_accuracy", "dead_group_rate", "replaced_rows", "declined_rows",
    "ess_fraction_mean", "log_ratio_mean_abs_mean", "wall_clock_seconds", "container_seconds", "dollars",
    "engine_stale_steps",
)
CONSISTENT_CONFIG_KEYS: tuple[str, ...] = ("max_steps", "reward", "num_generations", "per_device_train_batch_size",
                                           "gradient_accumulation_steps", "max_completion_length", "learning_rate")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    replay: bool
    max_policy_age: Optional[int]
    policy: str
    max_log_ratio: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


def arm_specs(policy_on: str = "gate", max_log_ratio: float = DEFAULT_MAX_LOG_RATIO,
              ages: Sequence[int] = AGES) -> list[ArmSpec]:
    """Plain GRPO, then Reservoir at each age with the policy off, then the same ages with it on."""
    plain = ArmSpec("grpo", False, None, "off", None)
    off = [ArmSpec(f"reservoir_age{a}_off", True, a, "off", None) for a in ages]
    on = [ArmSpec(f"reservoir_age{a}_on", True, a, policy_on, max_log_ratio if policy_on == "gate" else None)
          for a in ages]
    return [plain, *off, *on]


def arm_by_name(specs: Sequence[ArmSpec], name: str) -> ArmSpec:
    for spec in specs:
        if spec.name == name:
            return spec
    raise KeyError(f"no arm named {name!r}; choose from {[s.name for s in specs]}")


STALENESS_ESS_FLOOR = 0.5
STALENESS_MASS_CAP = 0.25
"""The ``_on`` arms under ``--policy-on staleness``: the ``conservative`` preset's ESS floor and group-mass cap,
without its age bound (the sweep's age axis is the buffer's ``max_policy_age``) and without the legacy gate."""
POLICY_KWARG_NAMES: tuple[str, ...] = ("staleness_policy", "policy")


def staleness_policy():
    """The ``StalenessPolicy`` the sweep's ``_on`` arms use: ESS floor and mass cap only."""
    from reservoir.integrations._trl_staleness import StalenessPolicy

    return StalenessPolicy(ess_floor=STALENESS_ESS_FLOOR, mass_cap=STALENESS_MASS_CAP)


def policy_kwarg_name(parameter_names: Sequence[str]) -> Optional[str]:
    """Which of ``ReservoirReplay.__init__``'s parameters takes a ``StalenessPolicy``, if any yet."""
    for name in POLICY_KWARG_NAMES:
        if name in parameter_names:
            return name
    return None


def replay_kwargs(spec: ArmSpec) -> dict:
    """The ``ReservoirReplay`` keyword arguments that turn the arm's policy on.

    ``gate`` is the legacy drift gate shipped in 0.6.0. ``staleness`` is
    S1's ``StalenessPolicy`` (planning/2026-10-06-staleness-controller.md
    D1), passed through whichever keyword the adapter accepts for it; until
    the adapter accepts one, the arm refuses to run rather than silently
    training with the policy off.
    """
    if spec.policy == "off":
        return {}
    if spec.policy == "gate":
        return {"max_log_ratio": spec.max_log_ratio, "max_declines_per_step": None}
    if spec.policy == "staleness":
        import inspect

        from reservoir.integrations.trl import ReservoirReplay

        name = policy_kwarg_name(list(inspect.signature(ReservoirReplay.__init__).parameters))
        if name is None:
            raise NotImplementedError(
                "ReservoirReplay does not accept a StalenessPolicy yet (none of "
                f"{POLICY_KWARG_NAMES} in its signature); update benchmarks.staleness.report.replay_kwargs "
                "when S1 wires the policy into the adapter, or run with --policy-on gate"
            )
        return {name: staleness_policy()}
    raise ValueError(f"unknown policy {spec.policy!r}; expected one of {POLICIES}")


def half_life_for(max_policy_age: int) -> int:
    """Priority half-life tied to the age bound: a row at the bound has decayed to a quarter."""
    return max(1, max_policy_age // 2)


# --- cost ----------------------------------------------------------------------


def estimate_cost(*, n_runs: int, max_steps: int, seconds_per_step: float, overhead_seconds: float,
                  price_per_hour: float) -> dict:
    per_run = max_steps * seconds_per_step + overhead_seconds
    hours = n_runs * per_run / 3600.0
    return {"runs": n_runs, "per_run_minutes": per_run / 60.0, "gpu_hours": hours, "dollars": hours * price_per_hour,
            "price_per_hour": price_per_hour, "seconds_per_step": seconds_per_step,
            "overhead_seconds": overhead_seconds}


def dollars_for(container_seconds: float, price_per_hour: float) -> float:
    """List-price dollars for the time inside ``run_arm``; a lower bound, image start is not counted."""
    return container_seconds / 3600.0 * price_per_hour


# --- per-run --------------------------------------------------------------------


def per_step_series(log_history: list[dict]) -> dict[str, list]:
    """Per-step series from TRL's ``log_history``; a metric an arm never logged is ``None`` at every step."""
    entries = [e for e in log_history if "step" in e and "reward" in e]
    series: dict[str, list] = {"step": [int(e["step"]) for e in entries]}
    for name, key in PER_STEP_KEYS.items():
        series[name] = [e.get(key) for e in entries]
    return series


def _mean(values: list) -> Optional[float]:
    present = [v for v in values if v is not None and math.isfinite(v)]
    return sum(present) / len(present) if present else None


def mean_std(values: list) -> dict:
    present = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not present:
        return {"mean": None, "std": None, "n": 0}
    mean = sum(present) / len(present)
    var = sum((v - mean) ** 2 for v in present) / (len(present) - 1) if len(present) > 1 else 0.0
    return {"mean": mean, "std": math.sqrt(var), "n": len(present)}


def summarize_run(record: dict) -> dict:
    series = per_step_series(record["log_history"])
    rewards = [r for r in series["reward"] if r is not None]
    tail = rewards[-max(1, len(rewards) // 4):] if rewards else []
    totals = record.get("totals")
    attestation = record.get("attestation")
    checker = (attestation or {}).get("checker") or {}
    engine = record.get("engine_check") or {}
    stale_steps = len(engine.get("stale_steps", [])) if engine.get("available") else None
    return {
        "seed": record["seed"],
        "steps_logged": len(rewards),
        "train_reward_last_quarter": _mean(tail),
        "train_reward_mean": _mean(rewards),
        "eval_accuracy": (record.get("eval") or {}).get("accuracy"),
        "dead_group_rate": _mean(series["dead_group_rate"]),
        "replaced_rows": int(totals.get("replaced_rows", 0)) if totals else None,
        "declined_rows": int(totals.get("declined_rows", 0)) if totals else None,
        "dead_groups": int(totals.get("dead_groups", 0)) if totals else None,
        "ess_fraction_mean": _mean(series["ess_fraction"]),
        "log_ratio_mean_abs_mean": _mean(series["log_ratio_mean_abs"]),
        "wall_clock_seconds": record.get("wall_clock_seconds"),
        "container_seconds": record.get("container_seconds"),
        "dollars": dollars_for(record["container_seconds"], record["price_per_hour"])
        if record.get("container_seconds") is not None and record.get("price_per_hour") is not None else None,
        "engine_stale_steps": stale_steps,
        "suspect": bool(stale_steps),
        "attestation_records": (attestation or {}).get("records"),
        "head_digest": (attestation or {}).get("head_digest"),
        "checker_returncode": checker.get("returncode") if attestation else None,
    }


# --- the report -------------------------------------------------------------------


def load_run_records(run_dir: Path) -> list[dict]:
    records = []
    for path in sorted(Path(run_dir).glob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "arm" in data and "log_history" in data:
            data["_path"] = str(path)
            records.append(data)
    return records


def reverify_log(record_path: Path, head_digest: str) -> int:
    """Run the independent checker on the log and manifest next to ``record_path``; return the record count.

    Raises if either file is missing, if the checker rejects the log, or if
    the log's last digest is not the ``head_digest`` the run reported.
    """
    from reservoir_checker.verify import verify_json_lines

    stem = Path(record_path).with_suffix("")
    attest, manifest = stem.with_name(stem.name + ".attest.jsonl"), stem.with_name(stem.name + ".manifest.jsonl")
    for path in (attest, manifest):
        if not path.exists():
            raise FileNotFoundError(f"{record_path} has no {path.name} next to it; the run cannot be re-verified")
    text = attest.read_text()
    verify_json_lines(text, manifest=manifest.read_text())
    lines = [line for line in text.splitlines() if line]
    last = json.loads(lines[-1])
    if last.get("digest") != head_digest:
        raise RuntimeError(f"{attest.name}: last record digest {last.get('digest')} is not the run's head {head_digest}")
    return len(lines)


def _check_run_verified(record: dict, reverify: bool) -> None:
    attestation = record.get("attestation")
    if attestation is None:
        raise RuntimeError(f"{record['_path']}: a Reservoir arm without an attestation log cannot be reported")
    code = (attestation.get("checker") or {}).get("returncode")
    if code != 0:
        raise RuntimeError(f"{record['_path']}: the checker rejected or never saw this log (returncode {code!r}); "
                           "the report must not include it")
    if reverify:
        n = reverify_log(Path(record["_path"]), attestation.get("head_digest"))
        if n != attestation.get("records"):
            raise RuntimeError(f"{record['_path']}: the log holds {n} records, the run reported {attestation.get('records')}")


def _check_consistent(records: list[dict]) -> None:
    configs = {tuple((k, r["config"].get(k)) for k in CONSISTENT_CONFIG_KEYS) + (("model", r.get("model")),)
               for r in records}
    if len(configs) > 1:
        raise RuntimeError(f"the records do not share one configuration (model, steps, batch, reward): {sorted(configs)}")
    seeds_by_arm = {}
    for r in records:
        seeds_by_arm.setdefault(r["arm"], set()).add(r["seed"])
    if len({frozenset(s) for s in seeds_by_arm.values()}) > 1:
        raise RuntimeError(f"the arms do not share one seed set: { {k: sorted(v) for k, v in seeds_by_arm.items()} }")


def build_report(run_dir: Path, policy_on: Optional[str] = None, max_log_ratio: Optional[float] = None,
                 reverify: bool = True) -> dict:
    """Aggregate every run record in ``run_dir`` by arm.

    Every Reservoir run must carry a checker verdict of 0 and, with
    ``reverify`` (the default), its log and manifest must sit next to the
    record and pass the checker again. Runs whose stale-engine check fired
    are listed as ``suspect`` and left out of the per-arm means. All records
    must share one configuration and one seed set per arm; arms that are
    absent are listed under ``missing_arms``.
    """
    records = load_run_records(run_dir)
    if not records:
        raise FileNotFoundError(f"no run records under {run_dir}")
    order = {s.name: i for i, s in enumerate(arm_specs())}
    records.sort(key=lambda r: (order.get(r["arm"], len(order)), r["seed"]))
    _check_consistent(records)
    for r in records:
        if r["replay"]:
            _check_run_verified(r, reverify)
    arms: dict[str, dict] = {}
    per_step: dict[str, dict] = {}
    for r in records:
        arm = arms.setdefault(r["arm"], {
            "replay": r["replay"], "policy": r["policy"], "max_policy_age": r["max_policy_age"],
            "max_log_ratio": r.get("max_log_ratio"), "runs": [],
        })
        arm["runs"].append({**summarize_run(r), "record": Path(r["_path"]).name})
        per_step.setdefault(r["arm"], {})[str(r["seed"])] = per_step_series(r["log_history"])
    for arm in arms.values():
        clean = [run for run in arm["runs"] if not run["suspect"]]
        arm["suspect_runs"] = [run["seed"] for run in arm["runs"] if run["suspect"]]
        arm["summary"] = {f: mean_std([run.get(f) for run in clean]) for f in SUMMARY_FIELDS}
    first = records[0]
    runs = [run for arm in arms.values() for run in arm["runs"]]
    return {
        "experiment": "staleness_sweep",
        "generated": _dt.date.today().isoformat(),
        "run_dir": str(run_dir),
        "model": first.get("model"), "gpu": first.get("gpu"), "versions": first.get("versions"),
        "max_steps": first["config"].get("max_steps"), "config": first.get("config"),
        "seeds": sorted({r["seed"] for r in records}),
        "policy_on": policy_on or next((r["policy"] for r in records if r["policy"] != "off"), None),
        "max_log_ratio": max_log_ratio if max_log_ratio is not None
        else next((r.get("max_log_ratio") for r in records if r.get("max_log_ratio") is not None), None),
        "price_per_hour": first.get("price_per_hour"),
        "arms": arms,
        "missing_arms": [name for name in order if name not in arms],
        "per_step": per_step,
        "verified_logs": sum(1 for r in records if r["replay"]),
        "reverified": reverify,
        "totals": {
            "runs": len(records),
            "suspect_runs": sum(1 for run in runs if run["suspect"]),
            "gpu_seconds": sum(run["container_seconds"] for run in runs if run["container_seconds"] is not None),
            "dollars": sum(run["dollars"] for run in runs if run["dollars"] is not None),
        },
    }


def _rolling(values: list, window: int) -> list:
    out = []
    for i in range(len(values)):
        chunk = [v for v in values[max(0, i - window + 1):i + 1] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def _mean_curve(seed_series: dict[str, dict], key: str, window: int) -> tuple[list, list]:
    """Rolling mean per seed, then the mean over seeds at every step any seed logged."""
    by_step: dict[int, list] = {}
    for series in seed_series.values():
        for step, value in zip(series["step"], _rolling(series[key], window)):
            by_step.setdefault(step, []).append(value)
    steps = sorted(by_step)
    return steps, [_mean(by_step[s]) for s in steps]


def plot_report(report: dict, path: Path, window: int = 10) -> None:
    """Two panels: reward curves per arm (seed mean, rolling window) and ESS fraction vs reward per step."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {8: "#1f77b4", 32: "#ff7f0e", 128: "#d62728", None: "#444444"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    for name, arm in report["arms"].items():
        steps, means = _mean_curve(report["per_step"][name], "reward", window)
        style = "--" if arm["policy"] != "off" else "-"
        left.plot(steps, means, style, color=colors.get(arm["max_policy_age"], "#444444"), label=name)
        if not arm["replay"]:
            continue
        xs, ys = [], []
        for series in report["per_step"][name].values():
            for e, r in zip(series["ess_fraction"], series["reward"]):
                if e is not None and r is not None:
                    xs.append(e)
                    ys.append(r)
        right.scatter(xs, ys, s=12, alpha=0.5, color=colors.get(arm["max_policy_age"]),
                      marker="o" if arm["policy"] != "off" else "x", label=name)
    left.set_xlabel("optimizer step")
    left.set_ylabel(f"train reward (rolling {window}, mean over seeds)")
    left.set_title(f"{report.get('model')}: reward by arm")
    left.legend(fontsize=7)
    right.set_xlabel("ESS fraction of replayed rows (exact, from the log)")
    right.set_ylabel("train reward at that step")
    right.set_title("ESS vs reward, replayed steps (x: policy off, o: policy on)")
    right.legend(fontsize=7)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def format_table(report: dict) -> str:
    """A fixed-width summary table, one row per arm (suspect runs excluded from the means, counted)."""
    header = (f"{'arm':<22}{'n':>3}{'susp':>5}{'eval acc':>10}{'reward q4':>11}{'dead rate':>11}{'replaced':>10}"
              f"{'declined':>10}{'ESS frac':>10}{'min/run':>9}{'$':>7}")
    lines = [header]
    for name, arm in report["arms"].items():
        s = arm["summary"]

        def f(key: str, fmt: str = "{:.3f}", scale: float = 1.0) -> str:
            v = s[key]["mean"]
            return "-" if v is None else fmt.format(v * scale)
        lines.append(
            f"{name:<22}{len(arm['runs']):>3}{len(arm['suspect_runs']):>5}{f('eval_accuracy'):>10}"
            f"{f('train_reward_last_quarter'):>11}{f('dead_group_rate'):>11}{f('replaced_rows', '{:.0f}'):>10}"
            f"{f('declined_rows', '{:.0f}'):>10}{f('ess_fraction_mean'):>10}"
            f"{f('container_seconds', '{:.1f}', 1 / 60):>9}{f('dollars', '{:.2f}'):>7}"
        )
    t = report["totals"]
    lines.append(f"total: {t['runs']} runs ({t['suspect_runs']} suspect), {t['gpu_seconds'] / 3600:.2f} GPU-hours, "
                 f"${t['dollars']:.2f}; verified logs {report['verified_logs']}"
                 f"{' (re-verified)' if report.get('reverified') else ''}; missing arms {report['missing_arms'] or 'none'}")
    return "\n".join(lines)
