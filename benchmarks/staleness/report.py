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
from pathlib import Path
from typing import Any, Optional

from benchmarks.staleness.stats import (
    at_rollout_budget,
    bootstrap_ci,
    paired_bootstrap_diff,
    reward_auc,
    steps_to_target,
)
from benchmarks.staleness.arms import (  # re-exported: the report and the driver read the arms from here
    DEFAULT_MAX_LOG_RATIO,
    POLICIES,
    STALENESS_ESS_FLOOR,
    STALENESS_MASS_CAP,
    ArmSpec,
    arm_by_name,
    arm_specs,
    half_life_for,
    half_life_for_spec,
    plan_jobs,
    priority_kwargs,
    replay_kwargs,
    staleness_policy,
)

PER_STEP_KEYS: dict[str, str] = {
    "reward": "reward",
    "dead_group_rate": "frac_reward_zero_std",
    "replaced_rows": "reservoir/replaced_rows",
    "declined_rows": "reservoir/declined_rows",
    "ess_fraction": "reservoir/sampling_design_ess_fraction",
    "log_ratio_mean_abs": "reservoir/log_ratio_mean_abs",
    "staleness_mean": "reservoir/staleness_mean",
    "sampling_design_ess": "reservoir/sampling_design_ess",
    "rollouts": "sweep/rollouts_generated",
    "dead_all_wrong": "sweep/dead_all_wrong_rate",
    "dead_all_correct": "sweep/dead_all_correct_rate",
    "replayed_row_fraction": "sweep/replayed_row_fraction",
    "same_step_fraction": "sweep/same_step_fraction",
    "replayed_age_mean": "sweep/replayed_age_mean",
    "replayed_age_p90": "sweep/replayed_age_p90",
    "replayed_age_max": "sweep/replayed_age_max",
    "clip_fresh_pos": "sweep/clip_fraction/fresh_pos",
    "clip_fresh_neg": "sweep/clip_fraction/fresh_neg",
    "clip_replayed_pos": "sweep/clip_fraction/replayed_pos",
    "clip_replayed_neg": "sweep/clip_fraction/replayed_neg",
    "entropy": "entropy",
    "completion_length": "completions/mean_length",
    "trained_completion_length": "sweep/trained_completion_length",
    "kl_to_init": "sweep/kl_to_init",
    "lm_head_grad_norm": "sweep/lm_head_grad_norm",
    "lm_head_grad_z": "sweep/lm_head_grad_z",
    "filter_rounds": "sweep/filter_rounds",
    "filter_unfilled_groups": "sweep/filter_unfilled_groups",
    "engine_confident_fraction": "sweep/engine_confident_fraction",
    "engine_mean_abs_diff": "sweep/engine_mean_abs_diff",
    "reward_all_rounds": "reward",
}
STALE_FRACTION = 0.01
"""A step is stale when more than this fraction of its completion tokens disagree confidently (``engine_probe``)."""
FIRST_ROUND_REWARD = "sweep/first_round_reward"
"""Preferred over TRL's ``reward`` for the ``reward`` series when logged: the mean over the step's first generation
round, the prompts every arm sees (``grpo_filter``'s TRL reward also averages its extra rounds)."""
LEGACY_PER_STEP_KEYS: dict[str, str] = {"reservoir/sampling_design_ess_fraction": "reservoir/ess_fraction"}
"""Metric names a log written before the sampling-design rename used; read when the new name is absent."""
SUMMARY_FIELDS: tuple[str, ...] = (
    "train_reward_last_quarter", "eval_accuracy", "math500_accuracy", "avg_at_k_last", "reward_auc",
    "steps_to_95pct_grpo", "dead_group_rate", "dead_all_wrong_rate", "dead_all_correct_rate",
    "replayed_row_fraction_mean", "replaced_rows", "declined_rows", "ess_fraction_mean", "log_ratio_mean_abs_mean",
    "replayed_age_p90_max", "kl_to_init_last", "rollouts_total", "steps_at_rollout_budget",
    "train_reward_last_quarter_at_rollout_budget", "reward_auc_at_rollout_budget", "avg_at_k_at_rollout_budget",
    "avg_at_k_step_at_rollout_budget", "steps_to_95pct_grpo_censored", "filter_unfilled_groups_total",
    "wall_clock_seconds", "container_seconds", "dollars", "engine_stale_steps", "peak_memory_gb",
)
GRPO_FILTER_STEP_FACTOR = 1.35
"""Prior step-time multiplier for ``grpo_filter``: generation is about half a step and dynamic sampling needs
about 1 / (1 - dead rate) generation rounds, ~1.7 at the 40% dead rate expected for Qwen2.5-0.5B on GSM8K."""
CONSISTENT_CONFIG_KEYS: tuple[str, ...] = (
    "max_steps", "reward", "num_generations", "per_device_train_batch_size", "gradient_accumulation_steps",
    "max_completion_length", "learning_rate", "bf16", "fp16", "vllm", "eval_size", "periodic_eval_size",
    "math500_size", "eval_every", "eval_samples", "train_size", "loss_type", "scale_rewards", "beta", "temperature",
    "math_max_new_tokens", "prompts_per_step",
)
PER_ARM_KEYS: tuple[str, ...] = ("policy", "max_log_ratio", "trainer", "priority", "max_policy_age", "stress")
PER_ARM_CONFIG_KEYS: tuple[str, ...] = ("capacity", "half_life", "beta_is")


# --- cost ----------------------------------------------------------------------


def estimate_cost(*, n_runs: int, max_steps: int, seconds_per_step: float, overhead_seconds: float,
                  price_per_hour: float) -> dict:
    per_run = max_steps * seconds_per_step + overhead_seconds
    hours = n_runs * per_run / 3600.0
    return {"runs": n_runs, "per_run_minutes": per_run / 60.0, "gpu_hours": hours, "dollars": hours * price_per_hour,
            "price_per_hour": price_per_hour, "seconds_per_step": seconds_per_step,
            "overhead_seconds": overhead_seconds}


def estimate_jobs_cost(arms: list, *, max_steps: int, seconds_per_step: float, overhead_seconds: float,
                       price_per_hour: float, eval_seconds: float) -> dict:
    """Cost of one run per entry of ``arms``: steps (``grpo_filter`` at ``GRPO_FILTER_STEP_FACTOR``), overhead, evals."""
    per_run = [max_steps * seconds_per_step * (GRPO_FILTER_STEP_FACTOR if a.trainer == "grpo_filter" else 1.0)
               + overhead_seconds + eval_seconds for a in arms]
    hours = sum(per_run) / 3600.0
    return {"runs": len(per_run), "per_run_minutes": sum(per_run) / len(per_run) / 60.0 if per_run else 0.0,
            "gpu_hours": hours, "dollars": hours * price_per_hour, "price_per_hour": price_per_hour,
            "seconds_per_step": seconds_per_step, "overhead_seconds": overhead_seconds, "eval_seconds": eval_seconds,
            "grpo_filter_step_factor": GRPO_FILTER_STEP_FACTOR}


def dollars_for(container_seconds: float, price_per_hour: float) -> float:
    """List-price dollars for the time inside ``run_arm``; a lower bound, image start is not counted."""
    return container_seconds / 3600.0 * price_per_hour


# --- per-run --------------------------------------------------------------------


def per_step_series(log_history: list[dict]) -> dict[str, list]:
    """Per-step series from TRL's ``log_history``; a metric an arm never logged is ``None`` at every step."""
    entries = [e for e in log_history if "step" in e and "reward" in e]
    series: dict[str, list] = {"step": [int(e["step"]) for e in entries]}
    for name, key in PER_STEP_KEYS.items():
        legacy = LEGACY_PER_STEP_KEYS.get(key)
        series[name] = [e.get(key, e.get(legacy) if legacy else None) for e in entries]
    series["reward"] = [e.get(FIRST_ROUND_REWARD, e["reward"]) for e in entries]
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
    control = [v for v in series["engine_confident_fraction"] if v is not None]
    if stale_steps is None and control:     # a control arm probed in its loss forward
        stale_steps = sum(1 for v in control if v > STALE_FRACTION)
    reasons = (["probe failed: " + engine["failure"]] if engine.get("failure") else []) + \
        ([f"{stale_steps} stale engine steps"] if stale_steps else [])
    if (record.get("config") or {}).get("vllm") and stale_steps is None and not engine.get("failure"):
        reasons.append("vLLM run was never engine-probed")
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
        "suspect": bool(reasons),
        "suspect_reasons": reasons,
        "eval_errors": eval_errors(record),
        "attestation_records": (attestation or {}).get("records"),
        "head_digest": (attestation or {}).get("head_digest"),
        "checker_returncode": checker.get("returncode") if attestation else None,
        **_diagnostic_summary(record, series),
    }


def eval_errors(record: dict) -> list[str]:
    """``"<evaluation>: <error>"`` for every evaluation of the run that raised (kept, not counted as suspect)."""
    evals = record.get("evals") or {}
    out = [f"{name}: {entry['error']}" for name, entry in evals.items()
           if isinstance(entry, dict) and entry.get("error")]
    out += [f"gsm8k_avg_at_k step {e.get('step')}: {e['error']}" for e in evals.get("gsm8k_avg_at_k") or []
            if e.get("error")]
    return out


def _last(values: list) -> Optional[float]:
    present = [v for v in values if v is not None]
    return present[-1] if present else None


def _diagnostic_summary(record: dict, series: dict) -> dict:
    """Per-run means of the sweep diagnostics and the evaluations (``None`` where a record predates them)."""
    evals = record.get("evals") or {}
    periodic = [e for e in evals.get("gsm8k_avg_at_k") or [] if "accuracy" in e]
    rollouts = [v for v in series["rollouts"] if v is not None]
    ages = [v for v in series["replayed_age_p90"] if v is not None]
    return {
        "math500_accuracy": (evals.get("math500") or {}).get("accuracy"),
        "avg_at_k_last": periodic[-1]["accuracy"] if periodic else None,
        "reward_auc": reward_auc(series["reward"]),
        "dead_all_wrong_rate": _mean(series["dead_all_wrong"]),
        "dead_all_correct_rate": _mean(series["dead_all_correct"]),
        "replayed_row_fraction_mean": _mean(series["replayed_row_fraction"]),
        "replayed_age_p90_max": max(ages) if ages else None,
        "kl_to_init_last": _last(series["kl_to_init"]),
        "rollouts_total": sum(rollouts) if rollouts else None,
        "filter_unfilled_groups_total": sum(v for v in series["filter_unfilled_groups"] if v is not None)
        if record.get("trainer") == "grpo_filter" else None,
        "peak_memory_gb": record.get("peak_memory_gb"),
    }


# --- the report -------------------------------------------------------------------


def load_run_records(run_dir: Path) -> list[dict]:
    records = []
    for path in sorted(Path(run_dir).glob("*.json")):
        if path.name.endswith(".failed.json"):      # a record the driver could not write; never reported
            continue
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
    floors = {json.loads(line).get("min_entry_age") for line in lines if '"op":"sample"' in line.replace(" ", "")}
    if floors and floors != {"1"}:
        raise RuntimeError(f"{attest.name}: sample records declare min_entry_age {sorted(map(str, floors))}, "
                           "not 1 on every draw; same-step replay is not excluded")
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
    """One configuration for every record, one setting per arm, one seed set; stress arms on a subset of it."""
    configs = {json.dumps([[k, r["config"].get(k)] for k in CONSISTENT_CONFIG_KEYS]
                          + [["model", r.get("model")], ["versions", r.get("versions")]], sort_keys=True)
               for r in records}
    if len(configs) > 1:
        raise RuntimeError(f"the records do not share one configuration (model, steps, batch, reward, precision, "
                           f"evaluation, versions): {sorted(configs)}")
    per_arm: dict[str, set] = {}
    for r in records:
        per_arm.setdefault(r["arm"], set()).add(json.dumps(
            [r.get(k) for k in PER_ARM_KEYS] + [r["config"].get(k) for k in PER_ARM_CONFIG_KEYS], sort_keys=True))
    mixed = sorted(arm for arm, settings in per_arm.items() if len(settings) > 1)
    if mixed:
        raise RuntimeError(f"arms {mixed} mix settings across seeds (policy, gate, trainer, priority, buffer)")
    main_seeds: dict[str, set] = {}
    stress_seeds: dict[str, set] = {}
    for r in records:
        (stress_seeds if r.get("stress") else main_seeds).setdefault(r["arm"], set()).add(r["seed"])
    if len({frozenset(s) for s in main_seeds.values()}) > 1:
        raise RuntimeError(f"the arms do not share one seed set: { {k: sorted(v) for k, v in main_seeds.items()} }")
    allowed = next(iter(main_seeds.values()), set())
    for arm, seeds in stress_seeds.items():
        if main_seeds and not seeds <= allowed:
            raise RuntimeError(f"stress arm {arm} ran seeds {sorted(seeds)} outside the main seed set {sorted(allowed)}")


def _check_run_usable(record: dict) -> None:
    """A run that replayed a same-step row is not reported (a failed or stale probe only makes it ``suspect``)."""
    same = [v for v in per_step_series(record["log_history"])["same_step_fraction"] if v]
    if same:
        raise RuntimeError(f"{record['_path']}: replayed rows from their own step (fractions {same}); "
                           "min_entry_age did not hold")


def _items_by_seed(records: list[dict], key: str) -> dict:
    return {r["seed"]: ((r.get("evals") or {}).get(key) or {}).get("per_item") for r in records
            if not summarize_run(r)["suspect"] and ((r.get("evals") or {}).get(key) or {}).get("per_item")}


def _cross_run(records: list[dict], arms: dict, per_step: dict) -> dict:
    """Bootstrap CIs, the difference from ``grpo``, steps to 95% of its reward and the rollout-matched cut.

    Fills each run summary with its steps-to-target and budget cut, and each
    arm with ``eval_ci``, ``math500_ci`` and ``eval_diff_vs_grpo``; returns
    the report-level target and budget.
    """
    by_arm: dict[str, list] = {}
    for r in records:
        by_arm.setdefault(r["arm"], []).append(r)
    base = _items_by_seed(by_arm.get("grpo", []), "gsm8k_test")
    grpo_tail = mean_std([run["train_reward_last_quarter"] for run in arms.get("grpo", {}).get("runs", [])
                          if not run["suspect"]])["mean"]
    target = 0.95 * grpo_tail if grpo_tail is not None else None
    budget_runs = [run["rollouts_total"] for run in arms.get("grpo", {}).get("runs", [])
                   if run["rollouts_total"] and not run["suspect"]]
    budget = sorted(budget_runs)[len(budget_runs) // 2] if budget_runs else None
    for name, arm in arms.items():
        items = _items_by_seed(by_arm[name], "gsm8k_test")
        math_items = _items_by_seed(by_arm[name], "math500")
        arm["eval_ci"] = bootstrap_ci(items) if items else None
        arm["math500_ci"] = bootstrap_ci(math_items) if math_items else None
        arm["eval_diff_vs_grpo"] = paired_bootstrap_diff(items, base) if items and base and name != "grpo" else None
        for run, record in zip(arm["runs"], by_arm[name]):
            series = per_step[name][str(record["seed"])]
            run["steps_to_95pct_grpo"] = (steps_to_target(series["step"], series["reward"], target)
                                          if target is not None else None)
            # censored at the run length when never reached, so slow seeds are not dropped from the mean
            run["steps_to_95pct_grpo_censored"] = (run["steps_to_95pct_grpo"] or record["config"].get("max_steps")
                                                   if target is not None else None)
            cut = at_rollout_budget(series, (record.get("evals") or {}).get("gsm8k_avg_at_k") or [], budget) \
                if budget else None
            run["rollout_budget_cut"] = cut
            for field, key in (("steps_at_rollout_budget", "step"),
                               ("train_reward_last_quarter_at_rollout_budget", "train_reward_last_quarter"),
                               ("reward_auc_at_rollout_budget", "reward_auc"), ("avg_at_k_at_rollout_budget", "avg_at_k"),
                               ("avg_at_k_step_at_rollout_budget", "avg_at_k_step")):
                run[field] = cut[key] if cut else None
        arm["reached_95pct_grpo"] = sum(1 for run in arm["runs"] if run["steps_to_95pct_grpo"] is not None)
    return {"steps_to_95pct_grpo_target": target, "rollout_budget": budget}


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
        _check_run_usable(r)
        if r["replay"]:
            _check_run_verified(r, reverify)
    arms: dict[str, dict] = {}
    per_step: dict[str, dict] = {}
    for r in records:
        arm = arms.setdefault(r["arm"], {
            "replay": r["replay"], "policy": r["policy"], "max_policy_age": r["max_policy_age"],
            "max_log_ratio": r.get("max_log_ratio"), "trainer": r.get("trainer"), "priority": r.get("priority"),
            "stress": bool(r.get("stress")), "runs": [],
        })
        arm["runs"].append({**summarize_run(r), "record": Path(r["_path"]).name})
        per_step.setdefault(r["arm"], {})[str(r["seed"])] = per_step_series(r["log_history"])
    cross = _cross_run(records, arms, per_step)
    for arm in arms.values():
        clean = [run for run in arm["runs"] if not run["suspect"]]
        arm["suspect_runs"] = [run["seed"] for run in arm["runs"] if run["suspect"]]
        arm["eval_error_runs"] = [run["seed"] for run in arm["runs"] if run["eval_errors"]]
        arm["summary"] = {f: mean_std([run.get(f) for run in clean]) for f in SUMMARY_FIELDS}
    first = records[0]
    runs = [run for arm in arms.values() for run in arm["runs"]]
    from benchmarks.provenance import gpu_of, stamped

    return stamped({
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
        **cross,
        "per_step": per_step,
        "verified_logs": sum(1 for r in records if r["replay"]),
        "reverified": reverify,
        "totals": {
            "runs": len(records),
            "suspect_runs": sum(1 for run in runs if run["suspect"]),
            "gpu_seconds": sum(run["container_seconds"] for run in runs if run["container_seconds"] is not None),
            "dollars": sum(run["dollars"] for run in runs if run["dollars"] is not None),
        },
    }, gpu=gpu_of(first))


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
    """A fixed-width summary table, one row per arm (suspect runs excluded from the means, counted).

    ``reach`` counts the runs whose rolling train reward reached 95% of
    ``grpo``'s final reward; ``eval CI`` is the seeds × items bootstrap CI of
    the final GSM8K accuracy.
    """
    header = (f"{'arm':<24}{'n':>3}{'susp':>5}{'eval acc':>10}{'eval CI':>15}{'reward q4':>11}{'AUC':>7}{'reach':>7}"
              f"{'dead rate':>11}{'replaced':>10}{'declined':>10}{'ESS frac':>10}{'min/run':>9}{'$':>7}")
    lines = [header]
    for name, arm in report["arms"].items():
        s = arm["summary"]

        def f(key: str, fmt: str = "{:.3f}", scale: float = 1.0) -> str:
            v = s[key]["mean"]
            return "-" if v is None else fmt.format(v * scale)
        ci = arm.get("eval_ci")
        ci_text = "-" if not ci else f"{ci['lo']:.3f}-{ci['hi']:.3f}"
        reach = f"{arm.get('reached_95pct_grpo', 0)}/{len(arm['runs'])}"
        lines.append(
            f"{name:<24}{len(arm['runs']):>3}{len(arm['suspect_runs']):>5}{f('eval_accuracy'):>10}{ci_text:>15}"
            f"{f('train_reward_last_quarter'):>11}{f('reward_auc'):>7}{reach:>7}{f('dead_group_rate'):>11}"
            f"{f('replaced_rows', '{:.0f}'):>10}{f('declined_rows', '{:.0f}'):>10}{f('ess_fraction_mean'):>10}"
            f"{f('container_seconds', '{:.1f}', 1 / 60):>9}{f('dollars', '{:.2f}'):>7}"
        )
    t = report["totals"]
    for name, arm in report["arms"].items():
        for run in arm["runs"]:
            for error in run.get("eval_errors") or []:
                lines.append(f"evaluation failed: {name} seed {run['seed']}: {error}")
    lines.append(f"total: {t['runs']} runs ({t['suspect_runs']} suspect), {t['gpu_seconds'] / 3600:.2f} GPU-hours, "
                 f"${t['dollars']:.2f}; verified logs {report['verified_logs']}"
                 f"{' (re-verified)' if report.get('reverified') else ''}; missing arms {report['missing_arms'] or 'none'}")
    return "\n".join(lines)
