"""The sweep's arms and how each one is built.

Main grid (every seed, same prompt stream and configuration)::

    grpo                     stock GRPOTrainer
    grpo_filter              DAPO dynamic sampling: dead groups dropped, prompts oversampled until the batch is full
    wreinforce               fixed-reference advantage (correct +0.1, wrong -1), no buffer
    reservoir_age{8,32}_off  replay, max_policy_age 8 / 32, policy off
    reservoir_age{8,32}_on   the same with the policy on
    reservoir_naive_age32    replay with uniform priority and no age decay, policy off

Stress arm (first seed only): ``reservoir_age128_on``. Nothing in the
literature trains well at that age; it is kept to show where replay breaks,
not to compete.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

from reservoir.decay import MAX_HALF_LIFE
from reservoir.priorities import PriorityStrategy

AGES: tuple[int, ...] = (8, 32)
STRESS_AGE = 128
NAIVE_AGE = 32
POLICIES: tuple[str, ...] = ("off", "gate", "staleness")
TRAINERS: tuple[str, ...] = ("grpo", "grpo_filter", "wreinforce", "reservoir")
DEFAULT_MAX_LOG_RATIO = 2.0
"""Legacy drift gate of the ``--policy-on gate`` arms: a replayed row whose absolute log-ratio exceeds this is
declined. The TRL adapter now measures the log-ratio as the mean over completion tokens of current minus
stored logprob (it was the sum when this value was chosen), so 2.0 bounds the per-token average: e^2 is a
7.4x sequence-level ratio either way, and a gate this loose rarely fires at any age in the sweep."""

STALENESS_ESS_FLOOR = 0.5
STALENESS_MASS_CAP = 0.25
STALENESS_MAX_AGE = 4
"""The ``_on`` arms under ``--policy-on staleness``: the ``conservative`` preset's sampling-design ESS floor and
group-mass cap, plus an age bound below every buffer age so the policy binds in the CPU smoke. The buffer's
``max_policy_age`` remains the sweep's age axis; the legacy drift gate stays off."""


@dataclass(frozen=True)
class ArmSpec:
    name: str
    replay: bool
    max_policy_age: Optional[int]
    policy: str
    max_log_ratio: Optional[float]
    trainer: str = "reservoir"
    priority: str = "advantage"   # "advantage" (the adapter's default) or "uniform"
    stress: bool = False          # run on the first seed only, outside the main grid

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class UniformPriority(PriorityStrategy):
    """Every rollout scores 1: replay draws uniformly among live rows (the naive arm)."""

    def score(self, rollout, group) -> float:
        return 1.0


def _replay_arm(name: str, age: int, policy: str, max_log_ratio: float, **extra) -> ArmSpec:
    gate = max_log_ratio if policy == "gate" else None
    return ArmSpec(name, True, age, policy, gate, **extra)


def arm_specs(policy_on: str = "staleness", max_log_ratio: float = DEFAULT_MAX_LOG_RATIO,
              ages: Sequence[int] = AGES) -> list[ArmSpec]:
    """The main grid in report order, then the stress arm."""
    controls = [ArmSpec(name, False, None, "off", None, trainer=name) for name in ("grpo", "grpo_filter", "wreinforce")]
    off = [_replay_arm(f"reservoir_age{a}_off", a, "off", max_log_ratio) for a in ages]
    on = [_replay_arm(f"reservoir_age{a}_on", a, policy_on, max_log_ratio) for a in ages]
    naive = _replay_arm(f"reservoir_naive_age{NAIVE_AGE}", NAIVE_AGE, "off", max_log_ratio, priority="uniform")
    stress = _replay_arm(f"reservoir_age{STRESS_AGE}_on", STRESS_AGE, policy_on, max_log_ratio, stress=True)
    return [*controls, *off, *on, naive, stress]


def arm_by_name(specs: Sequence[ArmSpec], name: str) -> ArmSpec:
    for spec in specs:
        if spec.name == name:
            return spec
    raise KeyError(f"no arm named {name!r}; choose from {[s.name for s in specs]}")


def _parse_list(text: str, cast=str) -> list:
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def plan_jobs(arms: str, seeds: str, policy_on: str, max_log_ratio: float) -> list[tuple[ArmSpec, int]]:
    """``(arm, seed)`` jobs in report order; a stress arm runs on the first seed only."""
    specs = arm_specs(policy_on=policy_on, max_log_ratio=max_log_ratio)
    chosen = specs if arms == "all" else [arm_by_name(specs, name) for name in _parse_list(arms)]
    seed_list = _parse_list(seeds, int)
    return [(spec, seed) for spec in chosen for seed in (seed_list[:1] if spec.stress else seed_list)]


def staleness_policy():
    """The ``StalenessPolicy`` the sweep's ``_on`` arms use."""
    from reservoir.integrations._trl_staleness import StalenessPolicy

    return StalenessPolicy(max_age=STALENESS_MAX_AGE, ess_floor=STALENESS_ESS_FLOOR,
                           mass_cap=STALENESS_MASS_CAP)


def replay_kwargs(spec: ArmSpec) -> dict:
    """The ``ReservoirReplay`` keyword arguments that turn the arm's policy on.

    ``gate`` is the legacy drift gate shipped in 0.6.0; ``staleness`` passes
    ``staleness_policy()`` as ``staleness_policy``.
    """
    if spec.policy == "off":
        return {}
    if spec.policy == "gate":
        return {"max_log_ratio": spec.max_log_ratio, "max_declines_per_step": None}
    if spec.policy == "staleness":
        return {"staleness_policy": staleness_policy()}
    raise ValueError(f"unknown policy {spec.policy!r}; expected one of {POLICIES}")


def priority_kwargs(spec: ArmSpec) -> dict:
    """``priority=UniformPriority()`` for the naive arm, nothing (the adapter's default) otherwise."""
    if spec.priority == "uniform":
        return {"priority": UniformPriority()}
    if spec.priority == "advantage":
        return {}
    raise ValueError(f"unknown priority {spec.priority!r}; expected 'advantage' or 'uniform'")


def half_life_for(max_policy_age: int) -> int:
    """Priority half-life tied to the age bound: a row at the bound has decayed to a quarter."""
    return max(1, max_policy_age // 2)


def half_life_for_spec(spec: ArmSpec) -> int:
    """The arm's half-life; the naive arm takes the library's maximum, 1024 steps.

    The buffer has no exact "no decay" setting: at half-life 1024 a row at
    age 32 keeps 2^(-32/1024), about 97.9%, of its weight.
    """
    if spec.priority == "uniform":
        return MAX_HALF_LIFE
    return half_life_for(spec.max_policy_age)
