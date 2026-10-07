"""A stale-engine check for TRL's colocated vLLM, at no extra forward.

Martingale's first real colocate run (its ``benchmarks/modal/results``,
2026-10-06) caught TRL 1.13's in-process vLLM serving the initial weights
for some generation steps while the trainer kept training. A sweep over
staleness cannot trust its axis if the engine itself is silently stale,
so every Reservoir arm compares, at every generation step, the logprobs
vLLM reported while sampling (``sampling_per_token_logps``, which TRL
attaches to every vLLM batch) with the trainer's own forward over the same
tokens. ``ReservoirReplay`` already runs that forward to obtain behavior
logprobs, so the comparison costs nothing.

The rule is Martingale's "confident disagreement": a token the engine gave
at least probability 0.5 and the trainer scores more than two nats lower.
bf16-vs-fp32 numerics cannot produce that; a weight mismatch does. A step
is ``stale`` when more than 1% of its completion tokens disagree that way.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from reservoir.integrations.trl import ReservoirReplay

CONFIDENT_LOGPROB = math.log(0.5)
GAP_NATS = 2.0
STALE_FRACTION = 0.01


def probe_stats(sampling: Any, trainer: Any, mask: Any, *, confident: float = CONFIDENT_LOGPROB,
                gap: float = GAP_NATS, stale_fraction: float = STALE_FRACTION) -> dict:
    """Per-step comparison of engine and trainer logprobs over the completion mask."""
    import torch

    sampling = sampling.detach().to("cpu", torch.float64)
    trainer = trainer.detach().to("cpu", torch.float64)
    valid = mask.detach().cpu().bool() & torch.isfinite(sampling) & torch.isfinite(trainer)
    tokens = int(valid.sum())
    if tokens == 0:
        return {"tokens": 0, "mean_abs_diff": None, "max_abs_diff": None, "confident_disagreements": 0,
                "confident_fraction": None, "stale": False}
    diff = (trainer - sampling)[valid]
    disagree = int(((sampling[valid] >= confident) & (diff < -gap)).sum())
    fraction = disagree / tokens
    return {
        "tokens": tokens,
        "mean_abs_diff": float(diff.abs().mean()),
        "max_abs_diff": float(diff.abs().max()),
        "confident_disagreements": disagree,
        "confident_fraction": fraction,
        "stale": fraction > stale_fraction,
    }


class ProbedReplay(ReservoirReplay):
    """``ReservoirReplay`` that records the engine check at every hook call.

    Only the single-process path is probed (the sweep runs one process).
    With HF generation there are no sampling logprobs and ``engine_check``
    reports ``available: False``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.engine_steps: list[dict] = []
        self._pending_sampling: Optional[Any] = None
        self._pending_step: Optional[int] = None

    def mix(self, output: dict, trainer: Any) -> dict:
        self._pending_sampling = output.get("sampling_per_token_logps")
        self._pending_step = int(trainer.state.global_step)
        if self._pending_sampling is not None and output.get("old_per_token_logps") is not None:
            self._record(output["old_per_token_logps"], output["completion_mask"])
        return super().mix(output, trainer)

    def behavior_logprobs(self, output: dict, trainer: Any) -> Any:
        logprobs = super().behavior_logprobs(output, trainer)
        if self._pending_sampling is not None:
            self._record(logprobs, output["completion_mask"])
        return logprobs

    def _record(self, trainer_logprobs: Any, mask: Any) -> None:
        sampling, self._pending_sampling = self._pending_sampling, None
        if tuple(sampling.shape) != tuple(trainer_logprobs.shape):
            self.engine_steps.append({"step": self._pending_step, "tokens": 0, "stale": False,
                                      "skipped": f"shape {tuple(sampling.shape)} vs {tuple(trainer_logprobs.shape)}"})
            return
        self.engine_steps.append({"step": self._pending_step, **probe_stats(sampling, trainer_logprobs, mask)})

    def engine_check(self) -> dict:
        return {
            "available": bool(self.engine_steps),
            "rule": {"confident_logprob": CONFIDENT_LOGPROB, "gap_nats": GAP_NATS, "stale_fraction": STALE_FRACTION},
            "steps": list(self.engine_steps),
            "stale_steps": [s["step"] for s in self.engine_steps if s.get("stale")],
            "skipped_steps": [s["step"] for s in self.engine_steps if "skipped" in s],
        }


def assert_probe_complete(check: dict, hook_calls: int) -> None:
    """Raise unless the engine was probed at every hook call; a vLLM run must never pass unprobed."""
    if not check.get("available"):
        raise RuntimeError("the stale-engine probe saw no sampling logprobs: TRL did not attach "
                           "sampling_per_token_logps, so the engine cannot be checked and the run is not usable")
    if check.get("skipped_steps"):
        raise RuntimeError(f"the stale-engine probe skipped steps {check['skipped_steps']} (shape mismatch)")
    if len(check["steps"]) != hook_calls:
        raise RuntimeError(f"the stale-engine probe ran {len(check['steps'])} times for {hook_calls} hook calls")
