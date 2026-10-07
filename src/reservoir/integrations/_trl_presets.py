"""GRPO replay presets and trainer attachment defaults.

Imported once at the end of ``trl.py`` so its public class can keep its
existing import path while the implementation lives outside that module.
"""

from __future__ import annotations

import sys
from pathlib import Path

from reservoir.decay import DecayParams, MAX_HALF_LIFE
from reservoir.integrations._trl_lifecycle import on_train_end

_UNSET = object()


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def for_grpo(cls, num_generations: int, max_steps: int,
             per_device_train_batch_size: int, profile: str = "conservative"):
    """Build a replay and return it with reasons for every derived setting."""
    g = _positive_int(num_generations, "num_generations")
    steps = _positive_int(max_steps, "max_steps")
    batch = _positive_int(per_device_train_batch_size, "per_device_train_batch_size")
    if profile not in ("conservative", "async", "off"):
        raise ValueError("profile must be 'conservative', 'async', or 'off'")

    # Keep roughly eight optimizer batches of rollouts, capped by the run.
    capacity = max(g, min(steps * batch, 8 * batch * g))
    half_life = min(MAX_HALF_LIFE, max(1, min(steps, 4 * g)))
    max_age = min(steps, 4 * half_life)
    # The uint64 tree has 31 bits after its default 32-bit priorities and
    # phase bit; reserve capacity bits before choosing the age window.
    capacity_bits = (capacity - 1).bit_length()
    max_age = min(max_age, max(0, 31 - capacity_bits) * half_life)
    DecayParams(half_life, max_age, capacity)

    gates = {
        "conservative": (5.0, 0),
        "async": (10.0, None),
        "off": (None, None),
    }
    max_log_ratio, max_declines = gates[profile]
    replay = cls(capacity=capacity, half_life=half_life,
                 max_policy_age=max_age, max_log_ratio=max_log_ratio,
                 max_declines_per_step=max_declines)
    explanation = {
        "capacity": f"{capacity} rows: eight batches of {batch} × {g} generations, capped by {steps} steps",
        "half_life": f"{half_life} optimizer steps: up to four generation groups, capped by run length",
        "max_policy_age": f"{max_age} optimizer steps: up to four half-lives, within the 64-bit tree budget",
        "max_log_ratio": f"{max_log_ratio}: {profile} drift gate; None disables it",
        "max_declines_per_step": f"{max_declines}: {profile} decline limit; None permits declines",
    }
    return replay, explanation


def _install() -> None:
    module = sys.modules["reservoir.integrations.trl"]
    replay_class = module.ReservoirReplay
    original_replay_init = replay_class.__init__
    original_attach = replay_class.attach
    original_mix_local = replay_class.mix_local
    original_build = module.build_trainer_class

    replay_class.for_grpo = classmethod(for_grpo)

    def replay_init(self, capacity=None, *args, attest=_UNSET, manifest=_UNSET, **kwargs):
        self._auto_capacity = capacity is None
        self._auto_logging = len(args) < 6 and attest is _UNSET and manifest is _UNSET
        if len(args) >= 6:
            original_replay_init(self, 1 if capacity is None else capacity, *args, **kwargs)
        else:
            original_replay_init(self, 1 if capacity is None else capacity, *args,
                                 attest=None if attest is _UNSET else attest,
                                 manifest=None if manifest is _UNSET else manifest, **kwargs)

    replay_class.__init__ = replay_init

    def attach(self, accelerator):
        output_dir = getattr(self, "_trainer_output_dir", None)
        schedule = getattr(self, "_trainer_schedule", None)
        if self._auto_capacity and schedule is not None:
            generated, steps, batch = schedule
            if steps < 1:  # TRL uses -1 when max_steps is not explicitly set.
                steps = 2000
            chosen, _ = for_grpo(replay_class, generated, steps, batch, "off")
            for key in ("capacity", "half_life", "max_policy_age"):
                self._buffer_kwargs[key] = chosen._buffer_kwargs[key]
            chosen.close()
            self._auto_capacity = False
            if self._buffer is not None:
                self._buffer.close()
                self._buffer = None
        if output_dir is not None and self._auto_logging:
            target = Path(output_dir) / "reservoir"
            if self.is_owner:
                target.mkdir(parents=True, exist_ok=True)
            self._buffer_kwargs["attest"] = target / "attest.jsonl"
            self._buffer_kwargs["manifest"] = target / "manifest.jsonl"
            self._auto_logging = False
            self._has_targets = True
            if self._buffer is not None:
                self._buffer.close()
                self._buffer = None
        return original_attach(self, accelerator)

    replay_class.attach = attach

    def mix_local(self, output, trainer):
        result = original_mix_local(self, output, trainer)
        self._generated_rows = getattr(self, "_generated_rows", 0) + output["advantages"].size(0)
        return result

    replay_class.mix_local = mix_local

    def build_trainer_class():
        trainer_class = original_build()
        if not getattr(trainer_class, "_developer_surface_installed", False):
            original_init = trainer_class.__init__

            def init(self, *args, replay_buffer, **kwargs):
                if not isinstance(replay_buffer, replay_class):
                    return original_init(self, *args, replay_buffer=replay_buffer, **kwargs)
                config = kwargs.get("args", args[2] if len(args) > 2 else None)
                replay_buffer._trainer_output_dir = getattr(config, "output_dir", None)
                if all(getattr(config, name, None) is not None for name in
                       ("num_generations", "max_steps", "per_device_train_batch_size")):
                    replay_buffer._trainer_schedule = (
                        config.num_generations, config.max_steps,
                        config.per_device_train_batch_size,
                    )
                original_init(self, *args, replay_buffer=replay_buffer, **kwargs)

            trainer_class.__init__ = init
            trainer_class.Callback.on_train_end = on_train_end
            trainer_class._developer_surface_installed = True
        return trainer_class

    module.build_trainer_class = build_trainer_class


_install()
