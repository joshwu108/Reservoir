"""reservoir: Certified Exact Prioritized Experience Replay
with Crash Atomicity and Sampling Attestation.

Package layout
--------------
Two product surfaces share one exact core.

LLM-RL rollout replay (the newer surface; README quick start):
    rollout.py          Rollout / RolloutGroup value types. Validation lives here.
    priorities.py       Pluggable scoring: a float score per rollout or per prompt.
    decay.py            Pure integer math for age-decayed priorities (design.md §7).
    decayed_tree.py     Sum-tree + min-tree holding (q, version) per leaf; evicts
                        stale entries and rebases. The only mutable tree state.
    rollout_buffer.py   RolloutBuffer: add_group / sample / update_priorities on top
                        of the tree. Keyed draws, IS weights, slot allocation.
    rollout_attest.py   Turns buffer events into attestation records; every insert
                        carries the example's content digest and source tag.
    rollout_manifest.py The manifest: the opening of every content digest, one
                        JSON line per insert, for the checker to recompute.
    rollout_snapshot.py state_dict / load_state_dict serialisation and validation.
    durable_rollout.py  Crash-atomic wrapper around RolloutBuffer.
    dataset_buffer.py   Which prompts to generate rollouts for next (float, numpy).

Classic transition replay (the original surface):
    buffer.py           ExactPERBuffer: big-integer PER over (s, a, r, s', done).
    fast_buffer.py / c_buffer.py   Float PER for real training; C-backed when built.
    sumtree.py          ExactSumTree / ExactMinTree used by both exact buffers.
    rational.py         float -> exact integer priority (the alpha boundary).
    durable.py          The intent/segment/commit protocol and DurableBuffer.
    nstep.py, her.py, gym_wrapper.py   Wrappers over the fast buffer.

Shared:
    draw.py             Keyed BLAKE2b uniform draws; no RNG anywhere else.
    attest.py           Hash-chained attestation log records.
    checker/            Independent verifier (verify), audit report (transcript),
                        two-log comparison (diff). Imports nothing from the rest
                        of this package; ships with console scripts.

Fine-tuning tools (separate from replay): prefcheck.py, trajectory.py,
report.py (preference-noise detection); anchor_set.py, forgetting_monitor.py,
replay_scheduler.py (forgetting measurement and replay).

See ``rollout_buffer.py`` ("Call structure"), ``decayed_tree.py`` and
``decay.py`` with docs/design.md §7 for the replay path. ``tests/`` mirrors
this layout, one file per module.
"""

__version__ = "0.6.0"

# Importing the package needs numpy only. The rollout replay surface is
# eager; the classic transition buffers, their wrappers and the
# fine-tuning tools need torch and are resolved on first attribute access
# (PEP 562), with an ImportError naming the extra to install when it is
# missing.
from reservoir.buffer import ExactPERBuffer
from reservoir.rollout import Rollout, RolloutGroup
from reservoir.rollout_buffer import RolloutBatch, RolloutBuffer
from reservoir.durable_rollout import DurableRolloutBuffer

# name -> (module, attribute, extra that provides its dependencies)
_LAZY: dict[str, tuple[str, str, str]] = {
    "FastPERBuffer": ("reservoir._classic", "FastPERBuffer", "classic"),
    "PyFastPERBuffer": ("reservoir.fast_buffer", "FastPERBuffer", "classic"),
    "backend": ("reservoir._classic", "backend", "classic"),
    "DatasetBuffer": ("reservoir.dataset_buffer", "DatasetBuffer", "classic"),
    "AnchorSet": ("reservoir.anchor_set", "AnchorSet", "anchor"),
    "ForgettingMonitor": ("reservoir.forgetting_monitor", "ForgettingMonitor", "anchor"),
    "ForgettingAlert": ("reservoir.forgetting_monitor", "ForgettingAlert", "anchor"),
    "ReplayScheduler": ("reservoir.replay_scheduler", "ReplayScheduler", "anchor"),
    "PreferenceQualityReport": ("reservoir.report", "PreferenceQualityReport", "prefcheck"),
    "NoiseLabel": ("reservoir.report", "NoiseLabel", "prefcheck"),
    "PreferenceNoiseDetector": ("reservoir.prefcheck", "PreferenceNoiseDetector", "prefcheck"),
}


_OPTIONAL_DEPENDENCIES = frozenset({"numpy", "torch", "gymnasium", "matplotlib", "transformers", "datasets", "accelerate", "trl"})


def __getattr__(name: str):
    """Resolve a classic or fine-tuning export on first use.

    A missing optional dependency becomes an ImportError naming the extra
    to install; any other ImportError (a defect in the module) is re-raised
    untouched.
    """
    if name not in _LAZY:
        raise AttributeError(f"module 'reservoir' has no attribute {name!r}")
    module_name, attribute, extra = _LAZY[name]
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        missing = (exc.name or "").split(".")[0]
        if missing in _OPTIONAL_DEPENDENCIES:
            raise ImportError(
                f"reservoir.{name} needs the '{extra}' extra: pip install \"reservoir-replay[{extra}]\" "
                f"(missing {missing})"
            ) from exc
        raise
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


# Only the eager names: ``from reservoir import *`` must not import torch.
# The lazy names in ``_LAZY`` are reached by attribute access.
__all__ = [
    "ExactPERBuffer",
    "Rollout",
    "RolloutGroup",
    "RolloutBatch",
    "RolloutBuffer",
    "DurableRolloutBuffer",
    "__version__",
]
