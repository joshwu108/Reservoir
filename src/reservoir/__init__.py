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
    rollout_attest.py   Turns buffer events into attestation records.
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
    checker/            Independent verifier. Imports nothing from this package.

Fine-tuning tools (separate from replay): prefcheck.py, trajectory.py,
report.py (preference-noise detection); anchor_set.py, forgetting_monitor.py,
replay_scheduler.py (forgetting measurement and replay).

See ``rollout_buffer.py`` ("Call structure"), ``decayed_tree.py`` and
``decay.py`` with docs/design.md §7 for the replay path. ``tests/`` mirrors
this layout, one file per module.
"""

__version__ = "0.4.0"

# Exact buffer (pure Python, arbitrary-precision integer arithmetic)
from reservoir.buffer import ExactPERBuffer

# Fast buffer - auto-selects C-backed or pure-Python implementation
try:
    from reservoir.c_buffer import CFastPERBuffer as FastPERBuffer
    _BACKEND = "c"
except (ImportError, OSError):
    # C extension not built, or ABI mismatch after Python upgrade - fall back
    from reservoir.fast_buffer import FastPERBuffer  # type: ignore[assignment]
    _BACKEND = "python"

# Always importable by explicit name
from reservoir.fast_buffer import FastPERBuffer as PyFastPERBuffer

# Which backend FastPERBuffer resolves to at import time.
# Read-only. Reflects import-time selection. Do not mutate at runtime.
backend: str = _BACKEND

# LLM-RL rollout replay: exact age-decayed priorities, attested, crash-atomic.
from reservoir.rollout import Rollout, RolloutGroup
from reservoir.rollout_buffer import RolloutBatch, RolloutBuffer
from reservoir.durable_rollout import DurableRolloutBuffer

__all__ = [
    "ExactPERBuffer",
    "FastPERBuffer",
    "PyFastPERBuffer",
    "Rollout",
    "RolloutGroup",
    "RolloutBatch",
    "RolloutBuffer",
    "DurableRolloutBuffer",
    "backend",
    "__version__",
]

# Optional modules — imported lazily so missing/broken deps don't crash the package
from reservoir.anchor_set import AnchorSet
from reservoir.forgetting_monitor import ForgettingMonitor, ForgettingAlert
from reservoir.replay_scheduler import ReplayScheduler
from reservoir.dataset_buffer import DatasetBuffer
from reservoir.report import PreferenceQualityReport, NoiseLabel

try:
    from reservoir.prefcheck import PreferenceNoiseDetector
except (ImportError, RuntimeError):
    # TRL not installed or incompatible (e.g. numpy version conflict on Colab).
    # PreferenceNoiseDetector is unavailable but the rest of the package works.
    PreferenceNoiseDetector = None  # type: ignore[assignment,misc]
