"""reservoir: Certified Exact Prioritized Experience Replay
with Crash Atomicity and Sampling Attestation.
"""

__version__ = "0.2.0"

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

__all__ = [
    "ExactPERBuffer",
    "FastPERBuffer",
    "PyFastPERBuffer",
    "backend",
    "__version__",
]

from reservoir.anchor_set import AnchorSet
from reservoir.forgetting_monitor import ForgettingMonitor, ForgettingAlert
from reservoir.replay_scheduler import ReplayScheduler
from reservoir.dataset_buffer import DatasetBuffer
from reservoir.prefcheck import PreferenceNoiseDetector
from reservoir.report import PreferenceQualityReport, NoiseLabel
