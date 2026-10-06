"""Alias of ``reservoir_checker.decay_replay``; keeps ``from checker.decay_replay import ...`` working from a checkout."""

import sys

from reservoir_checker import decay_replay as _impl

sys.modules[__name__] = _impl
