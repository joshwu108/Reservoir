"""Alias of ``reservoir_checker.staleness``; keeps ``from checker.staleness import ...`` working from a checkout."""

import sys

from reservoir_checker import staleness as _impl

sys.modules[__name__] = _impl
