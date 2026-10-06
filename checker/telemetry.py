"""Alias of ``reservoir_checker.telemetry``; keeps ``from checker.telemetry import ...`` working from a checkout."""

import sys

from reservoir_checker import telemetry as _impl

sys.modules[__name__] = _impl
