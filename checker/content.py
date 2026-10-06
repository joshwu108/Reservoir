"""Alias of ``reservoir_checker.content``; keeps ``from checker.content import ...`` working from a checkout."""

import sys

from reservoir_checker import content as _impl

sys.modules[__name__] = _impl
