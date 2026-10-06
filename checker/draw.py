"""Alias of ``reservoir_checker.draw``; keeps ``from checker.draw import ...`` working from a checkout."""

import sys

from reservoir_checker import draw as _impl

sys.modules[__name__] = _impl
