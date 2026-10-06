"""Alias of ``reservoir_checker.diff``; keeps ``from checker.diff import ...`` working from a checkout."""

import sys

from reservoir_checker import diff as _impl

if __name__ == "__main__":
    sys.exit(_impl.main())

sys.modules[__name__] = _impl
