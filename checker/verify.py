"""Alias of ``reservoir_checker.verify``; keeps ``from checker.verify import ...`` working from a checkout."""

import sys

from reservoir_checker import verify as _impl

if __name__ == "__main__":
    sys.exit(_impl.main())

sys.modules[__name__] = _impl
