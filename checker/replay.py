"""Alias of ``reservoir_checker.replay``; keeps ``python -m checker.replay`` working from a checkout."""

import sys

from reservoir_checker import replay as _impl

if __name__ == "__main__":
    sys.exit(_impl.main())

sys.modules[__name__] = _impl
