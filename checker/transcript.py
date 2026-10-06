"""Alias of ``reservoir_checker.transcript``; keeps ``from checker.transcript import ...`` working from a checkout."""

import sys

from reservoir_checker import transcript as _impl

if __name__ == "__main__":
    sys.exit(_impl.main())

sys.modules[__name__] = _impl
