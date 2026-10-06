"""
reservoir_checker — The independent verifier of attestation logs.

A sibling package of ``reservoir`` in the same distribution, not a
subpackage: importing it does not import the buffer, tree, draw or digest
code it verifies, so its independence holds at runtime and not only in
the source. Every module here imports only the standard library and other
modules of this package; the suite and ``make check-imports`` enforce
that, and the mutation campaign is the evidence it matters.

``pip install reservoir-replay`` installs it with console scripts
(``reservoir-verify``, ``reservoir-transcript``, ``reservoir-diff``). The
top-level ``checker`` package in the repository aliases these modules so
``python -m checker.verify`` keeps working from a checkout.
"""
