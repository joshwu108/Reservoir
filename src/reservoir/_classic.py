"""
reservoir._classic — Backend selection for the classic transition buffer.

``FastPERBuffer`` is the C-backed buffer when the extension built, else the
numpy one; ``backend`` says which. Importing this module needs torch, so
``reservoir`` resolves it lazily on first access to either name.
"""

try:
    from reservoir.c_buffer import CFastPERBuffer as FastPERBuffer
    backend: str = "c"
except (ImportError, OSError):
    # C extension not built, or ABI mismatch after a Python upgrade: fall back.
    from reservoir.fast_buffer import FastPERBuffer  # type: ignore[assignment]
    backend = "python"

__all__ = ["FastPERBuffer", "backend"]
