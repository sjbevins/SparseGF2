"""Thread-count control for the parallel (Numba ``prange``) kernels."""

from __future__ import annotations

import contextlib

import numba


@contextlib.contextmanager
def threads(num_threads: int | None):
    """Temporarily set Numba's thread count (``None`` keeps the current setting)."""
    if num_threads is None:
        yield
        return
    if num_threads < 1:
        raise ValueError(f"num_threads must be at least 1, got {num_threads}")
    previous = numba.get_num_threads()
    numba.set_num_threads(min(int(num_threads), numba.config.NUMBA_NUM_THREADS))
    try:
        yield
    finally:
        numba.set_num_threads(previous)
