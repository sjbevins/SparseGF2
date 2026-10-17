"""Thread-count control for the parallel (Numba ``prange``) kernels, and their turns.

Every parallel kernel of the package is launched inside :func:`threads`: the ensemble engine
(:meth:`sparsegf2.mipt.MonitoredCircuit.run`), shot sampling (:mod:`sparsegf2.sampling`) and
the batch canonicalization of :mod:`sparsegf2.tableau`. Numba releases the GIL while a
parallel region runs, whatever the kernel's ``nogil`` option says, so launches from several
Python threads can overlap. Numba's ``omp`` and ``tbb`` threading layers run overlapping
regions; its ``workqueue`` layer (the one Numba falls back to when neither OpenMP nor TBB is
available, or the one ``NUMBA_THREADING_LAYER`` names) cannot, and terminates the process when
a second region starts while one runs. So under any layer but ``omp`` and ``tbb``,
:func:`threads` holds one process-wide lock while its block runs, and launches from several
Python threads take turns; under ``omp`` and ``tbb`` it takes no lock and they overlap.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator

import numba

CONCURRENT_LAYERS = frozenset({"omp", "tbb"})
"""Numba threading layers that run parallel regions launched from several threads at once."""

# Held by the thread whose block of `threads` runs, under a layer outside CONCURRENT_LAYERS.
# Reentrant, so a block may open another one in the same thread.
_LAUNCH_LOCK = threading.RLock()


def concurrent_launches() -> bool:
    """Whether parallel kernels launched from several Python threads may run at once.

    True for Numba's ``omp`` and ``tbb`` threading layers, False for ``workqueue`` and any
    other. Numba picks the layer once per process; ``numba.get_num_threads`` makes it pick
    now if no parallel region has run yet, so the answer is the layer every launch will use.
    """
    numba.get_num_threads()  # starts the threading layer, as a first launch would
    return numba.threading_layer() in CONCURRENT_LAYERS


@contextlib.contextmanager
def threads(num_threads: int | None) -> Iterator[None]:
    """Run a block of parallel launches on ``num_threads`` threads, in turn if need be.

    Temporarily sets Numba's thread count for the calling thread (``None`` keeps the current
    setting). Unless :func:`concurrent_launches`, the block also holds the process-wide launch
    lock, so the blocks of several Python threads run one after another instead of
    terminating the process.
    """
    if num_threads is not None and num_threads < 1:
        raise ValueError(f"num_threads must be at least 1, got {num_threads}")
    turn = contextlib.nullcontext() if concurrent_launches() else _LAUNCH_LOCK
    with turn:
        if num_threads is None:
            yield
            return
        previous = numba.get_num_threads()
        numba.set_num_threads(min(int(num_threads), numba.config.NUMBA_NUM_THREADS))
        try:
            yield
        finally:
            numba.set_num_threads(previous)
