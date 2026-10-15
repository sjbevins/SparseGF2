"""Numba-compiled simulation kernels (private).

Nothing in here is public API; :class:`sparsegf2.Simulator` and the circuit
runner are the supported entry points. The layout of the state array is
documented in :mod:`sparsegf2._kernel.tableau`.

Importing this package, before any kernel module, deletes Numba's cached
kernels when any kernel source has changed since they were compiled
(:mod:`sparsegf2._kernel._cache`), so a cached caller never runs a stale callee.
"""

from sparsegf2._kernel._cache import clear_stale_caches

clear_stale_caches()
