"""Numba-compiled simulation kernels (private).

Nothing in here is public API; :class:`sparsegf2.Simulator` and the circuit
runner are the supported entry points. The layout of the state array is
documented in :mod:`sparsegf2._kernel.tableau`.

Importing this package, before any kernel module, stamps Numba's cache entries
for the kernels with a fingerprint of all kernel sources and deletes cached
kernels compiled from other sources (:mod:`sparsegf2._kernel._cache`), so a
cached caller does not load a callee compiled from older sources.
"""

from sparsegf2._kernel._cache import clear_stale_caches

clear_stale_caches()
