"""
Rehydration: reconstruct a live :class:`SparseGF2` from persisted tableau bits.

The circuits package stores end-of-circuit stabilizer tableaus as bit-packed
symplectic ``x_packed`` / ``z_packed`` arrays in ``tableaus.h5``. Analyses
are written against the existing ``SparseGF2`` public API (e.g.
``sim.compute_subsystem_entropy``, ``sparsegf2.analysis.observables.observe``),
so we load the packed bits back into a fresh simulator instead of
reimplementing every algorithm on raw arrays.

Rehydration flow:

1. Construct ``sim = SparseGF2(n, hybrid_mode=False)``. This runs
   ``_init_bell_pairs`` but we immediately overwrite the state.
2. Copy ``x_packed``/``z_packed`` into ``sim.x_packed``/``sim.z_packed``.
3. Run the packed->PLT->indices rebuild (kernels from
   ``sparsegf2.core.numba_kernels``). This is what the simulator's own
   ``_switch_to_sparse()`` method does after a dense-mode session.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from sparsegf2 import SparseGF2
from sparsegf2.core.numba_kernels import packed_to_plt, rebuild_indices_from_plt


def rehydrate_sim(
    n: int,
    x_packed: np.ndarray,
    z_packed: np.ndarray,
    *,
    use_min_weight_pivot: bool = True,
) -> SparseGF2:
    """Reconstruct a ``SparseGF2`` in sparse mode from packed tableau bits.

    Parameters
    ----------
    n : int
        Number of system qubits (2n = number of stabilizer generators).
    x_packed : ndarray, shape (2n, ceil(n/64)), dtype uint64
        Symplectic X part of the tableau.
    z_packed : ndarray, shape (2n, ceil(n/64)), dtype uint64
        Symplectic Z part of the tableau.
    use_min_weight_pivot : bool
        Forwarded to :class:`SparseGF2`; default matches the runner.

    Returns
    -------
    SparseGF2
        A fresh simulator in sparse mode whose state equals the loaded tableau.
    """
    N = 2 * n
    n_words = (n + 63) >> 6
    if x_packed.shape != (N, n_words):
        raise ValueError(
            f"x_packed shape must be ({N}, {n_words}); got {x_packed.shape}"
        )
    if z_packed.shape != (N, n_words):
        raise ValueError(
            f"z_packed shape must be ({N}, {n_words}); got {z_packed.shape}"
        )

    sim = SparseGF2(
        n,
        use_min_weight_pivot=use_min_weight_pivot,
        check_inputs=False,
        hybrid_mode=False,  # stay in sparse mode; we overwrite after init
    )
    # Overwrite the Bell-pair initial state with the loaded packed arrays.
    sim.x_packed[:] = np.asarray(x_packed, dtype=np.uint64)
    sim.z_packed[:] = np.asarray(z_packed, dtype=np.uint64)

    # packed -> PLT -> inverted indices (mirrors _switch_to_sparse)
    packed_to_plt(sim.x_packed, sim.z_packed, sim.plt, n, N)
    rebuild_indices_from_plt(
        sim.plt, sim.supp_q, sim.supp_len, sim.supp_pos,
        sim.inv, sim.inv_len, sim.inv_x, sim.inv_x_len,
        sim.inv_pos, sim.inv_x_pos, n, N,
    )
    sim._dense_mode = False
    return sim


def iter_rehydrated(
    n: int,
    x_stack: np.ndarray,
    z_stack: np.ndarray,
) -> Iterable[SparseGF2]:
    """Yield one :class:`SparseGF2` per sample in a cell.

    Parameters
    ----------
    n : int
        System size.
    x_stack : ndarray, shape (S, 2n, ceil(n/64))
        Stacked X-packed tableaus.
    z_stack : ndarray, shape (S, 2n, ceil(n/64))
        Stacked Z-packed tableaus.

    Yields
    ------
    SparseGF2
    """
    S = x_stack.shape[0]
    for i in range(S):
        yield rehydrate_sim(n, x_stack[i], z_stack[i])


__all__ = ["rehydrate_sim", "iter_rehydrated"]
