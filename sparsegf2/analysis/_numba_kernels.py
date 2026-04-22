"""Numba JIT-compiled kernels for weight statistics analysis.

Performance-critical inner loops compiled with @numba.njit(cache=True),
following the same pattern as sparsegf2/core/numba_kernels.py.
Falls back gracefully if numba is not installed.
"""
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@njit(cache=True)
def _compute_x_weights(plt, supp_q, supp_len, N):
    """Compute X-weight wt_X(r) for each generator r.

    wt_X(r) = |{q : (plt[r,q] >> 1) & 1 == 1}|  (qubits with X or Y)

    Iterates over each generator's support list for O(weight_mass) total cost.

    Parameters
    ----------
    plt : uint8[:, :]
        Pauli Lookup Table, shape (2n, n).
    supp_q : int32[:, :]
        Support lists, shape (2n, M_s).
    supp_len : int32[:]
        Weight of each generator, length 2n.
    N : int
        Total number of generators (2n).

    Returns
    -------
    int32[:]
        X-weight for each generator, length N.
    """
    x_weights = np.zeros(N, dtype=np.int32)
    for r in range(N):
        count = np.int32(0)
        for idx in range(supp_len[r]):
            q = supp_q[r, idx]
            if (plt[r, q] >> 1) & 1:
                count += 1
        x_weights[r] = count
    return x_weights


@njit(cache=True)
def _compute_pei_wcp(inv_x, inv_x_len, supp_len, n):
    """Compute Pivot Effectiveness Index and Worst-Case Pivot proxy sums.

    PEI_proxy = (1/n) * sum_{q : inv_x_len[q]>0} min_{r in inv_x[q]} supp_len[r]
    WCP_proxy = (1/n) * sum_{q : inv_x_len[q]>0} max_{r in inv_x[q]} supp_len[r]

    Returns raw sums and count; caller divides by n.

    Parameters
    ----------
    inv_x : int32[:, :]
        X-inverted index, shape (n, M).
    inv_x_len : int32[:]
        Length of inv_x[q], shape (n,).
    supp_len : int32[:]
        Weight of each generator, shape (2n,).
    n : int
        Number of qubits.

    Returns
    -------
    tuple of (int64, int64, int64)
        (pei_sum, wcp_sum, n_active_qubits)
    """
    pei_sum = np.int64(0)
    wcp_sum = np.int64(0)
    n_active = np.int64(0)
    for q in range(n):
        L = inv_x_len[q]
        if L == 0:
            continue
        n_active += 1
        min_w = supp_len[inv_x[q, 0]]
        max_w = min_w
        for idx in range(1, L):
            r = inv_x[q, idx]
            w = supp_len[r]
            if w < min_w:
                min_w = w
            if w > max_w:
                max_w = w
        pei_sum += min_w
        wcp_sum += max_w
    return pei_sum, wcp_sum, n_active


@njit(cache=True)
def _compute_x_weight_mass(plt, supp_q, supp_len, N):
    """Compute total X-weight mass sum_r wt_X(r) from PLT directly.

    Used by the consistency oracle to verify against sum_q inv_x_len[q].
    Cost: O(weight_mass).
    """
    total = np.int64(0)
    for r in range(N):
        for idx in range(supp_len[r]):
            q = supp_q[r, idx]
            if (plt[r, q] >> 1) & 1:
                total += 1
    return total


def _ensure_sparse_indices(sim):
    """Ensure sparse indices (inv_len, supp_len, etc.) reflect current state.

    If the simulator is in dense mode, rebuilds PLT and all sparse indices
    from the packed arrays without changing the mode flag. This is necessary
    because dense-mode gate operations do not maintain the sparse indices.
    """
    if sim._dense_mode:
        from sparsegf2.core.numba_kernels import packed_to_plt, rebuild_indices_from_plt
        packed_to_plt(sim.x_packed, sim.z_packed, sim.plt, sim.n, sim.N)
        rebuild_indices_from_plt(
            sim.plt, sim.supp_q, sim.supp_len, sim.supp_pos,
            sim.inv, sim.inv_len, sim.inv_x, sim.inv_x_len,
            sim.inv_pos, sim.inv_x_pos, sim.n, sim.N)
