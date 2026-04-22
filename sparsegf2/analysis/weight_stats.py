"""Weight statistics analysis for SparseGF2 stabilizer tableaux.

Implements the qubit-marginal distribution (P_a), generator-marginal
distribution (P_w), X-weight distribution, the fundamental duality
identity n*abar = 2n*wbar, the consistency oracle, and the pivot
effectiveness index.

All computations are read-only with respect to the quantum state.
If the simulator is in dense mode, sparse indices are temporarily
rebuilt for O(n) access without changing the mode flag.
"""
import numpy as np
from sparsegf2.analysis.observables import WeightStats
from sparsegf2.analysis._numba_kernels import (
    _compute_x_weights,
    _compute_pei_wcp,
    _compute_x_weight_mass,
    _ensure_sparse_indices,
)


def compute_weight_stats(sim, *, _indices_ensured=False) -> WeightStats:
    """Compute all weight distribution statistics from the current tableau.

    Computes qubit-marginal (P_a), generator-marginal (P_w), X-weight
    distribution, stabilizer/destabilizer decomposition, and verifies
    the fundamental duality n*abar = 2n*wbar.

    Cost: O(n) from sparse indices + O(weight_mass) for X-weights.
    O(n) additional memory (no n^2-sized arrays allocated).

    Parameters
    ----------
    sim : SparseGF2
        Simulator instance with accessible plt, inv_len, supp_len, etc.

    Returns
    -------
    WeightStats
        Frozen dataclass with all weight statistics.
    """
    if not _indices_ensured:
        _ensure_sparse_indices(sim)
    n = sim.n
    N = sim.N  # 2n

    inv_len = sim.inv_len[:n]    # int32, shape (n,)
    supp_len = sim.supp_len[:N]  # int32, shape (2n,)

    # === Qubit-marginal (variable-node degrees) ===
    # abar = (1/n) * sum_q inv_len[q]
    total_inv = int(np.sum(inv_len))
    abar = total_inv / n

    a_f = inv_len.astype(np.float64)
    var_a = float(np.var(a_f))
    std_a = float(np.sqrt(var_a))
    cv_a = std_a / abar if abar > 0 else 0.0
    a_max = int(np.max(inv_len))
    a_min = int(np.min(inv_len))

    # skew_a = (1/n) * sum_q ((a_q - abar) / std_a)^3
    if std_a > 1e-15:
        skew_a = float(np.mean(((a_f - abar) / std_a) ** 3))
    else:
        skew_a = 0.0

    # P_a(k) = |{q : inv_len[q] = k}| / n
    counts_a = np.bincount(inv_len, minlength=a_max + 1)
    hist_a = counts_a.astype(np.float64) / n

    # === Generator-marginal (check-node weights) ===
    # wbar = (1/(2n)) * sum_r supp_len[r]
    total_supp = int(np.sum(supp_len))
    wbar = total_supp / N

    w_f = supp_len.astype(np.float64)
    var_w = float(np.var(w_f))
    std_w = float(np.sqrt(var_w))
    cv_w = std_w / wbar if wbar > 0 else 0.0
    w_max = int(np.max(supp_len))
    w_min = int(np.min(supp_len))

    if std_w > 1e-15:
        skew_w = float(np.mean(((w_f - wbar) / std_w) ** 3))
    else:
        skew_w = 0.0

    # P_w(k) = |{r : supp_len[r] = k}| / (2n)
    counts_w = np.bincount(supp_len, minlength=w_max + 1)
    hist_w = counts_w.astype(np.float64) / N

    # === Stabilizer/destabilizer decomposition ===
    # wbar_stab = (1/n) * sum_{r=n}^{2n-1} supp_len[r]
    # wbar_destab = (1/n) * sum_{r=0}^{n-1} supp_len[r]
    wbar_stab = float(np.sum(supp_len[n:N])) / n
    wbar_destab = float(np.sum(supp_len[:n])) / n

    # === X-weight distribution ===
    # wt_X(r) = |{q : (plt[r,q] >> 1) & 1 == 1}|
    x_weights = _compute_x_weights(sim.plt, sim.supp_q, sim.supp_len, N)
    mean_wt_x = float(np.mean(x_weights.astype(np.float64)))
    x_max = int(np.max(x_weights)) if N > 0 else 0
    counts_x = np.bincount(x_weights, minlength=max(x_max + 1, 1))
    hist_wt_x = counts_x.astype(np.float64) / N

    # === Weight mass and identity ===
    # n*abar = total_inv, 2n*wbar = total_supp; must be equal (integer-exact)
    weight_mass = total_inv
    identity_holds = (total_inv == total_supp)

    return WeightStats(
        abar=float(abar), var_a=var_a, std_a=std_a, cv_a=cv_a,
        a_max=a_max, a_min=a_min, skew_a=skew_a, hist_a=hist_a,
        wbar=float(wbar), var_w=var_w, std_w=std_w, cv_w=cv_w,
        w_max=w_max, w_min=w_min, skew_w=skew_w, hist_w=hist_w,
        wbar_stab=wbar_stab, wbar_destab=wbar_destab,
        mean_wt_x=mean_wt_x, hist_wt_x=hist_wt_x,
        weight_mass=weight_mass, identity_holds=identity_holds,
    )


def verify_weight_mass_identity(sim, raise_on_failure=True) -> dict:
    """Verify the fundamental duality n*abar = 2n*wbar and X-weight consistency.

    Computes the total weight mass from BOTH sides independently:
      1. Qubit side:     total_inv   = sum_q inv_len[q]    (Invariant 3.1)
      2. Generator side: total_supp  = sum_r supp_len[r]   (Invariant 3.3)
      3. X-qubit side:   total_x_inv = sum_q inv_x_len[q]  (Invariant 3.2)
      4. X-generator:    total_x_supp = sum_r wt_X(r)      (from PLT)

    Asserts total_inv == total_supp and total_x_inv == total_x_supp
    (integer-exact comparisons).

    This is stronger than element-wise checking for scalar auditing
    but weaker than a full element-wise scan for debugging.

    Cost: O(n) for sums + O(weight_mass) for X-weight verification.

    Parameters
    ----------
    sim : SparseGF2
        Simulator instance.
    raise_on_failure : bool
        If True, raises AssertionError on mismatch.

    Returns
    -------
    dict
        Keys: total_inv, total_supp, inv_pass, total_x_inv, total_x_supp, x_pass.
    """
    _ensure_sparse_indices(sim)
    n = sim.n
    N = sim.N

    total_inv = int(np.sum(sim.inv_len[:n]))
    total_supp = int(np.sum(sim.supp_len[:N]))
    inv_pass = (total_inv == total_supp)

    total_x_inv = int(np.sum(sim.inv_x_len[:n]))
    total_x_supp = int(_compute_x_weight_mass(
        sim.plt, sim.supp_q, sim.supp_len, N))
    x_pass = (total_x_inv == total_x_supp)

    if raise_on_failure:
        if not inv_pass:
            raise AssertionError(
                f"Weight mass mismatch: sum inv_len={total_inv} "
                f"!= sum supp_len={total_supp}")
        if not x_pass:
            raise AssertionError(
                f"X-weight mass mismatch: sum inv_x_len={total_x_inv} "
                f"!= sum wt_X={total_x_supp}")

    return {
        'total_inv': total_inv, 'total_supp': total_supp, 'inv_pass': inv_pass,
        'total_x_inv': total_x_inv, 'total_x_supp': total_x_supp, 'x_pass': x_pass,
    }


def compute_pivot_effectiveness(sim, *, _indices_ensured=False) -> dict:
    """Compute the Pivot Effectiveness Index (PEI) and Worst-Case Pivot proxy.

    PEI_proxy = (1/n) * sum_{q : inv_x_len[q]>0} min_{r in inv_x[q]} supp_len[r]
    WCP_proxy = (1/n) * sum_{q : inv_x_len[q]>0} max_{r in inv_x[q]} supp_len[r]

    The ratio WCP_proxy / PEI_proxy quantifies the speedup from min-weight
    pivot selection (Section 5.5) over a worst-case pivot strategy.

    Parameters
    ----------
    sim : SparseGF2
        Simulator instance.

    Returns
    -------
    dict
        Keys: pei_proxy, wcp_proxy, speedup_ratio, n_active_qubits.
    """
    if not _indices_ensured:
        _ensure_sparse_indices(sim)
    n = sim.n

    pei_sum, wcp_sum, n_active = _compute_pei_wcp(
        sim.inv_x, sim.inv_x_len, sim.supp_len, n)

    pei_proxy = float(pei_sum) / n if n > 0 else 0.0
    wcp_proxy = float(wcp_sum) / n if n > 0 else 0.0
    speedup = wcp_proxy / pei_proxy if pei_proxy > 0 else 1.0

    return {
        'pei_proxy': pei_proxy,
        'wcp_proxy': wcp_proxy,
        'speedup_ratio': speedup,
        'n_active_qubits': int(n_active),
    }
