"""Observable extraction: WeightStats dataclass and observe() sweep utility."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True, eq=False)
class WeightStats:
    """Immutable container for all weight distribution statistics.

    Contains qubit-marginal (P_a), generator-marginal (P_w), X-weight,
    stabilizer/destabilizer decomposition, and weight mass quantities.

    Frozen: attribute reassignment is prevented. Numpy array fields
    (hist_a, hist_w, hist_wt_x) remain mutable in-place but should
    be treated as read-only.
    """
    # Qubit-marginal distribution (variable-node degrees)
    abar: float           # (1/n) * sum_q inv_len[q]
    var_a: float          # (1/n) * sum_q (a_q - abar)^2
    std_a: float          # sqrt(var_a)
    cv_a: float           # std_a / abar
    a_max: int            # max_q inv_len[q]
    a_min: int            # min_q inv_len[q]
    skew_a: float         # (1/n) * sum_q ((a_q - abar)/std_a)^3
    hist_a: np.ndarray    # P_a(k) = |{q : inv_len[q]=k}| / n

    # Generator-marginal distribution (check-node weights)
    wbar: float           # (1/(2n)) * sum_r supp_len[r]
    var_w: float          # (1/(2n)) * sum_r (w_r - wbar)^2
    std_w: float          # sqrt(var_w)
    cv_w: float           # std_w / wbar
    w_max: int            # max_r supp_len[r]
    w_min: int            # min_r supp_len[r]
    skew_w: float         # (1/(2n)) * sum_r ((w_r - wbar)/std_w)^3
    hist_w: np.ndarray    # P_w(k) = |{r : supp_len[r]=k}| / (2n)

    # Stabilizer/destabilizer decomposition
    wbar_stab: float      # (1/n) * sum_{r=n}^{2n-1} supp_len[r]
    wbar_destab: float    # (1/n) * sum_{r=0}^{n-1} supp_len[r]

    # X-weight distribution
    mean_wt_x: float      # (1/(2n)) * sum_r wt_X(r)
    hist_wt_x: np.ndarray # PMF of X-weights, normalized by 2n

    # Weight mass and consistency
    weight_mass: int      # n*abar = 2n*wbar (integer-exact)
    identity_holds: bool  # sum inv_len == sum supp_len


def observe(sim, p: Optional[float] = None) -> dict:
    """Extract all scalar observables as a flat dict for DataFrame construction.

    Parameters
    ----------
    sim : SparseGF2
        Simulator instance.
    p : float, optional
        Measurement rate metadata (included in output if provided).

    Returns
    -------
    dict
        Flat dictionary of scalar observables. All values are Python
        scalars (float, int, bool); no nested structures.
    """
    from sparsegf2.analysis._numba_kernels import _ensure_sparse_indices
    from sparsegf2.analysis.weight_stats import (
        compute_weight_stats, compute_pivot_effectiveness,
    )

    _ensure_sparse_indices(sim)
    stats = compute_weight_stats(sim, _indices_ensured=True)
    pei = compute_pivot_effectiveness(sim, _indices_ensured=True)

    d = {
        'abar': stats.abar,
        'var_a': stats.var_a,
        'std_a': stats.std_a,
        'cv_a': stats.cv_a,
        'a_max': stats.a_max,
        'a_min': stats.a_min,
        'skew_a': stats.skew_a,
        'wbar': stats.wbar,
        'var_w': stats.var_w,
        'std_w': stats.std_w,
        'cv_w': stats.cv_w,
        'w_max': stats.w_max,
        'w_min': stats.w_min,
        'skew_w': stats.skew_w,
        'wbar_stab': stats.wbar_stab,
        'wbar_destab': stats.wbar_destab,
        'mean_wt_x': stats.mean_wt_x,
        'weight_mass': stats.weight_mass,
        'identity_holds': stats.identity_holds,
        'pei_proxy': pei['pei_proxy'],
        'wcp_proxy': pei['wcp_proxy'],
        'pivot_speedup': pei['speedup_ratio'],
        'n_anticommuting_qubits': pei['n_active_qubits'],
    }
    if p is not None:
        d['p'] = float(p)
    return d
