"""
Error-metric implementations for the plotting layer.

All functions take a numpy array of per-sample values for one ``(n, x)``
group and return a ``(lower, upper)`` half-width pair suitable for
``ax.errorbar(yerr=(low, high))`` or ``ax.fill_between(y-low, y+high)``.

Metrics:

- :func:`sem`             — standard error of the mean; symmetric.
- :func:`std`             — sample standard deviation; symmetric.
- :func:`ci95_bootstrap`  — 95% non-parametric bootstrap CI; asymmetric.
- :func:`wilson`          — Wilson score interval for a binary proportion.
- :func:`pick_error_metric` — auto-selection: Wilson for binary, SEM otherwise.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

import numpy as np


def _as_finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def sem(values: np.ndarray) -> Tuple[float, float]:
    v = _as_finite(values)
    if v.size <= 1:
        return 0.0, 0.0
    s = float(np.std(v, ddof=1)) / math.sqrt(v.size)
    return s, s


def std(values: np.ndarray) -> Tuple[float, float]:
    v = _as_finite(values)
    if v.size <= 1:
        return 0.0, 0.0
    s = float(np.std(v, ddof=1))
    return s, s


def ci95_bootstrap(values: np.ndarray,
                   n_resamples: int = 1000,
                   seed: int = 42) -> Tuple[float, float]:
    v = _as_finite(values)
    if v.size <= 1:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_resamples, v.size))
    sampled_means = v[idx].mean(axis=1)
    lo, hi = np.quantile(sampled_means, [0.025, 0.975])
    mean = float(v.mean())
    return float(mean - lo), float(hi - mean)


def wilson(values: np.ndarray, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval half-widths for a binomial proportion.

    Values must be in ``{0, 1}``. Returns ``(phat - lower, upper - phat)``
    so that ``phat ± half-widths`` reproduces the Wilson interval.
    """
    v = _as_finite(values)
    if v.size == 0:
        return 0.0, 0.0
    n = int(v.size)
    k = int(v.sum())
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = phat + (z * z) / (2.0 * n)
    half = z * math.sqrt(phat * (1.0 - phat) / n + (z * z) / (4.0 * n * n))
    lower = (center - half) / denom
    upper = (center + half) / denom
    return float(phat - lower), float(upper - phat)


# ══════════════════════════════════════════════════════════════
# Registry + auto-selection
# ══════════════════════════════════════════════════════════════

ERROR_METRICS: Dict[str, Callable[[np.ndarray], Tuple[float, float]]] = {
    "sem": sem,
    "std": std,
    "ci95": ci95_bootstrap,
    "wilson": wilson,
    "none": lambda v: (0.0, 0.0),
}


def _is_binary(values: np.ndarray) -> bool:
    v = _as_finite(values)
    if v.size == 0:
        return False
    return bool(np.all((v == 0.0) | (v == 1.0)))


def pick_error_metric(metric: str, values: np.ndarray) -> str:
    """Resolve the ``"auto"`` metric to a concrete name using the values.

    Any metric other than ``"auto"`` passes through unchanged.
    """
    if metric != "auto":
        if metric not in ERROR_METRICS:
            raise ValueError(
                f"error_metric must be one of {sorted(ERROR_METRICS)} or 'auto'; "
                f"got {metric!r}"
            )
        return metric
    return "wilson" if _is_binary(values) else "sem"


__all__ = [
    "sem",
    "std",
    "ci95_bootstrap",
    "wilson",
    "ERROR_METRICS",
    "pick_error_metric",
]
