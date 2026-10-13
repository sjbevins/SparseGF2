"""Statistics helpers for ensemble data: survival curves and erasure thresholds.

These are small, dependency-free functions for the quantities monitored-circuit
studies need most often. They work on plain numpy arrays, such as the fields of
:class:`sparsegf2_v3.mipt.EnsembleResult`.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["bernoulli_erasure_curve", "kaplan_meier", "mean_and_error"]


def mean_and_error(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error of the mean along ``axis``, ignoring NaN entries."""
    v = np.asarray(values, dtype=np.float64)
    count = np.sum(~np.isnan(v), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(v, axis=axis) / count
        var = np.nansum((v - np.expand_dims(mean, axis)) ** 2, axis=axis) / (count - 1)
        err = np.sqrt(var / count)
    return mean, err


def kaplan_meier(times: np.ndarray, observed: np.ndarray) -> dict[str, np.ndarray]:
    """Kaplan-Meier estimate of a survival function from right-censored data.

    ``times[i]`` is the event time (e.g. purification layer) if ``observed[i]``,
    otherwise the censoring time (the last time the trajectory was checked).
    Returns ``times`` (distinct event times), ``survival`` (P(T > t) just after
    each), ``stderr`` (Greenwood's formula) and ``at_risk``. Use it to combine
    runs with different maximum depths; for a single run with one horizon,
    :meth:`EnsembleResult.survival` gives the same curve directly.

    ``EnsembleResult.purification_layer`` marks censored trajectories with -1,
    which is not a time: pass ``times=np.where(tau >= 0, tau, result.final_layer)``
    and ``observed=tau >= 0``. Negative or non-finite times raise ``ValueError``.
    """
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(observed, dtype=bool)
    if t.shape != e.shape:
        raise ValueError("times and observed must have the same shape")
    if not np.all(np.isfinite(t) & (t >= 0)):
        raise ValueError(
            "times must be finite and non-negative; censored entries need their censoring "
            "time, not a sentinel such as purification_layer = -1 (pass "
            "np.where(tau >= 0, tau, result.final_layer) with observed=tau >= 0)"
        )
    event_times = np.unique(t[e])
    survival = np.empty(event_times.size)
    stderr = np.empty(event_times.size)
    at_risk = np.empty(event_times.size, dtype=np.int64)
    s = 1.0
    greenwood = 0.0
    for i, ti in enumerate(event_times):
        n_risk = int(np.sum(t >= ti))
        d = int(np.sum((t == ti) & e))
        s *= 1.0 - d / n_risk
        if n_risk > d:
            greenwood += d / (n_risk * (n_risk - d))
        survival[i] = s
        stderr[i] = s * math.sqrt(greenwood)
        at_risk[i] = n_risk
    return {"times": event_times, "survival": survival, "stderr": stderr, "at_risk": at_risk}


def bernoulli_erasure_curve(per_size: np.ndarray, p: np.ndarray | float) -> np.ndarray:
    """Average of a per-erasure-size quantity when each qubit is erased with probability ``p``.

    ``per_size[m]`` (m = 0..n) is a quantity measured for uniformly random
    erasures of exactly ``m`` qubits, e.g. ``1 - correctable_fraction`` (the
    failure probability) or ``mean_lost``. Independent erasure of each of ``n``
    qubits with probability ``p`` erases ``m`` qubits with binomial probability,
    so the result is ``sum_m C(n, m) p**m (1 - p)**(n - m) * per_size[m]``.
    Accepts a scalar or an array of ``p``.
    """
    f = np.asarray(per_size, dtype=np.float64)
    n = f.size - 1
    ps = np.atleast_1d(np.asarray(p, dtype=np.float64))
    if np.any((ps < 0) | (ps > 1)):
        raise ValueError("p must be in [0, 1]")
    m = np.arange(n + 1)
    log_binom = np.array(
        [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) for k in m]
    )
    out = np.empty(ps.size)
    for i, pi in enumerate(ps):
        if pi == 0.0:
            out[i] = f[0]
        elif pi == 1.0:
            out[i] = f[n]
        else:
            w = np.exp(log_binom + m * math.log(pi) + (n - m) * math.log1p(-pi))
            out[i] = float(np.dot(w, f))
    return out if np.ndim(p) else out[0]
