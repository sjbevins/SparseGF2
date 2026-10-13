"""Statistics helpers for ensemble data: survival curves and erasure protection.

These are small, dependency-free functions for the quantities monitored-circuit
studies need most often. They work on plain numpy arrays, such as the fields of
:class:`sparsegf2_v3.mipt.EnsembleResult`.

Erasure protection of a code C on n qubits, from first-failure sizes D (one per
random ordering of the qubits; see :func:`sparsegf2_v3.mipt.erasure_first_failure`):

* ``failure_by_size``: P_C(m), the probability that a uniformly random m-qubit
  erasure is uncorrectable, estimated as the fraction of samples with D <= m;
* ``erasure_failure_probability``: Phi_C(e) = sum_m C(n,m) e^m (1-e)^(n-m) P_C(m),
  the failure probability when every qubit is erased independently with
  probability e, estimated as the mean over samples of Pr[Bin(n, e) >= D];
  with exhaustive counts for m <= w
  (:func:`sparsegf2_v3.mipt.uncorrectable_region_counts`) the m <= w terms are
  exact and the samples cover only m > w;
* ``erasure_tolerance``: e_delta = sup{e : Phi(e) <= delta} for an ensemble.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "bernoulli_erasure_curve",
    "erasure_failure_probability",
    "erasure_tolerance",
    "exact_failure_by_size",
    "failure_by_size",
    "kaplan_meier",
    "mean_and_error",
]


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


# -- erasure protection ---------------------------------------------------------------


def _log_binomial(n: int) -> np.ndarray:
    return np.array(
        [math.lgamma(n + 1) - math.lgamma(m + 1) - math.lgamma(n - m + 1) for m in range(n + 1)]
    )


def _binomial_weights(n: int, e: float, log_binom: np.ndarray) -> np.ndarray:
    """b_m(e) = C(n, m) e^m (1 - e)^(n - m) for m = 0..n."""
    w = np.zeros(n + 1)
    if e == 0.0:
        w[0] = 1.0
    elif e == 1.0:
        w[n] = 1.0
    else:
        m = np.arange(n + 1)
        w = np.exp(log_binom + m * math.log(e) + (n - m) * math.log1p(-e))
    return w


def _tail(weights: np.ndarray) -> np.ndarray:
    """T[d] = Pr[Bin >= d] for d = 0..n+1 (T[n + 1] = 0)."""
    t = np.zeros(weights.size + 1)
    t[:-1] = np.cumsum(weights[::-1])[::-1]
    return np.minimum(t, 1.0)


def _first_failure_array(first_failure, num_qubits: int) -> np.ndarray:
    d = np.asarray(first_failure, dtype=np.float64)
    if d.ndim == 0 or d.shape[-1] == 0:
        raise ValueError("first_failure needs a last axis of samples (at least one)")
    if not np.all(np.isfinite(d)) or np.any(d != np.round(d)):
        raise ValueError("first-failure sizes must be finite integers")
    if np.any((d < 1) | (d > num_qubits + 1)):
        raise ValueError(
            f"first-failure sizes must be in [1, {num_qubits + 1}] for n = {num_qubits}"
        )
    return d.astype(np.int64)


def _histogram(d: np.ndarray, num_qubits: int) -> np.ndarray:
    """Per-code fraction of samples with D = d, for d = 0..n+1: shape (..., n + 2)."""
    lead, samples = d.shape[:-1], d.shape[-1]
    flat = d.reshape(-1, samples)
    rows = np.repeat(np.arange(flat.shape[0]), samples)
    hist = np.bincount(
        rows * (num_qubits + 2) + flat.ravel(), minlength=flat.shape[0] * (num_qubits + 2)
    )
    return (hist.reshape(flat.shape[0], num_qubits + 2) / samples).reshape(*lead, num_qubits + 2)


def _matching_counts(exact_counts, d: np.ndarray) -> np.ndarray:
    """``exact_counts`` as an array whose leading axes are those of the samples ``d``."""
    c = np.asarray(exact_counts, dtype=np.float64)
    if c.ndim == 0 or c.shape[:-1] != d.shape[:-1]:
        raise ValueError(
            f"exact_counts must have shape {d.shape[:-1]} + (w,), one row per code of "
            f"first_failure; got {c.shape}"
        )
    return c


def _exact_part(counts, num_qubits: int, weights: np.ndarray) -> tuple[np.ndarray, int]:
    p_exact = exact_failure_by_size(counts, num_qubits)  # (..., w)
    w = p_exact.shape[-1]
    return p_exact @ weights[1 : w + 1], w


def failure_by_size(first_failure, num_qubits: int) -> np.ndarray:
    """Estimated P_C(m) for m = 0..n: the fraction of first-failure samples with D <= m.

    ``first_failure`` has shape ``(..., samples)``; the result has shape
    ``(..., n + 1)``. Because the erased regions of one ordering are nested,
    one ordering contributes to every m at once.
    """
    d = _first_failure_array(first_failure, num_qubits)
    hist = _histogram(d, num_qubits)
    return np.cumsum(hist, axis=-1)[..., : num_qubits + 1]


def exact_failure_by_size(counts, num_qubits: int) -> np.ndarray:
    """Exact P_C(m) for m = 1..w from exhaustive counts of uncorrectable m-qubit erasures.

    ``counts[..., m - 1]`` is the number of uncorrectable m-subsets (the
    output of :func:`sparsegf2_v3.mipt.uncorrectable_region_counts`); each is
    divided by C(n, m).
    """
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim == 0:
        raise ValueError("counts needs a last axis indexed by region size m = 1..w")
    w = c.shape[-1]
    if w > num_qubits:
        raise ValueError(f"counts cover sizes up to {w} > n = {num_qubits}")
    binom = np.array([math.comb(num_qubits, m) for m in range(1, w + 1)], dtype=np.float64)
    if not np.all(np.isfinite(c)) or np.any((c < 0) | (c > binom)):
        raise ValueError("counts[..., m - 1] must lie in [0, C(n, m)]")
    return c / binom


def erasure_failure_probability(first_failure, num_qubits: int, e, *, exact_counts=None):
    """Phi_C(e): the probability that independent erasure with probability e is uncorrectable.

    ``first_failure`` has shape ``(..., samples)`` (one row of samples per
    code, e.g. ``result.values["erasure_first_failure"][:, c]``); ``e`` is a
    number or a 1-D array. The estimate is the mean over samples of
    Pr[Bin(n, e) >= D]. With ``exact_counts`` (shape ``(..., w)``, from
    :func:`sparsegf2_v3.mipt.uncorrectable_region_counts` at the same
    checkpoint) the terms with m <= w use the exact P_C(m) and the samples
    contribute Pr[Bin(n, e) >= max(D, w + 1)] (the hybrid estimator).
    Returns shape ``(...)`` for a number ``e`` and ``(..., len(e))`` otherwise.
    A code without logical qubits (D = n + 1) has Phi_C = 0.
    """
    n = int(num_qubits)
    d = _first_failure_array(first_failure, n)
    es = np.atleast_1d(np.asarray(e, dtype=np.float64))
    if es.ndim != 1 or np.any((es < 0) | (es > 1)):
        raise ValueError("e must be a number or a 1-D array of probabilities in [0, 1]")
    hist = _histogram(d, n)  # (..., n + 2)
    if exact_counts is not None:
        exact_counts = _matching_counts(exact_counts, d)
    log_binom = _log_binomial(n)
    out = np.empty(hist.shape[:-1] + (es.size,))
    for i, ei in enumerate(es):
        weights = _binomial_weights(n, float(ei), log_binom)
        tail = _tail(weights)  # (n + 2,)
        if exact_counts is None:
            out[..., i] = hist @ tail
        else:
            exact, w = _exact_part(exact_counts, n, weights)
            sampled = tail.copy()
            sampled[: w + 1] = tail[w + 1]  # Pr[Bin >= max(D, w + 1)]
            out[..., i] = exact + hist @ sampled
    return out if np.ndim(e) else out[..., 0]


def erasure_tolerance(first_failure, num_qubits: int, delta: float, *, exact_counts=None) -> float:
    """e_delta = sup{e : Phi(e) <= delta} for the ensemble of codes in ``first_failure``.

    Phi(e) is the mean of Phi_C(e) over the codes (all leading axes of
    ``first_failure``; select the ensemble first, e.g. codes with k > 0, and
    the same rows of ``exact_counts``). The true Phi is non-decreasing in e,
    but the hybrid estimate need not be (with few samples a code's sampled
    P_C(w + 1) can fall below its exact P_C(w)), so the last crossing is
    located on a grid of 2048 steps in e and refined by bisection to about
    1e-12. Returns 1.0 if Phi stays <= delta up to e = 1, and NaN for an
    empty ensemble. With ``delta = 0`` the answer is 0 unless no code ever
    fails (Phi = 0 for every e), then 1.
    """
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be in [0, 1]")
    n = int(num_qubits)
    d = _first_failure_array(first_failure, n)
    if d.size == 0:
        return math.nan
    pooled = _histogram(d, n).reshape(-1, n + 2).mean(axis=0)
    log_binom = _log_binomial(n)
    mean_counts = None
    if exact_counts is not None:
        c = _matching_counts(exact_counts, d)
        mean_counts = exact_failure_by_size(c, n).reshape(-1, c.shape[-1]).mean(axis=0)

    def phi(e: float) -> float:
        weights = _binomial_weights(n, e, log_binom)
        tail = _tail(weights)
        if mean_counts is None:
            return float(pooled @ tail)
        w = mean_counts.size
        sampled = tail.copy()
        sampled[: w + 1] = tail[w + 1]  # Pr[Bin >= max(D, w + 1)]
        return float(mean_counts @ weights[1 : w + 1] + pooled @ sampled)

    if delta == 0.0:  # Phi(e) > 0 for every e > 0 as soon as some erasure can fail
        never_fails = pooled[: n + 1].sum() == 0 and (mean_counts is None or not mean_counts.any())
        return 1.0 if never_fails else 0.0
    grid = np.linspace(0.0, 1.0, 2049)
    below = np.array([phi(float(e)) <= delta for e in grid])
    if below[-1]:
        return 1.0
    i = int(np.flatnonzero(below)[-1])  # Phi(0) = P_C(0) = 0, so below[0] holds
    lo, hi = float(grid[i]), float(grid[i + 1])  # phi(lo) <= delta < phi(hi)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if phi(mid) <= delta:
            lo = mid
        else:
            hi = mid
    return lo
