"""Statistics helpers for ensemble data: survival curves and erasure protection.

These are small, dependency-free functions for the quantities monitored-circuit
studies need most often. They work on plain numpy arrays, such as the fields of
:class:`sparsegf2.mipt.EnsembleResult`.

Erasure protection of a code C on n qubits, from first-failure sizes D (one per
random ordering of the qubits; see :func:`sparsegf2.mipt.erasure_first_failure`):

* ``failure_by_size``: P_C(m), the probability that a uniformly random m-qubit
  erasure is uncorrectable, estimated as the fraction of samples with D <= m;
* ``erasure_failure_probability``: Phi_C(e) = sum_m C(n,m) e^m (1-e)^(n-m) P_C(m),
  the failure probability when every qubit is erased independently with
  probability e, estimated as the mean over samples of Pr[Bin(n, e) >= D];
  with exhaustive counts for m <= w
  (:func:`sparsegf2.mipt.uncorrectable_region_counts`) the m <= w terms are
  exact and the samples cover only m > w;
* ``erasure_tolerance``: e_delta = sup{e : Phi(e) <= delta} for an ensemble;
* ``exhaustive_cutoff``: the cutoff w = m*(n) that an enumeration budget of B
  regions allows, max{w : sum_{m=1}^{w} C(n, m) <= B}, and
  ``exhaustive_region_count`` that sum.
"""

from __future__ import annotations

import math
import operator

import numpy as np

__all__ = [
    "bernoulli_erasure_curve",
    "distance_bounds",
    "erasure_failure_probability",
    "erasure_tolerance",
    "exact_failure_by_size",
    "exhaustive_cutoff",
    "exhaustive_region_count",
    "failure_by_size",
    "hybrid_failure_by_size",
    "junction_dip",
    "kaplan_meier",
    "mean_and_error",
    "mean_first_failure",
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
    """Per-code number of samples with D = d, for d = 0..n+1: int64, shape (..., n + 2).

    Callers sum the integer counts first and divide by the number of samples
    once, so a fraction that should be exactly 1 (or equal another ratio) is.
    """
    lead, samples = d.shape[:-1], d.shape[-1]
    flat = d.reshape(-1, samples)
    rows = np.repeat(np.arange(flat.shape[0]), samples)
    hist = np.bincount(
        rows * (num_qubits + 2) + flat.ravel(), minlength=flat.shape[0] * (num_qubits + 2)
    )
    return hist.reshape(*lead, num_qubits + 2)


def _matching_counts(exact_counts, d: np.ndarray) -> np.ndarray:
    """``exact_counts`` as an array whose leading axes are those of the samples ``d``."""
    c = np.asarray(exact_counts, dtype=np.float64)
    if c.ndim == 0 or c.shape[:-1] != d.shape[:-1]:
        raise ValueError(
            f"exact_counts must have shape {d.shape[:-1]} + (w,), one row per code of "
            f"first_failure; got {c.shape}"
        )
    return c


def _hybrid_phi(hist, samples: int, p_exact, num_qubits: int, e: float, log_binom) -> np.ndarray:
    """Phi(e) from first-failure histograms and, if given, the exact P_C(m) for m = 1..w.

    ``hist`` (``(..., n + 2)``) counts the samples with D = d out of
    ``samples``; ``p_exact`` is ``(..., w)`` or None. Without it, Phi is the
    mean of Pr[Bin(n, e) >= D]; with it, the hybrid
    ``sum_{m <= w} b_m(e) P_C(m)`` plus the mean of Pr[Bin(n, e) >= max(D, w + 1)].
    Linear in both, so pooled data give the mean of the per-code values. The
    single implementation behind :func:`erasure_failure_probability` and
    :func:`erasure_tolerance`.
    """
    weights = _binomial_weights(num_qubits, e, log_binom)
    tail = _tail(weights)  # (n + 2,)
    if p_exact is None:
        return hist @ tail / samples
    w = p_exact.shape[-1]
    sampled = tail.copy()
    sampled[: w + 1] = tail[w + 1]  # Pr[Bin >= max(D, w + 1)]
    return p_exact @ weights[1 : w + 1] + hist @ sampled / samples


def failure_by_size(first_failure, num_qubits: int) -> np.ndarray:
    """Estimated P_C(m) for m = 0..n: the fraction of first-failure samples with D <= m.

    ``first_failure`` has shape ``(..., samples)``; the result has shape
    ``(..., n + 1)``. Because the erased regions of one ordering are nested,
    one ordering contributes to every m at once. Each entry is the integer
    number of samples with D <= m divided once by ``samples``, so P_C(n) is
    exactly 1 for a code with logical qubits.
    """
    d = _first_failure_array(first_failure, num_qubits)
    failed = np.cumsum(_histogram(d, num_qubits), axis=-1)[..., : num_qubits + 1]
    return failed / d.shape[-1]


def exact_failure_by_size(counts, num_qubits: int) -> np.ndarray:
    """Exact P_C(m) for m = 1..w from exhaustive counts of uncorrectable m-qubit erasures.

    ``counts[..., m - 1]`` is the number of uncorrectable m-subsets (the
    output of :func:`sparsegf2.mipt.uncorrectable_region_counts`); each is
    divided by C(n, m). Float counts above 2**53 are rounded and may lie a
    few ulps above C(n, m): counts up to C(n, m) (1 + 1e-12) are accepted and
    the ratio is clipped to [0, 1]; anything further out raises ``ValueError``.
    """
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim == 0:
        raise ValueError("counts needs a last axis indexed by region size m = 1..w")
    w = c.shape[-1]
    if w > num_qubits:
        raise ValueError(f"counts cover sizes up to {w} > n = {num_qubits}")
    binom = np.array([float(math.comb(num_qubits, m)) for m in range(1, w + 1)])
    if not np.all(np.isfinite(c)) or np.any((c < 0) | (c > binom * (1 + 1e-12))):
        raise ValueError(
            "counts[..., m - 1] must lie in [0, C(n, m)] (up to a relative 1e-12 of rounding)"
        )
    return np.clip(c / binom, 0.0, 1.0)


def _whole(value, what: str) -> int:
    """``value`` as a Python int: numpy integers and whole-number floats are accepted."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{what} must be an integer, got {value!r}")
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{what} must be an integer, got {value!r}") from None


def _qubit_count(num_qubits) -> int:
    n = _whole(num_qubits, "num_qubits")
    if n < 0:
        raise ValueError(f"num_qubits must be non-negative, got {n}")
    return n


def exhaustive_region_count(num_qubits: int, max_size: int) -> int:
    """N_exact(n, w) = sum_{m=1}^{w} C(n, m): the regions an exhaustive count up to size w covers.

    That is the number of m-subsets with 1 <= m <= w that
    :func:`sparsegf2.mipt.uncorrectable_region_counts` with ``max_size=w``
    accounts for, as an exact Python int (it exceeds 2**63 for large n and w).
    ``max_size`` must lie in [0, n]; 0 gives 0.
    """
    n = _qubit_count(num_qubits)
    w = _whole(max_size, "max_size")
    if not 0 <= w <= n:
        raise ValueError(f"max_size must be in [0, {n}], got {w}")
    return sum(math.comb(n, m) for m in range(1, w + 1))


def exhaustive_cutoff(num_qubits: int, budget: int) -> int:
    """m*(n) = max{w in 0..n : sum_{m=1}^{w} C(n, m) <= budget}: the largest exhaustive size a budget allows.

    P_C(m) is then exact for m <= m*(n) (every m-subset is enumerated) and
    sampled above. ``budget`` counts regions (:func:`exhaustive_region_count`),
    not seconds: the time per region depends on n and w, and every evaluation
    also pays a fixed elimination. 0 means that not even the n single-qubit
    regions fit; a budget of at least 2**n - 1 (every non-empty subset) gives n.
    The probe takes the budget directly:
    ``mipt.uncorrectable_region_counts(budget=B)``.
    """
    n = _qubit_count(num_qubits)
    b = _whole(budget, "budget")
    if b < 0:
        raise ValueError(f"budget must be non-negative, got {b}")
    total, term, w = 0, 1, 0
    while w < n:
        term = term * (n - w) // (w + 1)  # C(n, w + 1), exactly
        if total + term > b:
            break
        total += term
        w += 1
    return w


def hybrid_failure_by_size(first_failure, num_qubits: int, *, exact_counts=None) -> np.ndarray:
    """P_C(m) for m = 0..n as the hybrid estimators see it, shape ``(..., n + 1)``.

    Without ``exact_counts`` this is :func:`failure_by_size`. With counts of
    the uncorrectable m-subsets for m = 1..w, the entries m <= w are the exact
    P_C(m) and the entries m > w the sampled fractions. This is the curve
    behind :func:`erasure_failure_probability`, whose hybrid estimate equals
    ``sum_m C(n, m) e^m (1 - e)^(n - m) P_C(m)`` of this curve.
    """
    n = int(num_qubits)
    curve = failure_by_size(first_failure, n)
    if exact_counts is None:
        return curve
    d = _first_failure_array(first_failure, n)
    exact = exact_failure_by_size(_matching_counts(exact_counts, d), n)
    out = curve.copy()
    out[..., 1 : exact.shape[-1] + 1] = exact
    return out


def mean_first_failure(first_failure, num_qubits: int, *, exact_counts=None) -> np.ndarray:
    """mu_D = E[D] = sum_{m=0}^{n} (1 - P_C(m)) per code, shape ``(...)``.

    Uses :func:`hybrid_failure_by_size`, so it agrees with
    :func:`erasure_failure_probability` given the same data. A code without
    logical qubits (every D = n + 1) gives n + 1; condition on k > 0 before
    averaging over an ensemble.
    """
    curve = hybrid_failure_by_size(first_failure, num_qubits, exact_counts=exact_counts)
    return (1.0 - curve).sum(axis=-1)


def distance_bounds(
    first_failure, num_qubits: int, *, exact_counts=None
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds ``(low, high)`` on each code's distance d, as int64 arrays of shape ``(...)``.

    Erasing fewer than d qubits is always correctable, so every sampled
    first-failure size D_j >= d and ``high = min_j D_j``. Without counts
    ``low = 1``. With exhaustive counts for m = 1..w, the smallest m with an
    uncorrectable m-subset is d exactly (``low = high = m``); if there is none,
    ``low = w + 1``. A code without logical qubits (min D = n + 1) gets
    ``(n + 1, n + 1)``: it has no distance, so reports exclude it.
    """
    n = int(num_qubits)
    d = _first_failure_array(first_failure, n)
    high = d.min(axis=-1)
    low = np.ones_like(high)
    if exact_counts is not None:
        c = _matching_counts(exact_counts, d)
        w = c.shape[-1]
        positive = c > 0
        found = positive.any(axis=-1)
        first = np.argmax(positive, axis=-1) + 1
        low = np.where(found, first, w + 1)
        high = np.where(found, first, high)
    no_code = d.min(axis=-1) == n + 1
    low = np.where(no_code, n + 1, low)
    high = np.where(no_code, n + 1, high)
    return low.astype(np.int64), high.astype(np.int64)


def junction_dip(first_failure, num_qubits: int, exact_counts) -> np.ndarray:
    """Whether the sampled P_C(w + 1) falls below the exact P_C(w), per code.

    The true P_C(m) is non-decreasing; a dip at the junction of the hybrid
    curve is sampling noise. The estimators stay unbiased (nothing is
    clipped); reports flag it and suggest more orderings.
    """
    n = int(num_qubits)
    curve = hybrid_failure_by_size(first_failure, n, exact_counts=exact_counts)
    w = np.asarray(exact_counts).shape[-1]
    if w >= n:
        return np.zeros(curve.shape[:-1], dtype=bool)
    return curve[..., w + 1] < curve[..., w]


def erasure_failure_probability(first_failure, num_qubits: int, e, *, exact_counts=None):
    """Phi_C(e): the probability that independent erasure with probability e is uncorrectable.

    ``first_failure`` has shape ``(..., samples)`` (one row of samples per
    code, e.g. ``result.values["erasure_first_failure"][:, c]``); ``e`` is a
    number or a 1-D array. The estimate is the mean over samples of
    Pr[Bin(n, e) >= D]. With ``exact_counts`` (shape ``(..., w)``, from
    :func:`sparsegf2.mipt.uncorrectable_region_counts` at the same
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
    p_exact = None
    if exact_counts is not None:
        p_exact = exact_failure_by_size(_matching_counts(exact_counts, d), n)  # (..., w)
    log_binom = _log_binomial(n)
    out = np.empty(hist.shape[:-1] + (es.size,))
    for i, ei in enumerate(es):
        out[..., i] = _hybrid_phi(hist, d.shape[-1], p_exact, n, float(ei), log_binom)
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
    # Phi is linear in the histogram and in P_C(m): the samples of all codes pooled
    # (every code has the same number) and the mean exact P_C(m) give the mean Phi_C.
    pooled = _histogram(d, n).reshape(-1, n + 2).sum(axis=0)
    log_binom = _log_binomial(n)
    mean_exact = None
    if exact_counts is not None:
        c = _matching_counts(exact_counts, d)
        mean_exact = exact_failure_by_size(c, n).reshape(-1, c.shape[-1]).mean(axis=0)

    def phi(e: float) -> float:
        return float(_hybrid_phi(pooled, d.size, mean_exact, n, e, log_binom))

    if delta == 0.0:  # Phi(e) > 0 for every e > 0 as soon as some erasure can fail
        never_fails = pooled[: n + 1].sum() == 0 and (mean_exact is None or not mean_exact.any())
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
