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

Geometry-correlated erasures (:mod:`sparsegf2.erasures`): a region A of fixed
size m is erased with probability Q_J(A | m) = exp(-J b(A)) / Z_m(J), b(A) the
number of edges of the interaction graph leaving A, and
P_J(m) = sum_{|A| = m} Q_J(A | m) 1[A is uncorrectable] (P_0 is P_C):

* ``exact_correlated_failure``: P_J(m) exactly for m <= w and every J at once,
  sum_b U_m(b) e^{-J b} / sum_b N_m(b) e^{-J b}, from the boundary-resolved
  counts of :meth:`sparsegf2.codes.StabilizerCode.boundary_failure_counts`;
* ``correlated_failure_curve``: a full curve P_J(m), m = 0..n, from exact and
  sampled sizes (linear interpolation in between, flagged), and
  ``binomial_window``: the sizes a binomial number of erasures can reach;
* ``failure_probability_from_curve``: Phi_J(e) = sum_m C(n,m) e^m (1-e)^(n-m) P_J(m),
  and ``tolerance_from_curve``: e_delta(J) = sup{e : Phi_J(e) <= delta};
* ``mean_over_codes``: ensemble means and standard errors over the codes with
  logical qubits (k > 0).
"""

from __future__ import annotations

import math
import operator
from fractions import Fraction
from typing import Any

import numpy as np

__all__ = [
    "MONOTONE_SLACK",
    "bernoulli_erasure_curve",
    "binomial_window",
    "correlated_failure_curve",
    "distance_bounds",
    "erasure_failure_probability",
    "erasure_tolerance",
    "exact_correlated_failure",
    "exact_failure_by_size",
    "exhaustive_cutoff",
    "exhaustive_region_count",
    "failure_by_size",
    "failure_probability_from_curve",
    "hybrid_failure_by_size",
    "junction_dip",
    "kaplan_meier",
    "mean_and_error",
    "mean_first_failure",
    "mean_over_codes",
    "tolerance_from_curve",
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
    which is not a time: use ``time, observed = result.purification_times()``,
    which censors at the last layer where purification was tested. Negative or
    non-finite times raise ``ValueError``.

    ``survival`` is the exact product of (n_j - d_j) / n_j, rounded to float64 once
    (not a running float product), so a survival of exactly 1/2 is 0.5 and the
    median rule "first time with survival <= 0.5" does not miss it.
    """
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(observed, dtype=bool)
    if t.shape != e.shape:
        raise ValueError("times and observed must have the same shape")
    if not np.all(np.isfinite(t) & (t >= 0)):
        raise ValueError(
            "times must be finite and non-negative; censored entries need their censoring "
            "time, not a sentinel such as purification_layer = -1 (use "
            "time, observed = result.purification_times())"
        )
    event_times = np.unique(t[e])
    survival = np.empty(event_times.size)
    stderr = np.empty(event_times.size)
    at_risk = np.empty(event_times.size, dtype=np.int64)
    exact = Fraction(1)  # the exact product, rounded once per step by float()
    greenwood = 0.0
    for i, ti in enumerate(event_times):
        n_risk = int(np.sum(t >= ti))
        d = int(np.sum((t == ti) & e))
        exact *= Fraction(n_risk - d, n_risk)
        s = float(exact)
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


def _integer(value, what: str) -> int:
    """``value`` as a Python int: numpy integers and whole-number floats are accepted.

    Fractions, bools and strings raise ``TypeError``. :mod:`sparsegf2.mipt` validates
    every integer of a spec and its probes with this function too.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{what} must be an integer, got {value!r}")
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{what} must be an integer, got {value!r}") from None


def _qubit_count(num_qubits) -> int:
    n = _integer(num_qubits, "num_qubits")
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
    w = _integer(max_size, "max_size")
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
    b = _integer(budget, "budget")
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


# -- geometry-correlated erasures ------------------------------------------------------


def _couplings_1d(couplings) -> np.ndarray:
    try:
        js = np.atleast_1d(np.asarray(couplings, dtype=np.float64))
    except (TypeError, ValueError):
        raise ValueError(f"couplings must be numbers, got {couplings!r}") from None
    if js.ndim != 1 or js.size == 0 or not np.all(np.isfinite(js)):
        raise ValueError(
            "couplings must be one finite number J or a 1-D list of them (J -> +inf is a "
            "limit: on the ring use StabilizerCode.interval_failure_by_size)"
        )
    return js


def _count_array(values, name: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an array of counts") from None
    if arr.ndim < 2:
        raise ValueError(
            f"{name} needs shape (..., w, B + 1): region size m = 1..w on the second-to-last "
            f"axis, boundary b = 0..B on the last (BoundaryCounts.{name}); got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)) or np.any(arr < 0) or np.any(arr != np.round(arr)):
        raise ValueError(f"{name} must hold non-negative whole numbers (counts of regions)")
    return arr


def exact_correlated_failure(failed, total, couplings) -> np.ndarray:
    """P_J(m) exactly, for every coupling J and every size m = 1..w, from boundary-resolved counts.

    ``failed[..., m - 1, b] = U_m(b)``, the number of uncorrectable regions of
    size m with edge boundary b, and ``total[..., m - 1, b] = N_m(b)``, the
    number of all regions of size m with boundary b (the density of states of
    the graph), as :meth:`sparsegf2.codes.StabilizerCode.boundary_failure_counts`
    returns them. Under the correlated law Q_J(A | m) = exp(-J b(A)) / Z_m(J)
    the failure probability is, exactly and for every J at once::

        P_J(m) = sum_b U_m(b) exp(-J b) / sum_b N_m(b) exp(-J b)

    J = 0 gives sum_b U_m(b) / C(n, m), the uniform P_C(m)
    (:func:`exact_failure_by_size` of the summed counts, equal to it bit for
    bit while the counts are below 2**53). As J -> +inf it tends to
    U_m(b_min) / N_m(b_min), the failing fraction of the regions of smallest
    boundary; as J -> -inf to U_m(b_max) / N_m(b_max).

    Leading axes (``...``, for example one row per trajectory) broadcast
    between ``failed`` and ``total``, so one ``total`` (shape ``(w, B + 1)``)
    serves every code on the same graph. The boundary axes may differ in
    length: the shorter is padded with zeros (a boundary that occurs in
    neither changes nothing), so rows of :func:`sparsegf2.erasures.boundary_counts`,
    each trimmed to its own largest boundary, can be padded and stacked into a
    ``total``. ``couplings``: one finite J or a 1-D list, any order (negative J
    favors spread-out regions).

    Numerically stable for any finite J: each sum is shifted by the smallest
    boundary with N_m(b) > 0 when J >= 0 (the largest when J < 0), so every
    exponent is <= 0 and the denominator is at least 1. The counts are
    converted to float64 first, which is exact below 2**53 (far above any
    enumeration budget; larger counts would be rounded before the checks).
    Returns float64 of shape ``(..., c, w)``: entry ``[..., i, m - 1]`` is P at
    ``couplings[i]`` and size m.
    """
    u = _count_array(failed, "failed")
    t = _count_array(total, "total")
    if u.shape[-2] != t.shape[-2]:
        raise ValueError(
            f"failed and total must cover the same sizes m = 1..w (second-to-last axis): "
            f"got {u.shape[-2]} and {t.shape[-2]}"
        )
    width = max(u.shape[-1], t.shape[-1])
    u = np.concatenate([u, np.zeros(u.shape[:-1] + (width - u.shape[-1],))], axis=-1)
    t = np.concatenate([t, np.zeros(t.shape[:-1] + (width - t.shape[-1],))], axis=-1)
    try:
        u, t = np.broadcast_arrays(u, t)
    except ValueError:
        raise ValueError(
            f"the leading axes of failed {u.shape[:-2]} and total {t.shape[:-2]} do not broadcast"
        ) from None
    if np.any(u > t):
        raise ValueError(
            "failed[..., m - 1, b] (the failing regions) cannot exceed total[..., m - 1, b] "
            "(all regions of that size and boundary)"
        )
    present = t > 0
    if not np.all(present.any(axis=-1)):
        raise ValueError("total has a size m with no region at all: every row needs N_m(b) > 0")
    js = _couplings_1d(couplings)
    b = np.arange(width, dtype=np.float64)
    first = np.argmax(present, axis=-1).astype(np.float64)
    last = (width - 1 - np.argmax(present[..., ::-1], axis=-1)).astype(np.float64)
    out = np.empty(u.shape[:-2] + (js.size, u.shape[-2]))
    for i, coupling in enumerate(js.tolist()):
        shift = first if coupling >= 0 else last
        with np.errstate(over="ignore"):
            exponent = -coupling * (b - shift[..., None])
        weight = np.exp(np.where(present, exponent, -np.inf))  # <= 1 where N_m(b) > 0
        out[..., i, :] = (u * weight).sum(axis=-1) / (t * weight).sum(axis=-1)
    return out


def _probabilities(e, what: str = "e") -> np.ndarray:
    try:
        es = np.atleast_1d(np.asarray(e, dtype=np.float64))
    except (TypeError, ValueError):
        es = np.array([np.nan])
    if es.ndim != 1 or es.size == 0 or not np.all((es >= 0) & (es <= 1)):
        raise ValueError(f"{what} must be a number or a 1-D array of probabilities in [0, 1]")
    return es


def _curve_qubits(num_qubits) -> int:
    n = _qubit_count(num_qubits)
    if n < 1:
        raise ValueError("num_qubits must be at least 1")
    return n


def binomial_window(num_qubits: int, e, tol: float = 1e-12) -> tuple[int, int]:
    """The sizes ``(m_lo, m_hi)`` that a Bin(n, e) number of erased qubits can reach, up to ``tol``.

    With every one of n qubits erased independently with probability e, the
    number erased M follows Bin(n, e), and Phi(e) = sum_m Pr[M = m] P(m) only
    needs P(m) for m_lo <= m <= m_hi: the window is chosen so that each tail
    has probability at most tol / 2::

        Pr[M < m_lo] <= tol / 2    and    Pr[M > m_hi] <= tol / 2

    with m_lo as large and m_hi as small as that allows, so the sizes outside
    it carry at most ``tol`` in total and change Phi by at most ``tol`` (P lies
    in [0, 1]). ``e`` is a number or a 1-D array; for an array the window
    covers every e (the smallest m_lo and the largest m_hi). The binomial
    probabilities are computed in log space (lgamma), so large n is safe;
    e = 0 gives (0, 0), e = 1 gives (n, n), and tol = 0 gives (0, n) for any
    other e. ``tol`` must lie in [0, 1).
    """
    n = _qubit_count(num_qubits)
    es = _probabilities(e)
    try:
        cut = float(tol)
    except (TypeError, ValueError):
        cut = math.nan
    if not 0.0 <= cut < 1.0:
        raise ValueError(f"tol must be a number in [0, 1), got {tol!r}")
    log_binom = _log_binomial(n)
    lo, hi = n, 0
    for ei in es.tolist():
        if ei == 0.0:
            window = (0, 0)
        elif ei == 1.0:
            window = (n, n)
        elif cut == 0.0:
            window = (0, n)
        else:
            pmf = _binomial_weights(n, ei, log_binom)
            below = np.cumsum(pmf)  # below[m] = Pr[M <= m]
            beyond = np.append(np.cumsum(pmf[::-1])[::-1][1:], 0.0)  # beyond[m] = Pr[M > m]
            m_lo = int(np.count_nonzero(below <= cut / 2))  # Pr[M < m_lo] = below[m_lo - 1]
            m_hi = int(np.count_nonzero(beyond > cut / 2))  # first m with Pr[M > m] <= tol / 2
            window = (min(m_lo, m_hi), m_hi)
        lo, hi = min(lo, window[0]), max(hi, window[1])
    return lo, hi


def _unit_values(values, name: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an array of probabilities") from None
    known = arr[~np.isnan(arr)]
    if np.any((known < 0) | (known > 1)):
        raise ValueError(f"{name} must hold probabilities in [0, 1] (NaN marks a missing value)")
    return arr


def correlated_failure_curve(
    sampled, sizes, num_qubits: int, *, exact=None, holds_code=None
) -> tuple[np.ndarray, np.ndarray]:
    """A full failure curve P_J(m), m = 0..n, per coupling, from exact and sampled sizes.

    ``sampled`` has shape ``(..., c, s)``: estimates of P_J(m) at the ``s``
    region sizes ``sizes`` (distinct integers in [0, n], any order; for example
    ``CorrelatedFailure.failure``, or the mean of such over codes), one row per
    coupling. ``exact`` (optional) has shape ``(..., c, w)``: the exact P_J(m)
    for m = 1..w (:func:`exact_correlated_failure`). ``holds_code`` (optional,
    shape ``(...)``, booleans or code dimensions k, nonzero meaning a code)
    gives P(n). Leading axes broadcast. The curve, float64 ``(..., c, n + 1)``,
    takes each entry from the first source that has it:

    1. P(0) = 0 (erasing nothing never fails), and P(n) = 1[k > 0] (erasing
       everything fails exactly when there is logical information): from
       ``holds_code`` when given, else from ``exact`` when it covers m = n
       (w = n), else 1 (codes already conditioned on k > 0; pass
       ``holds_code`` for any other ensemble);
    2. the exact values for 1 <= m <= min(w, n - 1);
    3. the sampled values at the other sizes;
    4. linear interpolation in m between the nearest sizes known by 1-3 on
       either side, P(m) = P(m_left) + t (P(m_right) - P(m_left)),
       t = (m - m_left) / (m_right - m_left), which is exactly P(m_left) on a
       plateau.

    Returns ``(curve, interpolated)``, ``interpolated`` a boolean array of
    length n + 1, True where step 4 filled the entry (so a report can say that
    those sizes were not measured). The interpolation is a modeling choice,
    a value between the neighbors. P_0(m) is nondecreasing in m (a uniform
    (m + 1)-region is a uniform m-region plus one more qubit, and a superset
    of a failing region fails), and so are the J -> +inf limits on the ring
    and the path; for other J and graphs P_J(m) need not be (the regions
    favored at size m + 1 need not contain those favored at size m).
    :func:`binomial_window` says which sizes matter for Phi. NaN in the inputs
    propagates.
    """
    n = _curve_qubits(num_qubits)
    s = _unit_values(sampled, "sampled")
    if s.ndim < 2:
        raise ValueError(
            f"sampled needs shape (..., c, s): couplings on the second-to-last axis and sizes on "
            f"the last (use [values] for a single coupling); got shape {s.shape}"
        )
    try:
        m_sampled = np.asarray(sizes, dtype=np.int64).ravel()
        same = np.array_equal(m_sampled, np.asarray(sizes, dtype=np.float64).ravel())
    except (TypeError, ValueError):
        same = False
    if (
        not same
        or m_sampled.size != s.shape[-1]
        or np.any((m_sampled < 0) | (m_sampled > n))
        or np.unique(m_sampled).size != m_sampled.size
    ):
        raise ValueError(
            f"sizes must list {s.shape[-1]} distinct integers in [0, {n}], one per entry of the "
            f"last axis of sampled; got {sizes!r}"
        )
    lead = s.shape[:-1]
    ex = None
    if exact is not None:
        ex = _unit_values(exact, "exact")
        if ex.ndim < 2 or ex.shape[-1] > n:
            raise ValueError(
                f"exact needs shape (..., c, w) with w <= n = {n} (sizes m = 1..w); got {ex.shape}"
            )
        try:
            lead = np.broadcast_shapes(lead, ex.shape[:-1])
        except ValueError:
            raise ValueError(
                f"the leading axes of sampled {s.shape[:-1]} and exact {ex.shape[:-1]} do not "
                "broadcast"
            ) from None
    top: Any = 1.0
    if holds_code is not None:
        h = np.asarray(holds_code)
        try:
            top = np.broadcast_to((h != 0).astype(np.float64)[..., None], lead)
        except ValueError:
            raise ValueError(
                f"holds_code needs the shape of the leading axes {lead[:-1]} (one entry per code, "
                f"no coupling axis); got {h.shape}"
            ) from None
    elif ex is not None and ex.shape[-1] == n:
        top = np.broadcast_to(ex[..., n - 1], lead)  # the exact P(n) = 1[k > 0]
    curve = np.full(lead + (n + 1,), np.nan)
    known = np.zeros(n + 1, dtype=bool)
    curve[..., 0] = 0.0
    curve[..., n] = top
    known[[0, n]] = True
    if ex is not None:
        for m in range(1, min(ex.shape[-1], n - 1) + 1):
            curve[..., m] = ex[..., m - 1]
            known[m] = True
    for j, m in enumerate(m_sampled.tolist()):
        if not known[m]:
            curve[..., m] = s[..., j]
            known[m] = True
    idx = np.flatnonzero(known)
    gaps = np.flatnonzero(~known)
    if gaps.size:
        place = np.searchsorted(idx, gaps)
        left, right = idx[place - 1], idx[place]
        frac = (gaps - left) / (right - left)
        start = curve[..., left]
        curve[..., gaps] = start + frac * (curve[..., right] - start)
    return curve, ~known


_CHUNK = 1 << 22
"""Largest number of binomial weights held at once (32 MB of float64)."""

MONOTONE_SLACK = 1e-12
"""A step down in P(m) no larger than this still counts as monotone (:func:`tolerance_from_curve`)."""


def _weight_rows(n: int, es: np.ndarray, log_binom: np.ndarray) -> np.ndarray:
    """``W[i, m] = C(n, m) e_i^m (1 - e_i)^(n - m)`` in log space, shape ``(len(es), n + 1)``."""
    w = np.zeros((es.size, n + 1))
    w[es == 0.0, 0] = 1.0
    w[es == 1.0, n] = 1.0
    inner = (es > 0.0) & (es < 1.0)
    if inner.any():
        m = np.arange(n + 1)
        e = es[inner][:, None]
        w[inner] = np.exp(log_binom[None, :] + m * np.log(e) + (n - m) * np.log1p(-e))
    return w


def _phi_grid(flat: np.ndarray, n: int, es: np.ndarray) -> np.ndarray:
    """Phi of every row of ``flat`` (``(r, n + 1)``) at every e: ``(r, len(es))``, in chunks of e."""
    log_binom = _log_binomial(n)
    step = max(1, _CHUNK // (n + 1))
    out = np.empty((flat.shape[0], es.size))
    for a in range(0, es.size, step):
        out[:, a : a + step] = flat @ _weight_rows(n, es[a : a + step], log_binom).T
    return out


def _phi_rows(flat: np.ndarray, n: int, es: np.ndarray, log_binom: np.ndarray) -> np.ndarray:
    """Phi of row i of ``flat`` at ``es[i]``: ``(r,)``, in chunks of rows."""
    step = max(1, _CHUNK // (n + 1))
    out = np.empty(flat.shape[0])
    for a in range(0, flat.shape[0], step):
        weights = _weight_rows(n, es[a : a + step], log_binom)
        out[a : a + step] = np.einsum("rm,rm->r", flat[a : a + step], weights)
    return out


def _curve_array(curve, n: int) -> np.ndarray:
    p = _unit_values(curve, "curve")
    if p.ndim < 1 or p.shape[-1] != n + 1:
        raise ValueError(
            f"curve needs a last axis of n + 1 = {n + 1} sizes m = 0..n "
            f"(correlated_failure_curve gives one); got shape {p.shape}"
        )
    return p


def failure_probability_from_curve(curve, num_qubits: int, e):
    """Phi(e) = sum_m C(n, m) e^m (1 - e)^(n - m) P(m): the failure probability of a binomial erasure.

    ``curve`` has shape ``(..., n + 1)``: P(m) for m = 0..n, for example
    P_J(m) from :func:`correlated_failure_curve`. Erasing every qubit
    independently with probability e erases M ~ Bin(n, e) qubits; with the
    region of that size drawn from Q_J(. | M) (the binomially mixed correlated
    channel), the probability of losing logical information is Phi_J(e). At
    J = 0 this is the independent-erasure Phi_C(e) of
    :func:`erasure_failure_probability` (equal to it up to rounding for the
    same exact P_C(m)). ``e`` is a number or a 1-D array; returns float64 of
    shape ``(...)`` or ``(..., len(e))``. The weights are those of
    :func:`erasure_failure_probability`, in log space (lgamma), computed a
    few million at a time, so large n is safe.
    """
    n = _curve_qubits(num_qubits)
    p = _curve_array(curve, n)
    es = _probabilities(e)
    out = _phi_grid(p.reshape(-1, n + 1), n, es).reshape(p.shape[:-1] + (es.size,))
    return out if np.ndim(e) else out[..., 0]


def tolerance_from_curve(curve, num_qubits: int, delta: float, *, return_monotone: bool = False):
    """e_delta = sup{e in [0, 1] : Phi(e) <= delta} for each failure curve P(m), m = 0..n.

    ``curve`` has shape ``(..., n + 1)`` (one curve per code, per coupling, or
    an ensemble mean: Phi is linear in P, so the mean curve gives the mean
    Phi). Phi is :func:`failure_probability_from_curve`. When P(m) is
    nondecreasing in m, Phi is nondecreasing in e (a binomial count grows
    stochastically with e), {e : Phi(e) <= delta} is an interval [0, e_delta]
    and e_delta is its end. P_0(m) is monotone; a sampled or interpolated
    curve need not be, nor need P_J(m) for J != 0 on every graph
    (:func:`correlated_failure_curve`). The method works either way: Phi is
    evaluated on a grid of 2048
    steps in e, the last grid point with Phi <= delta is located (the
    supremum, up to excursions narrower than a step when Phi is not
    monotone), and the crossing after it is refined by 50 bisection steps
    (to about 1e-18). Returns 1.0 when Phi(1) = P(n) <= delta and, for
    delta = 0, 0.0 unless P is 0 everywhere (then 1.0). P(0) must be 0.

    Returns a float for a 1-D curve, else an array of shape ``(...)`` (NaN
    for a curve with a NaN entry). With ``return_monotone=True`` returns
    ``(e_delta, monotone)``, ``monotone`` True where the curve is
    nondecreasing in m up to rounding (no step down by more than
    :data:`MONOTONE_SLACK`, 1e-12, so a plateau that went through floating
    point still counts), so that e_delta is the unique crossing; where it is
    False, say so (the curve was sampled or interpolated and is not monotone,
    so Phi may cross delta more than once and e_delta is the last crossing).
    Memory stays below a few tens of MB at any n (Phi is evaluated in chunks).
    """
    n = _curve_qubits(num_qubits)
    p = _curve_array(curve, n)
    try:
        level = float(delta)
    except (TypeError, ValueError):
        level = math.nan
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"delta must be in [0, 1], got {delta!r}")
    flat = p.reshape(-1, n + 1)
    finite = ~np.isnan(flat).any(axis=1)
    if np.any(flat[finite, 0] != 0):
        raise ValueError("curve[..., 0] = P(0) must be 0: erasing no qubit never fails")
    monotone = np.all(np.diff(flat, axis=1) >= -MONOTONE_SLACK, axis=1) & finite
    out = np.full(flat.shape[0], np.nan)
    rows = np.flatnonzero(finite)
    if level == 0.0:  # Phi(e) > 0 for every e > 0 as soon as some P(m) > 0
        out[rows] = np.where(np.any(flat[rows] > 0, axis=1), 0.0, 1.0)
    elif rows.size:
        grid = np.linspace(0.0, 1.0, 2049)
        below = _phi_grid(flat[rows], n, grid) <= level  # (rows, 2049)
        last = 2048 - np.argmax(below[:, ::-1], axis=1)  # Phi(0) = 0 <= delta: always found
        done = below[:, -1]
        out[rows[done]] = 1.0
        todo = np.flatnonzero(~done)
        lo = grid[last[todo]]
        hi = grid[last[todo] + 1]
        values = flat[rows[todo]]
        log_binom = _log_binomial(n)
        for _ in range(50 if todo.size else 0):
            mid = 0.5 * (lo + hi)
            ok = _phi_rows(values, n, mid, log_binom) <= level
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        out[rows[todo]] = lo
    shaped = out.reshape(p.shape[:-1])
    flags = monotone.reshape(p.shape[:-1])
    result: Any = float(shaped) if p.ndim == 1 else shaped
    mono: Any = bool(flags) if p.ndim == 1 else flags
    return (result, mono) if return_monotone else result


def mean_over_codes(values, holds_code) -> tuple[np.ndarray, np.ndarray, int]:
    """Ensemble mean and standard error over the trajectories that hold a code (k > 0).

    ``values`` has shape ``(t, ...)``, one row per trajectory (for example
    per-trajectory curves P_J(m), shape ``(t, c, n + 1)``); ``holds_code`` has
    shape ``(t,)``: booleans or the code dimensions k (nonzero means a code).
    Returns ``(mean, std_error, count)``: the mean over the selected rows and
    its standard error (:func:`mean_and_error`, ignoring NaN entries), each of
    shape ``values.shape[1:]``, and how many rows were selected. Rows of codes
    with k = 0 are left out, as the summaries do: such a code loses nothing
    under any erasure, and averaging it in would mix the probability of having
    a code with how well a code protects. NaN when no row is selected (the
    error also when only one is).
    """
    v = np.asarray(values, dtype=np.float64)
    h = np.asarray(holds_code)
    if v.ndim < 1 or h.shape != v.shape[:1]:
        raise ValueError(
            f"holds_code needs one entry per row of values ({v.shape[:1]}), got shape {h.shape}"
        )
    chosen = v[h != 0]
    mean, error = mean_and_error(chosen, axis=0)
    return mean, error, int(chosen.shape[0])
