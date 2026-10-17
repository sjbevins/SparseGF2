"""Dynamic finite-size scaling of the Watts–Strogatz study (specification v1.4 §5.9, §9; design §7).

At fixed ``beta``, the near-critical scaling hypothesis for a time observable ``theta`` (for
example ``tau_ell`` or a peak time ``T*``) is, as an unverified ansatz generalized from
Hashizume, Bentsen and Daley (PRR 2022, Eq. (5)) by v1.4 §5.9,

    y(n, p) = E[theta | n, beta, p] ~= n^z F((p - p_c) n^(1/nu)),

with a scaling function ``F`` of its own for every observable. In the scaled coordinates

    x = (p - p_c) n^(1/nu),   Y = y n^(-z),   dY = se n^(-z)

the points of all sizes fall on the one curve ``Y = F(x)`` at the right ``(p_c, z, nu)``.

**Collapse quality** (Houdayer and Hartmann, PRB 70, 014418 (2004)): for every point ``i`` the
master curve at ``x_i`` is estimated from the points of the *other* sizes that bracket ``x_i``
(for each other size, its two consecutive points with ``x_j <= x_i <= x_{j+1}``; no
extrapolation), by a straight line fitted with weights ``1/dY^2``, which gives ``Ybar_i`` and
its variance ``dYbar_i^2``; then

    S = (1/N) sum_i (Y_i - Ybar_i)^2 / (dY_i^2 + dYbar_i^2)

over the ``N`` points that some other size brackets. ``S`` is about 1 for a collapse within the
errors and grows as the collapse fails. It jumps where a bracket switches.

**Polynomial master curve** (ruling R-FSS, a continuous alternative): on a fixed set of points,

    chi2(p_c, z, nu) = min_a sum_i (Y_i - F_a(x_i))^2 / dY_i^2,   F_a(x) = sum_{k<=d} a_k x^k,

with ``a`` solved in closed form (weighted linear least squares), so only ``(p_c, z, nu)`` are
searched. The set is a window in ``x`` (the overlap of the sizes, or ``|x| <= x_max``) drawn at
reference parameters and kept during a search, then drawn again at the fit until it settles
(:func:`window_points`, :func:`polynomial_cost`).

**Fits.** :func:`fit_collapse` minimizes either objective (``method``) over a box of
``(p_c, z, nu)`` with :func:`minimize_bounded` (Nelder–Mead in the unbounded variables ``phi``
of ``theta = lo + (hi - lo)(1 + sin phi)/2``, from several starts; deterministic given its
seed). :func:`fit_shared` fits one ``(p_c, z, nu)`` to several observables (each keeps its own
``F``), and :func:`compare_shared_independent` tests that hypothesis against independent fits
with a bootstrap null built by graph-cluster resampling paired across ``p``.
:func:`leave_one_size_out` refits without each size; :func:`bootstrap_fit` resamples graphs once
per ``(n, beta)`` and recomputes every rate's mean from the same graphs;
:func:`collapse_diagnostics` reports coverage, the drift of the fit when the smallest size is
removed, and passes the censoring and horizon-peak fractions of every point through;
:func:`compare_methods` fits both methods and compares them against the bootstrap spread, and
:func:`polynomial_degree_scan` shows the degree's effect.

The inputs are nested means and their graph-cluster errors (:mod:`.stats`), gathered in a
:class:`ScalingData` (one observable at one ``beta``). Exploratory and confirmatory analyses are
told apart only by the recorded ``label``. ``docs/study/07-finite-size-scaling.md`` has the
mathematics, the proofs and worked examples.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from . import stats as _stats

__all__ = [
    "DEFAULT_DEGREE",
    "HOUDAYER_HARTMANN",
    "MAX_DEGREE",
    "METHODS",
    "PARAMETERS",
    "POLYNOMIAL",
    "STATUSES",
    "BootstrapFit",
    "CollapseCost",
    "CollapseDiagnostics",
    "CollapseFit",
    "DegreeScan",
    "LeaveOneSizeOut",
    "MethodComparison",
    "MinimizeResult",
    "PolynomialCost",
    "ScalingData",
    "SharedComparison",
    "SharedFit",
    "bootstrap_fit",
    "collapse_cost",
    "collapse_diagnostics",
    "compare_methods",
    "compare_shared_independent",
    "fit_collapse",
    "fit_shared",
    "leave_one_size_out",
    "minimize_bounded",
    "polynomial_cost",
    "polynomial_degree_scan",
    "scaled_coordinates",
    "scaling_data",
    "window_points",
]

#: The fitted parameters, in this order everywhere (``params`` arrays, bounds, replicates).
PARAMETERS = ("p_c", "z", "nu")

#: The collapse objectives: Houdayer and Hartmann's quality (the default) and the polynomial
#: master curve of ruling R-FSS.
HOUDAYER_HARTMANN = "houdayer_hartmann"
POLYNOMIAL = "polynomial"
METHODS = (HOUDAYER_HARTMANN, POLYNOMIAL)

#: The polynomial master curve's default degree and the largest one allowed.
DEFAULT_DEGREE = 3
MAX_DEGREE = 10

#: Fit statuses. ``ok`` and ``at_bound`` (some parameter on its box bound: widen the box)
#: and ``not_converged`` (evaluation budget spent) return parameters; ``too_few_sizes`` (fewer
#: than two sizes) and ``too_few_points`` (no parameters in the box bracket enough points) do
#: not.
STATUSES = ("ok", "at_bound", "not_converged", "too_few_points", "too_few_sizes")

_STEP = 0.25  # initial simplex step in the unbounded variables phi (radians)
_RESTART_STEP = 0.05  # the simplex rebuilt at the best vertex after convergence
_BOUND_TOL = 1e-6  # a parameter within this fraction of its box width of a bound is "at" it

_plain = _stats._plain


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------


def _vector(name: str, values: Any, size: int | None = None) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array, one entry per point")
    if size is not None and a.size != size:
        raise ValueError(f"{name} must have one entry per point ({size}); got {a.size}")
    return a


_MEAN_RTOL = 1e-9  # y against the mean of its graph means, relative to the largest |value|


def _check_graph_mean(i: int, y: float, graph_means: np.ndarray) -> None:
    """Refuse a point whose ``y`` is not the mean of its defined graph means (to rounding)."""
    defined = graph_means[~np.isnan(graph_means)]
    if defined.size == 0:
        raise ValueError(
            f"graph_means[{i}] has no defined value, so it cannot be the graph values behind "
            f"y[{i}] = {y!r}: leave the point out"
        )
    mean = float(np.sum(defined)) / defined.size
    if abs(y - mean) > _MEAN_RTOL * float(np.max(np.abs(defined))):
        raise ValueError(
            f"y[{i}] = {y!r} is not the mean of graph_means[{i}] ({mean!r}): the bootstrap "
            "recomputes every replicate as the mean of resampled graph means, so graph_means must "
            "be the per-graph values whose equal-weight mean is y (a nested mean, or a restricted "
            "mean as the nested mean of min(tau, T_max)); a median or another estimator cannot "
            "carry graph_means: leave graphs and graph_means out (no bootstrap)"
        )


@dataclass(frozen=True, eq=False)
class ScalingData:
    """The points ``(n, p, y, se)`` of one observable at one ``beta``, ready for a collapse.

    ``y`` is the ensemble estimate of the time observable at size ``n`` and rate ``p`` (a
    nested mean; under censoring a survival-based estimate: say which in ``estimator``) and
    ``se`` its standard error (positive). ``graphs`` and ``graph_means`` (one array per point)
    are the per-graph values behind ``y``: they let :func:`bootstrap_fit`,
    :func:`compare_shared_independent` and :func:`compare_methods` resample graphs. Every point
    of a size must have the same graph labels, because the rates were measured on the same
    graphs, and ``y`` must be the equal-weight mean of the defined ``graph_means`` of its point
    (to rounding; a point that is not is refused), because the bootstrap recomputes every
    replicate as the mean of the resampled graph means. So only a graph-equal mean can carry
    graph values: a nested mean, or the restricted mean ``E[min(tau, T_max)]``, which is the
    nested mean of ``min(tau, T_max)`` (``docs/study/06-statistics.md`` §6.5). A Kaplan-Meier
    median has no per-graph values whose mean it is: such data can be fitted, with an error
    supplied by the user, but not bootstrapped here.
    ``censored_fraction`` and ``horizon_fraction`` (per point, optional) are passed through to
    the diagnostics. ``observable``, ``beta``, ``label`` ("exploratory" or "confirmatory") and
    ``estimator`` are recorded with every fit.
    """

    n: np.ndarray
    p: np.ndarray
    y: np.ndarray
    se: np.ndarray
    observable: str = "y"
    beta: str | None = None
    label: str | None = None
    estimator: str = "nested mean"
    censored_fraction: np.ndarray | None = None
    horizon_fraction: np.ndarray | None = None
    graphs: tuple[np.ndarray, ...] | None = None
    graph_means: tuple[np.ndarray, ...] | None = None

    def __post_init__(self) -> None:
        n = np.asarray(self.n)
        if n.ndim != 1 or n.size == 0:
            raise ValueError("n must be a non-empty one-dimensional array of system sizes")
        if n.dtype == np.bool_ or not (
            np.issubdtype(n.dtype, np.integer)
            or (np.issubdtype(n.dtype, np.floating) and np.all(n == np.floor(n)))
        ):
            raise ValueError("n must hold integer system sizes")
        if np.any(n < 1):
            raise ValueError("system sizes n must be positive")
        size = n.size
        p = _vector("p", self.p, size)
        y = _vector("y", self.y, size)
        se = _vector("se", self.se, size)
        if not np.all(np.isfinite(p)):
            raise ValueError("p must be finite")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must be finite: leave out the points without a value")
        if not np.all(np.isfinite(se) & (se > 0)):
            raise ValueError(
                "se must be positive and finite (the collapse weights points by 1/se^2); a point "
                "whose graph means are all equal has se = 0: give it a floor, for example the "
                "resolution of the depth grid"
            )
        set_ = object.__setattr__
        set_(self, "n", n.astype(np.int64))
        set_(self, "p", p)
        set_(self, "y", y)
        set_(self, "se", se)
        for name in ("censored_fraction", "horizon_fraction"):
            value = getattr(self, name)
            if value is not None:
                frac = _vector(name, value, size)
                if not np.all((frac >= 0) & (frac <= 1)):
                    raise ValueError(f"{name} must lie in [0, 1]")
                set_(self, name, frac)
        if (self.graphs is None) != (self.graph_means is None):
            raise ValueError("graphs and graph_means go together")
        if self.graphs is not None and self.graph_means is not None:
            if len(self.graphs) != size or len(self.graph_means) != size:
                raise ValueError("graphs and graph_means need one array per point")
            labels, values = [], []
            for i, (lab, gm) in enumerate(zip(self.graphs, self.graph_means, strict=True)):
                lab = _stats._labels(f"graphs[{i}]", lab)
                gm = _vector(f"graph_means[{i}]", gm, lab.size)
                if lab.size == 0 or np.any(np.diff(lab) <= 0):
                    raise ValueError(f"graphs[{i}] must be distinct increasing graph labels")
                if np.isinf(gm).any():
                    raise ValueError(f"graph_means[{i}] must be finite or NaN")
                _check_graph_mean(i, float(y[i]), gm)
                labels.append(lab)
                values.append(gm)
            set_(self, "graphs", tuple(labels))
            set_(self, "graph_means", tuple(values))
        if self.beta is not None and not isinstance(self.beta, str):
            set_(self, "beta", repr(float(self.beta)))

    @property
    def size(self) -> int:
        """The number of points."""
        return int(self.n.size)

    @property
    def sizes(self) -> np.ndarray:
        """The distinct system sizes, increasing."""
        return np.unique(self.n)

    def subset(self, keep: ArrayLike) -> ScalingData:
        """The points where ``keep`` (a boolean mask) is true, with their metadata."""
        mask = np.asarray(keep, dtype=bool)
        if mask.shape != (self.size,):
            raise ValueError(f"keep must be a boolean mask with one entry per point ({self.size})")
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            raise ValueError("the subset has no points")

        def pick(a: np.ndarray | None) -> np.ndarray | None:
            return None if a is None else a[idx]

        return ScalingData(
            n=self.n[idx],
            p=self.p[idx],
            y=self.y[idx],
            se=self.se[idx],
            observable=self.observable,
            beta=self.beta,
            label=self.label,
            estimator=self.estimator,
            censored_fraction=pick(self.censored_fraction),
            horizon_fraction=pick(self.horizon_fraction),
            graphs=None if self.graphs is None else tuple(self.graphs[i] for i in idx),
            graph_means=None
            if self.graph_means is None
            else tuple(self.graph_means[i] for i in idx),
        )

    def to_dict(self) -> dict[str, Any]:
        """The data as plain Python (lists and numbers; NaN as None)."""
        out: dict[str, Any] = _plain(self)
        return out


def scaling_data(
    n: ArrayLike,
    p: ArrayLike,
    means: Sequence[_stats.GraphMean],
    *,
    observable: str = "y",
    beta: str | float | None = None,
    label: str | None = None,
    estimator: str = "nested mean",
    censored_fraction: ArrayLike | None = None,
    horizon_fraction: ArrayLike | None = None,
) -> ScalingData:
    """:class:`ScalingData` from one scalar :class:`~.stats.GraphMean` per point.

    ``y`` and ``se`` are each estimate's ``mean`` and graph-cluster ``se``; the per-graph
    values are kept for the bootstrap. A point without a defined mean, or with fewer than two
    contributing graphs (no error), is refused: leave it out.
    """
    sizes = np.asarray(n)
    rates = np.asarray(p, dtype=np.float64)
    if len(means) != sizes.size or rates.size != sizes.size:
        raise ValueError("n, p and means need one entry per point")
    y, se = [], []
    for i, m in enumerate(means):
        if not isinstance(m, _stats.GraphMean) or m.mean.shape != ():
            raise ValueError(f"means[{i}] must be a scalar GraphMean (nested_mean of one value)")
        if not (np.isfinite(m.mean) and np.isfinite(m.se)):
            raise ValueError(
                f"means[{i}] (n = {sizes[i]}, p = {rates[i]}) has no defined mean and error "
                "(fewer than two contributing graphs): leave the point out"
            )
        y.append(float(m.mean))
        se.append(float(m.se))
    return ScalingData(
        n=sizes,
        p=rates,
        y=np.array(y),
        se=np.array(se),
        observable=observable,
        beta=None if beta is None else (beta if isinstance(beta, str) else repr(float(beta))),
        label=label,
        estimator=estimator,
        censored_fraction=None if censored_fraction is None else np.asarray(censored_fraction),
        horizon_fraction=None if horizon_fraction is None else np.asarray(horizon_fraction),
        graphs=tuple(m.graphs for m in means),
        graph_means=tuple(m.graph_means for m in means),
    )


def _check_params(p_c: float, z: float, nu: float) -> None:
    if not (math.isfinite(p_c) and math.isfinite(z) and math.isfinite(nu) and nu > 0):
        raise ValueError(f"p_c, z must be finite and nu positive; got {(p_c, z, nu)}")


def scaled_coordinates(
    data: ScalingData, p_c: float, z: float, nu: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``x = (p - p_c) n^(1/nu)``, ``Y = y n^(-z)``, ``dY = se n^(-z)`` for every point."""
    _check_params(p_c, z, nu)
    sizes = data.n.astype(np.float64)
    amp = np.power(sizes, -z)
    return (data.p - p_c) * np.power(sizes, 1.0 / nu), data.y * amp, data.se * amp


# ---------------------------------------------------------------------------------------------
# the collapse quality
# ---------------------------------------------------------------------------------------------


class _Collapse:
    """Points prepared for repeated evaluation of the Houdayer–Hartmann quality."""

    def __init__(self, n: np.ndarray, p: np.ndarray, y: np.ndarray, se: np.ndarray) -> None:
        self.p, self.y, self.se = p, y, se
        sizes, si = np.unique(n, return_inverse=True)
        self.sizes = sizes
        self.si = si
        num, s_count = n.size, sizes.size
        # within a size, x increases with p for any parameters: order by p (ties by y, se)
        order = np.lexsort((se, y, p, si))
        counts = np.bincount(si, minlength=s_count)
        start = np.concatenate(([0], np.cumsum(counts)[:-1]))
        rank = np.empty(num, dtype=np.int64)
        rank[order] = np.arange(num) - start[si[order]]
        width = int(np.max(counts))
        self.pad_p = np.full((s_count, width), np.nan)
        self.pad_y = np.full((s_count, width), np.nan)
        self.pad_e = np.full((s_count, width), np.nan)
        self.pad_p[si, rank] = p
        self.pad_y[si, rank] = y
        self.pad_e[si, rank] = se
        self.counts = counts
        self.width = width
        self.n_float = sizes.astype(np.float64)
        self.other = si[:, None] != np.arange(s_count)[None, :]
        self.rows = np.broadcast_to(np.arange(s_count), (num, s_count))
        self.single_point = counts == 1

    @classmethod
    def of(cls, data: ScalingData) -> _Collapse:
        return cls(data.n, data.p, data.y, data.se)

    def chi2(self, p_c: float, z: float, nu: float) -> np.ndarray:
        """``(Y_i - Ybar_i)^2 / (dY_i^2 + dYbar_i^2)`` per point; NaN where not bracketed."""
        num, s_count = self.si.size, self.sizes.size
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scale = np.power(self.n_float, 1.0 / nu)  # (pow is exact for n^1, n^0, ...)
            amp = np.power(self.n_float, -z)
            if not (np.all(np.isfinite(scale)) and np.all(np.isfinite(amp)) and np.all(amp > 0)):
                return np.full(num, np.nan)
            si = self.si
            x = (self.p - p_c) * scale[si]
            big_y = self.y * amp[si]
            big_dy = self.se * amp[si]
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(big_y)) and np.all(big_dy > 0)):
                return np.full(num, np.nan)
            xs = (self.pad_p - p_c) * scale[:, None]
            ys = self.pad_y * amp[:, None]
            es = self.pad_e * amp[:, None]
            # last[i, s]: the last point of size s with x_s <= x_i (-1 if none); each size's
            # abscissas are sorted, since x increases with p at fixed n
            last = np.empty((num, s_count), dtype=np.int64)
            for s in range(s_count):
                last[:, s] = np.searchsorted(xs[s, : self.counts[s]], x, side="right") - 1
            m = self.counts[None, :]
            top = last >= m - 1
            top_x = xs[np.arange(s_count), self.counts - 1]
            at_top = x[:, None] == top_x[None, :]
            valid = self.other & (last >= 0) & (~top | at_top)
            single = top & self.single_point[None, :]
            left = np.where(single, 0, np.where(top, m - 2, last))
            right = np.where(single, 0, np.where(top, m - 1, last + 1))
            li = np.clip(left, 0, self.width - 1)
            ri = np.clip(right, 0, self.width - 1)
            rows = self.rows
            mask = np.concatenate([valid, valid & ~single], axis=1)
            u = np.concatenate([xs[rows, li], xs[rows, ri]], axis=1) - x[:, None]
            yy = np.concatenate([ys[rows, li], ys[rows, ri]], axis=1)
            ee = np.concatenate([es[rows, li], es[rows, ri]], axis=1)
            w = np.where(mask, 1.0 / np.where(mask, ee, 1.0) ** 2, 0.0)
            u = np.where(mask, u, 0.0)
            yy = np.where(mask, yy, 0.0)
            k = w.sum(axis=1)
            used = k > 0
            k_safe = np.where(used, k, 1.0)
            ubar = (w * u).sum(axis=1) / k_safe
            ybar = (w * yy).sum(axis=1) / k_safe
            du = np.where(mask, u - ubar[:, None], 0.0)
            dy = np.where(mask, yy - ybar[:, None], 0.0)
            suu = (w * du * du).sum(axis=1)
            suy = (w * du * dy).sum(axis=1)
            line = suu > 0
            suu_safe = np.where(line, suu, 1.0)
            slope = np.where(line, suy / suu_safe, 0.0)
            master = ybar - slope * ubar
            master_var = 1.0 / k_safe + np.where(line, ubar * ubar / suu_safe, 0.0)
            chi2 = (big_y - master) ** 2 / (big_dy * big_dy + master_var)
        return np.where(used, chi2, np.nan)


@dataclass(frozen=True, eq=False)
class CollapseCost:
    """The Houdayer–Hartmann quality at given parameters.

    ``cost`` is the mean of ``chi2`` over the ``points_used`` bracketed points (NaN when none),
    ``chi2`` the normalized squared deviation of every point (NaN where no other size brackets
    it) and ``bracketed`` its mask.
    """

    cost: float
    points_used: int
    total_points: int
    chi2: np.ndarray
    bracketed: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _plain(self)
        return out


def collapse_cost(data: ScalingData, p_c: float, z: float, nu: float) -> CollapseCost:
    """The Houdayer–Hartmann collapse quality ``S`` of ``data`` at ``(p_c, z, nu)``.

    For point ``i`` of size ``n_i``, every other size ``n'`` with points on both sides of ``x_i``
    contributes its two consecutive points ``x_j <= x_i <= x_{j+1}`` (when ``x_i`` equals the
    largest ``x`` of ``n'``, its last two points; a size with one point contributes it only when
    its ``x`` equals ``x_i``). Through these points ``(u_l, Y_l)``, ``u_l = x_l - x_i``, with
    weights ``w_l = 1/dY_l^2``, the weighted least-squares line gives at ``u = 0``

        Ybar = ybar - b ubar,   dYbar^2 = 1/K + ubar^2 / S_uu,

    with ``K = sum w``, ``ubar, ybar`` the weighted means, ``S_uu = sum w (u - ubar)^2`` and
    ``b = sum w (u - ubar)(Y - ybar) / S_uu`` (``Ybar = ybar``, ``dYbar^2 = 1/K`` when every
    ``u_l = 0``). Then ``chi2_i = (Y_i - Ybar)^2 / (dY_i^2 + dYbar^2)`` and ``S`` is their mean.
    """
    _check_params(p_c, z, nu)
    chi2 = _Collapse.of(data).chi2(p_c, z, nu)
    bracketed = ~np.isnan(chi2)
    used = int(bracketed.sum())
    cost = float(np.mean(chi2[bracketed])) if used else math.nan
    return CollapseCost(
        cost=cost, points_used=used, total_points=data.size, chi2=chi2, bracketed=bracketed
    )


# ---------------------------------------------------------------------------------------------
# the polynomial master curve (ruling R-FSS)
# ---------------------------------------------------------------------------------------------


def window_points(
    data: ScalingData, p_c: float, z: float, nu: float, window: str | float = "overlap"
) -> np.ndarray:
    """The points in the polynomial method's window at ``(p_c, z, nu)``, as a boolean mask.

    ``window="overlap"``: the scaled abscissas that every size covers,
    ``x in [max_n min_i x_ni, min_n max_i x_ni]`` (no point when the sizes' ranges do not
    overlap); a number ``x_max > 0``: ``|x| <= x_max``. Both ends are included. A fit draws the
    window once and keeps that point set during a search (so its objective is continuous in the
    parameters), then draws it again at the fitted parameters until it no longer changes. Where
    ``n^(1/nu)`` overflows the window is empty. (``z`` does not enter the window.)
    """
    _check_params(p_c, z, nu)
    rule = _window(window)
    with np.errstate(over="ignore", invalid="ignore"):
        x = (data.p - p_c) * np.power(data.n.astype(np.float64), 1.0 / nu)
    if not np.all(np.isfinite(x)):
        return np.zeros(data.size, dtype=bool)
    if rule == "overlap":
        lo = max(float(x[data.n == s].min()) for s in data.sizes)
        hi = min(float(x[data.n == s].max()) for s in data.sizes)
        return (x >= lo) & (x <= hi)
    return np.abs(x) <= float(rule)


@dataclass(frozen=True, eq=False)
class _PolySolution:
    """The weighted least-squares master curve at fixed ``(p_c, z, nu)``.

    In the basis ``(x / x_scale)^k``: coefficients ``coef_scaled`` and their covariance
    ``cov_scaled`` (for unit-variance whitened residuals); ``residuals`` are
    ``(Y_i - F(x_i)) / dY_i``.
    """

    chi2: float
    residuals: np.ndarray
    x: np.ndarray
    big_y: np.ndarray
    sigma: np.ndarray
    x_scale: float
    coef_scaled: np.ndarray
    cov_scaled: np.ndarray
    rank: int

    @property
    def coefficients(self) -> np.ndarray:
        """``a_k`` of ``F(x) = sum_k a_k x^k``."""
        return self.coef_scaled / self.x_scale ** np.arange(self.coef_scaled.size)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``F(x)`` and the variance of the fitted value at each ``x``."""
        basis = np.vander(
            np.asarray(x, dtype=np.float64) / self.x_scale, self.coef_scaled.size, increasing=True
        )
        return basis @ self.coef_scaled, np.einsum("ij,jk,ik->i", basis, self.cov_scaled, basis)


class _Polynomial:
    """One observable's points on a fixed set, for the polynomial master-curve objective."""

    def __init__(
        self, n: np.ndarray, p: np.ndarray, y: np.ndarray, se: np.ndarray, degree: int
    ) -> None:
        self.n, self.p, self.y, self.se = n, p, y, se
        self.n_float = n.astype(np.float64)
        self.degree = degree
        self.count = int(n.size)

    @classmethod
    def subset(cls, data: ScalingData, mask: np.ndarray, degree: int) -> _Polynomial:
        return cls(data.n[mask], data.p[mask], data.y[mask], data.se[mask], degree)

    def solve(self, p_c: float, z: float, nu: float) -> _PolySolution | None:
        """The closed-form fit of ``F`` at ``(p_c, z, nu)``; None where it is undefined."""
        if self.count == 0:
            return None
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            x = (self.p - p_c) * np.power(self.n_float, 1.0 / nu)
            amp = np.power(self.n_float, -z)
            big_y = self.y * amp
            sigma = self.se * amp
            x_scale = float(np.max(np.abs(x)))
            if not math.isfinite(x_scale):
                return None
            x_scale = x_scale if x_scale > 0 else 1.0
            a = np.vander(x / x_scale, self.degree + 1, increasing=True) / sigma[:, None]
            b = big_y / sigma
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(sigma > 0)):
            return None
        try:
            u, s, vt = np.linalg.svd(a, full_matrices=False)
        except np.linalg.LinAlgError:  # pragma: no cover - LAPACK did not converge
            return None
        keep = s > s[0] * max(a.shape) * np.finfo(np.float64).eps
        inv = 1.0 / s[keep]
        coef = vt[keep].T @ (inv * (u[:, keep].T @ b))
        resid = b - a @ coef
        cov = (vt[keep].T * inv**2) @ vt[keep]
        return _PolySolution(
            chi2=float(resid @ resid),
            residuals=resid,
            x=x,
            big_y=big_y,
            sigma=sigma,
            x_scale=x_scale,
            coef_scaled=coef,
            cov_scaled=cov,
            rank=int(keep.sum()),
        )

    def chi2(self, p_c: float, z: float, nu: float) -> float:
        solution = self.solve(p_c, z, nu)
        return math.inf if solution is None else solution.chi2


@dataclass(frozen=True, eq=False)
class PolynomialCost:
    """The polynomial master-curve objective at given parameters on one point set.

    ``chi2 = min_a sum_i (Y_i - F_a(x_i))^2 / dY_i^2`` over the ``points_used`` points of
    ``selected``, ``F_a(x) = sum_{k=0}^{degree} a_k x^k`` (``coefficients``), solved in closed
    form; ``dof = points_used - (degree + 1)`` (the parameters are given here, not fitted) and
    ``chi2_per_dof``; ``residuals`` are ``(Y_i - F(x_i)) / dY_i`` (NaN outside the set).
    ``chi2`` is NaN where the objective is undefined (overflow, or no point).
    """

    chi2: float
    dof: int
    chi2_per_dof: float
    points_used: int
    total_points: int
    degree: int
    selected: np.ndarray
    residuals: np.ndarray
    coefficients: np.ndarray | None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _plain(self)
        return out


def polynomial_cost(
    data: ScalingData,
    p_c: float,
    z: float,
    nu: float,
    *,
    degree: int = DEFAULT_DEGREE,
    selected: ArrayLike | None = None,
    window: str | float = "overlap",
) -> PolynomialCost:
    """The polynomial objective of ruling R-FSS at ``(p_c, z, nu)``.

    The points are ``selected`` (a boolean mask) or, without it, :func:`window_points` at these
    parameters. In scaled coordinates the master curve ``F(x) = sum_k a_k x^k`` minimizing
    ``sum_i (Y_i - F(x_i))^2 / dY_i^2`` is a weighted linear least-squares problem in ``a``:
    with ``A_ik = x_i^k / dY_i`` and ``b_i = Y_i / dY_i``, ``a = A^+ b`` (the pseudoinverse,
    from the singular value decomposition of ``A`` in the better-conditioned basis
    ``(x / max|x|)^k``) and ``chi2 = |b - A a|^2``. Since ``(Y - F)/dY = (y - n^z F)/s``, the
    objective is in the units of the data's own errors.
    """
    _check_params(p_c, z, nu)
    deg = _degree(degree)
    if selected is None:
        mask = window_points(data, p_c, z, nu, window)
    else:
        mask = np.asarray(selected)
        if mask.shape != (data.size,) or mask.dtype != np.bool_:
            raise ValueError(
                f"selected must be a boolean mask with one entry per point ({data.size})"
            )
    solution = _Polynomial.subset(data, mask, deg).solve(p_c, z, nu)
    residuals = np.full(data.size, np.nan)
    used = int(mask.sum())
    dof = used - (deg + 1)
    if solution is None:
        return PolynomialCost(math.nan, dof, math.nan, used, data.size, deg, mask, residuals, None)
    residuals[mask] = solution.residuals
    return PolynomialCost(
        chi2=solution.chi2,
        dof=dof,
        chi2_per_dof=solution.chi2 / dof if dof > 0 else math.nan,
        points_used=used,
        total_points=data.size,
        degree=deg,
        selected=mask,
        residuals=residuals,
        coefficients=solution.coefficients,
    )


# ---------------------------------------------------------------------------------------------
# the bounded Nelder–Mead
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class MinimizeResult:
    """The best point found by :func:`minimize_bounded`.

    ``x`` and ``fun``; ``converged`` (the best start met the tolerances within its budget);
    ``at_bound`` (per coordinate, within ``1e-6`` of the box width of a bound; never for a fixed
    coordinate); ``evaluations`` over all starts; ``starts`` the number of starts run.
    """

    x: np.ndarray
    fun: float
    converged: bool
    at_bound: np.ndarray
    evaluations: int
    starts: int


def _nelder_mead(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    step: np.ndarray,
    xatol: float,
    fatol: float,
    budget: int,
) -> tuple[np.ndarray, float, int, bool]:
    """Nelder–Mead (Lagarias et al. 1998: reflection 1, expansion 2, contraction 1/2, shrink 1/2).

    Stops when the simplex's vertices are within ``xatol`` of the best one in every coordinate
    and their values within ``fatol * max(1, |f_best|)`` of it, or after ``budget``
    evaluations. Vertices are ordered by a stable sort, so ties keep their order.
    """
    dim = x0.size
    simplex = np.vstack([x0, x0 + np.diag(step)])
    values = np.array([f(v) for v in simplex])
    used = dim + 1
    while True:
        order = np.argsort(values, kind="stable")
        simplex, values = simplex[order], values[order]
        best = values[0]
        if not math.isfinite(best):
            return simplex[0], best, used, False
        spread_x = float(np.max(np.abs(simplex[1:] - simplex[0])))
        spread_f = float(values[-1] - best) if math.isfinite(values[-1]) else math.inf
        if spread_x <= xatol and spread_f <= fatol * max(1.0, abs(best)):
            return simplex[0], best, used, True
        if used >= budget:
            return simplex[0], best, used, False
        centroid = simplex[:-1].mean(axis=0)
        worst = simplex[-1]
        xr = centroid + (centroid - worst)
        fr = f(xr)
        used += 1
        if fr < values[0]:
            xe = centroid + 2.0 * (xr - centroid)
            fe = f(xe)
            used += 1
            simplex[-1], values[-1] = (xe, fe) if fe < fr else (xr, fr)
            continue
        if fr < values[-2]:
            simplex[-1], values[-1] = xr, fr
            continue
        if fr < values[-1]:  # outside contraction
            xc = centroid + 0.5 * (xr - centroid)
            fc = f(xc)
            used += 1
            if fc <= fr:
                simplex[-1], values[-1] = xc, fc
                continue
        else:  # inside contraction
            xc = centroid + 0.5 * (worst - centroid)
            fc = f(xc)
            used += 1
            if fc < values[-1]:
                simplex[-1], values[-1] = xc, fc
                continue
        simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])  # shrink
        values[1:] = [f(v) for v in simplex[1:]]
        used += dim


def _box(bounds: Any, names: Sequence[str] | None = None) -> np.ndarray:
    """``bounds`` as a ``(d, 2)`` array of finite ``lo <= hi`` pairs."""
    if isinstance(bounds, Mapping):
        if names is None:
            raise ValueError("bounds given as a mapping need parameter names")
        unknown = sorted(set(bounds) - set(names))
        missing = [k for k in names if k not in bounds]
        if unknown or missing:
            raise ValueError(
                f"bounds must give (lo, hi) for exactly {list(names)}; missing {missing}, "
                f"unknown {unknown}"
            )
        rows = [bounds[k] for k in names]
    else:
        rows = list(bounds)
    try:
        box = np.array(rows, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("bounds must be (lo, hi) pairs of numbers") from exc
    if box.ndim != 2 or box.shape[1] != 2 or (names is not None and box.shape[0] != len(names)):
        want = f"{len(names)} " if names is not None else ""
        raise ValueError(f"bounds must be {want}(lo, hi) pairs")
    if not np.all(np.isfinite(box)) or np.any(box[:, 0] > box[:, 1]):
        raise ValueError("every bound must be a finite pair with lo <= hi")
    return box


def minimize_bounded(
    fun: Callable[[np.ndarray], float],
    bounds: Any,
    *,
    initial: ArrayLike | None = None,
    starts: int = 8,
    seed: Any = 0,
    xatol: float = 1e-10,
    fatol: float = 1e-12,
    max_evaluations: int = 2000,
    restarts: int = 2,
) -> MinimizeResult:
    """Minimize ``fun`` over the box ``bounds`` by Nelder–Mead from several starts.

    Each free coordinate (``lo < hi``) is written ``theta = lo + (hi - lo)(1 + sin phi)/2`` with
    ``phi`` unbounded: a smooth map onto the closed box that reaches both bounds (07 §6.1,
    Proposition 5), so Nelder–Mead runs without constraints in ``phi``. A coordinate with
    ``lo == hi`` is held fixed. Non-finite values of ``fun`` count as ``+inf``.

    Starts: the rows of ``initial`` (one point or several) or, without them, the center of the
    box; then ``starts`` points uniform in the box from the stable stream of ``seed``
    (:func:`.stats.graph_bootstrap`'s generator). From each start the simplex has steps of
    0.25 in ``phi`` toward the interior; after it converges it is rebuilt at the best vertex up
    to ``restarts`` times while that still improves. ``max_evaluations`` bounds each start. The
    best start wins (the first on ties). Everything is deterministic given ``seed``.
    """
    box = _box(bounds)
    lo, hi = box[:, 0], box[:, 1]
    free = hi > lo
    width = hi - lo
    if isinstance(starts, bool) or not isinstance(starts, (int, np.integer)) or starts < 0:
        raise ValueError(f"starts must be a non-negative integer, not {starts!r}")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")

    def theta_of(phi: np.ndarray) -> np.ndarray:
        theta = lo.copy()
        theta[free] = lo[free] + width[free] * (1.0 + np.sin(phi)) / 2.0
        return np.clip(theta, lo, hi)

    def phi_of(theta: np.ndarray) -> np.ndarray:
        return np.arcsin(np.clip(2.0 * (theta[free] - lo[free]) / width[free] - 1.0, -1.0, 1.0))

    evaluations = 0

    def value(phi: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        v = float(fun(theta_of(phi)))
        return v if math.isfinite(v) else math.inf

    points: list[np.ndarray] = []
    if initial is not None:
        init = np.atleast_2d(np.asarray(initial, dtype=np.float64))
        if init.shape[1] != lo.size:
            raise ValueError(f"initial points must have {lo.size} coordinates")
        for row in init:
            if not (np.all(np.isfinite(row)) and np.all(row >= lo) and np.all(row <= hi)):
                raise ValueError(f"initial point {row.tolist()} is outside the box")
            points.append(row)
    else:
        points.append((lo + hi) / 2.0)
    if starts:
        u = _stats._uniforms(seed, int(starts) * lo.size).reshape(int(starts), lo.size)
        points.extend(lo + width * row for row in u)

    if not free.any():
        fixed = float(fun(lo.copy()))
        return MinimizeResult(
            x=lo.copy(),
            fun=fixed if math.isfinite(fixed) else math.inf,
            converged=True,
            at_bound=np.zeros(lo.size, dtype=bool),
            evaluations=1,
            starts=1,
        )
    best_phi, best_f, best_conv = phi_of(points[0]), math.inf, False
    for start in points:
        phi0 = phi_of(start)
        step = np.where(phi0 > 0, -_STEP, _STEP)
        phi, f, _, conv = _nelder_mead(value, phi0, step, xatol, fatol, max_evaluations)
        if f < best_f:
            best_phi, best_f, best_conv = phi, f, conv
    # polish the winner: rebuild a small simplex at its best vertex while that still improves
    for _ in range(restarts):
        if not math.isfinite(best_f):
            break
        step = np.where(best_phi > 0, -_RESTART_STEP, _RESTART_STEP)
        phi, f, _, conv = _nelder_mead(value, best_phi, step, xatol, fatol, max_evaluations)
        improved = f < best_f - fatol * max(1.0, abs(best_f))
        if f <= best_f:
            best_phi, best_f, best_conv = phi, f, conv
        if not improved:
            break
    best_theta = theta_of(best_phi)
    at_lo = free & (best_theta - lo <= _BOUND_TOL * width)
    at_hi = free & (hi - best_theta <= _BOUND_TOL * width)
    return MinimizeResult(
        x=np.asarray(best_theta, dtype=np.float64),
        fun=best_f,
        converged=bool(best_conv) and math.isfinite(best_f),
        at_bound=at_lo | at_hi,
        evaluations=evaluations,
        starts=len(points),
    )


# ---------------------------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------------------------

_USABLE = ("ok", "at_bound", "not_converged")


def _min_points(min_points: int | None, total: int) -> int:
    if min_points is None:
        return max(3, math.ceil(total / 2))
    if isinstance(min_points, bool) or not isinstance(min_points, (int, np.integer)):
        raise ValueError(f"min_points must be a positive integer or None, not {min_points!r}")
    if min_points < 1:
        raise ValueError("min_points must be at least 1")
    return int(min_points)


def _objective(collapses: Sequence[_Collapse], needs: Sequence[int]) -> Callable[..., float]:
    """The pooled Houdayer-Hartmann quality: the mean ``chi2`` over every observable's
    bracketed points; ``+inf`` when some observable has fewer bracketed points than it needs.
    """

    def objective(theta: np.ndarray) -> float:
        p_c, z, nu = (float(v) for v in theta)
        total, used = 0.0, 0
        for collapse, need in zip(collapses, needs, strict=True):
            chi2 = collapse.chi2(p_c, z, nu)
            ok = ~np.isnan(chi2)
            count = int(ok.sum())
            if count < need:
                return math.inf
            total += float(np.sum(chi2[ok]))
            used += count
        return total / used

    return objective


def _parameter_box(bounds: Any) -> np.ndarray:
    box = _box(bounds, PARAMETERS)
    if box[2, 0] <= 0:
        raise ValueError("the lower bound of nu must be positive")
    return box


def _method(method: Any) -> str:
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}, not {method!r}")
    return str(method)


def _degree(degree: Any) -> int:
    if (
        isinstance(degree, bool)
        or not isinstance(degree, (int, np.integer))
        or not 1 <= degree <= MAX_DEGREE
    ):
        raise ValueError(f"degree must be an integer from 1 to {MAX_DEGREE}, not {degree!r}")
    return int(degree)


def _window(window: Any) -> str | float:
    if isinstance(window, str):
        if window != "overlap":
            raise ValueError(f"window must be 'overlap' or a positive x_max, not {window!r}")
        return window
    if isinstance(window, bool) or not isinstance(window, (int, float, np.integer, np.floating)):
        raise ValueError(f"window must be 'overlap' or a positive x_max, not {window!r}")
    value = float(window)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"window must be 'overlap' or a positive x_max, not {window!r}")
    return value


@dataclass(frozen=True, eq=False)
class _Settings:
    box: np.ndarray
    starts: int
    seed: Any
    min_points: int | None
    max_evaluations: int
    method: str = HOUDAYER_HARTMANN
    degree: int = DEFAULT_DEGREE
    window: str | float = "overlap"
    window_iterations: int = 5


def _settings(
    bounds: Any,
    starts: int,
    seed: Any,
    min_points: int | None,
    max_evaluations: int,
    method: str = HOUDAYER_HARTMANN,
    degree: int = DEFAULT_DEGREE,
    window: str | float = "overlap",
    window_iterations: int = 5,
) -> _Settings:
    box = _parameter_box(bounds)
    if isinstance(starts, bool) or not isinstance(starts, (int, np.integer)) or starts < 0:
        raise ValueError(f"starts must be a non-negative integer, not {starts!r}")
    if min_points is not None:
        _min_points(min_points, 1)
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, (int, np.integer)):
        raise ValueError("max_evaluations must be a positive integer")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")
    if (
        isinstance(window_iterations, bool)
        or not isinstance(window_iterations, (int, np.integer))
        or window_iterations < 1
    ):
        raise ValueError(f"window_iterations must be a positive integer, not {window_iterations!r}")
    _stats._seed_sequence(seed)
    return _Settings(
        box,
        int(starts),
        seed,
        min_points,
        int(max_evaluations),
        _method(method),
        _degree(degree),
        _window(window),
        int(window_iterations),
    )


@dataclass(frozen=True)
class _Outcome:
    params: np.ndarray
    cost: float
    used: tuple[int, ...]
    status: str
    message: str
    at_bound: tuple[str, ...]
    evaluations: int
    starts: int
    needs: tuple[int, ...]
    chi2: float = math.nan
    dof: int = 0


def _no_fit(
    count: int, status: str, message: str, needs: tuple[int, ...], evaluations: int = 0
) -> _Outcome:
    return _Outcome(
        np.full(3, np.nan),
        math.nan,
        tuple(0 for _ in range(count)),
        status,
        message,
        (),
        evaluations,
        0,
        needs,
    )


def _status(result: MinimizeResult) -> tuple[str, str, tuple[str, ...]]:
    at_bound = tuple(name for name, flag in zip(PARAMETERS, result.at_bound, strict=True) if flag)
    if not result.converged:
        return "not_converged", "the evaluation budget ran out before convergence", at_bound
    if at_bound:
        return (
            "at_bound",
            f"{', '.join(at_bound)} at a bound of the box: widen it and refit",
            at_bound,
        )
    return "ok", "", at_bound


_TOO_FEW_SIZES = (
    "a collapse compares sizes: every observable needs at least two sizes (three to constrain "
    "p_c, z and nu); got {sizes}"
)


def _fit_hh(
    collapses: Sequence[_Collapse],
    totals: Sequence[int],
    settings: _Settings,
    initial: ArrayLike | None,
    starts: int | None = None,
) -> _Outcome:
    """Minimize the (pooled) Houdayer-Hartmann quality."""
    needs = tuple(_min_points(settings.min_points, t) for t in totals)
    sizes = [int(c.sizes.size) for c in collapses]
    if min(sizes) < 2:
        return _no_fit(len(collapses), "too_few_sizes", _TOO_FEW_SIZES.format(sizes=sizes), needs)
    short = [t for t, need in zip(totals, needs, strict=True) if t < need]
    if short:
        return _no_fit(
            len(collapses),
            "too_few_points",
            f"fewer points ({short}) than the {needs} bracketed points a fit needs (min_points)",
            needs,
        )
    result = minimize_bounded(
        _objective(collapses, needs),
        settings.box,
        initial=initial,
        starts=settings.starts if starts is None else starts,
        seed=settings.seed,
        max_evaluations=settings.max_evaluations,
    )
    if not math.isfinite(result.fun):
        return _no_fit(
            len(collapses),
            "too_few_points",
            f"no parameters in the box bracket at least {list(needs)} points (min_points) of "
            "every observable: the sizes' scaled ranges do not overlap enough; widen the box or "
            "the rate window, or add rates",
            needs,
            result.evaluations,
        )
    params = result.x
    used = tuple(int(np.sum(~np.isnan(c.chi2(*(float(v) for v in params))))) for c in collapses)
    status, message, at_bound = _status(result)
    return _Outcome(
        params,
        result.fun,
        used,
        status,
        message,
        at_bound,
        result.evaluations,
        result.starts,
        needs,
    )


def _free(box: np.ndarray) -> int:
    return int(np.sum(box[:, 1] > box[:, 0]))


def _poly_problem(poly: _Polynomial) -> str:
    """Why a point set cannot carry a polynomial collapse ("" when it can)."""
    sizes = np.unique(poly.n)
    if sizes.size < 2:
        return (
            f"the window holds points of {sizes.size} size(s): a collapse needs points of at "
            "least two sizes in it"
        )
    distinct = max(np.unique(poly.p[poly.n == s]).size for s in sizes)
    if distinct < poly.degree + 1:
        return (
            f"no size has {poly.degree + 1} distinct rates in the window: a degree-"
            f"{poly.degree} master curve would not be determined at every parameter value"
        )
    return ""


def _fit_poly(
    polys: Sequence[_Polynomial],
    settings: _Settings,
    initial: ArrayLike | None,
    starts: int | None = None,
) -> _Outcome:
    """Minimize the total polynomial chi2 over ``(p_c, z, nu)`` on fixed point sets."""
    degree = settings.degree
    n_free = _free(settings.box)
    needs = tuple(degree + 2 + n_free for _ in polys)
    problems = [message for message in map(_poly_problem, polys) if message]
    dof = sum(poly.count - (degree + 1) for poly in polys) - n_free
    if problems or dof < 1:
        total = sum(poly.count for poly in polys)
        message = (
            problems[0]
            if problems
            else (
                f"the window holds {total} points: a degree-{degree} master curve and {n_free} "
                f"fitted parameters leave {dof} degrees of freedom (at least 1 is needed)"
            )
        )
        out = _no_fit(len(polys), "too_few_points", message, needs)
        return _Outcome(
            out.params,
            out.cost,
            tuple(p.count for p in polys),
            out.status,
            out.message,
            (),
            0,
            0,
            needs,
            math.nan,
            max(dof, 0),
        )

    def objective(theta: np.ndarray) -> float:
        p_c, z, nu = (float(v) for v in theta)
        total = 0.0
        for poly in polys:
            total += poly.chi2(p_c, z, nu)
        return total

    result = minimize_bounded(
        objective,
        settings.box,
        initial=initial,
        starts=settings.starts if starts is None else starts,
        seed=settings.seed,
        max_evaluations=settings.max_evaluations,
    )
    if not math.isfinite(result.fun):
        out = _no_fit(
            len(polys),
            "too_few_points",
            "the polynomial objective is undefined everywhere the search went (n^z or n^(1/nu) "
            "overflows): narrow the box",
            needs,
            result.evaluations,
        )
        return out
    status, message, at_bound = _status(result)
    return _Outcome(
        result.x,
        result.fun / dof,
        tuple(p.count for p in polys),
        status,
        message,
        at_bound,
        result.evaluations,
        result.starts,
        needs,
        result.fun,
        dof,
    )


@dataclass(frozen=True, eq=False)
class _WindowFit:
    outcome: _Outcome
    selected: np.ndarray
    stable: bool | None  # None: a fixed set, never redrawn (so never checked)
    iterations: int
    solution: _PolySolution | None


def _rows(initial: ArrayLike | None) -> np.ndarray | None:
    return None if initial is None else np.atleast_2d(np.asarray(initial, dtype=np.float64))


def _reference(data: ScalingData, settings: _Settings, initial: ArrayLike | None) -> np.ndarray:
    """Where the polynomial window is first drawn: ``initial``'s first row, else the
    Houdayer-Hartmann fit with the same settings, else the center of the box."""
    rows = _rows(initial)
    if rows is not None:
        return rows[0]
    hh = _fit_hh([_Collapse.of(data)], [data.size], settings, None)
    return hh.params if hh.status in _USABLE else settings.box.mean(axis=1)


def _window_fit(
    data: ScalingData,
    settings: _Settings,
    initial: ArrayLike | None,
    reference: np.ndarray,
) -> _WindowFit:
    """The polynomial fit with a self-consistent window.

    The window is drawn at ``reference``; the fit keeps that point set during its search
    (``initial`` or the reference, and ``starts`` random starts); the window is drawn again at
    the fitted parameters, and the fit repeated from there (no random starts) until the set no
    longer changes, at most ``window_iterations`` fits.
    """
    theta = np.asarray(reference, dtype=np.float64)
    used = window_points(data, theta[0], theta[1], theta[2], window=settings.window)
    rows = _rows(initial)
    start_rows = rows if rows is not None else theta
    for iteration in range(1, settings.window_iterations + 1):
        poly = _Polynomial.subset(data, used, settings.degree)
        if iteration == 1:
            outcome = _fit_poly([poly], settings, start_rows)
        else:
            outcome = _fit_poly([poly], settings, theta, starts=0)
        if outcome.status not in _USABLE:
            return _WindowFit(outcome, used, False, iteration, None)
        p_c, z, nu = (float(v) for v in outcome.params)
        again = window_points(data, p_c, z, nu, window=settings.window)
        if np.array_equal(again, used) or iteration == settings.window_iterations:
            stable = bool(np.array_equal(again, used))
            return _WindowFit(outcome, used, stable, iteration, poly.solve(p_c, z, nu))
        used, theta = again, outcome.params
    raise AssertionError("unreachable")  # pragma: no cover


@dataclass(frozen=True, eq=False)
class CollapseFit:
    """A fitted collapse of one observable at one ``beta``.

    ``p_c``, ``z``, ``nu`` (NaN when the status is ``too_few_sizes`` or ``too_few_points``),
    the objective ``cost`` at them (``method="houdayer_hartmann"``: the mean ``chi2`` of the
    bracketed points; ``"polynomial"``: ``chi2 / dof``), ``points_used`` of ``total_points``
    (bracketed points, or the points in the window), ``status`` (one of :data:`STATUSES`) with
    a ``message``, the parameters ``at_bound``, and the settings that produced it (``bounds``;
    ``min_points``, the points required here, from ``min_points_setting``, the value given, None
    for the default rule; ``random_starts`` as given and ``starts`` run; ``seed``;
    ``max_evaluations``) with the data's ``observable``, ``beta``, ``label`` and ``estimator``.

    Polynomial fits also have the ``degree``, the ``window`` rule, the total ``chi2`` and
    ``dof = points_used - (degree + 1) - (free parameters)``, the point set ``selected`` (a mask
    over the data's points), the ``window_range`` ``(min x, max x)`` of the set at the fit,
    ``window_stable`` (True when the window drawn again at the fit is ``selected``, False when it
    is not after ``window_iterations`` fits; None for a refit that keeps a given set by design
    and never redraws it: leave-one-size-out, the drift of :func:`collapse_diagnostics`, a
    bootstrap replicate with ``reselect_window=False``), and the ``coefficients`` of
    ``F(x) = sum_k a_k x^k``. ``chi2 / dof`` is about 1 when the points
    scatter about the master curve as their errors say; the rates of a size share graphs, so
    their errors are correlated and ``chi2`` does not follow a chi-square law (the bootstrap
    gives the uncertainties).
    """

    p_c: float
    z: float
    nu: float
    cost: float
    points_used: int
    total_points: int
    status: str
    message: str
    at_bound: tuple[str, ...]
    sizes: tuple[int, ...]
    observable: str
    beta: str | None
    label: str | None
    estimator: str
    bounds: tuple[tuple[float, float], ...]
    min_points: int
    min_points_setting: int | None
    random_starts: int
    starts: int
    seed: Any
    max_evaluations: int
    evaluations: int
    method: str = HOUDAYER_HARTMANN
    degree: int | None = None
    window: str | float | None = None
    chi2: float = math.nan
    dof: int | None = None
    selected: np.ndarray | None = None
    window_range: tuple[float, float] | None = None
    window_stable: bool | None = None
    window_iterations: int = 0
    coefficients: np.ndarray | None = None
    _settings: Any = field(default=None, repr=False)

    @property
    def params(self) -> np.ndarray:
        """``[p_c, z, nu]``."""
        return np.array([self.p_c, self.z, self.nu])

    @property
    def usable(self) -> bool:
        """True when the fit returned parameters (``ok``, ``at_bound`` or ``not_converged``)."""
        return self.status in _USABLE

    @property
    def chi2_per_dof(self) -> float:
        """``chi2 / dof`` of a polynomial fit (NaN otherwise)."""
        return self.chi2 / self.dof if self.dof else math.nan

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _plain(self)
        return out


def _collapse_fit(
    data: ScalingData, settings: _Settings, outcome: _Outcome, window: _WindowFit | None = None
) -> CollapseFit:
    p_c, z, nu = (float(v) for v in outcome.params)
    extra: dict[str, Any] = {}
    if settings.method == POLYNOMIAL:
        extra = {
            "degree": settings.degree,
            "window": settings.window,
            "chi2": outcome.chi2,
            "dof": outcome.dof,
        }
        if window is not None:
            extra.update(
                selected=window.selected,
                window_stable=window.stable,
                window_iterations=window.iterations,
            )
            if window.solution is not None:
                x = window.solution.x
                extra.update(
                    window_range=(float(x.min()), float(x.max())),
                    coefficients=window.solution.coefficients,
                )
    return CollapseFit(
        p_c=p_c,
        z=z,
        nu=nu,
        cost=outcome.cost,
        points_used=outcome.used[0],
        total_points=data.size,
        status=outcome.status,
        message=outcome.message,
        at_bound=outcome.at_bound,
        sizes=tuple(int(s) for s in data.sizes),
        observable=data.observable,
        beta=data.beta,
        label=data.label,
        estimator=data.estimator,
        bounds=tuple((float(a), float(b)) for a, b in settings.box),
        min_points=outcome.needs[0],
        min_points_setting=settings.min_points,
        random_starts=settings.starts,
        starts=outcome.starts,
        seed=settings.seed,
        max_evaluations=settings.max_evaluations,
        evaluations=outcome.evaluations,
        method=settings.method,
        _settings=settings,
        **extra,
    )


def _settings_of(fit: CollapseFit, starts: int | None = None) -> _Settings:
    """The settings a fit was made with (``starts`` overrides its random starts)."""
    settings: _Settings = fit._settings
    return settings if starts is None else replace(settings, starts=int(starts))


def _fit_one(
    data: ScalingData,
    settings: _Settings,
    initial: ArrayLike | None,
    reference: np.ndarray | None = None,
) -> CollapseFit:
    if settings.method == HOUDAYER_HARTMANN:
        outcome = _fit_hh([_Collapse.of(data)], [data.size], settings, initial)
        return _collapse_fit(data, settings, outcome)
    if data.sizes.size < 2:
        needs = (settings.degree + 2 + _free(settings.box),)
        outcome = _no_fit(1, "too_few_sizes", _TOO_FEW_SIZES.format(sizes=[1]), needs)
        return _collapse_fit(data, settings, outcome)
    ref = _reference(data, settings, initial) if reference is None else reference
    window = _window_fit(data, settings, initial, ref)
    return _collapse_fit(data, settings, window.outcome, window)


def fit_collapse(
    data: ScalingData,
    *,
    bounds: Any,
    method: str = HOUDAYER_HARTMANN,
    degree: int = DEFAULT_DEGREE,
    window: str | float = "overlap",
    initial: ArrayLike | None = None,
    starts: int = 8,
    seed: Any = 0,
    min_points: int | None = None,
    max_evaluations: int = 2000,
    window_iterations: int = 5,
) -> CollapseFit:
    """Fit ``(p_c, z, nu)`` to one observable at one ``beta``.

    ``bounds``: ``{"p_c": (lo, hi), "z": (lo, hi), "nu": (lo, hi)}`` or three pairs in that
    order (``lo == hi`` fixes a parameter; ``nu``'s lower bound must be positive). The search is
    :func:`minimize_bounded` from ``initial`` (or the box center) and ``starts`` random starts
    drawn from ``seed``. The status says whether the result is usable (:data:`STATUSES`); a fit
    never raises for lack of data.

    ``method="houdayer_hartmann"`` (the default) minimizes the collapse quality of
    :func:`collapse_cost`. A parameter vector is admissible only when at least ``min_points``
    points are bracketed (default ``max(3, ceil(P/2))`` of the ``P`` points); inadmissible
    vectors count as ``+inf``.

    ``method="polynomial"`` (ruling R-FSS) minimizes ``chi2(p_c, z, nu) = min_a sum_i (Y_i -
    F_a(x_i))^2 / dY_i^2`` with a polynomial master curve of ``degree`` whose coefficients are
    solved in closed form (:func:`polynomial_cost`), on a fixed point set: the ``window``
    (:func:`window_points`) drawn at ``initial``'s first row, else at the Houdayer-Hartmann fit,
    else at the box center; after each search it is drawn again at the fitted parameters and
    the fit repeated until the set no longer changes (at most ``window_iterations`` fits;
    ``window_stable`` says whether it settled). The set needs points of two sizes, a size with
    ``degree + 1`` distinct rates and one degree of freedom; ``min_points`` does not apply.
    """
    settings = _settings(
        bounds,
        starts,
        seed,
        min_points,
        max_evaluations,
        method,
        degree,
        window,
        window_iterations,
    )
    return _fit_one(data, settings, initial)


@dataclass(frozen=True, eq=False)
class SharedFit:
    """One ``(p_c, z, nu)`` fitted to several observables, each with its own scaling function.

    Houdayer-Hartmann: ``cost`` is the pooled quality (the mean ``chi2`` over all observables'
    bracketed points), ``costs`` and ``points_used`` per observable at the shared parameters.
    Polynomial: ``cost = chi2 / dof`` over all observables (one master curve each, on the point
    sets ``selected`` that their own fits chose), ``costs`` each observable's ``chi2_o / (|S_o|
    - degree - 1)`` at the shared parameters, ``|S_o|`` the size of its set (study/07 §14).
    """

    p_c: float
    z: float
    nu: float
    cost: float
    costs: tuple[float, ...]
    points_used: tuple[int, ...]
    total_points: tuple[int, ...]
    status: str
    message: str
    at_bound: tuple[str, ...]
    observables: tuple[str, ...]
    beta: str | None
    labels: tuple[str | None, ...]
    bounds: tuple[tuple[float, float], ...]
    min_points: tuple[int, ...]
    starts: int
    seed: Any
    evaluations: int
    method: str = HOUDAYER_HARTMANN
    degree: int | None = None
    window: str | float | None = None
    chi2: float = math.nan
    dof: int | None = None
    selected: tuple[np.ndarray, ...] | None = None

    @property
    def params(self) -> np.ndarray:
        return np.array([self.p_c, self.z, self.nu])

    @property
    def usable(self) -> bool:
        return self.status in _USABLE

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _plain(self)
        return out


def _common_beta(datasets: Sequence[ScalingData]) -> str | None:
    betas = {d.beta for d in datasets}
    if len(betas) > 1:
        raise ValueError(
            f"the observables come from different beta values {sorted(map(str, betas))}: "
            "exponents are compared at one fixed beta (v1.4 §5.8)"
        )
    return next(iter(betas))


def _datasets(datasets: Sequence[ScalingData]) -> list[ScalingData]:
    items = list(datasets)
    if len(items) < 2:
        raise ValueError("a shared fit needs at least two observables")
    for d in items:
        if not isinstance(d, ScalingData):
            raise ValueError("datasets must be ScalingData")
    names = [d.observable for d in items]
    if len(set(names)) != len(names):
        raise ValueError(f"observable names must be distinct; got {names}")
    return items


def _shared_fit(
    items: Sequence[ScalingData],
    settings: _Settings,
    outcome: _Outcome,
    beta: str | None,
    polys: Sequence[_Polynomial] | None = None,
    selected: tuple[np.ndarray, ...] | None = None,
) -> SharedFit:
    params = outcome.params
    costs: list[float] = []
    for i, d in enumerate(items):
        if not np.all(np.isfinite(params)):
            costs.append(math.nan)
        elif polys is None:
            cc = _Collapse.of(d).chi2(*(float(v) for v in params))
            ok = ~np.isnan(cc)
            costs.append(float(np.mean(cc[ok])) if ok.any() else math.nan)
        else:
            own = polys[i].count - (settings.degree + 1)
            value = polys[i].chi2(*(float(v) for v in params))
            costs.append(value / own if own > 0 else math.nan)
    extra: dict[str, Any] = {}
    if settings.method == POLYNOMIAL:
        extra = {
            "degree": settings.degree,
            "window": settings.window,
            "chi2": outcome.chi2,
            "dof": outcome.dof,
            "selected": selected,
        }
    return SharedFit(
        p_c=float(params[0]),
        z=float(params[1]),
        nu=float(params[2]),
        cost=outcome.cost,
        costs=tuple(costs),
        points_used=outcome.used,
        total_points=tuple(d.size for d in items),
        status=outcome.status,
        message=outcome.message,
        at_bound=outcome.at_bound,
        observables=tuple(d.observable for d in items),
        beta=beta,
        labels=tuple(d.label for d in items),
        bounds=tuple((float(a), float(b)) for a, b in settings.box),
        min_points=outcome.needs,
        starts=outcome.starts,
        seed=settings.seed,
        evaluations=outcome.evaluations,
        method=settings.method,
        **extra,
    )


def _shared_polynomial(
    items: Sequence[ScalingData],
    settings: _Settings,
    initial: ArrayLike | None,
    independent: Sequence[CollapseFit],
    beta: str | None,
) -> SharedFit:
    """The shared polynomial fit on the point sets the observables' own fits chose."""
    failed = [f for f in independent if not f.usable]
    if failed:
        outcome = _no_fit(
            len(items),
            failed[0].status,
            f"the fit of {failed[0].observable} alone failed, so it has no window: "
            f"{failed[0].message}",
            tuple(f.min_points for f in independent),
        )
        return _shared_fit(items, settings, outcome, beta)
    selected = tuple(f.selected for f in independent if f.selected is not None)
    polys = [
        _Polynomial.subset(d, s, settings.degree) for d, s in zip(items, selected, strict=True)
    ]
    rows = np.array([f.params for f in independent])
    user = _rows(initial)
    if user is not None:
        rows = np.vstack([user, rows])
    outcome = _fit_poly(polys, settings, rows)
    return _shared_fit(items, settings, outcome, beta, polys, selected)


def fit_shared(
    datasets: Sequence[ScalingData],
    *,
    bounds: Any,
    method: str = HOUDAYER_HARTMANN,
    degree: int = DEFAULT_DEGREE,
    window: str | float = "overlap",
    initial: ArrayLike | None = None,
    starts: int = 8,
    seed: Any = 0,
    min_points: int | None = None,
    max_evaluations: int = 2000,
    window_iterations: int = 5,
) -> SharedFit:
    """Fit one ``(p_c, z, nu)`` to several observables at one ``beta`` (the shared hypothesis).

    Every observable keeps its own scaling function. Houdayer-Hartmann: the quality is evaluated
    per observable and pooled, ``S_shared = sum_o sum_i chi2_oi / sum_o N_o``, each observable
    needing its own ``min_points`` bracketed points. Polynomial: each observable is first fitted
    alone (:func:`fit_collapse`), which fixes its window; the shared fit minimizes
    ``sum_o chi2_o`` on those point sets, starting from the independent parameters. Same
    options as :func:`fit_collapse`.
    """
    items = _datasets(datasets)
    beta = _common_beta(items)
    settings = _settings(
        bounds,
        starts,
        seed,
        min_points,
        max_evaluations,
        method,
        degree,
        window,
        window_iterations,
    )
    if settings.method == HOUDAYER_HARTMANN:
        outcome = _fit_hh(
            [_Collapse.of(d) for d in items], [d.size for d in items], settings, initial
        )
        return _shared_fit(items, settings, outcome, beta)
    independent = [_fit_one(d, settings, initial) for d in items]
    return _shared_polynomial(items, settings, initial, independent, beta)


# ---------------------------------------------------------------------------------------------
# graph-cluster bootstrap, paired across p
# ---------------------------------------------------------------------------------------------


def _draws_by_size(
    datasets: Sequence[ScalingData], replicates: int, seed: Any
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """For each size: its graph labels and their bootstrap draws, shared by every point.

    The draws of size ``n`` come from ``graph_bootstrap(labels, replicates, seed=[seed, n])``:
    one resampling per ``(n, beta)``, applied to every rate and every observable.
    """
    labels: dict[int, np.ndarray] = {}
    for d in datasets:
        if d.graphs is None:
            raise ValueError(
                f"{d.observable}: the bootstrap needs the per-graph values of every point "
                "(build the data with scaling_data(n, p, means))"
            )
        for i in range(d.size):
            n = int(d.n[i])
            if n not in labels:
                labels[n] = d.graphs[i]
            elif not np.array_equal(labels[n], d.graphs[i]):
                raise ValueError(
                    f"the points at n = {n} do not have the same graphs; the bootstrap resamples "
                    "graphs once per size and applies the draws to every rate and observable"
                )
    out = {}
    for n, lab in sorted(labels.items()):
        out[n] = (lab, _stats.graph_bootstrap(lab, replicates, seed=_size_seed(seed, n)))
    return out


def _size_seed(seed: Any, n: int) -> list[int]:
    if isinstance(seed, (list, tuple, np.ndarray)):
        return [int(s) for s in np.asarray(seed, dtype=object).reshape(-1)] + [int(n)]
    return [int(seed), int(n)]


def _replicate_values(
    data: ScalingData, draws: Mapping[int, tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """``(B, P)`` replicate means and graph-cluster errors of every point."""
    assert data.graph_means is not None
    sample = next(iter(draws.values()))[1]
    reps = sample.shape[0]
    ystar = np.empty((reps, data.size))
    sestar = np.empty((reps, data.size))
    positions = {n: np.searchsorted(lab, d) for n, (lab, d) in draws.items()}
    for i in range(data.size):
        m, s = _stats._replicate(data.graph_means[i], positions[int(data.n[i])])
        assert s is not None
        ystar[:, i] = m
        sestar[:, i] = s
    return ystar, sestar


def _valid(y: np.ndarray, se: np.ndarray) -> np.ndarray:
    return np.isfinite(y) & np.isfinite(se) & (se > 0)


def _replicate_data(
    data: ScalingData, y: np.ndarray, se: np.ndarray, p: np.ndarray | None = None
) -> tuple[ScalingData, np.ndarray] | None:
    """A replicate's usable points (finite ``y``, positive ``se``) as data, with their mask over
    ``data``'s points (rates ``p`` replace ``data.p`` when given); None when no point is usable."""
    keep = _valid(y, se)
    if not keep.any():
        return None
    rates = data.p if p is None else p
    part = ScalingData(
        n=data.n[keep],
        p=rates[keep],
        y=y[keep],
        se=se[keep],
        observable=data.observable,
        beta=data.beta,
        label=data.label,
        estimator=data.estimator,
    )
    return part, keep


def _fixed_window_fit(
    data: ScalingData,
    mask: np.ndarray,
    settings: _Settings,
    initial: ArrayLike | None,
    anchor: np.ndarray,
) -> CollapseFit:
    """A polynomial fit on the given point set ``mask`` of ``data``, never redrawn: from
    ``initial`` (else ``anchor``) and the settings' random starts (``window_stable`` None)."""
    poly = _Polynomial.subset(data, mask, settings.degree)
    rows = _rows(initial)
    outcome = _fit_poly([poly], settings, rows if rows is not None else anchor)
    solution = (
        poly.solve(*(float(v) for v in outcome.params)) if outcome.status in _USABLE else None
    )
    return _collapse_fit(data, settings, outcome, _WindowFit(outcome, mask, None, 1, solution))


@dataclass(frozen=True, eq=False)
class _Replicate:
    fit: CollapseFit  # fitted on the replicate's usable points
    keep: np.ndarray  # those points, as a mask over the data's points


def _refit(
    data: ScalingData,
    fit: CollapseFit,
    settings: _Settings,
    y: np.ndarray,
    se: np.ndarray,
    *,
    initial: ArrayLike | None,
    reselect: bool,
    reference: np.ndarray | None = None,
) -> _Replicate | None:
    """One bootstrap replicate (values ``y``, ``se``) fitted the way ``fit`` fitted the data.

    The replicate's usable points are fitted by the same procedure with the same ``initial``,
    seed and bounds (``settings`` carries the replicate's random starts): Houdayer-Hartmann
    from ``initial`` or the box center and the random starts; polynomial with ``reselect`` its
    own window (drawn at ``reference``, else at the replicate's own Houdayer-Hartmann fit as for
    the data, and redrawn until it settles), without it the fit's window (from ``initial`` or the
    fit's parameters). None when no point is usable.
    """
    made = _replicate_data(data, y, se)
    if made is None:
        return None
    part, keep = made
    if settings.method == HOUDAYER_HARTMANN or reselect:
        return _Replicate(_fit_one(part, settings, initial, reference), keep)
    assert fit.selected is not None
    return _Replicate(
        _fixed_window_fit(part, fit.selected[keep], settings, initial, fit.params), keep
    )


@dataclass(frozen=True, eq=False)
class BootstrapFit:
    """Graph-cluster bootstrap of a collapse fit, paired across ``p``.

    ``fit`` is the fit to the data; ``replicates`` ``(B, 3)`` the refits of the resampled data
    (NaN rows where a refit returned no parameters), ``costs`` and ``statuses`` theirs; ``se``
    the standard deviation of the usable replicates per parameter and ``lower``/``upper`` their
    percentile interval at ``level``; ``failed`` the replicates without parameters and
    ``at_bound`` those with a parameter on its bound. ``replicate_starts`` is the number of
    random starts of every refit and ``reselect_window`` whether polynomial refits drew their
    own windows (None for Houdayer-Hartmann). Polynomial bootstraps also keep each replicate's
    point set, ``selected`` ``(B, P)`` (a mask over the data's points; all False where a
    replicate has none), and ``window_stable`` ``(B,)``, whether its window settled (False for
    a failed replicate; None with ``reselect_window=False``, where nothing is redrawn).
    """

    fit: CollapseFit
    replicates: np.ndarray
    costs: np.ndarray
    statuses: tuple[str, ...]
    se: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    level: float
    failed: int
    at_bound: int
    seed: Any
    replicate_starts: int = 0
    reselect_window: bool | None = None
    selected: np.ndarray | None = None
    window_stable: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _plain(self)
        return out


def _replicate_count(replicates: Any, smallest: int) -> int:
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, (int, np.integer))
        or replicates < smallest
    ):
        raise ValueError(f"replicates must be an integer >= {smallest}")
    return int(replicates)


def _replicate_starts(value: Any, starts: int) -> int:
    """The random starts of a refit: ``starts`` (the data fit's) when ``value`` is None."""
    if value is None:
        return starts
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(
            f"replicate_starts must be a non-negative integer, or None for as many as starts; "
            f"not {value!r}"
        )
    return int(value)


def _reselect(value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"reselect_window must be True or False, not {value!r}")
    return bool(value)


def bootstrap_fit(
    data: ScalingData,
    *,
    bounds: Any,
    method: str = HOUDAYER_HARTMANN,
    degree: int = DEFAULT_DEGREE,
    window: str | float = "overlap",
    replicates: int = 200,
    seed: Any = 0,
    level: float = 0.95,
    initial: ArrayLike | None = None,
    starts: int = 8,
    min_points: int | None = None,
    max_evaluations: int = 2000,
    window_iterations: int = 5,
    replicate_starts: int | None = None,
    reselect_window: bool = True,
) -> BootstrapFit:
    """The uncertainty of ``(p_c, z, nu)`` by resampling graphs (v1.4 §5.8, §9).

    For every size ``n`` the graphs are resampled once per replicate
    (``graph_bootstrap(labels_n, replicates, seed=[seed, n])``) and every rate's mean and error
    at that size are recomputed from the same drawn graphs, so the coupling of the rates
    (shared graphs, gates and mask uniforms) is kept. A point whose replicate error is 0 or
    undefined is left out of that replicate. Returns the replicate parameters, their standard
    deviations and percentile intervals. Needs the per-graph values (:func:`scaling_data`).

    Each replicate is fitted by the same procedure as the data (:func:`fit_collapse` with the
    same ``method``, ``bounds``, ``initial``, ``seed`` and settings), so the spread includes
    everything the estimate depends on: the multi-start search (``replicate_starts`` random
    starts, by default as many as ``starts``; fewer is cheaper and approximates the procedure)
    and, for the polynomial method with ``reselect_window`` (the default, ruling R-FSS), the
    window, which each replicate draws at its own Houdayer-Hartmann fit (or ``initial``) and
    redraws until it settles. ``reselect_window=False`` refits every replicate on the data fit's
    window instead (cheaper; its errors leave out the window's variability by construction).
    The seed is both the graph draws' (``[seed, n]``) and the fits' random starts', for the data
    and for every replicate alike. Measured calibration on synthetic data: 07 §11.
    """
    settings = _settings(
        bounds,
        starts,
        seed,
        min_points,
        max_evaluations,
        method,
        degree,
        window,
        window_iterations,
    )
    lvl = _stats._level(level)
    reps_count = _replicate_count(replicates, 2)
    rep_starts = _replicate_starts(replicate_starts, settings.starts)
    reselect = _reselect(reselect_window)
    draws = _draws_by_size([data], reps_count, seed)
    fit = _fit_one(data, settings, initial)
    rep_settings = replace(settings, starts=rep_starts)
    polynomial = settings.method == POLYNOMIAL
    reps = np.full((reps_count, 3), np.nan)
    costs = np.full(reps_count, np.nan)
    selected = np.zeros((reps_count, data.size), dtype=bool) if polynomial else None
    stable = np.zeros(reps_count, dtype=bool) if polynomial and reselect else None
    statuses: list[str] = []
    if fit.usable:
        ystar, sestar = _replicate_values(data, draws)
        for b in range(reps_count):
            out = _refit(
                data, fit, rep_settings, ystar[b], sestar[b], initial=initial, reselect=reselect
            )
            if out is None:
                statuses.append("too_few_points")
                continue
            statuses.append(out.fit.status)
            if out.fit.usable:
                reps[b] = out.fit.params
                costs[b] = out.fit.cost
            if selected is not None and out.fit.selected is not None:
                selected[b, np.flatnonzero(out.keep)] = out.fit.selected
            if stable is not None:
                stable[b] = out.fit.window_stable is True
    else:
        statuses = [fit.status] * reps_count
    usable = ~np.isnan(reps[:, 0])
    good = reps[usable]
    se = good.std(axis=0, ddof=1) if good.shape[0] > 1 else np.full(3, np.nan)
    lower, upper = (
        _stats.percentile_interval(good, lvl) if good.shape[0] else (np.full(3, np.nan),) * 2
    )
    return BootstrapFit(
        fit=fit,
        replicates=reps,
        costs=costs,
        statuses=tuple(statuses),
        se=se,
        lower=np.asarray(lower),
        upper=np.asarray(upper),
        level=lvl,
        failed=int((~usable).sum()),
        at_bound=sum(1 for s in statuses if s == "at_bound"),
        seed=seed,
        replicate_starts=rep_starts,
        reselect_window=reselect if polynomial else None,
        selected=selected,
        window_stable=stable,
    )


# ---------------------------------------------------------------------------------------------
# shared versus independent exponents
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SharedComparison:
    """Shared ``(p_c, z, nu)`` against independent fits, with a bootstrap null.

    ``delta`` measures how much worse the shared fit is: for the Houdayer-Hartmann method
    ``shared.cost - independent_cost`` (``independent_cost`` pools the independent fits'
    ``chi2`` the same way); for the polynomial method ``shared.chi2 - sum_o chi2_o`` on the same
    point sets (``independent_cost`` is then ``sum_o chi2_o / sum_o dof_o``). ``null`` holds its
    replicates under the shared hypothesis (NaN for failed refits),
    ``p_value = (1 + #{null >= delta}) / (1 + B')`` over the ``B'`` usable replicates, and
    ``prefers``: ``"independent"`` when ``p_value <= alpha``, ``"shared"`` otherwise, and
    ``"undetermined"`` when a fit gave no parameters. ``message`` explains an undetermined
    result. ``replicate_starts`` and ``reselect_window`` (None for Houdayer-Hartmann) are the
    null refits' settings.
    """

    shared: SharedFit
    independent: tuple[CollapseFit, ...]
    independent_cost: float
    delta: float
    null: np.ndarray
    p_value: float
    alpha: float
    prefers: str
    failed: int
    seed: Any
    message: str
    method: str = HOUDAYER_HARTMANN
    replicate_starts: int = 0
    reselect_window: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _plain(self)
        return out


def _pooled(fits: Sequence[CollapseFit]) -> float:
    total = sum(f.cost * f.points_used for f in fits)
    count = sum(f.points_used for f in fits)
    return total / count if count else math.nan


def compare_shared_independent(
    datasets: Sequence[ScalingData],
    *,
    bounds: Any,
    method: str = HOUDAYER_HARTMANN,
    degree: int = DEFAULT_DEGREE,
    window: str | float = "overlap",
    replicates: int = 200,
    seed: Any = 0,
    alpha: float = 0.05,
    initial: ArrayLike | None = None,
    starts: int = 8,
    min_points: int | None = None,
    max_evaluations: int = 2000,
    window_iterations: int = 5,
    replicate_starts: int | None = None,
    reselect_window: bool = True,
) -> SharedComparison:
    """Test one shared ``(p_c, z, nu)`` against independent exponents (v1.4 §5.8, §5.9).

    The statistic ``delta`` (see :class:`SharedComparison`) is the cost of sharing the
    parameters. Its null distribution is built by graph-cluster resampling of data mapped onto
    the shared hypothesis: every observable ``o`` is mapped point by point to

        p~ = p_c^s + (p - p_c^o) n^(1/nu_o - 1/nu_s),   y~ = y n^(z_s - z_o)

    (07 §8.3 writes them with a tilde; the graph means and errors are mapped by the factor of
    y), which gives at ``theta_s`` exactly the scaled coordinates the observable has at its own
    ``theta_o``: its collapse quality there
    (and, for the polynomial method, its window) is the one it had at ``theta_o``. The mapped
    data follow the shared scaling form with ``theta_s`` exactly when the data follow theirs
    without noise; with noise their own best parameters lie near ``theta_s``, not exactly on it
    (07 §8.3). Each replicate resamples the graphs once per size
    (``graph_bootstrap(labels_n, B, seed=[seed, n])``), applies the draws to every rate and
    observable, recomputes the means and errors, and computes ``delta*`` by the same procedure
    as ``delta``: the shared and the independent fits of the replicate, with the same
    ``initial``, seed and settings and ``replicate_starts`` random starts (default: as many as
    ``starts``); for the polynomial method each observable's window is drawn again for the
    replicate (``reselect_window``, the default) or kept from its fit to the data. The p-value
    is ``(1 + #{delta* >= delta}) / (1 + B')``. Needs the per-graph values of every point.
    """
    items = _datasets(datasets)
    beta = _common_beta(items)
    settings = _settings(
        bounds,
        starts,
        seed,
        min_points,
        max_evaluations,
        method,
        degree,
        window,
        window_iterations,
    )
    if not (isinstance(alpha, float) and 0 < alpha < 1):
        raise ValueError(f"alpha must be a probability strictly between 0 and 1, not {alpha!r}")
    reps = _replicate_count(replicates, 1)
    rep_settings = replace(settings, starts=_replicate_starts(replicate_starts, settings.starts))
    reselect = _reselect(reselect_window)
    draws = _draws_by_size(items, reps, seed)
    polynomial = settings.method == POLYNOMIAL
    recorded: dict[str, Any] = {
        "method": settings.method,
        "replicate_starts": rep_settings.starts,
        "reselect_window": reselect if polynomial else None,
    }
    if polynomial:
        independent = tuple(_fit_one(d, settings, initial) for d in items)
        shared = _shared_polynomial(items, settings, initial, independent, beta)
        dofs = sum(f.dof or 0 for f in independent)
        independent_cost = sum(f.chi2 for f in independent) / dofs if dofs else math.nan
    else:
        collapses = [_Collapse.of(d) for d in items]
        shared = _shared_fit(
            items, settings, _fit_hh(collapses, [d.size for d in items], settings, initial), beta
        )
        independent = tuple(
            _collapse_fit(d, settings, _fit_hh([c], [d.size], settings, initial))
            for d, c in zip(items, collapses, strict=True)
        )
        independent_cost = _pooled(independent)
    null = np.full(reps, np.nan)
    if not (shared.usable and all(f.usable for f in independent)):
        bad = [shared.status] + [f.status for f in independent]
        return SharedComparison(
            shared=shared,
            independent=independent,
            independent_cost=independent_cost,
            delta=math.nan,
            null=null,
            p_value=math.nan,
            alpha=alpha,
            prefers="undetermined",
            failed=reps,
            seed=seed,
            message=f"a fit returned no parameters (statuses {bad}): no comparison",
            **recorded,
        )
    if polynomial:
        delta = shared.chi2 - sum(f.chi2 for f in independent)
    else:
        delta = shared.cost - independent_cost
    theta_s = shared.params
    mapped = []
    for d, f in zip(items, independent, strict=True):
        n = d.n.astype(np.float64)
        factor = n ** (theta_s[1] - f.z)
        p_new = theta_s[0] + (d.p - f.p_c) * n ** (1.0 / f.nu - 1.0 / theta_s[2])
        ystar, sestar = _replicate_values(d, draws)
        mapped.append(_Mapped(d, f, p_new, ystar * factor, sestar * factor))
    for b in range(reps):
        value = _null_replicate(mapped, b, rep_settings, theta_s, initial, reselect, beta)
        if value is not None:
            null[b] = value
    valid = null[~np.isnan(null)]
    p_value = (1.0 + float(np.sum(valid >= delta))) / (1.0 + valid.size) if valid.size else math.nan
    if valid.size == 0:
        prefers, message = "undetermined", "every bootstrap refit failed"
    else:
        prefers = "independent" if p_value <= alpha else "shared"
        message = ""
    return SharedComparison(
        shared=shared,
        independent=independent,
        independent_cost=independent_cost,
        delta=delta,
        null=null,
        p_value=p_value,
        alpha=alpha,
        prefers=prefers,
        failed=int(reps - valid.size),
        seed=seed,
        message=message,
        **recorded,
    )


@dataclass(frozen=True, eq=False)
class _Mapped:
    """One observable's data mapped onto the shared parameters, and its bootstrap values."""

    data: ScalingData
    fit: CollapseFit  # the observable's own fit to the (unmapped) data
    p: np.ndarray  # the mapped rates p~ (07 §8.3)
    ystar: np.ndarray  # (B, P) replicate means, mapped
    sestar: np.ndarray  # (B, P) replicate errors, mapped


def _null_replicate(
    mapped: Sequence[_Mapped],
    b: int,
    settings: _Settings,
    theta_s: np.ndarray,
    initial: ArrayLike | None,
    reselect: bool,
    beta: str | None,
) -> float | None:
    """``delta*`` of replicate ``b`` of the mapped data, by the procedure that gave ``delta``
    (None when a fit fails).

    Houdayer-Hartmann: the shared and each independent fit from ``initial`` (or the box center)
    and the random starts. Polynomial with ``reselect``: each observable fitted alone with its
    own window (as :func:`fit_collapse`), then the shared fit on those windows from the
    independent parameters (as :func:`fit_shared`); without it, the windows of the observables'
    fits to the data, the independent fits from ``initial`` or ``theta_s`` (the mapped data's
    image of each observable's own parameters, Proposition 6) and the shared fit from theirs.
    """
    parts = []
    for m in mapped:
        made = _replicate_data(m.data, m.ystar[b], m.sestar[b], m.p)
        if made is None:
            return None
        parts.append(made)
    if settings.method == HOUDAYER_HARTMANN:
        cols = [_Collapse.of(part) for part, _ in parts]
        counts = [part.size for part, _ in parts]
        s_out = _fit_hh(cols, counts, settings, initial)
        i_outs = [_fit_hh([c], [k], settings, initial) for c, k in zip(cols, counts, strict=True)]
        if s_out.status not in _USABLE or any(o.status not in _USABLE for o in i_outs):
            return None
        ind_total = sum(o.cost * o.used[0] for o in i_outs)
        ind_count = sum(o.used[0] for o in i_outs)
        return float(s_out.cost - ind_total / ind_count)
    if reselect:
        independent = [_fit_one(part, settings, initial) for part, _ in parts]
        if not all(f.usable for f in independent):
            return None
        shared = _shared_polynomial(
            [part for part, _ in parts], settings, initial, independent, beta
        )
        if not shared.usable:
            return None
        return shared.chi2 - sum(f.chi2 for f in independent)
    polys = []
    for (part, keep), m in zip(parts, mapped, strict=True):
        assert m.fit.selected is not None
        polys.append(_Polynomial.subset(part, m.fit.selected[keep], settings.degree))
    rows = _rows(initial)
    i_outs = [_fit_poly([poly], settings, rows if rows is not None else theta_s) for poly in polys]
    if any(o.status not in _USABLE for o in i_outs):
        return None
    start = np.array([o.params for o in i_outs])
    s_out = _fit_poly(polys, settings, start if rows is None else np.vstack([rows, start]))
    if s_out.status not in _USABLE:
        return None
    return s_out.chi2 - sum(o.chi2 for o in i_outs)


# ---------------------------------------------------------------------------------------------
# leave one size out, diagnostics, the two methods side by side
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LeaveOneSizeOut:
    """Refits without each size, and how the left-out size collapses onto the others.

    ``fits[k]`` is the fit without ``sizes[k]``; ``params`` ``(S, 3)`` their parameters;
    ``held_out_cost[k]`` the mean ``chi2`` of the points of ``sizes[k]`` against the master
    curve of the other sizes at ``fits[k]``'s parameters (Houdayer-Hartmann: the bracketing
    line of the other sizes; polynomial: the fitted polynomial, with its prediction variance,
    at the held-out points inside the range it was fitted on), over ``held_out_points[k]``
    points: about 1 when the held-out size collapses within its errors. ``spread`` is the range
    (max - min) of each parameter over the refits.
    """

    full: CollapseFit
    sizes: tuple[int, ...]
    fits: tuple[CollapseFit, ...]
    params: np.ndarray
    held_out_cost: np.ndarray
    held_out_points: np.ndarray
    spread: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _plain(self)
        return out


def _without_size_polynomial(
    data: ScalingData, full: CollapseFit, size: int, settings: _Settings
) -> tuple[CollapseFit, float, int]:
    """A polynomial refit on the full fit's window minus one size (from the full fit and the
    settings' random starts; the set is kept, so ``window_stable`` is None), and the held-out
    cost."""
    assert full.selected is not None
    keep = data.n != size
    part = data.subset(keep)
    fit = _fixed_window_fit(part, full.selected[keep], settings, None, full.params)
    if not fit.usable:
        return fit, math.nan, 0
    solution = _Polynomial.subset(part, full.selected[keep], settings.degree).solve(
        fit.p_c, fit.z, fit.nu
    )
    if solution is None:  # pragma: no cover - a usable fit has a finite objective
        return fit, math.nan, 0
    held = _Polynomial.subset(data, data.n == size, settings.degree)
    x, big_y, dy = _held_coordinates(held, fit)
    inside = (x >= solution.x.min()) & (x <= solution.x.max())
    if not inside.any():
        return fit, math.nan, 0
    mean, var = solution.predict(x[inside])
    chi2 = (big_y[inside] - mean) ** 2 / (dy[inside] ** 2 + var)
    return fit, float(np.mean(chi2)), int(inside.sum())


def _held_coordinates(
    held: _Polynomial, fit: CollapseFit
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amp = np.power(held.n_float, -fit.z)
    return (held.p - fit.p_c) * np.power(held.n_float, 1.0 / fit.nu), held.y * amp, held.se * amp


def leave_one_size_out(
    data: ScalingData,
    *,
    bounds: Any,
    method: str = HOUDAYER_HARTMANN,
    degree: int = DEFAULT_DEGREE,
    window: str | float = "overlap",
    initial: ArrayLike | None = None,
    starts: int = 8,
    seed: Any = 0,
    min_points: int | None = None,
    max_evaluations: int = 2000,
    window_iterations: int = 5,
) -> LeaveOneSizeOut:
    """Refit without each size in turn (held-out-size collapse diagnostics, v1.4 §5.8).

    Each refit starts from the full fit (when it has parameters) and ``starts`` random starts;
    a polynomial refit keeps the full fit's window minus the left-out size. With fewer than
    three sizes the refits have ``status = "too_few_sizes"`` (or ``"too_few_points"``).
    """
    settings = _settings(
        bounds,
        starts,
        seed,
        min_points,
        max_evaluations,
        method,
        degree,
        window,
        window_iterations,
    )
    full = _fit_one(data, settings, initial)
    start = full.params if full.usable else initial
    whole = _Collapse.of(data)
    sizes = tuple(int(s) for s in data.sizes)
    fits, held_cost, held_points = [], [], []
    for size in sizes:
        if settings.method == POLYNOMIAL and full.usable:
            fit, cost, points = _without_size_polynomial(data, full, size, settings)
            fits.append(fit)
            held_cost.append(cost)
            held_points.append(points)
            continue
        part = data.subset(data.n != size)
        if settings.method == POLYNOMIAL:
            fit = _fit_one(part, settings, start)
        else:
            fit = _collapse_fit(
                part, settings, _fit_hh([_Collapse.of(part)], [part.size], settings, start)
            )
        fits.append(fit)
        if fit.usable and settings.method == HOUDAYER_HARTMANN:
            chi2 = whole.chi2(fit.p_c, fit.z, fit.nu)[data.n == size]
            ok = ~np.isnan(chi2)
            held_cost.append(float(np.mean(chi2[ok])) if ok.any() else math.nan)
            held_points.append(int(ok.sum()))
        else:
            held_cost.append(math.nan)
            held_points.append(0)
    params = np.array([f.params for f in fits])
    with np.errstate(invalid="ignore"):
        spread = (
            np.nanmax(params, axis=0) - np.nanmin(params, axis=0)
            if np.isfinite(params).any(axis=1).any()
            else np.full(3, np.nan)
        )
    return LeaveOneSizeOut(
        full=full,
        sizes=sizes,
        fits=tuple(fits),
        params=params,
        held_out_cost=np.array(held_cost),
        held_out_points=np.array(held_points, dtype=np.int64),
        spread=spread,
    )


@dataclass(frozen=True, eq=False)
class CollapseDiagnostics:
    """What a fitted collapse rests on, point by point.

    ``bracketed`` and ``chi2`` per point at the fit (Houdayer-Hartmann: the points another size
    brackets and their ``chi2``; polynomial: the points in the window and their squared
    whitened residuals); ``coverage`` the fraction of points used and ``coverage_by_size``;
    ``drift_fit`` the refit without the smallest size and ``drift`` its parameters minus the
    fit's (``drift_in_se`` divided by a bootstrap's errors when one is given); the
    ``censored_fraction`` and ``horizon_fraction`` of every point as given, and their maxima;
    ``notes`` in words.
    """

    fit: CollapseFit
    n: np.ndarray
    p: np.ndarray
    bracketed: np.ndarray
    chi2: np.ndarray
    coverage: float
    coverage_by_size: dict[int, float]
    drift_fit: CollapseFit | None
    drift: np.ndarray | None
    drift_in_se: np.ndarray | None
    censored_fraction: np.ndarray | None
    horizon_fraction: np.ndarray | None
    max_censored_fraction: float | None
    max_horizon_fraction: float | None
    label: str | None
    estimator: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _plain(self)
        return out


def collapse_diagnostics(
    data: ScalingData,
    fit: CollapseFit,
    *,
    bootstrap: BootstrapFit | None = None,
    starts: int | None = None,
) -> CollapseDiagnostics:
    """Coverage, drift without the smallest size, and the censoring and horizon-peak fractions
    of every point (v1.4 §5.9: report them per condition).

    The drift refit uses the fit's own method and settings (bounds, ``min_points``, seed,
    budget; ``starts`` overrides its number of random starts; a polynomial refit keeps the fit's
    window minus the smallest size) and starts from its parameters. ``bootstrap`` (a
    :func:`bootstrap_fit` of the same data) turns the drift into units of its errors.
    """
    if not isinstance(fit, CollapseFit):
        raise ValueError("fit must be a CollapseFit of the same data")
    notes: list[str] = []
    polynomial = fit.method == POLYNOMIAL
    if fit.usable and polynomial:
        assert fit.selected is not None and fit.degree is not None
        bracketed = fit.selected
        chi2 = (
            polynomial_cost(
                data, fit.p_c, fit.z, fit.nu, degree=fit.degree, selected=fit.selected
            ).residuals
            ** 2
        )
    elif fit.usable:
        cc = collapse_cost(data, fit.p_c, fit.z, fit.nu)
        bracketed, chi2 = cc.bracketed, cc.chi2
    else:
        bracketed, chi2 = np.zeros(data.size, dtype=bool), np.full(data.size, np.nan)
        notes.append(f"the fit has no parameters (status {fit.status}): {fit.message}")
    coverage = float(bracketed.mean())
    by_size = {int(s): float(bracketed[data.n == s].mean()) for s in data.sizes}
    missing = int((~bracketed).sum())
    if fit.usable and missing:
        notes.append(
            f"{missing} of {data.size} points are outside the polynomial window"
            if polynomial
            else f"{missing} of {data.size} points are not bracketed by another size and do not "
            "enter the quality (the ends of the scaled range)"
        )
    if polynomial and fit.window_stable is False:
        notes.append(f"the window did not settle in {fit.window_iterations} fits")
    drift_fit = drift = drift_in_se = None
    if fit.usable and data.sizes.size >= 3:
        settings = _settings_of(fit, starts)
        smallest = int(data.sizes[0])
        if polynomial:
            drift_fit = _without_size_polynomial(data, fit, smallest, settings)[0]
        else:
            part = data.subset(data.n != smallest)
            out = _fit_hh([_Collapse.of(part)], [part.size], settings, fit.params)
            drift_fit = _collapse_fit(part, settings, out)
        if drift_fit.usable:
            drift = drift_fit.params - fit.params
            if bootstrap is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    drift_in_se = drift / bootstrap.se
        else:
            notes.append(f"the refit without n = {smallest} failed: {drift_fit.message}")
    elif data.sizes.size < 3:
        notes.append("fewer than three sizes: no drift with the smallest size removed")
    max_c = None if data.censored_fraction is None else float(data.censored_fraction.max())
    max_h = None if data.horizon_fraction is None else float(data.horizon_fraction.max())
    if max_c:
        notes.append(
            f"censored fractions up to {max_c:.3g}: y is '{data.estimator}', which must treat "
            "censoring explicitly (a mean of observed times is not the ensemble mean)"
        )
    if max_h:
        notes.append(
            f"peaks at the horizon in up to {max_h:.3g} of the trajectories: their peak times "
            "are lower bounds"
        )
    return CollapseDiagnostics(
        fit=fit,
        n=data.n,
        p=data.p,
        bracketed=bracketed,
        chi2=chi2,
        coverage=coverage,
        coverage_by_size=by_size,
        drift_fit=drift_fit,
        drift=drift,
        drift_in_se=drift_in_se,
        censored_fraction=data.censored_fraction,
        horizon_fraction=data.horizon_fraction,
        max_censored_fraction=max_c,
        max_horizon_fraction=max_h,
        label=data.label,
        estimator=data.estimator,
        notes=tuple(notes),
    )


@dataclass(frozen=True, eq=False)
class MethodComparison:
    """The Houdayer-Hartmann and polynomial fits of the same data, side by side.

    ``difference`` is ``polynomial.params - houdayer_hartmann.params``. With ``replicates > 0``
    both methods refit every graph-bootstrap replicate (the same draws, paired across ``p``):
    ``se_difference`` is the standard deviation of the replicate differences, ``z`` the
    difference in its units (``|z|`` of a few or less: the methods agree within the sampling
    spread), and ``se_houdayer_hartmann``, ``se_polynomial`` each method's bootstrap error.
    ``failed`` counts replicates where either refit gave no parameters. ``replicate_starts``
    and ``reselect_window`` are the refits' settings (:func:`bootstrap_fit`).
    """

    houdayer_hartmann: CollapseFit
    polynomial: CollapseFit
    difference: np.ndarray
    se_difference: np.ndarray
    z: np.ndarray
    se_houdayer_hartmann: np.ndarray
    se_polynomial: np.ndarray
    replicates: int
    failed: int
    seed: Any
    replicate_starts: int = 0
    reselect_window: bool = True

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _plain(self)
        return out


def _method_reference(
    hh: CollapseFit | None, settings: _Settings, initial: ArrayLike | None
) -> np.ndarray:
    """Where :func:`compare_methods` draws the polynomial's first window: ``initial``'s first
    row, else the Houdayer-Hartmann fit of the same data, else the box center."""
    rows = _rows(initial)
    if rows is not None:
        return rows[0]
    return hh.params if hh is not None and hh.usable else settings.box.mean(axis=1)


def compare_methods(
    data: ScalingData,
    *,
    bounds: Any,
    degree: int = DEFAULT_DEGREE,
    window: str | float = "overlap",
    replicates: int = 100,
    seed: Any = 0,
    initial: ArrayLike | None = None,
    starts: int = 8,
    min_points: int | None = None,
    max_evaluations: int = 2000,
    window_iterations: int = 5,
    replicate_starts: int | None = None,
    reselect_window: bool = True,
) -> MethodComparison:
    """Fit with both methods and compare their parameters against the bootstrap spread.

    The polynomial fit draws its first window at the Houdayer-Hartmann fit (unless ``initial``
    is given). With ``replicates > 0`` (and per-graph values in the data) each graph-bootstrap
    replicate is fitted by both methods by the same procedure as the data (as in
    :func:`bootstrap_fit`: ``replicate_starts`` random starts, by default as many as ``starts``;
    the replicate's polynomial window drawn at its own Houdayer-Hartmann fit and redrawn until it
    settles, or with ``reselect_window=False`` the data fit's window); the spread of the paired
    replicate differences is the error of the difference, so ``z = difference / se_difference``
    measures the disagreement in units of what resampling the graphs explains (ruling R-FSS: the
    two methods' agreement is a reported diagnostic).
    """
    hh_settings = _settings(
        bounds,
        starts,
        seed,
        min_points,
        max_evaluations,
        HOUDAYER_HARTMANN,
        degree,
        window,
        window_iterations,
    )
    poly_settings = replace(hh_settings, method=POLYNOMIAL)
    count = _replicate_count(replicates, 0)
    rep_starts = _replicate_starts(replicate_starts, hh_settings.starts)
    reselect = _reselect(reselect_window)
    hh = _fit_one(data, hh_settings, initial)
    poly = _fit_one(data, poly_settings, initial, _method_reference(hh, hh_settings, initial))
    difference = poly.params - hh.params
    nan3 = np.full(3, np.nan)
    se_diff, se_hh, se_poly = nan3, nan3, nan3
    failed = 0
    if count and hh.usable and poly.usable:
        draws = _draws_by_size([data], count, seed)
        ystar, sestar = _replicate_values(data, draws)
        hh_rep = replace(hh_settings, starts=rep_starts)
        poly_rep = replace(poly_settings, starts=rep_starts)
        pairs = []
        for b in range(count):
            a = _refit(data, hh, hh_rep, ystar[b], sestar[b], initial=initial, reselect=True)
            c = _refit(
                data,
                poly,
                poly_rep,
                ystar[b],
                sestar[b],
                initial=initial,
                reselect=reselect,
                reference=_method_reference(None if a is None else a.fit, hh_rep, initial),
            )
            if a is None or c is None or not a.fit.usable or not c.fit.usable:
                failed += 1
                continue
            pairs.append((a.fit.params, c.fit.params))
        if len(pairs) > 1:
            first = np.array([pa for pa, _ in pairs])
            second = np.array([pc for _, pc in pairs])
            se_diff = (second - first).std(axis=0, ddof=1)
            se_hh = first.std(axis=0, ddof=1)
            se_poly = second.std(axis=0, ddof=1)
    elif count:
        failed = count
    with np.errstate(divide="ignore", invalid="ignore"):
        z = difference / se_diff
    return MethodComparison(
        houdayer_hartmann=hh,
        polynomial=poly,
        difference=difference,
        se_difference=se_diff,
        z=z,
        se_houdayer_hartmann=se_hh,
        se_polynomial=se_poly,
        replicates=count,
        failed=failed,
        seed=seed,
        replicate_starts=rep_starts,
        reselect_window=reselect,
    )


@dataclass(frozen=True, eq=False)
class DegreeScan:
    """Polynomial fits of the same data at several degrees: the degree's effect.

    ``fits[k]`` at ``degrees[k]``; ``params`` ``(K, 3)``, ``chi2_per_dof`` and ``dof`` per
    degree; ``spread`` the range of each parameter over the usable fits. A degree too low for
    the master curve shows as ``chi2_per_dof`` well above 1 and parameters that move with the
    degree; once the degree is enough, the parameters settle.
    """

    degrees: tuple[int, ...]
    fits: tuple[CollapseFit, ...]
    params: np.ndarray
    chi2_per_dof: np.ndarray
    dof: np.ndarray
    spread: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _plain(self)
        return out


def polynomial_degree_scan(
    data: ScalingData,
    *,
    bounds: Any,
    degrees: Sequence[int] = (1, 2, 3, 4, 5),
    window: str | float = "overlap",
    initial: ArrayLike | None = None,
    starts: int = 8,
    seed: Any = 0,
    max_evaluations: int = 2000,
    window_iterations: int = 5,
) -> DegreeScan:
    """Fit the polynomial method at each degree (the same first window for all of them).

    The first window is drawn at ``initial`` or at the Houdayer-Hartmann fit, once; each degree
    then settles its own window. Report the scan with a polynomial fit (ruling R-FSS: the
    degree's effect).
    """
    chosen = tuple(_degree(d) for d in degrees)
    if not chosen:
        raise ValueError("degrees must not be empty")
    base = _settings(
        bounds,
        starts,
        seed,
        None,
        max_evaluations,
        POLYNOMIAL,
        chosen[0],
        window,
        window_iterations,
    )
    reference = _reference(data, base, initial) if data.sizes.size >= 2 else None
    fits = tuple(_fit_one(data, replace(base, degree=d), initial, reference) for d in chosen)
    params = np.array([f.params for f in fits])
    chi2_per_dof = np.array([f.chi2_per_dof for f in fits])
    dof = np.array([f.dof if f.dof is not None else 0 for f in fits], dtype=np.int64)
    ok = np.isfinite(params).all(axis=1)
    spread = params[ok].max(axis=0) - params[ok].min(axis=0) if ok.any() else np.full(3, np.nan)
    return DegreeScan(
        degrees=chosen, fits=fits, params=params, chi2_per_dof=chi2_per_dof, dof=dof, spread=spread
    )
