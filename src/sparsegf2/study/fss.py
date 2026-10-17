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
errors and grows as the collapse fails.

**Fits.** :func:`fit_collapse` minimizes ``S`` over a box of ``(p_c, z, nu)`` with
:func:`minimize_bounded` (Nelder–Mead in the unbounded variables ``phi`` of
``theta = lo + (hi - lo)(1 + sin phi)/2``, from several starts; deterministic given its seed).
:func:`fit_shared` fits one ``(p_c, z, nu)`` to several observables (each keeps its own ``F``),
and :func:`compare_shared_independent` tests that hypothesis against independent fits with a
bootstrap null built by graph-cluster resampling paired across ``p``. :func:`leave_one_size_out`
refits without each size; :func:`bootstrap_fit` resamples graphs once per ``(n, beta)`` and
recomputes every rate's mean from the same graphs; :func:`collapse_diagnostics` reports
bracketing coverage, the drift of the fit when the smallest size is removed, and passes the
censoring and horizon-peak fractions of every point through.

The inputs are nested means and their graph-cluster errors (:mod:`.stats`), gathered in a
:class:`ScalingData` (one observable at one ``beta``). Exploratory and confirmatory analyses are
told apart only by the recorded ``label``. ``docs/study/07-finite-size-scaling.md`` has the
mathematics, the proofs and worked examples.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from . import stats as _stats

__all__ = [
    "PARAMETERS",
    "STATUSES",
    "BootstrapFit",
    "CollapseCost",
    "CollapseDiagnostics",
    "CollapseFit",
    "LeaveOneSizeOut",
    "MinimizeResult",
    "ScalingData",
    "SharedComparison",
    "SharedFit",
    "bootstrap_fit",
    "collapse_cost",
    "collapse_diagnostics",
    "compare_shared_independent",
    "fit_collapse",
    "fit_shared",
    "leave_one_size_out",
    "minimize_bounded",
    "scaled_coordinates",
    "scaling_data",
]

#: The fitted parameters, in this order everywhere (``params`` arrays, bounds, replicates).
PARAMETERS = ("p_c", "z", "nu")

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


@dataclass(frozen=True, eq=False)
class ScalingData:
    """The points ``(n, p, y, se)`` of one observable at one ``beta``, ready for a collapse.

    ``y`` is the ensemble estimate of the time observable at size ``n`` and rate ``p`` (for
    example a nested mean, or a restricted mean or median from a survival curve: say which in
    ``estimator``) and ``se`` its standard error (positive). ``graphs`` and ``graph_means`` (one
    array per point) are the per-graph values behind ``y``: they let :func:`bootstrap_fit` and
    :func:`compare_shared_independent` resample graphs; every point of a size must have the same
    graph labels, because the rates were measured on the same graphs.
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
    ``phi`` unbounded: a smooth map onto the closed box that reaches both bounds (MINUIT's
    transformation), so Nelder–Mead runs without constraints in ``phi``. A coordinate with
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


def _min_points(min_points: int | None, total: int) -> int:
    if min_points is None:
        return max(3, math.ceil(total / 2))
    if isinstance(min_points, bool) or not isinstance(min_points, (int, np.integer)):
        raise ValueError(f"min_points must be a positive integer or None, not {min_points!r}")
    if min_points < 1:
        raise ValueError("min_points must be at least 1")
    return int(min_points)


def _objective(collapses: Sequence[_Collapse], needs: Sequence[int]) -> Callable[..., float]:
    """The pooled quality: the mean ``chi2`` over every observable's bracketed points.

    ``+inf`` when some observable has fewer bracketed points than it needs.
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


@dataclass(frozen=True)
class _Settings:
    box: np.ndarray
    starts: int
    seed: Any
    min_points: int | None
    max_evaluations: int


def _settings(
    bounds: Any, starts: int, seed: Any, min_points: int | None, max_evaluations: int
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
    _stats._seed_sequence(seed)
    return _Settings(box, int(starts), seed, min_points, int(max_evaluations))


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


def _fit(
    collapses: Sequence[_Collapse],
    totals: Sequence[int],
    settings: _Settings,
    initial: ArrayLike | None,
    starts: int | None = None,
) -> _Outcome:
    needs = tuple(_min_points(settings.min_points, t) for t in totals)
    nan3 = np.full(3, np.nan)
    sizes = [c.sizes.size for c in collapses]
    if min(sizes) < 2:
        return _Outcome(
            nan3,
            math.nan,
            tuple(0 for _ in collapses),
            "too_few_sizes",
            "a collapse compares sizes: every observable needs at least two sizes (three to "
            f"constrain p_c, z and nu); got {sizes}",
            (),
            0,
            0,
            needs,
        )
    short = [t for t, need in zip(totals, needs, strict=True) if t < need]
    if short:
        return _Outcome(
            nan3,
            math.nan,
            tuple(0 for _ in collapses),
            "too_few_points",
            f"fewer points ({short}) than the {needs} bracketed points a fit needs (min_points)",
            (),
            0,
            0,
            needs,
        )
    objective = _objective(collapses, needs)
    result = minimize_bounded(
        objective,
        settings.box,
        initial=initial,
        starts=settings.starts if starts is None else starts,
        seed=settings.seed,
        max_evaluations=settings.max_evaluations,
    )
    if not math.isfinite(result.fun):
        return _Outcome(
            nan3,
            math.nan,
            tuple(0 for _ in collapses),
            "too_few_points",
            f"no parameters in the box bracket at least {list(needs)} points (min_points) of "
            "every observable: the sizes' scaled ranges do not overlap enough; widen the box or "
            "the rate window, or add rates",
            (),
            result.evaluations,
            result.starts,
            needs,
        )
    params = result.x
    used = tuple(int(np.sum(~np.isnan(c.chi2(*(float(v) for v in params))))) for c in collapses)
    at_bound = tuple(name for name, flag in zip(PARAMETERS, result.at_bound, strict=True) if flag)
    if not result.converged:
        status, message = "not_converged", "the evaluation budget ran out before convergence"
    elif at_bound:
        status = "at_bound"
        message = f"{', '.join(at_bound)} at a bound of the box: widen it and refit"
    else:
        status, message = "ok", ""
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


@dataclass(frozen=True, eq=False)
class CollapseFit:
    """A fitted collapse of one observable at one ``beta``.

    ``p_c``, ``z``, ``nu`` (NaN when the status is ``too_few_sizes`` or ``too_few_points``),
    the quality ``cost`` at them, ``points_used`` of ``total_points`` (bracketed points),
    ``status`` (one of :data:`STATUSES`) with a ``message``, the parameters ``at_bound``, and the
    settings that produced it (``bounds``; ``min_points``, the bracketed points required here,
    from ``min_points_setting``, the value given, None for the default rule; ``random_starts``
    as given and ``starts`` run; ``seed``; ``max_evaluations``) with the data's ``observable``,
    ``beta``, ``label`` and ``estimator``.
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

    @property
    def params(self) -> np.ndarray:
        """``[p_c, z, nu]``."""
        return np.array([self.p_c, self.z, self.nu])

    @property
    def usable(self) -> bool:
        """True when the fit returned parameters (``ok``, ``at_bound`` or ``not_converged``)."""
        return self.status in ("ok", "at_bound", "not_converged")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _plain(self)
        return out


def _collapse_fit(data: ScalingData, settings: _Settings, outcome: _Outcome) -> CollapseFit:
    p_c, z, nu = (float(v) for v in outcome.params)
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
    )


def _settings_of(fit: CollapseFit, starts: int | None = None) -> _Settings:
    """The settings a fit was made with (``starts`` overrides its random starts)."""
    return _Settings(
        np.array(fit.bounds),
        fit.random_starts if starts is None else int(starts),
        fit.seed,
        fit.min_points_setting,
        fit.max_evaluations,
    )


def fit_collapse(
    data: ScalingData,
    *,
    bounds: Any,
    initial: ArrayLike | None = None,
    starts: int = 8,
    seed: Any = 0,
    min_points: int | None = None,
    max_evaluations: int = 2000,
) -> CollapseFit:
    """Fit ``(p_c, z, nu)`` by minimizing the Houdayer–Hartmann quality (:func:`collapse_cost`).

    ``bounds``: ``{"p_c": (lo, hi), "z": (lo, hi), "nu": (lo, hi)}`` or three pairs in that
    order (``lo == hi`` fixes a parameter; ``nu``'s lower bound must be positive). A parameter
    vector is admissible only when at least ``min_points`` points are bracketed (default
    ``max(3, ceil(P/2))`` of the ``P`` points), since the quality of a few bracketed points
    says little; inadmissible vectors count as ``+inf``. The search is
    :func:`minimize_bounded` from ``initial`` (or the box center) and ``starts`` random starts
    drawn from ``seed``. The status says whether the result is usable (:data:`STATUSES`); a
    fit never raises for lack of data.
    """
    settings = _settings(bounds, starts, seed, min_points, max_evaluations)
    outcome = _fit([_Collapse.of(data)], [data.size], settings, initial)
    return _collapse_fit(data, settings, outcome)


@dataclass(frozen=True, eq=False)
class SharedFit:
    """One ``(p_c, z, nu)`` fitted to several observables, each with its own scaling function.

    ``cost`` is the pooled quality (the mean ``chi2`` over all observables' bracketed points),
    ``costs`` and ``points_used`` per observable at the shared parameters.
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

    @property
    def params(self) -> np.ndarray:
        return np.array([self.p_c, self.z, self.nu])

    @property
    def usable(self) -> bool:
        return self.status in ("ok", "at_bound", "not_converged")

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
    items: Sequence[ScalingData], settings: _Settings, outcome: _Outcome, beta: str | None
) -> SharedFit:
    params = outcome.params
    costs: list[float] = []
    if np.all(np.isfinite(params)):
        for d in items:
            cc = _Collapse.of(d).chi2(*(float(v) for v in params))
            ok = ~np.isnan(cc)
            costs.append(float(np.mean(cc[ok])) if ok.any() else math.nan)
    else:
        costs = [math.nan] * len(items)
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
    )


def fit_shared(
    datasets: Sequence[ScalingData],
    *,
    bounds: Any,
    initial: ArrayLike | None = None,
    starts: int = 8,
    seed: Any = 0,
    min_points: int | None = None,
    max_evaluations: int = 2000,
) -> SharedFit:
    """Fit one ``(p_c, z, nu)`` to several observables at one ``beta`` (the shared hypothesis).

    Every observable keeps its own scaling function: the quality is evaluated per observable
    and pooled, ``S_shared = sum_o sum_i chi2_oi / sum_o N_o``, with each observable needing its
    own ``min_points`` bracketed points. Same options as :func:`fit_collapse`.
    """
    items = _datasets(datasets)
    beta = _common_beta(items)
    settings = _settings(bounds, starts, seed, min_points, max_evaluations)
    outcome = _fit([_Collapse.of(d) for d in items], [d.size for d in items], settings, initial)
    return _shared_fit(items, settings, outcome, beta)


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
        ystar[:, i] = m
        sestar[:, i] = s
    return ystar, sestar


def _replicate_collapse(
    n: np.ndarray, p: np.ndarray, y: np.ndarray, se: np.ndarray
) -> tuple[_Collapse, int] | None:
    keep = np.isfinite(y) & np.isfinite(se) & (se > 0)
    if not keep.any():
        return None
    return _Collapse(n[keep], p[keep], y[keep], se[keep]), int(keep.sum())


@dataclass(frozen=True, eq=False)
class BootstrapFit:
    """Graph-cluster bootstrap of a collapse fit, paired across ``p``.

    ``fit`` is the fit to the data; ``replicates`` ``(B, 3)`` the refits of the resampled data
    (NaN rows where a refit returned no parameters), ``costs`` and ``statuses`` theirs; ``se``
    the standard deviation of the usable replicates per parameter and ``lower``/``upper`` their
    percentile interval at ``level``; ``failed`` the replicates without parameters and
    ``at_bound`` those with a parameter on its bound.
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

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _plain(self)
        return out


def bootstrap_fit(
    data: ScalingData,
    *,
    bounds: Any,
    replicates: int = 200,
    seed: Any = 0,
    level: float = 0.95,
    initial: ArrayLike | None = None,
    starts: int = 8,
    min_points: int | None = None,
    max_evaluations: int = 2000,
) -> BootstrapFit:
    """The uncertainty of ``(p_c, z, nu)`` by resampling graphs (v1.4 §5.8, §9).

    For every size ``n`` the graphs are resampled once per replicate
    (``graph_bootstrap(labels_n, replicates, seed=[seed, n])``) and every rate's mean and error
    at that size are recomputed from the same drawn graphs, so the coupling of the rates
    (shared graphs, gates and mask uniforms) is kept. Each replicate is refitted from the data's
    fit (no random starts); a point whose replicate error is 0 or undefined is left out of that
    replicate. Returns the replicate parameters, their standard deviations and percentile
    intervals. Needs the per-graph values (:func:`scaling_data`).
    """
    settings = _settings(bounds, starts, seed, min_points, max_evaluations)
    lvl = _stats._level(level)
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, (int, np.integer))
        or replicates < 2
    ):
        raise ValueError("replicates must be an integer >= 2")
    draws = _draws_by_size([data], int(replicates), seed)
    outcome = _fit([_Collapse.of(data)], [data.size], settings, initial)
    fit = _collapse_fit(data, settings, outcome)
    reps = np.full((int(replicates), 3), np.nan)
    costs = np.full(int(replicates), np.nan)
    statuses: list[str] = []
    if fit.usable:
        ystar, sestar = _replicate_values(data, draws)
        for b in range(int(replicates)):
            prepared = _replicate_collapse(data.n, data.p, ystar[b], sestar[b])
            if prepared is None:
                statuses.append("too_few_points")
                continue
            collapse, count = prepared
            out = _fit([collapse], [count], settings, fit.params, starts=0)
            statuses.append(out.status)
            if out.status in ("ok", "at_bound", "not_converged"):
                reps[b] = out.params
                costs[b] = out.cost
    else:
        statuses = [fit.status] * int(replicates)
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
    )


# ---------------------------------------------------------------------------------------------
# shared versus independent exponents
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SharedComparison:
    """Shared ``(p_c, z, nu)`` against independent fits, with a bootstrap null.

    ``delta = shared.cost - independent_cost`` (``independent_cost`` pools the independent fits'
    ``chi2`` the same way), ``null`` its replicates under the shared hypothesis (NaN for failed
    refits), ``p_value = (1 + #{null >= delta}) / (1 + B')`` over the ``B'`` usable replicates,
    and ``prefers``: ``"independent"`` when ``p_value <= alpha``, ``"shared"`` otherwise, and
    ``"undetermined"`` when a fit gave no parameters. ``message`` explains an undetermined
    result.
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
    replicates: int = 200,
    seed: Any = 0,
    alpha: float = 0.05,
    initial: ArrayLike | None = None,
    starts: int = 8,
    min_points: int | None = None,
    max_evaluations: int = 2000,
) -> SharedComparison:
    """Test one shared ``(p_c, z, nu)`` against independent exponents (v1.4 §5.8, §5.9).

    The statistic is the increase of the pooled quality when the parameters are shared,
    ``delta = S_shared(theta_s) - S_independent`` (``S_independent`` the pooled quality of the
    independent fits ``theta_o``). Its null distribution is built by graph-cluster resampling
    of data that satisfy the shared hypothesis exactly: every observable ``o`` is mapped point
    by point to

        p' = p_c^s + (p - p_c^o) n^(1/nu_o - 1/nu_s),   y' = y n^(z_s - z_o)

    (and its graph means and errors by the same factor), which gives at ``theta_s`` exactly the
    scaled coordinates the observable has at its own ``theta_o``, so its collapse quality is
    kept and the shared parameters are exact for the mapped data. Each replicate resamples the
    graphs once per size (``graph_bootstrap(labels_n, B, seed=[seed, n])``), applies the draws
    to every rate and observable, recomputes the means and errors, refits shared and
    independent parameters (starting from ``theta_s``) and records ``delta*``. The p-value is
    ``(1 + #{delta* >= delta}) / (1 + B')``. Needs the per-graph values of every point.
    """
    items = _datasets(datasets)
    beta = _common_beta(items)
    settings = _settings(bounds, starts, seed, min_points, max_evaluations)
    if not (isinstance(alpha, float) and 0 < alpha < 1):
        raise ValueError(f"alpha must be a probability strictly between 0 and 1, not {alpha!r}")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, (int, np.integer))
        or replicates < 1
    ):
        raise ValueError("replicates must be a positive integer")
    reps = int(replicates)
    draws = _draws_by_size(items, reps, seed)
    collapses = [_Collapse.of(d) for d in items]
    shared_out = _fit(collapses, [d.size for d in items], settings, initial)
    shared = _shared_fit(items, settings, shared_out, beta)
    independent = tuple(
        _collapse_fit(d, settings, _fit([c], [d.size], settings, initial))
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
        )
    delta = shared.cost - independent_cost
    theta_s = shared.params
    mapped = []
    for d, f in zip(items, independent, strict=True):
        n = d.n.astype(np.float64)
        factor = n ** (theta_s[1] - f.z)
        p_new = theta_s[0] + (d.p - f.p_c) * n ** (1.0 / f.nu - 1.0 / theta_s[2])
        ystar, sestar = _replicate_values(d, draws)
        mapped.append((d.n, p_new, ystar * factor, sestar * factor))
    for b in range(reps):
        prepared = [_replicate_collapse(n, p_new, ys[b], ss[b]) for n, p_new, ys, ss in mapped]
        if any(x is None for x in prepared):
            continue
        cols = [x[0] for x in prepared if x is not None]
        counts = [x[1] for x in prepared if x is not None]
        s_out = _fit(cols, counts, settings, theta_s, starts=0)
        i_outs = [
            _fit([c], [k], settings, theta_s, starts=0) for c, k in zip(cols, counts, strict=True)
        ]
        if s_out.status not in ("ok", "at_bound", "not_converged") or any(
            o.status not in ("ok", "at_bound", "not_converged") for o in i_outs
        ):
            continue
        ind_total = sum(o.cost * o.used[0] for o in i_outs)
        ind_count = sum(o.used[0] for o in i_outs)
        null[b] = s_out.cost - ind_total / ind_count
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
    )


# ---------------------------------------------------------------------------------------------
# leave one size out, diagnostics
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LeaveOneSizeOut:
    """Refits without each size, and how the left-out size collapses onto the others.

    ``fits[k]`` is the fit without ``sizes[k]``; ``params`` ``(S, 3)`` their parameters;
    ``held_out_cost[k]`` the mean ``chi2`` of the points of ``sizes[k]`` against the master
    curve of the other sizes at ``fits[k]``'s parameters (about 1 when the held-out size
    collapses within its errors), over ``held_out_points[k]`` bracketed points; ``spread`` the
    range (max - min) of each parameter over the refits.
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


def leave_one_size_out(
    data: ScalingData,
    *,
    bounds: Any,
    initial: ArrayLike | None = None,
    starts: int = 8,
    seed: Any = 0,
    min_points: int | None = None,
    max_evaluations: int = 2000,
) -> LeaveOneSizeOut:
    """Refit without each size in turn (held-out-size collapse diagnostics, v1.4 §5.8).

    Each refit starts from the full fit (when it has parameters) and ``starts`` random starts.
    With fewer than three sizes the refits have ``status = "too_few_sizes"``.
    """
    settings = _settings(bounds, starts, seed, min_points, max_evaluations)
    full_out = _fit([_Collapse.of(data)], [data.size], settings, initial)
    full = _collapse_fit(data, settings, full_out)
    start = full.params if full.usable else initial
    whole = _Collapse.of(data)
    sizes = tuple(int(s) for s in data.sizes)
    fits, held_cost, held_points = [], [], []
    for size in sizes:
        part = data.subset(data.n != size)
        out = _fit([_Collapse.of(part)], [part.size], settings, start)
        fit = _collapse_fit(part, settings, out)
        fits.append(fit)
        if fit.usable:
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

    ``bracketed`` and ``chi2`` per point at the fit; ``coverage`` the bracketed fraction and
    ``coverage_by_size``; ``drift_fit`` the refit without the smallest size and ``drift`` its
    parameters minus the fit's (``drift_in_se`` divided by a bootstrap's errors when one is
    given); the ``censored_fraction`` and ``horizon_fraction`` of every point as given, and their
    maxima; ``notes`` in words.
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
    """Bracketing coverage, drift without the smallest size, and the censoring and
    horizon-peak fractions of every point (v1.4 §5.9: report them per condition).

    The drift refit uses the fit's own settings (bounds, ``min_points``, seed, budget; ``starts``
    overrides its number of random starts) and starts from its parameters. ``bootstrap`` (a
    :func:`bootstrap_fit` of the same data) turns the drift into units of its errors.
    """
    if not isinstance(fit, CollapseFit):
        raise ValueError("fit must be a CollapseFit of the same data")
    notes: list[str] = []
    if fit.usable:
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
            f"{missing} of {data.size} points are not bracketed by another size and do not "
            "enter the quality (the ends of the scaled range)"
        )
    drift_fit = drift = drift_in_se = None
    if fit.usable and data.sizes.size >= 3:
        settings = _settings_of(fit, starts)
        part = data.subset(data.n != data.sizes[0])
        out = _fit([_Collapse.of(part)], [part.size], settings, fit.params)
        drift_fit = _collapse_fit(part, settings, out)
        if drift_fit.usable:
            drift = drift_fit.params - fit.params
            if bootstrap is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    drift_in_se = drift / bootstrap.se
        else:
            notes.append(f"the refit without n = {data.sizes[0]} failed: {drift_fit.message}")
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
