"""Observable-agnostic finite-size-scaling (FSS) data-collapse tools.

Locates a critical point ``p_c`` and the scaling exponents ``(nu, z)`` of a
continuous transition from finite-size data, for *any* scalar observable that
obeys the single-parameter FSS ansatz

    O(p, L) = L**z * F( (p - p_c) * L**(1/nu) ),                            (1)

where ``F`` is a universal (size-independent) scaling function, ``nu`` is the
correlation-length exponent, and ``z`` is the scaling dimension of the observable
``O`` (the dynamical exponent for a divergent characteristic time such as the
single-reference purification time; ``z = 0`` for a Binder-type dimensionless
cumulant; ``z = -beta/nu`` for an order parameter; ...).  The size-invariant
combination that actually collapses / crosses is therefore

    Y = O / L**z          (equivalently  Y = log O - z log L),

**not** ``O`` itself and **not** ``O / L`` unless ``z`` happens to equal 1.
Dividing by the fitted ``L**z`` is what makes a crossing/collapse of a
dimensionful observable well posed.

Nothing here is tied to a particular observable or ensemble: the input for a
single "cell" (all sizes at one external control value, e.g. one Watts-Strogatz
``beta``) is a plain mapping ``curves: {L: {p: value}}``.  The same procedure
serves the single-reference purification time, the code dimension, the half-cut
entropy, the average stabilizer weight, or any other scalar obeying (1).

Loss
----
The canonical residual is the Houdayer-Hartmann leave-one-size-out master-curve
loss (:func:`master_curve_loss`, per-point form :func:`hh_loss`): for each size,
predict its points by linear interpolation of the master curve built from the
*other* sizes, sum the squared residuals over the points inside the overlap, and
normalize by the point count and ``var(Y)``.  It is non-parametric in ``F`` (no
functional form is imposed on the scaling function) and is minimized by
scipy's Nelder-Mead from a grid of starts.

Estimators
----------
* :func:`robust_pc` -- a very robust ``p_c`` from the crossing spread: the ``z``
  that makes the consecutive-size ``O / L**z`` curves cross at a common ``p``
  (minimum spread) gives a ``p_c`` that barely depends on ``nu``.
* :func:`fit_fixed_pc` -- ``(nu, z)`` at a pinned ``p_c`` (far better conditioned
  than a free 3-parameter fit: no ``p_c``-``nu``-``z`` correlation).
* :func:`fit_three_param` -- the full ``(p_c, nu, z)`` joint fit, screening a
  multi-``z``-start grid so the high-``z`` global minimum is not missed.
* :func:`smooth_pc_collapse` -- fit ``p_c`` per control value, Savitzky-Golay
  smooth ``p_c`` across control values, and refit ``(nu, z)`` at the smoothed
  ``p_c`` (cuts the control-to-control ``z`` scatter that leaks in through the
  ``(p_c, z)`` correlation).
* :func:`graph_bootstrap` -- a disorder-aware bootstrap: resample the per-graph
  observable values with replacement (the same graph-index set across all ``p``
  of a size), recompute the per-cell median curve, and refit; the std over
  resamples is the uncertainty.

Only :mod:`numpy` is required at import time; the fitters lazily import
:mod:`scipy.optimize` / :mod:`scipy.signal` when called (install with
``pip install sparsegf2[analysis]``), preserving the package's numpy+numba
runtime floor.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..errors import InvalidArgumentError


def _minimize(*args, **kwargs):
    """Lazy ``scipy.optimize.minimize`` (scipy is an optional extra)."""
    try:
        from scipy.optimize import minimize
    except ImportError as e:  # pragma: no cover
        raise InvalidArgumentError(
            "the FSS fitters need the optional 'analysis' extra (scipy); "
            "install with: pip install sparsegf2[analysis]"
        ) from e
    return minimize(*args, **kwargs)


# A "cell" is all sizes at one external control value:  {L: {p: value}}.
Curves = Mapping[int, Mapping[float, float]]
# Raw (per-graph) cell:  {L: {p: [per-graph values]}}.
RawCell = Mapping[int, Mapping[float, Any]]

_LOSS_SENTINEL = 1e12


# ----------------------------------------------------------------------
# Houdayer-Hartmann leave-one-size-out master-curve loss
# ----------------------------------------------------------------------


def hh_loss(sizes, ps, ys, pc_of_point, nu, z, *, use_log=True) -> float:
    """Leave-one-size-out master-curve residual with a *per-point* ``p_c``.

    ``pc_of_point`` is an array aligned with ``sizes`` / ``ps`` / ``ys`` giving
    ``p_c(L)`` at each point (a scalar ``p_c`` is the special case of a constant
    array; see :func:`master_curve_loss`). For every distinct size, predict its
    points by linear interpolation of the collapse curve built from the *other*
    sizes, over the overlapping range in the scaling variable, and accumulate the
    squared residuals. Lower is better; a large finite sentinel is returned for
    degenerate inputs (``nu <= 0``, zero variance, fewer than two sizes).

    With ``use_log=True`` (the default), points with ``y <= 0`` are **dropped**
    before fitting: an exact zero (e.g. the code dimension's median in the
    area-law phase, or a censoring floor) carries no information about the
    multiplicative scaling form and its clipped logarithm would otherwise enter
    as a huge outlier that dominates the loss. If fewer than 4 positive points
    or fewer than 2 sizes survive, the sentinel is returned.
    """
    # nu <= 0 is nonsensical; a tiny positive nu makes L**(1/nu) overflow to inf,
    # which is a degenerate collapse either way -- reject both without warning.
    if not (nu >= 1e-3):
        return _LOSS_SENTINEL
    sizes = np.asarray(sizes, float)
    ps = np.asarray(ps, float)
    ys = np.asarray(ys, float)
    pcl = np.asarray(pc_of_point, float)
    if use_log:
        # Drop y <= 0: no information for a multiplicative scaling form, and a
        # clipped log would enter as a ~e^-27 outlier dominating var(y) and the
        # residuals (silently wrecking the fit). See the docstring.
        pos = ys > 0
        if pos.sum() < 4 or np.unique(sizes[pos]).size < 2:
            return _LOSS_SENTINEL
        if not pos.all():
            sizes, ps, ys, pcl = sizes[pos], ps[pos], ys[pos], pcl[pos]
    log_l = np.log(sizes)
    with np.errstate(over="ignore", invalid="ignore"):
        x = (ps - pcl) * sizes ** (1.0 / nu)
    if not np.all(np.isfinite(x)):
        return _LOSS_SENTINEL
    y = (np.log(ys) - z * log_l) if use_log else ys / sizes**z
    var_y = float(np.var(y))
    if not np.isfinite(var_y) or var_y == 0.0:
        return _LOSS_SENTINEL
    uniq = np.unique(sizes)
    if uniq.size < 2:
        return _LOSS_SENTINEL
    loss = 0.0
    npts = 0
    for size in uniq:
        m_in = sizes == size
        m_out = ~m_in
        if m_out.sum() < 2 or m_in.sum() < 1:
            continue
        order = np.argsort(x[m_out])
        ox, oy = x[m_out][order], y[m_out][order]
        in_x, in_y = x[m_in], y[m_in]
        overlap = (in_x >= ox[0]) & (in_x <= ox[-1])
        if not overlap.any():
            continue
        loss += float(np.sum((in_y[overlap] - np.interp(in_x[overlap], ox, oy)) ** 2))
        npts += int(overlap.sum())
    return (loss / npts) / var_y if npts else _LOSS_SENTINEL


def master_curve_loss(sizes, ps, ys, pc, nu, z, *, use_log=True) -> float:
    """Houdayer-Hartmann collapse loss for a **scalar** ``p_c``.

    A thin wrapper over :func:`hh_loss` that broadcasts the scalar ``pc`` to a
    per-point array. Use this for an ordinary single-``p_c`` collapse; use
    :func:`hh_loss` directly when ``p_c`` drifts with the system size.
    """
    sizes = np.asarray(sizes, float)
    return hh_loss(sizes, ps, ys, np.full_like(sizes, float(pc)), nu, z, use_log=use_log)


# ----------------------------------------------------------------------
# curves <-> flat per-point arrays
# ----------------------------------------------------------------------


def _arrays(curves: Curves) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten ``{L: {p: value}}`` to aligned ``(sizes, ps, ys)`` arrays."""
    sizes: list[float] = []
    ps: list[float] = []
    ys: list[float] = []
    for size, curve in curves.items():
        for p, value in curve.items():
            sizes.append(float(size))
            ps.append(float(p))
            ys.append(float(value))
    return np.array(sizes, float), np.array(ps, float), np.array(ys, float)


# ----------------------------------------------------------------------
# robust p_c from the crossing spread
# ----------------------------------------------------------------------


def robust_pc(curves: Curves, *, z_grid=None) -> tuple[float, float]:
    """Robust ``p_c`` (and crossing ``z``) from the consecutive-size crossings.

    For each candidate ``z``, rescale every size's curve to ``O / L**z`` and find
    the first crossing of each consecutive size pair. The ``z`` whose crossings
    have the smallest spread (they should all coincide at ``p_c``) gives a ``p_c``
    that barely depends on ``nu`` -- it is read straight off the data, with no
    functional form assumed. Returns ``(p_c, z_crossing)``, or ``(nan, nan)`` if
    no ``z`` yields enough coincident crossings: at least
    ``max(2, len(sizes) - 2)`` of the ``len(sizes) - 1`` consecutive pairs must
    cross, so **at least 3 sizes** are required (4+ recommended -- with 3 sizes
    the "spread" is the gap of just two crossings).

    ``z_grid`` overrides the default scan (``np.linspace(0.1, 1.15, 40)``); pass
    a wider grid when the expected ``z`` may exceed 1.15 (the fitters allow z up
    to ~1.2-1.25).
    """
    z_grid = np.linspace(0.1, 1.15, 40) if z_grid is None else np.asarray(z_grid, float)
    ns = sorted(curves)
    per_size = {
        n: (
            np.array(sorted(curves[n]), float),
            np.array([curves[n][p] for p in sorted(curves[n])], float),
        )
        for n in ns
    }
    best_spread, best_pc, best_z = np.inf, np.nan, np.nan
    for z in z_grid:
        crossings: list[float] = []
        for i in range(len(ns) - 1):
            (p1, t1), (p2, t2) = per_size[ns[i]], per_size[ns[i + 1]]
            lo, hi = max(p1[0], p2[0]), min(p1[-1], p2[-1])
            if hi <= lo:
                continue
            grid = np.linspace(lo, hi, 300)
            diff = np.interp(grid, p1, t1) / ns[i] ** z - np.interp(grid, p2, t2) / ns[i + 1] ** z
            sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
            if sign_change.size:
                crossings.append(float(grid[sign_change[0]]))
        # max(2, ...) not max(3, ...): with 3 sizes only 2 consecutive pairs
        # exist, so a floor of 3 could never be met and the estimator returned
        # (nan, nan) on perfectly good 3-size crossings.
        if len(crossings) >= max(2, len(ns) - 2):
            spread = float(np.std(crossings))
            if spread < best_spread:
                best_spread, best_pc, best_z = spread, float(np.median(crossings)), float(z)
    return best_pc, best_z


# ----------------------------------------------------------------------
# fixed-p_c (nu, z) fit
# ----------------------------------------------------------------------


def fit_fixed_pc(
    curves: Curves,
    pc: float,
    *,
    warm=None,
    use_log=True,
    nu_bounds=(0.4, 6.5),
    z_bounds=(0.05, 1.2),
) -> tuple[float, float]:
    """Fit ``(nu, z)`` of the collapse (1) at a **pinned** ``p_c``.

    With ``p_c`` held fixed the two exponents are far better conditioned than in
    a free 3-parameter fit (no ``p_c``-``nu``-``z`` correlation), so the fit is
    both faster and tighter. Minimizes :func:`master_curve_loss` by Nelder-Mead
    over a small ``(nu0, z0)`` start grid, or from ``warm=(nu, z)`` when given
    (used by the bootstraps). Returns ``(nu, z)``, or ``(nan, nan)`` when **no**
    start converges inside ``nu_bounds`` x ``z_bounds`` — the data's true
    exponents likely sit outside the window (widen the bounds); an in-band value
    is never fabricated.
    """
    sizes, ps, ys = _arrays(curves)
    pcl = np.full_like(sizes, float(pc))

    def loss(x):
        return hh_loss(sizes, ps, ys, pcl, x[0], x[1], use_log=use_log)

    opts = dict(xatol=1e-3, fatol=1e-8, maxiter=600)
    if warm is not None:
        res = _minimize(loss, warm, method="Nelder-Mead", options=opts)
        return float(res.x[0]), float(res.x[1])
    best_loss, best_x = np.inf, None
    for nu0 in (1.0, 1.5, 2.5, 4.0):
        for z0 in (0.3, 0.5, 0.7):
            res = _minimize(loss, (nu0, z0), method="Nelder-Mead", options=opts)
            if (
                res.fun < best_loss
                and nu_bounds[0] < res.x[0] < nu_bounds[1]
                and z_bounds[0] <= res.x[1] <= z_bounds[1]
            ):
                best_loss, best_x = res.fun, res.x
    if best_x is None:
        # Every start converged outside the bounds: the data's exponents are
        # not in the window. Returning a made-up in-band pair here would flow
        # into graph_bootstrap as a fabricated point with std = 0 — say NaN.
        return float("nan"), float("nan")
    return float(best_x[0]), float(best_x[1])


# ----------------------------------------------------------------------
# full 3-parameter (p_c, nu, z) fit
# ----------------------------------------------------------------------


def fit_three_param(
    curves: Curves,
    *,
    warm=None,
    use_log=True,
    pc_bounds=(0.02, 0.70),
    nu_bounds=(0.05, 6.5),
    z_bounds=(0.02, 1.25),
) -> tuple[float, float, float]:
    """Fit ``(p_c, nu, z)`` jointly by direct minimization of the collapse loss.

    The point estimate screens a grid of starts spanning several ``z0`` values,
    because the ``(p_c, z)`` valley has minima at both low and high ``z``: a
    single ``z0 = 0.5`` start can land in a worse basin and miss the clean global
    minimum. A ``warm=(p_c, nu, z)`` start does a single local refine (used by the
    bootstraps). Bounds are enforced as a soft penalty inside the objective.
    Returns ``(p_c, nu, z)``.
    """
    sizes, ps, ys = _arrays(curves)

    def loss(x):
        pc, nu, z = x
        if not (
            pc_bounds[0] < pc < pc_bounds[1]
            and nu_bounds[0] < nu < nu_bounds[1]
            and z_bounds[0] < z < z_bounds[1]
        ):
            return 1e6
        return hh_loss(sizes, ps, ys, np.full_like(sizes, pc), nu, z, use_log=use_log)

    opts = dict(xatol=1e-3, fatol=1e-9, maxiter=600)
    if warm is not None:
        res = _minimize(loss, warm, method="Nelder-Mead", options=opts)
        return float(res.x[0]), float(res.x[1]), float(res.x[2])
    best_loss, best_x = np.inf, (0.2, 2.0, 0.5)
    pc_lo = np.percentile(ps, 1)
    pc_hi = np.percentile(ps, 55)
    for pc0 in np.linspace(pc_lo, pc_hi, 12):
        for nu0 in (1.0, 2.0, 3.0, 4.5):
            # multiple z starts: the (p_c, z) valley has minima at both low and
            # high z; a single z0=0.5 start can miss the high-z (z~1) global min.
            for z0 in (0.4, 0.7, 1.0, 1.15):
                res = _minimize(loss, (pc0, nu0, z0), method="Nelder-Mead", options=opts)
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_x = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
    return best_x


# ----------------------------------------------------------------------
# smooth-p_c collapse across control values
# ----------------------------------------------------------------------


def smooth_pc_collapse(
    cells: Mapping[Any, Curves],
    *,
    window=11,
    polyorder=2,
    use_log=True,
) -> dict[Any, tuple[float, float, float]]:
    """Smooth-``p_c`` collapse across a family of external control values.

    ``cells`` maps each control value (e.g. a Watts-Strogatz ``beta``) to its
    :data:`Curves`. The procedure:

    1. fit ``(p_c, nu, z)`` per control value with :func:`fit_three_param`;
    2. Savitzky-Golay smooth the ``p_c(control)`` sequence (in control order),
       which removes the control-to-control ``p_c`` jitter;
    3. refit ``(nu, z)`` at the smoothed ``p_c`` with :func:`fit_fixed_pc`.

    Pinning ``p_c`` to a smooth trend cuts the ``z`` scatter that otherwise leaks
    in through the ``(p_c, z)`` correlation. Returns
    ``{control: (pc_smooth, nu, z)}`` in the input control order. For any control
    value where :func:`fit_fixed_pc` finds no converged in-bounds solution its
    ``nu`` and ``z`` come back ``nan`` (the refit never fabricates an in-band
    pair); ``pc_smooth`` is still the smoothed value.

    ``window`` (odd, ``>= polyorder + 2``) is clamped to the number of control
    values; if fewer than ``polyorder + 2`` values are present the raw ``p_c`` is
    used unchanged (no smoothing is possible).

    .. note::

       The smoothing runs over the controls **in the mapping's iteration
       order** and treats them as **uniformly spaced** (Savitzky-Golay operates
       on the index axis). Pass ``cells`` sorted in the physical control order;
       for a strongly non-uniform grid (e.g. log-spaced ``beta``) the smooth is
       a smooth in *index* space, which is usually what the jitter-removal
       intends — but it is not a smooth in the control variable itself.
    """
    controls = list(cells)
    if not controls:
        return {}
    pc_raw = np.array([fit_three_param(cells[c], use_log=use_log)[0] for c in controls], float)

    n = pc_raw.size
    win = int(window)
    if win % 2 == 0:
        win += 1
    win = min(win, n if n % 2 == 1 else n - 1)
    if win >= polyorder + 2 and win >= 3:
        try:
            from scipy.signal import savgol_filter
        except ImportError as e:  # pragma: no cover
            raise InvalidArgumentError(
                "smooth_pc_collapse needs the optional 'analysis' extra (scipy); "
                "install with: pip install sparsegf2[analysis]"
            ) from e
        pc_smooth = savgol_filter(pc_raw, win, polyorder)
    else:
        pc_smooth = pc_raw  # too few control values to smooth

    out: dict[Any, tuple[float, float, float]] = {}
    for c, pc in zip(controls, pc_smooth, strict=True):
        nu, z = fit_fixed_pc(cells[c], float(pc), use_log=use_log)
        out[c] = (float(pc), float(nu), float(z))
    return out


# ----------------------------------------------------------------------
# graph-level bootstrap
# ----------------------------------------------------------------------


def _median_curve(raw_cell: RawCell) -> dict[int, dict[float, float]]:
    """Per-cell median curve ``{L: {p: median over graphs}}`` from a raw cell."""
    return {
        n: {float(p): float(np.median(np.asarray(vals, float))) for p, vals in curve.items()}
        for n, curve in raw_cell.items()
    }


def graph_bootstrap(
    raw_cell: RawCell,
    estimator: Callable[[Curves], tuple],
    *,
    n_boot=100,
    seed=0,
) -> dict[str, Any]:
    """Disorder-aware bootstrap of an FSS estimator over the per-graph values.

    ``raw_cell`` is ``{L: {p: [per-graph values]}}`` -- the raw observable for
    every graph realization, before medianing. ``estimator`` is one of the fit
    functions (:func:`fit_fixed_pc` bound with its ``pc``, :func:`fit_three_param`,
    or any ``curves -> tuple`` callable) applied to a median curve.

    The point estimate is ``estimator`` on the full median curve. Each of the
    ``n_boot`` resamples draws, per size, an index set of graphs *with
    replacement* (the **same** set across all ``p`` of that size, so each
    resampled graph contributes its whole ``O(p)`` curve -- the rigorous,
    disorder-aware resampling), recomputes the median curve, and refits. The
    per-parameter std over resamples is the uncertainty.

    Returns ``{"point": tuple, "std": tuple, "samples": np.ndarray (n_ok, n_par),
    "n_boot_ok": int}``.
    """
    median = _median_curve(raw_cell)
    point = tuple(float(v) for v in estimator(median))

    sizes = sorted(raw_cell)
    # p-set and graph count per size, taken from the point-estimate curve so every
    # resample refits on the same (L, p) grid.
    p_by_size = {n: list(median[n]) for n in sizes}
    # One shared graph-index set is drawn per size and applied across every p of
    # that size, so all p at a size must carry the same number of per-graph values
    # (each realization contributes its whole O(p) curve). Ragged counts mean the
    # per-graph correspondence across p is already broken (e.g. censoring dropped
    # different graphs at different p), which makes shared-index resampling
    # ill-defined -- reject it cleanly rather than crashing with an IndexError.
    n_graphs = {}
    for n in sizes:
        counts = {len(np.asarray(raw_cell[n][p], float)) for p in p_by_size[n]}
        if len(counts) != 1:
            raise InvalidArgumentError(
                f"graph_bootstrap needs the same number of per-graph values at every "
                f"p of size {n}; got counts {sorted(counts)}. Each graph realization "
                f"must contribute its whole O(p) curve (drop or align censored graphs)."
            )
        n_graphs[n] = counts.pop()

    rng = np.random.default_rng(seed)
    samples: list[tuple[float, ...]] = []
    for _ in range(int(n_boot)):
        resampled: dict[int, dict[float, float]] = {}
        for n in sizes:
            idx = rng.integers(0, n_graphs[n], size=n_graphs[n])
            resampled[n] = {
                float(p): float(np.median(np.asarray(raw_cell[n][p], float)[idx]))
                for p in p_by_size[n]
            }
        try:
            est = tuple(float(v) for v in estimator(resampled))
        except (ValueError, RuntimeError):  # pragma: no cover (defensive)
            continue
        if all(np.isfinite(est)):
            samples.append(est)

    arr = np.array(samples, float) if samples else np.empty((0, len(point)), float)
    std = tuple(float(s) for s in (arr.std(axis=0) if arr.size else np.full(len(point), np.nan)))
    return {"point": point, "std": std, "samples": arr, "n_boot_ok": int(arr.shape[0])}


__all__ = [
    "hh_loss",
    "master_curve_loss",
    "robust_pc",
    "fit_fixed_pc",
    "fit_three_param",
    "smooth_pc_collapse",
    "graph_bootstrap",
]
