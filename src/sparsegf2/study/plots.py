"""Diagnostic figures of the Watts–Strogatz study (design §1 ``plots.py``; requirements R349, R350).

Every function draws one figure from a study store (through the same aggregation as
:func:`sparsegf2.study.api.aggregate`) or from a finite-size-scaling report, and writes four
files into a directory: the figure as ``<stem>.svg`` (vector) and ``<stem>.png`` (raster), and
the exact numbers it shows as ``<stem>.csv`` (one row per plotted point) and ``<stem>.json``
(the same numbers with the title, the caption and the settings). It returns those paths and the
caption. Every title starts with "Diagnostic:" and every caption says what is estimated, how
the band is computed, and that the figure alone does not establish a transition.

* :func:`k_trace`: mean ``k_t`` (or ``k_t / n``) at every layer, bands of one graph-cluster
  standard error;
* :func:`survival`: graph-weighted Kaplan-Meier survival of ``tau_ell``, censoring marked;
* :func:`d_cyc`: mean ``d_cyc`` versus depth with its zero extension (``d_cyc = 0`` where
  ``k_t = 0``, v1.4 §5.5);
* :func:`peak_distribution`: the graph-weighted distribution of ``T*`` with the fractions of
  trajectories without a peak and with the peak at the horizon;
* :func:`peak_vs_tau1`: ``T*`` against ``tau_1`` on the same trajectories;
* :func:`versus_p`: an observable versus ``p``, one line per size;
* :func:`graph_measures`: graph measurements versus ``beta``, one line per size;
* :func:`collapse`: the scaled points of an :func:`sparsegf2.study.api.fss` report.

matplotlib is imported when a figure is drawn (never by ``import sparsegf2``); the figures are
built without pyplot, so they draw headless and leave no global state. The colors are a
fixed-order categorical palette (blue, orange, aqua, yellow, magenta, green, violet, red; at
most eight series per figure), with gray chrome.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2.study import api as _sapi
from sparsegf2.study import stats as _stats
from sparsegf2.study.identity import Token

__all__ = [
    "PLOTS",
    "collapse",
    "d_cyc",
    "graph_measures",
    "k_trace",
    "peak_distribution",
    "peak_vs_tau1",
    "survival",
    "versus_p",
]

#: The categorical series colors, in their fixed order (never cycled).
SERIES_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "axes.edgecolor": _AXIS,
    "axes.labelcolor": _INK_2,
    "axes.facecolor": _SURFACE,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": _GRID,
    "grid.linewidth": 0.6,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
    "xtick.labelcolor": _INK_2,
    "ytick.labelcolor": _INK_2,
    "legend.frameon": False,
    "legend.fontsize": 8.0,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "figure.facecolor": _SURFACE,
    "savefig.facecolor": _SURFACE,
    "svg.hashsalt": "sparsegf2-study",
}
_BAND_ALPHA = 0.18
_MAX_SERIES = len(SERIES_COLORS)
_DIAGNOSTIC = (
    "Diagnostic figure: it summarizes the stored data and does not by itself establish a "
    "transition or a scaling law."
)


# ---------------------------------------------------------------------------------------------
# drawing and saving
# ---------------------------------------------------------------------------------------------


def _matplotlib() -> Any:
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - matplotlib is a test dependency
        raise ValueError(
            "the study plots need matplotlib: pip install matplotlib (it is in the test extra)"
        ) from exc
    return matplotlib


def _context() -> AbstractContextManager[Any]:
    rc: AbstractContextManager[Any] = _matplotlib().rc_context(_RC)
    return rc


def _figure(rows: int = 1, cols: int = 1, size: tuple[float, float] = (6.4, 4.0)) -> Any:
    from matplotlib.figure import Figure

    fig = Figure(figsize=size, layout="constrained")
    axes = fig.subplots(rows, cols, squeeze=False)
    for ax in axes.ravel():
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
    return fig, axes


def _colors(count: int) -> list[str]:
    if count > _MAX_SERIES:
        raise ValueError(
            f"this figure would draw {count} series; at most {_MAX_SERIES} are distinguishable: "
            "narrow where (fewer sizes, beta values or rates)"
        )
    return list(SERIES_COLORS[:count])


def _title(text: str) -> str:
    return "Diagnostic: " + text


def _save(
    fig: Any,
    directory: Path,
    stem: str,
    *,
    kind: str,
    title: str,
    caption: str,
    header: Sequence[str],
    rows: Sequence[Sequence[Any]],
    data: Mapping[str, Any],
    return_figure: bool,
) -> dict[str, Any]:
    """Write the figure (``.svg``, ``.png``) and its numbers (``.csv``, ``.json``)."""
    if not title.startswith("Diagnostic:"):  # R350: every figure is labeled a diagnostic
        raise ValueError(f"a study figure's title must start with 'Diagnostic:', got {title!r}")
    directory = Path(directory)
    svg, png = directory / f"{stem}.svg", directory / f"{stem}.png"
    fig.savefig(svg, format="svg", metadata={"Date": None})
    fig.savefig(png, format="png", dpi=150)
    csv_path = directory / f"{stem}.csv"
    _sapi._write_csv(csv_path, header, rows)
    json_path = directory / f"{stem}.json"
    _sapi._write_json(
        json_path,
        {
            "kind": kind,
            "title": title,
            "caption": caption,
            "columns": list(header),
            "rows": [list(r) for r in rows],
            **data,
        },  # fmt: skip
    )
    out: dict[str, Any] = {
        "kind": kind,
        "title": title,
        "caption": caption,
        "figures": {"svg": str(svg), "png": str(png)},
        "data": {"csv": str(csv_path), "json": str(json_path)},
        "points": len(rows),
    }
    if return_figure:
        out["figure_object"] = fig
    return out


def _label(cond: Mapping[str, Any], vary: Sequence[str]) -> str:
    names = {"n": "n", "beta": "β", "p": "p", "protocol": "protocol"}
    parts = [f"{names[k]} = {cond[k]}" for k in vary]
    return ", ".join(parts) if parts else "all"


def _varying(conds: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[str]:
    return [k for k in keys if len({str(c[k]) for c in conds}) > 1]


def _shared(conds: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str:
    fixed = [k for k in keys if len({str(c[k]) for c in conds}) == 1]
    names = {"n": "n", "beta": "beta", "p": "p", "protocol": "protocol"}
    return ", ".join(f"{names[k]} = {conds[0][k]}" for k in fixed)


def _band(ax: Any, x: np.ndarray, mean: np.ndarray, se: np.ndarray, color: str, **kw: Any) -> None:
    ok = np.isfinite(se)
    if ok.any():
        ax.fill_between(x, np.where(ok, mean - se, np.nan), np.where(ok, mean + se, np.nan),
                        color=color, alpha=_BAND_ALPHA, linewidth=0, **kw)  # fmt: skip


def _with_store(store: Any, fn: Callable[[Any], dict[str, Any]]) -> dict[str, Any]:
    with _sapi._opened(store) as s, _sapi._lookups():
        return fn(s)


# ---------------------------------------------------------------------------------------------
# the figures
# ---------------------------------------------------------------------------------------------


def _plain(axis: Any, log: bool = True) -> None:
    """Plain numbers on a log axis (``2``, ``10``, ``50``) at 1, 2 and 5 per decade, no minor
    labels: the default ``3 x 10^0`` labels collide on short decades."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    if log:
        axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def k_trace(
    store: Any,
    directory: Any,
    *,
    where: Any = None,
    normalized: bool = True,
    stem: str = "k_trace",
    return_figure: bool = False,
) -> dict[str, Any]:
    """Mean ``k_t`` (``normalized``: ``k_t / n``) at every layer ``t`` for each selected
    condition, with a band of one graph-cluster standard error (v1.4 §5.2, §6.3, §6.4)."""

    def draw(s: Any) -> dict[str, Any]:
        res = _sapi._aggregate(s, "k_trace", where=where, normalized=normalized)
        conds = res["conditions"]
        if not conds:
            raise ValueError("no stored k traces in this selection")
        colors = _colors(len(conds))
        vary = _varying(conds, ("protocol", "n", "beta", "p"))
        what = "k_t / n" if normalized else "k_t"
        rows: list[list[Any]] = []
        with _context():
            fig, axes = _figure()
            ax = axes[0, 0]
            for cond, color in zip(conds, colors, strict=True):
                x, y, se = cond["depths"], cond["mean"], cond["se"]
                ax.plot(x, y, color=color, label=_label(cond, vary))
                _band(ax, x, y, se, color)
                rows.extend([cond["protocol"], cond["n"], cond["beta"], cond["p"], int(t), m, e]
                            for t, m, e in zip(x.tolist(), y.tolist(), se.tolist(), strict=True))  # fmt: skip
            ax.set_xlabel("depth t (completed layers)")
            ax.set_ylabel(f"mean {what}" + (" (logical qubits per system qubit)" if normalized
                                            else " (logical qubits)"))  # fmt: skip
            title = _title(f"mean {what} versus depth" + (
                f" ({_shared(conds, ('protocol', 'n', 'beta', 'p'))})" if len(vary) < 4 else ""))  # fmt: skip
            ax.set_title(title, loc="left", color=_INK)
            if len(conds) > 1:
                ax.legend()
            graphs = sorted({int(c["graphs"]) for c in conds})
            caption = (
                f"Mean {what} at every layer t: the nested equal-graph-weight mean over "
                f"{graphs} graphs per condition (v1.4 §6.3); bands are plus or minus one "
                "graph-cluster standard error (§6.4). k_t is the number of logical qubits of the "
                f"code the dynamics left (v1.4 §5.2). {_DIAGNOSTIC}"
            )
            return _save(fig, Path(directory), stem, kind="k", title=title, caption=caption,
                         header=["protocol", "n", "beta", "p", "t", "mean", "se"], rows=rows,
                         data={"normalized": normalized, "where": where},
                         return_figure=return_figure)  # fmt: skip

    return _with_store(store, draw)


def survival(
    store: Any,
    directory: Any,
    *,
    where: Any = None,
    ell: int = 0,
    stem: str | None = None,
    return_figure: bool = False,
) -> dict[str, Any]:
    """The graph-weighted Kaplan-Meier survival ``S(t) = P(tau_ell > t)`` of each selected
    condition (``stats.nested_kaplan_meier``), with a band of one standard error and the
    censored trajectories marked at their censoring depth (v1.4 §5.2: censored, never infinite)."""
    ell = _sapi._int(ell, "ell", 0)

    def draw(s: Any) -> dict[str, Any]:
        res = _sapi._aggregate(s, "first_passage", where=where, ell=[ell])
        conds = [c for c in res["conditions"] if c["ells"]]
        if not conds:
            raise ValueError(f"no stored first passages with ell = {ell} in this selection")
        colors = _colors(len(conds))
        vary = _varying(conds, ("protocol", "n", "beta", "p"))
        rows: list[list[Any]] = []
        censoring: list[dict[str, Any]] = []
        with _context():
            fig, axes = _figure()
            ax = axes[0, 0]
            for cond, color in zip(conds, colors, strict=True):
                entry = cond["ells"][0]
                curve: _stats.SurvivalCurve = entry["_curve"]
                times = np.concatenate([[0.0], curve.times])
                surv = np.concatenate([[1.0], curve.survival])
                se = np.concatenate([[0.0], curve.se])
                end = float(entry["horizon"])  # T_max: S is known up to it
                x = np.concatenate([times, [end]]) if end > times[-1] else times
                y = np.concatenate([surv, [surv[-1]]]) if end > times[-1] else surv
                e = np.concatenate([se, [se[-1]]]) if end > times[-1] else se
                label = _label(cond, vary)
                ax.step(x, y, where="post", color=color, label=label)
                ax.fill_between(x, np.clip(y - e, 0, 1), np.clip(y + e, 0, 1), step="post",
                                color=color, alpha=_BAND_ALPHA, linewidth=0)  # fmt: skip
                observed = entry["_observed"]
                if (~observed).any():
                    depth = float(np.max(entry["_times"][~observed]))
                    at = float(curve.survival_at(depth))
                    ax.plot([depth], [at], marker="|", markersize=12, markeredgewidth=1.6,
                            color=color, linestyle="none")  # fmt: skip
                censoring.append(
                    {
                        **{k: cond[k] for k in ("protocol", "n", "beta", "p")},
                        "censored": entry["censored"],
                        "trajectories": entry["trajectories"],
                        "censored_fraction": entry["censored_fraction"],
                        "censored_fraction_se": entry["censored_fraction_se"],
                        "horizon": entry["horizon"],
                        "median": entry["median"],
                        "restricted_mean": entry["restricted_mean"],
                    }  # fmt: skip
                )
                rows.extend([cond["protocol"], cond["n"], cond["beta"], cond["p"], ell, t, v, u]
                            for t, v, u in zip(times.tolist(), surv.tolist(), se.tolist(),
                                               strict=True))  # fmt: skip
            ax.set_xlabel("depth t (completed layers)")
            ax.set_ylabel(f"S(t) = P(tau_{ell} > t)")
            ax.set_ylim(-0.02, 1.02)
            shared = _shared(conds, ("protocol", "n", "beta", "p"))
            title = _title(f"survival of tau_{ell}" + (f" ({shared})" if shared else ""))
            ax.set_title(title, loc="left", color=_INK)
            if len(conds) > 1:
                ax.legend()
            share = ", ".join(
                f"{_label(c, vary) or 'this condition'}: {c['censored']} of {c['trajectories']}"
                for c in censoring
            )
            caption = (
                f"Graph-weighted Kaplan-Meier estimate of P(tau_{ell} > t), tau_{ell} = min{{t : "
                f"k_t <= {ell}}} observed at every layer (v1.4 §5.2); every graph has the same "
                "weight, the band is plus or minus one graph-cluster delta-method standard error. "
                "A vertical tick marks the horizon T_max at which trajectories that had not "
                f"reached k_t <= {ell} are right-censored (censored: {share}); the curve is not "
                f"extended beyond it. {_DIAGNOSTIC}"
            )
            return _save(fig, Path(directory), stem or f"survival_tau{ell}", kind="survival",
                         title=title, caption=caption,
                         header=["protocol", "n", "beta", "p", "ell", "t", "survival", "se"],
                         rows=rows, data={"ell": ell, "censoring": censoring, "where": where},
                         return_figure=return_figure)  # fmt: skip

    return _with_store(store, draw)


def d_cyc(
    store: Any,
    directory: Any,
    *,
    where: Any = None,
    stem: str = "d_cyc",
    return_figure: bool = False,
) -> dict[str, Any]:
    """Mean ``d_cyc`` versus depth with its zero extension (v1.4 §5.5): a trajectory without a
    code (``k_t = 0``) counts as 0, so the mean is unconditional; bands of one graph-cluster
    standard error. The fraction of trajectories holding a code is in the data file."""

    def draw(s: Any) -> dict[str, Any]:
        res = _sapi._aggregate(s, "d_cyc", where=where, field="zero_extended")
        codes = _sapi._aggregate(s, "d_cyc", where=where, field="has_code")
        conds = res["conditions"]
        if not conds:
            raise ValueError("no stored d_cyc values in this selection")
        has = {(c["protocol"], c["n"], c["beta"], c["p"]): c for c in codes["conditions"]}
        colors = _colors(len(conds))
        vary = _varying(conds, ("protocol", "n", "beta", "p"))
        rows: list[list[Any]] = []
        with _context():
            fig, axes = _figure()
            ax = axes[0, 0]
            for cond, color in zip(conds, colors, strict=True):
                x, y, se = cond["depths"], cond["mean"], cond["se"]
                frac = has[(cond["protocol"], cond["n"], cond["beta"], cond["p"])]["mean"]
                ax.plot(x, y, color=color, marker="o", label=_label(cond, vary))
                _band(ax, x, y, se, color)
                rows.extend([cond["protocol"], cond["n"], cond["beta"], cond["p"], int(t), m, e, f]
                            for t, m, e, f in zip(x.tolist(), y.tolist(), se.tolist(),
                                                  frac.tolist(), strict=True))  # fmt: skip
            ax.set_xscale("symlog", linthresh=1.0)
            _plain(ax.xaxis, log=False)
            ax.set_xlabel("depth t (completed layers, symmetric log scale)")
            ax.set_ylabel("mean d_cyc, 0 where k_t = 0 (qubits)")
            shared = _shared(conds, ("protocol", "n", "beta", "p"))
            title = _title("zero-extended d_cyc versus depth" + (f" ({shared})" if shared else ""))
            ax.set_title(title, loc="left", color=_INK)
            if len(conds) > 1:
                ax.legend()
            caption = (
                "Mean original-ring contiguous distance d_cyc (the smallest cyclic interval whose "
                "erasure loses logical information, v1.4 §5.5) at the stored depths, with the "
                "zero extension stated there: d_cyc = 0 where k_t = 0, so the mean is over all "
                "trajectories (the raw d_cyc is undefined there; the fraction holding a code is "
                "the column has_code_fraction). Nested equal-graph-weight means; bands plus or "
                f"minus one graph-cluster standard error. {_DIAGNOSTIC}"
            )
            return _save(fig, Path(directory), stem, kind="d_cyc", title=title, caption=caption,
                         header=["protocol", "n", "beta", "p", "t", "mean", "se",
                                 "has_code_fraction"], rows=rows,
                         data={"zero_extension": "d_cyc := 0 where k_t = 0 (v1.4 §5.5)",
                               "where": where}, return_figure=return_figure)  # fmt: skip

    return _with_store(store, draw)


def _peak_rows(
    s: Any, observable: str, params: Any, where: Any, horizon_policy: str = "include"
) -> dict[str, Any]:
    return _sapi._aggregate(s, observable, params, where, horizon_policy=horizon_policy)


def peak_distribution(
    store: Any,
    directory: Any,
    *,
    observable: str = "peak_d_cyc",
    params: Any = None,
    where: Any = None,
    stem: str | None = None,
    return_figure: bool = False,
) -> dict[str, Any]:
    """The distribution of the peak time ``T*`` (v1.4 §5.8): for each condition the
    graph-weighted cumulative share of trajectories with ``T* <= t`` (every graph the same total
    weight, ``stats.graph_equal_weights``). The curve ends at one minus the share without a
    peak; trajectories whose peak is at the horizon (``T*`` a lower bound) are drawn as hollow
    markers, and both fractions are in the legend."""

    def draw(s: Any) -> dict[str, Any]:
        res = _peak_rows(s, observable, params, where)
        conds = res["conditions"]
        if not conds:
            raise ValueError(f"no stored {observable} values in this selection")
        colors = _colors(len(conds))
        vary = _varying(conds, ("protocol", "n", "beta", "p"))
        rows: list[list[Any]] = []
        summary: list[dict[str, Any]] = []
        with _context():
            fig, axes = _figure()
            ax = axes[0, 0]
            for cond, color in zip(conds, colors, strict=True):
                g, c, r = cond["_g"], cond["_c"], cond["_r"]
                weights = _stats.graph_equal_weights(g, c, r).astype(np.float64)
                weights /= weights.sum()
                tstar, horizon = cond["_t_star"], cond["_at_horizon"]
                defined = np.isfinite(tstar)
                order = np.argsort(np.where(defined, tstar, np.inf), kind="stable")
                order = order[defined[order]]
                cum = np.cumsum(weights[order])
                xs = tstar[order]
                label = (f"{_label(cond, vary)}: missing {cond['missing_fraction']:.2f}, at "
                         f"horizon {cond['horizon_fraction']:.2f}")  # fmt: skip
                if xs.size:
                    ax.step(np.concatenate([[xs[0]], xs]), np.concatenate([[0.0], cum]),
                            where="post", color=color, label=label)  # fmt: skip
                    hz = horizon[order]
                    ax.plot(xs[hz], cum[hz], marker="o", markerfacecolor="none",
                            markeredgecolor=color, linestyle="none")  # fmt: skip
                else:
                    ax.plot([], [], color=color, label=label)
                summary.append(
                    {
                        **{k: cond[k] for k in ("protocol", "n", "beta", "p")},
                        "trajectories": cond["trajectories"],
                        "missing_fraction": cond["missing_fraction"],
                        "horizon_fraction": cond["horizon_fraction"],
                    }  # fmt: skip
                )
                rows.extend(
                    [cond["protocol"], cond["n"], cond["beta"], cond["p"], int(g[i]), int(c[i]),
                     int(r[i]), float(tstar[i]), bool(horizon[i]), float(weights[i]), float(q)]
                    for i, q in zip(order.tolist(), cum.tolist(), strict=True)
                )  # fmt: skip
            ax.set_xscale("log")
            _plain(ax.xaxis)
            ax.set_xlabel("T* (layers, log scale)")
            ax.set_ylabel("graph-weighted share with T* <= t")
            ax.set_ylim(-0.02, 1.02)
            shared = _shared(conds, ("protocol", "n", "beta", "p"))
            title = _title(
                f"distribution of T* from {observable}" + (f" ({shared})" if shared else "")
            )
            ax.set_title(title, loc="left", color=_INK)
            ax.legend()
            caption = (
                f"Cumulative distribution of the peak time T* of {observable} (T* = exp of the "
                "mean of ln t over the maximizing checkpoint set, v1.4 §5.8), each graph weighted "
                "equally. The curve ends below 1 by the share of trajectories without a peak "
                "(missing: d* = 0, or an invalid trace); hollow markers are trajectories whose "
                "maximizing set contains T_max (horizon-limited: T* is a lower bound). "
                f"{_DIAGNOSTIC}"
            )
            return _save(fig, Path(directory), stem or f"distribution_{observable}",
                         kind="peak_distribution", title=title, caption=caption,
                         header=["protocol", "n", "beta", "p", "g", "c", "r", "t_star",
                                 "at_horizon", "weight", "cumulative"], rows=rows,
                         data={"observable": observable, "params": res.get("params"),
                               "summary": summary, "where": where},
                         return_figure=return_figure)  # fmt: skip

    return _with_store(store, draw)


def peak_vs_tau1(
    store: Any,
    directory: Any,
    *,
    observable: str = "peak_d_cyc",
    params: Any = None,
    where: Any = None,
    stem: str | None = None,
    return_figure: bool = False,
) -> dict[str, Any]:
    """``T*`` against ``tau_1`` on the same trajectories (v1.4 §5.9: compared only where
    ``tau_1`` is observed and positive and ``T*`` is defined); the diagonal ``T* = tau_1``;
    peaks at the horizon hollow. The excluded trajectories are counted in the caption."""

    def draw(s: Any) -> dict[str, Any]:
        res = _peak_rows(s, observable, params, where)
        fp = _sapi._aggregate(s, "first_passage", where=_strip_t(where), ell=[1])
        taus = {(c["protocol"], c["n"], c["beta"], c["p"]): c for c in fp["conditions"]}
        conds = [c for c in res["conditions"] if (c["protocol"], c["n"], c["beta"], c["p"]) in taus]
        if not conds:
            raise ValueError(f"no trajectories with both {observable} and first_passage stored")
        colors = _colors(len(conds))
        vary = _varying(conds, ("protocol", "n", "beta", "p"))
        rows: list[list[Any]] = []
        counts: list[dict[str, Any]] = []
        with _context():
            fig, axes = _figure(size=(5.2, 4.6))
            ax = axes[0, 0]
            lo, hi = math.inf, 0.0
            for cond, color in zip(conds, colors, strict=True):
                fp_cond = taus[(cond["protocol"], cond["n"], cond["beta"], cond["p"])]
                fp_rows = fp_cond["_rows"]
                entry = fp_cond["ells"][0]
                tau1 = {
                    (int(a), int(b), int(q)): (float(t), bool(o))
                    for a, b, q, t, o in zip(fp_rows["g"].tolist(), fp_rows["c"].tolist(),
                                             fp_rows["r"].tolist(), entry["_times"].tolist(),
                                             entry["_observed"].tolist(), strict=True)
                }  # fmt: skip
                xs, ys, hz = [], [], []
                censored = missing = zero = 0
                for i in range(cond["_g"].size):
                    key = (int(cond["_g"][i]), int(cond["_c"][i]), int(cond["_r"][i]))
                    t1, seen = tau1[key]
                    tstar = float(cond["_t_star"][i])
                    if not seen:
                        censored += 1
                        continue
                    if not math.isfinite(tstar):
                        missing += 1
                        continue
                    if t1 <= 0:
                        zero += 1
                        continue
                    xs.append(t1)
                    ys.append(tstar)
                    hz.append(bool(cond["_at_horizon"][i]))
                    rows.append([cond["protocol"], cond["n"], cond["beta"], cond["p"], *key, t1,
                                 tstar, bool(cond["_at_horizon"][i])])  # fmt: skip
                x, y, h = np.array(xs), np.array(ys), np.array(hz, dtype=bool)
                label = _label(cond, vary) or "trajectories"
                ax.plot(x[~h], y[~h], marker="o", linestyle="none", color=color, label=label,
                        alpha=0.85)  # fmt: skip
                if h.any():
                    ax.plot(x[h], y[h], marker="o", linestyle="none", markerfacecolor="none",
                            markeredgecolor=color)  # fmt: skip
                if x.size:
                    lo = min(lo, float(x.min()), float(y.min()))
                    hi = max(hi, float(x.max()), float(y.max()))
                counts.append({**{k: cond[k] for k in ("protocol", "n", "beta", "p")},
                               "pairs": int(x.size), "tau1_censored": censored,
                               "tstar_missing": missing, "tau1_zero": zero,
                               "tstar_above_tau1": int((y > x).sum()),
                               "horizon_peaks": int(h.sum())})  # fmt: skip
            if hi > 0:
                span = np.array([lo / 1.5, hi * 1.5])
                ax.plot(span, span, color=_MUTED, linestyle="--", linewidth=1.0, label="T* = tau_1")
            ax.set_xscale("log")
            ax.set_yscale("log")
            _plain(ax.xaxis)
            _plain(ax.yaxis)
            ax.set_xlabel("tau_1 (layers: first k_t <= 1)")
            ax.set_ylabel("T* (layers)")
            shared = _shared(conds, ("protocol", "n", "beta", "p"))
            title = _title("T* versus tau_1" + (f" ({shared})" if shared else ""))
            ax.set_title(title, loc="left", color=_INK)
            ax.legend()
            excluded = ", ".join(
                f"{c['tau1_censored']} tau_1 censored, {c['tstar_missing']} without a peak, "
                f"{c['tau1_zero']} with tau_1 = 0"
                for c in counts
            )
            caption = (
                f"Peak time T* of {observable} against the exact first passage tau_1 = min{{t : "
                "k_t <= 1}} on the same trajectories (v1.4 §5.8, §5.9), only where tau_1 is "
                "observed and positive and T* is defined; the dashed line is T* = tau_1. Hollow "
                f"markers: the peak is at the horizon (T* a lower bound). Excluded: {excluded}. "
                f"{_DIAGNOSTIC}"
            )
            return _save(fig, Path(directory), stem or f"{observable}_vs_tau1",
                         kind="peak_vs_tau1", title=title, caption=caption,
                         header=["protocol", "n", "beta", "p", "g", "c", "r", "tau_1", "t_star",
                                 "at_horizon"], rows=rows,
                         data={"observable": observable, "params": res.get("params"),
                               "counts": counts, "where": where},
                         return_figure=return_figure)  # fmt: skip

    return _with_store(store, draw)


def _strip_t(where: Any) -> Any:
    if isinstance(where, Mapping) and "t" in where:
        return {k: v for k, v in where.items() if k != "t"}
    return where


_STATISTICS = {
    "first_passage": ("median", "restricted_mean", "censored_fraction"),
    "peak": ("arithmetic", "geometric", "argmax", "missing_fraction", "horizon_fraction"),
}


def versus_p(
    store: Any,
    directory: Any,
    *,
    observable: str = "k",
    where: Any = None,
    depth: Any = "final",
    statistic: str | None = None,
    ell: int = 1,
    params: Any = None,
    field: str | None = None,
    index: int | None = None,
    nan_policy: str = "raise",
    horizon_policy: str = "exclude",
    stem: str | None = None,
    return_figure: bool = False,
) -> dict[str, Any]:
    """An observable versus the measurement rate ``p``, one line per ``(n, beta)``.

    Snapshot observables at ``depth`` (an integer, or ``"final"``: each condition's last
    stored depth), as nested means with one graph-cluster standard error; ``first_passage``
    (threshold ``ell``) as the Kaplan-Meier ``median`` (no band), the ``restricted_mean``
    ``E[min(tau, T_max)]`` or the ``censored_fraction`` (with bands); ``peak_*`` as the
    ``arithmetic`` or ``geometric`` mean of ``T*`` (bands), ``argmax`` (no band) or the
    ``missing_fraction`` / ``horizon_fraction``. ``params``, ``field``, ``index``,
    ``nan_policy`` and ``horizon_policy`` are :func:`sparsegf2.study.api.aggregate`'s."""

    def draw(s: Any) -> dict[str, Any]:
        from sparsegf2.study import observables as ob

        with _sapi._lookups():
            entry = ob.get(_sapi._text(observable, "observable"))
        kind = _sapi._aggregate_kind(entry)
        base = _strip_t(where)
        if kind == "snapshot":
            res = _sapi._aggregate(s, observable, params, base, field=field, index=index,
                                   nan_policy=nan_policy)  # fmt: skip
            stat = "mean"
        elif kind == "first_passage":
            res = _sapi._aggregate(s, observable, None, base, ell=[_sapi._int(ell, "ell", 0)])
            stat = _sapi._choice(statistic or "restricted_mean", "statistic",
                                 _STATISTICS["first_passage"])  # fmt: skip
        elif kind == "peak":
            res = _sapi._aggregate(s, observable, params, base, horizon_policy=horizon_policy)
            stat = _sapi._choice(statistic or "arithmetic", "statistic", _STATISTICS["peak"])
        else:
            raise ValueError("versus_p plots snapshot observables, first_passage or peak_*")
        if kind != "snapshot" and depth != "final":
            raise ValueError("depth applies to snapshot observables only")
        points: dict[tuple[str, int, str], list[tuple[float, str, float, float]]] = {}
        for cond in res["conditions"]:
            value, err = _point(cond, kind, stat, depth)
            points.setdefault((cond["protocol"], cond["n"], cond["beta"]), []).append(
                (Token.parse(cond["p"]).float64, cond["p"], value, err)
            )
        if not points:
            raise ValueError(f"no stored {observable} values in this selection")
        series = sorted(points)
        colors = _colors(len(series))
        groups = [{"protocol": a, "n": b, "beta": c} for a, b, c in series]
        vary = _varying(groups, ("protocol", "n", "beta"))
        rows: list[list[Any]] = []
        label_of = {"mean": f"mean {observable}" + (f".{field}" if field else ""),
                    "median": f"median tau_{ell}", "restricted_mean": f"E[min(tau_{ell}, T_max)]",
                    "censored_fraction": f"censored fraction of tau_{ell}",
                    "arithmetic": "E[T*]", "geometric": "exp(E[ln T*])",
                    "argmax": "argmax_t E[d(t)]", "missing_fraction": "fraction without a peak",
                    "horizon_fraction": "fraction with the peak at T_max"}  # fmt: skip
        with _context():
            fig, axes = _figure()
            ax = axes[0, 0]
            for key, color, group in zip(series, colors, groups, strict=True):
                pts = sorted(points[key])
                x = np.array([q[0] for q in pts])
                y = np.array([q[2] for q in pts], dtype=np.float64)
                e = np.array([q[3] for q in pts], dtype=np.float64)
                ax.plot(x, y, marker="o", color=color, label=_label(group, vary) or None)
                _band(ax, x, y, e, color)
                rows.extend([key[0], key[1], key[2], q[1], q[0], q[2], q[3]] for q in pts)
            ax.set_xlabel("measurement rate p")
            ax.set_ylabel(label_of[stat])
            shared = _shared(groups, ("protocol", "n", "beta"))
            where_depth = "" if kind != "snapshot" else f" at depth {depth}"
            title = _title(
                f"{label_of[stat]} versus p{where_depth}" + (f" ({shared})" if shared else "")
            )
            ax.set_title(title, loc="left", color=_INK)
            if len(series) > 1:
                ax.legend()
            band = (
                "no band (the Kaplan-Meier median and the argmax have no standard error here)"
                if stat in ("median", "argmax")
                else "bands are plus or minus one graph-cluster standard error"
            )
            caption = (
                f"{label_of[stat]} of {observable} at each measurement rate p, one line per "
                f"{'/'.join(vary) or 'condition'}; every graph has the same weight (v1.4 §6.3); "
                f"{band}. Rates of one (n, beta) share graphs, gates and mask uniforms (v1.4 "
                "§7.3), so neighboring points are positively correlated: compare rates with "
                f"aggregate's paired_rates, not by eye. {_DIAGNOSTIC}"
            )
            return _save(fig, Path(directory), stem or f"{observable}_{stat}_vs_p",
                         kind="vs_p", title=title, caption=caption,
                         header=["protocol", "n", "beta", "p", "p_float", "value", "se"],
                         rows=rows, data={"observable": observable, "statistic": stat,
                                          "depth": depth, "ell": ell, "field": field,
                                          "index": index, "where": where},
                         return_figure=return_figure)  # fmt: skip

    return _with_store(store, draw)


def _point(cond: Mapping[str, Any], kind: str, stat: str, depth: Any) -> tuple[float, float]:
    """One condition's plotted value and error (NaN where there is none)."""
    nan = math.nan
    if kind == "snapshot":
        depths = cond["depths"].tolist()
        if depth == "final":
            i = len(depths) - 1
        else:
            t = _sapi._int(depth, "depth", 0)
            if t not in depths:
                raise ValueError(f"depth {t} is not stored at {cond['n']}, {cond['beta']}, "
                                 f"{cond['p']}; it holds {depths}")  # fmt: skip
            i = depths.index(t)
        return float(cond["mean"][i]), float(cond["se"][i])
    if kind == "first_passage":
        entry = cond["ells"][0]
        if stat == "median":
            return (nan if entry["median"] is None else float(entry["median"])), nan
        if stat == "restricted_mean":
            return float(entry["restricted_mean"]), float(entry["restricted_mean_se"])
        return float(entry["censored_fraction"]), float(entry["censored_fraction_se"])
    if stat == "arithmetic":
        mean, se = cond["arithmetic_mean"], cond["arithmetic_se"]
    elif stat == "geometric":
        mean, se = cond["geometric_mean"], cond["geometric_se"]
    elif stat == "argmax":
        mean, se = cond["argmax"], None
    else:
        mean, se = cond[stat], None
    return (nan if mean is None else float(mean)), (nan if se is None else float(se))


_MEASURES = {
    "q": "q_G, rewired fraction of edges",
    "mean_distance": "mean shortest-path distance",
    "lambda2": "lambda_2 of D - A",
    "degree_variance": "degree variance",
    "components": "connected components",
}


def graph_measures(
    store: Any,
    directory: Any,
    *,
    where: Any = None,
    measures: Sequence[str] = ("q", "mean_distance", "lambda2", "degree_variance"),
    stem: str = "graph_measures",
    return_figure: bool = False,
) -> dict[str, Any]:
    """Graph measurements of the stored graphs versus ``beta``, one line per size (design §12):
    the equal-weight mean over the stored graphs of ``(n, beta)`` with the standard error of
    independent graphs (``stats.mean_over_graphs``). ``q`` is the fraction of edges outside
    ``C(n, 2)`` per ``2n``; ``mean_distance`` is over reachable pairs; ``lambda2`` is 0 for a
    disconnected graph."""
    measures = list(measures)
    for m in measures:
        if m not in _MEASURES:
            raise ValueError(f"measures must be among {sorted(_MEASURES)}, got {m!r}")

    def draw(s: Any) -> dict[str, Any]:
        filters = _sapi._where(where)
        filters.pop("p", None)
        filters.pop("t", None)
        manifest = s.manifest
        pairs = sorted({(c.n, c.beta.token) for c in manifest.conditions()
                        if ("n" not in filters or c.n in filters["n"])
                        and ("beta" not in filters or c.beta.token in filters["beta"])})  # fmt: skip
        values: dict[tuple[int, str], dict[str, Any]] = {}
        for n, beta in pairs:
            labels: list[int] = []
            per: dict[str, list[float]] = {m: [] for m in measures}
            for g in range(manifest.trials.graphs):
                if not s.has_graph(n, beta, g):
                    continue
                gm = s.graph_measures(n, beta, g)
                labels.append(g)
                for m in measures:
                    per[m].append(float(gm[m]))
            if labels:
                values[(n, beta)] = {"graphs": labels, **per}
        if not values:
            raise ValueError(
                "no stored graphs in this selection (graphs are stored when a unit runs)"
            )
        sizes = sorted({n for n, _ in values})
        colors = _colors(len(sizes))
        rows: list[list[Any]] = []
        cols = 2 if len(measures) > 1 else 1
        nrows = math.ceil(len(measures) / cols)
        with _context():
            fig, axes = _figure(nrows, cols, size=(6.8, 2.6 * nrows + 0.6))
            for k, m in enumerate(measures):
                ax = axes.ravel()[k]
                for n, color in zip(sizes, colors, strict=True):
                    pts = []
                    for (nn, beta), v in sorted(
                        values.items(), key=lambda kv: Token.parse(kv[0][1]).float64
                    ):
                        if nn != n:
                            continue
                        est = _stats.mean_over_graphs(v["graphs"], v[m])
                        pts.append((Token.parse(beta).float64, beta, float(est.mean),
                                    float(est.se), len(v["graphs"])))  # fmt: skip
                    x = np.array([q[0] for q in pts])
                    y = np.array([q[2] for q in pts])
                    e = np.array([q[3] for q in pts])
                    ax.plot(x, y, marker="o", color=color, label=f"n = {n}")
                    _band(ax, x, y, e, color)
                    rows.extend([n, q[1], q[0], m, q[2], q[3], q[4]] for q in pts)
                ax.set_xlabel("rewiring probability beta")
                ax.set_ylabel(_MEASURES[m])
            for ax in axes.ravel()[len(measures) :]:
                ax.set_visible(False)
            title = _title("graph measurements versus beta")
            fig.suptitle(title, x=0.01, ha="left", color=_INK, fontsize=10.0)
            if len(sizes) > 1:
                axes.ravel()[0].legend()
            caption = (
                "Measurements of the stored Watts-Strogatz graphs (design §12): the equal-weight "
                "mean over the graphs of each (n, beta) with the standard error of independent "
                "graphs (bands plus or minus one). q_G = |E(G) minus E(C(n, 2))| / (2n); the "
                "mean distance is over reachable distinct pairs; lambda_2 is the second-smallest "
                f"eigenvalue of the Laplacian D - A, 0 when the graph is disconnected. {_DIAGNOSTIC}"
            )
            return _save(fig, Path(directory), stem, kind="graph_measures", title=title,
                         caption=caption,
                         header=["n", "beta", "beta_float", "measure", "mean", "se", "graphs"],
                         rows=rows, data={"measures": measures, "where": where},
                         return_figure=return_figure)  # fmt: skip

    return _with_store(store, draw)


def collapse(
    report: Any,
    directory: Any,
    *,
    result: int = 0,
    stem: str | None = None,
    return_figure: bool = False,
) -> dict[str, Any]:
    """The scaling collapse of an :func:`sparsegf2.study.api.fss` report (the file it wrote, or
    the dict of :func:`sparsegf2.study.api.fss_report`): the used points in scaled coordinates
    ``x = (p - p_c) n^(1/nu)``, ``Y = y n^(-z)`` with error bars ``se n^(-z)``, one series per
    size, and the fitted master curve for the polynomial method. ``result`` picks the
    observable of a shared report."""
    if isinstance(report, (str, Path)):
        path = Path(report).expanduser()
        if not path.is_file():
            raise ValueError(f"report: no such file {str(report)!r} (fss writes it with out=DIR)")
        report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping) or report.get("schema") != _sapi.FSS_REPORT_SCHEMA:
        raise ValueError(f"report must be an fss report ({_sapi.FSS_REPORT_SCHEMA})")
    results = report["results"]
    index = _sapi._int(result, "result", 0)
    if index >= len(results):
        raise ValueError(f"result must be in 0..{len(results) - 1}")
    res = results[index]
    fit = res.get("fit")
    if fit is None or fit.get("p_c") is None or not math.isfinite(float(fit["p_c"])):
        raise ValueError(f"the report's fit of {res['observable']['theta']} has no parameters "
                         f"(status {res['status']})")  # fmt: skip
    p_c, z, nu = float(fit["p_c"]), float(fit["z"]), float(fit["nu"])
    used = [pt for pt in res["points"] if pt["used"]]
    sizes = sorted({int(pt["n"]) for pt in used})
    colors = _colors(len(sizes))
    theta = res["observable"]["theta"]
    rows: list[list[Any]] = []
    with _context():
        fig, axes = _figure()
        ax = axes[0, 0]
        for n, color in zip(sizes, colors, strict=True):
            pts = sorted((Token.parse(pt["p"]).float64, pt) for pt in used if int(pt["n"]) == n)
            p = np.array([q[0] for q in pts])
            y = np.array([float(q[1]["y"]) for q in pts])
            se = np.array([float(q[1]["se"]) for q in pts])
            x = (p - p_c) * float(n) ** (1.0 / nu)
            big_y = y * float(n) ** (-z)
            big_e = se * float(n) ** (-z)
            ax.errorbar(x, big_y, yerr=big_e, marker="o", linestyle="none", color=color,
                        label=f"n = {n}", elinewidth=1.0, capsize=2)  # fmt: skip
            rows.extend([theta, n, q[1]["p"], q[0], a, b, c]
                        for q, a, b, c in zip(pts, x.tolist(), big_y.tolist(), big_e.tolist(),
                                              strict=True))  # fmt: skip
        coefficients = fit.get("coefficients")
        if coefficients is not None and fit.get("window_range") is not None:
            lo, hi = (float(v) for v in fit["window_range"])
            grid = np.linspace(lo, hi, 200)
            curve = np.polynomial.polynomial.polyval(grid, np.asarray(coefficients, dtype=float))
            ax.plot(grid, curve, color=_INK_2, linewidth=1.2, label="fitted F (window)")
        ax.set_xlabel("x = (p - p_c) n^(1/nu)")
        ax.set_ylabel(f"Y = E[{theta}] n^(-z)")
        title = _title(f"scaling collapse of {theta} at beta = {report['beta']}")
        ax.set_title(title, loc="left", color=_INK)
        ax.legend()
        boot = res.get("bootstrap")
        intervals = ""
        if boot is not None:
            iv = boot["interval"]
            intervals = "; bootstrap {:g}% intervals p_c [{:.4g}, {:.4g}], z [{:.4g}, {:.4g}], nu " \
                        "[{:.4g}, {:.4g}]".format(100 * boot["level"], *iv["p_c"], *iv["z"],
                                                  *iv["nu"])  # fmt: skip
        caption = (
            f"Points (n, p) of E[{theta}] in the scaled coordinates of the unverified ansatz "
            f"E[theta] ~ n^z F((p - p_c) n^(1/nu)) (v1.4 §5.9) at the fitted p_c = {p_c:.4g}, "
            f"z = {z:.4g}, nu = {nu:.4g} ({fit['method']}, cost {float(fit['cost']):.3g}, "
            f"status {fit['status']}, {len(used)} points){intervals}. Error bars are one "
            f"graph-cluster standard error scaled by n^(-z). {_DIAGNOSTIC}"
        )
        return _save(fig, Path(directory), stem or f"collapse_{_safe(theta)}_beta{_safe(report['beta'])}",
                     kind="collapse", title=title, caption=caption,
                     header=["observable", "n", "p", "p_float", "x", "Y", "dY"], rows=rows,
                     data={"p_c": p_c, "z": z, "nu": nu, "method": fit["method"],
                           "beta": report["beta"]},
                     return_figure=return_figure)  # fmt: skip


_safe = _sapi.safe_name


#: The figures by the ``kind`` of :func:`sparsegf2.study.api.plot`.
PLOTS: dict[str, Callable[..., dict[str, Any]]] = {
    "k": k_trace,
    "survival": survival,
    "d_cyc": d_cyc,
    "peak_distribution": peak_distribution,
    "peak_vs_tau1": peak_vs_tau1,
    "vs_p": versus_p,
    "graph_measures": graph_measures,
    "collapse": collapse,
}
