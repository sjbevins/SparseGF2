"""Plotting helpers for studies / sweep results (lazy matplotlib + pandas).

Reads a :class:`~sparsegf2.analysis.Study` (or its rows / a DataFrame) and draws
the standard MIPT plot: an observable vs the measurement rate ``p``, one curve
per system size ``n`` (the shape you read a finite-size crossing off).

:func:`plot_study` **auto-detects** every numeric observable / analysis column,
so after :meth:`Study.augment` adds new columns (e.g. ``code_rate``,
``contiguous_distance``) the same call plots them too, with no edits needed.

Needs the ``viz`` (matplotlib) and ``data`` (pandas) extras; both are imported
lazily, so importing this module never pulls them in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2.errors import InvalidArgumentError

_FIXED_OBSERVABLES = ("code_dimension", "ref_entropy", "entropy_half_cut")


def _frame(source: Any):
    """Accept a Study, a list of row dicts, or a DataFrame; return a DataFrame."""
    if hasattr(source, "to_pandas"):
        return source.to_pandas()
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise InvalidArgumentError(
            "plotting needs the 'data' extra (pandas): pip install sparsegf2[data]"
        ) from e
    if isinstance(source, pd.DataFrame):
        return source
    return pd.DataFrame(list(source))


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise InvalidArgumentError(
            "plotting needs the 'viz' extra (matplotlib): pip install sparsegf2[viz]"
        ) from e
    return plt


def observable_columns(source: Any) -> list[str]:
    """The numeric observable / analysis columns worth plotting.

    These are the ``a.<name>`` analysis columns plus the fixed observables
    (``code_dimension`` / ``ref_entropy`` / ``entropy_half_cut``), keeping only
    columns that are numeric and not entirely null. When both ``a.X`` and the
    bare ``X`` are present (the runner fills both), the ``a.X`` one is kept.
    """
    df = _frame(source)
    candidates = [c for c in df.columns if c.startswith("a.")]
    have_analysis = {c[2:] for c in candidates}
    candidates += [c for c in _FIXED_OBSERVABLES if c in df.columns and c not in have_analysis]
    out: list[str] = []
    for c in candidates:
        s = df[c].dropna()
        if len(s) and np.issubdtype(np.asarray(s.to_numpy()).dtype, np.number):
            out.append(c)
    return sorted(set(out))


def aggregate(
    source: Any,
    ycol: str,
    *,
    xcol: str = "p",
    group: str = "n",
    err: str = "sem",
):
    """Per-(``group``, ``xcol``) mean, error, and sample count of ``ycol``.

    The central value is the mean over every row in the bin (the seeds, and any
    unswept repetition). ``err`` selects the uncertainty reported and tracked:

    * ``"sem"`` (default): standard error of the mean, ``std / sqrt(N)``, the
      error bar on an average over seeds.
    * ``"std"``: the sample standard deviation (the spread itself).

    Returns a tidy DataFrame ``[group?, xcol, "mean", "err", "count"]`` (the
    ``group`` column is present only when it varies; ``count`` is the number of
    samples in the bin). Single-sample bins get ``err = 0``.
    """
    df = _frame(source)
    if ycol not in df.columns:
        raise InvalidArgumentError(
            f"no column {ycol!r}; available observables: {observable_columns(df)}"
        )
    if xcol not in df.columns:
        raise InvalidArgumentError(f"no x-axis column {xcol!r}; columns: {list(df.columns)}")
    if err not in ("sem", "std"):
        raise InvalidArgumentError(f"err must be 'sem' or 'std', got {err!r}")
    keys = [group, xcol] if (group in df.columns and df[group].nunique() > 1) else [xcol]
    stats = df.groupby(keys)[ycol].agg(["mean", "std", "count"]).reset_index()
    spread = stats["std"].fillna(0.0)  # std is NaN for a single-sample bin
    stats["err"] = spread / np.sqrt(stats["count"]) if err == "sem" else spread
    return stats[[*keys, "mean", "err", "count"]]


def _draw_series(ax, x, y, yerr, *, errorstyle: str, label: str | None = None, **plot_kw: Any):
    """Draw one curve carrying its uncertainty as error bars or a shaded band."""
    if errorstyle == "bars":
        ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, label=label, **plot_kw)
    elif errorstyle == "band":
        (line,) = ax.plot(x, y, "o-", label=label, **plot_kw)
        ax.fill_between(x, y - yerr, y + yerr, color=line.get_color(), alpha=0.2, linewidth=0)
    else:
        raise InvalidArgumentError(f"errorstyle must be 'bars' or 'band', got {errorstyle!r}")


def plot_vs(
    source: Any,
    ycol: str,
    *,
    xcol: str = "p",
    group: str = "n",
    err: str = "sem",
    errorstyle: str = "bars",
    ax: Any = None,
    **plot_kw: Any,
):
    """Plot mean(``ycol``) vs ``xcol`` with uncertainty, one curve per ``group``.

    The error per point is computed across the rows in each bin (the seeds): the
    standard error of the mean by default (``err="sem"``), or the spread
    (``err="std"``). ``errorstyle="bars"`` (default) draws error bars;
    ``"band"`` shades a ``±err`` band around the mean. ``group`` is ignored when
    the column is absent or constant (a single curve is drawn). Returns the Axes.
    """
    df = _frame(source)
    plt = _plt()
    stats = aggregate(df, ycol, xcol=xcol, group=group, err=err)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    if group in stats.columns:
        for gval, sub in stats.groupby(group):
            _draw_series(
                ax,
                sub[xcol].to_numpy(),
                sub["mean"].to_numpy(),
                sub["err"].to_numpy(),
                errorstyle=errorstyle,
                label=f"{group}={gval}",
                **plot_kw,
            )
        ax.legend(title=group)
    else:
        _draw_series(
            ax,
            stats[xcol].to_numpy(),
            stats["mean"].to_numpy(),
            stats["err"].to_numpy(),
            errorstyle=errorstyle,
            **plot_kw,
        )
    ax.set_xlabel(xcol)
    ax.set_ylabel(f"mean {ycol}")
    ax.grid(True, ls=":", alpha=0.5)
    return ax


def plot_study(
    source: Any,
    *,
    xcol: str = "p",
    group: str = "n",
    ycols: list[str] | None = None,
    err: str = "sem",
    errorstyle: str = "bars",
    out: str | Path | None = None,
    ncols: int = 2,
):
    """Auto-plot every available observable vs ``xcol`` (grouped by ``group``).

    One panel per observable, each with uncertainty across seeds. ``ycols``
    overrides the auto-detected set; ``err`` (``"sem"``/``"std"``) and
    ``errorstyle`` (``"bars"`` default, or ``"band"``) are forwarded to
    :func:`plot_vs`. Saves to ``out`` (PNG/PDF/…) when given. Returns the Figure.
    Re-run after :meth:`Study.augment` to include newly-added columns.
    """
    df = _frame(source)
    plt = _plt()
    ycols = ycols or observable_columns(df)
    if not ycols:
        raise InvalidArgumentError("no numeric observable columns to plot")
    n = len(ycols)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    for ax, yc in zip(flat, ycols, strict=False):
        plot_vs(df, yc, xcol=xcol, group=group, err=err, errorstyle=errorstyle, ax=ax)
        ax.set_title(yc)
    for ax in flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=130, bbox_inches="tight")
    return fig


__all__ = ["aggregate", "observable_columns", "plot_study", "plot_vs"]
