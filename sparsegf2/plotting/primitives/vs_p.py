"""
``plot_vs_p`` — the single MVP plot primitive.

Draw one curve per size ``n`` showing the mean of ``y`` across samples as a
function of the measurement rate ``p`` (or any other partition-columned x).

Supports:

- sample-level filters (``filter="obs.k > 0"``) evaluated pre-aggregation,
- error bars (``errors="bar"``), error bands (``errors="band"``), or no errors,
- automatic error-metric selection: Wilson for binary columns, SEM otherwise,
- explicit override to ``std``, ``ci95`` (bootstrap 95% CI), or ``none``,
- ``sizes=[..]`` to filter which curves are drawn; default is every size present,
- derived-column aliases (``k_over_n``, ``d_cont_over_n``, ``d_min_over_n``, ...).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from sparsegf2.plotting.data import _aggregate, _apply_filter, _ensure_column, _load
from sparsegf2.plotting.style import rc_preset


Source = Union[Path, str, Iterable[Union[Path, str]]]


# ══════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════

def plot_vs_p(
    source: Source,
    y: str,
    *,
    x: str = "p",
    sizes: Optional[List[int]] = None,
    errors: str = "band",
    error_metric: str = "auto",
    filter: Optional[str] = None,
    label_fmt: str = "n={n}",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xscale: str = "linear",
    yscale: str = "linear",
    save: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (6.0, 4.0),
    style: Optional[str] = "research",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot ``y`` vs ``x`` (usually ``p``) grouped by system size ``n``.

    Parameters
    ----------
    source : path or list of paths
        One or more run directories produced by ``sparsegf2.circuits``.
    y : str
        Column name to plot on the y-axis. Accepts any column in
        ``samples.parquet`` or any ``analysis/<name>.parquet`` (prefixed with
        ``<name>.``, e.g. ``"distances.d_cont"``). Also accepts derived-column
        aliases (``"k_over_n"``, ``"d_cont_over_n"``, ``"d_min_over_n"``, ...).
    x : str
        X-axis column. Default ``"p"``; ``"n"`` also works for fixed-p slices.
    sizes : list of int, optional
        When set, only curves for these sizes are drawn. Default: all sizes
        present in the source.
    errors : {"band", "bar", "none"}
        How to render uncertainty. ``"band"`` uses a semi-transparent fill;
        ``"bar"`` uses matplotlib errorbars; ``"none"`` draws just the line.
    error_metric : {"auto", "sem", "std", "ci95", "wilson", "none"}
        How to compute the error size per point. ``"auto"`` picks Wilson for
        binary ``{0, 1}`` columns and SEM otherwise. ``"none"`` forces
        zero-width errors regardless of ``errors``.
    filter : str, optional
        SQL-style predicate applied to the sample-level DataFrame before
        aggregation. Example: ``"obs.k > 0"`` to get conditional means.
    label_fmt : str
        Format string used for legend labels; ``{n}`` is substituted.

    Returns
    -------
    (fig, ax) : matplotlib Figure and Axes.
    """
    with rc_preset(style):
        df = _load(source)
        df = _ensure_column(df, y)
        df = _apply_filter(df, filter)

        if sizes is not None:
            df = df.filter(pl.col("n").is_in(list(sizes)))
        if len(df) == 0:
            raise ValueError("No rows remain after filter + size selection.")

        agg = _aggregate(df, x=x, y=y, error_metric=error_metric)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        unique_sizes = sorted(set(agg["n"].to_list()))
        cmap = plt.get_cmap("viridis", max(len(unique_sizes), 1))

        for i, n in enumerate(unique_sizes):
            sub = agg.filter(pl.col("n") == n).sort(x)
            xs = sub[x].to_numpy()
            ys = sub["mean"].to_numpy()
            el = sub["err_low"].to_numpy()
            eh = sub["err_high"].to_numpy()
            color = cmap(i)
            label = label_fmt.format(n=int(n))

            if errors == "none" or error_metric == "none":
                ax.plot(xs, ys, marker="o", color=color, label=label)
            elif errors == "bar":
                ax.errorbar(xs, ys, yerr=(el, eh), marker="o",
                            linestyle="-", color=color, label=label,
                            capsize=2.5)
            elif errors == "band":
                ax.plot(xs, ys, marker="o", color=color, label=label)
                ax.fill_between(xs, ys - el, ys + eh,
                                color=color, alpha=0.18, linewidth=0)
            else:
                raise ValueError(
                    f"errors must be 'band', 'bar', or 'none'; got {errors!r}"
                )

        ax.set_xlabel(xlabel if xlabel is not None else x)
        ax.set_ylabel(ylabel if ylabel is not None else y)
        if title is not None:
            ax.set_title(title)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if unique_sizes:
            ax.legend(loc="best")

        if save is not None:
            fig.savefig(save, bbox_inches="tight")

    return fig, ax


__all__ = ["plot_vs_p"]
