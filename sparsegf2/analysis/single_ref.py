"""
Analysis utilities for the ``single_ref`` MIPT-probe picture.

This module reads the ``samples.parquet`` files produced by
:mod:`sparsegf2.circuits` when a sweep is run with ``picture="single_ref"``,
aggregates the single-qubit reference entropy ``obs.k`` by
``(n, p)``, computes the mean entropy ``<S>`` and its standard error,
and plots a "crossing" diagram: ``<S>`` versus ``p`` for different system
sizes ``n``. The crossing of the curves locates the MIPT critical point
``p_c``.

References
----------
Gullans & Huse 2020, Phys. Rev. X 10, 041020 (arXiv:1905.05195) -- the
purification-picture version; the single_ref probe is the minimal
analogue with one ancilla.

Skinner, Ruhman, Nahum 2019, Phys. Rev. X 9, 031009 (arXiv:1808.05953)
-- MIPT context.

Functions
---------
load_samples : polars-scan a whole run directory into a single frame.
aggregate_entropy : group by (n, p), return mean / std / sem / count.
plot_crossing : matplotlib crossing plot with error bars.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl


def load_samples(run_dir: Union[str, Path]) -> pl.DataFrame:
    """Load every ``samples.parquet`` under a run directory with hive
    partitioning, concatenated into one eager DataFrame.

    Parameters
    ----------
    run_dir : path-like
        Root of a run, i.e. the directory containing ``data/n=XXXX/p=YYYY/``
        cells. Typically produced by :class:`sparsegf2.circuits.SweepDriver`.

    Returns
    -------
    polars.DataFrame
        All per-sample rows across every cell in the run. Hive-partitioned
        columns ``n`` (int) and ``p`` (float) are lifted from the directory
        structure into the frame.
    """
    run_dir = Path(run_dir)
    pattern = str(run_dir / "data" / "**" / "samples.parquet")
    return pl.scan_parquet(pattern, hive_partitioning=True).collect()


def aggregate_entropy(
    df: pl.DataFrame, entropy_col: str = "obs.k"
) -> pl.DataFrame:
    """Aggregate entropy by ``(n, p)``: mean, sample std (ddof=1), SEM, count.

    The default ``entropy_col`` is ``obs.k``: for single_ref, the ``k``
    slot holds the entropy of the reference qubit (integer 0 or 1).

    Parameters
    ----------
    df : polars.DataFrame
        Long-format frame produced by :func:`load_samples`.
    entropy_col : str
        Column to aggregate. Default is the single_ref probe's entropy.

    Returns
    -------
    polars.DataFrame
        One row per ``(n, p)`` with columns
        ``S_mean``, ``S_std``, ``S_sem``, ``count``.
    """
    if entropy_col not in df.columns:
        raise KeyError(
            f"column {entropy_col!r} not in dataframe; available: "
            f"{sorted(df.columns)}"
        )
    agg = (
        df.group_by(["n", "p"])
        .agg(
            pl.col(entropy_col).mean().alias("S_mean"),
            pl.col(entropy_col).std(ddof=1).alias("S_std"),
            pl.col(entropy_col).count().alias("count"),
        )
        .with_columns(
            (pl.col("S_std") / pl.col("count").cast(pl.Float64).sqrt())
            .alias("S_sem")
        )
        .sort(["n", "p"])
    )
    return agg


def plot_crossing(
    agg: pl.DataFrame,
    *,
    out_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    title: Optional[str] = None,
    ax=None,
):
    """Crossing plot: ``<S>`` vs ``p`` for each system size ``n``.

    The curves for different ``n`` should intersect near the critical
    measurement rate ``p_c``; below ``p_c`` the probe entropy is pinned
    near 1 (volume-law side, the Bell-pair correlation survives) and
    above ``p_c`` it decays to 0 (area-law side, measurements collapse
    the correlation).

    Parameters
    ----------
    agg : polars.DataFrame
        Output of :func:`aggregate_entropy`.
    out_path : path-like, optional
        If given, save the figure to this path (png/pdf chosen by suffix).
    show : bool
        If True, call ``plt.show()``.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Plot onto this axis if provided; otherwise create a new figure.

    Returns
    -------
    (fig, ax) : matplotlib Figure and Axes.
    """
    import matplotlib.pyplot as plt  # imported lazily so non-plot callers
                                     # don't pay the matplotlib import cost.

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
    else:
        fig = ax.figure

    sizes = sorted(int(v) for v in agg["n"].unique().to_list())
    for n_val in sizes:
        sub = agg.filter(pl.col("n") == n_val).sort("p")
        ax.errorbar(
            sub["p"].to_numpy(),
            sub["S_mean"].to_numpy(),
            yerr=sub["S_sem"].to_numpy(),
            marker="o",
            linewidth=1.6,
            capsize=3,
            label=f"n={n_val}",
        )

    ax.set_xlabel("p  (measurement rate)")
    ax.set_ylabel(r"$\langle S \rangle$  (reference-qubit entropy)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.0, color="0.6", linestyle=":", linewidth=0.8)
    ax.axhline(1.0, color="0.6", linestyle=":", linewidth=0.8)
    ax.set_title(title if title is not None else "single_ref MIPT probe")
    ax.legend(title="system size", loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


__all__ = ["load_samples", "aggregate_entropy", "plot_crossing"]
