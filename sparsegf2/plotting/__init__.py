"""
sparsegf2.plotting — declarative plotting over run directories.

MVP: one primitive, :func:`plot_vs_p`, which loads any scalar column from
``samples.parquet`` or ``analysis/*.parquet`` (or ``aggregates.parquet``),
aggregates across samples with optional filtering, and draws one curve per
size ``n``.

Future primitives (heatmap, histogram, rate-distance Pareto, FSS collapse)
slot in as sibling functions in :mod:`sparsegf2.plotting.primitives` sharing
the loading, filtering, and aggregation helpers in :mod:`sparsegf2.plotting.data`.
"""
from __future__ import annotations

from sparsegf2.plotting.primitives.vs_p import plot_vs_p
from sparsegf2.plotting.errors import (
    sem, std, ci95_bootstrap, wilson,
    ERROR_METRICS, pick_error_metric,
)
from sparsegf2.plotting.aliases import DERIVED_ALIASES, resolve_alias

__all__ = [
    "plot_vs_p",
    "sem",
    "std",
    "ci95_bootstrap",
    "wilson",
    "ERROR_METRICS",
    "pick_error_metric",
    "DERIVED_ALIASES",
    "resolve_alias",
]
