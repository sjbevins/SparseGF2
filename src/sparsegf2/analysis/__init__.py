"""The analysis layer: named observables + offline tableau analysis + sweeps.

This subpackage sits *above* :mod:`sparsegf2.core` and :mod:`sparsegf2.circuits`
and consumes both. It provides:

* a registry of named, picture-aware analyses (:data:`ANALYSES`,
  :func:`register_analysis`, :func:`resolve_analyses`) used both online
  (``simulate(..., analyses=[...])``) and offline;
* offline helpers (:func:`reconstruct`, :func:`analyze`,
  :func:`analyze_tableaux`) that run the same analyses over saved tableaux;
* the parameter-sweep driver (:func:`sweep`, :class:`SweepResult`).

Built-in analysis names (pass any by name in ``analyses=[...]``):
``half_cut_entropy``, ``code_dimension``, ``code_rate``,
``contiguous_distance``, ``ref_entropy``, ``bipartite_mutual_info``,
``tripartite_mutual_info``, ``average_stabilizer_weight``,
``stabilizer_weight_spectrum``, ``entropy_profile`` (see :data:`ANALYSES` for
the live registry). Or pass your own ``fn(sim, spec) -> value`` callable.

The runtime floor stays numpy + numba: parquet/HDF5/parallel dependencies are
optional extras, imported lazily only when their feature is used.
"""

from __future__ import annotations

from sparsegf2.analysis._runtime import available_cores
from sparsegf2.analysis.fss_collapse import (
    fit_fixed_pc,
    fit_three_param,
    graph_bootstrap,
    hh_loss,
    master_curve_loss,
    robust_pc,
    smooth_pc_collapse,
)
from sparsegf2.analysis.offline import analyze, analyze_tableaux, reconstruct
from sparsegf2.analysis.plotting import aggregate, observable_columns, plot_study, plot_vs
from sparsegf2.analysis.registry import (
    ANALYSES,
    Analysis,
    register_analysis,
    resolve_analyses,
)
from sparsegf2.analysis.study import Study
from sparsegf2.analysis.sweep import SweepResult, sweep

__all__ = [
    "ANALYSES",
    "Analysis",
    "Study",
    "SweepResult",
    "aggregate",
    "analyze",
    "analyze_tableaux",
    "available_cores",
    "fit_fixed_pc",
    "fit_three_param",
    "graph_bootstrap",
    "hh_loss",
    "master_curve_loss",
    "observable_columns",
    "plot_study",
    "plot_vs",
    "reconstruct",
    "register_analysis",
    "resolve_analyses",
    "robust_pc",
    "smooth_pc_collapse",
    "sweep",
]
