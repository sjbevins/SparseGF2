"""Weight statistics and Tanner graph analysis for SparseGF2.

A pure read-only analysis layer that computes weight distributions,
pivot effectiveness indices, and Tanner graph representations from
the simulator's public state arrays.

Public API
----------
compute_weight_stats(sim) -> WeightStats
    All weight distribution statistics in a frozen dataclass.
verify_weight_mass_identity(sim) -> dict
    Consistency oracle: verifies n*abar = 2n*wbar.
compute_pivot_effectiveness(sim) -> dict
    PEI and WCP proxy values.
build_tanner_graph(sim) -> networkx.Graph
    Bipartite Tanner graph with Pauli-annotated edges.
build_tanner_hypergraph(sim) -> dict
    Hyperedge representation keyed by generator index.
plot_tanner_graph(sim, ...) -> matplotlib.figure.Figure
    Publication-quality Tanner graph visualization.
observe(sim, p=None) -> dict
    All scalar observables as a flat dict for pandas sweeps.
"""

from sparsegf2.analysis.observables import WeightStats, observe
from sparsegf2.analysis.weight_stats import (
    compute_weight_stats,
    verify_weight_mass_identity,
    compute_pivot_effectiveness,
)
from sparsegf2.analysis.tanner_graph import (
    build_tanner_graph,
    build_tanner_hypergraph,
    plot_tanner_graph,
)
from sparsegf2.analysis.single_ref import (
    load_samples as load_single_ref_samples,
    aggregate_entropy as aggregate_single_ref_entropy,
    plot_crossing as plot_single_ref_crossing,
    load_timeseries as load_single_ref_timeseries,
    compute_tau as compute_single_ref_tau,
    plot_purification_decay as plot_single_ref_purification_decay,
    plot_tau_scaling as plot_single_ref_tau_scaling,
    plot_psurv_vs_tn_grid,
    compute_tau_over_n_vs_p,
    plot_tau_over_n_vs_p,
    analyze_single_ref,
    detect_picture,
    has_timeseries,
)

__all__ = [
    'WeightStats',
    'compute_weight_stats',
    'verify_weight_mass_identity',
    'compute_pivot_effectiveness',
    'build_tanner_graph',
    'build_tanner_hypergraph',
    'plot_tanner_graph',
    'observe',
    'load_single_ref_samples',
    'aggregate_single_ref_entropy',
    'plot_single_ref_crossing',
    'load_single_ref_timeseries',
    'compute_single_ref_tau',
    'plot_single_ref_purification_decay',
    'plot_single_ref_tau_scaling',
    'plot_psurv_vs_tn_grid',
    'compute_tau_over_n_vs_p',
    'plot_tau_over_n_vs_p',
    'analyze_single_ref',
    'detect_picture',
    'has_timeseries',
]
