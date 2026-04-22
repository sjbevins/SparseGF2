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

__all__ = [
    'WeightStats',
    'compute_weight_stats',
    'verify_weight_mass_identity',
    'compute_pivot_effectiveness',
    'build_tanner_graph',
    'build_tanner_hypergraph',
    'plot_tanner_graph',
    'observe',
]
