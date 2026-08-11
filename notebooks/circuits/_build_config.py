"""Build and execute ``notebooks/circuits/config.ipynb``."""

from __future__ import annotations

from _nbtools import build_and_execute, code, md

CELLS = [
    md(
        r"""# `sparsegf2.circuits.config` - one validated circuit cell

`CircuitConfig` contains every non-sample knob for one circuit realization.
Construction validates the graph, picture, gate schedule, measurement rule,
depth, and simulator options before a run begins. Parameter sweeps and named
observables are already implemented in `sparsegf2.analysis`; they are not
future placeholders.

The six string graph families are `cycle`, `complete`, `path`, `lattice_2d`,
`newman_watts`, and `watts_strogatz`. A prebuilt `GraphTopology` or an arbitrary
simple undirected NetworkX graph can also be supplied through `from_networkx`.
"""
    ),
    md(
        r"""## Current mode surface

- Pictures: `pure_state`, `purification`, `single_ref`.
- Gating: `brickwork`, `random_edge`, `random_pool`, `all_edges`.
- Matching for brickwork: `round_robin`, `palette`, `fresh`.
- Measurements: `bernoulli`, `gated`, `random_pair`, `uniform_count`.
- Depth: `O(n)`, `O(log_n)`, `until_purified`, or a literal
  `total_layers_override`.

`random_edge` samples distinct graph edges; `random_pool` samples with
replacement; `all_edges` fires the graph's stored edge list in deterministic
order. `uniform_count` chooses `meas_count` distinct candidates and then applies
the Bernoulli measurement probability to those candidates.
"""
    ),
    code(
        "from sparsegf2.circuits import CircuitConfig\n"
        "from sparsegf2.circuits.config import GATING_MODES\n"
        "from sparsegf2.circuits.measurements import MEASUREMENT_MODES\n\n"
        "print('gating modes     :', GATING_MODES)\n"
        "print('measurement modes:', MEASUREMENT_MODES)\n"
        "for mode in GATING_MODES:\n"
        "    kwargs = {'gates_per_layer': 2} if mode == 'random_edge' else {}\n"
        "    cfg = CircuitConfig(\n"
        "        graph_spec='cycle', n=8, gating_mode=mode,\n"
        "        total_layers_override=3, **kwargs,\n"
        "    )\n"
        "    print(f'{mode:>11}: layers={cfg.total_layers()}, expected ratio={cfg.expected_gate_to_meas_ratio():.3g}')\n"
    ),
    md(
        r"""## Measurement modes and literal depth

The mode-specific fields are checked eagerly: `gates_per_layer` belongs only
to `random_edge` and `random_pool`, while `meas_count` belongs only to
`uniform_count`. A positive `total_layers_override` short-circuits both the
depth-mode formula and the random-edge gate-budget rescaling. This is the right
knob for an experiment specified at an exact measured depth.
"""
    ),
    code(
        "for mode in MEASUREMENT_MODES:\n"
        "    kwargs = {'meas_count': 3} if mode == 'uniform_count' else {}\n"
        "    cfg = CircuitConfig(\n"
        "        graph_spec='cycle', n=8, measurement_mode=mode, p=0.25,\n"
        "        total_layers_override=5, **kwargs,\n"
        "    )\n"
        "    print(f'{mode:>13}: layers={cfg.total_layers()}, ratio={cfg.expected_gate_to_meas_ratio():.3g}')\n\n"
        "scaled = CircuitConfig(\n"
        "    graph_spec='cycle', n=8, gating_mode='random_edge',\n"
        "    gates_per_layer=1, depth_factor=2,\n"
        ")\n"
        "literal = CircuitConfig(\n"
        "    graph_spec='cycle', n=8, gating_mode='random_edge',\n"
        "    gates_per_layer=1, depth_factor=2, total_layers_override=7,\n"
        ")\n"
        "print('formula / literal depth:', scaled.total_layers(), '/', literal.total_layers())\n"
    ),
    md(
        r"""## Reproducibility and analysis

Schema v2 keys every sample by the pair `(base_seed, sample_seed)`, never by
their sum. Distinct pairs therefore cannot alias when a sweep changes both the
quenched graph seed and the trajectory seed. `simulate(..., analyses=...)`
computes registered or custom observables on the live final tableau, while
`sparsegf2.analysis.sweep` handles many configurations and seeds.
"""
    ),
    code(
        "from sparsegf2.circuits import simulate\n\n"
        "cfg = CircuitConfig(\n"
        "    graph_spec='cycle', n=8, picture='purification', p=0.2,\n"
        "    total_layers_override=4, base_seed=17,\n"
        ")\n"
        "record = simulate(\n"
        "    cfg, sample_seed=3, analyses=['code_dimension', 'half_cut_entropy']\n"
        ")\n"
        "print('sample identity:', (cfg.base_seed, record.sample_seed))\n"
        "print('online analyses:', record.analyses)\n"
        "print('serialized depth override:', cfg.to_dict()['total_layers_override'])\n"
    ),
    md(
        r"""## Summary

`CircuitConfig` is the fail-fast description of one cell. It covers all six
named graph families, arbitrary NetworkX geometry, four gate schedules, four
measurement schedules, literal depth, schema-v2 pair seeding, and the current
online/offline analysis layer.
"""
    ),
]

if __name__ == "__main__":
    build_and_execute("config.ipynb", CELLS)
