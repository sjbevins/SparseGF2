"""Build and execute ``notebooks/circuits/graphs.ipynb``."""

from __future__ import annotations

from _nbtools import build_and_execute, code, md

CELLS = [
    md(
        r"""# `sparsegf2.circuits.graphs` - interaction geometry

A `GraphTopology` stores the allowed two-qubit edges, graph6 provenance, and
matching information used by brickwork schedules. The scheduler decides which
allowed edges fire. Graph construction itself does not touch the simulator.

`from_spec` supports six string families:

1. `cycle`
2. `complete`
3. `path`
4. `lattice_2d`
5. `newman_watts`
6. `watts_strogatz`

The two small-world families accept inline parameters. `from_networkx` adapts
any simple undirected NetworkX graph with arbitrary original node labels.
"""
    ),
    code(
        "from sparsegf2.circuits import from_spec\n\n"
        "specs = [\n"
        "    'cycle', 'complete', 'path', 'lattice_2d',\n"
        "    'newman_watts(k=2,p=0.2)',\n"
        "    'watts_strogatz(k=2,beta=0.25)',\n"
        "]\n"
        "graphs = [from_spec(spec, 16, seed=11) for spec in specs]\n"
        "for spec, graph in zip(specs, graphs, strict=True):\n"
        "    print(\n"
        "        f'{spec:<35} |E|={len(graph.edges):>3}  '\n"
        "        f'stochastic={graph.is_stochastic!s:<5}  graph6 chars={len(graph.graph6)}'\n"
        "    )\n"
    ),
    md(
        r"""## Matchings and gate schedules

`brickwork` requires a matching. `round_robin` and `palette` consume a stored
1-factorization; `fresh` draws a perfect matching when the geometry supplies a
sampler. The other gate schedules need only a nonempty edge set:
`random_edge` draws distinct edges, `random_pool` draws with replacement, and
`all_edges` fires every stored edge in order.

Irregular or stochastic graphs generally do not have a fixed
1-factorization, but remain valid with edge-based schedules.
"""
    ),
    code(
        "cycle = from_spec('cycle', 8)\n"
        "complete = from_spec('complete', 8)\n"
        "print('C8 factors / edges:', cycle.chi_prime, '/', len(cycle.edges))\n"
        "print('K8 factors / edges:', complete.chi_prime, '/', len(complete.edges))\n"
        "assert sorted(edge for factor in cycle.one_factorization for edge in factor) == cycle.edges\n"
        "print('cycle factors partition its edge set:', True)\n"
    ),
    md(
        r"""## Stochastic realization seeds

For a stochastic string specification, `CircuitConfig.base_seed` is the
quenched graph-realization seed. Reusing `(specification, n, seed)` reconstructs
the same sorted edge list and graph6 string. Changing `sample_seed` later changes
the circuit trajectory without changing this geometry.
"""
    ),
    code(
        "spec = 'watts_strogatz(k=2,beta=0.25)'\n"
        "a = from_spec(spec, 32, seed=7)\n"
        "b = from_spec(spec, 32, seed=7)\n"
        "c = from_spec(spec, 32, seed=8)\n"
        "print('same realization seed:', a.edges == b.edges)\n"
        "print('different realization seed:', a.edges != c.edges)\n"
        "print('edge count and mean degree:', len(a.edges), 2 * len(a.edges) / a.n)\n"
    ),
    md(
        r"""## Arbitrary NetworkX geometry

`from_networkx` relabels nodes to `0, ..., n-1`, rejects multigraphs and
self-loops, canonicalizes the edge list, and records graph6 provenance.
`CircuitConfig` also accepts the NetworkX object directly and performs this
adaptation during validation.
"""
    ),
    code(
        "import networkx as nx\n"
        "from sparsegf2.circuits import CircuitBuilder, CircuitConfig, from_networkx\n\n"
        "raw = nx.wheel_graph(8)\n"
        "adapted = from_networkx(raw, name='wheel8')\n"
        "cfg = CircuitConfig(\n"
        "    graph_spec=raw, n=8, gating_mode='random_edge',\n"
        "    total_layers_override=2,\n"
        ")\n"
        "print(adapted.name, '|E|=', len(adapted.edges), 'graph6=', adapted.graph6)\n"
        "print('direct NetworkX config resolves to:', CircuitBuilder(cfg, 0).graph.name)\n"
    ),
    md(
        r"""## Summary

The six named families and `from_networkx` share one canonical
`GraphTopology` representation. Deterministic graphs can expose reusable
1-factorizations; stochastic graphs preserve their exact realization through
the seed and graph6 metadata.
"""
    ),
]

if __name__ == "__main__":
    build_and_execute("graphs.ipynb", CELLS)
