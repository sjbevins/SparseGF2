"""Build + execute ``notebooks/circuits/graphs.ipynb`` - graph topologies,
1-factorization theory, graph6, and the topology-vs-simulatability finding."""

from __future__ import annotations

from _nbtools import build_and_execute, code, md

CELLS = [
    md(r"""# `sparsegf2.circuits.graphs` - who interacts with whom

[`graphs.py`](../../src/sparsegf2/circuits/graphs.py) defines the **graph**
whose edges are the only places a two-qubit gate may land. The graph fixes
the *connectivity*; the scheduler picks *which* edges fire each layer. This
module is pure combinatorics - **no simulator import** - so we can reason
about it entirely in graph theory.

## Contents
1. Graph-defined circuits & perfect matchings
2. 1-factorizations, edge colorings, and Vizing's theorem
3. The even cycle $C_n$ - two matchings
4. The complete graph $K_n$ - the round-robin 1-factorization
5. graph6 encoding (McKay)
6. `GraphTopology` and the constructors, line by line
7. Topology vs. *simulatability*: what the benchmark actually found

**Sources.** Vizing 1964 (chromatic index); König 1916 (bipartite edge
coloring); Anderson, *1-Factorizations of Complete Graphs*; McKay's
[graph6 format](https://users.cecs.anu.edu.au/~bdm/data/formats.txt)."""),
    md(r"""## 1. Graph-defined circuits & perfect matchings

A **matching** is a set of edges with no shared vertex; a **perfect
matching** covers every vertex exactly once. In a *brickwork* layer we want
to apply $n/2$ disjoint two-qubit gates simultaneously - that is precisely a
perfect matching of the qubit graph. So a graph supports brickwork iff it
has perfect matchings, and supports *deterministic* brickwork cycling iff
its edges partition into perfect matchings - a **1-factorization**."""),
    code(
        "import numpy as np\n"
        "from sparsegf2.circuits.graphs import (\n"
        "    GraphTopology, cycle_graph, complete_graph, from_spec, graph6_encode,\n"
        ")\n"
        "g = cycle_graph(8)\n"
        "print('cycle(8) edges:', g.edges)\n"
        "print('degree_max:', g.degree_max, '| has perfect matching:', g.has_perfect_matching)\n"
    ),
    md(r"""## 2. 1-factorizations, edge colorings, Vizing

A **1-factorization** of a graph is a partition of its edge set into perfect
matchings (called *1-factors*). Equivalently it is a *proper edge coloring*
using exactly $\Delta$ colors, where $\Delta$ is the maximum degree: each
color class is a perfect matching, and "proper" means no two edges of the
same color share a vertex.

**Vizing's theorem (1964):** every simple graph has chromatic index
$\chi' \in \{\Delta, \Delta+1\}$. Graphs with $\chi'=\Delta$ are *Class 1*;
those needing $\Delta+1$ are *Class 2*. A $\Delta$-regular graph is
1-factorable **iff** it is Class 1 (then the $\Delta$ color classes are each
perfect). Both supported graphs are regular and Class 1 for even $n$:

- $C_n$ (even): $2$-regular, $\chi'=2$ → two 1-factors.
- $K_n$ (even): $(n-1)$-regular, $\chi'=n-1$ → $n-1$ 1-factors.

For **odd** $n$ neither has a perfect matching at all (an odd number of
vertices can't be perfectly matched), so `one_factorization is None` and the
config rejects brickwork on it. `chi_prime` reports the number of 1-factors:"""),
    code(
        "for n in (6, 7, 8):\n"
        "    cg, kg = cycle_graph(n), complete_graph(n)\n"
        "    print(f'n={n}:  cycle chi_prime={cg.chi_prime}   complete chi_prime={kg.chi_prime}'\n"
        "          f'   (None = no 1-factorization, i.e. odd n)')\n"
    ),
    md(r"""## 3. The even cycle $C_n$ - two alternating matchings

Label the ring $0\!-\!1\!-\!\cdots\!-\!(n-1)\!-\!0$. The **even** 1-factor
takes edges $(0,1),(2,3),\dots$; the **odd** 1-factor takes
$(1,2),(3,4),\dots,(n-1,0)$. Each is a perfect matching, and together they
use every cycle edge exactly once - a 1-factorization with $\chi'=2$. The
code (`_cycle_one_factorization`) builds exactly these two lists, then
canonicalizes each edge to $(\min,\max)$. Let's verify the partition
property directly:"""),
    code(
        "g = cycle_graph(8)\n"
        "f0, f1 = g.one_factorization\n"
        "print('even 1-factor:', f0)\n"
        "print('odd  1-factor:', f1)\n"
        "# each is a perfect matching: every vertex appears exactly once\n"
        "for f in (f0, f1):\n"
        "    verts = sorted(q for e in f for q in e)\n"
        "    assert verts == list(range(8))\n"
        "# union = all edges, each once\n"
        "assert sorted(f0 + f1) == sorted(g.edges)\n"
        "print('two 1-factors partition E(C_8):', True)\n"
    ),
    md(r"""## 4. The complete graph $K_n$ - round-robin

For even $n$, $K_n$ decomposes into $n-1$ perfect matchings via the classic
**round-robin tournament** schedule: fix vertex $n-1$, arrange the other
$n-1$ on a circle, and rotate. In round $r$, pair $n-1$ with $r$, and for
$i=1,\dots,n/2-1$ pair $(r+i)\bmod(n-1)$ with $(r-i)\bmod(n-1)$. The $n-1$
rounds are exactly the 1-factors (Anderson). Each "round" is one layer of a
brickwork-on-$K_n$ circuit. Verify the partition:"""),
    code(
        "g = complete_graph(6)\n"
        "print('chi_prime (rounds):', g.chi_prime)        # n-1 = 5\n"
        "union = [tuple(e) for f in g.one_factorization for e in f]\n"
        "assert sorted(union) == sorted(g.edges)           # covers every edge once\n"
        "for f in g.one_factorization:\n"
        "    assert sorted(q for e in f for q in e) == list(range(6))   # perfect each round\n"
        "print('round 0:', g.one_factorization[0])\n"
        "print('n-1 rounds partition E(K_6):', True)\n"
    ),
    md(r"""## 5. graph6 - compact metadata encoding

`graph6` (McKay) stores a simple graph as a short ASCII string: a header
`N(n)` encoding the vertex count, then the bits of the **upper triangle** of
the adjacency matrix in column-major order, packed 6 bits per character with
a $+63$ offset (so bytes land in printable ASCII). `graph6_encode` implements
exactly this; `GraphTopology` stores the string so a realization's graph is
recoverable from metadata without re-deriving it. Two textbook checks: the
single vertex `@`, and the triangle `Bw`."""),
    code(
        "print('K1  :', repr(graph6_encode(1, [])))                       # '@'\n"
        "print('C3  :', repr(graph6_encode(3, [(0,1),(1,2),(0,2)])))      # 'Bw' (triangle)\n"
        "print('C8.graph6 :', repr(cycle_graph(8).graph6))\n"
        "# self-loops are not representable and are rejected\n"
        "from sparsegf2.errors import InvalidArgumentError\n"
        "try:\n"
        "    graph6_encode(3, [(0,0)])\n"
        "except InvalidArgumentError as e:\n"
        "    print('rejected self-loop:', e)\n"
    ),
    md(r"""## 6. `GraphTopology` and constructors, line by line

```python
@dataclass
class GraphTopology:
    name: str
    n: int
    edges: list[Edge]
    is_stochastic: bool
    one_factorization: list[Matching] | None
    graph6: str
    fresh_matching_sampler: Callable[[Generator], Matching] | None = field(default=None, repr=False)
```

- `edges` are canonical $(u,v)$ with $u<v$, sorted - a stable, comparable
  representation.
- `one_factorization` is `None` exactly when no 1-factorization exists (odd
  $n$); `round_robin`/`palette` matching need it.
- `fresh_matching_sampler` is a **closure** `rng -> matching` drawing a
  *uniformly random* perfect matching; `None` when there is no perfect
  matching. It's `repr=False` (closures don't print usefully).
- Derived properties (`degree_max`, `has_perfect_matching`,
  `has_one_factorization`, `chi_prime`) are computed, not stored.

The constructors guard their domains (`cycle_graph` needs $n\ge3$;
`complete_graph` needs $n\ge2$) and raise `InvalidArgumentError`. `from_spec`
dispatches a name string through `_GRAPH_CONSTRUCTORS`; new families register
there at the documented *extension point* (no stubs)."""),
    code(
        "# fresh sampler draws uniform perfect matchings; here, K_8\n"
        "g = complete_graph(8)\n"
        "rng = np.random.default_rng(0)\n"
        "for _ in range(3):\n"
        "    m = g.fresh_matching_sampler(rng)\n"
        "    assert sorted(q for e in m for q in e) == list(range(8))\n"
        "    print('fresh matching:', m)\n"
        "print('from_spec dispatch:', from_spec('cycle', 8).name, '|', from_spec('Complete', 6).name)\n"
        "try:\n"
        "    from_spec('hypercube', 8)\n"
        "except InvalidArgumentError as e:\n"
        "    print('unknown spec rejected:', e)\n"
    ),
    md(r"""## 7. Topology vs. *simulatability* - the benchmark surprise

Intuition: a nearest-neighbour **cycle** keeps gate supports short, so the
sparse tableau should stay cheap, while a **complete** graph fans supports
out and densifies - slower. We *measured* this
([`benchmark_topology_sparsity.py`](../../benchmarks/circuits/benchmark_topology_sparsity.py))
and the intuition **failed at $O(n)$ depth**: by $\sim\!4n$ layers even the
cycle's light-cone has scrambled the whole system many times, so the
stabilizer tableau *saturates* to near-maximal weight regardless of topology.
What topology *does* change is the **physics** - under measurement,
`complete` sustains far more entanglement (an MIPT-threshold effect). Re-run
the relevant slice live:"""),
    code(
        "import time\n"
        "from sparsegf2 import average_stabilizer_weight, entanglement_entropy\n"
        "from sparsegf2.circuits import CircuitConfig, Picture\n"
        "from sparsegf2.circuits.picture import setup_picture\n"
        "from sparsegf2.circuits.scheduler import CircuitBuilder\n"
        "from sparsegf2.circuits._clifford_table import sp4_table\n"
        "table = sp4_table()\n"
        "setup_picture(Picture.PURE_STATE, 4)[0].apply_gate_2q(0,1,table[5])  # JIT warmup\n"
        "def run(graph, n, p):\n"
        "    cfg = CircuitConfig(graph_spec=graph, n=n, p=p, depth_factor=4)\n"
        "    sim,_ = setup_picture(Picture.PURE_STATE, n, rng=np.random.default_rng([42,1]))\n"
        "    for layer in CircuitBuilder(cfg,0).layers():\n"
        "        for gi,(qi,qj) in enumerate(layer.gate_pairs):\n"
        "            sim.apply_gate_2q(qi,qj,table[int(layer.cliff_indices[gi])%len(table)])\n"
        "        for q in layer.meas_qubits: sim.measure_z(q)\n"
        "    return average_stabilizer_weight(sim), entanglement_entropy(sim, range(n//2))\n"
        "for graph in ('cycle','complete'):\n"
        "    w0,_ = run(graph, 64, 0.0)        # pure scrambling isolates topology\n"
        "    w1,s1 = run(graph, 64, 0.1)       # measured regime shows the physics\n"
        "    print(f'{graph:>9}: avg_weight(p=0)={w0:6.2f}   half_S(p=0.1)={s1}')\n"
    ),
    md(r"""## Summary

- Edges = allowed gate locations; brickwork layers are **perfect matchings**.
- A **1-factorization** (proper $\Delta$-edge-coloring; Vizing Class 1) lets
  brickwork cycle deterministically; even $C_n$ has 2, even $K_n$ has $n-1$.
- Odd $n$ → no perfect matching → no brickwork (config rejects it eagerly).
- Topology is a **physics** knob, not a speed knob, at $O(n)$ depth.

Next: [`matching`](matching.ipynb) - picking *which* 1-factor fires each
layer (round-robin / palette / fresh)."""),
]

if __name__ == "__main__":
    build_and_execute("graphs.ipynb", CELLS)
