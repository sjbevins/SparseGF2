"""Build + execute ``notebooks/circuits/scheduler.ipynb`` - CircuitBuilder,
the single-RNG model, the load-bearing draw order, and the layer stream."""

from __future__ import annotations

from _nbtools import build_and_execute, code, md

CELLS = [
    md(r"""# `sparsegf2.circuits.scheduler` - config + seed → layer stream

[`scheduler.py`](../../src/sparsegf2/circuits/scheduler.py) is the **single
source of truth for a circuit realization**. Given a `CircuitConfig` and a
`sample_seed`, `CircuitBuilder` emits a deterministic stream of
`CircuitLayer` records - gate pairs, their Clifford indices, and the qubits
to measure - *without touching the simulator*. That separation means the
same realization can be replayed on any backend (including a Stim parity
check).

## Contents
1. `CircuitLayer` - the per-layer record
2. The single-RNG model and the seed
3. The **load-bearing draw order**
4. `_place_gates` - brickwork vs random_edge
5. `layers`, `warmup_layers_iter`, `schedule`
6. Determinism & why the draw order can't move"""),
    md(r"""## 1. `CircuitLayer`

```python
@dataclass
class CircuitLayer:
    gate_pairs: list[tuple[int, int]]   # the 2q-gate qubit pairs this layer
    cliff_indices: np.ndarray           # int64 (n_gates,) - index into the Sp(4) table
    meas_qubits: list[int]              # sorted qubits measured this layer
```

`n_gates` and `n_measurements` are convenience properties. A layer is pure
data: *what* to do, not *how*. The runner reads it and calls
`apply_gate_2q` / `measure_z`."""),
    code(
        "import numpy as np\n"
        "from sparsegf2.circuits import CircuitConfig\n"
        "from sparsegf2.circuits.scheduler import CircuitBuilder, CircuitLayer\n"
        "cfg = CircuitConfig(graph_spec='cycle', n=8, depth_factor=2)\n"
        "layer = CircuitBuilder(cfg, sample_seed=0).schedule()[0]\n"
        "print('gate_pairs   :', layer.gate_pairs)\n"
        "print('cliff_indices:', layer.cliff_indices, '(', layer.cliff_indices.dtype, ')')\n"
        "print('meas_qubits  :', layer.meas_qubits)\n"
        "print('n_gates / n_measurements:', layer.n_gates, '/', layer.n_measurements)\n"
    ),
    md(r"""## 2. The single-RNG model and the seed

`CircuitBuilder` seeds **one** `np.random.default_rng(base_seed +
sample_seed)` and consumes it for all construction randomness. The
measurement *outcomes* are a **separate** stream living on the `SparseGF2`
instance (seeded independently by the runner), so the circuit realization
and the measurement coins are decoupled - you can replay the exact same
circuit with different outcomes by changing only the outcome stream. (A
3-way RNG split is a documented option; one stream keeps the mental model
simple. See the [`runner`](runner.ipynb) notebook for the outcome stream.)"""),
    code(
        "b = CircuitBuilder(CircuitConfig(graph_spec='cycle', n=8, depth_factor=1), sample_seed=5)\n"
        "print('builder seed = base_seed + sample_seed =', b.seed, '= 42 + 5')\n"
        "print('graph reused from config:', b.graph.name)\n"
    ),
    md(r"""## 3. The load-bearing draw order

Reproducibility from `(base_seed, sample_seed, layer_index)` requires the
RNG be consumed in a **fixed order each layer**. The contract:

1. **gate placement** - `brickwork`: matching selection (`palette`/`fresh`
   draw; `round_robin` draws nothing); `random_edge`: one edge index.
2. **Clifford indices** - one `rng.integers(0, n_cliffords)` per gate pair.
3. **measurement candidates** - mode-specific draws.

This order is identical in `warmup_layers_iter` and `layers` for steps 1-2.
Changing it (or the matching mode, which changes how many draws step 1
makes) reshuffles every later draw - which is why both are part of the
schema's reproducibility guarantee."""),
    md(r"""## 4. `_place_gates` - brickwork vs random_edge

```python
def _place_gates(self, t, edges, n_edges):
    if gating_mode == "brickwork":
        pairs = select_matching(self.graph, matching_mode, t, self.rng)   # whole matching
    elif gating_mode == "random_edge":
        m = min(cfg.resolved_gates_per_layer(), n_edges)        # m random edges
        idx = self.rng.choice(n_edges, size=m, replace=False)
        pairs = [edges[int(j)] for j in idx]
    cliff_idx = self.rng.integers(0, n_cliffords, size=len(pairs))         # step 2
    return pairs, cliff_idx
```

`brickwork` fires a whole perfect matching ($n/2$ gates); `random_edge`
fires $m = $ `gates_per_layer` distinct random edges (which may share
vertices - *not* a matching). $m=1$ is the single-edge model; $m=n/2$ is the
"O(n) random edges per step" model. Both then draw one Clifford index per
pair, in the fixed order. (Recall from [`config`](config.ipynb) that
`total_layers` scales with $m$, so a single-edge circuit runs $O(n^2)$
layers.)"""),
    code(
        "# brickwork: n/2 gates/layer; random_edge: m gates/layer\n"
        "bw = CircuitBuilder(CircuitConfig(graph_spec='cycle', n=8, gating_mode='brickwork', depth_factor=2), 0).schedule()\n"
        "re1 = CircuitBuilder(CircuitConfig(graph_spec='cycle', n=8, gating_mode='random_edge', gates_per_layer=1, depth_factor=2), 0).schedule()\n"
        "ren = CircuitBuilder(CircuitConfig(graph_spec='cycle', n=8, gating_mode='random_edge', gates_per_layer=4, depth_factor=2), 0).schedule()\n"
        "print('brickwork    gates/layer:', sorted({L.n_gates for L in bw}))   # {4}\n"
        "print('random m=1   gates/layer:', sorted({L.n_gates for L in re1}), '| layers:', len(re1), '(O(n^2))')\n"
        "print('random m=n/2 gates/layer:', sorted({L.n_gates for L in ren}), '| layers:', len(ren), '(matches brickwork)')\n"
        "from sparsegf2.circuits import cycle_graph\n"
        "edges = set(cycle_graph(8).edges)\n"
        "print('random_edge picks graph edges:', all((min(u,v),max(u,v)) in edges for L in ren for u,v in L.gate_pairs))\n"
    ),
    md(r"""## 5. `layers`, `warmup_layers_iter`, `schedule`

- `layers()` yields the $T=$ `total_layers()` measured layers (gates +
  measurements).
- `warmup_layers_iter()` yields `warmup_layers` **gate-only** layers
  (`meas_qubits` always empty) - pre-scrambling. It consumes the RNG
  *before* `layers()`, so the measured phase still begins at the scheduler's
  canonical $t=0$ for the gating pattern.
- `schedule()` is the eager list form of `layers()`.

Warmup matters for the purification picture (scramble the Bell pairs across
the system before measuring); for `pure_state` it just deepens the circuit."""),
    code(
        "cfg = CircuitConfig(graph_spec='cycle', n=8, picture='purification', depth_factor=2, warmup_layers=3)\n"
        "b = CircuitBuilder(cfg, 0)\n"
        "warm = list(b.warmup_layers_iter())\n"
        "print('warmup layers:', len(warm), '| all gate-only:', all(L.n_measurements == 0 for L in warm))\n"
        "print('measured layers:', len(b.schedule()), '= depth_factor*n = 2*8')\n"
    ),
    md(r"""## 6. Determinism

Same `(config, sample_seed)` → byte-identical schedule (gate pairs, Clifford
indices, measured qubits). Different seeds diverge. This is the property the
runner and the Stim-parity tests stand on:"""),
    code(
        "cfg = CircuitConfig(graph_spec='cycle', n=8, depth_factor=2, matching_mode='palette', p=0.3)\n"
        "a = CircuitBuilder(cfg, 7).schedule()\n"
        "b = CircuitBuilder(cfg, 7).schedule()\n"
        "same = all(la.gate_pairs == lb.gate_pairs and np.array_equal(la.cliff_indices, lb.cliff_indices)\n"
        "           and la.meas_qubits == lb.meas_qubits for la, lb in zip(a, b, strict=True))\n"
        "print('seed 7 == seed 7 :', same)\n"
        "c = CircuitBuilder(cfg, 8).schedule()\n"
        "print('seed 7 != seed 8 :', any(not np.array_equal(la.cliff_indices, lc.cliff_indices)\n"
        "                                 for la, lc in zip(a, c, strict=True)))\n"
    ),
    md(r"""## Summary

- `CircuitBuilder` is the deterministic, simulator-free source of a
  realization: a stream of `CircuitLayer` records.
- One construction RNG, consumed in a **fixed draw order** (placement →
  clifford → measurement) that is load-bearing for reproducibility.
- `brickwork` (a matching) vs `random_edge` (one edge); warmup prepends
  gate-only layers; `schedule()` is the eager form.

Next: [`runner`](runner.ipynb) - executing the layer stream on `SparseGF2`
and reading off the order parameter."""),
]

if __name__ == "__main__":
    build_and_execute("scheduler.ipynb", CELLS)
