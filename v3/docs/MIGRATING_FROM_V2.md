# Moving from SparseGF2 v2 to v3

v2 (`sparsegf2`) and v3 (`sparsegf2_v3`) install side by side, so you can move
one script at a time and cross-check as you go. This page maps v2 concepts to
v3, shows the production protocols in v3 terms, and says what changes for
reproducibility.

## Concepts

| v2 | v3 | notes |
|---|---|---|
| `SparseGF2(n, rng=...)` | `Simulator(n, phases=False, seed=...)` | v3 defaults to `phases=True` (signs tracked, outcomes exact); pass `phases=False` for v2's sign-free state |
| `sim.measure_z(q)` | `sim.measure(q)` | phase-free `measure()` returns `None`: there is no outcome without signs (v2 returned a coin flip). `RunResult.random` says which measurements were random in both modes |
| `_dispatch_2q_batch(pairs, idx, luts)` | `Circuit().append("CLIFFORD_2Q", pairs, indices)` then `sim.run(circuit)`, or `sim.clifford_2q(index, a, b)` | v3 indices are canonical two-qubit Clifford indices in `[0, 11520)`: symplectic part `index // 16` (position in `clifford.symplectic_group(2)`), signs `index % 16`. v2's Sp(4) table holds the same 720 matrices in a different order; see [Converting v2 gate indices](#converting-v2-gate-indices) |
| `_build_2q_lut(S)` | not needed | gates are compiled tables inside the kernel |
| `entanglement_entropy(sim, region)` | `sim.entropy(region)` | plus `observables.mutual_information`, `tripartite_information`, `entropy_profile`, ... |
| `sim.canonical_form()` | `sim.stabilizers()` (Stim's canonical form) or `sim.stabilizer_arrays(canonical=True)` | |
| `CircuitConfig` + `simulate(config, sample_seed)` | `mipt.MonitoredCircuit(...)` + `.run(trajectories, seed=...)` | every trajectory runs inside one compiled, parallel kernel; `spec.circuit(t, seed)` returns the exact circuit of trajectory `t` |
| pictures `pure_state` / `purification` / `single_ref` | `reference="none"` / `"purification"` / `"single"` | the single reference's partner is configurable: `reference_partner=n - 1` reproduces v2, `"random"` draws it per trajectory |
| gating `brickwork` / `round_robin` | `schedule="matching"` | an edge colouring of any graph; on an even ring it is the brickwork |
| gating `random_edge` | `schedule="random_edges"`, `gates_per_layer=...` (default `n // 2`) | **no depth rescaling**: a layer is always one round of gates plus one round of measurements (v2 rescaled the depth by n/(2m) in random gating modes) |
| warmup layers | `scramble_layers` (gate-only layers of the dynamics), `scramble_gates` (random pairs), `scramble_matchings` (random all-to-all perfect matchings) | applied in the order gates, matchings, layers |
| `base_seed` (graph and circuit together) | graph seeds in `graphs.*(seed=...)`, circuit seed in `run(seed=...)` | graph disorder and circuit disorder are independent axes |
| one graph per trajectory (production engines) | `edges=[g_0, g_1, ...]` | trajectory `t` runs on graph `t mod len(edges)` |
| `Study`, `sweep` | a loop over parameters, `EnsembleResult.save` / `load` / `concatenate` | or the agent interface's background jobs (`sparsegf2_v3.api.start_ensemble_job`), which save chunks and resume |
| `expurgation` | not ported yet | on the roadmap (DESIGN.md §14) |
| quantikz circuit drawing | `circuit.to_stim().diagram(...)` | Stim's diagrams |

## The production protocols

**Single reference, one graph per trajectory, matching scramble**
(`studies/prl_production/PROTOCOL.md`):
[`examples/purification_time.py`](../examples/purification_time.py) is the
whole thing; `production_spec(n, beta, p, graph_bank)` is the spec.
[`benchmarks/crosscheck_v2_production.py`](../benchmarks/crosscheck_v2_production.py)
runs it next to the v2 production engine on the same graph bank and checks
that the purification-time distributions agree.

**The generalized raw-tau protocol** (`single_ref/raw_tau/PROTOCOL.md`:
`N_circuits` circuits per graph, `q_scramble * n` random-pair scramble gates,
`T_max = q_max * n`, reference at `n - 1` or uniformly random):

```python
from sparsegf2_v3 import graphs, mipt

def raw_tau_spec(n, beta, p, graph_bank, q_scramble, q_max, random_reference):
    return mipt.MonitoredCircuit(
        num_system=n,
        edges=graph_bank,                       # G graphs: trajectory t uses graph t mod G
        layers=q_max * n,
        measurement_rate=p,
        schedule="random_edges",                # n // 2 edges per layer, with replacement
        reference="single",
        reference_partner="random" if random_reference else n - 1,
        scramble_gates=q_scramble * n,          # gates on uniformly random distinct pairs
        checkpoints="every_layer",
        stop_when_purified=True,
    )

bank = [graphs.watts_strogatz(64, 4, 0.1, seed=(2024, g)) for g in range(500)]
spec = raw_tau_spec(64, 0.1, 0.28, bank, q_scramble=2, q_max=8, random_reference=True)
result = spec.run(trajectories=len(bank) * 10, seed=7)   # 10 circuits per graph
graph_index = result.trajectories % len(bank)            # which graph each trajectory used
```

## Converting v2 gate indices

To replay a v2 gate sequence exactly, map each v2 Sp(4) index to the v3 index
of the same symplectic matrix (sign bits 0; signs do not matter phase-free):

```python
import numpy as np
from sparsegf2.circuits._clifford_table import sp4_table      # v2
from sparsegf2_v3.clifford import symplectic_group           # v3

v3_position = {m.tobytes(): i for i, m in enumerate(symplectic_group(2))}
V2_TO_V3 = np.array([16 * v3_position[np.asarray(m, np.uint8).tobytes()] for m in sp4_table()])
# v2 gate index g  ->  CLIFFORD_2Q(V2_TO_V3[g]) in v3
```

Both versions use the same matrix convention (rows are the images of
X_a, X_b, Z_a, Z_b in the basis x_a, x_b, z_a, z_b), and for uniformly random
gates the order does not matter at all.

## Reproducibility

* v3's random streams differ from v2's, so v3 reproduces v2's **statistics**,
  not its individual trajectories. Keep each `(n, beta, p)` point within one
  engine, and record the engine (`sparsegf2_v3.__version__` and the spec,
  both stored by `EnsembleResult.save`) with the data.
* In v3, trajectory `t` of `run(seed=s)` depends only on `(spec, s, t)`: it
  does not depend on the thread count or on how the run is split into chunks
  (`first_trajectory=`).
* **Common random numbers across p come for free.** The gate draws do not
  depend on `p`, and the measurement mask uses one uniform draw per qubit per
  layer whatever `p` is. So runs at different `p` with the same seed share
  every random number, which is v2's reserved `common_circuit_disorder`
  policy (lower-variance differences across p). For independent streams per
  `p`, give each `p` its own seed, e.g. `seed=(base, p_index)`.
* Graphs: `graphs.watts_strogatz(n, k, beta, seed=...)` follows the networkx
  convention (`k` = total degree, so C(n, 2) is `k=4`) with SparseGF2's own
  generator, independent of NumPy and networkx versions. Its rewiring rule is
  v2's (each lattice edge, in the classic order, moves its far endpoint to a
  uniformly random current non-neighbour with probability `beta`), so it
  samples the same ensemble of graphs, though not the same graphs. To reuse a
  v2 graph bank exactly, pass its edge arrays as `edges=[...]`.

## Performance

On this protocol v3 runs 27–60× more trajectories per second than the v2
production engine on one thread, and all cores are used by default (see the
[README](../README.md#performance)). Rough single-thread budgets for planning:
`api.estimate_ensemble(spec_dict, trajectories)` gives a deliberately
pessimistic estimate, and a short run (`spec.run(200, seed=0)`) reports
`trajectories_per_second` directly.
