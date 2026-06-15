"""Benchmark: does graph topology affect *simulatability*? (Honest answer: barely.)

A natural optimization intuition: SparseGF2 stores the stabilizer tableau
sparsely and a two-qubit gate only touches generators supported on its two
qubits, so a nearest-neighbour ``cycle`` (short supports) should simulate
faster than an all-to-all ``complete`` graph (supports fan out). This
script tests that intuition, and finds it **does not hold at O(n)
depth**.

For each (n, graph, p) it runs a fixed-depth brickwork circuit and reports

- wall-clock simulation time (gates + measurements),
- average stabilizer weight (mean non-identity Paulis per generator, a
  direct sparsity / tableau-density proxy), and
- half-cut entanglement entropy (the physics).

Two measurement regimes are compared:

- ``p = 0.0``: pure scrambling, isolating the topology's effect on the
  tableau with no measurement collapse.
- ``p = 0.1``: the MIPT-relevant regime.

Result (see the printed reading): at ``~4n`` depth the light-cone of even
the cycle has scrambled the whole system many times, so the tableau
*saturates* to near-maximal weight regardless of topology: cycle and
complete are within noise on both weight and time. Topology's real effect
is on the **entanglement** (complete sustains more under measurement),
which is physics, not simulation cost.

Run::

    .venv/bin/python benchmarks/circuits/benchmark_topology_sparsity.py
"""

from __future__ import annotations

import time

import numpy as np

from sparsegf2 import average_stabilizer_weight, entanglement_entropy
from sparsegf2.circuits import CircuitConfig, Picture
from sparsegf2.circuits._clifford_table import sp4_table
from sparsegf2.circuits.picture import setup_picture
from sparsegf2.circuits.scheduler import CircuitBuilder

# Distinct from the runner's measurement-outcome key; this benchmark just
# needs a fixed, reproducible outcome stream.
_BENCH_RNG_KEY = 1


def _warmup() -> None:
    """Trigger Numba JIT once so it doesn't pollute the first timing."""
    sim, _ = setup_picture(Picture.PURE_STATE, 4)
    sim.apply_gate_2q(0, 1, sp4_table()[5])
    sim.measure_z(0)


def bench(n: int, graph: str, p: float, *, depth_factor: int = 4, seed: int = 0):
    """Run one cell to completion; return (seconds, avg_weight, half_cut_S)."""
    cfg = CircuitConfig(
        graph_spec=graph,
        n=n,
        picture=Picture.PURE_STATE,
        gating_mode="brickwork",
        matching_mode="round_robin",
        measurement_mode="bernoulli",
        p=p,
        depth_factor=depth_factor,
    )
    sim, _ = setup_picture(
        Picture.PURE_STATE, n, rng=np.random.default_rng([cfg.base_seed + seed, _BENCH_RNG_KEY])
    )
    table = sp4_table()
    n_table = len(table)

    t0 = time.perf_counter()
    for layer in CircuitBuilder(cfg, sample_seed=seed).layers():
        for g, (qi, qj) in enumerate(layer.gate_pairs):
            sim.apply_gate_2q(qi, qj, table[int(layer.cliff_indices[g]) % n_table])
        for q in layer.meas_qubits:
            sim.measure_z(q)
    elapsed = time.perf_counter() - t0

    avg_w = average_stabilizer_weight(sim)
    half_s = entanglement_entropy(sim, range(n // 2))
    return elapsed, avg_w, half_s


def main() -> None:
    _warmup()
    sizes = [32, 64, 128]
    graphs = ["cycle", "complete"]
    ps = [0.0, 0.1]
    print(f"{'n':>5} {'graph':>9} {'p':>5} {'time_s':>9} {'avg_weight':>11} {'half_S':>7}")
    print("-" * 52)
    for n in sizes:
        for p in ps:
            for graph in graphs:
                elapsed, avg_w, half_s = bench(n, graph, p)
                print(f"{n:>5} {graph:>9} {p:>5.2f} {elapsed:>9.4f} {avg_w:>11.3f} {half_s:>7}")
    print(
        "\nReading:\n"
        "  * avg_weight & time are within noise between cycle and complete at\n"
        "    every (n, p): at ~4n depth the tableau saturates regardless of\n"
        "    topology, so locality buys ~no simulation speed here.\n"
        "  * half_S diverges under measurement (complete >> cycle): topology\n"
        "    changes the PHYSICS (MIPT threshold), not the sim cost.\n"
        "  Takeaway: choose the graph for the dynamics you want, not for speed."
    )


if __name__ == "__main__":
    main()
