#!/usr/bin/env python3
"""
Example 3: Measurement-induced phase transition (MIPT) sweep.

Runs nearest-neighbor brickwork circuits on the cycle graph across
measurement rates p = 0.05..0.30 to observe the transition in the code
dimension k and the active count abar.

Uses the ``sparsegf2.circuits`` subpackage to build the circuit schedule
and drive the SparseGF2 simulator one sample at a time.
"""
import time
import numpy as np

from sparsegf2 import warmup
from sparsegf2.circuits import CircuitConfig
from sparsegf2.circuits.runner import SimulationRunner, get_clifford_table

warmup()
cliff_table = get_clifford_table()  # 11,520-element two-qubit Clifford group

n = 64
n_samples = 20
p_values = [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30]

print(f"MIPT sweep: n={n}, {n_samples} samples/point, NN brickwork, depth=8n={8*n}")
print(f"Full Clifford group: {len(cliff_table)} elements\n")
print(f"  {'p':>6}  {'<k>':>8}  {'P(k>0)':>8}  {'<abar>':>8}  {'time/sample':>12}")
print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")

for p in p_values:
    cfg = CircuitConfig(
        graph_spec="cycle", n=n,
        matching_mode="round_robin",
        measurement_mode="uniform",
        depth_mode="O(n)", depth_factor=8,
        p=p, base_seed=42,
    )
    runner = SimulationRunner(cfg, clifford_table=cliff_table, warmup_jit=False)

    ks, abars = [], []
    t0 = time.perf_counter()
    for s in range(n_samples):
        rec = runner.run(sample_seed=s)
        ks.append(rec.k)
        abars.append(rec.final_abar)
    dt = (time.perf_counter() - t0) / n_samples

    print(f"  {p:>6.2f}  {np.mean(ks):>8.1f}  "
          f"{np.mean([int(k > 0) for k in ks]):>8.2f}  "
          f"{np.mean(abars):>8.1f}  {dt:>10.4f}s")

print(f"\nPhase transition: k drops to 0 and abar to O(1) near p_c ~ 0.16 (cycle).")
