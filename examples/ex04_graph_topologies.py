#!/usr/bin/env python3
"""
Example 4: Comparing graph topologies and matching modes.

Demonstrates how the two MVP graph topologies (cycle, complete) and the
three matching modes (round_robin, palette, fresh) differ in their
measurement-induced phase behaviour.

The emergent p_c is topology-dependent: the cycle has p_c ~ 0.16 in the
thermodynamic limit, while the all-to-all complete graph transitions much
later because every gate layer mixes the whole system.
"""
import time
import numpy as np

from sparsegf2 import warmup
from sparsegf2.circuits import CircuitConfig
from sparsegf2.circuits.runner import SimulationRunner, get_clifford_table

warmup()
cliff_table = get_clifford_table()

n = 32
n_samples = 10
configs = [
    # graph,      matching,       depth,       factor, p
    ("cycle",     "round_robin",  "O(n)",      8, 0.15),
    ("cycle",     "fresh",        "O(n)",      8, 0.15),
    ("complete",  "round_robin",  "O(log_n)",  8, 0.15),
    ("complete",  "palette",      "O(log_n)",  8, 0.15),
    ("complete",  "fresh",        "O(log_n)",  8, 0.15),
]

print(f"Topology / matching-mode comparison: n={n}, {n_samples} samples/row\n")
print(f"  {'graph':<10}  {'matching':<13}  {'depth':<10}  "
      f"{'<k>':>6}  {'P(k>0)':>8}  {'<abar>':>8}  {'time':>8}")
print(f"  {'-'*10}  {'-'*13}  {'-'*10}  "
      f"{'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")

for graph, matching, depth_mode, depth_factor, p in configs:
    cfg = CircuitConfig(
        graph_spec=graph, n=n,
        matching_mode=matching,
        measurement_mode="uniform",
        depth_mode=depth_mode, depth_factor=depth_factor,
        p=p, base_seed=42,
    )
    runner = SimulationRunner(cfg, clifford_table=cliff_table, warmup_jit=False)

    ks, abars = [], []
    t0 = time.perf_counter()
    for s in range(n_samples):
        rec = runner.run(sample_seed=s)
        ks.append(rec.k)
        abars.append(rec.final_abar)
    dt = time.perf_counter() - t0

    print(f"  {graph:<10}  {matching:<13}  {depth_mode:<10}  "
          f"{np.mean(ks):>6.1f}  {np.mean([int(k > 0) for k in ks]):>8.2f}  "
          f"{np.mean(abars):>8.1f}  {dt:>7.2f}s")

print("\nHigher-connectivity topologies survive larger measurement rates before the")
print("code is destroyed. The three matching modes are equivalent on the cycle")
print("(only two perfect matchings exist), but genuinely distinct on K_n.")
