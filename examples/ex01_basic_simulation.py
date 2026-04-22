#!/usr/bin/env python3
"""
Example 1: Basic SparseGF2 simulation — gates, measurements, observables.

Demonstrates:
- Initializing the simulator (|0>^n state with destabilizers + stabilizers)
- Applying 1-qubit and 2-qubit Clifford gates
- Z-measurement with reset
- Observable extraction (k, abar, entropy, TMI, bandwidth)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sparsegf2.core.sparse_tableau import SparseGF2, warmup

# JIT warmup (one-time cost, ~2s)
warmup()

# ── Initialize a 6-qubit simulator ──────────────────────────
n = 6
sim = SparseGF2(n)

print(f"Initial state: |0>^{n}")
print(f"  k = {sim.compute_k()}, abar = {sim.get_active_count():.2f}")
print(f"  S({{q0}}) = {sim.compute_subsystem_entropy([0])}")
print(f"  TMI = {sim.compute_tmi():.1f}")
print()

# ── Apply gates ─────────────────────────────────────────────
# CNOT(0,1): X_0 -> X_0 X_1, Z_1 -> Z_0 Z_1
CNOT = np.array([[1,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,1]], dtype=np.uint8)
sim.apply_gate(0, 1, CNOT)
print("After CNOT(0,1):")
print(f"  k = {sim.compute_k()}, abar = {sim.get_active_count():.2f}")

# Hadamard on qubit 2 (single-qubit gate)
sim.apply_h(2)
print("After H(2):")
print(f"  k = {sim.compute_k()}, abar = {sim.get_active_count():.2f}")

# S gate on qubit 3
sim.apply_s(3)
print("After S(3):")
print(f"  k = {sim.compute_k()}, abar = {sim.get_active_count():.2f}")
print()

# ── Measure qubit 0 ────────────────────────────────────────
print("Measuring Z on qubit 0...")
sim.apply_measurement_z(0)
print(f"  k = {sim.compute_k()} (decreased by 1)")
print(f"  abar = {sim.get_active_count():.2f}")
print()

# ── Measure all qubits ─────────────────────────────────────
print("Measuring all remaining qubits:")
for q in range(1, n):
    k_before = sim.compute_k()
    sim.apply_measurement_z(q)
    k_after = sim.compute_k()
    delta = k_after - k_before
    print(f"  Measure q{q}: k {k_before} -> {k_after} (delta={delta})")
print(f"\nFinal k = {sim.compute_k()} (all information destroyed)")
