"""Tests for the CircuitBuilder schedule generation."""

from __future__ import annotations

import numpy as np

from sparsegf2.circuits import CircuitConfig, cycle_graph
from sparsegf2.circuits.scheduler import CircuitBuilder, CircuitLayer


def test_brickwork_layer_count_and_gate_count():
    cfg = CircuitConfig(graph_spec="cycle", n=8, depth_factor=4)  # 32 layers
    b = CircuitBuilder(cfg, sample_seed=0)
    sched = b.schedule()
    assert len(sched) == 32
    assert all(layer.n_gates == 4 for layer in sched)  # full matching = n/2
    assert all(layer.cliff_indices.shape == (4,) for layer in sched)


def test_determinism_same_seed_identical_schedule():
    cfg = CircuitConfig(graph_spec="cycle", n=8, depth_factor=2, matching_mode="palette")
    a = CircuitBuilder(cfg, sample_seed=7).schedule()
    b = CircuitBuilder(cfg, sample_seed=7).schedule()
    for la, lb in zip(a, b, strict=True):
        assert la.gate_pairs == lb.gate_pairs
        assert np.array_equal(la.cliff_indices, lb.cliff_indices)
        assert la.meas_qubits == lb.meas_qubits


def test_different_seed_changes_schedule():
    cfg = CircuitConfig(graph_spec="cycle", n=8, depth_factor=2, p=0.5)
    a = CircuitBuilder(cfg, sample_seed=0).schedule()
    b = CircuitBuilder(cfg, sample_seed=1).schedule()
    # at least the clifford indices should differ somewhere
    assert any(
        not np.array_equal(la.cliff_indices, lb.cliff_indices) for la, lb in zip(a, b, strict=True)
    )


def test_random_edge_one_gate_per_layer():
    cfg = CircuitConfig(graph_spec="cycle", n=8, gating_mode="random_edge", depth_factor=3)
    sched = CircuitBuilder(cfg, sample_seed=0).schedule()
    assert all(layer.n_gates == 1 for layer in sched)
    # the chosen edge is always a graph edge
    edges = set(cycle_graph(8).edges)
    for layer in sched:
        u, v = layer.gate_pairs[0]
        assert (min(u, v), max(u, v)) in edges


def test_warmup_layers_are_gate_only():
    cfg = CircuitConfig(
        graph_spec="cycle", n=8, depth_factor=2, picture="purification", warmup_layers=5
    )
    b = CircuitBuilder(cfg, sample_seed=0)
    warm = list(b.warmup_layers_iter())
    assert len(warm) == 5
    assert all(layer.n_measurements == 0 for layer in warm)
    assert all(isinstance(layer, CircuitLayer) for layer in warm)


def test_prebuilt_graph_passthrough():
    g = cycle_graph(8)
    cfg = CircuitConfig(graph_spec=g, n=8, depth_factor=1)
    b = CircuitBuilder(cfg, sample_seed=0)
    assert b.graph is g
