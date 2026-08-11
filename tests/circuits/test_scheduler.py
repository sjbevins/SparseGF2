"""Tests for the CircuitBuilder schedule generation."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.circuits import CircuitConfig, cycle_graph, simulate
from sparsegf2.circuits.scheduler import CircuitBuilder, CircuitLayer
from sparsegf2.errors import InvalidArgumentError


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


@pytest.mark.parametrize("sample_seed", [True, 1.5, np.float64(3), "3", -1])
def test_sample_seed_rejects_non_integer_or_negative_values(sample_seed):
    cfg = CircuitConfig(graph_spec="cycle", n=8)
    with pytest.raises(InvalidArgumentError, match="sample_seed"):
        CircuitBuilder(cfg, sample_seed=sample_seed)
    with pytest.raises(InvalidArgumentError, match="sample_seed"):
        simulate(cfg, sample_seed=sample_seed)


def test_sample_seed_accepts_numpy_integer_without_changing_schedule():
    cfg = CircuitConfig(graph_spec="cycle", n=8, depth_factor=2)
    numpy_seed = CircuitBuilder(cfg, sample_seed=np.int64(7))
    python_seed = CircuitBuilder(cfg, sample_seed=7)
    assert numpy_seed.sample_seed == 7
    for left, right in zip(numpy_seed.layers(), python_seed.layers(), strict=True):
        assert left.gate_pairs == right.gate_pairs
        assert np.array_equal(left.cliff_indices, right.cliff_indices)
        assert left.meas_qubits == right.meas_qubits


def test_random_edge_one_gate_per_layer():
    cfg = CircuitConfig(graph_spec="cycle", n=8, gating_mode="random_edge", depth_factor=3)
    sched = CircuitBuilder(cfg, sample_seed=0).schedule()
    assert all(layer.n_gates == 1 for layer in sched)
    # the chosen edge is always a graph edge
    edges = set(cycle_graph(8).edges)
    for layer in sched:
        u, v = layer.gate_pairs[0]
        assert (min(u, v), max(u, v)) in edges


def test_all_edges_fires_fixed_edge_order_without_placement_rng():
    cfg = CircuitConfig(
        graph_spec="cycle",
        n=8,
        gating_mode="all_edges",
        p=0.5,
        total_layers_override=1,
        base_seed=13,
    )
    layer = next(CircuitBuilder(cfg, sample_seed=7).layers())
    assert layer.gate_pairs == list(cycle_graph(8).edges)

    # all_edges spends no random draw on placement: the first stream draw is
    # exactly the |E| Clifford-index vector, followed by Bernoulli measurements.
    rng = np.random.default_rng([13, 7])
    expected_cliffords = rng.integers(
        0, cfg.n_cliffords, size=len(cfg._graph.edges), dtype=np.int64
    )
    expected_measured = np.nonzero(rng.random(cfg.n) < cfg.p)[0].tolist()
    assert np.array_equal(layer.cliff_indices, expected_cliffords)
    assert layer.meas_qubits == expected_measured


def test_all_edges_warmup_also_fires_every_edge():
    cfg = CircuitConfig(graph_spec="path", n=6, gating_mode="all_edges", warmup_layers=2)
    warmup = list(CircuitBuilder(cfg, sample_seed=1).warmup_layers_iter())
    assert len(warmup) == 2
    assert all(layer.gate_pairs == list(cfg._graph.edges) for layer in warmup)
    assert all(layer.meas_qubits == [] for layer in warmup)


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
