"""
Tests for ``gating_mode='random_edge'`` and the new measurement modes
``gated`` and ``random_pair``.

- Random-edge gate placement: every layer applies exactly one gate on
  an edge drawn uniformly at random from the graph's edge list.
- Gated measurement: candidate qubits = qubits touched by this layer's
  gates; each is measured with probability ``p``.
- Random-pair measurement: two distinct qubits are drawn uniformly at
  random from the n-qubit system each layer; each is then measured
  with probability ``p``.

The long-run distribution properties (uniformity over edges, candidate
isolation) are tested statistically.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from sparsegf2.circuits import CircuitConfig, SimulationRunner
from sparsegf2.circuits.builder import CircuitBuilder


def test_random_edge_emits_one_gate_per_layer():
    cfg = CircuitConfig(
        graph_spec="complete", n=8, picture="single_ref",
        gating_mode="random_edge", measurement_mode="gated",
        p=0.3, depth_factor=4,
    )
    builder = CircuitBuilder(cfg, sample_seed=0)
    for layer in builder.layers():
        assert len(layer.gate_pairs) == 1, (
            f"random_edge must emit exactly 1 gate per layer; got "
            f"{len(layer.gate_pairs)}: {layer.gate_pairs}"
        )


def test_random_edge_uniformly_covers_k_n_edges():
    """Over many layers on K_n the marginal distribution of edges
    should be close to uniform across all n(n-1)/2 edges."""
    n = 8
    n_edges = n * (n - 1) // 2
    n_layers = 20000
    cfg = CircuitConfig(
        graph_spec="complete", n=n, picture="single_ref",
        gating_mode="random_edge", measurement_mode="gated",
        p=0.0, depth_factor=n_layers,  # => total_layers ~ n_layers
    )
    builder = CircuitBuilder(cfg, sample_seed=1234)
    counts = Counter()
    seen_layers = 0
    for layer in builder.layers():
        seen_layers += 1
        if seen_layers > n_layers:
            break
        (u, v) = layer.gate_pairs[0]
        key = (min(u, v), max(u, v))
        counts[key] += 1
    # Every one of the n(n-1)/2 edges should appear, and counts should
    # be within ~5 sigma of uniform.
    assert len(counts) == n_edges, (
        f"not all {n_edges} K_{n} edges were hit over {seen_layers} layers "
        f"(saw {len(counts)})"
    )
    expected = seen_layers / n_edges
    sigma = np.sqrt(expected * (1 - 1 / n_edges))
    for edge, c in counts.items():
        z = abs(c - expected) / sigma
        assert z < 5.0, (
            f"edge {edge}: observed {c}, expected {expected:.1f} "
            f"(z={z:.2f} sigma) — RNG distribution non-uniform"
        )


def test_gated_measurement_candidates_are_gated_qubits_only():
    """In gated mode, every measured qubit must be in the gate pair."""
    cfg = CircuitConfig(
        graph_spec="complete", n=16, picture="single_ref",
        gating_mode="random_edge", measurement_mode="gated",
        p=1.0, depth_factor=8,
    )
    builder = CircuitBuilder(cfg, sample_seed=7)
    for layer in builder.layers():
        gated = {q for pair in layer.gate_pairs for q in pair}
        # At p=1 every candidate is measured
        for q in layer.meas_qubits:
            assert q in gated, (
                f"gated mode leaked a non-gated qubit: gate={layer.gate_pairs}, "
                f"meas={layer.meas_qubits}, offending q={q}"
            )
        # And because p=1 all candidates ARE measured
        assert set(layer.meas_qubits) == gated, (
            f"gated+p=1 must measure all gated qubits: gated={gated}, "
            f"meas={set(layer.meas_qubits)}"
        )


def test_random_pair_measures_at_most_two_qubits_from_system_only():
    cfg = CircuitConfig(
        graph_spec="complete", n=16, picture="single_ref",
        gating_mode="random_edge", measurement_mode="random_pair",
        p=0.5, depth_factor=8,
    )
    builder = CircuitBuilder(cfg, sample_seed=11)
    for layer in builder.layers():
        assert len(layer.meas_qubits) <= 2
        for q in layer.meas_qubits:
            assert 0 <= q < 16
            assert q != 16  # never the reference qubit


@pytest.mark.parametrize("meas_mode", ["gated", "random_pair"])
def test_random_edge_with_until_purified_runs_end_to_end(meas_mode):
    """The new (gating, measurement) combos integrate cleanly with
    picture=single_ref + depth_mode=until_purified."""
    cfg = CircuitConfig(
        graph_spec="complete", n=16, picture="single_ref",
        gating_mode="random_edge", measurement_mode=meas_mode,
        p=0.5, depth_factor=32,
        depth_mode="until_purified", record_time_series=True,
    )
    runner = SimulationRunner(cfg, warmup_jit=False)
    records = [runner.run(sample_seed=s) for s in range(6)]
    for rec in records:
        assert rec.k in (0, 1)
        assert rec.ref_entropy_timeseries.shape == (cfg.total_layers() + 1,)
        assert int(rec.ref_entropy_timeseries[0]) == 1
        assert int(rec.ref_entropy_timeseries[-1]) == 0
