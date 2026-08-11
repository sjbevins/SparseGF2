"""Tests for the single_ref picture, gates_per_layer, and gating-aware depth."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2 import entanglement_entropy
from sparsegf2.circuits import CircuitConfig, Picture, simulate
from sparsegf2.circuits.picture import setup_picture
from sparsegf2.circuits.scheduler import CircuitBuilder
from sparsegf2.errors import InvalidArgumentError

# ----------------------------------------------------------------------
# single_ref picture
# ----------------------------------------------------------------------


def test_single_ref_layout_and_fresh_entropy():
    sim, spec = setup_picture(Picture.SINGLE_REF, 8)
    assert spec.picture is Picture.SINGLE_REF
    assert spec.total_qubits == 9  # n + 1
    assert spec.system_qubits.tolist() == list(range(8))
    assert spec.reference_qubits.tolist() == [8]
    assert spec.order_parameter == "ref_entropy"
    # reference (qubit 8) is Bell-paired with system qubit 7 -> S(ref) = 1
    assert entanglement_entropy(sim, [8]) == 1
    assert sim.n == 9


def test_single_ref_config_total_qubits():
    assert CircuitConfig(graph_spec="cycle", n=8, picture="single_ref").total_qubits() == 9


def test_single_ref_run_records_ref_entropy():
    cfg = CircuitConfig(graph_spec="cycle", n=8, picture="single_ref", p=0.16, depth_factor=4)
    rec = simulate(cfg, sample_seed=1)
    assert rec.picture is Picture.SINGLE_REF
    assert rec.ref_entropy in (0, 1)  # single tracked bit
    assert rec.code_dimension is None  # not the purification order parameter
    assert rec.entropy_half_cut is not None  # half-cut always recorded
    # reproducible
    assert simulate(cfg, sample_seed=1).ref_entropy == rec.ref_entropy


def test_single_ref_purifies_under_heavy_measurement():
    # Deep, strongly-monitored circuit -> the reference should purify
    # (ref_entropy 0) for at least some trajectories.
    cfg = CircuitConfig(graph_spec="cycle", n=8, picture="single_ref", p=0.7, depth_factor=8)
    outs = [simulate(cfg, sample_seed=s).ref_entropy for s in range(20)]
    assert all(o in (0, 1) for o in outs)
    assert min(outs) == 0  # at least one trajectory purifies


def test_single_ref_reference_qubit_is_never_measured():
    # The runner measures only system qubits [0, n); the reference (index n)
    # must never be a measurement candidate.
    cfg = CircuitConfig(
        graph_spec="cycle",
        n=8,
        picture="single_ref",
        measurement_mode="bernoulli",
        p=1.0,
        depth_factor=2,
    )
    for layer in CircuitBuilder(cfg, 0).schedule():
        assert all(q < 8 for q in layer.meas_qubits)


# ----------------------------------------------------------------------
# gates_per_layer (random_edge)
# ----------------------------------------------------------------------


def test_gates_per_layer_int_fires_m_edges():
    cfg = CircuitConfig(
        graph_spec="cycle", n=8, gating_mode="random_edge", gates_per_layer=3, depth_factor=1
    )
    assert cfg.resolved_gates_per_layer() == 3
    for layer in CircuitBuilder(cfg, 0).schedule():
        assert layer.n_gates == 3


def test_gates_per_layer_callable_scales_with_n():
    cfg = CircuitConfig(
        graph_spec="cycle",
        n=16,
        gating_mode="random_edge",
        gates_per_layer=lambda c: c.n // 2,
        depth_factor=1,
    )
    assert cfg.resolved_gates_per_layer() == 8  # n/2
    for layer in CircuitBuilder(cfg, 0).schedule():
        assert layer.n_gates == 8


def test_gates_per_layer_exceeding_edges_rejected():
    # cycle(8) has 8 edges; random_edge fires *distinct* edges, so asking for
    # more than |E| cannot be delivered. It is refused at construction (a silent
    # clamp would make total_layers()/the expected ratio disagree with reality).
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(
            graph_spec="cycle", n=8, gating_mode="random_edge", gates_per_layer=100, depth_factor=1
        )
    # exactly |E| is fine (every distinct edge fires each layer).
    cfg = CircuitConfig(
        graph_spec="cycle", n=8, gating_mode="random_edge", gates_per_layer=8, depth_factor=1
    )
    for layer in CircuitBuilder(cfg, 0).schedule():
        assert layer.n_gates == 8


@pytest.mark.parametrize("gpl", [5, lambda c: c.n])
def test_gates_per_layer_rejected_for_brickwork(gpl):
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(graph_spec="cycle", n=8, gating_mode="brickwork", gates_per_layer=gpl)


def test_gates_per_layer_invalid_values():
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(graph_spec="cycle", n=8, gating_mode="random_edge", gates_per_layer=0)
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(graph_spec="cycle", n=8, gating_mode="random_edge", gates_per_layer="lots")


def test_to_dict_resolves_callable_gates_per_layer():
    cfg = CircuitConfig(
        graph_spec="cycle", n=16, gating_mode="random_edge", gates_per_layer=lambda c: c.n // 2
    )
    d = cfg.to_dict()
    assert d["gates_per_layer"] == 8
    assert isinstance(d["gates_per_layer"], int)


# ----------------------------------------------------------------------
# gating-aware depth (the random_edge fix)
# ----------------------------------------------------------------------


def _gate_budget(cfg):
    """Total two-qubit gates a full run applies for this cell."""
    eg = cfg.n // 2 if cfg.gating_mode == "brickwork" else cfg.resolved_gates_per_layer()
    return cfg.total_layers() * eg


def test_single_edge_random_edge_needs_quadratic_depth():
    n, df = 8, 2
    cb = CircuitConfig(graph_spec="cycle", n=n, gating_mode="brickwork", depth_factor=df)
    c1 = CircuitConfig(
        graph_spec="cycle", n=n, gating_mode="random_edge", gates_per_layer=1, depth_factor=df
    )
    assert cb.total_layers() == df * n  # 16, O(n)
    assert c1.total_layers() == round(df * n * n / 2)  # 64, O(n^2)
    assert c1.total_layers() > cb.total_layers()


def test_on_random_edges_matches_brickwork_layer_count():
    n, df = 8, 2
    cb = CircuitConfig(graph_spec="cycle", n=n, gating_mode="brickwork", depth_factor=df)
    cn = CircuitConfig(
        graph_spec="cycle", n=n, gating_mode="random_edge", gates_per_layer=n // 2, depth_factor=df
    )
    assert cn.total_layers() == cb.total_layers()  # m=n/2 matches brickwork


def test_depth_normalized_to_equal_gate_budget():
    # All three gating styles apply the SAME total gate count (gates-per-qubit
    # invariant), so they're comparable at equal depth_factor.
    n, df = 8, 2
    cb = CircuitConfig(graph_spec="cycle", n=n, gating_mode="brickwork", depth_factor=df)
    c1 = CircuitConfig(
        graph_spec="cycle", n=n, gating_mode="random_edge", gates_per_layer=1, depth_factor=df
    )
    cn = CircuitConfig(
        graph_spec="cycle", n=n, gating_mode="random_edge", gates_per_layer=n // 2, depth_factor=df
    )
    assert _gate_budget(cb) == _gate_budget(c1) == _gate_budget(cn) == df * n * (n // 2)


def test_single_edge_run_total_gates_matches_total_layers():
    cfg = CircuitConfig(
        graph_spec="cycle", n=8, gating_mode="random_edge", gates_per_layer=1, p=0.1, depth_factor=2
    )
    rec = simulate(cfg, sample_seed=0)
    assert rec.total_gates == rec.total_layers  # 1 gate/layer
    assert rec.total_layers == 64  # O(n^2)


# ----------------------------------------------------------------------
# expected gate/measurement ratio (recorded before the run)
# ----------------------------------------------------------------------


def test_expected_ratio_random_edge_modes_recorded():
    # single edge + bernoulli: eg=1, em=n*p -> 1/(n*p)
    c1 = CircuitConfig(
        graph_spec="cycle",
        n=8,
        gating_mode="random_edge",
        gates_per_layer=1,
        measurement_mode="bernoulli",
        p=0.1,
    )
    assert c1.expected_gate_to_meas_ratio() == pytest.approx(1.0 / (8 * 0.1))
    assert simulate(c1, sample_seed=0).gate_to_meas_ratio_expected == pytest.approx(1.0 / (8 * 0.1))
    # m = n/2 + bernoulli: eg=n/2, em=n*p -> 1/(2p)
    cn = CircuitConfig(
        graph_spec="cycle",
        n=8,
        gating_mode="random_edge",
        gates_per_layer=4,
        measurement_mode="bernoulli",
        p=0.1,
    )
    assert cn.expected_gate_to_meas_ratio() == pytest.approx(1.0 / (2 * 0.1))


# ----------------------------------------------------------------------
# measurement-mode correctness with multi-edge random_edge
# ----------------------------------------------------------------------


def test_gated_with_multi_edge_random_edge_only_touches_gate_qubits():
    cfg = CircuitConfig(
        graph_spec="cycle",
        n=8,
        gating_mode="random_edge",
        gates_per_layer=3,
        measurement_mode="gated",
        p=1.0,
        depth_factor=1,
    )
    for layer in CircuitBuilder(cfg, 0).schedule():
        touched = {q for pair in layer.gate_pairs for q in pair}
        assert set(layer.meas_qubits) <= touched  # candidates are exactly the touched qubits


def test_random_edge_gates_are_valid_distinct_pairs():
    # every emitted pair is a real graph edge with qi != qj
    cfg = CircuitConfig(
        graph_spec="complete", n=8, gating_mode="random_edge", gates_per_layer=5, depth_factor=1
    )
    edges = {tuple(e) for e in cfg._graph.edges}
    for layer in CircuitBuilder(cfg, 0).schedule():
        for u, v in layer.gate_pairs:
            assert u != v
            assert (min(u, v), max(u, v)) in edges
        # distinct edges within the layer (replace=False)
        assert len(set(layer.gate_pairs)) == layer.n_gates


# ----------------------------------------------------------------------
# nearest-neighbor models run end-to-end in BOTH gating modes
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "gating,kw",
    [
        ("brickwork", {}),
        ("random_edge", {"gates_per_layer": 1}),
        ("random_edge", {"gates_per_layer": 4}),  # O(n) edges, n=8
    ],
)
def test_nearest_neighbor_cycle_runs_in_both_gating_modes(gating, kw):
    cfg = CircuitConfig(graph_spec="cycle", n=8, gating_mode=gating, p=0.15, depth_factor=2, **kw)
    rec = simulate(cfg, sample_seed=0)
    assert rec.total_gates > 0
    assert rec.entropy_half_cut is not None
    assert np.isfinite(rec.gate_to_meas_ratio_expected)
