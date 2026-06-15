"""Phase-4 round-out: random_pool gating, uniform_count, until_purified, time series."""

from __future__ import annotations

import pytest

from sparsegf2.circuits import CircuitConfig, simulate
from sparsegf2.errors import InvalidArgumentError

# ---- random_pool gating ----


def test_random_pool_default_matches_brickwork_ratio():
    n, p = 16, 0.2
    brick = CircuitConfig(graph_spec="cycle", n=n, p=p, gating_mode="brickwork")
    pool = CircuitConfig(graph_spec="cycle", n=n, p=p, gating_mode="random_pool")
    assert pool.resolved_gates_per_layer() == n // 2
    assert pool.expected_gate_to_meas_ratio() == pytest.approx(brick.expected_gate_to_meas_ratio())
    assert pool.expected_gate_to_meas_ratio() == pytest.approx(1.0 / (2 * p))


def test_random_pool_coefficient_changes_ratio():
    n, p = 16, 0.2
    pool_n = CircuitConfig(
        graph_spec="cycle", n=n, p=p, gating_mode="random_pool", gates_per_layer=n
    )
    assert pool_n.resolved_gates_per_layer() == n
    assert pool_n.expected_gate_to_meas_ratio() == pytest.approx(1.0 / p)


def test_random_pool_with_replacement_count():
    # n/2 gates per layer even though the cycle has only n edges (with replacement)
    cfg = CircuitConfig(graph_spec="cycle", n=12, p=0.2, gating_mode="random_pool")
    rec = simulate(cfg, sample_seed=0)
    assert rec.total_gates == rec.total_layers * (12 // 2)


def test_random_edge_default_unchanged():
    cfg = CircuitConfig(graph_spec="cycle", n=10, p=0.2, gating_mode="random_edge")
    assert cfg.resolved_gates_per_layer() == 1  # backward-compatible default


def test_brickwork_rejects_gates_per_layer():
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(graph_spec="cycle", n=8, gating_mode="brickwork", gates_per_layer=4)


# ---- uniform_count measurement ----


def test_uniform_count_ratio():
    cfg = CircuitConfig(
        graph_spec="cycle", n=20, p=0.3, measurement_mode="uniform_count", meas_count=5
    )
    assert cfg.resolved_meas_count() == 5
    assert cfg.expected_gate_to_meas_ratio() == pytest.approx((20 / 2) / (5 * 0.3))


def test_uniform_count_default_half():
    cfg = CircuitConfig(graph_spec="cycle", n=16, measurement_mode="uniform_count")
    assert cfg.resolved_meas_count() == 8


def test_uniform_count_candidates_size():
    from sparsegf2.circuits.scheduler import CircuitBuilder

    cfg = CircuitConfig(
        graph_spec="cycle", n=20, p=1.0, measurement_mode="uniform_count", meas_count=6
    )
    layer = next(CircuitBuilder(cfg, 0).layers())
    assert len(layer.meas_candidates) == 6
    assert len(layer.meas_qubits) == 6  # p=1 -> all candidates fire


def test_meas_count_without_uniform_raises():
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(graph_spec="cycle", n=8, meas_count=3)


def test_meas_count_too_large_raises():
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(graph_spec="cycle", n=8, measurement_mode="uniform_count", meas_count=99)


# ---- until_purified early-stop ----


def test_until_purified_stops_early():
    cfg = CircuitConfig(
        graph_spec="complete",
        n=12,
        picture="purification",
        p=0.4,
        depth_mode="until_purified",
        depth_factor=20,
    )
    rec = simulate(cfg, sample_seed=0)
    assert rec.purified_at_layer is not None
    assert rec.total_layers == rec.purified_at_layer
    assert rec.total_layers < cfg.total_layers()  # stopped before the cap
    assert rec.code_dimension == 0


def test_until_purified_pure_state_rejected():
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(graph_spec="cycle", n=8, picture="pure_state", depth_mode="until_purified")


# ---- record_time_series ----


def test_time_series_recorded():
    cfg = CircuitConfig(
        graph_spec="complete",
        n=12,
        picture="purification",
        p=0.25,
        record_time_series=True,
        depth_factor=4,
    )
    rec = simulate(cfg, sample_seed=0)
    assert rec.time_series is not None
    assert len(rec.time_series) == rec.total_layers
    assert all(isinstance(x, int) for x in rec.time_series)
    assert rec.time_series[-1] == rec.code_dimension  # last entry == final k


def test_time_series_none_by_default():
    cfg = CircuitConfig(graph_spec="cycle", n=8, picture="purification", p=0.2, depth_factor=2)
    rec = simulate(cfg, sample_seed=0)
    assert rec.time_series is None
    assert rec.purified_at_layer is None


def test_until_purified_hits_cap_when_never_purified():
    # p=0: no measurements, the system never purifies -> the loop runs to the cap.
    cfg = CircuitConfig(
        graph_spec="cycle",
        n=8,
        picture="purification",
        p=0.0,
        depth_mode="until_purified",
        depth_factor=2,
    )
    rec = simulate(cfg, sample_seed=0)
    assert rec.purified_at_layer is None
    assert rec.total_layers == cfg.total_layers()
    assert rec.code_dimension > 0


def test_time_series_single_ref_tracks_ref_entropy():
    cfg = CircuitConfig(
        graph_spec="complete",
        n=10,
        picture="single_ref",
        p=0.3,
        record_time_series=True,
        depth_factor=4,
    )
    rec = simulate(cfg, sample_seed=0)
    assert rec.time_series is not None
    assert len(rec.time_series) == rec.total_layers
    assert all(x in (0, 1) for x in rec.time_series)
    assert rec.time_series[-1] == rec.ref_entropy
    assert rec.code_dimension is None


def test_until_purified_with_time_series_truncates_together():
    cfg = CircuitConfig(
        graph_spec="complete",
        n=10,
        picture="purification",
        p=0.5,
        depth_mode="until_purified",
        record_time_series=True,
        depth_factor=20,
    )
    rec = simulate(cfg, sample_seed=0)
    assert rec.purified_at_layer is not None
    assert len(rec.time_series) == rec.total_layers == rec.purified_at_layer
    assert rec.time_series[-1] == 0


def test_uniform_count_full_n_candidates():
    from sparsegf2.circuits.scheduler import CircuitBuilder

    cfg = CircuitConfig(
        graph_spec="cycle", n=8, p=1.0, measurement_mode="uniform_count", meas_count=8
    )
    layer = next(CircuitBuilder(cfg, 0).layers())
    assert sorted(layer.meas_candidates) == list(range(8))
    assert len(layer.meas_qubits) == 8  # p=1 -> every candidate fires


def test_random_pool_exceeds_edge_count():
    # complete(4) has 6 edges; m=12 with replacement still fires 12 gates/layer.
    cfg = CircuitConfig(
        graph_spec="complete",
        n=4,
        p=0.2,
        gating_mode="random_pool",
        gates_per_layer=12,
        depth_factor=1,
    )
    rec = simulate(cfg, sample_seed=0)
    assert rec.total_gates == rec.total_layers * 12


def test_random_pool_repeated_edges_and_cliff_index_alignment():
    from sparsegf2.circuits.scheduler import CircuitBuilder

    cfg = CircuitConfig(
        graph_spec="cycle", n=4, p=0.0, gating_mode="random_pool", gates_per_layer=12
    )
    layer = next(CircuitBuilder(cfg, 0).layers())
    # one Clifford index per placed gate, even with duplicate edges
    assert len(layer.cliff_indices) == len(layer.gate_pairs) == 12
    # with replacement on a 4-edge graph, 12 draws must contain a duplicate edge
    assert len(set(layer.gate_pairs)) < len(layer.gate_pairs)
