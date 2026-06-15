"""The actual gate/measurement ratio should track the expected one."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.circuits import CircuitConfig, lattice_2d, simulate
from sparsegf2.circuits.config import GATING_MODES
from sparsegf2.errors import InvalidArgumentError


def test_actual_ratio_near_expected_bernoulli():
    # brickwork + bernoulli: expected = 1/(2p). Average actual over many
    # samples should land near it.
    cfg = CircuitConfig(
        graph_spec="cycle", n=16, p=0.2, depth_factor=4, measurement_mode="bernoulli"
    )
    expected = cfg.expected_gate_to_meas_ratio()
    ratios = [simulate(cfg, sample_seed=s).gate_to_meas_ratio_actual for s in range(40)]
    mean = float(np.mean(ratios))
    # n*p*T measurements expected; loose 25% band
    assert abs(mean - expected) / expected < 0.25


def test_gate_count_matches_brickwork_formula():
    cfg = CircuitConfig(graph_spec="cycle", n=10, p=0.1, depth_factor=3)
    rec = simulate(cfg, sample_seed=0)
    # T = 3*10 = 30 layers, n/2 = 5 gates/layer
    assert rec.total_layers == 30
    assert rec.total_gates == 30 * 5


def test_p_zero_gives_no_measurements_and_inf_ratio():
    cfg = CircuitConfig(graph_spec="cycle", n=8, p=0.0, depth_factor=2)
    rec = simulate(cfg, sample_seed=0)
    assert rec.total_measurements == 0
    assert rec.gate_to_meas_ratio_actual == float("inf")


def test_random_pool_bernoulli_realized_tracks_expected():
    # random_pool default m=n/2; bernoulli em=n*p does not depend on which
    # qubits the gates touch, so the realized ratio tracks expected exactly.
    cfg = CircuitConfig(graph_spec="cycle", n=16, p=0.2, gating_mode="random_pool", depth_factor=4)
    expected = cfg.expected_gate_to_meas_ratio()
    ratios = [simulate(cfg, sample_seed=s).gate_to_meas_ratio_actual for s in range(40)]
    assert abs(float(np.mean(ratios)) - expected) / expected < 0.25


def test_random_pool_gated_expected_is_lower_bound():
    # For 'gated', the expected ratio assumes 2*eg distinct touched qubits, an
    # over-estimate of candidates when edges repeat/share (random_pool). So the
    # documented contract: expected ratio <= realized ratio.
    cfg = CircuitConfig(
        graph_spec="cycle",
        n=16,
        p=0.2,
        gating_mode="random_pool",
        measurement_mode="gated",
        depth_factor=4,
    )
    expected = cfg.expected_gate_to_meas_ratio()
    realized = float(
        np.mean([simulate(cfg, sample_seed=s).gate_to_meas_ratio_actual for s in range(40)])
    )
    assert realized >= expected * 0.95  # realized is the (larger) ground truth


@pytest.mark.parametrize("mode", GATING_MODES)
def test_to_dict_round_trips_all_gating_modes(mode):
    kw = {} if mode == "brickwork" else {"gates_per_layer": 3}
    cfg = CircuitConfig(graph_spec="cycle", n=8, p=0.2, gating_mode=mode, **kw)
    cfg2 = CircuitConfig(**cfg.to_dict())  # must not raise for any mode
    assert cfg2.gating_mode == mode
    assert cfg2.total_layers() == cfg.total_layers()


def test_callable_gates_per_layer_wrong_arity_raises_at_construction():
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(
            graph_spec="cycle",
            n=8,
            p=0.2,
            gating_mode="random_edge",
            gates_per_layer=lambda: 4,  # wrong arity: should take the config
        )


def test_brickwork_gate_count_irregular_graph():
    # On an irregular graph the brickwork colour classes are not all n/2 edges;
    # total_gates must equal the sum of the cycled class sizes (not T * n/2).
    L = lattice_2d(4, 4)
    cfg = CircuitConfig(
        graph_spec=L,
        n=16,
        p=0.1,
        gating_mode="brickwork",
        matching_mode="round_robin",
        depth_factor=2,
    )
    rec = simulate(cfg, sample_seed=0)
    chi = L.chi_prime
    sizes = [len(m) for m in L.one_factorization]
    expected_gates = sum(sizes[t % chi] for t in range(cfg.total_layers()))
    assert rec.total_gates == expected_gates
    assert rec.total_gates != cfg.total_layers() * (16 // 2)  # not the perfect-matching count
