"""Tests for CircuitConfig validation and derived quantities."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sparsegf2.circuits import CircuitConfig, Picture, cycle_graph
from sparsegf2.errors import InvalidArgumentError


def test_defaults_and_picture_coercion():
    c = CircuitConfig(graph_spec="cycle", n=8)
    assert c.picture is Picture.PURE_STATE  # coerced from default
    assert c.gating_mode == "brickwork"
    assert c.n_cliffords == 720  # NOT 11520


def test_string_picture_coerced_to_enum():
    c = CircuitConfig(graph_spec="cycle", n=8, picture="purification")
    assert c.picture is Picture.PURIFICATION


def test_prebuilt_graph_topology_accepted():
    g = cycle_graph(8)
    c = CircuitConfig(graph_spec=g, n=8)
    assert c.graph_spec is g


def test_prebuilt_graph_n_mismatch_raises():
    g = cycle_graph(8)
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(graph_spec=g, n=16)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 1},
        {"graph_spec": 123},
        {"picture": "nope"},
        {"gating_mode": "nope"},
        {"matching_mode": "nope"},
        {"measurement_mode": "nope"},
        {"p": 1.5},
        {"p": -0.1},
        {"p": True},
        {"p": "0.1"},
        {"p": np.nan},
        {"p": np.inf},
        {"depth_mode": "nope"},
        {"depth_factor": 0},
        {"n_cliffords": 0},
        {"n_cliffords": 99999},
        {"warmup_layers": -1},
        {"hybrid": "yes"},
        {"hybrid": 1},
        {"use_numba": "yes"},
        {"use_numba": 1},
        {"base_seed": -1},  # streams are keyed by a non-negative entropy pair
        {"base_seed": 1.5},
    ],
)
def test_invalid_values_raise(kwargs):
    base = {"graph_spec": "cycle", "n": 8}
    base.update(kwargs)
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(**base)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": True},
        {"gating_mode": "random_edge", "gates_per_layer": True},
        {"measurement_mode": "uniform_count", "meas_count": True},
        {"depth_factor": True},
        {"n_cliffords": True},
        {"base_seed": True},
        {"warmup_layers": True},
    ],
)
def test_integer_knobs_reject_boolean_aliases(kwargs):
    base = {"graph_spec": "cycle", "n": 8}
    base.update(kwargs)
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(**base)


@pytest.mark.parametrize("resolved", [1.5, True, np.float64(2.0)])
def test_gates_per_layer_callable_must_return_exact_integer(resolved):
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(
            graph_spec="cycle",
            n=8,
            gating_mode="random_edge",
            gates_per_layer=lambda cfg: resolved,
        )


def test_integer_knobs_accept_numpy_integer_scalars():
    cfg = CircuitConfig(
        graph_spec="cycle",
        n=np.int64(8),
        gating_mode="random_edge",
        gates_per_layer=np.int32(2),
        measurement_mode="uniform_count",
        meas_count=np.int16(3),
        depth_factor=np.int64(2),
        n_cliffords=np.int32(100),
        base_seed=np.int64(7),
        warmup_layers=np.int16(1),
    )
    assert cfg.n == 8
    assert cfg.resolved_gates_per_layer() == 2
    assert cfg.resolved_meas_count() == 3


def test_measurement_rate_accepts_numpy_real_scalar():
    cfg = CircuitConfig(graph_spec="cycle", n=8, p=np.float32(0.25))
    assert cfg.p == pytest.approx(0.25)
    assert isinstance(cfg.p, float)

    fractional = CircuitConfig(graph_spec="cycle", n=8, p=Fraction(1, 4))
    assert fractional.p == pytest.approx(0.25)


def test_record_time_series_forbidden_for_pure_state():
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(graph_spec="cycle", n=8, picture="pure_state", record_time_series=True)
    # but allowed for purification
    CircuitConfig(graph_spec="cycle", n=8, picture="purification", record_time_series=True)


def test_until_purified_forbidden_for_pure_state():
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(graph_spec="cycle", n=8, picture="pure_state", depth_mode="until_purified")


def test_total_layers():
    assert (
        CircuitConfig(graph_spec="cycle", n=8, depth_mode="O(n)", depth_factor=4).total_layers()
        == 32
    )
    c = CircuitConfig(graph_spec="cycle", n=8, depth_mode="O(log_n)", depth_factor=4)
    assert c.total_layers() == 4 * math.ceil(math.log2(8))
    # all_edges defines a layer by firing the complete edge list and deliberately
    # keeps the base layer count rather than normalizing away that architecture.
    c = CircuitConfig(
        graph_spec="complete", n=8, gating_mode="all_edges", depth_mode="O(n)", depth_factor=4
    )
    assert c.total_layers() == 32


def test_total_layers_override_returns_exact_count():
    # An explicit override is the literal measured-layer count, bypassing the
    # depth_mode/depth_factor formula AND the random-edge gating rescale.
    c = CircuitConfig(
        graph_spec="cycle",
        n=8,
        gating_mode="random_pool",
        depth_mode="O(n)",
        depth_factor=4,
        total_layers_override=13,
    )
    assert c.total_layers() == 13
    # Same override under a different mode/factor still yields the literal count.
    c2 = CircuitConfig(
        graph_spec="cycle",
        n=8,
        depth_mode="O(log_n)",
        depth_factor=99,
        total_layers_override=1,
    )
    assert c2.total_layers() == 1


def test_total_layers_override_preserves_preexisting_positional_layout():
    cfg = CircuitConfig(
        "cycle",
        8,
        Picture.PURE_STATE,
        "brickwork",
        "round_robin",
        None,
        "bernoulli",
        None,
        0.2,
        "O(n)",
        3,
        100,
        9,
        False,
        1,
        False,
        True,
        None,
        None,
        True,
    )
    assert cfg.n_cliffords == 100
    assert cfg.base_seed == 9
    assert cfg.warmup_layers == 1
    assert cfg.hybrid is True
    assert cfg.total_layers_override is None


def test_total_layers_override_default_none_keeps_formula():
    c = CircuitConfig(graph_spec="cycle", n=8, depth_mode="O(n)", depth_factor=4)
    assert c.total_layers_override is None
    assert c.total_layers() == 32


@pytest.mark.parametrize("depth_mode", ["O(n)", "O(log_n)", "until_purified"])
def test_total_layers_override_composes_with_all_edges_and_every_depth_mode(depth_mode):
    cfg = CircuitConfig(
        graph_spec="cycle",
        n=8,
        picture="purification",
        gating_mode="all_edges",
        depth_mode=depth_mode,
        depth_factor=99,
        total_layers_override=np.int64(7),
    )
    assert cfg.total_layers_override == 7
    assert cfg.total_layers() == 7
    restored = CircuitConfig(**cfg.to_dict())
    assert restored.total_layers_override == 7
    assert restored.total_layers() == 7


@pytest.mark.parametrize("bad", [0, -1, 1.5, "10", True])
def test_total_layers_override_invalid_raises(bad):
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(graph_spec="cycle", n=8, total_layers_override=bad)


def test_total_qubits_by_picture():
    assert CircuitConfig(graph_spec="cycle", n=8, picture="pure_state").total_qubits() == 8
    assert CircuitConfig(graph_spec="cycle", n=8, picture="purification").total_qubits() == 16


def test_expected_ratio_modes():
    # brickwork + bernoulli: (n/2)/(n*p) = 1/(2p)
    c = CircuitConfig(
        graph_spec="cycle", n=8, gating_mode="brickwork", measurement_mode="bernoulli", p=0.1
    )
    assert c.expected_gate_to_meas_ratio() == pytest.approx(1.0 / (2 * 0.1))
    # p=0 -> inf
    c0 = CircuitConfig(graph_spec="cycle", n=8, p=0.0)
    assert c0.expected_gate_to_meas_ratio() == float("inf")
    # random_edge + random_pair: eg=1, em=2p -> 1/(2p)
    cr = CircuitConfig(
        graph_spec="cycle", n=8, gating_mode="random_edge", measurement_mode="random_pair", p=0.25
    )
    assert cr.expected_gate_to_meas_ratio() == pytest.approx(1.0 / (2 * 0.25))


def test_to_dict_serializes_enum():
    d = CircuitConfig(graph_spec="cycle", n=8, picture="purification").to_dict()
    assert d["picture"] == "purification"
    assert isinstance(d["picture"], str)
