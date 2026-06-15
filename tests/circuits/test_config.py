"""Tests for CircuitConfig validation and derived quantities."""

from __future__ import annotations

import math

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
        {"depth_mode": "nope"},
        {"depth_factor": 0},
        {"n_cliffords": 0},
        {"n_cliffords": 99999},
        {"warmup_layers": -1},
        {"hybrid": "yes"},
        {"hybrid": 1},
    ],
)
def test_invalid_values_raise(kwargs):
    base = {"graph_spec": "cycle", "n": 8}
    base.update(kwargs)
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(**base)


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
