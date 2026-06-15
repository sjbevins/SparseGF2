"""Newman-Watts small-world geometry: construction, reproducibility, and the
mode-availability matrix (every mode works except the 1-factorization modes,
which a stochastic non-regular graph cannot supply)."""

from __future__ import annotations

import pytest

from sparsegf2.circuits import CircuitConfig, simulate
from sparsegf2.circuits.graphs import from_spec, newman_watts
from sparsegf2.errors import InvalidArgumentError

pytest.importorskip("networkx")


# ---------------------------------------------------------------- construction


def test_newman_watts_is_stochastic_without_one_factorization():
    g = newman_watts(16, k=2, p=0.5, seed=0)
    assert g.n == 16
    assert g.is_stochastic is True
    assert g.has_one_factorization is False  # the whole point: no 1-factorization
    # built on a ring, so it is connected and has at least the n ring edges
    assert len(g.edges) >= 16


def test_from_spec_matches_direct_constructor():
    assert from_spec("newman_watts", 16, seed=3).edges == newman_watts(16, seed=3).edges


def test_realisation_is_reproducible_and_seed_dependent():
    assert newman_watts(20, seed=7).edges == newman_watts(20, seed=7).edges
    assert newman_watts(20, seed=7).edges != newman_watts(20, seed=8).edges


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 2},  # too few nodes
        {"n": 16, "k": 3},  # odd k
        {"n": 16, "k": 0},  # k < 2
        {"n": 16, "k": 16},  # k >= n
        {"n": 16, "p": 1.5},  # p out of range
    ],
)
def test_validation(kwargs):
    with pytest.raises(InvalidArgumentError):
        newman_watts(**kwargs)


# ----------------------------------------------------------- mode availability


def _runs(**cfg_kwargs) -> bool:
    cfg = CircuitConfig(graph_spec=newman_watts(16, k=2, p=0.5, seed=0), n=16, **cfg_kwargs)
    simulate(cfg, sample_seed=0)
    return True


def test_edge_gating_modes_available():
    assert _runs(gating_mode="random_edge", picture="purification", p=0.1)
    assert _runs(gating_mode="random_pool", picture="purification", p=0.1)


@pytest.mark.parametrize("matching_mode", ["round_robin", "palette"])
def test_one_factorization_modes_unavailable(matching_mode):
    with pytest.raises(InvalidArgumentError, match="1-factorization"):
        CircuitConfig(
            graph_spec=newman_watts(16, k=2, p=0.5, seed=0),
            n=16,
            gating_mode="brickwork",
            matching_mode=matching_mode,
        )


def test_brickwork_fresh_available_when_perfect_matching_exists():
    g = newman_watts(16, k=2, p=0.5, seed=0)
    assert g.has_perfect_matching  # this realisation has a PM
    assert _runs(gating_mode="brickwork", matching_mode="fresh", picture="purification", p=0.1)


@pytest.mark.parametrize("measurement_mode", ["bernoulli", "gated", "random_pair", "uniform_count"])
def test_all_measurement_modes(measurement_mode):
    assert _runs(
        gating_mode="random_pool", measurement_mode=measurement_mode, picture="purification", p=0.1
    )


@pytest.mark.parametrize("picture", ["pure_state", "purification", "single_ref"])
def test_all_pictures(picture):
    assert _runs(gating_mode="random_pool", picture=picture, p=0.1)


# --------------------------------------------------------- base_seed threading


def test_base_seed_selects_graph_realisation():
    # via the string spec, base_seed is the realisation seed: different base_seed
    # -> different geometry; same base_seed -> identical geometry. (random_pool
    # gating, since the default brickwork+round_robin needs a 1-factorization the
    # stochastic graph lacks.)
    common = dict(graph_spec="newman_watts", n=16, gating_mode="random_pool")
    a = CircuitConfig(**common, base_seed=1)
    b = CircuitConfig(**common, base_seed=1)
    c = CircuitConfig(**common, base_seed=2)
    assert a._graph.edges == b._graph.edges
    assert a._graph.edges != c._graph.edges
