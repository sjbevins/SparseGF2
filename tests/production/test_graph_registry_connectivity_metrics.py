from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from studies.prl_production.graph_registry.connectivity_metrics import (
    algebraic_connectivity_from_edges,
    circulant_c2_algebraic_connectivity,
    ring_algebraic_connectivity,
    watts_strogatz_algebraic_connectivity,
    watts_strogatz_c2_algebraic_connectivity,
)
from studies.prl_production.graph_registry.spec import beta_from_key, canonical_beta_key

from sparsegf2.circuits.graphs import _ws_rewire_edges, watts_strogatz


def _dense_full_spectrum_lambda2(n: int, edges: list[tuple[int, int]]) -> float:
    adjacency = np.zeros((n, n), dtype=np.float64)
    for left, right in edges:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    return float(np.linalg.eigvalsh(laplacian)[1])


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (64, 0.0480599858491453),
        (96, 0.0213924307751723),
        (128, 0.0120396342452613),
        (160, 0.00770726005229805),
        (192, 0.00535297857006167),
        (224, 0.00393311719826084),
        (256, 0.00301145019724691),
    ],
)
def test_circulant_formula_known_production_values(n: int, expected: float) -> None:
    assert circulant_c2_algebraic_connectivity(n) == pytest.approx(expected, abs=2e-15)
    assert ring_algebraic_connectivity(n) == pytest.approx(expected, abs=2e-15)


@pytest.mark.parametrize("n", [5, 8, 64, 128])
@pytest.mark.parametrize("seed", [0, 17, 2**40 + 39])
def test_beta_zero_matches_exact_circulant_formula(n: int, seed: int) -> None:
    assert watts_strogatz_c2_algebraic_connectivity(n, 0.0, seed) == pytest.approx(
        circulant_c2_algebraic_connectivity(n), rel=2e-13, abs=2e-14
    )
    assert watts_strogatz_algebraic_connectivity(n, 2, 0.0, seed) == pytest.approx(
        ring_algebraic_connectivity(n, 2), rel=2e-13, abs=2e-14
    )


@pytest.mark.parametrize(
    ("n", "beta", "seed"),
    [(8, 0.25, 3), (24, 0.031, 91), (64, 0.5, 2**40 + 39)],
)
def test_private_edge_path_matches_public_graph(n: int, beta: float, seed: int) -> None:
    canonical_beta = beta_from_key(canonical_beta_key(beta))
    private_edges = _ws_rewire_edges(n, 2, canonical_beta, seed)
    public_edges = watts_strogatz(n, k=2, beta=canonical_beta, seed=seed).edges

    assert private_edges == public_edges
    assert watts_strogatz_c2_algebraic_connectivity(n, beta, seed) == pytest.approx(
        algebraic_connectivity_from_edges(n, public_edges), rel=0.0, abs=0.0
    )


@pytest.mark.parametrize(
    ("n", "beta", "seed"),
    [(8, 0.1, 0), (16, 0.4, 7), (32, 1.0, 12345), (64, 0.005, 2**40 + 39)],
)
def test_subset_eigensolve_matches_dense_full_spectrum(n: int, beta: float, seed: int) -> None:
    edges = watts_strogatz(n, k=2, beta=beta, seed=seed).edges

    assert algebraic_connectivity_from_edges(n, edges) == pytest.approx(
        _dense_full_spectrum_lambda2(n, edges), rel=2e-12, abs=2e-13
    )


def test_connected_and_disconnected_graphs() -> None:
    path_edges = [(0, 1), (1, 2), (2, 3)]
    disconnected_edges = [(0, 1), (2, 3)]

    assert algebraic_connectivity_from_edges(4, path_edges) == pytest.approx(
        2.0 - math.sqrt(2.0), abs=2e-15
    )
    assert algebraic_connectivity_from_edges(4, disconnected_edges) == 0.0
    assert algebraic_connectivity_from_edges(3, []) == 0.0


def test_edge_order_and_repeated_calls_are_deterministic() -> None:
    edges = watts_strogatz(32, k=2, beta=0.375, seed=987654321).edges
    first = algebraic_connectivity_from_edges(32, edges)

    assert algebraic_connectivity_from_edges(32, list(reversed(edges))) == first
    assert algebraic_connectivity_from_edges(32, iter(edges)) == first
    assert watts_strogatz_c2_algebraic_connectivity(32, 0.375, 987654321) == first
    assert watts_strogatz_c2_algebraic_connectivity(32, 0.375, 987654321) == first


@pytest.mark.parametrize(
    ("n", "edges", "error", "message"),
    [
        (True, [], TypeError, "n must be an integer"),
        (1, [], ValueError, "n must be >= 2"),
        (4, "not edges", TypeError, "edges must be an iterable"),
        (4, [1], TypeError, "edge 0 must be an iterable"),
        (4, [(0, 1, 2)], ValueError, "exactly two endpoints"),
        (4, [(False, 1)], TypeError, "endpoint 0 must be an integer"),
        (4, [(0, 1.0)], TypeError, "endpoint 1 must be an integer"),
        (4, [(-1, 1)], ValueError, "endpoints must lie"),
        (4, [(0, 4)], ValueError, "endpoints must lie"),
        (4, [(1, 1)], ValueError, "self-loop"),
        (4, [(0, 1), (1, 0)], ValueError, "duplicates undirected edge"),
    ],
)
def test_invalid_edge_graphs_are_rejected(
    n: int,
    edges: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        algebraic_connectivity_from_edges(n, edges)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        ((4,), ValueError, "k must satisfy"),
        ((8.0,), TypeError, "n must be an integer"),
    ],
)
def test_invalid_circulant_parameters(
    args: tuple[object, ...], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        circulant_c2_algebraic_connectivity(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        ((8, True), TypeError, "k must be an integer"),
        ((8, 0), ValueError, "k must be >= 1"),
        ((8, 4), ValueError, "k must satisfy"),
    ],
)
def test_invalid_general_ring_parameters(
    args: tuple[object, ...], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        ring_algebraic_connectivity(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        ((4, 0.5, 0), ValueError, "k must satisfy"),
        ((8, True, 0), TypeError, "beta must be a real number"),
        ((8, math.nan, 0), ValueError, "beta must be finite"),
        ((8, -0.1, 0), ValueError, "beta must be finite"),
        ((8, 1.1, 0), ValueError, "beta must be finite"),
        ((8, 0.5, True), TypeError, "seed must be an integer"),
        ((8, 0.5, -1), ValueError, "seed must be >= 0"),
    ],
)
def test_invalid_watts_strogatz_parameters(
    args: tuple[object, ...], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        watts_strogatz_c2_algebraic_connectivity(*args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        ((8, True, 0.5, 0), TypeError, "k must be an integer"),
        ((8, 0, 0.5, 0), ValueError, "k must be >= 1"),
        ((8, 4, 0.5, 0), ValueError, "k must satisfy"),
    ],
)
def test_invalid_general_watts_strogatz_parameters(
    args: tuple[object, ...], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        watts_strogatz_algebraic_connectivity(*args)  # type: ignore[arg-type]
