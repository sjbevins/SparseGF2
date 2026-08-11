from __future__ import annotations

import math

import pytest
from studies.prl_production.graph_registry.rewiring_metrics import (
    RewiringCounts,
    _replay_watts_strogatz_rewiring,
    watts_strogatz_rewiring_counts,
)
from studies.prl_production.graph_registry.spec import beta_from_key, canonical_beta_key

from sparsegf2.circuits.graphs import _ws_rewire_edges


def _base_edges(n: int, k: int) -> set[tuple[int, int]]:
    return {
        (min(vertex, (vertex + offset) % n), max(vertex, (vertex + offset) % n))
        for offset in range(1, k + 1)
        for vertex in range(n)
    }


@pytest.mark.parametrize(
    ("n", "k"),
    [(4, 1), (5, 1), (6, 1), (6, 2), (7, 2), (8, 3), (12, 2), (64, 2)],
)
@pytest.mark.parametrize("beta", [0.0, 0.005, 0.031, 0.25, 0.7, 1.0])
@pytest.mark.parametrize("seed", [0, 17, 2**40 + 39])
def test_replay_matches_generator_and_final_displacement(
    n: int, k: int, beta: float, seed: int
) -> None:
    counts, final_edges = _replay_watts_strogatz_rewiring(n, k, beta, seed)
    canonical_beta = beta_from_key(canonical_beta_key(beta))
    expected_edges = _ws_rewire_edges(n, k, canonical_beta, seed)

    assert list(final_edges) == expected_edges
    assert counts.displaced_base_edges == len(_base_edges(n, k).difference(final_edges))
    assert counts.displaced_base_edges == (
        counts.successful_operations - counts.restored_base_edges
    )
    assert len(final_edges) == n * k


def test_beta_zero_has_no_rewiring() -> None:
    counts = watts_strogatz_rewiring_counts(64, 2, 0.0, 12345)

    assert counts == RewiringCounts(
        successful_operations=0,
        restored_base_edges=0,
        displaced_base_edges=0,
        skipped_full_neighbor=0,
    )


@pytest.mark.parametrize(("n", "k", "seed"), [(4, 1, 8), (6, 2, 3), (64, 2, 91)])
def test_beta_one_selects_every_base_edge(n: int, k: int, seed: int) -> None:
    counts = watts_strogatz_rewiring_counts(n, k, 1.0, seed)

    assert counts.successful_operations + counts.skipped_full_neighbor == n * k
    assert counts.displaced_base_edges == (
        counts.successful_operations - counts.restored_base_edges
    )


def test_replay_distinguishes_restorations_and_full_neighbor_skips() -> None:
    assert watts_strogatz_rewiring_counts(6, 2, 1.0, 0) == RewiringCounts(
        successful_operations=11,
        restored_base_edges=8,
        displaced_base_edges=3,
        skipped_full_neighbor=1,
    )


def test_repeat_and_equal_canonical_beta_keys_are_deterministic() -> None:
    args = (96, 2, 0.1234567894, 9_876_543_210)
    first = watts_strogatz_rewiring_counts(*args)

    assert first == watts_strogatz_rewiring_counts(*args)
    assert first == watts_strogatz_rewiring_counts(96, 2, 0.123456789, args[-1])


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        ((True, 1, 0.5, 0), TypeError, "n must be an integer"),
        ((8.0, 1, 0.5, 0), TypeError, "n must be an integer"),
        ((2, 1, 0.5, 0), ValueError, "n must be >= 3"),
        ((8, True, 0.5, 0), TypeError, "k must be an integer"),
        ((8, 0, 0.5, 0), ValueError, "k must be >= 1"),
        ((8, 4, 0.5, 0), ValueError, "k must satisfy"),
        ((8, 1, True, 0), TypeError, "beta must be a real number"),
        ((8, 1, math.nan, 0), ValueError, "beta must be finite"),
        ((8, 1, math.inf, 0), ValueError, "beta must be finite"),
        ((8, 1, -0.01, 0), ValueError, "beta must be finite"),
        ((8, 1, 1.01, 0), ValueError, "beta must be finite"),
        ((8, 1, 0.5, True), TypeError, "seed must be an integer"),
        ((8, 1, 0.5, 1.0), TypeError, "seed must be an integer"),
        ((8, 1, 0.5, -1), ValueError, "seed must be >= 0"),
    ],
)
def test_invalid_parameters_are_rejected(
    args: tuple[object, object, object, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        watts_strogatz_rewiring_counts(*args)  # type: ignore[arg-type]
