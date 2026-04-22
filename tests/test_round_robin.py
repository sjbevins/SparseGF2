"""
Validation of the round-robin 1-factorization used for the all-to-all
(complete-graph) gate schedule.

For any even ``n``, the 1-factorization of K_n returned by
:func:`sparsegf2.circuits.graphs.complete_graph` must satisfy the
combinatorial invariants of a proper 1-factorization (Anderson 2001):

- there are exactly ``n - 1`` perfect matchings (classes);
- each matching contains exactly ``n / 2`` edges;
- every edge of K_n appears in exactly one matching (no duplicates, no
  omissions), so the total edge count is ``n(n-1)/2``;
- within each matching every vertex is covered exactly once
  (no shared endpoints).
"""
from __future__ import annotations

from typing import Set, Tuple

import pytest

from sparsegf2.circuits.graphs import complete_graph


Edge = Tuple[int, int]


def _canon(e: Edge) -> Edge:
    a, b = e
    return (a, b) if a < b else (b, a)


@pytest.mark.parametrize("n", [2, 4, 6, 8, 10, 12, 16, 32])
def test_round_robin_is_a_valid_one_factorization(n: int):
    g = complete_graph(n)
    factorization = g.one_factorization
    assert factorization is not None, (
        f"K_{n}: no 1-factorization returned for even n"
    )

    # 1. n - 1 matchings
    assert len(factorization) == n - 1, (
        f"K_{n}: expected n-1={n-1} matchings, got {len(factorization)}"
    )

    all_edges: Set[Edge] = set()
    for layer_idx, matching in enumerate(factorization):
        # 2. Each matching has n/2 edges
        assert len(matching) == n // 2, (
            f"K_{n} layer {layer_idx}: expected {n // 2} edges, got "
            f"{len(matching)}"
        )
        # 3. Within a matching, each vertex appears at most once
        vertices = []
        for e in matching:
            vertices.extend(e)
        assert len(set(vertices)) == n, (
            f"K_{n} layer {layer_idx}: matching does not cover every "
            f"vertex exactly once (vertices seen: {sorted(vertices)})"
        )
        # 4. No edge is reused across matchings
        for e in matching:
            canon = _canon(e)
            assert canon not in all_edges, (
                f"K_{n} layer {layer_idx}: edge {canon} reused "
                "(violates 1-factorization)"
            )
            all_edges.add(canon)

    # 5. All n(n-1)/2 edges of K_n covered
    expected_edges = n * (n - 1) // 2
    assert len(all_edges) == expected_edges, (
        f"K_{n}: 1-factorization covers {len(all_edges)} edges, expected "
        f"{expected_edges}"
    )


@pytest.mark.parametrize("n", [3, 5, 7, 9, 13])
def test_odd_n_has_no_one_factorization(n: int):
    """K_n with odd n has no perfect matching, hence no 1-factorization."""
    g = complete_graph(n)
    assert g.one_factorization is None, (
        f"K_{n}: odd n should have no 1-factorization"
    )
