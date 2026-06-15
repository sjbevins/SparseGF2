"""Tests for graph topologies, 1-factorizations, and graph6 encoding."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.circuits.graphs import (
    GraphTopology,
    complete_graph,
    cycle_graph,
    from_spec,
    graph6_encode,
)

# ---- cycle ----


def test_cycle_edges_and_degree():
    g = cycle_graph(8)
    assert g.n == 8
    assert len(g.edges) == 8
    assert g.degree_max == 2
    assert all(u < v for u, v in g.edges)


def test_cycle_one_factorization_covers_all_edges_once():
    g = cycle_graph(8)
    assert g.has_one_factorization
    assert g.chi_prime == 2
    union = [e for m in g.one_factorization for e in m]
    # two disjoint perfect matchings, each n/2 edges, covering all 8 edges once
    assert len(union) == 8
    assert sorted(union) == sorted(g.edges)
    for m in g.one_factorization:
        assert len(m) == 4  # perfect matching on 8 vertices
        verts = [q for e in m for q in e]
        assert sorted(verts) == list(range(8))


def test_cycle_odd_n_has_no_perfect_matching():
    g = cycle_graph(7)
    assert not g.has_one_factorization
    assert not g.has_perfect_matching
    assert g.chi_prime is None


def test_cycle_fresh_sampler_returns_perfect_matchings():
    g = cycle_graph(8)
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = g.fresh_matching_sampler(rng)
        verts = [q for e in m for q in e]
        assert sorted(verts) == list(range(8))


def test_cycle_requires_n_at_least_3():
    with pytest.raises(ValueError):
        cycle_graph(2)


# ---- complete ----


def test_complete_edges_count():
    g = complete_graph(6)
    assert len(g.edges) == 6 * 5 // 2
    assert g.degree_max == 5


def test_complete_one_factorization_round_robin():
    g = complete_graph(6)
    assert g.chi_prime == 5  # n-1 rounds
    union = [tuple(e) for m in g.one_factorization for e in m]
    assert sorted(union) == sorted(g.edges)
    for m in g.one_factorization:
        verts = [q for e in m for q in e]
        assert sorted(verts) == list(range(6))


def test_complete_fresh_sampler_is_perfect_matching():
    g = complete_graph(8)
    rng = np.random.default_rng(1)
    m = g.fresh_matching_sampler(rng)
    verts = [q for e in m for q in e]
    assert sorted(verts) == list(range(8))


# ---- graph6 ----


def test_graph6_known_values():
    # K_1 (no edges) encodes to '@' (N=0 -> chr(63)='?'; here n=1 -> '@').
    assert graph6_encode(1, []) == "@"
    # The 3-cycle (triangle) is the standard graph6 'Bw'.
    assert graph6_encode(3, [(0, 1), (1, 2), (0, 2)]) == "Bw"


def test_graph6_rejects_self_loops():
    with pytest.raises(ValueError):
        graph6_encode(3, [(0, 0)])


def test_graph_carries_graph6():
    g = cycle_graph(4)
    assert isinstance(g.graph6, str) and len(g.graph6) >= 1


# ---- from_spec ----


def test_from_spec_dispatch():
    assert from_spec("cycle", 8).name == "cycle(8)"
    assert from_spec("Complete", 6).name == "complete(6)"  # case-insensitive


def test_from_spec_unknown_raises():
    with pytest.raises(ValueError):
        from_spec("hypercube", 8)


def test_topology_is_plain_dataclass_no_sim_coupling():
    # graphs.py must not import the simulator; a GraphTopology is pure data.
    import sparsegf2.circuits.graphs as gmod

    src = gmod.__file__
    with open(src, encoding="utf-8") as f:
        text = f.read()
    assert "import SparseGF2" not in text
    assert "from sparsegf2 import" not in text
    assert isinstance(cycle_graph(4), GraphTopology)
