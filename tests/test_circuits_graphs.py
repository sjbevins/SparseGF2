"""Unit tests for sparsegf2.circuits.graphs."""
from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.circuits.graphs import (
    cycle_graph, complete_graph, parse_graph_spec, graph6_encode,
)


# ══════════════════════════════════════════════════════════════
# Basic structure
# ══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_cycle_structure(n):
    g = cycle_graph(n)
    assert g.n == n
    assert len(g.edges) == n
    # Every vertex degree == 2
    deg = np.zeros(n, dtype=int)
    for u, v in g.edges:
        deg[u] += 1
        deg[v] += 1
    assert np.all(deg == 2)


@pytest.mark.parametrize("n", [4, 8, 16])
def test_complete_structure(n):
    g = complete_graph(n)
    assert g.n == n
    assert len(g.edges) == n * (n - 1) // 2
    deg = np.zeros(n, dtype=int)
    for u, v in g.edges:
        deg[u] += 1
        deg[v] += 1
    assert np.all(deg == n - 1)


# ══════════════════════════════════════════════════════════════
# 1-factorization
# ══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("n", [4, 6, 8, 12, 16])
def test_cycle_one_factorization(n):
    g = cycle_graph(n)
    assert g.has_one_factorization
    assert g.chi_prime == 2
    # Each matching covers every vertex exactly once
    for m in g.one_factorization:
        verts = set()
        for u, v in m:
            verts.add(u)
            verts.add(v)
        assert verts == set(range(n))
    # Union of matchings = all edges (each exactly once)
    union = set()
    for m in g.one_factorization:
        for e in m:
            assert e not in union
            union.add(e)
    assert union == set(g.edges)


@pytest.mark.parametrize("n", [4, 6, 8, 12])
def test_complete_one_factorization(n):
    g = complete_graph(n)
    assert g.has_one_factorization
    assert g.chi_prime == n - 1
    # Every matching is perfect
    for m in g.one_factorization:
        assert len(m) == n // 2
        verts = set()
        for u, v in m:
            verts.add(u)
            verts.add(v)
        assert verts == set(range(n))
    # Union = E(K_n)
    union = set()
    for m in g.one_factorization:
        for e in m:
            assert e not in union
            union.add(e)
    assert union == set(g.edges)


@pytest.mark.parametrize("n", [3, 5, 7])
def test_odd_n_no_1factor(n):
    # Cycles and complete graphs of odd n are legal graphs but have no
    # perfect matching, so no 1-factorization and no fresh sampler.
    gc = cycle_graph(n)
    assert gc.one_factorization is None
    assert gc.fresh_matching_sampler is None
    gk = complete_graph(n)
    assert gk.one_factorization is None
    assert gk.fresh_matching_sampler is None


def test_odd_complete_no_pm():
    g = complete_graph(5)
    assert g.one_factorization is None
    assert g.fresh_matching_sampler is None


# ══════════════════════════════════════════════════════════════
# Fresh matching samplers
# ══════════════════════════════════════════════════════════════

def test_complete_fresh_uniform_k4():
    """Chi-squared uniformity check on the 3 perfect matchings of K_4."""
    g = complete_graph(4)
    rng = np.random.default_rng(0)
    counts = {}
    N = 3000
    for _ in range(N):
        m = tuple(sorted(g.fresh_matching_sampler(rng)))
        counts[m] = counts.get(m, 0) + 1
    # K_4 has exactly 3 perfect matchings
    assert len(counts) == 3, f"expected 3 PMs of K_4, saw {len(counts)}"
    expected = N / 3
    chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
    # 5% critical value for chi^2 with df=2 is 5.991
    assert chi2 < 12.0, f"fresh sampler not uniform: chi2={chi2}"


def test_cycle_fresh_two_choices():
    """Fresh sampling on cycle(8) must return one of the 2 perfect matchings."""
    g = cycle_graph(8)
    rng = np.random.default_rng(0)
    palette = set(tuple(sorted(m)) for m in g.one_factorization)
    seen = set()
    for _ in range(200):
        m = tuple(sorted(g.fresh_matching_sampler(rng)))
        assert m in palette
        seen.add(m)
    assert seen == palette


# ══════════════════════════════════════════════════════════════
# graph6 encoding
# ══════════════════════════════════════════════════════════════

def test_graph6_small_cycle():
    # Known graph6 encoding for C_4
    # n=4, edges (0,1), (1,2), (2,3), (0,3)
    # Upper triangle column-major: (0,1), (0,2), (1,2), (0,3), (1,3), (2,3)
    # = 1 0 1 1 0 1 -> pad to 6 bits -> 101101 = 45 -> +63 = 108 -> 'l'
    # N prefix: n=4 -> chr(4+63) = 'C'
    s = graph6_encode(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
    assert s == "Cl"


def test_graph6_rejects_self_loop():
    with pytest.raises(ValueError):
        graph6_encode(3, [(0, 0), (1, 2)])


# ══════════════════════════════════════════════════════════════
# parse_graph_spec
# ══════════════════════════════════════════════════════════════

def test_parse_graph_spec_cycle():
    g = parse_graph_spec("cycle", 16)
    assert g.name == "cycle(16)"


def test_parse_graph_spec_complete():
    g = parse_graph_spec("complete", 8)
    assert g.name == "complete(8)"


def test_parse_graph_spec_unknown():
    with pytest.raises(ValueError):
        parse_graph_spec("pwr2k(2)", 16)   # not in MVP
