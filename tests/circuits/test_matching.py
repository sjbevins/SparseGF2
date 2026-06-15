"""Tests for per-layer matching selection."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.circuits.graphs import complete_graph, cycle_graph
from sparsegf2.circuits.matching import (
    MATCHING_MODES,
    available_modes,
    select_matching,
)


def test_modes_constant():
    assert MATCHING_MODES == ("round_robin", "palette", "fresh")


def test_round_robin_cycles_through_factorization():
    g = cycle_graph(8)
    rng = np.random.default_rng(0)
    chi = g.chi_prime
    m0 = select_matching(g, "round_robin", 0, rng)
    mchi = select_matching(g, "round_robin", chi, rng)  # wraps around
    assert m0 == mchi
    # consecutive layers differ (the two alternating matchings)
    assert select_matching(g, "round_robin", 0, rng) != select_matching(g, "round_robin", 1, rng)


def test_round_robin_draws_no_rng():
    g = cycle_graph(8)
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    select_matching(g, "round_robin", 3, rng)
    after = rng.bit_generator.state
    assert before == after  # round_robin must not consume the RNG


def test_palette_returns_a_factorization_member():
    g = complete_graph(6)
    rng = np.random.default_rng(2)
    fac = [sorted(m) for m in g.one_factorization]
    for _ in range(20):
        m = sorted(select_matching(g, "palette", 0, rng))
        assert m in fac


def test_fresh_returns_perfect_matching():
    g = complete_graph(8)
    rng = np.random.default_rng(3)
    m = select_matching(g, "fresh", 0, rng)
    verts = [q for e in m for q in e]
    assert sorted(verts) == list(range(8))


def test_unknown_mode_raises():
    g = cycle_graph(8)
    with pytest.raises(ValueError):
        select_matching(g, "nope", 0, np.random.default_rng(0))


def test_available_modes():
    g = cycle_graph(8)
    assert set(available_modes(g)) == {"round_robin", "palette", "fresh"}
    odd = cycle_graph(7)  # no 1-factorization, no perfect matching
    assert available_modes(odd) == []
