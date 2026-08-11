"""Arbitrary-geometry graph constructors: networkx adapter, path, lattice_2d."""

from __future__ import annotations

import pytest

from sparsegf2.circuits import (
    CircuitConfig,
    from_networkx,
    from_spec,
    lattice_2d,
    path_graph,
    simulate,
)
from sparsegf2.circuits.matching import available_modes
from sparsegf2.errors import InvalidArgumentError

nx = pytest.importorskip("networkx")

# (Floor-safety, that importing the package never imports networkx, is checked
#  robustly in a fresh subprocess by tests/test_floor_safety.py.)


# ---- path ----


def test_path_basic():
    p = path_graph(8)
    assert len(p.edges) == 7
    assert p.chi_prime == 2  # even / odd bond classes
    assert "round_robin" in available_modes(p)
    assert p.has_perfect_matching  # even n -> unique PM


def test_path_odd_no_pm():
    p = path_graph(7)
    assert not p.has_perfect_matching  # odd n
    assert p.has_one_factorization  # still has the bond classes for brickwork


@pytest.mark.parametrize("n", [4, 8, 12])
def test_path_brickwork_runs(n):
    cfg = CircuitConfig(graph_spec="path", n=n, gating_mode="brickwork", p=0.2, depth_factor=3)
    rec = simulate(cfg, sample_seed=0)
    assert rec.total_layers > 0


# ---- lattice ----


def test_lattice_square_from_spec():
    L = from_spec("lattice_2d", 9)
    assert L.name == "lattice_2d(3x3)"
    assert L.n == 9
    assert len(L.edges) == 12


def test_lattice_rectangular():
    L = lattice_2d(2, 4)
    assert L.n == 8
    assert L.has_perfect_matching  # cols even -> a PM exists


def test_lattice_nonsquare_from_spec_raises():
    with pytest.raises(InvalidArgumentError):
        from_spec("lattice_2d", 6)  # 6 is not a perfect square


def test_lattice_brickwork_runs():
    L = lattice_2d(4, 4)
    cfg = CircuitConfig(
        graph_spec=L, n=16, picture="purification", gating_mode="brickwork", p=0.2, depth_factor=2
    )
    rec = simulate(cfg, sample_seed=0, analyses=["code_dimension"])
    assert rec.total_layers > 0


# ---- from_networkx ----


def test_from_networkx_regular_has_factorization():
    g = nx.random_regular_graph(4, 16, seed=1)
    t = from_networkx(g)
    assert t.n == 16
    assert t.has_one_factorization  # edge coloring always available
    assert t.has_perfect_matching  # regular graph with even n


def test_from_networkx_no_perfect_matching():
    g = nx.star_graph(5)  # 6 nodes, no perfect matching
    t = from_networkx(g)
    assert not t.has_perfect_matching
    assert "fresh" not in available_modes(t)
    # random_edge still works (edges only)
    assert t.edges


def test_from_networkx_relabels_nonint_nodes():
    g = nx.grid_2d_graph(2, 3)  # nodes are (r, c) tuples
    t = from_networkx(g)
    assert t.n == 6
    assert all(isinstance(u, int) and isinstance(v, int) for u, v in t.edges)
    assert all(0 <= u < 6 and 0 <= v < 6 for u, v in t.edges)


@pytest.mark.parametrize("mode", ["round_robin", "palette"])
def test_from_networkx_brickwork_runs(mode):
    t = from_networkx(nx.random_regular_graph(3, 12, seed=2))
    cfg = CircuitConfig(
        graph_spec=t,
        n=12,
        picture="purification",
        gating_mode="brickwork",
        matching_mode=mode,
        p=0.2,
        depth_factor=3,
    )
    rec = simulate(cfg, sample_seed=0, analyses=["code_dimension"])
    assert rec.total_layers > 0


def test_nx_graph_as_graph_spec():
    cfg = CircuitConfig(graph_spec=nx.cycle_graph(10), n=10, p=0.2, depth_factor=2)
    assert cfg.n == 10
    rec = simulate(cfg, sample_seed=0)
    assert rec.total_layers > 0


def test_nx_graph_n_mismatch_raises():
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(graph_spec=nx.cycle_graph(10), n=8, p=0.2)


def test_from_networkx_rejects_directed():
    with pytest.raises(InvalidArgumentError):
        from_networkx(nx.DiGraph([(0, 1), (1, 2)]))


def test_from_networkx_rejects_multigraph():
    with pytest.raises(InvalidArgumentError):
        from_networkx(nx.MultiGraph([(0, 1), (0, 1)]))


def test_from_networkx_rejects_self_loops():
    g = nx.Graph([(0, 1), (1, 2)])
    g.add_edge(0, 0)
    with pytest.raises(InvalidArgumentError):
        from_networkx(g)


def test_from_networkx_fresh_is_perfect_matching():
    import numpy as np

    from sparsegf2.circuits.matching import select_matching

    t = from_networkx(nx.random_regular_graph(4, 12, seed=3))
    assert t.has_perfect_matching
    rng = np.random.default_rng(0)
    for _ in range(5):
        m = select_matching(t, "fresh", 0, rng)
        covered = sorted(q for e in m for q in e)
        assert covered == list(range(12))  # a perfect matching covers every vertex


def test_from_networkx_brickwork_fresh_runs():
    t = from_networkx(nx.random_regular_graph(4, 12, seed=4))
    cfg = CircuitConfig(
        graph_spec=t,
        n=12,
        picture="purification",
        gating_mode="brickwork",
        matching_mode="fresh",
        p=0.2,
        depth_factor=3,
    )
    assert simulate(cfg, sample_seed=0).total_layers > 0


def test_from_networkx_edgeless():
    t = from_networkx(nx.empty_graph(4))
    assert t.edges == []
    assert t.one_factorization is None
    assert not t.has_perfect_matching


@pytest.mark.parametrize("mode", ["brickwork", "random_edge", "random_pool", "all_edges"])
def test_edgeless_graph_rejected(mode):
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(graph_spec=nx.empty_graph(4), n=4, gating_mode=mode, p=0.2)


def test_path_single_edge():
    p = path_graph(2)
    assert p.edges == [(0, 1)]
    assert p.chi_prime == 1
    assert p.has_perfect_matching


def test_lattice_single_edge():
    L = lattice_2d(1, 2)
    assert L.n == 2
    assert L.edges == [(0, 1)]
    assert L.has_perfect_matching
