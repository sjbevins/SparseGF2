"""Graph generators: structure, determinism, and agreement with the standard definitions."""

import networkx as nx
import numpy as np
import pytest

from sparsegf2 import graphs


def _is_normalised(edges, n):
    assert edges.dtype == np.int64 and edges.ndim == 2 and edges.shape[1] == 2
    assert np.all(edges[:, 0] < edges[:, 1])
    assert edges.min() >= 0 and edges.max() < n
    assert len(np.unique(edges, axis=0)) == len(edges)


def test_simple_families():
    assert graphs.cycle(5).tolist() == [[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]]
    assert graphs.path(4).tolist() == [[0, 1], [1, 2], [2, 3]]
    assert len(graphs.complete(7)) == 21
    torus = graphs.lattice_2d(4, 5)
    _is_normalised(torus, 20)
    np.testing.assert_array_equal(graphs.degrees(torus, 20), 4)
    open_grid = graphs.lattice_2d(3, 3, periodic=False)
    assert len(open_grid) == 12


@pytest.mark.parametrize("beta", [0.0, 0.1, 0.5, 1.0])
def test_watts_strogatz(beta):
    n, k = 60, 4
    e = graphs.watts_strogatz(n, k, beta, seed=1)
    _is_normalised(e, n)
    assert len(e) == n * k // 2  # rewiring preserves the edge count
    np.testing.assert_array_equal(e, graphs.watts_strogatz(n, k, beta, seed=1))
    ring = graphs.watts_strogatz(n, k, 0.0, seed=1)
    rewired = len(set(map(tuple, e.tolist())) - set(map(tuple, ring.tolist())))
    if beta == 0:
        assert rewired == 0
    else:
        assert 0.4 * beta * len(e) < rewired < 1.6 * beta * len(e) + 5


def test_watts_strogatz_seeds_differ():
    a = graphs.watts_strogatz(40, 4, 0.3, seed=1)
    b = graphs.watts_strogatz(40, 4, 0.3, seed=2)
    assert not np.array_equal(a, b)


def test_newman_watts_adds_shortcuts():
    n, k = 50, 4
    ring = graphs.newman_watts(n, k, 0.0, seed=0)
    assert len(ring) == n * k // 2
    e = graphs.newman_watts(n, k, 0.3, seed=0)
    _is_normalised(e, n)
    assert set(map(tuple, ring.tolist())) <= set(map(tuple, e.tolist()))
    assert len(e) > len(ring)


def test_erdos_renyi_and_random_regular():
    e = graphs.erdos_renyi(30, 0.2, seed=4)
    _is_normalised(e, 30)
    assert 0.1 * 435 < len(e) < 0.3 * 435
    r = graphs.random_regular(20, 3, seed=5)
    _is_normalised(r, 20)
    np.testing.assert_array_equal(graphs.degrees(r, 20), 3)


@pytest.mark.parametrize(
    "edges, n",
    [
        (graphs.cycle(10), 10),
        (graphs.cycle(9), 9),
        (graphs.complete(8), 8),
        (graphs.lattice_2d(4, 4), 16),
        (graphs.watts_strogatz(30, 4, 0.3, seed=2), 30),
    ],
)
def test_edge_coloring_is_a_partition_into_matchings(edges, n):
    colors = graphs.edge_coloring(edges, n)
    all_idx = np.sort(np.concatenate(colors))
    np.testing.assert_array_equal(all_idx, np.arange(len(edges)))
    for idx in colors:
        nodes = edges[idx].ravel()
        assert len(np.unique(nodes)) == len(nodes)
    assert len(colors) <= 2 * graphs.degrees(edges, n).max() - 1


def test_even_cycle_coloring_is_brickwork():
    colors = graphs.edge_coloring(graphs.cycle(8), 8)
    assert len(colors) == 2
    e = graphs.cycle(8)
    layers = sorted(sorted(map(tuple, e[c].tolist())) for c in colors)
    assert [(0, 1), (2, 3), (4, 5), (6, 7)] in layers


def test_from_edges_normalises_and_validates():
    np.testing.assert_array_equal(graphs.from_edges([(3, 1), (1, 3), (0, 2)]), [[0, 2], [1, 3]])
    with pytest.raises(ValueError, match="self-loops"):
        graphs.from_edges([(1, 1)])
    with pytest.raises(ValueError, match="out of range"):
        graphs.from_edges([(0, 5)], num_nodes=3)
    with pytest.raises(ValueError):
        graphs.cycle(2)
    with pytest.raises(ValueError):
        graphs.watts_strogatz(10, 3, 0.1, seed=0)  # k must be even


def test_from_networkx():
    g = nx.cycle_graph(6)
    np.testing.assert_array_equal(graphs.from_networkx(g), graphs.cycle(6))
    with pytest.raises(ValueError, match="0..n-1"):
        graphs.from_networkx(nx.relabel_nodes(g, {0: 10}))
