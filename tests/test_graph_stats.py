"""graphs.stats and graphs.shortcut_fraction against closed forms."""

import math

import pytest

from sparsegf2 import graphs


def test_cycle():
    n = 10
    s = graphs.stats(graphs.cycle(n), n)
    assert s["lambda2"] == pytest.approx(2 - 2 * math.cos(2 * math.pi / n))
    assert s["connected"] is True
    assert s["mean_path_length"] == pytest.approx(n * n / (4 * (n - 1)))  # even n
    assert s["clustering"] == 0.0
    assert (s["degree_min"], s["degree_mean"], s["degree_max"]) == (2, 2.0, 2)


def test_complete_graph():
    s = graphs.stats(graphs.complete(6), 6)
    assert s["lambda2"] == pytest.approx(6.0)
    assert s["clustering"] == pytest.approx(1.0)
    assert s["mean_path_length"] == pytest.approx(1.0)


def test_ring_lattice_clustering():
    k = 4
    s = graphs.stats(graphs.watts_strogatz(20, k, 0.0, seed=0), 20)
    assert s["clustering"] == pytest.approx(3 * (k - 2) / (4 * (k - 1)))


def test_disconnected_graph():
    edges = graphs.from_edges([(0, 1), (2, 3)], 4)
    s = graphs.stats(edges, 4)
    assert s["connected"] is False
    assert math.isnan(s["mean_path_length"])
    assert s["lambda2"] == pytest.approx(0.0, abs=1e-12)


def test_too_large_is_refused():
    with pytest.raises(ValueError, match="at most 4096"):
        graphs.stats(graphs.cycle(5000), 5000)


def test_shortcut_fraction():
    assert graphs.shortcut_fraction(graphs.watts_strogatz(64, 4, 0.0, seed=1), 64, 4) == 0.0
    f = graphs.shortcut_fraction(graphs.watts_strogatz(64, 4, 1.0, seed=1), 64, 4)
    assert 0.5 < f <= 1.0
    with pytest.raises(ValueError, match="even"):
        graphs.shortcut_fraction(graphs.cycle(8), 8, 3)
