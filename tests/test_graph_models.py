"""Graph families checked against their models and against networkx.

Statistical tests use fixed seeds and accept a difference up to ``SIGMAS`` = 4.5
standard errors; the standard error is computed in each test from the samples
(unbiased variances) and stated in a comment. Expected values come from networkx,
numpy or closed forms, never from the function under test.
"""

import math

import networkx as nx
import numpy as np
import pytest

from sparsegf2 import graphs

SIGMAS = 4.5


def _simple(edges, n):
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    assert np.all(e[:, 0] != e[:, 1]), "self-loop"
    assert e.min() >= 0 and e.max() < n
    pairs = {tuple(sorted(p)) for p in e.tolist()}
    assert len(pairs) == len(e), "duplicate edge"
    return pairs


def _nx(edges, n):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(np.asarray(edges).tolist())
    return g


def _within_sigma(x, y):
    """|mean(x) - mean(y)| < 4.5 pooled standard errors, se = sqrt(var_x/N_x + var_y/N_y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    se = math.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    diff = x.mean() - y.mean()
    assert se > 0
    assert abs(diff) < SIGMAS * se, f"diff {diff:.5f}, se {se:.5f}, z {diff / se:.2f}"


def _gamma_q(a, x):
    """Regularized upper incomplete gamma Q(a, x) (series / continued fraction)."""
    if x <= 0:
        return 1.0
    lg = math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        ap = a
        for _ in range(2000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - lg)
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - lg) * h


def _chi2_sf(x, df):
    return _gamma_q(df / 2, x / 2)


def test_chi2_helper_matches_closed_forms():
    # df = 2: survival function exp(-x/2); df = 1: erfc(sqrt(x/2))
    for x in (0.5, 3.0, 10.0, 40.0):
        assert _chi2_sf(x, 2) == pytest.approx(math.exp(-x / 2), rel=1e-10)
        assert _chi2_sf(x, 1) == pytest.approx(math.erfc(math.sqrt(x / 2)), rel=1e-10)


def _two_sample_chi2_p(counts_a, counts_b, min_total=20):
    """Two-sample chi-square test of equal distributions; sparse bins pooled into one."""
    a, b = np.asarray(counts_a, float), np.asarray(counts_b, float)
    keep = (a + b) >= min_total
    a = np.append(a[keep], a[~keep].sum())
    b = np.append(b[keep], b[~keep].sum())
    nz = (a + b) > 0
    a, b = a[nz], b[nz]
    na, nb = a.sum(), b.sum()
    ea = (a + b) * na / (na + nb)
    eb = (a + b) * nb / (na + nb)
    stat = ((a - ea) ** 2 / ea).sum() + ((b - eb) ** 2 / eb).sum()
    return _chi2_sf(stat, len(a) - 1), stat, len(a) - 1


# ----------------------------------------------------------------------------
# Watts-Strogatz
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.0, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("n", [12, 101])
@pytest.mark.parametrize("k", [2, 4, 6])
def test_watts_strogatz_structure(n, k, beta):
    for seed in range(5):
        edges = graphs.watts_strogatz(n, k, beta, seed=seed)
        assert len(_simple(edges, n)) == n * k // 2
        # each node keeps its own k/2 forward edges in the networkx algorithm
        assert graphs.degrees(edges, n).min() >= k // 2


@pytest.mark.parametrize("n", [12, 101])
@pytest.mark.parametrize("k", [2, 4, 6])
def test_watts_strogatz_beta_zero_is_the_ring_lattice(n, k):
    edges = graphs.watts_strogatz(n, k, 0.0, seed=3)
    expected = {tuple(sorted((i, (i + j) % n))) for i in range(n) for j in range(1, k // 2 + 1)}
    assert _simple(edges, n) == expected
    np.testing.assert_array_equal(graphs.degrees(edges, n), k)
    assert graphs.shortcut_fraction(edges, n, k) == 0.0


def test_watts_strogatz_matches_networkx_ensemble():
    n, k, beta, trials = 100, 4, 0.2, 300
    ours, theirs = [], []
    for t in range(trials):
        ours.append(graphs.watts_strogatz(n, k, beta, seed=1000 + t))
        theirs.append(
            np.array(
                sorted(
                    tuple(sorted(e)) for e in nx.watts_strogatz_graph(n, k, beta, seed=t).edges()
                )
            )
        )
    # mean shortcut fraction: |difference| < 4.5 pooled standard errors (se from 300 + 300 values)
    _within_sigma(
        [graphs.shortcut_fraction(e, n, k) for e in ours],
        [graphs.shortcut_fraction(e, n, k) for e in theirs],
    )
    # mean average clustering: same criterion, networkx.average_clustering for both
    _within_sigma(
        [nx.average_clustering(_nx(e, n)) for e in ours],
        [nx.average_clustering(_nx(e, n)) for e in theirs],
    )
    # degree histograms: two-sample chi-square, p > 1e-6
    hist_a = np.bincount(np.concatenate([graphs.degrees(e, n) for e in ours]), minlength=30)
    hist_b = np.bincount(np.concatenate([graphs.degrees(e, n) for e in theirs]), minlength=30)
    p, stat, df = _two_sample_chi2_p(hist_a, hist_b)
    assert p > 1e-6, f"chi2 {stat:.1f} on {df} dof, p = {p:.3g}"


@pytest.mark.parametrize("extra", [1, 2])
@pytest.mark.parametrize("beta", [0.0, 0.3, 1.0])
def test_watts_strogatz_n_close_to_k(extra, beta):
    k = 4
    n = k + extra
    for seed in range(20):
        edges = graphs.watts_strogatz(n, k, beta, seed=seed)
        assert len(_simple(edges, n)) == n * k // 2
        assert graphs.degrees(edges, n).min() >= k // 2
        if extra == 1:  # n = k + 1: the complete graph, no free target to rewire to
            assert len(edges) == n * (n - 1) // 2


def test_watts_strogatz_odd_k_and_k_at_least_n_are_rejected():
    # The docstring says k is even; the code also requires 2 <= k < n, which the docstring
    # does not state (reported as a concern). Current behavior: ValueError.
    for n, k in [(10, 3), (10, 5), (10, 1), (10, 0), (6, 6), (6, 8), (4, 4)]:
        with pytest.raises(ValueError, match="k must be even"):
            graphs.watts_strogatz(n, k, 0.1, seed=0)
        with pytest.raises(ValueError, match="k must be even"):
            graphs.newman_watts(n, k, 0.1, seed=0)
    with pytest.raises(ValueError, match="beta"):
        graphs.watts_strogatz(10, 4, 1.5, seed=0)


# ----------------------------------------------------------------------------
# Other families
# ----------------------------------------------------------------------------


def test_random_regular_is_regular_simple_and_clusters_like_networkx():
    n, d, trials = 40, 3, 200
    ours, theirs = [], []
    for t in range(trials):
        e = graphs.random_regular(n, d, seed=t)
        assert len(_simple(e, n)) == n * d // 2
        np.testing.assert_array_equal(graphs.degrees(e, n), d)
        ours.append(nx.average_clustering(_nx(e, n)))
        theirs.append(nx.average_clustering(nx.random_regular_graph(d, n, seed=t)))
    # se = sqrt(var_ours/200 + var_nx/200) of the mean clustering; band 4.5 se
    _within_sigma(ours, theirs)


def test_newman_watts_mean_edge_count():
    n, k, p, trials = 100, 4, 0.1, 300
    counts = [len(graphs.newman_watts(n, k, p, seed=t)) for t in range(trials)]
    # Each of the n k / 2 ring edges adds one shortcut with probability p (a free target
    # always exists at this density), so E[edges] = n k / 2 (1 + p).
    expected = n * k / 2 * (1 + p)
    se = np.std(counts, ddof=1) / math.sqrt(trials)  # sample standard error of the mean
    assert abs(np.mean(counts) - expected) < SIGMAS * se
    ring = {tuple(sorted((i, (i + j) % n))) for i in range(n) for j in (1, 2)}
    assert ring <= _simple(graphs.newman_watts(n, k, p, seed=0), n)


def test_newman_watts_p_zero_is_the_ring_lattice():
    n, k = 30, 4
    assert _simple(graphs.newman_watts(n, k, 0.0, seed=1), n) == _simple(
        graphs.watts_strogatz(n, k, 0.0, seed=1), n
    )


def test_erdos_renyi_mean_edge_count():
    n, p, trials = 60, 0.15, 300
    counts = [len(graphs.erdos_renyi(n, p, seed=t)) for t in range(trials)]
    pairs = n * (n - 1) // 2
    expected = pairs * p
    # Binomial(pairs, p): exact standard error of the mean of 300 draws
    se = math.sqrt(pairs * p * (1 - p) / trials)
    assert abs(np.mean(counts) - expected) < SIGMAS * se
    for t in range(5):
        _simple(graphs.erdos_renyi(n, p, seed=t), n)
    assert len(graphs.erdos_renyi(n, 0.0, seed=0)) == 0
    assert len(graphs.erdos_renyi(n, 1.0, seed=0)) == pairs


def test_lattice_2d_degrees():
    n = 4 * 5
    torus = graphs.lattice_2d(4, 5)
    assert len(_simple(torus, n)) == 2 * n
    np.testing.assert_array_equal(graphs.degrees(torus, n), 4)
    # open boundaries: corners 2, other boundary sites 3, interior 4 (degrees from networkx)
    rows, cols = 4, 5
    open_edges = graphs.lattice_2d(rows, cols, periodic=False)
    grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(rows, cols), ordering="sorted")
    np.testing.assert_array_equal(graphs.degrees(open_edges, n), [grid.degree(v) for v in range(n)])
    assert sorted(set(graphs.degrees(open_edges, n).tolist())) == [2, 3, 4]
    assert len(open_edges) == rows * (cols - 1) + cols * (rows - 1)
    # a periodic direction of length 2 does not wrap, so no duplicate edge appears
    thin = graphs.lattice_2d(2, 5)
    _simple(thin, 10)
    np.testing.assert_array_equal(graphs.degrees(thin, 10), 3)


# ----------------------------------------------------------------------------
# graphs.stats against independent computations
# ----------------------------------------------------------------------------

_TWO_TRIANGLES = np.array([[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5]])  # disconnected
_ISOLATED = np.array([[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]])  # node 5 isolated

STATS_CASES = [
    ("ring", graphs.cycle(12), 12),
    ("path", graphs.path(9), 9),
    ("complete", graphs.complete(7), 7),
    ("torus", graphs.lattice_2d(4, 5), 20),
    ("open grid", graphs.lattice_2d(3, 4, periodic=False), 12),
    ("ws", graphs.watts_strogatz(40, 4, 0.2, seed=5), 40),
    ("ws dense", graphs.watts_strogatz(30, 6, 0.5, seed=6), 30),
    ("regular", graphs.random_regular(24, 3, seed=7), 24),
    ("two triangles", _TWO_TRIANGLES, 6),
    ("isolated node", _ISOLATED, 6),
]


@pytest.mark.parametrize("name,edges,n", STATS_CASES, ids=[c[0] for c in STATS_CASES])
def test_stats_against_networkx_and_numpy(name, edges, n):
    st = graphs.stats(edges, n)
    g = _nx(edges, n)
    deg = np.array([g.degree(v) for v in range(n)])
    assert st["degree_min"] == deg.min()
    assert st["degree_max"] == deg.max()
    assert st["degree_mean"] == pytest.approx(deg.mean())
    # docstring: average local clustering coefficient, nodes of degree < 2 counting 0,
    # which is networkx.average_clustering (not transitivity)
    assert st["clustering"] == pytest.approx(nx.average_clustering(g))
    # Laplacian built in the test
    adj = nx.to_numpy_array(g, nodelist=range(n))
    lap = np.diag(adj.sum(axis=1)) - adj
    lam2 = np.linalg.eigvalsh(lap)[1]
    connected = nx.is_connected(g)
    assert st["connected"] == connected
    if connected:
        assert st["lambda2"] == pytest.approx(lam2, abs=1e-9)
        assert st["mean_path_length"] == pytest.approx(nx.average_shortest_path_length(g))
        assert st["lambda2"] > 1e-9
    else:
        assert st["lambda2"] < 1e-12  # Task 18 tightens this to == 0.0
        assert abs(lam2) < 1e-9
        assert math.isnan(st["mean_path_length"])


def test_stats_cases_include_disconnected_and_isolated():
    assert len(STATS_CASES) == 10
    st = graphs.stats(_ISOLATED, 6)
    assert st["degree_min"] == 0 and not st["connected"]
    assert not graphs.stats(_TWO_TRIANGLES, 6)["connected"]
