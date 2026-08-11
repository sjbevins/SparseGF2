"""Watts-Strogatz **rewired** circulant C(n, k): construction invariants,
reproducibility, graph6 round-trip, spec parsing, and circuit integration.

The rewiring *moves* a lattice edge (keep the near endpoint, replace the far
endpoint with a uniform current non-neighbor), so the edge count |E| = n*k
and the mean degree 2k are preserved for every beta, but the degree *sequence*
is not (classic Watts-Strogatz). These tests therefore assert |E| and mean
degree, never per-vertex degree for beta > 0.

The core edge / graph6 tests use no networkx and must run without it; only the
graph6-decode round-trip and the perfect-matching-sampler paths need it, and
those are guarded with ``pytest.importorskip("networkx")``.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.circuits import CircuitConfig, simulate
from sparsegf2.circuits.graphs import from_spec, watts_strogatz
from sparsegf2.errors import InvalidArgumentError

# ---------------------------------------------------------------- helpers


def _base_circulant_edges(n: int, k: int) -> set[tuple[int, int]]:
    """The exact base circulant C(n, k) edge set: each vertex to (i +/- c) for
    c = 1..k, canonicalized to sorted (u, v) with u < v."""
    edges: set[tuple[int, int]] = set()
    for c in range(1, k + 1):
        for i in range(n):
            j = (i + c) % n
            edges.add((min(i, j), max(i, j)))
    return edges


def _degrees(n: int, edges) -> np.ndarray:
    """Per-vertex degree array from an edge list."""
    d = np.zeros(n, dtype=int)
    for u, v in edges:
        d[u] += 1
        d[v] += 1
    return d


# ---------------------------------------------------------------- beta = 0


@pytest.mark.parametrize("n,k", [(16, 2), (32, 3)])
def test_beta_zero_is_exact_circulant(n, k):
    g = watts_strogatz(n, k=k, beta=0.0, seed=0)
    base = _base_circulant_edges(n, k)
    # beta = 0 must not move any edge: exactly the base circulant.
    assert set(g.edges) == base
    # every vertex has degree 2k, and |E| = n*k.
    deg = _degrees(n, g.edges)
    assert np.all(deg == 2 * k)
    assert len(g.edges) == n * k


def test_beta_zero_independent_of_seed():
    # No rewiring happens, so the seed is irrelevant at beta = 0.
    a = watts_strogatz(24, k=2, beta=0.0, seed=1)
    b = watts_strogatz(24, k=2, beta=0.0, seed=999)
    assert a.edges == b.edges == sorted(_base_circulant_edges(24, 2))


# ---------------------------------------------------- edge-set invariants


@pytest.mark.parametrize("beta", [0.1, 0.3, 0.5, 0.8, 1.0])
@pytest.mark.parametrize("n,k", [(32, 2), (24, 3)])
def test_edge_set_invariants(beta, n, k):
    g = watts_strogatz(n, k=k, beta=beta, seed=7)
    edges = g.edges
    # edge count (hence mean degree 2k) is preserved for every beta.
    assert len(edges) == n * k
    assert _degrees(n, edges).sum() == 2 * n * k  # sum of degrees = 2|E|
    # no self-loops.
    assert all(u != v for u, v in edges)
    # every edge is sorted (u < v) and canonicalized.
    assert all(u < v for u, v in edges)
    # no duplicate edges.
    assert len(set(edges)) == len(edges)
    # graphs.py returns a sorted edge list.
    assert edges == sorted(edges)


def test_degree_sequence_not_preserved_at_high_beta():
    # Documented behavior: beta > 0 moves far endpoints, so the degree
    # sequence is NOT the flat 2k of the base circulant (only the mean is).
    g = watts_strogatz(64, k=2, beta=1.0, seed=3)
    deg = _degrees(g.n, g.edges)
    assert deg.mean() == 2 * 2  # mean degree 2k stays exactly 4
    assert not np.all(deg == 4)  # but the sequence spreads away from flat 2k


# --------------------------------------------------- rewired fraction ~ beta


def test_rewired_fraction_tracks_beta():
    # Over many seeds, the mean fraction of edges that moved off the base
    # circulant should approximate beta (each lattice edge rewires w.p. beta).
    n, k, beta = 128, 2, 0.2
    base = _base_circulant_edges(n, k)
    n_edges = n * k
    fractions = []
    for seed in range(40):
        g = watts_strogatz(n, k=k, beta=beta, seed=seed)
        moved = sum(1 for e in g.edges if e not in base)
        fractions.append(moved / n_edges)
    mean_frac = float(np.mean(fractions))
    assert abs(mean_frac - beta) < 0.03


# --------------------------------------------- reproducibility & distinctness


def test_reproducible_from_params():
    a = watts_strogatz(64, k=2, beta=0.3, seed=5)
    b = watts_strogatz(64, k=2, beta=0.3, seed=5)
    assert a.graph6 == b.graph6
    assert a.edges == b.edges


def test_distinct_seeds_give_distinct_graphs():
    graph6s = {watts_strogatz(128, k=2, beta=0.4, seed=s).graph6 for s in range(10)}
    # 10 independent realizations are almost surely all distinct.
    assert len(graph6s) == 10


def test_different_beta_or_k_changes_graph6():
    ref = watts_strogatz(64, k=2, beta=0.3, seed=0).graph6
    assert watts_strogatz(64, k=2, beta=0.6, seed=0).graph6 != ref
    assert watts_strogatz(64, k=3, beta=0.3, seed=0).graph6 != ref


# --------------------------------------------------------- graph6 round-trip


def test_graph6_round_trip_matches_edges():
    nx = pytest.importorskip("networkx")
    g = watts_strogatz(32, k=2, beta=0.5, seed=11)
    decoded = nx.from_graph6_bytes(g.graph6.encode("ascii"))
    decoded_edges = sorted((min(u, v), max(u, v)) for u, v in decoded.edges())
    assert decoded.number_of_nodes() == g.n
    assert decoded_edges == sorted(g.edges)


# ------------------------------------------------- GraphTopology attributes


def test_topology_attributes():
    n = 40
    g = watts_strogatz(n, k=2, beta=0.5, seed=2)
    assert g.n == n
    assert g.is_stochastic is True
    assert g.one_factorization is None  # stochastic / non-Class-1
    assert g.has_one_factorization is False
    assert isinstance(g.graph6, str) and len(g.graph6) >= 1
    # the auto-name carries the parameters for provenance.
    assert "watts_strogatz" in g.name
    for token in ("n=40", "k=2", "beta=0.5", "seed=2"):
        assert token in g.name


def test_name_override():
    g = watts_strogatz(16, k=2, beta=0.5, seed=0, name="my_ws")
    assert g.name == "my_ws"


# ------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 2},  # n < 3
        {"n": 16, "k": 0},  # k < 1
        {"n": 16, "k": 8},  # 2*k == n (need 2k < n)
        {"n": 16, "k": 20},  # 2*k > n
        {"n": 16, "beta": -0.1},  # beta below range
        {"n": 16, "beta": 1.5},  # beta above range
    ],
)
def test_validation_raises(kwargs):
    with pytest.raises(InvalidArgumentError):
        watts_strogatz(**kwargs)


# ------------------------------------------------------------- from_spec


def test_from_spec_with_inline_params():
    n, seed = 32, 4
    g = from_spec("watts_strogatz(k=2,beta=0.1)", n, seed)
    direct = watts_strogatz(n, k=2, beta=0.1, seed=seed)
    assert g.edges == direct.edges
    assert g.graph6 == direct.graph6


def test_from_spec_bare_uses_defaults():
    n, seed = 32, 4
    g = from_spec("watts_strogatz", n, seed)
    # bare spec must equal the documented defaults k=2, beta=0.5.
    default = watts_strogatz(n, k=2, beta=0.5, seed=seed)
    assert g.edges == default.edges
    assert g.graph6 == default.graph6


def test_from_spec_is_case_insensitive():
    n, seed = 24, 1
    g = from_spec("Watts_Strogatz(k=2,beta=0.2)", n, seed)
    assert g.edges == watts_strogatz(n, k=2, beta=0.2, seed=seed).edges


@pytest.mark.parametrize(
    "spec",
    [
        "watts_strogatz(k=2,beta=0.1",  # missing closing paren
        "watts_strogatz(beta)",  # no '=' in a parameter
        "watts_strogatz(beta=oops)",  # non-numeric value
    ],
)
def test_from_spec_malformed_raises(spec):
    with pytest.raises(InvalidArgumentError):
        from_spec(spec, 16, 0)


@pytest.mark.parametrize(
    "spec",
    [
        "watts_strogatz(p=0.5)",  # newman_watts key on watts_strogatz
        "watts_strogatz(gamma=1)",  # nonsense key
        "newman_watts(beta=0.1)",  # watts_strogatz key on newman_watts
    ],
)
def test_from_spec_unknown_param_raises(spec):
    # unknown parameter keys must surface as the package error, not a raw TypeError.
    with pytest.raises(InvalidArgumentError):
        from_spec(spec, 16, 0)


def test_from_spec_newman_watts_inline_params():
    pytest.importorskip("networkx")

    from sparsegf2.circuits.graphs import newman_watts

    n, seed = 32, 4
    g = from_spec("newman_watts(k=4,p=0.2)", n, seed)
    direct = newman_watts(n, k=4, p=0.2, seed=seed)
    assert g.edges == direct.edges
    assert g.graph6 == direct.graph6


# ---------------------------------------------------------- circuit integration


def test_simulate_records_analyses_and_graph6():
    n = 24
    cfg = CircuitConfig(
        graph_spec=watts_strogatz(n, 2, 0.3, seed=0),
        n=n,
        picture="single_ref",
        gating_mode="random_pool",
        p=0.2,
        depth_factor=2,
        base_seed=0,
    )
    rec = simulate(cfg, sample_seed=0, analyses=["ref_entropy", "half_cut_entropy"])
    # the requested analyses are present on the record.
    assert rec.analyses is not None
    assert "ref_entropy" in rec.analyses
    assert "half_cut_entropy" in rec.analyses
    # a stochastic graph records its graph6 for provenance.
    assert rec.graph6 is not None
    assert rec.graph_name is not None and "watts_strogatz" in rec.graph_name


def test_deterministic_graph_has_no_recorded_graph6():
    # Contrast: a deterministic graph (cycle) does not record a graph6 string.
    n = 24
    cfg = CircuitConfig(
        graph_spec="cycle",
        n=n,
        picture="single_ref",
        gating_mode="random_pool",
        p=0.2,
        depth_factor=2,
        base_seed=0,
    )
    rec = simulate(cfg, sample_seed=0, analyses=["ref_entropy", "half_cut_entropy"])
    assert rec.graph6 is None
