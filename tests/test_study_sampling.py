"""Conformance of the Watts–Strogatz study's graph, gate and measurement sampling (v1.4 §2–§4, §7).

Two kinds of checks, on the exact study protocol (``ExperimentManifest.monitored_circuit_kwargs``):

* **Exact** (deterministic): the native generator equals an independent reference rewiring
  trace for trace; the exported circuits equal both the protocol oracle and a direct
  transcription of the draws; masks are ``uniform < float(p)`` and nested across rates;
  prefixes do not depend on the horizon; the native Clifford numbering is ``16 s + b``.
* **Statistical**, at fixed seeds, with the false-alarm rate stated: every chi-square test
  passes when its p-value exceeds ``ALPHA = 1e-6``, every z-test when ``|z| < SIGMAS = 4.5``
  (two-sided false alarm 6.8e-6). Roughly 84 such assertions run here (about 40 chi-square
  tests and 44 z-tests), so a *correct* implementation at fresh seeds would fail one with
  probability below 4e-4; at the fixed seeds the outcome is deterministic. Statistics use exact
  reference laws: the occupancy law of with-replacement edge draws, Binomial(2n, beta)
  rewiring counts, and the exact distribution of the rewiring process for n = 6 and 7 (a
  dynamic program with no random numbers). The sensitivity of each family is shown by negative
  controls (a wrong beta, a biased target sampler, a biased Clifford sample, a wrong rate, and
  seeds that are shared across sizes or rewiring probabilities must fail).

Every graph and every circuit below is drawn from the study's own seeds (ruling R-SEED:
``identity.graph_seed(master, n, beta, g)`` and ``identity.run_seed(master, n, beta, mode, p)``),
not from ad hoc seeds, so the conformance tests exercise the seed derivation too: the
collections of different ``(n, beta)`` are independent samples, and only the rates of one
``(n, beta)`` are coupled (nested masks).

What the existing tests already prove, and is not repeated here:

* ``test_protocol.py``: for 20 specs (matching, random_edges, all_edges, weights, scrambles,
  banks) the exported circuit equals the oracle, and the laws of the draws (uniform unweighted
  edges, 64-bucket Clifford indices, per-qubit measurement rates, the layer shapes).
* ``test_mipt.py::test_measurement_draws_give_common_random_numbers_across_rates``: nesting at
  one seed on one cycle.
* ``test_invariance.py``: chunk, thread-count and replay invariance; appending graphs keeps
  the earlier trajectories.
* ``test_randomness_contract.py``: the pinned digests of the streams.

This file adds the study protocol itself (manifest keywords, even and odd n), the full
11,520-index and 720-class uniformity with the check of ``j // 16`` against ``clifford.py``,
the with-replacement law, independence diagnostics, mask correlations, nesting over a grid of
rates, horizon and probe independence, ``gate_mode="independent"``, the graph law, and the
independence of the samples of different sizes and rewiring probabilities.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import _reference_rng as ref
import _ws_reference as wsref
from _protocol_oracle import oracle_circuit_text
from sparsegf2 import graphs, mipt
from sparsegf2.clifford import Clifford, symplectic_group
from sparsegf2.study import identity as ident
from sparsegf2.study.identity import Token, build_graph
from sparsegf2.study.manifest import ExperimentManifest

ALPHA = 1e-6  # chi-square tests: p must exceed this
SIGMAS = 4.5  # z-tests: |z| must stay below this

MASTER_GRAPH = 17
MASTER_CIRCUIT = 29
PROPERTY = settings(max_examples=200, deadline=None, database=None, derandomize=True)


def graph_seed(n: int, beta, g: int) -> tuple[int, ...]:
    """The study's seed of graph ``g`` of ``(n, beta)`` under the master graph seed (ruling R-SEED)."""
    return ident.graph_seed(MASTER_GRAPH, n, beta, g)


def make_manifest(
    n: int,
    ps: list[str],
    layers: int,
    stride: int,
    *,
    beta: str = "0.3",
    graphs_count: int = 1,
    gate_mode: str = "shared",
) -> ExperimentManifest:
    """A manifest with one condition block: the study protocol at size ``n``."""
    return ExperimentManifest.from_json(
        {
            "experiment_id": "conformance",
            "conditions": [{"n": [n], "beta": [beta], "p": ps}],
            "depth": {"layers": {"fixed": layers}, "checkpoints": {"count": 3}},
            "trials": {"graphs": graphs_count, "circuits_per_graph": stride},
            "seeds": {
                "master_graph_seed": MASTER_GRAPH,
                "master_circuit_seed": MASTER_CIRCUIT,
                "gate_mode": gate_mode,
            },
            "observables": {"uniform_erasure": None, "correlated_erasure": []},
        }
    )


def make_spec(
    manifest: ExperimentManifest, n: int, p: str, *, beta: str = "0.3", probes=()
) -> tuple[mipt.MonitoredCircuit, list[ident.GraphRecord]]:
    """The study's ``MonitoredCircuit`` for one condition, on the manifest's graph collection."""
    records = [build_graph(n, beta, g, MASTER_GRAPH) for g in range(manifest.trials.graphs)]
    kwargs = manifest.monitored_circuit_kwargs(n, beta, p)
    return mipt.MonitoredCircuit(**kwargs, edges=[r.edges for r in records], probes=probes), records


class Sample:
    """The gates and masks of ``count`` trajectories of one spec, as arrays."""

    def __init__(self, spec, records, seed, us):
        n = spec.num_system
        h = n // 2
        t = spec.layers
        edge_id = [{tuple(e): i for i, e in enumerate(r.edges.tolist())} for r in records]
        self.n, self.h, self.layers, self.us = n, h, t, list(us)
        self.edge = np.zeros((len(self.us), t, h), dtype=np.int64)
        self.clifford = np.zeros_like(self.edge)
        self.mask = np.zeros((len(self.us), t, n), dtype=bool)
        self.touch = np.zeros((len(self.us), t, n), dtype=np.int64)  # gates touching each site
        for k, u in enumerate(self.us):
            graph = edge_id[u // spec.circuits_per_graph]
            layers = wsref.parse_circuit(str(spec.circuit(u, seed=seed)))
            assert len(layers) == t
            for layer_no, layer in enumerate(layers):
                assert len(layer.gates) == h
                for a, (x, y, c) in enumerate(layer.gates):
                    assert x < y  # low endpoint first: random_orientation is off
                    self.edge[k, layer_no, a] = graph[(x, y)]
                    self.clifford[k, layer_no, a] = c
                    self.touch[k, layer_no, x] += 1
                    self.touch[k, layer_no, y] += 1
                self.mask[k, layer_no, layer.measured] = True


# ---------------------------------------------------------------------------
# the statistics helpers
# ---------------------------------------------------------------------------


def test_chi_square_tail_matches_closed_forms():
    # even degrees of freedom: exp(-x/2) sum_{i < dof/2} (x/2)^i / i!
    for dof in range(2, 42, 2):
        for x in (0.3, 4.0, dof, 3.0 * dof, 80.0):
            h = x / 2
            exact = math.exp(-h) * sum(h**i / math.factorial(i) for i in range(dof // 2))
            assert wsref.chi2_sf(x, dof) == pytest.approx(exact, rel=1e-9, abs=1e-300)
    for x in (0.5, 2.0, 9.0, 30.0):
        assert wsref.chi2_sf(x, 1) == pytest.approx(math.erfc(math.sqrt(x / 2)), rel=1e-9)
        three = math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
        assert wsref.chi2_sf(x, 3) == pytest.approx(three, rel=1e-9)
    # published 5% critical values
    for x, dof in [(3.841459, 1), (5.991465, 2), (7.814728, 3), (82.528727, 63)]:
        assert wsref.chi2_sf(x, dof) == pytest.approx(0.05, rel=1e-5)
    assert wsref.chi2_sf(0.0, 7) == 1.0


@pytest.mark.parametrize("dof", [719, 4000, 11519])
def test_chi_square_tail_at_the_degrees_of_freedom_of_the_large_tests(dof):
    # no closed form here: compare with the Wilson-Hilferty approximation, which is accurate to
    # about 1e-3 relative around the 1e-3 tail and to about 1% around 5e-6, the levels that matter
    for k in (-2.0, 0.0, 1.0, 2.0, 3.0, 4.75):
        x = dof + k * math.sqrt(2 * dof)
        z = ((x / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
        approx = 0.5 * math.erfc(z / math.sqrt(2))
        assert wsref.chi2_sf(x, dof) == pytest.approx(approx, rel=0.02)


def test_chi_square_helper_pools_small_cells_and_detects_bias():
    # the three small cells (expected 3 in all) join the smaller of the two others: [13, 12] vs [14, 11]
    stat, dof, p = wsref.chi2_p([10, 12, 1, 1, 1], [11, 11, 1, 1, 1], min_expected=5)
    assert dof == 1 and stat == pytest.approx(1 / 14 + 1 / 11)
    with pytest.raises(ValueError, match="same shape and total"):
        wsref.chi2_p([1, 2], [2, 2])
    rng = np.random.default_rng(5)
    fair = rng.multinomial(100_000, np.full(40, 1 / 40))
    assert wsref.uniform_p(fair)[2] > ALPHA
    biased = rng.multinomial(100_000, np.array([1.1] * 20 + [0.9] * 20) / 40)
    assert wsref.uniform_p(biased)[2] < ALPHA  # a 10% bias in every cell is seen at once


# ---------------------------------------------------------------------------
# graphs: the reference against the native generator
# ---------------------------------------------------------------------------

SIZES = list(range(5, 41))  # odd and even n
BETAS = [0.0, 0.05, 0.3, 1.0]


def test_the_reference_rewiring_equals_the_native_generator_trace_for_trace():
    # v1.4 section 7.4, "Graph": the native generator equals an independent implementation
    # on fixed seeds. 36 sizes x 4 rewiring probabilities x 60 seeds = 8640 graphs.
    count = 0
    for n, beta in itertools.product(SIZES, BETAS):
        for g in range(60):
            seed = graph_seed(n, beta, g)  # the study's six-word seed of graph g of (n, beta)
            expected = wsref.rewire(n, beta, seed)
            assert np.array_equal(expected, graphs.watts_strogatz(n, 4, beta, seed=seed)), (
                n,
                beta,
                g,
            )
            count += 1
    assert count == 8640


@pytest.mark.parametrize(
    "seed",
    [
        0,
        1,
        2**32 - 1,
        2**64 + 5,
        [3, 1, 4],
        (0, 0),
        (2**32 - 1, 2**32 - 1),
        (ident.GRAPH_SEED_TAG, 17, 16, 2007713733, 3507754741, 3),  # a graph seed of the study
        (ident.CIRCUIT_SEED_TAG, 29, 16, 2007713733, 3507754741),  # a shared run seed: five words
        (ident.CIRCUIT_SEED_TAG, 29, 16, 2007713733, 3507754741, 3479488031, 1099992840),  # seven
    ],
)
def test_equality_holds_for_every_seed_form(seed):
    for n, beta in [(9, 0.4), (16, 0.2), (31, 0.7)]:
        assert np.array_equal(
            wsref.rewire(n, beta, seed), graphs.watts_strogatz(n, 4, beta, seed=seed)
        )
    spawned = np.random.SeedSequence(5).spawn(2)[1]
    assert np.array_equal(
        wsref.rewire(12, 0.5, spawned), graphs.watts_strogatz(12, 4, 0.5, seed=spawned)
    )


@PROPERTY
@given(
    st.integers(5, 60),
    st.sampled_from([0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]),
    st.integers(0, 2**32 - 1),
    st.integers(0, 2**32 - 1),
)
def test_equality_and_validity_for_arbitrary_seeds(n, beta, master, g):
    record = build_graph(n, beta, g, master)  # validated: simple, 2n edges, beta = 0 gives C(n, 2)
    seed = wsref.graph_seed(master, n, Token.parse(beta).token, g)  # the reference's own seed tuple
    assert record.seed == seed
    assert np.array_equal(record.edges, wsref.rewire(n, beta, seed))


def test_the_mapping_from_random_numbers_to_targets_matters_not_only_the_law():
    # v1.4 section 2.2: a target "uniform on A_j". Drawing next_below(|A_j|) and indexing the
    # sorted A_j has that law too, but another trace: the graphs differ from one seed, so a
    # graph record replays only under the mapping of its generator version.
    n, beta, seeds = 24, 0.5, 200
    same = sum(
        np.array_equal(
            wsref.rewire(n, beta, graph_seed(n, beta, g), mapping="index"),
            graphs.watts_strogatz(n, 4, beta, seed=graph_seed(n, beta, g)),
        )
        for g in range(seeds)
    )
    assert same <= 2  # about 24 rewirings per graph: equal only by luck
    assert all(  # with no rewiring there is no target to map
        np.array_equal(
            wsref.rewire(n, 0.0, graph_seed(n, 0.0, g), mapping="index"),
            graphs.watts_strogatz(n, 4, 0.0, seed=graph_seed(n, 0.0, g)),
        )
        for g in range(5)
    )
    # ... while the law is the same: the index mapping passes the same uniformity test below
    visits: list[wsref.Visit] = []
    for g in range(3000):
        wsref.rewire(n, beta, graph_seed(n, beta, g), mapping="index", record=visits)
    assert target_pit_p(visits) > ALPHA


@pytest.mark.parametrize("n", SIZES)
def test_graph_properties_hold_at_every_size_and_rewiring_probability(n):
    c0 = {tuple(e) for e in ident.circulant_edges(n).tolist()}
    assert c0 == {tuple(sorted(e)) for e in wsref.circulant_edges(n)}
    for beta in ("0", "0.05", "0.3", "1"):
        for g in range(8):
            record = build_graph(n, beta, g, MASTER_GRAPH)  # validates simple, 2n edges
            edges = record.edges.tolist()
            assert len({tuple(e) for e in edges}) == len(edges) == 2 * n  # no loop, no repeat
            assert all(a < b for a, b in edges)
            if beta == "0":
                assert {tuple(e) for e in edges} == c0  # beta = 0 gives C(n, 2)
            if n == 5:  # nothing to rewire to: K_5 at every beta
                assert len(edges) == 10 and {tuple(e) for e in edges} == c0
            degrees = np.bincount(np.array(edges).ravel(), minlength=n)
            assert degrees.min() >= 2 and degrees.sum() == 4 * n


def test_c_n_2_is_the_complete_graph_up_to_five_vertices_and_a_ring_after():
    for n in range(1, 6):
        assert len(wsref.circulant_edges(n)) == n * (n - 1) // 2
    for n in range(5, 41):
        edges = wsref.circulant_edges(n)
        assert len(edges) == 2 * n
        assert set(np.bincount(np.array([sorted(e) for e in edges]).ravel(), minlength=n)) == {4}


# ---------------------------------------------------------------------------
# graphs: the abort on an absent original edge is unreachable (v1.4 section 2.2)
# ---------------------------------------------------------------------------


def test_the_abort_fires_on_a_corrupted_order_and_never_on_the_real_one():
    order = wsref.original_edge_list(9)
    broken = order[:1] + order[:1] + order[2:]  # the first edge twice
    with pytest.raises(wsref.AbsentOriginalEdge, match="visit 1: the original edge"):
        wsref.rewire(9, 1.0, (MASTER_GRAPH, 0), order=broken)  # rewired at visit 0, absent at 1
    # with beta = 0 nothing is deleted, so even the broken order finds every edge present
    wsref.rewire(9, 0.0, (MASTER_GRAPH, 0), order=broken)
    # n = 4: the offset-major list repeats {0, 2} and {1, 3} (8 oriented edges, 6 pairs), but K_4
    # has no rewiring target, so nothing is ever deleted and no visited edge is absent
    listed = [(i, (i + d) % 4) for d in (1, 2) for i in range(4)]
    assert len(listed) == 8 and len({frozenset(e) for e in listed}) == 6
    wsref.rewire(4, 1.0, (MASTER_GRAPH, 0), order=listed)


@pytest.mark.parametrize("n", [5, 6, 7])
def test_no_reachable_state_lacks_the_visited_edge(n):
    # A computational proof for small n: the dynamic program visits every state that has
    # positive probability (at beta = 1/2 every branch is possible) and raises on an absent edge.
    law = wsref.exact_law(n, 0.5)
    assert sum(law.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(bin(mask).count("1") == 2 * n for mask in law)  # every reachable graph has 2n edges


# ---------------------------------------------------------------------------
# graphs: the exact law at n = 6 and n = 7
# ---------------------------------------------------------------------------


def test_the_exact_law_is_a_probability_distribution_with_the_expected_support():
    for n in (5, 6):
        for beta in (Fraction(1, 4), Fraction(1, 2), Fraction(1)):
            law = wsref.exact_law(n, beta, exact=True)  # exact rational arithmetic
            assert sum(law.values()) == 1 and all(q > 0 for q in law.values())
    assert len(wsref.exact_law(5, 0.5)) == 1  # K_5 stays K_5
    assert len(wsref.exact_law(6, 0.0)) == 1  # beta = 0: the circulant graph only
    assert len(wsref.exact_law(6, 0.5)) == 428 and len(wsref.exact_law(7, 0.5)) == 101_593
    law = wsref.exact_law(6, Fraction(1, 2), exact=True)
    # every graph keeps 2n edges, so the expected degree is exactly 4: sum_g mu(g) deg_g(v) = 4
    expected_degree = [
        sum(q * np.bincount(wsref.mask_edges(m, 6).ravel(), minlength=6)[v] for m, q in law.items())
        for v in range(6)
    ]
    assert all(d > 0 for d in expected_degree) and sum(expected_degree) == 24


@pytest.mark.parametrize("beta", [0.25, 0.5])
def test_the_native_generator_has_the_exact_law_at_n_6(beta):
    # Pearson chi-square over every graph the rewiring can produce (428 of them at n = 6):
    # N = 100,000 graphs, expected counts N mu(g), cells below 5 pooled. beta is dyadic, so the
    # float64 threshold realizes the Bernoulli(beta) law exactly.
    n, total = 6, 100_000
    law = wsref.exact_law(n, beta)
    counts: dict[int, int] = {}
    for g in range(total):
        mask = wsref.graph_mask(graphs.watts_strogatz(n, 4, beta, seed=graph_seed(n, beta, g)), n)
        counts[mask] = counts.get(mask, 0) + 1
    assert set(counts) <= set(law)  # never a graph outside the support
    keys = sorted(law)
    observed = np.array([counts.get(k, 0) for k in keys])
    expected = np.array([total * law[k] for k in keys])
    _, dof, p = wsref.chi2_p(observed, expected)
    assert dof > 300 and p > ALPHA
    # sensitivity: the same counts reject the law of a slightly different beta
    wrong = wsref.exact_law(n, beta + 0.05)
    wrong_expected = np.array([total * wrong.get(k, 0.0) for k in keys])
    assert wsref.chi2_p(observed, wrong_expected)[2] < ALPHA


@pytest.mark.parametrize("misreading", wsref.MISREADINGS)
def test_each_plausible_misreading_of_the_rewiring_rule_is_rejected(misreading):
    # v1.4 section 2.2 is easy to misread in three ways it warns about: the eligible set must be
    # taken before the deletion, the order is offset-major, the retained endpoint is a. Each
    # misreading gives a different law at n = 6 (graphs outside the support, or other
    # probabilities), which the exact-law test sees with 30,000 graphs; the correct reading,
    # on the same seeds, passes.
    n, beta, total = 6, 0.5, 30_000
    law = wsref.exact_law(n, beta)
    seeds = [graph_seed(n, beta, g) for g in range(total)]
    wrong = [wsref.graph_mask(wsref.rewire_misreading(n, beta, s, misreading), n) for s in seeds]
    assert wsref.law_p(wrong, law) < ALPHA
    right = [wsref.graph_mask(wsref.rewire(n, beta, s), n) for s in seeds]
    assert wsref.law_p(right, law) > ALPHA


@pytest.mark.parametrize("beta", [0.25, 0.5, 1.0])
def test_the_native_generator_has_the_exact_moments_at_n_7(beta):
    # 101,593 graphs can occur at n = 7: too many for cell counts, so compare the exact mean of
    # statistics with the sample mean, in standard errors computed from the exact variance.
    n, total = 7, 25_000
    law = wsref.exact_law(n, beta)
    degvar = np.array([wsref.degree_variance(wsref.mask_edges(m, n), n) for m in law])
    new = np.array([wsref.new_edge_count(wsref.mask_edges(m, n), n) for m in law], dtype=float)
    q = np.array(list(law.values()))
    assert q.sum() == pytest.approx(1.0, abs=1e-12)
    sample = [graphs.watts_strogatz(n, 4, beta, seed=graph_seed(n, beta, g)) for g in range(total)]
    for values, stat in [(degvar, wsref.degree_variance), (new, wsref.new_edge_count)]:
        mean = float(q @ values)
        var = float(q @ (values - mean) ** 2)
        got = np.array([stat(e, n) for e in sample], dtype=float)
        z = (got.mean() - mean) / math.sqrt(var / total)
        assert abs(z) < SIGMAS, (stat.__name__, mean, got.mean(), z)
        assert 0.9 < got.var() / var < 1.1  # the sample variance matches the exact one


# ---------------------------------------------------------------------------
# graphs: decisions and targets at a size where the law cannot be enumerated
# ---------------------------------------------------------------------------


def target_pit_p(visits: list[wsref.Visit], seed: int = 123) -> float:
    """p-value of the uniformity of the targets, by the randomized probability integral transform.

    For an event with eligible set of size s and the target at position r of the sorted set,
    u = (r + V) / s with V uniform from an independent generator. Given the past, the target is
    uniform on the eligible set, so u is uniform on (0, 1) and independent of the past: the
    sequence of u is exactly i.i.d. uniform, however the sizes s vary. 20 bins, chi-square.
    """
    rng = np.random.default_rng(seed)
    u = [
        (v.eligible.index(v.target) + rng.random()) / len(v.eligible)
        for v in visits
        if v.target is not None and v.eligible is not None
    ]
    bins = np.minimum((np.array(u) * 20).astype(int), 19)
    return wsref.uniform_p(np.bincount(bins, minlength=20))[2]


@pytest.fixture(scope="module", params=[0.1, 0.5])
def rewiring_trace(request):
    """20,000 graphs at n = 24 with every decision recorded; each equals the native graph."""
    beta, n, total = request.param, 24, 20_000
    records, attempts = [], np.zeros(total, dtype=int)
    for g in range(total):
        visits: list[wsref.Visit] = []
        edges = wsref.rewire(n, beta, graph_seed(n, beta, g), record=visits)
        if g < 2000:  # tie the recorded decisions to the native generator
            assert np.array_equal(
                edges, graphs.watts_strogatz(n, 4, beta, seed=graph_seed(n, beta, g))
            )
        records.append(visits)
        attempts[g] = sum(v.attempted for v in visits)
    return beta, n, records, attempts


def test_rewiring_decisions_are_bernoulli_beta(rewiring_trace):
    beta, n, records, attempts = rewiring_trace
    total, m = len(records), 2 * n
    # (a) the number of attempts is Binomial(2n G, beta); E = 2 n beta per graph
    z = (attempts.sum() - total * m * beta) / math.sqrt(total * m * beta * (1 - beta))
    assert abs(z) < SIGMAS
    # (b) per graph, Binomial(2n, beta): chi-square on the histogram
    _, dof, p = wsref.chi2_p(
        np.bincount(attempts, minlength=m + 1), total * wsref.binomial_pmf(m, beta)
    )
    assert dof >= 8 and p > ALPHA
    # (c) each of the 2n visit positions attempts with probability beta: sum of z^2 ~ chi2(2n)
    by_position = np.array([[v.attempted for v in visits] for visits in records], dtype=float).sum(
        axis=0
    )
    zs = (by_position - total * beta) / math.sqrt(total * beta * (1 - beta))
    assert abs(zs).max() < SIGMAS and wsref.chi2_sf(float(zs @ zs), m) > ALPHA
    # (d) no serial correlation between consecutive decisions
    b = np.array([[v.attempted for v in visits] for visits in records], dtype=float) - beta
    lag = float(np.sum(b[:, :-1] * b[:, 1:])) / (beta * (1 - beta))
    assert abs(lag / math.sqrt(total * (m - 1))) < SIGMAS
    # sensitivity: a wrong beta fails (b)
    wrong = wsref.chi2_p(
        np.bincount(attempts, minlength=m + 1), total * wsref.binomial_pmf(m, beta + 0.02)
    )
    assert wrong[2] < ALPHA


def test_targets_are_uniform_on_the_eligible_set_computed_before_the_deletion(rewiring_trace):
    beta, n, records, _ = rewiring_trace
    # (a) the first rewiring of each graph happens on the untouched circulant graph, where the
    # eligible set has exactly n - 5 vertices: the position of the target is uniform on n - 5
    first = []
    for visits in records:
        event = next((v for v in visits if v.target is not None), None)
        if event is not None:
            assert event.eligible is not None and len(event.eligible) == n - 5
            first.append(event.eligible.index(event.target))
            assert (
                event.b not in event.eligible and event.a not in event.eligible
            )  # N(a) u {a} excluded
    assert len(first) > 19_000
    assert wsref.uniform_p(np.bincount(first, minlength=n - 5))[2] > ALPHA
    # (b) over every event, whatever the size of the set: the randomized PIT is i.i.d. uniform
    flat = [v for visits in records for v in visits]
    assert sum(v.target is not None for v in flat) > 100_000 * beta
    assert target_pit_p(flat) > ALPHA
    # (c) the eligible set is always V minus the retained endpoint and its neighbors, so the new
    # edge is never a loop or a repeat: the target differs from a and from the old endpoint b
    assert all(v.target not in (v.a, v.b) for v in flat if v.target is not None)


def test_rejection_sampling_uses_n_over_the_eligible_size_draws_on_average(rewiring_trace):
    # rejecting next_below(n) until the value is eligible accepts with probability q = |A| / n, so the
    # number of draws is geometric with mean 1 / q and variance (1 - q) / q^2
    _, n, records, _ = rewiring_trace
    events = [v for visits in records for v in visits if v.target is not None]
    assert events
    q = np.array([len(v.eligible) / n for v in events])
    draws = np.array([v.draws for v in events])
    mean, var = float(np.sum(1 / q)), float(np.sum((1 - q) / q**2))
    assert abs(draws.sum() - mean) / math.sqrt(var) < SIGMAS
    assert draws.min() >= 1


def test_a_biased_target_sampler_fails_the_uniformity_test():
    visits: list[wsref.Visit] = []
    for g in range(3000):
        wsref.rewire(24, 0.5, graph_seed(24, 0.5, g), mapping="first", record=visits)
    assert target_pit_p(visits) < ALPHA  # always the smallest eligible vertex: seen at once


# ---------------------------------------------------------------------------
# circuits: the study protocol, exactly
# ---------------------------------------------------------------------------

SPEC_CASES = [
    (n, p, mode)
    for n in (5, 6, 9, 16, 17)
    for p in ("0.1", "0.4")
    for mode in ("shared", "independent")
]


@pytest.mark.parametrize("n, p, mode", SPEC_CASES)
def test_the_exported_circuit_is_the_oracle_and_the_direct_transcription(n, p, mode):
    manifest = make_manifest(n, [p], 12, 4, graphs_count=2, gate_mode=mode)
    spec, records = make_spec(manifest, n, p)
    seed = manifest.run_seed(n, "0.3", p)  # the manifest's beta is 0.3
    assert seed == wsref.run_seed(MASTER_CIRCUIT, n, "0.3", None if mode == "shared" else p)
    assert len(seed) == (5 if mode == "shared" else 7)
    for u in (0, 3, 4, 7):  # graphs 0 and 1, circuits 0 and 3
        g, c = divmod(u, manifest.stride)
        assert u == manifest.stream_index(g, c)
        text = str(spec.circuit(u, seed=seed))
        assert text == oracle_circuit_text(spec, u, seed)  # the protocol oracle (test_protocol)
        draws = wsref.study_draws(seed, u, len(records[g].edges), n, 12)  # v1.4 section 7.3
        for t, layer in enumerate(wsref.parse_circuit(text)):
            assert len(layer.gates) == n // 2  # h(n) = floor(n / 2), odd n included
            for a, (x, y, j) in enumerate(layer.gates):
                assert (x, y) == tuple(records[g].edges[draws.edge_index[t, a]])  # next_below(|E|)
                assert j == draws.clifford[t, a]  # next_below(11520)
            threshold = Token.parse(p).float64
            assert layer.measured == [i for i in range(n) if draws.uniform[t, i] < threshold]


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9, 10, 11, 16, 17, 24, 25])
def test_each_layer_has_floor_n_over_two_gates_on_edges_with_the_low_endpoint_first(n):
    manifest = make_manifest(n, ["0.2"], 25, 3)
    spec, records = make_spec(manifest, n, "0.2")
    assert spec.gates_per_layer == n // 2
    allowed = {tuple(e) for e in records[0].edges.tolist()}
    seed = manifest.run_seed(n, "0.3", "0.2")
    for u in range(3):
        layers = wsref.parse_circuit(str(spec.circuit(u, seed=seed)))
        assert len(layers) == 25
        for layer in layers:
            assert len(layer.gates) == n // 2
            assert all((x, y) in allowed and x < y and 0 <= j < 11520 for x, y, j in layer.gates)
            assert layer.measured == sorted(set(layer.measured)) and set(layer.measured) <= set(
                range(n)
            )


def test_the_circuit_follows_the_graph_edges_only():
    # an edge outside the graph is never drawn, however many gates are drawn
    manifest = make_manifest(20, ["0.1"], 100, 20, beta="0.5")
    spec, records = make_spec(manifest, 20, "0.1", beta="0.5")
    allowed = {tuple(e) for e in records[0].edges.tolist()}
    used = set()
    seed = manifest.run_seed(20, "0.5", "0.1")
    for u in range(20):
        for layer in wsref.parse_circuit(str(spec.circuit(u, seed=seed))):
            used |= {(x, y) for x, y, _ in layer.gates}
    assert used == allowed  # 20 x 100 x 10 draws reach all 40 edges


# ---------------------------------------------------------------------------
# circuits: the laws of the edge and Clifford draws
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def big():
    """n = 24, p = 0.3: 72 trajectories x 400 layers x 12 gates = 345,600 gates."""
    n, layers, k = 24, 400, 72
    manifest = make_manifest(n, ["0.3"], layers, k)
    spec, records = make_spec(manifest, n, "0.3")
    return Sample(spec, records, manifest.run_seed(n, "0.3", "0.3"), range(k)), records[0]


def test_edge_draws_are_uniform_over_the_graphs_edges(big):
    sample, record = big
    num_edges = len(record.edges)  # 48
    assert sample.edge.size == 345_600
    # pooled over positions and layers: 7200 expected per edge
    assert wsref.uniform_p(np.bincount(sample.edge.ravel(), minlength=num_edges))[2] > ALPHA
    # each of the 12 positions alone, combined as a sum of independent chi-squares
    total_stat, total_dof = 0.0, 0
    for a in range(sample.h):
        stat, dof, _ = wsref.uniform_p(
            np.bincount(sample.edge[:, :, a].ravel(), minlength=num_edges)
        )
        total_stat, total_dof = total_stat + stat, total_dof + dof
    assert wsref.chi2_sf(total_stat, total_dof) > ALPHA
    # the vertex marginal: an endpoint with probability deg(v) / |E| (v1.4 section 3)
    degree = np.bincount(record.edges.ravel(), minlength=sample.n)
    draws = sample.edge.size
    for v in range(sample.n):
        prob = degree[v] / num_edges
        hits = int(sample.touch[:, :, v].sum())
        assert abs(hits - draws * prob) < SIGMAS * math.sqrt(draws * prob * (1 - prob)), v


def occupancy_pmf(num_cells: int, draws: int) -> list[Fraction]:
    """Exact law of the number of distinct cells hit by ``draws`` uniform draws with replacement."""
    law = {0: Fraction(1)}
    for _ in range(draws):
        nxt: dict[int, Fraction] = {}
        for d, q in law.items():
            nxt[d] = nxt.get(d, 0) + q * Fraction(d, num_cells)
            nxt[d + 1] = nxt.get(d + 1, 0) + q * Fraction(num_cells - d, num_cells)
        law = nxt
    return [law.get(d, Fraction(0)) for d in range(draws + 1)]


def test_edges_are_drawn_with_replacement_in_sampled_order(big):
    sample, record = big
    num_edges, h = len(record.edges), sample.h
    layers = sample.edge.reshape(-1, h)
    distinct = np.array([len(set(row)) for row in layers.tolist()])
    law = occupancy_pmf(num_edges, h)  # P(D = d) for 12 draws among 48 edges
    assert float(law[h]) == pytest.approx(0.2231, abs=1e-4)  # 22% of the layers have no repeat
    assert (distinct < h).mean() > 0.7  # repeated edges are common: without replacement, never
    expected = np.array([float(q) for q in law]) * len(layers)
    _, dof, p = wsref.chi2_p(np.bincount(distinct, minlength=h + 1), expected)
    assert dof >= 3 and p > ALPHA
    # draws in one layer are independent of each other: the first two collide with probability 1/|E|
    same = int((layers[:, 0] == layers[:, 1]).sum())
    prob = 1 / num_edges
    assert abs(same - len(layers) * prob) < SIGMAS * math.sqrt(len(layers) * prob * (1 - prob))
    # and across layers: the last edge of layer t and the first of layer t + 1
    last, first = sample.edge[:, :-1, -1], sample.edge[:, 1:, 0]
    same_next = int((last == first).sum())
    pairs = last.size
    assert abs(same_next - pairs * prob) < SIGMAS * math.sqrt(pairs * prob * (1 - prob))


def test_disjoint_pairs_of_edge_draws_are_independent_and_uniform():
    # n = 8: 16 edges, 4 gates per layer; (e0, e1) and (e2, e3) are disjoint pairs, 256 cells
    n, layers, k = 8, 300, 100
    manifest = make_manifest(n, ["0.2"], layers, k)
    spec, records = make_spec(manifest, n, "0.2")
    sample = Sample(spec, records, manifest.run_seed(n, "0.3", "0.2"), range(k))
    num_edges = len(records[0].edges)
    assert num_edges == 16
    cells = np.concatenate(
        [
            (sample.edge[:, :, 0] * num_edges + sample.edge[:, :, 1]).ravel(),
            (sample.edge[:, :, 2] * num_edges + sample.edge[:, :, 3]).ravel(),
        ]
    )
    counts = np.bincount(cells, minlength=num_edges**2)
    assert counts.sum() == 60_000 and wsref.uniform_p(counts)[2] > ALPHA  # 234 per cell, 255 dof
    # the Clifford bucket (8 buckets of 1440) is independent of the edge of the same gate
    joint = np.bincount(
        (sample.edge * 8 + sample.clifford // 1440).ravel(), minlength=num_edges * 8
    )
    assert wsref.uniform_p(joint)[2] > ALPHA


def test_clifford_indices_are_uniform_over_11520_and_their_classes_over_720(big):
    sample, _ = big
    index = sample.clifford.ravel()
    assert index.size == 345_600 and index.min() >= 0 and index.max() < 11520
    # 30 expected per index: chi-square with 11,519 degrees of freedom
    assert wsref.uniform_p(np.bincount(index, minlength=11520))[2] > ALPHA
    # the symplectic class s = j // 16: 480 expected per class, 719 degrees of freedom
    assert wsref.uniform_p(np.bincount(index // 16, minlength=720))[2] > ALPHA
    # the four image-sign bits b = j % 16: 21,600 expected, 15 degrees of freedom
    assert wsref.uniform_p(np.bincount(index % 16, minlength=16))[2] > ALPHA
    # class and sign bits independent: 8 class buckets x 16 signs
    joint = np.bincount((index // 1440) * 16 + index % 16, minlength=8 * 16)
    assert wsref.uniform_p(joint)[2] > ALPHA
    # consecutive draws in one layer are independent: 8 x 8 buckets of disjoint pairs
    first, second = sample.clifford[:, :, 0::2], sample.clifford[:, :, 1::2]
    pairs = np.bincount(((first // 1440) * 8 + second // 1440).ravel(), minlength=64)
    assert wsref.uniform_p(pairs)[2] > ALPHA


def test_the_clifford_test_has_the_power_to_see_a_bias():
    # the sample size above (30 per index) cannot see a 2% bias but sees a 5% bias in every class
    rng = np.random.default_rng(7)
    prob = np.where(np.arange(720) % 2 == 0, 1.05, 0.95)
    biased = rng.multinomial(345_600, prob / prob.sum())
    assert wsref.uniform_p(biased)[2] < ALPHA
    fair = rng.multinomial(345_600, np.full(720, 1 / 720))
    assert wsref.uniform_p(fair)[2] > ALPHA


def test_j_over_16_is_the_symplectic_class_and_j_mod_16_the_sign_bits():
    # v1.4 section 9.1, against clifford.py and an independent enumeration of Sp(4, F_2):
    # every binary 4 x 4 matrix M with M Omega M^T = Omega (mod 2), sorted by the integer whose
    # binary digits are the entries row by row, first entry most significant
    omega = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    shifts = np.arange(15, -1, -1)
    ints = np.arange(1 << 16)
    mats = ((ints[:, None] >> shifts[None, :]) & 1).reshape(-1, 4, 4)
    product = np.einsum("aij,jk,alk->ail", mats, omega, mats) % 2
    group = mats[np.all(product == omega, axis=(1, 2))]
    assert len(group) == 720 and len(group) * 16 == 11520  # |Sp(4, 2)| = 720, 16 sign patterns
    assert np.array_equal(group, symplectic_group(2))  # the native list, same order
    counts = np.zeros(720, dtype=int)
    for j in range(11520):
        cliff = Clifford.from_index(2, j)
        s, b = divmod(j, 16)
        assert np.array_equal(cliff.symplectic_matrix(), group[s])
        assert cliff.sign_bits().tolist() == [(b >> k) & 1 for k in range(4)]  # X0, X1, Z0, Z1
        assert cliff.index() == j
        counts[s] += 1
    assert set(counts.tolist()) == {16}  # 16 preimages of each of the 720 classes


# ---------------------------------------------------------------------------
# circuits: the mask law
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def masks():
    """n = 12, p = 0.3: 40 trajectories x 200 layers, with the gates drawn in the same layers."""
    n, layers, k, p = 12, 200, 40, 0.3
    manifest = make_manifest(n, ["0.3"], layers, k)
    spec, records = make_spec(manifest, n, "0.3")
    return Sample(spec, records, manifest.run_seed(n, "0.3", "0.3"), range(k)), p


def frequency_pvalues(mask: np.ndarray, p: float) -> dict[str, float]:
    """Per-site frequencies against Bernoulli(p): the largest |z| (as a p-value) and the chi-square.

    ``mask`` is ``(trajectories, layers, sites)``. The count of a site is Binomial(K T, p), the
    sites are independent, so the sum of the squared z-scores is chi-square with n degrees of
    freedom.
    """
    k, layers, n = mask.shape
    count = k * layers
    z = (mask.sum(axis=(0, 1)) - count * p) / math.sqrt(count * p * (1 - p))
    return {
        "largest site": min(1.0, n * math.erfc(float(abs(z).max()) / math.sqrt(2))),
        "sites together": wsref.chi2_sf(float(z @ z), n),
    }


def correlation_pvalues(mask: np.ndarray, p: float) -> dict[str, float]:
    """p-values of the correlations a Bernoulli(p) mask must not have.

    With ``Y = mask - p`` the products ``Y_a Y_b`` of two different positions have mean 0 and
    variance ``(p (1 - p))^2``, and products of different pairs are uncorrelated, so each sum
    over the sample, scaled, is standard normal and the squares of a family add up to a
    chi-square. Families: two sites of one layer; one site, layers ``lag`` apart; two sites in
    consecutive layers; one site and layer in neighboring trajectories.
    """
    y = mask.astype(float) - p
    k, layers, n = y.shape
    scale = p * (1 - p)
    out = {}
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    z = np.array([(y[:, :, i] * y[:, :, j]).sum() for i, j in pairs]) / (
        math.sqrt(k * layers) * scale
    )
    out["site pairs"] = wsref.chi2_sf(float(z @ z), len(pairs))
    for lag in (1, 2, 3):
        terms = (y[:, :-lag, :] * y[:, lag:, :]).sum(axis=(0, 1))
        zs = terms / (math.sqrt(k * (layers - lag)) * scale)
        out[f"layer lag {lag}"] = wsref.chi2_sf(float(zs @ zs), n)
    cross = np.array(
        [(y[:, :-1, i] * y[:, 1:, j]).sum() for i in range(n) for j in range(n) if i != j]
    ) / (math.sqrt(k * (layers - 1)) * scale)
    out["sites in consecutive layers"] = wsref.chi2_sf(float(cross @ cross), len(cross))
    between = (y[:-1] * y[1:]).sum(axis=(0, 1)) / (math.sqrt((k - 1) * layers) * scale)
    out["neighboring trajectories"] = wsref.chi2_sf(float(between @ between), n)
    return out


def gate_dependence_pvalues(mask: np.ndarray, touch: np.ndarray, p: float) -> dict[str, float]:
    """p-values of the dependence of the mask on the gates drawn in the layers t - 1, t and t + 1.

    Given the gates, the masks are independent Bernoulli(p), so for any weights ``c`` that are a
    function of the gates (here: whether the site is touched by a gate, centered)
    ``z_i = sum Y c / sqrt(p (1 - p) sum c^2)`` is standard normal, and the ``z_i`` of different
    sites are independent: the sum of squares is chi-square with n degrees of freedom.
    """
    y = mask.astype(float) - p
    k, layers, n = y.shape
    scale = p * (1 - p)
    touched = (touch > 0).astype(float)
    out = {}
    for lag in (-1, 0, 1):
        lo, hi = max(0, -lag), layers - max(0, lag)
        weights = touched[:, lo + lag : hi + lag, :]
        weights = weights - weights.mean(axis=(0, 1))
        z = (y[:, lo:hi, :] * weights).sum(axis=(0, 1)) / np.sqrt(
            scale * (weights**2).sum(axis=(0, 1))
        )
        out[f"touched, layer offset {lag}"] = wsref.chi2_sf(float(z @ z), n)
    return out


def test_masks_are_independent_bernoulli_p_at_every_site(masks):
    sample, p = masks
    for name, value in frequency_pvalues(sample.mask, p).items():
        assert value > ALPHA, name
    count = sample.mask.shape[0] * sample.mask.shape[1]  # 8000 layers
    total = sample.mask.sum()
    assert abs(total - count * sample.n * p) < SIGMAS * math.sqrt(count * sample.n * p * (1 - p))
    # a wrong rate is seen: at p = 0.31 the total count would be 6.8 sigma away
    wrong = 0.31
    assert abs(total - count * sample.n * wrong) > SIGMAS * math.sqrt(
        count * sample.n * wrong * (1 - wrong)
    )
    # the per-site tests see a larger error (each site is 10 sigma away at p = 0.35)
    assert max(frequency_pvalues(sample.mask, 0.35).values()) < ALPHA


def test_masks_have_no_site_or_layer_correlation_beyond_chance(masks):
    sample, p = masks
    values = correlation_pvalues(sample.mask, p)
    assert len(values) == 6
    for name, value in values.items():
        assert value > ALPHA, name


def test_masks_are_independent_of_the_gate_draws(masks):
    sample, p = masks
    values = gate_dependence_pvalues(sample.mask, sample.touch, p)
    for name, value in values.items():
        assert value > ALPHA, name
    # the measured count of a layer against the mean Clifford class and the mean edge of its
    # gates, and the gates of the next layer against this layer's mask (the stream reads the
    # mask uniforms just before them): a correlation r of N independent layers is N(0, 1 / N)
    count = sample.mask.sum(axis=2)
    pairs = {
        "class": (count.ravel(), (sample.clifford // 16).mean(axis=2).ravel()),
        "edge": (count.ravel(), sample.edge.mean(axis=2).ravel()),
        "next class": (count[:, :-1].ravel(), (sample.clifford // 16).mean(axis=2)[:, 1:].ravel()),
        "next edge": (count[:, :-1].ravel(), sample.edge.mean(axis=2)[:, 1:].ravel()),
    }
    for name, (a, b) in pairs.items():
        r = np.corrcoef(a, b)[0, 1]
        assert abs(r) * math.sqrt(a.size) < SIGMAS, (name, r)


def correlated_masks(kind: str, shape=(40, 200, 12), p=0.3, share=0.2, seed=11) -> np.ndarray:
    """Bernoulli(p) masks with a planted dependence (negative controls): with probability
    ``share`` a position copies its neighbor (the previous site, or the previous layer)."""
    rng = np.random.default_rng(seed)
    k, layers, n = shape
    mask = rng.random(shape) < p
    copy = rng.random(shape) < share
    if kind == "site":
        for i in range(1, n):
            mask[:, :, i] = np.where(copy[:, :, i], mask[:, :, i - 1], mask[:, :, i])
    elif kind == "layer":
        for t in range(1, layers):
            mask[:, t] = np.where(copy[:, t], mask[:, t - 1], mask[:, t])
    return mask


def test_the_correlation_tests_have_the_power_to_see_planted_dependence():
    clean = correlated_masks("none")
    assert min(correlation_pvalues(clean, 0.3).values()) > ALPHA
    assert min(frequency_pvalues(clean, 0.3).values()) > ALPHA
    sites = correlation_pvalues(correlated_masks("site"), 0.3)
    assert sites["site pairs"] < ALPHA  # a 20% copy of the neighboring site
    layers = correlation_pvalues(correlated_masks("layer"), 0.3)
    assert layers["layer lag 1"] < ALPHA
    # masks that depend on whether a gate touched the site: 3% more likely when touched
    rng = np.random.default_rng(3)
    touch = rng.integers(0, 3, size=(40, 200, 12))
    biased = rng.random(touch.shape) < 0.3 + 0.03 * (touch > 0)
    assert gate_dependence_pvalues(biased, touch, 0.3)["touched, layer offset 0"] < ALPHA
    unbiased = rng.random(touch.shape) < 0.3
    assert min(gate_dependence_pvalues(unbiased, touch, 0.3).values()) > ALPHA


def test_the_extreme_rates_measure_nothing_or_everything():
    manifest = make_manifest(9, ["0", "1"], 15, 2)
    for p, expected in (("0", []), ("1", list(range(9)))):
        spec, _ = make_spec(manifest, 9, p)
        for u in range(2):
            seed = manifest.run_seed(9, "0.3", p)
            layers = wsref.parse_circuit(str(spec.circuit(u, seed=seed)))
            assert all(layer.measured == expected for layer in layers)


# ---------------------------------------------------------------------------
# circuits: nesting, shared uniforms, prefixes (v1.4 validation requirements 6, 10, 11)
# ---------------------------------------------------------------------------

RATE_GRID = ["0", "0.05", "0.1", "0.15", "0.2", "1/3", "0.5", "0.9", "1"]


def test_masks_are_uniform_below_the_float_threshold_and_nested_across_rates():
    n, layers, stride = 10, 40, 6
    manifest = make_manifest(n, RATE_GRID, layers, stride, graphs_count=2)
    specs = {p: make_spec(manifest, n, p)[0] for p in RATE_GRID}
    records = [build_graph(n, "0.3", g, MASTER_GRAPH) for g in range(2)]
    for g, c in [(0, 0), (0, 5), (1, 0), (1, 3), (1, 5)]:  # aligned (g, c, u)
        u = manifest.stream_index(g, c)
        seed = manifest.run_seed(n, "0.3", "0.1")
        assert seed == wsref.run_seed(MASTER_CIRCUIT, n, "0.3")  # five words, no rate in them
        assert all(manifest.run_seed(n, "0.3", p) == seed for p in RATE_GRID)  # shared gate mode
        draws = wsref.study_draws(seed, u, len(records[g].edges), n, layers)
        exports = {p: wsref.parse_circuit(str(specs[p].circuit(u, seed=seed))) for p in RATE_GRID}
        ordered = sorted(RATE_GRID, key=lambda p: Token.parse(p).value)
        for p in RATE_GRID:
            # the same gates at every rate ...
            assert [layer.gates for layer in exports[p]] == [
                layer.gates for layer in exports["0.1"]
            ]
            # ... and the mask is exactly "uniform < float64(p)" for the shared uniforms
            threshold = Token.parse(p).float64
            for t, layer in enumerate(exports[p]):
                assert layer.measured == [i for i in range(n) if draws.uniform[t, i] < threshold]
        for low, high in zip(ordered[:-1], ordered[1:], strict=True):  # M(p1) <= M(p2) sitewise
            for a, b in zip(exports[low], exports[high], strict=True):
                assert set(a.measured) <= set(b.measured)
        assert all(not layer.measured for layer in exports["0"])
        assert all(layer.measured == list(range(n)) for layer in exports["1"])
        # the measured fraction grows with the rate (a sanity check that the grid is not degenerate)
        sizes = [sum(len(layer.measured) for layer in exports[p]) for p in ordered]
        assert sizes == sorted(sizes) and sizes[0] == 0 and sizes[-1] == n * layers


def test_nesting_does_not_hold_when_the_gate_mode_is_independent():
    n, layers, stride = 10, 40, 4
    manifest = make_manifest(n, ["0.1", "0.2", "0.3"], layers, stride, gate_mode="independent")
    seeds = {p: manifest.run_seed(n, "0.3", p) for p in ("0.1", "0.2", "0.3")}
    assert len(set(seeds.values())) == 3
    assert all(s == wsref.run_seed(MASTER_CIRCUIT, n, "0.3", p) for p, s in seeds.items())
    records = [build_graph(n, "0.3", 0, MASTER_GRAPH)]
    u = manifest.stream_index(0, 2)
    out = {}
    for p, seed in seeds.items():
        spec, _ = make_spec(manifest, n, p)
        out[p] = wsref.parse_circuit(str(spec.circuit(u, seed=seed)))
        # reproducible: the same (spec, seed, u) gives the same circuit
        assert out[p] == wsref.parse_circuit(str(spec.circuit(u, seed=seed)))
        draws = wsref.study_draws(seed, u, len(records[0].edges), n, layers)
        assert [layer.measured for layer in out[p]] == [
            [i for i in range(n) if draws.uniform[t, i] < Token.parse(p).float64]
            for t in range(layers)
        ]
    gates = {p: [layer.gates for layer in out[p]] for p in out}
    assert (
        gates["0.1"] != gates["0.2"]
        and gates["0.2"] != gates["0.3"]
        and gates["0.1"] != gates["0.3"]
    )
    # masks are not nested: some site is measured at 0.1 and not at 0.2 (certain at 400 sites)
    assert any(
        set(a.measured) - set(b.measured) for a, b in zip(out["0.1"], out["0.2"], strict=True)
    )


@pytest.mark.parametrize("short, long", [(1, 7), (5, 17), (8, 8)])
def test_a_longer_horizon_extends_the_circuit_prefix(short, long):
    n, stride = 11, 3
    manifest_short = make_manifest(n, ["0.25"], short, stride)
    manifest_long = make_manifest(n, ["0.25"], long, stride)
    spec_short, _ = make_spec(manifest_short, n, "0.25")
    spec_long, _ = make_spec(manifest_long, n, "0.25")
    seed = manifest_short.run_seed(n, "0.3", "0.25")
    assert seed == manifest_long.run_seed(n, "0.3", "0.25")  # the horizon is in no seed
    for u in range(stride):
        a = wsref.parse_circuit(str(spec_short.circuit(u, seed=seed)))
        b = wsref.parse_circuit(str(spec_long.circuit(u, seed=seed)))
        assert len(a) == short and len(b) == long and a == b[:short]


def test_unrelated_choices_leave_the_circuit_unchanged():
    # probes, checkpoint schedule, the other graphs of the collection and other trajectories:
    # trajectory u depends on (graph u // stride, seed, u) only
    n, layers, stride = 12, 20, 4
    manifest = make_manifest(n, ["0.2"], layers, stride, graphs_count=2)
    spec, records = make_spec(manifest, n, "0.2")
    seed = manifest.run_seed(n, "0.3", "0.2")
    base = str(spec.circuit(5, seed=seed))
    with_probes, _ = make_spec(
        manifest, n, "0.2", probes=[mipt.code_dimension(), mipt.contiguous_distance()]
    )
    assert str(with_probes.circuit(5, seed=seed)) == base
    for checkpoints in ("final", 5, [0, 7, 20]):
        other = mipt.MonitoredCircuit(
            **{**manifest.monitored_circuit_kwargs(n, "0.3", "0.2"), "checkpoints": checkpoints},
            edges=[r.edges for r in records],
        )
        assert str(other.circuit(5, seed=seed)) == base
    # more graphs appended to the collection do not renumber or change trajectory 5 (graph 1)
    bigger = make_manifest(n, ["0.2"], layers, stride, graphs_count=6)
    spec_big, _ = make_spec(bigger, n, "0.2")
    assert bigger.run_seed(n, "0.3", "0.2") == seed  # the number of graphs is in no seed
    assert str(spec_big.circuit(5, seed=seed)) == base
    assert spec_big.num_trajectories == 6 * stride and spec.num_trajectories == 2 * stride
    # more circuits per graph (a larger stride) renumber, which is why the stride is frozen
    wider = make_manifest(n, ["0.2"], layers, stride + 4, graphs_count=2)
    spec_wide, _ = make_spec(wider, n, "0.2")
    assert str(spec_wide.circuit(5, seed=seed)) != base


def test_the_simulated_code_dimension_has_a_horizon_independent_prefix():
    # the circuit prefix is the trajectory prefix: k_t for t <= 6 does not depend on T_max, nor on
    # whether the other layers are observed (v1.4 validation requirement 6)
    n, stride = 10, 6
    graph = [build_graph(n, "0.3", 0, MASTER_GRAPH).edges]
    results = {}
    for layers, checkpoints in [(6, "every_layer"), (14, "every_layer"), (14, [0, 6, 14])]:
        manifest = make_manifest(n, ["0.3"], layers, stride)
        spec = mipt.MonitoredCircuit(
            **{**manifest.monitored_circuit_kwargs(n, "0.3", "0.3"), "checkpoints": checkpoints},
            edges=graph,
            probes=[mipt.code_dimension()],
        )
        values = spec.run(stride, seed=manifest.run_seed(n, "0.3", "0.3")).values["code_dimension"]
        results[(layers, str(checkpoints))] = values
    short, long = results[(6, "every_layer")], results[(14, "every_layer")]
    sparse = results[(14, "[0, 6, 14]")]
    assert short.shape == (stride, 7) and long.shape == (stride, 15) and sparse.shape == (stride, 3)
    assert np.array_equal(short, long[:, :7])
    assert np.array_equal(sparse[:, 1], long[:, 6]) and np.array_equal(sparse[:, 2], long[:, 14])
    assert (np.diff(long, axis=1) <= 0).all()  # k_t is nonincreasing (v1.4 section 5.2)
    assert (long[:, 0] == n).all()  # r_0 = 0, k_0 = n


# ---------------------------------------------------------------------------
# the samples of different (n, beta) are independent; only the rates are coupled (ruling R-SEED)
# ---------------------------------------------------------------------------
#
# v1.4 couples conditions across the rate p only (nested masks, shared gates). Graph g of (n, beta)
# and graph g of (n', beta') must not share raw random numbers: finite-size fits treat the sizes and
# the rewiring probabilities as independent samples. The first rewiring decisions and the first gate
# draws are where a shared seed shows at once. Each test has its null law (exact), a z-test with the
# false-alarm rate of the file, and a negative control: the convention the study first drafted, one
# seed for every condition, makes the very same statistic degenerate.


def z_binomial(count: int, trials: int, q: float) -> float:
    """The standard score of ``count`` successes in ``trials`` independent trials of probability ``q``."""
    return (count - trials * q) / math.sqrt(trials * q * (1 - q))


def first_attempts(n: int, beta: str, count: int, seed_of) -> np.ndarray:
    """For graphs ``0..count-1`` of ``(n, beta)``: the visit of the first rewiring attempt (``2 n`` if none).

    Before the first success every visit consumes exactly one ``next_double()``, the decision
    ``< beta``, so the index of the first success is read from the stream of the seed alone, with
    the reference generator (the trace-for-trace test above ties it to the native one).
    """
    threshold = Token.parse(beta).float64
    out = np.empty(count, dtype=int)
    for g in range(count):
        rng = ref.Xoshiro(wsref.seed_words(seed_of(n, beta, g)))
        j = 0
        while j < 2 * n and not rng.next_double() < threshold:
            j += 1
        out[g] = j
    return out


def first_attempt_pmf(n: int, beta: Fraction) -> list[Fraction]:
    """``P(first attempt at visit j)`` for ``j = 0..2n-1`` and ``P(no attempt)`` at index ``2n``."""
    return [(1 - beta) ** j * beta for j in range(2 * n)] + [(1 - beta) ** (2 * n)]


def shared_seed(n: int, beta, g: int) -> tuple[int, int]:
    """The convention of the first draft, kept as a negative control: one seed ``(master, g)`` for all."""
    return (MASTER_GRAPH, g)


def test_graph_collections_of_different_sizes_are_independent_samples():
    small, large, count = 12, 16, 3000
    pa, pb = first_attempt_pmf(small, Fraction(1, 2)), first_attempt_pmf(large, Fraction(1, 2))
    q = float(sum(x * y for x, y in zip(pa, pb, strict=False)))  # P(equal) for independent draws
    assert 0.33 < q < 0.34  # about 1/3: the first attempt is Geometric(1/2) at both sizes
    a = first_attempts(small, "0.5", count, graph_seed)
    b = first_attempts(large, "0.5", count, graph_seed)
    agree = int((a == b).sum())
    assert abs(z_binomial(agree, count, q)) < SIGMAS  # about 1000 of 3000 agree, by chance alone
    # negative control: with one seed per g for every size the first decisions are the same draws
    old_a = first_attempts(small, "0.5", count, shared_seed)
    old_b = first_attempts(large, "0.5", count, shared_seed)
    assert int((old_a == old_b).sum()) > 0.99 * count
    assert abs(z_binomial(int((old_a == old_b).sum()), count, q)) > SIGMAS


def test_graph_collections_of_different_betas_are_independent_samples():
    # One seed for every beta would nest the decisions (u < beta): the first rewiring at the larger
    # beta could only come at or before that at the smaller one. For independent collections it
    # does so with the probability P(J_high <= J_low), exactly computed from the two geometric laws.
    n, count = 16, 3000
    p_low, p_high = first_attempt_pmf(n, Fraction(3, 10)), first_attempt_pmf(n, Fraction(1, 2))
    q = float(sum(p_high[j] * sum(p_low[j:]) for j in range(2 * n + 1)))
    assert 0.76 < q < 0.78  # about 0.5 / (1 - 0.35)
    low = first_attempts(n, "0.3", count, graph_seed)
    high = first_attempts(n, "0.5", count, graph_seed)
    assert abs(z_binomial(int((high <= low).sum()), count, q)) < SIGMAS
    # negative control: a shared seed always nests (the decisions are one uniform per visit)
    assert (
        first_attempts(n, "0.5", count, shared_seed) <= first_attempts(n, "0.3", count, shared_seed)
    ).all()


def test_circuit_streams_of_different_betas_are_independent_samples():
    # the same size, rate and trajectory index u: the edge indices (uniform on 2n values) agree with
    # probability 1 / 2n when the run seeds are independent, and always when they are shared
    n, layers, k, p = 12, 40, 40, "0.2"
    low_manifest = make_manifest(n, [p], layers, k, beta="0.1")
    high_manifest = make_manifest(n, [p], layers, k, beta="0.3")
    (low_spec, low_records), (high_spec, high_records) = (
        make_spec(low_manifest, n, p, beta="0.1"),
        make_spec(high_manifest, n, p, beta="0.3"),
    )
    low_seed = low_manifest.run_seed(n, "0.1", p)
    high_seed = high_manifest.run_seed(n, "0.3", p)
    assert low_seed != high_seed
    low = Sample(low_spec, low_records, low_seed, range(k)).edge
    high = Sample(high_spec, high_records, high_seed, range(k)).edge
    assert low.size == 40 * 40 * (n // 2) == 9600
    assert abs(z_binomial(int((low == high).sum()), low.size, 1 / (2 * n))) < SIGMAS
    # negative control: the master circuit seed itself, shared by both betas, gives equal draws
    shared_low = Sample(low_spec, low_records, MASTER_CIRCUIT, range(k)).edge
    shared_high = Sample(high_spec, high_records, MASTER_CIRCUIT, range(k)).edge
    assert np.array_equal(shared_low, shared_high)


def test_circuit_streams_of_different_sizes_are_independent_samples():
    # Under one shared seed trajectory u of every size starts from the same first draw X, and the
    # first gate's edge index is floor(2 n X): the intervals [e / 2n, (e + 1) / 2n) of two sizes
    # always overlap (at X). For independent seeds they overlap with the probability computed here.
    small, large, k, p = 12, 16, 1000, "0.2"
    first, shared = {}, {}
    for n in (small, large):
        manifest = make_manifest(n, [p], 1, k)  # k trajectories on graph 0, one layer
        spec, records = make_spec(manifest, n, p)
        first[n] = Sample(spec, records, manifest.run_seed(n, "0.3", p), range(k)).edge[:, 0, 0]
        shared[n] = Sample(spec, records, MASTER_CIRCUIT, range(k)).edge[:, 0, 0]
    b, c = 2 * small, 2 * large

    def overlap(e: int, f: int) -> bool:
        return e * c < (f + 1) * b and f * b < (e + 1) * c

    q = sum(overlap(e, f) for e in range(b) for f in range(c)) / (b * c)
    assert q == (b + c - math.gcd(b, c)) / (b * c) == 0.0625  # the cells of the two grids that meet
    hits = sum(overlap(int(e), int(f)) for e, f in zip(first[small], first[large], strict=True))
    assert abs(z_binomial(hits, k, q)) < SIGMAS
    assert all(overlap(int(e), int(f)) for e, f in zip(shared[small], shared[large], strict=True))
