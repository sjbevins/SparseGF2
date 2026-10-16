"""Geometry-correlated erasure regions, graph side: ``sparsegf2.erasures`` and its kernel.

Checked here (G = (V, E) a graph on V = {0..n-1}, b(A) the number of edges leaving the
region A, Q_J(A) = exp(-J b(A)) / Z_m(J) on the regions of size m):

1. The kernel's incremental boundary change Δb equals recomputation for every swap
   (u in A, v not in A; adjacent pairs included) on random small graphs (Erdős–Rényi,
   cycles, complete graphs, a star, a path, Watts–Strogatz); ``region_boundary``,
   ``boundary`` and the CSR form against direct counts; the exact density of states
   N_m(b) and the minimum-boundary regions against brute force.
2. The chain keeps |A| = m and records sorted, distinct nodes whose boundaries are the
   recorded ones; m = 0, m = n and n = 1 are handled and draw nothing.
3. Stationarity: on graphs small enough to list every subset, the recorded states follow
   Q_J exactly (total variation within a bound set by the effective sample size, and a
   chi-square goodness of fit not rejected at 1e-4), with and without replica exchange,
   for J in {-1, 0, 0.7, 2.5}; J = 0 is the uniform law.
4. The ring C_n: the run count N(n, m, r) = (n/r) C(m-1, r-1) C(n-m-1, r-1), log Z and the
   law of b against brute force; the exact ring sampler against that law (chi-square on r
   and on single regions); the chain's <b> and Var b on C_64 within 4 standard errors of
   the closed forms.
5. Large J: on C_32 every state at J = 8 is an interval; on the open path only the two
   end intervals survive (J = 16; at J = 8 the law of b matches Q_8 itself); without
   replica exchange the chains of different seeds stick at different ends, which R-hat
   reveals, and with it R-hat is close to 1.
6. Proposition 2: <b> is nonincreasing in J and d<b>/dJ = -Var b, d log Z/dJ = -<b>
   (exact, from N_m(b)); the chain's mean boundary on a Watts–Strogatz graph decreases
   with J within errors.
7. ``chain_diagnostics``: tau_int of AR(1) series within 10% of (1 + rho)/(1 - rho);
   R-hat near 1 for agreeing chains and clearly above 1 for disagreeing ones.
8. Determinism: the same seed gives the same output, other seeds differ, the draw order
   equals a plain-Python transcription of the documented order, and a pinned digest
   guards the chain's stream.
9. Validation messages.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter

import numba
import numpy as np
import pytest

import _reference_rng as ref
from sparsegf2 import erasures, graphs
from sparsegf2._kernel import erasure_chain as kernel
from sparsegf2._kernel.rng import seed_state

# The weekly job runs the kernels as plain Python (NUMBA_DISABLE_JIT=1), several hundred
# times slower. There the chains are shorter; every tolerance below follows from the
# number of samples actually drawn (and their effective number), so the assertions are
# the same.
JIT_FREE = bool(numba.config.DISABLE_JIT)


def _size(compiled: int, plain: int) -> int:
    return plain if JIT_FREE else compiled


# ---------------------------------------------------------------------------------------
# Helpers: brute force, chi-square, effective sample sizes
# ---------------------------------------------------------------------------------------


def _brute_boundary(edges, region) -> int:
    inside = {int(q) for q in region}
    return sum((int(a) in inside) != (int(b) in inside) for a, b in edges)


def _small_graphs():
    """The graphs of the exhaustive checks: random, regular, extreme degrees."""
    return [
        ("erdos_renyi_9", graphs.erdos_renyi(9, 0.4, seed=1), 9),
        ("erdos_renyi_12", graphs.erdos_renyi(12, 0.3, seed=2), 12),
        ("cycle_7", graphs.cycle(7), 7),
        ("complete_6", graphs.complete(6), 6),
        ("star_8", graphs.from_edges([(0, v) for v in range(1, 8)], 8), 8),
        ("path_5", graphs.path(5), 5),
        ("watts_strogatz_13", graphs.watts_strogatz(13, 4, 0.3, seed=3), 13),
        ("no_edges_5", np.zeros((0, 2), dtype=np.int64), 5),
    ]


SMALL = _small_graphs()
SMALL_IDS = [name for name, _, _ in SMALL]


def _gamma_q(a: float, x: float) -> float:
    """The regularized upper incomplete gamma function Q(a, x) (series or continued fraction)."""
    if x <= 0.0:
        return 1.0
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:  # series for P(a, x)
        term = total = 1.0 / a
        ap = a
        for _ in range(100_000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return 1.0 - total * math.exp(log_prefactor)
    tiny = 1e-300  # modified Lentz for the continued fraction of Q(a, x)
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 100_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return math.exp(log_prefactor) * h


def _chi2_sf(stat: float, dof: int) -> float:
    """P(X >= stat) for X chi-square with ``dof`` degrees of freedom."""
    return _gamma_q(dof / 2.0, stat / 2.0)


def test_chi_square_helper_against_known_quantiles():
    assert _chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-12)
    assert _chi2_sf(18.307038053275146, 10) == pytest.approx(0.05, abs=1e-12)
    assert _chi2_sf(135.80672317102676, 100) == pytest.approx(0.01, abs=1e-12)
    assert _chi2_sf(0.0, 3) == 1.0


def _effective_samples(series_list, draws: int) -> float:
    """The smallest effective sample size of the given series, at most ``draws``."""
    ess = float(draws)
    for x in series_list:
        x = np.asarray(x, dtype=np.float64)
        if np.ptp(x) > 0:
            ess = min(ess, erasures.chain_diagnostics(x)["ess"])
    return ess


def _chi_square(counts, p, draws, ess):
    """Pearson's statistic on bins of at least 5 effective expected draws, rescaled to ``ess``.

    Draws of a Markov chain are correlated: the variance of each count is inflated by at
    most the largest integrated autocorrelation time, draws / ess, so dividing the
    statistic by it gives a chi-square variable (conservatively).
    """
    observed, expected = [], []
    acc_o = acc_e = 0.0
    for k in np.argsort(p, kind="stable"):
        acc_o += counts[k]
        acc_e += p[k] * draws
        if acc_e * ess / draws >= 5.0:
            observed.append(acc_o)
            expected.append(acc_e)
            acc_o = acc_e = 0.0
    if acc_e > 0.0:
        if observed:
            observed[-1] += acc_o
            expected[-1] += acc_e
        else:
            observed.append(acc_o)
            expected.append(acc_e)
    o, e = np.array(observed), np.array(expected)
    stat = float(np.sum((o - e) ** 2 / e)) * ess / draws
    return stat, len(o) - 1


def _assert_follows(ids, boundaries, p, label):
    """The recorded states (indices into the exact law ``p``) follow ``p``.

    Total variation: for N independent draws E[TV] <= sum_i sqrt(p_i (1 - p_i) / N) / 2,
    and one draw moves TV by at most 1/N, so P(TV > E[TV] + t) <= exp(-2 N t^2)
    (McDiarmid); with N the effective sample size and t at probability 1e-6. The chain's
    effective size is the smallest over b and the indicators of the likeliest states.
    """
    draws = len(ids)
    likely = [k for k in np.argsort(p)[::-1][:12] if p[k] >= 0.01]
    ess = _effective_samples([boundaries, *((ids == k) for k in likely)], draws)
    freq = np.bincount(ids, minlength=len(p)) / draws
    tv = 0.5 * float(np.abs(freq - p).sum())
    bound = 0.5 * float(np.sum(np.sqrt(p * (1 - p) / ess))) + math.sqrt(math.log(1e6) / (2 * ess))
    assert tv < bound, f"{label}: total variation {tv:.4f} >= {bound:.4f} (ESS {ess:.0f})"
    stat, dof = _chi_square(freq * draws, p, draws, ess)
    assert dof >= 1
    pvalue = _chi2_sf(stat, dof)
    assert pvalue > 1e-4, f"{label}: chi-square {stat:.1f} on {dof} dof, p = {pvalue:.2e}"


def _exact_law(edges, n, m, coupling):
    subsets = list(itertools.combinations(range(n), m))
    b = np.array([_brute_boundary(edges, a) for a in subsets], dtype=np.float64)
    logw = -coupling * b
    w = np.exp(logw - logw.max())
    return subsets, b, w / w.sum()


def _state_ids(regions, subsets, n):
    """Index of each recorded region (rows of ``regions``) in ``subsets``."""
    lookup = np.full(1 << n, -1, dtype=np.int64)
    for k, a in enumerate(subsets):
        lookup[sum(1 << q for q in a)] = k
    masks = np.sum(np.left_shift(np.int64(1), regions), axis=-1)
    ids = lookup[masks]
    assert np.all(ids >= 0)
    return ids


def _is_ring_interval(region, n) -> bool:
    inside = set(int(q) for q in region)
    starts = [q for q in inside if (q - 1) % n not in inside]
    return len(starts) == 1


# ---------------------------------------------------------------------------------------
# 1. Δb, boundaries, CSR, exact density of states
# ---------------------------------------------------------------------------------------


def test_csr_has_sorted_rows_with_the_neighbours_of_graphs_csr():
    for _, edges, n in SMALL:
        indptr, indices = kernel.csr(n, edges)
        ref_ptr, ref_idx = graphs._csr(edges, n)
        assert indptr.dtype == np.int64 and indices.dtype == np.int64
        assert np.array_equal(indptr, ref_ptr)
        for v in range(n):
            row = indices[indptr[v] : indptr[v + 1]]
            assert np.all(np.diff(row) > 0)
            assert set(row.tolist()) == set(ref_idx[ref_ptr[v] : ref_ptr[v + 1]].tolist())


@pytest.mark.parametrize(("name", "edges", "n"), SMALL, ids=SMALL_IDS)
def test_swap_delta_equals_recomputation_for_every_swap(name, edges, n, rng):
    indptr, indices = kernel.csr(n, edges)
    adjacent_pairs = 0
    for m in range(1, n):
        for _ in range(2):
            region = rng.choice(n, m, replace=False)
            member = np.zeros(n, dtype=np.uint8)
            member[region] = 1
            before = kernel.region_boundary(indptr, indices, member)
            assert before == _brute_boundary(edges, region)
            assert before == erasures.boundary(edges, n, region)
            for u in region:
                for v in np.flatnonzero(member == 0):
                    after = member.copy()
                    after[u] = 0
                    after[v] = 1
                    expected = _brute_boundary(edges, np.flatnonzero(after)) - before
                    assert kernel.swap_delta(indptr, indices, member, u, v) == expected
                    adjacent_pairs += v in indices[indptr[u] : indptr[u + 1]]
    assert adjacent_pairs > 0 or len(edges) == 0


def test_boundary_of_simple_regions():
    ring = graphs.cycle(10)
    assert erasures.boundary(ring, 10, []) == 0
    assert erasures.boundary(ring, 10, range(10)) == 0
    assert erasures.boundary(ring, 10, [3]) == 2
    assert erasures.boundary(ring, 10, [2, 3, 4]) == 2
    assert erasures.boundary(ring, 10, [0, 9]) == 2  # an interval across the seam
    assert erasures.boundary(ring, 10, np.array([1, 3, 5])) == 6
    assert erasures.boundary(graphs.path(10), 10, [0, 1, 2]) == 1
    assert isinstance(erasures.boundary(ring, 10, [1]), int)


@pytest.mark.parametrize(("name", "edges", "n"), SMALL, ids=SMALL_IDS)
def test_density_of_states_and_minimum_regions_match_brute_force(name, edges, n):
    for m in range(n + 1):
        subsets = list(itertools.combinations(range(n), m))
        bs = [_brute_boundary(edges, a) for a in subsets]
        counts = erasures.boundary_counts(edges, n, m)
        assert counts.dtype == np.int64
        assert np.array_equal(counts, np.bincount(bs))
        assert counts[-1] > 0 and counts.sum() == math.comb(n, m)
        b_min, regions = erasures.minimum_boundary_regions(edges, n, m)
        assert b_min == min(bs) and isinstance(b_min, int)
        assert regions.dtype == np.int64 and regions.shape[1] == m
        assert regions.tolist() == [list(a) for a, b in zip(subsets, bs, strict=True) if b == b_min]


def test_minimum_regions_of_the_ring_and_the_path():
    n, m = 16, 5
    b_min, regions = erasures.minimum_boundary_regions(graphs.cycle(n), n, m)
    assert b_min == 2
    intervals = sorted(sorted((s + k) % n for k in range(m)) for s in range(n))
    assert regions.tolist() == intervals  # the n intervals (lexicographic order)
    b_min, regions = erasures.minimum_boundary_regions(graphs.path(n), n, m)
    assert b_min == 1
    assert regions.tolist() == [list(range(m)), list(range(n - m, n))]  # the two end intervals


def test_enumeration_budget_is_enforced_with_the_number_of_subsets():
    with pytest.raises(ValueError, match=r"C\(40, 20\) = 137,846,528,820"):
        erasures.boundary_counts(graphs.cycle(40), 40, 20)
    with pytest.raises(ValueError, match="budget"):
        erasures.minimum_boundary_regions(graphs.cycle(12), 12, 6, budget=100)
    counts = erasures.boundary_counts(graphs.cycle(12), 12, 6, budget=924)  # exactly C(12, 6)
    assert counts.sum() == 924


def test_enumeration_kernel_writes_as_many_regions_as_fit_and_counts_them_all():
    n, m = 9, 3
    indptr, indices = kernel.csr(n, graphs.cycle(n))
    hist = np.zeros(n + 1, dtype=np.int64)
    rows = np.full((4, m), -1, dtype=np.int64)
    found = kernel.enumerate_boundaries(indptr, indices, n, m, hist, 2, rows)
    assert found == n  # the n intervals have b = 2
    assert rows.tolist() == [[0, 1, 2], [0, 1, 8], [0, 7, 8], [1, 2, 3]]  # the first four
    assert hist.sum() == math.comb(n, m) and hist[2] == n


# ---------------------------------------------------------------------------------------
# 2. Chain invariants and degenerate sizes
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("tempering", [True, False])
def test_chain_records_regions_of_size_m_with_their_boundaries(tempering):
    n = 16
    edges = graphs.watts_strogatz(n, 4, 0.2, seed=4)
    couplings = [0.0, 1.0, -0.5]
    for m in (1, 5, 8, 15):
        res = erasures.sample_regions(
            edges, n, m, couplings, 30, burn_in=3, thin=2, tempering=tempering, seed=3
        )
        assert res.regions.shape == (3, 30, m) and res.regions.dtype == np.int64
        assert res.boundaries.shape == (3, 30) and res.boundaries.dtype == np.int64
        assert np.all(np.diff(res.regions, axis=-1) > 0)  # sorted, distinct
        assert res.regions.min() >= 0 and res.regions.max() < n
        for i in range(3):
            for s in range(30):
                assert res.boundaries[i, s] == _brute_boundary(edges, res.regions[i, s])
        assert np.all((res.acceptance > 0) & (res.acceptance <= 1))
        assert res.acceptance[0] == 1.0  # J = 0 accepts every swap
        assert res.exchange_acceptance.shape == (2,)
        assert np.all(np.isnan(res.exchange_acceptance)) != tempering
        assert res.size == m and res.num_nodes == n and res.tempering == tempering
        assert res.couplings.tolist() == couplings


def test_degenerate_sizes_record_the_single_region_and_draw_nothing():
    edges = graphs.cycle(5)
    for m in (0, 5):
        res = erasures.sample_regions(edges, 5, m, [0.0, 2.0], 3, seed=1)
        assert res.regions.shape == (2, 3, m)
        assert np.all(res.regions == np.arange(m))
        assert np.all(res.boundaries == 0)
        assert np.all(np.isnan(res.acceptance)) and np.all(np.isnan(res.exchange_acceptance))
    for m in (0, 1):
        res = erasures.sample_regions(np.zeros((0, 2), dtype=np.int64), 1, m, 0.5, 2, seed=0)
        assert res.regions.shape == (1, 2, m) and np.all(res.boundaries == 0)
        assert res.exchange_acceptance.shape == (0,)
    indptr, indices = kernel.csr(4, graphs.cycle(4))
    state = seed_state(9)
    before = state.copy()
    regions = np.full((2, 2, 4), -1, dtype=np.int64)
    bnd = np.full((2, 2), -1, dtype=np.int64)
    moves = np.full((2, 4), -1, dtype=np.int64)
    kernel.sample_chain(
        indptr, indices, 4, 4, np.array([0.0, 1.0]), 2, 5, 1, True, state, regions, bnd, moves
    )
    assert np.array_equal(state, before)  # no draws
    assert np.all(regions == np.arange(4)) and np.all(bnd == 0) and np.all(moves == 0)


def test_a_graph_without_edges_accepts_every_swap():
    res = erasures.sample_regions(np.zeros((0, 2), dtype=np.int64), 6, 2, [3.0, -3.0], 20, seed=2)
    assert np.all(res.boundaries == 0) and np.all(res.acceptance == 1.0)
    assert np.all(res.exchange_acceptance == 1.0)


# ---------------------------------------------------------------------------------------
# 3. Stationarity on enumerable graphs
# ---------------------------------------------------------------------------------------

STATIONARY = {
    "erdos_renyi_8": (graphs.erdos_renyi(8, 0.45, seed=3), 8, 3),
    "watts_strogatz_10": (graphs.watts_strogatz(10, 4, 0.3, seed=2), 10, 4),
    "path_9": (graphs.path(9), 9, 4),
}
COUPLINGS = [2.5, -1.0, 0.7, 0.0]  # deliberately unsorted: results come back in this order


@pytest.mark.parametrize("tempering", [True, False], ids=["tempering", "independent"])
@pytest.mark.parametrize("name", list(STATIONARY))
def test_chain_samples_the_exact_law(name, tempering):
    edges, n, m = STATIONARY[name]
    draws = _size(20_000, 1_500)
    res = erasures.sample_regions(
        edges, n, m, COUPLINGS, draws, burn_in=100, thin=_size(3, 2), tempering=tempering, seed=11
    )
    for i, coupling in enumerate(COUPLINGS):
        subsets, b, p = _exact_law(edges, n, m, coupling)
        if coupling == 0.0:
            assert np.allclose(p, 1.0 / math.comb(n, m))  # J = 0: the uniform law
        ids = _state_ids(res.regions[i], subsets, n)
        assert np.array_equal(b[ids], res.boundaries[i])
        _assert_follows(ids, res.boundaries[i], p, f"{name} J={coupling}")


# ---------------------------------------------------------------------------------------
# 4. The ring: closed forms, the exact sampler, the chain against the closed forms
# ---------------------------------------------------------------------------------------


def _ring_runs(n, region) -> int:
    inside = set(region)
    return sum(1 for q in inside if (q - 1) % n not in inside)


@pytest.mark.parametrize("n", range(3, 13))
def test_ring_run_count_matches_brute_force(n):
    counts: Counter = Counter()
    ring = graphs.cycle(n)
    for m in range(n + 1):
        for a in itertools.combinations(range(n), m):
            r = _ring_runs(n, a)
            assert _brute_boundary(ring, a) == 2 * r  # b = 2r on the ring
            counts[m, r] += 1
    for m in range(n + 1):
        for r in range(n + 2):
            value = erasures.ring_run_count(n, m, r)
            assert isinstance(value, int) and value == counts[m, r], (m, r)


@pytest.mark.parametrize("n", [3, 4, 7, 10, 12])
def test_ring_law_and_partition_function_match_brute_force(n):
    ring = graphs.cycle(n)
    for m in range(n + 1):
        bs = np.array([_brute_boundary(ring, a) for a in itertools.combinations(range(n), m)])
        assert np.array_equal(
            erasures.boundary_counts(ring, n, m),
            np.bincount(bs),
        )
        for coupling in (-1.3, 0.0, 0.4, 2.0):
            weights = np.exp(-coupling * bs)
            log_z = math.log(weights.sum())
            assert erasures.ring_partition_function(n, m, coupling) == pytest.approx(
                log_z, rel=1e-12, abs=1e-12
            )
            values, probs = erasures.ring_boundary_distribution(n, m, coupling)
            assert values.dtype == np.int64 and probs.dtype == np.float64
            expected = [weights[bs == v].sum() / weights.sum() for v in values]
            assert probs == pytest.approx(expected, rel=1e-12, abs=1e-15)
            assert set(values.tolist()) == set(bs.tolist())


def test_ring_closed_forms_at_large_n_and_coupling_zero():
    n, m = 100_000, 30_000
    assert erasures.ring_partition_function(n, m, 0.0) == pytest.approx(
        math.lgamma(n + 1) - math.lgamma(m + 1) - math.lgamma(n - m + 1), rel=1e-12
    )
    values, probs = erasures.ring_boundary_distribution(n, m, 1.5)
    assert np.all(np.isfinite(probs)) and probs.sum() == pytest.approx(1.0)
    assert values[0] == 2 and values[-1] == 2 * m
    assert np.isfinite(erasures.ring_partition_function(n, m, -40.0))
    assert np.isfinite(erasures.ring_partition_function(n, m, 40.0))
    assert erasures.ring_run_count(n, 1, 1) == n
    assert erasures.ring_run_count(n, 0, 0) == erasures.ring_run_count(n, n, 0) == 1
    assert erasures.ring_run_count(n, 0, 1) == erasures.ring_run_count(n, 2, 3) == 0


@pytest.mark.parametrize("coupling", [-0.8, 0.0, 0.7])
def test_exact_ring_sampler_follows_the_law_of_single_regions(coupling):
    n, m = 8, 3
    draws = _size(20_000, 4_000)
    regions = erasures.sample_ring_regions(n, m, coupling, draws, seed=5)
    assert regions.shape == (draws, m) and regions.dtype == np.int64
    assert np.all(np.diff(regions, axis=1) > 0)
    subsets, _, p = _exact_law(graphs.cycle(n), n, m, coupling)
    ids = _state_ids(regions, subsets, n)
    stat, dof = _chi_square(np.bincount(ids, minlength=len(p)), p, draws, draws)
    assert _chi2_sf(stat, dof) > 1e-4


def test_exact_ring_sampler_follows_the_law_of_runs():
    n, m, coupling = 40, 12, 0.4
    draws = _size(20_000, 3_000)
    regions = erasures.sample_ring_regions(n, m, coupling, draws, seed=6)
    runs = np.array([_ring_runs(n, a.tolist()) for a in regions])
    values, probs = erasures.ring_boundary_distribution(n, m, coupling)
    counts = np.array([np.count_nonzero(runs == v // 2) for v in values])
    assert counts.sum() == draws
    stat, dof = _chi_square(counts, probs, draws, draws)
    assert _chi2_sf(stat, dof) > 1e-4
    ring = graphs.cycle(n)
    for a, r in zip(regions[:50], runs[:50], strict=True):
        assert erasures.boundary(ring, n, a) == 2 * r


def test_exact_ring_sampler_edge_cases():
    assert np.array_equal(erasures.sample_ring_regions(6, 0, 1.0, 3, seed=1), np.zeros((3, 0)))
    assert np.array_equal(
        erasures.sample_ring_regions(6, 6, 1.0, 2, seed=1), np.tile(np.arange(6), (2, 1))
    )
    one = erasures.sample_ring_regions(9, 1, 3.0, 50, seed=2)  # a single node: always r = 1
    assert one.shape == (50, 1) and len(set(one[:, 0].tolist())) > 1
    big = erasures.sample_ring_regions(1000, 300, 0.25, 5, seed=3)
    assert np.all(np.diff(big, axis=1) > 0) and big.max() < 1000
    same = erasures.sample_ring_regions(1000, 300, 0.25, 5, seed=3)
    assert np.array_equal(big, same)
    assert all(
        _is_ring_interval(a, 30) for a in erasures.sample_ring_regions(30, 7, 50.0, 20, seed=4)
    )


def _ring_moments(n, m, coupling):
    values, probs = erasures.ring_boundary_distribution(n, m, coupling)
    mean = float(np.sum(values * probs))
    return mean, float(np.sum((values - mean) ** 2 * probs))


def test_chain_on_the_ring_matches_the_closed_form_moments():
    n, m = 64, 16
    couplings = [0.0, 0.5, 1.0, 2.0]
    draws = _size(20_000, 500)
    res = erasures.sample_regions(
        graphs.cycle(n), n, m, couplings, draws, burn_in=_size(500, 100), seed=21
    )
    for i, coupling in enumerate(couplings):
        mean, var = _ring_moments(n, m, coupling)
        b = res.boundaries[i].astype(np.float64)
        first = erasures.chain_diagnostics(b)
        assert abs(first["mean"] - mean) < 4 * first["std_error"], (coupling, first, mean)
        second = erasures.chain_diagnostics((b - mean) ** 2)  # its mean estimates Var b
        assert abs(second["mean"] - var) < 4 * second["std_error"], (coupling, second, var)


# ---------------------------------------------------------------------------------------
# 5. Large coupling: intervals, end intervals, and what replica exchange buys
# ---------------------------------------------------------------------------------------


def test_ring_states_at_large_coupling_are_intervals():
    n, m = 32, 8
    ladder = [0.0, 1.0, 2.0, 4.0, 8.0]
    res = erasures.sample_regions(
        graphs.cycle(n), n, m, ladder, _size(2_000, 300), burn_in=200, seed=8
    )
    top = res.regions[-1]
    assert np.all(res.boundaries[-1] == 2)
    assert all(_is_ring_interval(a, n) for a in top)
    assert len({tuple(a) for a in top.tolist()}) > 1  # the interval moves around the ring
    assert np.all(res.exchange_acceptance > 0)


def test_open_path_at_large_coupling_keeps_only_the_two_end_intervals():
    n, m = 24, 6
    edges = graphs.path(n)
    ladder = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
    draws = _size(4_000, 400)
    res = erasures.sample_regions(edges, n, m, ladder, draws, burn_in=200, seed=12)
    left, right = list(range(m)), list(range(n - m, n))
    top = res.regions[-1].tolist()
    assert all(a in (left, right) for a in top)  # J = 16: P(other) < 2e-6 per draw
    on_left = np.array([a == left for a in top], dtype=np.float64)
    assert 0 < on_left.mean() < 1  # both ends appear

    # J = 8: the share of end intervals (b = 1) is Q_8(b = 1), with 4 standard errors.
    counts = erasures.boundary_counts(edges, n, m)
    exact = counts[1] * math.exp(-8.0) / np.sum(counts * np.exp(-8.0 * np.arange(len(counts))))
    ends = (res.boundaries[-2] == 1).astype(np.float64)
    d = erasures.chain_diagnostics(ends)
    assert abs(d["mean"] - exact) < 4 * math.sqrt(exact * (1 - exact) / d["ess"]) + 1e-12


def _left_share_at_top(seed, tempering, n, m, ladder, draws):
    res = erasures.sample_regions(
        graphs.path(n), n, m, ladder, draws, burn_in=100, tempering=tempering, seed=seed
    )
    return np.mean(res.regions[-1] < n // 2, axis=1), res


def test_without_replica_exchange_r_hat_reveals_stuck_chains():
    """At J = 8 on the open path the two end intervals are separated by a barrier
    exp(-8): a lone chain stays at the end it found, so chains of different seeds
    disagree (R-hat far above 1). Replica exchange carries states over the barrier
    through the lower couplings, and R-hat comes back to 1."""
    n, m = 12, 3
    ladder = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    draws = _size(4_000, 400)
    seeds = [1, 2, 3, 4]
    stuck = []
    for seed in seeds:
        share, res = _left_share_at_top(seed, False, n, m, ladder, draws)
        assert np.all(np.isnan(res.exchange_acceptance))
        stuck.append(share)
    assert erasures.chain_diagnostics(np.array(stuck))["rhat"] > 1.5
    mixed = []
    for seed in seeds:
        share, res = _left_share_at_top(seed, True, n, m, ladder, draws)
        assert np.all(res.exchange_acceptance > 0.05)
        mixed.append(share)
    assert erasures.chain_diagnostics(np.array(mixed))["rhat"] < _size(1.05, 1.2)


# ---------------------------------------------------------------------------------------
# 6. Proposition 2: the mean boundary never increases with J
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_mean_boundary_is_nonincreasing_with_derivative_minus_variance(seed):
    n = 10
    edges = graphs.erdos_renyi(n, 0.35, seed=seed)
    grid = np.linspace(-3.0, 3.0, 61)
    h = 1e-4
    for m in (3, 5):
        counts = erasures.boundary_counts(edges, n, m)
        mom = erasures.boundary_moments(counts, grid)
        assert set(mom) >= {"couplings", "log_partition", "mean", "variance"}
        assert np.all(np.diff(mom["mean"]) <= 1e-12)
        assert np.all(mom["variance"] >= 0)
        up = erasures.boundary_moments(counts, grid + h)
        down = erasures.boundary_moments(counts, grid - h)
        slope = (up["mean"] - down["mean"]) / (2 * h)
        assert slope == pytest.approx(-mom["variance"], rel=1e-5, abs=1e-6)
        slope = (up["log_partition"] - down["log_partition"]) / (2 * h)
        assert slope == pytest.approx(-mom["mean"], rel=1e-6, abs=1e-6)
        zero = erasures.boundary_moments(counts, 0.0)
        assert zero["log_partition"][0] == pytest.approx(math.log(math.comb(n, m)))
        assert zero["mean"][0] == pytest.approx(
            np.sum(np.arange(len(counts)) * counts) / counts.sum()
        )


def test_boundary_moments_are_stable_at_extreme_couplings():
    counts = np.array([0, 2, 0, 10**12, 7])
    mom = erasures.boundary_moments(counts, [-1e4, 1e4])
    assert np.all(np.isfinite(mom["log_partition"]))
    assert mom["mean"] == pytest.approx([4.0, 1.0])
    assert mom["variance"] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert mom["log_partition"][1] == pytest.approx(math.log(2) - 1e4)


def test_chain_mean_boundary_decreases_with_coupling_on_watts_strogatz():
    n, m = 64, 16
    edges = graphs.watts_strogatz(n, 4, 0.1, seed=1)
    couplings = [0.0, 0.5, 1.0, 2.0]
    res = erasures.sample_regions(edges, n, m, couplings, _size(4_000, 400), seed=31)
    stats = [erasures.chain_diagnostics(res.boundaries[i].astype(float)) for i in range(4)]
    for lo, hi in itertools.pairwise(stats):
        margin = 4 * math.hypot(lo["std_error"], hi["std_error"])
        assert hi["mean"] <= lo["mean"] + margin


# ---------------------------------------------------------------------------------------
# 7. chain_diagnostics
# ---------------------------------------------------------------------------------------


def _ar1(rho, size, rng):
    eps = rng.standard_normal(size)
    eps[0] /= math.sqrt(1 - rho * rho)  # start in the stationary law
    return np.fromiter(itertools.accumulate(eps, lambda a, e: rho * a + e), float, size)


@pytest.mark.parametrize("rho", [0.5, 0.9])
def test_integrated_autocorrelation_time_of_ar1(rho):
    rng = np.random.default_rng(12345)
    tau = (1 + rho) / (1 - rho)
    one = erasures.chain_diagnostics(_ar1(rho, 1_000_000, rng))
    assert one["tau_int"] == pytest.approx(tau, rel=0.1)
    assert one["ess"] == pytest.approx(1_000_000 / one["tau_int"])
    assert one["window"] >= 5 * one["tau_int"]
    assert one["std_error"] == pytest.approx(math.sqrt(one["variance"] / one["ess"]))
    assert "rhat" not in one
    chains = np.stack([_ar1(rho, 250_000, rng) for _ in range(4)])
    many = erasures.chain_diagnostics(chains)
    assert many["tau_int"] == pytest.approx(tau, rel=0.1)
    assert many["ess"] == pytest.approx(1_000_000 / many["tau_int"])
    assert many["chains"] == 4 and many["draws"] == 250_000
    assert abs(many["rhat"] - 1) < 0.01


def test_r_hat_of_agreeing_and_disagreeing_chains():
    rng = np.random.default_rng(7)
    iid = rng.standard_normal((4, 4_000))
    assert abs(erasures.chain_diagnostics(iid)["rhat"] - 1) < 0.01
    shifted = iid.copy()
    shifted[3] += 2.0
    assert erasures.chain_diagnostics(shifted)["rhat"] > 1.2
    wider = iid.copy()
    wider[3] *= 4.0  # same location, other scale: the folded R-hat sees it
    assert erasures.chain_diagnostics(wider)["rhat"] > 1.1
    trend = np.linspace(0.0, 3.0, 4_000)[None, :] + iid[:1]  # one drifting chain: split R-hat
    assert erasures.chain_diagnostics(trend)["rhat"] > 1.2


def test_chain_diagnostics_of_constant_series():
    flat = erasures.chain_diagnostics(np.full(50, 3.0))
    assert flat["mean"] == 3.0 and flat["variance"] == 0.0 and flat["std_error"] == 0.0
    assert flat["tau_int"] == 1.0 and flat["ess"] == 50.0
    same = erasures.chain_diagnostics(np.full((3, 20), 2.0))
    assert math.isnan(same["rhat"])
    apart = erasures.chain_diagnostics(np.repeat([[1.0], [2.0]], 20, axis=1))
    assert apart["rhat"] == math.inf


# ---------------------------------------------------------------------------------------
# 8. Determinism and the chain's stream
# ---------------------------------------------------------------------------------------


def _reference_chain(edges, n, m, couplings, samples, burn_in, thin, tempering, state):
    """The chain transcribed from the draw order in ``sample_chain``'s docstring."""
    gen = ref.Xoshiro([int(w) for w in state])
    c = len(couplings)
    chains = []
    for _ in range(c):
        order = list(range(n))
        for t in range(m):
            j = t + gen.next_below(n - t)
            order[t], order[j] = order[j], order[t]
        chains.append([order[:m], order[m:], _brute_boundary(edges, order[:m])])
    slot = list(range(c))
    moves = np.zeros((c, 4), dtype=np.int64)
    regions = np.zeros((c, samples, m), dtype=np.int64)
    bounds = np.zeros((c, samples), dtype=np.int64)
    done = 0
    for r in range(samples):
        while done < burn_in + r * thin:
            for i, coupling in enumerate(couplings):
                chain = chains[slot[i]]
                inside, outside = chain[0], chain[1]
                for _ in range(n):
                    a = gen.next_below(m)
                    p = gen.next_below(n - m)
                    u, v = inside[a], outside[p]
                    d = _brute_boundary(edges, [*inside[:a], v, *inside[a + 1 :]]) - chain[2]
                    moves[i, 0] += 1
                    if coupling * d > 0 and not gen.next_double() < math.exp(-abs(coupling * d)):
                        continue
                    inside[a], outside[p] = v, u
                    chain[2] += d
                    moves[i, 1] += 1
            if tempering and c > 1:
                for i in range(c - 1):
                    x = (couplings[i] - couplings[i + 1]) * (
                        chains[slot[i]][2] - chains[slot[i + 1]][2]
                    )
                    moves[i, 2] += 1
                    if x >= 0 or gen.next_double() < math.exp(x):
                        slot[i], slot[i + 1] = slot[i + 1], slot[i]
                        moves[i, 3] += 1
            done += 1
        for i in range(c):
            regions[i, r] = sorted(chains[slot[i]][0])
            bounds[i, r] = chains[slot[i]][2]
    return regions, bounds, moves, gen.s


@pytest.mark.parametrize("tempering", [True, False])
def test_kernel_follows_the_documented_draw_order(tempering):
    n, m = 11, 4
    edges = graphs.watts_strogatz(n, 4, 0.3, seed=5)
    couplings = np.array([-0.5, 0.0, 0.6, 1.4])
    samples, burn_in, thin = 5, 3, 2
    state = seed_state(77)
    expected = _reference_chain(
        edges, n, m, couplings.tolist(), samples, burn_in, thin, tempering, state
    )
    indptr, indices = kernel.csr(n, edges)
    regions = np.empty((4, samples, m), dtype=np.int64)
    bounds = np.empty((4, samples), dtype=np.int64)
    moves = np.empty((4, 4), dtype=np.int64)
    kernel.sample_chain(
        indptr, indices, n, m, couplings, samples, burn_in, thin, tempering,
        state, regions, bounds, moves,
    )  # fmt: skip
    assert np.array_equal(regions, expected[0])
    assert np.array_equal(bounds, expected[1])
    assert np.array_equal(moves, expected[2])
    assert [int(w) for w in state] == expected[3]  # the state advanced by exactly these draws
    assert moves[:, 1].min() > 0 and (moves[:3, 3].sum() > 0) == tempering


def test_same_seed_same_regions_other_seed_other_regions():
    edges = graphs.watts_strogatz(20, 4, 0.2, seed=1)
    run = lambda seed, **kw: erasures.sample_regions(  # noqa: E731
        edges, 20, 6, [0.0, 1.5, 0.5], 50, burn_in=10, thin=2, seed=seed, **kw
    )
    a, b, c = run(5), run(5), run(6)
    assert np.array_equal(a.regions, b.regions) and np.array_equal(a.boundaries, b.boundaries)
    assert np.array_equal(a.acceptance, b.acceptance)
    assert not np.array_equal(a.regions, c.regions)
    assert not np.array_equal(a.regions, run(5, tempering=False).regions)
    fresh = run(None)
    assert isinstance(fresh.seed, int) and 0 <= fresh.seed < 2**53
    assert np.array_equal(run(fresh.seed).regions, fresh.regions)
    assert a.seed == 5


# The chain's own stream contract: these bytes change only if the draw order documented in
# sparsegf2._kernel.erasure_chain.sample_chain (or the generator) changes.
PINNED_DIGEST = "c6067e313dd0b4a51b7f8e65b7adfafb3871890ea944a4e3ff639c0ff46696fe"


def test_pinned_digest_of_a_small_run():
    res = erasures.sample_regions(
        graphs.watts_strogatz(12, 4, 0.25, seed=7), 12, 4, [1.5, -0.5, 0.0, 0.75], 8,
        burn_in=3, thin=2, seed=2026,
    )  # fmt: skip
    h = hashlib.sha256()
    for array in (res.regions, res.boundaries):
        h.update(array.astype("<i8").tobytes())
    for array in (res.acceptance, res.exchange_acceptance):
        h.update(array.astype("<f8").tobytes())
    assert h.hexdigest() == PINNED_DIGEST


def test_results_are_read_only():
    res = erasures.sample_regions(graphs.cycle(6), 6, 2, [0.0, 1.0], 4, seed=1)
    with pytest.raises(ValueError):
        res.regions[0, 0, 0] = 5
    with pytest.raises(AttributeError):
        res.size = 3  # type: ignore[misc]


# ---------------------------------------------------------------------------------------
# 9. Validation
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"size": 7}, r"size must be in \[0, 6\]"),
        ({"size": -1}, r"size must be in \[0, 6\]"),
        ({"samples": 0}, "samples must be at least 1"),
        ({"burn_in": -1}, "burn_in must be at least 0"),
        ({"thin": 0}, "thin must be at least 1"),
        ({"samples": 2.5}, "samples must be an integer"),
        ({"thin": True}, "thin must be an integer"),
        ({"couplings": [0.0, math.inf]}, "finite"),
        ({"couplings": [math.nan]}, "finite"),
        ({"couplings": [0.5, 1.0, 0.5]}, "0.5 appears 2 times"),
        ({"couplings": []}, "at least one coupling"),
        ({"couplings": [[0.0, 1.0]]}, "one coupling or a list"),
        ({"couplings": ["a"]}, "couplings must be numbers"),
        ({"num_nodes": 0}, "num_nodes must be at least 1"),
        ({"edges": [(0, 9)]}, "out of range"),
        ({"edges": [(1, 1)]}, "self-loops"),
        ({"tempering": "yes"}, "tempering must be True or False"),
        ({"seed": -3}, "seed"),
    ],
)
def test_sample_regions_validation(kwargs, message):
    args = {
        "edges": graphs.cycle(6), "num_nodes": 6, "size": 2, "couplings": [0.0, 1.0],
        "samples": 3, "burn_in": 1, "thin": 1, "tempering": True, "seed": 1,
    }  # fmt: skip
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        erasures.sample_regions(
            args.pop("edges"), args.pop("num_nodes"), args.pop("size"),
            args.pop("couplings"), args.pop("samples"), **args,
        )  # fmt: skip


def test_other_validation_messages():
    ring = graphs.cycle(6)
    with pytest.raises(ValueError, match="node 3 appears twice"):
        erasures.boundary(ring, 6, [3, 1, 3])
    with pytest.raises(ValueError, match=r"node 6 outside \[0, 6\)"):
        erasures.boundary(ring, 6, [6])
    with pytest.raises(ValueError, match="region must be a list of node indices"):
        erasures.boundary(ring, 6, [[1, 2]])
    with pytest.raises(ValueError, match=r"size must be in \[0, 6\]"):
        erasures.boundary_counts(ring, 6, 9)
    with pytest.raises(ValueError, match="budget must be at least 1"):
        erasures.boundary_counts(ring, 6, 2, budget=0)
    with pytest.raises(ValueError, match="counts must be"):
        erasures.boundary_moments([1, -2], [0.0])
    with pytest.raises(ValueError, match="counts must be"):
        erasures.boundary_moments([0, 0], [0.0])
    with pytest.raises(ValueError, match="finite"):
        erasures.boundary_moments([1, 2], [math.inf])
    with pytest.raises(ValueError, match="n >= 3"):
        erasures.ring_run_count(2, 1, 1)
    with pytest.raises(ValueError, match=r"m must be in \[0, 8\]"):
        erasures.ring_boundary_distribution(8, 9, 0.0)
    with pytest.raises(ValueError, match="finite"):
        erasures.ring_partition_function(8, 3, math.inf)
    with pytest.raises(ValueError, match="samples must be at least 1"):
        erasures.sample_ring_regions(8, 3, 0.0, 0, seed=1)
    with pytest.raises(ValueError, match="at least 4 draws"):
        erasures.chain_diagnostics(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="at least 2 draws"):
        erasures.chain_diagnostics([1.0])
    with pytest.raises(ValueError, match="1-D series or a 2-D"):
        erasures.chain_diagnostics(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="finite"):
        erasures.chain_diagnostics([1.0, math.nan, 2.0])
