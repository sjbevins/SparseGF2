"""Geometry-correlated erasures, code side: which regions of a code fail, and the analysis.

A code C on n qubits with reference R loses l(A) = I(A:R) bits when the qubits A are
erased and fails when F(A) = 1[l(A) > 0]. On a graph G, b(A) is the number of edges
leaving A and Q_J(A | m) = exp(-J b(A)) / Z_m(J) on the regions of size m. Checked here:

1. ``StabilizerCode.boundary_failure_counts``: ``total`` is the graph's density of states
   N_m(b) (``erasures.boundary_counts``, and the ring's closed form), ``failed`` is
   U_m(b) from the brute-force GF(2) oracle over every subset, binned by boundary, on
   random codes (k = 0 included) and graphs (cycle, path, Watts-Strogatz, Erdős–Rényi),
   n up to 12 with full enumeration, and n = 70 (two words) for m <= 2; and
   sum_b U_m(b) equals ``uncorrectable_counts`` exactly. ``region_failures`` against the
   oracle on random regions.
2. ``analysis.exact_correlated_failure``: at J = 0 it equals ``exact_failure_by_size``
   bit for bit; for large |J| it equals U_m(b_min) / N_m(b_min) (or b_max for J < 0); it
   equals a direct sum of Q_J(A) F(A) over all subsets for J from -1.5 to 2.5; it is
   stable at J = +-1e4; leading axes broadcast and boundary axes are padded.
3. ``StabilizerCode.correlated_failure`` with the swap chain (a Watts-Strogatz graph, so
   no exact sampler): for J in {0, 1, 3} every size's estimate lies within 4 standard
   errors (from the effective sample size of the failure indicators, plus a
   discreteness allowance 2/ESS) of the exact P_J(m); the regions of size m are those of
   ``erasures.sample_regions(seed=[seed, m])``.
4. ``window_first_failure``, ``interval_failure_by_size`` and
   ``end_window_failure_by_size`` against brute force over every window (periodic and
   open, with a custom order), and the contiguous distance as the smallest first failure.
5. The brickwork ring (``graphs.cycle``, ``schedule="matching"``, purification picture,
   n = 12 and 16, p in {0.05, 0.2}): on real monitored-circuit codes, the identities of 1
   and 2, the ring's closed form for N_m(b), the J -> inf limit U_m(2)/N_m(2) equal to
   ``interval_failure_by_size``, the samplers (exact and chain) at J = 0 against the exact
   uniform curve and at J = 6 against ``interval_failure_by_size`` (allowing the exact
   weight Q_6(b > 2) of non-intervals), and Phi_{C,0}(e) from the exact curve equal to
   ``erasure_failure_probability`` from exact counts.
6. ``binomial_window``, ``correlated_failure_curve``, ``failure_probability_from_curve``,
   ``tolerance_from_curve`` (monotone and non-monotone curves) and ``mean_over_codes`` on
   hand-made inputs.
7. Validation messages, read-only results, determinism, and the kernels' edge cases.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numba
import numpy as np
import pytest

import _gf2_oracle as oracle
import sparsegf2 as sg
from sparsegf2 import analysis, erasures, graphs, mipt
from sparsegf2._kernel import erasure_failure as kernel
from sparsegf2.codes import BoundaryCounts, CorrelatedFailure, StabilizerCode

# The weekly job runs the kernels as plain Python (NUMBA_DISABLE_JIT=1); there the runs
# are shorter. Every statistical tolerance follows from the samples actually drawn.
JIT_FREE = bool(numba.config.DISABLE_JIT)


def _size(compiled, plain):
    return plain if JIT_FREE else compiled


# ---------------------------------------------------------------------------------------
# Codes, graphs and brute force
# ---------------------------------------------------------------------------------------


def _random_code(n: int, r: int, gates: int, measured: int, seed: int) -> StabilizerCode:
    """A random Clifford state on n + r qubits, ``measured`` system qubits measured in Z,
    the last r qubits the reference (r = 0: a pure state, k = 0)."""
    rng = np.random.default_rng(seed)
    sim = sg.Simulator(n + r, phases=False, seed=seed)
    for _ in range(gates):
        a, b = rng.choice(n + r, size=2, replace=False)
        sim.clifford_2q(int(rng.integers(11520)), int(a), int(b))
    for q in rng.choice(n, size=measured, replace=False):
        sim.measure(int(q))
    if r == 0:
        xs, zs, _ = sim.stabilizer_arrays()
        return StabilizerCode(np.concatenate([xs, zs], axis=1) % 2)
    return StabilizerCode.from_simulator(sim, reference=range(n, n + r))


# k = 3: [[4,2,2]] on qubits 0-3, [[5,1,3]] on 4-8, qubit 9 frozen
FOUR_FIVE_FROZEN = [
    "XXXX______",
    "ZZZZ______",
    "____XZZX__",
    "_____XZZX_",
    "____X_XZZ_",
    "____ZX_XZ_",
    "_________Z",
]

CODES = {
    "scrambled_k3": lambda: _random_code(10, 3, 40, 0, seed=1),
    "measured_k2": lambda: _random_code(12, 2, 50, 3, seed=2),
    "shallow_k1": lambda: _random_code(9, 1, 6, 0, seed=3),
    "pure_k0": lambda: _random_code(8, 0, 30, 0, seed=4),
    "blocks_k3": lambda: StabilizerCode(FOUR_FIVE_FROZEN),
}


def _graph(kind: str, n: int) -> np.ndarray:
    return {
        "cycle": lambda: graphs.cycle(n),
        "path": lambda: graphs.path(n),
        "watts_strogatz": lambda: graphs.watts_strogatz(n, 4, 0.3, seed=n),
        "erdos_renyi": lambda: graphs.erdos_renyi(n, 0.3, seed=n + 1),
    }[kind]()


CASES = [
    ("scrambled_k3", "cycle", 10),
    ("scrambled_k3", "erdos_renyi", 10),
    ("measured_k2", "watts_strogatz", 12),
    ("measured_k2", "path", 6),
    ("shallow_k1", "path", 9),
    ("shallow_k1", "watts_strogatz", 9),
    ("pure_k0", "cycle", 8),
    ("pure_k0", "erdos_renyi", 8),
    ("blocks_k3", "cycle", 10),
    ("blocks_k3", "watts_strogatz", 10),
]


def _brute_boundary(edges, region) -> int:
    inside = set(region)
    return sum((int(a) in inside) != (int(b) in inside) for a, b in edges)


def _oracle(code: StabilizerCode) -> oracle.Code:
    return oracle.Code(list(code.stabilizer_matrix()), code.num_qubits)


def _brute_counts(code: StabilizerCode, edges, w: int):
    """(failed, total) by listing every subset: U_m(b) and N_m(b), shape (w, |E| + 1)."""
    n = code.num_qubits
    gf2 = _oracle(code)
    failed = np.zeros((w, len(edges) + 1), dtype=np.int64)
    total = np.zeros_like(failed)
    for m in range(1, w + 1):
        for region in itertools.combinations(range(n), m):
            b = _brute_boundary(edges, region)
            total[m - 1, b] += 1
            failed[m - 1, b] += gf2.lost(list(region)) > 0
    return failed, total


def _pad(a: np.ndarray, width: int) -> np.ndarray:
    return np.concatenate([a, np.zeros(a.shape[:-1] + (width - a.shape[-1],), a.dtype)], -1)


# ---------------------------------------------------------------------------------------
# 1. Boundary-resolved counts and region failures
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("code_name", "graph", "max_size"), CASES)
def test_boundary_failure_counts_match_brute_force(code_name, graph, max_size):
    code = CODES[code_name]()
    n = code.num_qubits
    edges = _graph(graph, n)
    counts = code.boundary_failure_counts(edges, max_size)
    assert isinstance(counts, BoundaryCounts)
    assert (counts.max_size, counts.num_qubits) == (max_size, n)
    assert counts.num_logical == code.num_logical
    assert counts.failed.dtype == counts.total.dtype == np.int64
    assert counts.failed.shape == counts.total.shape
    assert counts.total[:, -1].any()  # trimmed to the largest boundary that occurs

    failed, total = _brute_counts(code, edges, max_size)
    width = counts.total.shape[1]
    assert not total[:, width:].any() and not failed[:, width:].any()
    np.testing.assert_array_equal(counts.total, total[:, :width])
    np.testing.assert_array_equal(counts.failed, failed[:, :width])
    for m in range(1, max_size + 1):
        dos = erasures.boundary_counts(edges, n, m)
        np.testing.assert_array_equal(counts.total[m - 1], _pad(dos, width))
        assert counts.total[m - 1].sum() == math.comb(n, m)
    np.testing.assert_array_equal(counts.failed.sum(axis=1), code.uncorrectable_counts(max_size))
    if code.num_logical == 0:
        assert not counts.failed.any()


def test_the_cases_cover_what_matters():
    """Codes with k = 0, failures at several boundaries of one size, and full enumeration."""
    seen = set()
    for code_name, graph, max_size in CASES:
        code = CODES[code_name]()
        counts = code.boundary_failure_counts(_graph(graph, code.num_qubits), max_size)
        if code.num_logical == 0:
            seen.add("k = 0")
        if max_size == code.num_qubits:
            seen.add("full")
        if any(np.count_nonzero(row) >= 2 for row in counts.failed):
            seen.add("several boundaries fail")
        partial = (counts.failed > 0) & (counts.failed < counts.total)
        if partial.any():
            seen.add("a boundary class partly fails")
    assert seen == {"k = 0", "full", "several boundaries fail", "a boundary class partly fails"}


def test_boundary_failure_counts_on_a_code_wider_than_one_word():
    """n = 70 with k = 2: the code's columns span two words (S part and N(S) part)."""
    code = _random_code(70, 2, 150, 4, seed=7)
    assert code.num_logical == 2
    edges = graphs.watts_strogatz(70, 4, 0.2, seed=1)
    counts = code.boundary_failure_counts(edges, 2)
    assert counts.failed.sum() > 0  # some pairs fail: the comparison is not vacuous
    failed, total = _brute_counts(code, edges, 2)
    width = counts.total.shape[1]
    np.testing.assert_array_equal(counts.total, total[:, :width])
    np.testing.assert_array_equal(counts.failed, failed[:, :width])
    np.testing.assert_array_equal(counts.failed.sum(axis=1), code.uncorrectable_counts(2))


@pytest.mark.parametrize("code_name", list(CODES))
def test_region_failures_match_the_oracle(code_name, rng):
    code = CODES[code_name]()
    n = code.num_qubits
    gf2 = _oracle(code)
    for m in range(n + 1):
        regions = np.array([rng.permutation(n)[:m] for _ in range(25)], dtype=np.int64)
        regions = regions.reshape(25, m)
        got = code.region_failures(regions)
        assert got.dtype == bool and got.shape == (25,)
        expected = [gf2.lost([int(q) for q in row]) > 0 for row in regions]
        np.testing.assert_array_equal(got, expected)
    assert code.region_failures([[0, 1], [1, 2]]).shape == (2,)  # a list of lists
    assert code.region_failures(np.zeros((0, 3), dtype=np.int64)).shape == (0,)


# ---------------------------------------------------------------------------------------
# 2. Exact P_J(m) from the counts
# ---------------------------------------------------------------------------------------


def _direct(code: StabilizerCode, edges, m: int, coupling: float) -> float:
    """sum_A Q_J(A | m) F(A) over every m-subset, by brute force (b and F per subset)."""
    gf2 = _oracle(code)
    rows = [
        (_brute_boundary(edges, a), gf2.lost(list(a)) > 0)
        for a in itertools.combinations(range(code.num_qubits), m)
    ]
    shift = min(b for b, _ in rows) if coupling >= 0 else max(b for b, _ in rows)
    weights = [math.exp(-coupling * (b - shift)) for b, _ in rows]
    return sum(w for w, (_, f) in zip(weights, rows, strict=True) if f) / sum(weights)


@pytest.mark.parametrize(
    ("code_name", "graph"), [("measured_k2", "watts_strogatz"), ("blocks_k3", "erdos_renyi")]
)
def test_exact_correlated_failure_equals_the_direct_sum_and_its_limits(code_name, graph):
    code = CODES[code_name]()
    n = code.num_qubits
    edges = _graph(graph, n)
    counts = code.boundary_failure_counts(edges, n)
    couplings = [-1.5, -0.3, 0.0, 0.4, 1.0, 2.5]
    exact = counts.exact(couplings)
    assert exact.shape == (len(couplings), n)
    np.testing.assert_array_equal(
        exact, analysis.exact_correlated_failure(counts.failed, counts.total, couplings)
    )
    for i, coupling in enumerate(couplings):
        for m in range(1, n + 1):
            assert exact[i, m - 1] == pytest.approx(_direct(code, edges, m, coupling), rel=1e-12)
    # J = 0: the uniform P_C(m), bit for bit
    uniform = analysis.exact_failure_by_size(code.uncorrectable_counts(n), n)
    np.testing.assert_array_equal(counts.exact(0.0)[0], uniform)
    # J -> +-inf: the failing fraction of the regions of smallest (largest) boundary. At
    # |J| = 30 the rest weighs at most C(n, m) e^-30 relative to it (every other boundary
    # differs by at least 1); at |J| = 1e4 that underflows and the limit is reached exactly.
    lowest = np.argmax(counts.total > 0, axis=1)
    highest = counts.total.shape[1] - 1 - np.argmax(counts.total[:, ::-1] > 0, axis=1)
    rows = np.arange(n)
    slack = np.array([math.comb(n, m) for m in range(1, n + 1)]) * math.exp(-30)
    for sign, b in ((1, lowest), (-1, highest)):
        limit = counts.failed[rows, b] / counts.total[rows, b]
        gap = np.abs(counts.exact(30.0 * sign)[0] - limit)
        assert np.all(gap <= slack)
        np.testing.assert_array_equal(counts.exact(1e4 * sign)[0], limit)
    assert np.all((exact >= 0) & (exact <= 1))


def test_exact_correlated_failure_broadcasts_and_pads():
    codes = [CODES["scrambled_k3"](), CODES["blocks_k3"]()]
    edges = graphs.cycle(10)
    counts = [c.boundary_failure_counts(edges, 4) for c in codes]
    failed = np.stack([c.failed for c in counts])  # (2, 4, B + 1), one shared total
    shared = analysis.exact_correlated_failure(failed, counts[0].total, [0.0, 1.5])
    assert shared.shape == (2, 2, 4)
    for t in range(2):
        np.testing.assert_array_equal(shared[t], counts[t].exact([0.0, 1.5]))
    # total built from erasures.boundary_counts rows of different lengths, padded
    rows = [erasures.boundary_counts(edges, 10, m) for m in range(1, 5)]
    width = max(len(r) for r in rows)
    total = np.stack([_pad(r, width) for r in rows])
    trimmed = counts[1].failed[:, : int(np.flatnonzero(counts[1].failed.any(axis=0))[-1]) + 1]
    np.testing.assert_array_equal(
        analysis.exact_correlated_failure(trimmed, total, 0.7), counts[1].exact(0.7)
    )
    # a single coupling as a number gives one row
    assert analysis.exact_correlated_failure(counts[1].failed, counts[1].total, 2.0).shape == (1, 4)


# ---------------------------------------------------------------------------------------
# 3. The chain estimate against the exact answer
# ---------------------------------------------------------------------------------------


def _within_errors(code, edges, result, exact, **sampler):
    """|estimate - exact| <= 4 se + 2 / ESS for every coupling and size, se from the
    effective sample size of the failure indicators of the very regions the estimate used
    (reproduced with seed [seed, m]). The 2 / ESS allows for the discreteness of rare
    failures (a single failing record when ESS p is 0.01, say). Returns the largest |z|
    where the normal approximation holds (ESS p (1 - p) >= 10)."""
    worst = 0.0
    n = code.num_qubits
    for j, m in enumerate(result.sizes.tolist()):
        drawn = erasures.sample_regions(
            edges,
            n,
            m,
            result.couplings,
            result.samples,
            seed=[result.seed, m],
            burn_in=result.burn_in,
            thin=result.thin,
            tempering=result.tempering,
            **sampler,
        )
        assert drawn.method == result.method
        for i in range(len(result.couplings)):
            fails = code.region_failures(drawn.regions[i])
            assert fails.sum() == result.failures[i, j]  # the seed contract
            p = exact[i, m - 1]
            if drawn.method == "exact":
                ess = float(result.samples)
            else:
                ess = erasures.chain_diagnostics(fails.astype(float))["ess"]
            se = math.sqrt(p * (1 - p) / ess)
            gap = abs(result.failure[i, j] - p)
            assert gap <= 4 * se + 2 / ess, (result.couplings[i], m, result.failure[i, j], p, se)
            if ess * p * (1 - p) >= 10:
                worst = max(worst, gap / se)
    return worst


def test_chain_estimate_agrees_with_the_exact_failure_probability():
    code = CODES["measured_k2"]()
    n = code.num_qubits
    edges = _graph("watts_strogatz", n)
    couplings = [0.0, 1.0, 3.0]
    exact = code.boundary_failure_counts(edges, n).exact(couplings)
    sizes = _size(list(range(1, n)), [1, 3, 6, 9])
    samples = _size(20_000, 400)
    result = code.correlated_failure(edges, couplings, sizes, samples, seed=11)
    assert isinstance(result, CorrelatedFailure)
    assert result.method == "chain" and result.geometry == "other"
    assert result.failure.shape == (3, len(sizes))
    np.testing.assert_array_equal(result.failure, result.failures / samples)
    # the comparison is not vacuous: P is strictly between 0 and 1 at several sizes, and
    # differs between couplings
    assert np.count_nonzero((exact > 0.05) & (exact < 0.95)) >= 6
    assert np.max(np.abs(exact[2] - exact[0])) > 0.05
    worst = _within_errors(code, edges, result, exact)
    assert worst < 4
    # J = 0 is the uniform P_C(m)
    np.testing.assert_array_equal(
        exact[0], analysis.exact_failure_by_size(code.uncorrectable_counts(n), n)
    )
    # Proposition 2 on the recorded boundaries
    assert np.all(np.diff(result.mean_boundary, axis=0) <= 0)


def test_correlated_failure_reports_its_sampler():
    code = CODES["scrambled_k3"]()
    edges = graphs.watts_strogatz(10, 4, 0.3, seed=2)
    r = code.correlated_failure(edges, [2.0, 0.0, 1.0], [3, 7], 50, burn_in=20, seed=3)
    assert r.couplings.tolist() == [2.0, 0.0, 1.0] and r.sizes.tolist() == [3, 7]
    assert r.exchange_pairs.tolist() == [[0.0, 1.0], [1.0, 2.0]]
    assert r.exchange_acceptance.shape == (2, 2) and np.all(np.isfinite(r.exchange_acceptance))
    assert np.all(r.rungs >= 3) and r.acceptance.shape == (3, 2)
    assert (r.num_qubits, r.num_logical) == (10, code.num_logical)
    assert (r.burn_in, r.thin, r.tempering) == (20, 1, True)
    # ladder="given": the pairs are the ladder's own, and a coarse one warns per size
    coarse = code.correlated_failure(edges, [0.0, 4.0], [5], 60, burn_in=10, seed=1, ladder="given")
    drawn = erasures.sample_regions(
        edges, 10, 5, [0.0, 4.0], 60, burn_in=10, seed=[1, 5], ladder="given"
    )
    assert coarse.exchange_acceptance[0, 0] == drawn.exchange_acceptance[0]
    assert coarse.rungs.tolist() == [2]
    assert coarse.warnings == tuple(f"size 5: {w}" for w in drawn.warnings)
    assert any("replica exchange between J = 0 and J = 4" in w for w in coarse.warnings)
    # exact samples on a cycle: no exchanges, no acceptance
    ring = code.correlated_failure(graphs.cycle(10), [0.0, 5.0], [4], 30, seed=2)
    assert (ring.method, ring.geometry) == ("exact", "cycle")
    assert np.all(np.isnan(ring.acceptance)) and np.all(np.isnan(ring.exchange_acceptance))
    assert ring.warnings == ()


def test_correlated_failure_is_reproducible_and_reports_a_fresh_seed():
    code = CODES["shallow_k1"]()
    edges = graphs.erdos_renyi(9, 0.4, seed=5)
    a = code.correlated_failure(edges, [0.0, 1.0], None, 40, burn_in=10, seed=[4, 2])
    b = code.correlated_failure(edges, [0.0, 1.0], None, 40, burn_in=10, seed=[4, 2])
    np.testing.assert_array_equal(a.failures, b.failures)
    assert a.seed == (4, 2) and a.sizes.tolist() == list(range(1, 9))
    other = code.correlated_failure(edges, [0.0, 1.0], None, 40, burn_in=10, seed=5)
    assert not np.array_equal(a.mean_boundary, other.mean_boundary)
    fresh = code.correlated_failure(edges, [0.5], [2, 4], 40, burn_in=10)
    again = code.correlated_failure(edges, [0.5], [2, 4], 40, burn_in=10, seed=fresh.seed)
    np.testing.assert_array_equal(fresh.failures, again.failures)
    np.testing.assert_array_equal(fresh.mean_boundary, again.mean_boundary)
    with pytest.raises(ValueError, match="read-only"):
        a.failure[0, 0] = 1.0


def test_a_code_without_logical_qubits_never_fails():
    code = CODES["pure_k0"]()
    edges = graphs.watts_strogatz(8, 4, 0.3, seed=1)
    r = code.correlated_failure(edges, [0.0, 2.0], [0, 1, 4, 8], 30, burn_in=10, seed=1)
    assert r.num_logical == 0 and not r.failures.any() and not r.failure.any()
    assert np.all(np.isfinite(r.mean_boundary))
    assert r.mean_boundary[:, [0, 3]].tolist() == [[0.0, 0.0], [0.0, 0.0]]  # m = 0 and m = n


def test_sizes_zero_and_n_are_exact():
    code = CODES["blocks_k3"]()
    r = code.correlated_failure(graphs.path(10), [1.0], [0, 10], 20, seed=1)
    assert r.failure.tolist() == [[0.0, 1.0]]


# ---------------------------------------------------------------------------------------
# 4. Contiguous windows
# ---------------------------------------------------------------------------------------


def _brute_windows(code: StabilizerCode, order, periodic: bool) -> np.ndarray:
    n = code.num_qubits
    gf2 = _oracle(code)
    out = []
    for s in range(n):
        limit = n if periodic else n - s
        found = limit + 1
        for length in range(1, limit + 1):
            window = [int(order[(s + j) % n]) for j in range(length)]
            if gf2.lost(window) > 0:
                found = length
                break
        out.append(found)
    return np.array(out)


@pytest.mark.parametrize("code_name", list(CODES))
@pytest.mark.parametrize("periodic", [True, False], ids=["ring", "open"])
@pytest.mark.parametrize("custom", [False, True], ids=["identity", "permuted"])
def test_windows_match_brute_force(code_name, periodic, custom, rng):
    code = CODES[code_name]()
    n = code.num_qubits
    order = rng.permutation(n) if custom else np.arange(n)
    arg = order if custom else None
    first = code.window_first_failure(arg, periodic=periodic)
    assert first.dtype == np.int64
    np.testing.assert_array_equal(first, _brute_windows(code, order, periodic))
    gf2 = _oracle(code)
    curve = code.interval_failure_by_size(arg, periodic=periodic)
    expected = [0.0]
    for m in range(1, n + 1):
        starts = range(n) if periodic else range(n - m + 1)
        fails = [gf2.lost([int(order[(s + j) % n]) for j in range(m)]) > 0 for s in starts]
        expected.append(Fraction(sum(fails), len(fails)))
    np.testing.assert_array_equal(curve, [float(x) for x in expected])
    assert curve[n] == (1.0 if code.num_logical else 0.0)
    ends = code.end_window_failure_by_size(arg)
    for m in range(n + 1):
        prefix = gf2.lost([int(q) for q in order[:m]]) > 0
        suffix = gf2.lost([int(q) for q in order[n - m :]]) > 0
        assert ends[m] == (prefix + suffix) / 2
    # the contiguous distance is the smallest first failure among starts where one occurs
    limits = np.full(n, n) if periodic else n - np.arange(n)
    occurs = first <= limits
    expected_distance = int(first[occurs].min()) if occurs.any() else 0
    assert code.contiguous_distance(arg, periodic=periodic) == expected_distance


def test_windows_on_a_code_wider_than_one_word():
    code = _random_code(70, 2, 150, 4, seed=7)
    order = np.random.default_rng(3).permutation(70)
    first = code.window_first_failure(order)
    for s in range(0, 70, 7):
        length = int(first[s])
        window = order[(s + np.arange(length)) % 70]
        assert code.lost_information(window) > 0
        assert code.lost_information(window[:-1]) == 0


# ---------------------------------------------------------------------------------------
# 5. The brickwork ring baseline
# ---------------------------------------------------------------------------------------


def _brickwork_codes(n: int, p: float, count: int, seed: int) -> list[StabilizerCode]:
    """Codes of the nearest-neighbour brickwork ring after 4n layers (p < 0.1, the mixed
    phase, where the code keeps distance 2 to 4 at these sizes) or n layers (p = 0.2,
    which purifies quickly)."""
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        schedule="matching",
        layers=4 * n if p < 0.1 else n,
        measurement_rate=p,
        reference="purification",
        checkpoints="final",
        probes=[mipt.code_snapshot()],
    )
    result = spec.run(count, seed=seed)
    return [result.code(t, 0) for t in range(count)]


BRICKWORK = [(12, 0.05), (12, 0.2), (16, 0.05), (16, 0.2)]


def _brickwork(n: int, p: float) -> list[StabilizerCode]:
    """Two codes per (n, p); JIT-free, n = 12 stands in for 16 (other seeds), since the
    plain-Python enumeration of 2^16 subsets per code takes too long."""
    size = 12 if JIT_FREE else n
    return _brickwork_codes(size, p, 2, seed=int(100 * p) + n)


@pytest.mark.parametrize(("n", "p"), BRICKWORK)
def test_brickwork_ring_codes(n, p):
    codes = _brickwork(n, p)
    n = codes[0].num_qubits
    ring = graphs.cycle(n)
    samples = _size(2000, 150)
    sizes = _size(None, [2, 5, 8])
    for code in codes:
        k = code.num_logical
        counts = code.boundary_failure_counts(ring, n)
        # the identities of 1 and 2, and the ring's closed form for N_m(b) (b = 2r)
        np.testing.assert_array_equal(counts.failed.sum(axis=1), code.uncorrectable_counts(n))
        width = counts.total.shape[1]
        for m in range(1, n):
            runs = [
                erasures.ring_run_count(n, m, b // 2) if b % 2 == 0 else 0 for b in range(width)
            ]
            np.testing.assert_array_equal(counts.total[m - 1], runs)
        uniform = analysis.exact_failure_by_size(code.uncorrectable_counts(n), n)
        np.testing.assert_array_equal(counts.exact(0.0)[0], uniform)
        # J -> +inf on the ring: the n intervals, U_m(2) / N_m(2) = interval_failure_by_size
        intervals = code.interval_failure_by_size()
        assert counts.total[: n - 1, 2].tolist() == [n] * (n - 1)
        np.testing.assert_array_equal(counts.failed[: n - 1, 2] / n, intervals[1:n])
        np.testing.assert_allclose(counts.exact(80.0)[0, : n - 1], intervals[1:n], rtol=1e-13)
        # Phi_{C,0} from the exact curve equals the IID failure probability from exact counts
        curve, filled = analysis.correlated_failure_curve(
            np.zeros((1, 0)), [], n, exact=counts.exact(0.0), holds_code=k
        )
        assert not filled.any()
        e = np.array([0.01, 0.05, 0.1, 0.3, 0.6])
        iid = analysis.erasure_failure_probability(
            code.first_failure_sizes(5, seed=0), n, e, exact_counts=code.uncorrectable_counts(n)
        )
        np.testing.assert_allclose(
            analysis.failure_probability_from_curve(curve[0], n, e), iid, rtol=1e-12, atol=1e-15
        )
        # the samplers at J = 0 (uniform) and J = 6 (nearly intervals)
        exact = np.stack([counts.exact(0.0)[0], intervals[1:]])
        bias = np.array(
            [0.0]
            + [1.0 - erasures.ring_boundary_distribution(n, m, 6.0)[1][0] for m in range(1, n)]
        )
        for method, ladder in (("exact", "auto"), ("chain", [1.0, 2.0, 4.0])):
            result = code.correlated_failure(
                ring,
                [0.0, 6.0],
                sizes,
                samples,
                burn_in=_size(200, 50),
                seed=n + k,
                method=method,
                ladder=ladder,
            )
            assert result.method == method
            if method == "chain":
                assert result.rungs.tolist() == [5] * len(result.sizes)
            _assert_brickwork_estimate(code, ring, result, exact, bias, method, ladder)


def _assert_brickwork_estimate(code, ring, result, target, bias, method, ladder):
    """Row 0 against the uniform P_0(m); row 1 (J = 6) against the interval limit, which
    differs from P_6(m) by at most Q_6(b > 2) = bias[m] (the weight of non-intervals)."""
    n = code.num_qubits
    for j, m in enumerate(result.sizes.tolist()):
        drawn = erasures.sample_regions(
            ring,
            n,
            m,
            [0.0, 6.0],
            result.samples,
            burn_in=result.burn_in,
            seed=[result.seed, m],
            method=method,
            ladder=ladder,
        )
        for i in range(2):
            fails = code.region_failures(drawn.regions[i])
            assert fails.sum() == result.failures[i, j]
            ess = (
                float(result.samples)
                if method == "exact"
                else erasures.chain_diagnostics(fails.astype(float))["ess"]
            )
            p = target[i, m - 1]
            allowance = bias[m] if i == 1 else 0.0
            se = math.sqrt(p * (1 - p) / ess)
            gap = abs(result.failure[i, j] - p)
            assert gap <= 4 * se + 2 / ess + allowance, (method, i, m, result.failure[i, j], p)


def test_the_brickwork_codes_are_not_trivial():
    """The baseline sees k = 0, codes of distance 2 or more, and curves that move between
    0 and 1 over several sizes, with the interval limit apart from the uniform curve."""
    seen = set()
    for size, p in BRICKWORK:
        for code in _brickwork(size, p):
            n = code.num_qubits
            if code.num_logical == 0:
                seen.add("k = 0")
                continue
            if code.uncorrectable_counts(1)[0] == 0:
                seen.add("distance >= 2")
            intervals = code.interval_failure_by_size()
            uniform = analysis.exact_failure_by_size(code.uncorrectable_counts(n), n)
            if np.count_nonzero((intervals > 0) & (intervals < 1)) >= 3:
                seen.add("partial over several sizes")
            if np.max(np.abs(intervals[1:] - uniform)) > 0.1:
                seen.add("intervals differ from uniform")
    assert seen == {
        "k = 0",
        "distance >= 2",
        "partial over several sizes",
        "intervals differ from uniform",
    }


# ---------------------------------------------------------------------------------------
# 6. Curves, Phi and the tolerance on hand-made inputs
# ---------------------------------------------------------------------------------------


def _binomial_tail_below(n, e, m) -> float:
    """Pr[Bin(n, e) < m] in exact rational arithmetic."""
    ef = Fraction(e)
    return float(sum(math.comb(n, j) * ef**j * (1 - ef) ** (n - j) for j in range(m)))


def _binomial_tail_above(n, e, m) -> float:
    ef = Fraction(e)
    return float(sum(math.comb(n, j) * ef**j * (1 - ef) ** (n - j) for j in range(m + 1, n + 1)))


@pytest.mark.parametrize(("n", "e", "tol"), [(100, 0.1, 1e-12), (60, 0.5, 1e-6), (40, 0.02, 1e-3)])
def test_binomial_window_cuts_tails_of_at_most_half_the_tolerance(n, e, tol):
    lo, hi = analysis.binomial_window(n, e, tol)
    assert 0 <= lo <= hi <= n
    assert _binomial_tail_below(n, e, lo) <= tol / 2 < _binomial_tail_below(n, e, lo + 1)
    assert _binomial_tail_above(n, e, hi) <= tol / 2 < _binomial_tail_above(n, e, hi - 1)


@pytest.mark.parametrize("m0", [2, 5])
def test_binomial_window_splits_the_tolerance_between_the_tails(m0):
    """With tol = 1.5 Pr[M <= m0], Pr[M <= m0] lies in (tol/2, tol]: m_lo must be m0 (a
    window that gave each tail the whole tol would start at m0 + 1)."""
    n, e = 100, 0.1
    tol = 1.5 * (1.0 - _binomial_tail_above(n, e, m0))
    lo, hi = analysis.binomial_window(n, e, tol)
    assert lo == m0
    assert _binomial_tail_below(n, e, lo) <= tol / 2 < _binomial_tail_below(n, e, lo + 1)
    assert _binomial_tail_above(n, e, hi) <= tol / 2 < _binomial_tail_above(n, e, hi - 1)
    # the same on the upper tail, by symmetry of Bin(n, e) and Bin(n, 1 - e)
    lo_up, hi_up = analysis.binomial_window(n, 1 - e, tol)
    assert hi_up == n - m0


def test_binomial_window_edge_cases():
    assert analysis.binomial_window(50, 0.0) == (0, 0)
    assert analysis.binomial_window(50, 1.0) == (50, 50)
    assert analysis.binomial_window(50, 0.3, tol=0.0) == (0, 50)
    a = analysis.binomial_window(200, 0.05)
    b = analysis.binomial_window(200, 0.4)
    assert analysis.binomial_window(200, [0.05, 0.4]) == (a[0], b[1])
    big = analysis.binomial_window(100_000, 0.25)
    assert big[0] < 25_000 < big[1] and big[1] - big[0] < 2_000
    assert all(isinstance(x, int) for x in big)


def test_correlated_failure_curve_fills_exact_sampled_and_interpolated_sizes():
    n = 10
    sampled = np.array([[0.5, 0.2, 0.9], [0.4, 0.1, 0.8]])  # sizes 5, 2, 8; two couplings
    exact = np.array([[0.0, 0.05], [0.0, 0.01]])  # m = 1, 2 (overrides the sampled m = 2)
    curve, filled = analysis.correlated_failure_curve(sampled, [5, 2, 8], n, exact=exact)
    assert curve.shape == (2, 11) and filled.shape == (11,)
    assert filled.tolist() == [
        False,
        False,
        False,
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        False,
    ]
    # known: 0 (m = 0), 0 and 0.05 (exact m = 1, 2), 0.5 (m = 5), 0.9 (m = 8), 1 (m = n)
    np.testing.assert_allclose(
        curve[0], [0, 0, 0.05, 0.2, 0.35, 0.5, 0.5 + 0.4 / 3, 0.5 + 0.8 / 3, 0.9, 0.95, 1.0]
    )
    assert curve[1, 2] == 0.01 and curve[1, 5] == 0.4 and curve[1, 8] == 0.8
    # holds_code sets P(n); a code dimension works as a flag; leading axes broadcast
    stacked = np.stack([sampled, sampled])  # (2 codes, 2 couplings, 3 sizes)
    both, _ = analysis.correlated_failure_curve(stacked, [5, 2, 8], n, holds_code=[3, 0])
    assert both.shape == (2, 2, 11)
    assert both[0, :, n].tolist() == [1.0, 1.0] and both[1, :, n].tolist() == [0.0, 0.0]
    assert both[1, 0, 9] == pytest.approx(0.45)  # halfway from 0.9 (m = 8) to 0 (m = 10)
    # sizes 0 and n are fixed by the endpoints; NaN propagates
    ends, filled = analysis.correlated_failure_curve([[0.3, 0.7]], [0, 10], n)
    assert ends[0, 0] == 0.0 and ends[0, n] == 1.0 and filled[1:n].all()
    holes, _ = analysis.correlated_failure_curve([[np.nan]], [5], n)
    assert np.isnan(holes[0, 4]) and np.isnan(holes[0, 5]) and holes[0, 0] == 0.0


def test_failure_probability_from_curve_is_the_binomial_mixture():
    n = 6
    curve = np.array([0.0, 0.0, 0.1, 0.3, 0.6, 0.9, 1.0])
    for e in (0.0, 0.07, 0.5, 1.0):
        expected = sum(math.comb(n, m) * e**m * (1 - e) ** (n - m) * curve[m] for m in range(7))
        assert analysis.failure_probability_from_curve(curve, n, e) == pytest.approx(expected)
    many = analysis.failure_probability_from_curve(np.stack([curve, curve / 2]), n, [0.1, 0.2])
    assert many.shape == (2, 2)
    np.testing.assert_allclose(many[1], many[0] / 2)
    # J = 0 with the uniform exact curve is erasure_failure_probability with exact counts
    code = CODES["blocks_k3"]()
    exact = code.boundary_failure_counts(graphs.cycle(10), 10).exact(0.0)
    full, _ = analysis.correlated_failure_curve(np.zeros((1, 0)), [], 10, exact=exact)
    e = np.linspace(0, 1, 11)
    iid = analysis.erasure_failure_probability(
        code.first_failure_sizes(3, seed=1), 10, e, exact_counts=code.uncorrectable_counts(10)
    )
    np.testing.assert_allclose(
        analysis.failure_probability_from_curve(full[0], 10, e), iid, rtol=1e-12, atol=1e-15
    )


def test_tolerance_from_a_monotone_curve_inverts_phi():
    n = 12
    curve = np.concatenate([[0.0], np.linspace(0.01, 1.0, n)])
    for delta in (0.001, 0.1, 0.5):
        eps, monotone = analysis.tolerance_from_curve(curve, n, delta, return_monotone=True)
        assert monotone is True and isinstance(eps, float)
        assert analysis.failure_probability_from_curve(curve, n, eps) == pytest.approx(
            delta, abs=1e-12
        )
    many = analysis.tolerance_from_curve(np.stack([curve, curve * 0.5]), n, 0.2)
    assert many.shape == (2,) and many[1] > many[0]
    assert analysis.tolerance_from_curve(curve * 0.5, n, 0.6) == 1.0  # Phi(1) = 0.5 <= 0.6
    assert analysis.tolerance_from_curve(curve, n, 0.0) == 0.0
    assert analysis.tolerance_from_curve(np.zeros(n + 1), n, 0.0) == 1.0
    nan = analysis.tolerance_from_curve(np.stack([curve, np.full(n + 1, np.nan)]), n, 0.1)
    assert np.isfinite(nan[0]) and np.isnan(nan[1])


def test_tolerance_from_a_non_monotone_curve_is_the_last_crossing():
    n = 4
    curve = np.array([0.0, 1.0, 0.0, 0.0, 1.0])  # Phi rises, falls, rises again
    delta = 0.35
    eps, monotone = analysis.tolerance_from_curve(curve, n, delta, return_monotone=True)
    assert monotone is False
    phi = analysis.failure_probability_from_curve(curve, n, np.linspace(0, 1, 4001))
    grid = np.linspace(0, 1, 4001)
    assert phi[(grid > 0.1) & (grid < 0.3)].max() > delta  # the first excursion above delta
    assert analysis.failure_probability_from_curve(curve, n, eps) == pytest.approx(delta, abs=1e-12)
    assert eps > 0.6 and np.all(phi[grid > eps + 1e-9] > delta)
    flags = analysis.tolerance_from_curve(
        np.stack([curve, np.linspace(0, 1, 5)]), n, delta, return_monotone=True
    )[1]
    assert flags.tolist() == [False, True]


def test_mean_over_codes_drops_codes_without_logical_qubits():
    values = np.array([[0.2, 0.4], [9.0, 9.0], [0.4, 0.8], [0.6, np.nan]])
    k = np.array([2, 0, 1, 3])
    mean, error, count = analysis.mean_over_codes(values, k)
    assert count == 3
    np.testing.assert_allclose(mean, [0.4, 0.6])
    np.testing.assert_allclose(error, [np.std([0.2, 0.4, 0.6], ddof=1) / math.sqrt(3), 0.2 / 1])
    same = analysis.mean_over_codes(values, k > 0)
    np.testing.assert_array_equal(same[0], mean)
    none = analysis.mean_over_codes(values, np.zeros(4, dtype=bool))
    assert none[2] == 0 and np.all(np.isnan(none[0]))


# ---------------------------------------------------------------------------------------
# 7. Validation, kernels' edge cases
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sizes": [0, 11]}, r"sizes must lie in \[0, 10\]"),
        ({"sizes": [-1]}, r"sizes must lie in \[0, 10\]"),
        ({"sizes": [2, 2]}, "sizes must be distinct"),
        ({"sizes": [1.5]}, "sizes must be a list of region sizes"),
        ({"sizes": [[1, 2]]}, "sizes must be a list of region sizes"),
        ({"sizes": []}, "give at least one region size"),
        ({"couplings": [0.0, math.inf]}, "couplings must be finite"),
        ({"couplings": [1.0, 1.0]}, "couplings must be distinct"),
        ({"couplings": "J"}, "couplings must be numbers"),
        ({"samples": 0}, "samples must be at least 1"),
        ({"seed": -3}, "seed must be a non-negative integer"),
        ({"method": "magic"}, "method must be 'auto', 'chain' or 'exact'"),
        ({"method": "exact", "edges": graphs.complete(10)}, "method='exact' needs a graph"),
        ({"edges": [(0, 10)]}, "edge endpoint 10 out of range for 10 nodes"),
    ],
)
def test_correlated_failure_validation(kwargs, message):
    code = CODES["blocks_k3"]()
    args = {"edges": graphs.cycle(10), "couplings": [0.0, 1.0], "sizes": [2], "samples": 5}
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        code.correlated_failure(**args)


def test_other_validation_messages():
    code = CODES["blocks_k3"]()
    ring = graphs.cycle(10)
    with pytest.raises(ValueError, match=r"max_size must be in \[1, 10\]"):
        code.boundary_failure_counts(ring, 0)
    with pytest.raises(ValueError, match=r"max_size must be in \[1, 10\]"):
        code.boundary_failure_counts(ring, 11)
    with pytest.raises(ValueError, match="max_size must be an integer"):
        code.boundary_failure_counts(ring, 2.5)
    with pytest.raises(ValueError, match=r"C\(10, 1\) \+ \.\.\. \+ C\(10, 4\) = 385 subsets"):
        code.boundary_failure_counts(ring, 4, budget=100)
    with pytest.raises(ValueError, match=r"exhaustive_cutoff\(10, 100\) = 2"):
        code.boundary_failure_counts(ring, 4, budget=100)
    assert code.boundary_failure_counts(ring, 4, budget=385).total.sum() == 385
    with pytest.raises(ValueError, match="budget must be at least 1"):
        code.boundary_failure_counts(ring, 2, budget=0)
    huge = StabilizerCode(np.zeros((0, 2 * 200), dtype=np.uint8), 200)
    with pytest.raises(ValueError, match="more than 10\\^30"):
        huge.boundary_failure_counts(graphs.cycle(200), 100)
    for bad, message in [
        ([[0, 1], [2]], "2-D integer array"),
        ([0, 1], "2-D integer array"),
        ([[0.0, 1.0]], "2-D integer array"),
        ([[0, 10]], r"in \[0, 10\)"),
        ([[-1, 2]], r"in \[0, 10\)"),
        ([[3, 3]], "distinct"),
    ]:
        with pytest.raises(ValueError, match=message):
            code.region_failures(bad)
    for bad in ([0, 1, 2], list(range(9)) + [9.5], [0] * 10, np.arange(10).reshape(2, 5)):
        with pytest.raises(ValueError, match=r"order must be a permutation of 0\.\.9"):
            code.window_first_failure(bad)
        with pytest.raises(ValueError, match="order must be a permutation"):
            code.end_window_failure_by_size(bad)
    one = StabilizerCode(["X"])
    with pytest.raises(ValueError, match="a code on one qubit"):
        one.correlated_failure(graphs.from_edges([], 1), [0.0])
    assert one.correlated_failure(graphs.from_edges([], 1), [0.0], [0, 1], 3).failure.tolist() == [
        [0.0, 0.0]
    ]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: analysis.exact_correlated_failure([1, 2], [[1, 2]], 0.0), r"failed needs shape"),
        (lambda: analysis.exact_correlated_failure([[1]], [[0, 1], [1, 1]], 0.0), "same sizes"),
        (lambda: analysis.exact_correlated_failure([[2, 0]], [[1, 1]], 0.0), "cannot exceed"),
        (lambda: analysis.exact_correlated_failure([[0.5]], [[1]], 0.0), "whole numbers"),
        (lambda: analysis.exact_correlated_failure([[-1]], [[1]], 0.0), "whole numbers"),
        (lambda: analysis.exact_correlated_failure([[0, 0]], [[0, 0]], 0.0), "no region at all"),
        (lambda: analysis.exact_correlated_failure([["a"]], [[1]], 0.0), "array of counts"),
        (
            lambda: analysis.exact_correlated_failure(np.zeros((2, 1, 1)), np.ones((3, 1, 1)), 0),
            "do not broadcast",
        ),
        (lambda: analysis.exact_correlated_failure([[0]], [[1]], [math.nan]), "finite number"),
        (lambda: analysis.exact_correlated_failure([[0]], [[1]], "x"), "couplings must be numbers"),
        (lambda: analysis.exact_correlated_failure([[0]], [[1]], []), "finite number"),
        (lambda: analysis.binomial_window(10, 1.5), "probabilities in"),
        (lambda: analysis.binomial_window(10, "x"), "probabilities in"),
        (lambda: analysis.binomial_window(10, 0.5, tol=1.0), r"tol must be a number in \[0, 1\)"),
        (lambda: analysis.binomial_window(10, 0.5, tol="x"), "tol must be"),
        (lambda: analysis.correlated_failure_curve([0.5], [3], 10), "sampled needs shape"),
        (lambda: analysis.correlated_failure_curve([[0.5]], [3, 4], 10), "sizes must list 1"),
        (lambda: analysis.correlated_failure_curve([[0.5]], [11], 10), "distinct integers"),
        (lambda: analysis.correlated_failure_curve([[0.5]], [2.5], 10), "distinct integers"),
        (lambda: analysis.correlated_failure_curve([[0.5, 0.2]], [3, 3], 10), "distinct"),
        (lambda: analysis.correlated_failure_curve([[0.5]], ["a"], 10), "distinct integers"),
        (lambda: analysis.correlated_failure_curve([[1.5]], [3], 10), r"in \[0, 1\]"),
        (lambda: analysis.correlated_failure_curve([["a"]], [3], 10), "array of probabilities"),
        (lambda: analysis.correlated_failure_curve([[0.5]], [3], 0), "at least 1"),
        (
            lambda: analysis.correlated_failure_curve([[0.5]], [3], 10, exact=[0.1]),
            "exact needs shape",
        ),
        (
            lambda: analysis.correlated_failure_curve([[0.5]], [1], 2, exact=[[0.1, 0.2, 0.3]]),
            "exact needs shape",
        ),
        (
            lambda: analysis.correlated_failure_curve([[0.5]] * 2, [3], 10, exact=[[0.1]] * 3),
            "do not broadcast",
        ),
        (
            lambda: analysis.correlated_failure_curve(
                np.zeros((2, 1, 1)), [3], 10, holds_code=[1, 1, 1]
            ),
            "holds_code needs the shape",
        ),
        (lambda: analysis.failure_probability_from_curve([0, 1], 2, 0.5), "n \\+ 1 = 3"),
        (lambda: analysis.failure_probability_from_curve([0, 0.5, 1], 2, -0.1), "probabilities"),
        (lambda: analysis.failure_probability_from_curve([0, 0.5, 1], 2, [[0.1]]), "1-D"),
        (lambda: analysis.tolerance_from_curve([0, 0.5, 1], 2, 1.5), r"delta must be in"),
        (lambda: analysis.tolerance_from_curve([0, 0.5, 1], 2, "x"), r"delta must be in"),
        (lambda: analysis.tolerance_from_curve([0.1, 0.5, 1], 2, 0.2), r"P\(0\) must be 0"),
        (lambda: analysis.mean_over_codes([[1.0], [2.0]], [True]), "one entry per row"),
    ],
)
def test_analysis_validation(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_kernels_leave_zero_counts_for_nothing_to_count():
    code = CODES["blocks_k3"]()
    failed = np.full((1, 3), 7, dtype=np.int64)
    total = np.full((1, 3), 7, dtype=np.int64)
    empty = np.zeros(1, dtype=np.int64)
    kernel.boundary_failure_counts(
        code._tab, code._ws, code._marked, empty, empty[:0], 0, 1, failed, total
    )
    assert not failed.any() and not total.any()
    indptr, indices = np.zeros(11, dtype=np.int64), np.zeros(0, dtype=np.int64)
    kernel.boundary_failure_counts(
        code._tab, code._ws, code._marked, indptr, indices, 10, 0, failed, total
    )
    assert not failed.any() and not total.any()
    out = np.full(2, 9, dtype=np.uint8)
    scratch = np.zeros((1, code._ws), dtype=np.uint64)
    nothing = np.zeros((2, 0), dtype=np.int64)
    kernel.region_failures(code._tab, code._ws, code._marked, 10, 3, nothing, scratch, out)
    assert out.tolist() == [0, 0]
    everything = np.tile(np.arange(10), (2, 1))  # the complement is empty: l = 2k > 0
    kernel.region_failures(code._tab, code._ws, code._marked, 10, 3, everything, scratch, out)
    assert out.tolist() == [1, 1]


def test_a_code_without_stabilizers_fails_on_every_nonempty_region():
    """S = {I} (the purification picture before any measurement): l(A) = 2|A| > 0."""
    code = StabilizerCode(np.zeros((0, 10), dtype=np.uint8), 5)
    assert (code.num_logical, code.num_stabilizers) == (5, 0)
    counts = code.boundary_failure_counts(graphs.cycle(5), 5)
    np.testing.assert_array_equal(counts.failed, counts.total)
    np.testing.assert_array_equal(code.window_first_failure(), np.ones(5))
    np.testing.assert_array_equal(code.window_first_failure(periodic=False), np.ones(5))
    np.testing.assert_array_equal(code.interval_failure_by_size(), [0, 1, 1, 1, 1, 1])
    assert code.region_failures([[0, 1, 2], [4, 3, 1]]).tolist() == [True, True]
    assert code.region_failures([[2]]).tolist() == [True]
    np.testing.assert_array_equal(counts.exact([-2.0, 0.0, 3.0]), np.ones((3, 5)))


def test_windows_of_a_code_without_logical_qubits():
    code = CODES["pure_k0"]()
    n = code.num_qubits
    np.testing.assert_array_equal(code.window_first_failure(), np.full(n, n + 1))
    np.testing.assert_array_equal(code.window_first_failure(periodic=False), n + 1 - np.arange(n))
    assert not code.interval_failure_by_size().any()
    assert not code.interval_failure_by_size(periodic=False).any()
    assert not code.end_window_failure_by_size().any()
