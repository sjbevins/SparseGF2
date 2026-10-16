"""The nearest-neighbour brickwork baseline of the geometry-correlated erasures, checked exactly.

The baseline circuit (the literature's monitored Clifford brickwork): the periodic ring
``graphs.cycle(n)`` with ``schedule="matching"`` (one matching per layer, even bonds then odd
bonds), uniform random two-qubit Cliffords, each system qubit measured in Z with probability
``p`` after every layer, ``4n`` layers, purification picture (``k = S(R)`` logical qubits).
On its codes, at ``n = 12`` to ``24`` (every region enumerable at ``n <= 16``):

1. **The J -> inf claim.** On the ring every minimum-boundary region of every size
   ``1 <= m <= n-1`` is one of the ``n`` intervals (``b = 2``), so ``lim P_{C,C_n,J}(m)`` is the
   failing fraction of the intervals of length ``m`` (``interval_failure_by_size``): checked
   as an integer identity on the boundary-resolved counts and at ``J = 30`` to 1e-9, and
   through the exact ring sampler at ``J = 12``. On the open path the limit is the two end
   windows (``end_window_failure_by_size``), likewise exactly and through the exact path
   sampler (``geometry="path"``). On a Watts-Strogatz graph (beta = 0.3) a
   minimum-boundary region need not be an interval of the qubit order: the claim is the ring's.
2. **J = 0** is the uniform law: ``P_{C,G,0}(m) = P_C(m)`` bit for bit, and ``Phi_{C,G,0} =
   Phi_C`` (the independent-erasure channel) and its tolerance.
3. **The samplers** (the in-run probe and ``StabilizerCode.correlated_failure``) agree with the
   exact ``P_{C,G,J}(m)`` within 4 standard errors for J in {0, 1, 3}: on the ring (exact
   sampler; binomial errors are exact) and on a Watts-Strogatz interaction graph (swap chain
   with replica exchange; errors from the spread over independent trajectories).
4. **Per-trajectory identities.** ``I(A:R)`` from the saved code equals the mutual information
   of the live simulator state (an integer number of bits); ``I(A:R) + I(Abar:R) = 2k``
   (the cleaning lemma, Li and Fisher's Theorem 1); the contiguous Singleton bound
   ``k <= n - 2 (d_cont - 1)``; ``d <= d_cont``.
5. **Permutation invariance.** For a code ensemble invariant under relabeling the qubits, the
   mean of ``P_{C,G,J}(m)`` does not depend on J (``E[F_C(A)]`` depends on ``|A|`` only), on
   any graph: checked on the ring geometry for codes of an all-to-all circuit, exactly per
   code and end to end through the probe. Brickwork codes are local and do see the
   geometry: at J = 3 the mean failure is higher than at J = 0 at intermediate m (clustered
   erasures hurt a local code; asserted only where the margin is above 5 standard errors).
6. **Random codes** (Gullans, Krastanov, Huse, Jiang and Flammia, PRX 11, 031066 (2021)):
   uniformly random ``[[64, 32]]`` stabilizer codes against their random-matrix formula,
   Eq. (23). Their failure probability is ``P(F) = 1 - E[2^{-I(A:R)}]`` (one minus the optimal
   recovery probability of Eq. (18), ``2^{-r_M}`` with ``r_M = I(A:R)``), not our
   ``F_C(A) = 1[I(A:R) > 0]``: at the critical size ``m = 16`` their 0.38968 against
   ``P[I(A:R) > 0] = 0.71121`` from the same rank distribution. Both are checked, on exactly
   uniform codes (an independent sampler of isotropic subspaces) and on codes from the
   simulator's own all-to-all random circuits.

Seeds are fixed. Exact identities are asserted exactly (or to 1e-9 where a limit is
approached at finite J); statistical checks use 4 standard errors, from exact binomial
variances where the samples are independent and from the spread over independent
trajectories or codes otherwise. ``benchmarks/correlated_erasure_baseline.py`` runs the same
baseline at n = 32 to 256 against the literature.
"""

from __future__ import annotations

import functools
import math

import numba
import numpy as np
import pytest

from sparsegf2 import Circuit, Simulator, analysis, erasures, graphs, mipt, observables
from sparsegf2.codes import StabilizerCode

JIT_FREE = bool(numba.config.DISABLE_JIT)


def _size(compiled, plain):
    """JIT-free (plain-Python kernels, several hundred times slower) runs use less work."""
    return plain if JIT_FREE else compiled


Z_MAX = 4.0  # statistical checks: |estimate - exact| below 4 standard errors
RATES = (0.05, 0.12, 0.3)  # mixed phase, near p_c ~ 0.16 (Gullans and Huse), purified
ENUMERATED = sorted({12, _size(16, 12)})  # every region of every size is listed
SMALL_RINGS = sorted({5, 12, _size(16, 12)})
IDENTITY_SIZES = sorted({12, 16, _size(20, 16), _size(24, 16)})


def _brickwork_spec(n: int, p: float, probes=()) -> mipt.MonitoredCircuit:
    return mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        layers=4 * n,
        measurement_rate=p,
        schedule="matching",
        reference="purification",
        checkpoints="final",
        probes=[mipt.code_dimension(), *probes, mipt.code_snapshot()],
    )


@functools.cache
def _brickwork(n: int, p: float, trajectories: int = 6, seed: int = 61):
    """``(spec, result, codes)``: brickwork codes at the final layer (k = 0 codes included)."""
    spec = _brickwork_spec(n, p)
    res = spec.run(trajectories, seed=seed)
    return spec, res, tuple(res.code(t, 0) for t in range(trajectories))


def _intervals(n: int, m: int) -> set[tuple[int, ...]]:
    return {tuple(sorted((s + j) % n for j in range(m))) for s in range(n)}


# ---------------------------------------------------------------------------
# 0. The baseline circuit is the brickwork
# ---------------------------------------------------------------------------


def test_matching_schedule_on_the_even_ring_is_the_brickwork():
    """Layer 1 acts on the bonds (0,1), (2,3), ..., layer 2 on (1,2), ..., (n-1,0), and so on;
    measurements follow each gate layer."""
    n = 12
    text = str(_brickwork_spec(n, 0.2).circuit(0, seed=3))
    blocks = text.split("TICK")[1:-1]  # the setup comes before the first TICK
    assert len(blocks) == 4 * n
    even = {(i, i + 1) for i in range(0, n, 2)}
    odd = {tuple(sorted((i, (i + 1) % n))) for i in range(1, n, 2)}
    for layer, block in enumerate(blocks, start=1):
        lines = block.strip().splitlines()
        gates = [ln for ln in lines if ln.startswith("CLIFFORD_2Q(")]
        assert len(gates) == 1 and lines[0] == gates[0]  # one gate layer, then measurements
        qubits = [int(q) for q in gates[0].partition(")")[2].split()]
        pairs = {tuple(sorted(qubits[i : i + 2])) for i in range(0, len(qubits), 2)}
        assert pairs == (even if layer % 2 else odd)
        assert all(ln.startswith("M ") for ln in lines[1:])


# ---------------------------------------------------------------------------
# 1. J -> inf: intervals on the ring, the two end windows on the open path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", SMALL_RINGS)
def test_ring_minimum_boundary_regions_are_exactly_the_intervals(n):
    for m in range(1, n):
        b_min, regions = erasures.minimum_boundary_regions(graphs.cycle(n), n, m)
        assert b_min == 2
        assert {tuple(r) for r in regions.tolist()} == _intervals(n, m)
        assert len(regions) == n == erasures.ring_run_count(n, m, 1)


@pytest.mark.parametrize("n", SMALL_RINGS)
def test_path_minimum_boundary_regions_are_exactly_the_end_windows(n):
    for m in range(1, n):
        b_min, regions = erasures.minimum_boundary_regions(graphs.path(n), n, m)
        assert b_min == 1
        assert {tuple(r) for r in regions.tolist()} == {
            tuple(range(m)),
            tuple(range(n - m, n)),
        }


def test_watts_strogatz_minimizers_need_not_be_intervals():
    """With shortcuts (beta = 0.3) a region of minimum boundary can be spread along the qubit
    order: the ring's J -> inf limit (the intervals) does not carry over."""
    n = 16
    ws = graphs.watts_strogatz(n, 4, 0.3, seed=3)
    b_min, regions = erasures.minimum_boundary_regions(ws, n, 4)
    rows = {tuple(r) for r in regions.tolist()}
    assert (3, 5, 7, 9) in rows and (3, 5, 7, 9) not in _intervals(n, 4)
    assert erasures.boundary(ws, n, [3, 5, 7, 9]) == b_min
    # and the J -> inf limit there is the failing fraction of those minimizers, whatever they are
    spec = mipt.MonitoredCircuit(
        num_system=n, edges=ws, layers=4 * n, measurement_rate=0.1, schedule="random_edges",
        reference="purification", probes=[mipt.code_snapshot()],
    )  # fmt: skip
    code = spec.run(1, seed=5).code(0, 0)
    assert code.num_logical > 0
    limit = code.boundary_failure_counts(ws, 4).exact(30.0)[0, 3]
    assert limit == pytest.approx(code.region_failures(regions).mean(), abs=1e-9)


@pytest.mark.parametrize("p", RATES)
@pytest.mark.parametrize("n", ENUMERATED)
def test_ring_limit_is_the_interval_failure_curve(n, p):
    _, _, codes = _brickwork(n, p)
    ring = graphs.cycle(n)
    for code in codes:
        counts = code.boundary_failure_counts(ring, n)  # every region, m = 1..n
        intervals = code.interval_failure_by_size()  # length n + 1
        # the regions of boundary 2 are the n intervals; their failing ones, exactly
        assert np.all(counts.total[: n - 1, 2] == n)
        assert np.all(counts.total[: n - 1, :2] == 0)
        np.testing.assert_array_equal(
            counts.failed[: n - 1, 2], np.rint(n * intervals[1:n]).astype(np.int64)
        )
        np.testing.assert_allclose(counts.exact(30.0)[0], intervals[1:], rtol=0, atol=1e-9)
        if code.num_logical == 0:
            assert not intervals.any()


@pytest.mark.parametrize("p", RATES)
@pytest.mark.parametrize("n", ENUMERATED)
def test_open_path_limit_is_the_two_end_windows(n, p):
    _, _, codes = _brickwork(n, p)
    path = graphs.path(n)
    differs = []
    for code in codes:
        counts = code.boundary_failure_counts(path, n)
        ends = code.end_window_failure_by_size()
        assert np.all(counts.total[: n - 1, 1] == 2) and np.all(counts.total[: n - 1, 0] == 0)
        np.testing.assert_array_equal(
            counts.failed[: n - 1, 1], np.rint(2 * ends[1:n]).astype(np.int64)
        )
        np.testing.assert_allclose(counts.exact(30.0)[0], ends[1:], rtol=0, atol=1e-9)
        if code.num_logical:
            open_windows = code.interval_failure_by_size(periodic=False)
            differs.append(not np.array_equal(ends, open_windows))
    # the open path's limit is not the uniform law on all intervals of the open order
    assert any(differs) or not differs


# ---------------------------------------------------------------------------
# 2. J = 0 is the uniform law; Phi_{C,0} is the independent-erasure channel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", RATES)
@pytest.mark.parametrize("n", ENUMERATED)
def test_uniform_limit_is_the_uniform_failure_probability(n, p):
    _, _, codes = _brickwork(n, p)
    es = np.array([0.02, 0.1, 0.25, 0.5])
    for i, code in enumerate(codes):
        unc = code.uncorrectable_counts(n)
        uniform = analysis.exact_failure_by_size(unc, n)
        for edges in (graphs.cycle(n), graphs.path(n), graphs.watts_strogatz(n, 4, 0.3, seed=1)):
            counts = code.boundary_failure_counts(edges, n)
            np.testing.assert_array_equal(counts.failed.sum(axis=1), unc)
            np.testing.assert_array_equal(counts.exact(0.0)[0], uniform)  # bit for bit
        curve, filled = analysis.correlated_failure_curve(
            np.empty((1, 0)), [], n, exact=uniform[None, :], holds_code=code.num_logical
        )
        assert not filled.any()
        first = code.first_failure_sizes(16, seed=[2, i])
        phi_iid = analysis.erasure_failure_probability(first, n, es, exact_counts=unc)
        phi_curve = analysis.failure_probability_from_curve(curve[0], n, es)
        np.testing.assert_allclose(phi_curve, phi_iid, rtol=1e-12, atol=1e-15)
        if code.num_logical:
            e10 = analysis.erasure_tolerance(first, n, 0.1, exact_counts=unc)
            assert analysis.tolerance_from_curve(curve[0], n, 0.1) == pytest.approx(e10, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. The samplers against the exact P_{C,G,J}(m)
# ---------------------------------------------------------------------------

COUPLINGS = (0.0, 1.0, 3.0)


def _z(diff, se):
    """diff / se, where an error of 0 (every row exactly 0 or 1) admits no difference at all."""
    safe = np.where(se > 0, se, 1.0)
    return np.where(se > 0, diff / safe, np.where(diff == 0, 0.0, np.inf))


def _binomial_z(estimates, exact, samples):
    """Pooled z of independent binomial fractions: sum(est - P) / sqrt(sum P(1 - P) / S)."""
    var = (exact * (1 - exact) / samples).sum(axis=0)
    diff = (estimates - exact).sum(axis=0)
    return _z(diff, np.sqrt(var)), diff


def _spread_z(estimates, exact, samples):
    """z of the mean difference with its error from the spread over independent rows.

    The chain's records are correlated within a row but rows (trajectories, codes) are
    independent, so the spread is a valid error; the floor (half the binomial error of
    independent samples) keeps rows that all agree from dividing by zero.
    """
    d = estimates - exact
    t = d.shape[0]
    se = d.std(axis=0, ddof=1) / math.sqrt(t)
    floor = 0.5 * np.sqrt((exact * (1 - exact) / samples).sum(axis=0)) / t
    mean = d.mean(axis=0)
    return _z(mean, np.maximum(se, floor)), mean


@pytest.mark.parametrize("p", [0.05, 0.12])
def test_ring_samplers_match_the_exact_failure(p):
    """The exact ring sampler: the probe and correlated_failure have exact binomial errors.
    J = 12 is the J -> inf limit through the sampler (all but ~1e-5 of the draws are
    intervals): it matches the interval curve."""
    n, samples, sizes = 16, _size(200, 40), [2, 4, 6, 8]
    couplings = [*COUPLINGS, 12.0]
    probe = mipt.correlated_erasure(couplings, sizes=sizes, samples=samples)
    spec = _brickwork_spec(n, p, probes=[probe])
    t = _size(16, 4)
    res = spec.run(t, seed=17)
    assert spec.correlated_sampler("correlated_erasure")["method"] == "exact"
    _, _, values = res.correlated()
    ring = graphs.cycle(n)
    exact = np.zeros((t, len(couplings), len(sizes)))
    lib = np.zeros_like(exact)
    limit = np.zeros((t, len(sizes)))
    for row in range(t):
        code = res.code(row, 0)
        exact[row] = code.boundary_failure_counts(ring, max(sizes)).exact(couplings)[
            :, np.array(sizes) - 1
        ]
        found = code.correlated_failure(ring, couplings, sizes, samples, seed=[5, row])
        assert found.method == "exact" and np.all(found.failure_ess == samples)
        lib[row] = found.failure
        limit[row] = code.interval_failure_by_size()[sizes]
        if code.num_logical == 0:
            assert not values[row].any()
    np.testing.assert_allclose(exact[:, 3], limit, rtol=0, atol=1e-6)
    for estimates in (values[:, -1], lib):
        z, _ = _binomial_z(estimates, exact, samples)
        assert np.all(np.abs(z) < Z_MAX), z
        z_limit, _ = _binomial_z(estimates[:, 3], limit, samples)
        assert np.all(np.abs(z_limit) < Z_MAX), z_limit


def test_path_sampler_reaches_the_two_end_windows():
    """The exact path sampler (geometry="path" over the qubit labels of a brickwork circuit)
    against the exact P_J(m) at J = 0 and 12, and J = 12 against the J -> inf limit, the mean
    failure of the two end windows (each draw is one of them but for about 1e-4)."""
    n, samples, sizes = 16, _size(200, 40), [2, 4, 6, 8]
    couplings = [0.0, 12.0]
    probe = mipt.correlated_erasure(couplings, sizes=sizes, samples=samples, geometry="path")
    spec = _brickwork_spec(n, 0.05, probes=[probe])
    assert spec.correlated_sampler("correlated_erasure")["geometries"] == ["path"]
    t = _size(16, 4)
    res = spec.run(t, seed=23)
    _, _, values = res.correlated()
    exact = np.zeros((t, len(couplings), len(sizes)))
    limit = np.zeros((t, len(sizes)))
    for row in range(t):
        code = res.code(row, 0)
        counts = code.boundary_failure_counts(graphs.path(n), max(sizes))
        exact[row] = counts.exact(couplings)[:, np.array(sizes) - 1]
        limit[row] = code.end_window_failure_by_size()[sizes]
    np.testing.assert_allclose(exact[:, 1], limit, rtol=0, atol=1e-3)
    z, _ = _binomial_z(values[:, -1], exact, samples)
    assert np.all(np.abs(z) < Z_MAX), z
    z_limit, _ = _binomial_z(values[:, -1, 1], limit, samples)
    assert np.all(np.abs(z_limit) < Z_MAX), z_limit


def test_watts_strogatz_chain_matches_the_exact_failure():
    """The swap chain with replica exchange on the circuit's own Watts-Strogatz graph.

    Thinned 4 sweeps apart: with every sweep recorded, the J = 3 chain at m = 8 has an
    integrated autocorrelation time of about 20 records (the sampler warns), still unbiased
    but with a tenth of the effective samples."""
    n, samples, sizes, thin = 16, _size(200, 30), [3, 5, 8], 4
    ws = graphs.watts_strogatz(n, 4, 0.3, seed=3)
    probe = mipt.correlated_erasure(COUPLINGS, sizes=sizes, samples=samples, thin=thin)
    spec = mipt.MonitoredCircuit(
        num_system=n, edges=ws, layers=4 * n, measurement_rate=0.12, schedule="random_edges",
        reference="purification", probes=[mipt.code_dimension(), probe, mipt.code_snapshot()],
    )  # fmt: skip
    assert spec.correlated_sampler("correlated_erasure")["method"] == "chain"
    t = _size(24, 4)
    res = spec.run(t, seed=29)
    _, _, values = res.correlated()
    exact = np.zeros((t, len(COUPLINGS), len(sizes)))
    lib = np.zeros_like(exact)
    for row in range(t):
        code = res.code(row, 0)
        exact[row] = code.boundary_failure_counts(ws, max(sizes)).exact(COUPLINGS)[
            :, np.array(sizes) - 1
        ]
        found = code.correlated_failure(ws, COUPLINGS, sizes, samples, thin=thin, seed=[9, row])
        assert found.method == "chain" and not found.warnings, found.warnings
        lib[row] = found.failure
    assert (res.values["code_dimension"][:, -1] > 0).sum() >= t // 2
    for estimates in (values[:, -1], lib):
        z, _ = _spread_z(estimates, exact, samples)
        assert np.all(np.abs(z) < Z_MAX), z


# ---------------------------------------------------------------------------
# 4. Per-trajectory identities
# ---------------------------------------------------------------------------


def test_lost_information_is_the_mutual_information_of_the_live_state(rng):
    """The saved code's I(A:R) equals S(A) + S(R) - S(AR) of the state the circuit leaves."""
    n, seed = 12, 61
    spec, res, codes = _brickwork(n, 0.05, seed=seed)
    reference = range(n, 2 * n)
    for t in range(_size(3, 1)):
        sim = Simulator(2 * n, phases=False)
        sim.run(spec.circuit(t, seed=seed))
        assert observables.code_dimension(sim, reference) == codes[t].num_logical
        for _ in range(_size(40, 8)):
            m = int(rng.integers(1, n + 1))
            region = rng.choice(n, size=m, replace=False).tolist()
            live = observables.mutual_information(sim, region, reference)
            assert isinstance(live, int)
            assert codes[t].lost_information(region) == live


@pytest.mark.parametrize("p", [0.05, 0.12])
@pytest.mark.parametrize("n", IDENTITY_SIZES)
def test_cleaning_lemma_and_the_contiguous_singleton_bound(n, p, rng):
    _, _, codes = _brickwork(n, p)
    for code in codes:
        k = code.num_logical
        for _ in range(_size(20, 4)):
            region = rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False)
            rest = np.setdiff1d(np.arange(n), region)
            lost, kept = code.lost_information(region.tolist()), code.lost_information(rest)
            assert lost + kept == 2 * k  # I(A:R) + I(Abar:R) = 2k
            if region.size:
                assert code.region_failures([region.tolist()])[0] == (lost > 0)
        d_cont = code.contiguous_distance()
        if k == 0:
            assert d_cont == 0
            continue
        assert d_cont == code.window_first_failure().min()
        assert k <= n - 2 * (d_cont - 1)  # two disjoint intervals of d_cont - 1 decouple from R
        counts = code.uncorrectable_counts(d_cont)
        assert counts[-1] > 0  # the failing interval itself
        assert int(np.flatnonzero(counts)[0]) + 1 <= d_cont  # d <= d_cont


# ---------------------------------------------------------------------------
# 5. Permutation invariance (and the geometry dependence of local codes)
# ---------------------------------------------------------------------------


def _all_to_all_spec(n: int, probes=()) -> mipt.MonitoredCircuit:
    """Gates on uniformly random pairs of the complete graph: a code ensemble invariant under
    every relabeling of the qubits."""
    return mipt.MonitoredCircuit(
        num_system=n, edges=graphs.complete(n), layers=6, measurement_rate=0.12,
        schedule="random_edges", gates_per_layer=2 * n, reference="purification",
        probes=[mipt.code_dimension(), *probes, mipt.code_snapshot()],
    )  # fmt: skip


def _exact_differences(res, n, sizes, couplings=(0.0, 3.0)):
    """Per code, exact P_{C,ring,3}(m) - P_{C,ring,0}(m) at ``sizes``."""
    ring = graphs.cycle(n)
    rows = []
    for t in range(res.num_trajectories):
        exact = res.code(t, 0).boundary_failure_counts(ring, max(sizes)).exact(couplings)
        rows.append(exact[1, np.array(sizes) - 1] - exact[0, np.array(sizes) - 1])
    return np.array(rows)


def _mean_z(d):
    mean, se = d.mean(axis=0), d.std(axis=0, ddof=1) / math.sqrt(d.shape[0])
    return mean, _z(mean, se)


def test_permutation_invariant_codes_do_not_see_the_geometry():
    """E[P_{C,G,J}(m)] = E[P_{C,G,0}(m)] for every J and graph when the code ensemble is
    invariant under relabeling: exactly per code (boundary-resolved counts on the ring) and
    end to end through the probe (exact ring sampler on geometry="cycle")."""
    n, sizes = 12, [2, 3, 4, 5, 6]
    res = _all_to_all_spec(n).run(_size(300, 20), seed=3)
    _, z = _mean_z(_exact_differences(res, n, sizes))
    assert np.all(np.abs(z) < Z_MAX), z
    probe = mipt.correlated_erasure([0.0, 3.0], sizes=sizes, samples=32, geometry="cycle")
    res = _all_to_all_spec(n, probes=[probe]).run(_size(400, 20), seed=4)
    _, _, values = res.correlated()
    _, z = _mean_z(values[:, -1, 1] - values[:, -1, 0])
    assert np.all(np.abs(z) < Z_MAX), z


def test_brickwork_codes_see_the_geometry():
    """The same comparison on brickwork codes (local): clustered erasures (J = 3) fail more often
    than uniform ones at intermediate sizes. Measured (exact per code, 200 codes, n = 16):
    p = 0.05, m = 2..6: +0.042, +0.083, +0.078, +0.028, +0.004 (z = 7.8, 11.3, 10.4, 5.9, 1.8);
    p = 0.12, m = 2..7: +0.013, +0.050, +0.076, +0.081, +0.058, +0.028 (z up to 10.7). Asserted
    only where the margin is above 5 standard errors."""
    n, sizes = 16, [2, 3, 4, 5, 6, 7]
    for p in (0.05, 0.12):
        res = _brickwork_spec(n, p).run(_size(200, 12), seed=4)
        mean, z = _mean_z(_exact_differences(res, n, sizes))
        if not JIT_FREE:
            assert z.max() > 5 and mean[np.argmax(z)] > 0, (mean, z)


# ---------------------------------------------------------------------------
# 6. Random codes against Gullans, Krastanov, Huse, Jiang and Flammia (2021)
# ---------------------------------------------------------------------------


def _rank_count(rows: int, cols: int, r: int) -> int:
    """The number of rows x cols matrices over GF(2) of rank r."""
    num = den = 1
    for i in range(r):
        num *= (2**rows - 2**i) * (2**cols - 2**i)
        den *= 2**r - 2**i
    return num // den


def _rmt_lost_distribution(m: int, ns: int) -> dict[int, float]:
    """The law of I(A:R) for |A| = m in the random-matrix ansatz of Gullans et al.: the
    2m x ns syndrome matrix M_S is uniform and I(A:R) = 2m - rank(M_S) (their r_M, with M of
    full rank 2m)."""
    total = 2 ** (2 * m * ns)
    return {2 * m - r: _rank_count(2 * m, ns, r) / total for r in range(min(2 * m, ns) + 1)}


def _gullans_failure(m: int, ns: int) -> float:
    """Their Eq. (23): P(F) = 1 - sum_r P(rank M_S = r) 2^(r - 2m) = 1 - E[2^-I(A:R)]."""
    return 1.0 - sum(prob * 2.0**-lost for lost, prob in _rmt_lost_distribution(m, ns).items())


def _rmt_any_loss(m: int, ns: int) -> float:
    """P[I(A:R) > 0] in the same ansatz: our failure F_C(A) = 1[I(A:R) > 0]."""
    return 1.0 - _rmt_lost_distribution(m, ns).get(0, 0.0)


def _symplectic(u: int, v: int, n: int) -> int:
    low = (1 << n) - 1
    return bin(((u & low) & (v >> n)) ^ ((u >> n) & (v & low))).count("1") & 1


def _uniform_code(n: int, s: int, rng: np.random.Generator) -> StabilizerCode:
    """A uniformly random stabilizer code with s generators on n qubits (no simulator involved).

    Draws v_i uniformly from span(v_1..v_{i-1})^perp minus span(v_1..v_{i-1}) (a uniform
    combination of a basis of the symplectic complement, kept up to date, until independent):
    every isotropic subspace of dimension s then arises from the same number of equally likely
    sequences. Bit q of a vector is X on qubit q, bit n + q is Z on qubit q.
    """
    complement = [1 << i for i in range(2 * n)]
    chosen: list[int] = []
    echelon: list[int] = []
    while len(chosen) < s:
        v = 0
        for bit, b in zip(
            rng.integers(0, 2, size=len(complement)).tolist(), complement, strict=True
        ):
            if bit:
                v ^= b
        w = v
        for b in echelon:
            w = min(w, w ^ b)
        if not w:
            continue  # in the span already
        echelon = sorted([*echelon, w], reverse=True)
        chosen.append(v)
        pivot = next(b for b in complement if _symplectic(b, v, n))
        complement = [b ^ pivot if _symplectic(b, v, n) else b for b in complement if b != pivot]
    rows = np.array([[(v >> q) & 1 for q in range(2 * n)] for v in chosen], dtype=np.uint8)
    return StabilizerCode(rows)


def _scrambled_code(n: int, k: int, gates: int, rng: np.random.Generator) -> StabilizerCode:
    """k qubits Bell-paired with a reference, n - k in |0>, then random all-to-all gates."""
    lines = [f"H {q}\nCX {q} {n + j}" for j, q in enumerate(range(n - k, n))]
    for index in rng.integers(0, 11520, size=gates).tolist():
        a, b = rng.choice(n, size=2, replace=False).tolist()
        lines.append(f"CLIFFORD_2Q({index}) {a} {b}")
    sim = Simulator(n + k, phases=False)
    sim.run(Circuit("\n".join(lines)))
    return StabilizerCode.from_simulator(sim, range(n, n + k))


def test_rmt_formula_reproduces_the_published_values():
    """Eq. (23) at N = 64, k = 32 (ns = 32): the values the literature memo lists, and the
    critical 0.38968 = 1 - r_c of the paper; the ansatz gives P[I > 0] = 0.71121 there."""
    ns = 32
    published = {14: 0.0308, 15: 0.1174, 16: 0.3897, 17: 0.7794}
    for m, value in published.items():
        assert _gullans_failure(m, ns) == pytest.approx(value, abs=5e-5)
    assert _gullans_failure(16, ns) == pytest.approx(0.38968, abs=5e-6)
    assert _gullans_failure(8, 16) == pytest.approx(0.389675, abs=5e-6)  # converging in ne
    assert _rmt_any_loss(16, ns) == pytest.approx(0.711212, abs=5e-6)
    assert _rmt_any_loss(17, ns) == 1.0  # rank M_S <= ns < 2m


@pytest.mark.parametrize("source", ["uniform", "scrambled"])
def test_random_codes_follow_the_random_matrix_formula(source, rng):
    """Uniformly random [[64, 32]] codes and codes of 16 random all-to-all gates per qubit:
    P(F) = 1 - E[2^-I] (their quantity) and P[I > 0] (ours) at m = 14..17, within 4 standard
    errors of the spread over codes."""
    n, k = 64, 32
    codes_count, samples = _size(240, 12), _size(64, 8)
    if source == "uniform":
        codes = [_uniform_code(n, n - k, rng) for _ in range(codes_count)]
    else:
        codes = [_scrambled_code(n, k, 16 * n, rng) for _ in range(codes_count)]
    assert {c.num_logical for c in codes} == {k}
    for m in (14, 15, 16, 17):
        lost = np.array(
            [
                [c.lost_information(rng.choice(n, size=m, replace=False).tolist())
                 for _ in range(samples)]
                for c in codes
            ]
        )  # fmt: skip
        theirs = (1.0 - 2.0**-lost).mean(axis=1)
        ours = (lost > 0).mean(axis=1)
        models = ((theirs, _gullans_failure(m, n - k)), (ours, _rmt_any_loss(m, n - k)))
        for per_code, model in models:
            se = per_code.std(ddof=1) / math.sqrt(len(per_code))
            floor = 1.0 / (per_code.size * samples)  # one draw's worth (ours is 1 at m = 17)
            tol = Z_MAX * max(se, floor) if not JIT_FREE else 0.25
            assert abs(per_code.mean() - model) <= tol, (source, m, per_code.mean(), model)
