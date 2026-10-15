"""Stabilizer codes (sparsegf2.codes) and the erasure estimators of sparsegf2.analysis."""

import itertools
import math

import numpy as np
import pytest

from sparsegf2 import analysis, graphs, mipt
from sparsegf2.codes import StabilizerCode

FIVE_QUBIT = ["XZZX_", "_XZZX", "X_XZZ", "ZX_XZ"]


def _gf2_rank(m: np.ndarray) -> int:
    m = (np.array(m, dtype=np.uint8) % 2).copy()
    rank = 0
    for col in range(m.shape[1]):
        rows = np.flatnonzero(m[rank:, col]) + rank
        if rows.size == 0:
            continue
        m[[rank, rows[0]]] = m[[rows[0], rank]]
        others = np.flatnonzero(m[:, col])
        others = others[others != rank]
        m[others] ^= m[rank]
        rank += 1
        if rank == m.shape[0]:
            break
    return rank


def _lost_by_formula(h: np.ndarray, erased) -> int:
    """l(A) = 2|A| - s - rank(H_A) + rank(H_Abar) from the check matrix H = (X | Z)."""
    n = h.shape[1] // 2
    a = sorted(erased)
    abar = [q for q in range(n) if q not in a]

    def cols(qs):
        return h[:, [*qs, *(n + q for q in qs)]]

    s = _gf2_rank(h)
    return 2 * len(a) - s - _gf2_rank(cols(a)) + _gf2_rank(cols(abar))


def _engine_codes(n, count, seed, reference="purification"):
    """Codes left by short monitored circuits, as StabilizerCode objects."""
    out = []
    rng = np.random.default_rng(seed)
    while len(out) < count:
        spec = mipt.MonitoredCircuit(
            num_system=n,
            edges=graphs.watts_strogatz(n, 4, 0.3, seed=int(rng.integers(1 << 30))),
            layers=int(rng.integers(1, 3 * n)),
            measurement_rate=float(rng.uniform(0.0, 0.3)),
            reference=reference,
            checkpoints="final",
            probes=[mipt.code_snapshot()],
        )
        r = spec.run(1, seed=int(rng.integers(1 << 30)))
        if r.snapshots["code_snapshot"][1][0, 0] >= 0:
            out.append(r.code(0, 0))
    return out


# -- StabilizerCode -------------------------------------------------------------------


def test_four_two_two_code():
    code = StabilizerCode(["XXXX", "ZZZZ"])
    assert (code.num_qubits, code.num_stabilizers, code.num_logical) == (4, 2, 2)
    assert code.lost_information([]) == 0
    assert all(code.is_correctable([q]) for q in range(4))
    assert all(code.lost_information(a) == 2 for a in itertools.combinations(range(4), 2))
    assert code.lost_information(range(4)) == 4  # all 2k bits
    assert code.uncorrectable_counts(2).tolist() == [0, 6]
    np.testing.assert_array_equal(code.first_failure_sizes(50, seed=1), 2)


def test_five_qubit_code_has_distance_three():
    code = StabilizerCode(FIVE_QUBIT)
    assert code.num_logical == 1
    assert code.uncorrectable_counts(3).tolist() == [0, 0, 10]
    np.testing.assert_array_equal(code.first_failure_sizes(40, seed=2), 3)


def test_bit_rows_pauli_strings_and_redundant_generators_agree():
    from_text = StabilizerCode(FIVE_QUBIT)
    rows = from_text.stabilizer_matrix()
    assert rows.shape == (4, 10)
    redundant = np.vstack([rows, rows[0] ^ rows[1]])  # a dependent generator
    code = StabilizerCode(redundant)
    assert code.num_stabilizers == 4 and code.num_logical == 1
    for a in itertools.combinations(range(5), 3):
        assert code.lost_information(a) == from_text.lost_information(a)
    packed = StabilizerCode.from_packed(from_text._packed, 5)
    np.testing.assert_array_equal(packed.stabilizer_matrix(), rows)
    assert "k=1" in repr(code)


def test_no_stabilizers_and_no_logical_qubits():
    free = StabilizerCode(np.zeros((0, 6), dtype=np.uint8))  # three bare qubits: k = n
    assert free.num_logical == 3
    assert free.uncorrectable_counts(2).tolist() == [3, 3]
    np.testing.assert_array_equal(free.first_failure_sizes(10, seed=0), 1)
    assert StabilizerCode([], num_qubits=2).num_logical == 2
    full = StabilizerCode(["ZI", "IZ"])  # k = 0: nothing to lose
    assert full.uncorrectable_counts(2).tolist() == [0, 0]
    np.testing.assert_array_equal(full.first_failure_sizes(5, seed=0), 3)
    assert full.lost_information([0, 1]) == 0


def test_invalid_codes_are_refused():
    with pytest.raises(ValueError, match="do not all commute"):
        StabilizerCode(["XI", "ZI"])
    with pytest.raises(ValueError, match="num_qubits"):
        StabilizerCode([])
    with pytest.raises(ValueError, match="0/1"):
        StabilizerCode(np.array([[2, 0]]))
    with pytest.raises(ValueError, match="same number of qubits"):
        StabilizerCode(["XX", "ZZZ"])
    code = StabilizerCode(["XXXX", "ZZZZ"])
    with pytest.raises(ValueError, match="distinct"):
        code.lost_information([0, 0])
    with pytest.raises(ValueError, match=r"in \[0, 4\)"):
        code.lost_information([4])
    with pytest.raises(ValueError, match="max_size"):
        code.uncorrectable_counts(5)


@pytest.mark.parametrize("reference", ["purification", "single"])
def test_lost_information_and_counts_match_brute_force(reference):
    for code in _engine_codes(7, 6, seed=11 if reference == "single" else 12, reference=reference):
        h = code.stabilizer_matrix()
        n = code.num_qubits
        expected = np.zeros(4)
        for m in range(1, 5):
            for a in itertools.combinations(range(n), m):
                lost = code.lost_information(a)
                assert lost == _lost_by_formula(h, a)
                expected[m - 1] += lost > 0
        np.testing.assert_array_equal(code.uncorrectable_counts(4), expected)


def test_counts_on_codes_wider_than_one_word():
    """n > 64: columns span several words; a few correctable pairs/triples are checked by hand."""
    for code in _engine_codes(70, 2, seed=5):
        counts = code.uncorrectable_counts(2)
        singles = sum(not code.is_correctable([q]) for q in range(70))
        assert counts[0] == singles
        rng = np.random.default_rng(0)
        pairs = [tuple(rng.choice(70, 2, replace=False)) for _ in range(200)]
        frac = np.mean([not code.is_correctable(p) for p in pairs])
        assert abs(frac - counts[1] / math.comb(70, 2)) < 0.15


def test_first_failure_law_matches_exact_failure_by_size():
    code = _engine_codes(8, 1, seed=3)[0]
    exact = [
        np.mean([not code.is_correctable(a) for a in itertools.combinations(range(8), m)])
        for m in range(9)
    ]
    sampled = analysis.failure_by_size(code.first_failure_sizes(20000, seed=4), 8)
    np.testing.assert_allclose(sampled, exact, atol=0.02)
    again = code.first_failure_sizes(20000, seed=4)
    np.testing.assert_array_equal(again, code.first_failure_sizes(20000, seed=4))


# -- analysis -------------------------------------------------------------------------


def test_failure_probability_is_the_binomial_mixture_of_failure_by_size():
    n = 12
    rng = np.random.default_rng(1)
    d = rng.integers(1, n + 2, size=(3, 50))  # including n + 1 (no logical qubits)
    p_m = analysis.failure_by_size(d, n)
    assert p_m.shape == (3, n + 1)
    np.testing.assert_allclose(p_m[:, 0], 0)
    for e in (0.0, 0.05, 0.3, 1.0):
        phi = analysis.erasure_failure_probability(d, n, e)
        weights = np.array([math.comb(n, m) * e**m * (1 - e) ** (n - m) for m in range(n + 1)])
        np.testing.assert_allclose(phi, p_m @ weights, atol=1e-13)
    curve = analysis.erasure_failure_probability(d, n, [0.1, 0.2])
    assert curve.shape == (3, 2)
    assert np.all(
        np.diff(analysis.erasure_failure_probability(d, n, np.linspace(0, 1, 30))) >= -1e-12
    )
    np.testing.assert_allclose(analysis.erasure_failure_probability(np.full(4, n + 1), n, 0.7), 0)


def test_hybrid_estimator_uses_exact_counts_for_small_regions():
    code = _engine_codes(10, 1, seed=8)[0]
    n = 10
    exact_all = np.array(
        [0.0]
        + [
            np.mean([not code.is_correctable(a) for a in itertools.combinations(range(n), m)])
            for m in range(1, n + 1)
        ]
    )
    d = code.first_failure_sizes(400, seed=1)
    counts = code.uncorrectable_counts(3)
    np.testing.assert_allclose(analysis.exact_failure_by_size(counts, n), exact_all[1:4])
    for e in (0.05, 0.2):
        weights = np.array([math.comb(n, m) * e**m * (1 - e) ** (n - m) for m in range(n + 1)])
        hybrid = analysis.erasure_failure_probability(d, n, e, exact_counts=counts)
        tail = np.array([weights[max(int(x), 4) :].sum() for x in d])  # Pr[Bin >= max(D, w + 1)]
        expected = weights[1:4] @ exact_all[1:4] + tail.mean()
        assert hybrid == pytest.approx(expected, abs=1e-13)
        assert hybrid == pytest.approx(exact_all @ weights, abs=0.05)


def test_erasure_tolerance_inverts_the_failure_curve():
    n = 16
    d = np.random.default_rng(2).integers(3, 12, size=(20, 30))
    for delta in (0.01, 0.1, 0.5):
        e = analysis.erasure_tolerance(d, n, delta)
        assert np.mean(analysis.erasure_failure_probability(d, n, e)) == pytest.approx(
            delta, abs=1e-9
        )
    assert analysis.erasure_tolerance(np.full((2, 3), n + 1), n, 0.1) == 1.0
    assert analysis.erasure_tolerance(np.ones((2, 3), dtype=int), n, 0.0) == 0.0
    with pytest.raises(ValueError, match="delta"):
        analysis.erasure_tolerance(d, n, 1.5)
    counts = np.zeros((20, 2))  # distance > 2 for every code: the hybrid can only lower Phi
    assert analysis.erasure_tolerance(d, n, 0.1, exact_counts=counts) >= analysis.erasure_tolerance(
        d, n, 0.1
    )


def test_estimators_check_their_inputs():
    with pytest.raises(ValueError):
        analysis.failure_by_size(np.array([0, 3]), 4)  # sizes are 1..n+1
    with pytest.raises(ValueError):
        analysis.failure_by_size(np.array([6]), 4)
    with pytest.raises(ValueError, match="e must be"):
        analysis.erasure_failure_probability(np.array([2]), 4, 1.5)
    with pytest.raises(ValueError, match="sizes up to"):
        analysis.exact_failure_by_size(np.zeros(5), 4)


# -- regressions from the review ------------------------------------------------------


def test_text_inputs_are_read_as_whole_generators():
    assert StabilizerCode("ZZ").num_qubits == 2  # one generator, not one per character
    assert StabilizerCode(["ixx", "izz"]).num_logical == 1  # lowercase identity, not a phase
    assert StabilizerCode(["-XX", "+ZZ"]).num_logical == 0
    with pytest.raises(ValueError, match="num_qubits = 3"):
        StabilizerCode(["XX"], num_qubits=3)
    rows = np.array([[0b11, 0], [0, 0b11]], dtype=np.uint64)  # X0, Z0 plus a stray bit
    with pytest.raises(ValueError, match="beyond qubit 0"):
        StabilizerCode.from_packed(rows, 1)


def test_counts_switch_to_floats_beyond_exact_integers():
    small = StabilizerCode(np.zeros((0, 16), dtype=np.uint8)).uncorrectable_counts(3)
    assert small.dtype == np.int64 and small.tolist() == [8, 28, 56]
    big = StabilizerCode(np.zeros((0, 400), dtype=np.uint8)).uncorrectable_counts(13)
    assert big.dtype == np.float64  # C(200, 13) > 2**63
    # every C(200, m), m <= 12, fits in int64: those counts are exact, then rounded once,
    # so they equal float(C(n, m)) and never exceed it (finding E1)
    assert big[:12].tolist() == [float(math.comb(200, m)) for m in range(1, 13)]
    assert abs(big[12] / math.comb(200, 13) - 1) < 1e-15


def _frozen_then_bare(n, frozen):
    """Qubits 0..frozen-1 frozen (stabilizer Z), the rest bare: k = n - frozen."""
    rows = np.zeros((frozen, 2 * n), dtype=np.uint8)
    rows[np.arange(frozen), n + np.arange(frozen)] = 1
    return StabilizerCode(rows)


def test_counts_stay_exact_integers_while_every_binomial_fits_in_int64():
    """E1: counts above 2**53 are accumulated in int64, so they are exact up to 2**63."""
    n, w = 200, 12
    assert 2**53 < math.comb(n, w) < 2**63
    code = _frozen_then_bare(n, 4)
    assert code.num_logical == n - 4
    counts = code.uncorrectable_counts(w)
    # an erasure is correctable exactly when it holds frozen qubits only
    assert counts.dtype == np.int64
    assert counts.tolist() == [math.comb(n, m) - math.comb(4, m) for m in range(1, w + 1)]


def test_float_counts_include_the_last_level_of_the_walk():
    """C(200, 13) > 2**63: size 13 is a float sum, reached by the walk through 12 frozen qubits."""
    n, w = 200, 13
    counts = _frozen_then_bare(n, 12).uncorrectable_counts(w)
    expected = [math.comb(n, m) - math.comb(12, m) for m in range(1, w + 1)]
    assert counts.dtype == np.float64
    assert counts[:-1].tolist() == [float(c) for c in expected[:-1]]
    assert abs(counts[-1] / expected[-1] - 1) < 1e-15


def test_binomial_tables_are_exact_below_two_to_the_63_and_marked_beyond():
    """The counts kernel's tables: int64 C(a, j) where it fits, -1 where it does not."""
    from sparsegf2._kernel.entropy import _binomial_tables

    n, w = 70, 40  # C(70, 35) ~ 1.1e20 > 2**63: rows from a = 67 on cross the boundary
    exact, approx = _binomial_tables(n, w)
    crossed = 0
    for a in range(n + 1):
        for j in range(w + 1):
            c = math.comb(a, j)
            if c < 2**63:
                assert exact[a, j] == c and approx[a, j] == float(c), (a, j)
            else:
                crossed += 1
                assert exact[a, j] == -1, (a, j)
                assert abs(approx[a, j] / c - 1) <= (2 * j + 1) * 2.0**-53, (a, j)
    assert crossed > 0


def test_engine_counts_beyond_two_to_the_53_pass_through_the_estimators():
    """E1 regression: n = 200, w = 13 at layer 0 of the purification picture (k = n)."""
    from sparsegf2 import api

    n, w = 200, 13
    spec = {
        "num_system": n,
        "graph": {"family": "cycle"},
        "layers": 0,
        "measurement_rate": 0.0,
        "reference": "purification",
        "checkpoints": "final",
        "probes": [{"kind": "uncorrectable_region_counts", "max_size": w}],
    }
    name = f"uncorrectable_region_counts[{w}]"
    counts = api._build_spec(spec).run(1, seed=0).values[name][:, -1]
    assert counts[0, :-1].tolist() == [float(math.comb(n, m)) for m in range(1, w)]
    p = analysis.exact_failure_by_size(counts, n)  # every erasure fails: P_C(m) = 1
    assert p[0, :-1].tolist() == [1.0] * (w - 1)
    assert 1.0 - 1e-15 <= p[0, -1] <= 1.0
    first = np.ones((1, 5))  # every ordering fails at its first qubit
    phi = analysis.erasure_failure_probability(first, n, 0.01, exact_counts=counts)
    assert phi[0] == pytest.approx(1 - 0.99**n, rel=1e-12)  # Pr[Bin(n, e) >= 1]
    summary = api.run_ensemble(spec, trajectories=1, seed=0)["probes"][name]
    assert summary["mean"] == pytest.approx([0.0] + [1.0] * w, abs=1e-12)


def test_erasure_tolerance_edge_cases():
    assert math.isnan(analysis.erasure_tolerance(np.zeros((0, 20)), 9, 0.1))
    assert analysis.erasure_tolerance(np.full(5, 64), 64, 0.0) == 0.0  # Phi(e) = e**64 > 0
    assert analysis.erasure_tolerance(np.full(5, 65), 64, 0.0) == 1.0  # no code ever fails
    # a hybrid estimate that rises and falls: the last crossing is the supremum
    d = np.random.default_rng(0).integers(15, 41, size=100)
    e = analysis.erasure_tolerance(d, 64, 8e-5, exact_counts=[0, 2])
    grid = np.linspace(0, 0.5, 20001)
    phi = analysis.erasure_failure_probability(d, 64, grid, exact_counts=[0, 2])
    assert e == pytest.approx(grid[np.flatnonzero(phi <= 8e-5)[-1]], abs=1e-4)
    with pytest.raises(ValueError, match="one row per code"):
        analysis.erasure_failure_probability(np.full((4, 20), 3), 9, 0.1, exact_counts=[0, 1])
    with pytest.raises(ValueError, match="one row per code"):
        analysis.erasure_tolerance(np.full((4, 20), 3), 9, 0.1, exact_counts=np.zeros((2, 2)))
    with pytest.raises(ValueError, match=r"\[0, C\(n, m\)\]"):
        analysis.exact_failure_by_size([-1, 0], 9)


def test_contiguous_distance_matches_the_engine_probe():
    from sparsegf2 import graphs, mipt

    n = 16
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        layers=40,
        measurement_rate=0.1,
        reference="single",
        reference_partner=0,
        scramble_gates=128,
        probes=[mipt.code_dimension(), mipt.contiguous_distance(), mipt.code_snapshot()],
    )
    res = spec.run(40, seed=3)
    k = res.values["code_dimension"][:, -1]
    engine = res.values["contiguous_distance"][:, -1]
    for t in range(res.num_trajectories):
        code = res.code(t, 0)
        if k[t] > 0:
            assert code.contiguous_distance() == engine[t]
        else:
            assert code.contiguous_distance() == 0
    # Guard against a vacuous test: the engine must produce distances above 1.
    assert np.nanmax(engine[k > 0]) > 1


def test_contiguous_distance_respects_order_and_periodic():
    code = StabilizerCode(["XXXX", "ZZZZ"])  # any pair is uncorrectable
    assert code.contiguous_distance() == 2
    assert code.contiguous_distance([3, 2, 1, 0], periodic=False) == 2
    with pytest.raises(ValueError, match="permutation"):
        code.contiguous_distance([0, 0, 1, 2])


def _rref(m):
    m = (np.asarray(m) % 2).astype(np.uint8)
    r = 0
    for c in range(m.shape[1]):
        nz = np.flatnonzero(m[r:, c])
        if not nz.size:
            continue
        p = r + nz[0]
        m[[r, p]] = m[[p, r]]
        others = np.flatnonzero(m[:, c])
        m[others[others != r]] ^= m[r]
        r += 1
        if r == m.shape[0]:
            break
    return m[:r]


def test_from_simulator_equals_the_engine_code_snapshot():
    import sparsegf2 as sg

    n = 12
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        layers=10,
        measurement_rate=0.2,
        reference="purification",
        probes=[mipt.code_snapshot()],
    )
    res = spec.run(6, seed=5)
    for t in range(6):
        sim = sg.Simulator(spec.num_qubits, phases=False, seed=0)
        sim.run(spec.circuit(t, seed=5))
        mine = StabilizerCode.from_simulator(sim, reference=range(n, 2 * n))
        engine = res.code(t, 0)
        assert mine.num_qubits == n and mine.num_logical == engine.num_logical
        np.testing.assert_array_equal(
            _rref(mine.stabilizer_matrix()), _rref(engine.stabilizer_matrix())
        )


def test_from_simulator_fresh_purification_state_has_no_stabilizers():
    import sparsegf2 as sg

    sim = sg.Simulator(4, phases=False)
    for q in (0, 1):
        sim.h(q)
        sim.cx(q, q + 2)
    code = StabilizerCode.from_simulator(sim, reference=[2, 3])
    assert (code.num_qubits, code.num_logical, code.num_stabilizers) == (2, 2, 0)
