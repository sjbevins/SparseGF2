"""Observables: checked against brute-force definitions and independent state vectors."""

import numpy as np
import pytest
import stim

from sparsegf2 import Simulator
from sparsegf2 import observables as ob


def _random_state(n: int, rng: np.random.Generator, phases: bool = False) -> Simulator:
    sim = Simulator.from_tableau(
        stim.Tableau.random(n), phases=phases, seed=int(rng.integers(1 << 30))
    )
    for _ in range(n // 3):
        sim.measure(int(rng.integers(n)))
    return sim


def _purification(n: int, rng: np.random.Generator, steps: int, p: float) -> Simulator:
    """n system qubits (0..n-1) Bell-paired with n references (n..2n-1), then monitored dynamics."""
    sim = Simulator(2 * n, phases=False, seed=int(rng.integers(1 << 30)))
    for i in range(n):
        sim.h(i)
        sim.cx(i, n + i)
    for _ in range(steps):
        a, b = (int(v) for v in rng.choice(n, 2, replace=False))
        sim.clifford_2q(int(rng.integers(11520)), a, b)
        if rng.random() < p:
            sim.measure(int(rng.integers(n)))
    return sim


def _statevector(sim: Simulator) -> np.ndarray:
    """The state vector stabilized by the simulator's generators (signs + if phase-free).

    Any consistent choice of signs gives a state with the same entanglement.
    """
    n = sim.num_qubits
    proj = np.eye(2**n, dtype=complex)
    for p in sim.stabilizers():
        proj = proj @ (np.eye(2**n) + p.to_matrix()) / 2
    w, v = np.linalg.eigh(proj)
    return v[:, np.argmax(w)]


def _sv_entropy(psi: np.ndarray, region, n: int) -> float:
    region = sorted(region)
    if not region or len(region) == n:
        return 0.0
    rest = [q for q in range(n) if q not in region]
    m = psi.reshape([2] * n).transpose(region + rest).reshape(2 ** len(region), -1)
    s = np.linalg.svd(m, compute_uv=False) ** 2
    s = s[s > 1e-12]
    return float(-(s * np.log2(s)).sum())


def test_mutual_and_tripartite_information_match_state_vector(rng):
    n = 6
    for _ in range(20):
        sim = _random_state(n, rng, phases=True)
        psi = _statevector(sim)
        perm = [int(q) for q in rng.permutation(n)]
        a, b, c = perm[:2], perm[2:4], perm[4:5]
        sv = {
            key: _sv_entropy(psi, reg, n)
            for key, reg in {
                "a": a,
                "b": b,
                "c": c,
                "ab": a + b,
                "ac": a + c,
                "bc": b + c,
                "abc": a + b + c,
            }.items()
        }
        assert ob.mutual_information(sim, a, b) == pytest.approx(sv["a"] + sv["b"] - sv["ab"])
        expected_i3 = sv["a"] + sv["b"] + sv["c"] - sv["ab"] - sv["ac"] - sv["bc"] + sv["abc"]
        assert ob.tripartite_information(sim, a, b, c) == pytest.approx(expected_i3)


@pytest.mark.parametrize("n", [1, 2, 5, 63, 64, 65, 130])
def test_entropy_profile_equals_prefix_entropies(n, rng):
    sim = _random_state(n, rng)
    profile = ob.entropy_profile(sim)
    assert profile.shape == (n + 1,)
    assert profile[0] == profile[n] == 0
    for k in range(n + 1):
        assert profile[k] == sim.entropy(range(k))
    order = [int(q) for q in rng.permutation(n)]
    custom = ob.entropy_profile(sim, order)
    for k in (0, n // 3, n // 2, n):
        assert custom[k] == sim.entropy(order[:k])


@pytest.mark.parametrize("n", [3, 8, 33, 70])
def test_lost_logical_information_matches_definition(n, rng):
    sim = _purification(n, rng, steps=4 * n, p=0.1)
    system, reference = list(range(n)), list(range(n, 2 * n))
    s_r = sim.entropy(reference)
    assert ob.code_dimension(sim, reference) == s_r
    for _ in range(20):
        m = int(rng.integers(0, n + 1))
        erased = [int(q) for q in rng.choice(system, m, replace=False)]
        expected = s_r + sim.entropy(erased) - sim.entropy(erased + reference)
        lost = ob.lost_logical_information(sim, erased, reference)
        assert lost == expected
        assert 0 <= lost <= 2 * s_r
        assert ob.erasure_is_correctable(sim, erased, reference) == (expected == 0)
        assert ob.recovery_probability(sim, erased, reference) == 2.0**-expected


def test_lost_logical_information_matches_state_vector(rng):
    n = 3  # 3 system + 3 reference = 6 qubits
    for _ in range(15):
        sim = _purification(n, rng, steps=6, p=0.2)
        signed = Simulator.from_tableau(
            stim.Tableau.from_stabilizers([p.to_stim() for p in sim.stabilizers()])
        )
        psi = _statevector(signed)
        reference = [3, 4, 5]
        erased = [int(q) for q in rng.choice(3, int(rng.integers(1, 3)), replace=False)]
        expected = (
            _sv_entropy(psi, reference, 6)
            + _sv_entropy(psi, erased, 6)
            - _sv_entropy(psi, erased + reference, 6)
        )
        assert ob.lost_logical_information(sim, erased, reference) == pytest.approx(expected)


def _brute_contiguous_distance(sim, system, reference, periodic):
    if sim.entropy(reference) == 0:
        return 0
    L = len(system)
    for length in range(1, L + 1):
        starts = range(L) if periodic else range(L - length + 1)
        for s in starts:
            run = [system[(s + t) % L] for t in range(length)]
            if ob.lost_logical_information(sim, run, reference) > 0:
                return length
    return L


@pytest.mark.parametrize("periodic", [True, False])
def test_contiguous_code_distance_matches_brute_force(periodic, rng):
    for n in (4, 7, 12, 20):
        for p in (0.0, 0.05, 0.15):
            sim = _purification(n, rng, steps=3 * n, p=p)
            system, reference = list(range(n)), list(range(n, 2 * n))
            assert ob.contiguous_code_distance(sim, system, reference, periodic=periodic) == (
                _brute_contiguous_distance(sim, system, reference, periodic)
            )


def test_fresh_purification_state_values():
    n = 10
    sim = Simulator(2 * n, phases=False)
    for i in range(n):
        sim.h(i)
        sim.cx(i, n + i)
    system, reference = list(range(n)), list(range(n, 2 * n))
    assert ob.code_dimension(sim, reference) == n
    assert ob.contiguous_code_distance(sim, system, reference) == 1
    stats = ob.random_erasure_statistics(sim, system, reference, [0, 1, 4, n], 30, seed=1)
    np.testing.assert_array_equal(stats["lost"][0], 0)
    np.testing.assert_array_equal(stats["lost"][1], 2)
    np.testing.assert_array_equal(stats["lost"][2], 8)
    np.testing.assert_array_equal(stats["lost"][3], 2 * n)
    np.testing.assert_allclose(stats["correctable_fraction"], [1, 0, 0, 0])


def test_random_erasure_statistics_reproducible_and_consistent(rng):
    n = 16
    sim = _purification(n, rng, steps=5 * n, p=0.05)
    system, reference = list(range(n)), list(range(n, 2 * n))
    a = ob.random_erasure_statistics(sim, system, reference, [2, 5, 8], 100, seed=9)
    b = ob.random_erasure_statistics(sim, system, reference, [2, 5, 8], 100, seed=9)
    np.testing.assert_array_equal(a["lost"], b["lost"])
    k = ob.code_dimension(sim, reference)
    assert a["lost"].min() >= 0 and a["lost"].max() <= 2 * k
    np.testing.assert_allclose(a["correctable_fraction"], (a["lost"] == 0).mean(axis=1))
    # larger erasures never lose less on average (monotonicity of I(R:E))
    means = a["lost"].mean(axis=1)
    assert means[0] <= means[1] <= means[2] + 1e-9


def test_stabilizer_weights():
    sim = Simulator(3)
    np.testing.assert_array_equal(ob.stabilizer_weights(sim), [1, 1, 1])
    sim.h(0)
    sim.cx(0, 1, 1, 2)
    assert sorted(ob.stabilizer_weights(sim).tolist()) == [2, 2, 3]


def test_argument_validation():
    sim = Simulator(4)
    with pytest.raises(ValueError, match="disjoint"):
        ob.mutual_information(sim, [0, 1], [1, 2])
    with pytest.raises(ValueError, match="duplicates"):
        ob.entropy_profile(sim, [0, 0, 1])
    with pytest.raises(IndexError):
        ob.code_dimension(sim, [9])
    with pytest.raises(ValueError, match="reference"):
        ob.lost_logical_information(sim, [0, 1], [1])
    with pytest.raises(ValueError, match="sizes"):
        ob.random_erasure_statistics(sim, [0, 1], [2], [5], 3)


# -- clipped gauge -------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 6, 31, 64, 65, 100])
def test_clipped_gauge_reproduces_every_bipartition(n, rng):
    sim = _random_state(n, rng)
    order = [int(q) for q in rng.permutation(n)]
    left, right = ob.clipped_gauge(sim, order)
    assert np.all((left >= 0) & (left <= right) & (right < n))
    profile = ob.entropy_profile(sim, order)
    crossing = np.array([np.sum((left < k) & (k <= right)) for k in range(n + 1)])
    np.testing.assert_array_equal(crossing, 2 * profile)
    # clipped gauge: at most two left endpoints and two right endpoints per site
    assert np.bincount(left, minlength=n).max() <= 2
    assert np.bincount(right, minlength=n).max() <= 2


def test_clipped_gauge_of_known_states():
    sim = Simulator(4)
    left, right = ob.clipped_gauge(sim)
    np.testing.assert_array_equal(left, right)  # product state: every stabilizer has length 1
    np.testing.assert_array_equal(ob.stabilizer_length_distribution(sim), [0, 4, 0, 0, 0])
    ghz = Simulator(4)
    ghz.h(0)
    ghz.cx(0, 1, 1, 2, 2, 3)
    lengths = ob.stabilizer_length_distribution(ghz)
    assert lengths[2] == 3 and lengths[4] == 1  # ZZ pairs and the XXXX string
    with pytest.raises(ValueError, match="all qubits"):
        ob.clipped_gauge(ghz, [0, 1])


# -- erasure profile -----------------------------------------------------------------------


def test_erasure_profile_matches_definition(rng):
    n = 14
    sim = _purification(n, rng, steps=5 * n, p=0.07)
    system, reference = list(range(n)), list(range(n, 2 * n))
    out = ob.erasure_profile(sim, system, reference, samples=8, seed=3)
    lost = out["lost"]
    k = ob.code_dimension(sim, reference)
    assert lost.shape == (8, n + 1)
    np.testing.assert_array_equal(lost[:, 0], 0)
    np.testing.assert_array_equal(lost[:, n], 2 * k)
    assert np.all(np.diff(lost, axis=1) >= 0)  # I(R:E) grows with E
    np.testing.assert_allclose(out["correctable_fraction"], (lost == 0).mean(axis=0))
    again = ob.erasure_profile(sim, system, reference, samples=8, seed=3)
    np.testing.assert_array_equal(again["lost"], lost)
    # every value is I(R:E) of some m-subset, so it lies within the min/max over all subsets
    for m in (1, 2):
        from itertools import combinations

        values = [
            ob.lost_logical_information(sim, list(c), reference) for c in combinations(system, m)
        ]
        assert min(values) <= lost[:, m].min() and lost[:, m].max() <= max(values)


# -- analysis ------------------------------------------------------------------------------


def test_kaplan_meier_known_answer():
    from sparsegf2 import analysis

    # events at 1, 2, 2; censored at 3 and 3
    km = analysis.kaplan_meier(np.array([1, 2, 2, 3, 3]), np.array([1, 1, 1, 0, 0], dtype=bool))
    np.testing.assert_array_equal(km["times"], [1, 2])
    np.testing.assert_allclose(km["survival"], [4 / 5, 4 / 5 * (1 - 2 / 4)])
    np.testing.assert_array_equal(km["at_risk"], [5, 4])
    no_censoring = analysis.kaplan_meier(np.array([3, 1, 2]), np.ones(3, dtype=bool))
    np.testing.assert_allclose(no_censoring["survival"], [2 / 3, 1 / 3, 0])


@pytest.mark.parametrize("n", [2, 7, 10, 11, 100, 101, 999])
def test_kaplan_meier_without_censoring_is_a_count_over_n(n):
    from sparsegf2 import analysis

    rng = np.random.default_rng(n)
    t = rng.integers(0, max(3, n // 3), size=n)
    km = analysis.kaplan_meier(t, np.ones(n, dtype=bool))
    # oracle: numpy counting, one float division of two integers
    expected = np.array([np.count_nonzero(t > x) / n for x in np.unique(t)])
    np.testing.assert_array_equal(km["survival"], expected)


def test_kaplan_meier_survival_one_half_is_exactly_one_half():
    from sparsegf2 import analysis

    # N = 10: events at 1, 1, 2, 2, 3, 4, 5; censored at 3, 4, 4. Hand arithmetic,
    # risk set = times >= t:
    #   t = 1: n = 10, d = 2 -> 8/10
    #   t = 2: n =  8, d = 2 -> 6/8
    #   t = 3: n =  6, d = 1 -> 5/6      (the censored entry at 3 is still at risk)
    # product 4/5 * 3/4 * 5/6 = 60/120 = 1/2 exactly. The float product
    # ((1 - 0.2) * 0.75) * (1 - 1/6) is not 0.5 (it is one ulp off).
    t = np.array([4, 1, 4, 2, 5, 3, 1, 3, 2, 4])
    o = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
    km = analysis.kaplan_meier(t, o)
    np.testing.assert_array_equal(km["times"], [1, 2, 3, 4, 5])
    assert km["survival"][2] == 0.5
    # no censoring: 5 of 10 have purified by t = 2 -> exactly 1/2
    clean = analysis.kaplan_meier(np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 4]), np.ones(10, dtype=bool))
    assert clean["survival"][1] == 0.5
    # censoring strictly before the event time: 8 trajectories;
    # t = 1: the 2 censored at 0 have left, 6 at risk, d = 3 -> 3/6 = 1/2
    early = analysis.kaplan_meier(
        np.array([0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
    )
    assert early["survival"][0] == 0.5


def test_bernoulli_erasure_curve():
    from sparsegf2 import analysis

    n = 20
    linear = np.arange(n + 1) / n  # per-size quantity m/n has mean p under Binomial(n, p)
    np.testing.assert_allclose(
        analysis.bernoulli_erasure_curve(linear, [0.0, 0.3, 1.0]), [0.0, 0.3, 1.0]
    )
    const = np.full(n + 1, 0.7)
    assert analysis.bernoulli_erasure_curve(const, 0.42) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        analysis.bernoulli_erasure_curve(const, 1.5)


def test_mean_and_error_ignores_nan():
    from sparsegf2 import analysis

    v = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 4.0]])
    mean, err = analysis.mean_and_error(v)
    np.testing.assert_allclose(mean, [3.0, 3.0])
    np.testing.assert_allclose(
        err, [np.std([1, 3, 5], ddof=1) / np.sqrt(3), np.std([2, 4], ddof=1) / np.sqrt(2)]
    )


# -- regressions from the review ------------------------------------------------------------


def test_erasure_docstrings_give_the_failure_probability_correctly():
    from sparsegf2 import analysis, mipt

    n, p = 10, 0.1
    sim = Simulator(2 * n, phases=False)
    for i in range(n):  # every system qubit Bell-paired: any non-empty erasure fails
        sim.h(i)
        sim.cx(i, n + i)
    profile = ob.erasure_profile(sim, range(n), range(n, 2 * n), samples=5, seed=1)
    failure = analysis.bernoulli_erasure_curve(1 - profile["correctable_fraction"], p)
    assert failure == pytest.approx(1 - (1 - p) ** n)
    doc = " ".join(ob.erasure_profile.__doc__.split())
    assert "1 - " in doc and "success probability" in doc
    # the engine's first-failure sizes give the same curve
    spec = mipt.MonitoredCircuit(
        num_system=n, edges=[(i, (i + 1) % n) for i in range(n)], layers=0, measurement_rate=0.0,
        reference="purification", probes=[mipt.erasure_first_failure(samples=5)],
    )  # fmt: skip
    first = spec.run(2, seed=1).values["erasure_first_failure"]
    assert analysis.erasure_failure_probability(first, n, p) == pytest.approx(1 - (1 - p) ** n)


def test_kaplan_meier_refuses_the_censoring_marker():
    from sparsegf2 import analysis, graphs, mipt

    spec = mipt.MonitoredCircuit(
        num_system=12, edges=graphs.cycle(12), layers=24, measurement_rate=0.05,
        reference="single", reference_partner=0, scramble_layers=12,
        checkpoints="every_layer", stop_when_purified=True,
    )  # fmt: skip
    r = spec.run(300, seed=5)
    tau = r.purification_layer
    assert (tau < 0).any() and (tau >= 0).any()
    with pytest.raises(ValueError, match=r"result\.purification_times\(\)"):
        analysis.kaplan_meier(tau, tau >= 0)
    km = analysis.kaplan_meier(*r.purification_times())
    layers, surviving = r.survival()
    assert km["survival"][-1] == pytest.approx(surviving[-1])
    with pytest.raises(ValueError, match="non-negative"):
        analysis.kaplan_meier(np.array([1.0, np.nan]), np.array([True, False]))


def test_contiguous_distance_when_the_reference_is_held_elsewhere():
    # qubits 0, 1 (the system) hold their own Bell pair; the reference 2 is
    # paired with qubit 3, outside system and reference: nothing is uncorrectable
    sim = Simulator(4, phases=False)
    sim.h(2)
    sim.cx(2, 3)
    sim.h(0)
    sim.cx(0, 1)
    assert ob.code_dimension(sim, [2]) == 1
    assert ob.lost_logical_information(sim, [0, 1], [2]) == 0
    for periodic in (True, False):
        assert ob.contiguous_code_distance(sim, [0, 1], [2], periodic=periodic) == 0
    # GHZ system, reference paired outside: still no uncorrectable run
    ghz = Simulator(5, phases=False)
    ghz.h(0)
    ghz.cx(0, 1, 1, 2)
    ghz.h(3)
    ghz.cx(3, 4)
    assert ob.contiguous_code_distance(ghz, [0, 1, 2], [3]) == 0
    # when the whole system is the shortest uncorrectable run, it is still reported
    bell = Simulator(3, phases=False)
    bell.h(0)
    bell.cx(0, 1)
    assert ob.contiguous_code_distance(bell, [0], [1]) == 1
    assert ob.contiguous_code_distance(bell, [0, 2], [1], periodic=False) == 1


def test_fractional_qubit_indices_are_refused():
    from sparsegf2 import mipt

    sim = Simulator(4, phases=False)
    sim.h(0)
    sim.cx(0, 1)
    for call in (
        lambda: ob.mutual_information(sim, [0.7], [1.9]),
        lambda: ob.entropy_profile(sim, [1.9]),
        lambda: ob.code_dimension(sim, [1.5]),
        lambda: ob.lost_logical_information(sim, [0.2], [1.2]),
        lambda: mipt.entropy([0.7, 1.9]),
        lambda: mipt.lost_information([0.5]),
    ):
        with pytest.raises(TypeError):
            call()
    assert ob.mutual_information(sim, np.array([0]), [np.int64(1)]) == 2  # numpy ints are fine
