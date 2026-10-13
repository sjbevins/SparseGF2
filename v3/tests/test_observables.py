"""Observables: checked against brute-force definitions and independent state vectors."""

import numpy as np
import pytest
import stim

from sparsegf2_v3 import Simulator
from sparsegf2_v3 import observables as ob


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
