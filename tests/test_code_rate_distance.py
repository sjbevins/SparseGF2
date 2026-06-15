"""code_rate and contiguous_distance observables."""

from __future__ import annotations

import pytest

from sparsegf2 import (
    SparseGF2,
    code_dimension,
    code_rate,
    contiguous_distance,
    from_bell_purification,
    from_zero_state,
)


def test_code_rate_fresh_purification_is_one():
    pur = from_bell_purification(6)
    assert code_dimension(pur, 6) == 6
    assert code_rate(pur, 6) == pytest.approx(1.0)


def test_code_rate_in_unit_interval():
    pur = from_bell_purification(8)
    # measure a few system qubits -> k drops, rate in [0, 1)
    for q in range(3):
        pur.measure_z(q)
    r = code_rate(pur, 8)
    assert 0.0 <= r < 1.0
    assert r == code_dimension(pur, 8) / 8


def test_code_rate_zero_system():
    assert code_rate(from_zero_state(0), 0) == 0.0


def test_contiguous_distance_fresh_purification_is_one():
    # every system qubit individually carries a logical -> minimum distance 1
    pur = from_bell_purification(6)
    assert contiguous_distance(pur, 6) == 1


def test_contiguous_distance_fully_purified_is_none():
    # a product |0^{2n}> state: the system is decoupled from the reference
    sim = from_zero_state(12)
    assert contiguous_distance(sim, 6) is None


def test_contiguous_distance_single_ref_picture():
    # single reference Bell-paired with the last system qubit: distance 1
    sim = SparseGF2(5)
    sim.apply_h(3)
    sim.apply_cx(3, 4)  # qubit 4 is the reference of system qubit 3
    d = contiguous_distance(sim, 4, reference=[4])
    assert d == 1


def test_contiguous_distance_is_the_minimum():
    # a GHZ-style purification where a single logical needs only one qubit
    pur = from_bell_purification(4)
    pur.measure_z(0)  # purify qubit 0; the rest still each hold a logical
    d = contiguous_distance(pur, 4)
    assert d == 1  # qubit 1 (say) still carries a logical, contiguous length 1


def test_periodic_vs_open_windows_run():
    pur = from_bell_purification(6)
    assert contiguous_distance(pur, 6, periodic=True) == 1
    assert contiguous_distance(pur, 6, periodic=False) == 1


def _contiguous_distance_bruteforce(sim, n_system, *, periodic=False):
    """Reference: scan windows and test I(A:R) > 0 directly (the definition).

    The fast :func:`contiguous_distance` factors the reference out of the
    stabilizer block once; this brute-force scan recomputes the mutual
    information per window. They must agree exactly.
    """
    from sparsegf2 import mutual_information

    ref = list(range(n_system, 2 * n_system))
    for length in range(1, n_system + 1):
        starts = range(n_system) if periodic else range(n_system - length + 1)
        for start in starts:
            window = [(start + j) % n_system for j in range(length)]
            if mutual_information(sim, window, ref) > 0:
                return length
    return None


@pytest.mark.parametrize("n", [4, 6, 8, 10])
@pytest.mark.parametrize("p", [0.0, 0.1, 0.2, 0.5])
def test_contiguous_distance_matches_bruteforce_definition(n, p):
    # The optimized scan must equal the per-window I(A:R) definition for both
    # open and periodic windows across the whole purification transition.
    from sparsegf2.circuits import CircuitConfig, simulate

    for seed in range(6):
        rec = simulate(
            CircuitConfig(graph_spec="cycle", n=n, picture="purification", p=p, depth_factor=4),
            sample_seed=seed,
            save_tableau=True,
        )
        sim = SparseGF2.from_symplectic(rec.final_tableau)
        for periodic in (False, True):
            assert contiguous_distance(
                sim, n, periodic=periodic
            ) == _contiguous_distance_bruteforce(sim, n, periodic=periodic)
