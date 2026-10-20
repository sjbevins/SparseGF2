"""The ensemble engine's reference-row bookkeeping, against the generic formulas.

With references that nothing acts on, the engine keeps the set of stabilizer
rows touching them (``ref_rows``) by choosing reference-free pivots when it can.
These tests evolve random states with that rule and check, at every step, that
the stabilizer group is unchanged and that S(R), I(R:E), erasure profiles and
contiguous distances computed from ``ref_rows`` equal the generic computations.
"""

import numpy as np
import pytest

from sparsegf2._kernel import entropy as ke
from sparsegf2._kernel import tableau as kt
from sparsegf2._kernel.tables import ANF1, ANF2, IDENTITY_SYM1, IDENTITY_SYM2, LIN1, LIN2
from sparsegf2.gates import gate

H = gate("H").clifford.index()
CX = gate("CX").clifford.index()


def _canonical(tab, n):
    xs, zs, signs = kt.rows_of(tab, n, 0, n)
    kt.canonicalize(xs, zs, signs, False)
    return xs, zs


def _bell_paired(n_sys, partners):
    """System 0..n_sys-1, reference n_sys + j Bell-paired with system qubit partners[j]."""
    n = n_sys + len(partners)
    ws = (n + 63) // 64
    tab = np.zeros((2 * n + 1, ws), dtype=np.uint64)
    kt.init_zero(tab, n, ws, False)
    for j, q in enumerate(partners):
        kt.apply_1q(tab, q, H, LIN1, ANF1, IDENTITY_SYM1, ws, False)
        kt.apply_2q(tab, q, n_sys + j, CX, LIN2, ANF2, IDENTITY_SYM2, ws, False)
    return tab, n, ws


@pytest.mark.parametrize(
    "n_sys, partners",
    [
        (10, list(range(10))),
        (12, [5]),
        (70, list(range(0, 70, 3))),
        (9, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ],
)
def test_reference_rows_track_the_code(n_sys, partners, rng):
    tab, n, ws = _bell_paired(n_sys, partners)
    plain = tab.copy()
    refs = np.arange(n_sys, n, dtype=np.int64)
    system = np.arange(n_sys, dtype=np.int64)
    ref_rows = np.zeros(ws, dtype=np.uint64)
    ke.reference_rows(tab, ws, refs, ref_rows)
    anti = np.zeros(ws, dtype=np.uint64)
    supp = np.zeros(n, dtype=np.int64)
    mask_buf = np.zeros(n, dtype=np.bool_)
    comp_buf = np.zeros(n, dtype=np.int64)
    scratch = np.zeros((2 * n, ws), dtype=np.uint64)
    union_buf = np.zeros(n, dtype=np.int64)
    profile_rows = np.zeros(n_sys + 1, dtype=np.int64)
    profile_generic = np.zeros(n_sys + 1, dtype=np.int64)
    checked_mixed = 0
    for step in range(6 * n_sys):
        for _ in range(n_sys // 2):
            a, b = rng.choice(n_sys, 2, replace=False)
            g = int(rng.integers(11520))
            for t in (tab, plain):
                kt.apply_2q(t, a, b, g, LIN2, ANF2, IDENTITY_SYM2, ws, False)
        for q in np.flatnonzero(rng.random(n_sys) < 0.15):
            r1 = kt.measure_1q_phase_free_tracked(tab, n, ws, q, 2, anti, supp, ref_rows)
            r2 = kt.measure_1q_phase_free(plain, n, ws, q, 2, anti, supp)
            assert r1 == r2
        if step % 3:
            continue
        for a, b in zip(_canonical(tab, n), _canonical(plain, n), strict=True):
            np.testing.assert_array_equal(a, b)
        s_r = ke.region_entropy(tab, n, ws, refs, mask_buf, comp_buf, scratch)
        k = sum(bin(int(w)).count("1") for w in ref_rows) - len(refs)
        assert k == s_r
        checked_mixed += k > 0
        erased = rng.choice(n_sys, int(rng.integers(0, n_sys + 1)), replace=False).astype(np.int64)
        assert ke.lost_information_rows(
            tab, ws, erased, ref_rows, scratch
        ) == ke.mutual_information_with(
            tab, n, ws, refs, erased, mask_buf, comp_buf, scratch, union_buf
        )
        order = rng.permutation(n_sys).astype(np.int64)
        ke.erasure_profile_rows(tab, ws, order, ref_rows, profile_rows)
        ke.erasure_profile(tab, n, ws, order, refs, profile_generic)
        np.testing.assert_array_equal(profile_rows, profile_generic)
        for periodic in (True, False):
            assert ke.contiguous_distance_rows(
                tab, ws, system, ref_rows, len(refs), periodic
            ) == ke.contiguous_distance(tab, n, ws, system, refs, periodic)
    assert checked_mixed > 0


def test_tracked_measurement_without_references_is_the_plain_one(rng):
    n = 20
    ws = 1
    a = np.zeros((2 * n + 1, ws), dtype=np.uint64)
    kt.init_zero(a, n, ws, False)
    b = a.copy()
    none = np.zeros(ws, dtype=np.uint64)
    anti = np.zeros(ws, dtype=np.uint64)
    supp = np.zeros(n, dtype=np.int64)
    for _ in range(200):
        x, y = rng.choice(n, 2, replace=False)
        g = int(rng.integers(11520))
        for t in (a, b):
            kt.apply_2q(t, x, y, g, LIN2, ANF2, IDENTITY_SYM2, ws, False)
        q = int(rng.integers(n))
        code = int(rng.integers(1, 4))
        kt.measure_1q_phase_free_tracked(a, n, ws, q, code, anti, supp, none)
        kt.measure_1q_phase_free(b, n, ws, q, code, anti, supp)
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "n_sys, partners, local, rate, reached",
    [
        (96, list(range(96)), True, 0.12, {"single", "gauge"}),  # purification, ring-local gates
        (64, list(range(64)), False, 0.05, {"single", "short"}),  # many logical qubits
        (80, list(range(0, 80, 4)), False, 0.12, {"gauge"}),  # a quarter of the system paired
        (33, [7], False, 0.12, {"gauge"}),  # one reference qubit
    ],
)
def test_contiguous_distance_rows_matches_the_scan(n_sys, partners, local, rate, reached, rng):
    """The clipped-gauge contiguous distance equals the scan it replaced, along whole trajectories.

    The cases reach every path of the kernel between them (``reached``): a failing single qubit,
    short runs when there are many logical qubits, and the gauges. Each state is checked periodic
    and open, in the ring order and in a random order.
    """
    tab, n, ws = _bell_paired(n_sys, partners)
    refs = np.arange(n_sys, n, dtype=np.int64)
    ref_rows = np.zeros(ws, dtype=np.uint64)
    ke.reference_rows(tab, ws, refs, ref_rows)
    anti = np.zeros(ws, dtype=np.uint64)
    supp = np.zeros(n, dtype=np.int64)
    orders = (np.arange(n_sys, dtype=np.int64), rng.permutation(n_sys).astype(np.int64))
    paths = set()
    for _ in range(6 * n_sys):
        for _ in range(n_sys // 2):
            if local:
                a = int(rng.integers(n_sys))
                b = (a + int(rng.integers(1, 3))) % n_sys
            else:
                a, b = (int(x) for x in rng.choice(n_sys, 2, replace=False))
            kt.apply_2q(tab, a, b, int(rng.integers(11520)), LIN2, ANF2, IDENTITY_SYM2, ws, False)
        for q in np.flatnonzero(rng.random(n_sys) < rate):
            kt.measure_1q_phase_free_tracked(tab, n, ws, q, 2, anti, supp, ref_rows)
        k = sum(bin(int(w)).count("1") for w in ref_rows) - len(refs)
        for order in orders:
            for periodic in (True, False):
                d = ke.contiguous_distance_rows(tab, ws, order, ref_rows, len(refs), periodic)
                assert d == ke.contiguous_distance_rows_scan(
                    tab, ws, order, ref_rows, len(refs), periodic
                )
                if k > 0:
                    short = 8 * k >= n_sys and d <= min(8, k // 12)
                    paths.add("single" if d == 1 else "short" if short else "gauge")
    assert reached <= paths
