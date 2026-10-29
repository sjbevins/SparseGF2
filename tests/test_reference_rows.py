"""The ensemble engine's reference-row bookkeeping, against the generic formulas.

With references that nothing acts on, the engine keeps the set of stabilizer
rows touching them (``ref_rows``) by choosing reference-free pivots when it can.
These tests evolve random states with that rule and check, at every step, that
the stabilizer group is unchanged and that S(R), I(R:E), erasure profiles and
contiguous distances computed from ``ref_rows`` equal the generic computations.
"""

import numba
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
    "n_sys, partners, local, rate",
    [
        (96, list(range(96)), True, 0.12),  # purification, ring-local gates
        (70, list(range(70)), False, 0.2),  # purification, all-to-all gates
        (64, list(range(64)), False, 0.05),  # many logical qubits: many reference pivots
        (80, list(range(0, 80, 4)), False, 0.12),  # a quarter of the system paired
        (33, [7], False, 0.12),  # one reference qubit
        (130, list(range(130)), False, 0.12),  # rows over five words
    ],
)
def test_searching_the_system_only_changes_nothing(n_sys, partners, local, rate, rng):
    """With ``n_sys`` given (the references follow the system, as in the engine) the tracked
    measurement seeks an unmarked pivot's support among the system qubits only. Along whole
    trajectories every measurement leaves the tableau, the reference rows and the return value bit
    for bit as the search over every qubit does. Both kinds of pivot occur: the first measurement,
    Z on a Bell-paired qubit before any gate, anticommutes only with the marked row X_q X_r."""
    tab, n, ws = _bell_paired(n_sys, partners)
    other = tab.copy()
    refs = np.arange(n_sys, n, dtype=np.int64)
    ref_rows = np.zeros(ws, dtype=np.uint64)
    ke.reference_rows(tab, ws, refs, ref_rows)
    other_rows = ref_rows.copy()
    anti = np.zeros(ws, dtype=np.uint64)
    supp = np.zeros(n, dtype=np.int64)
    pivots = set()

    def measure(q):
        before = ref_rows.copy()
        r1 = kt.measure_1q_phase_free_tracked(tab, n, ws, q, 2, anti, supp, ref_rows, n_sys)
        r2 = kt.measure_1q_phase_free_tracked(other, n, ws, q, 2, anti, supp, other_rows)
        assert r1 == r2
        np.testing.assert_array_equal(ref_rows, other_rows)
        np.testing.assert_array_equal(tab, other)
        if r1:  # a marked pivot leaves ref_rows; an unmarked one changes nothing there
            pivots.add("reference row" if (ref_rows != before).any() else "code row")

    measure(partners[0])
    assert pivots == {"reference row"}
    for _ in range(4 * n_sys):
        for _ in range(n_sys // 2):
            if local:
                a = int(rng.integers(n_sys))
                b = (a + int(rng.integers(1, 3))) % n_sys
            else:
                a, b = (int(x) for x in rng.choice(n_sys, 2, replace=False))
            g = int(rng.integers(11520))
            for t in (tab, other):
                kt.apply_2q(t, a, b, g, LIN2, ANF2, IDENTITY_SYM2, ws, False)
        for q in np.flatnonzero(rng.random(n_sys) < rate):
            measure(q)
    assert pivots == {"reference row", "code row"}


def test_the_search_bound_is_used():
    """``n_sys`` is trusted, not checked. With the precondition broken on purpose (the reference rows
    left empty, so the pivot X_1 X_4 counts as a code row although it acts on reference qubit 4), the
    search stopped at qubit 4 misses that support and the tableau differs from the full search. This
    only shows that the bound is applied: a change that ignored it would pass every exactness test."""
    tab, n, ws = _bell_paired(4, [1])
    other = tab.copy()
    anti = np.zeros(ws, dtype=np.uint64)
    supp = np.zeros(n, dtype=np.int64)
    empty = np.zeros(ws, dtype=np.uint64)
    assert kt.measure_1q_phase_free_tracked(tab, n, ws, 1, 2, anti, supp, empty.copy(), 4) == 2
    assert kt.measure_1q_phase_free_tracked(other, n, ws, 1, 2, anti, supp, empty.copy()) == 2
    assert not np.array_equal(tab, other)


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


@pytest.mark.parametrize(
    "n_sys, partners, local, rate, reached",
    [
        (96, list(range(96)), True, 0.12, {"one", "short", "long"}),  # ring-local gates
        (64, list(range(64)), False, 0.05, {"one", "short"}),  # many logical qubits
        (33, [7], False, 0.01, {"long"}),  # one reference qubit, kept by a low rate
    ],
)
def test_capped_contiguous_distance_is_the_full_one_capped(
    n_sys, partners, local, rate, reached, rng
):
    """contiguous_distance_rows_capped(..., cap) == min(contiguous_distance_rows(...), cap) at every
    step of whole trajectories, periodic and open, in two orders, for caps from 2 to above n (the
    full distance), and 0 without a code; the trajectories reach d = 1, d below the caps and d above
    them. Caps up to sqrt(2 n) + 1 check short runs, larger ones take the full distance. A system
    that leaves a qubit out (not the engine's layout) gives the reference scan, capped, as the full
    distance does. Without the JIT (the weekly job) fewer caps and one order are checked."""
    tab, n, ws = _bell_paired(n_sys, partners)
    refs = np.arange(n_sys, n, dtype=np.int64)
    ref_rows = np.zeros(ws, dtype=np.uint64)
    ke.reference_rows(tab, ws, refs, ref_rows)
    anti = np.zeros(ws, dtype=np.uint64)
    supp = np.zeros(n, dtype=np.int64)
    orders = (np.arange(n_sys, dtype=np.int64), rng.permutation(n_sys).astype(np.int64))
    jit_free = numba.config.DISABLE_JIT
    caps = (
        (2, 3, 8, 16, n_sys) if jit_free else (2, 3, 4, 5, 8, 14, 16, n_sys - 1, n_sys, n_sys + 5)
    )
    seen = set()
    for _ in range(4 * n_sys):
        for _ in range(n_sys // 2):
            if local:
                a = int(rng.integers(n_sys))
                b = (a + int(rng.integers(1, 3))) % n_sys
            else:
                a, b = (int(x) for x in rng.choice(n_sys, 2, replace=False))
            kt.apply_2q(tab, a, b, int(rng.integers(11520)), LIN2, ANF2, IDENTITY_SYM2, ws, False)
        for q in np.flatnonzero(rng.random(n_sys) < rate):
            kt.measure_1q_phase_free_tracked(tab, n, ws, q, 2, anti, supp, ref_rows)
        for order in orders[:1] if jit_free else orders:
            for periodic in (True, False):
                d = ke.contiguous_distance_rows(tab, ws, order, ref_rows, len(refs), periodic)
                for cap in caps:
                    got = ke.contiguous_distance_rows_capped(
                        tab, ws, order, ref_rows, len(refs), periodic, cap
                    )
                    assert got == min(d, cap), (d, cap, periodic)
                part = order[1:]
                scan = ke.contiguous_distance_rows_scan(
                    tab, ws, part, ref_rows, len(refs), periodic
                )
                for cap in (2, 3, 5):
                    got = ke.contiguous_distance_rows_capped(
                        tab, ws, part, ref_rows, len(refs), periodic, cap
                    )
                    assert got == min(scan, cap), (scan, cap, periodic)
                seen.add("none" if d == 0 else "one" if d == 1 else "short" if d < 8 else "long")
    assert reached <= seen, seen
