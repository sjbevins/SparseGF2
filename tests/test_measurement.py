"""Z-basis measurement parity vs Stim.

Verifies that :meth:`SparseGF2.measure_z` (and its X/Y conjugates)
reproduces ``stim.TableauSimulator`` for:

1. ``|0^n⟩`` baseline, where every ``Z_q`` measurement is deterministic.
2. ``H``-prepared ``|+^n⟩``, where every ``Z_q`` measurement is random.
3. Bell-pair correlations, where measuring the first qubit makes the second
   deterministic.
4. Long mixed gate + measurement circuits, step-by-step parity at every
   measurement (the headline MIPT-style workload).
5. ``measure_x`` / ``measure_y`` parity via the H- and SH- conjugations.
6. Data-structure invariants after long mixed sequences.
7. Repeated ``measure_z(q)`` collapses to the deterministic branch
   (idempotent symplectic).
8. ``is_deterministic_z`` agrees with Stim's ``peek_z`` non-zero check.

Parity contract: the simple tests (1-3) match Stim byte-for-byte at the
symplectic level, because the measurement has at most one anticommuting
stabilizer, so the pivot choice is forced. The multi-anticommuter tests
(4, 5) compare *stabilizer subspaces* via GF(2) RREF:
a different but equally valid pivot choice yields a different
destabilizer representation of the same physical state, so byte-for-byte
comparison would be over-strict.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2 import SparseGF2
from sparsegf2.core.symplectic import random_symplectic_2q

stim = pytest.importorskip("stim")

from _stim_parity import assert_states_equal, fresh_stim, stim_symplectic
from _stim_parity import symp_to_stim_tableau as _symp_to_stim_tableau  # noqa: E402

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _check_indices_consistent(sim: SparseGF2) -> None:
    n, N = sim.n, sim.N
    for q in range(n):
        expected_inv = {int(r) for r in range(N) if int(sim.plt[r, q]) != 0}
        actual_inv = {int(sim.inv[q, k]) for k in range(int(sim.inv_len[q]))}
        assert expected_inv == actual_inv, f"inv[{q}] inconsistent"

        expected_invx = {int(r) for r in range(N) if (int(sim.plt[r, q]) >> 1) & 1}
        actual_invx = {int(sim.inv_x[q, k]) for k in range(int(sim.inv_x_len[q]))}
        assert expected_invx == actual_invx, f"inv_x[{q}] inconsistent"

    for r in range(N):
        expected_supp = {int(q) for q in range(n) if int(sim.plt[r, q]) != 0}
        actual_supp = {int(sim.supp_q[r, k]) for k in range(int(sim.supp_len[r]))}
        assert expected_supp == actual_supp, f"supp_q[{r}] inconsistent"


# ----------------------------------------------------------------------
# 1. |0^n⟩: every Z_q measurement is deterministic
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_measure_zero_state_deterministic(n):
    ours = SparseGF2(n)
    theirs = fresh_stim(n)
    for q in range(n):
        assert ours.is_deterministic_z(q)
        # Stim's peek_z returns +1 / -1 / 0; non-zero ⇒ deterministic.
        assert theirs.peek_z(q) != 0
        ours.measure_z(q)
        theirs.measure(q)
        # Deterministic measurement leaves the tableau unchanged for both;
        # so byte-equality holds. Use the stabilizer-subspace contract
        # (assert_states_equal) to be future-proof against pivot-choice
        # divergences if any test below introduces non-deterministic paths.
        assert_states_equal(ours.to_symplectic(), stim_symplectic(theirs, n), n)


# ----------------------------------------------------------------------
# 2. |+^n⟩: every Z_q measurement is random
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_measure_plus_state_random(n):
    ours = SparseGF2(n)
    theirs = fresh_stim(n)
    for q in range(n):
        ours.apply_h(q)
        theirs.h(q)
    for q in range(n):
        assert not ours.is_deterministic_z(q)
        assert theirs.peek_z(q) == 0
        ours.measure_z(q)
        theirs.measure(q)
        # Each measurement has exactly one anticommuting stabilizer
        # (the freshly-Hadamarded X_q), so the pivot is forced and the
        # raw tableaus would match, but we use the subspace contract
        # for robustness against pivot-heuristic drift.
        assert_states_equal(ours.to_symplectic(), stim_symplectic(theirs, n), n)


# ----------------------------------------------------------------------
# 3. Bell-pair correlations: measure first ⇒ second becomes deterministic
# ----------------------------------------------------------------------


def test_bell_pair_measurement_correlation():
    ours = SparseGF2(2)
    theirs = fresh_stim(2)
    ours.apply_h(0)
    theirs.h(0)
    ours.apply_cx(0, 1)
    theirs.cx(0, 1)

    assert not ours.is_deterministic_z(0)
    assert not ours.is_deterministic_z(1)
    assert theirs.peek_z(0) == 0
    assert theirs.peek_z(1) == 0

    ours.measure_z(0)
    theirs.measure(0)
    assert_states_equal(ours.to_symplectic(), stim_symplectic(theirs, 2), 2)
    assert ours.is_deterministic_z(1)
    assert theirs.peek_z(1) != 0


# ----------------------------------------------------------------------
# 4. Mixed random gate + measurement circuit: step-by-step parity
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, n_steps, p_meas, seed",
    [
        (4, 200, 0.10, 0),
        (4, 200, 0.30, 1),
        (8, 400, 0.10, 2),
        (8, 400, 0.30, 3),
        (16, 600, 0.20, 4),
        (32, 400, 0.20, 5),
    ],
)
def test_mixed_circuit_with_measurements(n, n_steps, p_meas, seed):
    rng = np.random.default_rng(seed)
    ours = SparseGF2(n)
    theirs = fresh_stim(n)
    gate_1q = [
        ("apply_h", lambda s, q: s.h(q)),
        ("apply_s", lambda s, q: s.s(q)),
        ("apply_sqrt_x", lambda s, q: s.sqrt_x(q)),
    ]

    for step in range(n_steps):
        r = rng.random()
        if r < p_meas:
            q = int(rng.integers(0, n))
            det_ours = ours.is_deterministic_z(q)
            det_stim = theirs.peek_z(q) != 0
            assert det_ours == det_stim, (
                f"step {step}: determinism mismatch on q={q} (ours={det_ours}, stim={det_stim})"
            )
            ours.measure_z(q, rng=rng)
            theirs.measure(q)
        elif r < p_meas + 0.45 * (1 - p_meas):
            gname, stim_fn = gate_1q[int(rng.integers(0, 3))]
            q = int(rng.integers(0, n))
            getattr(ours, gname)(q)
            stim_fn(theirs, q)
        else:
            qi, qj = rng.choice(n, size=2, replace=False).tolist()
            S = random_symplectic_2q(rng)
            ours.apply_gate_2q(int(qi), int(qj), S)
            t = _symp_to_stim_tableau(S)
            theirs.do_tableau(t, [int(qi), int(qj)])

        try:
            assert_states_equal(ours.to_symplectic(), stim_symplectic(theirs, n), n)
        except AssertionError as e:
            pytest.fail(f"diverged at step {step} (n={n}, p={p_meas}, seed={seed}): {e}")


# ----------------------------------------------------------------------
# 5. measure_x / measure_y parity via conjugation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 4, 8])
def test_measure_x_parity(n):
    rng = np.random.default_rng(101)
    ours = SparseGF2(n)
    theirs = fresh_stim(n)
    # Prepare a generic state with a few random 2q gates.
    for _ in range(2 * n):
        qi, qj = rng.choice(n, size=2, replace=False).tolist()
        S = random_symplectic_2q(rng)
        ours.apply_gate_2q(int(qi), int(qj), S)
        theirs.do_tableau(_symp_to_stim_tableau(S), [int(qi), int(qj)])
    for q in range(n):
        ours.measure_x(q, rng=rng)
        theirs.do(stim.Circuit(f"MX {q}"))
        assert_states_equal(ours.to_symplectic(), stim_symplectic(theirs, n), n)


@pytest.mark.parametrize("n", [2, 4, 8])
def test_measure_y_parity(n):
    rng = np.random.default_rng(202)
    ours = SparseGF2(n)
    theirs = fresh_stim(n)
    for _ in range(2 * n):
        qi, qj = rng.choice(n, size=2, replace=False).tolist()
        S = random_symplectic_2q(rng)
        ours.apply_gate_2q(int(qi), int(qj), S)
        theirs.do_tableau(_symp_to_stim_tableau(S), [int(qi), int(qj)])
    for q in range(n):
        ours.measure_y(q, rng=rng)
        theirs.do(stim.Circuit(f"MY {q}"))
        assert_states_equal(ours.to_symplectic(), stim_symplectic(theirs, n), n)


# ----------------------------------------------------------------------
# 6. Data-structure invariants after long mixed sequences
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n, seed", [(6, 11), (12, 22), (24, 33)])
def test_indices_consistent_after_measurements(n, seed):
    rng = np.random.default_rng(seed)
    sim = SparseGF2(n)
    for _ in range(400):
        r = rng.random()
        if r < 0.2:
            sim.measure_z(int(rng.integers(0, n)), rng=rng)
        elif r < 0.6:
            g = int(rng.integers(0, 3))
            q = int(rng.integers(0, n))
            (sim.apply_h, sim.apply_s, sim.apply_sqrt_x)[g](q)
        else:
            qi, qj = rng.choice(n, size=2, replace=False).tolist()
            S = random_symplectic_2q(rng)
            sim.apply_gate_2q(int(qi), int(qj), S)
    _check_indices_consistent(sim)


# ----------------------------------------------------------------------
# 7. Repeated measurement is idempotent at the symplectic level
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 6])
def test_repeated_measure_z_idempotent(n):
    rng = np.random.default_rng(7)
    sim = SparseGF2(n)
    for q in range(n):
        sim.apply_h(q)
    for q in range(n):
        sim.measure_z(q, rng=rng)
    first = sim.to_symplectic().copy()
    # Re-measure each qubit; the projection is already onto Z_q, so the
    # symplectic should not change.
    for q in range(n):
        assert sim.is_deterministic_z(q)
        sim.measure_z(q, rng=rng)
    second = sim.to_symplectic()
    assert np.array_equal(first, second), "Re-measuring after collapse should be a no-op"


# ----------------------------------------------------------------------
# 8. is_deterministic_z agrees with Stim across long random circuits
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n, seed", [(6, 100), (12, 101), (24, 102)])
def test_is_deterministic_matches_stim(n, seed):
    rng = np.random.default_rng(seed)
    ours = SparseGF2(n)
    theirs = fresh_stim(n)
    for _ in range(200):
        r = rng.random()
        if r < 0.15:
            q = int(rng.integers(0, n))
            assert ours.is_deterministic_z(q) == (theirs.peek_z(q) != 0)
            ours.measure_z(q, rng=rng)
            theirs.measure(q)
        elif r < 0.55:
            g = int(rng.integers(0, 3))
            q = int(rng.integers(0, n))
            (ours.apply_h, ours.apply_s, ours.apply_sqrt_x)[g](q)
            (theirs.h, theirs.s, theirs.sqrt_x)[g](q)
        else:
            qi, qj = rng.choice(n, size=2, replace=False).tolist()
            S = random_symplectic_2q(rng)
            ours.apply_gate_2q(int(qi), int(qj), S)
            theirs.do_tableau(_symp_to_stim_tableau(S), [int(qi), int(qj)])
    # Final sweep: every qubit, on the final tableau.
    for q in range(n):
        assert ours.is_deterministic_z(q) == (theirs.peek_z(q) != 0)


# ----------------------------------------------------------------------
# 9. measure_z honors check_qubit
# ----------------------------------------------------------------------


def test_measure_z_boundary_errors():
    sim = SparseGF2(3)
    with pytest.raises(IndexError):
        sim.measure_z(-1)
    with pytest.raises(IndexError):
        sim.measure_z(3)
