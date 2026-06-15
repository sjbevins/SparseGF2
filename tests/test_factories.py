"""factories and resets.

Verifies the ergonomic constructors and the X / Y reset shims.

Coverage
--------

* ``from_zero_state(n)`` matches ``SparseGF2(n)`` byte-for-byte.
* ``from_bell_purification(n_system)`` matches a hand-built
  ``H + CX`` Bell-pair tableau and matches Stim byte-for-byte.
* ``reset_x`` / ``reset_y`` leave the state with the correct stabilizer
  (Stim parity).
"""

from __future__ import annotations

import numpy as np
import pytest

stim = pytest.importorskip("stim")  # skipped when the optional stim extra is absent

from _stim_parity import assert_states_equal, fresh_stim, stim_symplectic
from sparsegf2 import (
    SparseGF2,
    from_bell_purification,
    from_zero_state,
)

stim = pytest.importorskip("stim")


# ----------------------------------------------------------------------
# Identity-of-construction sanity
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 4, 16])
def test_from_zero_state_matches_constructor(n):
    """from_zero_state(n) and SparseGF2(n) produce the same tableau."""
    a = from_zero_state(n)
    b = SparseGF2(n)
    assert a.n == b.n
    assert np.array_equal(a.to_symplectic(), b.to_symplectic())


# ----------------------------------------------------------------------
# Bell-pair purification
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n_system", [1, 2, 3, 5, 8])
def test_from_bell_purification_byte_equal_to_handbuilt(n_system):
    """The factory should match a manual H(i); CX(i, i+n_system) sequence
    byte-for-byte."""
    a = from_bell_purification(n_system)
    b = SparseGF2(2 * n_system)
    for i in range(n_system):
        b.apply_h(i)
        b.apply_cx(i, i + n_system)
    assert np.array_equal(a.to_symplectic(), b.to_symplectic())


@pytest.mark.parametrize("n_system", [1, 2, 3, 5])
def test_from_bell_purification_stim_parity(n_system):
    """Bell-pair tableau matches Stim byte-for-byte (pre-measurement)."""
    n = 2 * n_system
    ours = from_bell_purification(n_system)
    theirs = fresh_stim(n)
    for i in range(n_system):
        theirs.h(i)
        theirs.cx(i, i + n_system)
    assert np.array_equal(ours.to_symplectic(), stim_symplectic(theirs, n))


@pytest.mark.parametrize("n_system", [1, 3, 5])
def test_from_bell_purification_z_correlations_deterministic(n_system):
    """In a Bell pair |Φ⁺⟩, Z_a Z_{a+n_system} is in the stabilizer
    group, so measuring Z_a makes Z_{a+n_system} deterministic.

    This is the load-bearing physics check for the factory.
    """
    sim = from_bell_purification(n_system)
    rng = np.random.default_rng(0)
    for i in range(n_system):
        assert not sim.is_deterministic_z(i)
        assert not sim.is_deterministic_z(i + n_system)
        sim.measure_z(i, rng=rng)
        assert sim.is_deterministic_z(i + n_system)


# ----------------------------------------------------------------------
# reset_x / reset_y
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 5])
def test_reset_x_makes_x_stabilizer_deterministic(n):
    """After reset_x(q), measuring X_q must be deterministic (it is the
    new stabilizer of qubit q)."""
    rng = np.random.default_rng(7)
    sim = SparseGF2(n)
    # Hadamard everything to non-trivial state, then reset_x.
    for q in range(n):
        sim.apply_h(q)
    sim.apply_cx(0, 1)  # mix in some entanglement
    sim.reset_x(0)
    # measure_x conjugates by H, then measure_z. After reset_x(0), the
    # tableau has Z_0 as a stab (because the H-conjugation of M_Z
    # leaves Z_0 in the stab block), which is X_0 in the original basis.
    # Simplest check: measure_x(0) is deterministic.
    sim2 = SparseGF2(n)
    # Apply the same prefix to sim2 then ask if measure_x is deterministic
    # right after reset_x via a separate path. We just confirm here that
    # measure_x(0) doesn't perturb the state further.
    pre = sim.to_symplectic().copy()
    sim.measure_x(0, rng=rng)
    post = sim.to_symplectic()
    assert np.array_equal(pre, post), "reset_x then measure_x should leave the symplectic unchanged"
    _ = sim2  # quiet linter


@pytest.mark.parametrize("n", [2, 5])
def test_reset_y_makes_y_stabilizer_deterministic(n):
    """Same idempotency check as reset_x but in the Y basis."""
    rng = np.random.default_rng(11)
    sim = SparseGF2(n)
    for q in range(n):
        sim.apply_h(q)
    sim.apply_cz(0, 1)
    sim.reset_y(0)
    pre = sim.to_symplectic().copy()
    sim.measure_y(0, rng=rng)
    post = sim.to_symplectic()
    assert np.array_equal(pre, post), "reset_y then measure_y should leave the symplectic unchanged"


# ----------------------------------------------------------------------
# Stim parity: reset_x / reset_y match the corresponding Stim circuit
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 4, 6])
def test_reset_x_matches_stim_subspace(n):
    """Apply some gates, then reset_x on every qubit; the resulting
    stabilizer subspace must equal Stim's after the same circuit."""
    rng = np.random.default_rng(2026)
    ours = SparseGF2(n)
    theirs = fresh_stim(n)
    # Random gates
    for _ in range(2 * n):
        q = int(rng.integers(0, n))
        ours.apply_h(q)
        theirs.h(q)
    # Reset every qubit in X
    for q in range(n):
        ours.reset_x(q)
        theirs.do(stim.Circuit(f"RX {q}"))
    assert_states_equal(ours.to_symplectic(), stim_symplectic(theirs, n), n)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_reset_y_matches_stim_subspace(n):
    rng = np.random.default_rng(2027)
    ours = SparseGF2(n)
    theirs = fresh_stim(n)
    for _ in range(2 * n):
        q = int(rng.integers(0, n))
        ours.apply_h(q)
        theirs.h(q)
    for q in range(n):
        ours.reset_y(q)
        theirs.do(stim.Circuit(f"RY {q}"))
    assert_states_equal(ours.to_symplectic(), stim_symplectic(theirs, n), n)
