"""
Ground-truth tests for single-qubit Clifford gate actions on Pauli operators.

These tests verify the ACTION of each gate (how it transforms specific Pauli
operators), not just that it produces some valid stabilizer group. Group-level
tests miss gate-swap bugs where two different Cliffords preserve the same
initial stabilizer group (e.g., Bell pairs) but act differently on arbitrary
Paulis.

References:
- Aaronson & Gottesman, "Improved Simulation of Stabilizer Circuits",
  Phys. Rev. A 70, 052328 (2004), Table I.
- Nielsen & Chuang, *Quantum Computation and Quantum Information*, §10.5.
"""
from __future__ import annotations

import numpy as np
import pytest
import stim

from sparsegf2 import SparseGF2


# PLT encoding: (x << 1) | z.  I = 0, Z = 1, X = 2, Y = 3.
I_PLT, Z_PLT, X_PLT, Y_PLT = 0, 1, 2, 3
PLT_LABEL = {0: "I", 1: "Z", 2: "X", 3: "Y"}


def _apply_and_read(gate_fn, initial_plt: int) -> int:
    """Put qubit 0 into a state whose row-0 system-part equals ``initial_plt``,
    apply ``gate_fn(sim, 0)``, read back row-0 system-part."""
    sim = SparseGF2(1)  # Bell pair: row 0 = X_0 (PLT=2), row 1 = Z_0 (PLT=1)
    # Rotate row 0 into the requested Pauli via Hadamards / CNOTs.
    # Easiest: use the known mapping I->no-op, X->no-op, Z->H, Y->HS (on the
    # generator representation of row 0 only).
    if initial_plt == X_PLT:
        pass
    elif initial_plt == Z_PLT:
        sim.apply_h(0)
    elif initial_plt == Y_PLT:
        # H then S sends X -> Z -> Z (wrong); we actually want row-0 = Y.
        # Easiest: apply S to row-0 directly. On initial Bell pair row 0 = X,
        # S sends X -> Y, so row 0 becomes Y.
        sim.apply_s(0)
    else:  # I_PLT is not achievable on a generator that already has support
        raise ValueError("cannot set initial to I: row must have support")
    # Sanity-check we set it up right.
    assert sim.plt[0, 0] == initial_plt, (
        f"setup failed: expected {PLT_LABEL[initial_plt]}, "
        f"got {PLT_LABEL[int(sim.plt[0, 0])]}"
    )
    gate_fn(sim, 0)
    return int(sim.plt[0, 0])


def _expected_action_H(p: int) -> int:
    # H: X <-> Z, Y -> -Y (phase-free: Y)
    return {X_PLT: Z_PLT, Z_PLT: X_PLT, Y_PLT: Y_PLT}[p]


def _expected_action_S(p: int) -> int:
    # S: X -> Y, Y -> X, Z -> Z (phase-free)
    return {X_PLT: Y_PLT, Y_PLT: X_PLT, Z_PLT: Z_PLT}[p]


def _expected_action_sqrt_x(p: int) -> int:
    # sqrt(X): X -> X, Z -> Y, Y -> Z (phase-free)
    return {X_PLT: X_PLT, Z_PLT: Y_PLT, Y_PLT: Z_PLT}[p]


@pytest.mark.parametrize("p_in", [X_PLT, Y_PLT, Z_PLT])
def test_apply_h_action(p_in):
    out = _apply_and_read(lambda s, q: s.apply_h(q), p_in)
    assert out == _expected_action_H(p_in), (
        f"H on {PLT_LABEL[p_in]}: got {PLT_LABEL[out]}, "
        f"expected {PLT_LABEL[_expected_action_H(p_in)]}"
    )


@pytest.mark.parametrize("p_in", [X_PLT, Y_PLT, Z_PLT])
def test_apply_s_action(p_in):
    out = _apply_and_read(lambda s, q: s.apply_s(q), p_in)
    assert out == _expected_action_S(p_in), (
        f"S on {PLT_LABEL[p_in]}: got {PLT_LABEL[out]}, "
        f"expected {PLT_LABEL[_expected_action_S(p_in)]} "
        f"(this would catch the S/sqrt(X) label-swap bug)"
    )


@pytest.mark.parametrize("p_in", [X_PLT, Y_PLT, Z_PLT])
def test_apply_sqrt_x_action(p_in):
    out = _apply_and_read(lambda s, q: s.apply_sqrt_x(q), p_in)
    assert out == _expected_action_sqrt_x(p_in), (
        f"sqrt(X) on {PLT_LABEL[p_in]}: got {PLT_LABEL[out]}, "
        f"expected {PLT_LABEL[_expected_action_sqrt_x(p_in)]}"
    )


@pytest.mark.parametrize("p_in", [X_PLT, Y_PLT, Z_PLT])
def test_apply_s_fast_matches_generic(p_in):
    """apply_s_fast (kernel path) must match apply_s (generic matrix path)."""
    out_gen = _apply_and_read(lambda s, q: s.apply_s(q), p_in)
    out_fast = _apply_and_read(lambda s, q: s.apply_s_fast(q), p_in)
    assert out_gen == out_fast, (
        f"apply_s_fast diverges from apply_s on {PLT_LABEL[p_in]}: "
        f"fast={PLT_LABEL[out_fast]}, generic={PLT_LABEL[out_gen]}"
    )


def test_s_squared_equals_z_via_stim():
    """S^2 = Z (up to global phase). Both produce a state with the same
    stabilizer group as applying Z directly."""
    n = 4

    sim_ours = SparseGF2(n)
    sim_ours.apply_s(0)
    sim_ours.apply_s(0)

    # Reference: apply Z -- but Z is not exposed on SparseGF2 directly; we use
    # Stim to compute the expected stabilizer group.
    stim_sim = stim.TableauSimulator()
    for i in range(n):
        stim_sim.h(i)
        stim_sim.cx(i, n + i)
    stim_sim.z(0)

    # Compare stabilizer groups via RREF
    m_ours = sim_ours.extract_sys_matrix()
    tab = stim_sim.current_inverse_tableau().inverse()
    m_stim = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    for r in range(n):
        for q in range(n):
            p_z = tab.z_output(r)[q]
            if p_z in (1, 2):
                m_stim[r, q] = 1
            if p_z in (2, 3):
                m_stim[r, n + q] = 1
            p_x = tab.x_output(r)[q]
            if p_x in (1, 2):
                m_stim[n + r, q] = 1
            if p_x in (2, 3):
                m_stim[n + r, n + q] = 1

    # RREF both; compare
    def rref(M):
        M = M.copy()
        R, C = M.shape
        r = 0
        for c in range(C):
            p = -1
            for i in range(r, R):
                if M[i, c]:
                    p = i
                    break
            if p < 0:
                continue
            M[[r, p]] = M[[p, r]]
            for i in range(R):
                if i != r and M[i, c]:
                    M[i] ^= M[r]
            r += 1
        return M[:r]

    assert np.array_equal(rref(m_ours), rref(m_stim))
