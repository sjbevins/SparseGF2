"""Regression test for apply_clifford_2q with track_inverse=True.

The previous implementation reused the forward symplectic matrix to update
the inverse tableau, which is correct only for GF(2) self-inverse gates
(CNOT, CZ, SWAP). A generic 2-qubit Clifford whose symplectic matrix is
not self-inverse would silently corrupt the inverse tableau.
"""
from __future__ import annotations

import numpy as np

from sparsegf2 import StabilizerTableau


def test_track_inverse_is_identity_after_gate_and_reverse_on_non_involutory():
    """Apply a non-GF(2)-self-inverse symplectic, then its inverse, and
    confirm the state returns to the initial tableau. If the inverse
    tableau is computed wrong for non-involutory gates the product
    would drift away from identity."""
    n = 4
    # A known non-self-inverse 4x4 symplectic: two CNOTs chained with a
    # permutation -- easier: just compose CNOT with a SWAP-free shift.
    # Concretely: CX followed by CX on different control/target pair, as
    # a single 4x4 gives a specific non-involutory symplectic.
    #
    # We construct the symplectic for "CX(0,1) then CX(1,0)", which is
    # NOT self-inverse at the GF(2) level. (The composition of two CXs
    # with swapped roles is an involution iff repeated.)
    #
    # Rather than hand-derive, sample an invertible GF(2) symplectic by
    # composing 3 CNOT-like matrices.
    CX_01 = np.array([
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
    ], dtype=np.uint8)
    CX_10 = np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ], dtype=np.uint8)

    S = (CX_01 @ CX_10) & 1  # non-involutory GF(2) symplectic

    def _gf2_inverse_4x4(M):
        A = (np.asarray(M) & 1).astype(np.uint8).copy()
        I = np.eye(4, dtype=np.uint8)
        aug = np.concatenate([A, I], axis=1)
        for col in range(4):
            pivot = -1
            for r in range(col, 4):
                if aug[r, col]:
                    pivot = r
                    break
            assert pivot >= 0, "matrix not invertible"
            if pivot != col:
                aug[[col, pivot]] = aug[[pivot, col]]
            for r in range(4):
                if r != col and aug[r, col]:
                    aug[r] ^= aug[col]
        return aug[:, 4:].astype(np.uint8)

    S_inv = _gf2_inverse_4x4(S)

    tab = StabilizerTableau.from_bell_pairs(n)
    initial = tab.to_symplectic().copy()

    tab.apply_clifford_2q(0, 1, S)
    tab.apply_clifford_2q(0, 1, S_inv)

    final = tab.to_symplectic()
    # After S then S_inv the stabilizer matrix must be identical to the
    # initial one.
    assert np.array_equal(initial, final), (
        "apply_clifford_2q + apply_clifford_2q(inverse) failed to return to "
        "the initial state for a non-involutory 4x4 symplectic"
    )
