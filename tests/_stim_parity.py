"""Stim parity helpers.

Pure test-time utilities for cross-checking ``SparseGF2`` against
``stim.TableauSimulator``. Importing this module
requires ``stim``; it must never be imported from runtime code.

The leading underscore on the filename keeps pytest from collecting it
as a test module.
"""

from __future__ import annotations

import numpy as np
import stim


def stim_symplectic(sim: stim.TableauSimulator, n: int) -> np.ndarray:
    """Extract the full ``(2n, 2n)`` symplectic ``[X | Z]`` from Stim.

    Built so the output matches :meth:`sparsegf2.SparseGF2.to_symplectic`
    exactly:

    * Rows ``0..n-1`` are destabilizers: ``forward_tableau.x_output(i)``.
    * Rows ``n..2n-1`` are stabilizers: ``forward_tableau.z_output(i)``.
    * Columns ``0..n-1`` carry the x-bits; columns ``n..2n-1`` carry
      the z-bits.

    Parameters
    ----------
    sim
        A ``stim.TableauSimulator`` representing the state to inspect.
    n
        Number of qubits the simulator should be acting on. Must equal
        the length of the simulator's current tableau.

    Returns
    -------
    ndarray
        A ``(2n, 2n)`` uint8 matrix.
    """
    fwd = sim.current_inverse_tableau().inverse(unsigned=True)
    if len(fwd) != n:
        raise ValueError(f"Stim tableau has {len(fwd)} qubits but n={n} was requested")

    mat = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    for i in range(n):
        x_xs, x_zs = fwd.x_output(i).to_numpy()  # destabilizer i
        z_xs, z_zs = fwd.z_output(i).to_numpy()  # stabilizer i
        mat[i, :n] = x_xs.astype(np.uint8)
        mat[i, n:] = x_zs.astype(np.uint8)
        mat[n + i, :n] = z_xs.astype(np.uint8)
        mat[n + i, n:] = z_zs.astype(np.uint8)
    return mat


def fresh_stim(n: int) -> stim.TableauSimulator:
    """Return a Stim simulator initialised to |0^n⟩."""
    sim = stim.TableauSimulator()
    sim.set_num_qubits(n)
    return sim


def _gf2_rref(A: np.ndarray) -> np.ndarray:
    """Reduce ``A`` to GF(2) row-reduced echelon form (in place).

    Zero rows are pushed to the bottom and the leading pivot column of
    each non-zero row is unique. Suitable for comparing the row span of
    two F_2 matrices via ``np.array_equal``.
    """
    A = (A.astype(np.uint8) & 1).copy()
    n_rows, n_cols = A.shape
    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        r_pivot = -1
        for r in range(pivot_row, n_rows):
            if A[r, col]:
                r_pivot = r
                break
        if r_pivot < 0:
            continue
        if r_pivot != pivot_row:
            A[[pivot_row, r_pivot]] = A[[r_pivot, pivot_row]]
        for r in range(n_rows):
            if r != pivot_row and A[r, col]:
                A[r] ^= A[pivot_row]
        pivot_row += 1
    return A


def stab_subspace_rref(M: np.ndarray, n: int) -> np.ndarray:
    """Row-reduce the stabilizer block (rows ``n..2n-1``) of ``[X|Z]``.

    Two valid stabilizer tableaus describe the same phase-free state iff
    their stabilizer-block RREFs are equal. Use this, not byte-for-byte
    comparison of the raw symplectic, whenever a measurement could have
    been resolved with different (but equally valid) pivot choices.
    """
    return _gf2_rref(M[n:])


def assert_states_equal(M_ours: np.ndarray, M_stim: np.ndarray, n: int) -> None:
    """Assert two symplectic tableaus describe the same phase-free state.

    Compares stabilizer subspaces in GF(2) RREF. Raises ``AssertionError``
    with a side-by-side diff on mismatch.
    """
    rref_ours = stab_subspace_rref(M_ours, n)
    rref_stim = stab_subspace_rref(M_stim, n)
    if not np.array_equal(rref_ours, rref_stim):
        raise AssertionError(
            f"stabilizer subspaces differ\nours RREF:\n{rref_ours}\nstim RREF:\n{rref_stim}"
        )


def stim_tableau_to_symplectic(tab: stim.Tableau) -> np.ndarray:
    """Convert a ``stim.Tableau`` to its ``(2n, 2n)`` symplectic projection.

    Phase-free: the sign bits are discarded. The layout matches
    :meth:`sparsegf2.SparseGF2.to_symplectic`:

    * rows ``0..n-1`` are ``C X_i C^†`` (destabilizers),
    * rows ``n..2n-1`` are ``C Z_i C^†`` (stabilizers),
    * columns ``0..n-1`` carry x-bits, columns ``n..2n-1`` carry z-bits.

    Used by the symplectic-group tests to verify that the native ``Sp(2n, F_2)`` enumeration
    matches Stim's set of Cliffords modulo signs.
    """
    xs2xs, xs2zs, zs2xs, zs2zs, *_signs = tab.to_numpy()
    n = xs2xs.shape[0]
    M = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    M[:n, :n] = xs2xs.astype(np.uint8)
    M[:n, n:] = xs2zs.astype(np.uint8)
    M[n:, :n] = zs2xs.astype(np.uint8)
    M[n:, n:] = zs2zs.astype(np.uint8)
    return M


def symp_to_stim_tableau(S: np.ndarray) -> stim.Tableau:
    """Build a stim.Tableau on 2 qubits whose symplectic part equals ``S``.

    Stim requires explicit signs; we always pick +1 because the simulator is
    phase-free, so its output depends only on the symplectic part (any
    consistent sign choice produces the same Stim symplectic). ``S`` is the
    4x4 GF(2) symplectic in the ``(X_qi, X_qj, Z_qi, Z_qj)`` row-vector basis.
    """
    rows = [S[i] for i in range(4)]

    def to_ps(row):
        return stim.PauliString.from_numpy(xs=row[:2].astype(bool), zs=row[2:].astype(bool))

    return stim.Tableau.from_conjugated_generators(
        xs=[to_ps(rows[0]), to_ps(rows[1])],
        zs=[to_ps(rows[2]), to_ps(rows[3])],
    )
