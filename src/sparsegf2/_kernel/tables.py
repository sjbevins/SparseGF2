"""Gate tables consumed by the compiled kernels.

Every 1- and 2-qubit Clifford has a canonical index (see
:mod:`sparsegf2.clifford`): ``index = sym * 4**k + s`` where ``sym`` picks a
symplectic matrix ``M`` and bit ``j`` of ``s`` negates image ``j``. The kernels
apply such a gate to the bit-packed tableau in two parts:

**Linear part.** With the kernel variables in basis order
``v = (x_0, .., x_{k-1}, z_0, .., z_{k-1})`` (for 2 qubits: ``x_a, x_b, z_a, z_b``),
the new bits are ``v @ M (mod 2)``. ``LIN[k][sym, i, j]`` stores ``M[i, j]`` as a
64-bit mask (all ones or all zeros), so one AND/XOR handles 64 generators.

**Sign part.** A generator whose restriction to the gate's qubits is the
unsigned Pauli ``P_v`` picks up a sign flip ``f(v)`` from ``C P_v C^dagger``.
For the all-plus-signs Clifford of symplectic ``sym`` this is ``sign0[sym, v]``.
Negating image ``j`` flips the result exactly when ``v_j = 1``, so for index
``(sym, s)``::

    f(v) = sign0[sym, v] XOR parity(v AND s)

The kernels evaluate ``f`` on 64 generators at once from its algebraic normal
form (ANF): ``f(v) = XOR over monomials m of a_m * AND_{j in m} v_j``.
``ANF[k][sym]`` stores the coefficients of ``sign0`` as a bit mask (bit ``m`` =
``a_m``); the ``s`` part only adds the linear monomials, computed on the fly by
:func:`sparsegf2._kernel.tableau.linear_monomials`.

The tables are derived here from the symplectic group alone, without Stim, and
the tests check them against :meth:`Clifford.conjugation_table` for every index.
"""

from __future__ import annotations

import numpy as np

from sparsegf2.clifford import symplectic_group

__all__ = ["ANF1", "ANF2", "IDENTITY_SYM1", "IDENTITY_SYM2", "LIN1", "LIN2"]

_ALL = np.uint64(0xFFFFFFFFFFFFFFFF)


def _product_phase_rows(x1, z1, x2, z2) -> np.ndarray:
    """Row-wise power of i from multiplying unsigned Paulis (rows of bit arrays)."""
    g = np.where(
        (x1 == 1) & (z1 == 1),
        z2 - x2,
        np.where(x1 == 1, z2 * (2 * x2 - 1), np.where(z1 == 1, x2 * (1 - 2 * z2), 0)),
    )
    return g.sum(axis=1)


def _sign0_table(k: int) -> np.ndarray:
    """``sign0[sym, v]`` for every symplectic matrix and every unsigned input Pauli."""
    group = symplectic_group(k).astype(np.int64)  # (S, 2k, 2k)
    num_sym = group.shape[0]
    d = 2 * k
    table = np.zeros((num_sym, 1 << d), dtype=np.uint8)
    for v in range(1 << d):
        bits = [(v >> j) & 1 for j in range(d)]
        # P_v = i**(sum_q x_q z_q) * prod_q X_q**x_q Z_q**z_q   (Y = i X Z)
        phase = np.full(num_sym, sum(bits[q] * bits[k + q] for q in range(k)), dtype=np.int64)
        acc_x = np.zeros((num_sym, k), dtype=np.int64)
        acc_z = np.zeros((num_sym, k), dtype=np.int64)
        for q in range(k):
            for j in (q, k + q):  # X_q before Z_q
                if bits[j]:
                    img_x, img_z = group[:, j, :k], group[:, j, k:]
                    phase += _product_phase_rows(acc_x, acc_z, img_x, img_z)
                    acc_x ^= img_x
                    acc_z ^= img_z
        phase %= 4
        if np.any(phase % 2):  # pragma: no cover (would mean a non-Hermitian image)
            raise AssertionError("conjugated Hermitian Pauli came out non-Hermitian")
        table[:, v] = phase // 2
    return table


def _anf_bits(truth: np.ndarray) -> np.ndarray:
    """Moebius transform of truth tables (rows) into packed ANF coefficient masks."""
    a = truth.astype(np.int64).copy()
    size = a.shape[1]
    j = 1
    while j < size:
        for v in range(size):
            if v & j:
                a[:, v] ^= a[:, v ^ j]
        j <<= 1
    weights = np.left_shift(np.int64(1), np.arange(size, dtype=np.int64))
    return (a * weights).sum(axis=1).astype(np.int64)


def _lin_masks(k: int) -> np.ndarray:
    group = symplectic_group(k)
    return np.where(group.astype(bool), _ALL, np.uint64(0)).astype(np.uint64)


def _identity_index(k: int) -> int:
    eye = np.eye(2 * k, dtype=np.uint8)
    for i, m in enumerate(symplectic_group(k)):
        if np.array_equal(m, eye):
            return i
    raise AssertionError("identity missing from symplectic group")  # pragma: no cover


LIN1 = _lin_masks(1)  # (6, 2, 2) uint64
LIN2 = _lin_masks(2)  # (720, 4, 4) uint64
ANF1 = _anf_bits(_sign0_table(1))  # (6,) int64, 4 coefficient bits each
ANF2 = _anf_bits(_sign0_table(2))  # (720,) int64, 16 coefficient bits each
IDENTITY_SYM1 = _identity_index(1)
IDENTITY_SYM2 = _identity_index(2)

for _arr in (LIN1, LIN2, ANF1, ANF2):
    _arr.flags.writeable = False
del _arr
