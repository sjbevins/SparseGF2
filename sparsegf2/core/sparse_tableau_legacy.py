"""
Sparse stabilizer tableau with inverted indices for O(a_bar) gate application.

.. deprecated::
    This module is superseded by ``sparse_tableau.py`` which uses the PLT
    + position map + min-weight pivot design. Use ``SparseGF2`` instead.

Extends StabilizerTableau with per-qubit inverted indices that track which
generators have support on each qubit. Gate application only touches generators
with support on the gate's qubits — O(a_bar) instead of O(n).
"""

import warnings as _warnings
_warnings.warn(
    "sparsegf2.core.sparse_tableau_legacy is deprecated. "
    "Use sparsegf2.core.sparse_tableau.SparseGF2 instead.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np
from sparsegf2.core.packed import PackedBitMatrix
from sparsegf2.core.tableau import StabilizerTableau, _ISWAP_SYMPLECTIC

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# Pre-converted uint64 symplectic matrices for named gates
_S_CNOT_U64 = np.array([[1,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,1]], dtype=np.uint64)
_S_CZ_U64   = np.array([[1,0,0,1],[0,1,1,0],[0,0,1,0],[0,0,0,1]], dtype=np.uint64)
_S_SWAP_U64 = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.uint64)
_S_ISWAP_U64 = _ISWAP_SYMPLECTIC.astype(np.uint64)


@njit(cache=True)
def _sparse_apply_2q(x_data, z_data, q0, q1, S,
                     x_gen_q0, x_gen_q1, z_gen_q0, z_gen_q1, n_gen):
    """Apply 2-qubit Clifford only to active generators."""
    w0 = q0 >> 6
    b0 = np.uint64(q0 & 63)
    w1 = q1 >> 6
    b1 = np.uint64(q1 & 63)
    mask0 = np.uint64(1) << b0
    mask1 = np.uint64(1) << b1
    one = np.uint64(1)

    n_idx_words = (n_gen + 63) // 64
    for iw in range(n_idx_words):
        active = x_gen_q0[iw] | x_gen_q1[iw] | z_gen_q0[iw] | z_gen_q1[iw]
        if active == np.uint64(0):
            continue
        base = iw * 64
        word = active
        while word != np.uint64(0):
            # Count trailing zeros to find lowest set bit
            bit_pos = np.uint64(0)
            tmp = word
            while (tmp & one) == np.uint64(0):
                tmp >>= one
                bit_pos += one
            r = base + int(bit_pos)
            if r < n_gen:
                xq0 = (x_data[r, w0] >> b0) & one
                xq1 = (x_data[r, w1] >> b1) & one
                zq0 = (z_data[r, w0] >> b0) & one
                zq1 = (z_data[r, w1] >> b1) & one

                n0 = (xq0*S[0,0] ^ xq1*S[1,0] ^ zq0*S[2,0] ^ zq1*S[3,0]) & one
                n1 = (xq0*S[0,1] ^ xq1*S[1,1] ^ zq0*S[2,1] ^ zq1*S[3,1]) & one
                n2 = (xq0*S[0,2] ^ xq1*S[1,2] ^ zq0*S[2,2] ^ zq1*S[3,2]) & one
                n3 = (xq0*S[0,3] ^ xq1*S[1,3] ^ zq0*S[2,3] ^ zq1*S[3,3]) & one

                x_data[r, w0] = (x_data[r, w0] & ~mask0) | (n0 << b0)
                x_data[r, w1] = (x_data[r, w1] & ~mask1) | (n1 << b1)
                z_data[r, w0] = (z_data[r, w0] & ~mask0) | (n2 << b0)
                z_data[r, w1] = (z_data[r, w1] & ~mask1) | (n3 << b1)

                rw = r >> 6
                rb = np.uint64(r & 63)
                rmask = one << rb
                if xq0 != n0:
                    x_gen_q0[rw] ^= rmask
                if xq1 != n1:
                    x_gen_q1[rw] ^= rmask
                if zq0 != n2:
                    z_gen_q0[rw] ^= rmask
                if zq1 != n3:
                    z_gen_q1[rw] ^= rmask

            word &= word - one


@njit(cache=True)
def _sparse_measure_z(x_data, z_data, x_gen, z_gen, q, n_gen, n_qubits):
    """Full sparse measurement: find anticomm, XOR, replace, update indices."""
    w = q >> 6
    b = np.uint64(q & 63)
    one = np.uint64(1)
    qmask = one << b
    n_words = x_data.shape[1]
    n_idx_words = (n_gen + 63) // 64

    # Find first anticommuting generator via x_gen[q]
    pivot = -1
    for iw in range(n_idx_words):
        bits = x_gen[q, iw]
        if bits != np.uint64(0):
            tmp = bits
            bit_pos = np.uint64(0)
            while (tmp & one) == np.uint64(0):
                tmp >>= one
                bit_pos += one
            pivot = iw * 64 + int(bit_pos)
            break

    if pivot == -1 or pivot >= n_gen:
        return

    pw = pivot >> 6
    pb = np.uint64(pivot & 63)
    pmask = one << pb

    # Find all OTHER anticommuting generators and XOR pivot into them
    for iw in range(n_idx_words):
        bits = x_gen[q, iw]
        if iw == pw:
            bits &= ~pmask  # exclude pivot
        if bits == np.uint64(0):
            continue
        word = bits
        while word != np.uint64(0):
            bit_pos = np.uint64(0)
            tmp = word
            while (tmp & one) == np.uint64(0):
                tmp >>= one
                bit_pos += one
            r = iw * 64 + int(bit_pos)
            if r < n_gen and r != pivot:
                rw = r >> 6
                rb = np.uint64(r & 63)
                rmask = one << rb

                # XOR pivot row into row r
                for j in range(n_words):
                    x_data[r, j] ^= x_data[pivot, j]
                    z_data[r, j] ^= z_data[pivot, j]

                # Update indices: for each qubit where pivot has support, toggle r
                for jww in range(n_words):
                    px = x_data[pivot, jww]
                    pz = z_data[pivot, jww]
                    combined = px | pz
                    if combined == np.uint64(0):
                        continue
                    bword = combined
                    while bword != np.uint64(0):
                        bp = np.uint64(0)
                        btmp = bword
                        while (btmp & one) == np.uint64(0):
                            btmp >>= one
                            bp += one
                        qq = jww * 64 + int(bp)
                        if qq < n_qubits:
                            if (px >> bp) & one:
                                x_gen[qq, rw] ^= rmask
                            if (pz >> bp) & one:
                                z_gen[qq, rw] ^= rmask
                        bword &= bword - one

            word &= word - one

    # Clear all index entries for pivot
    for qq in range(n_qubits):
        x_gen[qq, pw] &= ~pmask
        z_gen[qq, pw] &= ~pmask

    # Replace pivot with Z_q
    for j in range(n_words):
        x_data[pivot, j] = np.uint64(0)
        z_data[pivot, j] = np.uint64(0)
    z_data[pivot, w] = qmask

    # Set index for Z_q
    z_gen[q, pw] |= pmask


class SparseStabilizerTableau(StabilizerTableau):
    """Stabilizer tableau with inverted indices for O(a_bar) gate application."""

    def __init__(self, n: int, track_inverse: bool = False):
        super().__init__(n, track_inverse=track_inverse)
        self._n_idx_words = (n + 63) // 64
        self.x_gen = np.zeros((n, self._n_idx_words), dtype=np.uint64)
        self.z_gen = np.zeros((n, self._n_idx_words), dtype=np.uint64)
        for r in range(n):
            rw = r >> 6
            rb = np.uint64(r & 63)
            self.z_gen[r, rw] |= np.uint64(1) << rb

    def _rebuild_indices(self):
        """Rebuild inverted indices from tableau (slow, for init only)."""
        self.x_gen[:] = 0
        self.z_gen[:] = 0
        for q in range(self.n):
            x_col = self.x.get_column_bits(q)
            z_col = self.z.get_column_bits(q)
            for r in range(self.n):
                rw = r >> 6
                rb = np.uint64(r & 63)
                rmask = np.uint64(1) << rb
                if x_col[r]:
                    self.x_gen[q, rw] |= rmask
                if z_col[r]:
                    self.z_gen[q, rw] |= rmask

    @classmethod
    def from_zero_state(cls, n: int, track_inverse: bool = False):
        return cls(n, track_inverse=track_inverse)

    @classmethod
    def from_bell_pairs(cls, n_system: int, track_inverse: bool = False):
        n = n_system
        tab = cls(2 * n, track_inverse=track_inverse)
        tab.x = PackedBitMatrix.zeros(2 * n, 2 * n)
        tab.z = PackedBitMatrix.zeros(2 * n, 2 * n)
        for i in range(n):
            tab.x.set_bit(i, i, 1)
            tab.x.set_bit(i, n + i, 1)
            tab.z.set_bit(n + i, i, 1)
            tab.z.set_bit(n + i, n + i, 1)
        tab._n_idx_words = (2 * n + 63) // 64
        tab.x_gen = np.zeros((2 * n, tab._n_idx_words), dtype=np.uint64)
        tab.z_gen = np.zeros((2 * n, tab._n_idx_words), dtype=np.uint64)
        tab._rebuild_indices()
        return tab

    # ------------------------------------------------------------------
    # Fast gate methods — bypass apply_clifford_2q Python overhead
    # ------------------------------------------------------------------

    def _sparse_2q(self, q0, q1, S_u64):
        """Direct Numba sparse gate call. No validation, no astype, no dispatch."""
        _sparse_apply_2q(
            self.x.data, self.z.data, q0, q1, S_u64,
            self.x_gen[q0], self.x_gen[q1],
            self.z_gen[q0], self.z_gen[q1], self.n)

    def cnot(self, control: int, target: int):
        self._check_two_qubits(control, target)
        self._sparse_2q(control, target, _S_CNOT_U64)
        if self.track_inverse:
            self._apply_2q_to_pair(self.inv_x, self.inv_z, control, target,
                                   StabilizerTableau._S_CNOT)

    def cz(self, q0: int, q1: int):
        self._check_two_qubits(q0, q1)
        self._sparse_2q(q0, q1, _S_CZ_U64)
        if self.track_inverse:
            self._apply_2q_to_pair(self.inv_x, self.inv_z, q0, q1,
                                   StabilizerTableau._S_CZ)

    def swap(self, q0: int, q1: int):
        self._check_two_qubits(q0, q1)
        self._sparse_2q(q0, q1, _S_SWAP_U64)
        if self.track_inverse:
            self._apply_2q_to_pair(self.inv_x, self.inv_z, q0, q1,
                                   StabilizerTableau._S_SWAP)

    def iswap(self, q0: int, q1: int):
        self._check_two_qubits(q0, q1)
        self._sparse_2q(q0, q1, _S_ISWAP_U64)
        if self.track_inverse:
            self._apply_2q_to_pair(self.inv_x, self.inv_z, q0, q1,
                                   _ISWAP_SYMPLECTIC)

    def apply_clifford_2q(self, q0: int, q1: int, symplectic_4x4: np.ndarray):
        """Sparse arbitrary 2Q Clifford."""
        self._check_two_qubits(q0, q1)
        S = symplectic_4x4.astype(np.uint64)
        _sparse_apply_2q(
            self.x.data, self.z.data, q0, q1, S,
            self.x_gen[q0], self.x_gen[q1],
            self.z_gen[q0], self.z_gen[q1], self.n)
        if self.track_inverse:
            self._apply_2q_to_pair(self.inv_x, self.inv_z, q0, q1, symplectic_4x4)

    def h(self, q: int):
        self._check_qubit(q)
        self._apply_1q_to_pair(self.x, self.z, q, "H")
        self.x_gen[q], self.z_gen[q] = self.z_gen[q].copy(), self.x_gen[q].copy()
        if self.track_inverse:
            self._apply_1q_to_pair(self.inv_x, self.inv_z, q, "H")

    def s(self, q: int):
        self._check_qubit(q)
        self._apply_1q_to_pair(self.x, self.z, q, "S")
        self.z_gen[q] ^= self.x_gen[q]
        if self.track_inverse:
            self._apply_1q_to_pair(self.inv_x, self.inv_z, q, "S")

    def s_dag(self, q: int):
        self.s(q)

    def sqrt_x(self, q: int):
        self._check_qubit(q)
        self._apply_1q_to_pair(self.x, self.z, q, "SQRT_X")
        self.x_gen[q] ^= self.z_gen[q]
        if self.track_inverse:
            self._apply_1q_to_pair(self.inv_x, self.inv_z, q, "SQRT_X")

    def sqrt_x_dag(self, q: int):
        self.sqrt_x(q)

    # ------------------------------------------------------------------
    # Measurement — fully Numba-compiled with incremental index updates
    # ------------------------------------------------------------------

    def measure_z(self, q: int):
        self._check_qubit(q)
        if HAS_NUMBA:
            if self.track_inverse:
                raise NotImplementedError(
                    "SparseStabilizerTableau does not support track_inverse with "
                    "sparse measurement. Use StabilizerTableau or set track_inverse=False.")
            _sparse_measure_z(
                self.x.data, self.z.data, self.x_gen, self.z_gen,
                q, self.n, self.n)
        else:
            super().measure_z(q)
            self._rebuild_indices()

    def copy(self):
        tab = SparseStabilizerTableau.__new__(SparseStabilizerTableau)
        tab.n = self.n
        tab.track_inverse = self.track_inverse
        tab.x = self.x.copy()
        tab.z = self.z.copy()
        tab.inv_x = self.inv_x.copy() if self.inv_x is not None else None
        tab.inv_z = self.inv_z.copy() if self.inv_z is not None else None
        tab._n_idx_words = self._n_idx_words
        tab.x_gen = self.x_gen.copy()
        tab.z_gen = self.z_gen.copy()
        return tab
