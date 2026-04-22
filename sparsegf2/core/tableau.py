"""
Stabilizer tableau simulator over GF(2) with bit-packed uint64 storage.

Uses PackedBitMatrix for O(n/64) row operations. Each XOR on a generator
row processes 64 qubits at once instead of one, giving ~64x density
improvement over uint8-per-element storage.

Phase-free representation
-------------------------
This simulator tracks only the GF(2) symplectic part of the stabilizer
tableau (no phase bits). Gates like S and S_dag have identical GF(2)
effects. This is correct for rank, entropy, code distance, and weight
spectrum but not for observables that depend on generator signs.

Symplectic representation
-------------------------
Each generator row is a Pauli string encoded as two bit-vectors (x, z):
    x[r,q]=0, z[r,q]=0 -> I    x[r,q]=1, z[r,q]=0 -> X
    x[r,q]=1, z[r,q]=1 -> Y    x[r,q]=0, z[r,q]=1 -> Z

Gate application uses 4x4 GF(2) symplectic matrices. Measurement uses
the CHP algorithm (Aaronson & Gottesman 2004).
"""

import numpy as np
from sparsegf2.core.packed import PackedBitMatrix, packed_stabilizer_groups_equal
from sparsegf2.core.numba_kernels import (
    HAS_NUMBA, apply_clifford_2q_packed, measure_z_packed, gf2_rank_packed,
)

# iSWAP symplectic matrix, verified against stim.Tableau.from_named_gate("ISWAP")
_ISWAP_SYMPLECTIC = np.array([
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=np.uint8)


class StabilizerTableau:
    """GF(2) stabilizer tableau with bit-packed uint64 storage.

    Parameters
    ----------
    n : int
        Number of qubits (>= 0).
    """

    def __init__(self, n: int, track_inverse: bool = False):
        if n < 0:
            raise ValueError(f"Number of qubits must be non-negative, got {n}")
        self.n = n
        self.track_inverse = track_inverse
        # Forward tableau: x[r,q]=1 if generator r has X or Y on qubit q
        self.x = PackedBitMatrix.zeros(n, n)
        # z[r,q]=1 if generator r has Y or Z on qubit q
        # Initialize to |0...0>: stabilizers Z_0, ..., Z_{n-1}
        self.z = PackedBitMatrix.identity(n)
        # Inverse tableau (optional): inv * fwd^T = I over GF(2)
        # Updated on every gate so measurement can use O(1) pivot lookup.
        if track_inverse:
            self.inv_x = PackedBitMatrix.zeros(n, n)
            self.inv_z = PackedBitMatrix.identity(n)
        else:
            self.inv_x = None
            self.inv_z = None

    def _check_qubit(self, q: int):
        if not (0 <= q < self.n):
            raise IndexError(f"Qubit index {q} out of range [0, {self.n})")

    def _check_two_qubits(self, q0: int, q1: int):
        self._check_qubit(q0)
        self._check_qubit(q1)
        if q0 == q1:
            raise ValueError(
                f"Two-qubit gate requires distinct qubits, got q0={q0}, q1={q1}")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    @classmethod
    def from_zero_state(cls, n: int, track_inverse: bool = False) -> "StabilizerTableau":
        """Create |0...0>. Stabilizers: Z_0, ..., Z_{n-1}."""
        return cls(n, track_inverse=track_inverse)

    @classmethod
    def from_bell_pairs(cls, n_system: int, track_inverse: bool = False) -> "StabilizerTableau":
        """Create 2n-qubit Bell pair state.

        System qubits: 0..n-1, reference qubits: n..2n-1.
        Generators: X_i X_{n+i} (rows 0..n-1) and Z_i Z_{n+i} (rows n..2n-1).
        """
        n = n_system
        tab = cls(2 * n, track_inverse=track_inverse)
        tab.x = PackedBitMatrix.zeros(2 * n, 2 * n)
        tab.z = PackedBitMatrix.zeros(2 * n, 2 * n)
        for i in range(n):
            tab.x.set_bit(i, i, 1)          # X on system qubit i
            tab.x.set_bit(i, n + i, 1)      # X on reference qubit n+i
            tab.z.set_bit(n + i, i, 1)      # Z on system qubit i
            tab.z.set_bit(n + i, n + i, 1)  # Z on reference qubit n+i
        return tab

    # ------------------------------------------------------------------
    # Single-qubit Clifford gates (phase-free)
    # ------------------------------------------------------------------

    def _apply_1q_to_pair(self, x_mat, z_mat, q, gate):
        """Apply a single-qubit gate to columns of a (x, z) matrix pair."""
        if gate == "H":
            xc = x_mat.get_column_bits(q).copy()
            zc = z_mat.get_column_bits(q).copy()
            x_mat.set_column_bits(q, zc)
            z_mat.set_column_bits(q, xc)
        elif gate == "S":
            xc = x_mat.get_column_bits(q)
            zc = z_mat.get_column_bits(q)
            z_mat.set_column_bits(q, zc ^ xc)
        elif gate == "SQRT_X":
            xc = x_mat.get_column_bits(q)
            zc = z_mat.get_column_bits(q)
            x_mat.set_column_bits(q, xc ^ zc)

    def _apply_2q_to_pair(self, x_mat, z_mat, q0, q1, S):
        """Apply a 2-qubit Clifford to columns of a (x, z) matrix pair."""
        if HAS_NUMBA:
            apply_clifford_2q_packed(x_mat.data, z_mat.data, q0, q1,
                                     S.astype(np.uint64))
        else:
            b0 = x_mat.get_column_bits(q0)
            b1 = x_mat.get_column_bits(q1)
            b2 = z_mat.get_column_bits(q0)
            b3 = z_mat.get_column_bits(q1)
            bits = np.stack([b0, b1, b2, b3], axis=1).astype(np.int32)
            out = (bits @ S.astype(np.int32)).astype(np.uint8) & 1
            x_mat.set_column_bits(q0, out[:, 0])
            x_mat.set_column_bits(q1, out[:, 1])
            z_mat.set_column_bits(q0, out[:, 2])
            z_mat.set_column_bits(q1, out[:, 3])

    def h(self, q: int):
        """Hadamard: X <-> Z."""
        self._check_qubit(q)
        self._apply_1q_to_pair(self.x, self.z, q, "H")
        if self.track_inverse:
            self._apply_1q_to_pair(self.inv_x, self.inv_z, q, "H")

    def s(self, q: int):
        """Phase gate S: X -> Y (XZ), Z -> Z. (Phase-free: identical to S_dag.)"""
        self._check_qubit(q)
        self._apply_1q_to_pair(self.x, self.z, q, "S")
        if self.track_inverse:
            self._apply_1q_to_pair(self.inv_x, self.inv_z, q, "S")

    def s_dag(self, q: int):
        """S-dagger. (Phase-free: identical to S.)"""
        self.s(q)

    def x_gate(self, q: int):
        """Pauli X. No GF(2) effect (phase only)."""
        self._check_qubit(q)

    def y_gate(self, q: int):
        """Pauli Y. No GF(2) effect (phase only)."""
        self._check_qubit(q)

    def z_gate(self, q: int):
        """Pauli Z. No GF(2) effect (phase only)."""
        self._check_qubit(q)

    def sqrt_x(self, q: int):
        """sqrt(X): Z -> Y (XZ), X -> X. (Phase-free: identical to sqrt_X_dag.)"""
        self._check_qubit(q)
        self._apply_1q_to_pair(self.x, self.z, q, "SQRT_X")
        if self.track_inverse:
            self._apply_1q_to_pair(self.inv_x, self.inv_z, q, "SQRT_X")

    def sqrt_x_dag(self, q: int):
        """sqrt(X)-dagger. (Phase-free: identical to sqrt_X.)"""
        self.sqrt_x(q)

    # ------------------------------------------------------------------
    # Two-qubit Clifford gates
    # ------------------------------------------------------------------

    # Named-gate symplectic matrices (precomputed)
    _S_CNOT = np.array([[1,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,1]], dtype=np.uint8)
    _S_CZ   = np.array([[1,0,0,1],[0,1,1,0],[0,0,1,0],[0,0,0,1]], dtype=np.uint8)
    _S_SWAP = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.uint8)

    def cnot(self, control: int, target: int):
        """CNOT (CX): X_c -> X_c X_t, Z_t -> Z_c Z_t."""
        self._check_two_qubits(control, target)
        self._apply_2q_to_pair(self.x, self.z, control, target, self._S_CNOT)
        if self.track_inverse:
            self._apply_2q_to_pair(self.inv_x, self.inv_z, control, target, self._S_CNOT)

    def cz(self, q0: int, q1: int):
        """CZ (symmetric): X_0 -> X_0 Z_1, X_1 -> Z_0 X_1."""
        self._check_two_qubits(q0, q1)
        self._apply_2q_to_pair(self.x, self.z, q0, q1, self._S_CZ)
        if self.track_inverse:
            self._apply_2q_to_pair(self.inv_x, self.inv_z, q0, q1, self._S_CZ)

    def swap(self, q0: int, q1: int):
        """SWAP: exchange qubits q0, q1."""
        self._check_two_qubits(q0, q1)
        self._apply_2q_to_pair(self.x, self.z, q0, q1, self._S_SWAP)
        if self.track_inverse:
            self._apply_2q_to_pair(self.inv_x, self.inv_z, q0, q1, self._S_SWAP)

    def iswap(self, q0: int, q1: int):
        """iSWAP gate via verified 4x4 symplectic matrix."""
        self.apply_clifford_2q(q0, q1, _ISWAP_SYMPLECTIC)

    def apply_clifford_2q(self, q0: int, q1: int, symplectic_4x4: np.ndarray):
        """Apply arbitrary 2-qubit Clifford via its 4x4 GF(2) symplectic matrix."""
        self._check_two_qubits(q0, q1)
        self._apply_2q_to_pair(self.x, self.z, q0, q1, symplectic_4x4)
        if self.track_inverse:
            # All Clifford gates are self-inverse at GF(2) level
            self._apply_2q_to_pair(self.inv_x, self.inv_z, q0, q1, symplectic_4x4)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure_z(self, q: int):
        """Measure qubit q in Z basis and reset to |0>."""
        self._check_qubit(q)
        if HAS_NUMBA and not self.track_inverse:
            measure_z_packed(self.x.data, self.z.data, q)
        else:
            x_col = self.x.get_column_bits(q)
            anticomm = np.where(x_col == 1)[0]
            if len(anticomm) == 0:
                return
            pivot = anticomm[0]
            others = anticomm[1:]
            if len(others) > 0:
                self.x.xor_rows_broadcast(others, pivot)
                self.z.xor_rows_broadcast(others, pivot)
                if self.track_inverse:
                    self.inv_x.xor_rows_broadcast(others, pivot)
                    self.inv_z.xor_rows_broadcast(others, pivot)
            self.x.clear_row(pivot)
            self.z.clear_row(pivot)
            self.z.set_bit(pivot, q, 1)
            if self.track_inverse:
                self.inv_x.clear_row(pivot)
                self.inv_z.clear_row(pivot)
                self.inv_z.set_bit(pivot, q, 1)

    # ------------------------------------------------------------------
    # Tableau extraction
    # ------------------------------------------------------------------

    def to_symplectic(self) -> np.ndarray:
        """Return full symplectic matrix [X | Z] as dense uint8, shape (n, 2n)."""
        return np.hstack([self.x.to_dense(), self.z.to_dense()])

    def to_symplectic_packed(self) -> PackedBitMatrix:
        """Return full symplectic matrix [X | Z] as PackedBitMatrix."""
        return self.x.hstack(self.z)

    def gf2_rank(self) -> int:
        """GF(2) rank of the full symplectic matrix."""
        return self.to_symplectic_packed().rank()

    def system_rank(self, n_system: int) -> int:
        """GF(2) rank of the system-restricted symplectic matrix.

        For a 2n-qubit purification state, code dimension k = rank - n_system.
        """
        if not (0 <= n_system <= self.n):
            raise ValueError(f"n_system={n_system} out of range [0, {self.n}]")
        # Extract system columns only (0..n_sys-1) from both x and z
        sys_x_dense = self.x.to_dense()[:, :n_system]
        sys_z_dense = self.z.to_dense()[:, :n_system]
        sys_packed = PackedBitMatrix.from_dense(
            np.hstack([sys_x_dense, sys_z_dense]))
        return sys_packed.rank()

    def copy(self) -> "StabilizerTableau":
        """Deep copy."""
        tab = StabilizerTableau.__new__(StabilizerTableau)
        tab.n = self.n
        tab.track_inverse = self.track_inverse
        tab.x = self.x.copy()
        tab.z = self.z.copy()
        tab.inv_x = self.inv_x.copy() if self.inv_x is not None else None
        tab.inv_z = self.inv_z.copy() if self.inv_z is not None else None
        return tab


# ======================================================================
# GF(2) linear algebra (dense wrappers for backward compatibility)
# ======================================================================

def _gf2_rank(matrix: np.ndarray) -> int:
    """GF(2) rank of a dense uint8 matrix."""
    return PackedBitMatrix.from_dense(matrix.astype(np.uint8)).rank()


def gf2_rref(matrix: np.ndarray) -> np.ndarray:
    """GF(2) RREF of a dense uint8 matrix. Zero rows removed."""
    return PackedBitMatrix.from_dense(matrix.astype(np.uint8)).rref().to_dense()


def stabilizer_groups_equal(tab_a: np.ndarray, tab_b: np.ndarray) -> bool:
    """Check if two dense symplectic matrices span the same GF(2) subspace."""
    a = PackedBitMatrix.from_dense(tab_a.astype(np.uint8))
    b = PackedBitMatrix.from_dense(tab_b.astype(np.uint8))
    return packed_stabilizer_groups_equal(a, b)
