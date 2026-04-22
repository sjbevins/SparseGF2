"""
Stabilizer tableau simulator over GF(2).

A general-purpose stabilizer simulator that maintains the symplectic
representation of n-qubit stabilizer states. Supports arbitrary single-
and two-qubit Clifford gates, Z-basis measurement with reset, and
extraction of the full stabilizer tableau for analysis.

Phase-free representation
-------------------------
This simulator tracks only the GF(2) symplectic part of the stabilizer
tableau, NOT the phase bits. Gates like S and S_dag, or sqrt_X and
sqrt_X_dag, have identical GF(2) effects and are interchangeable here.
This is correct for all observables that depend only on the stabilizer
subspace (rank, entropy, code distance, weight spectrum) but NOT for
observables that depend on the sign of individual generators.

Symplectic representation
-------------------------
An n-qubit stabilizer state has n stabilizer generators. Each generator is
a Pauli string P = i^phase * X^{a} Z^{b} where a, b are n-bit vectors.
We store only the GF(2) part:

    x[r, q] = 1 if generator r has X or Y on qubit q
    z[r, q] = 1 if generator r has Y or Z on qubit q

So generator r acts on qubit q as:
    x=0, z=0 -> I
    x=1, z=0 -> X
    x=1, z=1 -> Y
    x=0, z=1 -> Z

Gate application
----------------
A Clifford gate on qubits (qi, qj) is described by a 4x4 GF(2) matrix S
that maps the input symplectic vector (x_qi, x_qj, z_qi, z_qj) to the
output. For each generator row, we extract 4 bits, multiply by S over
GF(2), and write back the 4 result bits. This is vectorized via NumPy
matrix multiply: ``out = (bits @ S) & 1``.

Measurement
-----------
Z-basis measurement on qubit q:
1. Find all generators with x[r, q] = 1 (anticommuting with Z_q).
2. If any exist: pick the first as pivot, XOR it into all others,
   then replace the pivot row with the single generator Z_q.
3. If none: the state is already an eigenstate of Z_q.

This matches the CHP algorithm (Aaronson & Gottesman 2004) and produces
the same stabilizer group as Stim.
"""

import numpy as np

# iSWAP 4x4 GF(2) symplectic matrix, verified against stim.Tableau.from_named_gate("ISWAP"):
#   X0 -> Z0 Y1,  X1 -> Y0 Z1,  Z0 -> Z1,  Z1 -> Z0
_ISWAP_SYMPLECTIC = np.array([
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=np.uint8)


class StabilizerTableauRef:
    """GF(2) stabilizer tableau for n qubits.

    Parameters
    ----------
    n : int
        Number of qubits (must be >= 0).

    Attributes
    ----------
    n : int
        Number of qubits.
    x : ndarray of uint8, shape (n, n)
        X part of the symplectic matrix.
    z : ndarray of uint8, shape (n, n)
        Z part of the symplectic matrix.
    """

    def __init__(self, n: int):
        if n < 0:
            raise ValueError(f"Number of qubits must be non-negative, got {n}")
        self.n = n
        self.x = np.zeros((n, n), dtype=np.uint8)
        self.z = np.eye(n, dtype=np.uint8)

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
    def from_zero_state(cls, n: int) -> "StabilizerTableauRef":
        """Create |0...0>. Stabilizers: Z_0, ..., Z_{n-1}."""
        return cls(n)

    @classmethod
    def from_bell_pairs(cls, n_system: int) -> "StabilizerTableauRef":
        """Create 2n-qubit Bell pair state.

        System qubits: 0..n-1, reference qubits: n..2n-1.
        Generators: X_i X_{n+i} (rows 0..n-1) and Z_i Z_{n+i} (rows n..2n-1).
        """
        n = n_system
        tab = cls(2 * n)
        tab.x[:] = 0
        tab.z[:] = 0
        idx = np.arange(n)
        tab.x[idx, idx] = 1
        tab.x[idx, n + idx] = 1
        tab.z[n + idx, idx] = 1
        tab.z[n + idx, n + idx] = 1
        return tab

    # ------------------------------------------------------------------
    # Single-qubit Clifford gates (phase-free)
    # ------------------------------------------------------------------

    def h(self, q: int):
        """Hadamard: X <-> Z."""
        self._check_qubit(q)
        self.x[:, q], self.z[:, q] = self.z[:, q].copy(), self.x[:, q].copy()

    def s(self, q: int):
        """Phase gate S: X -> Y (XZ), Z -> Z. (Phase-free: identical to S_dag.)"""
        self._check_qubit(q)
        self.z[:, q] ^= self.x[:, q]

    def s_dag(self, q: int):
        """S-dagger: X -> -Y, Z -> Z. (Phase-free: identical to S.)"""
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
        self.x[:, q] ^= self.z[:, q]

    def sqrt_x_dag(self, q: int):
        """sqrt(X)-dagger. (Phase-free: identical to sqrt_X.)"""
        self.sqrt_x(q)

    # ------------------------------------------------------------------
    # Two-qubit Clifford gates
    # ------------------------------------------------------------------

    def cnot(self, control: int, target: int):
        """CNOT (CX): X_c -> X_c X_t, Z_t -> Z_c Z_t."""
        self._check_two_qubits(control, target)
        self.x[:, target] ^= self.x[:, control]
        self.z[:, control] ^= self.z[:, target]

    def cz(self, q0: int, q1: int):
        """CZ (symmetric): X_0 -> X_0 Z_1, X_1 -> Z_0 X_1."""
        self._check_two_qubits(q0, q1)
        self.z[:, q1] ^= self.x[:, q0]
        self.z[:, q0] ^= self.x[:, q1]

    def swap(self, q0: int, q1: int):
        """SWAP: exchange qubits q0, q1."""
        self._check_two_qubits(q0, q1)
        self.x[:, q0], self.x[:, q1] = self.x[:, q1].copy(), self.x[:, q0].copy()
        self.z[:, q0], self.z[:, q1] = self.z[:, q1].copy(), self.z[:, q0].copy()

    def iswap(self, q0: int, q1: int):
        """iSWAP gate via verified 4x4 symplectic matrix."""
        self.apply_clifford_2q(q0, q1, _ISWAP_SYMPLECTIC)

    def apply_clifford_2q(self, q0: int, q1: int, symplectic_4x4: np.ndarray):
        """Apply arbitrary 2-qubit Clifford via its 4x4 GF(2) symplectic matrix.

        Vectorized: all generator rows are updated in a single NumPy operation.

        Parameters
        ----------
        q0, q1 : int
            The two qubits (must be distinct and in range).
        symplectic_4x4 : ndarray of uint8, shape (4, 4)
            GF(2) matrix: (x_q0, x_q1, z_q0, z_q1) -> (x'_q0, x'_q1, z'_q0, z'_q1).
        """
        self._check_two_qubits(q0, q1)
        S = symplectic_4x4
        # Gather 4 columns into (n_gen, 4) matrix
        bits = np.stack([self.x[:, q0], self.x[:, q1],
                         self.z[:, q0], self.z[:, q1]], axis=1)
        # GF(2) matrix multiply: cast to int32 to avoid platform-dependent
        # uint8 overflow behavior in NumPy matmul
        out = (bits.astype(np.int32) @ S.astype(np.int32)).astype(np.uint8) & 1
        self.x[:, q0] = out[:, 0]
        self.x[:, q1] = out[:, 1]
        self.z[:, q0] = out[:, 2]
        self.z[:, q1] = out[:, 3]

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure_z(self, q: int):
        """Measure qubit q in Z basis and reset to |0>.

        Modifies the tableau in place. The measurement outcome is random
        (not tracked since we only need the GF(2) stabilizer group).
        """
        self._check_qubit(q)
        # Find anticommuting rows (X support on qubit q)
        anticomm = np.where(self.x[:, q] == 1)[0]

        if len(anticomm) == 0:
            return

        pivot = anticomm[0]

        # XOR pivot into all other anticommuting rows (vectorized)
        others = anticomm[1:]
        if len(others) > 0:
            self.x[others] ^= self.x[pivot]
            self.z[others] ^= self.z[pivot]

        # Replace pivot with Z_q
        self.x[pivot, :] = 0
        self.z[pivot, :] = 0
        self.z[pivot, q] = 1

    # ------------------------------------------------------------------
    # Tableau extraction
    # ------------------------------------------------------------------

    def to_symplectic(self) -> np.ndarray:
        """Return full symplectic matrix [X | Z], shape (n_gen, 2*n_qubits)."""
        return np.hstack([self.x, self.z])

    def gf2_rank(self) -> int:
        """GF(2) rank of the full symplectic matrix."""
        return _gf2_rank(self.to_symplectic())

    def system_rank(self, n_system: int) -> int:
        """GF(2) rank of the system-restricted symplectic matrix.

        For a 2n-qubit purification state, code dimension k = rank - n_system.
        Extracts x[:, :n_sys] and z[:, :n_sys] (system qubit columns only).
        """
        if not (0 <= n_system <= self.n):
            raise ValueError(
                f"n_system={n_system} out of range [0, {self.n}]")
        sys_matrix = np.hstack([
            self.x[:, :n_system], self.z[:, :n_system]
        ])
        return _gf2_rank(sys_matrix)

    def copy(self) -> "StabilizerTableauRef":
        """Deep copy."""
        tab = StabilizerTableauRef.__new__(StabilizerTableauRef)
        tab.n = self.n
        tab.x = self.x.copy()
        tab.z = self.z.copy()
        return tab


# ======================================================================
# GF(2) linear algebra
# ======================================================================

def _gf2_eliminate(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """GF(2) Gaussian elimination (in-place on a copy).

    Returns the eliminated matrix and its rank.
    """
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")
    mat = matrix.astype(np.uint8, copy=True)
    rows, cols = mat.shape
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
        # Vectorized pivot search
        sub = mat[pivot_row:, col]
        idx = int(np.argmax(sub))
        if not sub[idx]:
            continue
        found = pivot_row + idx
        if found != pivot_row:
            mat[[pivot_row, found]] = mat[[found, pivot_row]]
        # Vectorized elimination of all other rows with a 1 in this column
        mask = mat[:, col].astype(bool)
        mask[pivot_row] = False
        mat[mask] ^= mat[pivot_row]
        pivot_row += 1
    return mat, pivot_row


def _gf2_rank(matrix: np.ndarray) -> int:
    """GF(2) rank via Gaussian elimination."""
    _, rank = _gf2_eliminate(matrix)
    return rank


def gf2_rref(matrix: np.ndarray) -> np.ndarray:
    """GF(2) reduced row echelon form. Zero rows removed."""
    mat, _ = _gf2_eliminate(matrix)
    return mat[np.any(mat != 0, axis=1)]


def stabilizer_groups_equal(tab_a: np.ndarray, tab_b: np.ndarray) -> bool:
    """Check if two symplectic matrices span the same GF(2) subspace via RREF."""
    rref_a = gf2_rref(tab_a)
    rref_b = gf2_rref(tab_b)
    if rref_a.shape != rref_b.shape:
        return False
    return np.array_equal(rref_a, rref_b)
