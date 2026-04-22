"""
Bit-packed GF(2) matrix using uint64 words.

Each row of the matrix is stored as a sequence of uint64 words, packing 64
columns into each word. This gives a 64x density improvement over uint8
per-element storage and allows XOR operations to process 64 bits at once.

Storage layout
--------------
For a matrix with n_cols columns:
    n_words = ceil(n_cols / 64)
    data[r, w] contains bits for columns [64*w, 64*w+64) of row r
    Bit 0 (LSB) of data[r, w] corresponds to column 64*w
    Bit 63 (MSB) of data[r, w] corresponds to column 64*w + 63

This is the same layout Stim uses internally and what a future C++ backend
will expect (contiguous uint64 array, row-major).
"""

import numpy as np


class PackedBitMatrix:
    """Bit-packed binary matrix over GF(2).

    Parameters
    ----------
    n_rows : int
        Number of rows.
    n_cols : int
        Number of logical columns (bits per row).

    Attributes
    ----------
    n_rows : int
    n_cols : int
    n_words : int
        Number of uint64 words per row: ceil(n_cols / 64).
    data : ndarray of uint64, shape (n_rows, n_words)
        The packed bit storage.
    """

    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_words = (n_cols + 63) // 64
        self.data = np.zeros((n_rows, self.n_words), dtype=np.uint64)

    # ------------------------------------------------------------------
    # Bit access
    # ------------------------------------------------------------------

    def get_bit(self, row: int, col: int) -> int:
        """Get a single bit value."""
        w = col >> 6           # col // 64
        b = col & 63           # col % 64
        return int((self.data[row, w] >> np.uint64(b)) & np.uint64(1))

    def set_bit(self, row: int, col: int, val: int):
        """Set a single bit value (0 or 1)."""
        w = col >> 6
        b = col & 63
        mask = np.uint64(1) << np.uint64(b)
        if val:
            self.data[row, w] |= mask
        else:
            self.data[row, w] &= ~mask

    def get_column_bits(self, col: int) -> np.ndarray:
        """Extract a single column as a uint8 array of shape (n_rows,).

        Returns 0 or 1 for each row.
        """
        w = col >> 6
        b = np.uint64(col & 63)
        return ((self.data[:, w] >> b) & np.uint64(1)).astype(np.uint8)

    def set_column_bits(self, col: int, bits: np.ndarray):
        """Set a column from a uint8 array of 0/1 values, shape (n_rows,)."""
        w = col >> 6
        b = np.uint64(col & 63)
        mask = np.uint64(1) << b
        # Clear the column bits
        self.data[:, w] &= ~mask
        # Set where bits == 1
        self.data[:, w] |= (bits.astype(np.uint64) << b)

    def xor_column_into(self, dst_col: int, src_col: int):
        """XOR src column into dst column: dst[:, dst_col] ^= src[:, src_col]."""
        sw = src_col >> 6
        sb = np.uint64(src_col & 63)
        dw = dst_col >> 6
        db = np.uint64(dst_col & 63)
        src_bits = (self.data[:, sw] >> sb) & np.uint64(1)
        self.data[:, dw] ^= src_bits << db

    # ------------------------------------------------------------------
    # Row operations
    # ------------------------------------------------------------------

    def xor_rows(self, dst: int, src: int):
        """XOR src row into dst row: data[dst] ^= data[src]."""
        self.data[dst] ^= self.data[src]

    def xor_rows_broadcast(self, dst_rows: np.ndarray, src: int):
        """XOR src row into multiple dst rows: data[dst_rows] ^= data[src].

        Uses NumPy broadcasting for efficiency.
        """
        if len(dst_rows) > 0:
            self.data[dst_rows] ^= self.data[src]

    def swap_rows(self, a: int, b: int):
        """Swap rows a and b."""
        self.data[[a, b]] = self.data[[b, a]]

    def clear_row(self, row: int):
        """Set all bits in a row to 0."""
        self.data[row] = 0

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dense(self) -> np.ndarray:
        """Convert to dense uint8 matrix of shape (n_rows, n_cols)."""
        dense = np.zeros((self.n_rows, self.n_cols), dtype=np.uint8)
        for col in range(self.n_cols):
            dense[:, col] = self.get_column_bits(col)
        return dense

    @classmethod
    def from_dense(cls, mat: np.ndarray) -> "PackedBitMatrix":
        """Create from a dense uint8 matrix of shape (rows, cols)."""
        n_rows, n_cols = mat.shape
        packed = cls(n_rows, n_cols)
        for col in range(n_cols):
            packed.set_column_bits(col, mat[:, col])
        return packed

    @classmethod
    def identity(cls, n: int) -> "PackedBitMatrix":
        """Create an n x n identity matrix."""
        mat = cls(n, n)
        for i in range(n):
            mat.set_bit(i, i, 1)
        return mat

    @classmethod
    def zeros(cls, n_rows: int, n_cols: int) -> "PackedBitMatrix":
        """Create an all-zeros matrix."""
        return cls(n_rows, n_cols)

    def copy(self) -> "PackedBitMatrix":
        """Deep copy."""
        new = PackedBitMatrix.__new__(PackedBitMatrix)
        new.n_rows = self.n_rows
        new.n_cols = self.n_cols
        new.n_words = self.n_words
        new.data = self.data.copy()
        return new

    def hstack(self, other: "PackedBitMatrix") -> "PackedBitMatrix":
        """Horizontal concatenation: [self | other].

        Returns a new matrix with n_cols = self.n_cols + other.n_cols.
        """
        assert self.n_rows == other.n_rows
        # Convert to dense, hstack, repack (simple but correct)
        dense = np.hstack([self.to_dense(), other.to_dense()])
        return PackedBitMatrix.from_dense(dense)

    # ------------------------------------------------------------------
    # GF(2) linear algebra
    # ------------------------------------------------------------------

    def rank(self) -> int:
        """GF(2) rank via Gaussian elimination on a copy."""
        try:
            from sparsegf2.core.numba_kernels import HAS_NUMBA, gf2_rank_packed
            if HAS_NUMBA:
                return gf2_rank_packed(self.data.copy(), self.n_cols)
        except ImportError:
            pass
        _, r = self._eliminate()
        return r

    def rref(self) -> "PackedBitMatrix":
        """GF(2) reduced row echelon form. Returns new matrix with zero rows removed."""
        mat, rank = self._eliminate()
        # Keep only the first `rank` rows (they are the nonzero RREF rows)
        result = PackedBitMatrix(rank, mat.n_cols)
        result.data = mat.data[:rank].copy()
        return result

    def _eliminate(self) -> tuple["PackedBitMatrix", int]:
        """GF(2) Gaussian elimination (RREF). Returns (eliminated_copy, rank)."""
        mat = self.copy()
        rows = mat.n_rows
        cols = mat.n_cols
        pivot_row = 0

        for col in range(cols):
            if pivot_row >= rows:
                break

            # Find pivot: first row >= pivot_row with bit set in this column
            w = col >> 6
            b = np.uint64(col & 63)
            col_bits = (mat.data[pivot_row:, w] >> b) & np.uint64(1)
            idx = int(np.argmax(col_bits))
            if not col_bits[idx]:
                continue
            found = pivot_row + idx

            # Swap to pivot position
            if found != pivot_row:
                mat.swap_rows(found, pivot_row)

            # Eliminate all other rows with a 1 in this column
            all_col_bits = (mat.data[:, w] >> b) & np.uint64(1)
            mask = all_col_bits.astype(bool)
            mask[pivot_row] = False
            # XOR pivot row into all rows that have bit set
            if np.any(mask):
                mat.data[mask] ^= mat.data[pivot_row]

            pivot_row += 1

        return mat, pivot_row


def packed_stabilizer_groups_equal(a: PackedBitMatrix, b: PackedBitMatrix) -> bool:
    """Check if two packed matrices span the same GF(2) subspace."""
    rref_a = a.rref()
    rref_b = b.rref()
    if rref_a.n_rows != rref_b.n_rows or rref_a.n_cols != rref_b.n_cols:
        return False
    return np.array_equal(rref_a.data, rref_b.data)
