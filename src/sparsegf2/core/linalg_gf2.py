"""Linear algebra over :math:`\\mathbb{F}_2 = \\{0, 1\\}` shared by the core simulator.

This module is the **single source of truth** for the GF(2) primitives
used in multiple places:

* :func:`gf2_rref`: row-reduced echelon form. Drives
  :meth:`sparsegf2.SparseGF2.canonical_form` and every observable that
  computes a subsystem rank.
* :func:`gf2_kernel_basis`: basis of the right kernel
  :math:`\\{v : Mv^\\top = 0 \\bmod 2\\}`. Drives the uniform sampler
  :func:`sparsegf2.random_symplectic`.

Both primitives have a public **Numba-JIT path** (preferred) and a
pure-Python fallback (used when Numba is not available, typically only
in `use_numba=False` instances during debugging). The two paths produce
**bit-identical** results.

Design contracts
================

* All inputs and outputs are ``np.uint8`` arrays whose entries are 0 or 1.
* All routines are pure (no caller-state mutation outside the named
  in-place inputs).
* :func:`gf2_rref` mutates its argument **in place**; callers wanting a
  fresh result should pass a copy.
* :func:`gf2_kernel_basis` returns a freshly-allocated matrix; the
  input ``M`` is not modified.

References
==========

* Aaronson & Gottesman, *Improved Simulation of Stabilizer Circuits*,
  Phys. Rev. A **70**, 052328 (2004). `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_.
* Standard textbook GF(2) Gaussian elimination; see e.g.
  Cormen *et al.*, *Introduction to Algorithms*, §28.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sparsegf2.errors import InvalidArgumentError

try:  # pragma: no cover (JIT path is the default; fallback is for debug)
    from sparsegf2.core.numba_kernels import (
        gf2_kernel_basis_jit as _gf2_kernel_basis_jit,
    )
    from sparsegf2.core.numba_kernels import (
        gf2_rank_words_jit as _gf2_rank_words_jit,
    )
    from sparsegf2.core.numba_kernels import (
        gf2_rref_jit as _gf2_rref_jit,
    )

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _gf2_rref_jit = None
    _gf2_kernel_basis_jit = None
    _gf2_rank_words_jit = None
    HAS_NUMBA = False


# ----------------------------------------------------------------------
# Reduced row-echelon form
# ----------------------------------------------------------------------


def gf2_rref(A: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Reduce ``A`` to GF(2) row-reduced echelon form **in place**.

    Standard pivot-and-eliminate Gaussian elimination over
    :math:`\\mathbb{F}_2`. Mutates ``A`` and returns it for chaining.
    The Numba JIT path is ~10-50× faster than the pure-Python
    fallback for moderate ``A`` shapes.

    Inputs are normalized to :math:`\\{0, 1\\}` in place (via ``A &= 1``)
    before reduction, so it accepts any ``uint8`` matrix even if entries
    happen to be 2 or 255.

    Parameters
    ----------
    A : ndarray of shape ``(m, k)``, ``uint8``
        Modulo-2 reduced and mutated in place.

    Returns
    -------
    ndarray of shape ``(m, k)``, ``uint8``
        The reduced row-echelon form of the input. Returned object is
        the same array as ``A``.
    """
    if A.size == 0:
        return A
    # Normalize entries to {0, 1} so the JIT path doesn't have to.
    A &= 1
    if HAS_NUMBA:
        return _gf2_rref_jit(A)
    return _gf2_rref_python(A)


def gf2_rank(A: NDArray[np.uint8]) -> int:
    """Return the rank of ``A`` over :math:`\\mathbb{F}_2`.

    Copies ``A`` (rank is a query, not a mutation) and counts non-zero
    rows of the resulting RREF.

    Parameters
    ----------
    A : ndarray of shape ``(m, k)``, ``uint8``
        Each entry is 0 or 1.

    Returns
    -------
    int
        :math:`\\mathrm{rank}_{\\mathbb{F}_2}(A) \\in [0, \\min(m, k)]`.
    """
    if A.size == 0:
        return 0
    # gf2_rref mutates in place and applies ``& 1`` internally, so a
    # single uint8 copy is sufficient, with no need to mask here.
    work = np.array(A, dtype=np.uint8, copy=True)
    gf2_rref(work)
    return int(np.any(work, axis=1).sum())


def gf2_rank_bits(A: NDArray[np.uint8]) -> int:
    """Rank of ``A`` over :math:`\\mathbb{F}_2` via bit-packed elimination.

    Identical result to :func:`gf2_rank`, but the matrix is first packed 64
    columns to a ``uint64`` word (LSB-first), so the row-XOR inner loop runs on
    whole words, ~64× less work than the uint8-per-bit elimination. This is
    the right call for repeated ranks of moderate-to-large matrices, e.g. the
    per-layer order parameter of ``until_purified`` circuit runs, where the
    uint8 rank dominated production wall-clock at large ``n``.

    Parameters
    ----------
    A : ndarray of shape ``(m, k)``, ``uint8``
        Entries are taken mod 2. ``A`` is not mutated.

    Returns
    -------
    int
        :math:`\\mathrm{rank}_{\\mathbb{F}_2}(A)`, exactly as :func:`gf2_rank`.
    """
    if A.size == 0:
        return 0
    # Coerce exactly like gf2_rank (np.array(dtype=uint8)) so the two accept
    # the same inputs: float entries truncate instead of raising on `& 1`.
    A = np.asarray(A, dtype=np.uint8)
    n_rows, n_cols = A.shape
    # Pack each row's bits LSB-first: column c -> bit (c & 63) of word (c >> 6).
    # np.packbits(bitorder="little") gives byte (c >> 3) bit (c & 7); padding the
    # byte rows to a multiple of 8 and viewing **little-endian** uint64 composes
    # to exactly the word layout the elimination kernel expects. The explicit
    # "<u8" dtype is load-bearing: on a hypothetical big-endian host a native
    # view would silently permute columns (wrong rank); "<u8" keeps the Python
    # fallback correct there and makes the JIT path fail loudly (non-native
    # dtype) instead of returning a wrong answer. On every supported platform
    # (little-endian) it is identical to np.uint64.
    packed8 = np.packbits(A & 1, axis=1, bitorder="little")
    n_bytes = packed8.shape[1]
    n_words = (n_bytes + 7) // 8
    if n_bytes != 8 * n_words:
        pad = np.zeros((n_rows, 8 * n_words - n_bytes), dtype=np.uint8)
        packed8 = np.concatenate([packed8, pad], axis=1)
    words = np.ascontiguousarray(packed8).view(np.dtype("<u8"))
    if HAS_NUMBA:
        return int(_gf2_rank_words_jit(words, n_cols))
    return _gf2_rank_words_python(words, n_cols)


def _gf2_rank_words_python(mat: NDArray[np.uint64], n_cols: int) -> int:
    """Pure-Python mirror of ``gf2_rank_words_jit`` (destructive on ``mat``)."""
    n_rows = mat.shape[0]
    one = np.uint64(1)
    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        w = col >> 6
        b = np.uint64(col & 63)
        found = -1
        for r in range(pivot_row, n_rows):
            if (mat[r, w] >> b) & one:
                found = r
                break
        if found < 0:
            continue
        if found != pivot_row:
            mat[[pivot_row, found]] = mat[[found, pivot_row]]
        col_bits = (mat[:, w] >> b) & one
        for r in np.nonzero(col_bits)[0]:
            if r != pivot_row:
                mat[r] ^= mat[pivot_row]
        pivot_row += 1
    return pivot_row


def _gf2_rref_python(A: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Pure-Python GF(2) RREF used when Numba is not importable (``HAS_NUMBA=False``).

    Note: this fallback is gated on module-level Numba availability, not
    on the per-instance ``SparseGF2(use_numba=False)`` flag. A
    ``SparseGF2`` constructed with ``use_numba=False`` still routes its
    ``canonical_form()`` calls through this module's :func:`gf2_rref`,
    which in turn prefers the JIT kernel whenever Numba is installed.
    Mutates ``A`` in place.
    """
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


# ----------------------------------------------------------------------
# Forward elimination on a chosen pivot-column set
# ----------------------------------------------------------------------


def gf2_eliminate_on_columns(
    mat: NDArray[np.uint8], cols: NDArray[np.int64]
) -> tuple[NDArray[np.uint8], int]:
    """Forward GF(2) elimination of ``mat`` using ``cols`` as pivot columns
    (left to right), applying every row operation across all of ``mat``.

    Returns a reduced copy and the rank ``r`` obtained on ``cols``: rows
    ``0 .. r-1`` carry the pivots (and span the column space of ``cols``), and
    rows ``r ..`` are zero in every column of ``cols``. Columns outside
    ``cols`` are carried along by the row operations but never pivoted on,
    which is what makes this the workhorse for block eliminations:
    :func:`sparsegf2.contiguous_distance` factors a fixed reference subsystem
    out of the stabilizer block once, and the expurgation package's witness
    extraction reduces ``[M | I]`` on the syndrome block, then the logical
    block, reading candidates off the identity block.

    Parameters
    ----------
    mat : ndarray of shape ``(m, k)``, ``uint8``
        Entries 0 or 1. Not mutated; a copy is reduced and returned.
    cols : ndarray of int
        Column indices to pivot on, processed left to right in the given
        order.

    Returns
    -------
    tuple of (ndarray, int)
        The reduced copy and the rank achieved on ``cols``.
    """
    m = np.asarray(mat)
    if m.ndim != 2:
        raise InvalidArgumentError(f"mat must be two-dimensional; got shape {m.shape}")
    m = np.array(m, dtype=np.uint8, copy=True)
    m &= 1
    raw_cols = np.asarray(list(cols), dtype=object)
    if raw_cols.ndim != 1:
        raise InvalidArgumentError(f"cols must be one-dimensional; got shape {raw_cols.shape}")
    for value in raw_cols:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise InvalidArgumentError(f"cols must contain exact integers; got {value!r}")
    col_indices = np.asarray([int(value) for value in raw_cols], dtype=np.int64)
    if col_indices.size and (col_indices.min() < 0 or col_indices.max() >= m.shape[1]):
        raise InvalidArgumentError(
            f"cols must lie in [0, {m.shape[1]}); got {col_indices.tolist()}"
        )
    r = 0
    n_rows = m.shape[0]
    for c in col_indices:
        nz = np.flatnonzero(m[r:, c])
        if nz.size == 0:
            continue
        piv = r + int(nz[0])
        if piv != r:
            m[[r, piv]] = m[[piv, r]]
        hits = m[:, c].astype(bool).copy()
        hits[r] = False
        if hits.any():
            m[hits] ^= m[r]
        r += 1
        if r == n_rows:
            break
    return m, r


# ----------------------------------------------------------------------
# Kernel basis (right kernel)
# ----------------------------------------------------------------------


def gf2_kernel_basis(M: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return a basis of the right kernel of ``M`` over :math:`\\mathbb{F}_2`.

    Computes a basis of
    :math:`\\{v \\in \\mathbb{F}_2^k : M v^\\top = 0 \\bmod 2\\}`.

    Algorithm (see master notebook §16.4 for the full derivation):

    1. Reduce ``M`` to RREF in place, recording the pivot columns.
    2. For each *free column* :math:`f` (not a pivot), emit a basis
       vector with :math:`v[f] = 1` and :math:`v[p_r] = A[r, f]` for
       each pivot column :math:`p_r`.

    Parameters
    ----------
    M : ndarray of shape ``(m, k)``, ``uint8``
        Each entry is 0 or 1.

    Returns
    -------
    ndarray of shape ``(k - rank(M), k)``, ``uint8``
        Rows form a basis of the right kernel. When ``M`` is empty
        (``m == 0``), returns the ``(k, k)`` identity. When
        ``rank(M) == k``, returns a ``(0, k)`` empty array.

    Notes
    -----
    Input is normalized to :math:`\\{0, 1\\}` before reduction, so it accepts
    any ``uint8`` matrix even if entries are 2 or 255. ``M`` itself is
    *not* mutated; a normalized copy is made internally.
    """
    # Normalize a contiguous uint8 copy; original M is not mutated.
    M_norm = np.ascontiguousarray(M, dtype=np.uint8) & 1
    if HAS_NUMBA:
        return _gf2_kernel_basis_jit(M_norm)
    return _gf2_kernel_basis_python(M_norm)


def _gf2_kernel_basis_python(M: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Pure-Python kernel basis. Reference for the JIT version."""
    n_rows, n_cols = M.shape
    if n_rows == 0:
        return np.eye(n_cols, dtype=np.uint8)
    A = (M.astype(np.uint8) & 1).copy()
    pivot_cols: list[int] = []
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
        pivot_cols.append(col)
        pivot_row += 1
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n_cols) if c not in pivot_set]
    if not free_cols:
        return np.zeros((0, n_cols), dtype=np.uint8)
    kernel = np.zeros((len(free_cols), n_cols), dtype=np.uint8)
    for k, free_col in enumerate(free_cols):
        kernel[k, free_col] = 1
        for r, pc in enumerate(pivot_cols):
            kernel[k, pc] = A[r, free_col]
    return kernel
