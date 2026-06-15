"""Numba-JIT compiled mirrors of the pure-Python kernels in
:mod:`sparsegf2.core.sparse_tableau`.

Every function here is a direct translation of its Python twin to a
loop body that Numba can compile with ``nopython=True``. The data layout
is identical (the same numpy arrays are passed by reference), so the
JIT version can be swapped in transparently per-instance via
``SparseGF2(n, use_numba=True)``.

Design notes
------------

* No Python lists, sets, or dicts inside ``@njit`` bodies. Active-set
    deduplication is done via the PLT (``plt[r, qi] != 0`` ⇔ ``r`` is
    already in ``inv[qi]``).
* LUTs are built in Python (16 entries each) and passed in as uint8
    arrays. Building a LUT is O(1) per gate, but doing it in pure Python
    keeps the kernels simple and the LUT-build code shared with the
    pure-Python path.
* The kernel signatures match the pure-Python kernels in
    :mod:`sparsegf2.core.sparse_tableau` so that tests can compare the
    two implementations on the same arrays.

The first call to any kernel triggers Numba's JIT compilation
(typically 1-3 s on first import). ``cache=True`` persists the compiled
code under ``__pycache__`` so subsequent process starts are instant. If
you change a kernel signature, wipe ``__pycache__``.
"""

from __future__ import annotations

import numba
import numpy as np
from numba import njit

# ----------------------------------------------------------------------
# Inverted-index O(1) primitives (njit)
# ----------------------------------------------------------------------


@njit(cache=True, inline="always")
def _add_to_inv(inv, inv_len, inv_pos, q, r):
    pos = inv_len[q]
    inv[q, pos] = r
    inv_pos[r, q] = pos
    inv_len[q] += 1


@njit(cache=True, inline="always")
def _remove_from_inv(inv, inv_len, inv_pos, q, r):
    pos = inv_pos[r, q]
    if pos < 0:
        return
    last = inv_len[q] - 1
    if pos != last:
        j = inv[q, last]
        inv[q, pos] = j
        inv_pos[j, q] = pos
    inv_len[q] -= 1
    inv_pos[r, q] = -1


@njit(cache=True, inline="always")
def _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, q, r):
    pos = inv_x_len[q]
    inv_x[q, pos] = r
    inv_x_pos[r, q] = pos
    inv_x_len[q] += 1


@njit(cache=True, inline="always")
def _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, q, r):
    pos = inv_x_pos[r, q]
    if pos < 0:
        return
    last = inv_x_len[q] - 1
    if pos != last:
        j = inv_x[q, last]
        inv_x[q, pos] = j
        inv_x_pos[j, q] = pos
    inv_x_len[q] -= 1
    inv_x_pos[r, q] = -1


@njit(cache=True, inline="always")
def _add_to_support(supp_q, supp_len, supp_pos, r, q):
    pos = supp_len[r]
    supp_q[r, pos] = q
    supp_pos[r, q] = pos
    supp_len[r] += 1


@njit(cache=True, inline="always")
def _remove_from_support(supp_q, supp_len, supp_pos, r, q):
    pos = supp_pos[r, q]
    if pos < 0:
        return
    last = supp_len[r] - 1
    if pos != last:
        j = supp_q[r, last]
        supp_q[r, pos] = j
        supp_pos[r, j] = pos
    supp_len[r] -= 1
    supp_pos[r, q] = -1


# ----------------------------------------------------------------------
# 1-qubit kernel
# ----------------------------------------------------------------------


@njit(cache=True)
def apply_1q_kernel_jit(
    plt,
    inv,
    inv_len,
    inv_x,
    inv_x_len,
    inv_x_pos,
    q,
    lut,
):
    """LUT-driven 1q gate at qubit ``q``. ``lut`` is a (4,) uint8 array."""
    n_active = inv_len[q]
    for idx in range(n_active):
        r = inv[q, idx]
        old_xz = plt[r, q]
        if old_xz == 0:
            continue
        new_xz = lut[old_xz]
        if new_xz == old_xz:
            continue
        old_x = (old_xz >> 1) & 1
        new_x = (new_xz >> 1) & 1
        plt[r, q] = new_xz
        if new_x and not old_x:
            _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, q, r)
        elif not new_x and old_x:
            _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, q, r)


# ----------------------------------------------------------------------
# 2-qubit kernel
# ----------------------------------------------------------------------


@njit(cache=True)
def apply_2q_kernel_jit(
    plt,
    supp_q,
    supp_len,
    supp_pos,
    inv,
    inv_len,
    inv_pos,
    inv_x,
    inv_x_len,
    inv_x_pos,
    qi,
    qj,
    lut,
    active_buf,
):
    """LUT-driven 2q gate at ``(qi, qj)``.

    ``lut`` is a (16,) uint8 array indexed by the packed old Pauli
    ``(xz_i << 2) | xz_j`` and returning the packed new Pauli, so the inner loop
    is one lookup on the two ``plt`` bytes with no bit unpacking (built by
    :func:`sparsegf2.core.sparse_tableau._build_2q_lut`).

    ``active_buf`` is a pre-allocated int32 scratch buffer of length
    ``>= 2 * inv.shape[1]``.
    """
    n_active = 0
    for idx in range(inv_len[qi]):
        active_buf[n_active] = inv[qi, idx]
        n_active += 1
    for idx in range(inv_len[qj]):
        r = inv[qj, idx]
        if plt[r, qi] != 0:
            continue
        active_buf[n_active] = r
        n_active += 1

    for a_idx in range(n_active):
        r = active_buf[a_idx]
        xz_i = plt[r, qi]
        xz_j = plt[r, qj]

        # Single packed lookup: index (xz_i<<2)|xz_j -> (new_xz_i<<2)|new_xz_j.
        out = lut[(xz_i << 2) | xz_j]
        new_xz_i = (out >> 2) & 3
        new_xz_j = out & 3
        if new_xz_i == xz_i and new_xz_j == xz_j:
            continue  # generator unaffected by the gate

        old_xi = (xz_i >> 1) & 1
        new_xi = (new_xz_i >> 1) & 1
        old_xj = (xz_j >> 1) & 1
        new_xj = (new_xz_j >> 1) & 1

        # qi
        was_i = xz_i != 0
        now_i = new_xz_i != 0
        if now_i and not was_i:
            _add_to_inv(inv, inv_len, inv_pos, qi, r)
            _add_to_support(supp_q, supp_len, supp_pos, r, qi)
        elif not now_i and was_i:
            _remove_from_inv(inv, inv_len, inv_pos, qi, r)
            _remove_from_support(supp_q, supp_len, supp_pos, r, qi)
        if new_xi and not old_xi:
            _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, qi, r)
        elif not new_xi and old_xi:
            _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, qi, r)
        plt[r, qi] = new_xz_i

        # qj
        was_j = xz_j != 0
        now_j = new_xz_j != 0
        if now_j and not was_j:
            _add_to_inv(inv, inv_len, inv_pos, qj, r)
            _add_to_support(supp_q, supp_len, supp_pos, r, qj)
        elif not now_j and was_j:
            _remove_from_inv(inv, inv_len, inv_pos, qj, r)
            _remove_from_support(supp_q, supp_len, supp_pos, r, qj)
        if new_xj and not old_xj:
            _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, qj, r)
        elif not new_xj and old_xj:
            _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, qj, r)
        plt[r, qj] = new_xz_j


# ----------------------------------------------------------------------
# Sparse XOR and clear (njit)
# ----------------------------------------------------------------------


@njit(cache=True)
def sparse_xor_rows_jit(
    plt,
    supp_q,
    supp_len,
    supp_pos,
    inv,
    inv_len,
    inv_pos,
    inv_x,
    inv_x_len,
    inv_x_pos,
    target,
    source,
):
    src_len = supp_len[source]
    for idx in range(src_len):
        q = supp_q[source, idx]
        s_xz = plt[source, q]
        if s_xz == 0:
            continue
        s_x = (s_xz >> 1) & 1
        s_z = s_xz & 1
        t_xz = plt[target, q]
        old_t_x = (t_xz >> 1) & 1
        if t_xz != 0:
            new_x = old_t_x ^ s_x
            new_z = (t_xz & 1) ^ s_z
            new_xz = (new_x << 1) | new_z
            if new_xz != 0:
                plt[target, q] = new_xz
                if new_x and not old_t_x:
                    _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, q, target)
                elif not new_x and old_t_x:
                    _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, q, target)
            else:
                plt[target, q] = 0
                _remove_from_inv(inv, inv_len, inv_pos, q, target)
                if old_t_x:
                    _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, q, target)
                _remove_from_support(supp_q, supp_len, supp_pos, target, q)
        else:
            plt[target, q] = s_xz
            _add_to_inv(inv, inv_len, inv_pos, q, target)
            if s_x:
                _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, q, target)
            _add_to_support(supp_q, supp_len, supp_pos, target, q)


@njit(cache=True)
def clear_generator_jit(
    plt,
    supp_q,
    supp_len,
    supp_pos,
    inv,
    inv_len,
    inv_pos,
    inv_x,
    inv_x_len,
    inv_x_pos,
    r,
):
    L = supp_len[r]
    for idx in range(L):
        q = supp_q[r, idx]
        xz = plt[r, q]
        _remove_from_inv(inv, inv_len, inv_pos, q, r)
        if (xz >> 1) & 1:
            _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, q, r)
        plt[r, q] = 0
        supp_pos[r, q] = -1
    supp_len[r] = 0


# ----------------------------------------------------------------------
# Measurement kernel (njit)
# ----------------------------------------------------------------------


# Module-level uint8 constant so the @njit body below stores the right
# dtype into `plt` without an implicit Python-int -> uint8 coercion that
# can trip Numba's type-unification under nopython.
PAULI_Z_U8 = numba.uint8(1)


@njit(cache=True)
def measure_z_kernel_jit(
    plt,
    supp_q,
    supp_len,
    supp_pos,
    inv,
    inv_len,
    inv_pos,
    inv_x,
    inv_x_len,
    inv_x_pos,
    q,
    use_min_weight_pivot,
    others_buf,
):
    """Z-basis measurement of qubit ``q``.

    Returns 1 if non-deterministic (a stabilizer anticommuter existed),
    0 otherwise.

    Aaronson-Gottesman CHP, phase-free:

    * **Deterministic** (no stabilizer row ``r >= n`` has ``x_{r,q} = 1``):
      Stim leaves the symplectic tableau unchanged, so we do the same.
    * **Non-deterministic**: pick stabilizer pivot ``p``, XOR it into every
      other anticommuter, copy old row ``p`` into row ``p - n`` (destab
      slot), and overwrite row ``p`` with the pure ``Z_q`` generator.
    """
    n = plt.shape[1]
    n_anti = inv_x_len[q]

    pivot = -1
    if use_min_weight_pivot:
        best_wt = -1
        for k in range(n_anti):
            r = inv_x[q, k]
            if r >= n:
                w = supp_len[r]
                if best_wt < 0 or w < best_wt:
                    best_wt = w
                    pivot = r
    else:
        for k in range(n_anti):
            r = inv_x[q, k]
            if r >= n:
                pivot = r
                break

    if pivot < 0:
        return 0

    # Skip destab = pivot - n from the snapshot: we are about to clear
    # and overwrite it with pivot's content, so XOR'ing pivot into it
    # first would be wasted work (the clear+copy below undoes it).
    destab = pivot - n
    n_others = 0
    for k in range(n_anti):
        r = inv_x[q, k]
        if r != pivot and r != destab:
            others_buf[n_others] = r
            n_others += 1

    for k in range(n_others):
        sparse_xor_rows_jit(
            plt,
            supp_q,
            supp_len,
            supp_pos,
            inv,
            inv_len,
            inv_pos,
            inv_x,
            inv_x_len,
            inv_x_pos,
            others_buf[k],
            pivot,
        )

    clear_generator_jit(
        plt,
        supp_q,
        supp_len,
        supp_pos,
        inv,
        inv_len,
        inv_pos,
        inv_x,
        inv_x_len,
        inv_x_pos,
        destab,
    )
    sparse_xor_rows_jit(
        plt,
        supp_q,
        supp_len,
        supp_pos,
        inv,
        inv_len,
        inv_pos,
        inv_x,
        inv_x_len,
        inv_x_pos,
        destab,
        pivot,
    )

    clear_generator_jit(
        plt,
        supp_q,
        supp_len,
        supp_pos,
        inv,
        inv_len,
        inv_pos,
        inv_x,
        inv_x_len,
        inv_x_pos,
        pivot,
    )
    plt[pivot, q] = PAULI_Z_U8
    supp_q[pivot, 0] = q
    supp_len[pivot] = 1
    supp_pos[pivot, q] = 0
    _add_to_inv(inv, inv_len, inv_pos, q, pivot)
    return 1


# ----------------------------------------------------------------------
# Warmup
# ----------------------------------------------------------------------


def warmup() -> None:
    """Trigger JIT compilation of every kernel.

    Call this once at the start of a long-running session to amortize
    Numba's compilation latency (1-3 s) before any timing measurement.
    Subsequent runs hit the on-disk cache and pay only the import cost.

    The warmup constructs a proper |0^4⟩ tableau, applies real Cliffords
    (H, then CX), then measures Z_0. Because H followed by CX leaves
    a stabilizer with x-bit on qubit 0, the measurement enters the
    **non-deterministic** branch, so every code path in
    ``measure_z_kernel_jit`` (pivot search, XOR loop, destab copy,
    Z_q overwrite) is JIT-compiled here, not lazily on first real use.
    """
    n = 4
    plt = np.zeros((2 * n, n), dtype=np.uint8)
    supp_q = np.zeros((2 * n, max(1, n)), dtype=np.int32)  # = _max_support(n)
    supp_len = np.zeros(2 * n, dtype=np.int32)
    supp_pos = np.full((2 * n, n), -1, dtype=np.int32)
    mi = max(1, 2 * n)  # tight bound (was 2*n + 64)
    inv = np.zeros((n, mi), dtype=np.int32)
    inv_len = np.zeros(n, dtype=np.int32)
    inv_pos = np.full((2 * n, n), -1, dtype=np.int32)
    inv_x = np.zeros((n, mi), dtype=np.int32)
    inv_x_len = np.zeros(n, dtype=np.int32)
    inv_x_pos = np.full((2 * n, n), -1, dtype=np.int32)
    active_buf = np.zeros(2 * mi, dtype=np.int32)
    others_buf = np.zeros(mi, dtype=np.int32)

    # Init |0^4⟩: rows 0..3 = X_i (destabs), rows 4..7 = Z_i (stabs).
    for i in range(n):
        plt[i, i] = 2  # X
        supp_q[i, 0] = i
        supp_len[i] = 1
        supp_pos[i, i] = 0
        inv[i, 0] = i
        inv_pos[i, i] = 0
        inv_x[i, 0] = i
        inv_x_pos[i, i] = 0
        inv_len[i] = 1
        inv_x_len[i] = 1
        plt[n + i, i] = 1  # Z
        supp_q[n + i, 0] = i
        supp_len[n + i] = 1
        supp_pos[n + i, i] = 0
        inv[i, 1] = n + i
        inv_pos[n + i, i] = 1
        inv_len[i] = 2

    # H LUT: I=0 → I, Z=1 → X=2, X=2 → Z=1, Y=3 → Y=3.
    lut_h = np.array([0, 2, 1, 3], dtype=np.uint8)
    # CX LUT (computed offline from CX_SYMP via _build_2q_lut; verified):
    # new_xi = xi; new_xj = xi ⊕ xj; new_zi = zi ⊕ zj; new_zj = zj.
    lut_cx = np.array(
        [0, 3, 2, 1, 4, 7, 6, 5, 12, 15, 14, 13, 8, 11, 10, 9],
        dtype=np.uint8,
    )

    # H(0): X_0 ↔ Z_0 on q=0. Row 0 (was X_0) → Z_0; row 4 (was Z_0) → X_0.
    apply_1q_kernel_jit(plt, inv, inv_len, inv_x, inv_x_len, inv_x_pos, 0, lut_h)
    # CX(0, 1): entangle. Stab on q=0 now has x-bit ⇒ measure_z(0) is random.
    apply_2q_kernel_jit(
        plt,
        supp_q,
        supp_len,
        supp_pos,
        inv,
        inv_len,
        inv_pos,
        inv_x,
        inv_x_len,
        inv_x_pos,
        0,
        1,
        lut_cx,
        active_buf,
    )
    # Measure Z_0: exercises the non-deterministic branch (pivot search,
    # XOR loop, destab copy, Z_q overwrite). Use min-weight pivot mode
    # (use_min_weight_pivot=True).
    was_random = measure_z_kernel_jit(
        plt,
        supp_q,
        supp_len,
        supp_pos,
        inv,
        inv_len,
        inv_pos,
        inv_x,
        inv_x_len,
        inv_x_pos,
        0,
        True,
        others_buf,
    )
    assert was_random == 1, "warmup: measure_z(0) on H·CX|00⟩ should be non-deterministic"

    # Second measurement: Z_0 is now deterministic (we just measured it).
    # Exercises the deterministic branch of the measurement kernel;
    # otherwise that branch was lazily JIT-compiled on first real use.
    was_random2 = measure_z_kernel_jit(
        plt,
        supp_q,
        supp_len,
        supp_pos,
        inv,
        inv_len,
        inv_pos,
        inv_x,
        inv_x_len,
        inv_x_pos,
        0,
        True,
        others_buf,
    )
    assert was_random2 == 0, "warmup: second measure_z(0) should be deterministic"

    # Third measurement: pivot_mode = first (use_min_weight_pivot=False).
    # The plain-first pivot loop is structurally different from the
    # min-weight branch, so we exercise it too. Re-prepare the state.
    apply_1q_kernel_jit(plt, inv, inv_len, inv_x, inv_x_len, inv_x_pos, 1, lut_h)
    apply_2q_kernel_jit(
        plt,
        supp_q,
        supp_len,
        supp_pos,
        inv,
        inv_len,
        inv_pos,
        inv_x,
        inv_x_len,
        inv_x_pos,
        1,
        2,
        lut_cx,
        active_buf,
    )
    was_random3 = measure_z_kernel_jit(
        plt,
        supp_q,
        supp_len,
        supp_pos,
        inv,
        inv_len,
        inv_pos,
        inv_x,
        inv_x_len,
        inv_x_pos,
        1,
        False,  # use_min_weight_pivot = False (plain-first pivot branch)
        others_buf,
    )
    assert was_random3 == 1, "warmup: measure_z(1) on H·CX|0...⟩ should be non-deterministic"


# ----------------------------------------------------------------------
# GF(2) row reduction (njit): drives SparseGF2.canonical_form()
# ----------------------------------------------------------------------


@njit(cache=True)
def gf2_rref_jit(A):
    """Reduce ``A`` to GF(2) row-reduced echelon form in-place, return ``A``.

    Standard pivot-and-eliminate scan over uint8 entries. The JIT'd
    version is ~10-50× faster than the pure-Python reference for
    moderate ``A`` (which is what :meth:`SparseGF2.canonical_form`
    feeds in: an ``(n, 2n)`` uint8 matrix). The body is in-place to
    avoid a re-allocation per call.
    """
    n_rows, n_cols = A.shape
    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        # Find a pivot in this column at or below pivot_row.
        r_pivot = -1
        for r in range(pivot_row, n_rows):
            if A[r, col]:
                r_pivot = r
                break
        if r_pivot < 0:
            continue
        # Swap pivot row up to pivot_row (in-place row swap).
        if r_pivot != pivot_row:
            for c in range(n_cols):
                tmp = A[pivot_row, c]
                A[pivot_row, c] = A[r_pivot, c]
                A[r_pivot, c] = tmp
        # Eliminate column col in every other row (above and below pivot_row).
        for r in range(n_rows):
            if r != pivot_row and A[r, col]:
                for c in range(n_cols):
                    A[r, c] ^= A[pivot_row, c]
        pivot_row += 1
    return A


# ----------------------------------------------------------------------
# GF(2) kernel basis (njit): drives sparsegf2.core.symplectic.random_symplectic
# ----------------------------------------------------------------------


@njit(cache=True)
def gf2_kernel_basis_jit(M):
    """Return a basis of the right kernel of ``M`` over GF(2).

    The pure-Python version :func:`sparsegf2.core.linalg_gf2._gf2_kernel_basis_python`
    is the reference; this one is a literal `@njit` translation. Used
    inside :func:`sparsegf2.core.symplectic.random_symplectic` where it's
    called once per qubit step ⇒ O(n^3) bit-ops; the JIT version is the
    only way to keep `random_symplectic` fast at n ≳ 50.

    Parameters
    ----------
    M : ndarray of shape (m, k), uint8
        The matrix whose right kernel we want.

    Returns
    -------
    ndarray of shape (k - rank(M), k), uint8
        Rows form a basis of ``{v ∈ F_2^k : M @ v.T = 0 (mod 2)}``.
        When ``m == 0``, returns the identity of shape (k, k).
    """
    n_rows, n_cols = M.shape
    if n_rows == 0:
        return np.eye(n_cols, dtype=np.uint8)

    # In-place RREF.
    A = M.copy()
    # pivot_cols tracked as fixed-size int32 array; -1 sentinel = unused.
    pivot_cols = np.full(n_cols, -1, dtype=np.int32)
    n_pivots = 0
    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        # Find a 1 at or below pivot_row in column col.
        r_pivot = -1
        for r in range(pivot_row, n_rows):
            if A[r, col] != 0:
                r_pivot = r
                break
        if r_pivot < 0:
            continue
        # Swap rows pivot_row <-> r_pivot (in-place).
        if r_pivot != pivot_row:
            for c in range(n_cols):
                tmp = A[pivot_row, c]
                A[pivot_row, c] = A[r_pivot, c]
                A[r_pivot, c] = tmp
        # Eliminate column col in every other row.
        for r in range(n_rows):
            if r != pivot_row and A[r, col] != 0:
                for c in range(n_cols):
                    A[r, c] ^= A[pivot_row, c]
        pivot_cols[n_pivots] = col
        n_pivots += 1
        pivot_row += 1

    n_free = n_cols - n_pivots
    if n_free == 0:
        return np.zeros((0, n_cols), dtype=np.uint8)

    # Build kernel basis. For each free column free_col:
    #   v[free_col] = 1
    #   v[pivot_cols[r]] = A[r, free_col]  for r in 0..n_pivots-1
    # Locate the free columns by scanning [0, n_cols) and skipping the
    # pivot columns recorded in pivot_cols[:n_pivots].
    kernel = np.zeros((n_free, n_cols), dtype=np.uint8)
    free_idx = 0
    for col in range(n_cols):
        # Is `col` a pivot column?
        is_pivot = False
        for r in range(n_pivots):
            if pivot_cols[r] == col:
                is_pivot = True
                break
        if is_pivot:
            continue
        kernel[free_idx, col] = 1
        for r in range(n_pivots):
            kernel[free_idx, pivot_cols[r]] = A[r, col]
        free_idx += 1
    return kernel


# ----------------------------------------------------------------------
# Dense bit-packed kernels (hybrid mode)
# ----------------------------------------------------------------------
#
# When the stabilizer tableau densifies (volume-law), the inverted-index
# bookkeeping of the sparse path costs more than it saves. The hybrid mode then
# mirrors the tableau into two bit-packed uint64 arrays (``x_packed`` and
# ``z_packed``, each shape ``(2n, ceil(n/64))``) and applies gates/measurements
# in O(n words) per row with no index maintenance (the Stim-style dense path).
#
# Bit layout: generator ``r``, qubit ``q`` -> word ``q >> 6``, bit ``q & 63``
# (LSB-first within each uint64). ``x_packed`` holds the X-bits, ``z_packed`` the
# Z-bits. The packed Pauli on (r, q) is ``(x_bit << 1) | z_bit``, the same nibble
# the sparse PLT stores, so the dense gate kernels reuse the very same packed
# LUTs (index ``(xz_i<<2)|xz_j`` -> ``(new_xz_i<<2)|new_xz_j``, row-vector
# convention) as the sparse path.


@njit(cache=True)
def apply_2q_packed_jit(x_packed, z_packed, qi, qj, lut):
    """Dense 2q Clifford via the packed LUT. ``lut`` is the same (16,) uint8
    table the sparse path uses. Rewrites the two affected bits of every row."""
    n_rows = x_packed.shape[0]
    one = np.uint64(1)
    two = np.uint64(2)
    three = np.uint64(3)
    wi = qi >> 6
    wj = qj >> 6
    bi = np.uint64(qi & 63)
    bj = np.uint64(qj & 63)
    keep_i = ~(one << bi)
    keep_j = ~(one << bj)
    for r in range(n_rows):
        xi = (x_packed[r, wi] >> bi) & one
        zi = (z_packed[r, wi] >> bi) & one
        xj = (x_packed[r, wj] >> bj) & one
        zj = (z_packed[r, wj] >> bj) & one
        xz_i = (xi << one) | zi
        xz_j = (xj << one) | zj
        out = np.uint64(lut[(xz_i << two) | xz_j])
        new_xz_i = (out >> two) & three
        new_xz_j = out & three
        if new_xz_i == xz_i and new_xz_j == xz_j:
            continue
        nxi = (new_xz_i >> one) & one
        nzi = new_xz_i & one
        nxj = (new_xz_j >> one) & one
        nzj = new_xz_j & one
        # Masked writes are order-safe even when qi, qj share a word.
        x_packed[r, wi] = (x_packed[r, wi] & keep_i) | (nxi << bi)
        z_packed[r, wi] = (z_packed[r, wi] & keep_i) | (nzi << bi)
        x_packed[r, wj] = (x_packed[r, wj] & keep_j) | (nxj << bj)
        z_packed[r, wj] = (z_packed[r, wj] & keep_j) | (nzj << bj)


@njit(cache=True)
def apply_1q_packed_jit(x_packed, z_packed, q, lut):
    """Dense 1q Clifford via the packed (4,) LUT (index ``(x<<1)|z``)."""
    n_rows = x_packed.shape[0]
    one = np.uint64(1)
    w = q >> 6
    b = np.uint64(q & 63)
    keep = ~(one << b)
    for r in range(n_rows):
        x = (x_packed[r, w] >> b) & one
        z = (z_packed[r, w] >> b) & one
        new_xz = np.uint64(lut[(x << one) | z])
        if new_xz == ((x << one) | z):
            continue
        nx = (new_xz >> one) & one
        nz = new_xz & one
        x_packed[r, w] = (x_packed[r, w] & keep) | (nx << b)
        z_packed[r, w] = (z_packed[r, w] & keep) | (nz << b)


@njit(cache=True)
def measure_z_packed_jit(x_packed, z_packed, q):
    """Dense Z-measurement (phase-free CHP). Returns 1 if non-deterministic.

    Standard Aaronson-Gottesman update on the bit-packed tableau: pick a
    stabilizer pivot (row in ``[n, 2n)`` with the x-bit set on ``q``), XOR it into
    every other anticommuter, copy the old pivot into its destabilizer slot, and
    overwrite the pivot with the pure ``Z_q`` generator. Deterministic outcome
    (no stabilizer anticommutes) leaves the tableau unchanged. The chosen pivot
    differs from the sparse min-weight pivot, so the generators differ, but the
    stabilizer group is identical (verified by canonical_form / Stim)."""
    n_rows = x_packed.shape[0]
    n = n_rows >> 1
    n_words = x_packed.shape[1]
    one = np.uint64(1)
    w = q >> 6
    b = np.uint64(q & 63)
    pivot = -1
    for r in range(n, n_rows):  # stabilizer rows only
        if (x_packed[r, w] >> b) & one:
            pivot = r
            break
    if pivot == -1:
        return 0
    destab = pivot - n
    for r in range(n_rows):
        if r != pivot and r != destab and ((x_packed[r, w] >> b) & one):
            for jj in range(n_words):
                x_packed[r, jj] ^= x_packed[pivot, jj]
                z_packed[r, jj] ^= z_packed[pivot, jj]
    for jj in range(n_words):  # destab slot <- old pivot; pivot <- Z_q
        x_packed[destab, jj] = x_packed[pivot, jj]
        z_packed[destab, jj] = z_packed[pivot, jj]
        x_packed[pivot, jj] = np.uint64(0)
        z_packed[pivot, jj] = np.uint64(0)
    z_packed[pivot, w] = one << b
    return 1


@njit(cache=True)
def plt_to_packed_jit(plt, x_packed, z_packed, n):
    """Sync the bit-packed arrays from the sparse PLT (sparse -> dense)."""
    n_rows = plt.shape[0]
    one = np.uint64(1)
    x_packed[:, :] = np.uint64(0)
    z_packed[:, :] = np.uint64(0)
    for r in range(n_rows):
        for qq in range(n):
            xz = plt[r, qq]
            if xz != 0:
                w = qq >> 6
                b = np.uint64(qq & 63)
                if (xz >> 1) & 1:
                    x_packed[r, w] |= one << b
                if xz & 1:
                    z_packed[r, w] |= one << b


@njit(cache=True)
def packed_to_plt_jit(x_packed, z_packed, plt, n):
    """Sync the sparse PLT from the bit-packed arrays (dense -> sparse)."""
    n_rows = plt.shape[0]
    one = np.uint64(1)
    plt[:, :] = np.uint8(0)
    for r in range(n_rows):
        for qq in range(n):
            w = qq >> 6
            b = np.uint64(qq & 63)
            xb = (x_packed[r, w] >> b) & one
            zb = (z_packed[r, w] >> b) & one
            if xb or zb:
                plt[r, qq] = np.uint8((xb << one) | zb)


@njit(cache=True)
def row_weights_packed_jit(x_packed, z_packed, n):
    """Per-generator Pauli weight (popcount of x|z over the n columns) from the
    packed arrays: the dense-mode equivalent of ``supp_len``."""
    n_rows = x_packed.shape[0]
    n_words = x_packed.shape[1]
    one = np.uint64(1)
    weights = np.zeros(n_rows, dtype=np.int32)
    tail = n & 63
    tail_mask = ((one << np.uint64(tail)) - one) if tail != 0 else ~np.uint64(0)
    for r in range(n_rows):
        cnt = 0
        for jj in range(n_words):
            v = x_packed[r, jj] | z_packed[r, jj]
            if jj == n_words - 1:
                v &= tail_mask
            while v != 0:
                cnt += 1
                v &= v - one
        weights[r] = cnt
    return weights


@njit(cache=True)
def gf2_rank_words_jit(mat, n_cols):
    """Rank over GF(2) of a bit-packed matrix, by word-wise elimination.

    ``mat`` is ``(n_rows, n_words)`` uint64 with column ``c`` at bit ``c & 63``
    of word ``c >> 6`` (LSB-first, the same layout as ``x_packed``/``z_packed``
    and ``np.packbits(..., bitorder="little")``). **Destructive**: eliminates in
    place, so the caller passes a copy. Row XORs run on whole words, making this
    ~64x cheaper than the uint8-per-bit elimination in ``gf2_rref``. It exists
    for the per-layer order-parameter rank in ``until_purified`` runs, where the
    uint8 rank dominated production wall-clock at large ``n``.
    """
    n_rows = mat.shape[0]
    n_words = mat.shape[1]
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
            for j in range(n_words):
                tmp = mat[found, j]
                mat[found, j] = mat[pivot_row, j]
                mat[pivot_row, j] = tmp
        for r in range(n_rows):
            if r != pivot_row and ((mat[r, w] >> b) & one):
                for j in range(n_words):
                    mat[r, j] ^= mat[pivot_row, j]
        pivot_row += 1
    return pivot_row
