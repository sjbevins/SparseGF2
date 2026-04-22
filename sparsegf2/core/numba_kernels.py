"""
Numba JIT-compiled kernels for hot paths in stabilizer tableau simulation.

These operate directly on the raw uint64 data arrays from PackedBitMatrix,
bypassing Python method dispatch overhead. Each function takes the .data
attribute (ndarray of uint64) plus column/row indices.

Falls back gracefully if numba is not installed.
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # No-op decorator when numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@njit(cache=True)
def build_gate_lut(S):
    """Build 16-entry LUT for 4x4 GF(2) symplectic matrix.

    Input index: (xi << 3) | (xj << 2) | (zi << 1) | zj
    Output value: (xi' << 3) | (xj' << 2) | (zi' << 1) | zj'
    """
    lut = np.zeros(16, dtype=np.uint8)
    for inp in range(16):
        b0 = (inp >> 3) & 1
        b1 = (inp >> 2) & 1
        b2 = (inp >> 1) & 1
        b3 = inp & 1
        n0 = (b0*S[0,0] + b1*S[1,0] + b2*S[2,0] + b3*S[3,0]) & 1
        n1 = (b0*S[0,1] + b1*S[1,1] + b2*S[2,1] + b3*S[3,1]) & 1
        n2 = (b0*S[0,2] + b1*S[1,2] + b2*S[2,2] + b3*S[3,2]) & 1
        n3 = (b0*S[0,3] + b1*S[1,3] + b2*S[2,3] + b3*S[3,3]) & 1
        lut[inp] = (n0 << 3) | (n1 << 2) | (n2 << 1) | n3
    return lut


@njit(cache=True)
def apply_clifford_2q_packed(x_data, z_data, q0, q1, S):
    """Apply a 2-qubit Clifford gate directly on packed uint64 arrays.

    Parameters
    ----------
    x_data : uint64[:, :]
        The x-part packed data, shape (n_gen, n_words).
    z_data : uint64[:, :]
        The z-part packed data, shape (n_gen, n_words).
    q0, q1 : int
        The two qubit indices.
    S : uint8[:, :]
        The 4x4 GF(2) symplectic matrix.
    """
    w0 = q0 >> 6
    b0 = np.uint64(q0 & 63)
    w1 = q1 >> 6
    b1 = np.uint64(q1 & 63)
    mask0 = np.uint64(1) << b0
    mask1 = np.uint64(1) << b1
    one = np.uint64(1)

    lut = build_gate_lut(S)

    n_gen = x_data.shape[0]
    for r in range(n_gen):
        # Extract 4 bits
        xq0 = (x_data[r, w0] >> b0) & one
        xq1 = (x_data[r, w1] >> b1) & one
        zq0 = (z_data[r, w0] >> b0) & one
        zq1 = (z_data[r, w1] >> b1) & one

        # LUT lookup: single table access replaces 16 muls + 12 XORs
        inp = int((xq0 << 3) | (xq1 << 2) | (zq0 << 1) | zq1)
        out = lut[inp]
        n0 = np.uint64((out >> 3) & 1)
        n1 = np.uint64((out >> 2) & 1)
        n2 = np.uint64((out >> 1) & 1)
        n3 = np.uint64(out & 1)

        # Write back (clear then set)
        x_data[r, w0] = (x_data[r, w0] & ~mask0) | (n0 << b0)
        x_data[r, w1] = (x_data[r, w1] & ~mask1) | (n1 << b1)
        z_data[r, w0] = (z_data[r, w0] & ~mask0) | (n2 << b0)
        z_data[r, w1] = (z_data[r, w1] & ~mask1) | (n3 << b1)


@njit(cache=True)
def measure_z_packed(x_data, z_data, q):
    """Z-basis measurement with reset on packed uint64 arrays.

    Parameters
    ----------
    x_data : uint64[:, :]
        The x-part packed data.
    z_data : uint64[:, :]
        The z-part packed data.
    q : int
        The qubit to measure.
    """
    w = q >> 6
    b = np.uint64(q & 63)
    one = np.uint64(1)
    mask = one << b
    n_gen = x_data.shape[0]
    n_words = x_data.shape[1]

    # Find first anticommuting row (x[r, q] == 1)
    pivot = -1
    for r in range(n_gen):
        if (x_data[r, w] >> b) & one:
            pivot = r
            break

    if pivot == -1:
        return  # Already eigenstate

    # XOR pivot into all other anticommuting rows
    for r in range(n_gen):
        if r != pivot and ((x_data[r, w] >> b) & one):
            for j in range(n_words):
                x_data[r, j] ^= x_data[pivot, j]
                z_data[r, j] ^= z_data[pivot, j]

    # Replace pivot with Z_q
    for j in range(n_words):
        x_data[pivot, j] = np.uint64(0)
        z_data[pivot, j] = np.uint64(0)
    z_data[pivot, w] = mask


@njit(cache=True)
def gf2_rank_packed(data, n_cols):
    """GF(2) rank of a packed uint64 matrix.

    Parameters
    ----------
    data : uint64[:, :]
        Packed matrix data, shape (n_rows, n_words).
    n_cols : int
        Number of logical columns.

    Returns
    -------
    int
        The rank.
    """
    n_rows = data.shape[0]
    n_words = data.shape[1]
    # Work on a copy
    mat = data.copy()
    one = np.uint64(1)
    pivot_row = 0

    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        cw = col >> 6
        cb = np.uint64(col & 63)

        # Find pivot
        found = -1
        for r in range(pivot_row, n_rows):
            if (mat[r, cw] >> cb) & one:
                found = r
                break
        if found == -1:
            continue

        # Swap
        if found != pivot_row:
            for j in range(n_words):
                mat[found, j], mat[pivot_row, j] = mat[pivot_row, j], mat[found, j]

        # Eliminate
        for r in range(n_rows):
            if r != pivot_row and ((mat[r, cw] >> cb) & one):
                for j in range(n_words):
                    mat[r, j] ^= mat[pivot_row, j]

        pivot_row += 1

    return pivot_row


# ═══════════════════════════════════════════════════════════════
# Hybrid dense/sparse conversion kernels
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def plt_to_packed(plt, x_packed, z_packed, n, N):
    """Convert PLT (uint8) to bit-packed x/z arrays (uint64). O(n*N)."""
    one = np.uint64(1)
    n_words = x_packed.shape[1]
    for r in range(N):
        for w in range(n_words):
            x_packed[r, w] = np.uint64(0)
            z_packed[r, w] = np.uint64(0)
    for r in range(N):
        for q in range(n):
            xz = plt[r, q]
            if xz != 0:
                w = q >> 6
                b = np.uint64(q & 63)
                if (xz >> 1) & 1:
                    x_packed[r, w] |= one << b
                if xz & 1:
                    z_packed[r, w] |= one << b


@njit(cache=True)
def packed_to_plt(x_packed, z_packed, plt, n, N):
    """Convert bit-packed x/z arrays to PLT. O(n*N)."""
    one = np.uint64(1)
    for r in range(N):
        for q in range(n):
            plt[r, q] = 0
    for r in range(N):
        for q in range(n):
            w = q >> 6
            b = np.uint64(q & 63)
            x_bit = int((x_packed[r, w] >> b) & one)
            z_bit = int((z_packed[r, w] >> b) & one)
            if x_bit or z_bit:
                plt[r, q] = (x_bit << 1) | z_bit


@njit(cache=True)
def rebuild_indices_from_plt(plt, supp_q, supp_len, supp_pos,
                             inv, inv_len, inv_x, inv_x_len,
                             inv_pos, inv_x_pos, n, N):
    """Rebuild ALL sparse indices from PLT. O(n*N). Called on dense->sparse transition."""
    # Clear all indices
    for q in range(n):
        inv_len[q] = 0
        inv_x_len[q] = 0
    for r in range(N):
        supp_len[r] = 0
        for q in range(n):
            supp_pos[r, q] = -1
            inv_pos[r, q] = -1
            inv_x_pos[r, q] = -1

    # Rebuild from PLT
    for r in range(N):
        for q in range(n):
            xz = plt[r, q]
            if xz != 0:
                # Add to inv[q]
                pos_inv = inv_len[q]
                inv[q, pos_inv] = r
                inv_pos[r, q] = pos_inv
                inv_len[q] += 1

                # Add to inv_x[q] if x-bit set
                if (xz >> 1) & 1:
                    pos_x = inv_x_len[q]
                    inv_x[q, pos_x] = r
                    inv_x_pos[r, q] = pos_x
                    inv_x_len[q] += 1

                # Add to supp[r]
                pos_s = supp_len[r]
                supp_q[r, pos_s] = q
                supp_pos[r, q] = pos_s
                supp_len[r] += 1


@njit(cache=True)
def compute_k_packed(x_packed, z_packed, n, N):
    """Compute code dimension k from packed arrays. Returns rank - n."""
    n_sys_cols = 2 * n
    n_out_words = (n_sys_cols + 63) >> 6
    n_src_words = (n + 63) >> 6
    one = np.uint64(1)

    mat = np.zeros((N, n_out_words), dtype=np.uint64)

    # X columns (0..n-1): direct word copy from x_packed
    n_full_words = n >> 6
    tail_bits = n & 63
    tail_mask = (one << np.uint64(tail_bits)) - one if tail_bits else ~np.uint64(0)

    for r in range(N):
        for w in range(n_full_words):
            mat[r, w] = x_packed[r, w]
        if tail_bits:
            mat[r, n_full_words] = x_packed[r, n_full_words] & tail_mask

        # Z columns (n..2n-1): bit-shift copy from z_packed
        z_dst_start = n
        z_dst_w0 = z_dst_start >> 6
        z_dst_shift = z_dst_start & 63

        if z_dst_shift == 0:
            for w in range(n_full_words):
                mat[r, z_dst_w0 + w] = z_packed[r, w]
            if tail_bits:
                mat[r, z_dst_w0 + n_full_words] |= z_packed[r, n_full_words] & tail_mask
        else:
            shift = np.uint64(z_dst_shift)
            rshift = np.uint64(64 - z_dst_shift)
            for sw in range(n_src_words):
                src = z_packed[r, sw]
                if sw == n_full_words and tail_bits:
                    src &= tail_mask
                dw = z_dst_w0 + sw
                mat[r, dw] |= src << shift
                if dw + 1 < n_out_words:
                    mat[r, dw + 1] |= src >> rshift

    rank = gf2_rank_packed(mat, n_sys_cols)
    return rank - n


@njit(cache=True)
def run_layer_dense(x_packed, z_packed, gate_qi, gate_qj, gate_symp,
                    n_gates, meas_qubits, n_meas):
    """Run one full layer in dense mode: all gates then all measurements."""
    for g in range(n_gates):
        apply_clifford_2q_packed(x_packed, z_packed,
                                gate_qi[g], gate_qj[g], gate_symp[g])
    for m in range(n_meas):
        measure_z_packed(x_packed, z_packed, meas_qubits[m])
