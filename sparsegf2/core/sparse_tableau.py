"""
SparseGF2: sparse stabilizer simulator with PLT, inverted indices, and
min-weight pivot selection.

Implements the CHP algorithm of Aaronson & Gottesman, "Improved Simulation
of Stabilizer Circuits", Phys. Rev. A 70, 052328 (2004),
arXiv:quant-ph/0406196. Gate updates (Table I there) are applied here
via 2x2 and 4x4 GF(2) symplectic matrices under the row-vector convention
(x', z') = (x, z) @ S; measurement follows Sec. V (rank-1 branch when an
anticommuting stabilizer exists, deterministic branch otherwise).

Phase tracking is not kept: the simulator represents the GF(2) symplectic
part only. This is correct for stabilizer-group-level observables (rank,
entropy via Fattal-Cubitt-Yamamoto-Bravyi-Chuang, code distance, weight
spectra) and for the MIPT observables used throughout sparsegf2.circuits.

Data structures (specification: docs/reference_manual.tex, Chapters 3-5):

1. Dense Pauli Lookup Table (PLT): 2n x n uint8 array, O(1) random access
   to any (generator, qubit) Pauli entry.
2. Position maps (inv_pos, inv_x_pos): 2n x n int32 arrays, O(1) removal
   from inverted indices.
3. Sparse support lists (supp_q, supp_len): per-generator list of qubits
   in its support.
4. Min-weight pivot: select lightest anticommuting generator during
   measurement, reducing per-layer measurement cost from O(n*p*abar^2)
   to O(n*abar).

Memory: O(n^2) allocated (PLT + position maps), O(n*abar) touched per layer.

The default mode is pure-sparse (no automatic mode switching). Pass
``hybrid_mode=True`` to enable sparse<->dense mode switching based on the
average active count a_bar.
"""
import numpy as np
import numba

from sparsegf2.core.numba_kernels import (
    apply_clifford_2q_packed, measure_z_packed, build_gate_lut,
    plt_to_packed, packed_to_plt, rebuild_indices_from_plt,
    compute_k_packed, run_layer_dense, gf2_rank_packed,
)

def _max_support(n):
    """Max qubits per generator. Must be > n so transients that reach
    weight n don't overflow ``supp_q[r, pos]`` (Numba does not bounds-check)."""
    return n + 1

def _max_inv(n):
    """Max generators per qubit in inverted index. Must be >= 2n."""
    return 2 * n + 64


# ═══════════════════════════════════════════════════════════════
# Initialization
# ═══════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _init_bell_pairs(plt, supp_q, supp_len, supp_pos, inv, inv_len, inv_x, inv_x_len,
                        inv_pos, inv_x_pos, n):
    """Initialize state: g_i = X_i (destabilizers), g_{n+i} = Z_i (stabilizers)."""
    N = 2 * n
    for r in range(N):
        supp_len[r] = 0
        for q in range(n):
            plt[r, q] = 0
            supp_pos[r, q] = -1
            inv_pos[r, q] = -1
            inv_x_pos[r, q] = -1
    for q in range(n):
        inv_len[q] = 0
        inv_x_len[q] = 0

    for i in range(n):
        # g_i = X_{s_i}: xz = (1<<1)|0 = 2
        plt[i, i] = 2
        supp_q[i, 0] = i
        supp_len[i] = 1
        supp_pos[i, i] = 0

        pos = inv_len[i]
        inv[i, pos] = i
        inv_pos[i, i] = pos
        inv_len[i] += 1

        pos_x = inv_x_len[i]
        inv_x[i, pos_x] = i
        inv_x_pos[i, i] = pos_x
        inv_x_len[i] += 1

        # g_{n+i} = Z_{s_i}: xz = (0<<1)|1 = 1
        plt[n + i, i] = 1
        supp_q[n + i, 0] = i
        supp_len[n + i] = 1
        supp_pos[n + i, i] = 0

        pos = inv_len[i]
        inv[i, pos] = n + i
        inv_pos[n + i, i] = pos
        inv_len[i] += 1
        # Z has x=0, no inv_x entry


# ═══════════════════════════════════════════════════════════════
# O(1) inverted-index operations via position maps
# ═══════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _add_to_inv(inv, inv_len, inv_pos, q, r):
    pos = inv_len[q]
    inv[q, pos] = r
    inv_pos[r, q] = pos
    inv_len[q] += 1

@numba.njit(cache=True)
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

@numba.njit(cache=True)
def _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, q, r):
    pos = inv_x_len[q]
    inv_x[q, pos] = r
    inv_x_pos[r, q] = pos
    inv_x_len[q] += 1

@numba.njit(cache=True)
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


# ═══════════════════════════════════════════════════════════════
# Support list operations
# ═══════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _add_to_support(supp_q, supp_len, supp_pos, r, q):
    """Add qubit q to generator r's support list. O(1)."""
    pos = supp_len[r]
    supp_q[r, pos] = q
    supp_pos[r, q] = pos
    supp_len[r] += 1

@numba.njit(cache=True)
def _remove_from_support(supp_q, supp_len, supp_pos, r, q):
    """Remove qubit q from generator r's support list. O(1) via position map."""
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


# LUT builder imported at module level as build_gate_lut
_build_gate_lut = build_gate_lut


# ═══════════════════════════════════════════════════════════════
# Single-qubit gate application: O(a_q) per gate
# ═══════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _apply_gate_1q_kernel(plt, inv_x, inv_x_len, inv_x_pos, inv, inv_len, q, S2):
    """Apply 1-qubit Clifford gate via 2x2 symplectic matrix S2 on qubit q.

    S2 maps (x_q, z_q) -> S2 @ (x_q, z_q) mod 2.
    Only updates PLT and inv_x (support/inv unchanged since gate preserves support).
    """
    # Build 4-entry LUT: input = (x<<1)|z, output = (x'<<1)|z'
    lut = np.empty(4, dtype=np.uint8)
    for inp in range(4):
        bx = (inp >> 1) & 1
        bz = inp & 1
        nx = (bx * S2[0, 0] + bz * S2[1, 0]) & 1
        nz = (bx * S2[0, 1] + bz * S2[1, 1]) & 1
        lut[inp] = (nx << 1) | nz

    # Process all generators with support on q
    for idx in range(inv_len[q]):
        r = inv[q, idx]
        old_xz = plt[r, q]
        if old_xz == 0:
            continue  # shouldn't happen but safety
        new_xz = lut[old_xz]
        if new_xz == old_xz:
            continue  # no change
        # new_xz != 0 guaranteed because S2 is invertible (Clifford) and old_xz != 0
        old_x = (old_xz >> 1) & 1
        new_x = (new_xz >> 1) & 1
        plt[r, q] = new_xz
        if new_x and not old_x:
            _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, q, r)
        elif not new_x and old_x:
            _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, q, r)


# Standard 2x2 symplectic matrices for common single-qubit Cliffords.
# Row-vector convention: (x', z') = (x, z) @ S. Aaronson-Gottesman 2004 Table I.
_H_SYMP = np.array([[0, 1], [1, 0]], dtype=np.uint8)       # H: X<->Z
_S_SYMP = np.array([[1, 1], [0, 1]], dtype=np.uint8)       # S: X->Y, Z->Z  (x'=x, z'=x^z)
_SQRT_X_SYMP = np.array([[1, 0], [1, 1]], dtype=np.uint8)  # sqrt(X): X->X, Z->Y (x'=x^z, z'=z)

# CX (CNOT) 4x4 symplectic matrix: X_c->X_c X_t, Z_t->Z_c Z_t
_CX_SYMP = np.array([[1,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,1]], dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════
# Specialized single-qubit gate kernels (H, S)
# ═══════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _apply_h_kernel(plt, inv, inv_len, inv_x, inv_x_len, inv_pos, inv_x_pos, q):
    """Apply Hadamard gate on qubit q by swapping X<->Z bits.

    H swaps X and Z: PLT encoding 2 (X) <-> 1 (Z), 3 (Y) unchanged.
    Since H swaps which generators have X-support vs Z-only support,
    we swap the inv_x index lists rather than updating entry by entry.

    O(a_q) to update PLT entries, O(1) to swap index arrays.
    """
    # Update PLT entries: swap X and Z bits for all generators on qubit q
    for idx in range(inv_len[q]):
        r = inv[q, idx]
        xz = plt[r, q]
        # xz encoding: (x<<1)|z. Swap x,z: new = (z<<1)|x
        x_bit = (xz >> 1) & 1
        z_bit = xz & 1
        plt[r, q] = (z_bit << 1) | x_bit

    # Swap inv_x and inv_x_len for qubit q.
    # After H, generators that had X-support now have Z-only, and vice versa.
    # Rather than removing/adding entries one by one, we swap the arrays.
    # inv_x[q] becomes the set of generators that had Z-only (now have X),
    # and the old inv_x[q] generators now have Z-only.
    #
    # Build new inv_x from scratch: generators with x-bit set after the swap.
    # After the PLT update above, the x-bit is the OLD z-bit.
    # A generator has new x-bit=1 iff it had old z-bit=1, i.e., old xz in {1, 3}.
    # Old inv_x had generators with old x-bit=1, i.e., old xz in {2, 3}.
    # So we need: generators with old xz in {1, 3} = Z or Y.
    #
    # Strategy: clear inv_x for q, rebuild from inv[q] by checking new PLT.
    # This is O(a_q) which matches the PLT update cost.

    # Clear all inv_x positions for qubit q
    old_inv_x_len = inv_x_len[q]
    for idx in range(old_inv_x_len):
        r = inv_x[q, idx]
        inv_x_pos[r, q] = -1
    inv_x_len[q] = 0

    # Rebuild inv_x from current PLT values
    for idx in range(inv_len[q]):
        r = inv[q, idx]
        new_xz = plt[r, q]
        if (new_xz >> 1) & 1:  # new x-bit is set
            _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, q, r)


@numba.njit(cache=True)
def _apply_s_kernel(plt, inv, inv_len, inv_x, inv_x_len, inv_x_pos, q):
    """Apply the S (phase) gate on qubit q: x' = x, z' = x XOR z.

    Follows Aaronson-Gottesman 2004, Table I (S row): the Pauli action is
      S X S^dagger = Y,  S Y S^dagger = -X,  S Z S^dagger = Z.
    In GF(2) symplectic form (phase-free) this is (x, z) -> (x, x^z):
      X (1,0) -> (1,1) = Y,  Y (1,1) -> (1,0) = X,  Z (0,1) -> (0,1) = Z.
    Equivalently ``(x, z) @ [[1, 1], [0, 1]] = (x, x+z)`` under the
    row-vector convention used throughout this file.

    Only the z-bit of generators with x=1 on qubit q changes; x is
    preserved for all generators. The x-support set (inv_x) is unchanged,
    and no generator gains or loses total support on q.

    Cost: O(inv_x_len[q]) -- iterate only the x-support list.
    """
    for idx in range(inv_x_len[q]):
        r = inv_x[q, idx]
        xz = plt[r, q]
        # x=1 is guaranteed by walking inv_x[q]; flip z.
        new_z = (xz & 1) ^ 1
        plt[r, q] = (1 << 1) | new_z  # x stays 1, z flipped


# ═══════════════════════════════════════════════════════════════
# Specialized CX (CNOT) kernel
# ═══════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _apply_cx_kernel(plt, supp_q, supp_len, supp_pos, inv, inv_len, inv_x, inv_x_len,
                     inv_pos, inv_x_pos, qc, qt, active_buf):
    """Apply CX (CNOT) with control qc, target qt.

    CX transformation (symplectic):
      x_target' = x_target XOR x_control
      z_control' = z_control XOR z_target
      x_control' = x_control (unchanged)
      z_target' = z_target (unchanged)

    PLT encoding: (x<<1)|z. For each generator r touching qc or qt:
      old_xc, old_zc from plt[r, qc]
      old_xt, old_zt from plt[r, qt]
      new_xt = old_xt ^ old_xc
      new_zc = old_zc ^ old_zt
      new_xc = old_xc (unchanged)
      new_zt = old_zt (unchanged)

    Index updates needed when support changes:
    - Generator gains support on qt: was I on qt, now has nonzero entry
    - Generator loses support on qt: had nonzero entry, now I on qt
    - Generator gains support on qc: was I on qc, now has nonzero entry
    - Generator loses support on qc: had nonzero entry, now I on qc

    active_buf: preallocated int32 scratch buffer, length >= 2 * inv.shape[1].
    """
    # Collect all generators touching qc OR qt (deduplicated)
    n_active = 0
    for idx in range(inv_len[qc]):
        active_buf[n_active] = inv[qc, idx]
        n_active += 1
    for idx in range(inv_len[qt]):
        r = inv[qt, idx]
        # Deduplicate: if r has support on qc, it's already in active_buf
        if plt[r, qc] != 0:
            continue
        active_buf[n_active] = r
        n_active += 1

    for a_idx in range(n_active):
        r = active_buf[a_idx]

        xz_c = plt[r, qc]
        xz_t = plt[r, qt]
        old_xc = (xz_c >> 1) & 1
        old_zc = xz_c & 1
        old_xt = (xz_t >> 1) & 1
        old_zt = xz_t & 1

        # CX transformation
        new_xt = old_xt ^ old_xc
        new_zc = old_zc ^ old_zt
        new_xc = old_xc      # unchanged
        new_zt = old_zt       # unchanged

        new_xz_c = (new_xc << 1) | new_zc
        new_xz_t = (new_xt << 1) | new_zt

        # --- Update control qubit qc ---
        was_present_c = (xz_c != 0)
        now_present_c = (new_xz_c != 0)

        if now_present_c and not was_present_c:
            _add_to_inv(inv, inv_len, inv_pos, qc, r)
            _add_to_support(supp_q, supp_len, supp_pos, r, qc)
        elif not now_present_c and was_present_c:
            _remove_from_inv(inv, inv_len, inv_pos, qc, r)
            _remove_from_support(supp_q, supp_len, supp_pos, r, qc)

        # x_control is unchanged, so inv_x[qc] only changes if z flipped
        # and that caused x to appear/disappear. But x_control is unchanged,
        # so inv_x[qc] never changes. However, the z-bit did change, so
        # we need to check: did the generator gain or lose presence on qc?
        # That's handled above. inv_x tracks x-bit, which is unchanged for qc.
        # So NO inv_x update needed for qc.

        plt[r, qc] = new_xz_c

        # --- Update target qubit qt ---
        was_present_t = (xz_t != 0)
        now_present_t = (new_xz_t != 0)

        if now_present_t and not was_present_t:
            _add_to_inv(inv, inv_len, inv_pos, qt, r)
            _add_to_support(supp_q, supp_len, supp_pos, r, qt)
        elif not now_present_t and was_present_t:
            _remove_from_inv(inv, inv_len, inv_pos, qt, r)
            _remove_from_support(supp_q, supp_len, supp_pos, r, qt)

        # inv_x for target: x_target changed (new_xt = old_xt ^ old_xc)
        if new_xt and not old_xt:
            _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, qt, r)
        elif not new_xt and old_xt:
            _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, qt, r)

        plt[r, qt] = new_xz_t


# ═══════════════════════════════════════════════════════════════
# Two-qubit gate application: O(a_bar) per gate
# ═══════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _apply_gate_kernel(plt, supp_q, supp_len, supp_pos, inv, inv_len, inv_x, inv_x_len,
                   inv_pos, inv_x_pos, qi, qj, S, active_buf):
    """Apply 2-qubit Clifford gate. O(a_bar) per gate.

    active_buf: preallocated int32 scratch buffer, length >= 2 * inv.shape[1].
    Passed in from SparseGF2 (self._active_buf) to avoid np.empty per-call.
    """
    # Build LUT once per gate — replaces per-generator 4x4 GF(2) multiply
    lut = _build_gate_lut(S)

    # Collect active generators via inverted index using caller-provided buffer
    n_active = 0

    for idx in range(inv_len[qi]):
        active_buf[n_active] = inv[qi, idx]
        n_active += 1

    for idx in range(inv_len[qj]):
        r = inv[qj, idx]
        # Deduplicate: check if r already active via PLT
        if plt[r, qi] != 0:
            continue
        active_buf[n_active] = r
        n_active += 1

    for a_idx in range(n_active):
        r = active_buf[a_idx]

        # O(1) Pauli read from PLT
        xz_i = plt[r, qi]
        xz_j = plt[r, qj]
        old_xi = (xz_i >> 1) & 1
        old_zi = xz_i & 1
        old_xj = (xz_j >> 1) & 1
        old_zj = xz_j & 1

        # LUT-based 4-bit symplectic transform (replaces 16 multiplies + 12 XORs)
        inp = (old_xi << 3) | (old_xj << 2) | (old_zi << 1) | old_zj
        out = lut[inp]
        new_xi = (out >> 3) & 1
        new_xj = (out >> 2) & 1
        new_zi = (out >> 1) & 1
        new_zj = out & 1

        new_xz_i = (new_xi << 1) | new_zi
        new_xz_j = (new_xj << 1) | new_zj

        # --- Update qubit qi ---
        was_present_i = (xz_i != 0)
        now_present_i = (new_xz_i != 0)

        if now_present_i and not was_present_i:
            _add_to_inv(inv, inv_len, inv_pos, qi, r)
            _add_to_support(supp_q, supp_len, supp_pos, r, qi)
        elif not now_present_i and was_present_i:
            _remove_from_inv(inv, inv_len, inv_pos, qi, r)
            _remove_from_support(supp_q, supp_len, supp_pos, r, qi)

        if new_xi and not old_xi:
            _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, qi, r)
        elif not new_xi and old_xi:
            _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, qi, r)

        plt[r, qi] = new_xz_i

        # --- Update qubit qj ---
        was_present_j = (xz_j != 0)
        now_present_j = (new_xz_j != 0)

        if now_present_j and not was_present_j:
            _add_to_inv(inv, inv_len, inv_pos, qj, r)
            _add_to_support(supp_q, supp_len, supp_pos, r, qj)
        elif not now_present_j and was_present_j:
            _remove_from_inv(inv, inv_len, inv_pos, qj, r)
            _remove_from_support(supp_q, supp_len, supp_pos, r, qj)

        if new_xj and not old_xj:
            _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, qj, r)
        elif not new_xj and old_xj:
            _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, qj, r)

        plt[r, qj] = new_xz_j


# ═══════════════════════════════════════════════════════════════
# Sparse XOR: O(wt(source)) per eliminated row
# ═══════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _sparse_xor_rows(plt, supp_q, supp_len, supp_pos, inv, inv_len, inv_x, inv_x_len,
                        inv_pos, inv_x_pos, target, source):
    """XOR source into target. Iterates over source's support list: O(wt(source))."""
    src_len = supp_len[source]
    for idx in range(src_len):
        q = supp_q[source, idx]
        s_xz = plt[source, q]
        if s_xz == 0:
            continue
        s_x = (s_xz >> 1) & 1

        t_xz = plt[target, q]
        old_t_x = (t_xz >> 1) & 1

        if t_xz != 0:
            new_x = old_t_x ^ s_x
            new_z = (t_xz & 1) ^ (s_xz & 1)
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


# ═══════════════════════════════════════════════════════════════
# Clear generator
# ═══════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _clear_generator(plt, supp_q, supp_len, supp_pos, inv, inv_len, inv_x, inv_x_len,
                        inv_pos, inv_x_pos, r):
    """Remove all support of generator r. O(wt(r))."""
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


# ═══════════════════════════════════════════════════════════════
# Measurement with minimum-weight pivot
# ═══════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _measure_z_kernel(plt, supp_q, supp_len, supp_pos, inv, inv_len, inv_x, inv_x_len,
                  inv_pos, inv_x_pos, q, use_min_weight_pivot,
                  comm_buf, others_buf):
    """Measure Z_q. Optional min-weight pivot.

    comm_buf, others_buf: preallocated int32 scratch buffers, each length >= inv.shape[1].
    Passed in from SparseGF2 (self._comm_buf, self._others_buf) to avoid np.empty per-call.
    """
    n_anti = inv_x_len[q]
    if n_anti == 0:
        # Deterministic: Z_q is already in the stabilizer group.
        # Canonicalization: ensure exactly one generator is the pure Z_q operator.
        # This is intentional for the sparse representation — it keeps inverted
        # indices clean for subsequent operations. The dense path skips this
        # (returns early) since it doesn't maintain inverted indices.
        # Early return when no canonicalization needed (0 or 1 Z-generators):
        n_comm = 0
        for idx in range(inv_len[q]):
            r = inv[q, idx]
            xz = plt[r, q]
            if (xz & 1) == 1:  # z-bit set
                comm_buf[n_comm] = r
                n_comm += 1
        if n_comm <= 1:
            return  # Already canonical (0 or 1 Z-generators on this qubit)
        pivot = comm_buf[0]
        for k in range(1, n_comm):
            _sparse_xor_rows(plt, supp_q, supp_len, supp_pos, inv, inv_len,
                                inv_x, inv_x_len, inv_pos, inv_x_pos,
                                comm_buf[k], pivot)
        _clear_generator(plt, supp_q, supp_len, supp_pos, inv, inv_len,
                           inv_x, inv_x_len, inv_pos, inv_x_pos, pivot)
        plt[pivot, q] = 1
        supp_q[pivot, 0] = q
        supp_len[pivot] = 1
        supp_pos[pivot, q] = 0
        _add_to_inv(inv, inv_len, inv_pos, q, pivot)
        return

    # Non-deterministic case
    if use_min_weight_pivot and n_anti > 1:
        best_r = inv_x[q, 0]
        best_wt = supp_len[best_r]
        for k in range(1, n_anti):
            r = inv_x[q, k]
            w = supp_len[r]
            if w < best_wt:
                best_wt = w
                best_r = r
        pivot = best_r
    else:
        pivot = inv_x[q, 0]

    # Copy anticommuting list (XOR modifies inv_x) into caller-provided buffer
    n_others = 0
    for k in range(n_anti):
        r = inv_x[q, k]
        if r != pivot:
            others_buf[n_others] = r
            n_others += 1

    # XOR pivot into all others
    for k in range(n_others):
        _sparse_xor_rows(plt, supp_q, supp_len, supp_pos, inv, inv_len,
                            inv_x, inv_x_len, inv_pos, inv_x_pos,
                            others_buf[k], pivot)

    # Replace pivot with Z_q
    _clear_generator(plt, supp_q, supp_len, supp_pos, inv, inv_len,
                       inv_x, inv_x_len, inv_pos, inv_x_pos, pivot)
    plt[pivot, q] = 1  # z=1, x=0
    supp_q[pivot, 0] = q
    supp_len[pivot] = 1
    supp_pos[pivot, q] = 0
    _add_to_inv(inv, inv_len, inv_pos, q, pivot)


# ═══════════════════════════════════════════════════════════════
# Bit manipulation helper (used by compute_subsystem_entropy)
# GF(2) rank is now consolidated in numba_kernels.gf2_rank_packed
# ═══════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _set_bit(packed, row, col, val):
    w = col >> 6
    b = numba.uint64(col & 63)
    mask = numba.uint64(1) << b
    if val:
        packed[row, w] |= mask
    else:
        packed[row, w] &= ~mask


# ═══════════════════════════════════════════════════════════════
# Main simulator class
# ═══════════════════════════════════════════════════════════════

class SparseGF2:
    """Pure-Python sparse stabilizer simulator with optional hybrid mode.

    The primary data structures are:

    - A dense Pauli Lookup Table (PLT) of shape ``(2n, n)`` for O(1)
      random access to any generator's Pauli at any qubit.
    - Sparse per-generator support lists (``supp_q``, ``supp_len``) plus a
      back-pointer position map (``supp_pos``) giving O(1) swap-with-last
      removal during gate application.
    - Per-qubit inverted indices (``inv`` for any Pauli, ``inv_x`` for
      generators with X-component) so that each gate touches only the
      generators that actually have support on the gated qubits.

    In the area-law phase (p > p_c), generators stay O(1)-sparse and the
    per-gate cost is O(a_bar) rather than O(n), giving an asymptotic
    speedup over dense tableau simulators like Stim.

    Parameters
    ----------
    n : int
        Number of system qubits. The tableau has 2n qubits in total
        (n system + n reference), initialized as n Bell pairs in the
        purification picture.
    use_min_weight_pivot : bool, default True
        When True, the measurement kernel picks the minimum-weight
        anticommuting generator as the pivot (reducing fill-in). When
        False, it picks the first anticommuting generator (faster but
        produces heavier generators in some cases). The stabilizer
        group is identical either way; only the generator representation
        differs.
    check_inputs : bool, default True
        Validate qubit indices at each gate call. Set to False in tight
        inner loops after correctness has been established.
    hybrid_mode : bool, default False
        When True, the simulator monitors the average active count
        ``a_bar`` and switches to a bit-packed dense representation if
        ``a_bar`` exceeds ``n/4`` (typical of volume-law circuits), then
        switches back to sparse when ``a_bar`` drops below ``n/8``. When
        False (the default), the simulator stays in sparse mode for the
        whole circuit — this is correct and faster whenever the circuit
        remains in the area-law phase.

    """

    def __init__(self, n, use_min_weight_pivot=True, check_inputs=True,
                 hybrid_mode=False):
        self.n = n
        self.N = 2 * n
        self.use_min_weight_pivot = use_min_weight_pivot
        self.check_inputs = check_inputs
        self._frozen_sparse = not hybrid_mode

        ms = _max_support(n)
        mi = _max_inv(n)

        # Dense PLT: O(1) Pauli access
        self.plt = np.zeros((self.N, n), dtype=np.uint8)

        # Sparse support lists + position map
        self.supp_q = np.zeros((self.N, ms), dtype=np.int32)
        self.supp_len = np.zeros(self.N, dtype=np.int32)
        self.supp_pos = np.full((self.N, n), -1, dtype=np.int32)

        # Inverted indices
        self.inv = np.zeros((n, mi), dtype=np.int32)
        self.inv_len = np.zeros(n, dtype=np.int32)
        self.inv_x = np.zeros((n, mi), dtype=np.int32)
        self.inv_x_len = np.zeros(n, dtype=np.int32)

        # Position maps for O(1) removal (int32 for large n)
        self.inv_pos = np.full((self.N, n), -1, dtype=np.int32)
        self.inv_x_pos = np.full((self.N, n), -1, dtype=np.int32)

        # Bit-packed arrays for dense mode
        n_words = (n + 63) // 64
        self.x_packed = np.zeros((self.N, n_words), dtype=np.uint64)
        self.z_packed = np.zeros((self.N, n_words), dtype=np.uint64)

        # Preallocated scratch buffers for kernels (avoid np.empty per-call).
        # Sized to mi (max inv row length); _apply_gate_kernel needs 2*mi for the
        # combined active set of two qubits.
        self._active_buf = np.zeros(mi * 2, dtype=np.int32)
        self._comm_buf = np.zeros(mi, dtype=np.int32)
        self._others_buf = np.zeros(mi, dtype=np.int32)

        # Hybrid mode state
        self._dense_mode = False
        self._ops_since_check = 0
        # Threshold: switch to dense when a_bar exceeds this value.
        # Empirically, the sparse overhead crossover is ~sqrt(2n), but we use
        # a conservative threshold to avoid switching on transients.
        self._dense_threshold = max(n // 4, 16)
        self._check_interval = max(n // 2, 32)

        _init_bell_pairs(self.plt, self.supp_q, self.supp_len, self.supp_pos,
                           self.inv, self.inv_len, self.inv_x, self.inv_x_len,
                           self.inv_pos, self.inv_x_pos, n)

    def _switch_to_dense(self):
        """Sync packed arrays from PLT and enter dense mode."""
        plt_to_packed(self.plt, self.x_packed, self.z_packed, self.n, self.N)
        self._dense_mode = True

    def _switch_to_sparse(self):
        """Rebuild PLT + all indices from packed arrays and enter sparse mode."""

        packed_to_plt(self.x_packed, self.z_packed, self.plt, self.n, self.N)
        rebuild_indices_from_plt(self.plt, self.supp_q, self.supp_len, self.supp_pos,
                                self.inv, self.inv_len, self.inv_x, self.inv_x_len,
                                self.inv_pos, self.inv_x_pos, self.n, self.N)
        self._dense_mode = False

    def _maybe_check_mode_switch(self):
        """Increment the per-op counter and run the mode-switch check if due.

        Used by per-gate entry points (apply_gate_1q, apply_gate,
        apply_measurement_z) and by run_layer. The check itself is guarded
        by _check_interval = max(n // 2, 32), so its O(n) cost is amortized
        across ~n/2 ops, giving O(1) amortized overhead per gate.

        Skipped entirely when frozen_sparse=True (the simulator is
        guaranteed to stay in sparse mode for the circuit's lifetime).
        """
        if self._frozen_sparse:
            return
        self._ops_since_check += 1
        if self._ops_since_check >= self._check_interval:
            self._ops_since_check = 0
            self._check_mode_switch()

    def _check_qubit(self, q, name="q"):
        """Validate qubit index is in range [0, n).

        No-op when self.check_inputs is False (opt-in debug validation).
        """
        if not self.check_inputs:
            return
        if not (0 <= q < self.n):
            raise IndexError(
                f"{name}={q} out of range for {self.n}-qubit system (valid: 0..{self.n - 1})")

    def _check_two_qubits(self, qi, qj):
        """Validate two distinct qubit indices in range [0, n).

        No-op when self.check_inputs is False (opt-in debug validation).
        """
        if not self.check_inputs:
            return
        if not (0 <= qi < self.n):
            raise IndexError(
                f"qi={qi} out of range for {self.n}-qubit system (valid: 0..{self.n - 1})")
        if not (0 <= qj < self.n):
            raise IndexError(
                f"qj={qj} out of range for {self.n}-qubit system (valid: 0..{self.n - 1})")
        if qi == qj:
            raise ValueError(f"Gate qubits must be distinct, got qi=qj={qi}")

    def apply_gate_1q(self, q, S2):
        """Apply a single-qubit Clifford gate via 2x2 GF(2) symplectic matrix.

        S2 maps (x_q, z_q) -> S2 @ (x_q, z_q) mod 2.
        """
        self._check_qubit(q, "q")
        S2 = np.asarray(S2, dtype=np.uint8)
        if self._dense_mode:
            # In dense mode: apply via packed arrays
            # 1Q gate doesn't change support, only x/z bits for qubit q
            w, b = q >> 6, np.uint64(q & 63)
            one = np.uint64(1)
            mask = one << b
            N = self.N
            for r in range(N):
                xbit = (self.x_packed[r, w] >> b) & one
                zbit = (self.z_packed[r, w] >> b) & one
                if xbit == 0 and zbit == 0:
                    continue
                nx = (xbit * np.uint64(S2[0, 0]) + zbit * np.uint64(S2[1, 0])) & one
                nz = (xbit * np.uint64(S2[0, 1]) + zbit * np.uint64(S2[1, 1])) & one
                if nx:
                    self.x_packed[r, w] |= mask
                else:
                    self.x_packed[r, w] &= ~mask
                if nz:
                    self.z_packed[r, w] |= mask
                else:
                    self.z_packed[r, w] &= ~mask
        else:
            _apply_gate_1q_kernel(self.plt, self.inv_x, self.inv_x_len,
                              self.inv_x_pos, self.inv, self.inv_len, q, S2)
        self._maybe_check_mode_switch()

    def apply_h(self, q):
        """Apply Hadamard gate on qubit q. Maps X<->Z."""
        self.apply_gate_1q(q, _H_SYMP)

    def apply_s(self, q):
        """Apply S gate on qubit q. Maps X->Y, Z->Z."""
        self.apply_gate_1q(q, _S_SYMP)

    def apply_sqrt_x(self, q):
        """Apply sqrt(X) gate on qubit q. Maps Z->Y, X->X."""
        self.apply_gate_1q(q, _SQRT_X_SYMP)

    def apply_h_fast(self, q):
        """Apply Hadamard gate on qubit q using specialized kernel.

        Bypasses LUT construction. In dense mode, falls back to generic path.
        """
        if self.check_inputs:
            self._check_qubit(q, "qubit")
        if self._dense_mode:
            self.apply_gate_1q(q, _H_SYMP)
            return
        _apply_h_kernel(self.plt, self.inv, self.inv_len, self.inv_x,
                        self.inv_x_len, self.inv_pos, self.inv_x_pos, q)
        self._maybe_check_mode_switch()

    def apply_s_fast(self, q):
        """Apply S gate on qubit q using specialized kernel.

        Bypasses LUT construction. In dense mode, falls back to generic path.
        """
        if self.check_inputs:
            self._check_qubit(q, "qubit")
        if self._dense_mode:
            self.apply_gate_1q(q, _S_SYMP)
            return
        _apply_s_kernel(self.plt, self.inv, self.inv_len, self.inv_x,
                        self.inv_x_len, self.inv_x_pos, q)
        self._maybe_check_mode_switch()

    def apply_cx_fast(self, qc, qt):
        """Apply CX (CNOT) gate with control qc, target qt using specialized kernel.

        Bypasses LUT construction and generic 4x4 symplectic multiply.
        In dense mode, falls back to generic path.
        """
        if self.check_inputs:
            self._check_two_qubits(qc, qt)
        if self._dense_mode:
            self.apply_gate(qc, qt, _CX_SYMP)
            return
        _apply_cx_kernel(self.plt, self.supp_q, self.supp_len, self.supp_pos,
                         self.inv, self.inv_len, self.inv_x, self.inv_x_len,
                         self.inv_pos, self.inv_x_pos, qc, qt,
                         self._active_buf)
        self._maybe_check_mode_switch()

    @property
    def backend(self) -> str:
        """Return ``'python'`` — identifier for this backend."""
        return "python"

    def apply_gate(self, qi, qj, S):
        """Apply a 2-qubit Clifford via a 4x4 GF(2) symplectic matrix.

        The gate maps ``(x_qi, x_qj, z_qi, z_qj)`` to
        ``S @ (x_qi, x_qj, z_qi, z_qj)`` (mod 2) on every generator with
        support on qi or qj. Generators with identity on both qubits are
        untouched (O(a_bar) per gate rather than O(n)).

        Parameters
        ----------
        qi, qj : int
            Distinct qubit indices in ``[0, n)``.
        S : ndarray, shape (4, 4), dtype uint8
            GF(2) symplectic matrix in the
            ``(X_qi, X_qj, Z_qi, Z_qj)`` basis. Extract one from a Stim
            ``Tableau`` via
            :func:`sparsegf2.symplectic_from_stim_tableau`.

        Raises
        ------
        IndexError
            If qi or qj is out of range (only when ``check_inputs=True``).
        ValueError
            If qi == qj.
        """
        self._check_two_qubits(qi, qj)  # no-op when check_inputs=False
        S = np.asarray(S, dtype=np.uint8)
        if self._dense_mode:
            apply_clifford_2q_packed(self.x_packed, self.z_packed, qi, qj, S)
        else:
            _apply_gate_kernel(self.plt, self.supp_q, self.supp_len, self.supp_pos,
                           self.inv, self.inv_len, self.inv_x, self.inv_x_len,
                           self.inv_pos, self.inv_x_pos, qi, qj, S,
                           self._active_buf)
        self._maybe_check_mode_switch()

    def apply_measurement_z(self, q):
        """Measure qubit q in the Z basis and reset to |0>."""
        self._check_qubit(q, "q")  # no-op when check_inputs=False
        if self._dense_mode:
            measure_z_packed(self.x_packed, self.z_packed, q)
        else:
            _measure_z_kernel(self.plt, self.supp_q, self.supp_len, self.supp_pos,
                          self.inv, self.inv_len, self.inv_x, self.inv_x_len,
                          self.inv_pos, self.inv_x_pos, q,
                          self.use_min_weight_pivot,
                          self._comm_buf, self._others_buf)
        self._maybe_check_mode_switch()

    def apply_measure_z(self, q):
        """Measure qubit q in the Z basis without explicit reset.

        In the phase-free GF(2) representation, this is identical to
        apply_measurement_z (measure + reset) because the reset only
        affects the sign bit which is not tracked. Provided for API
        parity with Stim's separate M and MR operations.
        """
        self.apply_measurement_z(q)

    def apply_reset_z(self, q):
        """Reset qubit q to |0> (project onto Z_q = +1 eigenstate).

        In the phase-free representation, this is equivalent to measurement
        followed by conditional correction (which only affects phases).
        """
        self.apply_measurement_z(q)

    def apply_measurement_x(self, q):
        """Measure qubit q in the X basis and reset.

        Implemented as H -> Z-measure -> H (conjugation to X eigenbasis).
        """
        self.apply_h(q)
        self.apply_measurement_z(q)
        self.apply_h(q)

    def apply_measurement_y(self, q):
        """Measure qubit q in the Y basis (phase-free).

        Rotates the Y eigenbasis to the Z eigenbasis via ``S`` then ``H``:
        under row-vector symplectic evolution, ``H S^dagger`` sends Y->Z
        (Aaronson-Gottesman 2004 Table I; phase-free, so S^dagger = S). The
        gate sequence applied is
        ``apply_s(q), apply_h(q), apply_measurement_z(q), apply_h(q), apply_s(q)``.
        """
        self.apply_s(q)
        self.apply_h(q)
        self.apply_measurement_z(q)
        self.apply_h(q)
        self.apply_s(q)

    def run_layer(self, gate_qi, gate_qj, gate_symp, meas_qubits):
        """Run a full layer of gates and measurements.

        Automatically selects dense or sparse mode and checks for mode transitions.

        Parameters
        ----------
        gate_qi, gate_qj : array-like of int, length n_gates
            Qubit indices for each 2-qubit gate.
        gate_symp : array-like of uint8, shape (n_gates, 4, 4)
            4x4 GF(2) symplectic matrices for each gate.
        meas_qubits : array-like of int
            Qubits to measure in the Z basis.
        """
        gate_symp = np.asarray(gate_symp, dtype=np.uint8)
        if len(gate_qi) > 0 and gate_symp.ndim != 3:
            raise ValueError(f"gate_symp must be (n_gates, 4, 4), got shape {gate_symp.shape}")
        # Run validations — each is a no-op when check_inputs=False
        if self.check_inputs:
            for g in range(len(gate_qi)):
                self._check_two_qubits(int(gate_qi[g]), int(gate_qj[g]))
            for m in range(len(meas_qubits)):
                self._check_qubit(int(meas_qubits[m]), "meas_qubit")
        if self._dense_mode:
            run_layer_dense(self.x_packed, self.z_packed,
                           np.asarray(gate_qi, dtype=np.int32),
                           np.asarray(gate_qj, dtype=np.int32),
                           gate_symp,
                           len(gate_qi),
                           np.asarray(meas_qubits, dtype=np.int32),
                           len(meas_qubits))
        else:
            for g in range(len(gate_qi)):
                _apply_gate_kernel(self.plt, self.supp_q, self.supp_len, self.supp_pos,
                               self.inv, self.inv_len, self.inv_x, self.inv_x_len,
                               self.inv_pos, self.inv_x_pos,
                               gate_qi[g], gate_qj[g], gate_symp[g],
                               self._active_buf)
            for m in range(len(meas_qubits)):
                _measure_z_kernel(self.plt, self.supp_q, self.supp_len, self.supp_pos,
                              self.inv, self.inv_len, self.inv_x, self.inv_x_len,
                              self.inv_pos, self.inv_x_pos, meas_qubits[m],
                              self.use_min_weight_pivot,
                              self._comm_buf, self._others_buf)

        # Periodic mode-switching check (once per layer, not once per op)
        self._maybe_check_mode_switch()

    def _check_mode_switch(self):
        """Check if we should switch between dense and sparse modes.

        Uses hysteresis: dense threshold is n/4, sparse threshold is n/8.
        This prevents rapid oscillation near the crossover point.
        """
        abar = self.get_active_count()
        if self._dense_mode:
            # Switch back to sparse if generators have become localized
            if abar < self._dense_threshold // 2:
                self._switch_to_sparse()
        else:
            # Switch to dense if a_bar is too large
            if abar > self._dense_threshold:
                self._switch_to_dense()

    def compute_k(self):
        """Code dimension k = rank(M_sys) - n.

        M_sys is the 2n x 2n symplectic matrix of the full purification
        tableau restricted to the n system-qubit columns. This is the
        reference-entropy S(R) = k computed via the Fattal-Cubitt-
        Yamamoto-Bravyi-Chuang formula (arXiv:quant-ph/0406168, Thm. 1):
        S(A) = rank(M|_A) - |A|. Computed via a Numba-JIT bit-packed
        GF(2) rank kernel.

        Returns
        -------
        int
            The code dimension k. For the initial Bell-pair state this
            is n (all qubits entangled with reference); after enough
            measurements it drops to 0.
        """
        if not self._dense_mode:
            # Sync packed arrays from PLT, then use fast Numba rank kernel
            plt_to_packed(self.plt, self.x_packed, self.z_packed, self.n, self.N)
        return compute_k_packed(self.x_packed, self.z_packed, self.n, self.N)

    def extract_sys_matrix(self):
        """Extract the 2n x 2n system sub-matrix for tableau comparison.

        In dense mode, updates self.plt as a side effect (packed arrays remain canonical).
        """
        if self._dense_mode:
            packed_to_plt(self.x_packed, self.z_packed, self.plt, self.n, self.N)
        n = self.n
        N = self.N
        mat = np.zeros((N, 2 * n), dtype=np.uint8)
        for r in range(N):
            for q in range(n):
                xz = self.plt[r, q]
                if xz != 0:
                    if (xz >> 1) & 1:
                        mat[r, q] = 1
                    if xz & 1:
                        mat[r, n + q] = 1
        return mat

    def get_active_count(self):
        """Average active count abar = (1/n) * sum_q |inv[q]|.

        Dense-mode implementation uses vectorized popcount over packed arrays:
        abar = popcount(x_packed | z_packed) / n, masking any stray bits
        beyond qubit n-1 in the last word. ~27x faster than the
        per-qubit Python loop for n=256.
        """
        if self._dense_mode:
            combined = self.x_packed | self.z_packed
            # Mask off stray bits in the last word beyond qubit n-1
            tail = self.n & 63
            if tail:
                mask_val = (np.uint64(1) << np.uint64(tail)) - np.uint64(1)
                last_col = combined[:, -1] & mask_val
                total_bits = int(
                    np.unpackbits(combined[:, :-1].view(np.uint8)).sum()
                    + np.unpackbits(last_col.view(np.uint8)).sum()
                )
            else:
                total_bits = int(
                    np.unpackbits(combined.view(np.uint8)).sum()
                )
            return total_bits / self.n
        # Sparse mode: sum inv_len across qubits via numpy for speed.
        return float(self.inv_len[:self.n].sum()) / self.n

    def compute_subsystem_entropy(self, qubits):
        """Compute entanglement entropy S(A) for a subsystem A.

        Uses the Fattal-Cubitt-Yamamoto-Bravyi-Chuang formula (arXiv:
        quant-ph/0406168, Theorem 1): ``S(A) = rank(M|_A) - |A|``, where
        ``M|_A`` is the stabilizer symplectic matrix restricted to the
        2|A| columns (X and Z) of the qubits in A.

        Parameters
        ----------
        qubits : list of int
            Qubit indices defining subsystem A.

        Returns
        -------
        int
            Entanglement entropy S(A) in units of log_2 (an integer for
            stabilizer states).
        """
        if self._dense_mode:
            packed_to_plt(self.x_packed, self.z_packed, self.plt, self.n, self.N)
        n = self.n
        N = self.N
        nA = len(qubits)
        n_cols = 2 * nA
        n_words = (n_cols + 63) >> 6
        mat = np.zeros((N, n_words), dtype=np.uint64)
        for r in range(N):
            for idx, q in enumerate(qubits):
                xz = self.plt[r, q]
                if xz != 0:
                    if (xz >> 1) & 1:
                        _set_bit(mat, r, idx, 1)
                    if xz & 1:
                        _set_bit(mat, r, nA + idx, 1)
        rank = gf2_rank_packed(mat, n_cols)
        return rank - nA

    def compute_tmi(self):
        """Compute tripartite mutual information I_3(A:B:C).

        For an approximately equal tripartition of the n system qubits:
          I_3 = S(A) + S(B) + S(C) - S(AB) - S(AC) - S(BC) + S(ABC)

        Reference: Hosur, Qi, Roberts, Yoshida, "Chaos in quantum
        channels", JHEP 2016:4 (arXiv:1511.04021), Eq. 24. See also
        Kitaev & Preskill 2006 and Levin & Wen 2006 in the topological-
        entanglement-entropy context.

        Returns
        -------
        float
            I_3(A:B:C). Negative in volume-law, zero in area-law.
        """
        n = self.n
        if n < 3:
            return 0.0
        # Split into 3 approximately equal parts
        s1 = n // 3
        s2 = 2 * n // 3
        A = list(range(0, s1))
        B = list(range(s1, s2))
        C = list(range(s2, n))

        SA = self.compute_subsystem_entropy(A)
        SB = self.compute_subsystem_entropy(B)
        SC = self.compute_subsystem_entropy(C)
        SAB = self.compute_subsystem_entropy(A + B)
        SAC = self.compute_subsystem_entropy(A + C)
        SBC = self.compute_subsystem_entropy(B + C)
        SABC = self.compute_subsystem_entropy(A + B + C)

        return SA + SB + SC - SAB - SAC - SBC + SABC

    def compute_bandwidth(self):
        """Compute maximum generator bandwidth on the ring C_n.

        Bandwidth of generator r = size of smallest contiguous arc on C_n
        containing all qubits in its support.

        Returns
        -------
        int
            Maximum bandwidth b* across all generators.
        """
        if self._dense_mode:
    
            packed_to_plt(self.x_packed, self.z_packed, self.plt, self.n, self.N)
            rebuild_indices_from_plt(self.plt, self.supp_q, self.supp_len, self.supp_pos,
                                    self.inv, self.inv_len, self.inv_x, self.inv_x_len,
                                    self.inv_pos, self.inv_x_pos, self.n, self.N)
        n = self.n
        N = self.N
        max_bw = 0
        for r in range(N):
            L = self.supp_len[r]
            if L <= 1:
                continue
            # Collect support qubits, sort
            qubits = sorted(self.supp_q[r, :L])
            # Compute gaps between consecutive support qubits on the ring
            max_gap = (qubits[0] + n - qubits[-1]) % n  # wrap-around gap
            for i in range(1, len(qubits)):
                gap = qubits[i] - qubits[i-1]
                if gap > max_gap:
                    max_gap = gap
            bw = n - max_gap
            if bw > max_bw:
                max_bw = bw
        return max_bw

    def run_circuit_batch(self, gate_qi, gate_qj, gate_symp, gate_layer_starts,
                          gate1q_q, gate1q_symp, gate1q_layer_starts,
                          meas_qubits, meas_layer_starts, meas_basis,
                          n_layers):
        """Execute a full circuit (gates + measurements) in one Numba call.

        Eliminates Python dispatch overhead per gate/measurement. Suitable
        for large circuits where layer-by-layer Python orchestration is the
        bottleneck.

        Parameters
        ----------
        gate_qi, gate_qj : int32 arrays, length N_gates_2q
            Qubit indices for each 2-qubit gate, flattened across all layers.
        gate_symp : uint8 array, shape (N_gates_2q, 4, 4)
            4x4 GF(2) symplectic matrices.
        gate_layer_starts : int32 array, length n_layers + 1
            Slice offsets into gate_qi/gate_qj/gate_symp per layer.
        gate1q_q : int32 array, length N_gates_1q
            Qubit indices for each 1-qubit gate, flattened.
        gate1q_symp : uint8 array, shape (N_gates_1q, 2, 2)
            2x2 GF(2) symplectic matrices for 1-qubit gates.
        gate1q_layer_starts : int32 array, length n_layers + 1
            Slice offsets into gate1q_q per layer.
        meas_qubits : int32 array, length N_meas
            Qubit indices for measurements, flattened.
        meas_layer_starts : int32 array, length n_layers + 1
            Slice offsets into meas_qubits per layer.
        meas_basis : uint8 array, length N_meas
            Per-measurement basis: 0=Z, 1=X, 2=Y.
        n_layers : int
            Number of layers.

        Notes
        -----
        Forces sparse mode at the start (the batch kernel doesn't operate
        on dense packed arrays). Mode-switch check runs at the end with
        amortized cost.
        """
        if self._dense_mode:
            self._switch_to_sparse()
        _run_circuit_batch_kernel(
            self.plt, self.supp_q, self.supp_len, self.supp_pos,
            self.inv, self.inv_len, self.inv_x, self.inv_x_len,
            self.inv_pos, self.inv_x_pos,
            np.asarray(gate_qi, dtype=np.int32),
            np.asarray(gate_qj, dtype=np.int32),
            np.asarray(gate_symp, dtype=np.uint8),
            np.asarray(gate_layer_starts, dtype=np.int32),
            np.asarray(gate1q_q, dtype=np.int32),
            np.asarray(gate1q_symp, dtype=np.uint8),
            np.asarray(gate1q_layer_starts, dtype=np.int32),
            np.asarray(meas_qubits, dtype=np.int32),
            np.asarray(meas_layer_starts, dtype=np.int32),
            np.asarray(meas_basis, dtype=np.uint8),
            n_layers, self.use_min_weight_pivot,
            self._active_buf, self._comm_buf, self._others_buf,
        )
        # After the batch, check mode-switch (cost amortized over whole circuit)
        if not self._frozen_sparse:
            self._check_mode_switch()

    def run_random_edge_circuit(self, edges_qi, edges_qj, cliff_symp,
                                edge_idx, cliff_idx, meas_r1, meas_r2, p):
        """Run a full random-edge circuit in a single compiled call.

        ~1.7x faster than a Python loop with per-step apply_gate/measure calls.
        All random numbers must be pre-generated and passed as arrays.

        Parameters
        ----------
        edges_qi, edges_qj : int32 arrays, length |E|
            Qubit pairs for each edge in the graph.
        cliff_symp : uint8 array, shape (n_cliffords, 4, 4)
            Clifford group symplectic matrices.
        edge_idx : int32 array, length T
            Random edge indices for each step.
        cliff_idx : int32 array, length T
            Random Clifford indices for each step.
        meas_r1, meas_r2 : float64 arrays, length T
            Random numbers for measurement decisions.
        p : float
            Measurement probability per gated qubit.
        """
        T = len(edge_idx)
        if self._dense_mode:
            self._switch_to_sparse()  # batch kernel operates in sparse mode
        _run_random_edge_circuit(
            self.plt, self.supp_q, self.supp_len, self.supp_pos,
            self.inv, self.inv_len, self.inv_x, self.inv_x_len,
            self.inv_pos, self.inv_x_pos,
            np.asarray(edges_qi, dtype=np.int32),
            np.asarray(edges_qj, dtype=np.int32),
            np.asarray(cliff_symp, dtype=np.uint8),
            np.asarray(edge_idx, dtype=np.int32),
            np.asarray(cliff_idx, dtype=np.int32),
            np.asarray(meas_r1, dtype=np.float64),
            np.asarray(meas_r2, dtype=np.float64),
            float(p), T, self.use_min_weight_pivot,
            self._active_buf, self._comm_buf, self._others_buf,
        )


# ═══════════════════════════════════════════════════════════════
# Batch random-edge circuit kernel (eliminates Python loop overhead)
# ═══════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _run_random_edge_circuit(plt, supp_q, supp_len, supp_pos,
                             inv, inv_len, inv_x, inv_x_len,
                             inv_pos, inv_x_pos,
                             edges_qi, edges_qj, cliff_symp,
                             edge_idx, cliff_idx, meas_r1, meas_r2,
                             p, T, use_min_weight_pivot,
                             active_buf, comm_buf, others_buf):
    """Run T random-edge steps entirely in compiled Numba code.

    Eliminates the Python loop overhead (~35% of runtime) by executing
    the full circuit in a single JIT-compiled function call.

    Parameters
    ----------
    edges_qi, edges_qj : int32 arrays of length |E|
        Qubit pairs for each edge in the graph.
    cliff_symp : uint8 array of shape (n_cliffords, 4, 4)
        Full Clifford group symplectic matrices.
    edge_idx : int32 array of length T
        Pre-generated random edge indices.
    cliff_idx : int32 array of length T
        Pre-generated random Clifford indices.
    meas_r1, meas_r2 : float64 arrays of length T
        Pre-generated random numbers for measurement decisions.
    p : float
        Measurement probability per gated qubit.
    T : int
        Number of circuit steps.
    active_buf, comm_buf, others_buf : int32 scratch buffers
        Preallocated per-simulator scratch passed down to kernels.
    """
    for t in range(T):
        ei = edge_idx[t]
        qi = edges_qi[ei]
        qj = edges_qj[ei]
        S = cliff_symp[cliff_idx[t]]
        _apply_gate_kernel(plt, supp_q, supp_len, supp_pos,
                       inv, inv_len, inv_x, inv_x_len,
                       inv_pos, inv_x_pos, qi, qj, S, active_buf)
        if meas_r1[t] < p:
            _measure_z_kernel(plt, supp_q, supp_len, supp_pos,
                          inv, inv_len, inv_x, inv_x_len,
                          inv_pos, inv_x_pos, qi, use_min_weight_pivot,
                          comm_buf, others_buf)
        if meas_r2[t] < p:
            _measure_z_kernel(plt, supp_q, supp_len, supp_pos,
                          inv, inv_len, inv_x, inv_x_len,
                          inv_pos, inv_x_pos, qj, use_min_weight_pivot,
                          comm_buf, others_buf)


# ═══════════════════════════════════════════════════════════════
# Generalized batched circuit kernel
# ═══════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _run_circuit_batch_kernel(plt, supp_q, supp_len, supp_pos,
                              inv, inv_len, inv_x, inv_x_len,
                              inv_pos, inv_x_pos,
                              gate_qi, gate_qj, gate_symp,
                              gate_layer_starts,
                              gate1q_q, gate1q_symp,
                              gate1q_layer_starts,
                              meas_qubits, meas_layer_starts,
                              meas_basis,
                              n_layers, use_min_weight_pivot,
                              active_buf, comm_buf, others_buf):
    """Run a full circuit in one Numba call, no per-gate Python dispatch.

    Layout uses "starts" arrays of length n_layers+1 giving slice ranges.

    Per layer L:
      - 1Q gates  in gate1q_q[gate1q_layer_starts[L]:gate1q_layer_starts[L+1]]
      - 2Q gates  in gate_qi[gate_layer_starts[L]:gate_layer_starts[L+1]]
      - meas      in meas_qubits[meas_layer_starts[L]:meas_layer_starts[L+1]]

    meas_basis[k] is 0=Z, 1=X (apply H, measure Z, apply H), 2=Y (SH, measure Z, HS).
    For surface codes M/MR/R all reduce to Z-basis measurement (basis=0).
    """
    for L in range(n_layers):
        # 1Q gates — dispatch to specialized kernels when possible
        g1_start = gate1q_layer_starts[L]
        g1_end = gate1q_layer_starts[L + 1]
        for g in range(g1_start, g1_end):
            q = gate1q_q[g]
            S2 = gate1q_symp[g]
            # Check for H: [[0,1],[1,0]]
            if (S2[0, 0] == 0 and S2[0, 1] == 1 and
                    S2[1, 0] == 1 and S2[1, 1] == 0):
                _apply_h_kernel(plt, inv, inv_len, inv_x, inv_x_len,
                                inv_pos, inv_x_pos, q)
            # Check for S: [[1,1],[0,1]] (Aaronson-Gottesman 2004 Table I)
            elif (S2[0, 0] == 1 and S2[0, 1] == 1 and
                      S2[1, 0] == 0 and S2[1, 1] == 1):
                _apply_s_kernel(plt, inv, inv_len, inv_x, inv_x_len,
                                inv_x_pos, q)
            else:
                _apply_gate_1q_kernel(plt, inv_x, inv_x_len, inv_x_pos,
                                      inv, inv_len, q, S2)

        # 2Q gates — dispatch to specialized CX kernel when possible
        g_start = gate_layer_starts[L]
        g_end = gate_layer_starts[L + 1]
        for g in range(g_start, g_end):
            qi = gate_qi[g]
            qj = gate_qj[g]
            S = gate_symp[g]
            # Check for CX: [[1,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,1]]
            if (S[0, 0] == 1 and S[0, 1] == 1 and S[0, 2] == 0 and S[0, 3] == 0 and
                    S[1, 0] == 0 and S[1, 1] == 1 and S[1, 2] == 0 and S[1, 3] == 0 and
                    S[2, 0] == 0 and S[2, 1] == 0 and S[2, 2] == 1 and S[2, 3] == 0 and
                    S[3, 0] == 0 and S[3, 1] == 0 and S[3, 2] == 1 and S[3, 3] == 1):
                _apply_cx_kernel(plt, supp_q, supp_len, supp_pos,
                                 inv, inv_len, inv_x, inv_x_len,
                                 inv_pos, inv_x_pos, qi, qj, active_buf)
            else:
                _apply_gate_kernel(plt, supp_q, supp_len, supp_pos,
                                   inv, inv_len, inv_x, inv_x_len,
                                   inv_pos, inv_x_pos, qi, qj, S, active_buf)

        # Measurements (all in Z basis after rotation)
        # Basis rotations use specialized H/S kernels directly
        m_start = meas_layer_starts[L]
        m_end = meas_layer_starts[L + 1]
        for m in range(m_start, m_end):
            q = meas_qubits[m]
            basis = meas_basis[m]
            if basis == 1:
                # X-basis: H, Z-meas, H
                _apply_h_kernel(plt, inv, inv_len, inv_x, inv_x_len,
                                inv_pos, inv_x_pos, q)
                _measure_z_kernel(plt, supp_q, supp_len, supp_pos,
                                  inv, inv_len, inv_x, inv_x_len,
                                  inv_pos, inv_x_pos, q, use_min_weight_pivot,
                                  comm_buf, others_buf)
                _apply_h_kernel(plt, inv, inv_len, inv_x, inv_x_len,
                                inv_pos, inv_x_pos, q)
            elif basis == 2:
                # Y-basis: S, H, measure_Z, H, S  (phase-free: S = S^dagger)
                _apply_s_kernel(plt, inv, inv_len, inv_x, inv_x_len,
                                inv_x_pos, q)
                _apply_h_kernel(plt, inv, inv_len, inv_x, inv_x_len,
                                inv_pos, inv_x_pos, q)
                _measure_z_kernel(plt, supp_q, supp_len, supp_pos,
                                  inv, inv_len, inv_x, inv_x_len,
                                  inv_pos, inv_x_pos, q, use_min_weight_pivot,
                                  comm_buf, others_buf)
                _apply_h_kernel(plt, inv, inv_len, inv_x, inv_x_len,
                                inv_pos, inv_x_pos, q)
                _apply_s_kernel(plt, inv, inv_len, inv_x, inv_x_len,
                                inv_x_pos, q)
            else:
                # Z-basis (also covers M, MR, R since phase is not tracked)
                _measure_z_kernel(plt, supp_q, supp_len, supp_pos,
                                  inv, inv_len, inv_x, inv_x_len,
                                  inv_pos, inv_x_pos, q, use_min_weight_pivot,
                                  comm_buf, others_buf)


def warmup():
    """JIT-compile all numba functions (sparse + dense modes)."""
    sim = SparseGF2(6)  # n=6 for TMI (needs n divisible by 3)
    S = np.eye(4, dtype=np.uint8)
    _build_gate_lut(S)  # Pre-compile LUT builder

    # Sparse mode warmup
    sim.apply_gate(0, 1, S)
    sim.apply_measurement_z(0)

    # Specialized kernel warmup (H, S, CX)
    sim.apply_h_fast(0)
    sim.apply_s_fast(0)
    sim.apply_cx_fast(0, 1)

    sim.compute_k()
    sim.extract_sys_matrix()
    sim.get_active_count()
    sim.compute_subsystem_entropy([0, 1, 2])
    sim.compute_tmi()
    sim.compute_bandwidth()

    # Dense mode warmup
    n_words = (6 + 63) // 64
    xp = np.zeros((12, n_words), dtype=np.uint64)
    zp = np.zeros((12, n_words), dtype=np.uint64)
    build_gate_lut(S)
    apply_clifford_2q_packed(xp, zp, 0, 1, S)
    measure_z_packed(xp, zp, 0)
    plt_to_packed(sim.plt, xp, zp, 6, 12)
    packed_to_plt(xp, zp, sim.plt, 6, 12)
    rebuild_indices_from_plt(sim.plt, sim.supp_q, sim.supp_len, sim.supp_pos,
                            sim.inv, sim.inv_len, sim.inv_x, sim.inv_x_len,
                            sim.inv_pos, sim.inv_x_pos, 6, 12)
    compute_k_packed(xp, zp, 6, 12)
    gqi = np.array([0], dtype=np.int32)
    gqj = np.array([1], dtype=np.int32)
    gs = np.eye(4, dtype=np.uint8).reshape(1, 4, 4)
    mq = np.array([0], dtype=np.int32)
    run_layer_dense(xp, zp, gqi, gqj, gs, 1, mq, 1)

    # Batched circuit kernel warmup
    sim2 = SparseGF2(6, check_inputs=False)
    g1q_q = np.array([0], dtype=np.int32)
    g1q_s = np.eye(2, dtype=np.uint8).reshape(1, 2, 2)
    g1q_starts = np.array([0, 1], dtype=np.int32)
    g_starts = np.array([0, 1], dtype=np.int32)
    m_starts = np.array([0, 1], dtype=np.int32)
    m_basis = np.array([0], dtype=np.uint8)
    sim2.run_circuit_batch(
        gqi, gqj, gs, g_starts,
        g1q_q, g1q_s, g1q_starts,
        mq, m_starts, m_basis,
        1)
