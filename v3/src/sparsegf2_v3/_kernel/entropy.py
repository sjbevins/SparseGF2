"""Compiled entanglement and erasure computations on the column-major tableau.

For a pure stabilizer state on ``N`` qubits, the entropy (in bits) of a set
``A`` of qubits is ``rank(stabilizers restricted to A) - |A|`` (Fattal et al.,
quant-ph/0406168). In the column layout the restriction is just the X and Z
bitsets of the qubits in ``A`` (stabilizer words only), so the rank is a
Gaussian elimination over ``2 |A|`` bitsets of ``ws`` words: cost
``O(|A|^2 * ws)``. Pure-state symmetry ``S(A) = S(complement)`` lets every
routine work with the smaller side.

The incremental :class:`basis` routines keep a reduced set of bitsets keyed by
their lowest set bit, so adding one qubit's two columns costs ``O(rank * ws)``.
That gives all prefix entropies of an ordering (an entanglement profile) and
contiguous-region scans in ``O(N * rank * ws)``.

The ``*_rows`` routines at the end serve the ensemble engine, which maintains
the *reference rows* of a code (see
:func:`~sparsegf2_v3._kernel.tableau.measure_1q_phase_free_tracked`): with
that invariant, ``S(R)`` is a popcount and ``I(R:E) = rank(E) - rank(E over
the other rows)``, so no reference column ever enters an elimination.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from sparsegf2_v3._kernel.bits import ZERO, ctz, popcount
from sparsegf2_v3._kernel.rng import next_below, stream_state
from sparsegf2_v3._kernel.tableau import columns_rank, columns_rank_without, single_qubit_entropy


@njit(cache=True)
def region_entropy(tab, n, ws, region, mask_buf, comp_buf, scratch):
    """Entropy of the qubits in ``region`` (distinct indices), using the smaller side.

    ``mask_buf`` (bool, length n), ``comp_buf`` (int64, length n) and
    ``scratch`` (uint64, shape >= (2 * (n // 2 + 1), ws)) are work arrays.
    """
    k = region.shape[0]
    if k == 0 or k == n:
        return 0
    if 2 * k <= n:
        if k == 1:
            return single_qubit_entropy(tab, region[0], ws)
        return columns_rank(tab, region, ws, scratch) - k
    for q in range(n):
        mask_buf[q] = True
    for j in range(k):
        mask_buf[region[j]] = False
    m = 0
    for q in range(n):
        if mask_buf[q]:
            comp_buf[m] = q
            m += 1
    if m == 1:
        return single_qubit_entropy(tab, comp_buf[0], ws)
    return columns_rank(tab, comp_buf[:m], ws, scratch) - m


# -- incremental basis ------------------------------------------------------------------


@njit(cache=True)
def basis_reset(pivot_owner, num):
    """Forget the first ``num`` basis vectors (pivot_owner maps bit -> vector index)."""
    for i in range(pivot_owner.shape[0]):
        pivot_owner[i] = -1
    num[0] = 0


@njit(cache=True)
def basis_add(vec, basis, pivot_owner, num, ws):
    """Reduce ``vec`` (modified) against ``basis``; add it if independent. Returns 1 if added."""
    while True:
        lead_w = -1
        for w in range(ws):
            if vec[w] != ZERO:
                lead_w = w
                break
        if lead_w < 0:
            return 0
        bit = (lead_w << 6) + np.int64(ctz(vec[lead_w]))
        owner = pivot_owner[bit]
        if owner < 0:
            idx = num[0]
            for w in range(ws):
                basis[idx, w] = vec[w]
            pivot_owner[bit] = idx
            num[0] = idx + 1
            return 1
        for w in range(lead_w, ws):
            vec[w] ^= basis[owner, w]


@njit(cache=True)
def basis_add_qubit(tab, q, ws, basis, pivot_owner, num, vec):
    """Add the X and Z stabilizer columns of qubit ``q``; return how many were independent."""
    added = 0
    for col in range(2):
        for w in range(ws):
            vec[w] = tab[2 * q + col, w]
        added += basis_add(vec, basis, pivot_owner, num, ws)
    return added


@njit(cache=True)
def basis_add_qubit_without(tab, q, ws, skip_rows, basis, pivot_owner, num, vec):
    """:func:`basis_add_qubit` with the rows set in ``skip_rows`` masked out."""
    added = 0
    for col in range(2):
        for w in range(ws):
            vec[w] = tab[2 * q + col, w] & ~skip_rows[w]
        added += basis_add(vec, basis, pivot_owner, num, ws)
    return added


@njit(cache=True)
def prefix_entropies(tab, n, ws, order, out):
    """``out[k] = S(order[:k])`` for k = 0..len(order); ``out`` has length len(order)+1."""
    basis = np.zeros((min(2 * order.shape[0], 64 * ws) + 1, ws), dtype=np.uint64)
    pivot_owner = np.full(64 * ws, -1, dtype=np.int64)
    num = np.zeros(1, dtype=np.int64)
    vec = np.zeros(ws, dtype=np.uint64)
    out[0] = 0
    rank = 0
    for k in range(order.shape[0]):
        rank += basis_add_qubit(tab, order[k], ws, basis, pivot_owner, num, vec)
        out[k + 1] = rank - (k + 1)


# -- erasure ----------------------------------------------------------------------------


@njit(cache=True)
def mutual_information_with(tab, n, ws, reference, erased, mask_buf, comp_buf, scratch, union_buf):
    """I(R:E) = S(R) + S(E) - S(R u E) for disjoint qubit sets R (reference) and E."""
    s_r = region_entropy(tab, n, ws, reference, mask_buf, comp_buf, scratch)
    s_e = region_entropy(tab, n, ws, erased, mask_buf, comp_buf, scratch)
    m = 0
    for j in range(reference.shape[0]):
        union_buf[m] = reference[j]
        m += 1
    for j in range(erased.shape[0]):
        union_buf[m] = erased[j]
        m += 1
    s_re = region_entropy(tab, n, ws, union_buf[:m], mask_buf, comp_buf, scratch)
    return s_r + s_e - s_re


@njit(cache=True)
def contiguous_distance(tab, n, ws, system, reference, periodic):
    """Smallest contiguous run of ``system`` (in the given order) whose erasure loses logical info.

    "Loses logical info" means I(R:A) > 0. Runs wrap around when ``periodic``.
    Returns 0 if the reference holds no logical information (S(R) = 0).

    Implementation: the reference columns are reduced once; then for every
    start the run is grown one qubit at a time, tracking ``rank(A)`` and
    ``rank(R u A)`` incrementally, and stopped as soon as it cannot beat the
    best length so far. I(R:A) = S(R) + rank(A) - rank(R u A) + |R|.
    """
    num_sys = system.shape[0]
    num_ref = reference.shape[0]
    cap = 64 * ws + 1
    vec = np.zeros(ws, dtype=np.uint64)
    # reference basis, built once
    ref_basis = np.zeros((cap, ws), dtype=np.uint64)
    ref_owner = np.full(64 * ws, -1, dtype=np.int64)
    ref_num = np.zeros(1, dtype=np.int64)
    rank_r = 0
    for j in range(num_ref):
        rank_r += basis_add_qubit(tab, reference[j], ws, ref_basis, ref_owner, ref_num, vec)
    s_r = rank_r - num_ref
    if s_r == 0:
        return 0
    a_basis = np.zeros((cap, ws), dtype=np.uint64)
    a_owner = np.full(64 * ws, -1, dtype=np.int64)
    a_num = np.zeros(1, dtype=np.int64)
    ra_basis = np.zeros((cap, ws), dtype=np.uint64)
    ra_owner = np.full(64 * ws, -1, dtype=np.int64)
    ra_num = np.zeros(1, dtype=np.int64)
    best = num_sys
    for s in range(num_sys):
        basis_reset(a_owner, a_num)
        # R u A starts as a copy of the reference basis
        for i in range(ra_owner.shape[0]):
            ra_owner[i] = ref_owner[i]
        for i in range(ref_num[0]):
            for w in range(ws):
                ra_basis[i, w] = ref_basis[i, w]
        ra_num[0] = ref_num[0]
        rank_a = 0
        rank_ra = rank_r
        max_len = best - 1
        if not periodic:
            max_len = min(max_len, num_sys - s)
        for length in range(1, max_len + 1):
            q = system[(s + length - 1) % num_sys]
            rank_a += basis_add_qubit(tab, q, ws, a_basis, a_owner, a_num, vec)
            rank_ra += basis_add_qubit(tab, q, ws, ra_basis, ra_owner, ra_num, vec)
            if s_r + rank_a - rank_ra + num_ref > 0:
                best = length
                break
    return best


@njit(cache=True)
def random_erasure_stats(tab, n, ws, system, reference, sizes, samples, key, lost):
    """For each size m, erase ``samples`` uniformly random m-subsets of ``system``.

    ``lost[i, j]`` receives I(R:E) for sample ``j`` of size ``sizes[i]``.
    Subsets come from random stream 0 of ``key`` (partial Fisher-Yates).
    """
    rng = np.empty(4, dtype=np.uint64)
    stream_state(key, 0, rng)
    num_sys = system.shape[0]
    perm = system.copy()
    mask_buf = np.zeros(n, dtype=np.bool_)
    comp_buf = np.zeros(n, dtype=np.int64)
    scratch = np.zeros((2 * (n // 2 + 1), ws), dtype=np.uint64)
    union_buf = np.zeros(n, dtype=np.int64)
    for i in range(sizes.shape[0]):
        m = sizes[i]
        for j in range(samples):
            for t in range(m):
                r = t + next_below(rng, num_sys - t)
                tmp = perm[t]
                perm[t] = perm[r]
                perm[r] = tmp
            lost[i, j] = mutual_information_with(
                tab, n, ws, reference, perm[:m], mask_buf, comp_buf, scratch, union_buf
            )


@njit(cache=True)
def erasure_profile(tab, n, ws, order, reference, out):
    """``out[m] = I(R : order[:m])`` for m = 0..len(order), in one incremental pass.

    Uses I(R:E) = S(R) + rank(E) - rank(R u E) + |R| with both ranks grown one
    qubit at a time; erasing a uniformly random ordering's prefixes gives
    uniformly random erasures of every size at once.
    """
    cap = 64 * ws + 1
    vec = np.zeros(ws, dtype=np.uint64)
    e_basis = np.zeros((cap, ws), dtype=np.uint64)
    e_owner = np.full(64 * ws, -1, dtype=np.int64)
    e_num = np.zeros(1, dtype=np.int64)
    re_basis = np.zeros((cap, ws), dtype=np.uint64)
    re_owner = np.full(64 * ws, -1, dtype=np.int64)
    re_num = np.zeros(1, dtype=np.int64)
    num_ref = reference.shape[0]
    rank_re = 0
    for j in range(num_ref):
        rank_re += basis_add_qubit(tab, reference[j], ws, re_basis, re_owner, re_num, vec)
    s_r = rank_re - num_ref
    rank_e = 0
    out[0] = 0
    for m in range(order.shape[0]):
        q = order[m]
        rank_e += basis_add_qubit(tab, q, ws, e_basis, e_owner, e_num, vec)
        rank_re += basis_add_qubit(tab, q, ws, re_basis, re_owner, re_num, vec)
        out[m + 1] = s_r + rank_e - rank_re + num_ref


# -- clipped gauge ----------------------------------------------------------------------


@njit(cache=True)
def _rightmost_bit(row, words):
    for w in range(words - 1, -1, -1):
        if row[w] != ZERO:
            x = row[w]
            b = 63
            while ((x >> np.uint64(b)) & np.uint64(1)) == ZERO:
                b -= 1
            return (w << 6) + b
    return -1


@njit(cache=True)
def clipped_gauge(tab, n, ws, order, left, right):
    """Endpoints of the stabilizers in the clipped gauge along ``order`` (all n qubits).

    Rows are packed site-major (column ``2 s`` = x, ``2 s + 1`` = z at site
    ``s`` of ``order``). Pass 1 (left gauge): eliminate column by column from
    the left so every generator has a distinct leftmost column. Pass 2 (right
    gauge): from the right, among generators sharing a rightmost column keep the
    one with the largest leftmost column and add it to the others, which never
    moves anyone's left endpoint. ``left[i]``/``right[i]`` receive the site
    indices (positions in ``order``) of generator i's endpoints. In this gauge
    ``S(order[:k]) = #{i : left[i] < k <= right[i]} / 2`` (Nahum et al.,
    PRX 7, 031016; Li, Chen and Fisher, PRB 98, 205136).
    """
    num_cols = 2 * n
    words = (num_cols + 63) // 64
    rows = np.zeros((n, words), dtype=np.uint64)
    for s in range(n):
        q = order[s]
        for col in range(2):
            c = 2 * s + col
            cw = c >> 6
            cbit = np.uint64(1) << np.uint64(c & 63)
            for r in range(n):
                if (tab[2 * q + col, r >> 6] >> np.uint64(r & 63)) & np.uint64(1):
                    rows[r, cw] |= cbit
    left_col = np.full(n, -1, dtype=np.int64)
    for c in range(num_cols):
        cw = c >> 6
        cbit = np.uint64(1) << np.uint64(c & 63)
        pivot = -1
        for r in range(n):
            if left_col[r] < 0 and (rows[r, cw] & cbit) != ZERO:
                pivot = r
                break
        if pivot < 0:
            continue
        left_col[pivot] = c
        for r in range(n):
            if r != pivot and left_col[r] < 0 and (rows[r, cw] & cbit) != ZERO:
                for w in range(cw, words):
                    rows[r, w] ^= rows[pivot, w]
    rightmost = np.zeros(n, dtype=np.int64)
    for r in range(n):
        rightmost[r] = _rightmost_bit(rows[r], words)
    right_col = np.full(n, -1, dtype=np.int64)
    for c in range(num_cols - 1, -1, -1):
        best = -1
        for r in range(n):
            if (
                right_col[r] < 0
                and rightmost[r] == c
                and (best < 0 or left_col[r] > left_col[best])
            ):
                best = r
        if best < 0:
            continue
        right_col[best] = c
        top = (c >> 6) + 1
        for r in range(n):
            if r != best and right_col[r] < 0 and rightmost[r] == c:
                for w in range(top):
                    rows[r, w] ^= rows[best, w]
                rightmost[r] = _rightmost_bit(rows[r], top)
    for r in range(n):
        left[r] = left_col[r] >> 1
        right[r] = right_col[r] >> 1


@njit(cache=True)
def erasure_profiles(tab, n, ws, system, reference, samples, key, out):
    """``out[j, m] = I(R : first m qubits of random ordering j of the system)``."""
    rng = np.empty(4, dtype=np.uint64)
    stream_state(key, 0, rng)
    num_sys = system.shape[0]
    perm = system.copy()
    for j in range(samples):
        for t in range(num_sys - 1):
            r = t + next_below(rng, num_sys - t)
            tmp = perm[t]
            perm[t] = perm[r]
            perm[r] = tmp
        erasure_profile(tab, n, ws, perm, reference, out[j])


# -- reference rows (ensemble engine) -----------------------------------------------------
#
# Precondition for everything below: ``ref_rows`` marks the stabilizer rows with
# support on the reference qubits R, and their R-parts are linearly independent
# (true right after Bell-pairing the references, and kept true by
# ``measure_1q_phase_free_tracked`` while nothing acts on R). Then, for a set E
# of system qubits, S(R) = popcount(ref_rows) - |R| and
# I(R:E) = rank(E) - rank(E restricted to the unmarked rows).


@njit(cache=True)
def reference_rows(tab, ws, reference, out):
    """``out`` = the stabilizer rows with support on ``reference`` (see the precondition)."""
    for w in range(ws):
        out[w] = ZERO
    for j in range(reference.shape[0]):
        r = reference[j]
        for w in range(ws):
            out[w] |= tab[2 * r, w] | tab[2 * r + 1, w]


@njit(cache=True)
def lost_information_rows(tab, ws, erased, ref_rows, scratch):
    """I(R:E) for system qubits ``erased``; ``scratch`` has shape >= (2 |E|, ws)."""
    return columns_rank(tab, erased, ws, scratch) - columns_rank_without(
        tab, erased, ws, ref_rows, scratch
    )


@njit(cache=True)
def erasure_profile_rows(tab, ws, order, ref_rows, out):
    """``out[m] = I(R : order[:m])`` for m = 0..len(order), in one incremental pass."""
    cap = 64 * ws + 1
    vec = np.zeros(ws, dtype=np.uint64)
    all_basis = np.zeros((cap, ws), dtype=np.uint64)
    all_owner = np.full(64 * ws, -1, dtype=np.int64)
    all_num = np.zeros(1, dtype=np.int64)
    free_basis = np.zeros((cap, ws), dtype=np.uint64)
    free_owner = np.full(64 * ws, -1, dtype=np.int64)
    free_num = np.zeros(1, dtype=np.int64)
    lost = 0
    out[0] = 0
    for m in range(order.shape[0]):
        q = order[m]
        lost += basis_add_qubit(tab, q, ws, all_basis, all_owner, all_num, vec)
        lost -= basis_add_qubit_without(tab, q, ws, ref_rows, free_basis, free_owner, free_num, vec)
        out[m + 1] = lost


@njit(cache=True)
def contiguous_distance_rows(tab, ws, system, ref_rows, num_ref, periodic):
    """:func:`contiguous_distance` for the engine: smallest run of ``system`` with I(R:A) > 0.

    Returns 0 if the code dimension is 0. Each start grows its run one qubit
    at a time and stops once it cannot beat the best length so far.
    """
    k = -num_ref
    for w in range(ws):
        k += np.int64(popcount(ref_rows[w]))
    if k == 0:
        return 0
    num_sys = system.shape[0]
    cap = 64 * ws + 1
    vec = np.zeros(ws, dtype=np.uint64)
    all_basis = np.zeros((cap, ws), dtype=np.uint64)
    all_owner = np.full(64 * ws, -1, dtype=np.int64)
    all_num = np.zeros(1, dtype=np.int64)
    free_basis = np.zeros((cap, ws), dtype=np.uint64)
    free_owner = np.full(64 * ws, -1, dtype=np.int64)
    free_num = np.zeros(1, dtype=np.int64)
    best = num_sys
    for s in range(num_sys):
        basis_reset(all_owner, all_num)
        basis_reset(free_owner, free_num)
        lost = 0
        max_len = best - 1
        if not periodic:
            max_len = min(max_len, num_sys - s)
        for length in range(1, max_len + 1):
            q = system[(s + length - 1) % num_sys]
            lost += basis_add_qubit(tab, q, ws, all_basis, all_owner, all_num, vec)
            lost -= basis_add_qubit_without(
                tab, q, ws, ref_rows, free_basis, free_owner, free_num, vec
            )
            if lost > 0:
                best = length
                break
    return best
