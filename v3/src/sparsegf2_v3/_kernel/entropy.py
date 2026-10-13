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


@njit(cache=True)
def basis_clear(basis, pivot_owner, num, ws):
    """Empty an incremental basis in O(num * ws) (only the owners of its pivots are reset).

    A stored vector never changes after insertion, so its pivot is its lowest set bit.
    """
    for i in range(num[0]):
        for w in range(ws):
            x = basis[i, w]
            if x != ZERO:
                pivot_owner[(w << 6) + np.int64(ctz(x))] = -1
                break
    num[0] = 0


@njit(cache=True)
def _small_add(vec, basis, pw, pm, top, ws):
    """Reduce ``vec`` against basis vectors [0, top) (insertion-order echelon); append if independent.

    Each stored vector has zeros at the pivots of all earlier ones, so one
    pass in insertion order clears every pivot. Returns the new top.
    """
    for b in range(top):
        if vec[pw[b]] & pm[b]:
            for w in range(ws):
                vec[w] ^= basis[b, w]
    for w in range(ws):
        x = vec[w]
        if x != ZERO:
            for v in range(ws):
                basis[top, v] = vec[v]
            pw[top] = w
            pm[top] = x & (~x + np.uint64(1))
            return top + 1
    return top


@njit(cache=True)
def code_columns(tab, ws, system, ref_rows):
    """The columns of ``system`` in a row basis adapted to the code; returns ``(cols, s, r)``.

    The generator rows restricted to ``system`` are row-reduced, the rows
    outside ``ref_rows`` first: the first ``s`` basis rows span the code's
    stabilizers S and all ``r`` span N(S). Row ``2 i + c`` of ``cols`` is the X
    (c = 0) or Z (c = 1) column of qubit ``system[i]`` over those ``r`` rows, so
    for any set A of qubits ``rank(N(S)|_A)`` is the rank of its columns and
    ``rank(S|_A)`` the rank of their first ``s`` bits. Ranks of column sets are
    unchanged by row operations, and the columns are ``ceil(r / 64)`` words
    instead of ``ws``: ``r = n + k`` rows instead of the tableau's ``n + n_ref``.
    """
    n = system.shape[0]
    nc = 2 * n
    wr = (nc + 63) // 64
    nrows = 64 * ws
    echelon = np.zeros((min(nrows, nc), wr), dtype=np.uint64)
    pw = np.zeros(min(nrows, nc), dtype=np.int64)
    pm = np.zeros(min(nrows, nc), dtype=np.uint64)
    row = np.zeros(wr, dtype=np.uint64)
    rank = 0
    s = 0
    for free_pass in range(2):
        for r in range(nrows):
            rw = r >> 6
            rb = np.uint64(1) << np.uint64(r & 63)
            if ((ref_rows[rw] & rb) == ZERO) != (free_pass == 0):
                continue
            for v in range(wr):
                row[v] = ZERO
            nonzero = False
            for j in range(nc):
                if tab[2 * system[j >> 1] + (j & 1), rw] & rb:
                    row[j >> 6] |= np.uint64(1) << np.uint64(j & 63)
                    nonzero = True
            if nonzero and rank < echelon.shape[0]:
                rank = _small_add(row, echelon, pw, pm, rank, wr)
        if free_pass == 0:
            s = rank
    wc = max((rank + 63) // 64, 1)
    cols = np.zeros((nc, wc), dtype=np.uint64)
    for i in range(rank):
        ib = np.uint64(1) << np.uint64(i & 63)
        for v in range(wr):
            x = echelon[i, v]
            while x != ZERO:
                j = 64 * v + ctz(x)
                cols[j, i >> 6] |= ib
                x &= x - np.uint64(1)
    return cols, s, rank


@njit(cache=True)
def _vector_hash(vec, length):
    h = np.uint64(0x9E3779B97F4A7C15)
    for w in range(length):
        h = (h ^ vec[w]) * np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(29)
    return h


@njit(cache=True)
def _column_table(cols, length, mask_last):
    """Open-addressing multimap from the X, Z and Y columns of every qubit to (qubit, kind).

    ``keys[3 c + j]`` is column j of qubit c (X, Z, X^Z), cut to ``length``
    words with ``mask_last`` applied to the last word; ``slots`` holds key
    indices (-1 empty) at their hash positions, probed linearly.
    """
    n = cols.shape[0] // 2
    keys = np.zeros((3 * n, length), dtype=np.uint64)
    hashes = np.zeros(3 * n, dtype=np.uint64)
    for c in range(n):
        for v in range(length):
            x = cols[2 * c, v]
            z = cols[2 * c + 1, v]
            if v == length - 1:
                x &= mask_last
                z &= mask_last
            keys[3 * c, v] = x
            keys[3 * c + 1, v] = z
            keys[3 * c + 2, v] = x ^ z
        for j in range(3):
            hashes[3 * c + j] = _vector_hash(keys[3 * c + j], length)
    size = 4
    while size < 4 * 3 * n:
        size *= 2
    slots = np.full(size, -1, dtype=np.int64)
    for e in range(3 * n):
        slot = np.int64(hashes[e] & np.uint64(size - 1))
        while slots[slot] >= 0:
            slot = (slot + 1) & (size - 1)
        slots[slot] = e
    return keys, hashes, slots


@njit(cache=True)
def _span_hits(basis, top, length, keys, hashes, slots, after, hits, require, cur):
    """Mark ``hits[c] |= 1 << j`` for every column (c, j), c > ``after``, inside span(basis[:top]).

    Walks the ``2**top`` elements of the span in Gray-code order and looks
    each up in the table of :func:`_column_table`. With ``require`` given
    (non-empty), only qubits with ``require[c] != 0`` are marked. Nothing is
    returned; ``cur`` is work space of the key length.
    """
    size = slots.shape[0]
    for v in range(length):
        cur[v] = ZERO
    for g in range(1 << top):
        if g > 0:
            b = ctz(np.uint64(g))
            for v in range(length):
                cur[v] ^= basis[b, v]
        h = _vector_hash(cur, length)
        slot = np.int64(h & np.uint64(size - 1))
        while slots[slot] >= 0:
            e = slots[slot]
            slot = (slot + 1) & (size - 1)
            if hashes[e] != h:
                continue
            c = e // 3
            if c <= after or (require.shape[0] > 0 and require[c] == 0):
                continue
            same = True
            for v in range(length):
                if keys[e, v] != cur[v]:
                    same = False
                    break
            if same:
                hits[c] |= np.uint8(1) << np.uint8(e - 3 * c)


@njit(cache=True)
def uncorrectable_counts_rows(tab, ws, system, ref_rows, max_size, out):
    """``out[m - 1]`` = number of m-subsets of ``system`` whose erasure leaks logical information.

    Exhaustive over m = 1..``max_size``, on the columns of :func:`code_columns`,
    with ``l(A) = rank(N(S)|_A) - rank(S|_A)``:

    * Prefixes of up to ``max_size - 1`` qubits are walked depth-first in
      increasing index order, keeping small echelon bases of their S and N(S)
      columns. ``l`` only grows with the set: once a prefix fails, all of its
      extensions fail too and are counted without being visited. A prefix
      whose S-columns have full rank ``2 |A|`` is correctable without its
      N(S) rank (both are at most ``2 |A|``), which is built only when needed.
    * The last qubit c of a correctable prefix P is not looped over. Adding c
      raises a rank by ``2 - dim K``, with K the pairs (a, b) such that
      ``a x_c + b z_c`` lies in the span of P's columns, so ``l(P + c) > 0``
      exactly when ``K_S != K_N``. Enumerating the at most ``4**|P|``
      elements of P's S-span and looking each up in a hash table of all
      X, Z and Y columns finds the few c with ``K_S`` non-empty; only those
      are checked against the N(S)-span.

    The counts are float64: exact while below 2**53 (always when every
    C(n, m), m <= ``max_size``, is), rounded beyond.
    """
    n = system.shape[0]
    w = max_size
    for m in range(w):
        out[m] = 0.0
    if w <= 0 or n == 0:
        return
    cols, s, r = code_columns(tab, ws, system, ref_rows)
    if r == s:
        return  # no logical qubits: every erasure is correctable
    wa = cols.shape[1]
    wf = max((s + 63) // 64, 1)
    fmask = ZERO
    if s > 0:
        fmask = ~ZERO if s % 64 == 0 else (np.uint64(1) << np.uint64(s % 64)) - np.uint64(1)
    skeys, shashes, sslots = _column_table(cols, wf, fmask)
    nkeys, nhashes, nslots = _column_table(cols, wa, ~ZERO)
    comb = np.zeros((n + 1, w + 1), dtype=np.float64)  # comb[a, j] = C(a, j), exact below 2**53
    for a in range(n + 1):
        comb[a, 0] = 1.0
        for j in range(1, min(a, w) + 1):
            comb[a, j] = comb[a - 1, j - 1] + (comb[a - 1, j] if j <= a - 1 else 0.0)
    cap = max(2 * (w - 1), 1)
    ab = np.zeros((cap, wa), dtype=np.uint64)
    apw = np.zeros(cap, dtype=np.int64)
    apm = np.zeros(cap, dtype=np.uint64)
    fb = np.zeros((cap, wf), dtype=np.uint64)
    fpw = np.zeros(cap, dtype=np.int64)
    fpm = np.zeros(cap, dtype=np.uint64)
    atop = np.zeros(w + 1, dtype=np.int64)
    ftop = np.zeros(w + 1, dtype=np.int64)
    built = np.zeros(w + 1, dtype=np.bool_)  # atop[j] holds the N(S) rank of the j-prefix
    built[0] = True
    cand = np.zeros(w + 1, dtype=np.int64)
    avec = np.zeros(wa, dtype=np.uint64)
    fvec = np.zeros(wf, dtype=np.uint64)
    hits_s = np.zeros(n, dtype=np.uint8)
    hits_n = np.zeros(n, dtype=np.uint8)
    no_filter = np.zeros(0, dtype=np.uint8)
    lvl = 0  # the prefix being extended has lvl qubits: cand[0..lvl-1]
    while lvl >= 0:
        size = lvl + 1
        if size == w:
            # count the uncorrectable one-qubit extensions of the correctable prefix
            after = cand[lvl - 1] if lvl > 0 else -1
            _span_hits(fb, ftop[lvl], wf, skeys, shashes, sslots, after, hits_s, no_filter, fvec)
            any_hit = False
            for c in range(after + 1, n):
                if hits_s[c] != 0:
                    any_hit = True
                    break
            if any_hit:
                if not built[lvl]:
                    j0 = lvl
                    while not built[j0]:
                        j0 -= 1
                    for j in range(j0, lvl):
                        t = atop[j]
                        for c in range(2 * cand[j], 2 * cand[j] + 2):
                            for v in range(wa):
                                avec[v] = cols[c, v]
                            t = _small_add(avec, ab, apw, apm, t, wa)
                        atop[j + 1] = t
                        built[j + 1] = True
                _span_hits(ab, atop[lvl], wa, nkeys, nhashes, nslots, after, hits_n, hits_s, avec)
                for c in range(after + 1, n):
                    if hits_s[c] != hits_n[c]:
                        out[w - 1] += 1.0
                    hits_s[c] = 0
                    hits_n[c] = 0
            lvl -= 1
            if lvl >= 0:
                cand[lvl] += 1
            continue
        i = cand[lvl]
        if i >= n:
            lvl -= 1
            if lvl >= 0:
                cand[lvl] += 1
            continue
        top = ftop[lvl]
        for c in range(2 * i, 2 * i + 2):
            for v in range(wf):
                fvec[v] = cols[c, v]
            fvec[wf - 1] &= fmask
            top = _small_add(fvec, fb, fpw, fpm, top, wf)
        ftop[size] = top
        failed = False
        if top < 2 * size:  # S-rank deficient: compare with the N(S) rank
            if not built[lvl]:
                j0 = lvl
                while not built[j0]:
                    j0 -= 1
                for j in range(
                    j0, lvl
                ):  # rebuild the prefix's N(S) basis from the last valid level
                    t = atop[j]
                    for c in range(2 * cand[j], 2 * cand[j] + 2):
                        for v in range(wa):
                            avec[v] = cols[c, v]
                        t = _small_add(avec, ab, apw, apm, t, wa)
                    atop[j + 1] = t
                    built[j + 1] = True
            t = atop[lvl]
            for c in range(2 * i, 2 * i + 2):
                for v in range(wa):
                    avec[v] = cols[c, v]
                t = _small_add(avec, ab, apw, apm, t, wa)
            atop[size] = t
            built[size] = True
            failed = t > top
        else:
            built[size] = False
        if failed:  # this prefix is uncorrectable, so is every extension
            remaining = n - 1 - i
            for j in range(0, w - size + 1):
                out[size - 1 + j] += comb[remaining, j]
            cand[lvl] += 1
        else:
            lvl += 1
            cand[lvl] = i + 1


@njit(cache=True)
def code_snapshot_rows(tab, n_sys, n_rows, ref_rows, out):
    """Write the code's stabilizer generators row-major into ``out``; return how many.

    The generators are the rows outside ``ref_rows`` (no support on the
    reference). Row ``i`` of ``out`` holds the X bits of qubits 0..n_sys-1 in
    words ``[0, wsys)`` and the Z bits in ``[wsys, 2 wsys)``, with
    ``wsys = out.shape[1] // 2``.
    """
    wsys = out.shape[1] // 2
    count = 0
    for r in range(n_rows):
        rw = r >> 6
        rb = np.uint64(1) << np.uint64(r & 63)
        if ref_rows[rw] & rb:
            continue
        for v in range(2 * wsys):
            out[count, v] = ZERO
        for q in range(n_sys):
            qb = np.uint64(1) << np.uint64(q & 63)
            if tab[2 * q, rw] & rb:
                out[count, q >> 6] |= qb
            if tab[2 * q + 1, rw] & rb:
                out[count, wsys + (q >> 6)] |= qb
        count += 1
    return count


@njit(cache=True)
def first_failure_samples(tab, ws, ref_rows, k, prng, system, out):
    """``out[i]`` = first-failure size along a fresh uniformly random ordering of ``system``.

    The smallest m such that erasing the first m qubits of the ordering
    leaks logical information, found by one incremental scan per ordering
    (Fisher-Yates, ``len - 1`` draws from ``prng`` per sample) on the columns
    of :func:`code_columns`. While the S-columns of the prefix have full rank
    ``2 m`` the prefix is correctable and the N(S) basis is skipped; it is
    built once, at the first deficient prefix, after which S-rank deficiency
    persists. With no logical qubits (``k == 0``) every entry is
    ``len(system) + 1`` and no draws are made.
    """
    n = system.shape[0]
    if k == 0:
        for i in range(out.shape[0]):
            out[i] = n + 1
        return
    cols, s, r = code_columns(tab, ws, system, ref_rows)
    wa = cols.shape[1]
    wf = max((s + 63) // 64, 1)
    fmask = ZERO
    if s > 0:
        fmask = ~ZERO if s % 64 == 0 else (np.uint64(1) << np.uint64(s % 64)) - np.uint64(1)
    fb = np.zeros((max(s, 1), wf), dtype=np.uint64)
    fo = np.full(64 * wf, -1, dtype=np.int64)
    fn = np.zeros(1, dtype=np.int64)
    ab = np.zeros((max(r, 1), wa), dtype=np.uint64)
    ao = np.full(64 * wa, -1, dtype=np.int64)
    an = np.zeros(1, dtype=np.int64)
    fvec = np.zeros(wf, dtype=np.uint64)
    avec = np.zeros(wa, dtype=np.uint64)
    idx = np.arange(n)
    for i in range(out.shape[0]):
        for t in range(n - 1):
            j = t + next_below(prng, n - t)
            tmp = idx[t]
            idx[t] = idx[j]
            idx[j] = tmp
        basis_clear(fb, fo, fn, wf)
        basis_clear(ab, ao, an, wa)
        built = False
        result = n + 1
        rank_s = 0
        rank_n = 0
        for m in range(n):
            c = idx[m]
            for col in range(2 * c, 2 * c + 2):
                for v in range(wf):
                    fvec[v] = cols[col, v]
                fvec[wf - 1] &= fmask
                rank_s += basis_add(fvec, fb, fo, fn, wf)
            if rank_s == 2 * (m + 1):
                continue  # full S-rank: correctable
            if not built:
                for e in range(m):
                    for col in range(2 * idx[e], 2 * idx[e] + 2):
                        for v in range(wa):
                            avec[v] = cols[col, v]
                        rank_n += basis_add(avec, ab, ao, an, wa)
                built = True
            for col in range(2 * c, 2 * c + 2):
                for v in range(wa):
                    avec[v] = cols[col, v]
                rank_n += basis_add(avec, ab, ao, an, wa)
            if rank_n > rank_s:
                result = m + 1
                break
        out[i] = result


@njit(cache=True)
def symplectic_complement(rows, num_qubits):
    """Basis of the Paulis commuting with every row (packed: X words, then Z words).

    ``rows`` is ``(s, 2 wsys)`` uint64. Returns ``(basis, rank)``: the
    ``2 n - rank`` basis vectors of {v : <h, v> = 0 for every row h} in the same
    packing, and the GF(2) rank of ``rows``. With ``h = (x, z)``, the
    symplectic product <h, v> = x.v_z + z.v_x is the dot product of
    ``(z | x)`` with ``v``, so this is the null space of the half-swapped rows.
    """
    s = rows.shape[0]
    wsys = rows.shape[1] // 2
    width = 2 * num_qubits
    m = np.zeros((s, 2 * wsys), dtype=np.uint64)
    for i in range(s):
        for w in range(wsys):
            m[i, w] = rows[i, wsys + w]  # (z | x)
            m[i, wsys + w] = rows[i, w]
    pivots = np.zeros(width, dtype=np.int64)
    rank = 0
    for col in range(width):
        word = (col // num_qubits) * wsys + ((col % num_qubits) >> 6)
        bit = np.uint64(1) << np.uint64((col % num_qubits) & 63)
        found = -1
        for i in range(rank, s):
            if m[i, word] & bit:
                found = i
                break
        if found < 0:
            continue
        if found != rank:
            for w in range(2 * wsys):
                tmp = m[found, w]
                m[found, w] = m[rank, w]
                m[rank, w] = tmp
        for i in range(s):
            if i != rank and (m[i, word] & bit):
                for w in range(2 * wsys):
                    m[i, w] ^= m[rank, w]
        pivots[rank] = col
        rank += 1
    is_pivot = np.zeros(width, dtype=np.bool_)
    for i in range(rank):
        is_pivot[pivots[i]] = True
    basis = np.zeros((width - rank, 2 * wsys), dtype=np.uint64)
    b = 0
    for col in range(width):
        if is_pivot[col]:
            continue
        word = (col // num_qubits) * wsys + ((col % num_qubits) >> 6)
        bit = np.uint64(1) << np.uint64((col % num_qubits) & 63)
        basis[b, word] |= bit
        for i in range(rank):
            if m[i, word] & bit:
                pc = pivots[i]
                basis[b, (pc // num_qubits) * wsys + ((pc % num_qubits) >> 6)] |= np.uint64(
                    1
                ) << np.uint64((pc % num_qubits) & 63)
        b += 1
    return basis, rank


@njit(cache=True)
def columns_from_rows(rows, num_qubits, out):
    """Column-major bitsets of packed rows: ``out[2q]`` / ``out[2q+1]`` = X / Z bit of qubit q per row."""
    wsys = rows.shape[1] // 2
    for c in range(out.shape[0]):
        for w in range(out.shape[1]):
            out[c, w] = ZERO
    for r in range(rows.shape[0]):
        rw = r >> 6
        rb = np.uint64(1) << np.uint64(r & 63)
        for q in range(num_qubits):
            qb = np.uint64(1) << np.uint64(q & 63)
            if rows[r, q >> 6] & qb:
                out[2 * q, rw] |= rb
            if rows[r, wsys + (q >> 6)] & qb:
                out[2 * q + 1, rw] |= rb


@njit(cache=True)
def generators_commute(rows):
    """Whether all packed rows (X words, then Z words) commute pairwise."""
    s = rows.shape[0]
    wsys = rows.shape[1] // 2
    for i in range(s):
        for j in range(i + 1, s):
            parity = 0
            for w in range(wsys):
                parity ^= (
                    popcount((rows[i, w] & rows[j, wsys + w]) ^ (rows[i, wsys + w] & rows[j, w]))
                    & 1
                )
            if parity:
                return False
    return True
