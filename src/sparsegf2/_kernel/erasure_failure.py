"""Compiled kernels for the code side of geometry-correlated erasures (private).

The code side of :mod:`sparsegf2.erasures`: which erasure regions destroy logical
information. Notation: a code on the system qubits ``0..n-1`` whose logical
information is held by a reference ``R``; erasing a set ``A`` of system qubits
loses ``l(A) = I(A:R)`` bits, and fails when ``F(A) = 1[l(A) > 0]``. ``F`` is
monotone: a superset of a failing set fails (``l`` never decreases when qubits
are added).

The tableau arguments ``(tab, ws, ref_rows)`` follow the ``*_rows`` routines
of :mod:`.entropy`: ``tab`` is a column-major tableau whose rows outside the
mask ``ref_rows`` restrict to the code's stabilizer group ``S`` on the system,
and all of whose rows restrict to its normalizer ``N(S)``, so
``l(A) = rank(N(S)|_A) - rank(S|_A)``. The ensemble engine's live tableau with
its reference rows, and :class:`sparsegf2.codes.StabilizerCode` (its generators
then a basis of ``N(S)``, the latter marked), both have that form.

* :func:`region_failures`: ``F(A)`` for many regions of one size (the regions
  a sampler drew), a region of more than half the system through its
  complement.
* :func:`boundary_failure_counts`: every region of size ``m = 1..w`` with its
  edge boundary ``b`` on a graph, counted by ``(m, b)``: all regions
  ``N_m(b)`` and the failing ones ``U_m(b)``. Then, for every coupling ``J`` at
  once, ``P_J(m) = sum_b U_m(b) exp(-J b) / sum_b N_m(b) exp(-J b)``.
* :func:`window_first_failure`: for every start along an order of the qubits,
  the length of the shortest failing contiguous window from there.

Nothing here draws random numbers.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from sparsegf2._kernel.bits import ZERO, ctz
from sparsegf2._kernel.entropy import basis_add, basis_clear, code_columns, lost_information_rows


@njit(cache=True)
def region_failures(tab, ws, ref_rows, n_sys, k, regions, scratch, out):
    """``out[r] = 1[I(A_r : R) > 0]`` for each row ``A_r`` of ``regions``.

    ``regions`` is int64[R, m]: each row lists ``m`` distinct system qubits of
    ``0..n_sys-1`` (any order); ``k = S(R)`` is the number of logical qubits.
    ``out`` is uint8[R]. ``scratch`` is uint64 of shape at least
    ``(max(2 min(m, n_sys - m), 1), ws)``. ``I(A:R)`` is
    :func:`.entropy.lost_information_rows`: the rank of the columns of ``A``
    over all rows minus their rank over the rows outside ``ref_rows``, two
    eliminations of ``2 m`` columns of ``ws`` words (``O(m^2 ws)``).

    A region of more than half the system is evaluated on its complement
    ``Ā`` (the other system qubits): the system and ``R`` together are in a
    pure state, so ``I(A:R) + I(Ā:R) = 2 S(R) = 2k`` (``S(AR) = S(Ā)`` and
    ``S(ĀR) = S(A)``; for a code this is the complementarity of the cleaning
    lemma, Bravyi and Terhal 2009), and ``I(A:R) > 0`` exactly when
    ``I(Ā:R) < 2k``. That bounds the eliminations by ``2 min(m, n_sys - m)``
    columns, about four times less work on average over ``m``. ``m = 0``
    gives 0, ``m = n_sys`` gives ``1[k > 0]``, and ``k = 0`` gives 0 without
    any elimination.
    """
    count = regions.shape[0]
    m = regions.shape[1]
    if k == 0:
        for r in range(count):
            out[r] = 0
        return
    flip = 2 * m > n_sys
    member = np.zeros(n_sys, dtype=np.uint8)
    other = np.empty(n_sys - m, dtype=np.int64)
    for r in range(count):
        if not flip:
            lost = lost_information_rows(tab, ws, regions[r], ref_rows, scratch)
            out[r] = 1 if lost > 0 else 0
            continue
        for j in range(m):
            member[regions[r, j]] = 1
        t = 0
        for q in range(n_sys):
            if member[q]:
                member[q] = 0
            else:
                other[t] = q
                t += 1
        lost = lost_information_rows(tab, ws, other, ref_rows, scratch)
        out[r] = 1 if lost < 2 * k else 0


@njit(cache=True)
def _truncate(basis, pivot_owner, num, keep, words):
    """Drop the basis vectors added after the first ``keep`` (an incremental basis is a stack).

    A vector stored by :func:`.entropy.basis_add` is nonzero, never changes, and
    its pivot is its lowest set bit, so forgetting it means clearing that bit's
    owner; the first ``keep`` vectors and their owners are exactly the basis as
    it was when it held ``keep`` vectors.
    """
    for i in range(keep, num[0]):
        w = 0
        while basis[i, w] == ZERO:
            w += 1
        pivot_owner[(w << 6) + np.int64(ctz(basis[i, w]))] = -1
    num[0] = keep


@njit(cache=True)
def _add_columns(cols, qubit, words, mask_last, basis, pivot_owner, num, vec):
    """Add the X and Z columns of ``qubit`` (cut to ``words`` words, the last masked); return the rank gain."""
    gained = 0
    for c in range(2 * qubit, 2 * qubit + 2):
        for v in range(words):
            vec[v] = cols[c, v]
        vec[words - 1] &= mask_last
        gained += basis_add(vec, basis, pivot_owner, num, words)
    return gained


@njit(cache=True)
def boundary_failure_counts(
    tab, ws, ref_rows, indptr, indices, n_sys, max_size, out_failed, out_total
):
    """Count every region of size ``m = 1..max_size`` by its boundary, and the failing ones.

    The regions are the ``m``-subsets ``A`` of the system qubits
    ``0..n_sys-1``; the graph (CSR, :func:`.erasure_chain.csr`) is on the same
    ``n_sys`` nodes, and ``b(A)`` is the number of its edges with exactly one
    end in ``A``. Outputs, int64 of shape ``(max_size, B + 1)`` with ``B`` at
    least the largest boundary of any region of size ``<= max_size`` (the sum
    of the ``max_size`` largest degrees, or the number of edges, bounds it)::

        out_total[m - 1, b]  = N_m(b) = #{A : |A| = m, b(A) = b}
        out_failed[m - 1, b] = U_m(b) = #{A : |A| = m, b(A) = b, I(A:R) > 0}

    so ``sum_b out_failed[m - 1, b]`` is the number of uncorrectable
    ``m``-subsets (:func:`.entropy.uncorrectable_counts_rows`) and
    ``sum_b out_total[m - 1, b] = C(n_sys, m)``.

    **Enumeration.** Depth first over the subsets in lexicographic order of
    their sorted qubits: the node for ``A = {a_1 < ... < a_j}`` extends the node
    for its first ``j - 1`` qubits by ``a_j``, so every subset of size
    ``1..max_size`` is visited exactly once. Along the current branch it keeps
    the boundary of each prefix, ``b(P ∪ {v}) = b(P) + deg(v) - 2 |N(v) ∩ P|``,
    and two incremental GF(2) bases on the columns of
    :func:`.entropy.code_columns` (``r = n + k`` bits; the first ``s`` of them
    span ``S``): the ``S``-part and the full columns of the prefix's qubits, so
    ``l(P) = rank_N - rank_S`` costs two column additions per node, undone
    (stack truncation) when the branch backs up.

    **Monotonicity.** ``F(A) = 1[l(A) > 0]`` never decreases when qubits are
    added, so once a prefix fails every region below it in the tree fails: its
    rank computations are skipped (its boundary never is). With no logical
    qubits (``r == s``) nothing fails and no rank is computed.
    """
    for i in range(out_total.shape[0]):
        for b in range(out_total.shape[1]):
            out_total[i, b] = 0
            out_failed[i, b] = 0
    w = max_size
    if w <= 0 or n_sys == 0:
        return
    system = np.arange(n_sys)
    cols, s, r = code_columns(tab, ws, system, ref_rows)
    has_code = r > s
    wa = cols.shape[1]
    wf = max((s + 63) // 64, 1)
    fmask = ZERO
    if s > 0:
        fmask = ~ZERO if s % 64 == 0 else (np.uint64(1) << np.uint64(s % 64)) - np.uint64(1)
    cap = 2 * w
    fb = np.zeros((cap, wf), dtype=np.uint64)
    fo = np.full(64 * wf, -1, dtype=np.int64)
    fn = np.zeros(1, dtype=np.int64)
    ab = np.zeros((cap, wa), dtype=np.uint64)
    ao = np.full(64 * wa, -1, dtype=np.int64)
    an = np.zeros(1, dtype=np.int64)
    fvec = np.zeros(wf, dtype=np.uint64)
    avec = np.zeros(wa, dtype=np.uint64)
    member = np.zeros(n_sys, dtype=np.uint8)
    cand = np.zeros(w + 1, dtype=np.int64)  # cand[j]: the qubit tried at depth j
    partial = np.zeros(w + 1, dtype=np.int64)  # partial[j]: b of the first j qubits
    failed = np.zeros(w + 1, dtype=np.bool_)  # failed[j]: the first j qubits fail
    fsave = np.zeros(w + 1, dtype=np.int64)  # basis sizes before the qubit at depth j
    asave = np.zeros(w + 1, dtype=np.int64)
    lvl = 0  # the committed prefix is cand[0..lvl-1]; member and the bases hold exactly it
    while lvl >= 0:
        i = cand[lvl]
        if i >= n_sys:  # no qubit left at this depth: back up and advance the parent
            lvl -= 1
            if lvl >= 0:
                q = cand[lvl]
                member[q] = 0
                _truncate(fb, fo, fn, fsave[lvl], wf)
                _truncate(ab, ao, an, asave[lvl], wa)
                cand[lvl] = q + 1
            continue
        size = lvl + 1
        inner = 0
        for j in range(indptr[i], indptr[i + 1]):
            inner += member[indices[j]]
        b = partial[lvl] + (indptr[i + 1] - indptr[i]) - 2 * inner
        partial[size] = b
        member[i] = 1
        fsave[lvl] = fn[0]
        asave[lvl] = an[0]
        f = failed[lvl]
        if has_code and not f:
            _add_columns(cols, i, wf, fmask, fb, fo, fn, fvec)
            _add_columns(cols, i, wa, ~ZERO, ab, ao, an, avec)
            f = an[0] > fn[0]  # rank(N(S)|_A) > rank(S|_A)
        failed[size] = f
        out_total[size - 1, b] += 1
        if f:
            out_failed[size - 1, b] += 1
        if size < w and i + 1 < n_sys:  # descend: keep i in the prefix
            lvl = size
            cand[lvl] = i + 1
            continue
        member[i] = 0
        _truncate(fb, fo, fn, fsave[lvl], wf)
        _truncate(ab, ao, an, asave[lvl], wa)
        cand[lvl] = i + 1


@njit(cache=True)
def window_first_failure(tab, ws, ref_rows, order, periodic, out):
    """``out[s]`` = the length of the shortest failing window of ``order`` that starts at ``s``.

    ``order`` (int64[n]) is a permutation of the system qubits; the window of
    length ``L`` at start ``s`` is ``order[s], order[s+1], ..., order[s+L-1]``,
    read cyclically (indices mod ``n``) when ``periodic`` and stopping at the
    end of the order when not. ``out`` is int64[n], one entry per start
    ``s = 0..n-1``: the smallest ``L`` (at most ``n``, or ``n - s`` for an open
    window) with ``I(window : R) > 0``, else that largest length plus one
    (``n + 1`` on the ring; on the ring a code with logical qubits always fails
    at ``L = n``, the whole system).

    Each start grows its window one qubit at a time on the columns of
    :func:`.entropy.code_columns`, keeping the rank of the window's
    ``S``-columns; while it is full (``2 L``) the window is correctable and the
    ``N(S)`` rank is not needed. At the first deficient length the ``N(S)``
    basis is built for the window and kept up from then on, and the scan stops
    at the first ``L`` with ``rank_N > rank_S``. Cost ``O(n)`` incremental steps
    per start. With no logical qubits nothing fails and no rank is computed.
    """
    n = order.shape[0]
    cols, s, r = code_columns(tab, ws, order, ref_rows)
    if r == s:
        for start in range(n):
            out[start] = n + 1 if periodic else n - start + 1
        return
    wa = cols.shape[1]
    wf = max((s + 63) // 64, 1)
    fmask = ZERO
    if s > 0:
        fmask = ~ZERO if s % 64 == 0 else (np.uint64(1) << np.uint64(s % 64)) - np.uint64(1)
    fb = np.zeros((max(s, 1), wf), dtype=np.uint64)
    fo = np.full(64 * wf, -1, dtype=np.int64)
    fn = np.zeros(1, dtype=np.int64)
    ab = np.zeros((r, wa), dtype=np.uint64)
    ao = np.full(64 * wa, -1, dtype=np.int64)
    an = np.zeros(1, dtype=np.int64)
    fvec = np.zeros(wf, dtype=np.uint64)
    avec = np.zeros(wa, dtype=np.uint64)
    for start in range(n):
        basis_clear(fb, fo, fn, wf)
        basis_clear(ab, ao, an, wa)
        limit = n if periodic else n - start
        result = limit + 1
        built = False
        for length in range(1, limit + 1):
            c = (start + length - 1) % n  # the column index of the newest qubit
            _add_columns(cols, c, wf, fmask, fb, fo, fn, fvec)
            if fn[0] == 2 * length:
                continue  # full S-rank: correctable
            if not built:
                for e in range(length - 1):
                    _add_columns(cols, (start + e) % n, wa, ~ZERO, ab, ao, an, avec)
                built = True
            _add_columns(cols, c, wa, ~ZERO, ab, ao, an, avec)
            if an[0] > fn[0]:
                result = length
                break
        out[start] = result
