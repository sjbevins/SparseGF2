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
* :func:`boundary_columns`: how many boundary columns the count arrays of
  :func:`boundary_failure_counts` need (size them with it, from one place).

**Arguments are checked.** Under the JIT an array that is too small is not an
IndexError but a write past its end, so every kernel here validates what it
is given before it writes anything, and raises ``ValueError`` (in compiled code
too) when: the tableau has fewer than ``2 n_sys`` column rows or ``ws``
words, or ``ref_rows`` fewer than ``ws`` words; an output or ``scratch`` is
smaller than documented; the graph is not a CSR of ``n_sys`` nodes without
self-loops; a region or order lists a qubit outside the system or twice; or
the ``k`` passed to :func:`region_failures` is not the number of logical qubits
of the tableau (``I(system : R) != 2k``). In the engine's layout
``k = popcount(ref_rows) - num_ref`` (as
:func:`.entropy.contiguous_distance_rows` computes it); for a
:class:`sparsegf2.codes.StabilizerCode` it is ``num_logical``.

Nothing here draws random numbers.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from sparsegf2._kernel.bits import ZERO, ctz
from sparsegf2._kernel.entropy import basis_add, basis_clear, code_columns, lost_information_rows


@njit(cache=True)
def _check_tableau(tab, ws, ref_rows, n_sys):
    """Refuse a tableau that cannot hold ``n_sys`` system qubits of ``ws`` words."""
    if n_sys < 0 or ws < 1 or tab.shape[0] < 2 * n_sys or tab.shape[1] < ws:
        raise ValueError("the tableau needs at least 2 n_sys column rows of ws >= 1 words")
    if ref_rows.shape[0] < ws:
        raise ValueError("ref_rows needs at least ws words")


@njit(cache=True)
def _logical_bits(tab, ws, ref_rows, n_sys):
    """``I(system : R) = 2 S(R) = 2k`` (the system and ``R`` together are pure); 0 for no system."""
    everything = np.arange(n_sys)
    scratch = np.zeros((2 * n_sys, ws), dtype=np.uint64)
    return lost_information_rows(tab, ws, everything, ref_rows, scratch)


@njit(cache=True)
def region_failures(tab, ws, ref_rows, n_sys, k, regions, scratch, out):
    """``out[r] = 1[I(A_r : R) > 0]`` for each row ``A_r`` of ``regions``.

    ``regions`` is int64[R, m]: each row lists ``m`` distinct system qubits of
    ``0..n_sys-1`` (any order); ``k = S(R)`` is the number of logical qubits.
    ``out`` is uint8[R] (at least). ``scratch`` is uint64 of shape at least
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
    any further elimination.

    Checked first (``ValueError``, see the module docstring): the tableau
    sizes, ``scratch`` and ``out``, every region's qubits (in range, distinct),
    and ``k`` itself, by one elimination of all ``2 n_sys`` system columns
    (``I(system : R)`` must be ``2k``) when there is at least one region.
    """
    count = regions.shape[0]
    m = regions.shape[1]
    _check_tableau(tab, ws, ref_rows, n_sys)
    if m > n_sys:
        raise ValueError("region_failures: a region of m > n_sys distinct system qubits")
    if scratch.shape[0] < max(2 * min(m, n_sys - m), 1) or scratch.shape[1] < ws:
        raise ValueError(
            "region_failures: scratch needs max(2 min(m, n_sys - m), 1) rows of ws words"
        )
    if out.shape[0] < count:
        raise ValueError("region_failures: out needs one entry per region")
    member = np.zeros(n_sys, dtype=np.uint8)
    for r in range(count):
        for j in range(m):
            q = regions[r, j]
            if q < 0 or q >= n_sys:
                raise ValueError("region_failures: a region lists a qubit outside 0..n_sys-1")
            if member[q]:
                raise ValueError("region_failures: a region lists a qubit twice")
            member[q] = 1
        for j in range(m):
            member[regions[r, j]] = 0
    if count > 0 and _logical_bits(tab, ws, ref_rows, n_sys) != 2 * k:
        raise ValueError(
            "region_failures: k is not the number of logical qubits of this tableau "
            "(I(system : R) != 2k)"
        )
    if k == 0:
        for r in range(count):
            out[r] = 0
        return
    flip = 2 * m > n_sys
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
def _check_csr(indptr, indices, n_sys):
    """Refuse anything but the CSR form of a graph on ``n_sys`` nodes without self-loops."""
    if indptr.shape[0] != n_sys + 1 or indptr[0] != 0 or indptr[n_sys] != indices.shape[0]:
        raise ValueError("the graph must be in CSR form on n_sys nodes (indptr of n_sys + 1)")
    for v in range(n_sys):  # first, so that every index read below lies in indices
        if indptr[v + 1] < indptr[v]:
            raise ValueError("the graph's indptr must be nondecreasing")
    for v in range(n_sys):
        for j in range(indptr[v], indptr[v + 1]):
            w = indices[j]
            if w < 0 or w >= n_sys or w == v:
                raise ValueError("the graph's neighbours must be other nodes in 0..n_sys-1")


@njit(cache=True)
def boundary_columns(indptr, max_size):
    """The number of boundary columns ``B + 1`` that counts up to size ``max_size`` need.

    ``indptr`` is the CSR row pointer of the graph (:func:`.erasure_chain.csr`;
    every edge appears in the rows of both its ends). A region ``A`` with
    ``|A| <= w`` has ``b(A) <= sum of the degrees in A <= the sum of the w
    largest degrees``, and ``b(A) <= |E|``, so::

        B = min(|E|, sum of the max_size largest degrees)

    bounds every boundary :func:`boundary_failure_counts` can meet, which
    refuses arrays with fewer columns. Size the count arrays with this, as
    :meth:`sparsegf2.codes.StabilizerCode.boundary_failure_counts` and
    :func:`sparsegf2.codes.boundary_columns` do. Callable from compiled code.
    """
    n = indptr.shape[0] - 1
    degrees = np.empty(n, dtype=np.int64)
    for v in range(n):
        degrees[v] = indptr[v + 1] - indptr[v]
    degrees.sort()
    top = 0
    for j in range(min(max_size, n)):
        top += degrees[n - 1 - j]
    return min(indptr[n] // 2, top) + 1


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
    end in ``A``. Outputs, int64 arrays of one shape, at least
    ``(max_size, boundary_columns(indptr, max_size))`` (:func:`boundary_columns`:
    ``B = min(|E|, sum of the max_size largest degrees)`` bounds every
    boundary, so no count can fall outside; smaller arrays are refused with
    ``ValueError``, as are a malformed graph and a tableau too small for
    ``n_sys`` qubits); entries beyond are zeroed::

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
    _check_tableau(tab, ws, ref_rows, n_sys)
    _check_csr(indptr, indices, n_sys)
    if max_size < 0:
        raise ValueError("boundary_failure_counts: max_size must be non-negative")
    if out_failed.shape[0] != out_total.shape[0] or out_failed.shape[1] != out_total.shape[1]:
        raise ValueError("boundary_failure_counts: out_failed and out_total need the same shape")
    if out_total.shape[0] < max_size or out_total.shape[1] < boundary_columns(indptr, max_size):
        raise ValueError(
            "boundary_failure_counts: the count arrays need max_size rows and "
            "boundary_columns(indptr, max_size) columns"
        )
    for i in range(out_total.shape[0]):
        for b in range(out_total.shape[1]):
            out_total[i, b] = 0
            out_failed[i, b] = 0
    w = max_size
    if w == 0 or n_sys == 0:
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
    ``order`` must be a permutation of ``0..n-1`` and ``out`` hold ``n``
    entries; otherwise, or for a tableau too small, ``ValueError``.
    """
    n = order.shape[0]
    _check_tableau(tab, ws, ref_rows, n)
    if out.shape[0] < n:
        raise ValueError("window_first_failure: out needs one entry per start")
    seen = np.zeros(n, dtype=np.uint8)
    for j in range(n):
        q = order[j]
        if q < 0 or q >= n or seen[q]:
            raise ValueError("window_first_failure: order must be a permutation of 0..n-1")
        seen[q] = 1
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
