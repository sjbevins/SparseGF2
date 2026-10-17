"""Compiled GF(2) row-space kernels on packed rows (private).

A *row* is a vector of GF(2)^width stored in ``W`` uint64 words: coordinate ``c``, its *packed
position*, is bit ``c & 63`` of word ``c >> 6``. A phase-free Pauli operator on ``N`` qubits
uses the snapshot layout of :func:`~sparsegf2._kernel.entropy.joint_snapshot_rows`:
``wq = ceil(N / 64)`` words of X bits (``x_q`` at packed position ``q``) followed by ``wq``
words of Z bits (``z_q`` at packed position ``64 wq + q``). The positions ``N .. 64 wq - 1`` and
``64 wq + N .. 128 wq - 1`` are padding and stay zero.

A *column order* is an int64 array of distinct packed positions: the columns in priority
order. :func:`rref` brings rows to the reduced row echelon form for a column order (for the
canonical form of :class:`sparsegf2.tableau.PhaseFreeTableau`: ``x_0 .. x_{N-1}``, then
``z_0 .. z_{N-1}``; for the physical code: the reference columns first), and
:func:`rank_of_columns` gives the rank of the rows restricted to a set of columns (the
entropy of a region). Everything is exact GF(2) arithmetic on words; nothing draws random
numbers.

:func:`pack_msb_first` writes rows in the byte order of
:meth:`sparsegf2.tableau.PhaseFreeTableau.to_bytes`.

The one-row-space kernels release the GIL (``nogil=True``), so several Python threads can
run them at once. :func:`rref_batch`, :func:`physical_rows_batch` and
:func:`pack_msb_first_batch` work on many row spaces in one call, in parallel over them
(``prange``, every Numba thread): the canonicalization and serialization of a whole
evaluation of snapshots (:meth:`sparsegf2.tableau.PhaseFreeTableau.from_words_batch`,
:func:`sparsegf2.tableau.physical_tableaux`, :func:`sparsegf2.tableau.to_bytes_batch`). They
keep the GIL, as the engine's parallel kernels do: two parallel launches from two Python
threads at once abort the process under Numba's ``workqueue`` threading layer, and holding the
GIL makes such calls take turns. Each item of a batch goes through the same :func:`rref` and
:func:`pack_msb_first` as one row space alone, so the results are the same bit for bit.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from sparsegf2._kernel.bits import ONE, ZERO

# REVERSED[b] is the byte b with its bit order reversed (bit i moves to bit 7 - i): a byte of
# little-endian packed bits read most significant bit first.
REVERSED = np.array([int(f"{b:08b}"[::-1], 2) for b in range(256)], dtype=np.uint8)


@njit(cache=True, nogil=True)
def rref(rows, cols, pivots):
    """Reduce ``rows`` in place to reduced row echelon form for the column order ``cols``.

    Gauss-Jordan elimination over GF(2): for each column ``cols[j]`` in turn, the first row
    at or after position ``rank`` with a 1 there becomes the next pivot row. It is swapped to
    position ``rank`` and added (XOR) to every other row with a 1 in that column, and
    ``pivots[rank] = j`` records the column's index in ``cols``. Returns the rank.

    Afterwards rows ``[0, rank)`` are the reduced row echelon form: row ``i`` is zero in the
    columns ``cols[:pivots[i]]``, has a 1 in ``cols[pivots[i]]``, and that column is zero in
    every other row; ``pivots[:rank]`` increases. Rows ``[rank, len(rows))`` are zero in every
    column of ``cols``, so they are zero rows when ``cols`` lists every column that can be
    nonzero. Only row operations are applied, so the row space is unchanged. ``pivots`` needs
    room for ``min(len(rows), len(cols))`` entries. Cost: ``O(len(cols) * len(rows) * W)``
    word operations.
    """
    num_rows = rows.shape[0]
    words = rows.shape[1]
    rank = 0
    for j in range(cols.shape[0]):
        if rank == num_rows:
            break
        c = cols[j]
        w = c >> 6
        bit = ONE << np.uint64(c & 63)
        found = -1
        for i in range(rank, num_rows):
            if (rows[i, w] & bit) != ZERO:
                found = i
                break
        if found < 0:
            continue
        if found != rank:
            for v in range(words):
                tmp = rows[found, v]
                rows[found, v] = rows[rank, v]
                rows[rank, v] = tmp
        for i in range(num_rows):
            if i != rank and (rows[i, w] & bit) != ZERO:
                for v in range(words):
                    rows[i, v] ^= rows[rank, v]
        pivots[rank] = j
        rank += 1
    return rank


@njit(cache=True, nogil=True)
def _rank_in_place(m, count, words):
    """Rank of the first ``count`` rows of ``m`` (destroyed).

    Each row in turn, if nonzero, takes its lowest set bit as its pivot and clears it from
    the rows after it. A row that is nonzero at its turn has a pivot bit that no later row
    has, so the rows that are nonzero at their turn are independent, and the row operations
    keep the rank: their number is the rank.
    """
    rank = 0
    for i in range(count):
        lead_w = -1
        for v in range(words):
            if m[i, v] != ZERO:
                lead_w = v
                break
        if lead_w < 0:
            continue
        rank += 1
        x = m[i, lead_w]
        lead = x & (~x + ONE)
        for j in range(i + 1, count):
            if (m[j, lead_w] & lead) != ZERO:
                for v in range(lead_w, words):
                    m[j, v] ^= m[i, v]
    return rank


@njit(cache=True, nogil=True)
def rank_of_columns(rows, count, cols, scratch):
    """GF(2) rank of the first ``count`` rows restricted to the packed positions ``cols``.

    The rows are copied into ``scratch`` (shape at least ``(count, W)``) with every column
    outside ``cols`` cleared, then eliminated (:func:`_rank_in_place`). Clearing columns is a
    projection, which commutes with row operations, so the result is the rank of the submatrix
    on the columns ``cols`` (in any order). Cost: ``O(count * W * (1 + rank))``.
    """
    words = rows.shape[1]
    mask = np.zeros(words, dtype=np.uint64)
    for j in range(cols.shape[0]):
        c = cols[j]
        mask[c >> 6] |= ONE << np.uint64(c & 63)
    for i in range(count):
        for v in range(words):
            scratch[i, v] = rows[i, v] & mask[v]
    return _rank_in_place(scratch, count, words)


@njit(parallel=True, cache=True)
def rref_batch(rows, counts, cols, ranks, pivots):
    """:func:`rref` of many row spaces at once, in parallel over them.

    ``rows`` is ``(B, R, W)``: item ``b`` is its first ``counts[b]`` rows (``counts[b] <= R``),
    reduced in place by :func:`rref` for the column order ``cols``; ``ranks[b]`` receives the
    rank and ``pivots[b, :rank]`` the pivots (``pivots`` is ``(B, P)`` with
    ``P >= min(R, len(cols))``). A negative ``counts[b]`` (a snapshot not saved) leaves item
    ``b`` alone and sets ``ranks[b] = -1``. Items share nothing, so the result does not depend
    on the number of threads.
    """
    for b in prange(rows.shape[0]):
        count = counts[b]
        if count < 0:
            ranks[b] = -1
            continue
        ranks[b] = rref(rows[b, :count], cols, pivots[b])


@njit(parallel=True, cache=True)
def physical_rows_batch(
    rows, ranks, ref_first, boundary, wq, wsys, system_cols, out, out_ranks, pivots
):
    """The physical rows of many joint row spaces at once, in parallel over them.

    Item ``b`` is ``rows[b, :ranks[b]]`` (destroyed; a negative rank gives
    ``out_ranks[b] = -1``), packed with ``wq`` words per half. It is reduced by :func:`rref`
    with the column order ``ref_first`` (the reference columns first, then the system
    columns); the rows whose pivot index is at least ``boundary`` (the number of reference
    columns) are zero on the reference, and their first ``wsys`` words of each half (the
    system qubits) are copied to ``out[b]`` (``(B, S, 2 wsys)``, ``S`` at least the number of
    such rows), which is then reduced by :func:`rref` for ``system_cols``. ``out_ranks[b]`` is
    the rank of the result. ``pivots`` is ``(B, P)`` with ``P`` at least every rank.
    """
    for b in prange(rows.shape[0]):
        r = ranks[b]
        if r < 0:
            out_ranks[b] = -1
            continue
        piv = pivots[b]
        rank = rref(rows[b, :r], ref_first, piv)
        first = 0
        while first < rank and piv[first] < boundary:
            first += 1
        count = rank - first
        for i in range(count):
            for v in range(wsys):
                out[b, i, v] = rows[b, first + i, v]
                out[b, i, wsys + v] = rows[b, first + i, wq + v]
        out_ranks[b] = rref(out[b, :count], system_cols, piv)


@njit(cache=True, nogil=True)
def pack_msb_first(rows, count, n, out):
    """Write the first ``count`` rows on ``n`` qubits as bytes, most significant bit first.

    ``rows`` are packed with ``wq = rows.shape[1] // 2`` words per half (bits beyond qubit
    ``n - 1`` zero). Row ``i`` of ``out`` (uint8, ``ceil(2 n / 8)`` bytes per row) receives the
    ``2 n`` bits ``x_0 .. x_{n-1}, z_0 .. z_{n-1}`` in this order, bit ``j`` at bit ``7 - (j & 7)``
    of byte ``j >> 3``, zero padded at the end: ``np.packbits(row, bitorder="big")``. Each byte
    of a half is one byte of its little-endian words with its bits reversed; the Z half starts
    at bit ``n``, so when ``n`` is not a multiple of 8 its bytes straddle two output bytes.
    Cost: ``O(count * n / 8)``.
    """
    wq = rows.shape[1] // 2
    width = out.shape[1]
    half_bytes = (n + 7) >> 3
    base = n >> 3
    shift = n & 7
    for i in range(count):
        for v in range(width):
            out[i, v] = 0
        for j in range(half_bytes):
            word = rows[i, j >> 3] >> np.uint64(8 * (j & 7))
            out[i, j] = REVERSED[np.int64(word & np.uint64(0xFF))]
        for j in range(half_bytes):
            word = rows[i, wq + (j >> 3)] >> np.uint64(8 * (j & 7))
            byte = np.int64(REVERSED[np.int64(word & np.uint64(0xFF))])
            if shift == 0:
                out[i, base + j] = byte
            else:
                out[i, base + j] |= byte >> shift
                if base + j + 1 < width:
                    out[i, base + j + 1] |= (byte << (8 - shift)) & 0xFF


@njit(parallel=True, cache=True)
def pack_msb_first_batch(rows, counts, n, out):
    """:func:`pack_msb_first` of ``rows[b, :counts[b]]`` into ``out[b]`` for every ``b``, in
    parallel (``rows`` ``(B, R, 2 wq)``, ``out`` ``(B, R, ceil(2 n / 8))``)."""
    for b in prange(rows.shape[0]):
        pack_msb_first(rows[b], counts[b], n, out[b])
