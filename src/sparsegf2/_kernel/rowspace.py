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
"""

from __future__ import annotations

import numpy as np
from numba import njit

from sparsegf2._kernel.bits import ONE, ZERO


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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
