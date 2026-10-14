"""Compiled kernels on the column-major, bit-packed stabilizer tableau.

State layout
------------
The whole state of an ``n``-qubit simulator is one ``uint64`` array ``tab`` of
shape ``(2 n + 1, W)``::

    tab[2 q]       X column of qubit q: bit r is set iff generator r has X or Y on q
    tab[2 q + 1]   Z column of qubit q: bit r is set iff generator r has Z or Y on q
    tab[2 n]       sign row: bit r is set iff generator r has sign -1

A generator ("row") ``r`` lives at bit ``r & 63`` of word ``r >> 6``:

* stabilizer ``i`` is row ``i`` (words ``[0, Ws)``, ``Ws = ceil(n / 64)``);
* in phase-tracking mode, destabilizer ``i`` is row ``64 * Ws + i`` (words
  ``[Ws, 2 Ws)``), i.e. the destabilizer block starts on a word boundary and
  destabilizer ``i`` sits at the same bit position as its stabilizer partner.

``W = Ws`` in phase-free mode and ``W = 2 Ws`` with phases. Only stabilizer
signs are maintained; destabilizer sign bits are never read. Padding bits
(rows ``>= n`` inside a block) are always zero.

Storing columns (one bitset over generators per qubit) means a gate touches four
bitsets with word-parallel boolean algebra, and a measurement finds every
anticommuting generator in one bitset read. DESIGN.md section 5 has the
reasoning and the benchmarks behind this choice.

Pauli codes
-----------
Single-qubit Paulis are coded ``x + 2 z``: X = 1, Z = 2, Y = 3 (0 = identity).

Conventions for measurement kernels
-----------------------------------
``coin`` is the outcome to use *if* the measurement turns out to be random
(0 means eigenvalue +1). Kernels return ``outcome | (was_random << 1)``. In
phase-free mode the outcome bit is meaningless and callers must not expose it.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from sparsegf2._kernel.bits import ALL, ONE, ZERO, ctz, popcount, prefix_parity_exclusive

_U = np.uint64


# ---------------------------------------------------------------------------
# initialisation
# ---------------------------------------------------------------------------


@njit(cache=True)
def init_zero(tab, n, ws, phases):
    """Reset ``tab`` to |0...0>: stabilizers Z_i, destabilizers X_i, all signs +."""
    tab[:, :] = ZERO
    for i in range(n):
        w = i >> 6
        bit = ONE << _U(i & 63)
        tab[2 * i + 1, w] |= bit
        if phases:
            tab[2 * i, ws + w] |= bit


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


@njit(cache=True, inline="always")
def linear_monomials(s, k):
    """ANF bits contributed by the sign pattern ``s`` (bit j of s negates image j)."""
    out = 0
    for j in range(2 * k):
        if (s >> j) & 1:
            out |= 1 << (1 << j)
    return out


@njit(cache=True, inline="always")
def _mask(flag):
    return ALL if flag else ZERO


@njit(cache=True)
def apply_1q(tab, q, g, lin1, anf1, ident1, ws, phases):
    """Apply the 1-qubit Clifford with canonical index ``g`` (0..23) to qubit ``q``."""
    sym = g >> 2
    sign_row = tab.shape[0] - 1
    width = tab.shape[1]
    xr = 2 * q
    zr = xr + 1
    linear = sym != ident1
    m00 = lin1[sym, 0, 0]
    m01 = lin1[sym, 0, 1]
    m10 = lin1[sym, 1, 0]
    m11 = lin1[sym, 1, 1]
    anf = anf1[sym] ^ linear_monomials(g & 3, 1)
    a1 = _mask((anf >> 1) & 1)  # monomial x
    a2 = _mask((anf >> 2) & 1)  # monomial z
    a3 = _mask((anf >> 3) & 1)  # monomial x z
    track = phases and anf != 0
    for w in range(width):
        v0 = tab[xr, w]
        v1 = tab[zr, w]
        if track and w < ws:
            tab[sign_row, w] ^= (v0 & a1) ^ (v1 & a2) ^ (v0 & v1 & a3)
        if linear:
            tab[xr, w] = (v0 & m00) ^ (v1 & m10)
            tab[zr, w] = (v0 & m01) ^ (v1 & m11)


@njit(cache=True)
def apply_2q(tab, a, b, g, lin2, anf2, ident2, ws, phases):
    """Apply the 2-qubit Clifford with canonical index ``g`` (0..11519) to ``(a, b)``.

    Qubit ``a`` is the gate's first target (qubit 0 of the Clifford).
    """
    sym = g >> 4
    sign_row = tab.shape[0] - 1
    width = tab.shape[1]
    ra = 2 * a
    rb = 2 * b
    linear = sym != ident2
    m = lin2[sym]
    anf = anf2[sym] ^ linear_monomials(g & 15, 2)
    # ANF coefficient masks; monomial bit order v0=x_a, v1=x_b, v2=z_a, v3=z_b.
    c1 = _mask((anf >> 1) & 1)
    c2 = _mask((anf >> 2) & 1)
    c3 = _mask((anf >> 3) & 1)
    c4 = _mask((anf >> 4) & 1)
    c5 = _mask((anf >> 5) & 1)
    c6 = _mask((anf >> 6) & 1)
    c7 = _mask((anf >> 7) & 1)
    c8 = _mask((anf >> 8) & 1)
    c9 = _mask((anf >> 9) & 1)
    c10 = _mask((anf >> 10) & 1)
    c11 = _mask((anf >> 11) & 1)
    c12 = _mask((anf >> 12) & 1)
    c13 = _mask((anf >> 13) & 1)
    c14 = _mask((anf >> 14) & 1)
    c15 = _mask((anf >> 15) & 1)
    track = phases and anf != 0
    for w in range(width):
        v0 = tab[ra, w]
        v1 = tab[rb, w]
        v2 = tab[ra + 1, w]
        v3 = tab[rb + 1, w]
        if track and w < ws:
            p3 = v0 & v1
            p5 = v0 & v2
            p6 = v1 & v2
            p7 = p3 & v2
            p9 = v0 & v3
            p10 = v1 & v3
            p11 = p3 & v3
            p12 = v2 & v3
            p13 = p5 & v3
            p14 = p6 & v3
            p15 = p7 & v3
            tab[sign_row, w] ^= (
                (v0 & c1)
                ^ (v1 & c2)
                ^ (p3 & c3)
                ^ (v2 & c4)
                ^ (p5 & c5)
                ^ (p6 & c6)
                ^ (p7 & c7)
                ^ (v3 & c8)
                ^ (p9 & c9)
                ^ (p10 & c10)
                ^ (p11 & c11)
                ^ (p12 & c12)
                ^ (p13 & c13)
                ^ (p14 & c14)
                ^ (p15 & c15)
            )
        if linear:
            tab[ra, w] = (v0 & m[0, 0]) ^ (v1 & m[1, 0]) ^ (v2 & m[2, 0]) ^ (v3 & m[3, 0])
            tab[rb, w] = (v0 & m[0, 1]) ^ (v1 & m[1, 1]) ^ (v2 & m[2, 1]) ^ (v3 & m[3, 1])
            tab[ra + 1, w] = (v0 & m[0, 2]) ^ (v1 & m[1, 2]) ^ (v2 & m[2, 2]) ^ (v3 & m[3, 2])
            tab[rb + 1, w] = (v0 & m[0, 3]) ^ (v1 & m[1, 3]) ^ (v2 & m[2, 3]) ^ (v3 & m[3, 3])


@njit(cache=True)
def apply_pauli(tab, q, code, ws):
    """Apply the Pauli with code ``code`` (X=1, Z=2, Y=3) to qubit ``q`` (signs only)."""
    sign_row = tab.shape[0] - 1
    for w in range(ws):
        flip = ZERO
        if code & 1:  # X anticommutes with generators that have z on q
            flip ^= tab[2 * q + 1, w]
        if code & 2:  # Z anticommutes with generators that have x on q
            flip ^= tab[2 * q, w]
        tab[sign_row, w] ^= flip


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


@njit(cache=True)
def anticommuting_rows(tab, qs, codes, anti):
    """Fill ``anti`` with the mask of generators anticommuting with the Pauli ``prod_j codes[j] on qs[j]``."""
    width = tab.shape[1]
    for w in range(width):
        anti[w] = ZERO
    for j in range(qs.shape[0]):
        c = qs[j]
        code = codes[j]
        if code & 2:
            for w in range(width):
                anti[w] ^= tab[2 * c, w]
        if code & 1:
            for w in range(width):
                anti[w] ^= tab[2 * c + 1, w]


@njit(cache=True)
def product_exponent(tab, n, ws, sel, nz):
    """Power of i (mod 4) in ``prod_{i in sel} s_i = i**e * (x)_c sigma(A_c, B_c)``.

    ``sel`` is a mask over stabilizer rows (``ws`` words); signs are included.
    The selected stabilizers commute, so the product is order independent; it
    is evaluated in increasing row order. Per qubit, a product of Hermitian
    single-qubit Paulis ``sigma(x_j, z_j)`` equals
    ``i**(#Y - A B + 2 #{j<l: z_j x_l}) sigma(A, B)`` with ``A, B`` the parities
    of the x and z bits; all three counts are taken with popcounts. ``nz`` is an
    int64 scratch array of length >= ``ws``.
    """
    sign_row = tab.shape[0] - 1
    k = 0
    total = 0
    for w in range(ws):
        if sel[w] != ZERO:
            nz[k] = w
            k += 1
            total += 2 * np.int64(popcount(sel[w] & tab[sign_row, w]))
    for c in range(n):
        xr = 2 * c
        num_y = 0
        a_par = 0
        z_par = 0
        pairs = 0
        for j in range(k):
            w = nz[j]
            m = sel[w]
            x = tab[xr, w] & m
            z = tab[xr + 1, w] & m
            if (x | z) == ZERO:
                continue
            num_y += np.int64(popcount(x & z))
            px = np.int64(popcount(x))
            pairs ^= np.int64(popcount(x & prefix_parity_exclusive(z))) & 1
            if z_par:
                pairs ^= px & 1
            z_par ^= np.int64(popcount(z)) & 1
            a_par ^= px & 1
        total += num_y - a_par * z_par + 2 * pairs
    return total & 3


@njit(cache=True)
def collapse(tab, n, ws, phases, anti, qs, codes, coin, supp, lo, hi):
    """Measure the Pauli ``P = prod_j codes[j] on qs[j]`` given its anticommutation mask.

    ``anti`` must hold :func:`anticommuting_rows` for ``P``; it is used as
    scratch. ``supp`` (int64, length >= n) and ``lo``/``hi`` (uint64, length >=
    ws) are scratch. Returns ``outcome | (was_random << 1)``.

    Random case (Aaronson-Gottesman): the first anticommuting stabilizer is the
    pivot ``p``; it is multiplied into every other anticommuting generator (one
    XOR of the remaining mask into each column in the pivot's support, with
    signs tracked by 2-bit counters per row); in phases mode it becomes its own
    destabilizer partner; finally row ``p`` is replaced by ``(-1)**coin P``.
    """
    width = tab.shape[1]
    sign_row = tab.shape[0] - 1
    pivot = -1
    for w in range(ws):
        if anti[w] != ZERO:
            pivot = (w << 6) + np.int64(ctz(anti[w]))
            break
    if pivot < 0:
        if not phases:
            return 0
        e = product_exponent(tab, n, ws, anti[ws:], supp)
        return (e >> 1) & 1

    pw = pivot >> 6
    pbit = ONE << _U(pivot & 63)
    nbit = ~pbit
    anti[pw] &= nbit
    if phases:
        anti[ws + pw] &= nbit  # the pivot's destabilizer is overwritten below

    ns = 0
    for c in range(n):
        if ((tab[2 * c, pw] | tab[2 * c + 1, pw]) & pbit) != ZERO:
            supp[ns] = c
            ns += 1

    if phases:
        for w in range(ws):
            lo[w] = ZERO
            hi[w] = ZERO
        for j in range(ns):
            c = supp[j]
            px = (tab[2 * c, pw] & pbit) != ZERO
            pz = (tab[2 * c + 1, pw] & pbit) != ZERO
            for w in range(ws):
                msk = anti[w]
                if msk == ZERO:
                    continue
                x = tab[2 * c, w]
                z = tab[2 * c + 1, w]
                # factor of i from (row's Pauli on c) * (pivot's Pauli on c)
                if px and pz:  # pivot has Y: X*Y = +iZ, Z*Y = -iX
                    plus = x & ~z
                    minus = ~x & z
                elif px:  # pivot has X: Z*X = +iY, Y*X = -iZ
                    plus = ~x & z
                    minus = x & z
                else:  # pivot has Z: Y*Z = +iX, X*Z = -iY
                    plus = x & z
                    minus = x & ~z
                plus &= msk
                minus &= msk
                hi[w] ^= lo[w] & plus
                lo[w] ^= plus
                hi[w] ^= ~lo[w] & minus
                lo[w] ^= minus
        sp = _mask((tab[sign_row, pw] & pbit) != ZERO)
        for w in range(ws):
            tab[sign_row, w] ^= anti[w] & (hi[w] ^ sp)

    for j in range(ns):
        c = supp[j]
        if (tab[2 * c, pw] & pbit) != ZERO:
            for w in range(width):
                tab[2 * c, w] ^= anti[w]
        if (tab[2 * c + 1, pw] & pbit) != ZERO:
            for w in range(width):
                tab[2 * c + 1, w] ^= anti[w]

    if phases:
        dw = ws + pw
        for c in range(n):
            tab[2 * c, dw] = (tab[2 * c, dw] & nbit) | (tab[2 * c, pw] & pbit)
            tab[2 * c + 1, dw] = (tab[2 * c + 1, dw] & nbit) | (tab[2 * c + 1, pw] & pbit)

    for j in range(ns):
        c = supp[j]
        tab[2 * c, pw] &= nbit
        tab[2 * c + 1, pw] &= nbit
    for j in range(qs.shape[0]):
        c = qs[j]
        if codes[j] & 1:
            tab[2 * c, pw] |= pbit
        if codes[j] & 2:
            tab[2 * c + 1, pw] |= pbit
    if phases:
        if coin:
            tab[sign_row, pw] |= pbit
        else:
            tab[sign_row, pw] &= nbit
    return (coin & 1) | 2


@njit(cache=True)
def measure_1q_phase_free(tab, n, ws, q, code, anti, supp):
    """Fast path: measure X/Z/Y (code 1/2/3) on ``q`` in phase-free mode.

    Same projection as :func:`collapse` restricted to stabilizer rows, fused
    into one loop (no signs, no destabilizers); the pivot is the first
    anticommuting row. Returns ``2`` if the outcome was random, else ``0``.
    ``anti`` (length >= ws) and ``supp`` (length >= n) are scratch.
    """
    _anticommuting_1q(tab, ws, q, code, anti)
    pivot = -1
    for w in range(ws):
        if anti[w] != ZERO:
            pivot = (w << 6) + np.int64(ctz(anti[w]))
            break
    if pivot < 0:
        return 0
    _collapse_phase_free(tab, n, ws, q, code, anti, supp, pivot)
    return 2


@njit(cache=True)
def measure_1q_phase_free_tracked(tab, n, ws, q, code, anti, supp, ref_rows):
    """:func:`measure_1q_phase_free` that maintains the *reference rows* of a code.

    ``ref_rows`` (length ws) marks the stabilizer rows with support on a set R
    of reference qubits that no operation ever touches. Invariant: the R-parts
    of the marked rows are linearly independent. Then

    * the unmarked rows generate the stabilizer group of the system's state
      (the code's stabilizers), and
    * ``S(R) = popcount(ref_rows) - |R|`` (the code dimension k).

    The pivot is the first anticommuting row *outside* ``ref_rows`` if there is
    one: multiplying by it leaves every R-part unchanged. Otherwise every
    anticommuting row is marked; multiplying the others by the pivot keeps
    their R-parts independent (and nonzero), and the pivot row, replaced by
    the measured Pauli, leaves the set. Either way the invariant holds and
    the stabilizer group is the same as with :func:`measure_1q_phase_free`.
    With ``ref_rows`` all zero the two functions are identical.
    """
    _anticommuting_1q(tab, ws, q, code, anti)
    pivot = -1
    for w in range(ws):
        free = anti[w] & ~ref_rows[w]
        if free != ZERO:
            pivot = (w << 6) + np.int64(ctz(free))
            break
    if pivot < 0:
        for w in range(ws):
            if anti[w] != ZERO:
                pivot = (w << 6) + np.int64(ctz(anti[w]))
                break
        if pivot < 0:
            return 0
        ref_rows[pivot >> 6] &= ~(ONE << _U(pivot & 63))
    _collapse_phase_free(tab, n, ws, q, code, anti, supp, pivot)
    return 2


@njit(cache=True)
def _anticommuting_1q(tab, ws, q, code, anti):
    """``anti`` = stabilizer rows anticommuting with X/Z/Y (code 1/2/3) on ``q``."""
    xr = 2 * q
    for w in range(ws):
        a = ZERO
        if code & 2:
            a ^= tab[xr, w]
        if code & 1:
            a ^= tab[xr + 1, w]
        anti[w] = a


@njit(cache=True)
def _collapse_phase_free(tab, n, ws, q, code, anti, supp, pivot):
    """Multiply the rows in ``anti`` by row ``pivot`` and replace it by the measured Pauli."""
    xr = 2 * q
    pw = pivot >> 6
    pbit = ONE << _U(pivot & 63)
    nbit = ~pbit
    anti[pw] &= nbit
    ns = 0
    for c in range(n):
        if ((tab[2 * c, pw] | tab[2 * c + 1, pw]) & pbit) != ZERO:
            supp[ns] = c
            ns += 1
    for j in range(ns):
        r = 2 * supp[j]
        if (tab[r, pw] & pbit) != ZERO:
            for w in range(ws):
                tab[r, w] ^= anti[w]
            tab[r, pw] &= nbit
        if (tab[r + 1, pw] & pbit) != ZERO:
            for w in range(ws):
                tab[r + 1, w] ^= anti[w]
            tab[r + 1, pw] &= nbit
    if code & 1:
        tab[xr, pw] |= pbit
    if code & 2:
        tab[xr + 1, pw] |= pbit


@njit(cache=True)
def measure_pauli(tab, n, ws, phases, qs, codes, coin, anti, supp, lo, hi):
    """Measure ``prod_j codes[j] on qs[j]``; see :func:`collapse` for the return value."""
    anticommuting_rows(tab, qs, codes, anti)
    return collapse(tab, n, ws, phases, anti, qs, codes, coin, supp, lo, hi)


@njit(cache=True)
def measure_1q(tab, n, ws, phases, q, code, coin, anti, supp, lo, hi, qbuf, cbuf):
    """Measure a single-qubit Pauli (X=1, Z=2, Y=3) on ``q``."""
    if not phases:
        return measure_1q_phase_free(tab, n, ws, q, code, anti, supp)
    width = tab.shape[1]
    for w in range(width):
        a = ZERO
        if code & 2:
            a ^= tab[2 * q, w]
        if code & 1:
            a ^= tab[2 * q + 1, w]
        anti[w] = a
    qbuf[0] = q
    cbuf[0] = code
    return collapse(tab, n, ws, phases, anti, qbuf[:1], cbuf[:1], coin, supp, lo, hi)


@njit(cache=True)
def peek_pauli(tab, n, ws, qs, codes, anti, nz):
    """Expectation of a Pauli without disturbing the state (phases mode).

    Returns 0 if a measurement would be random, else 1 for eigenvalue +1 and
    -1 for eigenvalue -1.
    """
    anticommuting_rows(tab, qs, codes, anti)
    for w in range(ws):
        if anti[w] != ZERO:
            return 0
    e = product_exponent(tab, n, ws, anti[ws:], nz)
    return 1 - 2 * ((e >> 1) & 1)


@njit(cache=True)
def is_deterministic(tab, ws, qs, codes, anti):
    """True if measuring the Pauli would give a deterministic outcome (any mode)."""
    anticommuting_rows(tab, qs, codes, anti)
    for w in range(ws):
        if anti[w] != ZERO:
            return False
    return True


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------


@njit(cache=True)
def single_qubit_entropy(tab, q, ws):
    """Entanglement entropy (0 or 1) of qubit ``q`` in a pure state.

    The stabilizer block restricted to one qubit has two columns; the entropy
    is ``rank - 1``, and the rank is 2 exactly when the columns are nonzero and
    different.
    """
    x_nonzero = False
    z_nonzero = False
    differ = False
    for w in range(ws):
        x = tab[2 * q, w]
        z = tab[2 * q + 1, w]
        x_nonzero |= x != ZERO
        z_nonzero |= z != ZERO
        differ |= x != z
    return 1 if (x_nonzero and z_nonzero and differ) else 0


@njit(cache=True)
def columns_rank(tab, qubits, ws, scratch):
    """GF(2) rank of the stabilizer block restricted to ``qubits``.

    The 2|A| column bitsets (X and Z for each qubit, restricted to stabilizer
    words) are copied into ``scratch`` (shape >= (2|A|, ws)) and row-reduced.
    """
    for i in range(qubits.shape[0]):
        q = qubits[i]
        for w in range(ws):
            scratch[2 * i, w] = tab[2 * q, w]
            scratch[2 * i + 1, w] = tab[2 * q + 1, w]
    return _echelon_rank(scratch, 2 * qubits.shape[0], ws)


@njit(cache=True)
def columns_rank_without(tab, qubits, ws, skip_rows, scratch):
    """:func:`columns_rank` over the stabilizer rows *not* set in ``skip_rows``."""
    for i in range(qubits.shape[0]):
        q = qubits[i]
        for w in range(ws):
            keep = ~skip_rows[w]
            scratch[2 * i, w] = tab[2 * q, w] & keep
            scratch[2 * i + 1, w] = tab[2 * q + 1, w] & keep
    return _echelon_rank(scratch, 2 * qubits.shape[0], ws)


@njit(cache=True)
def _echelon_rank(scratch, m, ws):
    """Rank of the first ``m`` bitset rows of ``scratch`` (destroyed)."""
    rank = 0
    for i in range(m):
        lead_w = -1
        for w in range(ws):
            if scratch[i, w] != ZERO:
                lead_w = w
                break
        if lead_w < 0:
            continue
        rank += 1
        word = scratch[i, lead_w]
        lead = word & (~word + ONE)
        for j in range(i + 1, m):
            if (scratch[j, lead_w] & lead) != ZERO:
                for w in range(lead_w, ws):
                    scratch[j, w] ^= scratch[i, w]
    return rank


# ---------------------------------------------------------------------------
# row-major extraction and canonical form
# ---------------------------------------------------------------------------


@njit(cache=True)
def rows_of(tab, n, first_row, count):
    """Row-major unpacked bits of ``count`` generators starting at ``first_row``.

    Returns ``(xs, zs, signs)`` with ``xs[i, q]`` the x bit of generator
    ``first_row + i`` on qubit ``q`` (uint8 arrays).
    """
    xs = np.zeros((count, n), dtype=np.uint8)
    zs = np.zeros((count, n), dtype=np.uint8)
    signs = np.zeros(count, dtype=np.uint8)
    sign_row = tab.shape[0] - 1
    for i in range(count):
        r = first_row + i
        w = r >> 6
        b = _U(r & 63)
        for q in range(n):
            xs[i, q] = np.uint8((tab[2 * q, w] >> b) & ONE)
            zs[i, q] = np.uint8((tab[2 * q + 1, w] >> b) & ONE)
        signs[i] = np.uint8((tab[sign_row, w] >> b) & ONE)
    return xs, zs, signs


@njit(cache=True)
def _row_mul(xs, zs, signs, target, source, track_signs):
    """Row ``target`` <- row ``target`` * row ``source`` (rows commute)."""
    n = xs.shape[1]
    e = 0
    for q in range(n):
        x1 = xs[target, q]
        z1 = zs[target, q]
        x2 = xs[source, q]
        z2 = zs[source, q]
        if track_signs and (x1 | z1) and (x2 | z2):
            # cyclic X -> Y -> Z: left * next = +i, left * previous = -i
            l_code = x1 + 2 * z1  # X=1, Z=2, Y=3
            r_code = x2 + 2 * z2
            if l_code != r_code:
                order_l = 0 if l_code == 1 else (1 if l_code == 3 else 2)
                order_r = 0 if r_code == 1 else (1 if r_code == 3 else 2)
                e += 1 if (order_r - order_l) % 3 == 1 else -1
        xs[target, q] = x1 ^ x2
        zs[target, q] = z1 ^ z2
    if track_signs:
        total = e + 2 * (np.int64(signs[target]) + np.int64(signs[source]))
        signs[target] = np.uint8((total & 3) >> 1)


@njit(cache=True)
def canonicalize(xs, zs, signs, track_signs):
    """Stim's canonical stabilizer form, in place.

    Gaussian elimination qubit by qubit, X before Z: the first remaining
    generator with an X (then a Z) on the qubit becomes the pivot, is
    multiplied into every other generator having that Pauli component, and is
    moved to the next pivot position. This matches
    ``stim.TableauSimulator.canonical_stabilizers``.
    """
    count, n = xs.shape
    min_pivot = 0
    for q in range(n):
        for basis in range(2):
            col = xs if basis == 0 else zs
            pivot = -1
            for r in range(min_pivot, count):
                if col[r, q]:
                    pivot = r
                    break
            if pivot < 0:
                continue
            for r in range(count):
                if r != pivot and col[r, q]:
                    _row_mul(xs, zs, signs, r, pivot, track_signs)
            if pivot != min_pivot:
                for qq in range(n):
                    t = xs[pivot, qq]
                    xs[pivot, qq] = xs[min_pivot, qq]
                    xs[min_pivot, qq] = t
                    t = zs[pivot, qq]
                    zs[pivot, qq] = zs[min_pivot, qq]
                    zs[min_pivot, qq] = t
                t = signs[pivot]
                signs[pivot] = signs[min_pivot]
                signs[min_pivot] = t
            min_pivot += 1
            if min_pivot == count:
                return
