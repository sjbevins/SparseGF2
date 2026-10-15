"""Brute-force GF(2) oracle for erasure questions about stabilizer codes (test-only).

It imports nothing from sparsegf2. Everything is plain Python on big integers: a Pauli
on n qubits is one integer of 2n bits, X part in bits 0..n-1 and Z part in bits n..2n-1.
A generator may be given as a pair ``(x_bits, z_bits)`` of Python integers, a pair of 0/1
numpy arrays, or one 0/1 numpy array of length 2n (X bits, then Z bits).

Definitions and identities used (S = the stabilizer group, r = its dimension, N = N(S)
the normalizer = symplectic complement of S, dim N = 2n - r, A = a set of qubits,
Abar its complement, M|_B the rows of M with every coordinate outside B cleared):

* dim of the elements of a group G supported in A is dim G - rank(G|_Abar). The map
  "clear A" has the elements supported in Abar as its kernel, and the rank-nullity
  theorem gives the dimension of the kernel.
* The symplectic complement of S|_B (inside the 2|B| coordinates of B) is the set of
  elements of N supported in B, so
  dim S_B = 2|B| - rank(N|_B) and dim N_B = 2|B| - rank(S|_B)
  (S_B, N_B: the elements of S, N supported in B). Both forms are implemented; the
  first is used when A is large, the second when it is small, and the tests check that
  they agree with an enumeration of all 2^r stabilizer elements.
* Cleaning lemma (Bravyi and Terhal 2009; Gottesman 1997 for the erasure form): the
  logical information destroyed by erasing A is l(A) = dim N_A - dim S_A, in bits. For
  the maximally mixed code state purified by the reference R this is I(A:R) = S(A) +
  S(R) - S(AR), and the erasure is correctable if and only if l(A) = 0.
* For a pure stabilizer state given by n independent generators, S(A) = rank(g|_A) - |A|
  (in bits).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence

import numpy as np


def rank(rows: Iterable[int]) -> int:
    """GF(2) rank of bit-rows given as Python integers."""
    basis: dict[int, int] = {}  # highest set bit -> row with that leading bit
    count = 0
    for v in rows:
        while v:
            top = v.bit_length() - 1
            if top in basis:
                v ^= basis[top]
            else:
                basis[top] = v
                count += 1
                break
    return count


def _to_int(gen, n: int) -> int:
    if isinstance(gen, tuple) and len(gen) == 2:
        x, z = gen
        if isinstance(x, (int, np.integer)):
            return int(x) | (int(z) << n)
        return _to_int(np.concatenate([np.asarray(x), np.asarray(z)]), n)
    bits = np.asarray(gen, dtype=np.uint8).ravel() & 1
    if bits.size != 2 * n:
        raise ValueError(f"expected 2n = {2 * n} bits, got {bits.size}")
    v = 0
    for j in np.flatnonzero(bits):
        v |= 1 << int(j)
    return v


def _rows(gens: Sequence, n: int) -> list[int]:
    return [_to_int(g, n) for g in gens]


def _mask(qubits: Iterable[int], n: int) -> int:
    m = 0
    for q in qubits:
        q = int(q)
        if not 0 <= q < n:
            raise ValueError(f"qubit {q} outside [0, {n})")
        m |= (1 << q) | (1 << (q + n))
    return m


def _swap_halves(v: int, n: int) -> int:
    low = (1 << n) - 1
    return (v >> n) | ((v & low) << n)


def _nullspace(rows: list[int], width: int) -> list[int]:
    """Basis of {v : <row, v> = 0 for every row} (dot product over GF(2)) by Gauss-Jordan."""
    pivots: dict[int, int] = {}  # pivot column -> fully reduced row
    for v in rows:
        for p, b in pivots.items():
            if v >> p & 1:
                v ^= b
        if not v:
            continue
        p = (v & -v).bit_length() - 1
        for q in list(pivots):
            if pivots[q] >> p & 1:
                pivots[q] ^= v
        pivots[p] = v
    out = []
    for f in range(width):
        if f in pivots:
            continue
        vec = 1 << f
        for p, b in pivots.items():
            if b >> f & 1:
                vec |= 1 << p
        out.append(vec)
    return out


def normalizer(gens: Sequence, n: int) -> list[int]:
    """Basis of N(S): all Paulis commuting with every generator (2n - r rows)."""
    swapped = [_swap_halves(g, n) for g in _rows(gens, n)]
    return _nullspace(swapped, 2 * n)


class Code:
    """Generators, their normalizer and cached ranks (private helper)."""

    def __init__(self, gens: Sequence, n: int):
        self.n = n
        rows = _rows(gens, n)
        self.s_rows = rows
        self.r = rank(rows)
        self.n_rows = normalizer(gens, n)
        self.dim_n = 2 * n - self.r
        if rank(self.n_rows) != self.dim_n:  # pragma: no cover (guards the oracle itself)
            raise AssertionError("normalizer has the wrong dimension")

    def _supported(self, group: list[int], dim: int, dual: list[int], region: list[int]) -> int:
        n = self.n
        a = len(region)
        rmask = _mask(region, n)
        cmask = _mask(range(n), n) & ~rmask
        if a <= n // 2:
            # elements supported in A: the complement (within A's 2|A| coordinates) of dual|_A
            return 2 * a - rank(v & rmask for v in dual)
        return dim - rank(v & cmask for v in group)

    def stabilizer_dim_on(self, region: list[int]) -> int:
        return self._supported(self.s_rows, self.r, self.n_rows, region)

    def normalizer_dim_on(self, region: list[int]) -> int:
        return self._supported(self.n_rows, self.dim_n, self.s_rows, region)

    def lost(self, region: list[int]) -> int:
        return self.normalizer_dim_on(region) - self.stabilizer_dim_on(region)


def _region(region: Iterable[int], n: int) -> list[int]:
    r = [int(q) for q in region]
    if len(set(r)) != len(r) or any(not 0 <= q < n for q in r):
        raise ValueError("region must be distinct qubits in [0, n)")
    return r


def stabilizer_dim_on(gens: Sequence, n: int, region: Iterable[int]) -> int:
    """Dimension of the stabilizer group restricted to elements supported in ``region``."""
    return Code(gens, n).stabilizer_dim_on(_region(region, n))


def normalizer_dim_on(gens: Sequence, n: int, region: Iterable[int]) -> int:
    """Dimension of the elements of N(S) supported in ``region``."""
    return Code(gens, n).normalizer_dim_on(_region(region, n))


def lost_information(gens: Sequence, n: int, region: Iterable[int]) -> int:
    """l(A) = dim N_A - dim S_A (cleaning lemma), equal to I(A:R) in bits."""
    return Code(gens, n).lost(_region(region, n))


def uncorrectable_counts(gens: Sequence, n: int, w: int) -> list[int]:
    """For m = 1..w, the number of m-subsets A of the n qubits with l(A) > 0."""
    code = Code(gens, n)
    if code.dim_n == code.r:  # k = 0: nothing to lose
        return [0] * w
    s_rows, n_rows = code.s_rows, code.n_rows
    out = []
    for m in range(1, w + 1):
        bad = 0
        for a in itertools.combinations(range(n), m):
            mask = _mask(a, n)
            # l(A) = rank(N|_A) - rank(S|_A), the small-region form of the identities above
            if rank(v & mask for v in n_rows) > rank(v & mask for v in s_rows):
                bad += 1
        out.append(bad)
    return out


def entropy(gens: Sequence, n: int, region: Iterable[int]) -> int:
    """Entropy in bits of the region for a pure stabilizer state (n independent generators)."""
    rows = _rows(gens, n)
    if rank(rows) != n:
        raise ValueError("a pure state needs n independent generators")
    reg = _region(region, n)
    mask = _mask(reg, n)
    return rank(v & mask for v in rows) - len(reg)


def all_stabilizer_elements(gens: Sequence, n: int) -> set[int]:
    """Every element of S as an integer (2^r of them): for tiny codes only."""
    basis: list[int] = []
    for v in _rows(gens, n):
        if rank(basis + [v]) > len(basis):
            basis.append(v)
    out = {0}
    for b in basis:
        out |= {v ^ b for v in out}
    return out
