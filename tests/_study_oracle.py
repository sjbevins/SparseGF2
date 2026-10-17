"""Independent small-system references for the study's observables (test-only).

Imports nothing from sparsegf2. Every quantity is computed from its definition by enumeration:
the whole stabilizer group of a joint state, every Pauli operator commuting with a code, every
subset, every permutation, every cyclic interval, every connected subset. Random joint states come
from Stim, an independent simulator.

Conventions:

* A Pauli operator on ``N`` qubits, up to its phase, is a Python int: bit ``q`` is ``x_q`` and bit
  ``N + q`` is ``z_q``.
* A joint state of the purification picture has ``N = 2n`` qubits: the system ``0..n-1``, then the
  reference ``n..2n-1``.
* A code on ``n`` qubits is given by stabilizer generators in the same int form with ``N = n``.
* A region is an int mask of system qubits (bit ``q`` = qubit ``q``).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable
from fractions import Fraction

import numpy as np
import stim

# ---------------------------------------------------------------------------------------------
# Pauli rows as ints
# ---------------------------------------------------------------------------------------------


def dense_to_ints(rows) -> list[int]:
    """The rows of a ``(s, 2N)`` 0/1 matrix (X bits then Z bits) as ints (bit j = column j)."""
    return [sum(1 << int(j) for j in np.flatnonzero(row)) for row in np.asarray(rows)]


def words_to_ints(words, num_qubits: int) -> list[int]:
    """Rows in the engine's packed layout as ints, read bit by bit.

    Row layout (``mipt.tableau_snapshot``): ``wq = ceil(N / 64)`` words of X bits (``x_q`` is bit
    ``q % 64`` of word ``q // 64``), then ``wq`` words of Z bits.
    """
    n = int(num_qubits)
    wq = (n + 63) // 64
    out = []
    for row in np.asarray(words, dtype=np.uint64):
        value = 0
        for q in range(n):
            if (int(row[q // 64]) >> (q % 64)) & 1:
                value |= 1 << q
            if (int(row[wq + q // 64]) >> (q % 64)) & 1:
                value |= 1 << (n + q)
        out.append(value)
    return out


def ints_to_dense(rows: Iterable[int], num_qubits: int) -> np.ndarray:
    rows = list(rows)
    out = np.zeros((len(rows), 2 * num_qubits), dtype=np.uint8)
    for i, r in enumerate(rows):
        for j in range(2 * num_qubits):
            out[i, j] = (r >> j) & 1
    return out


def symplectic(a: int, b: int, num_qubits: int) -> int:
    """``<a, b> = sum_q x_q(a) z_q(b) + z_q(a) x_q(b)`` mod 2."""
    mask = (1 << num_qubits) - 1
    ax, az = a & mask, a >> num_qubits
    bx, bz = b & mask, b >> num_qubits
    return (bin(ax & bz).count("1") + bin(az & bx).count("1")) & 1


def basis(rows: Iterable[int]) -> list[int]:
    """An independent basis of the span of ``rows`` (Gaussian elimination on ints)."""
    pivots: dict[int, int] = {}
    for r in rows:
        v = r
        while v:
            top = v.bit_length() - 1
            if top in pivots:
                v ^= pivots[top]
            else:
                pivots[top] = v
                break
    return list(pivots.values())


def group(generators: Iterable[int]) -> np.ndarray:
    """Every element of the span of ``generators`` (int64; ``2**rank`` of them)."""
    elements = np.zeros(1, dtype=np.int64)
    for g in basis(generators):
        elements = np.concatenate([elements, elements ^ np.int64(g)])
    return elements


def _parity(values: np.ndarray) -> np.ndarray:
    v = values.copy()
    for shift in (32, 16, 8, 4, 2, 1):
        v ^= v >> shift
    return v & 1


def supports(elements: np.ndarray, num_qubits: int) -> np.ndarray:
    """The support mask of each element: bit q set when the element acts on qubit q."""
    mask = (1 << num_qubits) - 1
    return (elements & mask) | (elements >> num_qubits)


def _log2(count: int) -> int:
    exponent = count.bit_length() - 1
    assert count == 1 << exponent, count
    return exponent


# ---------------------------------------------------------------------------------------------
# joint states: entropies and I(A:R) from the definition
# ---------------------------------------------------------------------------------------------


class Joint:
    """A pure joint state on ``2n`` qubits from its stabilizer generators, by enumerating its group.

    ``S(X) = |X| - log2 #{g in S : supp g within X}`` (the stabilizer entropy formula),
    ``I(A:R) = S(A) + S(R) - S(A u R)``.
    """

    def __init__(self, generators: Iterable[int], n: int):
        gens = list(generators)
        self.n = n
        self.N = 2 * n
        for a, b in itertools.combinations(gens, 2):
            assert symplectic(a, b, self.N) == 0, "generators must commute"
        self.elements = group(gens)
        assert self.elements.size == 1 << self.N, "the joint state must be pure"
        self.supp = supports(self.elements, self.N)
        self.reference = ((1 << n) - 1) << n

    def entropy(self, mask: int) -> int:
        inside = int(np.count_nonzero((self.supp & ~np.int64(mask)) == 0))
        return bin(mask).count("1") - _log2(inside)

    @property
    def k(self) -> int:
        return self.entropy(self.reference)

    def info(self, region: int) -> int:
        """I(A:R) for the system region ``region`` (a mask of system qubits)."""
        return self.entropy(region) + self.k - self.entropy(region | self.reference)

    def physical_generators(self) -> list[int]:
        """A basis of the elements with no reference support, as system-only ints (x bit q, z bit n + q)."""
        n, big = self.n, self.N
        low = (1 << n) - 1
        ref_free = self.elements[(self.supp & np.int64(self.reference)) == 0]
        rows = [(int(e) & low) | (((int(e) >> big) & low) << n) for e in ref_free]
        return basis(rows)


# ---------------------------------------------------------------------------------------------
# codes: I(A:R) = dim N_A - dim S_A by enumerating every Pauli
# ---------------------------------------------------------------------------------------------


class Code:
    """A stabilizer code on ``n`` qubits, by enumerating its stabilizer group and its normalizer.

    ``N_A`` (``S_A``): the normalizer (stabilizer) elements supported in ``A``;
    ``lost(A) = log2 |N_A| - log2 |S_A|``, the number of independent logical operators supported
    in ``A``, which is ``I(A:R)``. All ``4**n`` Paulis are enumerated: ``n <= 9``.
    """

    def __init__(self, generators: Iterable[int], n: int):
        assert n <= 9
        self.n = n
        self.gens = basis(generators)
        for a, b in itertools.combinations(self.gens, 2):
            assert symplectic(a, b, n) == 0, "generators must commute"
        self.k = n - len(self.gens)
        self.stabilizers = group(self.gens)
        everything = np.arange(1 << (2 * n), dtype=np.int64)
        commute = np.ones(everything.size, dtype=bool)
        low = (1 << n) - 1
        for g in self.gens:
            gx, gz = np.int64(g & low), np.int64(g >> n)
            commute &= _parity(((everything & low) & gz) ^ ((everything >> n) & gx)) == 0
        self.normalizer = everything[commute]
        self.s_supp = supports(self.stabilizers, n)
        self.n_supp = supports(self.normalizer, n)

    def lost(self, region: int) -> int:
        out = ~np.int64(region)
        normal = int(np.count_nonzero((self.n_supp & out) == 0))
        stab = int(np.count_nonzero((self.s_supp & out) == 0))
        return _log2(normal) - _log2(stab)


# ---------------------------------------------------------------------------------------------
# erasure statistics by enumeration
# ---------------------------------------------------------------------------------------------


def masks_of_size(n: int, m: int) -> list[int]:
    return [sum(1 << q for q in c) for c in itertools.combinations(range(n), m)]


def popcount(mask: int) -> int:
    return bin(mask).count("1")


def info_table(info: Callable[[int], int], n: int) -> dict[int, int]:
    """``{mask: I(A:R)}`` for every region of the n system qubits."""
    return {mask: info(mask) for mask in range(1 << n)}


def interval_masks(n: int, m: int) -> list[int]:
    return [sum(1 << ((a + j) % n) for j in range(m)) for a in range(n)]


def cyclic_distance(table: dict[int, int], n: int, k: int) -> float:
    """``d_cyc``: NaN for ``k = 0``, else the shortest failing cyclic interval (every interval tried)."""
    if k == 0:
        return math.nan
    for m in range(1, n + 1):
        if any(table[mask] > 0 for mask in interval_masks(n, m)):
            return float(m)
    raise AssertionError("no failing interval although k > 0")


def uniform_failure(table: dict[int, int], n: int) -> list[Fraction]:
    """``P_C(m) = #{failing m-subsets} / C(n, m)``, m = 0..n, exactly."""
    return [
        Fraction(sum(1 for mask in masks_of_size(n, m) if table[mask] > 0), math.comb(n, m))
        for m in range(n + 1)
    ]


def first_failure_distribution(table: dict[int, int], n: int) -> list[Fraction]:
    """``Pr(M_pi = m)`` for m = 0..n+1 over every permutation pi (``n + 1``: never fails)."""
    counts = [0] * (n + 2)
    for perm in itertools.permutations(range(n)):
        mask, first = 0, n + 1
        if table[0] > 0:
            first = 0
        else:
            for m, q in enumerate(perm, start=1):
                mask |= 1 << q
                if table[mask] > 0:
                    first = m
                    break
        counts[first] += 1
    total = math.factorial(n)
    return [Fraction(c, total) for c in counts]


def boundary(edges, mask: int) -> int:
    return sum(1 for a, b in edges if ((mask >> a) & 1) != ((mask >> b) & 1))


def correlated_law(edges, n: int, m: int, coupling: float) -> dict[int, float]:
    """``Q_{G,J}(A | m) = exp(-J b(A)) / Z_m(J)`` over every m-subset (shifted for stability)."""
    masks = masks_of_size(n, m)
    bounds = [boundary(edges, mask) for mask in masks]
    shift = min(bounds) if coupling >= 0 else max(bounds)
    weights = [math.exp(-coupling * (b - shift)) for b in bounds]
    total = sum(weights)
    return {mask: w / total for mask, w in zip(masks, weights, strict=True)}


def correlated_failure(table: dict[int, int], edges, n: int, coupling: float) -> list[float]:
    """``P_{C,G,J}(m) = sum_A Q(A | m) 1[I(A:R) > 0]``, m = 0..n."""
    return [
        sum(q for mask, q in correlated_law(edges, n, m, coupling).items() if table[mask] > 0)
        for m in range(n + 1)
    ]


def mean_leakage(table: dict[int, int], edges, n: int, m: int, coupling: float) -> float:
    """``l(m, J) = sum_A Q(A | m) I(A:R)``."""
    return sum(q * table[mask] for mask, q in correlated_law(edges, n, m, coupling).items())


def robustness(curve, n: int, k: int):
    """``D = 0`` for ``k = 0``, else ``sum_{m=0}^{n-1} (1 - P(m))``."""
    if k == 0:
        return 0
    return sum(1 - p for p in curve[:n])


def is_connected(mask: int, edges) -> bool:
    """Whether the subgraph induced on the nodes of ``mask`` is connected (breadth-first search)."""
    nodes = [q for q in range(mask.bit_length()) if (mask >> q) & 1]
    if not nodes:
        return False
    adjacent: dict[int, set[int]] = {q: set() for q in nodes}
    for a, b in edges:
        if a in adjacent and b in adjacent:
            adjacent[a].add(b)
            adjacent[b].add(a)
    seen = {nodes[0]}
    frontier = [nodes[0]]
    while frontier:
        q = frontier.pop()
        for r in adjacent[q] - seen:
            seen.add(r)
            frontier.append(r)
    return len(seen) == len(nodes)


def connected_distance(table: dict[int, int], edges, n: int, k: int) -> float:
    """``d_conn``: 0 for ``k = 0``, the smallest connected failing region, inf when there is none."""
    if k == 0:
        return 0.0
    best = math.inf
    for mask in range(1, 1 << n):
        if table[mask] > 0 and popcount(mask) < best and is_connected(mask, edges):
            best = popcount(mask)
    return float(best)


# ---------------------------------------------------------------------------------------------
# random joint states (Stim)
# ---------------------------------------------------------------------------------------------


def _scramble(sim: stim.TableauSimulator, qubits, rng, depth: int) -> None:
    qubits = list(qubits)
    for _ in range(depth):
        if len(qubits) > 1:
            a, b = (int(q) for q in rng.choice(qubits, size=2, replace=False))
            sim.cx(a, b)
        for q in qubits:
            gate = int(rng.integers(4))
            if gate == 0:
                sim.h(q)
            elif gate == 1:
                sim.s(q)


def random_joint_state(n: int, rng, measured) -> list[int]:
    """n Bell pairs (system q, reference n + q), a random circuit on the system with Z measurements
    of ``measured`` in the middle; the stabilizer generators over 2n qubits, signs dropped."""
    sim = stim.TableauSimulator(seed=int(rng.integers(2**31)))
    sim.set_num_qubits(2 * n)
    for q in range(n):
        sim.h(q)
        sim.cx(q, n + q)
    _scramble(sim, range(n), rng, 3 * n + 2)
    for q in measured:
        sim.measure(int(q))
    _scramble(sim, range(n), rng, 3 * n + 2)
    out = []
    for p in sim.canonical_stabilizers():
        xs, zs = p.to_numpy()
        x = sum(1 << int(q) for q in np.flatnonzero(xs))
        z = sum(1 << (2 * n + int(q)) for q in np.flatnonzero(zs))
        out.append(x | z)
    return out
