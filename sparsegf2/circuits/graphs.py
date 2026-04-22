"""
Graph topologies for graph-defined random Clifford circuits.

MVP scope: ``cycle`` (nearest-neighbor) and ``complete`` (all-to-all), both
requiring even ``n`` when used in matching gating mode (otherwise the graph has
no perfect matching). Stochastic graph families are deferred.

A :class:`GraphTopology` carries:

- the edge list,
- a canonical 1-factorization (list of perfect matchings) when one exists,
- metadata for the pre-flight compatibility validator,
- a graph6 string for metadata storage.

All per-layer sampling (fresh random matchings, random palette draws) lives in
:mod:`sparsegf2.circuits.matching`; this module is purely about graph structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


Edge = Tuple[int, int]
Matching = List[Edge]


# ══════════════════════════════════════════════════════════════
# graph6 encoding (Brendan McKay's standard format)
# Reference: https://users.cecs.anu.edu.au/~bdm/data/formats.txt
# ══════════════════════════════════════════════════════════════

def _graph6_N(n: int) -> str:
    """Encode the vertex count as the graph6 N(n) prefix."""
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}")
    if n <= 62:
        return chr(n + 63)
    if n <= 258047:
        return chr(126) + "".join(chr(((n >> s) & 0x3F) + 63) for s in (12, 6, 0))
    return chr(126) + chr(126) + "".join(
        chr(((n >> s) & 0x3F) + 63) for s in (30, 24, 18, 12, 6, 0)
    )


def graph6_encode(n: int, edges) -> str:
    """Encode an undirected graph on ``n`` vertices as a graph6 string.

    Parameters
    ----------
    n : int
        Number of vertices (nodes are ``0..n-1``).
    edges : iterable of (int, int)
        Each edge listed once; order within a pair does not matter.

    Returns
    -------
    str
        The graph6 ASCII representation (no trailing newline).
    """
    edge_set = set()
    for u, v in edges:
        if u == v:
            raise ValueError(f"graph6 does not permit self-loops; got ({u},{v})")
        a, b = (u, v) if u < v else (v, u)
        edge_set.add((a, b))

    bits = []
    for col in range(1, n):
        for row in range(col):
            bits.append(1 if (row, col) in edge_set else 0)

    while len(bits) % 6:
        bits.append(0)

    chars = []
    for i in range(0, len(bits), 6):
        v = 0
        for j in range(6):
            v = (v << 1) | bits[i + j]
        chars.append(chr(v + 63))

    return _graph6_N(n) + "".join(chars)


# ══════════════════════════════════════════════════════════════
# GraphTopology
# ══════════════════════════════════════════════════════════════

@dataclass
class GraphTopology:
    """A graph with optional 1-factorization, used for circuit construction.

    Attributes
    ----------
    name : str
        Human-readable name, e.g. ``"cycle(32)"``, ``"complete(64)"``.
    n : int
        Number of qubits (= vertices).
    edges : list of (int, int)
        Edge list, each edge listed once with ``u < v``.
    is_stochastic : bool
        Whether the graph realization depends on a per-sample seed.
        Deterministic for MVP graphs.
    one_factorization : list of list of (int, int) or None
        A canonical 1-factorization (list of perfect matchings) if one exists.
        ``None`` when the graph does not admit a 1-factorization at this ``n``.
    graph6 : str
        graph6-encoded representation for metadata storage.
    fresh_matching_sampler : callable(rng) -> list of (int, int)
        A function taking a numpy RNG and returning a uniformly-random perfect
        matching. ``None`` when the graph has no perfect matching.
    """

    name: str
    n: int
    edges: List[Edge]
    is_stochastic: bool
    one_factorization: Optional[List[Matching]]
    graph6: str
    fresh_matching_sampler: Optional[Callable[[np.random.Generator], Matching]] = \
        field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def degree_max(self) -> int:
        """Maximum vertex degree."""
        if not self.edges:
            return 0
        d = np.zeros(self.n, dtype=int)
        for u, v in self.edges:
            d[u] += 1
            d[v] += 1
        return int(d.max())

    @property
    def has_perfect_matching(self) -> bool:
        """True if the graph has at least one perfect matching."""
        return self.fresh_matching_sampler is not None

    @property
    def has_one_factorization(self) -> bool:
        """True if the graph decomposes into perfect matchings."""
        return self.one_factorization is not None

    @property
    def chi_prime(self) -> Optional[int]:
        """Number of matchings in the 1-factorization, or ``None`` if none exists."""
        if self.one_factorization is None:
            return None
        return len(self.one_factorization)


# ══════════════════════════════════════════════════════════════
# Cycle graph
# ══════════════════════════════════════════════════════════════

def _cycle_edges(n: int) -> List[Edge]:
    """Edges of the n-cycle: (i, i+1 mod n) for i in 0..n-1, canonicalized."""
    edges = []
    for i in range(n):
        u, v = i, (i + 1) % n
        edges.append((min(u, v), max(u, v)))
    return sorted(set(edges))


def _cycle_one_factorization(n: int) -> Optional[List[Matching]]:
    """Two alternating perfect matchings of the even-n cycle."""
    if n % 2 != 0:
        return None
    even = [(2 * j, 2 * j + 1) for j in range(n // 2)]
    odd = [(2 * j + 1, (2 * j + 2) % n) for j in range(n // 2)]
    odd_canon = [(min(a, b), max(a, b)) for a, b in odd]
    return [sorted(even), sorted(odd_canon)]


def _cycle_fresh_sampler(n: int):
    """Sampler returning one of the 2 perfect matchings of the even-n cycle at random."""
    if n % 2 != 0:
        return None
    matchings = _cycle_one_factorization(n)

    def sample(rng: np.random.Generator) -> Matching:
        return [tuple(e) for e in matchings[int(rng.integers(0, 2))]]

    return sample


def cycle_graph(n: int) -> GraphTopology:
    """Construct the cycle (nearest-neighbor) graph on ``n`` qubits."""
    if n < 3:
        raise ValueError(f"cycle graph requires n >= 3; got n={n}")
    edges = _cycle_edges(n)
    one_fac = _cycle_one_factorization(n)
    sampler = _cycle_fresh_sampler(n)
    return GraphTopology(
        name=f"cycle({n})",
        n=n,
        edges=edges,
        is_stochastic=False,
        one_factorization=one_fac,
        graph6=graph6_encode(n, edges),
        fresh_matching_sampler=sampler,
    )


# ══════════════════════════════════════════════════════════════
# Complete graph
# ══════════════════════════════════════════════════════════════

def _complete_edges(n: int) -> List[Edge]:
    """All n(n-1)/2 edges of K_n."""
    return [(u, v) for u in range(n) for v in range(u + 1, n)]


def _complete_one_factorization(n: int) -> Optional[List[Matching]]:
    """Standard round-robin 1-factorization of K_n for even n.

    The algorithm fixes vertex ``n-1`` and rotates the remaining ``n-1`` vertices
    through ``n-1`` rounds. In round ``r``:

      - pair ``n-1`` with ``r``,
      - for ``i in 1..n/2-1`` pair ``(r+i) mod (n-1)`` with ``(r-i) mod (n-1)``.

    This produces ``n-1`` perfect matchings whose union is exactly ``E(K_n)``.
    """
    if n % 2 != 0:
        return None
    matchings: List[Matching] = []
    m = n - 1  # ring size
    for r in range(m):
        pairs = [(min(n - 1, r), max(n - 1, r))]
        for i in range(1, n // 2):
            a = (r + i) % m
            b = (r - i) % m
            pairs.append((min(a, b), max(a, b)))
        matchings.append(sorted(pairs))
    return matchings


def _complete_fresh_sampler(n: int):
    """Uniformly-random perfect matching of K_n via random permutation pairing."""
    if n % 2 != 0:
        return None

    def sample(rng: np.random.Generator) -> Matching:
        perm = rng.permutation(n)
        pairs = []
        for j in range(n // 2):
            a, b = int(perm[2 * j]), int(perm[2 * j + 1])
            pairs.append((min(a, b), max(a, b)))
        return sorted(pairs)

    return sample


def complete_graph(n: int) -> GraphTopology:
    """Construct the complete (all-to-all) graph on ``n`` qubits."""
    if n < 2:
        raise ValueError(f"complete graph requires n >= 2; got n={n}")
    edges = _complete_edges(n)
    one_fac = _complete_one_factorization(n)
    sampler = _complete_fresh_sampler(n)
    return GraphTopology(
        name=f"complete({n})",
        n=n,
        edges=edges,
        is_stochastic=False,
        one_factorization=one_fac,
        graph6=graph6_encode(n, edges),
        fresh_matching_sampler=sampler,
    )


# ══════════════════════════════════════════════════════════════
# Parser
# ══════════════════════════════════════════════════════════════

_GRAPH_CONSTRUCTORS = {
    "cycle": cycle_graph,
    "complete": complete_graph,
}


def parse_graph_spec(spec: str, n: int, seed: int = 0) -> GraphTopology:
    """Build a graph topology from a string specification.

    MVP supports ``"cycle"`` and ``"complete"``. Stochastic specs are deferred.

    Parameters
    ----------
    spec : str
        One of ``"cycle"``, ``"complete"``.
    n : int
        Number of qubits.
    seed : int
        Reserved for stochastic specs; ignored for MVP.

    Returns
    -------
    GraphTopology
    """
    spec = spec.strip().lower()
    if spec not in _GRAPH_CONSTRUCTORS:
        supported = ", ".join(sorted(_GRAPH_CONSTRUCTORS))
        raise ValueError(
            f"Unknown graph_spec {spec!r}. Supported in MVP: {supported}."
        )
    return _GRAPH_CONSTRUCTORS[spec](n)


__all__ = [
    "Edge",
    "Matching",
    "GraphTopology",
    "cycle_graph",
    "complete_graph",
    "parse_graph_spec",
    "graph6_encode",
]
