"""Interaction graphs as plain edge arrays.

Every function returns an ``(m, 2)`` int64 array of undirected edges ``(a, b)``
with ``a < b``, sorted, without duplicates or self-loops. That array is all a
monitored circuit needs, so graphs from anywhere (networkx, a file, your own
code) plug in through :func:`from_edges`.

Random graphs take an explicit ``seed`` and use SparseGF2's own PRNG
(xoshiro256**), so a given ``(parameters, seed)`` produces the same graph on
every platform and every version of numpy or networkx.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from sparsegf2_v3._kernel.rng import next_below, next_double, seed_state

__all__ = [
    "complete",
    "cycle",
    "degrees",
    "edge_coloring",
    "erdos_renyi",
    "from_edges",
    "from_networkx",
    "lattice_2d",
    "newman_watts",
    "path",
    "power_law_weights",
    "random_regular",
    "watts_strogatz",
]


def from_edges(edges: Iterable[tuple[int, int]], num_nodes: int | None = None) -> np.ndarray:
    """Normalise any edge list: orient ``a < b``, drop duplicates, sort; reject self-loops."""
    arr = np.asarray(list(edges), dtype=np.int64).reshape(-1, 2)
    if arr.size and arr.min() < 0:
        raise ValueError("node indices must be non-negative")
    if num_nodes is not None and arr.size and arr.max() >= num_nodes:
        raise ValueError(f"edge endpoint {int(arr.max())} out of range for {num_nodes} nodes")
    if np.any(arr[:, 0] == arr[:, 1]):
        raise ValueError("self-loops are not allowed")
    arr = np.sort(arr, axis=1)
    return np.unique(arr, axis=0) if arr.size else arr


def cycle(n: int) -> np.ndarray:
    """Ring of ``n`` nodes (n >= 3): edges (i, i+1 mod n)."""
    if n < 3:
        raise ValueError("a cycle needs at least 3 nodes")
    i = np.arange(n)
    return from_edges(np.stack([i, (i + 1) % n], axis=1))


def path(n: int) -> np.ndarray:
    """Open chain of ``n`` nodes: edges (i, i+1)."""
    if n < 2:
        raise ValueError("a path needs at least 2 nodes")
    i = np.arange(n - 1)
    return from_edges(np.stack([i, i + 1], axis=1))


def complete(n: int) -> np.ndarray:
    """All ``n (n - 1) / 2`` pairs (all-to-all)."""
    if n < 2:
        raise ValueError("a complete graph needs at least 2 nodes")
    a, b = np.triu_indices(n, k=1)
    return np.stack([a, b], axis=1).astype(np.int64)


def lattice_2d(rows: int, cols: int, *, periodic: bool = True) -> np.ndarray:
    """Square lattice; node ``r * cols + c``. ``periodic`` wraps both directions (a torus)."""
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")
    edges = []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols or (periodic and cols > 2):
                edges.append((v, r * cols + (c + 1) % cols))
            if r + 1 < rows or (periodic and rows > 2):
                edges.append((v, ((r + 1) % rows) * cols + c))
    return from_edges(edges)


def _ring_lattice(n: int, k: int) -> list[tuple[int, int]]:
    if k % 2 or k < 2 or k >= n:
        raise ValueError(f"k must be even with 2 <= k < n, got k={k}, n={n}")
    return [(i, (i + j) % n) for j in range(1, k // 2 + 1) for i in range(n)]


def watts_strogatz(n: int, k: int, beta: float, *, seed) -> np.ndarray:
    """Watts-Strogatz small world (Nature 393, 440): ring lattice, then rewiring.

    Start from a ring where each node links to its ``k`` nearest neighbours
    (``k / 2`` on each side, ``k`` even). Visiting edges ``(i, i + j)`` for
    ``j = 1 .. k/2`` and ``i = 0 .. n-1`` in that order, each is rewired with
    probability ``beta`` to ``(i, w)`` with ``w`` uniform among nodes that are
    not ``i`` and not already adjacent to ``i``. The edge count stays
    ``n k / 2``. Same convention as ``networkx.watts_strogatz_graph``.
    """
    if not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    rng = seed_state(seed)
    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in _ring_lattice(n, k):
        adj[a].add(b)
        adj[b].add(a)
    for a, b in _ring_lattice(n, k):
        if next_double(rng) < beta:
            if len(adj[a]) >= n - 1:
                continue  # node already linked to everyone
            while True:
                w = int(next_below(rng, n))
                if w != a and w not in adj[a]:
                    break
            adj[a].discard(b)
            adj[b].discard(a)
            adj[a].add(w)
            adj[w].add(a)
    return from_edges((a, b) for a in range(n) for b in adj[a] if a < b)


def newman_watts(n: int, k: int, p: float, *, seed) -> np.ndarray:
    """Newman-Watts small world (Phys. Lett. A 263, 341): ring lattice plus shortcuts.

    Start from the ``k``-nearest-neighbour ring; for each of its edges
    ``(i, i + j)`` (same order as :func:`watts_strogatz`), with probability
    ``p`` add a shortcut ``(i, w)`` with ``w`` uniform among nodes not ``i``
    and not already adjacent to ``i``. No edge is removed.
    """
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    rng = seed_state(seed)
    ring = _ring_lattice(n, k)
    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in ring:
        adj[a].add(b)
        adj[b].add(a)
    for a, _ in ring:
        if next_double(rng) < p and len(adj[a]) < n - 1:
            while True:
                w = int(next_below(rng, n))
                if w != a and w not in adj[a]:
                    break
            adj[a].add(w)
            adj[w].add(a)
    return from_edges((a, b) for a in range(n) for b in adj[a] if a < b)


def erdos_renyi(n: int, p: float, *, seed) -> np.ndarray:
    """G(n, p): each of the n(n-1)/2 pairs is an edge independently with probability ``p``."""
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    rng = seed_state(seed)
    pairs = complete(n)
    keep = np.array([next_double(rng) < p for _ in range(len(pairs))], dtype=bool)
    return pairs[keep]


def random_regular(n: int, d: int, *, seed, max_tries: int = 1000) -> np.ndarray:
    """A uniformly random simple ``d``-regular graph (pairing model with rejection).

    Needs ``n * d`` even and ``d < n``. Rejection is efficient for small ``d``.
    """
    if (n * d) % 2 or d >= n or d < 1:
        raise ValueError("need n * d even and 1 <= d < n")
    rng = seed_state(seed)
    stubs = np.repeat(np.arange(n), d)
    for _ in range(max_tries):
        perm = stubs.copy()
        for i in range(len(perm) - 1, 0, -1):  # Fisher-Yates
            j = int(next_below(rng, i + 1))
            perm[i], perm[j] = perm[j], perm[i]
        pairs = perm.reshape(-1, 2)
        if np.any(pairs[:, 0] == pairs[:, 1]):
            continue
        normalised = np.sort(pairs, axis=1)
        if len(np.unique(normalised, axis=0)) == len(normalised):
            return from_edges(normalised)
    raise RuntimeError(f"no simple {d}-regular graph found in {max_tries} tries")


def from_networkx(graph) -> np.ndarray:
    """Edges of a networkx graph whose nodes are the integers 0..n-1."""
    nodes = sorted(graph.nodes())
    if nodes != list(range(len(nodes))):
        raise ValueError(
            "networkx graph nodes must be exactly 0..n-1 (use nx.convert_node_labels_to_integers)"
        )
    return from_edges(graph.edges(), len(nodes))


def power_law_weights(
    edges: np.ndarray, n: int, alpha: float, *, periodic: bool = True
) -> np.ndarray:
    """Weights ``r**-alpha`` for each edge, with ``r`` the distance along a ring (or chain).

    Combine with ``graphs.complete(n)`` and ``schedule="random_edges"`` to get
    the long-range monitored circuits of Block et al. (PRL 128, 010604), where a
    gate acts on a pair at distance ``r`` with probability proportional to
    ``r**-alpha``. ``periodic`` uses the ring distance ``min(|i-j|, n-|i-j|)``.
    """
    edges = np.asarray(edges, dtype=np.int64)
    d = np.abs(edges[:, 1] - edges[:, 0]).astype(np.float64)
    if periodic:
        d = np.minimum(d, n - d)
    if np.any(d <= 0):
        raise ValueError("edges must join distinct nodes")
    return d ** (-float(alpha))


def degrees(edges: np.ndarray, n: int) -> np.ndarray:
    """Degree of every node."""
    return np.bincount(np.asarray(edges, dtype=np.int64).ravel(), minlength=n)


def edge_coloring(edges: np.ndarray, n: int) -> list[np.ndarray]:
    """Split the edges into matchings (no two edges in a matching share a node).

    Applying one matching per time step generalises the brickwork circuit to any
    graph: on a cycle with even ``n`` this gives the usual even/odd bonds. Uses
    greedy colouring in edge order, so at most ``2 * max_degree - 1`` matchings
    (often ``max_degree`` or ``max_degree + 1``). Returns a list of index arrays
    into ``edges``.
    """
    edges = np.asarray(edges, dtype=np.int64)
    colours_at = [set() for _ in range(n)]
    colour_of = np.zeros(len(edges), dtype=np.int64)
    for e, (a, b) in enumerate(edges):
        c = 0
        while c in colours_at[a] or c in colours_at[b]:
            c += 1
        colour_of[e] = c
        colours_at[a].add(c)
        colours_at[b].add(c)
    num = int(colour_of.max()) + 1 if len(edges) else 0
    return [np.flatnonzero(colour_of == c) for c in range(num)]
