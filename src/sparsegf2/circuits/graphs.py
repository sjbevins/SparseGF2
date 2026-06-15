"""Graph topologies for graph-defined random Clifford circuits.

A circuit is "graph-defined" when the two-qubit gates of each layer are
placed on the **edges** of a fixed graph on the ``n`` qubits. The graph
decides *who can interact with whom*; the scheduler
(:mod:`sparsegf2.circuits.scheduler`) decides *which* edges fire each
layer (a perfect matching for brickwork, a single random edge for
``random_edge``).

A :class:`GraphTopology` carries:

- the edge list (each edge once, ``u < v``);
- a canonical **1-factorization** (a partition of the edges into perfect
  matchings) when one exists. Brickwork gating cycles through this;
- a **fresh-matching sampler** drawing a uniformly-random perfect matching;
- a graph6 string for compact metadata storage.

Topology vs. *simulatability* (the "construct circuits to be efficiently
simulatable" goal): the intuition is that locality should be cheaper.
SparseGF2 stores the tableau sparsely and a two-qubit gate only touches
generators supported on its two qubits, so a nearest-neighbour **cycle**
keeps supports short while a **complete** graph fans them out. But
``benchmarks/circuits/benchmark_topology_sparsity.py`` measured this and
found the effect is **negligible at the depths we run** (``O(n)``): by
``~4n`` layers even the cycle's light-cone has scrambled the whole system
many times, so the stabilizer tableau *saturates* to near-maximal weight
either way (n=128, p=0: cycle avg-weight 96.2 vs complete 96.2; runtimes
within noise). Locality would only buy speed at **shallow** depth, before
the light-cone spreads. What topology *does* change is the **physics**:
under measurement, ``complete`` sustains much more entanglement than
``cycle`` (a real MIPT-threshold effect), which is why both are offered,
not because one is cheaper to simulate. Don't pick a graph for speed at
``O(n)`` depth; pick it for the dynamics you want.

This module is pure graph structure, with **no** simulator import. All
per-layer sampling lives in :mod:`sparsegf2.circuits.matching`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# sparsegf2.errors is pure exception classes, so importing it does NOT couple
# this module to the simulator (no SparseGF2 / kernel import).
from sparsegf2.errors import InvalidArgumentError

Edge = tuple[int, int]
Matching = list[Edge]


# ----------------------------------------------------------------------
# graph6 encoding (Brendan McKay's standard format)
# Reference: https://users.cecs.anu.edu.au/~bdm/data/formats.txt
# ----------------------------------------------------------------------


def _graph6_N(n: int) -> str:
    """Encode the vertex count as the graph6 ``N(n)`` prefix."""
    if n < 0:
        raise InvalidArgumentError(f"n must be non-negative; got {n}")
    if n <= 62:
        return chr(n + 63)
    if n <= 258047:
        return chr(126) + "".join(chr(((n >> s) & 0x3F) + 63) for s in (12, 6, 0))
    return (
        chr(126) + chr(126) + "".join(chr(((n >> s) & 0x3F) + 63) for s in (30, 24, 18, 12, 6, 0))
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
        The graph6 ASCII representation (no trailing newline). The bit
        order is graph6's column-major upper triangle: for column
        ``col = 1..n-1`` and row ``0..col-1``, emit 1 iff ``(row, col)``
        is an edge, then pack 6 bits per character (offset by 63).
    """
    edge_set = set()
    for u, v in edges:
        if u == v:
            raise InvalidArgumentError(f"graph6 does not permit self-loops; got ({u},{v})")
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


# ----------------------------------------------------------------------
# GraphTopology
# ----------------------------------------------------------------------


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
        Deterministic (``False``) for both built-in graphs.
    one_factorization : list of list of (int, int) or None
        The canonical set of brickwork layers: a partition of the edges into
        matchings (a proper edge colouring) that ``round_robin`` / ``palette``
        cycle through. These are **perfect** matchings (a true 1-factorization)
        when the graph is regular and Class-1 (cycle, complete, regular graphs);
        for general geometries (path, lattice, arbitrary networkx graphs) they
        are colour classes that may leave some qubits idle on a layer. ``None``
        only for the empty edge set.
    graph6 : str
        graph6-encoded representation for metadata storage.
    fresh_matching_sampler : callable(rng) -> list of (int, int) or None
        Draws a uniformly-random perfect matching; ``None`` when the graph
        has no perfect matching. Kept out of ``repr`` (it's a closure).
    """

    name: str
    n: int
    edges: list[Edge]
    is_stochastic: bool
    one_factorization: list[Matching] | None
    graph6: str
    fresh_matching_sampler: Callable[[np.random.Generator], Matching] | None = field(
        default=None, repr=False
    )

    # ---- Derived properties ----

    @property
    def degree_max(self) -> int:
        """Maximum vertex degree (0 for the empty graph)."""
        if not self.edges:
            return 0
        d = np.zeros(self.n, dtype=int)
        for u, v in self.edges:
            d[u] += 1
            d[v] += 1
        return int(d.max())

    @property
    def has_perfect_matching(self) -> bool:
        """True iff the graph has at least one perfect matching."""
        return self.fresh_matching_sampler is not None

    @property
    def has_one_factorization(self) -> bool:
        """True iff the graph decomposes into perfect matchings."""
        return self.one_factorization is not None

    @property
    def chi_prime(self) -> int | None:
        """Number of matchings in the 1-factorization, or ``None``."""
        if self.one_factorization is None:
            return None
        return len(self.one_factorization)


# ----------------------------------------------------------------------
# Cycle graph (nearest-neighbour)
# ----------------------------------------------------------------------


def _cycle_edges(n: int) -> list[Edge]:
    """Edges of the n-cycle: ``(i, (i+1) mod n)``, canonicalized + sorted."""
    edges = []
    for i in range(n):
        u, v = i, (i + 1) % n
        edges.append((min(u, v), max(u, v)))
    return sorted(set(edges))


def _cycle_one_factorization(n: int) -> list[Matching] | None:
    """The two alternating perfect matchings of the even-n cycle.

    ``even`` pairs ``(0,1),(2,3),...``; ``odd`` pairs ``(1,2),(3,4),...,(n-1,0)``.
    Together they cover every cycle edge exactly once, so they form a valid
    1-factorization (and chi' = 2). Odd ``n`` has no perfect matching.
    """
    if n % 2 != 0:
        return None
    even = [(2 * j, 2 * j + 1) for j in range(n // 2)]
    odd = [(2 * j + 1, (2 * j + 2) % n) for j in range(n // 2)]
    odd_canon = [(min(a, b), max(a, b)) for a, b in odd]
    return [sorted(even), sorted(odd_canon)]


def _cycle_fresh_sampler(n: int):
    """Sampler returning one of the cycle's 2 perfect matchings at random."""
    if n % 2 != 0:
        return None
    matchings = _cycle_one_factorization(n)

    def sample(rng: np.random.Generator) -> Matching:
        return [tuple(e) for e in matchings[int(rng.integers(0, 2))]]

    return sample


def cycle_graph(n: int) -> GraphTopology:
    """Construct the cycle (nearest-neighbour ring) graph on ``n`` qubits."""
    if n < 3:
        raise InvalidArgumentError(f"cycle graph requires n >= 3; got n={n}")
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


# ----------------------------------------------------------------------
# Complete graph (all-to-all)
# ----------------------------------------------------------------------


def _complete_edges(n: int) -> list[Edge]:
    """All ``n(n-1)/2`` edges of ``K_n``, already ``u < v`` and sorted."""
    return [(u, v) for u in range(n) for v in range(u + 1, n)]


def _complete_one_factorization(n: int) -> list[Matching] | None:
    """Round-robin 1-factorization of ``K_n`` for even ``n``.

    The standard tournament-scheduling construction (Anderson 2001): fix
    vertex ``n-1``, rotate the remaining ``n-1`` through ``n-1`` rounds.
    In round ``r``: pair ``n-1`` with ``r``, then for ``i = 1..n/2-1``
    pair ``(r+i) mod (n-1)`` with ``(r-i) mod (n-1)``. The ``n-1`` rounds
    are perfect matchings whose union is exactly ``E(K_n)``.
    """
    if n % 2 != 0:
        return None
    matchings: list[Matching] = []
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
    """Uniformly-random perfect matching of ``K_n`` via a random pairing."""
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
        raise InvalidArgumentError(f"complete graph requires n >= 2; got n={n}")
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


# ----------------------------------------------------------------------
# Path (open chain): the textbook 1D brickwork geometry
# ----------------------------------------------------------------------


def path_graph(n: int) -> GraphTopology:
    """Construct the path (open chain) on ``n`` qubits: edges ``(i, i+1)``.

    Brickwork on a path is the standard 1D circuit: the two bond classes
    (even bonds ``(0,1),(2,3),…`` and odd bonds ``(1,2),(3,4),…``) alternate,
    with the boundary qubits idling on alternating layers. The even class is the
    unique perfect matching when ``n`` is even (so ``fresh`` returns it).
    """
    if n < 2:
        raise InvalidArgumentError(f"path graph requires n >= 2; got n={n}")
    edges = [(i, i + 1) for i in range(n - 1)]
    even = sorted((i, i + 1) for i in range(0, n - 1, 2))
    odd = sorted((i, i + 1) for i in range(1, n - 1, 2))
    one_fac = [c for c in (even, odd) if c] or None
    sampler = None
    if n % 2 == 0:  # even bonds cover every vertex -> the unique perfect matching
        pm = [tuple(e) for e in even]

        def sampler(rng: np.random.Generator, _pm=pm) -> Matching:
            return list(_pm)

    return GraphTopology(
        name=f"path({n})",
        n=n,
        edges=sorted(edges),
        is_stochastic=False,
        one_factorization=one_fac,
        graph6=graph6_encode(n, edges),
        fresh_matching_sampler=sampler,
    )


# ----------------------------------------------------------------------
# 2D lattice (open grid)
# ----------------------------------------------------------------------


def lattice_2d(rows: int, cols: int) -> GraphTopology:
    """Construct the open ``rows × cols`` grid; qubit ``(r, c) -> r*cols + c``.

    Edges are horizontal and vertical nearest neighbours. The grid is bipartite,
    so its edges split into four matchings by direction and coordinate parity
    (``H``/``V`` × even/odd), a proper edge colouring that brickwork cycles
    through. ``fresh`` returns a perfect matching when one exists (some axis
    has even length).
    """
    if rows < 1 or cols < 1 or rows * cols < 2:
        raise InvalidArgumentError(
            f"lattice_2d requires rows,cols >= 1 and >=2 nodes; got {rows}x{cols}"
        )
    n = rows * cols

    def idx(r: int, c: int) -> int:
        return r * cols + c

    h = [(idx(r, c), idx(r, c + 1)) for r in range(rows) for c in range(cols - 1)]
    v = [(idx(r, c), idx(r + 1, c)) for r in range(rows - 1) for c in range(cols)]
    edges = sorted(h + v)
    h_even = sorted((idx(r, c), idx(r, c + 1)) for r in range(rows) for c in range(0, cols - 1, 2))
    h_odd = sorted((idx(r, c), idx(r, c + 1)) for r in range(rows) for c in range(1, cols - 1, 2))
    v_even = sorted((idx(r, c), idx(r + 1, c)) for r in range(0, rows - 1, 2) for c in range(cols))
    v_odd = sorted((idx(r, c), idx(r + 1, c)) for r in range(1, rows - 1, 2) for c in range(cols))
    one_fac = [c for c in (h_even, h_odd, v_even, v_odd) if c] or None

    pm: Matching | None = None
    if cols % 2 == 0:
        pm = h_even  # every row paired (0,1),(2,3),... -> covers all vertices
    elif rows % 2 == 0:
        pm = v_even
    sampler = None
    if pm is not None:
        pm_t = [tuple(e) for e in pm]

        def sampler(rng: np.random.Generator, _pm=pm_t) -> Matching:
            return list(_pm)

    return GraphTopology(
        name=f"lattice_2d({rows}x{cols})",
        n=n,
        edges=edges,
        is_stochastic=False,
        one_factorization=one_fac,
        graph6=graph6_encode(n, edges),
        fresh_matching_sampler=sampler,
    )


# ----------------------------------------------------------------------
# Adapter: any simple, undirected networkx graph
# ----------------------------------------------------------------------


def _edge_coloring(edges: list[Edge]) -> list[Matching]:
    """Proper edge colouring (color classes = matchings) via networkx.

    Greedy node-colouring of the line graph: adjacent edges (sharing a vertex)
    get different colours, so each colour class is a matching and the classes
    partition the edges. Perfect matchings when the graph is regular & Class-1.
    """
    if not edges:
        return []
    import networkx as nx

    line = nx.line_graph(nx.Graph(edges))
    coloring = nx.greedy_color(line, strategy="largest_first")
    classes: dict[int, list[Edge]] = {}
    for e, color in coloring.items():
        a, b = e
        classes.setdefault(color, []).append((min(a, b), max(a, b)))
    return [sorted(classes[c]) for c in sorted(classes)]


def _make_pm_sampler(n: int, edges: list[Edge]):
    """A randomized perfect-matching sampler, or ``None`` if no PM exists.

    Draws a maximum-weight matching with fresh random edge weights each call,
    a perfect matching when one exists. **Not exactly uniform** for general
    graphs (exact uniform PM sampling is #P-hard); the structured cycle/complete
    samplers are uniform, this one is best-effort.
    """
    import networkx as nx

    base = nx.Graph()
    base.add_nodes_from(range(n))
    base.add_edges_from(edges)
    probe = nx.max_weight_matching(base, maxcardinality=True)
    if len(probe) * 2 != n:
        return None  # no perfect matching

    # Build the graph once; each call only re-randomizes the edge weights. The
    # sampler mutates this shared graph, so it is single-threaded (each circuit
    # builder runs its layers sequentially; parallel sweeps get a fresh copy
    # per worker process).
    edge_list = list(edges)

    def sample(rng: np.random.Generator) -> Matching:
        weights = dict(zip(edge_list, (float(w) for w in rng.random(len(edge_list))), strict=True))
        nx.set_edge_attributes(base, weights, "weight")
        m = nx.max_weight_matching(base, maxcardinality=True)
        return sorted((min(a, b), max(a, b)) for a, b in m)

    return sample


def from_networkx(g, *, name: str | None = None) -> GraphTopology:
    """Build a :class:`GraphTopology` from any simple, undirected networkx graph.

    Nodes are relabelled to ``0 .. n-1`` in iteration order. ``random_edge`` /
    ``random_pool`` gating works on any graph (edges only); brickwork uses a
    proper edge colouring (always available) for ``round_robin`` / ``palette``,
    and a best-effort perfect-matching sampler for ``fresh`` when the graph has
    a perfect matching.

    Raises :class:`InvalidArgumentError` for directed, multigraph, or
    self-looped inputs, or if the ``graph`` extra (networkx) is not installed.
    """
    try:
        import networkx as nx
    except ImportError as e:  # pragma: no cover
        raise InvalidArgumentError(
            "passing a networkx graph needs the 'graph' extra: pip install sparsegf2[graph]"
        ) from e
    if not isinstance(g, nx.Graph):
        raise InvalidArgumentError(
            f"from_networkx expects a networkx.Graph; got {type(g).__name__}"
        )
    if g.is_directed():
        raise InvalidArgumentError("from_networkx requires an undirected graph (no DiGraph)")
    if g.is_multigraph():
        raise InvalidArgumentError("from_networkx requires a simple graph (no MultiGraph)")
    if any(u == v for u, v in g.edges()):
        raise InvalidArgumentError("from_networkx requires a simple graph (no self-loops)")

    nodes = list(g.nodes())
    mapping = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    if n < 2:
        raise InvalidArgumentError(f"from_networkx requires >= 2 nodes; got {n}")
    edges = sorted(
        {(min(mapping[u], mapping[v]), max(mapping[u], mapping[v])) for u, v in g.edges()}
    )
    one_fac = _edge_coloring(edges) or None
    sampler = _make_pm_sampler(n, edges)
    return GraphTopology(
        name=name or f"networkx({n}n_{len(edges)}e)",
        n=n,
        edges=edges,
        is_stochastic=False,
        one_factorization=one_fac,
        graph6=graph6_encode(n, edges),
        fresh_matching_sampler=sampler,
    )


def newman_watts(
    n: int,
    k: int = 2,
    p: float = 0.5,
    *,
    seed: int = 0,
    name: str | None = None,
) -> GraphTopology:
    r"""Newman-Watts-Strogatz small-world graph (a **stochastic** geometry).

    Start from a ring lattice on ``n`` nodes where each node is joined to its
    ``k`` nearest neighbours, then *add* a shortcut across each lattice edge
    independently with probability ``p`` (no rewiring, no edge removal). The
    result is connected and small-world but, unlike the ring, **not regular**,
    so it admits **no 1-factorization** (no partition of the edges into perfect
    matchings).

    Consequence for the gating/matching modes:

    * ``random_edge`` and ``random_pool`` gating work (they need only the edge
      set), together with every measurement mode and picture.
    * ``brickwork`` with ``matching_mode="fresh"`` works **when the realisation
      has a perfect matching** (a fresh random perfect matching is drawn each
      layer).
    * ``brickwork`` with ``matching_mode="round_robin"`` or ``"palette"`` is
      **unavailable**: those consume a fixed 1-factorization, which a stochastic
      non-regular graph does not have. The config raises a clear error.

    Parameters
    ----------
    n : int
        Number of nodes (``>= 3``).
    k : int
        Ring connectivity: each node is joined to its ``k`` nearest neighbours
        (``k`` even, ``2 <= k < n``). ``k = 2`` is the bare ring.
    p : float
        Shortcut probability per lattice edge, in ``[0, 1]``.
    seed : int
        Realisation seed. A given ``(n, k, p, seed)`` is one fixed graph; vary
        the seed to draw another realisation from the ensemble. Used as a
        string spec (``graph_spec="newman_watts"``), the config's ``base_seed``
        is the realisation seed, so sweeping ``base_seed`` averages over graphs.
    name : str, optional
        Override the auto-generated name.

    References
    ----------
    Newman & Watts, *Renormalization group analysis of the small-world network
    model*, Phys. Lett. A **263**, 341 (1999),
    `cond-mat/9903357 <https://arxiv.org/abs/cond-mat/9903357>`_.
    """
    try:
        import networkx as nx
    except ImportError as e:  # pragma: no cover
        raise InvalidArgumentError(
            "newman_watts needs the 'graph' extra: pip install sparsegf2[graph]"
        ) from e
    if n < 3:
        raise InvalidArgumentError(f"newman_watts needs n >= 3; got {n}")
    if k < 2 or k >= n or k % 2 != 0:
        raise InvalidArgumentError(f"newman_watts needs even k with 2 <= k < n; got k={k}, n={n}")
    if not 0.0 <= p <= 1.0:
        raise InvalidArgumentError(f"newman_watts needs p in [0, 1]; got {p}")
    # nx labels nodes 0..n-1 already; the ring base guarantees connectivity.
    g = nx.newman_watts_strogatz_graph(n, k, p, seed=int(seed))
    edges = sorted({(min(u, v), max(u, v)) for u, v in g.edges()})
    sampler = _make_pm_sampler(n, edges)  # enables brickwork+fresh iff a PM exists
    return GraphTopology(
        name=name or f"newman_watts(n={n},k={k},p={p},seed={seed})",
        n=n,
        edges=edges,
        is_stochastic=True,
        one_factorization=None,  # stochastic / non-regular: no 1-factorization
        graph6=graph6_encode(n, edges),
        fresh_matching_sampler=sampler,
    )


# ----------------------------------------------------------------------
# Spec parser
# ----------------------------------------------------------------------


def _lattice_square(n: int) -> GraphTopology:
    """Square ``s x s`` grid for ``from_spec("lattice_2d", n)`` when ``n=s^2``."""
    import math

    s = math.isqrt(n)
    if s * s != n:
        raise InvalidArgumentError(
            f"from_spec('lattice_2d', n) needs a square n; n={n} is not a perfect square. "
            "Use lattice_2d(rows, cols) directly for a rectangular grid."
        )
    return lattice_2d(s, s)


_GRAPH_CONSTRUCTORS = {
    "cycle": cycle_graph,
    "complete": complete_graph,
    "path": path_graph,
    "lattice_2d": _lattice_square,
    # For an arbitrary geometry, pass a networkx graph to from_networkx(g) (or as
    # graph_spec directly); it is not a string-spec family.
}


def from_spec(spec: str, n: int, seed: int = 0) -> GraphTopology:
    """Build a graph topology from a string specification.

    Parameters
    ----------
    spec : str
        One of ``"cycle"``, ``"complete"``, ``"path"``, ``"lattice_2d"``,
        ``"newman_watts"`` (case-insensitive). ``"lattice_2d"`` builds the square
        ``s x s`` grid when ``n = s^2``; ``"newman_watts"`` builds a stochastic
        small-world graph seeded by ``seed`` (default ``k=2``, ``p=0.5``; pass
        :func:`newman_watts` directly to tune them). Use :func:`from_networkx`
        for an arbitrary geometry.
    n : int
        Number of qubits.
    seed : int
        Realisation seed for stochastic specs (``"newman_watts"``); ignored for
        the deterministic graphs.

    Returns
    -------
    GraphTopology
    """
    spec = spec.strip().lower()
    if spec == "newman_watts":
        return newman_watts(n, seed=seed)
    if spec not in _GRAPH_CONSTRUCTORS:
        supported = ", ".join(sorted([*_GRAPH_CONSTRUCTORS, "newman_watts"]))
        raise InvalidArgumentError(f"Unknown graph spec {spec!r}. Supported: {supported}.")
    return _GRAPH_CONSTRUCTORS[spec](n)


__all__ = [
    "Edge",
    "Matching",
    "GraphTopology",
    "cycle_graph",
    "complete_graph",
    "from_networkx",
    "from_spec",
    "graph6_encode",
    "lattice_2d",
    "newman_watts",
    "path_graph",
]
