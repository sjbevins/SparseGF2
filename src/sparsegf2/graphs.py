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

import json
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from sparsegf2._kernel.rng import next_below, next_double, seed_state

__all__ = [
    "complete",
    "cycle",
    "degrees",
    "edge_coloring",
    "erdos_renyi",
    "from_edges",
    "from_networkx",
    "MAX_STATS_NODES",
    "lattice_2d",
    "load",
    "newman_watts",
    "path",
    "power_law_weights",
    "random_regular",
    "save",
    "shortcut_fraction",
    "stats",
    "watts_strogatz",
]


def from_edges(edges: Iterable[tuple[int, int]], num_nodes: int | None = None) -> np.ndarray:
    """Normalize any edge list: orient ``a < b``, drop duplicates, sort; reject self-loops."""
    raw = edges if isinstance(edges, np.ndarray) else list(edges)
    arr = np.asarray(raw, dtype=np.int64).reshape(-1, 2)
    if arr.size and np.min(arr) < 0:
        raise ValueError("node indices must be non-negative")
    if num_nodes is not None and arr.size and np.max(arr) >= num_nodes:
        raise ValueError(f"edge endpoint {int(np.max(arr))} out of range for {num_nodes} nodes")
    if np.any(arr[:, 0] == arr[:, 1]):
        raise ValueError("self-loops are not allowed")
    if _is_normalized(arr):  # e.g. a graph from this module: skip the (slow) row-wise unique
        return arr.copy()
    arr = np.sort(arr, axis=1)
    return np.unique(arr, axis=0) if arr.size else arr


def _is_normalized(arr: np.ndarray) -> bool:
    """Every pair oriented ``a < b``, and the pairs strictly increasing (sorted, no repeats)."""
    a, b = arr[:, 0], arr[:, 1]
    if not np.all(a < b):
        return False
    return bool(np.all((a[1:] > a[:-1]) | ((a[1:] == a[:-1]) & (b[1:] > b[:-1]))))


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
    """Square lattice; node ``r * cols + c``. ``periodic`` wraps both directions (a torus).

    With ``periodic=True`` a direction of length 2 or less is not wrapped (wrapping it
    would duplicate edges).
    """
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

    Start from a ring where each node links to its ``k`` nearest neighbors
    (``k / 2`` on each side). ``k`` must be even with ``2 <= k < n``; anything
    else raises ``ValueError``. Visiting edges ``(i, i + j)`` for
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

    Start from the ``k``-nearest-neighbor ring (``k`` even with ``2 <= k < n``;
    anything else raises ``ValueError``); for each of its edges
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
        normalized = np.sort(pairs, axis=1)
        if len(np.unique(normalized, axis=0)) == len(normalized):
            return from_edges(normalized)
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

    Applying one matching per time step generalizes the brickwork circuit to any
    graph: on a cycle with even ``n`` this gives the usual even/odd bonds. Uses
    greedy coloring in edge order, so at most ``2 * max_degree - 1`` matchings
    (often ``max_degree`` or ``max_degree + 1``). Returns a list of index arrays
    into ``edges``.
    """
    edges = np.asarray(edges, dtype=np.int64)
    colors_at: list[set[int]] = [set() for _ in range(n)]
    color_of = np.zeros(len(edges), dtype=np.int64)
    for e, (a, b) in enumerate(edges):
        c = 0
        while c in colors_at[a] or c in colors_at[b]:
            c += 1
        color_of[e] = c
        colors_at[a].add(c)
        colors_at[b].add(c)
    num = int(np.max(color_of)) + 1 if len(edges) else 0
    return [np.flatnonzero(color_of == c) for c in range(num)]


MAX_STATS_NODES = 4096


def _adjacency(edges: np.ndarray, n: int) -> np.ndarray:
    adj = np.zeros((n, n))
    adj[edges[:, 0], edges[:, 1]] = 1.0
    adj[edges[:, 1], edges[:, 0]] = 1.0
    return adj


def _csr(edges: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.concatenate([edges[:, 0], edges[:, 1]])
    b = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(a, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(a, minlength=n))
    return indptr, b[order].astype(np.int64)


def stats(edges: np.ndarray, n: int) -> dict:
    """Spectral, distance and clustering statistics of one graph on ``n`` nodes.

    ``lambda2``: the algebraic connectivity, i.e. the second-smallest
    eigenvalue of the Laplacian (0 when disconnected). ``connected``.
    ``mean_path_length``: the mean shortest-path length over ordered pairs of
    distinct nodes (NaN when disconnected). ``clustering``: the average local
    clustering coefficient, with nodes of degree < 2 counting 0. Then
    ``degree_min``, ``degree_mean`` and ``degree_max``. Dense linear algebra,
    so ``n <= MAX_STATS_NODES``.
    """
    from sparsegf2._kernel.graphstats import path_length_sum

    n = int(n)
    if n > MAX_STATS_NODES:
        raise ValueError(
            f"graph statistics take at most {MAX_STATS_NODES} nodes (dense linear algebra), got n = {n}"
        )
    e = from_edges(np.asarray(edges, dtype=np.int64).reshape(-1, 2), n)
    deg = degrees(e, n)
    adj = _adjacency(e, n)
    lam = np.linalg.eigvalsh(np.diag(adj.sum(axis=1)) - adj)
    lambda2 = float(max(lam[1], 0.0)) if n > 1 else 0.0
    triangles = ((adj @ adj) * adj).sum(axis=1) / 2.0
    k = adj.sum(axis=1)
    possible = k * (k - 1) / 2.0
    local = np.divide(triangles, possible, out=np.zeros(n), where=possible > 0)
    indptr, indices = _csr(e, n)
    total, pairs = path_length_sum(indptr, indices, n)
    connected = bool(pairs == n * (n - 1))
    return {
        "lambda2": lambda2,
        "connected": connected,
        "mean_path_length": float(total / pairs) if connected and pairs else float("nan"),
        "clustering": float(local.mean()) if n else 0.0,
        "degree_min": int(deg.min()) if n else 0,
        "degree_mean": float(deg.mean()) if n else 0.0,
        "degree_max": int(deg.max()) if n else 0,
    }


def shortcut_fraction(edges: np.ndarray, n: int, k: int) -> float:
    """Fraction of edges whose ring distance exceeds k/2: the shortcuts of a ring-based graph.

    ``k`` is the ring lattice's total degree (``watts_strogatz`` /
    ``newman_watts``), so its own edges reach k/2 sites each way.
    """
    if k % 2:
        raise ValueError(f"k must be even (k/2 neighbors per side), got {k}")
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not len(e):
        return 0.0
    r = np.abs(e[:, 0] - e[:, 1])
    r = np.minimum(r, n - r)
    return float(np.mean(r > k // 2))


def _checked(edges, n: int, where: str) -> np.ndarray:
    arr = np.asarray(edges, dtype=np.int64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{where}: edges must be [a, b] pairs")
    bad = arr[(arr < 0) | (arr >= n)]
    if bad.size:
        raise ValueError(f"{where}: node {int(bad[0])} outside [0, {n})")
    loops = np.flatnonzero(arr[:, 0] == arr[:, 1])
    if loops.size:
        raise ValueError(f"{where}: self-loop at node {int(arr[loops[0], 0])}")
    return arr


def _normalized(arr: np.ndarray, weights, where: str):
    oriented = np.sort(arr, axis=1)
    order = np.lexsort((oriented[:, 1], oriented[:, 0]))
    ranked = oriented[order]
    if len(ranked) > 1:
        same = np.flatnonzero(np.all(ranked[1:] == ranked[:-1], axis=1))
        if same.size:
            a, b = ranked[same[0]]
            raise ValueError(f"{where}: repeated edge ({int(a)}, {int(b)})")
    if weights is None:
        return ranked, None
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != (len(arr),):
        raise ValueError(f"{where}: {w.size} weights for {len(arr)} edges")
    return ranked, w[order]


def save(path, graphs_, num_nodes: int, weights=None) -> Path:
    """Save a graph collection to ``.json`` or ``.npz`` (the format follows the suffix).

    ``graphs_`` is one edge array or a list of them; ``weights`` (optional)
    is one array per graph, one weight per edge, in the order of its edges.
    """
    path = Path(path)
    gs = [graphs_] if isinstance(graphs_, np.ndarray) and np.ndim(graphs_) == 2 else list(graphs_)
    ws = None if weights is None else list(weights)
    n = int(num_nodes)
    pairs = [
        _normalized(_checked(g, n, f"graph {i}"), None if ws is None else ws[i], f"graph {i}")
        for i, g in enumerate(gs)
    ]
    if path.suffix == ".json":
        doc = {"num_nodes": n, "graphs": [e.tolist() for e, _ in pairs]}
        if ws is not None:
            doc["weights"] = [w.tolist() for _, w in pairs]
        path.write_text(json.dumps(doc), encoding="utf-8")
    elif path.suffix == ".npz":
        edges = np.concatenate([e for e, _ in pairs]) if pairs else np.zeros((0, 2), np.int64)
        ptr = np.cumsum([0] + [len(e) for e, _ in pairs]).astype(np.int64)
        arrays = {"num_nodes": np.int64(n), "edges": edges, "ptr": ptr}
        if ws is not None:
            arrays["weights"] = np.concatenate([w for _, w in pairs])
        with path.open("wb") as fh:
            # numpy's stubs type the **arrays of savez as its allow_pickle flag
            np.savez(fh, **arrays)  # type: ignore[arg-type, unused-ignore]
    else:
        raise ValueError(f"graph files end in .json or .npz, got {path.name!r}")
    return path


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    """Every array of a ``.npz`` graph file, with a plain message for what is not an archive
    (numpy's own text for a non-zip file talks about pickles, and it leaves the file open when
    a zip is truncated, so the handle is ours)."""
    problem = (
        f"{path.name}: not a readable .npz graph file ({{why}}); a graph file is a zip archive "
        "written by graphs.save or numpy.savez with the arrays num_nodes and edges"
    )
    with path.open("rb") as handle:
        try:
            archive = np.load(handle, allow_pickle=False)
        except (ValueError, EOFError, zipfile.BadZipFile):
            raise ValueError(
                problem.format(why="it is not a zip archive: empty, truncated or in another format")
            ) from None
        if not isinstance(archive, np.lib.npyio.NpzFile):  # a .npy file renamed
            raise ValueError(problem.format(why="it holds one array (.npy), not an archive"))
        with archive as z:
            try:
                return {name: z[name] for name in z.files}
            except ValueError:  # object arrays, which numpy refuses without allow_pickle
                raise ValueError(
                    problem.format(why="it holds arrays of Python objects, not numbers")
                ) from None


def load(path) -> tuple[int, list[np.ndarray], list[np.ndarray] | None]:
    """Load a collection saved by :func:`save` (or written by hand in the same format).

    JSON: ``{"num_nodes": n, "graphs": [[[a, b], ...], ...], "weights": [[w, ...], ...]}``,
    or ``"edges"`` for a single graph. NPZ: ``num_nodes``, ``edges`` (E x 2), ``ptr``
    (G + 1 offsets; optional for one graph), optionally ``weights`` (E). Returns
    ``(num_nodes, graphs, weights or None)`` with edges normalized as by
    :func:`from_edges` and the weights permuted with them. A ``.npz`` that is not a zip
    archive (empty, truncated, text, a renamed ``.npy``) is a ``ValueError`` that says so.
    """
    path = Path(path)
    if path.suffix == ".json":
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "num_nodes" not in doc:
            raise ValueError(f"{path.name}: missing num_nodes")
        raw = doc.get("graphs") or ([doc["edges"]] if "edges" in doc else None)
        if raw is None:
            raise ValueError(f"{path.name}: give 'graphs' (a list of edge lists) or 'edges'")
        weights = doc.get("weights")
        if weights is not None and "edges" in doc and "graphs" not in doc:
            weights = [weights]
    elif path.suffix == ".npz":
        z = _read_npz(path)
        if "num_nodes" not in z or "edges" not in z:
            raise ValueError(f"{path.name}: needs arrays num_nodes and edges")
        doc = {"num_nodes": int(z["num_nodes"])}
        edges = z["edges"]
        ptr = z["ptr"] if "ptr" in z else np.array([0, len(edges)])
        raw = [edges[ptr[i] : ptr[i + 1]] for i in range(len(ptr) - 1)]
        flat_w = z.get("weights")
        weights = (
            None if flat_w is None else [flat_w[ptr[i] : ptr[i + 1]] for i in range(len(ptr) - 1)]
        )
    else:
        raise ValueError(f"graph files end in .json or .npz, got {path.name!r}")
    n = int(doc["num_nodes"])
    out_g, out_w = [], []
    for i, g in enumerate(raw):
        where = f"{path.name}, graph {i}"
        e, w = _normalized(_checked(g, n, where), None if weights is None else weights[i], where)
        out_g.append(e)
        out_w.append(w)
    return n, out_g, (out_w if weights is not None else None)
