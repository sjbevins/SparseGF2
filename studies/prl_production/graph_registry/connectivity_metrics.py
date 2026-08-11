"""Algebraic-connectivity metrics for production Watts--Strogatz graphs.

The functions here use the combinatorial Laplacian ``L = D - A``.  They are
pure computations: each call owns its arrays, no process-wide numerical state
is changed, and the Watts--Strogatz wrapper reconstructs exactly the graph
identified by the production registry's canonical ``(n, beta, seed)`` tuple.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Integral

import numpy as np
from scipy.linalg import eigh

from sparsegf2.circuits.graphs import Edge, _ws_rewire_edges

from .spec import _strict_integer, beta_from_key, canonical_beta_key


def _validated_size(n: int, *, minimum: int) -> int:
    """Return ``n`` as a built-in integer after strict validation."""
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError(f"n must be an integer, not {type(n).__name__}")
    value = int(n)
    if value < minimum:
        raise ValueError(f"n must be >= {minimum}; got {value}")
    return value


def ring_algebraic_connectivity(n: int, k: int = 2) -> float:
    r"""Return the exact-formula gap of the unrewired circulant ``C(n, k)``.

    The combinatorial-Laplacian eigenvalues are

    .. math::

        \lambda_j = 2k - 2\sum_{c=1}^k \cos(2\pi c j/n).

    For the connected range-power cycle validated here, the smallest nonzero
    member is attained at ``j = 1`` (and ``j = n - 1``).
    """
    size = _validated_size(n, minimum=3)
    neighbors = _strict_integer(k, "k", minimum=1)
    if 2 * neighbors >= size:
        raise ValueError(
            f"k must satisfy 1 <= k < n/2 (so degree 2k < n); got k={neighbors}, n={size}"
        )
    angle = math.tau / size
    return 2.0 * neighbors - 2.0 * math.fsum(
        math.cos(offset * angle) for offset in range(1, neighbors + 1)
    )


def circulant_c2_algebraic_connectivity(n: int) -> float:
    """Return the exact-formula gap of the degree-four ``C(n, 2)``."""
    return ring_algebraic_connectivity(n, k=2)


def _validated_edges(n: int, edges: Iterable[Edge]) -> tuple[tuple[Edge, ...], bool]:
    """Return canonical simple-undirected edges and whether the graph is connected."""
    if isinstance(edges, (str, bytes)):
        raise TypeError("edges must be an iterable of integer endpoint pairs")

    canonical: set[Edge] = set()
    adjacency: list[list[int]] = [[] for _ in range(n)]
    try:
        iterator = iter(edges)
    except TypeError as exc:
        raise TypeError("edges must be an iterable of integer endpoint pairs") from exc

    for index, edge in enumerate(iterator):
        try:
            endpoints = tuple(edge)
        except TypeError as exc:
            raise TypeError(f"edge {index} must be an iterable endpoint pair") from exc
        if len(endpoints) != 2:
            raise ValueError(f"edge {index} must contain exactly two endpoints")
        left, right = endpoints
        if isinstance(left, bool) or not isinstance(left, Integral):
            raise TypeError(f"edge {index} endpoint 0 must be an integer")
        if isinstance(right, bool) or not isinstance(right, Integral):
            raise TypeError(f"edge {index} endpoint 1 must be an integer")
        u, v = int(left), int(right)
        if not 0 <= u < n or not 0 <= v < n:
            raise ValueError(f"edge {index} endpoints must lie in [0, {n}); got ({u}, {v})")
        if u == v:
            raise ValueError(f"edge {index} is a self-loop at vertex {u}")
        canonical_edge = (min(u, v), max(u, v))
        if canonical_edge in canonical:
            raise ValueError(f"edge {index} duplicates undirected edge {canonical_edge}")
        canonical.add(canonical_edge)
        adjacency[u].append(v)
        adjacency[v].append(u)

    seen = {0}
    pending = [0]
    while pending:
        vertex = pending.pop()
        for neighbor in adjacency[vertex]:
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return tuple(sorted(canonical)), len(seen) == n


def algebraic_connectivity_from_edges(n: int, edges: Iterable[Edge]) -> float:
    """Return ``lambda_2`` of a validated simple undirected graph.

    The graph may be disconnected, in which case its algebraic connectivity is
    exactly zero.  Connected graphs are evaluated from only the two lowest
    eigenvalues of their dense combinatorial Laplacian.  This is the relevant
    exact dense calculation for the production sizes, rather than an iterative
    sparse eigensolver with convergence tolerances.
    """
    size = _validated_size(n, minimum=2)
    canonical_edges, connected = _validated_edges(size, edges)
    if not connected:
        return 0.0

    laplacian = np.zeros((size, size), dtype=np.float64)
    for left, right in canonical_edges:
        laplacian[left, left] += 1.0
        laplacian[right, right] += 1.0
        laplacian[left, right] = -1.0
        laplacian[right, left] = -1.0

    eigenvalues = eigh(
        laplacian,
        subset_by_index=(0, 1),
        check_finite=True,
        overwrite_a=False,
        driver="evr",
        eigvals_only=True,
    )
    value = float(eigenvalues[1])
    scale = max(1.0, float(np.max(np.diag(laplacian))))
    roundoff = 64.0 * np.finfo(np.float64).eps * size * scale
    if value < -roundoff:
        raise RuntimeError(f"Laplacian eigensolver returned negative lambda_2={value}")
    return max(0.0, value)


def watts_strogatz_algebraic_connectivity(n: int, k: int, beta: float, seed: int) -> float:
    """Return ``lambda_2`` for one canonical production ``C(n,k)`` rewiring.

    ``beta`` is quantized to the registry's ``1e-9`` identity before both graph
    reconstruction and the calculation.  The private edge-only constructor is
    used deliberately to avoid building an unused perfect-matching sampler;
    parity with the public constructor is covered by focused tests.
    """
    size = _validated_size(n, minimum=3)
    neighbors = _strict_integer(k, "k", minimum=1)
    if 2 * neighbors >= size:
        raise ValueError(
            f"k must satisfy 1 <= k < n/2 (so degree 2k < n); got k={neighbors}, n={size}"
        )
    seed_value = _strict_integer(seed, "seed")
    beta_value = beta_from_key(canonical_beta_key(beta))
    edges = _ws_rewire_edges(size, neighbors, beta_value, seed_value)
    return algebraic_connectivity_from_edges(size, edges)


def watts_strogatz_c2_algebraic_connectivity(n: int, beta: float, seed: int) -> float:
    """Return ``lambda_2`` for one canonical production ``C(n,2)`` rewiring."""
    return watts_strogatz_algebraic_connectivity(n, 2, beta, seed)


__all__ = [
    "algebraic_connectivity_from_edges",
    "circulant_c2_algebraic_connectivity",
    "ring_algebraic_connectivity",
    "watts_strogatz_algebraic_connectivity",
    "watts_strogatz_c2_algebraic_connectivity",
]
