"""Exact rewiring counts for the production Watts--Strogatz graphs.

The graph constructor stores only the final edge set, but a graph can undergo
more successful rewiring operations than the number of base-lattice edges that
are absent at the end.  A later operation can restore a base edge removed by an
earlier one.  This module replays the constructor once, with the same RNG and
adjacency updates, and reports both quantities without changing the generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sparsegf2.circuits.graphs import _WS_STREAM, Edge

from .spec import _strict_integer, beta_from_key, canonical_beta_key


@dataclass(frozen=True, slots=True)
class RewiringCounts:
    """Counts from one exact Watts--Strogatz rewiring replay.

    ``successful_operations`` counts Bernoulli-selected steps that found a
    valid nonneighbor and therefore moved an edge. ``restored_base_edges``
    counts successful steps whose new edge belongs to the original circulant.
    ``displaced_base_edges`` is the number of original edges absent from the
    final graph. ``skipped_full_neighbor`` counts selected steps for which the
    kept endpoint was already adjacent to every other vertex.
    """

    successful_operations: int
    restored_base_edges: int
    displaced_base_edges: int
    skipped_full_neighbor: int


def _validated_parameters(
    n: int, k: int, beta: float, seed: int
) -> tuple[int, int, int, float, int]:
    """Validate inputs and return the canonical generator parameters."""
    n_value = _strict_integer(n, "n", minimum=3)
    k_value = _strict_integer(k, "k", minimum=1)
    if 2 * k_value >= n_value:
        raise ValueError(
            f"k must satisfy 1 <= k < n/2 (so degree 2k < n); got k={k_value}, n={n_value}"
        )
    seed_value = _strict_integer(seed, "seed")
    beta_key = canonical_beta_key(beta)
    return n_value, k_value, beta_key, beta_from_key(beta_key), seed_value


def _replay_watts_strogatz_rewiring(
    n: int,
    k: int,
    beta: float,
    seed: int,
) -> tuple[RewiringCounts, tuple[Edge, ...]]:
    """Replay the current graph generator and also return its final edges.

    The edge tuple is private test support: comparing it with
    :func:`sparsegf2.circuits.graphs._ws_rewire_edges` guards the RNG call order
    and adjacency semantics on which the reported counts depend.
    """
    n_value, k_value, beta_key, beta_value, seed_value = _validated_parameters(n, k, beta, seed)
    rng = np.random.default_rng([seed_value, n_value, k_value, beta_key, _WS_STREAM])

    adjacency: list[set[int]] = [set() for _ in range(n_value)]
    order: list[Edge] = []
    for offset in range(1, k_value + 1):
        for kept in range(n_value):
            far = (kept + offset) % n_value
            adjacency[kept].add(far)
            adjacency[far].add(kept)
            order.append((kept, far))

    base_edges = {(min(left, right), max(left, right)) for left, right in order}
    successful_operations = 0
    restored_base_edges = 0
    skipped_full_neighbor = 0

    for kept, far in order:
        if rng.random() >= beta_value:
            continue
        forbidden = adjacency[kept] | {kept}
        if len(forbidden) >= n_value:
            skipped_full_neighbor += 1
            continue
        candidates = [vertex for vertex in range(n_value) if vertex not in forbidden]
        replacement = candidates[int(rng.integers(len(candidates)))]

        adjacency[kept].discard(far)
        adjacency[far].discard(kept)
        adjacency[kept].add(replacement)
        adjacency[replacement].add(kept)
        successful_operations += 1
        replacement_edge = (min(kept, replacement), max(kept, replacement))
        if replacement_edge in base_edges:
            restored_base_edges += 1

    final_edges = tuple(
        sorted(
            {
                (min(left, right), max(left, right))
                for left in range(n_value)
                for right in adjacency[left]
            }
        )
    )
    displaced_base_edges = len(base_edges.difference(final_edges))
    if displaced_base_edges != successful_operations - restored_base_edges:
        raise RuntimeError("Watts--Strogatz rewiring accounting invariant failed")

    return (
        RewiringCounts(
            successful_operations=successful_operations,
            restored_base_edges=restored_base_edges,
            displaced_base_edges=displaced_base_edges,
            skipped_full_neighbor=skipped_full_neighbor,
        ),
        final_edges,
    )


def watts_strogatz_rewiring_counts(
    n: int,
    k: int,
    beta: float,
    seed: int,
) -> RewiringCounts:
    """Return exact rewiring counts for one canonical WS realization.

    ``beta`` is first quantized to the registry's ``1e-9`` integer identity,
    then that canonical value is used both in the RNG seed and in each
    Bernoulli comparison. Consequently, inputs with the same canonical beta
    key produce identical counts, matching reconstruction from registry rows.
    """
    counts, _ = _replay_watts_strogatz_rewiring(n, k, beta, seed)
    return counts


__all__ = ["RewiringCounts", "watts_strogatz_rewiring_counts"]
