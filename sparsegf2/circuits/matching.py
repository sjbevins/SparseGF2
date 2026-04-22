"""
Matching mode implementations for graph-defined circuits.

Three modes are provided. See ``sparsegf2/circuits/README.md`` for the
verbatim specification of each. Briefly:

- ``round_robin`` — cycle deterministically through a fixed 1-factorization;
  layer ``t`` uses matching ``t mod chi'``.
- ``palette``     — at each layer, sample one matching uniformly at random
  from the same fixed 1-factorization ("palette").
- ``fresh``       — at each layer, sample a uniformly-random perfect matching
  of the graph from *all* perfect matchings (not restricted to a palette).

Requirements
------------

- ``round_robin`` and ``palette`` require the graph to admit a 1-factorization
  (regular, class 1 under Vizing's theorem). Both MVP graphs (cycle, complete)
  satisfy this when ``n`` is even.
- ``fresh`` requires the graph to have at least one perfect matching.

The pre-flight validator (``sparsegf2.circuits.validator``) checks these
requirements before any simulation is run.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from sparsegf2.circuits.graphs import GraphTopology, Matching


# Canonical set of matching-mode identifiers accepted throughout the package.
MATCHING_MODES: Tuple[str, ...] = ("round_robin", "palette", "fresh")


def select_matching(
    graph: GraphTopology,
    mode: str,
    layer_index: int,
    rng: np.random.Generator,
) -> Matching:
    """Select the matching to apply at ``layer_index``.

    Parameters
    ----------
    graph : GraphTopology
        The graph whose matchings are being drawn.
    mode : str
        One of ``"round_robin"``, ``"palette"``, ``"fresh"``.
    layer_index : int
        Index of the current circuit layer (0-based).
    rng : numpy.random.Generator
        RNG used by the stochastic modes (``palette``, ``fresh``). Unused for
        ``round_robin``, but the caller must still pass one for a uniform
        signature.

    Returns
    -------
    list of (int, int)
        The edges of the chosen matching. Edges are canonicalized as
        ``(u, v)`` with ``u < v``.

    Raises
    ------
    ValueError
        If ``mode`` is not a valid matching-mode name.
    RuntimeError
        If the requested mode is incompatible with the graph. The pre-flight
        validator should have caught this earlier; this is a defense-in-depth
        guard.
    """
    if mode == "round_robin":
        if graph.one_factorization is None:
            raise RuntimeError(
                f"round_robin requires a 1-factorization; graph {graph.name} has none"
            )
        chi = len(graph.one_factorization)
        return list(graph.one_factorization[layer_index % chi])

    if mode == "palette":
        if graph.one_factorization is None:
            raise RuntimeError(
                f"palette requires a 1-factorization; graph {graph.name} has none"
            )
        chi = len(graph.one_factorization)
        idx = int(rng.integers(0, chi))
        return list(graph.one_factorization[idx])

    if mode == "fresh":
        if graph.fresh_matching_sampler is None:
            raise RuntimeError(
                f"fresh requires a perfect-matching sampler; graph {graph.name} has none"
            )
        return list(graph.fresh_matching_sampler(rng))

    raise ValueError(
        f"Unknown matching_mode {mode!r}; valid modes: {MATCHING_MODES}"
    )


def available_modes(graph: GraphTopology) -> List[str]:
    """Return the matching-modes that are compatible with ``graph``.

    Used by the validator to produce the "available modes" part of its report
    when an incompatible mode is requested.
    """
    out: List[str] = []
    if graph.one_factorization is not None:
        out.extend(("round_robin", "palette"))
    if graph.fresh_matching_sampler is not None:
        out.append("fresh")
    return out


__all__ = [
    "MATCHING_MODES",
    "select_matching",
    "available_modes",
]
