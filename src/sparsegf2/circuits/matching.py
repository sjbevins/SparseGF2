"""Matching selection for graph-defined circuits.

Given a graph and a layer index, choose *which* perfect matching (set of
disjoint edges = the gate pairs) fires this layer. Three modes:

- ``round_robin``: cycle deterministically through a fixed
  1-factorization; layer ``t`` uses matching ``t mod chi'``. No RNG draw.
- ``palette``: at each layer sample one matching uniformly from the
  same fixed 1-factorization (the "palette"). One RNG draw per layer.
- ``fresh``: at each layer sample a uniformly-random perfect
  matching of the graph from *all* perfect matchings (not restricted to a
  palette). Uses the graph's ``fresh_matching_sampler``.

Requirements (checked here as defense-in-depth; the config validator
catches them earlier):

- ``round_robin`` / ``palette`` need the graph to admit a 1-factorization
  (regular + class 1 under Vizing). Both built-in graphs satisfy this for even ``n``.
- ``fresh`` needs the graph to have at least one perfect matching.

This module has **no** simulator coupling, just pure graph combinatorics.
"""

from __future__ import annotations

import numpy as np

from sparsegf2.circuits.graphs import GraphTopology, Matching
from sparsegf2.errors import InvalidArgumentError

# Canonical set of matching-mode identifiers accepted throughout the package.
MATCHING_MODES: tuple[str, ...] = ("round_robin", "palette", "fresh")


def select_matching(
    graph: GraphTopology,
    mode: str,
    layer_index: int,
    rng: np.random.Generator,
) -> Matching:
    """Select the matching (gate pairs) to apply at ``layer_index``.

    Parameters
    ----------
    graph : GraphTopology
        The graph whose matchings are drawn.
    mode : str
        One of :data:`MATCHING_MODES`.
    layer_index : int
        0-based index of the current circuit layer.
    rng : numpy.random.Generator
        RNG used by the stochastic modes (``palette``, ``fresh``). Unused
        by ``round_robin``, but the caller passes one for a uniform
        signature **and** so the RNG draw-order stays mode-independent
        upstream of this call.

    Returns
    -------
    list of (int, int)
        The chosen matching's edges, canonicalized ``(u, v)`` with ``u < v``.

    Raises
    ------
    InvalidArgumentError
        If ``mode`` is not a valid matching-mode name. (Subclasses
        ``ValueError``.)
    RuntimeError
        If the mode is incompatible with the graph (defense-in-depth; the
        config validator catches this eagerly, so this should be
        unreachable in normal use).
    """
    if mode == "round_robin":
        if graph.one_factorization is None:
            raise InvalidArgumentError(
                f"round_robin requires a 1-factorization; graph {graph.name} has none"
            )
        chi = len(graph.one_factorization)
        return list(graph.one_factorization[layer_index % chi])

    if mode == "palette":
        if graph.one_factorization is None:
            raise InvalidArgumentError(
                f"palette requires a 1-factorization; graph {graph.name} has none"
            )
        chi = len(graph.one_factorization)
        idx = int(rng.integers(0, chi))
        return list(graph.one_factorization[idx])

    if mode == "fresh":
        if graph.fresh_matching_sampler is None:
            raise InvalidArgumentError(
                f"fresh requires a perfect-matching sampler; graph {graph.name} has none"
            )
        return list(graph.fresh_matching_sampler(rng))

    raise InvalidArgumentError(f"Unknown matching_mode {mode!r}; valid modes: {MATCHING_MODES}")


def available_modes(graph: GraphTopology) -> list[str]:
    """Return the matching-modes compatible with ``graph``.

    Used by the validator to report "available modes" when an incompatible
    mode is requested.
    """
    out: list[str] = []
    if graph.one_factorization is not None:
        out.extend(("round_robin", "palette"))
    if graph.fresh_matching_sampler is not None:
        out.append("fresh")
    return out


__all__ = ["MATCHING_MODES", "select_matching", "available_modes"]
