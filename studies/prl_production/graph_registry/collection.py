"""Convenient read API for an indexed graph collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .database import GraphRegistryDatabase, RegisteredGraph
from .spec import canonical_beta_key


class GraphCollection:
    """Expose each ``(n, beta)`` cell as an immutable, indexable graph vector."""

    def __init__(self, database_path: str | Path, collection_id: str) -> None:
        self.registry = GraphRegistryDatabase(database_path)
        if not isinstance(collection_id, str) or not collection_id:
            raise ValueError("collection_id must be nonempty text")
        with self.registry.read_connection() as connection:
            row = connection.execute(
                """
                SELECT ensemble_id, expected_graphs_per_cell
                FROM ensembles WHERE ensemble_key = ?
                """,
                (collection_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown collection_id={collection_id!r}")
        self.collection_id = collection_id
        self.ensemble_id = int(row["ensemble_id"])
        expected = row["expected_graphs_per_cell"]
        if expected is None:
            raise ValueError("collection does not declare its expected graph count per cell")
        self.graphs_per_cell = int(expected)

    def cell_by_key(self, n: int, beta_key: int) -> tuple[RegisteredGraph, ...]:
        """Return one complete cell in direct ``graph_index`` order."""
        graphs = self.registry.graphs_for_cell(self.ensemble_id, n, beta_key)
        if not graphs:
            raise KeyError(f"collection has no cell for n={n}, beta_key={beta_key}")
        expected_indices = tuple(range(self.graphs_per_cell))
        indices = tuple(graph.graph_index for graph in graphs)
        if indices != expected_indices:
            raise RuntimeError(
                f"cell n={n}, beta_key={beta_key} is incomplete or noncanonical: "
                f"found {len(graphs)} indexed rows"
            )
        return graphs

    def cell(self, n: int, beta: float) -> tuple[RegisteredGraph, ...]:
        """Return the vector ``(G_0, ..., G_N-1)`` for one ``(n, beta)`` pair."""
        return self.cell_by_key(n, canonical_beta_key(beta))

    def seed_vector(self, n: int, beta: float) -> tuple[int, ...]:
        """Return the stored graph-seed vector for one complete cell."""
        return tuple(graph.graph_seed for graph in self.cell(n, beta))

    def graph(self, n: int, beta: float, graph_index: int) -> RegisteredGraph:
        """Return one graph by its direct vector index."""
        if isinstance(graph_index, bool) or not isinstance(graph_index, int):
            raise TypeError("graph_index must be an integer")
        if not 0 <= graph_index < self.graphs_per_cell:
            raise IndexError(
                f"graph_index must lie in [0, {self.graphs_per_cell}); got {graph_index}"
            )
        return self.cell(n, beta)[graph_index]

    def snapshot(self, graph: RegisteredGraph) -> dict[str, Any]:
        """Return the expandable invariant/result/artifact view for one graph."""
        if not isinstance(graph, RegisteredGraph):
            raise TypeError("graph must be a RegisteredGraph")
        if graph.ensemble_id != self.ensemble_id:
            raise ValueError("graph belongs to a different collection")
        local = self.cell_by_key(graph.n, graph.beta_key)[graph.graph_index]
        if local != graph:
            raise ValueError("graph does not match the stored record in this collection")
        return self.registry.graph_snapshot(graph.graph_id)


__all__ = ["GraphCollection"]
