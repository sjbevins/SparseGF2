from __future__ import annotations

from dataclasses import replace

import pytest
from studies.prl_production.graph_registry import GraphCollection
from studies.prl_production.graph_registry.build import build_collection
from studies.prl_production.graph_registry.spec import smoke_spec


def test_collection_exposes_complete_indexable_cells_and_snapshots(tmp_path) -> None:
    spec = smoke_spec()
    summary = build_collection(spec, tmp_path / "registry.sqlite3")
    collection = GraphCollection(summary.database_path, spec.collection_id)

    cell = collection.cell(8, 0.005)
    assert len(cell) == 4
    assert collection.seed_vector(8, 0.005) == tuple(graph.graph_seed for graph in cell)
    assert collection.graph(8, 0.005, 2) == cell[2]
    snapshot = collection.snapshot(cell[2])
    assert snapshot["graph"]["graph_index"] == 2
    assert snapshot["graph"]["graph_seed"] == cell[2].graph_seed
    assert snapshot["invariants"] == []
    assert snapshot["experiments"] == []


def test_collection_rejects_unknown_incomplete_and_foreign_graphs(tmp_path) -> None:
    spec = smoke_spec()
    first = build_collection(spec, tmp_path / "first.sqlite3")
    other_spec = replace(spec, name="different_smoke", master_seed=spec.master_seed + 1)
    second = build_collection(other_spec, tmp_path / "second.sqlite3")
    collection = GraphCollection(first.database_path, spec.collection_id)
    other = GraphCollection(second.database_path, other_spec.collection_id)

    with pytest.raises(KeyError, match="no cell"):
        collection.cell(8, 0.25)
    with pytest.raises(IndexError, match="graph_index"):
        collection.graph(8, 0.005, 4)
    with pytest.raises(TypeError, match="graph_index"):
        collection.graph(8, 0.005, True)
    with pytest.raises(ValueError, match="does not match"):
        collection.snapshot(other.graph(8, 0.005, 0))
