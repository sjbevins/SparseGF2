from __future__ import annotations

import json
import sqlite3

import pytest
from studies.prl_production.graph_registry import GraphRegistryDatabase
from studies.prl_production.graph_registry.build import (
    build_collection,
    main,
    validate_existing_collection,
    write_reports,
)
from studies.prl_production.graph_registry.database import RegistryConflictError
from studies.prl_production.graph_registry.spec import smoke_spec


def test_smoke_collection_build_resume_reports_and_cell_vectors(tmp_path) -> None:
    spec = smoke_spec()
    database_path = tmp_path / "collection" / "registry.sqlite3"

    first = build_collection(spec, database_path, batch_size=5)
    second = build_collection(spec, database_path, batch_size=7)

    assert first.collection_id == spec.collection_id
    assert second.ensemble_id == first.ensemble_id
    assert first.validation.graph_count == 24
    assert first.validation.cell_count == 6
    assert first.validation.sqlite_integrity == "ok"
    assert first.validation.foreign_key_violations == 0
    assert first.validation.reconstructed_graphs == 8
    assert second.validation.seed_content_sha256 == first.validation.seed_content_sha256

    registry = GraphRegistryDatabase(database_path)
    cell = registry.graphs_for_cell(first.ensemble_id, 8, 5_000_000)
    assert len(cell) == 4
    assert [graph.graph_index for graph in cell] == [0, 1, 2, 3]
    assert [graph.graph_seed for graph in cell] == [
        spec.graph_seed(8, 5_000_000, graph_index) for graph_index in range(4)
    ]

    manifest_path, status_path = write_reports(spec, second, publish_status=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["validation"]["graph_count"] == 24
    assert manifest["seed_derivation"] == "sha256_tuple_v1"
    assert "seed collection complete and validated" in status_path.read_text(encoding="utf-8")


def test_complete_collection_repairs_a_missing_row_but_rejects_seed_conflict(tmp_path) -> None:
    spec = smoke_spec()
    database_path = tmp_path / "registry.sqlite3"
    summary = build_collection(spec, database_path)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            DELETE FROM graphs
            WHERE ensemble_id = ? AND n = 8 AND beta_key = 0 AND graph_index = 3
            """,
            (summary.ensemble_id,),
        )
    repaired = build_collection(spec, database_path)
    assert repaired.validation.graph_count == spec.n_graphs

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            UPDATE graphs SET graph_seed = '17'
            WHERE ensemble_id = ? AND n = 8 AND beta_key = 0 AND graph_index = 3
            """,
            (summary.ensemble_id,),
        )
    with pytest.raises(RegistryConflictError, match="different beta, seed, or metadata"):
        build_collection(spec, database_path)


def test_validation_requires_an_existing_registry(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="registry does not exist"):
        validate_existing_collection(smoke_spec(), tmp_path / "missing.sqlite3")


def test_production_cli_requires_explicit_confirmation_but_dry_run_does_not(
    tmp_path, capsys
) -> None:
    database = tmp_path / "production.sqlite3"
    with pytest.raises(SystemExit, match="requires --confirm-production"):
        main(["--profile", "production", "--database", str(database)])
    assert not database.exists()

    assert (
        main(
            [
                "--profile",
                "production",
                "--database",
                str(database),
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_beta"] == 50
    assert payload["n_cells"] == 350
    assert payload["n_graphs"] == 350_000
    assert not database.exists()
