from __future__ import annotations

import sqlite3

import pytest
from studies.prl_production.graph_registry import (
    ArtifactRecord,
    ExperimentResultRecord,
    GraphRegistryDatabase,
    GraphSeedRecord,
    InvariantResultRecord,
    RegistryConflictError,
    RegistrySchemaError,
)
from studies.prl_production.graph_registry.database import APPLICATION_ID, SCHEMA_VERSION


def _registry(tmp_path) -> tuple[GraphRegistryDatabase, int]:
    registry = GraphRegistryDatabase(tmp_path / "graphs.sqlite3", busy_timeout_ms=1234)
    ensemble_id = registry.register_ensemble(
        "ws_mean_degree_4_v1",
        graph_family="watts_strogatz",
        expected_graphs_per_cell=1000,
        metadata={"mean_degree": 4},
    )
    return registry, ensemble_id


def test_versioned_schema_pragmas_foreign_keys_and_indexes(tmp_path) -> None:
    registry, _ = _registry(tmp_path)

    assert registry.schema_version == SCHEMA_VERSION
    with registry.read_connection() as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 1234
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "schema_migrations",
            "ensembles",
            "graphs",
            "experiments",
            "invariant_definitions",
            "invariant_results",
            "graph_experiment_results",
            "artifact_references",
        } <= tables
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        assert "idx_graphs_seed" in indexes
        assert "idx_experiment_results_experiment" in indexes


def test_graph_identity_is_not_seed_and_resume_is_idempotent(tmp_path) -> None:
    registry, ensemble_id = _registry(tmp_path)
    records = [GraphSeedRecord(64, 0.005, 5_000_000, index, 17) for index in (2, 0, 1)]

    graph_ids = registry.register_graphs(ensemble_id, records)
    assert len(set(graph_ids)) == 3
    assert registry.register_graphs(ensemble_id, records) == graph_ids

    cell = registry.graphs_for_cell(ensemble_id, 64, 5_000_000)
    assert [graph.graph_index for graph in cell] == [0, 1, 2]
    assert [graph.graph_seed for graph in cell] == [17, 17, 17]
    assert len({graph.graph_id for graph in cell}) == 3


def test_graph_batch_is_atomic_on_resume_conflict(tmp_path) -> None:
    registry, ensemble_id = _registry(tmp_path)
    registry.register_graphs(
        ensemble_id,
        [GraphSeedRecord(64, 0.005, 5_000_000, 0, 10)],
    )

    with pytest.raises(RegistryConflictError, match="different beta, seed, or metadata"):
        registry.register_graphs(
            ensemble_id,
            [
                GraphSeedRecord(64, 0.005, 5_000_000, 1, 11),
                GraphSeedRecord(64, 0.005, 5_000_000, 0, 99),
            ],
        )

    assert [
        graph.graph_index for graph in registry.graphs_for_cell(ensemble_id, 64, 5_000_000)
    ] == [0]


@pytest.mark.parametrize(
    "record",
    [
        GraphSeedRecord(True, 0.005, 5_000_000, 0, 1),
        GraphSeedRecord(64, float("nan"), 5_000_000, 0, 1),
        GraphSeedRecord(64, 0.005, 5_000_001, 0, 1),
        GraphSeedRecord(64, 0.005, 5_000_000, -1, 1),
        GraphSeedRecord(64, 0.005, 5_000_000, 0, 1 << 64),
    ],
)
def test_graph_input_ranges_are_strict(tmp_path, record) -> None:
    registry, ensemble_id = _registry(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        registry.register_graphs(ensemble_id, [record])


def test_expandable_graph_database_with_typed_results_and_artifact(tmp_path) -> None:
    registry, ensemble_id = _registry(tmp_path)
    graph_id = registry.register_graphs(
        ensemble_id,
        [GraphSeedRecord(64, 0.005, 5_000_000, 0, (1 << 64) - 1)],
    )[0]
    invariant_id = registry.define_invariant(
        "algebraic_connectivity",
        definition_version="1",
        value_kind="real",
        units="dimensionless",
    )
    experiment_id = registry.register_experiment(
        ensemble_id,
        "single_ref_tau_p_v1",
        kind="single_qubit_purification",
        protocol_version="1",
        parameters={"depth_cap": "8*n", "p": 0.2},
    )
    digest = "ab" * 32
    artifact_id = registry.register_artifacts(
        ensemble_id,
        [
            ArtifactRecord(
                "tau_p_raw_0",
                "artifacts/tau_p_raw_0.npz",
                digest,
                graph_id=graph_id,
                experiment_id=experiment_id,
                byte_size=123,
            )
        ],
    )[0]

    registry.upsert_invariant_results(
        [InvariantResultRecord(graph_id, invariant_id, status="running")]
    )
    registry.upsert_invariant_results([InvariantResultRecord(graph_id, invariant_id, value=0.125)])
    registry.upsert_experiment_results(
        [ExperimentResultRecord(graph_id, experiment_id, status="running")]
    )
    final = ExperimentResultRecord(
        graph_id,
        experiment_id,
        result={"event_observed": True, "tau_p": 31},
        artifact_id=artifact_id,
    )
    registry.upsert_experiment_results([final])
    registry.upsert_experiment_results([final])

    snapshot = registry.graph_snapshot(graph_id)
    assert snapshot["graph"]["graph_seed"] == (1 << 64) - 1
    assert snapshot["invariants"][0]["value"] == 0.125
    assert snapshot["experiments"][0]["result"]["tau_p"] == 31
    assert snapshot["artifacts"][0]["sha256"] == digest

    with pytest.raises(RegistryConflictError, match="terminal experiment result"):
        registry.upsert_experiment_results(
            [
                ExperimentResultRecord(
                    graph_id,
                    experiment_id,
                    result={"event_observed": True, "tau_p": 32},
                )
            ]
        )


def test_foreign_keys_and_artifact_checks_reject_invalid_data(tmp_path) -> None:
    registry, ensemble_id = _registry(tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        registry.register_artifacts(
            ensemble_id,
            [ArtifactRecord("bad", "bad.bin", "not-a-digest")],
        )

    with (
        registry._write_transaction() as connection,
        pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"),
    ):
        connection.execute(
            """
            INSERT INTO graphs(
                ensemble_id, n, beta, beta_key, graph_index, graph_seed,
                status, metadata_json, created_at, updated_at
            ) VALUES (999, 64, 0.0, 0, 0, '1', 'registered', '{}', 'now', 'now')
            """
        )


def test_newer_or_unrelated_database_is_refused(tmp_path) -> None:
    newer = tmp_path / "newer.sqlite3"
    with sqlite3.connect(newer) as connection:
        connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(RegistrySchemaError, match="newer"):
        GraphRegistryDatabase(newer)

    unrelated = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(unrelated) as connection:
        connection.execute("CREATE TABLE user_data(value TEXT)")
    with pytest.raises(RegistrySchemaError, match="non-registry"):
        GraphRegistryDatabase(unrelated)
