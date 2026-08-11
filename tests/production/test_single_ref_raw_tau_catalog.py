"""Strict catalog tests for generalized raw-tau artifact registration."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from studies.prl_production.single_ref.raw_tau.catalog import (
    CATALOG_APPLICATION_ID,
    CATALOG_SCHEMA_VERSION,
    RawTauCatalog,
    catalog_path,
)
from studies.prl_production.single_ref.raw_tau.io import file_sha256, write_deterministic_npz
from studies.prl_production.single_ref.raw_tau.providers import GridGraphProvider
from studies.prl_production.single_ref.raw_tau.storage import (
    WorkUnitProgress,
    logical_tau_digest,
    raw_tau_path,
)
from studies.prl_production.sweep_spec import (
    GraphCollectionGridSpec,
    GraphParameterGrid,
    ParameterAxis,
    ProbabilityGrid,
    ScientificEnvironmentContract,
    SingleReferenceProtocolSpec,
    SingleReferenceSweepSpec,
)


def _case():
    graphs = GraphCollectionGridSpec(
        name="catalog_graphs",
        graph_family="watts_strogatz",
        generator_name="tests.catalog_graph",
        generator_version="v1",
        sizes=(4,),
        parameter_grid=GraphParameterGrid(
            (
                ParameterAxis("beta", (0.0, 0.25)),
                ParameterAxis("range", (2,)),
            )
        ),
        graphs_per_cell=2,
        master_seed=17,
    )
    provider = GridGraphProvider(graphs, factory=lambda *_args: ((0, 1),))
    protocol = SingleReferenceProtocolSpec(
        n_circuits=3,
        q_scramble=1,
        q_max=2,
        p_grid=ProbabilityGrid("0.1", "0.2", "0.1"),
        master_seed=29,
    )
    sweep = SingleReferenceSweepSpec(
        name="catalog_test",
        graph_collection_sha256=graphs.specification_sha256,
        source_fingerprint_sha256="a" * 64,
        environment_contract=ScientificEnvironmentContract("3.test", "2.test", "0.test"),
        protocol=protocol,
    )
    cells = {cell.spec.cell_index: cell for cell in provider.cells()}
    units = tuple((cells[work.cell.cell_index], work) for work in sweep.work_units(graphs))
    return sweep, units


def _progress(data_root: Path, work):
    path = raw_tau_path(data_root, work)
    shape = work.raw_shape
    arrays = {
        "cell_sha256": np.str_(work.cell.cell_sha256),
        "p_decimal": np.str_(work.p_decimal),
        "graph_index": np.arange(shape[0], dtype=np.int32),
        "graph_seed": np.arange(shape[0], dtype=np.int64) + 100,
        "circuit_index": np.arange(shape[1], dtype=np.int32),
        "tau_p": np.ones(shape, dtype=np.int32),
        "stop_layer": np.ones(shape, dtype=np.int32),
        "event_observed": np.ones(shape, dtype=np.uint8),
        "complete": np.ones(shape, dtype=np.uint8),
        "reference_system_qubit": np.zeros(shape, dtype=np.int32),
    }
    write_deterministic_npz(path, arrays)
    total = work.graphs_per_cell * work.protocol.n_circuits
    return WorkUnitProgress(
        path=str(path),
        work_sha256=work.work_sha256,
        completed=total,
        total=total,
        events=total,
        censored=0,
        newly_completed=total,
        elapsed_s=0.1,
        artifact_sha256=file_sha256(path),
        logical_result_sha256=logical_tau_digest(arrays),
    )


def test_catalog_registers_generic_cells_and_index_layout_idempotently(
    tmp_path: Path,
) -> None:
    sweep, units = _case()
    with RawTauCatalog(tmp_path) as catalog:
        catalog.register_plan(sweep, units, expected_cell_count=2)
        catalog.register_plan(sweep, units, expected_cell_count=2)
        connection = catalog._connection
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA application_id").fetchone()[0] == (CATALOG_APPLICATION_ID)
        assert connection.execute("PRAGMA user_version").fetchone()[0] == (CATALOG_SCHEMA_VERSION)
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM graph_cells").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM work_units").fetchone()[0] == 4
        assert (
            connection.execute(
                "SELECT COUNT(DISTINCT generator_contract_sha256) FROM graph_cells"
            ).fetchone()[0]
            == 1
        )
        parameters = json.loads(
            connection.execute(
                "SELECT parameters_json FROM graph_cells ORDER BY cell_index LIMIT 1"
            ).fetchone()[0]
        )
        assert parameters == {
            "beta": {"type": "float64", "value": "0x0.0p+0"},
            "range": {"type": "integer", "value": "2"},
        }
        layout = json.loads(
            connection.execute("SELECT index_layout_json FROM work_units LIMIT 1").fetchone()[0]
        )
        assert layout["array_shape"] == [2, 3]
        assert layout["graph_index"] == "axis_0_zero_based"
        assert layout["circuit_index"] == "axis_1_zero_based"


def test_catalog_completion_verifies_artifact_and_is_resume_idempotent(
    tmp_path: Path,
) -> None:
    sweep, units = _case()
    cell, work = units[0]
    del cell
    with RawTauCatalog(tmp_path) as catalog:
        catalog.register_plan(sweep, units, expected_cell_count=2)
        progress = _progress(tmp_path, work)
        catalog.mark_complete(work, progress)
        catalog.mark_complete(work, progress)
        row = catalog.work_unit_record(work)
        assert row is not None
        assert row["status"] == "complete"
        assert row["completed_trajectories"] == 6
        assert row["artifact_sha256"] == progress.artifact_sha256
        assert row["logical_result_sha256"] == progress.logical_result_sha256
        assert row["artifact_relative_path"] == work.artifact_relative_path.as_posix()

    with RawTauCatalog(tmp_path) as reopened:
        reopened.register_plan(sweep, units, expected_cell_count=2)
        reopened.mark_complete(work, progress)
        assert reopened.work_unit_record(work)["status"] == "complete"


def test_catalog_uses_logical_result_not_container_bytes_for_equality(tmp_path: Path) -> None:
    sweep, units = _case()
    _cell, work = units[0]
    with RawTauCatalog(tmp_path) as catalog:
        catalog.register_plan(sweep, units, expected_cell_count=2)
        progress = _progress(tmp_path, work)
        catalog.mark_complete(work, progress)

        # ZIP readers permit trailing bytes.  This changes container provenance
        # while leaving every validated scientific array and its digest intact.
        path = Path(progress.path)
        with path.open("ab") as handle:
            handle.write(b"repacked-container")
        repacked = replace(progress, artifact_sha256=file_sha256(path))
        catalog.mark_complete(work, repacked)
        row = catalog.work_unit_record(work)
        assert row["artifact_sha256"] == repacked.artifact_sha256
        assert row["logical_result_sha256"] == progress.logical_result_sha256


def test_catalog_rejects_artifact_and_immutable_plan_conflicts(tmp_path: Path) -> None:
    sweep, units = _case()
    _cell, work = units[0]
    with RawTauCatalog(tmp_path) as catalog:
        catalog.register_plan(sweep, units, expected_cell_count=2)
        progress = _progress(tmp_path, work)
        bad_digest = replace(progress, artifact_sha256="0" * 64)
        with pytest.raises(ValueError, match="SHA-256"):
            catalog.mark_complete(work, bad_digest)

        bad_logical = replace(progress, logical_result_sha256="0" * 64)
        with pytest.raises(ValueError, match="logical-result"):
            catalog.mark_complete(work, bad_logical)

        catalog._connection.execute(
            "UPDATE work_units SET p_decimal = 'tampered' WHERE work_sha256 = ?",
            (work.work_sha256,),
        )
        with pytest.raises(ValueError, match="conflicts"):
            catalog.register_plan(sweep, units, expected_cell_count=2)


def test_catalog_rejects_an_unrecognized_database(tmp_path: Path) -> None:
    path = catalog_path(tmp_path)
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA application_id = 12345")
        connection.execute("PRAGMA user_version = 99")

    with pytest.raises(RuntimeError, match="unsupported catalog identity"):
        RawTauCatalog(tmp_path)


def test_catalog_does_not_claim_an_unversioned_nonempty_database(tmp_path: Path) -> None:
    path = catalog_path(tmp_path)
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated(value TEXT)")

    with pytest.raises(RuntimeError, match="unrecognized nonempty"):
        RawTauCatalog(tmp_path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert connection.execute(
            "SELECT name FROM sqlite_schema WHERE type = 'table'"
        ).fetchall() == [("unrelated",)]


def test_catalog_rejects_and_rolls_back_an_incomplete_cell_grid(tmp_path: Path) -> None:
    sweep, units = _case()
    with RawTauCatalog(tmp_path) as catalog:
        with pytest.raises(ValueError, match="complete measurement-rate grid"):
            catalog.register_plan(sweep, units[:-1], expected_cell_count=2)
        assert catalog._connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 0
        assert catalog._connection.execute("SELECT COUNT(*) FROM graph_cells").fetchone()[0] == 0
        assert catalog._connection.execute("SELECT COUNT(*) FROM work_units").fetchone()[0] == 0


def test_catalog_rejects_omission_of_a_complete_trailing_cell(tmp_path: Path) -> None:
    sweep, units = _case()
    with RawTauCatalog(tmp_path) as catalog:
        with pytest.raises(ValueError, match="complete zero-based graph-cell grid"):
            catalog.register_plan(sweep, units[:-2], expected_cell_count=2)
        assert catalog._connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 0
