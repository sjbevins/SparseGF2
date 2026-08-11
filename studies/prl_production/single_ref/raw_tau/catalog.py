"""Coordinator-owned SQLite inventory for raw-tau production artifacts.

The catalog contains only experiment identity, canonical graph cells, and one
row per ``(cell, p)`` artifact.  Trajectory arrays remain authoritative in the
deterministic NPZ shards; ``graph_index`` and ``circuit_index`` describe how a
catalog row joins to those arrays.  Worker processes never open this database.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from studies.prl_production.single_ref.shared_io import read_shared_bytes
from studies.prl_production.sweep_spec import SingleReferenceSweepSpec, TauWorkUnit

from .providers import ProviderCell
from .storage import WorkUnitProgress, logical_tau_digest, raw_tau_path

CATALOG_SCHEMA_VERSION = 1
CATALOG_APPLICATION_ID = 0x52544155  # ASCII "RTAU"


def catalog_path(data_root: str | Path) -> Path:
    """Return the generalized raw-tau result catalog path."""

    return Path(data_root) / "single_ref" / "raw_tau" / "catalog.sqlite3"


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _resolve_inside(root: Path, candidate: Path) -> tuple[Path, str]:
    resolved_root = root.resolve()
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"artifact {resolved} is outside data_root {resolved_root}") from exc
    return resolved, relative.as_posix()


class RawTauCatalog:
    """Strict, crash-safe artifact inventory owned by one coordinator process."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        self.path = catalog_path(self.data_root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._owner_pid = os.getpid()
        self._connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        try:
            self._initialize()
        except BaseException:
            self._connection.close()
            raise

    def __enter__(self) -> RawTauCatalog:
        self._assert_owner()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def close(self) -> None:
        self._assert_owner()
        self._connection.close()

    def _assert_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            raise RuntimeError("raw-tau catalog may only be used by its coordinator process")

    def _initialize(self) -> None:
        self._assert_owner()
        connection = self._connection
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if (application_id, user_version) not in {
            (0, 0),
            (CATALOG_APPLICATION_ID, CATALOG_SCHEMA_VERSION),
        }:
            raise RuntimeError(
                f"{self.path}: unsupported catalog identity/version "
                f"({application_id}, {user_version})"
            )
        if (application_id, user_version) == (0, 0):
            user_tables = connection.execute(
                """
                SELECT name FROM sqlite_schema
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
            if user_tables:
                raise RuntimeError(
                    f"{self.path}: refusing to claim an unrecognized nonempty SQLite database"
                )
        mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
        if mode != "wal":
            raise RuntimeError(f"{self.path}: could not enable WAL mode (got {mode!r})")
        connection.execute("PRAGMA synchronous = FULL")
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS catalog_metadata (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    catalog_kind TEXT NOT NULL CHECK (catalog_kind = 'single_ref_raw_tau'),
                    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
                    created_at_utc TEXT NOT NULL
                ) STRICT
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_sha256 TEXT PRIMARY KEY CHECK (length(experiment_sha256) = 64),
                    experiment_id TEXT NOT NULL UNIQUE,
                    collection_sha256 TEXT NOT NULL CHECK (length(collection_sha256) = 64),
                    source_fingerprint_sha256 TEXT NOT NULL
                        CHECK (length(source_fingerprint_sha256) = 64),
                    environment_contract_sha256 TEXT NOT NULL
                        CHECK (length(environment_contract_sha256) = 64),
                    specification_json TEXT NOT NULL,
                    registered_at_utc TEXT NOT NULL
                ) STRICT, WITHOUT ROWID
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_cells (
                    experiment_sha256 TEXT NOT NULL,
                    cell_sha256 TEXT NOT NULL CHECK (length(cell_sha256) = 64),
                    cell_index INTEGER NOT NULL CHECK (cell_index >= 0),
                    collection_id TEXT NOT NULL,
                    collection_sha256 TEXT NOT NULL CHECK (length(collection_sha256) = 64),
                    graph_family TEXT NOT NULL,
                    generator_name TEXT NOT NULL,
                    generator_version TEXT NOT NULL,
                    generator_contract_sha256 TEXT NOT NULL
                        CHECK (length(generator_contract_sha256) = 64),
                    n INTEGER NOT NULL CHECK (n >= 2),
                    parameters_json TEXT NOT NULL,
                    graphs_per_cell INTEGER NOT NULL CHECK (graphs_per_cell >= 1),
                    graph_index_start INTEGER NOT NULL CHECK (graph_index_start = 0),
                    graph_index_stop_exclusive INTEGER NOT NULL
                        CHECK (graph_index_stop_exclusive = graphs_per_cell),
                    PRIMARY KEY (experiment_sha256, cell_sha256),
                    UNIQUE (experiment_sha256, cell_index),
                    FOREIGN KEY (experiment_sha256)
                        REFERENCES experiments(experiment_sha256) ON DELETE RESTRICT
                ) STRICT, WITHOUT ROWID
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS work_units (
                    experiment_sha256 TEXT NOT NULL,
                    work_sha256 TEXT NOT NULL CHECK (length(work_sha256) = 64),
                    cell_sha256 TEXT NOT NULL CHECK (length(cell_sha256) = 64),
                    p_index INTEGER NOT NULL CHECK (p_index >= 0),
                    p_decimal TEXT NOT NULL,
                    graph_count INTEGER NOT NULL CHECK (graph_count >= 1),
                    circuit_count INTEGER NOT NULL CHECK (circuit_count >= 1),
                    expected_trajectories INTEGER NOT NULL CHECK (expected_trajectories >= 1),
                    index_layout_json TEXT NOT NULL,
                    artifact_relative_path TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('planned', 'complete')),
                    artifact_sha256 TEXT CHECK (
                        artifact_sha256 IS NULL OR length(artifact_sha256) = 64
                    ),
                    logical_result_sha256 TEXT CHECK (
                        logical_result_sha256 IS NULL OR length(logical_result_sha256) = 64
                    ),
                    completed_trajectories INTEGER NOT NULL
                        CHECK (completed_trajectories >= 0),
                    registered_at_utc TEXT NOT NULL,
                    completed_at_utc TEXT,
                    PRIMARY KEY (experiment_sha256, work_sha256),
                    UNIQUE (experiment_sha256, cell_sha256, p_index),
                    FOREIGN KEY (experiment_sha256, cell_sha256)
                        REFERENCES graph_cells(experiment_sha256, cell_sha256)
                        ON DELETE RESTRICT,
                    CHECK (expected_trajectories = graph_count * circuit_count),
                    CHECK (
                        (status = 'planned' AND artifact_sha256 IS NULL
                            AND logical_result_sha256 IS NULL
                            AND completed_trajectories = 0 AND completed_at_utc IS NULL)
                        OR
                        (status = 'complete' AND artifact_sha256 IS NOT NULL
                            AND logical_result_sha256 IS NOT NULL
                            AND completed_trajectories = expected_trajectories
                            AND completed_at_utc IS NOT NULL)
                    )
                ) STRICT, WITHOUT ROWID
                """
            )
            row = connection.execute(
                "SELECT catalog_kind, schema_version FROM catalog_metadata WHERE singleton = 1"
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO catalog_metadata(
                        singleton, catalog_kind, schema_version, created_at_utc
                    ) VALUES (1, 'single_ref_raw_tau', ?, ?)
                    """,
                    (CATALOG_SCHEMA_VERSION, _utc_now()),
                )
            elif tuple(row) != ("single_ref_raw_tau", CATALOG_SCHEMA_VERSION):
                raise RuntimeError(f"{self.path}: catalog metadata conflicts with this schema")
            connection.execute(f"PRAGMA application_id = {CATALOG_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {CATALOG_SCHEMA_VERSION}")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        quick_check = connection.execute("PRAGMA quick_check").fetchall()
        if quick_check != [("ok",)]:
            raise RuntimeError(f"{self.path}: SQLite quick_check failed: {quick_check!r}")
        foreign_key_check = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_check:
            raise RuntimeError(
                f"{self.path}: SQLite foreign_key_check failed: {foreign_key_check!r}"
            )

    @staticmethod
    def _insert_or_compare(
        connection: sqlite3.Connection,
        table: str,
        key_columns: tuple[str, ...],
        immutable: dict[str, object],
        insert_values: dict[str, object],
    ) -> None:
        where = " AND ".join(f"{column} = ?" for column in key_columns)
        key_values = tuple(immutable[column] for column in key_columns)
        columns = tuple(immutable)
        row = connection.execute(
            f"SELECT {', '.join(columns)} FROM {table} WHERE {where}",  # noqa: S608
            key_values,
        ).fetchone()
        if row is None:
            names = tuple(insert_values)
            placeholders = ", ".join("?" for _ in names)
            connection.execute(
                f"INSERT INTO {table}({', '.join(names)}) VALUES ({placeholders})",  # noqa: S608
                tuple(insert_values[name] for name in names),
            )
            return
        existing = dict(zip(columns, row, strict=True))
        if existing != immutable:
            raise ValueError(
                f"catalog {table} row {key_values!r} conflicts with the registered plan"
            )

    def register_plan(
        self,
        sweep: SingleReferenceSweepSpec,
        units: Iterable[tuple[ProviderCell, TauWorkUnit]],
        *,
        expected_cell_count: int,
    ) -> None:
        """Idempotently register one experiment and all canonical work units."""

        self._assert_owner()
        if (
            isinstance(expected_cell_count, bool)
            or not isinstance(expected_cell_count, int)
            or expected_cell_count < 1
        ):
            raise ValueError("expected_cell_count must be a positive integer")
        items = tuple(units)
        if not items:
            raise ValueError("a raw-tau plan must contain at least one work unit")
        experiment = {
            "experiment_sha256": sweep.specification_sha256,
            "experiment_id": sweep.experiment_id,
            "collection_sha256": sweep.graph_collection_sha256,
            "source_fingerprint_sha256": sweep.source_fingerprint_sha256,
            "environment_contract_sha256": sweep.environment_contract.specification_sha256,
            "specification_json": _canonical_json(sweep.canonical_payload()),
        }
        registered_at = _utc_now()
        connection = self._connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._insert_or_compare(
                connection,
                "experiments",
                ("experiment_sha256",),
                experiment,
                {**experiment, "registered_at_utc": registered_at},
            )
            seen_cells: set[str] = set()
            cell_rows: dict[str, dict[str, object]] = {}
            p_indices_by_cell: dict[str, set[int]] = {}
            seen_work: set[str] = set()
            for cell, work in items:
                if work.experiment_sha256 != sweep.specification_sha256:
                    raise ValueError("work unit belongs to a different experiment")
                if work.protocol != sweep.protocol:
                    raise ValueError("work unit protocol differs from the experiment protocol")
                if cell.spec != work.cell or cell.graphs_per_cell != work.graphs_per_cell:
                    raise ValueError("provider cell does not match its work unit")
                if cell.spec.collection_sha256 != sweep.graph_collection_sha256:
                    raise ValueError("provider cell belongs to a different graph collection")
                cell_row = {
                    "experiment_sha256": sweep.specification_sha256,
                    "cell_sha256": cell.cell_sha256,
                    "cell_index": cell.spec.cell_index,
                    "collection_id": cell.collection_id,
                    "collection_sha256": cell.spec.collection_sha256,
                    "graph_family": cell.graph_family,
                    "generator_name": cell.generator_name,
                    "generator_version": cell.generator_version,
                    "generator_contract_sha256": cell.generator_contract_sha256,
                    "n": cell.n,
                    "parameters_json": _canonical_json(cell.spec.parameters.canonical_payload()),
                    "graphs_per_cell": cell.graphs_per_cell,
                    "graph_index_start": 0,
                    "graph_index_stop_exclusive": cell.graphs_per_cell,
                }
                previous_cell_row = cell_rows.setdefault(cell.cell_sha256, cell_row)
                if previous_cell_row != cell_row:
                    raise ValueError("one cell hash was paired with conflicting provider metadata")
                if cell.cell_sha256 not in seen_cells:
                    self._insert_or_compare(
                        connection,
                        "graph_cells",
                        ("experiment_sha256", "cell_sha256"),
                        cell_row,
                        cell_row,
                    )
                    seen_cells.add(cell.cell_sha256)
                if work.work_sha256 in seen_work:
                    raise ValueError(f"duplicate work unit {work.work_sha256}")
                expected = work.graphs_per_cell * work.protocol.n_circuits
                relative_path = work.artifact_relative_path.as_posix()
                work_row = {
                    "experiment_sha256": sweep.specification_sha256,
                    "work_sha256": work.work_sha256,
                    "cell_sha256": cell.cell_sha256,
                    "p_index": work.p_index,
                    "p_decimal": work.p_decimal,
                    "graph_count": work.graphs_per_cell,
                    "circuit_count": work.protocol.n_circuits,
                    "expected_trajectories": expected,
                    "index_layout_json": _canonical_json(
                        {
                            "array_shape": [work.graphs_per_cell, work.protocol.n_circuits],
                            "circuit_index": "axis_1_zero_based",
                            "graph_index": "axis_0_zero_based",
                            "raw_arrays": [
                                "tau_p",
                                "stop_layer",
                                "event_observed",
                                "complete",
                                "reference_system_qubit",
                            ],
                        }
                    ),
                    "artifact_relative_path": relative_path,
                }
                self._insert_or_compare(
                    connection,
                    "work_units",
                    ("experiment_sha256", "work_sha256"),
                    work_row,
                    {
                        **work_row,
                        "status": "planned",
                        "artifact_sha256": None,
                        "logical_result_sha256": None,
                        "completed_trajectories": 0,
                        "registered_at_utc": registered_at,
                        "completed_at_utc": None,
                    },
                )
                seen_work.add(work.work_sha256)
                p_indices_by_cell.setdefault(cell.cell_sha256, set()).add(work.p_index)
            expected_p_indices = set(range(len(sweep.protocol.p_grid.canonical_values)))
            if any(indices != expected_p_indices for indices in p_indices_by_cell.values()):
                raise ValueError("every graph cell must contain the complete measurement-rate grid")
            cell_indices = sorted(int(row["cell_index"]) for row in cell_rows.values())
            if cell_indices != list(range(expected_cell_count)):
                raise ValueError(
                    "registered work units do not cover the complete zero-based graph-cell grid"
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def mark_complete(self, work: TauWorkUnit, progress: WorkUnitProgress) -> None:
        """Atomically mark a semantically validated final NPZ as complete."""

        self._assert_owner()
        if progress.work_sha256 != work.work_sha256:
            raise ValueError("worker result belongs to a different work unit")
        expected = work.graphs_per_cell * work.protocol.n_circuits
        if not progress.is_complete or progress.completed != expected:
            raise ValueError("only a complete work unit may be cataloged as terminal")
        expected_path = raw_tau_path(self.data_root, work)
        actual_path, relative_path = _resolve_inside(self.data_root, Path(progress.path))
        expected_resolved, expected_relative = _resolve_inside(self.data_root, expected_path)
        if actual_path != expected_resolved or relative_path != expected_relative:
            raise ValueError(
                f"worker returned artifact {actual_path}, expected {expected_resolved}"
            )
        payload = read_shared_bytes(actual_path)
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if progress.artifact_sha256 != actual_sha256:
            raise ValueError("worker artifact SHA-256 does not match the final artifact")
        with io.BytesIO(payload) as buffer, np.load(buffer, allow_pickle=False) as data:
            actual_logical_sha256 = logical_tau_digest(data)
        if progress.logical_result_sha256 != actual_logical_sha256:
            raise ValueError("worker logical-result SHA-256 does not match the final artifact")

        connection = self._connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT status, expected_trajectories, artifact_relative_path,
                       artifact_sha256, logical_result_sha256,
                       completed_trajectories
                FROM work_units
                WHERE experiment_sha256 = ? AND work_sha256 = ?
                """,
                (work.experiment_sha256, work.work_sha256),
            ).fetchone()
            if row is None:
                raise ValueError("work unit must be registered before completion")
            (
                status,
                expected_count,
                registered_path,
                old_sha256,
                old_logical_sha256,
                old_completed,
            ) = row
            if int(expected_count) != expected or str(registered_path) != expected_relative:
                raise ValueError("registered work-unit layout conflicts with the result")
            if status == "complete":
                if old_logical_sha256 != actual_logical_sha256 or int(old_completed) != expected:
                    raise ValueError("completed catalog row conflicts with the final artifact")
                if old_sha256 != actual_sha256:
                    # Container bytes are provenance, not scientific equality.
                    # A semantically validated repack with the same canonical
                    # logical digest updates only the container fingerprint.
                    connection.execute(
                        """
                        UPDATE work_units SET artifact_sha256 = ?
                        WHERE experiment_sha256 = ? AND work_sha256 = ?
                        """,
                        (actual_sha256, work.experiment_sha256, work.work_sha256),
                    )
            elif status == "planned":
                connection.execute(
                    """
                    UPDATE work_units
                    SET status = 'complete', artifact_sha256 = ?,
                        logical_result_sha256 = ?,
                        completed_trajectories = ?, completed_at_utc = ?
                    WHERE experiment_sha256 = ? AND work_sha256 = ?
                    """,
                    (
                        actual_sha256,
                        actual_logical_sha256,
                        expected,
                        _utc_now(),
                        work.experiment_sha256,
                        work.work_sha256,
                    ),
                )
            else:  # protected by the table CHECK; retained for corrupt databases
                raise RuntimeError(f"invalid catalog status {status!r}")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    def work_unit_record(self, work: TauWorkUnit) -> dict[str, object] | None:
        """Return one work-unit row for status tooling and tests."""

        self._assert_owner()
        cursor = self._connection.execute(
            """
            SELECT work_sha256, cell_sha256, p_index, p_decimal,
                   graph_count, circuit_count, expected_trajectories,
                   index_layout_json, artifact_relative_path, status,
                   artifact_sha256, logical_result_sha256,
                   completed_trajectories
            FROM work_units
            WHERE experiment_sha256 = ? AND work_sha256 = ?
            """,
            (work.experiment_sha256, work.work_sha256),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(zip((column[0] for column in cursor.description), row, strict=True))


__all__ = [
    "CATALOG_APPLICATION_ID",
    "CATALOG_SCHEMA_VERSION",
    "RawTauCatalog",
    "catalog_path",
]
