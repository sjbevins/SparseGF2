"""Versioned SQLite storage for graph-indexed production data.

The registry deliberately uses one database for every ``(n, beta)`` cell.  A
graph row has its own ``graph_id``; ``graph_seed`` is reproducibility metadata
and is *not* an identity key, so independently sampled rows may share a seed.

Writes must be owned by one coordinator process.  Simulation workers should
return records to that coordinator (or write immutable external artifacts) and
must not write this database concurrently.  WAL mode permits live readers, but
it does not turn SQLite into a multi-writer result store.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import json
import math
import re
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
APPLICATION_ID = 0x53474632  # "SGF2"
BETA_KEY_SCALE = 1_000_000_000
MAX_UINT64 = (1 << 64) - 1

_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_ENSEMBLE_STATUSES = frozenset({"planned", "active", "paused", "complete", "failed", "archived"})
_GRAPH_STATUSES = frozenset({"registered", "ready", "failed", "archived"})
_EXPERIMENT_STATUSES = frozenset({"planned", "running", "paused", "complete", "failed", "archived"})
_INVARIANT_RESULT_STATUSES = frozenset({"pending", "running", "complete", "failed"})
_EXPERIMENT_RESULT_STATUSES = frozenset(
    {"pending", "running", "complete", "censored", "failed", "skipped"}
)
_ARTIFACT_STATUSES = frozenset({"present", "missing", "corrupt", "archived"})
_VALUE_KINDS = frozenset({"integer", "real", "text", "json", "artifact"})
_IMMUTABLE_EXPERIMENT_RESULT_STATUSES = frozenset({"complete", "censored", "skipped"})


class RegistrySchemaError(RuntimeError):
    """Raised when a database is not a compatible graph registry."""


class RegistryConflictError(RuntimeError):
    """Raised when an idempotent identity maps to different immutable data."""


@dataclasses.dataclass(frozen=True, slots=True)
class GraphSeedRecord:
    """One independently sampled graph seed within an ``(n, beta)`` cell."""

    n: int
    beta: float
    beta_key: int
    graph_index: int
    graph_seed: int
    status: str = "registered"
    metadata: Mapping[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True)
class RegisteredGraph:
    """Stored graph identity returned in deterministic ``graph_index`` order."""

    graph_id: int
    ensemble_id: int
    n: int
    beta: float
    beta_key: int
    graph_index: int
    graph_seed: int
    status: str
    metadata: dict[str, object]


@dataclasses.dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Reference to an immutable external artifact and its content digest."""

    artifact_key: str
    uri: str
    sha256: str
    graph_id: int | None = None
    experiment_id: int | None = None
    invariant_id: int | None = None
    kind: str = "data"
    byte_size: int | None = None
    media_type: str | None = None
    status: str = "present"
    metadata: Mapping[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True)
class InvariantResultRecord:
    """A typed invariant result for one graph."""

    graph_id: int
    invariant_id: int
    value: object | None = None
    status: str = "complete"
    artifact_id: int | None = None
    error_message: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ExperimentResultRecord:
    """A graph-scoped result produced by one experiment."""

    graph_id: int
    experiment_id: int
    result: Mapping[str, object] | None = None
    status: str = "complete"
    artifact_id: int | None = None
    error_message: str | None = None


_MIGRATION_1 = (
    """
    CREATE TABLE schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE ensembles (
        ensemble_id INTEGER PRIMARY KEY,
        ensemble_key TEXT NOT NULL UNIQUE,
        graph_family TEXT NOT NULL,
        description TEXT,
        expected_graphs_per_cell INTEGER,
        status TEXT NOT NULL DEFAULT 'planned'
            CHECK (status IN ('planned','active','paused','complete','failed','archived')),
        metadata_json TEXT NOT NULL DEFAULT '{}'
            CHECK (json_valid(metadata_json) AND json_type(metadata_json) = 'object'),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        CHECK (length(trim(ensemble_key)) > 0),
        CHECK (length(trim(graph_family)) > 0),
        CHECK (expected_graphs_per_cell IS NULL OR expected_graphs_per_cell > 0)
    ) STRICT
    """,
    """
    CREATE TABLE graphs (
        graph_id INTEGER PRIMARY KEY,
        ensemble_id INTEGER NOT NULL,
        n INTEGER NOT NULL CHECK (n >= 3),
        beta REAL NOT NULL CHECK (beta >= 0.0 AND beta <= 1.0),
        beta_key INTEGER NOT NULL CHECK (beta_key >= 0 AND beta_key <= 1000000000),
        graph_index INTEGER NOT NULL CHECK (graph_index >= 0),
        graph_seed TEXT NOT NULL
            CHECK (length(graph_seed) BETWEEN 1 AND 20)
            CHECK (graph_seed NOT GLOB '*[^0-9]*')
            CHECK (graph_seed = '0' OR substr(graph_seed, 1, 1) != '0'),
        status TEXT NOT NULL DEFAULT 'registered'
            CHECK (status IN ('registered','ready','failed','archived')),
        metadata_json TEXT NOT NULL DEFAULT '{}'
            CHECK (json_valid(metadata_json) AND json_type(metadata_json) = 'object'),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (ensemble_id) REFERENCES ensembles(ensemble_id) ON DELETE RESTRICT,
        UNIQUE (ensemble_id, n, beta_key, graph_index),
        UNIQUE (graph_id, ensemble_id),
        CHECK (beta_key = CAST(round(beta * 1000000000.0) AS INTEGER))
    ) STRICT
    """,
    "CREATE INDEX idx_graphs_seed ON graphs (graph_seed)",
    "CREATE INDEX idx_graphs_cell_status ON graphs (ensemble_id, n, beta_key, status)",
    """
    CREATE TABLE experiments (
        experiment_id INTEGER PRIMARY KEY,
        ensemble_id INTEGER NOT NULL,
        experiment_key TEXT NOT NULL,
        kind TEXT NOT NULL,
        protocol_version TEXT NOT NULL,
        parameters_json TEXT NOT NULL DEFAULT '{}'
            CHECK (json_valid(parameters_json) AND json_type(parameters_json) = 'object'),
        status TEXT NOT NULL DEFAULT 'planned'
            CHECK (status IN ('planned','running','paused','complete','failed','archived')),
        description TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (ensemble_id) REFERENCES ensembles(ensemble_id) ON DELETE RESTRICT,
        UNIQUE (ensemble_id, experiment_key),
        UNIQUE (experiment_id, ensemble_id),
        CHECK (length(trim(experiment_key)) > 0),
        CHECK (length(trim(kind)) > 0),
        CHECK (length(trim(protocol_version)) > 0)
    ) STRICT
    """,
    "CREATE INDEX idx_experiments_status ON experiments (ensemble_id, status)",
    """
    CREATE TABLE invariant_definitions (
        invariant_id INTEGER PRIMARY KEY,
        invariant_key TEXT NOT NULL,
        definition_version TEXT NOT NULL,
        value_kind TEXT NOT NULL
            CHECK (value_kind IN ('integer','real','text','json','artifact')),
        units TEXT,
        description TEXT,
        parameters_json TEXT NOT NULL DEFAULT '{}'
            CHECK (json_valid(parameters_json) AND json_type(parameters_json) = 'object'),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (invariant_key, definition_version),
        CHECK (length(trim(invariant_key)) > 0),
        CHECK (length(trim(definition_version)) > 0)
    ) STRICT
    """,
    """
    CREATE TABLE artifact_references (
        artifact_id INTEGER PRIMARY KEY,
        ensemble_id INTEGER NOT NULL,
        artifact_key TEXT NOT NULL,
        graph_id INTEGER,
        experiment_id INTEGER,
        invariant_id INTEGER,
        kind TEXT NOT NULL,
        uri TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        byte_size INTEGER CHECK (byte_size IS NULL OR byte_size >= 0),
        media_type TEXT,
        status TEXT NOT NULL DEFAULT 'present'
            CHECK (status IN ('present','missing','corrupt','archived')),
        metadata_json TEXT NOT NULL DEFAULT '{}'
            CHECK (json_valid(metadata_json) AND json_type(metadata_json) = 'object'),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (ensemble_id) REFERENCES ensembles(ensemble_id) ON DELETE RESTRICT,
        FOREIGN KEY (graph_id, ensemble_id)
            REFERENCES graphs(graph_id, ensemble_id) ON DELETE RESTRICT,
        FOREIGN KEY (experiment_id, ensemble_id)
            REFERENCES experiments(experiment_id, ensemble_id) ON DELETE RESTRICT,
        FOREIGN KEY (invariant_id)
            REFERENCES invariant_definitions(invariant_id) ON DELETE RESTRICT,
        UNIQUE (ensemble_id, artifact_key),
        CHECK (length(trim(artifact_key)) > 0),
        CHECK (length(trim(kind)) > 0),
        CHECK (length(trim(uri)) > 0),
        CHECK (length(sha256) = 64),
        CHECK (sha256 = lower(sha256)),
        CHECK (sha256 NOT GLOB '*[^0-9a-f]*')
    ) STRICT
    """,
    "CREATE INDEX idx_artifacts_graph ON artifact_references (graph_id)",
    "CREATE INDEX idx_artifacts_experiment ON artifact_references (experiment_id)",
    "CREATE INDEX idx_artifacts_invariant ON artifact_references (invariant_id)",
    "CREATE INDEX idx_artifacts_status ON artifact_references (ensemble_id, status)",
    """
    CREATE TABLE invariant_results (
        graph_id INTEGER NOT NULL,
        invariant_id INTEGER NOT NULL,
        status TEXT NOT NULL
            CHECK (status IN ('pending','running','complete','failed')),
        value_json TEXT CHECK (value_json IS NULL OR json_valid(value_json)),
        artifact_id INTEGER,
        error_message TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT,
        PRIMARY KEY (graph_id, invariant_id),
        FOREIGN KEY (graph_id) REFERENCES graphs(graph_id) ON DELETE CASCADE,
        FOREIGN KEY (invariant_id)
            REFERENCES invariant_definitions(invariant_id) ON DELETE RESTRICT,
        FOREIGN KEY (artifact_id)
            REFERENCES artifact_references(artifact_id) ON DELETE RESTRICT,
        CHECK (status != 'complete' OR value_json IS NOT NULL OR artifact_id IS NOT NULL),
        CHECK (status != 'failed' OR length(trim(error_message)) > 0)
    ) STRICT
    """,
    "CREATE INDEX idx_invariant_results_definition ON invariant_results (invariant_id, status)",
    "CREATE INDEX idx_invariant_results_status ON invariant_results (status)",
    """
    CREATE TABLE graph_experiment_results (
        ensemble_id INTEGER NOT NULL,
        graph_id INTEGER NOT NULL,
        experiment_id INTEGER NOT NULL,
        status TEXT NOT NULL
            CHECK (status IN ('pending','running','complete','censored','failed','skipped')),
        result_json TEXT
            CHECK (result_json IS NULL OR
                   (json_valid(result_json) AND json_type(result_json) = 'object')),
        artifact_id INTEGER,
        error_message TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT,
        PRIMARY KEY (graph_id, experiment_id),
        FOREIGN KEY (graph_id, ensemble_id)
            REFERENCES graphs(graph_id, ensemble_id) ON DELETE CASCADE,
        FOREIGN KEY (experiment_id, ensemble_id)
            REFERENCES experiments(experiment_id, ensemble_id) ON DELETE RESTRICT,
        FOREIGN KEY (artifact_id)
            REFERENCES artifact_references(artifact_id) ON DELETE RESTRICT,
        CHECK (status NOT IN ('complete','censored') OR
               result_json IS NOT NULL OR artifact_id IS NOT NULL),
        CHECK (status != 'failed' OR length(trim(error_message)) > 0)
    ) STRICT
    """,
    """
    CREATE INDEX idx_experiment_results_experiment
        ON graph_experiment_results (experiment_id, status)
    """,
    "CREATE INDEX idx_experiment_results_graph ON graph_experiment_results (graph_id)",
)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _integer(value: object, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        upper = "" if maximum is None else f" and <= {maximum}"
        raise ValueError(f"{name} must be >= {minimum}{upper}")
    return value


def _optional_integer(
    value: object | None,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return _integer(value, name, minimum=minimum, maximum=maximum)


def _key(value: object, name: str) -> str:
    if not isinstance(value, str) or _KEY_RE.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be 1-128 ASCII letters, digits, '.', '_', ':', or '-', "
            "starting with a letter or digit"
        )
    return value


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be nonempty text without NUL characters")
    return value


def _optional_text(value: object | None, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_text(value, name)


def _status(value: object, allowed: frozenset[str], name: str = "status") -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}")
    return value


def _canonical_json(value: object, name: str, *, require_object: bool = False) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite, JSON-serializable data") from exc
    if require_object and not isinstance(decoded, dict):
        raise TypeError(f"{name} must be a mapping")
    return encoded


def _metadata(value: Mapping[str, object], name: str = "metadata") -> str:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return _canonical_json(dict(value), name, require_object=True)


class GraphRegistryDatabase:
    """Coordinator-owned, file-backed SQLite graph registry.

    Construction creates or migrates the database.  All write APIs use
    ``BEGIN IMMEDIATE`` and batch methods are all-or-nothing.  Repeating a
    registration returns the existing IDs when immutable fields agree; an
    identity collision with different data raises :class:`RegistryConflictError`.

    Do not share this object or writable connections with simulation workers.
    Use one coordinator writer and any number of short-lived readers.
    """

    def __init__(self, path: str | Path, *, busy_timeout_ms: int = 30_000) -> None:
        self.path = Path(path)
        if str(self.path) == ":memory:":
            raise ValueError("the graph registry must be one file-backed database")
        self.busy_timeout_ms = _integer(
            busy_timeout_ms,
            "busy_timeout_ms",
            minimum=1,
            maximum=600_000,
        )
        if self.path.exists() and self.path.is_dir():
            raise ValueError("database path points to a directory")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        if read_only:
            uri = self.path.resolve().as_uri() + "?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=self.busy_timeout_ms / 1000)
        else:
            connection = sqlite3.connect(self.path, timeout=self.busy_timeout_ms / 1000)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        if not read_only:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
        else:
            connection.execute("PRAGMA query_only = ON")
        return connection

    @contextlib.contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a short-lived read-only connection with foreign keys enabled."""

        connection = self._connect(read_only=True)
        try:
            yield connection
        finally:
            connection.close()

    @contextlib.contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the schema or verify/migrate a compatible registry."""

        connection = self._connect()
        try:
            application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            objects = connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%' AND type IN ('table','view','trigger')
                """
            ).fetchall()
            if application_id not in (0, APPLICATION_ID):
                raise RegistrySchemaError("database belongs to a different application")
            if version > SCHEMA_VERSION:
                raise RegistrySchemaError(
                    f"database schema v{version} is newer than supported v{SCHEMA_VERSION}"
                )
            if version == 0 and objects:
                raise RegistrySchemaError("refusing to overwrite a non-registry SQLite database")
            if version < 1:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    for statement in _MIGRATION_1:
                        connection.execute(statement)
                    applied_at = _now()
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (1, applied_at),
                    )
                    connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
                    connection.execute("PRAGMA user_version = 1")
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise
            recorded = connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[
                0
            ]
            final_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            final_application = int(connection.execute("PRAGMA application_id").fetchone()[0])
            if recorded != SCHEMA_VERSION or final_version != SCHEMA_VERSION:
                raise RegistrySchemaError("schema migration history is inconsistent")
            if final_application != APPLICATION_ID:
                raise RegistrySchemaError("graph-registry application ID is missing")
        finally:
            connection.close()

    @property
    def schema_version(self) -> int:
        """Return the on-disk schema version after validating the application ID."""

        with self.read_connection() as connection:
            application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if application_id != APPLICATION_ID:
            raise RegistrySchemaError("database is not a graph registry")
        return version

    def register_ensemble(
        self,
        ensemble_key: str,
        *,
        graph_family: str,
        expected_graphs_per_cell: int | None = None,
        description: str | None = None,
        metadata: Mapping[str, object] | None = None,
        status: str = "planned",
    ) -> int:
        """Create an ensemble or return its existing ID on an exact resume."""

        ensemble_key = _key(ensemble_key, "ensemble_key")
        graph_family = _nonempty_text(graph_family, "graph_family")
        expected = _optional_integer(
            expected_graphs_per_cell,
            "expected_graphs_per_cell",
            minimum=1,
        )
        description = _optional_text(description, "description")
        metadata_json = _metadata({} if metadata is None else metadata)
        status = _status(status, _ENSEMBLE_STATUSES)
        timestamp = _now()
        with self._write_transaction() as connection:
            connection.execute(
                """
                INSERT INTO ensembles(
                    ensemble_key, graph_family, description, expected_graphs_per_cell,
                    status, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ensemble_key) DO NOTHING
                """,
                (
                    ensemble_key,
                    graph_family,
                    description,
                    expected,
                    status,
                    metadata_json,
                    timestamp,
                    timestamp,
                ),
            )
            row = connection.execute(
                "SELECT * FROM ensembles WHERE ensemble_key = ?", (ensemble_key,)
            ).fetchone()
            assert row is not None
            expected_values = (graph_family, description, expected, metadata_json)
            actual_values = (
                row["graph_family"],
                row["description"],
                row["expected_graphs_per_cell"],
                row["metadata_json"],
            )
            if actual_values != expected_values:
                raise RegistryConflictError(
                    f"ensemble {ensemble_key!r} already exists with different immutable data"
                )
            return int(row["ensemble_id"])

    def set_ensemble_status(self, ensemble_id: int, status: str) -> None:
        ensemble_id = _integer(ensemble_id, "ensemble_id", minimum=1)
        status = _status(status, _ENSEMBLE_STATUSES)
        with self._write_transaction() as connection:
            cursor = connection.execute(
                "UPDATE ensembles SET status = ?, updated_at = ? WHERE ensemble_id = ?",
                (status, _now(), ensemble_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown ensemble_id={ensemble_id}")

    @staticmethod
    def _prepare_graph(record: GraphSeedRecord) -> tuple[object, ...]:
        if not isinstance(record, GraphSeedRecord):
            raise TypeError("graphs must contain GraphSeedRecord values")
        n = _integer(record.n, "n", minimum=3)
        if isinstance(record.beta, bool) or not isinstance(record.beta, (int, float)):
            raise TypeError("beta must be a real number")
        beta = float(record.beta)
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be finite and lie in [0, 1]")
        beta_key = _integer(record.beta_key, "beta_key", maximum=BETA_KEY_SCALE)
        expected_key = round(beta * BETA_KEY_SCALE)
        if beta_key != expected_key:
            raise ValueError(
                f"beta_key={beta_key} does not match round(beta * {BETA_KEY_SCALE})={expected_key}"
            )
        graph_index = _integer(record.graph_index, "graph_index")
        graph_seed = _integer(record.graph_seed, "graph_seed", maximum=MAX_UINT64)
        status = _status(record.status, _GRAPH_STATUSES)
        metadata_json = _metadata(record.metadata)
        return n, beta, beta_key, graph_index, str(graph_seed), status, metadata_json

    def register_graphs(
        self,
        ensemble_id: int,
        graphs: Iterable[GraphSeedRecord],
    ) -> list[int]:
        """Transactionally register graph rows and return IDs in input order.

        The unique identity is ``(ensemble_id, n, beta_key, graph_index)``.
        Equal seeds in different rows are intentionally accepted.
        """

        ensemble_id = _integer(ensemble_id, "ensemble_id", minimum=1)
        prepared = [self._prepare_graph(record) for record in graphs]
        if not prepared:
            return []
        timestamp = _now()
        graph_ids: list[int] = []
        with self._write_transaction() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM ensembles WHERE ensemble_id = ?", (ensemble_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(f"unknown ensemble_id={ensemble_id}")
            for n, beta, beta_key, graph_index, seed, status, metadata_json in prepared:
                connection.execute(
                    """
                    INSERT INTO graphs(
                        ensemble_id, n, beta, beta_key, graph_index, graph_seed,
                        status, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ensemble_id, n, beta_key, graph_index) DO NOTHING
                    """,
                    (
                        ensemble_id,
                        n,
                        beta,
                        beta_key,
                        graph_index,
                        seed,
                        status,
                        metadata_json,
                        timestamp,
                        timestamp,
                    ),
                )
                row = connection.execute(
                    """
                    SELECT * FROM graphs
                    WHERE ensemble_id = ? AND n = ? AND beta_key = ? AND graph_index = ?
                    """,
                    (ensemble_id, n, beta_key, graph_index),
                ).fetchone()
                assert row is not None
                immutable = (float(row["beta"]), row["graph_seed"], row["metadata_json"])
                if immutable != (beta, seed, metadata_json):
                    raise RegistryConflictError(
                        "graph identity already exists with different beta, seed, or metadata: "
                        f"ensemble_id={ensemble_id}, n={n}, beta_key={beta_key}, "
                        f"graph_index={graph_index}"
                    )
                graph_ids.append(int(row["graph_id"]))
        return graph_ids

    def graphs_for_cell(
        self, ensemble_id: int, n: int, beta_key: int
    ) -> tuple[RegisteredGraph, ...]:
        """Return a cell ordered by ``graph_index`` for direct vector indexing."""

        ensemble_id = _integer(ensemble_id, "ensemble_id", minimum=1)
        n = _integer(n, "n", minimum=3)
        beta_key = _integer(beta_key, "beta_key", maximum=BETA_KEY_SCALE)
        with self.read_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM graphs
                WHERE ensemble_id = ? AND n = ? AND beta_key = ?
                ORDER BY graph_index
                """,
                (ensemble_id, n, beta_key),
            ).fetchall()
        return tuple(
            RegisteredGraph(
                graph_id=int(row["graph_id"]),
                ensemble_id=int(row["ensemble_id"]),
                n=int(row["n"]),
                beta=float(row["beta"]),
                beta_key=int(row["beta_key"]),
                graph_index=int(row["graph_index"]),
                graph_seed=int(row["graph_seed"]),
                status=str(row["status"]),
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        )

    def set_graph_statuses(self, graph_ids: Iterable[int], status: str) -> None:
        ids = [_integer(value, "graph_id", minimum=1) for value in graph_ids]
        status = _status(status, _GRAPH_STATUSES)
        if not ids:
            return
        with self._write_transaction() as connection:
            timestamp = _now()
            for graph_id in ids:
                cursor = connection.execute(
                    "UPDATE graphs SET status = ?, updated_at = ? WHERE graph_id = ?",
                    (status, timestamp, graph_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"unknown graph_id={graph_id}")

    def register_experiment(
        self,
        ensemble_id: int,
        experiment_key: str,
        *,
        kind: str,
        protocol_version: str,
        parameters: Mapping[str, object] | None = None,
        description: str | None = None,
        status: str = "planned",
    ) -> int:
        """Create an experiment definition or resume an identical definition."""

        ensemble_id = _integer(ensemble_id, "ensemble_id", minimum=1)
        experiment_key = _key(experiment_key, "experiment_key")
        kind = _nonempty_text(kind, "kind")
        protocol_version = _nonempty_text(protocol_version, "protocol_version")
        parameters_json = _metadata({} if parameters is None else parameters, "parameters")
        description = _optional_text(description, "description")
        status = _status(status, _EXPERIMENT_STATUSES)
        timestamp = _now()
        with self._write_transaction() as connection:
            connection.execute(
                """
                INSERT INTO experiments(
                    ensemble_id, experiment_key, kind, protocol_version, parameters_json,
                    status, description, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ensemble_id, experiment_key) DO NOTHING
                """,
                (
                    ensemble_id,
                    experiment_key,
                    kind,
                    protocol_version,
                    parameters_json,
                    status,
                    description,
                    timestamp,
                    timestamp,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM experiments
                WHERE ensemble_id = ? AND experiment_key = ?
                """,
                (ensemble_id, experiment_key),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown ensemble_id={ensemble_id}")
            immutable = (
                row["kind"],
                row["protocol_version"],
                row["parameters_json"],
                row["description"],
            )
            if immutable != (kind, protocol_version, parameters_json, description):
                raise RegistryConflictError(
                    f"experiment {experiment_key!r} already exists with different immutable data"
                )
            return int(row["experiment_id"])

    def set_experiment_status(self, experiment_id: int, status: str) -> None:
        experiment_id = _integer(experiment_id, "experiment_id", minimum=1)
        status = _status(status, _EXPERIMENT_STATUSES)
        with self._write_transaction() as connection:
            cursor = connection.execute(
                "UPDATE experiments SET status = ?, updated_at = ? WHERE experiment_id = ?",
                (status, _now(), experiment_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown experiment_id={experiment_id}")

    def define_invariant(
        self,
        invariant_key: str,
        *,
        definition_version: str,
        value_kind: str,
        units: str | None = None,
        description: str | None = None,
        parameters: Mapping[str, object] | None = None,
    ) -> int:
        """Register a versioned invariant definition."""

        invariant_key = _key(invariant_key, "invariant_key")
        definition_version = _nonempty_text(definition_version, "definition_version")
        value_kind = _status(value_kind, _VALUE_KINDS, "value_kind")
        units = _optional_text(units, "units")
        description = _optional_text(description, "description")
        parameters_json = _metadata({} if parameters is None else parameters, "parameters")
        timestamp = _now()
        with self._write_transaction() as connection:
            connection.execute(
                """
                INSERT INTO invariant_definitions(
                    invariant_key, definition_version, value_kind, units, description,
                    parameters_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(invariant_key, definition_version) DO NOTHING
                """,
                (
                    invariant_key,
                    definition_version,
                    value_kind,
                    units,
                    description,
                    parameters_json,
                    timestamp,
                    timestamp,
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM invariant_definitions
                WHERE invariant_key = ? AND definition_version = ?
                """,
                (invariant_key, definition_version),
            ).fetchone()
            assert row is not None
            immutable = (
                row["value_kind"],
                row["units"],
                row["description"],
                row["parameters_json"],
            )
            if immutable != (value_kind, units, description, parameters_json):
                raise RegistryConflictError(
                    f"invariant {invariant_key!r} v{definition_version!r} already differs"
                )
            return int(row["invariant_id"])

    def register_artifacts(
        self,
        ensemble_id: int,
        artifacts: Iterable[ArtifactRecord],
    ) -> list[int]:
        """Transactionally register content-addressed external artifact references."""

        ensemble_id = _integer(ensemble_id, "ensemble_id", minimum=1)
        prepared: list[tuple[object, ...]] = []
        for record in artifacts:
            if not isinstance(record, ArtifactRecord):
                raise TypeError("artifacts must contain ArtifactRecord values")
            artifact_key = _key(record.artifact_key, "artifact_key")
            uri = _nonempty_text(record.uri, "uri")
            if not isinstance(record.sha256, str):
                raise TypeError("sha256 must be text")
            sha256 = record.sha256.lower()
            if _SHA256_RE.fullmatch(sha256) is None:
                raise ValueError("sha256 must contain exactly 64 hexadecimal digits")
            graph_id = _optional_integer(record.graph_id, "graph_id", minimum=1)
            experiment_id = _optional_integer(record.experiment_id, "experiment_id", minimum=1)
            invariant_id = _optional_integer(record.invariant_id, "invariant_id", minimum=1)
            kind = _nonempty_text(record.kind, "kind")
            byte_size = _optional_integer(record.byte_size, "byte_size")
            media_type = _optional_text(record.media_type, "media_type")
            status = _status(record.status, _ARTIFACT_STATUSES)
            metadata_json = _metadata(record.metadata)
            prepared.append(
                (
                    artifact_key,
                    graph_id,
                    experiment_id,
                    invariant_id,
                    kind,
                    uri,
                    sha256,
                    byte_size,
                    media_type,
                    status,
                    metadata_json,
                )
            )
        if not prepared:
            return []
        timestamp = _now()
        ids: list[int] = []
        with self._write_transaction() as connection:
            for values in prepared:
                connection.execute(
                    """
                    INSERT INTO artifact_references(
                        ensemble_id, artifact_key, graph_id, experiment_id, invariant_id,
                        kind, uri, sha256, byte_size, media_type, status, metadata_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ensemble_id, artifact_key) DO NOTHING
                    """,
                    (ensemble_id, *values, timestamp, timestamp),
                )
                row = connection.execute(
                    """
                    SELECT * FROM artifact_references
                    WHERE ensemble_id = ? AND artifact_key = ?
                    """,
                    (ensemble_id, values[0]),
                ).fetchone()
                if row is None:
                    raise KeyError("artifact references an unknown ensemble or scoped record")
                immutable = (
                    row["graph_id"],
                    row["experiment_id"],
                    row["invariant_id"],
                    row["kind"],
                    row["uri"],
                    row["sha256"],
                    row["byte_size"],
                    row["media_type"],
                    row["metadata_json"],
                )
                expected = (*values[1:9], values[10])
                if immutable != expected:
                    raise RegistryConflictError(
                        f"artifact {values[0]!r} already exists with different immutable data"
                    )
                ids.append(int(row["artifact_id"]))
        return ids

    @staticmethod
    def _validate_artifact_scope(
        connection: sqlite3.Connection,
        artifact_id: int | None,
        *,
        ensemble_id: int,
        graph_id: int,
        experiment_id: int | None = None,
        invariant_id: int | None = None,
    ) -> None:
        if artifact_id is None:
            return
        row = connection.execute(
            "SELECT * FROM artifact_references WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown artifact_id={artifact_id}")
        if int(row["ensemble_id"]) != ensemble_id:
            raise ValueError("artifact belongs to a different ensemble")
        if row["graph_id"] is not None and int(row["graph_id"]) != graph_id:
            raise ValueError("artifact belongs to a different graph")
        if (
            experiment_id is not None
            and row["experiment_id"] is not None
            and int(row["experiment_id"]) != experiment_id
        ):
            raise ValueError("artifact belongs to a different experiment")
        if (
            invariant_id is not None
            and row["invariant_id"] is not None
            and int(row["invariant_id"]) != invariant_id
        ):
            raise ValueError("artifact belongs to a different invariant")

    @staticmethod
    def _encode_invariant_value(value_kind: str, value: object | None) -> str | None:
        if value is None and value_kind != "json":
            return None
        if value_kind == "integer":
            value = _integer(value, "invariant value", minimum=-(1 << 63), maximum=(1 << 63) - 1)
        elif value_kind == "real":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("real invariant value must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("real invariant value must be finite")
        elif value_kind == "text":
            if not isinstance(value, str):
                raise TypeError("text invariant value must be text")
        elif value_kind == "artifact":
            if value is not None:
                raise ValueError("artifact invariant values use artifact_id, not an inline value")
            return None
        return _canonical_json(value, "invariant value")

    def upsert_invariant_results(self, records: Iterable[InvariantResultRecord]) -> None:
        """Transactionally insert progress or final graph-invariant results."""

        records = list(records)
        if not records:
            return
        with self._write_transaction() as connection:
            for record in records:
                if not isinstance(record, InvariantResultRecord):
                    raise TypeError("records must contain InvariantResultRecord values")
                graph_id = _integer(record.graph_id, "graph_id", minimum=1)
                invariant_id = _integer(record.invariant_id, "invariant_id", minimum=1)
                artifact_id = _optional_integer(record.artifact_id, "artifact_id", minimum=1)
                status = _status(record.status, _INVARIANT_RESULT_STATUSES)
                error = _optional_text(record.error_message, "error_message")
                if status == "failed" and error is None:
                    raise ValueError("failed invariant results require error_message")
                if status != "failed" and error is not None:
                    raise ValueError("error_message is only valid for failed invariant results")
                graph = connection.execute(
                    "SELECT ensemble_id FROM graphs WHERE graph_id = ?", (graph_id,)
                ).fetchone()
                definition = connection.execute(
                    "SELECT value_kind FROM invariant_definitions WHERE invariant_id = ?",
                    (invariant_id,),
                ).fetchone()
                if graph is None:
                    raise KeyError(f"unknown graph_id={graph_id}")
                if definition is None:
                    raise KeyError(f"unknown invariant_id={invariant_id}")
                value_json = self._encode_invariant_value(
                    str(definition["value_kind"]), record.value
                )
                if status == "complete" and value_json is None and artifact_id is None:
                    raise ValueError("complete invariant results require a value or artifact")
                self._validate_artifact_scope(
                    connection,
                    artifact_id,
                    ensemble_id=int(graph["ensemble_id"]),
                    graph_id=graph_id,
                    invariant_id=invariant_id,
                )
                timestamp = _now()
                completed_at = timestamp if status in {"complete", "failed"} else None
                existing = connection.execute(
                    """
                    SELECT * FROM invariant_results
                    WHERE graph_id = ? AND invariant_id = ?
                    """,
                    (graph_id, invariant_id),
                ).fetchone()
                values = (status, value_json, artifact_id, error, completed_at)
                if existing is not None:
                    old_values = (
                        existing["status"],
                        existing["value_json"],
                        existing["artifact_id"],
                        existing["error_message"],
                        existing["completed_at"],
                    )
                    same_payload = old_values[:4] == values[:4]
                    if same_payload:
                        continue
                    if existing["status"] == "complete":
                        raise RegistryConflictError(
                            "a complete invariant result is immutable unless the definition is versioned"
                        )
                    if existing["status"] == "running" and status == "pending":
                        raise RegistryConflictError(
                            "cannot move an invariant result back to pending"
                        )
                    connection.execute(
                        """
                        UPDATE invariant_results
                        SET status = ?, value_json = ?, artifact_id = ?, error_message = ?,
                            updated_at = ?, completed_at = ?
                        WHERE graph_id = ? AND invariant_id = ?
                        """,
                        (*values[:4], timestamp, completed_at, graph_id, invariant_id),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO invariant_results(
                            graph_id, invariant_id, status, value_json, artifact_id,
                            error_message, created_at, updated_at, completed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            graph_id,
                            invariant_id,
                            status,
                            value_json,
                            artifact_id,
                            error,
                            timestamp,
                            timestamp,
                            completed_at,
                        ),
                    )

    def upsert_experiment_results(self, records: Iterable[ExperimentResultRecord]) -> None:
        """Transactionally insert or advance graph-scoped experiment results."""

        records = list(records)
        if not records:
            return
        with self._write_transaction() as connection:
            for record in records:
                if not isinstance(record, ExperimentResultRecord):
                    raise TypeError("records must contain ExperimentResultRecord values")
                graph_id = _integer(record.graph_id, "graph_id", minimum=1)
                experiment_id = _integer(record.experiment_id, "experiment_id", minimum=1)
                artifact_id = _optional_integer(record.artifact_id, "artifact_id", minimum=1)
                status = _status(record.status, _EXPERIMENT_RESULT_STATUSES)
                error = _optional_text(record.error_message, "error_message")
                if status == "failed" and error is None:
                    raise ValueError("failed experiment results require error_message")
                if status != "failed" and error is not None:
                    raise ValueError("error_message is only valid for failed experiment results")
                result_json = None
                if record.result is not None:
                    if not isinstance(record.result, Mapping):
                        raise TypeError("experiment result must be a mapping")
                    result_json = _canonical_json(
                        dict(record.result), "experiment result", require_object=True
                    )
                if (
                    status in {"complete", "censored"}
                    and result_json is None
                    and artifact_id is None
                ):
                    raise ValueError(f"{status} experiment results require data or an artifact")
                row = connection.execute(
                    """
                    SELECT g.ensemble_id AS graph_ensemble, e.ensemble_id AS experiment_ensemble
                    FROM graphs AS g CROSS JOIN experiments AS e
                    WHERE g.graph_id = ? AND e.experiment_id = ?
                    """,
                    (graph_id, experiment_id),
                ).fetchone()
                if row is None:
                    graph_exists = connection.execute(
                        "SELECT 1 FROM graphs WHERE graph_id = ?", (graph_id,)
                    ).fetchone()
                    if graph_exists is None:
                        raise KeyError(f"unknown graph_id={graph_id}")
                    raise KeyError(f"unknown experiment_id={experiment_id}")
                ensemble_id = int(row["graph_ensemble"])
                if ensemble_id != int(row["experiment_ensemble"]):
                    raise ValueError("graph and experiment belong to different ensembles")
                self._validate_artifact_scope(
                    connection,
                    artifact_id,
                    ensemble_id=ensemble_id,
                    graph_id=graph_id,
                    experiment_id=experiment_id,
                )
                timestamp = _now()
                terminal = status in _IMMUTABLE_EXPERIMENT_RESULT_STATUSES or status == "failed"
                completed_at = timestamp if terminal else None
                existing = connection.execute(
                    """
                    SELECT * FROM graph_experiment_results
                    WHERE graph_id = ? AND experiment_id = ?
                    """,
                    (graph_id, experiment_id),
                ).fetchone()
                values = (status, result_json, artifact_id, error)
                if existing is not None:
                    old_values = (
                        existing["status"],
                        existing["result_json"],
                        existing["artifact_id"],
                        existing["error_message"],
                    )
                    if old_values == values:
                        continue
                    if existing["status"] in _IMMUTABLE_EXPERIMENT_RESULT_STATUSES:
                        raise RegistryConflictError(
                            "a terminal experiment result is immutable; create a new experiment version"
                        )
                    if existing["status"] == "running" and status == "pending":
                        raise RegistryConflictError(
                            "cannot move an experiment result back to pending"
                        )
                    connection.execute(
                        """
                        UPDATE graph_experiment_results
                        SET status = ?, result_json = ?, artifact_id = ?, error_message = ?,
                            updated_at = ?, completed_at = ?
                        WHERE graph_id = ? AND experiment_id = ?
                        """,
                        (*values, timestamp, completed_at, graph_id, experiment_id),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO graph_experiment_results(
                            ensemble_id, graph_id, experiment_id, status, result_json,
                            artifact_id, error_message, created_at, updated_at, completed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ensemble_id,
                            graph_id,
                            experiment_id,
                            status,
                            result_json,
                            artifact_id,
                            error,
                            timestamp,
                            timestamp,
                            completed_at,
                        ),
                    )

    def graph_snapshot(self, graph_id: int) -> dict[str, Any]:
        """Return the relational ``DATABASE`` view associated with one graph."""

        graph_id = _integer(graph_id, "graph_id", minimum=1)
        with self.read_connection() as connection:
            graph = connection.execute(
                "SELECT * FROM graphs WHERE graph_id = ?", (graph_id,)
            ).fetchone()
            if graph is None:
                raise KeyError(f"unknown graph_id={graph_id}")
            invariants = connection.execute(
                """
                SELECT r.*, d.invariant_key, d.definition_version, d.value_kind
                FROM invariant_results AS r
                JOIN invariant_definitions AS d USING (invariant_id)
                WHERE r.graph_id = ? ORDER BY d.invariant_key, d.definition_version
                """,
                (graph_id,),
            ).fetchall()
            results = connection.execute(
                """
                SELECT r.*, e.experiment_key, e.protocol_version
                FROM graph_experiment_results AS r
                JOIN experiments AS e USING (experiment_id)
                WHERE r.graph_id = ? ORDER BY e.experiment_key
                """,
                (graph_id,),
            ).fetchall()
            artifacts = connection.execute(
                "SELECT * FROM artifact_references WHERE graph_id = ? ORDER BY artifact_key",
                (graph_id,),
            ).fetchall()

        graph_data = dict(graph)
        graph_data["graph_seed"] = int(graph_data["graph_seed"])
        graph_data["metadata"] = json.loads(graph_data.pop("metadata_json"))
        invariant_data = [dict(row) for row in invariants]
        for item in invariant_data:
            item["value"] = (
                None if item["value_json"] is None else json.loads(item.pop("value_json"))
            )
            item.pop("value_json", None)
        result_data = [dict(row) for row in results]
        for item in result_data:
            item["result"] = (
                None if item["result_json"] is None else json.loads(item.pop("result_json"))
            )
            item.pop("result_json", None)
        artifact_data = [dict(row) for row in artifacts]
        for item in artifact_data:
            item["metadata"] = json.loads(item.pop("metadata_json"))
        return {
            "graph": graph_data,
            "invariants": invariant_data,
            "experiments": result_data,
            "artifacts": artifact_data,
        }
