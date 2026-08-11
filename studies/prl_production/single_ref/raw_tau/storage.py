"""Checksummed raw-tau shards and deterministic interrupted-run resume."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from studies.prl_production.single_ref.shared_io import load_npz_snapshot
from studies.prl_production.sweep_spec import (
    RAW_TAU_SCHEMA_VERSION,
    TAU_CENSORED,
    TAU_INCOMPLETE,
    SingleReferenceSweepSpec,
    TauWorkUnit,
)

from .engine import ENGINE_VERSION, simulate_trajectory
from .io import file_sha256, write_deterministic_npz
from .providers import CellEdgeBank, ProviderCell, load_edge_bank

JOURNAL_SCHEMA_VERSION = 1
_JOURNAL_REMOVE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)
_RAW_ARRAY_DTYPES = {
    "graph_index": ("i", 4),
    "graph_seed": ("i", 8),
    "circuit_index": ("i", 4),
    "tau_p": ("i", 4),
    "stop_layer": ("i", 4),
    "event_observed": ("u", 1),
    "complete": ("u", 1),
    "reference_system_qubit": ("i", 4),
}


@dataclass(frozen=True, slots=True)
class WorkUnitProgress:
    """Completed-count summary returned by one exclusive shard writer."""

    path: str
    work_sha256: str
    completed: int
    total: int
    events: int
    censored: int
    newly_completed: int
    elapsed_s: float
    artifact_sha256: str
    logical_result_sha256: str

    @property
    def is_complete(self) -> bool:
        return self.completed == self.total


def raw_tau_path(data_root: str | Path, work: TauWorkUnit) -> Path:
    """Return the deterministic path for one ``(cell, p)`` raw shard."""
    return Path(data_root).joinpath(*work.artifact_relative_path.parts)


def checkpoint_journal_path(data_root: str | Path, work: TauWorkUnit) -> Path:
    """Return the exclusive SQLite checkpoint journal for one raw shard."""
    path = raw_tau_path(data_root, work)
    return path.with_name(f"{path.name}.checkpoint.sqlite3")


def _parameters_json(cell: ProviderCell) -> str:
    return json.dumps(cell.parameters, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _environment_json(sweep: SingleReferenceSweepSpec) -> str:
    return json.dumps(
        sweep.environment_contract.canonical_payload(),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _new_arrays(
    sweep: SingleReferenceSweepSpec,
    work: TauWorkUnit,
    cell: ProviderCell,
    bank: CellEdgeBank,
) -> dict[str, object]:
    shape = work.raw_shape
    return {
        "schema_version": np.int32(RAW_TAU_SCHEMA_VERSION),
        "engine_version": np.str_(ENGINE_VERSION),
        "experiment_id": np.str_(sweep.experiment_id),
        "experiment_sha256": np.str_(sweep.specification_sha256),
        "source_fingerprint_sha256": np.str_(sweep.source_fingerprint_sha256),
        "environment_contract_json": np.str_(_environment_json(sweep)),
        "environment_contract_sha256": np.str_(sweep.environment_contract.specification_sha256),
        "protocol_sha256": np.str_(work.protocol.specification_sha256),
        "collection_id": np.str_(cell.collection_id),
        "collection_sha256": np.str_(cell.spec.collection_sha256),
        "cell_sha256": np.str_(cell.cell_sha256),
        "work_sha256": np.str_(work.work_sha256),
        "n": np.int32(cell.n),
        "parameters_json": np.str_(_parameters_json(cell)),
        "p_index": np.int32(work.p_index),
        "p_decimal": np.str_(work.p_decimal),
        "p": np.float64(float(work.p_decimal)),
        "n_graphs": np.int32(work.graphs_per_cell),
        "n_circuits": np.int32(work.protocol.n_circuits),
        "q_scramble": np.int32(work.protocol.q_scramble),
        "q_max": np.int32(work.protocol.q_max),
        "t_max": np.int32(work.protocol.t_max(cell.n)),
        "reference_system_qubit_policy": np.str_(work.protocol.reference_system_qubit_policy),
        "p_randomness_policy": np.str_(work.protocol.p_randomness_policy),
        "graph_index": np.arange(work.graphs_per_cell, dtype=np.int32),
        "graph_seed": np.asarray(bank.graph_seed, dtype=np.int64),
        "circuit_index": np.arange(work.protocol.n_circuits, dtype=np.int32),
        "edge_bank_sha256": np.str_(bank.artifact_sha256),
        "tau_p": np.full(shape, TAU_INCOMPLETE, dtype=np.int32),
        "stop_layer": np.zeros(shape, dtype=np.int32),
        "event_observed": np.zeros(shape, dtype=np.uint8),
        "complete": np.zeros(shape, dtype=np.uint8),
        "reference_system_qubit": np.full(shape, -1, dtype=np.int32),
    }


def _scalar(arrays: dict[str, object], key: str):
    value = np.asarray(arrays[key])
    if value.shape != ():
        raise ValueError(f"{key} must be scalar; got {value.shape}")
    return value.item()


def _raw_array(arrays: dict[str, object], key: str) -> np.ndarray:
    array = np.asarray(arrays[key])
    kind, itemsize = _RAW_ARRAY_DTYPES[key]
    if array.dtype.kind != kind or array.dtype.itemsize != itemsize:
        raise ValueError(f"{key} must have {kind}{itemsize} integer storage; got {array.dtype}")
    return array


def _journal_metadata(
    sweep: SingleReferenceSweepSpec,
    work: TauWorkUnit,
    cell: ProviderCell,
    bank: CellEdgeBank,
) -> str:
    payload = {
        "journal_schema_version": JOURNAL_SCHEMA_VERSION,
        "raw_tau_schema_version": RAW_TAU_SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "experiment_id": sweep.experiment_id,
        "experiment_sha256": sweep.specification_sha256,
        "source_fingerprint_sha256": sweep.source_fingerprint_sha256,
        "environment_contract_json": _environment_json(sweep),
        "environment_contract_sha256": sweep.environment_contract.specification_sha256,
        "protocol_sha256": work.protocol.specification_sha256,
        "collection_id": cell.collection_id,
        "collection_sha256": cell.spec.collection_sha256,
        "cell_sha256": cell.cell_sha256,
        "work_sha256": work.work_sha256,
        "n": cell.n,
        "parameters_json": _parameters_json(cell),
        "p_index": work.p_index,
        "p_decimal": work.p_decimal,
        "n_graphs": work.graphs_per_cell,
        "n_circuits": work.protocol.n_circuits,
        "q_scramble": work.protocol.q_scramble,
        "q_max": work.protocol.q_max,
        "t_max": work.protocol.t_max(cell.n),
        "reference_system_qubit_policy": work.protocol.reference_system_qubit_policy,
        "p_randomness_policy": work.protocol.p_randomness_policy,
        "edge_bank_sha256": bank.artifact_sha256,
    }
    return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _open_journal(path: Path, expected_metadata: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    try:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                singleton INTEGER NOT NULL PRIMARY KEY CHECK (singleton = 1),
                metadata_json TEXT NOT NULL
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                graph_index INTEGER NOT NULL CHECK (graph_index >= 0),
                circuit_index INTEGER NOT NULL CHECK (circuit_index >= 0),
                tau_p INTEGER NOT NULL,
                stop_layer INTEGER NOT NULL CHECK (stop_layer >= 1),
                event_observed INTEGER NOT NULL CHECK (event_observed IN (0, 1)),
                reference_system_qubit INTEGER NOT NULL CHECK (reference_system_qubit >= 0),
                PRIMARY KEY (graph_index, circuit_index)
            ) WITHOUT ROWID
            """
        )
        row = connection.execute(
            "SELECT metadata_json FROM metadata WHERE singleton = 1"
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO metadata(singleton, metadata_json) VALUES (1, ?)",
                (expected_metadata,),
            )
        elif str(row[0]) != expected_metadata:
            raise ValueError(f"{path}: checkpoint journal identity does not match this work unit")
        if connection.execute("SELECT COUNT(*) FROM metadata").fetchone()[0] != 1:
            raise ValueError(f"{path}: checkpoint journal has invalid metadata cardinality")
        connection.execute(f"PRAGMA user_version = {JOURNAL_SCHEMA_VERSION}")
        connection.commit()
        quick_check = connection.execute("PRAGMA quick_check").fetchall()
        if quick_check != [("ok",)]:
            raise ValueError(f"{path}: SQLite quick_check failed: {quick_check!r}")
        return connection
    except Exception:
        connection.rollback()
        connection.close()
        raise


def _apply_journal_row(
    path: Path,
    arrays: dict[str, object],
    row: tuple[object, ...],
    work: TauWorkUnit,
    cell: ProviderCell,
) -> None:
    graph_index, circuit_index, tau_p, stop_layer, event_observed, reference = (
        int(value) for value in row
    )
    if not 0 <= graph_index < work.graphs_per_cell:
        raise ValueError(f"{path}: journal graph_index={graph_index} is out of range")
    if not 0 <= circuit_index < work.protocol.n_circuits:
        raise ValueError(f"{path}: journal circuit_index={circuit_index} is out of range")
    if not 0 <= reference < cell.n:
        raise ValueError(f"{path}: journal reference site {reference} is out of range")
    cap = work.protocol.t_max(cell.n)
    if event_observed == 1:
        if tau_p != stop_layer or not 1 <= tau_p <= cap:
            raise ValueError(f"{path}: inconsistent observed journal row")
    elif event_observed == 0:
        if tau_p != TAU_CENSORED or stop_layer != cap:
            raise ValueError(f"{path}: inconsistent censored journal row")
    else:
        raise ValueError(f"{path}: journal event flag must be binary")
    complete = np.asarray(arrays["complete"], dtype=np.uint8)
    if complete[graph_index, circuit_index] != 0:
        raise ValueError(f"{path}: duplicate checkpoint result row")
    np.asarray(arrays["tau_p"])[graph_index, circuit_index] = tau_p
    np.asarray(arrays["stop_layer"])[graph_index, circuit_index] = stop_layer
    np.asarray(arrays["event_observed"])[graph_index, circuit_index] = event_observed
    np.asarray(arrays["reference_system_qubit"])[graph_index, circuit_index] = reference
    complete[graph_index, circuit_index] = 1


def _replay_journal(
    path: Path,
    connection: sqlite3.Connection,
    arrays: dict[str, object],
    work: TauWorkUnit,
    cell: ProviderCell,
) -> None:
    rows = connection.execute(
        """
        SELECT graph_index, circuit_index, tau_p, stop_layer,
               event_observed, reference_system_qubit
        FROM results ORDER BY graph_index, circuit_index
        """
    )
    for row in rows:
        _apply_journal_row(path, arrays, row, work, cell)


def _commit_journal_rows(
    path: Path,
    connection: sqlite3.Connection,
    rows: list[tuple[int, int, int, int, int, int]],
) -> None:
    if not rows:
        return
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany(
            """
            INSERT INTO results(
                graph_index, circuit_index, tau_p, stop_layer,
                event_observed, reference_system_qubit
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
    except Exception as exc:
        connection.rollback()
        raise RuntimeError(f"{path}: could not commit checkpoint batch") from exc
    rows.clear()


def _remove_journal(path: Path) -> None:
    for delay in (*_JOURNAL_REMOVE_RETRY_DELAYS, None):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def _validate_arrays(
    path: Path,
    arrays: dict[str, object],
    sweep: SingleReferenceSweepSpec,
    work: TauWorkUnit,
    cell: ProviderCell,
    bank: CellEdgeBank,
) -> None:
    expected_scalars = {
        "schema_version": RAW_TAU_SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "experiment_id": sweep.experiment_id,
        "experiment_sha256": sweep.specification_sha256,
        "source_fingerprint_sha256": sweep.source_fingerprint_sha256,
        "environment_contract_json": _environment_json(sweep),
        "environment_contract_sha256": sweep.environment_contract.specification_sha256,
        "protocol_sha256": work.protocol.specification_sha256,
        "collection_id": cell.collection_id,
        "collection_sha256": cell.spec.collection_sha256,
        "cell_sha256": cell.cell_sha256,
        "work_sha256": work.work_sha256,
        "n": cell.n,
        "parameters_json": _parameters_json(cell),
        "p_index": work.p_index,
        "p_decimal": work.p_decimal,
        "n_graphs": work.graphs_per_cell,
        "n_circuits": work.protocol.n_circuits,
        "q_scramble": work.protocol.q_scramble,
        "q_max": work.protocol.q_max,
        "t_max": work.protocol.t_max(cell.n),
        "reference_system_qubit_policy": work.protocol.reference_system_qubit_policy,
        "p_randomness_policy": work.protocol.p_randomness_policy,
        "edge_bank_sha256": bank.artifact_sha256,
    }
    for key, expected in expected_scalars.items():
        if key not in arrays:
            raise ValueError(f"{path}: missing {key}")
        actual = _scalar(arrays, key)
        if actual != expected:
            raise ValueError(f"{path}: {key}={actual!r}, expected {expected!r}")
    if not np.isclose(float(_scalar(arrays, "p")), float(work.p_decimal), rtol=0.0, atol=0.0):
        raise ValueError(f"{path}: floating p metadata mismatch")

    shape = work.raw_shape
    graph_index = _raw_array(arrays, "graph_index")
    graph_seed = _raw_array(arrays, "graph_seed")
    circuit_index = _raw_array(arrays, "circuit_index")
    if not np.array_equal(graph_index, np.arange(shape[0], dtype=np.int32)):
        raise ValueError(f"{path}: graph_index is not canonical")
    if not np.array_equal(graph_seed, bank.graph_seed):
        raise ValueError(f"{path}: graph seeds do not match the edge bank")
    if not np.array_equal(circuit_index, np.arange(shape[1], dtype=np.int32)):
        raise ValueError(f"{path}: circuit_index is not canonical")

    for key in ("tau_p", "stop_layer", "event_observed", "complete", "reference_system_qubit"):
        if np.asarray(arrays[key]).shape != shape:
            raise ValueError(f"{path}: {key} has invalid shape {np.asarray(arrays[key]).shape}")
    tau = _raw_array(arrays, "tau_p")
    stop = _raw_array(arrays, "stop_layer")
    event = _raw_array(arrays, "event_observed")
    complete = _raw_array(arrays, "complete")
    reference = _raw_array(arrays, "reference_system_qubit")
    if np.any((event != 0) & (event != 1)) or np.any((complete != 0) & (complete != 1)):
        raise ValueError(f"{path}: event and completion flags must be binary")
    done = complete.astype(bool)
    observed = done & event.astype(bool)
    censored = done & ~event.astype(bool)
    pending = ~done
    cap = work.protocol.t_max(cell.n)
    if (
        np.any(tau[observed] != stop[observed])
        or np.any(tau[observed] < 1)
        or np.any(tau[observed] > cap)
    ):
        raise ValueError(f"{path}: observed first-passage rows are inconsistent")
    if np.any(tau[censored] != TAU_CENSORED) or np.any(stop[censored] != cap):
        raise ValueError(f"{path}: censored rows are inconsistent")
    if (
        np.any(tau[pending] != TAU_INCOMPLETE)
        or np.any(stop[pending] != 0)
        or np.any(event[pending] != 0)
        or np.any(reference[pending] != -1)
    ):
        raise ValueError(f"{path}: incomplete rows are not pristine")
    if np.any(reference[done] < 0) or np.any(reference[done] >= cell.n):
        raise ValueError(f"{path}: completed rows have invalid Bell-pair sites")


def load_raw_tau_arrays(
    path: str | Path,
    sweep: SingleReferenceSweepSpec,
    work: TauWorkUnit,
    cell: ProviderCell,
    bank: CellEdgeBank,
) -> dict[str, object]:
    """Load one raw shard into memory after complete semantic validation."""
    resolved = Path(path)
    with load_npz_snapshot(resolved) as data:
        arrays = {key: np.array(data[key], copy=True) for key in data.files}
    _validate_arrays(resolved, arrays, sweep, work, cell, bank)
    return arrays


def run_work_unit(
    data_root: str | Path,
    sweep: SingleReferenceSweepSpec,
    work: TauWorkUnit,
    cell: ProviderCell,
    edge_bank_file: str | Path,
    *,
    checkpoint_every: int = 25,
    max_new_trajectories: int | None = None,
    execution: str = "batch",
    use_numba: bool | None = None,
    hybrid: bool = True,
) -> WorkUnitProgress:
    """Run or resume one exclusive ``(cell, p)`` artifact writer."""
    if not isinstance(sweep, SingleReferenceSweepSpec):
        raise TypeError("sweep must be a SingleReferenceSweepSpec")
    if not isinstance(work, TauWorkUnit):
        raise TypeError("work must be a TauWorkUnit")
    if sweep.specification_sha256 != work.experiment_sha256:
        raise ValueError("work unit belongs to a different experiment")
    if cell.spec != work.cell or cell.graphs_per_cell != work.graphs_per_cell:
        raise ValueError("provider cell does not match the work unit")
    if (
        isinstance(checkpoint_every, bool)
        or not isinstance(checkpoint_every, int)
        or checkpoint_every < 1
    ):
        raise ValueError("checkpoint_every must be a positive integer")
    if max_new_trajectories is not None and (
        isinstance(max_new_trajectories, bool)
        or not isinstance(max_new_trajectories, int)
        or max_new_trajectories < 1
    ):
        raise ValueError("max_new_trajectories must be positive when supplied")

    started = time.perf_counter()
    bank = load_edge_bank(edge_bank_file, cell)
    path = raw_tau_path(data_root, work)
    journal_path = checkpoint_journal_path(data_root, work)
    expected_metadata = _journal_metadata(sweep, work, cell, bank)

    if path.exists():
        arrays = load_raw_tau_arrays(path, sweep, work, cell, bank)
        complete = np.asarray(arrays["complete"], dtype=np.uint8)
        if not np.all(complete == 1):
            raise ValueError(f"{path}: published raw shard is incomplete")
        if journal_path.exists():
            replayed = _new_arrays(sweep, work, cell, bank)
            connection = _open_journal(journal_path, expected_metadata)
            try:
                _replay_journal(journal_path, connection, replayed, work, cell)
            finally:
                connection.close()
            _validate_arrays(journal_path, replayed, sweep, work, cell, bank)
            if not np.all(np.asarray(replayed["complete"], dtype=np.uint8) == 1):
                raise ValueError(f"{journal_path}: journal is incomplete beside a final shard")
            for key in arrays:
                if not np.array_equal(np.asarray(arrays[key]), np.asarray(replayed[key])):
                    raise ValueError(
                        f"{journal_path}: journal field {key} disagrees with the final shard"
                    )
            _remove_journal(journal_path)
        event = np.asarray(arrays["event_observed"], dtype=np.uint8)
        completed = int(complete.sum())
        events = int(event.sum())
        return WorkUnitProgress(
            path=str(path),
            work_sha256=work.work_sha256,
            completed=completed,
            total=work.raw_shape[0] * work.raw_shape[1],
            events=events,
            censored=completed - events,
            newly_completed=0,
            elapsed_s=time.perf_counter() - started,
            artifact_sha256=file_sha256(path),
            logical_result_sha256=logical_tau_digest(arrays),
        )

    arrays = _new_arrays(sweep, work, cell, bank)
    connection = _open_journal(journal_path, expected_metadata)
    try:
        _replay_journal(journal_path, connection, arrays, work, cell)
        _validate_arrays(journal_path, arrays, sweep, work, cell, bank)
    except BaseException:
        connection.close()
        raise
    complete = np.asarray(arrays["complete"], dtype=np.uint8)
    pending = np.argwhere(complete == 0)
    if max_new_trajectories is not None:
        pending = pending[:max_new_trajectories]

    newly_completed = 0
    checkpoint_rows: list[tuple[int, int, int, int, int, int]] = []
    try:
        for graph_value, circuit_value in pending:
            graph_index, circuit_index = int(graph_value), int(circuit_value)
            result = simulate_trajectory(
                work,
                graph_index,
                circuit_index,
                bank.graph_edges(graph_index),
                execution=execution,
                use_numba=use_numba,
                hybrid=hybrid,
            )
            tau_p = TAU_CENSORED if result.tau_p is None else result.tau_p
            event_observed = int(result.event_observed)
            np.asarray(arrays["tau_p"])[graph_index, circuit_index] = tau_p
            np.asarray(arrays["stop_layer"])[graph_index, circuit_index] = result.stop_layer
            np.asarray(arrays["event_observed"])[graph_index, circuit_index] = event_observed
            np.asarray(arrays["reference_system_qubit"])[graph_index, circuit_index] = (
                result.reference_system_qubit
            )
            # Completion is the in-memory commit marker and is always set last.
            complete[graph_index, circuit_index] = 1
            checkpoint_rows.append(
                (
                    graph_index,
                    circuit_index,
                    tau_p,
                    result.stop_layer,
                    event_observed,
                    result.reference_system_qubit,
                )
            )
            newly_completed += 1
            if len(checkpoint_rows) == checkpoint_every:
                _validate_arrays(journal_path, arrays, sweep, work, cell, bank)
                _commit_journal_rows(journal_path, connection, checkpoint_rows)
        _validate_arrays(journal_path, arrays, sweep, work, cell, bank)
        _commit_journal_rows(journal_path, connection, checkpoint_rows)
    except BaseException:
        connection.close()
        raise
    connection.close()

    event = np.asarray(arrays["event_observed"], dtype=np.uint8)
    completed = int(complete.sum())
    events = int(event[complete == 1].sum())
    censored = completed - events
    total = work.raw_shape[0] * work.raw_shape[1]
    if completed == total:
        write_deterministic_npz(path, arrays)
        published = load_raw_tau_arrays(path, sweep, work, cell, bank)
        for key in arrays:
            if not np.array_equal(np.asarray(arrays[key]), np.asarray(published[key])):
                raise ValueError(f"{path}: published field {key} failed round-trip validation")
        _remove_journal(journal_path)
        artifact_path = path
    else:
        artifact_path = journal_path
    return WorkUnitProgress(
        path=str(artifact_path),
        work_sha256=work.work_sha256,
        completed=completed,
        total=total,
        events=events,
        censored=censored,
        newly_completed=newly_completed,
        elapsed_s=time.perf_counter() - started,
        artifact_sha256=file_sha256(artifact_path),
        logical_result_sha256=logical_tau_digest(arrays),
    )


def logical_tau_digest(arrays: dict[str, object]) -> str:
    """Hash canonical raw results independently of ZIP and host byte order."""

    graph_index = _raw_array(arrays, "graph_index")
    graph_seed = _raw_array(arrays, "graph_seed")
    circuit_index = _raw_array(arrays, "circuit_index")
    result_shape = (graph_index.size, circuit_index.size)
    if graph_index.ndim != 1 or graph_seed.shape != graph_index.shape:
        raise ValueError("logical-result graph identity arrays have inconsistent shapes")
    if circuit_index.ndim != 1:
        raise ValueError("logical-result circuit_index must be one-dimensional")
    for key in (
        "tau_p",
        "stop_layer",
        "event_observed",
        "complete",
        "reference_system_qubit",
    ):
        if _raw_array(arrays, key).shape != result_shape:
            raise ValueError(f"logical-result {key} has an inconsistent shape")
    for key in ("event_observed", "complete"):
        flag = _raw_array(arrays, key)
        if np.any((flag != 0) & (flag != 1)):
            raise ValueError(f"logical-result {key} must be binary")

    digest = hashlib.sha256(b"single_ref_raw_tau_logical_result_v1\0")
    for key in ("cell_sha256", "p_decimal"):
        value = str(np.asarray(arrays[key]).item()).encode("utf-8")
        digest.update(len(key).to_bytes(2, "big"))
        digest.update(key.encode("ascii"))
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    dtypes = {
        "graph_index": "<i4",
        "graph_seed": "<i8",
        "circuit_index": "<i4",
        "tau_p": "<i4",
        "stop_layer": "<i4",
        "event_observed": "u1",
        "complete": "u1",
        "reference_system_qubit": "<i4",
    }
    for key, dtype in dtypes.items():
        raw = _raw_array(arrays, key)
        array = np.ascontiguousarray(raw.astype(np.dtype(dtype), copy=False))
        digest.update(len(key).to_bytes(2, "big"))
        digest.update(key.encode("ascii"))
        digest.update(array.ndim.to_bytes(2, "big"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        digest.update(array.tobytes())
    return digest.hexdigest()


__all__ = [
    "JOURNAL_SCHEMA_VERSION",
    "WorkUnitProgress",
    "checkpoint_journal_path",
    "load_raw_tau_arrays",
    "logical_tau_digest",
    "raw_tau_path",
    "run_work_unit",
]
