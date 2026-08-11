"""Compute, store, summarize, and plot exact rewiring counts for a graph collection."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import io
import json
import math
import os
import platform
import sqlite3
import struct
import sys
import uuid
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sparsegf2.circuits import graphs as graph_module

from .build import (
    default_database_path,
    validate_existing_collection,
    write_reports,
)
from .collection import GraphCollection
from .database import ArtifactRecord, InvariantResultRecord
from .plot_rewiring import SUMMARY_FIELDS, RewiringPlotPaths, plot_rewiring_summary
from .plot_rewiring_histograms import (
    RewiringHistogramPlotPaths,
    plot_rewiring_histograms,
)
from .rewiring_metrics import watts_strogatz_rewiring_counts
from .spec import GraphCollectionSpec, beta_from_key, production_spec, smoke_spec

CELL_SCHEMA_VERSION = 1
INVARIANT_KEY = "ws.realized_rewired_edge_count"
INVARIANT_VERSION = "c2_set_difference_v1"

PRL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PRL_ROOT.parents[1]
PUBLIC_REPORT = Path(__file__).resolve().parent / "REWIRED_EDGES.md"


@dataclass(frozen=True, slots=True)
class CellTask:
    """All immutable inputs for one ``(n, beta)`` cell."""

    collection_id: str
    n: int
    beta_key: int
    graph_indices: tuple[int, ...]
    graph_seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CellResult:
    """Exact per-graph construction and final-geometry counts for one cell."""

    task: CellTask
    operations: NDArray[np.uint16]
    displaced: NDArray[np.uint16]
    restored: NDArray[np.uint16]
    skipped: NDArray[np.uint16]


@dataclass(frozen=True, slots=True)
class AggregatedRewiringData:
    """Canonical full-collection arrays and summaries."""

    graph_seed: NDArray[np.uint64]
    operations: NDArray[np.uint16]
    displaced: NDArray[np.uint16]
    restored: NDArray[np.uint16]
    skipped: NDArray[np.uint16]
    summary_rows: tuple[dict[str, str], ...]
    displaced_logical_sha256: str


@dataclass(frozen=True, slots=True)
class RewiringAnalysisOutputs:
    """Published outputs from one complete rewiring analysis."""

    collection_id: str
    invariant_id: int
    summary_csv: Path
    raw_npz: Path
    run_manifest: Path
    report: Path
    plots: RewiringPlotPaths
    histograms: RewiringHistogramPlotPaths
    displaced_logical_sha256: str


def _set_below_normal_priority() -> None:
    """Best-effort priority reduction for this bounded auxiliary calculation."""
    try:
        if os.name == "nt":
            below_normal_priority_class = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(  # type: ignore[attr-defined]
                ctypes.windll.kernel32.GetCurrentProcess(),  # type: ignore[attr-defined]
                below_normal_priority_class,
            )
        else:
            os.nice(5)
    except (AttributeError, OSError):
        pass


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@lru_cache(maxsize=1)
def _metric_source_sha256() -> str:
    module_path = Path(sys.modules[watts_strogatz_rewiring_counts.__module__].__file__).resolve()
    return _file_sha256(module_path)


@lru_cache(maxsize=1)
def _generator_source_sha256() -> str:
    return _file_sha256(Path(graph_module.__file__).resolve())


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_deterministic_npz(path: Path, arrays: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as raw:
            with zipfile.ZipFile(
                raw,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                for key in sorted(arrays):
                    buffer = io.BytesIO()
                    np.lib.format.write_array(
                        buffer,
                        np.asanyarray(arrays[key]),
                        allow_pickle=False,
                    )
                    info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o600 << 16
                    archive.writestr(info, buffer.getvalue(), compresslevel=6)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> object:
    value = data[key]
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar; got shape {value.shape}")
    return value.item()


def _cell_path(output_dir: Path, task: CellTask) -> Path:
    return output_dir / "cells" / f"n{task.n}_b{task.beta_key:010d}.npz"


def _cell_logical_sha256(result: CellResult) -> str:
    digest = hashlib.sha256()
    digest.update(result.task.collection_id.encode("ascii"))
    digest.update(b"\0")
    digest.update(struct.pack(">IQ", result.task.n, result.task.beta_key))
    for values in zip(
        result.task.graph_indices,
        result.task.graph_seeds,
        result.operations,
        result.displaced,
        result.restored,
        result.skipped,
        strict=True,
    ):
        graph_index, graph_seed, operations, displaced, restored, skipped = values
        digest.update(
            struct.pack(
                ">IQHHHH",
                graph_index,
                graph_seed,
                int(operations),
                int(displaced),
                int(restored),
                int(skipped),
            )
        )
    return digest.hexdigest()


def _validate_cell_result(result: CellResult) -> None:
    task = result.task
    expected_shape = (len(task.graph_indices),)
    if task.graph_indices != tuple(range(len(task.graph_indices))):
        raise ValueError("graph indices must be contiguous and zero based")
    if len(task.graph_seeds) != len(task.graph_indices):
        raise ValueError("graph seed and index vectors differ in length")
    for name, array in (
        ("operations", result.operations),
        ("displaced", result.displaced),
        ("restored", result.restored),
        ("skipped", result.skipped),
    ):
        if array.dtype != np.dtype(np.uint16) or array.shape != expected_shape:
            raise ValueError(f"{name} must have dtype uint16 and shape {expected_shape}")
    edge_count = 2 * task.n
    if np.any(result.operations.astype(np.uint32) + result.skipped > edge_count):
        raise ValueError("operations plus skipped trials exceed 2n")
    if np.any(result.displaced > edge_count) or np.any(result.restored > edge_count):
        raise ValueError("a rewiring count exceeds 2n")
    if not np.array_equal(
        result.operations.astype(np.int32) - result.restored.astype(np.int32),
        result.displaced.astype(np.int32),
    ):
        raise ValueError("displaced must equal successful operations minus restorations")
    if task.beta_key == 0 and any(
        np.any(array)
        for array in (result.operations, result.displaced, result.restored, result.skipped)
    ):
        raise ValueError("all beta=0 rewiring counts must vanish")


def _compute_cell(task: CellTask) -> CellResult:
    operations = np.empty(len(task.graph_indices), dtype=np.uint16)
    displaced = np.empty_like(operations)
    restored = np.empty_like(operations)
    skipped = np.empty_like(operations)
    beta = beta_from_key(task.beta_key)
    for position, seed in enumerate(task.graph_seeds):
        counts = watts_strogatz_rewiring_counts(task.n, 2, beta, seed)
        operations[position] = counts.successful_operations
        displaced[position] = counts.displaced_base_edges
        restored[position] = counts.restored_base_edges
        skipped[position] = counts.skipped_full_neighbor
    result = CellResult(task, operations, displaced, restored, skipped)
    _validate_cell_result(result)
    return result


def _write_cell(path: Path, result: CellResult) -> None:
    _validate_cell_result(result)
    _write_deterministic_npz(
        path,
        {
            "beta": np.float64(beta_from_key(result.task.beta_key)),
            "beta_key": np.int64(result.task.beta_key),
            "collection_id": np.str_(result.task.collection_id),
            "displaced": result.displaced,
            "graph_index": np.asarray(result.task.graph_indices, dtype=np.int32),
            "graph_seed": np.asarray(result.task.graph_seeds, dtype=np.uint64),
            "logical_sha256": np.str_(_cell_logical_sha256(result)),
            "metric_source_sha256": np.str_(_metric_source_sha256()),
            "n": np.int32(result.task.n),
            "operations": result.operations,
            "restored": result.restored,
            "schema_version": np.int32(CELL_SCHEMA_VERSION),
            "skipped": result.skipped,
            "generator_source_sha256": np.str_(_generator_source_sha256()),
        },
    )


def _load_cell(path: Path, task: CellTask) -> CellResult:
    required = {
        "beta",
        "beta_key",
        "collection_id",
        "displaced",
        "graph_index",
        "graph_seed",
        "logical_sha256",
        "metric_source_sha256",
        "n",
        "operations",
        "restored",
        "schema_version",
        "skipped",
        "generator_source_sha256",
    }
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != required:
            raise ValueError(f"{path}: cell schema differs from {sorted(required)}")
        metadata = {
            "schema_version": int(_scalar(data, "schema_version")),
            "collection_id": str(_scalar(data, "collection_id")),
            "n": int(_scalar(data, "n")),
            "beta_key": int(_scalar(data, "beta_key")),
            "beta": float(_scalar(data, "beta")),
            "metric_source_sha256": str(_scalar(data, "metric_source_sha256")),
            "generator_source_sha256": str(_scalar(data, "generator_source_sha256")),
        }
        expected_metadata = {
            "schema_version": CELL_SCHEMA_VERSION,
            "collection_id": task.collection_id,
            "n": task.n,
            "beta_key": task.beta_key,
            "beta": beta_from_key(task.beta_key),
            "metric_source_sha256": _metric_source_sha256(),
            "generator_source_sha256": _generator_source_sha256(),
        }
        if metadata != expected_metadata:
            raise ValueError(f"{path}: cell metadata does not match its registry task")
        graph_index = np.array(data["graph_index"], copy=True)
        graph_seed = np.array(data["graph_seed"], copy=True)
        if graph_index.dtype != np.dtype(np.int32) or not np.array_equal(
            graph_index, np.asarray(task.graph_indices, dtype=np.int32)
        ):
            raise ValueError(f"{path}: graph_index does not match the registry")
        if graph_seed.dtype != np.dtype(np.uint64) or not np.array_equal(
            graph_seed, np.asarray(task.graph_seeds, dtype=np.uint64)
        ):
            raise ValueError(f"{path}: graph_seed does not match the registry")
        result = CellResult(
            task=task,
            operations=np.array(data["operations"], copy=True),
            displaced=np.array(data["displaced"], copy=True),
            restored=np.array(data["restored"], copy=True),
            skipped=np.array(data["skipped"], copy=True),
        )
        stored_sha256 = str(_scalar(data, "logical_sha256"))
    _validate_cell_result(result)
    actual_sha256 = _cell_logical_sha256(result)
    if stored_sha256 != actual_sha256:
        raise ValueError(f"{path}: logical SHA-256 mismatch")
    return result


def _collection_tasks(
    collection: GraphCollection, spec: GraphCollectionSpec
) -> tuple[CellTask, ...]:
    tasks: list[CellTask] = []
    for n in spec.sizes:
        for beta_key in spec.beta_keys:
            graphs = collection.cell_by_key(n, beta_key)
            tasks.append(
                CellTask(
                    collection_id=spec.collection_id,
                    n=n,
                    beta_key=beta_key,
                    graph_indices=tuple(graph.graph_index for graph in graphs),
                    graph_seeds=tuple(graph.graph_seed for graph in graphs),
                )
            )
    return tuple(tasks)


def _ensure_cells(
    tasks: tuple[CellTask, ...],
    output_dir: Path,
    *,
    workers: int,
    progress: bool,
) -> None:
    missing: list[CellTask] = []
    completed = 0
    for task in tasks:
        path = _cell_path(output_dir, task)
        if path.is_file():
            _load_cell(path, task)
            completed += 1
        else:
            missing.append(task)

    total = len(tasks)
    last_percent = completed * 100 // total

    def publish(result: CellResult) -> None:
        nonlocal completed, last_percent
        _write_cell(_cell_path(output_dir, result.task), result)
        completed += 1
        percent = completed * 100 // total
        if progress and (percent >= last_percent + 5 or completed == total):
            last_percent = percent
            print(f"rewiring cells complete: {completed}/{total} ({percent}%)", flush=True)

    missing.sort(
        key=lambda task: task.n + task.n * task.n * beta_from_key(task.beta_key),
        reverse=True,
    )
    if workers == 1:
        for task in missing:
            publish(_compute_cell(task))
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_set_below_normal_priority) as pool:
        futures = {pool.submit(_compute_cell, task): task for task in missing}
        for future in as_completed(futures):
            publish(future.result())


def _update_displaced_digest(
    digest: object,
    n: int,
    beta_key: int,
    graph_index: int,
    displaced: int,
) -> None:
    digest.update(struct.pack(">IQIH", n, beta_key, graph_index, displaced))


def _format_float(value: float) -> str:
    return format(float(value), ".17g")


def _aggregate(
    tasks: tuple[CellTask, ...], output_dir: Path, spec: GraphCollectionSpec
) -> AggregatedRewiringData:
    shape = (len(spec.sizes), len(spec.beta_keys), spec.graphs_per_cell)
    graph_seed = np.empty(shape, dtype=np.uint64)
    operations = np.empty(shape, dtype=np.uint16)
    displaced = np.empty(shape, dtype=np.uint16)
    restored = np.empty(shape, dtype=np.uint16)
    skipped = np.empty(shape, dtype=np.uint16)
    summary_rows: list[dict[str, str]] = []
    digest = hashlib.sha256()
    digest.update(spec.collection_id.encode("ascii") + b"\0")

    for task in tasks:
        n_index = spec.sizes.index(task.n)
        beta_index = spec.beta_keys.index(task.beta_key)
        result = _load_cell(_cell_path(output_dir, task), task)
        graph_seed[n_index, beta_index] = np.asarray(task.graph_seeds, dtype=np.uint64)
        operations[n_index, beta_index] = result.operations
        displaced[n_index, beta_index] = result.displaced
        restored[n_index, beta_index] = result.restored
        skipped[n_index, beta_index] = result.skipped
        for graph_index, value in zip(task.graph_indices, result.displaced, strict=True):
            _update_displaced_digest(digest, task.n, task.beta_key, graph_index, int(value))

        n_graphs = len(task.graph_indices)
        operation_values = result.operations.astype(np.float64)
        displaced_values = result.displaced.astype(np.float64)
        restored_values = result.restored.astype(np.float64)
        mean_operations = float(np.mean(operation_values))
        mean_displaced = float(np.mean(displaced_values))
        mean_restored = float(np.mean(restored_values))
        sem_operations = float(np.std(operation_values, ddof=1) / math.sqrt(n_graphs))
        sem_displaced = float(np.std(displaced_values, ddof=1) / math.sqrt(n_graphs))
        sem_restored = float(np.std(restored_values, ddof=1) / math.sqrt(n_graphs))
        edge_count = 2 * task.n
        summary_rows.append(
            {
                "collection_id": spec.collection_id,
                "n": str(task.n),
                "beta_key": str(task.beta_key),
                "beta": _format_float(beta_from_key(task.beta_key)),
                "n_graphs": str(n_graphs),
                "mean_displaced": _format_float(mean_displaced),
                "sem_displaced": _format_float(sem_displaced),
                "mean_operations": _format_float(mean_operations),
                "sem_operations": _format_float(sem_operations),
                "mean_restored": _format_float(mean_restored),
                "sem_restored": _format_float(sem_restored),
                "mean_displaced_fraction": _format_float(mean_displaced / edge_count),
                "sem_displaced_fraction": _format_float(sem_displaced / edge_count),
            }
        )

    return AggregatedRewiringData(
        graph_seed=graph_seed,
        operations=operations,
        displaced=displaced,
        restored=restored,
        skipped=skipped,
        summary_rows=tuple(summary_rows),
        displaced_logical_sha256=digest.hexdigest(),
    )


def _summary_csv_text(rows: tuple[dict[str, str], ...]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(SUMMARY_FIELDS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _write_aggregate_files(
    output_dir: Path,
    spec: GraphCollectionSpec,
    data: AggregatedRewiringData,
) -> tuple[Path, Path]:
    summary_path = output_dir / "rewired_edges_summary.csv"
    raw_path = output_dir / "rewiring_counts_raw.npz"
    _atomic_text(summary_path, _summary_csv_text(data.summary_rows))
    _write_deterministic_npz(
        raw_path,
        {
            "beta": np.asarray(spec.betas, dtype=np.float64),
            "beta_key": np.asarray(spec.beta_keys, dtype=np.int64),
            "collection_id": np.str_(spec.collection_id),
            "displaced": data.displaced,
            "displaced_logical_sha256": np.str_(data.displaced_logical_sha256),
            "graph_index": np.arange(spec.graphs_per_cell, dtype=np.int32),
            "graph_seed": data.graph_seed,
            "n": np.asarray(spec.sizes, dtype=np.int32),
            "operations": data.operations,
            "restored": data.restored,
            "schema_version": np.int32(CELL_SCHEMA_VERSION),
            "skipped": data.skipped,
        },
    )
    return summary_path, raw_path


def _register_and_verify_values(
    collection: GraphCollection,
    tasks: tuple[CellTask, ...],
    output_dir: Path,
    expected_sha256: str,
) -> int:
    invariant_id = collection.registry.define_invariant(
        INVARIANT_KEY,
        definition_version=INVARIANT_VERSION,
        value_kind="integer",
        units="edges",
        description=(
            "Number of final simple-undirected edges absent from the unrewired C(n,2) "
            "edge set; equivalently half the edge-set symmetric difference."
        ),
        parameters={
            "base_graph": "C(n,2)",
            "edge_type": "simple_undirected",
            "equivalent_formula": "cardinality(symmetric_difference(E_graph,E_base))/2",
            "formula": "cardinality(E_graph-E_base)",
            "k": 2,
        },
    )
    for task in tasks:
        result = _load_cell(_cell_path(output_dir, task), task)
        graphs = collection.cell_by_key(task.n, task.beta_key)
        records = [
            InvariantResultRecord(
                graph_id=graph.graph_id,
                invariant_id=invariant_id,
                value=int(value),
            )
            for graph, value in zip(graphs, result.displaced, strict=True)
        ]
        collection.registry.upsert_invariant_results(records)

    digest = hashlib.sha256()
    digest.update(collection.collection_id.encode("ascii") + b"\0")
    count = 0
    with collection.registry.read_connection() as connection:
        rows = connection.execute(
            """
            SELECT g.n, g.beta_key, g.graph_index, r.status, r.value_json
            FROM invariant_results AS r
            JOIN graphs AS g USING (graph_id)
            WHERE g.ensemble_id = ? AND r.invariant_id = ?
            ORDER BY g.n, g.beta_key, g.graph_index
            """,
            (collection.ensemble_id, invariant_id),
        )
        for row in rows:
            if str(row["status"]) != "complete":
                raise RuntimeError("rewired-edge invariant contains a noncomplete result")
            value = json.loads(str(row["value_json"]))
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError("rewired-edge invariant must contain integer values")
            n = int(row["n"])
            if not 0 <= value <= 2 * n:
                raise RuntimeError("stored rewired-edge invariant is outside [0, 2n]")
            _update_displaced_digest(
                digest,
                n,
                int(row["beta_key"]),
                int(row["graph_index"]),
                value,
            )
            count += 1
    expected_count = sum(len(task.graph_indices) for task in tasks)
    if count != expected_count:
        raise RuntimeError(f"stored {count} invariant values; expected {expected_count}")
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError("stored invariant values do not match the computed logical SHA-256")
    return invariant_id


def _relative_uri(path: Path) -> str:
    try:
        value = path.resolve().relative_to(REPOSITORY_ROOT)
    except ValueError:
        value = path.resolve()
    return str(value).replace("\\", "/")


def _ensemble_metadata(collection: GraphCollection) -> dict[str, object]:
    with collection.registry.read_connection() as connection:
        row = connection.execute(
            "SELECT metadata_json FROM ensembles WHERE ensemble_id = ?",
            (collection.ensemble_id,),
        ).fetchone()
    if row is None:
        raise RuntimeError("collection ensemble metadata disappeared")
    metadata = json.loads(str(row["metadata_json"]))
    if not isinstance(metadata, dict):
        raise RuntimeError("collection ensemble metadata is not an object")
    return metadata


def _write_run_manifest(
    output_dir: Path,
    spec: GraphCollectionSpec,
    collection: GraphCollection,
    data: AggregatedRewiringData,
    summary_path: Path,
    raw_path: Path,
) -> Path:
    ensemble_metadata = _ensemble_metadata(collection)
    source_files = {"analysis": Path(__file__).resolve()}
    payload = {
        "artifacts": {
            "raw_npz": {"sha256": _file_sha256(raw_path), "uri": _relative_uri(raw_path)},
            "summary_csv": {
                "sha256": _file_sha256(summary_path),
                "uri": _relative_uri(summary_path),
            },
        },
        "collection_id": spec.collection_id,
        "environment": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
        },
        "generator_source_sha256": ensemble_metadata.get("generator_source_sha256"),
        "generator_version": ensemble_metadata.get("generator_version"),
        "graph_count": spec.n_graphs,
        "invariant": {
            "definition_version": INVARIANT_VERSION,
            "formula": "cardinality(E_graph-E_C(n,2))",
            "key": INVARIANT_KEY,
            "units": "edges",
        },
        "logical_displaced_sha256": data.displaced_logical_sha256,
        "n_cells": spec.n_cells,
        "seed_content_sha256": ensemble_metadata.get("seed_content_sha256"),
        "source_sha256": {
            **{name: _file_sha256(path) for name, path in source_files.items()},
            "generator": _generator_source_sha256(),
            "metric": _metric_source_sha256(),
        },
        "specification_sha256": spec.specification_sha256,
        "total_skipped_full_neighbor": int(np.sum(data.skipped, dtype=np.uint64)),
    }
    path = output_dir / "rewiring_invariant_manifest.json"
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _register_artifacts(
    collection: GraphCollection,
    invariant_id: int,
    spec: GraphCollectionSpec,
    data: AggregatedRewiringData,
    summary_path: Path,
    raw_path: Path,
    manifest_path: Path,
) -> None:
    common_metadata = {
        "collection_id": spec.collection_id,
        "graph_count": spec.n_graphs,
        "logical_displaced_sha256": data.displaced_logical_sha256,
        "n_cells": spec.n_cells,
    }
    collection.registry.register_artifacts(
        collection.ensemble_id,
        [
            ArtifactRecord(
                artifact_key=f"{INVARIANT_KEY}.{INVARIANT_VERSION}.summary",
                uri=_relative_uri(summary_path),
                sha256=_file_sha256(summary_path),
                invariant_id=invariant_id,
                kind="invariant_summary",
                byte_size=summary_path.stat().st_size,
                media_type="text/csv",
                metadata=common_metadata,
            ),
            ArtifactRecord(
                artifact_key=f"{INVARIANT_KEY}.{INVARIANT_VERSION}.raw",
                uri=_relative_uri(raw_path),
                sha256=_file_sha256(raw_path),
                invariant_id=invariant_id,
                kind="invariant_raw",
                byte_size=raw_path.stat().st_size,
                media_type="application/x-npz",
                metadata=common_metadata,
            ),
            ArtifactRecord(
                artifact_key=f"{INVARIANT_KEY}.{INVARIANT_VERSION}.manifest",
                uri=_relative_uri(manifest_path),
                sha256=_file_sha256(manifest_path),
                invariant_id=invariant_id,
                kind="invariant_manifest",
                byte_size=manifest_path.stat().st_size,
                media_type="application/json",
                metadata=common_metadata,
            ),
        ],
    )


def _checkpoint_wal(path: Path) -> None:
    with sqlite3.connect(path, timeout=30.0) as connection:
        result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if result is None or int(result[0]) != 0:
        raise RuntimeError(f"could not checkpoint the SQLite WAL: {result!r}")


def _report_markdown(
    spec: GraphCollectionSpec,
    data: AggregatedRewiringData,
    summary_path: Path,
    raw_path: Path,
    manifest_path: Path,
    plots: RewiringPlotPaths,
    histograms: RewiringHistogramPlotPaths,
    report_path: Path,
) -> str:
    beta_one_rows = [row for row in data.summary_rows if int(row["beta_key"]) == 1_000_000_000]
    beta_min = min(beta for beta in spec.betas if beta > 0.0)
    beta_min_key = min(key for key in spec.beta_keys if key > 0)
    beta_min_rows = {
        int(row["n"]): row for row in data.summary_rows if int(row["beta_key"]) == beta_min_key
    }
    resolved_sizes = [n for n in spec.sizes if 1.0 / (2 * n) >= beta_min]
    unresolved_sizes = [n for n in spec.sizes if 1.0 / (2 * n) < beta_min]
    resolved_label = ", ".join(map(str, resolved_sizes)) or "none"
    unresolved_label = ", ".join(map(str, unresolved_sizes)) or "none"
    extension_lines = (
        [
            "For this production grid, the recommended extension is to preserve all",
            "existing points and prepend `geomspace(5e-4, 5e-3, 41)[:-1]`. This adds",
            "40 low-beta values with about 5.9% multiplicative spacing. The existing",
            "1,000 graphs per cell already give adequate statistics; the limiting factor",
            "is beta coverage.",
        ]
        if tuple(spec.sizes) == (64, 96, 128, 160, 192, 224, 256) and math.isclose(beta_min, 0.005)
        else []
    )
    lines = [
        "# Realized rewired edges",
        "",
        "Status: **complete and validated**",
        "",
        "The primary graph invariant is the number of final off-lattice edges",
        "",
        "```text",
        "N_rew(G) = |E(G) - E(C(n,2))|.",
        "```",
        "",
        "This differs from the number of accepted construction operations because a later",
        "rewiring can restore an earlier lattice edge. All plotted error bars are SEM across",
        f"the {spec.graphs_per_cell:,} indexed graph draws in each `(n, beta)` cell.",
        "",
        "![Rewired edges versus beta]({})".format(
            os.path.relpath(plots.png.resolve(), report_path.parent.resolve()).replace("\\", "/")
        ),
        "",
        "## Distribution across graph realizations",
        "",
        "Each vertical column below is the exact discrete histogram of `N_rew` across",
        f"the {spec.graphs_per_cell:,} graphs at one `(n, beta)` point. The ordinate is",
        "`f_rw = N_rew/(2n)`, so every full panel spans zero through one, while color gives",
        "the probability mass at each discrete edge count. Positive beta is logarithmic;",
        "beta equals zero is joined through conventional diagonal axis-break marks. The",
        "legend distinguishes",
        "the empirical final-edge mean, the nominal accepted-operation theory `f = beta`,",
        "and the one-edge level `f_rw = 1/(2n)`. Every size also has a low-beta zoom",
        "whose yellow diamond and annotation give the raw empirical mean at `beta_min`.",
        "",
        "![Conditional histograms of rewired edges]({})".format(
            os.path.relpath(
                histograms.overview_png.resolve(), report_path.parent.resolve()
            ).replace("\\", "/")
        ),
        "",
        "## One-edge grid test",
        "",
        "The nominal theory reaches one rewired edge at `beta_star = 1/(2n)`. A crossing",
        "below the smallest positive grid value cannot be bracketed by positive-beta data.",
        "",
        "| n | beta_star | mean N_rew at beta_min | relation to beta_min |",
        "|---:|---:|---:|:---|",
        *[
            "| {n} | {beta_star:.7f} | {mean:.3f} +/- {sem:.3f} | {relation} |".format(
                n=n,
                beta_star=1.0 / (2 * n),
                mean=float(beta_min_rows[n]["mean_displaced"]),
                sem=float(beta_min_rows[n]["sem_displaced"]),
                relation=("inside grid" if n in resolved_sizes else "below grid"),
            )
            for n in spec.sizes
        ],
        "",
        f"Here `beta_min = {beta_min:.7g}`. Positive-beta points resolve the crossing for ",
        f"`n = {resolved_label}`. They do not resolve it for ",
        f"`n = {unresolved_label}`.",
        *extension_lines,
        "",
        "## Beta equals one",
        "",
        "| n | final displaced edges | accepted operations | restored lattice edges | fraction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in beta_one_rows:
        lines.append(
            "| {n} | {mean_displaced:.4f} +/- {sem_displaced:.4f} | "
            "{mean_operations:.4f} | {mean_restored:.4f} | "
            "{mean_displaced_fraction:.6f} |".format(
                n=int(row["n"]),
                mean_displaced=float(row["mean_displaced"]),
                sem_displaced=float(row["sem_displaced"]),
                mean_operations=float(row["mean_operations"]),
                mean_restored=float(row["mean_restored"]),
                mean_displaced_fraction=float(row["mean_displaced_fraction"]),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Summary CSV: `{_relative_uri(summary_path)}`",
            f"- Per-graph raw counts: `{_relative_uri(raw_path)}`",
            f"- Run manifest: `{_relative_uri(manifest_path)}`",
            f"- Figure PNG: `{_relative_uri(plots.png)}`",
            f"- Figure PDF: `{_relative_uri(plots.pdf)}`",
            f"- Histogram overview PNG: `{_relative_uri(histograms.overview_png)}`",
            f"- Histogram overview PDF: `{_relative_uri(histograms.overview_pdf)}`",
            f"- Per-size histogram directory: `{_relative_uri(histograms.detail_pngs[0].parent)}`",
            f"- Logical result SHA-256: `{data.displaced_logical_sha256}`",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_collection_rewiring(
    spec: GraphCollectionSpec,
    database_path: Path,
    *,
    workers: int = 2,
    output_dir: Path | None = None,
    figure_dir: Path | None = None,
    progress: bool = True,
    publish_report: bool = False,
) -> RewiringAnalysisOutputs:
    """Run or resume the complete rewiring analysis for one sealed collection."""
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer in [1, 8]")
    _set_below_normal_priority()
    collection = GraphCollection(database_path, spec.collection_id)
    ensemble_metadata = _ensemble_metadata(collection)
    if ensemble_metadata.get("generator_source_sha256") != _generator_source_sha256():
        raise RuntimeError(
            "the current Watts-Strogatz generator source differs from the sealed collection"
        )
    tasks = _collection_tasks(collection, spec)
    resolved_output = (
        Path(output_dir)
        if output_dir is not None
        else database_path.parent / "invariants" / f"{INVARIANT_KEY}.{INVARIANT_VERSION}"
    )
    resolved_figure = (
        Path(figure_dir)
        if figure_dir is not None
        else PRL_ROOT / "figures" / "raw" / "graph_geometry" / spec.collection_id
    )
    _ensure_cells(tasks, resolved_output, workers=workers, progress=progress)
    data = _aggregate(tasks, resolved_output, spec)
    summary_path, raw_path = _write_aggregate_files(resolved_output, spec, data)
    plots = plot_rewiring_summary(
        summary_path,
        resolved_figure,
        expected_sizes=spec.sizes,
        expected_beta_keys=spec.beta_keys,
        expected_n_graphs=spec.graphs_per_cell,
    )
    histograms = plot_rewiring_histograms(
        raw_path,
        resolved_figure,
        expected_sizes=spec.sizes,
        expected_beta_keys=spec.beta_keys,
        expected_n_graphs=spec.graphs_per_cell,
    )
    invariant_id = _register_and_verify_values(
        collection,
        tasks,
        resolved_output,
        data.displaced_logical_sha256,
    )
    manifest_path = _write_run_manifest(
        resolved_output,
        spec,
        collection,
        data,
        summary_path,
        raw_path,
    )
    _register_artifacts(
        collection,
        invariant_id,
        spec,
        data,
        summary_path,
        raw_path,
        manifest_path,
    )
    _checkpoint_wal(database_path)

    refreshed = validate_existing_collection(spec, database_path)
    write_reports(spec, refreshed, publish_status=publish_report)
    report_path = resolved_output / "REWIRED_EDGES.md"
    report = _report_markdown(
        spec,
        data,
        summary_path,
        raw_path,
        manifest_path,
        plots,
        histograms,
        report_path,
    )
    _atomic_text(report_path, report)
    if publish_report:
        public_report = _report_markdown(
            spec,
            data,
            summary_path,
            raw_path,
            manifest_path,
            plots,
            histograms,
            PUBLIC_REPORT,
        )
        _atomic_text(PUBLIC_REPORT, public_report)
    return RewiringAnalysisOutputs(
        collection_id=spec.collection_id,
        invariant_id=invariant_id,
        summary_csv=summary_path,
        raw_npz=raw_path,
        run_manifest=manifest_path,
        report=report_path,
        plots=plots,
        histograms=histograms,
        displaced_logical_sha256=data.displaced_logical_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "production"), default="smoke")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--confirm-production", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = production_spec() if args.profile == "production" else smoke_spec()
    database_path = (args.database or default_database_path(spec)).resolve()
    if args.profile == "production" and not args.confirm_production and not args.dry_run:
        raise SystemExit("production rewiring analysis requires --confirm-production")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "collection_id": spec.collection_id,
                    "database": str(database_path),
                    "graph_count": spec.n_graphs,
                    "n_cells": spec.n_cells,
                    "workers": args.workers,
                },
                indent=2,
            )
        )
        return 0
    if not database_path.is_file():
        raise SystemExit(f"registry does not exist: {database_path}")
    outputs = analyze_collection_rewiring(
        spec,
        database_path,
        workers=args.workers,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        publish_report=args.profile == "production",
    )
    print(f"summary: {outputs.summary_csv}")
    print(f"raw counts: {outputs.raw_npz}")
    print(f"figure: {outputs.plots.png}")
    print(f"report: {outputs.report}")
    print(f"logical SHA-256: {outputs.displaced_logical_sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "AggregatedRewiringData",
    "CellResult",
    "CellTask",
    "INVARIANT_KEY",
    "INVARIANT_VERSION",
    "RewiringAnalysisOutputs",
    "analyze_collection_rewiring",
    "main",
]
