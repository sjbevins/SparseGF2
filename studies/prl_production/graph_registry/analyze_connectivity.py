# ruff: noqa: E402
"""Compute and publish algebraic connectivity for the sealed graph collection.

The calculation is graph indexed and resume safe.  Each ``(n, beta)`` cell is
an immutable NPZ checkpoint containing the 1,000 graph values in registry
order.  Complete cells are validated before reuse, then aggregated into the
raw tensor, cell summary, and cumulative normalized-gain analysis requested
for the paper.
"""

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
from typing import TYPE_CHECKING

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import scipy
from numpy.typing import NDArray

from sparsegf2.circuits import graphs as graph_module

from .build import (
    default_database_path,
    validate_existing_collection,
    write_reports,
)
from .collection import GraphCollection
from .connectivity_metrics import (
    ring_algebraic_connectivity,
    watts_strogatz_algebraic_connectivity,
)
from .database import ArtifactRecord, InvariantResultRecord
from .spec import GraphCollectionSpec, beta_from_key, production_spec, smoke_spec

if TYPE_CHECKING:
    from .plot_connectivity_gain import ConnectivityGainPlotPaths


CELL_SCHEMA_VERSION = 1
INVARIANT_KEY = "graph.algebraic_connectivity"
INVARIANT_VERSION = "combinatorial_laplacian_v1"
BOOTSTRAP_MASTER_SEED = 2_417_031_113
DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
PRODUCTION_NESTED_SIZE_SETS = (
    (64,),
    (64, 128),
    (64, 128, 192),
    (64, 128, 192, 256),
)

CELL_SUMMARY_FIELDS = (
    "collection_id",
    "n",
    "beta_key",
    "beta",
    "n_graphs",
    "lambda2_mean",
    "lambda2_std",
    "lambda2_sem",
    "lambda2_min",
    "lambda2_max",
    "lambda2_ring_exact",
    "gain_mean",
    "gain_sem",
)

NESTED_SUMMARY_FIELDS = (
    "collection_id",
    "set_size",
    "size_set",
    "beta_key",
    "beta",
    "n_graphs_per_cell",
    "g_lambda",
    "g_lambda_sem",
    "log_g_lambda",
    "log_g_lambda_sem",
    "ci68_low",
    "ci68_high",
)

PRL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PRL_ROOT.parents[1]
PUBLIC_REPORT = Path(__file__).resolve().parent / "ALGEBRAIC_CONNECTIVITY.md"


@dataclass(frozen=True, slots=True)
class ConnectivityCellTask:
    """Immutable registry inputs for one ``(n, beta)`` cell."""

    collection_id: str
    graph_k: int
    n: int
    beta_key: int
    graph_indices: tuple[int, ...]
    graph_seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ConnectivityCellResult:
    """Algebraic-connectivity values for one complete cell."""

    task: ConnectivityCellTask
    lambda2: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AggregatedConnectivityData:
    """Canonical full-collection values and derived summaries."""

    graph_seed: NDArray[np.uint64]
    lambda2: NDArray[np.float64]
    cell_rows: tuple[dict[str, str], ...]
    nested_rows: tuple[dict[str, str], ...]
    lambda2_logical_sha256: str


@dataclass(frozen=True, slots=True)
class ConnectivityAnalysisOutputs:
    """Published outputs of one complete connectivity analysis."""

    collection_id: str
    invariant_id: int
    cell_summary_csv: Path
    nested_summary_csv: Path
    raw_npz: Path
    run_manifest: Path
    report: Path
    plots: ConnectivityGainPlotPaths
    lambda2_logical_sha256: str


def _set_below_normal_priority() -> None:
    """Use one BLAS thread and best-effort BelowNormal process priority."""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
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
    module_path = Path(sys.modules[ring_algebraic_connectivity.__module__].__file__).resolve()
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


def _format_float(value: float) -> str:
    return format(float(value), ".17g")


def _cell_path(output_dir: Path, task: ConnectivityCellTask) -> Path:
    return output_dir / "cells" / f"n{task.n}_b{task.beta_key:010d}.npz"


def _update_lambda2_digest(
    digest: object,
    *,
    n: int,
    beta_key: int,
    graph_index: int,
    graph_seed: int,
    value: float,
) -> None:
    digest.update(struct.pack(">IQIQd", n, beta_key, graph_index, graph_seed, value))


def _cell_logical_sha256(result: ConnectivityCellResult) -> str:
    digest = hashlib.sha256()
    digest.update(result.task.collection_id.encode("ascii") + b"\0")
    for graph_index, graph_seed, value in zip(
        result.task.graph_indices,
        result.task.graph_seeds,
        result.lambda2,
        strict=True,
    ):
        _update_lambda2_digest(
            digest,
            n=result.task.n,
            beta_key=result.task.beta_key,
            graph_index=graph_index,
            graph_seed=graph_seed,
            value=float(value),
        )
    return digest.hexdigest()


def _validate_cell_result(result: ConnectivityCellResult) -> None:
    task = result.task
    expected_shape = (len(task.graph_indices),)
    if task.graph_indices != tuple(range(len(task.graph_indices))):
        raise ValueError("graph indices must be contiguous and zero based")
    if len(task.graph_seeds) != len(task.graph_indices):
        raise ValueError("graph seed and graph index vectors differ in length")
    if result.lambda2.dtype != np.dtype(np.float64) or result.lambda2.shape != expected_shape:
        raise ValueError(f"lambda2 must have dtype float64 and shape {expected_shape}")
    if not np.all(np.isfinite(result.lambda2)):
        raise ValueError("lambda2 values must be finite")
    if np.any(result.lambda2 < 0.0):
        raise ValueError("lambda2 values must be nonnegative")
    if task.beta_key == 0:
        ring = ring_algebraic_connectivity(task.n, task.graph_k)
        if not np.allclose(result.lambda2, ring, rtol=2e-12, atol=2e-13):
            raise ValueError("beta=0 lambda2 values do not match the exact C(n,k) gap")


def _compute_cell(task: ConnectivityCellTask) -> ConnectivityCellResult:
    beta = beta_from_key(task.beta_key)
    values = np.fromiter(
        (
            watts_strogatz_algebraic_connectivity(
                task.n,
                task.graph_k,
                beta,
                graph_seed,
            )
            for graph_seed in task.graph_seeds
        ),
        dtype=np.float64,
        count=len(task.graph_seeds),
    )
    result = ConnectivityCellResult(task=task, lambda2=values)
    _validate_cell_result(result)
    return result


def _write_cell(path: Path, result: ConnectivityCellResult) -> None:
    _validate_cell_result(result)
    task = result.task
    _write_deterministic_npz(
        path,
        {
            "beta": np.float64(beta_from_key(task.beta_key)),
            "beta_key": np.int64(task.beta_key),
            "collection_id": np.str_(task.collection_id),
            "generator_source_sha256": np.str_(_generator_source_sha256()),
            "graph_index": np.asarray(task.graph_indices, dtype=np.int32),
            "graph_k": np.int32(task.graph_k),
            "graph_seed": np.asarray(task.graph_seeds, dtype=np.uint64),
            "lambda2": result.lambda2,
            "logical_sha256": np.str_(_cell_logical_sha256(result)),
            "metric_source_sha256": np.str_(_metric_source_sha256()),
            "n": np.int32(task.n),
            "schema_version": np.int32(CELL_SCHEMA_VERSION),
        },
    )


def _load_cell(path: Path, task: ConnectivityCellTask) -> ConnectivityCellResult:
    required = {
        "beta",
        "beta_key",
        "collection_id",
        "generator_source_sha256",
        "graph_index",
        "graph_k",
        "graph_seed",
        "lambda2",
        "logical_sha256",
        "metric_source_sha256",
        "n",
        "schema_version",
    }
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != required:
            raise ValueError(f"{path}: cell schema differs from {sorted(required)}")
        metadata = {
            "beta": float(_scalar(data, "beta")),
            "beta_key": int(_scalar(data, "beta_key")),
            "collection_id": str(_scalar(data, "collection_id")),
            "generator_source_sha256": str(_scalar(data, "generator_source_sha256")),
            "graph_k": int(_scalar(data, "graph_k")),
            "metric_source_sha256": str(_scalar(data, "metric_source_sha256")),
            "n": int(_scalar(data, "n")),
            "schema_version": int(_scalar(data, "schema_version")),
        }
        expected_metadata = {
            "beta": beta_from_key(task.beta_key),
            "beta_key": task.beta_key,
            "collection_id": task.collection_id,
            "generator_source_sha256": _generator_source_sha256(),
            "graph_k": task.graph_k,
            "metric_source_sha256": _metric_source_sha256(),
            "n": task.n,
            "schema_version": CELL_SCHEMA_VERSION,
        }
        if metadata != expected_metadata:
            raise ValueError(f"{path}: cell metadata does not match its registry task")
        graph_index = np.array(data["graph_index"], copy=True)
        graph_seed = np.array(data["graph_seed"], copy=True)
        if graph_index.dtype != np.dtype(np.int32) or not np.array_equal(
            graph_index,
            np.asarray(task.graph_indices, dtype=np.int32),
        ):
            raise ValueError(f"{path}: graph_index does not match the registry")
        if graph_seed.dtype != np.dtype(np.uint64) or not np.array_equal(
            graph_seed,
            np.asarray(task.graph_seeds, dtype=np.uint64),
        ):
            raise ValueError(f"{path}: graph_seed does not match the registry")
        result = ConnectivityCellResult(
            task=task,
            lambda2=np.array(data["lambda2"], copy=True),
        )
        stored_digest = str(_scalar(data, "logical_sha256"))
    _validate_cell_result(result)
    if stored_digest != _cell_logical_sha256(result):
        raise ValueError(f"{path}: logical SHA-256 mismatch")
    return result


def _collection_tasks(
    collection: GraphCollection,
    spec: GraphCollectionSpec,
) -> tuple[ConnectivityCellTask, ...]:
    tasks: list[ConnectivityCellTask] = []
    for n in spec.sizes:
        for beta_key in spec.beta_keys:
            graphs = collection.cell_by_key(n, beta_key)
            tasks.append(
                ConnectivityCellTask(
                    collection_id=spec.collection_id,
                    graph_k=spec.graph_k,
                    n=n,
                    beta_key=beta_key,
                    graph_indices=tuple(graph.graph_index for graph in graphs),
                    graph_seeds=tuple(graph.graph_seed for graph in graphs),
                )
            )
    return tuple(tasks)


def _ensure_cells(
    tasks: tuple[ConnectivityCellTask, ...],
    output_dir: Path,
    *,
    workers: int,
    progress: bool,
) -> None:
    missing: list[ConnectivityCellTask] = []
    complete = 0
    for task in tasks:
        path = _cell_path(output_dir, task)
        if path.is_file():
            _load_cell(path, task)
            complete += 1
        else:
            missing.append(task)
    total = len(tasks)
    last_percent = complete * 100 // total

    def publish(result: ConnectivityCellResult) -> None:
        nonlocal complete, last_percent
        _write_cell(_cell_path(output_dir, result.task), result)
        complete += 1
        percent = complete * 100 // total
        if progress and (percent >= last_percent + 5 or complete == total):
            last_percent = percent
            print(f"connectivity cells complete: {complete}/{total} ({percent}%)", flush=True)

    missing.sort(
        key=lambda task: task.n**3 + task.n**2 * beta_from_key(task.beta_key),
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


def _default_nested_size_sets(spec: GraphCollectionSpec) -> tuple[tuple[int, ...], ...]:
    if all(size in spec.sizes for size in PRODUCTION_NESTED_SIZE_SETS[-1]):
        return PRODUCTION_NESTED_SIZE_SETS
    return tuple(tuple(spec.sizes[:index]) for index in range(1, len(spec.sizes) + 1))


def _validate_nested_size_sets(
    spec: GraphCollectionSpec,
    size_sets: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    if not size_sets:
        raise ValueError("nested_size_sets must be nonempty")
    previous: tuple[int, ...] = ()
    validated: list[tuple[int, ...]] = []
    for index, raw in enumerate(size_sets, 1):
        current = tuple(raw)
        if len(current) != index:
            raise ValueError("nested_size_sets must increase by one size at each step")
        if len(set(current)) != len(current) or tuple(sorted(current)) != current:
            raise ValueError("each nested size set must be strictly increasing")
        if any(size not in spec.sizes for size in current):
            raise ValueError("nested_size_sets contains a size outside the graph collection")
        if current[:-1] != previous:
            raise ValueError("nested_size_sets must be cumulative")
        validated.append(current)
        previous = current
    return tuple(validated)


def _bootstrap_cell_means(
    values: NDArray[np.float64],
    *,
    resamples: int,
    seed_components: tuple[int, ...],
) -> NDArray[np.float64]:
    rng = np.random.default_rng(np.random.SeedSequence(seed_components))
    output = np.empty(resamples, dtype=np.float64)
    n_graphs = len(values)
    chunk = max(1, min(256, 2_000_000 // n_graphs))
    for start in range(0, resamples, chunk):
        stop = min(resamples, start + chunk)
        indices = rng.integers(0, n_graphs, size=(stop - start, n_graphs), dtype=np.int32)
        output[start:stop] = np.mean(values[indices], axis=1)
    return output


def _nested_rows(
    spec: GraphCollectionSpec,
    lambda2: NDArray[np.float64],
    size_sets: tuple[tuple[int, ...], ...],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for beta_index, beta_key in enumerate(spec.beta_keys):
        means: dict[int, float] = {}
        sems: dict[int, float] = {}
        bootstrap_means: dict[int, NDArray[np.float64]] = {}
        for n in size_sets[-1]:
            n_index = spec.sizes.index(n)
            values = lambda2[n_index, beta_index]
            mean = float(np.mean(values))
            if mean <= 0.0:
                raise RuntimeError(
                    f"mean lambda2 is not positive for n={n}, beta_key={beta_key}; "
                    "the logarithmic gain is undefined"
                )
            means[n] = mean
            sems[n] = float(np.std(values, ddof=1) / math.sqrt(len(values)))
            bootstrap_means[n] = _bootstrap_cell_means(
                values,
                resamples=bootstrap_resamples,
                seed_components=(bootstrap_seed, beta_key, n),
            )

        for size_set in size_sets:
            if beta_key == 0:
                for n in size_set:
                    ring = ring_algebraic_connectivity(n, spec.graph_k)
                    if not math.isclose(means[n], ring, rel_tol=2e-12, abs_tol=2e-13):
                        raise RuntimeError("beta=0 ensemble mean does not match the exact ring gap")
                rows.append(
                    {
                        "collection_id": spec.collection_id,
                        "set_size": str(len(size_set)),
                        "size_set": json.dumps(list(size_set), separators=(",", ":")),
                        "beta_key": str(beta_key),
                        "beta": _format_float(beta_from_key(beta_key)),
                        "n_graphs_per_cell": str(spec.graphs_per_cell),
                        "g_lambda": "1",
                        "g_lambda_sem": "0",
                        "log_g_lambda": "0",
                        "log_g_lambda_sem": "0",
                        "ci68_low": "1",
                        "ci68_high": "1",
                    }
                )
                continue
            ratios = np.asarray(
                [means[n] / ring_algebraic_connectivity(n, spec.graph_k) for n in size_set],
                dtype=np.float64,
            )
            log_g = float(np.mean(np.log(ratios)))
            g_lambda = math.exp(log_g)
            relative_sem = np.asarray([sems[n] / means[n] for n in size_set])
            log_sem = float(np.sqrt(np.sum(relative_sem**2)) / len(size_set))
            g_sem = g_lambda * log_sem

            bootstrap_logs = np.zeros(bootstrap_resamples, dtype=np.float64)
            for n in size_set:
                ring = ring_algebraic_connectivity(n, spec.graph_k)
                values = bootstrap_means[n]
                if np.any(values <= 0.0):
                    raise RuntimeError("a bootstrap cell mean is nonpositive")
                bootstrap_logs += np.log(values / ring)
            bootstrap_gain = np.exp(bootstrap_logs / len(size_set))
            ci_low, ci_high = np.quantile(bootstrap_gain, (0.16, 0.84), method="linear")
            rows.append(
                {
                    "collection_id": spec.collection_id,
                    "set_size": str(len(size_set)),
                    "size_set": json.dumps(list(size_set), separators=(",", ":")),
                    "beta_key": str(beta_key),
                    "beta": _format_float(beta_from_key(beta_key)),
                    "n_graphs_per_cell": str(spec.graphs_per_cell),
                    "g_lambda": _format_float(g_lambda),
                    "g_lambda_sem": _format_float(g_sem),
                    "log_g_lambda": _format_float(log_g),
                    "log_g_lambda_sem": _format_float(log_sem),
                    "ci68_low": _format_float(float(ci_low)),
                    "ci68_high": _format_float(float(ci_high)),
                }
            )
    return tuple(sorted(rows, key=lambda row: (int(row["set_size"]), int(row["beta_key"]))))


def _aggregate(
    tasks: tuple[ConnectivityCellTask, ...],
    output_dir: Path,
    spec: GraphCollectionSpec,
    size_sets: tuple[tuple[int, ...], ...],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> AggregatedConnectivityData:
    shape = (len(spec.sizes), len(spec.beta_keys), spec.graphs_per_cell)
    graph_seed = np.empty(shape, dtype=np.uint64)
    lambda2 = np.empty(shape, dtype=np.float64)
    rows: list[dict[str, str]] = []
    digest = hashlib.sha256()
    digest.update(spec.collection_id.encode("ascii") + b"\0")
    for task in tasks:
        n_index = spec.sizes.index(task.n)
        beta_index = spec.beta_keys.index(task.beta_key)
        result = _load_cell(_cell_path(output_dir, task), task)
        graph_seed[n_index, beta_index] = np.asarray(task.graph_seeds, dtype=np.uint64)
        lambda2[n_index, beta_index] = result.lambda2
        for graph_index, graph_seed_value, value in zip(
            task.graph_indices,
            task.graph_seeds,
            result.lambda2,
            strict=True,
        ):
            _update_lambda2_digest(
                digest,
                n=task.n,
                beta_key=task.beta_key,
                graph_index=graph_index,
                graph_seed=graph_seed_value,
                value=float(value),
            )
        values = result.lambda2
        count = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        sem = std / math.sqrt(count)
        ring = ring_algebraic_connectivity(task.n, task.graph_k)
        rows.append(
            {
                "collection_id": spec.collection_id,
                "n": str(task.n),
                "beta_key": str(task.beta_key),
                "beta": _format_float(beta_from_key(task.beta_key)),
                "n_graphs": str(count),
                "lambda2_mean": _format_float(mean),
                "lambda2_std": _format_float(std),
                "lambda2_sem": _format_float(sem),
                "lambda2_min": _format_float(float(np.min(values))),
                "lambda2_max": _format_float(float(np.max(values))),
                "lambda2_ring_exact": _format_float(ring),
                "gain_mean": _format_float(mean / ring),
                "gain_sem": _format_float(sem / ring),
            }
        )
    nested_rows = _nested_rows(
        spec,
        lambda2,
        size_sets,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    return AggregatedConnectivityData(
        graph_seed=graph_seed,
        lambda2=lambda2,
        cell_rows=tuple(rows),
        nested_rows=nested_rows,
        lambda2_logical_sha256=digest.hexdigest(),
    )


def _csv_text(rows: tuple[dict[str, str], ...], fields: tuple[str, ...]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _write_aggregate_files(
    output_dir: Path,
    spec: GraphCollectionSpec,
    data: AggregatedConnectivityData,
) -> tuple[Path, Path, Path]:
    cell_summary = output_dir / "algebraic_connectivity_summary.csv"
    nested_summary = output_dir / "normalized_connectivity_gain.csv"
    raw_path = output_dir / "algebraic_connectivity_raw.npz"
    _atomic_text(cell_summary, _csv_text(data.cell_rows, CELL_SUMMARY_FIELDS))
    _atomic_text(nested_summary, _csv_text(data.nested_rows, NESTED_SUMMARY_FIELDS))
    _write_deterministic_npz(
        raw_path,
        {
            "beta": np.asarray(spec.betas, dtype=np.float64),
            "beta_key": np.asarray(spec.beta_keys, dtype=np.int64),
            "collection_id": np.str_(spec.collection_id),
            "graph_index": np.arange(spec.graphs_per_cell, dtype=np.int32),
            "graph_seed": data.graph_seed,
            "lambda2": data.lambda2,
            "lambda2_logical_sha256": np.str_(data.lambda2_logical_sha256),
            "n": np.asarray(spec.sizes, dtype=np.int32),
            "schema_version": np.int32(CELL_SCHEMA_VERSION),
        },
    )
    return cell_summary, nested_summary, raw_path


def _register_and_verify_values(
    collection: GraphCollection,
    tasks: tuple[ConnectivityCellTask, ...],
    output_dir: Path,
    expected_digest: str,
) -> int:
    invariant_id = collection.registry.define_invariant(
        INVARIANT_KEY,
        definition_version=INVARIANT_VERSION,
        value_kind="real",
        units="spectral_gap",
        description=(
            "Second-smallest eigenvalue of the simple undirected graph's "
            "combinatorial Laplacian L=D-A."
        ),
        parameters={
            "eigenvalue": "lambda_2",
            "graph_type": "simple_undirected",
            "laplacian": "combinatorial_D_minus_A",
        },
    )
    for task in tasks:
        result = _load_cell(_cell_path(output_dir, task), task)
        graphs = collection.cell_by_key(task.n, task.beta_key)
        collection.registry.upsert_invariant_results(
            InvariantResultRecord(
                graph_id=graph.graph_id,
                invariant_id=invariant_id,
                value=float(value),
            )
            for graph, value in zip(graphs, result.lambda2, strict=True)
        )

    digest = hashlib.sha256()
    digest.update(collection.collection_id.encode("ascii") + b"\0")
    count = 0
    with collection.registry.read_connection() as connection:
        rows = connection.execute(
            """
            SELECT g.n, g.beta_key, g.graph_index, g.graph_seed, r.status, r.value_json
            FROM invariant_results AS r
            JOIN graphs AS g USING (graph_id)
            WHERE g.ensemble_id = ? AND r.invariant_id = ?
            ORDER BY g.n, g.beta_key, g.graph_index
            """,
            (collection.ensemble_id, invariant_id),
        )
        for row in rows:
            if str(row["status"]) != "complete":
                raise RuntimeError("algebraic-connectivity invariant contains noncomplete data")
            value = json.loads(str(row["value_json"]))
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError("algebraic-connectivity invariant must contain real values")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise RuntimeError("stored algebraic connectivity is invalid")
            _update_lambda2_digest(
                digest,
                n=int(row["n"]),
                beta_key=int(row["beta_key"]),
                graph_index=int(row["graph_index"]),
                graph_seed=int(row["graph_seed"]),
                value=value,
            )
            count += 1
    if count != sum(len(task.graph_indices) for task in tasks):
        raise RuntimeError(f"stored {count} invariant values; expected the complete collection")
    if digest.hexdigest() != expected_digest:
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


def _artifact_payload(path: Path) -> dict[str, object]:
    return {"sha256": _file_sha256(path), "uri": _relative_uri(path)}


def _write_run_manifest(
    output_dir: Path,
    spec: GraphCollectionSpec,
    collection: GraphCollection,
    data: AggregatedConnectivityData,
    size_sets: tuple[tuple[int, ...], ...],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    cell_summary: Path,
    nested_summary: Path,
    raw_path: Path,
    plots: ConnectivityGainPlotPaths,
) -> Path:
    ensemble_metadata = _ensemble_metadata(collection)
    payload = {
        "artifacts": {
            "cell_summary_csv": _artifact_payload(cell_summary),
            "nested_gain_csv": _artifact_payload(nested_summary),
            "raw_npz": _artifact_payload(raw_path),
            "figure_png": _artifact_payload(plots.png),
            "figure_pdf": _artifact_payload(plots.pdf),
        },
        "bootstrap": {
            "ci_quantiles": [0.16, 0.84],
            "method": "deterministic_stratified_graph_resampling",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "shared_cell_resamples_across_cumulative_sets": True,
        },
        "collection_id": spec.collection_id,
        "environment": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "scipy": scipy.__version__,
            "sqlite": sqlite3.sqlite_version,
        },
        "generator_source_sha256": ensemble_metadata.get("generator_source_sha256"),
        "generator_version": ensemble_metadata.get("generator_version"),
        "graph_count": spec.n_graphs,
        "invariant": {
            "definition_version": INVARIANT_VERSION,
            "key": INVARIANT_KEY,
            "laplacian": "D-A",
            "units": "spectral_gap",
        },
        "lambda2_logical_sha256": data.lambda2_logical_sha256,
        "n_cells": spec.n_cells,
        "nested_size_sets": [list(values) for values in size_sets],
        "normalization": {
            "aggregate": "geometric_mean_across_sizes_of_ensemble_mean_gain",
            "denominator": "lambda2[C(n,2)]",
            "ring_formula": "4-2*cos(2*pi/n)-2*cos(4*pi/n)",
            "sem": "first_order_log_delta_method",
        },
        "seed_content_sha256": ensemble_metadata.get("seed_content_sha256"),
        "source_sha256": {
            "analysis": _file_sha256(Path(__file__).resolve()),
            "generator": _generator_source_sha256(),
            "metric": _metric_source_sha256(),
            "plot": _file_sha256(Path(sys.modules[plots.__class__.__module__].__file__).resolve()),
        },
        "specification_sha256": spec.specification_sha256,
    }
    path = output_dir / "connectivity_invariant_manifest.json"
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _register_artifacts(
    collection: GraphCollection,
    invariant_id: int,
    spec: GraphCollectionSpec,
    data: AggregatedConnectivityData,
    artifacts: tuple[tuple[str, Path, str, str], ...],
) -> None:
    common = {
        "collection_id": spec.collection_id,
        "graph_count": spec.n_graphs,
        "lambda2_logical_sha256": data.lambda2_logical_sha256,
        "n_cells": spec.n_cells,
    }
    collection.registry.register_artifacts(
        collection.ensemble_id,
        [
            ArtifactRecord(
                artifact_key=f"{INVARIANT_KEY}.{INVARIANT_VERSION}.{suffix}",
                uri=_relative_uri(path),
                sha256=_file_sha256(path),
                invariant_id=invariant_id,
                kind=kind,
                byte_size=path.stat().st_size,
                media_type=media_type,
                metadata=common,
            )
            for suffix, path, kind, media_type in artifacts
        ],
    )


def _checkpoint_wal(path: Path) -> None:
    with sqlite3.connect(path, timeout=30.0) as connection:
        result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if result is None or int(result[0]) != 0:
        raise RuntimeError(f"could not checkpoint the SQLite WAL: {result!r}")


def _report_markdown(
    spec: GraphCollectionSpec,
    data: AggregatedConnectivityData,
    size_sets: tuple[tuple[int, ...], ...],
    *,
    cell_summary: Path,
    nested_summary: Path,
    raw_path: Path,
    manifest_path: Path,
    plots: ConnectivityGainPlotPaths,
) -> str:
    lines = [
        "# Algebraic connectivity and normalized gain",
        "",
        "Status: **complete and validated**.",
        "",
        f"Collection: `{spec.collection_id}` ({spec.n_graphs:,} graphs in {spec.n_cells} cells).",
        "",
        "For each graph we compute the second-smallest eigenvalue of the combinatorial ",
        "Laplacian.  Each cell first forms the arithmetic graph mean.  The cumulative ",
        "normalized gain is then the geometric mean across its stated size set:",
        "",
        r"\[",
        r"g_\lambda(\beta)=\exp\!\left[\frac{1}{|\mathcal N|}",
        r"\sum_{n\in\mathcal N}\ln\frac{\overline{\lambda_2(G_{n,\beta})}}",
        r"{\lambda_2[C(n,2)]}\right].",
        r"\]",
        "",
        "Nested size sets: " + ", ".join(str(list(values)) for values in size_sets) + ".",
        "Error bars use the log-delta SEM; the 68% interval independently resamples ",
        "graphs within each cell and reuses those cell resamples across cumulative sets.",
        "",
        "## Artifacts",
        "",
        f"- Per-cell summary: `{_relative_uri(cell_summary)}`",
        f"- Nested normalized gain: `{_relative_uri(nested_summary)}`",
        f"- Per-graph raw values: `{_relative_uri(raw_path)}`",
        f"- Run manifest: `{_relative_uri(manifest_path)}`",
        f"- Figure PNG: `{_relative_uri(plots.png)}`",
        f"- Figure PDF: `{_relative_uri(plots.pdf)}`",
        f"- Logical result SHA-256: `{data.lambda2_logical_sha256}`",
        "",
    ]
    return "\n".join(lines)


def analyze_collection_connectivity(
    spec: GraphCollectionSpec,
    database_path: Path,
    *,
    workers: int = 2,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_MASTER_SEED,
    nested_size_sets: tuple[tuple[int, ...], ...] | None = None,
    output_dir: Path | None = None,
    figure_dir: Path | None = None,
    progress: bool = True,
    publish_report: bool = False,
) -> ConnectivityAnalysisOutputs:
    """Run or resume the complete algebraic-connectivity analysis."""
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer in [1, 8]")
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or not 2 <= bootstrap_resamples <= 100_000
    ):
        raise ValueError("bootstrap_resamples must be an integer in [2, 100000]")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if spec.graphs_per_cell < 2:
        raise ValueError("connectivity statistics require at least two graphs per cell")
    _set_below_normal_priority()
    size_sets = _validate_nested_size_sets(
        spec,
        _default_nested_size_sets(spec) if nested_size_sets is None else nested_size_sets,
    )
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
    data = _aggregate(
        tasks,
        resolved_output,
        spec,
        size_sets,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    cell_summary, nested_summary, raw_path = _write_aggregate_files(resolved_output, spec, data)

    from .plot_connectivity_gain import plot_connectivity_gain

    plots = plot_connectivity_gain(
        nested_summary,
        resolved_figure,
        expected_collection_id=spec.collection_id,
        expected_beta_keys=spec.beta_keys,
        expected_size_sets=size_sets,
        expected_n_graphs_per_cell=spec.graphs_per_cell,
    )
    invariant_id = _register_and_verify_values(
        collection,
        tasks,
        resolved_output,
        data.lambda2_logical_sha256,
    )
    manifest_path = _write_run_manifest(
        resolved_output,
        spec,
        collection,
        data,
        size_sets,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        cell_summary=cell_summary,
        nested_summary=nested_summary,
        raw_path=raw_path,
        plots=plots,
    )
    _register_artifacts(
        collection,
        invariant_id,
        spec,
        data,
        (
            ("cell_summary", cell_summary, "invariant_summary", "text/csv"),
            ("nested_gain", nested_summary, "derived_summary", "text/csv"),
            ("raw", raw_path, "invariant_raw", "application/x-npz"),
            ("manifest", manifest_path, "invariant_manifest", "application/json"),
            ("figure_png", plots.png, "invariant_figure", "image/png"),
            ("figure_pdf", plots.pdf, "invariant_figure", "application/pdf"),
        ),
    )
    _checkpoint_wal(database_path)
    refreshed = validate_existing_collection(spec, database_path)
    write_reports(spec, refreshed, publish_status=publish_report)
    report_path = resolved_output / "ALGEBRAIC_CONNECTIVITY.md"
    report = _report_markdown(
        spec,
        data,
        size_sets,
        cell_summary=cell_summary,
        nested_summary=nested_summary,
        raw_path=raw_path,
        manifest_path=manifest_path,
        plots=plots,
    )
    _atomic_text(report_path, report)
    if publish_report:
        _atomic_text(PUBLIC_REPORT, report)
    return ConnectivityAnalysisOutputs(
        collection_id=spec.collection_id,
        invariant_id=invariant_id,
        cell_summary_csv=cell_summary,
        nested_summary_csv=nested_summary,
        raw_npz=raw_path,
        run_manifest=manifest_path,
        report=report_path,
        plots=plots,
        lambda2_logical_sha256=data.lambda2_logical_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "production"), default="smoke")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_MASTER_SEED)
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
        raise SystemExit("production connectivity analysis requires --confirm-production")
    size_sets = _default_nested_size_sets(spec)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "bootstrap_resamples": args.bootstrap_resamples,
                    "collection_id": spec.collection_id,
                    "database": str(database_path),
                    "graph_count": spec.n_graphs,
                    "n_cells": spec.n_cells,
                    "nested_size_sets": [list(values) for values in size_sets],
                    "workers": args.workers,
                },
                indent=2,
            )
        )
        return 0
    if not database_path.is_file():
        raise SystemExit(f"registry does not exist: {database_path}")
    outputs = analyze_collection_connectivity(
        spec,
        database_path,
        workers=args.workers,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        nested_size_sets=size_sets,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        publish_report=args.profile == "production",
    )
    print(f"cell summary: {outputs.cell_summary_csv}")
    print(f"normalized gain: {outputs.nested_summary_csv}")
    print(f"raw values: {outputs.raw_npz}")
    print(f"figure: {outputs.plots.png}")
    print(f"report: {outputs.report}")
    print(f"logical SHA-256: {outputs.lambda2_logical_sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "AggregatedConnectivityData",
    "BOOTSTRAP_MASTER_SEED",
    "CELL_SUMMARY_FIELDS",
    "ConnectivityAnalysisOutputs",
    "ConnectivityCellResult",
    "ConnectivityCellTask",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "INVARIANT_KEY",
    "INVARIANT_VERSION",
    "NESTED_SUMMARY_FIELDS",
    "PRODUCTION_NESTED_SIZE_SETS",
    "analyze_collection_connectivity",
    "main",
]
