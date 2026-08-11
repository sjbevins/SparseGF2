"""Exact-layer single-reference purification simulation and durable storage."""

from __future__ import annotations

import hashlib
import io
import math
import os
import time
import uuid
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from studies.prl_production.campaign import (
    GRAPH_K,
    MASTER_SEED,
    MEAN_DEGREE,
    SCHEMA_VERSION,
    SCRAMBLE_DEPTH,
    TMAX_FACTOR,
)

from sparsegf2 import entanglement_entropy
from sparsegf2.circuits._clifford_table import SP4_SIZE, sp4_table
from sparsegf2.circuits.graphs import _ws_rewire_edges
from sparsegf2.circuits.picture import setup_picture
from sparsegf2.core.sparse_tableau import _build_2q_lut

_PLACE_STREAM_KEY = 0xB10C3
_MEAS_STREAM_KEY = 0x6D656173  # "meas"
_SCRAMBLE_STREAM_KEY = 0x73637262  # "scrb"
_ENGINE_NAME = "single_ref_exact_layer_v1"


@dataclass(frozen=True)
class PointSpec:
    """One independently writable production point."""

    n: int
    beta: float
    p: float
    n_graphs: int

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 6 or self.n % 2:
            raise ValueError(f"n must be an even integer >= 6; got {self.n!r}")
        if not (math.isfinite(self.beta) and 0.0 <= self.beta <= 1.0):
            raise ValueError(f"beta must lie in [0, 1]; got {self.beta!r}")
        if not (math.isfinite(self.p) and 0.0 <= self.p <= 1.0):
            raise ValueError(f"p must lie in [0, 1]; got {self.p!r}")
        if (
            isinstance(self.n_graphs, bool)
            or not isinstance(self.n_graphs, int)
            or self.n_graphs < 1
        ):
            raise ValueError(f"n_graphs must be a positive integer; got {self.n_graphs!r}")

    @property
    def cap(self) -> int:
        return TMAX_FACTOR * self.n

    @property
    def beta_key(self) -> int:
        return int(round(self.beta * 1_000_000_000))

    @property
    def p_key(self) -> int:
        return int(round(self.p * 1_000_000))


@dataclass(frozen=True)
class TrajectoryResult:
    """Exact first-passage result for one trajectory."""

    tau_p: int | None
    stop_layer: int
    event_observed: bool
    s_r_trace: tuple[int, ...] | None = None
    final_tableau_sha256: str | None = None


@dataclass(frozen=True)
class PointProgress:
    """Summary returned by a point worker."""

    point: PointSpec
    path: str
    completed: int
    events: int
    newly_completed: int
    elapsed_s: float

    @property
    def is_complete(self) -> bool:
        return self.completed == self.point.n_graphs


def _seed_words(point: PointSpec, graph_index: int, stream_key: int) -> list[int]:
    return [
        MASTER_SEED,
        point.n,
        point.beta_key,
        point.p_key,
        int(graph_index),
        stream_key,
    ]


@lru_cache(maxsize=1)
def _gate_luts() -> NDArray[np.uint8]:
    luts = np.ascontiguousarray([_build_2q_lut(matrix) for matrix in sp4_table()], dtype=np.uint8)
    luts.flags.writeable = False
    return luts


def _tableau_digest(sim) -> str:
    return hashlib.sha256(np.ascontiguousarray(sim.to_symplectic()).tobytes()).hexdigest()


def simulate_trajectory(
    point: PointSpec,
    graph_index: int,
    edges: NDArray[np.integer],
    *,
    execution: str = "batch",
    record_trace: bool = False,
    audit_tableau: bool = False,
    use_numba: bool | None = None,
    hybrid: bool = True,
) -> TrajectoryResult:
    """Simulate one trajectory and evaluate reference entropy after every layer."""
    if isinstance(graph_index, bool) or not isinstance(graph_index, (int, np.integer)):
        raise ValueError(f"graph_index must be an integer; got {graph_index!r}")
    graph_index = int(graph_index)
    if not 0 <= graph_index < point.n_graphs:
        raise ValueError(f"graph_index must lie in [0, {point.n_graphs}); got {graph_index}")
    if execution not in {"batch", "scalar"}:
        raise ValueError(f"execution must be 'batch' or 'scalar'; got {execution!r}")

    edge_array = np.asarray(edges, dtype=np.int64)
    expected_edges = point.n * GRAPH_K
    if edge_array.shape != (expected_edges, 2):
        raise ValueError(
            f"edges must have shape ({expected_edges}, 2) for C(n,{GRAPH_K}); "
            f"got {edge_array.shape}"
        )
    if (
        edge_array.min() < 0
        or edge_array.max() >= point.n
        or np.any(edge_array[:, 0] == edge_array[:, 1])
    ):
        raise ValueError("graph edges must be distinct-endpoint system-qubit pairs")

    place_rng = np.random.default_rng(_seed_words(point, graph_index, _PLACE_STREAM_KEY))
    meas_rng = np.random.default_rng(_seed_words(point, graph_index, _MEAS_STREAM_KEY))
    scramble_rng = np.random.default_rng(_seed_words(point, graph_index, _SCRAMBLE_STREAM_KEY))
    sim, spec = setup_picture(
        "single_ref",
        point.n,
        rng=meas_rng,
        use_numba=use_numba,
        hybrid=hybrid,
    )
    luts = _gate_luts()
    n_gates = point.n // 2

    for _ in range(SCRAMBLE_DEPTH):
        pairs = scramble_rng.permutation(point.n).reshape(n_gates, 2)
        cliff_indices = scramble_rng.integers(0, SP4_SIZE, size=n_gates, dtype=np.int64)
        if execution == "batch":
            sim._dispatch_2q_batch(pairs, cliff_indices, luts)
        else:
            for pair, cliff_index in zip(pairs, cliff_indices, strict=True):
                sim._dispatch_2q(
                    int(pair[0]),
                    int(pair[1]),
                    luts[int(cliff_index)],
                )

    initial_entropy = int(entanglement_entropy(sim, spec.reference_qubits))
    if initial_entropy != 1:
        raise RuntimeError(
            f"single reference must begin maximally mixed; got S(R)={initial_entropy}"
        )

    trace: list[int] | None = [] if record_trace else None
    for layer in range(1, point.cap + 1):
        edge_indices = place_rng.integers(0, len(edge_array), size=n_gates, dtype=np.int64)
        pairs = edge_array[edge_indices]
        cliff_indices = place_rng.integers(0, SP4_SIZE, size=n_gates, dtype=np.int64)
        if execution == "batch":
            sim._dispatch_2q_batch(pairs, cliff_indices, luts)
        else:
            for pair, cliff_index in zip(pairs, cliff_indices, strict=True):
                sim._dispatch_2q(
                    int(pair[0]),
                    int(pair[1]),
                    luts[int(cliff_index)],
                )

        measured = np.flatnonzero(meas_rng.random(point.n) < point.p).astype(np.int64, copy=False)
        if execution == "batch":
            sim._measure_z_batch(measured, rng=meas_rng)
        else:
            for qubit in measured:
                sim.measure_z(int(qubit), rng=meas_rng)

        # This read is deliberately unconditional.  The production observable
        # is the exact first zero of S(R) on the integer layer grid.
        s_r = int(entanglement_entropy(sim, spec.reference_qubits))
        if s_r not in {0, 1}:
            raise RuntimeError(f"single-qubit reference entropy must be binary; got {s_r}")
        if trace is not None:
            trace.append(s_r)
        if s_r == 0:
            return TrajectoryResult(
                tau_p=layer,
                stop_layer=layer,
                event_observed=True,
                s_r_trace=None if trace is None else tuple(trace),
                final_tableau_sha256=_tableau_digest(sim) if audit_tableau else None,
            )

    return TrajectoryResult(
        tau_p=None,
        stop_layer=point.cap,
        event_observed=False,
        s_r_trace=None if trace is None else tuple(trace),
        final_tableau_sha256=_tableau_digest(sim) if audit_tableau else None,
    )


def _beta_dir(beta_key: int) -> str:
    return f"b{beta_key:010d}"


def graph_bank_path(data_root: Path, point: PointSpec) -> Path:
    return (
        data_root
        / "single_ref"
        / "graphs"
        / f"g{point.n_graphs}"
        / f"n{point.n}_{_beta_dir(point.beta_key)}.npz"
    )


def point_path(data_root: Path, point: PointSpec) -> Path:
    return (
        data_root
        / "single_ref"
        / "points"
        / f"g{point.n_graphs}"
        / f"n{point.n}"
        / _beta_dir(point.beta_key)
        / f"p{point.p_key:06d}.npz"
    )


def _array_sha256(array: NDArray[np.generic]) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _write_deterministic_npz(path: Path, arrays: dict[str, object]) -> None:
    """Write a byte-reproducible NPZ and atomically replace the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("wb") as raw:
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
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _scalar(data: np.lib.npyio.NpzFile, key: str):
    value = data[key]
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar; got shape {value.shape}")
    return value.item()


def prepare_graph_bank(data_root: Path, point: PointSpec) -> str:
    """Create or validate the deterministic graph bank for one (n, beta)."""
    path = graph_bank_path(data_root, point)
    if path.exists():
        load_graph_bank(data_root, point)
        return str(path)

    edges = np.empty((point.n_graphs, point.n * GRAPH_K, 2), dtype=np.int32)
    for graph_index in range(point.n_graphs):
        generated = np.asarray(
            _ws_rewire_edges(point.n, GRAPH_K, point.beta, graph_index),
            dtype=np.int32,
        )
        if generated.shape != edges.shape[1:]:
            raise RuntimeError(
                f"graph {graph_index} has {len(generated)} edges; expected {point.n * GRAPH_K}"
            )
        edges[graph_index] = generated

    arrays: dict[str, object] = {
        "schema_version": np.int32(SCHEMA_VERSION),
        "engine": np.str_(_ENGINE_NAME),
        "n": np.int32(point.n),
        "k": np.int32(GRAPH_K),
        "mean_degree": np.float64(MEAN_DEGREE),
        "beta": np.float64(point.beta),
        "beta_key": np.int64(point.beta_key),
        "n_graphs": np.int32(point.n_graphs),
        "graph_index": np.arange(point.n_graphs, dtype=np.int32),
        "edges": edges,
        "edges_sha256": np.str_(_array_sha256(edges)),
    }
    _write_deterministic_npz(path, arrays)
    load_graph_bank(data_root, point)
    return str(path)


def load_graph_bank(data_root: Path, point: PointSpec) -> NDArray[np.int32]:
    """Load a graph bank after strict metadata and content validation."""
    path = graph_bank_path(data_root, point)
    with np.load(path, allow_pickle=False) as data:
        expected_scalars = {
            "schema_version": SCHEMA_VERSION,
            "engine": _ENGINE_NAME,
            "n": point.n,
            "k": GRAPH_K,
            "mean_degree": float(MEAN_DEGREE),
            "beta_key": point.beta_key,
            "n_graphs": point.n_graphs,
        }
        for key, expected in expected_scalars.items():
            actual = _scalar(data, key)
            if actual != expected:
                raise ValueError(f"{path}: {key}={actual!r}, expected {expected!r}")
        if not math.isclose(float(_scalar(data, "beta")), point.beta, abs_tol=5e-13):
            raise ValueError(f"{path}: beta metadata does not match {point.beta:g}")
        graph_index = np.asarray(data["graph_index"], dtype=np.int32)
        if not np.array_equal(graph_index, np.arange(point.n_graphs, dtype=np.int32)):
            raise ValueError(f"{path}: graph_index is not the canonical range")
        edges = np.asarray(data["edges"], dtype=np.int32)
        expected_shape = (point.n_graphs, point.n * GRAPH_K, 2)
        if edges.shape != expected_shape:
            raise ValueError(f"{path}: edges shape {edges.shape}, expected {expected_shape}")
        expected_hash = str(_scalar(data, "edges_sha256"))
        if _array_sha256(edges) != expected_hash:
            raise ValueError(f"{path}: graph-bank hash mismatch")
    return edges


def _new_point_arrays(point: PointSpec, *, record_traces: bool) -> dict[str, object]:
    arrays: dict[str, object] = {
        "schema_version": np.int32(SCHEMA_VERSION),
        "engine": np.str_(_ENGINE_NAME),
        "n": np.int32(point.n),
        "k": np.int32(GRAPH_K),
        "mean_degree": np.float64(MEAN_DEGREE),
        "beta": np.float64(point.beta),
        "beta_key": np.int64(point.beta_key),
        "p": np.float64(point.p),
        "p_key": np.int64(point.p_key),
        "n_graphs": np.int32(point.n_graphs),
        "tmax_factor": np.int32(TMAX_FACTOR),
        "t_max": np.int32(point.cap),
        "scramble_depth": np.int32(SCRAMBLE_DEPTH),
        "master_seed": np.int64(MASTER_SEED),
        "graph_index": np.arange(point.n_graphs, dtype=np.int32),
        "tau_p": np.full(point.n_graphs, -1, dtype=np.int32),
        "stop_layer": np.zeros(point.n_graphs, dtype=np.int32),
        "event_observed": np.zeros(point.n_graphs, dtype=np.uint8),
        "complete": np.zeros(point.n_graphs, dtype=np.uint8),
    }
    if record_traces:
        arrays["s_r_trace"] = np.full((point.n_graphs, point.cap), -1, dtype=np.int8)
    return arrays


def _load_point_arrays(path: Path, point: PointSpec, *, record_traces: bool) -> dict[str, object]:
    if not path.exists():
        return _new_point_arrays(point, record_traces=record_traces)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.array(data[key], copy=True) for key in data.files}

    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "engine": _ENGINE_NAME,
        "n": point.n,
        "k": GRAPH_K,
        "mean_degree": float(MEAN_DEGREE),
        "beta_key": point.beta_key,
        "p_key": point.p_key,
        "n_graphs": point.n_graphs,
        "tmax_factor": TMAX_FACTOR,
        "t_max": point.cap,
        "scramble_depth": SCRAMBLE_DEPTH,
        "master_seed": MASTER_SEED,
    }
    for key, expected in expected_scalars.items():
        actual = np.asarray(arrays[key]).item()
        if actual != expected:
            raise ValueError(f"{path}: {key}={actual!r}, expected {expected!r}")
    for key, expected in {"beta": point.beta, "p": point.p}.items():
        if not math.isclose(float(np.asarray(arrays[key]).item()), expected, abs_tol=5e-13):
            raise ValueError(f"{path}: {key} metadata does not match {expected:g}")

    for key in ("graph_index", "tau_p", "stop_layer", "event_observed", "complete"):
        if np.asarray(arrays[key]).shape != (point.n_graphs,):
            raise ValueError(f"{path}: {key} has invalid shape {np.asarray(arrays[key]).shape}")
    if not np.array_equal(
        np.asarray(arrays["graph_index"]), np.arange(point.n_graphs, dtype=np.int32)
    ):
        raise ValueError(f"{path}: graph_index is not the canonical range")
    if record_traces and "s_r_trace" not in arrays:
        raise ValueError(f"{path}: existing point lacks requested S(R) traces")
    if "s_r_trace" in arrays and np.asarray(arrays["s_r_trace"]).shape != (
        point.n_graphs,
        point.cap,
    ):
        raise ValueError(f"{path}: s_r_trace has invalid shape")

    complete = np.asarray(arrays["complete"], dtype=np.uint8)
    event = np.asarray(arrays["event_observed"], dtype=np.uint8)
    tau = np.asarray(arrays["tau_p"], dtype=np.int32)
    stop = np.asarray(arrays["stop_layer"], dtype=np.int32)
    if np.any((complete != 0) & (complete != 1)) or np.any((event != 0) & (event != 1)):
        raise ValueError(f"{path}: completion/event flags must be binary")
    done = complete.astype(bool)
    observed = done & event.astype(bool)
    censored = done & ~event.astype(bool)
    if np.any(tau[observed] != stop[observed]) or np.any(tau[observed] < 1):
        raise ValueError(f"{path}: observed tau_p and stop_layer are inconsistent")
    if np.any(tau[censored] != -1) or np.any(stop[censored] != point.cap):
        raise ValueError(f"{path}: censored rows are inconsistent")
    if np.any(stop[~done] != 0):
        raise ValueError(f"{path}: incomplete rows must have stop_layer=0")
    return arrays


def run_point(
    data_root: Path | str,
    point: PointSpec,
    *,
    checkpoint_every: int = 25,
    record_traces: bool = False,
    max_new_trajectories: int | None = None,
) -> PointProgress:
    """Run or resume one point, checkpointing deterministic arrays atomically."""
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    if max_new_trajectories is not None and max_new_trajectories < 1:
        raise ValueError("max_new_trajectories must be positive when provided")
    if record_traces and point.n_graphs > 32:
        raise ValueError("full S(R) traces are restricted to audit runs with <= 32 trajectories")

    started = time.perf_counter()
    data_root = Path(data_root)
    prepare_graph_bank(data_root, point)
    graph_edges = load_graph_bank(data_root, point)
    path = point_path(data_root, point)
    arrays = _load_point_arrays(path, point, record_traces=record_traces)
    complete = np.asarray(arrays["complete"], dtype=np.uint8)
    pending = np.flatnonzero(complete == 0)
    if max_new_trajectories is not None:
        pending = pending[:max_new_trajectories]

    newly_completed = 0
    for graph_index_value in pending:
        graph_index = int(graph_index_value)
        result = simulate_trajectory(
            point,
            graph_index,
            graph_edges[graph_index],
            record_trace=record_traces,
        )
        np.asarray(arrays["tau_p"])[graph_index] = -1 if result.tau_p is None else result.tau_p
        np.asarray(arrays["stop_layer"])[graph_index] = result.stop_layer
        np.asarray(arrays["event_observed"])[graph_index] = int(result.event_observed)
        if record_traces:
            trace = np.asarray(result.s_r_trace, dtype=np.int8)
            np.asarray(arrays["s_r_trace"])[graph_index, : len(trace)] = trace
        # Mark complete last so a saved row is never half-populated.
        complete[graph_index] = 1
        newly_completed += 1
        if newly_completed % checkpoint_every == 0:
            _write_deterministic_npz(path, arrays)

    if newly_completed or not path.exists():
        _write_deterministic_npz(path, arrays)
    completed = int(complete.sum())
    events = int(np.asarray(arrays["event_observed"], dtype=np.uint8)[complete == 1].sum())
    return PointProgress(
        point=point,
        path=str(path),
        completed=completed,
        events=events,
        newly_completed=newly_completed,
        elapsed_s=time.perf_counter() - started,
    )


__all__ = [
    "PointProgress",
    "PointSpec",
    "TrajectoryResult",
    "graph_bank_path",
    "load_graph_bank",
    "point_path",
    "prepare_graph_bank",
    "run_point",
    "simulate_trajectory",
]
