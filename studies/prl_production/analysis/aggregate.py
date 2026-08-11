"""Build preliminary live coverage and survival summaries for one run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from studies.prl_production.analysis.bootstrap import bootstrap_km_median
from studies.prl_production.analysis.survival import summarize_purification_point
from studies.prl_production.campaign import (
    GRAPH_K,
    MASTER_SEED,
    MEAN_DEGREE,
    SCHEMA_VERSION,
    SCRAMBLE_DEPTH,
    TMAX_FACTOR,
)
from studies.prl_production.single_ref.engine import PointSpec, point_path
from studies.prl_production.single_ref.shared_io import load_npz_snapshot

PRL_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PRL_ROOT / "manifests" / "runtime"
DEFAULT_ANALYSIS_ROOT = PRL_ROOT / "analysis" / "runs"

PRELIMINARY_LABEL = "PRELIMINARY"
_ENGINE_NAME = "single_ref_exact_layer_v1"
_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{16}")

PointStatus = Literal["absent", "partial", "complete"]


@dataclass(frozen=True, slots=True)
class CoverageRecord:
    """Validated live coverage for one manifest point."""

    point_index: int
    status: PointStatus
    n: int
    beta: float
    beta_key: int
    p: float
    p_key: int
    t_max: int
    n_graphs: int
    completed_trajectories: int
    observed_events: int
    censored_trajectories: int
    pending_trajectories: int
    path: str


@dataclass(frozen=True, slots=True)
class AggregationResult:
    """Locations and counts produced by one live aggregation pass."""

    run_id: str
    manifest_path: Path
    manifest_sha256: str
    output_dir: Path
    absent_points: int
    partial_points: int
    complete_points: int
    completed_trajectories: int
    total_trajectories: int


@dataclass(frozen=True, slots=True)
class _ResolvedManifest:
    run_id: str
    path: Path
    sha256: str
    data_root: Path
    n_graphs: int
    n_points: int
    n_trajectories: int
    record_traces: bool
    points: tuple[PointSpec, ...]


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer; got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")
    return value


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number; got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return result


def _manifest_point(raw: object, index: int, n_graphs: int) -> PointSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"point_order[{index}] must be an object")
    expected_keys = {"n", "beta", "p", "n_graphs"}
    if set(raw) != expected_keys:
        raise ValueError(
            f"point_order[{index}] keys must be {sorted(expected_keys)}; got {sorted(raw)}"
        )
    point = PointSpec(
        n=_integer(raw["n"], f"point_order[{index}].n", minimum=1),
        beta=_real(raw["beta"], f"point_order[{index}].beta"),
        p=_real(raw["p"], f"point_order[{index}].p"),
        n_graphs=_integer(raw["n_graphs"], f"point_order[{index}].n_graphs", minimum=1),
    )
    if point.n_graphs != n_graphs:
        raise ValueError(
            f"point_order[{index}].n_graphs={point.n_graphs}, expected manifest n_graphs={n_graphs}"
        )
    return point


def _load_manifest(path: Path, *, expected_run_id: str | None) -> _ResolvedManifest:
    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON manifest {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"manifest {path} must contain a JSON object")

    if _integer(payload.get("schema_version"), "schema_version") != 1:
        raise ValueError(f"manifest {path} has unsupported schema_version")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            f"manifest run_id must be 16 lowercase hexadecimal characters; got {run_id!r}"
        )
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError(f"manifest run_id={run_id!r}, expected {expected_run_id!r}")
    data_root_raw = payload.get("data_root")
    if not isinstance(data_root_raw, str) or not data_root_raw:
        raise ValueError("manifest data_root must be a nonempty path string")
    n_graphs = _integer(payload.get("n_graphs"), "n_graphs", minimum=1)
    n_points = _integer(payload.get("n_points"), "n_points")
    n_trajectories = _integer(payload.get("n_trajectories"), "n_trajectories")
    record_traces = payload.get("record_traces")
    if not isinstance(record_traces, bool):
        raise ValueError(f"record_traces must be boolean; got {record_traces!r}")
    point_order = payload.get("point_order")
    if not isinstance(point_order, list):
        raise ValueError("point_order must be a list")
    points = tuple(_manifest_point(raw, index, n_graphs) for index, raw in enumerate(point_order))
    if len(points) != n_points:
        raise ValueError(f"n_points={n_points}, but point_order contains {len(points)} entries")
    if n_trajectories != n_points * n_graphs:
        raise ValueError(
            f"n_trajectories={n_trajectories}, expected n_points*n_graphs={n_points * n_graphs}"
        )
    keys = [(point.n, point.beta_key, point.p_key) for point in points]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"point_order contains duplicate canonical points: {duplicates[:3]}")
    return _ResolvedManifest(
        run_id=run_id,
        path=path.resolve(),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        data_root=Path(data_root_raw),
        n_graphs=n_graphs,
        n_points=n_points,
        n_trajectories=n_trajectories,
        record_traces=record_traces,
        points=points,
    )


def _require_scalar(
    data: np.lib.npyio.NpzFile,
    key: str,
    dtype: np.dtype[Any],
) -> object:
    array = data[key]
    if array.shape != ():
        raise ValueError(f"{key} must be a scalar; got shape {array.shape}")
    if array.dtype != dtype:
        raise ValueError(f"{key} must have dtype {dtype}; got {array.dtype}")
    return array.item()


def _require_vector(
    data: np.lib.npyio.NpzFile,
    key: str,
    dtype: np.dtype[Any],
    length: int,
) -> np.ndarray:
    array = data[key]
    if array.shape != (length,):
        raise ValueError(f"{key} must have shape {(length,)}; got {array.shape}")
    if array.dtype != dtype:
        raise ValueError(f"{key} must have dtype {dtype}; got {array.dtype}")
    return np.asarray(array)


def _validate_point_file(
    path: Path,
    point: PointSpec,
    *,
    record_traces: bool,
) -> tuple[CoverageRecord, dict[str, np.ndarray]]:
    required_keys = {
        "schema_version",
        "engine",
        "n",
        "k",
        "mean_degree",
        "beta",
        "beta_key",
        "p",
        "p_key",
        "n_graphs",
        "tmax_factor",
        "t_max",
        "scramble_depth",
        "master_seed",
        "graph_index",
        "tau_p",
        "stop_layer",
        "event_observed",
        "complete",
    }
    expected_keys = required_keys | ({"s_r_trace"} if record_traces else set())
    try:
        with load_npz_snapshot(path) as data:
            if set(data.files) != expected_keys:
                missing = sorted(expected_keys - set(data.files))
                unexpected = sorted(set(data.files) - expected_keys)
                raise ValueError(f"NPZ keys differ; missing={missing}, unexpected={unexpected}")

            expected_scalars = {
                "schema_version": (np.dtype(np.int32), SCHEMA_VERSION),
                "n": (np.dtype(np.int32), point.n),
                "k": (np.dtype(np.int32), GRAPH_K),
                "mean_degree": (np.dtype(np.float64), float(MEAN_DEGREE)),
                "beta_key": (np.dtype(np.int64), point.beta_key),
                "p_key": (np.dtype(np.int64), point.p_key),
                "n_graphs": (np.dtype(np.int32), point.n_graphs),
                "tmax_factor": (np.dtype(np.int32), TMAX_FACTOR),
                "t_max": (np.dtype(np.int32), point.cap),
                "scramble_depth": (np.dtype(np.int32), SCRAMBLE_DEPTH),
                "master_seed": (np.dtype(np.int64), MASTER_SEED),
            }
            for key, (dtype, expected) in expected_scalars.items():
                actual = _require_scalar(data, key, dtype)
                if actual != expected:
                    raise ValueError(f"{key}={actual!r}, expected {expected!r}")
            engine = data["engine"]
            if engine.shape != () or engine.dtype.kind != "U" or engine.item() != _ENGINE_NAME:
                raise ValueError(f"engine metadata must equal {_ENGINE_NAME!r}")
            for key, expected in {"beta": point.beta, "p": point.p}.items():
                actual = float(_require_scalar(data, key, np.dtype(np.float64)))
                if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=5e-13):
                    raise ValueError(f"{key}={actual!r}, expected {expected!r}")

            graph_index = _require_vector(data, "graph_index", np.dtype(np.int32), point.n_graphs)
            tau = _require_vector(data, "tau_p", np.dtype(np.int32), point.n_graphs)
            stop = _require_vector(data, "stop_layer", np.dtype(np.int32), point.n_graphs)
            event = _require_vector(data, "event_observed", np.dtype(np.uint8), point.n_graphs)
            complete = _require_vector(data, "complete", np.dtype(np.uint8), point.n_graphs)
            if record_traces:
                trace = data["s_r_trace"]
                if trace.shape != (point.n_graphs, point.cap) or trace.dtype != np.dtype(np.int8):
                    raise ValueError(
                        "s_r_trace must have shape "
                        f"{(point.n_graphs, point.cap)} and dtype int8; "
                        f"got {trace.shape} and {trace.dtype}"
                    )

            if not np.array_equal(graph_index, np.arange(point.n_graphs, dtype=np.int32)):
                raise ValueError("graph_index is not the canonical range")
            if np.any((complete != 0) & (complete != 1)):
                raise ValueError("complete must contain only zero or one")
            if np.any((event != 0) & (event != 1)):
                raise ValueError("event_observed must contain only zero or one")
            done = complete == 1
            observed = done & (event == 1)
            censored = done & (event == 0)
            incomplete = ~done
            if (
                np.any(tau[incomplete] != -1)
                or np.any(stop[incomplete] != 0)
                or np.any(event[incomplete] != 0)
            ):
                raise ValueError("incomplete rows require tau_p=-1, stop_layer=0, event_observed=0")
            if (
                np.any(tau[observed] != stop[observed])
                or np.any(tau[observed] < 1)
                or np.any(tau[observed] > point.cap)
            ):
                raise ValueError(
                    f"observed complete rows require 1 <= tau_p == stop_layer <= {point.cap}"
                )
            if np.any(tau[censored] != -1) or np.any(stop[censored] != point.cap):
                raise ValueError(
                    f"censored complete rows require tau_p=-1 and stop_layer={point.cap}"
                )
            arrays = {
                "tau_p": np.array(tau, copy=True),
                "stop_layer": np.array(stop, copy=True),
                "event_observed": np.array(event, copy=True),
                "complete": np.array(complete, copy=True),
            }
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"invalid point file {path}: {exc}") from exc

    completed = int(arrays["complete"].sum())
    observed_count = int(arrays["event_observed"][arrays["complete"] == 1].sum())
    status: PointStatus = "complete" if completed == point.n_graphs else "partial"
    return (
        CoverageRecord(
            point_index=-1,
            status=status,
            n=point.n,
            beta=point.beta,
            beta_key=point.beta_key,
            p=point.p,
            p_key=point.p_key,
            t_max=point.cap,
            n_graphs=point.n_graphs,
            completed_trajectories=completed,
            observed_events=observed_count,
            censored_trajectories=completed - observed_count,
            pending_trajectories=point.n_graphs - completed,
            path=str(path),
        ),
        arrays,
    )


def _absent_record(index: int, point: PointSpec, path: Path) -> CoverageRecord:
    return CoverageRecord(
        point_index=index,
        status="absent",
        n=point.n,
        beta=point.beta,
        beta_key=point.beta_key,
        p=point.p,
        p_key=point.p_key,
        t_max=point.cap,
        n_graphs=point.n_graphs,
        completed_trajectories=0,
        observed_events=0,
        censored_trajectories=0,
        pending_trajectories=point.n_graphs,
        path=str(path),
    )


def _csv_text(fieldnames: list[str], rows: list[dict[str, object]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _float(value: float) -> str:
    return format(value, ".12g")


def _bootstrap_options(resamples: int, confidence: float) -> tuple[int, float]:
    if isinstance(resamples, bool) or not isinstance(resamples, int):
        raise ValueError(f"bootstrap_resamples must be an integer; got {resamples!r}")
    if resamples < 0:
        raise ValueError(f"bootstrap_resamples must be nonnegative; got {resamples}")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"bootstrap_confidence must be a real number; got {confidence!r}")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError(
            f"bootstrap_confidence must lie strictly between zero and one; got {confidence!r}"
        )
    return resamples, confidence


def _bootstrap_seed(run_id: str, point: PointSpec) -> int:
    """Derive a stable pointwise seed without depending on manifest order."""
    payload = f"{run_id}\0{point.n}\0{point.beta_key}\0{point.p_key}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


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


def _live_markdown(
    manifest: _ResolvedManifest,
    coverage: list[CoverageRecord],
    *,
    generated_unix: float,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
) -> str:
    status_counts = Counter(record.status for record in coverage)
    completed = sum(record.completed_trajectories for record in coverage)
    events = sum(record.observed_events for record in coverage)
    by_n: dict[int, Counter[str]] = defaultdict(Counter)
    for record in coverage:
        by_n[record.n][record.status] += 1
        by_n[record.n]["completed_trajectories"] += record.completed_trajectories
        by_n[record.n]["total_trajectories"] += record.n_graphs
    lines = [
        "# PRELIMINARY live single-reference analysis",
        "",
        "> **PRELIMINARY:** This live snapshot is incomplete and is not a final scientific result.",
        "",
        f"- Run ID: `{manifest.run_id}`",
        f"- Manifest: `{manifest.path}`",
        f"- Manifest SHA-256: `{manifest.sha256}`",
        f"- Complete points entering Kaplan-Meier summaries: {status_counts['complete']:,}",
        f"- Partial points excluded from summaries: {status_counts['partial']:,}",
        f"- Absent points excluded from summaries: {status_counts['absent']:,}",
        f"- Completed trajectories: {completed:,} / {manifest.n_trajectories:,}",
        f"- Observed purification events among completed trajectories: {events:,}",
        (
            "- Pointwise median bootstrap: disabled"
            if bootstrap_resamples == 0
            else "- Pointwise median bootstrap: "
            f"{bootstrap_resamples:,} resamples at confidence {_float(bootstrap_confidence)}"
        ),
        "",
        "Only fully complete point files enter `point_summary.csv` and the Kaplan-Meier summaries. "
        "Partial and absent points appear only in `coverage.csv`.",
        "",
        "| n | absent points | partial points | complete points | completed trajectories |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n in sorted(by_n):
        counts = by_n[n]
        lines.append(
            f"| {n} | {counts['absent']:,} | {counts['partial']:,} | "
            f"{counts['complete']:,} | {counts['completed_trajectories']:,} / "
            f"{counts['total_trajectories']:,} |"
        )
    lines.extend(["", f"Generated Unix time: {generated_unix:.6f}", ""])
    return "\n".join(lines)


def aggregate_manifest(
    manifest_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    expected_run_id: str | None = None,
    bootstrap_resamples: int = 0,
    bootstrap_confidence: float = 0.68,
) -> AggregationResult:
    """Validate every point and atomically publish a preliminary live snapshot."""
    bootstrap_resamples, bootstrap_confidence = _bootstrap_options(
        bootstrap_resamples,
        bootstrap_confidence,
    )
    manifest = _load_manifest(Path(manifest_path), expected_run_id=expected_run_id)
    output = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_ANALYSIS_ROOT / manifest.run_id / "live"
    )
    coverage: list[CoverageRecord] = []
    summary_rows: list[dict[str, object]] = []

    for index, point in enumerate(manifest.points):
        path = point_path(manifest.data_root, point)
        if not path.exists():
            coverage.append(_absent_record(index, point, path))
            continue
        record, arrays = _validate_point_file(
            path,
            point,
            record_traces=manifest.record_traces,
        )
        record = CoverageRecord(**{**asdict(record), "point_index": index})
        coverage.append(record)
        if record.status != "complete":
            continue
        analysis = summarize_purification_point(
            n=point.n,
            beta=point.beta,
            p=point.p,
            t_max=point.cap,
            tau_p=arrays["tau_p"],
            stop_layer=arrays["stop_layer"],
            event_observed=arrays["event_observed"],
        )
        summary = analysis.summary
        if bootstrap_resamples:
            bootstrap = bootstrap_km_median(
                arrays["stop_layer"],
                arrays["event_observed"],
                confidence=bootstrap_confidence,
                n_resamples=bootstrap_resamples,
                seed=_bootstrap_seed(manifest.run_id, point),
            )
            if bootstrap.central_median != summary.median_tau_p:
                raise RuntimeError("bootstrap and point summary disagree on the central median")
            median_ci_lower: int | str = (
                "" if bootstrap.lower_bound is None else bootstrap.lower_bound
            )
            median_ci_upper: int | str = (
                "" if bootstrap.upper_bound is None else bootstrap.upper_bound
            )
            median_ci_resolved = int(bootstrap.interval_resolved)
            bootstrap_resolved_fraction: float | str = _float(bootstrap.resolved_fraction)
            stored_bootstrap_confidence: float | str = _float(bootstrap.confidence)
        else:
            median_ci_lower = ""
            median_ci_upper = ""
            median_ci_resolved = 0
            bootstrap_resolved_fraction = ""
            stored_bootstrap_confidence = ""
        summary_rows.append(
            {
                "analysis_status": PRELIMINARY_LABEL,
                "run_id": manifest.run_id,
                "point_index": index,
                "n": summary.n,
                "beta": _float(summary.beta),
                "beta_key": point.beta_key,
                "p": _float(summary.p),
                "p_key": point.p_key,
                "t_max": summary.t_max,
                "n_trajectories": summary.n_trajectories,
                "n_events": summary.n_events,
                "n_censored": summary.n_censored,
                "event_fraction": _float(summary.event_fraction),
                "median_tau_p": "" if summary.median_tau_p is None else summary.median_tau_p,
                "median_resolved": int(summary.median_resolved),
                "median_ci_lower": median_ci_lower,
                "median_ci_upper": median_ci_upper,
                "median_ci_resolved": median_ci_resolved,
                "bootstrap_resolved_fraction": bootstrap_resolved_fraction,
                "bootstrap_resamples": bootstrap_resamples,
                "bootstrap_confidence": stored_bootstrap_confidence,
                "survival_at_cap": _float(summary.survival_at_cap),
            }
        )

    coverage_rows = [
        {
            "analysis_status": PRELIMINARY_LABEL,
            "run_id": manifest.run_id,
            **{
                key: _float(value) if key in {"beta", "p"} else value
                for key, value in asdict(record).items()
            },
        }
        for record in coverage
    ]
    coverage_fields = [
        "analysis_status",
        "run_id",
        "point_index",
        "status",
        "n",
        "beta",
        "beta_key",
        "p",
        "p_key",
        "t_max",
        "n_graphs",
        "completed_trajectories",
        "observed_events",
        "censored_trajectories",
        "pending_trajectories",
        "path",
    ]
    summary_fields = [
        "analysis_status",
        "run_id",
        "point_index",
        "n",
        "beta",
        "beta_key",
        "p",
        "p_key",
        "t_max",
        "n_trajectories",
        "n_events",
        "n_censored",
        "event_fraction",
        "median_tau_p",
        "median_resolved",
        "median_ci_lower",
        "median_ci_upper",
        "median_ci_resolved",
        "bootstrap_resolved_fraction",
        "bootstrap_resamples",
        "bootstrap_confidence",
        "survival_at_cap",
    ]
    generated_unix = time.time()
    # Complete the validation pass before replacing any prior live artifact.
    _atomic_text(output / "coverage.csv", _csv_text(coverage_fields, coverage_rows))
    _atomic_text(output / "point_summary.csv", _csv_text(summary_fields, summary_rows))
    _atomic_text(
        output / "LIVE_ANALYSIS.md",
        _live_markdown(
            manifest,
            coverage,
            generated_unix=generated_unix,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_confidence=bootstrap_confidence,
        ),
    )

    counts = Counter(record.status for record in coverage)
    return AggregationResult(
        run_id=manifest.run_id,
        manifest_path=manifest.path,
        manifest_sha256=manifest.sha256,
        output_dir=output.resolve(),
        absent_points=counts["absent"],
        partial_points=counts["partial"],
        complete_points=counts["complete"],
        completed_trajectories=sum(record.completed_trajectories for record in coverage),
        total_trajectories=manifest.n_trajectories,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the run selection and optional live-output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=0)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.68)
    args = parser.parse_args(argv)
    if _RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must be 16 lowercase hexadecimal characters")
    if args.bootstrap_resamples < 0:
        parser.error("--bootstrap-resamples must be nonnegative")
    if not math.isfinite(args.bootstrap_confidence) or not 0.0 < args.bootstrap_confidence < 1.0:
        parser.error("--bootstrap-confidence must lie strictly between zero and one")
    return args


def main(argv: list[str] | None = None) -> int:
    """Aggregate the selected immutable manifest once."""
    args = parse_args(argv)
    manifest_path = RUNTIME_ROOT / f"single_ref_{args.run_id}_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"run manifest not found: {manifest_path}")
    result = aggregate_manifest(
        manifest_path,
        output_dir=args.output,
        expected_run_id=args.run_id,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence=args.bootstrap_confidence,
    )
    print(
        f"{PRELIMINARY_LABEL}: {result.complete_points:,} complete points; "
        f"live analysis written to {result.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
