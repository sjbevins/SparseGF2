"""Bounded, resume-safe process coordinator for generalized raw-tau sweeps."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import sys
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from studies.prl_production.single_ref.benchmark import THREAD_LIMIT_VARIABLES
from studies.prl_production.sweep_spec import ScientificEnvironmentContract, TauWorkUnit

from .catalog import RawTauCatalog
from .config import (
    ResolvedRawTauConfig,
    current_scientific_environment_contract,
    load_config,
)
from .engine import _gate_luts
from .providers import ProviderCell, prepare_edge_bank
from .storage import WorkUnitProgress, run_work_unit

SCIENTIFIC_MANIFEST_SCHEMA_VERSION = 3
RUNTIME_AUDIT_SCHEMA_VERSION = 1
_REPLACE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)


@dataclass(frozen=True, slots=True)
class SweepPlan:
    """Write-free exact work counts resolved from one configuration."""

    experiment_id: str
    collection_id: str
    cells: int
    p_values: int
    work_units: int
    graphs: int
    circuits_per_graph_p: int
    trajectories: int
    max_layers: int
    max_dynamic_gates: int
    scramble_gates: int
    measurement_trials: int
    raw_tau_bytes: int


def build_work_units(config: ResolvedRawTauConfig) -> tuple[tuple[ProviderCell, TauWorkUnit], ...]:
    """Resolve canonical ``(cell, p)`` tasks without reading trajectory data."""
    if config.provider.collection_sha256 != config.sweep.graph_collection_sha256:
        raise ValueError("graph provider does not match the experiment fingerprint")
    units: list[tuple[ProviderCell, TauWorkUnit]] = []
    for cell in config.provider.cells():
        for p_index, p_decimal in enumerate(config.sweep.protocol.p_grid.canonical_values):
            units.append(
                (
                    cell,
                    TauWorkUnit(
                        experiment_sha256=config.sweep.specification_sha256,
                        protocol=config.sweep.protocol,
                        cell=cell.spec,
                        graphs_per_cell=cell.graphs_per_cell,
                        p_index=p_index,
                        p_decimal=p_decimal,
                    ),
                )
            )
    return tuple(units)


def make_plan(config: ResolvedRawTauConfig) -> SweepPlan:
    """Compute exact capped operation counts for the resolved Cartesian grid."""
    cells = config.provider.cells()
    n_p = len(config.sweep.protocol.p_grid.canonical_values)
    n_circuits = config.sweep.protocol.n_circuits
    trajectories = sum(cell.graphs_per_cell * n_p * n_circuits for cell in cells)
    max_layers = sum(
        cell.graphs_per_cell * n_p * n_circuits * config.sweep.protocol.t_max(cell.n)
        for cell in cells
    )
    max_dynamic_gates = sum(
        cell.graphs_per_cell
        * n_p
        * n_circuits
        * config.sweep.protocol.t_max(cell.n)
        * (cell.n // 2)
        for cell in cells
    )
    scramble_gates = sum(
        cell.graphs_per_cell * n_p * n_circuits * config.sweep.protocol.scramble_gate_count(cell.n)
        for cell in cells
    )
    measurement_trials = sum(
        cell.graphs_per_cell * n_p * n_circuits * config.sweep.protocol.t_max(cell.n) * cell.n
        for cell in cells
    )
    graph_count = sum(cell.graphs_per_cell for cell in cells)
    return SweepPlan(
        experiment_id=config.sweep.experiment_id,
        collection_id=config.provider.collection_id,
        cells=len(cells),
        p_values=n_p,
        work_units=len(cells) * n_p,
        graphs=graph_count,
        circuits_per_graph_p=n_circuits,
        trajectories=trajectories,
        max_layers=max_layers,
        max_dynamic_gates=max_dynamic_gates,
        scramble_gates=scramble_gates,
        measurement_trials=measurement_trials,
        raw_tau_bytes=4 * trajectories,
    )


def graph_source_summary(config: ResolvedRawTauConfig) -> dict[str, object]:
    """Return a bounded, human-reviewable summary of every canonical graph cell."""

    cells = config.provider.cells()
    parameter_values: dict[str, dict[str, object]] = {}
    cell_digest = hashlib.sha256(b"raw_tau_graph_cell_summary_v1\0")
    for cell in cells:
        payload = {
            "cell_index": cell.spec.cell_index,
            "n": cell.n,
            "parameters": cell.parameters,
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        cell_digest.update(len(encoded).to_bytes(8, "big"))
        cell_digest.update(encoded)
        for key, value in cell.parameters.items():
            token = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
            parameter_values.setdefault(key, {})[token] = value

    first = cells[0]
    provenance_sets = {
        "generator_contract_sha256": {cell.generator_contract_sha256 for cell in cells},
        "generator_name": {cell.generator_name for cell in cells},
        "generator_version": {cell.generator_version for cell in cells},
        "graph_family": {cell.graph_family for cell in cells},
        "graphs_per_cell": {cell.graphs_per_cell for cell in cells},
    }
    if any(len(values) != 1 for values in provenance_sets.values()):
        raise ValueError("one graph provider returned internally inconsistent cell provenance")
    values_summary = {
        key: [values[token] for token in sorted(values)]
        for key, values in sorted(parameter_values.items())
    }
    summary: dict[str, object] = {
        "cell_count": len(cells),
        "cell_definition_sha256": cell_digest.hexdigest(),
        "parameter_values": values_summary,
        "sizes": sorted({cell.n for cell in cells}),
        "source_kind": type(config.provider).__name__,
        **{key: next(iter(values)) for key, values in provenance_sets.items()},
    }
    if first.graph_family == "watts_strogatz":
        k_values = {int(cell.parameters["k"]) for cell in cells}
        if len(k_values) != 1:
            raise ValueError("Watts-Strogatz cells must use one reviewed k value")
        graph_k = next(iter(k_values))
        summary["graph_k"] = graph_k
        summary["mean_degree"] = 2 * graph_k
    return summary


def plan_review_payload(config: ResolvedRawTauConfig, plan: SweepPlan) -> dict[str, object]:
    """Return the complete write-free scientific identity for human review."""

    return {
        "collection_id": config.provider.collection_id,
        "collection_sha256": config.provider.collection_sha256,
        "environment_contract": config.sweep.environment_contract.canonical_payload(),
        "environment_contract_sha256": (config.sweep.environment_contract.specification_sha256),
        "exact_p_values": list(config.sweep.protocol.p_grid.canonical_values),
        "experiment_id": config.sweep.experiment_id,
        "experiment_sha256": config.sweep.specification_sha256,
        "graph_source_summary": graph_source_summary(config),
        "plan": asdict(plan),
        "protocol": config.sweep.protocol.canonical_payload(),
        "protocol_sha256": config.sweep.protocol.specification_sha256,
        "source_fingerprint_sha256": config.sweep.source_fingerprint_sha256,
    }


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        for delay in (*_REPLACE_RETRY_DELAYS, None):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if delay is None:
                    raise
                time.sleep(delay)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _lock_file_nonblocking(handle, path: Path) -> None:
    """Acquire one kernel-held exclusive lock without consulting marker contents."""
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover - exercised on POSIX CI
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise RuntimeError(f"raw-tau runner lock is already held: {path}") from exc


def _unlock_file(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:  # pragma: no cover - exercised on POSIX CI
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_lock_marker(handle, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    handle.seek(0)
    handle.truncate(0)
    handle.write(encoded)
    os.fsync(handle.fileno())


@contextlib.contextmanager
def _runner_lock(data_root: Path, experiment_id: str) -> Iterator[None]:
    runtime_root = data_root / "single_ref" / "raw_tau" / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    path = runtime_root / f"{experiment_id}.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    if os.fstat(handle.fileno()).st_size == 0:
        handle.write(b"\n")
        os.fsync(handle.fileno())
    lock_id = uuid.uuid4().hex
    try:
        _lock_file_nonblocking(handle, path)
        acquired_at = (
            dt.datetime.now(dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
        )
        marker: dict[str, object] = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "lock_id": lock_id,
            "pid": os.getpid(),
            "state": "locked",
            "acquired_at_utc": acquired_at,
            "released_at_utc": None,
        }
        _write_lock_marker(handle, marker)
        try:
            yield
        finally:
            marker["state"] = "released"
            marker["released_at_utc"] = (
                dt.datetime.now(dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
            )
            _write_lock_marker(handle, marker)
            _unlock_file(handle)
    finally:
        handle.close()


def _set_worker_thread_limits() -> None:
    # Called before ProcessPoolExecutor is created, so Windows spawn imports
    # NumPy/Numba under these limits rather than trying to repair oversubscription later.
    for name in THREAD_LIMIT_VARIABLES:
        os.environ[name] = "1"


def _worker_initializer(expected_environment: ScientificEnvironmentContract) -> None:
    for name in THREAD_LIMIT_VARIABLES:
        if os.environ.get(name) != "1":
            raise RuntimeError(f"worker inherited {name}={os.environ.get(name)!r}, expected '1'")
    try:
        import numba

        numba.set_num_threads(1)
    except ImportError:  # pragma: no cover - numba is a hard package dependency
        pass
    actual_environment = current_scientific_environment_contract()
    if actual_environment != expected_environment:
        raise RuntimeError(
            "worker scientific environment differs from the experiment contract: "
            f"got {actual_environment.canonical_payload()!r}, expected "
            f"{expected_environment.canonical_payload()!r}"
        )
    _gate_luts()


def _run_task(
    data_root: str,
    sweep,
    work: TauWorkUnit,
    cell: ProviderCell,
    bank_path: str,
    checkpoint_every: int,
) -> WorkUnitProgress:
    return run_work_unit(
        data_root,
        sweep,
        work,
        cell,
        bank_path,
        checkpoint_every=checkpoint_every,
    )


def _balanced_p_order(
    units: list[tuple[ProviderCell, TauWorkUnit]],
) -> list[tuple[ProviderCell, TauWorkUnit]]:
    """Interleave low/high p tasks to avoid a tail of capped trajectories."""
    ordered = sorted(units, key=lambda item: item[1].p_index)
    out: list[tuple[ProviderCell, TauWorkUnit]] = []
    left, right = 0, len(ordered) - 1
    while left <= right:
        out.append(ordered[left])
        left += 1
        if left <= right:
            out.append(ordered[right])
            right -= 1
    return out


def _terminate_executor(executor: ProcessPoolExecutor) -> None:
    for process in tuple(getattr(executor, "_processes", {}).values()):
        process.terminate()
    executor.shutdown(wait=True, cancel_futures=True)


def _manifest_payload(config: ResolvedRawTauConfig, plan: SweepPlan) -> dict[str, Any]:
    """Return only immutable scientific identity and exact work counts."""
    return {
        "schema_version": SCIENTIFIC_MANIFEST_SCHEMA_VERSION,
        "manifest_kind": "single_ref_raw_tau_scientific",
        "experiment_id": config.sweep.experiment_id,
        "experiment_sha256": config.sweep.specification_sha256,
        "collection_id": config.provider.collection_id,
        "collection_sha256": config.provider.collection_sha256,
        "source_fingerprint_sha256": config.sweep.source_fingerprint_sha256,
        "environment_contract": config.sweep.environment_contract.canonical_payload(),
        "environment_contract_sha256": (config.sweep.environment_contract.specification_sha256),
        "graph_source_summary": graph_source_summary(config),
        "protocol": config.sweep.protocol.canonical_payload(),
        "plan": asdict(plan),
    }


def _manifest_scientific_core(payload: object, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: scientific manifest must be a JSON object")
    schema_version = payload.get("schema_version")
    if schema_version != SCIENTIFIC_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(f"{path}: unsupported scientific manifest schema {schema_version!r}")
    allowed_keys = {
        "schema_version",
        "manifest_kind",
        "experiment_id",
        "experiment_sha256",
        "collection_id",
        "collection_sha256",
        "source_fingerprint_sha256",
        "environment_contract",
        "environment_contract_sha256",
        "graph_source_summary",
        "protocol",
        "plan",
    }
    if payload.get("manifest_kind") != "single_ref_raw_tau_scientific":
        raise RuntimeError(f"{path}: invalid scientific manifest kind")
    unknown = set(payload) - allowed_keys
    if unknown:
        raise RuntimeError(f"{path}: scientific manifest has unknown fields {sorted(unknown)}")
    keys = (
        "experiment_id",
        "experiment_sha256",
        "collection_id",
        "collection_sha256",
        "source_fingerprint_sha256",
        "environment_contract",
        "environment_contract_sha256",
        "graph_source_summary",
        "protocol",
        "plan",
    )
    missing = [key for key in keys if key not in payload]
    if missing:
        raise RuntimeError(f"{path}: scientific manifest is missing {missing}")
    return {key: payload[key] for key in keys}


def _ensure_scientific_manifest(path: Path, expected: dict[str, Any]) -> None:
    """Create the scientific manifest once or validate its immutable core."""
    if not path.exists():
        _atomic_json(path, expected)
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{path}: cannot read existing scientific manifest") from exc
    if _manifest_scientific_core(existing, path) != _manifest_scientific_core(expected, path):
        raise RuntimeError(f"{path}: existing scientific manifest conflicts with this config")


def _record_runtime_event(
    run_root: Path,
    config: ResolvedRawTauConfig,
    *,
    invocation_id: str,
    event: str,
    error: str | None = None,
) -> Path:
    """Append one immutable execution audit record outside the science manifest."""
    allowed = {"started", "complete", "failed", "interrupted"}
    if event not in allowed:
        raise ValueError(f"runtime event must be one of {sorted(allowed)}")
    recorded_at = dt.datetime.now(dt.UTC)
    payload = {
        "schema_version": RUNTIME_AUDIT_SCHEMA_VERSION,
        "invocation_id": invocation_id,
        "event": event,
        "recorded_at_utc": recorded_at.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "experiment_id": config.sweep.experiment_id,
        "config_path": str(config.config_path),
        "runtime": {
            "data_root": str(config.runtime.data_root),
            "workers": config.runtime.workers,
            "checkpoint_every": config.runtime.checkpoint_every,
            "max_in_flight": config.runtime.max_in_flight,
        },
        "environment": {
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "numba": importlib.metadata.version("numba"),
            "sparsegf2": importlib.metadata.version("sparsegf2"),
        },
        "error": error,
    }
    name = f"{time.time_ns():020d}_{event}_{invocation_id}.json"
    path = run_root / "runtime_history" / name
    _atomic_json(path, payload)
    return path


def _write_status(
    root: Path,
    config: ResolvedRawTauConfig,
    plan: SweepPlan,
    *,
    state: str,
    completed_work_units: int,
    completed_trajectories: int,
    current_cell: ProviderCell | None,
    started: float,
    error: str | None = None,
) -> None:
    elapsed = max(0.0, time.time() - started)
    payload = {
        "experiment_id": config.sweep.experiment_id,
        "state": state,
        "completed_work_units": completed_work_units,
        "total_work_units": plan.work_units,
        "completed_trajectories": completed_trajectories,
        "total_trajectories": plan.trajectories,
        "current_cell_index": None if current_cell is None else current_cell.spec.cell_index,
        "elapsed_seconds": elapsed,
        "error": error,
    }
    _atomic_json(root / "status.json", payload)
    cell_text = "none" if current_cell is None else str(current_cell.spec.cell_index)
    lines = [
        "# Raw single-reference sweep status",
        "",
        f"- Experiment: `{config.sweep.experiment_id}`",
        f"- State: **{state}**",
        f"- Work units: {completed_work_units:,} / {plan.work_units:,}",
        f"- Trajectories: {completed_trajectories:,} / {plan.trajectories:,}",
        f"- Current cell: {cell_text}",
        f"- Elapsed: {elapsed:.1f} s",
    ]
    if error is not None:
        lines.append(f"- Error: `{error}`")
    _atomic_text(root / "STATUS.md", "\n".join(lines) + "\n")


def run_sweep(config: ResolvedRawTauConfig) -> SweepPlan:
    """Run/resume all cells, completing every p shard before advancing cells."""
    if config.sweep.protocol.p_randomness_policy != "independent":
        raise ValueError("the production runner currently supports only independent p randomness")
    plan = make_plan(config)
    run_root = (
        config.runtime.data_root / "single_ref" / "raw_tau" / "runs" / config.sweep.experiment_id
    )
    manifest_path = run_root / "manifest.json"
    manifest = _manifest_payload(config, plan)

    started = time.time()
    invocation_id = uuid.uuid4().hex
    completed_work_units = 0
    completed_trajectories = 0
    _set_worker_thread_limits()
    with _runner_lock(config.runtime.data_root, config.sweep.experiment_id):
        _ensure_scientific_manifest(manifest_path, manifest)
        _record_runtime_event(
            run_root,
            config,
            invocation_id=invocation_id,
            event="started",
        )
        executor: ProcessPoolExecutor | None = None
        try:
            _write_status(
                run_root,
                config,
                plan,
                state="running",
                completed_work_units=0,
                completed_trajectories=0,
                current_cell=None,
                started=started,
            )
            all_units = build_work_units(config)
            with RawTauCatalog(config.runtime.data_root) as catalog:
                # The coordinator is the catalog's sole writer.  Register the
                # complete generic plan before any worker can publish results.
                catalog.register_plan(
                    config.sweep,
                    all_units,
                    expected_cell_count=len(config.provider.cells()),
                )
                context = mp.get_context("spawn")
                executor = ProcessPoolExecutor(
                    max_workers=config.runtime.workers,
                    mp_context=context,
                    initializer=_worker_initializer,
                    initargs=(config.sweep.environment_contract,),
                )
                by_cell: dict[int, list[tuple[ProviderCell, TauWorkUnit]]] = {}
                for item in all_units:
                    by_cell.setdefault(item[0].spec.cell_index, []).append(item)
                for cell in config.provider.cells():
                    bank_path = prepare_edge_bank(config.runtime.data_root, config.provider, cell)
                    pending_items = iter(_balanced_p_order(by_cell[cell.spec.cell_index]))
                    futures: dict[Future[WorkUnitProgress], TauWorkUnit] = {}
                    exhausted = False
                    while futures or not exhausted:
                        while not exhausted and len(futures) < config.runtime.max_in_flight:
                            try:
                                _cell, work = next(pending_items)
                            except StopIteration:
                                exhausted = True
                                break
                            future = executor.submit(
                                _run_task,
                                str(config.runtime.data_root),
                                config.sweep,
                                work,
                                cell,
                                bank_path,
                                config.runtime.checkpoint_every,
                            )
                            futures[future] = work
                        if not futures:
                            continue
                        done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                        for future in done:
                            work = futures.pop(future)
                            result = future.result()
                            if not result.is_complete:
                                raise RuntimeError(
                                    f"worker returned incomplete shard {result.path}"
                                )
                            # run_work_unit has semantically validated the NPZ;
                            # the coordinator independently verifies its path
                            # and digest before the terminal catalog commit.
                            catalog.mark_complete(work, result)
                            completed_work_units += 1
                            completed_trajectories += result.completed
                            _write_status(
                                run_root,
                                config,
                                plan,
                                state="running",
                                completed_work_units=completed_work_units,
                                completed_trajectories=completed_trajectories,
                                current_cell=cell,
                                started=started,
                            )
            executor.shutdown(wait=True, cancel_futures=False)
            executor = None
            _write_status(
                run_root,
                config,
                plan,
                state="complete",
                completed_work_units=completed_work_units,
                completed_trajectories=completed_trajectories,
                current_cell=None,
                started=started,
            )
            _record_runtime_event(
                run_root,
                config,
                invocation_id=invocation_id,
                event="complete",
            )
        except BaseException as exc:
            if executor is not None:
                _terminate_executor(executor)
            state = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            _write_status(
                run_root,
                config,
                plan,
                state=state,
                completed_work_units=completed_work_units,
                completed_trajectories=completed_trajectories,
                current_cell=None,
                started=started,
                error=f"{type(exc).__name__}: {exc}",
            )
            _record_runtime_event(
                run_root,
                config,
                invocation_id=invocation_id,
                event=state,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true", help="print the write-free plan (default)")
    action.add_argument("--run", action="store_true", help="execute/resume the sweep")
    parser.add_argument("--workers", type=int, help="override execution-only worker count")
    parser.add_argument(
        "--confirm-experiment-id",
        help="required with --run; must exactly equal the resolved experiment ID",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.workers is not None:
        if args.workers < 1:
            raise SystemExit("--workers must be positive")
        config = ResolvedRawTauConfig(
            provider=config.provider,
            sweep=config.sweep,
            runtime=type(config.runtime)(
                data_root=config.runtime.data_root,
                workers=args.workers,
                checkpoint_every=config.runtime.checkpoint_every,
                max_in_flight=max(args.workers, 2 * args.workers),
            ),
            config_path=config.config_path,
        )
    plan = make_plan(config)
    if not args.run:
        print(json.dumps(plan_review_payload(config, plan), indent=2, sort_keys=True))
        return 0
    if args.confirm_experiment_id != config.sweep.experiment_id:
        raise SystemExit("--run requires --confirm-experiment-id " + config.sweep.experiment_id)
    run_sweep(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RUNTIME_AUDIT_SCHEMA_VERSION",
    "SCIENTIFIC_MANIFEST_SCHEMA_VERSION",
    "SweepPlan",
    "build_work_units",
    "graph_source_summary",
    "main",
    "make_plan",
    "plan_review_payload",
    "run_sweep",
]
