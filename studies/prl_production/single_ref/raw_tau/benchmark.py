"""Bounded compute and process-scaling benchmark for the raw-tau v2 engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from studies.prl_production.single_ref.benchmark import (
    THREAD_LIMIT_VARIABLES,
    TrialMetrics,
    choose_circuit_tile,
    summarize_scaling,
)
from studies.prl_production.single_ref.raw_tau.config import (
    current_scientific_environment_contract,
    source_fingerprint_sha256,
)
from studies.prl_production.single_ref.raw_tau.engine import (
    ENGINE_VERSION,
    RawTauTrajectoryResult,
    _gate_luts,
    simulate_trajectory,
)
from studies.prl_production.sweep_spec import (
    GraphCollectionGridSpec,
    GraphParameterGrid,
    ProbabilityGrid,
    ScientificEnvironmentContract,
    SingleReferenceProtocolSpec,
    SingleReferenceSweepSpec,
    TauWorkUnit,
)

PCG64_CONTRACT = "PCG64"
ADMISSION_PILOT_TRAJECTORIES = 3
ADMISSION_SAFETY_FACTOR = 1.25


@dataclass(frozen=True, slots=True)
class ChunkMetrics:
    """Counts returned by one process-pool task."""

    pid: int
    cpu_seconds: float
    trajectories: int
    layers: int
    scramble_gates: int
    dynamics_gates: int
    measurement_trials: int
    measurements: int
    events: int
    peak_rss_bytes: int


@dataclass(frozen=True, slots=True)
class AdmissionPilot:
    """Deterministic capped-work timing guard used before process spawning."""

    circuit_indices: tuple[int, ...]
    requested_p: str
    capped_p: str
    t_max: int
    requested_stop_layers: tuple[int, ...]
    capped_stop_layers: tuple[int, ...]
    capped_event_observed: tuple[bool, ...]
    requested_seconds: tuple[float, ...]
    capped_seconds: tuple[float, ...]
    maximum_observed_seconds: float
    safety_factor: float
    seconds_per_trajectory_guard: float
    estimated_one_worker_repeat_seconds: float


_WORKER_WORK: TauWorkUnit | None = None
_WORKER_EDGES: NDArray[np.int64] | None = None
_WORKER_EXECUTION = "batch"
_WORKER_HYBRID = True


def _verified_environment_contract() -> ScientificEnvironmentContract:
    """Pin and verify the literal RNG/dependency contract used by trajectories."""
    default_bit_generator = type(np.random.default_rng(0).bit_generator).__name__
    explicit_bit_generator = np.random.PCG64.__name__
    if default_bit_generator != PCG64_CONTRACT or explicit_bit_generator != PCG64_CONTRACT:
        raise RuntimeError(
            "raw-tau benchmark requires NumPy default_rng and np.random.PCG64 to use "
            f"the literal {PCG64_CONTRACT} contract; got default_rng={default_bit_generator!r}, "
            f"PCG64={explicit_bit_generator!r}"
        )
    contract = current_scientific_environment_contract()
    if contract.bit_generator != PCG64_CONTRACT:
        raise RuntimeError(
            "scientific environment contract does not pin the literal PCG64 bit generator"
        )
    return contract


def _circulant_c2_edges(n: int) -> NDArray[np.int64]:
    """Return the undirected edge set of the local degree-four circulant."""

    if n < 5:
        raise ValueError("the built-in C(n,2) benchmark graph requires n >= 5")
    edges = {
        tuple(sorted((vertex, (vertex + distance) % n)))
        for vertex in range(n)
        for distance in (1, 2)
    }
    return np.asarray(sorted(edges), dtype=np.int64)


def make_benchmark_work(
    *,
    n: int,
    p: str,
    n_circuits: int,
    q_max: int,
    q_scramble: int,
    source_fingerprint: str | None = None,
    environment_contract: ScientificEnvironmentContract | None = None,
) -> TauWorkUnit:
    """Construct one fully fingerprinted, graph-family-neutral benchmark point."""

    graphs = GraphCollectionGridSpec(
        name="single-ref-benchmark",
        graph_family="circulant-c2-benchmark",
        generator_name="benchmark-circulant-c2",
        generator_version="1",
        sizes=(n,),
        parameter_grid=GraphParameterGrid(),
        graphs_per_cell=1,
        master_seed=8_108_202_026,
    )
    protocol = SingleReferenceProtocolSpec(
        n_circuits=n_circuits,
        q_scramble=q_scramble,
        q_max=q_max,
        p_grid=ProbabilityGrid(p, p, "1"),
        master_seed=8_118_202_026,
    )
    resolved_source = (
        source_fingerprint_sha256() if source_fingerprint is None else source_fingerprint
    )
    resolved_environment = (
        _verified_environment_contract() if environment_contract is None else environment_contract
    )
    if resolved_environment.bit_generator != PCG64_CONTRACT:
        raise ValueError(f"benchmark environment must pin {PCG64_CONTRACT}")
    sweep = SingleReferenceSweepSpec(
        name="single-ref-benchmark",
        graph_collection_sha256=graphs.specification_sha256,
        source_fingerprint_sha256=resolved_source,
        environment_contract=resolved_environment,
        protocol=protocol,
    )
    return next(sweep.work_units(graphs))


def run_admission_pilot(
    work: TauWorkUnit,
    edges: NDArray[np.int64],
    *,
    execution: str,
    hybrid: bool,
    pilot_trajectories: int = ADMISSION_PILOT_TRAJECTORIES,
    safety_factor: float = ADMISSION_SAFETY_FACTOR,
    max_estimated_seconds_per_repeat: float | None = None,
) -> tuple[RawTauTrajectoryResult, AdmissionPilot]:
    """Profile fixed requested-p and capped p=0 trajectories conservatively."""

    if (
        isinstance(pilot_trajectories, bool)
        or not isinstance(pilot_trajectories, int)
        or pilot_trajectories < 1
    ):
        raise ValueError("pilot_trajectories must be a positive integer")
    if not math.isfinite(safety_factor) or safety_factor < 1.0:
        raise ValueError("safety_factor must be finite and at least one")
    if max_estimated_seconds_per_repeat is not None and (
        isinstance(max_estimated_seconds_per_repeat, bool)
        or not math.isfinite(max_estimated_seconds_per_repeat)
        or max_estimated_seconds_per_repeat <= 0.0
    ):
        raise ValueError("max_estimated_seconds_per_repeat must be finite and positive")
    circuit_indices = tuple(range(min(pilot_trajectories, work.protocol.n_circuits)))
    capped_work = (
        work
        if work.p_decimal == "0"
        else make_benchmark_work(
            n=work.cell.n,
            p="0",
            n_circuits=work.protocol.n_circuits,
            q_max=work.protocol.q_max,
            q_scramble=work.protocol.q_scramble,
        )
    )
    warmup_work = (
        work
        if work.p_decimal == "1"
        else make_benchmark_work(
            n=work.cell.n,
            p="1",
            n_circuits=work.protocol.n_circuits,
            q_max=work.protocol.q_max,
            q_scramble=work.protocol.q_scramble,
        )
    )

    # A p=1 first-layer warm-up compiles the shared trajectory/gate kernels
    # without spending a capped trajectory. The first timed p=0 trajectory is
    # therefore the earliest capped run and immediate conservative check.
    simulate_trajectory(
        warmup_work,
        0,
        circuit_indices[0],
        edges,
        execution=execution,
        hybrid=hybrid,
    )

    def profile(point: TauWorkUnit, circuit_index: int) -> RawTauTrajectoryResult:
        result = simulate_trajectory(
            point,
            0,
            circuit_index,
            edges,
            execution=execution,
            hybrid=hybrid,
            profile=True,
        )
        if result.timings is None:
            raise AssertionError("profiled trajectory did not return stage timings")
        if not math.isfinite(result.timings.total_s) or result.timings.total_s <= 0.0:
            raise RuntimeError("profiled trajectory returned a nonpositive or nonfinite time")
        return result

    cap = work.protocol.t_max(work.cell.n)
    requested_results_list: list[RawTauTrajectoryResult] = []
    capped_results_list: list[RawTauTrajectoryResult] = []

    def enforce_time_cap() -> None:
        if max_estimated_seconds_per_repeat is None:
            return
        observed = [
            result.timings.total_s for result in (*requested_results_list, *capped_results_list)
        ]
        estimated = safety_factor * max(observed) * work.protocol.n_circuits
        if estimated > max_estimated_seconds_per_repeat:
            raise SystemExit(
                f"conservative pilot estimates one-worker repetition at {estimated:.2f} s, "
                "above the "
                f"{max_estimated_seconds_per_repeat:.2f} s safety cap; reduce --circuits, "
                "--n, or --q-max, or explicitly raise the cap"
            )

    for index in circuit_indices:
        capped_result = profile(capped_work, index)
        if (
            capped_result.event_observed
            or capped_result.tau_p is not None
            or capped_result.stop_layer != cap
        ):
            raise AssertionError("p=0 admission trajectory did not remain capped through T_max")
        capped_results_list.append(capped_result)
        enforce_time_cap()
        if capped_work is work:
            requested_results_list.append(capped_result)
        else:
            requested_results_list.append(profile(work, index))
            enforce_time_cap()

    requested_results = tuple(requested_results_list)
    capped_results = tuple(capped_results_list)

    requested_seconds = tuple(result.timings.total_s for result in requested_results)
    capped_seconds = tuple(result.timings.total_s for result in capped_results)
    maximum_observed = max((*requested_seconds, *capped_seconds))
    guard = safety_factor * maximum_observed
    pilot = AdmissionPilot(
        circuit_indices=circuit_indices,
        requested_p=work.p_decimal,
        capped_p="0",
        t_max=cap,
        requested_stop_layers=tuple(result.stop_layer for result in requested_results),
        capped_stop_layers=tuple(result.stop_layer for result in capped_results),
        capped_event_observed=tuple(result.event_observed for result in capped_results),
        requested_seconds=requested_seconds,
        capped_seconds=capped_seconds,
        maximum_observed_seconds=maximum_observed,
        safety_factor=safety_factor,
        seconds_per_trajectory_guard=guard,
        estimated_one_worker_repeat_seconds=guard * work.protocol.n_circuits,
    )
    return requested_results[0], pilot


def _peak_rss_bytes() -> int:
    """Return a best-effort process peak RSS without making psutil a dependency."""

    try:
        import psutil
    except ImportError:
        return 0
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def _worker_initializer(
    work: TauWorkUnit,
    edges: NDArray[np.int64],
    execution: str,
    hybrid: bool,
) -> None:
    """Install immutable worker state, thread limits, LUTs, and JIT signatures."""

    for name in THREAD_LIMIT_VARIABLES:
        if os.environ.get(name) != "1":
            raise RuntimeError(f"worker inherited {name}={os.environ.get(name)!r}, expected '1'")
    import numba

    numba.set_num_threads(1)
    if numba.get_num_threads() != 1:
        raise RuntimeError("Numba worker thread limit is not one")
    if type(np.random.default_rng(0).bit_generator).__name__ != PCG64_CONTRACT:
        raise RuntimeError(f"benchmark worker does not satisfy the {PCG64_CONTRACT} RNG contract")
    global _WORKER_WORK, _WORKER_EDGES, _WORKER_EXECUTION, _WORKER_HYBRID
    _WORKER_WORK = work
    _WORKER_EDGES = np.ascontiguousarray(edges, dtype=np.int64)
    _WORKER_EDGES.flags.writeable = False
    _WORKER_EXECUTION = execution
    _WORKER_HYBRID = hybrid
    _gate_luts()
    simulate_trajectory(
        work,
        0,
        0,
        _WORKER_EDGES,
        execution=execution,
        hybrid=hybrid,
    )


def _run_chunk(circuit_indices: tuple[int, ...]) -> ChunkMetrics:
    """Run one deterministic circuit-index tile inside an initialized worker."""

    if _WORKER_WORK is None or _WORKER_EDGES is None:
        raise RuntimeError("benchmark worker was not initialized")
    started = time.process_time()
    results = [
        simulate_trajectory(
            _WORKER_WORK,
            0,
            circuit_index,
            _WORKER_EDGES,
            execution=_WORKER_EXECUTION,
            hybrid=_WORKER_HYBRID,
        )
        for circuit_index in circuit_indices
    ]
    return ChunkMetrics(
        pid=os.getpid(),
        cpu_seconds=time.process_time() - started,
        trajectories=len(results),
        layers=sum(result.layers_executed for result in results),
        scramble_gates=sum(result.scramble_gates for result in results),
        dynamics_gates=sum(result.dynamic_gates for result in results),
        measurement_trials=sum(_WORKER_WORK.cell.n * result.layers_executed for result in results),
        measurements=sum(result.measurements for result in results),
        events=sum(result.event_observed for result in results),
        peak_rss_bytes=_peak_rss_bytes(),
    )


def _tiles(n_circuits: int, tile_size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(range(start, min(start + tile_size, n_circuits)))
        for start in range(0, n_circuits, tile_size)
    )


def _aggregate_trial(
    workers: int,
    wall_seconds: float,
    chunks: list[ChunkMetrics],
) -> TrialMetrics:
    peak_by_pid: dict[int, int] = {}
    for chunk in chunks:
        peak_by_pid[chunk.pid] = max(peak_by_pid.get(chunk.pid, 0), chunk.peak_rss_bytes)
    return TrialMetrics(
        workers=workers,
        wall_seconds=wall_seconds,
        cpu_seconds=sum(chunk.cpu_seconds for chunk in chunks),
        trajectories=sum(chunk.trajectories for chunk in chunks),
        layers=sum(chunk.layers for chunk in chunks),
        scramble_gates=sum(chunk.scramble_gates for chunk in chunks),
        dynamics_gates=sum(chunk.dynamics_gates for chunk in chunks),
        measurement_trials=sum(chunk.measurement_trials for chunk in chunks),
        measurements=sum(chunk.measurements for chunk in chunks),
        events=sum(chunk.events for chunk in chunks),
        peak_rss_bytes=sum(peak_by_pid.values()),
    )


def run_scaling_benchmark(
    work: TauWorkUnit,
    edges: NDArray[np.int64],
    *,
    worker_counts: tuple[int, ...],
    repetitions: int,
    target_task_seconds: float,
    seconds_per_trajectory: float,
    execution: str = "batch",
    hybrid: bool = True,
) -> tuple[tuple[TrialMetrics, ...], dict[int, float], dict[int, int]]:
    """Measure cold startup and steady strong scaling on a fixed circuit set."""

    if not worker_counts or len(set(worker_counts)) != len(worker_counts):
        raise ValueError("worker_counts must be nonempty and unique")
    if any(workers < 1 or workers > work.protocol.n_circuits for workers in worker_counts):
        raise ValueError("each worker count must lie in [1, n_circuits]")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    adaptive_tile = choose_circuit_tile(
        seconds_per_trajectory,
        n_circuits=work.protocol.n_circuits,
        target_seconds=target_task_seconds,
    )
    all_trials: list[TrialMetrics] = []
    cold_start: dict[int, float] = {}
    tile_sizes: dict[int, int] = {}
    context = mp.get_context("spawn")
    for workers in worker_counts:
        # Retain at least four tasks per worker when the circuit count permits it.
        load_balance_cap = max(1, math.ceil(work.protocol.n_circuits / (4 * workers)))
        tile_size = min(adaptive_tile, load_balance_cap)
        tiles = _tiles(work.protocol.n_circuits, tile_size)
        tile_sizes[workers] = tile_size
        cold_started = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_worker_initializer,
            initargs=(work, edges, execution, hybrid),
        ) as executor:
            # Exercise the already initialized pool once; neither JIT nor process
            # startup is included in the steady-state repetitions below.
            warm_tiles = tuple((index,) for index in range(workers))
            list(executor.map(_run_chunk, warm_tiles, chunksize=1))
            cold_start[workers] = time.perf_counter() - cold_started
            for _ in range(repetitions):
                started = time.perf_counter()
                chunks = list(executor.map(_run_chunk, tiles, chunksize=1))
                elapsed = time.perf_counter() - started
                all_trials.append(_aggregate_trial(workers, elapsed, chunks))
    return tuple(all_trials), cold_start, tile_sizes


def _thread_limits() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in THREAD_LIMIT_VARIABLES}


def _require_thread_limits() -> None:
    invalid = {name: value for name, value in _thread_limits().items() if value != "1"}
    if invalid:
        assignments = ", ".join(f"{name}={value!r}" for name, value in invalid.items())
        raise SystemExit(
            "numerical thread limits must be set before Python starts; use "
            f"benchmark_single_ref.ps1 ({assignments})"
        )


def _load_edges(path: Path | None, n: int) -> NDArray[np.int64]:
    if path is None:
        return _circulant_c2_edges(n)
    if path.suffix.lower() != ".npy":
        raise ValueError("--edges-npy must name one NumPy .npy array")
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--p", default="0")
    parser.add_argument("--q-max", type=int, default=1)
    parser.add_argument("--q-scramble", type=int, default=1)
    parser.add_argument("--circuits", type=int, default=16)
    parser.add_argument("--workers", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--target-task-seconds", type=float, default=0.5)
    parser.add_argument("--max-estimated-seconds-per-repeat", type=float, default=20.0)
    parser.add_argument("--execution", choices=("batch", "scalar"), default="batch")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--edges-npy", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.n < 5:
        parser.error("--n must be >= 5")
    if args.q_max < 1 or args.q_scramble < 0 or args.circuits < 1:
        parser.error("q_max and circuits must be positive; q_scramble must be non-negative")
    if not 1 <= args.repetitions <= 20:
        parser.error("--repetitions must lie in [1, 20]")
    if any(workers < 1 or workers > 32 for workers in args.workers):
        parser.error("--workers values must lie in [1, 32]")
    if max(args.workers) > args.circuits:
        parser.error("--circuits must be at least the largest worker count")
    if args.target_task_seconds <= 0 or args.max_estimated_seconds_per_repeat <= 0:
        parser.error("benchmark time limits must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    _require_thread_limits()
    args = parse_args(argv)
    edges = _load_edges(args.edges_npy, args.n)
    source_fingerprint = source_fingerprint_sha256()
    environment_contract = _verified_environment_contract()
    work = make_benchmark_work(
        n=args.n,
        p=args.p,
        n_circuits=args.circuits,
        q_max=args.q_max,
        q_scramble=args.q_scramble,
        source_fingerprint=source_fingerprint,
        environment_contract=environment_contract,
    )
    # Keep LUT construction out of both the admission pilot and steady worker
    # measurements. The pilot performs its own explicit untimed JIT warm-ups.
    _gate_luts()
    profile, admission = run_admission_pilot(
        work,
        edges,
        execution=args.execution,
        hybrid=not args.no_hybrid,
        max_estimated_seconds_per_repeat=args.max_estimated_seconds_per_repeat,
    )
    if profile.timings is None:
        raise AssertionError("profiled trajectory did not return stage timings")
    estimated_repeat = admission.estimated_one_worker_repeat_seconds
    if estimated_repeat > args.max_estimated_seconds_per_repeat:
        raise SystemExit(
            f"conservative pilot estimates one-worker repetition at {estimated_repeat:.2f} s, "
            "above the "
            f"{args.max_estimated_seconds_per_repeat:.2f} s safety cap; reduce --circuits, "
            "--n, or --q-max, or explicitly raise the cap"
        )
    trials, cold_start, tile_sizes = run_scaling_benchmark(
        work,
        edges,
        worker_counts=tuple(args.workers),
        repetitions=args.repetitions,
        target_task_seconds=args.target_task_seconds,
        seconds_per_trajectory=admission.seconds_per_trajectory_guard,
        execution=args.execution,
        hybrid=not args.no_hybrid,
    )
    payload = {
        "schema_version": 2,
        "engine_version": ENGINE_VERSION,
        "work_sha256": work.work_sha256,
        "source_fingerprint_sha256": source_fingerprint,
        "protocol_sha256": work.protocol.specification_sha256,
        "environment_contract": environment_contract.canonical_payload(),
        "environment_contract_sha256": environment_contract.specification_sha256,
        "edges_sha256": hashlib.sha256(np.ascontiguousarray(edges).tobytes()).hexdigest(),
        "case": {
            "n": args.n,
            "p": work.p_decimal,
            "q_max": args.q_max,
            "q_scramble": args.q_scramble,
            "circuits": args.circuits,
            "execution": args.execution,
            "hybrid": not args.no_hybrid,
            "n_edges": int(edges.shape[0]),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "numpy": np.__version__,
            "thread_limits": _thread_limits(),
        },
        "stage_profile": {
            "result": {
                key: value
                for key, value in asdict(profile).items()
                if key not in {"final_tableau_sha256", "timings"}
            },
            "timings": asdict(profile.timings),
        },
        "admission_pilot": asdict(admission),
        "cold_start_seconds": {str(key): value for key, value in cold_start.items()},
        "tile_sizes": {str(key): value for key, value in tile_sizes.items()},
        "trials": [asdict(trial) for trial in trials],
        "scaling": [asdict(row) for row in summarize_scaling(trials)],
    }
    text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())


__all__ = [
    "ADMISSION_PILOT_TRAJECTORIES",
    "ADMISSION_SAFETY_FACTOR",
    "PCG64_CONTRACT",
    "AdmissionPilot",
    "ChunkMetrics",
    "main",
    "make_benchmark_work",
    "run_admission_pilot",
    "run_scaling_benchmark",
]
