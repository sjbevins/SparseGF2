"""Bounded benchmark accounting for the single-reference production campaign.

The physics benchmark driver uses these helpers but lives beside the finalized
protocol-v2 leaf engine. This module deliberately imports only the standard
library, so a launcher can inspect and set numerical thread limits before NumPy
or Numba is imported.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

THREAD_LIMIT_VARIABLES = (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class WorkEstimate:
    """Exact upper-bound operation counts for a Cartesian campaign grid."""

    n_cells: int
    n_trajectories: int
    scramble_gates: int
    capped_layers: int
    capped_dynamics_gates: int
    capped_measurement_trials: int
    raw_tau_bytes: int


@dataclass(frozen=True)
class TrialMetrics:
    """One unaggregated steady-state throughput observation."""

    workers: int
    wall_seconds: float
    cpu_seconds: float
    trajectories: int
    layers: int
    scramble_gates: int
    dynamics_gates: int
    measurement_trials: int
    measurements: int
    events: int
    peak_rss_bytes: int = 0

    def __post_init__(self) -> None:
        integer_fields = (
            self.workers,
            self.trajectories,
            self.layers,
            self.scramble_gates,
            self.dynamics_gates,
            self.measurement_trials,
            self.measurements,
            self.events,
            self.peak_rss_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise TypeError("count fields must be integers")
        if self.workers < 1 or self.trajectories < 1:
            raise ValueError("workers and trajectories must be positive")
        if not (math.isfinite(self.wall_seconds) and self.wall_seconds > 0):
            raise ValueError("wall_seconds must be positive and finite")
        if not (math.isfinite(self.cpu_seconds) and self.cpu_seconds >= 0):
            raise ValueError("cpu_seconds must be non-negative and finite")
        if any(value < 0 for value in integer_fields[2:]):
            raise ValueError("operation counts must be non-negative")
        if self.measurements > self.measurement_trials:
            raise ValueError("measurements cannot exceed Bernoulli trials")
        if self.events > self.trajectories:
            raise ValueError("events cannot exceed trajectories")

    @property
    def trajectories_per_second(self) -> float:
        return self.trajectories / self.wall_seconds

    @property
    def layers_per_second(self) -> float:
        return self.layers / self.wall_seconds

    @property
    def effective_cores(self) -> float:
        return self.cpu_seconds / self.wall_seconds


@dataclass(frozen=True)
class ScalingSummary:
    """Median throughput and strong-scaling statistics at one worker count."""

    workers: int
    repetitions: int
    median_trajectories_per_second: float
    median_layers_per_second: float
    median_effective_cores: float
    median_peak_rss_bytes: float
    throughput_mad: float
    speedup: float
    efficiency: float


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def estimate_work(
    sizes: Iterable[int],
    *,
    graph_parameter_cells: int,
    n_graphs: int,
    n_p: int,
    n_circuits: int,
    q_max: int,
    q_scramble: int,
) -> WorkEstimate:
    """Return exact counts if every trajectory survives to its depth cap."""

    sizes_tuple = tuple(sizes)
    if not sizes_tuple:
        raise ValueError("sizes must not be empty")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in sizes_tuple):
        raise ValueError("sizes must contain integers >= 2")
    values = {
        "graph_parameter_cells": graph_parameter_cells,
        "n_graphs": n_graphs,
        "n_p": n_p,
        "n_circuits": n_circuits,
        "q_max": q_max,
    }
    for name, value in values.items():
        _positive_integer(value, name)
    _nonnegative_integer(q_scramble, "q_scramble")

    trajectories_per_size = graph_parameter_cells * n_graphs * n_p * n_circuits
    n_trajectories = len(sizes_tuple) * trajectories_per_size
    scramble_gates = trajectories_per_size * sum(q_scramble * n for n in sizes_tuple)
    capped_layers = trajectories_per_size * sum(q_max * n for n in sizes_tuple)
    capped_dynamics_gates = trajectories_per_size * sum(q_max * n * (n // 2) for n in sizes_tuple)
    capped_measurement_trials = trajectories_per_size * sum(q_max * n * n for n in sizes_tuple)
    return WorkEstimate(
        n_cells=len(sizes_tuple) * graph_parameter_cells,
        n_trajectories=n_trajectories,
        scramble_gates=scramble_gates,
        capped_layers=capped_layers,
        capped_dynamics_gates=capped_dynamics_gates,
        capped_measurement_trials=capped_measurement_trials,
        raw_tau_bytes=4 * n_trajectories,
    )


def choose_circuit_tile(
    seconds_per_trajectory: float,
    *,
    n_circuits: int,
    target_seconds: float = 1.0,
) -> int:
    """Choose a circuit tile targeting a bounded worker-task duration."""

    _positive_integer(n_circuits, "n_circuits")
    if not (math.isfinite(seconds_per_trajectory) and seconds_per_trajectory > 0):
        raise ValueError("seconds_per_trajectory must be positive and finite")
    if not (math.isfinite(target_seconds) and target_seconds > 0):
        raise ValueError("target_seconds must be positive and finite")
    return min(n_circuits, max(1, math.ceil(target_seconds / seconds_per_trajectory)))


def recommended_worker_grid(physical_cores: int) -> tuple[int, ...]:
    """Return the bounded strong-scaling grid for the available physical cores."""

    _positive_integer(physical_cores, "physical_cores")
    candidates = (1, 2, 4, 8, 12, 16)
    return tuple(
        sorted({*(value for value in candidates if value <= physical_cores), physical_cores})
    )


def summarize_scaling(trials: Iterable[TrialMetrics]) -> tuple[ScalingSummary, ...]:
    """Aggregate raw trials, using the one-worker median as the baseline."""

    grouped: dict[int, list[TrialMetrics]] = {}
    for trial in trials:
        grouped.setdefault(trial.workers, []).append(trial)
    if 1 not in grouped:
        raise ValueError("scaling trials require a one-worker baseline")
    baseline = statistics.median(item.trajectories_per_second for item in grouped[1])
    rows: list[ScalingSummary] = []
    for workers in sorted(grouped):
        values = grouped[workers]
        throughput = [item.trajectories_per_second for item in values]
        median_throughput = statistics.median(throughput)
        speedup = median_throughput / baseline
        rows.append(
            ScalingSummary(
                workers=workers,
                repetitions=len(values),
                median_trajectories_per_second=median_throughput,
                median_layers_per_second=statistics.median(
                    item.layers_per_second for item in values
                ),
                median_effective_cores=statistics.median(item.effective_cores for item in values),
                median_peak_rss_bytes=statistics.median(item.peak_rss_bytes for item in values),
                throughput_mad=statistics.median(
                    abs(value - median_throughput) for value in throughput
                ),
                speedup=speedup,
                efficiency=speedup / workers,
            )
        )
    return tuple(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bounded work-estimate command line."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--graph-parameter-cells", type=int, required=True)
    parser.add_argument("--graphs", type=int, required=True)
    parser.add_argument("--p-count", type=int, required=True)
    parser.add_argument("--circuits", type=int, required=True)
    parser.add_argument("--q-max", type=int, required=True)
    parser.add_argument("--q-scramble", type=int, required=True)
    parser.add_argument("--json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print a write-free upper-bound work estimate as JSON."""

    args = parse_args(argv)
    estimate = estimate_work(
        args.sizes,
        graph_parameter_cells=args.graph_parameter_cells,
        n_graphs=args.graphs,
        n_p=args.p_count,
        n_circuits=args.circuits,
        q_max=args.q_max,
        q_scramble=args.q_scramble,
    )
    payload = json.dumps(asdict(estimate), indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.json is not None:
        args.json.write_text(payload, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "THREAD_LIMIT_VARIABLES",
    "ScalingSummary",
    "TrialMetrics",
    "WorkEstimate",
    "choose_circuit_tile",
    "estimate_work",
    "main",
    "recommended_worker_grid",
    "summarize_scaling",
]
