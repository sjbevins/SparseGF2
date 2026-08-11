"""Exact-layer, graph-family-agnostic single-reference trajectory kernel."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from studies.prl_production.sweep_spec import TauWorkUnit

from sparsegf2.circuits._clifford_table import SP4_SIZE, sp4_table
from sparsegf2.core.observables import single_qubit_entropy
from sparsegf2.core.sparse_tableau import SparseGF2, _build_2q_lut

ENGINE_VERSION = "single_ref_raw_tau_exact_layer_v2"


@dataclass(frozen=True, slots=True)
class TrajectoryTimings:
    """Optional stage timings recorded only by explicit benchmark calls."""

    setup_s: float
    scramble_s: float
    dynamic_gates_s: float
    measurements_s: float
    entropy_s: float
    total_s: float


@dataclass(frozen=True, slots=True)
class RawTauTrajectoryResult:
    """First-passage result for one ``(graph, p, circuit)`` trajectory."""

    tau_p: int | None
    stop_layer: int
    event_observed: bool
    reference_system_qubit: int
    layers_executed: int
    scramble_gates: int
    dynamic_gates: int
    measurements: int
    final_tableau_sha256: str | None = None
    timings: TrajectoryTimings | None = None


@lru_cache(maxsize=1)
def _gate_luts() -> NDArray[np.uint8]:
    # Building the 720 phase-free Sp(4,F_2) representatives is deterministic.
    # Cache on the function without introducing another import-time global table.
    cached = np.ascontiguousarray([_build_2q_lut(matrix) for matrix in sp4_table()], dtype=np.uint8)
    if cached.shape != (SP4_SIZE, 16):
        raise RuntimeError(f"unexpected two-qubit Clifford LUT shape {cached.shape}")
    cached.flags.writeable = False
    return cached


def _tableau_digest(sim: SparseGF2) -> str:
    return hashlib.sha256(np.ascontiguousarray(sim.to_symplectic()).tobytes()).hexdigest()


def _validate_edges(n: int, edges: NDArray[np.integer] | object) -> NDArray[np.int64]:
    raw = np.asarray(edges)
    if raw.ndim != 2 or raw.shape[1:] != (2,) or raw.shape[0] < 1:
        raise ValueError(f"edges must have nonempty shape (m, 2); got {raw.shape}")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("edge endpoints must be integers")
    array = np.ascontiguousarray(raw, dtype=np.int64)
    if array.min() < 0 or array.max() >= n:
        raise ValueError(f"edge endpoints must lie in [0, n={n})")
    if np.any(array[:, 0] == array[:, 1]):
        raise ValueError("edges must join distinct system qubits")
    canonical = np.sort(array, axis=1)
    if np.unique(canonical, axis=0).shape[0] != canonical.shape[0]:
        raise ValueError("the graph edge set must not contain duplicate undirected edges")
    return array


def _uniform_distinct_pairs(
    rng: np.random.Generator,
    n: int,
    count: int,
) -> NDArray[np.int64]:
    """Sample ordered distinct system-qubit pairs uniformly with replacement."""
    if count == 0:
        return np.empty((0, 2), dtype=np.int64)
    first = rng.integers(0, n, size=count, dtype=np.int64)
    second = rng.integers(0, n - 1, size=count, dtype=np.int64)
    second += second >= first
    return np.column_stack((first, second))


def _pcg64(seed: int) -> np.random.Generator:
    """Construct the protocol's explicitly pinned NumPy bit generator."""

    return np.random.Generator(np.random.PCG64(seed))


def _apply_gate_batch(
    sim: SparseGF2,
    pairs: NDArray[np.int64],
    clifford_indices: NDArray[np.int64],
    luts: NDArray[np.uint8],
    execution: str,
) -> None:
    if execution == "batch":
        sim._dispatch_2q_batch(pairs, clifford_indices, luts)
        return
    for pair, clifford_index in zip(pairs, clifford_indices, strict=True):
        sim._dispatch_2q(int(pair[0]), int(pair[1]), luts[int(clifford_index)])


def _measure_qubits(
    sim: SparseGF2,
    measured: NDArray[np.int64],
    outcome_rng: np.random.Generator,
    execution: str,
) -> None:
    if execution == "batch":
        sim._measure_z_batch(measured, rng=outcome_rng)
        return
    for qubit in measured:
        sim.measure_z(int(qubit), rng=outcome_rng)


def simulate_trajectory(
    work: TauWorkUnit,
    graph_index: int,
    circuit_index: int,
    edges: NDArray[np.integer] | object,
    *,
    execution: str = "batch",
    use_numba: bool | None = None,
    hybrid: bool = True,
    profile: bool = False,
    audit_tableau: bool = False,
) -> RawTauTrajectoryResult:
    """Run one exact first-passage trajectory from ``t=0`` through ``q_max*n``.

    The reference is never acted on after Bell-pair preparation.  Scrambling
    applies exactly ``q_scramble*n`` independently sampled all-to-all two-qubit
    Cliffords on system qubits.  Every monitored layer then applies
    ``floor(n/2)`` graph-edge Cliffords sampled with replacement, Bernoulli-Z
    measurements on all system qubits, and one unconditional ``S(R)`` check.
    """
    if not isinstance(work, TauWorkUnit):
        raise TypeError("work must be a TauWorkUnit")
    if execution not in {"batch", "scalar"}:
        raise ValueError("execution must be 'batch' or 'scalar'")
    if work.protocol.p_randomness_policy != "independent":
        raise ValueError(
            "common_circuit_disorder is reserved until masks are pre-indexed by layer; "
            "use p_randomness_policy='independent'"
        )
    if isinstance(graph_index, bool) or not isinstance(graph_index, (int, np.integer)):
        raise TypeError("graph_index must be an integer")
    if isinstance(circuit_index, bool) or not isinstance(circuit_index, (int, np.integer)):
        raise TypeError("circuit_index must be an integer")
    graph_index = int(graph_index)
    circuit_index = int(circuit_index)
    if not 0 <= graph_index < work.graphs_per_cell:
        raise IndexError(f"graph_index must lie in [0, {work.graphs_per_cell})")
    if not 0 <= circuit_index < work.protocol.n_circuits:
        raise IndexError(f"circuit_index must lie in [0, {work.protocol.n_circuits})")

    started = time.perf_counter() if profile else 0.0
    n = work.cell.n
    edge_array = _validate_edges(n, edges)
    p = float(work.p_decimal)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must lie in [0, 1]; got {p}")

    pair_rng = _pcg64(work.trajectory_seed(graph_index, circuit_index, "scramble_pairs"))
    scramble_clifford_rng = _pcg64(
        work.trajectory_seed(graph_index, circuit_index, "scramble_cliffords")
    )
    edge_rng = _pcg64(work.trajectory_seed(graph_index, circuit_index, "dynamic_edges"))
    dynamic_clifford_rng = _pcg64(
        work.trajectory_seed(graph_index, circuit_index, "dynamic_cliffords")
    )
    mask_rng = _pcg64(work.trajectory_seed(graph_index, circuit_index, "measurement_mask"))
    outcome_rng = _pcg64(work.trajectory_seed(graph_index, circuit_index, "measurement_outcomes"))
    reference_rng = _pcg64(work.trajectory_seed(graph_index, circuit_index, "reference_placement"))

    reference_system_qubit = n - 1
    if work.protocol.reference_system_qubit_policy == "uniform_system_qubit_per_circuit":
        reference_system_qubit = int(reference_rng.integers(0, n))
    reference_qubit = n
    sim = SparseGF2(n + 1, rng=outcome_rng, use_numba=use_numba, hybrid=hybrid)
    sim.apply_h(reference_system_qubit)
    sim.apply_cx(reference_system_qubit, reference_qubit)
    setup_done = time.perf_counter() if profile else 0.0
    if single_qubit_entropy(sim, reference_qubit) != 1:
        raise RuntimeError("single reference must start with S(R)=1")

    luts = _gate_luts()
    n_scramble = work.protocol.scramble_gate_count(n)
    scramble_pairs = _uniform_distinct_pairs(pair_rng, n, n_scramble)
    scramble_cliffords = scramble_clifford_rng.integers(
        0, SP4_SIZE, size=n_scramble, dtype=np.int64
    )
    _apply_gate_batch(sim, scramble_pairs, scramble_cliffords, luts, execution)
    scramble_done = time.perf_counter() if profile else 0.0
    if single_qubit_entropy(sim, reference_qubit) != 1:
        raise RuntimeError("system-only scrambling changed S(R); this indicates a simulator bug")

    gates_per_layer = n // 2
    cap = work.protocol.t_max(n)
    dynamic_gates = 0
    measurements = 0
    gate_elapsed = 0.0
    measurement_elapsed = 0.0
    entropy_elapsed = 0.0
    tau_p: int | None = None
    stop_layer = cap
    for layer in range(1, cap + 1):
        gate_started = time.perf_counter() if profile else 0.0
        edge_indices = edge_rng.integers(
            0, edge_array.shape[0], size=gates_per_layer, dtype=np.int64
        )
        pairs = edge_array[edge_indices]
        clifford_indices = dynamic_clifford_rng.integers(
            0, SP4_SIZE, size=gates_per_layer, dtype=np.int64
        )
        _apply_gate_batch(sim, pairs, clifford_indices, luts, execution)
        dynamic_gates += gates_per_layer
        if profile:
            gate_elapsed += time.perf_counter() - gate_started

        measurement_started = time.perf_counter() if profile else 0.0
        measured = np.flatnonzero(mask_rng.random(n) < p).astype(np.int64, copy=False)
        _measure_qubits(sim, measured, outcome_rng, execution)
        measurements += int(measured.shape[0])
        if profile:
            measurement_elapsed += time.perf_counter() - measurement_started

        entropy_started = time.perf_counter() if profile else 0.0
        s_r = single_qubit_entropy(sim, reference_qubit)
        if profile:
            entropy_elapsed += time.perf_counter() - entropy_started
        if s_r == 0:
            tau_p = layer
            stop_layer = layer
            break
    layers_executed = stop_layer
    finished = time.perf_counter() if profile else 0.0
    timings = None
    if profile:
        timings = TrajectoryTimings(
            setup_s=setup_done - started,
            scramble_s=scramble_done - setup_done,
            dynamic_gates_s=gate_elapsed,
            measurements_s=measurement_elapsed,
            entropy_s=entropy_elapsed,
            total_s=finished - started,
        )
    return RawTauTrajectoryResult(
        tau_p=tau_p,
        stop_layer=stop_layer,
        event_observed=tau_p is not None,
        reference_system_qubit=reference_system_qubit,
        layers_executed=layers_executed,
        scramble_gates=n_scramble,
        dynamic_gates=dynamic_gates,
        measurements=measurements,
        final_tableau_sha256=_tableau_digest(sim) if audit_tableau else None,
        timings=timings,
    )


__all__ = [
    "ENGINE_VERSION",
    "RawTauTrajectoryResult",
    "TrajectoryTimings",
    "simulate_trajectory",
]
