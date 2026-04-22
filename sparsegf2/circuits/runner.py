"""
Simulation runner: executes one circuit schedule on a SparseGF2 simulator
and returns a fully-populated :class:`SampleRecord`.

The runner is the thinnest possible layer between the :class:`CircuitBuilder`
(which deterministically produces the schedule) and the :class:`SparseGF2`
simulator (which applies it). It collects the diagnostics and observables
that slot into ``samples.parquet`` and optionally extracts the end-of-circuit
tableau for ``tableaus.h5``.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from sparsegf2 import SparseGF2, warmup
from sparsegf2.circuits.builder import CircuitBuilder
from sparsegf2.circuits.config import CircuitConfig, SampleRecord
from sparsegf2.circuits.pictures import init_picture


# Clifford table cache

FULL_CLIFFORD_GROUP_SIZE = 11520
_CLIFFORD_CACHE: Optional[np.ndarray] = None


def get_clifford_table(
    n_cliffords: int = FULL_CLIFFORD_GROUP_SIZE,
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    """Return the two-qubit Clifford table as symplectic matrices.

    Deterministically built from ``stim.Tableau.iter_all(2)``, then cached
    at module level. Optionally persisted to ``<cache_dir>/clifford_cache.pkl``
    keyed by the Stim version.
    """
    global _CLIFFORD_CACHE
    if _CLIFFORD_CACHE is not None and len(_CLIFFORD_CACHE) >= n_cliffords:
        return _CLIFFORD_CACHE[:n_cliffords]

    import stim
    from sparsegf2.gates.clifford import symplectic_from_stim_tableau

    stim_version = stim.__version__

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / "clifford_cache.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                if (
                    data.get("stim_version") == stim_version
                    and "symp" in data
                    and len(data["symp"]) >= n_cliffords
                ):
                    _CLIFFORD_CACHE = data["symp"]
                    return _CLIFFORD_CACHE[:n_cliffords]
            except Exception:
                pass

    all_tabs = list(stim.Tableau.iter_all(2))
    symp = np.zeros((len(all_tabs), 4, 4), dtype=np.uint8)
    for i, tab in enumerate(all_tabs):
        symp[i] = symplectic_from_stim_tableau(tab)

    _CLIFFORD_CACHE = symp

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_dir / "clifford_cache.pkl", "wb") as f:
                pickle.dump({"symp": symp, "stim_version": stim_version}, f)
        except OSError:
            pass

    return symp[:n_cliffords]


# Tableau extraction

def _extract_xz_packed(sim: SparseGF2) -> Tuple[np.ndarray, np.ndarray]:
    """Extract bit-packed symplectic (X, Z) tableau from the simulator state.

    Returns ``(x_packed, z_packed)`` of shape ``(2n, ceil(n/64))`` with uint64
    dtype, matching the on-disk ``tableaus.h5`` layout.
    """
    n = sim.n
    N = sim.N
    n_words = (n + 63) >> 6

    # Sync PLT if the simulator is currently in dense mode, so plt is authoritative.
    if getattr(sim, "_dense_mode", False):
        from sparsegf2.core.numba_kernels import packed_to_plt
        packed_to_plt(sim.x_packed, sim.z_packed, sim.plt, n, N)

    plt = sim.plt  # uint8[N, n], bit 1 = X, bit 0 = Z
    x_packed = np.zeros((N, n_words), dtype=np.uint64)
    z_packed = np.zeros((N, n_words), dtype=np.uint64)
    for r in range(N):
        for q in range(n):
            xz = int(plt[r, q])
            if xz & 2:  # X-bit set -> X or Y
                x_packed[r, q >> 6] |= np.uint64(1) << np.uint64(q & 63)
            if xz & 1:  # Z-bit set -> Z or Y
                z_packed[r, q >> 6] |= np.uint64(1) << np.uint64(q & 63)
    return x_packed, z_packed


# SimulationRunner

class SimulationRunner:
    """Run one circuit schedule through SparseGF2 and collect observables.

    Typical use::

        runner = SimulationRunner(config)
        record = runner.run(sample_seed=0,
                            save_tableau=False,
                            save_realization=False)

    The Clifford table is loaded once per runner instance (lazy via
    :func:`get_clifford_table`), so constructing many runners across workers
    amortizes the table build.
    """

    def __init__(
        self,
        config: CircuitConfig,
        clifford_table: Optional[np.ndarray] = None,
        cache_dir: Optional[Path] = None,
        warmup_jit: bool = True,
    ) -> None:
        self.config = config
        if clifford_table is None:
            clifford_table = get_clifford_table(config.n_cliffords, cache_dir)
        self.clifford_table = clifford_table
        if warmup_jit:
            warmup()

    # --------------------------------------------------------------

    def run(
        self,
        sample_seed: int = 0,
        *,
        save_tableau: bool = False,
        save_realization: bool = False,
    ) -> SampleRecord:
        """Run one sample and return a :class:`SampleRecord`."""
        cfg = self.config
        builder = CircuitBuilder(cfg, sample_seed=sample_seed)
        sim = init_picture(cfg.picture, cfg.n, hybrid_mode=True)
        symp = self.clifford_table
        n_symp = len(symp)

        total_gates = 0
        total_measurements = 0
        total_layers = 0
        t_gate = 0.0
        t_meas = 0.0

        realization_layers = [] if save_realization else None

        t_all_0 = time.perf_counter()
        for layer in builder.layers():
            total_layers += 1
            if layer.n_gates:
                t0 = time.perf_counter()
                for i, (qi, qj) in enumerate(layer.gate_pairs):
                    ci = int(layer.cliff_indices[i]) % n_symp
                    sim.apply_gate(qi, qj, symp[ci])
                t_gate += time.perf_counter() - t0
                total_gates += layer.n_gates
            if layer.n_measurements:
                t0 = time.perf_counter()
                for q in layer.meas_qubits:
                    sim.apply_measurement_z(q)
                t_meas += time.perf_counter() - t0
                total_measurements += layer.n_measurements
            if realization_layers is not None:
                realization_layers.append({
                    "gate_pairs": list(layer.gate_pairs),
                    "cliff_indices": np.asarray(layer.cliff_indices, dtype=np.int64),
                    "meas_qubits": list(layer.meas_qubits),
                })
        t_total = time.perf_counter() - t_all_0

        # --- Observables ---
        k = int(sim.compute_k())
        abar = float(sim.get_active_count())
        bandwidth = int(sim.compute_bandwidth())
        tmi = float(sim.compute_tmi())
        half = list(range(cfg.n // 2))
        entropy_half = float(sim.compute_subsystem_entropy(half)) if half else 0.0

        # --- Diagnostics ---
        avg_gates = total_gates / max(total_layers, 1)
        actual_ratio = (
            total_gates / total_measurements if total_measurements > 0 else float("inf")
        )
        # expected ratio: n/2 gates per layer, n*p measurements per layer in uniform mode
        # -> gates/meas = (n/2) / (n*p) = 1/(2p); reported as 2p here for symmetry
        expected_ratio = 1.0 / (2.0 * cfg.p) if cfg.p > 0 else float("inf")

        record = SampleRecord(
            sample_seed=builder.seed,
            # diagnostics
            total_layers=total_layers,
            total_gates=total_gates,
            avg_gates_per_layer=float(avg_gates),
            total_measurements=total_measurements,
            gate_to_meas_ratio_expected=float(expected_ratio),
            gate_to_meas_ratio_actual=float(actual_ratio),
            final_abar=abar,
            runtime_total_s=float(t_total),
            runtime_gate_phase_s=float(t_gate),
            runtime_meas_phase_s=float(t_meas),
            # observables
            k=k,
            bandwidth=bandwidth,
            tmi=tmi,
            entropy_half_cut=entropy_half,
            p_k_gt_0=1 if k > 0 else 0,
            # optional side-data
            realization_layers=realization_layers,
        )

        if save_tableau:
            xp, zp = _extract_xz_packed(sim)
            record.tableau_x_packed = xp
            record.tableau_z_packed = zp
            # signs are not tracked by SparseGF2's phase-free GF(2) representation;
            # leave as None for MVP.

        return record


__all__ = [
    "FULL_CLIFFORD_GROUP_SIZE",
    "get_clifford_table",
    "SimulationRunner",
]
