"""
Configuration dataclasses and sample-record schema.

- :class:`CircuitConfig` — per-``(n, p)`` cell configuration.
- :class:`RunConfig`     — sweep-level driver configuration.
- :class:`SampleRecord`  — the per-sample data that runners produce and
  writers persist. This is the Python-side mirror of the ``samples.parquet``
  row schema.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np


# Enumerations (validated in __post_init__ of the dataclasses)

GATING_MODES = ("matching",)                          # MVP: only "matching"
MATCHING_MODE_NAMES = ("round_robin", "palette", "fresh")
MEASUREMENT_MODE_NAMES = ("uniform",)                 # MVP: only "uniform"
DEPTH_MODES = ("O(n)", "O(log_n)")                    # MVP
PICTURE_NAMES = ("purification",)                     # MVP: only "purification"
GRAPH_SPECS_MVP = ("cycle", "complete")


# CircuitConfig — per-cell knobs

@dataclass
class CircuitConfig:
    """Per-``(n, p)`` circuit configuration.

    Everything that defines *how the circuit is built* except the sample seed
    lives here. The sweep driver iterates over ``(n, p, sample_seed)`` keeping
    all other fields fixed.
    """

    # ---- Graph ----
    graph_spec: str                                   # "cycle" | "complete"
    n: int                                            # qubit count (even for MVP)

    # ---- Physics picture ----
    picture: str = "purification"

    # ---- Gating ----
    gating_mode: str = "matching"                     # MVP: "matching"
    matching_mode: str = "round_robin"                # "round_robin" | "palette" | "fresh"

    # ---- Measurements ----
    measurement_mode: str = "uniform"                 # MVP: "uniform"
    p: float = 0.15

    # ---- Depth ----
    depth_mode: str = "O(n)"                          # "O(n)" | "O(log_n)"
    depth_factor: int = 8

    # ---- Clifford group ----
    n_cliffords: int = 11520                          # full two-qubit Clifford group

    # ---- RNG ----
    base_seed: int = 42

    def __post_init__(self) -> None:
        # Graph spec
        if self.graph_spec not in GRAPH_SPECS_MVP:
            raise ValueError(
                f"graph_spec must be one of {GRAPH_SPECS_MVP}; got {self.graph_spec!r}"
            )
        # n
        if not isinstance(self.n, (int, np.integer)) or self.n < 2:
            raise ValueError(f"n must be an integer >= 2; got {self.n!r}")
        self.n = int(self.n)
        # Picture
        if self.picture not in PICTURE_NAMES:
            raise ValueError(
                f"picture must be one of {PICTURE_NAMES}; got {self.picture!r}"
            )
        # Gating
        if self.gating_mode not in GATING_MODES:
            raise ValueError(
                f"gating_mode must be one of {GATING_MODES}; got {self.gating_mode!r}"
            )
        # Matching
        if self.matching_mode not in MATCHING_MODE_NAMES:
            raise ValueError(
                f"matching_mode must be one of {MATCHING_MODE_NAMES}; "
                f"got {self.matching_mode!r}"
            )
        # Measurement
        if self.measurement_mode not in MEASUREMENT_MODE_NAMES:
            raise ValueError(
                f"measurement_mode must be one of {MEASUREMENT_MODE_NAMES}; "
                f"got {self.measurement_mode!r}"
            )
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"p must be in [0, 1]; got {self.p}")
        # Depth
        if self.depth_mode not in DEPTH_MODES:
            raise ValueError(
                f"depth_mode must be one of {DEPTH_MODES}; got {self.depth_mode!r}"
            )
        if not isinstance(self.depth_factor, (int, np.integer)) or self.depth_factor < 1:
            raise ValueError(
                f"depth_factor must be a positive integer; got {self.depth_factor!r}"
            )
        self.depth_factor = int(self.depth_factor)
        # Clifford table size
        if not isinstance(self.n_cliffords, (int, np.integer)) or self.n_cliffords < 1:
            raise ValueError(
                f"n_cliffords must be a positive integer; got {self.n_cliffords!r}"
            )
        self.n_cliffords = int(self.n_cliffords)
        # Seed
        if not isinstance(self.base_seed, (int, np.integer)):
            raise ValueError(f"base_seed must be an integer; got {self.base_seed!r}")
        self.base_seed = int(self.base_seed)

    # --------------------------------------------------------------
    # Derived quantities
    # --------------------------------------------------------------

    def total_layers(self) -> int:
        """Total number of circuit layers implied by depth_mode + depth_factor.

        - ``O(n)``     -> ``depth_factor * n``
        - ``O(log_n)`` -> ``depth_factor * max(1, ceil(log2(n)))``
        """
        if self.depth_mode == "O(n)":
            return max(1, self.depth_factor * self.n)
        if self.depth_mode == "O(log_n)":
            return max(1, self.depth_factor * max(1, int(math.ceil(math.log2(self.n)))))
        raise AssertionError(f"Unhandled depth_mode {self.depth_mode!r}")

    def expected_gate_to_meas_ratio(self) -> float:
        """Expected ratio of gate-applications to measurement-applications.

        In uniform measurement mode each of the n qubits is independently
        measured with probability p per layer, giving n*p measurements per
        layer on average; each layer applies n/2 two-qubit gates. So the
        expected gate-to-measurement ratio is (n/2) / (n*p) = 1/(2p) for
        p > 0. The runner stores this same value in
        SampleRecord.gate_to_meas_ratio_expected.
        """
        if self.p <= 0.0:
            return float("inf")
        return 1.0 / (2.0 * float(self.p))

    def to_dict(self) -> dict:
        """Serializable dict for manifest.json."""
        return asdict(self)


# RunConfig — sweep-level knobs

@dataclass
class RunConfig:
    """Sweep-level configuration for a whole run.

    One ``RunConfig`` produces one ``runs/<run_id>/`` directory.
    """

    circuit: CircuitConfig                            # shared knobs (n & p overridden per cell)
    sizes: List[int]                                  # e.g. [32, 64, 128]
    p_min: float
    p_max: float
    n_p: int                                          # linspace(p_min, p_max, n_p)
    n_samples_per_cell: int

    output_root: Path = field(default_factory=lambda: Path("runs"))
    run_id: Optional[str] = None                      # auto-generated when None

    save_tableaus: bool = False
    save_realizations: bool = False
    save_rng_state: bool = False

    n_workers: int = 1
    batch_size: int = 50

    # Free-form description persisted into manifest.json under a "notes" key.
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.sizes:
            raise ValueError("sizes must be a non-empty list of integers")
        if any((not isinstance(n, (int, np.integer)) or n < 2) for n in self.sizes):
            raise ValueError(f"all sizes must be integers >= 2; got {self.sizes}")
        self.sizes = [int(n) for n in self.sizes]

        if not (0.0 <= self.p_min <= self.p_max <= 1.0):
            raise ValueError(
                f"require 0 <= p_min <= p_max <= 1; got p_min={self.p_min}, p_max={self.p_max}"
            )
        if not isinstance(self.n_p, (int, np.integer)) or self.n_p < 1:
            raise ValueError(f"n_p must be a positive integer; got {self.n_p!r}")
        self.n_p = int(self.n_p)
        if not isinstance(self.n_samples_per_cell, (int, np.integer)) or self.n_samples_per_cell < 1:
            raise ValueError(
                f"n_samples_per_cell must be a positive integer; got {self.n_samples_per_cell!r}"
            )
        self.n_samples_per_cell = int(self.n_samples_per_cell)

        if not isinstance(self.n_workers, (int, np.integer)) or self.n_workers < 1:
            raise ValueError(f"n_workers must be a positive integer; got {self.n_workers!r}")
        self.n_workers = int(self.n_workers)
        if not isinstance(self.batch_size, (int, np.integer)) or self.batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer; got {self.batch_size!r}")
        self.batch_size = int(self.batch_size)

        self.output_root = Path(self.output_root)

    # --------------------------------------------------------------
    # Derived quantities
    # --------------------------------------------------------------

    def p_values(self) -> np.ndarray:
        """The full linspace of p values used in the sweep."""
        if self.n_p == 1:
            return np.array([self.p_min], dtype=np.float64)
        return np.linspace(self.p_min, self.p_max, self.n_p, dtype=np.float64)

    def total_cells(self) -> int:
        return len(self.sizes) * self.n_p

    def total_samples(self) -> int:
        return self.total_cells() * self.n_samples_per_cell

    def cell_config(self, n: int, p: float) -> CircuitConfig:
        """Return a copy of ``circuit`` with ``n`` and ``p`` overridden."""
        cfg_dict = self.circuit.to_dict()
        cfg_dict["n"] = int(n)
        cfg_dict["p"] = float(p)
        return CircuitConfig(**cfg_dict)


# SampleRecord — per-sample data produced by a SimulationRunner

@dataclass
class SampleRecord:
    """Everything one sample contributes to the run output.

    Fields map directly onto columns of ``samples.parquet``
    (identity + ``diag.*`` + ``obs.*``) plus optional tableau / realization
    side-data that the writer routes to ``tableaus.h5`` / ``realizations.h5``.
    """

    # --- Identity (slot 2) ---
    sample_seed: int
    rng_state_final: Optional[bytes] = None           # set only if RunConfig.save_rng_state
    graph6_per_sample: Optional[str] = None           # set only for stochastic graphs

    # --- Diagnostics (slot 3, diag.*) ---
    total_layers: int = 0
    total_gates: int = 0
    avg_gates_per_layer: float = 0.0
    total_measurements: int = 0
    gate_to_meas_ratio_expected: float = 0.0
    gate_to_meas_ratio_actual: float = 0.0
    final_abar: float = 0.0
    runtime_total_s: float = 0.0
    runtime_gate_phase_s: float = 0.0
    runtime_meas_phase_s: float = 0.0

    # --- Observables (slot 4, obs.*) ---
    k: int = 0
    bandwidth: int = 0
    tmi: float = 0.0
    entropy_half_cut: float = 0.0
    p_k_gt_0: int = 0

    # --- Optional side-data (written to separate slots) ---
    tableau_x_packed: Optional[np.ndarray] = None     # uint64[2n, ceil(n/64)]
    tableau_z_packed: Optional[np.ndarray] = None
    tableau_signs: Optional[np.ndarray] = None        # uint8[2n], if available
    realization_layers: Optional[list] = None         # list of CircuitLayer-like dicts


__all__ = [
    "GATING_MODES",
    "MATCHING_MODE_NAMES",
    "MEASUREMENT_MODE_NAMES",
    "DEPTH_MODES",
    "PICTURE_NAMES",
    "GRAPH_SPECS_MVP",
    "CircuitConfig",
    "RunConfig",
    "SampleRecord",
]
