"""
RunWriter: serializes a run to the standardized runs/<run_id>/ tree.

Notes:
- All per-cell writes are single-owner: the driver accumulates all
  SampleRecord objects for one (n, p) cell before invoking the writer.
- Parquet files use pyarrow; HDF5 files use h5py with gzip compression
  and byte-shuffle.
- Every file carries embedded metadata (schema_version, encoding, attrs)
  to be interpreted without external documentation.
- One graph_n{n:04d}.g6 per size for deterministic graphs; the canonical
  per-size mapping is also in manifest.json under n_to_graph6.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sparsegf2 import __version__ as _SPARSEGF2_VERSION
from sparsegf2.circuits.config import RunConfig, SampleRecord
from sparsegf2.circuits.graphs import parse_graph_spec

SCHEMA_VERSION = "1.0.0"


# Helpers

def _format_n(n: int) -> str:
    return f"n={int(n):04d}"


def _format_p(p: float) -> str:
    return f"p={float(p):.4f}"


def _git_info(repo_root: Path) -> Dict[str, object]:
    """Best-effort collection of git hash + dirty state."""
    info: Dict[str, object] = {"git_hash": None, "git_dirty": None}
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short=7", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=2.0
        ).strip()
        info["git_hash"] = out
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True, timeout=2.0
        )
        info["git_dirty"] = bool(out.strip())
    except Exception:
        pass
    return info


def _config_slug(cfg: RunConfig) -> str:
    """Short human-readable slug for the run_id."""
    c = cfg.circuit
    graph_shorthand = {"cycle": "NN", "complete": "ATA"}.get(c.graph_spec, c.graph_spec)
    depth_code = {"O(n)": "On", "O(log_n)": "Ologn"}.get(c.depth_mode, "depth")
    return f"{graph_shorthand}-{c.matching_mode}-{c.measurement_mode}-{depth_code}"


def auto_run_id(cfg: RunConfig, git_hash: Optional[str] = None) -> str:
    """Generate the default ``run_id`` used when ``RunConfig.run_id`` is None."""
    date = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%d")
    slug = _config_slug(cfg)
    if git_hash:
        return f"{date}_{slug}__{git_hash}"
    # Fall back to a short hash of the config so different configs don't collide
    h = hashlib.sha1(json.dumps(cfg.circuit.to_dict(), sort_keys=True).encode()).hexdigest()[:7]
    return f"{date}_{slug}__{h}"


# Manifest

def _build_manifest(
    cfg: RunConfig,
    run_id: str,
    *,
    started_utc: str,
    finished_utc: Optional[str],
    wall_seconds: Optional[float],
    slot_presence: Dict[str, object],
    graph_info: Dict[str, object],
    environment: Dict[str, object],
) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_utc": started_utc,
        "finished_utc": finished_utc,
        "notes": cfg.notes,
        "config": {
            **cfg.circuit.to_dict(),
            "sizes": list(cfg.sizes),
            "p_values": [float(p) for p in cfg.p_values().tolist()],
            "p_min": float(cfg.p_min),
            "p_max": float(cfg.p_max),
            "n_p": int(cfg.n_p),
            "n_samples_per_cell": int(cfg.n_samples_per_cell),
        },
        "graph_info": graph_info,
        "slot_presence": slot_presence,
        "environment": environment,
        "runtime": {
            "n_workers": int(cfg.n_workers),
            "total_samples": int(cfg.total_samples()),
            "total_cells": int(cfg.total_cells()),
            "wall_seconds": float(wall_seconds) if wall_seconds is not None else None,
        },
    }


def _collect_environment() -> Dict[str, object]:
    env: Dict[str, object] = {
        "sparsegf2_version": _SPARSEGF2_VERSION,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "hostname": socket.gethostname(),
    }
    try:
        import stim
        env["stim_version"] = stim.__version__
    except ImportError:
        env["stim_version"] = None
    try:
        env["numpy_version"] = np.__version__
    except Exception:
        pass
    try:
        env["h5py_version"] = h5py.__version__
    except Exception:
        pass
    try:
        env["pyarrow_version"] = pa.__version__
    except Exception:
        pass
    env["system"] = platform.platform()
    return env


def _collect_graph_info(cfg: RunConfig) -> Dict[str, object]:
    """Build the ``graph_info`` block of the manifest."""
    c = cfg.circuit
    # Determine whether the graph is stochastic by inspecting one size.
    probe = parse_graph_spec(c.graph_spec, cfg.sizes[0], seed=c.base_seed)
    info: Dict[str, object] = {
        "graph_spec": c.graph_spec,
        "is_stochastic": probe.is_stochastic,
    }
    if not probe.is_stochastic:
        info["n_to_graph6"] = {
            str(n): parse_graph_spec(c.graph_spec, n, seed=c.base_seed).graph6
            for n in cfg.sizes
        }
    return info


# Parquet schema for samples.parquet

_SAMPLES_COLUMNS = [
    # identity
    ("sample_seed", pa.int64()),
    # diagnostics
    ("diag.total_layers", pa.int64()),
    ("diag.total_gates", pa.int64()),
    ("diag.avg_gates_per_layer", pa.float64()),
    ("diag.total_measurements", pa.int64()),
    ("diag.gate_to_meas_ratio_expected", pa.float64()),
    ("diag.gate_to_meas_ratio_actual", pa.float64()),
    ("diag.final_abar", pa.float64()),
    ("diag.runtime_total_s", pa.float64()),
    ("diag.runtime_gate_phase_s", pa.float64()),
    ("diag.runtime_meas_phase_s", pa.float64()),
    # observables
    ("obs.k", pa.int32()),
    ("obs.bandwidth", pa.int32()),
    ("obs.tmi", pa.float64()),
    ("obs.entropy_half_cut", pa.float64()),
    ("obs.p_k_gt_0", pa.uint8()),
]

_SAMPLES_SCHEMA = pa.schema(_SAMPLES_COLUMNS)


def _records_to_arrow(records: List[SampleRecord]) -> pa.Table:
    data: Dict[str, list] = {name: [] for name, _ in _SAMPLES_COLUMNS}
    for r in records:
        data["sample_seed"].append(int(r.sample_seed))
        data["diag.total_layers"].append(int(r.total_layers))
        data["diag.total_gates"].append(int(r.total_gates))
        data["diag.avg_gates_per_layer"].append(float(r.avg_gates_per_layer))
        data["diag.total_measurements"].append(int(r.total_measurements))
        data["diag.gate_to_meas_ratio_expected"].append(float(r.gate_to_meas_ratio_expected))
        data["diag.gate_to_meas_ratio_actual"].append(float(r.gate_to_meas_ratio_actual))
        data["diag.final_abar"].append(float(r.final_abar))
        data["diag.runtime_total_s"].append(float(r.runtime_total_s))
        data["diag.runtime_gate_phase_s"].append(float(r.runtime_gate_phase_s))
        data["diag.runtime_meas_phase_s"].append(float(r.runtime_meas_phase_s))
        data["obs.k"].append(int(r.k))
        data["obs.bandwidth"].append(int(r.bandwidth))
        data["obs.tmi"].append(float(r.tmi))
        data["obs.entropy_half_cut"].append(float(r.entropy_half_cut))
        data["obs.p_k_gt_0"].append(int(r.p_k_gt_0))
    return pa.Table.from_pydict(data, schema=_SAMPLES_SCHEMA)


# HDF5 helpers

def _h5_compression() -> Dict[str, object]:
    """Preferred compression for HDF5 datasets."""
    return {"compression": "gzip", "compression_opts": 4, "shuffle": True}


# RunWriter

@dataclass
class _RunState:
    """Mutable state tracked by a :class:`RunWriter` during a run."""

    cells_written: List[Dict[str, object]] = field(default_factory=list)
    any_tableaus: bool = False
    any_realizations: bool = False


class RunWriter:
    """Writes a run's output to ``runs/<run_id>/`` following the standard layout.

    Lifecycle::

        writer = RunWriter(run_config, repo_root=<repo>)
        writer.begin_run()
        for (n, p, records) in cells:
            writer.write_cell(n=n, p=p, records=records,
                              save_tableaus=..., save_realizations=...)
        writer.finalize(wall_seconds=...)
    """

    def __init__(
        self,
        run_config: RunConfig,
        repo_root: Optional[Path] = None,
    ) -> None:
        self.cfg = run_config
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self._env = _collect_environment()
        self._git = _git_info(self.repo_root)
        self._env.update({k: v for k, v in self._git.items()})
        git_hash = self._git.get("git_hash")
        self.run_id = run_config.run_id or auto_run_id(run_config, git_hash if isinstance(git_hash, str) else None)
        self.run_dir: Path = run_config.output_root / self.run_id
        self._state = _RunState()
        self._started_utc: Optional[str] = None

    # --------------------------------------------------------------

    def begin_run(self) -> Path:
        """Create the run directory and persist the preliminary manifest.

        A placeholder ``manifest.json`` is written immediately (before any
        samples exist) so that an interrupted run still has inspectable
        metadata. :meth:`finalize` overwrites it with the final version.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "data").mkdir(exist_ok=True)

        # Write deterministic graph.g6 files if applicable. One file per size
        # (graph_n{n:04d}.g6) so that analysis tools reconstructing a graph for
        # a particular n pick up the correct encoding. The canonical per-size
        # mapping is also recorded in manifest.json under n_to_graph6.
        graph_info = _collect_graph_info(self.cfg)
        if not graph_info["is_stochastic"]:
            for n in self.cfg.sizes:
                probe = parse_graph_spec(
                    self.cfg.circuit.graph_spec, n,
                    seed=self.cfg.circuit.base_seed,
                )
                (self.run_dir / f"graph_n{n:04d}.g6").write_text(probe.graph6 + "\n")

        self._started_utc = _dt.datetime.now(tz=_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self._write_manifest(finished_utc=None, wall_seconds=None)
        return self.run_dir

    # --------------------------------------------------------------

    def cell_dir(self, n: int, p: float) -> Path:
        d = self.run_dir / "data" / _format_n(n) / _format_p(p)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_cell(
        self,
        *,
        n: int,
        p: float,
        records: List[SampleRecord],
        save_tableaus: bool = False,
        save_realizations: bool = False,
    ) -> None:
        """Persist one ``(n, p)`` cell's samples."""
        d = self.cell_dir(n, p)

        # samples.parquet — always
        table = _records_to_arrow(records)
        pq.write_table(table, d / "samples.parquet")

        # tableaus.h5 — optional
        if save_tableaus and any(r.tableau_x_packed is not None for r in records):
            self._write_tableaus_h5(d / "tableaus.h5", records, n)
            self._state.any_tableaus = True

        # realizations.h5 — optional
        if save_realizations and any(r.realization_layers is not None for r in records):
            self._write_realizations_h5(d / "realizations.h5", records)
            self._state.any_realizations = True

        # timeseries.h5 — present when single_ref was run with record_time_series
        if any(r.ref_entropy_timeseries is not None for r in records):
            self._write_timeseries_h5(d / "timeseries.h5", records, n)

        self._state.cells_written.append({
            "n": int(n), "p": float(p), "n_samples": len(records),
        })

    # --------------------------------------------------------------

    def _write_tableaus_h5(self, path: Path, records: List[SampleRecord], n: int) -> None:
        S = len(records)
        N = 2 * n
        n_words = (n + 63) >> 6
        x_stack = np.zeros((S, N, n_words), dtype=np.uint64)
        z_stack = np.zeros((S, N, n_words), dtype=np.uint64)
        seeds = np.zeros(S, dtype=np.int64)
        has_signs = any(r.tableau_signs is not None for r in records)
        signs_stack = np.zeros((S, N), dtype=np.uint8) if has_signs else None
        for i, r in enumerate(records):
            seeds[i] = r.sample_seed
            if r.tableau_x_packed is not None:
                x_stack[i] = r.tableau_x_packed
            if r.tableau_z_packed is not None:
                z_stack[i] = r.tableau_z_packed
            if signs_stack is not None and r.tableau_signs is not None:
                signs_stack[i] = r.tableau_signs

        with h5py.File(path, "w") as f:
            f.attrs["schema_version"] = SCHEMA_VERSION
            f.attrs["canonical_form"] = "as_produced"
            f.attrs["encoding"] = "symplectic_packed_uint64"
            f.attrs["encoding_version"] = "1.0"
            f.attrs["n"] = int(n)
            f.attrs["n_samples"] = int(S)
            comp = _h5_compression()
            f.create_dataset(
                "x_packed", data=x_stack, chunks=(1, N, n_words), **comp
            )
            f.create_dataset(
                "z_packed", data=z_stack, chunks=(1, N, n_words), **comp
            )
            f.create_dataset("sample_seed", data=seeds)
            if signs_stack is not None:
                f.create_dataset("signs", data=signs_stack, chunks=(1, N), **comp)

    def _write_timeseries_h5(
        self, path: Path, records: List[SampleRecord], n: int
    ) -> None:
        """Persist per-layer reference-qubit entropy traces for single_ref runs.

        Dataset layout:
          - ``S_of_t``      uint8[S, total_layers+1]: S(qubit n) at t=0..T.
          - ``sample_seed`` int64[S]: sample seeds, aligned with rows of S_of_t.
          - ``t_axis``      int32[total_layers+1]: 0, 1, ..., total_layers.
        """
        ts_records = [
            r for r in records if r.ref_entropy_timeseries is not None
        ]
        if not ts_records:
            return
        S = len(ts_records)
        T_plus_1 = int(ts_records[0].ref_entropy_timeseries.shape[0])
        ts_stack = np.zeros((S, T_plus_1), dtype=np.uint8)
        seeds = np.zeros(S, dtype=np.int64)
        for i, r in enumerate(ts_records):
            arr = np.asarray(r.ref_entropy_timeseries, dtype=np.uint8)
            if arr.shape != (T_plus_1,):
                raise ValueError(
                    "ref_entropy_timeseries has inconsistent length across "
                    f"samples: got {arr.shape} expected ({T_plus_1},)"
                )
            ts_stack[i] = arr
            seeds[i] = int(r.sample_seed)

        with h5py.File(path, "w") as f:
            f.attrs["schema_version"] = SCHEMA_VERSION
            f.attrs["encoding_version"] = "1.0"
            f.attrs["n"] = int(n)
            f.attrs["n_samples"] = int(S)
            f.attrs["total_layers"] = int(T_plus_1 - 1)
            comp = _h5_compression()
            f.create_dataset(
                "S_of_t", data=ts_stack, chunks=(1, T_plus_1), **comp
            )
            f.create_dataset("sample_seed", data=seeds)
            f.create_dataset(
                "t_axis", data=np.arange(T_plus_1, dtype=np.int32)
            )

    def _write_realizations_h5(
        self, path: Path, records: List[SampleRecord]
    ) -> None:
        with h5py.File(path, "w") as f:
            f.attrs["schema_version"] = SCHEMA_VERSION
            for r in records:
                if r.realization_layers is None:
                    continue
                g = f.create_group(f"seed_{int(r.sample_seed)}")
                for t, layer in enumerate(r.realization_layers):
                    gl = g.create_group(f"layer_{t:06d}")
                    pairs = np.asarray(layer["gate_pairs"], dtype=np.int32).reshape(-1, 2) \
                        if layer["gate_pairs"] else np.zeros((0, 2), dtype=np.int32)
                    gl.create_dataset("gate_pairs", data=pairs)
                    gl.create_dataset(
                        "cliff_indices",
                        data=np.asarray(layer["cliff_indices"], dtype=np.int64),
                    )
                    gl.create_dataset(
                        "meas_qubits",
                        data=np.asarray(layer["meas_qubits"], dtype=np.int32),
                    )

    # --------------------------------------------------------------

    def finalize(self, *, wall_seconds: Optional[float] = None) -> None:
        """Overwrite the manifest with final metadata and write ``index.parquet``."""
        finished_utc = _dt.datetime.now(tz=_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        self._write_manifest(finished_utc=finished_utc, wall_seconds=wall_seconds)
        self._write_index_parquet()

    def _write_manifest(
        self, finished_utc: Optional[str], wall_seconds: Optional[float]
    ) -> None:
        graph_info = _collect_graph_info(self.cfg)
        slot_presence: Dict[str, object] = {
            "samples": True,
            "graphs": bool(graph_info["is_stochastic"]),
            "tableaus": self._state.any_tableaus,
            "realizations": self._state.any_realizations,
            "analysis": {},
        }
        manifest = _build_manifest(
            self.cfg,
            self.run_id,
            started_utc=self._started_utc or finished_utc or "",
            finished_utc=finished_utc,
            wall_seconds=wall_seconds,
            slot_presence=slot_presence,
            graph_info=graph_info,
            environment=self._env,
        )
        with open(self.run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=False)

    def _write_index_parquet(self) -> None:
        """Write the redundant per-sample index. MVP form: one row per cell,
        with sample counts and slot-presence flags."""
        rows = {
            "n": [],
            "p": [],
            "n_samples": [],
            "has_tableau": [],
            "has_realization": [],
        }
        for cell in self._state.cells_written:
            d = self.cell_dir(cell["n"], cell["p"])
            rows["n"].append(int(cell["n"]))
            rows["p"].append(float(cell["p"]))
            rows["n_samples"].append(int(cell["n_samples"]))
            rows["has_tableau"].append((d / "tableaus.h5").exists())
            rows["has_realization"].append((d / "realizations.h5").exists())
        table = pa.table(rows, schema=pa.schema([
            ("n", pa.int32()),
            ("p", pa.float64()),
            ("n_samples", pa.int64()),
            ("has_tableau", pa.bool_()),
            ("has_realization", pa.bool_()),
        ]))
        pq.write_table(table, self.run_dir / "index.parquet")


__all__ = [
    "RunWriter",
    "auto_run_id",
    "SCHEMA_VERSION",
]
