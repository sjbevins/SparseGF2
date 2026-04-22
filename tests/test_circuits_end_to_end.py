"""End-to-end test: a small sweep produces a well-formed run directory
that loads correctly with polars+hive partitioning."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pytest

from sparsegf2.circuits import CircuitConfig, RunConfig, SweepDriver


def _make_cfg(tmp: Path) -> RunConfig:
    return RunConfig(
        circuit=CircuitConfig(
            graph_spec="cycle", n=8,
            matching_mode="round_robin", measurement_mode="uniform",
            depth_mode="O(n)", depth_factor=2,
            p=0.15, base_seed=42,
        ),
        sizes=[8, 12], p_min=0.0, p_max=0.3, n_p=3,
        n_samples_per_cell=3,
        output_root=tmp,
        save_tableaus=True,
        save_realizations=True,
        n_workers=1,
    )


def test_end_to_end_sweep(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    run_dir = SweepDriver(cfg, progress=False).run()
    assert run_dir.exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "graph.g6").exists()
    assert (run_dir / "index.parquet").exists()
    # Two sizes x three p values = 6 cell dirs
    cells = list((run_dir / "data").glob("n=*/p=*"))
    assert len(cells) == 6
    for c in cells:
        assert (c / "samples.parquet").exists()
        assert (c / "tableaus.h5").exists()
        assert (c / "realizations.h5").exists()


def test_manifest_contents(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    run_dir = SweepDriver(cfg, progress=False).run()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["config"]["graph_spec"] == "cycle"
    assert manifest["config"]["matching_mode"] == "round_robin"
    assert manifest["config"]["sizes"] == [8, 12]
    assert len(manifest["config"]["p_values"]) == 3
    assert manifest["slot_presence"]["samples"] is True
    assert manifest["slot_presence"]["tableaus"] is True
    assert manifest["slot_presence"]["realizations"] is True
    assert manifest["slot_presence"]["graphs"] is False          # deterministic graph
    assert "sparsegf2_version" in manifest["environment"]
    assert manifest["runtime"]["total_samples"] == 18
    assert manifest["runtime"]["total_cells"] == 6


def test_hive_partitioned_scan_materializes_n_and_p(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    run_dir = SweepDriver(cfg, progress=False).run()
    df = pl.scan_parquet(
        str(run_dir / "data/**/samples.parquet"),
        hive_partitioning=True,
    ).collect()
    assert "n" in df.columns and "p" in df.columns
    assert df.shape[0] == 18
    assert set(df["n"].unique()) == {8, 12}
    assert set(float(x) for x in df["p"].unique()) == {0.0, 0.15, 0.3}
    assert "obs.k" in df.columns
    assert "diag.total_layers" in df.columns


def test_samples_parquet_expected_columns(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    run_dir = SweepDriver(cfg, progress=False).run()
    df = pl.read_parquet(run_dir / "data/n=0008/p=0.0000/samples.parquet")
    required = {
        "sample_seed",
        "diag.total_layers", "diag.total_gates", "diag.avg_gates_per_layer",
        "diag.total_measurements",
        "diag.gate_to_meas_ratio_expected", "diag.gate_to_meas_ratio_actual",
        "diag.final_abar",
        "diag.runtime_total_s", "diag.runtime_gate_phase_s", "diag.runtime_meas_phase_s",
        "obs.k", "obs.bandwidth", "obs.tmi", "obs.entropy_half_cut", "obs.p_k_gt_0",
    }
    assert required <= set(df.columns)


def test_tableaus_h5_shape_and_attrs(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    run_dir = SweepDriver(cfg, progress=False).run()
    with h5py.File(run_dir / "data/n=0008/p=0.1500/tableaus.h5", "r") as f:
        assert f.attrs["schema_version"] == "1.0.0"
        assert f.attrs["encoding"] == "symplectic_packed_uint64"
        assert int(f.attrs["n"]) == 8
        S = int(f.attrs["n_samples"])
        assert S == 3
        assert f["x_packed"].shape == (S, 2 * 8, 1)
        assert f["z_packed"].shape == (S, 2 * 8, 1)
        assert f["sample_seed"].shape == (S,)


def test_round_trip_tableau_data(tmp_path: Path):
    """Bit-for-bit parity between in-memory SparseGF2 tableau and the one
    pulled back from tableaus.h5."""
    from sparsegf2.circuits.runner import SimulationRunner, _extract_xz_packed
    cfg = CircuitConfig(
        graph_spec="cycle", n=8, matching_mode="round_robin",
        measurement_mode="uniform", depth_mode="O(n)", depth_factor=2,
        p=0.15, base_seed=42,
    )
    runner = SimulationRunner(cfg, warmup_jit=True)
    record = runner.run(sample_seed=0, save_tableau=True)
    # Run the same sim again through the runner/driver -> file -> reload
    run_cfg = RunConfig(
        circuit=cfg, sizes=[8], p_min=0.15, p_max=0.15, n_p=1,
        n_samples_per_cell=1, output_root=tmp_path, save_tableaus=True,
    )
    run_dir = SweepDriver(run_cfg, progress=False).run()
    with h5py.File(run_dir / "data/n=0008/p=0.1500/tableaus.h5", "r") as f:
        x_disk = f["x_packed"][0]
        z_disk = f["z_packed"][0]
    assert np.array_equal(record.tableau_x_packed, x_disk)
    assert np.array_equal(record.tableau_z_packed, z_disk)


def test_realizations_h5_has_layer_groups(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    run_dir = SweepDriver(cfg, progress=False).run()
    with h5py.File(run_dir / "data/n=0008/p=0.1500/realizations.h5", "r") as f:
        seeds = [k for k in f.keys() if k.startswith("seed_")]
        assert len(seeds) == 3
        g = f[seeds[0]]
        layer_keys = sorted(g.keys())
        assert all(k.startswith("layer_") for k in layer_keys)
        # First layer contains the three expected datasets
        first = g[layer_keys[0]]
        assert set(first.keys()) == {"gate_pairs", "cliff_indices", "meas_qubits"}
