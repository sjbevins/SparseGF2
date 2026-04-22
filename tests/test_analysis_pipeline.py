"""Unit tests for the analysis pipeline."""
from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from sparsegf2 import SparseGF2
from sparsegf2.analysis.observables import observe
from sparsegf2.analysis_pipeline import (
    AnalysisConfig, run_pipeline, rehydrate_sim, ANALYSIS_REGISTRY,
)
from sparsegf2.circuits import CircuitConfig, RunConfig, SweepDriver


# Session fixture: one small run shared by most tests

@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("ap_small")
    cfg = RunConfig(
        circuit=CircuitConfig(
            graph_spec="cycle", n=8, matching_mode="round_robin",
            measurement_mode="uniform", depth_mode="O(n)", depth_factor=2,
            p=0.15, base_seed=42,
        ),
        sizes=[8, 12], p_min=0.0, p_max=0.3, n_p=3,
        n_samples_per_cell=3,
        output_root=root, save_tableaus=True, n_workers=1,
    )
    return SweepDriver(cfg, progress=False).run()


# Rehydrate parity

def test_rehydrate_parity(small_run: Path):
    """Rehydrated tableau reproduces the same observe() scalars as the live sim."""
    from sparsegf2.circuits.runner import SimulationRunner
    from sparsegf2.circuits import CircuitConfig

    cfg = CircuitConfig(
        graph_spec="cycle", n=8, matching_mode="round_robin",
        measurement_mode="uniform", depth_mode="O(n)", depth_factor=2,
        p=0.15, base_seed=42,
    )
    live_runner = SimulationRunner(cfg, warmup_jit=False)
    live_rec = live_runner.run(sample_seed=0, save_tableau=True)
    live_sim = SparseGF2(cfg.n, hybrid_mode=False)
    # Recreate live_sim state from the packed tableau the runner returned.
    from sparsegf2.core.numba_kernels import packed_to_plt, rebuild_indices_from_plt
    live_sim.x_packed[:] = live_rec.tableau_x_packed
    live_sim.z_packed[:] = live_rec.tableau_z_packed
    packed_to_plt(live_sim.x_packed, live_sim.z_packed, live_sim.plt,
                  live_sim.n, live_sim.N)
    rebuild_indices_from_plt(
        live_sim.plt, live_sim.supp_q, live_sim.supp_len, live_sim.supp_pos,
        live_sim.inv, live_sim.inv_len, live_sim.inv_x, live_sim.inv_x_len,
        live_sim.inv_pos, live_sim.inv_x_pos, live_sim.n, live_sim.N,
    )
    live_obs = observe(live_sim)

    # Rehydrate the same packed bits via the pipeline helper and compare.
    resim = rehydrate_sim(cfg.n,
                          live_rec.tableau_x_packed,
                          live_rec.tableau_z_packed)
    re_obs = observe(resim)

    for k, v in live_obs.items():
        if isinstance(v, bool) or isinstance(v, (int, np.integer)):
            assert int(re_obs[k]) == int(v), f"{k} mismatch"
        else:
            a = float(v); b = float(re_obs[k])
            if np.isnan(a) and np.isnan(b):
                continue
            assert abs(a - b) < 1e-9, f"{k}: live={a} rehydrated={b}"


# End-to-end cell-scope analyses

def test_full_pipeline_produces_all_outputs(small_run: Path):
    report = run_pipeline(AnalysisConfig(run_dir=small_run, verbose=False))
    assert not report.errors, report.errors

    cells = sorted((small_run / "data").glob("n=*/p=*"))
    for cell in cells:
        for name in ("distances", "weight_stats", "logical_weights"):
            assert (cell / "analysis" / f"{name}.parquet").exists(), \
                f"{name}.parquet missing in {cell}"
        for name in ("entropy_profile", "weight_spectrum"):
            assert (cell / "analysis" / f"{name}.h5").exists(), \
                f"{name}.h5 missing in {cell}"
        reg = json.loads((cell / "analysis" / "_registry.json").read_text())
        assert set(reg["entries"]) == {
            "distances", "weight_stats", "entropy_profile",
            "weight_spectrum", "logical_weights",
        }

    assert (small_run / "aggregates.parquet").exists()


def test_samples_parquet_still_parses(small_run: Path):
    """Pipeline must not touch samples.parquet."""
    df = pl.scan_parquet(
        str(small_run / "data" / "**" / "samples.parquet"),
        hive_partitioning=True,
    ).collect()
    assert len(df) == 2 * 3 * 3    # 2 sizes x 3 p x 3 samples


# Analysis-specific content checks

def test_distances_p_zero_sanity(tmp_path: Path):
    """At p=0 the reference state is maximally entangled with the system:
    k = n and any single qubit has I(A; R) > 0, so d_cont = d_min = 1.
    """
    cfg = RunConfig(
        circuit=CircuitConfig(
            graph_spec="cycle", n=8, matching_mode="round_robin",
            measurement_mode="uniform", depth_mode="O(n)", depth_factor=2,
            p=0.0, base_seed=42,
        ),
        sizes=[8], p_min=0.0, p_max=0.0, n_p=1,
        n_samples_per_cell=2,
        output_root=tmp_path, save_tableaus=True, n_workers=1,
    )
    run_dir = SweepDriver(cfg, progress=False).run()
    run_pipeline(AnalysisConfig(run_dir=run_dir, only=["distances"], verbose=False))
    df = pl.read_parquet(run_dir / "data" / "n=0008" / "p=0.0000"
                                       / "analysis" / "distances.parquet")
    # At p=0, every sample has k = n = 8, and any single qubit is in I(A; R) > 0.
    assert (df["d_cont"] == 1).all()
    assert (df["d_min"] == 1).all()


def test_weight_stats_has_expected_columns(small_run: Path):
    run_pipeline(AnalysisConfig(run_dir=small_run, only=["weight_stats"], verbose=False))
    df = pl.read_parquet(small_run / "data" / "n=0008" / "p=0.0000"
                                     / "analysis" / "weight_stats.parquet")
    for col in ("sample_seed", "abar", "wbar", "pei_proxy", "runtime_s"):
        assert col in df.columns, f"missing column {col!r}"


def test_entropy_profile_h5_shape(small_run: Path):
    run_pipeline(AnalysisConfig(run_dir=small_run, only=["entropy_profile"], verbose=False))
    path = (small_run / "data" / "n=0012" / "p=0.0000"
                      / "analysis" / "entropy_profile.h5")
    with h5py.File(path, "r") as f:
        assert f.attrs["n"] == 12
        S = f.attrs["n_samples"]
        assert f["S_of_A"].shape == (S, 6)     # ell = 1..n/2 = 6
        assert f["ell_values"][0] == 1


def test_weight_spectrum_h5_shape(small_run: Path):
    run_pipeline(AnalysisConfig(run_dir=small_run, only=["weight_spectrum"], verbose=False))
    path = (small_run / "data" / "n=0008" / "p=0.0000"
                      / "analysis" / "weight_spectrum.h5")
    with h5py.File(path, "r") as f:
        assert f["weights"].shape == (f.attrs["n_samples"], 2 * 8)


def test_logical_weights_schema(small_run: Path):
    run_pipeline(AnalysisConfig(run_dir=small_run, only=["logical_weights"], verbose=False))
    df = pl.read_parquet(small_run / "data" / "n=0008" / "p=0.0000"
                                     / "analysis" / "logical_weights.parquet")
    for col in ("d_logical_x", "d_logical_y", "d_logical_z",
                "d_logical_min", "method"):
        assert col in df.columns, f"missing column {col!r}"


def test_aggregates_has_wilson_for_binary(small_run: Path):
    run_pipeline(AnalysisConfig(run_dir=small_run, verbose=False))
    df = pl.read_parquet(small_run / "aggregates.parquet")
    # obs.p_k_gt_0 is binary => should have both unconditional and conditional Wilson columns
    wilson_cols = {c for c in df.columns
                   if c.startswith("obs.p_k_gt_0_wilson")}
    assert "obs.p_k_gt_0_wilson_lower" in wilson_cols
    assert "obs.p_k_gt_0_wilson_upper" in wilson_cols
    # obs.tmi is continuous => should NOT have Wilson columns
    tmi_wilson = [c for c in df.columns if c.startswith("obs.tmi_wilson")]
    assert tmi_wilson == []


def test_aggregates_has_conditional_means(small_run: Path):
    run_pipeline(AnalysisConfig(run_dir=small_run, verbose=False))
    df = pl.read_parquet(small_run / "aggregates.parquet")
    cond_cols = [c for c in df.columns if c.endswith("_cond")]
    assert cond_cols, "expected conditional columns suffixed with _cond"


# Resume / idempotency

def test_skip_and_force_behavior(tmp_path: Path):
    cfg = RunConfig(
        circuit=CircuitConfig(graph_spec="cycle", n=8,
                              matching_mode="round_robin",
                              measurement_mode="uniform",
                              depth_mode="O(n)", depth_factor=2,
                              p=0.1, base_seed=42),
        sizes=[8], p_min=0.0, p_max=0.2, n_p=2,
        n_samples_per_cell=2, output_root=tmp_path,
        save_tableaus=True, n_workers=1,
    )
    run_dir = SweepDriver(cfg, progress=False).run()

    # Pass 1: skip expensive
    r1 = run_pipeline(AnalysisConfig(run_dir=run_dir,
                                     skip=["weight_spectrum", "logical_weights"],
                                     verbose=False))
    statuses_by_name = {
        name: [r.status for r in r1.cell_results if r.name == name]
        for name in ("distances", "weight_spectrum", "logical_weights")
    }
    assert all(s == "computed" for s in statuses_by_name["distances"])
    assert statuses_by_name["weight_spectrum"] == []      # filtered out entirely
    assert statuses_by_name["logical_weights"] == []

    # Pass 2: default (fills in missing)
    r2 = run_pipeline(AnalysisConfig(run_dir=run_dir, verbose=False))
    dist_statuses = [r.status for r in r2.cell_results if r.name == "distances"]
    spec_statuses = [r.status for r in r2.cell_results if r.name == "weight_spectrum"]
    assert all(s == "existing" for s in dist_statuses)
    assert all(s == "computed" for s in spec_statuses)

    # Pass 3: force distances
    r3 = run_pipeline(AnalysisConfig(run_dir=run_dir, force=["distances"],
                                     verbose=False))
    dist_statuses = [r.status for r in r3.cell_results if r.name == "distances"]
    assert all(s == "computed" for s in dist_statuses)
    spec_statuses = [r.status for r in r3.cell_results if r.name == "weight_spectrum"]
    assert all(s == "existing" for s in spec_statuses)


def test_manifest_analysis_summary(small_run: Path):
    run_pipeline(AnalysisConfig(run_dir=small_run, verbose=False))
    manifest = json.loads((small_run / "manifest.json").read_text())
    assert "analysis_summary" in manifest
    summary = manifest["analysis_summary"]
    assert "per_cell" in summary and "run_level" in summary
    # Every cell should list its analyses
    for key, analyses in summary["per_cell"].items():
        assert "distances" in analyses
        assert "weight_stats" in analyses
    assert "aggregates" in summary["run_level"]
