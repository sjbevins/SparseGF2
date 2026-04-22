"""Smoke + correctness tests for the plotting primitive."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from sparsegf2.analysis_pipeline import AnalysisConfig, run_pipeline
from sparsegf2.circuits import CircuitConfig, RunConfig, SweepDriver
from sparsegf2.plotting import plot_vs_p, pick_error_metric
from sparsegf2.plotting.errors import sem, wilson, ci95_bootstrap


# ══════════════════════════════════════════════════════════════
# Fixture
# ══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def plot_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("plot_run")
    cfg = RunConfig(
        circuit=CircuitConfig(graph_spec="cycle", n=8,
                              matching_mode="round_robin",
                              measurement_mode="uniform",
                              depth_mode="O(n)", depth_factor=2,
                              p=0.15, base_seed=42),
        sizes=[8, 12, 16], p_min=0.0, p_max=0.3, n_p=4,
        n_samples_per_cell=4, output_root=root,
        save_tableaus=True, n_workers=1,
    )
    run_dir = SweepDriver(cfg, progress=False).run()
    run_pipeline(AnalysisConfig(run_dir=run_dir,
                                skip=["weight_spectrum", "logical_weights"],
                                verbose=False))
    return run_dir


# ══════════════════════════════════════════════════════════════
# Error metric unit tests
# ══════════════════════════════════════════════════════════════

def test_sem_matches_numpy():
    rng = np.random.default_rng(0)
    v = rng.normal(size=30)
    lo, hi = sem(v)
    expected = float(np.std(v, ddof=1)) / math.sqrt(v.size)
    assert abs(lo - expected) < 1e-12 and abs(hi - expected) < 1e-12


def test_wilson_known_case():
    """For k=50, n=100, z=1.96 -> CI approximately [0.404, 0.596]."""
    v = np.array([1] * 50 + [0] * 50)
    lo_half, hi_half = wilson(v)
    phat = 0.5
    assert abs((phat - lo_half) - 0.404) < 0.01
    assert abs((phat + hi_half) - 0.596) < 0.01


def test_bootstrap_ci_seed_reproducible():
    v = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    a = ci95_bootstrap(v, n_resamples=200, seed=7)
    b = ci95_bootstrap(v, n_resamples=200, seed=7)
    assert a == b


def test_pick_error_metric_auto_on_binary():
    binary = np.array([0, 1, 0, 1, 1, 0])
    cont = np.array([0.1, 0.2, 0.3, 0.4])
    assert pick_error_metric("auto", binary) == "wilson"
    assert pick_error_metric("auto", cont) == "sem"


def test_pick_error_metric_explicit_override():
    assert pick_error_metric("std", np.array([0, 1, 0])) == "std"
    with pytest.raises(ValueError):
        pick_error_metric("nonsense", np.array([0.0]))


# ══════════════════════════════════════════════════════════════
# Plot smoke tests
# ══════════════════════════════════════════════════════════════

def test_plot_vs_p_basic(plot_run: Path, tmp_path: Path):
    out = tmp_path / "basic.png"
    fig, ax = plot_vs_p(plot_run, y="obs.p_k_gt_0", save=out)
    assert out.exists() and out.stat().st_size > 0
    assert ax.get_xlabel() == "p"
    assert ax.get_ylabel() == "obs.p_k_gt_0"
    plt.close(fig)


def test_plot_vs_p_with_filter_conditional(plot_run: Path, tmp_path: Path):
    """filter='obs.k > 0' restricts rows before aggregation (conditional mean)."""
    out = tmp_path / "conditional.png"
    fig, _ = plot_vs_p(plot_run, y="obs.k", filter="obs.k > 0",
                       errors="bar", save=out)
    assert out.exists()
    plt.close(fig)


def test_plot_vs_p_size_filter(plot_run: Path, tmp_path: Path):
    """Only requested sizes are drawn."""
    out = tmp_path / "sized.png"
    fig, ax = plot_vs_p(plot_run, y="distances.d_cont",
                        sizes=[8, 16], errors="none", save=out)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert set(labels) == {"n=8", "n=16"}
    plt.close(fig)


def test_plot_vs_p_derived_k_over_n(plot_run: Path, tmp_path: Path):
    out = tmp_path / "k_over_n.png"
    fig, ax = plot_vs_p(plot_run, y="k_over_n",
                        filter="obs.k > 0", errors="band", save=out)
    assert out.exists()
    plt.close(fig)


def test_plot_vs_p_derived_d_cont_over_n(plot_run: Path, tmp_path: Path):
    out = tmp_path / "d_over_n.png"
    fig, _ = plot_vs_p(plot_run, y="d_cont_over_n",
                       filter="obs.k > 0", errors="band", save=out)
    assert out.exists()
    plt.close(fig)


def test_plot_vs_p_all_errors_modes(plot_run: Path, tmp_path: Path):
    for mode in ("band", "bar", "none"):
        out = tmp_path / f"mode_{mode}.png"
        fig, _ = plot_vs_p(plot_run, y="obs.k", errors=mode, save=out)
        assert out.exists(), f"errors={mode} did not produce a file"
        plt.close(fig)


def test_plot_vs_p_explicit_scales(plot_run: Path, tmp_path: Path):
    out = tmp_path / "logscale.png"
    fig, ax = plot_vs_p(
        plot_run, y="obs.p_k_gt_0",
        filter="p > 0",         # log x needs positive values
        xscale="log", errors="band", save=out,
    )
    assert ax.get_xscale() == "log"
    plt.close(fig)


def test_plot_vs_p_title_and_labels(plot_run: Path, tmp_path: Path):
    fig, ax = plot_vs_p(plot_run, y="obs.tmi",
                        title="TMI vs p", xlabel="p", ylabel="TMI",
                        save=tmp_path / "labels.png")
    assert ax.get_title() == "TMI vs p"
    assert ax.get_xlabel() == "p"
    assert ax.get_ylabel() == "TMI"
    plt.close(fig)


def test_plot_vs_p_missing_column_raises(plot_run: Path):
    with pytest.raises(KeyError):
        plot_vs_p(plot_run, y="does.not.exist")
