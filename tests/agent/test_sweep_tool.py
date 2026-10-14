"""ensemble_sweep: a grid of ensembles sharing graph and circuit seeds."""

import json

import numpy as np

from sparsegf2 import mipt

BASE = dict(
    num_system=8,
    layers=6,
    graph="watts_strogatz",
    graph_count=2,
    reference="purification",
    probes=["code_dimension"],
    seed=3,
)


def test_grid_runs_every_point_and_shares_seeds(run_tool, ctx):
    out = run_tool(
        "ensemble_sweep", **BASE, grid="measurement_rate=[0.05, 0.3]", background="never"
    )
    assert not out.is_error, out.text
    root = ctx.root / out.data["output_dir"]
    sweep = json.loads((root / "sweep.json").read_text(encoding="utf-8"))
    assert [p["params"] for p in sweep["points"]] == [
        {"measurement_rate": 0.05},
        {"measurement_rate": 0.3},
    ]
    lo, hi = (mipt.EnsembleResult.load(root / p["dir"] / "result.npz") for p in sweep["points"])
    assert np.all(lo.values["code_dimension"] >= hi.values["code_dimension"])  # nested measurements
    assert "next: analysis_compare" in out.text


def test_rerun_resumes_finished_points(run_tool):
    a = run_tool("ensemble_sweep", **BASE, grid="measurement_rate=[0.1]", background="never")
    b = run_tool("ensemble_sweep", **BASE, grid="measurement_rate=[0.1]", background="never")
    assert a.data["output_dir"] == b.data["output_dir"] and not b.is_error


def test_grid_validation(run_tool):
    assert (
        "cannot be swept"
        in run_tool(
            "ensemble_sweep", **BASE, measurement_rate=0.1, grid="reference=[1]", background="never"
        ).text
    )
    assert (
        "both in grid and as an argument"
        in run_tool(
            "ensemble_sweep",
            **BASE,
            measurement_rate=0.1,
            grid="measurement_rate=[0.2]",
            background="never",
        ).text
    )
    no_n = {k: v for k, v in BASE.items() if k != "num_system"}
    assert (
        "num_system"
        in run_tool(
            "ensemble_sweep", **no_n, grid="measurement_rate=[0.2]", background="never"
        ).text
    )


def test_default_is_a_background_job(run_tool):
    out = run_tool("ensemble_sweep", **BASE, grid="measurement_rate=[0.2, 0.25]")
    assert out.text.startswith("job ")
    done = run_tool("job_status", job_id=out.data["job_id"], wait_seconds=60)
    assert "completed" in done.text and "2/2 points" in done.text


def test_bad_grid_is_an_immediate_error_not_a_job(run_tool, ctx):
    before = ctx.jobs.list()
    out = run_tool("ensemble_sweep", **BASE, measurement_rate=0.1, grid="reference=[1]")
    assert out.is_error and "cannot be swept" in out.text
    assert not out.text.startswith("job ")
    assert ctx.jobs.list() == before
