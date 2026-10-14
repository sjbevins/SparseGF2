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


def test_rerun_does_not_recompute_finished_points(run_tool, ctx):
    kw = dict(**BASE, grid="measurement_rate=[0.1, 0.2]", background="never", output_dir="sw")
    assert not run_tool("ensemble_sweep", **kw).is_error
    files = sorted((ctx.root / "sw").glob("*/result.npz"))
    assert len(files) == 2
    before = [f.stat().st_mtime_ns for f in files]
    assert not run_tool("ensemble_sweep", **kw).is_error
    assert [f.stat().st_mtime_ns for f in files] == before


def test_reused_directory_with_other_seed_or_base_is_refused(run_tool, ctx):
    kw = dict(grid="measurement_rate=[0.1]", background="never", output_dir="sw")
    assert not run_tool("ensemble_sweep", **BASE, **kw).is_error
    manifest = ctx.root / "sw" / "sweep.json"
    text = manifest.read_text(encoding="utf-8")
    other_seed = run_tool("ensemble_sweep", **{**BASE, "seed": 4}, **kw)
    assert other_seed.is_error and "holds a different sweep" in other_seed.text
    other_base = run_tool("ensemble_sweep", **{**BASE, "layers": 7}, **kw)
    assert other_base.is_error and "holds a different sweep" in other_base.text
    other_grid = run_tool("ensemble_sweep", **BASE, **{**kw, "grid": "measurement_rate=[0.1, 0.2]"})
    assert other_grid.is_error and "holds a different sweep" in other_grid.text
    assert manifest.read_text(encoding="utf-8") == text
    no_seed = {k: v for k, v in BASE.items() if k != "seed"}
    assert not run_tool("ensemble_sweep", **no_seed, **kw).is_error  # stored seed is reused


def test_close_float_values_get_distinct_directories(run_tool, ctx):
    from sparsegf2.agent.tools.ensemble import _point_name

    a, b = (
        _point_name({"measurement_rate": 0.1234567}),
        _point_name({"measurement_rate": 0.1234568}),
    )
    assert a != b and _point_name({"measurement_rate": 0.05}) == "measurement_rate=0.05"
    out = run_tool(
        "ensemble_sweep", **BASE, grid="measurement_rate=[0.1234567, 0.1234568]", background="never"
    )
    assert not out.is_error, out.text
    dirs = [
        p["dir"]
        for p in json.loads(
            (ctx.root / out.data["output_dir"] / "sweep.json").read_text(encoding="utf-8")
        )["points"]
    ]
    assert len(set(dirs)) == 2


def test_duplicate_grid_values_are_refused_before_anything_runs(run_tool, ctx):
    before = ctx.jobs.list()
    out = run_tool("ensemble_sweep", **BASE, grid="measurement_rate=[0.1, 0.1]")
    assert out.is_error and "duplicate" in out.text
    assert ctx.jobs.list() == before and not (ctx.root / "runs").exists()
