"""ensemble_run: flat arguments, the same numbers as the library, reuse, jobs."""

import numpy as np

from sparsegf2 import graphs, mipt

ARGS = dict(
    num_system=8,
    layers=8,
    measurement_rate=0.15,
    graph="watts_strogatz",
    graph_count=3,
    graph_seed=2,
    circuits_per_graph=2,
    schedule="random_edges",
    reference="purification",
    checkpoints="4",
    seed=11,
    probes=["code_dimension", "erasure_first_failure(samples=16)"],
)


def test_matches_the_library(run_tool, ctx):
    out = run_tool("ensemble_run", **ARGS)
    assert not out.is_error, out.text
    assert out.text.startswith("ensemble_run: n=8, layers=8, p=0.15, 6 trajectories")
    saved = mipt.EnsembleResult.load(ctx.root / out.data["output_dir"] / "result.npz")
    direct = mipt.MonitoredCircuit(
        num_system=8,
        edges=[graphs.watts_strogatz(8, 4, 0.1, seed=(2, g)) for g in range(3)],
        circuits_per_graph=2,
        layers=8,
        measurement_rate=0.15,
        schedule="random_edges",
        reference="purification",
        checkpoints=4,
        probes=[mipt.code_dimension(), mipt.erasure_first_failure(16)],
    ).run(seed=11)
    np.testing.assert_array_equal(saved.values["code_dimension"], direct.values["code_dimension"])
    np.testing.assert_array_equal(
        saved.values["erasure_first_failure"], direct.values["erasure_first_failure"]
    )


def test_estimate_only(run_tool):
    out = run_tool("ensemble_run", **ARGS, estimate_only=True)
    assert "estimated_seconds" in out.text and "nothing was run" in out.text


def test_same_request_again_reuses_the_saved_run(run_tool):
    first = run_tool("ensemble_run", **ARGS)
    second = run_tool("ensemble_run", **ARGS)
    assert second.data["output_dir"] == first.data["output_dir"] and "reused" in second.text


def test_deterministic_graph_needs_trajectories(run_tool):
    out = run_tool("ensemble_run", num_system=8, layers=4, measurement_rate=0.1)
    assert out.is_error and "give trajectories" in out.text
    assert not run_tool(
        "ensemble_run", num_system=8, layers=4, measurement_rate=0.1, trajectories=5
    ).is_error


def test_background_always_then_job_status(run_tool):
    started = run_tool("ensemble_run", **{**ARGS, "seed": 12}, background="always")
    assert started.text.startswith("job ") and started.data["kind"] == "ensemble"
    done = run_tool("job_status", job_id=started.data["job_id"], wait_seconds=60)
    assert "completed" in done.text and "analysis_summary" in done.text


def test_second_request_joins_a_running_job(run_tool):
    big = dict(
        num_system=64,
        layers=400,
        measurement_rate=0.1,
        trajectories=4000,
        seed=1,
        probes=["code_dimension"],
        reference="single",
    )
    a = run_tool("ensemble_run", **big, background="always")
    b = run_tool("ensemble_run", **big, background="always")
    try:
        assert b.data["job_id"] == a.data["job_id"] and "already running" in b.text
    finally:
        run_tool("job_cancel", job_id=a.data["job_id"])
        run_tool("job_status", job_id=a.data["job_id"], wait_seconds=60)
