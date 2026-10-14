"""job_status, job_cancel, job_list, and generic background runs through dispatch."""

import threading

from sparsegf2.agent import dispatch, sessions
from sparsegf2.agent.registry import Bound, Param, ToolDef

GATE = threading.Event()


def _slow(ctx, args):
    while not GATE.wait(0.01):
        if sessions.cancelled():
            return {"value": -1}
    return {"value": args["x"] * 2}


SLOW = ToolDef(
    "plot_slow",
    "plot",
    "slow",
    "slow",
    (Param("x", "integer", "x", required=True),),
    _slow,
    lambda d, ctx: f"plot_slow: {d['value']}",
    long_running=True,
    estimate=lambda ctx, a: 1e9,
)


def test_background_job_then_status_returns_the_result(ctx, run_tool):
    GATE.clear()
    started = dispatch.call(ctx, Bound(SLOW), {"x": 21})
    assert not started.is_error and started.text.startswith("job ")
    job_id = started.data["job_id"]
    again = dispatch.call(ctx, Bound(SLOW), {"x": 21})  # same request while running
    assert again.data["job_id"] == job_id and "already running" in again.text
    GATE.set()
    done = run_tool("job_status", job_id=job_id, wait_seconds=10)
    assert "completed" in done.text and "plot_slow: 42" in done.text


def test_never_runs_in_place(ctx):
    GATE.set()
    out = dispatch.call(ctx, Bound(SLOW), {"x": 1, "background": "never"})
    assert out.text == "plot_slow: 2"


def test_cancel_and_list(ctx, run_tool):
    GATE.clear()
    job_id = dispatch.call(ctx, Bound(SLOW), {"x": 5}).data["job_id"]
    out = run_tool("job_cancel", job_id=job_id)
    assert not out.is_error
    assert "cancelled" in run_tool("job_status", job_id=job_id, wait_seconds=10).text
    listing = run_tool("job_list")
    assert job_id in listing.text
    GATE.set()


def test_unknown_job(run_tool):
    out = run_tool("job_status", job_id="zzz")
    assert out.is_error and "unknown job_id 'zzz'" in out.text
