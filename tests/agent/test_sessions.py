"""StateStore and JobStore."""

import threading
import time

import pytest

import sparsegf2 as sg
from sparsegf2.agent import sessions
from sparsegf2.agent.sessions import JobStore, StateStore, render_job


def test_state_store_ids_limits_and_listing():
    store = StateStore(max_states=2)
    a = store.add(sg.Simulator(2), "new")
    b = store.add(sg.Simulator(3, phases=False), "new")
    assert (a, b) == ("s1", "s2")
    with pytest.raises(ValueError, match="at most 2 open states"):
        store.add(sg.Simulator(1), "new")
    store.count_measurements(a, 3)
    assert store.info(a) == {
        "state_id": "s1",
        "num_qubits": 2,
        "phases": True,
        "origin": "new",
        "measurements": 3,
    }
    store.close(a)
    with pytest.raises(ValueError, match="unknown state_id 's1'.*open: \['s2'\]"):
        store.get(a)
    assert [s["state_id"] for s in store.list()] == ["s2"]


def test_job_runs_reports_progress_and_completes():
    jobs = JobStore()
    gate = threading.Event()

    def run(job):
        sessions.report(1, 2, "points")
        gate.wait(5)
        sessions.report(2, 2, "points")
        return {"answer": 42}

    status = jobs.start(tool="x_y", key="k", run=run, render=lambda d: f"x_y: {d['answer']}")
    assert status["status"] in ("queued", "running")
    time.sleep(0.05)
    assert jobs.status(status["job_id"])["progress"] == {"done": 1, "total": 2, "unit": "points"}
    gate.set()
    done = jobs.status(status["job_id"], wait_seconds=5)
    assert done["status"] == "completed" and done["result_text"] == "x_y: 42"


def test_same_key_while_running_returns_the_running_job():
    jobs = JobStore()
    gate = threading.Event()
    first = jobs.start(tool="x_y", key="same", run=lambda j: gate.wait(5) or {}, render=str)
    second = jobs.start(tool="x_y", key="same", run=lambda j: {}, render=str)
    assert second["job_id"] == first["job_id"] and "already running" in second["note"]
    gate.set()


def test_failure_and_cancel():
    jobs = JobStore()

    def boom(job):
        raise RuntimeError("bad")

    failed = jobs.status(jobs.start(tool="x_y", key="f", run=boom, render=str)["job_id"], 5)
    assert failed["status"] == "failed" and "RuntimeError: bad" in failed["error"]

    def loop(job):
        while not sessions.cancelled():
            time.sleep(0.01)
        return {"stopped": True}

    started = jobs.start(tool="x_y", key="c", run=loop, render=str)
    jobs.cancel(started["job_id"])
    assert jobs.status(started["job_id"], 5)["status"] == "cancelled"


def test_unknown_job_mentions_restarts():
    with pytest.raises(ValueError, match="unknown job_id"):
        JobStore().status("nope")


def test_render_job_has_next_step():
    text = render_job(
        {
            "job_id": "abc",
            "tool": "x_y",
            "status": "running",
            "kind": "tool",
            "progress": {"done": 1, "total": 4, "unit": "points"},
            "elapsed_seconds": 2.0,
            "output_dir": "runs/a",
            "error": None,
        }
    )
    assert text.startswith("job abc (x_y): running, 1/4 points")
    assert 'next: job_status(job_id="abc", wait_seconds=30)' in text
