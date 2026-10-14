"""Open stabilizer states and background jobs, held in the server process."""

from __future__ import annotations

import itertools
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["JobStore", "StateStore", "cancelled", "from_api_status", "render_job", "report"]


class StateStore:
    def __init__(self, max_states: int = 16):
        self.max_states = int(max_states)
        self._items: dict[str, dict[str, Any]] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def add(self, sim, origin: str) -> str:
        with self._lock:
            if len(self._items) >= self.max_states:
                raise ValueError(
                    f"at most {self.max_states} open states; close one with state_close "
                    f"(open: {sorted(self._items)})"
                )
            state_id = f"s{next(self._ids)}"
            self._items[state_id] = {"sim": sim, "origin": origin, "measurements": 0}
            return state_id

    def _entry(self, state_id: Any) -> dict[str, Any]:
        entry = self._items.get(state_id) if isinstance(state_id, str) else None
        if entry is None:
            raise ValueError(
                f"unknown state_id {state_id!r}; open: {sorted(self._items)} (states do not "
                "survive a server restart; save them with state_save)"
            )
        return entry

    def get(self, state_id: Any):
        return self._entry(state_id)["sim"]

    def count_measurements(self, state_id: str, k: int) -> None:
        self._entry(state_id)["measurements"] += int(k)

    def info(self, state_id: str) -> dict[str, Any]:
        e = self._entry(state_id)
        return {
            "state_id": state_id,
            "num_qubits": e["sim"].num_qubits,
            "phases": e["sim"].phases,
            "origin": e["origin"],
            "measurements": e["measurements"],
        }

    def close(self, state_id: str) -> dict[str, Any]:
        info = self.info(state_id)
        with self._lock:
            del self._items[state_id]
        return info

    def list(self) -> list[dict[str, Any]]:
        return [self.info(s) for s in sorted(self._items, key=lambda s: int(s[1:]))]


_local = threading.local()


def report(done: int, total: int, unit: str) -> None:
    """Progress of the job running in this thread (no-op outside a job)."""
    job = getattr(_local, "job", None)
    if job is not None:
        job.progress = {"done": int(done), "total": int(total), "unit": unit}


def cancelled() -> bool:
    """Whether the job running in this thread was asked to stop."""
    job = getattr(_local, "job", None)
    return bool(job is not None and job.cancel.is_set())


@dataclass
class _Job:
    job_id: str
    tool: str
    key: str
    output_dir: str | None
    status: str = "queued"
    started: float = field(default_factory=time.time)
    finished: float | None = None
    progress: dict[str, Any] | None = None
    error: str | None = None
    result_text: str | None = None
    data: dict[str, Any] | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tool": self.tool,
            "kind": "tool",
            "status": self.status,
            "progress": self.progress,
            "elapsed_seconds": round((self.finished or time.time()) - self.started, 2),
            "output_dir": self.output_dir,
            "error": self.error,
            "result_text": self.result_text,
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def start(
        self,
        *,
        tool: str,
        key: str,
        run: Callable[[Any], dict],
        render: Callable[[dict], str],
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            for job in self._jobs.values():
                if job.key == key and job.status in ("queued", "running"):
                    return {**job.snapshot(), "note": "already running; this is that job"}
            job = _Job(uuid.uuid4().hex[:12], tool, key, output_dir)
            self._jobs[job.job_id] = job
        job.thread = threading.Thread(
            target=self._run,
            args=(job, run, render),
            name=f"sparsegf2-{tool}-{job.job_id}",
            daemon=True,
        )
        job.thread.start()
        return job.snapshot()

    @staticmethod
    def _run(job: _Job, run: Callable[[Any], dict], render: Callable[[dict], str]) -> None:
        _local.job = job
        job.status = "running"
        try:
            data = run(job)
            job.data = data
            job.result_text = render(data)
            job.status = "cancelled" if job.cancel.is_set() else "completed"
        except Exception as exc:  # reported through job_status
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}\n" + "".join(
                traceback.format_exception(exc)[-3:]
            )
        finally:
            job.finished = time.time()
            _local.job = None

    def _get(self, job_id: Any) -> _Job | None:
        with self._lock:
            return self._jobs.get(job_id) if isinstance(job_id, str) else None

    def status(self, job_id: Any, wait_seconds: float = 0.0) -> dict[str, Any]:
        job = self._get(job_id)
        if job is None:
            return self._ensemble("status", job_id, wait_seconds)
        if wait_seconds and job.thread is not None:
            job.thread.join(timeout=float(wait_seconds))
        return job.snapshot()

    def cancel(self, job_id: Any) -> dict[str, Any]:
        job = self._get(job_id)
        if job is None:
            return self._ensemble("cancel", job_id)
        job.cancel.set()
        return job.snapshot()

    def list(self) -> list[dict[str, Any]]:
        from sparsegf2 import api

        ours = [j.snapshot() for j in sorted(self._jobs.values(), key=lambda j: -j.started)]
        theirs = [
            {**row, "kind": "ensemble", "tool": "ensemble"} for row in api.list_jobs()["jobs"]
        ]
        return ours + theirs

    def _ensemble(self, action: str, job_id: Any, wait_seconds: float = 0.0) -> dict[str, Any]:
        from sparsegf2 import api

        try:
            if action == "status":
                s = api.job_status(job_id, wait_seconds)
            else:
                s = api.cancel_job(job_id)
        except ValueError:
            raise ValueError(
                f"unknown job_id {job_id!r}; jobs in this server: {sorted(self._jobs)} plus the "
                "ensemble jobs job_list shows (jobs do not survive a server restart; run the same "
                "request again to resume its saved chunks)"
            ) from None
        return from_api_status(s)


def from_api_status(s: dict[str, Any]) -> dict[str, Any]:
    """An ensemble job's status from ``sparsegf2.api`` in this module's status format."""
    return {
        "job_id": s["job_id"],
        "tool": "ensemble",
        "kind": "ensemble",
        "status": s["status"],
        "progress": {
            "done": s["trajectories_done"],
            "total": s["trajectories_total"],
            "unit": "trajectories",
        },
        "elapsed_seconds": s.get("elapsed_seconds"),
        "eta_seconds": s.get("eta_seconds"),
        "output_dir": s.get("output_dir"),
        "error": s.get("error"),
        "result_text": None,
    }


def render_job(status: dict[str, Any]) -> str:
    from sparsegf2.agent.render import fmt

    p = status.get("progress")
    done = f", {p['done']}/{p['total']} {p['unit']}" if p else ""
    eta = f", eta {fmt(status['eta_seconds'])} s" if status.get("eta_seconds") else ""
    head = f"job {status['job_id']} ({status['tool']}): {status['status']}{done}{eta}"
    out = [head]
    if status.get("note"):
        out.append(status["note"])
    if status.get("output_dir"):
        out.append(f"output_dir: {status['output_dir']}")
    if status.get("error"):
        out.append(f"error: {status['error'].strip()}")
    if status["status"] in ("queued", "running"):
        out.append(f'next: job_status(job_id="{status["job_id"]}", wait_seconds=30)')
    elif status["status"] == "completed" and status.get("result_text"):
        out.append(status["result_text"])
    elif status["status"] == "completed" and status.get("kind") == "ensemble":
        out.append(f'next: analysis_summary(source="{status["output_dir"]}")')
    return "\n".join(out)
