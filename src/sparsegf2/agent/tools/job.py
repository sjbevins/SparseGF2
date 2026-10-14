"""job_status, job_cancel, job_list: background work started by any long-running tool."""

from __future__ import annotations

from typing import Any

from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.render import fmt, table
from sparsegf2.agent.sessions import render_job

JOB_ID = Param("job_id", "string", "The job_id a tool returned.", required=True)


def _status(ctx, a: dict[str, Any]) -> dict[str, Any]:
    return ctx.jobs.status(a["job_id"], a.get("wait_seconds", 0.0))


def _cancel(ctx, a: dict[str, Any]) -> dict[str, Any]:
    return ctx.jobs.cancel(a["job_id"])


def _list(ctx, a: dict[str, Any]) -> dict[str, Any]:
    return {"jobs": ctx.jobs.list()}


def _render_list(data: dict[str, Any], ctx) -> str:
    rows = [
        [
            j["job_id"],
            j.get("tool", "ensemble"),
            j["status"],
            (
                f"{j['progress']['done']}/{j['progress']['total']} {j['progress']['unit']}"
                if j.get("progress")
                else f"{fmt(j.get('trajectories_done'))}/{fmt(j.get('trajectories_total'))} trajectories"
            ),
            j.get("output_dir") or "-",
        ]
        for j in data["jobs"]
    ]
    if not rows:
        return "job_list: no jobs in this server process"
    return f"job_list: {len(rows)} job(s)\n" + table(
        ["job_id", "tool", "status", "progress", "output_dir"], rows
    )


register(
    ToolDef(
        name="job_status",
        stack="job",
        summary="Progress or result of a background job.",
        description=(
            "Progress of a background job (queued, running, completed, cancelled, failed). With "
            "wait_seconds (0-60) it waits for the job to finish first. A finished job's status "
            'contains the tool\'s result. Example: {"job_id": "3f2a9c1b7d4e", "wait_seconds": 30}'
        ),
        params=(
            JOB_ID,
            Param(
                "wait_seconds",
                "number",
                "Wait up to this long for the job to end.",
                default=0.0,
                minimum=0,
                maximum=60,
            ),
        ),
        handler=_status,
        render=lambda d, ctx: render_job(d),
        examples=(),
    )
)

register(
    ToolDef(
        name="job_cancel",
        stack="job",
        summary="Stop a background job (saved work is kept).",
        description=(
            "Stop a background job after its current chunk or point. Saved results are kept; "
            'running the same request again resumes. Example: {"job_id": "3f2a9c1b7d4e"}'
        ),
        params=(JOB_ID,),
        handler=_cancel,
        render=lambda d, ctx: render_job(d),
        read_only=False,
        examples=(),
    )
)

register(
    ToolDef(
        name="job_list",
        stack="job",
        summary="Background jobs of this server process.",
        description="List the background jobs started in this server process. Example: {}",
        params=(),
        handler=_list,
        render=_render_list,
        examples=({},),
    )
)
