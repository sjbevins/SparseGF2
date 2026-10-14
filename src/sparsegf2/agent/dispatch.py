"""Validate, run (here or as a background job), render, cap: one tool call."""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from typing import Any

from sparsegf2.agent.registry import Bound
from sparsegf2.agent.render import cap, jsonable
from sparsegf2.agent.sessions import render_job

__all__ = ["Outcome", "call"]


@dataclass(frozen=True)
class Outcome:
    text: str
    data: dict[str, Any] | None
    is_error: bool


def _start_generic(ctx, bound: Bound, args: dict[str, Any]) -> dict[str, Any]:
    tool = bound.tool
    key = tool.name + json.dumps(args, sort_keys=True, default=str)
    return ctx.jobs.start(
        tool=tool.name,
        key=key,
        run=lambda job: tool.handler(ctx, args),
        render=lambda data: tool.render(data, ctx),
        output_dir=args.get("output_dir"),
    )


def call(ctx, bound: Bound, raw_args: dict[str, Any] | None) -> Outcome:
    tool = bound.tool
    try:
        args = bound.validate(raw_args)
        mode = args.pop("background", "never") if tool.long_running else "never"
        background = mode == "always" or (
            mode == "auto"
            and tool.estimate is not None
            and tool.estimate(ctx, args) > ctx.sync_seconds
        )
        if background:
            data = (
                tool.background(ctx, args) if tool.background else _start_generic(ctx, bound, args)
            )
            text = render_job(data)
        else:
            data = tool.handler(ctx, args)
            text = tool.render(data, ctx)
        full = data.get("output_dir") if isinstance(data, dict) else None
        return Outcome(cap(text, ctx.max_output_chars, full=full), jsonable(data), False)
    except (ValueError, ImportError) as exc:
        return Outcome(f"{tool.name}: {exc}", None, True)
    except Exception as exc:  # a bug: report it plainly, never crash the server
        if not ctx.quiet:
            traceback.print_exc(file=sys.stderr)
        return Outcome(
            f"{tool.name}: internal error ({type(exc).__name__}): {exc}. This is a bug in "
            "SparseGF2; report it with the arguments used.",
            None,
            True,
        )
