"""The SparseGF2 MCP server (low-level SDK server with registry-generated flat schemas)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from sparsegf2._version import __version__
from sparsegf2.agent import INTERFACE_VERSION
from sparsegf2.agent.context import Context

__all__ = ["SERVER_NAME", "build", "instructions", "main"]

SERVER_NAME = "sparsegf2"


def instructions(ctx: Context) -> str:
    return (
        f"SparseGF2 {__version__}, interface version {INTERFACE_VERSION}, stack: {ctx.stack}. "
        "Fast stabilizer (Clifford) simulation: states gate by gate, Stim-language circuits, "
        "ensembles of monitored random circuits, single trajectories, codes and erasure, graphs, "
        "analysis of saved results. Structured arguments are call strings like entropy(0:8) or "
        'erasure_first_failure(samples=512)@[64,128]. Start with help_topic(topic="conventions"); '
        f"help_list names everything. Paths are relative to {ctx.root}. Long runs return a "
        "job_id: poll job_status."
    )


def build(
    stack: str = "all",
    *,
    root: str | Path = ".",
    stacks_file: str | Path | None = None,
    max_output_chars: int = 4000,
    sync_seconds: float = 60.0,
    max_states: int = 16,
    quiet: bool = False,
):
    """The server for ``stack``, its context, and the stack resolution (tools and notes)."""
    import anyio
    import anyio.to_thread
    from mcp import types
    from mcp.server.lowlevel import Server

    import sparsegf2.agent.tools  # noqa: F401  (registers every tool)
    from sparsegf2.agent import dispatch, stacks
    from sparsegf2.agent.tools.help import TOPICS

    resolution = stacks.resolve(
        stack, root=Path(root), stacks_file=Path(stacks_file) if stacks_file else None
    )
    ctx = Context(
        Path(root),
        max_output_chars=max_output_chars,
        sync_seconds=sync_seconds,
        max_states=max_states,
        quiet=quiet,
        stack=stack,
        tool_names=tuple(b.name for b in resolution.bounds),
    )
    by_name = {b.name: b for b in resolution.bounds}
    listing = [
        types.Tool(
            name=b.name,
            description=b.tool.description,
            input_schema=b.input_schema(),
            annotations=types.ToolAnnotations(
                read_only_hint=b.tool.read_only, destructive_hint=False, open_world_hint=False
            ),
        )
        for b in resolution.bounds
    ]

    async def on_list_tools(_rctx, _params) -> Any:
        return types.ListToolsResult(tools=listing)

    async def on_call_tool(_rctx, params) -> Any:
        bound = by_name.get(params.name)
        if bound is None:
            text = f"unknown tool {params.name!r}; this server's tools: {sorted(by_name)}"
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)], is_error=True
            )
        out = await anyio.to_thread.run_sync(dispatch.call, ctx, bound, params.arguments or {})
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=out.text)],
            structured_content=out.data,
            is_error=out.is_error,
        )

    def topic_text(topic: str) -> str:
        path = resources.files("sparsegf2.agent").joinpath("topics", f"{topic}.md")
        return path.read_text(encoding="utf-8")

    docs = [
        types.Resource(
            name=t,
            uri=f"docs://sparsegf2/{t}",
            mime_type="text/markdown",
            description=f"SparseGF2 guidance: {t}",
        )
        for t in TOPICS
    ]

    async def on_list_resources(_rctx, _params) -> Any:
        return types.ListResourcesResult(resources=docs)

    async def on_read_resource(_rctx, params) -> Any:
        uri = str(params.uri)
        topic = uri.removeprefix("docs://sparsegf2/")
        if topic not in TOPICS:
            raise ValueError(f"unknown resource {uri!r}; resources: docs://sparsegf2/<topic>")
        return types.ReadResourceResult(
            contents=[
                types.TextResourceContents(
                    uri=uri, mime_type="text/markdown", text=topic_text(topic)
                )
            ]
        )

    srv = Server(
        SERVER_NAME,
        version=__version__,
        title="SparseGF2 stabilizer simulator",
        instructions=instructions(ctx),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
        on_list_resources=on_list_resources,
        on_read_resource=on_read_resource,
    )
    return srv, ctx, resolution


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sparsegf2-mcp", description="Serve SparseGF2 over MCP (stdio)."
    )
    parser.add_argument("--stack", default="all", help="stack, bundle, preset or custom stack")
    parser.add_argument("--root", default=".", help="folder all paths are confined to")
    parser.add_argument("--stacks-file", default=None, help="a stacks.json to read first")
    parser.add_argument("--sync-seconds", type=float, default=60.0)
    parser.add_argument("--max-output-chars", type=int, default=4000)
    parser.add_argument("--max-states", type=int, default=16)
    parser.add_argument("--quiet", action="store_true", help="no stderr output except fatal errors")
    parser.add_argument("--list", action="store_true", help="print the tools and sizes, then exit")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        srv, ctx, res = build(
            args.stack,
            root=args.root,
            stacks_file=args.stacks_file,
            max_output_chars=args.max_output_chars,
            sync_seconds=args.sync_seconds,
            max_states=args.max_states,
            quiet=args.quiet,
        )
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.list:
        total = 0
        for b in res.bounds:
            total += b.size()
            print(f"{b.name:24s} {b.size():6d} chars  {b.tool.summary}")
        print(f"{'total':24s} {total:6d} chars (~{total // 4} tokens), {len(res.bounds)} tools")
        for note in res.notes:
            print(f"note: {note}")
        return 0
    if not args.quiet:
        for note in res.notes:
            print(f"note: {note}", file=sys.stderr)
    import anyio
    from mcp.server.stdio import stdio_server

    async def serve() -> None:
        async with stdio_server() as (read, write):
            await srv.run(read, write, srv.create_initialization_options())

    anyio.run(serve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
