"""Helpers for the agent tool tests: a context, direct dispatch and an MCP session."""

import asyncio

import pytest

import sparsegf2.agent.tools  # noqa: F401  (registers every tool)
from sparsegf2.agent import dispatch, server
from sparsegf2.agent.context import Context
from sparsegf2.agent.registry import REGISTRY, Bound


@pytest.fixture
def ctx(tmp_path):
    return Context(tmp_path, sync_seconds=60.0)


@pytest.fixture
def run_tool(ctx):
    """Call a tool through dispatch (validation, rendering, caps) and return the Outcome."""

    def call(name, **args):
        return dispatch.call(ctx, Bound(REGISTRY[name]), args)

    return call


@pytest.fixture
def session(tmp_path):
    """``session(stack, fn)`` runs ``await fn(client)`` against an in-process server."""
    import mcp

    def run(stack, fn):
        srv, _, _ = server.build(stack, root=tmp_path, quiet=True)

        async def go():
            async with mcp.Client(srv) as client:
                return await fn(client)

        return asyncio.run(go())

    return run
