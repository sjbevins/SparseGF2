"""The MCP server: listing, calling, errors, flags."""

import json

from sparsegf2.agent import server
from sparsegf2.agent.registry import check_flat


def test_list_tools_are_flat_and_named_by_stack(session):
    tools = session("help", lambda c: c.list_tools()).tools
    assert [t.name for t in tools] == ["help_topic", "help_list"]
    for t in tools:
        assert check_flat(t.input_schema) == []


def test_call_returns_text_and_structured_content(session):
    res = session("help", lambda c: c.call_tool("help_topic", {"topic": "conventions"}))
    assert not res.is_error and res.content[0].text.startswith("help_topic: conventions")
    json.dumps(res.structured_content, allow_nan=False)


def test_errors_are_tool_errors_not_protocol_errors(session):
    res = session("help", lambda c: c.call_tool("help_topic", {"topic": "x", "extra": 1}))
    assert res.is_error and "unknown argument(s) ['extra']" in res.content[0].text
    res = session("help", lambda c: c.call_tool("nope", {}))
    assert res.is_error and "unknown tool 'nope'" in res.content[0].text


def test_help_topics_are_resources(session):
    listed = session("help", lambda c: c.list_resources()).resources
    assert "docs://sparsegf2/conventions" in {str(r.uri) for r in listed}
    read = session("help", lambda c: c.read_resource("docs://sparsegf2/grammar"))
    assert "positional" in read.contents[0].text


def test_instructions_name_the_stack_and_version(tmp_path):
    srv, ctx, res = server.build("help", root=tmp_path, quiet=True)
    text = server.instructions(ctx)
    assert "stack: help" in text and "interface version 1" in text


def test_list_flag_prints_tools_and_sizes(tmp_path, capsys):
    assert server.main(["--stack", "help", "--root", str(tmp_path), "--list"]) == 0
    out = capsys.readouterr().out
    assert "help_topic" in out and "total" in out


def test_bad_stack_exits_2(tmp_path, capsys):
    assert server.main(["--stack", "nope", "--root", str(tmp_path)]) == 2
    assert "unknown stack 'nope'" in capsys.readouterr().err
