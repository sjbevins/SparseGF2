"""graph_build and graph_stats."""

import math

from sparsegf2 import graphs


def test_build_then_stats_from_the_file(run_tool, ctx):
    out = run_tool(
        "graph_build",
        num_nodes=16,
        graph="watts_strogatz",
        graph_count=3,
        graph_seed=1,
        output_file="ws.graphs.json",
    )
    assert not out.is_error and "3 graphs" in out.text
    n, gs, _ = graphs.load(ctx.root / "ws.graphs.json")
    assert n == 16 and len(gs) == 3
    stats = run_tool("graph_stats", num_nodes=16, graph="file", graph_file="ws.graphs.json")
    assert "lambda2" in stats.text and "3 graphs" in stats.text


def test_cycle_stats_match_the_closed_form(run_tool):
    out = run_tool("graph_stats", num_nodes=10, graph="cycle")
    assert f"{2 - 2 * math.cos(2 * math.pi / 10):.4g}" in out.text


def test_shortcut_fraction_for_watts_strogatz(run_tool):
    out = run_tool(
        "graph_stats", num_nodes=32, graph="watts_strogatz", graph_beta=0.0, graph_count=2
    )
    assert "shortcut_fraction" in out.text and "0+-0" in out.text.replace(" ", "")


def test_show_edges_small(run_tool):
    out = run_tool(
        "graph_build", num_nodes=4, graph="cycle", output_file="c.graphs.npz", show_edges=True
    )
    assert "[0,1]" in out.text.replace(" ", "")
