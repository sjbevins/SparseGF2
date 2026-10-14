"""graph_build and graph_stats."""

import math

import numpy as np

from sparsegf2 import api, graphs
from sparsegf2.agent import dispatch
from sparsegf2.agent.context import Context
from sparsegf2.agent.params import graph_dict
from sparsegf2.agent.registry import REGISTRY, Bound
from sparsegf2.agent.tools.graph import collection


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
    row = next(r for r in out.text.splitlines() if r.startswith("shortcut_fraction"))
    assert float(row.split()[-1].split("+-")[0]) == 0.0


def test_show_edges_small(run_tool):
    out = run_tool(
        "graph_build", num_nodes=4, graph="cycle", output_file="c.graphs.npz", show_edges=True
    )
    assert "[0,1]" in out.text.replace(" ", "")


def test_collection_equals_api_build_graph(ctx):
    args = {"graph": "watts_strogatz", "graph_beta": 0.3, "graph_count": 4, "graph_seed": 5}
    mine = collection(ctx, args, 20)
    ref = api._build_graph(graph_dict(args, 20, ctx), 20)
    assert len(mine) == len(ref) == 4
    for a, b in zip(mine, ref, strict=True):
        assert np.array_equal(a, b)


def test_build_reports_degree_extremes(run_tool):
    out = run_tool("graph_build", num_nodes=6, graph="cycle", output_file="c.graphs.json")
    assert out.data["degree_min"] == 2 and out.data["degree_max"] == 2


def test_stats_goes_to_the_background_when_the_estimate_is_large(tmp_path):
    small = Context(tmp_path, sync_seconds=1e-9)
    out = dispatch.call(
        small,
        Bound(REGISTRY["graph_stats"]),
        {"num_nodes": 16, "graph": "watts_strogatz", "graph_count": 2, "background": "auto"},
    )
    assert not out.is_error and out.data.get("job_id"), out.text


def test_stats_estimate_is_cheap_for_a_small_graph(run_tool):
    out = run_tool("graph_stats", num_nodes=16, graph="cycle", background="auto")
    assert not out.is_error and not out.data.get("job_id")


def test_stats_says_when_it_uses_only_the_first_1000(run_tool, ctx):
    gs = [np.array([[0, 1], [1, 2], [2, 0]])] * 1001
    graphs.save(ctx.root / "many.graphs.npz", gs, 3)
    out = run_tool("graph_stats", num_nodes=3, graph="file", graph_file="many.graphs.npz")
    assert not out.is_error and "first 1000 of 1001" in out.text
    assert out.data["graphs"] == 1000 and out.data["graphs_total"] == 1001


def test_too_large_build_is_refused_and_writes_nothing(run_tool, ctx):
    out = run_tool("graph_build", num_nodes=4096, graph="cycle", output_file="d/x.graphs.json")
    assert not out.is_error
    out = run_tool(
        "graph_build",
        num_nodes=2000,
        graph="watts_strogatz",
        graph_count=2000,
        output_file="big/x.graphs.json",
    )
    assert out.is_error and "graph_count" in out.text and "num_nodes" in out.text
    assert ".npz" in out.text and not (ctx.root / "big").exists()
    out = run_tool(
        "graph_build",
        num_nodes=10000,
        graph="watts_strogatz",
        graph_count=2001,
        output_file="big2/x.graphs.npz",
    )
    assert out.is_error and "graph_count" in out.text and not (ctx.root / "big2").exists()


def test_graph_count_is_capped_by_validation(run_tool):
    out = run_tool(
        "graph_build",
        num_nodes=8,
        graph="watts_strogatz",
        graph_count=api.MAX_GRAPH_REALIZATIONS + 1,
        output_file="x.graphs.npz",
    )
    assert out.is_error and "graph_count" in out.text


def test_bad_suffix_is_refused_before_anything_is_created(run_tool, ctx):
    out = run_tool("graph_build", num_nodes=8, graph="cycle", output_file="newdir/g.txt")
    assert out.is_error and ".json" in out.text and not (ctx.root / "newdir").exists()


def test_num_nodes_mismatch_message(run_tool):
    run_tool("graph_build", num_nodes=16, graph="cycle", output_file="c16.graphs.json")
    out = run_tool("graph_stats", num_nodes=8, graph="file", graph_file="c16.graphs.json")
    assert out.is_error
    assert (
        "graph_file 'c16.graphs.json' holds graphs on 16 nodes; call again with num_nodes=16"
        in out.text
    )


def test_stats_reports_degree_range(run_tool):
    out = run_tool("graph_stats", num_nodes=10, graph="cycle")
    assert out.data["degree_min"] == out.data["degree_max"] == 2
    assert "degree range over all graphs: 2..2" in out.text
