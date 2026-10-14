"""graph_build and graph_stats: interaction graphs of any family or file."""

from __future__ import annotations

from typing import Any

import numpy as np

from sparsegf2 import analysis, api, graphs
from sparsegf2.agent.params import GRAPH_PARAMS, graph_dict
from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.render import lines, pm, table

NUM_NODES = Param("num_nodes", "integer", "Number of nodes (qubits).", required=True, minimum=2)
MAX_GRAPHS = 1000


def collection(ctx, a: dict[str, Any], n: int) -> list[np.ndarray]:
    g = graph_dict(a, n, ctx)
    if g["family"] == "file":
        num, gs, _ = graphs.load(g["path"])
        if num != n:
            raise ValueError(f"graph_file has num_nodes {num}, not {n}")
        return gs
    built = api._build_graph(g, n)
    return built if isinstance(built, list) else [built]


def _build(ctx, a):
    n = a["num_nodes"]
    gs = collection(ctx, a, n)
    path = ctx.path(a["output_file"], what="output_file")
    path.parent.mkdir(parents=True, exist_ok=True)
    graphs.save(path, gs, n)
    deg = np.concatenate([graphs.degrees(g, n) for g in gs])
    out = {
        "output_file": ctx.show(path),
        "num_nodes": n,
        "graphs": len(gs),
        "edges_mean": float(np.mean([len(g) for g in gs])),
        "degree_min": int(deg.min()),
        "degree_mean": float(deg.mean()),
        "degree_max": int(deg.max()),
    }
    if a.get("show_edges"):
        if n > 64:
            raise ValueError("show_edges lists edges only for graphs of up to 64 nodes")
        out["edges_of_graph_0"] = gs[0].tolist()
    return out


def _render_build(d, ctx):
    head = (
        f"graph_build: {d['graphs']} graphs on {d['num_nodes']} nodes -> {d['output_file']} "
        f"(edges {d['edges_mean']:.4g} per graph, degree {d['degree_min']}..{d['degree_max']}, "
        f"mean {d['degree_mean']:.4g})"
    )
    edges = f"edges of graph 0: {d['edges_of_graph_0']}" if "edges_of_graph_0" in d else ""
    return lines(
        head,
        edges,
        f'next: graph_stats(num_nodes={d["num_nodes"]}, graph="file", '
        f'graph_file="{d["output_file"]}")',
    )


def _stats(ctx, a):
    n = a["num_nodes"]
    gs = collection(ctx, a, n)
    if len(gs) > MAX_GRAPHS:
        gs = gs[:MAX_GRAPHS]
    rows = [graphs.stats(g, n) for g in gs]
    keys = ("lambda2", "mean_path_length", "clustering", "degree_mean")
    out: dict[str, Any] = {
        "num_nodes": n,
        "graphs": len(gs),
        "connected_fraction": float(np.mean([r["connected"] for r in rows])),
    }
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            m, e = analysis.mean_and_error(vals)
            out[key] = (float(m), float(e))
    family = a.get("graph", "cycle")
    if family in ("watts_strogatz", "newman_watts"):
        k = graph_dict(a, n, ctx)["k"]
        sf = np.array([graphs.shortcut_fraction(g, n, k) for g in gs])
        m, e = analysis.mean_and_error(sf)
        out["shortcut_fraction"] = (float(m), float(e))
    return out


def _render_stats(d, ctx):
    rows = [
        [k, pm(*d[k])]
        for k in ("lambda2", "mean_path_length", "clustering", "degree_mean", "shortcut_fraction")
        if k in d
    ]
    return lines(
        f"graph_stats: {d['graphs']} graphs on {d['num_nodes']} nodes, "
        f"{d['connected_fraction']:.3g} connected (mean +- stderr over graphs)",
        table(["statistic", "value"], rows),
    )


register(
    ToolDef(
        name="graph_build",
        stack="graph",
        summary="Build a graph collection and save it to a file.",
        description=(
            "Build a collection of interaction graphs (any family; graph i from seed "
            "(graph_seed, i)) and save it as .json or .npz, to inspect or to reuse with "
            "graph=file. "
            'Example: {"num_nodes": 64, "graph": "watts_strogatz", "graph_k": 4, '
            '"graph_beta": 0.1, "graph_count": 50, "output_file": "graphs/ws64.graphs.json"}'
        ),
        params=(
            NUM_NODES,
            *GRAPH_PARAMS,
            Param("output_file", "string", "Destination (.json or .npz).", required=True),
            Param(
                "show_edges",
                "boolean",
                "Also list graph 0's edges (up to 64 nodes).",
                default=False,
            ),
        ),
        handler=_build,
        render=_render_build,
        read_only=False,
        examples=({"num_nodes": 8, "graph": "cycle", "output_file": "example.graphs.json"},),
    )
)

register(
    ToolDef(
        name="graph_stats",
        stack="graph",
        summary="lambda_2, path length, clustering, shortcuts.",
        description=(
            "Statistics of a graph collection, mean +- standard error over its graphs: "
            "algebraic connectivity lambda_2, fraction connected, mean shortest-path length, "
            "clustering, degrees, and for ring-based families the fraction of shortcut edges "
            "(n <= 4096). "
            'Example: {"num_nodes": 64, "graph": "watts_strogatz", "graph_beta": 0.1, '
            '"graph_count": 20}'
        ),
        params=(NUM_NODES, *GRAPH_PARAMS),
        handler=_stats,
        render=_render_stats,
        long_running=True,
        examples=({"num_nodes": 16, "graph": "watts_strogatz", "graph_count": 3},),
    )
)
