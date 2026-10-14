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
MAX_BUILD_CELLS = {".npz": 20_000_000, ".json": 2_000_000}
SECONDS_PER_N3 = 1e-10


def collection(ctx, a: dict[str, Any], n: int) -> list[np.ndarray]:
    g = graph_dict(a, n, ctx)
    if g["family"] == "file":
        num, gs, _ = graphs.load(g["path"])
        if num != n:
            raise ValueError(
                f"graph_file {a['graph_file']!r} holds graphs on {num} nodes; "
                f"call again with num_nodes={num}"
            )
        return gs
    built = api._build_graph(g, n)
    return built if isinstance(built, list) else [built]


def _build(ctx, a):
    n = a["num_nodes"]
    path = ctx.path(a["output_file"], what="output_file")
    suffix = path.suffix
    if suffix not in MAX_BUILD_CELLS:
        raise ValueError(f"output_file must end in .json or .npz, got {path.name!r}")
    count = 1 if a.get("graph") == "file" else a.get("graph_count", 1)
    if count * n > MAX_BUILD_CELLS[suffix]:
        hint = "; write .npz instead (it holds ten times more)" if suffix == ".json" else ""
        raise ValueError(
            f"graph_count x num_nodes = {count} x {n} exceeds the {suffix} limit of "
            f"{MAX_BUILD_CELLS[suffix]:,}; lower graph_count or num_nodes{hint}"
        )
    gs = collection(ctx, a, n)
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
    total = len(gs)
    gs = gs[:MAX_GRAPHS]
    rows = [graphs.stats(g, n) for g in gs]
    keys = ("lambda2", "mean_path_length", "clustering", "degree_mean")
    out: dict[str, Any] = {
        "num_nodes": n,
        "graphs": len(gs),
        "graphs_total": total,
        "connected_fraction": float(np.mean([r["connected"] for r in rows])),
    }
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            m, e = analysis.mean_and_error(vals)
            out[key] = (float(m), float(e))
    out["degree_min"] = min(r["degree_min"] for r in rows)
    out["degree_max"] = max(r["degree_max"] for r in rows)
    family = graph_dict(a, n, ctx)["family"]
    if family in ("watts_strogatz", "newman_watts"):
        k = graph_dict(a, n, ctx)["k"]
        sf = np.array([graphs.shortcut_fraction(g, n, k) for g in gs])
        m, e = analysis.mean_and_error(sf)
        out["shortcut_fraction"] = (float(m), float(e))
    return out


def _estimate(ctx, a):
    n = a["num_nodes"]
    if a.get("graph") == "file":
        count = len(graphs.load(ctx.path(a["graph_file"], what="graph_file", must_exist=True))[1])
    else:
        count = a.get("graph_count", 1)
    return min(count, MAX_GRAPHS) * SECONDS_PER_N3 * n**3


def _render_stats(d, ctx):
    rows = [
        [k, pm(*d[k])]
        for k in ("lambda2", "mean_path_length", "clustering", "degree_mean", "shortcut_fraction")
        if k in d
    ]
    if d["graphs_total"] > d["graphs"]:
        which = f"statistics over the first {d['graphs']} of {d['graphs_total']} graphs"
    else:
        which = f"{d['graphs']} graphs"
    degrees_line = f"degree range over all graphs: {d['degree_min']}..{d['degree_max']}"
    return lines(
        f"graph_stats: {which} on {d['num_nodes']} nodes, "
        f"{d['connected_fraction']:.3g} connected (mean +- stderr over graphs)",
        table(["statistic", "value"], rows),
        degrees_line,
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
        estimate=_estimate,
        examples=({"num_nodes": 16, "graph": "watts_strogatz", "graph_count": 3},),
    )
)
