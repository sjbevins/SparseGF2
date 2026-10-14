"""Analysis of saved ensemble results: summaries, any probe, erasure, purification, comparisons."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from sparsegf2 import analysis, api
from sparsegf2.agent.grammar import parse_checkpoints, parse_qubits
from sparsegf2.agent.params import SOURCE
from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.render import curve, fmt, lines, pm, table

_WHERE = re.compile(
    r"^\s*([A-Za-z_][\w\[\]#.]*)\s*(<=|>=|==|!=|<|>)\s*(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*$"
)
_OPS = {
    "<": np.less,
    ">": np.greater,
    "<=": np.less_equal,
    ">=": np.greater_equal,
    "==": np.equal,
    "!=": np.not_equal,
}

SELECTION_PARAMS = (
    Param(
        "where",
        "string",
        "Select trajectories: code (k>0), purified, unpurified, or '<scalar probe> <op> <number>' "
        "e.g. 'half_cut_entropy > 2'; evaluated at each depth.",
    ),
    Param("trajectory_range", "string", "Trajectory numbers to use, e.g. 0:100 or [3,7]."),
    Param("graph_range", "string", "Graph indices to use, e.g. 0:10."),
)


def load_result(ctx, source: str):
    path = ctx.path(source, what="source", must_exist=True)
    if path.is_dir():
        if (path / "sweep.json").exists():
            raise ValueError(
                f"{ctx.show(path)} is a sweep: use analysis_compare(source=...) or "
                "point source at one of its point folders"
            )
        if (
            path / "trajectory.json"
        ).exists():  # a traj_run record: one row, global trajectory number
            return _read(ctx, path, lambda: _load_record(path))
        res, meta = _read(ctx, path, lambda: api._load_directory(path))
        if res is None:
            raise ValueError(f"no saved results in {ctx.show(path)}")
        return res, meta, path
    return _read(ctx, path, lambda: api._load_result_file(path)), None, path


def _load_record(path):
    meta = json.loads((path / "trajectory.json").read_text(encoding="utf-8"))
    return api._load_result_file(path / "result.npz"), meta, path


def _read(ctx, path, load):
    try:
        return load()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"could not read {ctx.show(path)}: {exc}; is it a SparseGF2 result?"
        ) from None


def indices(value: Any, what: str) -> np.ndarray:
    if isinstance(value, str) and re.fullmatch(r"\s*\[?\s*-?\d*\s*:.*", value):
        parts = value.strip().split(":")
        if not parts[0].strip() or not parts[1].strip():
            raise ValueError(
                f"{what}: start and stop must be explicit (like 0:10 or 0:10:2), got {value!r}"
            )
    q = parse_qubits(value, what=what)
    if isinstance(q, str):
        parts = q.split(":")
        if not parts[0] or not parts[1]:
            raise ValueError(
                f"{what}: start and stop must be explicit (like 0:10 or 0:10:2), got {q!r}"
            )
        return np.arange(*[int(x) for x in parts if x])
    return np.asarray(q, dtype=np.int64)


def probe_by_kind(res, kind: str) -> str | None:
    for p in res.spec.probes:
        if p.kind == kind:
            return p.name
    return None


def at_layer(res, name: str, layer: int) -> int | None:
    hits = np.flatnonzero(np.asarray(res.layers(name)) == layer)
    return int(hits[0]) if hits.size else None


def _text(value: Any) -> str:
    return str(value).replace(" ", "") if isinstance(value, str) else str(value)


def _span(values: np.ndarray) -> str:
    u = np.unique(values)
    if u.size > 1 and u[-1] - u[0] + 1 == u.size:
        return f"{u[0]}..{u[-1]}"
    return str(u.tolist())


def select(
    res, *, layer: int, where: str | None = None, trajectories: Any = None, graphs: Any = None
) -> np.ndarray:
    mask = np.ones(res.num_trajectories, dtype=bool)
    if trajectories is not None:
        have = np.asarray(res.trajectories)
        asked = indices(trajectories, "trajectory_range")
        keep = np.isin(have, asked)
        if not keep.any():
            raise ValueError(
                f"trajectory_range {_text(trajectories)} matches none of the saved "
                f"trajectories {_span(have)}"
            )
        mask &= keep
    if graphs is not None:
        have = np.asarray(res.graph_index)
        asked = indices(graphs, "graph_range")
        keep = np.isin(have, asked)
        if not keep.any():
            raise ValueError(
                f"graph_range {_text(graphs)} matches none of the saved graphs {_span(have)}"
            )
        mask &= keep
    if not where or where == "all":
        return mask
    w = where.strip()
    if w in ("purified", "unpurified"):
        tau = res.purification_layer
        if tau is None:
            raise ValueError("where=purified needs a reference (purification data)")
        purified = (tau >= 0) & (tau <= layer)
        return mask & (purified if w == "purified" else ~purified)
    if w == "code":
        name = probe_by_kind(res, "code_dimension")
        if name is None:
            raise ValueError("where=code needs the code_dimension probe in this run")
        c = at_layer(res, name, layer)
        if c is None:
            raise ValueError(f"where=code: code_dimension was not recorded at layer {layer}")
        return mask & (res.values[name][:, c] > 0)
    m = _WHERE.match(w)
    if m is None:
        raise ValueError(
            f"where={where!r}: use code, purified, unpurified or "
            "'<scalar probe> <op> <number>' with op one of < > <= >= == !="
        )
    name, op, number = m.groups()
    if name not in res.values or np.ndim(res.values[name]) != 2:
        raise ValueError(
            f"where: {name!r} is not a scalar probe of this run "
            f"(scalar probes: {[k for k, v in res.values.items() if np.ndim(v) == 2]})"
        )
    c = at_layer(res, name, layer)
    if c is None:
        raise ValueError(f"where: {name} was not recorded at layer {layer}")
    return mask & _OPS[op](res.values[name][:, c], float(number))


def _summary(ctx, a):
    res, meta, path = load_result(ctx, a["source"])
    extra: dict[str, Any] = {"path": str(path)}
    if meta is not None and "spec" in meta:
        extra["spec"] = meta.get("spec")
        extra["trajectories_requested"] = meta.get("trajectories")
    return {"summary": api._bounded_summary(res, extra), "source": ctx.show(path)}


def _render_summary(d, ctx):
    from sparsegf2.agent.tools.ensemble import render_summary

    return render_summary("analysis_summary", d["summary"], output_dir=d["source"])


def _wanted_layers(available: np.ndarray, depths: Any) -> list[int]:
    if depths is None:
        if len(available) <= 12:
            return [int(x) for x in available]
        idx = np.unique(np.linspace(0, len(available) - 1, 12).round().astype(int))
        return [int(available[i]) for i in idx]
    asked = parse_checkpoints(depths, what="depths")
    if not isinstance(asked, list):
        raise ValueError("depths must be a list of layers, e.g. [64, 128]")
    missing = sorted(set(asked) - {int(x) for x in available})
    if missing:
        raise ValueError(f"depths {missing} were not recorded; recorded: {available.tolist()}")
    return asked


def _probe(ctx, a):
    res, _, path = load_result(ctx, a["source"])
    name = a["probe"]
    if name not in res.values:
        raise ValueError(f"probe {name!r} is not in this run; probes: {sorted(res.values)}")
    values = np.asarray(res.values[name], dtype=float)
    layers = np.asarray(res.layers(name))
    stat = a["statistic"]
    rows = []
    for layer in _wanted_layers(layers, a.get("depths")):
        c = int(np.flatnonzero(layers == layer)[0])
        mask = select(
            res,
            layer=layer,
            where=a.get("where"),
            trajectories=a.get("trajectory_range"),
            graphs=a.get("graph_range"),
        )
        v = values[mask, c]
        if values.ndim == 3:  # vector probe: per index
            v = v[~np.isnan(v).any(axis=-1)] if v.size else v
            mean = v.mean(axis=0) if len(v) else np.full(values.shape[-1], np.nan)
            vrow: dict[str, Any] = {"layer": layer, "count": int(len(v)), "curve": mean.tolist()}
            if not mask.any():
                vrow["note"] = f"no trajectories match where={a.get('where')}"
            rows.append(vrow)
            continue
        v = v[~np.isnan(v)]
        row: dict[str, Any] = {"layer": layer, "count": int(v.size)}
        if not mask.any():
            row["note"] = f"no trajectories match where={a.get('where')}"
        if not v.size:
            rows.append(row)
            continue
        if stat == "mean":
            m, e = analysis.mean_and_error(v)
            row["mean"], row["stderr"] = float(m), float(e)
        elif stat == "median":
            row["median"] = float(np.median(v))
        elif stat == "quantiles":
            for q in (10, 25, 50, 75, 90):
                row[f"q{q}"] = float(np.percentile(v, q))
        elif stat == "fraction_positive":
            row["fraction_positive"] = float(np.mean(v > 0))
        else:  # histogram of the most common values
            vals, counts = np.unique(v, return_counts=True)
            top = np.argsort(-counts)[:8]
            row["histogram"] = {fmt(vals[i]): int(counts[i]) for i in sorted(top)}
        rows.append(row)
    return {
        "source": ctx.show(path),
        "probe": name,
        "statistic": stat,
        "vector": values.ndim == 3,
        "rows": rows,
    }


_STAT_KEYS = {
    "median": ["median"],
    "quantiles": [f"q{q}" for q in (10, 25, 50, 75, 90)],
    "fraction_positive": ["fraction_positive"],
}


def _render_probe(d, ctx):
    head = f"analysis_probe: {d['probe']} ({d['statistic']}) in {d['source']}"
    notes = [f"layer {r['layer']}: {r['note']}" for r in d["rows"] if "note" in r]
    if d["vector"]:
        body = [
            f"layer {r['layer']} ({r['count']} trajectories): " + curve(r["curve"])
            for r in d["rows"]
        ]
        return lines(head, *body, *notes)
    if d["statistic"] == "mean":
        rows = [[r["layer"], r["count"], pm(r.get("mean"), r.get("stderr"))] for r in d["rows"]]
        return lines(head, table(["layer", "count", "mean"], rows), *notes)
    if d["statistic"] == "histogram":
        body = [f"layer {r['layer']}: {r.get('histogram')}" for r in d["rows"]]
        return lines(head, *body, *notes)
    keys = _STAT_KEYS[d["statistic"]]
    rows = [[r["layer"], r["count"], *[r.get(k) for k in keys]] for r in d["rows"]]
    return lines(head, table(["layer", "count", *keys], rows), *notes)


register(
    ToolDef(
        name="analysis_summary",
        stack="analysis",
        summary="Summary of a saved ensemble run.",
        description=(
            "Summarise a saved ensemble run (an output_dir or a result .npz, including the saved "
            "chunks of a running job): every probe vs depth, purification statistics, the spec. "
            'Example: {"source": "runs/ensemble_run-3f2a9c1b7d"}'
        ),
        params=(SOURCE,),
        handler=_summary,
        render=_render_summary,
    )
)

register(
    ToolDef(
        name="analysis_probe",
        stack="analysis",
        summary="Any statistic of any probe vs depth.",
        description=(
            "A statistic of one probe of a saved run at each depth (mean +- stderr, median, "
            "quantiles, histogram, fraction_positive), over the trajectories you select (where, "
            'trajectory_range, graph_range). A vector probe gives its mean curve. Example: {"source": '
            '"runs/ensemble_run-3f2a9c1b7d", "probe": "half_cut_entropy", "where": "code"}'
        ),
        params=(
            SOURCE,
            Param(
                "probe",
                "string",
                "Probe name as saved (analysis_summary lists them).",
                required=True,
            ),
            Param(
                "statistic",
                "string",
                "Which statistic (a vector probe always gives its mean curve).",
                default="mean",
                enum=("mean", "median", "quantiles", "histogram", "fraction_positive"),
            ),
            Param("depths", "string", "Layers to show, e.g. [64,128] (default: up to 12)."),
            *SELECTION_PARAMS,
        ),
        handler=_probe,
        render=_render_probe,
    )
)
