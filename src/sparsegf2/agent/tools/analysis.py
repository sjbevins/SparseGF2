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


DEFAULT_RATES = [0.01, 0.02, 0.05, 0.1]


def erasure_at(res, layer: int, *, rates, delta: float, mask=None) -> dict[str, Any]:
    """Erasure statistics at one depth over the selected rows (all circuits and those with k>0)."""
    n = res.spec.num_system
    ff = probe_by_kind(res, "erasure_first_failure")
    if ff is None:
        raise ValueError(
            "this run has no erasure_first_failure probe; add "
            "erasure_first_failure(samples=512) to its probes"
        )
    c = at_layer(res, ff, layer)
    rows = np.ones(res.num_trajectories, bool) if mask is None else mask
    first = np.asarray(res.values[ff])[rows, c]
    cn = probe_by_kind(res, "uncorrectable_counts")
    cc = at_layer(res, cn, layer) if cn else None
    counts = np.asarray(res.values[cn])[rows, cc] if cc is not None else None
    kn = probe_by_kind(res, "code_dimension")
    kc = at_layer(res, kn, layer) if kn else None
    k = np.asarray(res.values[kn])[rows, kc] if kc is not None else None
    code = (k > 0) if k is not None else (first.min(axis=-1) <= n)
    out: dict[str, Any] = {
        "layer": int(layer),
        "count": int(rows.sum()),
        "q_plus": float(code.mean()) if code.size else None,
        "k_over_n": analysis.mean_and_error(k / n) if k is not None and k.size else None,
    }
    if first.shape[0]:
        phi = analysis.erasure_failure_probability(first, n, rates, exact_counts=counts)
        out["phi_all"] = [analysis.mean_and_error(phi[:, i]) for i in range(len(rates))]
    if not code.any():
        out["note"] = f"no selected trajectory holds a code at layer {int(layer)}"
        return out
    cnt = counts[code] if counts is not None else None
    out["phi_code"] = [analysis.mean_and_error(phi[code, i]) for i in range(len(rates))]
    out["e_delta"] = analysis.erasure_tolerance(first[code], n, delta, exact_counts=cnt)
    out["mu_D"] = analysis.mean_and_error(
        analysis.mean_first_failure(first[code], n, exact_counts=cnt)
    )
    low, high = analysis.distance_bounds(first[code], n, exact_counts=cnt)
    exact = low == high
    dist: dict[str, Any] = {
        "exact_fraction": float(exact.mean()),
        "upper_median": float(np.median(high)),
    }
    if counts is not None:
        dist["w"] = int(counts.shape[-1])
        dist["d_values"] = {int(d): int((low[exact] == d).sum()) for d in np.unique(low[exact])}
    out["distance"] = dist
    out["p_c_code"] = analysis.hybrid_failure_by_size(first[code], n, exact_counts=cnt).mean(axis=0)
    out["dips"] = int(analysis.junction_dip(first[code], n, cnt).sum()) if cnt is not None else 0
    return out


def _erasure(ctx, a):
    res, _, path = load_result(ctx, a["source"])
    ff = probe_by_kind(res, "erasure_first_failure")
    if ff is None:
        erasure_at(res, 0, rates=[0.1], delta=0.01)  # raises the explanatory error
    rates = a.get("rates") or DEFAULT_RATES
    layers = np.asarray(res.layers(ff))
    rows = []
    for layer in _wanted_layers(layers, a.get("depths")):
        mask = select(
            res,
            layer=layer,
            where=a.get("where"),
            trajectories=a.get("trajectory_range"),
            graphs=a.get("graph_range"),
        )
        rows.append(erasure_at(res, layer, rates=rates, delta=a["delta"], mask=mask))
    return {
        "source": ctx.show(path),
        "n": res.spec.num_system,
        "rates": rates,
        "delta": a["delta"],
        "rows": rows,
    }


def _render_erasure(d, ctx):
    rates, n = d["rates"], d["n"]
    cols = [
        "layer",
        "count",
        "k/n",
        "q+",
        "mu_D(k>0)",
        *[f"Phi({fmt(e)})" for e in rates],
        f"Phi({fmt(rates[-1])})|k>0",
        f"e_{fmt(d['delta'])}|k>0",
    ]
    rows = []
    for r in d["rows"]:
        phi_all = r.get("phi_all") or [None] * len(rates)
        rows.append(
            [
                r["layer"],
                r["count"],
                pm(*r["k_over_n"]) if r.get("k_over_n") else "-",
                r.get("q_plus"),
                pm(*r["mu_D"]) if r.get("mu_D") else "-",
                *[pm(*p) if p else "-" for p in phi_all],
                pm(*r["phi_code"][-1]) if r.get("phi_code") else "-",
                r.get("e_delta"),
            ]
        )
    parts = [f"analysis_erasure: n={n}, {d['source']}", table(cols, rows)]
    parts += [r["note"] for r in d["rows"] if "note" in r]
    last = d["rows"][-1] if d["rows"] else None
    if last and last.get("distance"):
        dist = last["distance"]
        line = (
            f"distance at layer {last['layer']} (k>0): exact for {dist['exact_fraction']:.3g} of "
            "codes"
        )
        if "d_values" in dist:
            line += f" (d <= {dist['w']}: {dist['d_values']}); others d > {dist['w']}"
        line += f"; median upper bound min_j D_j = {fmt(dist['upper_median'])}"
        parts.append(line)
        parts.append(f"P_C(m) at layer {last['layer']} (k>0, hybrid): " + curve(last["p_c_code"]))
        if last.get("dips"):
            parts.append(
                f"note: {last['dips']} code(s) have sampled P_C(w+1) < exact P_C(w) "
                "(sampling noise): use more orderings (erasure_first_failure(samples=...))"
            )
    return lines(*parts)


def km_curve(res) -> dict[str, np.ndarray]:
    tau = res.purification_layer
    if tau is None:
        raise ValueError("this run has no reference, so no purification times")
    tau = np.asarray(tau)
    final = np.asarray(res.final_layer)
    return analysis.kaplan_meier(np.where(tau >= 0, tau, final), tau >= 0)


def tau_median_all(km: dict[str, np.ndarray]) -> float | None:
    below = np.flatnonzero(km["survival"] <= 0.5)
    return float(km["times"][below[0]]) if below.size else None


def _purification(ctx, a):
    res, _, path = load_result(ctx, a["source"])
    km = km_curve(res)
    tau = np.asarray(res.purification_layer)
    purified = tau[tau >= 0]
    out: dict[str, Any] = {
        "source": ctx.show(path),
        "trajectories": int(tau.size),
        "censored_fraction": float(np.mean(tau < 0)),
        "purified": int(purified.size),
        "tau_median_all": tau_median_all(km),
    }
    if purified.size:
        out["tau_mean_purified"] = analysis.mean_and_error(purified)
        out["tau_quantiles_purified"] = {
            f"q{round(q * 100)}": float(np.quantile(purified, q)) for q in a["quantiles"]
        }
    surv = {}
    for layer in a.get("at_layers") or []:
        i = np.searchsorted(km["times"], layer, side="right") - 1
        surv[int(layer)] = float(km["survival"][i]) if i >= 0 else 1.0
    out["survival_at"] = surv
    out["km_times"], out["km_survival"] = km["times"], km["survival"]
    return out


def _render_purification(d, ctx):
    parts = [
        f"analysis_purification: {d['trajectories']} trajectories in {d['source']}",
        f"censored {d['censored_fraction']:.3g}, purified {d['purified']}, tau median overall "
        f"{fmt(d['tau_median_all'])} (Kaplan-Meier; '-' if under half purified)",
    ]
    if "tau_mean_purified" in d:
        parts.append(
            "among purified: mean "
            + pm(*d["tau_mean_purified"])
            + ", quantiles "
            + ", ".join(f"{k}={fmt(v)}" for k, v in d["tau_quantiles_purified"].items())
        )
    for layer, s in d["survival_at"].items():
        parts.append(f"survival at layer {layer}: {fmt(s)}")
    if len(d["km_times"]):
        n_t = len(d["km_times"])
        idx = np.unique(np.linspace(0, n_t - 1, min(9, n_t)).round().astype(int))
        parts.append(
            "survival curve (layer: fraction unpurified): "
            + ", ".join(f"{fmt(d['km_times'][i])}: {fmt(d['km_survival'][i], 3)}" for i in idx)
        )
    return lines(*parts)


register(
    ToolDef(
        name="analysis_erasure",
        stack="analysis",
        summary="Phi(e), e_delta, mu_D, distance, P_C(m).",
        description=(
            "Erasure protection of the codes in a saved run with erasure_first_failure (and "
            "optionally uncorrectable_region_counts for exact small sizes): per depth, the code "
            "rate k/n, q+ (fraction with k>0), mu_D, Phi(e) = probability that erasing each qubit "
            "with probability e is uncorrectable (all circuits, and k>0), e_delta = sup{e : Phi(e) "
            '<= delta} for k>0, distance bounds and P_C(m). Example: {"source": '
            '"runs/ensemble_run-3f2a9c1b7d", "rates": [0.01, 0.05, 0.1], "delta": 0.01}'
        ),
        params=(
            SOURCE,
            Param(
                "rates",
                "array",
                "Erasure rates e (default 0.01, 0.02, 0.05, 0.1).",
                items="number",
                minimum=0,
                maximum=1,
            ),
            Param(
                "delta",
                "number",
                "Failure probability for e_delta.",
                default=0.01,
                minimum=0,
                maximum=1,
            ),
            Param("depths", "string", "Layers, e.g. [128,256] (default: up to 12)."),
            *SELECTION_PARAMS,
        ),
        handler=_erasure,
        render=_render_erasure,
        long_running=True,
    )
)

register(
    ToolDef(
        name="analysis_purification",
        stack="analysis",
        summary="Purification times with censoring.",
        description=(
            "Purification of the reference in a saved run: censored fraction, tau among purified "
            "trajectories, the median over all (Kaplan-Meier, defined when over half purified), "
            "survival at chosen layers and the survival curve. Never averages censored times. "
            'Example: {"source": "runs/ensemble_run-3f2a9c1b7d", "at_layers": [64, 256]}'
        ),
        params=(
            SOURCE,
            Param(
                "quantiles",
                "array",
                "Quantiles of tau among purified.",
                items="number",
                default=[0.1, 0.25, 0.5, 0.75, 0.9],
                minimum=0,
                maximum=1,
            ),
            Param("at_layers", "array", "Layers at which to report survival.", items="integer"),
        ),
        handler=_purification,
        render=_render_purification,
    )
)
