"""Analysis of saved ensemble results: summaries, any probe, erasure, purification, comparisons."""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2 import analysis, api
from sparsegf2.agent.grammar import bind, parse_calls, parse_checkpoints, parse_qubits, plain
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


def _mean_err(values) -> tuple[float, float]:
    """Mean and standard error as floats; an error below 1e-12 * max(1, |mean|) is 0."""
    m, e = analysis.mean_and_error(values)
    m, e = float(m), float(e)
    if e < 1e-12 * max(1.0, abs(m)):
        e = 0.0
    return m, e


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
        "exact_counts": counts is not None,
        "q_plus": float(code.mean()) if code.size else None,
        "k_over_n": _mean_err(k / n) if k is not None and k.size else None,
    }
    if first.shape[0]:
        phi = analysis.erasure_failure_probability(first, n, rates, exact_counts=counts)
        out["phi_all"] = [_mean_err(phi[:, i]) for i in range(len(rates))]
    if not code.any():
        out["note"] = (
            f"no selected trajectory holds a code at layer {int(layer)}"
            if first.shape[0]
            else f"no trajectory selected at layer {int(layer)}"
        )
        return out
    cnt = counts[code] if counts is not None else None
    out["phi_code"] = [_mean_err(phi[code, i]) for i in range(len(rates))]
    out["e_delta"] = float(analysis.erasure_tolerance(first[code], n, delta, exact_counts=cnt))
    out["mu_D"] = _mean_err(analysis.mean_first_failure(first[code], n, exact_counts=cnt))
    low, high = analysis.distance_bounds(first[code], n, exact_counts=cnt)
    exact = low == high
    dist: dict[str, Any] = {
        "codes": int(code.sum()),
        "exact": int(exact.sum()),
        "exact_fraction": float(exact.mean()),
        "exact_d": {int(d): int((low[exact] == d).sum()) for d in np.unique(low[exact])},
        "upper_median_inexact": float(np.median(high[~exact])) if (~exact).any() else None,
    }
    if counts is not None:
        dist["w"] = int(counts.shape[-1])
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
        "probes": {
            "erasure_first_failure": ff,
            "uncorrectable_counts": probe_by_kind(res, "uncorrectable_counts"),
            "code_dimension": probe_by_kind(res, "code_dimension"),
        },
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
    probes = d["probes"]
    used = ", ".join(f"{k}={v}" for k, v in probes.items() if v)
    parts.append(f"probes used (the first of each kind): {used}")
    if probes["uncorrectable_counts"] is None:
        parts.append("no uncorrectable_region_counts probe: all sizes are sampled estimates")
    elif not all(r["exact_counts"] for r in d["rows"]):
        have = [r["layer"] for r in d["rows"] if r["exact_counts"]]
        lack = [r["layer"] for r in d["rows"] if not r["exact_counts"]]
        parts.append(
            f"exact small-size counts used at layers {have}; not recorded at {lack} "
            "(sampled estimates there)"
        )
    last = d["rows"][-1] if d["rows"] else None
    if last and last.get("distance"):
        dist = last["distance"]
        line = (
            f"distance at layer {last['layer']} (k>0): exact for {dist['exact']}/{dist['codes']} "
            f"codes ({100 * dist['exact_fraction']:.3g}%); exact d: {dist['exact_d']}"
        )
        if "w" in dist:
            line += f" (counts are exhaustive up to {dist['w']}, so d = {dist['w'] + 1} is exact)"
        if dist["upper_median_inexact"] is not None:
            line += (
                f"; the other {dist['codes'] - dist['exact']} are only bounded, "
                f"median upper bound min_j D_j = {fmt(dist['upper_median_inexact'])}"
            )
        parts.append(line)
        parts.append(f"P_C(m) at layer {last['layer']} (k>0, hybrid): " + curve(last["p_c_code"]))
        if last.get("dips"):
            parts.append(
                f"note: {last['dips']} code(s) have sampled P_C(w+1) < exact P_C(w) "
                "(sampling noise): use more orderings (erasure_first_failure(samples=...))"
            )
    parts.append(
        f'next: analysis_purification(source="{d["source"]}"); '
        f'analysis_probe(source="{d["source"]}", probe="code_dimension", where="code")'
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


def survival_at(km: dict[str, np.ndarray], layer: int, *, horizon: int, censored: int):
    """S(layer) = P(tau > layer) from a Kaplan-Meier curve, or None where it is unknown.

    Beyond the horizon (the last layer at which purification was checked) the survival is
    unknown while censored trajectories remain: S(horizon) is only an upper bound. With no
    censored trajectory, S past the last event is exactly 0. Between checkpoints the value
    is that of the previous checkpoint, an upper bound unless every layer is checked.
    """
    if layer > horizon and censored > 0:
        return None
    i = np.searchsorted(km["times"], layer, side="right") - 1
    return float(km["survival"][i]) if i >= 0 else 1.0


def _purification(ctx, a):
    res, _, path = load_result(ctx, a["source"])
    km = km_curve(res)
    tau = np.asarray(res.purification_layer)
    purified = tau[tau >= 0]
    checks = {int(x) for x in res.checkpoint_layers}
    horizon = int(max(checks))
    censored = int((tau < 0).sum())
    out: dict[str, Any] = {
        "source": ctx.show(path),
        "trajectories": int(tau.size),
        "censored_fraction": float(np.mean(tau < 0)),
        "purified": int(purified.size),
        "tau_median_all": tau_median_all(km),
        "horizon": horizon,
        "survival_at_horizon": survival_at(km, horizon, horizon=horizon, censored=censored),
    }
    if purified.size:
        out["tau_mean_purified"] = _mean_err(purified)
        out["tau_quantiles_purified"] = {
            f"q{round(q * 100)}": float(np.quantile(purified, q)) for q in a["quantiles"]
        }
    surv, bounds = {}, []
    for layer in a.get("at_layers") or []:
        layer = int(layer)
        surv[layer] = survival_at(km, layer, horizon=horizon, censored=censored)
        if surv[layer] is not None and layer < horizon and layer not in checks:
            bounds.append(layer)
    out["survival_at"] = surv
    out["survival_upper_bound_layers"] = bounds
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
        if s is None:
            parts.append(
                f"survival at layer {layer}: unknown beyond the horizon (last checked layer "
                f"{d['horizon']}; S({d['horizon']}) = {fmt(d['survival_at_horizon'])} is an "
                "upper bound)"
            )
        elif layer in d["survival_upper_bound_layers"]:
            parts.append(
                f"survival at layer {layer}: {fmt(s)} (upper bound: layer {layer} is between "
                "checkpoints, this is S at the previous checkpoint)"
            )
        else:
            parts.append(f"survival at layer {layer}: {fmt(s)}")
    if len(d["km_times"]):
        n_t = len(d["km_times"])
        idx = np.unique(np.linspace(0, n_t - 1, min(9, n_t)).round().astype(int))
        parts.append(
            "survival curve (layer: fraction unpurified): "
            + ", ".join(f"{fmt(d['km_times'][i])}: {fmt(d['km_survival'][i], 3)}" for i in idx)
        )
    parts.append(
        f'next: analysis_erasure(source="{d["source"]}"); '
        f'analysis_compare(source="{d["source"]}", metrics=["mean(code_dimension)"])'
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
            "survival at chosen layers (unknown beyond the horizon when trajectories are "
            "censored) and the survival curve. Never averages censored times. "
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


class _Unknown(ValueError):
    """A metric has no value for this run; the message says why (shown as a note)."""


def _scalar(res, name: str, layer: int) -> np.ndarray:
    if name not in res.values or np.ndim(res.values[name]) != 2:
        raise ValueError(f"{name!r} is not a scalar probe of this run")
    c = at_layer(res, name, layer)
    if c is None:
        raise ValueError(f"{name} was not recorded at layer {layer}")
    v = np.asarray(res.values[name], dtype=float)[:, c]
    return v[~np.isnan(v)]


def _mean(v: np.ndarray, layer: int, name: str) -> tuple[float, float]:
    if not v.size:
        raise _Unknown(f"no trajectory has {name} at layer {layer}")
    return _mean_err(v)


def _m_mean(res, layer, probe):
    return _mean(_scalar(res, probe, layer), layer, probe)


def _m_stderr(res, layer, probe):
    return _mean(_scalar(res, probe, layer), layer, probe)[1], None


def _m_fraction(res, layer, probe, above=0):
    v = _scalar(res, probe, layer) > float(above)
    return _mean(v.astype(float), layer, probe)


def _erasure_metrics(res, layer, *, rates=(0.1,), delta=0.01) -> dict[str, Any]:
    ff = probe_by_kind(res, "erasure_first_failure")
    if ff is not None and at_layer(res, ff, layer) is None:
        raise ValueError(f"erasure_first_failure was not recorded at layer {layer}")
    return erasure_at(res, layer, rates=list(rates), delta=float(delta))


def _needs(r: dict[str, Any], key: str):
    if key not in r:
        raise _Unknown(r.get("note", f"{key} is not available"))
    return r[key]


def _m_phi(res, layer, e, where="all"):
    if where not in ("all", "code"):
        raise ValueError(f"where must be all or code, got {where!r}")
    r = _erasure_metrics(res, layer, rates=[float(e)])
    m, err = _needs(r, "phi_code" if where == "code" else "phi_all")[0]
    return m, err


def _m_e_delta(res, layer, delta=0.01):
    return _needs(_erasure_metrics(res, layer, delta=delta), "e_delta"), None


def _m_mu(res, layer):
    return tuple(_needs(_erasure_metrics(res, layer), "mu_D"))


def _m_q_plus(res, layer):
    q = _erasure_metrics(res, layer)["q_plus"]
    if q is None:
        raise _Unknown(f"no trajectory selected at layer {layer}")
    return q, None


def _m_tau_median(res, layer):
    tau = tau_median_all(km_curve(res))
    if tau is None:
        raise _Unknown("fewer than half the trajectories purified, so the median is undefined")
    return tau, None


def _m_censored(res, layer):
    tau = res.purification_layer
    if tau is None:
        raise ValueError("this run has no reference, so no purification times")
    return float(np.mean(np.asarray(tau) < 0)), None


def _m_survival(res, layer, at):
    km = km_curve(res)
    tau = np.asarray(res.purification_layer)
    horizon = int(max(int(x) for x in res.checkpoint_layers))
    s = survival_at(km, int(at), horizon=horizon, censored=int((tau < 0).sum()))
    if s is None:
        raise _Unknown(f"unknown beyond horizon {horizon}")
    return s, None


METRICS: dict[str, tuple[tuple[str, ...], str, Callable]] = {
    "mean": (("probe",), "mean of a scalar probe at the depth (+- stderr)", _m_mean),
    "stderr": (("probe",), "standard error of a scalar probe's mean", _m_stderr),
    "fraction": (("probe", "above"), "fraction of trajectories with probe > above", _m_fraction),
    "phi": (("e", "where"), "erasure failure probability Phi(e); where=all or code (k>0)", _m_phi),
    "e_delta": (
        ("delta",),
        "tolerable erasure rate: sup{e : Phi(e) <= delta}, codes with k>0",
        _m_e_delta,
    ),
    "mu_D": ((), "mean first-failure size of codes with k>0", _m_mu),
    "q_plus": ((), "fraction of trajectories that hold a code (k>0)", _m_q_plus),
    "tau_median": (
        (),
        "median purification time over all trajectories (Kaplan-Meier)",
        _m_tau_median,
    ),
    "censored_fraction": ((), "fraction never purified within the horizon", _m_censored),
    "survival": (
        ("at",),
        "fraction not purified after layer `at` (Kaplan-Meier); unknown beyond the horizon "
        "when trajectories are censored",
        _m_survival,
    ),
}


_REQUIRED = {"mean": "probe", "stderr": "probe", "fraction": "probe", "phi": "e", "survival": "at"}
_NUMBERS: dict[str, dict[str, type]] = {
    "fraction": {"above": float},
    "phi": {"e": float},
    "e_delta": {"delta": float},
    "survival": {"at": int},
}


def _signature(name: str) -> str:
    return f"{name}({', '.join(METRICS[name][0])})"


def bound_args(call, index: int | None = None) -> dict[str, Any]:
    """Validated, converted keyword arguments of a metric call, or a teaching ValueError."""
    where = f"metrics[{index}] " if index is not None else ""
    names = METRICS[call.name][0]
    try:
        args = plain(bind(call, names, what="metrics"))
    except ValueError as exc:
        raise ValueError(f"{where}{exc}; usage: {_signature(call.name)}") from None
    need = _REQUIRED.get(call.name)
    if need and need not in args:
        raise ValueError(
            f'{where}"{call.text}": {call.name} needs {need}; usage: {_signature(call.name)}'
        )
    for key, kind in _NUMBERS.get(call.name, {}).items():
        if key in args:
            try:
                value = float(args[key])
                if kind is int and value != int(value):
                    raise ValueError
                args[key] = kind(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f'{where}"{call.text}": {key} must be a number; usage: {_signature(call.name)}'
                ) from None
    if call.name == "phi" and args.get("where", "all") not in ("all", "code"):
        raise ValueError(
            f'{where}"{call.text}": where must be all or code, got {args["where"]!r}; usage: '
            f"{_signature('phi')}"
        )
    return args


def metric_value(res, call, layer: int, notes: list[str] | None = None):
    """(value, error) of one metric call; (None, None) with a note where it is undefined."""
    fn = METRICS[call.name][2]
    try:
        return fn(res, layer, **bound_args(call))
    except _Unknown as exc:
        if notes is not None:
            notes.append(f"{call.text}: {exc}")
        return None, None


def _sweep_points(ctx, root: Path) -> tuple[list[Path], list[str], int]:
    try:
        meta = json.loads((root / "sweep.json").read_text(encoding="utf-8"))
        points = [(p["dir"], p.get("params")) for p in meta["points"]]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            f"bad sweep.json in {ctx.show(root)}: {type(exc).__name__} {exc}; "
            "it should hold a 'points' list of {params, dir}"
        ) from None
    paths, skipped = [], []
    for name, params in points:
        d = root / name
        label = f"{name} ({json.dumps(params, sort_keys=True)})" if params else str(name)
        if (d / "result.npz").exists():
            paths.append(d)
        elif (d / "chunks").exists():
            skipped.append(f"{label}: incomplete (chunks only; rerun the sweep to finish it)")
        else:
            skipped.append(f"{label}: not run")
    return paths, skipped, len(points)


def sources_and_skipped(ctx, a) -> tuple[list[Path], list[str], int]:
    """Runs to compare, the sweep points left out (with the reason) and the sweep's size."""
    if ("sources" in a) == ("source" in a):
        raise ValueError("give sources (a list of runs) or source (a sweep folder), not both")
    if "source" in a:
        root = ctx.path(a["source"], what="source", must_exist=True)
        if (root / "sweep.json").exists():
            paths, skipped, total = _sweep_points(ctx, root)
            if not paths:
                raise ValueError(
                    f"none of the {total} points of sweep {ctx.show(root)} has results yet: "
                    "run ensemble_sweep again with the same arguments to finish it "
                    f"({'; '.join(skipped[:3])})"
                )
            return paths, skipped, total
        return [root], [], 1
    paths = [ctx.path(s, what="sources", must_exist=True) for s in a["sources"]]
    return paths, [], len(paths)


def sources_of(ctx, a) -> list[Path]:
    return sources_and_skipped(ctx, a)[0]


def _probe_names(probes: Any) -> str:
    names = []
    for p in probes if isinstance(probes, list) else []:
        names.append(str(p.get("name") or p.get("kind")) if isinstance(p, dict) else str(p))
    return "+".join(names) or "none"


def _flat(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict):
        out: dict[str, Any] = {}
        for k, v in d.items():
            if k == "probes" and not prefix:
                out["probes"] = json.dumps(_probe_names(v))
                continue
            out.update(_flat(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): json.dumps(d, sort_keys=True)}


def differing(specs: list[dict]) -> list[str]:
    flats = [_flat(s or {}) for s in specs]
    keys = sorted(set().union(*flats))
    return [k for k in keys if len({f.get(k) for f in flats}) > 1]


def _compare(ctx, a):
    calls = parse_calls(a["metrics"], what="metrics")
    for i, call in enumerate(calls):
        if call.name not in METRICS:
            close = difflib.get_close_matches(call.name, list(METRICS), n=2)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise ValueError(
                f"metrics[{i}]: unknown metric {call.name!r}{hint} (help_list(what=metrics))"
            )
    for i, call in enumerate(calls):
        bound_args(call, i)
    paths, skipped, total = sources_and_skipped(ctx, a)
    if not paths:
        raise ValueError("no finished runs to compare")
    loaded = [load_result(ctx, str(p)) for p in paths]
    specs = [(meta.get("spec") if meta else None) or {} for _, meta, _ in loaded]
    diff = differing(specs)
    rows = []
    for (res, _, path), spec in zip(loaded, specs, strict=True):
        layer = int(a["depth"]) if a.get("depth") is not None else int(res.checkpoint_layers[-1])
        flat = _flat(spec)
        row: dict[str, Any] = {
            "run": ctx.show(path),
            "params": {k: json.loads(flat[k]) if k in flat else None for k in diff},
            "layer": layer,
            "values": [],
            "notes": [],
        }
        for call in calls:
            try:
                row["values"].append(metric_value(res, call, layer, row["notes"]))
            except ValueError as exc:
                row["values"].append((None, None))
                row["notes"].append(f"{call.text}: {exc}")
        rows.append(row)
    notes = []
    layers = sorted({r["layer"] for r in rows})
    if a.get("depth") is None and len(layers) > 1:
        notes.append(
            "rows read at their own last checkpoint: layers differ "
            f"({', '.join(map(str, layers))}); pass depth=... to align"
        )
    return {
        "metrics": [c.text for c in calls],
        "differing": diff,
        "rows": rows,
        "skipped": skipped,
        "points": total,
        "notes": notes,
    }


def _render_compare(d, ctx):
    cols = (d["differing"] or ["run"]) + ["layer"] + d["metrics"]
    rows = []
    for r in d["rows"]:
        first = [r["params"][k] for k in d["differing"]] if d["differing"] else [r["run"]]
        rows.append([*[fmt(v) for v in first], r["layer"], *[pm(m, e) for m, e in r["values"]]])
    notes, seen = [*d["notes"], *(f"skipped {s}" for s in d["skipped"])], set()
    for r in d["rows"]:
        label = r["run"] if len(d["rows"]) > 1 else None
        for n in r["notes"]:
            text = f"{label}: {n}" if label else n
            if text not in seen:
                seen.add(text)
                notes.append(text)
    shown = notes[:8]
    if len(notes) > len(shown):
        shown.append(f"... and {len(notes) - len(shown)} more notes")
    head = (
        f"analysis_compare: {len(d['rows'])} of {d['points']} sweep points"
        if d["skipped"]
        else f"analysis_compare: {len(d['rows'])} runs"
    )
    return lines(head, table(cols, rows), *[f"note ('-' = no value): {n}" for n in shown])


register(
    ToolDef(
        name="analysis_compare",
        stack="analysis",
        summary="Compare runs or a sweep: a metrics table.",
        description=(
            "One row per run (a list of runs, or the points of an ensemble_sweep folder), with "
            "the parameters that differ between runs as columns, then metrics as call strings: "
            "mean(probe), stderr(probe), fraction(probe, above=0), phi(e=0.05, where=code), "
            "e_delta(0.01), mu_D, q_plus, tau_median, censored_fraction, survival(at=100) "
            "(unknown beyond the horizon when trajectories are censored; help_list(what=metrics)). "
            'Example: {"source": "runs/ensemble_sweep-1a2b", '
            '"metrics": ["mean(code_dimension)", "phi(e=0.05)"]}'
        ),
        params=(
            Param("sources", "array", "Runs to compare.", items="string"),
            Param("source", "string", "A sweep folder (or one run)."),
            Param("metrics", "array", "Metric calls.", required=True, items="string"),
            Param(
                "depth",
                "integer",
                "Layer to evaluate at (default each run's last). Not used by tau_median, "
                "censored_fraction and survival, which are whole-run quantities.",
                minimum=0,
            ),
        ),
        handler=_compare,
        render=_render_compare,
        long_running=True,
    )
)
