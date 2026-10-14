"""ensemble_run (and, in Task 19, ensemble_sweep): monitored random circuits, flat arguments."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2 import api
from sparsegf2.agent import sessions
from sparsegf2.agent.grammar import parse_grid
from sparsegf2.agent.params import (
    CHECKPOINTS,
    GRAPH_PARAMS,
    OUTPUT_DIR,
    PROBES,
    PROTOCOL_PARAMS,
    SEED,
    ensemble_spec,
)
from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.render import curve, fmt, lines, pm, table
from sparsegf2.agent.sessions import from_api_status

TRAJECTORIES = Param(
    "trajectories",
    "integer",
    "Number of trajectories; default for a random graph family: graph_count x circuits_per_graph.",
    minimum=1,
)
ENSEMBLE_PARAMS = (
    *PROTOCOL_PARAMS,
    *GRAPH_PARAMS,
    PROBES,
    CHECKPOINTS,
    TRAJECTORIES,
    SEED,
    OUTPUT_DIR,
)
MAX_ROWS = 12


def prepare(ctx, args: dict[str, Any], tool: str):
    spec = ensemble_spec(args, ctx)
    mc = api._build_spec(spec)
    if args.get("trajectories") is None and mc.num_trajectories is None:
        raise ValueError(
            "give trajectories (the number of random circuits); it may be left out only for a "
            "random graph family or a graph file with several graphs, which then run "
            "graph_count x circuits_per_graph circuits"
        )
    return spec, mc, ctx.output_dir(tool, args)


def _stored_spec(out: Path) -> dict | None:
    f = out / "spec.json"
    return json.loads(f.read_text(encoding="utf-8")).get("spec") if f.exists() else None


def _estimate(ctx, args):
    spec, mc, _ = prepare(ctx, args, "ensemble_run")
    return float(api.estimate_ensemble(spec, args.get("trajectories"))["estimated_seconds"] or 0.0)


def _stored_meta(out: Path) -> dict:
    f = out / "spec.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _run(ctx, args):
    spec, mc, out = prepare(ctx, args, "ensemble_run")
    if args.get("estimate_only"):
        return {
            "estimate": api.estimate_ensemble(spec, args.get("trajectories")),
            "output_dir": ctx.show(out),
        }
    if (out / "result.npz").exists() and not (out / "chunks").exists():
        if api._canonical(_stored_spec(out)) != api._canonical(spec):
            raise ValueError(f"{ctx.show(out)} holds a different run; give another output_dir")
        meta = _stored_meta(out)
        want_traj = args.get("trajectories")
        want_seed = args.get("seed")
        if want_traj is None:
            want_traj = mc.num_trajectories
        if (want_traj is not None and meta.get("trajectories") != want_traj) or (
            want_seed is not None and meta.get("seed") != want_seed
        ):
            raise ValueError(
                f"{ctx.show(out)} holds a different run (seed {meta.get('seed')} / "
                f"{meta.get('trajectories')} trajectories); give another output_dir"
            )
        return {
            "summary": api.load_results(str(out)),
            "output_dir": ctx.show(out),
            "reused": True,
            "seed_from_store": want_seed is None,
            "stored_seed": meta.get("seed"),
        }
    if (out / "chunks").exists():
        raise ValueError(
            f"{ctx.show(out)} holds an unfinished background run of this request; run it again "
            "with background=always to resume it"
        )
    summary = api.run_ensemble(
        spec,
        args.get("trajectories"),
        seed=args.get("seed"),
        output_dir=str(out),
        command="mcp:ensemble_run",
    )
    return {"summary": summary, "output_dir": ctx.show(out)}


def start_or_join(ctx, spec: dict, args: dict, out: Path, command: str) -> dict[str, Any]:
    """Start (or resume) a chunked ensemble job, or return the job already writing to ``out``."""
    target = out.resolve()
    with api._JOBS_LOCK:
        running = [
            j
            for j in api._JOBS.values()
            if j.output_dir == target and j.status in ("queued", "running")
        ]
    if running:
        status = from_api_status(api.job_status(running[0].job_id))
        return {**status, "output_dir": ctx.show(out), "note": "already running; this is that job"}
    status = from_api_status(
        api.start_ensemble_job(
            spec,
            args.get("trajectories"),
            output_dir=str(out),
            seed=args.get("seed"),
            command=command,
        )
    )
    return {**status, "output_dir": ctx.show(out)}


def _background(ctx, args):
    spec, mc, out = prepare(ctx, args, "ensemble_run")
    return start_or_join(ctx, spec, args, out, "mcp:ensemble_run")


def _rows(n: int) -> np.ndarray:
    if n <= MAX_ROWS:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, MAX_ROWS).round().astype(int))


def render_summary(
    title: str, s: dict[str, Any], *, output_dir: str | None = None, extra: tuple[str, ...] = ()
) -> str:
    def g(key, digits=None):
        v = s.get(key)
        if v is None:
            return "?"
        return fmt(v, digits) if digits else (fmt(v) if isinstance(v, float) else str(v))

    head = (
        f"{title}: n={g('num_system')}, layers={g('layers')}, p={g('measurement_rate')}, "
        f"{g('num_trajectories')} trajectories, seed={g('seed')}, {g('elapsed_seconds', 3)} s"
    )
    complete = "probes" in s and "checkpoints" in s
    probes = s.get("probes") or {}
    layers = (s.get("checkpoints") or {}).get("layers", [])
    scalar = {k: v for k, v in probes.items() if not v.get("vector") and "layers" not in v}
    own = {k: v for k, v in probes.items() if not v.get("vector") and "layers" in v}
    vector = {k: v for k, v in probes.items() if v.get("vector")}
    parts = [head, *extra]
    if not complete:
        parts.append(
            "summary truncated by the API; see the saved files (result_get, analysis_probe)"
        )
    if scalar:
        idx = _rows(len(layers))
        rows = [
            [layers[i], *[pm(v["mean"][i], v["stderr"][i]) for v in scalar.values()]] for i in idx
        ]
        parts.append(table(["layer", *scalar], rows))
    for name, v in own.items():
        idx = _rows(len(v["layers"]))
        parts.append(
            f"{name}:\n"
            + table(
                ["layer", name], [[v["layers"][i], pm(v["mean"][i], v["stderr"][i])] for i in idx]
            )
        )
    for name, v in vector.items():
        if "mean" not in v:
            parts.append(f"{name}: no data")
            continue
        start = int(v["index"][0]) if "index" in v else 0
        parts.append(
            f"{name} at layer {v['at_layer']} ({v['what']}):\n  " + curve(v["mean"], start=start)
        )
    pur = s.get("purification")
    if pur:
        line = (
            f"purification: {pur['purified']} purified, {pur['censored']} censored "
            f"(horizon {pur['horizon_layer']})"
        )
        if pur.get("tau_median_purified") is not None:
            line += f", tau median among purified {fmt(pur['tau_median_purified'])}"
        if pur.get("tau_median_all") is not None:
            line += f", tau median overall {pur['tau_median_all']}"
        parts.append(line)
    for note in (s.get("notes") or [])[:2]:
        parts.append(f"note: {note}")
    if output_dir:
        parts.append(f"saved: {output_dir}")
        nxt = [f'analysis_summary(source="{output_dir}")']
        kinds = " ".join(probes)
        if "erasure_first_failure" in kinds:
            nxt.append(f'analysis_erasure(source="{output_dir}")')
        if pur:
            nxt.append(f'analysis_purification(source="{output_dir}")')
        parts.append("next: " + "; ".join(nxt))
    return lines(*parts)


def _render(d, ctx):
    if "estimate" in d:
        e = d["estimate"]
        return lines(
            f"ensemble_run: estimate only (nothing was run) for {e['trajectories']} trajectories",
            f"estimated_seconds: {fmt(e['estimated_seconds'])} on {e['threads']} threads "
            f"(pessimistic); result_megabytes: {fmt(e['result_megabytes'])}",
            f"output_dir would be: {d['output_dir']}",
        )
    extra = ()
    if d.get("reused"):
        extra = ("reused the saved run of this exact request",)
        if d.get("seed_from_store"):
            extra = (
                f"reused the saved run of this exact request, same seed {d.get('stored_seed')}; "
                "pass a different seed or output_dir for fresh samples",
            )
    return render_summary("ensemble_run", d["summary"], output_dir=d["output_dir"], extra=extra)


register(
    ToolDef(
        name="ensemble_run",
        stack="ensemble",
        summary="Run an ensemble of monitored random circuits.",
        description=(
            "Run many trajectories of a monitored random Clifford circuit on a graph (or a "
            "collection of graphs): each layer applies random two-qubit Cliffords on graph edges, "
            "then measures each system qubit with probability measurement_rate. probes are call "
            "strings recorded at checkpoints (help_list(what=probes)). Returns means +- errors per "
            "checkpoint and purification statistics; everything is saved to output_dir. Long runs "
            'become background jobs. Example: {"num_system": 64, "layers": 256, '
            '"measurement_rate": 0.1, "graph": "watts_strogatz", "graph_count": 50, '
            '"circuits_per_graph": 4, "schedule": "random_edges", "reference": '
            '"purification", "probes": ["code_dimension", "erasure_first_failure(samples=512)"], '
            '"seed": 1}'
        ),
        params=(
            *ENSEMBLE_PARAMS,
            Param(
                "estimate_only",
                "boolean",
                "Only estimate time and memory; run nothing.",
                default=False,
            ),
        ),
        handler=_run,
        render=_render,
        read_only=False,
        long_running=True,
        estimate=_estimate,
        background=_background,
        examples=(
            {
                "num_system": 8,
                "layers": 8,
                "measurement_rate": 0.1,
                "graph": "watts_strogatz",
                "graph_count": 2,
                "reference": "purification",
                "probes": ["code_dimension"],
                "seed": 1,
            },
        ),
    )
)


SWEEPABLE = frozenset(
    {
        "num_system",
        "layers",
        "measurement_rate",
        "gates_per_layer",
        "circuits_per_graph",
        "scramble_gates",
        "scramble_matchings",
        "scramble_layers",
        "graph_k",
        "graph_beta",
        "graph_p",
        "graph_d",
        "graph_count",
        "trajectories",
    }
)
_REQUIRED = ("num_system", "layers", "measurement_rate")
SWEEP_PARAMS = tuple(
    dataclasses.replace(p, required=False) if p.name in _REQUIRED else p for p in ENSEMBLE_PARAMS
) + (
    Param(
        "grid",
        "string",
        "Parameters to sweep, every combination: e.g. measurement_rate=[0.05,0.1,0.15]; "
        "num_system=[64,128].",
        required=True,
    ),
)


def _point_name(point: dict[str, Any]) -> str:
    def text(v):
        return repr(float(v)) if isinstance(v, float) else str(v)

    return ",".join(f"{k}={text(v)}" for k, v in point.items())


def _grid_points(args: dict[str, Any]) -> list[dict[str, Any]]:
    grid = parse_grid(args["grid"])
    by_name = {p.name: p for p in ENSEMBLE_PARAMS}
    for key, values in grid.items():
        if key not in SWEEPABLE:
            raise ValueError(f"grid: {key} cannot be swept; sweepable: {sorted(SWEEPABLE)}")
        if key in args:
            raise ValueError(f"{key} is both in grid and as an argument; give it in one place")
        grid[key] = [by_name[key].check(v, "grid") for v in values]
    missing = [k for k in _REQUIRED if k not in args and k not in grid]
    if missing:
        raise ValueError(f"give {', '.join(missing)} as arguments or in grid")
    keys = list(grid)
    for key, values in grid.items():
        names = [_point_name({key: v}) for v in values]
        dup = sorted({n.split("=", 1)[1] for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"grid: duplicate value(s) for {key}: {', '.join(dup)}")
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]


def _sweep_estimate(ctx, args):
    _grid_points(args)  # a bad grid is an error now, not a failed background job
    return math.inf


def _sweep(ctx, args):
    points = _grid_points(args)
    root = ctx.output_dir("ensemble_sweep", args)
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / "sweep.json"
    skip = ("grid", "output_dir", "seed", "background")
    base = json.loads(json.dumps({k: v for k, v in args.items() if k not in skip}, default=str))
    seed = args["seed"] if args.get("seed") is not None else None
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        differs = []
        if seed is not None and seed != old["seed"]:
            differs.append(f"seed {old['seed']}")
        if old["grid"] != args["grid"]:
            differs.append(f"grid {old['grid']!r}")
        if old["base"] != base:
            differs.append("other parameters")
        if differs:
            raise ValueError(
                f"{ctx.show(root)} holds a different sweep ({', '.join(differs)}); "
                "give another output_dir"
            )
        seed = old["seed"]
    elif seed is None:
        seed = api._resolve_seed(None)
    meta = {
        "version": 1,
        "grid": args["grid"],
        "seed": seed,
        "base": base,
        "points": [{"params": p, "dir": _point_name(p)} for p in points],
    }
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    done: list[dict[str, Any]] = []
    for i, point in enumerate(points):
        if sessions.cancelled():
            break
        point_args = {**args, **point, "seed": seed}
        out = root / _point_name(point)
        spec, mc, _ = prepare(ctx, {**point_args, "output_dir": str(out)}, "ensemble_sweep")
        status = start_or_join(ctx, spec, point_args, out, "mcp:ensemble_sweep")
        job = api._get_job(status["job_id"])
        while job.thread is not None and job.thread.is_alive():
            job.thread.join(0.5)
            if sessions.cancelled():
                api.cancel_job(job.job_id)
        if job.status == "failed":
            raise RuntimeError(f"point {_point_name(point)} failed: {job.error}")
        done.append({"params": point, "dir": ctx.show(out), "status": job.status})
        sessions.report(i + 1, len(points), "points")
    return {
        "output_dir": ctx.show(root),
        "seed": seed,
        "points": done,
        "total_points": len(points),
        "cancelled": sessions.cancelled(),
    }


def _render_sweep(d, ctx):
    rows = [[p["dir"].rsplit("/", 1)[-1], p["status"]] for p in d["points"]]
    return lines(
        f"ensemble_sweep: {len(d['points'])}/{d['total_points']} points, seed {d['seed']} "
        f"-> {d['output_dir']}" + (" (cancelled)" if d["cancelled"] else ""),
        table(["point", "status"], rows),
        f'next: analysis_compare(source="{d["output_dir"]}", metrics=["mean(code_dimension)"])',
    )


register(
    ToolDef(
        name="ensemble_sweep",
        stack="ensemble",
        summary="A grid of ensembles (e.g. a scan in p).",
        description=(
            "Run ensemble_run at every combination of the grid, e.g. "
            'grid="measurement_rate=[0.05, 0.1, 0.15]; num_system=[64, 128]". All points share '
            "graph seeds and the circuit seed, so lower rates measure a subset of the qubits "
            "higher ones do. A background job by default; each point is saved as it finishes and "
            "rerunning resumes. Compare the points with analysis_compare(source=output_dir). "
            'Example: {"num_system": 32, "layers": 128, "grid": '
            '"measurement_rate=[0.05, 0.1, 0.2]", "reference": "purification", "probes": '
            '["code_dimension"], "trajectories": 200, "seed": 1}'
        ),
        params=SWEEP_PARAMS,
        handler=_sweep,
        render=_render_sweep,
        read_only=False,
        long_running=True,
        estimate=_sweep_estimate,
        examples=(
            {
                "num_system": 8,
                "layers": 4,
                "trajectories": 8,
                "reference": "purification",
                "grid": "measurement_rate=[0.1, 0.2]",
                "probes": ["code_dimension"],
                "background": "never",
                "seed": 1,
            },
        ),
    )
)
