"""ensemble_run (and, in Task 19, ensemble_sweep): monitored random circuits, flat arguments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2 import api
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
        return {"summary": api.load_results(str(out)), "output_dir": ctx.show(out), "reused": True}
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
    head = (
        f"{title}: n={s['num_system']}, layers={s['layers']}, p={fmt(s['measurement_rate'])}, "
        f"{s['num_trajectories']} trajectories, seed={s['seed']}, "
        f"{fmt(s['elapsed_seconds'], 3)} s"
    )
    layers = s["checkpoints"]["layers"]
    scalar = {k: v for k, v in s["probes"].items() if not v.get("vector") and "layers" not in v}
    own = {k: v for k, v in s["probes"].items() if not v.get("vector") and "layers" in v}
    vector = {k: v for k, v in s["probes"].items() if v.get("vector")}
    parts = [head, *extra]
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
        kinds = " ".join(s["probes"])
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
    extra = ("reused the saved run of this exact request",) if d.get("reused") else ()
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
