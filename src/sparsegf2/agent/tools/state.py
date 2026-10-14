"""Stabilizer states held by the server, acted on with circuit text and queried with observables."""

from __future__ import annotations

from typing import Any

import numpy as np

from sparsegf2 import api
from sparsegf2.agent.params import SEED, observable_dicts
from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.render import fmt, kv, lines, table
from sparsegf2.circuit import Circuit
from sparsegf2.simulator import Simulator

STATE_ID = Param("state_id", "string", "An open state's id (state_new returns it).", required=True)


def _new(ctx, a):
    if ("num_qubits" in a) == ("source" in a):
        raise ValueError("give exactly one of num_qubits or source (a .state.npz file)")
    if "source" in a:
        path = ctx.path(a["source"], what="source", must_exist=True)
        sim = Simulator.load(path)
        origin = f"loaded {ctx.show(path)}"
    else:
        sim = Simulator(a["num_qubits"], phases=a["phases"], seed=a.get("seed"))
        origin = "new"
    state_id = ctx.states.add(sim, origin)
    return {**ctx.states.info(state_id), "seed": str(sim.seed.entropy)}


def _render_new(d, ctx):
    return (
        f"state_new: {d['state_id']}, {d['num_qubits']} qubits, phases "
        f"{str(d['phases']).lower()} ({d['origin']})\n"
        f'next: state_apply(state_id="{d["state_id"]}", circuit="H 0\\nCX 0 1")'
    )


def _apply(ctx, a):
    sim = ctx.states.get(a["state_id"])
    result = sim.run(Circuit(a["circuit"]))
    m = int(np.asarray(result.random).size)
    ctx.states.count_measurements(a["state_id"], m)
    out: dict[str, Any] = {
        "state_id": a["state_id"],
        "measurements": m,
        "random": np.asarray(result.random).astype(int).tolist(),
    }
    if sim.phases:
        out["outcomes"] = np.asarray(result.measurements).astype(int).tolist()
        if np.asarray(result.detectors).size:
            out["detectors"] = np.asarray(result.detectors).astype(int).tolist()
    return out


def _render_apply(d, ctx):
    head = f"state_apply: {d['state_id']}, {d['measurements']} measurement" + (
        "s" if d["measurements"] != 1 else ""
    )
    return lines(head, kv(d, ["outcomes", "random", "detectors"], max_list=64))


def _destabilizers(sim) -> list[str]:
    if not sim.phases:
        raise ValueError("destabilizers exist only with phases=true")
    if sim.num_qubits > api.MAX_STABILIZER_QUBITS:
        raise ValueError(f"destabilizers are listed only up to {api.MAX_STABILIZER_QUBITS} qubits")
    return [str(p) for p in sim.destabilizers()]


def _query(ctx, a):
    sim = ctx.states.get(a["state_id"])
    items = list(a["observables"])
    extra = [i for i, s in enumerate(items) if s.strip() == "destabilizers"]
    rest = [s for i, s in enumerate(items) if i not in extra]
    compiled = api._compile_observables(observable_dicts(rest), sim.num_qubits, sim.phases, 0)
    values: list[tuple[str, Any]] = []
    it = iter(compiled)
    for i, text in enumerate(items):
        if i in extra:
            values.append(("destabilizers", _destabilizers(sim)))
            continue
        _name, fn, show = next(it)
        values.append((text.strip(), show(fn(sim))))
    return {"state_id": a["state_id"], "values": [{"observable": k, "value": v} for k, v in values]}


def _show_value(x):
    return x if isinstance(x, (list, dict)) else fmt(x)


def _render_query(d, ctx):
    out = [f"state_query: {d['state_id']}"]
    for item in d["values"]:
        v = item["value"]
        if isinstance(v, list) and v and isinstance(v[0], str):
            out.append(f"{item['observable']}: " + " ".join(v))
        elif isinstance(v, dict):
            body = ", ".join(f"{k}={_show_value(x)}" for k, x in v.items())
            out.append(f"{item['observable']}: {body}")
        else:
            out.append(f"{item['observable']}: {_show_value(v)}")
    return "\n".join(out)


def _save(ctx, a):
    path = ctx.path(a["path"], what="path")
    if not path.name.endswith(".state.npz"):
        raise ValueError("state files end in .state.npz (e.g. bell.state.npz)")
    path.parent.mkdir(parents=True, exist_ok=True)
    ctx.states.get(a["state_id"]).save(path)
    return {"state_id": a["state_id"], "path": ctx.show(path)}


def _close(ctx, a):
    return ctx.states.close(a["state_id"])


def _list(ctx, a):
    return {"states": ctx.states.list()}


def _render_list(d, ctx):
    if not d["states"]:
        return "state_list: no open states"
    rows = [
        [s["state_id"], s["num_qubits"], s["phases"], s["measurements"], s["origin"]]
        for s in d["states"]
    ]
    return "state_list:\n" + table(["state_id", "qubits", "phases", "measurements", "origin"], rows)


register(
    ToolDef(
        name="state_new",
        stack="state",
        summary="Open a stabilizer state (|0..0> or from a file).",
        description=(
            "Open a stabilizer state held by the server: num_qubits qubits in |0...0>, or a state "
            "saved with state_save or by traj_run(store=state) (source). Returns a state_id for "
            'state_apply/state_query. Example: {"num_qubits": 4, "seed": 1}'
        ),
        params=(
            Param(
                "num_qubits",
                "integer",
                "Qubits of a new |0...0> state.",
                minimum=1,
                maximum=api.MAX_QUBITS,
            ),
            Param("source", "string", "A .state.npz file under the root, instead."),
            Param("phases", "boolean", "Track signs (outcomes, expectations).", default=True),
            SEED,
        ),
        handler=_new,
        render=_render_new,
        read_only=False,
        examples=({"num_qubits": 4, "seed": 1},),
    )
)

register(
    ToolDef(
        name="state_apply",
        stack="state",
        summary="Apply circuit text (gates, measurements, ...).",
        description=(
            "Apply Stim-language circuit text to an open state: gates, measurements, resets, MPP, "
            "noise, REPEAT. Returns the new measurement outcomes (with phases) and which were "
            'random. Long circuits (large REPEAT counts) should run with background="always". '
            'Example: {"state_id": "s1", "circuit": "H 0\\nCX 0 1\\nM 0"}'
        ),
        params=(STATE_ID, Param("circuit", "string", "Circuit text to apply.", required=True)),
        handler=_apply,
        render=_render_apply,
        read_only=False,
        long_running=True,
    )
)

register(
    ToolDef(
        name="state_query",
        stack="state",
        summary="Observables of an open state.",
        description=(
            "Evaluate observables of an open state, as call strings: entropy(0:4), "
            "mutual_information(a=0:2, b=2:4), code_dimension(reference=4:8), "
            "expectation(Z0*Z1), stabilizers, destabilizers, ... (help_list(what=observables)). "
            'Example: {"state_id": "s1", "observables": ["entropy(0:1)", "stabilizers"]}'
        ),
        params=(
            STATE_ID,
            Param("observables", "array", "Observable calls.", required=True, items="string"),
        ),
        handler=_query,
        render=_render_query,
        long_running=True,
    )
)

register(
    ToolDef(
        name="state_save",
        stack="state",
        summary="Save an open state to a .state.npz file.",
        description=(
            "Save an open state exactly (tableau, signs, random-stream position) to a .state.npz "
            "file under the root; state_new(source=...) reopens it. "
            'Example: {"state_id": "s1", "path": "states/bell.state.npz"}'
        ),
        params=(STATE_ID, Param("path", "string", "Destination (.state.npz).", required=True)),
        handler=_save,
        render=lambda d, ctx: f"state_save: {d['state_id']} -> {d['path']}",
        read_only=False,
    )
)

register(
    ToolDef(
        name="state_close",
        stack="state",
        summary="Close an open state.",
        description='Close an open state and free its memory. Example: {"state_id": "s1"}',
        params=(STATE_ID,),
        handler=_close,
        render=lambda d, ctx: f"state_close: closed {d['state_id']}",
        read_only=False,
    )
)

register(
    ToolDef(
        name="state_list",
        stack="state",
        summary="Open states of this server.",
        description="List the open states (id, qubits, phases, measurements so far). Example: {}",
        params=(),
        handler=_list,
        render=_render_list,
        examples=({},),
    )
)
