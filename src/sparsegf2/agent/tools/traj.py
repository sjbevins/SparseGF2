"""traj_run and traj_probe: one trajectory of an ensemble, recorded and replayable."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2 import api, mipt
from sparsegf2._version import __version__
from sparsegf2.agent.grammar import parse_checkpoints
from sparsegf2.agent.params import (
    CHECKPOINTS,
    GRAPH_PARAMS,
    OUTPUT_DIR,
    PROBES,
    PROTOCOL_PARAMS,
    SEED,
    SOURCE,
    ensemble_spec,
    probe_dicts,
)
from sparsegf2.agent.registry import MISSING, Param, ToolDef, register
from sparsegf2.agent.render import lines, table
from sparsegf2.codes import StabilizerCode
from sparsegf2.simulator import Simulator

SNAPSHOT = "trajectory_code"
STORE = Param(
    "store",
    "string",
    "code: save the code at each checkpoint (needs a reference); state: save the full "
    "state (reopen with state_new); none.",
    default="code",
    enum=("code", "state", "none"),
)
COMPARE = Param(
    "compare_stim",
    "boolean",
    "Check the trajectory against Stim and the engine's code against the gate-by-gate "
    "simulator at every checkpoint.",
    default=False,
)
SAVE_CIRCUIT = Param(
    "save_circuit", "boolean", "Write the exact circuit as circuit.stim.", default=False
)


def canonical(matrix: np.ndarray) -> np.ndarray:
    """Reduced row echelon form over GF(2): equal for two generator sets of the same group."""
    m = (np.asarray(matrix) % 2).astype(np.uint8)
    r = 0
    for c in range(m.shape[1]):
        if r == m.shape[0]:
            break
        nz = np.flatnonzero(m[r:, c])
        if not nz.size:
            continue
        p = r + int(nz[0])
        m[[r, p]] = m[[p, r]]
        others = np.flatnonzero(m[:, c])
        m[others[others != r]] ^= m[r]
        r += 1
    return m[:r]


def _check_free(ctx, out: Path, spec: dict, seed: int, t: int) -> None:
    """Refuse to write into a directory that holds another trajectory's record."""
    f = out / "trajectory.json"
    if not f.exists():
        return
    old = json.loads(f.read_text(encoding="utf-8"))
    if (
        old.get("seed") != seed
        or old.get("trajectory") != t
        or api._canonical(old.get("spec")) != api._canonical(spec)
    ):
        raise ValueError(
            f"{ctx.show(out)} holds a different trajectory record; give another output_dir"
        )


def _record(
    ctx,
    spec: dict,
    seed: int,
    t: int,
    store: str,
    compare: bool,
    save_circuit: bool,
    out: Path,
) -> dict[str, Any]:
    mc = api._build_spec(spec)
    if store == "code" and mc.reference == "none":
        raise ValueError(
            "store=code needs a code: use reference=purification or single, or store=state"
        )
    probes = list(mc.probes)
    if store == "code":
        if any(p.kind == "code_snapshot" for p in probes):
            raise ValueError("store=code already saves the code; drop code_snapshot from probes")
        probes.append(mipt.code_snapshot(name=SNAPSHOT))
    _check_free(ctx, out, spec, seed, t)
    res = mc.replay(t, seed=seed, probes=probes)
    out.mkdir(parents=True, exist_ok=True)
    res.save(out / "result.npz")
    layers = [int(x) for x in res.checkpoint_layers]
    data: dict[str, Any] = {
        "output_dir": ctx.show(out),
        "trajectory": t,
        "seed": seed,
        "num_system": mc.num_system,
        "reference": mc.reference,
        "layers": layers,
        "files": ["result.npz", "trajectory.json"],
    }
    if store == "code":
        data["k"] = []
        for c in range(len(layers)):
            try:
                data["k"].append(res.code(0, c, SNAPSHOT).num_logical)
            except ValueError:  # stopped: the reference is pure
                data["k"].append(0)
    data["probes"] = {}
    for name, vals in res.values.items():
        arr = np.asarray(vals, dtype=float)
        if arr.ndim == 2:  # a scalar probe: (1 trajectory, evaluations)
            data["probes"][name] = [None if np.isnan(v) else v for v in arr[0].tolist()]
    data["probe_layers"] = {name: [int(x) for x in res.layers(name)] for name in data["probes"]}
    circuit = mc.circuit(t, seed=seed) if (store == "state" or compare or save_circuit) else None
    if store == "state":
        wanted = set(layers)

        def observe(sim, tick):
            if tick in wanted:
                sim.save(out / f"state_L{tick}.state.npz")

        Simulator(mc.num_qubits, phases=False, seed=0).run(circuit, observe=observe)
        data["files"] += [f"state_L{x}.state.npz" for x in layers]
    if compare:
        from sparsegf2 import crosscheck

        data["stim"] = crosscheck.compare_stabilizers(
            circuit, num_qubits=mc.num_qubits, phases=False, ticks=layers
        )
        if store == "code":
            codes: dict[int, np.ndarray] = {}

            def grab(sim, tick):
                if tick in set(layers):
                    codes[tick] = canonical(
                        StabilizerCode.from_simulator(sim, mc.reference_qubits).stabilizer_matrix()
                    )

            Simulator(mc.num_qubits, phases=False, seed=0).run(circuit, observe=grab)
            match = True
            for c, layer in enumerate(layers):
                try:
                    engine = canonical(res.code(0, c, SNAPSHOT).stabilizer_matrix())
                except ValueError:
                    continue
                match &= engine.shape == codes[layer].shape and np.array_equal(engine, codes[layer])
            data["engine_vs_simulator"] = bool(match)
    if save_circuit:
        (out / "circuit.stim").write_text(str(circuit.to_stim()) + "\n", encoding="utf-8")
        data["files"].append("circuit.stim")
    meta = {
        "version": 1,
        "spec": spec,
        "seed": seed,
        "trajectory": t,
        "checkpoints": layers,
        "store": store,
        "package_version": __version__,
        "randomness_contract": mipt.RANDOMNESS_CONTRACT,
    }
    (out / "trajectory.json").write_text(
        json.dumps(api._jsonable(meta, exact=True), indent=1), encoding="utf-8"
    )
    return data


def _run(ctx, a):
    spec = ensemble_spec(a, ctx)
    seed = api._resolve_seed(a.get("seed"))
    out = ctx.output_dir("traj_run", {**a, "seed": seed})
    return _record(
        ctx, spec, seed, a["trajectory"], a["store"], a["compare_stim"], a["save_circuit"], out
    )


def _probe(ctx, a):
    src = ctx.path(a["source"], what="source", must_exist=True)
    meta_path = src / "trajectory.json"
    if not meta_path.exists():
        raise ValueError(
            f"{ctx.show(src)} is not a trajectory record (no trajectory.json); traj_run makes one"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("randomness_contract") != mipt.RANDOMNESS_CONTRACT:
        raise ValueError(
            f"this record was made under randomness contract {meta.get('randomness_contract')} "
            f"(package {meta.get('package_version')}); this version uses contract "
            f"{mipt.RANDOMNESS_CONTRACT}, so replaying would give a different trajectory. Run "
            "traj_run again with this version."
        )
    spec = dict(meta["spec"])
    spec["probes"] = probe_dicts(a.get("probes") or [])
    if a.get("checkpoints") is not None:
        spec["checkpoints"] = parse_checkpoints(a["checkpoints"])
    out = (
        ctx.path(a["output_dir"], what="output_dir")
        if a.get("output_dir")
        else ctx.output_dir("traj_probe", {**a, "source": ctx.show(src)})
    )
    return _record(
        ctx, spec, meta["seed"], meta["trajectory"], a["store"], a["compare_stim"], False, out
    )


def _render(d, ctx, title):
    head = (
        f"{title}: trajectory {d['trajectory']} (seed {d['seed']}), n={d['num_system']}, "
        f"reference={d['reference']} -> {d['output_dir']}"
    )
    cols = ["layer"] + (["k"] if "k" in d else [])
    rows = [[layer] + ([d["k"][i]] if "k" in d else []) for i, layer in enumerate(d["layers"])]
    for name, values in d["probes"].items():
        if d["probe_layers"][name] == d["layers"] and name != SNAPSHOT:
            cols.append(name)
            for i, v in enumerate(values):
                rows[i].append(v)
    parts = [head, table(cols, rows)]
    if "stim" in d:
        s = d["stim"]
        verdict = "match" if s["match"] else "MISMATCH at tick " + str(s["first_mismatch"]["tick"])
        parts.append(f"stim: {verdict} ({s['ticks_compared']} checkpoints)")
    if "engine_vs_simulator" in d:
        parts.append(f"engine vs simulator: {'match' if d['engine_vs_simulator'] else 'MISMATCH'}")
    parts.append("files: " + ", ".join(d["files"]))
    parts.append(
        f'next: code_analyze(source="{d["output_dir"]}", properties=[...]); '
        f'traj_probe(source="{d["output_dir"]}", probes=[...], checkpoints=...)'
    )
    return lines(*parts)


TRAJ_PARAMS = (
    *PROTOCOL_PARAMS,
    *GRAPH_PARAMS,
    PROBES,
    CHECKPOINTS,
    SEED,
    OUTPUT_DIR,
    Param(
        "trajectory",
        "integer",
        "Which trajectory of the ensemble with this seed (the same numbering as ensemble_run).",
        default=0,
        minimum=0,
    ),
    STORE,
    COMPARE,
    SAVE_CIRCUIT,
)

register(
    ToolDef(
        name="traj_run",
        stack="traj",
        summary="One trajectory: store its code, probe, cross-check.",
        description=(
            "Simulate one trajectory of a monitored random circuit (the same trajectory "
            "ensemble_run would run as number `trajectory` with this seed) and store its code at "
            "each checkpoint (store=code, the default), or its full state (store=state). probes "
            "are optional call strings; compare_stim checks it against Stim; traj_probe replays "
            'it later at other depths. Example: {"num_system": 64, "layers": 256, '
            '"measurement_rate": 0.1, "graph": "watts_strogatz", "graph_count": 50, '
            '"reference": "purification", "trajectory": 17, "seed": 1, "checkpoints": '
            '"[64, 128, 256]"}'
        ),
        params=TRAJ_PARAMS,
        handler=_run,
        render=lambda d, ctx: _render(d, ctx, "traj_run"),
        read_only=False,
        long_running=True,
        examples=(
            {
                "num_system": 8,
                "layers": 6,
                "measurement_rate": 0.2,
                "reference": "purification",
                "seed": 1,
                "checkpoints": "[3, 6]",
            },
        ),
    )
)

register(
    ToolDef(
        name="traj_probe",
        stack="traj",
        summary="Replay a recorded trajectory with new probes/depths.",
        description=(
            "Replay a trajectory recorded by traj_run, exactly, with new probes at new "
            "checkpoints (and store its code there): probing a trajectory as a function of depth "
            'after the fact. Example: {"source": "runs/traj_run-1a2b3c", "probes": '
            '["code_dimension", "erasure_first_failure(samples=512)"], "checkpoints": "16"}'
        ),
        params=(
            SOURCE,
            PROBES,
            dataclasses.replace(CHECKPOINTS, default=MISSING),
            STORE,
            COMPARE,
            Param("output_dir", "string", "Where to save (default: a folder per request)."),
        ),
        handler=_probe,
        render=lambda d, ctx: _render(d, ctx, "traj_probe"),
        read_only=False,
        long_running=True,
    )
)
