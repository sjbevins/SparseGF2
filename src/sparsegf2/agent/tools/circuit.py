"""Stim-language circuits: inspect, run, convert, cross-check against Stim."""

from __future__ import annotations

from typing import Any

from sparsegf2 import api
from sparsegf2.agent.params import OUTPUT_DIR, SEED, observable_dicts
from sparsegf2.agent.registry import Param, ToolDef, register
from sparsegf2.agent.render import kv, lines, table
from sparsegf2.circuit import Circuit

CIRCUIT = Param(
    "circuit", "string", "Circuit text in Stim's language (CLIFFORD_2Q(i) a b is also accepted)."
)
CIRCUIT_FILE = Param("circuit_file", "string", "A circuit file under the root, instead of circuit.")


def circuit_text(ctx, a: dict[str, Any]) -> str:
    if ("circuit" in a) == ("circuit_file" in a):
        raise ValueError("give exactly one of circuit or circuit_file")
    if "circuit" in a:
        return a["circuit"]
    return ctx.path(a["circuit_file"], what="circuit_file", must_exist=True).read_text(
        encoding="utf-8"
    )


def _inspect(ctx, a):
    return api.inspect_circuit(circuit_text(ctx, a))


def _render_inspect(d, ctx):
    if not d.get("valid"):
        errs = "\n".join(f"line {e['line']}: {e['message']}" for e in d.get("errors", []))
        return f"circuit_inspect: invalid\n{errs}"
    hist = sorted(d["instruction_histogram"].items(), key=lambda kv_: -kv_[1])[:12]
    return lines(
        "circuit_inspect: valid",
        kv(
            d,
            [
                "num_qubits",
                "num_measurements",
                "num_detectors",
                "num_observables",
                "num_ticks",
                "num_operations",
                "stim_compatible",
            ],
        ),
        "instructions: " + ", ".join(f"{k} x{v}" for k, v in hist),
        "next: circuit_run(circuit=...)",
    )


def _run(ctx, a):
    out_dir = str(ctx.path(a["output_dir"], what="output_dir")) if a.get("output_dir") else None
    return api.run_circuit(
        circuit_text(ctx, a),
        shots=a["shots"],
        phases=a["phases"],
        seed=a.get("seed"),
        observables=observable_dicts(a["observables"]) if a.get("observables") else None,
        observe_ticks=a["observe_ticks"],
        tick_stride=a.get("tick_stride"),
        num_qubits=a.get("num_qubits"),
        output_dir=out_dir,
        command="mcp:circuit_run",
    )


def _render_run(d, ctx):
    shots = d["shots"]
    head = (
        f"circuit_run: {d['num_qubits']} qubits, {d['num_measurements']} measurements, "
        f"{shots} shot{'s' if shots != 1 else ''}, seed {d['seed']}, phases {str(d['phases']).lower()}"
    )
    body = kv(
        d,
        [
            "measurements",
            "random",
            "detectors",
            "observables_record",
            "random_fraction",
            "measurement_mean",
            "measurement_means",
            "first_shots",
            "detector_rate_mean",
            "detector_rate_max",
            "detector_rates",
            "any_detector_fraction",
            "observable_rates",
        ],
    )
    obs = d.get("observables")
    obs_text = ""
    if isinstance(obs, dict) and obs:
        obs_text = "observables (final state):\n" + "\n".join(f"  {k}: {v}" for k, v in obs.items())
    ticks = d.get("tick_observables")
    tick_text = ""
    if ticks:
        names = list(ticks["values"])
        rows = [[t, *[ticks["values"][n][i] for n in names]] for i, t in enumerate(ticks["ticks"])]
        tick_text = "per tick:\n" + table(["tick", *names], rows[:40])
    files = f"saved: {d['files']}" if d.get("files") else ""
    return lines(head, body, obs_text, tick_text, files)


def _convert(ctx, a):
    circuit = Circuit(circuit_text(ctx, a))
    text = str(circuit.to_stim()) if a["to"] == "stim" else str(circuit)
    out = {"to": a["to"], "text": text}
    if a.get("output_file"):
        path = ctx.path(a["output_file"], what="output_file")
        path.write_text(text + "\n", encoding="utf-8")
        out["output_file"] = ctx.show(path)
    return out


def _render_convert(d, ctx):
    if d.get("output_file"):
        return f"circuit_convert: wrote {d['to']} text to {d['output_file']}"
    return f"circuit_convert: {d['to']} text\n{d['text']}"


def _crosscheck(ctx, a):
    from sparsegf2 import crosscheck

    circuit = Circuit(circuit_text(ctx, a))
    out: dict[str, Any] = {}
    if a["compare"] in ("stabilizers", "both"):
        out["stabilizers"] = crosscheck.compare_stabilizers(
            circuit,
            num_qubits=a.get("num_qubits"),
            phases=a["phases"],
            ticks=a.get("at_ticks"),
            seed=a["seed"],
        )
    if a["compare"] in ("measurements", "both"):
        out["measurements"] = crosscheck.compare_measurements(
            circuit, num_qubits=a.get("num_qubits"), seed=a["seed"]
        )
    return out


def _render_crosscheck(d, ctx):
    parts = ["circuit_crosscheck: SparseGF2 vs Stim"]
    if "stabilizers" in d:
        s = d["stabilizers"]
        parts.append(
            f"stabilizers: {'match' if s['match'] else 'MISMATCH'} "
            f"({s['ticks_compared']} ticks compared)"
        )
        if not s["match"]:
            m = s["first_mismatch"]
            parts.append(
                f"  first at tick {m['tick']}:\n  sparsegf2 {m['sparsegf2'][:8]}\n  stim      {m['stim'][:8]}"
            )
    if "measurements" in d:
        m = d["measurements"]
        parts.append(
            f"measurements: {'match' if m['match'] else 'MISMATCH'} ({m['measurements']} "
            f"measurements, {m['detectors']} detectors, {m['observables']} observables)"
        )
        if not m["match"]:
            parts.append(f"  first: {m['first_mismatch']}")
    return "\n".join(parts)


register(
    ToolDef(
        name="circuit_inspect",
        stack="circuit",
        summary="Parse a circuit; counts or errors by line.",
        description=(
            "Check a Stim-language circuit without running it: qubit, measurement, detector and "
            "tick counts and an instruction histogram, or every bad line with its line number. "
            'Example: {"circuit": "H 0\\nCX 0 1\\nM 0 1"}'
        ),
        params=(CIRCUIT, CIRCUIT_FILE),
        handler=_inspect,
        render=_render_inspect,
        examples=({"circuit": "H 0\nCX 0 1\nM 0 1"},),
    )
)

register(
    ToolDef(
        name="circuit_run",
        stack="circuit",
        summary="Run a circuit: record, rates, observables.",
        description=(
            "Run a Stim-language circuit from |0...0>. shots=1 gives the measurement record; more "
            "shots give per-measurement means and detector/observable rates. observables are call "
            "strings evaluated on the final state, and after every TICK with observe_ticks "
            '(help_list(what=observables)). Example: {"circuit": "H 0\\nCX 0 1\\nTICK\\nM 0", '
            '"seed": 1, "observables": ["entropy(0:1)"], "observe_ticks": true}'
        ),
        params=(
            CIRCUIT,
            CIRCUIT_FILE,
            Param("shots", "integer", "Number of shots.", default=1, minimum=1),
            Param(
                "phases",
                "boolean",
                "Track signs (outcomes); false is faster, no outcomes.",
                default=True,
            ),
            SEED,
            Param(
                "observables",
                "array",
                "Observable calls, e.g. entropy(0:4), expectation(Z0*Z1).",
                items="string",
            ),
            Param(
                "observe_ticks",
                "boolean",
                "Also evaluate observables after every TICK.",
                default=False,
            ),
            Param("tick_stride", "integer", "With observe_ticks: every k-th TICK only.", minimum=1),
            Param(
                "num_qubits", "integer", "Simulate extra idle qubits up to this count.", minimum=1
            ),
            OUTPUT_DIR,
        ),
        handler=_run,
        render=_render_run,
        long_running=True,
        examples=(
            {
                "circuit": "H 0\nCX 0 1\nTICK\nM 0",
                "seed": 1,
                "observables": ["entropy(0:1)"],
                "observe_ticks": True,
            },
        ),
    )
)

register(
    ToolDef(
        name="circuit_convert",
        stack="circuit",
        summary="Convert to Stim text (expands CLIFFORD_*).",
        description=(
            "Convert a circuit to Stim's language (to=stim: CLIFFORD_1Q/2Q expanded into Stim "
            "gates) or to SparseGF2's normalised text (to=canonical); optionally write a file. "
            'Example: {"circuit": "CLIFFORD_2Q(77) 0 1", "to": "stim"}'
        ),
        params=(
            CIRCUIT,
            CIRCUIT_FILE,
            Param("to", "string", "Target text.", default="stim", enum=("stim", "canonical")),
            Param("output_file", "string", "Write the text to this file under the root."),
        ),
        handler=_convert,
        render=_render_convert,
        read_only=False,
        examples=({"circuit": "CLIFFORD_2Q(77) 0 1\nM 0", "to": "stim"},),
    )
)

register(
    ToolDef(
        name="circuit_crosscheck",
        stack="circuit",
        summary="Compare with Stim (stabilizers, records).",
        description=(
            "Run a noise-free circuit in SparseGF2 and in Stim and compare the canonical "
            "stabilizer groups after every TICK (or at_ticks) and, replaying Stim's random "
            "outcomes, every measurement, detector and observable. Needs Stim installed. "
            'Example: {"circuit": "H 0\\nTICK\\nCX 0 1\\nTICK\\nM 0 1"}'
        ),
        params=(
            CIRCUIT,
            CIRCUIT_FILE,
            Param(
                "compare",
                "string",
                "What to compare.",
                default="both",
                enum=("stabilizers", "measurements", "both"),
            ),
            Param("at_ticks", "array", "Ticks to compare (default every tick).", items="integer"),
            Param("phases", "boolean", "Compare signs too (stabilizers).", default=True),
            Param("seed", "integer", "Stim's seed.", default=0, minimum=0),
            Param(
                "num_qubits",
                "integer",
                "Total qubits (default: those the circuit uses).",
                minimum=1,
            ),
        ),
        handler=_crosscheck,
        render=_render_crosscheck,
        long_running=True,
        requires=("stim",),
        examples=({"circuit": "H 0\nTICK\nCX 0 1\nTICK\nM 0 1"},),
    )
)
