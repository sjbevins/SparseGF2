"""Parameter groups shared by several tools, and their translation to JSON API specs.

Protocol parameters carry the ``MonitoredCircuit`` field names exactly; graph
parameters are ``graph_`` plus the generator's parameter name (spec 5.3).
"""

from __future__ import annotations

import difflib
from typing import Any

from sparsegf2.agent import vocab
from sparsegf2.agent.grammar import (
    bind,
    parse_call,
    parse_calls,
    parse_checkpoints,
    parse_qubits,
    plain,
)
from sparsegf2.agent.registry import Param

_FAMILY_DEFAULTS = {"watts_strogatz": {"k": 4, "beta": 0.1}, "newman_watts": {"k": 4}}
_GRAPH_KEYS = (
    "graph_k",
    "graph_beta",
    "graph_p",
    "graph_d",
    "graph_rows",
    "graph_cols",
    "graph_periodic",
)


def _graph_enum() -> tuple[str, ...]:
    return (*vocab.graph_families(), "file")


GRAPH_PARAMS: tuple[Param, ...] = (
    Param(
        "graph",
        "string",
        "Interaction graph family; file = any geometry from graph_file.",
        default="cycle",
        enum=_graph_enum(),
    ),
    Param(
        "graph_k",
        "integer",
        "watts_strogatz/newman_watts: total degree K, even (K/2 neighbours per side). Default 4.",
        minimum=2,
    ),
    Param(
        "graph_beta",
        "number",
        "watts_strogatz: rewiring probability. Default 0.1.",
        minimum=0,
        maximum=1,
    ),
    Param(
        "graph_p",
        "number",
        "newman_watts: shortcut probability per edge; erdos_renyi: edge probability.",
        minimum=0,
        maximum=1,
    ),
    Param("graph_d", "integer", "random_regular: degree.", minimum=1),
    Param("graph_rows", "integer", "lattice_2d: rows (rows*cols = number of qubits).", minimum=1),
    Param("graph_cols", "integer", "lattice_2d: columns.", minimum=1),
    Param("graph_periodic", "boolean", "lattice_2d: periodic boundaries (default true)."),
    Param("graph_file", "string", "graph=file: a .json/.npz graph collection under the root."),
    Param(
        "graph_count",
        "integer",
        "Random families: number of graphs G; graph i comes from seed (graph_seed, i).",
        default=1,
        minimum=1,
    ),
    Param(
        "graph_seed",
        "integer",
        "Random families: seed of the graph collection (separate from the circuit seed).",
        default=0,
        minimum=0,
    ),
)

PROTOCOL_PARAMS: tuple[Param, ...] = (
    Param(
        "num_system",
        "integer",
        "System qubits n (references follow at n..).",
        required=True,
        minimum=2,
    ),
    Param(
        "layers",
        "integer",
        "Depth T: layers of gates then measurements (never rescaled).",
        required=True,
        minimum=0,
    ),
    Param(
        "measurement_rate",
        "number",
        "Probability p that each system qubit is measured in Z after each layer.",
        required=True,
        minimum=0,
        maximum=1,
    ),
    Param(
        "schedule",
        "string",
        "Gate placement: matching (edge colouring; brickwork on an even ring), random_edges "
        "(gates_per_layer edges with replacement), all_edges.",
        default="matching",
        enum=("matching", "random_edges", "all_edges"),
    ),
    Param("gates_per_layer", "integer", "random_edges: gates per layer (default n/2).", minimum=0),
    Param(
        "circuits_per_graph",
        "integer",
        "Random circuits per graph; trajectory t runs on graph t // circuits_per_graph.",
        default=1,
        minimum=1,
    ),
    Param(
        "reference",
        "string",
        "none: start in |0..0>; purification: each system qubit Bell-paired with its own "
        "reference (a code with n logical qubits); single: one reference qubit.",
        default="none",
        enum=("none", "purification", "single"),
    ),
    Param(
        "reference_partner",
        "integer",
        "reference=single: the system qubit paired with it; -1 = random per trajectory.",
        default=-1,
        minimum=-1,
    ),
    Param(
        "scramble_gates",
        "integer",
        "Random all-to-all gates before the dynamics.",
        default=0,
        minimum=0,
    ),
    Param(
        "scramble_matchings",
        "integer",
        "Layers of gates on random perfect matchings before the dynamics.",
        default=0,
        minimum=0,
    ),
    Param(
        "scramble_layers",
        "integer",
        "Gate-only layers of the schedule before the dynamics.",
        default=0,
        minimum=0,
    ),
    Param(
        "gate_set",
        "array",
        "Two-qubit gates to draw from (names like CX, or CLIFFORD_2Q indices); default all 11520.",
        items="string",
    ),
    Param(
        "random_orientation",
        "boolean",
        "Randomly swap each gate's qubits (matters for a non-uniform gate_set).",
        default=False,
    ),
    Param(
        "edge_weights", "string", "random_edges: edge weights as a call, e.g. power_law(alpha=2.0)."
    ),
    Param(
        "stop_when_purified",
        "boolean",
        "End a trajectory at the first checkpoint where the reference is pure.",
        default=False,
    ),
    Param(
        "system_order",
        "string",
        "Qubit order for half-cut/profile/contiguous probes, e.g. 0:16 or [3,1,0,2].",
    ),
    Param(
        "periodic", "boolean", "Whether contiguous runs of system_order wrap around.", default=True
    ),
)

PROBES = Param(
    "probes",
    "array",
    "Probe calls recorded at checkpoints, e.g. code_dimension, entropy(0:8), "
    "erasure_first_failure(samples=512)@[64,128] (help_list(what=probes)).",
    items="string",
)
CHECKPOINTS = Param(
    "checkpoints",
    "string",
    "Layers to record at: final, every_layer, a stride like 16, or a list like [0,64,128].",
    default="final",
)
SEED = Param("seed", "integer", "Circuit seed (omit for a fresh one; it is reported).", minimum=0)
OUTPUT_DIR = Param(
    "output_dir",
    "string",
    "Where results are saved, under the root (default runs/<tool>-<hash of the request>).",
)
SOURCE = Param(
    "source", "string", "A saved run, record or file under the root (result_list).", required=True
)

FIELD_MAP: dict[str, str] = {
    **{p.name: p.name for p in PROTOCOL_PARAMS},
    "edges": "graph",
    "probes": "probes",
    "checkpoints": "checkpoints",
}


def graph_dict(args: dict[str, Any], n: int, ctx) -> dict[str, Any]:
    """The JSON API's graph object for the graph parameters in ``args``."""
    family = args.get("graph", "cycle")
    families = vocab.graph_families()
    given = [k for k in _GRAPH_KEYS if k in args]
    if family == "file":
        if "graph_file" not in args:
            raise ValueError("graph=file needs graph_file (a .json or .npz graph collection)")
        if given:
            raise ValueError(f"{given[0]} does not apply to graph=file (the file fixes the graphs)")
        path = ctx.path(args["graph_file"], what="graph_file", must_exist=True)
        return {"family": "file", "path": str(path)}
    if "graph_file" in args:
        raise ValueError(f"graph_file needs graph=file (graph is {family})")
    takes = families[family]
    out: dict[str, Any] = {"family": family, **_FAMILY_DEFAULTS.get(family, {})}
    for key in given:
        name = key.removeprefix("graph_")
        if name not in takes:
            raise ValueError(
                f"{key} does not apply to graph={family}; {family} takes "
                f"{[f'graph_{t}' for t in takes] or 'no parameters'}"
            )
        out[name] = args[key]
    if family in vocab.random_graph_families():
        out["seed"] = args.get("graph_seed", 0)
        out["realizations"] = args.get("graph_count", 1)
    elif args.get("graph_count", 1) != 1:
        raise ValueError(f"graph_count applies to random families; {family} is deterministic")
    return out


def _suggest(name: str, options) -> str:
    close = difflib.get_close_matches(name, list(options), n=3)
    return f"; did you mean {', '.join(close)}?" if close else f"; choose from {sorted(options)}"


def probe_dicts(items: list[str]) -> list[dict[str, Any]]:
    sigs = vocab.probe_signatures()
    out = []
    for i, call in enumerate(parse_calls(items, allow_at=True, what="probes")):
        if call.name not in sigs:
            raise ValueError(f"probes[{i}]: unknown probe {call.name!r}{_suggest(call.name, sigs)}")
        bound = bind(call, sigs[call.name], what=f"probes[{i}]", keyword_only=["name"])
        d = {"kind": call.name, **plain(bound)}
        if call.at is not None:
            d["checkpoints"] = call.at
        out.append(d)
    return out


def observable_dicts(items: list[str]) -> list[dict[str, Any]]:
    sigs = vocab.observable_signatures()
    out = []
    for i, call in enumerate(parse_calls(items, what="observables")):
        if call.name not in sigs:
            raise ValueError(
                f"observables[{i}]: unknown observable {call.name!r}{_suggest(call.name, sigs)}"
            )
        bound = bind(call, sigs[call.name], what=f"observables[{i}]", keyword_only=["name"])
        out.append({"kind": call.name, **plain(bound)})
    return out


def ensemble_spec(args: dict[str, Any], ctx) -> dict[str, Any]:
    """The JSON API spec (``sparsegf2.api``) for flat ensemble/trajectory arguments."""
    special = {"edge_weights", "system_order", "reference_partner", "gate_set"}
    spec: dict[str, Any] = {
        p.name: args[p.name] for p in PROTOCOL_PARAMS if p.name in args and p.name not in special
    }
    spec["graph"] = graph_dict(args, args["num_system"], ctx)
    if "reference_partner" in args:
        rp = args["reference_partner"]
        spec["reference_partner"] = "random" if rp < 0 else rp
    if "edge_weights" in args:
        call = parse_call(args["edge_weights"], what="edge_weights")
        families = vocab.weight_families()
        if call.name not in families:
            raise ValueError(
                f"edge_weights: unknown family {call.name!r}{_suggest(call.name, families)}"
            )
        spec["edge_weights"] = {
            "family": call.name,
            **plain(bind(call, families[call.name], what="edge_weights")),
        }
    if "gate_set" in args:
        spec["gate_set"] = [int(g) if g.strip().isdigit() else g.strip() for g in args["gate_set"]]
    if "system_order" in args:
        spec["system_order"] = parse_qubits(args["system_order"], what="system_order")
    if "checkpoints" in args:
        spec["checkpoints"] = parse_checkpoints(args["checkpoints"])
    spec["probes"] = probe_dicts(args.get("probes") or [])
    return spec
