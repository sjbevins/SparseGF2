"""help_topic and help_list: conventions and exact names, generated from the library."""

from __future__ import annotations

from importlib import resources
from typing import Any

from sparsegf2.agent import vocab
from sparsegf2.agent.registry import Param, ToolDef, register

TOPICS = (
    "overview", "conventions", "grammar", "states", "circuits", "ensembles", "probes",
    "trajectories", "codes", "erasure", "purification", "entanglement", "graphs", "analysis",
    "files", "jobs", "stacks", "recipes",
)  # fmt: skip
LISTS = (
    "tools", "gates", "instructions", "observables", "probes", "code_properties", "metrics",
    "graph_families", "edge_weight_families", "protocol_parameters", "stacks",
)  # fmt: skip


def _topic(ctx, args: dict[str, Any]) -> dict[str, Any]:
    text = resources.files("sparsegf2.agent").joinpath("topics", f"{args['topic']}.md")
    return {"topic": args["topic"], "text": text.read_text(encoding="utf-8").strip()}


def _list(ctx, args: dict[str, Any]) -> dict[str, Any]:
    what = args["what"]
    rows: list[str]
    if what == "tools":
        from sparsegf2.agent import INTERFACE_VERSION
        from sparsegf2.agent.registry import REGISTRY

        rows = [f"interface version {INTERFACE_VERSION}, stack {ctx.stack}"]
        rows += [f"{n}: {REGISTRY[n].summary}" for n in ctx.tool_names or REGISTRY]
    elif what == "gates":
        g = vocab.gate_names()
        rows = [f"{k}: {' '.join(v)}" for k, v in g.items() if k in ("unitary", "clifford_index")]
    elif what == "instructions":
        g = vocab.gate_names()
        rows = [
            f"{k}: {' '.join(v)}" for k, v in g.items() if k not in ("unitary", "clifford_index")
        ]
    elif what == "observables":
        d = vocab.observable_descriptions()
        rows = [f"{k}({', '.join(s)}): {d[k]}" for k, s in vocab.observable_signatures().items()]
        rows.append("destabilizers(): the destabilizer generators (phases=true, up to 32 qubits)")
    elif what == "probes":
        d = vocab.probe_descriptions()
        rows = [f"{k}({', '.join(s)}): {d[k]}" for k, s in vocab.probe_signatures().items()]
    elif what == "code_properties":
        from sparsegf2.agent.tools.code import CODE_PROPERTIES

        rows = [f"{k}({', '.join(p)}): {desc}" for k, (p, desc, _) in CODE_PROPERTIES.items()]
    elif what == "metrics":
        from sparsegf2.agent.tools.analysis import METRICS

        rows = [f"{k}({', '.join(p)}): {desc}" for k, (p, desc, _) in METRICS.items()]
    elif what == "graph_families":
        rows = [
            f"{k}({', '.join('graph_' + p for p in s)})" for k, s in vocab.graph_families().items()
        ]
        rows.append(
            "file(graph_file): any geometry from a .json/.npz collection (graph_build saves one)"
        )
    elif what == "edge_weight_families":
        rows = [f"{k}({', '.join(s)})" for k, s in vocab.weight_families().items()]
    elif what == "protocol_parameters":
        from sparsegf2.agent.params import PROTOCOL_PARAMS

        rows = [f"{p.name}: {p.description}" for p in PROTOCOL_PARAMS]
    else:  # stacks
        from sparsegf2.agent import stacks

        rows = [
            f"{r['name']} ({r['kind']}): {len(r.get('tools', []))} tools"
            + (f" - {r['description']}" if r.get("description") else "")
            for r in stacks.catalog(ctx.root)
        ]
    return {"what": what, "rows": rows}


def _render_topic(data: dict[str, Any], ctx) -> str:
    return f"help_topic: {data['topic']}\n{data['text']}"


def _render_list(data: dict[str, Any], ctx) -> str:
    return f"help_list: {data['what']} ({len(data['rows'])})\n" + "\n".join(data["rows"])


register(
    ToolDef(
        name="help_topic",
        stack="help",
        summary="Guidance on one topic, with examples.",
        description=(
            "Read short guidance on a topic before using tools you have not used yet: the "
            "conventions (qubits, layers, seeds, units), the call grammar for probes and "
            'observables, and how each kind of study works. Example: {"topic": "conventions"}'
        ),
        params=(Param("topic", "string", "Which topic.", required=True, enum=TOPICS),),
        handler=_topic,
        render=_render_topic,
        examples=({"topic": "conventions"}, {"topic": "grammar"}),
    )
)

register(
    ToolDef(
        name="help_list",
        stack="help",
        summary="Exact names: tools, gates, probes, observables, ...",
        description=(
            "List exact names with their arguments and one-line meanings, generated from the "
            "simulator itself: tools, gates, instructions, observables, probes, code_properties, "
            "metrics, graph_families, edge_weight_families, protocol_parameters, stacks. "
            'Example: {"what": "probes"}'
        ),
        params=(Param("what", "string", "Which list.", required=True, enum=LISTS),),
        handler=_list,
        render=_render_list,
        examples=({"what": "probes"}, {"what": "observables"}),
    )
)
