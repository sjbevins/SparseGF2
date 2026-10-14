"""Names the tools accept, read from the library itself so that they never go stale."""

from __future__ import annotations

import dataclasses
import inspect
from functools import cache

__all__ = [
    "gate_names",
    "graph_families",
    "graph_required",
    "observable_descriptions",
    "observable_signatures",
    "probe_descriptions",
    "probe_signatures",
    "protocol_fields",
    "random_graph_families",
    "weight_families",
]

_PROBE_KEYWORDS = ("name", "checkpoints")


def _first_line(fn) -> str:
    return (inspect.getdoc(fn) or "").strip().splitlines()[0] if inspect.getdoc(fn) else ""


@cache
def probe_signatures() -> dict[str, tuple[str, ...]]:
    from sparsegf2 import api

    return {
        kind: tuple(p for p in inspect.signature(fn).parameters if p not in _PROBE_KEYWORDS)
        for kind, fn in api._probe_kinds().items()
    }


@cache
def probe_descriptions() -> dict[str, str]:
    from sparsegf2 import api

    return {kind: _first_line(fn) for kind, fn in api._probe_kinds().items()}


@cache
def observable_signatures() -> dict[str, tuple[str, ...]]:
    from sparsegf2 import api

    return {k: (*o.required, *o.optional) for k, o in api._OBSERVABLES.items()}


@cache
def observable_descriptions() -> dict[str, str]:
    from sparsegf2 import api

    return {k: o.description for k, o in api._OBSERVABLES.items()}


@cache
def graph_families() -> dict[str, tuple[str, ...]]:
    from sparsegf2 import api

    skip = {"n", "seed", "max_tries"}
    return {
        name: tuple(p for p in inspect.signature(fn).parameters if p not in skip)
        for name, fn in api._graph_families().items()
    }


def graph_required(family: str) -> tuple[str, ...]:
    """Parameters of ``family`` (besides ``n`` and ``seed``) that have no default."""
    from sparsegf2 import api

    skip = {"n", "seed"}
    return tuple(
        name
        for name, p in inspect.signature(api._graph_families()[family]).parameters.items()
        if name not in skip and p.default is inspect.Parameter.empty
    )


@cache
def random_graph_families() -> frozenset[str]:
    from sparsegf2 import api

    return frozenset(
        name
        for name, fn in api._graph_families().items()
        if "seed" in inspect.signature(fn).parameters
    )


@cache
def weight_families() -> dict[str, tuple[str, ...]]:
    from sparsegf2 import api

    skip = {"edges", "n"}
    return {
        name: tuple(p for p in inspect.signature(fn).parameters if p not in skip)
        for name, fn in api._weight_families().items()
    }


@cache
def protocol_fields() -> tuple[str, ...]:
    from sparsegf2 import mipt

    return tuple(
        f.name for f in dataclasses.fields(mipt.MonitoredCircuit) if not f.name.startswith("_")
    )


@cache
def gate_names() -> dict[str, list[str]]:
    import sparsegf2 as sg

    out: dict[str, list[str]] = {}
    for g in sg.all_gates():
        out.setdefault(g.kind, []).append(g.name)
    return {k: sorted(v) for k, v in out.items()}
