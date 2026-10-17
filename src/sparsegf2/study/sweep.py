"""Sweeps of phase-free monitored circuits over sizes, rewiring probabilities and rates.

A :class:`SweepSpec` names a grid of conditions ``(n, beta, p)`` on Watts–Strogatz graphs and
the protocol of v1.4 (``docs/misc/watts_strogatz_study_spec_v1_4.md``): ``n // 2`` gates per
layer on uniformly drawn edges, uniform two-qubit Cliffords, Z measurements with probability
``p``, phase-free. :func:`run` generates it into a directory with the existing engine
(:meth:`sparsegf2.mipt.MonitoredCircuit.run` with ``indices=``) and the seeds of
:mod:`sparsegf2.study.identity`: one graph file per ``(n, beta)``, one directory of saved
:class:`~sparsegf2.mipt.EnsembleResult` parts per condition, and an index, ``sweep.json``.
Runs resume, and a sweep grows (more sizes, rewiring probabilities, rates, graphs or circuits
per graph) without changing any saved trajectory. ``docs/SWEEPS.md`` is the full description.
"""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import dataclasses
import json
import math
import operator
import os
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from sparsegf2 import api, graphs, mipt
from sparsegf2._version import __version__
from sparsegf2.study import identity
from sparsegf2.study.identity import Token

__all__ = [
    "DEFAULT_PROBES",
    "INDEX_FILE",
    "SweepSpec",
    "geometric_checkpoints",
    "load",
    "parts",
    "replay",
    "rows",
    "run",
    "status",
    "verify",
]

FORMAT = "sparsegf2.sweep/1"
INDEX_FILE = "sweep.json"
GRAPH_DIR = "graphs"
#: The probes of a spec that names none: k at every layer, the joint tableau at the checkpoints.
DEFAULT_PROBES = (
    {"kind": "code_dimension", "checkpoints": "every_layer"},
    {"kind": "tableau_snapshot"},
)
DEFAULT_STRIDE = 1024
DEFAULT_CHECKPOINTS = {"count": 12}
PICTURES = ("purification", "single", "none")
#: The largest result one saved part holds in memory (``api._result_bytes``).
PART_BYTES = 2**28
#: How parts are compressed by default (``EnsembleResult.save``): zlib level 1, measured in
#: ``docs/SWEEPS.md`` section 12 against "none" and "default".
DEFAULT_COMPRESSION = "fast"
#: Writer threads that save parts while the next part computes; at most this many saves wait,
#: so at most ``DEFAULT_WRITERS + 1`` parts (of at most :data:`PART_BYTES`) are in memory.
DEFAULT_WRITERS = 4
#: Without a chunk size, every chunk after the first holds about this many seconds of
#: simulation at the rate the previous chunk measured.
CHUNK_SECONDS = 10.0
_PART = re.compile(r"result-(\d{12,})\.npz")  # the first u, zero-padded to 12 digits
_REQUIRED = (
    "sizes",
    "betas",
    "rates",
    "graphs",
    "circuits_per_graph",
    "layers",
    "master_graph_seed",
    "master_circuit_seed",
)
_OPTIONAL = ("stride", "checkpoints", "probes", "picture", "stop_when_purified")


# ---------------------------------------------------------------------------
# the checkpoint rule (v1.4 section 9)
# ---------------------------------------------------------------------------


def _half_up_root(s: int, q: int) -> int:
    """``floor(x + 1/2)`` for ``x = s^(1/q) / 2``, exactly: ``(r + 1) // 2`` with ``r`` the
    integer ``q``-th root of ``s`` (ties round upward)."""
    r = int(2.0 ** (math.log2(s) / q))  # an estimate, corrected exactly below
    while r**q > s:
        r -= 1
    while (r + 1) ** q <= s:
        r += 1
    return (r + 1) // 2


def round_half_up_power(t_max: int, j: int, q: int) -> int:
    """The nearest integer to ``t_max^(j/q)``, ties upward, in integer arithmetic.

    With ``a = T^(j/q)`` and ``s = 2^q T^j``, ``a = s^(1/q) / 2``: no floating-point error can
    move a value across a rounding boundary.
    """
    if t_max < 1 or q < 1 or not 0 <= j <= q:
        raise ValueError(f"need T >= 1, q >= 1 and 0 <= j <= q, got T={t_max}, j={j}, q={q}")
    return _half_up_root(2**q * t_max**j, q)


def geometric_checkpoints(t_max: int, count: int) -> tuple[int, ...]:
    """The geometric rule of v1.4 §9: ``a_j = T^(j/(N-1))``, ``j = 0..N-1``, each rounded to
    the nearest integer (ties upward), plus ``0, 1, T``; sorted, without repeats. ``N = 1``
    gives ``0, 1, T`` only."""
    for value, what, top in ((t_max, "T_max", None), (count, "checkpoint count", 1000)):
        identity._integer(value, what, minimum=1, maximum=top)
    t_max, count = operator.index(t_max), operator.index(count)
    points = {0, 1, t_max}
    if count >= 2:
        points.update(round_half_up_power(t_max, j, count - 1) for j in range(count))
    return tuple(sorted(points))


# ---------------------------------------------------------------------------
# the spec
# ---------------------------------------------------------------------------


def _per_size(value: Any, where: str, sizes: Sequence[int], parse: Callable[[Any, str], Any]):
    """``{n: parse(value)}`` for one value, or a table ``{"16": v, ...}`` covering every size."""
    if not isinstance(value, Mapping):
        one = parse(value, where)
        return {n: one for n in sizes}
    table = {}
    for key, v in value.items():
        if not isinstance(key, str) or not re.fullmatch(r"[1-9][0-9]*", key):
            raise ValueError(f"{where}: table keys are sizes such as '16', got {key!r}")
        table[int(key)] = parse(v, f"{where}[{key}]")
    if sorted(table) != list(sizes):
        raise ValueError(f"{where}: the table must have one entry per size {list(sizes)}")
    return table


def _count(value: Any, where: str) -> int:
    return identity._integer(value, where, minimum=1)


def _labels(value: Any, where: str) -> int:
    """A count of graphs or circuits: their labels g, c are 32-bit seed words."""
    return identity._integer(value, where, minimum=1, maximum=identity.SEED_LIMIT)


def _tokens(value: Any, where: str) -> tuple[Token, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{where}: give a non-empty list")
    out = sorted(Token.parse(v, what=f"{where}[{i}]") for i, v in enumerate(value))
    if len(set(out)) != len(out) or len({identity.token_words(t) for t in out}) != len(out):
        raise ValueError(f"{where}: a value appears twice")
    return tuple(out)


def _layers(value: Any, sizes: Sequence[int]) -> dict[int, int]:
    if isinstance(value, Mapping) and list(value) == ["per_n"]:
        factor = _count(value["per_n"], "layers.per_n")
        return {n: factor * n for n in sizes}
    if not isinstance(value, Mapping):
        raise ValueError('layers: give {"per_n": a} (a n layers) or a table {"16": 128, ...}')
    return _per_size(value, "layers", sizes, _count)


def _checkpoint_rule(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or len(value) != 1
        or next(iter(value))
        not in (
            "count",
            "list",
        )
    ):
        raise ValueError('checkpoints: give {"count": N} (the geometric rule) or {"list": [...]}')
    if "count" in value:
        return {"count": _count(value["count"], "checkpoints.count")}
    raw = value["list"]
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise ValueError("checkpoints.list: give a non-empty list of layers")
    depths = {identity._integer(d, f"checkpoints.list[{i}]", minimum=0) for i, d in enumerate(raw)}
    return {"list": sorted(depths)}


@dataclasses.dataclass(frozen=True)
class SweepSpec:
    """A sweep: the grid of conditions ``(n, beta, p)``, the graphs and circuits of each, the
    depth, the probes and the master seeds. Build it with :meth:`from_json`.

    Per-size fields (``rates``, ``graphs``, ``circuits_per_graph``, ``layers``) hold one value
    per size. :meth:`to_json` is the canonical form (every default written out, every per-size
    value resolved) and :attr:`digest` its SHA-256.
    """

    sizes: tuple[int, ...]
    betas: tuple[Token, ...]
    rates: dict[int, tuple[Token, ...]]
    graphs: dict[int, int]
    circuits_per_graph: dict[int, int]
    layers: dict[int, int]
    checkpoints: dict[str, Any]
    stride: int
    master_graph_seed: int
    master_circuit_seed: int
    probes: tuple[dict[str, Any], ...]
    picture: str
    stop_when_purified: bool

    @classmethod
    def from_json(cls, data: Any) -> SweepSpec:
        """The spec of a JSON object; every field is checked, and the engine checks the
        protocol and the probes at every size (``ValueError`` naming the field)."""
        try:
            return cls._from_json(data)
        except ValueError as exc:
            raise ValueError(f"sweep spec: {exc}") from None

    @classmethod
    def _from_json(cls, data: Any) -> SweepSpec:
        if not isinstance(data, Mapping):
            raise ValueError("expected an object")
        unknown = sorted(set(data) - {*_REQUIRED, *_OPTIONAL}, key=str)
        missing = [k for k in _REQUIRED if k not in data]
        if unknown or missing:
            raise ValueError(
                f"unknown key(s) {unknown}, missing key(s) {missing}; the keys are "
                f"{[*_REQUIRED, *_OPTIONAL]}"
            )
        raw = data["sizes"]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
            raise ValueError("sizes: give a non-empty list")
        sizes = sorted(identity._integer(v, f"sizes[{i}]", minimum=5) for i, v in enumerate(raw))
        if len(set(sizes)) != len(sizes):
            raise ValueError("sizes: a size appears twice")
        probes = data.get("probes", list(DEFAULT_PROBES))
        if (
            isinstance(probes, (str, bytes))
            or not isinstance(probes, Sequence)
            or not all(isinstance(q, Mapping) for q in probes)
        ):
            raise ValueError('probes: give a list of objects like [{"kind": "code_dimension"}]')
        circuits = _per_size(data["circuits_per_graph"], "circuits_per_graph", sizes, _labels)
        spec = cls(
            sizes=tuple(sizes),
            betas=_tokens(data["betas"], "betas"),
            rates=_per_size(data["rates"], "rates", sizes, _tokens),
            graphs=_per_size(data["graphs"], "graphs", sizes, _labels),
            circuits_per_graph=circuits,
            layers=_layers(data["layers"], sizes),
            checkpoints=_checkpoint_rule(data.get("checkpoints", DEFAULT_CHECKPOINTS)),
            stride=identity._integer(
                data.get("stride", DEFAULT_STRIDE), "stride", minimum=max(circuits.values())
            ),
            master_graph_seed=identity._seed_word(data["master_graph_seed"], "master_graph_seed"),
            master_circuit_seed=identity._seed_word(
                data["master_circuit_seed"], "master_circuit_seed"
            ),
            probes=tuple(json.loads(identity.canonical_json(q)) for q in probes),
            picture=data.get("picture", "purification"),
            stop_when_purified=data.get("stop_when_purified", False),
        )
        top = (max(spec.graphs.values()) - 1) * spec.stride + max(circuits.values()) - 1
        if max(top, spec.stride) >= 2**63:  # the engine also takes stride as an int64
            raise ValueError(
                f"stride: the stride {spec.stride} and the largest stream index (graphs - 1) * "
                f"stride + circuits_per_graph - 1 = {top} must fit the engine's int64 indices "
                "(below 2**63); lower stride or graphs"
            )
        if spec.picture not in PICTURES:
            raise ValueError(f"picture must be one of {list(PICTURES)}, got {spec.picture!r}")
        if not isinstance(spec.stop_when_purified, bool):
            raise ValueError("stop_when_purified must be true or false")
        for n in sizes:  # the engine's own validation of the protocol and the probes
            try:
                spec.monitored_circuit(
                    n, spec.betas[0], spec.rates[n][0], [identity.circulant_edges(n)]
                )
            except ValueError as exc:
                raise ValueError(f"at n = {n}: {exc}") from None
        return spec

    def to_json(self) -> dict[str, Any]:
        """The canonical JSON object (``from_json(to_json())`` is this spec)."""

        def table(values: Mapping[int, Any]) -> dict[str, Any]:
            return {str(n): values[n] for n in self.sizes}

        return {
            "sizes": list(self.sizes),
            "betas": [b.token for b in self.betas],
            "rates": table({n: [p.token for p in ps] for n, ps in self.rates.items()}),
            "graphs": table(self.graphs),
            "circuits_per_graph": table(self.circuits_per_graph),
            "layers": table(self.layers),
            "checkpoints": dict(self.checkpoints),
            "stride": self.stride,
            "master_graph_seed": self.master_graph_seed,
            "master_circuit_seed": self.master_circuit_seed,
            "probes": [dict(p) for p in self.probes],
            "picture": self.picture,
            "stop_when_purified": self.stop_when_purified,
        }

    @property
    def digest(self) -> str:
        return identity.json_digest(self.to_json())

    def conditions(self) -> list[tuple[int, Token, Token]]:
        """Every ``(n, beta, p)``, in the order runs go through them."""
        return [(n, b, p) for n in self.sizes for b in self.betas for p in self.rates[n]]

    def checkpoints_at(self, n: int) -> tuple[int, ...]:
        """The sparse checkpoints at size ``n``: the geometric rule, or the list's entries up
        to ``T``; always with ``0``, ``1`` and ``T``."""
        t = self.layers[n]
        if "count" in self.checkpoints:
            return geometric_checkpoints(t, self.checkpoints["count"])
        return tuple(sorted({0, 1, t, *(d for d in self.checkpoints["list"] if d <= t)}))

    def indices(self, n: int) -> np.ndarray:
        """The stream indices ``u = g * stride + c`` of every trajectory at size ``n``, sorted."""
        g = np.arange(self.graphs[n], dtype=np.int64)[:, None]
        c = np.arange(self.circuits_per_graph[n], dtype=np.int64)[None, :]
        return (g * self.stride + c).ravel()

    def run_seed(self, n: int, beta: Any) -> tuple[int, ...]:
        """The seed of every trajectory of ``(n, beta)``, the same at every rate."""
        return identity.run_seed(self.master_circuit_seed, n, beta, "shared")

    def engine_spec(self, n: int, beta: Any, p: Any) -> dict[str, Any]:
        """The condition's engine spec in the JSON form of :mod:`sparsegf2.api`, without the
        graph: the protocol of v1.4 §8.1 and the probes with the sparse checkpoints filled in."""
        Token.parse(beta, what="beta")
        sparse = list(self.checkpoints_at(n))
        partner = {"reference_partner": "random"} if self.picture == "single" else {}
        return {
            "num_system": n,
            "circuits_per_graph": self.stride,
            "layers": self.layers[n],
            "measurement_rate": Token.parse(p, what="p").float64,
            "reference": self.picture,
            **partner,
            "schedule": "random_edges",
            "gates_per_layer": n // 2,
            "random_orientation": False,
            "scramble_gates": 0,
            "scramble_matchings": 0,
            "scramble_layers": 0,
            "periodic": True,
            "checkpoints": "every_layer",
            "stop_when_purified": self.stop_when_purified,
            "probes": [
                {**q, "checkpoints": sparse} if q.get("checkpoints") is None else dict(q)
                for q in self.probes
            ],
        }

    def monitored_circuit(
        self, n: int, beta: Any, p: Any, collection: Sequence[np.ndarray]
    ) -> mipt.MonitoredCircuit:
        """The condition's :class:`~sparsegf2.mipt.MonitoredCircuit` on the graph collection
        ``collection`` (graph ``g`` is ``collection[g]``)."""
        spec = self.engine_spec(n, beta, p)
        probes = [api._build_probe(q, i, n) for i, q in enumerate(spec.pop("probes"))]
        return mipt.MonitoredCircuit(edges=cast(Any, list(collection)), probes=probes, **spec)


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------


def _name(n: int, beta: Token, p: Token | None = None) -> str:
    """``n16_beta0.1`` or ``n16_beta0.1_p1-3``: a '/' of a rational token becomes '-'."""
    text = f"n{n}_beta{beta.token}" + ("" if p is None else f"_p{p.token}")
    return text.replace("/", "-")


def _graph_file(root: Path, n: int, beta: Token) -> Path:
    return root / GRAPH_DIR / f"{_name(n, beta)}.npz"


def _read_index(root: Path) -> dict[str, Any] | None:
    path = root / INDEX_FILE
    if not path.exists():
        return None
    index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or index.get("format") != FORMAT:
        raise ValueError(f"{str(path)!r} is not a sweep index ({FORMAT})")
    return index


def _index(sweep: SweepSpec) -> dict[str, Any]:
    pairs = [(n, b) for n in sweep.sizes for b in sweep.betas]
    return {
        "format": FORMAT,
        "digest": sweep.digest,
        "spec": sweep.to_json(),
        "versions": identity.version_record(sweep.picture),
        "graphs": [
            {
                "n": n,
                "beta": b.token,
                "file": f"{GRAPH_DIR}/{_name(n, b)}.npz",
                "records": f"{GRAPH_DIR}/{_name(n, b)}.json",
                "graphs": sweep.graphs[n],
            }
            for n, b in pairs
        ],
        "conditions": [
            {
                "n": n,
                "beta": b.token,
                "p": p.token,
                "p_float64": p.hex,
                "path": _name(n, b, p),
                "run_seed": list(sweep.run_seed(n, b)),
                "trajectories": sweep.graphs[n] * sweep.circuits_per_graph[n],
            }
            for n, b, p in sweep.conditions()
        ],
    }


def _spec_of(root: Path) -> SweepSpec:
    index = _read_index(root)
    if index is None:
        raise ValueError(f"{str(root)!r} holds no sweep ({INDEX_FILE} is missing)")
    if index["versions"]["randomness_contract"] != mipt.RANDOMNESS_CONTRACT:
        raise ValueError(
            f"{str(root)!r} was made under randomness contract "
            f"{index['versions']['randomness_contract']}; this version uses "
            f"{mipt.RANDOMNESS_CONTRACT} and cannot reproduce its trajectories"
        )
    return SweepSpec.from_json(index["spec"])


def _collection(root: Path, sweep: SweepSpec, n: int, beta: Token) -> list[np.ndarray]:
    """Graphs ``0 .. G-1`` of ``(n, beta)`` from its graph file (written by :func:`run`)."""
    path = _graph_file(root, n, beta)
    if not path.exists():
        raise ValueError(f"no graph file {str(path)!r}: run the sweep first")
    _, collection, _ = graphs.load(path)
    return collection[: sweep.graphs[n]]


def _write_graphs(root: Path, sweep: SweepSpec, n: int, beta: Token) -> None:
    """Build graphs ``0 .. G-1`` of ``(n, beta)`` from their seeds and write the graph file and
    its records, refusing a graph whose recorded digest it no longer reproduces."""
    path = _graph_file(root, n, beta)
    records_path = path.with_suffix(".json")
    built = [
        identity.build_graph(n, beta, g, sweep.master_graph_seed) for g in range(sweep.graphs[n])
    ]
    if records_path.exists():
        old = json.loads(records_path.read_text(encoding="utf-8"))["graphs"]
        for record in old[: len(built)]:
            if record["edge_digest"] != built[record["g"]].edge_digest:
                raise ValueError(
                    f"graph {record['g']} of n = {n}, beta = {beta}: its seed now gives another "
                    "graph than the one the saved trajectories ran on (another graph "
                    "generator); use a new directory"
                )
        if len(old) >= len(built) and path.exists():
            return
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.npz")
    graphs.save(tmp, [r.edges for r in built], n)
    api._replace(tmp, path)
    records = {
        "n": n,
        "beta": beta.token,
        "generator": identity.GRAPH_GENERATOR_VERSION,
        "edge_order": identity.ORIGINAL_EDGE_ORDER,
        "graphs": [r.to_dict() for r in built],
    }
    api._atomic_write_text(records_path, json.dumps(records, indent=1))


def _part_paths(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    found = [p for p in folder.iterdir() if _PART.fullmatch(p.name)]
    return sorted(found, key=lambda p: int(p.name[7:-4]))  # by the first stream index


def _saved_indices(folder: Path) -> np.ndarray:
    """The stream indices the saved parts of a condition hold (reads one array of each)."""
    held = []
    for path in _part_paths(folder):
        with np.load(path, allow_pickle=False) as z:
            held.append(z["trajectories"])
    return np.concatenate(held).astype(np.int64) if held else np.zeros(0, dtype=np.int64)


# ---------------------------------------------------------------------------
# growing a sweep
# ---------------------------------------------------------------------------


_FROZEN = (
    "stride",
    "master_graph_seed",
    "master_circuit_seed",
    "picture",
    "stop_when_purified",
    "probes",
    "checkpoints",
)


def _check_growth(old: SweepSpec, new: SweepSpec) -> None:
    """Refuse a spec that would change a saved trajectory; it may only add to the sweep."""
    for name in _FROZEN:
        if getattr(old, name) != getattr(new, name):
            raise ValueError(
                f"sweep spec: {name} is {api._shown(getattr(old, name))} in {INDEX_FILE} and "
                f"{api._shown(getattr(new, name))} here; the saved trajectories were made with "
                "it, so it cannot change (use a new directory)"
            )
    for name, before, after in (
        ("sizes", set(old.sizes), set(new.sizes)),
        ("betas", set(old.betas), set(new.betas)),
    ):
        if not before <= after:
            gone = sorted(str(v) for v in before - after)
            raise ValueError(f"sweep spec: {name} drops {gone}; a sweep only grows")
    for n in old.sizes:
        if not set(old.rates[n]) <= set(new.rates[n]):
            gone = sorted(str(p) for p in set(old.rates[n]) - set(new.rates[n]))
            raise ValueError(f"sweep spec: rates at n = {n} drop {gone}; a sweep only grows")
        for name in ("graphs", "circuits_per_graph"):
            if getattr(new, name)[n] < getattr(old, name)[n]:
                raise ValueError(
                    f"sweep spec: {name} at n = {n} is {getattr(old, name)[n]} in "
                    f"{INDEX_FILE} and {getattr(new, name)[n]} here; a sweep only grows"
                )
        if new.layers[n] != old.layers[n]:
            raise ValueError(
                f"sweep spec: layers at n = {n} is {old.layers[n]} in {INDEX_FILE} and "
                f"{new.layers[n]} here; it fixes every saved trajectory (use a new directory)"
            )


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


def run(
    spec: SweepSpec | Mapping[str, Any],
    directory: str | Path,
    *,
    num_threads: int | None = None,
    chunk_size: int | None = None,
    compression: str = DEFAULT_COMPRESSION,
    writers: int = DEFAULT_WRITERS,
    log: Callable[[str], None] | None = None,
    command: str = "sweep.run",
) -> dict[str, Any]:
    """Generate (or resume, or grow) the sweep ``spec`` in ``directory``; returns :func:`status`.

    A directory made by another package version may grow when every other recorded version
    (randomness contract, graph generator, Clifford numbering, dynamics, row-space format, seed
    tags) is equal and, before anything is written, this code replays the first and the last
    saved trajectory of each condition that grows identically (:func:`verify`). Every graph
    file is written first; then each condition runs its missing trajectories in chunks of
    ``chunk_size`` trajectories (default: the first chunk from ``estimate_ensemble``'s cost
    model, each later one about :data:`CHUNK_SECONDS` at the rate the previous one measured;
    at most :data:`PART_BYTES` of result). Each chunk is saved as a part on one of ``writers``
    threads while the next one computes, with ``compression`` ``"fast"``, ``"none"`` or
    ``"default"`` (``EnsembleResult.save``; not part of the sweep's identity), and listed with
    this package version and its compression under ``parts`` in the condition's
    ``provenance.json``. Results do not depend on ``num_threads``, ``chunk_size``,
    ``compression`` or ``writers``. The directory's lock admits one writer.
    """
    sweep = spec if isinstance(spec, SweepSpec) else SweepSpec.from_json(spec)
    if num_threads is not None:
        num_threads = identity._integer(num_threads, "num_threads", minimum=1)
    if chunk_size is not None:
        chunk_size = identity._integer(chunk_size, "chunk_size", minimum=1)
    writers = identity._integer(writers, "writers", minimum=1)
    if compression not in mipt.SAVE_COMPRESSIONS:
        raise ValueError(
            f"compression must be one of {list(mipt.SAVE_COMPRESSIONS)}, got {compression!r}"
        )
    root = api._output_dir(directory)
    lock = api._lock_output_dir(root)
    started, ran = time.perf_counter(), 0
    try:
        index = _read_index(root)
        if index is not None:
            _check_growth(SweepSpec.from_json(index["spec"]), sweep)
            now = identity.version_record(sweep.picture)
            for key, value in index["versions"].items():
                if key != "sparsegf2" and now.get(key) != value:
                    raise ValueError(
                        f"{str(root)!r} was made with {key} {value!r}, this is {now.get(key)!r}; "
                        "trajectories of different versions need not agree: use a new directory"
                    )
            for n, beta, p in sweep.conditions():  # before anything is written
                done = _saved_indices(root / _name(n, beta, p))
                if done.size and np.setdiff1d(sweep.indices(n), done).size:
                    _replay_check(root, n, beta, p, done)
        for n in sweep.sizes:
            for beta in sweep.betas:
                _write_graphs(root, sweep, n, beta)
        api._atomic_write_text(root / INDEX_FILE, json.dumps(_index(sweep), indent=1))
        options = {
            "num_threads": num_threads,
            "chunk_size": chunk_size,
            "compression": compression,
            "writers": writers,
            "log": log,
            "command": command,
        }
        for n, beta, p in sweep.conditions():
            ran += _run_condition(root, sweep, n, beta, p, options)
    finally:
        lock.release()
    out = status(root)
    out.update(ran=ran, seconds=round(time.perf_counter() - started, 3))
    return out


def _replay_check(root: Path, n: int, beta: Token, p: Token, done: np.ndarray) -> None:
    """Refuse to add to a condition whose first and last saved trajectories this code does not
    replay exactly (every stored value and snapshot word): the saved parts were then made by a
    version that draws other trajectories."""
    for u in sorted({int(done.min()), int(done.max())}):
        try:
            same, why = verify(root, n, beta, p, u)["identical"], ""
        except ValueError as exc:
            same, why = False, f" ({exc})"
        if not same:
            raise ValueError(
                f"n = {n}, beta = {beta}, p = {p}: the saved trajectory u = {u} does not replay "
                f"identically with this code (sparsegf2 {__version__}){why}; the versions "
                "disagree, so this sweep cannot grow: use a new directory"
            )


_PROVENANCE_LOCK = threading.Lock()


def _record_part(folder: Path, name: str, compression: str) -> None:
    """List ``name`` with this package version and its compression under ``parts`` in
    ``provenance.json`` (the writers of one condition take turns)."""
    with _PROVENANCE_LOCK:
        path = folder / "provenance.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        entry = {"sparsegf2": __version__, "compression": compression}
        record.setdefault("parts", {})[name] = entry
        api._atomic_write_text(path, json.dumps(record, indent=2))


def _save_part(result: mipt.EnsembleResult, folder: Path, name: str, compression: str) -> None:
    """Save a part to a temporary file of this writer's own, then rename it (atomic)."""
    path = folder / name
    tmp = api._temporary(path)
    try:
        result.save(tmp, compression=compression)
        with open(tmp, "r+b") as fh:  # on disk before the rename makes it a part
            os.fsync(fh.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    api._replace(tmp, path)
    _record_part(folder, name, compression)


def _run_condition(
    root: Path,
    sweep: SweepSpec,
    n: int,
    beta: Token,
    p: Token,
    options: dict[str, Any],
) -> int:
    folder = root / _name(n, beta, p)
    folder.mkdir(exist_ok=True)
    for stale in folder.glob("*.tmp.npz"):  # left by a writer that was killed while saving
        stale.unlink()
    done = _saved_indices(folder)
    todo = np.setdiff1d(sweep.indices(n), done)
    if not todo.size:
        return 0
    engine = {**sweep.engine_spec(n, beta, p), "graph": {"family": "file"}}
    engine["graph"]["path"] = str(_graph_file(root, n, beta))
    api._write_run_metadata(
        folder,
        engine,
        int(sweep.indices(n).size),
        cast(Any, list(sweep.run_seed(n, beta))),
        options["command"],
        resumed_from=int(done.size) if (folder / "spec.json").exists() else None,
        graph_file=api._graph_file_record(engine),
    )
    mc = sweep.monitored_circuit(n, beta, p, _collection(root, sweep, n, beta))
    threads_, chosen = options["num_threads"], options["chunk_size"]
    cap = max(1, PART_BYTES // max(1, api._result_bytes(mc, 1)))
    size = min(
        cap, chosen or api._default_chunk_size(mc, int(todo.size), api._running_threads(threads_))
    )
    seed, writers, log = list(sweep.run_seed(n, beta)), options["writers"], options["log"]
    start = 0
    # Each part is saved on a writer thread while the next one computes (the kernels and
    # zlib release the GIL); at most ``writers`` saves wait, so at most writers + 1 parts
    # are in memory.
    with concurrent.futures.ThreadPoolExecutor(max_workers=writers) as pool:
        pending: collections.deque[concurrent.futures.Future[None]] = collections.deque()
        while start < todo.size:
            chunk = todo[start : start + size]
            result = mc.run(indices=chunk, seed=seed, num_threads=threads_)
            while len(pending) >= writers:
                pending.popleft().result()
            name = f"result-{int(chunk[0]):012d}.npz"
            pending.append(pool.submit(_save_part, result, folder, name, options["compression"]))
            start += int(chunk.size)
            if log is not None:
                log(
                    f"n={n} beta={beta} p={p}: {done.size + start}/{done.size + todo.size} "
                    f"trajectories ({result.trajectories_per_second:.0f}/s)"
                )
            if chosen is None and result.elapsed_seconds > 0:  # chunks of CHUNK_SECONDS
                rate = chunk.size / result.elapsed_seconds
                size = max(1, min(cap, int(CHUNK_SECONDS * rate)))
            del result
        for future in pending:
            future.result()
    return int(todo.size)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def status(directory: str | Path) -> dict[str, Any]:
    """Every condition's target and saved trajectories (reads only the stream indices)."""
    root = Path(directory).expanduser().resolve()
    index = _read_index(root)
    if index is None:
        raise ValueError(f"{str(root)!r} holds no sweep ({INDEX_FILE} is missing)")
    conditions = []
    for c in index["conditions"]:
        saved = int(_saved_indices(root / c["path"]).size)
        keys = ("n", "beta", "p", "path", "trajectories")
        conditions.append({**{k: c[k] for k in keys}, "saved": saved})
    target = sum(c["trajectories"] for c in conditions)
    saved = sum(c["saved"] for c in conditions)
    return {
        "directory": str(root),
        "digest": index["digest"],
        "complete": saved == target,
        "trajectories": target,
        "saved": saved,
        "conditions": conditions,
    }


def _condition(root: Path, n: int, beta: Any, p: Any) -> tuple[SweepSpec, Token, Token, Path]:
    sweep = _spec_of(root)
    b, q = Token.parse(beta, what="beta"), Token.parse(p, what="p")
    if (n, b, q) not in sweep.conditions():
        raise ValueError(f"the sweep has no condition n = {n}, beta = {b}, p = {q}")
    return sweep, b, q, root / _name(n, b, q)


def _rebased(part: mipt.EnsembleResult, mc: mipt.MonitoredCircuit, where: Path):
    """``part`` on the condition's current spec, after checking that it is that spec on the
    first graphs of the collection (a part saved before graphs were added holds fewer)."""
    count = part.spec.num_graphs
    if count > mc.num_graphs:
        difference: str | None = "edges"
    else:
        same = (
            mc
            if count == mc.num_graphs
            else dataclasses.replace(mc, edges=cast(Any, mc.graphs[:count]))
        )
        difference = mipt._protocol_difference(part.spec, same)
    if difference is not None:
        raise ValueError(f"{str(where)!r} was not made by this sweep's spec ({difference} differs)")
    return dataclasses.replace(part, spec=mc)


def _reader(directory: str | Path, n: int, beta: Any, p: Any):
    """The condition's folder and a loader of one saved part, rebased on the current spec."""
    root = Path(directory).expanduser().resolve()
    sweep, b, q, folder = _condition(root, n, beta, p)
    mc = sweep.monitored_circuit(n, b, q, _collection(root, sweep, n, b))
    seed = list(sweep.run_seed(n, b))

    def read(path: Path) -> mipt.EnsembleResult:
        part = mipt.EnsembleResult.load(path)
        if part.seed_entropy != seed:
            raise ValueError(f"{str(path)!r} was not made by this sweep (its seed differs)")
        return _rebased(part, mc, path)

    return folder, read


def parts(directory: str | Path, n: int, beta: Any, p: Any) -> Iterator[mipt.EnsembleResult]:
    """The saved parts of condition ``(n, beta, p)`` one at a time, each on the condition's
    current spec (for data too large to hold at once)."""
    folder, read = _reader(directory, n, beta, p)
    for path in _part_paths(folder):
        yield read(path)


def load(directory: str | Path, n: int, beta: Any, p: Any) -> mipt.EnsembleResult:
    """Every saved trajectory of condition ``(n, beta, p)`` as one result, rows sorted by the
    stream index ``u`` (row ``i`` is trajectory ``result.trajectories[i]``)."""
    found = list(parts(directory, n, beta, p))
    if not found:
        raise ValueError(f"nothing saved for n = {n}, beta = {beta}, p = {p}")
    # parts of other package versions passed the replay check of run before they were joined
    merged = mipt.EnsembleResult.concatenate(found, same_version=False)
    return _take(merged, np.argsort(merged.trajectories, kind="stable"))


def _take(result: mipt.EnsembleResult, rows_: np.ndarray) -> mipt.EnsembleResult:
    tau = result.purification_layer
    return dataclasses.replace(
        result,
        trajectories=np.asarray(result.trajectories)[rows_],
        values={name: v[rows_] for name, v in result.values.items()},
        purification_layer=None if tau is None else tau[rows_],
        final_layer=result.final_layer[rows_],
        num_measurements=result.num_measurements[rows_],
        snapshots={k: (g[rows_], c[rows_]) for k, (g, c) in result.snapshots.items()},
    )


def rows(directory: str | Path, n: int, beta: Any, p: Any) -> dict[str, np.ndarray]:
    """The identity of every row of :func:`load` (the same order): ``n``, ``beta``, ``p`` (the
    tokens), ``g``, ``c``, ``u`` and the ``graph_seed``, ``run_seed`` and ``outcome_seed`` words
    of each trajectory (one row each)."""
    root = Path(directory).expanduser().resolve()
    sweep, b, q, folder = _condition(root, n, beta, p)
    u = np.sort(_saved_indices(folder))
    g, c = u // sweep.stride, u % sweep.stride
    m, s = sweep.master_graph_seed, sweep.master_circuit_seed
    return {
        "n": np.full(u.size, n, dtype=np.int64),
        "beta": np.full(u.size, b.token),
        "p": np.full(u.size, q.token),
        "g": g,
        "c": c,
        "u": u,
        "graph_seed": np.array([identity.graph_seed(m, n, b, int(x)) for x in g], np.uint32),
        "run_seed": np.tile(np.array(sweep.run_seed(n, b), dtype=np.uint32), (u.size, 1)),
        "outcome_seed": np.array(
            [identity.outcome_seed(s, n, b, q, int(x), int(y)) for x, y in zip(g, c, strict=True)],
            np.uint32,
        ),
    }


def replay(
    directory: str | Path, n: int, beta: Any, p: Any, u: int, *, probes=None, checkpoints=None
) -> mipt.EnsembleResult:
    """Trajectory ``u`` of condition ``(n, beta, p)`` simulated again from its seed (a one-row
    result; with other ``probes`` or ``checkpoints`` if given). Nothing is read but the spec
    and the graph file, and nothing is written."""
    root = Path(directory).expanduser().resolve()
    sweep, b, q, _ = _condition(root, n, beta, p)
    u = identity._integer(u, "u", minimum=0)
    g, c = divmod(u, sweep.stride)
    if g >= sweep.graphs[n] or c >= sweep.circuits_per_graph[n]:
        raise ValueError(f"u = {u} is not a trajectory of the sweep at n = {n}")
    mc = sweep.monitored_circuit(n, b, q, _collection(root, sweep, n, b))
    return mc.replay(u, seed=list(sweep.run_seed(n, b)), probes=probes, checkpoints=checkpoints)


def verify(directory: str | Path, n: int, beta: Any, p: Any, u: int) -> dict[str, Any]:
    """Rerun trajectory ``u`` (:func:`replay`) and compare it with its saved row: every probe
    value, snapshot word, purification and final layer and measurement count. Returns
    ``identical`` and the digests of the saved row's tableaux (``None`` after a stop)."""
    replayed = replay(directory, n, beta, p, u)
    folder, read = _reader(directory, n, beta, p)
    for path in _part_paths(folder):
        with np.load(path, allow_pickle=False) as z:
            if u not in z["trajectories"]:
                continue
        part = read(path)
        row = int(np.flatnonzero(part.trajectories == u)[0])
        break
    else:
        raise ValueError(f"trajectory u = {u} of n = {n}, beta = {beta}, p = {p} is not saved")
    arrays = [(part.values[k][row], replayed.values[k][0]) for k in part.values]
    arrays += [(part.final_layer[row], replayed.final_layer[0])]
    arrays += [(part.num_measurements[row], replayed.num_measurements[0])]
    if part.purification_layer is not None and replayed.purification_layer is not None:
        arrays += [(part.purification_layer[row], replayed.purification_layer[0])]
    for k, (words, counts) in part.snapshots.items():
        arrays += [
            (words[row], replayed.snapshots[k][0][0]),
            (counts[row], replayed.snapshots[k][1][0]),
        ]
    out: dict[str, Any] = {
        "n": n,
        "beta": str(Token.parse(beta)),
        "p": str(Token.parse(p)),
        "u": u,
        "g": u // part.spec.circuits_per_graph,
        "c": u % part.spec.circuits_per_graph,
        "identical": all(np.array_equal(a, b, equal_nan=True) for a, b in arrays),
    }
    names = [q.name for q in part.spec.probes if q.kind == "tableau_snapshot"]
    if names:
        counts = part.snapshots[names[0]][1][row]
        out["tableau_digests"] = [
            part.tableau(row, e).digest() if counts[e] >= 0 else None for e in range(counts.size)
        ]
    return out
