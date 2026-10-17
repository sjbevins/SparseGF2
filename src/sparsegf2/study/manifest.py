"""The experiment manifest of the Watts–Strogatz study (v1.4 §1, §8, §9; design §3).

One immutable manifest freezes every choice that makes the study reproducible: the conditions
$(n, \\beta, p)$, the depth $T_{\\max}$ and the checkpoint schedule of each, the trial
allocation, the seeds, the dynamics options, the observables to record and the analysis
rules. ``ExperimentManifest.from_json`` reads it strictly (unknown keys are refused with the
allowed ones listed; types and ranges are checked; the message names the field), and
``to_json`` writes its canonical JSON, whose SHA-256 is ``digest()``.

Sections (all keys are listed in ``docs/study/02-identity-seeds-and-manifest.md``):

* ``conditions``: blocks ``{"n": [...], "beta": [...], "p": [...]}``, each the product of its
  lists. $\\beta$ and $p$ are tokens (:class:`~sparsegf2.study.identity.Token`): decimal strings,
  rational strings or JSON numbers. Duplicates are refused.
* ``depth``: ``layers`` (``{"per_n": a}`` gives $T_{\\max} = a n$, ``{"fixed": T}``, or
  ``{"table": {"16": 128}}``) and ``checkpoints`` (``{"count": N_T}``, the geometric rule of
  v1.4 §9, or ``{"list": [...]}``). The sparse checkpoints $\\mathcal T(n, \\beta)$ always include
  $0, 1, T_{\\max}$; the global checkpoints of the engine are every layer.
* ``trials``: graphs $G_*$, circuits per graph $C$, the frozen circuit index stride $s \\ge C$
  and the measurement replicas $R = 1$ (design D1).
* ``seeds``: ``master_graph_seed``, ``master_circuit_seed``, ``gate_mode``.
* ``dynamics``, ``observables``, ``storage``, ``resources``, ``analysis``: typed settings with
  the defaults of the design example; a later task may add fields.

How the manifest drives the engine. ``monitored_circuit_kwargs(n, beta, p)`` is the exact
keyword set of ``mipt.MonitoredCircuit`` for one condition (v1.4 §8.1) except ``edges`` (the
graph collection of the work unit, built by :func:`~sparsegf2.study.identity.build_graph`) and
``probes`` (added by the runner). The runner builds the probes from the typed settings:

====================================  ==========================================================
manifest                              probe
====================================  ==========================================================
(always)                              ``code_dimension()`` at every layer
``observables.tableau_snapshots``     ``tableau_snapshot(checkpoints=checkpoints(n, beta))``
``observables.contiguous_distance``   ``contiguous_distance(checkpoints=checkpoints(n, beta))``
``uniform_erasure.samples``           ``erasure_first_failure(samples, checkpoints=...)``
``uniform_erasure.exact_budget``      ``uncorrectable_region_counts(budget=..., checkpoints=...)``
``correlated_erasure[i]``             ``correlated_erasure(couplings, sizes, samples, ...)``
====================================  ==========================================================

Revisions. A store keeps every manifest revision. :func:`check_extension` says whether a new
manifest only extends an old one (more conditions, rates, graphs, circuits, a larger $T_{\\max}$,
more checkpoints, more observables) or changes something that would make the recorded data
inconsistent (seeds, ``gate_mode``, the stride, the protocol, the settings of a recorded
observable).
"""

from __future__ import annotations

import json
import math
import numbers
import operator
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from sparsegf2 import erasures, mipt
from sparsegf2.study.identity import (
    GATE_MODES,
    SEED_LIMIT,
    Token,
    TrajectoryKey,
    canonical_json,
    check_experiment_id,
    graph_seed,
    run_seed,
    sha256_hex,
    stream_index,
)

__all__ = [
    "MANIFEST_SCHEMA",
    "Analysis",
    "CheckpointRule",
    "Condition",
    "ConditionBlock",
    "CorrelatedErasure",
    "Depth",
    "Dynamics",
    "ExperimentManifest",
    "Layers",
    "Observables",
    "Peak",
    "Resources",
    "Seeds",
    "Storage",
    "Trials",
    "UniformErasure",
    "check_extension",
    "geometric_checkpoints",
    "round_half_up_power",
]

#: The schema identifier written in every canonical manifest.
MANIFEST_SCHEMA = "sparsegf2.study.manifest/1"

#: The two labels that keep exploratory and confirmatory runs apart (v1.4 §9, "Rate grid").
LABELS = ("exploratory", "confirmatory")

#: Most checkpoints the geometric rule is asked for (it is exact big-integer arithmetic).
MAX_CHECKPOINT_COUNT = 1000

_SIZE_RANGE = re.compile(r"\s*([0-9]+)\s*:\s*([0-9]+)\s*(?::\s*([0-9]+)\s*)?")


# ---------------------------------------------------------------------------
# strict readers
# ---------------------------------------------------------------------------


def _kind(value: Any) -> str:
    return "null" if value is None else type(value).__name__


def _object(
    value: Any, where: str, *, required: Sequence[str] = (), optional: Sequence[str] = ()
) -> dict[str, Any]:
    """``value`` as a dict whose keys are among ``required`` + ``optional`` (all of ``required``)."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: expected an object, got {_kind(value)}")
    allowed = sorted({*required, *optional})
    unknown = sorted((k for k in value if k not in allowed), key=str)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed keys are {allowed}")
    missing = [k for k in required if k not in value]
    if missing:
        raise ValueError(f"{where}: missing required key(s) {missing}")
    return dict(value)


def _int(value: Any, where: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{where}: must be an integer, got {value!r}")
    v = operator.index(value)
    if minimum is not None and v < minimum:
        raise ValueError(f"{where}: must be at least {minimum}, got {v}")
    if maximum is not None and v > maximum:
        raise ValueError(f"{where}: must be at most {maximum}, got {v}")
    return v


def _real(value: Any, where: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{where}: must be a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{where}: must be finite, got {v}")
    if minimum is not None and v < minimum:
        raise ValueError(f"{where}: must be at least {minimum:g}, got {v:g}")
    return v


def _flag(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{where}: must be true or false, got {value!r}")
    return value


def _choice(value: Any, where: str, choices: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{where}: must be one of {list(choices)}, got {value!r}")
    return value


def _list(value: Any, where: str, *, nonempty: bool = True) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{where}: expected a list, got {_kind(value)}")
    if nonempty and not value:
        raise ValueError(f"{where}: give at least one entry")
    return list(value)


def _token(value: Any, where: str) -> Token:
    try:
        return Token.parse(value, what=where)
    except ValueError as exc:
        text = str(exc)
        raise ValueError(text if text.startswith(where) else f"{where}: {text}") from None


# ---------------------------------------------------------------------------
# the checkpoint rule
# ---------------------------------------------------------------------------


def _half_up_root(s: int, q: int) -> int:
    r"""$\lfloor x + 1/2 \rfloor$ for $x = s^{1/q} / 2$, exactly, for integers $s \ge 1$, $q \ge 1$.

    With $r = \lfloor s^{1/q} \rfloor$ (the largest integer with $r^q \le s$) this is
    $\lfloor (r + 1) / 2 \rfloor$: write $2x = r + f$ with $0 \le f < 1$; then $(2x + 1) / 2$
    has the integer part $\lfloor (r + 1) / 2 \rfloor$ whatever the parity of $r$. A tie
    ($x$ a half-integer, $r$ odd and $f = 0$) rounds up.
    """
    r = int(2.0 ** (math.log2(s) / q))  # a float estimate, corrected exactly below
    while r**q > s:
        r -= 1
    while (r + 1) ** q <= s:
        r += 1
    return (r + 1) // 2


def round_half_up_power(t_max: int, j: int, q: int) -> int:
    r"""$\lfloor T^{j/q} + 1/2 \rfloor$ exactly: the nearest integer to $T_{\max}^{j/q}$, ties upward.

    With $a = T^{j/q}$ and $S = 2^q T^j$ we have $a = S^{1/q} / 2$, so the value is
    :func:`_half_up_root` of $S$ and $q$, in integer arithmetic: no floating-point error can
    move a value across a rounding boundary. (For integer $T \ge 1$ and $q \ge 1$ the number
    $a$ is never a half-integer, so ties do not actually arise; the rule is nevertheless
    exact.)
    """
    if t_max < 1 or q < 1 or not 0 <= j <= q:
        raise ValueError(f"need T >= 1, q >= 1 and 0 <= j <= q, got T={t_max}, j={j}, q={q}")
    return _half_up_root(2**q * t_max**j, q)


def geometric_checkpoints(t_max: int, count: int) -> tuple[int, ...]:
    """The sparse checkpoint schedule $\\mathcal T$ of v1.4 §9 for $T_{\\max}$ and $N_T$ = ``count``.

    For $N_T \\ge 2$: $a_j = \\exp(j \\ln T_{\\max} / (N_T - 1)) = T_{\\max}^{j/(N_T - 1)}$ for
    $j = 0, \\ldots, N_T - 1$, each rounded to the nearest integer with ties upward (computed
    exactly, :func:`round_half_up_power`), clamped to $[0, T_{\\max}]$; for $N_T = 1$ no
    geometric point. The result is the sorted union of these points and $\\{0, 1, T_{\\max}\\}$,
    without repeats.
    """
    if isinstance(t_max, bool) or not isinstance(t_max, numbers.Integral) or t_max < 1:
        raise ValueError(f"T_max must be an integer of at least 1, got {t_max!r}")
    if (
        isinstance(count, bool)
        or not isinstance(count, numbers.Integral)
        or not 1 <= count <= MAX_CHECKPOINT_COUNT
    ):
        raise ValueError(
            f"checkpoint count must be an integer in [1, {MAX_CHECKPOINT_COUNT}], got {count!r}"
        )
    t_max, count = operator.index(t_max), operator.index(count)
    points = {0, 1, t_max}
    if count >= 2:
        for j in range(count):
            a = round_half_up_power(t_max, j, count - 1)
            points.add(min(max(a, 0), t_max))
    return tuple(sorted(points))


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


class Condition(NamedTuple):
    """One resolved condition: a size, a rewiring probability and the rates run there."""

    n: int
    beta: Token
    rates: tuple[Token, ...]


@dataclass(frozen=True)
class ConditionBlock:
    """A block of the ``conditions`` list: the product of its sizes, $\\beta$ values and rates."""

    n: tuple[int, ...]
    beta: tuple[Token, ...]
    p: tuple[Token, ...]

    @classmethod
    def from_dict(cls, data: Any, where: str) -> ConditionBlock:
        d = _object(data, where, required=("n", "beta", "p"))
        sizes = [
            _int(v, f"{where}.n[{i}]", minimum=5) for i, v in enumerate(_list(d["n"], f"{where}.n"))
        ]
        betas = [
            _token(v, f"{where}.beta[{i}]") for i, v in enumerate(_list(d["beta"], f"{where}.beta"))
        ]
        rates = [_token(v, f"{where}.p[{i}]") for i, v in enumerate(_list(d["p"], f"{where}.p"))]
        groups: list[tuple[str, list[Any]]] = [("n", sizes), ("beta", betas), ("p", rates)]
        for name, values in groups:
            if len(set(values)) != len(values):
                twice = next(v for v in values if values.count(v) > 1)
                raise ValueError(f"{where}.{name}: {twice} appears twice")
        return cls(tuple(sorted(sizes)), tuple(sorted(betas)), tuple(sorted(rates)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": list(self.n),
            "beta": [t.token for t in self.beta],
            "p": [t.token for t in self.p],
        }


@dataclass(frozen=True)
class Layers:
    """How $T_{\\max}$ depends on $n$: ``per_n`` ($T_{\\max} = a n$), ``fixed`` or a ``table``."""

    kind: str
    value: int = 0
    table: tuple[tuple[int, int], ...] = ()

    @classmethod
    def from_dict(cls, data: Any, where: str) -> Layers:
        d = _object(data, where, optional=("per_n", "fixed", "table"))
        if len(d) != 1:
            raise ValueError(
                f"{where}: give exactly one of 'per_n', 'fixed' or 'table', got {sorted(d)}"
            )
        kind, raw = next(iter(d.items()))
        if kind == "table":
            if not isinstance(raw, Mapping) or not raw:
                raise ValueError(
                    f"{where}.table: expected an object {{'16': 128, ...}} with at least one size"
                )
            pairs = []
            for key, t in raw.items():
                if not isinstance(key, str) or not re.fullmatch(r"[0-9]+", key):
                    raise ValueError(
                        f"{where}.table: keys are sizes n as strings such as '16', got {key!r}"
                    )
                pairs.append((int(key), _int(t, f"{where}.table[{key}]", minimum=1)))
            return cls("table", table=tuple(sorted(pairs)))
        return cls(kind, value=_int(raw, f"{where}.{kind}", minimum=1))

    def t_max(self, n: int) -> int:
        if self.kind == "per_n":
            return self.value * n
        if self.kind == "fixed":
            return self.value
        for size, t in self.table:
            if size == n:
                return t
        raise ValueError(
            f"depth.layers.table has no entry for n = {n}; it covers {[s for s, _ in self.table]}"
        )

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "table":
            return {"table": {str(n): t for n, t in self.table}}
        return {self.kind: self.value}


@dataclass(frozen=True)
class CheckpointRule:
    """The sparse checkpoints: ``count`` (the geometric rule of v1.4 §9) or an explicit ``list``."""

    kind: str
    count: int = 0
    depths: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, data: Any, where: str) -> CheckpointRule:
        d = _object(data, where, optional=("count", "list"))
        if len(d) != 1:
            raise ValueError(f"{where}: give exactly one of 'count' or 'list', got {sorted(d)}")
        kind, raw = next(iter(d.items()))
        if kind == "count":
            return cls(
                "count", count=_int(raw, f"{where}.count", minimum=1, maximum=MAX_CHECKPOINT_COUNT)
            )
        depths = [
            _int(v, f"{where}.list[{i}]", minimum=0)
            for i, v in enumerate(_list(raw, f"{where}.list"))
        ]
        return cls("list", depths=tuple(sorted(set(depths))))

    def resolve(self, t_max: int) -> tuple[int, ...]:
        """The checkpoints for horizon ``t_max``: sorted, unique, containing $0, 1, T_{\\max}$."""
        if self.kind == "count":
            return geometric_checkpoints(t_max, self.count)
        return tuple(sorted({0, 1, t_max, *self.depths}))

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "count":
            return {"count": self.count}
        return {"list": list(self.depths)}


@dataclass(frozen=True)
class Depth:
    layers: Layers
    checkpoints: CheckpointRule

    @classmethod
    def from_dict(cls, data: Any, where: str) -> Depth:
        d = _object(data, where, required=("layers", "checkpoints"))
        return cls(
            Layers.from_dict(d["layers"], f"{where}.layers"),
            CheckpointRule.from_dict(d["checkpoints"], f"{where}.checkpoints"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"layers": self.layers.to_dict(), "checkpoints": self.checkpoints.to_dict()}


@dataclass(frozen=True)
class Trials:
    """Graphs $G_*$, circuits per graph $C$, the circuit index stride $s$ and replicas $R$."""

    graphs: int
    circuits_per_graph: int
    circuit_index_stride: int
    measurement_replicas: int = 1

    @classmethod
    def from_dict(cls, data: Any, where: str) -> Trials:
        d = _object(
            data,
            where,
            required=("graphs", "circuits_per_graph"),
            optional=("circuit_index_stride", "measurement_replicas"),
        )
        graphs_ = _int(d["graphs"], f"{where}.graphs", minimum=1)
        per_graph = _int(d["circuits_per_graph"], f"{where}.circuits_per_graph", minimum=1)
        stride = _int(
            d.get("circuit_index_stride", per_graph), f"{where}.circuit_index_stride", minimum=1
        )
        if stride < per_graph:
            raise ValueError(
                f"{where}.circuit_index_stride: must be at least circuits_per_graph "
                f"({per_graph}), got {stride}: trajectory u = g * stride + c needs c < stride"
            )
        replicas = _int(d.get("measurement_replicas", 1), f"{where}.measurement_replicas")
        if replicas != 1:
            raise ValueError(
                f"{where}.measurement_replicas: must be 1, got {replicas}: the native engine draws "
                "the gates and the measurement masks of a trajectory from one stream, so "
                "independent mask replicates on the same gates (r > 0) do not exist (design D1; "
                "v1.4 sections 7.2 and 8.1)"
            )
        return cls(graphs_, per_graph, stride, replicas)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graphs": self.graphs,
            "circuits_per_graph": self.circuits_per_graph,
            "circuit_index_stride": self.circuit_index_stride,
            "measurement_replicas": self.measurement_replicas,
        }


@dataclass(frozen=True)
class Seeds:
    """The master seeds (unsigned 32-bit) and how the circuit seed depends on the rate."""

    master_graph_seed: int
    master_circuit_seed: int
    gate_mode: str = "shared"

    @classmethod
    def from_dict(cls, data: Any, where: str) -> Seeds:
        d = _object(
            data,
            where,
            required=("master_graph_seed", "master_circuit_seed"),
            optional=("gate_mode",),
        )
        graph = _int(
            d["master_graph_seed"], f"{where}.master_graph_seed", minimum=0, maximum=SEED_LIMIT - 1
        )
        circuit = _int(
            d["master_circuit_seed"],
            f"{where}.master_circuit_seed",
            minimum=0,
            maximum=SEED_LIMIT - 1,
        )
        if graph == circuit:
            raise ValueError(
                f"{where}: master_graph_seed and master_circuit_seed must differ (both are {graph}): "
                "graph 0 is built from the seed (s, 0), which is the same SeedSequence as the plain "
                "circuit seed s"
            )
        return cls(
            graph, circuit, _choice(d.get("gate_mode", "shared"), f"{where}.gate_mode", GATE_MODES)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "master_graph_seed": self.master_graph_seed,
            "master_circuit_seed": self.master_circuit_seed,
            "gate_mode": self.gate_mode,
        }


@dataclass(frozen=True)
class Dynamics:
    stop_when_purified: bool = False

    @classmethod
    def from_dict(cls, data: Any, where: str) -> Dynamics:
        d = _object(data, where, optional=("stop_when_purified",))
        return cls(_flag(d.get("stop_when_purified", False), f"{where}.stop_when_purified"))

    def to_dict(self) -> dict[str, Any]:
        return {"stop_when_purified": self.stop_when_purified}


@dataclass(frozen=True)
class UniformErasure:
    """The uniform-erasure probes: ``samples`` first-failure orderings and an exact-count budget.

    ``exact_budget`` $B$ is the number of regions enumerated per evaluation
    (:func:`sparsegf2.mipt.uncorrectable_region_counts`, ``budget=B``); 0 switches the exact
    counts off, else $B \\ge n$ for every $n$ of the study.
    """

    samples: int = 200
    exact_budget: int = 5000

    @classmethod
    def from_dict(cls, data: Any, where: str) -> UniformErasure:
        d = _object(data, where, optional=("samples", "exact_budget"))
        return cls(
            _int(d.get("samples", 200), f"{where}.samples", minimum=1),
            _int(d.get("exact_budget", 5000), f"{where}.exact_budget", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"samples": self.samples, "exact_budget": self.exact_budget}


@dataclass(frozen=True)
class CorrelatedErasure:
    """One ``correlated_erasure`` probe (v1.4 §5.7): the arguments of ``mipt.correlated_erasure``.

    ``sizes`` is ``None`` (every $m = 1..n-1$), a list of sizes or a range string
    ``"a:b[:step]"`` (Python slice semantics). ``name`` is the probe name; with several
    entries the unnamed ones are ``correlated_erasure_0``, ``correlated_erasure_1``, ... (see
    :meth:`Observables.probe_names`).
    """

    couplings: tuple[float, ...]
    sizes: tuple[int, ...] | str | None = None
    samples: int = 100
    thin: int = 1
    geometry: str = "interaction"
    method: str = "auto"
    burn_in: int | None = None
    tempering: bool = True
    ladder: str | tuple[float, ...] = "auto"
    max_step: float | None = None
    name: str | None = None

    _KEYS = (
        "couplings",
        "sizes",
        "samples",
        "thin",
        "geometry",
        "method",
        "burn_in",
        "tempering",
        "ladder",
        "max_step",
        "name",
    )

    @classmethod
    def from_dict(cls, data: Any, where: str) -> CorrelatedErasure:
        d = _object(data, where, required=("couplings",), optional=cls._KEYS)
        couplings = tuple(
            _real(v, f"{where}.couplings[{i}]")
            for i, v in enumerate(_list(d["couplings"], f"{where}.couplings"))
        )
        sizes: tuple[int, ...] | str | None
        raw_sizes = d.get("sizes")
        if raw_sizes is None:
            sizes = None
        elif isinstance(raw_sizes, str):
            if not _SIZE_RANGE.fullmatch(raw_sizes):
                raise ValueError(
                    f"{where}.sizes: expected null, a list or a range such as '1:33', got {raw_sizes!r}"
                )
            sizes = raw_sizes.strip()
        else:
            sizes = tuple(
                _int(v, f"{where}.sizes[{i}]", minimum=1)
                for i, v in enumerate(_list(raw_sizes, f"{where}.sizes"))
            )
        ladder: str | tuple[float, ...]
        raw_ladder = d.get("ladder", "auto")
        if isinstance(raw_ladder, str):
            ladder = _choice(raw_ladder, f"{where}.ladder", ("auto", "given"))
        else:
            ladder = tuple(
                _real(v, f"{where}.ladder[{i}]")
                for i, v in enumerate(_list(raw_ladder, f"{where}.ladder"))
            )
        name = d.get("name")
        if name is not None:
            name = check_experiment_id(name, f"{where}.name")
        entry = cls(
            couplings=couplings,
            sizes=sizes,
            samples=_int(d.get("samples", 100), f"{where}.samples", minimum=1),
            thin=_int(d.get("thin", 1), f"{where}.thin", minimum=1),
            geometry=_choice(
                d.get("geometry", "interaction"), f"{where}.geometry", mipt.GEOMETRIES
            ),
            method=_choice(d.get("method", "auto"), f"{where}.method", erasures.METHODS),
            burn_in=None
            if d.get("burn_in") is None
            else _int(d["burn_in"], f"{where}.burn_in", minimum=0),
            tempering=_flag(d.get("tempering", True), f"{where}.tempering"),
            ladder=ladder,
            max_step=None
            if d.get("max_step") is None
            else _real(d["max_step"], f"{where}.max_step"),
            name=name,
        )
        try:  # the engine's own validation of couplings, sizes, ladder and step
            entry.probe()
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{where}: {exc}") from None
        return entry

    def expand_sizes(self, n: int) -> tuple[int, ...]:
        """The erasure sizes for ``n`` qubits: ``None`` is $1..n-1$, a range string is expanded."""
        if self.sizes is None:
            return tuple(range(1, n))
        if isinstance(self.sizes, str):
            match = _SIZE_RANGE.fullmatch(self.sizes)
            assert match is not None, "validated at parse time"
            step = int(match.group(3) or 1)
            if step < 1:
                raise ValueError(f"sizes {self.sizes!r}: a range step must be positive")
            return tuple(range(int(match.group(1)), int(match.group(2)), step))
        return self.sizes

    def probe(self, *, name: str | None = None, checkpoints: Any = None) -> mipt.Probe:
        """The ``mipt.correlated_erasure`` probe these settings describe (the runner's call)."""
        return mipt.correlated_erasure(
            list(self.couplings),
            None
            if self.sizes is None
            else (self.sizes if isinstance(self.sizes, str) else list(self.sizes)),
            self.samples,
            name or self.name,
            burn_in=self.burn_in,
            thin=self.thin,
            tempering=self.tempering,
            method=self.method,
            ladder=self.ladder if isinstance(self.ladder, str) else list(self.ladder),
            max_step=self.max_step,
            geometry=self.geometry,
            checkpoints=checkpoints,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "couplings": list(self.couplings),
            "sizes": list(self.sizes) if isinstance(self.sizes, tuple) else self.sizes,
            "samples": self.samples,
            "thin": self.thin,
            "geometry": self.geometry,
            "method": self.method,
            "burn_in": self.burn_in,
            "tempering": self.tempering,
            "ladder": list(self.ladder) if isinstance(self.ladder, tuple) else self.ladder,
            "max_step": self.max_step,
            "name": self.name,
        }


@dataclass(frozen=True)
class Observables:
    """Which observables the runner records and with which settings (v1.4 §5, design §5)."""

    tableau_snapshots: bool = True
    contiguous_distance: bool = True
    uniform_erasure: UniformErasure | None = field(default_factory=UniformErasure)
    correlated_erasure: tuple[CorrelatedErasure, ...] = ()

    @classmethod
    def from_dict(cls, data: Any, where: str) -> Observables:
        d = _object(
            data,
            where,
            optional=(
                "tableau_snapshots",
                "contiguous_distance",
                "uniform_erasure",
                "correlated_erasure",
            ),
        )
        uniform: UniformErasure | None
        if "uniform_erasure" in d:
            uniform = (
                None
                if d["uniform_erasure"] is None
                else UniformErasure.from_dict(d["uniform_erasure"], f"{where}.uniform_erasure")
            )
        else:
            uniform = UniformErasure()
        entries = tuple(
            CorrelatedErasure.from_dict(v, f"{where}.correlated_erasure[{i}]")
            for i, v in enumerate(
                _list(
                    d.get("correlated_erasure", []), f"{where}.correlated_erasure", nonempty=False
                )
            )
        )
        observables = cls(
            tableau_snapshots=_flag(d.get("tableau_snapshots", True), f"{where}.tableau_snapshots"),
            contiguous_distance=_flag(
                d.get("contiguous_distance", True), f"{where}.contiguous_distance"
            ),
            uniform_erasure=uniform,
            correlated_erasure=entries,
        )
        names = observables.probe_names()
        if len(set(names)) != len(names):
            twice = next(v for v in names if names.count(v) > 1)
            raise ValueError(
                f"{where}.correlated_erasure: the probe name {twice!r} is used twice; give each entry its own 'name'"
            )
        return observables

    def probe_names(self) -> tuple[str, ...]:
        """The names of the ``correlated_erasure`` probes: ``name``, else ``correlated_erasure``
        for a single entry or ``correlated_erasure_<i>`` among several."""
        count = len(self.correlated_erasure)
        return tuple(
            e.name or ("correlated_erasure" if count == 1 else f"correlated_erasure_{i}")
            for i, e in enumerate(self.correlated_erasure)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tableau_snapshots": self.tableau_snapshots,
            "contiguous_distance": self.contiguous_distance,
            "uniform_erasure": None
            if self.uniform_erasure is None
            else self.uniform_erasure.to_dict(),
            "correlated_erasure": [e.to_dict() for e in self.correlated_erasure],
        }


@dataclass(frozen=True)
class Storage:
    graphs_per_unit: int = 8
    compression_level: int = 6

    @classmethod
    def from_dict(cls, data: Any, where: str) -> Storage:
        d = _object(data, where, optional=("graphs_per_unit", "compression_level"))
        return cls(
            _int(d.get("graphs_per_unit", 8), f"{where}.graphs_per_unit", minimum=1),
            _int(d.get("compression_level", 6), f"{where}.compression_level", minimum=0, maximum=9),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "graphs_per_unit": self.graphs_per_unit,
            "compression_level": self.compression_level,
        }


@dataclass(frozen=True)
class Resources:
    num_threads: int | None = None
    max_unit_trajectories: int = 4096

    @classmethod
    def from_dict(cls, data: Any, where: str) -> Resources:
        d = _object(data, where, optional=("num_threads", "max_unit_trajectories"))
        threads = d.get("num_threads")
        return cls(
            None if threads is None else _int(threads, f"{where}.num_threads", minimum=1),
            _int(d.get("max_unit_trajectories", 4096), f"{where}.max_unit_trajectories", minimum=1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_threads": self.num_threads,
            "max_unit_trajectories": self.max_unit_trajectories,
        }


@dataclass(frozen=True)
class Peak:
    """The predeclared peak rule (v1.4 §5.8): plateau width $z$, its sensitivity grid, bootstrap size."""

    rule: str = "plateau"
    z: float = 1.0
    sensitivity_z: tuple[float, ...] = (0.0, 1.0, 2.0)
    bootstrap: int = 200

    @classmethod
    def from_dict(cls, data: Any, where: str) -> Peak:
        d = _object(data, where, optional=("rule", "z", "sensitivity_z", "bootstrap"))
        sensitivity = tuple(
            _real(v, f"{where}.sensitivity_z[{i}]", minimum=0.0)
            for i, v in enumerate(
                _list(d.get("sensitivity_z", [0, 1, 2]), f"{where}.sensitivity_z")
            )
        )
        return cls(
            _choice(d.get("rule", "plateau"), f"{where}.rule", ("plateau",)),
            _real(d.get("z", 1.0), f"{where}.z", minimum=0.0),
            sensitivity,
            _int(d.get("bootstrap", 200), f"{where}.bootstrap", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "z": self.z,
            "sensitivity_z": list(self.sensitivity_z),
            "bootstrap": self.bootstrap,
        }


@dataclass(frozen=True)
class Analysis:
    peak: Peak = field(default_factory=Peak)
    bootstrap: int = 1000
    censoring: str = "survival"

    @classmethod
    def from_dict(cls, data: Any, where: str) -> Analysis:
        d = _object(data, where, optional=("peak", "bootstrap", "censoring"))
        return cls(
            Peak.from_dict(d.get("peak", {}), f"{where}.peak"),
            _int(d.get("bootstrap", 1000), f"{where}.bootstrap", minimum=0),
            _choice(d.get("censoring", "survival"), f"{where}.censoring", ("survival",)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak": self.peak.to_dict(),
            "bootstrap": self.bootstrap,
            "censoring": self.censoring,
        }


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------


def _canonical_blocks(conditions: list[Condition]) -> tuple[ConditionBlock, ...]:
    r"""The canonical factorization of the resolved conditions into blocks (a function of the set
    of $(n, \beta, p)$ alone, so equivalent manifests have one digest).

    Group the pairs $(n, \beta)$ by their rate set $P$; within one rate set, group the sizes $n$
    by their set of $\beta$ values; each group is one block $\{n\} \times B \times P$. A
    manifest whose blocks are already a product (the design example) keeps its one block.
    """
    by_rates: dict[tuple[Token, ...], dict[int, list[Token]]] = {}
    for c in conditions:
        by_rates.setdefault(c.rates, {}).setdefault(c.n, []).append(c.beta)
    blocks = []
    for rates, per_size in by_rates.items():
        by_betas: dict[tuple[Token, ...], list[int]] = {}
        for n, beta_list in per_size.items():
            by_betas.setdefault(tuple(sorted(beta_list)), []).append(n)
        for beta_set, size_list in by_betas.items():
            blocks.append(ConditionBlock(tuple(sorted(size_list)), beta_set, rates))
    return tuple(sorted(blocks, key=lambda b: canonical_json(b.to_dict())))


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"the JSON object has the key {key!r} twice")
        out[key] = value
    return out


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"JSON has no {name}; use a finite number")


@dataclass(frozen=True)
class ExperimentManifest:
    """The immutable description of one study (see the module docstring).

    Build it with :meth:`from_json` (a dict, JSON text or a ``Path``). ``to_json()`` is the
    canonical text and ``digest()`` its SHA-256. The constructor validates the cross-field
    rules: no duplicate condition, a layer table that covers exactly the sizes in use,
    explicit checkpoints within $[0, T_{\\max}]$ for every size, correlated-erasure sizes
    within $[1, n - 1]$, an exact-count budget of at least $n$, and different master seeds.
    """

    experiment_id: str
    blocks: tuple[ConditionBlock, ...]
    depth: Depth
    trials: Trials
    seeds: Seeds
    label: str = "exploratory"
    dynamics: Dynamics = field(default_factory=Dynamics)
    observables: Observables = field(default_factory=Observables)
    storage: Storage = field(default_factory=Storage)
    resources: Resources = field(default_factory=Resources)
    analysis: Analysis = field(default_factory=Analysis)
    schema: str = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MANIFEST_SCHEMA:
            raise ValueError(
                f"manifest.schema: this version reads {MANIFEST_SCHEMA!r}, got {self.schema!r}"
            )
        check_experiment_id(self.experiment_id, "manifest.experiment_id")
        _choice(self.label, "manifest.label", LABELS)
        conditions = self.conditions()  # refuses duplicates
        object.__setattr__(self, "blocks", _canonical_blocks(conditions))
        sizes = sorted({c.n for c in conditions})
        if self.depth.layers.kind == "table":
            listed = [n for n, _ in self.depth.layers.table]
            missing = [n for n in sizes if n not in listed]
            extra = [n for n in listed if n not in sizes]
            if missing:
                raise ValueError(f"manifest.depth.layers.table: no entry for n = {missing}")
            if extra:
                raise ValueError(
                    f"manifest.depth.layers.table: n = {extra} is not in any condition"
                )
        rule = self.depth.checkpoints
        for n in sizes:
            t_max = self.depth.layers.t_max(n)
            if rule.kind == "list" and rule.depths and rule.depths[-1] > t_max:
                raise ValueError(
                    f"manifest.depth.checkpoints.list: checkpoint {rule.depths[-1]} exceeds "
                    f"T_max = {t_max} for n = {n}"
                )
            for entry in self.observables.correlated_erasure:
                bad = [m for m in entry.expand_sizes(n) if not 1 <= m <= n - 1]
                if bad or not entry.expand_sizes(n):
                    raise ValueError(
                        f"manifest.observables.correlated_erasure: sizes must lie in [1, {n - 1}] "
                        f"for n = {n}, got {bad[0] if bad else 'none'}"
                    )
        uniform = self.observables.uniform_erasure
        if uniform is not None and uniform.exact_budget and uniform.exact_budget < max(sizes):
            raise ValueError(
                f"manifest.observables.uniform_erasure.exact_budget: {uniform.exact_budget} regions "
                f"cannot even hold the {max(sizes)} single-qubit erasures of n = {max(sizes)}: use "
                "at least n, or 0 for no exact counts"
            )

    # -- construction -------------------------------------------------------------------

    @classmethod
    def from_json(cls, source: Mapping[str, Any] | str | Path) -> ExperimentManifest:
        """Read a manifest from a dict, from JSON text or from a file (a ``pathlib.Path``).

        Unknown keys, duplicate keys, wrong types and out-of-range values are refused with a
        message that names the field (``manifest.trials.graphs: ...``).
        """
        if isinstance(source, Path):
            try:
                text = source.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"manifest: cannot read {source}: {exc.strerror or exc}") from None
            return cls.from_json(text)
        if isinstance(source, str):
            if not source.lstrip().startswith("{"):
                raise ValueError(
                    "manifest: a string is read as JSON text and must start with '{'; "
                    "pass a pathlib.Path to read a file"
                )
            try:
                data = json.loads(
                    source, object_pairs_hook=_no_duplicate_keys, parse_constant=_refuse_constant
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"manifest: not valid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno})"
                ) from None
            except ValueError as exc:
                raise ValueError(f"manifest: {exc}") from None
            return cls.from_dict(data)
        return cls.from_dict(source)

    @classmethod
    def from_dict(cls, data: Any) -> ExperimentManifest:
        """Like :meth:`from_json` for data already parsed into dicts and lists."""
        d = _object(
            data,
            "manifest",
            required=("experiment_id", "conditions", "depth", "trials", "seeds"),
            optional=(
                "schema",
                "label",
                "dynamics",
                "observables",
                "storage",
                "resources",
                "analysis",
            ),
        )
        schema = d.get("schema", MANIFEST_SCHEMA)
        if schema != MANIFEST_SCHEMA:
            raise ValueError(
                f"manifest.schema: this version reads {MANIFEST_SCHEMA!r}, got {schema!r}"
            )
        blocks = tuple(
            ConditionBlock.from_dict(b, f"manifest.conditions[{i}]")
            for i, b in enumerate(_list(d["conditions"], "manifest.conditions"))
        )
        return cls(
            experiment_id=check_experiment_id(d["experiment_id"], "manifest.experiment_id"),
            blocks=blocks,
            depth=Depth.from_dict(d["depth"], "manifest.depth"),
            trials=Trials.from_dict(d["trials"], "manifest.trials"),
            seeds=Seeds.from_dict(d["seeds"], "manifest.seeds"),
            label=_choice(d.get("label", "exploratory"), "manifest.label", LABELS),
            dynamics=Dynamics.from_dict(d.get("dynamics", {}), "manifest.dynamics"),
            observables=Observables.from_dict(d.get("observables", {}), "manifest.observables"),
            storage=Storage.from_dict(d.get("storage", {}), "manifest.storage"),
            resources=Resources.from_dict(d.get("resources", {}), "manifest.resources"),
            analysis=Analysis.from_dict(d.get("analysis", {}), "manifest.analysis"),
            schema=schema,
        )

    # -- the canonical form -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The manifest as plain JSON data with every default written out."""
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "label": self.label,
            "conditions": [b.to_dict() for b in self.blocks],
            "depth": self.depth.to_dict(),
            "trials": self.trials.to_dict(),
            "seeds": self.seeds.to_dict(),
            "dynamics": self.dynamics.to_dict(),
            "observables": self.observables.to_dict(),
            "storage": self.storage.to_dict(),
            "resources": self.resources.to_dict(),
            "analysis": self.analysis.to_dict(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """The canonical JSON text (sorted keys, no spaces); ``indent`` pretty-prints it instead."""
        if indent is None:
            return canonical_json(self.to_dict())
        return json.dumps(json.loads(canonical_json(self.to_dict())), sort_keys=True, indent=indent)

    def digest(self) -> str:
        """The SHA-256 of the canonical JSON text: the identity of this revision of the manifest."""
        return sha256_hex(self.to_json().encode("utf-8"))

    # -- conditions, depth, checkpoints -------------------------------------------------

    def conditions(self) -> list[Condition]:
        """The resolved conditions $(n, \\beta, [p, \\ldots])$ in a stable order.

        Sorted by $n$, then $\\beta$; the rates of each pair are the sorted union over the
        blocks. A repeated $(n, \\beta, p)$, and two tokens with the same float64, are refused:
        they would run identical trajectories under two names.
        """
        rates: dict[tuple[int, Token], dict[Token, None]] = {}
        floats: dict[tuple[int, float], Token] = {}
        for block in self.blocks:
            for n in block.n:
                for beta in block.beta:
                    seen = floats.setdefault((n, beta.float64), beta)
                    if seen != beta:
                        raise ValueError(
                            f"manifest.conditions: beta = {seen} and beta = {beta} are the same "
                            f"float64 ({beta.hex}) and would build identical graphs for n = {n}"
                        )
                    bucket = rates.setdefault((n, beta), {})
                    for p in block.p:
                        if p in bucket:
                            raise ValueError(
                                f"manifest.conditions: (n = {n}, beta = {beta}, p = {p}) appears twice"
                            )
                        bucket[p] = None
        out = []
        for (n, beta), bucket in sorted(rates.items()):
            by_float: dict[float, Token] = {}
            for p in bucket:
                other = by_float.setdefault(p.float64, p)
                if other != p:
                    raise ValueError(
                        f"manifest.conditions: p = {other} and p = {p} are the same float64 "
                        f"({p.hex}) and would run identical trajectories (n = {n}, beta = {beta})"
                    )
            out.append(Condition(n, beta, tuple(sorted(bucket))))
        return out

    def _condition(self, n: Any, beta: Any, p: Any = None) -> tuple[int, Token, Token | None]:
        n = _int(n, "n")
        beta_t = _token(beta, "beta")
        p_t = None if p is None else _token(p, "p")
        for c in self.conditions():
            if c.n == n and c.beta == beta_t and (p_t is None or p_t in c.rates):
                return n, beta_t, p_t
        what = f"n = {n}, beta = {beta_t}" + ("" if p_t is None else f", p = {p_t}")
        raise ValueError(f"({what}) is not a condition of experiment {self.experiment_id!r}")

    def rates(self, n: int, beta: Token | str | float) -> tuple[Token, ...]:
        """The sorted rates run at $(n, \\beta)$."""
        n, beta_t, _ = self._condition(n, beta)
        return next(c.rates for c in self.conditions() if c.n == n and c.beta == beta_t)

    def layers(self, n: int, beta: Token | str | float) -> int:
        """The horizon $T_{\\max}(n, \\beta)$ in layers (the same for every rate; v1.4 §1)."""
        n, _, _ = self._condition(n, beta)
        return self.depth.layers.t_max(n)

    def checkpoints(self, n: int, beta: Token | str | float) -> tuple[int, ...]:
        """The sparse checkpoints $\\mathcal T(n, \\beta)$: sorted, unique, containing $0, 1, T_{\\max}$.

        The geometric rule of v1.4 §9 (``geometric_checkpoints``) or the explicit list, which
        was checked at construction to lie in $[0, T_{\\max}]$. Expensive probes run here; the
        reference purification and the code dimension are observed at every layer.
        """
        return self.depth.checkpoints.resolve(self.layers(n, beta))

    # -- seeds and keys -----------------------------------------------------------------

    @property
    def stride(self) -> int:
        """The frozen circuit index stride $s$."""
        return self.trials.circuit_index_stride

    @property
    def trajectories_per_condition(self) -> int:
        """$G_* C$ trajectories at every $(n, \\beta, p)$."""
        return self.trials.graphs * self.trials.circuits_per_graph

    def graph_seed(self, g: int) -> tuple[int, int]:
        return graph_seed(self.seeds.master_graph_seed, g)

    def run_seed(self, p: Token | str | float) -> int | tuple[int, int]:
        return run_seed(self.seeds.master_circuit_seed, self.seeds.gate_mode, p)

    def stream_index(self, g: int, c: int) -> int:
        return stream_index(g, c, self.stride)

    def trajectory_key(
        self, n: int, beta: Token | str | float, p: Token | str | float, g: int, c: int
    ) -> TrajectoryKey:
        n, beta_t, p_t = self._condition(n, beta, p)
        assert p_t is not None
        return TrajectoryKey(self.experiment_id, n, beta_t, p_t, g, c)

    def monitored_circuit_kwargs(
        self, n: int, beta: Token | str | float, p: Token | str | float
    ) -> dict[str, Any]:
        """The ``mipt.MonitoredCircuit`` keywords of v1.4 §8.1 for the condition $(n, \\beta, p)$.

        Everything except ``edges`` (the graphs of the work unit) and ``probes`` (the runner's):
        ``num_system=n``, ``circuits_per_graph=stride``, ``layers=T_max``,
        ``measurement_rate=float(p)``, ``reference="purification"``,
        ``schedule="random_edges"``, ``gates_per_layer=n // 2``, ``random_orientation=False``,
        no scramble, ``periodic=True``, ``checkpoints="every_layer"`` and
        ``stop_when_purified`` from ``dynamics``. Trajectory $u = g s + c$ then runs on graph
        $g$ (``u // stride``).
        """
        n, beta_t, p_t = self._condition(n, beta, p)
        assert p_t is not None
        return {
            "num_system": n,
            "circuits_per_graph": self.stride,
            "layers": self.depth.layers.t_max(n),
            "measurement_rate": p_t.float64,
            "reference": "purification",
            "schedule": "random_edges",
            "gates_per_layer": n // 2,
            "random_orientation": False,
            "scramble_gates": 0,
            "scramble_matchings": 0,
            "scramble_layers": 0,
            "periodic": True,
            "checkpoints": "every_layer",
            "stop_when_purified": self.dynamics.stop_when_purified,
        }

    @property
    def warnings(self) -> tuple[str, ...]:
        """Things that are allowed but probably not intended (the manifest is still valid)."""
        out = []
        if any(c.n == 5 for c in self.conditions()):
            out.append(
                "n = 5 is the complete graph K_5: rewiring cannot change it (v1.4 section 9), so "
                "every beta gives the same graph there"
            )
        if self.observables.tableau_snapshots and self.dynamics.stop_when_purified:
            out.append(
                "tableau_snapshots with stop_when_purified = true: a trajectory stops when the "
                "reference purifies, and its later snapshots are recorded as absent"
            )
        unit = self.storage.graphs_per_unit * self.trials.circuits_per_graph
        if unit > self.resources.max_unit_trajectories:
            out.append(
                f"a work unit of graphs_per_unit x circuits_per_graph = {unit} trajectories is "
                f"above max_unit_trajectories = {self.resources.max_unit_trajectories}"
            )
        return tuple(out)


# ---------------------------------------------------------------------------
# revisions
# ---------------------------------------------------------------------------


def check_extension(old: ExperimentManifest, new: ExperimentManifest) -> str | None:
    """``None`` if ``new`` only extends ``old``, else a message naming the first forbidden change.

    Allowed (design §3, "Revisions"): more conditions and rates (every old $(n, \\beta, p)$
    stays), more graphs, more circuits per graph (up to the frozen stride), a larger
    $T_{\\max}$, more checkpoints (every old checkpoint stays), more observables (every old
    observable keeps its settings), and any change of ``storage``, ``resources`` and
    ``analysis`` (they do not touch recorded data). Forbidden: a change of ``schema``,
    ``experiment_id``, ``label``, the seeds, ``gate_mode``, the stride, ``dynamics`` or the
    number of measurement replicas, and any removal or change of what the old manifest
    recorded. The checks run in the order of this list.
    """
    if not isinstance(old, ExperimentManifest) or not isinstance(new, ExperimentManifest):
        raise TypeError("check_extension compares two ExperimentManifest objects")

    def changed(what: str, before: Any, after: Any, why: str) -> str:
        return f"{what} changed from {before!r} to {after!r}: {why}"

    for name in ("schema", "experiment_id", "label"):
        a, b = getattr(old, name), getattr(new, name)
        if a != b:
            return changed(
                name, a, b, "it identifies the experiment and cannot change in a revision"
            )
    for name in ("master_graph_seed", "master_circuit_seed", "gate_mode"):
        a, b = getattr(old.seeds, name), getattr(new.seeds, name)
        if a != b:
            return changed(
                f"seeds.{name}",
                a,
                b,
                "seeds are frozen: every recorded graph and trajectory depends on them",
            )
    if old.stride != new.stride:
        return changed(
            "trials.circuit_index_stride",
            old.stride,
            new.stride,
            "the stride fixes every trajectory index u = g * stride + c",
        )
    if old.dynamics != new.dynamics:
        return changed(
            "dynamics.stop_when_purified",
            old.dynamics.stop_when_purified,
            new.dynamics.stop_when_purified,
            "the dynamics are part of the protocol",
        )
    if old.trials.measurement_replicas != new.trials.measurement_replicas:
        return changed(
            "trials.measurement_replicas",
            old.trials.measurement_replicas,
            new.trials.measurement_replicas,
            "replicas are not available (design D1)",
        )
    if new.trials.graphs < old.trials.graphs:
        return changed(
            "trials.graphs",
            old.trials.graphs,
            new.trials.graphs,
            "a revision may add graphs, not remove them",
        )
    if new.trials.circuits_per_graph < old.trials.circuits_per_graph:
        return changed(
            "trials.circuits_per_graph",
            old.trials.circuits_per_graph,
            new.trials.circuits_per_graph,
            "a revision may add circuits, not remove them",
        )
    old_conditions = old.conditions()
    new_rates = {(c.n, c.beta): set(c.rates) for c in new.conditions()}
    for c in old_conditions:
        for p in c.rates:
            if p not in new_rates.get((c.n, c.beta), ()):
                return (
                    f"condition (n = {c.n}, beta = {c.beta}, p = {p}) of the old manifest is missing "
                    "from the new one: a revision may add conditions and rates, not remove them"
                )
    for c in old_conditions:
        before, after = old.layers(c.n, c.beta), new.layers(c.n, c.beta)
        if after < before:
            return changed(
                f"T_max for n = {c.n}", before, after, "a revision may raise T_max, not lower it"
            )
    for c in old_conditions:
        lost = sorted(set(old.checkpoints(c.n, c.beta)) - set(new.checkpoints(c.n, c.beta)))
        if lost:
            return (
                f"the checkpoints of (n = {c.n}, beta = {c.beta}) lose the depths {lost}: a revision "
                "may only add checkpoints (when T_max grows, give an explicit list that keeps the "
                "old depths)"
            )
    a, b = old.observables, new.observables
    for name in ("tableau_snapshots", "contiguous_distance"):
        if getattr(a, name) and not getattr(b, name):
            return f"observables.{name} was recorded and cannot be switched off in a revision"
    if a.uniform_erasure is not None and a.uniform_erasure != b.uniform_erasure:
        return changed(
            "observables.uniform_erasure",
            a.uniform_erasure.to_dict(),
            None if b.uniform_erasure is None else b.uniform_erasure.to_dict(),
            "the settings of a recorded observable are frozen (add a new observable instead)",
        )
    new_entries = {canonical_json(e.to_dict()) for e in b.correlated_erasure}
    for i, entry in enumerate(a.correlated_erasure):
        if canonical_json(entry.to_dict()) not in new_entries:
            return (
                f"observables.correlated_erasure[{i}] ({a.probe_names()[i]!r}) was recorded and is "
                "missing or changed in the new manifest: the settings of a recorded observable are "
                "frozen (add a new entry with another name instead)"
            )
    return None
