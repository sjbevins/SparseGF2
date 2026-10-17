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
  v1.4 §9; ``{"list": [...]}``, one list for every size, whose entries above a size's
  $T_{\\max}$ are dropped for that size; or ``{"table": {"16": [...]}}``, one list per size,
  every size covered, every entry within $[0, T_{\\max}]$). The sparse checkpoints
  $S(n, \\beta)$ of a revision always include $0, 1, T_{\\max}$; the global checkpoints of the
  engine are every layer.
* ``trials``: graphs $N_{\\mathrm{graphs}}$, circuits per graph $C$, the frozen circuit index
  stride $\\bar C \\ge C$ and the measurement replicas $R = 1$ (design D1).
* ``seeds``: ``master_graph_seed``, ``master_circuit_seed``, ``gate_mode``. The seeds of
  graphs and runs are domain-tagged tuples that name the condition (ruling R-SEED,
  :func:`~sparsegf2.study.identity.graph_seed` and ``run_seed``), so the two master seeds may
  be equal, and different sizes and rewiring probabilities get independent samples; the
  manifest methods ``graph_seed(n, beta, g)`` and ``run_seed(n, beta, p)`` give them.
* ``dynamics``, ``observables``, ``storage``, ``resources``, ``analysis``: typed settings with
  the defaults of the design example; a later task may add fields.
* ``protocols`` (design §12, optional): ``["F"]`` by default; ``["F", "P"]`` or
  ``{"F": {}, "P": {"checkpoints": {...}}}`` adds the pure-product-state protocol P with its
  own checkpoint rule (:class:`ProtocolSettings`). It is written in the canonical JSON only when
  it is not the default, as is ``resources.max_unit_bytes``, so the digests of manifests
  without them did not change when they were added.

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
``correlated_erasure``                ``observables.correlated_probes(checkpoints=...)``
====================================  ==========================================================

``checkpoints(n, beta)`` in the table stands for the schedule used for work: while there is
one revision it is :meth:`ExperimentManifest.checkpoints`, and across revisions it is
:func:`effective_checkpoints` (below). The probes of the first five rows keep the names in
:data:`RESERVED_PROBE_NAMES`, which no ``correlated_erasure`` entry may take. Each correlated
entry's probe is named by :meth:`Observables.probe_names` (its ``name``, or a default fixed by
its position that appending entries never changes); the name keys the probe's random stream
and its stored values, so a revision keeps the name of every recorded probe.

Revisions. A store keeps every manifest revision. :func:`check_extension` says whether a new
manifest only extends an old one (more conditions, rates, graphs, circuits, a larger $T_{\\max}$,
any other checkpoint rule, more observables) or changes something that would make the recorded
data inconsistent (seeds, ``gate_mode``, the stride, the protocol, a lower $T_{\\max}$, the
settings or the name of a recorded observable). A manifest labeled ``"confirmatory"`` also
freezes its conditions and its ``analysis`` rules (v1.4 §5.8, §9: the rate grid is frozen and
the peak rule predeclared before confirmatory runs). Ruling R-CK (design §3): revision $r$ has its own schedule
$S_r(n, \\beta)$ from its own rule and $T_{\\max}$, and the schedule used for work under it is the
union $E_r = (S_1 \\cup \\dots \\cup S_r) \\cap [0, T_{\\max,r}]$, computed by
:func:`effective_checkpoints`. Because $T_{\\max}$ never decreases along a valid chain,
$E_r \\supseteq E_{r-1}$: a stored depth is never dropped.
"""

from __future__ import annotations

import json
import math
import numbers
import operator
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from sparsegf2 import erasures, mipt
from sparsegf2.study.identity import (
    GATE_MODES,
    PROTOCOLS,
    SEED_LIMIT,
    Token,
    TrajectoryKey,
    canonical_json,
    check_experiment_id,
    check_protocol,
    graph_seed,
    run_seed,
    sha256_hex,
    stream_index,
    token_words,
)

__all__ = [
    "MANIFEST_SCHEMA",
    "RESERVED_PROBE_NAMES",
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
    "ProtocolSettings",
    "Resources",
    "Seeds",
    "Storage",
    "Trials",
    "UniformErasure",
    "check_extension",
    "effective_checkpoints",
    "geometric_checkpoints",
    "round_half_up_power",
]

#: The schema identifier written in every canonical manifest.
MANIFEST_SCHEMA = "sparsegf2.study.manifest/1"

#: The two labels that keep exploratory and confirmatory runs apart (v1.4 §9, "Rate grid").
LABELS = ("exploratory", "confirmatory")

#: Most checkpoints the geometric rule is asked for (it is exact big-integer arithmetic).
MAX_CHECKPOINT_COUNT = 1000

#: The names of the probes the runner adds beside the ``correlated_erasure`` entries (the table
#: of the module docstring; ``code_snapshot`` replaces ``tableau_snapshot`` when snapshots are
#: off). A ``correlated_erasure`` entry may not take one of them: the engine refuses two probes
#: with one name, and only when a work unit runs.
RESERVED_PROBE_NAMES = (
    "code_dimension",
    "code_snapshot",
    "contiguous_distance",
    "erasure_first_failure",
    "tableau_snapshot",
    "uncorrectable_region_counts",
)

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
            _no_repeated_size([n for n, _ in pairs], f"{where}.table")
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


def _no_repeated_size(sizes: Sequence[int], where: str) -> None:
    """Refuse a table whose keys name one size twice (``"16"`` and ``"016"``)."""
    twice = next((n for n in sizes if list(sizes).count(n) > 1), None)
    if twice is not None:
        raise ValueError(f"{where}: the size n = {twice} appears twice")


@dataclass(frozen=True)
class CheckpointRule:
    r"""The sparse checkpoints $S(n,\beta)$ of one revision (v1.4 §9; design §3, ruling R-CK).

    * ``count``: the geometric rule with ``count`` $= N_T$ points (:func:`geometric_checkpoints`);
    * ``list``: one list of depths for every size. For a size with horizon $T_{\max}$ the
      entries above $T_{\max}$ are **dropped** (not an error), and $0, 1, T_{\max}$ are added;
    * ``table``: one list per size, every size of the conditions covered (the manifest refuses a
      missing or an extra size); each entry must lie in $[0, T_{\max}(n)]$, and $0, 1, T_{\max}$
      are added.
    """

    kind: str
    count: int = 0
    depths: tuple[int, ...] = ()
    table: tuple[tuple[int, tuple[int, ...]], ...] = ()

    @classmethod
    def from_dict(cls, data: Any, where: str) -> CheckpointRule:
        d = _object(data, where, optional=("count", "list", "table"))
        if len(d) != 1:
            raise ValueError(
                f"{where}: give exactly one of 'count', 'list' or 'table', got {sorted(d)}"
            )
        kind, raw = next(iter(d.items()))
        if kind == "count":
            return cls(
                "count", count=_int(raw, f"{where}.count", minimum=1, maximum=MAX_CHECKPOINT_COUNT)
            )
        if kind == "list":
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and not raw:
                raise ValueError(
                    f"{where}.list: give at least one entry (for just 0, 1 and T_max use "
                    "{'count': 1})"
                )
            depths = [
                _int(v, f"{where}.list[{i}]", minimum=0)
                for i, v in enumerate(_list(raw, f"{where}.list"))
            ]
            return cls("list", depths=tuple(sorted(set(depths))))
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError(
                f"{where}.table: expected an object {{'16': [1, 4, 16], ...}} with one list per size"
            )
        entries = []
        for key, values in raw.items():
            if not isinstance(key, str) or not re.fullmatch(r"[0-9]+", key):
                raise ValueError(
                    f"{where}.table: keys are sizes n as strings such as '16', got {key!r}"
                )
            depths = [
                _int(v, f"{where}.table[{key}][{i}]", minimum=0)
                for i, v in enumerate(_list(values, f"{where}.table[{key}]", nonempty=False))
            ]
            entries.append((int(key), tuple(sorted(set(depths)))))
        _no_repeated_size([n for n, _ in entries], f"{where}.table")
        return cls("table", table=tuple(sorted(entries)))

    def resolve(self, t_max: int, n: int) -> tuple[int, ...]:
        r"""The schedule $S(n,\beta)$ for horizon ``t_max`` at size ``n``: sorted, unique,
        containing $0, 1, T_{\max}$, within $[0, T_{\max}]$.

        ``count``: the geometric schedule. ``list``: the entries not above ``t_max``, plus
        $0, 1, T_{\max}$. ``table``: the entries of size ``n`` (each must be at most ``t_max``).
        """
        if self.kind == "count":
            return geometric_checkpoints(t_max, self.count)
        if self.kind == "list":
            return tuple(sorted({0, 1, t_max, *(d for d in self.depths if d <= t_max)}))
        for size, depths in self.table:
            if size == n:
                if depths and depths[-1] > t_max:
                    raise ValueError(
                        f"depth.checkpoints.table[{n}]: checkpoint {depths[-1]} exceeds "
                        f"T_max = {t_max} for n = {n}"
                    )
                return tuple(sorted({0, 1, t_max, *depths}))
        raise ValueError(
            f"depth.checkpoints.table has no entry for n = {n}; it covers "
            f"{[s for s, _ in self.table]}"
        )

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "count":
            return {"count": self.count}
        if self.kind == "list":
            return {"list": list(self.depths)}
        return {"table": {str(n): list(depths) for n, depths in self.table}}


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
    """Graphs $N_{\\mathrm{graphs}}$, circuits per graph $C$, the stride $\\bar C$ and replicas $R$."""

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
    """The master seeds (unsigned 32-bit) and how the run seed depends on the rate.

    Graph seeds and run seeds are tuples with different domain tags, so the two master seeds
    may be equal (ruling R-SEED; :func:`~sparsegf2.study.identity.graph_seed`,
    :func:`~sparsegf2.study.identity.run_seed`).
    """

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

    ``couplings`` are $J \\ge 0$: v1.4 §5.7 defines the region law
    $Q_{G,J}(A \\mid m) \\propto e^{-J\\,\\lvert\\partial A\\rvert}$ for $J \\ge 0$ only (the engine
    also samples $J < 0$, which the study does not define). ``sizes`` is ``None`` (every
    $m = 1..n-1$) or the tuple of sizes in the order given; :meth:`from_dict` also reads a range
    string ``"a:b[:step]"`` (Python slice semantics) and stores its expansion, so every spelling
    of the same sizes has one canonical form and one digest (a range does not depend on $n$).
    ``name`` is the probe name, which may not be one of :data:`RESERVED_PROBE_NAMES`; an unnamed
    entry is named by its position (:meth:`Observables.probe_names`).
    """

    couplings: tuple[float, ...]
    sizes: tuple[int, ...] | None = None
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
        for i, j in enumerate(couplings):
            if j < 0:
                raise ValueError(
                    f"{where}.couplings[{i}]: must be at least 0, got {j:g}: v1.4 section 5.7 "
                    "defines the correlated erasure law Q_(G,J) for J >= 0 only"
                )
        sizes: tuple[int, ...] | None
        raw_sizes = d.get("sizes")
        if raw_sizes is None:
            sizes = None
        elif isinstance(raw_sizes, str):
            match = _SIZE_RANGE.fullmatch(raw_sizes)
            if not match:
                raise ValueError(
                    f"{where}.sizes: expected null, a list or a range such as '1:33', got {raw_sizes!r}"
                )
            step = int(match.group(3) or 1)
            if step < 1:
                raise ValueError(f"{where}.sizes: a range step must be positive, got {raw_sizes!r}")
            sizes = tuple(range(int(match.group(1)), int(match.group(2)), step))
            if not sizes:
                raise ValueError(
                    f"{where}.sizes: the range {raw_sizes!r} is empty: give at least one size "
                    "(or null for every size 1..n-1)"
                )
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
            if name in RESERVED_PROBE_NAMES:
                raise ValueError(
                    f"{where}.name: {name!r} is the name of a probe the runner adds itself "
                    f"(the reserved names are {list(RESERVED_PROBE_NAMES)}); choose another name"
                )
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
        """The erasure sizes for ``n`` qubits: ``None`` is $1..n-1$, else the stored sizes."""
        if self.sizes is None:
            return tuple(range(1, n))
        return tuple(self.sizes)

    def probe(self, *, name: str | None = None, checkpoints: Any = None) -> mipt.Probe:
        """The ``mipt.correlated_erasure`` probe these settings describe, named ``name``.

        Without ``name`` the probe takes the entry's own ``name``, or the engine default
        ``"correlated_erasure"``, which is the manifest's name only for the first entry: the
        runner builds the probes of a manifest with :meth:`Observables.correlated_probes`, which
        passes each entry's resolved name (:meth:`Observables.probe_names`).
        """
        return mipt.correlated_erasure(
            list(self.couplings),
            None if self.sizes is None else list(self.sizes),
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
            "sizes": None if self.sizes is None else list(self.sizes),
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

    @staticmethod
    def default_probe_name(position: int) -> str:
        """The name of an unnamed ``correlated_erasure`` entry at ``position``:
        ``correlated_erasure`` for the first (the engine's default name) and
        ``correlated_erasure_<i>`` for the entry at position $i \\ge 1$. It depends on the position
        alone, not on the number of entries, so appending entries renames nothing."""
        return "correlated_erasure" if position == 0 else f"correlated_erasure_{position}"

    def probe_names(self) -> tuple[str, ...]:
        """The resolved names of the ``correlated_erasure`` probes: each entry's ``name``, else
        :meth:`default_probe_name` of its position. The name keys the probe's random stream
        (``mipt``: the hash of the name) and its stored values, so it is part of what a revision
        must keep (:func:`check_extension`)."""
        return tuple(
            e.name or self.default_probe_name(i) for i, e in enumerate(self.correlated_erasure)
        )

    def correlated_probes(self, checkpoints: Any = None) -> tuple[mipt.Probe, ...]:
        """The engine probes of the ``correlated_erasure`` entries, each under its resolved name
        (:meth:`probe_names`) and with ``checkpoints`` (the runner's call)."""
        return tuple(
            entry.probe(name=name, checkpoints=checkpoints)
            for entry, name in zip(self.correlated_erasure, self.probe_names(), strict=True)
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
    """Threads and the size of a work unit (they never change a recorded value).

    ``max_unit_bytes`` bounds the memory of one engine call: the snapshot buffer and the probe
    values the engine holds until the call returns (the runner splits units to fit; ``None``
    is the runner's default, ``runner.DEFAULT_UNIT_BYTES``). It is written in the canonical
    JSON only when set, so manifests written before it existed keep their digests.
    """

    num_threads: int | None = None
    max_unit_trajectories: int = 4096
    max_unit_bytes: int | None = None

    @classmethod
    def from_dict(cls, data: Any, where: str) -> Resources:
        d = _object(
            data, where, optional=("num_threads", "max_unit_trajectories", "max_unit_bytes")
        )
        threads = d.get("num_threads")
        limit = d.get("max_unit_bytes")
        return cls(
            None if threads is None else _int(threads, f"{where}.num_threads", minimum=1),
            _int(d.get("max_unit_trajectories", 4096), f"{where}.max_unit_trajectories", minimum=1),
            None if limit is None else _int(limit, f"{where}.max_unit_bytes", minimum=1),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "num_threads": self.num_threads,
            "max_unit_trajectories": self.max_unit_trajectories,
        }
        if self.max_unit_bytes is not None:
            out["max_unit_bytes"] = self.max_unit_bytes
        return out


@dataclass(frozen=True)
class ProtocolSettings:
    """One protocol of the study (design §12) and its own settings.

    ``name`` is ``"F"`` (the full reference of v1.4: the manifest's ``depth.checkpoints``,
    ``dynamics`` and ``observables`` apply to it, so it takes no settings here) or ``"P"`` (the
    pure product state |0...0> of the system, ``reference="none"``: no reference, so no code
    probes and no ``stop_when_purified``; the runner records its joint tableau, the pure
    n-qubit state, at ``checkpoints``). ``checkpoints`` is P's own checkpoint rule (the forms of
    ``depth.checkpoints``), ``None`` meaning the manifest's rule. P runs the same graphs, run
    seeds, stream indices and horizon as F, so its gates and measurement masks are F's.
    """

    name: str
    checkpoints: CheckpointRule | None = None

    @classmethod
    def from_dict(cls, name: Any, data: Any, where: str) -> ProtocolSettings:
        check_protocol(name, f"{where} key")
        if name == "F":  # F takes its settings from the manifest's sections
            if not isinstance(data, Mapping) or data:
                raise ValueError(
                    f"{where}.F: protocol F takes no settings of its own (depth.checkpoints, "
                    "dynamics and observables apply to it); give {}"
                )
            return cls("F")
        d = _object(data, f"{where}.{name}", optional=("checkpoints",))
        rule = d.get("checkpoints")
        return cls(
            name,
            None if rule is None else CheckpointRule.from_dict(rule, f"{where}.{name}.checkpoints"),
        )

    def to_dict(self) -> dict[str, Any]:
        if self.name == "F":
            return {}
        return {"checkpoints": None if self.checkpoints is None else self.checkpoints.to_dict()}


def _protocols(value: Any, where: str) -> tuple[ProtocolSettings, ...]:
    """``protocols``: a list of names (``["F", "P"]``) or an object of settings (``{"F": {}, "P":
    {"checkpoints": {"count": 8}}}``); sorted by name, each once."""
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{where}: give at least one protocol")
        entries = [ProtocolSettings.from_dict(k, v, f"{where}") for k, v in value.items()]
    else:
        names = _list(value, where)
        entries = [ProtocolSettings.from_dict(v, {}, f"{where}[{i}]") for i, v in enumerate(names)]
        if len({e.name for e in entries}) != len(entries):
            twice = next(e.name for e in entries if names.count(e.name) > 1)
            raise ValueError(f"{where}: protocol {twice!r} appears twice")
    return tuple(sorted(entries, key=lambda e: PROTOCOLS.index(e.name)))


_DEFAULT_PROTOCOLS = (ProtocolSettings("F"),)


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


def _distinct_seed_words(tokens: Iterable[Token], what: str) -> None:
    """Refuse two different tokens whose seed words (:func:`token_words`) coincide.

    The words of $\\beta$ name the condition inside every graph and run seed, and those of $p$
    inside the run seeds of ``gate_mode="independent"`` and the region seeds of the sampled
    observables (``study/observables.py``, in both modes), so a collision of two 64-bit hashes
    would give two conditions one seed. It has probability $2^{-64}$ for a given pair; refusing
    it makes "different conditions have different seeds" a fact about every accepted manifest
    and not a likelihood.
    """
    seen: dict[tuple[int, int], Token] = {}
    for token in tokens:
        other = seen.setdefault(token_words(token), token)
        if other != token:
            raise ValueError(
                f"manifest.conditions: the {what} values {other} and {token} have the same "
                "64-bit seed hash, so their conditions would share seeds (probability 2**-64 for "
                "a given pair): use another value for one of them"
            )


def _reparsed(value: Any, cls: Any, where: str, *, compare: bool = True) -> Any:
    """``value``, a section of class ``cls``, read back by ``cls.from_dict`` from its own
    ``to_dict()``: every rule of strict reading (types, ranges, $n \\ge 5$, the stride, the
    replicas of design D1, seed ranges, ...) then also holds for a section built directly or
    with ``dataclasses.replace``. With ``compare`` the section must also be in the canonical
    form reading gives (sorted, deduplicated, defaults filled in), so that its canonical JSON
    reads back as the same manifest."""
    if not isinstance(value, cls):
        raise ValueError(f"{where}: expected a {cls.__name__}, got {_kind(value)}")
    try:
        data = value.to_dict()
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            f"{where}: a field of this {cls.__name__} has the wrong type ({exc})"
        ) from None
    again = cls.from_dict(data, where)
    if compare and again != value:
        raise ValueError(
            f"{where}: {value!r} is not in the canonical form that reading it gives ({again!r}); "
            "build the manifest with ExperimentManifest.from_json"
        )
    return again


def _check_sections(manifest: ExperimentManifest) -> None:
    """Apply the rules of each section's ``from_dict`` to the sections of ``manifest`` (the
    constructor's first step), and replace its blocks by their validated form."""
    blocks = manifest.blocks
    if isinstance(blocks, (str, bytes)) or not isinstance(blocks, Sequence) or not blocks:
        raise ValueError("manifest.conditions: give a non-empty tuple of ConditionBlock objects")
    object.__setattr__(
        manifest,
        "blocks",
        tuple(
            _reparsed(b, ConditionBlock, f"manifest.conditions[{i}]", compare=False)
            for i, b in enumerate(blocks)
        ),
    )
    sections: tuple[tuple[str, Any], ...] = (
        ("depth", Depth),
        ("trials", Trials),
        ("seeds", Seeds),
        ("dynamics", Dynamics),
        ("observables", Observables),
        ("storage", Storage),
        ("resources", Resources),
        ("analysis", Analysis),
    )
    for name, cls in sections:
        _reparsed(getattr(manifest, name), cls, f"manifest.{name}")
    protocols = manifest.protocols  # design §12: the names are checked by the constructor
    if isinstance(protocols, (str, bytes)) or not isinstance(protocols, Sequence):
        raise ValueError("manifest.protocols: give a tuple of ProtocolSettings objects")
    for i, settings in enumerate(protocols):
        if not isinstance(settings, ProtocolSettings):
            raise ValueError(
                f"manifest.protocols[{i}]: expected a ProtocolSettings, got {_kind(settings)}"
            )
        if settings.checkpoints is not None:  # a protocol's own checkpoint rule
            _reparsed(
                settings.checkpoints,
                CheckpointRule,
                f"manifest.protocols.{settings.name}.checkpoints",
            )


class _Resolved(NamedTuple):
    """The resolved conditions of a manifest and their index by $(n, \\beta)$ (computed once)."""

    conditions: tuple[Condition, ...]
    by_pair: dict[tuple[int, Token], Condition]


def _resolve_conditions(blocks: Sequence[ConditionBlock]) -> _Resolved:
    """The conditions of ``blocks``: sorted by $n$, then $\\beta$, the rates of each pair the
    sorted union over the blocks; a repeated $(n, \\beta, p)$ and two tokens with one float64
    are refused."""
    rates: dict[tuple[int, Token], dict[Token, None]] = {}
    floats: dict[tuple[int, float], Token] = {}
    for block in blocks:
        for n in block.n:
            for beta in block.beta:
                seen = floats.setdefault((n, beta.float64), beta)
                if seen != beta:
                    raise ValueError(
                        f"manifest.conditions: beta = {seen} and beta = {beta} are two tokens of "
                        f"one float64 ({beta.hex}), the value the generator receives: they would "
                        f"be one rewiring probability under two names at n = {n}; keep one of them"
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
                    f"manifest.conditions: p = {other} and p = {p} are two tokens of one float64 "
                    f"({p.hex}), the measurement threshold the engine receives: they would be one "
                    f"rate under two names (n = {n}, beta = {beta}), and in gate_mode 'shared' "
                    "the very same trajectories; keep one of them"
                )
        out.append(Condition(n, beta, tuple(sorted(bucket))))
    return _Resolved(tuple(out), {(c.n, c.beta): c for c in out})


@dataclass(frozen=True)
class ExperimentManifest:
    """The immutable description of one study (see the module docstring).

    Build it with :meth:`from_json` (a dict, JSON text or a ``Path``). ``to_json()`` is the
    canonical text and ``digest()`` its SHA-256. The constructor first reads every section back
    through its ``from_dict`` (so a manifest built directly or with ``dataclasses.replace`` obeys
    the same rules as one read from JSON, and its canonical JSON reads back as itself), then
    validates the cross-field rules: no duplicate condition, a layer table and a checkpoint
    table that cover exactly the sizes in use, table entries within $[0, T_{\\max}(n)]$,
    correlated-erasure sizes within $[1, n - 1]$, an exact-count budget of at least $n$, and
    tokens ($\\beta$ and $p$) whose 64-bit seed hashes are all different (so that different
    conditions have different seeds; see ``graph_seed``). The two master seeds may be equal:
    graph seeds and run seeds carry different domain tags. The resolved conditions are computed
    once and indexed by $(n, \\beta)$.
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
    protocols: tuple[ProtocolSettings, ...] = _DEFAULT_PROTOCOLS

    def __post_init__(self) -> None:
        _check_sections(self)  # the rules of from_dict, also for direct construction and replace
        if self.schema != MANIFEST_SCHEMA:
            raise ValueError(
                f"manifest.schema: this version reads {MANIFEST_SCHEMA!r}, got {self.schema!r}"
            )
        check_experiment_id(self.experiment_id, "manifest.experiment_id")
        _choice(self.label, "manifest.label", LABELS)
        names = [s.name for s in self.protocols]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"manifest.protocols: give each protocol once, got {names}")
        for name in names:
            check_protocol(name, "manifest.protocols")
        conditions = self.conditions()  # refuses duplicates
        object.__setattr__(self, "blocks", _canonical_blocks(conditions))
        _distinct_seed_words((c.beta for c in conditions), "beta")
        _distinct_seed_words((p for c in conditions for p in c.rates), "p")
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
        rules = [(self.depth.checkpoints, "manifest.depth.checkpoints")] + [
            (s.checkpoints, f"manifest.protocols.{s.name}.checkpoints")
            for s in self.protocols
            if s.checkpoints is not None
        ]
        for rule, where in rules:
            if rule.kind == "table":
                listed = [n for n, _ in rule.table]
                missing = [n for n in sizes if n not in listed]
                extra = [n for n in listed if n not in sizes]
                if missing:
                    raise ValueError(f"{where}.table: no entry for n = {missing}")
                if extra:
                    raise ValueError(f"{where}.table: n = {extra} is not in any condition")
        for n in sizes:
            t_max = self.depth.layers.t_max(n)
            for rule, where in rules:
                if rule.kind == "table":
                    depths = dict(rule.table)[n]
                    if depths and depths[-1] > t_max:
                        raise ValueError(
                            f"{where}.table[{n}]: checkpoint {depths[-1]} "
                            f"exceeds T_max = {t_max} for n = {n}"
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
        message that names the field (``manifest.trials.graphs: ...``). A file is read as UTF-8,
        with or without a byte-order mark (Windows PowerShell 5.1 writes one).
        """
        if isinstance(source, Path):
            try:
                text = source.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError) as exc:
                reason = exc.strerror if isinstance(exc, OSError) else None
                raise ValueError(f"manifest: cannot read {source}: {reason or exc}") from None
            if not text.lstrip().startswith("{"):
                start = text.lstrip()[:20]
                raise ValueError(
                    f"manifest: the file {source} does not contain a JSON object (it starts "
                    f"with {start!r}, not '{{')"
                )
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
                "protocols",
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
            protocols=_protocols(d.get("protocols", ["F"]), "manifest.protocols"),
        )

    # -- the canonical form -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The manifest as plain JSON data with every default written out.

        ``protocols`` and ``resources.max_unit_bytes`` were added after the first version of
        the schema: they are written only when they differ from their defaults (``["F"]``,
        unset), so every manifest written before them keeps its canonical text and digest.
        """
        out = {
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
        if self.protocols != _DEFAULT_PROTOCOLS:
            out["protocols"] = {s.name: s.to_dict() for s in self.protocols}
        return out

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
        blocks. A repeated $(n, \\beta, p)$ is refused, and so are two tokens of one float64:
        the engine receives only the float, so they would be one condition under two names (in
        ``gate_mode="shared"`` two rates of one float64 would even run the same trajectories).
        Computed once per manifest (a new list is returned).
        """
        return list(self._resolved().conditions)

    def _resolved(self) -> _Resolved:
        """The conditions and their index, resolved at the first call and kept (the manifest is
        frozen, so they never change; the cache is not a field and plays no part in equality)."""
        cached = self.__dict__.get("_resolved_conditions")
        if cached is None:
            cached = _resolve_conditions(self.blocks)
            object.__setattr__(self, "_resolved_conditions", cached)
        return cached

    def _condition(self, n: Any, beta: Any, p: Any = None) -> tuple[int, Token, Token | None]:
        n = _int(n, "n")
        beta_t = _token(beta, "beta")
        p_t = None if p is None else _token(p, "p")
        c = self._resolved().by_pair.get((n, beta_t))
        if c is not None and (p_t is None or p_t in c.rates):
            return n, beta_t, p_t
        what = f"n = {n}, beta = {beta_t}" + ("" if p_t is None else f", p = {p_t}")
        raise ValueError(f"({what}) is not a condition of experiment {self.experiment_id!r}")

    def rates(self, n: int, beta: Token | str | float) -> tuple[Token, ...]:
        """The sorted rates run at $(n, \\beta)$."""
        n, beta_t, _ = self._condition(n, beta)
        return self._resolved().by_pair[(n, beta_t)].rates

    def layers(self, n: int, beta: Token | str | float) -> int:
        """The horizon $T_{\\max}(n, \\beta)$ in layers (the same for every rate; v1.4 §1)."""
        n, _, _ = self._condition(n, beta)
        return self.depth.layers.t_max(n)

    def checkpoints(
        self, n: int, beta: Token | str | float, protocol: str = "F"
    ) -> tuple[int, ...]:
        """The schedule $S(n, \\beta)$ of **this revision**: sorted, unique, containing $0, 1, T_{\\max}$.

        The geometric rule of v1.4 §9 (``geometric_checkpoints``), the list (entries above this
        size's $T_{\\max}$ dropped) or the size's entry of the table (checked at construction to
        lie in $[0, T_{\\max}]$). The schedule used for work under a revision that follows
        earlier ones is the union of the schedules of all of them,
        :func:`effective_checkpoints`. Expensive probes run at those depths; the reference
        purification and the code dimension are observed at every layer. For ``protocol="P"``
        the rule is P's own (:class:`ProtocolSettings`), or the manifest's rule if P has none.
        """
        n, _, _ = self._condition(n, beta)
        rule = self.protocol_settings(protocol).checkpoints or self.depth.checkpoints
        return rule.resolve(self.depth.layers.t_max(n), n)

    @property
    def protocol_names(self) -> tuple[str, ...]:
        """The protocols of the study (design §12), in the order of :data:`PROTOCOLS`."""
        return tuple(s.name for s in self.protocols)

    def protocol_settings(self, protocol: str) -> ProtocolSettings:
        """The settings of ``protocol``; ``ValueError`` if the study does not run it."""
        check_protocol(protocol)
        for s in self.protocols:
            if s.name == protocol:
                return s
        raise ValueError(
            f"protocol {protocol!r} is not run by experiment {self.experiment_id!r} "
            f"(manifest.protocols is {list(self.protocol_names)})"
        )

    def has_condition(self, n: int, beta: Token | str | float) -> bool:
        """Whether $(n, \\beta)$ is a condition of this manifest (with at least one rate)."""
        n = _int(n, "n")
        beta_t = _token(beta, "beta")
        return (n, beta_t) in self._resolved().by_pair

    # -- seeds and keys -----------------------------------------------------------------

    @property
    def stride(self) -> int:
        """The frozen circuit index stride $\\bar C$ (trajectory $u = g \\bar C + c$)."""
        return self.trials.circuit_index_stride

    @property
    def trajectories_per_condition(self) -> int:
        """$N_{\\mathrm{graphs}}\\, C$ trajectories at every $(n, \\beta, p)$."""
        return self.trials.graphs * self.trials.circuits_per_graph

    def graph_seed(self, n: int, beta: Token | str | float, g: int) -> tuple[int, ...]:
        """The seed of graph $g$ of the condition $(n, \\beta)$: ``identity.graph_seed``."""
        n, beta_t, _ = self._condition(n, beta)
        return graph_seed(self.seeds.master_graph_seed, n, beta_t, g)

    def run_seed(
        self, n: int, beta: Token | str | float, p: Token | str | float
    ) -> tuple[int, ...]:
        """The seed of the runs at the condition $(n, \\beta, p)$ in this manifest's gate mode:
        ``identity.run_seed``. In ``"shared"`` mode it is the same for every $p$."""
        n, beta_t, p_t = self._condition(n, beta, p)
        return run_seed(self.seeds.master_circuit_seed, n, beta_t, self.seeds.gate_mode, p_t)

    def stream_index(self, g: int, c: int) -> int:
        return stream_index(g, c, self.stride)

    def trajectory_key(
        self,
        n: int,
        beta: Token | str | float,
        p: Token | str | float,
        g: int,
        c: int,
        protocol: str = "F",
    ) -> TrajectoryKey:
        n, beta_t, p_t = self._condition(n, beta, p)
        assert p_t is not None
        self.protocol_settings(protocol)
        return TrajectoryKey(self.experiment_id, n, beta_t, p_t, g, c, 0, protocol)

    def monitored_circuit_kwargs(
        self, n: int, beta: Token | str | float, p: Token | str | float, protocol: str = "F"
    ) -> dict[str, Any]:
        """The ``mipt.MonitoredCircuit`` keywords of v1.4 §8.1 for the condition $(n, \\beta, p)$.

        Everything except ``edges`` (the graphs of the work unit) and ``probes`` (the runner's):
        ``num_system=n``, ``circuits_per_graph=stride``, ``layers=T_max``,
        ``measurement_rate=float(p)``, ``reference="purification"``,
        ``schedule="random_edges"``, ``gates_per_layer=n // 2``, ``random_orientation=False``,
        no scramble, ``periodic=True``, ``checkpoints="every_layer"`` and
        ``stop_when_purified`` from ``dynamics``. Trajectory $u = g s + c$ then runs on graph
        $g$ (``u // stride``). For ``protocol="P"`` (design §12) the only differences are
        ``reference="none"`` (the system starts in |0...0>) and ``stop_when_purified=False``
        (there is no reference to purify): the engine draws nothing that depends on the
        reference in these two pictures, so P's gates and masks are F's.
        """
        n, beta_t, p_t = self._condition(n, beta, p)
        assert p_t is not None
        self.protocol_settings(protocol)
        full = protocol == "F"
        return {
            "num_system": n,
            "circuits_per_graph": self.stride,
            "layers": self.depth.layers.t_max(n),
            "measurement_rate": p_t.float64,
            "reference": "purification" if full else "none",
            "schedule": "random_edges",
            "gates_per_layer": n // 2,
            "random_orientation": False,
            "scramble_gates": 0,
            "scramble_matchings": 0,
            "scramble_layers": 0,
            "periodic": True,
            "checkpoints": "every_layer",
            "stop_when_purified": self.dynamics.stop_when_purified if full else False,
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
        rule = self.depth.checkpoints
        if rule.kind == "list":
            horizon = max(self.depth.layers.t_max(c.n) for c in self.conditions())
            useless = [d for d in rule.depths if d > horizon]
            if useless:
                out.append(
                    f"checkpoints.list entries {useless} exceed T_max for every size and are "
                    "dropped everywhere: they have no effect in this revision"
                )
        return tuple(out)


# ---------------------------------------------------------------------------
# revisions
# ---------------------------------------------------------------------------


def check_extension(old: ExperimentManifest, new: ExperimentManifest) -> str | None:
    """``None`` if ``new`` only extends ``old``, else a message naming the first forbidden change.

    Two rules sharpen the list of design §3 below. A ``correlated_erasure`` entry is identified by
    its resolved name (:meth:`Observables.probe_names`) together with its settings: the name keys
    the probe's random stream and its stored values (v1.4 §7.3: "preserve probe names and
    versioned settings"), so a revision that renames a recorded probe, or gives its name to other
    settings, is refused, whether it does so by reordering or by naming entries. And a manifest
    labeled ``"confirmatory"`` freezes its conditions and its ``analysis`` section: v1.4 §9
    freezes the rate grid $\\mathcal P(n, \\beta)$ before confirmatory runs and v1.4 §5.8
    predeclares the peak rule (to change them, start a new experiment).

    Allowed (design §3, "Revisions" and ruling R-CK): more conditions and rates (every old
    $(n, \\beta, p)$ stays), more graphs, more circuits per graph (up to the frozen stride), a
    larger $T_{\\max}$ (per size; a lower one is forbidden), **any** change of the checkpoint
    rule (the schedule used for work is the union of the schedules of all revisions,
    :func:`effective_checkpoints`, so no stored depth is dropped), more observables (every old
    observable keeps its settings), a new protocol (design §12) or a new checkpoint rule for
    protocol P, and any change of ``storage``, ``resources`` and ``analysis`` (they do not
    touch recorded data). Forbidden: a change of ``schema``, ``experiment_id``, ``label``, the
    seeds, ``gate_mode``, the stride, ``dynamics`` or the number of measurement replicas, the
    removal of a protocol, and any removal or change of what the old manifest recorded. The
    checks run in the order of this list.
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
    dropped = [name for name in old.protocol_names if name not in new.protocol_names]
    if dropped:
        return (
            f"protocol {dropped[0]!r} of the old manifest is missing from the new one: a revision "
            "may add a protocol (design section 12), not remove one"
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
    if old.label == "confirmatory":
        old_rates = {(c.n, c.beta): set(c.rates) for c in old_conditions}
        for c in new.conditions():
            for p in c.rates:
                if p not in old_rates.get((c.n, c.beta), ()):
                    return (
                        f"condition (n = {c.n}, beta = {c.beta}, p = {p}) is new, but the manifest "
                        "is labeled 'confirmatory': v1.4 section 9 freezes the rate grid P(n, beta) "
                        "before confirmatory runs (start a new experiment to change it)"
                    )
        if old.analysis != new.analysis:
            return changed(
                "analysis",
                old.analysis.to_dict(),
                new.analysis.to_dict(),
                "a confirmatory manifest freezes its analysis rules (v1.4 sections 5.8 and 9: the "
                "peak rule is predeclared)",
            )
    for c in old_conditions:
        before, after = old.layers(c.n, c.beta), new.layers(c.n, c.beta)
        if after < before:
            return changed(
                f"T_max for n = {c.n}", before, after, "a revision may raise T_max, not lower it"
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

    def settings(entry: CorrelatedErasure, name: str | None) -> str:
        return canonical_json({**entry.to_dict(), "name": name})

    new_named = dict(zip(b.probe_names(), b.correlated_erasure, strict=True))
    for i, (name, entry) in enumerate(zip(a.probe_names(), a.correlated_erasure, strict=True)):
        if name in new_named and settings(new_named[name], name) == settings(entry, name):
            continue
        moved = [m for m, e in new_named.items() if settings(e, None) == settings(entry, None)]
        if moved:
            detail = f"names these settings {moved[0]!r}"
        elif name in new_named:
            detail = "gives this name other settings"
        else:
            detail = "has no probe of this name"
        return (
            f"observables.correlated_erasure[{i}] ({name!r}) was recorded and the new manifest "
            f"{detail}: the settings and the name of a recorded probe are frozen (the name keys "
            "its random stream and its stored values; an unnamed entry is named by its position, "
            f"so give it 'name': {name!r} to keep its name; add other settings as a new entry)"
        )
    return None


def effective_checkpoints(
    revisions: Sequence[ExperimentManifest],
    n: int,
    beta: Token | str | float,
    protocol: str = "F",
) -> tuple[int, ...]:
    r"""The checkpoints used for work under the last of ``revisions`` at $(n, \beta)$ (ruling R-CK).

    ``revisions`` are the manifest revisions of one experiment, oldest first. Revision $i$ has
    its own schedule $S_i(n,\beta)$ (:meth:`ExperimentManifest.checkpoints`); the effective
    schedule of the last revision $r$ is
    $$E_r(n,\beta) = \big(S_1 \cup \cdots \cup S_r\big)(n,\beta) \cap [0, T_{\max,r}(n,\beta)],$$
    returned sorted. A revision that does not contain $(n,\beta)$ contributes nothing. The list
    must be a valid chain: every consecutive pair satisfies :func:`check_extension` (so the
    horizon never decreases, no condition is dropped, ...), else ``ValueError`` names the pair
    and the first forbidden change. $(n,\beta)$ must be a condition of the last revision.

    Because $T_{\max}$ never decreases along a chain and each $S_i \subseteq [0, T_{\max,i}]$,
    $E_r = S_1 \cup \cdots \cup S_r$ and $E_r \supseteq E_{r-1}$: a depth stored under an
    earlier revision is never dropped, and it contains $0, 1, T_{\max,r}$. For one revision it
    is that revision's own schedule.

    ``protocol`` (design §12) selects whose schedules are united: each revision's own schedule
    for that protocol (:meth:`ExperimentManifest.checkpoints`); a revision that does not run the
    protocol contributes nothing, and the last revision must run it.
    """
    check_protocol(protocol)
    chain = list(revisions)
    if not chain:
        raise ValueError("effective_checkpoints needs at least one revision")
    for k, manifest in enumerate(chain):
        if not isinstance(manifest, ExperimentManifest):
            raise TypeError(f"revision {k} is not an ExperimentManifest")
    for k in range(1, len(chain)):
        message = check_extension(chain[k - 1], chain[k])
        if message is not None:
            raise ValueError(f"revision {k} does not extend revision {k - 1}: {message}")
    horizon = chain[-1].layers(n, beta)  # ValueError unless (n, beta) is a condition of the last
    chain[-1].protocol_settings(protocol)  # ValueError unless the last revision runs the protocol
    points: set[int] = set()
    for manifest in chain:
        if manifest.has_condition(n, beta) and protocol in manifest.protocol_names:
            points.update(manifest.checkpoints(n, beta, protocol))
    return tuple(sorted(p for p in points if 0 <= p <= horizon))
