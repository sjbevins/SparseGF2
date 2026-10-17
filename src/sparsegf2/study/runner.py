"""The experiment runner of the Watts–Strogatz study (v1.4 §5.3, §7.2–§7.4, §8; design §4, §12).

The runner turns the manifest revisions of a :class:`~sparsegf2.study.store.TableauStore` into
stored records, through the native engine only (``mipt.MonitoredCircuit.run``):

* :func:`requirements` says what the **simulation** of a trajectory of a condition records under
  the latest revision: its horizon $T_{\\max}$, the effective checkpoint schedule $E_r$ (ruling
  R-CK) and the engine probes with their settings and stable names. Its SHA-256 (canonical JSON)
  names it and keys the completions: a trajectory is run again only when it changes.
* :func:`derivation` says what is **derived** from a trajectory's stored data: the
  trajectory-level observables (first passages with their checkpoint bounds, the peaks of
  v1.4 §5.8) over the last revision's own schedule $S_r$ and the horizon, with the predeclared
  peak rule. A change of it (the rule, or a schedule with the same depths) is computed from the
  stored data by :func:`derive`, never by running a circuit again.
* :func:`plan` lists the work units still to do: for every protocol, condition and rate, the
  trajectories $(g, c)$ with $g < G_*$, $c < C$ that have no completion under the current
  requirements, grouped into rectangles of graphs and circuits sized by the manifest's
  ``storage`` and ``resources`` (and the memory of a unit). The order is deterministic.
* :func:`run` executes units: one engine call per unit on the trajectory indices
  ``u = g * stride + c`` (graph collection ``0 .. g1 - 1`` from the store, digest-checked); the
  records of every trajectory are derived and encoded on several threads (pure functions of the
  engine's row and the key), then written with the unit's completion in **one** transaction. A
  crash before the commit leaves nothing; :func:`run` again re-plans and redoes the unit.
* :func:`extend` adds a revision; the plan then holds the new conditions, rates, graphs and
  circuits, and reruns the trajectories whose simulation requirements changed (a larger
  horizon, new checkpoints, a new observable) from layer 0, every stored record being
  reproduced and compared exactly before the new ones are added (design D3).
* :func:`replay`, :func:`replay_signed`, :func:`circuit`: one trajectory again, from its stored
  identity: the phase-free tableau at any depth (checked against the stored digests), the
  outcome-aware signed run on the exported circuit, the exported circuit itself.
* :func:`compute`: a retrospective snapshot observable of the catalog over the stored snapshots,
  in batches, without simulation; :func:`derive`: the trajectory-level observables of the latest
  revision from the stored data; :func:`verify`: run a sample again and compare every stored
  record of it.

``docs/study/05-store-and-reproducibility.md`` has the definitions and proofs.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import hashlib
import json
import math
import struct
import time
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numba
import numpy as np

from sparsegf2 import analysis, mipt
from sparsegf2._parallel import threads
from sparsegf2._version import __version__
from sparsegf2.circuit import Circuit
from sparsegf2.codes import StabilizerCode
from sparsegf2.simulator import Simulator
from sparsegf2.study import observables as ob
from sparsegf2.study.identity import (
    GRAPH_GENERATOR_VERSION,
    PROTOCOLS,
    GraphRecord,
    Token,
    TrajectoryKey,
    build_graph,
    canonical_json,
    dynamics_version,
    json_digest,
    token_words,
)
from sparsegf2.study.manifest import CorrelatedErasure, ExperimentManifest, effective_checkpoints
from sparsegf2.study.store import (
    DIAGNOSTICS_SEED_TAG,
    OUTCOME_SEED_TAG,
    SNAPSHOT_ABSENT,
    SNAPSHOT_PHYS_ONLY,
    SNAPSHOT_STORED,
    ObservationRow,
    SnapshotRow,
    TableauStore,
    _compare_observation,
    _compare_snapshot,
    _now,
    environment,
    environment_digest,
    git_provenance,
    graph_measures,
    observation_row,
    snapshot_row,
)
from sparsegf2.tableau import PhaseFreeTableau, physical_tableaux, to_bytes_batch

__all__ = [
    "DEFAULT_UNIT_BYTES",
    "POOL_THREADS",
    "PROBE_NAMES",
    "ComputeSummary",
    "Current",
    "DeriveSummary",
    "Replay",
    "RunSummary",
    "SignedReplay",
    "Verification",
    "WorkUnit",
    "circuit",
    "compute",
    "current",
    "derivation",
    "derive",
    "diagnose_samplers",
    "engine_probes",
    "extend",
    "outcome_seed_of",
    "plan",
    "replay",
    "replay_signed",
    "requirements",
    "requirements_digest",
    "run",
    "signed_digest",
    "verify",
]

#: The memory budget of one work unit when ``resources.max_unit_bytes`` is not set: the
#: engine's snapshot buffer and probe values, the encoded records held until the commit, and
#: the canonical forms of one evaluation (:func:`_bytes_per_trajectory`).
DEFAULT_UNIT_BYTES = 1 << 30

#: The names of the engine probes. The sampled probes' random streams are keyed by their names
#: (``RANDOMNESS_CONTRACT = 2``), so these names are part of every stored sampled value: they
#: never change (a correlated_erasure probe is named by the manifest).
PROBE_NAMES = {
    "code_dimension": "code_dimension",
    "tableau_snapshot": "tableau_snapshot",
    "code_snapshot": "code_snapshot",
    "contiguous_distance": "contiguous_distance",
    "erasure_first_failure": "erasure_first_failure",
    "uncorrectable_region_counts": "uncorrectable_region_counts",
}

# A test hook: called inside a unit's transaction after every insert and before the commit
# (fault injection: an exception there, or a killed process, must leave the store unchanged).
_BEFORE_COMMIT: Callable[[WorkUnit], None] | None = None


# ---------------------------------------------------------------------------------------------
# requirements and derivations
# ---------------------------------------------------------------------------------------------


def requirements(
    revisions: Sequence[ExperimentManifest], protocol: str, n: int, beta: Any
) -> dict[str, Any]:
    """What the simulation of a trajectory of ``(protocol, n, beta)`` records under the last revision.

    A plain dictionary (its canonical JSON is hashed by :func:`requirements_digest`):
    ``protocol``, ``n``, ``beta`` (token), ``layers`` ($T_{\\max}$), ``checkpoints`` (the
    effective schedule $E_r$ of ruling R-CK, every probe's depths), ``dynamics`` (the dynamics
    version), and for protocol F ``stop_when_purified``, ``snapshot`` (``"tableau_snapshot"``,
    the joint state and its physical code, or ``"code_snapshot"``, the physical code only),
    ``contiguous_distance``, ``uniform_erasure`` (samples and exact budget, or None) and
    ``correlated_erasure`` (each entry's settings with its probe name). The rate is not part of
    it: every rate of a condition records the same things. The code dimension is always recorded
    at every layer (F). What is derived from these records (the trajectory-level observables,
    the peak rule, the peak domain) is :func:`derivation`, not part of the requirements: a
    change of it never runs a circuit again.
    """
    chain = list(revisions)
    latest = chain[-1]
    b = Token.parse(beta, what="beta")
    out: dict[str, Any] = {
        "protocol": protocol,
        "n": int(n),
        "beta": b.token,
        "layers": latest.layers(n, b),
        "checkpoints": list(effective_checkpoints(chain, n, b, protocol)),
        "dynamics": dynamics_version(protocol),
    }
    if protocol == "P":
        out["snapshot"] = "tableau_snapshot"
        return out
    obs = latest.observables
    uniform = obs.uniform_erasure
    out.update(
        stop_when_purified=latest.dynamics.stop_when_purified,
        snapshot="tableau_snapshot" if obs.tableau_snapshots else "code_snapshot",
        contiguous_distance=obs.contiguous_distance,
        uniform_erasure=None if uniform is None else uniform.to_dict(),
        correlated_erasure=[
            {**entry.to_dict(), "name": name}
            for entry, name in zip(obs.correlated_erasure, obs.probe_names(), strict=True)
        ],
    )
    return out


def requirements_digest(req: Mapping[str, Any]) -> str:
    """The SHA-256 of the canonical JSON of :func:`requirements` (or of a :func:`derivation`)."""
    return json_digest(dict(req))


def _peak_params(observable: str, rule: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    """The parameters of a peak observable: the manifest's peak rule restricted to the
    parameters the catalog entry has (``analysis.peak``), plus ``extra``."""
    names = ob.get(observable).params
    return {**{k: v for k, v in rule.items() if k in names}, **extra}


def _derived_keys(manifest: ExperimentManifest) -> list[tuple[str, dict[str, Any]]]:
    """The trajectory-level observables a protocol-F trajectory derives under ``manifest``, with
    their parameters: ``first_passage``; ``peak_d_cyc`` (exact rule) with ``contiguous_distance``;
    ``peak_D`` (``analysis.peak``) with ``uniform_erasure``; ``peak_D_J`` per correlated entry
    and coupling, keyed by the coupling ``J`` and the probe's ``name``."""
    obs = manifest.observables
    rule = manifest.analysis.peak.to_dict()
    keys: list[tuple[str, dict[str, Any]]] = [("first_passage", {})]
    if obs.contiguous_distance:
        keys.append(("peak_d_cyc", {"rule": "exact"}))
    if obs.uniform_erasure is not None:
        keys.append(("peak_D", _peak_params("peak_D", rule)))
    for entry, name in zip(obs.correlated_erasure, obs.probe_names(), strict=True):
        for coupling in entry.couplings:
            keys.append(("peak_D_J", _peak_params("peak_D_J", rule, J=float(coupling), name=name)))
    return keys


def derivation(
    revisions: Sequence[ExperimentManifest], protocol: str, n: int, beta: Any
) -> dict[str, Any] | None:
    """What is derived from a stored trajectory of ``(protocol, n, beta)`` under the last revision.

    For protocol F the dictionary ``layers`` (the horizon $T$), ``domain`` (the last revision's
    own schedule $S_r(n, \\beta)$, over which peaks and the checkpoint bounds of the first
    passages are taken; G-OBS-2: a store built through revisions and a store made directly
    under the last manifest give the same values; $S_r \\subseteq E_r$ is stored) and
    ``observables``: the sorted ``[id, version, params]`` of every trajectory-level observable
    it asks for (:func:`_derived_keys`). Its digest names it (a trajectory is *derived* under it
    when every one of these values is stored); the digest of ``layers`` and ``domain`` alone is
    the **context** of each stored trajectory-level value (:class:`Current`). Protocol P derives
    nothing (None).
    """
    if protocol != "F":
        return None
    chain = list(revisions)
    latest = chain[-1]
    b = Token.parse(beta, what="beta")
    observables = sorted(
        [name, ob.get(name).version, ob.params_key(name, params)]
        for name, params in _derived_keys(latest)
    )
    return {
        "layers": latest.layers(n, b),
        "domain": list(latest.checkpoints(n, b, protocol)),
        "observables": observables,
    }


@dataclass(frozen=True)
class Current:
    """What the latest revision asks of the trajectories of one ``(protocol, n, beta)``.

    ``requirements`` (:func:`requirements`) and its ``digest``; ``derivation``
    (:func:`derivation`, None for P) and its ``derivation_digest``; ``context``: the digest of
    the derivation's ``{"layers", "domain"}``, under which the trajectory-level values are
    stored (empty for P); ``derived``: the ``(id, version, params)`` of those values, with
    ``derived_params`` their resolved parameters.
    """

    protocol: str
    n: int
    beta: str
    requirements: dict[str, Any]
    digest: str
    derivation: dict[str, Any] | None
    derivation_digest: str
    context: str
    derived: tuple[tuple[str, str, str], ...]
    derived_params: tuple[dict[str, Any], ...]

    @property
    def context_json(self) -> dict[str, Any] | None:
        """The ``{"layers", "domain"}`` whose digest is :attr:`context` (None for P)."""
        if self.derivation is None:
            return None
        return {"layers": self.derivation["layers"], "domain": self.derivation["domain"]}


def current(revisions: Sequence[ExperimentManifest], protocol: str, n: int, beta: Any) -> Current:
    """The :class:`Current` requirements and derivation of ``(protocol, n, beta)``."""
    chain = list(revisions)
    req = requirements(chain, protocol, n, beta)
    der = derivation(chain, protocol, n, beta)
    if der is None:
        return Current(
            protocol=protocol, n=int(n), beta=req["beta"], requirements=req,
            digest=requirements_digest(req), derivation=None, derivation_digest="", context="",
            derived=(), derived_params=(),
        )  # fmt: skip
    keys = _derived_keys(chain[-1])
    derived = tuple(
        (name, ob.get(name).version, ob.params_key(name, params)) for name, params in keys
    )
    resolved = tuple(ob.resolve_params(name, params) for name, params in keys)
    return Current(
        protocol=protocol,
        n=int(n),
        beta=req["beta"],
        requirements=req,
        digest=requirements_digest(req),
        derivation=der,
        derivation_digest=requirements_digest(der),
        context=json_digest({"layers": der["layers"], "domain": der["domain"]}),
        derived=derived,
        derived_params=resolved,
    )


def engine_probes(req: Mapping[str, Any]) -> list[mipt.Probe]:
    """The engine probes of the requirements, with their stable names (:data:`PROBE_NAMES`)."""
    cps = list(req["checkpoints"])
    if req["protocol"] == "P":
        return [mipt.tableau_snapshot(PROBE_NAMES["tableau_snapshot"], checkpoints=cps)]
    probes = [mipt.code_dimension()]
    if req["snapshot"] == "tableau_snapshot":
        probes.append(mipt.tableau_snapshot(PROBE_NAMES["tableau_snapshot"], checkpoints=cps))
    else:
        probes.append(mipt.code_snapshot(PROBE_NAMES["code_snapshot"], checkpoints=cps))
    if req["contiguous_distance"]:
        probes.append(mipt.contiguous_distance(checkpoints=cps))
    uniform = req["uniform_erasure"]
    if uniform is not None:
        probes.append(
            mipt.erasure_first_failure(
                uniform["samples"], PROBE_NAMES["erasure_first_failure"], checkpoints=cps
            )
        )
        if uniform["exact_budget"]:
            probes.append(
                mipt.uncorrectable_region_counts(
                    name=PROBE_NAMES["uncorrectable_region_counts"],
                    budget=uniform["exact_budget"],
                    checkpoints=cps,
                )
            )
    for entry in req["correlated_erasure"]:
        probes.append(_correlated_entry(entry).probe(name=entry["name"], checkpoints=cps))
    return probes


def _correlated_entry(entry: Mapping[str, Any]) -> CorrelatedErasure:
    """A requirements' correlated entry (its settings with its ``name``) as the manifest's object."""
    return CorrelatedErasure.from_dict(
        {k: v for k, v in entry.items() if k != "name"}, "correlated_erasure"
    )


def _zlib_bound(size: int) -> int:
    """zlib's ``compressBound``: the largest compressed size of ``size`` bytes."""
    return size + (size >> 12) + (size >> 14) + (size >> 25) + 13


def _bytes_per_trajectory(req: Mapping[str, Any]) -> int:
    """The memory one trajectory of a unit holds at most (``resources.max_unit_bytes``).

    With ``E`` evaluations, ``N`` the qubits of the snapshot (``2n`` for the joint state of
    protocol F, ``n`` otherwise) and ``W = 2 ceil(N / 64)`` words per row:

    * the engine's snapshot rows, ``8 E N W`` bytes, held until every evaluation of the unit
      is canonicalized (then released);
    * the encoded snapshots held until the commit: per evaluation the compressed canonical
      bytes of the joint state (``16 + N ceil(2N / 8)`` bytes before zlib) and of the physical
      code (``16 + n ceil(2n / 8)``), at most zlib's bound of each;
    * the canonical forms of the one evaluation being encoded: its words (``8 N W``) and bytes
      for the joint state and the physical code;
    * the probe values (8 bytes each: the code dimension at every layer, the samples, counts
      and correlated fractions at every checkpoint), and their encoded observation data (the
      JSON of the samples, the failure curves and their errors: 64 bytes per value, a bound of
      the canonical JSON of a float or an integer and its zlib stream).

    A fixed 4 KiB per trajectory covers the record objects. ``tests/test_study_ingest.py``
    measures a unit's peak memory against this estimate.
    """
    n = int(req["n"])
    evaluations = len(req["checkpoints"])
    joint = req["snapshot"] == "tableau_snapshot"
    qubits = 2 * n if req["protocol"] == "F" and joint else n
    words = 2 * ((qubits + 63) // 64)
    engine = evaluations * qubits * words * 8
    canonical = 16 + qubits * ((2 * qubits + 7) // 8)
    phys = 16 + n * ((2 * n + 7) // 8) if req["protocol"] == "F" else 0
    encoded = evaluations * (_zlib_bound(canonical) + (_zlib_bound(phys) if phys else 0))
    one = qubits * words * 8 + canonical
    if phys:
        one += n * 2 * ((n + 63) // 64) * 8 + phys
    values = 0
    data = 0
    if req["protocol"] == "F":
        values += int(req["layers"]) + 1
        values += evaluations * (2 + int(bool(req["contiguous_distance"])))
        uniform = req["uniform_erasure"]
        if uniform is not None:
            values += evaluations * int(uniform["samples"])
            data += evaluations * (int(uniform["samples"]) + 2 * (n + 1))
            if uniform["exact_budget"]:
                values += evaluations * analysis.exhaustive_cutoff(n, int(uniform["exact_budget"]))
        for entry in req["correlated_erasure"]:
            sizes = _correlated_entry(entry).expand_sizes(n)
            cells = evaluations * len(entry["couplings"]) * len(sizes)
            values += cells
            data += 2 * cells + 2 * evaluations * len(entry["couplings"])
    return engine + encoded + one + 8 * values + 64 * (values + data) + 4096


# ---------------------------------------------------------------------------------------------
# work units and the plan
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkUnit:
    """A rectangle of trajectories run in one engine call (design §4).

    ``protocol``, ``n``, ``beta``, ``p``: the condition; graphs ``g0 <= g < g1`` and circuits
    ``c0 <= c < c1`` (``graphs = (g0, g1)``, ``circuits = (c0, c1)``); ``layers``, the horizon;
    ``requirements``, the digest of what each trajectory records; ``revision``, the manifest
    revision it was planned under; ``rerun``: its trajectories are stored already and run again
    for what a revision added (a deeper horizon, new depths, a new observable).
    """

    protocol: str
    n: int
    beta: Token
    p: Token
    graphs: tuple[int, int]
    circuits: tuple[int, int]
    layers: int
    requirements: str
    revision: int
    rerun: bool = False

    @property
    def trajectories(self) -> int:
        return (self.graphs[1] - self.graphs[0]) * (self.circuits[1] - self.circuits[0])

    def pairs(self) -> list[tuple[int, int]]:
        """The ``(g, c)`` of the unit, graph-major."""
        return [
            (g, c)
            for g in range(self.graphs[0], self.graphs[1])
            for c in range(self.circuits[0], self.circuits[1])
        ]

    def indices(self, stride: int) -> np.ndarray:
        """The native trajectory indices ``u = g * stride + c`` (v1.4 §7.2, design D2)."""
        return np.array([g * stride + c for g, c in self.pairs()], dtype=np.int64)

    def __str__(self) -> str:
        return (
            f"{self.protocol}:n={self.n}:beta={self.beta}:p={self.p}:g={self.graphs[0]}.."
            f"{self.graphs[1] - 1}:c={self.circuits[0]}..{self.circuits[1] - 1}:T={self.layers}"
        )


def _runs(values: Sequence[int]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive integers of a sorted list, as half-open ``(lo, hi)``."""
    out: list[tuple[int, int]] = []
    for v in values:
        if out and out[-1][1] == v:
            out[-1] = (out[-1][0], v + 1)
        else:
            out.append((v, v + 1))
    return out


def plan(
    store: TableauStore,
    *,
    graphs_per_unit: int | None = None,
    max_unit_trajectories: int | None = None,
    max_unit_bytes: int | None = None,
) -> list[WorkUnit]:
    """The work units still to do under the store's latest revision, in a deterministic order.

    For every protocol (F, then P), condition $(n, \\beta)$ in the manifest's order and rate
    $p$: the trajectories $(g, c)$, $g < G_*$, $c < C$, without a completion under the current
    :func:`requirements` (new ones, and stored ones whose simulation requirements changed; a
    change of what is derived from them is :func:`derive`'s, not the plan's). They are grouped
    into rectangles: for each run of consecutive missing circuits of a graph, graphs with the
    same run and consecutive labels go together, at most ``storage.graphs_per_unit`` graphs,
    ``resources.max_unit_trajectories`` trajectories and ``resources.max_unit_bytes`` (default
    :data:`DEFAULT_UNIT_BYTES`) of memory per unit (:func:`_bytes_per_trajectory`; a graph's
    circuits are split when even one graph exceeds them). A unit holds either stored
    trajectories run again (``rerun``: what a revision added, G-RUN-2) or new ones, and the
    reruns come first, so the stored trajectories receive the new depths before any new
    trajectory is added. Then units are sorted by protocol, ``n``, ``beta``, ``p``, first graph
    and first circuit. The three keywords override the manifest's sizes for this plan only; how
    trajectories are split into units never changes a stored value.
    """
    for name, value in (
        ("graphs_per_unit", graphs_per_unit),
        ("max_unit_trajectories", max_unit_trajectories),
        ("max_unit_bytes", max_unit_bytes),
    ):
        if value is not None and (not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    revisions = store.revisions()
    latest = revisions[-1]
    number = len(revisions)
    graphs_total = latest.trials.graphs
    circuits_total = latest.trials.circuits_per_graph
    budget = max_unit_bytes or latest.resources.max_unit_bytes or DEFAULT_UNIT_BYTES
    most = max_unit_trajectories or latest.resources.max_unit_trajectories
    per_unit_graphs = graphs_per_unit or latest.storage.graphs_per_unit
    units: list[WorkUnit] = []
    for protocol in latest.protocol_names:
        for cond in latest.conditions():
            req = requirements(revisions, protocol, cond.n, cond.beta)
            digest = requirements_digest(req)
            per_trajectory = _bytes_per_trajectory(req)
            cap = max(1, min(most, budget // per_trajectory))
            for p in cond.rates:
                done = store.completed(protocol, cond.n, cond.beta, p, digest)
                stored = store.stored_pairs(protocol, cond.n, cond.beta, p)
                pieces: dict[tuple[bool, int, int], list[int]] = {}
                for g in range(graphs_total):
                    missing = [c for c in range(circuits_total) if (g, c) not in done]
                    for rerun in (True, False):
                        kind = [c for c in missing if ((g, c) in stored) == rerun]
                        for lo, hi in _runs(kind):
                            for c0 in range(lo, hi, cap):
                                pieces.setdefault((rerun, c0, min(hi, c0 + cap)), []).append(g)
                for (rerun, c0, c1), labels in pieces.items():
                    per_unit = max(1, min(per_unit_graphs, cap // max(1, c1 - c0)))
                    for g0, g1 in _runs(labels):
                        for start in range(g0, g1, per_unit):
                            units.append(
                                WorkUnit(
                                    protocol=protocol,
                                    n=cond.n,
                                    beta=cond.beta,
                                    p=p,
                                    graphs=(start, min(g1, start + per_unit)),
                                    circuits=(c0, c1),
                                    layers=int(req["layers"]),
                                    requirements=digest,
                                    revision=number,
                                    rerun=rerun,
                                )
                            )
    units.sort(
        key=lambda u: (
            not u.rerun,
            PROTOCOLS.index(u.protocol),
            u.n,
            u.beta,
            u.p,
            u.graphs[0],
            u.circuits[0],
        )
    )
    return units


# ---------------------------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSummary:
    """What :func:`run` did: units, trajectories and snapshots written, time, units left, and
    the stored trajectories whose trajectory-level values :func:`derive` computed afterwards."""

    units: int
    trajectories: int
    snapshots: int
    elapsed_seconds: float
    remaining_units: int
    derived: int = 0


class _GraphCache:
    """Graph records of one run: read from the store (digest-checked) or built once."""

    def __init__(self, store: TableauStore, manifest: ExperimentManifest):
        self.store = store
        self.manifest = manifest
        self.records: dict[tuple[int, Token, int], GraphRecord] = {}
        self.new: set[tuple[int, Token, int]] = set()

    def get(self, n: int, beta: Token, g: int) -> GraphRecord:
        key = (n, beta, g)
        if key not in self.records:
            if self.store.has_graph(n, beta, g):
                record = self.store.graph(n, beta, g)
                expected = self.manifest.graph_seed(n, beta, g)
                if tuple(record.seed) != tuple(expected) or record.generator != (
                    GRAPH_GENERATOR_VERSION
                ):
                    raise ValueError(
                        f"graph (n = {n}, beta = {beta}, g = {g}) is stored with seed "
                        f"{record.seed} and generator {record.generator!r}; this manifest and "
                        f"code give {expected} and {GRAPH_GENERATOR_VERSION!r}"
                    )
            else:
                record = build_graph(n, beta, g, self.manifest.seeds.master_graph_seed)
                self.new.add(key)
            self.records[key] = record
        return self.records[key]

    def committed(self, keys: Sequence[tuple[int, Token, int]]) -> None:
        for key in keys:
            self.new.discard(key)


def run(
    store: TableauStore,
    *,
    num_threads: int | None = None,
    max_units: int | None = None,
    progress: Callable[[WorkUnit, int, int, Mapping[str, Any]], None] | None = None,
    units: Sequence[WorkUnit] | None = None,
) -> RunSummary:
    """Execute the plan (or the given ``units``), one engine call and one transaction per unit.

    ``num_threads`` (default: ``resources.num_threads``, else every thread) never changes a
    result: it sets the engine's threads and the threads that derive and encode the records.
    ``max_units`` stops after that many units (a resumable partial run). ``progress`` is called
    after each unit as ``progress(unit, index, total, info)``. A unit's records are committed
    together; an exception anywhere in a unit (building its graphs, the engine call, the
    canonical forms, the records, the writes) rolls it back, records it as failed (with the
    error, and no data) and is raised again. Running again re-plans: units already done are
    skipped. Without ``max_units``, :func:`derive` then computes the trajectory-level values of
    every stored trajectory that lacks those of the latest revision (a revision that changed
    only what is derived, such as the peak rule, runs no circuit).
    """
    store._require_writer()
    store.check_contract()
    todo = list(plan(store) if units is None else units)
    if max_units is not None:
        if max_units < 0:
            raise ValueError(f"max_units must be non-negative, got {max_units}")
        todo = todo[:max_units]
    manifest = store.manifest
    threads_ = manifest.resources.num_threads if num_threads is None else num_threads
    cache = _GraphCache(store, manifest)
    start = time.perf_counter()
    trajectories = snapshots = 0
    for index, unit in enumerate(todo):
        info = _run_unit(store, unit, cache, threads_)
        trajectories += unit.trajectories
        snapshots += int(info["snapshots"])
        if progress is not None:
            progress(unit, index, len(todo), info)
    derived = 0
    if max_units is None:
        derived = derive(store, num_threads=threads_).trajectories
    remaining = len(plan(store))
    return RunSummary(
        units=len(todo),
        trajectories=trajectories,
        snapshots=snapshots,
        elapsed_seconds=time.perf_counter() - start,
        remaining_units=remaining,
        derived=derived,
    )


def _unit_spec(
    manifest: ExperimentManifest,
    protocol: str,
    n: int,
    beta: Token,
    p: Token,
    collection: Sequence[np.ndarray],
    probes: Sequence[mipt.Probe],
    *,
    layers: int | None = None,
    stop: bool | None = None,
) -> mipt.MonitoredCircuit:
    kwargs = manifest.monitored_circuit_kwargs(n, beta, p, protocol)
    if layers is not None:
        kwargs["layers"] = int(layers)
    if stop is not None:
        kwargs["stop_when_purified"] = bool(stop)
    return mipt.MonitoredCircuit(edges=cast(Any, list(collection)), probes=list(probes), **kwargs)


#: The most threads that derive and encode records (:func:`_pool`). Their numpy, zlib, SHA-256
#: and isotropy work releases the GIL, but the Python between those calls does not: measured
#: (``benchmarks/study_store_costs.py``'s unit, 64 trajectories), the work per trajectory falls
#: from one thread to two and four and grows again beyond eight as the threads wait for the GIL.
POOL_THREADS = 4


@contextlib.contextmanager
def _pool(workers: int) -> Iterator[concurrent.futures.ThreadPoolExecutor]:
    """Threads that derive and encode records (at most :data:`POOL_THREADS`): pure functions of
    the engine's rows. The results never depend on the number of threads: each item is
    computed alone and they are written in plan order."""
    count = max(1, min(int(workers), POOL_THREADS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        yield pool


def _run_unit(
    store: TableauStore, unit: WorkUnit, cache: _GraphCache, num_threads: int | None
) -> dict[str, Any]:
    revisions = store.revisions()
    manifest = revisions[-1]
    cur = current(revisions, unit.protocol, unit.n, unit.beta)
    if cur.digest != unit.requirements or unit.revision != len(revisions):
        raise ValueError(
            f"unit {unit} was planned under other requirements or another revision: plan again "
            "(runner.plan) after a revision"
        )
    provenance = _provenance()
    _check_environment(store, manifest, provenance)
    started = _now()
    clock = time.perf_counter()
    with threads(num_threads):
        used_threads = numba.get_num_threads()
    snapshots = 0
    try:
        n, beta, p = unit.n, unit.beta, unit.p
        collection = [cache.get(n, beta, g) for g in range(unit.graphs[1])]
        new_graphs = [(n, beta, g) for g in range(unit.graphs[1]) if (n, beta, g) in cache.new]
        spec = _unit_spec(
            manifest, unit.protocol, n, beta, p, [r.edges for r in collection],
            engine_probes(cur.requirements),
        )  # fmt: skip
        indices = unit.indices(manifest.stride)
        result = spec.run(indices=indices, seed=list(manifest.run_seed(n, beta, p)),
                          num_threads=num_threads)  # fmt: skip
        engine_seconds = result.elapsed_seconds
        level = int(manifest.storage.compression_level)
        keys = [
            TrajectoryKey(store.experiment_id, n, beta, p, *divmod(int(u), manifest.stride), 0,
                          unit.protocol)
            for u in indices.tolist()
        ]  # fmt: skip
        with _pool(used_threads) as pool:
            measures = {
                key: pool.submit(graph_measures, cache.records[key].edges, n) for key in new_graphs
            }
            records = _unit_records(result, keys, cur, collection, level, num_threads, pool)
            graph_rows = {key: future.result() for key, future in measures.items()}
        del result
        with store.transaction() as tx:
            for key in new_graphs:
                tx.put_graph(cache.records[key], graph_rows[key])
            tx.put_requirements(cur.digest, canonical_json(cur.requirements))
            if cur.derivation is not None:
                tx.put_requirements(cur.derivation_digest, canonical_json(cur.derivation))
                tx.put_requirements(cur.context, canonical_json(cur.context_json))
            unit_id = tx.add_unit(
                protocol=unit.protocol,
                n=n,
                beta=beta.token,
                p=p.token,
                graphs=canonical_json(list(unit.graphs)),
                circuits=canonical_json(list(unit.circuits)),
                layers=unit.layers,
                requirements=cur.digest,
                status="done",
                revision=unit.revision,
                started_utc=started,
                threads=used_threads,
                trajectories=unit.trajectories,
                **provenance,
            )
            for rec in records:
                snapshots += _write_records(tx, rec, manifest, cur, unit_id, unit.revision)
            tx.finish_unit(
                unit_id, finished_utc=_now(), elapsed_seconds=time.perf_counter() - clock
            )
            if _BEFORE_COMMIT is not None:
                _BEFORE_COMMIT(unit)
    except Exception as exc:
        _record_failure(store, unit, started, used_threads, exc)
        raise
    cache.committed(new_graphs)
    return {
        "unit_id": unit_id,
        "trajectories": unit.trajectories,
        "snapshots": snapshots,
        "elapsed_seconds": time.perf_counter() - clock,
        "engine_seconds": engine_seconds,
        "threads": used_threads,
    }


def _record_failure(
    store: TableauStore, unit: WorkUnit, started: str, used_threads: int, exc: BaseException
) -> None:
    """A failed unit is recorded, without any data, in a transaction of its own."""
    try:
        with store.transaction() as tx:
            tx.add_unit(
                protocol=unit.protocol,
                n=unit.n,
                beta=unit.beta.token,
                p=unit.p.token,
                graphs=canonical_json(list(unit.graphs)),
                circuits=canonical_json(list(unit.circuits)),
                layers=unit.layers,
                requirements=unit.requirements,
                status="failed",
                revision=unit.revision,
                started_utc=started,
                finished_utc=_now(),
                threads=used_threads,
                trajectories=unit.trajectories,
                error=f"{type(exc).__name__}: {exc}"[:2000],
                **_provenance(),
            )
    except Exception:  # pragma: no cover - the original error matters more
        pass


# ---------------------------------------------------------------------------------------------
# the records of a trajectory (pure functions of the engine's row and the key)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _RowData:
    """The engine's values of one protocol-F trajectory at the checkpoints ``cps`` (E of them):
    the k trace through the horizon, the raw and zero-extended contiguous distance, ``D(0)``
    with its error and samples, and per correlated probe its couplings, ``D(J)`` and errors."""

    k: np.ndarray
    cps: np.ndarray
    zero_extended: np.ndarray | None
    d_value: np.ndarray | None
    d_se: np.ndarray | None
    first_failure: np.ndarray | None
    exact_counts: np.ndarray | None
    d_j: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray, bool]]

    def sources(self, layers: int, domain: Sequence[int]) -> _Sources:
        """The inputs of the trajectory-level values at horizon ``layers`` over ``domain``
        (every depth of it is one of ``cps``)."""
        dom = np.asarray(domain, dtype=np.int64)
        position = {int(t): j for j, t in enumerate(self.cps.tolist())}
        missing = [int(t) for t in dom.tolist() if int(t) not in position]
        if missing or layers >= self.k.size:
            raise ValueError(
                f"the run has no values at the depths {missing} or no k trace to layer {layers}"
            )
        on = np.array([position[int(t)] for t in dom.tolist()], dtype=np.int64)

        def pick(a: np.ndarray | None) -> np.ndarray | None:
            return None if a is None else a[on]

        return _Sources(
            k=self.k[: layers + 1],
            zero_extended=pick(self.zero_extended),
            d_value=pick(self.d_value),
            d_se=pick(self.d_se),
            first_failure=pick(self.first_failure),
            exact_counts=pick(self.exact_counts),
            d_j={name: (js, r[on], s[on], exact) for name, (js, r, s, exact) in self.d_j.items()},
        )


@dataclass(frozen=True, eq=False)
class _Sources:
    """The inputs of a trajectory's derived values: ``k_0..k_T`` and, at the domain's depths,
    the zero-extended ``d_cyc``, ``D(0)`` and its error, the first-failure samples and exact
    counts, and per correlated probe its couplings, ``D(J)``, errors and whether they are exact."""

    k: np.ndarray
    zero_extended: np.ndarray | None
    d_value: np.ndarray | None
    d_se: np.ndarray | None
    first_failure: np.ndarray | None
    exact_counts: np.ndarray | None
    d_j: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray, bool]]


def _derive_rows(
    key: TrajectoryKey,
    layers: int,
    domain: Sequence[int],
    context: str,
    wanted: Sequence[tuple[str, str, str, Mapping[str, Any]]],
    src: _Sources,
    level: int,
) -> list[ObservationRow]:
    """The trajectory-level values ``wanted`` (``(id, version, params text, params)``) of one
    trajectory at horizon ``layers`` over ``domain``, encoded under ``context``.

    A pure function of the stored data (:class:`_Sources`) and the key: the same rows whether
    the inputs come from the engine's arrays (:func:`run`) or from the store (:func:`derive`,
    :func:`verify`). ``first_passage``: the exact $\\tau_\\ell$ with the checkpoint bounds over
    the domain; ``peak_d_cyc``: the exact-tie peak of the zero-extended ``d_cyc``; ``peak_D``: the
    plateau peak of ``D(0)`` (true errors: ``se_exact`` True), its sensitivity and the bootstrap
    of $T^*$ seeded by ``bootstrap_seed(key, "peak_D", version)``; ``peak_D_J``: the plateau peak
    of probe ``name``'s ``D(J)`` at coupling ``J`` with the probe's ``se_exact``.
    """
    n = key.n
    dom = np.asarray(domain, dtype=np.int64)
    dom_k = src.k[dom]
    info = {"domain": dom, "schedule_digest": json_digest([int(t) for t in dom.tolist()])}
    out: list[ObservationRow] = []

    def put(name: str, version: str, text: str, **fields: Any) -> None:
        out.append(
            observation_row(
                -1, name, text, version=version, context=context, level=level, **fields
            )  # fmt: skip
        )

    for name, version, text, params in wanted:
        if name == "first_passage":
            fp = ob.first_passage(src.k, dom, num_system=n)
            observed = bool(fp.observed[0])
            put(name, version, text, value=float(fp.tau[0]) if observed else math.nan, se=0.0,
                data={**fp.to_dict(), **info}, method="k-trace",
                status="ok" if observed else "censored")  # fmt: skip
        elif name == "peak_d_cyc":
            if src.zero_extended is None:
                raise ValueError(f"{key}: peak_d_cyc needs the contiguous distance")
            pk = ob.peak(dom, src.zero_extended, rule="exact", k_at=dom_k, horizon=layers)
            put(name, version, text, value=pk.t_star, se=0.0, data={**pk.to_dict(), **info},
                method="exact", status="missing" if pk.missing else "ok")  # fmt: skip
        elif name == "peak_D":
            if src.d_value is None or src.d_se is None:
                raise ValueError(f"{key}: peak_D needs D(0)")
            z = float(params["z"])
            # D(0) from the orderings: its se are the true standard errors (se_exact True)
            pk = ob.peak(dom, src.d_value, src.d_se, rule="plateau", z=z, k_at=dom_k,
                         horizon=layers, se_exact=True)  # fmt: skip
            data = {**pk.to_dict(), **info}
            data["sensitivity"] = [
                s.to_dict()
                for s in ob.peak_sensitivity(dom, src.d_value, src.d_se, params["sensitivity_z"],
                                             k_at=dom_k, horizon=layers, se_exact=True)
            ]  # fmt: skip
            data["bootstrap"] = None
            if int(params["bootstrap"]) > 0:
                if src.first_failure is None:
                    raise ValueError(f"{key}: the bootstrap of peak_D needs the orderings")
                # the catalog's seed (Proposition 18(d) of study/04): the key, id and version only
                seed = ob.bootstrap_seed(key, "peak_D", version)
                boot = ob.bootstrap_peak(
                    dom, src.first_failure, n, z=z, replicates=int(params["bootstrap"]),
                    seed=list(seed), exact_counts=src.exact_counts, k_at=dom_k, horizon=layers,
                )  # fmt: skip
                lo, hi = boot.interval(0.95)
                finite = boot.t_star[np.isfinite(boot.t_star)]
                data["bootstrap"] = {
                    "replicates": boot.replicates, "seed": list(seed), "interval95": [lo, hi],
                    "missing_fraction": boot.missing_fraction,
                    "horizon_fraction": boot.horizon_fraction,
                    "sd": float(finite.std(ddof=1)) if finite.size > 1 else None,
                }  # fmt: skip
            put(name, version, text, value=pk.t_star, se=None, data=data, method="plateau",
                status="missing" if pk.missing else "ok")  # fmt: skip
        elif name == "peak_D_J":
            probe = str(params["name"])
            if probe not in src.d_j:
                raise ValueError(f"{key}: no D(J) of the correlated probe {probe!r}")
            couplings, rob_all, se_all, se_exact = src.d_j[probe]
            where = np.flatnonzero(couplings == float(params["J"]))
            if where.size != 1:
                raise ValueError(f"{key}: probe {probe!r} has no coupling J = {params['J']}")
            i = int(where[0])
            rob, rob_se = rob_all[:, i], se_all[:, i]
            z = float(params["z"])
            # the chain's binomial se are lower bounds: the record says so (se_exact False)
            pk = ob.peak(dom, rob, rob_se, rule="plateau", z=z, k_at=dom_k, horizon=layers,
                         se_exact=se_exact)  # fmt: skip
            data = {**pk.to_dict(), **info}
            data["sensitivity"] = [
                s.to_dict()
                for s in ob.peak_sensitivity(dom, rob, rob_se, params["sensitivity_z"],
                                             k_at=dom_k, horizon=layers, se_exact=se_exact)
            ]  # fmt: skip
            data["probe"] = probe
            put(name, version, text, value=pk.t_star, se=None, data=data, method="plateau",
                status="missing" if pk.missing else "ok")  # fmt: skip
        else:  # pragma: no cover - _derived_keys lists these only
            raise ValueError(f"no derivation of {name!r}")
    return out


@dataclass(frozen=True, eq=False)
class _Records:
    """Every record of one trajectory, validated and encoded, ready to be written."""

    key: TrajectoryKey
    graph_digest: str
    dynamics: str
    layers: int
    final_layer: int
    k: np.ndarray | None
    tau: np.ndarray | None
    status: str
    num_measurements: int
    snapshots: list[SnapshotRow]
    observations: list[ObservationRow]
    data: _RowData | None


def _trace_of(
    key: TrajectoryKey, result: mipt.EnsembleResult, row: int, req: Mapping[str, Any]
) -> tuple[np.ndarray | None, ob.FirstPassage | None, int, str]:
    """The k trace, its first passages, the final layer and the trace status of row ``row``,
    checked against the engine (purification layer, final layer, the stop)."""
    layers = int(req["layers"])
    final = int(result.final_layer[row])
    if key.protocol == "P":
        if final != layers:
            raise ValueError(f"{key}: a protocol-P trajectory must run to its horizon {layers}")
        return None, None, final, "no_reference"
    raw_k = np.asarray(result.values[PROBE_NAMES["code_dimension"]][row], dtype=np.float64)
    if raw_k.shape != (layers + 1,) or not np.all(np.isfinite(raw_k)):
        raise ValueError(f"{key}: the code dimension must be recorded at every layer 0..{layers}")
    k = raw_k.astype(np.int64)
    if k[0] != key.n or np.any(k != raw_k):
        raise ValueError(f"{key}: k_0 must be n = {key.n} (n Bell pairs), got {raw_k[0]}")
    fp = ob.first_passage(k, num_system=key.n)  # refuses an increasing trace
    assert result.purification_layer is not None
    if int(result.purification_layer[row]) != int(fp.tau[0]):
        raise ValueError(
            f"{key}: the engine's purification layer {int(result.purification_layer[row])} "
            f"differs from tau_0 = {int(fp.tau[0])} of the k trace"
        )
    stop = bool(req["stop_when_purified"])
    purified = bool(fp.observed[0])
    expected_final = int(fp.tau[0]) if stop and purified else layers
    if final != expected_final:
        raise ValueError(f"{key}: final layer {final}, expected {expected_final}")
    status = (
        "censored"
        if not purified
        else ("stopped_at_purification" if final < layers else "purified")
    )
    return k, fp, final, status


def _encode_snapshots(
    span: range,
    *,
    keys: Sequence[TrajectoryKey],
    traces: Sequence[tuple[np.ndarray | None, int]],
    saved: np.ndarray,
    joints: Sequence[PhaseFreeTableau | None],
    physs: Sequence[PhaseFreeTableau | None],
    t: int,
    req: Mapping[str, Any],
    level: int,
) -> list[SnapshotRow]:
    """The snapshot rows at depth ``t`` of the trajectories ``span`` of a unit (one task of the
    pool): ``saved[i] < 0`` where a stopped trajectory saved nothing (an absent row with
    ``k = 0``), otherwise the canonical joint state and physical code checked and encoded."""
    protocol = req["protocol"]
    joint_kind = req["snapshot"] == "tableau_snapshot"
    n = int(req["n"])
    stop = protocol == "F" and bool(req["stop_when_purified"])
    made = []
    for i in span:
        key = keys[i]
        k, final = traces[i]
        if int(saved[i]) < 0:
            if not (stop and t > final):
                raise ValueError(f"{key}: no snapshot at t = {t}, but the trajectory ran there")
            made.append(
                snapshot_row(
                    key,
                    t,
                    status=SNAPSHOT_ABSENT,
                    num_qubits=2 * n,
                    k=0,
                    joint=None,
                    phys=None,
                    level=level,
                )  # fmt: skip
            )
            continue
        joint, phys = joints[i], physs[i]
        if protocol == "P":
            if joint is None:
                raise ValueError(f"{key}: the snapshot at t = {t} was not canonicalized")
            made.append(
                snapshot_row(
                    key,
                    t,
                    status=SNAPSHOT_STORED,
                    num_qubits=n,
                    k=None,
                    joint=joint,
                    phys=None,
                    level=level,
                )  # fmt: skip
            )
            continue
        if (joint is None) == joint_kind or phys is None or k is None:
            raise ValueError(f"{key}: the snapshot at t = {t} was not canonicalized")
        made.append(
            snapshot_row(
                key, t, status=SNAPSHOT_STORED if joint_kind else SNAPSHOT_PHYS_ONLY,
                num_qubits=2 * n, k=int(k[t]), joint=joint, phys=phys, level=level,
                check_physical=False,
            )
        )  # fmt: skip
    return made


def _snapshot_rows(
    result: mipt.EnsembleResult,
    keys: Sequence[TrajectoryKey],
    traces: Sequence[tuple[np.ndarray | None, int]],
    req: Mapping[str, Any],
    level: int,
    num_threads: int | None,
    pool: concurrent.futures.ThreadPoolExecutor,
) -> list[list[SnapshotRow]]:
    """Every snapshot of a unit as validated, encoded rows, one evaluation at a time.

    For each evaluation the compiled batch routines of :mod:`sparsegf2.tableau`
    (``from_words_batch``, ``physical_tableaux``, ``to_bytes_batch``) put the snapshots of all
    trajectories in canonical form in parallel (exactly the objects and bytes of the
    one-at-a-time methods); the rows (checks, zlib, SHA-256) are made on the threads of
    ``pool`` (:func:`_encode_snapshots`); then the canonical forms of that evaluation are
    released. The engine's snapshot buffer is released once every evaluation is encoded, so a
    unit holds the encoded rows and one evaluation's canonical forms
    (:func:`_bytes_per_trajectory`).
    """
    name = PROBE_NAMES[req["snapshot"]]
    words, counts = result.snapshots[name]
    rows = counts.shape[0]
    joint_kind = req["snapshot"] == "tableau_snapshot"
    protocol = req["protocol"]
    n = int(req["n"])
    out: list[list[SnapshotRow]] = [[] for _ in range(rows)]
    size = max(1, -(-rows // 64))  # at most 64 tasks per evaluation
    chunks = [range(lo, min(rows, lo + size)) for lo in range(0, rows, size)]
    for e, t in enumerate(int(x) for x in req["checkpoints"]):
        joints: list[PhaseFreeTableau | None]
        physs: list[PhaseFreeTableau | None]
        if joint_kind:
            qubits = n if protocol == "P" else 2 * n
            joints = PhaseFreeTableau.from_words_batch(
                words[:, e], counts[:, e], qubits, num_threads=num_threads
            )
            physs = (
                physical_tableaux(joints, n, num_threads=num_threads)
                if protocol == "F"
                else [None] * rows
            )
        else:
            joints = [None] * rows
            physs = PhaseFreeTableau.from_words_batch(
                words[:, e], counts[:, e], n, num_threads=num_threads
            )
        to_bytes_batch(joints, num_threads=num_threads)
        to_bytes_batch(physs, num_threads=num_threads)
        task = functools.partial(
            _encode_snapshots, keys=keys, traces=traces, saved=counts[:, e], joints=joints,
            physs=physs, t=t, req=req, level=level,
        )  # fmt: skip
        for span, made in zip(chunks, pool.map(task, chunks), strict=True):
            for i, row in zip(span, made, strict=True):
                out[i].append(row)
        joints = physs = []  # this evaluation's canonical forms are released
    result.snapshots.pop(name)  # and, after the last one, the engine's buffer
    return out


def _unit_records(
    result: mipt.EnsembleResult,
    keys: Sequence[TrajectoryKey],
    cur: Current,
    collection: Sequence[GraphRecord],
    level: int,
    num_threads: int | None,
    pool: concurrent.futures.ThreadPoolExecutor,
) -> list[_Records]:
    """Every record of every trajectory of a unit (in the unit's order): the traces, the
    encoded snapshots (:func:`_snapshot_rows`), and on the threads of ``pool`` the snapshot-level
    observations and the derived values of each trajectory (:func:`_trajectory_records`)."""
    req = cur.requirements
    traced = [_trace_of(key, result, row, req) for row, key in enumerate(keys)]
    snaps = _snapshot_rows(
        result, keys, [(k, final) for k, _, final, _ in traced], req, level, num_threads, pool
    )
    samplers: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in req.get("correlated_erasure", []):
        for key in keys:
            _graph_sampler(samplers, entry, collection[key.g])

    derived = _derived_spec(cur)

    def make(row: int) -> _Records:
        return _trajectory_records(
            result, row, keys[row], req, collection[keys[row].g], traced[row], snaps[row],
            samplers, level, derived,
        )  # fmt: skip

    return list(pool.map(make, range(len(keys))))


#: What a trajectory derives in a unit: ``(layers, domain, context, wanted)`` with ``wanted``
#: the ``(id, version, params text, params)`` of :func:`_derive_rows`.
_Derived = tuple[int, Sequence[int], str, Sequence[tuple[str, str, str, Mapping[str, Any]]]]


def _derived_spec(cur: Current) -> _Derived | None:
    """The derivation of :class:`Current` as :func:`_trajectory_records` takes it (None for P)."""
    if cur.derivation is None:
        return None
    wanted = [(*k3, params) for k3, params in zip(cur.derived, cur.derived_params, strict=True)]
    return int(cur.derivation["layers"]), cur.derivation["domain"], cur.context, wanted


def _trajectory_records(
    result: mipt.EnsembleResult,
    row: int,
    key: TrajectoryKey,
    req: Mapping[str, Any],
    graph: GraphRecord,
    traced: tuple[np.ndarray | None, ob.FirstPassage | None, int, str],
    snapshots: list[SnapshotRow],
    samplers: Mapping[tuple[str, str], Mapping[str, Any]],
    level: int,
    derived: _Derived | None,
) -> _Records:
    """Derive, validate and encode every record of one trajectory (row ``row`` of ``result``).

    A pure function of the engine's row, the key, the graph and the requirements ``req``: it
    touches no database, so the rows of a unit are made on several threads and written
    afterwards. ``derived`` adds the trajectory-level values (:func:`_derive_rows`).
    """
    n = key.n
    layers = int(req["layers"])
    k, fp, final, status = traced
    common = {
        "key": key,
        "graph_digest": graph.edge_digest,
        "dynamics": req["dynamics"],
        "layers": layers,
        "final_layer": final,
        "status": status,
        "num_measurements": int(result.num_measurements[row]),
        "snapshots": snapshots,
    }
    if key.protocol == "P":
        return _Records(k=None, tau=None, observations=[], data=None, **common)
    assert k is not None and fp is not None
    cps = np.array(req["checkpoints"], dtype=np.int64)
    k_at = k[cps]
    out: list[ObservationRow] = []

    def put(t: int, name: str, params: str, **fields: Any) -> None:
        out.append(observation_row(t, name, params, level=level, **fields))

    plain = ob.params_key("k", None)  # "{}": k and d_cyc have no parameters
    for t in cps.tolist():
        put(t, "k", plain, value=int(k[t]), se=0.0, data=None, method="k-trace", status="ok")
    zero_extended = None
    if req["contiguous_distance"]:
        raw = np.asarray(result.values["contiguous_distance"][row], dtype=np.float64)
        cd = ob.cyclic_distance(raw, k_at, num_system=n)
        zero_extended = np.asarray(cd.zero_extended, dtype=np.int64)
        for j, t in enumerate(cps.tolist()):
            put(t, "d_cyc", plain, value=raw[j], se=0.0, method="engine-probe",
                status="ok" if cd.has_code[j] else "undefined_no_code",
                data={"has_code": bool(cd.has_code[j]), "zero_extended": int(cd.zero_extended[j])})  # fmt: skip
    uniform = req["uniform_erasure"]
    d_value = d_se = ff = exact = None
    if uniform is not None:
        ff = np.asarray(result.values[PROBE_NAMES["erasure_first_failure"]][row])
        if uniform["exact_budget"]:
            exact = np.asarray(result.values[PROBE_NAMES["uncorrectable_region_counts"]][row])
        profile = ob.uniform_failure(ff, n, exact_counts=exact, k=k_at)
        robust = ob.integrated_robustness(ff, n, exact_counts=exact, k=k_at)
        d_value, d_se = robust.value, robust.se
        method = "hybrid" if exact is not None else "orderings"
        params = ob.params_key(
            "D", {"samples": uniform["samples"], "exact_budget": uniform["exact_budget"]}
        )
        for j, t in enumerate(cps.tolist()):
            put(t, "uniform_failure", params, value=None, se=None, method=method, status="vector", data={
                "first_failure": ff[j].astype(np.int64),
                "exact_counts": None if exact is None else exact[j].astype(np.int64),
                "curve": profile.curve[j], "se": profile.se[j],
                "max_exact": profile.max_exact, "samples": profile.samples,
            })  # fmt: skip
            put(t, "D", params, value=robust.value[j], se=robust.se[j], method=method,
                status="ok" if np.isfinite(robust.value[j]) else "undefined", data={
                "has_code": bool(robust.has_code[j]), "samples": robust.samples,
                "max_exact": robust.max_exact,
            })  # fmt: skip
    d_j: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = {}
    for entry in req["correlated_erasure"]:
        name = entry["name"]
        couplings, sizes, values = result.correlated(name)
        expected = np.array([float(j) for j in entry["couplings"]], dtype=np.float64)
        if not np.array_equal(np.asarray(couplings, dtype=np.float64), expected):
            raise ValueError(f"{key}: probe {name!r} ran couplings {couplings}, not {expected}")
        prof = ob.correlated_failure(
            values[row], couplings, sizes, n, k_at, samples=entry["samples"],
            sampler=dict(samplers[(str(name), graph.edge_digest)]),
        )  # fmt: skip
        method = str(prof.sampler["method"]) if prof.sampler else "sampled"
        texts = {kind: ob.params_key(kind, entry) for kind in ("correlated_failure", "D_J")}
        for j, t in enumerate(cps.tolist()):
            put(t, "correlated_failure", texts["correlated_failure"], value=None, se=None, method=method, status="vector", data={
                "couplings": couplings, "sizes": sizes, "failure": prof.failure[j],
                "se": prof.se[j], "interpolated_sizes": prof.interpolated_sizes,
                "sampler": dict(prof.sampler or {}),
            })  # fmt: skip
            put(t, "D_J", texts["D_J"], value=None, se=None, method=method, status="vector", data={
                "couplings": couplings, "robustness": prof.robustness[j],
                "robustness_se": prof.robustness_se[j], "complete": prof.complete,
                "has_code": bool(prof.has_code[j]), "se_exact": prof.se_exact,
            })  # fmt: skip
        d_j[str(name)] = (expected, prof.robustness, prof.robustness_se, prof.se_exact)
    data = _RowData(
        k=k, cps=cps, zero_extended=zero_extended, d_value=d_value, d_se=d_se,
        first_failure=ff, exact_counts=exact, d_j=d_j,
    )  # fmt: skip
    if derived is not None:
        layers_d, domain, context, wanted = derived
        out.extend(
            _derive_rows(
                key, layers_d, domain, context, wanted, data.sources(layers_d, domain), level
            )  # fmt: skip
        )
    return _Records(k=k, tau=fp.tau, observations=out, data=data, **common)


def _write_records(
    tx: Any,
    rec: _Records,
    manifest: ExperimentManifest,
    cur: Current,
    unit_id: int,
    revision: int,
) -> int:
    """Write one trajectory's records in the unit's transaction: insert what is new, compare
    what is stored (a rerun meets every stored record again, and any difference raises)."""
    key = rec.key
    n, beta, p = key.n, key.beta, key.p
    tid = tx.put_trajectory(
        key,
        u=manifest.stream_index(key.g, key.c),
        p_float_hex=p.hex,
        run_seed=manifest.run_seed(n, beta, p),
        graph_digest=rec.graph_digest,
        dynamics_version=rec.dynamics,
        unit_id=unit_id,
        revision=revision,
    )
    fresh = tid in tx.inserted
    tx.put_trace(
        tid, key, layers=rec.layers, final_layer=rec.final_layer, k=rec.k, tau=rec.tau,
        status=rec.status, num_measurements=rec.num_measurements, unit_id=unit_id,
        revision=revision,
    )  # fmt: skip
    for row in rec.snapshots:
        tx.put_snapshot_row(tid, key, row, unit_id, fresh=fresh)
    if fresh:
        tx.put_fresh_observations(tid, rec.observations, unit_id)
    else:
        for obs_row in rec.observations:
            tx.put_observation_row(tid, key, obs_row, unit_id)
    tx.put_completion(tid, cur.digest, unit_id, revision)
    if cur.derivation is not None:
        tx.put_derivation(tid, cur.derivation_digest, unit_id, revision)
    return len(rec.snapshots)


def _graph_sampler(
    cache: dict[tuple[str, str], dict[str, Any]], entry: Mapping[str, Any], graph: GraphRecord
) -> dict[str, Any]:
    """How a correlated_erasure probe samples regions on one trajectory's graph.

    ``MonitoredCircuit.correlated_sampler`` describes a spec's whole graph collection (how many
    graphs, the range of ladder sizes over them), which depends on how trajectories were split
    into units; the record of a trajectory needs its own graph's: the same description from a
    spec holding that graph alone (the ladders depend only on the graph, the sizes and the
    settings, so they are the ones the run used).
    """
    name = str(entry["name"])
    key = (name, graph.edge_digest)
    if key not in cache:
        probe = _correlated_entry(entry).probe(name=name)
        spec = mipt.MonitoredCircuit(
            num_system=graph.n,
            edges=graph.edges,
            layers=1,
            measurement_rate=0.0,
            reference="purification",
            schedule="random_edges",
            probes=[probe],
        )
        cache[key] = spec.correlated_sampler(name)
    return dict(cache[key])


# ---------------------------------------------------------------------------------------------
# the trajectory-level values of stored trajectories (no simulation)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeriveSummary:
    """What :func:`derive` did: trajectories derived, rows inserted, rows already stored
    (compared and equal)."""

    trajectories: int
    inserted: int
    existing: int


def _stored_sources(
    store: TableauStore, tid: int, key: TrajectoryKey, req: Mapping[str, Any], layers: int,
    domain: Sequence[int], needs: set[str],
) -> _Sources:  # fmt: skip
    """The inputs of a trajectory's derived values read from the store: the k trace at horizon
    ``layers`` and, at the domain's depths, the stored ``d_cyc``, ``D``, ``uniform_failure`` and
    ``D_J`` records the requirements ``req`` name (``needs``: the derived ids asked for)."""
    k = store.k_trace(key, layers=layers)
    dom = [int(t) for t in domain]
    db = store._db()

    def rows(observable: str, params: str) -> dict[int, tuple[Any, Any, Any]]:
        version = ob.get(observable).version
        got = db.execute(
            "SELECT t, value, se, data FROM observations WHERE tid = ? AND observable = ? AND "
            "version = ? AND params = ? AND context = '' AND t >= 0",
            (tid, observable, version, params),
        ).fetchall()
        found = {int(t): (v, s, d) for t, v, s, d in got}
        missing = [t for t in dom if t not in found]
        if missing:
            raise ValueError(f"{key}: no stored {observable!r} at the depths {missing}")
        return found

    def data(blob: Any) -> dict[str, Any]:
        return dict(json.loads(zlib.decompress(blob).decode("utf-8")))

    zero = d_value = d_se = ff = exact = None
    if "peak_d_cyc" in needs:
        got = rows("d_cyc", ob.params_key("d_cyc", {}))
        zero = np.array([data(got[t][2])["zero_extended"] for t in dom], dtype=np.int64)
    uniform = req.get("uniform_erasure")
    if "peak_D" in needs and uniform is not None:
        params = ob.params_key(
            "D", {"samples": uniform["samples"], "exact_budget": uniform["exact_budget"]}
        )
        got = rows("D", params)
        d_value = np.array([_nan(got[t][0]) for t in dom], dtype=np.float64)
        d_se = np.array([_nan(got[t][1]) for t in dom], dtype=np.float64)
        prof = rows("uniform_failure", params)
        datas = [data(prof[t][2]) for t in dom]
        ff = np.array([d["first_failure"] for d in datas], dtype=np.int64)
        if datas and datas[0]["exact_counts"] is not None:
            exact = np.array([d["exact_counts"] for d in datas], dtype=np.int64)
    d_j: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = {}
    if "peak_D_J" in needs:
        for entry in req.get("correlated_erasure", []):
            got = rows("D_J", ob.params_key("D_J", entry))
            datas = [data(got[t][2]) for t in dom]
            js = np.array(datas[0]["couplings"], dtype=np.float64)
            rob = np.array([[_nan(x) for x in d["robustness"]] for d in datas], dtype=np.float64)
            se = np.array([[_nan(x) for x in d["robustness_se"]] for d in datas], dtype=np.float64)
            d_j[str(entry["name"])] = (js, rob.reshape(len(dom), js.size),
                                       se.reshape(len(dom), js.size), bool(datas[0]["se_exact"]))  # fmt: skip
    return _Sources(
        k=k, zero_extended=zero, d_value=d_value, d_se=d_se, first_failure=ff,
        exact_counts=exact, d_j=d_j,
    )  # fmt: skip


def _nan(x: Any) -> float:
    return math.nan if x is None else float(x)


def derive(
    store: TableauStore,
    *,
    n: int | None = None,
    beta: Any = None,
    p: Any = None,
    batch_size: int = 256,
    num_threads: int | None = None,
) -> DeriveSummary:
    """Compute the trajectory-level values of the latest revision from the stored data.

    For every protocol-F condition and rate of the latest manifest (``n``, ``beta``, ``p``
    filter them): the stored trajectories complete under the current :func:`requirements` but
    not derived under the current :func:`derivation` (a revision changed the peak rule, the
    schedule without adding a depth, or added a correlated probe whose trajectories ran
    before) get their first passages and peaks from the stored k trace and the stored values at
    the domain's depths (:func:`_derive_rows`, the function the run uses, so the values are
    those a run would store). ``batch_size`` trajectories at a time: their inputs are read, the
    values computed on ``num_threads`` threads (default: ``resources.num_threads``, else every
    thread) and written in one transaction under ``(id, version, params, context)``; a value
    stored already is compared, never replaced. No circuit is run: the bootstrap of ``peak_D``
    is seeded by ``bootstrap_seed(key, "peak_D", version)``, a function of the key alone.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    store._require_writer()
    revisions = store.revisions()
    latest = revisions[-1]
    number = len(revisions)
    level = int(latest.storage.compression_level)
    workers = latest.resources.num_threads if num_threads is None else num_threads
    with threads(workers):
        workers = numba.get_num_threads()
    trajectories = inserted = existing = 0
    if "F" not in latest.protocol_names:
        return DeriveSummary(0, 0, 0)
    for cond in latest.conditions():
        if n is not None and cond.n != int(n):
            continue
        if beta is not None and cond.beta != Token.parse(beta, what="beta"):
            continue
        cur = current(revisions, "F", cond.n, cond.beta)
        assert cur.derivation is not None
        layers = int(cur.derivation["layers"])
        domain = cur.derivation["domain"]
        wanted = [(*k3, params) for k3, params in zip(cur.derived, cur.derived_params, strict=True)]
        needs = {name for name, _, _ in cur.derived}
        for rate in cond.rates:
            if p is not None and rate != Token.parse(p, what="p"):
                continue
            complete = store.completed("F", cond.n, cond.beta, rate, cur.digest)
            done = store.derived_pairs("F", cond.n, cond.beta, rate, cur.derivation_digest)
            todo = sorted(complete - done)
            for lo in range(0, len(todo), batch_size):
                keys = [
                    store.key(n=cond.n, beta=cond.beta, p=rate, g=g, c=c, protocol="F")
                    for g, c in todo[lo : lo + batch_size]
                ]
                tids = [store._require_tid(key) for key in keys]
                sources = [
                    _stored_sources(store, tid, key, cur.requirements, layers, domain, needs)
                    for tid, key in zip(tids, keys, strict=True)
                ]
                task = functools.partial(
                    _derive_rows, layers=layers, domain=domain, context=cur.context,
                    wanted=wanted, level=level,
                )  # fmt: skip
                with _pool(workers) as pool:
                    made = list(
                        pool.map(
                            lambda item, f=task: f(item[0], src=item[1]),
                            zip(keys, sources, strict=True),
                        )  # fmt: skip
                    )
                with store.transaction() as tx:
                    tx.put_requirements(cur.digest, canonical_json(cur.requirements))
                    tx.put_requirements(cur.derivation_digest, canonical_json(cur.derivation))
                    tx.put_requirements(cur.context, canonical_json(cur.context_json))
                    for tid, key, rows in zip(tids, keys, made, strict=True):
                        for row in rows:
                            new = tx.put_observation_row(tid, key, row, None)
                            inserted += int(new)
                            existing += int(not new)
                        tx.put_derivation(tid, cur.derivation_digest, None, number)
                trajectories += len(keys)
    return DeriveSummary(trajectories=trajectories, inserted=inserted, existing=existing)


# ---------------------------------------------------------------------------------------------
# revisions
# ---------------------------------------------------------------------------------------------


def extend(store: TableauStore, new_manifest: ExperimentManifest) -> list[WorkUnit]:
    """Add ``new_manifest`` as the store's next revision and return the new plan.

    The revision must extend the latest one (:func:`~sparsegf2.study.manifest.check_extension`).
    The plan then holds the trajectories of new conditions, rates, graphs, circuits and
    protocols, and every stored trajectory whose simulation requirements changed (a larger
    horizon, checkpoints added by a new rule, a new observable): those are rerun from layer 0
    (the native stream has no mid-trajectory state, design D3) and every stored record (the k
    trace prefix, each snapshot digest, each value) must be reproduced exactly before the new
    ones are added. A revision that changes only what is derived (the peak rule, a schedule
    whose depths are stored) plans nothing: :func:`run` (or :func:`derive`) computes the new
    trajectory-level values from the stored data. Nothing runs until :func:`run`.
    """
    store.add_revision(new_manifest)
    return plan(store)


# ---------------------------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------------------------


def _depth_list(depths: Any) -> list[int]:
    values = sorted({int(t) for t in np.atleast_1d(np.asarray(depths, dtype=np.int64)).tolist()})
    if not values or values[0] < 0:
        raise ValueError(f"depths must be non-negative layers, at least one, got {depths!r}")
    return values


def _trajectory_context(
    store: TableauStore, key: TrajectoryKey
) -> tuple[Any, ExperimentManifest, list[np.ndarray]]:
    """The stored trajectory record, the latest manifest, and its graph collection 0..g, with
    the stored provenance checked against what the manifest gives, and the contract checked."""
    store.check_contract()
    rec = store.trajectory(key)
    manifest = store.manifest
    expected_seed = tuple(manifest.run_seed(key.n, key.beta, key.p))
    if tuple(rec.run_seed) != expected_seed or rec.p_float_hex != key.p.hex:
        raise ValueError(
            f"trajectory {key}: its stored run seed or rate does not match the manifest"
        )
    if rec.u != manifest.stream_index(key.g, key.c):
        raise ValueError(f"trajectory {key}: its stored index u = {rec.u} does not match")
    collection = [store.graph(key.n, key.beta, g).edges for g in range(key.g + 1)]
    if store.graph(key.n, key.beta, key.g).edge_digest != rec.graph_digest:
        raise ValueError(f"trajectory {key}: its graph digest does not match the graph record")
    return rec, manifest, collection


@dataclass(frozen=True, eq=False)
class Replay:
    """One trajectory rerun by :func:`replay`.

    ``tableaux[t]``: the joint phase-free tableau at each requested depth (system, then
    references; the ``n``-qubit pure state for protocol P); ``k``: ``k_0 .. k_T`` (F) for
    ``T = max(depths)``; ``verified_depths``: the depths whose stored digests were compared and
    matched; ``verified_layers``: the last layer of the stored k trace compared (``-1`` none).
    """

    key: TrajectoryKey
    depths: tuple[int, ...]
    tableaux: dict[int, PhaseFreeTableau]
    k: np.ndarray | None
    verified_depths: tuple[int, ...]
    verified_layers: int


def replay(
    store: TableauStore,
    key: TrajectoryKey | str,
    depths: Any,
    *,
    verify: bool = True,
    num_threads: int | None = None,
) -> Replay:
    """Rerun a stored trajectory from layer 0 and return its tableau at ``depths`` (design §4).

    The trajectory is ``MonitoredCircuit.run(indices=[u])`` with its stored run seed, its graph
    collection from the store and a ``tableau_snapshot`` at ``depths`` (with the code dimension
    at every layer for protocol F), never stopping at purification, so any depth can be asked
    for, stored or not, also beyond the horizon. With ``verify`` every requested depth whose
    snapshot is stored must give the stored joint digest (or, for a store that kept codes only,
    the stored physical digest), and the k trace must equal the stored trace on their common
    layers; a difference raises ``ValueError`` naming the depth. A store written under another
    randomness contract is refused (:meth:`TableauStore.check_contract`).
    """
    k = store.key(key)
    rec, manifest, collection = _trajectory_context(store, k)
    wanted = _depth_list(depths)
    probes: list[mipt.Probe] = [mipt.tableau_snapshot("replay", checkpoints=wanted)]
    if k.protocol == "F":
        probes.insert(0, mipt.code_dimension())
    spec = _unit_spec(
        manifest, k.protocol, k.n, k.beta, k.p, collection, probes, layers=max(wanted), stop=False
    )
    result = spec.run(indices=[rec.u], seed=list(rec.run_seed), num_threads=num_threads)
    tableaux = {t: result.tableau(0, j, "replay") for j, t in enumerate(wanted)}
    kk = None
    if k.protocol == "F":
        kk = np.asarray(result.values["code_dimension"][0]).astype(np.int64)
    verified: list[int] = []
    verified_layers = -1
    if verify:
        stored = set(store.depths(k))
        for t in wanted:
            if t not in stored:
                continue
            snap = store.snapshot(k, t)
            if snap.status == SNAPSHOT_ABSENT:
                continue
            tab = tableaux[t]
            if snap.joint_digest is not None:
                ok = tab.digest() == snap.joint_digest
            else:
                ok = tab.physical_tableau(k.n).digest() == snap.phys_digest
            if not ok:
                raise ValueError(
                    f"replay of {k}: the tableau at t = {t} does not match the stored digest "
                    "(the store, the code or the inputs changed)"
                )
            verified.append(t)
        if kk is not None:
            trace = store.trace(k)
            assert trace.k is not None
            common = min(trace.layers, int(kk.size) - 1)
            if not np.array_equal(trace.k[: common + 1], kk[: common + 1]):
                raise ValueError(f"replay of {k}: the k trace differs from the stored one")
            verified_layers = common
    return Replay(
        key=k,
        depths=tuple(wanted),
        tableaux=tableaux,
        k=kk,
        verified_depths=tuple(verified),
        verified_layers=verified_layers,
    )


def circuit(store: TableauStore, key: TrajectoryKey | str, layers: int | None = None) -> Circuit:
    """The exact circuit a stored trajectory simulates, through ``layers`` (default its horizon).

    ``MonitoredCircuit.circuit(u, seed=run_seed)`` (v1.4 §8.1): the setup (for protocol F the
    Bell pairs ``H i; CX i n+i``; nothing for P), ``TICK``, then per layer its ``CLIFFORD_2Q``
    gates, its ``M`` measurements and a ``TICK``: the ordered gate and mask history of the
    trajectory, for audit (v1.4 §4 "Tracking"; the on-demand ``layer`` record of v1.4 §8).
    """
    k = store.key(key)
    rec, manifest, collection = _trajectory_context(store, k)
    horizon = store.trace(k).layers if layers is None else int(layers)
    spec = _unit_spec(
        manifest, k.protocol, k.n, k.beta, k.p, collection, [], layers=horizon, stop=False
    )
    return spec.circuit(rec.u, seed=list(rec.run_seed))


def _text_words(text: str) -> tuple[int, int]:
    """The two 32-bit words (low, high) of ``blake2b(text, digest_size=8)`` read little-endian:
    the repository's convention for a string in a seed (``identity.token_words`` for tokens)."""
    value = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")
    return value & 0xFFFFFFFF, value >> 32


def outcome_seed_of(key: TrajectoryKey) -> tuple[int, ...]:
    """The default seed of the outcome-aware replay of trajectory ``key`` (G-TAB-2, SI-24).

    The twelve 32-bit words ``(OUTCOME_SEED_TAG, w(experiment_id), protocol, n, w(beta),
    w(p), g, c, r)`` with ``OUTCOME_SEED_TAG = 0x57534F31`` (``"WSO1"``), ``w`` the two words of
    the 8-byte BLAKE2b of the text (:func:`identity.token_words` for the tokens) and the
    protocol as its index in :data:`~sparsegf2.study.identity.PROTOCOLS`. A fixed length and its
    own tag keep it apart from the graph (WSG1), run (WSC1) and region (WSRO) seeds whatever the
    master seeds; it names every field of the key, so every trajectory has its own outcomes.
    """
    words = (
        OUTCOME_SEED_TAG,
        *_text_words(key.experiment_id),
        PROTOCOLS.index(key.protocol),
        key.n,
        *token_words(key.beta),
        *token_words(key.p),
        key.g,
        key.c,
        key.r,
    )
    if any(not 0 <= int(w) < 2**32 for w in words):
        raise ValueError(f"trajectory {key}: a field does not fit one 32-bit word of a seed")
    return tuple(int(w) for w in words)


def signed_digest(xs: np.ndarray, zs: np.ndarray, signs: np.ndarray) -> str:
    """The SHA-256 of a signed stabilizer group in Stim's canonical form (:data:`SIGNED_FORMAT`).

    The bytes are ``b"SGF2SG1\\0"``, the number of qubits as uint32 little-endian, then per
    generator its X and Z bits (``np.packbits(..., bitorder="big")`` of the ``2N`` columns)
    and one sign byte (1 for a minus sign).
    """
    rows = np.concatenate([np.asarray(xs, dtype=np.uint8), np.asarray(zs, dtype=np.uint8)], axis=1)
    body = np.packbits(rows, axis=1, bitorder="big")
    data = b"SGF2SG1\x00" + struct.pack("<I", int(xs.shape[1]))
    data += b"".join(r.tobytes() + bytes([int(s) & 1]) for r, s in zip(body, signs, strict=True))
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, eq=False)
class SignedReplay:
    """One trajectory rerun outcome-aware by :func:`replay_signed`.

    ``outcome_seed``: the seed of the Born outcomes (the recorded default :func:`outcome_seed_of`,
    or the one given); ``measurements``: the outcomes (0/1, one per ``M`` target of the
    circuit, in order); ``random``: which of them were random (the others were determined by
    the state); ``signed[t]``: the signed stabilizer generators at depth ``t`` as
    ``(xs, zs, signs)`` in Stim's canonical form, ``signed_digests[t]`` their
    :func:`signed_digest`; ``tableaux[t]``: the same group with its signs dropped;
    ``verified_depths``: the depths whose stored digests it matched.
    """

    key: TrajectoryKey
    depths: tuple[int, ...]
    outcome_seed: Any
    measurements: np.ndarray
    random: np.ndarray
    signed: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    signed_digests: dict[int, str]
    tableaux: dict[int, PhaseFreeTableau]
    verified_depths: tuple[int, ...]


def replay_signed(
    store: TableauStore,
    key: TrajectoryKey | str,
    depths: Any,
    *,
    outcome_seed: Any = None,
    verify: bool = True,
    record: bool = False,
) -> SignedReplay:
    """The outcome-aware replay of v1.4 §4, §7.4 and validation 5: the trajectory's exported
    circuit (:func:`circuit`) on ``Simulator(N, phases=True)``.

    Every random measurement outcome is a fair coin given the past and every determined one
    is the eigenvalue the state fixes (the Born law of a stabilizer state, v1.4 §4), drawn from
    the simulator's stream seeded by ``outcome_seed``: by default the trajectory's own
    :func:`outcome_seed_of` (a tagged domain of its own), so the replay is reproducible from the
    stored identity; the same seed gives the same outcomes and signed groups. Dropping the signs
    must give the stored phase-free row space at every stored depth (outcome independence,
    v1.4 §6.2; checked with ``verify``, ``ValueError`` otherwise). ``record`` (a writer store)
    keeps the outcomes, the random flags and the signed digests in ``signed_runs``, never in the
    phase-free records.
    """
    k = store.key(key)
    wanted = _depth_list(depths)
    circ = circuit(store, k, max(wanted))
    seed = outcome_seed_of(k) if outcome_seed is None else outcome_seed
    qubits = 2 * k.n if k.protocol == "F" else k.n
    sequence = np.random.SeedSequence(list(seed)) if isinstance(seed, (tuple, list)) else seed
    sim = Simulator(qubits, phases=True, seed=sequence)
    want = set(wanted)

    def observe(s: Simulator, tick: int) -> Any:
        if tick not in want:
            return None
        return PhaseFreeTableau.from_simulator(s), s.stabilizer_arrays(canonical=True)

    out = sim.run(circ, observe=observe)
    tableaux: dict[int, PhaseFreeTableau] = {}
    signed: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for tick, value in out.observations:
        if value is not None:
            tableaux[tick], arrays = value
            signed[tick] = (arrays[0].copy(), arrays[1].copy(), arrays[2].copy())
    digests = {t: signed_digest(*signed[t]) for t in sorted(signed)}
    verified: list[int] = []
    if verify:
        stored = set(store.depths(k))
        for t in wanted:
            if t in stored:
                snap = store.snapshot(k, t)
                if snap.status == SNAPSHOT_ABSENT:
                    continue
                tab = tableaux[t]
                digest = (
                    tab.digest()
                    if snap.joint_digest is not None
                    else tab.physical_tableau(k.n).digest()
                )
                expected = snap.joint_digest if snap.joint_digest is not None else snap.phys_digest
                if digest != expected:
                    raise ValueError(
                        f"signed replay of {k}: dropping the signs at t = {t} does not give the "
                        "stored row space"
                    )
                verified.append(t)
    measurements = np.asarray(out.measurements, dtype=np.uint8).copy()
    random_flags = np.asarray(out.random, dtype=bool).copy()
    if record:
        store._require_writer()
        with store.transaction() as tx:
            tx.put_signed_run(
                store._require_tid(k), k, outcome_seed=seed, layers=max(wanted), depths=wanted,
                outcomes=measurements, random=random_flags, signed_digests=digests,
            )  # fmt: skip
    return SignedReplay(
        key=k,
        depths=tuple(wanted),
        outcome_seed=seed,
        measurements=measurements,
        random=random_flags,
        signed=signed,
        signed_digests=digests,
        tableaux=tableaux,
        verified_depths=tuple(verified),
    )


# ---------------------------------------------------------------------------------------------
# retrospective observables
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeSummary:
    """What :func:`compute` did: the observable's key, rows inserted, rows already stored
    (compared and equal), snapshots read (absent ones included)."""

    observable: str
    version: str
    params: str
    inserted: int
    existing: int
    snapshots: int


#: The retrospective observables whose value at ``k = 0`` their definition fixes, whatever the
#: (pure) physical state: ``k = 0``, ``code_rate = 0``, ``stabilizer_rank = n`` (v1.4 §5.2),
#: ``d_cyc`` undefined (NaN, ``has_code`` False, zero extension 0; v1.4 §5.5), ``d_conn = 0``
#: (v1.4 §5.6) and ``mean_leakage = 0`` (``I(A:R) <= 2 S(R) = 0``; v1.4 §5.7). After a
#: ``stop_when_purified`` stop the state was not evolved, but ``k = 0`` holds at every later
#: depth (v1.4 §9: "code properties thereafter are exactly zero or undefined"), so these are
#: recorded with their exact values; an observable that reads the joint state (an entanglement)
#: or is not in this set stays absent there (SI-05).
DEFINED_AT_ZERO_K = frozenset(
    {"k", "code_rate", "stabilizer_rank", "d_cyc", "d_conn", "mean_leakage"}
)


def _zero_code(n: int) -> StabilizerCode:
    """A physical code with ``k = 0`` (the group of ``|0...0>``, generators ``Z_i``): the stand-in
    for the depths after a stop, used only for the observables of :data:`DEFINED_AT_ZERO_K`,
    whose values do not depend on which pure state it is."""
    return StabilizerCode(["I" * i + "Z" + "I" * (n - 1 - i) for i in range(n)])


def _retrospective_value(
    entry: ob.ObservableDef,
    resolved: Mapping[str, Any],
    key: TrajectoryKey,
    t: int,
    status: str,
    snap: Any,
    edges: np.ndarray | None,
) -> tuple[Any, Any, Any, str, str] | None:
    """``(value, se, data, method, status)`` of a retrospective observable at one stored snapshot
    (``snap``, a :class:`~sparsegf2.study.store.SnapshotRecord`), or None for an absent row
    (no state after a stop, and a value the definition does not fix there)."""
    needs_joint = "joint_tableau" in entry.inputs
    if status == SNAPSHOT_ABSENT:
        if needs_joint or entry.id not in DEFINED_AT_ZERO_K:
            return None
        snapshot = ob.SnapshotInput(num_system=key.n, t=t, key=key, code=_zero_code(key.n),
                                    edges=edges)  # fmt: skip
        value = ob.compute(entry.id, snapshot, resolved, version=entry.version)
        data = {**dict(value.data), "after_stop": True}
        return value.value, value.se, data, "k=0-after-stop", _value_status(entry.id, value)
    joint = snap.tableau() if needs_joint or key.protocol != "F" else None
    code = snap.code() if key.protocol == "F" and not needs_joint else None
    snapshot = ob.SnapshotInput(num_system=key.n, t=t, key=key, code=code, tableau=joint,
                                edges=edges)  # fmt: skip
    value = ob.compute(entry.id, snapshot, resolved, version=entry.version)
    return value.value, value.se, dict(value.data), value.method, _value_status(entry.id, value)


def compute(
    store: TableauStore,
    observable_id: str,
    params: Mapping[str, Any] | None = None,
    *,
    version: str | None = None,
    protocol: str = "F",
    n: int | None = None,
    beta: Any = None,
    p: Any = None,
    t: int | None = None,
    batch_size: int = 256,
) -> ComputeSummary:
    """Compute a retrospective snapshot observable of the catalog over the stored snapshots.

    ``observable_id`` must have a ``compute`` (:attr:`ObservableDef.retrospective`: ``k``,
    ``code_rate``, ``stabilizer_rank``, ``d_cyc``, ``d_conn``, ``half_cut_entropy``,
    ``mean_leakage``). Every snapshot that matches the filters is read
    (:meth:`TableauStore.iter_snapshots`, ``batch_size`` at a time) and given to
    :func:`sparsegf2.study.observables.compute` with the trajectory's key, the stored physical
    code, the joint tableau where the observable needs it, and the graph; each batch's values
    are written in one transaction under ``(observable, version, params)``. A depth after a
    ``stop_when_purified`` stop has no state but ``k = 0``: an observable whose value at
    ``k = 0`` its definition fixes (:data:`DEFINED_AT_ZERO_K`) gets that value (method
    ``k=0-after-stop``, ``after_stop`` in its data), so its ensemble means stay unconditional
    (v1.4 §9); any other gets a row of status ``absent_after_purification`` (value NULL; SI-05:
    an entanglement there is missing, never zero). A value already stored under that key (by
    an earlier call, or by the run for ``k`` and ``d_cyc``, which it records from the engine)
    is compared, never replaced: the recomputed value, error and status must equal it exactly,
    else ``ValueError``; so the call is idempotent, and on the run's values it is a cross-check
    of the stored snapshots. Nothing is simulated: the circuits' random numbers are never drawn
    (the sampled ``mean_leakage`` seeds its regions from the key,
    :func:`~sparsegf2.study.observables.region_seed`).
    """
    entry = ob.get(observable_id, version)
    if not entry.retrospective:
        raise ValueError(
            f"observable {entry.id!r} is computed from {list(entry.inputs)} when the trajectory "
            "is run (or derived from its stored data: runner.derive), not from a stored "
            "snapshot; retrospective ones are "
            f"{sorted(r['id'] for r in ob.catalog() if r['retrospective'])}"
        )
    resolved = ob.resolve_params(entry.id, params, version=entry.version)
    text = ob.params_key(entry.id, resolved, version=entry.version)
    needs_graph = "graph" in entry.inputs
    inserted = existing = count = 0
    graph_cache: dict[tuple[int, str, int], np.ndarray] = {}
    batch: list[tuple[TrajectoryKey, int, tuple[Any, Any, Any, str, str] | None]] = []

    def flush() -> None:
        nonlocal inserted, existing
        if not batch:
            return
        with store.transaction() as tx:
            for key, depth, got in batch:
                tid = store._require_tid(key)
                if got is None:  # absent after a stop, not fixed at k = 0
                    if store._has_observation(tid, depth, entry.id, entry.version, text):
                        existing += 1
                        continue
                    tx.put_observation(
                        tid, key, depth, entry.id, resolved, version=entry.version, value=None,
                        se=None, data=None, method="none", status="absent_after_purification",
                    )  # fmt: skip
                    inserted += 1
                    continue
                value, se, data, method, status = got
                new = tx.put_observation(
                    tid, key, depth, entry.id, resolved, version=entry.version, value=value,
                    se=se, data=data, method=method, compare="value", status=status,
                )  # fmt: skip
                inserted += int(new)
                existing += int(not new)
        batch.clear()

    for snap in store.iter_snapshots(
        protocol=protocol, n=n, beta=beta, p=p, t=t, batch_size=batch_size, status=None
    ):
        key = snap.key
        count += 1
        edges = None
        if needs_graph:
            gk = (key.n, key.beta.token, key.g)
            if gk not in graph_cache:
                graph_cache[gk] = store.graph(key.n, key.beta, key.g).edges
            edges = graph_cache[gk]
        batch.append(
            (
                key,
                snap.t,
                _retrospective_value(entry, resolved, key, snap.t, snap.status, snap, edges),
            )  # fmt: skip
        )
        if len(batch) >= batch_size:
            flush()
    flush()
    return ComputeSummary(
        observable=entry.id,
        version=entry.version,
        params=text,
        inserted=inserted,
        existing=existing,
        snapshots=count,
    )


def _value_status(observable: str, value: ob.ObservableValue) -> str:
    """The status of a retrospective value: ``undefined_no_code`` for a quantity without a code
    (``d_cyc`` at ``k = 0``, the engine's convention too), ``infinite``, ``undefined`` or ``ok``."""
    v = float(value.value)
    if math.isnan(v):
        has_code = value.data.get("has_code")
        return "undefined_no_code" if has_code is False else "undefined"
    return "infinite" if math.isinf(v) else "ok"


# ---------------------------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Verification:
    """The result of :func:`verify`: ``ok`` when nothing failed; the counts checked (trajectories,
    snapshots and observation rows recomputed and compared); one message per failure."""

    ok: bool
    trajectories: int
    snapshots: int
    failures: tuple[str, ...] = field(default_factory=tuple)
    observations: int = 0


def _native_codes(
    store: TableauStore, key: TrajectoryKey, depths: Sequence[int], num_threads: int | None
) -> dict[int, PhaseFreeTableau]:
    """The physical codes of a trajectory at ``depths`` from the engine's own ``code_snapshot``
    (v1.4 §9.2 (g), (h)): the native route, against which the stored codes (derived from the
    joint state, or stored from this probe) are checked."""
    rec, manifest, collection = _trajectory_context(store, key)
    spec = _unit_spec(
        manifest, key.protocol, key.n, key.beta, key.p, collection,
        [mipt.code_snapshot("native", checkpoints=list(depths))], layers=max(depths), stop=False,
    )  # fmt: skip
    result = spec.run(indices=[rec.u], seed=list(rec.run_seed), num_threads=num_threads)
    words, counts = result.snapshots["native"]
    return {
        t: PhaseFreeTableau.from_words(words[0, j, : counts[0, j]], key.n)
        for j, t in enumerate(depths)
    }


def _verification_requirements(
    store: TableauStore, tid: int, cur: Current
) -> dict[str, Any] | None:
    """The requirements a stored trajectory is checked under: the current ones if it is complete
    under them, else those of its completion with the largest horizon (then the most depths)."""
    digests = [
        str(r[0])
        for r in store._db().execute("SELECT requirements FROM completions WHERE tid = ?", (tid,))
    ]
    if cur.digest in digests:
        return dict(cur.requirements)
    reqs = [store.requirements(d) for d in digests]
    if not reqs:
        return None
    return max(reqs, key=lambda r: (int(r["layers"]), len(r["checkpoints"])))


def _verify_trajectory(
    store: TableauStore, key: TrajectoryKey, cur: Current, num_threads: int | None
) -> tuple[list[str], int, int]:
    """Run one stored trajectory again and compare every stored record of it (read only).

    Returns the failures, the snapshots and the observation rows compared."""
    failures: list[str] = []
    rec, manifest, collection = _trajectory_context(store, key)
    tid = rec.tid
    req = _verification_requirements(store, tid, cur)
    if req is None:
        return [f"{key}: no completion: the trajectory holds no complete record set"], 0, 0
    graph = store.graph(key.n, key.beta, key.g)
    spec = _unit_spec(
        manifest, key.protocol, key.n, key.beta, key.p, collection, engine_probes(req),
        layers=int(req["layers"]),
    )  # fmt: skip
    result = spec.run(indices=[rec.u], seed=list(rec.run_seed), num_threads=num_threads)
    level = int(manifest.storage.compression_level)
    traced = _trace_of(key, result, 0, req)
    with _pool(1) as pool:
        snaps = _snapshot_rows(result, [key], [(traced[0], traced[2])], req, level, num_threads,
                               pool)[0]  # fmt: skip
    samplers: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in req.get("correlated_erasure", []):
        _graph_sampler(samplers, entry, graph)
    # the snapshot-level records; the trajectory-level ones are checked per context below
    records = _trajectory_records(result, 0, key, req, graph, traced, snaps, samplers, level,
                                  None)  # fmt: skip
    db = store._db()
    # the trace at the verified horizon, and every shorter stored trace as its prefix
    trace = store.trace(key, layers=int(req["layers"]))
    if (
        (trace.final_layer, trace.status, trace.num_measurements)
        != (records.final_layer, records.status, records.num_measurements)
        or not _same(trace.k, records.k)
        or not _same(trace.tau, records.tau)
    ):
        failures.append(f"{key}: the trace at horizon {req['layers']} differs from the rerun")
    for row in db.execute("SELECT layers FROM traces WHERE tid = ? AND layers < ?",
                          (tid, int(req["layers"]))).fetchall():  # fmt: skip
        old = store.trace(key, layers=int(row[0]))
        if records.k is not None and (old.k is None or not np.array_equal(
            old.k, records.k[: old.layers + 1]
        )):  # fmt: skip
            failures.append(f"{key}: the trace at horizon {old.layers} is not the rerun's prefix")
    stored_depths = {
        int(r[0]): r[1:]
        for r in db.execute(
            "SELECT t, status, num_qubits, rank, k, joint_digest, phys_digest FROM snapshots "
            "WHERE tid = ?",
            (tid,),
        )
    }
    for row in records.snapshots:
        what = f"snapshot of {key} at t = {row.t}"
        if row.t not in stored_depths:
            failures.append(f"{what}: not stored")
            continue
        try:
            _compare_snapshot(what, stored_depths.pop(row.t), row)
        except ValueError as exc:
            failures.append(str(exc))
    for t in sorted(stored_depths):
        failures.append(f"snapshot of {key} at t = {t}: stored but not made by the rerun")
    # every stored observation: the rerun's rows, the derived rows under their own context,
    # and the retrospective rows recomputed from the stored snapshots
    expected = {
        (r.t, r.observable, r.version, r.params, r.context): r for r in records.observations
    }
    checked = 0
    stored_rows = db.execute(
        "SELECT t, observable, version, params, context, value, se, method, data, status "
        "FROM observations WHERE tid = ?",
        (tid,),
    ).fetchall()
    contexts: dict[str, list[tuple[Any, ...]]] = {}
    retrospective: list[tuple[Any, ...]] = []
    for r in stored_rows:
        ident = (int(r[0]), r[1], r[2], r[3], r[4])
        what = f"observation {r[1]!r} (version {r[2]}, params {r[3]}) of {key} at t = {r[0]}"
        row = expected.pop(ident, None)
        if row is not None:
            checked += 1
            try:
                _compare_observation(what, (r[5], r[6], r[7], r[8], r[9]), row)
            except ValueError as exc:
                failures.append(str(exc))
        elif int(r[0]) == -1:
            contexts.setdefault(r[4], []).append(r)
        elif ob.get(r[1], r[2]).retrospective:
            retrospective.append(r)
        else:
            failures.append(f"{what}: stored but not made by the rerun")
    for ident in sorted(expected, key=str):
        failures.append(f"observation {ident} of {key}: made by the rerun but not stored")
    if contexts and records.data is None:
        failures.append(f"{key}: trajectory-level values stored for a protocol without them")
    for context, rows in sorted(contexts.items()):
        try:
            spec_ctx = store.requirements(context)
            layers, domain = int(spec_ctx["layers"]), spec_ctx["domain"]
            assert records.data is not None
            wanted = [
                (r[1], r[2], r[3], ob.resolve_params(r[1], json.loads(r[3]), version=r[2]))
                for r in rows
            ]
            remade = _derive_rows(
                key, layers, domain, context, wanted,
                records.data.sources(layers, domain), level,
            )  # fmt: skip
        except (ValueError, KeyError) as exc:
            failures.append(f"{key}: trajectory-level values of context {context[:12]}: {exc}")
            continue
        for r, row in zip(rows, remade, strict=True):
            checked += 1
            what = f"observation {r[1]!r} (params {r[3]}) of {key} at t = -1 ({context[:12]})"
            try:
                _compare_observation(what, (r[5], r[6], r[7], r[8], r[9]), row)
            except ValueError as exc:
                failures.append(str(exc))
    graph_edges = graph.edges
    for r in retrospective:
        t, name, version, params = int(r[0]), r[1], r[2], r[3]
        entry = ob.get(name, version)
        resolved = ob.resolve_params(name, json.loads(params), version=version)
        snap = store.snapshot(key, t)
        got = _retrospective_value(entry, resolved, key, t, snap.status, snap,
                                   graph_edges if "graph" in entry.inputs else None)  # fmt: skip
        what = f"observation {name!r} (params {params}) of {key} at t = {t}"
        if got is None:
            row = observation_row(t, name, params, value=None, se=None, data=None, method="none",
                                  level=level, version=version, status="absent_after_purification")  # fmt: skip
        else:
            value, se, data, method, status = got
            row = observation_row(t, name, params, value=value, se=se, data=data, method=method,
                                  level=level, version=version, status=status)  # fmt: skip
        checked += 1
        try:
            _compare_observation(what, (r[5], r[6], r[7], r[8], r[9]), row, "value")
        except ValueError as exc:
            failures.append(str(exc))
    if key.protocol == "F":
        depths = [row.t for row in records.snapshots if row.status != SNAPSHOT_ABSENT]
        if depths:
            native = _native_codes(store, key, depths, num_threads)
            for row in records.snapshots:
                if row.t in native and native[row.t].digest() != row.phys_digest:
                    failures.append(
                        f"{key}: the native code_snapshot at t = {row.t} differs from the "
                        "physical code of the rerun's joint state"
                    )
    return failures, len(records.snapshots), checked


def _same(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def verify(
    store: TableauStore,
    sample: int | None = 8,
    seed: Any = 0,
    *,
    num_threads: int | None = None,
) -> Verification:
    """Run a sample of stored trajectories again and compare every record of them (v1.4 §7.4).

    ``sample`` trajectories (all of them for ``None``) are drawn without replacement from the
    stored ones in natural-key order with ``numpy.random.Generator(PCG64(seed))``. For each:

    * its graph is regenerated from its seed (same digest), and every stored snapshot is
      decoded and checked against its digest;
    * it is run again under the requirements it is complete under (the current ones, else its
      completion with the largest horizon), with the engine probes of those requirements, and
      its records are made exactly as :func:`run` makes them; every stored record of it must
      equal them: the trace at that horizon (and every shorter stored trace as its prefix),
      every snapshot (status, ``k``, both digests) and every observation (value, error,
      status, method and data), with no record missing on either side;
    * every stored trajectory-level value is derived again from the rerun under its own
      context (horizon and domain) and compared; every retrospective value (:func:`compute`)
      is recomputed from the stored snapshot and compared (value, error, status);
    * (protocol F) the engine's native ``code_snapshot`` must give the physical code of the
      rerun's joint state at every depth that has one.

    Returns a :class:`Verification` (it does not raise on a mismatch). A store of another
    randomness contract fails as a whole. Reading only: a reader store can be verified.
    """
    try:
        store.check_contract()
    except ValueError as exc:
        return Verification(ok=False, trajectories=0, snapshots=0, failures=(str(exc),))
    keys = store.trajectory_keys()
    if sample is not None and sample < len(keys):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        chosen = sorted(rng.choice(len(keys), size=int(sample), replace=False).tolist())
        keys = [keys[i] for i in chosen]
    failures: list[str] = []
    snapshots = observations = 0
    regenerated: set[tuple[int, str, int]] = set()
    revisions = store.revisions()
    currents: dict[tuple[str, int, str], Current] = {}
    for key in keys:
        try:
            gk = (key.n, key.beta.token, key.g)
            if gk not in regenerated:
                store.graph(key.n, key.beta, key.g).regenerate()
                regenerated.add(gk)
            for t in store.depths(key):
                snap = store.snapshot(key, t)
                if snap.status != SNAPSHOT_ABSENT:
                    if snap.joint_blob is not None:
                        snap.tableau()
                    if key.protocol == "F":
                        snap.physical_tableau()
            ck = (key.protocol, key.n, key.beta.token)
            if ck not in currents:
                currents[ck] = current(revisions, key.protocol, key.n, key.beta)
            found, snaps, rows = _verify_trajectory(store, key, currents[ck], num_threads)
            failures.extend(found)
            snapshots += snaps
            observations += rows
        except (ValueError, KeyError) as exc:
            failures.append(f"{key}: {exc}")
    return Verification(
        ok=not failures,
        trajectories=len(keys),
        snapshots=snapshots,
        failures=tuple(failures),
        observations=observations,
    )


# ---------------------------------------------------------------------------------------------
# provenance (G-RUN-3)
# ---------------------------------------------------------------------------------------------


def _provenance() -> dict[str, Any]:
    """The provenance of a unit: the package version, the commit and whether the checkout had
    uncommitted changes, ``RANDOMNESS_CONTRACT`` and the software environment with its digest."""
    env = environment()
    return {
        "sparsegf2_version": __version__,
        **git_provenance(),
        "randomness_contract": int(mipt.RANDOMNESS_CONTRACT),
        "environment": canonical_json(env),
        "environment_digest": environment_digest(env),
    }


def _check_environment(
    store: TableauStore, manifest: ExperimentManifest, provenance: Mapping[str, Any]
) -> None:
    """A confirmatory study runs every unit in one software environment (v1.4 §8.1: pin the
    commit and the dependencies before production): refuse a unit whose environment differs
    from the units already done, naming the differences. Exploratory studies record it only."""
    if manifest.label != "confirmatory":
        return
    rows = (
        store._db()
        .execute("SELECT DISTINCT environment, environment_digest FROM units WHERE status = 'done'")
        .fetchall()
    )
    for env_text, digest in rows:
        if digest != provenance["environment_digest"]:
            old, new = json.loads(env_text), json.loads(provenance["environment"])
            changed = sorted(k_ for k_ in set(old) | set(new) if old.get(k_) != new.get(k_))
            raise ValueError(
                f"this confirmatory study ran its units with {[(c, old.get(c)) for c in changed]}; "
                f"this process has {[(c, new.get(c)) for c in changed]}: run it in the recorded "
                "environment (or label the study exploratory)"
            )


# ---------------------------------------------------------------------------------------------
# sampler diagnostics (G-OBS-4)
# ---------------------------------------------------------------------------------------------


def _diagnostics_seed(n: int, beta: Token, g: int, probe: str, m: int) -> int:
    """The seed of the diagnostics of one sampler: the first 63 bits of the BLAKE2b of the
    canonical JSON of ``[DIAGNOSTICS_SEED_TAG, n, beta, g, probe, m]`` (chain ``i`` runs on
    ``[seed, i]``, as in :func:`sparsegf2.api.sample_erasure_regions`)."""
    text = canonical_json([DIAGNOSTICS_SEED_TAG, int(n), beta.token, int(g), probe, int(m)])
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") >> 1


def diagnose_samplers(
    store: TableauStore,
    *,
    chains: int = 4,
    samples: int | None = None,
    graphs: Sequence[int] | None = None,
    sizes: Sequence[int] | None = None,
    max_seconds: float = math.inf,
) -> int:
    """Archive the graph-side mixing diagnostics of every correlated_erasure sampler (G-OBS-4).

    The engine's correlated_erasure probe stores failing fractions only, so its sampler cannot
    be judged from the run. This runs, for each correlated entry of the latest manifest, each
    condition $(n, \\beta)$, each stored graph (``graphs``: a subset of labels; for the
    ``cycle`` and ``path`` geometries one erasure graph, stored with ``g = -1``) and each size
    (``sizes``: a subset), :func:`sparsegf2.api.sample_erasure_regions` with the probe's own
    settings (burn-in, thinning, tempering, ladder, method) and ``chains`` independent chains of
    ``samples`` draws (default the probe's, at least 8), and stores per coupling the method, the
    effective sample size, ``tau_int``, the acceptances, R-hat and its limit, and the
    ``mixing`` notes, in ``sampler_diagnostics`` keyed by the graph, the probe, a digest of its
    settings, the size and the coupling. Existing rows are compared (the seeds are derived:
    :func:`_diagnostics_seed`). Returns the number of rows written. An empty ``notes`` list is
    no proof of equilibrium.
    """
    from sparsegf2 import api
    from sparsegf2 import graphs as graphs_module

    store._require_writer()
    manifest = store.manifest
    entries = list(
        zip(
            manifest.observables.correlated_erasure, manifest.observables.probe_names(), strict=True
        )
    )
    written = 0
    for cond in manifest.conditions():
        n, beta = cond.n, cond.beta
        labels = range(manifest.trials.graphs) if graphs is None else graphs
        for entry, name in entries:
            config = json_digest({**entry.to_dict(), "name": name})
            if entry.geometry == "interaction":
                targets = [
                    (int(g), store.graph(n, beta, g).edges)
                    for g in labels
                    if store.has_graph(n, beta, g)
                ]
            else:
                maker = graphs_module.cycle if entry.geometry == "cycle" else graphs_module.path
                targets = [(-1, graphs_module.from_edges(maker(n), n))]
            wanted = entry.expand_sizes(n) if sizes is None else [m for m in sizes]
            for g, edges in targets:
                for m in wanted:
                    seed = _diagnostics_seed(n, beta, g, name, m)
                    count = max(entry.samples, 8) if samples is None else int(samples)
                    out = api.sample_erasure_regions(
                        {"edges": np.asarray(edges).tolist(), "num_nodes": n}, int(m),
                        [float(j) for j in entry.couplings], count, burn_in=entry.burn_in,
                        thin=entry.thin, tempering=entry.tempering, chains=chains, seed=seed,
                        method=entry.method,
                        ladder=entry.ladder if isinstance(entry.ladder, str) else list(entry.ladder),
                        max_step=entry.max_step, max_seconds=max_seconds,
                    )  # fmt: skip
                    notes = canonical_json(out.get("mixing") or [])
                    with store.transaction() as tx:
                        for row in out["couplings"]:
                            written += int(
                                tx.put_sampler_diagnostic(
                                    n=n,
                                    beta=beta.token,
                                    g=g,
                                    probe=name,
                                    config=config,
                                    m=int(m),
                                    coupling=float(row["coupling"]),
                                    method=out["method"],
                                    chains=chains,
                                    samples=count,
                                    seed=canonical_json(seed),
                                    ess=row["ess"],
                                    tau_int=row["tau_int"],
                                    acceptance=row["acceptance"],
                                    exchange_acceptance=row["exchange_acceptance"],
                                    r_hat=row["r_hat"],
                                    r_hat_limit=row["r_hat_limit"],
                                    notes=notes,
                                    data={
                                        k_: out.get(k_)
                                        for k_ in (
                                            "burn_in",
                                            "thin",
                                            "tempering",
                                            "ladder",
                                            "nonincreasing_in_J",
                                        )
                                    },
                                )  # fmt: skip
                            )
    return written
