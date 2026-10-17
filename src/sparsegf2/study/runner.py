"""The experiment runner of the Watts–Strogatz study (v1.4 §5.3, §7.2–§7.4, §8; design §4, §12).

The runner turns the manifest revisions of a :class:`~sparsegf2.study.store.TableauStore` into
stored records, through the native engine only (``mipt.MonitoredCircuit.run``):

* :func:`requirements` says what a complete trajectory of a condition records under the latest
  revision: its horizon $T_{\\max}$, the effective checkpoint schedule $E_r$ (ruling R-CK), the
  engine probes with their settings and stable names, and the peak rule. Its SHA-256 (canonical
  JSON) names it.
* :func:`plan` lists the work units still to do: for every protocol, condition and rate, the
  trajectories $(g, c)$ with $g < G_*$, $c < C$ that have no completion under the current
  requirements, grouped into rectangles of graphs and circuits sized by the manifest's
  ``storage`` and ``resources`` (and the engine's snapshot memory). The order is deterministic.
* :func:`run` executes units: one engine call per unit on the trajectory indices
  ``u = g * stride + c`` (graph collection ``0 .. g1 - 1`` from the store, digest-checked), then
  every record of every trajectory of the unit (graphs, trajectory, trace, snapshots,
  observations, completion and the unit itself) in **one** transaction. A crash before the
  commit leaves nothing; :func:`run` again re-plans and redoes the unit.
* :func:`extend` adds a revision; the plan then holds the new conditions, rates, graphs and
  circuits, and reruns the trajectories whose requirements changed (a larger horizon, new
  checkpoints, a new observable) from layer 0, every stored record being reproduced and
  compared exactly before the new ones are added (design D3).
* :func:`replay`, :func:`replay_signed`, :func:`circuit`: one trajectory again, from its stored
  identity: the phase-free tableau at any depth (checked against the stored digests), the
  outcome-aware signed run on the exported circuit, the exported circuit itself.
* :func:`compute`: a retrospective snapshot observable of the catalog over the stored snapshots,
  in batches, without simulation; :func:`verify`: replay a sample and compare digests.

``docs/study/05-store-and-reproducibility.md`` has the definitions and proofs.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numba
import numpy as np

from sparsegf2 import analysis, mipt
from sparsegf2._parallel import threads
from sparsegf2._version import __version__
from sparsegf2.circuit import Circuit
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
    TableauStore,
    _now,
    environment,
    environment_digest,
    git_provenance,
    graph_measures,
)
from sparsegf2.tableau import PhaseFreeTableau, physical_tableaux, to_bytes_batch

__all__ = [
    "DEFAULT_UNIT_BYTES",
    "PROBE_NAMES",
    "ComputeSummary",
    "Replay",
    "RunSummary",
    "SignedReplay",
    "Verification",
    "WorkUnit",
    "circuit",
    "compute",
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

#: The memory budget of one engine call when ``resources.max_unit_bytes`` is not set: the
#: snapshot buffer and the probe values held until the call returns.
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
# requirements
# ---------------------------------------------------------------------------------------------


def requirements(
    revisions: Sequence[ExperimentManifest], protocol: str, n: int, beta: Any
) -> dict[str, Any]:
    """What a complete trajectory of ``(protocol, n, beta)`` records under the last revision.

    A plain dictionary (its canonical JSON is hashed by :func:`requirements_digest`):
    ``protocol``, ``n``, ``beta`` (token), ``layers`` ($T_{\\max}$), ``checkpoints`` (the
    effective schedule $E_r$ of ruling R-CK, every probe's depths), ``dynamics`` (the dynamics
    version), and for protocol F ``stop_when_purified``, ``snapshot`` (``"tableau_snapshot"``,
    the joint state and its physical code, or ``"code_snapshot"``, the physical code only),
    ``contiguous_distance``, ``uniform_erasure`` (samples and exact budget, or None),
    ``correlated_erasure`` (each entry's settings with its probe name), ``peak`` (the
    predeclared peak rule of ``analysis.peak``, v1.4 §5.8) and ``domain``: the depths over which
    peaks and the checkpoint bounds of the first passages are taken, the last revision's own
    schedule $S_r(n, \\beta)$ (G-OBS-2: a store built through revisions and a store made
    directly under the last manifest give the same peaks; $S_r \\subseteq E_r$ is stored). The
    rate is not part of it: every rate of a condition records the same things. The code
    dimension is always recorded at every layer (F).
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
        peak=latest.analysis.peak.to_dict(),
        domain=list(latest.checkpoints(n, b, protocol)),
    )
    return out


def requirements_digest(req: Mapping[str, Any]) -> str:
    """The SHA-256 of the canonical JSON of :func:`requirements`."""
    return json_digest(dict(req))


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
        settings = {k: v for k, v in entry.items() if k != "name"}
        probes.append(
            CorrelatedErasure.from_dict(settings, "correlated_erasure").probe(
                name=entry["name"], checkpoints=cps
            )
        )
    return probes


def _bytes_per_trajectory(req: Mapping[str, Any]) -> int:
    """The engine memory one trajectory holds until its call returns: the snapshot rows of
    every checkpoint, ``8 E N 2 ceil(N / 64)`` bytes (``N`` the qubits of the snapshot), and
    the probe values, 8 bytes each (the code dimension at every layer, the samples, counts and
    correlated fractions at every checkpoint)."""
    n = int(req["n"])
    evaluations = len(req["checkpoints"])
    joint = req["protocol"] == "F" and req["snapshot"] == "tableau_snapshot"
    qubits = 2 * n if joint else n
    snap = evaluations * qubits * 2 * ((qubits + 63) // 64) * 8
    values = 0
    if req["protocol"] == "F":
        values += int(req["layers"]) + 1
        values += evaluations * (2 + int(bool(req["contiguous_distance"])))
        uniform = req["uniform_erasure"]
        if uniform is not None:
            values += evaluations * int(uniform["samples"])
            if uniform["exact_budget"]:
                values += evaluations * analysis.exhaustive_cutoff(n, int(uniform["exact_budget"]))
        for entry in req["correlated_erasure"]:
            sizes = CorrelatedErasure.from_dict(
                {k: v for k, v in entry.items() if k != "name"}, "correlated_erasure"
            ).expand_sizes(n)
            values += evaluations * len(entry["couplings"]) * len(sizes)
    return snap + 8 * values + 256


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
    :func:`requirements` (new ones, and stored ones whose requirements changed). They are
    grouped into rectangles: for each run of consecutive missing circuits of a graph, graphs
    with the same run and consecutive labels go together, at most ``storage.graphs_per_unit``
    graphs, ``resources.max_unit_trajectories`` trajectories and ``resources.max_unit_bytes``
    (default :data:`DEFAULT_UNIT_BYTES`) of engine memory per unit (a graph's circuits are split
    when even one graph exceeds them). A unit holds either stored trajectories run again
    (``rerun``: what a revision added, G-RUN-2) or new ones, and the reruns come first, so the
    stored trajectories receive the new depths before any new trajectory is added. Then units
    are sorted by protocol, ``n``, ``beta``, ``p``, first graph and first circuit. The three
    keywords override the manifest's sizes for this plan only; how trajectories are split into
    units never changes a stored value.
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
    """What :func:`run` did: units, trajectories and snapshots written, time, units left."""

    units: int
    trajectories: int
    snapshots: int
    elapsed_seconds: float
    remaining_units: int


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
    result. ``max_units`` stops after that many units (a resumable partial run). ``progress``
    is called after each unit as ``progress(unit, index, total, info)``. A unit's records are
    committed together; an exception rolls the unit back, records it as failed (with the error,
    and no data) and is raised again. Running again re-plans: units already done are skipped.
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
    remaining = len(plan(store))
    return RunSummary(
        units=len(todo),
        trajectories=trajectories,
        snapshots=snapshots,
        elapsed_seconds=time.perf_counter() - start,
        remaining_units=remaining,
    )


def _unit_spec(
    manifest: ExperimentManifest,
    req: Mapping[str, Any],
    unit_protocol: str,
    n: int,
    beta: Token,
    p: Token,
    collection: Sequence[np.ndarray],
    *,
    probes: Sequence[mipt.Probe] | None = None,
    layers: int | None = None,
    stop: bool | None = None,
) -> mipt.MonitoredCircuit:
    kwargs = manifest.monitored_circuit_kwargs(n, beta, p, unit_protocol)
    if layers is not None:
        kwargs["layers"] = int(layers)
    if stop is not None:
        kwargs["stop_when_purified"] = bool(stop)
    return mipt.MonitoredCircuit(
        edges=cast(Any, list(collection)),
        probes=list(engine_probes(req) if probes is None else probes),
        **kwargs,
    )


def _run_unit(
    store: TableauStore, unit: WorkUnit, cache: _GraphCache, num_threads: int | None
) -> dict[str, Any]:
    revisions = store.revisions()
    manifest = revisions[-1]
    req = requirements(revisions, unit.protocol, unit.n, unit.beta)
    digest = requirements_digest(req)
    if digest != unit.requirements or unit.revision != len(revisions):
        raise ValueError(
            f"unit {unit} was planned under other requirements or another revision: plan again "
            "(runner.plan) after a revision"
        )
    provenance = _provenance()
    _check_environment(store, manifest, provenance)
    n, beta, p = unit.n, unit.beta, unit.p
    collection = [cache.get(n, beta, g) for g in range(unit.graphs[1])]
    new_graphs = [(n, beta, g) for g in range(unit.graphs[1]) if (n, beta, g) in cache.new]
    spec = _unit_spec(manifest, req, unit.protocol, n, beta, p, [r.edges for r in collection])
    indices = unit.indices(manifest.stride)
    seed = manifest.run_seed(n, beta, p)
    started = _now()
    clock = time.perf_counter()
    with threads(num_threads):
        used_threads = numba.get_num_threads()
    result = spec.run(indices=indices, seed=list(seed), num_threads=num_threads)
    canon = _canonical_snapshots(result, req, n, num_threads)
    snapshots = 0
    samplers: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        with store.transaction() as tx:
            for key in new_graphs:
                record = cache.records[key]
                tx.put_graph(record, graph_measures(record.edges, record.n))
            tx.put_requirements(digest, canonical_json(req))
            unit_id = tx.add_unit(
                protocol=unit.protocol,
                n=n,
                beta=beta.token,
                p=p.token,
                graphs=canonical_json(list(unit.graphs)),
                circuits=canonical_json(list(unit.circuits)),
                layers=unit.layers,
                requirements=digest,
                status="done",
                revision=unit.revision,
                started_utc=started,
                threads=used_threads,
                trajectories=unit.trajectories,
                **provenance,
            )
            for row, u in enumerate(indices.tolist()):
                snapshots += _ingest(
                    tx,
                    store,
                    manifest,
                    unit,
                    req,
                    digest,
                    spec,
                    result,
                    row,
                    int(u),
                    collection,
                    unit_id,
                    samplers,
                    canon,
                )
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
        "engine_seconds": result.elapsed_seconds,
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


def _ingest(
    tx: Any,
    store: TableauStore,
    manifest: ExperimentManifest,
    unit: WorkUnit,
    req: Mapping[str, Any],
    digest: str,
    spec: mipt.MonitoredCircuit,
    result: mipt.EnsembleResult,
    row: int,
    u: int,
    collection: Sequence[GraphRecord],
    unit_id: int,
    samplers: dict[tuple[str, str], dict[str, Any]],
    canon: Sequence[tuple[list[PhaseFreeTableau | None], list[PhaseFreeTableau | None]]],
) -> int:
    """Derive, validate and write every record of one trajectory (row ``row`` of ``result``)."""
    n, beta, p = unit.n, unit.beta, unit.p
    stride = manifest.stride
    g, c = divmod(u, stride)
    key = TrajectoryKey(store.experiment_id, n, beta, p, g, c, 0, unit.protocol)
    revision = unit.revision
    layers = int(req["layers"])
    cps = np.array(req["checkpoints"], dtype=np.int64)
    tid = tx.put_trajectory(
        key,
        u=u,
        p_float_hex=p.hex,
        run_seed=manifest.run_seed(n, beta, p),
        graph_digest=collection[g].edge_digest,
        dynamics_version=req["dynamics"],
        unit_id=unit_id,
        revision=revision,
    )
    final = int(result.final_layer[row])
    measurements = int(result.num_measurements[row])
    snap_name = PROBE_NAMES[req["snapshot"]]
    words, counts = result.snapshots[snap_name]
    if unit.protocol == "P":
        if final != layers:
            raise ValueError(f"{key}: a protocol-P trajectory must run to its horizon {layers}")
        tx.put_trace(
            tid, key, layers=layers, final_layer=final, k=None, tau=None, status="no_reference",
            num_measurements=measurements, unit_id=unit_id, revision=revision,
        )  # fmt: skip
        for j, t in enumerate(cps.tolist()):
            rows = int(counts[row, j])
            if rows < 0:
                raise ValueError(f"{key}: protocol P never stops, but t = {t} has no snapshot")
            state = canon[j][0][row]
            assert state is not None
            tx.put_snapshot(
                tid, key, t, status=SNAPSHOT_STORED, num_qubits=n, k=None, joint=state,
                phys=None, unit_id=unit_id,
            )  # fmt: skip
        tx.put_completion(tid, digest, unit_id, revision)
        return len(cps)

    # protocol F: the k trace at every layer, exact first passages, consistency with the engine
    raw_k = np.asarray(result.values[PROBE_NAMES["code_dimension"]][row], dtype=np.float64)
    if raw_k.shape != (layers + 1,) or not np.all(np.isfinite(raw_k)):
        raise ValueError(f"{key}: the code dimension must be recorded at every layer 0..{layers}")
    k = raw_k.astype(np.int64)
    if k[0] != n or np.any(k != raw_k):
        raise ValueError(f"{key}: k_0 must be n = {n} (n Bell pairs), got {raw_k[0]}")
    fp = ob.first_passage(k, cps, num_system=n)  # refuses an increasing trace
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
    tx.put_trace(
        tid, key, layers=layers, final_layer=final, k=k, tau=fp.tau, status=status,
        num_measurements=measurements, unit_id=unit_id, revision=revision,
    )  # fmt: skip
    k_at = k[cps]
    joint_qubits = 2 * n
    tableau_kind = req["snapshot"] == "tableau_snapshot"
    for j, t in enumerate(cps.tolist()):
        rows = int(counts[row, j])
        if rows < 0:
            if not (stop and t > final):
                raise ValueError(f"{key}: no snapshot at t = {t}, but the trajectory ran there")
            tx.put_snapshot(
                tid, key, t, status=SNAPSHOT_ABSENT, num_qubits=joint_qubits, k=0, joint=None,
                phys=None, unit_id=unit_id,
            )  # fmt: skip
            continue
        joint, phys = canon[j][0][row], canon[j][1][row]
        if (joint is None) == tableau_kind or phys is None:
            raise ValueError(f"{key}: the snapshot at t = {t} was not canonicalized")
        tx.put_snapshot(
            tid, key, t, status=SNAPSHOT_STORED if tableau_kind else SNAPSHOT_PHYS_ONLY,
            num_qubits=joint_qubits, k=int(k[t]), joint=joint, phys=phys, unit_id=unit_id,
            check_physical=False,
        )  # fmt: skip

    # observations at the checkpoints
    def put(t: int, name: str, params: Mapping[str, Any] | None, **fields: Any) -> None:
        tx.put_observation(tid, key, t, name, params, unit_id=unit_id, **fields)

    for t in cps.tolist():
        put(t, "k", None, value=int(k[t]), se=0.0, data=None, method="k-trace", status="ok")
    context = digest
    # the domain of peaks and checkpoint bounds: the last revision's own schedule (G-OBS-2)
    domain = np.array(req["domain"], dtype=np.int64)
    on = np.isin(cps, domain)
    dom_k = k_at[on]
    dom_info = {"domain": domain, "schedule_digest": json_digest([int(t) for t in domain])}
    fp_dom = ob.first_passage(k, domain, num_system=n)
    put(-1, "first_passage", None, value=_tau0(fp), se=0.0,
        data={**fp_dom.to_dict(), **dom_info}, method="k-trace", context=context,
        status="ok" if fp.observed[0] else "censored")  # fmt: skip
    peak_rule = req["peak"]
    if req["contiguous_distance"]:
        raw = np.asarray(result.values["contiguous_distance"][row], dtype=np.float64)
        cd = ob.cyclic_distance(raw, k_at, num_system=n)
        for j, t in enumerate(cps.tolist()):
            put(t, "d_cyc", None, value=raw[j], se=0.0, method="engine-probe",
                status="ok" if cd.has_code[j] else "undefined_no_code",
                data={"has_code": bool(cd.has_code[j]), "zero_extended": int(cd.zero_extended[j])})  # fmt: skip
        pk = ob.peak(domain, cd.zero_extended[on], rule="exact", k_at=dom_k, horizon=layers)
        put(-1, "peak_d_cyc", {"rule": "exact"}, value=pk.t_star, se=0.0,
            data={**pk.to_dict(), **dom_info}, method="exact", context=context,
            status="missing" if pk.missing else "ok")  # fmt: skip
    uniform = req["uniform_erasure"]
    if uniform is not None:
        ff = np.asarray(result.values[PROBE_NAMES["erasure_first_failure"]][row])
        exact = None
        if uniform["exact_budget"]:
            exact = np.asarray(result.values[PROBE_NAMES["uncorrectable_region_counts"]][row])
        profile = ob.uniform_failure(ff, n, exact_counts=exact, k=k_at)
        robust = ob.integrated_robustness(ff, n, exact_counts=exact, k=k_at)
        method = "hybrid" if exact is not None else "orderings"
        params = {"samples": uniform["samples"], "exact_budget": uniform["exact_budget"]}
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
        plateau = _peak_params("peak_D", peak_rule)
        # D(0) from the orderings: its se are the true standard errors (se_exact True)
        pk = ob.peak(domain, robust.value[on], robust.se[on], rule="plateau", z=peak_rule["z"],
                     k_at=dom_k, horizon=layers, se_exact=True)  # fmt: skip
        data = {**pk.to_dict(), **dom_info}
        data["sensitivity"] = [
            s.to_dict()
            for s in ob.peak_sensitivity(
                domain,
                robust.value[on],
                robust.se[on],
                peak_rule["sensitivity_z"],
                k_at=dom_k,
                horizon=layers,
                se_exact=True,
            )  # fmt: skip
        ]
        data["bootstrap"] = None
        if peak_rule["bootstrap"] > 0:
            entry = ob.get("peak_D")
            # the catalog's seed (Proposition 18(d) of study/04): the key, id and version only
            seed = ob.bootstrap_seed(key, "peak_D", entry.version)
            boot = ob.bootstrap_peak(
                domain, ff[on], n, z=peak_rule["z"], replicates=peak_rule["bootstrap"],
                seed=list(seed), exact_counts=None if exact is None else exact[on], k_at=dom_k,
                horizon=layers,
            )  # fmt: skip
            lo, hi = boot.interval(0.95)
            finite = boot.t_star[np.isfinite(boot.t_star)]
            data["bootstrap"] = {
                "replicates": boot.replicates, "seed": list(seed), "interval95": [lo, hi],
                "missing_fraction": boot.missing_fraction,
                "horizon_fraction": boot.horizon_fraction,
                "sd": float(finite.std(ddof=1)) if finite.size > 1 else None,
            }  # fmt: skip
        put(-1, "peak_D", plateau, value=pk.t_star, se=None, data=data, method="plateau",
            context=context, status="missing" if pk.missing else "ok")  # fmt: skip
    for entry in req["correlated_erasure"]:
        name = entry["name"]
        couplings, sizes, values = result.correlated(name)
        prof = ob.correlated_failure(
            values[row], couplings, sizes, n, k_at, samples=entry["samples"],
            sampler=_graph_sampler(samplers, entry, collection[g]),
        )  # fmt: skip
        method = str(prof.sampler["method"]) if prof.sampler else "sampled"
        for j, t in enumerate(cps.tolist()):
            put(t, "correlated_failure", entry, value=None, se=None, method=method, status="vector", data={
                "couplings": couplings, "sizes": sizes, "failure": prof.failure[j],
                "se": prof.se[j], "interpolated_sizes": prof.interpolated_sizes,
                "sampler": dict(prof.sampler or {}),
            })  # fmt: skip
            put(t, "D_J", entry, value=None, se=None, method=method, status="vector", data={
                "couplings": couplings, "robustness": prof.robustness[j],
                "robustness_se": prof.robustness_se[j], "complete": prof.complete,
                "has_code": bool(prof.has_code[j]),
            })  # fmt: skip
        for i, coupling in enumerate(couplings.tolist()):
            params = _peak_params("peak_D_J", peak_rule, J=float(coupling))
            rob, rob_se = prof.robustness[on, i], prof.robustness_se[on, i]
            # the chain's binomial se are lower bounds: the record says so (se_exact False)
            pk = ob.peak(domain, rob, rob_se, rule="plateau", z=peak_rule["z"], k_at=dom_k,
                         horizon=layers, se_exact=prof.se_exact)  # fmt: skip
            data = {**pk.to_dict(), **dom_info}
            data["sensitivity"] = [
                s.to_dict()
                for s in ob.peak_sensitivity(
                    domain,
                    rob,
                    rob_se,
                    peak_rule["sensitivity_z"],
                    k_at=dom_k,
                    horizon=layers,
                    se_exact=prof.se_exact,
                )  # fmt: skip
            ]
            data["probe"] = name
            put(-1, "peak_D_J", params, value=pk.t_star, se=None, data=data, method="plateau",
                context=f"{context}/{name}", status="missing" if pk.missing else "ok")  # fmt: skip
    tx.put_completion(tid, digest, unit_id, revision)
    return len(cps)


def _canonical_snapshots(
    result: mipt.EnsembleResult,
    req: Mapping[str, Any],
    n: int,
    num_threads: int | None,
) -> list[tuple[list[PhaseFreeTableau | None], list[PhaseFreeTableau | None]]]:
    """Every snapshot of a unit in canonical form: per evaluation, the joint tableaux and the
    physical codes of all its trajectories (``None`` where a stopped trajectory saved nothing,
    and for the parts a snapshot kind does not have), with their canonical bytes.

    The compiled batch routines of :mod:`sparsegf2.tableau` (``from_words_batch``,
    ``physical_tableaux``, ``to_bytes_batch``) run in parallel over the trajectories and give
    exactly the objects and bytes of the one-at-a-time methods.
    """
    words, counts = result.snapshots[PROBE_NAMES[req["snapshot"]]]
    rows = counts.shape[0]
    out: list[tuple[list[PhaseFreeTableau | None], list[PhaseFreeTableau | None]]] = []
    for e in range(counts.shape[1]):
        joints: list[PhaseFreeTableau | None]
        physs: list[PhaseFreeTableau | None]
        if req["snapshot"] == "tableau_snapshot":
            qubits = n if req["protocol"] == "P" else 2 * n
            joints = PhaseFreeTableau.from_words_batch(
                words[:, e], counts[:, e], qubits, num_threads=num_threads
            )
            physs = (
                physical_tableaux(joints, n, num_threads=num_threads)
                if req["protocol"] == "F"
                else [None] * rows
            )
        else:
            joints = [None] * rows
            physs = PhaseFreeTableau.from_words_batch(
                words[:, e], counts[:, e], n, num_threads=num_threads
            )
        to_bytes_batch(joints, num_threads=num_threads)
        to_bytes_batch(physs, num_threads=num_threads)
        out.append((joints, physs))
    return out


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
        settings = {k: v for k, v in entry.items() if k != "name"}
        probe = CorrelatedErasure.from_dict(settings, "correlated_erasure").probe(name=name)
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


def _peak_params(observable: str, rule: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    """The parameters of a peak observable: the manifest's peak rule restricted to the
    parameters the catalog entry has (``analysis.peak``), plus ``extra``."""
    names = ob.get(observable).params
    return {**{k: v for k, v in rule.items() if k in names}, **extra}


def _tau0(fp: ob.FirstPassage) -> float:
    return float(fp.tau[0]) if fp.observed[0] else math.nan


# ---------------------------------------------------------------------------------------------
# revisions
# ---------------------------------------------------------------------------------------------


def extend(store: TableauStore, new_manifest: ExperimentManifest) -> list[WorkUnit]:
    """Add ``new_manifest`` as the store's next revision and return the new plan.

    The revision must extend the latest one (:func:`~sparsegf2.study.manifest.check_extension`).
    The plan then holds the trajectories of new conditions, rates, graphs, circuits and
    protocols, and every stored trajectory whose requirements changed (a larger horizon,
    checkpoints added by a new rule, a new observable): those are rerun from layer 0 (the native
    stream has no mid-trajectory state, design D3) and every stored record (the k trace prefix,
    each snapshot digest, each value) must be reproduced exactly before the new ones are added.
    Nothing runs until :func:`run`.
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
    req = requirements(store.revisions(), k.protocol, k.n, k.beta)
    probes: list[mipt.Probe] = [mipt.tableau_snapshot("replay", checkpoints=wanted)]
    if k.protocol == "F":
        probes.insert(0, mipt.code_dimension())
    spec = _unit_spec(
        manifest, req, k.protocol, k.n, k.beta, k.p, collection, probes=probes,
        layers=max(wanted), stop=False,
    )  # fmt: skip
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
    req = requirements(store.revisions(), k.protocol, k.n, k.beta)
    spec = _unit_spec(
        manifest, req, k.protocol, k.n, k.beta, k.p, collection, probes=[], layers=horizon,
        stop=False,
    )  # fmt: skip
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
    ``stop_when_purified`` stop has no state: it gets a row of status
    ``absent_after_purification`` (value NULL; SI-05: missing, never zero), unless the run
    already stored a value there. A value already stored under that key (by an earlier call, or
    by the run for ``k`` and ``d_cyc``, which it records from the engine) is compared, never
    replaced: the recomputed value, error and status must equal it exactly, else
    ``ValueError``; so the call is idempotent, and on the run's values it is a cross-check of
    the stored snapshots. Nothing is simulated: the circuits' random numbers are never drawn
    (the sampled ``mean_leakage`` seeds its regions from the key,
    :func:`~sparsegf2.study.observables.region_seed`).
    """
    entry = ob.get(observable_id, version)
    if not entry.retrospective:
        raise ValueError(
            f"observable {entry.id!r} is computed from {list(entry.inputs)} when the trajectory "
            "is run, not from a stored snapshot; retrospective ones are "
            f"{sorted(r['id'] for r in ob.catalog() if r['retrospective'])}"
        )
    resolved = ob.resolve_params(entry.id, params, version=entry.version)
    text = ob.params_key(entry.id, resolved, version=entry.version)
    needs_joint = "joint_tableau" in entry.inputs
    needs_graph = "graph" in entry.inputs
    inserted = existing = count = 0
    graph_cache: dict[tuple[int, str, int], np.ndarray] = {}
    batch: list[tuple[TrajectoryKey, int, ob.ObservableValue | None]] = []

    def flush() -> None:
        nonlocal inserted, existing
        if not batch:
            return
        with store.transaction() as tx:
            for key, depth, value in batch:
                tid = store._require_tid(key)
                if value is None:  # absent after a stop
                    if store._has_observation(tid, depth, entry.id, entry.version, text):
                        existing += 1
                        continue
                    tx.put_observation(
                        tid, key, depth, entry.id, resolved, version=entry.version, value=None,
                        se=None, data=None, method="none", status="absent_after_purification",
                    )  # fmt: skip
                    inserted += 1
                    continue
                new = tx.put_observation(
                    tid, key, depth, entry.id, resolved, version=entry.version,
                    value=value.value, se=value.se, data=dict(value.data), method=value.method,
                    compare="value", status=_value_status(entry.id, value),
                )  # fmt: skip
                inserted += int(new)
                existing += int(not new)
        batch.clear()

    for snap in store.iter_snapshots(
        protocol=protocol, n=n, beta=beta, p=p, t=t, batch_size=batch_size, status=None
    ):
        key = snap.key
        count += 1
        if snap.status == SNAPSHOT_ABSENT:
            batch.append((key, snap.t, None))
        else:
            edges = None
            if needs_graph:
                gk = (key.n, key.beta.token, key.g)
                if gk not in graph_cache:
                    graph_cache[gk] = store.graph(key.n, key.beta, key.g).edges
                edges = graph_cache[gk]
            joint = snap.tableau() if needs_joint or key.protocol != "F" else None
            code = snap.code() if key.protocol == "F" and not needs_joint else None
            snapshot = ob.SnapshotInput(
                num_system=key.n, t=snap.t, key=key, code=code, tableau=joint, edges=edges
            )
            value = ob.compute(entry.id, snapshot, resolved, version=entry.version)
            batch.append((key, snap.t, value))
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
    """The result of :func:`verify`: ``ok`` when nothing failed; the counts checked; one message
    per failure."""

    ok: bool
    trajectories: int
    snapshots: int
    failures: tuple[str, ...] = field(default_factory=tuple)


def _native_codes(
    store: TableauStore, key: TrajectoryKey, depths: Sequence[int], num_threads: int | None
) -> dict[int, PhaseFreeTableau]:
    """The physical codes of a trajectory at ``depths`` from the engine's own ``code_snapshot``
    (v1.4 §9.2 (g), (h)): the native route, against which the stored codes (derived from the
    joint state, or stored from this probe) are checked."""
    rec, manifest, collection = _trajectory_context(store, key)
    req = requirements(store.revisions(), key.protocol, key.n, key.beta)
    spec = _unit_spec(
        manifest, req, key.protocol, key.n, key.beta, key.p, collection,
        probes=[mipt.code_snapshot("native", checkpoints=list(depths))], layers=max(depths),
        stop=False,
    )  # fmt: skip
    result = spec.run(indices=[rec.u], seed=list(rec.run_seed), num_threads=num_threads)
    words, counts = result.snapshots["native"]
    return {
        t: PhaseFreeTableau.from_words(words[0, j, : counts[0, j]], key.n)
        for j, t in enumerate(depths)
    }


def verify(
    store: TableauStore,
    sample: int | None = 8,
    seed: Any = 0,
    *,
    num_threads: int | None = None,
) -> Verification:
    """Replay a sample of stored trajectories and compare everything with the store (v1.4 §7.4).

    ``sample`` trajectories (all of them for ``None``) are drawn without replacement from the
    stored ones in natural-key order with ``numpy.random.Generator(PCG64(seed))``. For each:
    its graph is regenerated from its seed (same digest), every stored snapshot is decoded and
    checked against its digest, :func:`replay` must reproduce the stored k trace and every
    stored digest, and (protocol F) the engine's native ``code_snapshot`` must give the stored
    physical code at every stored depth. Returns a :class:`Verification` (it does not raise on
    a mismatch). A store of another randomness contract fails as a whole.
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
    snapshots = 0
    regenerated: set[tuple[int, str, int]] = set()
    for key in keys:
        try:
            gk = (key.n, key.beta.token, key.g)
            if gk not in regenerated:
                store.graph(key.n, key.beta, key.g).regenerate()
                regenerated.add(gk)
            stored = []
            for t in store.depths(key):
                snap = store.snapshot(key, t)
                if snap.status != SNAPSHOT_ABSENT:
                    if snap.joint_blob is not None:
                        snap.tableau()
                    if key.protocol == "F":
                        snap.physical_tableau()
                    stored.append(t)
                    snapshots += 1
            replay(store, key, stored or [0], verify=True, num_threads=num_threads)
            if key.protocol == "F" and stored:
                native = _native_codes(store, key, stored, num_threads)
                for t in stored:
                    if native[t].digest() != store.snapshot(key, t).phys_digest:
                        raise ValueError(
                            f"the native code_snapshot at t = {t} differs from the stored "
                            "physical code"
                        )
        except (ValueError, KeyError) as exc:
            failures.append(f"{key}: {exc}")
    return Verification(
        ok=not failures, trajectories=len(keys), snapshots=snapshots, failures=tuple(failures)
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
