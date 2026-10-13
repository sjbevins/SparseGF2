"""Ensembles of random monitored Clifford circuits, at full speed.

Describe the protocol once with :class:`MonitoredCircuit`, choose what to record
with probes, and run as many trajectories as you need::

    from sparsegf2_v3 import graphs, mipt

    spec = mipt.MonitoredCircuit(
        num_system=128,
        edges=graphs.watts_strogatz(128, k=4, beta=0.1, seed=1),
        layers=1024,
        measurement_rate=0.1,
        schedule="random_edges",          # 64 random edges per layer (default n // 2)
        reference="single",               # one reference qubit ...
        reference_partner="random",       # ... Bell-paired with a random system qubit
        scramble_gates=256,               # random all-to-all gates before the dynamics
        checkpoints="every_layer",
        probes=[mipt.code_dimension()],
        stop_when_purified=True,
    )
    result = spec.run(trajectories=100_000, seed=7)   # parallel over all cores
    result.purification_layer                          # tau per trajectory (-1 = censored)
    layers, surviving = result.survival()              # P(tau > layer)
    spec.circuit(trajectory=3, seed=7)                 # the exact circuit of trajectory 3

The protocol, step by step (qubits ``0..n-1`` are the system; references
follow):

1. **Setup.** ``reference="purification"``: system qubit ``i`` is Bell-paired
   with reference ``n + i`` (the maximally mixed initial state). ``"single"``:
   one reference ``n`` is Bell-paired with ``reference_partner``. ``"none"``:
   just |0...0>.
2. **Scramble** (optional). ``scramble_gates`` random two-qubit gates on
   uniformly random distinct pairs of system qubits, then
   ``scramble_matchings`` layers of gates on uniformly random perfect
   matchings of the system (all-to-all), then ``scramble_layers`` gate-only
   layers of the dynamics below.
3. **Dynamics**, ``layers`` times. Gates on edges of the graph, chosen by
   ``schedule``:

   * ``"matching"``: cycle through an edge colouring of the graph (on an even
     ring this is the standard brickwork, alternating even and odd bonds);
   * ``"random_edges"``: ``gates_per_layer`` edges drawn with replacement,
     uniformly or with probabilities ``edge_weights`` (long-range circuits:
     ``graphs.complete(n)`` with ``graphs.power_law_weights``);
   * ``"all_edges"``: every edge, in order.

   Each gate is a uniformly random two-qubit Clifford (or uniform over
   ``gate_set``). Then each system qubit is measured in the Z basis with
   probability ``measurement_rate``.
4. **Checkpoints.** After the layers listed in ``checkpoints`` (layer 0 is the
   state after setup and scramble), every probe is evaluated. With a reference,
   the first checkpoint at which the reference is pure (code dimension 0) is
   the purification layer; ``stop_when_purified`` ends the trajectory there.

A "layer" is exactly one step 3, never rescaled. Everything runs phase-free:
entanglement does not depend on measurement outcomes.

Graph disorder: ``edges`` may be a list of graphs (a graph bank); trajectory
``t`` then runs on graph ``t mod len(edges)``, so one realization per
trajectory is ``edges=[graphs.watts_strogatz(n, 4, beta, seed=(17, i)) for i in
range(T)]``. Build the graphs with their own seed: graph disorder never draws
from the circuit's random streams.

Randomness: trajectory ``t`` of ``run(..., seed=s)`` is a deterministic function
of ``(spec, s, t)``, independent of the thread count and of the other
trajectories, so ensembles can be split into chunks (``first_trajectory``) and
merged without changing any result. The measurement mask uses one uniform draw per
qubit per layer whatever the rate, so runs at different ``measurement_rate``
with the same seed share their random numbers (the measured set at a lower rate
is a subset of that at a higher rate).
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from sparsegf2_v3._kernel import ensemble as _ens
from sparsegf2_v3._kernel.tables import ANF1, ANF2, IDENTITY_SYM1, IDENTITY_SYM2, LIN1, LIN2
from sparsegf2_v3._parallel import threads
from sparsegf2_v3.circuit import Circuit
from sparsegf2_v3.clifford import num_cliffords
from sparsegf2_v3.gates import gate
from sparsegf2_v3.graphs import edge_coloring, from_edges

__all__ = [
    "EnsembleResult",
    "MonitoredCircuit",
    "Probe",
    "code_dimension",
    "contiguous_distance",
    "entropy",
    "entropy_profile",
    "erasure_correctable_fraction",
    "erasure_correctable_profile",
    "erasure_lost_information",
    "erasure_lost_profile",
    "half_cut_entropy",
    "lost_information",
    "mutual_information",
    "tripartite_information",
]

_SCHEDULES = {
    "matching": _ens.SCHED_MATCHING,
    "random_edges": _ens.SCHED_RANDOM_EDGES,
    "all_edges": _ens.SCHED_ALL_EDGES,
}
_REFERENCES = {
    "none": _ens.REF_NONE,
    "purification": _ens.REF_PURIFICATION,
    "single": _ens.REF_SINGLE,
}
_H_INDEX = gate("H").clifford.index()
_CX_INDEX = gate("CX").clifford.index()


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """Something to record at every checkpoint. Build one with the functions below.

    Scalar probes record one number per checkpoint; vector probes (profiles)
    record ``num_system + 1`` numbers, indexed by region size.
    """

    kind: str
    name: str
    regions: tuple[tuple[int, ...], ...] = ()
    size: int = 0
    samples: int = 0

    @property
    def is_vector(self) -> bool:
        return self.kind in _VECTOR_KINDS


def _region(qubits: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(q) for q in qubits)


def entropy(region: Sequence[int], name: str | None = None) -> Probe:
    """Entanglement entropy S(A) of the given qubits (system and/or reference)."""
    r = _region(region)
    return Probe("entropy", name or f"entropy{list(r)}", (r,))


def half_cut_entropy() -> Probe:
    """S of the first half of the system (in ``system_order``), the usual MIPT order parameter."""
    return Probe("half_cut", "half_cut_entropy")


def mutual_information(a: Sequence[int], b: Sequence[int], name: str | None = None) -> Probe:
    """I(A:B) = S(A) + S(B) - S(AB) for disjoint regions."""
    ra, rb = _region(a), _region(b)
    return Probe(
        "mutual_information", name or f"mutual_information{[list(ra), list(rb)]}", (ra, rb)
    )


def tripartite_information(
    a: Sequence[int] | None = None,
    b: Sequence[int] | None = None,
    c: Sequence[int] | None = None,
    name: str | None = None,
) -> Probe:
    """I3(A:B:C) for disjoint regions; the standard critical-point locator.

    Without arguments, A, B, C are the first three quarters of the system in
    ``system_order`` (the usual choice for a ring split into four arcs).
    """
    if a is None and b is None and c is None:
        return Probe("tripartite", name or "tripartite_information")
    if a is None or b is None or c is None:
        raise ValueError("give all three regions, or none for the default quarters")
    regions = (_region(a), _region(b), _region(c))
    return Probe(
        "tripartite", name or f"tripartite_information{[list(r) for r in regions]}", regions
    )


def code_dimension() -> Probe:
    """Number of logical qubits still protected: k = S(R). Needs a reference."""
    return Probe("code_dimension", "code_dimension")


def contiguous_distance() -> Probe:
    """Shortest contiguous run (in ``system_order``, wrapping if ``periodic``) whose erasure loses logical info."""
    return Probe("contiguous_distance", "contiguous_distance")


def lost_information(region: Sequence[int], name: str | None = None) -> Probe:
    """I(R:E) for a fixed set E of system qubits: logical information lost by erasing E."""
    r = _region(region)
    return Probe("lost_information", name or f"lost_information{list(r)}", (r,))


def erasure_lost_information(size: int, samples: int = 100, name: str | None = None) -> Probe:
    """Mean I(R:E) over ``samples`` random erasures of ``size`` system qubits."""
    return Probe(
        "erasure_mean_lost",
        name or f"erasure_lost_information[{size}]",
        (),
        int(size),
        int(samples),
    )


def erasure_correctable_fraction(size: int, samples: int = 100, name: str | None = None) -> Probe:
    """Fraction of ``samples`` random erasures of ``size`` system qubits that are correctable."""
    return Probe(
        "erasure_correctable",
        name or f"erasure_correctable_fraction[{size}]",
        (),
        int(size),
        int(samples),
    )


def entropy_profile() -> Probe:
    """Vector: ``S(system_order[:k])`` for every k = 0..n (entanglement of every bipartition)."""
    return Probe("entropy_profile", "entropy_profile")


def erasure_lost_profile(samples: int = 10, name: str | None = None) -> Probe:
    """Vector: mean I(R:E_m) over random erasures E_m of every size m = 0..n.

    Each sample erases the prefixes of one uniformly random ordering of the
    system, which gives a uniformly random erasure of every size in a single
    incremental pass.
    """
    return Probe("erasure_lost_profile", name or "erasure_lost_profile", (), 0, int(samples))


def erasure_correctable_profile(samples: int = 10, name: str | None = None) -> Probe:
    """Vector: fraction of random erasures of each size m = 0..n that are correctable.

    Feed it to :func:`sparsegf2_v3.analysis.bernoulli_erasure_curve` to get the
    failure probability when each qubit is erased independently with
    probability p.
    """
    return Probe(
        "erasure_correctable_profile", name or "erasure_correctable_profile", (), 0, int(samples)
    )


_VECTOR_KINDS = {
    "entropy_profile": _ens.VPROBE_ENTROPY_PROFILE,
    "erasure_lost_profile": _ens.VPROBE_ERASURE_LOST,
    "erasure_correctable_profile": _ens.VPROBE_ERASURE_CORRECTABLE,
}
_SCALAR_KINDS = {
    "entropy": _ens.PROBE_ENTROPY,
    "half_cut": _ens.PROBE_ENTROPY,
    "mutual_information": _ens.PROBE_MUTUAL_INFORMATION,
    "tripartite": _ens.PROBE_TRIPARTITE,
    "code_dimension": _ens.PROBE_CODE_DIMENSION,
    "contiguous_distance": _ens.PROBE_CONTIGUOUS_DISTANCE,
    "lost_information": _ens.PROBE_LOST_INFORMATION,
    "erasure_mean_lost": _ens.PROBE_ERASURE_MEAN_LOST,
    "erasure_correctable": _ens.PROBE_ERASURE_CORRECTABLE,
}
_NEEDS_REFERENCE = {
    "code_dimension",
    "contiguous_distance",
    "lost_information",
    "erasure_mean_lost",
    "erasure_correctable",
    "erasure_lost_profile",
    "erasure_correctable_profile",
}


# ---------------------------------------------------------------------------
# the protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class MonitoredCircuit:
    """A random monitored-circuit protocol (see the module docstring for the steps).

    Parameters
    ----------
    num_system
        Number of system qubits ``n``.
    edges
        Interaction graph on the system qubits, an ``(m, 2)`` array (see
        :mod:`sparsegf2_v3.graphs`), or a list of them for graph disorder:
        trajectory ``t`` runs on graph ``t mod len(edges)``.
    layers
        Number of dynamics layers.
    measurement_rate
        Probability that each system qubit is measured (Z basis) in each layer.
    schedule, gates_per_layer, edge_weights
        How gates are placed in a layer: ``"matching"`` (default),
        ``"random_edges"`` (``gates_per_layer`` edges with replacement, default
        ``n // 2``, uniform or proportional to ``edge_weights``) or
        ``"all_edges"``.
    reference, reference_partner
        ``"none"`` (default), ``"purification"`` or ``"single"``; for
        ``"single"``, the Bell partner is a system qubit index or ``"random"``
        (drawn per trajectory).
    scramble_gates, scramble_matchings, scramble_layers
        Optional pre-scrambling, in this order: gates on random distinct pairs,
        layers of random all-to-all perfect matchings, gate-only layers of the
        dynamics.
    gate_set
        Canonical two-qubit Clifford indices to draw gates from uniformly
        (default: all 11520, i.e. the uniform two-qubit Clifford group).
    random_orientation
        Randomly swap each gate's two qubits (matters only for non-uniform
        ``gate_set``).
    checkpoints
        ``"final"`` (default), ``"every_layer"``, an int stride ``k`` (layers
        0, k, 2k, ... and the last), or an explicit list of layers in
        ``[0, layers]``.
    probes
        What to record at each checkpoint (:func:`code_dimension`, ...).
    stop_when_purified
        End each trajectory at the first checkpoint where the reference is pure.
    system_order, periodic
        Qubit ordering used by the half-cut entropy, default tripartite regions,
        entanglement profile and contiguous distance (default ``0..n-1``), and
        whether contiguous runs wrap around.
    """

    num_system: int
    edges: np.ndarray
    layers: int
    measurement_rate: float
    schedule: str = "matching"
    gates_per_layer: int | None = None
    edge_weights: np.ndarray | None = None
    reference: str = "none"
    reference_partner: int | str = "random"
    scramble_gates: int = 0
    scramble_matchings: int = 0
    scramble_layers: int = 0
    gate_set: np.ndarray | None = None
    random_orientation: bool = False
    checkpoints: str | int | Sequence[int] = "final"
    probes: Sequence[Probe] = ()
    stop_when_purified: bool = False
    system_order: Sequence[int] | None = None
    periodic: bool = True
    _compiled: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        n = int(self.num_system)
        if n < 2:
            raise ValueError("num_system must be at least 2")
        raw_graphs, is_bank = _graph_list(self.edges)
        graphs_ = [from_edges(g, n) for g in raw_graphs]
        if not graphs_ or any(len(g) == 0 for g in graphs_):
            raise ValueError("every graph needs at least one edge")
        object.__setattr__(self, "num_system", n)
        if self.layers < 0:
            raise ValueError("layers must be non-negative")
        if not 0.0 <= self.measurement_rate <= 1.0:
            raise ValueError("measurement_rate must be in [0, 1]")
        if self.schedule not in _SCHEDULES:
            raise ValueError(f"schedule must be one of {list(_SCHEDULES)}, got {self.schedule!r}")
        if self.gates_per_layer is not None and self.schedule != "random_edges":
            raise ValueError("gates_per_layer only applies to schedule='random_edges'")
        if self.edge_weights is not None:
            if self.schedule != "random_edges":
                raise ValueError("edge_weights only applies to schedule='random_edges'")
            raw_weights = list(self.edge_weights) if is_bank else [self.edge_weights]
            if len(raw_weights) != len(graphs_):
                raise ValueError(
                    "with a list of graphs, edge_weights must be a list of the same length"
                )
            weights = [
                _checked_weights(w, raw, g)
                for w, raw, g in zip(raw_weights, raw_graphs, graphs_, strict=True)
            ]
            object.__setattr__(self, "edge_weights", tuple(weights) if is_bank else weights[0])
        object.__setattr__(self, "edges", tuple(graphs_) if is_bank else graphs_[0])
        if self.schedule == "random_edges":
            gpl = n // 2 if self.gates_per_layer is None else int(self.gates_per_layer)
            if gpl < 0:
                raise ValueError("gates_per_layer must be non-negative")
            object.__setattr__(self, "gates_per_layer", gpl)
        if self.reference not in _REFERENCES:
            raise ValueError(
                f"reference must be one of {list(_REFERENCES)}, got {self.reference!r}"
            )
        if self.reference == "single" and self.reference_partner != "random":
            partner = int(self.reference_partner)
            if not 0 <= partner < n:
                raise ValueError(f"reference_partner must be 'random' or in [0, {n})")
            object.__setattr__(self, "reference_partner", partner)
        if min(self.scramble_gates, self.scramble_matchings, self.scramble_layers) < 0:
            raise ValueError("scramble_gates, scramble_matchings and scramble_layers must be >= 0")
        pool = (
            np.arange(num_cliffords(2), dtype=np.int64)
            if self.gate_set is None
            else np.asarray(self.gate_set, dtype=np.int64).ravel()
        )
        if pool.size == 0 or pool.min() < 0 or pool.max() >= num_cliffords(2):
            raise ValueError("gate_set must be a non-empty list of indices in [0, 11520)")
        object.__setattr__(self, "gate_set", pool)
        order = (
            np.arange(n, dtype=np.int64)
            if self.system_order is None
            else np.asarray(self.system_order, dtype=np.int64)
        )
        if sorted(order.tolist()) != list(range(n)):
            raise ValueError("system_order must be a permutation of the system qubits")
        object.__setattr__(self, "system_order", order)
        object.__setattr__(self, "checkpoints", self._resolve_checkpoints(self.checkpoints))
        object.__setattr__(self, "probes", tuple(self._resolve_probe(p) for p in self.probes))
        names = [p.name for p in self.probes]
        if len(set(names)) != len(names):
            raise ValueError(f"probe names must be unique, got {names}")
        if self.stop_when_purified and self.reference == "none":
            raise ValueError("stop_when_purified needs a reference")

    def _resolve_probe(self, p: Probe) -> Probe:
        """Validate a probe and fill in default regions."""
        if not isinstance(p, Probe):
            raise TypeError("probes must be created with the mipt probe functions")
        n = self.num_system
        order = self.system_order.tolist()
        if p.kind in _NEEDS_REFERENCE and self.reference == "none":
            raise ValueError(
                f"probe {p.name!r} needs a reference (reference='purification' or 'single')"
            )
        if p.kind == "half_cut":
            return replace(p, regions=(tuple(order[: n // 2]),))
        if p.kind == "tripartite" and not p.regions:
            if n < 4:
                raise ValueError("default tripartite regions need at least 4 system qubits")
            q = n // 4
            return replace(
                p, regions=(tuple(order[:q]), tuple(order[q : 2 * q]), tuple(order[2 * q : 3 * q]))
            )
        if p.regions:
            flat = [q for r in p.regions for q in r]
            limit = n if p.kind == "lost_information" else self.num_qubits
            if any(not 0 <= q < limit for q in flat):
                what = "system qubits" if p.kind == "lost_information" else "valid qubit indices"
                raise ValueError(f"probe {p.name!r}: regions must contain {what}")
            if len(set(flat)) != len(flat):
                raise ValueError(
                    f"probe {p.name!r}: regions must be disjoint and without duplicates"
                )
        if p.kind in ("erasure_mean_lost", "erasure_correctable") and not (
            0 <= p.size <= n and p.samples >= 1
        ):
            raise ValueError(f"probe {p.name!r}: need 0 <= size <= {n} and samples >= 1")
        if p.kind in ("erasure_lost_profile", "erasure_correctable_profile") and p.samples < 1:
            raise ValueError(f"probe {p.name!r}: samples must be at least 1")
        if p.kind not in _SCALAR_KINDS and p.kind not in _VECTOR_KINDS:
            raise ValueError(f"unknown probe kind {p.kind!r}")
        return p

    def _resolve_checkpoints(self, spec) -> np.ndarray:
        if isinstance(spec, np.ndarray) and spec.dtype == np.int64 and spec.ndim == 1:
            layers = spec
        elif isinstance(spec, str):
            if spec == "final":
                layers = np.array([self.layers], dtype=np.int64)
            elif spec == "every_layer":
                layers = np.arange(self.layers + 1, dtype=np.int64)
            else:
                raise ValueError(
                    f"checkpoints must be 'final', 'every_layer', a stride or a list, got {spec!r}"
                )
        elif isinstance(spec, (int, np.integer)):
            if spec < 1:
                raise ValueError("a checkpoint stride must be positive")
            layers = np.unique(np.append(np.arange(0, self.layers + 1, int(spec)), self.layers))
        else:
            layers = np.unique(np.asarray(list(spec), dtype=np.int64))
        if layers.size == 0 or layers.min() < 0 or layers.max() > self.layers:
            raise ValueError(f"checkpoints must lie in [0, layers={self.layers}]")
        return layers.astype(np.int64)

    # -- derived quantities ----------------------------------------------------------

    @property
    def graphs(self) -> tuple[np.ndarray, ...]:
        """The interaction graphs (one, or the graph bank); trajectory t uses ``graphs[t % G]``."""
        return self.edges if isinstance(self.edges, tuple) else (self.edges,)

    @property
    def num_graphs(self) -> int:
        return len(self.graphs)

    @property
    def num_reference(self) -> int:
        return {"none": 0, "purification": self.num_system, "single": 1}[self.reference]

    @property
    def num_qubits(self) -> int:
        """System plus reference qubits."""
        return self.num_system + self.num_reference

    @property
    def system_qubits(self) -> list[int]:
        return list(range(self.num_system))

    @property
    def reference_qubits(self) -> list[int]:
        return list(range(self.num_system, self.num_qubits))

    @property
    def checkpoint_layers(self) -> np.ndarray:
        return self.checkpoints

    @property
    def scalar_probes(self) -> list[Probe]:
        return [p for p in self.probes if not p.is_vector]

    @property
    def vector_probes(self) -> list[Probe]:
        return [p for p in self.probes if p.is_vector]

    def with_edges(
        self, edges: np.ndarray, edge_weights: np.ndarray | None = None
    ) -> MonitoredCircuit:
        """The same protocol on other graph(s) (one graph, or a list of graphs)."""
        return replace(self, edges=edges, edge_weights=edge_weights, _compiled={})

    def _arrays(self) -> dict:
        if not self._compiled:
            graph_ptr = np.zeros(self.num_graphs + 1, dtype=np.int64)
            color_ptr_ptr = np.zeros(self.num_graphs + 1, dtype=np.int64)
            color_ptr, color_idx, edge_cum = [], [], []
            weights = self.edge_weights
            if weights is not None and not isinstance(weights, tuple):
                weights = (weights,)
            for g, edges in enumerate(self.graphs):
                graph_ptr[g + 1] = graph_ptr[g] + len(edges)
                colors = edge_coloring(edges, self.num_system)
                color_ptr.append(np.concatenate([[0], np.cumsum([len(c) for c in colors])]))
                color_ptr_ptr[g + 1] = color_ptr_ptr[g] + len(colors) + 1
                color_idx.append(np.concatenate(colors))
                if weights is not None:
                    cum = np.cumsum(weights[g] / weights[g].sum())
                    cum[-1] = 1.0
                    edge_cum.append(cum)
            self._compiled.update(
                edges=np.concatenate(self.graphs).astype(np.int64),
                graph_ptr=graph_ptr,
                color_ptr=np.concatenate(color_ptr).astype(np.int64),
                color_ptr_ptr=color_ptr_ptr,
                color_idx=np.concatenate(color_idx).astype(np.int64),
                edge_cum=np.concatenate(edge_cum) if edge_cum else np.zeros(0, dtype=np.float64),
            )
            scalars = self.scalar_probes
            ptr, idx = [], []
            region_len = np.zeros((len(scalars), 3), dtype=np.int64)
            for j, p in enumerate(scalars):
                ptr.append(len(idx))
                for r, region in enumerate(p.regions[:3]):
                    idx.extend(region)
                    region_len[j, r] = len(region)
            vectors = self.vector_probes
            self._compiled.update(
                kinds=np.array([_SCALAR_KINDS[p.kind] for p in scalars], dtype=np.int64),
                ptr=np.array(ptr, dtype=np.int64),
                region_len=region_len,
                idx=np.array(idx, dtype=np.int64),
                size=np.array([p.size for p in scalars], dtype=np.int64),
                samples=np.array([p.samples for p in scalars], dtype=np.int64),
                vkinds=np.array([_VECTOR_KINDS[p.kind] for p in vectors], dtype=np.int64),
                vsamples=np.array([p.samples for p in vectors], dtype=np.int64),
                partner=-1 if self.reference_partner == "random" else int(self.reference_partner),
                gpl=int(self.gates_per_layer or 0),
            )
        return self._compiled

    # -- running ---------------------------------------------------------------------

    def run(
        self,
        trajectories: int,
        *,
        seed,
        first_trajectory: int = 0,
        num_threads: int | None = None,
    ) -> EnsembleResult:
        """Simulate trajectories ``first_trajectory .. first_trajectory + trajectories - 1``.

        Runs in parallel over CPU cores (``num_threads`` limits that). Returns an
        :class:`EnsembleResult`.
        """
        if trajectories < 0:
            raise ValueError("trajectories must be non-negative")
        a = self._arrays()
        key_struct, key_probe = _keys(seed)
        num_ck = len(self.checkpoints)
        scalars, vectors = self.scalar_probes, self.vector_probes
        values = np.zeros((trajectories, num_ck, len(scalars)), dtype=np.float64)
        vec_values = np.zeros(
            (trajectories, num_ck, len(vectors), self.num_system + 1), dtype=np.float64
        )
        tau = np.zeros(trajectories, dtype=np.int64)
        final = np.zeros(trajectories, dtype=np.int64)
        meas = np.zeros(trajectories, dtype=np.int64)
        start = time.perf_counter()
        if trajectories:
            with threads(num_threads):
                _ens.run_ensemble(
                    self.num_system, self.num_reference, _REFERENCES[self.reference], a["partner"],
                    a["edges"], a["graph_ptr"], a["edge_cum"], a["color_ptr"], a["color_ptr_ptr"],
                    a["color_idx"], _SCHEDULES[self.schedule], a["gpl"], self.gate_set,
                    float(self.measurement_rate), int(self.layers), int(self.scramble_gates),
                    int(self.scramble_matchings), int(self.scramble_layers),
                    bool(self.random_orientation), self.checkpoints,
                    bool(self.stop_when_purified), a["kinds"], a["ptr"], a["region_len"], a["idx"],
                    a["size"], a["samples"], a["vkinds"], a["vsamples"], self.system_order,
                    bool(self.periodic), key_struct, key_probe, int(first_trajectory),
                    values, vec_values, tau, final, meas,
                    _H_INDEX, _CX_INDEX, LIN1, ANF1, IDENTITY_SYM1, LIN2, ANF2, IDENTITY_SYM2,
                )  # fmt: skip
        elapsed = time.perf_counter() - start
        out = {p.name: values[:, :, j] for j, p in enumerate(scalars)}
        out.update({p.name: vec_values[:, :, v, :] for v, p in enumerate(vectors)})
        return EnsembleResult(
            spec=self,
            seed_entropy=_entropy_of(seed),
            trajectories=np.arange(
                first_trajectory, first_trajectory + trajectories, dtype=np.int64
            ),
            checkpoint_layers=self.checkpoints.copy(),
            values={p.name: out[p.name] for p in self.probes},
            purification_layer=tau if self.reference != "none" else None,
            final_layer=final,
            num_measurements=meas,
            elapsed_seconds=elapsed,
        )

    def circuit(self, trajectory: int, *, seed) -> Circuit:
        """The exact circuit that trajectory ``trajectory`` of ``run(seed=seed)`` simulates.

        Layout: setup gates, scramble (CLIFFORD_2Q), ``TICK`` (checkpoint layer
        0), then per layer ``CLIFFORD_2Q`` gates, ``M`` measurements and a
        ``TICK``. So tick ``t`` of ``Simulator.run(circuit, observe=...)``
        sees the state after layer ``t``.
        """
        a = self._arrays()
        key_struct, _ = _keys(seed)
        n = self.num_system
        max_gates = int(_ens._max_gates(n, a["graph_ptr"], a["gpl"], self.scramble_gates))
        scramble_blocks = self.scramble_matchings + self.scramble_layers
        total_gate_blocks = 1 + scramble_blocks + self.layers
        ga = np.zeros(
            self.scramble_gates + (scramble_blocks + self.layers) * max_gates, dtype=np.int64
        )
        gb = np.zeros_like(ga)
        gg = np.zeros_like(ga)
        gate_ptr = np.zeros(total_gate_blocks + 1, dtype=np.int64)
        mq = np.zeros(self.layers * n, dtype=np.int64)
        meas_ptr = np.zeros(self.layers + 1, dtype=np.int64)
        partner = _ens.generate_ops(
            n, _REFERENCES[self.reference], a["partner"], a["edges"], a["graph_ptr"],
            a["edge_cum"], a["color_ptr"], a["color_ptr_ptr"], a["color_idx"],
            _SCHEDULES[self.schedule], a["gpl"], self.gate_set, float(self.measurement_rate),
            int(self.layers), int(self.scramble_gates), int(self.scramble_matchings),
            int(self.scramble_layers), bool(self.random_orientation), key_struct, int(trajectory),
            ga, gb, gg, gate_ptr, mq, meas_ptr,
        )  # fmt: skip
        c = Circuit()
        if self.reference == "purification":
            c.append("H", range(n))
            c.append("CX", [q for i in range(n) for q in (i, n + i)])
        elif self.reference == "single":
            c.append("H", [int(partner)])
            c.append("CX", [int(partner), n])

        def gates(block: int) -> None:
            lo, hi = gate_ptr[block], gate_ptr[block + 1]
            if hi > lo:
                pairs = np.stack([ga[lo:hi], gb[lo:hi]], axis=1).ravel().tolist()
                c.append("CLIFFORD_2Q", pairs, gg[lo:hi].tolist())

        for block in range(1 + scramble_blocks):
            gates(block)
        c.append("TICK")
        for layer in range(self.layers):
            gates(1 + scramble_blocks + layer)
            lo, hi = meas_ptr[layer], meas_ptr[layer + 1]
            if hi > lo:
                c.append("M", mq[lo:hi].tolist())
            c.append("TICK")
        return c


def _graph_list(edges) -> tuple[list[np.ndarray], bool]:
    """Split ``edges`` into a list of ``(m, 2)`` arrays; the flag says it was a list of graphs."""
    if isinstance(edges, np.ndarray) and edges.ndim != 3:
        return [edges.reshape(-1, 2)], False
    try:
        arr = np.asarray(edges, dtype=np.int64)
    except ValueError:  # ragged: graphs with different numbers of edges
        return [np.asarray(e, dtype=np.int64).reshape(-1, 2) for e in edges], True
    if arr.ndim == 3:
        return list(arr), True
    return [arr.reshape(-1, 2)], False


def _checked_weights(weights, raw_edges: np.ndarray, edges: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != (len(raw_edges),) or len(raw_edges) != len(edges):
        raise ValueError(
            "edge_weights needs one weight per edge of an already normalised edge "
            "list (pass edges from sparsegf2_v3.graphs)"
        )
    if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("edge_weights must be finite, non-negative and not all zero")
    if not np.array_equal(raw_edges, edges):
        raise ValueError("edge_weights requires edges in normalised order (sorted, a < b)")
    return w


def _keys(seed) -> tuple[np.ndarray, np.ndarray]:
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    state = ss.generate_state(4, dtype=np.uint64)
    return state[:2].copy(), state[2:].copy()


def _entropy_of(seed):
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    e = ss.entropy
    return e.tolist() if isinstance(e, np.ndarray) else e


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


_SPEC_FIELDS = (
    "num_system",
    "layers",
    "measurement_rate",
    "schedule",
    "gates_per_layer",
    "reference",
    "reference_partner",
    "scramble_gates",
    "scramble_matchings",
    "scramble_layers",
    "random_orientation",
    "stop_when_purified",
    "periodic",
)


@dataclass
class EnsembleResult:
    """Everything an ensemble run produced. Arrays have one row per trajectory.

    Attributes
    ----------
    values
        ``values[probe_name]`` has shape ``(trajectories, checkpoints)`` for
        scalar probes and ``(trajectories, checkpoints, num_system + 1)`` for
        vector probes. Entries after a trajectory stopped
        (``stop_when_purified``) are NaN.
    purification_layer
        First checkpoint layer at which the reference was pure, or -1 if it
        never was (right-censored at the last checkpoint). ``None`` without a
        reference.
    final_layer
        Last layer simulated for each trajectory.
    num_measurements
        Number of measurements made in each trajectory.
    """

    spec: MonitoredCircuit
    seed_entropy: object
    trajectories: np.ndarray
    checkpoint_layers: np.ndarray
    values: dict[str, np.ndarray]
    purification_layer: np.ndarray | None
    final_layer: np.ndarray
    num_measurements: np.ndarray
    elapsed_seconds: float = 0.0

    @property
    def num_trajectories(self) -> int:
        return int(self.trajectories.size)

    @property
    def censored(self) -> np.ndarray | None:
        """True where the reference never purified (right-censored)."""
        return None if self.purification_layer is None else self.purification_layer < 0

    @property
    def trajectories_per_second(self) -> float:
        return (
            self.num_trajectories / self.elapsed_seconds
            if self.elapsed_seconds > 0
            else float("inf")
        )

    def mean(self, name: str) -> np.ndarray:
        """Average of a probe over trajectories at each checkpoint (ignoring stopped ones)."""
        v = self.values[name]
        if v.shape[0] == 0:
            return np.full(v.shape[1:], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns after stops
            return np.nanmean(v, axis=0)

    def survival(self) -> tuple[np.ndarray, np.ndarray]:
        """Survival function of the purification time at the checkpoints.

        Returns ``(layers, S)`` with ``S[i] = P(tau > layers[i])``, the fraction
        of trajectories whose reference is still mixed after that layer. Every
        trajectory is observed up to the last checkpoint, so this is exact
        (censoring only affects layers beyond the last checkpoint); with a
        reference-qubit picture ``S(t)`` equals the mean reference entropy.
        """
        if self.purification_layer is None:
            raise ValueError("survival needs a reference (purification or single)")
        tau = self.purification_layer
        layers = self.checkpoint_layers
        alive = (tau[None, :] < 0) | (tau[None, :] > layers[:, None])
        return layers.copy(), alive.mean(axis=1)

    def __repr__(self) -> str:
        return (
            f"EnsembleResult(trajectories={self.num_trajectories}, checkpoints={len(self.checkpoint_layers)}, "
            f"probes={list(self.values)}, {self.trajectories_per_second:,.0f} trajectories/s)"
        )

    # -- persistence --------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the arrays and run metadata to a compressed ``.npz`` file."""
        spec = self.spec
        meta = {
            "format": "sparsegf2_v3.EnsembleResult/1",
            "seed_entropy": self.seed_entropy,
            "elapsed_seconds": self.elapsed_seconds,
            "probes": [
                {
                    "kind": p.kind,
                    "name": p.name,
                    "regions": [list(r) for r in p.regions],
                    "size": p.size,
                    "samples": p.samples,
                }
                for p in spec.probes
            ],
            "spec": {k: getattr(spec, k) for k in _SPEC_FIELDS},
        }
        sizes = [len(g) for g in spec.graphs]
        arrays = {
            "edges": np.concatenate(spec.graphs),
            "graph_ptr": np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
            "gate_set": spec.gate_set,
            "system_order": spec.system_order,
            "trajectories": self.trajectories,
            "checkpoint_layers": self.checkpoint_layers,
            "final_layer": self.final_layer,
            "num_measurements": self.num_measurements,
        }
        if spec.edge_weights is not None:
            w = spec.edge_weights
            arrays["edge_weights"] = np.concatenate(w) if isinstance(w, tuple) else w
        meta["graph_bank"] = isinstance(spec.edges, tuple)
        for name, v in self.values.items():
            arrays[f"value::{name}"] = v
        if self.purification_layer is not None:
            arrays["purification_layer"] = self.purification_layer
        np.savez_compressed(path, __meta__=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> EnsembleResult:
        """Read a file written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
            s = meta["spec"]
            probes = [
                Probe(
                    p["kind"],
                    p["name"],
                    tuple(tuple(r) for r in p["regions"]),
                    p["size"],
                    p["samples"],
                )
                for p in meta["probes"]
            ]
            edges, weights = z["edges"], z["edge_weights"] if "edge_weights" in z.files else None
            if meta.get("graph_bank"):
                cuts = z["graph_ptr"][1:-1]
                edges = list(np.split(edges, cuts))
                weights = None if weights is None else list(np.split(weights, cuts))
            spec = MonitoredCircuit(
                edges=edges,
                gate_set=z["gate_set"],
                system_order=z["system_order"],
                edge_weights=weights,
                checkpoints=z["checkpoint_layers"].astype(np.int64),
                probes=probes,
                **{
                    k: s[k]
                    for k in _SPEC_FIELDS
                    if k in s and not (k == "gates_per_layer" and s["schedule"] != "random_edges")
                },
            )
            return cls(
                spec=spec,
                seed_entropy=meta["seed_entropy"],
                trajectories=z["trajectories"],
                checkpoint_layers=z["checkpoint_layers"],
                values={p.name: z[f"value::{p.name}"] for p in probes},
                purification_layer=z["purification_layer"]
                if "purification_layer" in z.files
                else None,
                final_layer=z["final_layer"],
                num_measurements=z["num_measurements"],
                elapsed_seconds=meta["elapsed_seconds"],
            )

    @staticmethod
    def concatenate(results: Sequence[EnsembleResult]) -> EnsembleResult:
        """Merge chunks of the same protocol and seed (e.g. from separate jobs)."""
        if not results:
            raise ValueError("nothing to concatenate")
        first = results[0]
        for r in results[1:]:
            if r.seed_entropy != first.seed_entropy or list(r.values) != list(first.values):
                raise ValueError("can only concatenate results of the same protocol and seed")
            if not np.array_equal(r.checkpoint_layers, first.checkpoint_layers):
                raise ValueError("checkpoint layers differ")

        def cat(get):
            parts = [get(r) for r in results]
            return None if parts[0] is None else np.concatenate(parts)

        return EnsembleResult(
            spec=first.spec,
            seed_entropy=first.seed_entropy,
            trajectories=cat(lambda r: r.trajectories),
            checkpoint_layers=first.checkpoint_layers,
            values={k: np.concatenate([r.values[k] for r in results]) for k in first.values},
            purification_layer=cat(lambda r: r.purification_layer),
            final_layer=cat(lambda r: r.final_layer),
            num_measurements=cat(lambda r: r.num_measurements),
            elapsed_seconds=sum(r.elapsed_seconds for r in results),
        )
