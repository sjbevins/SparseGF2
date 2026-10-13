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
    spec.circuit(trajectory=3, seed=7)                 # the exact circuit of trajectory 3

The protocol, step by step (qubits ``0..n-1`` are the system; references
follow):

1. **Setup.** ``reference="purification"``: system qubit ``i`` is Bell-paired
   with reference ``n + i`` (the maximally mixed initial state). ``"single"``:
   one reference ``n`` is Bell-paired with ``reference_partner``. ``"none"``:
   just |0...0>.
2. **Scramble** (optional). ``scramble_gates`` random two-qubit gates on
   uniformly random distinct pairs of system qubits, then ``scramble_layers``
   gate-only layers of the dynamics below.
3. **Dynamics**, ``layers`` times. Gates on edges of the graph, chosen by
   ``schedule``:

   * ``"matching"``: cycle through an edge colouring of the graph (on an even
     ring this is the standard brickwork, alternating even and odd bonds);
   * ``"random_edges"``: ``gates_per_layer`` edges drawn uniformly with
     replacement;
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

Randomness: trajectory ``t`` of ``run(..., seed=s)`` is a deterministic function
of ``(spec, s, t)``, independent of the thread count and of the other
trajectories, so ensembles can be split into chunks (``first_trajectory``) and
merged without changing any result. Graph disorder is separate: build the
``edges`` with their own seed.
"""

from __future__ import annotations

import json
import time
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
    "erasure_correctable_fraction",
    "erasure_lost_information",
    "half_cut_entropy",
    "lost_information",
    "mean_stabilizer_weight",
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
    """Something to record at every checkpoint. Build one with the functions below."""

    kind: str
    name: str
    region: tuple[int, ...] = ()
    size: int = 0
    samples: int = 0


def entropy(region: Sequence[int], name: str | None = None) -> Probe:
    """Entanglement entropy S(A) of the given qubits (system and/or reference)."""
    region = tuple(int(q) for q in region)
    return Probe("entropy", name or f"entropy{list(region)}", region)


def half_cut_entropy() -> Probe:
    """S of the first half of the system (in ``system_order``), the usual MIPT order parameter."""
    return Probe("half_cut", "half_cut_entropy")


def code_dimension() -> Probe:
    """Number of logical qubits still protected: k = S(R). Needs a reference."""
    return Probe("code_dimension", "code_dimension")


def contiguous_distance() -> Probe:
    """Shortest contiguous run (in ``system_order``, wrapping if ``periodic``) whose erasure loses logical info."""
    return Probe("contiguous_distance", "contiguous_distance")


def lost_information(region: Sequence[int], name: str | None = None) -> Probe:
    """I(R:E) for a fixed set E of system qubits: logical information lost by erasing E."""
    region = tuple(int(q) for q in region)
    return Probe("lost_information", name or f"lost_information{list(region)}", region)


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
    """Fraction of ``samples`` random erasures of ``size`` system qubits that are correctable (I(R:E) = 0)."""
    return Probe(
        "erasure_correctable",
        name or f"erasure_correctable_fraction[{size}]",
        (),
        int(size),
        int(samples),
    )


def mean_stabilizer_weight() -> Probe:
    """Average number of qubits per stabilizer generator: a sparsity diagnostic (basis dependent)."""
    return Probe("mean_weight", "mean_stabilizer_weight")


_NEEDS_REFERENCE = {
    "code_dimension",
    "contiguous_distance",
    "lost_information",
    "erasure_mean_lost",
    "erasure_correctable",
}
_PROBE_CODE = {
    "entropy": _ens.PROBE_ENTROPY,
    "half_cut": _ens.PROBE_ENTROPY,
    "code_dimension": _ens.PROBE_CODE_DIMENSION,
    "contiguous_distance": _ens.PROBE_CONTIGUOUS_DISTANCE,
    "lost_information": _ens.PROBE_LOST_INFORMATION,
    "erasure_mean_lost": _ens.PROBE_ERASURE_MEAN_LOST,
    "erasure_correctable": _ens.PROBE_ERASURE_CORRECTABLE,
    "mean_weight": _ens.PROBE_MEAN_WEIGHT,
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
        :mod:`sparsegf2_v3.graphs`).
    layers
        Number of dynamics layers.
    measurement_rate
        Probability that each system qubit is measured (Z basis) in each layer.
    schedule, gates_per_layer
        How gates are placed in a layer: ``"matching"`` (default),
        ``"random_edges"`` (``gates_per_layer`` edges with replacement; default
        ``n // 2``) or ``"all_edges"``.
    reference, reference_partner
        ``"none"`` (default), ``"purification"`` or ``"single"``; for
        ``"single"``, the Bell partner is a system qubit index or ``"random"``
        (drawn per trajectory).
    scramble_gates, scramble_layers
        Optional pre-scrambling: random all-to-all gates, then gate-only layers.
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
        Qubit ordering used by the half-cut entropy, the entanglement profile
        and the contiguous distance (default ``0..n-1``), and whether contiguous
        runs wrap around.
    record_profile
        Also record the entanglement profile ``S(system_order[:k])`` for every
        ``k`` at each checkpoint.
    """

    num_system: int
    edges: np.ndarray
    layers: int
    measurement_rate: float
    schedule: str = "matching"
    gates_per_layer: int | None = None
    reference: str = "none"
    reference_partner: int | str = "random"
    scramble_gates: int = 0
    scramble_layers: int = 0
    gate_set: np.ndarray | None = None
    random_orientation: bool = False
    checkpoints: str | int | Sequence[int] = "final"
    probes: Sequence[Probe] = ()
    stop_when_purified: bool = False
    system_order: Sequence[int] | None = None
    periodic: bool = True
    record_profile: bool = False
    _compiled: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        n = int(self.num_system)
        if n < 2:
            raise ValueError("num_system must be at least 2")
        edges = from_edges(self.edges, n)
        if len(edges) == 0:
            raise ValueError("the graph has no edges")
        object.__setattr__(self, "num_system", n)
        object.__setattr__(self, "edges", edges)
        if self.layers < 0:
            raise ValueError("layers must be non-negative")
        if not 0.0 <= self.measurement_rate <= 1.0:
            raise ValueError("measurement_rate must be in [0, 1]")
        if self.schedule not in _SCHEDULES:
            raise ValueError(f"schedule must be one of {list(_SCHEDULES)}, got {self.schedule!r}")
        if self.gates_per_layer is not None and self.schedule != "random_edges":
            raise ValueError("gates_per_layer only applies to schedule='random_edges'")
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
        if self.scramble_gates < 0 or self.scramble_layers < 0:
            raise ValueError("scramble_gates and scramble_layers must be non-negative")
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
        probes = tuple(self.probes)
        names = [p.name for p in probes]
        if len(set(names)) != len(names):
            raise ValueError(f"probe names must be unique, got {names}")
        for p in probes:
            if not isinstance(p, Probe):
                raise TypeError("probes must be created with the mipt probe functions")
            if p.kind in _NEEDS_REFERENCE and self.reference == "none":
                raise ValueError(
                    f"probe {p.name!r} needs a reference (reference='purification' or 'single')"
                )
            if p.kind in ("lost_information",) and any(not 0 <= q < n for q in p.region):
                raise ValueError(f"probe {p.name!r}: erased qubits must be system qubits")
            if p.kind == "entropy" and any(not 0 <= q < self.num_qubits for q in p.region):
                raise ValueError(f"probe {p.name!r}: region out of range")
            if p.kind == "entropy" and len(set(p.region)) != len(p.region):
                raise ValueError(f"probe {p.name!r}: region has duplicates")
            if p.kind in ("erasure_mean_lost", "erasure_correctable") and not (
                0 <= p.size <= n and p.samples >= 1
            ):
                raise ValueError(f"probe {p.name!r}: need 0 <= size <= {n} and samples >= 1")
        object.__setattr__(self, "probes", probes)
        if self.stop_when_purified and self.reference == "none":
            raise ValueError("stop_when_purified needs a reference")

    def _resolve_checkpoints(self, spec) -> np.ndarray:
        if isinstance(spec, np.ndarray) and spec.dtype == np.int64 and spec.ndim == 1:
            layers = spec
        elif spec == "final":
            layers = np.array([self.layers], dtype=np.int64)
        elif spec == "every_layer":
            layers = np.arange(self.layers + 1, dtype=np.int64)
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

    def with_edges(self, edges: np.ndarray) -> MonitoredCircuit:
        """The same protocol on another graph (e.g. a new graph-disorder realization)."""
        return replace(self, edges=edges, _compiled={})

    def _arrays(self) -> dict:
        if not self._compiled:
            colors = edge_coloring(self.edges, self.num_system)
            color_ptr = np.zeros(len(colors) + 1, dtype=np.int64)
            color_ptr[1:] = np.cumsum([len(c) for c in colors])
            color_idx = np.concatenate(colors).astype(np.int64) if colors else np.zeros(0, np.int64)
            ptr = [0]
            idx: list[int] = []
            n = self.num_system
            half = self.system_order[: n // 2].tolist()
            for p in self.probes:
                region = half if p.kind == "half_cut" else list(p.region)
                idx.extend(region)
                ptr.append(len(idx))
            self._compiled.update(
                color_ptr=color_ptr,
                color_idx=color_idx,
                kinds=np.array([_PROBE_CODE[p.kind] for p in self.probes], dtype=np.int64),
                ptr=np.array(ptr, dtype=np.int64),
                idx=np.array(idx, dtype=np.int64),
                size=np.array([p.size for p in self.probes], dtype=np.int64),
                samples=np.array([p.samples for p in self.probes], dtype=np.int64),
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
        keys = _keys(seed)
        num_ck = len(self.checkpoints)
        values = np.zeros((trajectories, num_ck, len(self.probes)), dtype=np.float64)
        prof_len = self.num_system + 1
        profiles = np.zeros(
            (trajectories, num_ck, prof_len) if self.record_profile else (trajectories, 0, 0),
            dtype=np.int64,
        )
        tau = np.zeros(trajectories, dtype=np.int64)
        final = np.zeros(trajectories, dtype=np.int64)
        meas = np.zeros(trajectories, dtype=np.int64)
        start = time.perf_counter()
        if trajectories:
            with threads(num_threads):
                _ens.run_ensemble(
                    self.num_system, self.num_reference, _REFERENCES[self.reference], a["partner"],
                    self.edges, a["color_ptr"], a["color_idx"], _SCHEDULES[self.schedule], a["gpl"],
                    self.gate_set, float(self.measurement_rate), int(self.layers),
                    int(self.scramble_gates), int(self.scramble_layers), bool(self.random_orientation),
                    self.checkpoints, bool(self.stop_when_purified), a["kinds"], a["ptr"], a["idx"],
                    a["size"], a["samples"], self.system_order, bool(self.periodic),
                    bool(self.record_profile), keys[0], keys[1], int(first_trajectory),
                    values, profiles, tau, final, meas,
                    _H_INDEX, _CX_INDEX, LIN1, ANF1, IDENTITY_SYM1, LIN2, ANF2, IDENTITY_SYM2,
                )  # fmt: skip
        elapsed = time.perf_counter() - start
        return EnsembleResult(
            spec=self,
            seed_entropy=_entropy_of(seed),
            trajectories=np.arange(
                first_trajectory, first_trajectory + trajectories, dtype=np.int64
            ),
            checkpoint_layers=self.checkpoints.copy(),
            values={p.name: values[:, :, j] for j, p in enumerate(self.probes)},
            profiles=profiles if self.record_profile else None,
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
        keys = _keys(seed)
        n = self.num_system
        max_gates = max(a["gpl"], len(self.edges), self.scramble_gates, 1)
        total_gate_blocks = 1 + self.scramble_layers + self.layers
        ga = np.zeros(
            self.scramble_gates + (self.scramble_layers + self.layers) * max_gates, dtype=np.int64
        )
        gb = np.zeros_like(ga)
        gg = np.zeros_like(ga)
        gate_ptr = np.zeros(total_gate_blocks + 1, dtype=np.int64)
        mq = np.zeros(self.layers * n, dtype=np.int64)
        meas_ptr = np.zeros(self.layers + 1, dtype=np.int64)
        partner = _ens.generate_ops(
            n, _REFERENCES[self.reference], a["partner"], self.edges, a["color_ptr"], a["color_idx"],
            _SCHEDULES[self.schedule], a["gpl"], self.gate_set, float(self.measurement_rate),
            int(self.layers), int(self.scramble_gates), int(self.scramble_layers),
            bool(self.random_orientation), keys[0], int(trajectory), ga, gb, gg, gate_ptr, mq, meas_ptr,
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

        for block in range(1 + self.scramble_layers):
            gates(block)
        c.append("TICK")
        for layer in range(self.layers):
            gates(1 + self.scramble_layers + layer)
            lo, hi = meas_ptr[layer], meas_ptr[layer + 1]
            if hi > lo:
                c.append("M", mq[lo:hi].tolist())
            c.append("TICK")
        return c


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


@dataclass
class EnsembleResult:
    """Everything an ensemble run produced. Arrays have one row per trajectory.

    Attributes
    ----------
    values
        ``values[probe_name]`` has shape ``(trajectories, checkpoints)``.
        Entries after a trajectory stopped (``stop_when_purified``) are NaN.
    profiles
        ``(trajectories, checkpoints, n + 1)`` entanglement profiles if
        ``record_profile`` was set (-1 after a stop), else ``None``.
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
    profiles: np.ndarray | None
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
        with np.errstate(invalid="ignore"):
            return np.nanmean(v, axis=0) if v.size else np.full(v.shape[1], np.nan)

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
            "seed_entropy": self.seed_entropy,
            "elapsed_seconds": self.elapsed_seconds,
            "probes": [p.__dict__ for p in spec.probes],
            "spec": {
                "num_system": spec.num_system,
                "layers": spec.layers,
                "measurement_rate": spec.measurement_rate,
                "schedule": spec.schedule,
                "gates_per_layer": spec.gates_per_layer,
                "reference": spec.reference,
                "reference_partner": spec.reference_partner,
                "scramble_gates": spec.scramble_gates,
                "scramble_layers": spec.scramble_layers,
                "random_orientation": spec.random_orientation,
                "stop_when_purified": spec.stop_when_purified,
                "periodic": spec.periodic,
                "record_profile": spec.record_profile,
            },
        }
        arrays = {
            "edges": spec.edges,
            "gate_set": spec.gate_set,
            "system_order": spec.system_order,
            "trajectories": self.trajectories,
            "checkpoint_layers": self.checkpoint_layers,
            "final_layer": self.final_layer,
            "num_measurements": self.num_measurements,
        }
        for name, v in self.values.items():
            arrays[f"value::{name}"] = v
        if self.profiles is not None:
            arrays["profiles"] = self.profiles
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
                Probe(p["kind"], p["name"], tuple(p["region"]), p["size"], p["samples"])
                for p in meta["probes"]
            ]
            spec = MonitoredCircuit(
                num_system=s["num_system"],
                edges=z["edges"],
                layers=s["layers"],
                measurement_rate=s["measurement_rate"],
                schedule=s["schedule"],
                gates_per_layer=s["gates_per_layer"] if s["schedule"] == "random_edges" else None,
                reference=s["reference"],
                reference_partner=s["reference_partner"],
                scramble_gates=s["scramble_gates"],
                scramble_layers=s["scramble_layers"],
                gate_set=z["gate_set"],
                random_orientation=s["random_orientation"],
                checkpoints=z["checkpoint_layers"].astype(np.int64),
                probes=probes,
                stop_when_purified=s["stop_when_purified"],
                system_order=z["system_order"],
                periodic=s["periodic"],
                record_profile=s["record_profile"],
            )
            values = {p.name: z[f"value::{p.name}"] for p in probes}
            return cls(
                spec=spec,
                seed_entropy=meta["seed_entropy"],
                trajectories=z["trajectories"],
                checkpoint_layers=z["checkpoint_layers"],
                values=values,
                profiles=z["profiles"] if "profiles" in z.files else None,
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
            profiles=cat(lambda r: r.profiles),
            purification_layer=cat(lambda r: r.purification_layer),
            final_layer=cat(lambda r: r.final_layer),
            num_measurements=cat(lambda r: r.num_measurements),
            elapsed_seconds=sum(r.elapsed_seconds for r in results),
        )
