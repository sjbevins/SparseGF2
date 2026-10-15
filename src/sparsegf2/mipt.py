"""Ensembles of random monitored Clifford circuits, at full speed.

Describe the protocol once with :class:`MonitoredCircuit`, choose what to record
with probes, and run as many trajectories as you need::

    from sparsegf2 import graphs, mipt

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
   layers of the dynamics below (the first ``scramble_layers`` layers of the
   schedule, without measurements; the dynamics continue from there).
3. **Dynamics**, ``layers`` times. Gates on edges of the graph, chosen by
   ``schedule``:

   * ``"matching"``: cycle through an edge colouring of the graph (on an even
     ring this is the standard brickwork, alternating even and odd bonds). The
     cycle runs on through the scramble layers into the dynamics, so with
     ``scramble_layers=s`` the first measured layer uses colour ``s mod C``;
   * ``"random_edges"``: ``gates_per_layer`` edges drawn with replacement,
     uniformly or with probabilities ``edge_weights`` (long-range circuits:
     ``graphs.complete(n)`` with ``graphs.power_law_weights``);
   * ``"all_edges"``: every edge, in order.

   Each gate is a uniformly random two-qubit Clifford (or uniform over
   ``gate_set``). Then each system qubit is measured in the Z basis with
   probability ``measurement_rate``.
4. **Checkpoints.** After the layers listed in ``checkpoints`` (layer 0 is the
   state after setup and scramble), every probe is evaluated, except probes
   given their own ``checkpoints`` (e.g. a cheap probe every layer, an
   expensive one at a few depths). With a reference, the first checkpoint at
   which the reference is pure (code dimension 0) is the purification layer;
   ``stop_when_purified`` ends the trajectory there, and later evaluations
   record each probe's exact value for a pure reference (k = 0, nothing lost,
   first failure n + 1), or NaN for entanglement probes, which keep changing.

A "layer" is exactly one step 3, never rescaled. Everything runs phase-free:
entanglement does not depend on measurement outcomes.

Graph disorder: ``edges`` may be a collection of G graphs (a list), each run
with ``circuits_per_graph`` random circuits (default 1): trajectory ``t`` uses
graph ``t // circuits_per_graph`` and ``run()`` runs all ``G *
circuits_per_graph`` trajectories by default. Build the graphs with their own
seed (``[graphs.watts_strogatz(n, 4, beta, seed=(17, i)) for i in range(G)]``):
graph disorder never draws from the circuit's random streams, so the same
collection can be reused at every measurement rate.

Erasure: with a reference, the code the circuit has generated at a checkpoint
is probed by ``erasure_first_failure`` (first-failure sizes over random
orderings, from which :mod:`sparsegf2.analysis` estimates P_C(m), the
failure probability Phi_C(e) under independent erasure, and e_delta),
``uncorrectable_region_counts`` (exhaustive small erasures: exact P_C(m)) and
``code_snapshot`` (the code itself, for :class:`sparsegf2.codes.StabilizerCode`).

Randomness: trajectory ``t`` of ``run(..., seed=s)`` is a deterministic function
of ``(spec, s, t)``, independent of the thread count and of the other
trajectories, so ensembles can be split into chunks (``first_trajectory``) and
merged without changing any result. The measurement mask uses one uniform draw per
qubit per layer whatever the rate, so runs at different ``measurement_rate``
with the same seed share their random numbers (the measured set at a lower rate
is a subset of that at a higher rate). The sampled probes draw their orderings
from a stream of their own at each evaluation, keyed by the trajectory, the
probe's name and the layer, so other probes and other checkpoints never change
them (:data:`RANDOMNESS_CONTRACT`).
"""

from __future__ import annotations

import hashlib
import json
import operator
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2._kernel import ensemble as _ens
from sparsegf2._kernel.tables import ANF1, ANF2, IDENTITY_SYM1, IDENTITY_SYM2, LIN1, LIN2
from sparsegf2._parallel import threads
from sparsegf2._version import __version__
from sparsegf2.analysis import exhaustive_cutoff
from sparsegf2.circuit import Circuit
from sparsegf2.clifford import num_cliffords
from sparsegf2.gates import gate
from sparsegf2.graphs import edge_coloring, from_edges

__all__ = [
    "EnsembleResult",
    "RANDOMNESS_CONTRACT",
    "MonitoredCircuit",
    "Probe",
    "code_dimension",
    "code_snapshot",
    "contiguous_distance",
    "entropy",
    "entropy_profile",
    "erasure_first_failure",
    "erasure_lost_profile",
    "half_cut_entropy",
    "lost_information",
    "mutual_information",
    "tripartite_information",
    "uncorrectable_region_counts",
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
_H_CLIFFORD, _CX_CLIFFORD = gate("H").clifford, gate("CX").clifford
assert _H_CLIFFORD is not None and _CX_CLIFFORD is not None, "H and CX are unitary gates"
_H_INDEX = _H_CLIFFORD.index()
_CX_INDEX = _CX_CLIFFORD.index()

#: Version of the package's random streams: the same contract and seed give the
#: same random numbers. It covers
#:
#: * the generator: xoshiro256** and its draw conventions (``next_u64``,
#:   ``next_double``, ``next_below``, ``next_bit`` in ``sparsegf2._kernel.rng``);
#: * the stream derivation ``stream_state(key, index)`` and the index mix
#:   ``mix(a, b, c)`` (exact formulas in the ``_kernel.rng`` docstring);
#: * every draw made through ``stream_state``: the ensemble circuit streams
#:   (trajectory ``t`` uses stream ``t`` of words 0 and 1 of the run's
#:   ``SeedSequence``, which fixes ``run``, ``replay`` and ``circuit``), the
#:   ensemble probe streams (one per sampled-probe evaluation: stream
#:   ``mix(t, name hash, layer)`` of words 2 and 3), the shot streams of
#:   ``sparsegf2.sample`` (shot ``s`` uses stream ``s`` of ``stream_key(seed)``)
#:   and the sampled observables ``random_erasure_statistics`` and
#:   ``erasure_profile`` (stream 0 of ``stream_key(seed)``);
#: * the order of the draws within a trajectory (``_kernel.ensemble``
#:   docstring) and within a sampled evaluation.
#:
#: Generators seeded directly from a ``SeedSequence`` (``Simulator``'s own
#: stream, the graph families, ``StabilizerCode.first_failure_sizes``) use the
#: same generator but not ``stream_state``; contract 2 left them unchanged.
#: Bump this whenever a change gives any covered draw other random numbers;
#: results and output directories record it and refuse to merge, resume or
#: replay (``EnsembleResult.replay``) across the change: this version cannot
#: reproduce a run made under another contract. Contract 2 mixes all 128 key
#: bits into every stream and gives each sampled-probe evaluation its own
#: stream; contract 1 folded the key to ``key0 ^ key1`` and read one probe
#: stream per trajectory in sequence.
RANDOMNESS_CONTRACT = 2


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """Something to record at checkpoints. Build one with the functions below.

    Each probe records a fixed number of values per evaluation (one for a
    scalar probe; one per region size, permutation sample or erased-region
    size for a vector probe). It is evaluated at the spec's ``checkpoints``,
    or at its own ``checkpoints`` (same forms: ``"final"``, ``"every_layer"``,
    a stride or a list of layers), so expensive probes can run at a few
    depths while cheap ones run at every layer.
    """

    kind: str
    name: str
    regions: tuple[tuple[int, ...], ...] = ()
    size: int = 0
    samples: int = 0
    checkpoints: str | int | tuple[int, ...] | None = None
    # uncorrectable_region_counts given a budget or a mapping from n: the input as
    # ("max_size" or "budget", one int or sorted (n, value) pairs). The spec resolves it
    # into ``size`` (and the default name) for its num_system; it stays on the resolved
    # probe so that a spec of another size resolves it again. It is not compared: probes
    # that record the same sizes under the same name are the same probe, however w was
    # chosen.
    cutoff: tuple[str, int | tuple[tuple[int, int], ...]] | None = field(
        default=None, compare=False
    )

    @property
    def is_vector(self) -> bool:
        return _KINDS[self.kind].length != "one"


def _region(qubits: Sequence[int]) -> tuple[int, ...]:
    # operator.index, not int(): a fractional index is an error, never truncated
    return tuple(operator.index(q) for q in qubits)


def _schedule(checkpoints) -> str | int | tuple[int, ...] | None:
    """A probe's own checkpoints in a hashable form (lists become tuples)."""
    if checkpoints is None or isinstance(checkpoints, str):
        return checkpoints
    if isinstance(checkpoints, (int, np.integer, float, np.floating)):
        return _integer(checkpoints, "checkpoints (a stride)")
    return tuple(_integer(c, "checkpoints entries") for c in checkpoints)


def entropy(region: Sequence[int], name: str | None = None, *, checkpoints=None) -> Probe:
    """Entanglement entropy S(A) of the given qubits (system and/or reference)."""
    r = _region(region)
    return Probe("entropy", name or f"entropy{list(r)}", (r,), checkpoints=_schedule(checkpoints))


def half_cut_entropy(*, checkpoints=None) -> Probe:
    """S of the first half of the system (in ``system_order``), the usual MIPT order parameter."""
    return Probe("half_cut", "half_cut_entropy", checkpoints=_schedule(checkpoints))


def mutual_information(
    a: Sequence[int], b: Sequence[int], name: str | None = None, *, checkpoints=None
) -> Probe:
    """I(A:B) = S(A) + S(B) - S(AB) for disjoint regions."""
    ra, rb = _region(a), _region(b)
    return Probe(
        "mutual_information",
        name or f"mutual_information{[list(ra), list(rb)]}",
        (ra, rb),
        checkpoints=_schedule(checkpoints),
    )


def tripartite_information(
    a: Sequence[int] | None = None,
    b: Sequence[int] | None = None,
    c: Sequence[int] | None = None,
    name: str | None = None,
    *,
    checkpoints=None,
) -> Probe:
    """I3(A:B:C) for disjoint regions; the standard critical-point locator.

    Without arguments, A, B, C are the first three quarters of the system in
    ``system_order`` (the usual choice for a ring split into four arcs).
    """
    if a is None and b is None and c is None:
        return Probe(
            "tripartite", name or "tripartite_information", checkpoints=_schedule(checkpoints)
        )
    if a is None or b is None or c is None:
        raise ValueError("give all three regions, or none for the default quarters")
    regions = (_region(a), _region(b), _region(c))
    return Probe(
        "tripartite",
        name or f"tripartite_information{[list(r) for r in regions]}",
        regions,
        checkpoints=_schedule(checkpoints),
    )


def entropy_profile(*, checkpoints=None) -> Probe:
    """Vector: ``S(system_order[:m])`` for every m = 0..n (entanglement of every bipartition)."""
    return Probe("entropy_profile", "entropy_profile", checkpoints=_schedule(checkpoints))


def code_dimension(*, checkpoints=None) -> Probe:
    """Number of logical qubits k = S(R) of the code the circuit has generated. Needs a reference.

    The code rate is k / n; ``k > 0`` says whether the circuit produced a code at all.
    """
    return Probe("code_dimension", "code_dimension", checkpoints=_schedule(checkpoints))


def contiguous_distance(*, checkpoints=None) -> Probe:
    """Shortest contiguous run (in ``system_order``, wrapping if ``periodic``) whose erasure loses logical info; NaN without a code.

    The distance is undefined without a code: where the code dimension is 0 the
    probe records NaN, so means (:meth:`EnsembleResult.mean`, the API
    summaries) average only over trajectories that still hold a code.
    """
    return Probe("contiguous_distance", "contiguous_distance", checkpoints=_schedule(checkpoints))


def lost_information(region: Sequence[int], name: str | None = None, *, checkpoints=None) -> Probe:
    """l_C(A) = I(A:R) for a fixed set A of system qubits: logical information lost by erasing A.

    The erasure of A is correctable exactly when this is 0.
    """
    r = _region(region)
    return Probe(
        "lost_information",
        name or f"lost_information{list(r)}",
        (r,),
        checkpoints=_schedule(checkpoints),
    )


def erasure_lost_profile(samples: int = 10, name: str | None = None, *, checkpoints=None) -> Probe:
    """Vector: mean I(R:A_m) over random erasures A_m of every size m = 0..n.

    Each sample erases the prefixes of one uniformly random ordering of the
    system, which gives a uniformly random erasure of every size in a single
    incremental pass.
    """
    return Probe(
        "erasure_lost_profile",
        name or "erasure_lost_profile",
        (),
        0,
        _integer(samples, "samples"),
        _schedule(checkpoints),
    )


def erasure_first_failure(
    samples: int = 100, name: str | None = None, *, checkpoints=None
) -> Probe:
    """Vector of ``samples`` first-failure sizes, one per uniformly random ordering of the system.

    For an ordering pi, the first-failure size is the smallest m such that
    erasing pi_1..pi_m is uncorrectable (I(A:R) > 0); it is n + 1 when the code
    has no logical qubits (then no erasure ever fails). The erased regions
    are nested, so the fraction of samples with D <= m estimates P_C(m), the
    probability that a uniformly random m-qubit erasure is uncorrectable, for
    every m at once, and ``analysis.erasure_failure_probability`` turns the
    samples into Phi_C(e) for independent erasure with probability e. Each
    scan stops at its first failure.
    """
    return Probe(
        "erasure_first_failure",
        name or "erasure_first_failure",
        (),
        0,
        _integer(samples, "samples"),
        _schedule(checkpoints),
    )


def uncorrectable_region_counts(
    max_size: int | Mapping[int, int] | None = None,
    name: str | None = None,
    *,
    budget: int | Mapping[int, int] | None = None,
    checkpoints=None,
) -> Probe:
    """Vector: the number of uncorrectable m-qubit erasures for m = 1..w, by exhaustion.

    Entry ``m - 1`` counts the m-subsets A of the system with I(A:R) > 0, so
    ``count / C(n, m)`` is the exact P_C(m), and all counts are zero exactly
    when every erasure of at most w qubits is correctable (code distance > w).

    The cutoff w = m*(n) splits P_C(m) into an exact regime, m <= w, and a
    sampled one above (:func:`erasure_first_failure`). Give at most one of

    * ``max_size``: w itself (default 2);
    * ``budget``: the number B of regions to enumerate per evaluation, which
      gives w = max{w : sum_{m=1}^{w} C(n, m) <= B}
      (:func:`sparsegf2.analysis.exhaustive_cutoff`). B counts regions, not
      seconds: the time per region depends on n and w.

    Either may be one int or a mapping from the system size n to a value, so
    one probe serves a sweep over n: ``max_size={64: 3, 128: 2, 256: 2}`` or
    ``budget={64: 20_000, 256: 50_000}``. A budget or a mapping is resolved by
    :class:`MonitoredCircuit` for its ``num_system``. The default name is
    ``uncorrectable_region_counts[w]`` with the resolved w; pass ``name`` to
    keep one key at every n.

    The cost grows like C(n, w - 1): for n = 256 (one thread), about 1 ms for
    w = 2, 40 to 50 ms for 3 and about 7 s for 4 per evaluation, so give
    larger sizes their own sparse ``checkpoints``.
    """
    if max_size is not None and budget is not None:
        raise ValueError("give max_size or budget, not both")
    if budget is None and not isinstance(max_size, Mapping):
        w = _integer(2 if max_size is None else max_size, "max_size")
        return Probe(
            "uncorrectable_counts",
            name or f"uncorrectable_region_counts[{w}]",
            (),
            w,
            0,
            _schedule(checkpoints),
        )
    form = "max_size" if budget is None else "budget"
    cutoff = (form, _cutoff_value(max_size if budget is None else budget, form))
    return Probe(
        "uncorrectable_counts",
        name or _unresolved_name(cutoff),
        (),
        0,
        0,
        _schedule(checkpoints),
        cutoff,
    )


def _cutoff_value(value, form: str) -> int | tuple[tuple[int, int], ...]:
    """A budget (an int) or a mapping {n: value} as sorted ``(n, value)`` pairs, validated.

    A budget must be non-negative. A mapping's entries are checked against
    their own n: a ``max_size`` must lie in [1, n].
    """
    if not isinstance(value, Mapping):  # one budget (one max_size needs no resolving)
        b = _integer(value, form)
        if b < 0:
            raise ValueError(f"{form} must be non-negative, got {b}")
        return b
    if not value:
        raise ValueError(f"{form}: give at least one system size")
    pairs = []
    for key, v in value.items():
        try:
            n = _integer(key, form)
        except TypeError:
            raise TypeError(
                f"the system sizes (keys) of {form} must be integers, got {key!r}"
            ) from None
        if n < 2:
            raise ValueError(f"the system sizes (keys) of {form} must be at least 2, got {n}")
        x = _integer(v, f"{form}[{n}]")
        if form == "max_size" and not 1 <= x <= n:
            raise ValueError(f"max_size[{n}] must be in [1, {n}], got {x}")
        if form == "budget" and x < 0:
            raise ValueError(f"budget[{n}] must be non-negative, got {x}")
        pairs.append((n, x))
    return tuple(sorted(pairs))


def _unresolved_name(cutoff: tuple) -> str:
    """The default name of a counts probe before its size is known: its cutoff, as given."""
    form, value = cutoff
    shown = dict(value) if isinstance(value, tuple) else value
    return f"uncorrectable_region_counts({form}={shown})"


def _resolve_cutoff(p: Probe, n: int) -> Probe:
    """``p`` with ``size`` (and a default name) set from its budget or mapping for ``n`` qubits."""
    assert p.cutoff is not None, "only probes with a cutoff are resolved"
    form, value = p.cutoff
    if isinstance(value, tuple):
        given = dict(value)
        if n not in given:
            raise ValueError(
                f"probe {p.name!r}: {form} has no entry for num_system = {n}; it covers the "
                f"system sizes {sorted(given)}"
            )
        value = given[n]
    if form == "budget":
        w = exhaustive_cutoff(n, value)
        if w == 0:
            raise ValueError(
                f"probe {p.name!r}: budget {value} allows no exhaustive region size at n = {n} "
                f"(C({n}, 1) = {n} > {value}); raise the budget or drop the probe"
            )
    else:
        w = value
    # A default name follows the resolved size; a name the user chose stays.
    default = p.name in (_unresolved_name(p.cutoff), f"uncorrectable_region_counts[{p.size}]")
    return replace(p, size=w, name=f"uncorrectable_region_counts[{w}]" if default else p.name)


def code_snapshot(name: str | None = None, *, checkpoints=None) -> Probe:
    """The code itself: its stabilizer generators, saved for analysis after the run.

    The recorded value is the number of generators (n - k); the generators
    are in ``EnsembleResult.snapshots`` and ``EnsembleResult.code(...)`` returns
    a :class:`~sparsegf2.codes.StabilizerCode`. Without a reference this is
    the stabilizer group of the (pure) system state. One snapshot probe per
    spec.
    """
    return Probe("code_snapshot", name or "code_snapshot", checkpoints=_schedule(checkpoints))


@dataclass(frozen=True)
class _Kind:
    """How the engine treats one probe kind (one table, so adding a probe is one entry here
    plus its kernel branch)."""

    code: int  # kernel probe code (``_kernel.ensemble.PROBE_*``)
    reference: bool  # needs a reference qubit system
    length: str  # values per evaluation: "one", "profile" (n + 1), "samples", "size"
    after_purification: str  # value once the reference is pure: "nan", "zero", "n", "n+1"
    sampled: bool = False  # draws random orderings: Probe.samples must be at least 1


_KINDS = {
    "entropy": _Kind(_ens.PROBE_ENTROPY, False, "one", "nan"),
    "half_cut": _Kind(_ens.PROBE_ENTROPY, False, "one", "nan"),
    "mutual_information": _Kind(_ens.PROBE_MUTUAL_INFORMATION, False, "one", "nan"),
    "tripartite": _Kind(_ens.PROBE_TRIPARTITE, False, "one", "nan"),
    "entropy_profile": _Kind(_ens.PROBE_ENTROPY_PROFILE, False, "profile", "nan"),
    "code_dimension": _Kind(_ens.PROBE_CODE_DIMENSION, True, "one", "zero"),
    "contiguous_distance": _Kind(_ens.PROBE_CONTIGUOUS_DISTANCE, True, "one", "nan"),
    "lost_information": _Kind(_ens.PROBE_LOST_INFORMATION, True, "one", "zero"),
    "erasure_lost_profile": _Kind(_ens.PROBE_ERASURE_LOST_PROFILE, True, "profile", "zero", True),
    "erasure_first_failure": _Kind(_ens.PROBE_FIRST_FAILURE, True, "samples", "n+1", True),
    "uncorrectable_counts": _Kind(_ens.PROBE_UNCORRECTABLE_COUNTS, True, "size", "zero"),
    "code_snapshot": _Kind(_ens.PROBE_CODE_SNAPSHOT, False, "one", "n"),
}


def _probe_length(p: Probe, n: int) -> int:
    rule = _KINDS[p.kind].length
    return {"one": 1, "profile": n + 1, "samples": p.samples, "size": p.size}[rule]


def _probe_fill(p: Probe, n: int) -> float:
    """The probe's exact value once the reference is pure (k = 0), or NaN if not determined."""
    rule = _KINDS[p.kind].after_purification
    return {"nan": np.nan, "zero": 0.0, "n": float(n), "n+1": float(n + 1)}[rule]


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
        :mod:`sparsegf2.graphs`), or a collection of graphs (a list of such
        arrays) for graph disorder.
    circuits_per_graph
        With a collection of G graphs: how many random circuits run on each
        (default 1). Trajectory ``t`` runs on graph ``t // circuits_per_graph``,
        so the ensemble has ``G * circuits_per_graph`` trajectories, and the
        same collection can be reused unchanged at every measurement rate.
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
        dynamics (the ``"matching"`` colour cycle continues from them into the
        measured layers).
    gate_set
        Canonical two-qubit Clifford indices to draw gates from uniformly
        (default: all 11520, i.e. the uniform two-qubit Clifford group).
    random_orientation
        Randomly swap each gate's two qubits (matters only for non-uniform
        ``gate_set``).
    checkpoints
        ``"final"`` (default), ``"every_layer"``, an int stride ``k`` (layers
        0, k, 2k, ... and the last), or an explicit list of layers in
        ``[0, layers]`` (sorted and deduplicated). These are the layers where
        purification is tested and where probes without their own
        ``checkpoints`` are evaluated. :attr:`checkpoint_layers` holds the
        resulting layers; ``"final"``, ``"every_layer"`` and strides are kept
        as given, so a copy made with ``dataclasses.replace(spec, layers=...)``
        follows the new depth.
    probes
        What to record (:func:`code_dimension`, ...), each at the spec's
        checkpoints or at its own.
    stop_when_purified
        End each trajectory at the first checkpoint where the reference is
        pure. Probe evaluations after that hold the value they would have had
        (k = 0: no code, no failing erasure) or NaN where it is not determined
        (entanglement of the still-evolving system).
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
    circuits_per_graph: int = 1
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
    # Per-instance caches, never copied by dataclasses.replace (init=False): a
    # derived spec always resolves and compiles its own fields.
    _checkpoint_layers: np.ndarray | None = field(
        init=False, repr=False, compare=False, default=None
    )
    _probe_layers: tuple = field(init=False, repr=False, compare=False, default=())
    _compiled: dict = field(init=False, default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        n = _integer(self.num_system, "num_system")
        if n < 2:
            raise ValueError("num_system must be at least 2")
        raw_graphs, is_bank = _graph_list(self.edges)
        graphs_ = [_read_only(from_edges(g, n)) for g in raw_graphs]
        if not graphs_ or any(len(g) == 0 for g in graphs_):
            raise ValueError("every graph needs at least one edge")
        object.__setattr__(self, "num_system", n)
        layers = _integer(self.layers, "layers")
        if layers < 0:
            raise ValueError("layers must be non-negative")
        object.__setattr__(self, "layers", layers)
        rate = _number(self.measurement_rate, "measurement_rate")
        if not 0.0 <= rate <= 1.0:
            raise ValueError("measurement_rate must be in [0, 1]")
        object.__setattr__(self, "measurement_rate", rate)
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
        per_graph = _integer(self.circuits_per_graph, "circuits_per_graph")
        if per_graph < 1:
            raise ValueError("circuits_per_graph must be at least 1")
        if not is_bank and per_graph != 1:
            raise ValueError(
                "circuits_per_graph applies to a collection of graphs (a list); with a single "
                "graph, choose the number of circuits with run(trajectories=...)"
            )
        object.__setattr__(self, "circuits_per_graph", per_graph)
        if self.schedule == "random_edges":
            gpl = (
                n // 2
                if self.gates_per_layer is None
                else _integer(self.gates_per_layer, "gates_per_layer")
            )
            if gpl < 0:
                raise ValueError("gates_per_layer must be non-negative")
            object.__setattr__(self, "gates_per_layer", gpl)
        if self.reference not in _REFERENCES:
            raise ValueError(
                f"reference must be one of {list(_REFERENCES)}, got {self.reference!r}"
            )
        if self.reference_partner != "random":
            if isinstance(self.reference_partner, str):
                raise ValueError(f"reference_partner must be 'random' or in [0, {n})")
            partner = _integer(self.reference_partner, "reference_partner")
            if self.reference == "single" and not 0 <= partner < n:
                raise ValueError(f"reference_partner must be 'random' or in [0, {n})")
            object.__setattr__(self, "reference_partner", partner)
        for name in ("scramble_gates", "scramble_matchings", "scramble_layers"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if min(self.scramble_gates, self.scramble_matchings, self.scramble_layers) < 0:
            raise ValueError("scramble_gates, scramble_matchings and scramble_layers must be >= 0")
        for name in ("random_orientation", "stop_when_purified", "periodic"):
            object.__setattr__(self, name, _flag(getattr(self, name), name))
        pool = (
            _read_only(np.arange(num_cliffords(2), dtype=np.int64))
            if self.gate_set is None
            else _int_array(self.gate_set, "gate_set")
        )
        if pool.size == 0 or pool.min() < 0 or pool.max() >= num_cliffords(2):
            raise ValueError("gate_set must be a non-empty list of indices in [0, 11520)")
        object.__setattr__(self, "gate_set", pool)
        order = (
            _read_only(np.arange(n, dtype=np.int64))
            if self.system_order is None
            else _int_array(self.system_order, "system_order")
        )
        if sorted(order.tolist()) != list(range(n)):
            raise ValueError("system_order must be a permutation of the system qubits")
        object.__setattr__(self, "system_order", order)
        checkpoints, layers_ = self._resolve_checkpoints(self.checkpoints)
        object.__setattr__(self, "checkpoints", checkpoints)
        object.__setattr__(self, "_checkpoint_layers", layers_)
        object.__setattr__(self, "probes", tuple(self._resolve_probe(p) for p in self.probes))
        names = [p.name for p in self.probes]
        if len(set(names)) != len(names):
            raise ValueError(f"probe names must be unique, got {names}")
        if sum(p.kind == "code_snapshot" for p in self.probes) > 1:
            raise ValueError("use at most one code_snapshot probe (give it all the checkpoints)")
        object.__setattr__(
            self,
            "_probe_layers",
            tuple(
                layers_ if p.checkpoints is None else self._resolve_checkpoints(p.checkpoints)[1]
                for p in self.probes
            ),
        )
        if self.stop_when_purified and self.reference == "none":
            raise ValueError("stop_when_purified needs a reference")

    def _resolve_probe(self, p: Probe) -> Probe:
        """Validate a probe and fill in default regions and the exhaustive size for this n."""
        if not isinstance(p, Probe):
            raise TypeError("probes must be created with the mipt probe functions")
        if p.kind not in _KINDS:
            raise ValueError(f"unknown probe kind {p.kind!r}")
        n = self.num_system
        assert self.system_order is not None, "system_order is resolved before probes"
        order = np.asarray(self.system_order).tolist()
        if _KINDS[p.kind].reference and self.reference == "none":
            raise ValueError(
                f"probe {p.name!r} needs a reference (reference='purification' or 'single')"
            )
        if p.checkpoints is not None:
            self._resolve_checkpoints(p.checkpoints)  # validates against layers
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
        if _KINDS[p.kind].sampled and p.samples < 1:
            raise ValueError(f"probe {p.name!r}: samples must be at least 1")
        if p.kind == "uncorrectable_counts":
            if p.cutoff is not None:  # a budget or a mapping: its size at this n
                p = _resolve_cutoff(p, n)
            if not 1 <= p.size <= n:
                raise ValueError(f"probe {p.name!r}: max_size must be in [1, {n}], got {p.size}")
        return p

    def _resolve_checkpoints(self, spec) -> tuple[str | int | np.ndarray, np.ndarray]:
        """The checkpoint spec in normal form, and the (read-only) layers it selects.

        ``"final"``, ``"every_layer"`` and strides stay symbolic, so a copy with
        another ``layers`` resolves them again; an explicit list becomes its
        sorted, deduplicated layers (the engine relies on that order).
        """
        explicit = False
        if isinstance(spec, str):
            if spec == "final":
                layers = np.array([self.layers], dtype=np.int64)
            elif spec == "every_layer":
                layers = np.arange(self.layers + 1, dtype=np.int64)
            else:
                raise ValueError(
                    f"checkpoints must be 'final', 'every_layer', a stride or a list, got {spec!r}"
                )
        elif isinstance(spec, (int, np.integer, float, np.floating)) and not isinstance(
            spec, (bool, np.bool_)
        ):
            spec = _integer(spec, "checkpoints (a stride)")
            if spec < 1:
                raise ValueError("a checkpoint stride must be positive")
            layers = np.unique(np.append(np.arange(0, self.layers + 1, spec), self.layers))
        else:
            # np.unique sorts and drops repeats: the engine matches checkpoints in
            # increasing layer order, so a repeated or out-of-order entry would
            # silently skip every later checkpoint.
            layers = np.unique(_int_array(spec, "checkpoints"))
            explicit = True
        if layers.size == 0 or np.min(layers) < 0 or np.max(layers) > self.layers:
            raise ValueError(f"checkpoints must lie in [0, layers={self.layers}]")
        layers = _read_only(layers.astype(np.int64))
        return (layers if explicit else spec), layers

    # -- derived quantities ----------------------------------------------------------

    @property
    def graphs(self) -> tuple[np.ndarray, ...]:
        """The interaction graphs (one, or the collection); trajectory t uses graph t // circuits_per_graph."""
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
        """The layers at which probes are evaluated (sorted, read-only)."""
        assert self._checkpoint_layers is not None, "resolved in __post_init__"
        return self._checkpoint_layers

    @property
    def num_trajectories(self) -> int | None:
        """``len(edges) * circuits_per_graph`` for a collection of graphs; None for a single graph."""
        if not isinstance(self.edges, tuple):
            return None
        return len(self.edges) * self.circuits_per_graph

    def probe_layers(self, name: str) -> np.ndarray:
        """The layers at which the probe ``name`` is evaluated (sorted, read-only)."""
        for p, layers in zip(self.probes, self._probe_layers, strict=True):
            if p.name == name:
                return layers
        raise KeyError(f"no probe named {name!r}; probes: {[p.name for p in self.probes]}")

    @property
    def scalar_probes(self) -> list[Probe]:
        return [p for p in self.probes if not p.is_vector]

    @property
    def vector_probes(self) -> list[Probe]:
        return [p for p in self.probes if p.is_vector]

    def with_edges(
        self,
        edges: np.ndarray,
        edge_weights: np.ndarray | None = None,
        circuits_per_graph: int | None = None,
    ) -> MonitoredCircuit:
        """The same protocol on other graph(s) (one graph, or a list of graphs).

        ``circuits_per_graph`` defaults to this protocol's value for a list of
        graphs and to 1 for a single graph.
        """
        if circuits_per_graph is None:
            circuits_per_graph = self.circuits_per_graph if _graph_list(edges)[1] else 1
        return replace(
            self, edges=edges, edge_weights=edge_weights, circuits_per_graph=circuits_per_graph
        )

    def _arrays(self) -> dict:
        if not self._compiled:
            sizes = np.array([len(g) for g in self.graphs], dtype=np.int64)
            graph_ptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
            if self.schedule == "matching":
                color_ptr_ptr = np.zeros(self.num_graphs + 1, dtype=np.int64)
                ptr_parts, idx_parts = [], []
                for g, edges in enumerate(self.graphs):
                    colors = edge_coloring(edges, self.num_system)
                    ptr_parts.append(np.concatenate([[0], np.cumsum([len(c) for c in colors])]))
                    color_ptr_ptr[g + 1] = color_ptr_ptr[g] + len(colors) + 1
                    idx_parts.append(np.concatenate(colors))
                color_ptr = np.concatenate(ptr_parts).astype(np.int64)
                color_idx = np.concatenate(idx_parts).astype(np.int64)
            else:
                # Only the matching schedule reads the colouring: give every graph
                # one colour holding all its edges instead of colouring it (pure
                # Python, and the main cost of a large graph bank).
                color_ptr = np.stack([np.zeros_like(sizes), sizes], axis=1).ravel()
                color_ptr_ptr = 2 * np.arange(self.num_graphs + 1, dtype=np.int64)
                color_idx = np.arange(graph_ptr[-1], dtype=np.int64) - np.repeat(
                    graph_ptr[:-1], sizes
                )
            edge_cum = []
            weights = self.edge_weights
            if weights is not None:
                for w in weights if isinstance(weights, tuple) else (weights,):
                    cum = np.cumsum(w / w.sum())
                    cum[-1] = 1.0
                    edge_cum.append(cum)
            self._compiled.update(
                edges=np.concatenate(self.graphs).astype(np.int64),
                graph_ptr=graph_ptr,
                color_ptr=color_ptr,
                color_ptr_ptr=color_ptr_ptr,
                color_idx=color_idx,
                edge_cum=np.concatenate(edge_cum) if edge_cum else np.zeros(0, dtype=np.float64),
                # writable copies of the spec's read-only arrays, for the kernels
                pool=np.array(self.gate_set, dtype=np.int64),
                order=np.array(self.system_order, dtype=np.int64),
                checkpoints=np.array(self.checkpoint_layers, dtype=np.int64),
            )
            n = self.num_system
            ptr: list[int] = []
            idx: list[int] = []
            region_len = np.zeros((len(self.probes), 3), dtype=np.int64)
            for j, p in enumerate(self.probes):
                ptr.append(len(idx))
                for r, region in enumerate(p.regions[:3]):
                    idx.extend(region)
                    region_len[j, r] = len(region)
            plen = np.array([_probe_length(p, n) for p in self.probes], dtype=np.int64)
            counts = np.array([len(layers) for layers in self._probe_layers], dtype=np.int64)
            poff = np.concatenate([[0], np.cumsum(counts * plen)]).astype(np.int64)
            pck_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            union = np.unique(np.concatenate([self.checkpoint_layers, *self._probe_layers]))
            snapshot = [j for j, p in enumerate(self.probes) if p.kind == "code_snapshot"]
            self._compiled.update(
                pkind=np.array([_KINDS[p.kind].code for p in self.probes], dtype=np.int64),
                plen=plen,
                poff=poff[:-1].copy(),
                total_len=int(poff[-1]),
                pck_ptr=pck_ptr,
                pck=np.concatenate([np.zeros(0, np.int64), *self._probe_layers]).astype(np.int64),
                pfill=np.array([_probe_fill(p, n) for p in self.probes], dtype=np.float64),
                gck=union.astype(np.int64),
                detect=np.isin(union, self.checkpoint_layers),
                ptr=np.array(ptr, dtype=np.int64),
                region_len=region_len,
                idx=np.array(idx, dtype=np.int64),
                size=np.array([p.size for p in self.probes], dtype=np.int64),
                samples=np.array([p.samples for p in self.probes], dtype=np.int64),
                phash=np.array([_name_hash(p.name) for p in self.probes], dtype=np.uint64),
                snapshot=snapshot[0] if snapshot else -1,
                partner=-1 if self.reference_partner == "random" else int(self.reference_partner),
                gpl=int(self.gates_per_layer or 0),
            )
        return self._compiled

    # -- running ---------------------------------------------------------------------

    def _check_trajectories(self, first: int, count: int) -> None:
        total = self.num_trajectories
        if first < 0 or count < 0:
            raise ValueError("first_trajectory and trajectories must be non-negative")
        if total is not None and first + count > total:
            raise ValueError(
                f"trajectories {first}..{first + count - 1} do not exist: this collection of "
                f"{len(self.edges)} graphs with circuits_per_graph={self.circuits_per_graph} "
                f"has {total} trajectories (raise circuits_per_graph for more circuits per graph)"
            )

    def run(
        self,
        trajectories: int | None = None,
        *,
        seed,
        first_trajectory: int = 0,
        num_threads: int | None = None,
    ) -> EnsembleResult:
        """Simulate trajectories ``first_trajectory .. first_trajectory + trajectories - 1``.

        With a collection of graphs, ``trajectories`` defaults to every
        remaining trajectory (all ``len(edges) * circuits_per_graph`` of them
        for ``first_trajectory=0``); with a single graph it is required. Runs in
        parallel over CPU cores (``num_threads`` limits that). Returns an
        :class:`EnsembleResult`.
        """
        first_trajectory = _integer(first_trajectory, "first_trajectory")
        if trajectories is None:
            if self.num_trajectories is None:
                raise ValueError("with a single graph, give the number of trajectories")
            trajectories = self.num_trajectories - first_trajectory
        trajectories = _integer(trajectories, "trajectories")
        self._check_trajectories(first_trajectory, trajectories)
        start = time.perf_counter()  # includes the one-off graph preprocessing below
        a = self._arrays()
        # One SeedSequence for both the random streams and the record, so the
        # recorded seed is the one used (also for seed=None or spawned sequences).
        ss = _seed_sequence(seed)
        key_struct, key_probe = _keys(ss)
        out = np.full((trajectories, a["total_len"]), np.nan, dtype=np.float64)
        n = self.num_system
        if a["snapshot"] >= 0:
            wsys = (n + 63) // 64
            evaluations = len(self._probe_layers[a["snapshot"]])
            snap = np.zeros((trajectories, evaluations, n, 2 * wsys), dtype=np.uint64)
            snap_rows = np.zeros((trajectories, evaluations), dtype=np.int64)
        else:  # one dummy slot: the kernel always has somewhere to point
            snap = np.zeros((trajectories, 1, 1, 2), dtype=np.uint64)
            snap_rows = np.zeros((trajectories, 1), dtype=np.int64)
        tau = np.zeros(trajectories, dtype=np.int64)
        final = np.zeros(trajectories, dtype=np.int64)
        meas = np.zeros(trajectories, dtype=np.int64)
        if trajectories:
            with threads(num_threads):
                _ens.run_ensemble(
                    n, self.num_reference, _REFERENCES[self.reference], a["partner"],
                    a["edges"], a["graph_ptr"], self.circuits_per_graph, a["edge_cum"],
                    a["color_ptr"], a["color_ptr_ptr"], a["color_idx"],
                    _SCHEDULES[self.schedule], a["gpl"], a["pool"], float(self.measurement_rate),
                    int(self.layers), int(self.scramble_gates), int(self.scramble_matchings),
                    int(self.scramble_layers), bool(self.random_orientation), a["gck"],
                    a["detect"], bool(self.stop_when_purified), a["pkind"], a["plen"], a["poff"],
                    a["pck_ptr"], a["pck"], a["pfill"], a["ptr"], a["region_len"], a["idx"],
                    a["size"], a["samples"], a["order"], bool(self.periodic), key_struct,
                    key_probe, a["phash"], first_trajectory, out, snap, snap_rows, tau, final,
                    meas,
                    _H_INDEX, _CX_INDEX, LIN1, ANF1, IDENTITY_SYM1, LIN2, ANF2, IDENTITY_SYM2,
                )  # fmt: skip
        elapsed = time.perf_counter() - start
        values, snapshots = {}, {}
        for j, p in enumerate(self.probes):
            count, length = len(self._probe_layers[j]), int(a["plen"][j])
            block = out[:, a["poff"][j] : a["poff"][j] + count * length]
            block = block.reshape(trajectories, count, length)
            values[p.name] = block if p.is_vector else block[:, :, 0]
            if j == a["snapshot"]:
                snapshots[p.name] = (snap, snap_rows)
        return EnsembleResult(
            spec=self,
            seed_entropy=ss.entropy,
            seed_spawn_key=ss.spawn_key,
            seed_pool_size=ss.pool_size,
            trajectories=np.arange(
                first_trajectory, first_trajectory + trajectories, dtype=np.int64
            ),
            checkpoint_layers=self.checkpoint_layers.copy(),
            values=values,
            purification_layer=tau if self.reference != "none" else None,
            final_layer=final,
            num_measurements=meas,
            elapsed_seconds=elapsed,
            snapshots=snapshots,
            randomness_contract=RANDOMNESS_CONTRACT,
        )

    def replay(self, trajectory: int, *, seed, probes=None, checkpoints=None) -> EnsembleResult:
        """Trajectory ``trajectory`` of ``run(seed=seed)`` again, with other probes or checkpoints.

        Probes and checkpoints never change a trajectory's circuit or random
        numbers, so this is exactly that trajectory, observed differently: a
        one-row :class:`EnsembleResult`. This is how a trajectory is probed at
        new depths after the fact. The sampled probes are reproduced too: an
        evaluation's orderings depend only on the trajectory, the probe's name,
        its ``samples`` and the layer, so the same probe at a layer the original
        run evaluated gives the same values.

        All of this holds only when the run was made under this version's
        :data:`RANDOMNESS_CONTRACT` (``result.randomness_contract ==
        RANDOMNESS_CONTRACT``). This version cannot reproduce the trajectories
        of a run made under contract 1: the same seed gives other circuits and
        samples, and nothing here can tell. For a saved run use
        :meth:`EnsembleResult.replay`, which takes the run's own spec and seed
        and refuses a result made under another contract.
        """
        trajectory = _integer(trajectory, "trajectory")
        self._check_trajectories(trajectory, 1)
        spec = replace(
            self,
            probes=self.probes if probes is None else tuple(probes),
            checkpoints=self.checkpoints if checkpoints is None else checkpoints,
        )
        return spec.run(1, seed=seed, first_trajectory=trajectory)

    def circuit(self, trajectory: int, *, seed) -> Circuit:
        """The exact circuit that trajectory ``trajectory`` of ``run(seed=seed)`` simulates.

        Layout: setup gates, scramble (CLIFFORD_2Q), ``TICK`` (checkpoint layer
        0), then per layer ``CLIFFORD_2Q`` gates, ``M`` measurements and a
        ``TICK``. So tick ``t`` of ``Simulator.run(circuit, observe=...)``
        sees the state after layer ``t``.

        It is the circuit of a run made under this version's
        :data:`RANDOMNESS_CONTRACT` only: for a result made under contract 1
        (``result.randomness_contract == 1``) the same seed gives another
        circuit, and this version cannot rebuild the one that run simulated.
        """
        trajectory = _integer(trajectory, "trajectory")
        self._check_trajectories(trajectory, 1)
        a = self._arrays()
        key_struct, _ = _keys(_seed_sequence(seed))
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
            self.circuits_per_graph, a["edge_cum"], a["color_ptr"], a["color_ptr_ptr"],
            a["color_idx"],
            _SCHEDULES[self.schedule], a["gpl"], a["pool"], float(self.measurement_rate),
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
    w = np.array(weights, dtype=np.float64).ravel()  # a copy: never the caller's array
    if w.shape != (len(raw_edges),) or len(raw_edges) != len(edges):
        raise ValueError(
            "edge_weights needs one weight per edge of an already normalised edge "
            "list (pass edges from sparsegf2.graphs)"
        )
    if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("edge_weights must be finite, non-negative and not all zero")
    if not np.array_equal(raw_edges, edges):
        raise ValueError("edge_weights requires edges in normalised order (sorted, a < b)")
    return _read_only(w)


def _read_only(arr: np.ndarray) -> np.ndarray:
    """Mark an array the spec owns as read-only (the spec is frozen) and return it."""
    arr.flags.writeable = False
    return arr


def _integer(value, what: str) -> int:
    """``value`` as a Python int: numpy integers and whole-number floats are accepted."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{what} must be an integer, got {value!r}")
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{what} must be an integer, got {value!r}") from None


def _number(value, what: str) -> float:
    """``value`` as a Python float (numpy scalars accepted, strings and bools refused)."""
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise TypeError(f"{what} must be a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{what} must be a number, got {value!r}") from None


def _flag(value, what: str) -> bool:
    """``value`` as a Python bool (numpy booleans and 0/1 accepted)."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise TypeError(f"{what} must be True or False, got {value!r}")


def _int_array(values, what: str) -> np.ndarray:
    """A new read-only 1-D int64 array; non-integer entries are an error, never truncated."""
    if isinstance(values, np.ndarray):
        raw = np.array(values)
    elif isinstance(values, (str, bytes)):
        raise ValueError(f"{what} must hold integers, got {values!r}")
    else:
        try:
            raw = np.array(list(values))
        except TypeError:  # a single number
            raw = np.array([values])
    if raw.size and not np.issubdtype(raw.dtype, np.integer):
        whole = np.issubdtype(raw.dtype, np.floating) and bool(
            np.all(np.isfinite(raw)) and np.all(raw == np.round(raw))
        )
        if not whole:
            raise ValueError(f"{what} must hold integers, got {raw.tolist()}")
    return _read_only(raw.astype(np.int64).reshape(-1))


def _seed_sequence(seed) -> np.random.SeedSequence:
    return seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)


def _keys(ss: np.random.SeedSequence) -> tuple[np.ndarray, np.ndarray]:
    state = ss.generate_state(4, dtype=np.uint64)
    return state[:2].copy(), state[2:].copy()


def _name_hash(name: str) -> int:
    """The 64-bit key of a probe's sampling streams: blake2b of its name, read little-endian."""
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _plain_ints(value):
    """Seed entropy as JSON-stable Python values: an int, or a list of ints."""
    if value is None or isinstance(value, (int, np.integer)):
        return None if value is None else int(value)
    return [int(v) for v in value]


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


_SPEC_FIELDS = (
    "num_system",
    "layers",
    "measurement_rate",
    "schedule",
    "gates_per_layer",
    "circuits_per_graph",
    "reference",
    "reference_partner",
    "scramble_gates",
    "scramble_matchings",
    "scramble_layers",
    "random_orientation",
    "stop_when_purified",
    "periodic",
)


def _protocol_difference(a: MonitoredCircuit, b: MonitoredCircuit) -> str | None:
    """The first protocol field in which two specs differ (with both values), or None."""
    for name in _SPEC_FIELDS:
        va, vb = getattr(a, name), getattr(b, name)
        if va != vb:
            return f"{name} ({va!r} vs {vb!r})"
    if a.num_graphs != b.num_graphs or not all(
        np.array_equal(g, h) for g, h in zip(a.graphs, b.graphs, strict=True)
    ):
        return "edges"
    wa, wb = a.edge_weights, b.edge_weights
    if (wa is None) != (wb is None) or (
        wa is not None
        and wb is not None
        and not all(
            np.array_equal(x, y)
            for x, y in zip(
                wa if isinstance(wa, tuple) else (wa,),
                wb if isinstance(wb, tuple) else (wb,),
                strict=True,
            )
        )
    ):
        return "edge_weights"
    for name in ("gate_set", "system_order", "checkpoint_layers"):
        if not np.array_equal(getattr(a, name), getattr(b, name)):
            return name
    if a.probes != b.probes:
        return "probes"
    return None


def _seed_record(result: EnsembleResult) -> tuple:
    """What identifies the seed of a run: the SeedSequence's entropy, spawn key and pool size."""
    return result.seed_entropy, result.seed_spawn_key, result.seed_pool_size


def _cutoff_record(cutoff: tuple) -> dict:
    """A probe's cutoff as saved, in its JSON probe form: ``{"budget": 100}``, ``{"max_size": {"64": 3}}``."""
    form, value = cutoff
    return {form: {str(n): v for n, v in value} if isinstance(value, tuple) else value}


def _cutoff_from_record(record: dict | None) -> tuple | None:
    """The cutoff saved by :func:`_cutoff_record` (None for probes without one, and old files)."""
    if record is None:
        return None
    ((form, value),) = record.items()
    if isinstance(value, dict):
        value = {int(n): v for n, v in value.items()}
    return form, _cutoff_value(value, form)


def _json_default(obj):
    """``json.dumps`` fallback: numpy scalars and arrays become plain Python values."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass
class EnsembleResult:
    """Everything an ensemble run produced. Arrays have one row per trajectory.

    Attributes
    ----------
    values
        ``values[probe_name]`` has shape ``(trajectories, evaluations)`` for a
        scalar probe and ``(trajectories, evaluations, length)`` for a vector
        probe, one evaluation per layer in :meth:`layers` (the probe's own
        checkpoints, or the spec's). Evaluations after a trajectory stopped
        (``stop_when_purified``) hold the probe's exact value for a pure
        reference (k = 0: code dimension and lost information 0, first-failure
        size n + 1, no uncorrectable regions) or NaN where that is not
        determined (entanglement probes); ``contiguous_distance`` is NaN
        wherever k = 0.
    snapshots
        For a :func:`code_snapshot` probe: ``(generators, counts)``, where
        ``generators[t, c, i]`` is generator ``i`` of the code at evaluation
        ``c`` of trajectory ``t`` (X bits of the system qubits in the first half
        of the words, Z bits in the second) and ``counts[t, c]`` how many rows
        are used (-1 for evaluations after a stop). :meth:`code` decodes one.
    purification_layer
        First checkpoint layer at which the reference was pure, or -1 if it
        never was (right-censored at the last checkpoint). ``None`` without a
        reference.
    final_layer
        Last layer simulated for each trajectory.
    num_measurements
        Number of measurements made in each trajectory.
    seed_entropy, seed_spawn_key, seed_pool_size
        The ``numpy.random.SeedSequence`` the run used (also when it was
        ``seed=None`` or a spawned child): :attr:`seed_sequence` rebuilds it,
        so ``spec.run(..., seed=result.seed_sequence)`` repeats the run.
    version
        The ``sparsegf2`` version that produced the result (``None`` for
        files saved before versions were recorded).
    randomness_contract
        The :data:`RANDOMNESS_CONTRACT` the run was made under (1 for files
        saved before contracts were recorded). The same seed gives other
        trajectories under another contract, so :meth:`concatenate` refuses to
        mix contracts.
    """

    spec: MonitoredCircuit
    seed_entropy: int | Sequence[int] | None
    trajectories: np.ndarray
    checkpoint_layers: np.ndarray
    values: dict[str, np.ndarray]
    purification_layer: np.ndarray | None
    final_layer: np.ndarray
    num_measurements: np.ndarray
    elapsed_seconds: float = 0.0
    seed_spawn_key: tuple[int, ...] = ()
    seed_pool_size: int = 4
    version: str | None = __version__
    snapshots: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    randomness_contract: int = RANDOMNESS_CONTRACT

    def __post_init__(self) -> None:
        # One normal form in memory and after a JSON round trip (tuples, lists and
        # numpy integers otherwise compare unequal when chunks are merged).
        self.seed_entropy = _plain_ints(self.seed_entropy)
        self.seed_spawn_key = tuple(int(k) for k in self.seed_spawn_key)
        self.seed_pool_size = int(self.seed_pool_size)
        self.randomness_contract = _integer(self.randomness_contract, "randomness_contract")

    @property
    def num_trajectories(self) -> int:
        return int(self.trajectories.size)

    @property
    def seed_sequence(self) -> np.random.SeedSequence:
        """The ``SeedSequence`` the run used; pass it as ``seed=`` to repeat the run.

        The repeat is exact only under the run's :attr:`randomness_contract`;
        :meth:`replay` checks it.
        """
        return np.random.SeedSequence(
            self.seed_entropy, spawn_key=self.seed_spawn_key, pool_size=self.seed_pool_size
        )

    @property
    def graph_index(self) -> np.ndarray:
        """The graph each trajectory ran on (all 0 with a single graph)."""
        if not isinstance(self.spec.edges, tuple):
            return np.zeros(self.num_trajectories, dtype=np.int64)
        return np.asarray(self.trajectories, dtype=np.int64) // self.spec.circuits_per_graph

    def layers(self, name: str) -> np.ndarray:
        """The layers at which probe ``name`` was evaluated (the columns of ``values[name]``)."""
        return self.spec.probe_layers(name)

    def code(self, trajectory: int, evaluation: int, name: str | None = None):
        """The saved code of row ``trajectory`` at evaluation ``evaluation`` of a snapshot probe.

        Returns a :class:`~sparsegf2.codes.StabilizerCode`; ``trajectory`` is a
        row index into this result (see :attr:`trajectories` for the global
        index) and ``evaluation`` indexes ``layers(name)``.
        """
        from sparsegf2.codes import StabilizerCode

        if not self.snapshots:
            raise ValueError("this result has no code snapshots (add mipt.code_snapshot())")
        if name is None:
            (name,) = self.snapshots
        generators, counts = self.snapshots[name]
        rows = int(counts[trajectory, evaluation])
        if rows < 0:
            raise ValueError(
                f"trajectory row {trajectory} had stopped (reference pure) before evaluation "
                f"{evaluation}: there is no code to return"
            )
        return StabilizerCode.from_packed(
            generators[trajectory, evaluation, :rows], self.spec.num_system
        )

    def replay(self, trajectory: int, *, probes=None, checkpoints=None) -> EnsembleResult:
        """Trajectory ``trajectory`` of this run again, with other probes or checkpoints.

        :meth:`MonitoredCircuit.replay` with this result's own spec and seed
        (``self.spec.replay(trajectory, seed=self.seed_sequence, ...)``): the
        same circuit and, for the same probe at a layer the run evaluated, the
        same values, sampled probes included. ``trajectory`` is a trajectory
        index (an entry of :attr:`trajectories`; it is also the row when the
        run started at trajectory 0).

        Raises ``ValueError`` if the run was made under another
        :data:`RANDOMNESS_CONTRACT`: under another contract the same seed gives
        other circuits and samples, so this version cannot reproduce that run.
        """
        trajectory = _integer(trajectory, "trajectory")
        if self.randomness_contract != RANDOMNESS_CONTRACT:
            raise ValueError(
                f"this result was made under randomness contract {self.randomness_contract}; "
                f"this version uses {RANDOMNESS_CONTRACT}, under which its seed gives other "
                "circuits and samples, so its trajectories cannot be replayed. Rerun the "
                "protocol with this version (result.spec.run(...)) and replay that result"
            )
        held = np.asarray(self.trajectories, dtype=np.int64)
        if not np.any(held == trajectory):
            where = (
                f"{held.size} trajectories between {held.min()} and {held.max()}"
                if held.size
                else "no trajectories"
            )
            raise ValueError(
                f"trajectory {trajectory} is not in this result, which holds {where} (pass a "
                "trajectory index from result.trajectories, not a row of a chunk)"
            )
        return self.spec.replay(
            trajectory, seed=self.seed_sequence, probes=probes, checkpoints=checkpoints
        )

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
        """Average of a probe over trajectories at each of its evaluations, ignoring NaN entries.

        NaN marks entanglement probes of trajectories that had stopped
        (``stop_when_purified``) and ``contiguous_distance`` without a code.
        """
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
        """Write the arrays and run metadata (spec, seed, package version) to a compressed ``.npz`` file."""
        spec = self.spec
        meta = {
            "format": "sparsegf2.EnsembleResult/1",
            "version": self.version,
            "randomness_contract": self.randomness_contract,
            "seed_entropy": self.seed_entropy,
            "seed_spawn_key": list(self.seed_spawn_key),
            "seed_pool_size": self.seed_pool_size,
            "elapsed_seconds": self.elapsed_seconds,
            "probes": [
                {
                    "kind": p.kind,
                    "name": p.name,
                    "regions": [list(r) for r in p.regions],
                    "size": p.size,
                    "samples": p.samples,
                    "checkpoints": list(p.checkpoints)
                    if isinstance(p.checkpoints, tuple)
                    else p.checkpoints,
                    # the budget or mapping that chose ``size``, kept for another n
                    **({"cutoff": _cutoff_record(p.cutoff)} if p.cutoff is not None else {}),
                }
                for p in spec.probes
            ],
            "spec": {k: getattr(spec, k) for k in _SPEC_FIELDS},
            # "final", "every_layer" or a stride, kept symbolic; None for a list
            "checkpoints": spec.checkpoints if isinstance(spec.checkpoints, (str, int)) else None,
        }
        sizes = [len(g) for g in spec.graphs]
        arrays: dict[str, Any] = {
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
        for name, (generators, counts) in self.snapshots.items():
            arrays[f"snapshot::{name}"] = generators
            arrays[f"snapshot_counts::{name}"] = counts
        if self.purification_layer is not None:
            arrays["purification_layer"] = self.purification_layer
        text = json.dumps(meta, default=_json_default)
        np.savez_compressed(path, __meta__=np.array(text), **arrays)

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
                    _schedule(p.get("checkpoints")),
                    _cutoff_from_record(p.get("cutoff")),
                )
                for p in meta["probes"]
            ]
            edges, weights = z["edges"], z["edge_weights"] if "edge_weights" in z.files else None
            if meta.get("graph_bank"):
                cuts = z["graph_ptr"][1:-1]
                edges = list(np.split(edges, cuts))
                weights = None if weights is None else list(np.split(weights, cuts))
            layers = z["checkpoint_layers"].astype(np.int64)
            symbolic = meta.get("checkpoints")
            spec = MonitoredCircuit(
                edges=edges,
                gate_set=z["gate_set"],
                system_order=z["system_order"],
                edge_weights=weights,
                checkpoints=layers if symbolic is None else symbolic,
                probes=probes,
                **{
                    k: s[k]
                    for k in _SPEC_FIELDS
                    if k in s and not (k == "gates_per_layer" and s["schedule"] != "random_edges")
                },
            )
            if not np.array_equal(spec.checkpoint_layers, layers):
                raise ValueError(f"{path}: the stored checkpoints do not match the stored layers")
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
                seed_spawn_key=tuple(meta.get("seed_spawn_key", ())),
                seed_pool_size=meta.get("seed_pool_size", 4),
                version=meta.get("version"),
                # files from before contracts were recorded were made under contract 1
                randomness_contract=meta.get("randomness_contract", 1),
                snapshots={
                    p.name: (z[f"snapshot::{p.name}"], z[f"snapshot_counts::{p.name}"])
                    for p in probes
                    if f"snapshot::{p.name}" in z.files
                },
            )

    @staticmethod
    def concatenate(results: Sequence[EnsembleResult]) -> EnsembleResult:
        """Merge chunks of the same protocol and seed (e.g. from separate jobs).

        The chunks must have identical specs, seeds, package versions and
        randomness contracts and cover different trajectories
        (``first_trajectory=``); anything else raises ``ValueError`` rather
        than mixing protocols or counting a trajectory twice.
        """
        if not results:
            raise ValueError("nothing to concatenate")
        first = results[0]
        refuse = "can only concatenate results of the same protocol and seed"
        for r in results[1:]:
            if r.randomness_contract != first.randomness_contract:
                raise ValueError(
                    f"{refuse}: they were made under randomness contracts "
                    f"{first.randomness_contract} and {r.randomness_contract}, under which the "
                    "same seed gives other trajectories; rerun the older part with this version "
                    f"(randomness contract {RANDOMNESS_CONTRACT})"
                )
            if _seed_record(r) != _seed_record(first):
                raise ValueError(
                    f"{refuse}: the seeds differ ({_seed_record(first)} vs {_seed_record(r)})"
                )
            difference = _protocol_difference(first.spec, r.spec)
            if difference is not None:
                raise ValueError(f"{refuse}: {difference} differs")
            if list(r.values) != list(first.values):
                raise ValueError(f"{refuse}: the probes differ")
            if not np.array_equal(r.checkpoint_layers, first.checkpoint_layers):
                raise ValueError("checkpoint layers differ")
            if r.version != first.version:
                raise ValueError(
                    f"{refuse}: they come from different sparsegf2 versions ({first.version} "
                    f"vs {r.version}), whose trajectories need not agree; rerun with one version"
                )
        ids = np.concatenate([np.asarray(r.trajectories) for r in results])
        if np.unique(ids).size != ids.size:
            raise ValueError(
                "the results share trajectories (each would be counted twice); run each chunk "
                "with its own range, e.g. spec.run(n, seed=s, first_trajectory=k)"
            )

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
            seed_spawn_key=first.seed_spawn_key,
            seed_pool_size=first.seed_pool_size,
            version=first.version,
            randomness_contract=first.randomness_contract,
            snapshots={
                name: (
                    np.concatenate([r.snapshots[name][0] for r in results]),
                    np.concatenate([r.snapshots[name][1] for r in results]),
                )
                for name in first.snapshots
            },
        )
