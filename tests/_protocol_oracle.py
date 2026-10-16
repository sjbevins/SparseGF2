"""An independent transcription of the ensemble engine's draw order (the protocol oracle).

``oracle_circuit_text(spec, t, seed)`` rebuilds, in plain Python, the text of the circuit
that trajectory ``t`` of ``spec.run(seed=seed)`` simulates, so a test can require it to
equal ``str(spec.circuit(t, seed=seed))``, and ``probe_orderings`` rebuilds the random
orderings a sampled probe draws at one evaluation. It is written from the documented protocol:
the ``sparsegf2._kernel.ensemble`` module docstring ("Random draws per trajectory, in
order", items 1-4) and the ``MonitoredCircuit`` docstrings (setup, scramble, schedules,
the color cycle that runs on from the scramble layers). It draws from
``_reference_rng`` (a pure-Python xoshiro256**), never imports ``sparsegf2._kernel``,
never calls ``MonitoredCircuit.circuit`` or ``run``, and reads only the spec's public
fields (``graphs``, ``edge_weights``, ``gate_set``, ...).

Where the documents are silent, these conventions were read from the engine and are
transcribed here (the equality test pins each of them):

* Keys: ``SeedSequence(seed).generate_state(4, uint64)``; words 0 and 1 key the circuit
  streams, words 2 and 3 the probe streams. Trajectory ``t`` draws its circuit from
  stream ``t``, derived with ``_reference_rng.stream_state`` under the current contract
  (``RANDOMNESS_CONTRACT_MODEL``) or the one passed as ``contract=``. The draw order is
  the same in every contract, so ``contract=1`` rebuilds contract-1 circuits.
* Probe streams (contract 2): the evaluation of the sampled probe named ``name`` at
  layer ``L`` of trajectory ``t`` draws only from stream ``mix(t, h, L)`` of the probe
  key, where ``h`` is the first 8 bytes of ``blake2b(name.encode("utf-8"),
  digest_size=8)`` read little-endian. Its orderings start from ``0 .. n - 1`` at every
  evaluation, and each is a Fisher-Yates shuffle of the previous one (``r = i +
  below(n - i)`` for ``i = 0 .. n - 2``, then swap entries ``i`` and ``r``).
* Scramble pair: ``a = below(n)``, then ``b = below(n - 1)``, plus one if ``b >= a``.
* Scramble matching: a fresh identity permutation each time, shuffled front to back
  (``r = i + below(n - i)`` for ``i = 0 .. n - 2``, then swap entries ``i`` and ``r``),
  paired as ``(perm[2j], perm[2j + 1])``; an odd ``n`` leaves ``perm[n - 1]`` idle.
* Edge coloring (``"matching"``): greedy in edge order (each edge takes the smallest
  color free at both of its ends); color class ``c`` lists its edges in edge order.
  Layer ``L`` of the count shared by the scramble layers (``0 .. s - 1``) and the
  dynamics (``s ..``) uses class ``L mod C``.
* Weighted edge: ``u = double()``; the edge is the smallest ``e`` with ``u < cum[e]``,
  where ``cum`` is the running sum of ``w / sum(w)`` with its last entry set to 1. The
  total is numpy's (pairwise) sum, as in the engine: Python's ``sum`` adds naively on
  3.11 and with compensation from 3.12, and either can differ from it in the last bit.
* Orientation: a bit of 1 swaps the edge's two qubits (edges are stored ``a < b``).
* Gate: ``gate_set[below(len(gate_set))]``, with ``gate_set`` as the spec resolved it.
* Measurement: qubits ``0 .. n - 1`` in order, each measured when ``double() < p``.
* Text: the setup as ``H`` and ``CX`` lines; one ``CLIFFORD_2Q(g, ...) a b ...`` line per
  non-empty block (the scramble pairs, each scramble matching, each scramble layer, each
  dynamics layer); ``TICK`` after the scramble and after every layer, with that layer's
  ``M`` line (qubits in increasing order) before its ``TICK`` when anything was measured.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np

import _reference_rng as ref

Gate = tuple[int, int, int]  # (first qubit, second qubit, two-qubit Clifford index)


@dataclass
class TrajectoryDraws:
    """Everything one trajectory draws, in the order it is applied."""

    partner: int | None  # the single reference's Bell partner (None without one)
    scramble: list[list[Gate]] = field(default_factory=list)  # pairs, matchings, layers
    gates: list[list[Gate]] = field(default_factory=list)  # one list per dynamics layer
    measured: list[list[int]] = field(default_factory=list)  # one list per dynamics layer


def circuit_key(seed) -> tuple[int, int]:
    """The circuit-stream key of ``run(seed=seed)``: the first two of four SeedSequence words."""
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    words = ss.generate_state(4, dtype=np.uint64)
    return int(words[0]), int(words[1])


def probe_key(seed) -> tuple[int, int]:
    """The probe-stream key of ``run(seed=seed)``: the last two of four SeedSequence words."""
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    words = ss.generate_state(4, dtype=np.uint64)
    return int(words[2]), int(words[3])


def probe_name_hash(name: str) -> int:
    """The 64-bit hash of a probe name: blake2b with an 8-byte digest, read little-endian."""
    return int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "little")


def probe_orderings(
    seed, trajectory: int, name: str, layer: int, n: int, samples: int
) -> list[list[int]]:
    """The ``samples`` orderings of ``0 .. n - 1`` one sampled-probe evaluation draws.

    For ``erasure_first_failure`` they order positions in the spec's ``system_order``
    (the qubits themselves for the default order), for ``erasure_lost_profile`` the
    system qubits.
    """
    index = ref.mix(int(trajectory), probe_name_hash(name), int(layer))
    rng = ref.Xoshiro(ref.stream_state(probe_key(seed), index))
    order = list(range(n))
    out = []
    for _ in range(samples):
        for i in range(n - 1):
            r = i + rng.next_below(n - i)
            order[i], order[r] = order[r], order[i]
        out.append(list(order))
    return out


def color_classes(edges: list[tuple[int, int]], n: int) -> list[list[int]]:
    """Greedy edge coloring in edge order; each color class lists its edge indices in order."""
    used: list[set[int]] = [set() for _ in range(n)]
    classes: list[list[int]] = []
    for e, (a, b) in enumerate(edges):
        c = 0
        while c in used[a] or c in used[b]:
            c += 1
        used[a].add(c)
        used[b].add(c)
        if c == len(classes):  # every smaller color is already in use at a or b
            classes.append([])
        classes[c].append(e)
    return classes


def cumulative(weights) -> list[float]:
    """Running sum of the normalized weights, with the last entry exactly 1."""
    total = float(np.sum(np.asarray(weights, dtype=np.float64)))
    cum: list[float] = []
    running = 0.0  # added in order, as np.cumsum does
    for w in (float(x) for x in weights):
        running += w / total
        cum.append(running)
    cum[-1] = 1.0
    return cum


def oracle_draws(
    spec, trajectory: int, seed, contract: int = ref.RANDOMNESS_CONTRACT_MODEL
) -> TrajectoryDraws:
    """Replay trajectory ``trajectory``'s random draws from the documented protocol."""
    n = int(spec.num_system)
    t = int(trajectory)
    rng = ref.Xoshiro(ref.stream_state(circuit_key(seed), t, contract=contract))
    graph = 0 if spec.num_graphs == 1 else t // int(spec.circuits_per_graph)
    edges = [(int(a), int(b)) for a, b in np.asarray(spec.graphs[graph]).tolist()]
    weights = spec.edge_weights
    if isinstance(weights, tuple):
        weights = weights[graph]
    cum = None if weights is None else cumulative(weights)
    pool = [int(g) for g in np.asarray(spec.gate_set)]
    orient = bool(spec.random_orientation)
    p = float(spec.measurement_rate)
    classes = color_classes(edges, n) if spec.schedule == "matching" else []

    def gate_on(a: int, b: int) -> Gate:
        if orient and rng.next_bit():
            a, b = b, a
        return a, b, pool[rng.next_below(len(pool))]

    def layer(count: int, measure: bool) -> tuple[list[Gate], list[int]]:
        # item 4: per gate [edge] [orientation bit] gate index; then one uniform per qubit
        if spec.schedule == "matching":
            chosen = classes[count % len(classes)]
            num = len(chosen)
        elif spec.schedule == "random_edges":
            num = int(spec.gates_per_layer)
        else:
            num = len(edges)
        gates = []
        for j in range(num):
            if spec.schedule == "matching":
                e = chosen[j]
            elif spec.schedule == "random_edges":
                if cum is None:
                    e = rng.next_below(len(edges))
                else:
                    e = bisect_right(cum, rng.next_double())  # smallest e with u < cum[e]
            else:
                e = j
            gates.append(gate_on(*edges[e]))
        measured = [q for q in range(n) if rng.next_double() < p] if measure else []
        return gates, measured

    # item 1: the single reference's partner, drawn only when it is random
    partner = None
    if spec.reference == "single":
        if spec.reference_partner == "random":
            partner = rng.next_below(n)
        else:
            partner = int(spec.reference_partner)
    draws = TrajectoryDraws(partner)

    # item 2: scramble pairs, per gate a, then b, then the gate index
    pairs = []
    for _ in range(int(spec.scramble_gates)):
        a = rng.next_below(n)
        b = rng.next_below(n - 1)
        if b >= a:
            b += 1
        pairs.append((a, b, pool[rng.next_below(len(pool))]))
    draws.scramble.append(pairs)

    # item 3: scramble matchings, a Fisher-Yates shuffle, then one gate index per pair
    for _ in range(int(spec.scramble_matchings)):
        perm = list(range(n))
        for i in range(n - 1):
            r = i + rng.next_below(n - i)
            perm[i], perm[r] = perm[r], perm[i]
        draws.scramble.append(
            [(perm[2 * j], perm[2 * j + 1], pool[rng.next_below(len(pool))]) for j in range(n // 2)]
        )

    # item 4: scramble layers, then the dynamics, on one layer count
    s = int(spec.scramble_layers)
    for count in range(s):
        draws.scramble.append(layer(count, False)[0])
    for count in range(s, s + int(spec.layers)):
        gates, measured = layer(count, True)
        draws.gates.append(gates)
        draws.measured.append(measured)
    return draws


def _gate_line(gates: list[Gate]) -> list[str]:
    if not gates:
        return []
    indices = ", ".join(str(g) for _, _, g in gates)
    targets = " ".join(f"{a} {b}" for a, b, _ in gates)
    return [f"CLIFFORD_2Q({indices}) {targets}"]


def oracle_circuit_text(
    spec, trajectory: int, seed, contract: int = ref.RANDOMNESS_CONTRACT_MODEL
) -> str:
    """The text of ``spec.circuit(trajectory, seed=seed)``, from :func:`oracle_draws`.

    With ``contract`` other than the package's, it is the circuit the package drew
    under that contract.
    """
    draws = oracle_draws(spec, trajectory, seed, contract)
    n = int(spec.num_system)
    lines = []
    if spec.reference == "purification":
        lines.append("H " + " ".join(str(i) for i in range(n)))
        lines.append("CX " + " ".join(f"{i} {n + i}" for i in range(n)))
    elif spec.reference == "single":
        lines.append(f"H {draws.partner}")
        lines.append(f"CX {draws.partner} {n}")
    for block in draws.scramble:
        lines += _gate_line(block)
    lines.append("TICK")
    for gates, measured in zip(draws.gates, draws.measured, strict=True):
        lines += _gate_line(gates)
        if measured:
            lines.append("M " + " ".join(str(q) for q in measured))
        lines.append("TICK")
    return "\n".join(lines)
