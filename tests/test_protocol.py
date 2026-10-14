"""The ensemble protocol, checked from outside the engine.

``_protocol_oracle`` transcribes the documented draw order in plain Python, so a
protocol bug in the shared generator (``gen_layer`` builds both the run and the
exported circuit) no longer cancels out: every exported circuit must equal the
oracle's text. The laws of every draw are then checked on the engine's circuits
at fixed seeds: chi-square p-values above 1e-6, and normal bands of 4.5 sigma
with sigma computed in the test.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from _protocol_oracle import colour_classes, oracle_circuit_text
from sparsegf2 import graphs, mipt

SEED = 2024

ORACLE_SPECS = {
    "matching_cycle_none": dict(
        num_system=6, edges=graphs.cycle(6), layers=6, measurement_rate=0.3
    ),
    "matching_scramble_layers_purification": dict(
        num_system=9,
        edges=graphs.lattice_2d(3, 3),  # 5 colours, so 3 scramble layers shift the cycle
        layers=7,
        measurement_rate=0.2,
        reference="purification",
        scramble_layers=3,
    ),
    "random_edges_default_count_single_random": dict(
        num_system=10,
        edges=graphs.watts_strogatz(10, 4, 0.3, seed=1),
        layers=6,
        measurement_rate=0.2,
        schedule="random_edges",
        reference="single",
    ),
    "random_edges_odd_count_single_last_qubit": dict(
        num_system=9,
        edges=graphs.cycle(9),
        layers=5,
        measurement_rate=0.25,
        schedule="random_edges",
        gates_per_layer=7,
        reference="single",
        reference_partner=8,
    ),
    "random_edges_no_gates": dict(
        num_system=5,
        edges=graphs.path(5),
        layers=4,
        measurement_rate=0.5,
        schedule="random_edges",
        gates_per_layer=0,
        reference="purification",
    ),
    "weighted_power_law_purification": dict(
        num_system=8,
        edges=graphs.complete(8),
        layers=5,
        measurement_rate=0.15,
        schedule="random_edges",
        edge_weights=graphs.power_law_weights(graphs.complete(8), 8, 2.0),
        reference="purification",
    ),
    "weighted_integers_with_zero_weights": dict(
        num_system=6,
        edges=graphs.complete(6),
        layers=6,
        measurement_rate=0.1,
        schedule="random_edges",
        gates_per_layer=9,
        edge_weights=[0, 1, 2, 0, 3, 1, 1, 0, 5, 2, 1, 1, 4, 0, 2],
    ),
    "all_edges_orientation_gate_set_single_first_qubit": dict(
        num_system=6,
        edges=graphs.complete(6),
        layers=3,
        measurement_rate=0.3,
        schedule="all_edges",
        reference="single",
        reference_partner=0,
        gate_set=[0, 5, 77, 1000, 9999],
        random_orientation=True,
    ),
    "scramble_pairs_and_matchings_even": dict(
        num_system=8,
        edges=graphs.cycle(8),
        layers=3,
        measurement_rate=0.2,
        reference="purification",
        scramble_gates=9,
        scramble_matchings=2,
    ),
    "scramble_matchings_odd_idle_qubit": dict(
        num_system=7, edges=graphs.cycle(7), layers=3, measurement_rate=0.4, scramble_matchings=3
    ),
    "every_scramble_matching_schedule_single_random": dict(
        num_system=10,
        edges=graphs.watts_strogatz(10, 4, 0.2, seed=2),
        layers=4,
        measurement_rate=0.3,
        reference="single",
        scramble_gates=4,
        scramble_matchings=1,
        scramble_layers=5,
        gate_set=[3, 11519],
        random_orientation=True,
    ),
    "ragged_bank_matching": dict(
        num_system=11,
        edges=[graphs.newman_watts(11, 2, 0.3, seed=i) for i in range(4)],
        circuits_per_graph=2,
        layers=4,
        measurement_rate=0.2,
        reference="purification",
        scramble_layers=1,
    ),
    "weighted_bank_single_random": dict(
        num_system=8,
        edges=[graphs.complete(8)] * 3,
        circuits_per_graph=3,
        edge_weights=[graphs.power_law_weights(graphs.complete(8), 8, a) for a in (0.5, 1.5, 3.0)],
        layers=4,
        measurement_rate=0.2,
        schedule="random_edges",
        gates_per_layer=5,
        reference="single",
    ),
    "all_edges_bank_orientation": dict(
        num_system=6,
        edges=[graphs.cycle(6), graphs.complete(6)],
        circuits_per_graph=4,
        layers=2,
        measurement_rate=0.3,
        schedule="all_edges",
        scramble_gates=2,
        random_orientation=True,
    ),
    "every_qubit_measured": dict(
        num_system=4, edges=graphs.cycle(4), layers=3, measurement_rate=1.0, scramble_gates=3
    ),
    "no_layers_single_random": dict(
        num_system=5,
        edges=graphs.cycle(5),
        layers=0,
        measurement_rate=0.5,
        reference="single",
        scramble_gates=3,
        scramble_matchings=1,
    ),
    "two_qubits": dict(
        num_system=2,
        edges=graphs.path(2),
        layers=3,
        measurement_rate=0.5,
        schedule="random_edges",
        gates_per_layer=3,
        reference="single",
        random_orientation=True,
    ),
    "word_boundary_65_single_random": dict(
        num_system=65,
        edges=graphs.cycle(65),
        layers=2,
        measurement_rate=0.1,
        schedule="random_edges",
        reference="single",
        scramble_matchings=1,
    ),
}


@pytest.mark.parametrize("trajectory", [0, 1, 7])
@pytest.mark.parametrize("name", list(ORACLE_SPECS))
def test_exported_circuit_equals_the_oracle(name, trajectory):
    spec = mipt.MonitoredCircuit(**ORACLE_SPECS[name])
    assert str(spec.circuit(trajectory, seed=SEED)) == oracle_circuit_text(spec, trajectory, SEED)


@pytest.mark.parametrize(
    "seed", [0, [3, 1, 4], np.random.SeedSequence(5).spawn(2)[1]], ids=["0", "list", "spawned"]
)
@pytest.mark.parametrize("trajectory", [2, 10**9 + 7])
@pytest.mark.parametrize(
    "name",
    [
        "every_scramble_matching_schedule_single_random",
        "weighted_power_law_purification",
        "word_boundary_65_single_random",
    ],
)
def test_oracle_agrees_for_other_seeds_and_far_trajectories(name, trajectory, seed):
    # first_trajectory-style indices: a trajectory's circuit depends only on (spec, seed, t)
    spec = mipt.MonitoredCircuit(**ORACLE_SPECS[name])
    assert str(spec.circuit(trajectory, seed=seed)) == oracle_circuit_text(spec, trajectory, seed)


# ---------------------------------------------------------------------------
# the laws of the draws, on the engine's exported circuits
# ---------------------------------------------------------------------------


def _segments(circuit) -> list[dict]:
    """The circuit split at its TICKs: per segment, the H targets, the CLIFFORD_2Q blocks
    (lists of (a, b, gate index)) and the measured qubits. Segment 0 is the setup and
    scramble; segment l is dynamics layer l."""
    segments: list[dict] = [dict(h=[], blocks=[], measured=[])]
    for ins in circuit:
        qubits = [t.value for t in ins.targets]
        if ins.name == "TICK":
            segments.append(dict(h=[], blocks=[], measured=[]))
        elif ins.name == "H":
            segments[-1]["h"].extend(qubits)
        elif ins.name == "CLIFFORD_2Q":
            pairs = zip(qubits[0::2], qubits[1::2], (int(g) for g in ins.args), strict=True)
            segments[-1]["blocks"].append(list(pairs))
        elif ins.name == "M":
            segments[-1]["measured"].extend(qubits)
        else:
            assert ins.name == "CX", ins.name
    assert segments.pop() == dict(h=[], blocks=[], measured=[])  # nothing after the last TICK
    return segments


def _layer_gates(segment: dict) -> list[tuple[int, int, int]]:
    """The gates of one dynamics layer (at most one CLIFFORD_2Q line)."""
    assert len(segment["blocks"]) <= 1
    return segment["blocks"][0] if segment["blocks"] else []


def _chi2_sf(x: float, dof: int) -> float:
    """P(chi-square with ``dof`` degrees of freedom > x): Q(dof / 2, x / 2) in closed form."""
    h = x / 2
    if dof % 2 == 0:  # e^-h sum_{i < dof/2} h^i / i!
        term = total = math.exp(-h)
        for i in range(1, dof // 2):
            term *= h / i
            total += term
        return total
    # erfc(sqrt h) + e^-h sum_{i < (dof-1)/2} h^(i + 1/2) / Gamma(i + 3/2)
    total = math.erfc(math.sqrt(h))
    term = math.sqrt(h) * math.exp(-h) / math.gamma(1.5)
    for i in range(dof // 2):
        total += term
        term *= h / (i + 1.5)
    return total


def _chi2_p(counts, expected) -> float:
    """Pearson's chi-square p-value of ``counts`` against ``expected`` (same total)."""
    counts = np.asarray(counts, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert counts.shape == expected.shape and abs(counts.sum() - expected.sum()) < 1e-6
    return _chi2_sf(float(np.sum((counts - expected) ** 2 / expected)), counts.size - 1)


def _uniform_p(counts) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    return _chi2_p(counts, np.full(counts.shape, counts.sum() / counts.size))


def test_chi_square_tail_matches_known_values():
    # 5% critical values (tables) and the exact chi-square(2) tail e^(-x/2)
    for x, dof in [(3.841459, 1), (5.991465, 2), (7.814728, 3), (82.528727, 63)]:
        assert _chi2_sf(x, dof) == pytest.approx(0.05, rel=1e-5)
    assert _chi2_sf(10.0, 2) == pytest.approx(math.exp(-5.0), rel=1e-12)
    assert _chi2_sf(0.0, 7) == pytest.approx(1.0)


def test_every_qubit_is_measured_at_the_rate():
    n, p, layers, trajectories = 12, 0.3, 400, 5
    spec = mipt.MonitoredCircuit(
        num_system=n, edges=graphs.cycle(n), layers=layers, measurement_rate=p
    )
    counts = np.zeros(n)
    for t in range(trajectories):
        segments = _segments(spec.circuit(t, seed=SEED))
        assert len(segments) == layers + 1 and segments[0]["measured"] == []
        for segment in segments[1:]:
            measured = segment["measured"]
            assert measured == sorted(set(measured)) and set(measured) <= set(range(n))
            counts[measured] += 1
    trials = layers * trajectories
    # per qubit: binomial(2000, 0.3), mean 600, sigma = sqrt(2000 * 0.3 * 0.7) = 20.5
    sigma = math.sqrt(trials * p * (1 - p))
    for q in range(n):  # qubit n - 1 included
        assert abs(counts[q] - trials * p) <= 4.5 * sigma, (q, counts[q])
    # total: binomial(24000, 0.3), mean 7200, sigma = sqrt(24000 * 0.21) = 71.0
    total_sigma = math.sqrt(n * trials * p * (1 - p))
    assert abs(counts.sum() - n * trials * p) <= 4.5 * total_sigma


def test_random_edges_places_exactly_gates_per_layer_on_graph_edges():
    edges = graphs.watts_strogatz(10, 4, 0.3, seed=1)
    spec = mipt.MonitoredCircuit(
        num_system=10,
        edges=edges,
        layers=50,
        measurement_rate=0.2,
        schedule="random_edges",
        gates_per_layer=7,
        scramble_layers=2,
    )
    allowed = set(map(tuple, edges.tolist()))
    for t in range(3):
        segments = _segments(spec.circuit(t, seed=SEED))
        layers = segments[0]["blocks"] + [_layer_gates(s) for s in segments[1:]]
        assert len(layers) == 2 + 50  # scramble layers, then dynamics
        for gates in layers:
            assert len(gates) == 7
            assert all((min(a, b), max(a, b)) in allowed for a, b, _ in gates)


def test_all_edges_places_every_edge_once_in_order():
    edges = graphs.watts_strogatz(12, 4, 0.3, seed=2)
    spec = mipt.MonitoredCircuit(
        num_system=12, edges=edges, layers=20, measurement_rate=0.2, schedule="all_edges"
    )
    for t in range(3):
        for segment in _segments(spec.circuit(t, seed=SEED))[1:]:
            assert [(a, b) for a, b, _ in _layer_gates(segment)] == list(map(tuple, edges.tolist()))


def test_matching_places_one_colour_class_per_layer():
    n, scramble = 12, 3
    edges = graphs.watts_strogatz(n, 4, 0.3, seed=2)
    edge_list = list(map(tuple, edges.tolist()))
    classes = colour_classes(edge_list, n)
    num_colours = len(classes)
    assert num_colours == 6  # the cycle does not divide the scramble layers
    spec = mipt.MonitoredCircuit(
        num_system=n, edges=edges, layers=20, measurement_rate=0.2, scramble_layers=scramble
    )
    segments = _segments(spec.circuit(0, seed=SEED))
    layers = segments[0]["blocks"] + [_layer_gates(s) for s in segments[1:]]
    assert len(layers) == scramble + 20
    sets = [frozenset((min(a, b), max(a, b)) for a, b, _ in gates) for gates in layers]
    for count, gates in enumerate(layers):
        qubits = [q for a, b, _ in gates for q in (a, b)]
        assert len(qubits) == len(set(qubits))  # each layer is a matching
        assert sets[count] == {edge_list[e] for e in classes[count % num_colours]}
        assert len(sets[count]) == len(gates)
    # one period of layers covers every edge exactly once, and the cycle repeats
    assert sum(len(s) for s in sets[:num_colours]) == len(edge_list)
    assert frozenset().union(*sets[:num_colours]) == set(edge_list)
    assert all(sets[c] == sets[c + num_colours] for c in range(len(sets) - num_colours))


def test_gate_indices_and_unweighted_edges_are_uniform():
    edges = graphs.watts_strogatz(40, 4, 0.2, seed=1)
    spec = mipt.MonitoredCircuit(
        num_system=40,
        edges=edges,
        layers=400,
        measurement_rate=0.0,
        schedule="random_edges",
        gates_per_layer=50,
    )
    gates = [g for s in _segments(spec.circuit(0, seed=SEED))[1:] for g in _layer_gates(s)]
    assert len(gates) == 20_000
    index = np.array([g for _, _, g in gates])
    assert index.min() >= 0 and index.max() < 11520
    # 11520 = 64 * 180: both bucketings are uniform over 64 buckets (dof 63)
    assert _uniform_p(np.bincount(index % 64, minlength=64)) > 1e-6
    assert _uniform_p(np.bincount(index // 180, minlength=64)) > 1e-6
    edge_id = {e: i for i, e in enumerate(map(tuple, edges.tolist()))}
    picked = [edge_id[(min(a, b), max(a, b))] for a, b, _ in gates]
    assert _uniform_p(np.bincount(picked, minlength=len(edges))) > 1e-6  # 80 edges, dof 79


def test_weighted_edges_follow_the_weights():
    edges = graphs.cycle(20)
    weights = np.arange(20.0)  # edge 0 has weight 0 and must never be drawn
    spec = mipt.MonitoredCircuit(
        num_system=20,
        edges=edges,
        layers=400,
        measurement_rate=0.0,
        schedule="random_edges",
        gates_per_layer=50,
        edge_weights=weights,
    )
    gates = [g for s in _segments(spec.circuit(0, seed=SEED))[1:] for g in _layer_gates(s)]
    edge_id = {e: i for i, e in enumerate(map(tuple, edges.tolist()))}
    counts = np.bincount([edge_id[(min(a, b), max(a, b))] for a, b, _ in gates], minlength=20)
    assert counts[0] == 0 and counts.sum() == 20_000
    # expected 20000 * w / 190: at least 105 per edge of positive weight (dof 18)
    assert _chi2_p(counts[1:], 20_000 * weights[1:] / weights.sum()) > 1e-6


def test_partner_scramble_pairs_and_matchings_are_uniform():
    n, scramble_gates, scramble_matchings, trajectories = 8, 20, 4, 2000
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        layers=0,
        measurement_rate=0.0,
        reference="single",
        scramble_gates=scramble_gates,
        scramble_matchings=scramble_matchings,
    )
    partners: list[int] = []
    pairs: Counter = Counter()
    partner_of_0: list[int] = []
    for t in range(trajectories):
        (segment,) = _segments(spec.circuit(t, seed=SEED))
        partners.extend(segment["h"])
        blocks = segment["blocks"]
        assert len(blocks) == 1 + scramble_matchings
        pairs.update((a, b) for a, b, _ in blocks[0])
        for matching in blocks[1:]:
            qubits = [q for a, b, _ in matching for q in (a, b)]
            assert sorted(qubits) == list(range(n))  # a perfect matching (n is even)
            partner_of_0.extend(b if a == 0 else a for a, b, _ in matching if 0 in (a, b))
    assert len(partners) == trajectories
    assert _uniform_p(np.bincount(partners, minlength=n)) > 1e-6  # 250 per qubit, dof 7
    assert all(a != b for a, b in pairs)
    ordered = [pairs[(a, b)] for a in range(n) for b in range(n) if a != b]
    assert sum(ordered) == trajectories * scramble_gates
    assert _uniform_p(ordered) > 1e-6  # 56 ordered pairs, 714 each, dof 55
    assert len(partner_of_0) == trajectories * scramble_matchings
    assert _uniform_p(np.bincount(partner_of_0, minlength=n)[1:]) > 1e-6  # 1143 each, dof 6


def test_orientation_flips_half_the_gates_and_gate_set_is_uniform():
    n, layers, trajectories = 16, 200, 2
    gate_set = [0, 5, 77, 1000, 9999]
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        layers=layers,
        measurement_rate=0.0,
        schedule="all_edges",
        gate_set=gate_set,
        random_orientation=True,
    )
    gates = [
        g
        for t in range(trajectories)
        for s in _segments(spec.circuit(t, seed=SEED))[1:]
        for g in _layer_gates(s)
    ]
    draws = n * layers * trajectories
    assert len(gates) == draws
    # edges are stored a < b, so a gate with a > b was flipped. Binomial(6400, 1/2):
    # sigma = sqrt(6400 / 4) = 40 gates
    flipped = sum(a > b for a, b, _ in gates)
    assert abs(flipped - draws / 2) <= 4.5 * math.sqrt(draws / 4)
    counts = Counter(g for _, _, g in gates)
    assert set(counts) == set(gate_set)
    assert _uniform_p([counts[g] for g in gate_set]) > 1e-6  # 1280 each, dof 4
