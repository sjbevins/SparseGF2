"""The monitored-circuit ensemble engine.

The central test runs each trajectory's exported circuit through the general
``Simulator`` (itself tested against Stim) and requires the engine's recorded
values to match exactly at every checkpoint.
"""

import dataclasses
import itertools
import json
import re

import numpy as np
import pytest

from sparsegf2 import Simulator, StabilizerCode, analysis, gate, graphs, mipt
from sparsegf2 import observables as ob

RANDOM_KINDS = ("erasure_first_failure", "erasure_lost_profile")  # checked separately


def _replay(spec: mipt.MonitoredCircuit, trajectory: int, seed):
    """Purity at every tick and each probe's value at its own layers, from running the
    exported circuit."""
    circuit = spec.circuit(trajectory, seed=seed)
    refs = spec.reference_qubits
    order = spec.system_order.tolist()
    probe_layers = {p.name: set(spec.probe_layers(p.name).tolist()) for p in spec.probes}

    def probe(sim, p):
        if p.kind in ("entropy", "half_cut"):
            return sim.entropy(p.regions[0])
        if p.kind == "mutual_information":
            return ob.mutual_information(sim, *p.regions)
        if p.kind == "tripartite":
            return ob.tripartite_information(sim, *p.regions)
        if p.kind == "code_dimension":
            return sim.entropy(refs)
        if p.kind == "contiguous_distance":
            # the engine records NaN, not the observable's sentinel 0, without a code
            if sim.entropy(refs) == 0:
                return np.nan
            return ob.contiguous_code_distance(sim, order, refs, periodic=spec.periodic)
        if p.kind == "lost_information":
            return ob.lost_logical_information(sim, p.regions[0], refs)
        if p.kind == "entropy_profile":
            return ob.entropy_profile(sim, order)
        if p.kind == "uncorrectable_counts":
            counts = np.zeros(p.size)
            for m in range(1, p.size + 1):
                for region in itertools.combinations(spec.system_qubits, m):
                    counts[m - 1] += ob.lost_logical_information(sim, region, refs) > 0
            return counts
        if p.kind == "code_snapshot":  # the recorded value: n - k generators
            return spec.num_system - (sim.entropy(refs) if refs else 0)
        raise AssertionError(p.kind)

    def observe(sim, tick):
        values = {
            p.name: probe(sim, p)
            for p in spec.probes
            if p.kind not in RANDOM_KINDS and tick in probe_layers[p.name]
        }
        for p in spec.probes:  # the code itself, as a canonical basis of its stabilizer group
            if p.kind == "code_snapshot" and refs and tick in probe_layers[p.name]:
                code = StabilizerCode.from_simulator(sim, reference=refs)
                values[(p.name, "code")] = _row_space(code.stabilizer_matrix())
        pure = sim.entropy(refs) == 0 if refs else False
        return values, pure

    sim = Simulator(spec.num_qubits, phases=False, seed=0)
    observations = dict(sim.run(circuit, observe=observe).observations)
    return observations


def _row_space(rows: np.ndarray) -> np.ndarray:
    """The reduced row echelon form of ``rows`` over GF(2) without its zero rows, which is
    the same matrix for any two generating sets of one group."""
    m = (np.asarray(rows) % 2).astype(np.uint8)
    pivot = 0
    for col in range(m.shape[1]):
        if pivot == m.shape[0]:
            break
        nonzero = np.flatnonzero(m[pivot:, col])
        if not nonzero.size:
            continue
        row = pivot + int(nonzero[0])
        m[[pivot, row]] = m[[row, pivot]]
        others = np.flatnonzero(m[:, col])
        m[others[others != pivot]] ^= m[pivot]
        pivot += 1
    return m[:pivot]


def _code_probes(region: list[int], layers: int) -> list[mipt.Probe]:
    """The code probes of the specs with references above 64 qubits."""
    return [
        mipt.code_dimension(),
        mipt.lost_information(region),
        mipt.uncorrectable_region_counts(2, checkpoints=[0, layers // 2, layers]),
        mipt.contiguous_distance(),
        mipt.code_snapshot(checkpoints=[0, layers]),
    ]


SPECS = {
    "brickwork_cycle_purification": dict(
        num_system=12,
        edges=graphs.cycle(12),
        layers=20,
        measurement_rate=0.1,
        reference="purification",
        checkpoints="every_layer",
        probes=[
            mipt.code_dimension(),
            mipt.half_cut_entropy(),
            mipt.contiguous_distance(),
            mipt.lost_information([0, 1, 5]),
            mipt.lost_information(list(range(7))),
            mipt.entropy([2, 3, 13]),
            mipt.entropy_profile(),
            mipt.mutual_information([0, 1, 2], [6, 7, 8]),
            mipt.tripartite_information(),
            mipt.uncorrectable_region_counts(2, checkpoints=[0, 5, 20]),
            mipt.code_snapshot(checkpoints=[10, 20]),
        ],
    ),
    "random_edges_ws_single_random_partner": dict(
        num_system=20,
        edges=graphs.watts_strogatz(20, 4, 0.2, seed=3),
        layers=40,
        measurement_rate=0.12,
        schedule="random_edges",
        reference="single",
        scramble_gates=30,
        checkpoints=3,
        probes=[
            mipt.code_dimension(),
            mipt.half_cut_entropy(),
            mipt.tripartite_information([0, 1], [5, 6], [10, 11, 12]),
        ],
    ),
    "long_range_power_law": dict(
        num_system=16,
        edges=graphs.complete(16),
        layers=10,
        measurement_rate=0.15,
        schedule="random_edges",
        edge_weights=graphs.power_law_weights(graphs.complete(16), 16, 2.0),
        reference="purification",
        checkpoints="every_layer",
        probes=[mipt.code_dimension(), mipt.entropy_profile()],
    ),
    "all_edges_complete_with_stop": dict(
        num_system=9,
        edges=graphs.complete(9),
        layers=30,
        measurement_rate=0.3,
        schedule="all_edges",
        reference="single",
        reference_partner=4,
        checkpoints="every_layer",
        probes=[
            mipt.code_dimension(),
            mipt.contiguous_distance(),
            mipt.lost_information([0, 4]),
            mipt.uncorrectable_region_counts(2, checkpoints=2),
            mipt.code_snapshot(checkpoints=3),
            mipt.half_cut_entropy(checkpoints=[0, 7, 30]),
        ],
        stop_when_purified=True,
    ),
    "lattice_scramble_layers_gate_set_orientation": dict(
        num_system=16,
        edges=graphs.lattice_2d(4, 4),
        layers=15,
        measurement_rate=0.2,
        reference="purification",
        scramble_layers=3,
        gate_set=[0, 5, 77, 1000, 9999],
        random_orientation=True,
        checkpoints=[0, 1, 7, 15],
        probes=[mipt.code_dimension(), mipt.contiguous_distance(), mipt.entropy_profile()],
        system_order=list(range(15, -1, -1)),
        periodic=False,
    ),
    "single_reference_ws_collection_matching_scramble": dict(
        num_system=16,
        edges=[graphs.watts_strogatz(16, 4, 0.2, seed=(5, i)) for i in range(4)],
        circuits_per_graph=2,
        layers=30,
        measurement_rate=0.15,
        schedule="random_edges",
        reference="single",
        reference_partner=15,
        scramble_matchings=4,
        checkpoints="every_layer",
        probes=[mipt.code_dimension()],
        stop_when_purified=True,
    ),
    "ragged_bank_matching_schedule_all_scrambles": dict(
        num_system=11,
        edges=[graphs.newman_watts(11, 2, 0.3, seed=i) for i in range(3)],
        circuits_per_graph=2,
        layers=12,
        measurement_rate=0.2,
        reference="purification",
        scramble_gates=5,
        scramble_matchings=2,
        scramble_layers=2,
        checkpoints=[0, 3, 12],
        probes=[
            mipt.code_dimension(),
            mipt.half_cut_entropy(checkpoints=5),
            mipt.contiguous_distance(checkpoints="every_layer"),
        ],
    ),
    "weighted_bank": dict(
        num_system=10,
        edges=[graphs.complete(10)] * 2,
        circuits_per_graph=3,
        edge_weights=[
            graphs.power_law_weights(graphs.complete(10), 10, 1.0),
            graphs.power_law_weights(graphs.complete(10), 10, 3.0),
        ],
        layers=8,
        measurement_rate=0.1,
        schedule="random_edges",
        reference="purification",
        checkpoints="every_layer",
        probes=[mipt.code_dimension(), mipt.entropy_profile()],
    ),
    "no_reference_word_boundary": dict(
        num_system=65,
        edges=graphs.cycle(65),
        layers=12,
        measurement_rate=0.25,
        schedule="random_edges",
        gates_per_layer=40,
        checkpoints="every_layer",
        probes=[mipt.half_cut_entropy(), mipt.entropy([0, 64])],
    ),
    # References above 64 qubits and at the 64-bit word boundaries. The lost_information
    # region crosses qubit 64 wherever the system has one (n > 64); for n <= 64 the
    # references (qubits n .. 2n - 1) cross it instead. At rate 0.03 some single-reference
    # trajectories purify (k = 0) and some keep their code.
    "purification_n40_80_qubits": dict(
        num_system=40,
        edges=graphs.watts_strogatz(40, 4, 0.2, seed=40),
        layers=8,
        measurement_rate=0.2,
        reference="purification",
        checkpoints="every_layer",
        probes=_code_probes([0, 20, 39], 8),
    ),
    "single_n70_partner_64": dict(
        num_system=70,
        edges=graphs.watts_strogatz(70, 4, 0.2, seed=70),
        layers=8,
        measurement_rate=0.03,
        schedule="random_edges",
        reference="single",
        reference_partner=64,
        checkpoints="every_layer",
        probes=_code_probes(list(range(60, 68)), 8),
    ),
    "purification_n63": dict(
        num_system=63,
        edges=graphs.watts_strogatz(63, 4, 0.2, seed=63),
        layers=6,
        measurement_rate=0.2,
        schedule="random_edges",
        reference="purification",
        checkpoints="every_layer",
        probes=_code_probes([0, 31, 62], 6),
    ),
    "purification_n64": dict(
        num_system=64,
        edges=graphs.cycle(64),
        layers=6,
        measurement_rate=0.2,
        reference="purification",
        checkpoints="every_layer",
        probes=_code_probes([0, 62, 63], 6),
    ),
    "purification_n65": dict(
        num_system=65,
        edges=graphs.cycle(65),
        layers=6,
        measurement_rate=0.2,
        schedule="all_edges",
        reference="purification",
        checkpoints="every_layer",
        probes=_code_probes([0, 63, 64], 6),
    ),
    "single_n127_partner_63": dict(
        num_system=127,
        edges=graphs.cycle(127),
        layers=4,
        measurement_rate=0.03,
        reference="single",
        reference_partner=63,
        checkpoints="every_layer",
        probes=_code_probes([0, 63, 64, 126], 4),
    ),
    "single_n128_partner_64": dict(
        num_system=128,
        edges=graphs.cycle(128),
        layers=4,
        measurement_rate=0.03,
        schedule="random_edges",
        reference="single",
        reference_partner=64,
        checkpoints="every_layer",
        probes=_code_probes([0, 63, 64, 127], 4),
    ),
    "single_n129_random_partner": dict(
        num_system=129,
        edges=graphs.watts_strogatz(129, 4, 0.2, seed=129),
        layers=4,
        measurement_rate=0.03,
        schedule="random_edges",
        reference="single",
        scramble_gates=16,
        checkpoints="every_layer",
        probes=_code_probes([0, 63, 64, 127, 128], 4),
    ),
}

# Specs replayed with 3 trajectories instead of 6: their replays count every 1- and 2-qubit
# erasure of up to 129 qubits in Python.
FEW_REPLAYS = {
    "purification_n40_80_qubits",
    "single_n70_partner_64",
    "purification_n63",
    "purification_n64",
    "purification_n65",
    "single_n127_partner_63",
    "single_n128_partner_64",
    "single_n129_random_partner",
}


@pytest.mark.parametrize("name", list(SPECS))
def test_engine_matches_exported_circuits_exactly(name):
    """Every deterministic probe, at its own layers, equals the replayed circuit.

    After a trajectory stops (stop_when_purified) the engine records the
    probe's exact value for a pure reference: the replay, which keeps
    simulating, must agree with it; entanglement probes are NaN there. A
    saved code (with a reference) must span the same stabilizer group as the
    replayed state's code.
    """
    spec = mipt.MonitoredCircuit(**SPECS[name])
    seed = 2024
    trajectories = 3 if name in FEW_REPLAYS else 6
    result = spec.run(trajectories, seed=seed)
    for t in range(trajectories):
        obs = _replay(spec, t, seed)
        tau_expected = -1
        for layer in spec.checkpoint_layers:
            _, pure = obs[int(layer)]
            if pure and spec.reference != "none":
                tau_expected = int(layer)
                break
        if spec.reference != "none":
            assert result.purification_layer[t] == tau_expected
        stop = tau_expected if (spec.stop_when_purified and tau_expected >= 0) else None
        for p in spec.probes:
            if p.kind in RANDOM_KINDS:
                continue
            for c, layer in enumerate(result.layers(p.name)):
                got = result.values[p.name][t, c]
                expected = obs[int(layer)][0][p.name]
                if (
                    stop is not None
                    and layer > stop
                    and np.isnan(mipt._probe_fill(p, spec.num_system))
                ):
                    assert np.all(np.isnan(got)), f"{p.name} t={t} layer={layer}"
                    continue
                np.testing.assert_allclose(got, expected, err_msg=f"{p.name} t={t} layer={layer}")
                if p.kind == "code_snapshot" and spec.reference != "none":
                    if result.snapshots[p.name][1][t, c] < 0:  # stopped: no code was saved
                        continue
                    np.testing.assert_array_equal(
                        _row_space(result.code(t, c, p.name).stabilizer_matrix()),
                        obs[int(layer)][0][(p.name, "code")],
                        err_msg=f"{p.name} code t={t} layer={layer}",
                    )


def test_results_independent_of_threads_and_chunking():
    spec = mipt.MonitoredCircuit(**SPECS["random_edges_ws_single_random_partner"])
    full = spec.run(40, seed=5, num_threads=4)
    one = spec.run(40, seed=5, num_threads=1)
    chunks = mipt.EnsembleResult.concatenate(
        [spec.run(15, seed=5), spec.run(25, seed=5, first_trajectory=15)]
    )
    for name in full.values:
        np.testing.assert_array_equal(full.values[name], one.values[name])
        np.testing.assert_array_equal(full.values[name], chunks.values[name])
    np.testing.assert_array_equal(full.purification_layer, chunks.purification_layer)
    np.testing.assert_array_equal(full.trajectories, np.arange(40))
    other = spec.run(40, seed=6)
    assert not np.array_equal(full.values["half_cut_entropy"], other.values["half_cut_entropy"])


def test_measurement_draws_give_common_random_numbers_across_rates():
    # Circuit structure draws are independent of p, so lowering p only removes
    # measurements: the measured set at p=0.1 is a subset of that at p=0.3.
    base = dict(num_system=10, edges=graphs.cycle(10), layers=5, reference="none")
    lo = mipt.MonitoredCircuit(**base, measurement_rate=0.1).circuit(0, seed=1)
    hi = mipt.MonitoredCircuit(**base, measurement_rate=0.3).circuit(0, seed=1)

    def measured(c):
        out, layer = [], []
        for ins in c:
            if ins.name == "M":
                layer = sorted(t.value for t in ins.targets)
            if ins.name == "TICK":
                out.append(set(layer))
                layer = []
        return out

    for a, b in zip(measured(lo), measured(hi), strict=True):
        assert a <= b


def test_erasure_profile_probes():
    n = 12
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        layers=8,
        measurement_rate=0.1,
        reference="purification",
        checkpoints=[0, 8],
        probes=[
            mipt.code_dimension(),
            mipt.erasure_lost_profile(samples=6),
            mipt.erasure_first_failure(samples=40),
        ],
    )
    r = spec.run(10, seed=4)
    lost = r.values["erasure_lost_profile"]
    first = r.values["erasure_first_failure"]
    k = r.values["code_dimension"]
    assert lost.shape == (10, 2, n + 1) and first.shape == (10, 2, 40)
    np.testing.assert_array_equal(lost[:, :, 0], 0)
    np.testing.assert_allclose(lost[:, :, n], 2 * k)
    assert np.all(np.diff(lost, axis=2) >= -1e-12)  # mean of monotone curves is monotone
    assert np.all(lost[:, 0, 1:] == 2 * np.arange(1, n + 1))  # fresh Bell pairs: 2 bits per qubit
    np.testing.assert_array_equal(first[:, 0], 1)  # every single qubit carries logical info
    assert np.all((first >= 1) & (first <= n))  # k > 0: some prefix always fails


def test_first_failure_agrees_with_the_saved_code():
    """The engine's first-failure samples follow the same law as the saved code's."""
    n = 16
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        layers=16,
        measurement_rate=0.08,
        reference="purification",
        checkpoints=[16],
        probes=[
            mipt.code_dimension(),
            mipt.erasure_first_failure(samples=4000),
            mipt.uncorrectable_region_counts(3),
            mipt.code_snapshot(),
        ],
    )
    r = spec.run(3, seed=11)
    for t in range(3):
        code = r.code(t, 0)
        assert code.num_logical == r.values["code_dimension"][t, 0]
        np.testing.assert_array_equal(
            code.uncorrectable_counts(3), r.values["uncorrectable_region_counts[3]"][t, 0]
        )
        engine = analysis.failure_by_size(r.values["erasure_first_failure"][t, 0], n)
        offline = analysis.failure_by_size(code.first_failure_sizes(4000, seed=t), n)
        assert np.max(np.abs(engine - offline)) < 0.05  # 4000 samples each: ~0.01 noise
        exact = analysis.exact_failure_by_size(code.uncorrectable_counts(3), n)
        assert np.max(np.abs(engine[1:4] - exact)) < 0.05


def test_long_range_weights_follow_power_law():
    n = 32
    edges = graphs.complete(n)
    w = graphs.power_law_weights(edges, n, 2.0)
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=edges,
        layers=400,
        measurement_rate=0.0,
        schedule="random_edges",
        edge_weights=w,
    )
    c = spec.circuit(0, seed=1)
    dists = []
    for ins in c:
        if ins.name == "CLIFFORD_2Q":
            for a, b in zip(ins.targets[::2], ins.targets[1::2], strict=True):
                d = abs(a.value - b.value)
                dists.append(min(d, n - d))
    dists = np.array(dists)
    expected = np.zeros(n // 2 + 1)
    for (a, b), wi in zip(edges, w, strict=True):
        d = min(b - a, n - (b - a))
        expected[d] += wi
    expected /= expected.sum()
    observed = np.bincount(dists, minlength=n // 2 + 1) / len(dists)
    assert np.abs(observed - expected).max() < 0.02


def test_survival_function():
    spec = mipt.MonitoredCircuit(
        num_system=8,
        edges=graphs.cycle(8),
        layers=40,
        measurement_rate=0.3,
        reference="single",
        checkpoints="every_layer",
        probes=[mipt.code_dimension()],
    )
    r = spec.run(200, seed=2)
    layers, surv = r.survival()
    np.testing.assert_array_equal(layers, np.arange(41))
    assert surv[0] == 1.0 and np.all(np.diff(surv) <= 0)
    # without stopping, S(t) equals the mean reference entropy (it is 0 or 1)
    np.testing.assert_allclose(surv, r.mean("code_dimension"))


def test_erasure_probes_without_measurements():
    n = 10
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        layers=6,
        measurement_rate=0.0,
        reference="purification",
        checkpoints=[0, 6],
        probes=[
            mipt.code_dimension(),
            mipt.erasure_first_failure(samples=5),
            mipt.uncorrectable_region_counts(2),
            mipt.lost_information([1, 2, 3]),
        ],
    )
    r = spec.run(5, seed=0)
    np.testing.assert_array_equal(r.values["code_dimension"], n)  # nothing purifies
    np.testing.assert_array_equal(r.values["erasure_first_failure"], 1)
    counts = r.values["uncorrectable_region_counts[2]"]
    np.testing.assert_array_equal(counts, np.broadcast_to([n, n * (n - 1) / 2], counts.shape))
    np.testing.assert_array_equal(r.values["lost_information[1, 2, 3]"], 6)


def test_mean_and_censoring_helpers():
    spec = mipt.MonitoredCircuit(**SPECS["all_edges_complete_with_stop"])
    r = spec.run(50, seed=3)
    assert r.censored.dtype == bool
    assert np.all((r.purification_layer >= 0) | r.censored)
    m = r.mean("code_dimension")
    assert m.shape == (len(spec.checkpoint_layers),)
    assert m[0] == 1.0  # the single reference starts entangled
    assert "trajectories/s" in repr(r)


def test_save_and_load_round_trip(tmp_path):
    spec = mipt.MonitoredCircuit(**SPECS["lattice_scramble_layers_gate_set_orientation"])
    r = spec.run(7, seed=[1, 2, 3])
    path = tmp_path / "result.npz"
    r.save(path)
    back = mipt.EnsembleResult.load(path)
    for name in r.values:
        np.testing.assert_array_equal(back.values[name], r.values[name])
    np.testing.assert_array_equal(back.purification_layer, r.purification_layer)
    assert back.seed_entropy == r.seed_entropy
    # the reloaded spec reproduces the run
    again = back.spec.run(7, seed=[1, 2, 3])
    for name in r.values:
        np.testing.assert_array_equal(again.values[name], r.values[name])


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(num_system=1), "at least 2"),
        (dict(measurement_rate=1.5), "measurement_rate"),
        (dict(schedule="nope"), "schedule"),
        (dict(gates_per_layer=3), "random_edges"),
        (dict(reference="ancilla"), "reference"),
        (dict(reference="single", reference_partner=99), "reference_partner"),
        (dict(probes=[mipt.code_dimension()]), "needs a reference"),
        (dict(stop_when_purified=True), "needs a reference"),
        (dict(checkpoints=[0, 99]), "checkpoints"),
        (dict(gate_set=[11520]), "gate_set"),
        (dict(system_order=[0, 0, 1, 2]), "permutation"),
        (dict(probes=[mipt.half_cut_entropy(), mipt.half_cut_entropy()]), "unique"),
        (dict(edges=[[0, 0]]), "self-loops"),
    ],
)
def test_validation(overrides, message):
    base = dict(num_system=4, edges=graphs.cycle(4), layers=3, measurement_rate=0.1)
    base.update(overrides)
    with pytest.raises((ValueError, TypeError), match=message):
        mipt.MonitoredCircuit(**base)


def test_with_edges_and_derived_properties():
    spec = mipt.MonitoredCircuit(
        num_system=6,
        edges=graphs.cycle(6),
        layers=2,
        measurement_rate=0.0,
        reference="purification",
    )
    assert spec.num_qubits == 12
    assert spec.reference_qubits == list(range(6, 12))
    other = spec.with_edges(graphs.complete(6))
    assert len(other.edges) == 15 and len(spec.edges) == 6
    np.testing.assert_array_equal(spec.checkpoint_layers, [2])


def test_graph_collection_gives_each_graph_a_block_of_circuits():
    n = 12
    bank = [graphs.watts_strogatz(n, 4, 0.3, seed=(9, i)) for i in range(3)]
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=bank,
        circuits_per_graph=4,
        layers=6,
        measurement_rate=0.2,
        schedule="random_edges",
    )
    assert spec.num_graphs == 3 and spec.num_trajectories == 12
    for t in (0, 3, 4, 7, 11):
        single = spec.with_edges(bank[t // 4])
        # trajectory t of the collection is trajectory t on graph t // circuits_per_graph
        assert str(spec.circuit(t, seed=4)) == str(single.circuit(t, seed=4))
    everything = spec.run(seed=4)  # all 3 * 4 trajectories by default
    np.testing.assert_array_equal(everything.graph_index, np.repeat(np.arange(3), 4))
    chunks = mipt.EnsembleResult.concatenate(
        [spec.run(5, seed=4), spec.run(seed=4, first_trajectory=5)]
    )
    np.testing.assert_array_equal(everything.num_measurements, chunks.num_measurements)
    # the same collection at another rate: same graphs, same circuit randomness
    other = mipt.MonitoredCircuit(
        num_system=n, edges=bank, circuits_per_graph=4, layers=6, measurement_rate=0.5,
        schedule="random_edges",
    )  # fmt: skip
    np.testing.assert_array_equal(other.run(seed=4).graph_index, everything.graph_index)
    with pytest.raises(ValueError, match="has 12 trajectories"):
        spec.run(13, seed=4)
    with pytest.raises(ValueError, match="has 12 trajectories"):
        spec.circuit(12, seed=4)
    with pytest.raises(ValueError, match="circuits_per_graph applies"):
        mipt.MonitoredCircuit(
            num_system=n, edges=bank[0], circuits_per_graph=2, layers=1, measurement_rate=0.1
        )
    with pytest.raises(ValueError, match="give the number of trajectories"):
        mipt.MonitoredCircuit(num_system=n, edges=bank[0], layers=1, measurement_rate=0.1).run(
            seed=1
        )


def test_probes_run_on_their_own_checkpoints():
    n = 10
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        layers=20,
        measurement_rate=0.3,
        reference="single",
        reference_partner=0,
        checkpoints="final",  # purification is tested only here
        probes=[
            mipt.code_dimension(checkpoints="every_layer"),
            mipt.uncorrectable_region_counts(2, checkpoints=[5, 10]),
            mipt.erasure_first_failure(samples=3, checkpoints=4),
        ],
        stop_when_purified=True,
    )
    r = spec.run(20, seed=3)
    np.testing.assert_array_equal(r.layers("code_dimension"), np.arange(21))
    np.testing.assert_array_equal(r.layers("uncorrectable_region_counts[2]"), [5, 10])
    np.testing.assert_array_equal(r.layers("erasure_first_failure"), [0, 4, 8, 12, 16, 20])
    assert r.values["code_dimension"].shape == (20, 21)
    assert r.values["uncorrectable_region_counts[2]"].shape == (20, 2, 2)
    assert r.values["erasure_first_failure"].shape == (20, 6, 3)
    # detection only at the spec's checkpoints: tau is the final layer or censored
    assert set(np.unique(r.purification_layer)) <= {-1, 20}
    k = r.values["code_dimension"]
    first_zero = [int(np.argmax(row == 0)) if (row == 0).any() else -1 for row in k]
    assert any(z not in (-1, 20) for z in first_zero)  # purified earlier, but not tested there


def test_code_probes_keep_exact_values_after_a_stop():
    n = 8
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        layers=40,
        measurement_rate=0.5,
        reference="purification",
        checkpoints="every_layer",
        probes=[
            mipt.code_dimension(),
            mipt.erasure_first_failure(samples=4),
            mipt.uncorrectable_region_counts(2),
            mipt.lost_information([0, 1]),
            mipt.erasure_lost_profile(samples=2),
            mipt.half_cut_entropy(),
            mipt.code_snapshot(checkpoints=[39, 40]),
        ],
        stop_when_purified=True,
    )
    r = spec.run(30, seed=5)
    tau = r.purification_layer
    assert (tau >= 0).mean() > 0.5
    for t in np.flatnonzero(tau >= 0):
        after = np.arange(41) > tau[t]
        np.testing.assert_array_equal(r.values["code_dimension"][t, after], 0)
        np.testing.assert_array_equal(r.values["erasure_first_failure"][t, after], n + 1)
        np.testing.assert_array_equal(r.values["uncorrectable_region_counts[2]"][t, after], 0)
        np.testing.assert_array_equal(r.values["lost_information[0, 1]"][t, after], 0)
        np.testing.assert_array_equal(r.values["erasure_lost_profile"][t, after], 0)
        assert np.all(np.isnan(r.values["half_cut_entropy"][t, after]))
        if tau[t] < 39:
            np.testing.assert_array_equal(r.values["code_snapshot"][t], n)
            np.testing.assert_array_equal(r.snapshots["code_snapshot"][1][t], -1)
            with pytest.raises(ValueError, match="had stopped"):
                r.code(t, 0)
    # the code rate counts purified circuits as k = 0 (no NaN to drop)
    assert not np.isnan(r.values["code_dimension"]).any()


def test_scramble_matchings_are_perfect_matchings():
    n = 9
    spec = mipt.MonitoredCircuit(
        num_system=n, edges=graphs.cycle(n), layers=0, measurement_rate=0.0, scramble_matchings=3
    )
    circuit = spec.circuit(0, seed=1)
    blocks = [ins for ins in circuit.flattened() if ins.name == "CLIFFORD_2Q"]
    assert len(blocks) == 3
    for ins in blocks:
        qubits = [t.value for t in ins.targets]
        assert len(qubits) == 2 * (n // 2) and len(set(qubits)) == len(qubits)


def test_graph_bank_save_load_round_trip(tmp_path):
    bank = [graphs.newman_watts(10, 2, 0.4, seed=i) for i in range(3)]
    spec = mipt.MonitoredCircuit(
        num_system=10,
        edges=bank,
        circuits_per_graph=2,
        layers=5,
        measurement_rate=0.2,
        reference="single",
        scramble_matchings=2,
        probes=[
            mipt.code_dimension(),
            mipt.uncorrectable_region_counts(2, checkpoints=[2, 5]),
            mipt.code_snapshot(checkpoints=[3]),
        ],
    )
    res = spec.run(seed=2)
    res.save(tmp_path / "bank.npz")
    back = mipt.EnsembleResult.load(tmp_path / "bank.npz")
    assert back.spec.num_graphs == 3 and back.spec.scramble_matchings == 2
    assert back.spec.circuits_per_graph == 2
    for g, h in zip(back.spec.graphs, spec.graphs, strict=True):
        np.testing.assert_array_equal(g, h)
    np.testing.assert_array_equal(back.layers("uncorrectable_region_counts[2]"), [2, 5])
    for name in res.values:
        np.testing.assert_array_equal(back.values[name], res.values[name])
    for a, b in zip(back.snapshots["code_snapshot"], res.snapshots["code_snapshot"], strict=True):
        np.testing.assert_array_equal(a, b)
    rerun = back.spec.run(seed=2)
    np.testing.assert_array_equal(rerun.values["code_dimension"], res.values["code_dimension"])


# -- regressions from the review ------------------------------------------------------------


def test_checkpoint_arrays_are_sorted_and_deduplicated():
    # np.geomspace(...).astype(int) has repeats; an int64 array used to skip the
    # sorting, so every checkpoint after the repeat was never evaluated.
    ck = np.geomspace(1, 60, 12).astype(np.int64)
    base = dict(
        num_system=8,
        edges=graphs.cycle(8),
        layers=60,
        measurement_rate=0.3,
        reference="single",
        reference_partner=0,
        probes=[mipt.code_dimension()],
    )
    from_array = mipt.MonitoredCircuit(**base, checkpoints=ck).run(100, seed=1)
    from_list = mipt.MonitoredCircuit(**base, checkpoints=ck.tolist()).run(100, seed=1)
    np.testing.assert_array_equal(from_array.checkpoint_layers, np.unique(ck))
    np.testing.assert_array_equal(from_array.purification_layer, from_list.purification_layer)
    assert not np.isnan(from_array.values["code_dimension"]).any()
    unsorted = mipt.MonitoredCircuit(**base, checkpoints=np.array([30, 0, 10], dtype=np.int64))
    np.testing.assert_array_equal(unsorted.checkpoint_layers, [0, 10, 30])
    assert not np.isnan(unsorted.run(5, seed=2).values["code_dimension"]).any()
    with pytest.raises(ValueError, match="integers"):
        mipt.MonitoredCircuit(**base, checkpoints=[2.5])


def test_dataclasses_replace_gives_an_independent_spec():
    import dataclasses

    base = mipt.MonitoredCircuit(
        num_system=12,
        edges=graphs.cycle(12),
        layers=8,
        measurement_rate=0.1,
        reference="purification",
        checkpoints="final",
        probes=[mipt.code_dimension(), mipt.half_cut_entropy()],
    )
    base.run(3, seed=1)  # fills base's compiled cache
    fields = dict(
        num_system=12, layers=8, measurement_rate=0.1, reference="purification", checkpoints="final"
    )
    for edges in (graphs.complete(12), graphs.path(12)):
        derived = dataclasses.replace(base, edges=edges, probes=[mipt.half_cut_entropy()])
        fresh = mipt.MonitoredCircuit(**fields, edges=edges, probes=[mipt.half_cut_entropy()])
        assert derived._compiled is not base._compiled
        np.testing.assert_array_equal(
            derived.run(20, seed=3).values["half_cut_entropy"],
            fresh.run(20, seed=3).values["half_cut_entropy"],
        )
        assert str(derived.circuit(0, seed=3)) == str(fresh.circuit(0, seed=3))
    single = mipt.MonitoredCircuit(
        num_system=12, edges=graphs.cycle(12), layers=2, measurement_rate=0.1,
        reference="single", reference_partner=0,
    )  # fmt: skip
    single.circuit(0, seed=1)
    assert "CX 9 12" in str(dataclasses.replace(single, reference_partner=9).circuit(0, seed=1))
    # symbolic checkpoints follow a new depth; explicit ones are kept
    deeper = dataclasses.replace(base, layers=20)
    np.testing.assert_array_equal(deeper.checkpoint_layers, [20])
    assert deeper.run(2, seed=1).final_layer.tolist() == [20, 20]
    every = mipt.MonitoredCircuit(
        **{**fields, "checkpoints": "every_layer"}, edges=graphs.cycle(12)
    )
    np.testing.assert_array_equal(dataclasses.replace(every, layers=3).checkpoint_layers, range(4))
    strided = mipt.MonitoredCircuit(**{**fields, "checkpoints": 4}, edges=graphs.cycle(12))
    np.testing.assert_array_equal(
        dataclasses.replace(strided, layers=10).checkpoint_layers, [0, 4, 8, 10]
    )
    listed = mipt.MonitoredCircuit(**{**fields, "checkpoints": [0, 8]}, edges=graphs.cycle(12))
    np.testing.assert_array_equal(dataclasses.replace(listed, layers=9).checkpoint_layers, [0, 8])


def test_numpy_scalar_fields_and_seeds_save(tmp_path):
    for n in np.array([8]):  # sizes from a numpy array
        spec = mipt.MonitoredCircuit(
            num_system=n,
            edges=graphs.cycle(int(n)),
            layers=4 * n,
            measurement_rate=np.float32(0.25),
            reference="single",
            reference_partner=n - 1,
            scramble_gates=np.int64(4),
            random_orientation=np.True_,
            checkpoints=np.int64(8),
            probes=[mipt.code_dimension()],
            stop_when_purified=np.True_,
        )
        result = spec.run(6, seed=np.int64(5))
        result.save(tmp_path / "r.npz")
        back = mipt.EnsembleResult.load(tmp_path / "r.npz")
        assert type(back.spec.layers) is int and back.spec.layers == 32
        assert back.spec.checkpoints == 8 and back.seed_entropy == 5
        np.testing.assert_array_equal(back.purification_layer, result.purification_layer)
        again = back.spec.run(6, seed=back.seed_sequence)
        np.testing.assert_array_equal(again.purification_layer, result.purification_layer)


def test_seed_record_identifies_the_seed_used(tmp_path):
    spec = mipt.MonitoredCircuit(
        num_system=12, edges=graphs.cycle(12), layers=20, measurement_rate=0.15,
        reference="single", checkpoints="every_layer", probes=[mipt.code_dimension()],
    )  # fmt: skip
    fresh = spec.run(30, seed=None)
    np.testing.assert_array_equal(
        spec.run(30, seed=fresh.seed_entropy).purification_layer, fresh.purification_layer
    )
    children = np.random.SeedSequence(42).spawn(2)
    a, b = spec.run(30, seed=children[0]), spec.run(30, seed=children[1])
    assert a.seed_spawn_key == (0,) and b.seed_spawn_key == (1,)
    np.testing.assert_array_equal(
        spec.run(30, seed=b.seed_sequence).purification_layer, b.purification_layer
    )
    with pytest.raises(ValueError, match="seeds differ"):
        mipt.EnsembleResult.concatenate([a, spec.run(30, seed=children[1], first_trajectory=30)])
    # tuple seeds: a chunk reloaded from disk still merges with an in-memory chunk
    first = spec.run(10, seed=(7, 3))
    first.save(tmp_path / "first.npz")
    second = spec.run(10, seed=(7, 3), first_trajectory=10)
    merged = mipt.EnsembleResult.concatenate(
        [mipt.EnsembleResult.load(tmp_path / "first.npz"), second]
    )
    np.testing.assert_array_equal(merged.trajectories, np.arange(20))
    np.testing.assert_array_equal(
        merged.purification_layer, spec.run(20, seed=(7, 3)).purification_layer
    )


def test_concatenate_refuses_other_protocols_and_repeated_trajectories():
    base = dict(
        num_system=12, edges=graphs.cycle(12), layers=10, reference="single",
        checkpoints="every_layer", probes=[mipt.code_dimension()],
    )  # fmt: skip
    lo = mipt.MonitoredCircuit(**base, measurement_rate=0.05).run(20, seed=1)
    hi = mipt.MonitoredCircuit(**base, measurement_rate=0.4).run(20, seed=1, first_trajectory=20)
    with pytest.raises(ValueError, match="measurement_rate"):
        mipt.EnsembleResult.concatenate([lo, hi])
    ring_bank = mipt.MonitoredCircuit(
        **{**base, "edges": graphs.complete(12)}, measurement_rate=0.05
    )
    with pytest.raises(ValueError, match="edges"):
        mipt.EnsembleResult.concatenate([lo, ring_bank.run(20, seed=1, first_trajectory=20)])
    purif = mipt.MonitoredCircuit(**{**base, "reference": "purification"}, measurement_rate=0.05)
    with pytest.raises(ValueError, match="reference"):
        mipt.EnsembleResult.concatenate([lo, purif.run(20, seed=1, first_trajectory=20)])
    with pytest.raises(ValueError, match="share trajectories"):
        mipt.EnsembleResult.concatenate([lo, lo])
    other_version = mipt.MonitoredCircuit(**base, measurement_rate=0.05).run(
        5, seed=1, first_trajectory=20
    )
    other_version.version = "0.0.0"
    with pytest.raises(ValueError, match="versions"):
        mipt.EnsembleResult.concatenate([lo, other_version])


def test_save_records_the_package_version(tmp_path):
    import json

    import sparsegf2

    spec = mipt.MonitoredCircuit(
        num_system=8, edges=graphs.cycle(8), layers=2, measurement_rate=0.1
    )
    result = spec.run(2, seed=1)
    assert result.version == sparsegf2.__version__
    result.save(tmp_path / "r.npz")
    with np.load(tmp_path / "r.npz") as z:
        assert json.loads(str(z["__meta__"]))["version"] == sparsegf2.__version__
    assert mipt.EnsembleResult.load(tmp_path / "r.npz").version == sparsegf2.__version__


def test_scramble_layers_continue_the_colour_cycle():
    # Restarting the cycle repeated the last scramble matching in the first
    # dynamics layer, so scramble_layers=1 on a brickwork was absorbed.
    def blocks(scramble_layers):
        spec = mipt.MonitoredCircuit(
            num_system=8, edges=graphs.cycle(8), layers=3, measurement_rate=0.0,
            scramble_layers=scramble_layers, gate_set=[gate("CX").clifford.index()],
        )  # fmt: skip
        return [
            sorted(zip(*[iter(t.value for t in ins.targets)] * 2, strict=True))
            for ins in spec.circuit(0, seed=1)
            if ins.name == "CLIFFORD_2Q"
        ]

    even, odd = [(0, 1), (2, 3), (4, 5), (6, 7)], [(0, 7), (1, 2), (3, 4), (5, 6)]
    assert blocks(0) == [even, odd, even]
    assert blocks(1) == [even, odd, even, odd]
    assert blocks(2) == [even, odd, even, odd, even]
    # with CX gates on the reference's partner, one scramble layer is no longer undone
    spec = mipt.MonitoredCircuit(
        num_system=8, edges=graphs.cycle(8), layers=1, measurement_rate=0.0,
        reference="single", reference_partner=0, scramble_layers=1,
        gate_set=[gate("CX").clifford.index()], checkpoints="every_layer",
        probes=[mipt.entropy([1])],
    )  # fmt: skip
    assert spec.run(1, seed=0).values["entropy[1]"].tolist() == [[1.0, 1.0]]


def test_spec_owns_read_only_copies_of_its_arrays():
    n = 12
    order = np.arange(n)
    rng = np.random.default_rng(0)
    specs = []
    for _ in range(2):
        rng.shuffle(order)  # one buffer reused for several orderings
        specs.append(
            mipt.MonitoredCircuit(
                num_system=n,
                edges=graphs.cycle(n),
                layers=4,
                measurement_rate=0.1,
                system_order=order,
                probes=[mipt.half_cut_entropy(), mipt.entropy_profile()],
            )  # fmt: skip
        )
    spec = specs[0]
    assert spec.system_order is not order
    assert spec.probes[0].regions[0] == tuple(spec.system_order[: n // 2].tolist())
    r = spec.run(50, seed=1)
    np.testing.assert_array_equal(
        r.values["half_cut_entropy"], r.values["entropy_profile"][:, :, n // 2]
    )
    gs = np.array([0, 1, 2])
    weights = np.ones(n)
    other = mipt.MonitoredCircuit(
        num_system=n, edges=graphs.cycle(n), layers=2, measurement_rate=0.1,
        schedule="random_edges", edge_weights=weights, gate_set=gs,
    )  # fmt: skip
    gs[:] = 11519
    weights[:] = 0
    np.testing.assert_array_equal(other.gate_set, [0, 1, 2])
    np.testing.assert_array_equal(other.edge_weights, np.ones(n))
    for arr in (other.gate_set, other.system_order, other.edge_weights, other.edges,
                other.checkpoint_layers):  # fmt: skip
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 1


def test_contiguous_distance_is_nan_without_a_code():
    spec = mipt.MonitoredCircuit(
        num_system=16, edges=graphs.cycle(16), layers=40, measurement_rate=0.25,
        reference="purification", checkpoints=[10, 20, 40],
        probes=[mipt.code_dimension(), mipt.contiguous_distance()],
    )  # fmt: skip
    r = spec.run(200, seed=1)
    k, d = r.values["code_dimension"], r.values["contiguous_distance"]
    assert (k == 0).any() and (k > 0).any()
    assert np.isnan(d[k == 0]).all() and (d[k > 0] >= 1).all()
    with_code = np.where(k > 0, d, 0).sum(axis=0) / np.maximum((k > 0).sum(axis=0), 1)
    np.testing.assert_allclose(r.mean("contiguous_distance"), with_code)


def test_probe_arguments_are_validated_like_the_spec():
    assert mipt.uncorrectable_region_counts(2.0).name == "uncorrectable_region_counts[2]"
    with pytest.raises(TypeError, match="max_size must be an integer"):
        mipt.uncorrectable_region_counts(2.7)
    with pytest.raises(TypeError, match="samples must be an integer"):
        mipt.erasure_first_failure(2.5)
    with pytest.raises(TypeError, match="checkpoints"):
        mipt.code_dimension(checkpoints=2.5)
    base = dict(num_system=8, edges=graphs.cycle(8), layers=256, measurement_rate=0.1)
    spec = mipt.MonitoredCircuit(
        **base, checkpoints=256 / 8, probes=[mipt.half_cut_entropy(checkpoints=64.0)]
    )
    np.testing.assert_array_equal(spec.checkpoint_layers, np.arange(0, 257, 32))  # a stride
    np.testing.assert_array_equal(spec.probe_layers("half_cut_entropy"), np.arange(0, 257, 64))
    with pytest.raises(ValueError, match="samples must be at least 1"):
        mipt.MonitoredCircuit(
            **base, reference="purification", probes=[mipt.erasure_lost_profile(samples=0)]
        )


# -- the exhaustive cutoff m*(n): a budget, or one value per system size --------------------


def _counts_spec(n: int, *probes: mipt.Probe) -> mipt.MonitoredCircuit:
    return mipt.MonitoredCircuit(
        num_system=n, edges=graphs.cycle(n), layers=6, measurement_rate=0.1,
        reference="purification", checkpoints=[3, 6], probes=list(probes),
    )  # fmt: skip


def test_uncorrectable_counts_default_cutoff_is_unchanged():
    p = mipt.uncorrectable_region_counts()
    assert (p.size, p.name, p.cutoff) == (2, "uncorrectable_region_counts[2]", None)
    assert mipt.uncorrectable_region_counts(3) == mipt.uncorrectable_region_counts(max_size=3)
    # a budget is resolved by the spec; until then, different budgets are different probes
    assert mipt.uncorrectable_region_counts(budget=100) != mipt.uncorrectable_region_counts(
        budget=200
    )


def test_budget_gives_the_same_probe_and_values_as_its_cutoff():
    n, budget = 12, 100  # C(12, 1) + C(12, 2) = 78 <= 100 < 78 + C(12, 3) = 298: m* = 2
    w = analysis.exhaustive_cutoff(n, budget)
    assert w == 2
    by_budget = _counts_spec(n, mipt.uncorrectable_region_counts(budget=budget))
    by_size = _counts_spec(n, mipt.uncorrectable_region_counts(max_size=w))
    assert by_budget.probes == by_size.probes
    assert mipt._protocol_difference(by_budget, by_size) is None
    a, b = by_budget.run(20, seed=4), by_size.run(20, seed=4)
    assert list(a.values) == list(b.values) == ["uncorrectable_region_counts[2]"]
    counts = a.values["uncorrectable_region_counts[2]"]
    np.testing.assert_array_equal(counts, b.values["uncorrectable_region_counts[2]"])
    assert counts.shape == (20, 2, 2) and counts.max() > 0  # some codes have small failures
    again = by_budget.replay(7, seed=4)  # replaying resolves the budget again, identically
    np.testing.assert_array_equal(again.values["uncorrectable_region_counts[2]"][0], counts[7])


def test_cutoff_mappings_resolve_for_each_system_size():
    sizes = {12: 3, 16: 2}
    # m*(12) = 1 (12 <= 20 < 12 + 66); m*(16) = 2 (16 + 120 = 136 <= 200 < 136 + 560)
    budgets = {12: 20, 16: 200}
    for n, w_size, w_budget in ((12, 3, 1), (16, 2, 2)):
        spec = _counts_spec(
            n,
            mipt.uncorrectable_region_counts(sizes),
            mipt.uncorrectable_region_counts(budget=budgets, name="counts"),
        )
        by_size, by_budget = spec.probes
        assert (by_size.name, by_size.size) == (f"uncorrectable_region_counts[{w_size}]", w_size)
        assert (by_budget.name, by_budget.size) == ("counts", w_budget)  # one key for every n
        res = spec.run(5, seed=1)
        assert res.values["counts"].shape == (5, 2, w_budget)
        # both count the uncorrectable m-subsets, so they agree on the sizes they share
        np.testing.assert_array_equal(
            res.values["counts"], res.values[by_size.name][..., :w_budget]
        )


def test_resolved_cutoff_probes_resolve_again_for_another_size():
    probes = [
        mipt.uncorrectable_region_counts(budget={12: 100, 16: 20}),
        mipt.uncorrectable_region_counts(max_size={12: 3, 16: 1}, name="pc"),
        # renamed after construction (as the JSON layer does): the name is kept
        dataclasses.replace(mipt.uncorrectable_region_counts(budget=20), name="renamed"),
    ]
    spec12 = _counts_spec(12, *probes)
    assert [(p.name, p.size) for p in spec12.probes] == [
        ("uncorrectable_region_counts[2]", 2),
        ("pc", 3),
        ("renamed", 1),
    ]
    # the resolved probes keep their cutoff, so a spec for another n resolves them again
    spec16 = _counts_spec(16, *spec12.probes)
    assert [(p.name, p.size) for p in spec16.probes] == [
        ("uncorrectable_region_counts[1]", 1),
        ("pc", 1),
        ("renamed", 1),
    ]


def test_cutoff_arguments_are_validated():
    with pytest.raises(ValueError, match="give max_size or budget, not both"):
        mipt.uncorrectable_region_counts(2, budget=100)
    with pytest.raises(ValueError, match=r"no entry for num_system = 12.*\[16, 32\]"):
        _counts_spec(12, mipt.uncorrectable_region_counts(max_size={32: 2, 16: 2}))
    with pytest.raises(ValueError, match=r"no entry for num_system = 12.*\[16\]"):
        _counts_spec(12, mipt.uncorrectable_region_counts(budget={16: 100}))
    message = (
        "budget 11 allows no exhaustive region size at n = 12 (C(12, 1) = 12 > 11); "
        "raise the budget or drop the probe"
    )
    probe = mipt.uncorrectable_region_counts(budget=11)  # n is not known yet
    with pytest.raises(ValueError, match=re.escape(f"probe {probe.name!r}: {message}")):
        _counts_spec(12, probe)
    with pytest.raises(ValueError, match=re.escape(message)):  # each entry knows its n
        mipt.uncorrectable_region_counts(budget={64: 50_000, 12: 11})
    with pytest.raises(ValueError, match=r"max_size must be in \[1, 12\], got 13"):
        _counts_spec(12, mipt.uncorrectable_region_counts(max_size=13))
    with pytest.raises(ValueError, match=r"max_size\[12\] must be in \[1, 12\], got 13"):
        mipt.uncorrectable_region_counts(max_size={12: 13})
    with pytest.raises(ValueError, match=r"max_size\[12\] must be in \[1, 12\], got 0"):
        mipt.uncorrectable_region_counts(max_size={12: 0})
    with pytest.raises(ValueError, match="budget must be non-negative"):
        mipt.uncorrectable_region_counts(budget=-1)
    with pytest.raises(ValueError, match=r"budget\[12\] must be non-negative"):
        mipt.uncorrectable_region_counts(budget={12: -1})
    with pytest.raises(ValueError, match="at least one system size"):
        mipt.uncorrectable_region_counts(max_size={})
    with pytest.raises(ValueError, match="system sizes .* at least 2, got 1"):
        mipt.uncorrectable_region_counts(budget={1: 100})
    # non-integer keys and values are errors, never truncated or parsed
    for bad in ({"12": 2}, {12.5: 2}, {True: 2}):
        with pytest.raises(TypeError, match="system sizes"):
            mipt.uncorrectable_region_counts(max_size=bad)
    with pytest.raises(TypeError, match=r"max_size\[12\] must be an integer, got 2.5"):
        mipt.uncorrectable_region_counts(max_size={12: 2.5})
    with pytest.raises(TypeError, match=r"budget\[12\] must be an integer, got '100'"):
        mipt.uncorrectable_region_counts(budget={12: "100"})
    with pytest.raises(TypeError, match="budget must be an integer"):
        mipt.uncorrectable_region_counts(budget=100.5)
    with pytest.raises(TypeError, match="max_size must be an integer"):
        mipt.uncorrectable_region_counts(max_size=[2, 3])
    # whole numbers of other types are accepted, as everywhere in the spec
    probe = mipt.uncorrectable_region_counts(budget={12.0: 1e2})
    assert _counts_spec(12, probe).probes[0].size == 2


def test_cutoff_probes_survive_save_and_load(tmp_path):
    spec = _counts_spec(
        12,
        mipt.uncorrectable_region_counts(budget=100),
        mipt.uncorrectable_region_counts(max_size={12: 3, 64: 2}, checkpoints=[6]),
        mipt.uncorrectable_region_counts(budget={12: 20, 64: 50_000}, name="pc"),
    )
    res = spec.run(6, seed=3)
    res.save(tmp_path / "cutoff.npz")
    back = mipt.EnsembleResult.load(tmp_path / "cutoff.npz")
    names = ["uncorrectable_region_counts[2]", "uncorrectable_region_counts[3]", "pc"]
    assert list(back.values) == list(res.values) == names
    for name in names:
        np.testing.assert_array_equal(back.values[name], res.values[name])
    assert back.spec.probes == spec.probes
    assert [p.cutoff for p in back.spec.probes] == [p.cutoff for p in spec.probes]
    more = spec.run(2, seed=3, first_trajectory=6)
    merged = mipt.EnsembleResult.concatenate([back, more])  # the same protocol
    assert merged.num_trajectories == 8


def _edit_saved_probe(src, dst, **changes) -> None:
    """Copy the saved result ``src`` to ``dst`` with fields of its first stored probe changed."""
    with np.load(src, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["__meta__"]))
    meta["probes"][0].update(changes)
    arrays["__meta__"] = np.array(json.dumps(meta))
    np.savez_compressed(dst, **arrays)


def test_load_refuses_a_probe_whose_size_disagrees_with_its_cutoff(tmp_path):
    spec = _counts_spec(12, mipt.uncorrectable_region_counts(budget=100))  # 78 <= 100: w = 2
    res = spec.run(3, seed=1)
    res.save(tmp_path / "good.npz")
    # A cutoff that no longer gives the stored size (an edit, or a future rule change):
    # budget 1000 resolves to w = 4, but the file holds 2 columns under "[2]".
    for i, changes in enumerate(({"cutoff": {"budget": 1000}}, {"size": 3})):
        bad = tmp_path / f"bad{i}.npz"
        _edit_saved_probe(tmp_path / "good.npz", bad, **changes)
        with pytest.raises(ValueError, match="the stored probe sizes do not match their cutoff"):
            mipt.EnsembleResult.load(bad)
    # another budget with the same cutoff (78 <= 90 < 298) agrees with the file
    _edit_saved_probe(tmp_path / "good.npz", tmp_path / "same.npz", cutoff={"budget": 90})
    back = mipt.EnsembleResult.load(tmp_path / "same.npz")
    assert back.spec.probes[0].cutoff == ("budget", 90)
    np.testing.assert_array_equal(
        back.values["uncorrectable_region_counts[2]"], res.values["uncorrectable_region_counts[2]"]
    )
