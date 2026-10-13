"""The monitored-circuit ensemble engine.

The central test runs each trajectory's exported circuit through the general
``Simulator`` (itself tested against Stim) and requires the engine's recorded
values to match exactly at every checkpoint.
"""

import numpy as np
import pytest

from sparsegf2_v3 import Simulator, graphs, mipt
from sparsegf2_v3 import observables as ob


def _replay(spec: mipt.MonitoredCircuit, trajectory: int, seed):
    """Probe values, profiles and purification layer from running the exported circuit."""
    circuit = spec.circuit(trajectory, seed=seed)
    refs = spec.reference_qubits
    order = spec.system_order.tolist()
    n = spec.num_system

    def probe(sim, p):
        if p.kind == "entropy":
            return sim.entropy(p.region)
        if p.kind == "half_cut":
            return sim.entropy(order[: n // 2])
        if p.kind == "code_dimension":
            return sim.entropy(refs)
        if p.kind == "contiguous_distance":
            return ob.contiguous_code_distance(sim, order, refs, periodic=spec.periodic)
        if p.kind == "lost_information":
            return ob.lost_logical_information(sim, p.region, refs)
        if p.kind == "mean_weight":
            return ob.stabilizer_weights(sim).mean()
        raise AssertionError(p.kind)

    def observe(sim, tick):
        values = [probe(sim, p) for p in spec.probes if not p.kind.startswith("erasure")]
        profile = ob.entropy_profile(sim, order) if spec.record_profile else None
        pure = sim.entropy(refs) == 0 if refs else False
        return values, profile, pure

    sim = Simulator(spec.num_qubits, phases=False, seed=0)
    observations = dict(sim.run(circuit, observe=observe).observations)
    return observations


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
            mipt.mean_stabilizer_weight(),
            mipt.entropy([2, 3, 13]),
        ],
        record_profile=True,
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
        probes=[mipt.code_dimension(), mipt.half_cut_entropy()],
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
        probes=[mipt.code_dimension(), mipt.mean_stabilizer_weight()],
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
        probes=[mipt.code_dimension(), mipt.contiguous_distance()],
        system_order=list(range(15, -1, -1)),
        periodic=False,
        record_profile=True,
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
}


@pytest.mark.parametrize("name", list(SPECS))
def test_engine_matches_exported_circuits_exactly(name):
    spec = mipt.MonitoredCircuit(**SPECS[name])
    seed = 2024
    result = spec.run(6, seed=seed)
    exact = [j for j, p in enumerate(spec.probes) if not p.kind.startswith("erasure")]
    for t in range(6):
        obs = _replay(spec, t, seed)
        tau_expected = -1
        stopped = False
        for ci, layer in enumerate(spec.checkpoint_layers):
            values, profile, pure = obs[int(layer)]
            if stopped:
                assert np.all(np.isnan(result.values[spec.probes[0].name][t, ci]))
                continue
            for j_out, j in enumerate(exact):
                got = result.values[spec.probes[j].name][t, ci]
                assert got == pytest.approx(values[j_out]), (spec.probes[j].name, t, int(layer))
            if spec.record_profile:
                np.testing.assert_array_equal(result.profiles[t, ci], profile)
            if pure and tau_expected < 0 and spec.reference != "none":
                tau_expected = int(layer)
                stopped = spec.stop_when_purified
        if spec.reference != "none":
            assert result.purification_layer[t] == tau_expected


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


def test_erasure_probes_are_consistent():
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
            mipt.erasure_lost_information(0, samples=5),
            mipt.erasure_lost_information(n, samples=5),
            mipt.erasure_lost_information(3, samples=50),
            mipt.erasure_correctable_fraction(1, samples=20),
        ],
    )
    r = spec.run(5, seed=0)
    k = r.values["code_dimension"]
    np.testing.assert_array_equal(k, n)  # no measurements: nothing purifies
    np.testing.assert_array_equal(r.values["erasure_lost_information[0]"], 0)
    np.testing.assert_array_equal(r.values[f"erasure_lost_information[{n}]"], 2 * k)
    np.testing.assert_array_equal(r.values["erasure_lost_information[3]"], 6)
    np.testing.assert_array_equal(r.values["erasure_correctable_fraction[1]"], 0)


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
    np.testing.assert_array_equal(back.profiles, r.profiles)
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
