"""The monitored-circuit ensemble engine.

The central test runs each trajectory's exported circuit through the general
``Simulator`` (itself tested against Stim) and requires the engine's recorded
values to match exactly at every checkpoint.
"""

import numpy as np
import pytest

from sparsegf2_v3 import Simulator, gate, graphs, mipt
from sparsegf2_v3 import observables as ob


def _replay(spec: mipt.MonitoredCircuit, trajectory: int, seed):
    """Probe values and purity at every tick, from running the exported circuit."""
    circuit = spec.circuit(trajectory, seed=seed)
    refs = spec.reference_qubits
    order = spec.system_order.tolist()

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
        raise AssertionError(p.kind)

    def observe(sim, tick):
        values = {p.name: probe(sim, p) for p in spec.probes if not p.kind.startswith("erasure")}
        pure = sim.entropy(refs) == 0 if refs else False
        return values, pure

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
            mipt.lost_information(list(range(7))),
            mipt.entropy([2, 3, 13]),
            mipt.entropy_profile(),
            mipt.mutual_information([0, 1, 2], [6, 7, 8]),
            mipt.tripartite_information(),
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
        probes=[mipt.code_dimension(), mipt.contiguous_distance(), mipt.lost_information([0, 4])],
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
    "production_protocol_ws_bank_matching_scramble": dict(
        num_system=16,
        edges=[graphs.watts_strogatz(16, 4, 0.2, seed=(5, i)) for i in range(4)],
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
        layers=12,
        measurement_rate=0.2,
        reference="purification",
        scramble_gates=5,
        scramble_matchings=2,
        scramble_layers=2,
        checkpoints=[0, 3, 12],
        probes=[mipt.code_dimension(), mipt.half_cut_entropy(), mipt.contiguous_distance()],
    ),
    "weighted_bank": dict(
        num_system=10,
        edges=[graphs.complete(10)] * 2,
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
}


@pytest.mark.parametrize("name", list(SPECS))
def test_engine_matches_exported_circuits_exactly(name):
    spec = mipt.MonitoredCircuit(**SPECS[name])
    seed = 2024
    result = spec.run(6, seed=seed)
    for t in range(6):
        obs = _replay(spec, t, seed)
        tau_expected = -1
        stopped = False
        for ci, layer in enumerate(spec.checkpoint_layers):
            values, pure = obs[int(layer)]
            if stopped:
                for p in spec.probes:
                    assert np.all(np.isnan(result.values[p.name][t, ci]))
                continue
            for name, expected in values.items():
                got = result.values[name][t, ci]
                np.testing.assert_allclose(got, expected, err_msg=f"{name} t={t} layer={layer}")
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
            mipt.erasure_correctable_profile(samples=6),
        ],
    )
    r = spec.run(10, seed=4)
    lost = r.values["erasure_lost_profile"]
    frac = r.values["erasure_correctable_profile"]
    k = r.values["code_dimension"]
    assert lost.shape == (10, 2, n + 1)
    np.testing.assert_array_equal(lost[:, :, 0], 0)
    np.testing.assert_allclose(lost[:, :, n], 2 * k)
    np.testing.assert_array_equal(frac[:, :, 0], 1)
    assert np.all(np.diff(lost, axis=2) >= -1e-12)  # mean of monotone curves is monotone
    assert np.all(lost[:, 0, 1:] == 2 * np.arange(1, n + 1))  # fresh Bell pairs: 2 bits per qubit


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


def test_graph_bank_assigns_graph_t_mod_g():
    n = 12
    bank = [graphs.watts_strogatz(n, 4, 0.3, seed=(9, i)) for i in range(3)]
    spec = mipt.MonitoredCircuit(
        num_system=n, edges=bank, layers=6, measurement_rate=0.2, schedule="random_edges"
    )
    assert spec.num_graphs == 3 and isinstance(spec.edges, tuple)
    for t in (0, 1, 2, 4, 7):
        single = spec.with_edges(bank[t % 3])
        # the circuit of trajectory t on the bank equals trajectory t on graph t mod 3
        assert str(spec.circuit(t, seed=4)) == str(single.circuit(t, seed=4))
    a = spec.run(7, seed=4)
    b = mipt.EnsembleResult.concatenate(
        [spec.run(3, seed=4), spec.run(4, seed=4, first_trajectory=3)]
    )
    np.testing.assert_array_equal(a.num_measurements, b.num_measurements)


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
        layers=5,
        measurement_rate=0.2,
        reference="single",
        scramble_matchings=2,
        probes=[mipt.code_dimension()],
    )
    res = spec.run(5, seed=2)
    res.save(tmp_path / "bank.npz")
    back = mipt.EnsembleResult.load(tmp_path / "bank.npz")
    assert back.spec.num_graphs == 3 and back.spec.scramble_matchings == 2
    for g, h in zip(back.spec.graphs, spec.graphs, strict=True):
        np.testing.assert_array_equal(g, h)
    rerun = back.spec.run(5, seed=2)
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

    import sparsegf2_v3

    spec = mipt.MonitoredCircuit(
        num_system=8, edges=graphs.cycle(8), layers=2, measurement_rate=0.1
    )
    result = spec.run(2, seed=1)
    assert result.version == sparsegf2_v3.__version__
    result.save(tmp_path / "r.npz")
    with np.load(tmp_path / "r.npz") as z:
        assert json.loads(str(z["__meta__"]))["version"] == sparsegf2_v3.__version__
    assert mipt.EnsembleResult.load(tmp_path / "r.npz").version == sparsegf2_v3.__version__


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
