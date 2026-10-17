"""The tableau_snapshot probe and runs on explicit trajectory indices.

Every snapshot is checked against an independent signed simulator (Stim, running the exported
circuit of the trajectory, signs dropped) and against brute-force enumeration of the stabilizer
group for small systems; the physical code against the engine's own ``code_snapshot``; and a
run on ``indices`` against the rows of a range run. Every comparison is exact.
"""

import itertools
import math

import numba
import numpy as np
import pytest

from _rowspace_oracle import (
    canonical,
    dense_rows,
    entropy_by_enumeration,
    int_rows,
    log2_exact,
    pack,
    rref,
    span,
    stim_tick_states,
    support,
    symplectic,
)
from sparsegf2 import PhaseFreeTableau, Simulator, api, graphs, mipt
from sparsegf2._kernel import entropy as kernel_entropy


def ws_spec(n, layers, rate, probes, *, beta=0.2, graph_seed=1, **kwargs):
    """The study's protocol: a Watts-Strogatz graph (k = 4), n // 2 random edges per layer,
    n Bell pairs with the references, no scramble."""
    return mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.watts_strogatz(n, 4, beta, seed=graph_seed),
        layers=layers,
        measurement_rate=rate,
        schedule="random_edges",
        reference=kwargs.pop("reference", "purification"),
        probes=probes,
        **kwargs,
    )


def bell_pairs(n: int) -> np.ndarray:
    """The canonical rows of n Bell pairs (system i, reference n + i), written out: the rows
    X_i X_{n+i} (pivot x_i) for i = 0..n-1, then Z_i Z_{n+i} (pivot z_i)."""
    rows = np.zeros((2 * n, 4 * n), dtype=np.uint8)
    for i in range(n):
        rows[i, [i, n + i]] = 1
        rows[n + i, [2 * n + i, 3 * n + i]] = 1
    return rows


# -- the snapshot itself -----------------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 8, 11])
def test_layer_zero_holds_the_bell_pairs(n):
    probes = [mipt.code_dimension(checkpoints=[0]), mipt.tableau_snapshot(checkpoints=[0, 6])]
    res = ws_spec(n, 6, 0.2, probes).run(4, seed=1)
    expected = bell_pairs(n)
    assert canonical(expected).tolist() == expected.tolist()  # already canonical
    for t in range(4):
        tab = res.tableau(t, 0)
        np.testing.assert_array_equal(tab.canonical(), expected)
        assert tab.digest() == PhaseFreeTableau(expected).digest()
        assert tab.physical_tableau(n).rank == 0  # r_0 = 0
        assert res.code(t, 0).num_logical == n == res.values["code_dimension"][t, 0]  # k_0 = n
    np.testing.assert_array_equal(res.values["tableau_snapshot"], 2 * n)
    np.testing.assert_array_equal(res.snapshots["tableau_snapshot"][1], 2 * n)


CASES = [  # n, beta, measurement rate, layers: odd and even n, several rates and graphs
    (5, 0.0, 0.1, 16),
    (6, 0.3, 0.3, 12),
    (7, 0.2, 0.05, 24),
    (9, 0.6, 0.2, 18),
    (12, 0.1, 0.15, 20),
    (16, 1.0, 0.1, 16),
    # several 64-bit words per half row: N = 66, 80 and 140 (ceil(N / 64) = 2, 2, 3 while the
    # system alone takes 1, 1, 2), the layout of every study run with n >= 33
    (33, 0.2, 0.15, 6),
    (40, 0.1, 0.1, 6),
    (70, 0.3, 0.1, 4),
]


@pytest.mark.parametrize(("n", "beta", "rate", "layers"), CASES)
def test_every_snapshot_is_a_pure_state_whose_code_has_the_probed_dimension(n, beta, rate, layers):
    spec = ws_spec(
        n,
        layers,
        rate,
        [mipt.code_dimension(), mipt.tableau_snapshot(checkpoints="every_layer")],
        beta=beta,
        checkpoints="every_layer",
    )
    trajectories = 5 if n <= 16 else 3
    res = spec.run(trajectories, seed=n)
    generators, counts = res.snapshots["tableau_snapshot"]
    # every generator of the joint state over all N = 2n qubits: ceil(N / 64) words per half
    assert generators.shape == (trajectories, layers + 1, 2 * n, 2 * ((2 * n + 63) // 64))
    seen = set()
    for t in range(trajectories):
        for layer in range(layers + 1):
            assert counts[t, layer] == 2 * n
            written = unpack_raw(generators[t, layer], 2 * n)
            rows = int_rows(written)
            # 2n independent commuting rows, as the engine wrote them (before canonicalization)
            assert len(rref(rows, range(4 * n))) == 2 * n
            assert all(symplectic(a, b, 2 * n) == 0 for a, b in itertools.combinations(rows, 2))
            tab = res.tableau(t, layer)
            assert tab.is_pure() and tab.rank == 2 * n and tab.num_qubits == 2 * n
            np.testing.assert_array_equal(tab.canonical(), canonical(written))
            r = tab.physical_tableau(n).rank
            k = int(res.values["code_dimension"][t, layer])
            assert n - r == k == tab.entropy(range(n, 2 * n))  # k_t = n - r_t = S(R)
            seen.add(k)
    assert len(seen) > 2, seen  # the codes shrink: the comparison is not vacuous


def unpack_raw(raw: np.ndarray, num_qubits: int) -> np.ndarray:
    """The engine's packed rows as 0/1 rows, bit by bit (independent of the package)."""
    wq = (num_qubits + 63) // 64
    out = np.zeros((raw.shape[0], 2 * num_qubits), dtype=np.uint8)
    for i in range(raw.shape[0]):
        for q in range(num_qubits):
            for half in range(2):
                out[i, half * num_qubits + q] = (int(raw[i, half * wq + q // 64]) >> (q % 64)) & 1
    return out


@pytest.mark.parametrize(
    ("n", "reference"),
    [
        (10, "purification"),
        (10, "single"),
        (10, "none"),
        # several words per half row: the joint state on N = 66, 80, 140 or 65 qubits, its
        # physical code on n, with fewer words per half (2 against 1 at n = 33, 40 and 64)
        (33, "purification"),
        (40, "purification"),
        (70, "purification"),
        (64, "single"),
        (65, "single"),
        (65, "none"),
    ],
)
def test_the_physical_code_is_the_native_code_snapshot(n, reference):
    layers, checkpoints = 24, [0, 3, 8, 16, 24]
    common = dict(beta=0.3, checkpoints="every_layer", reference=reference)
    first = [mipt.code_dimension()] if reference != "none" else []
    joint = ws_spec(
        n, layers, 0.12, [*first, mipt.tableau_snapshot(checkpoints=checkpoints)], **common
    ).run(6, seed=3)
    native = ws_spec(
        n, layers, 0.12, [*first, mipt.code_snapshot(checkpoints=checkpoints)], **common
    ).run(6, seed=3)
    big = joint.spec.num_qubits
    assert big == {"purification": 2 * n, "single": n + 1, "none": n}[reference]
    generators = joint.snapshots["tableau_snapshot"][0]
    assert generators.shape == (6, len(checkpoints), big, 2 * ((big + 63) // 64))
    assert native.snapshots["code_snapshot"][0].shape[2:] == (n, 2 * ((n + 63) // 64))
    for name in joint.values.keys() - {"tableau_snapshot"}:
        np.testing.assert_array_equal(joint.values[name], native.values[name])
    np.testing.assert_array_equal(joint.final_layer, native.final_layer)
    for t in range(6):
        for e, layer in enumerate(checkpoints):
            ours, theirs = joint.code(t, e), native.code(t, e)
            assert ours.num_qubits == theirs.num_qubits == n
            np.testing.assert_array_equal(
                canonical(ours.stabilizer_matrix()), canonical(theirs.stabilizer_matrix())
            )
            assert ours.num_logical == theirs.num_logical
            if first:  # k_t = n - r_t = the code_dimension probe = S(R)
                tab = joint.tableau(t, e)
                k = int(joint.values["code_dimension"][t, layer])
                assert ours.num_logical == k == tab.entropy(range(n, big))


STIM_CASES = [  # n, beta, rate, layers, reference
    (5, 0.0, 0.1, 10, "purification"),
    (6, 0.3, 0.25, 10, "purification"),
    (7, 0.2, 0.05, 14, "purification"),
    (9, 0.5, 0.3, 10, "purification"),
    (12, 0.1, 0.15, 12, "purification"),
    (16, 1.0, 0.1, 8, "purification"),
    (11, 0.2, 0.1, 12, "single"),
    (8, 0.2, 0.2, 10, "none"),
    # several words per half row (N = 66, 80, 140, 65, 70)
    (33, 0.2, 0.15, 6, "purification"),
    (40, 0.1, 0.1, 6, "purification"),
    (70, 0.2, 0.1, 4, "purification"),
    (64, 0.2, 0.1, 6, "single"),
    (70, 0.2, 0.1, 6, "none"),
]


@pytest.mark.parametrize(("n", "beta", "rate", "layers", "reference"), STIM_CASES)
def test_snapshots_equal_stim_on_the_exported_circuit(n, beta, rate, layers, reference):
    """v1.4 §9.2 (h): the stored joint row space equals an independent signed simulator's state
    at every tested depth, signs dropped (Stim, running the exported circuit, Bell pairs and
    layer structure included: tick t is the state after layer t)."""
    spec = ws_spec(
        n,
        layers,
        rate,
        [mipt.tableau_snapshot(checkpoints="every_layer")],
        beta=beta,
        reference=reference,
    )
    rows_of = [3, 0, 8]
    res = spec.run(seed=17, indices=rows_of)
    for row, u in enumerate(rows_of):
        circuit = spec.circuit(u, seed=17).to_stim()
        states = stim_tick_states(circuit, spec.num_qubits, seed=u)
        assert len(states) == layers + 1
        for layer in range(layers + 1):
            tab = res.tableau(row, layer)
            np.testing.assert_array_equal(
                tab.canonical(), canonical(states[layer]), err_msg=f"u={u} layer={layer}"
            )
            assert tab.digest() == PhaseFreeTableau(states[layer]).digest()


@pytest.mark.parametrize("n", [3, 4])
def test_entropies_of_engine_snapshots_against_enumeration(n):
    """The joint state's entropies, I(A:R) by v1.4 §5.4's physical-rank formula, and
    I(A:R) + I(V\\A:R) = 2k (v1.4 §5.8), on engine snapshots, by enumerating 2^(2n) elements."""
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.complete(n),
        layers=6,
        measurement_rate=0.25,
        schedule="random_edges",
        reference="purification",
        checkpoints="every_layer",
        probes=[mipt.code_dimension(), mipt.tableau_snapshot(checkpoints="every_layer")],
    )
    res = spec.run(4, seed=2)
    big, system, reference = 2 * n, list(range(n)), list(range(n, 2 * n))
    low, ref_mask = (1 << n) - 1, sum(1 << q for q in range(n, 2 * n))
    ks = set()
    for t in range(4):
        for layer in range(7):
            tab = res.tableau(t, layer)
            elements = span(int_rows(tab.canonical()))
            assert len(elements) == 2**big
            for size in range(big + 1):
                for region in itertools.combinations(range(big), size):
                    assert tab.entropy(region) == entropy_by_enumeration(elements, big, region)
            physical = {
                (g & low) | (((g >> big) & low) << n)
                for g in elements
                if support(g, big) & ref_mask == 0
            }
            r = log2_exact(len(physical))
            k = n - r
            assert k == res.values["code_dimension"][t, layer]
            ks.add(k)

            def r_of(region, physical=physical):
                mask = sum(1 << q for q in region)
                return log2_exact(sum(1 for p in physical if support(p, n) & ~mask == 0))

            for size in range(n + 1):
                for a in itertools.combinations(system, size):
                    b = [q for q in system if q not in a]
                    info = tab.mutual_information(a, reference)
                    s_a = entropy_by_enumeration(elements, big, a)
                    s_ar = entropy_by_enumeration(elements, big, [*a, *reference])
                    assert info == s_a + k - s_ar
                    assert info == k + 2 * len(a) - n - r_of(a) + r_of(b)
                    assert info + tab.mutual_information(b, reference) == 2 * k
    assert len(ks) > 1, ks


def test_the_kernel_writes_every_stabilizer_row_and_never_a_destabilizer(rng):
    """joint_snapshot_rows on a phase-tracking tableau (destabilizer words after the
    stabilizer words) writes exactly the stabilizers: the rows of stabilizer_arrays."""
    for n in (3, 64, 70, 130):
        sim = Simulator(n, phases=True, seed=1)
        for _ in range(3 * n):
            a, b = (int(q) for q in rng.choice(n, size=2, replace=False))
            sim.do("CX", a, b)
            sim.do(["H", "S"][int(rng.integers(2))], a)
            if rng.random() < 0.2:
                sim.measure(b)
        tab = sim._tab
        assert tab.shape[1] == 2 * ((n + 63) // 64)  # the destabilizer words are there
        wq = (n + 63) // 64
        out = np.full((n, 2 * wq), np.uint64(0xFFFF), dtype=np.uint64)  # garbage is cleared
        assert kernel_entropy.joint_snapshot_rows(tab, n, n, out) == n
        xs, zs, _ = sim.stabilizer_arrays()
        np.testing.assert_array_equal(out, pack(np.concatenate([xs, zs], axis=1), n))


# -- runs on explicit indices -----------------------------------------------------------------


def collection_spec(**changes):
    """A collection of 4 graphs with 3 circuits each, and a probe of every kind that draws
    from a stream of its own: the circuit stream, the sampled orderings of
    erasure_lost_profile and erasure_first_failure, and both correlated_erasure branches
    (the exact ring sampler and the chain)."""
    bank = [graphs.watts_strogatz(10, 4, 0.3, seed=(3, g)) for g in range(4)]
    settings = dict(
        num_system=10,
        edges=bank,
        circuits_per_graph=3,
        layers=12,
        measurement_rate=0.15,
        schedule="random_edges",
        reference="purification",
        checkpoints="every_layer",
        probes=[
            mipt.code_dimension(),
            mipt.contiguous_distance(checkpoints=[6, 12]),
            mipt.erasure_lost_profile(3, checkpoints=[5, 12]),
            mipt.erasure_first_failure(5, checkpoints=[4, 12]),
            mipt.correlated_erasure([0.0, 1.0], [2, 3], 4, "ring", geometry="cycle"),
            mipt.correlated_erasure([0.5], [3], 3, "chain", burn_in=5, checkpoints=[8]),
            mipt.tableau_snapshot(checkpoints=[0, 6, 12]),
        ],
    )
    settings.update(changes)
    return mipt.MonitoredCircuit(**settings)


def assert_rows_equal(part, whole, rows):
    np.testing.assert_array_equal(part.trajectories, np.asarray(whole.trajectories)[rows])
    assert list(part.values) == list(whole.values)
    for name in whole.values:
        np.testing.assert_array_equal(part.values[name], whole.values[name][rows], err_msg=name)
    for name, (generators, counts) in whole.snapshots.items():
        np.testing.assert_array_equal(part.snapshots[name][0], generators[rows])
        np.testing.assert_array_equal(part.snapshots[name][1], counts[rows])
    for field in ("purification_layer", "final_layer", "num_measurements"):
        np.testing.assert_array_equal(getattr(part, field), getattr(whole, field)[rows])


def test_a_run_on_indices_equals_those_rows_of_a_range_run():
    spec = collection_spec()
    whole = spec.run(seed=4)
    assert whole.num_trajectories == 12
    part = spec.run(seed=4, indices=[5, 2, 9])
    assert part.trajectories.tolist() == [5, 2, 9]
    assert part.graph_index.tolist() == [1, 0, 3]
    assert_rows_equal(part, whole, [5, 2, 9])
    # the streams keyed by the trajectory index are exercised: the samples vary
    assert np.unique(whole.values["erasure_first_failure"]).size > 3
    assert np.unique(whole.values["erasure_lost_profile"]).size > 3
    assert np.unique(whole.values["ring"]).size > 1
    assert np.unique(whole.values["chain"]).size > 1
    one = spec.run(seed=4, indices=np.array([11]))
    assert_rows_equal(one, whole, [11])
    empty = spec.run(seed=4, indices=[])
    assert empty.num_trajectories == 0 and empty.values["code_dimension"].shape == (0, 13)


def test_indices_on_a_single_graph_reach_any_trajectory():
    spec = ws_spec(8, 10, 0.2, [mipt.code_dimension(), mipt.tableau_snapshot()])
    far = 10**12 + 7
    by_range = spec.run(2, seed=6, first_trajectory=far)
    by_index = spec.run(seed=6, indices=[far + 1, far])
    assert_rows_equal(by_index, by_range, [1, 0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(indices=[1, 1]), "distinct: trajectory 1 appears 2 times"),
        (dict(indices=[0, -3]), "non-negative"),
        (dict(indices=[12]), "trajectory 12 does not exist"),
        (dict(indices=[[1, 2]]), "1-D"),
        (dict(indices=[1.5]), "integers"),
        (dict(indices=5), "1-D list"),
        (dict(indices=[1], trajectories=1), "not both"),
        (dict(indices=[1], first_trajectory=0), "not both"),
    ],
)
def test_bad_indices_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        collection_spec().run(seed=1, **kwargs)


def test_prefixes_checkpoints_and_threads_do_not_change_a_trajectory():
    def spec(layers, schedule, probes=None):
        return ws_spec(
            12,
            layers,
            0.1,
            probes
            if probes is not None
            else [mipt.code_dimension(), mipt.tableau_snapshot(checkpoints=schedule)],
            checkpoints="every_layer",
        )

    indices = [4, 0, 7, 2]
    short = spec(40, [10, 20, 30, 40]).run(seed=9, indices=indices)
    long = spec(80, [10, 20, 30, 40, 80]).run(seed=9, indices=indices)
    # prefix: the first 40 layers of an 80-layer run are the 40-layer run
    np.testing.assert_array_equal(
        long.snapshots["tableau_snapshot"][0][:, :4], short.snapshots["tableau_snapshot"][0]
    )
    np.testing.assert_array_equal(
        long.values["code_dimension"][:, :41], short.values["code_dimension"]
    )
    # checkpoints: moving the snapshots (or dropping them) leaves the trajectory alone
    moved = spec(40, [5, 20, 25, 40]).run(seed=9, indices=indices)
    for a, b in ((1, 1), (3, 3)):
        np.testing.assert_array_equal(
            moved.snapshots["tableau_snapshot"][0][:, a],
            short.snapshots["tableau_snapshot"][0][:, b],
        )
    bare = spec(40, None, probes=[mipt.code_dimension()]).run(seed=9, indices=indices)
    np.testing.assert_array_equal(bare.values["code_dimension"], short.values["code_dimension"])
    np.testing.assert_array_equal(bare.num_measurements, short.num_measurements)
    # threads: one thread and several give the same snapshots
    threads = min(4, numba.config.NUMBA_NUM_THREADS)
    assert threads > 1 or numba.config.DISABLE_JIT, "set NUMBA_NUM_THREADS to at least 2"
    one = spec(40, [10, 20, 30, 40]).run(seed=9, indices=indices, num_threads=1)
    many = spec(40, [10, 20, 30, 40]).run(seed=9, indices=indices, num_threads=threads)
    for res in (one, many):
        assert_rows_equal(res, short, list(range(4)))


# -- stops, persistence, replay -----------------------------------------------------------------


def test_after_a_stop_nothing_is_saved():
    n = 6
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        layers=30,
        measurement_rate=0.5,
        reference="purification",
        checkpoints="every_layer",
        probes=[mipt.code_dimension(), mipt.tableau_snapshot(checkpoints=[2, 10, 20, 30])],
        stop_when_purified=True,
    )
    res = spec.run(20, seed=5)
    tau = res.purification_layer
    assert (tau >= 0).any() and ((tau >= 0) & (tau < 20)).any()
    counts = res.snapshots["tableau_snapshot"][1]
    for t in range(20):
        for e, layer in enumerate([2, 10, 20, 30]):
            stopped = 0 <= tau[t] < layer
            assert counts[t, e] == (-1 if stopped else 2 * n)
            assert res.values["tableau_snapshot"][t, e] == 2 * n  # N generators, either way
            if stopped:
                with pytest.raises(ValueError, match="had stopped"):
                    res.tableau(t, e)
                with pytest.raises(ValueError, match="had stopped"):
                    res.code(t, e)
                assert not res.snapshots["tableau_snapshot"][0][t, e].any()
            else:
                tab = res.tableau(t, e)
                assert tab.is_pure()
                k = n - tab.physical_tableau(n).rank
                assert k == res.values["code_dimension"][t, layer]


def test_save_load_concatenate_and_replay(tmp_path):
    spec = collection_spec()
    res = spec.run(seed=4, indices=[5, 2, 9])
    res.save(tmp_path / "part.npz")
    back = mipt.EnsembleResult.load(tmp_path / "part.npz")
    assert_rows_equal(back, res, [0, 1, 2])
    assert [p.kind for p in back.spec.probes][-1] == "tableau_snapshot"
    for row, e in itertools.product(range(3), range(3)):
        assert back.tableau(row, e) == res.tableau(row, e)
        assert back.tableau(row, e).digest() == res.tableau(row, e).digest()
    merged = mipt.EnsembleResult.concatenate([back, spec.run(seed=4, indices=[11, 0])])
    assert merged.trajectories.tolist() == [5, 2, 9, 11, 0]
    assert_rows_equal(merged, spec.run(seed=4), [5, 2, 9, 11, 0])
    replayed = back.replay(9, probes=[mipt.tableau_snapshot(checkpoints=[6])])
    assert replayed.tableau(0, 0) == res.tableau(2, 1)  # trajectory 9, layer 6
    np.testing.assert_array_equal(
        replayed.snapshots["tableau_snapshot"][0][0, 0], res.snapshots["tableau_snapshot"][0][2, 1]
    )


def test_a_result_without_the_probe_saves_as_before(tmp_path):
    spec = collection_spec(probes=[mipt.code_dimension(), mipt.code_snapshot(checkpoints=[12])])
    res = spec.run(seed=4, indices=[3, 1])
    res.save(tmp_path / "plain.npz")
    with np.load(tmp_path / "plain.npz") as z:
        assert sorted(z.files) == sorted(
            [
                "__meta__",
                "edges",
                "graph_ptr",
                "gate_set",
                "system_order",
                "trajectories",
                "checkpoint_layers",
                "final_layer",
                "num_measurements",
                "value::code_dimension",
                "value::code_snapshot",
                "snapshot::code_snapshot",
                "snapshot_counts::code_snapshot",
                "purification_layer",
            ]
        )
        assert z["snapshot::code_snapshot"].shape == (2, 1, 10, 2)
    back = mipt.EnsembleResult.load(tmp_path / "plain.npz")
    assert back.code(0, 0).num_logical == res.values["code_dimension"][0, 12]
    with pytest.raises(ValueError, match="no tableau_snapshot probe"):
        back.tableau(0, 0)


def test_one_snapshot_probe_per_spec():
    with pytest.raises(ValueError, match="at most one snapshot probe"):
        ws_spec(6, 4, 0.1, [mipt.code_snapshot(), mipt.tableau_snapshot()])
    with pytest.raises(ValueError, match="at most one snapshot probe"):
        ws_spec(6, 4, 0.1, [mipt.tableau_snapshot(), mipt.tableau_snapshot("other")])


# -- the JSON service ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "shape"),
    [(6, [3, 2, 12, 2]), (40, [3, 2, 80, 4])],  # N = 80: two words per half row
)
def test_the_json_probe_its_layout_summary_and_saved_file(tmp_path, n, shape):
    spec = {
        "num_system": n,
        "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 1},
        "layers": 8,
        "measurement_rate": 0.2,
        "schedule": "random_edges",
        "reference": "purification",
        "checkpoints": "every_layer",
        "probes": [{"kind": "code_dimension"}, {"kind": "tableau_snapshot", "checkpoints": [0, 8]}],
    }
    estimate = api.estimate_ensemble(spec, 3)
    rows = {row["name"]: row for row in estimate["layout"]["arrays"]}
    assert rows["snapshot::tableau_snapshot"]["shape"] == shape
    assert rows["snapshot_counts::tableau_snapshot"]["shape"] == [3, 2]
    assert "tableau_snapshot" in estimate["layout"]["note"]
    assert estimate["result_megabytes"] * 1e6 >= math.prod(shape) * 8
    out = api.run_ensemble(spec, 3, seed=1, output_dir=str(tmp_path / "run"))
    assert out["probes"]["tableau_snapshot"]["mean"] == [2.0 * n, 2.0 * n]
    assert any("EnsembleResult.tableau" in note for note in out["notes"])
    with np.load(tmp_path / "run" / "result.npz") as z:
        for name, row in rows.items():
            assert list(z[name].shape) == row["shape"], name
    saved = mipt.EnsembleResult.load(tmp_path / "run" / "result.npz")
    direct = api._build_spec(spec).run(3, seed=1)
    for t, e in itertools.product(range(3), range(2)):
        assert saved.tableau(t, e) == direct.tableau(t, e)
    assert api._probe_us(api._build_spec(spec).probes[1], api._build_spec(spec)) > 0
    # the saved run read back by load_results: the same probe summary and its storage rows
    loaded = api.load_results(str(tmp_path / "run"))
    assert loaded["probes"]["tableau_snapshot"] == out["probes"]["tableau_snapshot"]
    assert loaded["probes"]["code_dimension"] == out["probes"]["code_dimension"]
    stored = {row["name"]: row["megabytes"] for row in loaded["storage"]["arrays"]}
    assert {"snapshot::tableau_snapshot", "snapshot_counts::tableau_snapshot"} <= set(stored)
    assert 0 < stored["snapshot::tableau_snapshot"] * 1e6 <= math.prod(shape) * 8  # compressed
    with pytest.raises(ValueError, match="at most one snapshot probe"):
        api.run_ensemble(
            dict(spec, probes=[{"kind": "code_snapshot"}, {"kind": "tableau_snapshot"}]), 1, seed=1
        )


def test_rows_of_the_bell_state_unpack_independently():
    """The helper used above reads the engine packing as documented (a hand-made example)."""
    raw = pack(dense_rows([0b0101, 0b1010], 4), 2)  # X_0 Z_0 = Y_0 and X_1 Z_1 = Y_1
    np.testing.assert_array_equal(raw, [[0b01, 0b01], [0b10, 0b10]])
    np.testing.assert_array_equal(unpack_raw(raw, 2), dense_rows([0b0101, 0b1010], 4))


def test_the_json_probe_accepts_a_name():
    """(test_api.py's every-probe naming test cannot hold both snapshot kinds in one spec.)"""
    spec = {
        "num_system": 5,
        "graph": {"family": "cycle"},
        "layers": 3,
        "measurement_rate": 0.2,
        "reference": "purification",
        "probes": [{"kind": "tableau_snapshot", "name": "my_tableau_snapshot"}],
    }
    out = api.run_ensemble(spec, 2, seed=1)
    assert set(out["probes"]) == {"my_tableau_snapshot"}
    assert out["probes"]["my_tableau_snapshot"]["mean"] == [10.0]
