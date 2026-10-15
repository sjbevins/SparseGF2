"""Replay: probes and checkpoints never change a trajectory (the randomness contract)."""

import json

import numpy as np
import pytest

import _protocol_oracle as oracle
from sparsegf2 import graphs, mipt
from sparsegf2.clifford import Clifford


def _spec(**kw):
    n = 16
    base = dict(
        num_system=n,
        edges=[graphs.watts_strogatz(n, 4, 0.2, seed=(0, g)) for g in range(3)],
        circuits_per_graph=2,
        layers=24,
        measurement_rate=0.15,
        schedule="random_edges",
        reference="purification",
        checkpoints="final",
        probes=[mipt.code_snapshot()],
    )
    base.update(kw)
    return mipt.MonitoredCircuit(**base)


def test_contract_constant():
    assert mipt.RANDOMNESS_CONTRACT == 2


@pytest.mark.parametrize("t", [0, 3, 5])
def test_replay_with_new_probes_reproduces_the_code_bit_for_bit(t):
    spec = _spec()
    full = spec.run(seed=9)
    again = spec.replay(
        t,
        seed=9,
        checkpoints=8,
        probes=[
            mipt.code_dimension(),
            mipt.erasure_first_failure(20, checkpoints=[8, 16]),
            mipt.code_snapshot(checkpoints="final"),
        ],
    )
    assert again.num_trajectories == 1 and int(again.trajectories[0]) == t
    np.testing.assert_array_equal(
        full.code(t, 0).stabilizer_matrix(), again.code(0, 0).stabilizer_matrix()
    )
    assert again.values["code_dimension"][0, -1] == full.code(t, 0).num_logical


def test_replay_validates_the_index():
    with pytest.raises(ValueError):
        _spec().replay(6, seed=1)  # 3 graphs x 2 circuits = trajectories 0..5


# --- sampled probes: one stream per evaluation (finding N4) ----------------------------

SAMPLED = ("erasure_first_failure", "erasure_lost_profile")
SAMPLES = {"erasure_first_failure": 8, "erasure_lost_profile": 4}
TRAJECTORIES = 4


def _sampled_spec(**kw):
    # the spec of the N4 repro (review/scripts/ensemble/n_replay_samples.py)
    base = dict(
        num_system=16,
        edges=graphs.cycle(16),
        layers=12,
        measurement_rate=0.05,
        reference="purification",
        checkpoints=[4, 12],
        probes=[mipt.code_dimension(), mipt.erasure_first_failure(samples=8)],
    )
    base.update(kw)
    return mipt.MonitoredCircuit(**base)


def _full_spec():
    """Both sampled probes behind another sampled probe, among unsampled ones."""
    return _sampled_spec(
        probes=[
            mipt.code_dimension(),
            mipt.half_cut_entropy(),
            mipt.erasure_lost_profile(3, name="another"),
            mipt.erasure_first_failure(SAMPLES["erasure_first_failure"]),
            mipt.erasure_lost_profile(SAMPLES["erasure_lost_profile"]),
        ]
    )


def _probe(kind, **kw):
    return getattr(mipt, kind)(SAMPLES[kind], **kw)


@pytest.fixture(scope="module")
def full_run():
    return _full_spec().run(TRAJECTORIES, seed=3)


def test_every_evaluation_draws(full_run):
    # k > 0 everywhere, so no evaluation is the k = 0 fill (which draws nothing)
    assert (full_run.values["code_dimension"] > 0).all()
    assert len(np.unique(full_run.values["erasure_first_failure"])) > 2
    assert len(np.unique(full_run.values["erasure_lost_profile"][:, :, 1:-1])) > 2


@pytest.mark.parametrize("t", range(TRAJECTORIES))
@pytest.mark.parametrize("kind", SAMPLED)
def test_replay_reproduces_sampled_probes_with_other_probes_and_checkpoints(full_run, kind, t):
    spec = _full_spec()
    layers = list(full_run.layers(kind))
    assert layers == [4, 12]
    for c, layer in enumerate(layers):
        one = spec.replay(t, seed=3, probes=[_probe(kind)], checkpoints=[layer])
        np.testing.assert_array_equal(one.values[kind][0, 0], full_run.values[kind][t, c])
    final = spec.replay(t, seed=3, probes=[_probe(kind)], checkpoints="final")
    np.testing.assert_array_equal(final.values[kind][0, 0], full_run.values[kind][t, 1])
    # on its own schedule, behind other probes, with the spec checking every layer
    own = spec.replay(
        t,
        seed=3,
        probes=[mipt.code_dimension(), _probe(kind, checkpoints=[1, 12])],
        checkpoints="every_layer",
    )
    np.testing.assert_array_equal(own.values[kind][0, 1], full_run.values[kind][t, 1])


@pytest.mark.parametrize("kind", SAMPLED)
def test_a_second_sampled_probe_does_not_change_the_first(kind):
    alone = _sampled_spec(probes=[_probe(kind)]).run(TRAJECTORIES, seed=3)
    others = [
        mipt.erasure_first_failure(5, name="extra_first_failure"),
        mipt.erasure_lost_profile(2, name="extra_lost_profile"),
    ]
    for probes in ([*others, _probe(kind)], [_probe(kind), *others]):
        both = _sampled_spec(probes=probes).run(TRAJECTORIES, seed=3)
        np.testing.assert_array_equal(both.values[kind], alone.values[kind])


def test_probe_names_key_the_streams():
    twins = _sampled_spec(
        probes=[mipt.erasure_first_failure(8), mipt.erasure_first_failure(8, name="twin")]
    ).run(TRAJECTORIES, seed=3)
    assert not np.array_equal(twins.values["erasure_first_failure"], twins.values["twin"])
    # one stream per evaluation, read in order: fewer samples are a prefix of more
    fewer = _sampled_spec(probes=[mipt.erasure_first_failure(3)]).run(TRAJECTORIES, seed=3)
    np.testing.assert_array_equal(
        fewer.values["erasure_first_failure"], twins.values["erasure_first_failure"][:, :, :3]
    )


def test_probe_streams_follow_the_documented_derivation():
    # A code that never changes: one reference Bell-paired with qubit 5, identity gates,
    # no measurements. Erasing a set loses 2 bits if it contains qubit 5 and nothing
    # otherwise, so the first-failure size of an ordering is 1 + the position of qubit 5,
    # and the lost-information profile is 2 x the fraction of orderings whose first m
    # qubits contain it. The orderings come from the oracle's transcription.
    n, partner = 12, 5
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        layers=4,
        measurement_rate=0.0,
        gate_set=[Clifford.identity(2).index()],
        reference="single",
        reference_partner=partner,
        checkpoints="every_layer",
        probes=[
            mipt.erasure_first_failure(6),
            mipt.erasure_lost_profile(3, checkpoints=[1, 3]),
            mipt.erasure_first_failure(4, name="ff/other:näme"),
        ],
    )
    seed, first, count = (4, 2), 5, 3
    result = spec.run(count, seed=seed, first_trajectory=first)
    assert len(np.unique(result.values["erasure_first_failure"])) > 3  # orderings vary
    for row in range(count):
        t = first + row
        for p in spec.probes:  # each with its own name, schedule and number of samples
            for c, layer in enumerate(result.layers(p.name)):
                orders = oracle.probe_orderings(seed, t, p.name, int(layer), n, p.samples)
                if p.kind == "erasure_first_failure":
                    want = [o.index(partner) + 1 for o in orders]
                else:
                    want = [
                        2 * sum(partner in o[:m] for o in orders) / p.samples for m in range(n + 1)
                    ]
                got = result.values[p.name][row, c]
                np.testing.assert_array_equal(got, want, err_msg=f"{p.name} t={t} L={layer}")


# --- the contract travels with results ---------------------------------------------------


def _drop_contract(src, dst):
    """Copy a saved result without its randomness_contract (a file from before the field)."""
    with np.load(src, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["__meta__"]))
    del meta["randomness_contract"]
    arrays["__meta__"] = np.array(json.dumps(meta))
    np.savez_compressed(dst, **arrays)


def test_results_carry_the_contract_and_never_mix_contracts(tmp_path):
    spec = _sampled_spec()
    a = spec.run(2, seed=3)
    assert a.randomness_contract == mipt.RANDOMNESS_CONTRACT
    assert spec.replay(1, seed=3).randomness_contract == mipt.RANDOMNESS_CONTRACT
    path = tmp_path / "a.npz"
    a.save(path)
    assert mipt.EnsembleResult.load(path).randomness_contract == mipt.RANDOMNESS_CONTRACT
    old_a = tmp_path / "old_a.npz"
    _drop_contract(path, old_a)
    assert mipt.EnsembleResult.load(old_a).randomness_contract == 1
    b = spec.run(2, seed=3, first_trajectory=2)
    merged = mipt.EnsembleResult.concatenate([a, b])
    assert merged.randomness_contract == mipt.RANDOMNESS_CONTRACT
    b.save(tmp_path / "b.npz")
    old_b = tmp_path / "old_b.npz"
    _drop_contract(tmp_path / "b.npz", old_b)
    # the same protocol, seed and version, other trajectories: only the contract differs
    for parts in ([mipt.EnsembleResult.load(old_a), b], [a, mipt.EnsembleResult.load(old_b)]):
        with pytest.raises(ValueError, match="randomness contract"):
            mipt.EnsembleResult.concatenate(parts)
    old = mipt.EnsembleResult.concatenate(
        [mipt.EnsembleResult.load(old_a), mipt.EnsembleResult.load(old_b)]
    )
    assert old.randomness_contract == 1  # one contract throughout: merged as before
