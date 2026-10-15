"""Trajectory t depends only on (spec, seed, t): chunks, threads and replay agree exactly.

One spec switches on every feature that touches the per-trajectory streams: a
collection of graphs of different sizes, several circuits per graph, a random
single-reference partner, stop_when_purified, per-probe schedules and every
probe kind. A chunk that forgets the ``first_trajectory`` offset anywhere (graph
index, circuit stream, probe stream) changes some array here.
"""

import tempfile
from pathlib import Path

import numba
import numpy as np
import pytest

from sparsegf2 import graphs, mipt

N = 20
SEED = 5
GRAPHS = 4
PER_GRAPH = 3
TRAJECTORIES = GRAPHS * PER_GRAPH


def _bank(count=GRAPHS):
    return [graphs.newman_watts(N, 4, 0.3 + 0.2 * i, seed=(11, i)) for i in range(count)]


def _probes():
    return [
        mipt.half_cut_entropy(),
        mipt.entropy_profile(checkpoints=4),
        mipt.code_dimension(),
        mipt.lost_information([0, 1, 2], "lost_information"),
        mipt.contiguous_distance(),
        mipt.erasure_first_failure(6),
        mipt.erasure_lost_profile(3),
        mipt.uncorrectable_region_counts(2),
        mipt.code_snapshot(checkpoints="final"),
    ]


def _spec(edges=None, **overrides):
    kw = dict(
        num_system=N,
        edges=_bank() if edges is None else edges,
        circuits_per_graph=PER_GRAPH,
        layers=12,
        measurement_rate=0.25,
        schedule="random_edges",
        reference="single",
        reference_partner="random",
        checkpoints=2,
        probes=_probes(),
        stop_when_purified=True,
    )
    kw.update(overrides)
    return mipt.MonitoredCircuit(**kw)


@pytest.fixture(scope="module")
def spec():
    return _spec()


@pytest.fixture(scope="module")
def whole(spec):
    return spec.run(TRAJECTORIES, seed=SEED)


def assert_same(a, b):
    """Every array of two results is equal (NaN equals NaN)."""
    assert list(a.values) == list(b.values)
    for name in a.values:
        np.testing.assert_array_equal(a.values[name], b.values[name], err_msg=name)
    for field in ("purification_layer", "final_layer", "num_measurements", "trajectories"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field), err_msg=field)
    assert list(a.snapshots) == list(b.snapshots)
    for name in a.snapshots:
        for x, y in zip(a.snapshots[name], b.snapshots[name], strict=True):
            np.testing.assert_array_equal(x, y, err_msg=name)


def test_spec_is_nontrivial(spec, whole):
    sizes = {len(g) for g in _bank()}
    assert len(sizes) == GRAPHS  # different edge counts
    assert whole.num_trajectories == TRAJECTORIES
    tau = whole.purification_layer
    assert (tau >= 0).any() and (tau < 0).any()  # some, but not all, purified
    assert (whole.final_layer < spec.layers).any()  # stop_when_purified acted
    # the partner is random, so trajectories differ in more than the graph
    assert len(np.unique(whole.values["lost_information"], axis=0)) > 1
    # the per-probe schedules differ from the spec's
    assert list(whole.layers("entropy_profile")) == [0, 4, 8, 12]
    assert list(whole.layers("code_snapshot")) == [12]
    assert list(whole.layers("code_dimension")) == [0, 2, 4, 6, 8, 10, 12]
    # a code survives somewhere and is saved
    counts = whole.snapshots["code_snapshot"][1]
    assert (counts >= 0).any() and (counts < 0).any()


def test_chunks_equal_one_run(spec, whole):
    chunks = [
        spec.run(1, seed=SEED),
        spec.run(6, seed=SEED, first_trajectory=1),
        spec.run(4, seed=SEED, first_trajectory=7),
        spec.run(1, seed=SEED, first_trajectory=11),
    ]
    assert_same(whole, mipt.EnsembleResult.concatenate(chunks))


def test_thread_count_does_not_matter(spec, whole):
    threads = min(4, numba.config.NUMBA_NUM_THREADS)
    one = spec.run(TRAJECTORIES, seed=SEED, num_threads=1)
    many = spec.run(TRAJECTORIES, seed=SEED, num_threads=threads)
    assert_same(one, many)
    assert_same(one, whole)


@pytest.mark.parametrize("t", [0, 5, 11])
def test_replay_equals_its_row(spec, whole, t):
    one = spec.replay(t, seed=SEED, probes=_probes(), checkpoints=2)
    row = mipt.EnsembleResult.concatenate([spec.run(1, seed=SEED, first_trajectory=t)])
    assert_same(one, row)
    for name, v in whole.values.items():
        np.testing.assert_array_equal(one.values[name][0], v[t], err_msg=name)
    for field in ("purification_layer", "final_layer", "num_measurements"):
        assert getattr(one, field)[0] == getattr(whole, field)[t], field
    for name, (gens, counts) in whole.snapshots.items():
        np.testing.assert_array_equal(one.snapshots[name][0][0], gens[t])
        np.testing.assert_array_equal(one.snapshots[name][1][0], counts[t])


def test_replay_with_default_arguments(spec, whole):
    one = spec.replay(5, seed=SEED)
    np.testing.assert_array_equal(
        one.values["code_dimension"][0], whole.values["code_dimension"][5]
    )


# --- collections (DESIGN 8.1) -------------------------------------------------


def test_collection_of_one_graph_equals_that_graph_alone():
    g = _bank(1)[0]
    kw = dict(probes=[mipt.code_dimension(), mipt.erasure_first_failure(4)], checkpoints=2)
    in_list = _spec([g], **kw).run(seed=SEED)
    alone = _spec(g, circuits_per_graph=1, **kw).run(PER_GRAPH, seed=SEED)
    assert in_list.num_trajectories == PER_GRAPH
    for name in in_list.values:
        np.testing.assert_array_equal(in_list.values[name], alone.values[name], err_msg=name)
    np.testing.assert_array_equal(in_list.purification_layer, alone.purification_layer)
    np.testing.assert_array_equal(in_list.final_layer, alone.final_layer)


@pytest.mark.parametrize("schedule", ["matching", "random_edges", "all_edges"])
def test_appending_graphs_keeps_earlier_trajectories(schedule):
    bank = _bank()
    kw = dict(schedule=schedule, checkpoints="every_layer")
    small = _spec(bank[:2], **kw).run(seed=SEED)
    big = _spec(bank, **kw).run(seed=SEED)
    rows = small.num_trajectories
    assert rows == 2 * PER_GRAPH and big.num_trajectories == TRAJECTORIES
    for name in small.values:
        np.testing.assert_array_equal(small.values[name], big.values[name][:rows], err_msg=name)
    for field in ("purification_layer", "final_layer", "num_measurements"):
        np.testing.assert_array_equal(getattr(small, field), getattr(big, field)[:rows])
    gens, counts = small.snapshots["code_snapshot"]
    np.testing.assert_array_equal(gens, big.snapshots["code_snapshot"][0][:rows])
    np.testing.assert_array_equal(counts, big.snapshots["code_snapshot"][1][:rows])


# --- edge cases ---------------------------------------------------------------


def _cycle_spec(**kw):
    base = dict(num_system=6, edges=graphs.cycle(6), measurement_rate=0.5)
    base.update(kw)
    return mipt.MonitoredCircuit(**base)


def _round_trip(result):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "r.npz"
        result.save(path)
        back = mipt.EnsembleResult.load(path)
    assert_same(result, back)
    return back


def test_zero_trajectory_run_saves_loads_and_concatenates():
    spec = _cycle_spec(
        reference="single",
        layers=6,
        checkpoints=2,
        probes=[mipt.code_dimension(), mipt.erasure_first_failure(3), mipt.code_snapshot()],
        stop_when_purified=True,
    )
    empty = spec.run(0, seed=1)
    assert empty.num_trajectories == 0
    back = _round_trip(empty)
    assert back.num_trajectories == 0
    full = spec.run(5, seed=1)
    merged = mipt.EnsembleResult.concatenate([back, full])
    assert_same(merged, full)
    assert_same(mipt.EnsembleResult.concatenate([full, back]), full)


def test_layers_zero_with_stop_when_purified():
    spec = _cycle_spec(
        reference="purification",
        layers=0,
        stop_when_purified=True,
        probes=[mipt.code_dimension(), mipt.code_snapshot()],
    )
    result = spec.run(5, seed=1)
    assert list(spec.checkpoint_layers) == [0]
    np.testing.assert_array_equal(result.final_layer, 0)
    np.testing.assert_array_equal(result.values["code_dimension"], 6)  # nothing purified yet
    _round_trip(result)
    split = mipt.EnsembleResult.concatenate(
        [spec.run(2, seed=1), spec.run(3, seed=1, first_trajectory=2)]
    )
    assert_same(split, result)


def test_checkpoint_stride_larger_than_layers():
    spec = _cycle_spec(
        reference="single",
        layers=5,
        checkpoints=100,
        probes=[mipt.code_dimension(), mipt.entropy_profile(checkpoints=7)],
    )
    assert list(spec.checkpoint_layers) == [0, 5]
    result = spec.run(5, seed=1)
    assert result.values["code_dimension"].shape == (5, 2)
    assert (result.purification_layer[result.purification_layer >= 0] <= 5).all()
    _round_trip(result)
    assert_same(
        mipt.EnsembleResult.concatenate(
            [spec.run(2, seed=1), spec.run(3, seed=1, first_trajectory=2)]
        ),
        result,
    )


def test_probe_schedule_disjoint_from_specs_under_stop_when_purified():
    spec = _cycle_spec(
        reference="single",
        layers=8,
        checkpoints=[0, 8],
        probes=[
            mipt.code_dimension(checkpoints=[3, 5]),
            mipt.erasure_first_failure(3, checkpoints=[1, 7]),
            mipt.half_cut_entropy(checkpoints=[2, 4]),
        ],
        stop_when_purified=True,
    )
    result = spec.run(20, seed=3)
    assert list(result.layers("code_dimension")) == [3, 5]
    assert list(result.layers("half_cut_entropy")) == [2, 4]
    tau = result.purification_layer
    # purification is only tested at the spec's checkpoints: 0 or 8, or never
    assert set(tau.tolist()) <= {-1, 0, 8}
    for t in range(result.num_trajectories):
        np.testing.assert_array_equal(
            spec.replay(t, seed=3).values["code_dimension"][0], result.values["code_dimension"][t]
        )
    _round_trip(result)
    assert_same(
        mipt.EnsembleResult.concatenate(
            [spec.run(7, seed=3), spec.run(13, seed=3, first_trajectory=7)]
        ),
        result,
    )
