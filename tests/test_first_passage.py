"""Exact online first passages, ``EnsembleResult.first_passage``.

``tau_ell = min{t : k_t <= ell}`` (v1.4 section 5.2) is read by the engine after every layer
whatever the checkpoints. Here it is compared with the minimum over a dense every-layer
``code_dimension`` trace of the same trajectories, at a dense grid of rates from 0 to 1, odd
and even n, beta 0 and > 0, both reference pictures, with and without ``stop_when_purified``
and at depths up to 10^5; censored entries (-1) must sit exactly where the dense trace never
reaches the threshold. Stim checks the same quantities at depth 10^4 in ``test_sweep.py``.
Every test runs at fixed seeds and asserts nothing about time.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from sparsegf2 import analysis, api, graphs, mipt

#: 33 rates from 0 to 1: 0, then 32 log-spaced from 10^-4 (crossings thousands of layers
#: deep at these sizes) to 1 (every qubit measured every layer).
RATES = [0.0, *np.geomspace(1e-4, 1.0, 32).tolist()]


def dense_first_passage(k: np.ndarray, n: int) -> np.ndarray:
    """``tau_ell`` from a dense trace ``k[t, L]`` (layers 0, 1, ...), by definition: the first
    column with ``k <= ell``, or -1."""
    out = np.full((k.shape[0], n + 1), -1, dtype=np.int64)
    for ell in range(n + 1):
        hit = k <= ell
        reached = hit.any(axis=1)
        out[reached, ell] = hit[reached].argmax(axis=1)
    return out


def protocol(n, beta, picture, layers, p, *, checkpoints, probes, stop=False, graph_seed=3):
    collection = [graphs.watts_strogatz(n, 4, beta, seed=(graph_seed, g)) for g in range(2)]
    partner = {"reference_partner": "random"} if picture == "single" else {}
    return mipt.MonitoredCircuit(
        num_system=n,
        edges=collection,
        circuits_per_graph=3,
        layers=layers,
        measurement_rate=p,
        schedule="random_edges",
        gates_per_layer=n // 2,
        random_orientation=False,
        reference=picture,
        **partner,
        checkpoints=checkpoints,
        probes=probes,
        stop_when_purified=stop,
    )


def check_structure(first: np.ndarray, k0: np.ndarray, n: int) -> None:
    """tau_ell is nonincreasing in ell (censored entries, -1, read as infinity), tau_n = 0 and
    tau_ell = 0 exactly for ell >= k_0."""
    as_inf = np.where(first < 0, np.iinfo(np.int64).max, first)
    assert np.all(np.diff(as_inf, axis=1) <= 0)
    assert np.all(first[:, n] == 0)
    ell = np.arange(n + 1)[None, :]
    assert np.array_equal(first == 0, ell >= k0[:, None])


CASES = [  # n, beta, picture, stop_when_purified
    (7, 0.0, "purification", False),
    (8, 0.2, "purification", True),
    (9, 0.1, "single", False),
    (6, 0.3, "single", True),
]


@pytest.mark.parametrize(("n", "beta", "picture", "stop"), CASES)
def test_first_passages_are_the_minimum_over_the_dense_trace(n, beta, picture, stop):
    layers = 10_000
    sparse = sorted({0, 1, 3, 10, 30, 100, 300, 1000, 3000, layers})
    seen_censored = seen_deep = 0
    for p in RATES:
        dense = protocol(n, beta, picture, layers, p, checkpoints="every_layer",
                         probes=[mipt.code_dimension()]).run(seed=[5, n])  # fmt: skip
        k = dense.values["code_dimension"]
        assert np.all(np.diff(k, axis=1) <= 0)  # k never increases, layer by layer
        expected = dense_first_passage(k, n)
        assert dense.first_passage is not None
        assert np.array_equal(dense.first_passage, expected)
        check_structure(dense.first_passage, k[:, 0].astype(np.int64), n)
        # censored exactly where the trace never reaches the threshold, through final_layer
        assert np.array_equal(dense.first_passage < 0, k.min(axis=1)[:, None] > np.arange(n + 1))
        assert np.all(dense.final_layer == layers)
        # and they hold the whole trace: k_t = #{ell < n : tau_ell > t or censored}
        t = np.arange(layers + 1)
        rebuilt = ((expected[:, :-1, None] < 0) | (expected[:, :-1, None] > t)).sum(axis=1)
        assert np.array_equal(rebuilt, k)
        # tau_0 is the exact purification layer the every-layer checkpoints detect
        assert np.array_equal(dense.first_passage[:, 0], dense.purification_layer)
        # the same trajectories on sparse checkpoints, stopping or not: the same first passages
        runs = [
            protocol(n, beta, picture, layers, p, checkpoints=sparse,
                     probes=[mipt.code_dimension()], stop=stop),
            protocol(n, beta, picture, layers, p, checkpoints="final", probes=[], stop=stop),
        ]  # fmt: skip
        if stop:  # stopping at the exact purification layer, every layer a checkpoint
            runs.append(
                protocol(n, beta, picture, layers, p, checkpoints="every_layer",
                         probes=[mipt.code_dimension()], stop=True)
            )  # fmt: skip
        for spec in runs:
            other = spec.run(seed=[5, n])
            assert np.array_equal(other.first_passage, expected)
            if stop:  # every threshold was reached by the stop: none is censored early
                stopped = other.final_layer < layers
                assert np.all(other.first_passage[stopped] >= 0)
                assert np.all(other.first_passage[stopped, 0] <= other.final_layer[stopped])
        seen_censored += int(np.count_nonzero(expected < 0))
        seen_deep += int(np.count_nonzero(expected > 1000))
    assert seen_censored > 0 and seen_deep > 0  # the grid reached both regimes


def test_first_passages_at_depth_one_hundred_thousand_and_their_replay():
    n, layers = 6, 100_000
    for p in (0.0, 1e-5, 3e-5, 1e-4):
        spec = protocol(n, 0.1, "purification", layers, p, checkpoints="every_layer",
                        probes=[mipt.code_dimension(checkpoints="every_layer")])  # fmt: skip
        res = spec.run(seed=[9, 1])
        k = res.values["code_dimension"]
        assert np.all(np.diff(k, axis=1) <= 0)
        assert np.array_equal(res.first_passage, dense_first_passage(k, n))
        check_structure(res.first_passage, k[:, 0].astype(np.int64), n)
        if p == 1e-4:
            assert np.any(res.first_passage[:, :n] > 10_000)  # crossings deep in the run
        # a replay at this depth, observed on sparse checkpoints, gives the same row
        u = int(res.trajectories[-1])
        again = res.replay(u, checkpoints=[0, 10, 1000, layers],
                           probes=[mipt.code_dimension(checkpoints=[7, 70_000])])  # fmt: skip
        assert np.array_equal(again.first_passage[0], res.first_passage[-1])
        assert again.values["code_dimension"][0].tolist() == [k[-1, 7], k[-1, 70_000]]
        same = res.replay(u)
        assert np.array_equal(same.values["code_dimension"][0], k[-1])
        assert np.array_equal(same.first_passage[0], res.first_passage[-1])


def test_first_passages_are_saved_loaded_concatenated_and_censored(tmp_path):
    n, layers = 7, 400
    spec = protocol(n, 0.2, "purification", layers, 0.004, checkpoints=[0, 50, layers],
                    probes=[mipt.code_dimension()])  # fmt: skip
    whole = spec.run(seed=4)
    parts = [spec.run(3, seed=4, first_trajectory=f) for f in (0, 3)]
    joined = mipt.EnsembleResult.concatenate(parts)
    assert np.array_equal(joined.first_passage, whole.first_passage)
    whole.save(tmp_path / "r.npz", compression="fast")
    loaded = mipt.EnsembleResult.load(tmp_path / "r.npz")
    assert loaded.first_passage.dtype == np.int64
    assert np.array_equal(loaded.first_passage, whole.first_passage)
    # (time, observed): censored entries take the censoring depth, the last layer simulated
    time, observed = whole.first_passage_times()
    assert np.array_equal(observed, whole.first_passage >= 0) and not observed.all()
    censor = np.broadcast_to(whole.final_layer[:, None], time.shape)
    assert np.array_equal(time[~observed], censor[~observed]) and np.all(censor == layers)
    km = analysis.kaplan_meier(time[:, 0], observed[:, 0])
    assert km["survival"].size == np.unique(time[observed[:, 0], 0]).size
    # a part saved before first passages were recorded has none, and a merge with it neither
    parts[1].first_passage = None
    parts[1].save(tmp_path / "old.npz")
    old = mipt.EnsembleResult.load(tmp_path / "old.npz")
    assert old.first_passage is None
    assert mipt.EnsembleResult.concatenate([parts[0], old]).first_passage is None
    with pytest.raises(ValueError, match="no first passages"):
        old.first_passage_times()
    # the API keeps them with a selection of rows and lists them in the layout
    kept = api._rows(whole, np.arange(whole.num_trajectories) % 2 == 0)
    assert np.array_equal(kept.first_passage, whole.first_passage[::2])
    layout = api._result_layout(spec, 6)["arrays"]
    row = next(r for r in layout if r["name"] == "first_passage")
    assert row["shape"] == [6, n + 1] and row["dtype"] == "int64"


def test_without_a_reference_there_are_no_first_passages():
    spec = mipt.MonitoredCircuit(num_system=6, edges=graphs.cycle(6), layers=10,
                                 measurement_rate=0.2, reference="none",
                                 probes=[mipt.half_cut_entropy()])  # fmt: skip
    res = spec.run(2, seed=1)
    assert res.first_passage is None
    assert all(r["name"] != "first_passage" for r in api._result_layout(spec, 2)["arrays"])
    with pytest.raises(ValueError, match="no first passages"):
        res.first_passage_times()


# ---------------------------------------------------------------------------
# every existing output is unchanged
# ---------------------------------------------------------------------------

#: Digests of these runs made by the code before first passages were recorded (commit
#: 374608c): every probe value, snapshot word and count, purification and final layer and
#: measurement count. Reading k after every layer draws nothing, so none may change.
UNCHANGED = {
    "purification_ws9_dense": "9406be805eb17a84720aaf296b1cf209ac439c6e78b4857e0cb086880f0f7228",
    "purification_ws9_sparse_stop": "b42b2264837a7c4098b47a7c59d961212bf2fa32a2f7fab57e9d9a0af37b6208",
    "single_ring8_stop": "c5ccb9638e97753cefa5f2946725d9edb4c0d37161269add37d8d366ee456ca2",
    "none_path6": "294002aad6a8ab96f57c3909449e73514d61e33bb4c29d441dd0df127ba8bc74",
    "matching_scramble_cycle8": "637ab8818402ae20a73abd761c6c987975d69a89bfb0ae61aad3036422d61759",
    "single_fixed_stop_sparse": "5e94f40f8f73163d38a9d8877ff62bc37671f340aa5eaef83328a68721038a0e",
}


def _digest(res: mipt.EnsembleResult) -> str:
    h = hashlib.sha256()
    for name in sorted(res.values):
        h.update(name.encode())
        h.update(np.ascontiguousarray(res.values[name]).astype("<f8").tobytes())
    for name in sorted(res.snapshots):
        words, counts = res.snapshots[name]
        h.update(name.encode())
        h.update(np.ascontiguousarray(words).astype("<u8").tobytes())
        h.update(np.ascontiguousarray(counts).astype("<i8").tobytes())
    if res.purification_layer is not None:
        h.update(np.ascontiguousarray(res.purification_layer).astype("<i8").tobytes())
    h.update(np.ascontiguousarray(res.final_layer).astype("<i8").tobytes())
    h.update(np.ascontiguousarray(res.num_measurements).astype("<i8").tobytes())
    return h.hexdigest()


def _unchanged_cases():
    ws9 = [graphs.watts_strogatz(9, 4, 0.1, seed=(3, g)) for g in range(3)]
    common = dict(num_system=9, edges=ws9, circuits_per_graph=4, layers=300,
                  schedule="random_edges", gates_per_layer=4, random_orientation=False,
                  reference="purification")  # fmt: skip
    yield "purification_ws9_dense", 12, mipt.MonitoredCircuit(
        **common, measurement_rate=0.12, checkpoints="every_layer",
        probes=[mipt.code_dimension(), mipt.half_cut_entropy(checkpoints=[0, 3, 30, 300]),
                mipt.contiguous_distance(checkpoints=[1, 10, 100, 300]),
                mipt.erasure_first_failure(samples=10, checkpoints=[5, 50, 300]),
                mipt.tableau_snapshot(checkpoints=[0, 7, 70, 300])],
    )  # fmt: skip
    yield "purification_ws9_sparse_stop", 12, mipt.MonitoredCircuit(
        **common, measurement_rate=0.2, checkpoints=[0, 1, 5, 50, 120, 300],
        stop_when_purified=True,
        probes=[mipt.code_dimension(), mipt.half_cut_entropy(),
                mipt.contiguous_distance(checkpoints=[1, 10, 100, 300]),
                mipt.erasure_first_failure(samples=6),
                mipt.tableau_snapshot(checkpoints=[0, 7, 70, 300])],
    )  # fmt: skip
    yield "single_ring8_stop", 10, mipt.MonitoredCircuit(
        num_system=8, edges=graphs.watts_strogatz(8, 4, 0.0, seed=1), layers=200,
        measurement_rate=0.3, schedule="random_edges", reference="single",
        reference_partner="random", checkpoints="every_layer", stop_when_purified=True,
        probes=[mipt.code_dimension(), mipt.code_snapshot(checkpoints=[0, 2, 20, 200])],
    )  # fmt: skip
    yield "none_path6", 5, mipt.MonitoredCircuit(
        num_system=6, edges=graphs.path(6), layers=40, measurement_rate=0.25,
        schedule="all_edges", reference="none", checkpoints=8,
        probes=[mipt.half_cut_entropy(), mipt.entropy_profile()],
    )  # fmt: skip
    yield "matching_scramble_cycle8", 7, mipt.MonitoredCircuit(
        num_system=8, edges=graphs.cycle(8), layers=64, measurement_rate=0.1,
        schedule="matching", reference="purification", scramble_gates=5, scramble_matchings=1,
        scramble_layers=2, checkpoints=[0, 4, 16, 64],
        probes=[mipt.code_dimension(), mipt.uncorrectable_region_counts(max_size=2),
                mipt.erasure_lost_profile(samples=3)],
    )  # fmt: skip
    yield "single_fixed_stop_sparse", 9, mipt.MonitoredCircuit(
        num_system=7, edges=graphs.cycle(7), layers=500, measurement_rate=0.05,
        schedule="random_edges", reference="single", reference_partner=3, scramble_gates=20,
        checkpoints=[0, 10, 100, 250, 500], stop_when_purified=True,
        probes=[mipt.code_dimension(), mipt.mutual_information([0, 1], [3, 4])],
    )  # fmt: skip


def test_every_existing_output_is_unchanged():
    got = {name: _digest(spec.run(count, seed=[41, 7])) for name, count, spec in _unchanged_cases()}
    assert got == UNCHANGED
