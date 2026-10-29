"""The distance stop, ``stop_when_distance`` / ``stop_not_before``: a trajectory ends at the first
layer L >= H by which its contiguous distance has reached c, and everything it recorded through L
is what the run without the rule records.

Each run here is compared with the same protocol without the rule (same seed): values, first
passages, purification times and snapshots up to the stop, NaN after it, the stop layer
max(H, T) (T: the first layer with d >= c), together with ``stop_when_purified``, chunks,
replay, save and load, the JSON API, the summaries and every validation error.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sparsegf2 import analysis, api, graphs, mipt

LAYERS = 72


def _spec(n=24, beta=1.0, p=0.1, *, reference="purification", rule=None, **changes):
    """A protocol with the stop's probe at every layer plus probes on other schedules."""
    probes = [
        mipt.code_dimension(checkpoints="every_layer"),
        mipt.contiguous_distance(cap=2, checkpoints="every_layer"),
        mipt.half_cut_entropy(),  # the spec's checkpoints
        mipt.erasure_first_failure(samples=12, checkpoints=[0, 10, 30, 50, LAYERS]),
        mipt.code_snapshot(checkpoints=[20, 40]),
    ]
    kw = dict(
        num_system=n,
        edges=[graphs.watts_strogatz(n, 4, beta, seed=(11, g)) for g in range(4)],
        circuits_per_graph=10,
        layers=LAYERS,
        measurement_rate=p,
        schedule="random_edges",
        reference=reference,
        checkpoints=6,  # purification is tested every 6 layers only
        probes=probes,
    )
    if reference == "single":
        kw["reference_partner"] = "random"
        kw["scramble_gates"] = 4 * n
    kw.update(changes)
    if rule is not None:
        kw["stop_when_distance"], kw["stop_not_before"] = rule
    return mipt.MonitoredCircuit(**kw)


def _with_rule(spec, at_least=2, not_before=0):
    return dataclasses.replace(spec, stop_when_distance=at_least, stop_not_before=not_before)


def _expected_stop(base, not_before, at_least=2):
    """The stop layer each row of ``base`` (a run without the rule) gets under the rule, or -1."""
    first = base.first_distance_layer(at_least, name="contiguous_distance[cap=2]")
    k = base.values["code_dimension"]  # every layer
    out = np.full(base.num_trajectories, -1, dtype=np.int64)
    for t in range(base.num_trajectories):
        if first[t] < 0:
            continue
        stop = max(not_before, int(first[t]))
        # still a code there (no purification first); a stop at the horizon itself is none
        if stop < base.spec.layers and stop <= base.final_layer[t] and k[t, stop] > 0:
            out[t] = stop
    return out


def _assert_same_until_the_stop(base, rule):
    """Every value of ``rule`` through its final layer equals ``base``'s, and is NaN after."""
    final = rule.final_layer
    assert np.all(final <= base.final_layer)
    for name, v in rule.values.items():
        layers = rule.layers(name)
        b = base.values[name]
        for t in range(rule.num_trajectories):
            known = layers <= final[t]
            np.testing.assert_array_equal(v[t, known], b[t, known])
            if rule.stopped_on_distance[t]:
                assert np.isnan(v[t, ~known]).all()
            else:
                np.testing.assert_array_equal(v[t], b[t])
    for t in range(rule.num_trajectories):  # first passages: censored at the stop
        fp = base.first_passage[t]
        np.testing.assert_array_equal(
            rule.first_passage[t], np.where((fp >= 0) & (fp <= final[t]), fp, -1)
        )
    stopped = rule.stopped_on_distance
    assert np.all(rule.purification_layer[stopped] == -1)
    np.testing.assert_array_equal(
        rule.purification_layer[~stopped], base.purification_layer[~stopped]
    )
    np.testing.assert_array_equal(rule.final_layer[~stopped], base.final_layer[~stopped])
    assert np.all(rule.num_measurements <= base.num_measurements)
    (gens, counts), (bgens, bcounts) = (
        rule.snapshots["code_snapshot"],
        base.snapshots["code_snapshot"],
    )
    snap_layers = rule.layers("code_snapshot")
    for t in range(rule.num_trajectories):
        known = snap_layers <= final[t]
        np.testing.assert_array_equal(counts[t, known], bcounts[t, known])
        np.testing.assert_array_equal(gens[t, known], bgens[t, known])
        if stopped[t]:
            assert np.all(counts[t, ~known] == -1)


CASES = [  # n, beta, p, reference, not_before, stop_when_purified
    (16, 0.0, 0.05, "purification", 0, False),
    (24, 1.0, 0.1, "purification", 0, False),
    (24, 1.0, 0.1, "purification", 30, False),
    (32, 0.0, 0.15, "purification", 12, True),
    (32, 1.0, 0.2, "purification", 50, True),
    (20, 1.0, 0.05, "single", 0, False),
    (20, 0.0, 0.1, "single", 25, True),
]


@pytest.mark.parametrize("n, beta, p, reference, not_before, stop", CASES)
def test_identical_to_the_run_without_the_rule_up_to_the_stop(
    n, beta, p, reference, not_before, stop
):
    spec = _spec(n, beta, p, reference=reference, stop_when_purified=stop)
    base = spec.run(seed=5)
    rule = _with_rule(spec, 2, not_before).run(seed=5)
    _assert_same_until_the_stop(base, rule)
    expected = _expected_stop(base, not_before)
    np.testing.assert_array_equal(rule.stopped_on_distance, expected >= 0)
    np.testing.assert_array_equal(rule.final_layer[expected >= 0], expected[expected >= 0])
    assert rule.stopped_on_distance.any()  # the cases exercise the rule
    if not stop:
        # what is left ran to the horizon: no distance >= 2, or purified before the stop
        assert np.all(rule.final_layer[expected < 0] == LAYERS)


def test_not_before_zero_stops_at_the_first_distance_of_two():
    spec = _spec()
    base = spec.run(seed=2)
    rule = _with_rule(spec).run(seed=2)
    first = base.first_distance_layer(2, name="contiguous_distance[cap=2]")
    stopped = rule.stopped_on_distance
    assert stopped.sum() > 20
    np.testing.assert_array_equal(rule.final_layer[stopped], first[stopped])
    np.testing.assert_array_equal(rule.first_distance_layer()[stopped], first[stopped])
    # the stop layer is where d first reached 2: d is below 2 at every earlier layer
    d = rule.values["contiguous_distance[cap=2]"]
    for t in np.flatnonzero(stopped):
        L = rule.final_layer[t]
        assert d[t, L] == 2 and not np.any(d[t, :L] >= 2)


def test_the_stop_waits_for_not_before_and_then_comes_at_once():
    spec = _spec()
    base = spec.run(seed=4)
    for H in (10, 40, LAYERS, LAYERS + 5):
        rule = _with_rule(spec, 2, H).run(seed=4)
        stopped = rule.stopped_on_distance
        assert np.all(rule.final_layer[stopped] >= H)
        np.testing.assert_array_equal(
            rule.final_layer[stopped],
            np.maximum(H, rule.first_distance_layer()[stopped]),
        )
        np.testing.assert_array_equal(stopped, _expected_stop(base, H) >= 0)
        if H > LAYERS:  # beyond the horizon: the rule never fires
            assert not stopped.any()
            np.testing.assert_array_equal(rule.final_layer, base.final_layer)


def test_a_full_distance_probe_and_larger_thresholds():
    spec = _spec(
        probes=[
            mipt.code_dimension(checkpoints="every_layer"),
            mipt.contiguous_distance(checkpoints="every_layer"),
        ]
    )
    base = spec.run(seed=3)
    for c in (2, 3):
        rule = _with_rule(spec, c, 5).run(seed=3)
        first = base.first_distance_layer(c)
        k = base.values["code_dimension"]
        for t in range(rule.num_trajectories):
            L = max(5, first[t])
            if first[t] >= 0 and k[t, L] > 0:
                assert rule.stopped_on_distance[t] and rule.final_layer[t] == L
            else:
                assert not rule.stopped_on_distance[t]
        np.testing.assert_array_equal(
            rule.values["contiguous_distance"][~rule.stopped_on_distance],
            base.values["contiguous_distance"][~rule.stopped_on_distance],
        )


def test_together_with_stop_when_purified_whichever_comes_first():
    spec = _spec(p=0.2, checkpoints="every_layer")  # purification tested at every layer
    base = spec.run(seed=6)  # neither rule
    both = dataclasses.replace(
        spec, stop_when_purified=True, stop_when_distance=2, stop_not_before=30
    ).run(seed=6)
    tau = base.purification_layer
    first = base.first_distance_layer(2, name="contiguous_distance[cap=2]")
    for t in range(base.num_trajectories):
        purify = tau[t] if tau[t] >= 0 else np.inf
        distance = max(30, first[t]) if first[t] >= 0 else np.inf
        expected = min(purify, distance, LAYERS)
        assert both.final_layer[t] == expected
        if purify <= distance and purify <= LAYERS:  # purification first: an observed time
            assert both.purification_layer[t] == tau[t] and not both.stopped_on_distance[t]
        elif distance < purify:
            assert both.stopped_on_distance[t] and both.purification_layer[t] == -1
    assert both.stopped_on_distance.any() and (both.purification_layer >= 0).any()
    # after a purification stop the code probes keep their exact fill; after a distance stop NaN
    k = both.values["code_dimension"]
    for t in range(both.num_trajectories):
        after = k[t, both.final_layer[t] + 1 :]
        if both.stopped_on_distance[t]:
            assert np.isnan(after).all()
        elif both.purification_layer[t] >= 0:
            assert np.all(after == 0)


def test_a_stop_at_layer_zero():
    # one reference after an all-to-all scramble: its code usually has distance >= 2 at once
    spec = _spec(20, 1.0, 0.1, reference="single", rule=(2, 0), scramble_gates=200)
    res = spec.run(seed=1)
    at_zero = res.stopped_on_distance & (res.final_layer == 0)
    assert at_zero.any()
    assert np.all(res.num_measurements[at_zero] == 0)
    assert np.all(res.first_passage[at_zero, 1:] == 0)  # k_0 = 1: tau_ell = 0 for ell >= 1
    assert np.all(res.first_passage[at_zero, 0] == -1)
    np.testing.assert_array_equal(res.censoring_layer[at_zero], 0)


def test_chunks_indices_and_threads_give_the_single_run():
    spec = _spec(rule=(2, 20))
    whole = spec.run(seed=8)
    parts = mipt.EnsembleResult.concatenate(
        [spec.run(15, seed=8), spec.run(25, seed=8, first_trajectory=15)]
    )
    shuffled = spec.run(indices=np.random.default_rng(0).permutation(40), seed=8, num_threads=1)
    order = np.argsort(shuffled.trajectories)
    for other, rows in ((parts, slice(None)), (shuffled, order)):
        np.testing.assert_array_equal(other.final_layer[rows], whole.final_layer)
        np.testing.assert_array_equal(other.purification_layer[rows], whole.purification_layer)
        np.testing.assert_array_equal(other.first_passage[rows], whole.first_passage)
        np.testing.assert_array_equal(other.stopped_on_distance[rows], whole.stopped_on_distance)
        for name in whole.values:
            np.testing.assert_array_equal(other.values[name][rows], whole.values[name])


def test_replay_stops_where_the_run_stopped(tmp_path):
    spec = _spec(p=0.2, rule=(2, 15))  # 36 of 40 rows stop on distance, 4 purify first
    res = spec.run(seed=9)
    path = tmp_path / "run.npz"
    res.save(path)
    back = mipt.EnsembleResult.load(path)
    assert back.spec.stop_when_distance == 2 and back.spec.stop_not_before == 15
    np.testing.assert_array_equal(back.stopped_on_distance, res.stopped_on_distance)
    for u in np.flatnonzero(res.stopped_on_distance)[:3].tolist() + [
        int(np.flatnonzero(~res.stopped_on_distance)[0])
    ]:
        again = back.replay(u)
        assert again.final_layer[0] == res.final_layer[u]
        for name in res.values:
            np.testing.assert_array_equal(again.values[name][0], res.values[name][u])
        # other probes and checkpoints: the stop's probe is kept (added, at every layer)
        other = back.replay(u, probes=[mipt.half_cut_entropy()], checkpoints="final")
        assert other.final_layer[0] == res.final_layer[u]
        assert list(other.values) == ["half_cut_entropy", "contiguous_distance[cap=2]"]
        np.testing.assert_array_equal(
            other.values["contiguous_distance[cap=2]"][0],
            res.values["contiguous_distance[cap=2]"][u],
        )
        # without the rule the same trajectory runs on to the horizon, identical up to the stop
        free = dataclasses.replace(back.spec, stop_when_distance=None, stop_not_before=0).replay(
            u, seed=back.seed_sequence
        )
        assert free.final_layer[0] == LAYERS
        L = res.final_layer[u]
        np.testing.assert_array_equal(
            free.values["code_dimension"][0, : L + 1], res.values["code_dimension"][u, : L + 1]
        )


def test_concatenate_refuses_another_rule():
    spec = _spec(rule=(2, 15))
    a = spec.run(10, seed=1)
    b = _with_rule(spec, 2, 16).run(10, seed=1, first_trajectory=10)
    with pytest.raises(ValueError, match="stop_not_before"):
        mipt.EnsembleResult.concatenate([a, b])
    c = dataclasses.replace(spec, stop_when_distance=None, stop_not_before=0)
    with pytest.raises(ValueError, match="stop_when_distance"):
        mipt.EnsembleResult.concatenate([a, c.run(10, seed=1, first_trajectory=10)])


def test_censoring_survival_and_first_passage_times_use_each_rows_stop():
    spec = _spec(rule=(2, 10), checkpoints=4)
    res = spec.run(seed=3)
    stopped = res.stopped_on_distance
    assert stopped.any()
    time, observed = res.purification_times()
    assert not observed[stopped].any()
    # censored at the last purification test (every 4 layers) not beyond the stop
    np.testing.assert_array_equal(time[stopped], (res.final_layer[stopped] // 4) * 4)
    fp_time, fp_observed = res.first_passage_times()
    assert np.all(
        fp_time[stopped & ~fp_observed[:, 0], 0] == res.final_layer[stopped & ~fp_observed[:, 0]]
    )
    analysis.kaplan_meier(time, observed)  # valid input
    layers, surviving = res.survival()
    earliest = res.final_layer[stopped].min()
    assert np.all(np.isnan(surviving[layers > earliest]))
    assert not np.isnan(surviving[layers <= earliest]).any()
    base = dataclasses.replace(spec, stop_when_distance=None, stop_not_before=0).run(seed=3)
    _, full = base.survival()
    np.testing.assert_array_equal(surviving[layers <= earliest], full[layers <= earliest])
    with pytest.raises(ValueError, match="cannot tell"):
        res.first_distance_layer(3)
    with pytest.raises(ValueError, match="give at_least"):
        base.first_distance_layer()


def test_snapshots_after_a_distance_stop_are_absent():
    res = _spec(rule=(2, 0)).run(seed=2)
    counts = res.snapshots["code_snapshot"][1]
    t = int(np.flatnonzero(res.stopped_on_distance & (res.final_layer < 20))[0])
    assert counts[t, 0] == -1
    with pytest.raises(ValueError, match="stop_when_distance"):
        res.code(t, 0)


def test_validation_errors_name_what_to_change():
    good = _spec()
    with pytest.raises(ValueError, match="at least 2"):
        _with_rule(good, 1)
    with pytest.raises(TypeError, match="stop_when_distance must be an integer"):
        _with_rule(good, 2.5)
    with pytest.raises(TypeError, match="stop_when_distance must be an integer"):
        _with_rule(good, True)
    with pytest.raises(ValueError, match="stop_not_before must be non-negative"):
        _with_rule(good, 2, -1)
    with pytest.raises(TypeError, match="stop_not_before must be an integer"):
        _with_rule(good, 2, 1.5)
    with pytest.raises(ValueError, match="stop_not_before applies only with stop_when_distance"):
        dataclasses.replace(good, stop_not_before=5)
    with pytest.raises(ValueError, match="needs a reference"):
        _spec(reference="none", probes=[], rule=(2, 0))
    k = mipt.code_dimension(checkpoints="every_layer")
    with pytest.raises(ValueError, match="needs a contiguous_distance probe.*every_layer"):
        _spec(probes=[k], rule=(2, 0))
    with pytest.raises(ValueError, match="exactly one contiguous_distance probe, got 2"):
        _spec(
            probes=[
                mipt.contiguous_distance(cap=2, checkpoints="every_layer"),
                mipt.contiguous_distance(checkpoints="every_layer"),
            ],
            rule=(2, 0),
        )
    with pytest.raises(ValueError, match="has cap 2, below stop_when_distance=3"):
        _spec(rule=(3, 0))
    with pytest.raises(ValueError, match="must be evaluated at every layer"):
        _spec(probes=[mipt.contiguous_distance(cap=2, checkpoints=2)], rule=(2, 0))
    with pytest.raises(ValueError, match="must be evaluated at every layer"):
        _spec(probes=[mipt.contiguous_distance(cap=2)], rule=(2, 0))  # the spec's stride 6
    # the spec's checkpoints serve when they are every layer; a cap above c is fine
    ok = _spec(probes=[mipt.contiguous_distance(cap=4)], checkpoints="every_layer", rule=(3, 0))
    assert ok.stop_probe.name == "contiguous_distance[cap=4]"
    assert _spec(rule=(2.0, 7.0)).stop_not_before == 7  # whole-number floats, as elsewhere
    assert good.stop_probe is None


JSON_SPEC = {
    "num_system": 24,
    "graph": {"family": "watts_strogatz", "k": 4, "beta": 1.0, "seed": 1, "realizations": 3},
    "circuits_per_graph": 10,
    "layers": 60,
    "measurement_rate": 0.1,
    "schedule": "random_edges",
    "reference": "purification",
    "checkpoints": 5,
    "probes": [
        {"kind": "code_dimension", "checkpoints": "every_layer"},
        {"kind": "contiguous_distance", "cap": 2, "checkpoints": "every_layer"},
        {"kind": "erasure_first_failure", "samples": 10, "checkpoints": [0, 20, 40, 60]},
        {"kind": "entropy_profile", "checkpoints": [0, 30, 60]},
    ],
    "stop_when_purified": True,
    "stop_when_distance": 2,
    "stop_not_before": 20,
}


def test_the_json_api_runs_and_summarizes_the_rule(tmp_path):
    out = api.run_ensemble(JSON_SPEC, seed=4, output_dir=str(tmp_path / "run"))
    res = mipt.EnsembleResult.load(tmp_path / "run" / "result.npz")
    stop = out["stop_when_distance"]
    assert stop["at_least"] == 2 and stop["not_before"] == 20
    assert stop["probe"] == "contiguous_distance[cap=2]"
    assert stop["stopped"] == int(res.stopped_on_distance.sum()) > 0
    assert stop["stop_layer_max"] == int(res.final_layer[res.stopped_on_distance].max())
    assert any(note.startswith("stop_when_distance:") for note in out["notes"])
    # the code dimension averages only the rows still running after the stops
    k = out["probes"]["code_dimension"]
    assert "count" in k and min(k["count"]) < res.num_trajectories
    survival = out["purification"]["survival"]["fraction_unpurified"]
    assert survival[0] == 1.0 and survival[-1] is None  # unknown after the first distance stop
    # the saved directory summarizes the same stops
    again = api.load_results(str(tmp_path / "run"))
    assert again["stop_when_distance"]["stopped"] == stop["stopped"]
    # erasure curves count only the rows that reached the layer (the others are unknown)
    curves = api.erasure_curves(str(tmp_path / "run"))["uniform"]["erasure_first_failure"]
    reached = ~np.isnan(res.values["erasure_first_failure"][:, -1, 0])
    assert curves["count"] == int(reached.sum()) < res.num_trajectories
    # the rule is part of the protocol: resuming the directory with another one is refused
    for change in ({"stop_not_before": 21}, {"stop_when_distance": None, "stop_not_before": 0}):
        with pytest.raises(ValueError, match="different spec"):
            api.start_ensemble_job(
                {**JSON_SPEC, **change}, output_dir=str(tmp_path / "run"), seed=4
            )
    assert api.capabilities("ensemble_spec")["ensemble_spec"]["stop_when_distance"] is None
    assert api.capabilities("ensemble_spec")["ensemble_spec"]["stop_not_before"] == 0


@pytest.mark.parametrize(
    "change, message",
    [
        ({"stop_when_distance": 1}, "stop_when_distance must be at least 2"),
        ({"stop_when_distance": "2"}, "stop_when_distance must be an integer or null"),
        ({"stop_not_before": -3}, "stop_not_before must be non-negative"),
        (
            {"stop_when_distance": None, "stop_not_before": 5},
            "applies only with stop_when_distance",
        ),
        ({"reference": "none", "stop_when_purified": False, "probes": []}, "needs a reference"),
        ({"probes": [{"kind": "code_dimension"}]}, "needs a contiguous_distance probe"),
        (
            {"probes": [{"kind": "contiguous_distance", "cap": 2}]},
            "must be evaluated at every layer",
        ),
        ({"stop_when_distance": 3}, "has cap 2, below stop_when_distance=3"),
        ({"stop_when_distance": 25}, "exceeds num_system=24"),  # d <= n: it could never stop
        ({"stop_when_distnace": 2}, "unknown key"),
    ],
)
def test_the_json_api_reports_every_bad_rule(change, message):
    with pytest.raises(ValueError, match=message):
        api.run_ensemble({**JSON_SPEC, **change}, trajectories=2, seed=1)


def test_a_stop_before_the_first_checkpoint_is_censored_at_the_stop():
    """With checkpoints="final" purification is tested at the horizon only, so a row the rule
    stops earlier was never tested; it still held a code at its stop, so its purification time
    is censored there, never at a layer it did not reach."""
    res = _spec(p=0.2, rule=(2, 40), checkpoints="final").run(seed=1)
    stopped = res.stopped_on_distance
    assert 0 < stopped.sum() < res.num_trajectories
    assert np.all(res.final_layer[stopped] < LAYERS)  # every stop comes before the one checkpoint
    time, observed = res.purification_times()
    np.testing.assert_array_equal(time[stopped], res.final_layer[stopped])
    assert not observed[stopped].any()
    assert np.all(time <= res.final_layer)
    np.testing.assert_array_equal(res.censoring_layer[~stopped], LAYERS)  # the one checkpoint
    # the same rows' first passages agree: k > 0 at the stop, so tau_0 > final_layer
    fp_time, fp_observed = res.first_passage_times()
    np.testing.assert_array_equal(fp_time[stopped, 0], time[stopped])
    assert not fp_observed[stopped, 0].any()
    survival = analysis.kaplan_meier(time, observed)
    assert survival is not None


def test_the_summaries_count_and_describe_only_the_rows_still_running(tmp_path):
    """After distance stops, a probe's mean is over the rows known at its layer: the
    correlated_erasure 'count' says how many, and no text claims all trajectories."""
    spec = {
        **JSON_SPEC,
        "stop_when_purified": False,
        "probes": [
            *JSON_SPEC["probes"],
            {"kind": "erasure_lost_profile", "samples": 4, "checkpoints": [0, 30, 60]},
            {
                "kind": "correlated_erasure",
                "couplings": [0, 1],
                "sizes": "1:5",
                "samples": 6,
                "checkpoints": [0, 30, 60],
            },
        ],
    }
    out = api.run_ensemble(spec, seed=4, output_dir=str(tmp_path / "run"))
    res = mipt.EnsembleResult.load(tmp_path / "run" / "result.npz")
    assert out["stop_when_distance"]["stopped"] > 0
    corr = out["probes"]["correlated_erasure"]
    e = [0, 30, 60].index(corr["at_layer"])  # the last evaluation where some row holds a code
    known = ~np.isnan(res.values["correlated_erasure"][:, e]).all(axis=1)
    assert e > 0 and corr["count"] == int(known.sum()) < res.num_trajectories
    assert corr["count_with_code"] <= corr["count"]
    texts = [p["what"] for p in out["probes"].values() if isinstance(p, dict) and "what" in p]
    assert not any("averaged over all trajectories" in t for t in texts + out["notes"])
    assert "still running" in out["probes"]["erasure_lost_profile"]["what"]
