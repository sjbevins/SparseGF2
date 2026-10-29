"""Untouched-qubit tracking, ``record_first_touch``: ``EnsembleResult.first_touch`` and the
helpers ``bare_counts`` (N_bare(T)) and ``bare_times`` (tau_bare).

``first_touch[t, q]`` must be the first layer in which system qubit q took part in a drawn gate
or was measured (0 for a scramble gate, -1 if never through ``final_layer``). It is compared
with the touches read from the circuit text of ``MonitoredCircuit.circuit(t)`` and with an
independent regeneration of the draws (``_protocol_oracle.oracle_draws``), for every schedule,
scramble, reference picture and rates 0 and 1; with stops it must be the run without the stop
truncated at ``final_layer``; chunks, indices, threads, replay, save and load, concatenate and
the JSON API (resume and extend included) must give the same rows; in the purification picture
every untouched qubit carries 2 bits of I(i:R), so N_bare(T) <= N_log(T) and N_bare(T) <= k(T)
on every layer. Runs with the option give bit-identical trajectories (test_logical_sites.py).
Past a stop, ``bare_counts`` and ``bare_times`` are checked row by row against the touches read
from the circuit text (NaN where a row stopped with bare qubits left, censoring at the stop), and
the summary's ``first_touch`` block against the saved arrays and a median taken by definition.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pytest

import _protocol_oracle as oracle
from sparsegf2 import analysis, api, graphs, mipt

_LINE = re.compile(r"^([A-Z_0-9]+)(?:\([^)]*\))?\s*(.*)$")


def touches_from_text(text: str, n: int) -> np.ndarray:
    """First touches read from the text of ``spec.circuit(t)``: the setup's H and CX lines do
    not count, gates before the first TICK are the scramble (layer 0), and layer L ends at the
    (L + 1)-th TICK."""
    out = np.full(n, -1, dtype=np.int64)
    layer = 0
    for line in text.splitlines():
        match = _LINE.match(line.strip())
        assert match, line
        name, targets = match.group(1), match.group(2).split()
        if name == "TICK":
            layer += 1
        elif name in ("CLIFFORD_2Q", "M"):
            for q in map(int, targets):
                assert 0 <= q < n, "gates and measurements act on the system only"
                if out[q] < 0:
                    out[q] = layer
        else:
            assert name in ("H", "CX") and layer == 0, f"unexpected line {line!r}"
    return out


def touches_from_draws(spec: mipt.MonitoredCircuit, t: int, seed) -> np.ndarray:
    """First touches from the protocol oracle's independent regeneration of the draws."""
    draws = oracle.oracle_draws(spec, t, seed)
    out = np.full(spec.num_system, -1, dtype=np.int64)

    def touch(qubits, layer):
        for q in qubits:
            if out[q] < 0:
                out[q] = layer

    for block in draws.scramble:
        touch([q for a, b, _ in block for q in (a, b)], 0)
    for layer, (gates, measured) in enumerate(zip(draws.gates, draws.measured, strict=True), 1):
        touch([q for a, b, _ in gates for q in (a, b)], layer)
        touch(measured, layer)
    return out


def _ws(n, beta, seed):
    return graphs.watts_strogatz(n, 4, beta, seed=seed)


CASES = [  # name, spec keywords
    ("random_edges_purification", dict(num_system=16, edges=_ws(16, 0.3, 1),
                                       measurement_rate=0.05)),
    ("random_edges_p0", dict(num_system=12, edges=_ws(12, 1.0, 2), measurement_rate=0.0)),
    ("random_edges_p1", dict(num_system=12, edges=_ws(12, 0.0, 3), measurement_rate=1.0)),
    ("one_gate_per_layer", dict(num_system=10, edges=_ws(10, 0.3, 4), measurement_rate=0.02,
                                gates_per_layer=1)),
    ("no_gates_no_measurements", dict(num_system=6, edges=graphs.cycle(6), measurement_rate=0.0,
                                      gates_per_layer=0)),
    ("weighted_edges", dict(num_system=10, edges=graphs.complete(10), measurement_rate=0.05,
                            edge_weights=graphs.power_law_weights(graphs.complete(10), 10, 2.0))),
    ("matching_scramble_layers", dict(num_system=9, edges=graphs.cycle(9), measurement_rate=0.03,
                                      schedule="matching", scramble_layers=2)),
    ("all_edges_path", dict(num_system=7, edges=graphs.path(7), measurement_rate=0.1,
                            schedule="all_edges")),
    ("scramble_pairs_and_matchings", dict(num_system=11, edges=_ws(11, 0.3, 5),
                                          measurement_rate=0.05, scramble_gates=4,
                                          scramble_matchings=1)),
    ("single_random_partner", dict(num_system=10, edges=_ws(10, 1.0, 6), measurement_rate=0.1,
                                   reference="single", reference_partner="random")),
    ("no_reference", dict(num_system=8, edges=graphs.path(8), measurement_rate=0.1,
                          reference="none")),
    ("collection_oriented_gate_set", dict(num_system=8,
                                          edges=[_ws(8, 0.5, (7, g)) for g in range(3)],
                                          circuits_per_graph=2, measurement_rate=0.1,
                                          random_orientation=True, gate_set=[0, 5, 77, 11519])),
]  # fmt: skip


def _spec(kw, layers=20, **changes):
    kw = {"schedule": "random_edges", "reference": "purification", **kw, **changes}
    return mipt.MonitoredCircuit(layers=layers, record_first_touch=True, **kw)


@pytest.mark.parametrize("name, kw", CASES, ids=[c[0] for c in CASES])
def test_first_touches_are_those_of_the_circuit_and_of_the_draws(name, kw):
    spec = _spec(kw)
    seed = [17, len(name)]
    res = spec.run(6, seed=seed)
    n = spec.num_system
    assert res.first_touch.shape == (6, n) and res.first_touch.dtype == np.int64
    for t in range(6):
        text = str(spec.circuit(t, seed=seed))
        assert text == oracle.oracle_circuit_text(spec, t, seed)
        np.testing.assert_array_equal(res.first_touch[t], touches_from_text(text, n), err_msg=name)
        np.testing.assert_array_equal(res.first_touch[t], touches_from_draws(spec, t, seed))
    ft = res.first_touch
    assert np.all((ft >= -1) & (ft <= spec.layers))
    if name == "no_gates_no_measurements":
        assert np.all(ft == -1)
    if name == "random_edges_p1":  # every qubit is measured in layer 1
        assert np.all((ft >= 0) & (ft <= 1))
    if name in ("matching_scramble_layers", "scramble_pairs_and_matchings"):
        assert np.any(ft == 0)
    if name == "random_edges_purification":
        assert np.any(ft > 1)


@pytest.mark.parametrize("stop", ["purified", "distance"])
def test_with_stops_the_touches_are_the_free_run_truncated(stop):
    n = 16
    kw = dict(
        num_system=n,
        edges=[_ws(n, 0.3, (9, g)) for g in range(2)],
        circuits_per_graph=10,
        measurement_rate=0.2 if stop == "purified" else 0.05,
        checkpoints="every_layer",
        probes=[mipt.code_dimension(), mipt.contiguous_distance(cap=2)],
    )
    free = _spec(kw, layers=60).run(seed=4)
    rule = (
        {"stop_when_purified": True}
        if stop == "purified"
        else {"stop_when_distance": 2, "stop_not_before": 3}
    )
    stopped = _spec(kw, layers=60, **rule).run(seed=4)
    final = stopped.final_layer
    assert np.any(final < 60)
    truncated = np.where(free.first_touch <= final[:, None], free.first_touch, -1)
    np.testing.assert_array_equal(stopped.first_touch, truncated)
    # N_bare is known through each row's stop, and after it only where it was already 0
    layers, counts = stopped.bare_counts()
    _, full = free.bare_counts()
    for t in range(stopped.num_trajectories):
        known = layers <= final[t]
        np.testing.assert_array_equal(counts[t, known], full[t, known])
        after = counts[t, ~known]
        if full[t, final[t]] == 0:
            assert np.all(after == 0)
        else:
            assert np.isnan(after).all()


def test_bare_counts_and_times_by_definition():
    spec = _spec(dict(CASES[0][1]), layers=30)
    res = spec.run(12, seed=3)
    ft = res.first_touch
    layers, counts = res.bare_counts()
    assert np.array_equal(layers, np.arange(31))
    for t in range(12):
        for L in layers:
            assert counts[t, L] == sum(1 for q in range(16) if ft[t, q] < 0 or ft[t, q] > L)
    assert np.all(np.diff(counts, axis=1) <= 0)
    sub_layers, sub = res.bare_counts([3, 0, 30])
    np.testing.assert_array_equal(sub, counts[:, [3, 0, 30]])
    assert sub_layers.tolist() == [3, 0, 30]
    time, observed = res.bare_times()
    for t in range(12):
        if observed[t]:
            assert time[t] == ft[t].max() and counts[t, time[t]] == 0
            assert time[t] == 0 or counts[t, time[t] - 1] > 0
        else:
            assert time[t] == res.final_layer[t] and counts[t, -1] > 0
    km = analysis.kaplan_meier(time, observed)
    assert km["survival"][-1] <= 1.0
    with pytest.raises(ValueError, match="layers must lie in"):
        res.bare_counts([31])
    # p = 0 and no gates: nothing is ever touched, every tau_bare is censored at the horizon
    idle = _spec(dict(CASES[4][1]), layers=5).run(3, seed=1)
    assert np.all(idle.bare_counts()[1] == 6)
    t_idle, seen = idle.bare_times()
    assert not seen.any() and np.all(t_idle == 5)


def _bare_by_definition(touches: np.ndarray, final: int, layers) -> np.ndarray:
    """N_bare(T) of one row from its first touches: the qubits untouched after layer T, known
    through the row's final layer and after it only where none was left there (NaN otherwise)."""
    left = int(np.sum(touches < 0))  # first touches lie in [0, final] or are -1
    out = []
    for layer in layers:
        known = layer <= final or left == 0
        out.append(float(np.sum((touches < 0) | (touches > layer))) if known else np.nan)
    return np.array(out)


STOP_BASE = dict(
    num_system=12,
    edges=[_ws(12, 0.3, (21, g)) for g in range(3)],
    circuits_per_graph=8,
    measurement_rate=0.05,
    gates_per_layer=2,
    checkpoints="every_layer",
    probes=[mipt.code_dimension(), mipt.contiguous_distance(cap=2)],
)
SINGLE = dict(reference="single", reference_partner="random")
STOP_CASES = [  # name, spec changes, layers
    # one reference qubit: a row can stop (purified, or d >= 2) with untouched qubits left
    ("single_purified", dict(**SINGLE, measurement_rate=0.08, stop_when_purified=True), 14),
    ("single_distance", dict(**SINGLE, measurement_rate=0.03, stop_when_distance=2), 12),
    ("single_both", dict(**SINGLE, stop_when_purified=True, stop_when_distance=2), 14),
    # purification picture: an untouched qubit carries I(i:R) = 2, so k = 0 (purified) and
    # N_log = 0 (d >= 2) both mean that no qubit is left untouched at the stop
    ("purification_purified", dict(measurement_rate=0.1, stop_when_purified=True), 40),
    ("purification_distance", dict(measurement_rate=0.1, stop_when_distance=2,
                                   stop_not_before=3), 40),
]  # fmt: skip


def _stopped_run(kw, layers):
    """A run with stops, and its first touches read from the circuit text (the whole circuit,
    whatever the stop) and cut at each row's final layer."""
    spec = _spec({**STOP_BASE, **kw}, layers=layers)
    seed = [8, 1]
    res = spec.run(seed=seed)
    text = np.array(
        [
            touches_from_text(str(spec.circuit(t, seed=seed)), spec.num_system)
            for t in range(res.num_trajectories)
        ]
    )
    return res, np.where(text <= res.final_layer[:, None], text, -1)


@pytest.mark.parametrize("name, kw, layers", STOP_CASES, ids=[c[0] for c in STOP_CASES])
def test_bare_counts_past_each_rows_stop(name, kw, layers):
    """N_bare(T) row by row: the count through the row's final layer, after it 0 where no qubit
    was left at the stop and NaN where one was (it might have been touched later), at every layer
    and at a list of layers in any order, with repeats."""
    res, touches = _stopped_run(kw, layers)
    np.testing.assert_array_equal(res.first_touch, touches)
    final = res.final_layer
    for at in (None, [layers, 0, 5, 5, 1, layers - 1, 9]):
        got, counts = res.bare_counts(at)
        want = np.arange(layers + 1) if at is None else np.array(at)
        np.testing.assert_array_equal(got, want)
        assert counts.shape == (res.num_trajectories, want.size) and counts.dtype == np.float64
        for t in range(res.num_trajectories):
            np.testing.assert_array_equal(
                counts[t], _bare_by_definition(touches[t], final[t], want), err_msg=f"row {t}"
            )
    stopped = final < layers
    left = (touches < 0).sum(axis=1)
    unknown = np.isnan(res.bare_counts()[1]).any(axis=1)
    np.testing.assert_array_equal(unknown, stopped & (left > 0))
    assert (stopped & (left == 0)).any()  # stopped with none left: zeros past the stop
    if name.startswith("single"):
        assert unknown.any()
    else:
        assert stopped.any() and not unknown.any()


@pytest.mark.parametrize("name, kw, layers", STOP_CASES, ids=[c[0] for c in STOP_CASES])
def test_bare_times_are_censored_at_each_rows_stop(name, kw, layers):
    """tau_bare is the last first touch where every qubit was touched by the row's final layer,
    and otherwise censored at that layer (a stop or the horizon); kaplan_meier takes the pair,
    and below the earliest censoring depth its survival is the fraction with N_bare > 0."""
    res, touches = _stopped_run(kw, layers)
    final = res.final_layer
    time, observed = res.bare_times()
    assert time.dtype == np.int64 and observed.dtype == bool
    for t in range(res.num_trajectories):
        if (touches[t] >= 0).all():
            assert observed[t] and time[t] == touches[t].max() <= final[t], t
        else:  # some qubit untouched through the stop or the horizon
            assert not observed[t] and time[t] == final[t], t
    stopped = final < layers
    if name.startswith("single"):
        assert (stopped & ~observed).any()  # censored at a stop below the horizon
    else:  # pure at a stop (k = 0) or d >= 2 there (N_log = 0): nothing left untouched
        purified = res.purification_layer >= 0
        assert observed[purified].all()
        assert np.all(time[purified] <= res.purification_layer[purified])
        assert observed[res.stopped_on_distance].all()
    km = analysis.kaplan_meier(time, observed)
    survival = km["survival"]
    assert survival.size and np.all((survival >= 0) & (survival <= 1))
    assert np.all(np.diff(survival) <= 0)
    _, counts = res.bare_counts()
    horizon = time[~observed].min() if (~observed).any() else layers
    for when, value in zip(km["times"], survival, strict=True):
        if when <= horizon:  # every row's tau_bare is known to be at most, or above, `when`
            column = counts[:, int(when)]
            assert not np.isnan(column).any()
            assert value == pytest.approx(np.mean(column > 0), rel=1e-12)


def test_without_the_option_there_is_nothing_and_it_costs_nothing():
    kw = dict(num_system=8, edges=graphs.cycle(8), layers=10, measurement_rate=0.1,
              reference="purification", probes=[mipt.code_dimension()])  # fmt: skip
    off = mipt.MonitoredCircuit(**kw)
    res = off.run(3, seed=1)
    assert res.first_touch is None and not off.record_first_touch
    assert all(r["name"] != "first_touch" for r in api._result_layout(off, 3)["arrays"])
    with pytest.raises(ValueError, match="record_first_touch"):
        res.bare_counts()
    with pytest.raises(ValueError, match="record_first_touch"):
        res.bare_times()
    on = mipt.MonitoredCircuit(**kw, record_first_touch=True)
    assert api._result_bytes(on, 10) - api._result_bytes(off, 10) == 10 * 8 * 8
    with pytest.raises(TypeError, match="record_first_touch must be True or False"):
        mipt.MonitoredCircuit(**kw, record_first_touch="yes")
    # a replay with the option gives the first touches of a run made without it
    again = dataclasses.replace(off, record_first_touch=True).replay(2, seed=1)
    np.testing.assert_array_equal(
        again.values["code_dimension"][0], res.values["code_dimension"][2]
    )
    assert again.first_touch.shape == (1, 8)


def test_chunks_indices_threads_replay_save_load_and_concatenate(tmp_path):
    spec = _spec(dict(CASES[11][1]), layers=25)
    whole = spec.run(seed=5)
    assert whole.num_trajectories == 6
    parts = mipt.EnsembleResult.concatenate(
        [spec.run(2, seed=5), spec.run(4, seed=5, first_trajectory=2)]
    )
    np.testing.assert_array_equal(parts.first_touch, whole.first_touch)
    order = np.random.default_rng(1).permutation(6)
    shuffled = spec.run(indices=order, seed=5, num_threads=1)
    np.testing.assert_array_equal(shuffled.first_touch, whole.first_touch[order])
    for u in (0, 5):
        np.testing.assert_array_equal(whole.replay(u).first_touch[0], whole.first_touch[u])
    whole.save(tmp_path / "r.npz")
    back = mipt.EnsembleResult.load(tmp_path / "r.npz")
    assert back.spec.record_first_touch and back.first_touch.dtype == np.int64
    np.testing.assert_array_equal(back.first_touch, whole.first_touch)
    np.testing.assert_array_equal(back.replay(3).first_touch[0], whole.first_touch[3])
    kept = api._rows(whole, np.arange(6) % 2 == 0)
    np.testing.assert_array_equal(kept.first_touch, whole.first_touch[::2])
    other = dataclasses.replace(spec, record_first_touch=False).run(2, seed=5, first_trajectory=2)
    with pytest.raises(ValueError, match="record_first_touch"):
        mipt.EnsembleResult.concatenate([spec.run(2, seed=5), other])


@pytest.mark.parametrize("n", [16, 64])
@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_untouched_qubits_carry_two_bits_and_bound_n_log_and_k(n, beta):
    """Purification picture: an untouched qubit is still Bell-paired with its reference, so
    I(i:R) = 2 there, N_bare(T) <= N_log(T) and N_bare(T) <= k(T) on every layer."""
    for p in (0.0, 0.02, 0.1):
        spec = mipt.MonitoredCircuit(
            num_system=n, edges=[_ws(n, beta, (13, g)) for g in range(2)], circuits_per_graph=4,
            layers=24, measurement_rate=p, schedule="random_edges", reference="purification",
            checkpoints="every_layer", record_first_touch=True,
            probes=[mipt.code_dimension(), mipt.logical_sites(), mipt.site_information()],
        )  # fmt: skip
        res = spec.run(seed=[n, int(1000 * p)])
        layers, bare = res.bare_counts()
        assert np.all(bare <= res.values["logical_sites"])
        assert np.all(bare <= res.values["code_dimension"])
        sites = res.values["site_information"]
        ft = res.first_touch
        for L in (0, 1, 2, 5):
            untouched = (ft < 0) | (ft > L)
            assert np.all(sites[:, L][untouched] == 2)
        assert bare[:, 0].tolist() == [n] * res.num_trajectories  # no scramble: all bare at 0
        assert np.any(bare[:, 1] > 0)


JSON_SPEC = {
    "num_system": 12,
    "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.3, "seed": 1},
    "layers": 30,
    "measurement_rate": 0.05,
    "schedule": "random_edges",
    "reference": "purification",
    "checkpoints": 5,
    "record_first_touch": True,
    "probes": [{"kind": "code_dimension"}, {"kind": "logical_sites"}],
}


def test_the_json_api_summarizes_saves_resumes_and_extends(tmp_path):
    out = api.run_ensemble(JSON_SPEC, trajectories=20, seed=3, output_dir=str(tmp_path / "run"))
    res = mipt.EnsembleResult.load(tmp_path / "run" / "result.npz")
    block = out["first_touch"]
    layers, counts = res.bare_counts(res.checkpoint_layers)
    assert block["layers"] == layers.tolist()
    np.testing.assert_allclose(block["bare_mean"], counts.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(block["fraction_with_bare"], (counts > 0).mean(axis=0), rtol=1e-5)
    _, observed = res.bare_times()
    assert block["all_touched"] == int(observed.sum())
    assert block["censored"] == 20 - int(observed.sum())
    assert any(note.startswith("first_touch:") for note in out["notes"])
    assert api.load_results(str(tmp_path / "run"))["first_touch"] == block
    assert api.capabilities("ensemble_spec")["ensemble_spec"]["record_first_touch"] is False
    layout = api.estimate_ensemble(JSON_SPEC, trajectories=7)["layout"]["arrays"]
    row = next(r for r in layout if r["name"] == "first_touch")
    assert row["shape"] == [7, 12] and row["dtype"] == "int64"
    # a resume with the option changed is another protocol
    with pytest.raises(ValueError, match="different spec"):
        api.start_ensemble_job(
            {**JSON_SPEC, "record_first_touch": False}, output_dir=str(tmp_path / "run"), seed=3
        )
    # extending the run in chunks gives the rows of one direct run
    more = api.start_ensemble_job(JSON_SPEC, 50, output_dir=str(tmp_path / "run"), chunk_size=9)
    assert more["resumed_from"] == 20
    assert api.job_status(more["job_id"], wait_seconds=60)["status"] == "completed"
    extended = mipt.EnsembleResult.load(tmp_path / "run" / "result.npz")
    direct = api._build_spec(JSON_SPEC).run(50, seed=3)
    np.testing.assert_array_equal(extended.first_touch, direct.first_touch)
    np.testing.assert_array_equal(extended.values["logical_sites"], direct.values["logical_sites"])
    with pytest.raises(ValueError, match="record_first_touch must be"):
        api.run_ensemble({**JSON_SPEC, "record_first_touch": 1}, trajectories=2, seed=1)


def _median_by_layer(res: mipt.EnsembleResult) -> int | None:
    """The median of tau_bare over all rows read off N_bare(T) at every layer: the first T with
    P(N_bare(T) > 0) <= 1/2, over the layers known for every row, when some is below 1/2."""
    layers, counts = res.bare_counts()
    known = ~np.isnan(counts).any(axis=0)
    some = (counts[:, known] > 0).mean(axis=0)
    below = np.flatnonzero(some <= 0.5)
    return int(layers[known][below[0]]) if (some < 0.5).any() else None


@pytest.mark.parametrize(
    "extra",
    [
        {"checkpoints": "final"},
        {"checkpoints": [0, 30]},
        {"checkpoints": "every_layer"},
        {"checkpoints": "final", "measurement_rate": 0.0},  # too few touches: no median
        {
            "checkpoints": "final",
            "probes": [
                {"kind": "code_dimension"},
                {"kind": "contiguous_distance", "cap": 2, "checkpoints": "every_layer"},
            ],
            "stop_when_distance": 2,
        },
    ],
)
def test_the_summary_median_of_tau_bare_is_exact_at_any_checkpoints(extra):
    # tau_bare is known at every layer, so its median must not be rounded up to a checkpoint
    spec = {**JSON_SPEC, "gates_per_layer": 2, **extra}
    out = api.run_ensemble(spec, trajectories=24, seed=5)
    res = api._build_spec(spec).run(24, seed=5)
    assert out["first_touch"]["tau_bare_median_all"] == _median_by_layer(res)


def _median_from_touches(touches: np.ndarray, final: np.ndarray, layers: int) -> int | None:
    """tau_bare_median_all from the first touches alone. tau_bare is the last first touch of a row
    whose every qubit was touched, and is only known to exceed final_layer otherwise, so
    P(tau_bare > T) is known for every row through the earliest such final_layer (the horizon
    when there is none). The median is the first T with at most half of the rows above T; it is
    given only when strictly more than half of the rows had tau_bare <= that depth."""
    total = len(touches)
    tau = [int(row.max()) if (row >= 0).all() else None for row in touches]
    known_to = min((int(f) for f, x in zip(final, tau, strict=True) if x is None), default=layers)

    def done(T):
        return sum(1 for x in tau if x is not None and x <= T)

    if 2 * done(known_to) <= total:
        return None
    return next(T for T in range(known_to + 1) if 2 * done(T) >= total)


STOP_JSON = {
    "num_system": 12,
    "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.3, "seed": 1, "realizations": 3},
    "circuits_per_graph": 8,
    "layers": 30,
    "measurement_rate": 0.05,
    "schedule": "random_edges",
    "gates_per_layer": 2,
    "reference": "purification",
    "checkpoints": "every_layer",
    "record_first_touch": True,
    "probes": [{"kind": "code_dimension"}, {"kind": "contiguous_distance", "cap": 2}],
}
SUMMARY_CASES = [  # name, spec changes, what the median does
    # one row censored at a distance stop (layer 11), past the median
    ("distance_stops", {"reference": "single", "reference_partner": "random", "layers": 20,
                        "stop_when_distance": 2, "stop_not_before": 10}, "defined"),
    # no stop, rows censored at the horizon
    ("censored_at_the_horizon", {"measurement_rate": 0.0, "gates_per_layer": 1, "layers": 24},
     "defined"),
    # at most half of the rows run out of bare qubits
    ("few_touched", {"measurement_rate": 0.0, "gates_per_layer": 1, "layers": 10}, "few"),
    # most rows run out, but one stops (purified) with bare qubits at layer 1, and the median
    # is not known from what lies below that depth
    ("censored_early", {"reference": "single", "reference_partner": "random",
                        "measurement_rate": 0.03, "stop_when_purified": True}, "early"),
]  # fmt: skip


@pytest.mark.parametrize("name, changes, median", SUMMARY_CASES, ids=[c[0] for c in SUMMARY_CASES])
def test_the_summary_block_against_the_saved_arrays(tmp_path, name, changes, median):
    """all_touched, censored, count, bare_mean and fraction_with_bare at every shown layer, and
    tau_bare_median_all, from the saved first touches and final layers, with stops."""
    spec = {**STOP_JSON, **changes}
    out = api.run_ensemble(spec, seed=1, output_dir=str(tmp_path / "run"))
    res = mipt.EnsembleResult.load(tmp_path / "run" / "result.npz")
    touches, final, layers = res.first_touch, res.final_layer, spec["layers"]
    block = out["first_touch"]
    observed = (touches >= 0).all(axis=1)
    assert block["all_touched"] == int(observed.sum())
    assert block["censored"] == int((~observed).sum())
    assert block["layers"] == res.checkpoint_layers.tolist()  # every layer is shown here
    for j, T in enumerate(block["layers"]):
        bare = np.array(
            [_bare_by_definition(row, f, [T])[0] for row, f in zip(touches, final, strict=True)]
        )
        known = ~np.isnan(bare)
        assert block["count"][j] == int(known.sum()) > 0
        assert block["bare_mean"][j] == pytest.approx(bare[known].mean(), rel=1e-5)
        if known.all():  # P(N_bare(T) > 0) = P(tau_bare > T)
            assert block["fraction_with_bare"][j] == pytest.approx(np.mean(bare > 0), rel=1e-5)
        else:  # some row stopped before T with bare qubits left
            assert block["fraction_with_bare"][j] is None
    stopped_bare = (final < layers) & ~observed
    assert (None in block["fraction_with_bare"]) == bool(stopped_bare.any())
    expected = _median_from_touches(touches, final, layers)
    assert block["tau_bare_median_all"] == expected
    if median == "defined":
        assert expected is not None and block["censored"] > 0
    elif median == "few":  # at most half of the rows ran out of bare qubits
        assert expected is None and 2 * observed.sum() <= observed.size
    else:  # more than half ran out, but not by the earliest censoring depth
        assert expected is None and 2 * observed.sum() > observed.size
        assert stopped_bare.any()


def test_the_summary_median_on_constructed_touches():
    """The summary helper on first touches and final layers set by hand (12 layers, 6 qubits):
    the exact-half boundary, odd and even numbers of rows, censoring at the horizon and at a stop,
    then random rows, against _median_from_touches and _bare_by_definition."""
    base = _spec(dict(CASES[4][1]), layers=12).run(1, seed=1)
    assert base.checkpoint_layers.tolist() == [12]

    def summary(touches, final):
        result = dataclasses.replace(
            base,
            trajectories=np.arange(len(touches)),
            first_touch=np.array(touches, dtype=np.int64),
            final_layer=np.array(final, dtype=np.int64),
        )
        return api._first_touch_summary(result, result.checkpoint_layers, np.arange(1))

    def done_at(last):  # every qubit touched, the last one in layer `last`
        return [0, 0, min(1, last), 0, 0, last]

    bare = [3, -1, 0, 0, 1, 2]  # a qubit untouched through the row's final layer
    cases = [  # rows, final layers, median
        ([done_at(3), done_at(5), bare, bare], [12] * 4, None),  # exactly half: no median
        ([done_at(3), done_at(5), done_at(9), bare], [12] * 4, 5),
        ([done_at(2), done_at(4), done_at(6), bare, bare], [12] * 5, 6),  # (5 + 1) // 2 = third
        ([done_at(3), done_at(5), done_at(9), bare], [12, 12, 12, 4], None),  # stop at 4
        ([done_at(3), done_at(5), done_at(9), bare], [12, 12, 12, 5], None),  # 2 by 5: half
        ([done_at(3), done_at(5), done_at(9), bare], [12, 12, 12, 9], 5),
        ([done_at(7), done_at(0), done_at(12)], [7, 12, 12], 7),
        ([done_at(4)], [12], 4),
        ([bare], [12], None),
    ]
    for touches, final, median in cases:
        block = summary(touches, final)
        assert block["tau_bare_median_all"] == median, (touches, final)
        assert block["tau_bare_median_all"] == _median_from_touches(np.array(touches), final, 12)
        assert block["censored"] == sum(1 for row in touches if min(row) < 0)
    rng = np.random.default_rng(29)
    medians = set()
    for _ in range(300):
        rows = int(rng.integers(1, 10))
        final = rng.integers(0, 13, rows)
        final[rng.random(rows) < 0.5] = 12  # many rows reach the horizon
        touches = rng.integers(0, 13, (rows, 6)) % (final[:, None] + 1)
        touches[rng.random((rows, 6)) < 0.08] = -1
        block = summary(touches, final)
        expected = _median_from_touches(touches, final, 12)
        assert block["tau_bare_median_all"] == expected, (touches.tolist(), final.tolist())
        medians.add(expected is None)
        assert block["all_touched"] == int((touches >= 0).all(axis=1).sum())
        assert block["censored"] == rows - block["all_touched"]
        counts = np.array([_bare_by_definition(touches[t], final[t], [12])[0] for t in range(rows)])
        if np.isnan(counts).any():
            assert np.isnan(block["fraction_with_bare"][0])
        else:
            assert block["fraction_with_bare"][0] == np.mean(counts > 0)
    assert medians == {True, False}
