"""Ensemble and circuit summaries checked against numpy on the saved arrays.

The P_C(m) curves condition on a code (k > 0) or say that they do not, the
purification statistics follow their definitions, and nothing is cut from a
summary without ``truncated: true`` and a note that survives the cut.
"""

import json
import math

import numpy as np
import pytest

from sparsegf2 import api, mipt

CONDITIONED = "among trajectories that hold a code (k > 0); count_with_code says how many"


def _close(shown, expected) -> None:
    """Summary numbers (rounded to 6 significant digits, null for NaN) against numpy."""
    got = np.array(shown, dtype=np.float64)  # null -> NaN
    np.testing.assert_allclose(got, np.asarray(expected, dtype=np.float64), rtol=1e-5, atol=1e-9)


def _mean_and_stderr(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, standard error and count over axis 0, leaving NaN entries out (numpy only)."""
    ok = ~np.isnan(values)
    count = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok, values, 0.0).sum(axis=0) / count
        squares = (np.where(ok, values - mean, 0.0) ** 2).sum(axis=0)
        stderr = np.sqrt(squares / (count - 1) / count)
    return mean, stderr, count


# ---------------------------------------------------------------------------
# oracles on one saved run
# ---------------------------------------------------------------------------

# Seed 1 with 20 trajectories: 13 purify within the horizon and 7 are censored, and the
# survival curve sits at exactly 1/2 from layer 11 to 21 before it drops below. Each test
# asserts the property it relies on (censoring, codes and k = 0, the 1/2 boundary) on the
# saved arrays first, so a change of the random stream cannot make it pass vacuously.
ORACLE_SPEC = {
    "num_system": 12,
    "graph": {"family": "cycle"},
    "layers": 40,
    "measurement_rate": 0.15,
    "reference": "single",
    "checkpoints": "every_layer",
    "stop_when_purified": True,
    "probes": [
        {"kind": "code_dimension"},
        {"kind": "half_cut_entropy"},
        {"kind": "contiguous_distance"},
        {"kind": "entropy_profile", "checkpoints": 10},
        {"kind": "erasure_lost_profile", "samples": 4, "checkpoints": 10},
        {"kind": "erasure_first_failure", "samples": 30, "checkpoints": 10},
        {"kind": "uncorrectable_region_counts", "max_size": 2, "checkpoints": [10, 20, 40]},
    ],
}
ORACLE_N = 12
ORACLE_TRAJECTORIES = 20


@pytest.fixture(scope="module")
def oracle_run(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("oracle") / "run"
    out = api.run_ensemble(ORACLE_SPEC, ORACLE_TRAJECTORIES, seed=1, output_dir=str(out_dir))
    return out, mipt.EnsembleResult.load(out_dir / "result.npz")


def _columns(result: mipt.EnsembleResult, name: str, layers) -> np.ndarray:
    own = np.asarray(result.layers(name))
    cols = np.searchsorted(own, layers)
    assert np.array_equal(own[cols], layers)
    return cols


def test_scalar_means_errors_and_counts_match_numpy(oracle_run):
    out, res = oracle_run
    shown = out["checkpoints"]["layers"]
    assert np.isnan(res.values["half_cut_entropy"]).any()  # stopped trajectories
    assert np.isnan(res.values["contiguous_distance"]).any()  # no code, no distance
    for name in ("code_dimension", "half_cut_entropy", "contiguous_distance"):
        entry = out["probes"][name]
        values = res.values[name][:, _columns(res, name, entry.get("layers", shown))]
        mean, stderr, count = _mean_and_stderr(values)
        _close(entry["mean"], mean)
        _close(entry["stderr"], stderr)
        if np.isnan(res.values[name]).any():
            assert entry["count"] == count.tolist()
        else:
            assert "count" not in entry


def test_vector_errors_match_numpy(oracle_run):
    out, res = oracle_run
    assert np.isnan(res.values["entropy_profile"]).any()  # stopped trajectories have no profile
    for name in ("entropy_profile", "erasure_lost_profile"):
        entry = out["probes"][name]
        (c,) = _columns(res, name, [entry["at_layer"]])
        values = res.values[name][:, c, :]
        rows = ~np.isnan(values[:, 0])
        assert entry["count"] == rows.sum() and "count_with_code" not in entry
        mean, stderr, _ = _mean_and_stderr(values[rows])
        index = entry.get("index", np.arange(ORACLE_N + 1))
        _close(entry["mean"], mean[index])
        _close(entry["stderr"], stderr[index])


def test_purification_statistics_match_numpy(oracle_run):
    out, res = oracle_run
    pur = out["purification"]
    tau = res.purification_layer
    done = tau[tau >= 0]
    assert 0 < done.size < tau.size  # censored trajectories: a mean with the -1s would differ
    assert pur["purified"] == done.size and pur["censored"] == tau.size - done.size
    assert pur["tau_mean_purified"] == pytest.approx(done.sum() / done.size, rel=1e-5)
    assert pur["tau_median_purified"] == pytest.approx(np.median(done), rel=1e-5)
    p10, p25, p75, p90 = np.percentile(done, [10, 25, 75, 90], method="linear")  # as api.py
    assert pur["tau_quantiles_purified"] == pytest.approx(
        {"p10": p10, "p25": p25, "p75": p75, "p90": p90}, rel=1e-5
    )
    # The median over all trajectories, censored ones counted as later than any layer:
    # with 20 trajectories it is the 10th smallest tau. Here the 10th and 11th differ, so
    # the survival curve sits at exactly 1/2 between them: the boundary case.
    ordered = np.sort(np.where(tau >= 0, tau, np.inf))
    half = tau.size // 2
    assert tau.size % 2 == 0 and np.isfinite(ordered[half]) and ordered[half - 1] < ordered[half]
    assert pur["tau_median_all"] == ordered[half - 1]


def test_failure_by_size_is_conditioned_on_a_code_and_matches_numpy(oracle_run):
    out, res = oracle_run
    n = ORACLE_N
    entry = out["probes"]["erasure_first_failure"]
    assert CONDITIONED in entry["what"]
    first = res.values["erasure_first_failure"]  # (trajectories, evaluations, samples)
    code = first.min(axis=2) <= n  # k = 0 gives D = n + 1 for every sample
    k = res.values["code_dimension"][:, _columns(res, "code_dimension", res.layers(
        "erasure_first_failure"))]  # fmt: skip
    assert np.array_equal(code, k > 0)
    (c,) = _columns(res, "erasure_first_failure", [entry["at_layer"]])
    assert 2 <= code[:, c].sum() < ORACLE_TRAJECTORIES  # codes, and trajectories without one
    curves = (first[code[:, c], c, :, None] <= np.arange(n + 1)).mean(axis=1)  # P_C(m) per code
    mean, stderr, _ = _mean_and_stderr(curves)
    assert entry["count"] == ORACLE_TRAJECTORIES and entry["count_with_code"] == code[:, c].sum()
    _close(entry["mean"], mean)
    _close(entry["stderr"], stderr)
    assert entry["mean"][-1] == 1.0  # erasing every qubit of a code always fails
    grid = entry["vs_depth"]
    cols = _columns(res, "erasure_first_failure", grid["layers"])
    assert grid["count_with_code"] == code[:, cols].sum(axis=0).tolist()
    for row, col in zip(grid["mean"], cols, strict=True):
        held = first[code[:, col], col, :, None] <= np.asarray(grid["index"])
        _close(row, held.mean(axis=1).mean(axis=0))


def test_exact_failure_by_size_is_conditioned_by_code_dimension_at_its_layers(oracle_run):
    out, res = oracle_run
    n, name = ORACLE_N, "uncorrectable_region_counts[2]"
    entry = out["probes"][name]
    assert CONDITIONED in entry["what"]  # code_dimension runs at every layer: a superset
    counts = res.values[name]  # (trajectories, evaluations, 2): m = 1, 2
    k = res.values["code_dimension"][:, _columns(res, "code_dimension", res.layers(name))]
    (c,) = _columns(res, name, [entry["at_layer"]])
    code = k[:, c] > 0
    assert 2 <= code.sum() < ORACLE_TRAJECTORIES
    exact = counts[code, c, :] / np.array([math.comb(n, 1), math.comb(n, 2)])
    mean, stderr, _ = _mean_and_stderr(np.column_stack([np.zeros(code.sum()), exact]))
    assert entry["count_with_code"] == code.sum()
    _close(entry["mean"], mean)
    _close(entry["stderr"], stderr)
    assert entry["vs_depth"]["count_with_code"] == (k > 0).sum(axis=0).tolist()
    assert not any(note.startswith(name) for note in out.get("notes", []))


# ---------------------------------------------------------------------------
# A1: P_C(m) with trajectories that have no code (review repro e15)
# ---------------------------------------------------------------------------

E15_SPEC = {
    "num_system": 16,
    "graph": {"family": "cycle"},
    "layers": 48,
    "measurement_rate": 0.15,
    "reference": "purification",
    "checkpoints": [16, 48],
    "probes": [
        {"kind": "erasure_first_failure", "samples": 50},
        {"kind": "uncorrectable_region_counts", "max_size": 2},
    ],
}


def test_e15_failure_curves_condition_on_a_code(tmp_path):
    out = api.run_ensemble(E15_SPEC, 40, seed=3, output_dir=str(tmp_path / "run"))
    res = mipt.EnsembleResult.load(tmp_path / "run" / "result.npz")
    first = res.values["erasure_first_failure"]
    code = first.min(axis=2) <= 16
    assert 0 < code[:, -1].sum() < 40  # some trajectories lost every logical qubit
    # the same trajectories with a code_dimension probe (probes never change a trajectory)
    k = api._build_spec(dict(E15_SPEC, probes=[{"kind": "code_dimension"}])).run(40, seed=3)
    assert np.array_equal(code, k.values["code_dimension"] > 0)
    entry = out["probes"]["erasure_first_failure"]
    assert CONDITIONED in entry["what"] and entry["at_layer"] == 48
    assert entry["count"] == 40 and entry["count_with_code"] == code[:, -1].sum()
    assert entry["mean"][-1] == 1.0  # the unconditioned mean would be the fraction with a code
    assert entry["vs_depth"]["count_with_code"] == code.sum(axis=0).tolist()
    # Counts alone cannot tell k = 0 from a large distance. With no code_dimension probe,
    # the first-failure probe at the same layers tells which trajectories hold a code.
    counts = out["probes"]["uncorrectable_region_counts[2]"]
    assert CONDITIONED in counts["what"] and counts["count_with_code"] == code[:, -1].sum()
    assert counts["vs_depth"]["count_with_code"] == code.sum(axis=0).tolist()
    exact = res.values["uncorrectable_region_counts[2]"][code[:, -1], -1] / np.array([16, 120])
    mean, stderr, _ = _mean_and_stderr(np.column_stack([np.zeros(len(exact)), exact]))
    _close(counts["mean"], mean)
    _close(counts["stderr"], stderr)
    assert not any(n.startswith("uncorrectable_region_counts") for n in out.get("notes", []))


def test_counts_are_conditioned_only_by_a_probe_at_every_one_of_their_layers(tmp_path):
    counts_probe = E15_SPEC["probes"][1]
    # nothing tells which trajectories hold a code: every trajectory is averaged, and said so
    alone = dict(E15_SPEC, probes=[counts_probe])
    out = api.run_ensemble(alone, 10, seed=3, output_dir=str(tmp_path / "alone"))
    res = mipt.EnsembleResult.load(tmp_path / "alone" / "result.npz")
    counts = out["probes"]["uncorrectable_region_counts[2]"]
    assert "count_with_code" not in counts
    assert "trajectories without a code (k = 0) count as P_C = 0" in counts["what"]
    exact = res.values["uncorrectable_region_counts[2]"][:, -1] / np.array([16, 120])
    _close(counts["mean"], [0.0, *exact.mean(axis=0)])
    note = [n for n in out["notes"] if n.startswith("uncorrectable_region_counts[2]:")]
    assert len(note) == 1 and "code_dimension or erasure_first_failure" in note[0]
    # probes evaluated at layer 48 only cannot condition the counts at layer 16
    late = dict(E15_SPEC, probes=[counts_probe,
                                  {"kind": "code_dimension", "checkpoints": "final"},
                                  {"kind": "erasure_first_failure", "samples": 5,
                                   "checkpoints": "final"}])  # fmt: skip
    out = api.run_ensemble(late, 10, seed=3)
    assert "count_with_code" not in out["probes"]["uncorrectable_region_counts[2]"]
    # code_dimension at the spec's checkpoints (a probe of another name works too)
    same = dict(E15_SPEC, probes=[counts_probe, {"kind": "code_dimension", "name": "k"}])
    out = api.run_ensemble(same, 10, seed=3)
    counts = out["probes"]["uncorrectable_region_counts[2]"]
    assert CONDITIONED in counts["what"] and "count_with_code" in counts
    assert not any(n.startswith("uncorrectable_region_counts") for n in out.get("notes", []))


def test_conditioned_curves_show_the_last_evaluation_with_a_code(tmp_path):
    spec = {
        "num_system": 8, "graph": {"family": "cycle"}, "layers": 30, "measurement_rate": 0.5,
        "reference": "single", "checkpoints": "every_layer", "stop_when_purified": True,
        "probes": [{"kind": "code_dimension"}, {"kind": "erasure_first_failure", "samples": 5},
                   {"kind": "uncorrectable_region_counts", "max_size": 1}],
    }  # fmt: skip
    out = api.run_ensemble(spec, 6, seed=1, output_dir=str(tmp_path / "run"))
    res = mipt.EnsembleResult.load(tmp_path / "run" / "result.npz")
    code = res.values["code_dimension"] > 0  # (trajectories, layers 0..30)
    held = code.sum(axis=0)
    assert held[-1] == 0 and held[0] == 6  # codes at the start, none at the last layer
    c = int(np.flatnonzero(held)[-1])
    first = res.values["erasure_first_failure"][code[:, c], c, :]
    curves = (first[:, :, None] <= np.arange(9)).mean(axis=1)
    counts = res.values["uncorrectable_region_counts[1]"][code[:, c], c, :] / 8
    expected = {
        "erasure_first_failure": curves.mean(axis=0),
        "uncorrectable_region_counts[1]": [0.0, counts.mean()],
    }
    for name, mean in expected.items():
        entry = out["probes"][name]
        assert entry["at_layer"] == c and entry["count_with_code"] == held[c] > 0, name
        assert entry["count"] == 6
        _close(entry["mean"], mean)


def test_lost_information_summaries_say_that_trajectories_without_a_code_lose_nothing(tmp_path):
    spec = dict(E15_SPEC, probes=[{"kind": "lost_information", "region": "0:4", "name": "lost"},
                                  {"kind": "erasure_lost_profile", "samples": 4},
                                  {"kind": "code_dimension"}])  # fmt: skip
    out = api.run_ensemble(spec, 40, seed=3, output_dir=str(tmp_path / "run"))
    res = mipt.EnsembleResult.load(tmp_path / "run" / "result.npz")
    assert (res.values["code_dimension"][:, -1] == 0).any()  # trajectories without a code
    profile = out["probes"]["erasure_lost_profile"]
    assert "trajectories without a code (k = 0) count as 0 bits lost" in profile["what"]
    assert "count_with_code" not in profile  # not conditioned: averaged over every trajectory
    _close(profile["mean"], res.values["erasure_lost_profile"][:, -1].mean(axis=0))
    _close(out["probes"]["lost"]["mean"], res.values["lost"].mean(axis=0))
    for key in ("lost", "erasure_lost_profile"):
        note = [n for n in out["notes"] if n.startswith(f"{key}:")]
        assert len(note) == 1 and "count as 0 bits lost" in note[0], key


def test_failure_curves_without_any_code_are_null(tmp_path):
    spec = {
        "num_system": 8, "graph": {"family": "cycle"}, "layers": 30, "measurement_rate": 0.5,
        "reference": "single", "checkpoints": "final", "stop_when_purified": True,
        "probes": [{"kind": "code_dimension"}, {"kind": "erasure_first_failure", "samples": 5},
                   {"kind": "uncorrectable_region_counts", "max_size": 1}],
    }  # fmt: skip
    out = api.run_ensemble(spec, 6, seed=1, output_dir=str(tmp_path / "run"))
    res = mipt.EnsembleResult.load(tmp_path / "run" / "result.npz")
    assert np.all(res.values["code_dimension"] == 0)  # every reference purified
    for name in ("erasure_first_failure", "uncorrectable_region_counts[1]"):
        entry = out["probes"][name]
        assert entry["count"] == 6 and entry["count_with_code"] == 0
        assert entry["mean"] == [None] * len(entry["mean"])
        assert entry["stderr"] == [None] * len(entry["stderr"])


# ---------------------------------------------------------------------------
# A3: tau_median_all needs strictly more than half purified (review repro e06)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tau, median",
    [
        ([10, -1, 4, -1], None),  # exactly half purified within the horizon: undefined
        ([1, 2, 3, -1, -1, -1], None),
        ([-1, -1], None),
        ([10, 12, 4, -1], 10),  # S = 1/2 exactly at layer 10, 1/4 at 12
        ([5, 7], 5),
    ],
)
def test_tau_median_all_needs_more_than_half_purified(tau, median):
    summary = api._purification_summary(np.array(tau), np.arange(13))
    assert summary["tau_median_all"] == median


def test_e06_exactly_half_purified_has_no_overall_median():
    spec = {"num_system": 8, "graph": {"family": "cycle"}, "layers": 12, "measurement_rate": 0.2,
            "reference": "single", "checkpoints": "every_layer"}  # fmt: skip
    pur = api.run_ensemble(spec, 4, seed=1)["purification"]  # tau = [10, -1, 4, -1]
    assert pur["purified"] == 2 and pur["trajectories"] == 4
    assert pur["tau_median_all"] is None


# ---------------------------------------------------------------------------
# A10: notes survive the size cap (review repro e07)
# ---------------------------------------------------------------------------


def test_bounded_keeps_every_note_whole_and_adds_its_own_once():
    notes = [f"note {i}: " + "x" * 700 for i in range(4)]
    out = api._bounded({"seed": 1, "blob": ["y" * 3000] * 10, "notes": notes})
    assert out["truncated"] and len(json.dumps(out)) <= api.MAX_OUTPUT_CHARS
    assert out["notes"][:4] == notes and len(out["notes"]) == 5
    marked = api._mark_truncated(api._mark_truncated({"notes": ["a"]}, "cut"), "cut")
    assert marked == {"notes": ["a", "cut"], "truncated": True}


def test_e07_interpretive_notes_survive_when_the_summary_is_cut():
    spec = {
        "num_system": 70, "graph": {"family": "cycle"}, "layers": 60, "measurement_rate": 0.2,
        "reference": "purification", "checkpoints": "every_layer", "stop_when_purified": True,
        "probes": [{"kind": "entropy", "region": "0:70:2"}, {"kind": "contiguous_distance"},
                   {"kind": "code_snapshot", "checkpoints": "final"},
                   {"kind": "code_dimension"}, {"kind": "half_cut_entropy"}]
        + [{"kind": "entropy", "region": [i, i + 1], "name": f"S{i}"} for i in range(8)],
    }  # fmt: skip
    out = api.run_ensemble(spec, 6, seed=1)
    assert out["truncated"] and len(json.dumps(out)) <= api.MAX_OUTPUT_CHARS
    for start in ("long probe names", "contiguous_distance:", "code_snapshot:",
                  "stop_when_purified:"):  # fmt: skip
        assert sum(note.startswith(start) for note in out["notes"]) == 1, start
    assert sum("evaluations per probe are shown" in note for note in out["notes"]) == 1


# ---------------------------------------------------------------------------
# A11: automatic names in run_circuit are unique (review repro e11)
# ---------------------------------------------------------------------------


def test_e11_automatic_observable_names_never_take_a_used_name():
    obs = [{"kind": "entropy", "region": [0]},
           {"kind": "entropy", "region": [2], "name": "entropy#2"},
           {"kind": "entropy", "region": [0, 1]}]  # fmt: skip
    out = api.run_circuit("H 0\nCX 0 1\nH 2\nCX 2 3", seed=1, observables=obs)
    assert out["observables"] == {"entropy": 1, "entropy#2": 1, "entropy#3": 0}
    # a name given explicitly is kept, even after an automatic one of the same text
    obs = [{"kind": "entropy", "region": [0]},
           {"kind": "entropy", "region": [4], "name": "entropy"}]  # fmt: skip
    out = api.run_circuit("H 0\nCX 0 1\nH 4", seed=1, observables=obs)
    assert out["observables"] == {"entropy#2": 1, "entropy": 0}
    # two explicit names alike: the second gets the first free suffix
    obs = [{"kind": "entropy", "region": [0], "name": "s"},
           {"kind": "entropy", "region": [4], "name": "s"},
           {"kind": "entropy", "region": [2], "name": "s#2"}]  # fmt: skip
    out = api.run_circuit("H 0\nCX 0 1\nH 4", seed=1, observables=obs)
    assert out["observables"] == {"s": 1, "s#3": 0, "s#2": 0}


# ---------------------------------------------------------------------------
# A15: subsampled vector curves and depth grids set truncated (review repro e11)
# ---------------------------------------------------------------------------

PROFILE_SPEC = {
    "num_system": 100,
    "graph": {"family": "cycle"},
    "layers": 4,
    "measurement_rate": 0.1,
    "probes": [{"kind": "entropy_profile"}],
}


def _flagged(out) -> bool:
    return bool(out.get("truncated")) and any(
        "vector" in note and "output_dir" in note for note in out.get("notes", [])
    )


def test_e11_subsampled_vector_curves_are_flagged():
    out = api.run_ensemble(PROFILE_SPEC, 2, seed=1)
    entry = out["probes"]["entropy_profile"]
    assert len(entry["mean"]) < 101 and entry["index"][-1] == 100
    assert _flagged(out)


def test_subsampled_depth_grids_are_flagged():
    # the curve is whole (21 sizes), the grid has 17 of them
    sizes = api.run_ensemble(dict(PROFILE_SPEC, num_system=20, checkpoints=2), 2, seed=1)
    entry = sizes["probes"]["entropy_profile"]
    assert "index" not in entry and len(entry["vs_depth"]["index"]) < 21
    assert _flagged(sizes)
    # 13 evaluations, at most 8 in the grid
    depths = api.run_ensemble(
        dict(PROFILE_SPEC, num_system=8, layers=12, checkpoints="every_layer"), 2, seed=1
    )
    assert len(depths["probes"]["entropy_profile"]["vs_depth"]["layers"]) < 13
    assert _flagged(depths)
    # nothing left out: 9 sizes at 3 evaluations
    whole = api.run_ensemble(dict(PROFILE_SPEC, num_system=8, checkpoints=2), 2, seed=1)
    assert "truncated" not in whole and "notes" not in whole


# ---------------------------------------------------------------------------
# A16: the tick limit counts the last tick (review repro e20)
# ---------------------------------------------------------------------------


def _evaluated(num_ticks: int, stride: int) -> int:
    """Ticks run_circuit evaluates: 0, stride, 2 stride, ... and the last one."""
    return len(set(range(0, num_ticks, stride)) | {num_ticks - 1})


@pytest.mark.parametrize(
    "num_ticks", [api.MAX_TICK_ROWS, api.MAX_TICK_ROWS + 1, 2 * api.MAX_TICK_ROWS]
)
def test_e20_tick_limit_counts_the_last_tick(num_ticks):
    text = "TICK\n" * num_ticks
    obs = [{"kind": "entropy", "region": [0]}]
    out = api.run_circuit(text, num_qubits=1, seed=1, observables=obs, observe_ticks=True)
    series = out["tick_observables"]
    stride = series["tick_stride"]
    assert series["num_evaluated"] == _evaluated(num_ticks, stride) <= api.MAX_TICK_ROWS
    if stride > 1:  # the smallest stride that fits, for the default and for a given one
        assert _evaluated(num_ticks, stride - 1) > api.MAX_TICK_ROWS
        with pytest.raises(ValueError, match=f"use tick_stride >= {stride}"):
            api.run_circuit(text, num_qubits=1, observables=obs, observe_ticks=True,
                            tick_stride=stride - 1)  # fmt: skip
