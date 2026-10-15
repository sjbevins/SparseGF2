"""The JSON service layer: validation, bounded output, equivalence with direct library calls."""

import json
import re
import time

import numpy as np
import pytest

import sparsegf2 as sg
from sparsegf2 import api, graphs, mipt
from sparsegf2 import observables as ob

BELL = "H 0\nCX 0 1\nTICK\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"

SMALL_SPEC = {
    "num_system": 10,
    "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 3},
    "layers": 30,
    "measurement_rate": 0.15,
    "reference": "single",
    "checkpoints": "every_layer",
    "probes": [{"kind": "code_dimension"}, {"kind": "half_cut_entropy"}],
    "stop_when_purified": True,
}


def _size(obj) -> int:
    return len(json.dumps(obj))


def _direct_spec(**overrides) -> mipt.MonitoredCircuit:
    params = dict(
        num_system=10,
        edges=graphs.watts_strogatz(10, k=4, beta=0.2, seed=3),
        layers=30,
        measurement_rate=0.15,
        reference="single",
        checkpoints="every_layer",
        probes=[mipt.code_dimension(), mipt.half_cut_entropy()],
        stop_when_purified=True,
    )
    params.update(overrides)
    return mipt.MonitoredCircuit(**params)


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


def test_capabilities_lists_everything_and_fits():
    caps = api.capabilities()
    assert _size(caps) <= api.MAX_OUTPUT_CHARS
    assert caps["version"] == sg.__version__
    assert "CX" in caps["instructions"]["unitary"]
    assert "CLIFFORD_2Q" in caps["instructions"]["clifford_index"]
    assert set(caps["schedules"]) == set(api.SCHEDULES)
    probes = caps["probes"].get("names", caps["probes"])  # names only if the overview is full
    for name in ("code_dimension", "erasure_first_failure", "uncorrectable_region_counts"):
        assert name in probes
    for name in ("entropy", "code_dimension", "random_erasure_statistics", "expectation"):
        assert name in caps["observables"]
    assert "watts_strogatz" in caps["graph_families"]
    assert "num_system" in caps["ensemble_spec"]
    json.dumps(caps)  # plain JSON


def test_capabilities_section_and_unknown_section():
    probes = api.capabilities("probes")
    assert list(probes) == ["probes"]
    assert probes["probes"]["lost_information"]["params"]["region"] == "required"
    assert probes["probes"]["uncorrectable_region_counts"]["params"]["max_size"] == 2
    with pytest.raises(ValueError, match="unknown section"):
        api.capabilities("nope")


def test_schedules_and_references_match_the_engine():
    for schedule in api.SCHEDULES:
        _direct_spec(schedule=schedule, stop_when_purified=False, probes=[])
    for reference in api.REFERENCES:
        _direct_spec(reference=reference, stop_when_purified=False, probes=[])


def test_quickstart_is_markdown():
    text = api.quickstart()
    assert text.startswith("# SparseGF2")
    assert "start_ensemble_job" in text


# ---------------------------------------------------------------------------
# inspect_circuit
# ---------------------------------------------------------------------------


def test_inspect_valid_circuit_counts():
    text = "H 0\nREPEAT 3 {\n  CX 0 1 2 3\n  M 0 1\n  TICK\n}\nCLIFFORD_2Q(7) 1 2\nDETECTOR rec[-1]"
    info = api.inspect_circuit(text)
    assert info["valid"]
    assert info["num_qubits"] == 4
    assert info["num_measurements"] == 6
    assert info["num_detectors"] == 1
    assert info["num_ticks"] == 3
    assert info["repeat_blocks"] == 1
    assert info["instruction_histogram"]["CX"] == 6
    assert info["instruction_histogram"]["M"] == 6
    assert info["extensions"] == ["CLIFFORD_2Q"]
    assert not info["stim_compatible"]
    assert sg.Circuit(info["text"]) == sg.Circuit(text)


def test_inspect_reports_every_bad_line():
    text = "H 0\nFOO 1\nCX 0 0\nREPEAT 2 {\n  M 0\n}\nM(2) 0"
    info = api.inspect_circuit(text)
    assert not info["valid"]
    lines = {e["line"]: e["message"] for e in info["errors"]}
    assert set(lines) == {2, 3, 7}
    assert "FOO" in lines[2]
    assert "differ" in lines[3]


def test_inspect_structural_error_without_line():
    info = api.inspect_circuit("REPEAT 2 {\n  H 0\n")
    assert not info["valid"]
    assert info["errors"][-1]["line"] is None
    assert "REPEAT" in info["errors"][-1]["message"]


def test_inspect_rejects_non_text():
    with pytest.raises(ValueError, match="string"):
        api.inspect_circuit(["H 0"])


def test_inspect_large_circuit_is_bounded():
    text = "\n".join(f"CX {i} {i + 1}" for i in range(3000))
    info = api.inspect_circuit(text)
    assert info["text_truncated"]
    assert _size(info) <= api.MAX_OUTPUT_CHARS


# ---------------------------------------------------------------------------
# run_circuit
# ---------------------------------------------------------------------------


def test_run_single_shot_matches_simulator():
    out = api.run_circuit(BELL, seed=11)
    direct = sg.Simulator(2, seed=11).run(sg.Circuit(BELL))
    assert out["measurements"] == "".join(map(str, direct.measurements))
    assert out["random"] == "".join(str(int(b)) for b in direct.random)
    assert out["detectors"] == "0"
    assert out["num_measurements"] == 2
    assert out["seed"] == 11
    assert out["random_fraction"] == 0.5


def test_run_many_shots_matches_sample():
    out = api.run_circuit(BELL, shots=400, seed=5)
    direct = sg.sample(sg.Circuit(BELL), 400, seed=5)
    assert out["measurement_means"] == pytest.approx(direct.measurements.mean(axis=0), abs=1e-6)
    assert out["detector_rates"] == [0.0]
    assert out["observable_flip_rates"] == pytest.approx(direct.observables.mean(axis=0), abs=1e-6)
    assert 0.3 < out["measurement_mean"] < 0.7


def test_run_fresh_seed_is_reported_and_reproducible():
    first = api.run_circuit("H 0 1 2 3 4 5 6 7\nM 0 1 2 3 4 5 6 7")
    again = api.run_circuit("H 0 1 2 3 4 5 6 7\nM 0 1 2 3 4 5 6 7", seed=first["seed"])
    assert first["measurements"] == again["measurements"]


def test_run_observables_match_library():
    text = "H 0 1 2\nCX 0 3 1 4 2 5\nCX 3 4\nTICK\nM 5"
    specs = [
        {"kind": "entropy", "region": [0, 1]},
        {"kind": "mutual_information", "a": "0:2", "b": [3, 4]},
        {"kind": "code_dimension", "reference": [0, 1, 2]},
        {"kind": "lost_logical_information", "erased": [3], "reference": [0, 1, 2]},
        {"kind": "stabilizers"},
        {"kind": "expectation", "pauli": "Z2*Z5"},
        {"kind": "entropy", "region": [0], "name": "s0"},
    ]
    out = api.run_circuit(text, seed=2, observables=specs)
    sim = sg.Simulator(6, seed=2)
    sim.run(sg.Circuit(text))
    values = out["observables"]
    assert values["entropy"] == sim.entropy([0, 1])
    assert values["mutual_information"] == ob.mutual_information(sim, [0, 1], [3, 4])
    assert values["code_dimension"] == ob.code_dimension(sim, [0, 1, 2])
    assert values["lost_logical_information"] == ob.lost_logical_information(sim, [3], [0, 1, 2])
    assert values["stabilizers"] == [str(p) for p in sim.stabilizers()]
    assert values["expectation"] == sim.peek_pauli("Z2*Z5")
    assert values["s0"] == sim.entropy([0])


def test_run_observe_ticks_gives_time_series():
    text = "H 0\nTICK\nCX 0 1\nTICK\nCX 1 2\nTICK\nM 0\nTICK"
    out = api.run_circuit(
        text, seed=1, observables=[{"kind": "entropy", "region": [0]}], observe_ticks=True
    )
    series = out["tick_observables"]
    assert series["ticks"] == [0, 1, 2, 3]
    assert series["values"]["entropy"] == [0, 1, 1, 0]
    assert out["observables"]["entropy"] == 0


def test_run_observe_ticks_subsamples_long_series():
    text = "H 0\nCX 0 1\n" + "TICK\nSWAP 0 1\n" * 200
    out = api.run_circuit(
        text, seed=1, observables=[{"kind": "entropy", "region": [0]}], observe_ticks=True
    )
    series = out["tick_observables"]
    assert series["num_ticks"] == 200
    assert series["subsampled"]
    assert len(series["ticks"]) <= api.MAX_CHECKPOINTS_SHOWN
    assert series["ticks"][0] == 0 and series["ticks"][-1] == 199


def test_run_erasure_observables():
    # Five-qubit-code-like random encoding: 2 logical qubits held by references.
    text = "H 0 1\nCX 0 2 1 3\n" + "".join(f"CLIFFORD_2Q({k}) {a} {b}\n" for k, a, b in [
        (100, 2, 4), (2000, 3, 5), (9000, 4, 6), (777, 5, 7), (4321, 2, 7), (11000, 3, 6),
    ])  # fmt: skip
    specs = [
        {"kind": "random_erasure_statistics", "system": "2:8", "reference": [0, 1],
         "sizes": [0, 1, 6], "samples": 20},
        {"kind": "contiguous_code_distance", "system": "2:8", "reference": [0, 1]},
        {"kind": "erasure_profile", "system": "2:8", "reference": [0, 1], "samples": 4},
    ]  # fmt: skip
    out = api.run_circuit(text, seed=4, observables=specs)
    stats = out["observables"]["random_erasure_statistics"]
    assert stats["sizes"] == [0, 1, 6]
    assert stats["correctable_fraction"][0] == 1.0
    assert stats["correctable_fraction"][2] == 0.0
    profile = out["observables"]["erasure_profile"]["correctable_fraction"]["values"]
    assert len(profile) == 7 and profile[0] == 1.0 and profile[-1] == 0.0
    sim = sg.Simulator(8, seed=4)
    sim.run(sg.Circuit(text))
    assert out["observables"]["contiguous_code_distance"] == ob.contiguous_code_distance(
        sim, range(2, 8), [0, 1]
    )


def test_run_phase_free_reports_only_random_flags():
    out = api.run_circuit(BELL, seed=3, phases=False)
    assert "measurements" not in out and "detectors" not in out
    assert out["random"] == "10"
    assert any("phases=false" in note for note in out["notes"])
    with pytest.raises(ValueError, match="phases"):
        api.run_circuit(BELL, phases=False, observables=[{"kind": "expectation", "pauli": "Z0"}])


def test_run_large_record_is_summarized_and_saved(tmp_path):
    text = "H " + " ".join(map(str, range(300))) + "\nM " + " ".join(map(str, range(300)))
    out = api.run_circuit(text, seed=9, output_dir=str(tmp_path / "run"))
    assert "measurements" not in out
    assert _size(out) <= api.MAX_OUTPUT_CHARS
    saved = np.load(out["files"]["arrays"])
    direct = sg.Simulator(300, seed=9).run(sg.Circuit(text))
    assert np.array_equal(saved["measurements"][0], direct.measurements)
    assert (tmp_path / "run" / "circuit.stim").exists()
    prov = json.loads((tmp_path / "run" / "provenance.json").read_text(encoding="utf-8"))
    assert prov["seed"] == 9 and prov["sparsegf2"] == sg.__version__


def test_run_many_detectors_bounded():
    n = 200
    text = "\n".join(
        ["M " + " ".join(map(str, range(n)))] + [f"DETECTOR rec[-{i + 1}]" for i in range(n)]
    )
    out = api.run_circuit(text, shots=50, seed=1)
    assert "detector_rates" not in out
    assert out["detector_rate_max"] == 0.0
    assert _size(out) <= api.MAX_OUTPUT_CHARS


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"circuit": "H 0\nBOGUS 1"}, "line 2"),
        ({"circuit": 5}, "string"),
        ({"circuit": BELL, "shots": 0}, "shots must be at least 1"),
        ({"circuit": BELL, "seed": -1}, "seed"),
        ({"circuit": BELL, "num_qubits": 1}, "smaller"),
        ({"circuit": BELL, "observables": {"kind": "entropy"}}, "list"),
        ({"circuit": BELL, "observables": [{"kind": "volume"}]}, "unknown observable kind"),
        ({"circuit": BELL, "observables": [{"kind": "entropy"}]}, "missing required"),
        ({"circuit": BELL, "observables": [{"kind": "entropy", "region": [5]}]}, "out of range"),
        (
            {"circuit": BELL, "observables": [{"kind": "entropy", "region": [0], "qubits": [1]}]},
            "unknown key",
        ),
        ({"circuit": BELL, "observables": [{"kind": "entropy", "region": "x"}]}, "range like"),
        ({"circuit": BELL, "observe_ticks": True}, "observe_ticks needs"),
    ],
)
def test_run_validation_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        api.run_circuit(**kwargs)


# ---------------------------------------------------------------------------
# ensembles
# ---------------------------------------------------------------------------


def test_run_ensemble_matches_library():
    out = api.run_ensemble(SMALL_SPEC, 60, seed=7)
    direct = _direct_spec().run(60, seed=7)
    assert out["num_trajectories"] == 60
    assert out["seed"] == 7
    mean, err = sg.analysis.mean_and_error(direct.values["code_dimension"])
    assert out["probes"]["code_dimension"]["mean"] == pytest.approx(mean, abs=1e-5, nan_ok=True)
    tau = direct.purification_layer
    pur = out["purification"]
    assert pur["purified"] == int((tau >= 0).sum())
    assert pur["censored_fraction"] == pytest.approx((tau < 0).mean(), abs=1e-6)
    if (tau >= 0).any():
        assert pur["tau_median_purified"] == pytest.approx(np.median(tau[tau >= 0]), abs=1e-6)
    layers, surviving = direct.survival()
    assert pur["survival"]["fraction_unpurified"] == pytest.approx(surviving, abs=1e-6)
    assert out["checkpoints"]["layers"] == layers.tolist()
    assert _size(out) <= api.MAX_OUTPUT_CHARS


def test_run_ensemble_explicit_edges_and_gate_names():
    spec = dict(SMALL_SPEC, probes=[], stop_when_purified=False, checkpoints="final")
    spec.pop("graph")
    spec["edges"] = graphs.cycle(10).tolist()
    spec["gate_set"] = ["CX", "SWAP", "ISWAP"]
    out = api.run_ensemble(spec, 5, seed=1)
    direct = mipt.MonitoredCircuit(
        num_system=10,
        edges=graphs.cycle(10),
        layers=30,
        measurement_rate=0.15,
        reference="single",
        gate_set=[sg.gate(g).clifford.index() for g in ("CX", "SWAP", "ISWAP")],
    ).run(5, seed=1)
    assert out["purification"]["purified"] == int((direct.purification_layer >= 0).sum())


def test_run_ensemble_subsamples_checkpoints_and_vector_probes():
    spec = {
        "num_system": 80,
        "graph": {"family": "cycle"},
        "layers": 120,
        "measurement_rate": 0.2,
        "reference": "purification",
        "checkpoints": "every_layer",
        "probes": [{"kind": "entropy_profile"}, {"kind": "code_dimension"}],
    }
    out = api.run_ensemble(spec, 2, seed=3)
    assert out["checkpoints"]["subsampled"]
    assert out["checkpoints"]["num_total"] == 121
    assert len(out["probes"]["code_dimension"]["mean"]) <= api.MAX_CHECKPOINTS_SHOWN
    profile = out["probes"]["entropy_profile"]
    assert profile["vector"] and profile["at_layer"] == 120
    assert len(profile["mean"]) <= api.MAX_PROFILE_SHOWN
    assert profile["index"][-1] == 80
    assert _size(out) <= api.MAX_OUTPUT_CHARS


def test_run_ensemble_power_law_weights():
    spec = {
        "num_system": 12,
        "graph": {"family": "complete"},
        "edge_weights": {"family": "power_law", "alpha": 2.0},
        "schedule": "random_edges",
        "layers": 10,
        "measurement_rate": 0.1,
        "probes": [{"kind": "half_cut_entropy"}],
    }
    out = api.run_ensemble(spec, 4, seed=2)
    edges = graphs.complete(12)
    direct = mipt.MonitoredCircuit(
        num_system=12,
        edges=edges,
        edge_weights=graphs.power_law_weights(edges, 12, 2.0),
        schedule="random_edges",
        layers=10,
        measurement_rate=0.1,
        probes=[mipt.half_cut_entropy()],
    ).run(4, seed=2)
    assert out["probes"]["half_cut_entropy"]["mean"][0] == pytest.approx(
        direct.mean("half_cut_entropy")[0]
    )


def test_run_ensemble_saves_files(tmp_path):
    out = api.run_ensemble(SMALL_SPEC, 20, seed=4, output_dir=str(tmp_path / "e"))
    files = out["files"]
    saved = mipt.EnsembleResult.load(files["result"])
    direct = _direct_spec().run(20, seed=4)
    assert np.array_equal(saved.purification_layer, direct.purification_layer)
    stored = json.loads((tmp_path / "e" / "spec.json").read_text(encoding="utf-8"))
    assert stored == {"spec": SMALL_SPEC, "trajectories": 20, "seed": 4}
    prov = json.loads((tmp_path / "e" / "provenance.json").read_text(encoding="utf-8"))
    for key in ("python", "numpy", "numba", "sparsegf2", "timestamp", "command", "seed"):
        assert key in prov
    assert api.load_results(str(tmp_path / "e"))["purification"] == out["purification"]
    assert api.load_results(files["result"])["num_trajectories"] == 20
    with pytest.raises(ValueError, match="already holds a run"):
        api.run_ensemble(SMALL_SPEC, 20, seed=4, output_dir=str(tmp_path / "e"))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"num_sytem": 10}, "unknown key"),
        ({"graph": {"family": "torus"}}, "unknown graph family"),
        ({"graph": {"family": "watts_strogatz", "k": 4, "beta": 0.1}}, "own 'seed'"),
        ({"graph": {"family": "cycle", "k": 2}}, "unknown key"),
        ({"graph": {"family": "lattice_2d", "rows": 3, "cols": 4}}, "rows \\* cols"),
        ({"probes": [{"kind": "volume"}]}, "unknown probe kind"),
        ({"probes": [{"kind": "entropy"}]}, "missing required"),
        ({"probes": [{"kind": "code_dimension", "size": 3}]}, "unknown key"),
        ({"reference": "none"}, "invalid spec"),
        ({"schedule": "spiral"}, "schedule"),
        ({"measurement_rate": 2.0}, "measurement_rate"),
        ({"gate_set": ["H"]}, "two-qubit"),
        ({"edges": [[0, 1]]}, "exactly one"),
    ],
)
def test_ensemble_spec_validation(change, message):
    spec = dict(SMALL_SPEC, **change)
    with pytest.raises(ValueError, match=message):
        api.run_ensemble(spec, 2, seed=1)


def test_ensemble_spec_missing_keys_and_bad_trajectories():
    with pytest.raises(ValueError, match="missing required"):
        api.run_ensemble({"num_system": 8, "graph": {"family": "cycle"}}, 2)
    with pytest.raises(ValueError, match="trajectories"):
        api.run_ensemble(SMALL_SPEC, 0)
    with pytest.raises(ValueError, match="object"):
        api.run_ensemble([1, 2], 2)


def test_estimate_ensemble():
    small = api.estimate_ensemble(SMALL_SPEC, 10)
    assert small["fits_sync_limit"]
    huge = api.estimate_ensemble(
        {"num_system": 1024, "graph": {"family": "cycle"}, "layers": 8192,
         "measurement_rate": 0.1, "reference": "single"},
        1_000_000,
    )  # fmt: skip
    assert not huge["fits_sync_limit"]
    assert huge["estimated_seconds"] > small["estimated_seconds"]


def test_load_results_errors(tmp_path):
    with pytest.raises(ValueError, match="no such"):
        api.load_results(str(tmp_path / "missing"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no saved results"):
        api.load_results(str(tmp_path / "empty"))
    np.savez(tmp_path / "other.npz", x=np.zeros(3))
    with pytest.raises(ValueError, match="not a saved SparseGF2 ensemble result"):
        api.load_results(str(tmp_path / "other.npz"))
    (tmp_path / "text.txt").write_text("root:x:0:0:root:/root:/bin/bash\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a saved SparseGF2 ensemble result"):
        api.load_results(str(tmp_path / "text.txt"))


# ---------------------------------------------------------------------------
# background jobs
# ---------------------------------------------------------------------------


def test_job_lifecycle_matches_single_run(tmp_path):
    out = tmp_path / "job"
    status = api.start_ensemble_job(SMALL_SPEC, 50, output_dir=str(out), seed=6, chunk_size=15)
    assert status["status"] in ("queued", "running", "completed")
    final = api.job_status(status["job_id"], wait_seconds=60)
    assert final["status"] == "completed", final
    assert final["trajectories_done"] == 50 and final["progress"] == 1.0
    assert (out / "result.npz").exists() and not (out / "chunks").exists()
    saved = mipt.EnsembleResult.load(out / "result.npz")
    direct = _direct_spec().run(50, seed=6)
    assert np.array_equal(saved.trajectories, np.arange(50))
    assert np.array_equal(saved.purification_layer, direct.purification_layer)
    assert np.array_equal(
        saved.values["code_dimension"], direct.values["code_dimension"], equal_nan=True
    )
    assert status["job_id"] in {j["job_id"] for j in api.list_jobs()["jobs"]}

    # Extending the finished run continues from trajectory 50 with the stored seed.
    more = api.start_ensemble_job(SMALL_SPEC, 80, output_dir=str(out), chunk_size=20)
    assert more["resumed_from"] == 50
    assert api.job_status(more["job_id"], wait_seconds=60)["status"] == "completed"
    extended = mipt.EnsembleResult.load(out / "result.npz")
    assert np.array_equal(
        extended.purification_layer, _direct_spec().run(80, seed=6).purification_layer
    )


def test_job_cancel_and_resume(tmp_path):
    out = tmp_path / "cancel"
    spec = dict(SMALL_SPEC, num_system=24, layers=200, stop_when_purified=False,
                graph={"family": "cycle"}, measurement_rate=0.05)  # fmt: skip
    status = api.start_ensemble_job(spec, 2000, output_dir=str(out), seed=2, chunk_size=1)
    deadline = time.monotonic() + 60
    while api.job_status(status["job_id"])["trajectories_done"] == 0:
        assert time.monotonic() < deadline
        time.sleep(0.005)
    cancelled = api.cancel_job(status["job_id"])
    assert cancelled["job_id"] == status["job_id"]
    final = api.job_status(status["job_id"], wait_seconds=60)
    assert final["status"] == "cancelled"
    assert 0 < final["trajectories_done"] < 2000
    partial = api.load_results(str(out))
    assert partial["num_trajectories"] == final["trajectories_done"]

    resumed = api.start_ensemble_job(spec, 30, output_dir=str(out), chunk_size=10)
    assert resumed["resumed_from"] == final["trajectories_done"]
    done = api.job_status(resumed["job_id"], wait_seconds=60)
    assert done["status"] == "completed"
    total = max(30, final["trajectories_done"])
    saved = mipt.EnsembleResult.load(out / "result.npz")
    assert saved.num_trajectories == total
    direct = api._build_spec(spec).run(total, seed=2)
    assert np.array_equal(
        saved.values["half_cut_entropy"], direct.values["half_cut_entropy"], equal_nan=True
    )


def test_job_errors(tmp_path):
    with pytest.raises(ValueError, match="output_dir"):
        api.start_ensemble_job(SMALL_SPEC, 10, output_dir=None)
    with pytest.raises(ValueError, match="unknown job_id"):
        api.job_status("nope")
    with pytest.raises(ValueError, match="unknown job_id"):
        api.cancel_job("nope")
    out = tmp_path / "j"
    status = api.start_ensemble_job(SMALL_SPEC, 10, output_dir=str(out), seed=1)
    api.job_status(status["job_id"], wait_seconds=60)
    with pytest.raises(ValueError, match="different spec"):
        api.start_ensemble_job(dict(SMALL_SPEC, layers=31), 10, output_dir=str(out))
    with pytest.raises(ValueError, match="seed"):
        api.start_ensemble_job(SMALL_SPEC, 10, output_dir=str(out), seed=2)
    with pytest.raises(ValueError, match="wait_seconds"):
        api.job_status(status["job_id"], wait_seconds=1000)


# ---------------------------------------------------------------------------
# regressions: bounded output, validation, robustness
# ---------------------------------------------------------------------------


def test_large_probe_regions_get_short_keys(tmp_path):
    n = 1200
    spec = {
        "num_system": n,
        "graph": {"family": "cycle"},
        "layers": 1,
        "measurement_rate": 0.1,
        "probes": [
            {"kind": "entropy", "region": "0:600"},
            {"kind": "mutual_information", "a": "0:300", "b": "600:900"},
            {"kind": "entropy", "region": "0:1200:2"},
        ],
    }
    out = api.run_ensemble(spec, 2, seed=1, output_dir=str(tmp_path / "big"))
    assert set(out["probes"]) == {
        "entropy[0:600]",
        "mutual_information[[0:300], [600:900]]",
        "entropy[0:1200:2]",
    }
    assert out["seed"] == 1 and out["num_trajectories"] == 2
    assert "files" in out and not out.get("truncated")
    assert _size(out) <= api.MAX_OUTPUT_CHARS
    direct = mipt.MonitoredCircuit(
        num_system=n, edges=graphs.cycle(n), layers=1, measurement_rate=0.1,
        probes=[mipt.entropy(range(0, n, 2))],
    ).run(2, seed=1)  # fmt: skip
    assert out["probes"]["entropy[0:1200:2]"]["mean"] == pytest.approx(
        direct.mean(f"entropy{list(range(0, n, 2))}")
    )


def test_bounded_last_resort_keeps_scalars_and_files():
    huge = {
        "seed": 3,
        "num_trajectories": 10,
        "files": {"result": "/x/result.npz"},
        "probes": {"x" * 9000: {"mean": [1.0]}},
        "blob": ["y" * 5000] * 5,
    }
    out = api._bounded(huge)
    assert out["truncated"] and out["seed"] == 3 and out["num_trajectories"] == 10
    assert out["files"] == {"result": "/x/result.npz"}
    assert _size(out) <= api.MAX_OUTPUT_CHARS


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"stop_when_purified": "false"}, "stop_when_purified must be true or false"),
        ({"random_orientation": "yes"}, "random_orientation must be true or false"),
        ({"periodic": "no"}, "periodic must be true or false"),
        ({"schedule": "random_edges", "gates_per_layer": 2.5}, "gates_per_layer must be"),
        ({"scramble_layers": 1.5}, "scramble_layers must be an integer"),
        ({"measurement_rate": True}, "measurement_rate must be a number"),
        ({"measurement_rate": "0.1"}, "measurement_rate must be a number"),
        ({"reference_partner": "3"}, "reference_partner must be"),
        ({"graph": {"family": "watts_strogatz", "k": "x", "beta": 0.1, "seed": 1}}, "k must be"),
        ({"graph": {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": -1}}, "seed"),
        ({"checkpoints": 4.5}, "checkpoints"),
        ({"checkpoints": None}, "checkpoints must be"),
        ({"probes": [{"kind": "uncorrectable_region_counts", "max_size": "2"}]}, "max_size must"),
        ({"probes": [{"kind": "code_dimension", "checkpoints": 2.5}]}, "checkpoints"),
        ({"probes": [{"kind": "code_dimension", "name": ""}]}, "name must be"),
        ({"schedule": 3}, "schedule must be a string"),
    ],
)
def test_ensemble_spec_types_are_checked(change, message):
    spec = dict(SMALL_SPEC, **change)
    with pytest.raises(ValueError, match=message):
        api.run_ensemble(spec, 2, seed=1)


def test_ensemble_spec_accepts_whole_number_floats():
    out = api.run_ensemble(
        dict(SMALL_SPEC, checkpoints=10.0, reference_partner=3.0, layers=30.0), 3, seed=1
    )
    assert out["checkpoints"]["layers"] == [0, 10, 20, 30]


def test_every_probe_accepts_a_name():
    caps = api.capabilities("probes")["probes"]
    spec = dict(SMALL_SPEC, reference="purification", stop_when_purified=False, layers=4)
    spec["checkpoints"] = "final"
    probes = []
    for kind, info in caps.items():
        assert "name" in info["params"]
        args = {k: v for k, v in info["params"].items() if v == "required"}
        params = {"region": [0, 1], "a": [0, 1], "b": [2, 3], "size": 2}
        probes.append({"kind": kind, "name": f"my_{kind}", **{k: params[k] for k in args}})
    out = api.run_ensemble(dict(spec, probes=probes), 2, seed=1)
    assert set(out["probes"]) == {f"my_{kind}" for kind in caps}


def test_vector_probe_with_stop_when_purified_has_data_and_depth_grid():
    spec = dict(
        SMALL_SPEC,
        probes=[{"kind": "entropy_profile"}, {"kind": "erasure_lost_profile", "samples": 3}],
    )
    out = api.run_ensemble(spec, 40, seed=2)
    direct = _direct_spec(
        probes=[mipt.entropy_profile(), mipt.erasure_lost_profile(samples=3)]
    ).run(40, seed=2)
    for name in ("entropy_profile", "erasure_lost_profile"):
        entry = out["probes"][name]
        values = direct.values[name]
        alive = ~np.isnan(values[:, :, 0])  # the lost profile is exact (0) after a stop
        assert alive.all() == (name == "erasure_lost_profile")
        last = np.flatnonzero(alive.any(axis=0))[-1]
        assert entry["at_layer"] == int(direct.checkpoint_layers[last])
        assert entry["count"] == int(alive[:, last].sum())
        assert entry["mean"] == pytest.approx(np.nanmean(values[:, last, :], axis=0), abs=1e-5)
        grid = entry["vs_depth"]
        assert grid["layers"][0] == 0 and len(grid["layers"]) <= api.MAX_VECTOR_DEPTHS
        assert grid["mean"][0] == pytest.approx(
            np.nanmean(values[:, 0, grid["index"]], axis=0), abs=1e-5
        )
    assert _size(out) <= api.MAX_OUTPUT_CHARS


def test_erasure_probe_summaries_give_failure_by_size():
    from sparsegf2 import analysis

    spec = dict(
        SMALL_SPEC,
        reference="purification",
        stop_when_purified=False,
        checkpoints=10,
        probes=[
            {"kind": "erasure_first_failure", "samples": 20},
            {"kind": "uncorrectable_region_counts", "max_size": 2, "checkpoints": [5, 30]},
            {"kind": "code_snapshot", "checkpoints": "final"},
        ],
    )
    out = api.run_ensemble(spec, 12, seed=3)
    direct = api._build_spec(spec).run(12, seed=3)
    first = out["probes"]["erasure_first_failure"]
    expected = analysis.failure_by_size(direct.values["erasure_first_failure"][:, -1], 10).mean(0)
    assert first["mean"] == pytest.approx(expected, abs=1e-5) and first["at_layer"] == 30
    assert "uncorrectable" in first["what"]
    counts = out["probes"]["uncorrectable_region_counts[2]"]
    exact = analysis.exact_failure_by_size(
        direct.values["uncorrectable_region_counts[2]"][:, -1], 10
    )
    assert counts["mean"] == pytest.approx([0.0, *exact.mean(0)], abs=1e-5)
    assert counts["vs_depth"]["layers"] == [5, 30]
    snap = out["probes"]["code_snapshot"]
    assert snap["layers"] == [30] and any("code_snapshot" in note for note in out["notes"])


def test_ensemble_result_memory_is_bounded():
    spec = dict(SMALL_SPEC, probes=[{"kind": "erasure_first_failure", "samples": 100000}])
    est = api.estimate_ensemble(spec, 10)
    assert est["result_megabytes"] == pytest.approx(10 * 8 * (31 * 100000 + 4) / 1e6, rel=1e-4)
    with pytest.raises(ValueError, match="GB of memory"):
        api.run_ensemble(spec, 5000, seed=1)


def test_inspect_and_run_catch_bad_record_lookbacks():
    info = api.inspect_circuit("M 0\nDETECTOR rec[-2]")
    assert not info["valid"] and info["errors"][0]["line"] == 2
    info = api.inspect_circuit("M 0\nREPEAT 3 {\n  M 1\n  DETECTOR rec[-1] rec[-3]\n}")
    assert not info["valid"] and info["errors"][0]["line"] == 4
    assert api.inspect_circuit("M 0\nREPEAT 3 {\n  M 1\n  DETECTOR rec[-1] rec[-2]\n}")["valid"]
    with pytest.raises(ValueError, match="invalid circuit: line 2: rec\\[-2\\]"):
        api.run_circuit("M 0\nDETECTOR rec[-2]")


def test_huge_repeat_counts_are_fast_and_refused_to_run():
    text = "REPEAT 1000000000 {\n  H 0\n  TICK\n  OBSERVABLE_INCLUDE(2) rec[-1]\n  M 0\n}"
    start = time.perf_counter()
    info = api.inspect_circuit(text)
    assert time.perf_counter() - start < 1.0
    assert not info["valid"] and "operations" in info["errors"][-1]["message"]
    with pytest.raises(ValueError, match="operations"):
        api.run_circuit("REPEAT 1000000000 {\n  H 0\n  TICK\n}")
    ok = api.inspect_circuit("M 0\nREPEAT 100000 {\n  OBSERVABLE_INCLUDE(2) rec[-1]\n  TICK\n}")
    assert ok["valid"] and ok["num_observables"] == 3 and ok["num_ticks"] == 100000


def test_observe_ticks_stride():
    text = "H 0\nCX 0 1\n" + "TICK\nSWAP 0 1\n" * 30
    obs = [{"kind": "entropy", "region": [0]}]
    out = api.run_circuit(text, seed=1, observables=obs, observe_ticks=True, tick_stride=7)
    series = out["tick_observables"]
    assert series["ticks"] == [0, 7, 14, 21, 28, 29] and series["tick_stride"] == 7
    with pytest.raises(ValueError, match="tick_stride only applies"):
        api.run_circuit(text, observables=obs, tick_stride=2)
    long = "TICK\n" * (api.MAX_TICK_ROWS * 2 + 10)
    out = api.run_circuit(long, observables=obs, observe_ticks=True, num_qubits=1)
    assert out["tick_observables"]["num_evaluated"] <= api.MAX_TICK_ROWS
    with pytest.raises(ValueError, match="tick_stride >= 3"):
        api.run_circuit(long, observables=obs, observe_ticks=True, num_qubits=1, tick_stride=1)


def test_run_circuit_flags_every_cut():
    text = "H 0\n" + "M 0\n" * 300 + "OBSERVABLE_INCLUDE(300) rec[-1]"
    single = api.run_circuit(text, seed=1)
    assert single["truncated"] and "measurements" not in single
    assert single["num_ones"] == round(single["measurement_mean"] * 300)
    assert any("observable_flips" in note for note in single["notes"])
    many = api.run_circuit(text, shots=8, seed=1)
    assert many["truncated"] and len(many["observable_flip_rates"]) == api.MAX_LIST_SHOWN
    assert any("per-measurement means" in note for note in many["notes"])


def test_inspect_flags_a_cut_histogram():
    lines = []
    for g in sg.all_gates():
        if g.kind in ("unitary", "measure", "measure_reset", "reset", "measure_pair"):
            lines.append(f"{g.name} 0" + (" 1" if g.num_qubits == 2 else ""))
        elif g.name in ("X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1"):
            lines.append(f"{g.name}(0.1) 0")
    lines += ["DEPOLARIZE2(0.1) 0 1", "CLIFFORD_1Q(3) 0", "CLIFFORD_2Q(5) 0 1", "MPP X0*Z1"]
    info = api.inspect_circuit("\n".join(lines))
    assert info["valid"] and len(lines) > api.MAX_LIST_SHOWN
    assert info["truncated"] and len(info["instruction_histogram"]) == api.MAX_LIST_SHOWN
    assert sum(info["kind_totals"].values()) == len(lines)


def test_load_directory_with_chunks_left_over_from_an_interrupted_merge(tmp_path):
    out = tmp_path / "crash"
    status = api.start_ensemble_job(SMALL_SPEC, 30, output_dir=str(out), seed=5, chunk_size=10)
    assert api.job_status(status["job_id"], wait_seconds=60)["status"] == "completed"
    # Recreate the state of a crash between writing result.npz and deleting chunks.
    (out / "chunks").mkdir()
    direct = _direct_spec()
    for first in (0, 10, 20):
        direct.run(10, seed=5, first_trajectory=first).save(
            out / "chunks" / f"traj_{first:012d}.npz"
        )
    loaded = api.load_results(str(out))
    assert loaded["num_trajectories"] == 30
    resumed = api.start_ensemble_job(SMALL_SPEC, 40, output_dir=str(out), chunk_size=10)
    assert resumed["resumed_from"] == 30
    assert api.job_status(resumed["job_id"], wait_seconds=60)["status"] == "completed"
    saved = mipt.EnsembleResult.load(out / "result.npz")
    assert np.array_equal(saved.purification_layer, direct.run(40, seed=5).purification_layer)
    assert not (out / "chunks").exists()


def test_failed_metadata_write_does_not_leave_a_stuck_job(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(api, "_write_run_metadata", fail)
    with pytest.raises(OSError, match="disk full"):
        api.start_ensemble_job(SMALL_SPEC, 10, output_dir=str(tmp_path / "m"), seed=1)
    assert all(j["output_dir"] != str(tmp_path / "m") for j in api.list_jobs()["jobs"])
    monkeypatch.undo()
    status = api.start_ensemble_job(SMALL_SPEC, 10, output_dir=str(tmp_path / "m"), seed=1)
    assert api.job_status(status["job_id"], wait_seconds=60)["status"] == "completed"


def test_estimate_accounts_for_idle_threads():
    import numba

    spec = {"num_system": 64, "graph": {"family": "complete"}, "schedule": "all_edges",
            "layers": 16, "measurement_rate": 0.1}  # fmt: skip
    one = api.estimate_ensemble(spec, 1)["estimated_seconds"]
    many = api.estimate_ensemble(spec, numba.get_num_threads())["estimated_seconds"]
    assert many == pytest.approx(one, rel=1e-3)


def test_graph_realizations_give_a_collection_of_graphs():
    spec = {
        "num_system": 16,
        "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 3, "realizations": 5},
        "circuits_per_graph": 2,
        "layers": 24,
        "measurement_rate": 0.2,
        "schedule": "random_edges",
        "reference": "single",
        "reference_partner": 15,
        "scramble_matchings": 4,
        "checkpoints": "every_layer",
        "stop_when_purified": True,
    }
    mc = api._build_spec(spec)
    assert mc.num_graphs == 5 and mc.num_trajectories == 10
    for i, g in enumerate(mc.graphs):
        np.testing.assert_array_equal(g, graphs.watts_strogatz(16, 4, 0.2, seed=[3, i]))
    direct = mc.run(seed=4)
    summary = api.run_ensemble(spec, seed=4)  # trajectories omitted: all 5 x 2 circuits
    assert summary["num_trajectories"] == 10
    assert summary["purification"]["purified"] == int((direct.purification_layer >= 0).sum())
    assert api.estimate_ensemble(spec)["trajectories"] == 10
    with pytest.raises(ValueError, match="give 10"):
        api.run_ensemble(spec, trajectories=11, seed=4)
    with pytest.raises(ValueError, match="give trajectories"):
        api.run_ensemble({**spec, "graph": {"family": "cycle"}, "circuits_per_graph": 1}, seed=4)
    with pytest.raises(ValueError, match="circuits_per_graph"):
        api._build_spec({**spec, "graph": {"family": "cycle"}})  # one graph: c must be 1
    with pytest.raises(ValueError, match="random family"):
        api._build_spec({**spec, "graph": {"family": "cycle", "realizations": 2}})
    with pytest.raises(ValueError, match="at least 1"):
        api._build_spec({**spec, "graph": {**spec["graph"], "realizations": 0}})


# ---------------------------------------------------------------------------
# regressions from the review
# ---------------------------------------------------------------------------


def _gate_pairs(circuit) -> set[tuple[int, int]]:
    return {
        (ins.targets[2 * j].value, ins.targets[2 * j + 1].value)
        for ins in circuit
        if ins.name == "CLIFFORD_2Q"
        for j in range(len(ins.targets) // 2)
    }


def test_explicit_edge_weights_stay_with_their_edges():
    # weight only on (2, 3), listed before (0, 1): the weights must follow the edges
    spec = {
        "num_system": 4, "edges": [[2, 3], [0, 1]], "layers": 20, "measurement_rate": 0.0,
        "schedule": "random_edges", "edge_weights": [1.0, 0.0], "checkpoints": "final",
        "probes": [{"kind": "entropy", "region": [0], "name": "S0"},
                   {"kind": "entropy", "region": [2], "name": "S2"}],
    }  # fmt: skip
    out = api.run_ensemble(spec, 200, seed=1)
    assert out["probes"]["S0"]["mean"] == [0.0] and out["probes"]["S2"]["mean"][0] > 0
    assert _gate_pairs(api._build_spec(spec).circuit(0, seed=1)) == {(2, 3)}
    # reversed pairs and a ring listed nearest neighbours first, then next-nearest
    n = 8
    edges = [[(i + 1) % n, i] for i in range(n)] + [[i, (i + 2) % n] for i in range(n)]
    ring = {**spec, "num_system": n, "edges": edges, "edge_weights": [1.0] * n + [0.0] * n}
    used = _gate_pairs(api._build_spec(ring).circuit(0, seed=2))
    assert used == {tuple(sorted((i, (i + 1) % n))) for i in range(n)}
    with pytest.raises(ValueError, match="more than once"):
        api._build_spec({**spec, "edges": [[0, 1], [1, 0]]})
    with pytest.raises(ValueError, match="one weight per edge"):
        api._build_spec({**spec, "edge_weights": [1.0]})


def test_list_edge_weights_need_a_known_edge_order():
    base = {"num_system": 8, "layers": 4, "measurement_rate": 0.1, "schedule": "random_edges"}
    ws = {"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 1}
    with pytest.raises(ValueError, match="known edge order"):
        api._build_spec({**base, "graph": ws, "edge_weights": [1.0] * 16})
    with pytest.raises(ValueError, match="known edge order"):
        api._build_spec({**base, "graph": {**ws, "realizations": 3}, "edge_weights": [1.0] * 16})
    # deterministic families have the documented sorted order: weights apply as given
    mc = api._build_spec({**base, "graph": {"family": "cycle"}, "edge_weights": [1.0] * 8})
    np.testing.assert_array_equal(mc.edges, graphs.cycle(8))
    # and weight families work with random graphs and banks
    api._build_spec(
        {**base, "graph": {**ws, "realizations": 2}, "edge_weights": {"family": "power_law",
                                                                   "alpha": 2.0}}
    )  # fmt: skip


def test_saved_spec_is_exact_and_resume_compares_exactly(tmp_path):
    spec = dict(SMALL_SPEC, measurement_rate=1 / 3)
    api.run_ensemble(spec, 20, seed=5, output_dir=str(tmp_path / "a"))
    stored = json.loads((tmp_path / "a" / "spec.json").read_text(encoding="utf-8"))
    assert stored["spec"]["measurement_rate"] == 1 / 3
    api.run_ensemble(
        stored["spec"], stored["trajectories"], seed=stored["seed"], output_dir=str(tmp_path / "b")
    )
    a = mipt.EnsembleResult.load(tmp_path / "a" / "result.npz")
    b = mipt.EnsembleResult.load(tmp_path / "b" / "result.npz")
    assert b.spec.measurement_rate == 1 / 3
    np.testing.assert_array_equal(a.num_measurements, b.num_measurements)
    job = api.start_ensemble_job(
        dict(SMALL_SPEC, measurement_rate=0.12345649), 20, output_dir=str(tmp_path / "j"), seed=3
    )
    api.job_status(job["job_id"], wait_seconds=60)
    with pytest.raises(ValueError, match="different spec"):
        api.start_ensemble_job(
            dict(SMALL_SPEC, measurement_rate=0.12345551), 40, output_dir=str(tmp_path / "j")
        )


def test_resume_keeps_provenance_and_refuses_another_version(tmp_path, monkeypatch):
    out = str(tmp_path / "prov")
    first = api.start_ensemble_job(SMALL_SPEC, 20, output_dir=out, seed=4, command="first")
    api.job_status(first["job_id"], wait_seconds=60)
    more = api.start_ensemble_job(SMALL_SPEC, 40, output_dir=out, command="second")
    api.job_status(more["job_id"], wait_seconds=60)
    prov = json.loads((tmp_path / "prov" / "provenance.json").read_text(encoding="utf-8"))
    assert prov["command"] == "first" and prov["sparsegf2"] == sg.__version__
    assert [(r["command"], r["resumed_from"]) for r in prov["resumes"]] == [("second", 20)]
    monkeypatch.setattr(api, "__version__", "0.0.0.other")
    with pytest.raises(ValueError, match=r"started with sparsegf2 .*need not agree"):
        api.start_ensemble_job(SMALL_SPEC, 60, output_dir=out)


def _rewrite_without_contract(path):
    """Rewrite a saved result as a file from before ``randomness_contract`` was recorded."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["__meta__"]))
    del meta["randomness_contract"]
    arrays["__meta__"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)


def test_resume_refuses_another_randomness_contract(tmp_path):
    api.run_ensemble(SMALL_SPEC, 5, seed=1, output_dir=str(tmp_path / "sync"))
    prov = json.loads((tmp_path / "sync" / "provenance.json").read_text(encoding="utf-8"))
    assert prov["randomness_contract"] == mipt.RANDOMNESS_CONTRACT
    out = tmp_path / "contract"
    job = api.start_ensemble_job(SMALL_SPEC, 20, output_dir=str(out), seed=4)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    prov_path = out / "provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov["randomness_contract"] == mipt.RANDOMNESS_CONTRACT
    refused = re.escape(
        "this run was made under randomness contract 1; this version uses "
        f"{mipt.RANDOMNESS_CONTRACT}; start a new output_dir"
    )
    # a provenance written under contract 1, and one from before the contract was recorded
    old = {k: v for k, v in prov.items() if k != "randomness_contract"}
    for record in ({**prov, "randomness_contract": 1}, old):
        prov_path.write_text(json.dumps(record), encoding="utf-8")
        for trajectories in (20, 40):  # resume (nothing left to do) and extend
            with pytest.raises(ValueError, match=refused):
                api.start_ensemble_job(SMALL_SPEC, trajectories, output_dir=str(out))
        assert api.load_results(str(out))["num_trajectories"] == 20  # reading is still fine
    # results saved under contract 1 are refused even beside a current provenance
    prov_path.write_text(json.dumps(prov), encoding="utf-8")
    _rewrite_without_contract(out / "result.npz")
    with pytest.raises(ValueError, match=refused):
        api.start_ensemble_job(SMALL_SPEC, 40, output_dir=str(out))
    assert not (out / "chunks").exists()  # nothing was run


def test_summaries_show_the_randomness_contract(tmp_path):
    out = tmp_path / "shown"
    summary = api.run_ensemble(SMALL_SPEC, 5, seed=1, output_dir=str(out))
    assert summary["randomness_contract"] == mipt.RANDOMNESS_CONTRACT
    assert api.load_results(str(out))["randomness_contract"] == mipt.RANDOMNESS_CONTRACT
    _rewrite_without_contract(out / "result.npz")  # saved before the contract was recorded
    assert api.load_results(str(out))["randomness_contract"] == 1
    assert api.load_results(str(out / "result.npz"))["randomness_contract"] == 1


def test_estimate_includes_graph_generation_and_times_are_wall_clock():
    one = dict(SMALL_SPEC, schedule="random_edges", checkpoints="final")
    bank = dict(one, graph={**SMALL_SPEC["graph"], "realizations": 2000})
    small, large = api.estimate_ensemble(one, 10), api.estimate_ensemble(bank, 10)
    assert large["estimated_seconds"] > small["estimated_seconds"] + 0.5  # 2000 graphs
    start = time.perf_counter()
    out = api.run_ensemble(bank, 200, seed=1)
    wall = time.perf_counter() - start
    assert wall * 0.8 <= out["elapsed_seconds"] <= wall
    assert out["elapsed_seconds"] <= large["estimated_seconds"] * 1.1 + 0.5
    with pytest.raises(ValueError, match="start_ensemble_job"):
        api.run_ensemble(dict(bank, circuits_per_graph=50_000), 10**8, seed=1, max_seconds=1.0)


def test_phase_free_stabilizers_carry_no_signs():
    free = api.run_circuit(
        "X 0\nH 1\nCX 1 2\nZ 1", phases=False, observables=[{"kind": "stabilizers"}]
    )
    assert free["observables"]["stabilizers"] == ["Z__", "_XX", "_ZZ"]
    assert any("without signs" in note for note in free["notes"])
    signed = api.run_circuit("X 0\nH 1\nCX 1 2\nZ 1", observables=[{"kind": "stabilizers"}])
    assert signed["observables"]["stabilizers"] == ["-Z__", "-_XX", "+_ZZ"]


def test_long_profiles_are_flagged_and_saved_in_full(tmp_path):
    rng = np.random.default_rng(1)
    n = 120
    text = "\n".join(
        f"CLIFFORD_2Q({rng.integers(11520)}) {a} {a + 1}"
        for layer in range(12)
        for a in range(layer % 2, n - 1, 2)
    )
    obs = [{"kind": "entropy_profile"}, {"kind": "stabilizer_length_distribution"},
           {"kind": "erasure_profile", "system": "0:60", "reference": "60:120", "samples": 2}]  # fmt: skip
    out = api.run_circuit(text, seed=1, observables=obs, output_dir=str(tmp_path / "run"))
    assert out["truncated"] and any("profile entries" in note for note in out["notes"])
    shown = out["observables"]["entropy_profile"]
    assert len(shown["values"]) == api.MAX_PROFILE_SHOWN and shown["subsampled"]
    saved = json.loads((tmp_path / "run" / "observables.json").read_text(encoding="utf-8"))["final"]
    sim = sg.Simulator(n, seed=1)
    sim.run(sg.Circuit(text))
    assert saved["entropy_profile"] == ob.entropy_profile(sim).tolist()
    assert len(saved["stabilizer_length_distribution"]) == n + 1
    assert len(saved["erasure_profile"]["correctable_fraction"]) == 61
    short = api.run_circuit("H 0\nCX 0 1", observables=[{"kind": "entropy_profile"}])
    assert short["observables"]["entropy_profile"] == {"values": [0, 1, 0]}
    assert "truncated" not in short


def test_out_of_range_integers_are_input_errors():
    base = {"num_system": 8, "edges": [[0, 1], [1, 2]], "layers": 8, "measurement_rate": 0.2}
    for bad in ({"edges": [[0, 2**70]]}, {"gate_set": [2**70]}, {"checkpoints": [2**70]},
                {"layers": 2**70}, {"scramble_gates": 2**70}):  # fmt: skip
        with pytest.raises(ValueError):
            api.run_ensemble({**base, **bad}, 2, seed=1)
    out = api.run_ensemble(base, 2, seed=2**70)  # seeds may be any size
    assert out["seed"] == 2**70


def test_inspect_reports_anti_hermitian_mpp_with_its_line():
    out = api.inspect_circuit("H 0\nMPP X0*Z0\n")
    assert out["valid"] is False
    assert out["errors"][0]["line"] == 2 and "anti-Hermitian" in out["errors"][0]["message"]
    with pytest.raises(ValueError, match="line 2"):
        api.run_circuit("H 0\nMPP X0*Z0\n", seed=1)


def test_size_cap_keeps_checkpoints_evenly_spaced():
    spec = {
        "num_system": 32, "graph": {"family": "cycle"}, "layers": 100, "measurement_rate": 0.2,
        "reference": "single", "checkpoints": "every_layer", "stop_when_purified": True,
        "probes": [{"kind": "entropy", "region": [i, i + 1], "name": f"S{i}"} for i in range(13)],
    }  # fmt: skip
    out = api.run_ensemble(spec, 100, seed=1)
    assert out["truncated"] and _size(out) <= api.MAX_OUTPUT_CHARS
    layers = out["checkpoints"]["layers"]
    assert layers[0] == 0 and layers[-1] == 100 and len(layers) < 101
    survival = out["purification"]["survival"]
    assert survival["layers"] == layers
    assert all(len(p["mean"]) == len(layers) for p in out["probes"].values())
    result = api._build_spec(spec).run(100, seed=1)
    all_layers, surviving = result.survival()
    np.testing.assert_allclose(
        survival["fraction_unpurified"], surviving[np.searchsorted(all_layers, layers)], atol=1e-6
    )


def test_contiguous_distance_summary_leaves_out_trajectories_without_a_code():
    spec = {
        "num_system": 16, "graph": {"family": "cycle"}, "layers": 40, "measurement_rate": 0.25,
        "reference": "purification", "checkpoints": [10, 20, 40],
        "probes": [{"kind": "code_dimension"}, {"kind": "contiguous_distance"}],
    }  # fmt: skip
    out = api.run_ensemble(spec, 200, seed=1)
    entry = out["probes"]["contiguous_distance"]
    assert min(entry["mean"]) >= 1 and "count" in entry
    assert any("contiguous_distance" in note for note in out["notes"])


def test_summary_of_dense_probe_schedules_is_flagged_and_keeps_every_probe():
    spec = dict(
        SMALL_SPEC,
        num_system=64,
        graph={"family": "cycle"},
        layers=200,
        reference="purification",
        stop_when_purified=False,
        checkpoints="final",
        probes=[{"kind": "entropy", "region": [i, i + 1], "checkpoints": "every_layer"}
                for i in range(0, 20, 2)]
        + [{"kind": "code_dimension", "checkpoints": "every_layer"},
           {"kind": "erasure_first_failure", "samples": 10, "checkpoints": 3},
           {"kind": "uncorrectable_region_counts", "checkpoints": 3},
           {"kind": "entropy_profile", "checkpoints": 3}],
    )  # fmt: skip
    out = api.run_ensemble(spec, 8, seed=1)
    assert _size(out) <= api.MAX_OUTPUT_CHARS
    assert len(out["probes"]) == 14  # nothing cut: fewer evaluations are shown instead
    assert out["truncated"] and any("evaluations per probe" in n for n in out["notes"])


def test_huge_exhaustive_sizes_give_a_huge_estimate_not_an_error():
    spec = dict(
        SMALL_SPEC,
        num_system=600,
        graph={"family": "cycle"},
        reference="purification",
        probes=[{"kind": "uncorrectable_region_counts", "max_size": 520}],
    )
    est = api.estimate_ensemble(spec, 1)
    assert not est["fits_sync_limit"] and est["estimated_seconds"] > 1e100
