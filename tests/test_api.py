"""The JSON service layer: validation, bounded output, equivalence with direct library calls."""

import errno
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

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
    assert not (out / ".lock").exists()  # a cancelled job lets the next writer in
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
    samples = direct.values["erasure_first_failure"][:, -1]
    code = samples.min(axis=1) <= 10  # the curve averages the trajectories that hold a code
    expected = analysis.failure_by_size(samples[code], 10).mean(0)
    assert first["mean"] == pytest.approx(expected, abs=1e-5) and first["at_layer"] == 30
    assert first["count_with_code"] == code.sum() and "uncorrectable" in first["what"]
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
    with pytest.raises(ValueError, match="is not writable: disk full"):
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


# ---------------------------------------------------------------------------
# the exhaustive cutoff m*(n) in JSON: a budget, or one value per system size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "cutoff", "w"),
    [
        # m*(n) = max{w : sum_{m=1}^{w} C(n, m) <= B}; the sums: n = 64: 64, 2,080, 43,744,
        # 679,120; n = 128: 128, 8,256, 349,632; n = 256: 256, 32,896, 2,796,416
        (64, {"budget": 50000}, 3),
        (256, {"budget": 50000}, 2),
        (64, {"max_size": {"64": 3, "128": 2}}, 3),
        (128, {"max_size": {"64": 3, "128": 2}}, 2),
        (64, {"budget": {"64": 20000, "256": 50000}}, 2),
        (256, {"budget": {"64": 20000, "256": 50000}}, 2),
    ],
)
def test_json_cutoff_forms_resolve_per_system_size(n, cutoff, w):
    spec = dict(
        SMALL_SPEC,
        num_system=n,
        graph={"family": "cycle"},
        reference="purification",
        stop_when_purified=False,
        probes=[{"kind": "uncorrectable_region_counts", **cutoff}],
    )
    probe = api._build_spec(spec).probes[0]
    assert (probe.name, probe.size) == (f"uncorrectable_region_counts[{w}]", w)
    explicit = dict(spec, probes=[{"kind": "uncorrectable_region_counts", "max_size": w}])
    assert api.estimate_ensemble(spec, 100) == api.estimate_ensemble(explicit, 100)


def test_json_cutoff_forms_run_like_their_max_size():
    base = dict(SMALL_SPEC, reference="purification", stop_when_purified=False, checkpoints=10)
    explicit = {
        w: api.run_ensemble(
            dict(base, probes=[{"kind": "uncorrectable_region_counts", "max_size": w}]), 6, seed=2
        )["probes"]
        for w in (1, 2, 3)
    }
    # n = 10: C(10, 1) = 10, then 55, then 175 regions
    for cutoff, w in (
        ({"budget": 100}, 2),
        ({"max_size": {"10": 3, "64": 2}}, 3),
        ({"budget": {"10": 20, "64": 20000}}, 1),
    ):
        probe = {"kind": "uncorrectable_region_counts", **cutoff}
        out = api.run_ensemble(dict(base, probes=[probe]), 6, seed=2)
        assert out["probes"] == explicit[w], cutoff
    named = {"kind": "uncorrectable_region_counts", "budget": {"10": 100}, "name": "counts"}
    out = api.run_ensemble(dict(base, probes=[named]), 6, seed=2)
    assert out["probes"] == {"counts": explicit[2]["uncorrectable_region_counts[2]"]}


RESOLVING = r"probes\[0\] \(uncorrectable_region_counts\): "  # errors at num_system = 10


@pytest.mark.parametrize(
    ("cutoff", "message"),
    [
        (
            {"max_size": {"64": 3, "128": 2}},
            RESOLVING
            + r"max_size has no entry for num_system = 10; it covers the system sizes \[64, 128\]",
        ),
        (
            {"budget": 5},
            RESOLVING
            + re.escape(
                "budget 5 allows no exhaustive region size at n = 10 (C(10, 1) = 10 > 5); "
                "raise the budget or drop the probe"
            ),
        ),
        ({"max_size": 11}, RESOLVING + r"max_size must be in \[1, 10\], got 11"),
        (  # an entry below its own n fails at once, whatever num_system is
            {"budget": {"10": 50000, "64": 50}},
            re.escape("budget 50 allows no exhaustive region size at n = 64 (C(64, 1) = 64 > 50)"),
        ),
        ({"max_size": 2, "budget": 100}, "give max_size or budget, not both"),
        ({"max_size": {"ten": 3}}, r"probes\[0\]\.max_size: .*integers like \"64\", got 'ten'"),
        ({"budget": {"10.5": 100}}, r"probes\[0\]\.budget: .*integers like \"64\", got '10.5'"),
        (
            {"budget": {"-5": 100}},
            r"probes\[0\]\.budget: .*positive integers like \"64\", got '-5'",
        ),
        (
            {"max_size": {"10": 2, "010": 3}},
            r"probes\[0\]\.max_size gives the system size 10 twice",
        ),
        ({"max_size": {"10": 2.5}}, r"max_size\[10\] must be an integer, got 2.5"),
        ({"max_size": {"10": 11}}, r"max_size\[10\] must be in \[1, 10\], got 11"),
        ({"budget": {"10": "100"}}, r"budget\[10\] must be an integer, got '100'"),
        ({"budget": {"10": True}}, r"budget\[10\] must be an integer, got True"),
        ({"budget": -1}, "budget must be non-negative"),
        ({"budget": 2.5}, "budget must be an integer"),
        ({"max_size": [2, 3]}, "max_size must be an integer"),
        ({"max_size": {}}, "max_size: give at least one system size"),
    ],
)
def test_json_cutoff_errors_name_the_field(cutoff, message):
    spec = dict(SMALL_SPEC, probes=[{"kind": "uncorrectable_region_counts", **cutoff}])
    with pytest.raises(ValueError, match=message) as info:
        api.run_ensemble(spec, 2, seed=1)
    assert str(info.value).startswith("probes[0]")  # which probe, like every spec error


@pytest.mark.parametrize("extra", [{"budget": {"x": 1}}, {"max_size": 2}, {"region": "bad"}])
def test_probe_keys_are_checked_before_their_values(extra):
    spec = dict(SMALL_SPEC, probes=[{"kind": "code_dimension", **extra}])
    key = re.escape(str([next(iter(extra))]))
    with pytest.raises(
        ValueError, match=rf"probes\[0\] \(code_dimension\): unknown key\(s\) {key}"
    ):
        api.run_ensemble(spec, 2, seed=1)


def test_capabilities_document_the_cutoff_forms():
    entry = api.capabilities("probes")["probes"]["uncorrectable_region_counts"]
    assert entry["params"]["max_size"] == 2 and entry["params"]["budget"] is None
    for form in (
        {"budget": 50000},
        {"max_size": {"64": 3, "128": 2}},
        {"budget": {"64": 20000, "256": 50000}},
    ):
        assert form in entry["forms"]
    assert "sum_{m=1}^{w} C(n, m) <= B" in entry["note"]


# ---------------------------------------------------------------------------
# resume and extend: the same run, checked before anything runs
# ---------------------------------------------------------------------------


def _file_spec(path) -> dict:
    return {"num_system": 12, "graph": {"family": "file", "path": str(path)}, "layers": 10,
            "measurement_rate": 0.2, "reference": "single", "checkpoints": 5,
            "probes": [{"kind": "code_dimension"}]}  # fmt: skip


def _direct_file_run(edges: np.ndarray, trajectories: int, seed: int) -> mipt.EnsembleResult:
    return mipt.MonitoredCircuit(
        num_system=12, edges=edges, layers=10, measurement_rate=0.2, reference="single",
        checkpoints=5, probes=[mipt.code_dimension()],
    ).run(trajectories, seed=seed)  # fmt: skip


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_resume_refuses_a_changed_graph_file_before_running(tmp_path):
    n, gfile = 12, tmp_path / "ws.npz"
    first_graph = graphs.watts_strogatz(n, 4, 0.1, seed=1)
    graphs.save(gfile, [first_graph], n)
    original = gfile.read_bytes()
    spec, out = _file_spec(gfile), tmp_path / "run"
    job = api.start_ensemble_job(spec, 20, output_dir=str(out), seed=3, chunk_size=10)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    meta_path = out / "spec.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["spec"] == spec  # the spec as given, with the file recorded beside it
    assert meta["graph_file"] == {"path": str(gfile.resolve()), "sha256": _sha256(gfile)}

    # the graph file is regenerated with other graphs, then the run is extended
    other_graph = graphs.watts_strogatz(n, 4, 0.1, seed=2)
    assert not np.array_equal(other_graph, first_graph)
    graphs.save(gfile, [other_graph], n)
    changed = re.escape(f"the graph file {str(gfile.resolve())!r} changed since the run started")
    before = {name: (out / name).read_bytes() for name in ("spec.json", "provenance.json")}
    with pytest.raises(ValueError, match=changed + r".*edges differs"):
        api.start_ensemble_job(spec, 40, output_dir=str(out), chunk_size=10)
    assert not (out / "chunks").exists()  # refused before anything ran or was written
    assert {name: (out / name).read_bytes() for name in before} == before
    assert api.load_results(str(out))["num_trajectories"] == 20

    # a spec.json from before the hash was recorded: the saved graphs are compared instead
    del meta["graph_file"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match=r"no longer holds the graphs .*edges differs"):
        api.start_ensemble_job(spec, 40, output_dir=str(out), chunk_size=10)
    assert not (out / "chunks").exists()

    # with the original file back, the extension runs and equals one run of 40
    gfile.write_bytes(original)
    more = api.start_ensemble_job(spec, 40, output_dir=str(out), chunk_size=10)
    assert more["resumed_from"] == 20
    assert api.job_status(more["job_id"], wait_seconds=60)["status"] == "completed"
    saved = mipt.EnsembleResult.load(out / "result.npz")
    direct = _direct_file_run(first_graph, 40, seed=3)
    np.testing.assert_array_equal(saved.purification_layer, direct.purification_layer)
    np.testing.assert_array_equal(saved.values["code_dimension"], direct.values["code_dimension"])
    assert json.loads(meta_path.read_text(encoding="utf-8"))["graph_file"]["sha256"] == (
        hashlib.sha256(original).hexdigest()
    )


def test_a_graph_file_rewritten_with_the_same_graphs_resumes_only_against_saved_results(
    tmp_path,
):
    n, gfile = 12, tmp_path / "ws.npz"
    graph = graphs.watts_strogatz(n, 4, 0.1, seed=1)
    graphs.save(gfile, [graph], n)
    spec, out = _file_spec(gfile), tmp_path / "run"
    job = api.start_ensemble_job(spec, 10, output_dir=str(out), seed=3)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    # the same graph in other bytes (with an extra array that graphs.load ignores)
    np.savez(gfile, num_nodes=n, edges=graph, comment=np.array("regenerated"))
    more = api.start_ensemble_job(spec, 20, output_dir=str(out))
    assert any("holds the graphs the saved results used" in note for note in more["notes"])
    assert api.job_status(more["job_id"], wait_seconds=60)["status"] == "completed"
    saved = mipt.EnsembleResult.load(out / "result.npz")
    direct = _direct_file_run(graph, 20, seed=3)
    np.testing.assert_array_equal(saved.purification_layer, direct.purification_layer)
    meta = json.loads((out / "spec.json").read_text(encoding="utf-8"))
    assert meta["graph_file"]["sha256"] == _sha256(gfile)  # the bytes now in use

    # with nothing saved yet there are no graphs to compare: other bytes are refused
    (out / "result.npz").unlink()
    graphs.save(gfile, [graph], n)
    with pytest.raises(ValueError, match="changed since the run started"):
        api.start_ensemble_job(spec, 20, output_dir=str(out))


def test_a_relative_graph_path_read_from_another_directory_is_checked(tmp_path, monkeypatch):
    n, first, second = 12, tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    graph = graphs.watts_strogatz(n, 4, 0.1, seed=1)
    graphs.save(first / "g.npz", [graph], n)
    graphs.save(second / "g.npz", [graphs.watts_strogatz(n, 4, 0.1, seed=2)], n)
    spec, out = _file_spec("g.npz"), str(tmp_path / "run")  # resolved from the cwd
    monkeypatch.chdir(first)
    job = api.start_ensemble_job(spec, 10, output_dir=out, seed=3)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    monkeypatch.chdir(second)  # the same spec text now reads another file
    named = re.escape(
        f"the graph file {str((second / 'g.npz').resolve())!r} changed since the run started "
        f"(then {str((first / 'g.npz').resolve())!r})"
    )
    with pytest.raises(ValueError, match=named):
        api.start_ensemble_job(spec, 20, output_dir=out)
    # a byte-identical copy at the new path is the same file: the run continues
    (second / "g.npz").write_bytes((first / "g.npz").read_bytes())
    more = api.start_ensemble_job(spec, 20, output_dir=out)
    assert api.job_status(more["job_id"], wait_seconds=60)["status"] == "completed"
    saved = mipt.EnsembleResult.load(tmp_path / "run" / "result.npz")
    direct = _direct_file_run(graph, 20, seed=3)
    np.testing.assert_array_equal(saved.purification_layer, direct.purification_layer)


def _rewrite_meta(path, **changes):
    """Rewrite entries of a saved result's metadata (as a file from elsewhere would have)."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["__meta__"]))
    meta.update(changes)
    arrays["__meta__"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)


def test_version_and_seed_are_checked_against_the_saved_results(tmp_path):
    out = tmp_path / "copied"
    job = api.start_ensemble_job(SMALL_SPEC, 10, output_dir=str(out), seed=4)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    # copied without provenance.json, from another version
    (out / "provenance.json").unlink()
    _rewrite_meta(out / "result.npz", version="3.0.0.dev11")
    with pytest.raises(ValueError, match=r"started with sparsegf2 3\.0\.0\.dev11, but this is"):
        api.start_ensemble_job(SMALL_SPEC, 20, output_dir=str(out))
    assert not (out / "chunks").exists()
    # this version's results, but spec.json gives another seed than they were made with
    _rewrite_meta(out / "result.npz", version=sg.__version__)
    meta = json.loads((out / "spec.json").read_text(encoding="utf-8"))
    (out / "spec.json").write_text(json.dumps({**meta, "seed": 5}), encoding="utf-8")
    with pytest.raises(
        ValueError, match="gives seed 5, but the saved results were made with seed 4"
    ):
        api.start_ensemble_job(SMALL_SPEC, 20, output_dir=str(out))
    assert not (out / "chunks").exists()
    (out / "spec.json").write_text(json.dumps(meta), encoding="utf-8")
    more = api.start_ensemble_job(SMALL_SPEC, 20, output_dir=str(out))
    assert api.job_status(more["job_id"], wait_seconds=60)["status"] == "completed"
    saved = mipt.EnsembleResult.load(out / "result.npz")
    np.testing.assert_array_equal(
        saved.purification_layer, _direct_spec().run(20, seed=4).purification_layer
    )


def test_a_merge_that_fails_names_the_chunk_files_that_do_not_belong(tmp_path):
    out = tmp_path / "mixed"
    api.run_ensemble(SMALL_SPEC, 20, seed=4, output_dir=str(out))
    chunks = out / "chunks"
    chunks.mkdir()
    _direct_spec().run(10, seed=4, first_trajectory=20).save(chunks / "traj_000000000020.npz")
    stray = chunks / "traj_000000000030.npz"
    _direct_spec(measurement_rate=0.3).run(10, seed=4, first_trajectory=30).save(stray)
    with pytest.raises(ValueError, match="do not belong to the same run") as info:
        api.load_results(str(out))
    message = str(info.value)
    assert "traj_000000000030.npz" in message and "traj_000000000020.npz" not in message
    assert "delete them" in message
    stray.unlink()  # as the message says: the earlier trajectories are readable again
    assert api.load_results(str(out))["num_trajectories"] == 30


def test_load_results_returns_the_exact_spec_which_extends_the_run(tmp_path):
    spec = {"num_system": 8,
            "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.123456789, "seed": 1},
            "layers": 10, "measurement_rate": 1 / 3, "reference": "single", "checkpoints": 5,
            "probes": [{"kind": "code_dimension"}]}  # fmt: skip
    out = str(tmp_path / "exact")
    job = api.start_ensemble_job(spec, 10, output_dir=out, seed=2)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    shown = api.load_results(out)["spec"]
    assert shown == spec  # not rounded for display: it is the run's identity
    assert shown["measurement_rate"] == 1 / 3 and shown["graph"]["beta"] == 0.123456789
    more = api.start_ensemble_job(shown, 20, output_dir=out)
    assert more["resumed_from"] == 10
    assert api.job_status(more["job_id"], wait_seconds=60)["status"] == "completed"
    assert api.load_results(out)["num_trajectories"] == 20
    # another spec is refused with the first differing key path and both values
    for change, where in (
        ({"measurement_rate": 0.333333},
         "measurement_rate is 0.3333333333333333 in its spec.json and 0.333333 here"),
        ({"graph": {**spec["graph"], "k": 2}}, "graph.k is 4 in its spec.json and 2 here"),
        ({"probes": [{"kind": "half_cut_entropy"}]},
         'probes[0].kind is "code_dimension" in its spec.json and "half_cut_entropy" here'),
        ({"checkpoints": "final"}, 'checkpoints is 5 in its spec.json and "final" here'),
    ):  # fmt: skip
        with pytest.raises(ValueError, match="different spec: " + re.escape(where)):
            api.start_ensemble_job({**spec, **change}, 20, output_dir=out)
    without = {k: v for k, v in spec.items() if k != "checkpoints"}  # "final" by default
    with pytest.raises(ValueError, match=re.escape("checkpoints is 5 in its spec.json and absent")):
        api.start_ensemble_job(without, 20, output_dir=out)
    # a harmless difference first in the file (10.0 for 10) does not hide the one that counts
    rate = "different spec: measurement_rate is 0.3333333333333333 in its spec.json and 0.25 here"
    with pytest.raises(ValueError, match=re.escape(rate)):
        api.start_ensemble_job(
            {**spec, "layers": 10.0, "measurement_rate": 0.25}, 20, output_dir=out
        )


def test_bounded_keeps_an_exact_field_whole_when_it_cuts_the_rest():
    spec = {"measurement_rate": 1 / 3, "edges": [[i, i + 1] for i in range(40)]}
    out = api._bounded({"spec": spec, "values": list(range(5000))}, exact=("spec",))
    assert out["truncated"] and len(out["values"]) < 5000  # the rest was cut to fit
    assert out["spec"] == spec  # neither rounded nor cut to 32 edges


def test_resume_accepts_an_equivalent_cutoff_form_and_refuses_another_size(tmp_path):
    # n = 10: C(10, 1) + C(10, 2) = 55 <= 100 < 55 + C(10, 3) = 175, so budget 100 gives w = 2
    def counts(**cutoff) -> dict:
        probe = {"kind": "uncorrectable_region_counts", **cutoff}
        return dict(SMALL_SPEC, reference="purification", stop_when_purified=False,
                    checkpoints=10, probes=[probe])  # fmt: skip

    out = str(tmp_path / "cutoff")
    job = api.start_ensemble_job(counts(budget=100), 10, output_dir=out, seed=2)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    same = counts(max_size={"10": 2, "64": 3})  # w = 2 at n = 10 too: the same probe
    more = api.start_ensemble_job(same, 20, output_dir=out)
    assert more["resumed_from"] == 10
    assert any("builds the same protocol" in note for note in more["notes"])
    assert api.job_status(more["job_id"], wait_seconds=60)["status"] == "completed"
    saved = mipt.EnsembleResult.load(tmp_path / "cutoff" / "result.npz")
    direct = _direct_spec(
        reference="purification", stop_when_purified=False, checkpoints=10,
        probes=[mipt.uncorrectable_region_counts(max_size=2)],
    ).run(20, seed=2)  # fmt: skip
    key = "uncorrectable_region_counts[2]"
    np.testing.assert_array_equal(saved.values[key], direct.values[key])
    stored = json.loads((tmp_path / "cutoff" / "spec.json").read_text(encoding="utf-8"))
    assert stored["spec"] == same  # the form given last
    refused = r'different spec: probes\[0\]\.max_size is \{"10": 2, "64": 3\} in its spec\.json '
    with pytest.raises(ValueError, match=refused + r"and 3 here.*protocol differs in probes"):
        api.start_ensemble_job(counts(max_size=3), 30, output_dir=out)


def test_resume_without_trajectories_runs_to_the_stored_target(tmp_path):
    out = tmp_path / "crashed"
    job = api.start_ensemble_job(SMALL_SPEC, 30, output_dir=str(out), seed=5, chunk_size=10)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    # the state a crash leaves: spec.json asks for 30 trajectories, one chunk of 10 is saved
    (out / "result.npz").unlink()
    (out / "chunks").mkdir()
    _direct_spec().run(10, seed=5).save(out / "chunks" / "traj_000000000000.npz")
    resumed = api.start_ensemble_job(SMALL_SPEC, output_dir=str(out))  # no trajectories, seed
    assert (resumed["resumed_from"], resumed["trajectories_total"]) == (10, 30)
    assert api.job_status(resumed["job_id"], wait_seconds=60)["status"] == "completed"
    saved = mipt.EnsembleResult.load(out / "result.npz")
    np.testing.assert_array_equal(
        saved.purification_layer, _direct_spec().run(30, seed=5).purification_layer
    )
    again = api.start_ensemble_job(SMALL_SPEC, output_dir=str(out))  # finished: nothing to do
    assert (again["resumed_from"], again["trajectories_total"]) == (30, 30)
    assert api.job_status(again["job_id"], wait_seconds=60)["status"] == "completed"
    with pytest.raises(ValueError, match="give trajectories"):  # a new directory needs a count
        api.start_ensemble_job(SMALL_SPEC, output_dir=str(tmp_path / "new"))
    assert not (tmp_path / "new").exists()  # and is not created by the refusal


def test_a_graph_collection_resumes_to_its_stored_target_too(tmp_path):
    # 5 graphs x 4 circuits = 20 trajectories; the run is started with 8 of them
    bank = {**SMALL_SPEC, "graph": {**SMALL_SPEC["graph"], "realizations": 5},
            "circuits_per_graph": 4}  # fmt: skip
    direct = _direct_spec(
        edges=[graphs.watts_strogatz(10, k=4, beta=0.2, seed=[3, i]) for i in range(5)],
        circuits_per_graph=4,
    )
    out = tmp_path / "bank"
    job = api.start_ensemble_job(bank, 8, output_dir=str(out), seed=1, chunk_size=4)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    # a crash after the first chunk of 4: the resume reaches the target 8, not all 20
    (out / "result.npz").unlink()
    (out / "chunks").mkdir()
    direct.run(4, seed=1).save(out / "chunks" / "traj_000000000000.npz")
    resumed = api.start_ensemble_job(bank, output_dir=str(out))
    assert (resumed["resumed_from"], resumed["trajectories_total"]) == (4, 8)
    assert api.job_status(resumed["job_id"], wait_seconds=60)["status"] == "completed"
    saved = mipt.EnsembleResult.load(out / "result.npz")
    np.testing.assert_array_equal(
        saved.purification_layer, direct.run(8, seed=1).purification_layer
    )
    # the rest needs an explicit count; a new run without one runs (and stores) all 20
    rest = api.start_ensemble_job(bank, 20, output_dir=str(out))
    assert (rest["resumed_from"], rest["trajectories_total"]) == (8, 20)
    assert api.job_status(rest["job_id"], wait_seconds=60)["status"] == "completed"
    fresh = api.start_ensemble_job(bank, output_dir=str(tmp_path / "fresh"), seed=1)
    assert fresh["trajectories_total"] == 20
    assert api.job_status(fresh["job_id"], wait_seconds=60)["status"] == "completed"
    stored = json.loads((tmp_path / "fresh" / "spec.json").read_text(encoding="utf-8"))
    assert stored["trajectories"] == 20


def test_contract_refusal_comes_before_the_spec_comparison(tmp_path):
    out = tmp_path / "old"
    api.run_ensemble(SMALL_SPEC, 5, seed=1, output_dir=str(out))
    _rewrite_without_contract(out / "result.npz")
    with pytest.raises(ValueError, match="made under randomness contract 1"):
        api.start_ensemble_job(dict(SMALL_SPEC, layers=31), 10, output_dir=str(out))


# ---------------------------------------------------------------------------
# one writer per output directory, and merges that survive readers
# ---------------------------------------------------------------------------

_PENDING = (
    "merge pending: result.npz is in use; the chunk files hold the data, and the next "
    "load_results or start_ensemble_job merges them"
)

# a long job (2000 chunks of one trajectory), cancelled by the test that starts it
_LONG_SPEC = dict(SMALL_SPEC, num_system=24, layers=200, stop_when_purified=False,
                  graph={"family": "cycle"}, measurement_rate=0.05)  # fmt: skip


def _write_lock(out: Path, pid: int, host: str | None = None, **fields) -> dict:
    """A lock file as another writer would have left it (``fields``: ``os``, ``created``)."""
    started = datetime.now(UTC).isoformat(timespec="seconds")
    record = {"pid": pid, "host": host or socket.gethostname(), "started": started, **fields}
    out.mkdir(parents=True, exist_ok=True)
    (out / ".lock").write_text(json.dumps(record), encoding="utf-8")
    return record


def _held(out: Path, record: dict, who: str = "another process") -> str:
    """The refusal for a lock held by ``record``, as a regular expression (it names the
    lock's system when that is not this one)."""
    system = record.get("os")
    shown = f" ({system})" if system not in (None, api._os_identity()) else ""
    return re.escape(
        f"{who} (pid {record['pid']} on host {record['host']}{shown}, started "
        f"{record['started']}) is writing to {str(out.resolve())!r}; wait for it to finish "
        "or cancel it"
    )


def _processes(monkeypatch, states: dict) -> None:
    """Processes as the system reports them, ``{pid: (alive, creation time)}``; other pids
    (this process's) are asked for real."""
    real = api._process_state
    monkeypatch.setattr(api, "_process_state", lambda pid: states.get(pid) or real(pid))


def test_a_live_writer_holds_the_output_dir_against_every_other_writer(tmp_path, monkeypatch):
    _processes(monkeypatch, {4242: (True, None), 9999: (False, None)})
    out = tmp_path / "held"
    record = _write_lock(out, 4242, os=api._os_identity())  # another process, running
    text = (out / ".lock").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=_held(out, record)):
        api.start_ensemble_job(SMALL_SPEC, 10, output_dir=str(out), seed=1)
    with pytest.raises(ValueError, match=_held(out, record)):
        api.run_ensemble(SMALL_SPEC, 10, seed=1, output_dir=str(out))
    # nothing was written, and the holder's lock is untouched
    assert [p.name for p in out.iterdir()] == [".lock"]
    assert (out / ".lock").read_text(encoding="utf-8") == text
    # a lock of this process that no job holds (a run_ensemble in another thread, say)
    mine = _write_lock(out, os.getpid())
    with pytest.raises(ValueError, match=_held(out, mine, who="this process")):
        api.start_ensemble_job(SMALL_SPEC, 10, output_dir=str(out), seed=1)
    # another machine's lock always holds, even when its pid runs nothing here: another
    # hostname, or this hostname under another system (WSL and containers share it)
    for other in ({"host": "cluster-node-7"}, {"os": "Linux pidns:1"}):
        record = _write_lock(out, 9999, **other)
        with pytest.raises(ValueError, match=_held(out, record)):
            api.start_ensemble_job(SMALL_SPEC, 10, output_dir=str(out), seed=1)
    (out / ".lock").unlink()  # the holder finished
    job = api.start_ensemble_job(SMALL_SPEC, 10, output_dir=str(out), seed=1)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    assert not (out / ".lock").exists()


def test_a_lock_left_by_a_dead_process_is_taken_over(tmp_path, monkeypatch):
    dead = 4242
    _processes(monkeypatch, {dead: (False, None)})
    out = tmp_path / "stale"
    # a lock written before the system was recorded is judged by its hostname alone
    for fields in ({}, {"os": api._os_identity()}):
        _write_lock(out, dead, **fields)
        lock = api._lock_output_dir(out)
        held = json.loads((out / ".lock").read_text(encoding="utf-8"))
        assert (held["pid"], held["host"], held["os"]) == (
            os.getpid(), socket.gethostname(), api._os_identity()
        )  # fmt: skip
        assert held.get("created") == api._process_state(os.getpid())[1]  # None off Windows
        assert [p.name for p in out.iterdir()] == [".lock"]  # no turn or temporary file left
        lock.release()
        assert not (out / ".lock").exists()
    # through a job and through run_ensemble
    _write_lock(out, dead)
    job = api.start_ensemble_job(SMALL_SPEC, 10, output_dir=str(out), seed=1)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    assert not (out / ".lock").exists()
    _write_lock(tmp_path / "stale2", dead)
    api.run_ensemble(SMALL_SPEC, 5, seed=1, output_dir=str(tmp_path / "stale2"))
    assert (tmp_path / "stale2" / "result.npz").exists()
    assert not (tmp_path / "stale2" / ".lock").exists()


def test_a_lock_whose_pid_now_names_another_process_is_taken_over(tmp_path, monkeypatch):
    # Windows soon gives the pid of a process that ended to another one; the lock records
    # its holder's creation time (the kernel's), which the new process does not share
    _processes(monkeypatch, {4242: (True, 1001)})
    out = tmp_path / "reused"
    record = _write_lock(out, 4242, created=1001)  # the running process took it: it holds
    with pytest.raises(ValueError, match=_held(out, record)):
        api._lock_output_dir(out)
    _write_lock(out, 4242, created=1000)  # an earlier process with that pid took it
    api._lock_output_dir(out).release()
    assert not (out / ".lock").exists()
    # without a creation time on either side (a POSIX writer, say) liveness decides
    record = _write_lock(out, 4242)
    with pytest.raises(ValueError, match=_held(out, record)):
        api._lock_output_dir(out)
    _processes(monkeypatch, {4242: (True, None)})
    record = _write_lock(out, 4242, created=1000)
    with pytest.raises(ValueError, match=_held(out, record)):
        api._lock_output_dir(out)


def test_a_lock_without_a_record_is_judged_by_its_age(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_RETRY_DELAYS", (0.001,) * len(api._RETRY_DELAYS))
    out = tmp_path / "empty"
    out.mkdir()
    # created but not yet written (a drive without hard links): its writer may be alive
    (out / ".lock").write_bytes(b"")
    with pytest.raises(ValueError, match="could not take the lock .*retry, or delete it if no"):
        api._lock_output_dir(out)
    assert (out / ".lock").read_bytes() == b""
    # still empty long after: its writer died while writing it
    old = time.time() - 60
    os.utime(out / ".lock", (old, old))
    api._lock_output_dir(out).release()
    assert not (out / ".lock").exists()


def test_the_lock_appears_with_its_whole_record(tmp_path, monkeypatch):
    out = tmp_path / "atomic"
    out.mkdir()
    real_link, linked = os.link, []

    def link(src, dst, *args, **kwargs):
        # the record is complete before the lock appears under its name
        linked.append(json.loads(Path(src).read_text(encoding="utf-8")))
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", link)
    lock = api._lock_output_dir(out)
    assert linked == [json.loads((out / ".lock").read_text(encoding="utf-8"))]
    assert [p.name for p in out.iterdir()] == [".lock"]
    lock.release()
    # without hard links (FAT drives, some shares; EPERM on Linux) it is created exclusively
    for error in (OSError(errno.EOPNOTSUPP, "not supported"), PermissionError(1, "not permitted")):

        def unsupported(src, dst, *args, error=error, **kwargs):
            raise error

        monkeypatch.setattr(os, "link", unsupported)
        lock = api._lock_output_dir(out)
        assert json.loads((out / ".lock").read_text(encoding="utf-8"))["pid"] == os.getpid()
        assert [p.name for p in out.iterdir()] == [".lock"]
        with pytest.raises(ValueError, match="this process"):  # and still exclusively
            api._lock_output_dir(out)
        lock.release()


def test_a_stale_lock_is_deleted_only_while_it_is_unchanged(tmp_path):
    out = tmp_path / "race"
    _write_lock(out, 4242)
    path = out / ".lock"
    seen = path.read_bytes()
    # another process took the stale lock over meanwhile: its lock is kept
    mine = _write_lock(out, os.getpid())
    api._remove_stale_lock(path, seen)
    assert json.loads(path.read_text(encoding="utf-8")) == mine
    # while another process takes its turn (.lock.takeover exists), nothing is deleted
    path.write_bytes(seen)
    guard = out / ".lock.takeover"
    guard.touch()
    api._remove_stale_lock(path, seen)
    assert path.read_bytes() == seen
    # a turn left behind by a process that died in it is freed after a while
    old = time.time() - 60
    os.utime(guard, (old, old))
    api._remove_stale_lock(path, seen)
    assert not guard.exists()
    api._remove_stale_lock(path, seen)
    assert not path.exists() and not guard.exists()


def test_process_state_tells_a_running_process_from_a_finished_one():
    alive, created = api._process_state(os.getpid())
    assert alive and (created is None or isinstance(created, int))  # known on Windows
    assert api._process_state(os.getpid()) == (alive, created)  # the kernel's, so it is stable
    with subprocess.Popen([sys.executable, "-c", "pass"]) as proc:
        proc.wait()
    assert not api._process_state(proc.pid)[0]
    assert api._process_state(0) == api._process_state(-1) == (False, None)  # no os.kill(0)


def test_lock_records_are_validated_and_a_lock_that_cannot_be_taken_says_so(tmp_path, monkeypatch):
    good = {"pid": 4242, "host": "h", "started": "2026-10-15T08:00:00+00:00"}
    assert api._lock_record(json.dumps(good).encode()) == good
    for bad in (b"", b"\xff", b"[1]", json.dumps({"pid": 4242, "host": "h"}).encode(),
                *(json.dumps({**good, **change}).encode()
                  for change in ({"pid": True}, {"pid": 0}, {"pid": 2**31}, {"pid": "4242"},
                                 {"host": 7}))):  # fmt: skip
        assert api._lock_record(bad) is None, bad
    monkeypatch.setattr(api, "_RETRY_DELAYS", (0.001,) * len(api._RETRY_DELAYS))
    # a stale lock that keeps coming back (another process takes turns that never end)
    _processes(monkeypatch, {4242: (False, None)})
    monkeypatch.setattr(api, "_remove_stale_lock", lambda path, seen: None)
    out = tmp_path / "contended"
    _write_lock(out, 4242)
    with pytest.raises(ValueError, match="could not take the lock .*retry, or delete it if no"):
        api._lock_output_dir(out)
    # a directory this process may not write to: an input error, with the last attempt's
    # error (a PermissionError is retried: on Windows a lock being deleted gives one too)
    (out / ".lock").unlink()
    attempts = []

    def denied(path, data, error=None):
        attempts.append(path)
        raise error or PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(api, "_create_lock", denied)
    not_writable = re.escape(f"output_dir {str(out)!r} is not writable: [Errno 13] Permission")
    with pytest.raises(ValueError, match=not_writable):
        api._lock_output_dir(out)
    assert len(attempts) == len(api._RETRY_DELAYS) + 1
    # any other error (a read-only or full drive) at once: retrying cannot help
    attempts.clear()
    read_only = OSError(errno.EROFS, "Read-only file system")
    monkeypatch.setattr(api, "_create_lock", lambda path, data: denied(path, data, read_only))
    with pytest.raises(ValueError, match=re.escape(f"{str(out)!r} is not writable: [Errno")):
        api._lock_output_dir(out)
    assert len(attempts) == 1


def test_a_running_job_is_named_to_a_second_writer_in_the_same_process(tmp_path):
    out = tmp_path / "busy"
    job = api.start_ensemble_job(_LONG_SPEC, 2000, output_dir=str(out), seed=2, chunk_size=1)
    try:
        named = f"job {job['job_id']} is already writing to {str(out.resolve())!r}; poll it"
        for _ in range(2):  # the second time with .lock deleted by hand: still refused
            with pytest.raises(ValueError, match=re.escape(named)):
                api.start_ensemble_job(_LONG_SPEC, 2000, output_dir=str(out))
            with pytest.raises(ValueError, match=re.escape(named)):
                api.run_ensemble(_LONG_SPEC, 2, seed=2, output_dir=str(out))
            (out / ".lock").unlink(missing_ok=True)
    finally:
        api.cancel_job(job["job_id"])
        final = api.job_status(job["job_id"], wait_seconds=60)
    assert final["status"] == "cancelled" and not (out / ".lock").exists()


def test_an_interrupt_after_the_job_started_leaves_it_the_lock(tmp_path, monkeypatch):
    real_start = threading.Thread.start

    def start(self, fail=None):
        if fail is not None and self.name.startswith("sparsegf2-job-"):
            raise fail
        real_start(self)
        if self.name.startswith("sparsegf2-job-"):
            raise KeyboardInterrupt  # Ctrl-C right after the job started

    monkeypatch.setattr(threading.Thread, "start", start)
    out = tmp_path / "handover"
    with pytest.raises(KeyboardInterrupt):
        api.start_ensemble_job(_LONG_SPEC, 2000, output_dir=str(out), seed=2, chunk_size=1)
    monkeypatch.undo()
    (job,) = [j for j in api.list_jobs()["jobs"] if j["output_dir"] == str(out.resolve())]
    try:  # the running job keeps its lock
        assert json.loads((out / ".lock").read_text(encoding="utf-8"))["pid"] == os.getpid()
    finally:
        api.cancel_job(job["job_id"])
        final = api.job_status(job["job_id"], wait_seconds=60)
    assert final["status"] == "cancelled" and not (out / ".lock").exists()
    # a thread that cannot be started owns nothing: the lock is released, the job forgotten
    monkeypatch.setattr(
        threading.Thread, "start", lambda self: start(self, RuntimeError("can't start"))
    )
    with pytest.raises(RuntimeError, match="can't start"):
        api.start_ensemble_job(SMALL_SPEC, 10, output_dir=str(tmp_path / "no_thread"), seed=1)
    assert not (tmp_path / "no_thread" / ".lock").exists()
    assert all(j["output_dir"] != str((tmp_path / "no_thread").resolve())
               for j in api.list_jobs()["jobs"])  # fmt: skip


def test_the_lock_is_released_on_every_path(tmp_path, monkeypatch):
    out = tmp_path / "paths"
    job = api.start_ensemble_job(SMALL_SPEC, 10, output_dir=str(out), seed=1)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    # a refused resume
    with pytest.raises(ValueError, match="different spec"):
        api.start_ensemble_job(dict(SMALL_SPEC, layers=31), 20, output_dir=str(out))
    assert not (out / ".lock").exists()

    # a job that fails
    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(api, "_atomic_save_result", disk_full)
    job = api.start_ensemble_job(SMALL_SPEC, 20, output_dir=str(out))
    final = api.job_status(job["job_id"], wait_seconds=60)
    assert final["status"] == "failed" and "disk full" in final["error"]
    assert not (out / ".lock").exists()
    monkeypatch.undo()

    # run_ensemble interrupted by Ctrl-C
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(mipt.MonitoredCircuit, "run", interrupted)
    with pytest.raises(KeyboardInterrupt):
        api.run_ensemble(SMALL_SPEC, 5, seed=1, output_dir=str(tmp_path / "ctrl_c"))
    assert not (tmp_path / "ctrl_c" / ".lock").exists()


def test_overlapping_chunks_load_and_merge(tmp_path):
    snapshot = {"kind": "code_snapshot", "checkpoints": "final"}
    spec = dict(SMALL_SPEC, probes=[*SMALL_SPEC["probes"], snapshot])
    direct = _direct_spec(
        probes=[mipt.code_dimension(), mipt.half_cut_entropy(),
                mipt.code_snapshot(checkpoints="final")]
    )  # fmt: skip
    out = tmp_path / "overlap"
    job = api.start_ensemble_job(spec, 26, output_dir=str(out), seed=5, chunk_size=13)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    # two writers with chunk sizes 13 and 8 left overlapping chunks, and no result.npz
    (out / "result.npz").unlink()
    (out / "chunks").mkdir()
    for first, count in ((0, 13), (8, 8), (13, 13), (16, 8)):
        direct.run(count, seed=5, first_trajectory=first).save(
            out / "chunks" / f"traj_{first:012d}.npz"
        )

    def same_as_direct(result: mipt.EnsembleResult, count: int) -> None:
        full = direct.run(count, seed=5)
        np.testing.assert_array_equal(result.trajectories, np.arange(count))
        np.testing.assert_array_equal(result.purification_layer, full.purification_layer)
        np.testing.assert_array_equal(result.final_layer, full.final_layer)
        for key, value in full.values.items():
            np.testing.assert_array_equal(result.values[key], value)
        for a, b in zip(result.snapshots["code_snapshot"], full.snapshots["code_snapshot"],
                        strict=True):  # fmt: skip
            np.testing.assert_array_equal(a, b)

    loaded, _ = api._load_directory(out)
    same_as_direct(loaded, 26)
    # the run is complete, so load_results also merges the chunks
    assert api.load_results(str(out))["num_trajectories"] == 26
    assert not (out / "chunks").exists()
    same_as_direct(mipt.EnsembleResult.load(out / "result.npz"), 26)
    # chunks that overlap result.npz (20..33 against 0..25) and each other
    (out / "chunks").mkdir()
    for first, count in ((20, 14), (30, 10)):
        direct.run(count, seed=5, first_trajectory=first).save(
            out / "chunks" / f"traj_{first:012d}.npz"
        )
    more = api.start_ensemble_job(spec, 40, output_dir=str(out))
    assert more["resumed_from"] == 40
    assert api.job_status(more["job_id"], wait_seconds=60)["status"] == "completed"
    same_as_direct(mipt.EnsembleResult.load(out / "result.npz"), 40)
    assert not (out / "chunks").exists()
    # a part of another run is refused even when every trajectory in it is covered
    (out / "chunks").mkdir()
    stray = out / "chunks" / "traj_000000000003.npz"
    _direct_spec(measurement_rate=0.3).run(5, seed=5, first_trajectory=3).save(stray)
    with pytest.raises(ValueError, match="do not belong to the same run") as info:
        api.load_results(str(out))
    assert "traj_000000000003.npz" in str(info.value) and "delete them" in str(info.value)


def _blocked_replace(monkeypatch, failures: int) -> list:
    """``os.replace`` onto result.npz fails ``failures`` times, as on Windows while a reader
    holds the file open; returns the list of attempts."""
    real = os.replace
    attempts: list = []

    def replace(src, dst):
        if Path(dst).name == "result.npz":
            attempts.append(dst)
            if len(attempts) <= failures:
                raise PermissionError(13, "the file is in use by another process", str(dst))
        return real(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    return attempts


def test_the_final_merge_retries_while_a_reader_holds_the_files(tmp_path, monkeypatch):
    assert 4.0 <= sum(api._RETRY_DELAYS) <= 5.5  # the backoff gives up after about 5 s
    monkeypatch.setattr(api, "_RETRY_DELAYS", (0.001,) * len(api._RETRY_DELAYS))
    attempts = _blocked_replace(monkeypatch, failures=2)
    real_unlink, unlinks = os.unlink, []

    def unlink(path, *args, **kwargs):
        if Path(path).name == "traj_000000000000.npz":
            unlinks.append(path)
            if len(unlinks) == 1:  # a reader has this chunk open for a moment
                raise PermissionError(13, "the file is in use by another process", str(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    out = tmp_path / "busy"
    job = api.start_ensemble_job(SMALL_SPEC, 20, output_dir=str(out), seed=4, chunk_size=10)
    final = api.job_status(job["job_id"], wait_seconds=60)
    assert final["status"] == "completed" and "notes" not in final
    assert len(attempts) == 3 and len(unlinks) == 2
    saved = mipt.EnsembleResult.load(out / "result.npz")
    np.testing.assert_array_equal(
        saved.purification_layer, _direct_spec().run(20, seed=4).purification_layer
    )
    assert not (out / "chunks").exists() and not (out / ".lock").exists()


def test_a_merge_a_reader_blocks_is_pending_and_the_next_call_merges(tmp_path, monkeypatch):
    out = tmp_path / "pending"
    job = api.start_ensemble_job(SMALL_SPEC, 10, output_dir=str(out), seed=4, chunk_size=10)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"

    def block() -> list:
        monkeypatch.setattr(api, "_RETRY_DELAYS", (0.001,) * len(api._RETRY_DELAYS))
        return _blocked_replace(monkeypatch, failures=10**9)

    def chunk_files() -> int:
        return len(list((out / "chunks").glob("traj_*.npz")))

    attempts = block()
    more = api.start_ensemble_job(SMALL_SPEC, 20, output_dir=str(out), chunk_size=10)
    final = api.job_status(more["job_id"], wait_seconds=60)
    assert final["status"] == "completed" and final["notes"] == [_PENDING], final
    assert len(attempts) == len(api._RETRY_DELAYS) + 1
    # the reader's file is untouched, the chunk is kept, nothing temporary is left
    assert mipt.EnsembleResult.load(out / "result.npz").num_trajectories == 10
    assert chunk_files() == 1
    assert not list(out.glob("*.tmp*")) and not (out / ".lock").exists()
    # while the file stays in use, load_results reads everything and leaves the merge pending
    assert api.load_results(str(out))["num_trajectories"] == 20
    assert chunk_files() == 1 and not (out / ".lock").exists()
    monkeypatch.undo()
    # while another process writes there, load_results only reads: that writer merges
    _write_lock(out, os.getpid())
    held = (out / ".lock").read_bytes()
    assert api.load_results(str(out))["num_trajectories"] == 20
    assert chunk_files() == 1 and (out / ".lock").read_bytes() == held
    (out / ".lock").unlink()
    # the reader is done: start_ensemble_job merges
    again = api.start_ensemble_job(SMALL_SPEC, output_dir=str(out))
    assert (again["resumed_from"], again["trajectories_total"]) == (20, 20)
    assert api.job_status(again["job_id"], wait_seconds=60)["status"] == "completed"
    assert not (out / "chunks").exists()
    assert mipt.EnsembleResult.load(out / "result.npz").num_trajectories == 20

    block()
    more = api.start_ensemble_job(SMALL_SPEC, 30, output_dir=str(out), chunk_size=10)
    assert api.job_status(more["job_id"], wait_seconds=60)["notes"] == [_PENDING]
    monkeypatch.undo()  # and so does load_results
    assert api.load_results(str(out))["num_trajectories"] == 30
    assert not (out / "chunks").exists() and not (out / ".lock").exists()
    saved = mipt.EnsembleResult.load(out / "result.npz")
    np.testing.assert_array_equal(
        saved.purification_layer, _direct_spec().run(30, seed=4).purification_layer
    )


def test_load_results_retries_while_a_merge_replaces_a_file(tmp_path, monkeypatch):
    out = tmp_path / "reading"
    api.run_ensemble(SMALL_SPEC, 10, seed=4, output_dir=str(out))
    monkeypatch.setattr(api, "_RETRY_DELAYS", (0.001,) * len(api._RETRY_DELAYS))
    real_load, calls = mipt.EnsembleResult.load, []

    def load(path, failures=2):
        calls.append(path)
        if len(calls) <= failures:
            raise PermissionError(13, "the file is being replaced", str(path))
        return real_load(path)

    monkeypatch.setattr(mipt.EnsembleResult, "load", staticmethod(load))
    assert api.load_results(str(out))["num_trajectories"] == 10
    assert len(calls) == 3
    calls.clear()
    monkeypatch.setattr(
        mipt.EnsembleResult, "load", staticmethod(lambda path: load(path, failures=10**9))
    )
    with pytest.raises(ValueError, match="could not read results from"):
        api.load_results(str(out))
    assert len(calls) == len(api._RETRY_DELAYS) + 1


def test_a_merge_that_cannot_write_its_file_fails_instead_of_waiting(tmp_path, monkeypatch):
    real_save = mipt.EnsembleResult.save

    def save(self, path):
        if Path(path).name.startswith("result.npz"):  # the merged file, not a chunk
            raise PermissionError(13, "Permission denied", str(path))
        return real_save(self, path)

    monkeypatch.setattr(mipt.EnsembleResult, "save", save)
    out = tmp_path / "unwritable"
    job = api.start_ensemble_job(SMALL_SPEC, 20, output_dir=str(out), seed=4, chunk_size=10)
    final = api.job_status(job["job_id"], wait_seconds=60)
    assert final["status"] == "failed" and "Permission denied" in final["error"]
    assert "notes" not in final  # not "result.npz is in use"
    assert len(list((out / "chunks").glob("traj_*.npz"))) == 2
    assert not list(out.rglob("*.tmp*")) and not (out / ".lock").exists()


def test_chunks_that_result_npz_covers_are_deleted_without_rewriting_it(tmp_path, monkeypatch):
    out = tmp_path / "covered"
    job = api.start_ensemble_job(SMALL_SPEC, 20, output_dir=str(out), seed=4, chunk_size=10)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    # a chunk a held delete left behind: result.npz holds its trajectories already
    (out / "chunks").mkdir()
    _direct_spec().run(10, seed=4, first_trajectory=10).save(
        out / "chunks" / "traj_000000000010.npz"
    )
    written = (out / "result.npz").stat().st_mtime_ns
    attempts = _blocked_replace(monkeypatch, failures=0)  # records every replace of result.npz
    assert api.load_results(str(out))["num_trajectories"] == 20
    assert attempts == [] and (out / "result.npz").stat().st_mtime_ns == written
    assert not (out / "chunks").exists()


def test_chunks_a_reader_holds_share_one_backoff(tmp_path, monkeypatch):
    out = tmp_path / "held_chunks"
    job = api.start_ensemble_job(SMALL_SPEC, 20, output_dir=str(out), seed=4, chunk_size=10)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    chunks = out / "chunks"
    chunks.mkdir()
    for first in (0, 10):
        _direct_spec().run(10, seed=4, first_trajectory=first).save(
            chunks / f"traj_{first:012d}.npz"
        )
    real_unlink, sleeps = os.unlink, []

    def unlink(path, *args, **kwargs):
        if Path(path).name.startswith("traj_"):  # a sync client holds every chunk
            raise PermissionError(13, "the file is in use by another process", str(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    assert api._merge_chunks(out)
    assert sleeps == list(api._RETRY_DELAYS)  # one backoff of about 5 s for both chunks
    assert len(list(chunks.glob("traj_*.npz"))) == 2  # left: repeats that a load drops
    monkeypatch.undo()
    assert api.load_results(str(out))["num_trajectories"] == 20  # and the next merge deletes
    assert not chunks.exists()


def test_temporary_files_are_each_writers_own_and_never_read(tmp_path):
    out = tmp_path / "temporaries"
    first, second = api._temporary(out / "result.npz"), api._temporary(out / "result.npz")
    assert first != second
    assert first.name.startswith(f"result.npz.{os.getpid()}.") and first.name.endswith(".tmp.npz")
    job = api.start_ensemble_job(SMALL_SPEC, 10, output_dir=str(out), seed=4)
    assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
    # complete files that killed writers left under temporary names, of another run
    (out / "chunks").mkdir()
    other = _direct_spec(measurement_rate=0.3).run(5, seed=4, first_trajectory=10)
    other.save(out / "chunks" / "traj_000000000010.npz.999.deadbeef.tmp.npz")
    other.save(out / "result.npz.999.deadbeef.tmp.npz")
    assert api.load_results(str(out))["num_trajectories"] == 10  # never read as parts
    more = api.start_ensemble_job(SMALL_SPEC, 20, output_dir=str(out))
    assert api.job_status(more["job_id"], wait_seconds=60)["status"] == "completed"
    assert not list(out.rglob("*.tmp*")) and not (out / "chunks").exists()  # merged away
    saved = mipt.EnsembleResult.load(out / "result.npz")
    np.testing.assert_array_equal(
        saved.purification_layer, _direct_spec().run(20, seed=4).purification_layer
    )


def test_the_refusal_names_the_system_of_a_lock_taken_on_another(tmp_path, monkeypatch):
    # WSL and containers report the Windows (or host) name, so the pid alone would send a
    # Windows user to Task Manager for a Linux process
    _processes(monkeypatch, {9999: (False, None)})
    out = tmp_path / "wsl"
    record = _write_lock(out, 9999, os="Linux pidns:4026532396")
    shown = f"(pid 9999 on host {record['host']} (Linux pidns:4026532396), started "
    with pytest.raises(ValueError, match=re.escape(shown)):
        api._lock_output_dir(out)
    # a lock of this system, or one written before the system was recorded, says no more
    for fields in ({"os": api._os_identity()}, {}):
        record = _write_lock(out, os.getpid(), **fields)
        with pytest.raises(ValueError, match=re.escape(f"on host {record['host']}, started")):
            api._lock_output_dir(out)


def test_a_lock_with_a_malformed_creation_time_is_never_taken_over(tmp_path, monkeypatch):
    # a creation time that is not an int counts as absent, never as another process's
    created = 133_000_000_000_000_001  # a Windows FILETIME; as a float it is another number
    _processes(monkeypatch, {4242: (True, created)})
    out = tmp_path / "malformed"
    for bad in (float(created), str(created), True, None, [created]):
        record = _write_lock(out, 4242, os=api._os_identity(), created=bad)
        held = api._lock_record((out / ".lock").read_bytes())
        assert held is not None and "created" not in held, bad
        with pytest.raises(ValueError, match=_held(out, record)):
            api._lock_output_dir(out)
    record = _write_lock(out, 4242, os=api._os_identity(), created=created)
    assert api._lock_record((out / ".lock").read_bytes())["created"] == created


def test_a_lock_that_cannot_be_read_is_refused_and_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_RETRY_DELAYS", (0.001,) * len(api._RETRY_DELAYS))
    out = tmp_path / "unreadable_lock"
    _write_lock(out, 4242)
    old = time.time() - 60  # old enough to be judged stale, if it could be read
    os.utime(out / ".lock", (old, old))
    real_read_bytes = Path.read_bytes

    def read_bytes(self):
        if self.name == ".lock":  # another account's lock, say
            raise PermissionError(13, "Permission denied", str(self))
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    unreadable = r"could not take the lock .*cannot be read \(check its permissions\); retry"
    with pytest.raises(ValueError, match=unreadable):
        api._lock_output_dir(out)
    monkeypatch.undo()
    assert api._lock_record((out / ".lock").read_bytes())["pid"] == 4242  # never deleted unread


# ---------------------------------------------------------------------------
# every failure carries a message: names that are not strings, sizes, directories that
# cannot be written or read, seeds that survive JSON
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [["cycle"], {"name": "cycle"}, 3])
def test_names_that_are_not_strings_are_input_errors_naming_the_field(bad):
    spec = dict(SMALL_SPEC, probes=[{"kind": "code_dimension"}])
    collection = {"family": bad, "seed": 1, "realizations": 2}
    cases = [
        (lambda: api.estimate_ensemble(dict(spec, graph={"family": bad}), 2), r"graph\.family"),
        (lambda: api.estimate_ensemble(dict(spec, graph=collection), 2), r"graph\.family"),
        (lambda: api.estimate_ensemble(dict(spec, edge_weights={"family": bad}), 2),
         r"edge_weights\.family"),
        (lambda: api.estimate_ensemble(dict(spec, probes=[{"kind": bad}]), 2),
         r"probes\[0\]\.kind"),
        (lambda: api.run_circuit("H 0", observables=[{"kind": bad}]), r"observables\[0\]\.kind"),
        (lambda: api.capabilities(bad), "section"),
    ]  # fmt: skip
    for run, field in cases:
        message = f"{field} must be a string .*, got {re.escape(repr(bad))}"
        with pytest.raises(ValueError, match=message):
            run()


def test_num_system_is_bounded_before_any_graph_is_generated(monkeypatch):
    def generated(*args, **kwargs):
        raise AssertionError("a graph was generated")

    monkeypatch.setattr(api, "_build_graph", generated)
    too_large = re.escape(f"num_system is too large (at most {api.MAX_QUBITS:,})")
    for n in (10**12, api.MAX_QUBITS + 1):  # 10**12 qubits once ran out of memory
        big = dict(SMALL_SPEC, num_system=n)
        with pytest.raises(ValueError, match=too_large):
            api.estimate_ensemble(big, 2)
        with pytest.raises(ValueError, match=too_large):
            api.run_ensemble(big, 2, seed=1)


def test_an_output_dir_that_cannot_be_written_is_an_input_error(tmp_path, monkeypatch):
    # a path below a regular file can never be created, on any system
    (tmp_path / "a_file").write_text("", encoding="utf-8")
    out = tmp_path / "a_file" / "out"
    not_writable = re.escape(f"output_dir {str(out.resolve())!r} is not writable: ")
    with pytest.raises(ValueError, match=not_writable):
        api.run_circuit("H 0\nM 0", seed=1, output_dir=str(out))
    with pytest.raises(ValueError, match=not_writable):
        api.run_ensemble(SMALL_SPEC, 2, seed=1, output_dir=str(out))
    with pytest.raises(ValueError, match=not_writable):
        api.start_ensemble_job(SMALL_SPEC, 2, seed=1, output_dir=str(out))

    # a directory whose files cannot be written (a full drive, say)
    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(api, "_atomic_savez", full)
    full_dir = tmp_path / "full"
    message = f"output_dir {str(full_dir.resolve())!r} is not writable: [Errno {errno.ENOSPC}] "
    with pytest.raises(ValueError, match=re.escape(message + "No space left on device")):
        api.run_circuit("H 0\nM 0", seed=1, output_dir=str(full_dir))
    monkeypatch.setattr(mipt.EnsembleResult, "save", full)
    with pytest.raises(ValueError, match="is not writable: .*No space left on device"):
        api.run_ensemble(SMALL_SPEC, 2, seed=1, output_dir=str(tmp_path / "full2"))
    assert not (tmp_path / "full2" / ".lock").exists()


def test_saved_parts_that_cannot_be_read_are_input_errors_naming_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_RETRY_DELAYS", (0.001,) * len(api._RETRY_DELAYS))
    out = tmp_path / "unreadable"
    api.run_ensemble(SMALL_SPEC, 10, seed=4, output_dir=str(out))

    def denied(path):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(mipt.EnsembleResult, "load", staticmethod(denied))
    reads = (
        lambda: api.start_ensemble_job(SMALL_SPEC, 20, output_dir=str(out)),
        lambda: api.start_ensemble_job(SMALL_SPEC, output_dir=str(out)),  # to the stored target
        lambda: api.load_results(str(out)),
    )
    unreadable = (
        re.escape(f"could not read results from {str(out.resolve())!r}: [Errno 13] Permission")
        + ".*result.npz.*; close any program that holds the file open, or check its permissions"
    )
    for read in reads:
        with pytest.raises(ValueError, match=unreadable):
            read()
    assert not (out / ".lock").exists()
    assert all(j["output_dir"] != str(out.resolve()) for j in api.list_jobs()["jobs"])
    monkeypatch.undo()
    # a spec.json that holds no run's record
    spec_json = out.resolve() / "spec.json"
    for text, problem in (("{not json", "is not valid JSON"), ("[1, 2]", "holds no run's record")):
        spec_json.write_text(text, encoding="utf-8")
        for read in reads:
            with pytest.raises(ValueError, match=re.escape(f"{str(spec_json)!r} {problem}")):
                read()
    spec_json.write_text(json.dumps({"spec": SMALL_SPEC, "trajectories": 10}), encoding="utf-8")
    with pytest.raises(ValueError, match="its spec.json records no seed; restore it"):
        api.start_ensemble_job(SMALL_SPEC, 20, output_dir=str(out))
    assert not (out / ".lock").exists()
    # run_ensemble's check that a directory holds no run reads it too
    fresh = tmp_path / "fresh"

    def unlistable(path):
        raise PermissionError(13, "Permission denied", str(path / "chunks"))

    monkeypatch.setattr(api, "_result_parts", unlistable)
    with pytest.raises(ValueError, match="could not read results from .*Permission denied"):
        api.run_ensemble(SMALL_SPEC, 2, seed=1, output_dir=str(fresh))
    assert not (fresh / ".lock").exists()


def test_load_results_reads_on_while_a_merge_removes_the_chunks(tmp_path, monkeypatch):
    # a merge replaces result.npz before it deletes chunks/, so a chunks/ that is gone, or
    # (Windows) being deleted, while load_results lists it holds nothing result.npz lacks
    monkeypatch.setattr(api, "_RETRY_DELAYS", (0.001,) * len(api._RETRY_DELAYS))
    out = tmp_path / "vanishing"
    api.run_ensemble(SMALL_SPEC, 10, seed=4, output_dir=str(out))
    (out / "chunks").mkdir()
    real_iterdir = Path.iterdir
    listings: list = []

    def listing(self, error=None, after=0):
        if self.name == "chunks":
            listings.append(self)
            if error is not None and len(listings) > after:
                raise error
        return real_iterdir(self)

    gone = FileNotFoundError(errno.ENOENT, "No such file or directory")
    monkeypatch.setattr(Path, "iterdir", lambda self: listing(self, gone))
    assert api.load_results(str(out))["num_trajectories"] == 10
    # being deleted: the read retries, the check for a pending merge leaves it
    listings.clear()
    deleted = PermissionError(13, "Access is denied")
    monkeypatch.setattr(Path, "iterdir", lambda self: listing(self, deleted, after=1))
    assert api.load_results(str(out))["num_trajectories"] == 10
    assert len(listings) == 2


def test_fresh_seeds_fit_in_a_double(monkeypatch):
    # JavaScript MCP clients (Claude Code among them) read JSON numbers as doubles: a
    # fresh seed above 2**53 would come back rounded, and repeat another run
    monkeypatch.setattr(api.secrets, "randbits", lambda bits: 2**bits - 1)  # the largest draw
    assert api._resolve_seed(None) == 2**53 - 1
    monkeypatch.undo()
    seeds = [api.run_circuit("H 0\nM 0")["seed"] for _ in range(20)]
    assert all(0 <= seed < 2**53 and float(seed) == seed for seed in seeds)


def test_explicit_seeds_of_any_size_are_used_and_reported_exactly(tmp_path):
    direct = _direct_spec()
    for seed in (2**53 + 1, 2**64 + 3):
        assert api.run_circuit("H 0\nM 0", seed=seed)["seed"] == seed
        out = tmp_path / f"seed_{seed}"
        assert api.run_ensemble(SMALL_SPEC, 4, seed=seed, output_dir=str(out))["seed"] == seed
        for record in ("spec.json", "provenance.json"):
            assert json.loads((out / record).read_text(encoding="utf-8"))["seed"] == seed
        assert api.load_results(str(out))["seed"] == seed
        job = api.start_ensemble_job(SMALL_SPEC, 8, output_dir=str(out))  # the stored seed
        assert job["seed"] == seed
        assert api.job_status(job["job_id"], wait_seconds=60)["status"] == "completed"
        np.testing.assert_array_equal(
            mipt.EnsembleResult.load(out / "result.npz").purification_layer,
            direct.run(8, seed=seed).purification_layer,
        )
