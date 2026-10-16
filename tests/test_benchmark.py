"""``benchmark_ensemble``: measured throughput on this machine, projected to other thread counts.

The measurements themselves vary from machine to machine, so the tests check structure,
invariants and the projection arithmetic (a pure function), not speeds.
"""

import json

import numba
import pytest

from sparsegf2 import api, cli

SPEC = {
    "num_system": 8,
    "graph": {"family": "cycle"},
    "layers": 20,
    "measurement_rate": 0.2,
    "reference": "single",
    "checkpoints": 4,
    "probes": [{"kind": "code_dimension"}, {"kind": "entropy_profile", "checkpoints": "final"}],
}


def available() -> int:
    return numba.config.NUMBA_NUM_THREADS


def test_projection_arithmetic():
    f = api._parallel_threads
    # one thread measured: the model alone
    assert api._projected_rate(10.0, None, 1, 1) == 10.0
    assert api._projected_rate(10.0, None, 1, 4) == pytest.approx(10.0 * f(4) / f(1))
    # two measured: the model is calibrated to reproduce the P-thread rate
    assert api._projected_rate(10.0, 30.0, 4, 4) == pytest.approx(30.0)
    assert api._projected_rate(10.0, 30.0, 4, 1) == pytest.approx(10.0)
    c = 30.0 / (10.0 * f(4) / f(1))
    assert api._projected_rate(10.0, 30.0, 4, 2) == pytest.approx(10.0 * f(2) / f(1) * c)


def test_the_structure_and_invariants(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = api.benchmark_ensemble(SPEC, trajectories=50, threads=[1, 64], seconds=1)
    p = min(64, available(), 50)
    assert out["trajectories"] == 50 and out["seed"] == 0
    assert out["threads_available"] == available() and out["cpus"] >= 1
    assert out["warmup_seconds"] > 0 and out["estimated_seconds"] > 0
    assert out["estimate_ratio"] > 0
    measured = out["measured"]
    assert [m["threads"] for m in measured] == ([1] if p == 1 else [1, p])
    for m in measured:
        assert 1 <= m["trajectories"] <= 50
        assert m["seconds"] > 0 and m["trajectories_per_second"] > 0
    projected = out["projected"]
    assert [row["threads"] for row in projected] == [1, 64]
    assert all(row["seconds"] > 0 for row in projected)
    assert projected[0]["basis"] == "measured"
    assert projected[0]["seconds"] >= projected[1]["seconds"]
    busy = min(64, 50)  # threads beyond the trajectories stay idle
    assert projected[1]["basis"] == ("measured" if busy in (1, p) else "projected")
    notes = " ".join(out.get("notes", []))
    assert ("projected beyond the" in notes) == (available() < 64)
    storage = out["storage"]
    assert 1 <= storage["sample_trajectories"] <= 50
    assert storage["disk_megabytes"] > 0 and storage["memory_megabytes"] > 0
    assert list(tmp_path.iterdir()) == []  # nothing is written to the working directory
    json.dumps(out)


def test_a_measured_count_is_reproduced_by_the_projection():
    out = api.benchmark_ensemble(SPEC, trajectories=40, threads=[1, 3], seconds=1)
    by = {m["threads"]: m for m in out["measured"]}
    for row in out["projected"]:
        if row["basis"] == "measured":
            m = by[min(row["threads"], 40, available())]
            assert row["seconds"] == pytest.approx(40 / m["trajectories_per_second"], rel=1e-4)


def test_one_thread_and_one_trajectory():
    out = api.benchmark_ensemble(SPEC, trajectories=1, threads=[1], seconds=1)
    assert [m["threads"] for m in out["measured"]] == [1]
    assert out["measured"][0]["trajectories"] == 1
    assert [(r["threads"], r["basis"]) for r in out["projected"]] == [(1, "measured")]
    assert "notes" not in out or "projected beyond" not in " ".join(out["notes"])


def test_default_threads_are_one_and_all():
    out = api.benchmark_ensemble(SPEC, trajectories=20, seconds=1)
    assert [r["threads"] for r in out["projected"]] == sorted({1, available()})


def test_a_graph_collection_runs_all_its_circuits_by_default():
    spec = dict(
        SPEC,
        graph={"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 1, "realizations": 3},
        circuits_per_graph=2,
    )
    out = api.benchmark_ensemble(spec, threads=[1], seconds=1)
    assert out["trajectories"] == 6


@pytest.mark.parametrize(
    ("kwargs", "word"),
    [
        ({"threads": [0]}, "threads"),
        ({"threads": "4"}, "threads"),
        ({"threads": []}, "threads"),
        ({"threads": [True]}, "threads"),
        ({"seconds": 0}, "seconds"),
        ({"seconds": -1}, "seconds"),
        ({"seconds": 601}, "seconds"),
        ({"seconds": "9"}, "seconds"),
        ({"seconds": True}, "seconds"),
        ({"seed": -1}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"trajectories": 0}, "trajectories"),
    ],
)
def test_bad_arguments_are_refused(kwargs, word):
    arguments = {"trajectories": 4, "seconds": 1, **kwargs}
    with pytest.raises(ValueError, match=word):
        api.benchmark_ensemble(SPEC, **arguments)


def test_a_single_graph_needs_trajectories():
    with pytest.raises(ValueError, match="trajectories"):
        api.benchmark_ensemble(SPEC, seconds=1)


def test_the_command_line(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(SPEC), encoding="utf-8")
    argv = ["benchmark", str(path), "--trajectories", "30", "--threads", "1", "64"]
    code = cli.main([*argv, "--seconds", "1", "--seed", "2", "--json"])
    captured = capsys.readouterr()
    assert code == 0 and captured.out.count("\n") == 1
    data = json.loads(captured.out)
    assert data["trajectories"] == 30 and data["seed"] == 2
    assert [r["threads"] for r in data["projected"]] == [1, 64]

    code = cli.main([*argv, "--seconds", "1"])
    captured = capsys.readouterr()
    assert code == 0
    assert "threads" in captured.out and "disk" in captured.out

    code = cli.main([*argv, "--seconds", "0", "--json"])
    assert code == 2
    assert "seconds" in json.loads(capsys.readouterr().out)["error"]["message"]
