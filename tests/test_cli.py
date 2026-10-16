"""The ``sparsegf2`` command line, run in-process (plus one subprocess smoke test)."""

import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import sparsegf2 as sg
from sparsegf2 import api, cli, graphs, mipt

BELL = "H 0\nCX 0 1\nM 0 1\n"

SPEC = {
    "num_system": 8,
    "graph": {"family": "cycle"},
    "layers": 20,
    "measurement_rate": 0.2,
    "reference": "single",
    "checkpoints": 5,
    "probes": [{"kind": "code_dimension"}],
}


def run_cli(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run_json(capsys, *argv: str) -> tuple[int, dict]:
    code, out, _ = run_cli(capsys, *argv, "--json")
    assert out.count("\n") == 1, "--json must print exactly one line"
    return code, json.loads(out)


@pytest.fixture
def bell_file(tmp_path: Path) -> Path:
    path = tmp_path / "bell.stim"
    path.write_text(BELL, encoding="utf-8")
    return path


@pytest.fixture
def spec_file(tmp_path: Path) -> Path:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(SPEC), encoding="utf-8")
    return path


def test_capabilities(capsys):
    code, data = run_json(capsys, "capabilities")
    assert code == 0
    assert data == api.capabilities()
    code, data = run_json(capsys, "capabilities", "--section", "probes")
    assert code == 0 and list(data) == ["probes"]
    code, out, err = run_cli(capsys, "capabilities")
    assert code == 0 and "version:" in out and "done in" in err


def test_inspect(capsys, bell_file):
    code, data = run_json(capsys, "inspect", str(bell_file))
    assert code == 0
    assert data["valid"] and data["num_measurements"] == 2


def test_inspect_invalid_circuit_prints_errors_and_exits_2(capsys, tmp_path):
    bad = tmp_path / "bad.stim"
    bad.write_text("H 0\nNOPE 1\n", encoding="utf-8")
    code, data = run_json(capsys, "inspect", str(bad))
    assert code == 2
    assert not data["valid"] and data["errors"][0]["line"] == 2
    bad.write_text("M 0\nDETECTOR rec[-2]\n", encoding="utf-8")
    code, data = run_json(capsys, "inspect", str(bad))
    assert code == 2 and data["errors"][0]["line"] == 2


def test_inspect_stdin(capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(BELL))
    code, data = run_json(capsys, "inspect", "-")
    assert code == 0 and data["num_qubits"] == 2


def test_run_matches_api(capsys, bell_file):
    code, data = run_json(capsys, "run", str(bell_file), "--shots", "100", "--seed", "4")
    assert code == 0
    expected = api.run_circuit(BELL, shots=100, seed=4)
    for key in ("measurement_means", "first_shots", "random_fraction", "seed"):
        assert data[key] == expected[key]


def test_run_no_phases_and_observables(capsys, bell_file, tmp_path):
    obs = tmp_path / "obs.json"
    obs.write_text(json.dumps([{"kind": "entropy", "region": [0]}]), encoding="utf-8")
    code, data = run_json(
        capsys, "run", str(bell_file), "--seed", "1", "--no-phases", "--observables", f"@{obs}"
    )
    assert code == 0
    assert not data["phases"] and "measurements" not in data
    assert data["observables"] == {"entropy": 0}
    code, data = run_json(
        capsys,
        "run",
        str(bell_file),
        "--seed",
        "1",
        "--observables",
        '[{"kind": "entropy", "region": "0:1"}]',
        "--observe-ticks",
    )
    assert code == 0 and data["tick_observables"]["num_ticks"] == 0


def test_run_output_dir_records_command(capsys, bell_file, tmp_path):
    code, data = run_json(
        capsys, "run", str(bell_file), "--seed", "2", "--output-dir", str(tmp_path / "out")
    )
    assert code == 0
    prov = json.loads(Path(data["files"]["provenance"]).read_text(encoding="utf-8"))
    assert prov["seed"] == 2 and prov["command"].startswith("sparsegf2")


def test_run_validation_error_exit_code(capsys, tmp_path):
    bad = tmp_path / "bad.stim"
    bad.write_text("H 0\nNOPE 1\n", encoding="utf-8")
    code, data = run_json(capsys, "run", str(bad))
    assert code == 2
    assert data["error"]["type"] == "ValidationError"
    assert "line 2" in data["error"]["message"]
    code, out, err = run_cli(capsys, "run", str(bad))
    assert code == 2 and out == "" and "line 2" in err


def test_bad_json_and_missing_file(capsys, bell_file, tmp_path):
    code, data = run_json(capsys, "run", str(bell_file), "--observables", "[not json")
    assert code == 2 and "not valid JSON" in data["error"]["message"]
    code, data = run_json(capsys, "run", str(tmp_path / "missing.stim"))
    assert code == 2 and "no such file" in data["error"]["message"]


def test_usage_errors(capsys):
    code, out, err = run_cli(capsys)
    assert code == 2 and "COMMAND" in err
    code, data = run_json(capsys, "run")
    assert code == 2 and data["error"]["type"] == "UsageError"
    code, out, err = run_cli(capsys, "frobnicate")
    assert code == 2


def test_help_starts_with_examples(capsys):
    for argv in (["--help"], ["run", "--help"], ["ensemble", "--help"], ["inspect", "--help"]):
        code, out, _ = run_cli(capsys, *argv)
        assert code == 0
        description = out.split("\n\n", 1)[1]
        assert description.startswith("examples:"), argv


def test_version(capsys):
    code, out, _ = run_cli(capsys, "--version")
    assert code == 0 and sg.__version__ in out


def test_ensemble_matches_library(capsys, spec_file, tmp_path):
    out_dir = tmp_path / "ens"
    code, data = run_json(
        capsys, "ensemble", str(spec_file), "--trajectories", "30", "--seed", "5",
        "--output-dir", str(out_dir),
    )  # fmt: skip
    assert code == 0
    assert data == {**api.run_ensemble(SPEC, 30, seed=5), "files": data["files"],
                    "elapsed_seconds": data["elapsed_seconds"],
                    "trajectories_per_second": data["trajectories_per_second"]}  # fmt: skip
    # The saved spec.json repeats the run exactly.
    code, again = run_json(capsys, "ensemble", str(out_dir / "spec.json"))
    assert code == 0
    assert again["purification"] == data["purification"]
    code, loaded = run_json(capsys, "results", str(out_dir))
    assert code == 0 and loaded["purification"] == data["purification"]


def test_ensemble_chunked_is_resumable(capsys, spec_file, tmp_path):
    out_dir = tmp_path / "chunked"
    argv = ["ensemble", str(spec_file), "--trajectories", "40", "--seed", "3",
            "--output-dir", str(out_dir), "--chunk-size", "15"]  # fmt: skip
    code, out, err = run_cli(capsys, *argv, "--json")
    assert code == 0, err
    data = json.loads(out)
    assert data["job"]["status"] == "completed" and data["num_trajectories"] == 40
    saved = mipt.EnsembleResult.load(out_dir / "result.npz")
    direct = api._build_spec(SPEC).run(40, seed=3)
    assert np.array_equal(saved.purification_layer, direct.purification_layer)
    assert "40/40 trajectories" in err
    code, data = run_json(capsys, *argv)
    assert code == 0 and data["job"]["resumed_from"] == 40


def test_ensemble_ctrl_c_exits_130_with_partial_summary(capsys, spec_file, tmp_path, monkeypatch):
    real_status = api.job_status
    calls = []

    def interrupted_once(job_id, wait_seconds=0.0):
        calls.append(job_id)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return real_status(job_id, wait_seconds)

    monkeypatch.setattr(api, "job_status", interrupted_once)
    out_dir = tmp_path / "interrupted"
    code, out, err = run_cli(
        capsys, "ensemble", str(spec_file), "--trajectories", "200", "--seed", "1",
        "--output-dir", str(out_dir), "--chunk-size", "10", "--json",
    )  # fmt: skip
    assert code == cli.EXIT_INTERRUPTED, err
    data = json.loads(out)
    assert data["job"]["status"] in ("cancelled", "completed")
    assert data["num_trajectories"] == data["job"]["trajectories_done"] > 0
    assert "rerun the same command to resume" in err
    assert not (out_dir / ".lock").exists()  # the rerun is not refused


def test_ensemble_chunked_shows_the_job_notes(capsys, spec_file, tmp_path, monkeypatch):
    out_dir = tmp_path / "notes"
    argv = ["ensemble", str(spec_file), "--trajectories", "10", "--seed", "1",
            "--output-dir", str(out_dir), "--chunk-size", "5"]  # fmt: skip
    # a reader holds result.npz through the final merge (and the summary's load)
    monkeypatch.setattr(api, "_RETRY_DELAYS", (0.001,) * len(api._RETRY_DELAYS))
    real_replace = os.replace

    def replace(src, dst):
        if Path(dst).name == "result.npz":
            raise PermissionError(13, "the file is in use by another process", str(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    code, data = run_json(capsys, *argv)
    assert code == 0 and data["num_trajectories"] == 10
    assert data["job"]["notes"] == [api._MERGE_PENDING]
    monkeypatch.undo()
    # a resume's notes: a spec written differently that builds the same protocol
    spec_file.write_text(json.dumps({**SPEC, "schedule": "matching"}), encoding="utf-8")
    code, data = run_json(capsys, *argv)
    assert code == 0 and data["job"]["resumed_from"] == 10
    assert any("written differently" in note for note in data["job"]["notes"])
    assert not (out_dir / "chunks").exists()  # the pending merge was done


def test_ensemble_refuses_an_output_dir_another_process_writes_to(
    capsys, spec_file, tmp_path, monkeypatch
):
    real_state = api._process_state
    monkeypatch.setattr(
        api, "_process_state", lambda pid: (True, None) if pid == 4242 else real_state(pid)
    )
    out_dir = tmp_path / "held"
    out_dir.mkdir()
    started = datetime.now(UTC).isoformat(timespec="seconds")
    holder = {"pid": 4242, "host": socket.gethostname(), "started": started}  # running
    (out_dir / ".lock").write_text(json.dumps(holder), encoding="utf-8")
    for extra in ([], ["--chunk-size", "5"]):
        code, data = run_json(
            capsys, "ensemble", str(spec_file), "--trajectories", "10", "--seed", "1",
            "--output-dir", str(out_dir), *extra,
        )  # fmt: skip
        assert code == 2, data
        assert (
            f"another process (pid 4242 on host {socket.gethostname()}, started {started})"
            in (data["error"]["message"])
        )
    assert [p.name for p in out_dir.iterdir()] == [".lock"]


def test_ensemble_graph_file_collection_runs_every_circuit_without_trajectories(capsys, tmp_path):
    n = 6
    bank = [graphs.watts_strogatz(n, 2, 0.3, seed=s) for s in range(3)]
    graphs.save(tmp_path / "bank.npz", bank, n)
    graph = {"family": "file", "path": str(tmp_path / "bank.npz")}
    spec = {**SPEC, "num_system": n, "graph": graph}
    spec_path = tmp_path / "bank.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    out_dir = tmp_path / "bank"
    code, data = run_json(
        capsys, "ensemble", str(spec_path), "--seed", "1", "--output-dir", str(out_dir)
    )
    assert code == 0 and data["num_trajectories"] == 3  # one circuit per graph of the file
    direct = mipt.MonitoredCircuit(
        num_system=n, edges=bank, layers=20, measurement_rate=0.2, reference="single",
        checkpoints=5, probes=[mipt.code_dimension()],
    ).run(seed=1)  # fmt: skip
    saved = mipt.EnsembleResult.load(out_dir / "result.npz")
    assert np.array_equal(saved.purification_layer, direct.purification_layer)
    # the run's spec.json, which records the graph file beside the spec, repeats it
    code, again = run_json(capsys, "ensemble", str(out_dir / "spec.json"))
    assert code == 0 and again["purification"] == data["purification"]
    # a spec the CLI cannot read as a collection is the API's to refuse, as invalid input
    spec_path.write_text(json.dumps({**SPEC, "graph": 5}), encoding="utf-8")
    code, data = run_json(capsys, "ensemble", str(spec_path))
    assert code == 2 and "graph must be an object" in data["error"]["message"]


def test_ensemble_chunked_resume_runs_to_the_stored_target(capsys, spec_file, tmp_path):
    out_dir = tmp_path / "crashed"
    argv = ["ensemble", str(spec_file), "--output-dir", str(out_dir), "--chunk-size", "2"]
    code, data = run_json(capsys, *argv, "--trajectories", "6", "--seed", "3")
    assert code == 0 and data["num_trajectories"] == 6
    # a crash after the first chunk: spec.json asks for 6, one chunk of 2 is saved
    direct = mipt.MonitoredCircuit(
        num_system=8, edges=graphs.cycle(8), layers=20, measurement_rate=0.2,
        reference="single", checkpoints=5, probes=[mipt.code_dimension()],
    )  # fmt: skip
    (out_dir / "result.npz").unlink()
    (out_dir / "chunks").mkdir()
    direct.run(2, seed=3).save(out_dir / "chunks" / "traj_000000000000.npz")
    code, data = run_json(capsys, *argv)  # the same command, without --trajectories or --seed
    assert code == 0 and data["job"]["resumed_from"] == 2 and data["num_trajectories"] == 6
    saved = mipt.EnsembleResult.load(out_dir / "result.npz")
    assert np.array_equal(saved.purification_layer, direct.run(6, seed=3).purification_layer)


def test_ensemble_errors(capsys, spec_file, tmp_path):
    code, data = run_json(capsys, "ensemble", str(spec_file))
    message = data["error"]["message"]  # the API's, for one graph
    assert code == 2 and "give trajectories" in message and "--trajectories" in message
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"spec": SPEC, "trajectories": 2, "extra": 1}), encoding="utf-8")
    code, data = run_json(capsys, "ensemble", str(wrapped))
    assert code == 2 and "['extra']" in data["error"]["message"]
    assert "graph_file" in data["error"]["message"]
    code, data = run_json(
        capsys, "ensemble", str(spec_file), "--trajectories", "5", "--chunk-size", "2"
    )
    assert code == 2 and "--output-dir" in data["error"]["message"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**SPEC, "layrs": 3}), encoding="utf-8")
    code, data = run_json(capsys, "ensemble", str(bad), "--trajectories", "5")
    assert code == 2 and "unknown key" in data["error"]["message"]


def test_internal_errors_exit_1(capsys, monkeypatch, bell_file):
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(api, "run_circuit", boom)
    code, data = run_json(capsys, "run", str(bell_file))
    assert code == 1 and data["error"]["type"] == "RuntimeError"
    assert data["error"]["message"] == "internal error: kaboom"

    def wrong_type(*args, **kwargs):  # the API raises ValueError for input: this is a bug
        raise TypeError("'NoneType' object is not subscriptable")

    monkeypatch.setattr(api, "run_circuit", wrong_type)
    code, data = run_json(capsys, "run", str(bell_file))
    assert code == 1 and data["error"] == {
        "type": "TypeError",
        "message": "internal error: 'NoneType' object is not subscriptable",
    }


def test_names_that_are_not_strings_exit_2(capsys, bell_file, tmp_path):
    spec_path = tmp_path / "list_family.json"
    spec_path.write_text(json.dumps({**SPEC, "graph": {"family": ["cycle"]}}), encoding="utf-8")
    code, data = run_json(capsys, "ensemble", str(spec_path), "--trajectories", "2")
    assert code == 2 and "graph.family must be a string" in data["error"]["message"]
    observables = '[{"kind": ["entropy"]}]'
    code, data = run_json(capsys, "run", str(bell_file), "--observables", observables)
    assert code == 2 and "observables[0].kind must be a string" in data["error"]["message"]


def test_paths_that_cannot_be_read_or_written_exit_2(capsys, monkeypatch, bell_file, spec_file,
                                                      tmp_path):  # fmt: skip
    (tmp_path / "a_file").write_text("", encoding="utf-8")
    real_mkdir = Path.mkdir

    def mkdir(self, *args, **kwargs):  # a directory this process may not create
        if self.name == "denied":
            raise PermissionError(13, "Permission denied", str(self))
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    for out, expected in (
        (tmp_path / "a_file" / "out", "is a file, not a directory"),  # below a regular file
        (tmp_path / "denied", "is not writable: [Errno 13] Permission denied"),
    ):
        for argv in (
            ["run", str(bell_file)],
            ["ensemble", str(spec_file), "--trajectories", "2"],
            ["ensemble", str(spec_file), "--trajectories", "2", "--chunk-size", "1"],
        ):
            code, data = run_json(capsys, *argv, "--output-dir", str(out))
            assert code == 2 and expected in data["error"]["message"], argv
    monkeypatch.undo()
    # a file this process may not read (Windows: one another program holds), or not UTF-8
    real_read_text = Path.read_text

    def read_text(self, *args, **kwargs):
        if self.name == "bell.stim":
            raise PermissionError(13, "Permission denied", str(self))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    code, data = run_json(capsys, "run", str(bell_file))
    assert code == 2 and data["error"]["type"] == "ValidationError"
    assert data["error"]["message"].startswith(f"could not read {str(bell_file)!r}: [Errno 13]")
    monkeypatch.undo()
    latin1 = tmp_path / "latin1.stim"
    latin1.write_bytes(b"H 0\n# caf\xe9\n")
    code, data = run_json(capsys, "run", str(latin1))
    assert code == 2 and data["error"]["message"].startswith(f"{str(latin1)!r} is not UTF-8")


def test_python_m_cli_subprocess(tmp_path):
    env = dict(os.environ)
    src = str(Path(sg.__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "sparsegf2.cli", "run", "-", "--seed", "1", "--json"],
        input=BELL,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["measurements"] == api.run_circuit(BELL, seed=1)["measurements"]


_BLOCK_MCP = """
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("No module named 'mcp' (blocked for this test)")
sys.meta_path.insert(0, _Block())
import sparsegf2, sparsegf2.api, sparsegf2.cli, sparsegf2.mcp_server
assert "mcp" not in sys.modules
sys.exit(sparsegf2.cli.main(["mcp", "--json"]))
"""


def test_core_works_without_mcp_installed():
    env = dict(os.environ)
    src = str(Path(sg.__file__).resolve().parents[1])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCK_MCP], capture_output=True, text=True, env=env,
        timeout=120, check=False,
    )  # fmt: skip
    assert proc.returncode == 2, proc.stderr
    assert "pip install 'sparsegf2[mcp]'" in json.loads(proc.stdout)["error"]["message"]


# ---------------------------------------------------------------------------
# regions and curves: the erasure tools
# ---------------------------------------------------------------------------


def regions_argv(**flags: list[str] | None) -> list[str]:
    """``sparsegf2 regions`` on a 32-node ring, with the given options replaced (None drops one)."""
    options: dict[str, list[str] | None] = {
        "family": ["cycle"], "n": ["32"], "size": ["8"], "couplings": ["0", "1", "2"],
        "samples": ["100"], "seed": ["3"],
    }  # fmt: skip
    options.update(flags)
    argv = ["regions"]
    for name, values in options.items():
        if values is not None:
            argv += ["--" + name.replace("_", "-"), *values]
    return argv


def _timeless(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in ("elapsed_seconds", "estimated_seconds")}


def test_regions_matches_the_api(capsys):
    code, data = run_json(capsys, *regions_argv())
    assert code == 0
    expected = api.sample_erasure_regions(
        {"family": "cycle"}, 8, [0, 1, 2], 100, num_nodes=32, seed=3
    )
    assert _timeless(data) == _timeless(json.loads(json.dumps(expected)))
    assert data["method"] == "exact" and data["seed"] == 3


def test_regions_family_parameters_and_options(capsys):
    argv = regions_argv(
        family=["watts_strogatz"], n=["24"], k=["4"], beta=["0.2"], graph_seed=["5"], size=["6"],
        couplings=["-1", "0", "1.5"], samples=["40"], chains=["2"], burn_in=["30"], thin=["2"],
        seed=["9"], method=["chain"], max_step=["0.5"], no_tempering=[],
    )  # fmt: skip
    code, data = run_json(capsys, *argv)
    assert code == 0, data
    graph = {"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 5}
    expected = api.sample_erasure_regions(
        graph, 6, [-1, 0, 1.5], 40, num_nodes=24, seed=9, chains=2, burn_in=30, thin=2,
        tempering=False, method="chain", max_step=0.5,
    )  # fmt: skip
    assert _timeless(data) == _timeless(json.loads(json.dumps(expected)))
    assert data["tempering"] is False and data["burn_in"] == 30 and data["thin"] == 2
    # another family parameter, as JSON
    argv = regions_argv(
        family=["erdos_renyi"], n=["16"], graph_param=["p=0.3"], graph_seed=["1"], size=["4"],
        couplings=["0", "1"], samples=["30"],
    )  # fmt: skip
    code, data = run_json(capsys, *argv)
    assert code == 0, data
    assert data["graph"]["family"] == "erdos_renyi"


def test_regions_reads_a_graph_file(capsys, tmp_path):
    n = 12
    graphs.save(tmp_path / "g.npz", [graphs.watts_strogatz(n, 4, 0.3, seed=2)], n)
    argv = regions_argv(
        family=None, n=None, graph=[str(tmp_path / "g.npz")], size=["4"], couplings=["0", "1"],
        samples=["30"], seed=["1"],
    )  # fmt: skip
    code, data = run_json(capsys, *argv)
    assert code == 0, data
    assert data["graph"]["num_nodes"] == n and data["graph"]["family"] == "file"


def test_the_regions_time_limit_is_the_api_limit(capsys):
    assert cli.REGIONS_MAX_SECONDS == api.SYNC_ENSEMBLE_MAX_SECONDS
    code, out, _ = run_cli(capsys, "regions", "--help")
    assert f"({api.SYNC_ENSEMBLE_MAX_SECONDS:g})" in out


def test_regions_table(capsys):
    code, out, err = run_cli(capsys, *regions_argv())
    assert code == 0 and "done in" in err
    assert "mean b" in out and "R-hat" in out and "Proposition 2" in out and "limit" in out
    assert "against z = " in out and "the step nearest a rise" in out
    assert "ring closed form" in out and "example regions at J = 2" in out
    rows = [line for line in out.splitlines() if line.lstrip().startswith(("0 ", "1 ", "2 "))]
    assert len(rows) >= 3  # one line per coupling
    argv = regions_argv(
        family=["watts_strogatz"], n=["64"], k=["4"], beta=["0.1"], graph_seed=["1"],
        size=["16"], couplings=["0", "2"], samples=["40"], burn_in=["5"], seed=["1"],
    )  # fmt: skip
    code, out, _ = run_cli(capsys, *argv)
    assert code == 0 and "swap chain, burn-in 5 sweeps" in out and "mixing:" in out


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ({"size": ["40"]}, "size must be between 1 and 31"),
        ({"couplings": ["0", "0"]}, "distinct"),
        ({"samples": ["2"]}, "samples must be at least 8"),
        ({"graph_param": ["oops"]}, "KEY=VALUE"),
        ({"graph": ["x.npz"]}, "carries its own graph"),
        ({"method": ["magic"]}, "method must be"),
        ({"ladder": ["dense"]}, "--ladder takes"),
        ({"max_seconds": ["0"]}, "max_seconds=0"),
    ],
)
def test_regions_errors_exit_2(capsys, flags, message):
    code, data = run_json(capsys, *regions_argv(**flags))
    assert code == 2, data
    assert data["error"]["type"] == "ValidationError" and message in data["error"]["message"]


def test_regions_needs_a_graph_and_the_required_options(capsys):
    code, data = run_json(capsys, *regions_argv(family=None, n=None))
    assert code == 2 and "give the graph" in data["error"]["message"]
    code, data = run_json(capsys, *regions_argv(size=None))
    assert code == 2 and data["error"]["type"] == "UsageError"
    assert "--size" in data["error"]["message"]
    code, data = run_json(capsys, *regions_argv(n=None))
    assert code == 2 and "give num_nodes" in data["error"]["message"]


def test_regions_refuses_what_would_take_minutes(capsys):
    argv = regions_argv(
        family=["watts_strogatz"], n=["2000"], k=["4"], beta=["0.1"], graph_seed=["1"],
        size=["500"], samples=["2000"], chains=["8"],
    )  # fmt: skip
    code, data = run_json(capsys, *argv)
    assert code == 2 and "estimated at about" in data["error"]["message"]
    assert "--max-seconds" in data["error"]["message"]


@pytest.fixture(scope="module")
def erasure_run(tmp_path_factory) -> Path:
    spec = {
        "num_system": 12, "graph": {"family": "cycle"}, "layers": 16, "measurement_rate": 0.1,
        "schedule": "matching", "reference": "purification", "checkpoints": [8, 16],
        "probes": [
            {"kind": "code_dimension"},
            {"kind": "erasure_first_failure", "samples": 100},
            {"kind": "uncorrectable_region_counts", "max_size": 2},
            {"kind": "correlated_erasure", "couplings": [0, 1.5], "samples": 60},
        ],
    }  # fmt: skip
    out = tmp_path_factory.mktemp("cli_curves") / "run"
    api.run_ensemble(spec, 30, seed=4, output_dir=str(out))
    return out


def test_curves_matches_the_api(capsys, erasure_run):
    code, data = run_json(
        capsys, "curves", str(erasure_run), "--eps", "0.05", "0.1", "0.3", "--delta", "0.05",
        "--layer", "8",
    )  # fmt: skip
    assert code == 0, data
    expected = api.erasure_curves(str(erasure_run), [0.05, 0.1, 0.3], 0.05, 8)
    assert _timeless(data) == _timeless(json.loads(json.dumps(expected)))
    assert data["layer"] == 8 and data["uniform"]["erasure_first_failure"]["layer"] == 8
    code, data = run_json(capsys, "curves", str(erasure_run / "result.npz"))
    assert code == 0 and len(data["e"]) >= 4


def test_curves_table(capsys, erasure_run):
    code, out, err = run_cli(capsys, "curves", str(erasure_run), "--eps", "0.05", "0.1", "0.3")
    assert code == 0 and "done in" in err
    assert "independent erasure" in out and "correlated erasure" in out
    assert "Phi J=0" in out and "Phi J=1.5" in out and "e_delta" in out and "stderr" in out
    assert "hold a code" in out


def test_curves_errors_exit_2(capsys, erasure_run, tmp_path):
    code, data = run_json(capsys, "curves", str(tmp_path / "nowhere"))
    assert code == 2 and "no such file or directory" in data["error"]["message"]
    code, data = run_json(capsys, "curves", str(erasure_run), "--layer", "7")
    assert code == 2 and "layer 7 was not evaluated" in data["error"]["message"]
    code, data = run_json(capsys, "curves", str(erasure_run), "--delta", "3")
    assert code == 2 and "delta must be in [0, 1]" in data["error"]["message"]
    code, data = run_json(capsys, "curves", str(erasure_run), "--eps", "2")
    assert code == 2 and "e must be numbers in [0, 1]" in data["error"]["message"]
    plain = {**SPEC, "probes": [{"kind": "code_dimension"}]}
    api.run_ensemble(plain, 3, seed=1, output_dir=str(tmp_path / "plain"))
    code, data = run_json(capsys, "curves", str(tmp_path / "plain"))
    assert code == 2
    assert "no erasure_first_failure or correlated_erasure probe" in data["error"]["message"]
    code, data = run_json(capsys, "curves")
    assert code == 2 and data["error"]["type"] == "UsageError"


def test_erasure_commands_help_starts_with_examples(capsys):
    for argv in (["regions", "--help"], ["curves", "--help"]):
        code, out, _ = run_cli(capsys, *argv)
        assert code == 0
        assert out.split("\n\n", 1)[1].startswith("examples:"), argv


def test_internal_errors_of_the_erasure_commands_exit_1(capsys, monkeypatch, erasure_run):
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(api, "erasure_curves", boom)
    code, data = run_json(capsys, "curves", str(erasure_run))
    assert code == 1
    assert data["error"] == {"type": "RuntimeError", "message": "internal error: kaboom"}
    monkeypatch.setattr(api, "sample_erasure_regions", boom)
    code, data = run_json(capsys, *regions_argv())
    assert code == 1 and data["error"]["message"] == "internal error: kaboom"
