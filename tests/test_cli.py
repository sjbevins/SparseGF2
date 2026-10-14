"""The ``sparsegf2`` command line, run in-process (plus one subprocess smoke test)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import sparsegf2 as sg
from sparsegf2 import api, cli, mipt

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


def test_ensemble_errors(capsys, spec_file, tmp_path):
    code, data = run_json(capsys, "ensemble", str(spec_file))
    assert code == 2 and "--trajectories" in data["error"]["message"]
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
