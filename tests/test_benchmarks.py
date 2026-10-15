"""The benchmark scripts' smoke modes run and write well-formed results."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_thread_scaling_quick(tmp_path):
    out = tmp_path / "ts"
    # benchmark this checkout's src, not whatever an editable install points at
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]),
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "thread_scaling.py"),
            "--quick",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads((out / "thread_scaling.json").read_text(encoding="utf-8"))
    assert (out / "thread_scaling.md").read_text(encoding="utf-8").startswith("# Thread scaling")
    env = data["environment"]
    for key in ("cpu_model", "physical_cores", "logical_processors", "numba_num_threads"):
        assert key in env
    assert env["threading_layer"] in ("omp", "tbb", "workqueue")
    assert data["identical_across_threads"] is True
    # the identity check is only meaningful with two or more thread counts
    assert len(data["thread_counts"]) >= min(2, env["numba_num_threads"])
    for w in data["workloads"]:
        assert [r["threads"] for r in w["rows"]] == data["thread_counts"]
    assert [w["name"] for w in data["workloads"]] == ["W1", "W2", "W3", "W4"]
    for w in data["workloads"]:
        assert w["rows"][0]["threads"] == 1 and w["rows"][0]["speedup"] == 1.0
        for row in w["rows"]:
            assert {"seconds", "throughput", "speedup", "efficiency"} <= row.keys()
            assert row["seconds"] > 0
