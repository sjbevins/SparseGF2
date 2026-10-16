"""The benchmark scripts' smoke modes run and write well-formed results."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _src_env() -> dict:
    """The environment for a benchmark subprocess: this checkout's src first on the path."""
    # benchmark this checkout's src, not whatever an editable install points at
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]),
    }


def test_thread_scaling_quick(tmp_path):
    out = tmp_path / "ts"
    env = _src_env()
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


def test_erasure_cost_quick(tmp_path):
    out = tmp_path / "ec"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "erasure_cost.py"),
            "--quick",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
        env=_src_env(),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads((out / "erasure_cost.json").read_text(encoding="utf-8"))
    md = (out / "erasure_cost.md").read_text(encoding="utf-8")
    assert md.startswith("# Erasure enumeration and sampling cost")
    for heading in ("## Exhaustive counts", "## First-failure orderings", "## Budgets"):
        assert heading in md
    assert data["quick"] is True and data["environment"]["threads"] == 1
    sizes, rates = data["settings"]["sizes"], data["settings"]["rates"]
    assert [(c["n"], c["p"]) for c in data["cases"]] == [(n, p) for n in sizes for p in rates]
    for case in data["cases"]:
        n = case["n"]
        assert case["codes"] and all(c["k"] > 0 for c in case["codes"])
        for c in case["codes"]:
            assert 1 <= c["distance_low"] <= c["distance_high"] <= n
        # quick mode stops at w = 3 and extrapolates w = 4, labelled as such
        assert [lv["w"] for lv in case["levels"]] == [1, 2, 3]
        assert case["next_level"]["w"] == 4 and "not measured" in case["next_level"]["basis"]
        for lv in case["levels"]:
            assert lv["n_exact"] == sum(math.comb(n, m) for m in range(1, lv["w"] + 1))
            assert 0 < lv["library"]["median"] <= lv["library"]["max"]
            assert lv["engine"]["median"] <= lv["engine"]["max"]
            assert lv["seconds_per_region"] == lv["library"]["median"] / lv["n_exact"]
        assert [o["orderings"] for o in case["orderings"]] == [100, 512, 2048]
    budgets = {(b["budget"], b["n"]): b for b in data["budgets"]}
    assert {b for b, _ in budgets} == {10_000, 50_000, 100_000, 1_000_000, 10_000_000}
    for (b, n), row in budgets.items():
        w = row["m_star"]  # m*(n) = max{w : N_exact(n, w) <= B}, recomputed here
        assert sum(math.comb(n, m) for m in range(1, w + 1)) <= b
        assert w == n or sum(math.comb(n, m) for m in range(1, w + 2)) > b
        assert row["seconds_per_code"] > 0
        assert row["source"] == "measured" or row["source"].startswith("extrapolated")
    study = data["study"]
    assert study["rows"] and study["cpu_seconds"] > 0
    assert study["wall_seconds_low"] <= study["wall_seconds_high"]


def test_probe_cost_quick(tmp_path):
    out = tmp_path / "pc"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "probe_cost.py"), "--quick", "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
        env=_src_env(),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads((out / "probe_cost.json").read_text(encoding="utf-8"))
    md = (out / "probe_cost.md").read_text(encoding="utf-8")
    assert md.startswith("# Probe cost against the runtime estimate")
    assert data["quick"] is True and data["environment"]["threads"] == 1
    sizes, states = data["settings"]["sizes"], data["settings"]["states"]
    assert [(b["n"], b["state"]) for b in data["bases"]] == [(n, s) for n in sizes for s in states]
    for b in data["bases"]:
        # the states: k = 1 with a single reference, k = n scrambled in the purification picture
        if b["state"] == "single":
            assert b["k"] == 1
        if b["state"] == "purification":
            assert b["k"] == b["n"]
        assert b["seconds"] > 0 and b["estimate_seconds"] > 0
    labels = data["settings"]["labels"]
    rows = [r for r in data["rows"] if "measured_seconds" in r]
    assert len(rows) == len(sizes) * len(states) * len(labels)
    for r in rows:
        assert r["measured_seconds"] >= 0 and r["estimate_seconds"] > 0 and r["copies"] >= 1
        assert r["ratio"] == r["estimate_seconds"] / max(r["measured_seconds"], 1e-9)
    assert "## All threads" in md
    assert [a["n"] for a in data["all_threads"]] == [16] * 6
    for a in data["all_threads"]:
        assert a["threads"] == data["settings"]["all_threads"] >= 1
        assert a["seconds"] > 0 and a["alone_seconds"] > 0 and a["estimate_seconds"] > 0
        assert a["speed_under_load"] == a["alone_seconds"] / a["seconds"]
    # re-pricing the saved measurements reproduces every estimate
    again = tmp_path / "again"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "probe_cost.py"), "--from-json",
         str(out / "probe_cost.json"), "--out", str(again)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
        env=_src_env(),
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    repriced = json.loads((again / "probe_cost.json").read_text(encoding="utf-8"))
    assert [r.get("estimate_seconds") for r in repriced["rows"]] == [
        r.get("estimate_seconds") for r in data["rows"]
    ]
    assert [a["estimate_seconds"] for a in repriced["all_threads"]] == [
        a["estimate_seconds"] for a in data["all_threads"]
    ]


def test_correlated_erasure_baseline_quick(tmp_path):
    out = tmp_path / "ceb"
    script = str(ROOT / "benchmarks" / "correlated_erasure_baseline.py")
    proc = subprocess.run(
        [sys.executable, script, "--quick", "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
        env=_src_env(),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads((out / "correlated_erasure_baseline.json").read_text(encoding="utf-8"))
    md = (out / "correlated_erasure_baseline.md").read_text(encoding="utf-8")
    assert md.startswith("# Correlated-erasure baseline: the brickwork ring against the literature")
    for heading in ("## Summary", "## 2. Scaling with n", "## 5. Φ_{C,J}(ε) and ε_δ(J)"):
        assert heading in md
    cfg = data["settings"]
    assert data["quick"] is True
    ring = [c for c in data["cases"] if c["family"] == "ring"]
    assert [(c["n"], c["p"]) for c in ring] == [
        (n, p) for n in cfg["sizes"] + cfg["scaling_sizes"] for p in cfg["rates"]
    ]
    for c in ring:
        assert c["correlated"] == (c["n"] in cfg["sizes"])
        if c["q_plus"] == 0:
            continue
        # l_x >= d_cont for every site, and the J -> inf (interval) curve grows with m
        assert c["d_cont"]["mean"] <= c["gh_l_code"]["mean"] + 1e-12
        limit = c["curves"]["J = inf"]["mean"]
        assert limit == sorted(limit)
    for row in data["interval_probability"]:
        probs = [row[f"J = {j}"] for j in ("0.5", "1", "2", "4", "8")]
        assert probs == sorted(probs) and probs[0] > 0 and probs[-1] <= 1
    assert any(f["lf_star"] for f in data["fits"])
    # re-rendering the saved results gives the same report
    again = tmp_path / "again"
    proc = subprocess.run(
        [sys.executable, script, "--from-json", str(out / "correlated_erasure_baseline.json"),
         "--out", str(again)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=ROOT,
        env=_src_env(),
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (again / "correlated_erasure_baseline.md").read_text(encoding="utf-8") == md
