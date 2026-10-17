"""What a depth scan of Watts–Strogatz sweeps to T_max = 10^6 costs: the owner's workload.

The workload (``docs/SWEEPS.md``; v1.4 sections 5.2, 5.5, 5.7, 5.8): fixed (n, beta, p), the
sweep's protocol (purification picture, ``n // 2`` random-edge gates per layer, every layer a
purification test, k and its exact first passages read after every layer) and probes on the
61-point geometric grid of T_max (v1.4 section 9). Every logical processor, one trajectory per
thread at a time (a round), default settings. Seeds are fixed; nothing here depends on a
result's values except the counts of codes.

1. **Engine.** Layers per second of the machine (every thread busy, a round) and per thread
   (the same divided by the threads), at T = 10^4, 10^5 and 10^6 where affordable, so that the
   rate is seen not to depend on T; and of one thread alone (one trajectory). Seconds per
   trajectory to 10^6 are measured where T = 10^6 ran, and otherwise extrapolated from the
   deepest run (10^6 over its per-thread rate), labeled so.
2. **stop_when_purified.** A stopped trajectory costs min(tau_0, T_max) layers, tau_0 from its
   exact first passages; modeled at the measured rate, and checked against rounds run with the
   stop. Censored at a depth below 10^6, a trajectory is counted to 10^6 (an upper bound).
3. **Probes.** Thread-seconds per evaluation with every thread busy, on the codes at the last
   E layers up to 4n at p = 0.05: the run with the probe minus the run without (minimum over
   repeats), over the evaluations that held a code (k > 0; ``tableau_snapshot``: over every
   evaluation, it does not depend on k). ``correlated_erasure``: every size 1..n-1, J = 0.5 and
   2, 100 samples; the exact ring sampler (geometry ``cycle``) at beta = 0, the swap chain on
   the interaction graph at beta = 0.1. At n = 512 it runs every 32nd size and is scaled to
   every size (extrapolated, labeled); n = 256 runs both, which checks the scaling. Also the
   compressed bytes (zlib level 1, the sweep's default) per evaluation of each probe's arrays.
4. **Disk.** ``sweep.run`` of one condition (beta = 0.1, p = 0.05, without the stop, so every
   checkpoint is evaluated) with ``code_dimension``, ``contiguous_distance``,
   ``erasure_first_failure`` and ``tableau_snapshot`` on the 61-point grid of T = 10^6 (n = 512:
   10^5; a result's size does not depend on T), default compression: bytes per trajectory of
   each stored array of its parts.
5. **Campaign.** Trajectories per hour of the machine for (a) k, first passages and d_cyc at
   the 61 checkpoints, (b) (a) + D(0) (``erasure_first_failure``, 100 orderings), (c) (b) + D(J)
   (``correlated_erasure`` as above), with ``stop_when_purified`` (EXPECTATION_VALUES.md section
   9): thread-seconds per trajectory = the engine to min(tau_0, T_max) + the expected number of
   checkpoints with a code, sum over the grid of P(tau_0 > t) from the first passages, times each
   probe's cost per evaluation measured at p = 0.05 (taken for every p). The sweep's saving of
   parts runs on other threads and is not counted.

Usage, from the repository root, on an otherwise idle machine::

    python benchmarks/depth_scan.py                       # about 18 minutes, all cores
    python benchmarks/depth_scan.py --quick --out <dir>   # smoke test, under a minute
    python benchmarks/depth_scan.py --from-json benchmarks/results/depth_scan.json

Writes ``depth_scan.json`` and ``depth_scan.md`` into ``--out`` (default
``benchmarks/results/``); ``--from-json`` measures nothing and renders the markdown again.
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numba
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from thread_scaling import cpu_info  # noqa: E402

import sparsegf2 as sg  # noqa: E402
from sparsegf2.study import identity, sweep  # noqa: E402
from sparsegf2.study.sweep import SweepSpec, geometric_checkpoints  # noqa: E402

GRID = 61
CIRCUITS = 4
MASTER_GRAPH, MASTER_CIRCUIT = 3, 5
REPEATS = 3
FULL = {
    "t_max": 10**6,
    "sizes": (32, 64, 128, 256, 512),
    "betas": ("0", "0.1"),
    "rates": ("0.02", "0.05", "0.1", "0.15", "0.2", "0.3", "0.5"),
    # (n, depth) -> the (beta, p) run at that depth ("all": every one)
    "engine": {
        **{(n, t): "all" for n in (32, 64, 128) for t in (10**4, 10**5, 10**6)},
        (256, 10**4): "all",
        (256, 10**5): "all",
        (256, 10**6): [("0.1", p) for p in ("0.02", "0.05", "0.1", "0.15", "0.2", "0.3", "0.5")]
        + [("0", p) for p in ("0.02", "0.1", "0.3")],
        (512, 10**4): "all",
        (512, 10**5): [("0.1", p) for p in ("0.02", "0.05", "0.1", "0.15", "0.2", "0.3", "0.5")]
        + [("0", p) for p in ("0.02", "0.1", "0.3")],
    },
    "single_thread_layers": {32: 10**4, 64: 10**4, 128: 10**4, 256: 10**4, 512: 2000},
    "probe_rate": "0.05",
    "cheap_evaluations": {32: 64, 64: 64, 128: 64, 256: 64, 512: 16},
    "snapshot_evaluations": 8,
    "correlated_evaluations": {32: 4, 64: 2, 128: 1, 256: 1, 512: 1},
    "correlated_stride": {512: 32},  # every 32nd size, scaled to all of them
    "correlated_check": {256: 16},  # also every 16th size, to check that scaling
    "disk_layers": {32: 10**6, 64: 10**6, 128: 10**6, 256: 10**6, 512: 10**5},
    "stop_checks": 2,
}
QUICK = {
    "t_max": 1000,
    "sizes": (8, 12),
    "betas": ("0", "0.1"),
    "rates": ("0.1", "0.3"),
    "engine": {(n, t): "all" for n in (8, 12) for t in (100, 1000)},
    "single_thread_layers": {8: 1000, 12: 1000},
    "probe_rate": "0.1",
    "cheap_evaluations": {8: 8, 12: 8},
    "snapshot_evaluations": 4,
    "correlated_evaluations": {8: 2, 12: 1},
    "correlated_stride": {12: 4},
    "correlated_check": {8: 2},
    "disk_layers": {8: 1000, 12: 1000},
    "stop_checks": 1,
}
PROBES = ("code_dimension", "contiguous_distance", "erasure_first_failure", "tableau_snapshot",
          "correlated_erasure")  # fmt: skip


@functools.cache
def collection(n: int, beta: str, count: int) -> tuple[np.ndarray, ...]:
    return tuple(identity.build_graph(n, beta, g, MASTER_GRAPH).edges for g in range(count))


def condition(n, beta, p, layers, probes, *, stop=False, trajectories=None) -> dict:
    count = trajectories or numba.get_num_threads()
    return {
        "sizes": [n],
        "betas": [beta],
        "rates": [p],
        "graphs": -(-count // CIRCUITS),
        "circuits_per_graph": CIRCUITS,
        "stride": CIRCUITS,
        "layers": {str(n): layers},
        "checkpoints": {"count": GRID},
        "probes": probes,
        "stop_when_purified": stop,
        "master_graph_seed": MASTER_GRAPH,
        "master_circuit_seed": MASTER_CIRCUIT,
    }


def run(n, beta, p, layers, probes, *, stop=False, trajectories=None, num_threads=None):
    """``trajectories`` (default: one per thread) of the condition through the engine, as the
    sweep runs them; returns (result, wall seconds)."""
    count = trajectories or numba.get_num_threads()
    spec = SweepSpec.from_json(condition(n, beta, p, layers, probes, stop=stop, trajectories=count))
    mc = spec.monitored_circuit(n, beta, p, list(collection(n, beta, spec.graphs[n])))
    index, seed = spec.indices(n)[:count], list(spec.run_seed(n, beta))
    start = time.perf_counter()
    result = mc.run(indices=index, seed=seed, num_threads=num_threads)
    return result, time.perf_counter() - start


def sig(x: float) -> float:
    return float(f"{x:.4g}")


# ---------------------------------------------------------------------------
# 1. the engine
# ---------------------------------------------------------------------------


def engine_row(n, beta, p, layers, t_max) -> dict:
    res, wall = run(n, beta, p, layers, [{"kind": "code_dimension"}])
    threads, count = numba.get_num_threads(), res.num_trajectories
    tau0 = res.first_passage[:, 0]
    observed = tau0 >= 0
    grid = [t for t in geometric_checkpoints(t_max, GRID) if t <= layers]
    return {
        "n": n,
        "beta": beta,
        "p": p,
        "layers": layers,
        "trajectories": count,
        "seconds": round(wall, 3),
        "machine_layers_per_second": sig(count * layers / wall),
        "thread_layers_per_second": sig(count * layers / wall / min(count, threads)),
        "purified": round(float(observed.mean()), 4),
        "tau0_median": float(np.median(tau0[observed])) if observed.any() else None,
        "tau0": tau0.tolist(),
        # P(tau_0 > t) at the grid points of T_max up to this depth
        "alive": [round(float(np.mean(~observed | (tau0 > t))), 4) for t in grid],
    }


def single_thread_row(n, p, layers) -> dict:
    res, wall = run(n, "0.1", p, layers, [{"kind": "code_dimension"}], trajectories=1,
                    num_threads=1)  # fmt: skip
    return {"n": n, "beta": "0.1", "p": p, "layers": layers, "seconds": round(wall, 3),
            "layers_per_second": sig(layers / wall)}  # fmt: skip


# ---------------------------------------------------------------------------
# 3. probes
# ---------------------------------------------------------------------------


def probe_json(kind: str, n: int, beta: str, sizes: str | None) -> dict:
    if kind == "erasure_first_failure":
        return {"kind": kind, "samples": 100}
    if kind == "correlated_erasure":
        return {"kind": kind, "couplings": [0.5, 2], "sizes": sizes, "samples": 100,
                "geometry": "cycle" if beta == "0" else "interaction"}  # fmt: skip
    return {"kind": kind}


def stored_bytes(result, names: list[str], scratch: Path) -> int:
    """Compressed bytes (zlib level 1) of the arrays ``names`` when ``result`` is saved."""
    path = scratch / "probe.npz"
    result.save(path, compression="fast")
    with zipfile.ZipFile(path) as z:
        size = sum(i.compress_size for i in z.infolist() if i.filename[:-4] in names)
    path.unlink()
    return size


def probe_row(n, beta, kind, plan, scratch, *, stride=None) -> dict:
    p, last = plan["probe_rate"], 4 * n
    if kind == "tableau_snapshot":
        evaluations = plan["snapshot_evaluations"]
    elif kind == "correlated_erasure":
        evaluations = plan["correlated_evaluations"][n]
    elif kind == "code_dimension":
        evaluations = min(last, 512)
    else:
        evaluations = plan["cheap_evaluations"][n]
    window = list(range(last - evaluations + 1, last + 1))
    k = {"kind": "code_dimension", "name": "k", "checkpoints": window}
    base = [] if kind == "code_dimension" else [k]
    sizes = None if stride is None else f"1:{n}:{stride}"
    probe = {**probe_json(kind, n, beta, sizes), "checkpoints": window}
    repeats = 1 if kind == "correlated_erasure" else REPEATS
    walls = {"without": [], "with": []}
    for _ in range(repeats):
        res0, wall0 = run(n, beta, p, last, base)
        res1, wall1 = run(n, beta, p, last, base + [probe])
        walls["without"].append(wall0)
        walls["with"].append(wall1)
    k_values = res1.values["code_dimension" if kind == "code_dimension" else "k"]
    with_code = float(np.mean(np.sum(k_values > 0, axis=1)))  # per trajectory
    per = evaluations if kind in ("tableau_snapshot", "code_dimension") else with_code
    extra = min(walls["with"]) - min(walls["without"])
    sized = n - 1 if stride is None else len(range(1, n, stride))
    names = [f"value::{kind}", f"snapshot::{kind}", f"snapshot_counts::{kind}"]
    return {
        "n": n,
        "beta": beta,
        "p": p,
        "probe": kind,
        "sizes": sized if kind == "correlated_erasure" else None,
        "stride": stride,
        "evaluations": evaluations,
        "evaluations_with_code": round(with_code, 2),
        "trajectories": res1.num_trajectories,
        "seconds_without": round(min(walls["without"]), 4),
        "seconds_with": round(min(walls["with"]), 4),
        "seconds_per_evaluation": sig(extra / per) if per > 0 else None,
        "bytes_per_evaluation": round(
            stored_bytes(res1, names, scratch) / (res1.num_trajectories * evaluations), 1
        ),
        "sampler": res1.spec.correlated_sampler(kind)["method"]
        if kind == "correlated_erasure"
        else None,
    }


# ---------------------------------------------------------------------------
# 4. disk
# ---------------------------------------------------------------------------


def disk_row(n, layers, scratch) -> dict:
    probes = [{"kind": "code_dimension"}, {"kind": "contiguous_distance"},
              {"kind": "erasure_first_failure", "samples": 100}, {"kind": "tableau_snapshot"}]  # fmt: skip
    directory = scratch / f"disk_{n}"
    start = time.perf_counter()
    out = sweep.run(condition(n, "0.1", "0.05", layers, probes), directory)
    wall = time.perf_counter() - start
    folder = directory / out["conditions"][0]["path"]
    parts = sorted(folder.glob("result-*.npz"))
    members: dict[str, int] = {}
    for path in parts:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                members[info.filename[:-4]] = (
                    members.get(info.filename[:-4], 0) + info.compress_size
                )
    total = sum(path.stat().st_size for path in parts)
    count = out["saved"]
    shutil.rmtree(directory)
    # what every part held before the global checkpoint rule was stored compactly: the
    # T + 1 layers of "every_layer" as int64, compressed the same way (zlib level 1)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        with z.open("checkpoint_layers.npy", "w") as fh:
            np.lib.format.write_array(fh, np.arange(layers + 1, dtype=np.int64))
        every_layer = z.infolist()[0].compress_size
    return {
        "n": n,
        "layers": layers,
        "trajectories": count,
        "parts": len(parts),
        "seconds": round(wall, 2),
        "bytes_per_trajectory": round(total / count),
        "every_layer_array_bytes_per_part": every_layer,
        "members_per_trajectory": {k: round(v / count, 1) for k, v in sorted(members.items())},
    }


SNAPSHOT_MEMBERS = ("snapshot::tableau_snapshot", "snapshot_counts::tableau_snapshot",
                    "value::tableau_snapshot")  # fmt: skip


def disk_configs(row: dict, correlated_bytes: float | None) -> dict:
    """Bytes per trajectory of each configuration, from the members of the measured parts."""
    m = row["members_per_trajectory"]
    zip_overhead = row["bytes_per_trajectory"] - sum(m.values())  # headers and directory
    a = zip_overhead + sum(
        v for k, v in m.items() if k not in SNAPSHOT_MEMBERS and k != "value::erasure_first_failure"
    )
    b = a + m.get("value::erasure_first_failure", 0.0)
    return {
        "a": round(a),
        "b": round(b),
        "c": None if correlated_bytes is None else round(b + GRID * correlated_bytes),
        "a_snapshots": round(a + sum(m.get(k, 0.0) for k in SNAPSHOT_MEMBERS)),
    }


# ---------------------------------------------------------------------------
# 2 and 5. the stop and the campaign (derived)
# ---------------------------------------------------------------------------


def deepest(results: dict) -> dict:
    """The deepest engine row of each (n, beta, p)."""
    out: dict = {}
    for r in results["engine"]:
        key = (r["n"], r["beta"], r["p"])
        if key not in out or r["layers"] > out[key]["layers"]:
            out[key] = r
    return out


def stop_model(r: dict, t_max: int) -> dict:
    """Layers per trajectory with the stop (min(tau_0, T_max); censored at a depth below T_max:
    T_max) and the expected checkpoints with a code (sum over the grid of P(tau_0 > t); beyond
    the run's depth, P(tau_0 > depth))."""
    tau0 = np.array(r["tau0"])
    layers = np.where(tau0 >= 0, np.minimum(tau0, t_max), t_max)
    grid = geometric_checkpoints(t_max, GRID)
    alive = r["alive"] + [r["alive"][-1]] * (len(grid) - len(r["alive"]))
    return {
        "layers": float(layers.mean()),
        "seconds": float(layers.mean()) / r["thread_layers_per_second"],
        "with_code": float(sum(alive)),
        "bounded": r["layers"] < t_max and bool(np.any(tau0 < 0)),
    }


def probe_cost(results: dict, n: int, beta: str, kind: str) -> tuple[float | None, str]:
    """A probe's thread-seconds per evaluation with a code, and whether it was extrapolated."""
    rows = [r for r in results["probes"] if (r["n"], r["beta"], r["probe"]) == (n, beta, kind)]
    full = [r for r in rows if r["stride"] is None]
    row = full[0] if full else (rows[0] if rows else None)
    if row is None or row["seconds_per_evaluation"] is None:
        return None, "not measured"
    if row["stride"] is None:
        return max(row["seconds_per_evaluation"], 0.0), "measured"
    return row["seconds_per_evaluation"] * (n - 1) / row["sizes"], "extrapolated"


def campaign(results: dict) -> list[dict]:
    t_max, threads = results["plan"]["t_max"], results["environment"]["numba_num_threads"]
    rows = []
    for (n, beta, p), r in sorted(deepest(results).items()):
        model = stop_model(r, t_max)
        costs = {k: probe_cost(results, n, beta, k) for k in PROBES}
        engine = model["seconds"]
        per_eval = {
            "a": (costs["code_dimension"][0] or 0.0) + (costs["contiguous_distance"][0] or 0.0),
            "b": costs["erasure_first_failure"][0],
            "c": costs["correlated_erasure"][0],
        }
        seconds = {"a": engine + model["with_code"] * per_eval["a"]}
        seconds["b"] = (
            None if per_eval["b"] is None else seconds["a"] + model["with_code"] * per_eval["b"]
        )
        seconds["c"] = (
            None if per_eval["c"] is None or seconds["b"] is None
            else seconds["b"] + model["with_code"] * per_eval["c"]
        )  # fmt: skip
        rows.append({
            "n": n, "beta": beta, "p": p,
            "engine_layers": r["layers"],
            "stop_layers": round(model["layers"]),
            "stop_seconds": sig(engine),
            "no_stop_seconds": sig(t_max / r["thread_layers_per_second"]),
            "checkpoints_with_code": round(model["with_code"], 2),
            "upper_bound": model["bounded"],
            "seconds": {k: None if v is None else sig(v) for k, v in seconds.items()},
            "per_hour": {k: None if v is None else sig(3600 * threads / v) for k, v in seconds.items()},
            "correlated_basis": costs["correlated_erasure"][1],
        })  # fmt: skip
    return rows


def derive(results: dict) -> dict:
    """The derived tables from the measured rows: the disk of each configuration (D(J)'s bytes
    per evaluation at beta = 0.1, every size, or scaled from a subset) and the campaign."""
    correlated: dict = {}
    for r in results["probes"]:
        if r["probe"] == "correlated_erasure" and r["beta"] == "0.1":
            if r["stride"] is None:
                correlated[r["n"]] = (r["bytes_per_evaluation"], False)
            elif r["n"] not in correlated:
                correlated[r["n"]] = (r["bytes_per_evaluation"] * (r["n"] - 1) / r["sizes"], True)
    for d in results["disk"]:
        size, scaled = correlated.get(d["n"], (None, False))
        d["configs"] = {**disk_configs(d, size), "c_extrapolated": scaled}
    results["campaign"] = campaign(results)
    return results


def stop_checks(results: dict, count: int) -> list[dict]:
    """Rounds run with the stop, against the model, in the conditions whose trajectories all
    purified at the largest median tau_0 (where the stop matters and the timing is long)."""
    t_max, threads = results["plan"]["t_max"], numba.get_num_threads()
    rows = [r for r in deepest(results).values() if r["layers"] == t_max and r["purified"] == 1]
    rows.sort(key=lambda r: -r["tau0_median"])
    out = []
    for r in rows[:count]:
        rounds = 4
        res, wall = run(r["n"], r["beta"], r["p"], t_max, [{"kind": "code_dimension"}],
                        stop=True, trajectories=rounds * threads)  # fmt: skip
        layers = res.final_layer.astype(float)
        model = layers.sum() / r["thread_layers_per_second"] / threads
        out.append({
            "n": r["n"], "beta": r["beta"], "p": r["p"], "trajectories": res.num_trajectories,
            "mean_final_layer": round(float(layers.mean()), 1),
            "seconds": round(wall, 3), "model_seconds": round(float(model), 3),
        })  # fmt: skip
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def fmt(x, digits: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and x <= 0:
        return "~0"  # below the timing resolution
    if isinstance(x, float) and (abs(x) >= 1e5 or (0 < abs(x) < 1e-3)):
        return f"{x:.{digits - 1}e}"
    return f"{x:,.{digits}g}" if isinstance(x, float) else f"{x:,}"


def render_markdown(results: dict) -> str:
    env, plan = results["environment"], results["plan"]
    t_max, threads = plan["t_max"], env["numba_num_threads"]
    exp = len(str(t_max)) - 1
    lines = [
        "# Depth scan to T_max",
        "",
        f"CPU: {env['cpu_model']} ({env['physical_cores']} physical cores, "
        f"{env['logical_processors']} logical processors), {env['platform']}. Python "
        f"{env['python']}, NumPy {env['numpy']}, Numba {env['numba']} (`NUMBA_NUM_THREADS` = "
        f"{threads}, threading layer `{env['threading_layer']}`), sparsegf2 {env['sparsegf2']}.",
        "",
        f"The sweep's protocol at fixed (n, beta, p): purification picture, n // 2 random-edge "
        f"gates per layer, every layer a purification test, k and its exact first passages "
        f"read after every layer; T_max = 10^{exp}, probes on its {GRID}-point geometric grid. "
        f"Every thread busy, one trajectory per thread (a round of {threads}); "
        "`benchmarks/depth_scan.py` describes each measurement.",
        "",
        "## 1. Engine",
        "",
        "Millions of layers per second of the machine (every thread busy), by the depth of the "
        "run, and per thread at the deepest run; one thread alone (one trajectory). Seconds per "
        f"trajectory to 10^{exp} (thread-seconds, every thread busy): measured where the run "
        "reached it, otherwise extrapolated from the deepest run (marked `*`).",
        "",
    ]
    depths = sorted({r["layers"] for r in results["engine"]})
    single = {(r["n"], r["p"]): r for r in results["single_thread"]}
    deep = deepest(results)
    head = " | ".join(f"machine M/s, T = {d:,}" for d in depths)
    lines += [
        f"| n | beta | p | {head} | thread layers/s | one thread alone, layers/s | "
        f"s per trajectory to {t_max:,} | purified by the deepest run |",
        "|---:|---:|---:|" + "---:|" * len(depths) + "---:|---:|---:|---:|",
    ]
    by_depth = {(r["n"], r["beta"], r["p"], r["layers"]): r for r in results["engine"]}
    for (n, beta, p), r in sorted(deep.items()):
        cells = " | ".join(
            fmt(by_depth[(n, beta, p, d)]["machine_layers_per_second"] / 1e6)
            if (n, beta, p, d) in by_depth
            else "-"
            for d in depths
        )
        alone = single.get((n, p)) if beta == "0.1" else None
        secs = t_max / r["thread_layers_per_second"]
        mark = "" if r["layers"] == t_max else "*"
        lines.append(
            f"| {n} | {beta} | {p} | {cells} | {fmt(r['thread_layers_per_second'])} | "
            f"{fmt(alone['layers_per_second']) if alone else '-'} | {fmt(secs)}{mark} | "
            f"{r['purified']:.2f} by {r['layers']:,} |"
        )
    lines += [
        "",
        "## 2. stop_when_purified",
        "",
        "With the stop a trajectory runs min(tau_0, T_max) layers; per (n, beta, p) the mean of "
        "that over the trajectories (tau_0 exact, from the first passages; censored below "
        "T_max: counted to T_max, an upper bound marked `<=`), its thread-seconds at the "
        "measured rate, against the run without the stop. Checked against rounds run with the "
        "stop (model: the layers run over the per-thread rate and the threads):",
        "",
        "| n | beta | p | trajectories | mean final layer | seconds | model seconds | "
        "measured / model |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in results["stop_checks"]:
        lines.append(
            f"| {c['n']} | {c['beta']} | {c['p']} | {c['trajectories']} | "
            f"{c['mean_final_layer']:,} | {c['seconds']} | {c['model_seconds']} | "
            f"{c['seconds'] / c['model_seconds']:.2f} |"
        )
    lines += [
        "",
        "The model assumes the threads stay busy; a call ends with its slowest thread, and "
        "Numba gives each thread an equal block of trajectories, so where tau_0 spreads over "
        "the whole horizon the threads that finish early wait (the ratio above). The campaign "
        "below uses the model: divide its rates by up to that ratio where tau_0 is comparable "
        "to T_max.",
    ]
    lines += [
        "",
        f"| n | beta | p | mean layers with the stop | s per trajectory with the stop | "
        f"without (to {t_max:,}) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for c in results["campaign"]:
        bound = "<= " if c["upper_bound"] else ""
        lines.append(
            f"| {c['n']} | {c['beta']} | {c['p']} | {bound}{c['stop_layers']:,} | "
            f"{bound}{fmt(c['stop_seconds'])} | {fmt(c['no_stop_seconds'])} |"
        )
    lines += [
        "",
        "## 3. Probes per evaluation",
        "",
        f"Thread-seconds per evaluation with every thread busy, on codes at p = "
        f"{plan['probe_rate']} at the last layers up to 4n (over the evaluations with k > 0; "
        "`tableau_snapshot` and `code_dimension` over every evaluation); compressed bytes per "
        "evaluation (zlib level 1). `correlated_erasure`: J = 0.5 and 2, 100 samples, every "
        "size, or every `stride`-th size (then scaled to every size in the campaign).",
        "",
        "| n | beta | probe | sizes | evaluations (with a code) | s per evaluation | "
        "bytes per evaluation | sampler |",
        "|---:|---:|:---|---:|---:|---:|---:|:---|",
    ]
    for r in results["probes"]:
        sizes = (
            "-"
            if r["sizes"] is None
            else f"{r['sizes']}" + (f" (1 in {r['stride']})" if r["stride"] else "")
        )
        lines.append(
            f"| {r['n']} | {r['beta']} | {r['probe']} | {sizes} | {r['evaluations']} "
            f"({r['evaluations_with_code']}) | {fmt(r['seconds_per_evaluation'])} | "
            f"{fmt(r['bytes_per_evaluation'])} | {r['sampler'] or '-'} |"
        )
    lines += [
        "",
        "## 4. Disk",
        "",
        f"`sweep.run` of one condition (beta = 0.1, p = 0.05, without the stop, every checkpoint "
        f"evaluated), {GRID}-point grid, default compression; bytes per trajectory of the parts "
        "(a: k, first passages, d_cyc and the run's records; b: + D(0); c: + D(J) at every "
        "checkpoint, from its bytes per evaluation above, `*` scaled from a subset of sizes; with "
        "tableau snapshots: a + the joint tableau at every checkpoint). The last column is what "
        "every part held before the checkpoint rule was stored compactly, per trajectory:",
        "",
        "| n | T | trajectories (parts) | seconds | (a) | (b) | (c) | (a) + snapshots | "
        "first passages | snapshots | every-layer array per trajectory, no longer stored |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in results["disk"]:
        c = d["configs"]
        m = d["members_per_trajectory"]
        lines.append(
            f"| {d['n']} | {d['layers']:,} | {d['trajectories']} ({d['parts']}) | "
            f"{d['seconds']} | {fmt(c['a'])} | {fmt(c['b'])} | {fmt(c['c'])}"
            f"{'*' if c['c_extrapolated'] else ''} | "
            f"{fmt(c['a_snapshots'])} | {fmt(m.get('first_passage', 0.0))} | "
            f"{fmt(m.get('snapshot::tableau_snapshot', 0.0))} | "
            f"{fmt(d['every_layer_array_bytes_per_part'] * d['parts'] / d['trajectories'])} |"
        )
    lines += [
        "",
        "## 5. Campaign",
        "",
        f"Trajectories per hour of the machine ({threads} threads) with `stop_when_purified`, "
        f"probes on the {GRID}-point grid of T_max = {t_max:,} "
        f"({len(geometric_checkpoints(t_max, GRID))} distinct layers, 0 included): (a) k, "
        "first passages and d_cyc; (b) + D(0); (c) + D(J). Thread-seconds per trajectory: the "
        "engine to min(tau_0, T_max) (section 2) plus, per probe, its cost per evaluation "
        "(section 3, measured at p = "
        f"{plan['probe_rate']} and used at every p) times `checkpoints with a code`, the "
        "expected number of grid layers with k > 0 (`<=`: an upper bound, the run censored "
        "below T_max). `(c)` marked `*` uses a D(J) cost scaled to every size from a subset "
        "(section 3). Rows of a few milliseconds per trajectory are bounds the sweep does not "
        "reach: its calls, chunks and saved parts cost more than such trajectories.",
        "",
        "| n | beta | p | checkpoints with a code | s per trajectory, engine | (a) per hour | "
        "(b) per hour | (c) per hour |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in results["campaign"]:
        bound = "<= " if c["upper_bound"] else ""
        star = "*" if c["correlated_basis"] == "extrapolated" else ""
        ph = c["per_hour"]
        lines.append(
            f"| {c['n']} | {c['beta']} | {c['p']} | {bound}{c['checkpoints_with_code']} | "
            f"{bound}{fmt(c['stop_seconds'])} | {fmt(ph['a'])} | {fmt(ph['b'])} | "
            f"{fmt(ph['c'])}{star} |"
        )
    lines += ["", f"Total wall time {results['total_seconds']:.0f} s.", ""]
    return "\n".join(lines)


def write(results: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("depth_scan.json", json.dumps(results, indent=1) + "\n"),
        ("depth_scan.md", render_markdown(results)),
    ):
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def measure(plan: dict, scratch: Path, log) -> dict:
    t_max = plan["t_max"]
    results: dict = {"engine": [], "single_thread": [], "probes": [], "disk": []}
    for (n, layers), which in plan["engine"].items():
        pairs = [(b, p) for b in plan["betas"] for p in plan["rates"]] if which == "all" else which
        for beta, p in pairs:
            results["engine"].append(engine_row(n, beta, p, layers, t_max))
            log(results["engine"][-1] | {"tau0": "...", "alive": "..."})
    for n, layers in plan["single_thread_layers"].items():
        for p in plan["rates"]:
            results["single_thread"].append(single_thread_row(n, p, layers))
            log(results["single_thread"][-1])
    for n in plan["sizes"]:
        for beta in plan["betas"]:
            for kind in PROBES:
                strides = (
                    [plan["correlated_stride"].get(n)] if kind == "correlated_erasure" else [None]
                )
                if kind == "correlated_erasure" and n in plan["correlated_check"]:
                    strides.append(plan["correlated_check"][n])
                for stride in strides:
                    results["probes"].append(probe_row(n, beta, kind, plan, scratch, stride=stride))
                    log(results["probes"][-1])
    for n, layers in plan["disk_layers"].items():
        results["disk"].append(disk_row(n, layers, scratch))
        log(results["disk"][-1])
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(HERE / "results"), help="output directory")
    ap.add_argument("--quick", action="store_true", help="tiny sizes and depths, a smoke test")
    ap.add_argument("--from-json", default=None, help="render a saved results file again")
    args = ap.parse_args()
    out = Path(args.out)
    if args.from_json:  # the derived tables are computed again from the measured rows
        write(derive(json.loads(Path(args.from_json).read_text("utf-8"))), out)
        return
    plan = QUICK if args.quick else FULL
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="depth_scan_") as tmp:
        scratch = Path(tmp)
        for kind in PROBES:  # compile every kernel first
            probe_row(8, "0.1", kind, {**QUICK, "probe_rate": "0.1"}, scratch, stride=None)
        env = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **cpu_info(),
            "logical_processors": os.cpu_count(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "numba_num_threads": numba.config.NUMBA_NUM_THREADS,
            "threading_layer": numba.threading_layer(),
            "sparsegf2": sg.__version__,
        }
        print(f"{env['cpu_model']}; {env['numba_num_threads']} threads", flush=True)
        results = measure(plan, scratch, lambda row: print(row, flush=True))
        results = {
            "environment": env,
            "quick": args.quick,
            "plan": {"t_max": plan["t_max"], "probe_rate": plan["probe_rate"]},
            **results,
        }
        results["stop_checks"] = stop_checks(results, plan["stop_checks"])
        print(results["stop_checks"], flush=True)
    results["total_seconds"] = round(time.perf_counter() - start, 1)
    write(derive(results), out)


if __name__ == "__main__":
    main()
