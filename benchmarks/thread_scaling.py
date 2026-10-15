"""Thread scaling of ensembles and shot sampling, and a check that results do not depend on it.

Four workloads, each timed at 1, 2, 4, 8, 16 and ``NUMBA_NUM_THREADS`` threads:

* **W1** monitored circuits, Watts-Strogatz n = 64 (k = 4, beta = 0.1), purification picture,
  p = 0.1, 256 layers, ``random_edges``; probes ``code_dimension`` every 32 layers,
  ``erasure_first_failure(samples=512)`` and ``uncorrectable_region_counts(max_size=2)`` at the end;
* **W2** the same at n = 256;
* **W3** the purification-time workload: cycle n = 128, single reference, p = 0.1,
  ``8 n`` layers, ``stop_when_purified``, ``code_dimension`` every layer;
* **W4** ``sparsegf2.sample`` of a noisy repetition-code memory circuit.

Wall time is the median of ``--repeats`` runs after a warm-up (which also compiles). Every thread
count must give identical results (all arrays equal, samples equal for a fixed seed); the script
aborts if not. Usage, from the repository root, on an idle machine::

    python benchmarks/thread_scaling.py
    python benchmarks/thread_scaling.py --quick --out /tmp/ts     # smoke test

Writes ``thread_scaling.json`` and ``thread_scaling.md`` into ``--out``
(default ``benchmarks/results/``).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numba
import numpy as np

import sparsegf2 as sg
from sparsegf2 import Circuit, graphs, mipt

HERE = Path(__file__).resolve().parent
SEED = 2024


def cpu_info() -> dict:
    """CPU model name and physical core count, best effort per platform."""
    info: dict = {"cpu_model": None, "physical_cores": None}
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if sys.platform == "win32":
        script = (
            "Get-CimInstance Win32_Processor | ForEach-Object "
            "{ $_.Name.Trim() + '|' + $_.NumberOfCores }"
        )
        try:
            lines = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            ).stdout.splitlines()
            rows = [ln.rsplit("|", 1) for ln in lines if "|" in ln]
            info["cpu_model"] = rows[0][0]
            info["physical_cores"] = sum(int(r[1]) for r in rows)
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
    elif sys.platform == "darwin":
        with contextlib.suppress(OSError, subprocess.SubprocessError, ValueError):
            info["physical_cores"] = int(
                subprocess.run(
                    ["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True, check=True
                ).stdout
            )
    if info["cpu_model"] is None:
        info["cpu_model"] = platform.processor() or platform.machine()
    return info


def digest(*arrays) -> str:
    """Hash of the bytes of arrays (NaN bit patterns included): equal digest, equal arrays."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes())
    return h.hexdigest()


def ensemble_digest(res: mipt.EnsembleResult) -> str:
    arrays = [res.values[name] for name in sorted(res.values)]
    arrays += [res.purification_layer, res.final_layer, res.num_measurements]
    return digest(*arrays)


def erasure_spec(n: int) -> mipt.MonitoredCircuit:
    return mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.watts_strogatz(n, 4, 0.1, seed=1),
        layers=256,
        measurement_rate=0.1,
        schedule="random_edges",
        reference="purification",
        checkpoints=32,
        probes=[
            mipt.code_dimension(),
            mipt.erasure_first_failure(samples=512, checkpoints="final"),
            mipt.uncorrectable_region_counts(max_size=2, checkpoints="final"),
        ],
    )


def purification_spec(n: int) -> mipt.MonitoredCircuit:
    return mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        layers=8 * n,
        measurement_rate=0.1,
        schedule="random_edges",
        reference="single",
        reference_partner="random",
        checkpoints="every_layer",
        stop_when_purified=True,
        probes=[mipt.code_dimension()],
    )


def repetition_memory(distance: int, rounds: int, p: float) -> Circuit:
    """Bit-flip repetition-code memory: data qubits 0..d-1, checks d..2d-2, noise before each round."""
    d = distance
    data = " ".join(str(i) for i in range(d))
    cx1 = " ".join(f"{i} {d + i}" for i in range(d - 1))
    cx2 = " ".join(f"{i + 1} {d + i}" for i in range(d - 1))
    checks = " ".join(str(d + i) for i in range(d - 1))
    rnd = f"X_ERROR({p}) {data}\nCX {cx1}\nCX {cx2}\nMR {checks}\n"
    dets = "\n".join(f"DETECTOR rec[-{d - 1 - i}] rec[-{2 * (d - 1) - i}]" for i in range(d - 1))
    first = rnd + "\n".join(f"DETECTOR rec[-{d - 1 - i}]" for i in range(d - 1)) + "\n"
    body = rnd + dets + "\n"
    final = f"M {data}\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    return Circuit(first + f"REPEAT {rounds - 1} {{\n{body}}}\n" + final)


def make_workloads(quick: bool) -> list[dict]:
    """Each workload: name, description, unit, size and ``run(threads, units) -> digest``."""
    # units: trajectories for W1 to W3, shots for W4
    w1, w2, w3, w4 = (16, 16, 16, 2000) if quick else (51_200, 9_000, 184_000, 200_000)
    spec64 = erasure_spec(16 if quick else 64)
    spec256 = erasure_spec(24 if quick else 256)
    spec_pt = purification_spec(16 if quick else 128)
    rounds = 4 if quick else 400
    circuit = repetition_memory(5, rounds, 0.02)

    def ensemble(spec):
        def run(threads: int, units: int):
            return ensemble_digest(spec.run(units, seed=SEED, num_threads=threads))

        return run

    def shots(threads: int, units: int):
        res = sg.sample(circuit, units, seed=SEED, num_threads=threads)
        return digest(res.measurements, res.detectors, res.observables, res.random)

    return [
        dict(
            name="W1",
            what=f"ensemble, Watts-Strogatz n={spec64.num_system} k=4 beta=0.1, purification, "
            "p=0.1, 256 layers, random_edges; code_dimension every 32 layers, "
            "erasure_first_failure(512) and uncorrectable_region_counts(2) at the end",
            unit="trajectories",
            units=w1,
            run=ensemble(spec64),
        ),
        dict(
            name="W2",
            what=f"as W1 at n={spec256.num_system}",
            unit="trajectories",
            units=w2,
            run=ensemble(spec256),
        ),
        dict(
            name="W3",
            what=f"purification time, cycle n={spec_pt.num_system}, single reference, p=0.1, "
            f"{spec_pt.layers} layers, stop_when_purified, code_dimension every layer",
            unit="trajectories",
            units=w3,
            run=ensemble(spec_pt),
        ),
        dict(
            name="W4",
            what="sparsegf2.sample of a noisy bit-flip repetition-code memory (distance 5, "
            f"{rounds} rounds, X_ERROR 0.02), phases on",
            unit="shots",
            units=w4,
            run=shots,
        ),
    ]


def timed(fn, threads: int, units: int, repeats: int):
    times, digests = [], set()
    for _ in range(repeats):
        t0 = time.perf_counter()
        digests.add(fn(threads, units))
        times.append(time.perf_counter() - t0)
    return times, digests


def render_markdown(results: dict) -> str:
    env = results["environment"]
    safe = "safe" if env["threading_layer_concurrency_safe"] else "not safe"
    lines = [
        "# Thread scaling",
        "",
        f"CPU: {env['cpu_model']} ({env['physical_cores']} physical cores, "
        f"{env['logical_processors']} logical processors). Numba {env['numba']}, "
        f"`NUMBA_NUM_THREADS` = {env['numba_num_threads']}, threading layer "
        f"`{env['threading_layer']}` ({safe} for concurrent calls from several Python threads). "
        f"Wall time is the median of {results['repeats']} runs after a warm-up (one run at "
        "fewer than 8 threads, where each run lasts tens of seconds). "
        "Results were identical at every thread count.",
        "",
    ]
    for w in results["workloads"]:
        lines += [
            f"## {w['name']}: {w['what']}",
            "",
            f"{w['units']} {w['unit']} per run.",
            "",
            f"| threads | runs | wall time (s) | {w['unit']}/s | speedup | efficiency |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in w["rows"]:
            lines.append(
                f"| {r['threads']} | {r['runs']} | {r['seconds']:.3f} | {r['throughput']:.4g} | "
                f"{r['speedup']:.2f} | {r['efficiency']:.2f} |"
            )
        lines += ["", f"Most threads versus 16: {w['beats_16']}.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(HERE / "results"), help="output directory")
    ap.add_argument("--quick", action="store_true", help="tiny sizes, for a smoke test")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every workload size")
    ap.add_argument("--only", nargs="*", default=None, help="run only these, e.g. W1 W4")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    repeats = 1 if args.quick else args.repeats
    maximum = numba.config.NUMBA_NUM_THREADS
    counts = sorted({1, 2, 4, 8, 16, maximum} if not args.quick else {1, min(2, maximum)})
    counts = [t for t in counts if t <= maximum]

    workloads = make_workloads(args.quick)
    if args.only:
        workloads = [w for w in workloads if w["name"] in args.only]
    for w in workloads:
        w["units"] = max(1, int(w["units"] * args.scale))

    # Warm-up at the full thread count compiles every kernel; then the layer can be read.
    for w in workloads:
        w["run"](maximum, 8 if w["unit"] == "trajectories" else 64)
    layer = numba.threading_layer()
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **cpu_info(),
        "logical_processors": os.cpu_count(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "sparsegf2": sg.__version__,
        "numba_num_threads": maximum,
        "threading_layer": layer,
        "threading_layer_concurrency_safe": layer in ("tbb", "omp"),
        "threading_layer_note": "Numba documents tbb and omp as safe for concurrent calls from "
        "several Python threads (as MCP background jobs make) and workqueue as not.",
    }
    print(
        f"{env['cpu_model']}: {env['physical_cores']} cores, {env['logical_processors']} logical; "
        f"numba {env['numba']}, {maximum} threads, layer {layer}",
        flush=True,
    )

    results = {
        "environment": env,
        "repeats": repeats,
        "seed": SEED,
        "quick": args.quick,
        "identical_across_threads": True,
        "workloads": [],
    }
    start = time.perf_counter()
    for w in workloads:
        rows, reference = [], None
        for t in counts:
            # Runs of tens of seconds on few threads are steady: one run each keeps the
            # whole benchmark near 20 minutes; every run at 8 or more threads is repeated.
            reps = repeats if t >= 8 or args.quick else 1
            times, digests = timed(w["run"], t, w["units"], reps)
            if len(digests) != 1 or (reference is not None and digests != reference):
                raise SystemExit(
                    f"{w['name']}: results at {t} threads differ from those at {counts[0]} "
                    "thread(s) or between repeats; results must not depend on the thread count"
                )
            reference = digests
            sec = statistics.median(times)
            base = rows[0]["seconds"] if rows else sec
            rows.append(
                {
                    "threads": t,
                    "seconds": sec,
                    "runs": reps,
                    "all_seconds": times,
                    "throughput": w["units"] / sec,
                    "speedup": base / sec,
                    "efficiency": base / sec / (t / counts[0]),
                }
            )
            print(
                f"{w['name']} {t:>2} threads {sec:8.3f} s  {w['units'] / sec:12.4g} "
                f"{w['unit']}/s  speedup {base / sec:5.2f}",
                flush=True,
            )
        by = {r["threads"]: r for r in rows}
        if 16 in by and maximum > 16:
            ratio = by[16]["seconds"] / by[maximum]["seconds"]
            beats = (
                f"{maximum} threads take {by[maximum]['seconds']:.3f} s against "
                f"{by[16]['seconds']:.3f} s at 16 ({ratio:.2f}x, "
                f"{'faster' if ratio > 1.0 else 'not faster'})"
            )
        else:
            beats = "not applicable (no thread count above 16)"
        results["workloads"].append(
            {
                "name": w["name"],
                "what": w["what"],
                "unit": w["unit"],
                "units": w["units"],
                "digest": next(iter(reference)),
                "rows": rows,
                "beats_16": beats,
            }
        )
    results["total_seconds"] = time.perf_counter() - start
    (out / "thread_scaling.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "thread_scaling.md").write_text(render_markdown(results), encoding="utf-8")
    print(f"identical at every thread count; total {results['total_seconds']:.0f} s; wrote {out}")


if __name__ == "__main__":
    main()
