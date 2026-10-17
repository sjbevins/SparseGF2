"""Baseline throughput of Watts–Strogatz sweeps through ``sparsegf2.study.sweep.run``.

One condition per run, each in a fresh directory: n = 64, 128, 256, 512; p = 0.05 and 0.2;
beta = 0.1; the purification picture; ``n // 2`` random-edge gates per layer; T = 4n layers;
``code_dimension`` at every layer, without and with a ``tableau_snapshot`` on the geometric
checkpoints (count 12); default threads and chunks; four circuits per graph. The wall time
covers the whole ``sweep.run`` call (graph file, index, engine, saving the compressed parts);
``engine`` is the engine's own share (the parts' ``elapsed_seconds``). Disk bytes per trajectory
are the condition folder's part files divided by its trajectories (the graph files and the small
JSON records are not counted). A warm-up run compiles the kernels first. Each directory is
deleted after it is measured. Usage, from the repository root, on an idle machine::

    python benchmarks/sweep_throughput.py              # about 5 minutes, all cores
    python benchmarks/sweep_throughput.py --quick --out <dir>   # smoke test, seconds
    python benchmarks/sweep_throughput.py --from-json benchmarks/results/sweep_throughput.json

Writes ``sweep_throughput.json`` and ``sweep_throughput.md`` into ``--out`` (default
``benchmarks/results/``); ``--from-json`` measures nothing and renders the markdown again.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numba
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from thread_scaling import cpu_info  # noqa: E402

import sparsegf2 as sg  # noqa: E402
from sparsegf2 import mipt  # noqa: E402
from sparsegf2.study import sweep  # noqa: E402

RATES = ("0.05", "0.2")
CIRCUITS_PER_GRAPH = 4
#: Trajectories per condition, chosen for 5 to 45 s each on a 32-thread machine (16 cores).
FULL = {  # n: (without snapshots, with snapshots)
    64: (49152, 24576),
    128: (24576, 6144),
    256: (12288, 3072),
    512: (3072, 768),
}
QUICK = {16: (64, 32), 24: (64, 32)}


def condition_spec(n: int, p: str, snapshots: bool, trajectories: int) -> dict:
    probes = [{"kind": "code_dimension", "checkpoints": "every_layer"}]
    if snapshots:
        probes.append({"kind": "tableau_snapshot"})
    return {
        "sizes": [n],
        "betas": ["0.1"],
        "rates": [p],
        "graphs": trajectories // CIRCUITS_PER_GRAPH,
        "circuits_per_graph": CIRCUITS_PER_GRAPH,
        "stride": CIRCUITS_PER_GRAPH,
        "layers": {"per_n": 4},
        "checkpoints": {"count": 12},
        "probes": probes,
        "master_graph_seed": 1,
        "master_circuit_seed": 2,
    }


def measure(n: int, p: str, snapshots: bool, trajectories: int, scratch: Path) -> dict:
    directory = scratch / f"n{n}_p{p}_{'snap' if snapshots else 'k'}"
    start = time.perf_counter()
    out = sweep.run(condition_spec(n, p, snapshots, trajectories), directory)
    seconds = time.perf_counter() - start
    assert out["complete"] and out["saved"] == trajectories
    folder = directory / out["conditions"][0]["path"]
    files = sorted(folder.glob("result-*.npz"))
    engine = sum(mipt.EnsembleResult.load(f).elapsed_seconds for f in files)
    part_bytes = sum(f.stat().st_size for f in files)
    shutil.rmtree(directory)
    return {
        "n": n,
        "p": p,
        "snapshots": snapshots,
        "layers": 4 * n,
        "trajectories": trajectories,
        "parts": len(files),
        "seconds": round(seconds, 3),
        "trajectories_per_second": round(trajectories / seconds, 2),
        "engine_seconds": round(engine, 3),
        "disk_bytes_per_trajectory": round(part_bytes / trajectories),
    }


def render_markdown(results: dict) -> str:
    env = results["environment"]
    lines = [
        "# Sweep throughput",
        "",
        f"CPU: {env['cpu_model']} ({env['physical_cores']} physical cores, "
        f"{env['logical_processors']} logical processors), {env['platform']}. Python "
        f"{env['python']}, NumPy {env['numpy']}, Numba {env['numba']} "
        f"(`NUMBA_NUM_THREADS` = {env['numba_num_threads']}, threading layer "
        f"`{env['threading_layer']}`), sparsegf2 {env['sparsegf2']}.",
        "",
        "One condition per `sweep.run` call, in a fresh directory: Watts–Strogatz graphs "
        "(k = 4, beta = 0.1), the purification picture, n // 2 random-edge gates per layer, "
        "T = 4n layers, `code_dimension` at every layer, and with `snapshots` a "
        "`tableau_snapshot` on the geometric checkpoints (count 12); default threads and "
        "chunks, four circuits per graph. `seconds` is the wall time of the whole call (graph "
        "file, index, engine, compressed parts), `engine` the engine's share; disk bytes per "
        "trajectory count the condition's part files only.",
        "",
        "| n | p | snapshots | trajectories | seconds | trajectories/s | engine s | "
        "disk bytes/trajectory |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for r in results["rows"]:
        lines.append(
            f"| {r['n']} | {r['p']} | {'yes' if r['snapshots'] else 'no'} | "
            f"{r['trajectories']} | {r['seconds']:.2f} | {r['trajectories_per_second']:.4g} | "
            f"{r['engine_seconds']:.2f} | {r['disk_bytes_per_trajectory']:,} |"
        )
    lines += ["", f"Total wall time {results['total_seconds']:.0f} s.", ""]
    return "\n".join(lines)


def write(results: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("sweep_throughput.json", json.dumps(results, indent=1) + "\n"),
        ("sweep_throughput.md", render_markdown(results)),
    ):
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(HERE / "results"), help="output directory")
    ap.add_argument("--quick", action="store_true", help="tiny sizes, for a smoke test")
    ap.add_argument("--from-json", default=None, help="render a saved results file again")
    args = ap.parse_args()
    out = Path(args.out)
    if args.from_json:
        write(json.loads(Path(args.from_json).read_text("utf-8")), out)
        return
    plan = QUICK if args.quick else FULL
    with tempfile.TemporaryDirectory(prefix="sweep_throughput_") as tmp:
        scratch = Path(tmp)
        sweep.run(condition_spec(8, "0.1", True, 8), scratch / "warm-up")  # compiles
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
        rows, start = [], time.perf_counter()
        for n, counts in plan.items():
            for p in RATES:
                for snapshots, trajectories in zip((False, True), counts, strict=True):
                    row = measure(n, p, snapshots, trajectories, scratch)
                    rows.append(row)
                    print(row, flush=True)
    results = {
        "environment": env,
        "quick": args.quick,
        "rows": rows,
        "total_seconds": round(time.perf_counter() - start, 1),
    }
    write(results, out)


if __name__ == "__main__":
    main()
