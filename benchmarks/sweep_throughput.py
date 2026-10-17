"""Baseline throughput of Watts–Strogatz sweeps through ``sparsegf2.study.sweep.run``.

One condition per run, each in a fresh directory: n = 64, 128, 256, 512; p = 0.05 and 0.2;
beta = 0.1; the purification picture; ``n // 2`` random-edge gates per layer; T = 4n layers;
``code_dimension`` at every layer, without and with a ``tableau_snapshot`` on the geometric
checkpoints (count 12); default threads, chunks, compression ("fast") and writer threads;
four circuits per graph. The wall time covers the whole ``sweep.run`` call (graph file, index,
engine, saving the parts); ``engine`` is the engine's own share (the parts'
``elapsed_seconds``). Disk bytes per trajectory are the condition folder's part files divided
by its trajectories (the graph files and the small JSON records are not counted). A warm-up
run compiles the kernels first. Each directory is deleted after it is measured.

A second table times saving one part with snapshots, of the largest size a sweep makes
(``sweep.PART_BYTES``), at n = 128 and 256 with each compression of ``EnsembleResult.save``
(``fsync`` included), against computing it. Usage, from the repository root, on an idle
machine::

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
from sparsegf2 import api, mipt  # noqa: E402
from sparsegf2.study import identity, sweep  # noqa: E402
from sparsegf2.study.sweep import SweepSpec  # noqa: E402

RATES = ("0.05", "0.2")
CIRCUITS_PER_GRAPH = 4
#: Trajectories per condition, chosen for 5 to 60 s each on a 32-thread machine (16 cores).
FULL = {  # n: (without snapshots, with snapshots)
    64: (73728, 49152),
    128: (36864, 12288),
    256: (18432, 6144),
    512: (4608, 1536),
}
QUICK = {16: (64, 32), 24: (64, 32)}
SAVE_SIZES = (128, 256)
QUICK_SAVE_SIZES = (16, 24)
COMPRESSIONS = ("none", "fast", "default")


def condition_spec(n: int, p: str, snapshots: bool, trajectories: int) -> dict:
    probes = [{"kind": "code_dimension", "checkpoints": "every_layer"}]
    if snapshots:
        probes.append({"kind": "tableau_snapshot"})
    return {
        "sizes": [n],
        "betas": ["0.1"],
        "rates": [p],
        "graphs": -(-trajectories // CIRCUITS_PER_GRAPH),
        "circuits_per_graph": CIRCUITS_PER_GRAPH,
        "stride": CIRCUITS_PER_GRAPH,
        "layers": {"per_n": 4},
        "checkpoints": {"count": 12},
        "probes": probes,
        "master_graph_seed": 1,
        "master_circuit_seed": 2,
    }


def measure(n: int, p: str, snapshots: bool, trajectories: int, scratch: Path, **options) -> dict:
    """One condition through ``sweep.run`` in a fresh directory (``options``: its keyword
    arguments, such as ``writers`` or ``performance``)."""
    directory = scratch / f"n{n}_p{p}_{'snap' if snapshots else 'k'}"
    start = time.perf_counter()
    out = sweep.run(condition_spec(n, p, snapshots, trajectories), directory, **options)
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


def measure_saves(n: int, scratch: Path, limit: int | None = None) -> dict:
    """One part of the largest size a sweep saves with snapshots (at most ``limit``
    trajectories): compute it, then save it with every compression (``fsync`` included)."""
    first = [identity.build_graph(n, "0.1", 0, 1).edges]
    probe = SweepSpec.from_json(condition_spec(n, "0.05", True, CIRCUITS_PER_GRAPH))
    per_trajectory = api._result_bytes(probe.monitored_circuit(n, "0.1", "0.05", first), 1)
    size = min(max(1, sweep.PART_BYTES // per_trajectory), limit or sweep.PART_BYTES)
    spec = SweepSpec.from_json(condition_spec(n, "0.05", True, size))
    collection = [identity.build_graph(n, "0.1", g, 1).edges for g in range(spec.graphs[n])]
    mc = spec.monitored_circuit(n, "0.1", "0.05", collection)
    start = time.perf_counter()
    result = mc.run(indices=np.arange(size), seed=list(spec.run_seed(n, "0.1")))
    row: dict = {"n": n, "trajectories": size}
    row["compute_seconds"] = round(time.perf_counter() - start, 3)
    for compression in COMPRESSIONS:
        path = scratch / f"part_{n}_{compression}.npz"
        start = time.perf_counter()
        result.save(path, compression=compression)
        with open(path, "r+b") as fh:
            os.fsync(fh.fileno())
        row[compression] = {
            "seconds": round(time.perf_counter() - start, 3),
            "megabytes": round(path.stat().st_size / 1e6, 1),
        }
        path.unlink()
    return row


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
        "`tableau_snapshot` on the geometric checkpoints (count 12); default threads, chunks, "
        "compression (`fast`) and writer threads, four circuits per graph. `seconds` is the "
        "wall time of the whole call (graph file, index, engine, saved parts), `engine` the "
        "engine's share; disk bytes per trajectory count the condition's part files only.",
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
    lines += [
        "",
        "Saving one part with snapshots (p = 0.05, the largest part a sweep makes) against "
        "computing it, per compression of `EnsembleResult.save`, `fsync` included (seconds, "
        "megabytes on disk):",
        "",
        "| n | trajectories | compute s | none | fast | default |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results["saves"]:
        cells = " | ".join(
            f"{r[c]['seconds']:.2f} s, {r[c]['megabytes']:.1f} MB" for c in COMPRESSIONS
        )
        lines.append(f"| {r['n']} | {r['trajectories']} | {r['compute_seconds']:.2f} | {cells} |")
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
        rows, saves, start = [], [], time.perf_counter()
        for n, counts in plan.items():
            for p in RATES:
                for snapshots, trajectories in zip((False, True), counts, strict=True):
                    rows.append(measure(n, p, snapshots, trajectories, scratch))
                    print(rows[-1], flush=True)
        for n in QUICK_SAVE_SIZES if args.quick else SAVE_SIZES:
            saves.append(measure_saves(n, scratch, 64 if args.quick else None))
            print(saves[-1], flush=True)
    results = {
        "environment": env,
        "quick": args.quick,
        "rows": rows,
        "saves": saves,
        "total_seconds": round(time.perf_counter() - start, 1),
    }
    write(results, out)


if __name__ == "__main__":
    main()
