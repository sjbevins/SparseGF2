"""What the study's tableau store costs: ingestion, storage, memory and lookup.

The numbers of docs/study/05-store-and-reproducibility.md §5.8 come from this script.

**Ingestion.** For n in {32, 64, 128, 256}: one condition (beta = 0.1, p = 0.1), 64
trajectories (64 graphs, one circuit each) in one work unit, T = 2n layers, the geometric
checkpoint rule with 12 points, ``contiguous_distance``, ``uniform_erasure`` with 100 orderings
and an exact budget of n, no correlated probe; the T* bootstrap of ``peak_D`` with 200
replicates (the manifest's default) and without it. A warm-up store of the same shape runs
first (kernels compiled and cached). Reported per n: the engine's seconds (the unit's
``engine_seconds``), the rest of the unit (canonical forms, records, the transaction), their
ratio, trajectories and snapshots per second, best of ``--repeats``.

**Storage.** From the same stores: the compressed joint and physical bytes per snapshot (the
blobs) and the file bytes per snapshot (the SQLite file after a checkpoint of its log).

**Memory.** ``tracemalloc``'s peak while one unit runs, against the estimate the plan sizes
units by (``runner._bytes_per_trajectory`` times the trajectories).

**Lookup.** ``store.tableau`` and ``store.snapshot`` by key (2000 random keys and depths) in
stores of 10^3, 10^4 (and with ``--large`` 10^5) snapshots at n = 6.

    python benchmarks/study_store_costs.py                   # a few minutes
    python benchmarks/study_store_costs.py --quick           # smoke test, n = 32 only
    python benchmarks/study_store_costs.py --from-json results/study_store_costs.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import shutil
import sqlite3
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

from sparsegf2.study import runner
from sparsegf2.study.manifest import ExperimentManifest
from sparsegf2.study.store import STORE_FILE, TableauStore

HERE = Path(__file__).resolve().parent


def _manifest(name: str, n: int, bootstrap: int, graphs: int = 64) -> ExperimentManifest:
    return ExperimentManifest.from_json(
        {
            "experiment_id": name,
            "conditions": [{"n": [n], "beta": ["0.1"], "p": ["0.1"]}],
            "depth": {"layers": {"per_n": 2}, "checkpoints": {"count": 12}},
            "trials": {"graphs": graphs, "circuits_per_graph": 1},
            "seeds": {"master_graph_seed": 1, "master_circuit_seed": 2},
            "observables": {
                "tableau_snapshots": True,
                "contiguous_distance": True,
                "uniform_erasure": {"samples": 100, "exact_budget": n},
            },
            "storage": {"graphs_per_unit": graphs},
            "analysis": {"peak": {"bootstrap": bootstrap}},
        }
    )


def _ingest(root: Path, n: int, bootstrap: int, repeats: int) -> dict[str, Any]:
    with TableauStore.create(root / f"warm{n}-{bootstrap}", _manifest("warm", n, bootstrap)) as s:
        runner.run(s)
    best: dict[str, Any] | None = None
    for rep in range(repeats):
        path = root / f"s{n}-{bootstrap}-{rep}"
        with TableauStore.create(path, _manifest("bench", n, bootstrap)) as store:
            infos: list[dict[str, Any]] = []
            runner.run(store, progress=lambda u, i, t, info: infos.append(dict(info)))
            engine = sum(float(i["engine_seconds"]) for i in infos)
            unit = sum(float(i["elapsed_seconds"]) for i in infos)
            db = store._db()
            snaps = int(db.execute("SELECT count(*) FROM snapshots").fetchone()[0])
            joint, phys = db.execute(
                "SELECT avg(length(joint)), avg(length(phys)) FROM snapshots WHERE joint IS NOT NULL"
            ).fetchone()
            trajectories = int(db.execute("SELECT count(*) FROM trajectories").fetchone()[0])
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        size = (path / STORE_FILE).stat().st_size
        row = {
            "n": n,
            "bootstrap": bootstrap,
            "trajectories": trajectories,
            "snapshots": snaps,
            "engine_seconds": engine,
            "ingestion_seconds": unit - engine,
            "ratio": (unit - engine) / engine,
            "trajectories_per_second": trajectories / (unit - engine),
            "snapshots_per_second": snaps / (unit - engine),
            "joint_bytes": float(joint),
            "phys_bytes": float(phys),
            "file_bytes_per_snapshot": size / snaps,
        }
        if best is None or row["ingestion_seconds"] < best["ingestion_seconds"]:
            best = row
        shutil.rmtree(path, ignore_errors=True)
    assert best is not None
    return best


def _memory(root: Path, n: int) -> dict[str, Any]:
    with TableauStore.create(root / f"memwarm{n}", _manifest("m", n, 0, 32)) as s:
        runner.run(s)
    with TableauStore.create(root / f"mem{n}", _manifest("m", n, 0, 32)) as store:
        units = runner.plan(store)
        req = runner.requirements(store.revisions(), "F", n, "0.1")
        estimate = runner._bytes_per_trajectory(req) * units[0].trajectories
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        runner.run(store, units=units)
        peak = tracemalloc.get_traced_memory()[1] - base
        tracemalloc.stop()
    return {"n": n, "trajectories": 32, "estimate_mib": estimate / 2**20, "peak_mib": peak / 2**20}


def _lookup(root: Path, trajectories: int) -> dict[str, Any]:
    m = ExperimentManifest.from_json(
        {
            "experiment_id": "lookup",
            "conditions": [{"n": [6], "beta": ["0.25"], "p": ["0.1"]}],
            "depth": {"layers": {"fixed": 10}, "checkpoints": {"list": [2, 5]}},
            "trials": {"graphs": trajectories // 2, "circuits_per_graph": 2},
            "seeds": {"master_graph_seed": 1, "master_circuit_seed": 2},
            "observables": {"tableau_snapshots": True, "contiguous_distance": False},
            "storage": {"graphs_per_unit": 256},
            "analysis": {"peak": {"bootstrap": 0}},
        }
    )
    with TableauStore.create(root / f"lookup{trajectories}", m) as store:
        runner.run(store)
        keys = store.trajectory_keys()
        snaps = int(store._db().execute("SELECT count(*) FROM snapshots").fetchone()[0])
        rng = random.Random(0)
        picks = [(rng.choice(keys), rng.choice((0, 1, 2, 5, 10))) for _ in range(2000)]
        out: dict[str, Any] = {"snapshots": snaps, "trajectories": len(keys)}
        for name, call in (
            ("tableau_us", lambda k, t: store.tableau(k, t)),
            ("snapshot_us", lambda k, t: store.snapshot(k, t)),
        ):
            start = time.perf_counter()
            for k, t in picks:
                call(k, t)
            out[name] = (time.perf_counter() - start) / len(picks) * 1e6
    return out


def _markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Costs of the study's tableau store",
        "",
        f"Machine: {data['machine']}; {data['threads']} threads. Made by "
        "`benchmarks/study_store_costs.py`.",
        "",
        "## Ingestion (64 trajectories, one unit)",
        "",
        "| n | T* bootstrap | engine (s) | rest of the unit (s) | ratio | trajectories/s | "
        "snapshots/s | joint bytes/snapshot | physical bytes | file bytes/snapshot |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in data["ingestion"]:
        lines.append(
            f"| {r['n']} | {r['bootstrap']} | {r['engine_seconds']:.3f} | "
            f"{r['ingestion_seconds']:.2f} | {r['ratio']:.0f} | "
            f"{r['trajectories_per_second']:.0f} | {r['snapshots_per_second']:.0f} | "
            f"{r['joint_bytes']:.0f} | {r['phys_bytes']:.0f} | {r['file_bytes_per_snapshot']:.0f} |"
        )
    lines += ["", "## Memory of one unit (32 trajectories)", "",
              "| n | estimate (MiB) | traced peak (MiB) |", "|---|---|---|"]  # fmt: skip
    for r in data["memory"]:
        lines.append(f"| {r['n']} | {r['estimate_mib']:.1f} | {r['peak_mib']:.1f} |")
    lines += ["", "## Lookup by key (n = 6)", "",
              "| snapshots | trajectories | `store.tableau` (us) | `store.snapshot` (us) |",
              "|---|---|---|---|"]  # fmt: skip
    for r in data["lookup"]:
        lines.append(
            f"| {r['snapshots']} | {r['trajectories']} | {r['tableau_us']:.0f} | "
            f"{r['snapshot_us']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="n = 32 only, one repeat")
    parser.add_argument("--large", action="store_true", help="add a lookup store of 10^5")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--from-json", type=Path, default=None)
    args = parser.parse_args()
    if args.from_json is not None:
        data = json.loads(args.from_json.read_text(encoding="utf-8"))
    else:
        sizes = [32] if args.quick else [32, 64, 128, 256]
        repeats = 1 if args.quick else args.repeats
        root = Path(tempfile.mkdtemp(prefix="study-costs-"))
        try:
            ingestion = [_ingest(root, n, b, repeats) for n in sizes for b in (200, 0)]
            memory = [_memory(root, n) for n in sizes]
            counts = [200] if args.quick else [200, 2000] + ([20000] if args.large else [])
            lookup = [_lookup(root, c) for c in counts]
        finally:
            shutil.rmtree(root, ignore_errors=True)
        data = {
            "machine": f"{platform.processor() or platform.machine()}, {platform.platform()}",
            "threads": os.cpu_count(),
            "sqlite": sqlite3.sqlite_version,
            "ingestion": ingestion,
            "memory": memory,
            "lookup": lookup,
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "study_store_costs.json").write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
    text = _markdown(data)
    if args.from_json is None:
        (args.out / "study_store_costs.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
