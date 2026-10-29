"""One-thread cost of the weight-one logical-operator checks, beside a layer and the full distance.

The encoding-onset question asks, at every layer, whether the code still has a qubit carrying a
weight-one logical operator: N_log = #{i : I(i:R) > 0} (``mipt.logical_sites``), whose zero with
k > 0 is exactly ``min(d_cont, 2) == 2`` (``mipt.contiguous_distance(cap=2)``). This measures,
on one thread and per evaluation, on codes of monitored Watts-Strogatz trajectories:

* ``layer``: one dynamics layer (``n // 2`` random edges, measurements at rate p), from runs
  without probes;
* ``cap2``: ``contiguous_distance_rows_capped(cap=2)``, which stops at the first qubit with
  I(i:R) > 0 (cheapest with an early defect, ``2 n W`` words without one);
* ``logical_sites``: ``site_information_rows`` counting every such qubit (always ``2 n W``
  words: what cap 2 spends without a defect);
* ``site_information``: the same pass writing the n values;
* ``full``: the exact contiguous distance (``contiguous_distance_rows``, two gauge
  eliminations), skipped above ``--full-max-n``.

**States**: ``reference="purification"``, ``schedule="random_edges"``, one Watts-Strogatz graph
per beta (k = 4), rate p; snapshots of the joint tableau at each depth in ``--depths``
(``tableau_snapshot``), rebuilt as the engine's column tableau with its reference rows. Each
kernel runs ``R`` times in a compiled loop (R chosen so a timing lasts about
``--target-seconds``), medians over ``--repeats`` timings after a warm-up. The answers are
checked against each other on every state (N_log = 0 iff cap 2 gives 2 when k > 0; the full
distance is 1 iff N_log > 0).

    python benchmarks/logical_sites_cost.py                      # minutes, one thread; idle machine
    python benchmarks/logical_sites_cost.py --quick --out <dir>  # smoke test, a few seconds

Writes ``logical_sites_cost.json`` and ``logical_sites_cost.md`` into ``--out`` (default
``benchmarks/results/``).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numba
import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from thread_scaling import cpu_info  # noqa: E402

import sparsegf2 as sg  # noqa: E402
from sparsegf2 import graphs, mipt  # noqa: E402
from sparsegf2._kernel import entropy as ke  # noqa: E402

SEED = 29
KINDS = ("cap2", "logical_sites", "site_information", "full")


@njit(cache=False)
def _repeat_cap2(tab, ws, system, ref_rows, num_ref, reps):
    total = 0
    for _ in range(reps):
        total += ke.contiguous_distance_rows_capped(tab, ws, system, ref_rows, num_ref, True, 2)
    return total


@njit(cache=False)
def _repeat_sites(tab, ws, n, ref_rows, out, reps):
    total = 0
    for _ in range(reps):
        total += ke.site_information_rows(tab, ws, n, ref_rows, out)
    return total


@njit(cache=False)
def _repeat_full(tab, ws, system, ref_rows, num_ref, reps):
    total = 0
    for _ in range(reps):
        total += ke.contiguous_distance_rows(tab, ws, system, ref_rows, num_ref, True)
    return total


def _states(n: int, beta: float, p: float, depths: list[int], trajectories: int):
    """Engine tableaux (tab, ws, ref_rows, k, depth) of monitored trajectories at ``depths``."""
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.watts_strogatz(n, 4, beta, seed=(SEED, n)),
        layers=max(depths),
        measurement_rate=p,
        schedule="random_edges",
        reference="purification",
        checkpoints=sorted(set(depths)),
        probes=[mipt.code_dimension(), mipt.tableau_snapshot()],
    )
    res = spec.run(trajectories, seed=[SEED, n, int(1000 * beta), int(1000 * p)], num_threads=1)
    words, counts = res.snapshots["tableau_snapshot"]
    num = spec.num_qubits
    ws = (num + 63) // 64
    refs = np.arange(n, num, dtype=np.int64)
    out = []
    for t in range(trajectories):
        for c, depth in enumerate(spec.checkpoint_layers.tolist()):
            if counts[t, c] < 0:
                continue
            tab = np.zeros((2 * num + 1, ws), dtype=np.uint64)
            ke.columns_from_rows(np.ascontiguousarray(words[t, c]), num, tab)
            ref_rows = np.zeros(ws, dtype=np.uint64)
            ke.reference_rows(tab, ws, refs, ref_rows)
            out.append((tab, ws, ref_rows, int(res.values["code_dimension"][t, c]), depth))
    return out


def _time(fn, target: float, repeats: int) -> float:
    """Median seconds per call of ``fn(reps)``, with reps chosen to last about ``target``."""
    fn(1)  # warm-up (and compilation)
    reps = 1
    while True:
        start = time.perf_counter()
        fn(reps)
        elapsed = time.perf_counter() - start
        if elapsed >= target / 4 or reps >= 1 << 22:
            break
        reps *= 4
    reps = max(1, int(reps * target / max(elapsed, 1e-9)))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(reps)
        times.append((time.perf_counter() - start) / reps)
    return statistics.median(times)


def _layer_seconds(n: int, beta: float, p: float, layers: int, trajectories: int) -> float:
    """One-thread seconds per dynamics layer, from runs without probes (minus a zero-layer run)."""
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.watts_strogatz(n, 4, beta, seed=(SEED, n)),
        layers=layers,
        measurement_rate=p,
        schedule="random_edges",
        reference="purification",
    )
    empty = mipt.MonitoredCircuit(**{**_fields(spec), "layers": 0})
    spec.run(1, seed=SEED, num_threads=1)  # warm-up
    empty.run(1, seed=SEED, num_threads=1)
    start = time.perf_counter()
    spec.run(trajectories, seed=SEED, num_threads=1)
    full = time.perf_counter() - start
    start = time.perf_counter()
    empty.run(trajectories, seed=SEED, num_threads=1)
    base = time.perf_counter() - start
    return max(full - base, 0.0) / (trajectories * layers)


def _fields(spec: mipt.MonitoredCircuit) -> dict:
    return dict(
        num_system=spec.num_system,
        edges=spec.edges,
        measurement_rate=spec.measurement_rate,
        schedule=spec.schedule,
        reference=spec.reference,
        layers=spec.layers,
    )


def measure(args) -> dict:
    rows = []
    for n in args.sizes:
        for beta in args.betas:
            layer = _layer_seconds(n, beta, args.rate, args.layer_layers, args.layer_trajectories)
            for tab, ws, ref_rows, k, depth in _states(
                n, beta, args.rate, args.depths, args.trajectories
            ):
                system = np.arange(n, dtype=np.int64)
                out = np.zeros(n, dtype=np.int64)
                empty = np.zeros(0, dtype=np.int64)
                n_log = int(ke.site_information_rows(tab, ws, n, ref_rows, empty))
                cap2 = int(
                    ke.contiguous_distance_rows_capped(tab, ws, system, ref_rows, n, True, 2)
                )
                if k > 0 and (cap2 == 2) != (n_log == 0):
                    raise AssertionError(f"cap 2 and N_log disagree at n={n}, depth={depth}")
                seconds = {
                    "cap2": _time(
                        lambda r, tab=tab, ws=ws, s=system, rr=ref_rows, n=n: _repeat_cap2(
                            tab, ws, s, rr, n, r
                        ),
                        args.target_seconds,
                        args.repeats,
                    ),
                    "logical_sites": _time(
                        lambda r, tab=tab, ws=ws, rr=ref_rows, n=n, e=empty: _repeat_sites(
                            tab, ws, n, rr, e, r
                        ),
                        args.target_seconds,
                        args.repeats,
                    ),
                    "site_information": _time(
                        lambda r, tab=tab, ws=ws, rr=ref_rows, n=n, o=out: _repeat_sites(
                            tab, ws, n, rr, o, r
                        ),
                        args.target_seconds,
                        args.repeats,
                    ),
                }
                full = None
                if n <= args.full_max_n:
                    full = int(ke.contiguous_distance_rows(tab, ws, system, ref_rows, n, True))
                    if k > 0 and (full == 1) != (n_log > 0):
                        raise AssertionError(f"the full distance and N_log disagree at n={n}")
                    seconds["full"] = _time(
                        lambda r, tab=tab, ws=ws, s=system, rr=ref_rows, n=n: _repeat_full(
                            tab, ws, s, rr, n, r
                        ),
                        args.target_seconds,
                        min(args.repeats, 3),
                    )
                rows.append(
                    {
                        "n": n, "beta": beta, "p": args.rate, "depth": depth, "k": k,
                        "n_log": n_log, "cap2": cap2, "distance": full,
                        "layer_seconds": layer, "seconds": seconds,
                    }
                )  # fmt: skip
                print(
                    f"n={n:5d} beta={beta:g} T={depth:5d} k={k:5d} N_log={n_log:5d} "
                    + " ".join(f"{name}={1e6 * s:9.2f}us" for name, s in seconds.items())
                    + f" layer={1e6 * layer:9.2f}us",
                    flush=True,
                )
    return {
        "quick": args.quick,
        "environment": {
            "sparsegf2": sg.__version__,
            "numba": numba.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "threads": 1,
            **cpu_info(),
        },
        "settings": {
            "rate": args.rate,
            "depths": args.depths,
            "trajectories": args.trajectories,
            "target_seconds": args.target_seconds,
            "repeats": args.repeats,
            "full_max_n": args.full_max_n,
        },
        "rows": rows,
    }


def markdown(data: dict) -> str:
    env = data["environment"]
    lines = [
        "# Cost of the weight-one logical-operator checks",
        "",
        f"One thread, {env.get('cpu_model') or 'unknown CPU'}, sparsegf2 {env['sparsegf2']}, "
        f"numba {env['numba']}; rate p = {data['settings']['rate']}, purification picture, "
        "Watts-Strogatz k = 4, `random_edges`. Microseconds per evaluation (median); `layer` is "
        "one dynamics layer. N_log is the number of qubits with I(i:R) > 0; cap 2 stops at the "
        "first of them."
        + (" **Quick mode: smoke-test sizes, not for decisions.**" if data["quick"] else ""),
        "",
        "| n | beta | depth | k | N_log | layer | cap 2 | logical_sites | site_information "
        "| full d_cont | full / logical_sites |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in data["rows"]:
        s = r["seconds"]
        full = s.get("full")
        full_us = "" if full is None else f"{1e6 * full:.2f}"
        ratio = "" if full is None else f"{full / s['logical_sites']:.1f}"
        lines.append(
            f"| {r['n']} | {r['beta']:g} | {r['depth']} | {r['k']} | {r['n_log']} "
            f"| {1e6 * r['layer_seconds']:.2f} | {1e6 * s['cap2']:.2f} "
            f"| {1e6 * s['logical_sites']:.2f} | {1e6 * s['site_information']:.2f} "
            f"| {full_us} | {ratio} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 16, 64, 256])
    parser.add_argument("--trajectories", type=int, default=2)
    parser.add_argument("--target-seconds", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--full-max-n", type=int, default=1024)
    parser.add_argument("--layer-layers", type=int, default=64)
    parser.add_argument("--layer-trajectories", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="tiny sizes: a smoke test")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)
    if args.quick:
        args.sizes, args.depths, args.trajectories = [32, 64], [2, 8, 48], 1
        args.target_seconds, args.repeats = 0.01, 2
        args.layer_layers, args.layer_trajectories = 8, 2
    numba.set_num_threads(1)
    data = measure(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logical_sites_cost.json").write_text(json.dumps(data, indent=1), "utf-8")
    (args.out / "logical_sites_cost.md").write_text(markdown(data), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
