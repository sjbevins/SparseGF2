"""One-thread cost of every probe kind per evaluation, against the runtime estimate.

``api.estimate_ensemble`` and the ``max_seconds`` guard of ``api.run_ensemble`` price a
trajectory with ``api._trajectory_us``, whose probe terms are ``api._probe_us``. The estimate is
meant to be deliberately pessimistic: at least 1.5 times the measured cost on the machine that
fitted it (the fit aims at about 3 times), so that slower machines are still covered. This
measures what every probe kind costs per evaluation, on one thread, in the states that make it
expensive, beside its estimate.

**States**, one trajectory each, probes at the final layer:

* ``single``: ``reference="single"``, ``2 n ceil(log2 n)`` random all-to-all gates, no layers.
  k = 1 and a random code whose erasures fail only at about n / 2 qubits: the worst case of the
  erasure scans (contiguous distance, first-failure orderings, exhaustive counts), whose work
  grows with the distance.
* ``purification``: ``reference="purification"``, the same scramble: k = n and a volume-law
  state on 2n qubits, the worst case of the entanglement probes (their eliminations run over
  twice the words) and of the elimination every erasure probe starts with.
* ``monitored``: ``reference="purification"``, the same scramble, then ``min(n / 2, 256)``
  layers on a Watts-Strogatz graph (k = 4, beta = 0.1), ``random_edges``, measurement rate 0.05:
  an intermediate code, as in erasure studies.

**Measured**: the trajectory rerun with R copies of the probe at its final layer, minus the same
run without probes, divided by R (R makes the copies take about as long as the run without
probes, at most 4096, chosen again from the measured cost when a noisy first guess gave too few;
R = 1 for slow probes; ``code_snapshot``, of which a spec holds one, is evaluated at the last R
layers instead). Medians over repeats, after a warm-up. Where all the copies together take under
5 % of the run without probes, the difference is mostly timing noise: the row is marked
unresolved and left out of the ratios. The run without probes is also set against its estimate
(the dynamics and setup terms). A probe is not run at a size where the growth from the sizes
below predicts more than ``--max-eval-seconds`` per evaluation.

**Reported**: measured and estimated seconds per evaluation and estimate / measured, per kind,
n and state, and the smallest ratio per kind. **All threads**: a few expensive probes at
n = 512 and 2048 with one trajectory per thread on every thread, against ``estimate_ensemble``'s
kernel term, which prices busy threads with ``api._parallel_threads``; with the speed of a busy
thread relative to a thread alone.

    python benchmarks/probe_cost.py                       # about 6 minutes, idle machine
    python benchmarks/probe_cost.py --quick --out <dir>   # smoke test, a few seconds
    python benchmarks/probe_cost.py --from-json results/probe_cost.json  # re-price only

``--from-json`` measures nothing: it prices the measured rows again with the current
``api._trajectory_us`` and rewrites both outputs (to compare a refitted model with the same
measurements).

Writes ``probe_cost.json`` and ``probe_cost.md`` into ``--out`` (default
``benchmarks/results/``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numba
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from thread_scaling import cpu_info  # noqa: E402

import sparsegf2 as sg  # noqa: E402
from sparsegf2 import api, graphs, mipt  # noqa: E402

SEED = 21
SIZES = (64, 128, 256, 512, 1024, 2048)
STATES = ("single", "purification", "monitored")
MARGIN = 1.5  # the documented pessimism: estimate >= MARGIN x measured on the fitting machine
# Run with one trajectory per thread on every thread: (probe label, state).
ALL_THREADS = (
    ("entropy_profile", "purification"),
    ("erasure_lost_profile (10 samples)", "purification"),
    ("erasure_first_failure (100 samples)", "single"),
    ("contiguous_distance", "single"),
    ("uncorrectable_counts (w = 2)", "purification"),
    ("tripartite (quarters)", "purification"),
)
ALL_THREADS_SIZES = (512, 2048)


# ---------------------------------------------------------------------------
# states and probes
# ---------------------------------------------------------------------------


def state_spec(n: int, state: str) -> mipt.MonitoredCircuit:
    """The probe-free spec of one state (see the module docstring)."""
    scramble = 2 * n * math.ceil(math.log2(n))
    if state == "monitored":
        return mipt.MonitoredCircuit(
            num_system=n,
            edges=graphs.watts_strogatz(n, 4, 0.1, seed=(SEED, n)),
            layers=min(n // 2, 256),
            measurement_rate=0.05,
            schedule="random_edges",
            reference="purification",
            scramble_gates=scramble,
            checkpoints="final",
        )
    return mipt.MonitoredCircuit(
        num_system=n,
        edges=graphs.cycle(n),
        layers=0,
        measurement_rate=0.0,
        reference=state,
        scramble_gates=scramble,
        checkpoints="final",
    )


def probe_makers(n: int) -> dict[str, Callable[[], mipt.Probe]]:
    """One probe per measured case, by label; copies are renamed."""
    q = n // 4
    return {
        "code_dimension": mipt.code_dimension,
        "entropy (all n system qubits)": lambda: mipt.entropy(range(n)),
        "half_cut": mipt.half_cut_entropy,
        "mutual_information (two quarters)": lambda: mipt.mutual_information(
            range(q), range(2 * q, 3 * q)
        ),
        "tripartite (quarters)": mipt.tripartite_information,
        "lost_information (n / 2 qubits)": lambda: mipt.lost_information(range(n // 2)),
        "contiguous_distance": mipt.contiguous_distance,
        "entropy_profile": mipt.entropy_profile,
        "erasure_lost_profile (10 samples)": lambda: mipt.erasure_lost_profile(samples=10),
        "erasure_first_failure (100 samples)": lambda: mipt.erasure_first_failure(samples=100),
        "uncorrectable_counts (w = 1)": lambda: mipt.uncorrectable_region_counts(max_size=1),
        "uncorrectable_counts (w = 2)": lambda: mipt.uncorrectable_region_counts(max_size=2),
        "uncorrectable_counts (w = 3)": lambda: mipt.uncorrectable_region_counts(max_size=3),
        "code_snapshot": mipt.code_snapshot,
    }


def with_copies(base: mipt.MonitoredCircuit, make: Callable[[], mipt.Probe], copies: int):
    """``base`` with ``copies`` evaluations of the probe: copies at the final layer, or, for
    ``code_snapshot`` (one per spec), one probe at the last ``copies`` layers."""
    probe = make()
    if probe.kind == "code_snapshot":
        assert copies <= base.layers + 1, "a snapshot has at most one evaluation per layer"
        layers = list(range(base.layers + 1 - copies, base.layers + 1))
        return replace(base, probes=(replace(probe, name="p0", checkpoints=layers),))
    return replace(
        base,
        probes=tuple(replace(probe, name=f"p{i}", checkpoints="final") for i in range(copies)),
    )


def all_threads_spec(
    n: int, state: str, make: Callable[[], mipt.Probe], evaluations: int
) -> mipt.MonitoredCircuit:
    """A scrambled state, then ``evaluations - 1`` layers without gates or measurements and the
    probe at every layer: the state stays as it is and is evaluated ``evaluations`` times."""
    return replace(
        state_spec(n, state),
        layers=evaluations - 1,
        schedule="random_edges",
        gates_per_layer=0,
        checkpoints="every_layer",
        probes=(make(),),
    )


def estimate_seconds(base: mipt.MonitoredCircuit, make: Callable[[], mipt.Probe]) -> float:
    """The model's one-thread seconds for one evaluation: what one copy adds to the trajectory."""
    one = with_copies(base, make, 1)
    return (api._trajectory_us(one) - api._trajectory_us(base)) * 1e-6


def describe(base: mipt.MonitoredCircuit, seed: int) -> dict:
    """k, the half-cut entropy and the mean first-failure size of the state (16 orderings)."""
    spec = replace(
        base,
        probes=(
            mipt.code_dimension(),
            mipt.half_cut_entropy(),
            mipt.erasure_first_failure(samples=16, name="first"),
        ),
    )
    res = run(spec, seed)
    return {
        "k": int(res.values["code_dimension"][0, -1]),
        "half_cut_entropy": int(res.values["half_cut_entropy"][0, -1]),
        "mean_first_failure": float(np.mean(res.values["first"][0, -1])),
    }


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def run(spec: mipt.MonitoredCircuit, seed: int) -> mipt.EnsembleResult:
    return spec.run(1, seed=seed, num_threads=1)


def seconds(fn, cfg: dict) -> tuple[float, int]:
    """Median wall time of ``fn()`` and the number of runs.

    Repeated until ``min_total`` seconds are spent (at least ``min_runs``, at most
    ``max_runs`` runs); a call of at least ``once_above`` seconds is not repeated.
    """
    times: list[float] = []
    while True:
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
        if times[0] >= cfg["once_above"] or len(times) >= cfg["max_runs"]:
            break
        if len(times) >= cfg["min_runs"] and sum(times) >= cfg["min_total"]:
            break
    return statistics.median(times), len(times)


def measure_probe(base: mipt.MonitoredCircuit, base_s: float, make, seed: int, cfg: dict):
    """Seconds per evaluation of ``make()`` on the state of ``base``; ``(seconds, R, runs)``."""
    t0 = time.perf_counter()
    run(with_copies(base, make, 1), seed)  # warm-up, and a first guess of the cost
    guess = max(time.perf_counter() - t0 - base_s, 1e-7)
    if guess >= cfg["once_above"] and guess > 4 * base_s:
        return guess, 1, 1  # slow: the warm-up run is the measurement
    most = base.layers + 1 if make().kind == "code_snapshot" else cfg["max_copies"]

    def enough(cost: float) -> int:  # copies that take about as long as the run without probes
        return max(1, min(most, math.ceil(max(base_s, 0.01) / max(cost, 1e-7))))

    # A guess below the timing noise could ask for thousands of copies of a probe that is not
    # cheap; floored at 1 % of the run without probes, it asks for at most 100 at first.
    copies = enough(max(guess, 0.01 * base_s))
    for _ in range(3):  # a cheap probe's first guess is mostly noise: too few copies
        spec = with_copies(base, make, copies)
        total, runs = seconds(lambda spec=spec: run(spec, seed), cfg)
        per = max(total - base_s, 0.0) / copies
        if enough(per) <= 2 * copies:
            break
        copies = enough(per)
    return per, copies, runs


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------


def measure(cfg: dict) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    bases: list[dict] = []
    # (label, state) -> the measured (n, seconds), in increasing n
    history: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for n in cfg["sizes"]:
        for state in cfg["states"]:
            base = state_spec(n, state)
            seed = SEED * 100_000 + n
            run(base, seed)  # warm-up
            base_s, base_runs = seconds(lambda base=base, seed=seed: run(base, seed), cfg)
            info = describe(base, seed)
            bases.append({"n": n, "state": state, "seconds": base_s, "runs": base_runs, **info})
            print(
                f"n={n:>5} {state:<12} run without probes {human(base_s):>9} "
                f"k={info['k']} S_half={info['half_cut_entropy']} "
                f"mean first failure {info['mean_first_failure']:.1f}",
                flush=True,
            )
            for label, make in probe_makers(n).items():
                if label not in cfg["labels"]:
                    continue
                done = history.setdefault((label, state), [])
                if done:
                    predicted = predict(done, n)
                    if predicted > cfg["max_eval_seconds"]:
                        rows.append({"label": label, "n": n, "state": state, "skipped": predicted})
                        print(f"   {label:<38} not run: predicted {human(predicted)}", flush=True)
                        continue
                measured, copies, runs = measure_probe(base, base_s, make, seed, cfg)
                # all copies together below a small fraction of the run without probes: the
                # difference is mostly timing noise
                unresolved = copies * measured < cfg["resolution"] * base_s
                if not unresolved:
                    done.append((n, measured))
                rows.append(
                    {
                        "label": label,
                        "kind": make().kind,
                        "n": n,
                        "state": state,
                        "measured_seconds": measured,
                        "copies": copies,
                        "runs": runs,
                        "unresolved": unresolved,
                    }
                )
                print(
                    f"   {label:<38} {human(measured):>9}  (R={copies}, {runs} runs)"
                    + (" unresolved" if unresolved else ""),
                    flush=True,
                )
    return rows, bases


def measure_all_threads(cfg: dict) -> list[dict]:
    """Each probe of ``ALL_THREADS``: one trajectory on one thread, then one trajectory per
    thread on every thread (the run ``estimate_ensemble`` prices with
    ``api._parallel_threads``)."""
    threads = cfg["all_threads"]
    rows = []
    for n in cfg["all_threads_sizes"]:
        evaluations = 9 if n <= 512 else 3
        for label, state in ALL_THREADS:
            if label == "contiguous_distance" and n > 1024:
                continue  # about a minute per evaluation at n = 2048
            spec = all_threads_spec(n, state, probe_makers(n)[label], evaluations)
            seed = SEED * 100_000 + n
            numba.set_num_threads(1)
            run(spec, seed)  # warm-up
            alone, _ = seconds(lambda spec=spec, seed=seed: run(spec, seed), cfg)
            numba.set_num_threads(threads)

            def busy(spec=spec, seed=seed):
                return spec.run(threads, seed=seed, num_threads=threads)

            if alone < 1.0:
                busy()  # warm-up of the thread pool
            wall, runs = seconds(busy, cfg)
            rows.append(
                {
                    "label": label,
                    "n": n,
                    "state": state,
                    "evaluations": evaluations,
                    "threads": threads,
                    "alone_seconds": alone,
                    "seconds": wall,
                    "runs": runs,
                }
            )
            print(
                f"all threads: {label:<38} n={n:>5} {state:<12} one alone {human(alone):>9} "
                f"{threads} at once {human(wall):>9} ({runs} runs)",
                flush=True,
            )
    numba.set_num_threads(1)
    return rows


def predict(done: list[tuple[int, float]], n: int) -> float:
    """Seconds at ``n`` from the measured sizes below: the last growth per doubling (at
    least 8, the growth of an n**3 cost) carried on."""
    n1, t1 = done[-1]
    growth = 8.0
    if len(done) >= 2:
        n0, t0 = done[-2]
        if t0 > 1e-4:  # smaller times are too noisy to extrapolate
            growth = max(growth, (t1 / t0) ** (1 / math.log2(n1 / n0)))
    return t1 * growth ** math.log2(n / n1)


def price(results: dict) -> dict:
    """Add the current model's estimate and estimate / measured to every row."""
    makers: dict[int, dict] = {}
    specs: dict[tuple[int, str], mipt.MonitoredCircuit] = {}
    for b in results["bases"]:
        spec = specs.setdefault((b["n"], b["state"]), state_spec(b["n"], b["state"]))
        b["estimate_seconds"] = api._trajectory_us(spec) * 1e-6
        b["ratio"] = b["estimate_seconds"] / b["seconds"]
    for r in results["rows"]:
        if "measured_seconds" not in r:
            continue
        n = r["n"]
        spec = specs.setdefault((n, r["state"]), state_spec(n, r["state"]))
        make = makers.setdefault(n, probe_makers(n))[r["label"]]
        r["estimate_seconds"] = estimate_seconds(spec, make)
        r["ratio"] = r["estimate_seconds"] / max(r["measured_seconds"], 1e-9)
    measured = [r for r in results["rows"] if "measured_seconds" in r and not r["unresolved"]]
    by_label: dict[str, list[dict]] = {}
    for r in measured:
        by_label.setdefault(r["label"], []).append(r)
    results["smallest_ratio"] = {
        label: min(rs, key=lambda r: r["ratio"]) | {"rows": len(rs)}
        for label, rs in by_label.items()
    }
    results["below_margin"] = [r for r in measured if r["ratio"] < MARGIN]
    for r in results.get("all_threads", []):
        spec = all_threads_spec(
            r["n"], r["state"], probe_makers(r["n"])[r["label"]], r["evaluations"]
        )
        one = api._trajectory_us(spec) * 1e-6
        # estimate_ensemble's kernel term for one trajectory per thread
        r["estimate_seconds"] = r["threads"] * one / api._parallel_threads(r["threads"])
        r["ratio"] = r["estimate_seconds"] / r["seconds"]
        r["alone_ratio"] = one / r["alone_seconds"]
        r["speed_under_load"] = r["alone_seconds"] / r["seconds"]
    return results


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def human(x: float | None) -> str:
    """Seconds in a readable unit."""
    if x is None:
        return "-"
    a = abs(x)
    if a < 1e-6:
        return f"{x * 1e9:.3g} ns"
    if a < 1e-3:
        return f"{x * 1e6:.3g} µs"
    if a < 1:
        return f"{x * 1e3:.3g} ms"
    if a < 600:
        return f"{x:.3g} s"
    return f"{x / 60:.3g} min"


def render_markdown(r: dict) -> str:
    env, cfg = r["environment"], r["settings"]
    lines = [
        "# Probe cost against the runtime estimate",
        "",
        f"Recorded {env['date']} on {env['cpu_model']} ({env['physical_cores']} cores, "
        f"{env['logical_processors']} logical processors), Python {env['python']}, "
        f"Numba {env['numba']}, sparsegf2 {env['sparsegf2']}. Every time is **one thread** "
        "(except in the all-threads section), after a warm-up; single-thread times are "
        "inflated directly by any background load."
        + (f" {env['note']}" if env.get("note") else ""),
        "",
        f"Generated by `python benchmarks/probe_cost.py{' --quick' if r['quick'] else ''}`"
        + (f"; total {r['total_seconds'] / 60:.1f} min" if "total_seconds" in r else "")
        + f". Estimates are `api._trajectory_us` of sparsegf2 {r['priced_with']}"
        + (" (re-priced from the measured rows)" if r.get("repriced") else "")
        + ".",
        "",
        "States (one trajectory, probes at the final layer): **single**, reference `single` "
        "after 2 n ceil(log2 n) random all-to-all gates (k = 1, erasures fail at about n / 2 "
        "qubits: the worst case of the erasure scans); **purification**, the same scramble "
        "with every system qubit paired (k = n, a volume-law state on 2n qubits: the worst "
        "case of the entanglement eliminations); **monitored**, the purification state after "
        "min(n / 2, 256) further layers on a Watts-Strogatz graph (k = 4, beta = 0.1, "
        "`random_edges`) at measurement rate 0.05. A measurement is the trajectory with R copies "
        "of the probe at its final layer minus the trajectory without probes, divided by R; "
        "medians over repeats. Where the copies take under 5 % of the trajectory without probes "
        "the row is timing noise, marked unresolved and left out of the smallest ratios. Rows "
        f"with estimate / measured below {MARGIN:g} are marked **!**.",
        "",
        "## States",
        "",
        "| n | state | run without probes | its estimate | estimate / measured | k "
        "| half-cut entropy | mean first-failure size |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for b in r["bases"]:
        lines.append(
            f"| {b['n']} | {b['state']} | {human(b['seconds'])} | {human(b['estimate_seconds'])} "
            f"| {b['ratio']:.2f} | {b['k']} | {b['half_cut_entropy']} "
            f"| {b['mean_first_failure']:.1f} |"
        )
    lines += [
        "",
        "## Smallest estimate / measured per probe",
        "",
        "| probe | smallest ratio | at n | state | rows |",
        "|---|---:|---:|---|---:|",
    ]
    for label, m in r["smallest_ratio"].items():
        flag = " **!**" if m["ratio"] < MARGIN else ""
        lines.append(
            f"| {label} | {m['ratio']:.2f}{flag} | {m['n']} | {m['state']} | {m['rows']} |"
        )
    largest = max(cfg["sizes"])
    lines += [
        "",
        f"## n = {largest}",
        "",
        "| probe | state | measured | estimate | estimate / measured |",
        "|---|---|---:|---:|---:|",
    ]
    for row in r["rows"]:
        if row["n"] == largest:
            lines.append(row_line(row, with_n=False))
    if r.get("all_threads"):
        lines += [
            "",
            "## All threads",
            "",
            "The same states, the probe evaluated at every layer of a stretch without gates or "
            "measurements (9 evaluations at n <= 512, 3 above): one trajectory alone on one "
            "thread, then one trajectory per thread on every thread. Each busy thread runs at "
            "the speed under load (one alone / all at once) of a thread alone: all-core clocks, "
            "SMT siblings and the shared memory bandwidth slow it. `estimate_ensemble` prices "
            "the busy threads with `api._parallel_threads`; the estimate here is its kernel term "
            "(no call overhead).",
            "",
            "| probe | n | state | threads | one alone | its estimate / measured | all at once "
            "| speed under load | estimate | estimate / measured |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for a in r["all_threads"]:
            flag = " **!**" if a["ratio"] < MARGIN else ""
            lines.append(
                f"| {a['label']} | {a['n']} | {a['state']} | {a['threads']} "
                f"| {human(a['alone_seconds'])} | {a['alone_ratio']:.2f} | {human(a['seconds'])} "
                f"| {a['speed_under_load']:.2f} | {human(a['estimate_seconds'])} "
                f"| {a['ratio']:.2f}{flag} |"
            )
    lines += [
        "",
        "## Every measurement",
        "",
        "| probe | n | state | measured | R | runs | estimate | estimate / measured |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in r["rows"]:
        lines.append(row_line(row, with_n=True))
    lines.append("")
    return "\n".join(lines)


def row_line(row: dict, with_n: bool) -> str:
    n = f" {row['n']} |" if with_n else ""
    if "measured_seconds" not in row:
        extra = " - | - |" if with_n else ""
        return (
            f"| {row['label']} |{n} {row['state']} | not run (predicted "
            f"{human(row['skipped'])}) |{extra} - | - |"
        )
    flag = " **!**" if row["ratio"] < MARGIN else ""
    if row["unresolved"]:
        flag = " (unresolved)"
    middle = f" {row['copies']} | {row['runs']} |" if with_n else ""
    return (
        f"| {row['label']} |{n} {row['state']} | {human(row['measured_seconds'])} |{middle} "
        f"{human(row['estimate_seconds'])} | {row['ratio']:.2f}{flag} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(HERE / "results"), help="output directory")
    ap.add_argument("--quick", action="store_true", help="tiny sizes, for a smoke test")
    ap.add_argument("--sizes", type=int, nargs="*", default=None, help="system sizes n")
    ap.add_argument("--states", nargs="*", default=None, choices=STATES, help="states to run")
    ap.add_argument("--labels", nargs="*", default=None, help="probe labels to run (default all)")
    ap.add_argument("--max-eval-seconds", type=float, default=150.0)
    ap.add_argument("--note", default="", help="a sentence about the machine's state")
    ap.add_argument(
        "--from-json",
        default=None,
        help="measure nothing: price the measured rows of this results file with the current model",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.from_json:
        results = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        results["priced_with"] = sg.__version__
        results["repriced"] = True
        results = price(results)
        write(out, results)
        print(f"re-priced {args.from_json}; wrote {out}", flush=True)
        return
    labels = list(probe_makers(64))
    cfg = {
        "sizes": list(args.sizes or ((8, 16) if args.quick else SIZES)),
        "states": list(args.states or STATES),
        "labels": list(args.labels or labels),
        "max_eval_seconds": args.max_eval_seconds,
        "min_runs": 3,
        "max_runs": 25,
        "min_total": 0.01 if args.quick else 0.2,
        "once_above": 5.0,
        "max_copies": 4096,
        "resolution": 0.05,
        "all_threads": min(4, numba.config.NUMBA_NUM_THREADS)
        if args.quick
        else numba.config.NUMBA_NUM_THREADS,
        "all_threads_sizes": [16] if args.quick else list(ALL_THREADS_SIZES),
    }
    unknown = sorted(set(cfg["labels"]) - set(labels))
    if unknown:
        raise SystemExit(f"unknown labels {unknown}; choose from {labels}")
    start = time.perf_counter()
    numba.set_num_threads(1)
    env = {
        "date": time.strftime("%Y-%m-%d"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        **cpu_info(),
        "logical_processors": os.cpu_count(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "sparsegf2": sg.__version__,
        "threads": 1,
        "note": args.note,
    }
    print(f"{env['cpu_model']}; one thread; sizes {cfg['sizes']}", flush=True)
    rows, bases = measure(cfg)
    results = {
        "environment": env,
        "quick": args.quick,
        "seed": SEED,
        "settings": cfg,
        "priced_with": sg.__version__,
        "bases": bases,
        "rows": rows,
        "all_threads": measure_all_threads(cfg),
    }
    results["total_seconds"] = time.perf_counter() - start
    results = price(results)
    write(out, results)
    print(f"total {results['total_seconds'] / 60:.1f} min; wrote {out}", flush=True)


def write(out: Path, results: dict) -> None:
    (out / "probe_cost.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "probe_cost.md").write_text(render_markdown(results), encoding="utf-8")
    for r in results["below_margin"]:
        print(
            f"below {MARGIN:g}: {r['label']} n={r['n']} {r['state']} ratio {r['ratio']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
