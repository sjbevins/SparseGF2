"""The cost of exhaustive erasure enumeration and of sampled orderings, by n and budget.

P_C(m) is exact for m <= w = m*(n) (``uncorrectable_region_counts``: every m-subset is counted)
and sampled above (``erasure_first_failure``: M uniformly random orderings). This measures what
each costs per code, on one thread, so that w and M can be chosen from times, not region counts.

**Codes** come from real trajectories: for n in {32, 64, 128, 256, 512} and p in {0.05, 0.1},
Watts-Strogatz graphs (k = 4, beta = 0.1, one graph per trajectory), purification picture,
``random_edges``, 4n layers, ``code_snapshot`` at the final layer. The first 8 trajectories with
k > 0 give the codes; their k and distance bounds are recorded.

**Measured**, one thread, per code:

* the exhaustive counts for w = 1, 2, ..., by two paths:

  - the library, ``StabilizerCode.uncorrectable_counts(w)``;
  - the engine: the trajectory rerun with R copies of the counts probe at its final layer, minus
    the same run without probes, divided by R. R is chosen so that the probes take about as long
    as the evolution, which keeps the difference well above the timing noise (R = 1 for slow
    levels). The engine's counts must equal the library's, or the script aborts.

  w rises until one evaluation took over ``--max-eval-seconds`` (10 s). A level is not run when
  N_exact(n, w) = sum_{m<=w} C(n, m) exceeds ``--max-regions`` (1e9), or, from w = 4 on, when the
  walk model predicts more than ``--max-predicted`` seconds for one evaluation. Levels slower
  than a few seconds are measured on the first codes only, until ``--level-seconds`` are spent
  (at least ``--min-codes`` codes);
* ``code_columns`` alone, the elimination every evaluation of either probe pays;
* first-failure orderings, M in {100, 512, 2048}, by the same two paths
  (``StabilizerCode.first_failure_sizes`` and ``erasure_first_failure(samples=M)``).

**Reported:** per (n, p, w) the median and max seconds per code, N_exact and the seconds per
region; per budget B in {1e4, 5e4, 1e5, 1e6, 1e7} the cutoff m*(n) and the expected seconds per
code (engine path, what an ensemble pays; library alongside); an example study total, which
divides by the effective threads measured in ``results/thread_scaling.json``. Levels that were
not measured are extrapolated with the walk model of docs/observables/random-region-erasure.md
§3.3 and labeled as extrapolations; a per-code model check shows how far to trust them.

    python benchmarks/erasure_cost.py                      # about 20 minutes, idle machine
    python benchmarks/erasure_cost.py --quick --out <dir>  # smoke test, a few seconds
    python benchmarks/erasure_cost.py --from-json results/erasure_cost.json  # re-render only

``--from-json`` measures nothing: it recomputes the derived tables (model check, budgets, study)
from the measured cases of a results file and rewrites both outputs.

Writes ``erasure_cost.json`` and ``erasure_cost.md`` into ``--out`` (default
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
from dataclasses import replace
from pathlib import Path

import numba
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from thread_scaling import cpu_info  # noqa: E402

import sparsegf2 as sg  # noqa: E402
from sparsegf2 import analysis, graphs, mipt  # noqa: E402
from sparsegf2._kernel.entropy import code_columns  # noqa: E402

SEED = 27
SIZES = (32, 64, 128, 256, 512)
RATES = (0.05, 0.1)
ORDERINGS = (100, 512, 2048)
BUDGETS = (10_000, 50_000, 100_000, 1_000_000, 10_000_000)
# The example study: a sweep over n at both rates, counts at the cutoff of one budget and
# M orderings, both at every checkpoint.
STUDY = {
    "sizes": [64, 128, 256, 512],
    "rates": list(RATES),
    "trajectories": 1000,
    "checkpoints": 8,
    "budget": 50_000,
    "orderings": 512,
}


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def seconds(fn, cfg: dict, min_total: float | None = None):
    """Median wall time of ``fn()`` and its last value.

    Repeated until ``min_total`` seconds are spent (at least ``min_runs``, at most ``max_runs``
    runs); a call of at least ``once_above`` seconds is not repeated.
    """
    floor = cfg["min_total"] if min_total is None else min_total
    times, value = [], None
    while True:
        t0 = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - t0)
        if times[0] >= cfg["once_above"] or len(times) >= cfg["max_runs"]:
            break
        if len(times) >= cfg["min_runs"] and sum(times) >= floor:
            break
    return statistics.median(times), value


def engine_run(spec: mipt.MonitoredCircuit, trajectory: int, seed: int) -> mipt.EnsembleResult:
    """Trajectory ``trajectory`` of ``spec.run(seed=seed)``, on one thread."""
    return spec.run(1, seed=seed, first_trajectory=trajectory, num_threads=1)


def engine_seconds(case: dict, code: dict, make, guess: float, cfg: dict):
    """Engine seconds per evaluation of the probe ``make(name)`` on ``code``'s trajectory.

    The trajectory is rerun with R copies of the probe at its final layer; the time without
    probes (``code["evolve"]``) is subtracted and the rest divided by R. Returns
    ``(seconds, R, result)``.
    """
    base = code["evolve"]
    copies = max(1, min(cfg["max_copies"], math.ceil(max(base, 1e-4) / max(guess, 1e-7))))
    variant = replace(
        case["spec"],
        probes=tuple(make(f"p{i}") for i in range(copies)),
        checkpoints="final",
    )
    t, seed = code["trajectory"], case["seed"]
    if guess * copies < cfg["once_above"]:
        engine_run(variant, t, seed)  # warm-up (fills the spec's per-instance caches)
    total, res = seconds(lambda: engine_run(variant, t, seed), cfg)
    return (total - base) / copies, copies, res


# ---------------------------------------------------------------------------
# codes
# ---------------------------------------------------------------------------


def make_case(n: int, p: float, cfg: dict) -> dict:
    """The spec of one (n, p) and its first ``cfg["codes"]`` codes with k > 0."""
    pool = 4 * cfg["codes"]
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=[graphs.watts_strogatz(n, 4, 0.1, seed=(SEED, n, i)) for i in range(pool)],
        layers=4 * n,
        measurement_rate=p,
        schedule="random_edges",
        reference="purification",
        checkpoints="final",
        probes=[mipt.code_dimension(), mipt.code_snapshot()],
    )
    seed = SEED * 1_000_000 + n * 1000 + round(1000 * p)
    res = spec.run(seed=seed)  # every graph once, on all threads
    k = res.values["code_dimension"][:, -1]
    rows = np.flatnonzero(k > 0)[: cfg["codes"]]
    if rows.size == 0:
        raise SystemExit(f"n={n} p={p}: no trajectory kept a code (k > 0); nothing to time")
    codes = [{"trajectory": int(t), "k": int(k[t]), "code": res.code(int(t), 0)} for t in rows]
    return {
        "n": n,
        "p": p,
        "seed": seed,
        "spec": spec,
        "trajectories_run": int(k.size),
        "fraction_with_code": float(np.mean(k > 0)),
        "codes": codes,
    }


def warm_up() -> None:
    """Compile (or load) and call every kernel once, so no timing includes a first call."""
    case = make_case(12, 0.05, {"codes": 1})
    code = case["codes"][0]
    c = code["code"]
    code_columns(c._tab, c._ws, c._system, c._marked)
    c.uncorrectable_counts(3)
    c.first_failure_sizes(16, seed=0)
    probes = [
        mipt.uncorrectable_region_counts(max_size=2, name="c", checkpoints="final"),
        mipt.erasure_first_failure(samples=16, name="f", checkpoints="final"),
    ]
    engine_run(replace(case["spec"], probes=(), checkpoints="final"), code["trajectory"], 1)
    engine_run(replace(case["spec"], probes=probes, checkpoints="final"), code["trajectory"], 1)


# ---------------------------------------------------------------------------
# the walk model (docs/observables/random-region-erasure.md §3.3), for extrapolation only
# ---------------------------------------------------------------------------


def walk_growth(n: int, w: int, k: int) -> float:
    """Model ratio T_walk(n, w) / T_walk(n, w - 1), for w >= 2.

    The walk costs about C(n, w-1) (4^(w-1) ceil(r/64) + n) word operations, r = n + k.
    """
    words = math.ceil((n + k) / 64)
    prefixes = math.comb(n, w - 1) / math.comb(n, w - 2)
    return prefixes * (4 ** (w - 1) * words + n) / (4 ** (w - 2) * words + n)


def extrapolate(t_prev: float, t_cols: float, n: int, w: int, k: int) -> float:
    """Model time at w from the time at w - 1: the elimination plus the walk scaled by the model."""
    return t_cols + max(t_prev - t_cols, 0.0) * walk_growth(n, w, k)


def stats(values: list[float]) -> dict:
    return {"median": statistics.median(values), "max": max(values), "min": min(values)}


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------


def measure_level(case: dict, w: int, cfg: dict) -> dict:
    """Both paths of the exhaustive counts at cutoff w, code by code (fewer codes if slow)."""
    n = case["n"]
    rows, spent = [], 0.0
    for code in case["codes"]:
        start = time.perf_counter()
        c = code["code"]
        lib, counts = seconds(lambda c=c: c.uncorrectable_counts(w), cfg)
        eng, copies, res = engine_seconds(
            case,
            code,
            lambda name: mipt.uncorrectable_region_counts(
                max_size=w, name=name, checkpoints="final"
            ),
            lib,
            cfg,
        )
        engine_counts = res.values["p0"][0, -1]
        if not np.array_equal(engine_counts, np.asarray(counts, dtype=np.float64)):
            raise SystemExit(
                f"n={n} p={case['p']} trajectory {code['trajectory']} w={w}: engine counts "
                f"{engine_counts.tolist()} differ from the library's {list(counts)}"
            )
        code["counts"] = [int(x) for x in counts]
        rows.append(
            {
                "trajectory": code["trajectory"],
                "library_seconds": lib,
                "engine_seconds": eng,
                "engine_copies": copies,
                "counts": code["counts"],
            }
        )
        spent += time.perf_counter() - start
        if len(rows) >= cfg["min_codes"] and spent >= cfg["level_seconds"]:
            break
    n_exact = analysis.exhaustive_region_count(n, w)
    lib_s = stats([r["library_seconds"] for r in rows])
    eng_s = stats([r["engine_seconds"] for r in rows])
    return {
        "w": w,
        "n_exact": n_exact,
        "codes_measured": len(rows),
        "library": lib_s,
        "engine": eng_s,
        "engine_over_library": statistics.median(
            r["engine_seconds"] / r["library_seconds"] for r in rows
        ),
        "seconds_per_region": lib_s["median"] / n_exact,
        "slowest": max(max(r["library_seconds"], r["engine_seconds"]) for r in rows),
        "spent_seconds": spent,
        "per_code": rows,
    }


def measure_case(case: dict, cfg: dict) -> dict:
    n, p = case["n"], case["p"]
    bare = replace(case["spec"], probes=(), checkpoints="final")
    for code in case["codes"]:
        c, t = code["code"], code["trajectory"]
        engine_run(bare, t, case["seed"])  # warm-up
        code["evolve"], _ = seconds(
            lambda t=t: engine_run(bare, t, case["seed"]), cfg, min_total=cfg["evolve_total"]
        )
        code["code_columns"], _ = seconds(
            lambda c=c: code_columns(c._tab, c._ws, c._system, c._marked), cfg
        )
    cols = stats([code["code_columns"] for code in case["codes"]])
    k_med = int(statistics.median(code["k"] for code in case["codes"]))

    # first-failure orderings
    ordering_rows = []
    for m in cfg["orderings"]:
        per_code = []
        for code in case["codes"]:
            c = code["code"]
            lib, first = seconds(lambda c=c, m=m: c.first_failure_sizes(m, seed=SEED), cfg)
            eng, copies, res = engine_seconds(
                case,
                code,
                lambda name, m=m: mipt.erasure_first_failure(
                    samples=m, name=name, checkpoints="final"
                ),
                lib,
                cfg,
            )
            sample = res.values["p0"][0, -1]
            if sample.shape != (m,) or sample.min() < 1 or sample.max() > n:
                raise SystemExit(f"n={n} p={p}: engine first-failure sizes out of range")
            code[f"first_{m}"] = first
            per_code.append(
                {
                    "trajectory": code["trajectory"],
                    "library_seconds": lib,
                    "engine_seconds": eng,
                    "engine_copies": copies,
                }
            )
        ordering_rows.append(
            {
                "orderings": m,
                "library": stats([r["library_seconds"] for r in per_code]),
                "engine": stats([r["engine_seconds"] for r in per_code]),
                "per_code": per_code,
            }
        )
    lo, hi = cfg["orderings"][0], cfg["orderings"][-1]
    by_m = {r["orderings"]: r for r in ordering_rows}
    marginal = [
        (a["library_seconds"] - b["library_seconds"]) / (hi - lo)
        for a, b in zip(by_m[hi]["per_code"], by_m[lo]["per_code"], strict=True)
    ]

    # exhaustive counts, w = 1, 2, ...
    levels: list[dict] = []
    w, stop = 0, None
    while stop is None:
        w += 1
        n_exact = analysis.exhaustive_region_count(n, w) if w <= n else None
        if w > n:
            stop = f"w = n = {n} reached"
        elif w > cfg["max_w"]:
            stop = f"w capped at {cfg['max_w']} (--max-w)"
        elif n_exact > cfg["max_regions"]:
            stop = f"N_exact({n}, {w}) = {n_exact:.3g} > {cfg['max_regions']:.0e}: not run"
        elif w >= 4 and levels:
            prev = levels[-1]
            predicted = extrapolate(prev["library"]["max"], cols["max"], n, w, k_med)
            if predicted > cfg["max_predicted"]:
                stop = (
                    f"w = {w} predicted at {predicted:.0f} s per evaluation "
                    f"> {cfg['max_predicted']:g} s: not run"
                )
        if stop is not None:
            break
        level = measure_level(case, w, cfg)
        levels.append(level)
        print(
            f"n={n:>3} p={p:<4} w={w} N_exact={n_exact:>11,} codes={level['codes_measured']} "
            f"library median {level['library']['median']:.4g} s max {level['library']['max']:.4g} s"
            f" | engine median {level['engine']['median']:.4g} s",
            flush=True,
        )
        if level["slowest"] > cfg["max_eval_seconds"]:
            stop = (
                f"an evaluation at w = {w} took {level['slowest']:.1f} s "
                f"> {cfg['max_eval_seconds']:g} s"
            )
            w += 1
    next_level = None
    if levels and w <= n:
        last = levels[-1]
        next_level = {
            "w": w,
            "n_exact": analysis.exhaustive_region_count(n, w),
            "extrapolated_median_seconds": extrapolate(
                last["library"]["median"], cols["median"], n, w, k_med
            ),
            "extrapolated_max_seconds": extrapolate(
                last["library"]["max"], cols["max"], n, w, k_med
            ),
            "basis": f"walk model (§3.3) from the measured w = {last['w']}; not measured",
        }

    # per code: k and distance bounds, from the largest w measured on it and the most orderings
    codes_out = []
    for code in case["codes"]:
        first = code[f"first_{hi}"]
        counts = np.asarray(code.get("counts", []), dtype=np.int64)
        exact = counts if counts.size else None
        low, high = analysis.distance_bounds(first, n, exact_counts=exact)
        codes_out.append(
            {
                "trajectory": code["trajectory"],
                "k": code["k"],
                "distance_low": int(low),
                "distance_high": int(high),
                "counts_up_to": int(counts.size),
                "evolve_seconds": code["evolve"],
                "code_columns_seconds": code["code_columns"],
            }
        )
    return {
        "n": n,
        "p": p,
        "seed": case["seed"],
        "layers": 4 * n,
        "trajectories_run": case["trajectories_run"],
        "fraction_with_code": case["fraction_with_code"],
        "codes": codes_out,
        "evolve": stats([code["evolve"] for code in case["codes"]]),
        "code_columns": cols,
        "orderings": ordering_rows,
        "seconds_per_ordering": stats(marginal),
        "levels": levels,
        "stop": stop,
        "next_level": next_level,
    }


# ---------------------------------------------------------------------------
# derived: the model check, budgets and the study example (``--from-json`` redoes these)
# ---------------------------------------------------------------------------


def model_check(case: dict) -> list[dict]:
    """Measured growth of the walk from w - 1 to w against the model, per code, for w >= 3.

    Growth of a code = (T(w) - T_cols) / (T(w - 1) - T_cols) on the library path, for the codes
    measured at both levels; the model is :func:`walk_growth` with the code's own k.
    """
    n = case["n"]
    codes = {c["trajectory"]: c for c in case["codes"]}
    out = []
    for a, b in zip(case["levels"], case["levels"][1:], strict=False):
        if b["w"] < 3:
            continue
        before = {r["trajectory"]: r["library_seconds"] for r in a["per_code"]}
        ratios = []
        for r in b["per_code"]:
            t = r["trajectory"]
            cols = codes[t]["code_columns_seconds"]
            if t in before and before[t] > cols:
                measured = (r["library_seconds"] - cols) / (before[t] - cols)
                ratios.append(measured / walk_growth(n, b["w"], codes[t]["k"]))
        if ratios:
            out.append({"w": b["w"], "codes": len(ratios), **stats(ratios)})
    return out


def level_seconds(case: dict, w: int, path: str = "engine") -> tuple[float, str]:
    """Median seconds per code at cutoff w by ``path``: measured, or extrapolated level by level.

    The extrapolation keeps the path's fixed cost (the library's ``code_columns``; the engine's
    w = 1 time, which is almost all elimination) and scales the rest with the walk model.
    """
    levels = {lv["w"]: lv for lv in case["levels"]}
    if w in levels:
        return levels[w][path]["median"], "measured"
    last = max(levels)
    t = levels[last][path]["median"]
    fixed = case["code_columns"]["median"] if path == "library" else levels[1][path]["median"]
    k_med = int(statistics.median(c["k"] for c in case["codes"]))
    for v in range(last + 1, w + 1):
        t = extrapolate(t, fixed, case["n"], v, k_med)
    return t, f"extrapolated from w = {last} (walk model)"


def budget_table(cases: list[dict], budgets) -> list[dict]:
    """m*(n) per budget and the seconds per code there, the larger of the rates' medians."""
    out = []
    sizes = sorted({c["n"] for c in cases})
    for b in budgets:
        for n in sizes:
            w = analysis.exhaustive_cutoff(n, b)
            row = {"budget": b, "n": n, "m_star": w}
            if w == 0:
                row.update(
                    n_exact=0,
                    seconds_per_code=None,
                    library_seconds_per_code=None,
                    source="B < n: no exhaustive size",
                )
            else:
                mine = [c for c in cases if c["n"] == n]
                engine = max(level_seconds(c, w, "engine") for c in mine)
                library = max(level_seconds(c, w, "library") for c in mine)
                row.update(
                    n_exact=analysis.exhaustive_region_count(n, w),
                    seconds_per_code=engine[0],
                    library_seconds_per_code=library[0],
                    source=engine[1],
                )
            out.append(row)
    return out


def effective_threads() -> dict:
    """Measured speedups at the most threads, for the ensemble workloads W1 and W2 (Task 25)."""
    path = HERE / "results" / "thread_scaling.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = [w["rows"][-1] for w in data["workloads"] if w["name"] in ("W1", "W2")]
        return {
            "source": "benchmarks/results/thread_scaling.json (W1, W2)",
            "threads": rows[0]["threads"],
            "low": min(r["speedup"] for r in rows),
            "high": max(r["speedup"] for r in rows),
        }
    except (OSError, KeyError, IndexError, ValueError):
        return {
            "source": "thread_scaling.json missing: one thread assumed",
            "threads": 1,
            "low": 1.0,
            "high": 1.0,
        }


def study_example(cases: list[dict], study: dict) -> dict:
    eff = effective_threads()
    rows, cpu = [], 0.0
    t, c = study["trajectories"], study["checkpoints"]
    for case in cases:
        if case["n"] not in study["sizes"] or case["p"] not in study["rates"]:
            continue
        w = analysis.exhaustive_cutoff(case["n"], study["budget"])
        counts, source = level_seconds(case, w, "engine")
        orders = {r["orderings"]: r for r in case["orderings"]}
        if study["orderings"] not in orders:
            continue
        first = orders[study["orderings"]]["engine"]["median"]
        evolve = case["evolve"]["median"]
        seconds_ = (counts + first) * t * c + evolve * t
        cpu += seconds_
        rows.append(
            {
                "n": case["n"],
                "p": case["p"],
                "m_star": w,
                "counts_seconds": counts,
                "counts_source": source,
                "first_failure_seconds": first,
                "evolve_seconds": evolve,
                "cpu_seconds": seconds_,
            }
        )
    return {
        **study,
        "effective_threads": eff,
        "rows": rows,
        "cpu_seconds": cpu,
        "wall_seconds_low": cpu / eff["high"],
        "wall_seconds_high": cpu / eff["low"],
    }


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
    if a < 36_000:
        return f"{x / 60:.3g} min"
    return f"{x / 3600:.3g} h"


def render_markdown(r: dict) -> str:
    env, cfg = r["environment"], r["settings"]
    lines = [
        "# Erasure enumeration and sampling cost",
        "",
        f"Recorded {env['date']} on {env['cpu_model']} ({env['physical_cores']} cores, "
        f"{env['logical_processors']} logical processors), Python {env['python']}, "
        f"Numba {env['numba']}, sparsegf2 {env['sparsegf2']}. Every time is **one thread**, "
        "after a warm-up; single-thread times are inflated directly by any background load."
        + (f" {env['note']}" if env.get("note") else ""),
        "",
        f"Codes: Watts-Strogatz graphs (k = 4, beta = 0.1, one graph per trajectory), "
        "purification picture, `random_edges`, 4n layers, `code_snapshot` at the final layer; "
        f"the first {cfg['codes']} trajectories with k > 0 per (n, p). "
        f"Generated by `python benchmarks/erasure_cost.py{' --quick' if r['quick'] else ''}`; "
        f"total {r['total_seconds'] / 60:.1f} min.",
        "",
        "* **library**: `StabilizerCode.uncorrectable_counts(w)` or "
        "`first_failure_sizes(M)` on the saved code;",
        "* **engine**: the trajectory rerun with R copies of the probe at its final layer, minus "
        "the same run without probes, divided by R (R makes the probes take about as long as "
        "the evolution; R = 1 for slow levels). The engine's counts equal the library's for "
        "every code (checked).",
        "",
        "Medians and maxima are over codes. A time includes the fixed `code_columns` "
        "elimination that every evaluation pays.",
        "",
        *paths_paragraph(r),
        "## Codes",
        "",
        "| n | p | layers | trajectories with k > 0 | k (min / median / max) "
        "| distance d | evolution per trajectory | `code_columns` median (max) |",
        "|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for c in r["cases"]:
        ks = [x["k"] for x in c["codes"]]
        dist = ", ".join(
            str(x["distance_low"])
            if x["distance_low"] == x["distance_high"]
            else f"[{x['distance_low']}, {x['distance_high']}]"
            for x in c["codes"]
        )
        lines.append(
            f"| {c['n']} | {c['p']} | {c['layers']} | {c['fraction_with_code']:.0%} of "
            f"{c['trajectories_run']} | {min(ks)} / {statistics.median(ks):g} / {max(ks)} | "
            f"{dist} | {human(c['evolve']['median'])} | "
            f"{human(c['code_columns']['median'])} ({human(c['code_columns']['max'])}) |"
        )
    lines += [
        "",
        "Distance d per code: exact where an exhaustive count found an uncorrectable region; "
        "otherwise [low, high], with low = the largest w counted on that code plus one and high "
        f"= the smallest of {cfg['orderings'][-1]} sampled first-failure sizes.",
        "",
        "## Exhaustive counts, per (n, p, w)",
        "",
        "| n | p | w | N_exact(n, w) | codes | library median | library max | engine median "
        "| engine max | engine / library | seconds per region |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in r["cases"]:
        for lv in c["levels"]:
            lines.append(
                f"| {c['n']} | {c['p']} | {lv['w']} | {lv['n_exact']:,} | {lv['codes_measured']} | "
                f"{human(lv['library']['median'])} | {human(lv['library']['max'])} | "
                f"{human(lv['engine']['median'])} | {human(lv['engine']['max'])} | "
                f"{lv['engine_over_library']:.2f} | {human(lv['seconds_per_region'])} |"
            )
    lines += [
        "",
        "Seconds per region = library median / N_exact(n, w), elimination included (at w = 1 "
        "the elimination is almost all of it). Where fewer codes than the rest were measured, "
        f"the level was slow: codes were added until {cfg['level_seconds']:g} s were spent, "
        f"at least {cfg['min_codes']}.",
        "",
        "Why w stopped rising, and the next level **extrapolated, not measured** (walk model of "
        "docs/observables/random-region-erasure.md §3.3, scaled from the measured level below):",
        "",
        "| n | p | stopped because | next w | N_exact | extrapolated median (max) |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for c in r["cases"]:
        nx = c["next_level"]
        tail = (
            f"{nx['w']} | {nx['n_exact']:,} | {human(nx['extrapolated_median_seconds'])} "
            f"({human(nx['extrapolated_max_seconds'])})"
            if nx
            else "- | - | -"
        )
        lines.append(f"| {c['n']} | {c['p']} | {c['stop']} | {tail} |")
    lines += [
        "",
        "Model check: the measured growth of the walk from w - 1 to w, (T(w) - T_cols) / "
        "(T(w - 1) - T_cols) per code (library path), divided by the model's growth for that "
        "code's k. An extrapolation is accurate to about the factors seen at the same n.",
        "",
        "| n | p | w | codes | measured / model: median | min | max |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in r["cases"]:
        for g in c["model_check"]:
            lines.append(
                f"| {c['n']} | {c['p']} | {g['w']} | {g['codes']} | {g['median']:.2f} | "
                f"{g['min']:.2f} | {g['max']:.2f} |"
            )
    lines += [
        "",
        "## First-failure orderings",
        "",
        "| n | p | M | library median | library max | engine median | engine max |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in r["cases"]:
        for o in c["orderings"]:
            lines.append(
                f"| {c['n']} | {c['p']} | {o['orderings']} | {human(o['library']['median'])} | "
                f"{human(o['library']['max'])} | {human(o['engine']['median'])} | "
                f"{human(o['engine']['max'])} |"
            )
    lo, hi = cfg["orderings"][0], cfg["orderings"][-1]
    lines += [
        "",
        f"Time per ordering (library, (T(M = {hi}) - T(M = {lo})) / {hi - lo} per code), "
        "beyond the elimination every evaluation pays:",
        "",
        "| n | p | per ordering, median | max |",
        "|---:|---:|---:|---:|",
    ]
    for c in r["cases"]:
        s = c["seconds_per_ordering"]
        lines.append(f"| {c['n']} | {c['p']} | {human(s['median'])} | {human(s['max'])} |")
    lines += [
        "",
        "## Budgets",
        "",
        "m*(n) = max{w : N_exact(n, w) <= B} (`analysis.exhaustive_cutoff`). Expected seconds per "
        "code: the median at that w, the larger of the two rates, for the engine probe (what an "
        "ensemble pays per evaluation) and for `StabilizerCode.uncorrectable_counts` on a saved "
        "code.",
        "",
        "| B | n | m*(n) | N_exact | seconds per code, engine | library | source |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for b in r["budgets"]:
        lines.append(
            f"| {b['budget']:,} | {b['n']} | {b['m_star']} | {b['n_exact']:,} | "
            f"{human(b['seconds_per_code'])} | {human(b['library_seconds_per_code'])} | "
            f"{b['source']} |"
        )
    s = r["study"]
    eff = s["effective_threads"]
    lines += [
        "",
        "## Expected total for a study",
        "",
        "total CPU time = sum over (n, p) of (counts per code + orderings per code) x "
        "trajectories x checkpoints + evolution per trajectory x trajectories; wall time = CPU "
        f"time / effective threads. Effective threads: {eff['low']:.1f} to {eff['high']:.1f} at "
        f"{eff['threads']} threads ({eff['source']}, the measured speedups of the ensemble "
        "workloads).",
        "",
        f"Example: n in {s['sizes']}, p in {s['rates']}, {s['trajectories']:,} trajectories per "
        f"(n, p), {s['checkpoints']} checkpoints, counts at the cutoff of B = {s['budget']:,} and "
        f"M = {s['orderings']} orderings at every checkpoint, 4n layers (evolution as measured "
        "here). Counts and orderings per code are engine-path medians.",
        "",
        "| n | p | m*(n) | counts per code | orderings per code | evolution per trajectory "
        "| CPU time |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in s["rows"]:
        lines.append(
            f"| {row['n']} | {row['p']} | {row['m_star']} | {human(row['counts_seconds'])} | "
            f"{human(row['first_failure_seconds'])} | {human(row['evolve_seconds'])} | "
            f"{human(row['cpu_seconds'])} |"
        )
    lines += [
        "",
        f"Total: {human(s['cpu_seconds'])} of CPU time, about {human(s['wall_seconds_low'])} to "
        f"{human(s['wall_seconds_high'])} of wall time on {eff['threads']} threads.",
        "",
    ]
    return "\n".join(lines)


def paths_paragraph(r: dict) -> list[str]:
    """What the engine/library ratios show, computed from the levels (empty if nothing to say)."""
    fixed = [
        lv["engine_over_library"]
        for c in r["cases"]
        if c["n"] >= 128
        for lv in c["levels"]
        if lv["w"] == 1
    ]
    walk = [lv["engine_over_library"] for c in r["cases"] for lv in c["levels"] if lv["w"] >= 4]
    if not fixed or not walk:
        return []
    return [
        "The two paths run the same kernels, but the engine's fixed elimination costs more: it "
        "reduces the trajectory's tableau rows, while a saved code's normalizer rows arrive "
        "already reduced (`symplectic_complement`). At w = 1, almost all elimination, the "
        f"engine took {min(fixed):.1f} to {max(fixed):.1f} times the library's time for "
        "n >= 128 (also visible in the first-failure rows at M = 100). From w = 4 on the walk "
        f"dominates and the paths agree: engine / library {min(walk):.2f} to {max(walk):.2f}.",
        "",
    ]


def derive(results: dict) -> dict:
    """The derived parts of the results, from the measured cases: model check, budgets, study."""
    for case in results["cases"]:
        case["model_check"] = model_check(case)
    cfg = results["settings"]
    results["budgets"] = budget_table(results["cases"], cfg["budgets"])
    results["study"] = study_example(results["cases"], {**STUDY, "sizes": cfg["sizes"][-4:]})
    return results


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(HERE / "results"), help="output directory")
    ap.add_argument("--quick", action="store_true", help="tiny sizes, for a smoke test")
    ap.add_argument("--sizes", type=int, nargs="*", default=None, help="system sizes n")
    ap.add_argument("--codes", type=int, default=8, help="codes with k > 0 per (n, p)")
    ap.add_argument("--max-eval-seconds", type=float, default=10.0)
    ap.add_argument("--max-regions", type=float, default=1e9)
    ap.add_argument("--max-predicted", type=float, default=90.0)
    ap.add_argument("--level-seconds", type=float, default=60.0)
    ap.add_argument("--min-codes", type=int, default=2)
    ap.add_argument("--max-w", type=int, default=64)
    ap.add_argument("--note", default="", help="a sentence about the machine's state")
    ap.add_argument(
        "--from-json",
        default=None,
        help="measure nothing: redo the derived tables and the markdown from this results file",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.from_json:
        results = jsonable(derive(json.loads(Path(args.from_json).read_text(encoding="utf-8"))))
        (out / "erasure_cost.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (out / "erasure_cost.md").write_text(render_markdown(results), encoding="utf-8")
        print(f"rederived from {args.from_json}; wrote {out}", flush=True)
        return
    cfg = {
        "sizes": list(args.sizes or ((16, 24) if args.quick else SIZES)),
        "rates": list(RATES),
        "codes": 2 if args.quick else args.codes,
        "orderings": list(ORDERINGS),
        "budgets": list(BUDGETS),
        "max_eval_seconds": args.max_eval_seconds,
        "max_regions": args.max_regions,
        "max_predicted": args.max_predicted,
        "level_seconds": 0.5 if args.quick else args.level_seconds,
        "min_codes": args.min_codes,
        "max_w": 3 if args.quick else args.max_w,
        "min_runs": 3,
        "max_runs": 25,
        "min_total": 0.01 if args.quick else 0.2,
        "evolve_total": 0.02 if args.quick else 0.5,
        "once_above": 1.0,
        "max_copies": 64,
    }
    start = time.perf_counter()
    print("warming up (compiles or loads the kernels)", flush=True)
    warm_up()
    cases = [make_case(n, p, cfg) for n in cfg["sizes"] for p in cfg["rates"]]
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
    measured = [measure_case(case, cfg) for case in cases]
    results = {
        "environment": env,
        "quick": args.quick,
        "seed": SEED,
        "settings": cfg,
        "cases": measured,
    }
    results = derive(results)
    results["total_seconds"] = time.perf_counter() - start
    results = jsonable(results)
    (out / "erasure_cost.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "erasure_cost.md").write_text(render_markdown(results), encoding="utf-8")
    print(f"total {results['total_seconds'] / 60:.1f} min; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
