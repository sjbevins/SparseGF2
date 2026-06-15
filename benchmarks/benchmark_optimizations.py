"""Does every optimization actually pay off? An A/B speedup matrix.

The simulator carries several independent optimizations. Each one is supposed to
make things faster, and this script *verifies* that, by timing the same work with
the optimization on and off, across several system sizes and circuit
architectures. An optimization that doesn't earn its complexity should show a
speedup near 1 (or below), and this script will say so plainly.

Optimizations measured
-----------------------
1. **Numba kernels**: the JIT-compiled inverted-index gate/measurement kernels
   vs the pure-Python fallback (``CircuitConfig(use_numba=...)``).
2. **Min-weight pivot**: choosing the lightest generator as the measurement
   pivot vs the first available one (``pivot_mode="min_weight"`` vs ``"first"``).
3. **Symplectic sampler**: the Bravyi-Maslov O(n²) canonical-form sampler vs the
   incremental kernel-basis sampler (``random_symplectic(sampler=...)``).

For each we sweep system size and (where it matters) the circuit architecture:
nearest-neighbour ``cycle`` brickwork, all-to-all ``complete`` brickwork, and the
``random_pool`` gating mode. The JIT is warmed up before timing so compile time
never pollutes a measurement.

Run::

    .venv/bin/python benchmarks/benchmark_optimizations.py            # default grid
    .venv/bin/python benchmarks/benchmark_optimizations.py --quick    # small + fast
    .venv/bin/python benchmarks/benchmark_optimizations.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import numpy as np

from sparsegf2.circuits import CircuitConfig, simulate
from sparsegf2.core.symplectic import random_symplectic

# ----------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------


def best_time(fn: Callable[[], object], *, repeat: int = 3, warmup: int = 1) -> float:
    """Median wall-clock seconds over ``repeat`` runs, after ``warmup`` untimed
    runs (so a JIT compile on the first call is never timed)."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


# ----------------------------------------------------------------------
# architectures
# ----------------------------------------------------------------------

# (label, kwargs merged into the base config): the architecture selections.
ARCHITECTURES = {
    "cycle/brickwork": {"graph_spec": "cycle", "gating_mode": "brickwork"},
    "complete/brickwork": {"graph_spec": "complete", "gating_mode": "brickwork"},
    "cycle/random_pool": {"graph_spec": "cycle", "gating_mode": "random_pool"},
}


def _config(n: int, arch: dict, **over) -> CircuitConfig:
    base = dict(
        n=n,
        picture="purification",
        p=0.1,
        depth_factor=4,
        **arch,
    )
    base.update(over)
    return CircuitConfig(**base)


# ----------------------------------------------------------------------
# per-optimization A/B sweeps
# ----------------------------------------------------------------------


def bench_numba(sizes: list[int], archs: dict, *, repeat: int) -> list[dict]:
    """use_numba=True vs False, a full circuit, per (n, architecture)."""
    rows = []
    for arch_name, arch in archs.items():
        for n in sizes:
            on = _config(n, arch, use_numba=True)
            off = _config(n, arch, use_numba=False)
            t_on = best_time(lambda c=on: simulate(c, sample_seed=0), repeat=repeat)
            t_off = best_time(lambda c=off: simulate(c, sample_seed=0), repeat=repeat)
            rows.append(
                {
                    "optimization": "numba",
                    "arch": arch_name,
                    "n": n,
                    "t_on_s": t_on,
                    "t_off_s": t_off,
                    "speedup": t_off / t_on if t_on else float("nan"),
                }
            )
    return rows


def bench_pivot(sizes: list[int], archs: dict, *, repeat: int) -> list[dict]:
    """pivot_mode='min_weight' vs 'first', a measurement-heavy circuit."""
    rows = []
    for arch_name, arch in archs.items():
        for n in sizes:
            mw = _config(n, arch, pivot_mode="min_weight", p=0.3)
            first = _config(n, arch, pivot_mode="first", p=0.3)
            t_mw = best_time(lambda c=mw: simulate(c, sample_seed=0), repeat=repeat)
            t_first = best_time(lambda c=first: simulate(c, sample_seed=0), repeat=repeat)
            rows.append(
                {
                    "optimization": "min_weight_pivot",
                    "arch": arch_name,
                    "n": n,
                    "t_on_s": t_mw,
                    "t_off_s": t_first,
                    "speedup": t_first / t_mw if t_mw else float("nan"),
                }
            )
    return rows


def bench_sampler(sizes: list[int], *, repeat: int) -> list[dict]:
    """random_symplectic: Bravyi-Maslov vs kernel-basis (a primitive, no arch)."""
    rows = []
    for n in sizes:

        def bm(n=n):
            random_symplectic(n, rng=np.random.default_rng(0), sampler="bravyi_maslov")

        def kb(n=n):
            random_symplectic(n, rng=np.random.default_rng(0), sampler="kernel_basis")

        t_bm = best_time(bm, repeat=repeat)
        t_kb = best_time(kb, repeat=repeat)
        rows.append(
            {
                "optimization": "bravyi_maslov_sampler",
                "arch": "-",
                "n": n,
                "t_on_s": t_bm,
                "t_off_s": t_kb,
                "speedup": t_kb / t_bm if t_bm else float("nan"),
            }
        )
    return rows


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------


def print_table(rows: list[dict]) -> None:
    by_opt: dict[str, list[dict]] = {}
    for r in rows:
        by_opt.setdefault(r["optimization"], []).append(r)
    for opt, group in by_opt.items():
        print(f"\n=== {opt}: faster=on/off ({'>1 means the optimization helps'}) ===")
        print(f"{'arch':<20}{'n':>6}{'on (ms)':>12}{'off (ms)':>12}{'speedup':>10}")
        for r in group:
            print(
                f"{r['arch']:<20}{r['n']:>6}"
                f"{r['t_on_s'] * 1e3:>12.2f}{r['t_off_s'] * 1e3:>12.2f}"
                f"{r['speedup']:>9.2f}x"
            )
        speeds = [r["speedup"] for r in group if np.isfinite(r["speedup"])]
        if speeds:
            verdict = "PAYS OFF" if np.median(speeds) > 1.1 else "marginal / no win"
            print(f"  -> median speedup {np.median(speeds):.2f}x  [{verdict}]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="small sizes, fast")
    ap.add_argument("--repeat", type=int, default=3, help="timed repeats (median)")
    ap.add_argument("--json", default=None, help="also write the raw rows here")
    args = ap.parse_args(argv)

    if args.quick:
        circ_sizes = [12, 24]
        samp_sizes = [16, 32]
    else:
        circ_sizes = [16, 32, 64]
        samp_sizes = [16, 32, 64, 128]

    print(f"system sizes (circuit): {circ_sizes}   (sampler): {samp_sizes}")
    rows: list[dict] = []
    rows += bench_numba(circ_sizes, ARCHITECTURES, repeat=args.repeat)
    rows += bench_pivot(circ_sizes, ARCHITECTURES, repeat=args.repeat)
    rows += bench_sampler(samp_sizes, repeat=args.repeat)
    print_table(rows)

    if args.json is not None:
        with open(args.json, "w") as f:
            json.dump({"rows": rows}, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
