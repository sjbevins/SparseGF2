"""Measurement-rate sweep: SparseGF2 vs Stim as the measurement rate ``p`` varies.

The companion to ``benchmark_random_clifford.py`` (which sweeps system size
``n``). This one fixes ``n`` and sweeps ``p`` to expose *why* SparseGF2 is fast
for monitored circuits: measurements keep the stabilizer tableau **sparse**
(area-law), and the sparse representation's per-measurement cost is
``O(anticommuters · weight)`` rather than the dense ``O(n^2)``. So the speedup
over Stim **grows with the measurement rate**:

- ``p = 0``: pure scrambling, volume-law, the tableau densifies → Stim wins.
- larger ``p``: area-law, the tableau stays sparse → SparseGF2 wins, by more
  and more.

This is the regime MIPT studies live in, so it is the relevant benchmark.

Run from the project root::

    .venv/bin/python benchmarks/benchmark_measurement_rate.py --n 192
    .venv/bin/python benchmarks/benchmark_measurement_rate.py --n 256 --ps 0 0.1 0.25 0.5 --json out.json

Reuses the circuit construction + timing from ``benchmark_random_clifford``;
Stim is optional (the script still times SparseGF2 alone if Stim is absent).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_base():
    """Import the sibling benchmark module (registered in sys.modules so its
    ``@dataclass`` definitions resolve ``__module__`` correctly)."""
    name = "benchmark_random_clifford"
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class RateResult:
    n: int
    depth: int
    p_meas: float
    sparsegf2_seconds: float
    stim_seconds: float | None
    speedup: float | None


def run_rate_sweep(n: int, ps: list[float], depth_coeff: float, reps: int) -> list[RateResult]:
    b = _load_base()
    if b._numba_warmup is not None:
        b._numba_warmup()
    depth = max(1, int(depth_coeff * n))
    results: list[RateResult] = []
    print(f"{'n':>5} {'depth':>6} {'p':>5} {'ours_s':>9} {'stim_s':>10} {'speedup':>9}")
    print("-" * 52)
    for p in ps:
        ours_runs, stim_runs = [], []
        for rep in range(reps):
            circ = b.make_circuit(n, depth, p, seed=rep)
            if b._HAS_STIM:
                b.precompile_stim_tableaux(circ)
            ours_runs.append(b.run_sparsegf2(circ))
            if b._HAS_STIM:
                stim_runs.append(b.run_stim(circ))
        ours = min(ours_runs)  # best of reps (least noise)
        stim = min(stim_runs) if stim_runs else None
        speedup = (stim / ours) if (stim and ours > 0) else None
        results.append(RateResult(n, depth, p, ours, stim, speedup))
        sp = f"{speedup:>8.2f}x" if speedup is not None else "      n/a"
        st = f"{stim:>10.3f}" if stim is not None else "       n/a"
        print(f"{n:>5} {depth:>6} {p:>5.2f} {ours:>9.3f} {st} {sp}")
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=192, help="system size (fixed)")
    p.add_argument(
        "--ps",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.25, 0.5],
        help="measurement rates to sweep",
    )
    p.add_argument("--depth-coeff", type=float, default=1.0, help="depth = coeff * n")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args(argv)

    results = run_rate_sweep(args.n, args.ps, args.depth_coeff, args.reps)
    print(
        "\nReading: at p=0 the tableau is dense (volume law) and Stim wins; as p "
        "rises the\ntableau stays sparse (area law) and SparseGF2's advantage grows."
    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
