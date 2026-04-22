"""
All-to-all MIPT sweep using the single-qubit reference probe.

Runs a full ``(n, p)`` parameter sweep on the complete graph K_n with:

- picture = single_ref (n system + 1 reference qubit),
- gating = round-robin 1-factorization of K_n (n-1 matchings, each n/2
  edges, covering all n(n-1)/2 edges per "period"),
- measurement = uniform (every system qubit gets measured with prob p
  after every gate layer),

and writes a standardised hive-partitioned run directory.

Usage
-----
Full sweep as requested (warning: expensive at large n):

    python scripts/run_all_to_all_mipt.py \
        --sizes 16 32 64 128 256 512 \
        --p-min 0.0 --p-max 0.75 --p-step 0.01 \
        --samples 500 --workers 8 \
        --output runs/ --run-id all_to_all_mipt

Quick smoke test (suitable for laptops):

    python scripts/run_all_to_all_mipt.py --quick

The script prints an estimated ``total_cells x total_samples`` before
running so the operator can sanity-check cost.

Output lands at ``<output>/<run_id>/``; point
``scripts/generate_report.py`` at that directory to produce the
``crossing_plot.pdf`` and ``report.tex``.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from sparsegf2.circuits import CircuitConfig, RunConfig, SweepDriver


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Sweep single_ref MIPT on the all-to-all graph K_n with a "
            "round-robin 1-factorization gate schedule."
        )
    )
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[16, 32, 64, 128, 256, 512],
                   help="System sizes (must be even). Default: 16..512.")
    p.add_argument("--p-min", type=float, default=0.0,
                   help="Minimum measurement rate. Default: 0.0.")
    p.add_argument("--p-max", type=float, default=0.75,
                   help="Maximum measurement rate. Default: 0.75.")
    p.add_argument("--p-step", type=float, default=0.01,
                   help="Step between consecutive p values. Default: 0.01 "
                        "(76 p values over [0.0, 0.75]).")
    p.add_argument("--samples", type=int, default=500,
                   dest="n_samples_per_cell",
                   help="Samples per (n, p) cell. Default: 500.")
    p.add_argument("--depth-factor", type=int, default=8,
                   help="Total layers = depth_factor * n. Default: 8.")
    p.add_argument("--workers", type=int, default=1, dest="n_workers",
                   help="Parallel ProcessPoolExecutor workers. Default: 1.")
    p.add_argument("--seed", type=int, default=42, dest="base_seed",
                   help="Base RNG seed (per-sample seed adds to this).")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Samples per worker batch.")
    p.add_argument("--output", type=Path, default=Path("runs"),
                   dest="output_root",
                   help="Parent directory for the run output.")
    p.add_argument("--run-id", default="all_to_all_mipt",
                   help="Subdirectory name under --output.")
    p.add_argument("--notes", default="",
                   help="Free-form note persisted in manifest.json.")
    p.add_argument("--quick", action="store_true",
                   help="Shortcut for a small test sweep (sizes 8 16, p "
                        "0..0.5 step 0.1, 32 samples). Overrides the "
                        "related flags.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit without running.")
    return p


def _apply_quick_overrides(args: argparse.Namespace) -> None:
    args.sizes = [8, 16]
    args.p_min = 0.0
    args.p_max = 0.5
    args.p_step = 0.1
    args.n_samples_per_cell = 32
    args.depth_factor = 4
    args.run_id = f"{args.run_id}_quick"


def _make_p_grid(p_min: float, p_max: float, step: float) -> np.ndarray:
    n_p = int(round((p_max - p_min) / step)) + 1
    return np.linspace(p_min, p_max, n_p, dtype=np.float64)


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.quick:
        _apply_quick_overrides(args)

    # Even-n guard for the complete-graph 1-factorization.
    for n in args.sizes:
        if n % 2 != 0:
            print(
                f"error: K_n 1-factorization requires even n; got {n}",
                file=sys.stderr,
            )
            return 2

    p_values = _make_p_grid(args.p_min, args.p_max, args.p_step)
    n_p = int(len(p_values))

    cc = CircuitConfig(
        graph_spec="complete",
        n=int(args.sizes[0]),                 # placeholder; driver overrides
        picture="single_ref",
        gating_mode="matching",
        matching_mode="round_robin",          # K_n 1-factorization
        measurement_mode="uniform",
        p=float(p_values[0]),                 # placeholder
        depth_mode="O(n)",
        depth_factor=int(args.depth_factor),
        base_seed=int(args.base_seed),
    )
    rc = RunConfig(
        circuit=cc,
        sizes=list(args.sizes),
        p_min=float(args.p_min),
        p_max=float(args.p_max),
        n_p=n_p,
        n_samples_per_cell=int(args.n_samples_per_cell),
        output_root=args.output_root,
        run_id=args.run_id,
        save_tableaus=False,
        save_realizations=False,
        n_workers=int(args.n_workers),
        batch_size=int(args.batch_size),
        notes=args.notes or (
            "All-to-all (K_n) MIPT sweep with single_ref probe and "
            "round-robin 1-factorization."
        ),
    )

    total_cells = rc.total_cells()
    total_samples = rc.total_samples()
    # Back-of-envelope cost: per-sample ~ depth_factor * n^2 / 2 gates (K_n).
    approx_gates_per_sample = int(
        args.depth_factor * max(args.sizes) ** 2 / 2
    )
    print("=" * 64)
    print("All-to-all MIPT sweep (single_ref, round_robin on K_n)")
    print("=" * 64)
    print(f"  sizes                : {list(args.sizes)}")
    print(f"  p grid               : {n_p} values over "
          f"[{args.p_min}, {args.p_max}] step {args.p_step}")
    print(f"  samples per (n, p)   : {args.n_samples_per_cell}")
    print(f"  total cells          : {total_cells}")
    print(f"  total samples        : {total_samples:,}")
    print(f"  depth_factor         : {args.depth_factor}  "
          f"(total_layers = depth_factor * n)")
    print(f"  workers              : {args.n_workers}")
    print(f"  gates/sample (largest n, approx) : {approx_gates_per_sample:,}")
    print(f"  output               : "
          f"{args.output_root / args.run_id}")
    print("=" * 64, flush=True)

    if args.dry_run:
        print("dry-run: not executing.")
        return 0

    t0 = time.perf_counter()
    run_dir = SweepDriver(rc, progress=True).run()
    dt = time.perf_counter() - t0
    print(
        f"\nDone in {dt:.1f} s. Run directory: {run_dir}\n"
        f"Next step: python scripts/generate_report.py {run_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
