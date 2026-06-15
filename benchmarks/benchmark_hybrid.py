"""Hybrid sparse/dense vs pure-sparse: the speedup across the measurement rate.

The hybrid mode (``SparseGF2(..., hybrid=True)`` / ``CircuitConfig(..., hybrid=
True)``) monitors the stabilizer density and switches to a bit-packed dense
representation while the state is volume-law, switching back to sparse when
measurements thin it out. This benchmark shows *where* that helps: it fixes ``n``
and sweeps the measurement rate ``p``, timing a full circuit with ``hybrid=False``
vs ``hybrid=True`` (same config, same seeds).

- low ``p``  -> volume-law -> the hybrid switches to dense -> big speedup.
- high ``p`` -> area-law   -> the hybrid stays sparse -> parity (monitoring is ~free).

The two paths produce identical physical results (asserted in the test suite); this
script only measures wall-clock. No Stim dependency.

Run from the project root::

    .venv/bin/python benchmarks/benchmark_hybrid.py --n 128
    .venv/bin/python benchmarks/benchmark_hybrid.py --n 192 --ps 0 0.05 0.1 0.25 0.5 --seeds 3
"""

from __future__ import annotations

import argparse
import time

from sparsegf2.circuits import CircuitConfig, simulate


def _best_ms(cfg, seeds: int) -> float:
    """Best-of-``seeds`` wall-clock (ms) for one config, after a JIT warmup."""
    simulate(cfg, sample_seed=0)  # warm the numba kernels (not timed)
    best = float("inf")
    for s in range(1, seeds + 1):
        t0 = time.perf_counter()
        simulate(cfg, sample_seed=s)
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.25, 0.5])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--depth-factor", type=int, default=8)
    args = ap.parse_args()

    base = dict(
        graph_spec="cycle",
        n=args.n,
        picture="single_ref",
        gating_mode="brickwork",
        matching_mode="round_robin",
        measurement_mode="bernoulli",
        depth_mode="O(n)",
        depth_factor=args.depth_factor,
    )
    print(
        f"n={args.n} single_ref brickwork O(n) depth_factor={args.depth_factor}, "
        f"best of {args.seeds} seeds"
    )
    print(f"  {'p':>6} {'sparse (ms)':>12} {'hybrid (ms)':>12} {'speedup':>8}")
    for p in args.ps:
        ds = _best_ms(CircuitConfig(**base, p=p, hybrid=False), args.seeds)
        dh = _best_ms(CircuitConfig(**base, p=p, hybrid=True), args.seeds)
        print(f"  {p:>6.3f} {ds:>12.0f} {dh:>12.0f} {ds / dh:>7.2f}x")


if __name__ == "__main__":
    main()
