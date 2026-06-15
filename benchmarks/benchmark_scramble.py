"""Global random Clifford vs the gate scrambler: which is faster to scramble?

The single-qubit purification probe (Block et al., long-range MIPT) starts by
scrambling the system with a highly-scrambling random unitary. Two ways to do it:

1. **Global Clifford**: one uniform random ``n``-qubit Clifford applied in a
   single shot (``CircuitConfig(scramble=True)`` /
   ``SparseGF2.apply_clifford(random_symplectic(n, rng))``). Exact maximal
   scramble, but ``O(n^3)`` (generate + dense apply + sparse-index rebuild).
2. **Gate scrambler**: the unitary circuit run with no measurements for a
   scrambling depth (``warmup_layers``). For an all-to-all, 1 gate/layer circuit
   the depth needed to converge is ``~c * n * log2(n)`` gates.

This benchmarks both across system sizes. The two produce statistically identical
scrambled entanglement (verified separately; their Page curves agree to within
finite-depth convergence), so this script is only about *speed*. The result has a
crossover: the global Clifford wins for small ``n`` (fewer, denser operations),
the gate scrambler wins for large ``n`` (sparse gates beat a dense ``O(n^3)``).

Run::

    .venv/bin/python benchmarks/benchmark_scramble.py
    .venv/bin/python benchmarks/benchmark_scramble.py --ns 16 32 64 128 256 512
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from sparsegf2 import SparseGF2, enumerate_sp4, random_symplectic

_TABLE = enumerate_sp4()


def _single_ref(nsys: int, seed: int) -> SparseGF2:
    s = SparseGF2(nsys + 1, rng=np.random.default_rng(seed))
    s.apply_h(nsys - 1)
    s.apply_cx(nsys - 1, nsys)
    return s


def scramble_global(nsys: int, seed: int) -> None:
    s = _single_ref(nsys, seed)
    s.apply_clifford(random_symplectic(nsys, np.random.default_rng(seed)), qubits=range(nsys))


def scramble_gates(nsys: int, seed: int, depth: int) -> None:
    s = _single_ref(nsys, seed)
    rng = np.random.default_rng(seed)
    for _ in range(depth):
        i, j = rng.choice(nsys, size=2, replace=False)
        s.apply_gate_2q(int(i), int(j), _TABLE[int(rng.integers(720))])


def _best(fn, reps: int) -> float:
    fn()  # warm up the JIT
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ns", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument(
        "--depth-coeff", type=float, default=5.0, help="gate depth = c * n * ceil(log2 n)"
    )
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args(argv)

    print(f"{'n':>6}{'depth(gates)':>14}{'global (ms)':>14}{'gates (ms)':>14}{'gates/global':>14}")
    for n in args.ns:
        depth = int(args.depth_coeff * n * np.ceil(np.log2(n)))
        tg = _best(lambda n=n: scramble_global(n, 0), args.reps) * 1e3
        tc = _best(lambda n=n, d=depth: scramble_gates(n, 0, d), args.reps) * 1e3
        winner = "global" if tg < tc else "gates"
        print(f"{n:>6}{depth:>14}{tg:>14.2f}{tc:>14.2f}{tc / tg:>12.1f}x  [{winner} faster]")
    print(
        "\nReading: ratio > 1 means the global Clifford is faster. Expect a crossover "
        "around n ~ 100; below it the global one-shot wins, above it the sparse gate "
        "scrambler wins (the global Clifford is a dense O(n^3) generate-and-apply)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
