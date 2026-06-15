"""Asymptotic-speedup benchmark: SparseGF2 vs Stim on random Clifford +
measurement circuits.

Workload
--------

Brick-wall random 2q Cliffords on a 1D chain at depth ``d = c * n``,
interleaved with Z-basis measurements at rate ``p`` per qubit per layer.
This is the canonical MIPT setup. The interesting regime for the
asymptotic-speedup story is ``p >= p_c ≈ 0.16`` (area-law phase): each
generator stays O(1)-sparse, so SparseGF2's per-gate cost is O(1) while
Stim's is O(n).

Usage
-----

::

    .venv/bin/python benchmarks/benchmark_random_clifford.py \\
        --ns 16 32 64 128 \\
        --depth-coeff 1.0 \\
        --p 0.25 \\
        --reps 3 \\
        --json results.json

If ``stim`` is importable in the same venv, both simulators are
benchmarked head-to-head and the speedup factor is reported.

Notes
-----

* The 2q gates are drawn uniformly from the native ``Sp(4, F_2)`` table,
  the same sampler the simulator uses internally.
* Warmup: the first call into the JIT path triggers Numba compilation
  (1-3 s). The benchmark calls ``warmup()`` before timing so the warmup
  cost isn't charged to ``SparseGF2``.
* Outcomes are not compared; this is a pure throughput benchmark. The
  step-by-step Stim parity is already locked in by the pytest suite.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

from sparsegf2 import SparseGF2
from sparsegf2.core.symplectic import random_symplectic_2q_batch

try:
    from sparsegf2.core.numba_kernels import warmup as _numba_warmup
except ImportError:  # pragma: no cover
    _numba_warmup = None

try:
    import stim

    _HAS_STIM = True
except ImportError:
    stim = None
    _HAS_STIM = False


# ----------------------------------------------------------------------
# Circuit generator
# ----------------------------------------------------------------------


@dataclass
class Circuit:
    n: int
    depth: int
    p_meas: float
    pairs: np.ndarray  # (depth, n//2, 2) int32
    sympls: np.ndarray  # (depth, n//2, 4, 4) uint8
    meas_mask: np.ndarray  # (depth, n) bool
    stim_tableaux: list | None = None


def make_circuit(n: int, depth: int, p_meas: float, seed: int) -> Circuit:
    """Brick-wall random Clifford + measurement circuit on a 1D chain."""
    rng = np.random.default_rng(seed)
    n_pairs = n // 2
    pairs = np.zeros((depth, n_pairs, 2), dtype=np.int32)
    for d in range(depth):
        offset = d & 1  # alternating brick pattern
        for k in range(n_pairs):
            qi = (2 * k + offset) % n
            qj = (2 * k + 1 + offset) % n
            pairs[d, k, 0] = qi
            pairs[d, k, 1] = qj
    sympls = np.empty((depth, n_pairs, 4, 4), dtype=np.uint8)
    for d in range(depth):
        sympls[d] = random_symplectic_2q_batch(n_pairs, rng=rng)
    meas_mask = rng.random((depth, n)) < p_meas
    return Circuit(n=n, depth=depth, p_meas=p_meas, pairs=pairs, sympls=sympls, meas_mask=meas_mask)


def _symp_to_stim_tableau(S: np.ndarray) -> "stim.Tableau":  # noqa: UP037  (quotes load-bearing: stim may be None)
    rows = [S[i] for i in range(4)]

    def to_ps(row):
        xs = row[:2].astype(bool)
        zs = row[2:].astype(bool)
        return stim.PauliString.from_numpy(xs=xs, zs=zs)

    return stim.Tableau.from_conjugated_generators(
        xs=[to_ps(rows[0]), to_ps(rows[1])],
        zs=[to_ps(rows[2]), to_ps(rows[3])],
    )


def precompile_stim_tableaux(circ: Circuit) -> None:
    """Convert every 4x4 symplectic to a stim.Tableau (one-time cost).

    Done before timing so that ``run_stim`` is purely the simulator's
    work, not numpy→stim conversion overhead.
    """
    if not _HAS_STIM:
        raise RuntimeError("stim is not importable")
    tableaux = []
    for d in range(circ.depth):
        layer = []
        for k in range(circ.pairs.shape[1]):
            layer.append(_symp_to_stim_tableau(circ.sympls[d, k]))
        tableaux.append(layer)
    circ.stim_tableaux = tableaux


# ----------------------------------------------------------------------
# Runners
# ----------------------------------------------------------------------


def run_sparsegf2(circ: Circuit, use_numba: bool = True) -> float:
    """Run the circuit on SparseGF2 and return elapsed wall-clock seconds."""
    sim = SparseGF2(circ.n, use_numba=use_numba)
    t0 = time.perf_counter()
    for d in range(circ.depth):
        for k in range(circ.pairs.shape[1]):
            qi, qj = int(circ.pairs[d, k, 0]), int(circ.pairs[d, k, 1])
            sim.apply_gate_2q(qi, qj, circ.sympls[d, k])
        for q in range(circ.n):
            if circ.meas_mask[d, q]:
                sim.measure_z(q)
    return time.perf_counter() - t0


def run_stim(circ: Circuit) -> float:
    if not _HAS_STIM:
        raise RuntimeError("stim is not importable")
    if circ.stim_tableaux is None:
        precompile_stim_tableaux(circ)
    sim = stim.TableauSimulator()
    sim.set_num_qubits(circ.n)
    t0 = time.perf_counter()
    for d in range(circ.depth):
        layer = circ.stim_tableaux[d]
        for k in range(circ.pairs.shape[1]):
            qi, qj = int(circ.pairs[d, k, 0]), int(circ.pairs[d, k, 1])
            sim.do_tableau(layer[k], [qi, qj])
        for q in range(circ.n):
            if circ.meas_mask[d, q]:
                sim.measure(q)
    return time.perf_counter() - t0


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


@dataclass
class Result:
    n: int
    depth: int
    p_meas: float
    rep: int
    seed: int
    sparsegf2_seconds: float
    stim_seconds: float | None
    speedup: float | None


def run_benchmark(
    ns: list[int],
    depth_coeff: float,
    p_meas: float,
    reps: int,
    seed_base: int = 0,
    use_numba: bool = True,
    quiet: bool = False,
) -> list[Result]:
    if use_numba and _numba_warmup is not None:
        _numba_warmup()  # amortize JIT compilation before timing

    results: list[Result] = []
    for n in ns:
        depth = max(1, int(depth_coeff * n))
        for rep in range(reps):
            seed = seed_base + rep
            circ = make_circuit(n, depth, p_meas, seed=seed)
            if _HAS_STIM:
                precompile_stim_tableaux(circ)
            t_ours = run_sparsegf2(circ, use_numba=use_numba)
            t_stim: float | None = None
            speedup: float | None = None
            if _HAS_STIM:
                t_stim = run_stim(circ)
                if t_ours > 0:
                    speedup = t_stim / t_ours
            r = Result(
                n=n,
                depth=depth,
                p_meas=p_meas,
                rep=rep,
                seed=seed,
                sparsegf2_seconds=t_ours,
                stim_seconds=t_stim,
                speedup=speedup,
            )
            results.append(r)
            if not quiet:
                msg = f"n={n:4d}  d={depth:4d}  p={p_meas:.2f}  rep={rep}  ours={t_ours:.3f}s"
                if t_stim is not None:
                    msg += f"  stim={t_stim:.3f}s  speedup={speedup:.2f}x"
                print(msg, flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ns", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--depth-coeff", type=float, default=1.0)
    p.add_argument("--p", type=float, default=0.25, help="measurement rate per qubit per layer")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-numba", action="store_true")
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args(argv)

    results = run_benchmark(
        ns=args.ns,
        depth_coeff=args.depth_coeff,
        p_meas=args.p,
        reps=args.reps,
        seed_base=args.seed,
        use_numba=not args.no_numba,
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
