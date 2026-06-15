"""Runtime: SparseGF2 vs Stim on a nearest-neighbour brickwork circuit at depth 8n.

Random two-qubit Cliffords in a 1D brick pattern, with Z-measurements at rate
p=0.25 (the area-law / measurement-heavy regime), run to depth d = 8n across
growing system sizes. The same circuit is replayed in Stim for a like-for-like
wall-clock comparison.

This is the regime where the sparse representation pays off: the measurements
hold the stabilizers at low weight, so SparseGF2's measurement cost stays far
below Stim's dense O(n^2) step, and the gap widens with n.

Reuses ``benchmarks/benchmark_random_clifford.py`` (which builds the matched
circuit and times both backends). Reads ``runtime_data.json`` if present, else
runs the benchmark. Run::

    .venv/bin/python docs/figures/benchmark_runtime.py
"""

from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

NS = [32, 64, 128, 256, 512]
DEPTH_COEFF = 8
P_MEAS = 0.25
REPS = 2

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "runtime_data.json")


def load_or_run() -> list[dict]:
    if os.path.exists(JSON):
        with open(JSON) as f:
            return json.load(f)
    sys.path.insert(0, os.path.join(HERE, "..", "..", "benchmarks"))
    from benchmark_random_clifford import run_benchmark

    results = run_benchmark(ns=NS, depth_coeff=DEPTH_COEFF, p_meas=P_MEAS, reps=REPS, seed_base=0)
    data = [r.__dict__ for r in results]
    with open(JSON, "w") as f:
        json.dump(data, f)
    return data


def main() -> int:
    data = load_or_run()
    by_n: dict[int, dict[str, list[float]]] = collections.defaultdict(
        lambda: {"sg": [], "st": []}
    )
    for r in data:
        by_n[r["n"]]["sg"].append(r["sparsegf2_seconds"])
        by_n[r["n"]]["st"].append(r["stim_seconds"])
    ns = np.array(sorted(by_n))
    sg = np.array([min(by_n[int(n)]["sg"]) for n in ns])  # best of reps
    st = np.array([min(by_n[int(n)]["st"]) for n in ns])

    print(f"{'n':>6}{'depth':>8}{'SparseGF2 (s)':>15}{'Stim (s)':>12}{'speedup':>10}")
    for n, a, b in zip(ns, sg, st, strict=True):
        print(f"{int(n):>6}{DEPTH_COEFF * int(n):>8}{a:>15.3f}{b:>12.3f}{b / a:>9.1f}x")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.loglog(ns, sg, "o-", label="SparseGF2 (Numba)")
    ax1.loglog(ns, st, "s-", label="Stim")
    ax1.set_xlabel("system size n (qubits)")
    ax1.set_ylabel("wall-clock time (s)")
    ax1.set_title(f"Runtime, brickwork depth 8n, p={P_MEAS}")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend()

    ax2.semilogx(ns, st / sg, "o-", color="C2")
    ax2.axhline(1.0, color="0.6", ls="--", lw=1)
    ax2.set_xlabel("system size n (qubits)")
    ax2.set_ylabel("speedup (Stim time / SparseGF2 time)")
    ax2.set_title("SparseGF2 speedup over Stim")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig("docs/figures/runtime_vs_stim.png", dpi=150, bbox_inches="tight")
    print("wrote docs/figures/runtime_vs_stim.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
