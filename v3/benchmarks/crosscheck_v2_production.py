"""Cross-check SparseGF2 v3 against the v2 production engine on the PRL protocol.

Both engines run the single-reference purification protocol
(``studies/prl_production/PROTOCOL.md``) on the *same* graph bank (v2's own
Watts-Strogatz generator). Their random streams differ, so the comparison is
statistical: the purification-time distributions must agree (two-sample
Kolmogorov-Smirnov test, censored times placed at T_max + 1). It also reports
trajectories per second for each engine on one thread, and for v3 on all cores.

Usage::

    python v3/benchmarks/crosscheck_v2_production.py --n 32 --beta 0.1 --graphs 2000
    python v3/benchmarks/crosscheck_v2_production.py --n 32 64 128 256 --graphs 2000 500 200 100 \
        --out v3/benchmarks/results/crosscheck_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "v3" / "src"))
sys.path.insert(0, str(REPO / "v3" / "examples"))

import numba  # noqa: E402
from purification_time import production_spec  # noqa: E402
from sparsegf2.circuits.graphs import _ws_rewire_edges  # noqa: E402
from studies.prl_production.single_ref.engine import PointSpec, simulate_trajectory  # noqa: E402


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """KS statistic and asymptotic p-value (no SciPy dependency)."""
    a, b = np.sort(a), np.sort(b)
    grid = np.union1d(a, b)
    d = np.max(
        np.abs(
            np.searchsorted(a, grid, side="right") / a.size
            - np.searchsorted(b, grid, side="right") / b.size
        )
    )
    en = np.sqrt(a.size * b.size / (a.size + b.size))
    lam = (en + 0.12 + 0.11 / en) * d
    j = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1.0) ** (j - 1) * np.exp(-2 * lam**2 * j**2)), 0.0, 1.0))
    return float(d), p


def p_center(beta: float) -> float:
    """The production grid's steering centre for ``beta`` (nearest row of refinement_centers.csv)."""
    rows = np.loadtxt(
        REPO / "studies" / "prl_production" / "inputs" / "refinement_centers.csv",
        delimiter=",",
        skiprows=1,
    )
    return float(rows[np.argmin(np.abs(rows[:, 0] - beta)), 1])


def compare(n: int, beta: float, p: float, num_graphs: int) -> dict:
    cap = 8 * n
    bank = [np.asarray(_ws_rewire_edges(n, 2, beta, g), dtype=np.int64) for g in range(num_graphs)]
    point = PointSpec(n=n, beta=beta, p=p, n_graphs=num_graphs)

    simulate_trajectory(point, 0, bank[0])  # JIT warm-up
    t0 = time.perf_counter()
    v2 = [simulate_trajectory(point, g, bank[g]) for g in range(num_graphs)]
    v2_seconds = time.perf_counter() - t0
    v2_tau = np.array([r.tau_p if r.event_observed else cap + 1 for r in v2])

    spec = production_spec(n, beta, p, bank)
    spec.run(2, seed=0)
    numba.set_num_threads(1)
    one = spec.run(num_graphs, seed=1)
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    allc = spec.run(num_graphs, seed=1)
    assert np.array_equal(one.purification_layer, allc.purification_layer)
    v3_tau = np.where(one.purification_layer >= 0, one.purification_layer, cap + 1)

    d, pval = ks_two_sample(v2_tau, v3_tau)
    return {
        "n": n,
        "beta": beta,
        "p": p,
        "graphs": num_graphs,
        "cap": cap,
        "v2_censored": float(np.mean(v2_tau > cap)),
        "v3_censored": float(np.mean(v3_tau > cap)),
        "v2_median_tau": float(np.median(v2_tau)),
        "v3_median_tau": float(np.median(v3_tau)),
        "ks_statistic": d,
        "ks_pvalue": pval,
        "v2_traj_per_s_1thread": num_graphs / v2_seconds,
        "v3_traj_per_s_1thread": one.trajectories_per_second,
        "v3_traj_per_s_all_threads": allc.trajectories_per_second,
        "threads": numba.config.NUMBA_NUM_THREADS,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, nargs="+", default=[32])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--p", type=float, default=None, help="default: the production grid centre")
    ap.add_argument("--graphs", type=int, nargs="+", default=[2000], help="one value, or one per n")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    p = p_center(args.beta) if args.p is None else args.p
    graphs = args.graphs * len(args.n) if len(args.graphs) == 1 else args.graphs
    reports = []
    for n, g in zip(args.n, graphs, strict=True):
        reports.append(compare(n, args.beta, p, g))
        print(json.dumps(reports[-1], indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=1))
    bad = [r["n"] for r in reports if r["ks_pvalue"] < 1e-3]
    if bad:
        raise SystemExit(f"purification-time distributions disagree for n = {bad}")


if __name__ == "__main__":
    main()
