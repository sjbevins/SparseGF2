"""Cross-check SparseGF2 v3 against the v2 production engine on the PRL protocol.

Both engines run the single-reference purification protocol
(``studies/prl_production/PROTOCOL.md``) on the *same* graph bank (v2's own
Watts-Strogatz generator). Their random streams differ, so the comparison is
statistical: the purification-time distributions must agree (two-sample
Kolmogorov-Smirnov test, censored times placed at T_max + 1). It also reports
trajectories per second for each engine on one thread, and for v3 on all cores.

Usage::

    python v3/benchmarks/crosscheck_v2_production.py --n 32 --beta 0.1 --p 0.293 --graphs 2000
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--p", type=float, default=0.293)
    ap.add_argument("--graphs", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    n, cap = args.n, 8 * args.n

    bank = [
        np.asarray(_ws_rewire_edges(n, 2, args.beta, g), dtype=np.int64) for g in range(args.graphs)
    ]
    point = PointSpec(n=n, beta=args.beta, p=args.p, n_graphs=args.graphs)

    simulate_trajectory(point, 0, bank[0])  # JIT warm-up
    t0 = time.perf_counter()
    v2 = [simulate_trajectory(point, g, bank[g]) for g in range(args.graphs)]
    v2_seconds = time.perf_counter() - t0
    v2_tau = np.array([r.tau_p if r.event_observed else cap + 1 for r in v2])

    spec = production_spec(n, args.beta, args.p, bank)
    spec.run(2, seed=0)
    numba.set_num_threads(1)
    one = spec.run(args.graphs, seed=1)
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    allc = spec.run(args.graphs, seed=1)
    assert np.array_equal(one.purification_layer, allc.purification_layer)
    v3_tau = np.where(one.purification_layer >= 0, one.purification_layer, cap + 1)

    d, pval = ks_two_sample(v2_tau, v3_tau)
    report = {
        "n": n,
        "beta": args.beta,
        "p": args.p,
        "graphs": args.graphs,
        "cap": cap,
        "v2_censored": float(np.mean(v2_tau > cap)),
        "v3_censored": float(np.mean(v3_tau > cap)),
        "v2_median_tau": float(np.median(v2_tau)),
        "v3_median_tau": float(np.median(v3_tau)),
        "ks_statistic": d,
        "ks_pvalue": pval,
        "v2_traj_per_s_1thread": args.graphs / v2_seconds,
        "v3_traj_per_s_1thread": one.trajectories_per_second,
        "v3_traj_per_s_all_threads": allc.trajectories_per_second,
        "threads": numba.config.NUMBA_NUM_THREADS,
    }
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    if pval < 1e-3:
        raise SystemExit("purification-time distributions disagree")


if __name__ == "__main__":
    main()
