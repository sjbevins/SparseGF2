"""Geometry-correlated erasures of the codes monitored circuits generate, on a ring and a small world.

An erasure of m qubits strikes a region A with probability proportional to
exp(-J b(A)), b(A) the number of edges of an interaction graph G leaving A:
J = 0 is the uniform (independent-erasure) case, J > 0 favours compact
regions, and J -> inf keeps only the regions of smallest boundary. For two
ensembles in the purification picture,

* the nearest-neighbour brickwork ring (graphs.cycle, schedule "matching"),
  whose regions are drawn exactly (no Markov chain), and
* a Watts-Strogatz graph (schedule "random_edges"), whose regions come from a
  swap chain with replica exchange,

this runs the in-run probe ``mipt.correlated_erasure`` at the final layer and
prints, over the trajectories that hold a code (k > 0):

* P_J(m), the probability that an m-qubit erasure fails, per coupling J;
* the exact P_J(m) for small m from the saved codes (boundary-resolved counts),
  next to the probe's estimate;
* the J -> inf limit: on the ring the failing fraction of the n intervals
  (StabilizerCode.interval_failure_by_size), on the small world that of the
  minimum-boundary regions (erasures.minimum_boundary_regions);
* Phi_J(e), the failure probability when each qubit is erased with
  probability e and the region is then drawn from that law, and the tolerance
  e_delta(J) = sup{e : Phi_J(e) <= delta} of the mean curve;
* for the chain, a graph-side mixing check with the probe's settings
  (erasures.sample_regions on two seeds: the chain does not depend on the code).

By default the sizes are the binomial window of --eps (the sizes that carry
all but 1e-6 of the probability at those rates). The results are saved as
``ring.npz`` and ``small_world.npz`` in --out-dir (``mipt.EnsembleResult.load``;
``sparsegf2 curves`` and ``api.erasure_curves`` read them too).

    python examples/correlated_erasure.py --n 32 --trajectories 200 --out-dir correlated

See docs/observables/correlated-erasure.md for the mathematics and the algorithms.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from sparsegf2 import analysis, erasures, graphs, mipt


def _sizes(text: str | None, n: int, eps: list[float]) -> str:
    if text:
        return text
    lo, hi = analysis.binomial_window(n, eps, 1e-6)
    return f"{max(lo, 1)}:{min(hi, n - 1) + 1}"


def _limit_curves(res: mipt.EnsembleResult, rows: np.ndarray, edges: np.ndarray, ring: bool):
    """The exact J -> inf curve P_inf(m), m = 0..n, of each code in ``rows`` (None if too large)."""
    n = res.spec.num_system
    if ring:
        return np.array([res.code(int(t), 0).interval_failure_by_size() for t in rows])
    if max(math.comb(n, m) for m in range(1, n)) > erasures.ENUMERATION_BUDGET:
        return None
    minimizers = [erasures.minimum_boundary_regions(edges, n, m)[1] for m in range(1, n)]
    curves = np.zeros((rows.size, n + 1))
    for i, t in enumerate(rows):
        code = res.code(int(t), 0)
        curves[i, 1:n] = [code.region_failures(regions).mean() for regions in minimizers]
        curves[i, n] = 1.0  # k > 0
    return curves


def study(name: str, edges: np.ndarray, schedule: str, args: argparse.Namespace) -> None:
    n = args.n
    ring = name == "ring"
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=edges,
        layers=args.layers or 4 * n,
        measurement_rate=args.p,
        schedule=schedule,
        reference="purification",
        checkpoints="final",
        probes=[
            mipt.code_dimension(),
            mipt.correlated_erasure(
                args.couplings, sizes=_sizes(args.sizes, n, args.eps), samples=args.samples
            ),
            mipt.code_snapshot(),
        ],
    )
    sampler = spec.correlated_sampler("correlated_erasure")
    res = spec.run(args.trajectories, seed=args.seed)
    path = Path(args.out_dir) / f"{name}.npz"
    res.save(path)  # first, so the run is never lost to the summary below

    k = res.values["code_dimension"][:, -1]
    holds = k > 0
    couplings, sizes, values = res.correlated()  # values[t, e, c, i]
    print(f"\n== {name}: n = {n}, p = {args.p}, {res.num_trajectories} trajectories, "
          f"{int(holds.sum())} hold a code (k > 0), mean k/n = {k.mean() / n:.3f}")  # fmt: skip
    print(f"sampler: {sampler['method']}"
          + (f", {sampler['rungs'][0]}-{sampler['rungs'][1]} rungs, burn-in {sampler['burn_in']}"
             if sampler["method"] != "exact" else f" ({', '.join(sampler['geometries'])})"))  # fmt: skip
    if not holds.any():
        print("no trajectory holds a code: nothing to report")
        return
    rows = np.flatnonzero(holds)
    mean, err, _ = analysis.mean_over_codes(values[:, -1], holds)  # (couplings, sizes)
    shown = np.unique(np.linspace(0, sizes.size - 1, min(sizes.size, 8)).round().astype(int))
    print("P_J(m) over codes with k > 0, m = " + " ".join(f"{m:6d}" for m in sizes[shown]))
    for i, coupling in enumerate(couplings):
        print(f"  J = {coupling:5.2f}          " + " ".join(f"{x:6.3f}" for x in mean[i, shown]))

    # Exact P_J(m) for the small sizes, from the saved codes (every region enumerated).
    w = min(analysis.exhaustive_cutoff(n, erasures.ENUMERATION_BUDGET), n - 1, 3)
    exact = np.array([
        res.code(int(t), 0).boundary_failure_counts(edges, w).exact(couplings) for t in rows
    ]).mean(axis=0)  # (couplings, w)  # fmt: skip
    small = [i for i, m in enumerate(sizes.tolist()) if m <= w]
    if small:
        print(f"exact P_J(m), m <= {w}, vs the probe (+- its standard error):")
        for i, coupling in enumerate(couplings):
            cells = [f"m={sizes[j]}: {exact[i, sizes[j] - 1]:.3f} vs {mean[i, j]:.3f} +- {err[i, j]:.3f}"
                     for j in small]  # fmt: skip
            print(f"  J = {coupling:5.2f}   " + ";  ".join(cells))

    # Phi_J(e) and e_delta(J) of the mean curve (sizes the probe skipped are interpolated).
    eps = np.array(args.eps)
    per_code, interpolated = analysis.correlated_failure_curve(values[rows, -1], sizes, n)
    phi = analysis.failure_probability_from_curve(per_code, n, eps).mean(axis=0)
    mean_curve = per_code.mean(axis=0)
    e_delta, monotone = analysis.tolerance_from_curve(
        mean_curve, n, args.delta, return_monotone=True
    )
    labels = [f"{c:g}" for c in couplings]
    limit = _limit_curves(res, rows, edges, ring)
    if limit is not None:  # the exact J -> inf limit, from the saved codes
        phi = np.vstack([phi, analysis.failure_probability_from_curve(limit, n, eps).mean(axis=0)])
        e_inf, mono_inf = analysis.tolerance_from_curve(
            limit.mean(axis=0), n, args.delta, return_monotone=True
        )
        e_delta, monotone = np.append(e_delta, e_inf), np.append(monotone, mono_inf)
        labels.append("inf (exact)")
    print("   J           Phi_J(e) for e = " + " ".join(f"{e:6.3f}" for e in eps)
          + f"   e_{args.delta:g}(J)  monotone in m")  # fmt: skip
    for label, row, tol, mono in zip(labels, phi, e_delta, monotone, strict=True):
        print(f"  {label:>11}                  " + " ".join(f"{x:6.3f}" for x in row)
              + f"   {tol:9.4f}  {bool(mono)}")  # fmt: skip
    if interpolated[1:n].any():
        print(f"(sizes {np.flatnonzero(interpolated).tolist()} were interpolated)")

    if sampler["method"] != "exact":  # the chain: check its mixing on the graph alone
        m = int(sizes[sizes.size // 2])
        for seed in (1, 2):
            check = erasures.sample_regions(
                edges, n, m, couplings, args.samples, seed=seed, burn_in=sampler["burn_in"]
            )
            where = erasures.region_diagnostics(check.regions[int(np.argmax(couplings))], n)
            print(f"mixing check, m = {m}, seed {seed}: mean b per J "
                  f"{np.round(check.boundaries.mean(axis=1), 2).tolist()}, slowest node's tau "
                  f"{where['tau_int_max']:.1f}; warnings: {list(check.warnings) or 'none'}")  # fmt: skip
    print(f"saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--p", type=float, default=0.1, help="measurement rate per qubit and layer")
    ap.add_argument("--layers", type=int, default=None, help="default 4n")
    ap.add_argument("--trajectories", type=int, default=40)
    ap.add_argument("--couplings", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--sizes", default=None, help="sizes as 'a:b[:step]' (default: the window)")
    ap.add_argument("--samples", type=int, default=200, help="regions per coupling and size")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--delta", type=float, default=0.05, help="failure probability for e_delta")
    ap.add_argument("--beta", type=float, default=0.1, help="Watts-Strogatz rewiring probability")
    ap.add_argument("--graph-seed", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default="correlated_erasure")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    study("ring", graphs.cycle(args.n), "matching", args)
    small_world = graphs.watts_strogatz(args.n, 4, args.beta, seed=args.graph_seed)
    study("small_world", small_world, "random_edges", args)


if __name__ == "__main__":
    main()
