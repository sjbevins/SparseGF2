"""Erasure protection of the code a monitored circuit generates, versus depth.

Purification picture: every system qubit starts Bell-paired with its own
reference, so the system holds ``n`` logical qubits; measurements purify it.
A collection of Watts-Strogatz graphs is evolved with ``circuits_per_graph``
random circuits each. At every checkpoint this records

* the code dimension k (the code rate is k / n);
* first-failure sizes D over random orderings of the qubits: erasing the first
  m qubits of an ordering is uncorrectable exactly when m >= D;
* exhaustive counts of the uncorrectable erasures of 1..w qubits (exact
  P_C(m) for m <= w);
* the code itself at the final layer, to analyse again later.

From these, :mod:`sparsegf2_v3.analysis` gives the failure probability under
independent erasure of each qubit with probability e,
Phi(e) = sum_m C(n, m) e^m (1 - e)^(n - m) P_C(m) (hybrid estimator: exact for
m <= w, sampled above), over all circuits (a purified circuit, k = 0, never
fails) and conditioned on k > 0 at e = e0, and the tolerance
e_delta = sup{e : Phi(e) <= delta} of the codes with k > 0.

    python v3/examples/code_and_erasure.py --n 64 --p 0.05 --layers 256 --every 32 --out code.npz

The result file is an ``EnsembleResult`` (``mipt.EnsembleResult.load``).
"""

from __future__ import annotations

import argparse

import numpy as np

from sparsegf2_v3 import analysis, graphs, mipt


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.1, help="Watts-Strogatz rewiring probability")
    ap.add_argument("--layers", type=int, default=256)
    ap.add_argument("--every", type=int, default=32, help="checkpoint stride in layers")
    ap.add_argument("--graphs", type=int, default=50)
    ap.add_argument("--circuits-per-graph", type=int, default=4)
    ap.add_argument("--samples", type=int, default=100, help="random orderings per code")
    ap.add_argument("--exhaustive", type=int, default=2, help="largest exhaustively tested size w")
    ap.add_argument("--e0", type=float, default=0.1, help="erasure probability of the k>0 column")
    ap.add_argument("--delta", type=float, default=0.1, help="failure probability for e_delta")
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="code_and_erasure.npz")
    args = ap.parse_args()

    n, w = args.n, args.exhaustive
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=[graphs.watts_strogatz(n, 4, args.beta, seed=(args.graph_seed, g)) for g in range(args.graphs)],
        circuits_per_graph=args.circuits_per_graph,
        layers=args.layers,
        measurement_rate=args.p,
        schedule="random_edges",
        reference="purification",
        checkpoints=args.every,
        probes=[
            mipt.code_dimension(),
            mipt.erasure_first_failure(samples=args.samples),
            mipt.uncorrectable_region_counts(w),
            mipt.code_snapshot(checkpoints="final"),
        ],
    )  # fmt: skip
    res = spec.run(seed=args.seed)  # every graph, circuits_per_graph circuits each
    res.save(args.out)  # first, so the run is never lost to the summary below
    print(res)

    k = res.values["code_dimension"]  # (trajectories, checkpoints)
    first = res.values["erasure_first_failure"]  # (trajectories, checkpoints, samples)
    counts = res.values[f"uncorrectable_region_counts[{w}]"]  # (trajectories, checkpoints, w)
    rates = np.array([0.05, 0.1, 0.2])
    print(
        " layer     r=k/n    q+   Phi(e) for e = "
        + " ".join(f"{e:4.2f}" for e in rates)
        + f"   Phi({args.e0:g})|k>0   e_{args.delta:g}|k>0"
    )
    for c, layer in enumerate(res.checkpoint_layers):
        code = k[:, c] > 0
        phi = analysis.erasure_failure_probability(
            first[:, c], n, rates, exact_counts=counts[:, c]
        ).mean(axis=0)  # purified circuits (k = 0) count as Phi = 0
        line = f"{layer:6d}  {k[:, c].mean() / n:7.3f}  {code.mean():5.2f}   "
        line += " ".join(f"{f:5.3f}" for f in phi)
        if code.any():
            phi_code = analysis.erasure_failure_probability(
                first[code, c], n, args.e0, exact_counts=counts[code, c]
            ).mean()
            e_delta = analysis.erasure_tolerance(
                first[code, c], n, args.delta, exact_counts=counts[code, c]
            )
            line += f"   {phi_code:11.3f}   {e_delta:9.3f}"
        print(line)

    # The saved codes can be analysed again, e.g. with many more orderings.
    alive = np.flatnonzero(res.snapshots["code_snapshot"][1][:, 0] >= 0)
    if alive.size and k[alive[0], -1] > 0:
        code = res.code(int(alive[0]), 0)
        more = code.first_failure_sizes(10 * args.samples, seed=0)
        print(
            f"trajectory {alive[0]}: [[{code.num_qubits}, {code.num_logical}]] code, "
            f"P_C(m) for m = 1..4 from {more.size} orderings: "
            f"{np.round(analysis.failure_by_size(more, n)[1:5], 3).tolist()}"
        )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
