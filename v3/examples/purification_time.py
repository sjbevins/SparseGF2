"""Single-reference purification time on Watts-Strogatz graphs (the PRL production protocol).

This is ``studies/prl_production/PROTOCOL.md`` expressed with SparseGF2 v3:

* ``n`` system qubits plus one reference, Bell-paired with system qubit ``n - 1``;
* one Watts-Strogatz realization of the circulant C(n, 2) (``2n`` edges, mean
  degree 4) per trajectory;
* scramble: 32 layers of random all-to-all perfect matchings;
* each measured layer: ``n / 2`` graph edges drawn uniformly with replacement,
  a uniform two-qubit Clifford on each, then a Z measurement of every system
  qubit with probability ``p``;
* S(R) after every layer; the trajectory stops at its first zero, and is
  right-censored at ``T_max = 8 n``.

Run it::

    python v3/examples/purification_time.py --n 64 --beta 0.1 --p 0.29 --trajectories 10000

It prints the purification statistics and writes ``purification_n64_b0.1_p0.29.npz``
(an :class:`~sparsegf2_v3.mipt.EnsembleResult`; reload with
``mipt.EnsembleResult.load``). Points are independent: run many ``(n, beta, p)``
in parallel jobs, or split one point into chunks with ``first_trajectory`` and
merge them with ``EnsembleResult.concatenate``.
"""

from __future__ import annotations

import argparse

import numpy as np

from sparsegf2_v3 import analysis, graphs, mipt


def production_spec(
    n: int, beta: float, p: float, graph_bank, t_max_factor: int = 8
) -> mipt.MonitoredCircuit:
    """The protocol for one point; trajectory ``t`` uses ``graph_bank[t % len(graph_bank)]``."""
    return mipt.MonitoredCircuit(
        num_system=n,
        edges=graph_bank,
        layers=t_max_factor * n,
        measurement_rate=p,
        schedule="random_edges",  # n // 2 edges per layer, uniformly with replacement
        reference="single",
        reference_partner=n - 1,
        scramble_matchings=32,
        checkpoints="every_layer",
        stop_when_purified=True,
    )


def watts_strogatz_bank(n: int, beta: float, count: int, seed: int = 0) -> list[np.ndarray]:
    """``count`` independent realizations of WS-rewired C(n, 2), graph ``i`` from seed ``(seed, i)``."""
    return [graphs.watts_strogatz(n, 4, beta, seed=(seed, i)) for i in range(count)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--p", type=float, default=0.29)
    ap.add_argument("--trajectories", type=int, default=10_000)
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bank = watts_strogatz_bank(args.n, args.beta, args.trajectories, args.graph_seed)
    spec = production_spec(args.n, args.beta, args.p, bank)
    result = spec.run(args.trajectories, seed=args.seed)

    tau = result.purification_layer
    observed = tau >= 0
    times = np.where(observed, tau, result.final_layer)
    km = analysis.kaplan_meier(times, observed)
    print(result)
    print(
        f"observed purifications: {observed.sum()} / {tau.size}  (censored at T_max = {spec.layers})"
    )
    if observed.any():
        print(
            f"median tau (Kaplan-Meier): {km['times'][np.searchsorted(-km['survival'], -0.5)]:.0f} layers"
        )
    out = args.out or f"purification_n{args.n}_b{args.beta}_p{args.p}.npz"
    result.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
