"""Single-reference purification time on Watts-Strogatz graphs.

* ``n`` system qubits plus one reference, Bell-paired with system qubit ``n - 1``;
* a collection of ``--graphs`` Watts-Strogatz realizations of the circulant
  C(n, 2) (``2n`` edges, mean degree 4), each evolved with
  ``--circuits-per-graph`` random circuits (the same graphs serve every
  measurement rate when ``--graph-seed`` is kept);
* scramble: 32 layers of random all-to-all perfect matchings;
* each measured layer: ``n / 2`` graph edges drawn uniformly with replacement,
  a uniform two-qubit Clifford on each, then a Z measurement of every system
  qubit with probability ``p``;
* S(R) after every layer; the trajectory stops at its first zero, and is
  right-censored at ``T_max = 8 n``.

Run it::

    python examples/purification_time.py --n 64 --beta 0.1 --p 0.29 --graphs 10000

It prints the purification statistics and writes ``purification_n64_b0.1_p0.29.npz``
(an :class:`~sparsegf2.mipt.EnsembleResult`; reload with
``mipt.EnsembleResult.load``). Points are independent: run many ``(n, beta, p)``
in parallel jobs, or split one point into chunks with ``first_trajectory`` and
merge them with ``EnsembleResult.concatenate``.
"""

from __future__ import annotations

import argparse

import numpy as np

from sparsegf2 import analysis, graphs, mipt


def purification_spec(
    n: int, p: float, graph_list, circuits_per_graph: int = 1, t_max_factor: int = 8
) -> mipt.MonitoredCircuit:
    """The protocol for one point; trajectory ``t`` uses graph ``t // circuits_per_graph``."""
    return mipt.MonitoredCircuit(
        num_system=n,
        edges=graph_list,
        circuits_per_graph=circuits_per_graph,
        layers=t_max_factor * n,
        measurement_rate=p,
        schedule="random_edges",  # n // 2 edges per layer, uniformly with replacement
        reference="single",
        reference_partner=n - 1,
        scramble_matchings=32,
        checkpoints="every_layer",
        stop_when_purified=True,
    )


def watts_strogatz_graphs(n: int, beta: float, count: int, seed: int = 0) -> list[np.ndarray]:
    """``count`` independent realizations of WS-rewired C(n, 2), graph ``i`` from seed ``(seed, i)``."""
    return [graphs.watts_strogatz(n, 4, beta, seed=(seed, i)) for i in range(count)]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--p", type=float, default=0.29)
    ap.add_argument("--graphs", type=int, default=10_000)
    ap.add_argument("--circuits-per-graph", type=int, default=1)
    ap.add_argument("--graph-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    graph_list = watts_strogatz_graphs(args.n, args.beta, args.graphs, args.graph_seed)
    spec = purification_spec(args.n, args.p, graph_list, args.circuits_per_graph)
    result = spec.run(seed=args.seed)  # every graph, circuits_per_graph circuits each

    out = args.out or f"purification_n{args.n}_b{args.beta}_p{args.p}.npz"
    result.save(out)  # first, so that the run is never lost to the summary below
    print(result)
    print(f"saved {out}")

    tau = result.purification_layer
    observed = tau >= 0
    print(
        f"observed purifications: {observed.sum()} / {tau.size}  (censored at T_max = {spec.layers})"
    )
    print(f"median tau (Kaplan-Meier): {median_tau(result)}")


def median_tau(result: mipt.EnsembleResult) -> str:
    """The Kaplan-Meier median of tau, or why there is none (censored entries are tau = -1)."""
    time, observed = result.purification_times()  # censored at the last purification test
    if not observed.any():
        return "undefined (no trajectory purified)"
    km = analysis.kaplan_meier(time, observed)
    below = np.flatnonzero(km["survival"] <= 0.5)  # the survival is correctly rounded
    if not below.size:  # the survival curve never reaches 1/2 within the last checkpoint
        return f"> last checkpoint ({int(result.checkpoint_layers[-1])}) (more than half censored)"
    if not np.any(km["survival"] < 0.5):  # it stops at 1/2: the median lies beyond
        return f">= {km['times'][below[0]]:.0f} layers (not determined: only half purified)"
    return f"{km['times'][below[0]]:.0f} layers"


if __name__ == "__main__":
    main()
