"""Measurement-induced transition: entanglement probes across a p-sweep.

For each system size and measurement rate, runs brickwork random Clifford
circuits on a ring (or any graph family below) from |0...0> for ``depth * n``
layers and records the half-cut entropy and the tripartite information of
the four quarters. ``-I3`` curves for different ``n`` cross at p_c.

    python examples/transition_scan.py --sizes 32 64 128 --rates 0.10 0.20 0.01 \\
        --trajectories 1000 --out scan.json

``--graph ws --beta 0.1`` runs on Watts-Strogatz graphs instead (with
``--schedule random_edges`` for layers of random graph edges). Every
(n, p) point uses the same seed, so the measurement masks at different rates
share their random numbers (common random numbers reduce the noise of
differences across p).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from sparsegf2 import analysis, graphs, mipt


def graph_for(kind: str, n: int, beta: float, seed: int):
    if kind == "ring":
        return graphs.cycle(n)
    if kind == "ws":
        return graphs.watts_strogatz(n, 4, beta, seed=seed)
    if kind == "complete":
        return graphs.complete(n)
    raise ValueError(f"unknown graph {kind!r}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument(
        "--rates",
        type=float,
        nargs=3,
        default=[0.08, 0.24, 0.01],
        metavar=("START", "STOP", "STEP"),
    )
    ap.add_argument("--trajectories", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=4, help="layers per qubit")
    ap.add_argument("--graph", choices=["ring", "ws", "complete"], default="ring")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--schedule", choices=["matching", "random_edges"], default="matching")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="transition_scan.json")
    args = ap.parse_args()

    start, stop, step = args.rates
    rates = np.round(np.arange(start, stop + step / 2, step), 6).tolist()
    out = {"args": vars(args), "rates": rates, "points": []}
    for n in args.sizes:
        edges = graph_for(args.graph, n, args.beta, seed=n)
        for p in rates:
            spec = mipt.MonitoredCircuit(
                num_system=n,
                edges=edges,
                layers=args.depth * n,
                measurement_rate=p,
                schedule=args.schedule,
                checkpoints="final",
                probes=[mipt.half_cut_entropy(), mipt.tripartite_information()],
            )
            res = spec.run(args.trajectories, seed=args.seed)
            s_half, s_err = analysis.mean_and_error(res.values["half_cut_entropy"][:, -1])
            i3, i3_err = analysis.mean_and_error(-res.values["tripartite_information"][:, -1])
            out["points"].append(
                {
                    "n": n,
                    "p": p,
                    "half_cut": float(s_half),
                    "half_cut_err": float(s_err),
                    "neg_i3": float(i3),
                    "neg_i3_err": float(i3_err),
                    "trajectories_per_second": res.trajectories_per_second,
                }  # fmt: skip
            )
            print(
                f"n={n:4d} p={p:.3f}  S_half={s_half:7.2f}±{s_err:.2f}  -I3={i3:6.2f}±{i3_err:.2f}",
                flush=True,
            )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
