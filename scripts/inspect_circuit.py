#!/usr/bin/env python
"""Print the first N layers of a graph-defined circuit for visual inspection.

A thin CLI over :func:`sparsegf2.circuits.inspect_circuit`. It builds a
``CircuitConfig`` from the flags and renders the deterministic setup gates
plus the first ``--layers`` layers of the random Clifford + measurement
schedule, marking what is random vs deterministic.

Examples
--------
    # the single-reference picture, nearest-neighbour brickwork
    python scripts/inspect_circuit.py --n 8 --picture single_ref --layers 12

    # purification with O(n) random edges per step
    python scripts/inspect_circuit.py --n 8 --picture purification \\
        --gating random_edge --gates-per-layer 4 --layers 10

    # complete graph, fresh matchings, gated measurements
    python scripts/inspect_circuit.py --graph complete --n 8 \\
        --matching fresh --measurement gated --layers 6

Run with ``-h`` for the full flag list. Any future picture / graph / gating /
measurement mode is inspectable through the same flags with no script change.
"""

from __future__ import annotations

import argparse
import sys

from sparsegf2.circuits import CircuitConfig, inspect_circuit
from sparsegf2.errors import SparseGF2Error


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inspect the first N layers of a graph-defined circuit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--graph", default="cycle", help="graph spec: cycle | complete")
    p.add_argument("--n", type=int, required=True, help="number of system qubits")
    p.add_argument(
        "--picture",
        default="pure_state",
        help="pure_state | purification | single_ref",
    )
    p.add_argument("--gating", default="brickwork", help="brickwork | random_edge | random_pool")
    p.add_argument(
        "--matching",
        default="round_robin",
        help="round_robin | palette | fresh (brickwork only)",
    )
    p.add_argument(
        "--gates-per-layer",
        type=int,
        default=None,
        help="random edges per layer (random_edge/random_pool only; "
        "default 1 for random_edge, n/2 for random_pool)",
    )
    p.add_argument(
        "--measurement",
        default="bernoulli",
        help="bernoulli | gated | random_pair | uniform_count",
    )
    p.add_argument(
        "--meas-count",
        type=int,
        default=None,
        help="candidate qubits per layer (uniform_count only; default n/2)",
    )
    p.add_argument("--p", type=float, default=0.15, help="measurement probability")
    p.add_argument("--depth-mode", default="O(n)", help="O(n) | O(log_n) | until_purified")
    p.add_argument("--depth-factor", type=int, default=8, help="depth multiplier")
    p.add_argument("--warmup", type=int, default=0, help="gate-only warmup layers")
    p.add_argument("--seed", type=int, default=0, help="sample seed")
    p.add_argument("--layers", type=int, default=20, help="max measured layers to show")
    p.add_argument("--no-warmup", action="store_true", help="hide warmup layers")
    return p


def main(argv: list[str] | None = None) -> int:
    # The inspector output is intentionally Unicode (×, →, ↔, measurement meters).
    # On a non-UTF-8 console (Windows defaults to a legacy code page) printing it
    # would raise UnicodeEncodeError, so emit UTF-8 explicitly.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        config = CircuitConfig(
            graph_spec=args.graph,
            n=args.n,
            picture=args.picture,
            gating_mode=args.gating,
            matching_mode=args.matching,
            gates_per_layer=args.gates_per_layer,
            measurement_mode=args.measurement,
            meas_count=args.meas_count,
            p=args.p,
            depth_mode=args.depth_mode,
            depth_factor=args.depth_factor,
            warmup_layers=args.warmup,
        )
    except SparseGF2Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        inspect_circuit(
            config,
            sample_seed=args.seed,
            max_layers=args.layers,
            include_warmup=not args.no_warmup,
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
