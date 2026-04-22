"""
argparse entry point for ``python -m sparsegf2.circuits``.

Maps CLI flags onto :class:`CircuitConfig` + :class:`RunConfig` and invokes
:class:`SweepDriver`. Pre-flight validation failures exit with code 2 and
print the :class:`ValidationReport` text; any other failure exits with 1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from sparsegf2.circuits.config import (
    CircuitConfig, RunConfig,
    DEPTH_MODES, MATCHING_MODE_NAMES, MEASUREMENT_MODE_NAMES,
    GATING_MODES, PICTURE_NAMES, GRAPH_SPECS_MVP,
)
from sparsegf2.circuits.driver import SweepDriver
from sparsegf2.circuits.validator import CompatibilityError


# Depth-mode CLI aliases

_DEPTH_ALIASES = {
    "O(n)": "O(n)",
    "O_n": "O(n)",
    "o_n": "O(n)",
    "on": "O(n)",
    "O(log_n)": "O(log_n)",
    "O_log_n": "O(log_n)",
    "o_log_n": "O(log_n)",
    "ologn": "O(log_n)",
}


def _parse_depth_mode(s: str) -> str:
    canonical = _DEPTH_ALIASES.get(s)
    if canonical is None:
        raise argparse.ArgumentTypeError(
            f"--depth-mode must parse to one of {DEPTH_MODES} "
            f"(accepts aliases: {sorted(_DEPTH_ALIASES)}); got {s!r}"
        )
    return canonical


# argparse builder

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sparsegf2.circuits",
        description=(
            "Run a graph-defined random-Clifford + measurement sweep and "
            "write the standardized run directory to disk."
        ),
    )

    # Graph
    p.add_argument("--graph", required=True, choices=GRAPH_SPECS_MVP,
                   help="Graph topology.")
    p.add_argument("--picture", default="purification", choices=PICTURE_NAMES,
                   help="Physics picture: purification (2n qubits, full tableau) "
                        "or single_ref (n+1 qubits, single-qubit probe).")
    p.add_argument("--gating", default="matching", choices=GATING_MODES,
                   dest="gating_mode",
                   help="Gating mode (MVP: matching).")
    p.add_argument("--matching-mode", default="round_robin",
                   choices=MATCHING_MODE_NAMES,
                   help="How per-layer matchings are selected.")
    p.add_argument("--measurement", default="uniform",
                   choices=MEASUREMENT_MODE_NAMES,
                   dest="measurement_mode",
                   help="Measurement mode (MVP: uniform).")

    # Depth
    p.add_argument("--depth-mode", default="O(n)", type=_parse_depth_mode,
                   help=f"Depth scaling: one of {DEPTH_MODES} "
                        "(aliases: O_n, O_log_n, ologn, etc.).")
    p.add_argument("--depth-factor", type=int, default=8,
                   help="Multiplicative constant for total layers.")

    # Cliffords
    p.add_argument("--n-cliffords", type=int, default=11520,
                   help="Size of the 2-qubit Clifford palette (default: full group).")

    # Sweep parameters
    p.add_argument("--sizes", type=int, nargs="+", required=True,
                   help="System sizes (number of system qubits); must be even for MVP graphs.")
    p.add_argument("--p-min", type=float, required=True,
                   help="Minimum measurement rate in the sweep.")
    p.add_argument("--p-max", type=float, required=True,
                   help="Maximum measurement rate in the sweep.")
    p.add_argument("--n-p", type=int, required=True,
                   help="Number of p-values (linspace between p_min and p_max).")
    p.add_argument("--samples", type=int, required=True, dest="n_samples_per_cell",
                   help="Number of samples per (n, p) cell.")

    # Seeds + parallelism
    p.add_argument("--seed", type=int, default=42, dest="base_seed",
                   help="Base RNG seed; sample seed is added to this.")
    p.add_argument("--workers", type=int, default=1, dest="n_workers",
                   help="Parallel worker count (1 = inline).")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Samples per worker batch.")

    # Output slots
    p.add_argument("--save-tableaus", action="store_true",
                   help="Persist end-of-circuit tableaus to tableaus.h5 per cell.")
    p.add_argument("--save-realizations", action="store_true",
                   help="Persist full circuit layer trace to realizations.h5 per cell.")
    p.add_argument("--record-time-series", action="store_true",
                   help="single_ref only: record S(qubit n) at every layer and "
                        "write timeseries.h5 per cell (for purification-time "
                        "analysis).")

    # Output location
    p.add_argument("--output", type=Path, default=Path("runs"),
                   dest="output_root",
                   help="Directory under which runs/<run_id>/ is created.")
    p.add_argument("--run-id", default=None,
                   help="Explicit run_id (default: auto-generate).")
    p.add_argument("--notes", default="",
                   help="Free-form description to persist in manifest.json.")
    p.add_argument("--no-progress", action="store_true",
                   help="Suppress per-cell progress logs.")
    return p


# main()

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    circuit = CircuitConfig(
        graph_spec=args.graph,
        n=args.sizes[0],                              # placeholder; driver overrides
        picture=args.picture,
        gating_mode=args.gating_mode,
        matching_mode=args.matching_mode,
        measurement_mode=args.measurement_mode,
        p=args.p_min,                                 # placeholder; driver overrides
        depth_mode=args.depth_mode,
        depth_factor=args.depth_factor,
        n_cliffords=args.n_cliffords,
        base_seed=args.base_seed,
        record_time_series=args.record_time_series,
    )
    run_cfg = RunConfig(
        circuit=circuit,
        sizes=args.sizes,
        p_min=args.p_min,
        p_max=args.p_max,
        n_p=args.n_p,
        n_samples_per_cell=args.n_samples_per_cell,
        output_root=args.output_root,
        run_id=args.run_id,
        save_tableaus=args.save_tableaus,
        save_realizations=args.save_realizations,
        n_workers=args.n_workers,
        batch_size=args.batch_size,
        notes=args.notes,
    )

    driver = SweepDriver(run_cfg, progress=not args.no_progress)
    try:
        run_dir = driver.run()
    except CompatibilityError as e:
        print(str(e), file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise
    print(f"\nDONE: {run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["build_parser", "main"]
