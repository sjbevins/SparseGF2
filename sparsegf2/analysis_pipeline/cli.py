"""CLI for ``python -m sparsegf2.analysis_pipeline``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sparsegf2.analysis_pipeline.analyses import ANALYSIS_REGISTRY
from sparsegf2.analysis_pipeline.config import AnalysisConfig
from sparsegf2.analysis_pipeline.orchestrator import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    names = sorted(ANALYSIS_REGISTRY)
    p = argparse.ArgumentParser(
        prog="python -m sparsegf2.analysis_pipeline",
        description=(
            "Run the analysis suite over an existing runs/<run_id>/ tree "
            "and write results into data/n=.../p=.../analysis/ plus "
            "run-level aggregates.parquet."
        ),
    )
    p.add_argument("run_dir", type=Path, help="Path to runs/<run_id>/")
    p.add_argument("--only", nargs="+", metavar="NAME",
                   choices=names, default=None,
                   help=f"Run only these analyses. Choices: {names}")
    p.add_argument("--skip", nargs="+", metavar="NAME",
                   choices=names, default=None,
                   help="Run every default analysis except these.")
    p.add_argument("--force", nargs="+", metavar="NAME",
                   choices=names, default=None,
                   help="Recompute these even when output files exist.")
    p.add_argument("--sizes", nargs="+", type=int, default=None,
                   help="Only process cells whose n is in this list.")
    p.add_argument("--p-values", nargs="+", type=float, default=None,
                   help="Only process cells whose p is in this list.")
    p.add_argument("--workers", type=int, default=1, dest="n_workers",
                   help="Parallel worker count (1 = inline).")
    p.add_argument("--quiet", action="store_true", dest="quiet",
                   help="Suppress per-cell progress output.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    cfg = AnalysisConfig(
        run_dir=args.run_dir,
        only=args.only,
        skip=args.skip,
        force=args.force,
        sizes=args.sizes,
        p_values=args.p_values,
        n_workers=args.n_workers,
        verbose=not args.quiet,
    )
    report = run_pipeline(cfg)
    print("")
    print(report.summary())
    return 0 if not report.errors else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["build_parser", "main"]
