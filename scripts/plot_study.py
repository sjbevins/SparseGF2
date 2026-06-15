#!/usr/bin/env python
"""Plot a saved Study: each observable vs the measurement rate, one curve per n.

Examples
--------
    # plot every observable column to a file
    python scripts/plot_study.py runs/transition --out transition.png

    # just list what is available to plot (incl. columns added by Study.augment)
    python scripts/plot_study.py runs/transition --list

    # plot specific columns, sweeping a different x-axis
    python scripts/plot_study.py runs/depth_scan --x depth_factor --cols a.code_rate

The plot auto-detects every numeric observable / analysis column, so after you
run ``study.augment([...])`` to add new observables this script picks them up
with no changes.
"""

from __future__ import annotations

import argparse
import sys

from sparsegf2.analysis import Study, observable_columns, plot_study


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("study_dir", help="path to a Study directory (rows.parquet + manifest.json)")
    p.add_argument("--out", default=None, help="save the figure here (PNG/PDF/...); else show it")
    p.add_argument("--x", default="p", help="x-axis column (default: p)")
    p.add_argument("--group", default="n", help="one curve per distinct value of this column")
    p.add_argument(
        "--cols", nargs="+", default=None, help="observable columns to plot (default: all)"
    )
    p.add_argument(
        "--errorstyle",
        default="bars",
        choices=["bars", "band"],
        help="show uncertainty as error bars (default) or a shaded band",
    )
    p.add_argument(
        "--err",
        default="sem",
        choices=["sem", "std"],
        help="error metric across seeds: standard error of the mean (default) or std",
    )
    p.add_argument(
        "--list", action="store_true", help="list the plottable observable columns and exit"
    )
    args = p.parse_args(argv)

    study = Study.open(args.study_dir)
    if args.list:
        cols = observable_columns(study)
        print("\n".join(cols) if cols else "(no numeric observable columns)")
        return 0

    if args.out is not None:
        import matplotlib

        matplotlib.use("Agg")  # headless

    plot_study(
        study,
        xcol=args.x,
        group=args.group,
        ycols=args.cols,
        err=args.err,
        errorstyle=args.errorstyle,
        out=args.out,
    )
    if args.out is not None:
        print(f"wrote {args.out}")
    else:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
