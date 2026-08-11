"""Plot realized Watts--Strogatz rewiring statistics from a strict summary CSV."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .spec import beta_from_key, canonical_beta_key

SUMMARY_FIELDS = (
    "collection_id",
    "n",
    "beta_key",
    "beta",
    "n_graphs",
    "mean_displaced",
    "sem_displaced",
    "mean_operations",
    "sem_operations",
    "mean_restored",
    "sem_restored",
    "mean_displaced_fraction",
    "sem_displaced_fraction",
)

_NONNEGATIVE_INTEGER = re.compile(r"0|[1-9][0-9]*")


@dataclass(frozen=True, slots=True)
class RewiringSummaryPoint:
    """One complete ``(n, beta)`` cell in a rewiring summary."""

    collection_id: str
    n: int
    beta_key: int
    beta: float
    n_graphs: int
    mean_displaced: float
    sem_displaced: float
    mean_operations: float
    sem_operations: float
    mean_restored: float
    sem_restored: float
    mean_displaced_fraction: float
    sem_displaced_fraction: float


@dataclass(frozen=True, slots=True)
class RewiringPlotPaths:
    """Atomically published rewiring-figure paths."""

    png: Path
    pdf: Path


def _parse_integer(raw: str | None, name: str, row_number: int, *, minimum: int) -> int:
    if raw is None or _NONNEGATIVE_INTEGER.fullmatch(raw) is None:
        raise ValueError(f"row {row_number}: {name} must be a canonical nonnegative integer")
    value = int(raw)
    if value < minimum:
        raise ValueError(f"row {row_number}: {name} must be >= {minimum}; got {value}")
    return value


def _parse_finite(raw: str | None, name: str, row_number: int) -> float:
    if raw is None or not raw or raw.strip() != raw:
        raise ValueError(f"row {row_number}: {name} must be a finite number")
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"row {row_number}: {name} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"row {row_number}: {name} must be finite")
    return value


def _close(left: float, right: float, *, scale: float = 1.0) -> bool:
    return math.isclose(left, right, rel_tol=2e-12, abs_tol=2e-12 * scale)


def _parse_row(row: dict[str | None, str | None], row_number: int) -> RewiringSummaryPoint:
    if None in row:
        raise ValueError(f"row {row_number}: too many CSV fields")
    collection_id = row["collection_id"]
    if collection_id is None or not collection_id or collection_id.strip() != collection_id:
        raise ValueError(f"row {row_number}: collection_id must be nonempty and canonical")

    n = _parse_integer(row["n"], "n", row_number, minimum=3)
    beta_key = _parse_integer(row["beta_key"], "beta_key", row_number, minimum=0)
    try:
        canonical_beta = beta_from_key(beta_key)
    except (TypeError, ValueError) as error:
        raise ValueError(f"row {row_number}: invalid beta_key={beta_key}") from error
    beta = _parse_finite(row["beta"], "beta", row_number)
    if canonical_beta_key(beta) != beta_key or beta != canonical_beta:
        raise ValueError(
            f"row {row_number}: beta={beta!r} is not the canonical value for beta_key={beta_key}"
        )

    n_graphs = _parse_integer(row["n_graphs"], "n_graphs", row_number, minimum=2)
    numeric_names = SUMMARY_FIELDS[5:]
    values = {name: _parse_finite(row[name], name, row_number) for name in numeric_names}
    for name, value in values.items():
        if value < 0.0:
            raise ValueError(f"row {row_number}: {name} must be nonnegative")

    edge_count = 2 * n
    for name in ("mean_displaced", "mean_operations", "mean_restored"):
        if values[name] > edge_count:
            raise ValueError(f"row {row_number}: {name} exceeds the {edge_count} lattice edges")
    if values["mean_displaced_fraction"] > 1.0:
        raise ValueError(f"row {row_number}: mean_displaced_fraction must be <= 1")
    if not _close(
        values["mean_operations"] - values["mean_restored"],
        values["mean_displaced"],
        scale=edge_count,
    ):
        raise ValueError(
            f"row {row_number}: mean_operations - mean_restored must equal mean_displaced"
        )
    if not _close(
        values["mean_displaced_fraction"],
        values["mean_displaced"] / edge_count,
    ):
        raise ValueError(
            f"row {row_number}: mean_displaced_fraction is inconsistent with mean_displaced/(2n)"
        )
    if not _close(
        values["sem_displaced_fraction"],
        values["sem_displaced"] / edge_count,
    ):
        raise ValueError(
            f"row {row_number}: sem_displaced_fraction is inconsistent with sem_displaced/(2n)"
        )
    if beta_key == 0 and any(values[name] != 0.0 for name in numeric_names):
        raise ValueError(f"row {row_number}: all beta=0 rewiring statistics must be zero")

    return RewiringSummaryPoint(
        collection_id=collection_id,
        n=n,
        beta_key=beta_key,
        beta=beta,
        n_graphs=n_graphs,
        **values,
    )


def _canonical_expected_grid(
    points: Sequence[RewiringSummaryPoint],
    *,
    expected_sizes: Sequence[int] | None,
    expected_beta_keys: Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    sizes = (
        tuple(expected_sizes)
        if expected_sizes is not None
        else tuple(sorted({point.n for point in points}))
    )
    beta_keys = (
        tuple(expected_beta_keys)
        if expected_beta_keys is not None
        else tuple(sorted({point.beta_key for point in points}))
    )
    if not sizes or sizes != tuple(sorted(set(sizes))):
        raise ValueError("expected_sizes must be nonempty, unique, and strictly increasing")
    if not beta_keys or beta_keys != tuple(sorted(set(beta_keys))):
        raise ValueError("expected_beta_keys must be nonempty, unique, and strictly increasing")
    for key in beta_keys:
        beta_from_key(key)
    return sizes, beta_keys


def read_rewiring_summary(
    path: Path,
    *,
    expected_sizes: Sequence[int] | None = None,
    expected_beta_keys: Sequence[int] | None = None,
    expected_n_graphs: int | None = None,
) -> tuple[RewiringSummaryPoint, ...]:
    """Read and validate a complete, canonically ordered rectangular summary."""
    summary_path = Path(path)
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(SUMMARY_FIELDS):
            raise ValueError(
                "rewiring summary schema/order differs from the required fields: "
                + ",".join(SUMMARY_FIELDS)
            )
        points = tuple(_parse_row(row, row_number) for row_number, row in enumerate(reader, 2))
    if not points:
        raise ValueError("rewiring summary contains no data rows")

    collection_ids = {point.collection_id for point in points}
    if len(collection_ids) != 1:
        raise ValueError("rewiring summary mixes collection_id values")
    graph_counts = {point.n_graphs for point in points}
    if len(graph_counts) != 1:
        raise ValueError("rewiring summary mixes n_graphs values across cells")
    if expected_n_graphs is not None:
        if (
            isinstance(expected_n_graphs, bool)
            or not isinstance(expected_n_graphs, int)
            or expected_n_graphs < 2
        ):
            raise ValueError("expected_n_graphs must be an integer >= 2")
        if graph_counts != {expected_n_graphs}:
            raise ValueError(
                f"rewiring summary requires n_graphs={expected_n_graphs}; found {sorted(graph_counts)}"
            )

    sizes, beta_keys = _canonical_expected_grid(
        points,
        expected_sizes=expected_sizes,
        expected_beta_keys=expected_beta_keys,
    )
    expected_order = tuple((n, beta_key) for n in sizes for beta_key in beta_keys)
    actual_order = tuple((point.n, point.beta_key) for point in points)
    if actual_order != expected_order:
        duplicate_count = len(actual_order) - len(set(actual_order))
        if duplicate_count:
            raise ValueError(
                f"rewiring summary contains {duplicate_count} duplicate (n, beta) rows"
            )
        missing = sorted(set(expected_order) - set(actual_order))
        extra = sorted(set(actual_order) - set(expected_order))
        if missing or extra:
            raise ValueError(
                "rewiring summary is not a complete rectangular grid; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        raise ValueError("rewiring summary rows are not in canonical n-major, beta-major order")
    return points


def _beta_ticks(points: Sequence[RewiringSummaryPoint]) -> tuple[list[float], list[str]]:
    maximum = max(point.beta for point in points)
    positives = sorted({point.beta for point in points if point.beta > 0.0})
    if not positives:
        return [0.0], ["0"]
    minimum = positives[0]
    candidates = [0.0, minimum, 0.01, 0.03, 0.1, 0.3, 1.0, maximum]
    ticks: list[float] = []
    for candidate in candidates:
        if (
            candidate <= maximum
            and (candidate == 0.0 or candidate >= minimum)
            and not any(_close(candidate, tick) for tick in ticks)
        ):
            ticks.append(candidate)
    ticks.sort()
    labels = []
    for tick in ticks:
        label = "0" if tick == 0.0 else format(tick, ".3g")
        if 0.0 < tick < 1.0:
            label = label.removeprefix("0")
        labels.append(label)
    return ticks, labels


def _build_figure(points: Sequence[RewiringSummaryPoint]):
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with plt.rc_context(mpl.rcParamsDefault):
        figure, (raw_axis, fraction_axis) = plt.subplots(
            1,
            2,
            figsize=(7.3, 3.05),
            constrained_layout=True,
            sharex=True,
        )
        colors = plt.get_cmap("tab10")
        sizes = sorted({point.n for point in points})
        for color_index, n in enumerate(sizes):
            cell = [point for point in points if point.n == n]
            beta = [point.beta for point in cell]
            color = colors(color_index % 10)
            raw_axis.errorbar(
                beta,
                [point.mean_displaced for point in cell],
                yerr=[point.sem_displaced for point in cell],
                color=color,
                marker="o",
                markersize=2.7,
                linewidth=0.8,
                elinewidth=0.65,
                capsize=1.5,
                capthick=0.65,
                label=rf"$n={n}$",
                zorder=3,
            )
            raw_axis.plot(
                beta,
                [2.0 * n * value for value in beta],
                color=color,
                linestyle=":",
                linewidth=0.65,
                alpha=0.32,
                zorder=1,
            )
            fraction_axis.errorbar(
                beta,
                [point.mean_displaced_fraction for point in cell],
                yerr=[point.sem_displaced_fraction for point in cell],
                color=color,
                marker="o",
                markersize=2.7,
                linewidth=0.8,
                elinewidth=0.65,
                capsize=1.5,
                capthick=0.65,
                zorder=3,
            )

        all_beta = sorted({point.beta for point in points})
        fraction_axis.plot(
            all_beta,
            all_beta,
            color="0.25",
            linestyle=":",
            linewidth=0.75,
            alpha=0.55,
            zorder=1,
        )
        minimum_positive = min((beta for beta in all_beta if beta > 0.0), default=0.005)
        linear_threshold = minimum_positive / 2.0
        ticks, tick_labels = _beta_ticks(points)
        for axis in (raw_axis, fraction_axis):
            axis.set_xscale("symlog", linthresh=linear_threshold, linscale=0.6)
            axis.set_xlim(-0.12 * linear_threshold, max(all_beta) * 1.08)
            axis.set_xticks(ticks)
            axis.set_xticklabels(tick_labels)
            axis.set_xlabel(r"Rewiring probability $\beta$")
            axis.grid(axis="y", linewidth=0.4, alpha=0.22)
            axis.tick_params(axis="both", labelsize=7.3)

        raw_axis.set_ylabel(r"Mean final off-lattice edges $N_{\rm rew}$")
        raw_axis.set_ylim(bottom=0.0)
        raw_axis.text(
            0.98,
            0.04,
            r"dotted: $\langle N_{\rm op}\rangle=2n\beta$",
            transform=raw_axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.38",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
        )
        fraction_axis.set_ylabel(r"Mean final off-lattice fraction $N_{\rm rew}/(2n)$")
        fraction_axis.set_ylim(0.0, 1.03)
        fraction_axis.text(
            0.98,
            0.04,
            r"dotted: $\langle N_{\rm op}\rangle/(2n)=\beta$",
            transform=fraction_axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="0.38",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
        )
        raw_axis.legend(
            loc="upper left",
            frameon=False,
            fontsize=7,
            ncol=2,
            columnspacing=0.8,
            handlelength=1.4,
            handletextpad=0.35,
        )
        return figure


def _atomic_save_pair(figure, png_path: Path, pdf_path: Path) -> None:
    import matplotlib as mpl

    png_path.parent.mkdir(parents=True, exist_ok=True)
    if png_path.parent.resolve() != pdf_path.parent.resolve():
        raise ValueError("PNG and PDF outputs must share one directory")
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    png_temporary = png_path.with_name(f".{png_path.name}.{token}.tmp")
    pdf_temporary = pdf_path.with_name(f".{pdf_path.name}.{token}.tmp")
    try:
        with mpl.rc_context(mpl.rcParamsDefault):
            figure.savefig(
                png_temporary,
                format="png",
                dpi=300,
                bbox_inches="tight",
                metadata={"Software": "SparseGF2 rewiring analysis"},
            )
            figure.savefig(
                pdf_temporary,
                format="pdf",
                bbox_inches="tight",
                metadata={
                    "Creator": "SparseGF2 rewiring analysis",
                    "Producer": "Matplotlib",
                    "CreationDate": None,
                    "ModDate": None,
                },
            )
        os.replace(png_temporary, png_path)
        os.replace(pdf_temporary, pdf_path)
    finally:
        png_temporary.unlink(missing_ok=True)
        pdf_temporary.unlink(missing_ok=True)


def plot_rewiring_summary(
    summary_csv: Path,
    output_dir: Path,
    *,
    stem: str = "rewired_edges_vs_beta",
    expected_sizes: Sequence[int] | None = None,
    expected_beta_keys: Sequence[int] | None = None,
    expected_n_graphs: int | None = None,
) -> RewiringPlotPaths:
    """Validate one summary and atomically write side-by-side PNG and PDF plots."""
    if not stem or Path(stem).name != stem or Path(stem).suffix:
        raise ValueError("stem must be a nonempty extension-free file name")
    points = read_rewiring_summary(
        Path(summary_csv),
        expected_sizes=expected_sizes,
        expected_beta_keys=expected_beta_keys,
        expected_n_graphs=expected_n_graphs,
    )
    output = Path(output_dir)
    paths = RewiringPlotPaths(output / f"{stem}.png", output / f"{stem}.pdf")
    figure = _build_figure(points)
    try:
        _atomic_save_pair(figure, paths.png, paths.pdf)
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--stem", default="rewired_edges_vs_beta")
    parser.add_argument("--expected-n-graphs", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = plot_rewiring_summary(
        args.summary_csv,
        args.output_dir,
        stem=args.stem,
        expected_n_graphs=args.expected_n_graphs,
    )
    print(f"PNG: {paths.png}")
    print(f"PDF: {paths.pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "RewiringPlotPaths",
    "RewiringSummaryPoint",
    "SUMMARY_FIELDS",
    "main",
    "plot_rewiring_summary",
    "read_rewiring_summary",
]
