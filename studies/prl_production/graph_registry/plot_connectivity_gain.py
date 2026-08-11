"""Plot cumulative algebraic-connectivity gain from a strict production summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .spec import beta_from_key, canonical_beta_key, production_beta_keys

SUMMARY_FIELDS = (
    "collection_id",
    "set_size",
    "size_set",
    "beta_key",
    "beta",
    "n_graphs_per_cell",
    "g_lambda",
    "g_lambda_sem",
    "log_g_lambda",
    "log_g_lambda_sem",
    "ci68_low",
    "ci68_high",
)

NESTED_SIZE_SETS = (
    (64,),
    (64, 128),
    (64, 128, 192),
    (64, 128, 192, 256),
)
REPRESENTATIVE_BETA_TARGETS = (0.005, 0.01, 0.03, 0.1, 0.3, 1.0)

_NONNEGATIVE_INTEGER = re.compile(r"0|[1-9][0-9]*")


@dataclass(frozen=True, slots=True)
class ConnectivityGainPoint:
    """One cumulative-size estimate at a single canonical rewiring probability."""

    collection_id: str
    set_size: int
    size_set: tuple[int, ...]
    beta_key: int
    beta: float
    n_graphs_per_cell: int
    g_lambda: float
    g_lambda_sem: float
    log_g_lambda: float
    log_g_lambda_sem: float
    ci68_low: float
    ci68_high: float


@dataclass(frozen=True, slots=True)
class ConnectivityGainPlotPaths:
    """Atomically published cumulative connectivity-gain figure paths."""

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
    return math.isclose(left, right, rel_tol=5e-11, abs_tol=5e-12 * scale)


def _parse_size_set(
    raw: str | None,
    set_size: int,
    row_number: int,
    expected_size_sets: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    if raw is None or not raw or raw.strip() != raw:
        raise ValueError(f"row {row_number}: size_set must be canonical compact JSON")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"row {row_number}: size_set must be canonical compact JSON") from error
    if not isinstance(decoded, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in decoded
    ):
        raise ValueError(f"row {row_number}: size_set must be a JSON list of integers")
    values = tuple(decoded)
    canonical = json.dumps(decoded, separators=(",", ":"))
    if raw != canonical:
        raise ValueError(f"row {row_number}: size_set must be canonical compact JSON")
    expected = expected_size_sets[set_size - 1]
    if values != expected:
        expected_text = json.dumps(expected, separators=(",", ":"))
        raise ValueError(f"row {row_number}: set_size={set_size} requires size_set={expected_text}")
    return values


def _parse_row(
    row: dict[str | None, str | None],
    row_number: int,
    expected_size_sets: tuple[tuple[int, ...], ...],
) -> ConnectivityGainPoint:
    if None in row:
        raise ValueError(f"row {row_number}: too many CSV fields")
    collection_id = row["collection_id"]
    if collection_id is None or not collection_id or collection_id.strip() != collection_id:
        raise ValueError(f"row {row_number}: collection_id must be nonempty and canonical")

    set_size = _parse_integer(row["set_size"], "set_size", row_number, minimum=1)
    if set_size > len(expected_size_sets):
        raise ValueError(f"row {row_number}: set_size must be in [1, {len(expected_size_sets)}]")
    size_set = _parse_size_set(row["size_set"], set_size, row_number, expected_size_sets)

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

    n_graphs = _parse_integer(row["n_graphs_per_cell"], "n_graphs_per_cell", row_number, minimum=2)
    numeric_names = SUMMARY_FIELDS[6:]
    values = {name: _parse_finite(row[name], name, row_number) for name in numeric_names}
    g_lambda = values["g_lambda"]
    g_sem = values["g_lambda_sem"]
    log_g = values["log_g_lambda"]
    log_sem = values["log_g_lambda_sem"]
    ci_low = values["ci68_low"]
    ci_high = values["ci68_high"]

    if g_lambda <= 0.0 or ci_low <= 0.0 or ci_high <= 0.0:
        raise ValueError(f"row {row_number}: gain and CI endpoints must be positive")
    if g_sem < 0.0 or log_sem < 0.0:
        raise ValueError(f"row {row_number}: SEM values must be nonnegative")
    if g_sem >= g_lambda and g_sem != 0.0:
        raise ValueError(f"row {row_number}: g_lambda_sem must be smaller than g_lambda")
    if not _close(log_g, math.log(g_lambda), scale=max(1.0, abs(log_g))):
        raise ValueError(f"row {row_number}: log_g_lambda is inconsistent with ln(g_lambda)")
    if not _close(g_sem, g_lambda * log_sem, scale=max(1.0, g_sem)):
        raise ValueError(f"row {row_number}: g_lambda_sem is inconsistent with the log-scale SEM")
    if not ci_low <= g_lambda <= ci_high:
        raise ValueError(f"row {row_number}: ci68_low <= g_lambda <= ci68_high is required")
    if log_sem == 0.0:
        if g_sem != 0.0 or ci_low != g_lambda or ci_high != g_lambda:
            raise ValueError(f"row {row_number}: zero uncertainty requires a collapsed CI")
    elif not ci_low < g_lambda < ci_high:
        raise ValueError(f"row {row_number}: positive uncertainty requires a nondegenerate CI")

    if beta_key == 0 and (
        g_lambda != 1.0
        or g_sem != 0.0
        or log_g != 0.0
        or log_sem != 0.0
        or ci_low != 1.0
        or ci_high != 1.0
    ):
        raise ValueError(f"row {row_number}: beta=0 gain must be exactly one with zero uncertainty")

    return ConnectivityGainPoint(
        collection_id=collection_id,
        set_size=set_size,
        size_set=size_set,
        beta_key=beta_key,
        beta=beta,
        n_graphs_per_cell=n_graphs,
        **values,
    )


def _canonical_expected_size_sets(
    expected_size_sets: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...]:
    raw_sets = NESTED_SIZE_SETS if expected_size_sets is None else expected_size_sets
    size_sets: list[tuple[int, ...]] = []
    for set_size, raw_values in enumerate(raw_sets, 1):
        values = tuple(raw_values)
        if (
            len(values) != set_size
            or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
            or values != tuple(sorted(set(values)))
            or any(value < 3 for value in values)
        ):
            raise ValueError(
                "expected_size_sets must contain strictly increasing integer sets whose "
                "length equals their one-based position"
            )
        if size_sets and values[:-1] != size_sets[-1]:
            raise ValueError("expected_size_sets must be cumulative nested prefixes")
        size_sets.append(values)
    if not size_sets:
        raise ValueError("expected_size_sets must be nonempty")
    return tuple(size_sets)


def _canonical_expected_beta_keys(expected_beta_keys: Sequence[int] | None) -> tuple[int, ...]:
    raw_keys = production_beta_keys() if expected_beta_keys is None else expected_beta_keys
    keys = tuple(raw_keys)
    if (
        len(keys) < 2
        or any(isinstance(key, bool) or not isinstance(key, int) for key in keys)
        or keys != tuple(sorted(set(keys)))
        or keys[0] != 0
    ):
        raise ValueError(
            "expected_beta_keys must begin at zero and contain at least one positive, "
            "strictly increasing canonical integer key"
        )
    for key in keys:
        beta_from_key(key)
    return keys


def read_connectivity_gain_summary(
    path: Path,
    *,
    expected_size_sets: Sequence[Sequence[int]] | None = None,
    expected_beta_keys: Sequence[int] | None = None,
    expected_collection_id: str | None = None,
    expected_n_graphs_per_cell: int | None = None,
) -> tuple[ConnectivityGainPoint, ...]:
    """Read a strict rectangular grid, defaulting to the production sets and betas."""
    size_sets = _canonical_expected_size_sets(expected_size_sets)
    beta_keys = _canonical_expected_beta_keys(expected_beta_keys)
    summary_path = Path(path)
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(SUMMARY_FIELDS):
            raise ValueError(
                "connectivity-gain summary schema/order differs from the required fields: "
                + ",".join(SUMMARY_FIELDS)
            )
        points = tuple(
            _parse_row(row, row_number, size_sets) for row_number, row in enumerate(reader, 2)
        )
    if not points:
        raise ValueError("connectivity-gain summary contains no data rows")

    collection_ids = {point.collection_id for point in points}
    if len(collection_ids) != 1:
        raise ValueError("connectivity-gain summary mixes collection_id values")
    if expected_collection_id is not None:
        if (
            not isinstance(expected_collection_id, str)
            or not expected_collection_id
            or expected_collection_id.strip() != expected_collection_id
        ):
            raise ValueError("expected_collection_id must be nonempty canonical text")
        if collection_ids != {expected_collection_id}:
            raise ValueError(
                f"connectivity-gain summary requires collection_id={expected_collection_id!r}; "
                f"found {sorted(collection_ids)!r}"
            )

    graph_counts = {point.n_graphs_per_cell for point in points}
    if len(graph_counts) != 1:
        raise ValueError("connectivity-gain summary mixes n_graphs_per_cell values")
    if expected_n_graphs_per_cell is not None:
        if (
            isinstance(expected_n_graphs_per_cell, bool)
            or not isinstance(expected_n_graphs_per_cell, int)
            or expected_n_graphs_per_cell < 2
        ):
            raise ValueError("expected_n_graphs_per_cell must be an integer >= 2")
        if graph_counts != {expected_n_graphs_per_cell}:
            raise ValueError(
                "connectivity-gain summary requires "
                f"n_graphs_per_cell={expected_n_graphs_per_cell}; found {sorted(graph_counts)}"
            )

    expected_order = tuple(
        (set_size, beta_key) for set_size in range(1, len(size_sets) + 1) for beta_key in beta_keys
    )
    actual_order = tuple((point.set_size, point.beta_key) for point in points)
    if actual_order != expected_order:
        duplicate_count = len(actual_order) - len(set(actual_order))
        if duplicate_count:
            raise ValueError(
                f"connectivity-gain summary contains {duplicate_count} duplicate grid rows"
            )
        missing = sorted(set(expected_order) - set(actual_order))
        extra = sorted(set(actual_order) - set(expected_order))
        if missing or extra:
            raise ValueError(
                "connectivity-gain summary is not the complete expected rectangular grid; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        raise ValueError(
            "connectivity-gain summary rows are not in canonical set-major, beta-major order"
        )
    return points


def _add_horizontal_axis_break(left_axis, right_axis) -> None:
    """Mark the omitted interval between beta zero and the positive log axis."""
    left_axis.spines["right"].set_visible(False)
    right_axis.spines["left"].set_visible(False)
    left_axis.tick_params(right=False)
    right_axis.tick_params(left=False)
    marker = [(-0.45, -1.0), (0.45, 1.0)]
    for axis, x_value, side in (
        (left_axis, 1.0, "left"),
        (right_axis, 0.0, "right"),
    ):
        for y_value, edge in ((0.0, "bottom"), (1.0, "top")):
            artist = axis.plot(
                [x_value],
                [y_value],
                marker=marker,
                markersize=5.0,
                markeredgewidth=0.75,
                color="black",
                linestyle="none",
                transform=axis.transAxes,
                clip_on=False,
                zorder=20,
            )[0]
            artist.set_gid(f"beta-axis-break-{side}-{edge}")


def _representative_beta_keys(points: Sequence[ConnectivityGainPoint]) -> tuple[int, ...]:
    positive = {point.beta_key: point.beta for point in points if point.beta > 0.0}
    selected = [
        min(positive, key=lambda key: (abs(math.log(positive[key] / target)), key))
        for target in REPRESENTATIVE_BETA_TARGETS
    ]
    return tuple(dict.fromkeys(selected))


def _build_figure(points: Sequence[ConnectivityGainPoint]):
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with plt.rc_context(mpl.rcParamsDefault):
        plt.rcParams.update({"font.family": "sans-serif", "mathtext.fontset": "dejavusans"})
        figure = plt.figure(figsize=(7.3, 3.25), constrained_layout=False)
        figure.subplots_adjust(left=0.09, right=0.985, bottom=0.17, top=0.96)
        outer = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.0), wspace=0.34)
        left = outer[0, 0].subgridspec(1, 2, width_ratios=(0.095, 1.0), wspace=0.025)
        zero_axis = figure.add_subplot(left[0, 0])
        positive_axis = figure.add_subplot(left[0, 1], sharey=zero_axis)
        cardinality_axis = figure.add_subplot(outer[0, 1])

        size_sets = tuple(dict.fromkeys(point.size_set for point in points))
        set_count = len(size_sets)
        color_positions = [
            0.12 + 0.78 * index / max(1, set_count - 1) for index in range(set_count)
        ]
        colors = plt.get_cmap("viridis")(color_positions)
        markers = ("o", "s", "^", "D", "v", "P", "X")
        by_set = {
            set_size: [point for point in points if point.set_size == set_size]
            for set_size in range(1, set_count + 1)
        }
        for set_size, color, marker in zip(
            range(1, set_count + 1), colors, markers[:set_count], strict=True
        ):
            cell = by_set[set_size]
            zero = cell[0]
            zero_container = zero_axis.errorbar(
                [0.0],
                [zero.g_lambda],
                yerr=[zero.g_lambda_sem],
                fmt=marker,
                color=color,
                markersize=3.5,
                linewidth=0.8,
                zorder=3 + set_size,
            )
            zero_container.lines[0].set_gid(f"connectivity-gain-beta-m{set_size}-zero")
            positive = cell[1:]
            container = positive_axis.errorbar(
                [point.beta for point in positive],
                [point.g_lambda for point in positive],
                yerr=[point.g_lambda_sem for point in positive],
                fmt=f"{marker}-",
                color=color,
                markersize=2.8,
                markeredgewidth=0.45,
                linewidth=0.9,
                elinewidth=0.55,
                capsize=1.2,
                capthick=0.55,
                label=rf"$m={set_size}$",
                zorder=3 + set_size,
            )
            container.lines[0].set_gid(f"connectivity-gain-beta-m{set_size}")

        positive_beta = [point.beta for point in by_set[1][1:]]
        zero_axis.set_xlim(-0.45, 0.45)
        zero_axis.set_xticks([0.0], ["0"])
        positive_axis.set_xscale("log")
        positive_axis.set_xlim(min(positive_beta) / 1.18, max(positive_beta) * 1.08)
        beta_ticks = [0.005, 0.01, 0.03, 0.1, 0.3, 1.0]
        positive_axis.set_xticks(beta_ticks, [".005", ".01", ".03", ".1", ".3", "1"])
        positive_axis.minorticks_off()
        positive_axis.tick_params(axis="y", labelleft=False)
        zero_axis.set_ylabel(r"Cumulative gain $g_\lambda^{(m)}(\beta)$")
        positive_axis.set_xlabel(r"Rewiring probability $\beta$")
        positive_axis.legend(
            title=r"Cumulative sets $\mathcal{N}_m$",
            loc="upper left",
            frameon=False,
            fontsize=7.0,
            title_fontsize=7.0,
            ncol=2,
            handlelength=1.4,
            handletextpad=0.35,
            columnspacing=0.7,
        )
        _add_horizontal_axis_break(zero_axis, positive_axis)

        representative_keys = _representative_beta_keys(points)
        beta_colors = plt.get_cmap("plasma")(
            [
                index / max(1, len(representative_keys) - 1)
                for index in range(len(representative_keys))
            ]
        )
        point_lookup = {(point.set_size, point.beta_key): point for point in points}
        for beta_key, color, marker in zip(
            representative_keys,
            beta_colors,
            ("o", "s", "^", "D", "v", "P")[: len(representative_keys)],
            strict=True,
        ):
            selected = [point_lookup[(set_size, beta_key)] for set_size in range(1, set_count + 1)]
            container = cardinality_axis.errorbar(
                range(1, set_count + 1),
                [point.g_lambda for point in selected],
                yerr=[point.g_lambda_sem for point in selected],
                fmt=f"{marker}-",
                color=color,
                markersize=3.2,
                markeredgewidth=0.45,
                linewidth=0.9,
                elinewidth=0.55,
                capsize=1.2,
                capthick=0.55,
                label=rf"$\beta={selected[0].beta:.3g}$",
                zorder=3,
            )
            container.lines[0].set_gid(f"connectivity-gain-cardinality-beta-{beta_key}")

        all_ci = [value for point in points for value in (point.ci68_low, point.ci68_high)]
        y_min = min(all_ci)
        y_max = max(all_ci)
        lower = min(0.96, y_min / 1.08)
        upper = max(1.04, y_max * 1.12)
        for axis in (zero_axis, positive_axis, cardinality_axis):
            axis.set_yscale("log")
            axis.set_ylim(lower, upper)
            axis.axhline(1.0, color="0.55", linestyle=":", linewidth=0.75, zorder=1)
            axis.grid(axis="y", which="both", linewidth=0.4, alpha=0.22)
            axis.tick_params(axis="both", labelsize=7.5)
        cardinality_axis.set_xticks(range(1, set_count + 1))
        cardinality_axis.set_xlabel(r"Number of sizes $m=|\mathcal{N}_m|$")
        cardinality_axis.set_ylabel(r"Cumulative gain $g_\lambda^{(m)}(\beta)$")
        cardinality_axis.legend(
            loc="upper left",
            frameon=False,
            fontsize=6.7,
            ncol=2,
            handlelength=1.35,
            handletextpad=0.3,
            columnspacing=0.55,
        )
        figure.text(
            zero_axis.get_position().x0,
            0.982,
            "(a)",
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
        figure.text(
            cardinality_axis.get_position().x0,
            0.982,
            "(b)",
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
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
                metadata={"Software": "SparseGF2 connectivity-gain analysis"},
            )
            figure.savefig(
                pdf_temporary,
                format="pdf",
                bbox_inches="tight",
                metadata={
                    "Creator": "SparseGF2 connectivity-gain analysis",
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


def plot_connectivity_gain(
    summary_csv: Path,
    output_dir: Path,
    *,
    stem: str = "algebraic_connectivity_gain_convergence",
    expected_size_sets: Sequence[Sequence[int]] | None = None,
    expected_beta_keys: Sequence[int] | None = None,
    expected_collection_id: str | None = None,
    expected_n_graphs_per_cell: int | None = None,
) -> ConnectivityGainPlotPaths:
    """Validate a cumulative-gain summary and atomically publish PNG and PDF."""
    if not stem or Path(stem).name != stem or Path(stem).suffix:
        raise ValueError("stem must be a nonempty extension-free file name")
    points = read_connectivity_gain_summary(
        Path(summary_csv),
        expected_size_sets=expected_size_sets,
        expected_beta_keys=expected_beta_keys,
        expected_collection_id=expected_collection_id,
        expected_n_graphs_per_cell=expected_n_graphs_per_cell,
    )
    output = Path(output_dir)
    paths = ConnectivityGainPlotPaths(output / f"{stem}.png", output / f"{stem}.pdf")
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
    parser.add_argument("--stem", default="algebraic_connectivity_gain_convergence")
    parser.add_argument("--expected-collection-id")
    parser.add_argument("--expected-n-graphs-per-cell", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = plot_connectivity_gain(
        args.summary_csv,
        args.output_dir,
        stem=args.stem,
        expected_collection_id=args.expected_collection_id,
        expected_n_graphs_per_cell=args.expected_n_graphs_per_cell,
    )
    print(f"PNG: {paths.png}")
    print(f"PDF: {paths.pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ConnectivityGainPlotPaths",
    "ConnectivityGainPoint",
    "NESTED_SIZE_SETS",
    "REPRESENTATIVE_BETA_TARGETS",
    "SUMMARY_FIELDS",
    "main",
    "plot_connectivity_gain",
    "read_connectivity_gain_summary",
]
