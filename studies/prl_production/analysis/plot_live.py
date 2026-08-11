"""Plot validated preliminary purification-time curves from a live summary CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
from studies.prl_production.campaign import TMAX_FACTOR

if TYPE_CHECKING:
    from matplotlib.figure import Figure

matplotlib.use("Agg")

PRL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_ROOT = PRL_ROOT / "analysis" / "runs"
DEFAULT_FIGURE_ROOT = PRL_ROOT / "figures" / "raw"
DEFAULT_MINIMUM_POINTS = 5

_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{16}")
_UNSIGNED_INTEGER_PATTERN = re.compile(r"0|[1-9][0-9]*")
_BASE_FIELDS = {
    "analysis_status",
    "run_id",
    "point_index",
    "n",
    "beta",
    "beta_key",
    "p",
    "p_key",
    "t_max",
    "n_trajectories",
    "n_events",
    "n_censored",
    "event_fraction",
    "median_tau_p",
    "median_resolved",
    "survival_at_cap",
}
_BOOTSTRAP_FIELDS = {
    "median_ci_lower",
    "median_ci_upper",
    "median_ci_resolved",
    "bootstrap_resolved_fraction",
    "bootstrap_resamples",
    "bootstrap_confidence",
}


@dataclass(frozen=True, slots=True)
class LiveSummaryPoint:
    """One strictly validated, fully complete point summary."""

    point_index: int
    n: int
    beta: float
    beta_key: int
    p: float
    p_key: int
    t_max: int
    n_trajectories: int
    n_events: int
    n_censored: int
    event_fraction: float
    median_tau_p: int | None
    median_resolved: bool
    survival_at_cap: float
    median_ci_lower: int | None = None
    median_ci_upper: int | None = None
    median_ci_resolved: bool = False
    bootstrap_resolved_fraction: float | None = None
    bootstrap_resamples: int = 0
    bootstrap_confidence: float | None = None

    @property
    def upper_interval_cap_limited(self) -> bool:
        """Whether a resolved median has a bootstrapped upper bound beyond the cap."""
        return (
            self.median_resolved and self.bootstrap_resamples > 0 and self.median_ci_upper is None
        )


@dataclass(frozen=True, slots=True)
class LivePlotResult:
    """Artifacts produced from one immutable read of a live summary."""

    run_id: str
    summary_path: Path
    summary_sha256: str
    output_dir: Path
    index_path: Path
    plot_paths: tuple[Path, ...]
    plotted_betas: tuple[float, ...]
    skipped_betas: tuple[float, ...]
    validated_rows: int


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def _parse_integer(raw: str, name: str, *, minimum: int = 0) -> int:
    if _UNSIGNED_INTEGER_PATTERN.fullmatch(raw) is None:
        raise ValueError(f"{name} must be an unsigned integer; got {raw!r}")
    result = int(raw)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {result}")
    return result


def _parse_optional_integer(raw: str, name: str) -> int | None:
    if raw == "":
        return None
    return _parse_integer(raw, name, minimum=1)


def _parse_flag(raw: str, name: str) -> bool:
    if raw not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1; got {raw!r}")
    return raw == "1"


def _parse_float(raw: str, name: str, *, minimum: float, maximum: float) -> float:
    if raw == "":
        raise ValueError(f"{name} must not be blank")
    try:
        result = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a real number; got {raw!r}") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]; got {raw!r}")
    return result


def _parse_optional_float(
    raw: str,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    if raw == "":
        return None
    return _parse_float(raw, name, minimum=minimum, maximum=maximum)


def _row_value(row: dict[str, str | None], field: str, row_number: int) -> str:
    value = row.get(field)
    if value is None:
        raise ValueError(f"row {row_number}: missing {field}")
    if value != value.strip():
        raise ValueError(f"row {row_number}: {field} contains surrounding whitespace")
    return value


def _parse_row(
    row: dict[str, str | None],
    row_number: int,
    *,
    expected_run_id: str,
    has_bootstrap: bool,
) -> LiveSummaryPoint:
    prefix = f"row {row_number}"
    if _row_value(row, "analysis_status", row_number) != "PRELIMINARY":
        raise ValueError(f"{prefix}: analysis_status must be PRELIMINARY")
    row_run_id = _row_value(row, "run_id", row_number)
    if row_run_id != expected_run_id:
        raise ValueError(f"{prefix}: run_id={row_run_id!r}, expected {expected_run_id!r}")

    point_index = _parse_integer(
        _row_value(row, "point_index", row_number), f"{prefix}.point_index"
    )
    n = _parse_integer(_row_value(row, "n", row_number), f"{prefix}.n", minimum=1)
    if n % 2:
        raise ValueError(f"{prefix}: n must be even; got {n}")
    beta = _parse_float(
        _row_value(row, "beta", row_number), f"{prefix}.beta", minimum=0.0, maximum=1.0
    )
    beta_key = _parse_integer(_row_value(row, "beta_key", row_number), f"{prefix}.beta_key")
    if beta_key != round(beta * 1_000_000_000):
        raise ValueError(f"{prefix}: beta and beta_key are inconsistent")
    p = _parse_float(_row_value(row, "p", row_number), f"{prefix}.p", minimum=0.0, maximum=1.0)
    p_key = _parse_integer(_row_value(row, "p_key", row_number), f"{prefix}.p_key")
    if p_key != round(p * 1_000_000):
        raise ValueError(f"{prefix}: p and p_key are inconsistent")
    t_max = _parse_integer(_row_value(row, "t_max", row_number), f"{prefix}.t_max", minimum=1)
    if t_max != TMAX_FACTOR * n:
        raise ValueError(f"{prefix}: t_max={t_max}, expected {TMAX_FACTOR * n}")

    n_trajectories = _parse_integer(
        _row_value(row, "n_trajectories", row_number),
        f"{prefix}.n_trajectories",
        minimum=1,
    )
    n_events = _parse_integer(_row_value(row, "n_events", row_number), f"{prefix}.n_events")
    n_censored = _parse_integer(_row_value(row, "n_censored", row_number), f"{prefix}.n_censored")
    if n_events + n_censored != n_trajectories:
        raise ValueError(f"{prefix}: event and censor counts do not sum to n_trajectories")
    event_fraction = _parse_float(
        _row_value(row, "event_fraction", row_number),
        f"{prefix}.event_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    if not math.isclose(
        event_fraction,
        n_events / n_trajectories,
        rel_tol=0.0,
        abs_tol=5e-12,
    ):
        raise ValueError(f"{prefix}: event_fraction disagrees with event counts")

    median_resolved = _parse_flag(
        _row_value(row, "median_resolved", row_number), f"{prefix}.median_resolved"
    )
    median_tau_p = _parse_optional_integer(
        _row_value(row, "median_tau_p", row_number), f"{prefix}.median_tau_p"
    )
    if median_resolved != (median_tau_p is not None):
        raise ValueError(f"{prefix}: median_resolved disagrees with median_tau_p")
    if median_tau_p is not None and median_tau_p > t_max:
        raise ValueError(f"{prefix}: median_tau_p exceeds t_max")
    survival_at_cap = _parse_float(
        _row_value(row, "survival_at_cap", row_number),
        f"{prefix}.survival_at_cap",
        minimum=0.0,
        maximum=1.0,
    )
    if median_resolved and survival_at_cap > 0.5 + 1e-12:
        raise ValueError(f"{prefix}: a resolved median requires survival_at_cap <= 0.5")
    if not median_resolved and survival_at_cap <= 0.5:
        raise ValueError(f"{prefix}: an unresolved median requires survival_at_cap > 0.5")

    median_ci_lower: int | None = None
    median_ci_upper: int | None = None
    median_ci_resolved = False
    bootstrap_resolved_fraction: float | None = None
    bootstrap_resamples = 0
    bootstrap_confidence: float | None = None
    if has_bootstrap:
        median_ci_lower = _parse_optional_integer(
            _row_value(row, "median_ci_lower", row_number), f"{prefix}.median_ci_lower"
        )
        median_ci_upper = _parse_optional_integer(
            _row_value(row, "median_ci_upper", row_number), f"{prefix}.median_ci_upper"
        )
        median_ci_resolved = _parse_flag(
            _row_value(row, "median_ci_resolved", row_number),
            f"{prefix}.median_ci_resolved",
        )
        bootstrap_resolved_fraction = _parse_optional_float(
            _row_value(row, "bootstrap_resolved_fraction", row_number),
            f"{prefix}.bootstrap_resolved_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        bootstrap_resamples = _parse_integer(
            _row_value(row, "bootstrap_resamples", row_number),
            f"{prefix}.bootstrap_resamples",
        )
        bootstrap_confidence = _parse_optional_float(
            _row_value(row, "bootstrap_confidence", row_number),
            f"{prefix}.bootstrap_confidence",
            minimum=0.0,
            maximum=1.0,
        )
        if bootstrap_confidence is not None and not 0.0 < bootstrap_confidence < 1.0:
            raise ValueError(f"{prefix}: bootstrap_confidence must lie strictly between 0 and 1")
        if bootstrap_resamples == 0:
            if (
                any(
                    value is not None
                    for value in (
                        median_ci_lower,
                        median_ci_upper,
                        bootstrap_resolved_fraction,
                        bootstrap_confidence,
                    )
                )
                or median_ci_resolved
            ):
                raise ValueError(f"{prefix}: disabled bootstrap fields must be blank or zero")
        else:
            if bootstrap_resolved_fraction is None or bootstrap_confidence is None:
                raise ValueError(f"{prefix}: enabled bootstrap metadata must not be blank")
            if median_ci_upper is not None and median_ci_lower is None:
                raise ValueError(
                    f"{prefix}: a finite upper confidence bound requires a lower bound"
                )
            if median_ci_resolved != (median_ci_lower is not None and median_ci_upper is not None):
                raise ValueError(f"{prefix}: median_ci_resolved disagrees with confidence bounds")
            if median_ci_resolved and not median_resolved:
                raise ValueError(f"{prefix}: a resolved confidence interval requires a median")
            if median_ci_lower is not None and median_ci_lower > t_max:
                raise ValueError(f"{prefix}: median_ci_lower exceeds t_max")
            if median_ci_upper is not None and median_ci_upper > t_max:
                raise ValueError(f"{prefix}: median_ci_upper exceeds t_max")
            if (
                median_ci_lower is not None
                and median_ci_upper is not None
                and median_ci_lower > median_ci_upper
            ):
                raise ValueError(f"{prefix}: confidence bounds are reversed")

    return LiveSummaryPoint(
        point_index=point_index,
        n=n,
        beta=beta,
        beta_key=beta_key,
        p=p,
        p_key=p_key,
        t_max=t_max,
        n_trajectories=n_trajectories,
        n_events=n_events,
        n_censored=n_censored,
        event_fraction=event_fraction,
        median_tau_p=median_tau_p,
        median_resolved=median_resolved,
        survival_at_cap=survival_at_cap,
        median_ci_lower=median_ci_lower,
        median_ci_upper=median_ci_upper,
        median_ci_resolved=median_ci_resolved,
        bootstrap_resolved_fraction=bootstrap_resolved_fraction,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence=bootstrap_confidence,
    )


def _read_summary(path: Path, *, expected_run_id: str) -> tuple[list[LiveSummaryPoint], str]:
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read point summary {path}: {exc}") from exc
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"point summary {path} is not valid UTF-8") from exc

    reader = csv.DictReader(text.splitlines())
    fields = reader.fieldnames
    if fields is None:
        raise ValueError(f"point summary {path} has no header")
    if len(fields) != len(set(fields)):
        raise ValueError(f"point summary {path} has duplicate header fields")
    field_set = set(fields)
    if field_set == _BASE_FIELDS:
        has_bootstrap = False
    elif field_set == _BASE_FIELDS | _BOOTSTRAP_FIELDS:
        has_bootstrap = True
    else:
        missing = sorted(_BASE_FIELDS - field_set)
        unexpected = sorted(field_set - (_BASE_FIELDS | _BOOTSTRAP_FIELDS))
        partial_bootstrap = sorted(field_set & _BOOTSTRAP_FIELDS)
        raise ValueError(
            f"point summary schema differs; missing={missing}, unexpected={unexpected}, "
            f"partial_bootstrap={partial_bootstrap}"
        )

    points: list[LiveSummaryPoint] = []
    for row_number, row in enumerate(reader, start=2):
        if None in row:
            raise ValueError(f"row {row_number}: too many CSV fields")
        points.append(
            _parse_row(
                row,
                row_number,
                expected_run_id=expected_run_id,
                has_bootstrap=has_bootstrap,
            )
        )

    point_indices = [point.point_index for point in points]
    if len(point_indices) != len(set(point_indices)):
        raise ValueError("point summary contains duplicate point_index values")
    canonical_points = [(point.beta_key, point.n, point.p_key) for point in points]
    if len(canonical_points) != len(set(canonical_points)):
        raise ValueError("point summary contains duplicate (beta, n, p) points")
    bootstrap_options = {
        (point.bootstrap_resamples, point.bootstrap_confidence) for point in points
    }
    if len(bootstrap_options) > 1:
        raise ValueError("point summary mixes different bootstrap configurations")
    return points, hashlib.sha256(raw_bytes).hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_figure(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        figure.savefig(temporary, format="png", dpi=240, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _beta_label(beta: float) -> str:
    return format(beta, ".9g")


def _plot_beta(points: list[LiveSummaryPoint], path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    by_n: dict[int, list[LiveSummaryPoint]] = defaultdict(list)
    for point in points:
        by_n[point.n].append(point)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    ):
        figure, axis = plt.subplots(figsize=(3.35, 2.55), constrained_layout=True)
        colors = plt.get_cmap("tab10")
        has_unresolved = False
        has_cap_limited = False
        all_positive_values: list[float] = []

        for color_index, n in enumerate(sorted(by_n)):
            size_points = sorted(by_n[n], key=lambda point: point.p_key)
            color = colors(color_index % 10)
            resolved_x = [point.p for point in size_points if point.median_resolved]
            resolved_y = [
                float(point.median_tau_p) for point in size_points if point.median_tau_p is not None
            ]
            if resolved_x:
                axis.plot(
                    resolved_x,
                    resolved_y,
                    color=color,
                    linewidth=0.75,
                    alpha=0.9,
                    label=rf"$n={n}$",
                    zorder=2,
                )
                all_positive_values.extend(resolved_y)
            else:
                axis.plot([], [], color=color, linewidth=0.75, label=rf"$n={n}$")

            for point in size_points:
                if not point.median_resolved:
                    has_unresolved = True
                    all_positive_values.append(float(point.t_max))
                    axis.scatter(
                        [point.p],
                        [point.t_max],
                        marker=r"$\uparrow$",
                        s=24,
                        color=color,
                        linewidths=0.7,
                        zorder=5,
                    )
                    continue

                assert point.median_tau_p is not None
                if point.median_ci_lower is not None and point.median_ci_upper is not None:
                    lower = float(point.median_ci_lower)
                    upper = float(point.median_ci_upper)
                    axis.vlines(point.p, lower, upper, color=color, linewidth=0.75, zorder=3)
                    cap_width = 0.0012
                    axis.hlines(
                        [lower, upper],
                        point.p - cap_width,
                        point.p + cap_width,
                        color=color,
                        linewidth=0.75,
                        zorder=3,
                    )
                    all_positive_values.extend((lower, upper))

                if point.upper_interval_cap_limited:
                    has_cap_limited = True
                    lower = float(point.median_ci_lower or point.median_tau_p)
                    arrow_top = point.t_max * 1.08
                    axis.vlines(
                        point.p,
                        lower,
                        point.t_max,
                        color=color,
                        linewidth=0.7,
                        linestyles=":",
                        zorder=2,
                    )
                    axis.annotate(
                        "",
                        xy=(point.p, arrow_top),
                        xytext=(point.p, point.t_max * 0.98),
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.7},
                        annotation_clip=False,
                    )
                    axis.scatter(
                        [point.p],
                        [point.median_tau_p],
                        marker="^",
                        s=17,
                        facecolors="white",
                        edgecolors=[color],
                        linewidths=0.8,
                        zorder=6,
                    )
                    all_positive_values.append(arrow_top)
                else:
                    axis.scatter(
                        [point.p],
                        [point.median_tau_p],
                        marker="o",
                        s=10,
                        facecolors=[color],
                        edgecolors="white",
                        linewidths=0.25,
                        zorder=5,
                    )

        beta = points[0].beta
        axis.set_title(rf"$\beta={_beta_label(beta)}$", fontsize=9, pad=2)
        axis.set_xlabel(r"Measurement probability $p$", fontsize=8)
        axis.set_ylabel(r"Median $\tau_p$ (layers)", fontsize=8)
        axis.set_yscale("log")
        axis.tick_params(axis="both", which="major", labelsize=7, length=2.5)
        axis.tick_params(axis="both", which="minor", length=1.5)
        axis.grid(axis="y", which="both", linewidth=0.35, alpha=0.25)
        if all_positive_values:
            axis.set_ylim(
                max(0.8, min(all_positive_values) / 1.35),
                max(all_positive_values) * 1.25,
            )
        axis.text(
            0.015,
            0.985,
            "PRELIMINARY",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=5.5,
            color="0.4",
        )

        handles, labels = axis.get_legend_handles_labels()
        if has_unresolved:
            handles.append(
                Line2D(
                    [],
                    [],
                    color="0.25",
                    marker=r"$\uparrow$",
                    linestyle="None",
                    markersize=6,
                )
            )
            labels.append(r"median $>T_{\max}$")
        if has_cap_limited:
            handles.append(
                Line2D(
                    [],
                    [],
                    color="0.25",
                    marker="^",
                    markerfacecolor="white",
                    linestyle=":",
                    markersize=4,
                )
            )
            labels.append("upper CI cap-limited")
        axis.legend(
            handles,
            labels,
            loc="best",
            ncol=2,
            fontsize=5.4,
            frameon=False,
            handlelength=1.5,
            columnspacing=0.8,
            handletextpad=0.35,
        )
        _atomic_figure(figure, path)
        plt.close(figure)


def _index_text(
    *,
    run_id: str,
    summary_path: Path,
    summary_sha256: str,
    output_dir: Path,
    groups: dict[int, list[LiveSummaryPoint]],
    plotted_keys: list[int],
    skipped_keys: list[int],
    minimum_points: int,
) -> str:
    lines = [
        "# PRELIMINARY live purification-time plots",
        "",
        "> **PRELIMINARY:** These plots update while the campaign runs and are not final results.",
        "",
        f"- Run ID: `{run_id}`",
        f"- Validated point summary: `{summary_path.resolve()}`",
        f"- Point-summary SHA-256: `{summary_sha256}`",
        f"- Minimum complete p points required for every available size: {minimum_points}",
        "- Curves show raw Kaplan-Meier medians joined by straight segments; no smoothing or shaded bands are used.",
        "- Upward arrows at `T_max` are unresolved median lower limits. Open triangles identify resolved medians with cap-limited upper bootstrap intervals.",
        "",
        "## Plots",
        "",
    ]
    if plotted_keys:
        lines.extend(
            [
                "| beta | sizes | complete points by size | plot |",
                "|---:|:---|:---|:---|",
            ]
        )
        for beta_key in plotted_keys:
            points = groups[beta_key]
            counts = Counter(point.n for point in points)
            sizes = ", ".join(str(n) for n in sorted(counts))
            point_counts = ", ".join(f"n={n}: {counts[n]}" for n in sorted(counts))
            name = f"tau_p_b{beta_key:010d}.png"
            lines.append(
                f"| {_beta_label(points[0].beta)} | {sizes} | {point_counts} | [{name}]({name}) |"
            )
    else:
        lines.append("No beta currently meets the minimum-points requirement.")

    lines.extend(["", "## Skipped betas", ""])
    if skipped_keys:
        for beta_key in skipped_keys:
            points = groups[beta_key]
            counts = Counter(point.n for point in points)
            reason = ", ".join(f"n={n}: {counts[n]}" for n in sorted(counts))
            lines.append(f"- beta={_beta_label(points[0].beta)} ({reason})")
    else:
        lines.append("None.")
    lines.extend(["", f"Output directory: `{output_dir.resolve()}`", ""])
    return "\n".join(lines)


def plot_live_summaries(
    summary_path: Path | str,
    *,
    run_id: str,
    output_dir: Path | str | None = None,
    minimum_points: int = DEFAULT_MINIMUM_POINTS,
) -> LivePlotResult:
    """Validate one complete-point summary and atomically publish live raw plots."""
    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id must be 16 lowercase hexadecimal characters")
    minimum_points = _positive_integer(minimum_points, "minimum_points")
    summary = Path(summary_path)
    points, summary_sha256 = _read_summary(summary, expected_run_id=run_id)
    output = Path(output_dir) if output_dir is not None else DEFAULT_FIGURE_ROOT / run_id

    groups: dict[int, list[LiveSummaryPoint]] = defaultdict(list)
    beta_values: dict[int, float] = {}
    for point in points:
        previous = beta_values.setdefault(point.beta_key, point.beta)
        if not math.isclose(previous, point.beta, rel_tol=0.0, abs_tol=5e-13):
            raise ValueError(f"beta_key={point.beta_key} maps to inconsistent beta values")
        groups[point.beta_key].append(point)

    plotted_keys: list[int] = []
    skipped_keys: list[int] = []
    plot_paths: list[Path] = []
    for beta_key in sorted(groups):
        beta_points = groups[beta_key]
        counts = Counter(point.n for point in beta_points)
        if not counts or min(counts.values()) < minimum_points:
            skipped_keys.append(beta_key)
            continue
        plotted_keys.append(beta_key)
        plot_path = output / f"tau_p_b{beta_key:010d}.png"
        _plot_beta(beta_points, plot_path)
        plot_paths.append(plot_path.resolve())

    index_path = output / "LIVE_PLOTS.md"
    _atomic_text(
        index_path,
        _index_text(
            run_id=run_id,
            summary_path=summary,
            summary_sha256=summary_sha256,
            output_dir=output,
            groups=groups,
            plotted_keys=plotted_keys,
            skipped_keys=skipped_keys,
            minimum_points=minimum_points,
        ),
    )
    return LivePlotResult(
        run_id=run_id,
        summary_path=summary.resolve(),
        summary_sha256=summary_sha256,
        output_dir=output.resolve(),
        index_path=index_path.resolve(),
        plot_paths=tuple(plot_paths),
        plotted_betas=tuple(groups[key][0].beta for key in plotted_keys),
        skipped_betas=tuple(groups[key][0].beta for key in skipped_keys),
        validated_rows=len(points),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse live-plot input, output, and completeness threshold options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-points", type=int, default=DEFAULT_MINIMUM_POINTS)
    args = parser.parse_args(argv)
    if _RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must be 16 lowercase hexadecimal characters")
    if args.minimum_points <= 0:
        parser.error("--minimum-points must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    """Render one preliminary plot snapshot without reading point NPZ files."""
    args = parse_args(argv)
    summary = (
        args.summary
        if args.summary is not None
        else DEFAULT_ANALYSIS_ROOT / args.run_id / "live" / "point_summary.csv"
    )
    result = plot_live_summaries(
        summary,
        run_id=args.run_id,
        output_dir=args.output,
        minimum_points=args.minimum_points,
    )
    print(
        f"PRELIMINARY: wrote {len(result.plot_paths)} beta plots to {result.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MINIMUM_POINTS",
    "LivePlotResult",
    "LiveSummaryPoint",
    "plot_live_summaries",
]
