"""Create bounded preliminary scaling diagnostics for one beta value.

This command consumes only the validated ``point_summary.csv`` produced by
``analysis.aggregate``.  It never opens trajectory NPZ files.  Unresolved
Kaplan-Meier medians and cap-limited confidence intervals are accounted for by
``prepare_collapse_data`` but are not imputed or included in the fit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np

from .scaling import (
    CollapseBounds,
    CollapseData,
    CollapseFit,
    ParameterPair,
    ProfileLossLandscape,
    fit_three_parameter_collapse,
    prepare_collapse_data,
    profile_pairwise_landscapes,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

matplotlib.use("Agg")

PRL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_ROOT = PRL_ROOT / "analysis" / "runs"
DEFAULT_FIGURE_ROOT = PRL_ROOT / "figures" / "diagnostics"
PRELIMINARY_LABEL = "PRELIMINARY"

MAX_FIT_STARTS = 16
MAX_FIT_ITERATIONS = 1_000
MAX_FIT_WORK = 20_000
MAX_LANDSCAPE_GRID_SIZE = 15
MAX_PROFILE_INTERVALS = 3
MAX_LANDSCAPE_ITERATIONS = 100
MAX_LANDSCAPE_CELLS = 3 * MAX_LANDSCAPE_GRID_SIZE**2
MAX_LANDSCAPE_WORK = 250_000

_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{16}")
_SUMMARY_FIELDS = (
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
    "median_ci_lower",
    "median_ci_upper",
    "median_ci_resolved",
    "bootstrap_resolved_fraction",
    "bootstrap_resamples",
    "bootstrap_confidence",
    "survival_at_cap",
)
_PAIRS: tuple[ParameterPair, ...] = (("pc", "nu"), ("pc", "z"), ("nu", "z"))
_PARAMETER_LABELS = {
    "pc": r"$p_c$",
    "nu": r"$\nu$",
    "z": r"$z$",
}


@dataclass(frozen=True, slots=True)
class FitBetaResult:
    """Paths and identifiers for one completed preliminary diagnostic run."""

    run_id: str
    beta: float
    beta_key: int
    output_dir: Path
    summary_path: Path
    figure_paths: tuple[Path, ...]
    landscapes_included: bool


@dataclass(frozen=True, slots=True)
class _RawPoint:
    point_index: int
    n: int
    p: float
    t_max: int
    median: int | None
    lower: int | None
    upper: int | None
    bootstrap_resamples: int

    @property
    def median_resolved(self) -> bool:
        return self.median is not None

    @property
    def interval_resolved(self) -> bool:
        return self.lower is not None and self.upper is not None

    @property
    def upper_cap_limited(self) -> bool:
        return self.median is not None and self.bootstrap_resamples > 0 and self.upper is None


def _positive_integer(value: int, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]; got {value}")
    return value


def _finite_float(value: float, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number; got {value!r}")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = "finite" if minimum is None else f"finite and at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}; got {value!r}")
    return result


def _parse_ascii_integer(raw: str, name: str, *, minimum: int = 0) -> int:
    if raw != raw.strip() or not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"{name} must be an unsigned integer; got {raw!r}")
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")
    return value


def _parse_finite_float(raw: str, name: str, *, minimum: float, maximum: float) -> float:
    if raw == "" or raw != raw.strip():
        raise ValueError(f"{name} must be a finite real number; got {raw!r}")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite real number; got {raw!r}") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]; got {raw!r}")
    return value


def _validate_run_and_beta(run_id: str, beta_key: int) -> None:
    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id must be 16 lowercase hexadecimal characters")
    _positive_integer(beta_key, "beta_key", minimum=0, maximum=1_000_000_000)


def _read_beta_records(
    path: Path,
    *,
    run_id: str,
    beta_key: int,
) -> tuple[list[dict[str, str]], str]:
    """Validate the aggregate CSV and return exactly one beta's records."""
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read point summary {path}: {exc}") from exc
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"point summary {path} is not valid UTF-8") from exc

    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise ValueError(f"point summary {path} has no header")
    if tuple(reader.fieldnames) != _SUMMARY_FIELDS:
        missing = sorted(set(_SUMMARY_FIELDS) - set(reader.fieldnames))
        unexpected = sorted(set(reader.fieldnames) - set(_SUMMARY_FIELDS))
        raise ValueError(
            "point summary schema or field order differs from the validated aggregate schema; "
            f"missing={missing}, unexpected={unexpected}"
        )

    selected: list[dict[str, str]] = []
    point_indices: set[int] = set()
    canonical_points: set[tuple[int, int, int]] = set()
    beta_values: dict[int, float] = {}
    for row_number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"row {row_number}: malformed CSV field count")
        if row["analysis_status"] != PRELIMINARY_LABEL:
            raise ValueError(f"row {row_number}: analysis_status must be PRELIMINARY")
        if row["run_id"] != run_id:
            raise ValueError(f"row {row_number}: run_id={row['run_id']!r}, expected {run_id!r}")

        point_index = _parse_ascii_integer(
            row["point_index"],
            f"row {row_number}.point_index",
        )
        if point_index in point_indices:
            raise ValueError(f"duplicate point_index {point_index}")
        point_indices.add(point_index)
        row_beta_key = _parse_ascii_integer(
            row["beta_key"],
            f"row {row_number}.beta_key",
        )
        beta = _parse_finite_float(
            row["beta"],
            f"row {row_number}.beta",
            minimum=0.0,
            maximum=1.0,
        )
        if row_beta_key != round(beta * 1_000_000_000):
            raise ValueError(f"row {row_number}: beta and beta_key are inconsistent")
        previous_beta = beta_values.setdefault(row_beta_key, beta)
        if previous_beta != beta:
            raise ValueError(f"beta_key={row_beta_key} maps to inconsistent beta values")
        n = _parse_ascii_integer(row["n"], f"row {row_number}.n", minimum=1)
        p_key = _parse_ascii_integer(row["p_key"], f"row {row_number}.p_key")
        p = _parse_finite_float(
            row["p"],
            f"row {row_number}.p",
            minimum=0.0,
            maximum=1.0,
        )
        if p_key != round(p * 1_000_000):
            raise ValueError(f"row {row_number}: p and p_key are inconsistent")
        canonical = (row_beta_key, n, p_key)
        if canonical in canonical_points:
            raise ValueError(f"duplicate (beta, n, p) point {canonical}")
        canonical_points.add(canonical)
        if row_beta_key == beta_key:
            selected.append(dict(row))

    if not selected:
        raise ValueError(f"point summary contains no rows for beta_key={beta_key}")
    return selected, hashlib.sha256(raw_bytes).hexdigest()


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
        figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _optional_integer(raw: str) -> int | None:
    return None if raw == "" else int(raw)


def _raw_points(records: list[dict[str, str]]) -> list[_RawPoint]:
    points: list[_RawPoint] = []
    for row in records:
        point_index = int(row["point_index"])
        median = _optional_integer(row["median_tau_p"])
        lower = _optional_integer(row["median_ci_lower"])
        upper = _optional_integer(row["median_ci_upper"])
        if lower is not None and median is not None and lower > median:
            raise ValueError(f"point_index={point_index}: lower interval exceeds the median")
        if upper is not None and median is not None and upper < median:
            raise ValueError(f"point_index={point_index}: upper interval is below the median")
        points.append(
            _RawPoint(
                point_index=point_index,
                n=int(row["n"]),
                p=float(row["p"]),
                t_max=int(row["t_max"]),
                median=median,
                lower=lower,
                upper=upper,
                bootstrap_resamples=int(row["bootstrap_resamples"]),
            )
        )
    return points


def _mark_preliminary(axis: Any) -> None:
    axis.figure.suptitle(
        "PRELIMINARY DIAGNOSTIC",
        x=0.01,
        y=1.04,
        ha="left",
        va="top",
        fontsize=7,
        color="0.4",
    )


def _placeholder(axis: Any, message: str) -> None:
    axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center")
    _mark_preliminary(axis)


def _core_figures(
    data: CollapseData,
    fit: CollapseFit,
    raw_points: list[_RawPoint],
) -> dict[str, Figure]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figures: dict[str, Figure] = {}
    colors = plt.get_cmap("tab10")
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "text.usetex": False,
    }
    with plt.rc_context(rc):
        raw_figure, raw_axis = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
        raw_by_n: dict[int, list[_RawPoint]] = defaultdict(list)
        for point in raw_points:
            raw_by_n[point.n].append(point)
        has_unresolved = False
        has_cap_limited = False
        has_unavailable_interval = False
        for color_index, n in enumerate(sorted(raw_by_n)):
            color = colors(color_index % 10)
            points = sorted(raw_by_n[n], key=lambda point: point.p)
            resolved = [point for point in points if point.median_resolved]
            if resolved:
                raw_axis.plot(
                    [point.p for point in resolved],
                    [point.median for point in resolved],
                    color=color,
                    linewidth=0.7,
                    label=rf"$n={n}$",
                    zorder=1,
                )
            else:
                raw_axis.plot([], [], color=color, linewidth=0.7, label=rf"$n={n}$")
            for point in points:
                if not point.median_resolved:
                    has_unresolved = True
                    raw_axis.scatter(
                        [point.p],
                        [point.t_max],
                        color=color,
                        marker=r"$\uparrow$",
                        s=30,
                        zorder=4,
                    )
                    continue
                assert point.median is not None
                if point.interval_resolved:
                    assert point.lower is not None and point.upper is not None
                    raw_axis.errorbar(
                        [point.p],
                        [point.median],
                        yerr=np.asarray(
                            [[point.median - point.lower], [point.upper - point.median]]
                        ),
                        color=color,
                        marker="o",
                        markersize=3,
                        linewidth=0.8,
                        capsize=1.5,
                        zorder=3,
                    )
                elif point.upper_cap_limited:
                    has_cap_limited = True
                    lower = point.median if point.lower is None else point.lower
                    raw_axis.vlines(
                        point.p,
                        lower,
                        point.t_max,
                        color=color,
                        linewidth=0.7,
                        linestyles=":",
                        zorder=2,
                    )
                    raw_axis.scatter(
                        [point.p],
                        [point.median],
                        marker="^",
                        s=22,
                        facecolors="white",
                        edgecolors=[color],
                        linewidths=0.8,
                        zorder=4,
                    )
                else:
                    has_unavailable_interval = True
                    raw_axis.scatter(
                        [point.p],
                        [point.median],
                        marker="s",
                        s=18,
                        facecolors="white",
                        edgecolors=[color],
                        linewidths=0.8,
                        zorder=4,
                    )
        raw_axis.set_xlabel(r"Measurement probability $p$")
        raw_axis.set_ylabel(r"Median purification time $\tau_p$ (layers)")
        raw_axis.set_yscale("log")
        raw_axis.grid(alpha=0.2)
        handles, labels = raw_axis.get_legend_handles_labels()
        if has_unresolved:
            handles.append(Line2D([], [], color="0.25", marker=r"$\uparrow$", linestyle="None"))
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
                )
            )
            labels.append("upper CI cap-limited")
        if has_unavailable_interval:
            handles.append(
                Line2D(
                    [],
                    [],
                    color="0.25",
                    marker="s",
                    markerfacecolor="white",
                    linestyle="None",
                )
            )
            labels.append("interval unavailable")
        raw_axis.legend(handles, labels, frameon=False, ncol=2, fontsize=7)
        _mark_preliminary(raw_axis)
        figures["raw.png"] = raw_figure

        collapse_figure, collapse_axis = plt.subplots(
            figsize=(5.0, 3.6),
            constrained_layout=True,
        )
        residual_figure, residual_axis = plt.subplots(
            figsize=(5.0, 3.6),
            constrained_layout=True,
        )
        if fit.success and fit.master_curve is not None:
            for color_index, n in enumerate(data.sizes):
                mask = data.n == n
                order = np.argsort(fit.scaling_coordinate[mask])
                x = fit.scaling_coordinate[mask][order]
                scaled_tau = data.tau[mask][order] / n**fit.z
                collapse_axis.scatter(
                    x,
                    scaled_tau,
                    color=colors(color_index % 10),
                    s=12,
                    label=rf"$n={n}$",
                )
                residual_axis.scatter(
                    x,
                    fit.standardized_residual[mask][order],
                    color=colors(color_index % 10),
                    s=12,
                    label=rf"$n={n}$",
                )
            x_grid = np.linspace(
                float(np.min(fit.scaling_coordinate)),
                float(np.max(fit.scaling_coordinate)),
                300,
            )
            collapse_axis.plot(
                x_grid,
                fit.master_curve.predict(x_grid),
                color="black",
                linewidth=1.0,
                label="profiled master curve",
            )
            residual_axis.axhline(0.0, color="black", linewidth=0.8)
            collapse_axis.set_yscale("log")
            collapse_axis.legend(frameon=False, ncol=2, fontsize=7)
            residual_axis.legend(frameon=False, ncol=2, fontsize=7)
        else:
            _placeholder(collapse_axis, "No validated fit")
            _placeholder(residual_axis, "No validated fit")
        collapse_axis.set_xlabel(r"$(p-p_c)n^{1/\nu}$")
        collapse_axis.set_ylabel(r"$\tau_p/n^z$")
        collapse_axis.grid(alpha=0.2)
        residual_axis.set_xlabel(r"$(p-p_c)n^{1/\nu}$")
        residual_axis.set_ylabel("Standardized residual")
        residual_axis.grid(alpha=0.2)
        _mark_preliminary(collapse_axis)
        _mark_preliminary(residual_axis)
        figures["collapse.png"] = collapse_figure
        figures["residual.png"] = residual_figure
    return figures


def _landscape_figure(
    landscape: ProfileLossLandscape,
    fit: CollapseFit,
) -> Figure:
    import matplotlib.pyplot as plt

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(5.0, 3.8), constrained_layout=True)
        masked = np.ma.masked_invalid(landscape.delta_loss)
        image = axis.pcolormesh(
            landscape.x_values,
            landscape.y_values,
            masked,
            shading="auto",
            cmap="viridis",
        )
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label(r"Profiled $\Delta$ loss")
        axis.set_xlabel(_PARAMETER_LABELS[landscape.x_parameter])
        axis.set_ylabel(_PARAMETER_LABELS[landscape.y_parameter])
        axis.set_title(f"profiled over {_PARAMETER_LABELS[landscape.optimized_parameter]}")
        if fit.success:
            axis.scatter(
                [getattr(fit, landscape.x_parameter)],
                [getattr(fit, landscape.y_parameter)],
                marker="*",
                s=65,
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
                label="best fit",
            )
            axis.legend(loc="upper right", frameon=False, fontsize=7)
        _mark_preliminary(axis)
        return figure


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _fit_payload(fit: CollapseFit) -> dict[str, Any]:
    diagnostics = fit.diagnostics
    attempts = [
        {
            **asdict(attempt),
            "reported_loss": _finite_or_none(attempt.reported_loss),
            "parameters": [_finite_or_none(value) for value in attempt.parameters],
        }
        for attempt in diagnostics.attempts
    ]
    return {
        "success": fit.success,
        "parameters": {
            "pc": _finite_or_none(fit.pc),
            "nu": _finite_or_none(fit.nu),
            "z": _finite_or_none(fit.z),
        },
        "loss": _finite_or_none(fit.loss),
        "diagnostics": {
            "message": diagnostics.message,
            "n_points": diagnostics.n_points,
            "n_sizes": diagnostics.n_sizes,
            "n_starts": diagnostics.n_starts,
            "n_valid_starts": diagnostics.n_valid_starts,
            "best_start_index": diagnostics.best_start_index,
            "objective": _finite_or_none(diagnostics.objective),
            "weighted_rmse": _finite_or_none(diagnostics.weighted_rmse),
            "chi_square": _finite_or_none(diagnostics.chi_square),
            "effective_master_parameters": _finite_or_none(diagnostics.effective_master_parameters),
            "effective_degrees_of_freedom": _finite_or_none(
                diagnostics.effective_degrees_of_freedom
            ),
            "reduced_chi_square": _finite_or_none(diagnostics.reduced_chi_square),
            "spline_roughness": _finite_or_none(diagnostics.spline_roughness),
            "condition_number": _finite_or_none(diagnostics.condition_number),
            "pc_inside_common_window": diagnostics.pc_inside_common_window,
            "boundary_parameters": list(diagnostics.boundary_parameters),
            "attempts": attempts,
        },
    }


def _landscape_payload(landscape: ProfileLossLandscape) -> dict[str, Any]:
    def nullable_matrix(values: np.ndarray) -> list[list[float | None]]:
        return [
            [_finite_or_none(value) for value in row]
            for row in np.asarray(values, dtype=np.float64)
        ]

    return {
        "x_parameter": landscape.x_parameter,
        "y_parameter": landscape.y_parameter,
        "profiled_parameter": landscape.optimized_parameter,
        "x_values": [float(value) for value in landscape.x_values],
        "y_values": [float(value) for value in landscape.y_values],
        "delta_loss": nullable_matrix(landscape.delta_loss),
        "profiled_values": nullable_matrix(landscape.optimized_values),
        "valid": np.asarray(landscape.valid, dtype=bool).tolist(),
        "optimizer_calls": int(landscape.optimizer_calls),
    }


def _validate_work_limits(
    *,
    n_starts: int,
    maxiter: int,
    include_landscapes: bool,
    grid_size: int,
    profile_intervals: int,
    landscape_maxiter: int,
) -> None:
    n_starts = _positive_integer(n_starts, "n_starts", minimum=1, maximum=MAX_FIT_STARTS)
    maxiter = _positive_integer(
        maxiter,
        "maxiter",
        minimum=10,
        maximum=MAX_FIT_ITERATIONS,
    )
    if n_starts * maxiter > MAX_FIT_WORK:
        raise ValueError(f"fit work n_starts*maxiter must not exceed {MAX_FIT_WORK}")
    if not include_landscapes:
        return
    grid_size = _positive_integer(
        grid_size,
        "landscape_grid_size",
        minimum=2,
        maximum=MAX_LANDSCAPE_GRID_SIZE,
    )
    profile_intervals = _positive_integer(
        profile_intervals,
        "profile_intervals",
        minimum=1,
        maximum=MAX_PROFILE_INTERVALS,
    )
    landscape_maxiter = _positive_integer(
        landscape_maxiter,
        "landscape_maxiter",
        minimum=10,
        maximum=MAX_LANDSCAPE_ITERATIONS,
    )
    cells = len(_PAIRS) * grid_size**2
    work = cells * profile_intervals * landscape_maxiter
    if cells > MAX_LANDSCAPE_CELLS or work > MAX_LANDSCAPE_WORK:
        raise ValueError(
            "requested landscape exceeds the bounded diagnostic budget: "
            f"cells={cells}/{MAX_LANDSCAPE_CELLS}, work={work}/{MAX_LANDSCAPE_WORK}"
        )


def fit_beta_diagnostics(
    summary_path: Path | str,
    *,
    run_id: str,
    beta_key: int,
    output_dir: Path | str | None = None,
    bounds: CollapseBounds = CollapseBounds(),
    interior_knots: int = 3,
    smoothing: float = 0.02,
    n_starts: int = 6,
    maxiter: int = 300,
    include_landscapes: bool = False,
    landscape_grid_size: int = 11,
    profile_intervals: int = 1,
    landscape_maxiter: int = 60,
) -> FitBetaResult:
    """Fit one beta and atomically publish preliminary diagnostic artifacts."""
    _validate_run_and_beta(run_id, beta_key)
    if not isinstance(bounds, CollapseBounds):
        raise TypeError(f"bounds must be CollapseBounds; got {type(bounds).__name__}")
    _positive_integer(interior_knots, "interior_knots", minimum=0, maximum=8)
    smoothing = _finite_float(smoothing, "smoothing", minimum=np.finfo(float).tiny)
    _validate_work_limits(
        n_starts=n_starts,
        maxiter=maxiter,
        include_landscapes=include_landscapes,
        grid_size=landscape_grid_size,
        profile_intervals=profile_intervals,
        landscape_maxiter=landscape_maxiter,
    )

    summary = Path(summary_path)
    records, summary_sha256 = _read_beta_records(
        summary,
        run_id=run_id,
        beta_key=beta_key,
    )
    data = prepare_collapse_data(records, expected_beta=beta_key / 1_000_000_000)
    fit = fit_three_parameter_collapse(
        data,
        bounds=bounds,
        interior_knots=interior_knots,
        smoothing=smoothing,
        n_starts=n_starts,
        maxiter=maxiter,
    )

    landscapes: dict[ParameterPair, ProfileLossLandscape] = {}
    if include_landscapes:
        grids = {
            name: np.linspace(*bounds.interval(name), landscape_grid_size)
            for name in ("pc", "nu", "z")
        }
        landscapes = profile_pairwise_landscapes(
            data,
            pc_values=grids["pc"],
            nu_values=grids["nu"],
            z_values=grids["z"],
            bounds=bounds,
            interior_knots=interior_knots,
            smoothing=smoothing,
            profile_intervals=profile_intervals,
            maxiter=landscape_maxiter,
            max_cells=landscape_grid_size**2,
        )
        if set(landscapes) != set(_PAIRS):
            raise RuntimeError("profile_pairwise_landscapes did not return all three pairs")

    output = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_FIGURE_ROOT / run_id / f"beta_{beta_key}"
    )
    raw_points = _raw_points(records)
    core_figures = _core_figures(data, fit, raw_points)
    all_figures = dict(core_figures)
    for pair, landscape in landscapes.items():
        all_figures[f"profile_{pair[0]}_{pair[1]}.png"] = _landscape_figure(landscape, fit)

    import matplotlib.pyplot as plt

    figure_paths: list[Path] = []
    try:
        for name, figure in all_figures.items():
            path = output / name
            _atomic_figure(figure, path)
            figure_paths.append(path.resolve())
    finally:
        for figure in all_figures.values():
            plt.close(figure)

    payload = {
        "analysis_status": PRELIMINARY_LABEL,
        "interpretation": (
            "Diagnostic point estimates only. No smoothing across beta and no final uncertainty "
            "claim are made. Unresolved or cap-limited medians are not imputed."
        ),
        "generated_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "beta": data.beta,
        "beta_key": data.beta_key,
        "input": {
            "point_summary_csv": str(summary.resolve()),
            "sha256": summary_sha256,
            "source_kind": "validated point_summary.csv only",
        },
        "selection": asdict(data.selection),
        "raw_plot_accounting": {
            "total_rows": len(raw_points),
            "resolved_medians": sum(point.median_resolved for point in raw_points),
            "unresolved_medians": sum(not point.median_resolved for point in raw_points),
            "resolved_two_sided_intervals": sum(point.interval_resolved for point in raw_points),
            "upper_ci_cap_limited": sum(point.upper_cap_limited for point in raw_points),
            "interval_unavailable": sum(
                point.median_resolved
                and not point.interval_resolved
                and not point.upper_cap_limited
                for point in raw_points
            ),
        },
        "sizes": list(data.sizes),
        "common_p_window": list(data.common_p_window),
        "fit_options": {
            "bounds": asdict(bounds),
            "interior_knots": interior_knots,
            "smoothing": smoothing,
            "n_starts": n_starts,
            "maxiter": maxiter,
        },
        "fit": _fit_payload(fit),
        "landscape_options": {
            "included": include_landscapes,
            "grid_size_per_axis": landscape_grid_size if include_landscapes else None,
            "total_cells": len(_PAIRS) * landscape_grid_size**2 if include_landscapes else 0,
            "profile_intervals": profile_intervals if include_landscapes else None,
            "maxiter": landscape_maxiter if include_landscapes else None,
            "each_grid_spans_the_explicit_fit_bounds": include_landscapes,
        },
        "landscapes": {
            f"{pair[0]}_{pair[1]}": _landscape_payload(landscape)
            for pair, landscape in landscapes.items()
        },
        "figures": [path.name for path in figure_paths],
    }
    summary_output = output / "summary.json"
    _atomic_text(
        summary_output,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return FitBetaResult(
        run_id=run_id,
        beta=data.beta,
        beta_key=data.beta_key,
        output_dir=output.resolve(),
        summary_path=summary_output.resolve(),
        figure_paths=tuple(figure_paths),
        landscapes_included=include_landscapes,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a bounded one-beta diagnostic request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--beta-key", required=True, type=int)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pc-min", type=float, default=0.01)
    parser.add_argument("--pc-max", type=float, default=0.60)
    parser.add_argument("--nu-min", type=float, default=0.40)
    parser.add_argument("--nu-max", type=float, default=6.00)
    parser.add_argument("--z-min", type=float, default=0.00)
    parser.add_argument("--z-max", type=float, default=3.00)
    parser.add_argument("--interior-knots", type=int, default=3)
    parser.add_argument("--smoothing", type=float, default=0.02)
    parser.add_argument("--fit-starts", type=int, default=6)
    parser.add_argument("--fit-maxiter", type=int, default=300)
    parser.add_argument("--landscapes", action="store_true")
    parser.add_argument("--landscape-grid-size", type=int, default=11)
    parser.add_argument("--profile-intervals", type=int, default=1)
    parser.add_argument("--landscape-maxiter", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one bounded diagnostic and report the output directory."""
    args = parse_args(argv)
    summary = (
        args.summary
        if args.summary is not None
        else DEFAULT_ANALYSIS_ROOT / args.run_id / "live" / "point_summary.csv"
    )
    bounds = CollapseBounds(
        pc=(args.pc_min, args.pc_max),
        nu=(args.nu_min, args.nu_max),
        z=(args.z_min, args.z_max),
    )
    result = fit_beta_diagnostics(
        summary,
        run_id=args.run_id,
        beta_key=args.beta_key,
        output_dir=args.output,
        bounds=bounds,
        interior_knots=args.interior_knots,
        smoothing=args.smoothing,
        n_starts=args.fit_starts,
        maxiter=args.fit_maxiter,
        include_landscapes=args.landscapes,
        landscape_grid_size=args.landscape_grid_size,
        profile_intervals=args.profile_intervals,
        landscape_maxiter=args.landscape_maxiter,
    )
    print(
        f"{PRELIMINARY_LABEL}: beta={result.beta:.9g}; wrote "
        f"{len(result.figure_paths)} plots and {result.summary_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["FitBetaResult", "fit_beta_diagnostics", "main", "parse_args"]
