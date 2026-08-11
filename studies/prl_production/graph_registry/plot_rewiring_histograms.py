"""Plot discrete rewired-edge distributions from a validated raw-count archive."""

from __future__ import annotations

import argparse
import os
import re
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .spec import beta_from_key, canonical_beta_key

RAW_FIELDS = (
    "beta",
    "beta_key",
    "collection_id",
    "displaced",
    "displaced_logical_sha256",
    "graph_index",
    "graph_seed",
    "n",
    "operations",
    "restored",
    "schema_version",
    "skipped",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class RewiringRawData:
    """Validated per-graph rewiring counts on a rectangular ``(n, beta)`` grid."""

    collection_id: str
    displaced_logical_sha256: str
    sizes: NDArray[np.int32]
    beta: NDArray[np.float64]
    beta_key: NDArray[np.int64]
    graph_index: NDArray[np.int32]
    displaced: NDArray[np.uint16]

    @property
    def n_graphs(self) -> int:
        """Return the number of indexed graph draws in every cell."""
        return int(self.graph_index.size)


@dataclass(frozen=True, slots=True)
class RewiringHistogramPlotPaths:
    """Paths for the overview and seven detailed conditional histograms."""

    overview_png: Path
    overview_pdf: Path
    detail_pngs: tuple[Path, ...]
    detail_pdfs: tuple[Path, ...]


def _scalar_string(array: NDArray[np.generic], name: str) -> str:
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(f"{name} must be a scalar Unicode string")
    value = str(array.item())
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be nonempty and canonical")
    return value


def _validate_expected(
    actual: NDArray[np.integer],
    expected: Sequence[int] | None,
    name: str,
) -> None:
    if expected is None:
        return
    canonical = tuple(int(value) for value in expected)
    if canonical != tuple(sorted(set(canonical))):
        raise ValueError(f"expected_{name} must be unique and strictly increasing")
    if tuple(int(value) for value in actual) != canonical:
        raise ValueError(f"raw archive does not match expected_{name}")


def read_rewiring_raw(
    path: Path,
    *,
    expected_sizes: Sequence[int] | None = None,
    expected_beta_keys: Sequence[int] | None = None,
    expected_n_graphs: int | None = None,
) -> RewiringRawData:
    """Read and strictly validate the complete per-graph rewiring archive."""
    raw_path = Path(path)
    with np.load(raw_path, allow_pickle=False) as archive:
        if tuple(archive.files) != RAW_FIELDS:
            raise ValueError("raw rewiring archive schema/order differs from the required fields")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}

    scalar_version = arrays["schema_version"]
    if scalar_version.shape != () or scalar_version.dtype != np.dtype(np.int32):
        raise ValueError("schema_version must be a scalar int32")
    if int(scalar_version) != 1:
        raise ValueError(f"unsupported raw rewiring schema_version={int(scalar_version)}")

    collection_id = _scalar_string(arrays["collection_id"], "collection_id")
    logical_sha = _scalar_string(arrays["displaced_logical_sha256"], "displaced_logical_sha256")
    if _SHA256.fullmatch(logical_sha) is None:
        raise ValueError("displaced_logical_sha256 must be lowercase hexadecimal")

    required_dtypes = {
        "beta": np.dtype(np.float64),
        "beta_key": np.dtype(np.int64),
        "graph_index": np.dtype(np.int32),
        "graph_seed": np.dtype(np.uint64),
        "n": np.dtype(np.int32),
        "operations": np.dtype(np.uint16),
        "displaced": np.dtype(np.uint16),
        "restored": np.dtype(np.uint16),
        "skipped": np.dtype(np.uint16),
    }
    for name, dtype in required_dtypes.items():
        if arrays[name].dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")

    sizes = arrays["n"]
    beta = arrays["beta"]
    beta_key = arrays["beta_key"]
    graph_index = arrays["graph_index"]
    if sizes.ndim != 1 or sizes.size == 0:
        raise ValueError("n must be a nonempty one-dimensional array")
    if tuple(int(value) for value in sizes) != tuple(sorted(set(map(int, sizes)))):
        raise ValueError("n must be unique and strictly increasing")
    if np.any(sizes < 3):
        raise ValueError("all system sizes must be at least 3")
    if beta.ndim != 1 or beta.size < 2 or not np.all(np.isfinite(beta)):
        raise ValueError("beta must be a finite one-dimensional grid with at least two values")
    if beta_key.shape != beta.shape:
        raise ValueError("beta_key and beta must have the same shape")
    if tuple(int(value) for value in beta_key) != tuple(sorted(set(map(int, beta_key)))):
        raise ValueError("beta_key must be unique and strictly increasing")
    if int(beta_key[0]) != 0 or float(beta[0]) != 0.0 or np.any(beta[1:] <= 0.0):
        raise ValueError("beta grid must start at zero and then contain positive values")
    for key, value in zip(beta_key, beta, strict=True):
        expected_beta = beta_from_key(int(key))
        if canonical_beta_key(float(value)) != int(key) or float(value) != expected_beta:
            raise ValueError("beta is inconsistent with its canonical beta_key")

    if graph_index.ndim != 1 or graph_index.size < 2:
        raise ValueError("graph_index must contain at least two draws")
    if not np.array_equal(graph_index, np.arange(graph_index.size, dtype=np.int32)):
        raise ValueError("graph_index must be the canonical contiguous range starting at zero")
    shape = (sizes.size, beta.size, graph_index.size)
    for name in ("graph_seed", "operations", "displaced", "restored", "skipped"):
        if arrays[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}")

    operations = arrays["operations"]
    displaced = arrays["displaced"]
    restored = arrays["restored"]
    if not np.array_equal(
        operations.astype(np.int64) - restored.astype(np.int64),
        displaced.astype(np.int64),
    ):
        raise ValueError("operations - restored must equal displaced for every graph")
    for size_index, n_value in enumerate(sizes):
        maximum = 2 * int(n_value)
        for name in ("displaced", "operations", "restored", "skipped"):
            if np.any(arrays[name][size_index] > maximum):
                raise ValueError(f"{name} exceeds 2n for n={int(n_value)}")
    for name in ("displaced", "operations", "restored", "skipped"):
        if np.any(arrays[name][:, 0, :] != 0):
            raise ValueError(f"beta=0 {name} values must all be zero")

    if expected_n_graphs is not None:
        if (
            isinstance(expected_n_graphs, bool)
            or not isinstance(expected_n_graphs, int)
            or expected_n_graphs < 2
        ):
            raise ValueError("expected_n_graphs must be an integer >= 2")
        if graph_index.size != expected_n_graphs:
            raise ValueError(
                f"raw archive contains {graph_index.size} graphs per cell; "
                f"expected {expected_n_graphs}"
            )
    _validate_expected(sizes, expected_sizes, "sizes")
    _validate_expected(beta_key, expected_beta_keys, "beta_keys")

    return RewiringRawData(
        collection_id=collection_id,
        displaced_logical_sha256=logical_sha,
        sizes=sizes,
        beta=beta,
        beta_key=beta_key,
        graph_index=graph_index,
        displaced=displaced,
    )


def conditional_histograms(data: RewiringRawData) -> tuple[NDArray[np.float64], ...]:
    """Return exact-integer probability masses with one column per beta value."""
    histograms: list[NDArray[np.float64]] = []
    for size_index, n_value in enumerate(data.sizes):
        histogram = np.zeros((2 * int(n_value) + 1, data.beta.size), dtype=np.float64)
        for beta_index in range(data.beta.size):
            counts = np.bincount(
                data.displaced[size_index, beta_index],
                minlength=histogram.shape[0],
            )
            histogram[:, beta_index] = counts / data.n_graphs
        if not np.allclose(np.sum(histogram, axis=0), 1.0, rtol=0.0, atol=1e-15):
            raise RuntimeError("a conditional rewiring histogram does not sum to one")
        histograms.append(histogram)
    return tuple(histograms)


def rewired_fraction_support(n_value: int) -> NDArray[np.float64]:
    """Return the exact support of ``N_rewired / (2 n)`` from zero through one."""
    if isinstance(n_value, bool) or not isinstance(n_value, int) or n_value < 1:
        raise ValueError("n_value must be a positive integer")
    return np.arange(2 * n_value + 1, dtype=np.float64) / (2 * n_value)


def one_rewired_edge_fraction(n_value: int) -> float:
    """Return the normalized ordinate corresponding to exactly one rewired edge."""
    return float(rewired_fraction_support(n_value)[1])


def _fraction_edges(n_value: int) -> NDArray[np.float64]:
    """Return clipped cell edges for the exact discrete normalized support."""
    support = rewired_fraction_support(n_value)
    edges = np.empty(support.size + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges[1:-1] = 0.5 * (support[:-1] + support[1:])
    return edges


def _positive_beta_edges(beta: NDArray[np.float64]) -> NDArray[np.float64]:
    positive = beta[1:]
    if positive.size < 2 or np.any(np.diff(positive) <= 0.0):
        raise ValueError("at least two strictly increasing positive beta values are required")
    edges = np.empty(positive.size + 1, dtype=np.float64)
    edges[1:-1] = np.sqrt(positive[:-1] * positive[1:])
    edges[0] = positive[0] ** 2 / edges[1]
    edges[-1] = positive[-1] ** 2 / edges[-2]
    return edges


def _beta_ticks(beta: NDArray[np.float64]) -> tuple[list[float], list[str]]:
    candidates = (0.005, 0.01, 0.03, 0.1, 0.3, 1.0)
    lower = float(beta[1])
    upper = float(beta[-1])
    ticks = [value for value in candidates if lower <= value <= upper]
    labels = [format(value, ".3g").removeprefix("0") for value in ticks]
    return ticks, labels


def _masked(histogram: NDArray[np.float64]) -> np.ma.MaskedArray:
    return np.ma.masked_less_equal(histogram, 0.0)


def _zoom_limits(n_value: int, beta: NDArray[np.float64]) -> tuple[float, float, float]:
    """Return log-beta and normalized-y limits centered on the one-edge crossing."""
    beta_min = float(beta[1])
    threshold = one_rewired_edge_fraction(n_value)
    x_lower = min(beta_min, threshold) / 1.6
    x_upper = min(float(beta[-1]), 3.0 * max(beta_min, threshold))
    y_upper = min(1.0, max(4.5 * threshold, 1.25 * x_upper))
    return x_lower, x_upper, y_upper


def _line_effects(linewidth: float):
    from matplotlib import patheffects

    return [
        patheffects.Stroke(linewidth=linewidth, foreground="black"),
        patheffects.Normal(),
    ]


def _add_horizontal_axis_break(left_axis, right_axis, *, detail: bool) -> None:
    """Mark the omitted interval between zero and the positive log-beta axis."""
    # This is the canonical Matplotlib broken-axis construction: hide the
    # touching spines and draw diagonal markers in the axes' own coordinates.
    # A marker path keeps the slash angle and length independent of data limits.
    left_axis.spines["right"].set_visible(False)
    right_axis.spines["left"].set_visible(False)
    left_axis.tick_params(right=False)
    right_axis.tick_params(left=False)
    marker = [(-0.45, -1.0), (0.45, 1.0)]
    marker_size = 5.5 if detail else 4.5
    marker_width = 0.8 if detail else 0.65
    for axis, x_value, side in (
        (left_axis, 1.0, "left"),
        (right_axis, 0.0, "right"),
    ):
        for y_value, edge in ((0.0, "bottom"), (1.0, "top")):
            artist = axis.plot(
                [x_value],
                [y_value],
                marker=marker,
                markersize=marker_size,
                markeredgewidth=marker_width,
                color="black",
                linestyle="none",
                transform=axis.transAxes,
                clip_on=False,
                zorder=20,
            )[0]
            artist.set_gid(f"beta-axis-break-{side}-{edge}")


def _add_beta_minimum_mean(
    axis,
    *,
    n_value: int,
    beta: NDArray[np.float64],
    means: NDArray[np.float64],
    detail: bool,
) -> None:
    """Mark and label the empirical rewired-edge mean at ``beta_min``."""
    beta_min = float(beta[1])
    mean_fraction = float(means[1])
    mean_edges = mean_fraction * (2 * n_value)
    marker = axis.plot(
        [beta_min],
        [mean_fraction],
        marker="D",
        markersize=4.1 if detail else 2.8,
        markerfacecolor="#ffd166",
        markeredgecolor="black",
        markeredgewidth=0.5 if detail else 0.35,
        linestyle="none",
        label=r"Mean at $\beta_{\min}$",
        zorder=8,
    )[0]
    marker.set_gid("beta-min-empirical-mean")
    annotation = axis.annotate(
        rf"$\langle N_{{\rm rw}}\rangle_{{\beta_{{\min}}}}={mean_edges:.3f}$",
        xy=(beta_min, mean_fraction),
        xycoords="data",
        xytext=(0.97, 0.91),
        textcoords="axes fraction",
        ha="right",
        va="top",
        fontsize=6.6 if detail else 4.7,
        arrowprops={
            "arrowstyle": "-",
            "color": "#ffd166",
            "linewidth": 0.7 if detail else 0.45,
        },
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "0.35",
            "linewidth": 0.4,
            "alpha": 0.88,
        },
        zorder=9,
    )
    annotation.set_gid("beta-min-mean-annotation")


def _plot_reference_lines(axis, *, n_value: int, beta: NDArray[np.float64], detail: bool):
    threshold = one_rewired_edge_fraction(n_value)
    theory_width = 0.95 if detail else 0.65
    threshold_width = 0.9 if detail else 0.65
    theory_line = axis.plot(
        beta[1:],
        beta[1:],
        color="#ff6f61",
        linestyle="--",
        linewidth=theory_width,
        label=r"Accepted-operation theory: $f=\beta$",
        zorder=5,
    )[0]
    theory_line.set_path_effects(_line_effects(1.8 if detail else 1.35))
    threshold_line = axis.axhline(
        threshold,
        color="#67e8f9",
        linestyle="-.",
        linewidth=threshold_width,
        label=r"One edge: $f_{\rm rw}=1/(2n)$",
        zorder=5,
    )
    threshold_line.set_path_effects(_line_effects(1.7 if detail else 1.3))
    return theory_line, threshold_line


def _add_zoom_panel(
    axis,
    *,
    n_value: int,
    beta: NDArray[np.float64],
    means: NDArray[np.float64],
    histogram: NDArray[np.float64],
    norm,
    cmap,
    detail: bool,
) -> None:
    from matplotlib.ticker import NullFormatter

    threshold = one_rewired_edge_fraction(n_value)
    beta_min = float(beta[1])
    x_lower, x_upper, y_upper = _zoom_limits(n_value, beta)
    axis.axvspan(
        x_lower,
        min(beta_min, x_upper),
        facecolor="0.82",
        alpha=0.35,
        hatch="////",
        edgecolor="0.55",
        linewidth=0.0,
        zorder=0,
    )
    axis.pcolormesh(
        _positive_beta_edges(beta),
        _fraction_edges(n_value),
        _masked(histogram[:, 1:]),
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
        zorder=1,
    )
    mean_line = axis.plot(
        beta[1:],
        means[1:],
        color="white",
        linewidth=1.0 if detail else 0.7,
        label="Empirical mean",
        zorder=5,
    )[0]
    mean_line.set_path_effects(_line_effects(2.0 if detail else 1.45))
    _plot_reference_lines(axis, n_value=n_value, beta=beta, detail=detail)
    _add_beta_minimum_mean(
        axis,
        n_value=n_value,
        beta=beta,
        means=means,
        detail=detail,
    )
    axis.axvline(
        threshold,
        color="#67e8f9",
        linestyle=":",
        linewidth=0.8 if detail else 0.55,
        zorder=4,
    )
    axis.axvline(
        beta_min,
        color="0.3",
        linestyle=":",
        linewidth=0.8 if detail else 0.55,
        zorder=4,
    )
    axis.scatter(
        [threshold],
        [threshold],
        s=15 if detail else 7,
        color="#67e8f9",
        edgecolor="black",
        linewidth=0.45,
        zorder=6,
    )
    axis.set_xscale("log")
    axis.set_xlim(x_lower, x_upper)
    axis.set_ylim(0.0, y_upper)
    axis.set_yticks([0.0, threshold], ["0", "1 edge"])
    axis.tick_params(axis="both", labelsize=6.8 if detail else 5.3, pad=1.2)
    axis.xaxis.set_minor_formatter(NullFormatter())
    relation = r"$\geq\beta_{\min}$" if threshold >= beta_min else r"$<\beta_{\min}$"
    axis.set_title(
        rf"One-edge zoom: $\beta_\star={threshold:.4g}$ {relation}",
        fontsize=7.4 if detail else 5.5,
        pad=2.0,
    )
    axis.set_xlabel(r"$\beta$", fontsize=7.2 if detail else 5.5, labelpad=0.7)
    axis.grid(linewidth=0.3, alpha=0.2)


def _add_distribution_panel(
    figure,
    slot,
    *,
    n_value: int,
    beta: NDArray[np.float64],
    samples: NDArray[np.uint16],
    histogram: NDArray[np.float64],
    norm,
    cmap,
    detail: bool,
    zoom_axis=None,
):
    inner = slot.subgridspec(1, 2, width_ratios=(0.085, 1.0), wspace=0.018)
    zero_axis = figure.add_subplot(inner[0, 0])
    positive_axis = figure.add_subplot(inner[0, 1], sharey=zero_axis)
    y_edges = _fraction_edges(n_value)
    zero_mesh = zero_axis.pcolormesh(
        (-0.5, 0.5),
        y_edges,
        _masked(histogram[:, :1]),
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    positive_axis.pcolormesh(
        _positive_beta_edges(beta),
        y_edges,
        _masked(histogram[:, 1:]),
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    means = np.mean(samples.astype(np.float64), axis=1) / (2 * n_value)
    zero_axis.plot([0.0], [means[0]], marker="o", markersize=2.2, color="white", zorder=4)
    mean_line = positive_axis.plot(
        beta[1:],
        means[1:],
        color="white",
        linewidth=0.8 if detail else 0.65,
        zorder=4,
    )[0]
    mean_line.set_label("Empirical mean")
    mean_line.set_path_effects(_line_effects(1.8 if detail else 1.35))
    theory_line, threshold_line = _plot_reference_lines(
        positive_axis,
        n_value=n_value,
        beta=beta,
        detail=detail,
    )
    threshold = one_rewired_edge_fraction(n_value)
    zero_axis.axhline(
        threshold,
        color="#67e8f9",
        linestyle="-.",
        linewidth=0.9 if detail else 0.65,
        zorder=5,
    )

    zero_axis.set_xlim(-0.5, 0.5)
    zero_axis.set_xticks([0.0], ["0"])
    zero_axis.tick_params(axis="y", labelsize=7.2)
    zero_axis.tick_params(axis="x", labelsize=7.2, pad=2)
    positive_axis.set_xscale("log")
    positive_axis.set_xlim(_positive_beta_edges(beta)[[0, -1]])
    ticks, labels = _beta_ticks(beta)
    positive_axis.set_xticks(ticks, labels)
    positive_axis.tick_params(axis="both", labelsize=7.2)
    positive_axis.tick_params(axis="y", labelleft=False)
    positive_axis.set_ylim(0.0, 1.0)
    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0] if detail else [0.0, 0.5, 1.0]
    zero_axis.set_yticks(y_ticks)
    zero_axis.set_ylabel(r"$f_{\rm rw}=N_{\rm rw}/(2n)$", fontsize=8.2)
    positive_axis.set_title(rf"$n={n_value}$", fontsize=8.5, pad=2.5)
    zero_axis.grid(axis="y", linewidth=0.35, alpha=0.2)
    positive_axis.grid(axis="y", linewidth=0.35, alpha=0.2)
    if detail:
        positive_axis.set_xlabel(r"Rewiring probability $\beta$ (log scale)", fontsize=8.2)
        positive_axis.legend(
            handles=(mean_line, theory_line, threshold_line),
            loc="upper left",
            fontsize=6.8,
            framealpha=0.88,
            borderpad=0.35,
            handlelength=2.1,
        )
    _add_horizontal_axis_break(zero_axis, positive_axis, detail=detail)
    if zoom_axis is None:
        zoom_axis = positive_axis.inset_axes((0.06, 0.56, 0.52, 0.37))
    _add_zoom_panel(
        zoom_axis,
        n_value=n_value,
        beta=beta,
        means=means,
        histogram=histogram,
        norm=norm,
        cmap=cmap,
        detail=detail,
    )
    return zero_mesh


def _build_overview(data: RewiringRawData, histograms: tuple[NDArray[np.float64], ...]):
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    with plt.rc_context(mpl.rcParamsDefault):
        figure = plt.figure(figsize=(7.3, 8.5), constrained_layout=False)
        figure.subplots_adjust(left=0.072, right=0.985, bottom=0.065, top=0.975)
        grid = figure.add_gridspec(4, 2, wspace=0.22, hspace=0.39)
        norm = LogNorm(vmin=1.0 / data.n_graphs, vmax=1.0)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(alpha=0.0)
        mesh = None
        for index, (n_value, histogram) in enumerate(zip(data.sizes, histograms, strict=True)):
            mesh = _add_distribution_panel(
                figure,
                grid[index // 2, index % 2],
                n_value=int(n_value),
                beta=data.beta,
                samples=data.displaced[index],
                histogram=histogram,
                norm=norm,
                cmap=cmap,
                detail=False,
            )
        legend_axis = figure.add_subplot(grid[3, 1])
        legend_axis.axis("off")
        if mesh is None:
            raise RuntimeError("no histogram panels were constructed")
        colorbar_axis = legend_axis.inset_axes((0.13, 0.67, 0.74, 0.10))
        colorbar = figure.colorbar(mesh, cax=colorbar_axis, orientation="horizontal")
        colorbar.set_label("Probability mass", fontsize=8.2)
        colorbar.ax.xaxis.set_label_position("top")
        colorbar.ax.xaxis.labelpad = 2
        colorbar.set_ticks([1e-3, 1e-2, 1e-1, 1.0])
        colorbar.ax.tick_params(labelsize=7.2)
        handles = (
            Line2D([0], [0], color="white", linewidth=1.1, path_effects=_line_effects(2.0)),
            Line2D([0], [0], color="#ff6f61", linestyle="--", linewidth=1.0),
            Line2D([0], [0], color="#67e8f9", linestyle="-.", linewidth=1.0),
        )
        legend_axis.legend(
            handles,
            (
                "Empirical mean",
                r"Accepted-operation theory: $f=\beta$",
                r"One edge: $f_{\rm rw}=1/(2n)$",
            ),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.08),
            fontsize=7.4,
            frameon=False,
            handlelength=2.5,
        )
        figure.supxlabel(r"Rewiring probability $\beta$", fontsize=9)
        return figure


def _build_detail(
    data: RewiringRawData,
    size_index: int,
    histogram: NDArray[np.float64],
):
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    with plt.rc_context(mpl.rcParamsDefault):
        figure = plt.figure(figsize=(7.3, 3.25), constrained_layout=False)
        figure.subplots_adjust(left=0.073, right=0.965, bottom=0.17, top=0.91)
        grid = figure.add_gridspec(
            1,
            3,
            width_ratios=(1.45, 0.78, 0.035),
            wspace=0.25,
        )
        norm = LogNorm(vmin=1.0 / data.n_graphs, vmax=1.0)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(alpha=0.0)
        zoom_axis = figure.add_subplot(grid[0, 1])
        mesh = _add_distribution_panel(
            figure,
            grid[0, 0],
            n_value=int(data.sizes[size_index]),
            beta=data.beta,
            samples=data.displaced[size_index],
            histogram=histogram,
            norm=norm,
            cmap=cmap,
            detail=True,
            zoom_axis=zoom_axis,
        )
        colorbar_axis = figure.add_subplot(grid[0, 2])
        colorbar = figure.colorbar(mesh, cax=colorbar_axis)
        colorbar.set_label("Probability mass", fontsize=8.2)
        colorbar.set_ticks([1e-3, 1e-2, 1e-1, 1.0])
        colorbar.ax.tick_params(labelsize=7.2)
        return figure


def _save_figure(figure, png_path: Path, pdf_path: Path) -> None:
    import matplotlib as mpl

    png_path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    png_temporary = png_path.with_name(f".{png_path.name}.{token}.tmp")
    pdf_temporary = pdf_path.with_name(f".{pdf_path.name}.{token}.tmp")
    try:
        with mpl.rc_context(mpl.rcParamsDefault):
            figure.savefig(
                png_temporary,
                format="png",
                dpi=250,
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


def plot_rewiring_histograms(
    raw_npz: Path,
    output_dir: Path,
    *,
    stem: str = "rewired_edges_histograms_vs_beta",
    detail_dir_name: str = "rewired_edges_histograms_by_n",
    expected_sizes: Sequence[int] | None = None,
    expected_beta_keys: Sequence[int] | None = None,
    expected_n_graphs: int | None = None,
) -> RewiringHistogramPlotPaths:
    """Validate raw counts and atomically publish overview and per-size histograms."""
    for value, name in ((stem, "stem"), (detail_dir_name, "detail_dir_name")):
        if not value or Path(value).name != value or Path(value).suffix:
            raise ValueError(f"{name} must be a nonempty extension-free file name")
    data = read_rewiring_raw(
        raw_npz,
        expected_sizes=expected_sizes,
        expected_beta_keys=expected_beta_keys,
        expected_n_graphs=expected_n_graphs,
    )
    histograms = conditional_histograms(data)
    output = Path(output_dir)
    overview_png = output / f"{stem}.png"
    overview_pdf = output / f"{stem}.pdf"
    detail_output = output / detail_dir_name
    detail_pngs = tuple(
        detail_output / f"rewired_edges_histogram_n{int(n_value):03d}.png" for n_value in data.sizes
    )
    detail_pdfs = tuple(path.with_suffix(".pdf") for path in detail_pngs)

    import matplotlib.pyplot as plt

    overview = _build_overview(data, histograms)
    try:
        _save_figure(overview, overview_png, overview_pdf)
    finally:
        plt.close(overview)
    for size_index, histogram in enumerate(histograms):
        detail = _build_detail(data, size_index, histogram)
        try:
            _save_figure(detail, detail_pngs[size_index], detail_pdfs[size_index])
        finally:
            plt.close(detail)
    return RewiringHistogramPlotPaths(
        overview_png=overview_png,
        overview_pdf=overview_pdf,
        detail_pngs=detail_pngs,
        detail_pdfs=detail_pdfs,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_npz", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-n-graphs", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = plot_rewiring_histograms(
        args.raw_npz,
        args.output_dir,
        expected_n_graphs=args.expected_n_graphs,
    )
    print(f"overview PNG: {paths.overview_png}")
    print(f"overview PDF: {paths.overview_pdf}")
    for png_path in paths.detail_pngs:
        print(f"detail PNG: {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "RAW_FIELDS",
    "RewiringHistogramPlotPaths",
    "RewiringRawData",
    "conditional_histograms",
    "main",
    "one_rewired_edge_fraction",
    "plot_rewiring_histograms",
    "read_rewiring_raw",
    "rewired_fraction_support",
]
