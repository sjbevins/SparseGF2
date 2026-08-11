from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from studies.prl_production.graph_registry.plot_connectivity_gain import (
    NESTED_SIZE_SETS,
    SUMMARY_FIELDS,
    _build_figure,
    plot_connectivity_gain,
    read_connectivity_gain_summary,
)
from studies.prl_production.graph_registry.spec import production_beta_keys


def _row(
    set_size: int,
    beta_key: int,
    *,
    size_sets: tuple[tuple[int, ...], ...] = NESTED_SIZE_SETS,
    n_graphs: int = 8,
) -> dict[str, object]:
    beta = beta_key / 1_000_000_000
    log_gain = 0.45 * set_size * math.sqrt(beta)
    gain = math.exp(log_gain)
    log_sem = 0.0 if beta_key == 0 else 0.006 / math.sqrt(set_size)
    gain_sem = gain * log_sem
    return {
        "collection_id": "synthetic_collection",
        "set_size": set_size,
        "size_set": json.dumps(size_sets[set_size - 1], separators=(",", ":")),
        "beta_key": beta_key,
        "beta": beta,
        "n_graphs_per_cell": n_graphs,
        "g_lambda": gain,
        "g_lambda_sem": gain_sem,
        "log_g_lambda": log_gain,
        "log_g_lambda_sem": log_sem,
        "ci68_low": math.exp(log_gain - log_sem),
        "ci68_high": math.exp(log_gain + log_sem),
    }


def _complete_rows(
    *,
    size_sets: tuple[tuple[int, ...], ...] = NESTED_SIZE_SETS,
    beta_keys: tuple[int, ...] | None = None,
) -> list[dict[str, object]]:
    keys = tuple(production_beta_keys()) if beta_keys is None else beta_keys
    return [
        _row(set_size, beta_key, size_sets=size_sets)
        for set_size in range(1, len(size_sets) + 1)
        for beta_key in keys
    ]


def _write_summary(
    path: Path,
    rows: list[dict[str, object]],
    fields: tuple[str, ...] = SUMMARY_FIELDS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_reader_accepts_exact_production_grid_and_nested_size_sets(tmp_path: Path) -> None:
    summary = tmp_path / "connectivity_gain_summary.csv"
    _write_summary(summary, _complete_rows())

    points = read_connectivity_gain_summary(
        summary,
        expected_collection_id="synthetic_collection",
        expected_n_graphs_per_cell=8,
    )

    assert len(points) == 4 * 50
    assert tuple(dict.fromkeys(point.size_set for point in points)) == NESTED_SIZE_SETS
    assert tuple(point.beta_key for point in points[:50]) == tuple(production_beta_keys())
    for set_size in range(1, 5):
        zero = points[(set_size - 1) * 50]
        assert zero.beta_key == 0
        assert zero.g_lambda == zero.ci68_low == zero.ci68_high == 1.0
        assert zero.g_lambda_sem == zero.log_g_lambda == zero.log_g_lambda_sem == 0.0


def test_reader_accepts_explicit_smoke_grid(tmp_path: Path) -> None:
    size_sets = ((8,), (8, 12))
    beta_keys = (0, 100_000_000, 1_000_000_000)
    summary = tmp_path / "smoke.csv"
    _write_summary(summary, _complete_rows(size_sets=size_sets, beta_keys=beta_keys))

    points = read_connectivity_gain_summary(
        summary,
        expected_size_sets=size_sets,
        expected_beta_keys=beta_keys,
        expected_n_graphs_per_cell=8,
    )

    assert len(points) == 6
    assert tuple(dict.fromkeys(point.size_set for point in points)) == size_sets


def test_figure_has_broken_beta_axis_and_cardinality_slices(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    summary = tmp_path / "connectivity_gain_summary.csv"
    _write_summary(summary, _complete_rows())
    points = read_connectivity_gain_summary(summary)

    figure = _build_figure(points)
    try:
        figure.canvas.draw()
        zero_axis, positive_axis, cardinality_axis = figure.axes
        assert figure.get_size_inches().tolist() == [7.3, 3.25]
        assert positive_axis.get_xscale() == "log"
        assert all(axis.get_yscale() == "log" for axis in figure.axes)
        assert not zero_axis.spines["right"].get_visible()
        assert not positive_axis.spines["left"].get_visible()
        assert {line.get_gid() for line in zero_axis.lines if line.get_gid()} >= {
            "beta-axis-break-left-bottom",
            "beta-axis-break-left-top",
            "connectivity-gain-beta-m1-zero",
        }
        assert {line.get_gid() for line in positive_axis.lines if line.get_gid()} >= {
            "beta-axis-break-right-bottom",
            "beta-axis-break-right-top",
            "connectivity-gain-beta-m1",
            "connectivity-gain-beta-m4",
        }
        np.testing.assert_array_equal(cardinality_axis.get_xticks(), [1, 2, 3, 4])
        cardinality_lines = [
            line
            for line in cardinality_axis.lines
            if (line.get_gid() or "").startswith("connectivity-gain-cardinality-beta-")
        ]
        assert len(cardinality_lines) == 6
        assert all(np.array_equal(line.get_xdata(), [1, 2, 3, 4]) for line in cardinality_lines)

        left_x0 = zero_axis.get_position().x0
        left_x1 = positive_axis.get_position().x1
        right_width = cardinality_axis.get_position().width
        assert abs((left_x1 - left_x0) - right_width) < 1e-12
    finally:
        plt.close(figure)


def test_plot_writes_deterministic_atomic_png_and_pdf(tmp_path: Path) -> None:
    summary = tmp_path / "connectivity_gain_summary.csv"
    _write_summary(summary, _complete_rows())

    paths = plot_connectivity_gain(
        summary,
        tmp_path / "figures",
        expected_collection_id="synthetic_collection",
        expected_n_graphs_per_cell=8,
    )

    assert paths.png.name == "algebraic_connectivity_gain_convergence.png"
    assert paths.pdf.name == "algebraic_connectivity_gain_convergence.pdf"
    assert paths.png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert paths.pdf.read_bytes().startswith(b"%PDF-")
    assert paths.png.stat().st_size > 10_000
    assert paths.pdf.stat().st_size > 1_000
    first_png = paths.png.read_bytes()
    first_pdf = paths.pdf.read_bytes()

    repeated = plot_connectivity_gain(
        summary,
        tmp_path / "figures",
        expected_collection_id="synthetic_collection",
        expected_n_graphs_per_cell=8,
    )
    assert repeated.png.read_bytes() == first_png
    assert repeated.pdf.read_bytes() == first_pdf
    assert not list(paths.png.parent.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("schema", "schema/order differs"),
        ("size_set", "requires size_set"),
        ("missing", "complete expected rectangular grid"),
        ("order", "canonical set-major"),
        ("duplicate", "duplicate grid rows"),
        ("beta", "not the canonical value"),
        ("nonpositive", "gain and CI endpoints must be positive"),
        ("log", "inconsistent with ln"),
        ("sem", "inconsistent with the log-scale SEM"),
        ("ci", "ci68_low <= g_lambda"),
        ("beta_zero", "beta=0 gain must be exactly one"),
        ("mixed_collection", "mixes collection_id"),
    ],
)
def test_reader_rejects_invalid_summaries(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    rows = _complete_rows()
    fields = SUMMARY_FIELDS
    if mutation == "schema":
        fields = tuple(reversed(SUMMARY_FIELDS))
    elif mutation == "size_set":
        rows[50]["size_set"] = "[64,96]"
    elif mutation == "missing":
        rows.pop(1)
    elif mutation == "order":
        rows[1], rows[2] = rows[2], rows[1]
    elif mutation == "duplicate":
        rows[1] = dict(rows[0])
    elif mutation == "beta":
        rows[1]["beta"] = float(rows[1]["beta"]) + 1e-7
    elif mutation == "nonpositive":
        rows[1]["g_lambda"] = 0.0
    elif mutation == "log":
        rows[1]["log_g_lambda"] = float(rows[1]["log_g_lambda"]) + 0.1
    elif mutation == "sem":
        rows[1]["g_lambda_sem"] = float(rows[1]["g_lambda_sem"]) * 2.0
    elif mutation == "ci":
        rows[1]["ci68_low"] = float(rows[1]["g_lambda"]) * 1.1
    elif mutation == "beta_zero":
        rows[0]["g_lambda"] = 1.01
        rows[0]["log_g_lambda"] = math.log(1.01)
        rows[0]["ci68_low"] = 1.01
        rows[0]["ci68_high"] = 1.01
    elif mutation == "mixed_collection":
        rows[1]["collection_id"] = "another_collection"
    summary = tmp_path / "bad.csv"
    _write_summary(summary, rows, fields)

    with pytest.raises(ValueError, match=match):
        read_connectivity_gain_summary(summary)


def test_plot_validates_before_creating_output_directory(tmp_path: Path) -> None:
    rows = _complete_rows()
    rows.pop()
    summary = tmp_path / "incomplete.csv"
    _write_summary(summary, rows)
    output = tmp_path / "figures"

    with pytest.raises(ValueError, match="complete expected rectangular grid"):
        plot_connectivity_gain(summary, output)
    assert not output.exists()
