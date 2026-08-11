from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from studies.prl_production.graph_registry.plot_rewiring_histograms import (
    RAW_FIELDS,
    _build_detail,
    _build_overview,
    conditional_histograms,
    one_rewired_edge_fraction,
    plot_rewiring_histograms,
    read_rewiring_raw,
    rewired_fraction_support,
)


def _arrays() -> dict[str, np.ndarray]:
    sizes = np.asarray([8, 12], dtype=np.int32)
    beta_key = np.asarray([0, 100_000_000, 1_000_000_000], dtype=np.int64)
    beta = beta_key.astype(np.float64) / 1_000_000_000
    graph_index = np.arange(8, dtype=np.int32)
    shape = (sizes.size, beta.size, graph_index.size)
    operations = np.zeros(shape, dtype=np.uint16)
    restored = np.zeros(shape, dtype=np.uint16)
    for size_index, n_value in enumerate(sizes):
        operations[size_index, 1] = np.arange(8, dtype=np.uint16) % 4
        operations[size_index, 2] = 2 * n_value - np.arange(8, dtype=np.uint16) % 3
        restored[size_index, 2] = np.arange(8, dtype=np.uint16) % 2
    displaced = operations - restored
    return {
        "beta": beta,
        "beta_key": beta_key,
        "collection_id": np.str_("synthetic_collection"),
        "displaced": displaced,
        "displaced_logical_sha256": np.str_("a" * 64),
        "graph_index": graph_index,
        "graph_seed": np.arange(np.prod(shape), dtype=np.uint64).reshape(shape),
        "n": sizes,
        "operations": operations,
        "restored": restored,
        "schema_version": np.int32(1),
        "skipped": np.zeros(shape, dtype=np.uint16),
    }


def _write(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def test_histograms_are_exact_integer_pmfs_and_preserve_means(tmp_path: Path) -> None:
    raw = tmp_path / "raw.npz"
    arrays = _arrays()
    _write(raw, arrays)
    data = read_rewiring_raw(
        raw,
        expected_sizes=(8, 12),
        expected_beta_keys=(0, 100_000_000, 1_000_000_000),
        expected_n_graphs=8,
    )

    histograms = conditional_histograms(data)

    assert len(histograms) == 2
    for size_index, (n_value, histogram) in enumerate(zip(data.sizes, histograms, strict=True)):
        assert histogram.shape == (2 * int(n_value) + 1, 3)
        np.testing.assert_allclose(histogram.sum(axis=0), 1.0, rtol=0.0, atol=1e-15)
        support = np.arange(histogram.shape[0])[:, None]
        np.testing.assert_allclose(
            np.sum(histogram * support, axis=0),
            np.mean(arrays["displaced"][size_index], axis=1),
            rtol=0.0,
            atol=1e-15,
        )
        assert histogram[0, 0] == 1.0
        assert np.count_nonzero(histogram[:, 0]) == 1
        fraction_support = rewired_fraction_support(int(n_value))[:, None]
        np.testing.assert_allclose(
            np.sum(histogram * fraction_support, axis=0),
            np.mean(arrays["displaced"][size_index], axis=1) / (2 * int(n_value)),
            rtol=0.0,
            atol=1e-15,
        )


@pytest.mark.parametrize("n_value", [8, 12, 64, 256])
def test_normalized_support_and_one_edge_threshold(n_value: int) -> None:
    support = rewired_fraction_support(n_value)

    assert support.shape == (2 * n_value + 1,)
    assert support[0] == 0.0
    assert support[-1] == 1.0
    assert support[1] == one_rewired_edge_fraction(n_value) == 1.0 / (2 * n_value)
    np.testing.assert_allclose(np.diff(support), 1.0 / (2 * n_value), rtol=0.0, atol=1e-16)


@pytest.mark.parametrize("bad_n", [True, 0, -1, 2.5])
def test_normalized_support_rejects_invalid_size(bad_n: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        rewired_fraction_support(bad_n)  # type: ignore[arg-type]


def test_detail_plot_contains_normalized_theory_and_one_edge_crossing(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    raw = tmp_path / "raw.npz"
    _write(raw, _arrays())
    data = read_rewiring_raw(raw)
    histogram = conditional_histograms(data)[0]
    n_value = int(data.sizes[0])
    threshold = one_rewired_edge_fraction(n_value)

    figure = _build_detail(data, 0, histogram)
    try:
        zoom_axis, zero_axis, positive_axis, _ = figure.axes
        assert positive_axis.get_ylim() == (0.0, 1.0)
        assert zero_axis.get_ylim() == (0.0, 1.0)
        assert not zero_axis.spines["right"].get_visible()
        assert not positive_axis.spines["left"].get_visible()
        assert not any(text.get_text() == r"$\approx$" for text in zero_axis.texts)
        assert {line.get_gid() for line in zero_axis.lines if line.get_gid()} == {
            "beta-axis-break-left-bottom",
            "beta-axis-break-left-top",
        }
        assert {line.get_gid() for line in positive_axis.lines if line.get_gid()} == {
            "beta-axis-break-right-bottom",
            "beta-axis-break-right-top",
        }
        for axis in (zero_axis, positive_axis):
            break_lines = [
                line for line in axis.lines if (line.get_gid() or "").startswith("beta-axis-break-")
            ]
            assert len(break_lines) == 2
            assert all(line.get_transform() == axis.transAxes for line in break_lines)
        for axis in (positive_axis, zoom_axis):
            lines = {line.get_label(): line for line in axis.lines}
            theory = lines[r"Accepted-operation theory: $f=\beta$"]
            one_edge = lines[r"One edge: $f_{\rm rw}=1/(2n)$"]
            np.testing.assert_array_equal(theory.get_xdata(), data.beta[1:])
            np.testing.assert_array_equal(theory.get_ydata(), data.beta[1:])
            np.testing.assert_array_equal(one_edge.get_ydata(), [threshold, threshold])
        beta_min_marker = next(
            line for line in zoom_axis.lines if line.get_gid() == "beta-min-empirical-mean"
        )
        expected_mean_edges = float(np.mean(data.displaced[0, 1]))
        np.testing.assert_array_equal(beta_min_marker.get_xdata(), [data.beta[1]])
        np.testing.assert_allclose(
            beta_min_marker.get_ydata(),
            [expected_mean_edges / (2 * n_value)],
            rtol=0.0,
            atol=1e-15,
        )
        annotation = next(
            text for text in zoom_axis.texts if text.get_gid() == "beta-min-mean-annotation"
        )
        assert f"={expected_mean_edges:.3f}" in annotation.get_text()
        x_lower, x_upper = zoom_axis.get_xlim()
        assert x_lower < threshold < x_upper
        assert x_lower < float(data.beta[1]) < x_upper
    finally:
        plt.close(figure)


def test_overview_uses_equal_fixed_panel_geometry(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    raw = tmp_path / "raw.npz"
    _write(raw, _arrays())
    data = read_rewiring_raw(raw)

    figure = _build_overview(data, conditional_histograms(data))
    try:
        figure.canvas.draw()
        positive_axes = [axis for axis in figure.axes if axis.get_title().startswith("$n=")]
        assert len(positive_axes) == len(data.sizes)
        widths = [axis.get_position().width for axis in positive_axes]
        heights = [axis.get_position().height for axis in positive_axes]
        assert max(widths) - min(widths) < 1e-12
        assert max(heights) - min(heights) < 1e-12
    finally:
        plt.close(figure)


def test_plot_writes_deterministic_overview_and_per_size_outputs(tmp_path: Path) -> None:
    raw = tmp_path / "raw.npz"
    _write(raw, _arrays())

    paths = plot_rewiring_histograms(
        raw,
        tmp_path / "figures",
        expected_sizes=(8, 12),
        expected_beta_keys=(0, 100_000_000, 1_000_000_000),
        expected_n_graphs=8,
    )

    assert paths.overview_png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert paths.overview_pdf.read_bytes().startswith(b"%PDF-")
    assert [path.name for path in paths.detail_pngs] == [
        "rewired_edges_histogram_n008.png",
        "rewired_edges_histogram_n012.png",
    ]
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in paths.detail_pngs)
    assert all(path.read_bytes().startswith(b"%PDF-") for path in paths.detail_pdfs)
    first = {
        path: path.read_bytes()
        for path in (
            paths.overview_png,
            paths.overview_pdf,
            *paths.detail_pngs,
            *paths.detail_pdfs,
        )
    }

    repeated = plot_rewiring_histograms(
        raw,
        tmp_path / "figures",
        expected_sizes=(8, 12),
        expected_beta_keys=(0, 100_000_000, 1_000_000_000),
        expected_n_graphs=8,
    )
    for path in (
        repeated.overview_png,
        repeated.overview_pdf,
        *repeated.detail_pngs,
        *repeated.detail_pdfs,
    ):
        assert path.read_bytes() == first[path]
    assert not list((tmp_path / "figures").rglob(".*.tmp"))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("schema", "schema/order differs"),
        ("dtype", "displaced must have dtype"),
        ("shape", "operations must have shape"),
        ("identity", "operations - restored"),
        ("beta_zero", "beta=0 displaced"),
        ("range", "displaced exceeds 2n"),
        ("beta_key", "beta is inconsistent"),
        ("graph_index", "canonical contiguous"),
    ],
)
def test_reader_rejects_corrupt_archives(tmp_path: Path, mutation: str, match: str) -> None:
    arrays = _arrays()
    if mutation == "schema":
        arrays = {name: arrays[name] for name in reversed(RAW_FIELDS)}
    elif mutation == "dtype":
        arrays["displaced"] = arrays["displaced"].astype(np.int64)
    elif mutation == "shape":
        arrays["operations"] = arrays["operations"][:, :, :-1]
    elif mutation == "identity":
        arrays["operations"][0, 1, 0] += 1
    elif mutation == "beta_zero":
        arrays["operations"][0, 0, 0] = 1
        arrays["displaced"][0, 0, 0] = 1
    elif mutation == "range":
        arrays["operations"][0, 1, 0] = 17
        arrays["displaced"][0, 1, 0] = 17
    elif mutation == "beta_key":
        arrays["beta"][1] = 0.11
    elif mutation == "graph_index":
        arrays["graph_index"][1] = 2
    raw = tmp_path / "bad.npz"
    _write(raw, arrays)

    with pytest.raises(ValueError, match=match):
        read_rewiring_raw(raw)


def test_validation_finishes_before_output_creation(tmp_path: Path) -> None:
    arrays = _arrays()
    arrays["displaced"] = arrays["displaced"].astype(np.int64)
    raw = tmp_path / "bad.npz"
    _write(raw, arrays)
    output = tmp_path / "figures"

    with pytest.raises(ValueError, match="displaced must have dtype"):
        plot_rewiring_histograms(raw, output)
    assert not output.exists()
