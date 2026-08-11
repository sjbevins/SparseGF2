from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from studies.prl_production.graph_registry.plot_rewiring import (
    SUMMARY_FIELDS,
    plot_rewiring_summary,
    read_rewiring_summary,
)


def _row(n: int, beta_key: int, n_graphs: int = 8) -> dict[str, object]:
    beta = beta_key / 1_000_000_000
    edges = 2 * n
    operations = edges * beta * 0.9
    restored = operations * 0.08
    displaced = operations - restored
    sem = 0.0 if beta == 0.0 else 0.12 + n / 1_000
    return {
        "collection_id": "synthetic_collection",
        "n": n,
        "beta_key": beta_key,
        "beta": beta,
        "n_graphs": n_graphs,
        "mean_displaced": displaced,
        "sem_displaced": sem,
        "mean_operations": operations,
        "sem_operations": sem * 1.1,
        "mean_restored": restored,
        "sem_restored": sem * 0.25,
        "mean_displaced_fraction": displaced / edges,
        "sem_displaced_fraction": sem / edges,
    }


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


def _complete_rows() -> list[dict[str, object]]:
    return [_row(n, beta_key) for n in (8, 12) for beta_key in (0, 100_000_000, 1_000_000_000)]


def test_plot_writes_atomic_png_and_pdf_from_complete_synthetic_grid(tmp_path: Path) -> None:
    summary = tmp_path / "rewiring_summary.csv"
    _write_summary(summary, _complete_rows())

    paths = plot_rewiring_summary(
        summary,
        tmp_path / "figures",
        expected_sizes=(8, 12),
        expected_beta_keys=(0, 100_000_000, 1_000_000_000),
        expected_n_graphs=8,
    )

    assert paths.png.name == "rewired_edges_vs_beta.png"
    assert paths.pdf.name == "rewired_edges_vs_beta.pdf"
    assert paths.png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert paths.pdf.read_bytes().startswith(b"%PDF-")
    assert paths.png.stat().st_size > 10_000
    assert paths.pdf.stat().st_size > 1_000
    assert not list(paths.png.parent.glob(".*.tmp"))

    first_png = paths.png.read_bytes()
    first_pdf = paths.pdf.read_bytes()
    repeated = plot_rewiring_summary(
        summary,
        tmp_path / "figures",
        expected_sizes=(8, 12),
        expected_beta_keys=(0, 100_000_000, 1_000_000_000),
        expected_n_graphs=8,
    )
    assert repeated.png.read_bytes() == first_png
    assert repeated.pdf.read_bytes() == first_pdf


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("schema", "schema/order differs"),
        ("order", "canonical n-major"),
        ("missing", "complete rectangular grid"),
        ("duplicate", "duplicate .* rows"),
        ("nan", "must be finite"),
        ("fraction", "mean_displaced_fraction is inconsistent"),
        ("identity", "mean_operations - mean_restored"),
        ("mixed_collection", "mixes collection_id"),
    ],
)
def test_reader_rejects_invalid_or_incomplete_summary(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    rows = _complete_rows()
    fields = SUMMARY_FIELDS
    if mutation == "schema":
        fields = tuple(reversed(SUMMARY_FIELDS))
    elif mutation == "order":
        rows[1], rows[2] = rows[2], rows[1]
    elif mutation == "missing":
        rows.pop(1)
    elif mutation == "duplicate":
        rows[1] = dict(rows[0])
    elif mutation == "nan":
        rows[1]["sem_displaced"] = "nan"
    elif mutation == "fraction":
        rows[1]["mean_displaced_fraction"] = 0.99
    elif mutation == "identity":
        rows[1]["mean_restored"] = float(rows[1]["mean_restored"]) + 1.0
    elif mutation == "mixed_collection":
        rows[1]["collection_id"] = "another_collection"
    summary = tmp_path / "rewiring_summary.csv"
    _write_summary(summary, rows, fields)

    with pytest.raises(ValueError, match=match):
        read_rewiring_summary(
            summary,
            expected_sizes=(8, 12),
            expected_beta_keys=(0, 100_000_000, 1_000_000_000),
            expected_n_graphs=8,
        )


def test_reader_rejects_noncanonical_beta_and_wrong_graph_count(tmp_path: Path) -> None:
    rows = _complete_rows()
    rows[1]["beta"] = 0.1000000004
    summary = tmp_path / "bad_beta.csv"
    _write_summary(summary, rows)
    with pytest.raises(ValueError, match="not the canonical value"):
        read_rewiring_summary(summary)

    rows = _complete_rows()
    rows[1]["n_graphs"] = 7
    _write_summary(summary, rows)
    with pytest.raises(ValueError, match="mixes n_graphs"):
        read_rewiring_summary(summary)


def test_plot_validates_before_creating_output_directory(tmp_path: Path) -> None:
    summary = tmp_path / "incomplete.csv"
    _write_summary(summary, _complete_rows()[:-1])
    output = tmp_path / "figures"

    with pytest.raises(ValueError, match="complete rectangular grid"):
        plot_rewiring_summary(
            summary,
            output,
            expected_sizes=(8, 12),
            expected_beta_keys=(0, 100_000_000, 1_000_000_000),
        )
    assert not output.exists()
