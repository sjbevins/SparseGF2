from __future__ import annotations

import csv
import hashlib
import json
import sqlite3

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from studies.prl_production.graph_registry.analyze_rewiring import (
    INVARIANT_KEY,
    analyze_collection_rewiring,
)
from studies.prl_production.graph_registry.build import build_collection
from studies.prl_production.graph_registry.plot_rewiring import read_rewiring_summary
from studies.prl_production.graph_registry.spec import smoke_spec


def _sha256(path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def test_smoke_rewiring_analysis_is_complete_registered_and_idempotent(tmp_path) -> None:
    spec = smoke_spec()
    database_path = tmp_path / "registry" / "graph_registry.sqlite3"
    build = build_collection(spec, database_path)
    output_dir = tmp_path / "rewiring"
    figure_dir = tmp_path / "figures"

    first = analyze_collection_rewiring(
        spec,
        database_path,
        workers=1,
        output_dir=output_dir,
        figure_dir=figure_dir,
        progress=False,
    )
    first_hashes = {
        path.name: _sha256(path)
        for path in (
            first.summary_csv,
            first.raw_npz,
            first.run_manifest,
            first.plots.png,
            first.plots.pdf,
            first.histograms.overview_png,
            first.histograms.overview_pdf,
            *first.histograms.detail_pngs,
            *first.histograms.detail_pdfs,
        )
    }
    second = analyze_collection_rewiring(
        spec,
        database_path,
        workers=1,
        output_dir=output_dir,
        figure_dir=figure_dir,
        progress=False,
    )

    points = read_rewiring_summary(
        second.summary_csv,
        expected_sizes=spec.sizes,
        expected_beta_keys=spec.beta_keys,
        expected_n_graphs=spec.graphs_per_cell,
    )
    assert len(points) == spec.n_cells
    assert all(
        point.mean_displaced == point.sem_displaced == 0.0 for point in points if not point.beta
    )
    assert first.displaced_logical_sha256 == second.displaced_logical_sha256
    assert first_hashes == {
        path.name: _sha256(path)
        for path in (
            second.summary_csv,
            second.raw_npz,
            second.run_manifest,
            second.plots.png,
            second.plots.pdf,
            second.histograms.overview_png,
            second.histograms.overview_pdf,
            *second.histograms.detail_pngs,
            *second.histograms.detail_pdfs,
        )
    }

    with np.load(second.raw_npz, allow_pickle=False) as raw:
        assert raw["displaced"].shape == (2, 3, 4)
        assert np.array_equal(raw["operations"] - raw["restored"], raw["displaced"])
        assert not np.any(raw["displaced"][:, 0])
        assert str(raw["displaced_logical_sha256"].item()) == second.displaced_logical_sha256

    with sqlite3.connect(database_path) as connection:
        invariant = connection.execute(
            "SELECT invariant_id FROM invariant_definitions WHERE invariant_key = ?",
            (INVARIANT_KEY,),
        ).fetchone()
        assert invariant is not None
        result_count = connection.execute(
            "SELECT count(*) FROM invariant_results WHERE invariant_id = ? AND status = 'complete'",
            (invariant[0],),
        ).fetchone()[0]
        artifact_count = connection.execute(
            "SELECT count(*) FROM artifact_references WHERE invariant_id = ?",
            (invariant[0],),
        ).fetchone()[0]
    assert result_count == spec.n_graphs
    assert artifact_count == 3
    assert "Status: **complete and validated**" in second.report.read_text(encoding="utf-8")
    manifest = json.loads(second.run_manifest.read_text(encoding="utf-8"))
    assert manifest["logical_displaced_sha256"] == second.displaced_logical_sha256
    assert manifest["graph_count"] == spec.n_graphs
    assert build.validation.seed_content_sha256 == manifest["seed_content_sha256"]

    with second.summary_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(int(row["n"]), int(row["beta_key"])) for row in rows] == [
        (n, beta_key) for n in spec.sizes for beta_key in spec.beta_keys
    ]
