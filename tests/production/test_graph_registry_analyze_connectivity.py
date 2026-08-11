from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")

from studies.prl_production.graph_registry.analyze_connectivity import (
    CELL_SUMMARY_FIELDS,
    INVARIANT_KEY,
    NESTED_SUMMARY_FIELDS,
    _default_nested_size_sets,
    analyze_collection_connectivity,
    main,
)
from studies.prl_production.graph_registry.build import build_collection
from studies.prl_production.graph_registry.connectivity_metrics import (
    ring_algebraic_connectivity,
)
from studies.prl_production.graph_registry.spec import production_spec, smoke_spec


def _sha256(path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _csv_rows(path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_smoke_connectivity_analysis_is_registered_resumable_and_exact(tmp_path) -> None:
    spec = smoke_spec()
    database_path = tmp_path / "registry" / "graph_registry.sqlite3"
    build_collection(spec, database_path)
    output_dir = tmp_path / "connectivity"
    figure_dir = tmp_path / "figures"

    first = analyze_collection_connectivity(
        spec,
        database_path,
        workers=1,
        bootstrap_resamples=64,
        bootstrap_seed=17,
        output_dir=output_dir,
        figure_dir=figure_dir,
        progress=False,
    )
    published = (
        first.cell_summary_csv,
        first.nested_summary_csv,
        first.raw_npz,
        first.run_manifest,
        first.plots.png,
        first.plots.pdf,
    )
    first_hashes = {path.name: _sha256(path) for path in published}
    first_cell_hashes = {
        path.name: _sha256(path) for path in sorted((output_dir / "cells").glob("*.npz"))
    }

    second = analyze_collection_connectivity(
        spec,
        database_path,
        workers=1,
        bootstrap_resamples=64,
        bootstrap_seed=17,
        output_dir=output_dir,
        figure_dir=figure_dir,
        progress=False,
    )
    assert first.lambda2_logical_sha256 == second.lambda2_logical_sha256
    assert first_hashes == {
        path.name: _sha256(path)
        for path in (
            second.cell_summary_csv,
            second.nested_summary_csv,
            second.raw_npz,
            second.run_manifest,
            second.plots.png,
            second.plots.pdf,
        )
    }
    assert first_cell_hashes == {
        path.name: _sha256(path) for path in sorted((output_dir / "cells").glob("*.npz"))
    }

    with np.load(second.raw_npz, allow_pickle=False) as raw:
        assert raw["lambda2"].shape == (2, 3, 4)
        assert raw["graph_seed"].shape == raw["lambda2"].shape
        assert np.array_equal(raw["n"], np.asarray(spec.sizes, dtype=np.int32))
        assert np.array_equal(raw["beta_key"], np.asarray(spec.beta_keys, dtype=np.int64))
        for n_index, n in enumerate(spec.sizes):
            assert np.allclose(
                raw["lambda2"][n_index, 0],
                ring_algebraic_connectivity(n, spec.graph_k),
                rtol=2e-12,
                atol=2e-13,
            )
        lambda2 = np.array(raw["lambda2"], copy=True)

    cell_rows = _csv_rows(second.cell_summary_csv)
    assert tuple(cell_rows[0]) == CELL_SUMMARY_FIELDS
    assert len(cell_rows) == spec.n_cells
    assert [(int(row["n"]), int(row["beta_key"])) for row in cell_rows] == [
        (n, beta_key) for n in spec.sizes for beta_key in spec.beta_keys
    ]

    nested_rows = _csv_rows(second.nested_summary_csv)
    size_sets = _default_nested_size_sets(spec)
    assert tuple(nested_rows[0]) == NESTED_SUMMARY_FIELDS
    assert len(nested_rows) == len(size_sets) * len(spec.beta_keys)
    assert [(int(row["set_size"]), int(row["beta_key"])) for row in nested_rows] == [
        (set_size, beta_key)
        for set_size in range(1, len(size_sets) + 1)
        for beta_key in spec.beta_keys
    ]
    for row in nested_rows:
        size_set = tuple(json.loads(row["size_set"]))
        assert size_set == size_sets[int(row["set_size"]) - 1]
        beta_index = spec.beta_keys.index(int(row["beta_key"]))
        means = np.asarray([np.mean(lambda2[spec.sizes.index(n), beta_index]) for n in size_set])
        sems = np.asarray(
            [
                np.std(lambda2[spec.sizes.index(n), beta_index], ddof=1)
                / math.sqrt(spec.graphs_per_cell)
                for n in size_set
            ]
        )
        rings = np.asarray([ring_algebraic_connectivity(n, spec.graph_k) for n in size_set])
        if int(row["beta_key"]) == 0:
            assert float(row["g_lambda"]) == 1.0
            assert float(row["g_lambda_sem"]) == 0.0
            assert float(row["log_g_lambda"]) == 0.0
            assert float(row["log_g_lambda_sem"]) == 0.0
            assert float(row["ci68_low"]) == float(row["ci68_high"]) == 1.0
            continue
        expected_log = float(np.mean(np.log(means / rings)))
        expected_log_sem = float(np.sqrt(np.sum((sems / means) ** 2)) / len(size_set))
        assert float(row["log_g_lambda"]) == pytest.approx(expected_log, abs=1e-14)
        assert float(row["log_g_lambda_sem"]) == pytest.approx(expected_log_sem, abs=1e-14)
        assert float(row["g_lambda"]) == pytest.approx(math.exp(expected_log), rel=1e-14)
        assert float(row["g_lambda_sem"]) == pytest.approx(
            math.exp(expected_log) * expected_log_sem,
            abs=1e-14,
        )
        assert float(row["ci68_low"]) <= float(row["g_lambda"]) <= float(row["ci68_high"])

    with sqlite3.connect(database_path) as connection:
        invariant_id = connection.execute(
            "SELECT invariant_id FROM invariant_definitions WHERE invariant_key = ?",
            (INVARIANT_KEY,),
        ).fetchone()[0]
        complete = connection.execute(
            "SELECT count(*) FROM invariant_results WHERE invariant_id = ? AND status='complete'",
            (invariant_id,),
        ).fetchone()[0]
        artifacts = connection.execute(
            "SELECT count(*) FROM artifact_references WHERE invariant_id = ?",
            (invariant_id,),
        ).fetchone()[0]
    assert complete == spec.n_graphs
    assert artifacts == 6
    manifest = json.loads(second.run_manifest.read_text(encoding="utf-8"))
    assert manifest["bootstrap"]["shared_cell_resamples_across_cumulative_sets"] is True
    assert manifest["nested_size_sets"] == [list(values) for values in size_sets]
    assert manifest["lambda2_logical_sha256"] == second.lambda2_logical_sha256
    assert "Status: **complete and validated**" in second.report.read_text(encoding="utf-8")


def test_existing_corrupt_cell_is_rejected_instead_of_silently_recomputed(tmp_path) -> None:
    spec = smoke_spec()
    database_path = tmp_path / "registry.sqlite3"
    build_collection(spec, database_path)
    output_dir = tmp_path / "connectivity"
    analyze_collection_connectivity(
        spec,
        database_path,
        workers=1,
        bootstrap_resamples=8,
        output_dir=output_dir,
        figure_dir=tmp_path / "figures",
        progress=False,
    )
    cell = sorted((output_dir / "cells").glob("*.npz"))[0]
    cell.write_bytes(b"not an NPZ archive")
    with pytest.raises((OSError, ValueError)):
        analyze_collection_connectivity(
            spec,
            database_path,
            workers=1,
            bootstrap_resamples=8,
            output_dir=output_dir,
            figure_dir=tmp_path / "figures",
            progress=False,
        )


def test_production_cli_requires_confirmation_and_dry_run_is_write_free(tmp_path, capsys) -> None:
    database = tmp_path / "must_not_exist.sqlite3"
    with pytest.raises(SystemExit, match="requires --confirm-production"):
        main(["--profile", "production", "--database", str(database)])
    assert main(["--profile", "production", "--database", str(database), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["collection_id"] == production_spec().collection_id
    assert payload["nested_size_sets"] == [
        list(values) for values in ((64,), (64, 128), (64, 128, 192), (64, 128, 192, 256))
    ]
    assert not database.exists()
