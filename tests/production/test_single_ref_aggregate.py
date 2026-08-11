from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from studies.prl_production.analysis import aggregate as aggregate_module
from studies.prl_production.analysis.aggregate import aggregate_manifest
from studies.prl_production.campaign import (
    GRAPH_K,
    MASTER_SEED,
    MEAN_DEGREE,
    SCHEMA_VERSION,
    SCRAMBLE_DEPTH,
    TMAX_FACTOR,
)
from studies.prl_production.single_ref.engine import PointSpec, point_path

RUN_ID = "0123456789abcdef"


def _point_arrays(point: PointSpec) -> dict[str, object]:
    return {
        "schema_version": np.int32(SCHEMA_VERSION),
        "engine": np.str_("single_ref_exact_layer_v1"),
        "n": np.int32(point.n),
        "k": np.int32(GRAPH_K),
        "mean_degree": np.float64(MEAN_DEGREE),
        "beta": np.float64(point.beta),
        "beta_key": np.int64(point.beta_key),
        "p": np.float64(point.p),
        "p_key": np.int64(point.p_key),
        "n_graphs": np.int32(point.n_graphs),
        "tmax_factor": np.int32(TMAX_FACTOR),
        "t_max": np.int32(point.cap),
        "scramble_depth": np.int32(SCRAMBLE_DEPTH),
        "master_seed": np.int64(MASTER_SEED),
        "graph_index": np.arange(point.n_graphs, dtype=np.int32),
        "tau_p": np.full(point.n_graphs, -1, dtype=np.int32),
        "stop_layer": np.zeros(point.n_graphs, dtype=np.int32),
        "event_observed": np.zeros(point.n_graphs, dtype=np.uint8),
        "complete": np.zeros(point.n_graphs, dtype=np.uint8),
    }


def _write_point(data_root: Path, point: PointSpec, arrays: dict[str, object]) -> Path:
    path = point_path(data_root, point)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def _write_manifest(path: Path, data_root: Path, points: list[PointSpec]) -> Path:
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "data_root": str(data_root),
        "n_graphs": points[0].n_graphs,
        "n_points": len(points),
        "n_trajectories": len(points) * points[0].n_graphs,
        "record_traces": False,
        "point_order": [
            {"n": point.n, "beta": point.beta, "p": point.p, "n_graphs": point.n_graphs}
            for point in points
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_live_aggregation_classifies_all_points_and_summarizes_only_complete(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    points = [
        PointSpec(n=8, beta=0.01, p=0.20, n_graphs=4),
        PointSpec(n=8, beta=0.01, p=0.21, n_graphs=4),
        PointSpec(n=8, beta=0.01, p=0.22, n_graphs=4),
    ]
    complete = _point_arrays(points[0])
    complete["tau_p"] = np.asarray([2, 64, -1, -1], dtype=np.int32)
    complete["stop_layer"] = np.asarray([2, 64, 64, 64], dtype=np.int32)
    complete["event_observed"] = np.asarray([1, 1, 0, 0], dtype=np.uint8)
    complete["complete"] = np.ones(4, dtype=np.uint8)
    _write_point(data_root, points[0], complete)

    partial = _point_arrays(points[1])
    partial["tau_p"] = np.asarray([5, -1, -1, -1], dtype=np.int32)
    partial["stop_layer"] = np.asarray([5, 0, 0, 0], dtype=np.int32)
    partial["event_observed"] = np.asarray([1, 0, 0, 0], dtype=np.uint8)
    partial["complete"] = np.asarray([1, 0, 0, 0], dtype=np.uint8)
    _write_point(data_root, points[1], partial)

    manifest = _write_manifest(tmp_path / "manifest.json", data_root, points)
    output = tmp_path / "analysis"
    result = aggregate_manifest(
        manifest,
        output_dir=output,
        expected_run_id=RUN_ID,
    )

    assert (result.absent_points, result.partial_points, result.complete_points) == (1, 1, 1)
    assert result.completed_trajectories == 5
    coverage = _read_csv(output / "coverage.csv")
    assert [row["status"] for row in coverage] == ["complete", "partial", "absent"]
    assert [row["point_index"] for row in coverage] == ["0", "1", "2"]
    assert all(row["analysis_status"] == "PRELIMINARY" for row in coverage)
    assert coverage[1]["completed_trajectories"] == "1"
    assert coverage[2]["pending_trajectories"] == "4"

    summaries = _read_csv(output / "point_summary.csv")
    assert len(summaries) == 1
    assert summaries[0]["analysis_status"] == "PRELIMINARY"
    assert summaries[0]["point_index"] == "0"
    assert summaries[0]["median_tau_p"] == "64"
    assert summaries[0]["n_events"] == "2"
    assert summaries[0]["survival_at_cap"] == "0.5"
    assert summaries[0]["median_ci_lower"] == ""
    assert summaries[0]["median_ci_upper"] == ""
    assert summaries[0]["median_ci_resolved"] == "0"
    assert summaries[0]["bootstrap_resolved_fraction"] == ""
    assert summaries[0]["bootstrap_resamples"] == "0"
    assert summaries[0]["bootstrap_confidence"] == ""

    markdown = (output / "LIVE_ANALYSIS.md").read_text(encoding="utf-8")
    assert markdown.count("PRELIMINARY") >= 2
    assert "Complete points entering Kaplan-Meier summaries: 1" in markdown
    assert "Partial and absent points appear only in `coverage.csv`" in markdown
    assert not list(output.glob(".*.tmp"))


def test_enabled_bootstrap_summary_is_pointwise_and_deterministic(tmp_path: Path) -> None:
    point = PointSpec(n=8, beta=0.01, p=0.20, n_graphs=5)
    arrays = _point_arrays(point)
    arrays["tau_p"] = np.asarray([1, 2, 3, -1, -1], dtype=np.int32)
    arrays["stop_layer"] = np.asarray([1, 2, 3, 64, 64], dtype=np.int32)
    arrays["event_observed"] = np.asarray([1, 1, 1, 0, 0], dtype=np.uint8)
    arrays["complete"] = np.ones(5, dtype=np.uint8)
    data_root = tmp_path / "data"
    _write_point(data_root, point, arrays)
    manifest = _write_manifest(tmp_path / "manifest.json", data_root, [point])

    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        aggregate_manifest(
            manifest,
            output_dir=output,
            bootstrap_resamples=128,
            bootstrap_confidence=0.8,
        )

    first_bytes = (outputs[0] / "point_summary.csv").read_bytes()
    assert first_bytes == (outputs[1] / "point_summary.csv").read_bytes()
    row = _read_csv(outputs[0] / "point_summary.csv")[0]
    assert row["median_tau_p"] == "3"
    assert row["bootstrap_resamples"] == "128"
    assert row["bootstrap_confidence"] == "0.8"
    assert 0.0 <= float(row["bootstrap_resolved_fraction"]) <= 1.0
    assert row["median_ci_lower"]
    assert row["median_ci_resolved"] == str(int(bool(row["median_ci_upper"])))
    assert "128 resamples at confidence 0.8" in (outputs[0] / "LIVE_ANALYSIS.md").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("graph_index", "canonical range"),
        ("incomplete", "incomplete rows require"),
        ("observed", "observed complete rows require"),
        ("censored", "censored complete rows require"),
        ("metadata", "p=.*expected"),
    ],
)
def test_live_aggregation_rejects_malformed_point_rows_and_metadata(
    tmp_path: Path,
    corruption: str,
    match: str,
) -> None:
    point = PointSpec(n=8, beta=0.01, p=0.20, n_graphs=2)
    arrays = _point_arrays(point)
    arrays["tau_p"] = np.asarray([2, -1], dtype=np.int32)
    arrays["stop_layer"] = np.asarray([2, 64], dtype=np.int32)
    arrays["event_observed"] = np.asarray([1, 0], dtype=np.uint8)
    arrays["complete"] = np.ones(2, dtype=np.uint8)
    if corruption == "graph_index":
        arrays["graph_index"] = np.asarray([1, 0], dtype=np.int32)
    elif corruption == "incomplete":
        arrays["complete"] = np.asarray([1, 0], dtype=np.uint8)
        arrays["stop_layer"] = np.asarray([2, 0], dtype=np.int32)
        arrays["tau_p"] = np.asarray([2, 0], dtype=np.int32)
    elif corruption == "observed":
        arrays["tau_p"] = np.asarray([65, -1], dtype=np.int32)
        arrays["stop_layer"] = np.asarray([65, 64], dtype=np.int32)
    elif corruption == "censored":
        arrays["stop_layer"] = np.asarray([2, 63], dtype=np.int32)
    elif corruption == "metadata":
        arrays["p"] = np.float64(0.201)
    data_root = tmp_path / "data"
    _write_point(data_root, point, arrays)
    manifest = _write_manifest(tmp_path / "manifest.json", data_root, [point])

    with pytest.raises(ValueError, match=match):
        aggregate_manifest(manifest, output_dir=tmp_path / "analysis")
    assert not (tmp_path / "analysis" / "coverage.csv").exists()


def test_live_aggregation_requires_canonical_unique_manifest_points(tmp_path: Path) -> None:
    point = PointSpec(n=8, beta=0.01, p=0.20, n_graphs=2)
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        tmp_path / "data",
        [point, point],
    )

    with pytest.raises(ValueError, match="duplicate canonical points"):
        aggregate_manifest(manifest, output_dir=tmp_path / "analysis")


def test_cli_resolves_the_requested_run_manifest_and_custom_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    point = PointSpec(n=8, beta=0.01, p=0.20, n_graphs=2)
    runtime = tmp_path / "runtime"
    _write_manifest(
        runtime / f"single_ref_{RUN_ID}_manifest.json",
        tmp_path / "data",
        [point],
    )
    monkeypatch.setattr(aggregate_module, "RUNTIME_ROOT", runtime)
    output = tmp_path / "custom"

    assert aggregate_module.main(["--run-id", RUN_ID, "--output", str(output)]) == 0
    assert "PRELIMINARY" in capsys.readouterr().out
    assert (output / "coverage.csv").exists()
    assert _read_csv(output / "point_summary.csv") == []
