from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

from studies.prl_production.analysis import fit_beta
from studies.prl_production.analysis.scaling import ProfileLossLandscape

RUN_ID = "0123456789abcdef"
BETA_KEY = 10_000_000


def _row(point_index: int, *, n: int, p: float, beta: float = 0.01) -> dict[str, object]:
    x = (p - 0.25) * n ** (1.0 / 1.4)
    median = max(2, int(round(n**0.75 * math.exp(0.25 - 0.18 * x))))
    lower = max(1, int(math.floor(0.88 * median)))
    upper = min(8 * n, int(math.ceil(1.12 * median)))
    return {
        "analysis_status": "PRELIMINARY",
        "run_id": RUN_ID,
        "point_index": point_index,
        "n": n,
        "beta": format(beta, ".9g"),
        "beta_key": round(beta * 1_000_000_000),
        "p": format(p, ".6f"),
        "p_key": round(p * 1_000_000),
        "t_max": 8 * n,
        "n_trajectories": 500,
        "n_events": 400,
        "n_censored": 100,
        "event_fraction": "0.8",
        "median_tau_p": median,
        "median_resolved": 1,
        "median_ci_lower": lower,
        "median_ci_upper": upper,
        "median_ci_resolved": 1,
        "bootstrap_resolved_fraction": "1",
        "bootstrap_resamples": 50,
        "bootstrap_confidence": "0.68",
        "survival_at_cap": "0.2",
    }


def _write_summary(path: Path, *, extra_field: bool = False) -> None:
    fields = list(fit_beta._SUMMARY_FIELDS)
    if extra_field:
        fields.append("unexpected")
    rows: list[dict[str, object]] = []
    index = 0
    for beta in (0.01, 0.02):
        for n in (32, 64, 128):
            for p in (0.22, 0.24, 0.26, 0.28):
                row = _row(index, n=n, p=p, beta=beta)
                if extra_field:
                    row["unexpected"] = "value"
                rows.append(row)
                index += 1
            if beta == 0.01:
                unresolved = _row(index, n=n, p=0.20, beta=beta)
                unresolved.update(
                    {
                        "median_tau_p": "",
                        "median_resolved": 0,
                        "median_ci_lower": "",
                        "median_ci_upper": "",
                        "median_ci_resolved": 0,
                    }
                )
                rows.append(unresolved)
                index += 1
                cap_limited = _row(index, n=n, p=0.30, beta=beta)
                cap_limited.update(
                    {
                        "median_ci_upper": "",
                        "median_ci_resolved": 0,
                    }
                )
                rows.append(cap_limited)
                index += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fake_fit(data):
    pc, nu, z = 0.25, 1.4, 0.75
    coordinate = (data.p - pc) * data.n ** (1.0 / nu)
    residual = np.linspace(-0.02, 0.02, len(data.n))
    diagnostics = SimpleNamespace(
        message="synthetic validated fit",
        n_points=len(data.n),
        n_sizes=len(data.sizes),
        n_starts=2,
        n_valid_starts=2,
        best_start_index=0,
        objective=0.01,
        weighted_rmse=0.02,
        chi_square=1.0,
        effective_master_parameters=3.0,
        effective_degrees_of_freedom=float(len(data.n) - 6),
        reduced_chi_square=0.2,
        spline_roughness=0.01,
        condition_number=20.0,
        pc_inside_common_window=True,
        boundary_parameters=(),
        attempts=(),
    )
    master = SimpleNamespace(
        predict=lambda values: np.exp(0.25 - 0.18 * np.asarray(values, dtype=float))
    )
    return SimpleNamespace(
        success=True,
        beta=data.beta,
        pc=pc,
        nu=nu,
        z=z,
        loss=0.01,
        master_curve=master,
        scaling_coordinate=coordinate,
        fitted_log_tau=data.log_tau - residual,
        residual=residual,
        standardized_residual=residual / data.log_sigma,
        diagnostics=diagnostics,
    )


def _fake_landscapes(**kwargs):
    pairs = (("pc", "nu", "z"), ("pc", "z", "nu"), ("nu", "z", "pc"))
    result = {}
    grids = {
        "pc": np.asarray(kwargs["pc_values"]),
        "nu": np.asarray(kwargs["nu_values"]),
        "z": np.asarray(kwargs["z_values"]),
    }
    for x_name, y_name, hidden in pairs:
        shape = (len(grids[y_name]), len(grids[x_name]))
        loss = np.arange(np.prod(shape), dtype=float).reshape(shape)
        result[(x_name, y_name)] = ProfileLossLandscape(
            x_parameter=x_name,
            y_parameter=y_name,
            optimized_parameter=hidden,
            x_values=grids[x_name],
            y_values=grids[y_name],
            loss=loss,
            delta_loss=loss - np.min(loss),
            optimized_values=np.full(shape, float(np.mean(grids[hidden]))),
            valid=np.ones(shape, dtype=bool),
            optimizer_calls=np.prod(shape),
        )
    return result


def test_diagnostics_use_only_selected_summary_rows_and_publish_labeled_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "point_summary.csv"
    _write_summary(summary)
    output = tmp_path / "figures" / f"beta_{BETA_KEY:010d}"
    observed: dict[str, object] = {}

    def fake_fit(data, **kwargs):
        observed["fit_point_indices"] = tuple(int(value) for value in data.point_index)
        observed["fit_options"] = kwargs
        return _fake_fit(data)

    def fake_landscape(data, **kwargs):
        observed["landscape_points"] = len(data.n)
        observed["landscape_options"] = kwargs
        return _fake_landscapes(**kwargs)

    original_atomic_figure = fit_beta._atomic_figure
    labels: list[tuple[str, str, str]] = []
    legend_labels: dict[str, list[str]] = {}

    def capture_figure(figure, path):
        axis = figure.axes[0]
        labels.append((axis.get_xlabel(), axis.get_ylabel(), axis.get_title()))
        legend = axis.get_legend()
        legend_labels[path.name] = (
            [] if legend is None else [text.get_text() for text in legend.get_texts()]
        )
        original_atomic_figure(figure, path)

    monkeypatch.setattr(fit_beta, "fit_three_parameter_collapse", fake_fit)
    monkeypatch.setattr(fit_beta, "profile_pairwise_landscapes", fake_landscape)
    monkeypatch.setattr(fit_beta, "_atomic_figure", capture_figure)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("raw NPZ read"))

    result = fit_beta.fit_beta_diagnostics(
        summary,
        run_id=RUN_ID,
        beta_key=BETA_KEY,
        output_dir=output,
        include_landscapes=True,
        landscape_grid_size=3,
        profile_intervals=1,
        landscape_maxiter=10,
    )

    assert len(observed["fit_point_indices"]) == 12
    assert max(observed["fit_point_indices"]) < 18
    assert observed["landscape_points"] == 12
    assert observed["landscape_options"]["max_cells"] == 9
    assert result.beta == pytest.approx(0.01)
    assert len(result.figure_paths) == 6
    assert {path.name for path in result.figure_paths} == {
        "raw.png",
        "collapse.png",
        "residual.png",
        "profile_pc_nu.png",
        "profile_pc_z.png",
        "profile_nu_z.png",
    }
    for path in result.figure_paths:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (
        r"Measurement probability $p$",
        r"Median purification time $\tau_p$ (layers)",
        "",
    ) in labels
    assert (r"$(p-p_c)n^{1/\nu}$", r"$\tau_p/n^z$", "") in labels
    assert (r"$(p-p_c)n^{1/\nu}$", "Standardized residual", "") in labels
    assert any(x == r"$p_c$" and y == r"$\nu$" and "profiled" in title for x, y, title in labels)
    assert r"median $>T_{\max}$" in legend_labels["raw.png"]
    assert "upper CI cap-limited" in legend_labels["raw.png"]

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["analysis_status"] == "PRELIMINARY"
    assert payload["input"]["source_kind"] == "validated point_summary.csv only"
    assert payload["selection"]["total_records"] == 18
    assert payload["selection"]["usable_points"] == 12
    assert payload["selection"]["unresolved_medians"] == 3
    assert payload["selection"]["unresolved_bootstrap_intervals"] == 3
    assert payload["raw_plot_accounting"] == {
        "interval_unavailable": 0,
        "resolved_medians": 15,
        "resolved_two_sided_intervals": 12,
        "total_rows": 18,
        "unresolved_medians": 3,
        "upper_ci_cap_limited": 3,
    }
    assert payload["landscape_options"]["total_cells"] == 27
    assert set(payload["landscapes"]) == {"pc_nu", "pc_z", "nu_z"}
    assert "No smoothing across beta" in payload["interpretation"]
    assert not list(output.glob(".*.tmp"))


def test_default_fit_is_light_and_does_not_call_landscapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = tmp_path / "point_summary.csv"
    _write_summary(summary)
    monkeypatch.setattr(
        fit_beta,
        "fit_three_parameter_collapse",
        lambda data, **kwargs: _fake_fit(data),
    )
    monkeypatch.setattr(
        fit_beta,
        "profile_pairwise_landscapes",
        lambda *args, **kwargs: pytest.fail("landscapes must be opt-in"),
    )

    result = fit_beta.fit_beta_diagnostics(
        summary,
        run_id=RUN_ID,
        beta_key=BETA_KEY,
        output_dir=tmp_path / "output",
    )

    assert not result.landscapes_included
    assert {path.name for path in result.figure_paths} == {
        "raw.png",
        "collapse.png",
        "residual.png",
    }


def test_schema_and_work_limits_fail_before_publishing(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.csv"
    _write_summary(malformed, extra_field=True)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="schema or field order"):
        fit_beta.fit_beta_diagnostics(
            malformed,
            run_id=RUN_ID,
            beta_key=BETA_KEY,
            output_dir=output,
        )
    assert not output.exists()

    max_grid = fit_beta.MAX_LANDSCAPE_GRID_SIZE + 1
    with pytest.raises(ValueError, match="landscape_grid_size"):
        fit_beta.fit_beta_diagnostics(
            tmp_path / "does-not-exist.csv",
            run_id=RUN_ID,
            beta_key=BETA_KEY,
            include_landscapes=True,
            landscape_grid_size=max_grid,
        )
    assert max_grid == 16
