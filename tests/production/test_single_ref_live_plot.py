from __future__ import annotations

import csv
from pathlib import Path

import pytest
from studies.prl_production.analysis import plot_live as plot_module
from studies.prl_production.analysis.plot_live import plot_live_summaries

RUN_ID = "0123456789abcdef"
FIELDS = [
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
]


def _row(
    point_index: int,
    *,
    n: int,
    beta: float,
    p: float,
    median: int | None,
    ci_lower: int | None = None,
    ci_upper: int | None = None,
) -> dict[str, object]:
    n_events = 70 if median is not None else 40
    return {
        "analysis_status": "PRELIMINARY",
        "run_id": RUN_ID,
        "point_index": point_index,
        "n": n,
        "beta": format(beta, ".12g"),
        "beta_key": round(beta * 1_000_000_000),
        "p": format(p, ".12g"),
        "p_key": round(p * 1_000_000),
        "t_max": 8 * n,
        "n_trajectories": 100,
        "n_events": n_events,
        "n_censored": 100 - n_events,
        "event_fraction": n_events / 100,
        "median_tau_p": "" if median is None else median,
        "median_resolved": int(median is not None),
        "median_ci_lower": "" if ci_lower is None else ci_lower,
        "median_ci_upper": "" if ci_upper is None else ci_upper,
        "median_ci_resolved": int(ci_lower is not None and ci_upper is not None),
        "bootstrap_resolved_fraction": 0.8,
        "bootstrap_resamples": 200,
        "bootstrap_confidence": 0.95,
        "survival_at_cap": 0.3 if median is not None else 0.6,
    }


def _write_summary(path: Path, rows: list[dict[str, object]], fields: list[str] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_live_plot_writes_one_raw_plot_per_eligible_beta_and_atomic_index(
    tmp_path: Path,
) -> None:
    rows = [
        _row(0, n=8, beta=0.01, p=0.20, median=22, ci_lower=18, ci_upper=28),
        _row(1, n=8, beta=0.01, p=0.21, median=None, ci_lower=64),
        _row(2, n=16, beta=0.01, p=0.20, median=46, ci_lower=38, ci_upper=None),
        _row(3, n=16, beta=0.01, p=0.21, median=35, ci_lower=30, ci_upper=44),
        _row(4, n=8, beta=0.02, p=0.20, median=20, ci_lower=16, ci_upper=25),
    ]
    summary = tmp_path / "point_summary.csv"
    _write_summary(summary, rows)
    output = tmp_path / "figures"

    result = plot_live_summaries(
        summary,
        run_id=RUN_ID,
        output_dir=output,
        minimum_points=2,
    )

    assert result.validated_rows == 5
    assert result.plotted_betas == (0.01,)
    assert result.skipped_betas == (0.02,)
    assert [path.name for path in result.plot_paths] == ["tau_p_b0010000000.png"]
    assert result.plot_paths[0].stat().st_size > 1_000
    assert sorted(path.name for path in output.glob("*.png")) == ["tau_p_b0010000000.png"]

    index = (output / "LIVE_PLOTS.md").read_text(encoding="utf-8")
    assert "PRELIMINARY" in index
    assert f"Run ID: `{RUN_ID}`" in index
    assert "beta=0.02" in index
    assert "n=8: 2" in index and "n=16: 2" in index
    assert "[tau_p_b0010000000.png](tau_p_b0010000000.png)" in index
    assert "no smoothing or shaded bands" in index
    assert not list(output.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("partial_bootstrap", "schema differs"),
        ("count_mismatch", "do not sum"),
        ("duplicate_point", "duplicate .* points"),
        ("partial_status", "analysis_status must be PRELIMINARY"),
    ],
)
def test_live_plot_rejects_malformed_or_noncomplete_summaries(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    rows = [_row(0, n=8, beta=0.01, p=0.20, median=22, ci_lower=18, ci_upper=28)]
    fields = list(FIELDS)
    if mutation == "partial_bootstrap":
        fields.remove("median_ci_upper")
        rows[0].pop("median_ci_upper")
    elif mutation == "count_mismatch":
        rows[0]["n_censored"] = 29
    elif mutation == "duplicate_point":
        rows.append(dict(rows[0], point_index=1))
    elif mutation == "partial_status":
        rows[0]["analysis_status"] = "PARTIAL"
    summary = tmp_path / "point_summary.csv"
    _write_summary(summary, rows, fields)

    with pytest.raises(ValueError, match=match):
        plot_live_summaries(
            summary,
            run_id=RUN_ID,
            output_dir=tmp_path / "figures",
            minimum_points=1,
        )
    assert not (tmp_path / "figures" / "LIVE_PLOTS.md").exists()


def test_live_plot_cli_accepts_explicit_summary_and_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = tmp_path / "point_summary.csv"
    _write_summary(
        summary,
        [_row(0, n=8, beta=0.01, p=0.20, median=22, ci_lower=18, ci_upper=28)],
    )
    output = tmp_path / "custom"

    assert (
        plot_module.main(
            [
                "--run-id",
                RUN_ID,
                "--summary",
                str(summary),
                "--output",
                str(output),
                "--minimum-points",
                "1",
            ]
        )
        == 0
    )
    assert "wrote 1 beta plots" in capsys.readouterr().out
    assert (output / "LIVE_PLOTS.md").exists()
