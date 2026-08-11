from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from studies.prl_production.analysis import scaling
from studies.prl_production.analysis.scaling import (
    CollapseBounds,
    fit_three_parameter_collapse,
    prepare_collapse_data,
    profile_pairwise_landscapes,
)


def _synthetic_records(
    *,
    include_excluded: bool = True,
) -> list[dict[str, object]]:
    pc, nu, z = 0.25, 1.45, 0.82
    rows: list[dict[str, object]] = []
    point_index = 0
    for n in (32, 64, 128, 256):
        for p in np.linspace(0.205, 0.295, 11):
            x = (p - pc) * n ** (1.0 / nu)
            log_master = 0.45 - 0.38 * x + 0.10 * x**2 + 0.025 * x**3
            median = int(round(n**z * math.exp(log_master)))
            lower = max(1, int(round(median * 0.92)))
            upper = min(8 * n, int(round(median * 1.08)))
            rows.append(
                {
                    "point_index": point_index,
                    "n": n,
                    "beta": "0.01",
                    "beta_key": "10000000",
                    "p": format(float(p), ".6f"),
                    "p_key": str(round(float(p) * 1_000_000)),
                    "t_max": 8 * n,
                    "median_tau_p": median,
                    "median_resolved": 1,
                    "median_ci_lower": lower,
                    "median_ci_upper": upper,
                    "median_ci_resolved": 1,
                    "bootstrap_resamples": 500,
                    "bootstrap_confidence": "0.68",
                }
            )
            point_index += 1
        if include_excluded:
            # The unresolved median is represented only as censoring; it must
            # not enter the fit as tau=8n.
            rows.append(
                {
                    "point_index": point_index,
                    "n": n,
                    "beta": "0.01",
                    "beta_key": "10000000",
                    "p": "0.190000",
                    "p_key": "190000",
                    "t_max": 8 * n,
                    "median_tau_p": "",
                    "median_resolved": 0,
                    "median_ci_lower": "",
                    "median_ci_upper": "",
                    "median_ci_resolved": 0,
                    "bootstrap_resamples": 500,
                    "bootstrap_confidence": "0.68",
                }
            )
            point_index += 1
    return rows


@pytest.fixture
def synthetic_data():
    return prepare_collapse_data(_synthetic_records())


def _tight_bounds() -> CollapseBounds:
    return CollapseBounds(pc=(0.225, 0.275), nu=(0.90, 2.20), z=(0.55, 1.10))


def test_profiled_fit_recovers_synthetic_collapse_without_cap_imputation(synthetic_data) -> None:
    data = synthetic_data
    assert data.selection.total_records == 48
    assert data.selection.usable_points == 44
    assert data.selection.unresolved_medians == 4
    assert data.selection.unresolved_bootstrap_intervals == 0
    assert len(data.tau) == 44
    assert np.all(data.tau < 8 * data.n)

    fit = fit_three_parameter_collapse(
        data,
        bounds=_tight_bounds(),
        interior_knots=1,
        smoothing=0.01,
        n_starts=6,
        maxiter=250,
    )

    assert fit.success
    assert fit.pc == pytest.approx(0.25, abs=0.012)
    assert fit.nu == pytest.approx(1.45, abs=0.40)
    assert fit.z == pytest.approx(0.82, abs=0.12)
    assert fit.master_curve is not None
    assert fit.diagnostics.n_valid_starts > 0
    assert fit.diagnostics.pc_inside_common_window
    assert fit.diagnostics.weighted_rmse < 0.04
    assert np.all(np.isfinite(fit.standardized_residual))


def test_input_rejects_mixed_beta_and_all_unresolved_records() -> None:
    mixed = _synthetic_records(include_excluded=False)
    mixed[-1] = {
        **mixed[-1],
        "beta": "0.02",
        "beta_key": "20000000",
    }
    with pytest.raises(ValueError, match="exactly one beta"):
        prepare_collapse_data(mixed)

    unresolved = _synthetic_records(include_excluded=False)
    for row in unresolved:
        row.update(
            {
                "median_tau_p": "",
                "median_resolved": 0,
                "median_ci_lower": "",
                "median_ci_upper": "",
                "median_ci_resolved": 0,
            }
        )
    with pytest.raises(ValueError, match="no resolved medians"):
        prepare_collapse_data(unresolved)


def test_bounded_multistart_is_deterministic(synthetic_data) -> None:
    options = {
        "bounds": _tight_bounds(),
        "interior_knots": 1,
        "smoothing": 0.01,
        "n_starts": 4,
        "maxiter": 180,
    }
    first = fit_three_parameter_collapse(synthetic_data, **options)
    second = fit_three_parameter_collapse(synthetic_data, **options)

    assert first.success and second.success
    assert first.parameters == pytest.approx(second.parameters, rel=0.0, abs=1e-12)
    assert first.loss == pytest.approx(second.loss, rel=0.0, abs=1e-14)
    assert first.diagnostics.best_start_index == second.diagnostics.best_start_index
    assert first.diagnostics.n_valid_starts == second.diagnostics.n_valid_starts


def test_failed_optimizer_results_return_nan_not_a_fallback(
    synthetic_data,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_optimizer(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            success=False,
            x=np.asarray([0.25, 1.45, 0.82]),
            fun=0.0,
            message="forced failure",
        )

    monkeypatch.setattr(scaling, "_minimize", fail_optimizer)
    fit = fit_three_parameter_collapse(
        synthetic_data,
        bounds=_tight_bounds(),
        interior_knots=1,
        n_starts=3,
        maxiter=20,
    )

    assert not fit.success
    assert all(math.isnan(value) for value in fit.parameters)
    assert math.isnan(fit.loss)
    assert fit.master_curve is None
    assert fit.diagnostics.n_valid_starts == 0
    assert all(not attempt.accepted for attempt in fit.diagnostics.attempts)


def test_all_pairwise_landscapes_reoptimize_the_hidden_parameter(synthetic_data) -> None:
    bounds = _tight_bounds()
    landscapes = profile_pairwise_landscapes(
        synthetic_data,
        pc_values=np.asarray([0.245, 0.255]),
        nu_values=np.asarray([1.30, 1.60]),
        z_values=np.asarray([0.76, 0.88]),
        bounds=bounds,
        interior_knots=1,
        smoothing=0.01,
        profile_intervals=1,
        maxiter=60,
    )

    assert set(landscapes) == {("pc", "nu"), ("pc", "z"), ("nu", "z")}
    expected_hidden = {("pc", "nu"): "z", ("pc", "z"): "nu", ("nu", "z"): "pc"}
    for pair, landscape in landscapes.items():
        assert landscape.optimized_parameter == expected_hidden[pair]
        assert landscape.loss.shape == (2, 2)
        assert np.all(landscape.valid)
        assert np.all(np.isfinite(landscape.optimized_values))
        assert float(np.nanmin(landscape.delta_loss)) == pytest.approx(0.0, abs=1e-14)
        hidden_lower, hidden_upper = bounds.interval(landscape.optimized_parameter)
        assert np.all(landscape.optimized_values >= hidden_lower)
        assert np.all(landscape.optimized_values <= hidden_upper)
