from __future__ import annotations

import math

import numpy as np
import pytest
from studies.prl_production.analysis.survival import (
    kaplan_meier,
    summarize_purification_point,
)


def test_kaplan_meier_known_curve_with_tied_censoring() -> None:
    curve = kaplan_meier(
        stop_times=[1, 2, 2, 3, 4],
        event_observed=[1, 1, 0, 1, 0],
    )

    assert curve.times == (1, 2, 3, 4)
    assert curve.n_at_risk == (5, 4, 2, 1)
    assert curve.n_events == (1, 1, 1, 0)
    assert curve.n_censored == (0, 1, 0, 1)
    assert curve.survival == pytest.approx((0.8, 0.6, 0.3, 0.3))
    expected_se = (
        0.8 * math.sqrt(1 / 20),
        0.6 * math.sqrt(1 / 20 + 1 / 12),
        0.3 * math.sqrt(1 / 20 + 1 / 12 + 1 / 2),
        0.3 * math.sqrt(1 / 20 + 1 / 12 + 1 / 2),
    )
    assert curve.greenwood_se == pytest.approx(expected_se)
    assert curve.median == 3
    assert curve.sample_size == 5
    assert curve.event_count == 3
    assert curve.censored_count == 2
    assert curve.survival_at(0) == 1.0
    assert curve.survival_at(2) == pytest.approx(0.6)
    assert curve.greenwood_se_at(0) == 0.0


def test_cap_event_and_cap_censor_share_the_risk_set() -> None:
    curve = kaplan_meier(stop_times=[64, 64], event_observed=[1, 0])

    assert curve.n_at_risk == (2,)
    assert curve.n_events == (1,)
    assert curve.n_censored == (1,)
    assert curve.survival == pytest.approx((0.5,))
    assert curve.median == 64


def test_all_censored_median_is_unresolved() -> None:
    curve = kaplan_meier(stop_times=[64, 64, 64], event_observed=[0, 0, 0])

    assert curve.survival == (1.0,)
    assert curve.greenwood_se == (0.0,)
    assert curve.median is None


def test_complete_event_has_zero_survival_and_finite_error() -> None:
    curve = kaplan_meier(stop_times=[1, 1], event_observed=[1, 1])

    assert curve.survival == (0.0,)
    assert curve.greenwood_se == (0.0,)
    assert curve.median == 1


def test_point_summary_preserves_exact_cap_semantics() -> None:
    result = summarize_purification_point(
        n=8,
        beta=0.1,
        p=0.25,
        t_max=64,
        tau_p=np.asarray([2, 64, -1, -1], dtype=np.int32),
        stop_layer=np.asarray([2, 64, 64, 64], dtype=np.int32),
        event_observed=np.asarray([1, 1, 0, 0], dtype=np.uint8),
    )

    summary = result.summary
    assert summary.n_trajectories == 4
    assert summary.n_events == 2
    assert summary.n_censored == 2
    assert summary.event_fraction == 0.5
    assert summary.median_tau_p == 64
    assert summary.median_resolved
    assert summary.survival_at_cap == 0.5


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tau_p": [1.0]}, "tau_p must contain integers"),
        ({"stop_layer": [0]}, "stop_layer must lie"),
        ({"event_observed": [2]}, "only zero or one"),
        ({"tau_p": [-1], "stop_layer": [32]}, "censored rows require"),
        ({"tau_p": [2], "stop_layer": [3], "event_observed": [1]}, "observed rows"),
        ({"t_max": 63}, "must equal 8\\*n"),
    ],
)
def test_point_summary_rejects_malformed_records(
    kwargs: dict[str, object],
    match: str,
) -> None:
    inputs: dict[str, object] = {
        "n": 8,
        "beta": 0.1,
        "p": 0.25,
        "t_max": 64,
        "tau_p": [-1],
        "stop_layer": [64],
        "event_observed": [0],
    }
    inputs.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=match):
        summarize_purification_point(**inputs)


@pytest.mark.parametrize(
    ("stop_times", "events", "error", "match"),
    [
        ([], [], ValueError, "at least one"),
        ([[1]], [1], ValueError, "one-dimensional"),
        ([1.0], [1], TypeError, "contain integers"),
        ([0], [1], ValueError, "positive"),
        ([1, 2], [1], ValueError, "same length"),
        ([1], [0.0], TypeError, "booleans or binary integers"),
        ([1], [2], ValueError, "only zero or one"),
    ],
)
def test_kaplan_meier_rejects_invalid_inputs(
    stop_times: object,
    events: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        kaplan_meier(stop_times, events)
