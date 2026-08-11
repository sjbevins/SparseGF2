from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from studies.prl_production.analysis.bootstrap import (
    BootstrapMedianResult,
    bootstrap_km_median,
)


def test_bootstrap_is_deterministic_and_preserves_full_sample_median() -> None:
    kwargs = {
        "stop_times": [2, 3, 5, 8, 8],
        "event_observed": [1, 1, 1, 0, 0],
        "confidence": 0.8,
        "n_resamples": 200,
        "seed": 1729,
    }

    first = bootstrap_km_median(**kwargs)
    second = bootstrap_km_median(**kwargs)

    assert first == second
    assert first.central_median == 5
    assert first.n_resamples == 200
    assert first.seed == 1729
    assert first.n_resolved / first.n_resamples == first.resolved_fraction


def test_finite_inverted_cdf_bounds_remain_integer_layers() -> None:
    result = bootstrap_km_median(
        stop_times=[1, 2, 3, 4, 5],
        event_observed=[1, 1, 1, 1, 1],
        confidence=0.8,
        n_resamples=101,
        seed=4,
    )

    assert result.central_median == 3
    assert result.lower_bound == 2
    assert result.upper_bound == 4
    assert type(result.lower_bound) is int
    assert type(result.upper_bound) is int
    assert result.interval_resolved
    assert result.resolved_fraction == 1.0


def test_unresolved_replicates_produce_an_unresolved_upper_bound() -> None:
    result = bootstrap_km_median(
        stop_times=[64, 64],
        event_observed=[1, 0],
        confidence=0.95,
        n_resamples=256,
        seed=11,
    )

    assert result.central_median == 64
    assert result.lower_bound == 64
    assert result.upper_bound is None
    assert 0.0 < result.resolved_fraction < 1.0
    assert not result.interval_resolved


def test_all_censored_sample_retains_unresolved_central_median_and_bounds() -> None:
    result = bootstrap_km_median(
        stop_times=[64, 64, 64],
        event_observed=[0, 0, 0],
        n_resamples=32,
        seed=5,
    )

    assert result.central_median is None
    assert result.lower_bound is None
    assert result.upper_bound is None
    assert result.n_resolved == 0
    assert result.resolved_fraction == 0.0


def test_result_is_immutable() -> None:
    result = bootstrap_km_median([1, 1], [1, 1], n_resamples=2)

    with pytest.raises(FrozenInstanceError):
        result.central_median = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("option", "value", "error", "match"),
    [
        ("confidence", True, TypeError, "real number"),
        ("confidence", "0.95", TypeError, "real number"),
        ("confidence", np.nan, ValueError, "strictly between"),
        ("confidence", 0.0, ValueError, "strictly between"),
        ("confidence", 1.0, ValueError, "strictly between"),
        ("n_resamples", True, TypeError, "must be an integer"),
        ("n_resamples", 1.5, TypeError, "must be an integer"),
        ("n_resamples", 0, ValueError, "must be positive"),
        ("seed", False, TypeError, "must be an integer"),
        ("seed", 1.0, TypeError, "must be an integer"),
        ("seed", -1, ValueError, "must be nonnegative"),
    ],
)
def test_bootstrap_rejects_invalid_options(
    option: str,
    value: object,
    error: type[Exception],
    match: str,
) -> None:
    kwargs: dict[str, object] = {"n_resamples": 2}
    kwargs[option] = value
    with pytest.raises(error, match=match):
        bootstrap_km_median([1, 2], [1, 0], **kwargs)


@pytest.mark.parametrize(
    ("stop_times", "events", "error", "match"),
    [
        ([], [], ValueError, "at least one"),
        ([1.0], [1], TypeError, "contain integers"),
        ([0], [1], ValueError, "positive"),
        ([1, 2], [1], ValueError, "same length"),
        ([1], [2], ValueError, "only zero or one"),
    ],
)
def test_bootstrap_reuses_strict_survival_input_validation(
    stop_times: object,
    events: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        bootstrap_km_median(stop_times, events, n_resamples=2)


def test_result_type_is_explicit() -> None:
    result = bootstrap_km_median([1], [1], n_resamples=np.int64(1), seed=np.int64(3))

    assert isinstance(result, BootstrapMedianResult)
    assert result.central_median == 1
    assert result.lower_bound == 1
    assert result.upper_bound == 1
