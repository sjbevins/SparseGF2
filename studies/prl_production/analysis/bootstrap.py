"""Pointwise graph-bootstrap uncertainty for purification-time medians."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from .survival import kaplan_meier


@dataclass(frozen=True, slots=True)
class BootstrapMedianResult:
    """Percentile interval for one point's Kaplan-Meier median.

    ``central_median`` is always estimated from the complete input sample, not
    from the bootstrap replicates.  A bound is ``None`` when its empirical
    quantile is unresolved because the corresponding bootstrap survival curve
    does not reach one half within the observation window.
    """

    central_median: int | None
    lower_bound: int | None
    upper_bound: int | None
    confidence: float
    n_resamples: int
    n_resolved: int
    resolved_fraction: float
    seed: int

    @property
    def interval_resolved(self) -> bool:
        """Whether both percentile bounds are finite observed layers."""
        return self.lower_bound is not None and self.upper_bound is not None


def _confidence_level(value: Real) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"confidence must be a real number; got {value!r}")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly between zero and one; got {value!r}")
    return confidence


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive; got {result}")
    return result


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative; got {result}")
    return result


def _inverted_cdf_quantile(
    medians: list[int | None],
    probability: float,
) -> int | None:
    """Return an integer empirical quantile, ordering ``None`` as infinity."""
    finite = sorted(median for median in medians if median is not None)
    order_index = math.ceil(probability * len(medians)) - 1
    if order_index >= len(finite):
        return None
    return finite[order_index]


def bootstrap_km_median(
    stop_times: ArrayLike,
    event_observed: ArrayLike,
    *,
    confidence: Real = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
) -> BootstrapMedianResult:
    """Bootstrap one point's Kaplan-Meier median over independent graphs.

    Each input row must represent one independently sampled graph trajectory.
    Rows are resampled with replacement, preserving the original sample size.
    This pointwise interval is intended for raw-curve error bars; it does not
    implement the shared-index, paired-across-``p`` bootstrap used by the final
    finite-size-scaling analysis.

    Unresolved replicate medians are ordered as positive infinity for the
    percentile calculation.  Therefore a depth-cap-limited quantile is
    returned as ``None`` rather than being imputed at the depth cap.
    """
    confidence = _confidence_level(confidence)
    n_resamples = _positive_integer(n_resamples, "n_resamples")
    seed = _nonnegative_integer(seed, "seed")

    central_curve = kaplan_meier(stop_times, event_observed)
    durations = np.asarray(stop_times)
    events = np.asarray(event_observed, dtype=np.bool_)
    sample_size = len(durations)

    rng = np.random.default_rng(seed)
    bootstrap_medians: list[int | None] = []
    for _ in range(n_resamples):
        indices = rng.integers(0, sample_size, size=sample_size)
        bootstrap_medians.append(kaplan_meier(durations[indices], events[indices]).median)

    n_resolved = sum(median is not None for median in bootstrap_medians)
    tail_probability = (1.0 - confidence) / 2.0
    return BootstrapMedianResult(
        central_median=central_curve.median,
        lower_bound=_inverted_cdf_quantile(bootstrap_medians, tail_probability),
        upper_bound=_inverted_cdf_quantile(bootstrap_medians, 1.0 - tail_probability),
        confidence=confidence,
        n_resamples=n_resamples,
        n_resolved=n_resolved,
        resolved_fraction=n_resolved / n_resamples,
        seed=seed,
    )


__all__ = ["BootstrapMedianResult", "bootstrap_km_median"]
