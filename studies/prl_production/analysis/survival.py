"""Kaplan-Meier analysis for exact single-reference purification times."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class KaplanMeierCurve:
    """A right-continuous Kaplan-Meier survival curve.

    Each tuple has one entry per distinct observed stop layer.  ``n_at_risk``
    is evaluated immediately before events and censoring at that layer.  The
    standard error is Greenwood's estimate for the survival probability.
    ``median`` is ``None`` when survival never reaches one half within the
    observation window.
    """

    times: tuple[int, ...]
    n_at_risk: tuple[int, ...]
    n_events: tuple[int, ...]
    n_censored: tuple[int, ...]
    survival: tuple[float, ...]
    greenwood_se: tuple[float, ...]
    median: int | None

    @property
    def sample_size(self) -> int:
        """Return the number of trajectories entering the first risk set."""
        return self.n_at_risk[0]

    @property
    def event_count(self) -> int:
        """Return the total number of observed purification events."""
        return sum(self.n_events)

    @property
    def censored_count(self) -> int:
        """Return the total number of right-censored trajectories."""
        return sum(self.n_censored)

    def survival_at(self, layer: int) -> float:
        """Evaluate the right-continuous survival estimate at an integer layer."""
        layer = _validate_nonnegative_integer(layer, "layer")
        index = bisect_right(self.times, layer) - 1
        return 1.0 if index < 0 else self.survival[index]

    def greenwood_se_at(self, layer: int) -> float:
        """Evaluate Greenwood's survival standard error at an integer layer."""
        layer = _validate_nonnegative_integer(layer, "layer")
        index = bisect_right(self.times, layer) - 1
        return 0.0 if index < 0 else self.greenwood_se[index]


@dataclass(frozen=True, slots=True)
class PurificationPointSummary:
    """Scalar survival summary for one complete ``(n, beta, p)`` point."""

    n: int
    beta: float
    p: float
    t_max: int
    n_trajectories: int
    n_events: int
    n_censored: int
    event_fraction: float
    median_tau_p: int | None
    survival_at_cap: float

    @property
    def median_resolved(self) -> bool:
        """Whether the Kaplan-Meier median was reached by ``t_max``."""
        return self.median_tau_p is not None


@dataclass(frozen=True, slots=True)
class PointSurvivalAnalysis:
    """Kaplan-Meier curve and scalar summary for one production point."""

    summary: PurificationPointSummary
    curve: KaplanMeierCurve


def _validate_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative; got {value}")
    return value


def _positive_integer(value: int, name: str) -> int:
    value = _validate_nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _integer_vector(values: ArrayLike, name: str) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if not array.size:
        return np.asarray(array, dtype=np.int64)
    if array.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain integers; got dtype {array.dtype}")
    if array.dtype.kind == "u" and int(array.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} contains a value too large for int64")
    return np.asarray(array, dtype=np.int64)


def _event_vector(values: ArrayLike, expected_length: int) -> NDArray[np.bool_]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"event_observed must be one-dimensional; got shape {array.shape}")
    if len(array) != expected_length:
        raise ValueError(
            "stop times and event_observed must have the same length; "
            f"got {expected_length} and {len(array)}"
        )
    if not array.size:
        return np.asarray(array, dtype=np.bool_)
    if array.dtype.kind == "b":
        return np.asarray(array, dtype=np.bool_)
    if array.dtype.kind not in {"i", "u"}:
        raise TypeError(
            f"event_observed must contain booleans or binary integers; got dtype {array.dtype}"
        )
    if np.any((array != 0) & (array != 1)):
        raise ValueError("event_observed must contain only zero or one")
    return np.asarray(array, dtype=np.bool_)


def _probability(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number; got {value!r}")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]; got {value!r}")
    return value


def kaplan_meier(
    stop_times: ArrayLike,
    event_observed: ArrayLike,
) -> KaplanMeierCurve:
    """Estimate survival from integer stop layers and right-censoring flags.

    Events and censoring tied at a layer share the same pre-layer risk set, as
    required by the Kaplan-Meier product-limit estimator.  This is essential
    at the production cap, where an exact purification event at ``8n`` must be
    distinguished from censoring at the same layer.
    """
    durations = _integer_vector(stop_times, "stop_times")
    events = _event_vector(event_observed, len(durations))
    if not len(durations):
        raise ValueError("at least one trajectory is required")
    if np.any(durations < 1):
        raise ValueError("stop_times must be positive integer layers")

    times, inverse = np.unique(durations, return_inverse=True)
    event_counts = np.bincount(inverse[events], minlength=len(times)).astype(np.int64)
    censor_counts = np.bincount(inverse[~events], minlength=len(times)).astype(np.int64)

    risk = len(durations)
    survival_probability = 1.0
    greenwood_sum = 0.0
    risk_counts: list[int] = []
    survival_values: list[float] = []
    standard_errors: list[float] = []
    median: int | None = None

    for time, n_events, n_censored in zip(
        times,
        event_counts,
        censor_counts,
        strict=True,
    ):
        n_events = int(n_events)
        n_censored = int(n_censored)
        risk_counts.append(risk)
        if n_events:
            if n_events == risk:
                survival_probability = 0.0
            else:
                survival_probability *= 1.0 - n_events / risk
                greenwood_sum += n_events / (risk * (risk - n_events))
        standard_error = (
            0.0 if survival_probability == 0.0 else survival_probability * math.sqrt(greenwood_sum)
        )
        survival_values.append(survival_probability)
        standard_errors.append(standard_error)
        if median is None and survival_probability <= 0.5:
            median = int(time)
        risk -= n_events + n_censored

    if risk != 0:
        raise RuntimeError(f"internal risk-set accounting failed; {risk} trajectories remain")

    return KaplanMeierCurve(
        times=tuple(int(value) for value in times),
        n_at_risk=tuple(risk_counts),
        n_events=tuple(int(value) for value in event_counts),
        n_censored=tuple(int(value) for value in censor_counts),
        survival=tuple(survival_values),
        greenwood_se=tuple(standard_errors),
        median=median,
    )


def summarize_purification_point(
    *,
    n: int,
    beta: Real,
    p: Real,
    t_max: int,
    tau_p: ArrayLike,
    stop_layer: ArrayLike,
    event_observed: ArrayLike,
) -> PointSurvivalAnalysis:
    """Validate and summarize one complete single-reference production point.

    The production encoding is strict: an observed event has
    ``tau_p == stop_layer`` in ``[1, t_max]``; a censored trajectory has
    ``tau_p == -1`` and ``stop_layer == t_max``.  Incomplete rows therefore
    fail validation instead of being mistaken for censoring.
    """
    n = _positive_integer(n, "n")
    t_max = _positive_integer(t_max, "t_max")
    if t_max != 8 * n:
        raise ValueError(f"t_max must equal 8*n={8 * n}; got {t_max}")
    beta = _probability(beta, "beta")
    p = _probability(p, "p")

    tau = _integer_vector(tau_p, "tau_p")
    stops = _integer_vector(stop_layer, "stop_layer")
    if len(stops) != len(tau):
        raise ValueError(
            f"tau_p and stop_layer must have the same length; got {len(tau)} and {len(stops)}"
        )
    events = _event_vector(event_observed, len(tau))
    if not len(tau):
        raise ValueError("at least one completed trajectory is required")
    if np.any((stops < 1) | (stops > t_max)):
        raise ValueError(f"stop_layer must lie in [1, {t_max}]")

    observed = events
    censored = ~events
    if np.any(tau[observed] != stops[observed]) or np.any(tau[observed] < 1):
        raise ValueError("observed rows require tau_p == stop_layer >= 1")
    if np.any(tau[censored] != -1) or np.any(stops[censored] != t_max):
        raise ValueError("censored rows require tau_p == -1 and stop_layer == t_max")

    curve = kaplan_meier(stops, events)
    n_events = int(events.sum())
    n_trajectories = len(events)
    summary = PurificationPointSummary(
        n=n,
        beta=beta,
        p=p,
        t_max=t_max,
        n_trajectories=n_trajectories,
        n_events=n_events,
        n_censored=n_trajectories - n_events,
        event_fraction=n_events / n_trajectories,
        median_tau_p=curve.median,
        survival_at_cap=curve.survival[-1],
    )
    return PointSurvivalAnalysis(summary=summary, curve=curve)


__all__ = [
    "KaplanMeierCurve",
    "PointSurvivalAnalysis",
    "PurificationPointSummary",
    "kaplan_meier",
    "summarize_purification_point",
]
