"""Profiled three-parameter scaling of single-reference purification times.

The input is the validated row representation of one beta value from
``point_summary.csv``.  Only rows with a resolved Kaplan-Meier median and a
finite two-sided bootstrap interval enter the fit.  In particular, an
unresolved median is never replaced by the depth cap.

For every trial ``(p_c, nu, z)``, this module profiles the nuisance master
curve in

``log(tau_50) - z*log(n) = log(F((p-p_c)*n**(1/nu)))``

as a weighted cubic regression spline.  The spline coefficients are obtained
by deterministic penalized least squares, and only the three physical
parameters require numerical optimization.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral, Real
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

type ParameterName = Literal["pc", "nu", "z"]
type ParameterPair = tuple[ParameterName, ParameterName]

_PARAMETERS: tuple[ParameterName, ...] = ("pc", "nu", "z")
_INVALID_LOSS = 1.0e100


def _minimize(*args: Any, **kwargs: Any) -> Any:
    """Call scipy's multivariate optimizer without importing scipy at startup."""
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - scipy is present in the research env
        raise RuntimeError("purification-time scaling requires scipy") from exc
    return minimize(*args, **kwargs)


def _minimize_scalar(*args: Any, **kwargs: Any) -> Any:
    """Call scipy's bounded scalar optimizer without importing scipy at startup."""
    try:
        from scipy.optimize import minimize_scalar
    except ImportError as exc:  # pragma: no cover - scipy is present in the research env
        raise RuntimeError("purification-time scaling requires scipy") from exc
    return minimize_scalar(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class CollapseBounds:
    """Closed parameter bounds for the three-parameter collapse."""

    pc: tuple[float, float] = (0.01, 0.60)
    nu: tuple[float, float] = (0.40, 6.00)
    z: tuple[float, float] = (0.00, 3.00)

    def __post_init__(self) -> None:
        _validate_bound(self.pc, "pc", positive_lower=False)
        _validate_bound(self.nu, "nu", positive_lower=True)
        _validate_bound(self.z, "z", positive_lower=False)
        if self.pc[0] < 0.0 or self.pc[1] > 1.0:
            raise ValueError(f"pc bounds must lie in [0, 1]; got {self.pc!r}")
        if self.z[0] < 0.0:
            raise ValueError(f"z lower bound must be nonnegative; got {self.z[0]!r}")

    def interval(self, name: ParameterName) -> tuple[float, float]:
        """Return the bound associated with one parameter name."""
        return getattr(self, name)

    def scipy_bounds(self) -> tuple[tuple[float, float], ...]:
        """Return bounds in canonical ``(pc, nu, z)`` order."""
        return (self.pc, self.nu, self.z)


@dataclass(frozen=True, slots=True)
class PointSelectionDiagnostics:
    """Accounting for point-summary rows entering a collapse."""

    total_records: int
    unresolved_medians: int
    unresolved_bootstrap_intervals: int
    usable_points: int
    sigma_floored_points: int
    bootstrap_resamples: int
    bootstrap_confidence: float


@dataclass(frozen=True, slots=True)
class CollapseData:
    """Strictly selected and transformed data for one beta value."""

    beta: float
    beta_key: int
    point_index: NDArray[np.int64]
    n: NDArray[np.float64]
    p: NDArray[np.float64]
    tau: NDArray[np.float64]
    log_tau: NDArray[np.float64]
    log_sigma: NDArray[np.float64]
    selection: PointSelectionDiagnostics

    @property
    def sizes(self) -> tuple[int, ...]:
        """Return the system sizes represented in the selected points."""
        return tuple(int(value) for value in np.unique(self.n))

    @property
    def common_p_window(self) -> tuple[float, float]:
        """Return the overlap of the selected p ranges across sizes."""
        lows = [float(np.min(self.p[self.n == size])) for size in np.unique(self.n)]
        highs = [float(np.max(self.p[self.n == size])) for size in np.unique(self.n)]
        return max(lows), min(highs)


@dataclass(frozen=True, slots=True)
class ProfiledMasterCurve:
    """Best-fit cubic regression spline for one physical parameter vector."""

    x_center: float
    x_scale: float
    knots: tuple[float, ...]
    coefficients: tuple[float, ...]

    def predict_log_scaled(self, x: ArrayLike) -> NDArray[np.float64]:
        """Evaluate ``log(F(x))`` at one or more scaling coordinates."""
        values = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("master-curve coordinates must be finite")
        normalized = (values - self.x_center) / self.x_scale
        basis = _spline_basis(normalized, np.asarray(self.knots, dtype=np.float64))
        return np.asarray(basis @ np.asarray(self.coefficients), dtype=np.float64)

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the positive master curve ``F(x)``."""
        with np.errstate(over="ignore"):
            result = np.exp(self.predict_log_scaled(x))
        return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class OptimizerAttempt:
    """Audit record for one deterministic multistart optimization attempt."""

    start: tuple[float, float, float]
    optimizer_success: bool
    accepted: bool
    parameters: tuple[float, float, float]
    reported_loss: float
    message: str


@dataclass(frozen=True, slots=True)
class CollapseFitDiagnostics:
    """Goodness-of-fit and optimizer diagnostics for one collapse."""

    success: bool
    message: str
    n_points: int
    n_sizes: int
    n_starts: int
    n_valid_starts: int
    best_start_index: int | None
    objective: float
    weighted_rmse: float
    chi_square: float
    effective_master_parameters: float
    effective_degrees_of_freedom: float
    reduced_chi_square: float
    spline_roughness: float
    condition_number: float
    pc_inside_common_window: bool
    boundary_parameters: tuple[ParameterName, ...]
    attempts: tuple[OptimizerAttempt, ...]


@dataclass(frozen=True, slots=True)
class CollapseFit:
    """Three-parameter collapse result and pointwise residual diagnostics."""

    beta: float
    pc: float
    nu: float
    z: float
    loss: float
    master_curve: ProfiledMasterCurve | None
    scaling_coordinate: NDArray[np.float64]
    fitted_log_tau: NDArray[np.float64]
    residual: NDArray[np.float64]
    standardized_residual: NDArray[np.float64]
    diagnostics: CollapseFitDiagnostics

    @property
    def success(self) -> bool:
        """Whether at least one valid bounded optimizer result was found."""
        return self.diagnostics.success

    @property
    def parameters(self) -> tuple[float, float, float]:
        """Return ``(p_c, nu, z)`` in canonical order."""
        return self.pc, self.nu, self.z


@dataclass(frozen=True, slots=True)
class ProfileLossLandscape:
    """A two-dimensional loss surface profiled over its omitted parameter."""

    x_parameter: ParameterName
    y_parameter: ParameterName
    optimized_parameter: ParameterName
    x_values: NDArray[np.float64]
    y_values: NDArray[np.float64]
    loss: NDArray[np.float64]
    delta_loss: NDArray[np.float64]
    optimized_values: NDArray[np.float64]
    valid: NDArray[np.bool_]
    optimizer_calls: int


@dataclass(frozen=True, slots=True)
class _ProfileResult:
    loss: float
    data_loss: float
    chi_square: float
    roughness: float
    condition_number: float
    effective_parameters: float
    x: NDArray[np.float64]
    fitted_log_tau: NDArray[np.float64]
    residual: NDArray[np.float64]
    standardized_residual: NDArray[np.float64]
    master_curve: ProfiledMasterCurve


def _validate_bound(
    bound: tuple[float, float],
    name: str,
    *,
    positive_lower: bool,
) -> None:
    if not isinstance(bound, tuple) or len(bound) != 2:
        raise TypeError(f"{name} bounds must be a two-element tuple; got {bound!r}")
    lower, upper = bound
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in bound):
        raise TypeError(f"{name} bounds must contain real numbers; got {bound!r}")
    lower, upper = float(lower), float(upper)
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError(f"{name} bounds must be finite and increasing; got {bound!r}")
    if positive_lower and lower <= 0.0:
        raise ValueError(f"{name} lower bound must be positive; got {lower!r}")


def _parse_integer(raw: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(raw, str):
        if raw != raw.strip() or not raw.isascii() or not raw.isdecimal():
            raise ValueError(f"{name} must be an unsigned integer; got {raw!r}")
        value = int(raw)
    elif isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Integral):
        raise ValueError(f"{name} must be an integer; got {raw!r}")
    else:
        value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")
    return value


def _parse_optional_integer(raw: object, name: str) -> int | None:
    if raw is None or raw == "":
        return None
    return _parse_integer(raw, name, minimum=1)


def _parse_float(
    raw: object,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(raw, str):
        if raw == "" or raw != raw.strip():
            raise ValueError(f"{name} must be a finite real number; got {raw!r}")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a finite real number; got {raw!r}") from exc
    elif isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Real):
        raise ValueError(f"{name} must be a finite real number; got {raw!r}")
    else:
        value = float(raw)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]; got {raw!r}")
    return value


def _parse_optional_float(
    raw: object,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    if raw is None or raw == "":
        return None
    return _parse_float(raw, name, minimum=minimum, maximum=maximum)


def _parse_flag(raw: object, name: str) -> bool:
    if isinstance(raw, (bool, np.bool_)):
        return bool(raw)
    if isinstance(raw, Integral) and not isinstance(raw, bool) and int(raw) in {0, 1}:
        return bool(raw)
    if raw in {"0", "1"}:
        return raw == "1"
    raise ValueError(f"{name} must be zero or one; got {raw!r}")


def _row_value(row: Mapping[str, object], field: str, row_number: int) -> object:
    if field not in row:
        raise ValueError(f"row {row_number}: missing {field}")
    return row[field]


def _readonly(values: ArrayLike, dtype: np.dtype[Any]) -> NDArray[Any]:
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def prepare_collapse_data(
    records: Iterable[Mapping[str, object]],
    *,
    expected_beta: Real | None = None,
    log_sigma_floor: Real = 0.005,
    minimum_sizes: int = 3,
    minimum_points_per_size: int = 4,
) -> CollapseData:
    """Validate and select one beta's rows from ``point_summary.csv``.

    A usable row has a resolved Kaplan-Meier median and a resolved two-sided
    bootstrap interval.  Rows with an unresolved median or cap-limited upper
    interval remain accounted for in ``selection`` but do not enter the fit.
    No value is synthesized for either case.
    """
    rows = list(records)
    if not rows:
        raise ValueError("at least one point-summary record is required")
    if isinstance(log_sigma_floor, (bool, np.bool_)) or not isinstance(log_sigma_floor, Real):
        raise TypeError(f"log_sigma_floor must be a real number; got {log_sigma_floor!r}")
    log_sigma_floor = float(log_sigma_floor)
    if not math.isfinite(log_sigma_floor) or log_sigma_floor <= 0.0:
        raise ValueError(f"log_sigma_floor must be finite and positive; got {log_sigma_floor!r}")
    minimum_sizes = _parse_integer(minimum_sizes, "minimum_sizes", minimum=3)
    minimum_points_per_size = _parse_integer(
        minimum_points_per_size,
        "minimum_points_per_size",
        minimum=2,
    )
    if expected_beta is not None:
        expected_beta_value = _parse_float(
            expected_beta,
            "expected_beta",
            minimum=0.0,
            maximum=1.0,
        )
        expected_beta_key = round(expected_beta_value * 1_000_000_000)
    else:
        expected_beta_key = None

    parsed_keys: set[tuple[int, int]] = set()
    point_indices: set[int] = set()
    beta_values: dict[int, float] = {}
    bootstrap_options: set[tuple[int, float | None]] = set()
    selected: list[tuple[int, int, float, float, float]] = []
    unresolved_medians = 0
    unresolved_intervals = 0
    sigma_floored = 0

    for row_number, row in enumerate(rows, start=2):
        if not isinstance(row, Mapping):
            raise TypeError(f"row {row_number} must be a mapping; got {type(row).__name__}")
        prefix = f"row {row_number}"
        point_index = _parse_integer(
            _row_value(row, "point_index", row_number),
            f"{prefix}.point_index",
        )
        if point_index in point_indices:
            raise ValueError(f"duplicate point_index {point_index}")
        point_indices.add(point_index)

        n = _parse_integer(_row_value(row, "n", row_number), f"{prefix}.n", minimum=1)
        if n % 2:
            raise ValueError(f"{prefix}.n must be even; got {n}")
        beta = _parse_float(
            _row_value(row, "beta", row_number),
            f"{prefix}.beta",
            minimum=0.0,
            maximum=1.0,
        )
        beta_key = _parse_integer(
            _row_value(row, "beta_key", row_number),
            f"{prefix}.beta_key",
        )
        if beta_key != round(beta * 1_000_000_000):
            raise ValueError(f"{prefix}: beta and beta_key are inconsistent")
        if expected_beta_key is not None and beta_key != expected_beta_key:
            raise ValueError(
                f"{prefix}: beta={beta!r} does not match expected_beta={expected_beta_value!r}"
            )
        previous_beta = beta_values.setdefault(beta_key, beta)
        if previous_beta != beta:
            raise ValueError(f"beta_key={beta_key} maps to inconsistent beta values")

        p = _parse_float(
            _row_value(row, "p", row_number),
            f"{prefix}.p",
            minimum=0.0,
            maximum=1.0,
        )
        p_key = _parse_integer(
            _row_value(row, "p_key", row_number),
            f"{prefix}.p_key",
        )
        if p_key != round(p * 1_000_000):
            raise ValueError(f"{prefix}: p and p_key are inconsistent")
        canonical_key = (n, p_key)
        if canonical_key in parsed_keys:
            raise ValueError(f"duplicate (n, p) point {canonical_key}")
        parsed_keys.add(canonical_key)

        t_max = _parse_integer(
            _row_value(row, "t_max", row_number),
            f"{prefix}.t_max",
            minimum=1,
        )
        if t_max != 8 * n:
            raise ValueError(f"{prefix}.t_max={t_max}, expected {8 * n}")
        median_resolved = _parse_flag(
            _row_value(row, "median_resolved", row_number),
            f"{prefix}.median_resolved",
        )
        median = _parse_optional_integer(
            _row_value(row, "median_tau_p", row_number),
            f"{prefix}.median_tau_p",
        )
        if median_resolved != (median is not None):
            raise ValueError(f"{prefix}: median_resolved disagrees with median_tau_p")
        if median is not None and median > t_max:
            raise ValueError(f"{prefix}: median_tau_p exceeds t_max")

        interval_resolved = _parse_flag(
            _row_value(row, "median_ci_resolved", row_number),
            f"{prefix}.median_ci_resolved",
        )
        lower = _parse_optional_integer(
            _row_value(row, "median_ci_lower", row_number),
            f"{prefix}.median_ci_lower",
        )
        upper = _parse_optional_integer(
            _row_value(row, "median_ci_upper", row_number),
            f"{prefix}.median_ci_upper",
        )
        if upper is not None and lower is None:
            raise ValueError(f"{prefix}: a finite upper interval bound requires a lower bound")
        if interval_resolved != (lower is not None and upper is not None):
            raise ValueError(f"{prefix}: median_ci_resolved disagrees with interval bounds")
        if interval_resolved and not median_resolved:
            raise ValueError(f"{prefix}: a resolved interval requires a resolved median")
        if lower is not None and lower > t_max:
            raise ValueError(f"{prefix}: median_ci_lower exceeds t_max")
        if upper is not None and upper > t_max:
            raise ValueError(f"{prefix}: median_ci_upper exceeds t_max")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"{prefix}: bootstrap interval bounds are reversed")

        n_resamples = _parse_integer(
            _row_value(row, "bootstrap_resamples", row_number),
            f"{prefix}.bootstrap_resamples",
        )
        confidence = _parse_optional_float(
            _row_value(row, "bootstrap_confidence", row_number),
            f"{prefix}.bootstrap_confidence",
            minimum=0.0,
            maximum=1.0,
        )
        if confidence is not None and not 0.0 < confidence < 1.0:
            raise ValueError(f"{prefix}: bootstrap_confidence must lie strictly between 0 and 1")
        if n_resamples == 0:
            if (
                confidence is not None
                or lower is not None
                or upper is not None
                or interval_resolved
            ):
                raise ValueError(f"{prefix}: disabled bootstrap fields must be blank or zero")
        elif confidence is None:
            raise ValueError(f"{prefix}: enabled bootstrap_confidence must not be blank")
        bootstrap_options.add((n_resamples, confidence))

        if not median_resolved:
            unresolved_medians += 1
            continue
        if not interval_resolved:
            unresolved_intervals += 1
            continue
        if median is None or lower is None or upper is None or confidence is None:
            raise RuntimeError("internal resolved-row accounting failed")
        normal_quantile = NormalDist().inv_cdf(0.5 + confidence / 2.0)
        raw_log_sigma = (math.log(upper) - math.log(lower)) / (2.0 * normal_quantile)
        if not math.isfinite(raw_log_sigma) or raw_log_sigma < 0.0:
            raise ValueError(f"{prefix}: bootstrap interval gives an invalid log uncertainty")
        log_sigma = max(raw_log_sigma, log_sigma_floor)
        sigma_floored += int(raw_log_sigma < log_sigma_floor)
        selected.append((point_index, n, p, float(median), log_sigma))

    if len(beta_values) != 1:
        raise ValueError(
            f"collapse records must contain exactly one beta; got {sorted(beta_values.values())}"
        )
    if len(bootstrap_options) != 1:
        raise ValueError("collapse records mix different bootstrap configurations")
    n_resamples, confidence = next(iter(bootstrap_options))
    if n_resamples == 0 or confidence is None:
        raise ValueError("collapse fitting requires enabled bootstrap uncertainty")
    if not selected:
        raise ValueError("no resolved medians with finite two-sided bootstrap intervals")

    selected.sort(key=lambda item: (item[1], item[2], item[0]))
    indices = _readonly([item[0] for item in selected], np.dtype(np.int64))
    sizes = _readonly([item[1] for item in selected], np.dtype(np.float64))
    ps = _readonly([item[2] for item in selected], np.dtype(np.float64))
    tau = _readonly([item[3] for item in selected], np.dtype(np.float64))
    log_sigma = _readonly([item[4] for item in selected], np.dtype(np.float64))
    log_tau = _readonly(np.log(tau), np.dtype(np.float64))

    unique_sizes, counts = np.unique(sizes, return_counts=True)
    if len(unique_sizes) < minimum_sizes:
        raise ValueError(
            f"collapse needs at least {minimum_sizes} sizes with usable points; "
            f"got {len(unique_sizes)}"
        )
    short_sizes = {
        int(size): int(count)
        for size, count in zip(unique_sizes, counts, strict=True)
        if count < minimum_points_per_size
    }
    if short_sizes:
        raise ValueError(
            f"each size needs at least {minimum_points_per_size} usable points; got {short_sizes}"
        )
    p_lows = [float(np.min(ps[sizes == size])) for size in unique_sizes]
    p_highs = [float(np.max(ps[sizes == size])) for size in unique_sizes]
    if max(p_lows) >= min(p_highs):
        raise ValueError("usable p ranges do not overlap across sizes")

    beta_key, beta = next(iter(beta_values.items()))
    return CollapseData(
        beta=beta,
        beta_key=beta_key,
        point_index=indices,
        n=sizes,
        p=ps,
        tau=tau,
        log_tau=log_tau,
        log_sigma=log_sigma,
        selection=PointSelectionDiagnostics(
            total_records=len(rows),
            unresolved_medians=unresolved_medians,
            unresolved_bootstrap_intervals=unresolved_intervals,
            usable_points=len(selected),
            sigma_floored_points=sigma_floored,
            bootstrap_resamples=n_resamples,
            bootstrap_confidence=confidence,
        ),
    )


def _positive_integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    parsed = _parse_integer(value, name, minimum=minimum)
    if parsed > maximum:
        raise ValueError(f"{name} must be at most {maximum}; got {parsed}")
    return parsed


def _positive_real(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number; got {value!r}")
    result = float(value)
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not lower_ok:
        relation = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}; got {value!r}")
    return result


def _validate_data(data: CollapseData) -> None:
    if not isinstance(data, CollapseData):
        raise TypeError(f"data must be CollapseData; got {type(data).__name__}")
    arrays = (data.point_index, data.n, data.p, data.tau, data.log_tau, data.log_sigma)
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("CollapseData arrays must be nonempty and aligned")
    if any(np.asarray(array).ndim != 1 for array in arrays):
        raise ValueError("CollapseData arrays must be one-dimensional")
    numeric = (data.n, data.p, data.tau, data.log_tau, data.log_sigma)
    if any(not np.all(np.isfinite(array)) for array in numeric):
        raise ValueError("CollapseData arrays must be finite")
    if np.any(data.n <= 0.0) or np.any(data.tau <= 0.0) or np.any(data.log_sigma <= 0.0):
        raise ValueError("CollapseData sizes, times, and uncertainties must be positive")
    if np.any((data.p < 0.0) | (data.p > 1.0)):
        raise ValueError("CollapseData p values must lie in [0, 1]")
    if not np.allclose(data.log_tau, np.log(data.tau), rtol=0.0, atol=1e-12):
        raise ValueError("CollapseData log_tau is inconsistent with tau")
    keys = [(int(n), float(p)) for n, p in zip(data.n, data.p, strict=True)]
    if len(keys) != len(set(keys)):
        raise ValueError("CollapseData contains duplicate (n, p) points")
    if len(np.unique(data.n)) < 3:
        raise ValueError("CollapseData must contain at least three sizes")


def _spline_knots(interior_knots: int) -> NDArray[np.float64]:
    if interior_knots == 0:
        return np.empty(0, dtype=np.float64)
    return np.linspace(-0.75, 0.75, interior_knots + 2, dtype=np.float64)[1:-1]


def _spline_basis(
    normalized_x: ArrayLike,
    knots: NDArray[np.float64],
) -> NDArray[np.float64]:
    u = np.asarray(normalized_x, dtype=np.float64)
    columns = [np.ones_like(u), u, u * u, u * u * u]
    columns.extend(np.maximum(u - knot, 0.0) ** 3 for knot in knots)
    return np.column_stack(columns)


def _spline_second_derivative_basis(
    normalized_x: NDArray[np.float64],
    knots: NDArray[np.float64],
) -> NDArray[np.float64]:
    u = np.asarray(normalized_x, dtype=np.float64)
    columns = [np.zeros_like(u), np.zeros_like(u), np.full_like(u, 2.0), 6.0 * u]
    columns.extend(6.0 * np.maximum(u - knot, 0.0) for knot in knots)
    return np.column_stack(columns)


@lru_cache(maxsize=16)
def _roughness_matrix(interior_knots: int) -> NDArray[np.float64]:
    knots = _spline_knots(interior_knots)
    nodes, weights = np.polynomial.legendre.leggauss(48)
    second = _spline_second_derivative_basis(nodes, knots)
    result = second.T @ (weights[:, None] * second)
    result.setflags(write=False)
    return result


def _parameter_array(parameters: Mapping[ParameterName, float]) -> NDArray[np.float64]:
    return np.asarray([parameters[name] for name in _PARAMETERS], dtype=np.float64)


def _parameters_in_bounds(values: NDArray[np.float64], bounds: CollapseBounds) -> bool:
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        return False
    return all(
        lower <= float(value) <= upper
        for value, (lower, upper) in zip(values, bounds.scipy_bounds(), strict=True)
    )


def _profile_master_curve(
    data: CollapseData,
    parameters: NDArray[np.float64],
    *,
    bounds: CollapseBounds,
    interior_knots: int,
    smoothing: float,
) -> _ProfileResult | None:
    if not _parameters_in_bounds(parameters, bounds):
        return None
    pc, nu, z = (float(value) for value in parameters)
    log_n = np.log(data.n)
    with np.errstate(over="ignore", invalid="ignore"):
        size_scale = np.exp(log_n / nu)
        x = (data.p - pc) * size_scale
    if not np.all(np.isfinite(x)):
        return None
    x_min, x_max = float(np.min(x)), float(np.max(x))
    x_scale = (x_max - x_min) / 2.0
    if not math.isfinite(x_scale) or x_scale <= 1e-12:
        return None
    x_center = (x_max + x_min) / 2.0
    normalized_x = (x - x_center) / x_scale
    knots = _spline_knots(interior_knots)
    basis = _spline_basis(normalized_x, knots)
    if len(data.n) <= basis.shape[1] + 3:
        return None

    scaled_log_tau = data.log_tau - z * log_n
    relative_weights = 1.0 / np.square(data.log_sigma)
    relative_weights /= float(np.mean(relative_weights))
    normal_matrix = basis.T @ (relative_weights[:, None] * basis)
    roughness_matrix = _roughness_matrix(interior_knots)
    trace_scale = float(np.trace(normal_matrix)) / basis.shape[1]
    ridge = max(trace_scale, 1.0) * 1e-12
    penalized_matrix = normal_matrix + smoothing * roughness_matrix
    penalized_matrix = penalized_matrix + ridge * np.eye(basis.shape[1])
    condition_number = float(np.linalg.cond(penalized_matrix))
    if not math.isfinite(condition_number) or condition_number > 1e14:
        return None
    try:
        coefficients = np.linalg.solve(
            penalized_matrix,
            basis.T @ (relative_weights * scaled_log_tau),
        )
        effective_parameters = float(np.trace(np.linalg.solve(penalized_matrix, normal_matrix)))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coefficients)) or not math.isfinite(effective_parameters):
        return None

    fitted_scaled = basis @ coefficients
    fitted_log_tau = fitted_scaled + z * log_n
    residual = data.log_tau - fitted_log_tau
    standardized = residual / data.log_sigma
    weighted_sum = float(np.sum(relative_weights * np.square(residual)))
    roughness = float(coefficients @ roughness_matrix @ coefficients)
    data_loss = weighted_sum / len(residual)
    loss = (weighted_sum + smoothing * roughness) / len(residual)
    chi_square = float(np.sum(np.square(standardized)))
    diagnostics = (loss, data_loss, chi_square, roughness)
    if not all(math.isfinite(value) and value >= 0.0 for value in diagnostics):
        return None

    master = ProfiledMasterCurve(
        x_center=x_center,
        x_scale=x_scale,
        knots=tuple(float(value) for value in knots),
        coefficients=tuple(float(value) for value in coefficients),
    )
    return _ProfileResult(
        loss=loss,
        data_loss=data_loss,
        chi_square=chi_square,
        roughness=roughness,
        condition_number=condition_number,
        effective_parameters=effective_parameters,
        x=np.asarray(x, dtype=np.float64),
        fitted_log_tau=np.asarray(fitted_log_tau, dtype=np.float64),
        residual=np.asarray(residual, dtype=np.float64),
        standardized_residual=np.asarray(standardized, dtype=np.float64),
        master_curve=master,
    )


def _halton(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        result += digit * fraction
        fraction /= base
    return result


def _value_from_fraction(name: ParameterName, fraction: float, bounds: CollapseBounds) -> float:
    lower, upper = bounds.interval(name)
    if name == "nu":
        return math.exp(math.log(lower) + fraction * (math.log(upper) - math.log(lower)))
    return lower + fraction * (upper - lower)


def _deterministic_starts(
    data: CollapseData,
    bounds: CollapseBounds,
    n_starts: int,
) -> tuple[tuple[float, float, float], ...]:
    pc_lower, pc_upper = bounds.pc
    pc_center = min(max(float(np.median(data.p)), pc_lower), pc_upper)
    nu_center = math.sqrt(bounds.nu[0] * bounds.nu[1])
    z_center = min(max(1.0, bounds.z[0]), bounds.z[1])
    starts: list[tuple[float, float, float]] = [(pc_center, nu_center, z_center)]
    index = 1
    while len(starts) < n_starts:
        candidate = (
            _value_from_fraction("pc", _halton(index, 2), bounds),
            _value_from_fraction("nu", _halton(index, 3), bounds),
            _value_from_fraction("z", _halton(index, 5), bounds),
        )
        if not any(np.allclose(candidate, previous, rtol=0.0, atol=1e-14) for previous in starts):
            starts.append(candidate)
        index += 1
    return tuple(starts)


def _result_message(result: object) -> str:
    message = getattr(result, "message", "")
    return str(message).replace("\n", " ")


def _optimizer_candidate(
    result: object,
    data: CollapseData,
    *,
    bounds: CollapseBounds,
    interior_knots: int,
    smoothing: float,
) -> tuple[NDArray[np.float64] | None, _ProfileResult | None, float]:
    try:
        reported_loss = float(result.fun)
        candidate = np.asarray(result.x, dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        return None, None, math.nan
    if not bool(getattr(result, "success", False)):
        return candidate, None, reported_loss
    if not math.isfinite(reported_loss) or not _parameters_in_bounds(candidate, bounds):
        return candidate, None, reported_loss
    profile = _profile_master_curve(
        data,
        candidate,
        bounds=bounds,
        interior_knots=interior_knots,
        smoothing=smoothing,
    )
    if profile is None or not math.isclose(
        reported_loss,
        profile.loss,
        rel_tol=1e-5,
        abs_tol=1e-10,
    ):
        return candidate, None, reported_loss
    return candidate, profile, reported_loss


def _failure_fit(
    data: CollapseData,
    attempts: tuple[OptimizerAttempt, ...],
    message: str,
) -> CollapseFit:
    nan_array = _readonly(np.full(len(data.n), np.nan), np.dtype(np.float64))
    diagnostics = CollapseFitDiagnostics(
        success=False,
        message=message,
        n_points=len(data.n),
        n_sizes=len(data.sizes),
        n_starts=len(attempts),
        n_valid_starts=0,
        best_start_index=None,
        objective=math.nan,
        weighted_rmse=math.nan,
        chi_square=math.nan,
        effective_master_parameters=math.nan,
        effective_degrees_of_freedom=math.nan,
        reduced_chi_square=math.nan,
        spline_roughness=math.nan,
        condition_number=math.nan,
        pc_inside_common_window=False,
        boundary_parameters=(),
        attempts=attempts,
    )
    return CollapseFit(
        beta=data.beta,
        pc=math.nan,
        nu=math.nan,
        z=math.nan,
        loss=math.nan,
        master_curve=None,
        scaling_coordinate=nan_array,
        fitted_log_tau=nan_array,
        residual=nan_array,
        standardized_residual=nan_array,
        diagnostics=diagnostics,
    )


def fit_three_parameter_collapse(
    data: CollapseData,
    *,
    bounds: CollapseBounds = CollapseBounds(),
    interior_knots: int = 3,
    smoothing: Real = 0.02,
    n_starts: int = 12,
    maxiter: int = 500,
) -> CollapseFit:
    """Fit ``p_c``, ``nu``, and ``z`` with a profiled smooth master curve.

    Starts are a deterministic low-discrepancy sequence.  Every optimizer
    result is independently checked for convergence, finite in-bound
    parameters, finite loss, and agreement with a fresh profile evaluation.
    If no result passes those checks, all reported parameters are ``nan``.
    """
    _validate_data(data)
    if not isinstance(bounds, CollapseBounds):
        raise TypeError(f"bounds must be CollapseBounds; got {type(bounds).__name__}")
    interior_knots = _positive_integer(
        interior_knots,
        "interior_knots",
        minimum=0,
        maximum=8,
    )
    smoothing = _positive_real(smoothing, "smoothing")
    n_starts = _positive_integer(n_starts, "n_starts", minimum=1, maximum=64)
    maxiter = _positive_integer(maxiter, "maxiter", minimum=10, maximum=10_000)
    if len(data.n) <= 7 + interior_knots:
        raise ValueError(
            "collapse has too few usable points for the spline and three physical parameters"
        )

    def objective(values: ArrayLike) -> float:
        candidate = np.asarray(values, dtype=np.float64)
        profile = _profile_master_curve(
            data,
            candidate,
            bounds=bounds,
            interior_knots=interior_knots,
            smoothing=smoothing,
        )
        return _INVALID_LOSS if profile is None else profile.loss

    starts = _deterministic_starts(data, bounds, n_starts)
    attempts: list[OptimizerAttempt] = []
    valid: list[tuple[int, NDArray[np.float64], _ProfileResult]] = []
    for index, start in enumerate(starts):
        result = _minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds.scipy_bounds(),
            options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": maxiter, "maxls": 40},
        )
        candidate, profile, reported_loss = _optimizer_candidate(
            result,
            data,
            bounds=bounds,
            interior_knots=interior_knots,
            smoothing=smoothing,
        )
        accepted = profile is not None and candidate is not None
        attempt_parameters = (
            tuple(float(value) for value in candidate)
            if candidate is not None and candidate.shape == (3,)
            else (math.nan, math.nan, math.nan)
        )
        attempts.append(
            OptimizerAttempt(
                start=start,
                optimizer_success=bool(getattr(result, "success", False)),
                accepted=accepted,
                parameters=attempt_parameters,
                reported_loss=reported_loss,
                message=_result_message(result),
            )
        )
        if accepted:
            valid.append((index, candidate, profile))

    attempt_tuple = tuple(attempts)
    if not valid:
        return _failure_fit(data, attempt_tuple, "no optimizer result passed validation")
    best_start_index, best_parameters, best_profile = min(valid, key=lambda item: item[2].loss)
    pc, nu, z = (float(value) for value in best_parameters)
    effective_dof = len(data.n) - best_profile.effective_parameters - 3.0
    reduced_chi_square = (
        best_profile.chi_square / effective_dof if effective_dof > 0.0 else math.nan
    )
    boundary_parameters = tuple(
        name
        for name, value in zip(_PARAMETERS, best_parameters, strict=True)
        if min(
            value - bounds.interval(name)[0],
            bounds.interval(name)[1] - value,
        )
        <= 1e-4 * (bounds.interval(name)[1] - bounds.interval(name)[0])
    )
    common_lower, common_upper = data.common_p_window
    diagnostics = CollapseFitDiagnostics(
        success=True,
        message="bounded profiled fit converged",
        n_points=len(data.n),
        n_sizes=len(data.sizes),
        n_starts=len(starts),
        n_valid_starts=len(valid),
        best_start_index=best_start_index,
        objective=best_profile.loss,
        weighted_rmse=math.sqrt(best_profile.data_loss),
        chi_square=best_profile.chi_square,
        effective_master_parameters=best_profile.effective_parameters,
        effective_degrees_of_freedom=effective_dof,
        reduced_chi_square=reduced_chi_square,
        spline_roughness=best_profile.roughness,
        condition_number=best_profile.condition_number,
        pc_inside_common_window=common_lower < pc < common_upper,
        boundary_parameters=boundary_parameters,
        attempts=attempt_tuple,
    )
    return CollapseFit(
        beta=data.beta,
        pc=pc,
        nu=nu,
        z=z,
        loss=best_profile.loss,
        master_curve=best_profile.master_curve,
        scaling_coordinate=_readonly(best_profile.x, np.dtype(np.float64)),
        fitted_log_tau=_readonly(best_profile.fitted_log_tau, np.dtype(np.float64)),
        residual=_readonly(best_profile.residual, np.dtype(np.float64)),
        standardized_residual=_readonly(
            best_profile.standardized_residual,
            np.dtype(np.float64),
        ),
        diagnostics=diagnostics,
    )


def _grid_values(
    values: ArrayLike,
    name: ParameterName,
    bounds: CollapseBounds,
) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) < 2:
        raise ValueError(f"{name} landscape grid must be one-dimensional with at least two values")
    if len(result) > 200:
        raise ValueError(f"{name} landscape grid is limited to 200 values; got {len(result)}")
    if not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{name} landscape grid must be finite and strictly increasing")
    lower, upper = bounds.interval(name)
    if float(result[0]) < lower or float(result[-1]) > upper:
        raise ValueError(f"{name} landscape grid must lie within {bounds.interval(name)!r}")
    result.setflags(write=False)
    return result


def _scalar_profile_candidate(
    result: object,
    fixed: Mapping[ParameterName, float],
    hidden: ParameterName,
    data: CollapseData,
    *,
    bounds: CollapseBounds,
    interior_knots: int,
    smoothing: float,
) -> tuple[float, _ProfileResult] | None:
    try:
        value = float(result.x)
        reported_loss = float(result.fun)
    except (AttributeError, TypeError, ValueError):
        return None
    lower, upper = bounds.interval(hidden)
    if (
        not bool(getattr(result, "success", False))
        or not math.isfinite(value)
        or not lower <= value <= upper
        or not math.isfinite(reported_loss)
    ):
        return None
    parameters = dict(fixed)
    parameters[hidden] = value
    profile = _profile_master_curve(
        data,
        _parameter_array(parameters),
        bounds=bounds,
        interior_knots=interior_knots,
        smoothing=smoothing,
    )
    if profile is None or not math.isclose(
        reported_loss,
        profile.loss,
        rel_tol=1e-5,
        abs_tol=1e-10,
    ):
        return None
    return value, profile


def profile_loss_landscape(
    data: CollapseData,
    *,
    pair: ParameterPair,
    x_values: ArrayLike,
    y_values: ArrayLike,
    bounds: CollapseBounds = CollapseBounds(),
    interior_knots: int = 3,
    smoothing: Real = 0.02,
    profile_intervals: int = 3,
    maxiter: int = 100,
    max_cells: int = 10_000,
) -> ProfileLossLandscape:
    """Profile a pairwise loss surface over the omitted physical parameter.

    At every grid cell, the hidden parameter is independently minimized over
    several deterministic bounded intervals.  The spline master curve is
    solved again at every scalar objective evaluation.  Therefore these are
    profiled surfaces, not fixed-parameter slices.
    """
    _validate_data(data)
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise TypeError(f"pair must be a two-parameter tuple; got {pair!r}")
    if pair[0] not in _PARAMETERS or pair[1] not in _PARAMETERS or pair[0] == pair[1]:
        raise ValueError(f"pair must contain two distinct names from {_PARAMETERS!r}; got {pair!r}")
    if not isinstance(bounds, CollapseBounds):
        raise TypeError(f"bounds must be CollapseBounds; got {type(bounds).__name__}")
    interior_knots = _positive_integer(
        interior_knots,
        "interior_knots",
        minimum=0,
        maximum=8,
    )
    smoothing = _positive_real(smoothing, "smoothing")
    profile_intervals = _positive_integer(
        profile_intervals,
        "profile_intervals",
        minimum=1,
        maximum=12,
    )
    maxiter = _positive_integer(maxiter, "maxiter", minimum=10, maximum=10_000)
    max_cells = _positive_integer(max_cells, "max_cells", minimum=4, maximum=40_000)
    x_grid = _grid_values(x_values, pair[0], bounds)
    y_grid = _grid_values(y_values, pair[1], bounds)
    if len(x_grid) * len(y_grid) > max_cells:
        raise ValueError(
            f"landscape has {len(x_grid) * len(y_grid)} cells, exceeding max_cells={max_cells}"
        )
    hidden = next(name for name in _PARAMETERS if name not in pair)
    hidden_lower, hidden_upper = bounds.interval(hidden)
    interval_edges = np.linspace(hidden_lower, hidden_upper, profile_intervals + 1)
    losses = np.full((len(y_grid), len(x_grid)), np.nan, dtype=np.float64)
    optimized = np.full_like(losses, np.nan)
    valid = np.zeros_like(losses, dtype=np.bool_)
    optimizer_calls = 0

    for y_index, y_value in enumerate(y_grid):
        for x_index, x_value in enumerate(x_grid):
            fixed: dict[ParameterName, float] = {
                pair[0]: float(x_value),
                pair[1]: float(y_value),
            }

            def objective(
                hidden_value: float,
                fixed_parameters: Mapping[ParameterName, float] = fixed,
            ) -> float:
                parameters = dict(fixed_parameters)
                parameters[hidden] = float(hidden_value)
                profile = _profile_master_curve(
                    data,
                    _parameter_array(parameters),
                    bounds=bounds,
                    interior_knots=interior_knots,
                    smoothing=smoothing,
                )
                return _INVALID_LOSS if profile is None else profile.loss

            candidates: list[tuple[float, _ProfileResult]] = []
            for lower, upper in zip(interval_edges[:-1], interval_edges[1:], strict=True):
                optimizer_calls += 1
                result = _minimize_scalar(
                    objective,
                    bounds=(float(lower), float(upper)),
                    method="bounded",
                    options={"xatol": 1e-6, "maxiter": maxiter},
                )
                candidate = _scalar_profile_candidate(
                    result,
                    fixed,
                    hidden,
                    data,
                    bounds=bounds,
                    interior_knots=interior_knots,
                    smoothing=smoothing,
                )
                if candidate is not None:
                    candidates.append(candidate)
            if not candidates:
                continue
            for endpoint in (hidden_lower, hidden_upper):
                parameters = dict(fixed)
                parameters[hidden] = endpoint
                profile = _profile_master_curve(
                    data,
                    _parameter_array(parameters),
                    bounds=bounds,
                    interior_knots=interior_knots,
                    smoothing=smoothing,
                )
                if profile is not None:
                    candidates.append((endpoint, profile))
            hidden_value, best_profile = min(candidates, key=lambda item: item[1].loss)
            optimized[y_index, x_index] = hidden_value
            losses[y_index, x_index] = best_profile.loss
            valid[y_index, x_index] = True

    finite = losses[valid]
    delta = np.full_like(losses, np.nan)
    if finite.size:
        delta[valid] = losses[valid] - float(np.min(finite))
    for array in (losses, optimized, valid, delta):
        array.setflags(write=False)
    return ProfileLossLandscape(
        x_parameter=pair[0],
        y_parameter=pair[1],
        optimized_parameter=hidden,
        x_values=x_grid,
        y_values=y_grid,
        loss=losses,
        delta_loss=delta,
        optimized_values=optimized,
        valid=valid,
        optimizer_calls=optimizer_calls,
    )


def profile_pairwise_landscapes(
    data: CollapseData,
    *,
    pc_values: ArrayLike,
    nu_values: ArrayLike,
    z_values: ArrayLike,
    bounds: CollapseBounds = CollapseBounds(),
    interior_knots: int = 3,
    smoothing: Real = 0.02,
    profile_intervals: int = 3,
    maxiter: int = 100,
    max_cells: int = 10_000,
) -> dict[ParameterPair, ProfileLossLandscape]:
    """Return all three pairwise surfaces with the third parameter profiled."""
    grids: dict[ParameterName, ArrayLike] = {
        "pc": pc_values,
        "nu": nu_values,
        "z": z_values,
    }
    pairs: tuple[ParameterPair, ...] = (("pc", "nu"), ("pc", "z"), ("nu", "z"))
    return {
        pair: profile_loss_landscape(
            data,
            pair=pair,
            x_values=grids[pair[0]],
            y_values=grids[pair[1]],
            bounds=bounds,
            interior_knots=interior_knots,
            smoothing=smoothing,
            profile_intervals=profile_intervals,
            maxiter=maxiter,
            max_cells=max_cells,
        )
        for pair in pairs
    }


__all__ = [
    "CollapseBounds",
    "CollapseData",
    "CollapseFit",
    "CollapseFitDiagnostics",
    "OptimizerAttempt",
    "PointSelectionDiagnostics",
    "ProfileLossLandscape",
    "ProfiledMasterCurve",
    "fit_three_parameter_collapse",
    "prepare_collapse_data",
    "profile_loss_landscape",
    "profile_pairwise_landscapes",
]
