"""The observable catalog of the Watts–Strogatz study (specification v1.4 §5.2–§5.8; design §5).

Every observable the study records, or computes later from stored snapshots, is a pure function
here, with a registry entry (:class:`ObservableDef`) that names it, versions it and states its
definition. ``docs/study/04-observables.md`` has the definitions, the proof of every claim the
code relies on, the algorithms and worked examples; this docstring fixes the conventions.

**The objects** (v1.4 §5). After ``t`` monitored layers the joint system–reference state is pure;
its physical stabilizer group has rank ``r_t`` and the code holds ``k_t = n - r_t`` logical
qubits. For a set ``A`` of system qubits, ``I_t(A:R)`` (bits) is the logical information an
erasure of ``A`` destroys, and ``f_t(A) = 1[I_t(A:R) > 0]`` says whether the erasure fails (is
not correctable). ``StabilizerCode.lost_information`` computes ``I_t(A:R)`` from the physical
code and ``PhaseFreeTableau.mutual_information`` from the joint state.

**What is here.**

* The ``k`` trace (v1.4 §5.2, §5.3): :func:`k_change_points` (lossless) and its inverse
  :func:`k_from_change_points`; :func:`first_passage`, the exact
  ``tau_ell = min{t : k_t <= ell}`` for ``ell = 0..n`` with right-censoring at the horizon, and
  the checkpoint upper bounds ``tau_chk``; :func:`code_parameters` (``k``, ``k / n``, ``n - k``).
* Distances: :func:`cyclic_distance` (``d_cyc``, v1.4 §5.5, from the engine's
  ``contiguous_distance`` probe: raw, ``has_code``, zero-extended),
  :func:`cyclic_distance_from_code` (the same from a stored code) and
  :func:`connected_distance` (``d_conn``, v1.4 §5.6, by exact enumeration).
* Uniform erasures (v1.4 §5.7 at ``J = 0``): :func:`uniform_failure` (``P_C(m)``) and
  :func:`integrated_robustness` (``D(0) = sum_{m<n} (1 - P_C(m)) = E_pi[M_pi]``).
* Geometry-correlated erasures (v1.4 §5.7): :func:`correlated_failure` (``P_{C,G,J}(m)`` and
  ``D(J)``), :func:`mean_leakage` and :func:`mean_leakage_exact` (``l(m, J)``), and
  :func:`region_seed` (the seed of a retrospective region sampler).
* The joint state: :func:`half_cut_entropy`.
* Peaks (v1.4 §5.8): :func:`peak`, :func:`peak_sensitivity`, :func:`bootstrap_peak`.
* The registry: :class:`ObservableDef`, :func:`register`, :func:`get`, :func:`catalog`,
  :func:`resolve_params`, :func:`params_key` and, for the observables computed later from a
  stored snapshot, :func:`compute` with :class:`SnapshotInput` and :class:`ObservableValue`.

**Versions.** An entry's ``version`` is part of every stored value's key (with the id and the
canonical parameters, :func:`params_key`), so a changed implementation gets a new version and its
values never mix with the old ones. A retrospective sampler's seed contains the version too
(:func:`region_seed`).

>>> fp = first_passage([3, 3, 1, 1, 0, 0])
>>> fp.tau.tolist(), fp.observed.tolist()
([4, 2, 2, 0], [True, True, True, True])
>>> k_change_points([3, 3, 1, 1, 0, 0]).tolist()
[[0, 3], [2, 1], [4, 0], [5, 0]]
"""

from __future__ import annotations

import hashlib
import itertools
import math
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

import numpy as np

from sparsegf2 import analysis, erasures, graphs
from sparsegf2.codes import StabilizerCode
from sparsegf2.study.identity import TrajectoryKey, canonical_json, token_words
from sparsegf2.tableau import PhaseFreeTableau

__all__ = [
    "INPUTS",
    "LEVELS",
    "PEAK_RULES",
    "REGION_SEED_TAG",
    "REGION_SEED_WORDS",
    "REQUIRED",
    "ChannelFailure",
    "CorrelatedProfile",
    "CyclicDistance",
    "FailureProfile",
    "FirstPassage",
    "Leakage",
    "ObservableDef",
    "ObservableValue",
    "Peak",
    "PeakBootstrap",
    "Robustness",
    "SnapshotInput",
    "bootstrap_peak",
    "catalog",
    "catalog_markdown",
    "code_parameters",
    "compute",
    "connected_distance",
    "correlated_channel_failure",
    "correlated_failure",
    "cyclic_distance",
    "cyclic_distance_from_code",
    "first_passage",
    "get",
    "half_cut_entropy",
    "integrated_robustness",
    "k_change_points",
    "k_from_change_points",
    "mean_leakage",
    "mean_leakage_exact",
    "params_key",
    "peak",
    "peak_sensitivity",
    "region_seed",
    "register",
    "resolve_params",
    "uniform_channel_failure",
    "uniform_failure",
]


def _read_only(obj: Any) -> None:
    for f in fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, np.ndarray):
            value.flags.writeable = False


# ---------------------------------------------------------------------------------------------
# input checks (every message says what to change)
# ---------------------------------------------------------------------------------------------


def _int(value: Any, what: str, *, minimum: int | None = None) -> int:
    """``value`` as a Python int: ints and numpy integers only (never bools, floats or strings)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    v = int(value)
    if minimum is not None and v < minimum:
        raise ValueError(f"{what} must be at least {minimum}, got {v}")
    return v


def _index(idx: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(int(i)) for i in idx) + "]"


def _numbers(values: Any, what: str) -> np.ndarray:
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be numbers") from None


def _whole(values: Any, what: str) -> np.ndarray:
    """``values`` as int64; NaN, infinities, fractions and negative numbers are refused (the first is named)."""
    arr = _numbers(values, what)
    bad = ~np.isfinite(arr) | (arr < 0)
    bad |= np.where(np.isfinite(arr), arr != np.round(arr), False)
    if bad.any():
        idx = tuple(int(i) for i in np.argwhere(bad)[0])
        raise ValueError(f"{what}{_index(idx)} = {arr[idx]!r} is not a non-negative whole number")
    return arr.astype(np.int64)


def _trace(k: Any, what: str = "k") -> np.ndarray:
    arr = _numbers(k, what)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(
            f"{what} must be a 1-D trace k_0, ..., k_T with one entry per layer t = 0..T "
            f"(code_dimension at every layer), got shape {arr.shape}"
        )
    return _whole(arr, what)


def _depths(values: Any, what: str) -> np.ndarray:
    """Strictly increasing non-negative whole depths (a 1-D array, at least one entry)."""
    arr = _numbers(values, what)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{what} must be a 1-D list of depths, got shape {arr.shape}")
    out = _whole(arr, what)
    if np.any(np.diff(out) <= 0):
        raise ValueError(f"{what} must increase strictly (each depth once, in order)")
    return out


# ---------------------------------------------------------------------------------------------
# the k trace and first passage (v1.4 §5.2, §5.3)
# ---------------------------------------------------------------------------------------------


def code_parameters(k: Any, num_system: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(k_t, k_t / n, r_t = n - k_t)``: the logical qubits, the code rate and the stabilizer rank (v1.4 §5.2).

    ``k`` holds whole numbers in ``[0, n]`` (any shape); the three arrays have its shape. The
    rate ``R_t = k_t / n`` is the code rate, not the reference register.
    """
    n = _int(num_system, "num_system", minimum=1)
    kk = _whole(k, "k")
    if np.any(kk > n):
        idx = tuple(int(i) for i in np.argwhere(kk > n)[0])
        raise ValueError(f"k{_index(idx)} = {kk[idx]} exceeds n = {n}")
    return kk, kk / n, n - kk


def k_change_points(k: Any) -> np.ndarray:
    """The trace ``k_0, ..., k_T`` as its change points, losslessly (v1.4 §5.2, §5.3).

    ``k[t]`` is the number of logical qubits after ``t`` completed layers, for every
    ``t = 0..T`` (the engine's ``code_dimension`` with ``checkpoints="every_layer"``). It must
    be nonincreasing (Pauli measurements without feedback never raise ``k_t``; proof in
    ``docs/study/04-observables.md``, Proposition 1); an increase is refused with the layer
    named, since it means the input is not such a trace.

    Returns an int64 array of shape ``(J, 2)`` with rows ``(t_j, k_{t_j})``: the layer ``0``,
    every layer ``t >= 1`` with ``k_t < k_{t-1}`` in increasing order, and the horizon ``T`` as a
    last row when it is not already one of them. So ``k_t = k_{t_j}`` for
    ``t_j <= t < t_{j+1}``, the last row's layer is always ``T``, and
    :func:`k_from_change_points` recovers the trace exactly. The form is canonical: one trace
    has one array. With ``stop_when_purified`` the engine fills the layers after the stop with
    ``k = 0``, the exact value (``k`` is nonincreasing and nonnegative, so it stays 0).

    >>> k_change_points([4, 4, 4]).tolist()
    [[0, 4], [2, 4]]
    >>> k_change_points([4, 2, 2, 1]).tolist()
    [[0, 4], [1, 2], [3, 1]]
    """
    kk = _trace(k)
    step = np.diff(kk)
    up = np.flatnonzero(step > 0)
    if up.size:
        t = int(up[0]) + 1
        raise ValueError(
            f"k must be nonincreasing in t (v1.4 §5.2: Pauli measurements without feedback never "
            f"raise k_t), but k[{t}] = {kk[t]} > k[{t - 1}] = {kk[t - 1]}: layer {t} gained a "
            "logical qubit, so this is not a trace of one trajectory at every layer"
        )
    horizon = kk.size - 1
    layers = np.concatenate([np.zeros(1, dtype=np.int64), np.flatnonzero(step < 0) + 1])
    if layers[-1] != horizon:
        layers = np.append(layers, horizon)
    return np.stack([layers, kk[layers]], axis=1).astype(np.int64)


def k_from_change_points(points: Any) -> np.ndarray:
    """The trace ``k_0, ..., k_T`` from :func:`k_change_points`: its exact inverse.

    ``points`` is ``(J, 2)``: layers ``t_0 = 0 < t_1 < ... < t_{J-1} = T`` and values that
    decrease strictly at every row except possibly the last, which may repeat the previous value
    (the horizon marker). Anything else (a first layer other than 0, unsorted layers, an
    increase, a repeated value that is not the last row) is refused: it is not the canonical form
    of any trace. Returns int64 of length ``T + 1``.

    >>> k_from_change_points([[0, 4], [1, 2], [3, 1]]).tolist()
    [4, 2, 2, 1]
    """
    raw = _numbers(points, "points")
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] == 0:
        raise ValueError(
            f"points must be a (J, 2) array of rows (t_j, k_j) from k_change_points, got shape {raw.shape}"
        )
    pts = _whole(raw, "points")
    t, kv = pts[:, 0], pts[:, 1]
    if t[0] != 0:
        raise ValueError(f"the first change point must be at layer 0, got layer {t[0]}")
    if np.any(np.diff(t) <= 0):
        raise ValueError("the layers of the change points must increase strictly")
    dk = np.diff(kv)
    if np.any(dk > 0):
        j = int(np.flatnonzero(dk > 0)[0]) + 1
        raise ValueError(
            f"k must be nonincreasing: row {j} has k = {kv[j]} after k = {kv[j - 1]} (v1.4 §5.2)"
        )
    if dk.size > 1 and np.any(dk[:-1] == 0):
        j = int(np.flatnonzero(dk[:-1] == 0)[0]) + 1
        raise ValueError(
            f"row {j} repeats k = {kv[j]}: only the last row (the horizon) may repeat the value "
            "before it; every other row must be a change"
        )
    horizon = int(t[-1])
    out = np.empty(horizon + 1, dtype=np.int64)
    out[:horizon] = np.repeat(kv[:-1], np.diff(t))
    out[horizon] = kv[-1]
    return out


@dataclass(frozen=True, eq=False)
class FirstPassage:
    """Exact first-passage depths of one trajectory (v1.4 §5.2), with right-censoring.

    * ``tau``: int64 ``(n + 1,)``, ``tau[ell] = min{t : k_t <= ell}`` where ``observed[ell]``;
      ``-1`` where the threshold was not reached by the horizon: a marker for a censored
      threshold, never a time (do not average it; use :meth:`survival_data`).
    * ``observed``: bool ``(n + 1,)``, ``k_T <= ell`` (the event happened within the horizon).
    * ``horizon``: ``T``, the last layer of the trace and the censoring depth of every
      unobserved threshold (v1.4's ``censor_depth``; also :attr:`censor_depth`).
    * ``checkpoints``, ``tau_chk``, ``observed_chk``: with checkpoints, the checkpoint upper
      bound ``tau_chk[ell] = min{t in checkpoints : k_t <= ell}`` (``-1`` where no checkpoint
      reached the threshold) and whether it was observed; ``tau <= tau_chk`` wherever both are
      observed (Proposition 3 of the document). Primary analyses use ``tau`` (v1.4 §5.2).
    * ``num_system``: ``n``.
    """

    num_system: int
    horizon: int
    tau: np.ndarray
    observed: np.ndarray
    checkpoints: np.ndarray | None = None
    tau_chk: np.ndarray | None = None
    observed_chk: np.ndarray | None = None

    def __post_init__(self) -> None:
        _read_only(self)

    @property
    def censor_depth(self) -> int:
        """The depth at which every unobserved threshold is right-censored: the horizon ``T``."""
        return self.horizon

    def survival_data(self, ell: int) -> tuple[int, bool]:
        """``(time, observed)`` of threshold ``ell`` for survival analysis.

        ``time`` is ``tau_ell`` when observed and the censoring depth ``T`` otherwise (the event
        happens after ``T``; :func:`sparsegf2.study.stats.weighted_kaplan_meier` takes such
        pairs).
        """
        e = _int(ell, "ell", minimum=0)
        if e > self.num_system:
            raise ValueError(f"ell must be in [0, {self.num_system}], got {e}")
        if self.observed[e]:
            return int(self.tau[e]), True
        return self.horizon, False

    def to_dict(self) -> dict[str, Any]:
        """Plain Python: censored entries are ``None`` (not ``-1``)."""
        out: dict[str, Any] = {
            "num_system": self.num_system,
            "horizon": self.horizon,
            "censor_depth": self.horizon,
            "tau": [int(v) if o else None for v, o in zip(self.tau, self.observed, strict=True)],
            "observed": self.observed.tolist(),
        }
        if self.checkpoints is not None:
            assert self.tau_chk is not None and self.observed_chk is not None
            out["checkpoints"] = self.checkpoints.tolist()
            out["tau_chk"] = [
                int(v) if o else None for v, o in zip(self.tau_chk, self.observed_chk, strict=True)
            ]
            out["observed_chk"] = self.observed_chk.tolist()
        return out


def first_passage(
    k: Any, checkpoints: Any = None, *, num_system: int | None = None
) -> FirstPassage:
    """The exact first-passage depths ``tau_ell = min{t >= 0 : k_t <= ell}``, ``ell = 0..n`` (v1.4 §5.2).

    ``k`` is the trace ``k_0, ..., k_T`` at every layer (1-D) or its change points (2-D, from
    :func:`k_change_points`); ``n`` is ``num_system`` (default ``k_0``, which is ``n`` in the
    purification picture; ``k_0 <= n`` is required).

    **Exactness.** ``k_t`` is nonincreasing (refused otherwise), so ``{t : k_t <= ell}`` is a
    final segment ``[tau_ell, T]`` of the observed layers, and the layer ``t`` where ``k`` drops
    from ``k_{t-1}`` to ``k_t`` is the first passage of exactly the thresholds
    ``ell = k_t, ..., k_{t-1} - 1`` (several at once when ``k`` drops by more than one). Every
    layer is observed, so this is ``tau_ell`` exactly, whatever the checkpoint schedule:
    the ``n + 1`` crossings and the trace determine each other (``k_t = min{ell : tau_ell <= t}``).

    **Censoring.** A threshold with ``k_T > ell`` was not reached by the horizon ``T``: it is
    right-censored at ``T`` (``observed`` False, ``tau`` ``-1``), never given a value.
    ``tau_n = 0``.

    **Checkpoint bounds.** With ``checkpoints`` (strictly increasing layers in ``[0, T]``, the
    sparse schedule ``T(n, beta)``), ``tau_chk[ell] = min{t in checkpoints : k_t <= ell}``, which
    is the first checkpoint at or after ``tau_ell``; hence ``tau_ell <= tau_chk[ell]`` whenever
    both are observed. Cost ``O(T + n)``.

    >>> fp = first_passage([5, 5, 2, 2, 2, 1], checkpoints=[0, 4, 5])
    >>> fp.tau.tolist(), fp.tau_chk.tolist(), fp.horizon
    ([-1, 5, 2, 2, 2, 0], [-1, 5, 4, 4, 4, 0], 5)
    """
    arr = _numbers(k, "k")
    kk = k_from_change_points(arr) if arr.ndim == 2 else _trace(arr)
    points = k_change_points(kk)  # validates the monotonicity
    n = int(kk[0]) if num_system is None else _int(num_system, "num_system", minimum=0)
    if kk[0] > n:
        raise ValueError(f"k_0 = {kk[0]} exceeds num_system = {n} (k_t <= n always)")
    horizon = int(kk.size - 1)
    tau = np.full(n + 1, -1, dtype=np.int64)
    tau[int(kk[0]) :] = 0
    for (t_j, k_j), k_before in zip(points[1:].tolist(), points[:-1, 1].tolist(), strict=True):
        if k_j < k_before:  # thresholds k_j .. k_before - 1 are first met at layer t_j
            tau[k_j:k_before] = t_j
    observed = tau >= 0
    cps = tau_chk = observed_chk = None
    if checkpoints is not None:
        cps = _depths(checkpoints, "checkpoints")
        if cps[-1] > horizon:
            raise ValueError(
                f"checkpoint {cps[-1]} lies beyond the horizon T = {horizon} of the trace"
            )
        pos = np.searchsorted(cps, tau, side="left")  # the first checkpoint >= tau_ell
        observed_chk = observed & (pos < cps.size)
        tau_chk = np.where(observed_chk, cps[np.minimum(pos, cps.size - 1)], -1).astype(np.int64)
    return FirstPassage(
        num_system=n,
        horizon=horizon,
        tau=tau,
        observed=observed,
        checkpoints=cps,
        tau_chk=tau_chk,
        observed_chk=observed_chk,
    )


# ---------------------------------------------------------------------------------------------
# distances (v1.4 §5.5, §5.6)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CyclicDistance:
    """``d_cyc`` in the three forms of v1.4 §5.5 and §9 ("Native contiguous distance").

    * ``raw``: float64, the engine's value, NaN where ``k_t = 0`` (undefined without a code);
    * ``has_code``: bool, ``k_t > 0``;
    * ``zero_extended``: int64, ``raw`` where ``k_t > 0`` and 0 where ``k_t = 0`` (the
      analysis-only zero extension, for unconditional peak traces).
    """

    raw: np.ndarray
    has_code: np.ndarray
    zero_extended: np.ndarray

    def __post_init__(self) -> None:
        _read_only(self)


def cyclic_distance(raw: Any, k: Any, *, num_system: int | None = None) -> CyclicDistance:
    """``d_cyc`` from the engine's ``contiguous_distance`` probe: raw, ``has_code`` and zero-extended (v1.4 §5.5).

    For ``k_t > 0``,
    ``d_cyc(t) = min{m in 1..n : some interval A_{a,m} = {a, a+1, ..., a+m-1 mod n} has f_t = 1}``,
    the shortest failing interval of the original ring order (identity order, periodic); the
    minimum exists because ``I_t(V:R) = 2 k_t > 0``. The engine records NaN where ``k_t = 0``.
    ``raw`` and ``k`` (the code dimension at the same evaluations) have one shape; they are
    checked against each other (a whole number in ``[1, n]`` where ``k > 0``, NaN where
    ``k = 0``), so data from mismatched evaluations is refused.
    """
    r = _numbers(raw, "raw")
    kk = _whole(k, "k")
    if r.shape != kk.shape:
        raise ValueError(f"raw {r.shape} and k {kk.shape} must have the same shape")
    has = kk > 0
    nan = np.isnan(r)
    finite = np.isfinite(r)
    bad = has & (~finite | (r < 1) | np.where(finite, r != np.round(r), False))
    if num_system is not None:
        bad |= has & finite & (r > _int(num_system, "num_system", minimum=1))
    if bad.any():
        idx = tuple(int(i) for i in np.argwhere(bad)[0])
        raise ValueError(
            f"raw{_index(idx)} = {r[idx]!r} where k = {kk[idx]}: with a code d_cyc is a whole number "
            "in [1, n]"
        )
    wrong = ~has & ~nan
    if wrong.any():
        idx = tuple(int(i) for i in np.argwhere(wrong)[0])
        raise ValueError(
            f"raw{_index(idx)} = {r[idx]!r} where k = 0: the engine records NaN without a code; "
            "pass the probe's raw values and the code dimension at the same evaluations"
        )
    zero = np.where(has, np.where(nan, 0.0, r), 0.0).astype(np.int64)
    return CyclicDistance(raw=r.copy(), has_code=has, zero_extended=zero)


def cyclic_distance_from_code(code: StabilizerCode) -> float:
    """``d_cyc`` of a stored code (v1.4 §5.5): NaN when ``k = 0``, else the shortest failing interval.

    Uses :meth:`StabilizerCode.contiguous_distance` with the identity order ``0..n-1`` and
    periodic windows, the study's ring (v1.4 §5.5: the original cyclic order at every ``beta``),
    so it equals the engine probe at the same evaluation (a retrospective cross-check).
    """
    if not isinstance(code, StabilizerCode):
        raise TypeError(f"code must be a StabilizerCode, got {type(code).__name__}")
    if code.num_logical == 0:
        return math.nan
    return float(code.contiguous_distance(periodic=True))


def connected_distance(code: StabilizerCode, edges: Any, *, max_n: int = 20) -> float:
    """``d_conn(G, t)`` of v1.4 §5.6 by exact enumeration of connected regions in increasing size.

    For a code with ``k > 0``,
    ``d_conn = min{|A| : A subset of V, G[A] connected, f(A) = 1}`` (``inf`` when no connected
    region fails, which needs a disconnected ``G``); ``0`` when ``k = 0`` (v1.4's definition). The
    connected regions of each size ``m + 1`` are the connected regions of size ``m`` with one
    neighbor added (every connected graph has a vertex whose removal leaves it connected: a leaf
    of a spanning tree), each kept once; the regions of one size are tested for failure in one
    batch (:meth:`StabilizerCode.region_failures`), and the first size with a failing region is
    returned. The number of connected regions can grow exponentially with ``n``, so ``n`` above
    ``max_n`` (default 20) is refused; raise ``max_n`` to accept the cost. On ``C(n, 2)`` every
    interval is connected, so ``d_conn <= d_cyc`` there; the two need not be equal (v1.4 §5.6).
    """
    if not isinstance(code, StabilizerCode):
        raise TypeError(f"code must be a StabilizerCode, got {type(code).__name__}")
    n = code.num_qubits
    limit = _int(max_n, "max_n", minimum=1)
    if n > limit:
        raise ValueError(
            f"d_conn enumerates connected regions, whose number can grow exponentially with n: "
            f"n = {n} is above max_n = {limit}. Pass a larger max_n to accept the cost (time and "
            "memory grow with the number of connected regions), or use d_cyc, P_C(m) or D(J) "
            "at this size"
        )
    if n > 62:
        raise ValueError(f"connected_distance holds regions as 64-bit masks: n = {n} is above 62")
    e = graphs.from_edges(edges, n)
    if code.num_logical == 0:
        return 0.0
    neighbors = np.zeros(n, dtype=np.uint64)
    for a, b in e.tolist():
        neighbors[a] |= np.uint64(1) << np.uint64(b)
        neighbors[b] |= np.uint64(1) << np.uint64(a)
    bit = np.uint64(1) << np.arange(n, dtype=np.uint64)
    level = bit.copy()  # the connected regions of size 1
    for m in range(1, n + 1):
        members = (level[:, None] & bit[None, :]) != 0  # (R, n), m True per row
        regions = np.nonzero(members)[1].reshape(level.size, m).astype(np.int64)
        if code.region_failures(regions).any():
            return float(m)
        if m == n:
            break
        reach = np.zeros(level.size, dtype=np.uint64)  # the neighbors of each region
        for v in range(n):
            reach[members[:, v]] |= neighbors[v]
        reach &= ~level
        grown = [level[(reach & bit[v]) != 0] | bit[v] for v in range(n)]
        level = np.unique(np.concatenate(grown))
        if level.size == 0:
            break
    return math.inf


def half_cut_entropy(tableau: PhaseFreeTableau, num_system: int) -> int:
    """``S(A)`` in bits of the system qubits ``A = {0, ..., floor(n/2) - 1}``, from the stored joint state.

    The engine's ``half_cut_entropy`` probe takes the first half of ``system_order``, the
    identity order in the study, so the two agree. ``tableau`` is the joint phase-free row space
    (system ``0..n-1`` first, then the references); ``S(A) = rank(S|_A) - |A|`` for the pure joint
    state (``PhaseFreeTableau.entropy``).
    """
    if not isinstance(tableau, PhaseFreeTableau):
        raise TypeError(f"tableau must be a PhaseFreeTableau, got {type(tableau).__name__}")
    n = _int(num_system, "num_system", minimum=1)
    if n > tableau.num_qubits:
        raise ValueError(f"num_system = {n} exceeds the tableau's {tableau.num_qubits} qubits")
    return int(tableau.entropy(range(n // 2)))


# ---------------------------------------------------------------------------------------------
# uniform erasures (v1.4 §5.7 at J = 0)
# ---------------------------------------------------------------------------------------------


def _first_failure(first_failure: Any, n: int) -> np.ndarray:
    d = _numbers(first_failure, "first_failure")
    if d.ndim == 0 or d.shape[-1] == 0:
        raise ValueError("first_failure needs a last axis of samples (at least one ordering)")
    ok = np.isfinite(d) & (d >= 1) & (d <= n + 1)
    ok &= np.where(np.isfinite(d), d == np.round(d), False)
    if not ok.all():
        idx = tuple(int(i) for i in np.argwhere(~ok)[0])
        raise ValueError(
            f"first_failure{_index(idx)} = {d[idx]!r}: first-failure sizes are whole numbers in "
            f"[1, n + 1] = [1, {n + 1}]"
        )
    di = d.astype(np.int64)
    never = di == n + 1
    mixed = never.any(axis=-1) & ~never.all(axis=-1)
    if mixed.any():
        idx = tuple(int(i) for i in np.argwhere(mixed)[0])
        raise ValueError(
            f"first_failure{_index(idx)} mixes n + 1 (an ordering that never fails) with sizes <= n: "
            "every ordering of a code with k > 0 fails by m = n (I(V:R) = 2k > 0) and no ordering "
            "of a code with k = 0 fails, so the samples of one code must agree"
        )
    return di


def _erasure_data(
    first_failure: Any, n: int, exact_counts: Any, k: Any = None
) -> tuple[np.ndarray, Any]:
    """The first-failure sizes and the exhaustive counts of the same codes, checked against each other.

    Consequences of the definitions that hold exactly, so data from mismatched codes or
    evaluations is refused: a count of 0 at size ``m`` (no ``m``-region fails) means no ordering
    fails by ``m`` (every ``D_j > m``); a code with ``k = 0`` (every ``D_j = n + 1``) has no
    failing region at all (every count 0); the count at ``m = n``, when given, is ``1[k > 0]``;
    and the code dimension ``k``, when given, is positive exactly when the orderings fail.
    """
    d = _first_failure(first_failure, n)
    if k is not None:
        kk = _whole(k, "k")
        if kk.shape != d.shape[:-1]:
            raise ValueError(f"k must have shape {d.shape[:-1]} (one per code), got {kk.shape}")
        bad = (kk > 0) != (d[..., 0] <= n)
        if bad.any():
            idx = tuple(int(i) for i in np.argwhere(bad)[0])
            raise ValueError(
                f"k{_index(idx)} = {kk[idx]} contradicts first_failure: the orderings of a code fail "
                "(D <= n) exactly when k > 0; give k at the same evaluations as the samples"
            )
    if exact_counts is None:
        return d, None
    counts = _numbers(exact_counts, "exact_counts")
    if counts.ndim == 0 or counts.shape[:-1] != d.shape[:-1] or counts.shape[-1] > n:
        raise ValueError(
            f"exact_counts must have shape {d.shape[:-1]} + (w,) with w <= n = {n} (counts of the "
            f"failing regions of sizes 1..w, one row per code of first_failure), got {counts.shape}"
        )
    sizes = np.arange(1, counts.shape[-1] + 1)
    has = d[..., 0] <= n
    zero = counts == 0
    clash = zero & (d.min(axis=-1)[..., None] <= sizes)
    clash |= ~has[..., None] & ~zero
    if clash.any():
        idx = tuple(int(i) for i in np.argwhere(clash)[0])
        raise ValueError(
            f"exact_counts{_index(idx)} = {counts[idx]!r} contradicts first_failure: a count of 0 at "
            "size m means no ordering fails by m, and a code whose orderings never fail (k = 0) "
            "has no failing region; give the counts and the samples of the same codes and evaluations"
        )
    return d, counts


@dataclass(frozen=True, eq=False)
class FailureProfile:
    """``P_C(m)`` for ``m = 0..n`` with its standard errors (v1.4 §5.7 at ``J = 0``).

    * ``curve``: float64 ``(..., n + 1)``, the hybrid estimate: exact for ``m = 0`` (0),
      ``1 <= m <= w`` (exhaustive counts) and ``m = n`` (``1[k > 0]``), the fraction of the
      ``S`` orderings with first-failure size ``D_j <= m`` otherwise;
    * ``se``: float64 ``(..., n + 1)``, 0 on the exact entries, ``sqrt(P (1 - P) / S)`` on the
      sampled ones (binomial: the orderings are independent); a sampled 0 or 1 gets 0, which
      reads as "within about ``3 / S`` of it";
    * ``exact``: bool ``(n + 1,)``, which entries are exact;
    * ``samples`` (``S``), ``max_exact`` (``w``, 0 without counts), ``num_system``.
    """

    num_system: int
    curve: np.ndarray
    se: np.ndarray
    exact: np.ndarray
    samples: int
    max_exact: int

    def __post_init__(self) -> None:
        _read_only(self)


def uniform_failure(
    first_failure: Any, num_system: int, *, exact_counts: Any = None, k: Any = None
) -> FailureProfile:
    """``P_C(m)``, the probability that a uniformly random ``m``-qubit erasure fails, ``m = 0..n`` (v1.4 §5.7).

    ``P_C(m) = C(n, m)^{-1} sum_{|A| = m} f(A)``, the ``J = 0`` case of ``P_{C,G,J}(m)``.
    ``first_failure`` has shape ``(..., S)``: the engine's ``erasure_first_failure`` sizes
    ``D_j`` (``n + 1`` for a code with ``k = 0``); ``exact_counts`` (optional, ``(..., w)``) the
    engine's ``uncorrectable_region_counts``. The estimate is
    :func:`sparsegf2.analysis.hybrid_failure_by_size`: the prefix of length ``m`` of a uniformly
    random ordering is a uniformly random ``m``-subset and it fails exactly when ``D_j <= m``
    (``f`` is monotone along nested prefixes), so ``mean_j 1[D_j <= m]`` is unbiased, with
    binomial variance; the entries ``m <= w`` are exact. Counts that contradict the samples (a
    zero count at a size some ordering already failed at, or failing regions of a code with
    ``k = 0``) are refused, and so is a code dimension ``k`` (optional, ``(...)``, at the same
    evaluations) that is positive where the orderings never fail or zero where they do.
    """
    n = _int(num_system, "num_system", minimum=1)
    d, counts = _erasure_data(first_failure, n, exact_counts, k)
    curve = analysis.hybrid_failure_by_size(d, n, exact_counts=counts)
    w = 0 if counts is None else int(counts.shape[-1])
    exact = np.zeros(n + 1, dtype=bool)
    exact[[0, n]] = True
    exact[1 : w + 1] = True
    samples = int(d.shape[-1])
    se = np.sqrt(curve * (1.0 - curve) / samples)
    se[..., exact] = 0.0
    return FailureProfile(
        num_system=n, curve=curve, se=se, exact=exact, samples=samples, max_exact=min(w, n)
    )


@dataclass(frozen=True, eq=False)
class Robustness:
    """``D(0)`` per code with its standard error (v1.4 §5.7).

    * ``value``: float64 ``(...)``, ``D(0) = sum_{m=0}^{n-1} (1 - P_C(m))``, 0 where ``k = 0``;
    * ``se``: float64 ``(...)``, the standard error of the estimate (0 where ``k = 0`` and where
      the exact counts cover every size; NaN with a single ordering);
    * ``has_code``: bool ``(...)``; ``samples`` (``S``); ``max_exact`` (``w``); ``num_system``.
    """

    num_system: int
    value: np.ndarray
    se: np.ndarray
    has_code: np.ndarray
    samples: int
    max_exact: int

    def __post_init__(self) -> None:
        _read_only(self)


def integrated_robustness(
    first_failure: Any, num_system: int, *, exact_counts: Any = None, k: Any = None
) -> Robustness:
    """The integrated first-failure robustness ``D(0) = sum_{m=0}^{n-1} (1 - P_C(m)) = E_pi[M_pi]`` (v1.4 §5.7).

    ``D(0) = 0`` when ``k = 0`` (v1.4's definition). For ``k > 0``, with ``M_pi`` the first failing
    prefix size of a uniformly random ordering ``pi`` of the qubits:

    *Proof that D(0) = E_pi[M_pi].* ``f`` is monotone under inclusion and the prefixes of ``pi``
    are nested, so ``M_pi > m`` exactly when the prefix of length ``m`` is correctable; that
    prefix is a uniformly random ``m``-subset, so ``Pr(M_pi > m) = 1 - P_C(m)``. ``M_pi`` is an
    integer in ``[1, n]`` (``f(V) = 1`` since ``I(V:R) = 2k > 0``), and for such a variable
    ``E[M] = sum_{m >= 0} Pr(M > m) = sum_{m=0}^{n-1} Pr(M > m)``. ∎

    **Estimator.** ``first_failure`` ``(..., S)`` holds independent draws ``D_1..D_S`` of
    ``M_pi`` (the engine's ``erasure_first_failure``). Without counts the estimate is their mean,
    unbiased, with standard error ``s_D / sqrt(S)`` (``s_D`` the sample standard deviation). With
    exhaustive counts for ``m = 1..w`` (``exact_counts``, ``(..., w)``), the terms ``m <= w`` are
    exact and the rest is sampled:

        D_hat = 1 + sum_{m=1}^{w} (1 - P_C(m)) + mean_j Y_j,    Y_j = max(D_j, w + 1) - (w + 1),

    because ``sum_{m=w+1}^{n-1} 1[D_j > m] = Y_j`` for ``D_j <= n``; it is unbiased (each term
    is) and its standard error is ``s_Y / sqrt(S)``, never larger than ``s_D / sqrt(S)`` since
    the exact terms carry no noise. The value is the hybrid curve of :func:`uniform_failure`
    (:func:`sparsegf2.analysis.hybrid_failure_by_size`) summed over ``m = 0..n-1``, which is
    :func:`sparsegf2.analysis.mean_first_failure` for ``k > 0`` (that sum runs to ``m = n``,
    where ``P_C(n) = 1``) and 0 for ``k = 0``. Counts or a code dimension ``k`` that contradict
    the samples are refused (see :func:`uniform_failure`).
    """
    n = _int(num_system, "num_system", minimum=1)
    d, counts = _erasure_data(first_failure, n, exact_counts, k)
    has = d[..., 0] <= n
    w = 0 if counts is None else int(counts.shape[-1])
    curve = analysis.hybrid_failure_by_size(d, n, exact_counts=counts)
    value = np.where(has, (1.0 - curve[..., :n]).sum(axis=-1), 0.0)
    samples = int(d.shape[-1])
    if samples >= 2:
        y = (np.maximum(d, w + 1) - (w + 1)).astype(np.float64)
        se = np.sqrt(y.var(axis=-1, ddof=1) / samples)
    else:
        se = np.full(d.shape[:-1], np.nan)
    se = np.where(has, se, 0.0)
    return Robustness(
        num_system=n, value=value, se=se, has_code=has, samples=samples, max_exact=min(w, n)
    )


@dataclass(frozen=True, eq=False)
class ChannelFailure:
    """``Phi(e)``, the failure probability of the binomially mixed erasure channel (v1.4 §5.7).

    * ``e``: float64 ``(q,)``, the erasure probabilities;
    * ``value``: float64 ``(..., q)`` (``(..., c, q)`` for a correlated profile), the estimate of
      ``Phi(e) = sum_{m=0}^{n} C(n, m) e^m (1 - e)^(n - m) P(m)``; 0 where ``k = 0``;
    * ``se``: its standard error (same shape);
    * ``interpolated_mass``: float64 ``(q,)``, ``sum_{m interpolated} C(n, m) e^m (1 - e)^(n-m)``:
      the binomial weight of the sizes whose ``P(m)`` was interpolated (always 0 for the uniform
      estimate); ``num_system``.
    """

    num_system: int
    e: np.ndarray
    value: np.ndarray
    se: np.ndarray
    interpolated_mass: np.ndarray

    def __post_init__(self) -> None:
        _read_only(self)


def _rates(e: Any) -> np.ndarray:
    es = np.atleast_1d(_numbers(e, "e"))
    if es.ndim != 1 or es.size == 0 or not np.all((es >= 0) & (es <= 1)):
        raise ValueError("e must be a probability in [0, 1] or a 1-D list of them")
    return es


def _binomial_tails(n: int, es: np.ndarray) -> np.ndarray:
    """``T[d, i] = Pr[Bin(n, e_i) >= d]`` for ``d = 0..n+1``: the channel of the step curves ``1[m >= d]``."""
    steps = (np.arange(n + 1)[None, :] >= np.arange(1, n + 1)[:, None]).astype(np.float64)
    tails = np.empty((n + 2, es.size))
    tails[0] = 1.0
    tails[1 : n + 1] = analysis.failure_probability_from_curve(steps, n, es)
    tails[n + 1] = 0.0
    return tails


def uniform_channel_failure(
    first_failure: Any, num_system: int, e: Any, *, exact_counts: Any = None, k: Any = None
) -> ChannelFailure:
    """``Phi_C(e) = sum_m C(n, m) e^m (1 - e)^(n - m) P_C(m)``: independent erasure with probability ``e`` (v1.4 §5.7).

    Each qubit erased independently with probability ``e`` erases ``M ~ Bin(n, e)`` qubits, a
    uniformly random set given its size, so the failure probability is the binomial mixture of
    ``P_C(m)`` (``J = 0``; ``e`` is not the measurement rate ``p``). The estimate is
    :func:`sparsegf2.analysis.erasure_failure_probability` (the binomial mixture of the hybrid
    curve of :func:`uniform_failure`):

        Phi_hat(e) = sum_{m=1}^{w} b_m(e) P_C(m) + mean_j T_e(max(D_j, w + 1)),

    ``b_m(e) = C(n, m) e^m (1 - e)^(n - m)``, ``T_e(d) = Pr[Bin(n, e) >= d]`` (``T_e(n + 1) = 0``),
    since ``sum_{m >= w+1} b_m 1[D_j <= m] = T_e(max(D_j, w + 1))``. Unbiased; its standard error
    is ``s_Z / sqrt(S)`` with ``Z_j = T_e(max(D_j, w + 1))`` (the orderings are independent). A
    code with ``k = 0`` gives 0. Inputs as :func:`uniform_failure`.
    """
    n = _int(num_system, "num_system", minimum=1)
    d, counts = _erasure_data(first_failure, n, exact_counts, k)
    es = _rates(e)
    value = np.asarray(analysis.erasure_failure_probability(d, n, es, exact_counts=counts))
    w = 0 if counts is None else int(counts.shape[-1])
    z = _binomial_tails(n, es)[np.maximum(d, w + 1)]  # (..., S, q)
    samples = int(d.shape[-1])
    spread = np.sqrt(z.var(axis=-2, ddof=1) / samples) if samples >= 2 else np.nan
    se = np.where((d[..., 0] <= n)[..., None], spread, 0.0)
    return ChannelFailure(
        num_system=n, e=es, value=value, se=se, interpolated_mass=np.zeros(es.size)
    )


# ---------------------------------------------------------------------------------------------
# geometry-correlated erasures (v1.4 §5.7)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CorrelatedProfile:
    """``P_{C,G,J}(m)`` and ``D(J)`` from a ``correlated_erasure`` probe (v1.4 §5.7).

    With ``c`` couplings, ``s`` sampled sizes and any leading axes ``...`` (trajectories,
    evaluations):

    * ``failure``: float64 ``(..., c, s)``, the probe's estimates ``P_hat_J(m)`` (the failing
      fraction of ``samples`` regions drawn from ``Q_{G,J}(. | m)``); ``se``: their binomial
      errors ``sqrt(P (1 - P) / S)``: exact for independent samples (``sampler["method"]`` is
      ``"exact"``), an underestimate by the factor ``sqrt(tau_int)`` for the chain, whose records
      are correlated (the probe stores no ``tau_int``; the spread over trajectories, whose
      chains are independent, is the valid error of an ensemble mean);
    * ``curve``: float64 ``(..., c, n + 1)``, ``P_J(m)`` for ``m = 0..n``: ``P(0) = 0``,
      ``P(n) = 1[k > 0]``, the sampled sizes, and linear interpolation in ``m`` at the other
      sizes (:func:`sparsegf2.analysis.correlated_failure_curve`); ``interpolated``: bool
      ``(n + 1,)``, where the curve was interpolated; ``interpolated_sizes``: those ``m``;
    * ``robustness``: float64 ``(..., c)``, ``D(J) = sum_{m=0}^{n-1} (1 - P_J(m))`` of the curve
      (0 where ``k = 0``); ``robustness_se``: its standard error from the independent sizes;
      ``complete``: True when no size in ``1..n-1`` was interpolated (otherwise ``D(J)`` leans
      on the interpolation and is marked incomplete);
    * ``couplings``, ``sizes`` (as the probe lists them), ``samples``, ``has_code``,
      ``num_system``, and ``sampler``: the probe's region-sampling method and chain settings as
      given (``MonitoredCircuit.correlated_sampler(name)``), carried through for the record.
    """

    num_system: int
    couplings: np.ndarray
    sizes: np.ndarray
    samples: int
    failure: np.ndarray
    se: np.ndarray
    has_code: np.ndarray
    curve: np.ndarray
    interpolated: np.ndarray
    robustness: np.ndarray
    robustness_se: np.ndarray
    complete: bool
    sampler: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        _read_only(self)

    @property
    def interpolated_sizes(self) -> np.ndarray:
        """The sizes ``m`` in ``1..n-1`` whose ``P_J(m)`` was interpolated, not sampled."""
        return np.flatnonzero(self.interpolated).astype(np.int64)


def correlated_failure(
    values: Any,
    couplings: Any,
    sizes: Any,
    num_system: int,
    k: Any,
    *,
    samples: int,
    sampler: Mapping[str, Any] | None = None,
) -> CorrelatedProfile:
    """``P_{C,G,J}(m)`` with binomial errors and ``D(J) = sum_{m=0}^{n-1} (1 - P_J(m))`` (v1.4 §5.7).

    ``values`` ``(..., c, s)`` are a ``correlated_erasure`` probe's estimates
    (``EnsembleResult.correlated(name)``): at coupling ``couplings[i]`` and size ``sizes[j]``,
    the fraction of ``samples`` regions drawn from
    ``Q_{G,J}(A | m) = exp(-J b_G(A)) / Z_{G,m}(J)`` that fail. ``k`` ``(...)`` is the code
    dimension at the same evaluations: where ``k = 0`` the probe records 0 (checked) and
    ``D(J) = 0``. At ``J = 0`` the law is uniform and ``P_{C,G,0}(m) = P_C(m)``.

    **D(J).** The full curve takes ``P(0) = 0``, ``P(n) = 1[k > 0]``, the sampled sizes, and
    linear interpolation between the nearest known sizes elsewhere; ``D(J)`` sums ``1 - P_J(m)``
    over ``m = 0..n-1``. It is unbiased when every size ``1..n-1`` was sampled (exact samplers;
    the chain up to its equilibration) and ``complete`` is then True; otherwise the interpolated
    sizes are flagged and ``complete`` is False. ``D(J) <= (n + 1) / 2`` for the exact profile
    (v1.4 §5.8); at ``J > 0`` it is a weighted integrated correctability score, not an expected
    first-failure size.

    **Errors.** Each size draws from its own stream (``RANDOMNESS_CONTRACT = 2``), so the
    ``P_hat_J(m)`` of different sizes are independent; ``D_hat(J)`` is linear in them,
    ``D_hat = n - sum_j a_j P_hat_J(m_j)`` with ``a_j`` the weight of size ``m_j`` in the
    interpolated sum, so ``Var D_hat = sum_j a_j^2 Var P_hat_J(m_j)``, with the binomial
    variances (see :class:`CorrelatedProfile` for the chain).
    """
    n = _int(num_system, "num_system", minimum=1)
    count = _int(samples, "samples", minimum=1)
    js = _numbers(couplings, "couplings")
    if js.ndim != 1 or js.size == 0 or not np.all(np.isfinite(js)):
        raise ValueError("couplings must be a 1-D list of finite numbers (the probe's couplings)")
    if np.unique(js).size != js.size:
        raise ValueError("couplings must be distinct")
    ms = _whole(sizes, "sizes")
    if ms.ndim != 1 or ms.size == 0 or np.any(ms > n) or np.unique(ms).size != ms.size:
        raise ValueError(f"sizes must be a 1-D list of distinct sizes in [0, {n}]")
    v = _numbers(values, "values")
    if v.ndim < 2 or v.shape[-2:] != (js.size, ms.size):
        raise ValueError(
            f"values must have shape (..., {js.size}, {ms.size}) (couplings, sizes), got {v.shape}"
        )
    kk = _whole(k, "k")
    if kk.shape != v.shape[:-2]:
        raise ValueError(f"k must have the leading shape of values, {v.shape[:-2]}, got {kk.shape}")
    if not np.all(np.isfinite(v)) or np.any((v < 0) | (v > 1)):
        raise ValueError("values must be failing fractions in [0, 1]")
    scaled = v * count
    if np.any(np.abs(scaled - np.round(scaled)) > 1e-9 * count):
        raise ValueError(
            f"values must be multiples of 1 / samples = 1 / {count} (failing fractions of the "
            "recorded regions): check samples"
        )
    has = kk > 0
    if np.any(v[~has] != 0):
        raise ValueError("the probe records 0 for a code with k = 0, but values are nonzero there")
    curve, interpolated = analysis.correlated_failure_curve(v, ms, n, holds_code=has)
    robustness = np.where(has[..., None], (1.0 - curve[..., :n]).sum(axis=-1), 0.0)
    se = np.sqrt(v * (1.0 - v) / count)
    basis, _ = analysis.correlated_failure_curve(
        np.eye(ms.size)[:, None, :], ms, n, holds_code=np.zeros(ms.size)
    )
    weight = basis[:, 0, :n].sum(axis=-1)  # a_j: the weight of size m_j in sum_{m<n} P(m)
    robustness_se = np.where(has[..., None], np.sqrt((se**2 * weight**2).sum(axis=-1)), 0.0)
    return CorrelatedProfile(
        num_system=n,
        couplings=js.copy(),
        sizes=ms.copy(),
        samples=count,
        failure=v.copy(),
        se=se,
        has_code=has,
        curve=curve,
        interpolated=interpolated,
        robustness=robustness,
        robustness_se=robustness_se,
        complete=not bool(interpolated[1:n].any()),
        sampler=None if sampler is None else MappingProxyType(dict(sampler)),
    )


def correlated_channel_failure(profile: CorrelatedProfile, e: Any) -> ChannelFailure:
    """``Phi_{C,G,J}(e) = sum_m C(n, m) e^m (1 - e)^(n - m) P_{C,G,J}(m)`` of a correlated profile (v1.4 §5.7).

    The binomially mixed correlated channel: ``M ~ Bin(n, e)`` erased qubits, the region drawn
    from ``Q_{G,J}(. | M)``. The value is
    :func:`sparsegf2.analysis.failure_probability_from_curve` of the profile's full curve
    (``(..., c, q)``; 0 where ``k = 0``). It is linear in the sampled ``P_hat_J(m_j)``, whose sizes
    are independent, so ``Var Phi_hat = sum_j a_j(e)^2 Var P_hat_J(m_j)`` with ``a_j(e)`` the
    binomial weight that size ``m_j`` receives directly and through the interpolation (binomial
    variances: see :class:`CorrelatedProfile` for the chain). ``interpolated_mass`` is the
    binomial weight of the interpolated sizes: where it is not small, ``Phi`` leans on the
    interpolation (:func:`sparsegf2.analysis.binomial_window` gives the sizes that matter).
    """
    if not isinstance(profile, CorrelatedProfile):
        raise TypeError(f"profile must be a CorrelatedProfile, got {type(profile).__name__}")
    n = profile.num_system
    es = _rates(e)
    value = analysis.failure_probability_from_curve(profile.curve, n, es)
    ms = profile.sizes
    basis, _ = analysis.correlated_failure_curve(
        np.eye(ms.size)[:, None, :], ms, n, holds_code=np.zeros(ms.size)
    )
    weight = analysis.failure_probability_from_curve(basis[:, 0, :], n, es)  # a_j(e): (s, q)
    se = np.sqrt(np.einsum("...j,jq->...q", profile.se**2, weight**2))
    se = np.where(profile.has_code[..., None, None], se, 0.0)
    mass = analysis.failure_probability_from_curve(profile.interpolated.astype(np.float64), n, es)
    return ChannelFailure(num_system=n, e=es, value=value, se=se, interpolated_mass=mass)


REGION_SEED_TAG = 0x5753524F
"""The first word of every retrospective region-sampler seed: ``"WSRO"`` read as a big-endian word.

It is the domain of these seeds, distinct from the graph seeds' ``0x57534731`` (``"WSG1"``) and the
run seeds' ``0x57534331`` (``"WSC1"``) of design §3 (ruling R-SEED)."""

REGION_SEED_WORDS = 19
"""The length of every region-sampler seed (one fixed length: ``SeedSequence`` never pads it)."""


def _label(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{what} must be a non-empty string without NUL characters, got {value!r}")
    return value


def _text_words(text: str) -> tuple[int, int]:
    """``w(x)``: the 32-bit words (low, high) of ``blake2b(x.encode(), digest_size=8)`` read little-endian.

    The repository's convention for a string in a seed (design §3, ruling R-SEED: the probe-name
    hash of :mod:`sparsegf2.mipt`, :func:`~sparsegf2.study.identity.rate_hash`), for any string:
    :func:`~sparsegf2.study.identity.token_words` gives the same words for a probability token
    (and is used for the tokens), but parses its input as one, so the experiment id, the
    observable id and the version go through this function.
    """
    value = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")
    return value & 0xFFFFFFFF, value >> 32


def _seed_word(value: Any, what: str) -> int:
    v = _int(value, what, minimum=0)
    if v >= 2**32:
        raise ValueError(f"{what} must be below 2**32 (one 32-bit seed word), got {v}")
    return v


def _float_words(value: float) -> tuple[int, int]:
    """The IEEE-754 binary64 bits of ``value`` as two 32-bit words (low, high): exact, one to one."""
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    return bits & 0xFFFFFFFF, bits >> 32


def _coupling(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"a coupling J must be a number, got {value!r}")
    try:
        j = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"a coupling J must be a number, got {value!r}") from None
    if not math.isfinite(j) or abs(j) > erasures.MAX_ABS_COUPLING:
        raise ValueError(erasures.coupling_bound_message(j))
    return 0.0 if j == 0 else j


def _trajectory(key: Any) -> TrajectoryKey:
    if isinstance(key, TrajectoryKey):
        return key
    if isinstance(key, str):
        return TrajectoryKey.parse(key)
    raise TypeError(
        f"key must be a TrajectoryKey (or its string form), got {type(key).__name__}: the seed of "
        "a retrospective sampler is derived from the trajectory's full identity"
    )


def region_seed(
    key: TrajectoryKey | str,
    t: int,
    m: int,
    coupling: float,
    observable_id: str,
    version: str,
) -> tuple[int, ...]:
    """The seed of a retrospective region sampler: a fixed-length, domain-tagged tuple of 32-bit words.

    For the trajectory ``key = (experiment_id, n, beta, p, g, c, r)`` (a
    :class:`~sparsegf2.study.identity.TrajectoryKey`), the depth ``t``, the region size ``m``, the
    coupling ``J``, the observable ``observable_id`` and its ``version`` (design §3, ruling
    R-SEED)::

        seed = (TAG, w(experiment_id), n, w(beta_token), w(p_token), g, c, r, t, m,
                bits(J), w(observable_id), w(version))                       # 19 words

    with ``TAG`` = :data:`REGION_SEED_TAG` (``0x5753524F``, ``"WSRO"``), ``w(x)`` the two words
    (low, high) of ``blake2b(x.encode(), digest_size=8)`` read little-endian (the repository's
    convention), ``bits(J)`` the two words (low, high) of ``J``'s IEEE-754 binary64 pattern
    (``-0.0`` written as ``0.0``), and every other field one word (refused at ``2**32`` and above,
    so that each field is exactly one word of the ``SeedSequence`` entropy). The tuple goes to
    ``erasures.sample_regions(..., seed=list(seed))``.

    *Distinct inputs, distinct seeds.* The integer fields and ``bits(J)`` are one-to-one, and the
    fields sit at fixed positions of a fixed-length tuple, so two inputs give the same tuple only
    if two different strings of one field (experiment ids, tokens, ids or versions) share a 64-bit
    blake2b digest. *No collision with the graph and run seeds.* Those tuples start with
    ``0x57534731`` (``"WSG1"``) and ``0x57534331`` (``"WSC1"``) and these with ``0x5753524F``, so
    no region seed equals a graph or run seed as a ``SeedSequence`` entropy, whatever the master
    seeds; every one of these tuples has at least 4 words, so ``SeedSequence``'s zero padding
    (to its 4-word pool) never applies. The streams then coincide only through a collision of
    ``SeedSequence``'s 128-bit hash. The seed depends on every field of the key: two trajectories
    (also the same ``(g, c)`` at two rates) draw independent regions; a new ``version`` draws new
    ones.

    >>> from sparsegf2.study.identity import TrajectoryKey
    >>> key = TrajectoryKey("demo", 8, "0.1", "0.2", 3, 1)
    >>> seed = region_seed(key, 16, 4, 1.0, "mean_leakage", "1")
    >>> len(seed), hex(seed[0]), seed == region_seed(str(key), 16, 4, 1.0, "mean_leakage", "1")
    (19, '0x5753524f', True)
    """
    trajectory = _trajectory(key)
    words = (
        REGION_SEED_TAG,
        *_text_words(trajectory.experiment_id),
        _seed_word(trajectory.n, "n"),
        *token_words(trajectory.beta),
        *token_words(trajectory.p),
        _seed_word(trajectory.g, "g"),
        _seed_word(trajectory.c, "c"),
        _seed_word(trajectory.r, "r"),
        _seed_word(t, "t"),
        _seed_word(m, "m"),
        *_float_words(_coupling(coupling)),
        *_text_words(_label(observable_id, "observable_id")),
        *_text_words(_label(version, "version")),
    )
    assert len(words) == REGION_SEED_WORDS
    return tuple(int(w) for w in words)


@dataclass(frozen=True, eq=False)
class Leakage:
    """One estimate of the mean leaked information ``l(m, J) = E_{A ~ Q_{G,J}(.|m)}[I(A:R)]`` (v1.4 §5.7).

    * ``value``: the estimate (bits); ``se``: its standard error (0 when exact);
    * ``ess``: the effective number of independent regions behind it (``samples`` for exact
      samples, the chain's ``samples / tau_int`` of the series of ``I(A_s:R)`` capped at
      ``samples``);
    * ``method``: ``"exact"`` or ``"chain"`` (the region sampler), ``"enumerate"`` (every region,
      exact) or ``"none"`` (``k = 0`` or ``m = 0``: ``l = 0`` exactly, nothing drawn);
    * ``size`` (``m``), ``coupling`` (``J``), ``samples``, ``seed`` (:func:`region_seed`, None for
      ``"enumerate"``), ``warnings`` (the sampler's), ``burn_in``, ``thin``, ``tempering``,
      ``num_logical`` (``k``).
    """

    size: int
    coupling: float
    value: float
    se: float
    ess: float
    samples: int
    method: str
    seed: tuple[int, ...] | None
    warnings: tuple[str, ...]
    burn_in: int
    thin: int
    tempering: bool
    num_logical: int


MEAN_LEAKAGE_VERSION = "1"


def mean_leakage(
    code: StabilizerCode,
    edges: Any,
    size: int,
    coupling: float,
    *,
    key: TrajectoryKey | str,
    t: int,
    samples: int = 100,
    burn_in: int | None = None,
    thin: int = 1,
    tempering: bool = True,
    method: str = "auto",
    ladder: str | Sequence[float] = "auto",
    max_step: float | None = None,
    observable_id: str = "mean_leakage",
    version: str = MEAN_LEAKAGE_VERSION,
) -> Leakage:
    """Estimate ``l(m, J) = E_{A ~ Q_{G,J}(. | m)}[I(A:R)]``, the mean leaked information (v1.4 §5.7).

    ``code`` is the stored physical code of the trajectory ``key`` at depth ``t`` and ``edges``
    its interaction graph ``G``. ``samples`` regions of size ``m = size`` are drawn from
    ``Q_{G,J}(A | m) ∝ exp(-J b_G(A))`` by :func:`sparsegf2.erasures.sample_regions` (exact
    independent samples on a cycle or a path, else the swap chain with replica exchange; the
    keywords are passed on) with the seed :func:`region_seed` of
    ``(key, t, m, J, observable_id, version)``, so the estimate is a pure function of the stored
    data and its identity. ``I(A_s:R)`` is computed exactly for each region
    (:meth:`StabilizerCode.lost_information`) and the estimate is their mean.

    **Error.** For exact samples the records are independent and ``se = s / sqrt(S)``; for the
    chain ``se = s / sqrt(ess)``, ``ess = min(S / tau_int, S)`` with ``tau_int`` the integrated
    autocorrelation time of the series ``I(A_s:R)`` (:func:`sparsegf2.erasures.chain_diagnostics`,
    capped so the error is never below the independent one). ``s`` is the sample standard
    deviation. The chain's error does not include a bias from incomplete equilibration; the
    sampler's warnings are carried in ``warnings``.

    A code with ``k = 0`` and the size ``m = 0`` give ``l = 0`` exactly (nothing is drawn);
    ``m = n`` gives ``2k``. The exact identities ``l(m, J) + l(n - m, J) = 2k`` and
    ``l(n/2, J) = k`` hold for the exact values (:func:`mean_leakage_exact`) and in expectation
    for these estimates.
    """
    if not isinstance(code, StabilizerCode):
        raise TypeError(f"code must be a StabilizerCode, got {type(code).__name__}")
    n = code.num_qubits
    m = _int(size, "size", minimum=0)
    if m > n:
        raise ValueError(f"size must be in [0, {n}], got {m}")
    j = _coupling(coupling)
    count = _int(samples, "samples", minimum=1)
    trajectory = _trajectory(key)
    if trajectory.n != n:
        raise ValueError(f"the key is a trajectory of n = {trajectory.n} qubits, the code has {n}")
    seed = region_seed(trajectory, t, m, j, observable_id, version)
    k = code.num_logical
    if k == 0 or m == 0:
        graphs.from_edges(edges, n)  # still validate the graph
        return Leakage(
            size=m,
            coupling=j,
            value=0.0,
            se=0.0,
            ess=0.0,
            samples=0,
            method="none",
            seed=seed,
            warnings=(),
            burn_in=0,
            thin=_int(thin, "thin", minimum=1),
            tempering=False,
            num_logical=k,
        )
    drawn = erasures.sample_regions(
        edges,
        n,
        m,
        [j],
        count,
        burn_in=burn_in,
        thin=thin,
        tempering=tempering,
        seed=list(seed),
        method=method,
        ladder=ladder,
        max_step=max_step,
    )
    info = np.array([code.lost_information(r) for r in drawn.regions[0]], dtype=np.float64)
    value = float(info.mean())
    if count >= 2:
        variance = float(info.var(ddof=1))
        ess = (
            float(count)
            if drawn.method == "exact"
            else min(float(erasures.chain_diagnostics(info)["ess"]), float(count))
        )
        se = math.sqrt(variance / ess)
    else:
        ess, se = 1.0, math.nan
    return Leakage(
        size=m,
        coupling=j,
        value=value,
        se=se,
        ess=ess,
        samples=count,
        method=drawn.method,
        seed=seed,
        warnings=drawn.warnings,
        burn_in=drawn.burn_in,
        thin=drawn.thin,
        tempering=drawn.tempering,
        num_logical=k,
    )


def mean_leakage_exact(
    code: StabilizerCode, edges: Any, size: int, couplings: Any, *, budget: int | None = None
) -> np.ndarray:
    """``l(m, J)`` exactly, by enumerating every ``m``-subset: ``sum_A exp(-J b(A)) I(A:R) / Z_m(J)``.

    For each coupling in ``couplings`` (a number or a 1-D list); returns float64 of shape
    ``(len(couplings),)``. Every region ``A`` with ``|A| = m`` is listed once, with its boundary
    ``b_G(A)`` in the graph ``edges`` and ``I(A:R)`` from the code; the weights are shifted by the
    smallest boundary (largest for ``J < 0``) so every exponent is ``<= 0``. ``C(n, m)`` may not
    exceed ``budget`` (default :data:`sparsegf2.erasures.ENUMERATION_BUDGET`); the time is
    about ``C(n, m)`` calls of :meth:`StabilizerCode.lost_information`. For small systems and
    validation: ``l(m, J) + l(n - m, J) = 2k`` and ``l(n/2, J) = k`` hold exactly (up to
    rounding) for these values.
    """
    if not isinstance(code, StabilizerCode):
        raise TypeError(f"code must be a StabilizerCode, got {type(code).__name__}")
    n = code.num_qubits
    m = _int(size, "size", minimum=0)
    if m > n:
        raise ValueError(f"size must be in [0, {n}], got {m}")
    js = _numbers(couplings, "couplings")
    js = np.atleast_1d(js)
    if js.ndim != 1:
        raise ValueError("couplings must be a number or a 1-D list")
    for value in js.tolist():
        _coupling(value)
    limit = erasures.ENUMERATION_BUDGET if budget is None else _int(budget, "budget", minimum=1)
    total = math.comb(n, m)
    if total > limit:
        raise ValueError(
            f"enumerating every region of size {m} on {n} qubits means C({n}, {m}) = {total:,} "
            f"regions, above the budget of {limit:,}: pass a larger budget or use mean_leakage"
        )
    e = graphs.from_edges(edges, n)
    regions = np.array(list(itertools.combinations(range(n), m)), dtype=np.int64).reshape(total, m)
    member = np.zeros((total, n), dtype=bool)
    member[np.repeat(np.arange(total), m), regions.ravel()] = True
    boundary = (member[:, e[:, 0]] != member[:, e[:, 1]]).sum(axis=1).astype(np.float64)
    if code.num_logical == 0 or m == 0:
        info = np.zeros(total)
    else:
        info = np.array([code.lost_information(r) for r in regions], dtype=np.float64)
    out = np.empty(js.size)
    for i, coupling in enumerate(js.tolist()):
        shift = boundary.min() if coupling >= 0 else boundary.max()
        weight = np.exp(-coupling * (boundary - shift))
        out[i] = float(weight @ info / weight.sum())
    return out


# ---------------------------------------------------------------------------------------------
# peaks (v1.4 §5.8)
# ---------------------------------------------------------------------------------------------


PEAK_RULES = ("exact", "plateau")
"""The peak rules: exact ties (integer or exact traces) and the plateau rule (Monte Carlo traces)."""


@dataclass(frozen=True, eq=False)
class Peak:
    """The peak of one depth trace ``d(t)`` over the positive checkpoints (v1.4 §5.8).

    * ``d_star``: ``max_{t in T+} d(t)`` (NaN for an invalid trace);
    * ``maximizing_set``: int64, the set ``A`` of the rule (exact maximizers, or the plateau);
      empty for an invalid trace;
    * ``t_star``: ``exp(mean_{t in A} ln t)``, the log midpoint of ``A``; NaN when ``missing``;
    * ``missing``: True when ``d* = 0`` or the trace is invalid; ``reason`` says which;
    * ``at_horizon``: whether the horizon ``T_max`` is in ``A`` (the peak may lie beyond it);
      False when ``missing``; ``horizon``: ``T_max``;
    * ``k_on_set``: ``k_t`` at the depths of ``A`` (int64) when the ``k`` values were given;
      ``t >= tau_1`` exactly when ``k_t <= 1``, since ``tau_1`` is exact;
    * ``first_argmax``: the smallest depth with ``d(t) = d*``, the checkpoint peak (-1 for an
      invalid trace); ``t_star_is_checkpoint``: whether ``T*`` (the interpolated summary)
      equals one of the checkpoints (to a relative ``1e-12``), which it does when ``A`` has one
      element; otherwise ``k(T*)`` is undefined (v1.4 §5.8) and ``k_on_set`` is the record to use;
    * ``rule`` and ``z``: the rule and the plateau width (None for ``"exact"``).
    """

    rule: str
    z: float | None
    d_star: float
    maximizing_set: np.ndarray
    t_star: float
    missing: bool
    reason: str
    at_horizon: bool
    horizon: int
    k_on_set: np.ndarray | None
    first_argmax: int
    t_star_is_checkpoint: bool = False

    def __post_init__(self) -> None:
        _read_only(self)

    def to_dict(self) -> dict[str, Any]:
        """Plain Python (NaN as None), for a stored trajectory observable."""
        return {
            "rule": self.rule,
            "z": self.z,
            "d_star": None if math.isnan(self.d_star) else self.d_star,
            "maximizing_set": self.maximizing_set.tolist(),
            "t_star": None if math.isnan(self.t_star) else self.t_star,
            "missing": self.missing,
            "reason": self.reason,
            "at_horizon": self.at_horizon,
            "horizon": self.horizon,
            "k_on_set": None if self.k_on_set is None else self.k_on_set.tolist(),
            "first_argmax": self.first_argmax,
            "t_star_is_checkpoint": self.t_star_is_checkpoint,
        }


def _invalid_peak(
    rule: str, z: float | None, horizon: int, reason: str, k_given: bool, d_star: float = math.nan
) -> Peak:
    return Peak(
        rule=rule,
        z=z,
        d_star=d_star,
        maximizing_set=np.zeros(0, dtype=np.int64),
        t_star=math.nan,
        missing=True,
        reason=reason,
        at_horizon=False,
        horizon=horizon,
        k_on_set=np.zeros(0, dtype=np.int64) if k_given else None,
        first_argmax=-1,
    )


def peak(
    t: Any,
    d: Any,
    se: Any = None,
    *,
    rule: str,
    z: float | None = None,
    k_at: Any = None,
    horizon: int | None = None,
) -> Peak:
    """The peak depth of a nonnegative trace ``d(t)`` (v1.4 §5.8), with the rule fixed in advance.

    ``t`` are the checkpoints (strictly increasing depths ``>= 0``), ``d`` the trace there
    (for example the zero-extended ``d_cyc``, or ``D(J)``), ``se`` its standard errors for the
    plateau rule, ``k_at`` the ``k_t`` at the checkpoints (optional), ``horizon`` ``T_max``
    (default ``max(t)``). Over ``T+ = {t in checkpoints : t > 0}``:

    * ``d* = max_{T+} d(t)``;
    * ``rule="exact"`` (integer or exact traces): ``A = {t in T+ : d(t) = d*}``, exact ties;
    * ``rule="plateau"`` (Monte Carlo traces; ``se`` and ``z >= 0`` required):
      ``A = {t in T+ : d(t) >= d* - z sqrt(se(t)^2 + se(t*)^2)}`` with ``t*`` the first
      maximizer. Conditional on the trajectory, the estimates at different depths come from
      different probe streams and are independent, so ``d(t*) - d(t)`` has standard deviation
      ``sqrt(se(t)^2 + se(t*)^2)``: ``t`` stays in ``A`` unless it is more than ``z`` such
      standard deviations below the maximum. ``z = 0`` gives the exact rule; ``A`` grows with
      ``z``.
    * ``T* = exp(mean_{t in A} ln t)``, the log midpoint (geometric mean) of ``A``: it lies in
      ``[min A, max A]`` and is generally not a checkpoint.

    ``T*`` is missing when ``d* = 0`` (no robustness to place in time: a purified trajectory)
    or when the trace is invalid: no positive checkpoint, a value in ``T+`` that is NaN,
    infinite or negative (pass the zero-extended ``d_cyc``, not the raw one), or, for the
    plateau rule, a missing or invalid ``se``. ``at_horizon`` says whether ``T_max`` is in ``A``
    (the peak may lie beyond the horizon). The entry at ``t = 0`` is ignored.

    >>> p = peak([0, 1, 2, 4, 8], [0, 1, 3, 3, 2], rule="exact")
    >>> p.d_star, p.maximizing_set.tolist(), round(p.t_star, 6), p.at_horizon
    (3.0, [2, 4], 2.828427, False)
    """
    if rule not in PEAK_RULES:
        raise ValueError(f"rule must be 'exact' or 'plateau', got {rule!r}")
    times = _depths(t, "t")
    values = _numbers(d, "d")
    if values.shape != times.shape:
        raise ValueError(f"d must have one value per checkpoint: {times.shape}, got {values.shape}")
    end = times[-1] if horizon is None else _int(horizon, "horizon", minimum=0)
    if end < times[-1]:
        raise ValueError(f"horizon {end} is below the last checkpoint {times[-1]}")
    width: float | None = None
    errors = None
    if rule == "plateau":
        if z is None:
            raise ValueError("the plateau rule needs z (its width in standard errors)")
        width = float(z)
        if not math.isfinite(width) or width < 0:
            raise ValueError(f"z must be a finite number >= 0, got {z!r}")
        if se is None:
            raise ValueError("the plateau rule needs se, the standard error at each checkpoint")
        errors = _numbers(se, "se")
        if errors.shape != times.shape:
            raise ValueError(
                f"se must have one value per checkpoint: {times.shape}, got {errors.shape}"
            )
    k_values = None
    if k_at is not None:
        k_values = _whole(k_at, "k_at")
        if k_values.shape != times.shape:
            raise ValueError(
                f"k_at must have one value per checkpoint: {times.shape}, got {k_values.shape}"
            )
    given = k_values is not None
    positive = times > 0
    if not positive.any():
        return _invalid_peak(rule, width, int(end), "no positive checkpoint", given)
    tp, dp = times[positive], values[positive]
    if not np.all(np.isfinite(dp)) or np.any(dp < 0):
        i = int(np.flatnonzero(~np.isfinite(dp) | (dp < 0))[0])
        return _invalid_peak(
            rule,
            width,
            int(end),
            f"d({tp[i]}) = {dp[i]!r} is not a finite nonnegative number (pass the zero-extended "
            "trace, v1.4 §5.5)",
            given,
        )
    d_star = float(dp.max())
    first = int(np.flatnonzero(dp == d_star)[0])
    if errors is not None:
        ep = errors[positive]
        if not np.all(np.isfinite(ep)) or np.any(ep < 0):
            i = int(np.flatnonzero(~np.isfinite(ep) | (ep < 0))[0])
            return _invalid_peak(
                rule,
                width,
                int(end),
                f"se({tp[i]}) = {ep[i]!r} is not a finite nonnegative number",
                given,
            )
        assert width is not None
        keep = dp >= d_star - width * np.sqrt(ep**2 + ep[first] ** 2)
        keep[first] = True
    else:
        keep = dp == d_star
    chosen = tp[keep]
    k_set = None if k_values is None else k_values[positive][keep]
    if d_star == 0:
        return Peak(
            rule=rule,
            z=width,
            d_star=0.0,
            maximizing_set=chosen,
            t_star=math.nan,
            missing=True,
            reason="d* = 0: the trace is zero at every positive checkpoint",
            at_horizon=False,
            horizon=int(end),
            k_on_set=k_set,
            first_argmax=int(tp[first]),
        )
    if chosen.size == 1:
        t_star = float(chosen[0])  # exactly the checkpoint, not exp(ln t) rounded
    else:
        t_star = float(np.exp(np.log(chosen.astype(np.float64)).mean()))
    return Peak(
        rule=rule,
        z=width,
        d_star=d_star,
        maximizing_set=chosen,
        t_star=t_star,
        missing=False,
        reason="",
        at_horizon=bool(np.any(chosen == end)),
        horizon=int(end),
        k_on_set=k_set,
        first_argmax=int(tp[first]),
        t_star_is_checkpoint=bool(np.any(np.abs(times - t_star) <= 1e-12 * t_star)),
    )


def peak_sensitivity(
    t: Any,
    d: Any,
    se: Any,
    zs: Iterable[float],
    *,
    k_at: Any = None,
    horizon: int | None = None,
) -> tuple[Peak, ...]:
    """The plateau-rule peak at every ``z`` of ``zs`` (v1.4 §5.8: report the rule's sensitivity).

    One :class:`Peak` per ``z``, in the order given. The maximizing sets are nested: a larger
    ``z`` keeps every checkpoint a smaller one keeps.
    """
    widths = [float(z) for z in zs]
    if not widths:
        raise ValueError("zs must list at least one plateau width")
    return tuple(peak(t, d, se, rule="plateau", z=z, k_at=k_at, horizon=horizon) for z in widths)


@dataclass(frozen=True, eq=False)
class PeakBootstrap:
    """The bootstrap distribution of ``T*`` for a ``D(0)`` trace (resampled orderings per checkpoint).

    * ``estimate``: the :class:`Peak` of the original samples;
    * ``t_star``: float64 ``(B,)``, ``T*`` of each replicate (NaN where its peak is missing);
    * ``at_horizon``: bool ``(B,)``; ``missing_fraction``, ``horizon_fraction``: their means
      (the latter over all replicates);
    * ``replicates`` (``B``), ``seed``, ``z``.

    It describes the Monte Carlo uncertainty of ``T*`` given the trajectory's codes (the
    orderings), not the ensemble's: graphs and circuits are resampled by
    :func:`sparsegf2.study.stats.graph_bootstrap`.
    """

    estimate: Peak
    t_star: np.ndarray
    at_horizon: np.ndarray
    missing_fraction: float
    horizon_fraction: float
    replicates: int
    seed: Any
    z: float

    def __post_init__(self) -> None:
        _read_only(self)

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """The percentile interval of the replicates' ``T*`` that have a peak (linear interpolation)."""
        if not 0.0 < float(level) < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level!r}")
        kept = self.t_star[~np.isnan(self.t_star)]
        if kept.size == 0:
            return math.nan, math.nan
        tail = (1.0 - float(level)) / 2.0
        lo, hi = np.quantile(kept, [tail, 1.0 - tail])
        return float(lo), float(hi)


def _resample_indices(seed: Any, count: int, bound: int) -> np.ndarray:
    """``count`` indices in ``[0, bound)``: ``floor(u * bound)``, ``u`` the top 53 bits of PCG64's raw stream.

    numpy guarantees the raw output of ``PCG64(SeedSequence(seed))`` across versions; the map
    to ``[0, bound)`` has a bias below ``bound * 2**-53`` per draw.
    """
    raw = np.random.PCG64(np.random.SeedSequence(seed)).random_raw(count)
    unit = (raw >> np.uint64(11)).astype(np.float64) * (2.0**-53)
    return np.floor(unit * bound).astype(np.int64)


def bootstrap_peak(
    t: Any,
    first_failure: Any,
    num_system: int,
    *,
    z: float,
    replicates: int,
    seed: Any,
    exact_counts: Any = None,
    k_at: Any = None,
    horizon: int | None = None,
) -> PeakBootstrap:
    """Bootstrap ``T*`` of a ``D(0)`` trace by resampling each checkpoint's orderings (v1.4 §5.8).

    ``first_failure`` is ``(E, S)``: the first-failure sizes at each of the ``E`` checkpoints
    ``t`` of one trajectory; ``exact_counts`` (optional) ``(E, w)``. The estimate is the plateau
    peak (:func:`peak` with ``z``) of :func:`integrated_robustness` at the checkpoints. Each
    replicate ``b`` resamples, independently at every checkpoint ``e``, ``S`` orderings with
    replacement from that checkpoint's ``S`` (the estimates at different checkpoints are
    independent given the codes, so they are resampled independently; the exact counts are
    kept), recomputes ``D(0)`` and its error, and takes the plateau peak. The indices are
    ``floor(u * S)`` from :func:`numpy.random.PCG64` seeded by ``SeedSequence(seed)`` (raw
    stream, ``u`` its top 53 bits), drawn in the order replicate, checkpoint, sample.
    """
    times = _depths(t, "t")
    n = _int(num_system, "num_system", minimum=1)
    d = _first_failure(first_failure, n)
    if d.ndim != 2 or d.shape[0] != times.size:
        raise ValueError(
            f"first_failure must be (checkpoints, samples) = ({times.size}, S), got {d.shape}"
        )
    counts = None
    if exact_counts is not None:
        counts = _numbers(exact_counts, "exact_counts")
        if counts.ndim != 2 or counts.shape[0] != times.size:
            raise ValueError(
                f"exact_counts must be (checkpoints, w) = ({times.size}, w), got {counts.shape}"
            )
    b = _int(replicates, "replicates", minimum=1)
    samples = d.shape[1]
    base = integrated_robustness(d, n, exact_counts=counts)
    estimate = peak(times, base.value, base.se, rule="plateau", z=z, k_at=k_at, horizon=horizon)
    picks = _resample_indices(seed, b * times.size * samples, samples).reshape(
        b, times.size, samples
    )
    resampled = np.take_along_axis(np.broadcast_to(d, (b, *d.shape)), picks, axis=2)
    reps = integrated_robustness(
        resampled,
        n,
        exact_counts=None if counts is None else np.broadcast_to(counts, (b, *counts.shape)),
    )
    t_star = np.full(b, np.nan)
    at_horizon = np.zeros(b, dtype=bool)
    for i in range(b):
        p = peak(times, reps.value[i], reps.se[i], rule="plateau", z=z, horizon=horizon)
        t_star[i] = p.t_star
        at_horizon[i] = p.at_horizon
    return PeakBootstrap(
        estimate=estimate,
        t_star=t_star,
        at_horizon=at_horizon,
        missing_fraction=float(np.isnan(t_star).mean()),
        horizon_fraction=float(at_horizon.mean()),
        replicates=b,
        seed=seed,
        z=float(z),
    )


# ---------------------------------------------------------------------------------------------
# the registry (design §5)
# ---------------------------------------------------------------------------------------------


class _Required:
    """The default of a parameter that must be given."""

    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED: Any = _Required()
"""The default of a parameter without one (it must be given)."""

LEVELS = ("snapshot", "trajectory", "aggregate", "scaling")
"""Where an observable lives: one snapshot, one trajectory, an aggregate, a scaling fit (design §5)."""

INPUTS: Mapping[str, str] = MappingProxyType(
    {
        "k_trace": "k_t at every layer (code_dimension with checkpoints='every_layer'), or its change points",
        "joint_tableau": "the stored joint phase-free row space (tableau_snapshot)",
        "physical_code": "the physical code (a code_snapshot, or the joint tableau's physical_code)",
        "graph": "the trajectory's interaction graph",
        "probe:code_dimension": "the engine's code_dimension probe at the observable's evaluations",
        "probe:contiguous_distance": "the engine's contiguous_distance probe",
        "probe:erasure_first_failure": "the engine's erasure_first_failure probe (first-failure sizes)",
        "probe:uncorrectable_region_counts": "the engine's exhaustive uncorrectable-region counts",
        "probe:correlated_erasure": "the engine's correlated_erasure probe (failing fractions)",
        "trace": "a depth trace of another observable at the checkpoints (trace:<id>)",
    }
)
"""The input kinds an entry may list (``trace:<id>`` names the observable whose trace it reads)."""


@dataclass(frozen=True, eq=False)
class ObservableValue:
    """One computed value of a snapshot observable, in the store's columns (design §4 ``observations``).

    ``value`` (float; NaN where undefined), ``se`` (its standard error, 0 when exact),
    ``method`` (how it was computed) and ``data`` (everything else, plain Python values).
    """

    value: float
    se: float
    method: str
    data: Mapping[str, Any]


@dataclass(frozen=True, eq=False)
class SnapshotInput:
    """What a retrospective snapshot observable reads: one stored snapshot of one trajectory.

    ``num_system`` (``n``), ``t`` (the depth), ``key`` (the trajectory's
    :class:`~sparsegf2.study.identity.TrajectoryKey`, or its string form; needed by the sampled
    observables, whose seeds derive from it: :func:`region_seed`), and the stored data it has:
    the physical ``code``, the joint ``tableau`` (system ``0..n-1``, then the references) and the
    interaction graph ``edges``. :meth:`physical_code` falls back to the joint tableau's physical
    code.
    """

    num_system: int
    t: int
    key: TrajectoryKey | str | None = None
    code: StabilizerCode | None = None
    tableau: PhaseFreeTableau | None = None
    edges: Any = None

    def __post_init__(self) -> None:
        n = _int(self.num_system, "num_system", minimum=1)
        _int(self.t, "t", minimum=0)
        if self.key is not None:
            key = _trajectory(self.key)
            if key.n != n:
                raise ValueError(f"the key is a trajectory of n = {key.n}, not num_system = {n}")
            object.__setattr__(self, "key", key)

    def trajectory(self) -> TrajectoryKey:
        if self.key is None:
            raise ValueError(
                "this observable draws regions from a seed of the trajectory's identity: give key="
            )
        assert isinstance(self.key, TrajectoryKey)
        return self.key

    def physical_code(self) -> StabilizerCode:
        if self.code is not None:
            return self.code
        if self.tableau is not None:
            return self.tableau.physical_code(self.num_system)
        raise ValueError("this observable needs the physical code: give code= or tableau=")

    def joint(self) -> PhaseFreeTableau:
        if self.tableau is None:
            raise ValueError("this observable needs the joint tableau: give tableau=")
        return self.tableau

    def graph(self) -> np.ndarray:
        if self.edges is None:
            raise ValueError(
                "this observable needs the trajectory's interaction graph: give edges="
            )
        return graphs.from_edges(self.edges, self.num_system)


@dataclass(frozen=True, eq=False)
class ObservableDef:
    """One entry of the observable catalog (design §5).

    * ``id``, ``version``: the name and the implementation version; with the canonical
      parameters (:func:`params_key`) they key every stored value, so a changed implementation
      needs a new version;
    * ``level``: one of :data:`LEVELS`; ``inputs``: kinds of :data:`INPUTS` (``trace:<id>`` for a
      trace of another observable);
    * ``params``: each parameter's default (:data:`REQUIRED` when it must be given), read-only;
    * ``definition``: the formula and its v1.4 section, in words;
    * ``function``: the pure function of this module that computes it;
    * ``compute``: for an observable computed later from a stored snapshot,
      ``compute(snapshot, params) -> ObservableValue`` with :class:`SnapshotInput` and the
      resolved parameters (None for the observables computed from engine probes or traces).
    """

    id: str
    version: str
    level: str
    inputs: tuple[str, ...]
    params: Mapping[str, Any]
    definition: str
    function: Callable[..., Any]
    compute: Callable[[SnapshotInput, Mapping[str, Any]], ObservableValue] | None = None

    def __post_init__(self) -> None:
        _label(self.id, "id")
        _label(self.version, "version")
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {list(LEVELS)}, got {self.level!r}")
        for kind in self.inputs:
            base = kind.split(":", 1)[0] if kind.startswith("trace:") else kind
            if base not in INPUTS or kind == "trace":
                raise ValueError(
                    f"input {kind!r} is not one of {sorted(INPUTS)} (use trace:<id> for a trace)"
                )
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    @property
    def retrospective(self) -> bool:
        """Whether it is computed later from a stored snapshot (it has ``compute``)."""
        return self.compute is not None

    def row(self) -> dict[str, Any]:
        """The catalog row: plain Python values."""
        return {
            "id": self.id,
            "version": self.version,
            "level": self.level,
            "inputs": list(self.inputs),
            "params": {k: ("required" if v is REQUIRED else v) for k, v in self.params.items()},
            "definition": self.definition,
            "function": self.function.__name__,
            "retrospective": self.retrospective,
        }

    def _same(self, other: ObservableDef) -> bool:
        return (
            self.level == other.level
            and self.inputs == other.inputs
            and dict(self.params) == dict(other.params)
            and self.definition == other.definition
            and self.function is other.function
            and self.compute is other.compute
        )


_REGISTRY: dict[tuple[str, str], ObservableDef] = {}


def register(entry: ObservableDef) -> ObservableDef:
    """Add an entry to the catalog; registering the same entry again changes nothing.

    A different entry under an existing ``(id, version)`` is refused: a changed definition or
    implementation needs a new version, so that values computed under the old one are never
    mistaken for the new one's.
    """
    if not isinstance(entry, ObservableDef):
        raise TypeError(f"register takes an ObservableDef, got {type(entry).__name__}")
    key = (entry.id, entry.version)
    old = _REGISTRY.get(key)
    if old is not None:
        if old._same(entry):
            return old
        raise ValueError(
            f"observable {entry.id!r} version {entry.version!r} is already registered with "
            "another definition: give the changed observable a new version"
        )
    _REGISTRY[key] = entry
    return entry


def get(observable_id: str, version: str | None = None) -> ObservableDef:
    """The catalog entry ``observable_id`` (its latest registered version unless ``version`` is given)."""
    found = [e for (i, _), e in _REGISTRY.items() if i == observable_id]
    if not found:
        raise KeyError(
            f"no observable {observable_id!r} in the catalog; it has {sorted({i for i, _ in _REGISTRY})}"
        )
    if version is None:
        return found[-1]
    for e in found:
        if e.version == version:
            return e
    raise KeyError(
        f"observable {observable_id!r} has no version {version!r}; it has {[e.version for e in found]}"
    )


def catalog() -> tuple[dict[str, Any], ...]:
    """Every entry as a plain row (:meth:`ObservableDef.row`), sorted by id and version: the table for docs and the CLI."""
    return tuple(_REGISTRY[key].row() for key in sorted(_REGISTRY))


def catalog_markdown() -> str:
    """:func:`catalog` as a Markdown table (id, version, level, inputs, parameters, definition)."""
    lines = [
        "| id | version | level | inputs | parameters | definition |",
        "|---|---|---|---|---|---|",
    ]
    for row in catalog():
        params = ", ".join(f"{k}={v!r}" for k, v in row["params"].items()) or "none"
        lines.append(
            f"| `{row['id']}` | {row['version']} | {row['level']}"
            + (" (retrospective)" if row["retrospective"] else "")
            + f" | {', '.join(row['inputs'])} | {params} | {row['definition']} |"
        )
    return "\n".join(lines)


def resolve_params(
    observable_id: str, params: Mapping[str, Any] | None = None, *, version: str | None = None
) -> dict[str, Any]:
    """The parameters of an observable with its defaults filled in; unknown or missing ones refused."""
    entry = get(observable_id, version)
    given = dict(params or {})
    unknown = sorted(set(given) - set(entry.params))
    if unknown:
        raise ValueError(
            f"unknown parameter(s) {unknown} for observable {entry.id!r}; its parameters are "
            f"{sorted(entry.params)}"
        )
    out = {k: given.get(k, default) for k, default in entry.params.items()}
    missing = sorted(k for k, v in out.items() if v is REQUIRED)
    if missing:
        raise ValueError(f"observable {entry.id!r} needs the parameter(s) {missing}")
    return out


def params_key(
    observable_id: str, params: Mapping[str, Any] | None = None, *, version: str | None = None
) -> str:
    """The canonical JSON of the resolved parameters: the ``params`` part of a stored value's key.

    Sorted keys, no spaces, tuples as lists (:func:`sparsegf2.study.identity.canonical_json`),
    so equal parameters give equal text whatever their order or the defaults left out.
    """
    return canonical_json(resolve_params(observable_id, params, version=version))


def compute(
    observable_id: str,
    snapshot: SnapshotInput,
    params: Mapping[str, Any] | None = None,
    *,
    version: str | None = None,
) -> ObservableValue:
    """Compute a retrospective snapshot observable from a stored snapshot (design §4 ``runner.compute``).

    Only entries with a ``compute`` (``ObservableDef.retrospective``) qualify; the others come
    from engine probes or traces and are computed by their ``function`` from those arrays.
    """
    entry = get(observable_id, version)
    if entry.compute is None:
        raise ValueError(
            f"observable {entry.id!r} is computed from {list(entry.inputs)} by "
            f"{entry.function.__name__}, not from a stored snapshot"
        )
    if not isinstance(snapshot, SnapshotInput):
        raise TypeError(f"snapshot must be a SnapshotInput, got {type(snapshot).__name__}")
    return entry.compute(snapshot, resolve_params(entry.id, params, version=entry.version))


# -- the retrospective computations -------------------------------------------------------------


def _compute_k(snapshot: SnapshotInput, params: Mapping[str, Any]) -> ObservableValue:
    code = snapshot.physical_code()
    k = code.num_logical
    return ObservableValue(
        float(k),
        0.0,
        "stored-code",
        {"num_system": snapshot.num_system, "rank": code.num_stabilizers},
    )


def _compute_rate(snapshot: SnapshotInput, params: Mapping[str, Any]) -> ObservableValue:
    k = snapshot.physical_code().num_logical
    return ObservableValue(k / snapshot.num_system, 0.0, "stored-code", {"k": k})


def _compute_rank(snapshot: SnapshotInput, params: Mapping[str, Any]) -> ObservableValue:
    k = snapshot.physical_code().num_logical
    return ObservableValue(float(snapshot.num_system - k), 0.0, "stored-code", {"k": k})


def _compute_d_cyc(snapshot: SnapshotInput, params: Mapping[str, Any]) -> ObservableValue:
    code = snapshot.physical_code()
    raw = cyclic_distance_from_code(code)
    has = code.num_logical > 0
    return ObservableValue(
        raw,
        0.0,
        "stored-code",
        {"has_code": has, "zero_extended": int(raw) if has else 0, "k": code.num_logical},
    )


def _compute_d_conn(snapshot: SnapshotInput, params: Mapping[str, Any]) -> ObservableValue:
    code = snapshot.physical_code()
    value = connected_distance(code, snapshot.graph(), max_n=params["max_n"])
    return ObservableValue(
        value, 0.0, "enumeration", {"has_code": code.num_logical > 0, "k": code.num_logical}
    )


def _compute_half_cut(snapshot: SnapshotInput, params: Mapping[str, Any]) -> ObservableValue:
    value = half_cut_entropy(snapshot.joint(), snapshot.num_system)
    return ObservableValue(float(value), 0.0, "joint-tableau", {"qubits": snapshot.num_system // 2})


def _compute_leakage(snapshot: SnapshotInput, params: Mapping[str, Any]) -> ObservableValue:
    code = snapshot.physical_code()
    route = params["route"]
    if route == "enumerate":
        value = float(
            mean_leakage_exact(
                code, snapshot.graph(), params["m"], [params["J"]], budget=params["budget"]
            )[0]
        )
        return ObservableValue(value, 0.0, "enumerate", {"k": code.num_logical})
    if route != "sample":
        raise ValueError(f"route must be 'sample' or 'enumerate', got {route!r}")
    est = mean_leakage(
        code,
        snapshot.graph(),
        params["m"],
        params["J"],
        key=snapshot.trajectory(),
        t=snapshot.t,
        samples=params["samples"],
        burn_in=params["burn_in"],
        thin=params["thin"],
        tempering=params["tempering"],
        method=params["method"],
        ladder=params["ladder"],
        max_step=params["max_step"],
        observable_id="mean_leakage",
        version=get("mean_leakage").version,
    )
    return ObservableValue(
        est.value,
        est.se,
        est.method,
        {
            "ess": est.ess,
            "samples": est.samples,
            "seed": None if est.seed is None else list(est.seed),
            "warnings": list(est.warnings),
            "burn_in": est.burn_in,
            "thin": est.thin,
            "tempering": est.tempering,
            "k": est.num_logical,
        },
    )


# -- the built-in entries -------------------------------------------------------------------------

_UNIFORM = {"samples": 200, "exact_budget": 5000}
_CORRELATED = {
    "name": None,
    "couplings": REQUIRED,
    "sizes": None,
    "samples": 100,
    "thin": 1,
    "geometry": "interaction",
    "method": "auto",
    "burn_in": None,
    "tempering": True,
    "ladder": "auto",
    "max_step": None,
}
_PLATEAU = {"rule": "plateau", "z": 1.0, "sensitivity_z": (0.0, 1.0, 2.0), "bootstrap": 200}

for _entry in (
    ObservableDef(
        "k",
        "1",
        "snapshot",
        ("k_trace", "physical_code"),
        {},
        "k_t = n - r_t = S(R), the logical qubits after t layers (v1.4 §5.2)",
        code_parameters,
        _compute_k,
    ),
    ObservableDef(
        "code_rate",
        "1",
        "snapshot",
        ("k_trace", "physical_code"),
        {},
        "R_t = k_t / n, the code rate (v1.4 §5.2)",
        code_parameters,
        _compute_rate,
    ),
    ObservableDef(
        "stabilizer_rank",
        "1",
        "snapshot",
        ("k_trace", "physical_code"),
        {},
        "r_t = dim W_t^phys = n - k_t (v1.4 §5.2)",
        code_parameters,
        _compute_rank,
    ),
    ObservableDef(
        "k_trace",
        "1",
        "trajectory",
        ("k_trace",),
        {},
        "the change points (t_j, k_{t_j}) of k_t, t = 0..T, and the horizon: lossless (v1.4 §5.2, §5.3)",
        k_change_points,
    ),
    ObservableDef(
        "first_passage",
        "1",
        "trajectory",
        ("k_trace",),
        {},
        "tau_ell = min{t : k_t <= ell}, ell = 0..n, right-censored at T_max; checkpoint bounds "
        "tau_chk = min{t in T(n, beta) : k_t <= ell} (v1.4 §5.2)",
        first_passage,
    ),
    ObservableDef(
        "d_cyc",
        "1",
        "snapshot",
        ("probe:contiguous_distance", "probe:code_dimension", "physical_code"),
        {},
        "first failing cyclic interval of the ring order: raw (NaN at k = 0), has_code, "
        "zero-extended (v1.4 §5.5)",
        cyclic_distance,
        _compute_d_cyc,
    ),
    ObservableDef(
        "uniform_failure",
        "1",
        "snapshot",
        ("probe:erasure_first_failure", "probe:uncorrectable_region_counts"),
        _UNIFORM,
        "P_C(m) = C(n, m)^-1 sum_{|A|=m} f(A), m = 0..n: exact counts for m <= w, first-failure "
        "orderings above (v1.4 §5.7, J = 0)",
        uniform_failure,
    ),
    ObservableDef(
        "D",
        "1",
        "snapshot",
        ("probe:erasure_first_failure", "probe:uncorrectable_region_counts"),
        _UNIFORM,
        "D(0) = sum_{m=0}^{n-1} (1 - P_C(m)) = E_pi[M_pi], 0 at k = 0, with its standard error "
        "(v1.4 §5.7)",
        integrated_robustness,
    ),
    ObservableDef(
        "Phi",
        "1",
        "snapshot",
        ("probe:erasure_first_failure", "probe:uncorrectable_region_counts"),
        {**_UNIFORM, "e": REQUIRED},
        "Phi_C(e) = sum_m C(n, m) e^m (1 - e)^(n - m) P_C(m), independent erasure with probability "
        "e (not the rate p), with its standard error (v1.4 §5.7)",
        uniform_channel_failure,
    ),
    ObservableDef(
        "correlated_failure",
        "1",
        "snapshot",
        ("probe:correlated_erasure", "probe:code_dimension"),
        _CORRELATED,
        "P_{C,G,J}(m) = sum_{|A|=m} Q_{G,J}(A|m) f(A), Q ∝ exp(-J b_G(A)), with binomial errors "
        "per size; interpolated sizes flagged (v1.4 §5.7)",
        correlated_failure,
    ),
    ObservableDef(
        "D_J",
        "1",
        "snapshot",
        ("probe:correlated_erasure", "probe:code_dimension"),
        _CORRELATED,
        "D(J) = sum_{m=0}^{n-1} (1 - P_{C,G,J}(m)), 0 at k = 0; incomplete when a size was "
        "interpolated (v1.4 §5.7)",
        correlated_failure,
    ),
    ObservableDef(
        "Phi_J",
        "1",
        "snapshot",
        ("probe:correlated_erasure", "probe:code_dimension"),
        {**_CORRELATED, "e": REQUIRED},
        "Phi_{C,G,J}(e) = sum_m C(n, m) e^m (1 - e)^(n - m) P_{C,G,J}(m), the binomially mixed "
        "correlated channel, with the binomial mass of interpolated sizes (v1.4 §5.7)",
        correlated_channel_failure,
    ),
    ObservableDef(
        "mean_leakage",
        MEAN_LEAKAGE_VERSION,
        "snapshot",
        ("physical_code", "graph"),
        {
            "m": REQUIRED,
            "J": REQUIRED,
            "samples": 100,
            "burn_in": None,
            "thin": 1,
            "tempering": True,
            "method": "auto",
            "ladder": "auto",
            "max_step": None,
            "route": "sample",
            "budget": None,
        },
        "l(m, J) = E_{A ~ Q_{G,J}(.|m)}[I(A:R)], regions seeded by (trajectory key, t, m, J, id, "
        "version); "
        "route 'enumerate' for every region (v1.4 §5.7)",
        mean_leakage,
        _compute_leakage,
    ),
    ObservableDef(
        "d_conn",
        "1",
        "snapshot",
        ("physical_code", "graph"),
        {"max_n": 20},
        "min{|A| : G[A] connected, f(A) = 1}, 0 at k = 0, inf if none; exact enumeration, "
        "refused above max_n (v1.4 §5.6)",
        connected_distance,
        _compute_d_conn,
    ),
    ObservableDef(
        "half_cut_entropy",
        "1",
        "snapshot",
        ("joint_tableau",),
        {},
        "S(A) of the system qubits A = {0, ..., floor(n/2) - 1} from the joint state (v1.4 §5.1, "
        "§5.4)",
        half_cut_entropy,
        _compute_half_cut,
    ),
    ObservableDef(
        "peak_d_cyc",
        "1",
        "trajectory",
        ("trace:d_cyc", "k_trace"),
        {"rule": "exact"},
        "d* = max_{t > 0} d_cyc(t) (zero-extended), A = exact maximizers, T* = exp(mean_A ln t), "
        "missing at d* = 0, horizon flag, k_t on A (v1.4 §5.8)",
        peak,
    ),
    ObservableDef(
        "peak_D",
        "1",
        "trajectory",
        ("trace:D", "k_trace"),
        _PLATEAU,
        "the plateau peak of D(0)(t): A = {t : D(t) >= D* - z sqrt(se(t)^2 + se(t*)^2)}, "
        "T* = exp(mean_A ln t), sensitivity over z, bootstrap of T* (v1.4 §5.8)",
        peak,
    ),
    ObservableDef(
        "peak_D_J",
        "1",
        "trajectory",
        ("trace:D_J", "k_trace"),
        {**_PLATEAU, "J": REQUIRED},
        "the plateau peak of D(J)(t) at one coupling J (v1.4 §5.8)",
        peak,
    ),
    ObservableDef(
        "peak_d_conn",
        "1",
        "trajectory",
        ("trace:d_conn", "k_trace"),
        {"rule": "exact"},
        "the exact-tie peak of d_conn(t) where finite (v1.4 §5.8)",
        peak,
    ),
):
    register(_entry)
del _entry
