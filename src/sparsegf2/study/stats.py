"""Nested ensemble statistics of the Watts–Strogatz study (specification v1.4 §§5.8, 5.9, 6.3, 6.4).

The study's samples are nested (v1.4 §1, §6.3): graphs ``g`` are independent draws from the
Watts–Strogatz law, gate sequences ``c`` are drawn conditionally on their graph, and measurement
replicas ``r`` conditionally on their gate sequence (the native engine has ``r = 0`` only, design
§9 D1, but every function here accepts any replica count). Every function takes plain arrays with
one row per trajectory: its labels ``g, c, r`` and the observable's value(s). Rows may come in
any order and in any number of chunks: the results depend only on the set of rows.

* :func:`nested_mean` is the estimator of v1.4 §6.3,

  ``f_gc = mean_r F_gcr``, ``f_g = mean_c f_gc``, ``mu = mean_g f_g``,

  which gives every graph the same weight whatever its numbers of circuits and replicas, with the
  graph-cluster standard error of v1.4 §6.4 (:func:`graph_cluster_se`),

  ``SE = sqrt( sum_g (f_g - mu)^2 / (G (G - 1)) )``, defined for ``G > 1``.

* :func:`paired_difference` compares two rates on the same graphs: the difference is formed within
  each graph first, then its cluster error is taken (v1.4 §6.4), so the positive correlation of the
  coupled rates (shared graphs, gates and mask uniforms, v1.4 §7.3) shrinks the error instead of
  being ignored.
* Censored first-passage times (v1.4 §5.2, §5.8, §5.9): :func:`weighted_kaplan_meier` and
  :func:`nested_kaplan_meier` (graph-equal weights) with a graph-cluster variance, the median, the
  restricted mean to a horizon, and :func:`censoring_summary`. Nothing here returns a plain mean
  of the observed times when some times are censored.
* :func:`graph_bootstrap` draws graph labels with replacement from a stable stream; the same draws
  are applied to every rate and observable, so pairing survives; :func:`percentile_interval` reads
  the replicates.
* :func:`peak_time_summaries` keeps the three peak-time statistics of v1.4 §5.8 apart: the
  arithmetic mean ``E[T*]``, the geometric mean ``exp(E[ln T*])`` and ``argmax_t E[d(t)]``.

``docs/study/06-statistics.md`` has the definitions, the proofs and worked examples.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "CensoringSummary",
    "KM_VARIANCE",
    "GraphMean",
    "NestedMean",
    "PeakTimeSummary",
    "SurvivalCurve",
    "censoring_summary",
    "graph_bootstrap",
    "graph_cluster_se",
    "graph_equal_weights",
    "mean_over_graphs",
    "nested_kaplan_meier",
    "nested_mean",
    "paired_difference",
    "peak_time_summaries",
    "percentile_interval",
    "student_t_quantile",
    "weighted_kaplan_meier",
]

NanPolicy = Literal["raise", "omit"]
HorizonPolicy = Literal["refuse", "exclude", "include"]

#: The variance estimate of :class:`SurvivalCurve`, recorded with every curve.
KM_VARIANCE = (
    "graph-cluster delta method: the influence of each cluster's weights on the estimate, "
    "summed within clusters, variance = G/(G-1) * sum of squares"
)

_EXACT_WEIGHT_LIMIT = 1 << 62  # integer weights whose total stays below this are summed exactly


# ---------------------------------------------------------------------------------------------
# input checks
# ---------------------------------------------------------------------------------------------


def _labels(name: str, labels: ArrayLike) -> np.ndarray:
    a = np.asarray(labels)
    if a.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array of labels, one per trajectory")
    if a.dtype == np.bool_ or not (
        np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating)
    ):
        raise ValueError(f"{name} must hold integer labels, not {a.dtype}")
    if np.issubdtype(a.dtype, np.floating) and not np.all(np.isfinite(a) & (a == np.floor(a))):
        raise ValueError(f"{name} must hold integer labels")
    if a.size and a.min() < 0:
        raise ValueError(f"{name} labels must be non-negative (v1.4 §1: g, c, r >= 0)")
    return a.astype(np.int64)


def _rows(g: ArrayLike, c: ArrayLike, r: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gg, cc, rr = _labels("g", g), _labels("c", c), _labels("r", r)
    if not gg.size == cc.size == rr.size:
        raise ValueError(
            f"g, c and r must have one entry per trajectory; got {gg.size}, {cc.size} and {rr.size}"
        )
    if gg.size == 0:
        raise ValueError("no trajectories: g, c and r are empty")
    return gg, cc, rr


def _values(values: ArrayLike, rows: int, name: str = "values") -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 0 or v.shape[0] != rows:
        raise ValueError(
            f"{name} must have one row per trajectory (first axis of length {rows}); "
            f"got shape {v.shape}"
        )
    if np.isinf(v).any():
        raise ValueError(f"{name} must be finite or NaN (an undefined value); found an infinity")
    return v


def _nan_policy(policy: str) -> bool:
    if policy not in ("raise", "omit"):
        raise ValueError(f"nan_policy must be 'raise' or 'omit', not {policy!r}")
    return policy == "omit"


def _check_nan(v: np.ndarray, omit: bool, name: str = "values") -> None:
    bad = int(np.isnan(v).sum())
    if bad and not omit:
        raise ValueError(
            f"{bad} of the {name} are NaN (undefined). Pass nan_policy='omit' to average only the "
            "defined values; that changes the target to the equal-graph mean of the conditional "
            "means over the graphs with a defined value (see NestedMean)"
        )


def _level(level: float) -> float:
    if not isinstance(level, (int, float)) or isinstance(level, bool) or not 0 < level < 1:
        raise ValueError(f"level must be a probability strictly between 0 and 1, not {level!r}")
    return float(level)


def _times(times: ArrayLike, name: str = "times") -> np.ndarray:
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array, one entry per trajectory")
    if not np.all(np.isfinite(t) & (t >= 0)):
        raise ValueError(
            f"{name} must be finite and non-negative: a censored trajectory needs its censoring "
            "depth (T_max), not a sentinel such as -1 or infinity (v1.4 §5.2)"
        )
    return t


def _flags(flags: ArrayLike, rows: int, name: str) -> np.ndarray:
    a = np.asarray(flags)
    if a.shape != (rows,):
        raise ValueError(f"{name} must have one entry per trajectory ({rows}); got {a.shape}")
    if a.dtype != np.bool_:
        if not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating)) or (
            a.size and not np.all((a == 0) | (a == 1))
        ):
            raise ValueError(f"{name} must hold booleans (or 0 and 1)")
        a = a.astype(bool)
    return a


def _divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``num / den`` with NaN where ``den == 0`` (and no warning)."""
    num = np.asarray(num, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=np.asarray(den) > 0)
    return out


# ---------------------------------------------------------------------------------------------
# JSON-friendly results
# ---------------------------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """``value`` as plain Python: lists, floats (non-finite as None), ints, bools, dicts."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _plain(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if not f.name.startswith("_")
        }
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# ---------------------------------------------------------------------------------------------
# stable random streams (graph bootstrap, multi-start fits)
# ---------------------------------------------------------------------------------------------


def _seed_sequence(seed: Any) -> np.random.SeedSequence:
    def entry(s: Any) -> int:
        if isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or s < 0:
            raise ValueError(
                f"seed must be a non-negative integer or a sequence of them, not {seed!r}"
            )
        return int(s)

    if isinstance(seed, (list, tuple, np.ndarray)):
        items = [entry(s) for s in np.asarray(seed, dtype=object).reshape(-1)]
        if not items:
            raise ValueError("seed must not be an empty sequence")
        return np.random.SeedSequence(items)
    return np.random.SeedSequence(entry(seed))


def _bounded_integers(seed: Any, count: int, bound: int) -> np.ndarray:
    """``count`` integers uniform on ``0..bound-1`` from the raw PCG64 stream of ``seed``.

    numpy keeps the raw output of its bit generators and ``SeedSequence`` stable across
    versions (``Generator`` methods may change), so the draws are reproducible. A raw word
    ``x`` maps to ``x mod bound`` after rejecting the ``2^64 mod bound`` largest words, which
    makes every residue equally likely.
    """
    bitgen = np.random.PCG64(_seed_sequence(seed))
    raw = bitgen.random_raw(count)
    rem = (1 << 64) % bound
    if rem:
        limit = np.uint64((1 << 64) - rem)
        bad = np.flatnonzero(raw >= limit)
        for i in bad:  # probability below bound / 2^64 per word
            x = bitgen.random_raw()
            while x >= limit:
                x = bitgen.random_raw()
            raw[i] = x
    return (raw % np.uint64(bound)).astype(np.int64)


def _uniforms(seed: Any, count: int) -> np.ndarray:
    """``count`` doubles uniform on [0, 1) (53 random bits each) from the raw PCG64 stream."""
    raw = np.random.PCG64(_seed_sequence(seed)).random_raw(count)
    return (raw >> np.uint64(11)).astype(np.float64) * 2.0**-53


# ---------------------------------------------------------------------------------------------
# Student's t quantile (no SciPy)
# ---------------------------------------------------------------------------------------------


def _t_two_sided(theta: float, dof: int) -> float:
    """Student's ``A(theta) = P(|X_nu| <= sqrt(nu) tan(theta))``, ``nu = dof`` (doc 06 §4.2)."""
    s, c = math.sin(theta), math.cos(theta)
    c2 = c * c
    if dof % 2:
        total, term = 0.0, 1.0
        if dof > 1:
            total = 1.0
            for k in range(1, (dof - 3) // 2 + 1):
                term *= c2 * (2 * k) / (2 * k + 1)
                total += term
        return (2.0 / math.pi) * (theta + s * c * total) if dof > 1 else 2.0 * theta / math.pi
    total, term = 1.0, 1.0
    for k in range(1, (dof - 2) // 2 + 1):
        term *= c2 * (2 * k - 1) / (2 * k)
        total += term
    return s * total


def student_t_quantile(probability: float, dof: int) -> float:
    """The quantile ``t_q`` of Student's t distribution with ``dof`` (an integer >= 1) degrees of freedom.

    With ``X_nu`` a Student variable on ``nu = dof`` degrees of freedom, ``s >= 0`` and
    ``theta = arctan(s / sqrt(nu))``, the two-sided probability ``A(theta) = P(|X_nu| <= s)`` is an
    exact finite sum (Abramowitz and Stegun, 26.7.3 and 26.7.4; doc 06 §4.2):

    * ``nu = 1``: ``A = 2 theta / pi``;
    * ``nu`` odd: ``A = (2/pi) [theta + sin(theta) cos(theta) sum_{k=0}^{(nu-3)/2} alpha_k
      cos^(2k) theta]``, ``alpha_0 = 1``, ``alpha_k = alpha_{k-1} 2k / (2k + 1)``;
    * ``nu`` even: ``A = sin(theta) sum_{k=0}^{(nu-2)/2} kappa_k cos^(2k) theta``, ``kappa_0 = 1``,
      ``kappa_k = kappa_{k-1} (2k - 1) / (2k)``.

    ``A`` increases from 0 to 1 on ``theta in [0, pi/2)``; the quantile for ``q > 1/2`` is
    ``t_{q,nu} = sqrt(nu) tan(theta*)`` with ``A(theta*) = 2q - 1``, found by bisection to the last
    bit, and ``t_{q,nu} = -t_{1-q,nu}``. Used for the intervals
    ``mean +- t_{(1+level)/2, G'-1} SE`` of :meth:`GraphMean.interval`. The arguments are checked
    on every call; only the computation is cached (by ``(q, int(dof))``), so a refused ``dof``
    such as ``2.0`` or ``True`` is refused whatever was computed before.
    """
    if isinstance(dof, bool) or not isinstance(dof, (int, np.integer)) or dof < 1:
        raise ValueError(f"dof must be an integer >= 1, not {dof!r}")
    q = float(probability)
    if not 0 < q < 1:
        raise ValueError(f"probability must be strictly between 0 and 1, not {probability!r}")
    return _t_quantile(q, int(dof))


@functools.lru_cache(maxsize=256)
def _t_quantile(q: float, dof: int) -> float:
    """:func:`student_t_quantile` for a checked ``0 < q < 1`` and an ``int`` ``dof >= 1``."""
    if q == 0.5:
        return 0.0
    if q < 0.5:
        return -_t_quantile(1.0 - q, dof)
    target = 2.0 * q - 1.0
    lo, hi = 0.0, math.pi / 2
    while True:  # halves [lo, hi] until no double lies strictly inside (at most ~60 steps)
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if _t_two_sided(mid, int(dof)) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(dof) * math.tan(0.5 * (lo + hi))


# ---------------------------------------------------------------------------------------------
# the nested (equal graph weight) estimator
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Hierarchy:
    """The rows of a sample in canonical order (by ``g``, then ``c``, then ``r``) and its levels."""

    order: np.ndarray  # row indices in canonical order
    circuit_start: np.ndarray  # (M,) first canonical row of each (g, c)
    graph_start: np.ndarray  # (G,) first circuit of each graph
    graphs: np.ndarray  # (G,) graph labels, increasing
    circuit_keys: np.ndarray  # (M, 2) the (g, c) labels
    replicas: np.ndarray  # (M,) rows per (g, c): R_gc
    circuits_per_graph: np.ndarray  # (G,) C_g
    trajectories_per_graph: np.ndarray  # (G,)


def _hierarchy(g: np.ndarray, c: np.ndarray, r: np.ndarray) -> _Hierarchy:
    order = np.lexsort((r, c, g))
    gs, cs, rs = g[order], c[order], r[order]
    same = (gs[1:] == gs[:-1]) & (cs[1:] == cs[:-1])
    dup = same & (rs[1:] == rs[:-1])
    if dup.any():
        k = int(np.flatnonzero(dup)[0]) + 1
        raise ValueError(
            f"trajectory (g, c, r) = ({gs[k]}, {cs[k]}, {rs[k]}) appears more than once; "
            "each trajectory must appear exactly once"
        )
    n = gs.size
    circuit_start = np.flatnonzero(np.concatenate(([True], ~same)))
    circuit_graph = gs[circuit_start]
    graph_start = np.flatnonzero(np.concatenate(([True], circuit_graph[1:] != circuit_graph[:-1])))
    replicas = np.diff(np.append(circuit_start, n))
    circuits_per_graph = np.diff(np.append(graph_start, circuit_start.size))
    return _Hierarchy(
        order=order,
        circuit_start=circuit_start,
        graph_start=graph_start,
        graphs=circuit_graph[graph_start],
        circuit_keys=np.stack((circuit_graph, cs[circuit_start]), axis=1),
        replicas=replicas,
        circuits_per_graph=circuits_per_graph,
        trajectories_per_graph=np.add.reduceat(replicas, graph_start),
    )


def _graph_means(h: _Hierarchy, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-graph nested means ``(G, K)`` of canonical-order rows ``flat (N, K)``, omitting NaN.

    Returns the graph means (NaN where a graph has no defined value) and the number of defined
    rows per column.
    """
    defined = ~np.isnan(flat)
    sums = np.add.reduceat(np.where(defined, flat, 0.0), h.circuit_start, axis=0)
    counts = np.add.reduceat(defined.astype(np.int64), h.circuit_start, axis=0)
    circuit_means = _divide(sums, counts)
    has = counts > 0
    graph_sums = np.add.reduceat(np.where(has, circuit_means, 0.0), h.graph_start, axis=0)
    graph_counts = np.add.reduceat(has.astype(np.int64), h.graph_start, axis=0)
    return _divide(graph_sums, graph_counts), defined.sum(axis=0)


def _summary(graph_means: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, graph-cluster SE and count of the defined graph means along axis 0."""
    defined = ~np.isnan(graph_means)
    count = defined.sum(axis=0)
    mean = _divide(np.where(defined, graph_means, 0.0).sum(axis=0), count)
    dev = np.where(defined, graph_means - mean, 0.0)
    var = _divide((dev * dev).sum(axis=0), count * (count - 1))
    return mean, np.sqrt(var), count


def graph_cluster_se(graph_means: ArrayLike) -> np.ndarray:
    """The graph-cluster standard error of v1.4 §6.4 from per-graph means (axis 0 = graphs).

    ``SE = sqrt( sum_g (f_g - mu)^2 / (G (G - 1)) )`` with ``mu`` the mean of the ``G`` defined
    (non-NaN) graph means; NaN where fewer than two graphs are defined. For independent graphs
    ``E[SE^2] = Var(mu)`` exactly, whatever the variances of the individual ``f_g``
    (``docs/study/06-statistics.md`` §4.1, Proposition 3).
    """
    gm = np.asarray(graph_means, dtype=np.float64)
    if gm.ndim == 0:
        raise ValueError("graph_means must have one entry per graph along axis 0")
    return _summary(gm)[1]


@dataclass(frozen=True, eq=False)
class GraphMean:
    """An equal-weight mean over independent graphs and its graph-cluster standard error.

    ``mean`` is ``(1/G') sum_g f_g`` over the ``G'`` graphs whose value is defined (not NaN),
    ``se`` the graph-cluster standard error (v1.4 §6.4; NaN when ``G' < 2``: with one graph the
    graph-ensemble uncertainty cannot be inferred). The arrays ``mean``, ``se`` and
    ``contributing_graphs`` have the shape of one value (0-d for a scalar observable;
    ``float(result.mean)`` gives the number), ``graph_means`` has one more leading axis, aligned
    with ``graphs`` (increasing labels).
    """

    mean: np.ndarray
    se: np.ndarray
    graphs: np.ndarray
    graph_means: np.ndarray
    contributing_graphs: np.ndarray

    @property
    def num_graphs(self) -> int:
        return int(self.graphs.size)

    def interval(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """``mean -+ t_{(1+level)/2, G'-1} se``: Student's t on ``G' - 1`` degrees of freedom.

        Exact coverage needs normal graph means of equal variance; the tests measure 0.948 at
        level 0.95 for ``G' = 20`` normal graph means of unequal variances. NaN where
        ``G' < 2``.
        """
        q = (1.0 + _level(level)) / 2.0
        count = self.contributing_graphs.reshape(-1)
        flat = np.full(count.shape, np.nan)
        se = self.se.reshape(-1)
        for dof in np.unique(count[count >= 2] - 1):
            sel = count - 1 == dof
            flat[sel] = student_t_quantile(q, int(dof)) * se[sel]
        half = flat.reshape(self.mean.shape)
        return self.mean - half, self.mean + half

    def _positions(self, draws: ArrayLike) -> np.ndarray:
        d = np.asarray(draws)
        if d.ndim != 2 or not np.issubdtype(d.dtype, np.integer):
            raise ValueError("draws must be a (replicates, graphs) integer array (graph_bootstrap)")
        pos = np.searchsorted(self.graphs, d)
        ok = (pos < self.graphs.size) & (self.graphs[np.minimum(pos, self.graphs.size - 1)] == d)
        if not ok.all():
            missing = sorted(set(d[~ok].tolist()))[:5]
            raise ValueError(
                f"drawn graph labels {missing} are not among this estimate's graphs; draw from "
                "the same graph labels for every rate and observable (graph_bootstrap)"
            )
        return pos

    def bootstrap(self, draws: ArrayLike) -> np.ndarray:
        """The estimate on each bootstrap replicate: ``(B, *shape)``.

        ``draws`` are graph labels from :func:`graph_bootstrap` (one row per replicate). A
        replicate's estimate is the mean of the drawn graphs' values (a graph drawn twice counts
        twice), so the same draws applied to two rates give replicate differences that are the
        differences of the replicates: the pairing survives.
        """
        return _replicate(self.graph_means, self._positions(draws), with_se=False)[0]

    def to_dict(self) -> dict[str, Any]:
        """The result as plain Python (lists and numbers; NaN as None)."""
        out: dict[str, Any] = _plain(self)
        return out


@dataclass(frozen=True, eq=False)
class NestedMean(GraphMean):
    """The nested estimator of v1.4 §6.3 and its counts per level.

    ``graph_means[g]`` is ``f_g = (1/C_g) sum_c (1/R_gc) sum_r F_gcr``; ``mean`` gives every
    graph the same weight. ``circuits_per_graph`` (``C_g``), ``trajectories_per_graph``,
    ``circuit_keys`` (the ``(g, c)`` pairs, ``(M, 2)``) and ``replicas_per_circuit`` (``R_gc``)
    describe the sample; ``contributing_trajectories`` counts the defined values (all rows
    unless ``nan_policy='omit'``).

    With ``nan_policy='omit'`` a NaN row is left out of its circuit, a circuit with no defined
    row out of its graph, and a graph with no defined circuit out of the mean. The target then
    changes: the estimate is the equal-graph mean, over the graphs with a defined value, of each
    graph's mean of the defined values, which is neither ``E[F]`` nor ``E[F | defined]`` (the
    latter weights graphs by how often they are defined). ``contributing_graphs`` says how many
    graphs entered.
    """

    circuits_per_graph: np.ndarray
    trajectories_per_graph: np.ndarray
    circuit_keys: np.ndarray
    replicas_per_circuit: np.ndarray
    contributing_trajectories: np.ndarray
    nan_policy: str

    @property
    def num_circuits(self) -> int:
        return int(self.circuit_keys.shape[0])

    @property
    def num_trajectories(self) -> int:
        return int(self.trajectories_per_graph.sum())


def nested_mean(
    g: ArrayLike, c: ArrayLike, r: ArrayLike, values: ArrayLike, *, nan_policy: NanPolicy = "raise"
) -> NestedMean:
    """The equal-graph-weight nested mean of v1.4 §6.3 with its graph-cluster error (§6.4).

    ``g, c, r`` label each row (trajectory) with its graph, gate sequence and replica;
    ``values`` has one row per trajectory and any trailing shape (for example one column per
    checkpoint, or replicates of a simulation study). Then

        f_gc = (1/R_gc) sum_r F_gcr,   f_g = (1/C_g) sum_c f_gc,   mu = (1/G) sum_g f_g,

    and ``se = sqrt(sum_g (f_g - mu)^2 / (G (G - 1)))``. Under a nonadaptive allocation (the
    numbers ``C_g`` and ``R_gc`` fixed before the values are seen) ``mu`` is unbiased for the
    ensemble mean and every graph has weight ``1/G``; averaging all rows instead weights graphs
    by their row counts. With equal ``C_g = C`` and ``R_gc = R`` it is the plain mean of the
    rows.

    The rows are put in canonical order (by ``g``, then ``c``, then ``r``) before any sum, so
    the result is bitwise the same for any order or chunking of the rows. A repeated
    ``(g, c, r)`` is refused, as are infinite values. NaN values are refused unless
    ``nan_policy='omit'`` (see :class:`NestedMean` for what that estimates).
    """
    omit = _nan_policy(nan_policy)
    gg, cc, rr = _rows(g, c, r)
    v = _values(values, gg.size)
    _check_nan(v, omit)
    h = _hierarchy(gg, cc, rr)
    shape = v.shape[1:]
    flat = v[h.order].reshape(gg.size, -1)
    graph_means, defined_rows = _graph_means(h, flat)
    mean, se, count = _summary(graph_means)
    return NestedMean(
        mean=mean.reshape(shape),
        se=se.reshape(shape),
        graphs=h.graphs,
        graph_means=graph_means.reshape((h.graphs.size, *shape)),
        contributing_graphs=count.reshape(shape),
        circuits_per_graph=h.circuits_per_graph,
        trajectories_per_graph=h.trajectories_per_graph,
        circuit_keys=h.circuit_keys,
        replicas_per_circuit=h.replicas,
        contributing_trajectories=defined_rows.reshape(shape),
        nan_policy=nan_policy,
    )


def mean_over_graphs(graphs: ArrayLike, graph_means: ArrayLike) -> GraphMean:
    """A :class:`GraphMean` from one value per graph (axis 0), NaN for an undefined value.

    For data already reduced to graph level (for example the per-graph differences of
    :func:`paired_difference`, or a simulation study). Labels must be distinct; they are sorted
    together with their values.
    """
    labels = _labels("graphs", graphs)
    gm = np.asarray(graph_means, dtype=np.float64)
    if gm.ndim == 0 or gm.shape[0] != labels.size:
        raise ValueError(
            f"graph_means must have one entry per graph label along axis 0 ({labels.size}); "
            f"got shape {gm.shape}"
        )
    if labels.size == 0:
        raise ValueError("no graphs")
    if np.isinf(gm).any():
        raise ValueError("graph_means must be finite or NaN")
    order = np.argsort(labels, kind="stable")
    labels, gm = labels[order], gm[order]
    if (labels[1:] == labels[:-1]).any():
        raise ValueError("graph labels must be distinct")
    mean, se, count = _summary(gm)
    return GraphMean(mean=mean, se=se, graphs=labels, graph_means=gm, contributing_graphs=count)


def paired_difference(
    first: GraphMean, second: GraphMean, *, graphs: Literal["same", "common"] = "same"
) -> GraphMean:
    """``second - first`` on the same graphs: differences within each graph, then the cluster error.

    v1.4 §6.4: for rates compared on shared graphs, form ``D_g = f_g(second) - f_g(first)`` within
    each graph first and take the mean and graph-cluster error of the ``D_g``. The graphs are
    independent, so the ``D_g`` are, and ``Var(mean D) = (1/G^2) sum_g [Var f_g(1) + Var f_g(2)
    - 2 Cov(f_g(1), f_g(2))]``: the positive covariance that coupled rates share (graph, gates and
    mask uniforms, v1.4 §7.3) shrinks the error. Treating the two means as independent would
    drop the covariance term and overstate the error.

    Both estimates must have the same graph labels (``graphs='same'``, the default) or are
    restricted to the labels they share (``graphs='common'``). A graph whose value is NaN at
    either rate is left out; ``contributing_graphs`` counts the pairs used. The shapes of the two
    values must agree.
    """
    if not isinstance(first, GraphMean) or not isinstance(second, GraphMean):
        raise ValueError("paired_difference takes two GraphMean (or NestedMean) results")
    if first.mean.shape != second.mean.shape:
        raise ValueError(
            f"the two estimates have different shapes: {first.mean.shape} and {second.mean.shape}"
        )
    if graphs not in ("same", "common"):
        raise ValueError(f"graphs must be 'same' or 'common', not {graphs!r}")
    a, b = first.graphs, second.graphs
    if graphs == "same" and not np.array_equal(a, b):
        only_a = np.setdiff1d(a, b)[:5].tolist()
        only_b = np.setdiff1d(b, a)[:5].tolist()
        raise ValueError(
            "paired_difference compares the same graphs at both rates; graphs "
            f"{only_a} are only in the first and {only_b} only in the second (pass "
            "graphs='common' to use the graphs both have)"
        )
    common = np.intersect1d(a, b)
    if common.size == 0:
        raise ValueError("the two estimates have no graph in common")
    diff = (
        second.graph_means[np.searchsorted(b, common)]
        - first.graph_means[np.searchsorted(a, common)]
    )
    return mean_over_graphs(common, diff)


# ---------------------------------------------------------------------------------------------
# bootstrap over graphs
# ---------------------------------------------------------------------------------------------


def graph_bootstrap(graphs: int | ArrayLike, replicates: int, seed: Any) -> np.ndarray:
    """Graph labels drawn with replacement: a ``(replicates, G)`` integer array.

    ``graphs`` is the number ``G`` of graphs (labels ``0..G-1``, the study's ``g``) or the array
    of graph labels. Row ``b`` is replicate ``b``: ``G`` labels drawn uniformly with replacement.
    The draws come from the raw PCG64 stream of ``np.random.SeedSequence(seed)`` (``seed`` a
    non-negative integer or a sequence of them, for example ``(master, n)``), which numpy keeps
    stable across versions, so the same seed gives the same draws.

    Apply the *same* draws to every rate, observable and checkpoint of a condition (all of them
    were measured on the same graphs): :meth:`GraphMean.bootstrap` then gives replicates whose
    differences across rates are the replicates of the differences, so the pairing of v1.4 §6.4
    survives. Draws for different system sizes come from different graph collections and use
    separate seeds.
    """
    if isinstance(graphs, (int, np.integer)) and not isinstance(graphs, bool):
        if graphs < 1:
            raise ValueError("graphs must be at least 1")
        labels = np.arange(int(graphs), dtype=np.int64)
    else:
        labels = _labels("graphs", graphs)
        if labels.size == 0:
            raise ValueError("graphs must not be empty")
        if np.unique(labels).size != labels.size:
            raise ValueError("graph labels must be distinct")
        labels = np.sort(labels)
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, (int, np.integer))
        or replicates < 1
    ):
        raise ValueError(f"replicates must be a positive integer, not {replicates!r}")
    count = int(replicates) * labels.size
    positions = _bounded_integers(seed, count, labels.size)
    return labels[positions].reshape(int(replicates), labels.size)


_REPLICATE_CHUNK = 1 << 20  # gathered values per block of replicates (8 MB of float64)


def _replicate(
    graph_means: np.ndarray, positions: np.ndarray, *, with_se: bool = True
) -> tuple[np.ndarray, np.ndarray | None]:
    """Replicate means (and, with ``with_se``, graph-cluster SEs) of the graphs at ``positions``.

    ``positions`` is ``(B, G')``; the results are ``(B, *shape)``. The gathered values
    ``(rows, G', *shape)`` are built for a block of replicates at a time, at most about
    ``_REPLICATE_CHUNK`` values per block, so memory does not grow as ``B G' K`` (``K`` the size
    of one value). Each replicate's sums run over its own row in the same order whatever the
    blocks, so the result does not depend on the block size.
    """
    reps, width = positions.shape
    shape = graph_means.shape[1:]
    per_rep = max(1, width * int(np.prod(shape, dtype=np.int64)))
    step = max(1, _REPLICATE_CHUNK // per_rep)
    mean = np.empty((reps, *shape))
    se = np.empty((reps, *shape)) if with_se else None
    for first in range(0, reps, step):
        block = slice(first, min(reps, first + step))
        vals = graph_means[positions[block]]  # (rows, G', *shape)
        defined = ~np.isnan(vals)
        count = defined.sum(axis=1)
        m = _divide(np.where(defined, vals, 0.0).sum(axis=1), count)
        mean[block] = m
        if se is not None:
            dev = np.where(defined, vals - np.expand_dims(m, 1), 0.0)
            se[block] = np.sqrt(_divide((dev * dev).sum(axis=1), count * (count - 1)))
    return mean, se


def _quantile_sorted(xs: np.ndarray, count: np.ndarray, q: float) -> np.ndarray:
    """The type-7 (linear) quantile of the first ``count`` entries of each sorted column."""
    pos = (count - 1) * q
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, np.maximum(count - 1, 0))
    lo = np.maximum(lo, 0)
    frac = pos - lo
    a = np.take_along_axis(xs, lo[None, ...], axis=0)[0]
    b = np.take_along_axis(xs, hi[None, ...], axis=0)[0]
    out = a + frac * (b - a)
    return np.where(count > 0, out, np.nan)


def percentile_interval(
    replicates: ArrayLike, level: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """The percentile bootstrap interval along axis 0 (replicates), ignoring NaN replicates.

    The ``(1 - level)/2`` and ``(1 + level)/2`` quantiles of the replicates, each the linear
    interpolation between order statistics (numpy's default ``"linear"`` method, Hyndman and Fan
    type 7), computed here so it does not change with the numpy version. NaN where no replicate
    is defined.
    """
    lvl = _level(level)
    a = np.asarray(replicates, dtype=np.float64)
    if a.ndim == 0 or a.shape[0] == 0:
        raise ValueError("replicates must have at least one replicate along axis 0")
    xs = np.sort(a, axis=0)  # NaN sort last
    count = (~np.isnan(a)).sum(axis=0)
    return _quantile_sorted(xs, count, (1 - lvl) / 2), _quantile_sorted(xs, count, (1 + lvl) / 2)


# ---------------------------------------------------------------------------------------------
# censored first-passage times
# ---------------------------------------------------------------------------------------------


def _equal_weights(
    gg: np.ndarray, cc: np.ndarray, rr: np.ndarray, *, exact: bool
) -> np.ndarray | None:
    """Trajectory weights ``L / (C_g R_gc)`` as integers (``exact``; None when ``L N`` would
    reach :data:`_EXACT_WEIGHT_LIMIT`) or ``1 / (C_g R_gc)`` as floats (``exact=False``)."""
    h = _hierarchy(gg, cc, rr)
    circuit_graph = np.repeat(np.arange(h.graphs.size), h.circuits_per_graph)
    denom = h.circuits_per_graph[circuit_graph] * h.replicas  # (M,) C_g R_gc
    if exact:
        lcm = math.lcm(*(int(x) for x in np.unique(denom)))
        if lcm * gg.size >= _EXACT_WEIGHT_LIMIT:
            return None
        per_circuit: np.ndarray = lcm // denom
        weights = np.empty(gg.size, dtype=np.int64)
    else:
        per_circuit = 1.0 / denom
        weights = np.empty(gg.size, dtype=np.float64)
    weights[h.order] = np.repeat(per_circuit, h.replicas)
    return weights


def graph_equal_weights(g: ArrayLike, c: ArrayLike, r: ArrayLike) -> np.ndarray:
    """Integer trajectory weights proportional to ``1 / (C_g R_gc)``, equal in total per graph.

    The weight of trajectory ``(g, c, r)`` is ``L / (C_g R_gc)`` with ``L`` the least common
    multiple of all ``C_g R_gc``: every graph's weights sum to ``L``, so every graph has the same
    total weight, and every gate sequence of a graph has the same share of it (the weights of
    v1.4 §6.3's nested estimator). A weighted statistic that only uses ratios of weights (the
    Kaplan-Meier estimate, a weighted mean) is unchanged by the common factor; divide by
    ``weights.sum()`` for probabilities. Integers keep sums of weights exact, which lets
    :func:`weighted_kaplan_meier` decide a survival of exactly 1/2 exactly. Raises when ``L``
    times the number of trajectories reaches ``2^62`` (many different, large ``C_g R_gc``);
    :func:`nested_kaplan_meier` then uses float weights ``1 / (C_g R_gc)`` instead.
    """
    gg, cc, rr = _rows(g, c, r)
    weights = _equal_weights(gg, cc, rr, exact=True)
    if weights is None:
        raise ValueError(
            "the allocation's numbers of circuits and replicas have a least common multiple too "
            "large for exact integer weights; pass float weights 1/(C_g R_gc) to "
            "weighted_kaplan_meier instead (with clusters=g), or call nested_kaplan_meier, "
            "which does"
        )
    return weights


@dataclass(frozen=True, eq=False)
class SurvivalCurve:
    """A (weighted) Kaplan-Meier estimate of ``S(t) = P(tau > t)`` and what it supports.

    At the distinct event times ``t_j`` (``times``): ``survival`` is
    ``S(t_j) = prod_{i <= j} (1 - D_i / Y_i)`` (right-continuous, the value just after ``t_j``),
    ``se`` its standard error (``variance`` names the estimate), ``at_risk`` and ``events`` the
    shares ``Y_j / W`` and ``D_j / W`` of the total weight ``W``, and ``num_at_risk`` /
    ``num_events`` the unweighted counts. Before the first event time ``S = 1``; after the
    largest follow-up time ``horizon`` it is unknown (:meth:`survival_at` returns NaN there).

    * ``median``: ``min{t : S(t) <= 1/2}``, None when the curve stays above 1/2 within the
      follow-up. ``median_upper``: ``min{t : S(t) < 1/2}``; when the curve sits exactly at 1/2
      from ``median`` on, every time in ``[median, median_upper]`` is a median of the
      estimated law, and ``median_upper`` None means that interval reaches past the follow-up.
    * ``restricted_mean``: ``int_0^h S(t) dt = E[min(tau, h)]`` at ``h = restricted_horizon``,
      with ``restricted_mean_se``.
    * ``mean`` and ``mean_se``: the weighted mean of the times, given only when no trajectory is
      censored (then it equals the restricted mean at the largest time); None otherwise, since a
      mean of observed times under censoring is not the ensemble mean (v1.4 §5.8, §5.9).
    """

    times: np.ndarray
    survival: np.ndarray
    se: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    num_at_risk: np.ndarray
    num_events: np.ndarray
    trajectories: int
    censored: int
    clusters: int
    horizon: float
    median: float | None
    median_upper: float | None
    restricted_horizon: float
    restricted_mean: float
    restricted_mean_se: float
    mean: float | None
    mean_se: float | None
    exact_weights: bool
    variance: str = KM_VARIANCE

    def survival_at(self, t: ArrayLike) -> np.ndarray:
        """``S(t)`` at arbitrary times: a step function, NaN beyond ``horizon`` (the largest
        follow-up time) unless the curve has reached 0 there."""
        tt = np.asarray(t, dtype=np.float64)
        idx = np.searchsorted(self.times, tt, side="right") - 1
        s = np.where(idx >= 0, self.survival[np.maximum(idx, 0)] if self.times.size else 1.0, 1.0)
        known = self.times.size > 0 and self.survival[-1] == 0.0
        return s if known else np.where(tt > self.horizon, np.nan, s)

    def se_at(self, t: ArrayLike) -> np.ndarray:
        """The standard error of ``S(t)`` at arbitrary times (0 before the first event; NaN
        where :meth:`survival_at` is NaN)."""
        tt = np.asarray(t, dtype=np.float64)
        idx = np.searchsorted(self.times, tt, side="right") - 1
        s = np.where(idx >= 0, self.se[np.maximum(idx, 0)] if self.times.size else 0.0, 0.0)
        known = self.times.size > 0 and self.survival[-1] == 0.0
        return s if known else np.where(tt > self.horizon, np.nan, s)

    def to_dict(self) -> dict[str, Any]:
        """The result as plain Python (lists and numbers; NaN as None)."""
        out: dict[str, Any] = _plain(self)
        return out


def _exact_weights(w: np.ndarray) -> np.ndarray | None:
    """``w`` as int64 when every weight is an integer and their total fits exactly, else None."""
    if not (np.issubdtype(w.dtype, np.integer) or np.all(w == np.floor(w))):
        return None
    if int(np.max(w)) * w.size >= _EXACT_WEIGHT_LIMIT:
        return None
    return w.astype(np.int64)


def _snap_half(s: np.ndarray, kept: np.ndarray, at_risk: np.ndarray) -> None:
    """Make ``s[j] <= 0.5`` and ``s[j] < 0.5`` agree with the exact products (integer weights).

    ``s`` is the float64 cumulative product of ``kept[j] / at_risk[j]``; each factor and product
    is rounded at most four times, so ``|s[j] - S_j| <= 2 (j + 1) u`` (``u = 2^-53``) near 1/2.
    Only the ``j`` within ``(4 j + 8) u`` of 1/2 are checked, with Python integers: ``2 prod kept``
    against ``prod at_risk``. A value exactly 1/2 becomes 0.5; one on the wrong side of 1/2 is
    moved by one ulp.
    """
    u = 2.0**-53
    tol = (4.0 * np.arange(s.size) + 8.0) * u
    candidates = np.flatnonzero(np.abs(s - 0.5) <= tol)
    num, den, k = 1, 1, 0
    for j in candidates.tolist():
        while k <= j:
            num *= int(kept[k])
            den *= int(at_risk[k])
            k += 1
        twice = 2 * num
        if twice == den:
            s[j] = 0.5
        elif twice < den:
            s[j] = min(s[j], np.nextafter(0.5, 0.0))
        else:
            s[j] = max(s[j], np.nextafter(0.5, 1.0))


def _km_influence_chunks(
    ts: np.ndarray,
    es: np.ndarray,
    wn: np.ndarray,
    cluster: np.ndarray,
    n_clusters: int,
    event_times: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    kept: np.ndarray,
    s: np.ndarray,
    chunk: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(first_cluster, U)``: each cluster's influence ``U (k, J)`` on ``S(t_j)``.

    Derived in ``docs/study/06-statistics.md`` §6.3 (Proposition 6): for cluster ``g`` with
    at-risk and event weights ``Y_gj`` and ``D_gj`` (shares of the total weight),

        U_g(t_j) = S(t_j) sum_{i <= j} (Y_gi D_i - D_gi Y_i) / (Y_i (Y_i - D_i)),

    the derivative of ``S(t_j)`` along a scaling of cluster ``g``'s weights; ``U = 0`` from the
    first ``i`` with ``Y_i = D_i`` on (the curve is pinned at 0 there).
    """
    J = event_times.size
    pos = np.searchsorted(event_times, ts, side="right")  # at risk at j < pos; event index pos-1
    alive = kept > 0
    coef = np.zeros(J)
    coef[alive] = 1.0 / (y[alive] * kept[alive])
    if not alive.all():
        coef[int(np.flatnonzero(~alive)[0]) :] = 0.0
    by_cluster = np.argsort(cluster, kind="stable")
    starts = np.searchsorted(cluster[by_cluster], np.arange(n_clusters + 1))
    step = chunk or max(1, (1 << 20) // (J + 1))  # about 2^20 entries per (clusters, J) array
    for first in range(0, n_clusters, step):
        last = min(n_clusters, first + step)
        rows = by_cluster[starts[first] : starts[last]]
        local = cluster[rows] - first
        k = last - first
        exits = np.zeros((k, J + 1))
        np.add.at(exits, (local, pos[rows]), wn[rows])
        yg = np.cumsum(exits[:, ::-1], axis=1)[:, ::-1][:, 1:]
        dg = np.zeros((k, J))
        ev = es[rows]
        np.add.at(dg, (local[ev], pos[rows][ev] - 1), wn[rows][ev])
        u = np.cumsum((yg * d - dg * y) * coef, axis=1) * s
        yield first, u


def weighted_kaplan_meier(
    times: ArrayLike,
    observed: ArrayLike,
    weights: ArrayLike,
    *,
    clusters: ArrayLike | None = None,
    horizon: float | None = None,
) -> SurvivalCurve:
    """The weighted Kaplan-Meier estimate of ``S(t) = P(tau > t)`` with a cluster variance.

    ``times[i]`` is trajectory ``i``'s first-passage depth if ``observed[i]``, else its censoring
    depth (``T_max``, v1.4 §5.2); ``weights`` are positive (for the study
    :func:`graph_equal_weights`, or use :func:`nested_kaplan_meier`). At the distinct event
    times ``t_j`` with weighted numbers at risk ``Y_j = sum_{T_i >= t_j} w_i`` and of events
    ``D_j = sum_{T_i = t_j, observed} w_i``,

        S(t) = prod_{t_j <= t} (1 - D_j / Y_j).

    With censoring only at one horizon ``T_max`` this is exactly the weighted fraction of
    trajectories with ``tau > t`` for every ``t <= T_max`` (the product telescopes), so with
    graph-equal weights it is the nested mean of ``1[tau > t]``.

    **Variance.** ``clusters`` labels the independent units (the graphs ``g``; None makes every
    trajectory its own unit). The standard error is the graph-cluster delta method (sandwich):
    each cluster's influence on ``S(t)`` (see :func:`_km_influence_chunks`), summed within the
    cluster, with variance ``G/(G-1) sum_g U_g(t)^2``. With censoring only at a common horizon it
    equals the graph-cluster error of v1.4 §6.4 of the nested mean of ``1[tau > t]`` exactly.

    **Exactness.** With integer weights (``graph_equal_weights`` returns integers, unit weights
    are integers) the sums are exact and a survival of exactly 1/2 is decided exactly, so the
    median rule does not depend on rounding; other weights give a float64 curve.

    ``horizon`` (default: the largest time) sets the restricted mean ``int_0^h S(t) dt``; it may
    not exceed the largest follow-up time, past which ``S`` is unknown.
    """
    t = _times(times)
    n = t.size
    if n == 0:
        raise ValueError("no trajectories: times is empty")
    e = _flags(observed, n, "observed")
    w = np.asarray(weights)
    if w.shape != (n,):
        raise ValueError(f"weights must have one entry per trajectory ({n}); got {w.shape}")
    if w.dtype == np.bool_ or not (
        np.issubdtype(w.dtype, np.integer) or np.issubdtype(w.dtype, np.floating)
    ):
        raise ValueError("weights must be numbers")
    if not np.all(np.isfinite(w) & (w > 0)):
        raise ValueError("weights must be positive and finite")
    if clusters is None:
        cl = np.arange(n, dtype=np.int64)
    else:
        cl = _labels("clusters", clusters)
        if cl.size != n:
            raise ValueError(f"clusters must have one entry per trajectory ({n})")
    return _kaplan_meier(t, e, w, cl, horizon)


def _kaplan_meier(
    t: np.ndarray, e: np.ndarray, w: np.ndarray, cl: np.ndarray, horizon: float | None
) -> SurvivalCurve:
    n = t.size
    wf = w.astype(np.float64)
    exact = _exact_weights(w)
    # canonical order: time, events before censorings at a time, cluster, weight
    order = np.lexsort((wf, cl, ~e, t))
    ts, es, ws, cs = t[order], e[order], wf[order], cl[order]
    event_times = np.unique(ts[es])
    J = event_times.size
    first = np.searchsorted(ts, event_times, side="left")
    last = np.searchsorted(ts, event_times, side="right")
    event_cum = np.concatenate(([0], np.cumsum(es)))
    num_events = event_cum[last] - event_cum[first]
    if exact is not None:
        wi = exact[order]
        suffix_i = np.zeros(n + 1, dtype=np.int64)
        suffix_i[:n] = np.cumsum(wi[::-1])[::-1]
        y_int, kept_int = suffix_i[first], suffix_i[first + num_events]
        total = float(suffix_i[0])
        y_abs = y_int.astype(np.float64)
        kept_abs = kept_int.astype(np.float64)
    else:
        suffix = np.zeros(n + 1)
        suffix[:n] = np.cumsum(ws[::-1])[::-1]
        total = float(suffix[0])
        y_abs, kept_abs = suffix[first], suffix[first + num_events]
    s = np.cumprod(kept_abs / y_abs) if J else np.zeros(0)
    if exact is not None and J:
        _snap_half(s, kept_int, y_int)
    y = y_abs / total
    kept = kept_abs / total
    d = y - kept

    h = float(ts[-1]) if horizon is None else float(horizon)
    reached_zero = bool(J and s[-1] == 0.0)  # then S = 0 is known beyond the last time too
    if not (math.isfinite(h) and h >= 0.0 and (h <= ts[-1] or reached_zero)):
        raise ValueError(
            f"horizon must lie in [0, {float(ts[-1])!r}] (the largest follow-up time; S(t) is "
            f"unknown beyond it unless the curve has reached 0); got {horizon!r}"
        )
    seg_start = np.minimum(np.concatenate(([0.0], event_times)), h)
    seg_end = np.minimum(np.concatenate((event_times, [np.inf])), h)
    lengths = np.maximum(seg_end - seg_start, 0.0)
    restricted = float(np.sum(np.concatenate(([1.0], s)) * lengths))

    uniq, cluster = np.unique(cs, return_inverse=True)
    n_clusters = int(uniq.size)
    wn = ws / total
    var = np.zeros(J)
    rm_var = 0.0
    if J:
        for _, u in _km_influence_chunks(
            ts, es, wn, cluster, n_clusters, event_times, y, d, kept, s
        ):
            var += np.sum(u * u, axis=0)
            infl = np.sum(u * lengths[1:], axis=1)
            rm_var += float(np.sum(infl * infl))
    factor = n_clusters / (n_clusters - 1) if n_clusters > 1 else math.nan
    se = np.sqrt(factor * var)
    restricted_se = math.sqrt(factor * rm_var) if n_clusters > 1 else math.nan

    censored = int(n - es.sum())
    mean = mean_se = None
    if censored == 0:
        mean = float(np.sum(wn * ts))
        infl = np.bincount(cluster, weights=wn * (ts - mean), minlength=n_clusters)
        mean_se = math.sqrt(factor * float(np.sum(infl * infl))) if n_clusters > 1 else math.nan

    below = np.flatnonzero(s <= 0.5)
    strictly = np.flatnonzero(s < 0.5)
    return SurvivalCurve(
        times=event_times,
        survival=s,
        se=se,
        at_risk=y,
        events=d,
        num_at_risk=n - first,
        num_events=num_events,
        trajectories=n,
        censored=censored,
        clusters=n_clusters,
        horizon=float(ts[-1]),
        median=float(event_times[below[0]]) if below.size else None,
        median_upper=float(event_times[strictly[0]]) if strictly.size else None,
        restricted_horizon=h,
        restricted_mean=restricted,
        restricted_mean_se=restricted_se,
        mean=mean,
        mean_se=mean_se,
        exact_weights=exact is not None,
    )


def nested_kaplan_meier(
    g: ArrayLike,
    c: ArrayLike,
    r: ArrayLike,
    times: ArrayLike,
    observed: ArrayLike,
    *,
    horizon: float | None = None,
) -> SurvivalCurve:
    """The Kaplan-Meier curve of a first-passage time with graph-equal weights, clustered by graph.

    :func:`weighted_kaplan_meier` with :func:`graph_equal_weights` (each trajectory
    ``1 / (C_g R_gc)``, every graph the same total) and ``clusters = g``: the survival-curve
    analog of :func:`nested_mean`. With censoring only at ``T_max`` the curve, its error and the
    restricted mean to ``T_max`` equal the nested means (and v1.4 §6.4 errors) of ``1[tau > t]``
    and ``min(tau, T_max)``.

    Any allocation is accepted (v1.4 §6.3). The weights are integers when their least common
    multiple allows (``exact_weights`` True: a survival of exactly 1/2 is decided exactly);
    otherwise, for allocations with many different large ``C_g R_gc``, they are the floats
    ``1 / (C_g R_gc)`` (``exact_weights`` False: the same curve to rounding, with the median
    decided in float64).
    """
    gg, cc, rr = _rows(g, c, r)
    t = _times(times)
    if t.size != gg.size:
        raise ValueError(f"times must have one entry per trajectory ({gg.size})")
    e = _flags(observed, gg.size, "observed")
    weights = _equal_weights(gg, cc, rr, exact=True)
    if weights is None:
        weights = _equal_weights(gg, cc, rr, exact=False)
    assert weights is not None
    return _kaplan_meier(t, e, weights, gg, horizon)


@dataclass(frozen=True, eq=False)
class CensoringSummary:
    """How much of a first-passage sample is censored (v1.4 §5.9: report it per condition).

    ``censored_fraction`` is the graph-equal-weight fraction of censored trajectories (the
    nested mean of the censoring indicator) with its graph-cluster error ``censored_fraction_se``;
    ``raw_censored_fraction`` the plain fraction of rows. ``horizon`` is the largest time and
    ``censoring_times`` the distinct censoring depths (one, ``T_max``, in a single-horizon run).
    ``graphs_all_censored`` / ``graphs_any_censored`` count graphs whose every / some trajectory
    is censored.
    """

    trajectories: int
    observed: int
    censored: int
    observed_fraction: float
    censored_fraction: float
    censored_fraction_se: float
    raw_censored_fraction: float
    horizon: float
    censoring_times: np.ndarray
    graphs: int
    graphs_all_censored: int
    graphs_any_censored: int

    def to_dict(self) -> dict[str, Any]:
        """The result as plain Python (lists and numbers; NaN as None)."""
        out: dict[str, Any] = _plain(self)
        return out


def censoring_summary(
    g: ArrayLike, c: ArrayLike, r: ArrayLike, times: ArrayLike, observed: ArrayLike
) -> CensoringSummary:
    """Counts and graph-equal-weight fractions of observed and censored first passages."""
    gg, cc, rr = _rows(g, c, r)
    t = _times(times)
    if t.size != gg.size:
        raise ValueError(f"times must have one entry per trajectory ({gg.size})")
    e = _flags(observed, gg.size, "observed")
    est = nested_mean(gg, cc, rr, (~e).astype(np.float64))
    frac = float(est.mean)
    per_graph = est.graph_means
    return CensoringSummary(
        trajectories=int(gg.size),
        observed=int(e.sum()),
        censored=int((~e).sum()),
        observed_fraction=1.0 - frac,
        censored_fraction=frac,
        censored_fraction_se=float(est.se),
        raw_censored_fraction=float((~e).mean()),
        horizon=float(t.max()),
        censoring_times=np.unique(t[~e]),
        graphs=est.num_graphs,
        graphs_all_censored=int(np.sum(per_graph == 1.0)),
        graphs_any_censored=int(np.sum(per_graph > 0.0)),
    )


# ---------------------------------------------------------------------------------------------
# peak times (v1.4 §5.8)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PeakTimeSummary:
    """The three peak-time statistics of v1.4 §5.8, kept apart.

    * ``arithmetic``: the nested mean of ``T*`` (:class:`NestedMean`), an estimate of ``E[T*]``;
    * ``geometric`` = ``exp(log_mean.mean)``: ``exp(E[ln T*])``, with the delta-method error
      ``geometric_se = geometric * log_mean.se``; never larger than the arithmetic mean of the
      same values (Jensen);
    * ``argmax``: ``argmax_t E[d(t)]`` over the positive checkpoints of the nested mean trace
      ``mean_trace`` (the first maximizer; ``argmax_set`` lists every exact maximizer of the
      averaged values, ``argmax_at_horizon`` whether the largest checkpoint is one). It is the
      peak of the mean, not a mean of peaks.

    ``missing_fraction`` is the graph-equal-weight fraction of trajectories without a peak time
    (``d* = 0`` or an invalid trace), which the means leave out (a change of target: they are
    conditional on a defined peak). ``horizon_fraction`` is the fraction (over all trajectories)
    whose maximizing set contains ``T_max``: those peaks may lie beyond the horizon, so their
    ``T*`` is a lower bound. ``horizon_policy`` says what the means did with them:
    ``"refuse"`` (means None when any defined peak is at the horizon), ``"exclude"`` (left out:
    the means are conditional on an interior peak) or ``"include"`` (kept: the means are then lower
    bounds). ``note`` explains a missing mean.
    """

    arithmetic: NestedMean | None
    log_mean: NestedMean | None
    geometric: float | None
    geometric_se: float | None
    mean_trace: NestedMean | None
    checkpoints: np.ndarray | None
    argmax: float | None
    argmax_set: np.ndarray | None
    argmax_at_horizon: bool | None
    missing_fraction: float
    horizon_fraction: float
    horizon_policy: str
    note: str

    def argmax_replicates(self, draws: ArrayLike) -> np.ndarray:
        """``argmax_t`` of the mean trace on each bootstrap replicate (graph draws), ``(B,)``."""
        if self.mean_trace is None or self.checkpoints is None:
            raise ValueError("argmax_replicates needs the traces (pass traces and checkpoints)")
        reps = self.mean_trace.bootstrap(draws)
        return np.array([_trace_argmax(row, self.checkpoints)[0] for row in reps], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """The result as plain Python (lists and numbers; NaN as None)."""
        out: dict[str, Any] = _plain(self)
        return out


def _trace_argmax(trace: np.ndarray, checkpoints: np.ndarray) -> tuple[float, np.ndarray]:
    positive = checkpoints > 0
    values = trace[positive]
    times = checkpoints[positive]
    if values.size == 0 or np.isnan(values).all():
        return math.nan, np.zeros(0)
    best = np.nanmax(values)
    tied = times[values == best]
    return float(tied[0]), tied


def peak_time_summaries(
    g: ArrayLike,
    c: ArrayLike,
    r: ArrayLike,
    peak_times: ArrayLike,
    *,
    at_horizon: ArrayLike,
    traces: ArrayLike | None = None,
    checkpoints: ArrayLike | None = None,
    horizon_policy: HorizonPolicy = "refuse",
    trace_nan_policy: NanPolicy = "raise",
) -> PeakTimeSummary:
    """``E[T*]``, ``exp(E[ln T*])`` and ``argmax_t E[d(t)]`` (v1.4 §5.8), as different statistics.

    ``peak_times`` is each trajectory's ``T* = exp(mean_{t in A} ln t)`` over its maximizing set
    ``A`` of positive checkpoints (NaN where missing: ``d* = 0`` or an invalid trace);
    ``at_horizon[i]`` says whether ``T_max`` is in trajectory ``i``'s ``A``. ``traces``
    ``(N, len(checkpoints))`` are the traces ``d(t)`` at the checkpoints, zero-extended where
    ``k_t = 0`` for the unconditional averaged trace (v1.4 §5.5); with them the summary has the
    nested mean trace and its argmax over the positive checkpoints. All means give every graph
    the same weight (:func:`nested_mean`). ``horizon_policy`` is described in
    :class:`PeakTimeSummary`; ``trace_nan_policy`` is :func:`nested_mean`'s ``nan_policy`` for
    the traces.
    """
    if horizon_policy not in ("refuse", "exclude", "include"):
        raise ValueError(
            f"horizon_policy must be 'refuse', 'exclude' or 'include', not {horizon_policy!r}"
        )
    gg, cc, rr = _rows(g, c, r)
    tstar = _values(peak_times, gg.size, "peak_times")
    if tstar.ndim != 1:
        raise ValueError("peak_times must have one value per trajectory")
    defined = ~np.isnan(tstar)
    if np.any(tstar[defined] <= 0):
        raise ValueError("peak times must be positive (T* is a geometric mean of positive depths)")
    horizon = _flags(at_horizon, gg.size, "at_horizon")
    missing_fraction = float(nested_mean(gg, cc, rr, (~defined).astype(np.float64)).mean)
    horizon_fraction = float(nested_mean(gg, cc, rr, (horizon & defined).astype(np.float64)).mean)

    note = ""
    use = defined.copy()
    if horizon_policy == "exclude":
        use &= ~horizon
    blocked = horizon_policy == "refuse" and bool(np.any(horizon & defined))
    arithmetic = log_mean = None
    geometric = geometric_se = None
    if blocked:
        note = (
            "some peaks are at the horizon T_max (their T* is a lower bound): the means are "
            "withheld; choose horizon_policy='exclude' (condition on an interior peak) or "
            "'include' (lower bounds)"
        )
    elif not use.any():
        note = "no trajectory has a peak time that the policy keeps"
    else:
        vals = np.where(use, tstar, np.nan)
        arithmetic = nested_mean(gg, cc, rr, vals, nan_policy="omit")
        log_mean = nested_mean(
            gg, cc, rr, np.log(vals, where=use, out=np.full(vals.shape, np.nan)), nan_policy="omit"
        )
        geometric = math.exp(float(log_mean.mean))
        geometric_se = geometric * float(log_mean.se) if np.isfinite(log_mean.se) else math.nan
        if horizon_policy == "include" and np.any(horizon & defined):
            note = "peaks at the horizon are included: the means are lower bounds"

    mean_trace = None
    cps = None
    argmax = None
    argmax_set = None
    argmax_at_horizon = None
    if traces is not None:
        if checkpoints is None:
            raise ValueError("traces need their checkpoints")
        cps = _times(checkpoints, "checkpoints")
        tr = _values(traces, gg.size, "traces")
        if tr.ndim != 2 or tr.shape[1] != cps.size:
            raise ValueError(
                f"traces must be (trajectories, checkpoints) = ({gg.size}, {cps.size}); got {tr.shape}"
            )
        if np.any(np.diff(cps) <= 0):
            raise ValueError("checkpoints must be strictly increasing")
        omit = _nan_policy(trace_nan_policy)
        _check_nan(tr, omit, "trace values")
        mean_trace = nested_mean(gg, cc, rr, tr, nan_policy=trace_nan_policy)
        best, tied = _trace_argmax(mean_trace.mean, cps)
        if math.isfinite(best):
            argmax, argmax_set = best, tied
            argmax_at_horizon = bool(cps[-1] in tied)
    elif checkpoints is not None:
        raise ValueError("checkpoints were given without traces")
    return PeakTimeSummary(
        arithmetic=arithmetic,
        log_mean=log_mean,
        geometric=geometric,
        geometric_se=geometric_se,
        mean_trace=mean_trace,
        checkpoints=cps,
        argmax=argmax,
        argmax_set=argmax_set,
        argmax_at_horizon=argmax_at_horizon,
        missing_fraction=missing_fraction,
        horizon_fraction=horizon_fraction,
        horizon_policy=horizon_policy,
        note=note,
    )
