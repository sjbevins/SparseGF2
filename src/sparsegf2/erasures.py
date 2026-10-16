"""Geometry-correlated erasure regions on a graph (the graph side: no codes here).

An erasure that strikes qubits close to each other in the interaction graph is
modelled by a law on regions of a fixed size that favours small edge boundaries.

**Notation.** ``G = (V, E)`` is the interaction graph, ``V = {0, ..., n-1}``
(the qubit labels), ``E`` a set of undirected edges without self-loops (an
``(E, 2)`` array as in :mod:`sparsegf2.graphs`; repeated edges count once). A
region ``A`` of ``V`` has size ``m = |A|`` and edge boundary::

    ∂A = {{u, v} in E : u in A, v not in A},        b(A) = |∂A|

**The correlated law.** At coupling ``J`` (any finite real number)::

    Q_J(A | m) = exp(-J b(A)) / Z_m(J),        Z_m(J) = sum over |A| = m of exp(-J b(A))

on the ``C(n, m)`` regions of size ``m``. ``J = 0`` is the uniform law (the
independent-erasure observable ``P_C(m)``); ``J > 0`` favours compact regions,
whose erased qubits cluster along the graph; ``J < 0`` favours spread-out ones.
With ``s_v = -1`` on ``A`` and ``+1`` off it, ``b(A) = sum over {u,v} in E of
(1 - s_u s_v) / 2``, so ``Q_J`` is the Ising model with coupling ``K = J / 2``
on ``G`` at fixed magnetization ``sum_v s_v = n - 2m`` (ferromagnetic for
``J > 0``).

**Density of states.** ``N_m(b) = #{A : |A| = m, b(A) = b}`` depends on ``G``
only, and ``Z_m(J) = sum_b N_m(b) exp(-J b)``, ``<b>_J = -d log Z_m / dJ``,
``d<b>_J / dJ = -Var_J(b) <= 0``: the mean boundary never increases with ``J``
(Proposition 2 of the study).

**Limits.** As ``J -> +inf``, ``Q_J`` becomes uniform on the regions of
minimum boundary (:func:`minimum_boundary_regions`): on the ring ``C_n`` the
``n`` intervals (``b = 2``), on the open path only the two intervals that touch
an end (``b = 1``). On graphs with shortcuts (Watts-Strogatz, ``beta > 0``) the
minimum-boundary regions need not be contiguous in the qubit order.

What is here:

* :func:`boundary`; :func:`boundary_counts` (exact ``N_m(b)`` by enumeration),
  :func:`boundary_moments` (``log Z``, ``<b>``, ``Var b`` from ``N_m``),
  :func:`minimum_boundary_regions` (the ``J -> +inf`` support);
* :func:`sample_regions` (:class:`RegionSample`): regions drawn from ``Q_J``,
  exactly and independently when the graph is one cycle or one path through
  all nodes (in any labeling), otherwise by a Metropolis swap chain per
  coupling with replica exchange between couplings;
* :func:`chain_diagnostics` (integrated autocorrelation time, effective sample
  size, R-hat) and :func:`region_diagnostics` (the same for the membership of
  a few nodes: on a ring the boundary looks mixed long before the regions have
  moved);
* the ring ``C_n`` and the open path ``P_n`` in closed form:
  :func:`ring_run_count`, :func:`ring_boundary_distribution`,
  :func:`ring_partition_function`, :func:`path_run_count`,
  :func:`path_boundary_distribution`, :func:`path_partition_function`, and
  the exact samplers :func:`sample_ring_regions` and
  :func:`sample_path_regions`.

>>> from sparsegf2 import erasures, graphs
>>> erasures.boundary(graphs.cycle(8), 8, [2, 3, 4])
2
>>> erasures.ring_run_count(8, 3, 2)
32
>>> erasures.path_run_count(8, 3, 1, 1)  # the prefix {0, 1, 2} and the suffix {5, 6, 7}
2
"""

from __future__ import annotations

import math
import operator
import secrets
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from sparsegf2 import graphs
from sparsegf2._kernel import erasure_chain as _chain
from sparsegf2._kernel.rng import seed_state

__all__ = [
    "ENUMERATION_BUDGET",
    "METHODS",
    "REGION_PROBES",
    "RegionSample",
    "SOKAL_WINDOW",
    "boundary",
    "boundary_counts",
    "boundary_moments",
    "chain_diagnostics",
    "minimum_boundary_regions",
    "path_boundary_distribution",
    "path_partition_function",
    "path_run_count",
    "region_diagnostics",
    "ring_boundary_distribution",
    "ring_partition_function",
    "ring_run_count",
    "sample_path_regions",
    "sample_regions",
    "sample_ring_regions",
]

ENUMERATION_BUDGET = 2_000_000
"""Default largest number of subsets :func:`boundary_counts` lists (``budget=``)."""

SOKAL_WINDOW = 5.0
"""The constant ``c`` of Sokal's automatic window in :func:`chain_diagnostics`."""


# ---------------------------------------------------------------------------------------
# Argument checks (every message says what to change)
# ---------------------------------------------------------------------------------------


def _integer(value: Any, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        number = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {number}")
    return number


def _graph(edges: Any, num_nodes: Any) -> tuple[int, np.ndarray]:
    n = _integer(num_nodes, "num_nodes", 1)
    return n, graphs.from_edges(edges, n)


def _size(size: Any, n: int) -> int:
    m = _integer(size, "size")
    if not 0 <= m <= n:
        raise ValueError(f"size must be in [0, {n}] (the number of erased nodes), got {m}")
    return m


def _region_mask(region: Any, n: int) -> np.ndarray:
    arr = np.asarray(region)
    if arr.size == 0:
        return np.zeros(n, dtype=bool)
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"region must be a list of node indices (integers), got {region!r}")
    outside = arr[(arr < 0) | (arr >= n)]
    if outside.size:
        raise ValueError(f"region: node {int(outside[0])} outside [0, {n})")
    values, counts = np.unique(arr, return_counts=True)
    if np.any(counts > 1):
        raise ValueError(f"region: node {int(values[counts > 1][0])} appears twice")
    mask = np.zeros(n, dtype=bool)
    mask[arr] = True
    return mask


def _coupling(coupling: Any) -> float:
    try:
        value = float(coupling)
    except (TypeError, ValueError):
        raise ValueError(f"coupling must be a number, got {coupling!r}") from None
    if not math.isfinite(value):
        raise ValueError(
            f"coupling must be finite, got {value} (J -> +inf is a limit, not a coupling: "
            "its regions are minimum_boundary_regions)"
        )
    return value


def _couplings(couplings: Any, *, distinct: bool) -> np.ndarray:
    try:
        arr = np.array(couplings, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"couplings must be numbers, got {couplings!r}") from None
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError("couplings must be one coupling or a list of couplings (1-D)")
    if arr.size == 0:
        raise ValueError("give at least one coupling")
    bad = np.flatnonzero(~np.isfinite(arr))
    if bad.size:
        raise ValueError(
            f"couplings must be finite, got {arr[bad[0]]} at position {int(bad[0])} "
            "(J -> +inf is a limit, not a coupling: its regions are minimum_boundary_regions)"
        )
    if distinct:
        values, counts = np.unique(arr, return_counts=True)
        if np.any(counts > 1):
            k = int(np.argmax(counts > 1))
            raise ValueError(
                f"couplings must be distinct: {values[k]:g} appears {int(counts[k])} times"
            )
    return arr


def _seed(seed: Any) -> int | tuple[int, ...]:
    """A concrete seed; ``None`` draws a fresh one below 2**53 (exact as a JSON number)."""
    message = "seed must be a non-negative integer, a list of them, or None (a fresh seed)"
    if seed is None:
        return secrets.randbits(53)
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError(f"{message}, got {seed!r}")
    if isinstance(seed, (int, np.integer)):
        if seed < 0:
            raise ValueError(f"{message}, got {seed}")
        return int(seed)
    if isinstance(seed, Sequence) and not isinstance(seed, str):
        words = tuple(_integer(s, "seed", 0) for s in seed)
        if words:
            return words
    raise ValueError(f"{message}, got {seed!r}")


def _csr(n: int, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _chain.csr(n, np.ascontiguousarray(edges, dtype=np.int64))


# ---------------------------------------------------------------------------------------
# Boundaries and the exact density of states
# ---------------------------------------------------------------------------------------


def boundary(edges, num_nodes: int, region) -> int:
    """The edge boundary ``b(A) = #{{u, v} in E : u in A, v not in A}`` of one region.

    ``region`` lists distinct nodes of ``0..num_nodes-1`` (any order; empty is
    allowed). The number of edges with exactly one end in ``A``; ``b(A) =
    b(V \\ A)``, and ``b`` is 0 for ``A`` empty or ``A = V``.
    """
    n, e = _graph(edges, num_nodes)
    mask = _region_mask(region, n)
    return int(np.count_nonzero(mask[e[:, 0]] != mask[e[:, 1]]))


def _check_budget(n: int, m: int, budget: Any) -> None:
    limit = ENUMERATION_BUDGET if budget is None else _integer(budget, "budget", 1)
    total = math.comb(n, m)
    if total > limit:
        raise ValueError(
            f"listing every region of size {m} on {n} nodes means C({n}, {m}) = {total:,} "
            f"subsets, above the budget of {limit:,}; pass budget={total} (the time grows "
            "in proportion to the number of subsets) or sample regions with sample_regions"
        )


def _enumerate(n: int, e: np.ndarray, m: int, target: int, rows: int):
    indptr, indices = _csr(n, e)
    hist = np.zeros(len(e) + 1, dtype=np.int64)
    regions = np.empty((rows, m), dtype=np.int64)
    found = _chain.enumerate_boundaries(indptr, indices, n, m, hist, target, regions)
    return hist, regions, found


def boundary_counts(edges, num_nodes: int, size: int, *, budget: int | None = None) -> np.ndarray:
    """The exact density of states ``N_m(b) = #{A : |A| = m, b(A) = b}``.

    Returns an int64 array ``N`` with ``N[b] = N_m(b)`` for ``b = 0, 1, ...,
    b_max``, where ``b_max`` is the largest boundary of a region of size
    ``m = size`` (so ``N[-1] > 0``); ``N.sum() == C(n, m)``. It gives
    ``Z_m(J) = sum_b N[b] exp(-J b)`` for every ``J`` at once
    (:func:`boundary_moments`). Every ``m``-subset is listed (lexicographic
    order, boundaries updated incrementally), so ``C(n, m)`` may not exceed
    ``budget`` (default :data:`ENUMERATION_BUDGET`, two million: well under a
    second compiled); a larger count is refused with the number of subsets.
    """
    n, e = _graph(edges, num_nodes)
    m = _size(size, n)
    _check_budget(n, m, budget)
    hist, _, _ = _enumerate(n, e, m, -1, 0)
    return hist[: int(np.flatnonzero(hist)[-1]) + 1].copy()


def minimum_boundary_regions(
    edges, num_nodes: int, size: int, *, budget: int | None = None
) -> tuple[int, np.ndarray]:
    """The regions of size ``m`` with the smallest boundary: ``(b_min, regions)``.

    ``b_min = min over |A| = m of b(A)``, and ``regions`` is an int64 array of
    shape ``(N_m(b_min), m)``: every region attaining it, one per row, nodes
    sorted, rows in lexicographic order. ``Q_J(. | m)`` tends to the uniform law
    on these rows as ``J -> +inf``. On the ring ``C_n`` (``1 <= m <= n-1``) they
    are the ``n`` intervals (``b_min = 2``); on the open path ``P_n`` the two
    intervals that touch an end (``b_min = 1``). Enumerates every subset, twice,
    within ``budget`` as :func:`boundary_counts`.
    """
    n, e = _graph(edges, num_nodes)
    m = _size(size, n)
    _check_budget(n, m, budget)
    hist, _, _ = _enumerate(n, e, m, -1, 0)
    b_min = int(np.flatnonzero(hist)[0])
    _, regions, found = _enumerate(n, e, m, b_min, int(hist[b_min]))
    assert found == len(regions)
    return b_min, regions


def boundary_moments(counts, couplings) -> dict[str, np.ndarray]:
    """``log Z_m(J)``, ``<b>_J`` and ``Var_J(b)`` from the density of states, per coupling.

    ``counts[b] = N_m(b)`` (from :func:`boundary_counts` or a closed form);
    ``couplings`` one finite ``J`` or a list. With ``w_J(b) = N_m(b)
    exp(-J b)``::

        log_partition = log Z_m(J) = log sum_b w_J(b)        (natural log; log C(n, m) at J = 0)
        mean          = <b>_J      = sum_b b w_J(b) / Z_m(J) = -d log Z_m / dJ
        variance      = Var_J(b)   = sum_b (b - <b>_J)^2 w_J(b) / Z_m(J) = -d<b>_J / dJ

    computed exactly (in floating point; the sums are shifted by their largest
    term, so any finite ``J`` is safe). Returns a dict of float64 arrays, one
    entry per coupling: ``couplings``, ``log_partition``, ``mean``,
    ``variance``. ``<b>_J`` is nonincreasing in ``J`` (``Var >= 0``).
    """
    try:
        weights = np.asarray(counts, dtype=np.float64)
    except (TypeError, ValueError):
        weights = np.array([np.nan])
    if (
        weights.ndim != 1
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.any(weights > 0)
    ):
        raise ValueError(
            "counts must be a 1-D list of non-negative numbers N_m(b), b = 0, 1, ..., with "
            "at least one positive entry (boundary_counts gives one)"
        )
    js = _couplings(couplings, distinct=False)
    b = np.flatnonzero(weights > 0)
    exponent = np.log(weights[b])[None, :] - js[:, None] * b[None, :]
    top = exponent.max(axis=1)
    w = np.exp(exponent - top[:, None])
    total = w.sum(axis=1)
    p = w / total[:, None]
    mean = p @ b
    variance = np.sum(p * (b[None, :] - mean[:, None]) ** 2, axis=1)
    return {
        "couplings": js,
        "log_partition": top + np.log(total),
        "mean": mean,
        "variance": variance,
    }


# ---------------------------------------------------------------------------------------
# The swap chain with replica exchange
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RegionSample:
    """Regions drawn from ``Q_J`` by :func:`sample_regions`, with how they were drawn.

    With ``c`` couplings, ``S`` recorded states per coupling and region size
    ``m`` (every array is read-only, and every per-coupling axis follows the
    order of ``couplings`` as given):

    * ``couplings``: float64[c], the couplings ``J`` as given.
    * ``size``: ``m``; ``num_nodes``: ``n``.
    * ``regions``: int64[c, S, m]; ``regions[i, s]`` is the ``s``-th recorded
      region at ``couplings[i]``, nodes sorted ascending.
    * ``boundaries``: int64[c, S], ``b`` of each recorded region.
    * ``method``: ``"chain"`` (the swap chain: records of one coupling are
      correlated) or ``"exact"`` (independent exact samples: a cycle or a path).
    * ``geometry``: ``"cycle"`` or ``"path"`` when the graph is one cycle or one
      path through all nodes (in any labeling), else ``"other"``.
    * ``acceptance``: float64[c], the fraction of swap proposals accepted at
      each coupling (NaN for exact samples, and when ``m = 0`` or ``m = n``:
      nothing to propose).
    * ``exchange_acceptance``: float64[c-1]; entry ``i`` is the fraction of
      accepted replica exchanges between the ``i``-th and ``(i+1)``-th
      smallest couplings (``np.sort(couplings)``), NaN without exchanges.
    * ``seed``: the seed used (a fresh one when ``seed=None`` was passed).
    * ``burn_in``, ``thin``: as passed (the exact samplers ignore them);
      ``tempering``: whether exchanges ran (False with one coupling and for
      exact samples).
    """

    couplings: np.ndarray
    size: int
    num_nodes: int
    regions: np.ndarray
    boundaries: np.ndarray
    method: str
    geometry: str
    acceptance: np.ndarray
    exchange_acceptance: np.ndarray
    seed: int | tuple[int, ...]
    burn_in: int
    thin: int
    tempering: bool

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                value.flags.writeable = False


METHODS = ("auto", "chain", "exact")
"""The ``method`` values of :func:`sample_regions`."""

_GEOMETRIES = {0: "other", 1: "cycle", 2: "path"}


def sample_regions(
    edges,
    num_nodes: int,
    size: int,
    couplings,
    samples: int,
    *,
    burn_in: int = 200,
    thin: int = 1,
    tempering: bool = True,
    seed: int | Sequence[int] | None = None,
    method: str = "auto",
) -> RegionSample:
    """Draw regions of size ``m`` from ``Q_J(A) ∝ exp(-J b(A))`` at each coupling.

    ``couplings``: finite, distinct, any order. ``method``:

    * ``"auto"`` (default): ``"exact"`` when the graph is one cycle or one path
      through all ``n`` nodes, in any labeling (the ring and the open chain, the
      brickwork geometries), else ``"chain"``.
    * ``"exact"``: independent exact samples (:func:`sample_ring_regions` and
      :func:`sample_path_regions`, laid along the cycle or path the edges
      form); refused for any other graph. ``burn_in``, ``thin`` and
      ``tempering`` do not apply.
    * ``"chain"``: the swap chain below, on any graph.

    **The chain.** One Metropolis swap chain (Kawasaki dynamics) per coupling.
    A proposal picks ``u`` uniformly in ``A`` and ``v`` uniformly outside it and
    moves to ``A' = A \\ {u} ∪ {v}``; the proposal probability
    ``1 / (m (n - m))`` is symmetric, and the move is accepted with probability
    ``min(1, exp(-J Δb))``, ``Δb = b(A') - b(A)``, so ``Q_J`` is stationary
    (detailed balance). Swaps connect all ``m``-subsets whatever the graph, so
    the chain is irreducible. ``Δb`` costs ``O(deg u + deg v)``::

        Δb = (deg_A(u) - deg_Ā(u)) + (deg_Ā(v) - deg_A(v)) + 2 * 1[u ~ v]

    (degrees into ``A`` and its complement ``Ā``, before the move; the edge
    ``{u, v}`` is a boundary edge before and after). A sweep is ``n`` proposals.

    **Replica exchange** (``tempering``, two or more couplings): after every
    sweep, the chains at neighbouring couplings ``J_i < J_{i+1}`` (sorted)
    exchange their states with probability ``min(1, exp((J_i - J_{i+1})
    (b_i - b_{i+1})))`` (Swendsen and Wang 1986; Geyer 1991; Hukushima and
    Nemoto 1996). The product of the laws is invariant, so each chain still
    samples exactly its own ``Q_J``, but states cross barriers at low ``J`` and
    climb the ladder: at large ``J`` a lone chain can stay for very long in one
    of several separated minima (on the open path, at one end), which R-hat
    over seeds reveals.

    **Recording.** Each chain starts from a uniformly random region (exact at
    ``J = 0``), runs ``burn_in`` sweeps, then records its state ``samples``
    times, ``thin`` sweeps apart. Records of one chain are correlated: use
    :func:`chain_diagnostics` on ``boundaries`` and :func:`region_diagnostics`
    on ``regions`` for errors and mixing. The boundary alone can mislead: at
    large ``J`` on a ring every state is an interval (``b = 2``) long before the
    interval has moved round the ring (about ``m (n - m) n / (4 pi^2)`` sweeps),
    which is why ``"auto"`` samples cycles and paths exactly.

    **Randomness.** The only source is the xoshiro256** state
    ``seed_state(seed)`` (:mod:`sparsegf2._kernel.rng`), drawn in the orders
    documented in :mod:`sparsegf2._kernel.erasure_chain`: ``sample_chain``
    (start states, then per sweep each chain's ``n`` proposals in increasing
    coupling order, then the exchanges), or ``ring_exact_sample`` /
    ``path_exact_sample`` (coupling by coupling in increasing order, sample by
    sample). The same arguments and seed give the same result on every
    platform. ``seed=None`` draws a fresh seed, reported as ``seed``.

    Returns a :class:`RegionSample`. Arrays are in the order of ``couplings``
    as given; ``exchange_acceptance`` follows the sorted couplings.
    """
    n, e = _graph(edges, num_nodes)
    m = _size(size, n)
    given = _couplings(couplings, distinct=True)
    samples = _integer(samples, "samples", 1)
    burn_in = _integer(burn_in, "burn_in", 0)
    thin = _integer(thin, "thin", 1)
    if not isinstance(tempering, (bool, np.bool_)):
        raise ValueError(f"tempering must be True or False, got {tempering!r}")
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"method must be 'auto', 'chain' or 'exact', got {method!r}")
    used = _seed(seed)

    indptr, indices = _csr(n, e)
    path_order, kind = _chain.geometry_order(indptr, indices, n)
    if method == "exact" and kind == 0:
        raise ValueError(
            "method='exact' needs a graph that is one cycle or one path through all "
            f"{n} nodes (in any labeling); this one is neither: use method='chain' (or 'auto')"
        )
    exact = method == "exact" or (method == "auto" and kind != 0)

    c = len(given)
    order = np.argsort(given, kind="stable")
    ladder = np.ascontiguousarray(given[order])
    regions = np.empty((c, samples, m), dtype=np.int64)
    bounds = np.empty((c, samples), dtype=np.int64)
    if exact:
        sampler = _chain.ring_exact_sample if kind == 1 else _chain.path_exact_sample
        sampler(n, m, ladder, samples, seed_state(used), path_order, regions, bounds)
        local = np.full(c, np.nan)
        exchange = np.full(max(c - 1, 0), np.nan)
    else:
        moves = np.zeros((c, 4), dtype=np.int64)
        _chain.sample_chain(
            indptr,
            indices,
            n,
            m,
            ladder,
            samples,
            burn_in,
            thin,
            bool(tempering),
            seed_state(used),
            regions,
            bounds,
            moves,
        )
        local = np.where(moves[:, 0] > 0, moves[:, 1] / np.maximum(moves[:, 0], 1), np.nan)
        exchange = np.where(moves[:-1, 2] > 0, moves[:-1, 3] / np.maximum(moves[:-1, 2], 1), np.nan)
    out_regions = np.empty_like(regions)
    out_bounds = np.empty_like(bounds)
    acceptance = np.empty(c, dtype=np.float64)
    out_regions[order] = regions
    out_bounds[order] = bounds
    acceptance[order] = local
    return RegionSample(
        couplings=given.copy(),
        size=m,
        num_nodes=n,
        regions=out_regions,
        boundaries=out_bounds,
        method="exact" if exact else "chain",
        geometry=_GEOMETRIES[int(kind)],
        acceptance=acceptance,
        exchange_acceptance=exchange.astype(np.float64),
        seed=used,
        burn_in=burn_in,
        thin=thin,
        tempering=bool(tempering) and c >= 2 and not exact,
    )


# ---------------------------------------------------------------------------------------
# Chain diagnostics
# ---------------------------------------------------------------------------------------


def _autocovariance(centred: np.ndarray) -> np.ndarray:
    """``C(t) = (1/N) sum_{s < N-t} x_s x_{s+t}`` per row, ``t = 0..N-1`` (FFT, zero-padded)."""
    draws = centred.shape[1]
    size = 1 << (2 * draws - 1).bit_length()
    spectrum = np.fft.rfft(centred, n=size, axis=1)
    power = spectrum.real**2 + spectrum.imag**2
    return np.fft.irfft(power, n=size, axis=1)[:, :draws] / draws


def _rank_normal(x: np.ndarray) -> np.ndarray:
    """``z = Phi^-1((r - 3/8) / (S + 1/4))`` with ``r`` the average rank among all ``S`` draws."""
    flat = x.ravel()
    values, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts - 1) / 2.0  # average rank of each tie group (1-based)
    normal = statistics.NormalDist()
    total = flat.size + 0.25
    z = np.array([normal.inv_cdf((r - 0.375) / total) for r in ranks.tolist()])
    return z[inverse.ravel()].reshape(x.shape)


def _rhat(z: np.ndarray) -> float:
    """Gelman-Rubin ``sqrt(((N-1)/N W + B/N) / W)`` on rows (chains) of ``N`` draws."""
    draws = z.shape[1]
    # A constant row has variance 0 and mean equal to its value exactly (float sums
    # would leave a rounding residue of about 1e-32, turning inf into 1e15).
    constant = np.ptp(z, axis=1) == 0
    means = np.where(constant, z[:, 0], np.mean(z, axis=1))
    within = float(np.mean(np.where(constant, 0.0, np.var(z, axis=1, ddof=1))))
    between_over_n = 0.0 if np.ptp(means) == 0 else float(np.var(means, ddof=1))
    if within == 0.0:
        return math.nan if between_over_n == 0.0 else math.inf
    return math.sqrt(((draws - 1) / draws * within + between_over_n) / within)


def _split_rhat(chains: np.ndarray) -> float:
    half = chains.shape[1] // 2
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    bulk = _rhat(_rank_normal(split))
    tail = _rhat(_rank_normal(np.abs(split - np.median(split))))
    values = [v for v in (bulk, tail) if not math.isnan(v)]
    return max(values) if values else math.nan


def chain_diagnostics(series) -> dict[str, Any]:
    """Mean, integrated autocorrelation time, effective sample size and R-hat of MCMC output.

    ``series`` is one chain's draws (1-D, at least 2) or independent chains of
    equal length (2-D: chains x draws, at least 4 draws each), for example
    ``RegionSample.boundaries[i]``, or one coupling's boundaries from several
    seeds stacked. Returns a dict:

    * ``mean``: the mean of all draws; ``variance``: their variance (``1/total``).
    * ``tau_int``: the integrated autocorrelation time ``tau = 1 + 2 sum_{t >= 1}
      rho(t)`` (1 for independent draws; AR(1) with lag-one correlation ``r``
      has ``(1 + r) / (1 - r)``). ``rho(t) = C(t) / C(0)`` with ``C`` the
      autocovariance of each chain about its own mean (``1/N`` normalization,
      by FFT), averaged over chains. The sum is cut at Sokal's automatic window
      (Madras and Sokal 1988, J. Stat. Phys. 50, 109; Sokal 1997, in
      Functional Integration, Plenum): ``tau(W) = 1 + 2 sum_{t=1}^{W} rho(t)``
      at the smallest ``W`` with ``W >= c tau(W)``, ``c = 5``
      (:data:`SOKAL_WINDOW`; the normalization and constant of emcee,
      Foreman-Mackey et al. 2013, PASP 125, 306), or ``W = N - 1`` when there is
      none. The estimate is reliable when the chain is much longer than
      ``tau`` (``N >= 50 tau`` is the usual rule), and meant for positively
      correlated draws (Metropolis chains). It is floored at
      ``1 / max(1, log10(total))`` (an effective size at most
      ``total log10(total)``, as Stan caps it), and is 1 for a series with no
      variation.
    * ``window``: ``W``; ``ess``: the effective sample size ``total / tau``;
      ``std_error``: the standard error of ``mean``, ``sqrt(variance tau /
      total)``; ``chains`` and ``draws``.
    * ``rhat`` (2-D only): the rank-normalized split R-hat of Vehtari, Gelman,
      Simpson, Carpenter and Bürkner (2021, Bayesian Analysis 16(2), 667), the
      larger of the bulk and the folded (tail) version. Each chain is split
      into halves (the middle draw dropped when ``N`` is odd); the draws are
      replaced by normal scores ``Phi^-1((r - 3/8) / (S + 1/4))`` of their
      average ranks ``r`` among all ``S`` draws (folded: of ``|x - median|``);
      then ``R = sqrt(((N'-1)/N' W + B/N') / W)`` (Gelman and Rubin 1992) with
      ``W`` the mean within-half variance and ``B/N'`` the variance of the half
      means. Near 1 (below 1.01 is the authors' threshold) when the chains
      agree; NaN when every draw is equal, inf when each half is constant but
      the halves differ.

    ``tau_int`` and ``ess`` describe the correlation inside the chains; across
    chains that disagree (R-hat well above 1) they are not to be trusted. For
    the swap chain, diagnostics of the boundary alone can say "mixed" while the
    regions have not moved (a ring at large ``J``): check
    :func:`region_diagnostics` too.
    """
    try:
        x = np.asarray(series, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("series must be numbers (one chain, or chains x draws)") from None
    if x.ndim == 1:
        if x.size < 2:
            raise ValueError(f"chain_diagnostics needs at least 2 draws, got {x.size}")
        chains = x[None, :]
    elif x.ndim == 2:
        if x.shape[0] < 1 or x.shape[1] < 4:
            raise ValueError(
                "chains x draws input needs at least one chain of at least 4 draws (split "
                f"R-hat halves each chain), got shape {x.shape}"
            )
        chains = x
    else:
        raise ValueError(
            f"series must be a 1-D series or a 2-D array (chains x draws), got {x.ndim} dimensions"
        )
    if not np.all(np.isfinite(chains)):
        raise ValueError("series must be finite numbers (no NaN or inf)")
    num_chains, draws = chains.shape
    total = num_chains * draws
    acov = _autocovariance(chains - chains.mean(axis=1, keepdims=True)).mean(axis=0)
    if acov[0] > 0:
        taus = 2.0 * np.cumsum(acov / acov[0]) - 1.0
        ok = np.arange(draws) >= SOKAL_WINDOW * taus
        window = int(np.argmax(ok)) if ok.any() else draws - 1
        tau = float(taus[window])
    else:
        window, tau = 0, 1.0
    tau = max(tau, 1.0 / max(1.0, math.log10(total)))
    variance = float(np.var(chains))
    ess = total / tau
    out: dict[str, Any] = {
        "mean": float(np.mean(chains)),
        "variance": variance,
        "tau_int": tau,
        "window": window,
        "ess": ess,
        "std_error": math.sqrt(variance / ess),
        "chains": num_chains,
        "draws": draws,
    }
    if x.ndim == 2:
        out["rhat"] = _split_rhat(chains)
    return out


REGION_PROBES = 8
"""How many nodes :func:`region_diagnostics` follows by default."""


def _nodes(values: Any, n: int, what: str) -> list[int]:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size == 0 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{what} must be a non-empty list of node indices, got {values!r}")
    bad = arr[(arr < 0) | (arr >= n)]
    if bad.size:
        raise ValueError(f"{what}: node {int(bad[0])} outside [0, {n})")
    return sorted({int(v) for v in arr})


def region_diagnostics(regions, num_nodes: int, *, vertices=None) -> dict[str, Any]:
    """Mixing diagnostics of where the regions are: the membership of a few nodes over the draws.

    ``regions``: int array ``(draws, m)``, the records of one chain
    (``RegionSample.regions[i]``), or ``(chains, draws, m)``, independent
    chains of one coupling stacked (for example from several seeds). For each
    followed node ``v`` the series ``x_s = 1[v in A_s]`` goes through
    :func:`chain_diagnostics`. ``vertices``: the nodes to follow (default
    :data:`REGION_PROBES` = 8 spread evenly in label order, ``floor(k n / 8)``
    for ``k = 0..7``, or every node when ``n < 8``).

    Why: the boundary ``b`` can look mixed while the regions have not moved.
    On a ring at large ``J`` every state is an interval (``b = 2``) almost at
    once, but the swap chain moves an interval by one place at a time, so its
    position relaxes only after about ``m (n - m) n / (4 pi^2)`` sweeps
    (``1.2e3`` at ``n = 64, m = 16``; ``8e4`` at ``n = 256, m = 64``). The
    membership of fixed nodes follows the position.

    Returns a dict (arrays have one entry per followed node, in ``vertices``
    order):

    * ``vertices``: the followed nodes (sorted).
    * ``mean``: the fraction of draws containing each node (its marginal).
    * ``tau_int``, ``ess``: per node, as in :func:`chain_diagnostics`.
    * ``changes``: how often each node's membership changed between
      consecutive draws (summed over chains).
    * ``slowest``: the node with the largest ``tau_int``; ``tau_int_max`` and
      ``ess_min`` over the nodes.
    * ``frozen``: the nodes whose membership never changed. A frozen node says
      nothing about mixing (``tau_int`` is 1 for a constant series): the run
      may be too short to see the region move. Several chains settle it.
    * ``rhat`` and ``rhat_max`` (several chains only): per node and the
      largest (ignoring NaN), as in :func:`chain_diagnostics`. Far above 1 when
      the chains hold their regions in different places.
    """
    n = _integer(num_nodes, "num_nodes", 1)
    arr = np.asarray(regions)
    if arr.ndim not in (2, 3) or (arr.size and not np.issubdtype(arr.dtype, np.integer)):
        raise ValueError(
            "regions must be an integer array of shape (draws, m) for one chain or "
            f"(chains, draws, m) for several, got shape {arr.shape}"
        )
    chains = arr[None] if arr.ndim == 2 else arr
    bad = chains[(chains < 0) | (chains >= n)]
    if bad.size:
        raise ValueError(f"regions: node {int(bad[0])} outside [0, {n})")
    if vertices is None:
        probes = sorted({k * n // REGION_PROBES for k in range(REGION_PROBES)})
    else:
        probes = _nodes(vertices, n, "vertices")
    member = np.stack([np.any(chains == v, axis=2) for v in probes], axis=-1).astype(np.float64)
    per = [
        chain_diagnostics(member[0, :, k] if arr.ndim == 2 else member[:, :, k])
        for k in range(len(probes))
    ]
    tau = np.array([d["tau_int"] for d in per])
    changes = np.count_nonzero(np.diff(member, axis=1), axis=(0, 1)).astype(np.int64)
    out: dict[str, Any] = {
        "vertices": probes,
        "mean": np.array([d["mean"] for d in per]),
        "tau_int": tau,
        "ess": np.array([d["ess"] for d in per]),
        "changes": changes,
        "slowest": probes[int(np.argmax(tau))],
        "tau_int_max": float(tau.max()),
        "ess_min": float(min(d["ess"] for d in per)),
        "frozen": [v for v, c in zip(probes, changes.tolist(), strict=True) if c == 0],
    }
    if arr.ndim == 3:
        rhat = np.array([d["rhat"] for d in per])
        out["rhat"] = rhat
        known = rhat[~np.isnan(rhat)]
        out["rhat_max"] = float(known.max()) if known.size else math.nan
    return out


# ---------------------------------------------------------------------------------------
# The ring C_n in closed form
# ---------------------------------------------------------------------------------------


def _ring_args(n: Any, m: Any) -> tuple[int, int]:
    n = _integer(n, "n")
    if n < 3:
        raise ValueError(f"the ring C_n needs n >= 3 nodes, got n = {n}")
    m = _integer(m, "m")
    if not 0 <= m <= n:
        raise ValueError(f"m must be in [0, {n}] (the number of erased nodes), got {m}")
    return n, m


def ring_run_count(n: int, m: int, r: int) -> int:
    """``N(n, m, r)``: the number of regions of size ``m`` on the ring ``C_n`` with ``r`` runs.

    A run is a maximal block of consecutive erased nodes (cyclically), and on
    the ring ``b(A) = 2r``. For ``1 <= m <= n-1``::

        N(n, m, r) = (n / r) C(m-1, r-1) C(n-m-1, r-1),    1 <= r <= min(m, n-m)

    and 0 for any other ``r`` (an exact integer). Reason: the ``r`` run lengths
    form a composition of ``m`` into ``r`` positive parts (``C(m-1, r-1)``), the
    gap lengths one of ``n - m`` (``C(n-m-1, r-1)``), and the first run may start
    at any of ``n`` nodes; each region arises ``r`` times this way, once per
    choice of its first run. For ``m = 0`` or ``m = n`` there is one region and
    no boundary: ``N = 1`` for ``r = 0``, else 0. Requires ``n >= 3``.
    """
    n, m = _ring_args(n, m)
    r = _integer(r, "r")
    if m in (0, n):
        return 1 if r == 0 else 0
    if not 1 <= r <= min(m, n - m):
        return 0
    count, rest = divmod(n * math.comb(m - 1, r - 1) * math.comb(n - m - 1, r - 1), r)
    assert rest == 0
    return count


def _log_comb(a: int, k: int) -> float:
    return math.lgamma(a + 1) - math.lgamma(k + 1) - math.lgamma(a - k + 1)


def _ring_log_weights(n: int, m: int, coupling: float) -> tuple[np.ndarray, np.ndarray]:
    """Runs ``r`` and ``log(N(n, m, r) exp(-2 J r))`` (exact counts up to n = 1000, lgamma above)."""
    if m in (0, n):
        return np.zeros(1, dtype=np.int64), np.zeros(1)
    runs = np.arange(1, min(m, n - m) + 1, dtype=np.int64)
    if n <= 1000:
        log_counts = [math.log(ring_run_count(n, m, r)) for r in runs.tolist()]
    else:
        log_counts = [
            math.log(n / r) + _log_comb(m - 1, r - 1) + _log_comb(n - m - 1, r - 1)
            for r in runs.tolist()
        ]
    return runs, np.array(log_counts) - 2.0 * coupling * runs


def ring_boundary_distribution(n: int, m: int, coupling: float) -> tuple[np.ndarray, np.ndarray]:
    """The law of ``b`` under ``Q_J`` on the ring ``C_n``: ``(values, probabilities)``.

    ``values`` (int64) are ``b = 2r`` for ``r = 1..min(m, n-m)`` (``[0]`` when
    ``m = 0`` or ``m = n``), and ``probabilities`` (float64, summing to 1)::

        P_J(b = 2r) = N(n, m, r) exp(-2 J r) / Z_m(J)

    with ``N`` from :func:`ring_run_count`, computed in log space (any ``n``,
    any finite ``J``).
    """
    n, m = _ring_args(n, m)
    runs, logw = _ring_log_weights(n, m, _coupling(coupling))
    w = np.exp(logw - logw.max())
    return 2 * runs, w / w.sum()


def ring_partition_function(n: int, m: int, coupling: float) -> float:
    """``log Z_m(J)`` on the ring ``C_n``: ``log sum_r N(n, m, r) exp(-2 J r)``.

    The natural logarithm (``Z`` itself overflows for large ``n``), computed in
    log space; ``log C(n, m)`` at ``J = 0`` and 0 for ``m = 0`` or ``m = n``.
    """
    n, m = _ring_args(n, m)
    _, logw = _ring_log_weights(n, m, _coupling(coupling))
    top = float(logw.max())
    return top + math.log(float(np.sum(np.exp(logw - top))))


def _exact_samples(kind: int, n: int, m: int, coupling: float, samples: int, seed) -> np.ndarray:
    regions = np.empty((1, samples, m), dtype=np.int64)
    bounds = np.empty((1, samples), dtype=np.int64)
    sampler = _chain.ring_exact_sample if kind == 1 else _chain.path_exact_sample
    order = np.arange(n, dtype=np.int64)
    sampler(n, m, np.array([coupling]), samples, seed_state(_seed(seed)), order, regions, bounds)
    return regions[0]


def sample_ring_regions(n: int, m: int, coupling: float, samples: int, *, seed) -> np.ndarray:
    """Exact independent samples of ``Q_J`` on the ring ``C_n``: int64 ``[samples, m]``, rows sorted.

    Each sample: draw the number of runs ``r`` from ``P(r) = N(n, m, r)
    exp(-2 J r) / Z`` (:func:`ring_boundary_distribution`); draw the run lengths,
    a uniform composition of ``m`` into ``r`` positive parts, and the gap lengths,
    one of ``n - m`` into ``r`` parts (stars and bars: a uniform ``(r-1)``-subset
    of the ``m - 1``, resp. ``n - m - 1``, places between consecutive units); draw
    the start of the first run uniformly among the ``n`` nodes; lay runs and gaps
    alternately around the ring. Every region with ``r`` runs comes from
    exactly ``r`` of the ``n C(m-1, r-1) C(n-m-1, r-1)`` equally likely triples
    (one per choice of which of its runs is laid first, which fixes the start
    and both compositions), so given ``r`` the region is uniform, and overall it
    has probability ``exp(-2 J r) / Z = Q_J(A)``. No Markov chain, so no burn-in
    and no correlation between samples. The ring is ``graphs.cycle(n)`` (node
    ``i`` next to ``i +- 1 mod n``); :func:`sample_regions` uses the same
    sampler on any graph that is a cycle, whatever its labels. ``seed`` as in
    :func:`sample_regions` (``None``: a fresh, unreported seed); the draw order
    is in :func:`sparsegf2._kernel.erasure_chain.ring_exact_sample`.
    """
    n, m = _ring_args(n, m)
    value = _coupling(coupling)
    samples = _integer(samples, "samples", 1)
    return _exact_samples(1, n, m, value, samples, seed)


# ---------------------------------------------------------------------------------------
# The open path P_n in closed form
# ---------------------------------------------------------------------------------------


def _path_args(n: Any, m: Any) -> tuple[int, int]:
    n = _integer(n, "n")
    if n < 2:
        raise ValueError(f"the path closed forms need n >= 2 nodes, got n = {n}")
    m = _integer(m, "m")
    if not 0 <= m <= n:
        raise ValueError(f"m must be in [0, {n}] (the number of erased nodes), got {m}")
    return n, m


def path_run_count(n: int, m: int, r: int, ends: int) -> int:
    """The number of regions of size ``m`` on the path ``P_n`` with ``r`` runs touching ``ends`` ends.

    A run is a maximal block of consecutive erased nodes; ``ends`` (0, 1 or 2)
    counts the ends of the path (nodes 0 and ``n - 1``) the region contains. On
    the path ``b(A) = 2 r - ends``. For ``1 <= m <= n-1``::

        ends = 0:  C(m-1, r-1) C(n-m-1, r)
        ends = 1:  2 C(m-1, r-1) C(n-m-1, r-1)
        ends = 2:  C(m-1, r-1) C(n-m-1, r-2)          (r >= 2)

    (an exact integer, 0 where a binomial is out of range). Reason: the ``r``
    run lengths are a composition of ``m`` (``C(m-1, r-1)``); the ``n - m``
    unerased nodes fill ``r + 1`` gaps, the ``r - 1`` inner ones positive and
    the two end gaps positive or empty as ``ends`` says, so ``n - m`` is split
    into ``r + 1``, ``r`` (either end empty: the factor 2) or ``r - 1`` positive
    parts. The empty region has ``(r, ends) = (0, 0)`` and the full one
    ``(1, 2)`` (both ``b = 0``): 1 each. Requires ``n >= 2``.
    """
    n, m = _path_args(n, m)
    r = _integer(r, "r")
    ends = _integer(ends, "ends")
    if ends not in (0, 1, 2):
        raise ValueError(
            f"ends must be 0, 1 or 2 (how many ends of the path the region touches), got {ends}"
        )
    if m == 0:
        return 1 if (r, ends) == (0, 0) else 0
    if m == n:
        return 1 if (r, ends) == (1, 2) else 0
    k = r - ends
    if not 1 <= r <= m or not 0 <= k <= n - m - 1:
        return 0
    count = math.comb(m - 1, r - 1) * math.comb(n - m - 1, k)
    return 2 * count if ends == 1 else count


def _path_log_weights(n: int, m: int, coupling: float) -> tuple[np.ndarray, np.ndarray]:
    """``b = 2r - e`` and ``log(count(r, e) exp(-J b))`` over the possible ``(r, e)``."""
    if m in (0, n):
        return np.zeros(1, dtype=np.int64), np.zeros(1)
    values, logs = [], []
    for r in range(1, min(m, n - m + 1) + 1):
        for e in range(3):
            k = r - e
            if not 0 <= k <= n - m - 1:
                continue
            if n <= 1000:
                log_count = math.log(path_run_count(n, m, r, e))
            else:
                log_count = _log_comb(m - 1, r - 1) + _log_comb(n - m - 1, k)
                log_count += math.log(2) if e == 1 else 0.0
            values.append(2 * r - e)
            logs.append(log_count - coupling * (2 * r - e))
    return np.array(values, dtype=np.int64), np.array(logs)


def path_boundary_distribution(n: int, m: int, coupling: float) -> tuple[np.ndarray, np.ndarray]:
    """The law of ``b`` under ``Q_J`` on the path ``P_n``: ``(values, probabilities)``.

    ``values`` (int64, increasing) are the boundaries that occur, ``b = 2r - e``
    (``1..2m`` at most; ``[0]`` when ``m = 0`` or ``m = n``), and
    ``probabilities`` (float64, summing to 1)::

        P_J(b) = sum over (r, e) with 2r - e = b of path_run_count(n, m, r, e) exp(-J b) / Z_m(J)

    computed in log space (any ``n``, any finite ``J``).
    """
    n, m = _path_args(n, m)
    bvals, logw = _path_log_weights(n, m, _coupling(coupling))
    values, inverse = np.unique(bvals, return_inverse=True)
    w = np.exp(logw - logw.max())
    return values.astype(np.int64), np.bincount(inverse.ravel(), weights=w) / w.sum()


def path_partition_function(n: int, m: int, coupling: float) -> float:
    """``log Z_m(J)`` on the path ``P_n``: ``log sum_{r, e} path_run_count(n, m, r, e) exp(-J (2r - e))``.

    The natural logarithm, in log space; ``log C(n, m)`` at ``J = 0`` and 0 for
    ``m = 0`` or ``m = n``.
    """
    n, m = _path_args(n, m)
    _, logw = _path_log_weights(n, m, _coupling(coupling))
    top = float(logw.max())
    return top + math.log(float(np.sum(np.exp(logw - top))))


def sample_path_regions(n: int, m: int, coupling: float, samples: int, *, seed) -> np.ndarray:
    """Exact independent samples of ``Q_J`` on the path ``P_n``: int64 ``[samples, m]``, rows sorted.

    Each sample: draw ``(r, e)`` (runs, ends touched) with probability
    ``path_run_count(n, m, r, e) exp(-J (2r - e)) / Z``; for ``e = 1`` the
    touched end, uniformly; the run lengths, a uniform composition of ``m``
    into ``r`` positive parts; the gap lengths, a uniform composition of
    ``n - m`` into the gaps that must be positive (``r + 1``, ``r`` or ``r - 1``
    of them: the inner gaps and the untouched ends); then lay gap 0, run 1,
    gap 1, ..., run ``r``, gap ``r`` from node 0. Each region arises from
    exactly one such choice, so given ``(r, e)`` it is uniform, and overall it
    has probability ``exp(-J b) / Z = Q_J(A)``. The path is
    ``graphs.path(n)``; :func:`sample_regions` uses the same sampler on any
    graph that is a path, whatever its labels. ``seed`` as in
    :func:`sample_regions` (``None``: a fresh, unreported seed); the draw order
    is in :func:`sparsegf2._kernel.erasure_chain.path_exact_sample`.
    """
    n, m = _path_args(n, m)
    value = _coupling(coupling)
    samples = _integer(samples, "samples", 1)
    return _exact_samples(2, n, m, value, samples, seed)
