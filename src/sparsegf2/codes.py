"""Stabilizer codes as objects: erasure analysis of a code after the circuit that made it.

The ensemble engine can save the code a monitored circuit has generated
(:func:`sparsegf2.mipt.code_snapshot`); :meth:`EnsembleResult.code
<sparsegf2.mipt.EnsembleResult.code>` returns it as a :class:`StabilizerCode`,
so erasure statistics can be recomputed later with more random orderings or
larger exhaustive searches than the run used. A code can also be built
directly from its stabilizer generators.

How it is computed: for a code with stabilizer group S on n qubits and its
normalizer N(S) (the Paulis commuting with all of S), erasing a set A of qubits
loses ``l(A) = rank(N(S)|_A) - rank(S|_A)`` bits of logical information, where
``G|_A`` is the restriction of a group's generators to the qubits in A. This
equals ``I(A:R)`` for a reference R holding the logical information, and
``2|A| - s - rank(H_A) + rank(H_Abar)`` in terms of the check matrix H. The
code keeps the generators of S and of N(S) as one column-major tableau with
the N(S) rows marked, so the same compiled routines that the engine uses with
a live reference compute ``l(A)``, first-failure sizes and exhaustive counts.

Geometry-correlated erasures (:mod:`sparsegf2.erasures` is the graph side):
regions A of a fixed size m erased with probability
``Q_J(A | m) = exp(-J b(A)) / Z_m(J)``, ``b(A)`` the number of edges of an
interaction graph with exactly one end in A, and the code's failure
probability ``P_J(m) = sum over |A| = m of Q_J(A | m) F(A)``,
``F(A) = 1[l(A) > 0]``. :meth:`StabilizerCode.correlated_failure` estimates it
from sampled regions, :meth:`StabilizerCode.boundary_failure_counts` gives it
exactly for small m and every J at once, and
:meth:`StabilizerCode.interval_failure_by_size` and
:meth:`StabilizerCode.end_window_failure_by_size` give its exact
``J -> +inf`` limits on the ring and the open path.
"""

from __future__ import annotations

import math
import operator
import re
from collections.abc import Iterable
from dataclasses import dataclass, fields

import numpy as np

from sparsegf2 import analysis, erasures, graphs
from sparsegf2._kernel import entropy as _ent
from sparsegf2._kernel import erasure_chain as _chain
from sparsegf2._kernel import erasure_failure as _fail
from sparsegf2._kernel.rng import seed_state
from sparsegf2.pauli import PauliString

__all__ = ["BoundaryCounts", "CorrelatedFailure", "StabilizerCode", "boundary_columns"]


def _read_only(obj) -> None:
    for field in fields(obj):
        value = getattr(obj, field.name)
        if isinstance(value, np.ndarray):
            value.flags.writeable = False


@dataclass(frozen=True, eq=False)
class CorrelatedFailure:
    """Estimates of a code's failure probability under geometry-correlated erasures.

    From :meth:`StabilizerCode.correlated_failure`. With ``c`` couplings, ``s``
    region sizes and ``S`` recorded regions per coupling and size (every array
    read-only; per-coupling axes follow ``couplings`` as given):

    * ``failure``: float64[c, s], ``failure[i, j] = failures[i, j] / samples``,
      the fraction of the recorded regions of size ``sizes[j]`` at coupling
      ``couplings[i]`` whose erasure loses logical information: the estimate
      of ``P_J(m) = sum over |A| = m of Q_J(A | m) 1[I(A:R) > 0]``. 0 for a
      code without logical qubits.
    * ``failures``: int64[c, s], how many of the ``S`` recorded regions failed.
    * ``failure_ess``: float64[c, s], the effective number of independent
      records behind each entry: ``S`` for exact samples (independent), and
      for the chain ``S / tau_int`` of the series of failure indicators
      (:func:`sparsegf2.erasures.chain_diagnostics`), capped at ``S``: the
      estimated ``tau_int`` of a short series can fall below 1 from noise
      (``chain_diagnostics`` floors it only at ``1 / log10(S)``), and records
      cannot be worth more than independent ones, so the cap keeps the error
      bar from shrinking below the binomial one (conservative). A standard error is
      ``sqrt(p (1 - p) / failure_ess)``; it is an estimate from the run
      itself (it can be too small when the chain has not mixed: compare seeds),
      and when no record or every record failed it gives 0, so read such an
      entry as "below about 3 / failure_ess" (or above 1 minus that).
    * ``couplings``: float64[c]; ``sizes``: int64[s]; ``samples``: ``S``.
    * ``mean_boundary``: float64[c, s], the mean boundary ``b`` of the recorded
      regions, an estimate of ``<b>_J`` (Proposition 2: the exact ``<b>_J``
      does not increase with ``J``; compare along increasing couplings, within
      errors).
    * ``acceptance``: float64[c, s], the fraction of swap proposals accepted
      (NaN for exact samples).
    * ``exchange_pairs``: float64[c-1, 2], the requested couplings in
      increasing order, as neighboring pairs (shape ``(0, 2)`` for one
      coupling); ``exchange_acceptance``: float64[c-1, s], for each pair the
      smallest replica-exchange acceptance among the rungs of the ladder
      between them (they are neighbors in the ladder only with
      ``ladder="given"``; NaN without exchanges, as for exact samples).
    * ``min_exchange_acceptance``: float64[s], the smallest exchange acceptance
      over every neighboring pair of the whole ladder of each size, including
      the links to inserted rungs such as ``J = 0`` (the one number to check;
      NaN without exchanges). ``rungs``: int64[s], how many couplings the
      ladder of each size ran (the cost is proportional to it).
    * ``method`` (``"chain"`` or ``"exact"``) and ``geometry`` (``"cycle"``,
      ``"path"`` or ``"other"``), as :class:`sparsegf2.erasures.RegionSample`.
    * ``size_warnings``: every warning of the sampler, as ``(m, sentence)``
      pairs in the order of ``sizes``; ``warnings``: the same merged for
      reading, one line per kind of warning with the sizes it occurred at
      (``"sizes 3-9, 12: replica exchange between J = 0 and J = 4 was accepted
      in 2.1% ..."``, with the figures of the first of those sizes and a note
      when they differ at the others; sentences that differ only in their
      figures are one kind, sentences about other couplings are not). Both
      empty when nothing was flagged.
    * ``seed``: the seed used; the regions of size ``m`` are those of
      ``erasures.sample_regions(..., seed=[seed, m])`` (for a list seed, its
      entries followed by ``m``). ``burn_in``: the sweeps before the first
      record (the default :func:`sparsegf2.erasures.default_burn_in` of n
      resolved, or as passed; the exact samplers ignore it); ``thin``: as
      passed; ``tempering``: whether replica exchange ran at any size (False
      for exact samples and for a one-rung ladder, whatever was passed).
      ``num_qubits``, ``num_logical``: the code's n and k.

    Records of one chain are correlated, so ``failures`` is not binomial:
    use ``failure_ess`` for errors, and for an ensemble the spread over codes
    (independent chains). To look closer at one size, reproduce its regions
    with ``erasures.sample_regions(..., seed=[seed, m])`` and
    :meth:`StabilizerCode.region_failures`.
    """

    couplings: np.ndarray
    sizes: np.ndarray
    samples: int
    failure: np.ndarray
    failures: np.ndarray
    failure_ess: np.ndarray
    mean_boundary: np.ndarray
    acceptance: np.ndarray
    exchange_pairs: np.ndarray
    exchange_acceptance: np.ndarray
    min_exchange_acceptance: np.ndarray
    rungs: np.ndarray
    method: str
    geometry: str
    warnings: tuple[str, ...]
    size_warnings: tuple[tuple[int, str], ...]
    seed: int | tuple[int, ...]
    burn_in: int
    thin: int
    tempering: bool
    num_qubits: int
    num_logical: int

    def __post_init__(self) -> None:
        _read_only(self)


@dataclass(frozen=True, eq=False)
class BoundaryCounts:
    """Every erasure region of size ``m = 1..w`` counted by its edge boundary, and the failing ones.

    From :meth:`StabilizerCode.boundary_failure_counts`. Arrays (int64,
    read-only) of shape ``(w, B + 1)``, ``w = max_size``, row ``m - 1`` for
    regions of size ``m`` and column ``b`` for boundary ``b``::

        total[m - 1, b]  = N_m(b) = #{A : |A| = m, b(A) = b}               (the graph only)
        failed[m - 1, b] = U_m(b) = #{A : |A| = m, b(A) = b, I(A:R) > 0}   (graph and code)

    ``B`` is the largest boundary of any counted region (the columns are
    trimmed to it, so ``total[:, -1]`` has a nonzero entry); a row is zero
    beyond its own largest boundary, so ``total[m - 1]`` is
    :func:`sparsegf2.erasures.boundary_counts` of size ``m`` padded with zeros.
    ``total[m - 1].sum() = C(n, m)`` and ``failed[m - 1].sum()`` is the number
    of uncorrectable ``m``-subsets
    (:meth:`StabilizerCode.uncorrectable_counts`). ``max_size``,
    ``num_qubits`` (n) and ``num_logical`` (k) describe the run.

    :meth:`exact` turns the counts into ``P_J(m)`` for any couplings. Arrays
    for counts of one's own (for example per trajectory, sharing one graph)
    need :func:`boundary_columns` columns before trimming.
    """

    failed: np.ndarray
    total: np.ndarray
    max_size: int
    num_qubits: int
    num_logical: int

    def __post_init__(self) -> None:
        _read_only(self)

    def exact(self, couplings) -> np.ndarray:
        """``P_J(m) = sum_b U_m(b) exp(-J b) / sum_b N_m(b) exp(-J b)``, float64 ``(c, w)``.

        Exactly (in floating point, stable for any finite ``J``), for each
        coupling in ``couplings`` (a number or a list) and ``m = 1..w``:
        :func:`sparsegf2.analysis.exact_correlated_failure`. ``J = 0`` is the
        uniform ``P_C(m)``. For a full curve with
        :func:`sparsegf2.analysis.correlated_failure_curve`, pass
        ``holds_code=num_logical`` (or ``max_size = n``, whose last entry is
        ``P(n) = 1[k > 0]``).
        """
        return analysis.exact_correlated_failure(self.failed, self.total, couplings)


def boundary_columns(edges, num_qubits: int, max_size: int) -> int:
    """How many boundary columns ``B + 1`` the counts of regions of size ``<= max_size`` need.

    For the graph ``edges`` on ``num_qubits`` nodes (``(E, 2)`` as in
    :mod:`sparsegf2.graphs`), every region ``A`` with ``|A| <= w = max_size``
    has ``b(A) <= min(|E|, sum of the w largest degrees) = B``, so count
    arrays of shape ``(w, B + 1)`` hold every boundary. The compiled kernel
    refuses smaller arrays; this is the one place the bound is computed
    (:func:`sparsegf2._kernel.erasure_failure.boundary_columns`, callable from
    compiled code with the CSR row pointer). :class:`BoundaryCounts` trims its
    columns afterwards to the largest boundary that occurs.
    """
    n = erasures._integer(num_qubits, "num_qubits", 1)
    e = np.ascontiguousarray(graphs.from_edges(edges, n), dtype=np.int64)
    w = erasures._integer(max_size, "max_size", 0)
    indptr, _ = _chain.csr(n, e)
    return int(_fail.boundary_columns(indptr, w))


_FIGURES = re.compile(r"(?<!J = )(?<![\w.\-])-?\d+(?:\.\d+)?")
"""Numbers in a warning other than the couplings (``J = ...``): percentages, counts."""


def _size_ranges(sizes: list[int]) -> str:
    """``[3, 4, 5, 7]`` as ``"3-5, 7"``."""
    parts, start = [], 0
    for i in range(1, len(sizes) + 1):
        if i == len(sizes) or sizes[i] != sizes[i - 1] + 1:
            a, b = sizes[start], sizes[i - 1]
            parts.append(str(a) if a == b else f"{a}-{b}")
            start = i
    return ", ".join(parts)


def _merge_warnings(notes: list[tuple[int, str]]) -> tuple[str, ...]:
    """One line per kind of warning (the sentence with its figures masked), with its sizes."""
    groups: dict[str, list[tuple[int, str]]] = {}
    for m, sentence in notes:
        groups.setdefault(_FIGURES.sub("#", sentence), []).append((m, sentence))
    lines = []
    for items in groups.values():
        sizes = sorted({m for m, _ in items})
        first_m, text = items[0]
        if len(sizes) == 1:
            label = f"size {sizes[0]}"
        else:
            label = f"sizes {_size_ranges(sizes)} ({len(sizes)} sizes)"
        if len({sentence for _, sentence in items}) > 1:
            text += f" (figures as at size {first_m}; they differ at the other sizes)"
        lines.append(f"{label}: {text}")
    return tuple(lines)


def _bounded(value, name: str, low: int, high: int) -> int:
    number = erasures._integer(value, name)
    if not low <= number <= high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {number}")
    return number


def _check_region_budget(n: int, w: int, budget) -> None:
    """Refuse an enumeration of sum_{m=1}^{w} C(n, m) regions above the budget."""
    limit = (
        erasures.ENUMERATION_BUDGET if budget is None else erasures._integer(budget, "budget", 1)
    )
    total, term = 0, 1
    for m in range(1, w + 1):
        term = term * (n - m + 1) // m  # C(n, m), exactly
        total += term
        if total > 10**30:
            break
    if total > limit:
        count = f"{total:,}" if total <= 10**30 else "more than 10^30"
        raise ValueError(
            f"listing every region of size 1..{w} on {n} qubits means C({n}, 1) + ... + "
            f"C({n}, {w}) = {count} subsets, above the budget of {limit:,}: pass a larger "
            f"budget (the time grows in proportion), a smaller max_size (the largest that fits is "
            f"analysis.exhaustive_cutoff({n}, {limit}) = {analysis.exhaustive_cutoff(n, limit)}), "
            "or sample regions with correlated_failure"
        )


class StabilizerCode:
    """A stabilizer code on ``n`` qubits, given by its stabilizer generators.

    Parameters
    ----------
    stabilizers
        The generators: an ``(s, 2n)`` 0/1 array whose row holds the X bits of
        qubits 0..n-1 then the Z bits, or Pauli strings (``PauliString`` or
        text such as ``"XZ_Y"``, case-insensitive; a leading sign is ignored);
        a single string is one generator. The generators must commute; they
        need not be independent.
    num_qubits
        ``n``; needed only when there are no generators.

    Examples
    --------
    >>> code = StabilizerCode(["XXXX", "ZZZZ"])      # the [[4, 2, 2]] code
    >>> code.num_logical
    2
    >>> code.lost_information([0]), code.lost_information([0, 1])
    (0, 2)
    >>> code.uncorrectable_counts(2).tolist()       # all 6 pairs are uncorrectable
    [0, 6]
    """

    def __init__(self, stabilizers, num_qubits: int | None = None):
        rows = _bit_rows(stabilizers, num_qubits)
        n = rows.shape[1] // 2
        self._init_packed(_pack(rows), n)

    @classmethod
    def from_packed(cls, rows: np.ndarray, num_qubits: int) -> StabilizerCode:
        """A code from packed generators: ``(s, 2 ceil(n / 64))`` uint64, X words then Z words."""
        code = cls.__new__(cls)
        code._init_packed(np.ascontiguousarray(rows, dtype=np.uint64), int(num_qubits))
        return code

    @classmethod
    def from_simulator(cls, sim, reference) -> StabilizerCode:
        """The code a simulator's system holds, the system being every qubit not in ``reference``.

        Gaussian elimination on the reference columns of the stabilizer
        generators leaves generators of the subgroup acting trivially on the
        reference; restricted to the system (renumbered 0..n_sys-1 in
        increasing order) they are the code's stabilizers, so ``num_logical``
        is S(reference). This is the same group the ensemble engine's
        ``code_snapshot`` saves.
        """
        n = sim.num_qubits
        ref = sorted({int(q) for q in reference})
        if not ref or ref[0] < 0 or ref[-1] >= n:
            raise ValueError(f"reference must be a non-empty set of qubits in [0, {n})")
        refset = set(ref)
        system = [q for q in range(n) if q not in refset]
        if not system:
            raise ValueError("the reference covers every qubit; there is no system")
        xs, zs, _ = sim.stabilizer_arrays()
        m = np.concatenate([xs[:, ref], zs[:, ref], xs[:, system], zs[:, system]], axis=1)
        m = (m % 2).astype(np.uint8)
        r, rows = 2 * len(ref), m.shape[0]
        pivot = 0
        for col in range(r):
            nz = np.flatnonzero(m[pivot:, col])
            if not nz.size:
                continue
            p = pivot + int(nz[0])
            m[[pivot, p]] = m[[p, pivot]]
            others = np.flatnonzero(m[:, col])
            m[others[others != pivot]] ^= m[pivot]
            pivot += 1
            if pivot == rows:
                break
        return cls(m[pivot:, r:], num_qubits=len(system))

    def _init_packed(self, packed: np.ndarray, n: int) -> None:
        if n < 1:
            raise ValueError("a code needs at least one qubit")
        wsys = (n + 63) // 64
        if packed.ndim != 2 or packed.shape[1] != 2 * wsys:
            raise ValueError(f"packed generators need shape (s, {2 * wsys}) for n = {n}")
        if n % 64 and packed.size:
            pad = ~((np.uint64(1) << np.uint64(n % 64)) - np.uint64(1))
            if np.any(packed[:, [wsys - 1, 2 * wsys - 1]] & pad):
                raise ValueError(f"packed generators have bits beyond qubit {n - 1}")
        if not _ent.generators_commute(packed):
            raise ValueError("the stabilizer generators do not all commute")
        normalizer, rank = _ent.symplectic_complement(packed, n)
        stack = np.concatenate([packed, normalizer])
        ws = (stack.shape[0] + 63) // 64
        tab = np.zeros((2 * n + 1, max(ws, 1)), dtype=np.uint64)
        _ent.columns_from_rows(stack, n, tab)
        marked = np.zeros(max(ws, 1), dtype=np.uint64)
        for r in range(packed.shape[0], stack.shape[0]):
            marked[r >> 6] |= np.uint64(1) << np.uint64(r & 63)
        self._n = n
        self._packed = packed
        self._rank = rank
        self._tab = tab
        self._ws = tab.shape[1]
        self._marked = marked  # the N(S) rows (the "reference" rows of the shared routines)
        self._system = np.arange(n, dtype=np.int64)

    # -- basic data --------------------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        return self._n

    @property
    def num_stabilizers(self) -> int:
        """The number of independent generators s (the rank of the check matrix)."""
        return self._rank

    @property
    def num_logical(self) -> int:
        """The number of logical qubits k = n - s."""
        return self._n - self._rank

    def stabilizer_matrix(self) -> np.ndarray:
        """The generators as given, as an ``(s, 2n)`` uint8 array (X bits, then Z bits)."""
        return _unpack(self._packed, self._n)

    def __repr__(self) -> str:
        return (
            f"StabilizerCode(n={self._n}, k={self.num_logical}, generators={self._packed.shape[0]})"
        )

    # -- erasure -----------------------------------------------------------------------

    def lost_information(self, erased: Iterable[int]) -> int:
        """``l(A) = I(A:R)``: bits of logical information destroyed by erasing the qubits A."""
        a = self._qubits(erased)
        scratch = np.zeros((max(2 * a.size, 1), self._ws), dtype=np.uint64)
        return int(_ent.lost_information_rows(self._tab, self._ws, a, self._marked, scratch))

    def is_correctable(self, erased: Iterable[int]) -> bool:
        """Whether the erasure of A can be corrected perfectly (``l(A) = 0``)."""
        return self.lost_information(erased) == 0

    def site_information(self) -> np.ndarray:
        """``l({i}) = I(i:R)`` for every qubit i (int64, 0, 1 or 2): the single-qubit erasures.

        Entry i is above 0 exactly when some single-qubit Pauli on i commutes with S
        without being in S (a weight-one logical operator); the same values as the engine's
        :func:`sparsegf2.mipt.site_information` probe, from two columns per qubit.
        """
        out = np.zeros(self._n, dtype=np.int64)
        _ent.site_information_rows(self._tab, self._ws, self._n, self._marked, out)
        return out

    def logical_sites(self) -> int:
        """N_log: the number of qubits that carry a weight-one logical operator (``I(i:R) > 0``).

        0 for a code without logical qubits; the code's contiguous distance (in any order) is
        at least 2 exactly when it has logical qubits and this is 0. The same value as the
        engine's :func:`sparsegf2.mipt.logical_sites` probe.
        """
        empty = np.zeros(0, dtype=np.int64)
        return int(_ent.site_information_rows(self._tab, self._ws, self._n, self._marked, empty))

    def first_failure_sizes(self, samples: int, *, seed) -> np.ndarray:
        """First-failure sizes along ``samples`` uniformly random orderings of the qubits.

        Entry i is the smallest m such that erasing the first m qubits of
        ordering i is uncorrectable, or n + 1 if no erasure fails (k = 0); the
        same estimator as :func:`sparsegf2.mipt.erasure_first_failure`.
        """
        if samples < 0:
            raise ValueError("samples must be non-negative")
        out = np.zeros(int(samples), dtype=np.float64)
        _ent.first_failure_samples(
            self._tab, self._ws, self._marked, self.num_logical, seed_state(seed), self._system, out
        )
        return out.astype(np.int64)

    def uncorrectable_counts(self, max_size: int) -> np.ndarray:
        """Number of uncorrectable m-qubit erasures for m = 1..``max_size``, by exhaustion.

        Entry ``m - 1`` divided by C(n, m) is the exact probability that a
        uniformly random m-qubit erasure is uncorrectable. The code distance
        exceeds ``max_size`` exactly when every count is 0.

        The counts are int64, and exact, while every C(n, m), m <= ``max_size``,
        is below 2**63. Beyond that they are float64: a count whose C(n, m) is
        below 2**63 is still computed exactly and rounded once, so it never
        exceeds float(C(n, m)); a larger one is a compensated float64 sum with
        relative error below about (2 ``max_size`` + 2) 2**-53, which
        :func:`sparsegf2.analysis.exact_failure_by_size` tolerates.
        """
        w = operator.index(max_size)
        if not 1 <= w <= self._n:
            raise ValueError(f"max_size must be in [1, {self._n}]")
        out = np.zeros(w, dtype=np.float64)
        exact = np.zeros(w, dtype=np.int64)
        if self.num_logical > 0:
            _ent.uncorrectable_counts_rows(
                self._tab, self._ws, self._system, self._marked, w, out, exact
            )
        if max(math.comb(self._n, m) for m in range(1, w + 1)) < 2**63:
            return exact
        return out

    def contiguous_distance(self, order=None, *, periodic: bool = True) -> int:
        """Length of the shortest contiguous window of ``order`` whose erasure is uncorrectable.

        ``order`` is a permutation of the qubits (default ``0..n-1``); windows
        wrap around its end when ``periodic``. Returns 0 when no window is
        uncorrectable (in particular when the code has no logical qubits).
        Uncorrectability is monotone in the window length, so the length is
        found by bisection, with all windows of a length checked.
        """
        n = self._n
        seq = np.arange(n) if order is None else np.asarray(list(order), dtype=np.int64)
        if seq.shape != (n,) or not np.array_equal(np.sort(seq), np.arange(n)):
            raise ValueError(f"order must be a permutation of 0..{n - 1}")
        if self.num_logical == 0:
            return 0

        def some_window_fails(length: int) -> bool:
            starts = range(n) if periodic and length < n else range(n - length + 1)
            for s in starts:
                window = seq[np.arange(s, s + length) % n]
                if self.lost_information(window) > 0:
                    return True
            return False

        if not some_window_fails(n):
            return 0
        lo, hi = 1, n  # some window of length hi fails
        while lo < hi:
            mid = (lo + hi) // 2
            if some_window_fails(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    # -- geometry-correlated erasures ----------------------------------------------------

    def region_failures(self, regions) -> np.ndarray:
        """``F(A) = 1[l(A) > 0]`` for many regions of one size: a bool array, one entry per region.

        ``regions`` is an integer array of shape ``(R, m)`` (or a list of ``R``
        lists of ``m`` qubits): each row lists ``m`` distinct qubits, any
        order. Entry ``r`` is True when erasing the qubits of row ``r`` loses
        logical information, ``l(A) = I(A:R) > 0`` (:meth:`lost_information`),
        by two GF(2) eliminations per region; a region of more than ``n / 2``
        qubits is evaluated on its complement, since ``l(A) + l(Ā) = 2k``
        (the cleaning lemma), so ``l(A) > 0`` exactly when ``l(Ā) < 2k``. All
        False for a code without logical qubits and for ``m = 0``; all True for
        ``m = n`` when ``k > 0``.
        """
        a = self._regions(regions)
        out = np.zeros(a.shape[0], dtype=np.uint8)
        self._fill_failures(a, out)
        return out.astype(bool)

    def _fill_failures(self, regions: np.ndarray, out: np.ndarray) -> None:
        n, m = self._n, regions.shape[1]
        scratch = np.zeros((max(2 * min(m, n - m), 1), self._ws), dtype=np.uint64)
        k = self.num_logical
        _fail.region_failures(self._tab, self._ws, self._marked, n, k, regions, scratch, out)

    def correlated_failure(
        self,
        edges,
        couplings,
        sizes=None,
        samples: int = 100,
        *,
        burn_in: int | None = None,
        thin: int = 1,
        tempering: bool = True,
        seed=None,
        method: str = "auto",
        ladder="auto",
        max_step: float | None = None,
    ) -> CorrelatedFailure:
        """Estimate ``P_J(m)``, the failure probability under geometry-correlated erasures of size ``m``.

        The erasure law on the regions ``A`` of size ``m`` is
        ``Q_J(A | m) = exp(-J b(A)) / Z_m(J)``, with ``b(A)`` the number of edges
        of the graph ``edges`` (on this code's ``n`` qubits, ``(E, 2)`` as in
        :mod:`sparsegf2.graphs`) with exactly one end in ``A``: ``J > 0`` favors
        compact regions, ``J = 0`` is uniform, ``J < 0`` favors spread-out
        ones. With ``l(A) = I(A:R)`` the bits of logical information erasing
        ``A`` destroys (:meth:`lost_information`), the failure probability is::

            P_J(m) = sum over |A| = m of Q_J(A | m) F(A),     F(A) = 1[l(A) > 0]

        For each size ``m`` in ``sizes`` (distinct integers in ``[0, n]``, any
        order; default every ``m = 1..n-1``, since ``P(0) = 0`` and
        ``P(n) = 1[k > 0]`` are known), :func:`sparsegf2.erasures.sample_regions`
        draws ``samples`` regions at every coupling in ``couplings`` (distinct, any
        order, ``|J| <= erasures.MAX_ABS_COUPLING``: beyond it use the exact J -> inf routes
        of this class) with ``burn_in`` (default ``None``: that
        function's size-aware default), ``thin``, ``tempering``, ``method``,
        ``ladder`` and ``max_step`` passed on (exact independent samples when
        the graph is one cycle or one path, in any labeling; otherwise a swap
        chain per coupling with replica exchange along an automatic ladder of
        couplings: see that function for the algorithm, its diagnostics and
        warnings). Each region's failure is computed
        exactly (:meth:`region_failures`), and the estimate is the failing
        fraction::

            failure[i, j] = (1 / S) sum_s F(A_s),   A_1..A_S the regions of size sizes[j] at couplings[i]

        It converges to ``P_J(m)`` as ``S`` grows: unbiased for exact samples,
        and for the chain biased only by incomplete equilibration (a lone
        chain at ``J = 0`` starts in its stationary law and stays there; with
        replica exchange it also receives states from the other rungs, which
        start uniform too, so the bias vanishes as they equilibrate). Each
        size runs its own sampler, burn-in included, so the cost is about
        ``len(sizes)`` times that of one
        :func:`~sparsegf2.erasures.sample_regions` call plus ``c S len(sizes)``
        failure evaluations of two eliminations of ``2 min(m, n - m)`` columns
        of ``n / 32`` words each. ``failure_ess`` gives each entry's effective
        number of independent records, for error bars.

        **Seeds.** One ``seed`` drives every size: the regions of size ``m``
        are drawn with the sequence seed ``[seed, m]`` (a list seed's entries
        followed by ``m``), so they are exactly
        ``erasures.sample_regions(edges, n, m, couplings, samples, ...,
        seed=[seed, m])``, and the sizes ``m >= 1`` of one seed use different,
        independent streams. (numpy's ``SeedSequence`` pads with zeros, so
        ``[seed, 0]`` is the stream of ``seed`` itself; size 0 draws nothing.)
        ``seed=None`` draws a fresh seed, reported as ``seed``.

        A code without logical qubits gives 0 everywhere (its regions are
        still drawn, so the diagnostics are filled in). The chain does not
        depend on the code: its diagnostics are those of the sampler on the
        graph alone. Returns a :class:`CorrelatedFailure`.
        """
        n = self._n
        e = graphs.from_edges(edges, n)
        given = erasures._couplings(couplings, distinct=True)
        m_list = self._sizes(sizes)
        samples = erasures._integer(samples, "samples", 1)
        used = erasures._seed(seed)
        base = [used] if isinstance(used, int) else list(used)
        c, s = len(given), len(m_list)
        requested = np.sort(given)
        failures = np.zeros((c, s), dtype=np.int64)
        ess = np.full((c, s), float(samples))
        mean_boundary = np.empty((c, s))
        acceptance = np.empty((c, s))
        exchange = np.full((c - 1, s), np.nan)
        weakest = np.full(s, np.nan)
        rungs = np.zeros(s, dtype=np.int64)
        notes: list[tuple[int, str]] = []
        method_used = geometry = ""
        used_burn_in = 0
        exchanged = False
        for j, m in enumerate(m_list.tolist()):
            drawn = erasures.sample_regions(
                e,
                n,
                m,
                given,
                samples,
                burn_in=burn_in,
                thin=thin,
                tempering=tempering,
                seed=[*base, m],
                method=method,
                ladder=ladder,
                max_step=max_step,
            )
            if self.num_logical > 0:
                flat = np.array(drawn.regions.reshape(c * samples, m))
                failed = np.zeros(c * samples, dtype=np.uint8)
                self._fill_failures(flat, failed)
                indicators = failed.reshape(c, samples)
                failures[:, j] = indicators.sum(axis=1)
                if drawn.method == "chain" and samples >= 2:
                    for i in range(c):
                        series = indicators[i].astype(np.float64)
                        # capped at the records: a short series can get tau < 1 from noise
                        # (chain_diagnostics floors it only at 1 / log10(S)), which would
                        # shrink the error bar below the binomial one
                        ess[i, j] = min(erasures.chain_diagnostics(series)["ess"], float(samples))
            mean_boundary[:, j] = drawn.boundaries.mean(axis=1)
            acceptance[:, j] = drawn.acceptance
            pairs = drawn.exchange_acceptance
            place = np.searchsorted(drawn.ladder, requested)
            for i in range(c - 1):
                between = pairs[place[i] : place[i + 1]]
                finite = between[np.isfinite(between)]
                exchange[i, j] = finite.min() if finite.size else np.nan
            if np.isfinite(pairs).any():
                weakest[j] = pairs[np.isfinite(pairs)].min()
            rungs[j] = drawn.ladder.size
            notes.extend((m, note) for note in drawn.warnings)
            method_used, geometry = drawn.method, drawn.geometry
            used_burn_in = drawn.burn_in  # the same for every size (it depends on n only)
            exchanged = exchanged or drawn.tempering
        return CorrelatedFailure(
            couplings=given.copy(),
            sizes=m_list,
            samples=samples,
            failure=failures / samples,
            failures=failures,
            failure_ess=ess,
            mean_boundary=mean_boundary,
            acceptance=acceptance,
            exchange_pairs=np.stack([requested[:-1], requested[1:]], axis=1),
            exchange_acceptance=exchange,
            min_exchange_acceptance=weakest,
            rungs=rungs,
            method=method_used,
            geometry=geometry,
            warnings=_merge_warnings(notes),
            size_warnings=tuple(notes),
            seed=used,
            burn_in=used_burn_in,
            thin=thin,
            tempering=bool(exchanged),
            num_qubits=n,
            num_logical=self.num_logical,
        )

    def boundary_failure_counts(self, edges, max_size: int, *, budget=None) -> BoundaryCounts:
        """Count every region of size ``m = 1..max_size`` by its edge boundary, and the failing ones.

        For the graph ``edges`` on this code's ``n`` qubits (``(E, 2)`` as in
        :mod:`sparsegf2.graphs`), ``b(A)`` is the number of edges with exactly
        one end in ``A``. Every ``m``-subset with ``1 <= m <= w = max_size`` is
        listed once (depth first, in lexicographic order) with its boundary,
        and its failure ``F(A) = 1[l(A) > 0]`` is computed incrementally along
        the listing (each region adds its last qubit's two columns to two GF(2)
        bases kept along the branch; once a set fails, so do all its supersets
        in the listing, and their ranks are skipped). The result
        (:class:`BoundaryCounts`) holds::

            total[m - 1, b]  = N_m(b) = #{A : |A| = m, b(A) = b}
            failed[m - 1, b] = U_m(b) = #{A : |A| = m, b(A) = b, F(A) = 1}

        from which, exactly and for every coupling ``J`` at once
        (:meth:`BoundaryCounts.exact`)::

            P_J(m) = sum_b U_m(b) exp(-J b) / sum_b N_m(b) exp(-J b)

        ``J = 0`` gives ``uncorrectable_counts(max_size) / C(n, m)``, and
        ``J -> +inf`` the failing fraction of the regions of smallest boundary.
        The number of regions ``sum_{m=1}^{w} C(n, m)`` may not exceed
        ``budget`` (default :data:`sparsegf2.erasures.ENUMERATION_BUDGET`, two
        million); a larger count is refused with the number and the largest
        ``max_size`` that fits (:func:`sparsegf2.analysis.exhaustive_cutoff`).
        A code without logical qubits has ``failed`` all zero.
        """
        n = self._n
        e = np.ascontiguousarray(graphs.from_edges(edges, n), dtype=np.int64)
        w = _bounded(max_size, "max_size", 1, n)
        _check_region_budget(n, w, budget)
        indptr, indices = _chain.csr(n, e)
        columns = int(_fail.boundary_columns(indptr, w))  # holds every b(A), |A| <= w
        failed = np.zeros((w, columns), dtype=np.int64)
        total = np.zeros((w, columns), dtype=np.int64)
        _fail.boundary_failure_counts(
            self._tab, self._ws, self._marked, indptr, indices, n, w, failed, total
        )
        width = int(np.flatnonzero(total.any(axis=0))[-1]) + 1
        return BoundaryCounts(
            failed=failed[:, :width].copy(),
            total=total[:, :width].copy(),
            max_size=w,
            num_qubits=n,
            num_logical=self.num_logical,
        )

    def window_first_failure(self, order=None, *, periodic: bool = True) -> np.ndarray:
        """For every start ``s``, the length of the shortest uncorrectable contiguous window from it.

        ``order`` is a permutation of the qubits (default ``0..n-1``). The
        window of length ``L`` at start ``s`` is ``order[s], order[s+1], ...,
        order[s+L-1]``, read cyclically when ``periodic`` (``L <= n``) and
        stopping at the end of the order otherwise (``L <= n - s``). Returns
        int64 of length ``n``: entry ``s`` is the smallest ``L`` whose window
        loses logical information (``l > 0``), or the largest possible length
        plus one when none does (``n + 1`` on the ring, ``n - s + 1`` on the
        open order). Windows from one start are nested and failure is
        monotone, so the window of length ``m`` at ``s`` fails exactly when
        ``L(s) <= m``. The contiguous distance (:meth:`contiguous_distance`)
        is the smallest ``L(s)`` among the starts from which some window fails
        (``L(s) <= n`` on the ring, ``L(s) <= n - s`` on the open order). One
        incremental scan per start, ``O(n)`` column additions each.
        """
        seq = self._order(order)
        out = np.zeros(self._n, dtype=np.int64)
        _fail.window_first_failure(self._tab, self._ws, self._marked, seq, bool(periodic), out)
        return out

    def interval_failure_by_size(self, order=None, *, periodic: bool = True) -> np.ndarray:
        """The failing fraction of the windows of each length: the ring's J -> inf limit with periodic=True only.

        Entry ``m`` (float64, length ``n + 1``) is ``#{failing windows of
        length m} / #{windows of length m}`` along ``order`` (default
        ``0..n-1``): the ``n`` windows ``order[s..s+m-1]`` read cyclically when
        ``periodic``, the ``n - m + 1`` windows inside the order otherwise
        (from :meth:`window_first_failure`: the window at ``s`` fails exactly
        when ``L(s) <= m``). Entry 0 is 0 and entry ``n`` is ``1[k > 0]``.

        **The ring's J -> +inf limit.** On the ring ``C_n`` (``graphs.cycle(n)``
        with the default order, or any cycle listed along ``order``,
        ``n >= 3``), the regions of size ``1 <= m <= n-1`` with the smallest
        boundary (``b = 2``) are exactly the ``n`` intervals, so ``Q_J(. | m)``
        tends to the uniform law on them and, exactly,
        ``lim_{J -> +inf} P_J(m)`` is this entry with ``periodic=True``.

        On an **open path** the ``J -> +inf`` limit is not this curve: the
        smallest boundary (``b = 1``) belongs only to the two intervals that
        touch an end, so the limit is
        ``(1[L_first <= m] + 1[L_last <= m]) / 2`` with ``L_first`` and
        ``L_last`` the first-failure lengths of the prefix and of the suffix
        (:meth:`end_window_failure_by_size`). With ``periodic=False`` this
        method gives the uniform law on all intervals of the open order, which
        is NOT the open path's ``J -> +inf`` limit (that is the two end windows).
        """
        n = self._n
        first = self.window_first_failure(order, periodic=periodic)
        out = np.zeros(n + 1)
        sizes = np.arange(1, n + 1)
        if periodic:
            failing = np.cumsum(np.bincount(first, minlength=n + 2))[1 : n + 1]  # L(s) <= m
            out[1:] = failing / n
        else:
            starts = np.arange(n)
            inside = starts[None, :] <= n - sizes[:, None]
            failing = np.count_nonzero(inside & (first[None, :] <= sizes[:, None]), axis=1)
            out[1:] = failing / (n - sizes + 1)
        return out

    def end_window_failure_by_size(self, order=None) -> np.ndarray:
        """The open path's ``J -> +inf`` limit of ``P_J(m)``: the mean failure of its two end windows.

        On the path laid along ``order`` (default ``0..n-1``; ``graphs.path(n)``
        for the default) the regions of size ``1 <= m <= n-1`` with the
        smallest boundary (``b = 1``) are only the prefix ``order[:m]`` and the
        suffix ``order[n-m:]``, so ``Q_J(. | m)`` tends to the uniform law on
        those two and, exactly::

            lim_{J -> +inf} P_J(m) = (1[L_first <= m] + 1[L_last <= m]) / 2

        with ``L_first`` (``L_last``) the smallest length of a failing prefix
        (suffix). Returns float64 of length ``n + 1``; entry 0 is 0 and entry
        ``n`` is ``1[k > 0]``. One incremental scan from each end.
        """
        seq = self._order(order)
        sizes = np.arange(self._n + 1)
        first = self._prefix_first_failure(seq)
        last = self._prefix_first_failure(seq[::-1].copy())
        return ((first <= sizes).astype(np.float64) + (last <= sizes)) / 2

    def _prefix_first_failure(self, seq: np.ndarray) -> int:
        lost = np.zeros(self._n + 1, dtype=np.int64)
        _ent.erasure_profile_rows(self._tab, self._ws, seq, self._marked, lost)
        failing = np.flatnonzero(lost > 0)
        return int(failing[0]) if failing.size else self._n + 1

    def _order(self, order) -> np.ndarray:
        n = self._n
        seq = np.arange(n, dtype=np.int64) if order is None else np.asarray(order)
        if (
            seq.shape != (n,)
            or not np.issubdtype(seq.dtype, np.integer)
            or not np.array_equal(np.sort(seq), np.arange(n))
        ):
            raise ValueError(f"order must be a permutation of 0..{n - 1}")
        return np.ascontiguousarray(seq, dtype=np.int64)

    def _sizes(self, sizes) -> np.ndarray:
        n = self._n
        if sizes is None:
            m_list = np.arange(1, n, dtype=np.int64)
        else:
            raw = np.asarray(sizes)
            if raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)):
                raise ValueError(
                    f"sizes must be a list of region sizes (integers in [0, {n}]), got {sizes!r}"
                )
            m_list = raw.astype(np.int64)
            bad = m_list[(m_list < 0) | (m_list > n)]
            if bad.size:
                raise ValueError(f"sizes must lie in [0, {n}] (the number of qubits), got {bad[0]}")
            if np.unique(m_list).size != m_list.size:
                raise ValueError(f"sizes must be distinct, got {sizes!r}")
        if m_list.size == 0:
            raise ValueError(
                f"give at least one region size (in [0, {n}]); a code on one qubit has no "
                "size strictly between 0 and n, so pass sizes explicitly"
            )
        return m_list

    def _regions(self, regions) -> np.ndarray:
        n = self._n
        try:
            arr = np.asarray(regions)
        except ValueError:
            arr = np.empty(0, dtype=object)
        if arr.ndim != 2 or (arr.size and not np.issubdtype(arr.dtype, np.integer)):
            raise ValueError(
                "regions must be a 2-D integer array (or a list of lists), one region per row, "
                "all of the same size"
            )
        a = np.array(arr, dtype=np.int64, order="C")  # a writable copy (one kernel signature)
        if a.size and (np.min(a) < 0 or np.max(a) >= n):
            raise ValueError(f"qubit indices must be in [0, {n})")
        if a.shape[1] > 1 and np.any(np.diff(np.sort(a, axis=1), axis=1) == 0):
            raise ValueError("the qubits of each region must be distinct")
        return a

    def _qubits(self, qubits: Iterable[int]) -> np.ndarray:
        a = np.array([int(q) for q in qubits], dtype=np.int64)
        if a.size and (np.min(a) < 0 or np.max(a) >= self._n):
            raise ValueError(f"qubit indices must be in [0, {self._n})")
        if np.unique(a).size != a.size:
            raise ValueError("qubit indices must be distinct")
        return a


def _bit_rows(stabilizers, num_qubits: int | None) -> np.ndarray:
    """Generators as an ``(s, 2n)`` uint8 array from bit rows or Pauli strings."""
    if isinstance(stabilizers, (str, PauliString)):
        stabilizers = [stabilizers]  # one generator, not one per character
    if isinstance(stabilizers, np.ndarray) and stabilizers.dtype != object:
        rows = np.asarray(stabilizers)
    else:
        items = list(stabilizers)
        if items and all(isinstance(p, (str, PauliString)) for p in items):
            paulis = [p if isinstance(p, PauliString) else _pauli_text(p) for p in items]
            n = len(paulis[0])
            if any(len(p) != n for p in paulis):
                raise ValueError("all stabilizers must act on the same number of qubits")
            if num_qubits is not None and n != num_qubits:
                raise ValueError(
                    f"the stabilizers act on {n} qubits, not num_qubits = {num_qubits}"
                )
            return np.array([np.concatenate([p.xs, p.zs]) for p in paulis], dtype=np.uint8).reshape(
                len(paulis), 2 * n
            )
        rows = np.asarray(items)
    if rows.size == 0:
        if num_qubits is None and rows.ndim == 2 and rows.shape[1] > 0 and rows.shape[1] % 2 == 0:
            num_qubits = rows.shape[1] // 2  # an (0, 2n) array still says n
        if num_qubits is None:
            raise ValueError("give num_qubits for a code without stabilizers")
        return np.zeros((0, 2 * int(num_qubits)), dtype=np.uint8)
    if rows.ndim != 2 or rows.shape[1] % 2 or not np.isin(rows, (0, 1)).all():
        raise ValueError("stabilizers must be an (s, 2n) array of 0/1 (X bits, then Z bits)")
    if num_qubits is not None and rows.shape[1] != 2 * num_qubits:
        raise ValueError(f"rows have {rows.shape[1] // 2} qubits, not num_qubits = {num_qubits}")
    return rows.astype(np.uint8)


def _pauli_text(text: str) -> PauliString:
    """A stabilizer written as text: a leading sign is dropped and the letters upper-cased,
    so a lowercase identity ``i`` is never read as a phase."""
    return PauliString(text.strip().lstrip("+-").upper())


def _pack(rows: np.ndarray) -> np.ndarray:
    s, width = rows.shape
    n = width // 2
    wsys = (n + 63) // 64
    out = np.zeros((s, 2 * wsys), dtype=np.uint64)
    for half in range(2):
        for q in np.flatnonzero(rows[:, half * n : (half + 1) * n].any(axis=0)):
            bit = np.uint64(1) << np.uint64(int(q) & 63)
            out[rows[:, half * n + q].astype(bool), half * wsys + (int(q) >> 6)] |= bit
    return out


def _unpack(packed: np.ndarray, n: int) -> np.ndarray:
    wsys = (n + 63) // 64
    out = np.zeros((packed.shape[0], 2 * n), dtype=np.uint8)
    for half in range(2):
        for q in range(n):
            word = packed[:, half * wsys + (q >> 6)]
            out[:, half * n + q] = (word >> np.uint64(q & 63)) & np.uint64(1)
    return out
