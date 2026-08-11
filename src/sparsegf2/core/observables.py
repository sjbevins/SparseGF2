"""Phase-free stabilizer observables for :class:`SparseGF2`.

This module is the **physics deliverable** of the core simulator: the
MIPT order parameters and structural diagnostics, computed directly
from the symplectic tableau.

Available observables
=====================

============================= ================================================
``subsystem_rank``            ``rank_{F_2}([X|Z]|_A)``, the dimension of the
                              local stabilizer subgroup on subsystem ``A``.
``entanglement_entropy``      ``S(A) = rank - |A|``, the Fattal–Cubitt–Yamamoto–
                              Bravyi–Chuang formula (`quant-ph/0406168`_).
``mutual_information``        ``I(A:B) = S(A) + S(B) - S(A ∪ B)``.
``tripartite_mutual_info``    ``I_3(A:B:C) = I(A:B) + I(A:C) - I(A:B ∪ C)``.
``code_dimension``            Number of *purified* logical qubits in an
                              explicit purification setup. The partition is
                              an explicit argument, with no hidden ``n // 2``.
``generator_weights``         Array of Hamming weights for all generators.
``stabilizer_weight_spectrum`` Histogram of stabilizer weights, the
                              area-vs-volume law diagnostic.
============================= ================================================

Phase-free contract
===================

All of the above are functions of the **stabilizer subspace** alone (the
row span of rows ``n..2n-1`` of ``[X | Z]``). The simulator's
deliberately-dropped sign bits do not affect any of them; see the master
notebook §2.3 for the formal statement of the contract.

References
==========

.. _quant-ph/0406168: https://arxiv.org/abs/quant-ph/0406168

Each observable's docstring cites the specific source it implements; the
collected list:

* Aaronson & Gottesman, *Improved Simulation of Stabilizer Circuits*,
  Phys. Rev. A **70**, 052328 (2004),
  `quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_,
  the symplectic tableau these observables read.
* Fattal, Cubitt, Yamamoto, Bravyi, Chuang, *Entanglement in the
  Stabilizer Formalism*, `quant-ph/0406168`_ (2004),
  the entanglement-entropy formula ``S(A) = rank - |A|``.
* Groisman, Popescu, Winter, *Quantum, classical, and total amount of
  correlations in a quantum state*, Phys. Rev. A **72**, 032317 (2005),
  `quant-ph/0410091 <https://arxiv.org/abs/quant-ph/0410091>`_,
  the operational meaning of the mutual information.
* Hosur, Qi, Roberts, Yoshida, *Chaos in quantum channels*,
  JHEP **02** (2016) 004,
  `arXiv:1511.04021 <https://arxiv.org/abs/1511.04021>`_,
  the tripartite information as a scrambling diagnostic.
* Gullans & Huse, *Dynamical Purification Phase Transition Induced by
  Quantum Measurements*, Phys. Rev. X **10**, 041020 (2020),
  `arXiv:1905.05195 <https://arxiv.org/abs/1905.05195>`_,
  the purification picture and code dimension.
* Gullans & Huse, *Scalable Probes of Measurement-Induced Criticality*,
  Phys. Rev. Lett. **125**, 070606 (2020),
  `arXiv:1910.00020 <https://arxiv.org/abs/1910.00020>`_,
  the (contiguous) code distance.
* Choi, Bao, Qi, Altman, *Quantum Error Correction in Scrambling
  Dynamics and Measurement-Induced Phase Transition*,
  Phys. Rev. Lett. **125**, 030505 (2020),
  `arXiv:1903.05124 <https://arxiv.org/abs/1903.05124>`_,
  the error-correcting-code framing of the monitored dynamics.
* Zabalo, Gullans, Wilson, Gopalakrishnan, Huse, Pixley, *Critical
  properties of the measurement-induced transition in random quantum
  circuits*, Phys. Rev. B **101**, 060301(R) (2020),
  `arXiv:1911.00008 <https://arxiv.org/abs/1911.00008>`_,
  the tripartite information as the MIPT order parameter.
* Schumacher & Nielsen, *Quantum data processing and error correction*,
  Phys. Rev. A **54**, 2629 (1996),
  `quant-ph/9604022 <https://arxiv.org/abs/quant-ph/9604022>`_,
  erasure correctability ``<=> I(A:R) = 0`` (decoupling).
* Li, Chen, Fisher, *Measurement-driven entanglement transition in
  hybrid quantum circuits*, Phys. Rev. B **100**, 134306 (2019),
  `arXiv:1901.08092 <https://arxiv.org/abs/1901.08092>`_,
  the stabilizer-weight (area- vs volume-law) diagnostic.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from sparsegf2.core._protocol import SimulatorProtocol, SubsystemFastExtractor

# Bit-packed elimination: identical results to ``gf2_rank``, ~7-27x faster at
# the (n, 2|A|) shapes the observables produce (and never slower, even tiny).
# Load-bearing for ``until_purified`` runs, which take a rank per measured layer.
from sparsegf2.core.linalg_gf2 import gf2_rank_bits as _gf2_rank
from sparsegf2.errors import InvalidArgumentError, TableauCorruption

# Public type alias for the subsystem argument accepted by every
# observable. Locked in here so future packages can write
# ``def my_observable(sim, A: Subsystem) -> int: ...`` and stay forward-
# compatible with any widening of the accepted forms.
#
# Currently: any iterable of integers (list / tuple / range / ndarray of
# ints). Forward-compatible: a future widening to slice / boolean mask
# will live behind this alias. Uses the PEP 695 ``type`` syntax (Python
# 3.12+, our minimum supported version).
type Subsystem = Iterable[int]


def complement(n: int, subsystem: Subsystem) -> NDArray[np.int64]:
    """Return the complement of ``subsystem`` within ``range(n)``.

    Convenience for ``np.setdiff1d(np.arange(n), subsystem)`` with input
    validation. Returns the qubits in ``[0, n)`` *not* in ``subsystem``
    as a sorted int64 ndarray.

    Parameters
    ----------
    n : int
        Total qubit count. Must be non-negative.
    subsystem : Subsystem
        Indices to exclude. Duplicates and out-of-range entries raise.

    Returns
    -------
    ndarray of int64
        Sorted complement.

    Examples
    --------
    >>> complement(4, [1, 3]).tolist()
    [0, 2]
    >>> complement(4, []).tolist()
    [0, 1, 2, 3]

    Raises
    ------
    InvalidArgumentError
        If ``n < 0`` or ``subsystem`` has duplicates / out-of-range entries.
    """
    if n < 0:
        raise InvalidArgumentError(f"n must be non-negative, got {n}")
    A = _validate_subsystem(subsystem, n)
    return np.setdiff1d(np.arange(n, dtype=np.int64), A, assume_unique=True)


# (The ``isinstance(sim, _SparseGF2)`` fast-path coupling that lived
# here previously has been promoted to a public optional protocol,
# :class:`SubsystemFastExtractor` in :mod:`sparsegf2.core._protocol`,
# so alternative backends can opt into the fast path without observables
# duck-typing on attribute names.)

# ----------------------------------------------------------------------
# Subsystem-rank machinery
# ----------------------------------------------------------------------


def _validate_subsystem(subsystem: Iterable[int], n: int) -> NDArray[np.int64]:
    """Coerce ``subsystem`` to a sorted, dedup'd int64 array in ``[0, n)``."""
    arr = np.asarray(list(subsystem), dtype=np.int64)
    if arr.size == 0:
        return arr
    if arr.min() < 0 or arr.max() >= n:
        raise InvalidArgumentError(
            f"subsystem entries must be in [0, n={n}); got min={arr.min()}, max={arr.max()}"
        )
    unique = np.unique(arr)
    if unique.shape[0] != arr.shape[0]:
        raise InvalidArgumentError(f"subsystem must not have duplicates: got {arr.tolist()}")
    return unique


def subsystem_rank(sim: SimulatorProtocol, subsystem: Iterable[int]) -> int:
    r"""Return :math:`\mathrm{rank}_{\mathbb{F}_2}([X|Z]\big|_A)` where ``A = subsystem``.

    The stabilizer block (rows ``n..2n-1``) of the symplectic matrix
    is restricted to the column set ``A ∪ (A + n)`` (the x-bit columns
    for ``A``, then the z-bit columns for ``A``) and the GF(2) rank of
    the restricted matrix is returned. Column order does not affect
    rank, so the layout is `[x_A, z_A]` for implementation convenience.

    This is the load-bearing primitive: every observable in this module
    is a thin wrapper around :func:`subsystem_rank`.

    Parameters
    ----------
    sim
        A simulator satisfying :class:`SimulatorProtocol` (e.g. a
        :class:`sparsegf2.SparseGF2` instance).
    subsystem
        Iterable of qubit indices in ``[0, sim.n)``. Order doesn't
        matter; duplicates raise.

    Returns
    -------
    int
        Non-negative integer. The loose envelope is ``[0, min(n, 2|A|)]``
        from the matrix shape; for a **pure** stabilizer state the
        Fattal et al. contract sharpens this to ``[|A|, min(2|A|, n + |A|)]``,
        and :func:`entanglement_entropy` relies on the lower bound to
        guarantee ``S(A) >= 0``.

    Raises
    ------
    ValueError
        If ``subsystem`` has duplicates or out-of-range entries.
    """
    n = sim.n
    A = _validate_subsystem(subsystem, n)
    return _subsystem_rank_arr(sim, A, n)


def _subsystem_rank_arr(sim: SimulatorProtocol, A: NDArray[np.int64], n: int) -> int:
    """Internal subsystem_rank that takes a *validated* int64 array.

    Skips :func:`_validate_subsystem` to avoid double work when called
    from internal chains (``entanglement_entropy``, ``mutual_information``,
    ``tripartite_mutual_info``) that have already validated their inputs.
    Public callers should use :func:`subsystem_rank`.
    """
    if A.shape[0] == 0:
        return 0
    # Build the (n, 2|A|) stabilizer-restricted matrix. Backends that
    # implement the optional :class:`SubsystemFastExtractor` protocol
    # (any :class:`SparseGF2` does) deliver this slice directly and
    # avoid the ``(2n, 2n)`` allocation ``to_symplectic()`` would make.
    # Protocol-only backends go through ``to_symplectic()``.
    if isinstance(sim, SubsystemFastExtractor):
        sub = sim._subsystem_xz_block(A)
    else:
        full = sim.to_symplectic()
        cols = np.concatenate([A, A + n])
        sub = full[n:, cols].astype(np.uint8) & 1
    return _gf2_rank(sub)


def _entropy_from_validated(sim: SimulatorProtocol, A: NDArray[np.int64], n: int) -> int:
    """Entropy for a validated subsystem, using the smaller pure-state side."""
    if A.shape[0] == 0 or A.shape[0] == n:
        return 0
    if A.shape[0] > n // 2:
        mask = np.ones(n, dtype=np.bool_)
        mask[A] = False
        A = np.flatnonzero(mask).astype(np.int64, copy=False)
    return int(_subsystem_rank_arr(sim, A, n) - A.shape[0])


# ----------------------------------------------------------------------
# Entropy + mutual information
# ----------------------------------------------------------------------


def entanglement_entropy(sim: SimulatorProtocol, subsystem: Iterable[int]) -> int:
    r"""Entanglement entropy of subsystem ``A`` in units of $\log_2$.

    Fattal–Cubitt–Yamamoto–Bravyi–Chuang 2004 (`quant-ph/0406168`_):

    .. math::

        S(A) = \mathrm{rank}_{\mathbb{F}_2}\bigl([X|Z]\big|_A\bigr) - |A|.

    The right-hand side is an integer; phase-free, so the sign bit
    cannot change it.

    Parameters
    ----------
    sim
        A :class:`SparseGF2` instance.
    subsystem
        Iterable of qubit indices in ``[0, sim.n)``. Duplicates raise.

    Returns
    -------
    int
        Non-negative integer in ``[0, min(|A|, sim.n - |A|)]`` for any
        pure state. Returns ``0`` when ``A`` is empty.

    Raises
    ------
    ValueError
        If ``subsystem`` has duplicate or out-of-range indices.
    """
    n = sim.n
    A = _validate_subsystem(subsystem, n)
    if A.shape[0] == 0:
        return 0
    s = int(_subsystem_rank_arr(sim, A, n) - A.shape[0])
    # Sanity: for a *pure* stabilizer state the FCYBC formula guarantees
    # S(A) ∈ [0, min(|A|, n - |A|)]. Out-of-range results imply
    # tableau corruption (the kernels maintain rank-n stabilizer
    # subgroups on n qubits by construction); catch them here, not in
    # downstream MIPT analyses that would compute nonsensical phases.
    upper = min(A.shape[0], n - A.shape[0])
    if s < 0 or s > upper:
        raise TableauCorruption(
            f"entanglement_entropy: FCYBC contract violated: got S(|A|={A.shape[0]}) = {s}, "
            f"expected in [0, {upper}]; likely tableau corruption"
        )
    return s


def mutual_information(sim: SimulatorProtocol, A: Iterable[int], B: Iterable[int]) -> int:
    r"""Mutual information $I(A:B) = S(A) + S(B) - S(A \cup B)$.

    Parameters
    ----------
    sim
        A :class:`SparseGF2` instance.
    A, B
        Disjoint iterables of qubit indices in ``[0, sim.n)``.

    Returns
    -------
    int
        Non-negative integer (under unitary evolution; phase-free).

    Raises
    ------
    ValueError
        If ``A ∩ B ≠ ∅`` or either has out-of-range / duplicate indices.
    """
    n = sim.n
    Aarr = _validate_subsystem(A, n)
    Barr = _validate_subsystem(B, n)
    if np.intersect1d(Aarr, Barr).size:
        raise InvalidArgumentError("A and B must be disjoint")
    # I(A:B) = S(A) + S(B) - S(A∪B). With disjoint A, B the size terms
    # |A|+|B| = |AB| cancel, so we only need the three ranks.
    AB = Aarr if Barr.size == 0 else Barr if Aarr.size == 0 else np.concatenate([Aarr, Barr])
    rA = _subsystem_rank_arr(sim, Aarr, n)
    rB = _subsystem_rank_arr(sim, Barr, n)
    rAB = _subsystem_rank_arr(sim, AB, n)
    return int(rA + rB - rAB)


def tripartite_mutual_info(
    sim: SimulatorProtocol,
    A: Iterable[int],
    B: Iterable[int],
    C: Iterable[int],
) -> int:
    r"""Tripartite mutual information $I_3(A:B:C) = I(A:B) + I(A:C) - I(A:B \cup C)$.

    A canonical MIPT order parameter: negative in the volume-law
    phase, near-zero in the area-law phase, scales with the boundary
    of ``A`` at the critical point. Introduced as a scrambling measure
    by Hosur, Qi, Roberts & Yoshida, *Chaos in quantum channels*,
    JHEP 02 (2016) 004 (`arXiv:1511.04021`_); used as the
    measurement-induced-transition order parameter / data-collapse
    variable by Zabalo et al., Phys. Rev. B **101**, 060301(R) (2020)
    (`arXiv:1911.00008`_).

    .. _arXiv:1511.04021: https://arxiv.org/abs/1511.04021
    .. _arXiv:1911.00008: https://arxiv.org/abs/1911.00008

    Parameters
    ----------
    sim
        A simulator satisfying :class:`SimulatorProtocol`.
    A, B, C
        Pairwise-disjoint iterables of qubit indices in ``[0, sim.n)``.

    Returns
    -------
    int
        :math:`I_3 \in \mathbb{Z}`, which can be negative (volume-law),
        zero (area-law product), or positive (rare for stabilizer
        states; happens at the MIPT critical point).

    Raises
    ------
    ValueError
        If any of ``A ∩ B``, ``A ∩ C``, ``B ∩ C`` is non-empty, or
        if any of ``A``, ``B``, ``C`` has duplicate / out-of-range
        indices.
    """
    return _tripartite_mutual_info_and_entropy_ab(sim, A, B, C)[0]


def _tripartite_mutual_info_and_entropy_ab(
    sim: SimulatorProtocol,
    A: Iterable[int],
    B: Iterable[int],
    C: Iterable[int],
) -> tuple[int, int]:
    """Return ``(I3(A:B:C), S(A union B))`` with shared rank work.

    This internal helper underpins :func:`tripartite_mutual_info` and the
    dynamics studies that also record the half-cut entropy ``S(A union B)``.
    The former implementation evaluated three mutual informations separately,
    repeating the rank of ``A`` three times.  Here each of the seven distinct
    entropies is evaluated once.  Pure-state complementarity is used whenever a
    union is larger than half the system, reducing matrix width without changing
    the integer observable.
    """
    n = sim.n
    Aarr = _validate_subsystem(A, n)
    Barr = _validate_subsystem(B, n)
    Carr = _validate_subsystem(C, n)
    if np.intersect1d(Aarr, Barr).size:
        raise InvalidArgumentError("A and B must be disjoint")
    if np.intersect1d(Aarr, Carr).size:
        raise InvalidArgumentError("A and C must be disjoint")
    if np.intersect1d(Barr, Carr).size:
        raise InvalidArgumentError("B and C must be disjoint")

    AB = np.concatenate([Aarr, Barr])
    AC = np.concatenate([Aarr, Carr])
    BC = np.concatenate([Barr, Carr])
    ABC = np.concatenate([Aarr, Barr, Carr])
    sA = _entropy_from_validated(sim, Aarr, n)
    sB = _entropy_from_validated(sim, Barr, n)
    sC = _entropy_from_validated(sim, Carr, n)
    sAB = _entropy_from_validated(sim, AB, n)
    sAC = _entropy_from_validated(sim, AC, n)
    sBC = _entropy_from_validated(sim, BC, n)
    sABC = _entropy_from_validated(sim, ABC, n)
    return sA + sB + sC - sAB - sAC - sBC + sABC, sAB


# ----------------------------------------------------------------------
# Code dimension
# ----------------------------------------------------------------------


def code_dimension(sim: SimulatorProtocol, n_system: int) -> int:
    r"""System-half entanglement entropy in the MIPT purification picture.

    For a stabilizer state on ``2 * n_system`` qubits (typically
    prepared via :func:`sparsegf2.from_bell_purification(n_system)`),
    returns the entanglement entropy of the system half:

    .. math::

        k \;\equiv\; S(\text{system})
          \;=\; \mathrm{rank}_{\mathbb{F}_2}\bigl([X|Z]\big|_{\text{system}}\bigr)
                - n_{\text{system}}.

    Reading: ``k`` counts the system qubits that remain entangled with
    the reference (the "uncollapsed" logical qubits). For a fresh Bell
    purification, ``k == n_system`` (fully entangled, no purification
    yet). After enough measurements, ``k → 0`` (system fully purified).
    The **purification time** is the MIPT order parameter of Gullans
    & Huse 2020.

    This is *literally* ``entanglement_entropy(sim, range(n_system))``;
    the alternate name is kept for readability at purification-picture
    call sites. The partition is an explicit argument (rather than an
    assumed ``sim.n // 2``), so the purification picture is never implicit.

    Parameters
    ----------
    sim
        A :class:`SparseGF2` on ``2 * n_system`` qubits in an explicit
        purification picture.
    n_system
        Number of system qubits. Must satisfy ``2 * n_system == sim.n``.

    Returns
    -------
    int
        :math:`S(\text{system}) \in [0, n_{\text{system}}]`.

    See Also
    --------
    entanglement_entropy : the general subsystem entropy.
    """
    if n_system < 0:
        raise InvalidArgumentError(f"n_system must be non-negative, got {n_system}")
    if 2 * n_system != sim.n:
        raise InvalidArgumentError(
            f"code_dimension expects sim.n == 2 * n_system; got sim.n={sim.n}, n_system={n_system}"
        )
    return entanglement_entropy(sim, range(n_system))


def code_rate(sim: SimulatorProtocol, n_system: int) -> float:
    r"""Encoding rate ``k / n`` of the purification code.

    ``k = code_dimension(sim, n_system)`` is the number of system qubits still
    entangled with the reference (the protected logical qubits) and ``n`` is
    ``n_system``, so the rate is :math:`R = k / n \in [0, 1]`. It starts at
    ``1`` for a fresh Bell purification and falls to ``0`` as the monitored
    dynamics purifies the system. Returns ``0.0`` for ``n_system == 0``.

    The rate ``k / n`` of an error-correcting code is standard (Nielsen &
    Chuang, ch. 10); as an intensive order parameter of the monitored
    dynamics it follows Gullans & Huse, Phys. Rev. X **10**, 041020 (2020)
    and Choi et al., Phys. Rev. Lett. **125**, 030505 (2020).
    """
    if n_system == 0:
        return 0.0
    return code_dimension(sim, n_system) / n_system


def _eliminate_on_columns(
    mat: NDArray[np.uint8], cols: NDArray[np.int64]
) -> tuple[NDArray[np.uint8], int]:
    """Forward GF(2) elimination of ``mat`` using ``cols`` as pivot columns
    (left to right), applying every row operation across all of ``mat``.

    Returns a reduced copy and the rank ``r`` obtained on ``cols``: rows
    ``0 .. r-1`` carry the pivots (and span the column space of ``cols``), and
    rows ``r ..`` are zero in every column of ``cols``. Used by
    :func:`contiguous_distance` to factor a fixed reference subsystem out of the
    stabilizer block once, so each window costs two small ranks.
    """
    m = mat.copy()
    r = 0
    n_rows = m.shape[0]
    for c in cols:
        nz = np.flatnonzero(m[r:, c])
        if nz.size == 0:
            continue
        piv = r + int(nz[0])
        if piv != r:
            m[[r, piv]] = m[[piv, r]]
        hits = m[:, c].astype(bool).copy()
        hits[r] = False
        if hits.any():
            m[hits] ^= m[r]
        r += 1
        if r == n_rows:
            break
    return m, r


def contiguous_distance(
    sim: SimulatorProtocol,
    n_system: int,
    *,
    reference: Iterable[int] | None = None,
    periodic: bool = False,
) -> int | None:
    r"""Smallest contiguous system region that carries protected logical info.

    For a purification code, a *contiguous* region :math:`A` of the system
    "supports a logical operator" (equivalently, is not erasure-correctable)
    exactly when it shares mutual information with the reference,
    :math:`I(A : R) > 0`. The contiguous distance is the minimum length of such
    a region:

    .. math::

        d_{\mathrm{cont}} \;=\;
            \min\{\, |A| : A \text{ contiguous}, \ I(A : R) > 0 \,\}.

    Scans windows of increasing length and returns the first length with a hit,
    so it is the genuine minimum. ``None`` when the system is fully purified
    (``k == 0``, so no region carries logical information).

    The contiguous code distance is from Gullans & Huse, *Scalable Probes of
    Measurement-Induced Criticality*, Phys. Rev. Lett. **125**, 070606 (2020)
    (`arXiv:1910.00020 <https://arxiv.org/abs/1910.00020>`_). The
    erasure-correctability criterion ``A correctable <=> I(A:R) = 0`` is the
    decoupling condition of Schumacher & Nielsen, Phys. Rev. A **54**, 2629
    (1996); see also Choi et al., Phys. Rev. Lett. **125**, 030505 (2020).

    Parameters
    ----------
    sim
        A :class:`SparseGF2` in a purification picture (system qubits
        ``0 .. n_system-1``, the reference following).
    n_system
        Number of system qubits.
    reference
        The reference qubits. Defaults to ``range(n_system, 2 * n_system)``
        (the standard Bell purification); pass e.g. ``[n_system]`` for the
        single-reference picture.
    periodic
        If true, windows wrap around the system ring (suitable for a cycle
        geometry); otherwise they are open intervals (a chain).

    Returns
    -------
    int or None
        :math:`d_{\mathrm{cont}} \in [1, n_{\text{system}}]`, or ``None`` when
        the system is decoupled from the reference.
    """
    if n_system <= 0:
        return None
    ref = (
        np.arange(n_system, 2 * n_system, dtype=np.int64)
        if reference is None
        else _validate_subsystem(reference, sim.n)
    )
    if ref.size == 0:
        return None
    n = sim.n
    # Take the stabilizer block once and forward-eliminate the reference's
    # columns out of it. Afterwards the first ``rank_R`` rows span R and the rest
    # ("tail") are zero on R, so for any window A:
    #   rank(A ∪ R) = rank_R + rank(tail[:, A]),
    #   I(A : R)    = rank(A) + rank_R - rank(A ∪ R) = rank(A) - rank(tail[:, A]).
    # Each window is then two cheap O(rows * |A|) ranks rather than a fresh rank
    # over ~n columns, and S(R) = 0 (a fully purified cell) bails out at once.
    stab = np.asarray(sim.to_symplectic()[n:, :], dtype=np.uint8) & 1
    idx_R = np.concatenate([ref, ref + n])
    reduced, rank_R = _eliminate_on_columns(stab, idx_R)
    if rank_R <= ref.size:  # S(R) = rank_R - |R| = 0  ->  k = 0, no logical info
        return None
    tail = reduced[rank_R:]
    for length in range(1, n_system + 1):
        starts = range(n_system) if periodic else range(n_system - length + 1)
        for start in starts:
            win = np.fromiter(((start + j) % n_system for j in range(length)), np.int64, length)
            idx_A = np.concatenate([win, win + n])
            rank_A = _gf2_rank(stab[:, idx_A])
            rank_A_in_tail = _gf2_rank(tail[:, idx_A]) if tail.shape[0] else 0
            if rank_A - rank_A_in_tail > 0:  # I(A:R) > 0
                return length
    return None  # fully purified: no contiguous region carries logical info


# ----------------------------------------------------------------------
# Generator-weight diagnostics
# ----------------------------------------------------------------------


def generator_weights(sim: SimulatorProtocol) -> NDArray[np.int32]:
    """Return per-generator Pauli weights (number of non-identity qubits).

    Parameters
    ----------
    sim
        A simulator satisfying :class:`SimulatorProtocol`.

    Returns
    -------
    ndarray of shape ``(2 * sim.n,)``, ``int32``
        ``weights[r]`` is the Pauli weight (popcount of the symplectic
        row) of generator ``r``. Rows ``0..n-1`` are destabilizers;
        ``n..2n-1`` are stabilizers. The simulator maintains these
        live via ``supp_len``, so this is **O(2n)** (a copy + cast)
        rather than a recount.

    Notes
    -----
    A fresh copy is returned (read-only-safe for the caller to mutate
    without affecting the simulator's internal state).
    """
    return np.asarray(sim.supp_len, dtype=np.int32).copy()


def stabilizer_weight_spectrum(sim: SimulatorProtocol) -> NDArray[np.int64]:
    """Histogram of stabilizer-row weights.

    Used by the circuits package to track the area- vs volume-law
    diagnostic via the average stabilizer weight ``<wt>``: area-law
    states have ``<wt> = O(1)`` while volume-law states have
    ``<wt> = O(n)``.

    Parameters
    ----------
    sim
        A simulator satisfying :class:`SimulatorProtocol`.

    Returns
    -------
    ndarray of shape ``(sim.n + 1,)``, ``int64``
        ``spec[w]`` is the number of stabilizer rows (indices
        ``n..2n-1``) with exactly ``w`` non-identity Paulis. The sum
        is always ``sim.n`` (one row per qubit).

    Raises
    ------
    RuntimeError
        If any stabilizer-row weight is outside ``[0, n]``, which would
        indicate index corruption (the kernels guarantee
        ``0 ≤ wt ≤ n`` by construction).
    """
    n = sim.n
    if n == 0:
        return np.zeros(1, dtype=np.int64)
    weights = np.asarray(sim.supp_len[n:], dtype=np.int64)
    if weights.size and (weights.min() < 0 or weights.max() > n):
        raise TableauCorruption(
            f"stabilizer_weight_spectrum: weight out of [0, {n}]; "
            f"likely index desync (min={weights.min()}, max={weights.max()})"
        )
    return np.bincount(weights, minlength=n + 1).astype(np.int64)


def average_stabilizer_weight(sim: SimulatorProtocol) -> float:
    """Mean stabilizer weight ``<wt>``.

    For area-law states ``<wt> = O(1)``; for volume-law states
    ``<wt> = O(n)``. Cheap to compute (one numpy reduction over
    ``supp_len[n:]``).
    """
    n = sim.n
    if n == 0:
        return 0.0
    return float(np.mean(sim.supp_len[n:]))
