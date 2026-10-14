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
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable

import numpy as np

from sparsegf2._kernel import entropy as _ent
from sparsegf2._kernel.rng import seed_state
from sparsegf2.pauli import PauliString

__all__ = ["StabilizerCode"]


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
        exceeds ``max_size`` exactly when every count is 0. The counts are
        int64 while every C(n, m) is below 2**53 (so the counts are exact);
        beyond that they are float64, exact to double precision.
        """
        w = operator.index(max_size)
        if not 1 <= w <= self._n:
            raise ValueError(f"max_size must be in [1, {self._n}]")
        out = np.zeros(w, dtype=np.float64)
        if self.num_logical > 0:
            _ent.uncorrectable_counts_rows(self._tab, self._ws, self._system, self._marked, w, out)
        if max(math.comb(self._n, m) for m in range(1, w + 1)) < 2**53:
            return out.astype(np.int64)
        return out

    def _qubits(self, qubits: Iterable[int]) -> np.ndarray:
        a = np.array([int(q) for q in qubits], dtype=np.int64)
        if a.size and (a.min() < 0 or a.max() >= self._n):
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
            bit = np.uint64(1) << np.uint64(q & 63)
            out[rows[:, half * n + q].astype(bool), half * wsys + (q >> 6)] |= bit
    return out


def _unpack(packed: np.ndarray, n: int) -> np.ndarray:
    wsys = (n + 63) // 64
    out = np.zeros((packed.shape[0], 2 * n), dtype=np.uint8)
    for half in range(2):
        for q in range(n):
            word = packed[:, half * wsys + (q >> 6)]
            out[:, half * n + q] = (word >> np.uint64(q & 63)) & np.uint64(1)
    return out
