"""Phase-free stabilizer tableaux as canonical GF(2) row spaces.

A Pauli operator on ``N`` qubits, up to its phase, is a row vector
``(x_0, ..., x_{N-1}, z_0, ..., z_{N-1})`` over GF(2) (``X = (1, 0)``, ``Z = (0, 1)``,
``Y = (1, 1)`` on each qubit), and the product of two Paulis is, up to a phase, the sum of
their vectors. So a stabilizer group with its signs removed (the *phase-free* stabilizer
group of v1.4 §5.1) is a subspace of GF(2)^{2N}: the row space of its generators. A
stabilizer state's phase-free group is *isotropic* (its elements commute pairwise); a pure
state's has dimension ``N``. :class:`PhaseFreeTableau` holds any such row space in a
canonical form, so that two generating sets of the same group, from any basis, any order or
any simulator, give the same object, the same bytes and the same digest.

Conventions
-----------
* **Qubits.** ``0 .. N-1``. A snapshot of the ensemble engine
  (:func:`sparsegf2.mipt.tableau_snapshot`) lists the system qubits ``0 .. n-1`` first and the
  reference qubits ``n .. N-1`` after them (``N = 2n`` in the purification picture, where
  reference ``n + i`` starts Bell-paired with system qubit ``i``).
* **Columns.** A row is ``(x_0, ..., x_{N-1}, z_0, ..., z_{N-1})``, in this order. The
  canonical form is the reduced row echelon form over GF(2) for this column order: the first
  nonzero column of each row (its pivot) holds a 1, the pivots increase down the rows, every
  pivot column is zero in every other row, and zero rows are dropped. Every subspace has
  exactly one such basis (``docs/study/03-phase-free-tableau.md`` has the proof).
* **Words.** :meth:`PhaseFreeTableau.from_words` and :attr:`PhaseFreeTableau.words` use the
  engine's packing: ``wq = ceil(N / 64)`` uint64 words of X bits (``x_q`` is bit ``q & 63`` of
  word ``q >> 6``), then ``wq`` words of Z bits (``z_q`` is bit ``q & 63`` of word
  ``wq + (q >> 6)``); the bits beyond qubit ``N - 1`` are zero.
* **Bytes.** :meth:`PhaseFreeTableau.to_bytes` writes the 8-byte header :data:`MAGIC`
  (``b"SGF2RS1\\0"``), then ``N`` and the rank as little-endian uint32, then the canonical rows
  in order, each row's ``2N`` bits in the column order above packed most significant bit first
  (``np.packbits(row, bitorder="big")``: ``ceil(2N / 8)`` bytes, zero padding at the end).
  :meth:`PhaseFreeTableau.digest` is the SHA-256 of these bytes, in hexadecimal: the canonical
  phase-free row-space digest of v1.4 §5, independent of the generating set and of the
  simulator that produced it (the raw engine words are not: they depend on the basis).
* **Entropies** are in bits, of the stabilizer state whose phase-free group is the row space
  (the signs do not change them): ``S(A) = |A| - dim S_A`` with ``S_A`` the elements supported
  in ``A``, which is ``rank(S|_A) - |A|`` for a pure state.

The eliminations are compiled (:mod:`sparsegf2._kernel.rowspace`) and exact.
"""

from __future__ import annotations

import hashlib
import operator
import struct
from collections.abc import Iterable

import numpy as np

from sparsegf2._kernel import entropy as _ent
from sparsegf2._kernel import rowspace as _rs
from sparsegf2.codes import StabilizerCode, _bit_rows

__all__ = ["MAGIC", "PhaseFreeTableau"]

MAGIC = b"SGF2RS1\x00"
"""The first 8 bytes of :meth:`PhaseFreeTableau.to_bytes` (format 1 of the canonical row space)."""

_HEADER = struct.Struct("<II")  # N, rank


def _words_per_half(n: int) -> int:
    return (n + 63) // 64


def _canonical_columns(n: int) -> np.ndarray:
    """The packed positions of ``x_0 .. x_{N-1}, z_0 .. z_{N-1}``, in this order."""
    q = np.arange(n, dtype=np.int64)
    return np.concatenate([q, q + 64 * _words_per_half(n)])


def _dense_to_words(rows: np.ndarray, n: int) -> np.ndarray:
    """``(s, 2N)`` 0/1 rows as ``(s, 2 wq)`` uint64 words in the engine packing."""
    wq = _words_per_half(n)
    out = np.empty((rows.shape[0], 2 * wq), dtype=np.uint64)
    padded = np.zeros((rows.shape[0], 64 * wq), dtype=np.uint8)
    for half in range(2):
        padded[:, :n] = rows[:, half * n : (half + 1) * n]
        packed = np.packbits(padded, axis=1, bitorder="little")  # x_q is bit q of byte q // 8
        out[:, half * wq : (half + 1) * wq] = packed.view("<u8")
    return out


def _words_to_dense(words: np.ndarray, n: int) -> np.ndarray:
    """``(s, 2 wq)`` uint64 words in the engine packing as ``(s, 2N)`` 0/1 uint8 rows."""
    wq = _words_per_half(n)
    little = np.ascontiguousarray(words, dtype="<u8")
    out = np.empty((little.shape[0], 2 * n), dtype=np.uint8)
    for half in range(2):
        block = np.ascontiguousarray(little[:, half * wq : (half + 1) * wq]).view(np.uint8)
        out[:, half * n : (half + 1) * n] = np.unpackbits(block, axis=1, bitorder="little")[:, :n]
    return out


def _checked_words(words, n: int) -> np.ndarray:
    """Engine words as a new writable C-contiguous uint64 array, their shape and padding checked."""
    wq = _words_per_half(n)
    raw = np.asarray(words)
    if raw.ndim != 2 or raw.shape[1] != 2 * wq:
        raise ValueError(
            f"words need shape (rows, {2 * wq}) for {n} qubits (X words, then Z words), "
            f"got {raw.shape}"
        )
    if (
        raw.size
        and raw.dtype != np.uint64
        and (not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0))
    ):
        raise ValueError(f"words must be uint64 (non-negative integers), got {raw.dtype}")
    out = np.array(raw, dtype=np.uint64, order="C")  # always a copy: rref works in place
    if n % 64 and out.size:
        pad = ~((np.uint64(1) << np.uint64(n % 64)) - np.uint64(1))
        if np.any(out[:, [wq - 1, 2 * wq - 1]] & pad):
            raise ValueError(f"words have bits beyond qubit {n - 1}")
    return out


def _qubit_count(num_qubits) -> int:
    n = operator.index(num_qubits)
    if n < 1:
        raise ValueError(f"num_qubits must be at least 1, got {n}")
    return n


class PhaseFreeTableau:
    """A GF(2) row space of Pauli operators on ``N`` qubits, held in canonical form.

    Parameters
    ----------
    stabilizers
        Generators of the row space: an ``(s, 2N)`` 0/1 array whose rows hold the X bits of
        qubits ``0 .. N-1`` and then their Z bits, or Pauli strings (text such as ``"XZ_Y"``,
        or :class:`~sparsegf2.PauliString`; a sign is ignored). They may be dependent,
        repeated or zero: only their span matters.
    num_qubits
        ``N``; needed only when there are no generators.

    Construct also with :meth:`from_words` (the engine's packed rows, as
    :meth:`sparsegf2.mipt.EnsembleResult.tableau` does), :meth:`from_simulator` (the
    stabilizers of a :class:`~sparsegf2.Simulator`, signs dropped) or :meth:`from_bytes`.
    The module docstring fixes the qubit order, the column order, the packing and the byte
    format. Two tableaux are equal (``==``, :meth:`same_space`) exactly when their row spaces
    are, and then their :meth:`to_bytes` and :meth:`digest` agree.

    Examples
    --------
    A Bell pair, from two generating sets (``YY`` is ``XX`` times ``ZZ`` up to a phase):

    >>> a = PhaseFreeTableau(["XX", "ZZ"])
    >>> b = PhaseFreeTableau(["YY", "XX", "ZZ", "__"])
    >>> a.same_space(b), a.rank, a.digest() == b.digest()
    (True, 2, True)
    >>> a.entropy([0]), a.mutual_information([0], [1]), a.is_pure()
    (1, 2, True)
    """

    def __init__(self, stabilizers, num_qubits: int | None = None):
        rows = _bit_rows(stabilizers, num_qubits)
        n = _qubit_count(rows.shape[1] // 2)
        self._init(_dense_to_words(rows, n), n)

    def _init(self, words: np.ndarray, n: int) -> None:
        """Canonicalize ``words`` (writable, C-contiguous uint64 of shape ``(s, 2 wq)``) in place."""
        pivots = np.empty(max(words.shape[0], 1), dtype=np.int64)
        rank = int(_rs.rref(words, _canonical_columns(n), pivots))
        self._n = n
        self._wq = _words_per_half(n)
        self._words = words[:rank].copy()  # never modified after this
        self._bytes: bytes | None = None
        self._isotropic: bool | None = None

    # -- constructors ----------------------------------------------------------------------

    @classmethod
    def from_words(cls, words, num_qubits: int) -> PhaseFreeTableau:
        """The row space of packed rows: ``(s, 2 ceil(N / 64))`` uint64, X words then Z words.

        This is the layout of a ``tableau_snapshot`` (and of a ``code_snapshot``, whose rows
        cover the system qubits only); bits beyond qubit ``N - 1`` must be zero.
        """
        n = _qubit_count(num_qubits)
        tab = cls.__new__(cls)
        tab._init(_checked_words(words, n), n)
        return tab

    @classmethod
    def from_simulator(cls, sim) -> PhaseFreeTableau:
        """The phase-free stabilizer group of a :class:`~sparsegf2.Simulator`'s state.

        The signs are dropped, so a phase-tracking simulator and a phase-free one in the
        same stabilizer state (any measurement outcomes) give the same tableau.
        """
        xs, zs, _ = sim.stabilizer_arrays()
        n = int(sim.num_qubits)
        return cls.from_words(_dense_to_words(np.concatenate([xs, zs], axis=1), n), n)

    @classmethod
    def from_bytes(cls, data: bytes) -> PhaseFreeTableau:
        """The tableau written by :meth:`to_bytes` (and only canonical bytes are accepted).

        Raises ``ValueError`` for another header, a length that does not match ``N`` and the
        rank, nonzero padding bits, or rows that are not the canonical form of their row
        space (so ``PhaseFreeTableau.from_bytes(b).to_bytes() == b`` whenever it returns).
        """
        data = bytes(data)
        head = len(MAGIC) + _HEADER.size
        if len(data) < head or data[: len(MAGIC)] != MAGIC:
            raise ValueError(f"not a canonical row space: the bytes must start with {MAGIC!r}")
        n, rank = _HEADER.unpack_from(data, len(MAGIC))
        if n < 1 or rank > 2 * n:
            raise ValueError(f"not a canonical row space: N = {n}, rank = {rank}")
        width = (2 * n + 7) // 8
        if len(data) != head + rank * width:
            raise ValueError(
                f"not a canonical row space: {len(data)} bytes, but N = {n} and rank {rank} "
                f"need {head + rank * width}"
            )
        body = np.frombuffer(data, dtype=np.uint8, offset=head).reshape(rank, width)
        bits = np.unpackbits(body, axis=1, bitorder="big")
        if bits[:, 2 * n :].any():
            raise ValueError("not a canonical row space: nonzero padding bits")
        tab = cls.from_words(_dense_to_words(bits[:, : 2 * n], n), n)
        if tab.to_bytes() != data:
            raise ValueError(
                "not a canonical row space: the rows are not the reduced row echelon form of "
                "their span (or are not independent)"
            )
        return tab

    # -- the row space ---------------------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        """``N``, the number of qubits."""
        return self._n

    @property
    def rank(self) -> int:
        """The dimension of the row space (``N`` for a pure state)."""
        return int(self._words.shape[0])

    @property
    def words(self) -> np.ndarray:
        """The canonical rows, packed as in :meth:`from_words`: ``(rank, 2 wq)`` uint64, read-only."""
        view = self._words.view()
        view.flags.writeable = False
        return view

    def canonical(self) -> np.ndarray:
        """The canonical rows as a ``(rank, 2N)`` uint8 0/1 matrix (X bits, then Z bits).

        The reduced row echelon form over GF(2) for the columns ``x_0 .. x_{N-1}, z_0 ..
        z_{N-1}``, zero rows dropped: the same matrix for every generating set of the space.
        """
        return _words_to_dense(self._words, self._n)

    def to_bytes(self) -> bytes:
        """The canonical serialization (module docstring, Bytes): header, ``N``, rank, rows."""
        if self._bytes is None:
            body = np.packbits(self.canonical(), axis=1, bitorder="big")
            self._bytes = MAGIC + _HEADER.pack(self._n, self.rank) + body.tobytes()
        return self._bytes

    def digest(self) -> str:
        """SHA-256 of :meth:`to_bytes` in hexadecimal: the canonical row-space digest."""
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def is_isotropic(self) -> bool:
        """Whether all rows commute pairwise (``x.z' + z.x' = 0``): a phase-free stabilizer group."""
        if self._isotropic is None:
            self._isotropic = bool(_ent.generators_commute(self._words))
        return self._isotropic

    def is_pure(self) -> bool:
        """Whether this is the phase-free group of a pure state: isotropic, of rank ``N``."""
        return self.rank == self._n and self.is_isotropic()

    def same_space(self, other: PhaseFreeTableau) -> bool:
        """Whether ``other`` is the same row space on the same number of qubits."""
        if not isinstance(other, PhaseFreeTableau):
            raise TypeError(f"compare with a PhaseFreeTableau, got {type(other).__name__}")
        return self._n == other._n and np.array_equal(self._words, other._words)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PhaseFreeTableau):
            return NotImplemented
        return self.same_space(other)

    def __hash__(self) -> int:
        return hash(self.to_bytes())

    def __repr__(self) -> str:
        return f"PhaseFreeTableau(num_qubits={self._n}, rank={self.rank})"

    # -- entropies ---------------------------------------------------------------------------

    def _qubits(self, qubits: Iterable[int], what: str) -> np.ndarray:
        if isinstance(qubits, (str, bytes)):
            raise TypeError(f"{what} must be qubit indices, got {qubits!r}")
        a = np.array([operator.index(q) for q in qubits], dtype=np.int64)
        if np.any(a < 0) or np.any(a >= self._n):
            raise ValueError(f"{what}: qubit indices must be in [0, {self._n})")
        if np.unique(a).size != a.size:
            raise ValueError(f"{what}: qubit indices must be distinct")
        return np.sort(a)

    def _rank_on(self, qubits: np.ndarray) -> int:
        """The rank of the rows restricted to the X and Z columns of ``qubits``."""
        if not qubits.size or not self.rank:
            return 0
        cols = np.concatenate([qubits, qubits + 64 * self._wq])
        scratch = np.empty_like(self._words)
        return int(_rs.rank_of_columns(self._words, self.rank, cols, scratch))

    def _require_group(self, what: str) -> None:
        if not self.is_isotropic():
            raise ValueError(
                f"{what} needs a stabilizer group, but the rows do not all commute "
                "(is_isotropic() is False)"
            )

    def entropy(self, region: Iterable[int]) -> int:
        """The entropy ``S(A)`` in bits of the qubits ``A = region``.

        For the stabilizer state whose phase-free group ``S`` (dimension ``r``) is this row
        space, ``S(A) = |A| - dim S_A`` with ``S_A`` the elements of ``S`` supported in ``A``,
        and ``dim S_A = r - rank(S|_Abar)`` (the restriction to the complement ``Abar`` has
        kernel ``S_A``). For a pure state (``r = N``) this is ``rank(S|_A) - |A|``, which is
        used when ``A`` is the smaller side. Needs an isotropic row space.
        """
        a = self._qubits(region, "region")
        self._require_group("entropy")
        if 2 * a.size <= self._n and self.is_pure():
            return self._rank_on(a) - int(a.size)
        rest = np.setdiff1d(np.arange(self._n, dtype=np.int64), a)
        return int(a.size) - self.rank + self._rank_on(rest)

    def mutual_information(self, a: Iterable[int], b: Iterable[int]) -> int:
        """``I(A:B) = S(A) + S(B) - S(A u B)`` in bits, for disjoint sets of qubits ``a`` and ``b``.

        With ``a`` system qubits and ``b`` the reference of a snapshot, this is ``I(A:R)``,
        the logical information an erasure of ``A`` loses (v1.4 §5.4).
        """
        qa, qb = self._qubits(a, "a"), self._qubits(b, "b")
        if np.intersect1d(qa, qb).size:
            raise ValueError("the regions a and b must be disjoint")
        return self.entropy(qa) + self.entropy(qb) - self.entropy(np.concatenate([qa, qb]))

    # -- the physical code -------------------------------------------------------------------

    def physical_tableau(self, num_system: int) -> PhaseFreeTableau:
        """The elements with no support on qubits ``>= num_system``, as a row space of the system.

        Reduces the rows to reduced row echelon form with the reference columns (the X and Z
        columns of qubits ``num_system .. N-1``) first: the rows whose pivot is a system column
        are zero on the reference and span every element of the row space that is
        (``docs/study/03-phase-free-tableau.md`` has the proof). Restricted to the system
        qubits ``0 .. num_system - 1`` they span ``S_phys`` (v1.4 §5), of dimension ``r_t``.
        """
        ns = operator.index(num_system)
        if not 1 <= ns <= self._n:
            raise ValueError(f"num_system must be in [1, {self._n}], got {ns}")
        if ns == self._n:
            return self
        wq = self._wq
        ref = np.arange(ns, self._n, dtype=np.int64)
        system = np.arange(ns, dtype=np.int64)
        cols = np.concatenate([ref, ref + 64 * wq, system, system + 64 * wq])
        rows = self._words.copy()
        pivots = np.empty(max(rows.shape[0], 1), dtype=np.int64)
        rank = int(_rs.rref(rows, cols, pivots))
        first = int(np.searchsorted(pivots[:rank], 2 * (self._n - ns)))  # the reference pivots
        physical = rows[first:rank]
        wsys = _words_per_half(ns)
        return PhaseFreeTableau.from_words(
            np.concatenate([physical[:, :wsys], physical[:, wq : wq + wsys]], axis=1), ns
        )

    def physical_code(self, num_system: int) -> StabilizerCode:
        """The physical code of a joint state: :meth:`physical_tableau` as a ``StabilizerCode``.

        Its generators are the canonical rows of ``S_phys`` on the ``num_system`` system qubits;
        ``num_stabilizers`` is ``r_t`` and ``num_logical`` is ``k_t = num_system - r_t``, the
        reference entropy ``S(R)`` of a pure joint state (v1.4 §5, §5.2). It is the code a
        ``code_snapshot`` of the same state saves, as a row space.
        """
        self._require_group("physical_code")
        physical = self.physical_tableau(num_system)
        return StabilizerCode.from_packed(np.array(physical._words), physical.num_qubits)
