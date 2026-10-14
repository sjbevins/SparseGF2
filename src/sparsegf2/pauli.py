"""Pauli strings with exact phase tracking.

Conventions (shared by the whole package)
-----------------------------------------
Each qubit carries two bits ``(x, z)``::

    I = (0, 0)    X = (1, 0)    Z = (0, 1)    Y = (1, 1)

A :class:`PauliString` represents the operator

    i**phase * P_0 (x) P_1 (x) ... (x) P_{n-1}

where each ``P_q`` is one of the Hermitian matrices I, X, Y, Z (so Y is the
usual Hermitian Pauli Y, not XZ) and ``phase`` is an integer mod 4::

    phase 0 -> +1     phase 1 -> +i     phase 2 -> -1     phase 3 -> -i

This is the same convention as ``stim.PauliString`` (its ``sign`` is
``1j**phase``), so strings convert to and from Stim without any sign fix-ups.

Text format matches Stim's dense format: an optional sign prefix
(``+``, ``-``, ``i``, ``+i``, ``-i``) followed by one character per qubit from
``IXYZ_`` (``_`` and ``I`` both mean identity). ``str()`` always writes ``_``
for identity, exactly like Stim.

This module is pure numpy. It is used to define gates and to build lookup
tables, never inside a hot loop, so it favours clarity over speed.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

__all__ = ["PauliString", "pauli_product_phase"]

_CHAR_TO_BITS = {
    "I": (0, 0),
    "_": (0, 0),
    "X": (1, 0),
    "Y": (1, 1),
    "Z": (0, 1),
}
_BITS_TO_CHAR = {(0, 0): "_", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}
_SIGN_PREFIX = {"": 0, "+": 0, "-": 2, "i": 1, "+i": 1, "-i": 3}
_PHASE_TO_PREFIX = {0: "+", 1: "+i", 2: "-", 3: "-i"}


def pauli_product_phase(x1: np.ndarray, z1: np.ndarray, x2: np.ndarray, z2: np.ndarray) -> int:
    """Return the power of ``i`` picked up when multiplying two unsigned Pauli strings.

    For Hermitian single-qubit Paulis ``s(a)`` and ``s(b)`` with bits ``a, b``,
    ``s(a) s(b) = i**g(a, b) s(a XOR b)``, where ``g`` is the Aaronson-Gottesman
    function (values -1, 0, +1). Summing ``g`` over qubits gives the phase of the
    full product, reduced mod 4.
    """
    x1 = np.asarray(x1, dtype=np.int64)
    z1 = np.asarray(z1, dtype=np.int64)
    x2 = np.asarray(x2, dtype=np.int64)
    z2 = np.asarray(z2, dtype=np.int64)
    g = np.where(
        (x1 == 1) & (z1 == 1),
        z2 - x2,  # left factor is Y
        np.where(
            x1 == 1,
            z2 * (2 * x2 - 1),  # left factor is X
            np.where(z1 == 1, x2 * (1 - 2 * z2), 0),  # left factor is Z (or I)
        ),
    )
    return int(g.sum()) % 4


class PauliString:
    """An immutable n-qubit Pauli operator with a phase in {+1, +i, -1, -i}.

    Examples
    --------
    >>> p = PauliString("+XZ_Y")
    >>> q = PauliString("-ZZ__")
    >>> str(p * q)
    '+iY__Y'
    >>> p.commutes(q)
    False
    """

    __slots__ = ("_xs", "_zs", "_phase")

    def __init__(self, text: str):
        text = text.strip()
        body_start = 0
        for prefix in ("+i", "-i", "+", "-", "i"):
            if text.startswith(prefix):
                body_start = len(prefix)
                break
        prefix, body = text[:body_start], text[body_start:]
        try:
            bits = [_CHAR_TO_BITS[c] for c in body.upper()]
        except KeyError as exc:
            raise ValueError(
                f"invalid Pauli string {text!r}: characters must be from 'IXYZ_'"
            ) from exc
        xs = np.array([b[0] for b in bits], dtype=np.uint8)
        zs = np.array([b[1] for b in bits], dtype=np.uint8)
        self._init(xs, zs, _SIGN_PREFIX[prefix])

    # -- construction -------------------------------------------------------

    def _init(self, xs: np.ndarray, zs: np.ndarray, phase: int) -> None:
        xs = np.ascontiguousarray(xs, dtype=np.uint8) & 1
        zs = np.ascontiguousarray(zs, dtype=np.uint8) & 1
        if xs.shape != zs.shape or xs.ndim != 1:
            raise ValueError(
                f"xs and zs must be equal-length 1-D arrays, got {xs.shape}, {zs.shape}"
            )
        xs.flags.writeable = False
        zs.flags.writeable = False
        self._xs = xs
        self._zs = zs
        self._phase = int(phase) % 4

    @classmethod
    def from_bits(cls, xs: Iterable[int], zs: Iterable[int], phase: int = 0) -> PauliString:
        """Build from x and z bit arrays and a phase (power of i)."""
        obj = cls.__new__(cls)
        xs = xs if isinstance(xs, np.ndarray) else list(xs)
        zs = zs if isinstance(zs, np.ndarray) else list(zs)
        obj._init(np.asarray(xs), np.asarray(zs), phase)
        return obj

    @classmethod
    def identity(cls, n: int) -> PauliString:
        """The n-qubit identity with phase +1."""
        return cls.from_bits(np.zeros(n, np.uint8), np.zeros(n, np.uint8), 0)

    @classmethod
    def single(cls, n: int, qubit: int, pauli: str) -> PauliString:
        """The n-qubit string with ``pauli`` ('X', 'Y' or 'Z') on ``qubit`` and identity elsewhere."""
        if not 0 <= qubit < n:
            raise ValueError(f"qubit {qubit} out of range for n={n}")
        xs = np.zeros(n, np.uint8)
        zs = np.zeros(n, np.uint8)
        xs[qubit], zs[qubit] = _CHAR_TO_BITS[pauli.upper()]
        return cls.from_bits(xs, zs, 0)

    # -- accessors ------------------------------------------------------------

    @property
    def xs(self) -> np.ndarray:
        """Read-only uint8 array of x bits, one per qubit."""
        return self._xs

    @property
    def zs(self) -> np.ndarray:
        """Read-only uint8 array of z bits, one per qubit."""
        return self._zs

    @property
    def phase(self) -> int:
        """The phase as a power of i: 0 (+1), 1 (+i), 2 (-1), 3 (-i)."""
        return self._phase

    @property
    def sign(self) -> complex:
        """The phase as a complex number (same as ``stim.PauliString.sign``)."""
        return (1, 1j, -1, -1j)[self._phase]

    @property
    def is_hermitian(self) -> bool:
        """True when the phase is +1 or -1."""
        return self._phase % 2 == 0

    @property
    def weight(self) -> int:
        """Number of qubits where the operator is not the identity."""
        return int(np.count_nonzero(self._xs | self._zs))

    def __len__(self) -> int:
        return int(self._xs.shape[0])

    def __getitem__(self, qubit: int) -> str:
        """The single-qubit Pauli on ``qubit`` as one of 'I', 'X', 'Y', 'Z'."""
        c = _BITS_TO_CHAR[(int(self._xs[qubit]), int(self._zs[qubit]))]
        return "I" if c == "_" else c

    # -- algebra --------------------------------------------------------------

    def __mul__(self, other: PauliString) -> PauliString:
        if not isinstance(other, PauliString):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError(f"length mismatch: {len(self)} vs {len(other)}")
        phase = (
            self._phase
            + other._phase
            + pauli_product_phase(self._xs, self._zs, other._xs, other._zs)
        )
        return PauliString.from_bits(self._xs ^ other._xs, self._zs ^ other._zs, phase)

    def __neg__(self) -> PauliString:
        return PauliString.from_bits(self._xs, self._zs, self._phase + 2)

    def times_phase(self, k: int) -> PauliString:
        """Return ``i**k * self``."""
        return PauliString.from_bits(self._xs, self._zs, self._phase + k)

    def commutes(self, other: PauliString) -> bool:
        """True when the two operators commute (symplectic product is 0)."""
        if len(self) != len(other):
            raise ValueError(f"length mismatch: {len(self)} vs {len(other)}")
        s = np.count_nonzero(self._xs & other._zs) + np.count_nonzero(self._zs & other._xs)
        return bool(s % 2 == 0)

    def unsigned(self) -> PauliString:
        """The same operator with phase reset to +1."""
        return PauliString.from_bits(self._xs, self._zs, 0)

    def to_matrix(self) -> np.ndarray:
        """Dense ``2**n x 2**n`` complex matrix, qubit 0 as the most significant kron factor.

        Intended for tests on a handful of qubits.
        """
        mats = {
            (0, 0): np.eye(2, dtype=complex),
            (1, 0): np.array([[0, 1], [1, 0]], dtype=complex),
            (1, 1): np.array([[0, -1j], [1j, 0]], dtype=complex),
            (0, 1): np.array([[1, 0], [0, -1]], dtype=complex),
        }
        out = np.ones((1, 1), dtype=complex)
        for x, z in zip(self._xs, self._zs, strict=True):
            out = np.kron(out, mats[(int(x), int(z))])
        return self.sign * out

    # -- conversions ----------------------------------------------------------

    def __str__(self) -> str:
        body = "".join(
            _BITS_TO_CHAR[(int(x), int(z))] for x, z in zip(self._xs, self._zs, strict=True)
        )
        return _PHASE_TO_PREFIX[self._phase] + body

    def __repr__(self) -> str:
        return f"PauliString({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PauliString):
            return NotImplemented
        return (
            self._phase == other._phase
            and np.array_equal(self._xs, other._xs)
            and np.array_equal(self._zs, other._zs)
        )

    def __hash__(self) -> int:
        return hash((self._xs.tobytes(), self._zs.tobytes(), self._phase))

    def to_stim(self):
        """Convert to ``stim.PauliString`` (requires Stim)."""
        import stim

        return stim.PauliString(str(self))

    @classmethod
    def from_stim(cls, pauli) -> PauliString:
        """Convert from ``stim.PauliString`` (requires Stim)."""
        return cls(str(pauli))
