"""Small Clifford unitaries as signed tableaux, plus the 1- and 2-qubit Clifford groups.

A k-qubit Clifford ``C`` (modulo global phase) is fully described by where it
sends the 2k generators of the Pauli group under conjugation::

    C X_j C^dagger = x_image(j)        C Z_j C^dagger = z_image(j)

Each image is a Hermitian :class:`~sparsegf2_v3.pauli.PauliString` with sign
+1 or -1. This is exactly the information in a ``stim.Tableau``, and the
conversion between the two is lossless.

Symplectic matrix convention
----------------------------
``Clifford.symplectic_matrix()`` returns the ``(2k, 2k)`` GF(2) matrix ``M``
whose row ``i`` is the ``[x | z]`` bit vector of the image of basis element
``i``, with basis order ``X_0 .. X_{k-1}, Z_0 .. Z_{k-1}``. An unsigned Pauli
with bit row vector ``v = [x | z]`` maps to ``v @ M (mod 2)``. This is the same
row-vector convention SparseGF2 v2 used.

Canonical indexing of the 1- and 2-qubit groups
-----------------------------------------------
Random-circuit generators need to name a Clifford with a single integer. We fix
a version-stable order that does not depend on Stim:

* ``symplectic_group(k)`` lists all of ``Sp(2k, F_2)`` (6 for k=1, 720 for k=2),
  sorted by the integer whose binary digits are the row-major entries of ``M``
  (first entry most significant).
* A Clifford's index is ``symplectic_index * 4**k + sign_bits`` where bit ``j``
  of ``sign_bits`` is set when image ``j`` (same ``X_0..X_{k-1}, Z_0..Z_{k-1}``
  order) has sign -1.

So ``Clifford.from_index(2, i)`` for ``i`` in ``range(11520)`` enumerates the
two-qubit Clifford group exactly once each, and a uniform integer in that range
is a uniformly random two-qubit Clifford. For entanglement dynamics without
phases only the symplectic part matters; it is ``i // 4**k``.

Conjugation tables
------------------
``Clifford.conjugation_table()`` gives, for every unsigned input Pauli on the k
qubits, the output Pauli and whether a minus sign appears. Pauli codes are
``code_q = x_q + 2 * z_q`` per qubit (I=0, X=1, Z=2, Y=3) packed as
``sum_q code_q << (2 q)``. Simulation kernels are built from these tables.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import numpy as np

from sparsegf2_v3.pauli import PauliString

__all__ = [
    "Clifford",
    "num_cliffords",
    "random_clifford_indices",
    "symplectic_form",
    "symplectic_group",
]


def symplectic_form(k: int) -> np.ndarray:
    """The ``(2k, 2k)`` GF(2) matrix ``[[0, I], [I, 0]]``."""
    om = np.zeros((2 * k, 2 * k), dtype=np.uint8)
    om[:k, k:] = np.eye(k, dtype=np.uint8)
    om[k:, :k] = np.eye(k, dtype=np.uint8)
    return om


def _is_symplectic(m: np.ndarray) -> bool:
    k = m.shape[0] // 2
    om = symplectic_form(k).astype(np.int64)
    mm = m.astype(np.int64)
    return bool(np.array_equal((mm @ om @ mm.T) % 2, om))


class Clifford:
    """A k-qubit Clifford unitary modulo global phase, stored as a signed tableau.

    Build one from generator images, from a gate name (see
    :func:`sparsegf2_v3.gates.gate`), from its canonical index, or from a
    ``stim.Tableau``.

    Examples
    --------
    >>> h = Clifford.from_strings(xs=["+Z"], zs=["+X"])     # Hadamard
    >>> str(h(PauliString("+Y")))
    '-Y'
    """

    __slots__ = ("_x_images", "_z_images")

    def __init__(self, x_images: Sequence[PauliString], z_images: Sequence[PauliString]):
        xs = tuple(x_images)
        zs = tuple(z_images)
        k = len(xs)
        if len(zs) != k:
            raise ValueError(f"need the same number of X and Z images, got {k} and {len(zs)}")
        for img in xs + zs:
            if len(img) != k:
                raise ValueError(f"every image must act on {k} qubits, got {img}")
            if not img.is_hermitian:
                raise ValueError(f"images must be Hermitian (sign +1 or -1), got {img}")
        for i in range(k):
            for j in range(k):
                if not xs[i].commutes(xs[j]) or not zs[i].commutes(zs[j]):
                    raise ValueError("images do not satisfy the Pauli commutation relations")
                if xs[i].commutes(zs[j]) == (i == j):
                    raise ValueError("images do not satisfy the Pauli commutation relations")
        self._x_images = xs
        self._z_images = zs

    # -- constructors -----------------------------------------------------------

    @classmethod
    def from_strings(cls, xs: Sequence[str], zs: Sequence[str]) -> Clifford:
        """Build from images written as Pauli strings, e.g. ``xs=["+XX", "+_X"]``."""
        return cls([PauliString(s) for s in xs], [PauliString(s) for s in zs])

    @classmethod
    def identity(cls, k: int) -> Clifford:
        return cls(
            [PauliString.single(k, j, "X") for j in range(k)],
            [PauliString.single(k, j, "Z") for j in range(k)],
        )

    @classmethod
    def from_symplectic(
        cls, matrix: np.ndarray, sign_bits: Sequence[int] | None = None
    ) -> Clifford:
        """Build from a ``(2k, 2k)`` symplectic matrix and optional per-image sign bits."""
        m = np.asarray(matrix, dtype=np.uint8) & 1
        if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] % 2:
            raise ValueError(f"matrix must be (2k, 2k), got {m.shape}")
        if not _is_symplectic(m):
            raise ValueError("matrix is not symplectic")
        k = m.shape[0] // 2
        signs = np.zeros(2 * k, np.uint8) if sign_bits is None else np.asarray(sign_bits, np.uint8)
        images = [
            PauliString.from_bits(m[i, :k], m[i, k:], 2 * int(signs[i])) for i in range(2 * k)
        ]
        return cls(images[:k], images[k:])

    @classmethod
    def from_index(cls, k: int, index: int) -> Clifford:
        """The Clifford at position ``index`` of the canonical order (k = 1 or 2)."""
        total = num_cliffords(k)
        if not 0 <= index < total:
            raise ValueError(f"index must be in [0, {total}), got {index}")
        sym_index, sign_word = divmod(int(index), 4**k)
        signs = [(sign_word >> j) & 1 for j in range(2 * k)]
        return cls.from_symplectic(symplectic_group(k)[sym_index], signs)

    @classmethod
    def from_stim(cls, tableau) -> Clifford:
        """Convert from a ``stim.Tableau``."""
        k = len(tableau)
        return cls(
            [PauliString.from_stim(tableau.x_output(j)) for j in range(k)],
            [PauliString.from_stim(tableau.z_output(j)) for j in range(k)],
        )

    # -- accessors ------------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        return len(self._x_images)

    def x_image(self, j: int) -> PauliString:
        """``C X_j C^dagger``."""
        return self._x_images[j]

    def z_image(self, j: int) -> PauliString:
        """``C Z_j C^dagger``."""
        return self._z_images[j]

    def symplectic_matrix(self) -> np.ndarray:
        """The ``(2k, 2k)`` GF(2) matrix whose rows are the unsigned images."""
        k = self.num_qubits
        m = np.zeros((2 * k, 2 * k), dtype=np.uint8)
        for i, img in enumerate(self._x_images + self._z_images):
            m[i, :k] = img.xs
            m[i, k:] = img.zs
        return m

    def sign_bits(self) -> np.ndarray:
        """Length-2k uint8 array: 1 where the image (X_0..X_{k-1}, Z_0..Z_{k-1}) has sign -1."""
        return np.array([img.phase // 2 for img in self._x_images + self._z_images], dtype=np.uint8)

    def index(self) -> int:
        """Position in the canonical order (k = 1 or 2). Inverse of :meth:`from_index`."""
        k = self.num_qubits
        lookup = _symplectic_lookup(k)
        sym_index = lookup[self.symplectic_matrix().tobytes()]
        sign_word = sum(int(b) << j for j, b in enumerate(self.sign_bits()))
        return sym_index * 4**k + sign_word

    # -- action -----------------------------------------------------------------

    def __call__(self, pauli: PauliString) -> PauliString:
        """Conjugate: return ``C P C^dagger``."""
        k = self.num_qubits
        if len(pauli) != k:
            raise ValueError(f"expected a {k}-qubit Pauli, got {len(pauli)} qubits")
        # P = i**phase * prod_q i**(x_q z_q) X_q**x_q Z_q**z_q   (Y = i X Z)
        extra = int(np.count_nonzero(pauli.xs & pauli.zs))
        out = PauliString.identity(k).times_phase(pauli.phase + extra)
        for q in range(k):
            if pauli.xs[q]:
                out = out * self._x_images[q]
            if pauli.zs[q]:
                out = out * self._z_images[q]
        return out

    def then(self, other: Clifford) -> Clifford:
        """The Clifford that applies ``self`` first and ``other`` second (operator ``other @ self``)."""
        if other.num_qubits != self.num_qubits:
            raise ValueError("qubit-count mismatch")
        return Clifford(
            [other(img) for img in self._x_images],
            [other(img) for img in self._z_images],
        )

    def inverse(self) -> Clifford:
        """The inverse Clifford."""
        k = self.num_qubits
        om = symplectic_form(k).astype(np.int64)
        m_inv = (om @ self.symplectic_matrix().astype(np.int64).T @ om) % 2
        images = []
        for i in range(2 * k):
            candidate = PauliString.from_bits(m_inv[i, :k], m_inv[i, k:], 0)
            # self(candidate) is +/- the i-th generator; pick the sign that makes it +.
            images.append(candidate if self(candidate).phase == 0 else -candidate)
        return Clifford(images[:k], images[k:])

    def conjugation_table(self) -> tuple[np.ndarray, np.ndarray]:
        """Output codes and sign flips for every unsigned input Pauli (see module docs).

        Returns ``(out_codes, sign_flips)``, both uint8 arrays of length ``4**k``.
        """
        k = self.num_qubits
        out_codes = np.zeros(4**k, np.uint8)
        sign_flips = np.zeros(4**k, np.uint8)
        for code in range(4**k):
            xs = [(code >> (2 * q)) & 1 for q in range(k)]
            zs = [(code >> (2 * q + 1)) & 1 for q in range(k)]
            img = self(PauliString.from_bits(xs, zs, 0))
            out_codes[code] = sum(
                (int(img.xs[q]) | (int(img.zs[q]) << 1)) << (2 * q) for q in range(k)
            )
            sign_flips[code] = img.phase // 2
        return out_codes, sign_flips

    # -- dunder / conversions -------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Clifford):
            return NotImplemented
        return self._x_images == other._x_images and self._z_images == other._z_images

    def __hash__(self) -> int:
        return hash((self._x_images, self._z_images))

    def __str__(self) -> str:
        k = self.num_qubits
        lines = [f"X{j} -> {self._x_images[j]}" for j in range(k)]
        lines += [f"Z{j} -> {self._z_images[j]}" for j in range(k)]
        return "\n".join(lines)

    def __repr__(self) -> str:
        xs = [str(p) for p in self._x_images]
        zs = [str(p) for p in self._z_images]
        return f"Clifford.from_strings(xs={xs}, zs={zs})"

    def to_stim(self):
        """Convert to ``stim.Tableau`` (requires Stim)."""
        import stim

        return stim.Tableau.from_conjugated_generators(
            xs=[p.to_stim() for p in self._x_images],
            zs=[p.to_stim() for p in self._z_images],
        )


# -- the 1- and 2-qubit groups ---------------------------------------------------


@functools.cache
def symplectic_group(k: int) -> np.ndarray:
    """All of ``Sp(2k, F_2)`` for k = 1 or 2, in canonical order (read-only array).

    Shape ``(6, 2, 2)`` for k=1 and ``(720, 4, 4)`` for k=2. Found by brute force
    over all ``2**(4 k**2)`` bit matrices, which takes a few milliseconds.
    """
    if k not in (1, 2):
        raise ValueError("symplectic_group is only provided for k = 1 or 2")
    d = 2 * k
    nbits = d * d
    ints = np.arange(1 << nbits, dtype=np.int64)
    shifts = np.arange(nbits - 1, -1, -1, dtype=np.int64)  # first entry = most significant bit
    mats = ((ints[:, None] >> shifts[None, :]) & 1).astype(np.int64).reshape(-1, d, d)
    om = symplectic_form(k).astype(np.int64)
    prod = np.einsum("aij,jk,alk->ail", mats, om, mats) % 2
    keep = np.all(prod == om, axis=(1, 2))
    group = mats[keep].astype(np.uint8)  # already sorted because ints is increasing
    group.flags.writeable = False
    return group


@functools.cache
def _symplectic_lookup(k: int) -> dict[bytes, int]:
    return {m.tobytes(): i for i, m in enumerate(symplectic_group(k))}


def num_cliffords(k: int) -> int:
    """Size of the k-qubit Clifford group modulo phase: 24 for k=1, 11520 for k=2."""
    return len(symplectic_group(k)) * 4**k


def random_clifford_indices(
    k: int, size: int | tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """Uniformly random canonical Clifford indices (k = 1 or 2) as an int64 array."""
    return rng.integers(0, num_cliffords(k), size=size, dtype=np.int64)
