"""Erasure decoding on a role-labeled tableau: M, r_M, recovery, candidates.

Implements the erasure-decoding formalism of Gullans, Krastanov, Huse,
Jiang, and Flammia, PRX **11**, 031066 (2021), Eqs. (16)-(18), in
tableau coordinates:

* :func:`uncorrectable_matrix` builds :math:`M(S, L, e)`: one row per
  local error basis element on the erased sites (:math:`Z_i`, then
  :math:`X_i`, per site), columns the syndrome bits against the checks
  followed by the logical-action bits against the logical pairs. Its
  row space enumerates the syndrome and logical action of every error
  supported on the erasure.
* :func:`uncorrectable_rank` is
  :math:`r_M = \\mathrm{rank}\\, M - \\mathrm{rank}\\, M_S` (the syndrome
  block alone), the number of independent zero-syndrome error
  directions with nontrivial logical action.
* :func:`recovery_probability` is the exact optimal-decoding recovery
  :math:`P(R \\mid e) = 2^{-r_M}` for the erasure pattern.
* :func:`expurgation_candidates` extracts the offending operators
  themselves via an augmented elimination
  :math:`[M \\mid I]`: eliminate on the syndrome columns, then the
  logical columns; each pivot row of the second block has zero
  syndrome and nontrivial logical action, and its identity-block
  witness bits reconstruct a Pauli supported on the erased sites.

Everything here is a function of the phase-free row data alone, which
is why no signed simulator is needed anywhere in the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from sparsegf2.core.linalg_gf2 import gf2_eliminate_on_columns, gf2_rank_bits
from sparsegf2.core.sparse_tableau import PAULI_X, PAULI_Z
from sparsegf2.errors import InvalidArgumentError
from sparsegf2.expurgation.roles import StabilizerCode, _exact_integer_array


@dataclass(frozen=True)
class Candidate:
    """A zero-syndrome, nontrivially-logical Pauli supported on an erasure.

    ``qubits`` and ``letters`` follow the sparse-Pauli convention of
    :meth:`sparsegf2.SparseGF2.measure_pauli`; ``weight`` is the Pauli
    weight ``len(qubits)``.
    """

    qubits: tuple[int, ...]
    letters: tuple[int, ...]

    @property
    def weight(self) -> int:
        return len(self.qubits)


def sample_erasure(
    n: int,
    rng: np.random.Generator,
    *,
    rate: float | None = None,
    count: int | None = None,
    sites: NDArray[np.int64] | None = None,
) -> NDArray[np.int64]:
    """Sample an erasure pattern: a sorted set of erased qubit indices.

    Exactly one of ``rate`` (each site erased independently with this
    probability) and ``count`` (a uniform random subset of exactly this
    many sites) must be given. ``sites`` restricts the pool of erasable
    sites; the default pool is ``range(n)``.
    """
    if (rate is None) == (count is None):
        raise InvalidArgumentError("give exactly one of rate= or count=")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise InvalidArgumentError(f"n must be a non-negative exact integer, got {n!r}")
    n = int(n)
    pool = (
        np.arange(n, dtype=np.int64) if sites is None else _exact_integer_array(sites, name="sites")
    )
    if pool.size:
        if pool.min() < 0 or pool.max() >= n:
            raise InvalidArgumentError(f"sites must be in [0, n={n})")
        if np.unique(pool).shape[0] != pool.shape[0]:
            raise InvalidArgumentError("sites must not repeat")
    if rate is not None:
        if (
            not isinstance(rate, Real)
            or isinstance(rate, (bool, np.bool_))
            or not math.isfinite(float(rate))
            or not 0.0 <= float(rate) <= 1.0
        ):
            raise InvalidArgumentError(f"rate must be in [0, 1], got {rate}")
        picked = pool[rng.random(pool.shape[0]) < float(rate)]
    else:
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
            raise InvalidArgumentError(f"count must be an exact integer, got {count!r}")
        count = int(count)
        if not 0 <= count <= pool.shape[0]:
            raise InvalidArgumentError(f"count must be in [0, {pool.shape[0]}], got {count}")
        picked = rng.choice(pool, size=count, replace=False)
    return np.sort(picked.astype(np.int64))


def _validate_erasure(code: StabilizerCode, erased: NDArray[np.int64]) -> NDArray[np.int64]:
    e = _exact_integer_array(erased, name="erased sites")
    if e.size:
        if e.min() < 0 or e.max() >= code.n:
            raise InvalidArgumentError(f"erased sites must be in [0, n={code.n})")
        if np.unique(e).shape[0] != e.shape[0]:
            raise InvalidArgumentError(f"erased sites must not repeat: got {e.tolist()}")
    return e


def uncorrectable_matrix(code: StabilizerCode, erased: NDArray[np.int64]) -> NDArray[np.uint8]:
    """Assemble :math:`M(S, L, e)` for an erasure pattern.

    Rows: the local error basis on the erased sites, :math:`Z_{i}` then
    :math:`X_{i}` for each site in ascending order (so row ``2t`` and
    ``2t + 1`` belong to site ``erased[t]``). Columns: syndrome bits
    against the checks (in :meth:`StabilizerCode.check_pairs` order),
    then logical bits (two per pair in
    :meth:`StabilizerCode.logical_pairs` order:
    :math:`\\bar{Z}` bit, then :math:`\\bar{X}` bit). Gauge pairs
    contribute no columns.

    Each row is one commutation query on the simulator, so assembly
    costs one pass over the erased sites' inverted indices and is
    independent of ``n`` at fixed tableau density.
    """
    e = _validate_erasure(code, erased)
    n_cols = len(code.check_pairs()) + 2 * code.k
    M = np.zeros((2 * e.shape[0], n_cols), dtype=np.uint8)
    for t in range(e.shape[0]):
        site = int(e[t])
        for j, letter in enumerate((PAULI_Z, PAULI_X)):
            syndrome, logical = code.commutation_bits([site], [letter])
            M[2 * t + j, : syndrome.shape[0]] = syndrome
            M[2 * t + j, syndrome.shape[0] :] = logical
    return M


def uncorrectable_rank(code: StabilizerCode, erased: NDArray[np.int64]) -> int:
    """Return :math:`r_M = \\mathrm{rank}\\, M - \\mathrm{rank}\\, M_S`.

    The number of independent error directions on the erasure that have
    zero syndrome but nontrivial logical action. ``0`` means the
    erasure is perfectly correctable.
    """
    M = uncorrectable_matrix(code, erased)
    n_s = len(code.check_pairs())
    return gf2_rank_bits(M) - gf2_rank_bits(M[:, :n_s])


def recovery_probability(code: StabilizerCode, erased: NDArray[np.int64]) -> float:
    """Exact optimal-decoding recovery probability :math:`2^{-r_M}`.

    For an erasure the erased qubits are replaced by maximally mixed
    states and their locations are known, so the optimal decoder
    succeeds with probability exactly :math:`2^{-r_M}` (Gullans et al.,
    Eqs. (17)-(18)). Phase-free data suffices: only ranks enter.
    """
    return float(2.0 ** (-uncorrectable_rank(code, erased)))


def expurgation_candidates(code: StabilizerCode, erased: NDArray[np.int64]) -> list[Candidate]:
    """Extract the uncorrectable error operators for an erasure pattern.

    Augment :math:`M` with an identity block, eliminate on the syndrome
    columns, then on the logical columns of the remaining rows. The
    logical-block pivot rows have zero syndrome and nontrivial logical
    action; because every row operation was mirrored on the identity
    block, their witness bits are the coefficients of the local error
    basis whose product realizes them. Witness bit ``2t`` contributes
    :math:`Z` and bit ``2t + 1`` contributes :math:`X` at site
    ``erased[t]`` (both set: :math:`Y`).

    Returns exactly :math:`r_M` candidates, sorted lightest first,
    which feeds the minimum-weight pivot's sparsity preservation and
    removes the lightest failure modes first.
    """
    e = _validate_erasure(code, erased)
    if e.shape[0] == 0:
        return []
    M = uncorrectable_matrix(code, e)
    n_s = len(code.check_pairs())
    two_k = 2 * code.k
    n_err = 2 * e.shape[0]
    A = np.concatenate([M, np.eye(n_err, dtype=np.uint8)], axis=1)
    A, r_s = gf2_eliminate_on_columns(A, np.arange(n_s))
    tail = A[r_s:]
    tail, r_m = gf2_eliminate_on_columns(tail, np.arange(n_s, n_s + two_k))
    out: list[Candidate] = []
    for row in tail[:r_m]:
        witness = row[n_s + two_k :]
        qubits: list[int] = []
        letters: list[int] = []
        for t in range(e.shape[0]):
            z_bit = int(witness[2 * t])
            x_bit = int(witness[2 * t + 1])
            if z_bit or x_bit:
                qubits.append(int(e[t]))
                letters.append((x_bit << 1) | z_bit)
        out.append(Candidate(qubits=tuple(qubits), letters=tuple(letters)))
    out.sort(key=lambda c: c.weight)
    return out
