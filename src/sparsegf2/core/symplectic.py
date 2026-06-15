"""Symplectic group ``Sp(2n, F_2)`` algebra and enumeration.

Generate, enumerate, and uniformly sample from
:math:`\\mathrm{Sp}(2n, \\mathbb{F}_2)` *without* importing Stim at
runtime. Stim is kept exclusively as a test-time cross-checker that the
elements and the sampling distribution agree.

This is what lets the circuit machinery build its 2-qubit Clifford cache
natively (each element converts to a ``(4, 4)`` GF(2) symplectic) instead
of going through an external library.

Conventions
===========

We work in the **(X_0, X_1, …, X_{n-1}, Z_0, Z_1, …, Z_{n-1})** basis
throughout. A 4-dim Pauli vector for 2 qubits is therefore
``(x_0, x_1, z_0, z_1)``. The symplectic form is

.. math::

    \\Omega = \\begin{pmatrix} 0_n & I_n \\\\ I_n & 0_n \\end{pmatrix},

and two Pauli strings :math:`u, v \\in \\mathbb{F}_2^{2n}` anticommute
iff :math:`u\\,\\Omega\\,v^T = 1 \\bmod 2`. A ``(2n, 2n)`` matrix ``S``
is symplectic iff :math:`S\\,\\Omega\\,S^T = \\Omega \\bmod 2`.

Row-vector convention: a single-Pauli row evolves as
``(x', z') = (x, z) @ S`` mod 2. The rows of ``S`` are therefore the
images of the basis vectors, in the same order
``(X_0, …, X_{n-1}, Z_0, …, Z_{n-1})``.

Algorithm: nested symplectic basis construction
================================================

For each index :math:`i = 1, \\dots, n`, pick a fresh symplectic pair
:math:`(X_i', Z_i')` in the symplectic complement of all previously
chosen pairs:

1. ``X_i'`` is any **nonzero** vector in the complement.
2. ``Z_i'`` is any vector in the complement that anticommutes with ``X_i'`` (symplectic product ``= 1``).

At step :math:`i` the complement has dimension :math:`2(n - i + 1)`, so
the number of choices is :math:`(2^{2(n-i+1)} - 1) \\cdot 2^{2(n-i+1)-1}`.
Taking the product over :math:`i = 1, \\dots, n` gives the standard
order

.. math::

    |\\mathrm{Sp}(2n, \\mathbb{F}_2)| = \\prod_{i=1}^{n} \\bigl(2^{2(n-i+1)} - 1\\bigr) \\, 2^{2(n-i+1)-1}.

Sanity check (n=2): :math:`(2^4 - 1)\\cdot 2^3 \\cdot (2^2 - 1)\\cdot 2^1
= 15 \\cdot 8 \\cdot 3 \\cdot 2 = 720`.

References
==========

* Aaronson & Gottesman, *Improved Simulation of Stabilizer Circuits*, Phys. Rev. A **70**, 052328 (2004).
* Koenig & Smolin, *How to efficiently select an arbitrary Clifford group element*, `arXiv:1406.2170 <https://arxiv.org/abs/1406.2170>`_.
* Bravyi & Maslov, *Hadamard-free circuits expose the structure of the Clifford group*, `arXiv:2003.09412 <https://arxiv.org/abs/2003.09412>`_.
"""

from __future__ import annotations

import threading as _threading

import numpy as np

from sparsegf2.core.linalg_gf2 import gf2_kernel_basis as _gf2_kernel_basis
from sparsegf2.errors import InvalidArgumentError, SimulatorBackendError

# ----------------------------------------------------------------------
# Symplectic form + primitives
# ----------------------------------------------------------------------


def symplectic_form(n: int) -> np.ndarray:
    """Return the ``(2n, 2n)`` symplectic form ``Ω`` in the X-Z block basis.

    :math:`\\Omega = \\begin{pmatrix} 0 & I_n \\\\ I_n & 0 \\end{pmatrix}` over GF(2).
    """
    if n < 0:
        raise InvalidArgumentError(f"n must be non-negative, got {n}")
    Omega = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    if n > 0:
        Omega[:n, n:] = np.eye(n, dtype=np.uint8)
        Omega[n:, :n] = np.eye(n, dtype=np.uint8)
    return Omega


def symplectic_product(u: np.ndarray, v: np.ndarray) -> int:
    """Compute ``⟨u, v⟩_Ω = u·Ω·v^T mod 2`` in the X-Z block basis.

    Both vectors must have even length ``2n``. Returns 0 if ``u`` and
    ``v`` (interpreted as Pauli strings) commute, 1 if they anticommute.
    """
    if u.shape != v.shape or u.ndim != 1 or u.shape[0] % 2 != 0:
        raise InvalidArgumentError(
            f"u and v must be 1-D F_2 vectors of even length, got {u.shape} and {v.shape}"
        )
    n = u.shape[0] // 2
    return (
        int(
            np.dot(u[:n].astype(np.int64), v[n:].astype(np.int64))
            + np.dot(u[n:].astype(np.int64), v[:n].astype(np.int64))
        )
        & 1
    )


def is_symplectic(S: np.ndarray) -> bool:
    """Check whether ``S`` is in ``Sp(2n, F_2)`` for the X-Z block form.

    Verifies :math:`S\\,\\Omega\\,S^T \\equiv \\Omega \\bmod 2`.
    """
    S = np.asarray(S, dtype=np.uint8)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] % 2 != 0:
        return False
    n = S.shape[0] // 2
    Omega = symplectic_form(n)
    lhs = ((S @ Omega @ S.T) & 1).astype(np.uint8)
    return bool(np.array_equal(lhs, Omega))


# ----------------------------------------------------------------------
# Group order
# ----------------------------------------------------------------------


def symplectic_group_order(n: int) -> int:
    """Return ``|Sp(2n, F_2)| = prod_{i=1..n} (2^{2(n-i+1)} - 1) · 2^{2(n-i+1)-1}``.

    Examples: ``order(1) = 6``, ``order(2) = 720``, ``order(3) = 1_451_520``.
    """
    if n < 0:
        raise InvalidArgumentError(f"n must be non-negative, got {n}")
    if n == 0:
        return 1
    total = 1
    for i in range(1, n + 1):
        d = 2 * (n - i + 1)
        total *= (2**d - 1) * (2 ** (d - 1))
    return total


# ----------------------------------------------------------------------
# Enumeration via the nested symplectic basis construction
# ----------------------------------------------------------------------


def _all_vectors(d: int) -> np.ndarray:
    """All ``2^d`` vectors in ``F_2^d`` as a ``(2^d, d)`` uint8 array.

    Row ``k`` holds the little-endian bits of integer ``k``: bit 0 at
    column 0, bit ``d-1`` at column ``d-1``. Row order is therefore the
    numeric order of ``k`` from 0 to ``2^d - 1``.
    """
    bits = np.zeros((1 << d, d), dtype=np.uint8)
    for k in range(1 << d):
        for b in range(d):
            bits[k, b] = (k >> b) & 1
    return bits


def _assemble_matrix(pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Assemble S from a list of n symplectic pairs ``(X_i', Z_i')``.

    Returns a ``(2n, 2n)`` matrix whose rows are
    ``[X_0', X_1', …, X_{n-1}', Z_0', Z_1', …, Z_{n-1}']``.
    """
    rows = [p[0] for p in pairs] + [p[1] for p in pairs]
    # All ``rows`` are uint8 by construction (``_all_vectors`` emits uint8),
    # so np.stack alone preserves the dtype without an extra cast/copy.
    return np.stack(rows, axis=0)


def _enumerate_recursive(
    n: int,
    chosen_pairs: list[tuple[np.ndarray, np.ndarray]],
    accumulator: list[np.ndarray],
) -> None:
    """Recursively pick all symplectic pairs ``(X_i', Z_i')`` in the
    complement of the already-chosen ones, appending the assembled
    ``(2n, 2n)`` matrix to ``accumulator`` whenever ``n`` pairs are
    in place."""
    if len(chosen_pairs) == n:
        accumulator.append(_assemble_matrix(chosen_pairs))
        return

    # Build the complement: vectors that commute with every previously
    # chosen X_j' and Z_j'. We work in the full 2n-dim space and just
    # filter; the complement has dimension 2(n - len(chosen)).
    all_vecs = _all_vectors(2 * n)
    constraints = []
    for X_prev, Z_prev in chosen_pairs:
        constraints.append(X_prev)
        constraints.append(Z_prev)
    if constraints:
        C = np.stack(constraints, axis=0).astype(np.uint8)
        # ⟨v, c⟩_Ω = c·Ω·v^T = 0 for every c ∈ C
        # We want v such that all (c·Ω) · v^T == 0.
        Omega = symplectic_form(n)
        # Each constraint maps v -> ⟨c, v⟩_Ω
        # We need the dot product of (c · Omega) with v to be 0.
        A = (C @ Omega) & 1  # (k, 2n)
        # Filter all_vecs to those satisfying A v^T == 0
        prods = (all_vecs @ A.T) & 1  # (2^{2n}, k)
        mask = np.all(prods == 0, axis=1)
        complement = all_vecs[mask]
    else:
        complement = all_vecs

    nonzero_complement = [v for v in complement if v.any()]
    for X_new in nonzero_complement:
        for Z_new in complement:
            if symplectic_product(X_new, Z_new) != 1:
                continue
            chosen_pairs.append((X_new, Z_new))
            _enumerate_recursive(n, chosen_pairs, accumulator)
            chosen_pairs.pop()


def enumerate_symplectic_group(n: int) -> np.ndarray:
    """Enumerate every element of ``Sp(2n, F_2)``.

    Returns a ``(|Sp(2n,F_2)|, 2n, 2n)`` uint8 array. Practical only for
    small ``n``: ``n=1`` gives 6, ``n=2`` gives 720, ``n=3`` gives
    1,451,520 (≈ 12 MB). ``n=4`` is infeasible (≈ 47 billion).

    The order in which matrices are emitted is deterministic but
    implementation-defined; do not rely on it being preserved across
    versions.
    """
    if n < 0:
        raise InvalidArgumentError(f"n must be non-negative, got {n}")
    if n == 0:
        return np.zeros((1, 0, 0), dtype=np.uint8)
    if n > 3:
        raise InvalidArgumentError(
            f"enumerate_symplectic_group(n={n}) is infeasible (>= 10^10 matrices). "
            "Use the Bravyi-Maslov uniform sampler instead."
        )
    matrices: list[np.ndarray] = []
    _enumerate_recursive(n, [], matrices)
    expected = symplectic_group_order(n)
    if len(matrices) != expected:
        raise SimulatorBackendError(
            f"enumeration produced {len(matrices)} matrices but expected {expected}"
        )
    return np.stack(matrices, axis=0).astype(np.uint8)


# ----------------------------------------------------------------------
# Cached 2-qubit enumeration + uniform sampler
# ----------------------------------------------------------------------


_SP4_CACHE: np.ndarray | None = None
_SP4_LOCK = _threading.Lock()


def enumerate_sp4() -> np.ndarray:
    """Return the cached ``(720, 4, 4)`` uint8 enumeration of ``Sp(4, F_2)``.

    The first call materializes the table; subsequent calls return the
    cached array. Safe to mutate a *copy* but never the returned view;
    treat it as read-only.

    **Thread-safety.** Initialization is guarded by a module-level lock
    with double-checked locking: once the cache is populated, all
    subsequent calls take the lock-free fast path. Pre-warm by calling
    :func:`enumerate_sp4` once from the main thread before fanning out
    if you want every worker to skip the lock entirely.
    """
    global _SP4_CACHE
    # Lock-free fast path once the cache is populated.
    cache = _SP4_CACHE
    if cache is not None:
        return cache
    with _SP4_LOCK:
        if _SP4_CACHE is None:
            tab = enumerate_symplectic_group(2)
            tab.setflags(write=False)
            _SP4_CACHE = tab
        return _SP4_CACHE


def random_symplectic_2q(rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample one element uniformly at random from ``Sp(4, F_2)``.

    Parameters
    ----------
    rng
        A ``numpy.random.Generator``. If ``None``, ``np.random.default_rng()``
        is used (non-reproducible). Pass an explicit ``rng`` in tests
        for determinism.

    Returns
    -------
    ndarray
        A ``(4, 4)`` uint8 GF(2) symplectic matrix.
    """
    if rng is None:
        rng = np.random.default_rng()
    cache = enumerate_sp4()
    idx = int(rng.integers(0, cache.shape[0]))
    return cache[idx].copy()


def random_symplectic_2q_batch(
    n_samples: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Draw ``n_samples`` uniform samples from ``Sp(4, F_2)``.

    Returns a ``(n_samples, 4, 4)`` uint8 array. Faster than calling
    :func:`random_symplectic_2q` in a loop because it issues a single
    batched ``rng.integers`` call and a single advanced-index lookup.
    """
    if rng is None:
        rng = np.random.default_rng()
    cache = enumerate_sp4()
    idx = rng.integers(0, cache.shape[0], size=n_samples)
    # Advanced indexing already returns a fresh writable array, so no
    # ``.copy()`` is needed here, unlike ``random_symplectic_2q`` which
    # uses a scalar index and returns a read-only view into the cache.
    return cache[idx]


# ----------------------------------------------------------------------
# Uniform sampler for arbitrary n via incremental kernel-basis sampling
# ----------------------------------------------------------------------
#
# Algorithm (informal). To sample uniformly from Sp(2n, F_2), build a
# symplectic basis {X_1', Z_1', ..., X_n', Z_n'} of F_2^{2n} one pair at
# a time. At step i:
#
#   1. Let W_i be the symplectic complement of {X_j', Z_j' : j < i}.
#      W_0 = F_2^{2n}; W_i has dimension 2(n - i + 1).
#   2. Pick X_i' = a uniformly-random nonzero vector in W_i.
#   3. Pick Z_i' = a uniformly-random vector in W_i with
#      <X_i', Z_i'>_Ω = 1.
#
# Counts: at step i, |W_i| - 1 = 2^{2(n-i+1)} - 1 nonzero choices for
# X_i', and 2^{2(n-i+1)-1} choices for Z_i' (the constraint slices the
# complement in half). The product over i recovers the standard
# |Sp(2n, F_2)| = prod_{i=1..n} (2^{2(n-i+1)} - 1) · 2^{2(n-i+1)-1}.
#
# Implementation: we maintain a basis K of W_i as the kernel of the
# constraint matrix that encodes "u is symplectically orthogonal to
# every previous pair". K is recomputed at each step via GF(2) Gauss
# elimination of an (2(i-1), 2n) constraint matrix; O(n^3) total.
#
# This is the Stim-disconnect deliverable (no runtime Stim). It is the
# ``kernel_basis`` sampler; the asymptotically-faster Bravyi-Maslov sampler
# (O(n^2) bit-ops via the canonical decomposition C = F_1 H S F_2,
# arXiv:2003.09412) is also implemented below in
# ``_random_symplectic_bravyi_maslov`` and is the ``auto`` default for n != 2.
# Both are uniform over Sp(2n, F_2); kernel_basis is the simpler to verify.


def _constraints_from_pair(x_row: np.ndarray, z_row: np.ndarray, n: int) -> np.ndarray:
    """Pack the two symplectic-orthogonality constraints into 2 rows.

    For a vector u ∈ F_2^{2n}, the constraint ``<u, w>_Ω = 0`` is
    ``u_x · w_z + u_z · w_x = 0 mod 2``, which is the linear
    constraint ``[w_z | w_x] · u^T = 0``. So the constraint matrix
    row for ``w`` is ``[w_z | w_x]``, i.e. ``w`` with its two halves
    swapped.

    Returns a ``(2, 2n)`` matrix.
    """
    row_x = np.concatenate([x_row[n:], x_row[:n]])
    row_z = np.concatenate([z_row[n:], z_row[:n]])
    return np.stack([row_x, row_z], axis=0).astype(np.uint8) & 1


_SAMPLERS = ("auto", "kernel_basis", "bravyi_maslov")


def random_symplectic(
    n: int,
    rng: np.random.Generator | None = None,
    *,
    sampler: str = "auto",
) -> np.ndarray:
    """Sample one element uniformly at random from ``Sp(2n, F_2)``.

    Two uniform samplers are available, selected by ``sampler``:

    - ``"bravyi_maslov"``: the canonical-form sampler of Bravyi & Maslov
      (`arXiv:2003.09412 <https://arxiv.org/abs/2003.09412>`_): a
      Mallows-distributed permutation + Hadamard layer wrapped by two random
      Hadamard-free factors. ``O(n^2)`` bit-operations (``O(n^3)`` here, dense
      numpy matrix products), so much faster than the kernel-basis construction
      at large ``n``.
    - ``"kernel_basis"``: the incremental nested-symplectic-basis construction
      (pick ``X_i'`` then ``Z_i'`` in the running symplectic complement). Simple
      to verify; the historical default.
    - ``"auto"`` (default): ``kernel_basis`` for the cached ``n = 2`` table
      (so the RNG state matches :func:`random_symplectic_2q`), and
      ``bravyi_maslov`` otherwise.

    Both are uniform over ``Sp(2n, F_2)`` (verified by chi-squared against the
    enumeration and by cross-check with Stim); they differ only in speed and in
    the exact RNG draws they consume.

    Parameters
    ----------
    n
        Number of qubits. Must be non-negative.
    rng
        A ``numpy.random.Generator``. If ``None``, ``np.random.default_rng()``
        is used (non-reproducible). Pass an explicit ``rng`` in tests
        for determinism.
    sampler
        One of ``"auto"`` / ``"kernel_basis"`` / ``"bravyi_maslov"``.

    Returns
    -------
    ndarray of shape (2n, 2n), uint8
        A GF(2) symplectic matrix in the ``(X_0, …, X_{n-1}, Z_0, …, Z_{n-1})``
        row-vector basis.
    """
    if n < 0:
        raise InvalidArgumentError(f"n must be non-negative, got {n}")
    if sampler not in _SAMPLERS:
        raise InvalidArgumentError(f"sampler must be one of {_SAMPLERS}; got {sampler!r}")
    if rng is None:
        rng = np.random.default_rng()
    if n == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if sampler == "bravyi_maslov":
        return _random_symplectic_bravyi_maslov(n, rng)
    # auto / kernel_basis keep the cached-table short-circuit at n = 2 so the RNG
    # state advances identically to random_symplectic_2q.
    if n == 2:
        return random_symplectic_2q(rng)
    if sampler == "kernel_basis":
        return _random_symplectic_kernel_basis(n, rng)
    return _random_symplectic_bravyi_maslov(n, rng)  # auto, n != 2


def _random_symplectic_kernel_basis(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform ``Sp(2n, F_2)`` via the incremental nested-symplectic-basis
    construction (the verified historical sampler)."""
    rows_x: list[np.ndarray] = []
    rows_z: list[np.ndarray] = []

    # No constraints initially; complement_basis is the standard basis of F_2^{2n}.
    constraint_M = np.zeros((0, 2 * n), dtype=np.uint8)

    for i in range(n):
        # ``complement_basis`` spans W_i = the symplectic complement of all
        # previously chosen (X_j', Z_j'). It is computed as the right
        # null-space ("GF(2) kernel") of the constraint matrix, hence
        # the call to ``_gf2_kernel_basis``; these are two names for
        # the same subspace (algebra vs the function's name).
        complement_basis = _gf2_kernel_basis(constraint_M)
        n_kernel = complement_basis.shape[0]
        # By construction, n_kernel = 2 * (n - i).
        # Sample X_i' = random nonzero linear combination of basis rows.
        while True:
            coeffs = rng.integers(0, 2, size=n_kernel, dtype=np.uint8)
            if int(coeffs.sum()) > 0:
                break
        x_new = (coeffs @ complement_basis) & 1  # shape (2n,)

        # Sample Z_i' = random vector with <x_new, z_new>_Ω = 1.
        # Compute <x_new, complement_basis[k]> for each k.
        # <x_new, w> = x_new[:n] · w[n:] + x_new[n:] · w[:n].
        x_lo, x_hi = x_new[:n], x_new[n:]
        prods = (complement_basis[:, n:] @ x_lo + complement_basis[:, :n] @ x_hi) & 1
        # The set of coefficient vectors c with <x_new, c @ complement_basis>_Ω = 1
        # is exactly {c : c · prods = 1 (mod 2)}, which is half of all
        # 2^{n_kernel} coefficient vectors. Sample uniformly: pick c
        # randomly, accept if the parity matches; if it doesn't, flip
        # any one bit in c where prods[k] = 1 (then parity flips).
        coeffs2 = rng.integers(0, 2, size=n_kernel, dtype=np.uint8)
        parity = int((coeffs2 * prods).sum()) & 1
        if parity == 0:
            # Find any index where prods = 1 and flip coeffs2 there.
            # `prods` must contain at least one 1: if every basis row
            # commuted with x_new, then x_new would lie in the complement
            # subspace's symplectic radical, but the complement is a
            # non-degenerate symplectic subspace by construction. `raise`
            # so the check survives `python -O`.
            anti = np.where(prods == 1)[0]
            if anti.size == 0:
                raise SimulatorBackendError(
                    "random_symplectic: complement subspace has degenerate symplectic form "
                    f"(non-degeneracy invariant broken at step {i})"
                )
            flip = int(anti[rng.integers(0, anti.size)])
            coeffs2[flip] ^= 1
        z_new = (coeffs2 @ complement_basis) & 1

        # Defensive: <x_new, z_new>_Ω must be 1.
        if symplectic_product(x_new, z_new) != 1:
            raise SimulatorBackendError(
                f"random_symplectic: sampler bug at step {i}: <X_i, Z_i>_Ω != 1"
            )

        rows_x.append(x_new)
        rows_z.append(z_new)
        constraint_M = np.concatenate(
            [constraint_M, _constraints_from_pair(x_new, z_new, n)], axis=0
        )

    S = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    for i in range(n):
        S[i] = rows_x[i]
        S[n + i] = rows_z[i]
    return S


# ----------------------------------------------------------------------
# Bravyi-Maslov canonical-form sampler (arXiv:2003.09412)
# ----------------------------------------------------------------------


def _gf2_inverse(M: np.ndarray) -> np.ndarray:
    """Inverse of an invertible square GF(2) matrix via Gauss-Jordan on [M | I]."""
    n = M.shape[0]
    aug = np.concatenate([np.asarray(M, dtype=np.uint8) & 1, np.eye(n, dtype=np.uint8)], axis=1)
    # M is always invertible (unit lower-triangular delta), so the pivot for
    # column ``col`` sits at or below row ``col``, so the pivot row tracks ``col``.
    for col in range(n):
        piv = next((r for r in range(col, n) if aug[r, col]), -1)
        if piv < 0:  # pragma: no cover - delta is always unit lower-triangular
            raise SimulatorBackendError("random_symplectic (Bravyi-Maslov): singular matrix")
        if piv != col:
            aug[[col, piv]] = aug[[piv, col]]
        for r in range(n):
            if r != col and aug[r, col]:
                aug[r] ^= aug[col]
    return aug[:, n:]


def _sample_qmallows(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``(had, perm)`` from the quantum-Mallows distribution.

    ``had[i]`` marks a Hadamard on the qubit placed at position ``i``; ``perm``
    is the qubit permutation. The non-uniform Mallows weights are what make the
    overall canonical form sample uniformly despite the free Hadamard-free
    factors (Bravyi & Maslov 2020, App.).
    """
    had = np.zeros(n, dtype=bool)
    perm = np.zeros(n, dtype=np.int64)
    inds = list(range(n))
    for i in range(n):
        m = n - i
        eps = 4.0 ** (-m)
        r = rng.uniform(0.0, 1.0)
        index = -int(np.ceil(np.log2(r + (1.0 - r) * eps)))
        # ``index`` lies in [0, 2m-1]; clamp the boundary value (reachable only
        # when r is exactly 0.0) so ``k`` below never becomes -1 (which would
        # silently wrap to the last index of ``inds``).
        index = min(index, 2 * m - 1)
        had[i] = index < m
        k = index if index < m else 2 * m - index - 1
        perm[i] = inds[k]
        del inds[k]
    return had, perm


def _fill_tril(mat: np.ndarray, rng: np.random.Generator, *, symmetric: bool = False) -> None:
    """Fill the strict lower triangle of ``mat`` with random bits (in place);
    mirror to the upper triangle when ``symmetric``."""
    dim = mat.shape[0]
    if dim < 2:
        return
    rows, cols = np.tril_indices(dim, -1)
    vals = rng.integers(0, 2, size=rows.size, dtype=np.int64)
    mat[rows, cols] = vals
    if symmetric:
        mat[cols, rows] = vals


def _random_symplectic_bravyi_maslov(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform ``Sp(2n, F_2)`` via the Bravyi-Maslov canonical form.

    Builds two Hadamard-free factors from random ``delta`` (unit lower-triangular)
    and ``gamma`` (symmetric), a Hadamard layer ``had`` and a Mallows-distributed
    permutation ``perm``, and composes them. Returns the **phase-free** ``(2n, 2n)``
    symplectic part directly; unlike qiskit's ``random_clifford`` this never
    samples the random sign/phase vector at all.
    """
    had, perm = _sample_qmallows(n, rng)
    gamma1 = np.diag(rng.integers(0, 2, size=n, dtype=np.int64))
    gamma2 = np.diag(rng.integers(0, 2, size=n, dtype=np.int64))
    delta1 = np.eye(n, dtype=np.int64)
    delta2 = np.eye(n, dtype=np.int64)
    _fill_tril(gamma1, rng, symmetric=True)
    _fill_tril(gamma2, rng, symmetric=True)
    _fill_tril(delta1, rng)
    _fill_tril(delta2, rng)

    zero = np.zeros((n, n), dtype=np.int64)
    prod1 = (gamma1 @ delta1) % 2
    prod2 = (gamma2 @ delta2) % 2
    inv1 = _gf2_inverse(delta1).astype(np.int64).T % 2
    inv2 = _gf2_inverse(delta2).astype(np.int64).T % 2
    table1 = np.block([[delta1, zero], [prod1, inv1]]) % 2
    table2 = np.block([[delta2, zero], [prod2, inv2]]) % 2

    # Apply the permutation (row-permute) then the Hadamard layer (swap the X/Z
    # rows of the Hadamarded qubits), then compose with the left factor.
    table = table2[np.concatenate([perm, n + perm])]
    inds = had * np.arange(1, n + 1)
    inds = inds[inds > 0] - 1
    lhs = np.concatenate([inds, inds + n])
    rhs = np.concatenate([inds + n, inds])
    table[lhs, :] = table[rhs, :]

    return ((table1 @ table) % 2).astype(np.uint8)


def random_symplectic_batch(
    n: int,
    n_samples: int,
    rng: np.random.Generator | None = None,
    *,
    sampler: str = "auto",
) -> np.ndarray:
    """Draw ``n_samples`` uniform samples from ``Sp(2n, F_2)``.

    Returns a ``(n_samples, 2n, 2n)`` uint8 array. For ``n = 2`` (with the
    table-backed samplers) this routes to :func:`random_symplectic_2q_batch`;
    otherwise it calls :func:`random_symplectic` in a loop with ``sampler``.
    """
    # Validate up front so the contract is independent of n_samples (an empty
    # batch would otherwise never reach the per-sample validation).
    if n < 0:
        raise InvalidArgumentError(f"n must be non-negative, got {n}")
    if sampler not in _SAMPLERS:
        raise InvalidArgumentError(f"sampler must be one of {_SAMPLERS}; got {sampler!r}")
    if rng is None:
        rng = np.random.default_rng()
    if n == 2 and sampler in ("auto", "kernel_basis"):
        return random_symplectic_2q_batch(n_samples, rng=rng)
    out = np.empty((n_samples, 2 * n, 2 * n), dtype=np.uint8)
    for i in range(n_samples):
        out[i] = random_symplectic(n, rng=rng, sampler=sampler)
    return out
