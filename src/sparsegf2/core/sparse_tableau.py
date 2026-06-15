"""Sparse stabilizer tableau over GF(2).

This module provides :class:`SparseGF2`, a phase-free Aaronson-Gottesman
stabilizer simulator that stores the symplectic part only of the
tableau using four complementary representations of the same bits:

1. **Dense Pauli Lookup Table** (``plt``) of shape ``(2n, n)``, encoding
   each generator's Pauli at each qubit as a 2-bit integer
   ``xz = (x << 1) | z`` (so ``I = 0, Z = 1, X = 2, Y = 3``). O(1)
   random read.
2. **Per-generator support list** (``supp_q``, ``supp_len``) plus an
   O(1) back-pointer ``supp_pos``, so we can iterate a generator's
   nonzero qubits in O(weight) and remove a qubit in O(1).
3. **Per-qubit inverted index of all touching generators**
   (``inv``, ``inv_len``) plus back-pointer ``inv_pos``. Lets every
   gate touch only the generators with support on its qubits.
4. **Per-qubit inverted index of generators with x-bit set**
   (``inv_x``, ``inv_x_len``) plus back-pointer ``inv_x_pos``. Used
   by the measurement kernel to find anticommuting generators in O(1).

Initialization
==============

* The full set of data structures is allocated up-front to the bounds
  documented in :func:`_max_support` and :func:`_max_inv`, so the JIT
  kernels can be dropped in without re-shaping.
* Initialization to the pure :math:`|0^n\\rangle` state on exactly ``n``
  qubits (rows ``0..n-1`` = destabilizers :math:`X_i`, rows ``n..2n-1``
  = stabilizers :math:`Z_i`).
* Symplectic extraction (:meth:`SparseGF2.to_symplectic`): a
  ``(2n, 2n)`` ``uint8`` matrix ``[X | Z]`` in the Aaronson-Gottesman
  layout.

Implicit purification semantics are deliberately absent. ``SparseGF2(n)``
is exactly ``n`` qubits. To purify, use
:func:`from_bell_purification(n_system)` for a ready-made Bell-pair
purification or build one by hand: ``SparseGF2(2 * n)`` followed by
``H(i); CX(i, i + n)`` for ``i ∈ [0, n)``.

Public kernels
==============

* **Single-qubit** Cliffords: :meth:`SparseGF2.apply_h`, :meth:`apply_s`,
  :meth:`apply_sqrt_x`, and :meth:`apply_gate_1q` for arbitrary
  user-supplied ``(2, 2)`` symplectic matrices. The kernel walks
  ``inv[q]`` (the inverted index of all generators touching qubit ``q``)
  and rewrites each PLT entry via a 4-entry LUT.
* **Two-qubit** Cliffords: :meth:`apply_cx`, :meth:`apply_cz`,
  :meth:`apply_swap`, and :meth:`apply_gate_2q` for arbitrary ``(4, 4)``
  symplectics produced by :mod:`sparsegf2.core.symplectic` (the
  Stim-free :math:`\\mathrm{Sp}(4, \\mathbb{F}_2)` sampler). The kernel
  collects the deduplicated set of generators touching ``qi`` or
  ``qj`` and pushes each through a 16-entry LUT, patching ``supp``,
  ``inv``, and ``inv_x`` on both qubits in O(1) per affected bit.
* **Measurement**: :meth:`measure_z`, :meth:`measure_x`, :meth:`measure_y`,
  :meth:`is_deterministic_z`. Phase-free contract: deterministic
  outcomes return ``0`` regardless of the physical eigenvalue.
* **Reset/projection**: :meth:`reset_z`, :meth:`reset_x`, :meth:`reset_y`.
* **State extraction**: :meth:`to_symplectic` (raw ``[X | Z]``),
  :meth:`canonical_form` (GF(2) RREF of the stabilizer subspace),
  :meth:`copy` (snapshot/restore for trajectory branching).

Numba JIT mirrors of every kernel live in
:mod:`sparsegf2.core.numba_kernels`; both paths take the same numpy
arrays so the JIT version is a drop-in
(``SparseGF2(n, use_numba=True/False)`` toggles per-instance).

Pauli encoding
==============

Each PLT entry is a 2-bit integer xz::

    I = 0      Z = 1      X = 2      Y = 3
    x bit = (xz >> 1) & 1
    z bit = xz & 1

A generator's ``X`` bit on qubit *q* is set when its Pauli on *q* is
``X`` or ``Y``; the ``Z`` bit is set when it is ``Y`` or ``Z``.

Reference
=========

Aaronson & Gottesman, "Improved Simulation of Stabilizer Circuits",
Phys. Rev. A 70, 052328 (2004), `arXiv:quant-ph/0406196
<https://arxiv.org/abs/quant-ph/0406196>`_.
"""

from __future__ import annotations

import copy as _copy
import threading as _threading
from collections.abc import Iterable
from typing import Self

import numpy as np

from sparsegf2.core.linalg_gf2 import gf2_rref
from sparsegf2.core.symplectic import is_symplectic as _is_symplectic
from sparsegf2.errors import (
    InvalidArgumentError,
    SymplecticConditionError,
    TableauCorruption,
)

try:  # pragma: no cover, exercised by the Numba-parity tests
    from sparsegf2.core import numba_kernels as _nk

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _nk = None
    _HAS_NUMBA = False

# Capacity heuristics
# -------------------
#
# These size the scratch arrays that back the inverted indices and the
# per-generator support lists. Upper bounds, with proofs:
#
# - ``_max_support(n) = n``. A single generator has support on at most
#   every qubit. There is no `supp_q[r, n]` write site anywhere in the
#   kernels, because every kernel writes support entries indexed by
#   `supp_len[r]` and increments after, with `supp_len[r] <= n` after
#   any sequence of `_add_to_support` / `_remove_from_support` calls.
# - ``_max_inv(n) = 2n``. The inverted index `inv[q]` holds at most one
#   entry per generator, and there are `N = 2n` generators total.
#   Identical bound for `inv_x[q]`. This tight bound is provably
#   sufficient (no need for extra slack).


def _max_support(n: int) -> int:
    """Max qubits a single generator can touch.

    A generator has support on at most every qubit, so the tight bound
    is ``n``. We return ``max(1, n)`` so that arrays for ``n = 0`` are
    at least 1-wide (numpy disallows zero-stride arithmetic on shape-0
    axes in some JIT paths).
    """
    return max(1, n)


def _max_inv(n: int) -> int:
    """Max generators that can be indexed under a single qubit.

    There are ``N = 2n`` generators total, so any per-qubit inverted
    index holds at most ``2n`` entries. We return ``max(1, 2n)`` for
    the same shape-0 reason as :func:`_max_support`.
    """
    return max(1, 2 * n)


# Pauli encoding
# --------------
#
# PLT entries are 2-bit integers (x << 1) | z. Constants are exported so
# tests and the notebooks can refer to them by name.
PAULI_I = np.uint8(0)
PAULI_Z = np.uint8(1)
PAULI_X = np.uint8(2)
PAULI_Y = np.uint8(3)


# Named two-qubit Clifford generators (4x4 GF(2) symplectic matrices)
# -------------------------------------------------------------------
#
# Row-vector convention: rows are images of basis vectors
# (X_qi, X_qj, Z_qi, Z_qj). For each named gate the matrix is verified
# against Stim in the two-qubit-gate parity tests.
#
# CX (control qi, target qj):  X_qi -> X_qi X_qj,  Z_qj -> Z_qi Z_qj
CX_SYMP = np.array(
    [
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
    ],
    dtype=np.uint8,
)

# CZ (symmetric):  X_qi -> X_qi Z_qj,  X_qj -> Z_qi X_qj
CZ_SYMP = np.array(
    [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.uint8,
)

# SWAP: exchange qubits
SWAP_SYMP = np.array(
    [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=np.uint8,
)


# Named single-qubit Clifford generators (2x2 GF(2) symplectic matrices)
# -----------------------------------------------------------------------
#
# Row-vector convention: ``(x', z') = (x, z) @ S2``. Each matrix is in
# the ``(X, Z)`` basis, so column 0 is the image of X under the gate
# (in symplectic coords) and column 1 is the image of Z.
#
# Reference: Aaronson & Gottesman 2004, Table I (phase bits ignored
# since we are phase-free).
#
# H : X <-> Z
# S : X -> Y, Z -> Z       => (x, z) -> (x, x ^ z)
# √X: Z -> Y, X -> X       => (x, z) -> (x ^ z, z)
H_SYMP = np.array([[0, 1], [1, 0]], dtype=np.uint8)
S_SYMP = np.array([[1, 1], [0, 1]], dtype=np.uint8)
SQRT_X_SYMP = np.array([[1, 0], [1, 1]], dtype=np.uint8)


def _build_1q_lut_uncached(S2: np.ndarray) -> np.ndarray:
    """Build a 4-entry LUT for the 1q gate given by 2x2 symplectic ``S2``.

    Input index encodes the *old* Pauli at the gated qubit as
    ``(x << 1) | z``; the LUT entry encodes the *new* Pauli the same way.

    The kernel uses this LUT instead of a per-generator matrix multiply
    so the inner loop becomes a single array lookup. Pure-Python (not
    `@njit`-friendly), so callers should go through :func:`_build_1q_lut`
    which caches by `S2.tobytes()`.
    """
    if S2.shape != (2, 2):
        raise InvalidArgumentError(f"S2 must be (2, 2), got {S2.shape}")
    S = S2.astype(np.uint8) & 1
    lut = np.empty(4, dtype=np.uint8)
    for inp in range(4):
        bx = (inp >> 1) & 1
        bz = inp & 1
        nx = (bx * int(S[0, 0]) + bz * int(S[1, 0])) & 1
        nz = (bx * int(S[0, 1]) + bz * int(S[1, 1])) & 1
        lut[inp] = (nx << 1) | nz
    return lut


# LUT cache. Each entry is the 4-entry uint8 LUT for the symplectic
# whose 4 bytes are the dict key. Random gates can reuse cache slots
# (there are only 6 elements of Sp(2, F_2)) and named gates hit the
# precomputed slots populated at module load (see bottom of module).
#
# **Thread-safety.** The cache is mutated through the read-check-then-write
# pattern in :func:`_build_1q_lut`. CPython's GIL makes each individual
# dict op atomic, but two threads can both observe a cache miss and
# both build+insert the same LUT: wasted work, never corruption (the
# resulting LUTs are deterministic). SparseGF2 is single-threaded by
# contract; if you fan-out a workload across threads, import
# ``sparsegf2`` once from the main thread first so the named-gate slots
# are populated before any worker starts.
_LUT1Q_CACHE: dict[tuple[tuple[int, ...], bytes], np.ndarray] = {}
# Lock guarding ``_LUT1Q_CACHE`` writes. Reads are GIL-atomic in CPython,
# so we lock only the build-and-insert critical section. The double-
# checked-locking pattern below is intentional: a thread that observes
# a cache hit doesn't need to acquire the lock at all.
_LUT1Q_LOCK = _threading.Lock()


def _build_1q_lut(S2: np.ndarray) -> np.ndarray:
    """Cached version of :func:`_build_1q_lut_uncached`.

    Returns the **same** read-only LUT object across calls with the
    same ``(shape, bytes)`` pair. Cache hits skip shape + symplectic
    validation, because once we have a LUT for these bytes at this
    shape, the input *is* a valid ``Sp(2, F_2)`` element.

    The shape is part of the cache key so a 1D 4-byte array cannot
    collide with a ``(2, 2)`` 4-byte array (the bytes are identical
    but only the latter is a valid 1q symplectic).

    Thread-safe: read-check-then-write is guarded by a module-level
    lock with double-checked locking; a cache hit is lock-free.
    """
    arr = np.asarray(S2, dtype=np.uint8)
    key = (arr.shape, arr.tobytes())
    cached = _LUT1Q_CACHE.get(key)
    if cached is not None:
        return cached
    if arr.shape != (2, 2):
        raise InvalidArgumentError(f"S2 must be (2, 2), got {arr.shape}")
    if not _is_symplectic(arr):
        raise SymplecticConditionError(
            f"S2 is not symplectic (S2 @ Omega @ S2.T != Omega mod 2):\n{arr}"
        )
    with _LUT1Q_LOCK:
        # Re-check after acquiring the lock: another thread may have
        # built and inserted this entry between our check and acquisition.
        cached = _LUT1Q_CACHE.get(key)
        if cached is not None:
            return cached
        lut = _build_1q_lut_uncached(arr)
        lut.flags.writeable = False
        _LUT1Q_CACHE[key] = lut
    return lut


# ----------------------------------------------------------------------
# Inverted-index O(1) primitives (pure Python)
# ----------------------------------------------------------------------
#
# These keep ``inv``, ``inv_x``, ``supp_q`` and their position maps
# consistent under add/remove with a swap-with-last trick. They are
# called from the gate kernels below; the JIT path uses byte-identical
# mirrors in :mod:`sparsegf2.core.numba_kernels`
# (``_add_to_inv``, ``_remove_from_inv``, etc.).


def _add_to_inv(
    inv: np.ndarray,
    inv_len: np.ndarray,
    inv_pos: np.ndarray,
    q: int,
    r: int,
) -> None:
    pos = inv_len[q]
    inv[q, pos] = r
    inv_pos[r, q] = pos
    inv_len[q] += 1


def _remove_from_inv(
    inv: np.ndarray,
    inv_len: np.ndarray,
    inv_pos: np.ndarray,
    q: int,
    r: int,
) -> None:
    pos = inv_pos[r, q]
    if pos < 0:
        return
    last = inv_len[q] - 1
    if pos != last:
        j = inv[q, last]
        inv[q, pos] = j
        inv_pos[j, q] = pos
    inv_len[q] -= 1
    inv_pos[r, q] = -1


def _add_to_inv_x(
    inv_x: np.ndarray,
    inv_x_len: np.ndarray,
    inv_x_pos: np.ndarray,
    q: int,
    r: int,
) -> None:
    pos = inv_x_len[q]
    inv_x[q, pos] = r
    inv_x_pos[r, q] = pos
    inv_x_len[q] += 1


def _remove_from_inv_x(
    inv_x: np.ndarray,
    inv_x_len: np.ndarray,
    inv_x_pos: np.ndarray,
    q: int,
    r: int,
) -> None:
    pos = inv_x_pos[r, q]
    if pos < 0:
        return
    last = inv_x_len[q] - 1
    if pos != last:
        j = inv_x[q, last]
        inv_x[q, pos] = j
        inv_x_pos[j, q] = pos
    inv_x_len[q] -= 1
    inv_x_pos[r, q] = -1


def _add_to_support(
    supp_q: np.ndarray,
    supp_len: np.ndarray,
    supp_pos: np.ndarray,
    r: int,
    q: int,
) -> None:
    pos = supp_len[r]
    supp_q[r, pos] = q
    supp_pos[r, q] = pos
    supp_len[r] += 1


def _remove_from_support(
    supp_q: np.ndarray,
    supp_len: np.ndarray,
    supp_pos: np.ndarray,
    r: int,
    q: int,
) -> None:
    pos = supp_pos[r, q]
    if pos < 0:
        return
    last = supp_len[r] - 1
    if pos != last:
        j = supp_q[r, last]
        supp_q[r, pos] = j
        supp_pos[r, j] = pos
    supp_len[r] -= 1
    supp_pos[r, q] = -1


# ----------------------------------------------------------------------
# Single-qubit gate kernel
# ----------------------------------------------------------------------


def _build_2q_lut_uncached(S: np.ndarray) -> np.ndarray:
    """Build a 16-entry LUT for the 2q gate given by 4x4 symplectic ``S``.

    The index is the *old* two-qubit Pauli packed as ``(xz_i << 2) | xz_j`` with
    ``xz_q = (x_q << 1) | z_q`` in ``[0, 4)``, i.e. exactly the two ``plt``
    bytes the kernel already holds for qubits ``(qi, qj)``. The entry is the
    *new* Pauli packed the same way (``(new_xz_i << 2) | new_xz_j``). So the
    kernel's inner loop is one array lookup with the packed bytes, with no
    per-generator unpack/repack of the four bits. Callers should go through
    :func:`_build_2q_lut`, which caches by ``S.tobytes()``.
    """
    if S.shape != (4, 4):
        raise InvalidArgumentError(f"S must be (4, 4), got {S.shape}")
    Sm = S.astype(np.uint8) & 1
    lut = np.empty(16, dtype=np.uint8)
    for xz_i in range(4):
        for xz_j in range(4):
            # basis order is (X_qi, X_qj, Z_qi, Z_qj); (x', z') = (x, z) @ S
            b = [(xz_i >> 1) & 1, (xz_j >> 1) & 1, xz_i & 1, xz_j & 1]
            out = [0, 0, 0, 0]
            for col in range(4):
                acc = 0
                for row in range(4):
                    acc ^= b[row] & int(Sm[row, col])
                out[col] = acc
            new_xz_i = (out[0] << 1) | out[2]
            new_xz_j = (out[1] << 1) | out[3]
            lut[(xz_i << 2) | xz_j] = (new_xz_i << 2) | new_xz_j
    return lut


# 2q LUT cache. Sp(4, F_2) has 720 elements, so this dict tops out at
# 720 entries no matter how long the simulation runs.
_LUT2Q_CACHE: dict[tuple[tuple[int, ...], bytes], np.ndarray] = {}
_LUT2Q_LOCK = _threading.Lock()


def _build_2q_lut(S: np.ndarray) -> np.ndarray:
    """Cached version of :func:`_build_2q_lut_uncached`.

    Returns the **same** read-only LUT object across calls with the same
    ``(shape, bytes)`` pair. The cache is bounded by ``|Sp(4, F_2)|`` (720);
    even an infinite stream of uniformly-sampled gates only populates 720
    cache slots. See :func:`_build_1q_lut` for the rationale on including
    ``shape`` in the cache key, and for the thread-safety contract.
    """
    arr = np.asarray(S, dtype=np.uint8)
    key = (arr.shape, arr.tobytes())
    cached = _LUT2Q_CACHE.get(key)
    if cached is not None:
        return cached
    if arr.shape != (4, 4):
        raise InvalidArgumentError(f"S must be (4, 4), got {arr.shape}")
    if not _is_symplectic(arr):
        raise SymplecticConditionError(
            f"S is not symplectic (S @ Omega @ S.T != Omega mod 2):\n{arr}"
        )
    with _LUT2Q_LOCK:
        cached = _LUT2Q_CACHE.get(key)
        if cached is not None:
            return cached
        lut = _build_2q_lut_uncached(arr)
        lut.flags.writeable = False
        _LUT2Q_CACHE[key] = lut
    return lut


# Pre-warm the LUT caches with the named gates so the first apply_*
# call doesn't pay the build/validation cost.
for _S2 in (H_SYMP, S_SYMP, SQRT_X_SYMP):
    _build_1q_lut(_S2)
for _S in (CX_SYMP, CZ_SYMP, SWAP_SYMP):
    _build_2q_lut(_S)
del _S2, _S


def _apply_1q_kernel(
    plt: np.ndarray,
    inv: np.ndarray,
    inv_len: np.ndarray,
    inv_x: np.ndarray,
    inv_x_len: np.ndarray,
    inv_x_pos: np.ndarray,
    q: int,
    lut: np.ndarray,
) -> None:
    """Apply a single-qubit Clifford gate at qubit ``q``.

    Walks the per-qubit inverted index ``inv[q]``, rewrites each touching
    generator's PLT entry via a 4-entry LUT, and patches the ``inv_x[q]``
    membership whenever the x-bit at qubit ``q`` flips.

    ``lut`` is the 4-entry uint8 LUT for the gate's 2x2 symplectic,
    built and cached by :func:`_build_1q_lut`. The kernel signature
    matches :func:`sparsegf2.core.numba_kernels.apply_1q_kernel_jit` so
    the cached LUT can be shared between the Python and JIT paths.

    The gate is invertible on F_2, so a nonzero PLT entry stays nonzero,
    and we never have to touch ``inv``, ``supp_q``, or ``supp_len``.

    Cost: ``O(inv_len[q])`` per call.
    """
    n_active = int(inv_len[q])
    for idx in range(n_active):
        r = int(inv[q, idx])
        old_xz = int(plt[r, q])
        # r in inv[q] is an invariant: plt[r, q] must be non-zero. A zero
        # here means inv has desynced from plt: the four-data-structure
        # bug class the inverted-index design is most vulnerable to.
        # The Numba mirror keeps a silent `continue` for performance.
        # `raise` (not `assert`) so the check survives `python -O`.
        if old_xz == 0:
            raise TableauCorruption(
                f"_apply_1q_kernel: index desync, r={r} is in inv[{q}] but plt[{r},{q}]=0"
            )
        new_xz = int(lut[old_xz])
        if new_xz == old_xz:
            continue
        old_x = (old_xz >> 1) & 1
        new_x = (new_xz >> 1) & 1
        plt[r, q] = new_xz
        if new_x and not old_x:
            _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, q, r)
        elif not new_x and old_x:
            _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, q, r)


# ----------------------------------------------------------------------
# Two-qubit gate kernel
# ----------------------------------------------------------------------


def _apply_2q_kernel(
    plt: np.ndarray,
    supp_q: np.ndarray,
    supp_len: np.ndarray,
    supp_pos: np.ndarray,
    inv: np.ndarray,
    inv_len: np.ndarray,
    inv_pos: np.ndarray,
    inv_x: np.ndarray,
    inv_x_len: np.ndarray,
    inv_x_pos: np.ndarray,
    qi: int,
    qj: int,
    lut: np.ndarray,
    active_buf: np.ndarray,
) -> None:
    """Apply a two-qubit Clifford gate at qubits ``(qi, qj)``.

    ``lut`` is the 16-entry uint8 LUT for the gate's 4x4 symplectic,
    built and cached by :func:`_build_2q_lut`. The kernel signature
    matches :func:`sparsegf2.core.numba_kernels.apply_2q_kernel_jit` so
    the cached LUT can be shared between the Python and JIT paths.

    Algorithm:

    1. Collect the deduplicated set of generators touching ``qi`` or ``qj`` via ``inv[qi]`` and ``inv[qj]``, deduplicated through a PLT lookup (a generator is already in ``inv[qi]`` iff ``plt[r, qi] != 0``).
    2. For each such generator, transform its 4 Pauli bits via ``lut`` and patch ``plt``, ``supp_q``, ``inv``, and ``inv_x`` on both qubits whenever a Pauli appears or disappears.

    Cost: ``O(|inv[qi]| + |inv[qj]|)`` per call. In an area-law circuit each generator is sparse, so this is much smaller than the ``O(N)`` cost a dense tableau pays per gate.

    ``active_buf`` is a caller-supplied int32 scratch buffer of length ``>= 2 * inv.shape[1]`` (re-used across gates so we don't allocate per call).
    """
    # Step 1: collect deduplicated active set into active_buf
    n_active = 0
    for idx in range(int(inv_len[qi])):
        active_buf[n_active] = int(inv[qi, idx])
        n_active += 1
    for idx in range(int(inv_len[qj])):
        r = int(inv[qj, idx])
        if int(plt[r, qi]) != 0:
            continue  # already in active_buf via the qi pass
        active_buf[n_active] = r
        n_active += 1

    # Step 2: per-generator transform. The LUT is indexed by the packed pair
    # (xz_i << 2) | xz_j and returns the new pair packed the same way, so there
    # is no per-generator bit unpack/repack; an unchanged generator is skipped.
    for a_idx in range(n_active):
        r = int(active_buf[a_idx])
        xz_i = int(plt[r, qi])
        xz_j = int(plt[r, qj])

        out = int(lut[(xz_i << 2) | xz_j])
        new_xz_i = (out >> 2) & 3
        new_xz_j = out & 3
        if new_xz_i == xz_i and new_xz_j == xz_j:
            continue  # this generator is unaffected by the gate

        old_xi = (xz_i >> 1) & 1
        new_xi = (new_xz_i >> 1) & 1
        old_xj = (xz_j >> 1) & 1
        new_xj = (new_xz_j >> 1) & 1

        # --- Update qubit qi ---
        was_present_i = xz_i != 0
        now_present_i = new_xz_i != 0
        if now_present_i and not was_present_i:
            _add_to_inv(inv, inv_len, inv_pos, qi, r)
            _add_to_support(supp_q, supp_len, supp_pos, r, qi)
        elif not now_present_i and was_present_i:
            _remove_from_inv(inv, inv_len, inv_pos, qi, r)
            _remove_from_support(supp_q, supp_len, supp_pos, r, qi)
        if new_xi and not old_xi:
            _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, qi, r)
        elif not new_xi and old_xi:
            _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, qi, r)
        plt[r, qi] = new_xz_i

        # --- Update qubit qj ---
        was_present_j = xz_j != 0
        now_present_j = new_xz_j != 0
        if now_present_j and not was_present_j:
            _add_to_inv(inv, inv_len, inv_pos, qj, r)
            _add_to_support(supp_q, supp_len, supp_pos, r, qj)
        elif not now_present_j and was_present_j:
            _remove_from_inv(inv, inv_len, inv_pos, qj, r)
            _remove_from_support(supp_q, supp_len, supp_pos, r, qj)
        if new_xj and not old_xj:
            _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, qj, r)
        elif not new_xj and old_xj:
            _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, qj, r)
        plt[r, qj] = new_xz_j


# ----------------------------------------------------------------------
# Sparse row XOR and generator clear (building blocks for measurement)
# ----------------------------------------------------------------------


def _sparse_xor_rows(
    plt: np.ndarray,
    supp_q: np.ndarray,
    supp_len: np.ndarray,
    supp_pos: np.ndarray,
    inv: np.ndarray,
    inv_len: np.ndarray,
    inv_pos: np.ndarray,
    inv_x: np.ndarray,
    inv_x_len: np.ndarray,
    inv_x_pos: np.ndarray,
    target: int,
    source: int,
) -> None:
    """XOR ``source`` into ``target`` (mod 2), maintaining all indices.

    Iterates only over ``supp_q[source]``, so the cost is ``O(wt(source))``
    rather than ``O(n)``. Every PLT bit that changes is reflected in
    ``inv``, ``inv_x``, and ``supp_q`` on the affected qubit; entries that
    appear or disappear are spliced in/out of the swap-with-last position
    maps in O(1).

    **Precondition: ``source != target``.** The function snapshots
    ``supp_len[source]`` once at entry and walks ``supp_q[source]``
    while mutating ``supp_q[target]`` via the swap-with-last helpers.
    If source and target are the same row, the snapshot becomes stale
    as the row is XOR'd into itself (which zeros every bit), so the
    iteration would read shuffled/cleared entries and the indices
    would silently desync. Callers are responsible for the constraint.
    """
    src_len = int(supp_len[source])
    # Walk a snapshot of the source's support; supp_len[source] is fixed
    # for the duration of this call (we only touch the target row).
    for idx in range(src_len):
        q = int(supp_q[source, idx])
        s_xz = int(plt[source, q])
        if s_xz == 0:
            continue
        s_x = (s_xz >> 1) & 1
        s_z = s_xz & 1

        t_xz = int(plt[target, q])
        old_t_x = (t_xz >> 1) & 1

        if t_xz != 0:
            new_x = old_t_x ^ s_x
            new_z = (t_xz & 1) ^ s_z
            new_xz = (new_x << 1) | new_z
            if new_xz != 0:
                plt[target, q] = new_xz
                if new_x and not old_t_x:
                    _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, q, target)
                elif not new_x and old_t_x:
                    _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, q, target)
            else:
                plt[target, q] = 0
                _remove_from_inv(inv, inv_len, inv_pos, q, target)
                if old_t_x:
                    _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, q, target)
                _remove_from_support(supp_q, supp_len, supp_pos, target, q)
        else:
            plt[target, q] = s_xz
            _add_to_inv(inv, inv_len, inv_pos, q, target)
            if s_x:
                _add_to_inv_x(inv_x, inv_x_len, inv_x_pos, q, target)
            _add_to_support(supp_q, supp_len, supp_pos, target, q)


def _clear_generator(
    plt: np.ndarray,
    supp_q: np.ndarray,
    supp_len: np.ndarray,
    supp_pos: np.ndarray,
    inv: np.ndarray,
    inv_len: np.ndarray,
    inv_pos: np.ndarray,
    inv_x: np.ndarray,
    inv_x_len: np.ndarray,
    inv_x_pos: np.ndarray,
    r: int,
) -> None:
    """Zero out generator row ``r`` and all index entries. ``O(wt(r))``."""
    L = int(supp_len[r])
    for idx in range(L):
        q = int(supp_q[r, idx])
        xz = int(plt[r, q])
        _remove_from_inv(inv, inv_len, inv_pos, q, r)
        if (xz >> 1) & 1:
            _remove_from_inv_x(inv_x, inv_x_len, inv_x_pos, q, r)
        plt[r, q] = 0
        supp_pos[r, q] = -1
    supp_len[r] = 0


# ----------------------------------------------------------------------
# Z-basis measurement kernel
# ----------------------------------------------------------------------


def _measure_z_kernel(
    plt: np.ndarray,
    supp_q: np.ndarray,
    supp_len: np.ndarray,
    supp_pos: np.ndarray,
    inv: np.ndarray,
    inv_len: np.ndarray,
    inv_pos: np.ndarray,
    inv_x: np.ndarray,
    inv_x_len: np.ndarray,
    inv_x_pos: np.ndarray,
    q: int,
    use_min_weight_pivot: bool,
    others_buf: np.ndarray,
) -> int:
    """Project the state onto a ``Z_q``-eigenspace (Aaronson-Gottesman CHP).

    Returns ``1`` when the measurement was *non-deterministic* and ``0``
    when it was deterministic. The caller draws a fair coin in the
    non-deterministic case (phase-free: we cannot derive the outcome).

    Determinism condition (A&G): ``Z_q`` measurement is deterministic
    iff **no stabilizer row anticommutes with Z_q** (i.e., no row index
    ``r >= n`` has the x-bit on ``q`` set). Destabilizers anticommuting
    with ``Z_q`` is irrelevant; that's their job in the canonical
    destab/stab pairing.

    Non-deterministic branch (stabilizer pivot ``p >= n`` exists with
    ``x_{p,q} = 1``):

    1. XOR ``p`` into every other row (destab or stab) with ``x_{·,q} = 1`` so that no other row anticommutes with ``Z_q`` post-call.
    2. Copy the *old* row ``p`` into row ``p - n`` (the destabilizer slot paired with stabilizer ``p``). This preserves the canonical destab/stab pairing.
    3. Replace row ``p`` with the pure ``Z_q`` generator.

    Deterministic branch: leave the symplectic tableau unchanged. This
    matches Stim byte-for-byte (Stim only flips a sign bit in this case).

    ``others_buf`` is a caller-supplied int32 scratch buffer of length
    ``>= inv.shape[1]``.
    """
    n = plt.shape[1]
    n_anti = int(inv_x_len[q])

    # Find a stabilizer anticommuter (row index >= n). If
    # use_min_weight_pivot, pick the lightest; otherwise pick the first.
    pivot = -1
    if use_min_weight_pivot:
        best_wt = -1
        for k in range(n_anti):
            r = int(inv_x[q, k])
            if r >= n:
                w = int(supp_len[r])
                if best_wt < 0 or w < best_wt:
                    best_wt = w
                    pivot = r
    else:
        for k in range(n_anti):
            r = int(inv_x[q, k])
            if r >= n:
                pivot = r
                break

    if pivot < 0:
        # Deterministic: no stabilizer anticommutes with Z_q. Stim's
        # symplectic is unchanged in this case, so we must be too.
        return 0

    # Snapshot the other anticommuters (inv_x[q] is mutated by XOR). Skip
    # the destabilizer slot at pivot - n: we are about to clear and
    # overwrite it with pivot's content, so XOR'ing pivot into it first
    # would be wasted work (the clear+copy below undoes the XOR exactly).
    destab = pivot - n
    n_others = 0
    for k in range(n_anti):
        r = int(inv_x[q, k])
        if r != pivot and r != destab:
            others_buf[n_others] = r
            n_others += 1

    # Step 1: XOR pivot into each other anticommuter.
    for k in range(n_others):
        _sparse_xor_rows(
            plt,
            supp_q,
            supp_len,
            supp_pos,
            inv,
            inv_len,
            inv_pos,
            inv_x,
            inv_x_len,
            inv_x_pos,
            int(others_buf[k]),
            pivot,
        )

    # Step 2: copy pivot into the destabilizer slot at pivot - n.
    # _sparse_xor_rows into a freshly-cleared row is a copy.
    _clear_generator(
        plt,
        supp_q,
        supp_len,
        supp_pos,
        inv,
        inv_len,
        inv_pos,
        inv_x,
        inv_x_len,
        inv_x_pos,
        destab,
    )
    _sparse_xor_rows(
        plt,
        supp_q,
        supp_len,
        supp_pos,
        inv,
        inv_len,
        inv_pos,
        inv_x,
        inv_x_len,
        inv_x_pos,
        destab,
        pivot,
    )

    # Step 3: replace pivot with the pure Z_q generator.
    _clear_generator(
        plt,
        supp_q,
        supp_len,
        supp_pos,
        inv,
        inv_len,
        inv_pos,
        inv_x,
        inv_x_len,
        inv_x_pos,
        pivot,
    )
    plt[pivot, q] = PAULI_Z
    supp_q[pivot, 0] = q
    supp_len[pivot] = 1
    supp_pos[pivot, q] = 0
    _add_to_inv(inv, inv_len, inv_pos, q, pivot)
    return 1


def _init_zero_state(
    plt: np.ndarray,
    supp_q: np.ndarray,
    supp_len: np.ndarray,
    supp_pos: np.ndarray,
    inv: np.ndarray,
    inv_len: np.ndarray,
    inv_pos: np.ndarray,
    inv_x: np.ndarray,
    inv_x_len: np.ndarray,
    inv_x_pos: np.ndarray,
    n: int,
) -> None:
    """Initialize the seven arrays to the standard |0^n⟩ A&G tableau.

    Layout:

    * Row ``i`` (destabilizer): ``X_i``, i.e. ``plt[i, i] = PAULI_X``.
    * Row ``n + i`` (stabilizer): ``Z_i``, i.e. ``plt[n+i, i] = PAULI_Z``.

    This writes the |0^n⟩ tableau (the all-zero computational basis
    state), *not* a Bell-pair tableau; the name says exactly what it does.
    """
    # The caller allocates everything zeroed (numpy default). Position
    # maps need -1 sentinels, which the caller also fills via np.full.
    # We only have to set the n destabilizers + n stabilizers.

    for i in range(n):
        # Destabilizer g_i = X_i
        plt[i, i] = PAULI_X
        supp_q[i, 0] = i
        supp_len[i] = 1
        supp_pos[i, i] = 0

        # X_i has x-bit set => appears in inv and in inv_x
        pos = inv_len[i]
        inv[i, pos] = i
        inv_pos[i, i] = pos
        inv_len[i] += 1

        pos_x = inv_x_len[i]
        inv_x[i, pos_x] = i
        inv_x_pos[i, i] = pos_x
        inv_x_len[i] += 1

        # Stabilizer g_{n+i} = Z_i
        plt[n + i, i] = PAULI_Z
        supp_q[n + i, 0] = i
        supp_len[n + i] = 1
        supp_pos[n + i, i] = 0

        # Z_i has only z-bit set => appears in inv but NOT inv_x
        pos = inv_len[i]
        inv[i, pos] = n + i
        inv_pos[n + i, i] = pos
        inv_len[i] += 1


# ----------------------------------------------------------------------
# Dense bit-packed kernels: pure-Python mirrors of the @njit versions in
# numba_kernels.py. Used when ``use_numba=False`` (the hybrid dense path) and
# exercised by the numba-vs-Python parity tests. Result-identical to the JIT
# kernels: same packed LUTs, same "first stabilizer anticommuter" pivot.
# ----------------------------------------------------------------------

_U1 = np.uint64(1)
_U2 = np.uint64(2)
_U3 = np.uint64(3)


def _apply_2q_packed(x_packed, z_packed, qi, qj, lut):
    """Pure-Python mirror of :func:`numba_kernels.apply_2q_packed_jit`."""
    wi, bi = qi >> 6, np.uint64(qi & 63)
    wj, bj = qj >> 6, np.uint64(qj & 63)
    xi = (x_packed[:, wi] >> bi) & _U1
    zi = (z_packed[:, wi] >> bi) & _U1
    xj = (x_packed[:, wj] >> bj) & _U1
    zj = (z_packed[:, wj] >> bj) & _U1
    idx = (((xi << _U1) | zi) << _U2) | ((xj << _U1) | zj)
    out = lut[idx.astype(np.intp)].astype(np.uint64)
    new_xz_i = (out >> _U2) & _U3
    new_xz_j = out & _U3
    keep_i = ~(_U1 << bi)
    keep_j = ~(_U1 << bj)
    x_packed[:, wi] = (x_packed[:, wi] & keep_i) | (((new_xz_i >> _U1) & _U1) << bi)
    z_packed[:, wi] = (z_packed[:, wi] & keep_i) | ((new_xz_i & _U1) << bi)
    x_packed[:, wj] = (x_packed[:, wj] & keep_j) | (((new_xz_j >> _U1) & _U1) << bj)
    z_packed[:, wj] = (z_packed[:, wj] & keep_j) | ((new_xz_j & _U1) << bj)


def _apply_1q_packed(x_packed, z_packed, q, lut):
    """Pure-Python mirror of :func:`numba_kernels.apply_1q_packed_jit`."""
    w, b = q >> 6, np.uint64(q & 63)
    x = (x_packed[:, w] >> b) & _U1
    z = (z_packed[:, w] >> b) & _U1
    out = lut[((x << _U1) | z).astype(np.intp)].astype(np.uint64)
    keep = ~(_U1 << b)
    x_packed[:, w] = (x_packed[:, w] & keep) | (((out >> _U1) & _U1) << b)
    z_packed[:, w] = (z_packed[:, w] & keep) | ((out & _U1) << b)


def _measure_z_packed(x_packed, z_packed, q):
    """Pure-Python mirror of :func:`numba_kernels.measure_z_packed_jit`."""
    n_rows = x_packed.shape[0]
    n = n_rows >> 1
    w, b = q >> 6, np.uint64(q & 63)
    xbit = (x_packed[:, w] >> b) & _U1
    pivot = -1
    for r in range(n, n_rows):
        if xbit[r]:
            pivot = r
            break
    if pivot == -1:
        return 0
    destab = pivot - n
    pivot_x = x_packed[pivot].copy()
    pivot_z = z_packed[pivot].copy()
    for r in np.nonzero(xbit)[0]:
        if r != pivot and r != destab:
            x_packed[r] ^= pivot_x
            z_packed[r] ^= pivot_z
    x_packed[destab] = pivot_x
    z_packed[destab] = pivot_z
    x_packed[pivot] = np.uint64(0)
    z_packed[pivot] = np.uint64(0)
    z_packed[pivot, w] = _U1 << b
    return 1


def _plt_to_packed(plt, x_packed, z_packed, n):
    """Pure-Python mirror of :func:`numba_kernels.plt_to_packed_jit`."""
    x_packed[:, :] = np.uint64(0)
    z_packed[:, :] = np.uint64(0)
    for r in range(plt.shape[0]):
        for q in range(n):
            xz = int(plt[r, q])
            if xz:
                w, b = q >> 6, np.uint64(q & 63)
                if (xz >> 1) & 1:
                    x_packed[r, w] |= _U1 << b
                if xz & 1:
                    z_packed[r, w] |= _U1 << b


def _packed_to_plt(x_packed, z_packed, plt, n):
    """Pure-Python mirror of :func:`numba_kernels.packed_to_plt_jit`."""
    plt[:, :] = np.uint8(0)
    for r in range(plt.shape[0]):
        for q in range(n):
            w, b = q >> 6, np.uint64(q & 63)
            xb = int((x_packed[r, w] >> b) & _U1)
            zb = int((z_packed[r, w] >> b) & _U1)
            if xb or zb:
                plt[r, q] = (xb << 1) | zb


def _row_weights_packed(x_packed, z_packed, n):
    """Pure-Python mirror of :func:`numba_kernels.row_weights_packed_jit`."""
    combined = (x_packed | z_packed).copy()
    tail = n & 63
    if tail:
        combined[:, -1] &= (_U1 << np.uint64(tail)) - _U1
    bits = np.unpackbits(combined.view(np.uint8), axis=1)
    return bits.sum(axis=1).astype(np.int32)


class SparseGF2:
    """Phase-free sparse stabilizer simulator over GF(2).

    See the module-level docstring for the full capability list (1q/2q
    Cliffords, Z/X/Y measurement + reset, canonical-form extraction,
    copy/restore, factory constructors).

    Parameters
    ----------
    n : int
        Number of qubits in the state. The tableau carries ``2 * n``
        generator rows: rows ``0..n-1`` are destabilizers, rows
        ``n..2n-1`` are stabilizers.

        ``SparseGF2(n)`` builds exactly ``n`` qubits; it does **not**
        silently double ``n``. To model an ``n``-qubit system entangled
        with an ``n``-qubit reference, construct the purification
        explicitly: build a ``SparseGF2(2 * n_system)`` and apply the
        Hadamard + CNOT pairs by hand (or use
        :func:`from_bell_purification`).

    Attributes
    ----------
    n : int
        Number of qubits.
    N : int
        Number of generator rows in the tableau, equal to ``2 * n``.
    plt : ndarray, shape (N, n), dtype uint8
        Pauli Lookup Table. ``plt[r, q]`` is the 2-bit Pauli of
        generator ``r`` on qubit ``q``.
    supp_q, supp_len, supp_pos : ndarray
        Per-generator support lists with O(1) removal.
    inv, inv_len, inv_pos : ndarray
        Per-qubit inverted index of all generators touching that qubit.
    inv_x, inv_x_len, inv_x_pos : ndarray
        Per-qubit inverted index of generators with x-bit set on that
        qubit (used by the Z-basis measurement kernel ``_measure_z_kernel``
        to find anticommuters in O(1)).
    """

    # Supported ``pivot_mode`` values. Adding modes is forward-compatible;
    # callers must not assume the set is closed.
    _PIVOT_MODES: tuple[str, ...] = ("min_weight", "first")

    # Hard upper bound on qubit count. ``supp_q``, ``inv``, and ``inv_x``
    # store generator indices as int32, so a 2n-row tableau caps at
    # n = 2^31 / 2 - 1. The realistic bound is memory long before this,
    # but the assertion documents the contract and catches a
    # hypothetical int32 → int8 dtype regression at __init__ time.
    _MAX_N: int = 1 << 30

    def __init__(
        self,
        n: int,
        *,
        rng: np.random.Generator | None = None,
        pivot_mode: str | None = None,
        use_numba: bool | None = None,
        hybrid: bool = False,
    ):
        """Construct an ``n``-qubit :math:`|0^n\\rangle` simulator.

        Parameters
        ----------
        n
            Number of qubits. Must be non-negative and below
            :attr:`SparseGF2._MAX_N` (the int32-index ceiling, far above
            any realistic memory bound).
        rng
            Default ``numpy.random.Generator`` used by measurement-method
            calls that don't pass an explicit ``rng=`` kwarg. If ``None``
            (the default), a fresh ``np.random.default_rng()`` is created
            and stored on the instance; this gives one seed per instance
            (still non-reproducible across constructions, but reproducible
            *within* a single instance's measurement sequence).
            Pass an explicit seeded ``rng`` for fully reproducible
            trajectories.
        pivot_mode
            How the measurement kernel chooses among multiple
            anticommuting stabilizers:

            * ``"min_weight"`` (default): pick the lightest anticommuter,
              which keeps the post-measurement tableau sparser.
            * ``"first"``: pick the first anticommuter, which is cheaper per
              measurement and generally produces denser stabilizers.

            Forward-compatible: future modes (e.g. ``"random"``) may be
            added; the set of valid modes is :attr:`SparseGF2._PIVOT_MODES`.
        use_numba
            When ``True``, gates and measurement dispatch to the JIT
            kernels in :mod:`sparsegf2.core.numba_kernels`; when ``False``,
            the pure-Python kernels in this module are used. ``None``
            (default) auto-detects: JIT if Numba imported successfully,
            pure-Python otherwise.
        hybrid
            When ``True``, the simulator monitors the tableau's stabilizer
            density and **switches to a bit-packed dense representation** while
            the state is volume-law (where the sparse inverted-index bookkeeping
            costs more than it saves), switching back to sparse when it thins.
            Gates and measurements then run via the dense kernels with no index
            maintenance, much faster on dense/low-measurement-rate circuits.
            Produces the identical physical state either way: the same stabilizer
            group and therefore every gauge-invariant observable (entropy, code
            dimension, mutual information, distance). The generator-weight
            *diagnostics* (``generator_weights`` / ``stabilizer_weight_spectrum`` /
            ``average_stabilizer_weight``) are basis-dependent and may differ:
            the dense CHP measurement pivots to a different (equally valid)
            generating set, exactly as ``pivot_mode`` already does. Which set you
            read also depends on whether the sim is dense at that moment, so do
            **not** aggregate those weight diagnostics across hybrid and
            pure-sparse runs. The dense path is also somewhat sticky: the dense
            first-anticommuter pivot keeps ``a_bar`` high, so a hybrid sim tends to
            stay dense once it has densified (the automatic switch back to sparse
            rarely fires on its own). ``False`` (default) keeps the pure-sparse
            engine; the dense mirror is only allocated when this is ``True``.
        """
        if n < 0:
            raise InvalidArgumentError(f"n must be non-negative, got {n}")
        if n > self._MAX_N:
            raise InvalidArgumentError(
                f"n={n} exceeds SparseGF2._MAX_N={self._MAX_N}; the int32 "
                "generator-index dtype would overflow"
            )

        if pivot_mode is None:
            pivot_mode = "min_weight"
        if pivot_mode not in self._PIVOT_MODES:
            raise InvalidArgumentError(f"pivot_mode={pivot_mode!r} not in {self._PIVOT_MODES}")

        if use_numba is None:
            use_numba = _HAS_NUMBA
        if use_numba and not _HAS_NUMBA:
            raise RuntimeError(
                "use_numba=True but sparsegf2.core.numba_kernels failed to import; "
                "numba is required for the JIT kernels (it is a core "
                "dependency); reinstall sparsegf2, or pass use_numba=False"
            )
        self.n: int = n
        self.N: int = 2 * n
        self.pivot_mode: str = pivot_mode
        # ``use_min_weight_pivot`` is kept as a derived bool so the JIT
        # kernel signature (which takes a bool, not a string) stays
        # unchanged. When new pivot modes land, they will branch off
        # this slot in the dispatch.
        self.use_min_weight_pivot: bool = pivot_mode == "min_weight"
        self.use_numba: bool = use_numba
        self.hybrid: bool = bool(hybrid)
        # --- Hybrid sparse/dense switching state (only acts when hybrid=True) ---
        # Switch to the dense bit-packed path when the average column weight
        # (a_bar) climbs above _dense_threshold, and back to sparse below half
        # that. The 2x band plus the coarse check interval give hysteresis so
        # transients near the crossover do not thrash. The decision is evaluated
        # only once every _check_interval ops, amortizing the O(n) a_bar scan.
        self._dense_threshold: int = max(n // 4, 16)
        self._check_interval: int = max(n // 2, 32)
        # Per-instance default RNG for measurement calls. Reproducibility
        # contract: within one SparseGF2 instance, measurements draw from
        # this RNG in deterministic call order. Pass an explicit
        # ``rng=...`` to override at the call site.
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

        self._allocate_buffers(n)

        _init_zero_state(
            self.plt,
            self.supp_q,
            self._supp_len,
            self.supp_pos,
            self.inv,
            self.inv_len,
            self.inv_pos,
            self.inv_x,
            self.inv_x_len,
            self.inv_x_pos,
            n,
        )

    def _allocate_buffers(self, n: int) -> None:
        """Allocate the four inverted-index arrays + scratch buffers.

        Extracted from ``__init__`` so a subclass (e.g. a future
        ``BatchedSparseGF2``) can override storage without copy-pasting
        the allocation block. Mutates ``self``; assumes ``self.n`` /
        ``self.N`` are already set.

        Subclass contract: any override must populate the same attribute
        names (``plt``, ``supp_q``, ``supp_len``, ``supp_pos``, ``inv``,
        ``inv_len``, ``inv_pos``, ``inv_x``, ``inv_x_len``, ``inv_x_pos``,
        ``_active_buf``, ``_others_buf``) with shapes that the kernel
        helpers in :mod:`sparsegf2.core.numba_kernels` accept.
        """
        ms = _max_support(n)
        mi = _max_inv(n)

        # Position maps hold values in [-1, mi). For mi <= 32767 the
        # tight type is int16, which halves the L2 working set of the
        # three large pos arrays, a free memory-bandwidth win at
        # n = O(100..1000) where the hot loops scatter-read these.
        # For larger n we fall back to int32.
        # The strict ``<`` is load-bearing: the maximum stored value is
        # ``mi - 1`` and we need both that and the ``-1`` sentinel to
        # fit in the chosen dtype. A future edit to ``<=`` would silently
        # overflow int16 at the boundary, so we assert it explicitly.
        pos_dtype = np.int16 if mi < (1 << 15) else np.int32
        assert mi - 1 <= np.iinfo(pos_dtype).max, (
            f"position dtype {pos_dtype} cannot hold mi-1={mi - 1}; bump to int32"
        )

        # Dense PLT
        self.plt = np.zeros((self.N, n), dtype=np.uint8)

        # Sparse support lists + O(1) position map. ``supp_len`` is the backing
        # array behind the ``supp_len`` property, which serves the dense-mode
        # weights transparently (see the property below).
        self.supp_q = np.zeros((self.N, ms), dtype=np.int32)
        self._supp_len = np.zeros(self.N, dtype=np.int32)
        self.supp_pos = np.full((self.N, n), -1, dtype=pos_dtype)

        # Inverted index of all touching generators per qubit
        self.inv = np.zeros((n, mi), dtype=np.int32)
        self.inv_len = np.zeros(n, dtype=np.int32)
        self.inv_pos = np.full((self.N, n), -1, dtype=pos_dtype)

        # Inverted index of x-supporting generators per qubit
        self.inv_x = np.zeros((n, mi), dtype=np.int32)
        self.inv_x_len = np.zeros(n, dtype=np.int32)
        self.inv_x_pos = np.full((self.N, n), -1, dtype=pos_dtype)

        # Scratch buffers re-used across kernel calls.
        # _active_buf: 2q gate dedup'd active set. Worst-case length is
        # |inv[qi]| + |inv[qj]| <= 2 * inv.shape[1] (= 4n).
        # _others_buf: anticommuter snapshot for the measurement kernel.
        # Fits within |inv_x[q]| <= mi.
        self._active_buf = np.zeros(2 * mi, dtype=np.int32)
        self._others_buf = np.zeros(mi, dtype=np.int32)

        # Hybrid dense mirror: two bit-packed uint64 arrays of shape
        # ``(2n, ceil(n/64))``, allocated only when hybrid is enabled (a
        # pure-sparse instance pays nothing). Re-allocated on every rebuild
        # (``_load_from_symplectic``), which also resets us to sparse mode.
        #
        # INVARIANT: while ``_dense_mode`` is True, the bit-packed arrays are
        # authoritative and the sparse structures (``plt``, ``supp_q``,
        # ``supp_pos``, ``inv*``) are STALE. Any method that reads them directly
        # must branch on ``_dense_mode`` (as ``to_symplectic``, ``canonical_form``,
        # ``_subsystem_xz_block``, ``is_deterministic_z``, and the ``supp_len``
        # property already do) or first sync via ``_switch_to_sparse``.
        # self.hybrid is always set before allocation (__init__ assigns it ahead
        # of _allocate_buffers; _load_from_symplectic runs on built instances).
        if self.hybrid:
            n_words = (n + 63) // 64
            self.x_packed = np.zeros((self.N, n_words), dtype=np.uint64)
            self.z_packed = np.zeros((self.N, n_words), dtype=np.uint64)
        else:
            self.x_packed = None
            self.z_packed = None
        self._dense_mode = False
        self._ops_since_check = 0

    @property
    def supp_len(self) -> np.ndarray:
        """Per-generator Pauli weight (the count of non-identity columns).

        Backed by the maintained sparse array in sparse mode (returns the live
        ``self._supp_len`` reference); in dense mode it is computed on demand from
        the packed arrays (popcount), so weight-reading observables stay correct
        without forcing a sparse rebuild. Note the dense-mode return is a *fresh*
        array each access, not a live view, so do not cache it across mutations.
        """
        if self._dense_mode:
            if self.use_numba:
                return _nk.row_weights_packed_jit(self.x_packed, self.z_packed, self.n)
            return _row_weights_packed(self.x_packed, self.z_packed, self.n)
        return self._supp_len

    # ------------------------------------------------------------------
    # Hybrid sparse <-> dense mode switching
    # ------------------------------------------------------------------

    def active_count(self) -> float:
        """The mean number of active generators per qubit (``a_bar``).

        ``a_bar = (1/n) * sum_r weight(g_r)``, the average **column** weight of
        the tableau: how many of the ``2n`` generators touch a typical qubit.
        It is the standard tableau-density diagnostic (low in the area-law /
        measurement-dominated regime, extensive in the volume-law regime) and
        the metric that drives the hybrid sparse/dense mode switch. Reads the
        dense-aware ``supp_len``, so it is correct in both engine modes.

        Note this is a property of the *generating set*, not the state: like
        the generator-weight diagnostics it can differ between equivalent
        tableaus (e.g. across ``pivot_mode`` or sparse/dense measurement
        pivots) while every physical observable is identical.
        """
        if self.n == 0:
            return 0.0
        return float(self.supp_len.sum()) / self.n

    # Backwards-compatible internal alias (the hybrid switch predates the
    # public accessor).
    _active_count = active_count

    def _maybe_check_mode_switch(self) -> None:
        """Hybrid hook called after each gate/measurement. Amortized: the O(n)
        density scan + any switch only runs once every ``_check_interval`` ops.

        Note the counter ticks per *primitive* op, so a composite measurement
        (``measure_x`` = H, measure_z, H) advances it by 3 and a switch can fall
        between those primitives, always correct (every kernel reads the same
        logical tableau), just more frequent than per-logical-op."""
        if not self.hybrid:
            return
        self._ops_since_check += 1
        if self._ops_since_check < self._check_interval:
            return
        self._ops_since_check = 0
        abar = self._active_count()
        if self._dense_mode:
            if abar < self._dense_threshold / 2:
                self._switch_to_sparse()
        elif abar > self._dense_threshold:
            self._switch_to_dense()

    def _switch_to_dense(self) -> None:
        """Mirror the sparse PLT into the bit-packed arrays and enter dense mode.
        The sparse indices are left stale until a later switch back rebuilds them."""
        if self.use_numba:
            _nk.plt_to_packed_jit(self.plt, self.x_packed, self.z_packed, self.n)
        else:
            _plt_to_packed(self.plt, self.x_packed, self.z_packed, self.n)
        self._dense_mode = True

    def _switch_to_sparse(self) -> None:
        """Rebuild the PLT + all sparse indices from the dense state and resume
        the sparse kernels. ``to_symplectic`` is dense-aware, so it reads the
        packed arrays; ``_load_from_symplectic`` then re-derives everything and
        clears ``_dense_mode`` (via ``_allocate_buffers``)."""
        self._load_from_symplectic(self.to_symplectic())

    # ------------------------------------------------------------------
    # Single-qubit Clifford gates
    # ------------------------------------------------------------------

    def _check_qubit(self, q: int, name: str = "q") -> None:
        if not (0 <= q < self.n):
            raise IndexError(
                f"{name}={q} out of range for {self.n}-qubit system (valid: 0..{self.n - 1})"
            )

    def apply_gate_1q(self, q: int, S2: np.ndarray) -> None:
        """Apply a single-qubit Clifford gate via a 2x2 GF(2) symplectic matrix.

        ``S2`` is interpreted under the row-vector convention
        ``(x', z') = (x, z) @ S2`` on every generator with support on
        qubit ``q``. The simulator is phase-free; gates that differ only
        by phase (e.g. ``S`` vs ``S†``) collapse to the same symplectic.

        ``S2`` is validated against the symplectic condition
        :math:`S_2\\,\\Omega_1\\,S_2^\\top \\equiv \\Omega_1 \\pmod 2` on every call;
        a non-symplectic matrix would silently corrupt the tableau into a
        non-Clifford state. The check is ~1µs and worth the safety net.

        Parameters
        ----------
        q
            Qubit index in ``[0, n)``.
        S2
            ``(2, 2)`` uint8 GF(2) symplectic matrix.

        Raises
        ------
        ValueError
            If ``S2`` is not a valid element of ``Sp(2, F_2)``.
        """
        self._check_qubit(q)
        # _build_1q_lut validates shape + symplecticity on cache miss and
        # returns the (read-only) LUT. Cache hits skip validation entirely.
        lut = _build_1q_lut(S2)
        if self._dense_mode:
            if self.use_numba:
                _nk.apply_1q_packed_jit(self.x_packed, self.z_packed, q, lut)
            else:
                _apply_1q_packed(self.x_packed, self.z_packed, q, lut)
        elif self.use_numba:
            _nk.apply_1q_kernel_jit(
                self.plt,
                self.inv,
                self.inv_len,
                self.inv_x,
                self.inv_x_len,
                self.inv_x_pos,
                q,
                lut,
            )
        else:
            _apply_1q_kernel(
                self.plt,
                self.inv,
                self.inv_len,
                self.inv_x,
                self.inv_x_len,
                self.inv_x_pos,
                q,
                lut,
            )
        if self.hybrid:  # skip the bound-method call entirely on the default path
            self._maybe_check_mode_switch()

    def apply_h(self, q: int) -> None:
        """Hadamard: ``X <-> Z`` on qubit ``q``."""
        self.apply_gate_1q(q, H_SYMP)

    def apply_s(self, q: int) -> None:
        """Phase gate ``S``: ``X -> Y``, ``Z -> Z`` on qubit ``q``.

        Phase-free, so identical to ``S†`` at the symplectic level.
        """
        self.apply_gate_1q(q, S_SYMP)

    def apply_sqrt_x(self, q: int) -> None:
        """``√X``: ``Z -> Y``, ``X -> X`` on qubit ``q``.

        Phase-free, so identical to ``√X†`` at the symplectic level.
        """
        self.apply_gate_1q(q, SQRT_X_SYMP)

    # ------------------------------------------------------------------
    # Two-qubit Clifford gates
    # ------------------------------------------------------------------

    def _check_two_qubits(self, qi: int, qj: int) -> None:
        if not (0 <= qi < self.n):
            raise IndexError(
                f"qi={qi} out of range for {self.n}-qubit system (valid: 0..{self.n - 1})"
            )
        if not (0 <= qj < self.n):
            raise IndexError(
                f"qj={qj} out of range for {self.n}-qubit system (valid: 0..{self.n - 1})"
            )
        if qi == qj:
            raise InvalidArgumentError(f"Two-qubit gate requires distinct qubits, got qi=qj={qi}")

    def apply_gate_2q(self, qi: int, qj: int, S: np.ndarray) -> None:
        """Apply a two-qubit Clifford gate via a 4x4 GF(2) symplectic matrix.

        ``S`` is interpreted under the row-vector convention
        ``(x', z') = (x, z) @ S`` on every generator with support on
        ``qi`` or ``qj``, where the 4-vector ``(x, z)`` packs the bits
        as ``(x_qi, x_qj, z_qi, z_qj)``.

        Pass any element of :mod:`sparsegf2.core.symplectic` (e.g.
        :func:`~sparsegf2.core.symplectic.random_symplectic_2q`) or one
        of the named constants :data:`CX_SYMP`, :data:`CZ_SYMP`,
        :data:`SWAP_SYMP`.

        ``S`` is validated against the symplectic condition
        :math:`S\\,\\Omega_2\\,S^\\top \\equiv \\Omega_2 \\pmod 2` on every call;
        a non-symplectic matrix would silently corrupt the tableau into a
        non-Clifford state. The check is ~1µs and worth the safety net.

        Parameters
        ----------
        qi, qj
            Distinct qubit indices in ``[0, n)``.
        S
            ``(4, 4)`` uint8 GF(2) symplectic matrix in the
            ``(X_qi, X_qj, Z_qi, Z_qj)`` basis.

        Raises
        ------
        ValueError
            If ``S`` is not a valid element of ``Sp(4, F_2)``.
        """
        # _build_2q_lut validates shape + symplecticity on cache miss and
        # returns the (read-only) LUT. Cache hits skip validation entirely.
        self._dispatch_2q(qi, qj, _build_2q_lut(S))

    def _dispatch_2q(self, qi: int, qj: int, lut: np.ndarray) -> None:
        """Apply a 2q gate from an **already-built** LUT (the internal fast path).

        ``apply_gate_2q`` builds the LUT from a symplectic on every call (a
        ``tobytes`` re-key + cache lookup). A caller that replays gates from a
        fixed table (the circuit runner) builds each LUT once and calls this
        directly, removing that per-gate overhead on the hottest loop.
        """
        self._check_two_qubits(qi, qj)
        if self._dense_mode:
            if self.use_numba:
                _nk.apply_2q_packed_jit(self.x_packed, self.z_packed, qi, qj, lut)
            else:
                _apply_2q_packed(self.x_packed, self.z_packed, qi, qj, lut)
        elif self.use_numba:
            _nk.apply_2q_kernel_jit(
                self.plt,
                self.supp_q,
                self._supp_len,
                self.supp_pos,
                self.inv,
                self.inv_len,
                self.inv_pos,
                self.inv_x,
                self.inv_x_len,
                self.inv_x_pos,
                qi,
                qj,
                lut,
                self._active_buf,
            )
        else:
            _apply_2q_kernel(
                self.plt,
                self.supp_q,
                self._supp_len,
                self.supp_pos,
                self.inv,
                self.inv_len,
                self.inv_pos,
                self.inv_x,
                self.inv_x_len,
                self.inv_x_pos,
                qi,
                qj,
                lut,
                self._active_buf,
            )
        if self.hybrid:  # skip the bound-method call entirely on the default path
            self._maybe_check_mode_switch()

    def apply_cx(
        self,
        control: int | None = None,
        target: int | None = None,
        *,
        qi: int | None = None,
        qj: int | None = None,
    ) -> None:
        """CNOT: ``X_c -> X_c X_t``, ``Z_t -> Z_c Z_t``.

        Argument names: ``control`` and ``target`` (positional, the
        natural CX nomenclature) **or** ``qi`` and ``qj`` (keyword-only,
        consistent with :meth:`apply_cz`, :meth:`apply_swap`,
        :meth:`apply_gate_2q`). Mixing the two raises ``TypeError``.

        Examples
        --------
        >>> sim.apply_cx(0, 1)                # positional (control=0, target=1)
        >>> sim.apply_cx(control=0, target=1) # keyword, CX nomenclature
        >>> sim.apply_cx(qi=0, qj=1)          # keyword, sibling-method nomenclature
        """
        if qi is not None or qj is not None:
            if control is not None or target is not None:
                raise TypeError("apply_cx: pass either (control, target) or (qi, qj), not both")
            if qi is None or qj is None:
                raise TypeError("apply_cx: qi and qj must both be supplied")
            control, target = qi, qj
        if control is None or target is None:
            raise TypeError("apply_cx requires control and target (or qi, qj) qubit indices")
        self.apply_gate_2q(control, target, CX_SYMP)

    def apply_cz(self, qi: int, qj: int) -> None:
        """CZ (symmetric): ``X_i -> X_i Z_j``, ``X_j -> Z_i X_j``."""
        self.apply_gate_2q(qi, qj, CZ_SYMP)

    def apply_swap(self, qi: int, qj: int) -> None:
        """SWAP: exchange qubits ``qi`` and ``qj``."""
        self.apply_gate_2q(qi, qj, SWAP_SYMP)

    def apply_clifford(self, symp: np.ndarray, qubits: Iterable[int] | None = None) -> None:
        r"""Apply an arbitrary multi-qubit Clifford, given by its symplectic.

        Transforms the tableau by the ``(2k, 2k)`` GF(2) symplectic ``symp``
        acting on the ``k`` qubits in ``qubits`` (default: all ``n`` qubits, in
        which case ``symp`` must be ``(2n, 2n)``). This generalizes
        :meth:`apply_gate_1q` / :meth:`apply_gate_2q` to any ``k``; ``symp`` is in
        the same ``[x | z]`` column convention as the named gate constants and
        :func:`sparsegf2.random_symplectic`, so e.g.
        ``apply_clifford(random_symplectic(k, rng), qubits)`` applies a uniform
        random ``k``-qubit Clifford (a scrambling unitary).

        A many-qubit Clifford densifies the tableau, so this does **not** exploit
        sparsity: it densifies via :meth:`to_symplectic`, updates the ``2k``
        affected columns (``O(n * k^2)``), then rebuilds the sparse indices with
        :meth:`_load_from_symplectic` (an ``O(n^2)`` pass over the tableau). Use it
        for one-shot work (a global scramble), not the inner gate loop. In hybrid
        mode, calling this while dense reads the packed state (``to_symplectic`` is
        dense-aware) and leaves the sim in sparse mode afterwards; the next switch
        check re-densifies if warranted.

        Parameters
        ----------
        symp : ndarray, shape ``(2k, 2k)``, uint8
            The Clifford's symplectic. Must satisfy the symplectic condition.
        qubits : iterable of int, optional
            The ``k`` distinct qubits it acts on; coordinate order matches
            ``symp`` (x-block then z-block). Defaults to ``range(n)``.

        Raises
        ------
        InvalidArgumentError
            If the shapes/qubit count are inconsistent, or qubits are out of
            range or duplicated.
        SymplecticConditionError
            If ``symp`` is not symplectic.
        """
        M = np.asarray(symp, dtype=np.uint8) & 1
        if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] % 2 != 0:
            raise InvalidArgumentError(f"symp must be (2k, 2k); got shape {M.shape}")
        k = M.shape[0] // 2
        q = (
            np.arange(self.n, dtype=np.int64)
            if qubits is None
            else np.asarray(list(qubits), dtype=np.int64)
        )
        if q.shape[0] != k:
            raise InvalidArgumentError(f"symp is (2k, 2k) with k={k} but got {q.shape[0]} qubits")
        if q.size and (q.min() < 0 or q.max() >= self.n):
            raise InvalidArgumentError(f"qubits must be in [0, n={self.n}); got {q.tolist()}")
        if np.unique(q).shape[0] != q.shape[0]:
            raise InvalidArgumentError(f"qubits must be distinct; got {q.tolist()}")
        if not _is_symplectic(M):
            raise SymplecticConditionError(
                "apply_clifford: symp is not symplectic (S @ Omega @ S.T != Omega mod 2)"
            )
        # Generators are rows of the symplectic; the codebase's gate convention is
        # row-vector (v -> v @ M), so the full transform is T @ M_full where M_full
        # is the identity except the (idx, idx) block = M. Only the idx columns
        # change, so update that column block in place: T[:, idx] = T[:, idx] @ M.
        n = self.n
        idx = np.concatenate([q, n + q])
        tableau = self.to_symplectic()
        tableau[:, idx] = (tableau[:, idx] @ M) % 2
        self._load_from_symplectic(tableau)

    # ------------------------------------------------------------------
    # Z-basis measurement
    # ------------------------------------------------------------------

    def is_deterministic_z(self, q: int) -> bool:
        """Return ``True`` iff measuring ``Z_q`` is deterministic.

        Aaronson-Gottesman: a ``Z_q`` measurement is deterministic iff
        no **stabilizer** (row index ``>= n``) anticommutes with
        ``Z_q``. Destabilizers anticommuting with ``Z_q`` is structural
        (that is the destabilizer's job in the canonical pairing) and
        does not make the measurement random.

        Agrees with Stim's phase-free predicate ``peek_z(q) != 0``.
        (Note: Stim's ``peek_z`` returns +1 / −1 / 0 and the sign
        distinguishes the two deterministic eigenstates; we collapse
        both into ``True`` because we don't track signs.)
        """
        self._check_qubit(q)
        n = self.n
        if self._dense_mode:
            # No inverted index in dense mode: scan the stabilizer rows' x-bit on q.
            w, b = q >> 6, np.uint64(q & 63)
            return not bool(((self.x_packed[n:, w] >> b) & np.uint64(1)).any())
        n_anti = int(self.inv_x_len[q])
        if n_anti == 0:
            return True
        # Vectorised: ask numpy whether any of the first n_anti entries
        # of inv_x[q] is a stabilizer row (index >= n). ~5-10× faster
        # than the Python generator + all().
        return not bool(np.any(self.inv_x[q, :n_anti] >= n))

    def measure_z(self, q: int, rng: np.random.Generator | None = None) -> int:
        """Measure qubit ``q`` in the Z basis and project the state.

        The simulator tracks only the symplectic part of the tableau,
        so the *value* of the outcome is sign-bit information we
        deliberately don't carry. For a deterministic
        measurement we return ``0``; for a random measurement we draw a
        fair coin from ``rng``.

        Either way, the post-measurement *symplectic* tableau is the
        unique projection of the pre-measurement tableau onto the
        ``Z_q``-eigenspace, which matches Stim's symplectic step-for-step.

        .. warning::

            **The return value is not the physical outcome.** A deterministic
            measurement of a state stabilized by ``-Z_q`` (e.g. ``|1⟩_q``)
            would physically yield ``1``, but this method returns ``0``
            regardless, because the sign that would distinguish ``|0⟩_q``
            from ``|1⟩_q`` lives in the phase bit we do not track. For
            *non-deterministic* measurements the return value *is* a fair
            phase-free coin: it has the right distribution but is not
            correlated with any physical outcome.

            For a downstream user this means:

            * Phase-free observables (entropy, code dimension, rank,
              correlations between qubits that are measured *together*) are fully accurate.
            * Single-qubit signed Pauli expectations
              :math:`\\langle\\psi|Z_q|\\psi\\rangle` are **not** available. Use Stim for those.

            Use :meth:`is_deterministic_z` to ask whether the measurement
            is deterministic *without* projecting.

        Parameters
        ----------
        q
            Qubit index in ``[0, n)``.
        rng
            ``numpy.random.Generator`` used to pick the random outcome
            when the measurement is non-deterministic. **Defaults to the
            per-instance `self._rng`** set at construction (so measurements
            are reproducible within one ``SparseGF2`` instance with no
            extra plumbing). Pass an explicit ``rng`` to override at the
            call site, useful when branching trajectories from a copied
            instance and you need each branch's RNG seeded separately.

        Returns
        -------
        int
            ``0`` or ``1``, the (phase-free) outcome of the measurement.
        """
        self._check_qubit(q)
        if self._dense_mode:
            if self.use_numba:
                was_random = _nk.measure_z_packed_jit(self.x_packed, self.z_packed, q)
            else:
                was_random = _measure_z_packed(self.x_packed, self.z_packed, q)
        elif self.use_numba:
            was_random = _nk.measure_z_kernel_jit(
                self.plt,
                self.supp_q,
                self._supp_len,
                self.supp_pos,
                self.inv,
                self.inv_len,
                self.inv_pos,
                self.inv_x,
                self.inv_x_len,
                self.inv_x_pos,
                q,
                self.use_min_weight_pivot,
                self._others_buf,
            )
        else:
            was_random = _measure_z_kernel(
                self.plt,
                self.supp_q,
                self._supp_len,
                self.supp_pos,
                self.inv,
                self.inv_len,
                self.inv_pos,
                self.inv_x,
                self.inv_x_len,
                self.inv_x_pos,
                q,
                self.use_min_weight_pivot,
                self._others_buf,
            )
        if self.hybrid:  # skip the bound-method call entirely on the default path
            self._maybe_check_mode_switch()
        if was_random:
            # Fall back to the per-instance RNG (set at construction) so
            # reproducibility is preserved within an instance.
            if rng is None:
                rng = self._rng
            return int(rng.integers(0, 2))
        return 0

    def reset_z(self, q: int, rng: np.random.Generator | None = None) -> None:
        """Project ``q`` onto the ``Z_q``-eigenspace.

        In the phase-free representation this is identical to a Z-basis
        measurement (the difference would live in the sign bit, which we
        don't track). The outcome bit is discarded.

        When ``rng=None`` (the default) the underlying :meth:`measure_z`
        falls back to ``self._rng`` (set at construction), so the
        sequence is reproducible within one ``SparseGF2`` instance. Pass
        an explicit ``rng`` to override at the call site.

        .. note::

            This is **not** a forced ``|0⟩_q`` projection. Phase-free,
            we cannot distinguish ``|0⟩_q`` from ``|1⟩_q``; they share
            the same symplectic. The state is left in *some* eigenstate
            of ``Z_q``; which one is a sign-bit fact we don't carry.
        """
        self.measure_z(q, rng=rng)

    def reset_x(self, q: int, rng: np.random.Generator | None = None) -> None:
        """Project ``q`` onto the ``X_q``-eigenspace. Phase-free version
        of :meth:`reset_z` in the X basis: ``H · reset_z · H``.

        RNG semantics match :meth:`reset_z`: falls back to ``self._rng``.
        """
        self.apply_h(q)
        self.measure_z(q, rng=rng)
        self.apply_h(q)

    def reset_y(self, q: int, rng: np.random.Generator | None = None) -> None:
        """Project ``q`` onto the ``Y_q``-eigenspace.

        Conjugation: ``S · H · reset_z · H · S`` (phase-free, so
        ``S = S†`` at the symplectic level). RNG semantics match
        :meth:`reset_z`: falls back to ``self._rng``.
        """
        self.apply_s(q)
        self.apply_h(q)
        self.measure_z(q, rng=rng)
        self.apply_h(q)
        self.apply_s(q)

    def measure_x(self, q: int, rng: np.random.Generator | None = None) -> int:
        """Measure qubit ``q`` in the X basis: ``H · M_Z · H``."""
        self.apply_h(q)
        out = self.measure_z(q, rng)
        self.apply_h(q)
        return out

    def measure_y(self, q: int, rng: np.random.Generator | None = None) -> int:
        """Measure qubit ``q`` in the Y basis.

        Conjugates the Y eigenbasis to the Z eigenbasis via ``S H`` (so
        the inverse sequence is ``H S`` after the measurement). The
        simulator is phase-free, so ``S†`` collapses to ``S`` at the
        symplectic level.
        """
        self.apply_s(q)
        self.apply_h(q)
        out = self.measure_z(q, rng)
        self.apply_h(q)
        self.apply_s(q)
        return out

    # ------------------------------------------------------------------
    # Symplectic extraction
    # ------------------------------------------------------------------

    def canonical_form(self) -> np.ndarray:
        """Return the canonical $\\mathbb{F}_2$ RREF of the stabilizer subspace.

        Two stabilizer tableaus represent the same phase-free state iff
        their stabilizer subspaces (the row span of rows ``n..2n-1`` in
        ``[X | Z]``) are equal. The reduced row-echelon form is a
        canonical representative of that subspace, so this method gives
        a **byte-identical** answer for any two tableaus describing the
        same state, regardless of generator-set or destabilizer choice.

        Returns
        -------
        ndarray
            An ``(n, 2n)`` ``uint8`` matrix in GF(2) RREF. Rows are the
            canonical stabilizer generators in the ``[X | Z]`` layout.

        Notes
        -----
        Destabilizers are *not* canonicalized; they are auxiliary data
        (coset representatives), and any valid choice represents the
        same physical state. The canonical comparison contract is
        therefore on the stabilizer block alone.
        """
        # Walk the stabilizer block (rows n..2n-1) of the PLT directly,
        # bypassing to_symplectic() to avoid materializing the destabs.
        # Then reduce via the centralized GF(2) RREF (JIT-accelerated;
        # ~10-50× faster than a Python row-reduction at moderate n).
        n = self.n
        if self._dense_mode:
            stab = self.to_symplectic()[n:].copy()
        else:
            stab = np.empty((n, 2 * n), dtype=np.uint8)
            stab[:, :n] = (self.plt[n:] >> 1) & 1
            stab[:, n:] = self.plt[n:] & 1
        return gf2_rref(stab)

    def copy(self, *, copy_rng: bool = True) -> Self:
        """Return a deep copy of this simulator (snapshot/restore primitive).

        Every internal numpy array is copied (including the packed dense mirror
        when ``hybrid`` is on); the resulting instance is independent of ``self``
        and safe to mutate. ``use_numba``, ``pivot_mode``, ``use_min_weight_pivot``,
        ``hybrid``, the dense-mode switching state, ``n``, and ``N`` are preserved.

        This is the load-bearing primitive for the circuits package's
        trajectory machinery: split a state into branches, evolve each
        independently, and tabulate observables.

        Parameters
        ----------
        copy_rng
            When ``True`` (default), deep-copy the per-instance
            ``_rng`` so the copy advances independently of the parent.
            When ``False``, the copy is left with ``_rng = None`` and
            the caller must assign a fresh generator before any
            measurement. This skips the ``copy.deepcopy`` cost (a
            few microseconds) and is the right call for circuits-style
            trajectory branching where every branch needs a fresh
            seeded generator anyway.

        Returns
        -------
        SparseGF2
            An independent copy of this simulator in the same state.
        """
        # Construct a bare instance and overwrite its arrays; this bypasses
        # _init_zero_state since we're going to overwrite everything.
        # Using object.__new__ avoids re-running __init__.
        cls = type(self)
        other = cls.__new__(cls)
        other.n = self.n
        other.N = self.N
        other.pivot_mode = self.pivot_mode
        other.use_min_weight_pivot = self.use_min_weight_pivot
        other.use_numba = self.use_numba
        # Hybrid mode + switching state.
        other.hybrid = self.hybrid
        other._dense_mode = self._dense_mode
        other._dense_threshold = self._dense_threshold
        other._check_interval = self._check_interval
        other._ops_since_check = self._ops_since_check
        # Deep-copy the RNG so the copy advances independently of the
        # parent. To branch trajectories with *fresh* randomness, pass
        # ``copy_rng=False`` and assign a new generator post-copy.
        other._rng = _copy.deepcopy(self._rng) if copy_rng else None  # type: ignore[assignment]
        other.plt = self.plt.copy()
        other.supp_q = self.supp_q.copy()
        other._supp_len = self._supp_len.copy()
        other.supp_pos = self.supp_pos.copy()
        other.inv = self.inv.copy()
        other.inv_len = self.inv_len.copy()
        other.inv_pos = self.inv_pos.copy()
        other.inv_x = self.inv_x.copy()
        other.inv_x_len = self.inv_x_len.copy()
        other.inv_x_pos = self.inv_x_pos.copy()
        other._active_buf = self._active_buf.copy()
        other._others_buf = self._others_buf.copy()
        other.x_packed = self.x_packed.copy() if self.x_packed is not None else None
        other.z_packed = self.z_packed.copy() if self.z_packed is not None else None
        return other

    def __copy__(self) -> Self:
        return self.copy()

    def __deepcopy__(self, memo) -> Self:
        return self.copy()

    # ------------------------------------------------------------------
    # Pickling (serialization)
    # ------------------------------------------------------------------

    # Bump this when the on-disk array layout / dtypes change in a
    # backwards-incompatible way. ``__setstate__`` checks the value to
    # refuse mismatched pickles up front.
    _STATE_VERSION: int = 2

    def __getstate__(self) -> dict:
        """Capture the simulator's full state for ``pickle`` / ``deepcopy``.

        Note: the state explicitly excludes module-level caches (LUT,
        ``_SP4_CACHE``); those are re-warmed on demand in the
        reconstituting process.
        """
        return {
            "_state_version": self._STATE_VERSION,
            "n": self.n,
            "N": self.N,
            "pivot_mode": self.pivot_mode,
            "use_min_weight_pivot": self.use_min_weight_pivot,
            "use_numba": self.use_numba,
            "hybrid": self.hybrid,
            "_dense_mode": self._dense_mode,
            "_dense_threshold": self._dense_threshold,
            "_check_interval": self._check_interval,
            "_ops_since_check": self._ops_since_check,
            "_rng": self._rng,
            "plt": self.plt,
            "supp_q": self.supp_q,
            "_supp_len": self._supp_len,
            "supp_pos": self.supp_pos,
            "inv": self.inv,
            "inv_len": self.inv_len,
            "inv_pos": self.inv_pos,
            "inv_x": self.inv_x,
            "inv_x_len": self.inv_x_len,
            "inv_x_pos": self.inv_x_pos,
            "_active_buf": self._active_buf,
            "_others_buf": self._others_buf,
            "x_packed": self.x_packed,
            "z_packed": self.z_packed,
        }

    def __setstate__(self, state: dict) -> None:
        """Restore from a ``__getstate__`` dict; rejects mismatched versions."""
        v = state.get("_state_version", 0)
        if v != self._STATE_VERSION:
            raise InvalidArgumentError(
                f"SparseGF2 pickle has _state_version={v}; this build "
                f"expects {self._STATE_VERSION}. Re-pickle on the current "
                "version or migrate the on-disk array layout."
            )
        for k, v in state.items():
            if k == "_state_version":
                continue
            setattr(self, k, v)

    # ------------------------------------------------------------------
    # Optional Protocol method: SubsystemFastExtractor
    # ------------------------------------------------------------------

    def _subsystem_xz_block(self, A: np.ndarray) -> np.ndarray:
        """Return the ``(n, 2|A|)`` stabilizer-restricted ``[X | Z]`` block.

        Fast path for :func:`sparsegf2.subsystem_rank`: slice the PLT
        directly (rows ``n..2n-1``, qubit columns ``A``), unpack the xz
        nibble into separate X and Z bit-planes, and return them
        concatenated. Skips the ``(2n, 2n)`` allocation that
        :meth:`to_symplectic` would otherwise make.

        Implements :class:`sparsegf2.core._protocol.SubsystemFastExtractor`.

        Parameters
        ----------
        A : ndarray of int64
            Validated, sorted, in-range subsystem indices.

        Returns
        -------
        ndarray of uint8
            Shape ``(n, 2 * len(A))``; left half is X-bits, right half
            is Z-bits on qubits ``A``.
        """
        n = self.n
        if self._dense_mode:
            # No PLT in dense mode: gather the requested columns from the
            # bit-packed stabilizer rows by fancy-indexing whole words and
            # shifting per-column (no Python loop, no full (2n,2n) unpack).
            A = np.asarray(A)
            words = (A >> 6).astype(np.intp)
            bits = (A & 63).astype(np.uint64)
            one = np.uint64(1)
            x_block = ((self.x_packed[n:][:, words] >> bits) & one).astype(np.uint8)
            z_block = ((self.z_packed[n:][:, words] >> bits) & one).astype(np.uint8)
            return np.concatenate([x_block, z_block], axis=1)
        stab_plt = self.plt[n:, A]
        x_block = (stab_plt >> 1) & 1
        z_block = stab_plt & 1
        return np.concatenate([x_block, z_block], axis=1)

    # ------------------------------------------------------------------
    # Inverse extraction: rebuild from a (2n, 2n) symplectic matrix
    # ------------------------------------------------------------------

    @classmethod
    def from_symplectic(
        cls,
        symp: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
        pivot_mode: str | None = None,
        use_numba: bool | None = None,
        hybrid: bool = False,
    ) -> Self:
        """Construct a ``SparseGF2`` from an explicit ``(2n, 2n)`` symplectic.

        Closes the round-trip with :meth:`to_symplectic`. The input must
        be a GF(2) symplectic matrix in the Aaronson-Gottesman
        ``[X | Z]`` layout: rows ``0..n-1`` are destabilizers, rows
        ``n..2n-1`` are stabilizers, ``[X | Z]`` columns split at ``n``.

        Parameters
        ----------
        symp : ndarray, shape ``(2n, 2n)``, uint8
            The full symplectic matrix. Must satisfy
            :math:`S\\,\\Omega\\,S^T \\equiv \\Omega \\pmod 2`.
        rng, pivot_mode, use_numba, hybrid
            Forwarded to :meth:`__init__`.

        Returns
        -------
        SparseGF2
            A simulator whose state equals the input tableau.

        Raises
        ------
        InvalidArgumentError
            If ``symp`` does not have shape ``(2n, 2n)`` or is not
            symplectic.
        """
        symp = np.asarray(symp, dtype=np.uint8)
        if symp.ndim != 2 or symp.shape[0] != symp.shape[1] or symp.shape[0] % 2 != 0:
            raise InvalidArgumentError(f"symp must be (2n, 2n), got {symp.shape}")
        twoN = symp.shape[0]
        n = twoN // 2
        if not _is_symplectic(symp):
            raise SymplecticConditionError(
                "from_symplectic: input is not symplectic (S @ Omega @ S.T != Omega mod 2)"
            )
        sim = cls(n, rng=rng, pivot_mode=pivot_mode, use_numba=use_numba, hybrid=hybrid)
        # Overwrite the |0^n⟩ tableau with the requested state. We use
        # the same allocator as __init__, then re-walk the PLT to
        # populate the inverted indices.
        sim._load_from_symplectic(symp)
        return sim

    def _load_from_symplectic(self, symp: np.ndarray) -> None:
        """Overwrite ``self`` with the state encoded by a ``(2n, 2n)`` symplectic.

        Internal helper for :meth:`from_symplectic`. Assumes the input
        has already been validated.
        """
        n = self.n
        # Reset the inverted indices.
        self._allocate_buffers(n)
        # Pack X/Z bits into the (x << 1 | z) PLT layout.
        x_block = symp[:, :n] & 1
        z_block = symp[:, n:] & 1
        self.plt[:] = (x_block << 1) | z_block
        # Rebuild supp_q / supp_len / supp_pos and inv / inv_x indices
        # by walking every nonzero PLT entry.
        for r in range(2 * n):
            for q in range(n):
                xz = int(self.plt[r, q])
                if xz == 0:
                    continue
                _add_to_support(self.supp_q, self._supp_len, self.supp_pos, r, q)
                _add_to_inv(self.inv, self.inv_len, self.inv_pos, q, r)
                if (xz >> 1) & 1:
                    _add_to_inv_x(self.inv_x, self.inv_x_len, self.inv_x_pos, q, r)

    # ------------------------------------------------------------------
    # Repr (single-line summary)
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        # Show hybrid (and the current dense/sparse mode) only when enabled, so
        # the common pure-sparse repr is unchanged.
        hybrid = f", hybrid=True, dense={self._dense_mode}" if self.hybrid else ""
        return (
            f"{type(self).__name__}(n={self.n}, "
            f"pivot_mode={self.pivot_mode!r}, "
            f"use_numba={self.use_numba}{hybrid})"
        )

    def to_symplectic(self) -> np.ndarray:
        """Return the full symplectic matrix ``[X | Z]`` as ``(2n, 2n)`` uint8.

        Column layout follows the canonical Aaronson-Gottesman convention::

            mat[r, q]       = x-bit of generator r on qubit q   for q ∈ [0, n)
            mat[r, n + q]   = z-bit of generator r on qubit q   for q ∈ [0, n)

        Row layout: rows ``0..n-1`` are destabilizers; rows ``n..2n-1``
        are stabilizers.

        For the initial |0^n⟩ state, this matrix is exactly the
        ``2n × 2n`` identity.
        """
        # Vectorised: split the (x << 1 | z) byte into its two bit planes.
        # ~100× faster than a Python double loop at n=512.
        n = self.n
        if self._dense_mode:
            # Unpack the bit-packed arrays into a fresh PLT (no mutation of
            # self.plt) via the verified packed->plt kernel, then split.
            plt = np.zeros((self.N, n), dtype=np.uint8)
            if self.use_numba:
                _nk.packed_to_plt_jit(self.x_packed, self.z_packed, plt, n)
            else:
                _packed_to_plt(self.x_packed, self.z_packed, plt, n)
        else:
            plt = self.plt
        mat = np.empty((self.N, 2 * n), dtype=np.uint8)
        mat[:, :n] = (plt >> 1) & 1
        mat[:, n:] = plt & 1
        return mat


# ----------------------------------------------------------------------
# Module-level factories
# ----------------------------------------------------------------------
#
# These are ergonomic constructors so the circuits package (and any other
# downstream) doesn't have to write the same H + CX boilerplate every
# time. Each factory returns a fully-constructed :class:`SparseGF2`
# instance; the kernels are exactly the same as for any direct gate
# application.


def from_zero_state(n: int, **kwargs) -> SparseGF2:
    """Return a :class:`SparseGF2` representing the pure ``|0^n⟩`` state.

    Identical to ``SparseGF2(n)`` but spelled out for clarity at call
    sites that pair this with :func:`from_bell_purification` etc.

    Parameters
    ----------
    n
        Number of qubits.
    **kwargs
        Forwarded to :class:`SparseGF2` (e.g. ``use_numba``, ``pivot_mode``).
    """
    return SparseGF2(n, **kwargs)


def from_bell_purification(n_system: int, **kwargs) -> SparseGF2:
    """Build a ``2 * n_system``-qubit state in the Bell-pair purification picture.

    For each system qubit ``i ∈ [0, n_system)``, a Bell pair
    ``|Φ⁺⟩ = (|00⟩ + |11⟩) / √2`` is prepared between qubit ``i`` (the
    "system" side) and qubit ``i + n_system`` (the "reference" side).
    Equivalent to applying ``H(i); CX(i, i + n_system)`` for each
    ``i ∈ [0, n_system)`` to a ``SparseGF2(2 * n_system)`` initialized
    in ``|0^{2 * n_system}⟩``.

    This builds the purification explicitly, so the entangled
    system+reference state is constructed by hand rather than implied by a
    qubit-count convention.

    Parameters
    ----------
    n_system
        Number of system qubits. The returned simulator has
        ``2 * n_system`` total qubits.
    **kwargs
        Forwarded to :class:`SparseGF2`.

    Returns
    -------
    SparseGF2
        A ``2 * n_system``-qubit simulator in the Bell-pair state.
    """
    sim = SparseGF2(2 * n_system, **kwargs)
    for i in range(n_system):
        sim.apply_h(i)
        sim.apply_cx(i, i + n_system)
    return sim


# ``from_product_state`` was removed: it silently discarded its ``bits``
# argument (returned the same tableau for every input), which is correct
# under the phase-free contract but a footgun for downstream code that
# wanted classical-bit tracking. The same effect is spelled directly:
# ``SparseGF2(len(bits))``.
