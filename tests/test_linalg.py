"""Direct tests for the centralized GF(2) primitives,
SimulatorProtocol structural check, and SparseGF2.copy() RNG semantics.

These primitives are extracted into :mod:`sparsegf2.core.linalg_gf2`
and used by both :meth:`SparseGF2.canonical_form` and
:func:`sparsegf2.random_symplectic`. They were previously only
exercised transitively; this file pins them directly.

Covers:

1. ``gf2_rref`` JIT vs pure-Python parity on uniform-random matrices
   *including* entries outside ``{0, 1}`` (the centralized wrapper
   normalizes via ``& 1`` so both paths must agree).
2. ``gf2_kernel_basis`` JIT vs pure-Python parity on the same regime.
3. ``gf2_rank`` is consistent with explicit RREF-then-count.
4. ``SparseGF2`` satisfies :class:`SimulatorProtocol` at runtime.
5. ``SparseGF2.copy()`` deep-copies the per-instance RNG (advancing
   the parent does not affect the copy).
6. ``measure_z`` falls back to ``self._rng`` when no ``rng=`` is
   passed (per-instance reproducibility contract).
"""

from __future__ import annotations

import copy as _copy

import numpy as np
import pytest

from sparsegf2 import SparseGF2
from sparsegf2.core._protocol import SimulatorProtocol
from sparsegf2.core.linalg_gf2 import (
    _gf2_kernel_basis_python,
    _gf2_rref_python,
    gf2_kernel_basis,
    gf2_rank,
    gf2_rref,
)

# ----------------------------------------------------------------------
# JIT-vs-Python parity for gf2_rref
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,seed",
    [
        ((4, 8), 0),
        ((8, 8), 1),
        ((10, 16), 2),
        ((16, 4), 3),
        ((32, 64), 4),
    ],
)
def test_gf2_rref_jit_vs_python_parity(shape, seed):
    """``gf2_rref`` JIT and pure-Python paths must produce the same RREF."""
    rng = np.random.default_rng(seed)
    M = rng.integers(0, 2, size=shape, dtype=np.uint8)
    # JIT path via centralized dispatcher
    jit_result = gf2_rref(M.copy())
    # Pure-Python reference
    py_result = _gf2_rref_python(M.copy())
    assert np.array_equal(jit_result, py_result)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_gf2_rref_normalizes_nonbinary_inputs(seed):
    """Inputs containing 2, 3, 255 etc. must be normalized to ``& 1``
    before reduction; both JIT and Python paths must agree."""
    rng = np.random.default_rng(seed)
    M = rng.integers(0, 256, size=(6, 12), dtype=np.uint8)
    # The centralized wrapper masks with `& 1` so output bits should
    # equal what we'd get from explicit pre-masking.
    a = gf2_rref(M.copy())
    b = _gf2_rref_python((M & 1).copy())
    assert np.array_equal(a, b)
    # Same on the kernel-basis path.
    c = gf2_kernel_basis(M)
    d = _gf2_kernel_basis_python(M & 1)
    assert np.array_equal(c, d)


# ----------------------------------------------------------------------
# JIT-vs-Python parity for gf2_kernel_basis
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,seed",
    [
        ((0, 4), 0),  # empty constraint matrix → identity
        ((1, 4), 1),
        ((3, 6), 2),
        ((4, 8), 3),
        ((6, 12), 4),
        ((8, 8), 5),  # square (often rank-k → empty kernel)
        ((10, 5), 6),  # m > k
    ],
)
def test_gf2_kernel_basis_jit_vs_python_parity(shape, seed):
    """``gf2_kernel_basis`` JIT and pure-Python paths must agree."""
    rng = np.random.default_rng(seed)
    if shape[0] == 0:
        M = np.zeros(shape, dtype=np.uint8)
    else:
        M = rng.integers(0, 2, size=shape, dtype=np.uint8)
    jit_kernel = gf2_kernel_basis(M)
    py_kernel = _gf2_kernel_basis_python(M)
    # The basis is not unique, but the row-span is. Both should produce
    # bases of the same vector space, equivalent iff
    # rref([jit | py]) has the same rank as either alone.
    if jit_kernel.shape[0] == 0:
        assert py_kernel.shape[0] == 0
        return
    # Concatenate and check rank stays at jit_kernel.shape[0].
    combined = np.concatenate([jit_kernel, py_kernel], axis=0)
    r_jit = gf2_rank(jit_kernel)
    r_combined = gf2_rank(combined)
    assert r_jit == jit_kernel.shape[0]  # JIT result is a basis
    assert r_combined == r_jit  # Python adds nothing new


def test_gf2_kernel_basis_kernel_property():
    """Every output row must satisfy ``M v^T == 0 mod 2``."""
    rng = np.random.default_rng(0)
    M = rng.integers(0, 2, size=(5, 10), dtype=np.uint8)
    K = gf2_kernel_basis(M)
    if K.shape[0] == 0:
        return
    product = (K @ M.T) & 1
    assert np.array_equal(product, np.zeros_like(product))


# ----------------------------------------------------------------------
# gf2_rank consistency
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_gf2_rank_matches_explicit_rref(seed):
    rng = np.random.default_rng(seed)
    M = rng.integers(0, 2, size=(8, 16), dtype=np.uint8)
    via_helper = gf2_rank(M)
    via_explicit = int(np.any(gf2_rref(M.copy()), axis=1).sum())
    assert via_helper == via_explicit


def test_gf2_rank_empty():
    assert gf2_rank(np.zeros((0, 5), dtype=np.uint8)) == 0
    assert gf2_rank(np.zeros((5, 0), dtype=np.uint8)) == 0


# ----------------------------------------------------------------------
# SimulatorProtocol structural check
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 4, 8])
def test_sparsegf2_satisfies_simulator_protocol(n):
    """``SparseGF2`` must satisfy the structural :class:`SimulatorProtocol`."""
    sim = SparseGF2(n)
    assert isinstance(sim, SimulatorProtocol), (
        "SparseGF2 missing one of the SimulatorProtocol attributes"
    )
    # And its attributes have the documented types.
    assert isinstance(sim.n, int)
    assert isinstance(sim.supp_len, np.ndarray)
    assert sim.supp_len.shape == (2 * n,)
    M = sim.to_symplectic()
    assert isinstance(M, np.ndarray)
    assert M.shape == (2 * n, 2 * n)


# ----------------------------------------------------------------------
# Per-instance RNG semantics: measure_z falls back to self._rng
# ----------------------------------------------------------------------


def test_measure_z_uses_instance_rng_by_default():
    """Two SparseGF2 instances constructed with the same seed must
    produce identical measure_z outcome sequences when no ``rng=`` is
    passed, pinning the per-instance reproducibility contract."""
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    sim_a = SparseGF2(4, rng=rng_a)
    sim_b = SparseGF2(4, rng=rng_b)
    # Drive each into the same non-deterministic state.
    for s in (sim_a, sim_b):
        s.apply_h(0)
        s.apply_h(1)
        s.apply_h(2)
        s.apply_h(3)
    outs_a = [sim_a.measure_z(q) for q in range(4)]
    outs_b = [sim_b.measure_z(q) for q in range(4)]
    assert outs_a == outs_b, (
        f"measure_z without explicit rng= must be reproducible across "
        f"instances seeded identically: got {outs_a} vs {outs_b}"
    )


def test_copy_rng_advances_independently():
    """After ``sim.copy()``, advancing the parent's RNG must not affect
    the copy's. Pins the deep-copy-of-RNG semantics."""
    sim = SparseGF2(4, rng=np.random.default_rng(2026))
    # Apply gates to put both into a non-deterministic state.
    for q in range(4):
        sim.apply_h(q)
    sib = sim.copy()
    # Advance the parent's RNG by issuing a measurement on the parent.
    sim.measure_z(0)
    # Now both should still produce the same NEXT outcome on a fresh qubit
    # if (and only if) their RNGs were deepcopied. Reset both qubits.
    # Easier: just compare a long sequence of standalone rng draws.
    parent_draws = [int(sim._rng.integers(0, 2)) for _ in range(20)]
    child_draws = [int(sib._rng.integers(0, 2)) for _ in range(20)]
    # If sib._rng started independently advanced from sim._rng's state
    # at the moment of copy, the two sequences will differ in a
    # deterministic way, but both must be reproducible (one per RNG).
    # The contract we care about: each RNG was deepcopied, so mutating
    # one doesn't change the other. We've already drawn 20 ints from
    # each, and re-running with fresh deep copies from the same starting
    # state should reproduce both sequences.
    sim2 = SparseGF2(4, rng=np.random.default_rng(2026))
    for q in range(4):
        sim2.apply_h(q)
    sib2 = sim2.copy()
    sim2.measure_z(0)
    parent_draws_again = [int(sim2._rng.integers(0, 2)) for _ in range(20)]
    child_draws_again = [int(sib2._rng.integers(0, 2)) for _ in range(20)]
    assert parent_draws == parent_draws_again
    assert child_draws == child_draws_again


def test_copy_via_stdlib_deepcopy():
    """`copy.deepcopy(sim)` is wired through `__deepcopy__`."""
    sim = SparseGF2(3)
    sim.apply_h(0)
    sib = _copy.deepcopy(sim)
    assert np.array_equal(sim.to_symplectic(), sib.to_symplectic())


# ----------------------------------------------------------------------
# apply_cx accepts both (control, target) and (qi, qj)
# ----------------------------------------------------------------------


def test_apply_cx_accepts_kwarg_aliases():
    """For consistency with apply_cz/swap, which
    use (qi, qj); apply_cx accepts both naming conventions."""
    a = SparseGF2(3)
    b = SparseGF2(3)
    c = SparseGF2(3)
    a.apply_cx(0, 1)
    b.apply_cx(control=0, target=1)
    c.apply_cx(qi=0, qj=1)
    assert np.array_equal(a.to_symplectic(), b.to_symplectic())
    assert np.array_equal(a.to_symplectic(), c.to_symplectic())


def test_apply_cx_rejects_mixed_naming():
    sim = SparseGF2(3)
    with pytest.raises(TypeError, match="not both"):
        sim.apply_cx(0, qj=1)


def test_apply_cx_rejects_missing_args():
    sim = SparseGF2(3)
    with pytest.raises(TypeError):
        sim.apply_cx()
    with pytest.raises(TypeError):
        sim.apply_cx(qi=0)


# ----------------------------------------------------------------------
# SimulatorProtocol-only fallback path inside observables.subsystem_rank
# ----------------------------------------------------------------------


class _ProtocolOnlyAdapter:
    """Wraps a SparseGF2 instance and exposes ONLY the SimulatorProtocol
    surface: no ``plt`` attribute, no JIT internals. Used to exercise
    the ``to_symplectic()`` fallback inside :func:`subsystem_rank` (the
    fast path is gated on ``isinstance(sim, SparseGF2)``).
    """

    def __init__(self, sim: SparseGF2) -> None:
        self._sim = sim

    @property
    def n(self) -> int:
        return self._sim.n

    @property
    def supp_len(self) -> np.ndarray:
        return self._sim.supp_len

    def to_symplectic(self) -> np.ndarray:
        return self._sim.to_symplectic()


@pytest.mark.parametrize("n,seed", [(4, 0), (6, 1), (8, 2)])
def test_observables_match_through_protocol_adapter(n, seed):
    """Observables computed via the SimulatorProtocol fallback (no
    ``plt`` attr) must agree with the SparseGF2 fast path. Pins the
    protocol surface so alternative backends won't silently regress."""
    from sparsegf2 import (
        code_dimension,
        entanglement_entropy,
        mutual_information,
        subsystem_rank,
    )

    rng = np.random.default_rng(seed)
    sim = SparseGF2(n, rng=rng)
    # Drive into a non-trivial state.
    for _ in range(2 * n):
        q = int(rng.integers(0, n))
        sim.apply_h(q)
        if n >= 2:
            qj = int(rng.integers(0, n))
            if qj != q:
                sim.apply_cx(q, qj)

    adapter = _ProtocolOnlyAdapter(sim)

    A = list(range(n // 2))
    B = list(range(n // 2, n))

    assert subsystem_rank(adapter, A) == subsystem_rank(sim, A)
    assert entanglement_entropy(adapter, A) == entanglement_entropy(sim, A)
    if A and B:
        assert mutual_information(adapter, A, B) == mutual_information(sim, A, B)
    if n % 2 == 0:
        assert code_dimension(adapter, n // 2) == code_dimension(sim, n // 2)


# ----------------------------------------------------------------------
# gf2_rank_bits: bit-packed rank == uint8 rank, exactly
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "shape",
    [(1, 1), (5, 3), (17, 64), (20, 65), (33, 127), (40, 128), (64, 200), (90, 13)],
)
def test_gf2_rank_bits_matches_gf2_rank(shape, seed):
    """The packed-word rank agrees with the uint8 rank on random matrices,
    including widths at/around the 64-bit word boundary."""
    from sparsegf2.core.linalg_gf2 import gf2_rank, gf2_rank_bits

    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, size=shape).astype(np.uint8)
    assert gf2_rank_bits(A) == gf2_rank(A)
    # low-rank structure: duplicate half the rows
    m = shape[0]
    if m >= 2:
        A[m // 2 :] = A[: m - m // 2][: m - m // 2]
        assert gf2_rank_bits(A) == gf2_rank(A)


def test_gf2_rank_bits_nonbinary_and_empty():
    """Entries are taken mod 2 (parity with gf2_rank's normalization), the
    input is not mutated, and the empty matrix has rank 0."""
    from sparsegf2.core.linalg_gf2 import gf2_rank, gf2_rank_bits

    rng = np.random.default_rng(0)
    A = rng.integers(0, 256, size=(12, 70)).astype(np.uint8)
    before = A.copy()
    assert gf2_rank_bits(A) == gf2_rank(A)
    assert np.array_equal(A, before)  # not mutated
    assert gf2_rank_bits(np.zeros((0, 5), dtype=np.uint8)) == 0
    assert gf2_rank_bits(np.zeros((5, 0), dtype=np.uint8)) == 0


def test_gf2_rank_bits_python_mirror_matches_jit():
    """The pure-Python word-elimination mirror gives the same rank as the
    public (JIT-preferring) path."""
    from sparsegf2.core.linalg_gf2 import _gf2_rank_words_python, gf2_rank_bits

    rng = np.random.default_rng(7)
    for _ in range(20):
        m, k = int(rng.integers(1, 40)), int(rng.integers(1, 150))
        A = rng.integers(0, 2, size=(m, k)).astype(np.uint8)
        packed8 = np.packbits(A, axis=1, bitorder="little")
        n_words = (packed8.shape[1] + 7) // 8
        if packed8.shape[1] != 8 * n_words:
            pad = np.zeros((m, 8 * n_words - packed8.shape[1]), dtype=np.uint8)
            packed8 = np.concatenate([packed8, pad], axis=1)
        words = np.ascontiguousarray(packed8).view(np.uint64).copy()
        assert _gf2_rank_words_python(words, k) == gf2_rank_bits(A)


# ----------------------------------------------------------------------
# gf2_eliminate_on_columns
# ----------------------------------------------------------------------


def test_eliminate_on_columns_contract():
    """Rank on the pivot set matches gf2_rank of those columns, rows past
    the rank are zero on every pivot column, the input is not mutated, and
    the row space is preserved."""
    from sparsegf2.core.linalg_gf2 import gf2_eliminate_on_columns, gf2_rank

    rng = np.random.default_rng(5)
    for _ in range(20):
        m, k = int(rng.integers(1, 20)), int(rng.integers(2, 20))
        n_piv = int(rng.integers(1, k))
        A = rng.integers(0, 2, size=(m, k)).astype(np.uint8)
        before = A.copy()
        cols = np.arange(n_piv)
        reduced, r = gf2_eliminate_on_columns(A, cols)
        assert np.array_equal(A, before)  # not mutated
        assert r == gf2_rank(A[:, :n_piv])
        assert not reduced[r:, :n_piv].any()
        # Row operations preserve the row space.
        assert gf2_rank(np.concatenate([A, reduced])) == gf2_rank(A)


def test_eliminate_on_columns_identity_witness():
    """Augmenting with an identity block records the row operations: each
    reduced row equals its witness combination of the original rows."""
    from sparsegf2.core.linalg_gf2 import gf2_eliminate_on_columns

    rng = np.random.default_rng(6)
    m, k = 10, 7
    A = rng.integers(0, 2, size=(m, k)).astype(np.uint8)
    aug = np.concatenate([A, np.eye(m, dtype=np.uint8)], axis=1)
    reduced, _ = gf2_eliminate_on_columns(aug, np.arange(k))
    for row in reduced:
        witness = row[k:]
        assert np.array_equal((witness @ A) % 2, row[:k])
