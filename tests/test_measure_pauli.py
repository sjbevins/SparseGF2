"""General Pauli measurement: ``measure_pauli`` and ``pauli_anticommuting_rows``.

Covers:

1. ``pauli_anticommuting_rows`` agrees with brute-force symplectic
   products computed from ``to_symplectic()``, in sparse and dense mode.
2. ``measure_pauli([q], [PAULI_Z])`` matches ``measure_z(q)`` at the
   stabilizer-subspace level (pivot tie-breaking may differ, so
   byte-for-byte comparison would be over-strict), and likewise for
   single-qubit X and Y against the conjugation-based methods.
3. Multi-qubit measurements preserve the symplectic (canonical pairing)
   condition and the four-structure index consistency, install ``g`` at
   the pivot row, and leave no stabilizer anticommuting with ``g``.
4. Re-measuring the same Pauli is deterministic: returns ``None`` and
   leaves the tableau unchanged.
5. Dense (hybrid) mode returns the same pivot pair and the same
   stabilizer subspace as sparse mode.
6. Stim parity via ``MPP`` on random circuits (skips without stim).
7. Input validation: identity, repeated qubits, bad letters, range and
   length errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2 import PAULI_X, PAULI_Y, PAULI_Z, SparseGF2
from sparsegf2.core.symplectic import is_symplectic, random_symplectic_2q
from sparsegf2.errors import InvalidArgumentError

_LETTER_NAMES = {1: "Z", 2: "X", 3: "Y"}

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _scrambled(n: int, seed: int, **kwargs) -> tuple[SparseGF2, np.random.Generator]:
    """A well-scrambled n-qubit state and the generator that built it."""
    rng = np.random.default_rng(seed)
    sim = SparseGF2(n, rng=np.random.default_rng(seed + 1), **kwargs)
    for _ in range(4 * n):
        qi, qj = rng.choice(n, size=2, replace=False).tolist()
        sim.apply_gate_2q(int(qi), int(qj), random_symplectic_2q(rng))
    return sim, rng


def _random_pauli(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    w = int(rng.integers(1, n + 1))
    qubits = np.sort(rng.choice(n, size=w, replace=False)).astype(np.int64)
    letters = rng.integers(1, 4, size=w).astype(np.uint8)
    return qubits, letters


def _brute_anticommuting(sim: SparseGF2, qubits, letters) -> set[int]:
    """Anticommuting rows via dense symplectic products (independent path)."""
    n = sim.n
    g = np.zeros(2 * n, dtype=np.uint8)
    for q, xz in zip(qubits, letters, strict=True):
        g[int(q)] = (int(xz) >> 1) & 1
        g[n + int(q)] = int(xz) & 1
    mat = sim.to_symplectic()
    out = set()
    for r in range(2 * n):
        prod = int(mat[r, :n] @ g[n:] + mat[r, n:] @ g[:n]) & 1
        if prod:
            out.add(r)
    return out


def _check_indices_consistent(sim: SparseGF2) -> None:
    n, N = sim.n, sim.N
    for q in range(n):
        expected_inv = {int(r) for r in range(N) if int(sim.plt[r, q]) != 0}
        actual_inv = {int(sim.inv[q, k]) for k in range(int(sim.inv_len[q]))}
        assert expected_inv == actual_inv, f"inv[{q}] inconsistent"
        expected_invx = {int(r) for r in range(N) if (int(sim.plt[r, q]) >> 1) & 1}
        actual_invx = {int(sim.inv_x[q, k]) for k in range(int(sim.inv_x_len[q]))}
        assert expected_invx == actual_invx, f"inv_x[{q}] inconsistent"
    for r in range(N):
        expected_supp = {int(q) for q in range(n) if int(sim.plt[r, q]) != 0}
        actual_supp = {int(sim.supp_q[r, k]) for k in range(int(sim.supp_len[r]))}
        assert expected_supp == actual_supp, f"supp_q[{r}] inconsistent"


# ----------------------------------------------------------------------
# 1. Anticommuting-row query vs brute force
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n, seed", [(4, 0), (8, 1), (12, 2)])
def test_anticommuting_rows_brute_force(n, seed):
    sim, rng = _scrambled(n, seed)
    for _ in range(20):
        qubits, letters = _random_pauli(n, rng)
        got = set(int(r) for r in sim.pauli_anticommuting_rows(qubits, letters))
        assert got == _brute_anticommuting(sim, qubits, letters)


def test_anticommuting_rows_dense_mode_matches_sparse():
    sim, rng = _scrambled(8, 5, hybrid=True)
    twin = sim.copy()
    sim._switch_to_dense()
    for _ in range(20):
        qubits, letters = _random_pauli(8, rng)
        got_dense = sim.pauli_anticommuting_rows(qubits, letters)
        got_sparse = twin.pauli_anticommuting_rows(qubits, letters)
        assert np.array_equal(got_dense, got_sparse)


def test_anticommuting_rows_identity_is_empty():
    sim, _ = _scrambled(6, 7)
    assert sim.pauli_anticommuting_rows([], []).shape == (0,)


# ----------------------------------------------------------------------
# 2. Single-letter agreement with the dedicated methods
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_measure_pauli_z_matches_measure_z(seed):
    sim, rng = _scrambled(8, seed)
    twin = sim.copy()
    q = int(rng.integers(0, 8))
    sim.measure_z(q)
    pair = twin.measure_pauli([q], [PAULI_Z])
    assert np.array_equal(sim.canonical_form(), twin.canonical_form())
    if pair is not None:
        # The installed pivot row is exactly Z_q.
        assert int(twin.plt[twin.n + pair, q]) == int(PAULI_Z)
        assert int(twin.supp_len[twin.n + pair]) == 1


@pytest.mark.parametrize("letter, method", [(PAULI_X, "measure_x"), (PAULI_Y, "measure_y")])
def test_measure_pauli_xy_match_conjugated_methods(letter, method):
    for seed in range(4):
        sim, rng = _scrambled(8, 40 + seed)
        twin = sim.copy()
        q = int(rng.integers(0, 8))
        getattr(sim, method)(q)
        twin.measure_pauli([q], [letter])
        assert np.array_equal(sim.canonical_form(), twin.canonical_form())


# ----------------------------------------------------------------------
# 3. Multi-qubit measurements: invariants and installation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n, seed", [(6, 10), (10, 11), (16, 12)])
def test_measure_pauli_preserves_invariants(n, seed):
    sim, rng = _scrambled(n, seed)
    for _ in range(3 * n):
        qubits, letters = _random_pauli(n, rng)
        pair = sim.measure_pauli(qubits, letters)
        assert is_symplectic(sim.to_symplectic())
        _check_indices_consistent(sim)
        # Post-measurement, nothing in the stabilizer half anticommutes.
        anti = sim.pauli_anticommuting_rows(qubits, letters)
        assert not np.any(anti >= n)
        if pair is not None:
            # g sits verbatim at the pivot row.
            row = sim.n + pair
            assert int(sim.supp_len[row]) == qubits.shape[0]
            for q, xz in zip(qubits, letters, strict=True):
                assert int(sim.plt[row, int(q)]) == int(xz)


def test_remeasuring_is_deterministic_noop():
    sim, rng = _scrambled(8, 21)
    qubits, letters = _random_pauli(8, rng)
    first = sim.measure_pauli(qubits, letters)
    assert first is not None  # overwhelmingly likely on a scrambled state
    before = sim.to_symplectic().copy()
    assert sim.measure_pauli(qubits, letters) is None
    assert np.array_equal(sim.to_symplectic(), before)


def test_zero_state_z_products_deterministic():
    sim = SparseGF2(4)
    before = sim.to_symplectic().copy()
    assert sim.measure_pauli([0, 1], [PAULI_Z, PAULI_Z]) is None
    assert sim.measure_pauli([0, 1, 2, 3], [PAULI_Z] * 4) is None
    assert np.array_equal(sim.to_symplectic(), before)
    # X_0 anticommutes with stabilizer Z_0 at pair 0.
    assert sim.measure_pauli([0], [PAULI_X]) == 0


# ----------------------------------------------------------------------
# 4. Dense (hybrid) mode parity
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_dense_mode_matches_sparse(seed):
    sim, rng = _scrambled(10, 60 + seed, hybrid=True)
    twin = sim.copy()
    sim._switch_to_dense()
    for _ in range(10):
        qubits, letters = _random_pauli(10, rng)
        p_dense = sim.measure_pauli(qubits, letters)
        p_sparse = twin.measure_pauli(qubits, letters)
        assert p_dense == p_sparse
        assert np.array_equal(sim.canonical_form(), twin.canonical_form())


# ----------------------------------------------------------------------
# 5. Stim parity via MPP
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n, seed", [(4, 100), (8, 101), (12, 102)])
def test_measure_pauli_stim_mpp_parity(n, seed):
    stim = pytest.importorskip("stim")
    from _stim_parity import assert_states_equal, fresh_stim, stim_symplectic
    from _stim_parity import symp_to_stim_tableau as to_stim

    rng = np.random.default_rng(seed)
    ours = SparseGF2(n)
    theirs = fresh_stim(n)
    for _ in range(3 * n):
        qi, qj = rng.choice(n, size=2, replace=False).tolist()
        S = random_symplectic_2q(rng)
        ours.apply_gate_2q(int(qi), int(qj), S)
        theirs.do_tableau(to_stim(S), [int(qi), int(qj)])
    for _ in range(2 * n):
        qubits, letters = _random_pauli(n, rng)
        ours.measure_pauli(qubits, letters)
        targets = "*".join(
            f"{_LETTER_NAMES[int(xz)]}{int(q)}" for q, xz in zip(qubits, letters, strict=True)
        )
        theirs.do(stim.Circuit(f"MPP {targets}"))
        assert_states_equal(ours.to_symplectic(), stim_symplectic(theirs, n), n)


# ----------------------------------------------------------------------
# 6. Validation
# ----------------------------------------------------------------------


def test_measure_pauli_rejects_bad_inputs():
    sim = SparseGF2(4)
    with pytest.raises(InvalidArgumentError):
        sim.measure_pauli([], [])  # identity
    with pytest.raises(InvalidArgumentError):
        sim.measure_pauli([0, 0], [PAULI_X, PAULI_X])  # repeated qubit
    with pytest.raises(InvalidArgumentError):
        sim.measure_pauli([0], [0])  # identity letter
    with pytest.raises(InvalidArgumentError):
        sim.measure_pauli([0], [4])  # out-of-range letter
    with pytest.raises(InvalidArgumentError):
        sim.measure_pauli([4], [PAULI_X])  # qubit out of range
    with pytest.raises(InvalidArgumentError):
        sim.measure_pauli([-1], [PAULI_X])
    with pytest.raises(InvalidArgumentError):
        sim.measure_pauli([0, 1], [PAULI_X])  # length mismatch


@pytest.mark.parametrize(
    ("qubits", "letters"),
    [
        ([1.9], [PAULI_X]),
        ([True], [PAULI_X]),
        ([0], [2.9]),
        ([0], [True]),
    ],
)
def test_pauli_validation_rejects_lossy_or_boolean_integer_aliases(qubits, letters):
    sim = SparseGF2(2)
    with pytest.raises(InvalidArgumentError, match="exact integers"):
        sim.pauli_anticommuting_rows(qubits, letters)
