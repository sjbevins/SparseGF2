"""Initial-state Stim parity.

:class:`sparsegf2.SparseGF2` must produce, on construction, the
canonical ``2n × 2n`` identity symplectic for the pure |0^n⟩ state on
exactly *n* qubits, bit-for-bit equal to what
``stim.TableauSimulator(n)`` reports for the same state.

This pins the no-implicit-purification rule (``SparseGF2(n)`` is *n*
qubits, never silently doubled) and the Stim parity contract at the very
first construction step.
"""

from __future__ import annotations

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from _stim_parity import fresh_stim, stim_symplectic
from sparsegf2 import SparseGF2
from sparsegf2.core.sparse_tableau import PAULI_X, PAULI_Z

# ----------------------------------------------------------------------
# Data-structure sanity (independent of Stim)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 8, 13, 21, 32])
def test_shapes(n):
    sim = SparseGF2(n)
    assert sim.n == n
    assert 2 * n == sim.N
    assert sim.plt.shape == (2 * n, n)
    # supp_q is sized to _max_support(n) = max(1, n).
    expected_ms = max(1, n)
    assert sim.supp_q.shape == (2 * n, expected_ms)
    assert sim.supp_len.shape == (2 * n,)
    assert sim.supp_pos.shape == (2 * n, n)
    # inv / inv_x are sized to _max_inv(n) = max(1, 2n), the tight upper
    # bound. (Was 2n+64 with an undocumented slack; see ROADMAP.md §5.5.)
    expected_mi = max(1, 2 * n)
    assert sim.inv.shape == (n, expected_mi)
    assert sim.inv_len.shape == (n,)
    assert sim.inv_pos.shape == (2 * n, n)
    assert sim.inv_x.shape == (n, expected_mi)
    assert sim.inv_x_len.shape == (n,)
    assert sim.inv_x_pos.shape == (2 * n, n)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
def test_plt_zero_state_layout(n):
    """Row i = X_i (destab), row n+i = Z_i (stab); all other PLT entries 0."""
    sim = SparseGF2(n)
    expected = np.zeros((2 * n, n), dtype=np.uint8)
    for i in range(n):
        expected[i, i] = PAULI_X
        expected[n + i, i] = PAULI_Z
    np.testing.assert_array_equal(sim.plt, expected)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
def test_support_lists_zero_state(n):
    """Every generator has support on exactly one qubit."""
    sim = SparseGF2(n)
    np.testing.assert_array_equal(sim.supp_len, np.ones(2 * n, dtype=np.int32))
    for r in range(2 * n):
        q = r % n  # destab i and stab i both live on qubit i
        assert sim.supp_q[r, 0] == q
        assert sim.supp_pos[r, q] == 0


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
def test_inverted_indices_zero_state(n):
    """Each qubit's inv list has exactly its destab + stab; inv_x has only the destab."""
    sim = SparseGF2(n)
    np.testing.assert_array_equal(sim.inv_len, np.full(n, 2, dtype=np.int32))
    np.testing.assert_array_equal(sim.inv_x_len, np.ones(n, dtype=np.int32))
    for q in range(n):
        # inv: destab on this qubit added first, then stab
        assert sim.inv[q, 0] == q
        assert sim.inv[q, 1] == n + q
        assert sim.inv_pos[q, q] == 0
        assert sim.inv_pos[n + q, q] == 1
        # inv_x: only the destab (X_q) has x-bit set
        assert sim.inv_x[q, 0] == q
        assert sim.inv_x_pos[q, q] == 0
        assert sim.inv_x_pos[n + q, q] == -1


# ----------------------------------------------------------------------
# Symplectic extraction
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 8, 13, 21, 32])
def test_symplectic_is_identity(n):
    """The |0^n⟩ symplectic must be the 2n × 2n identity."""
    sim = SparseGF2(n)
    mat = sim.to_symplectic()
    np.testing.assert_array_equal(mat, np.eye(2 * n, dtype=np.uint8))


# ----------------------------------------------------------------------
# Stim parity (the actual parity contract)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "n",
    [1, 2, 3, 4, 5, 7, 8, 13, 16, 21, 32, 50, 64, 100],
)
def test_stim_parity_initial_state(n):
    """SparseGF2(n).to_symplectic() must exactly equal Stim's |0^n⟩ tableau."""
    ours = SparseGF2(n).to_symplectic()
    theirs = stim_symplectic(fresh_stim(n), n)
    assert ours.shape == theirs.shape == (2 * n, 2 * n)
    np.testing.assert_array_equal(ours, theirs)


# ----------------------------------------------------------------------
# Snapshot / restore (SparseGF2.copy)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 3, 8])
def test_copy_produces_independent_instance(n):
    """`sim.copy()` returns a separate instance with identical state."""
    sim = SparseGF2(n)
    # Drive some non-trivial state if n >= 2
    if n >= 2:
        sim.apply_h(0)
        sim.apply_cx(0, 1)
    sib = sim.copy()
    assert sib is not sim
    np.testing.assert_array_equal(sim.to_symplectic(), sib.to_symplectic())
    # Mutate sib; sim must be untouched.
    if n >= 1:
        sib.apply_h(0)
        if n >= 1:
            assert not np.array_equal(sim.to_symplectic(), sib.to_symplectic())


def test_copy_preserves_construction_flags():
    sim = SparseGF2(4, pivot_mode="first", use_numba=True)
    sib = sim.copy()
    assert sib.n == sim.n
    assert sib.N == sim.N
    assert sib.pivot_mode == sim.pivot_mode
    assert sib.use_min_weight_pivot == sim.use_min_weight_pivot
    assert sib.use_numba == sim.use_numba


def test_copy_is_deepcopy_safe():
    """`copy.deepcopy(sim)` routes through `__deepcopy__` and works."""
    import copy

    sim = SparseGF2(5)
    sim.apply_h(0)
    sim.apply_cx(0, 1)
    sim.apply_s(2)
    sib = copy.deepcopy(sim)
    np.testing.assert_array_equal(sim.to_symplectic(), sib.to_symplectic())
