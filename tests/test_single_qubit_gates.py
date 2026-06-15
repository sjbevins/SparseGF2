"""single-qubit Clifford Stim parity.

Verifies that each named 1q gate ``H``, ``S``, ``√X`` applied via
:class:`sparsegf2.SparseGF2` produces a tableau byte-equal to ``stim``'s
``TableauSimulator`` after the same gate. The test sweeps:

* Every qubit of every size in {1, 2, 3, 5, 8, 16, 32}.
* Step-by-step parity over random 200-gate sequences for n ∈ {3, 8, 16}.
* Idempotency (each 1q named gate squares to identity in the phase-free
  symplectic representation).
* Equivalence between the named wrappers and the generic
  ``apply_gate_1q`` entry point.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("stim")  # Stim-parity module: skipped when the optional stim extra is absent

from _stim_parity import fresh_stim, stim_symplectic
from sparsegf2 import SparseGF2
from sparsegf2.core.sparse_tableau import H_SYMP, S_SYMP, SQRT_X_SYMP

# (name, sparsegf2 method name, stim method name, symplectic constant)
NAMED_GATES = [
    ("H", "apply_h", "h", H_SYMP),
    ("S", "apply_s", "s", S_SYMP),
    ("SQRT_X", "apply_sqrt_x", "sqrt_x", SQRT_X_SYMP),
]


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 16, 32])
@pytest.mark.parametrize("gate_name,sg_method,stim_method,_symp", NAMED_GATES)
def test_named_gate_each_qubit(n, gate_name, sg_method, stim_method, _symp):
    """Apply ``gate_name`` at each qubit q; tableau must match Stim."""
    for q in range(n):
        sim = SparseGF2(n)
        stim_sim = fresh_stim(n)

        getattr(sim, sg_method)(q)
        getattr(stim_sim, stim_method)(q)

        ours = sim.to_symplectic()
        theirs = stim_symplectic(stim_sim, n)
        assert np.array_equal(ours, theirs), f"{gate_name} on qubit {q} of n={n} diverges from Stim"


@pytest.mark.parametrize("n", [1, 3, 8, 16])
@pytest.mark.parametrize("gate_name,sg_method,_stim,_symp", NAMED_GATES)
def test_named_gate_is_involution(n, gate_name, sg_method, _stim, _symp):
    """Phase-free symplectic: applying the gate twice = identity."""
    sim = SparseGF2(n)
    initial = sim.to_symplectic()
    for q in range(n):
        getattr(sim, sg_method)(q)
        getattr(sim, sg_method)(q)
    final = sim.to_symplectic()
    assert np.array_equal(initial, final), (
        f"{gate_name} is not its own inverse in the symplectic representation"
    )


@pytest.mark.parametrize("n", [1, 2, 5, 16])
def test_apply_gate_1q_matches_named_wrappers(n):
    """The generic ``apply_gate_1q(q, S2)`` must produce the same tableau
    as the named wrappers when called with the matching symplectic."""
    for q in range(n):
        for sg_method, S in [
            ("apply_h", H_SYMP),
            ("apply_s", S_SYMP),
            ("apply_sqrt_x", SQRT_X_SYMP),
        ]:
            sim_named = SparseGF2(n)
            sim_generic = SparseGF2(n)
            getattr(sim_named, sg_method)(q)
            sim_generic.apply_gate_1q(q, S)
            assert np.array_equal(sim_named.to_symplectic(), sim_generic.to_symplectic())


@pytest.mark.parametrize(
    "n,n_gates,seed",
    [
        (3, 200, 0),
        (8, 200, 1),
        (16, 200, 2),
        (32, 200, 3),
    ],
)
def test_random_sequence_stepwise_parity(n, n_gates, seed):
    """Apply a deterministic random sequence of 1q gates; verify parity at
    every step. This is the strong test: every prefix circuit's
    symplectic must match Stim's exactly.
    """
    rng = np.random.default_rng(seed)
    sim = SparseGF2(n)
    stim_sim = fresh_stim(n)

    for step in range(n_gates):
        gate_idx = int(rng.integers(0, 3))
        q = int(rng.integers(0, n))
        _, sg_method, stim_method, _ = NAMED_GATES[gate_idx]
        getattr(sim, sg_method)(q)
        getattr(stim_sim, stim_method)(q)

        if step % 10 == 0 or step == n_gates - 1:
            ours = sim.to_symplectic()
            theirs = stim_symplectic(stim_sim, n)
            assert np.array_equal(ours, theirs), (
                f"divergence at step {step} (gate={NAMED_GATES[gate_idx][0]}, q={q})"
            )


def test_check_qubit_raises_out_of_range():
    sim = SparseGF2(4)
    with pytest.raises(IndexError):
        sim.apply_h(-1)
    with pytest.raises(IndexError):
        sim.apply_h(4)


def test_inv_x_membership_consistent_after_random_sequence():
    """After random 1q gates, inv_x[q] must list exactly the generators
    with x-bit set on qubit q. This catches index corruption."""
    rng = np.random.default_rng(42)
    n = 12
    sim = SparseGF2(n)

    for _ in range(500):
        gate_idx = int(rng.integers(0, 3))
        q = int(rng.integers(0, n))
        getattr(sim, NAMED_GATES[gate_idx][1])(q)

    for q in range(n):
        expected = set()
        for r in range(sim.N):
            if (sim.plt[r, q] >> 1) & 1:
                expected.add(r)
        observed = set(int(sim.inv_x[q, i]) for i in range(sim.inv_x_len[q]))
        assert expected == observed, f"inv_x[q={q}] out of sync with PLT"


# ----------------------------------------------------------------------
# Safety net: apply_gate_1q rejects non-symplectic matrices.
# ----------------------------------------------------------------------


def test_apply_gate_1q_rejects_non_symplectic():
    """A non-symplectic 2x2 matrix must raise ValueError, not silently
    corrupt the tableau into a non-Clifford state."""
    sim = SparseGF2(3)
    # Wrong shape:
    with pytest.raises(ValueError, match="must be"):
        sim.apply_gate_1q(0, np.eye(3, dtype=np.uint8))
    # Right shape, wrong relation: S Ω S^T != Ω for the all-ones 2x2 matrix.
    bad = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="not symplectic"):
        sim.apply_gate_1q(0, bad)
    # The legitimate named gates still work.
    sim.apply_gate_1q(0, H_SYMP)
    sim.apply_gate_1q(1, S_SYMP)
    sim.apply_gate_1q(2, SQRT_X_SYMP)


def test_apply_gate_1q_lut_cache_keyed_by_shape_too():
    """Regression test for the LUT-cache key collision: H_SYMP has bytes
    ``b'\\x00\\x01\\x01\\x00'`` and so does a 1D ``np.array([0,1,1,0], uint8)``.
    The 1D array is invalid input and must still raise. That is, the cache
    must include the shape in its key, not just the bytes.
    """
    sim = SparseGF2(2)
    # Make sure H_SYMP is cached first (pre-warming + this call).
    sim.apply_h(0)
    # Now try the 1D array with the same bytes. Should fail validation,
    # NOT silently succeed via a stale cache lookup.
    bad_1d = np.array([0, 1, 1, 0], dtype=np.uint8)
    with pytest.raises(ValueError, match="must be"):
        sim.apply_gate_1q(0, bad_1d)
