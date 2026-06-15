"""SparseGF2.apply_clifford: apply an arbitrary multi-qubit Clifford by its
symplectic. Validated against the trusted 1-/2-qubit gate methods, by round-trip
inversion, and by the physical invariant that a system-only Clifford cannot
change the entropy of a disjoint reference."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2 import (
    CX_SYMP,
    CZ_SYMP,
    SWAP_SYMP,
    SparseGF2,
    entanglement_entropy,
    random_symplectic,
)
from sparsegf2.core.symplectic import _gf2_inverse
from sparsegf2.errors import InvalidArgumentError, SymplecticConditionError


def _nontrivial(n, seed):
    s = SparseGF2(n, rng=np.random.default_rng(seed))
    for q in range(n):
        s.apply_h(q)
    for q in range(n - 1):
        s.apply_cx(q, q + 1)
    s.apply_s(0)
    return s


@pytest.mark.parametrize("M,name", [(CX_SYMP, "cx"), (CZ_SYMP, "cz"), (SWAP_SYMP, "swap")])
def test_matches_named_two_qubit_gate(M, name):
    qi, qj = 1, 3
    a = _nontrivial(5, 0)
    b = _nontrivial(5, 0)
    a.apply_clifford(M, [qi, qj])
    getattr(b, f"apply_{name}")(qi, qj)
    assert np.array_equal(a.to_symplectic(), b.to_symplectic())


def test_global_default_acts_on_all_qubits():
    n = 6
    s = _nontrivial(n, 1)
    before = s.to_symplectic().copy()
    s.apply_clifford(random_symplectic(n, np.random.default_rng(2)))  # qubits=None -> all n
    assert not np.array_equal(s.to_symplectic(), before)


def test_round_trip_inverse_is_identity():
    n = 8
    s = _nontrivial(n, 3)
    before = s.to_symplectic().copy()
    M = random_symplectic(n, np.random.default_rng(4))
    s.apply_clifford(M)
    s.apply_clifford(_gf2_inverse(M))
    assert np.array_equal(s.to_symplectic(), before)


def test_system_scramble_preserves_reference_entropy():
    # qubit n is a reference Bell-paired with system qubit n-1; scrambling the
    # system (qubits 0..n-1) is a unitary on the system, so S(reference) is invariant.
    nsys = 8
    sim = SparseGF2(nsys + 1, rng=np.random.default_rng(5))
    sim.apply_h(nsys - 1)
    sim.apply_cx(nsys - 1, nsys)
    assert entanglement_entropy(sim, [nsys]) == 1
    sim.apply_clifford(random_symplectic(nsys, np.random.default_rng(6)), qubits=range(nsys))
    assert entanglement_entropy(sim, [nsys]) == 1


def test_global_clifford_scrambles_to_volume_law():
    # a random global Clifford on |0^n> gives a near-maximal half-cut entropy
    n = 12
    vals = []
    for seed in range(60):
        s = SparseGF2(n, rng=np.random.default_rng(seed))
        s.apply_clifford(random_symplectic(n, np.random.default_rng(seed)))
        vals.append(entanglement_entropy(s, range(n // 2)))
    # half-cut max is n/2 = 6; a random stabilizer state sits just below it
    assert 5.0 < np.mean(vals) <= 6.0


def test_non_symplectic_rejected():
    with pytest.raises(SymplecticConditionError):
        SparseGF2(2).apply_clifford(np.ones((4, 4), dtype=np.uint8))  # n=2, k=2, but not symplectic


@pytest.mark.parametrize(
    "n,M,qubits",
    [
        (3, np.eye(4, dtype=np.uint8), [0, 1, 2]),  # k=2 but 3 qubits
        (3, np.eye(4, dtype=np.uint8), [0, 5]),  # qubit out of range
        (4, np.eye(4, dtype=np.uint8), [1, 1]),  # duplicate qubit
        (3, np.eye(3, dtype=np.uint8), None),  # odd-dimension symp
    ],
)
def test_shape_and_qubit_validation(n, M, qubits):
    with pytest.raises(InvalidArgumentError):
        SparseGF2(n).apply_clifford(M, qubits)
