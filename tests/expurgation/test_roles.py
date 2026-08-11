"""StabilizerCode: role bookkeeping and the elementary expurgation move."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2 import PAULI_X, PAULI_Z, SparseGF2
from sparsegf2.core.symplectic import is_symplectic
from sparsegf2.errors import InvalidArgumentError
from sparsegf2.expurgation import (
    ROLE_CHECK,
    ROLE_GAUGE,
    ROLE_LOGICAL,
    StabilizerCode,
)

# ----------------------------------------------------------------------
# Construction and structure queries
# ----------------------------------------------------------------------


def test_from_encoding_roles(example_a_code):
    code = example_a_code
    assert code.n == 4
    assert code.k == 2
    assert code.check_pairs().tolist() == [0, 1]
    assert code.logical_pairs().tolist() == [2, 3]
    assert code.gauge_pairs().tolist() == []
    assert np.array_equal(code.roles, [ROLE_CHECK, ROLE_CHECK, ROLE_LOGICAL, ROLE_LOGICAL])


def test_constructor_validation():
    sim = SparseGF2(3)
    with pytest.raises(InvalidArgumentError):
        StabilizerCode(sim, [0, 1])  # wrong length
    with pytest.raises(InvalidArgumentError):
        StabilizerCode(sim, [0, 1, 3])  # bad role value
    with pytest.raises(InvalidArgumentError):
        StabilizerCode.from_encoding(sim, [0, 0])  # repeated data qubit
    with pytest.raises(InvalidArgumentError):
        StabilizerCode.from_encoding(sim, [3])  # out of range
    with pytest.raises(InvalidArgumentError, match="exact integers"):
        StabilizerCode(sim, [0, 1, 1.9])
    with pytest.raises(InvalidArgumentError, match="exact integers"):
        StabilizerCode.from_encoding(sim, [True])


def test_roles_are_copied():
    sim = SparseGF2(2)
    roles = np.array([ROLE_CHECK, ROLE_LOGICAL], dtype=np.uint8)
    code = StabilizerCode(sim, roles)
    roles[0] = ROLE_GAUGE
    assert code.roles[0] == ROLE_CHECK


def test_copy_is_independent(example_a_code):
    code = example_a_code
    snap = code.copy()
    assert code.measure([3], [PAULI_X], strategy="gauge") == 3
    assert code.k == 1
    assert snap.k == 2
    assert np.array_equal(snap.roles, [ROLE_CHECK, ROLE_CHECK, ROLE_LOGICAL, ROLE_LOGICAL])


# ----------------------------------------------------------------------
# Commutation bits
# ----------------------------------------------------------------------


def test_commutation_bits_example_a(example_a_code):
    code = example_a_code
    # E = Z_3: anticommutes with check XXXX, with logical X-bar_a = IIXX,
    # and with logical X-bar_b = IIIX.
    syndrome, logical = code.commutation_bits([3], [PAULI_Z])
    assert syndrome.tolist() == [1, 0]
    assert logical.tolist() == [0, 1, 0, 1]
    # E = X_3: undetectable, acts as the logical Z-bar_b.
    syndrome, logical = code.commutation_bits([3], [PAULI_X])
    assert syndrome.tolist() == [0, 0]
    assert logical.tolist() == [0, 0, 1, 0]
    # E = Z-bar_a = IZZI: undetectable, acts as a logical (pair a).
    syndrome, logical = code.commutation_bits([1, 2], [PAULI_Z, PAULI_Z])
    assert syndrome.tolist() == [0, 0]
    assert logical.tolist() == [0, 1, 0, 0]
    # E = the check XXXX itself: undetectable and logically trivial.
    syndrome, logical = code.commutation_bits(range(4), [PAULI_X] * 4)
    assert syndrome.tolist() == [0, 0]
    assert logical.tolist() == [0, 0, 0, 0]


def test_gauge_pairs_contribute_no_bits(example_a_code):
    code = example_a_code
    code.measure([3], [PAULI_X], strategy="gauge")
    assert code.gauge_pairs().tolist() == [3]
    syndrome, logical = code.commutation_bits([3], [PAULI_Z])
    # Checks unchanged under the gauge strategy; only pair a is logical now.
    assert syndrome.shape == (2,)
    assert logical.shape == (2,)


# ----------------------------------------------------------------------
# The elementary move
# ----------------------------------------------------------------------


def test_measure_stabilizer_strategy(example_a_code):
    code = example_a_code
    pair = code.measure([3], [PAULI_X], strategy="stabilizer")
    assert pair == 3
    assert code.k == 1
    assert code.check_pairs().tolist() == [0, 1, 3]
    # The measured operator is now an explicit check row.
    assert int(code.sim.plt[code.n + 3, 3]) == int(PAULI_X)


def test_measure_gauge_strategy_keeps_checks(example_a_code):
    code = example_a_code
    pair = code.measure([3], [PAULI_X], strategy="gauge")
    assert pair == 3
    assert code.k == 1
    assert code.check_pairs().tolist() == [0, 1]
    assert code.gauge_pairs().tolist() == [3]


def test_measure_rejects_detectable_operator(example_a_code):
    # X_0 anticommutes with the check ZZII.
    with pytest.raises(InvalidArgumentError):
        example_a_code.measure([0], [PAULI_X])


def test_measure_skips_logically_trivial(example_a_code):
    code = example_a_code
    roles_before = code.roles.copy()
    assert code.measure(range(4), [PAULI_X] * 4) is None  # the check XXXX
    assert np.array_equal(code.roles, roles_before)


def test_measure_consumes_logical_z_already_in_pure_state_stabilizer(example_a_code):
    # Z-bar_a = IZZI is a pure-state stabilizer row, but it is logically
    # nontrivial for the code. Code-space projection must therefore spend
    # its logical pair instead of treating the measurement as deterministic.
    code = example_a_code
    before = code.sim.to_symplectic().copy()
    assert code.measure([1, 2], [PAULI_Z, PAULI_Z]) == 2
    assert code.k == 1
    assert code.gauge_pairs().tolist() == [2]
    assert np.array_equal(code.sim.to_symplectic(), before)


def test_measure_never_spends_preexisting_gauge_pair():
    # First expurgate pair 0.  The second operator anticommutes with that
    # gauge pair and with logical pair 1, so an unconstrained pure-state
    # measurement can pivot on pair 0 and fail to reduce k.  Code-space
    # projection must target the remaining logical pair.
    code = StabilizerCode(SparseGF2(2), [ROLE_LOGICAL, ROLE_LOGICAL])
    assert code.measure([0], [PAULI_X], strategy="gauge") == 0
    assert code.k == 1
    assert code.measure([0, 1], [PAULI_X, PAULI_X], strategy="gauge") == 1
    assert code.k == 0
    assert code.gauge_pairs().tolist() == [0, 1]
    assert is_symplectic(code.sim.to_symplectic())


@pytest.mark.parametrize("hybrid", [False, True])
def test_code_projection_preserves_canonical_tableau_in_sparse_and_dense_modes(hybrid):
    sim = SparseGF2(3, hybrid=hybrid)
    sim.apply_h(0)
    sim.apply_cx(0, 1)
    sim.apply_cx(1, 2)
    if hybrid:
        sim._switch_to_dense()
    code = StabilizerCode(sim, [ROLE_LOGICAL] * 3)
    assert code.measure([0, 2], [PAULI_X, PAULI_Z], strategy="stabilizer") is not None
    assert code.k == 2
    assert is_symplectic(code.sim.to_symplectic())


def test_measure_validates_strategy(example_a_code):
    with pytest.raises(InvalidArgumentError):
        example_a_code.measure([3], [PAULI_X], strategy="bogus")
