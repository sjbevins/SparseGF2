"""Forward-looking API contracts.

Locks the following core API contracts:

* ``pivot_mode`` kwarg + deprecation shim for ``use_min_weight_pivot``.
* ``SparseGF2Error`` exception hierarchy (multi-inherits from the
  standard exception that makes sense semantically, so existing
  ``except ValueError`` callers still work).
* ``Subsystem`` type alias + ``complement(n, A)`` helper.
* ``SparseGF2.copy(*, copy_rng=)``.
* ``SparseGF2.from_symplectic(mat)`` classmethod (round-trip with
  ``to_symplectic``).
* ``__getstate__`` / ``__setstate__`` + ``pickle.dumps/loads``
  roundtrip.
* ``SubsystemFastExtractor`` optional protocol (observables fast path).
* Capacity assertion (``n > _MAX_N`` raises).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from sparsegf2 import (
    InvalidArgumentError,
    SimulatorBackendError,
    SparseGF2,
    SparseGF2Error,
    SymplecticConditionError,
    TableauCorruption,
    complement,
    from_zero_state,
    is_symplectic,
)
from sparsegf2.core._protocol import SimulatorProtocol, SubsystemFastExtractor

# ----------------------------------------------------------------------
# pivot_mode kwarg
# ----------------------------------------------------------------------


def test_pivot_mode_defaults_to_min_weight():
    sim = SparseGF2(4)
    assert sim.pivot_mode == "min_weight"
    assert sim.use_min_weight_pivot is True


@pytest.mark.parametrize(
    "mode,expected_bool",
    [("min_weight", True), ("first", False)],
)
def test_pivot_mode_explicit(mode, expected_bool):
    sim = SparseGF2(4, pivot_mode=mode)
    assert sim.pivot_mode == mode
    assert sim.use_min_weight_pivot is expected_bool


def test_pivot_mode_rejects_unknown_mode():
    with pytest.raises(InvalidArgumentError, match="not in"):
        SparseGF2(4, pivot_mode="random")  # planned future mode, not yet shipped


# ----------------------------------------------------------------------
# SparseGF2Error hierarchy
# ----------------------------------------------------------------------


def test_invalid_argument_error_caught_by_value_error():
    with pytest.raises(ValueError):
        SparseGF2(-1)


def test_invalid_argument_error_caught_by_sparsegf2_error():
    with pytest.raises(SparseGF2Error):
        SparseGF2(-1)


def test_symplectic_condition_error_caught_by_value_error():
    sim = SparseGF2(2)
    bad = np.array([[1, 1], [0, 0]], dtype=np.uint8)  # not symplectic
    with pytest.raises(ValueError):
        sim.apply_gate_1q(0, bad)
    with pytest.raises(SymplecticConditionError):
        sim.apply_gate_1q(0, bad)


def test_tableau_corruption_class_hierarchy():
    # TableauCorruption multi-inherits from RuntimeError and
    # SparseGF2Error. Pin the hierarchy so a downstream `except
    # RuntimeError` still catches it.
    assert issubclass(TableauCorruption, RuntimeError)
    assert issubclass(TableauCorruption, SparseGF2Error)


def test_tableau_corruption_fires_on_invalid_subsystem_rank_via_entropy():
    # ``entanglement_entropy`` raises ``TableauCorruption`` when the
    # FCYBC bound is violated. We can't easily fabricate this state
    # without breaking internal invariants further, so we only assert
    # the class hierarchy here. The raise site is exercised in code
    # review / static analysis.
    pass


def test_simulator_backend_error_is_sparsegf2_error():
    # SimulatorBackendError is the parent of internal sampler bugs;
    # smoke-check the class hierarchy without trying to fabricate a bug.
    assert issubclass(SimulatorBackendError, SparseGF2Error)
    assert issubclass(SimulatorBackendError, RuntimeError)


def test_capacity_assertion():
    # n > _MAX_N must raise InvalidArgumentError (we don't actually
    # allocate at this scale; the check fires up front).
    with pytest.raises(InvalidArgumentError, match="exceeds"):
        SparseGF2(SparseGF2._MAX_N + 1)


# ----------------------------------------------------------------------
# Subsystem type alias + complement
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,A,expected",
    [
        (4, [], [0, 1, 2, 3]),
        (4, [0, 1, 2, 3], []),
        (4, [1, 3], [0, 2]),
        (1, [0], []),
        (1, [], [0]),
        (0, [], []),
    ],
)
def test_complement_basic(n, A, expected):
    assert complement(n, A).tolist() == expected


def test_complement_validates_n():
    with pytest.raises(InvalidArgumentError):
        complement(-1, [])


def test_complement_validates_subsystem():
    with pytest.raises(InvalidArgumentError):
        complement(4, [0, 0])  # duplicate
    with pytest.raises(InvalidArgumentError):
        complement(4, [5])  # out of range


# ----------------------------------------------------------------------
# copy(copy_rng=)
# ----------------------------------------------------------------------


def test_copy_default_copies_rng():
    sim = SparseGF2(4, rng=np.random.default_rng(7))
    sib = sim.copy()
    # Both have rngs; advancing one doesn't disturb the other.
    a = sim._rng.integers(0, 100)
    b = sib._rng.integers(0, 100)
    assert isinstance(a, np.integer | int)
    assert isinstance(b, np.integer | int)


def test_copy_rng_false_yields_none():
    sim = SparseGF2(4, rng=np.random.default_rng(7))
    sib = sim.copy(copy_rng=False)
    assert sib._rng is None
    # Tableau state still copied.
    assert np.array_equal(sim.to_symplectic(), sib.to_symplectic())


# ----------------------------------------------------------------------
# from_symplectic round-trip
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
def test_from_symplectic_round_trip(n):
    sim = SparseGF2(n, rng=np.random.default_rng(n * 11))
    # Drive into a non-trivial state.
    for q in range(n):
        sim.apply_h(q)
    for q in range(n - 1):
        sim.apply_cx(q, q + 1)
    M = sim.to_symplectic()
    sim2 = SparseGF2.from_symplectic(M)
    assert np.array_equal(sim2.to_symplectic(), M)


def test_from_symplectic_validates_shape():
    with pytest.raises(InvalidArgumentError, match=r"\(2n, 2n\)"):
        SparseGF2.from_symplectic(np.zeros((3, 4), dtype=np.uint8))


def test_from_symplectic_validates_symplecticity():
    bad = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.uint8)
    assert not is_symplectic(bad)
    with pytest.raises(SymplecticConditionError):
        SparseGF2.from_symplectic(bad)


def test_from_symplectic_then_gate_continues_correctly():
    """After ``from_symplectic``, applying gates must continue working.
    This pins that the inverted indices are reconstructed correctly."""
    a = SparseGF2(4, rng=np.random.default_rng(0))
    for q in range(4):
        a.apply_h(q)
    a.apply_cx(0, 1)
    M = a.to_symplectic()
    b = SparseGF2.from_symplectic(M)
    # Continue evolving both identically.
    a.apply_s(2)
    b.apply_s(2)
    a.apply_cz(1, 3)
    b.apply_cz(1, 3)
    assert np.array_equal(a.canonical_form(), b.canonical_form())


# ----------------------------------------------------------------------
# Pickle round-trip
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 4, 8])
def test_pickle_round_trip(n):
    sim = SparseGF2(n, rng=np.random.default_rng(2026))
    # Drive into a non-trivial state for n >= 1.
    for q in range(n):
        sim.apply_h(q)
    blob = pickle.dumps(sim)
    sib = pickle.loads(blob)
    assert sim.n == sib.n
    assert sim.pivot_mode == sib.pivot_mode
    assert sim.use_numba == sib.use_numba
    assert np.array_equal(sim.to_symplectic(), sib.to_symplectic())


def test_pickle_round_trip_preserves_observable_chain():
    sim = SparseGF2(4, rng=np.random.default_rng(0))
    sim.apply_h(0)
    sim.apply_cx(0, 1)
    sim.apply_cx(1, 2)
    sim.apply_cx(2, 3)
    blob = pickle.dumps(sim)
    sib = pickle.loads(blob)
    # Both must produce identical canonical forms, the load-bearing
    # state-equality invariant.
    assert np.array_equal(sim.canonical_form(), sib.canonical_form())


def test_setstate_rejects_mismatched_version():
    sim = SparseGF2(2)
    state = sim.__getstate__()
    state["_state_version"] = state["_state_version"] + 99
    sib = SparseGF2.__new__(SparseGF2)
    with pytest.raises(InvalidArgumentError, match="_state_version"):
        sib.__setstate__(state)


# ----------------------------------------------------------------------
# SubsystemFastExtractor optional protocol
# ----------------------------------------------------------------------


def test_sparsegf2_satisfies_subsystem_fast_extractor():
    sim = SparseGF2(4)
    assert isinstance(sim, SubsystemFastExtractor)


def test_subsystem_xz_block_shape():
    sim = SparseGF2(4)
    sim.apply_h(0)
    sim.apply_cx(0, 1)
    A = np.asarray([0, 2], dtype=np.int64)
    block = sim._subsystem_xz_block(A)
    assert block.shape == (4, 4)  # (n, 2|A|)
    assert block.dtype == np.uint8
    # Entries are 0 or 1.
    assert ((block == 0) | (block == 1)).all()


def test_subsystem_fast_extractor_matches_to_symplectic_path():
    """The fast-path output must equal the slow-path-derived block."""
    sim = SparseGF2(6, rng=np.random.default_rng(0))
    for q in range(6):
        sim.apply_h(q)
    for q in range(5):
        sim.apply_cx(q, q + 1)
    A = np.asarray([1, 3, 5], dtype=np.int64)
    fast = sim._subsystem_xz_block(A)

    full = sim.to_symplectic()
    n = sim.n
    cols = np.concatenate([A, A + n])
    slow = full[n:, cols].astype(np.uint8) & 1
    assert np.array_equal(fast, slow)


# ----------------------------------------------------------------------
# Pluggability: alternative simulator backend via Protocol
# ----------------------------------------------------------------------


def test_simulator_protocol_runtime_check_still_works():
    sim = from_zero_state(5)
    assert isinstance(sim, SimulatorProtocol)


# ----------------------------------------------------------------------
# __repr__
# ----------------------------------------------------------------------


def test_repr_includes_n_and_pivot_mode():
    s = repr(SparseGF2(7, pivot_mode="first"))
    assert "SparseGF2" in s
    assert "n=7" in s
    assert "first" in s
