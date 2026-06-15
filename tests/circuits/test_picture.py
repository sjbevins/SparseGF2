"""Tests for picture setup (pure_state + purification)."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2 import SparseGF2, code_dimension
from sparsegf2.circuits.picture import Picture, PictureSpec, setup_picture
from sparsegf2.errors import InvalidArgumentError


def test_picture_strenum_equality():
    assert Picture.PURE_STATE == "pure_state"
    assert Picture("purification") is Picture.PURIFICATION
    assert str(Picture.PURE_STATE) == "pure_state"


def test_pure_state_layout():
    sim, spec = setup_picture(Picture.PURE_STATE, 8)
    assert isinstance(sim, SparseGF2)
    assert sim.n == 8
    assert spec.total_qubits == 8
    assert spec.system_qubits.tolist() == list(range(8))
    assert spec.reference_qubits.tolist() == []
    assert spec.order_parameter == "none"
    # fresh |0^n>: half-cut entropy is 0
    assert isinstance(spec, PictureSpec)


def test_purification_layout_and_full_entanglement():
    sim, spec = setup_picture("purification", 8)
    assert sim.n == 16  # 2 * n_system
    assert spec.total_qubits == 16
    assert spec.system_qubits.tolist() == list(range(8))
    assert spec.reference_qubits.tolist() == list(range(8, 16))
    assert spec.order_parameter == "code_dimension"
    # fresh Bell purification: every system qubit entangled with reference
    assert code_dimension(sim, 8) == 8


def test_rng_is_forwarded_for_reproducibility():
    # same seed -> identical measurement behavior
    sim_a, _ = setup_picture(Picture.PURE_STATE, 6, rng=np.random.default_rng(123))
    sim_b, _ = setup_picture(Picture.PURE_STATE, 6, rng=np.random.default_rng(123))
    # put both into a random-outcome situation: H then measure
    for s in (sim_a, sim_b):
        s.apply_h(0)
    out_a = [sim_a.measure_z(0) for _ in range(5)]
    out_b = [sim_b.measure_z(0) for _ in range(5)]
    assert out_a == out_b


def test_pivot_and_numba_forwarded():
    sim, _ = setup_picture(Picture.PURE_STATE, 4, pivot_mode="first", use_numba=False)
    assert sim.pivot_mode == "first"
    assert sim.use_numba is False


def test_invalid_picture_and_n():
    with pytest.raises(InvalidArgumentError):
        setup_picture("teleport", 8)
    with pytest.raises(InvalidArgumentError):
        setup_picture(Picture.PURE_STATE, 0)
