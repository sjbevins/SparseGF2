"""The random-encoding convenience builder."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.errors import InvalidArgumentError
from sparsegf2.expurgation import random_encoding

# ----------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------


def test_depth_zero_is_trivial():
    code = random_encoding(6, [4, 5], 0, rng=np.random.default_rng(0))
    assert code.n == 6
    assert code.k == 2
    assert code.logical_pairs().tolist() == [4, 5]
    # No gates applied: the tableau is the identity.
    assert np.array_equal(code.sim.to_symplectic(), np.eye(12, dtype=np.uint8))


@pytest.mark.parametrize("geometry", ["brickwork", "all_to_all"])
def test_seeded_encoding_is_reproducible(geometry):
    a = random_encoding(10, range(5, 10), 4, geometry=geometry, rng=np.random.default_rng(42))
    b = random_encoding(10, range(5, 10), 4, geometry=geometry, rng=np.random.default_rng(42))
    assert np.array_equal(a.sim.to_symplectic(), b.sim.to_symplectic())


def test_odd_n_all_to_all():
    code = random_encoding(7, [6], 3, geometry="all_to_all", rng=np.random.default_rng(1))
    assert code.n == 7
    assert code.k == 1


def test_sim_kwargs_forwarded():
    code = random_encoding(
        6, [5], 2, rng=np.random.default_rng(2), use_numba=False, pivot_mode="first"
    )
    assert code.sim.use_numba is False
    assert code.sim.pivot_mode == "first"


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_encoding_validation():
    with pytest.raises(InvalidArgumentError):
        random_encoding(6, [0], -1)
    with pytest.raises(InvalidArgumentError):
        random_encoding(6, [0], 2, geometry="ring")
