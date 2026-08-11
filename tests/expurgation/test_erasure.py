"""Erasure decoding: M assembly, r_M, recovery, candidate extraction."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.errors import InvalidArgumentError
from sparsegf2.expurgation import (
    expurgation_candidates,
    random_encoding,
    recovery_probability,
    sample_erasure,
    uncorrectable_matrix,
    uncorrectable_rank,
)

# ----------------------------------------------------------------------
# Worked example A: the matrix from the notes, verbatim
# ----------------------------------------------------------------------


def test_uncorrectable_matrix_example_a(example_a_code):
    M = uncorrectable_matrix(example_a_code, np.array([3]))
    # Rows (Z_3, X_3); columns: checks (XXXX, ZZII), then logical bits
    # (Z-bar_a, X-bar_a, Z-bar_b, X-bar_b).
    expected = np.array(
        [
            [1, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(M, expected)


def test_rank_and_recovery_example_a(example_a_code):
    code = example_a_code
    assert uncorrectable_rank(code, np.array([3])) == 1
    assert recovery_probability(code, np.array([3])) == 0.5
    cands = expurgation_candidates(code, np.array([3]))
    assert len(cands) == 1
    assert cands[0].qubits == (3,)
    assert cands[0].letters == (2,)  # X_3 = IIIX
    assert cands[0].weight == 1
    # Measuring the candidate repairs this erasure exactly.
    code.measure(cands[0].qubits, cands[0].letters, strategy="stabilizer")
    assert uncorrectable_rank(code, np.array([3])) == 0
    assert recovery_probability(code, np.array([3])) == 1.0


# ----------------------------------------------------------------------
# Cross-check M against dense symplectic products
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_uncorrectable_matrix_brute_force(seed):
    rng = np.random.default_rng(seed)
    code = random_encoding(10, range(5, 10), 3, geometry="all_to_all", rng=rng)
    erased = sample_erasure(10, rng, count=3)
    M = uncorrectable_matrix(code, erased)
    n = code.n
    mat = code.sim.to_symplectic()
    checks = code.check_pairs()
    logicals = code.logical_pairs()
    # Generator columns: check stab rows, then per logical pair its
    # stabilizer (Z-bar) and destabilizer (X-bar) rows.
    gen_rows = list(n + checks)
    for j in logicals:
        gen_rows.extend([n + j, j])
    for t, site in enumerate(erased):
        for j, (x, z) in enumerate([(0, 1), (1, 0)]):  # Z_site, X_site
            for c, r in enumerate(gen_rows):
                prod = (int(mat[r, site]) * z + int(mat[r, n + site]) * x) & 1
                assert M[2 * t + j, c] == prod


# ----------------------------------------------------------------------
# Candidate contract on random codes
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_candidates_contract(seed):
    rng = np.random.default_rng(100 + seed)
    code = random_encoding(12, range(6, 12), 4, geometry="all_to_all", rng=rng)
    erased = sample_erasure(12, rng, count=3)
    cands = expurgation_candidates(code, erased)
    assert len(cands) == uncorrectable_rank(code, erased)
    weights = [c.weight for c in cands]
    assert weights == sorted(weights)
    for c in cands:
        assert set(c.qubits) <= set(int(s) for s in erased)
        assert all(1 <= xz <= 3 for xz in c.letters)
        syndrome, logical = code.commutation_bits(c.qubits, c.letters)
        assert not syndrome.any()
        assert logical.any()


def test_empty_erasure(example_a_code):
    code = example_a_code
    empty = np.array([], dtype=np.int64)
    assert uncorrectable_matrix(code, empty).shape == (0, 6)
    assert uncorrectable_rank(code, empty) == 0
    assert recovery_probability(code, empty) == 1.0
    assert expurgation_candidates(code, empty) == []


def test_erasure_validation(example_a_code):
    with pytest.raises(InvalidArgumentError):
        uncorrectable_matrix(example_a_code, np.array([0, 0]))
    with pytest.raises(InvalidArgumentError):
        uncorrectable_matrix(example_a_code, np.array([4]))
    with pytest.raises(InvalidArgumentError, match="exact integers"):
        uncorrectable_matrix(example_a_code, np.array([1.9]))


# ----------------------------------------------------------------------
# Erasure sampling
# ----------------------------------------------------------------------


def test_sample_erasure_count():
    rng = np.random.default_rng(0)
    for _ in range(20):
        e = sample_erasure(10, rng, count=4)
        assert e.shape == (4,)
        assert np.array_equal(e, np.sort(e))
        assert np.unique(e).shape == (4,)
        assert e.min() >= 0 and e.max() < 10


def test_sample_erasure_rate_extremes():
    rng = np.random.default_rng(1)
    assert sample_erasure(8, rng, rate=0.0).shape == (0,)
    assert np.array_equal(sample_erasure(8, rng, rate=1.0), np.arange(8))


def test_sample_erasure_sites_pool():
    rng = np.random.default_rng(2)
    pool = np.array([1, 3, 5, 7])
    for _ in range(10):
        e = sample_erasure(8, rng, count=2, sites=pool)
        assert set(e.tolist()) <= set(pool.tolist())


def test_sample_erasure_validation():
    rng = np.random.default_rng(3)
    with pytest.raises(InvalidArgumentError):
        sample_erasure(8, rng)  # neither
    with pytest.raises(InvalidArgumentError):
        sample_erasure(8, rng, rate=0.1, count=2)  # both
    with pytest.raises(InvalidArgumentError):
        sample_erasure(8, rng, rate=1.5)
    with pytest.raises(InvalidArgumentError):
        sample_erasure(8, rng, count=9)
    with pytest.raises(InvalidArgumentError):
        sample_erasure(8, rng, count=1, sites=np.array([8]))
    with pytest.raises(InvalidArgumentError, match="exact integer"):
        sample_erasure(8, rng, count=1.9)
    with pytest.raises(InvalidArgumentError, match="exact integers"):
        sample_erasure(8, rng, count=1, sites=np.array([1.9]))
    with pytest.raises(InvalidArgumentError):
        sample_erasure(8, rng, rate=True)
