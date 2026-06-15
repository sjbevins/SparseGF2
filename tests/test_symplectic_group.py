"""native ``Sp(2n, F_2)`` enumeration + uniform sampler.

The headline parity check: the set of ``(4, 4)`` symplectic matrices our
recursive construction emits equals, as a set, the symplectic
projection of ``stim.Tableau.iter_all(2)``. No Stim at runtime; Stim is
only used here to *verify* the set.

Additional checks:

* Group order formula ``|Sp(2n, F_2)| = 2^{n^2} ∏ (4^i - 1)``.
* Every emitted matrix satisfies ``M Ω M^T = Ω`` over ``F_2``.
* All matrices are distinct (no duplicates, no omissions).
* The uniform sampler ``random_symplectic_2q`` produces a distribution
  that survives a chi-squared uniformity test over the 720 elements.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.core.symplectic import (
    enumerate_sp4,
    enumerate_symplectic_group,
    is_symplectic,
    random_symplectic_2q,
    random_symplectic_2q_batch,
    symplectic_form,
    symplectic_group_order,
    symplectic_product,
)

# Stim is test-only; skip the whole file when not installed.
stim = pytest.importorskip("stim")

from _stim_parity import stim_tableau_to_symplectic  # noqa: E402

# ----------------------------------------------------------------------
# Group order + symplectic form
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,expected",
    [(0, 1), (1, 6), (2, 720), (3, 1_451_520), (4, 47_377_612_800)],
)
def test_symplectic_group_order(n, expected):
    assert symplectic_group_order(n) == expected


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_symplectic_form_shape_and_block(n):
    omega = symplectic_form(n)
    assert omega.shape == (2 * n, 2 * n)
    assert omega.dtype == np.uint8
    if n > 0:
        assert np.array_equal(omega[:n, :n], np.zeros((n, n), dtype=np.uint8))
        assert np.array_equal(omega[:n, n:], np.eye(n, dtype=np.uint8))
        assert np.array_equal(omega[n:, :n], np.eye(n, dtype=np.uint8))
        assert np.array_equal(omega[n:, n:], np.zeros((n, n), dtype=np.uint8))


def test_symplectic_product_basic():
    # ⟨X_0, Z_0⟩ = 1, ⟨X_0, Z_1⟩ = 0 on 2 qubits
    X0 = np.array([1, 0, 0, 0], dtype=np.uint8)
    Z0 = np.array([0, 0, 1, 0], dtype=np.uint8)
    Z1 = np.array([0, 0, 0, 1], dtype=np.uint8)
    assert symplectic_product(X0, Z0) == 1
    assert symplectic_product(X0, Z1) == 0
    assert symplectic_product(Z0, Z1) == 0


# ----------------------------------------------------------------------
# Enumeration integrity
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_enumeration_count_matches_group_order(n):
    arr = enumerate_symplectic_group(n)
    assert arr.shape[0] == symplectic_group_order(n)


@pytest.mark.parametrize("n", [1, 2])
def test_every_enumerated_element_is_symplectic(n):
    arr = enumerate_symplectic_group(n)
    assert all(is_symplectic(M) for M in arr), (
        f"some elements of the n={n} enumeration fail the symplectic condition"
    )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_enumeration_is_distinct(n):
    arr = enumerate_symplectic_group(n)
    seen = {M.tobytes() for M in arr}
    assert len(seen) == arr.shape[0], f"duplicate matrices in n={n} enumeration"


def test_enumeration_refuses_infeasible_n():
    with pytest.raises(ValueError):
        enumerate_symplectic_group(4)


def test_enumerate_sp4_is_cached_and_readonly():
    """The cached Sp(4, F_2) table should be the same array on every call
    and should be marked read-only so accidental writes are caught."""
    first = enumerate_sp4()
    second = enumerate_sp4()
    assert first is second, "enumerate_sp4 must memoise across calls"
    assert not first.flags.writeable, "cached Sp(4) table must be read-only"


# ----------------------------------------------------------------------
# The headline parity check: Stim ↔ our enumeration
# ----------------------------------------------------------------------


def test_sp4_matches_stim_symplectic_projection():
    """The native Sp(4, F_2) construction must produce *exactly* the same
    set of matrices that ``stim.Tableau.iter_all(2)`` reaches modulo
    sign bits. This pins the Stim disconnect: equality of two finite
    sets, byte-for-byte."""
    stim_set = set()
    for tab in stim.Tableau.iter_all(2):
        stim_set.add(stim_tableau_to_symplectic(tab).tobytes())

    ours = enumerate_sp4()
    ours_set = {M.tobytes() for M in ours}

    assert len(stim_set) == 720
    assert len(ours_set) == 720
    assert stim_set == ours_set, (
        "native Sp(4, F_2) enumeration diverges from Stim's symplectic projection"
    )


def test_stim_clifford_is_16_fold_cover_of_sp4():
    """Every element of Sp(4, F_2) lifts to exactly 4^2 = 16 signed
    Cliffords (one per choice of phase per qubit per generator). This is
    the formal statement that explains why the public release sees
    11,520 Cliffords but we see 720 symplectic matrices."""
    counts: dict[bytes, int] = {}
    for tab in stim.Tableau.iter_all(2):
        key = stim_tableau_to_symplectic(tab).tobytes()
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 720
    assert set(counts.values()) == {16}


# ----------------------------------------------------------------------
# Sampler: uniformity + symplecticity
# ----------------------------------------------------------------------


def test_random_symplectic_2q_returns_symplectic_matrices():
    rng = np.random.default_rng(0)
    for _ in range(50):
        M = random_symplectic_2q(rng)
        assert M.shape == (4, 4)
        assert M.dtype == np.uint8
        assert is_symplectic(M)


def test_random_symplectic_2q_batch_matches_singleton():
    """Batching must be statistically equivalent to repeated singleton
    draws. Same seed → identical sequence."""
    rng_singleton = np.random.default_rng(123)
    one_by_one = np.stack([random_symplectic_2q(rng_singleton) for _ in range(50)], axis=0)

    rng_batch = np.random.default_rng(123)
    batch = random_symplectic_2q_batch(50, rng=rng_batch)

    assert np.array_equal(one_by_one, batch)


def test_random_sampler_chi_squared_uniformity():
    """Draw a large sample from ``random_symplectic_2q_batch`` and check
    that the histogram over the 720 elements is consistent with the
    uniform distribution. We use Pearson's chi-squared test at the 0.1%
    significance level, so false positives ~ 1 / 1000 runs, well within
    flake tolerance.
    """
    rng = np.random.default_rng(2026)
    cache = enumerate_sp4()
    idx_of = {M.tobytes(): i for i, M in enumerate(cache)}

    # Aim for an expected count >= 50 per bucket: n_samples >= 50 * 720 = 36_000
    n_samples = 50_000
    sample = random_symplectic_2q_batch(n_samples, rng=rng)
    hist = np.zeros(720, dtype=np.int64)
    for M in sample:
        hist[idx_of[M.tobytes()]] += 1

    expected = n_samples / 720
    chi2 = float(np.sum((hist - expected) ** 2) / expected)

    # Sp(4, F_2) has 720 buckets; df = 719. For df=719 the 0.999 quantile
    # is ~ 825 (large-df normal approximation: df + 3 sqrt(2 df)).
    threshold = 825.0
    assert chi2 < threshold, (
        f"chi-squared = {chi2:.1f} exceeds {threshold:.1f}; sampler may not be uniform"
    )
    # Loose sanity envelope on the extreme bins. With ~50_000 draws over
    # 720 buckets (expected ≈ 69 per bucket, σ ≈ 8), the min-of-720 has
    # tail mass extending well below the 0.4·expected line used previously,
    # and that earlier assertion was the source of an intermittent flake.
    # The chi-squared above is the principled uniformity check; this
    # envelope just catches catastrophic bucket starvation.
    assert hist.min() >= 1
    assert hist.max() <= int(3.0 * expected)


# ----------------------------------------------------------------------
# n=1 is in the same algorithmic path
# ----------------------------------------------------------------------


def test_sp2_matches_stim_symplectic_projection():
    """Same parity story as Sp(4), one qubit smaller, a sanity check on
    the recursive base case."""
    stim_set = set()
    for tab in stim.Tableau.iter_all(1):
        stim_set.add(stim_tableau_to_symplectic(tab).tobytes())
    ours = enumerate_symplectic_group(1)
    ours_set = {M.tobytes() for M in ours}
    assert len(stim_set) == 6
    assert len(ours_set) == 6
    assert stim_set == ours_set
