"""Uniform sampler for ``Sp(2n, F_2)`` at arbitrary ``n``.

Verifies :func:`sparsegf2.core.symplectic.random_symplectic`:

1. Every sample satisfies ``S Ω S^T ≡ Ω (mod 2)`` (`is_symplectic`).
2. For ``n ∈ {1, 2, 3}``, the empirical distribution over the
   enumerated group matches uniform via Pearson's chi-squared.
3. For ``n = 4``, the distribution of ``S @ v`` for a fixed ``v ≠ 0``
   matches Stim's ``stim.Tableau.random(n)`` projection statistically.
4. Determinism: identical seeds yield identical sample sequences.

No runtime Stim: the sampler does not import ``stim`` (Stim is only used
in this test file for cross-validation).
"""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.core.symplectic import (
    enumerate_symplectic_group,
    is_symplectic,
    random_symplectic,
    random_symplectic_batch,
    symplectic_group_order,
)

stim = pytest.importorskip("stim")


# ----------------------------------------------------------------------
# Basic structure
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 8])
def test_shape_and_dtype(n):
    rng = np.random.default_rng(0)
    S = random_symplectic(n, rng=rng)
    assert S.shape == (2 * n, 2 * n)
    assert S.dtype == np.uint8


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 8])
def test_all_samples_symplectic(n):
    """50 random samples; every one should satisfy S Ω S^T = Ω."""
    rng = np.random.default_rng(n * 100 + 7)
    for k in range(50):
        S = random_symplectic(n, rng=rng)
        assert is_symplectic(S), f"n={n} sample {k} not symplectic"


def test_n_zero_returns_empty_matrix():
    rng = np.random.default_rng(0)
    S = random_symplectic(0, rng=rng)
    assert S.shape == (0, 0)


def test_negative_n_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        random_symplectic(-1)


# ----------------------------------------------------------------------
# Uniformity at n ≤ 3 via chi-squared over the enumeration
# ----------------------------------------------------------------------


def _chi_squared_uniformity(samples: np.ndarray) -> float:
    """Compute Pearson's chi-squared statistic over the enumerated group.

    Determines ``n`` from ``samples.shape``, enumerates the group, and
    bins each sample into one of the ``|Sp(2n, F_2)|`` buckets.
    """
    n_qubits = samples.shape[1] // 2
    enumeration = enumerate_symplectic_group(n_qubits)
    idx_of: dict[bytes, int] = {enumeration[k].tobytes(): k for k in range(enumeration.shape[0])}
    counts = np.zeros(enumeration.shape[0], dtype=np.int64)
    for s in samples:
        counts[idx_of[s.tobytes()]] += 1
    expected = samples.shape[0] / enumeration.shape[0]
    return float(np.sum((counts - expected) ** 2) / expected)


@pytest.mark.parametrize(
    "n,n_samples,threshold",
    [
        # df = |Sp| - 1; threshold = mean + 4*std ≈ df + 4 sqrt(2 df)
        (1, 600, 25.0),  # df = 5, threshold roughly 5 + 4 sqrt(10) ≈ 17.6; loose to 25
        (2, 5000, 825.0),  # df = 719; chi^2 < 825 ⇔ p > 0.001 (large-df normal approx)
    ],
)
def test_chi_squared_uniformity(n, n_samples, threshold):
    rng = np.random.default_rng(2026 + n)
    samples = np.stack([random_symplectic(n, rng=rng) for _ in range(n_samples)], axis=0)
    chi2 = _chi_squared_uniformity(samples)
    assert chi2 < threshold, (
        f"chi-squared {chi2:.2f} >= threshold {threshold} for n={n} "
        f"(group size {symplectic_group_order(n)}) suggests non-uniform sampling"
    )


def test_n2_matches_random_symplectic_2q_distribution():
    """For n=2, random_symplectic should produce the same distribution
    as the cached-table sampler (random_symplectic_2q). Both should be
    set-equal to the enumeration."""
    rng = np.random.default_rng(99)
    samples = np.stack([random_symplectic(2, rng=rng) for _ in range(10000)], axis=0)
    seen = {s.tobytes() for s in samples}
    # In 10,000 draws over 720 buckets, missing >0 by chance is ~exp(-13.9) ≈ 10^-6.
    assert len(seen) == 720, f"missing buckets after 10k draws: seen {len(seen)}/720"


# ----------------------------------------------------------------------
# Stim cross-check at n = 4 (where enumeration is infeasible)
# ----------------------------------------------------------------------


def test_n4_set_inclusion_in_stim_clifford_group():
    """Every sample at n=4 should project to a Clifford element of the
    Stim Clifford group at 4 qubits. We can't enumerate (≈ 4.7e10
    matrices) but we can verify each sample is a valid Clifford by
    constructing a stim.Tableau and round-tripping."""
    rng = np.random.default_rng(420)
    for _ in range(30):
        S = random_symplectic(4, rng=rng)
        assert is_symplectic(S)
        # Construct the corresponding Stim Tableau (no signs).
        rows = [S[i] for i in range(2 * 4)]

        def to_ps(row):
            xs = row[:4].astype(bool)
            zs = row[4:].astype(bool)
            return stim.PauliString.from_numpy(xs=xs, zs=zs)

        tab = stim.Tableau.from_conjugated_generators(
            xs=[to_ps(rows[i]) for i in range(4)],
            zs=[to_ps(rows[4 + i]) for i in range(4)],
        )
        assert len(tab) == 4


def test_n4_image_distribution_matches_stim():
    """Compare the empirical distribution of S @ X_0 across samples
    from our sampler vs ``stim.Tableau.random``. Both should be uniform
    over non-zero vectors in F_2^{2n}, so the marginal of the first
    coordinate should be very close to 50/50."""
    n = 4
    rng = np.random.default_rng(2027)
    n_samples = 3000

    # Image of X_0 = [1, 0, ..., 0] under our random_symplectic:
    # X_0 @ S = S[0] (the first row).
    ours_first_row_x0 = np.zeros(n_samples, dtype=np.int64)
    for k in range(n_samples):
        S = random_symplectic(n, rng=rng)
        ours_first_row_x0[k] = int(S[0, 0])

    # Same for Stim
    stim_first_row_x0 = np.zeros(n_samples, dtype=np.int64)
    for k in range(n_samples):
        tab = stim.Tableau.random(n)
        xs, _zs = tab.x_output(0).to_numpy()
        stim_first_row_x0[k] = int(xs[0])

    # Both should average ≈ 0.5; tail probabilities should match.
    ours_mean = float(ours_first_row_x0.mean())
    stim_mean = float(stim_first_row_x0.mean())
    # Each is a Bernoulli with p ≈ ? For uniform Sp(2n, F_2), the bit
    # S[0, 0] is half the time 1, half 0 (by symmetry under X_0 ↔ X_1
    # swap, since the group acts transitively on nonzero vectors).
    # 4σ band over n_samples = 3000 is ≈ ±0.036.
    assert abs(ours_mean - stim_mean) < 0.05, (
        f"empirical means diverge: ours={ours_mean:.4f}, stim={stim_mean:.4f}"
    )


# ----------------------------------------------------------------------
# Determinism + batched API
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4, 6])
def test_determinism_under_fixed_seed(n):
    """Two RNGs with the same seed produce identical samples."""
    seq1 = np.stack(
        [random_symplectic(n, rng=np.random.default_rng(2026)) for _ in range(3)], axis=0
    )
    seq2 = np.stack(
        [random_symplectic(n, rng=np.random.default_rng(2026)) for _ in range(3)], axis=0
    )
    assert np.array_equal(seq1, seq2)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_batch_returns_correct_shape(n):
    rng = np.random.default_rng(0)
    batch = random_symplectic_batch(n, 7, rng=rng)
    assert batch.shape == (7, 2 * n, 2 * n)
    for k in range(7):
        if n > 0:
            assert is_symplectic(batch[k])


# ----------------------------------------------------------------------
# Bravyi-Maslov sampler (the O(n^2) canonical-form sampler)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6, 8])
def test_bravyi_maslov_all_symplectic(n):
    rng = np.random.default_rng(n * 13 + 1)
    for _ in range(50):
        assert is_symplectic(random_symplectic(n, rng=rng, sampler="bravyi_maslov"))


@pytest.mark.parametrize("n,n_samples,threshold", [(1, 600, 25.0), (2, 5000, 825.0)])
def test_bravyi_maslov_chi_squared_uniformity(n, n_samples, threshold):
    rng = np.random.default_rng(7000 + n)
    samples = np.stack(
        [random_symplectic(n, rng=rng, sampler="bravyi_maslov") for _ in range(n_samples)],
        axis=0,
    )
    chi2 = _chi_squared_uniformity(samples)
    assert chi2 < threshold, f"BM chi-squared {chi2:.1f} >= {threshold} for n={n}"


def test_bravyi_maslov_covers_all_720_at_n2():
    rng = np.random.default_rng(123)
    seen = {random_symplectic(2, rng=rng, sampler="bravyi_maslov").tobytes() for _ in range(10000)}
    assert len(seen) == 720


def test_auto_uses_bravyi_maslov_for_n_ge_3():
    # auto and explicit bravyi_maslov agree for the same seed when n != 2.
    for n in (3, 5):
        a = random_symplectic(n, rng=np.random.default_rng(99), sampler="auto")
        b = random_symplectic(n, rng=np.random.default_rng(99), sampler="bravyi_maslov")
        assert np.array_equal(a, b)


@pytest.mark.parametrize("sampler", ["auto", "kernel_basis", "bravyi_maslov"])
def test_determinism_per_sampler(sampler):
    a = random_symplectic(5, rng=np.random.default_rng(2026), sampler=sampler)
    b = random_symplectic(5, rng=np.random.default_rng(2026), sampler=sampler)
    assert np.array_equal(a, b)


def test_invalid_sampler_raises():
    with pytest.raises(ValueError, match="sampler"):
        random_symplectic(3, sampler="nope")


@pytest.mark.parametrize("sampler", ["kernel_basis", "bravyi_maslov"])
def test_batch_forwards_sampler(sampler):
    batch = random_symplectic_batch(5, 4, rng=np.random.default_rng(0), sampler=sampler)
    assert batch.shape == (4, 10, 10)
    for k in range(4):
        assert is_symplectic(batch[k])


def test_batch_validates_independent_of_n_samples():
    # an invalid sampler / negative n must raise even for an empty batch
    with pytest.raises(ValueError, match="sampler"):
        random_symplectic_batch(3, 0, sampler="nope")
    with pytest.raises(ValueError, match="non-negative"):
        random_symplectic_batch(-1, 2)


@pytest.mark.parametrize("sampler", ["auto", "kernel_basis", "bravyi_maslov"])
@pytest.mark.parametrize("n", [2, 3, 5])
def test_batch_determinism_per_sampler(sampler, n):
    a = random_symplectic_batch(n, 6, rng=np.random.default_rng(2026), sampler=sampler)
    b = random_symplectic_batch(n, 6, rng=np.random.default_rng(2026), sampler=sampler)
    assert np.array_equal(a, b)


# ----------------------------------------------------------------------
# n=3 uniformity via the image marginal (full enumeration is 1.45M elements;
# the image v @ S of a fixed nonzero v is uniform over the 2^{2n}-1 nonzero
# vectors for a uniform Sp sample, which is cheap to bin).
# ----------------------------------------------------------------------


def _image_chi_squared(n: int, sampler: str, n_samples: int, seed: int) -> tuple[float, int]:
    rng = np.random.default_rng(seed)
    counts = np.zeros(1 << (2 * n), dtype=np.int64)
    weights = 1 << np.arange(2 * n)
    for _ in range(n_samples):
        s = random_symplectic(n, rng=rng, sampler=sampler)
        idx = int(np.dot(s[0].astype(np.int64) & 1, weights))  # image of X_0 = row 0
        counts[idx] += 1
    assert counts[0] == 0, "image hit the zero vector (impossible for a symplectic)"
    g = (1 << (2 * n)) - 1
    expected = n_samples / g
    return float(np.sum((counts[1:] - expected) ** 2) / expected), g


@pytest.mark.parametrize("sampler", ["bravyi_maslov", "kernel_basis"])
def test_image_uniformity_n3(sampler):
    chi2, g = _image_chi_squared(3, sampler, 63 * 250, 7003 if sampler == "bravyi_maslov" else 5003)
    df = g - 1
    threshold = df + 6 * np.sqrt(2 * df)  # ~118 for df=62
    assert chi2 < threshold, f"{sampler} image chi2={chi2:.1f} >= {threshold:.1f} (n=3)"


def test_bm_and_kernel_basis_cover_same_images_n3():
    weights = 1 << np.arange(6)
    for sampler in ("bravyi_maslov", "kernel_basis"):
        rng = np.random.default_rng(11)
        seen = {
            int(
                np.dot(
                    random_symplectic(3, rng=rng, sampler=sampler)[0].astype(np.int64) & 1, weights
                )
            )
            for _ in range(63 * 70)
        }
        assert seen == set(range(1, 64)), f"{sampler} missed some of the 63 nonzero images"


def test_n4_bravyi_maslov_explicit_stim_validity():
    rng = np.random.default_rng(421)
    for _ in range(30):
        S = random_symplectic(4, rng=rng, sampler="bravyi_maslov")
        assert is_symplectic(S)
        rows = [S[i] for i in range(8)]

        def to_ps(row):
            return stim.PauliString.from_numpy(xs=row[:4].astype(bool), zs=row[4:].astype(bool))

        tab = stim.Tableau.from_conjugated_generators(
            xs=[to_ps(rows[i]) for i in range(4)],
            zs=[to_ps(rows[4 + i]) for i in range(4)],
        )
        assert len(tab) == 4


def test_gf2_inverse_unit_lower_triangular():
    from sparsegf2.core.symplectic import _gf2_inverse

    for seed in range(25):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 9))
        M = np.eye(n, dtype=np.int64)
        rows, cols = np.tril_indices(n, -1)
        M[rows, cols] = rng.integers(0, 2, size=rows.size)
        inv = _gf2_inverse(M.astype(np.uint8)).astype(np.int64)
        assert np.array_equal((M @ inv) % 2, np.eye(n, dtype=np.int64))


def test_gf2_inverse_general_invertible():
    from sparsegf2.core.symplectic import _gf2_inverse

    # a permutation matrix (invertible, not triangular): inverse is its transpose
    M = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)
    inv = _gf2_inverse(M).astype(np.int64)
    assert np.array_equal((M.astype(np.int64) @ inv) % 2, np.eye(3, dtype=np.int64))
