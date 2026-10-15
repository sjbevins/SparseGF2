"""mu_D, distance bounds and the hybrid P_C(m) curve (analysis.py)."""

import math

import numpy as np
import pytest

from sparsegf2 import StabilizerCode, analysis

# [[10, 4]]: the [[5,1,3]] code on qubits 0-4, [[4,2,2]] on 5-8 and a bare qubit 9. A
# correctable erasure holds at most 2 qubits of the first block, 1 of the second and not
# qubit 9, so every 4-qubit erasure is uncorrectable: P_C(4) = 1 and every D <= 4.
TEN_FOUR = [
    "XZZX______",
    "_XZZX_____",
    "X_XZZ_____",
    "ZX_XZ_____",
    "_____XXXX_",
    "_____ZZZZ_",
]
# k = 3: [[4,2,2]] on qubits 0-3, [[5,1,3]] on 4-8, qubit 9 frozen; only pairs in 0-3 fail
FOUR_FIVE_FROZEN = [
    "XXXX______",
    "ZZZZ______",
    "____XZZX__",
    "_____XZZX_",
    "____X_XZZ_",
    "____ZX_XZ_",
    "_________Z",
]
# k = 2: two [[5,1,3]] codes, distance 3
FIVE_FIVE = [
    "XZZX______",
    "_XZZX_____",
    "X_XZZ_____",
    "ZX_XZ_____",
    "_____XZZX_",
    "______XZZX",
    "_____X_XZZ",
    "_____ZX_XZ",
]


def _code_data(code, samples=200, w=2):
    first = code.first_failure_sizes(samples, seed=0)
    counts = code.uncorrectable_counts(w)
    return first, counts


def test_422_code_has_distance_two_and_mean_first_failure_two():
    code = StabilizerCode(["XXXX", "ZZZZ"])  # [[4, 2, 2]]: every pair is uncorrectable
    first, counts = _code_data(code)
    assert np.all(first == 2)
    assert analysis.mean_first_failure(first, 4) == pytest.approx(2.0)
    assert analysis.mean_first_failure(first, 4, exact_counts=counts) == pytest.approx(2.0)
    low, high = analysis.distance_bounds(first, 4, exact_counts=counts)
    assert (int(low), int(high)) == (2, 2)


def test_code_without_logical_qubits_gives_n_plus_one():
    code = StabilizerCode(["XX", "ZZ"])  # k = 0
    first, counts = _code_data(code, w=1)
    assert np.all(first == 3)
    assert analysis.mean_first_failure(first, 2, exact_counts=counts) == pytest.approx(3.0)
    low, high = analysis.distance_bounds(first, 2, exact_counts=counts)
    assert (int(low), int(high)) == (3, 3)


def test_without_counts_mean_first_failure_is_the_sample_mean():
    rng = np.random.default_rng(1)
    first = rng.integers(1, 12, size=(5, 40))  # n = 10, D in [1, 11]
    np.testing.assert_allclose(analysis.mean_first_failure(first, 10), first.mean(axis=-1))


def test_hybrid_curve_uses_exact_counts_up_to_w_and_reproduces_phi():
    rng = np.random.default_rng(2)
    n, w = 12, 2
    first = rng.integers(3, 14, size=(4, 64))  # no failures at m <= 2 in the samples
    counts = np.array([[0, 3], [0, 0], [1, 5], [0, 1]], dtype=float)
    curve = analysis.hybrid_failure_by_size(first, n, exact_counts=counts)
    assert curve.shape == (4, n + 1)
    np.testing.assert_allclose(curve[:, 1 : w + 1], analysis.exact_failure_by_size(counts, n))
    np.testing.assert_allclose(curve[:, w + 1 :], analysis.failure_by_size(first, n)[:, w + 1 :])
    for e in (0.05, 0.2, 0.5):
        m = np.arange(n + 1)
        b = np.array([np.exp(analysis._log_binomial(n)[i]) for i in m]) * e**m * (1 - e) ** (n - m)
        np.testing.assert_allclose(
            curve @ b, analysis.erasure_failure_probability(first, n, e, exact_counts=counts)
        )


def test_distance_bounds_without_counts_and_with_zero_counts():
    first = np.array([[5, 7, 6], [4, 4, 9]])
    low, high = analysis.distance_bounds(first, 10)
    np.testing.assert_array_equal(low, [1, 1])
    np.testing.assert_array_equal(high, [5, 4])
    low, high = analysis.distance_bounds(first, 10, exact_counts=np.zeros((2, 3)))
    np.testing.assert_array_equal(low, [4, 4])  # w + 1 = 4
    np.testing.assert_array_equal(high, [5, 4])


def test_junction_dip_flags_a_sampled_value_below_the_exact_one():
    n = 6
    counts = np.array([[0, 15]], dtype=float)  # every pair fails: exact P(2) = 1
    first = np.array([[6, 6, 6, 6]])  # samples say P(3) = 0 < 1
    assert analysis.junction_dip(first, n, counts).tolist() == [True]
    first_ok = np.array([[2, 2, 2, 2]])
    assert analysis.junction_dip(first_ok, n, counts).tolist() == [False]


# -- exact arithmetic of P_C(m) (E1, E2) and the e_delta junction ---------------------


def test_sampled_failure_by_size_divides_integer_counts_once():
    """E2 regression: P_C(m) is cumsum(counts) / samples, never a sum of fractions."""
    code = StabilizerCode(TEN_FOUR)
    n, w = 10, 4
    counts = code.uncorrectable_counts(w)
    assert code.num_logical == 4 and counts.tolist() == [1, 15, 80, 210]  # exact P_C(4) = 1
    first = np.repeat([1, 2, 3, 4], [13, 21, 30, 36])  # D histogram [13, 21, 30, 36]
    p = analysis.failure_by_size(first, n)
    assert p.tolist() == [0.0, 13 / 100, 34 / 100, 64 / 100] + [1.0] * 7
    assert p[n] == 1.0
    assert not analysis.junction_dip(first, n, counts)  # P_C(5) = 1 = P_C(4): no dip
    for seed in range(100):  # every D <= w and P_C(w) = 1, so no sample can ever dip
        assert not analysis.junction_dip(code.first_failure_sizes(100, seed=seed), n, counts)


def test_failure_by_size_reaches_one_exactly_for_codes_with_logical_qubits():
    rng = np.random.default_rng(5)
    for n in (5, 10, 64, 129):
        for samples in (3, 7, 100, 999):
            d = rng.integers(1, n + 1, size=(20, samples))  # D <= n: every code has k > 0
            p = analysis.failure_by_size(d, n)
            assert np.all(p[:, n] == 1.0)
            hist = np.stack([np.bincount(row, minlength=n + 2) for row in d])
            np.testing.assert_array_equal(p, np.cumsum(hist, axis=1)[:, : n + 1] / samples)


def test_exact_failure_by_size_tolerates_rounded_float_counts():
    """E1: float counts beyond 2**53 may round above C(n, m); the ratio is clipped to 1."""
    n = 200
    binom = np.array([float(math.comb(n, m)) for m in range(1, 14)])
    over = binom * (1 + 5e-13)
    assert np.all(over > binom)
    np.testing.assert_array_equal(analysis.exact_failure_by_size(over, n), 1.0)
    np.testing.assert_array_equal(analysis.exact_failure_by_size(binom, n), 1.0)
    for bad in (binom * (1 + 1e-9), binom * 1.5, -binom):
        with pytest.raises(ValueError, match=r"\[0, C\(n, m\)\]"):
            analysis.exact_failure_by_size(bad, n)
    for bad in ([7, 0], [6, 16], [-1e-300, 0], [np.nan, 0], [np.inf, 0]):  # C(6, 1..2) = 6, 15
        with pytest.raises(ValueError, match=r"\[0, C\(n, m\)\]"):
            analysis.exact_failure_by_size(bad, 6)


def test_erasure_tolerance_inverts_the_hybrid_curve_with_samples_below_w():
    """e_delta comes from the same hybrid Phi as erasure_failure_probability, junction included.

    The samples contain D <= w and the counts are nonzero, so the hybrid's replacement of
    Pr[Bin >= D] by Pr[Bin >= w + 1] for those samples matters (audit mutant
    ``tolerance_hybrid_junction`` moved e_0.05 by 15 % on such data). The hybrid Phi is
    first checked against a transcription of its definition, then e_delta against the
    grid inversion of erasure_failure_probability.
    """
    n, w = 10, 2
    codes = [StabilizerCode(g) for g in (TEN_FOUR, FOUR_FIVE_FROZEN, FIVE_FIVE)]
    assert [c.num_logical for c in codes] == [4, 3, 2]
    first = np.stack([c.first_failure_sizes(300, seed=i) for i, c in enumerate(codes)])
    counts = np.stack([c.uncorrectable_counts(w) for c in codes])
    assert counts.tolist() == [[1, 15], [0, 6], [0, 0]]
    assert np.mean(first <= w) > 0.1  # the junction is exercised
    grid = np.linspace(0.0, 1.0, 10001)
    phi = analysis.erasure_failure_probability(first, n, grid, exact_counts=counts).mean(axis=0)
    # mean Phi_C(e) = sum_{m <= w} b_m(e) mean P_C(m) + mean Pr[Bin(n, e) >= max(D, w + 1)]
    m = np.arange(n + 1)
    binom = np.array([math.comb(n, k) for k in m], dtype=np.float64)
    b = binom * grid[:, None] ** m * (1 - grid[:, None]) ** (n - m)  # (grid, n + 1)
    at_least = np.zeros((grid.size, n + 2))  # Pr[Bin >= s] for s = 0..n+1
    for s in range(n + 1):
        at_least[:, s] = b[:, s:].sum(axis=1)
    starts = np.bincount(np.maximum(first, w + 1).ravel(), minlength=n + 2) / first.size
    exact_mean = (counts / binom[1 : w + 1]).mean(axis=0)
    np.testing.assert_allclose(phi, b[:, 1 : w + 1] @ exact_mean + at_least @ starts, atol=1e-12)
    for delta in (0.01, 0.05, 0.2, 0.6):
        e = analysis.erasure_tolerance(first, n, delta, exact_counts=counts)
        i = int(np.flatnonzero(phi <= delta)[-1])  # last grid point with Phi <= delta
        assert grid[i] - 1e-9 <= e <= grid[i + 1], delta
