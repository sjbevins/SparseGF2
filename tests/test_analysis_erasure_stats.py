"""mu_D, distance bounds and the hybrid P_C(m) curve (analysis.py)."""

import numpy as np
import pytest

from sparsegf2 import StabilizerCode, analysis


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
