"""Tests for measurement-candidate selection."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.circuits.measurements import (
    MEASUREMENT_MODES,
    sample_measurements,
    sample_measurements_detailed,
)


def test_modes_constant():
    assert MEASUREMENT_MODES == ("bernoulli", "gated", "random_pair", "uniform_count")


# ---- candidates vs fired (sample_measurements_detailed) ----


def test_detailed_candidates_by_mode():
    rng = np.random.default_rng(0)
    cand, _ = sample_measurements_detailed("bernoulli", 8, 0.5, [], rng)
    assert cand == list(range(8))  # every qubit is a candidate
    cand, _ = sample_measurements_detailed("gated", 8, 0.5, [(0, 1), (4, 5)], rng)
    assert cand == [0, 1, 4, 5]  # only gate-touched qubits
    cand, _ = sample_measurements_detailed("random_pair", 8, 0.5, [], rng)
    assert len(cand) == 2 and cand == sorted(cand)  # exactly 2 random qubits


def test_detailed_fired_is_subset_of_candidates():
    rng = np.random.default_rng(1)
    for mode in MEASUREMENT_MODES:
        for _ in range(20):
            cand, fired = sample_measurements_detailed(mode, 8, 0.4, [(0, 1), (2, 3)], rng)
            assert set(fired) <= set(cand)
            assert fired == sorted(fired)


def test_detailed_p_extremes():
    # p=0 -> nothing fires (but candidates still listed); p=1 -> all fire
    cand0, fired0 = sample_measurements_detailed("bernoulli", 8, 0.0, [], np.random.default_rng(0))
    assert cand0 == list(range(8)) and fired0 == []
    cand1, fired1 = sample_measurements_detailed("bernoulli", 8, 1.0, [], np.random.default_rng(0))
    assert fired1 == cand1 == list(range(8))


def test_sample_measurements_equals_detailed_fired_same_rng():
    # the wrapper must consume the RNG identically to the detailed version
    for mode in MEASUREMENT_MODES:
        a, b = np.random.default_rng(7), np.random.default_rng(7)
        fired = sample_measurements(mode, 8, 0.4, [(0, 1), (2, 3)], a)
        _, fired2 = sample_measurements_detailed(mode, 8, 0.4, [(0, 1), (2, 3)], b)
        assert fired == fired2
        # and the RNG is left in the same state
        assert a.bit_generator.state == b.bit_generator.state


def test_bernoulli_p0_and_p1():
    rng = np.random.default_rng(0)
    assert sample_measurements("bernoulli", 8, 0.0, [], rng) == []
    assert sample_measurements("bernoulli", 8, 1.0, [], rng) == list(range(8))


def test_bernoulli_sorted_unique():
    rng = np.random.default_rng(1)
    out = sample_measurements("bernoulli", 16, 0.5, [], rng)
    assert out == sorted(set(out))
    assert all(0 <= q < 16 for q in out)


def test_bernoulli_mean_close_to_np():
    rng = np.random.default_rng(2)
    n, p, trials = 32, 0.25, 400
    total = sum(len(sample_measurements("bernoulli", n, p, [], rng)) for _ in range(trials))
    mean = total / trials
    assert abs(mean - n * p) < 1.0  # ~8 expected; loose tolerance


def test_gated_only_touches_gate_qubits():
    rng = np.random.default_rng(3)
    pairs = [(0, 1), (4, 5)]
    out = sample_measurements("gated", 8, 1.0, pairs, rng)
    assert out == [0, 1, 4, 5]


def test_gated_empty_when_no_gates():
    rng = np.random.default_rng(4)
    assert sample_measurements("gated", 8, 1.0, [], rng) == []


def test_random_pair_at_most_two():
    rng = np.random.default_rng(5)
    for _ in range(20):
        out = sample_measurements("random_pair", 8, 1.0, [], rng)
        assert len(out) == 2  # p=1 -> both kept
        assert out == sorted(set(out))


def test_random_pair_needs_two_qubits():
    rng = np.random.default_rng(6)
    assert sample_measurements("random_pair", 1, 1.0, [], rng) == []


def test_invalid_mode_and_p():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError):
        sample_measurements("nope", 8, 0.1, [], rng)
    with pytest.raises(ValueError):
        sample_measurements("bernoulli", 8, 1.5, [], rng)
