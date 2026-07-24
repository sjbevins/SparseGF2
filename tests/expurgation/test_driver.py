"""The expurgation loop: worked examples end to end, monotonicity, stopping."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.errors import InvalidArgumentError
from sparsegf2.expurgation import (
    ExpurgationConfig,
    expurgate,
    expurgation_candidates,
    mean_recovery,
    random_encoding,
    uncorrectable_rank,
)

# ----------------------------------------------------------------------
# Worked example B: one measurement buys a distance increase
# ----------------------------------------------------------------------


def test_example_b_distance_increase(example_b_code, distance):
    code = example_b_code
    assert code.k == 2
    assert distance(code) == 1
    erased = np.array([0, 1])
    assert uncorrectable_rank(code, erased) == 3
    cands = expurgation_candidates(code, erased)
    assert len(cands) == 3
    lightest = cands[0]
    assert lightest.qubits == (0, 1)
    assert lightest.letters == (3, 3)  # Y_0 Y_1 = YYII
    pair = code.measure(lightest.qubits, lightest.letters, strategy="stabilizer")
    assert pair is not None
    assert code.k == 1
    assert distance(code) == 2


# ----------------------------------------------------------------------
# Worked example A chain: the 1D miniature (expurgation runs to k = 0)
# ----------------------------------------------------------------------


def test_example_a_chain_dies(example_a_code):
    result = expurgate(
        example_a_code,
        ExpurgationConfig(
            strategy="stabilizer",
            erasure_count=1,
            max_rounds=50,
            validation_patterns=16,
            seed=0,
        ),
    )
    assert result.stop_reason == "k_zero"
    assert result.k_initial == 2
    assert result.k_final == 0
    # A k = 0 code has nothing left to fail on: recovery is trivially 1.
    assert result.recovery_after == 1.0
    assert result.recovery_after >= result.recovery_before


# ----------------------------------------------------------------------
# Random all-to-all codes: accounting and monotone recovery
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(3))
def test_loop_accounting_and_monotonicity(seed):
    code = random_encoding(
        16, range(8, 16), 4, geometry="all_to_all", rng=np.random.default_rng(seed)
    )
    k0 = code.k
    result = expurgate(
        code,
        ExpurgationConfig(
            erasure_count=2,
            k_target=4,
            max_rounds=40,
            validation_patterns=24,
            seed=seed,
        ),
    )
    # k falls by exactly the number of measurements performed.
    assert result.k_initial == k0
    assert result.k_final == code.k == k0 - len(result.measured_pairs)
    assert len(result.measured_pairs) == len(result.measured_weights)
    assert len(result.measured_pairs) == len(result.measured_rounds)
    assert len(result.k_trajectory) == result.rounds
    # Gauge strategy: original checks untouched, spent pairs become gauge.
    assert len(code.check_pairs()) == 16 - k0
    assert len(code.gauge_pairs()) == len(result.measured_pairs)
    # The frozen-pattern recovery metric never decreases (Prop. 2).
    assert result.recovery_after >= result.recovery_before
    if result.stop_reason == "k_target":
        assert result.k_final <= 4


def test_stabilizer_strategy_grows_checks():
    code = random_encoding(12, range(6, 12), 4, geometry="all_to_all", rng=np.random.default_rng(7))
    result = expurgate(
        code,
        ExpurgationConfig(
            strategy="stabilizer",
            erasure_count=2,
            k_target=3,
            max_rounds=30,
            validation_patterns=0,
            seed=7,
        ),
    )
    assert result.recovery_before is None
    assert result.recovery_after is None
    assert len(code.check_pairs()) == 6 + len(result.measured_pairs)
    assert len(code.gauge_pairs()) == 0


def test_recovery_target_stops_early():
    code = random_encoding(
        12, range(6, 12), 4, geometry="all_to_all", rng=np.random.default_rng(11)
    )
    result = expurgate(
        code,
        ExpurgationConfig(
            erasure_count=1,
            recovery_target=0.9,
            max_rounds=60,
            validation_patterns=16,
            seed=11,
        ),
    )
    if result.stop_reason == "recovery_target":
        assert result.recovery_after >= 0.9


def test_mean_recovery_bounds(example_a_code):
    patterns = [np.array([3]), np.array([2, 3])]
    m = mean_recovery(example_a_code, patterns)
    assert 0.0 < m <= 1.0
    assert mean_recovery(example_a_code, []) == 1.0


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------


def test_config_validation(example_a_code):
    with pytest.raises(InvalidArgumentError):
        expurgate(example_a_code, ExpurgationConfig())  # no erasure model
    with pytest.raises(InvalidArgumentError):
        expurgate(example_a_code, ExpurgationConfig(erasure_rate=0.1, erasure_count=1))
    with pytest.raises(InvalidArgumentError):
        expurgate(example_a_code, ExpurgationConfig(erasure_count=1, strategy="bogus"))
    with pytest.raises(InvalidArgumentError):
        expurgate(example_a_code, ExpurgationConfig(erasure_count=1, k_target=-1))
    with pytest.raises(InvalidArgumentError):
        expurgate(
            example_a_code,
            ExpurgationConfig(erasure_count=1, recovery_target=0.9, validation_patterns=0),
        )
