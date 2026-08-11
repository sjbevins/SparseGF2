"""Exact parity tests for the runner's batched Numba fast path."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2 import SparseGF2, random_symplectic
from sparsegf2.core.observables import (
    _tripartite_mutual_info_and_entropy_ab,
    entanglement_entropy,
    mutual_information,
    tripartite_mutual_info,
)
from sparsegf2.core.sparse_tableau import _build_2q_lut


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("use_numba", [False, True])
def test_batched_gates_and_measurements_match_scalar_byte_for_byte(hybrid, use_numba):
    n = 24
    schedule_rng = np.random.default_rng(8181)
    scalar = SparseGF2(n, rng=np.random.default_rng(9191), hybrid=hybrid, use_numba=use_numba)
    batched = SparseGF2(n, rng=np.random.default_rng(9191), hybrid=hybrid, use_numba=use_numba)
    # Make mode-switch boundaries cut through both gate and measurement batches.
    scalar._check_interval = 7
    batched._check_interval = 7
    luts = np.ascontiguousarray(
        [
            _build_2q_lut(
                random_symplectic(2, np.random.default_rng(int(schedule_rng.integers(1 << 30))))
            )
            for _ in range(16)
        ],
        dtype=np.uint8,
    )

    for _ in range(18):
        pairs = np.empty((12, 2), dtype=np.int64)
        for g in range(pairs.shape[0]):
            pairs[g] = schedule_rng.choice(n, size=2, replace=False)
        cliff_indices = schedule_rng.integers(0, len(luts), size=len(pairs), dtype=np.int64)
        meas_qubits = np.flatnonzero(schedule_rng.random(n) < 0.3).astype(np.int64)

        for g, (qi, qj) in enumerate(pairs):
            scalar._dispatch_2q(int(qi), int(qj), luts[cliff_indices[g]])
        scalar_outcomes = np.asarray(
            [scalar.measure_z(int(q)) for q in meas_qubits], dtype=np.uint8
        )

        batched._dispatch_2q_batch(pairs, cliff_indices, luts)
        batch_outcomes = batched._measure_z_batch(meas_qubits)

        assert np.array_equal(batch_outcomes, scalar_outcomes)
        assert np.array_equal(batched.to_symplectic(), scalar.to_symplectic())
        assert batched._dense_mode == scalar._dense_mode
        assert batched._ops_since_check == scalar._ops_since_check
        assert batched.active_count() == scalar.active_count()

    # Delayed batch outcome draws must leave the generator at the identical point.
    assert int(batched._rng.integers(1 << 62)) == int(scalar._rng.integers(1 << 62))


def test_shared_tmi_helper_matches_three_mutual_informations():
    n = 12
    rng = np.random.default_rng(7171)
    sim = SparseGF2(n, hybrid=True)
    for _ in range(8 * n):
        qi, qj = (int(x) for x in rng.choice(n, size=2, replace=False))
        sim.apply_gate_2q(qi, qj, random_symplectic(2, rng))
        if rng.random() < 0.25:
            sim.measure_z(int(rng.integers(n)), rng=rng)

    # ABC is larger than half the system, exercising S(ABC) = S(complement).
    A, B, C = range(0, 3), range(3, 6), range(6, 10)
    expected_i3 = (
        mutual_information(sim, A, B)
        + mutual_information(sim, A, C)
        - mutual_information(sim, A, list(B) + list(C))
    )
    expected_sab = entanglement_entropy(sim, list(A) + list(B))
    got_i3, got_sab = _tripartite_mutual_info_and_entropy_ab(sim, A, B, C)
    assert got_i3 == expected_i3 == tripartite_mutual_info(sim, A, B, C)
    assert got_sab == expected_sab
