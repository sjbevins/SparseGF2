"""Tests for sparsegf2.analysis.weight_stats module.

Covers: identity invariants, variance decomposition, consistency oracle,
histograms, stabilizer/destabilizer decomposition, X-weight consistency,
PEI/WCP proxies, and the observe() flat dict API.
"""
import numpy as np
import pytest

from sparsegf2.core.sparse_tableau import SparseGF2
from sparsegf2.analysis.weight_stats import (
    compute_weight_stats,
    verify_weight_mass_identity,
    compute_pivot_effectiveness,
)
from sparsegf2.analysis.observables import observe

# Symplectic matrices for CNOT and CZ gates
CNOT = np.array([[1, 1, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 1, 1]], dtype=np.uint8)

CZ = np.array([[1, 0, 0, 1],
               [0, 1, 1, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]], dtype=np.uint8)


# ── A1: Identity Test — Initial State ────────────────────────────

class TestA1InitialState:
    def test_initial_state_n8(self):
        sim = SparseGF2(8)
        stats = compute_weight_stats(sim)

        assert stats.abar == 2.0
        assert stats.wbar == 1.0
        assert stats.weight_mass == 16
        assert stats.var_a == 0.0
        assert stats.var_w == 0.0
        assert stats.identity_holds is True
        assert stats.wbar_stab == 1.0
        assert stats.wbar_destab == 1.0


# ── A2: Identity Test — After Gates ──────────────────────────────

class TestA2AfterGates:
    def test_cnot_then_cz_n4(self):
        sim = SparseGF2(4)

        # After CNOT(0,1)
        sim.apply_gate(0, 1, CNOT)
        stats1 = compute_weight_stats(sim)
        assert stats1.weight_mass == 10
        assert stats1.abar == 2.50

        # After CZ(2,3)
        sim.apply_gate(2, 3, CZ)
        stats2 = compute_weight_stats(sim)
        assert stats2.weight_mass == 12
        assert stats2.abar == 3.00

    def test_duality_identity_integer_exact(self):
        """Verify n*abar == 2n*wbar at each step (integer-exact on weight_mass)."""
        sim = SparseGF2(4)
        n = sim.n

        sim.apply_gate(0, 1, CNOT)
        stats1 = compute_weight_stats(sim)
        # weight_mass = n*abar = 2n*wbar (all integer)
        assert stats1.weight_mass == int(n * stats1.abar)
        assert stats1.weight_mass == int(2 * n * stats1.wbar)

        sim.apply_gate(2, 3, CZ)
        stats2 = compute_weight_stats(sim)
        assert stats2.weight_mass == int(n * stats2.abar)
        assert stats2.weight_mass == int(2 * n * stats2.wbar)


# ── A3: Identity Test — After Measurement ────────────────────────

class TestA3AfterMeasurement:
    def _build_n6_state(self):
        """n=6, apply 5 CNOTs: (0,1), (1,2), (3,4), (4,5), (2,3)."""
        sim = SparseGF2(6)
        sim.apply_gate(0, 1, CNOT)
        sim.apply_gate(1, 2, CNOT)
        sim.apply_gate(3, 4, CNOT)
        sim.apply_gate(4, 5, CNOT)
        sim.apply_gate(2, 3, CNOT)
        return sim

    def test_before_measurement(self):
        sim = self._build_n6_state()
        stats = compute_weight_stats(sim)
        # 6 * 4.5 = 27
        assert stats.weight_mass == 27

    def test_after_measurement(self):
        sim = self._build_n6_state()
        sim.apply_measurement_z(2)
        stats = compute_weight_stats(sim)
        # After measurement: weight_mass == 22
        assert stats.weight_mass == 22

    def test_identity_holds_after_measurement(self):
        sim = self._build_n6_state()
        sim.apply_measurement_z(2)
        stats = compute_weight_stats(sim)
        assert stats.identity_holds is True

    def test_stab_destab_may_differ_after_measurement(self):
        sim = self._build_n6_state()
        sim.apply_measurement_z(2)
        stats = compute_weight_stats(sim)
        # wbar_stab and wbar_destab are computed; just check they are valid
        assert stats.wbar_stab >= 0.0
        assert stats.wbar_destab >= 0.0
        # They sum correctly (tested in A7), but may differ from each other
        # (we don't assert equality between them)


# ── A4: Variance Decomposition Test ──────────────────────────────

class TestA4VarianceDecomposition:
    def test_variance_nonnegative_under_random_gates(self):
        sim = SparseGF2(16)
        rng = np.random.default_rng(42)

        for _ in range(100):
            qi, qj = rng.choice(16, size=2, replace=False)
            sim.apply_gate(int(qi), int(qj), CNOT)

            stats = compute_weight_stats(sim)
            assert stats.var_a >= 0.0, f"var_a negative: {stats.var_a}"
            assert stats.var_w >= 0.0, f"var_w negative: {stats.var_w}"
            assert stats.cv_a >= 0.0, f"cv_a negative: {stats.cv_a}"
            assert stats.cv_w >= 0.0, f"cv_w negative: {stats.cv_w}"


# ── A5: Consistency Oracle Test ──────────────────────────────────

class TestA5ConsistencyOracle:
    def test_corrupted_inv_len_raises(self):
        sim = SparseGF2(8)
        sim.apply_gate(0, 1, CNOT)
        sim.apply_gate(2, 3, CZ)

        original = int(sim.inv_len[0])
        sim.inv_len[0] += 1

        with pytest.raises(AssertionError):
            verify_weight_mass_identity(sim, raise_on_failure=True)

        sim.inv_len[0] = original

    def test_corrupted_inv_len_returns_false(self):
        sim = SparseGF2(8)
        sim.apply_gate(0, 1, CNOT)
        sim.apply_gate(2, 3, CZ)

        original = int(sim.inv_len[0])
        sim.inv_len[0] += 1

        result = verify_weight_mass_identity(sim, raise_on_failure=False)
        assert result['inv_pass'] is False

        sim.inv_len[0] = original

    def test_uncorrupted_passes(self):
        sim = SparseGF2(8)
        sim.apply_gate(0, 1, CNOT)
        sim.apply_gate(2, 3, CZ)

        result = verify_weight_mass_identity(sim, raise_on_failure=True)
        assert result['inv_pass'] is True
        assert result['x_pass'] is True


# ── A6: Histogram Test ───────────────────────────────────────────

class TestA6Histogram:
    def test_histograms_sum_to_one_initial(self):
        sim = SparseGF2(8)
        stats = compute_weight_stats(sim)

        assert abs(np.sum(stats.hist_a) - 1.0) < 1e-10
        assert abs(np.sum(stats.hist_w) - 1.0) < 1e-10
        assert np.all(stats.hist_a >= 0)
        assert np.all(stats.hist_w >= 0)

    def test_histograms_sum_to_one_after_gates(self):
        sim = SparseGF2(8)
        for i in range(7):
            sim.apply_gate(i, (i + 1) % 8, CNOT)

        stats = compute_weight_stats(sim)

        assert abs(np.sum(stats.hist_a) - 1.0) < 1e-10
        assert abs(np.sum(stats.hist_w) - 1.0) < 1e-10
        assert np.all(stats.hist_a >= 0)
        assert np.all(stats.hist_w >= 0)


# ── A7: Stabilizer/Destabilizer Decomposition Test ───────────────

class TestA7StabDestabDecomposition:
    def test_decomposition_identity(self):
        """Verify abar == wbar_stab + wbar_destab and 2*wbar == wbar_stab + wbar_destab."""
        sim = SparseGF2(8)

        # Check initial state
        stats = compute_weight_stats(sim)
        assert abs(stats.abar - (stats.wbar_stab + stats.wbar_destab)) < 1e-10
        assert abs(2 * stats.wbar - (stats.wbar_stab + stats.wbar_destab)) < 1e-10

        # Check after several gates
        sim.apply_gate(0, 1, CNOT)
        sim.apply_gate(2, 3, CZ)
        sim.apply_gate(4, 5, CNOT)
        sim.apply_gate(1, 3, CZ)
        stats = compute_weight_stats(sim)
        assert abs(stats.abar - (stats.wbar_stab + stats.wbar_destab)) < 1e-10
        assert abs(2 * stats.wbar - (stats.wbar_stab + stats.wbar_destab)) < 1e-10

        # Check after measurement
        sim.apply_measurement_z(3)
        stats = compute_weight_stats(sim)
        assert abs(stats.abar - (stats.wbar_stab + stats.wbar_destab)) < 1e-10
        assert abs(2 * stats.wbar - (stats.wbar_stab + stats.wbar_destab)) < 1e-10

    def test_decomposition_under_random_gates(self):
        """Verify decomposition holds across many random operations."""
        sim = SparseGF2(12)
        rng = np.random.default_rng(99)

        for step in range(50):
            qi, qj = rng.choice(12, size=2, replace=False)
            sim.apply_gate(int(qi), int(qj), CNOT)

            stats = compute_weight_stats(sim)
            assert abs(stats.abar - (stats.wbar_stab + stats.wbar_destab)) < 1e-10, \
                f"Step {step}: abar={stats.abar}, stab+destab={stats.wbar_stab + stats.wbar_destab}"
            assert abs(2 * stats.wbar - (stats.wbar_stab + stats.wbar_destab)) < 1e-10, \
                f"Step {step}: 2*wbar={2*stats.wbar}, stab+destab={stats.wbar_stab + stats.wbar_destab}"


# ── A8: X-Weight Consistency Test ────────────────────────────────

class TestA8XWeightConsistency:
    def test_x_weight_initial_state(self):
        """Initial state: each qubit has 1 X-type generator (destabilizer), total_x = n."""
        n = 8
        sim = SparseGF2(n)
        result = verify_weight_mass_identity(sim, raise_on_failure=True)
        assert result['x_pass'] is True
        # total_x_inv = sum_q inv_x_len[q] = n (one destab X_i per qubit i)
        assert result['total_x_inv'] == n
        assert result['total_x_supp'] == n

    def test_x_weight_after_gates(self):
        """X-weight identity holds after gates."""
        sim = SparseGF2(8)
        sim.apply_gate(0, 1, CNOT)
        sim.apply_gate(2, 3, CZ)
        sim.apply_gate(4, 5, CNOT)

        result = verify_weight_mass_identity(sim, raise_on_failure=True)
        assert result['x_pass'] is True
        assert result['total_x_inv'] == result['total_x_supp']


# ── A9: PEI Proxy Test ──────────────────────────────────────────

class TestA9PEIProxy:
    def test_pei_leq_wcp_always(self):
        """PEI proxy <= WCP proxy always."""
        sim = SparseGF2(8)
        pei = compute_pivot_effectiveness(sim)
        assert pei['pei_proxy'] <= pei['wcp_proxy']

    def test_pei_eq_wcp_initial(self):
        """At initial state, each qubit has exactly 1 anticommuting generator,
        so min == max, hence pei_proxy == wcp_proxy."""
        sim = SparseGF2(8)
        pei = compute_pivot_effectiveness(sim)
        assert abs(pei['pei_proxy'] - pei['wcp_proxy']) < 1e-10

    def test_pei_lt_wcp_after_gates(self):
        """After gates create variance, pei_proxy < wcp_proxy (typically)."""
        sim = SparseGF2(16)
        rng = np.random.default_rng(7)

        # Apply many gates to create variance in generator weights
        for _ in range(50):
            qi, qj = rng.choice(16, size=2, replace=False)
            sim.apply_gate(int(qi), int(qj), CNOT)

        pei = compute_pivot_effectiveness(sim)
        # After many gates, we expect min != max for at least some qubits
        assert pei['pei_proxy'] <= pei['wcp_proxy']
        # With 50 random CNOTs on 16 qubits, variance should appear
        assert pei['wcp_proxy'] > pei['pei_proxy'], \
            "Expected pei_proxy < wcp_proxy after 50 random gates"


# ── A10: observe() Flat Dict Test ────────────────────────────────

class TestA10ObserveFlatDict:
    def test_returns_dict(self):
        sim = SparseGF2(8)
        d = observe(sim)
        assert isinstance(d, dict)

    def test_all_values_are_scalars(self):
        sim = SparseGF2(8)
        d = observe(sim)
        for key, val in d.items():
            assert isinstance(val, (int, float, bool, np.integer, np.floating, np.bool_)), \
                f"Key '{key}' has non-scalar value of type {type(val)}: {val}"

    def test_all_floats_are_finite(self):
        sim = SparseGF2(8)
        d = observe(sim)
        for key, val in d.items():
            if isinstance(val, (float, np.floating)):
                assert np.isfinite(val), f"Key '{key}' has non-finite value: {val}"

    def test_observe_after_gates(self):
        """observe() works after gate application too."""
        sim = SparseGF2(8)
        sim.apply_gate(0, 1, CNOT)
        sim.apply_gate(2, 3, CZ)

        d = observe(sim)
        assert isinstance(d, dict)
        for key, val in d.items():
            assert isinstance(val, (int, float, bool, np.integer, np.floating, np.bool_)), \
                f"Key '{key}' has non-scalar value of type {type(val)}: {val}"
            if isinstance(val, (float, np.floating)):
                assert np.isfinite(val), f"Key '{key}' has non-finite value: {val}"
