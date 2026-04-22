"""
Tests for specialized H/S/CX Numba kernels.

Verifies that the specialized kernels (_apply_h_kernel, _apply_s_kernel,
_apply_cx_kernel) produce identical stabilizer groups as the generic
gate path. Tests use random circuits with both paths and compare full
GF(2) RREF tableaux row by row.

Usage:
    py -3.13 -m pytest tests/test_surface_optimizations.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from sparsegf2.core.sparse_tableau import (
    SparseGF2, warmup, _H_SYMP, _S_SYMP, _CX_SYMP,
)


# ======================================================================
# Helpers
# ======================================================================

def gf2_rref(mat):
    """Compute Reduced Row Echelon Form over GF(2)."""
    m = mat.copy()
    nrows, ncols = m.shape
    rank = 0
    for col in range(ncols):
        pivot = -1
        for row in range(rank, nrows):
            if m[row, col]:
                pivot = row
                break
        if pivot == -1:
            continue
        if pivot != rank:
            m[[rank, pivot]] = m[[pivot, rank]]
        for row in range(nrows):
            if row != rank and m[row, col]:
                m[row] ^= m[rank]
        rank += 1
    return m[:rank]


def assert_rref_match(sim_a, sim_b, msg=""):
    """Assert that two SparseGF2 simulators have identical RREF tableaux."""
    mat_a = sim_a.extract_sys_matrix()
    mat_b = sim_b.extract_sys_matrix()
    rref_a = gf2_rref(mat_a)
    rref_b = gf2_rref(mat_b)
    assert rref_a.shape == rref_b.shape, (
        f"RREF rank mismatch: {rref_a.shape[0]} vs {rref_b.shape[0]}. {msg}"
    )
    assert np.array_equal(rref_a, rref_b), (
        f"RREF content mismatch (rank={rref_a.shape[0]}). {msg}"
    )


def clone_state(sim):
    """Create a SparseGF2 with identical state (deep copy of arrays)."""
    n = sim.n
    clone = SparseGF2(n, use_min_weight_pivot=sim.use_min_weight_pivot,
                      check_inputs=sim.check_inputs)
    clone.plt[:] = sim.plt
    clone.supp_q[:] = sim.supp_q
    clone.supp_len[:] = sim.supp_len
    clone.supp_pos[:] = sim.supp_pos
    clone.inv[:] = sim.inv
    clone.inv_len[:] = sim.inv_len
    clone.inv_x[:] = sim.inv_x
    clone.inv_x_len[:] = sim.inv_x_len
    clone.inv_pos[:] = sim.inv_pos
    clone.inv_x_pos[:] = sim.inv_x_pos
    clone.x_packed[:] = sim.x_packed
    clone.z_packed[:] = sim.z_packed
    clone._dense_mode = sim._dense_mode
    return clone


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="session", autouse=True)
def jit_warmup():
    """JIT-compile all Numba kernels before tests run."""
    warmup()


# ======================================================================
# H gate tests
# ======================================================================

class TestHadamardKernel:
    """Test _apply_h_kernel against generic apply_gate_1q path."""

    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_h_single_qubit_all_positions(self, n):
        """Apply H on each qubit individually, compare to generic."""
        for q in range(n):
            sim_gen = SparseGF2(n, check_inputs=False)
            sim_fast = clone_state(sim_gen)
            sim_gen.apply_h(q)
            sim_fast.apply_h_fast(q)
            assert_rref_match(sim_gen, sim_fast,
                              f"H on q={q}, n={n}, initial state")

    @pytest.mark.parametrize("n", [8, 16, 32])
    @pytest.mark.parametrize("seed", [42, 123, 777])
    def test_h_after_random_circuit(self, n, seed):
        """Apply random gates, then H on each qubit, compare both paths."""
        rng = np.random.default_rng(seed)
        sim = SparseGF2(n, check_inputs=False)

        # Build up entanglement with random CX gates
        for _ in range(n * 2):
            qc = rng.integers(0, n)
            qt = rng.integers(0, n - 1)
            if qt >= qc:
                qt += 1
            sim.apply_gate(qc, qt, _CX_SYMP)

        # Apply some measurements to get sparse structure
        for q in range(n):
            if rng.random() < 0.3:
                sim.apply_measurement_z(q)

        # Now test H on each qubit
        for q in range(n):
            sim_gen = clone_state(sim)
            sim_fast = clone_state(sim)
            sim_gen.apply_h(q)
            sim_fast.apply_h_fast(q)
            assert_rref_match(sim_gen, sim_fast,
                              f"H on q={q}, n={n}, seed={seed}")

    @pytest.mark.parametrize("n", [8, 16])
    def test_h_double_application_identity(self, n):
        """H^2 = I: applying H twice should return to original state."""
        rng = np.random.default_rng(99)
        sim = SparseGF2(n, check_inputs=False)
        for _ in range(n):
            qc, qt = rng.integers(0, n, size=2)
            if qc == qt:
                qt = (qt + 1) % n
            sim.apply_gate(qc, qt, _CX_SYMP)

        sim_orig = clone_state(sim)
        for q in range(n):
            sim.apply_h_fast(q)
            sim.apply_h_fast(q)
        assert_rref_match(sim_orig, sim, "H^2 = I")


# ======================================================================
# S gate tests
# ======================================================================

class TestSGateKernel:
    """Test _apply_s_kernel against generic apply_gate_1q path."""

    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_s_single_qubit_all_positions(self, n):
        """Apply S on each qubit individually, compare to generic."""
        for q in range(n):
            sim_gen = SparseGF2(n, check_inputs=False)
            sim_fast = clone_state(sim_gen)
            sim_gen.apply_s(q)
            sim_fast.apply_s_fast(q)
            assert_rref_match(sim_gen, sim_fast,
                              f"S on q={q}, n={n}, initial state")

    @pytest.mark.parametrize("n", [8, 16, 32])
    @pytest.mark.parametrize("seed", [42, 123, 777])
    def test_s_after_random_circuit(self, n, seed):
        """Apply random gates, then S on each qubit, compare both paths."""
        rng = np.random.default_rng(seed)
        sim = SparseGF2(n, check_inputs=False)

        for _ in range(n * 2):
            qc = rng.integers(0, n)
            qt = rng.integers(0, n - 1)
            if qt >= qc:
                qt += 1
            sim.apply_gate(qc, qt, _CX_SYMP)

        for q in range(n):
            if rng.random() < 0.3:
                sim.apply_measurement_z(q)

        for q in range(n):
            sim_gen = clone_state(sim)
            sim_fast = clone_state(sim)
            sim_gen.apply_s(q)
            sim_fast.apply_s_fast(q)
            assert_rref_match(sim_gen, sim_fast,
                              f"S on q={q}, n={n}, seed={seed}")


# ======================================================================
# CX gate tests
# ======================================================================

class TestCXKernel:
    """Test _apply_cx_kernel against generic _apply_gate_kernel path."""

    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_cx_all_pairs_initial(self, n):
        """Apply CX on every qubit pair from initial state."""
        for qc in range(min(n, 6)):
            for qt in range(min(n, 6)):
                if qc == qt:
                    continue
                sim_gen = SparseGF2(n, check_inputs=False)
                sim_fast = clone_state(sim_gen)
                sim_gen.apply_gate(qc, qt, _CX_SYMP)
                sim_fast.apply_cx_fast(qc, qt)
                assert_rref_match(sim_gen, sim_fast,
                                  f"CX({qc},{qt}), n={n}, initial state")

    @pytest.mark.parametrize("n", [8, 16, 32])
    @pytest.mark.parametrize("seed", [42, 123, 777])
    def test_cx_after_random_circuit(self, n, seed):
        """Apply random gates + measurements, then CX on pairs, compare."""
        rng = np.random.default_rng(seed)
        sim = SparseGF2(n, check_inputs=False)

        # Build entangled state
        for _ in range(n * 2):
            qc = rng.integers(0, n)
            qt = rng.integers(0, n - 1)
            if qt >= qc:
                qt += 1
            sim.apply_gate(qc, qt, _CX_SYMP)

        # Measure some qubits
        for q in range(n):
            if rng.random() < 0.3:
                sim.apply_measurement_z(q)

        # Test CX on random pairs
        n_tests = min(n * 2, 20)
        for _ in range(n_tests):
            qc = rng.integers(0, n)
            qt = rng.integers(0, n - 1)
            if qt >= qc:
                qt += 1
            sim_gen = clone_state(sim)
            sim_fast = clone_state(sim)
            sim_gen.apply_gate(qc, qt, _CX_SYMP)
            sim_fast.apply_cx_fast(qc, qt)
            assert_rref_match(sim_gen, sim_fast,
                              f"CX({qc},{qt}), n={n}, seed={seed}")

    @pytest.mark.parametrize("n", [8, 16])
    @pytest.mark.parametrize("seed", [42, 200])
    def test_cx_chain_sequence(self, n, seed):
        """Apply a long chain of CX gates, comparing at each step."""
        rng = np.random.default_rng(seed)
        sim_gen = SparseGF2(n, check_inputs=False)
        sim_fast = clone_state(sim_gen)

        for step in range(n * 4):
            qc = rng.integers(0, n)
            qt = rng.integers(0, n - 1)
            if qt >= qc:
                qt += 1
            sim_gen.apply_gate(qc, qt, _CX_SYMP)
            sim_fast.apply_cx_fast(qc, qt)

            if step % n == 0:
                assert_rref_match(sim_gen, sim_fast,
                                  f"CX chain step {step}, n={n}, seed={seed}")

        assert_rref_match(sim_gen, sim_fast,
                          f"CX chain final, n={n}, seed={seed}")


# ======================================================================
# Mixed gate tests (H + S + CX together)
# ======================================================================

class TestMixedGates:
    """Test sequences mixing H, S, and CX using specialized kernels."""

    @pytest.mark.parametrize("n", [8, 16, 32])
    @pytest.mark.parametrize("seed", [42, 99, 314])
    def test_random_hscx_circuit(self, n, seed):
        """Random circuit using only H, S, CX -- compare generic vs fast."""
        rng = np.random.default_rng(seed)
        sim_gen = SparseGF2(n, check_inputs=False)
        sim_fast = clone_state(sim_gen)

        n_ops = n * 6
        for _ in range(n_ops):
            gate_type = rng.integers(0, 3)
            if gate_type == 0:
                # H gate
                q = rng.integers(0, n)
                sim_gen.apply_h(q)
                sim_fast.apply_h_fast(q)
            elif gate_type == 1:
                # S gate
                q = rng.integers(0, n)
                sim_gen.apply_s(q)
                sim_fast.apply_s_fast(q)
            else:
                # CX gate
                qc = rng.integers(0, n)
                qt = rng.integers(0, n - 1)
                if qt >= qc:
                    qt += 1
                sim_gen.apply_gate(qc, qt, _CX_SYMP)
                sim_fast.apply_cx_fast(qc, qt)

        assert_rref_match(sim_gen, sim_fast,
                          f"Random H/S/CX circuit, n={n}, seed={seed}")

    @pytest.mark.parametrize("n", [8, 16, 32])
    @pytest.mark.parametrize("seed", [42, 99])
    def test_hscx_with_measurements(self, n, seed):
        """H/S/CX circuit with interleaved Z measurements."""
        rng = np.random.default_rng(seed)
        sim_gen = SparseGF2(n, check_inputs=False)
        sim_fast = clone_state(sim_gen)

        n_ops = n * 8
        for i in range(n_ops):
            gate_type = rng.integers(0, 4)
            if gate_type == 0:
                q = rng.integers(0, n)
                sim_gen.apply_h(q)
                sim_fast.apply_h_fast(q)
            elif gate_type == 1:
                q = rng.integers(0, n)
                sim_gen.apply_s(q)
                sim_fast.apply_s_fast(q)
            elif gate_type == 2:
                qc = rng.integers(0, n)
                qt = rng.integers(0, n - 1)
                if qt >= qc:
                    qt += 1
                sim_gen.apply_gate(qc, qt, _CX_SYMP)
                sim_fast.apply_cx_fast(qc, qt)
            else:
                q = rng.integers(0, n)
                sim_gen.apply_measurement_z(q)
                sim_fast.apply_measurement_z(q)

            # Periodic RREF checks
            if i % (n * 2) == 0:
                assert_rref_match(sim_gen, sim_fast,
                                  f"Mixed+meas step {i}, n={n}, seed={seed}")

        assert_rref_match(sim_gen, sim_fast,
                          f"Mixed+meas final, n={n}, seed={seed}")


# ======================================================================
# Surface code circuit test
# ======================================================================

class TestSurfaceCodeCircuit:
    """Test with a surface code circuit structure (H, CX, M only)."""

    def _build_surface_code_round(self, n, sim_gen, sim_fast, rng):
        """Simulate one round of a surface-code-like circuit.

        Applies H gates, CX gates in a brickwork pattern, and measurements.
        """
        # H on all qubits
        for q in range(n):
            sim_gen.apply_h(q)
            sim_fast.apply_h_fast(q)

        # CX brickwork layer (even pairs)
        for i in range(0, n - 1, 2):
            sim_gen.apply_gate(i, i + 1, _CX_SYMP)
            sim_fast.apply_cx_fast(i, i + 1)

        # CX brickwork layer (odd pairs)
        for i in range(1, n - 1, 2):
            sim_gen.apply_gate(i, i + 1, _CX_SYMP)
            sim_fast.apply_cx_fast(i, i + 1)

        # Measure roughly half the qubits
        for q in range(n):
            if rng.random() < 0.5:
                sim_gen.apply_measurement_z(q)
                sim_fast.apply_measurement_z(q)

    @pytest.mark.parametrize("n", [8, 16, 32, 64])
    @pytest.mark.parametrize("seed", [42, 99])
    def test_surface_code_rounds(self, n, seed):
        """Multiple rounds of surface-code-like circuit."""
        rng = np.random.default_rng(seed)
        sim_gen = SparseGF2(n, check_inputs=False)
        sim_fast = clone_state(sim_gen)

        n_rounds = 8
        for r in range(n_rounds):
            self._build_surface_code_round(n, sim_gen, sim_fast, rng)
            assert_rref_match(sim_gen, sim_fast,
                              f"Surface code round {r}, n={n}, seed={seed}")

    @pytest.mark.parametrize("n", [16, 32])
    @pytest.mark.parametrize("seed", [42])
    def test_deep_surface_code(self, n, seed):
        """Deep surface-code-like circuit (many rounds)."""
        rng = np.random.default_rng(seed)
        sim_gen = SparseGF2(n, check_inputs=False)
        sim_fast = clone_state(sim_gen)

        n_rounds = 32
        for r in range(n_rounds):
            self._build_surface_code_round(n, sim_gen, sim_fast, rng)

        assert_rref_match(sim_gen, sim_fast,
                          f"Deep surface code, n={n}, seed={seed}")


# ======================================================================
# Edge case tests
# ======================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions for specialized kernels."""

    def test_h_on_identity_qubit(self):
        """H on a qubit with no generators (all I)."""
        n = 8
        sim = SparseGF2(n, check_inputs=False)
        # Measure qubit 3 to clear its generators
        sim.apply_measurement_z(3)
        sim_gen = clone_state(sim)
        sim_fast = clone_state(sim)
        sim_gen.apply_h(3)
        sim_fast.apply_h_fast(3)
        assert_rref_match(sim_gen, sim_fast, "H on measured qubit")

    def test_s_on_z_only_qubit(self):
        """S on a qubit where all generators have Z-only support (no X)."""
        n = 8
        sim = SparseGF2(n, check_inputs=False)
        # Initial state: qubit 0 has X (destabilizer) and Z (stabilizer)
        # After measurement, only Z remains
        sim.apply_measurement_z(0)
        sim_gen = clone_state(sim)
        sim_fast = clone_state(sim)
        sim_gen.apply_s(0)
        sim_fast.apply_s_fast(0)
        assert_rref_match(sim_gen, sim_fast, "S on Z-only qubit")

    def test_cx_both_qubits_identity(self):
        """CX where neither qubit has support from any generator."""
        n = 8
        sim = SparseGF2(n, check_inputs=False)
        sim.apply_measurement_z(2)
        sim.apply_measurement_z(3)
        sim_gen = clone_state(sim)
        sim_fast = clone_state(sim)
        sim_gen.apply_gate(2, 3, _CX_SYMP)
        sim_fast.apply_cx_fast(2, 3)
        assert_rref_match(sim_gen, sim_fast, "CX on two measured qubits")

    def test_cx_creates_support(self):
        """CX that creates new support on target qubit."""
        n = 8
        sim = SparseGF2(n, check_inputs=False)
        # Qubit 0 has X support (destabilizer), qubit 1 has X support
        # CX(0,1): x_1' = x_1 XOR x_0. The X generator on qubit 0
        # will spread its X to qubit 1.
        sim_gen = clone_state(sim)
        sim_fast = clone_state(sim)
        sim_gen.apply_gate(0, 1, _CX_SYMP)
        sim_fast.apply_cx_fast(0, 1)
        assert_rref_match(sim_gen, sim_fast, "CX creates support on target")

    def test_cx_removes_support(self):
        """CX that removes support from a qubit."""
        n = 8
        sim = SparseGF2(n, check_inputs=False)
        # First create shared X support via CX
        sim.apply_gate(0, 1, _CX_SYMP)
        # Now CX(0,1) again should undo: x_1' = x_1 XOR x_0
        # If both had X, XOR cancels
        sim_gen = clone_state(sim)
        sim_fast = clone_state(sim)
        sim_gen.apply_gate(0, 1, _CX_SYMP)
        sim_fast.apply_cx_fast(0, 1)
        assert_rref_match(sim_gen, sim_fast, "CX removes support")

    def test_small_n(self):
        """Test on minimal system sizes."""
        for n in [2, 3, 4]:
            rng = np.random.default_rng(42)
            sim_gen = SparseGF2(n, check_inputs=False)
            sim_fast = clone_state(sim_gen)
            for _ in range(n * 4):
                op = rng.integers(0, 4)
                if op == 0:
                    q = rng.integers(0, n)
                    sim_gen.apply_h(q)
                    sim_fast.apply_h_fast(q)
                elif op == 1:
                    q = rng.integers(0, n)
                    sim_gen.apply_s(q)
                    sim_fast.apply_s_fast(q)
                elif op == 2 and n > 1:
                    qc = rng.integers(0, n)
                    qt = rng.integers(0, n - 1)
                    if qt >= qc:
                        qt += 1
                    sim_gen.apply_gate(qc, qt, _CX_SYMP)
                    sim_fast.apply_cx_fast(qc, qt)
                else:
                    q = rng.integers(0, n)
                    sim_gen.apply_measurement_z(q)
                    sim_fast.apply_measurement_z(q)
            assert_rref_match(sim_gen, sim_fast, f"Small n={n}")


# ======================================================================
# hybrid_mode tests (pure sparse default vs hybrid auto-switching)
# ======================================================================

class TestHybridMode:
    """Test that hybrid_mode=True (auto dense-switching) produces identical results
    to the default pure-sparse mode."""

    @staticmethod
    def _run_surface_circuit(circuit, hybrid):
        """Run a Stim surface code circuit through SparseGF2."""
        n = circuit.num_qubits
        sim = SparseGF2(n, use_min_weight_pivot=True, hybrid_mode=hybrid)
        CX = _CX_SYMP
        for inst in circuit.flattened():
            name = inst.name
            targets = inst.targets_copy()
            if name in ("TICK", "DETECTOR", "OBSERVABLE_INCLUDE",
                        "QUBIT_COORDS", "SHIFT_COORDS",
                        "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"):
                continue
            elif name == "H":
                for t in targets:
                    sim.apply_h(t.value)
            elif name == "S":
                for t in targets:
                    sim.apply_s(t.value)
            elif name in ("CX", "CNOT", "ZCX"):
                for i in range(0, len(targets), 2):
                    sim.apply_gate(targets[i].value, targets[i + 1].value, CX)
            elif name in ("R", "RZ", "M", "MZ", "MR", "MRZ"):
                for t in targets:
                    sim.apply_measurement_z(t.value)
            elif name in ("MX", "MRX"):
                for t in targets:
                    sim.apply_h(t.value)
                    sim.apply_measurement_z(t.value)
                    sim.apply_h(t.value)
        return sim

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_hybrid_matches_pure_sparse(self, d):
        """hybrid_mode=True produces identical k and RREF as pure sparse (default)."""
        import stim
        circuit = stim.Circuit.generated(
            "surface_code:unrotated_memory_z", distance=d, rounds=d,
            after_clifford_depolarization=0)

        sim_pure = self._run_surface_circuit(circuit, hybrid=False)
        sim_hybrid = self._run_surface_circuit(circuit, hybrid=True)

        k_pure = sim_pure.compute_k()
        k_hybrid = sim_hybrid.compute_k()
        assert k_pure == k_hybrid, (
            f"d={d}: k mismatch hybrid={k_hybrid} vs pure={k_pure}")
        assert_rref_match(sim_pure, sim_hybrid,
                          f"d={d}: hybrid mode RREF mismatch")

    @pytest.mark.parametrize("d", [3, 5, 7])
    def test_hybrid_fast_kernels(self, d):
        """Pure sparse (default) with fast kernels matches hybrid mode."""
        import stim
        circuit = stim.Circuit.generated(
            "surface_code:unrotated_memory_z", distance=d, rounds=d,
            after_clifford_depolarization=0)

        n = circuit.num_qubits
        CX = _CX_SYMP

        sim_hybrid = SparseGF2(n, use_min_weight_pivot=True, hybrid_mode=True)
        sim_pure = SparseGF2(n, use_min_weight_pivot=True, check_inputs=False)

        for inst in circuit.flattened():
            name = inst.name
            targets = inst.targets_copy()
            if name in ("TICK", "DETECTOR", "OBSERVABLE_INCLUDE",
                        "QUBIT_COORDS", "SHIFT_COORDS",
                        "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"):
                continue
            elif name == "H":
                for t in targets:
                    sim_hybrid.apply_h(t.value)
                    sim_pure.apply_h_fast(t.value)
            elif name == "S":
                for t in targets:
                    sim_hybrid.apply_s(t.value)
                    sim_pure.apply_s_fast(t.value)
            elif name in ("CX", "CNOT", "ZCX"):
                for i in range(0, len(targets), 2):
                    sim_hybrid.apply_gate(
                        targets[i].value, targets[i + 1].value, CX)
                    sim_pure.apply_cx_fast(
                        targets[i].value, targets[i + 1].value)
            elif name in ("R", "RZ", "M", "MZ", "MR", "MRZ"):
                for t in targets:
                    sim_hybrid.apply_measurement_z(t.value)
                    sim_pure.apply_measurement_z(t.value)
            elif name in ("MX", "MRX"):
                for t in targets:
                    sim_hybrid.apply_h(t.value)
                    sim_hybrid.apply_measurement_z(t.value)
                    sim_hybrid.apply_h(t.value)
                    sim_pure.apply_h_fast(t.value)
                    sim_pure.apply_measurement_z(t.value)
                    sim_pure.apply_h_fast(t.value)

        assert_rref_match(sim_hybrid, sim_pure,
                          f"d={d}: hybrid + fast kernels RREF mismatch")

    def test_pure_sparse_stays_sparse(self):
        """Default mode (pure sparse) never enters dense mode even with many ops."""
        n = 16
        sim = SparseGF2(n, check_inputs=False)
        rng = np.random.default_rng(42)

        # Apply many gates (would normally trigger mode-switch check)
        for _ in range(n * 20):
            qc = rng.integers(0, n)
            qt = rng.integers(0, n - 1)
            if qt >= qc:
                qt += 1
            sim.apply_cx_fast(qc, qt)

        assert not sim._dense_mode, "pure sparse sim switched to dense mode"
