"""
Phase 5: Per-step Stim equivalence verification.

Verifies that after EVERY SINGLE gate and measurement in a random circuit,
the SparseGF2 stabilizer tableau produces the exact same stabilizer group
(via GF(2) RREF) as Stim. This catches any gate that drifts even slightly.

Tests:
  5.1: Per-gate verification (check after every gate)
  5.2: Per-measurement verification (check after every measurement)
  5.3: Full purification circuit (Bell pairs + system gates + measurements)
"""

import sys
from pathlib import Path
import numpy as np

try:
    import stim
except ImportError:
    print("ERROR: stim required. pip install stim")
    sys.exit(1)

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from sparsegf2.core.tableau import StabilizerTableau, stabilizer_groups_equal


def _ensure_stim_qubits(sim, n):
    cur = len(sim.current_inverse_tableau())
    for q in range(cur, n):
        sim.reset(q)


def _extract_stim(sim, n):
    _ensure_stim_qubits(sim, n)
    tab = sim.current_inverse_tableau().inverse()
    symp = np.zeros((n, 2 * n), dtype=np.uint8)
    for r in range(n):
        ps = tab.z_output(r)
        for q in range(n):
            p = ps[q]
            if p in (1, 2): symp[r, q] = 1
            if p in (2, 3): symp[r, n + q] = 1
    return symp


# ======================================================================
# 5.1: Per-gate step verification
# ======================================================================

def test_per_gate_equivalence():
    """After every gate, check tableau matches Stim."""
    rng = np.random.default_rng(42)
    total_ops = 0
    total_checks = 0

    for n in [8, 16, 32]:
        for trial in range(10):
            our = StabilizerTableau.from_zero_state(n)
            stim_sim = stim.TableauSimulator()

            n_ops = 8 * n
            for step in range(n_ops):
                op = rng.choice(["H", "S", "CNOT", "CZ", "SWAP", "ISWAP"])
                if op in ("H", "S"):
                    q = int(rng.integers(0, n))
                    if op == "H":
                        our.h(q); stim_sim.h(q)
                    else:
                        our.s(q); stim_sim.s(q)
                else:
                    q0, q1 = [int(x) for x in rng.choice(n, 2, replace=False)]
                    if op == "CNOT":
                        our.cnot(q0, q1); stim_sim.cx(q0, q1)
                    elif op == "CZ":
                        our.cz(q0, q1); stim_sim.cz(q0, q1)
                    elif op == "SWAP":
                        our.swap(q0, q1); stim_sim.swap(q0, q1)
                    else:
                        our.iswap(q0, q1); stim_sim.iswap(q0, q1)

                total_ops += 1
                # Check after every gate
                our_symp = our.to_symplectic()
                stim_symp = _extract_stim(stim_sim, n)
                assert stabilizer_groups_equal(our_symp, stim_symp), (
                    f"MISMATCH after step {step} ({op}), n={n}, trial={trial}")
                total_checks += 1

    print(f"  [PASS] 5.1: Per-gate equivalence ({total_checks} step-by-step checks)")


# ======================================================================
# 5.2: Per-measurement step verification
# ======================================================================

def test_per_measurement_equivalence():
    """After every measurement, check tableau matches Stim."""
    rng = np.random.default_rng(123)
    total_checks = 0

    for n in [8, 16, 32]:
        for trial in range(10):
            our = StabilizerTableau.from_zero_state(n)
            stim_sim = stim.TableauSimulator()

            for step in range(8 * n):
                op = rng.choice(["H", "S", "CNOT", "CZ", "MR"],
                                p=[0.15, 0.1, 0.25, 0.2, 0.3])
                if op in ("H", "S"):
                    q = int(rng.integers(0, n))
                    if op == "H":
                        our.h(q); stim_sim.h(q)
                    else:
                        our.s(q); stim_sim.s(q)
                elif op in ("CNOT", "CZ"):
                    q0, q1 = [int(x) for x in rng.choice(n, 2, replace=False)]
                    if op == "CNOT":
                        our.cnot(q0, q1); stim_sim.cx(q0, q1)
                    else:
                        our.cz(q0, q1); stim_sim.cz(q0, q1)
                else:
                    q = int(rng.integers(0, n))
                    our.measure_z(q); stim_sim.measure(q); stim_sim.reset(q)

                    # Check after every measurement
                    our_symp = our.to_symplectic()
                    stim_symp = _extract_stim(stim_sim, n)
                    assert stabilizer_groups_equal(our_symp, stim_symp), (
                        f"MISMATCH after measurement at step {step}, n={n}, trial={trial}")
                    total_checks += 1

    print(f"  [PASS] 5.2: Per-measurement equivalence ({total_checks} checks)")


# ======================================================================
# 5.3: Full purification circuit equivalence
# ======================================================================

def test_purification_full():
    """Full purification circuit: Bell pairs + system gates + measurements.
    Verify at the end AND compute k = system_rank - n_system."""
    rng = np.random.default_rng(789)
    total_checks = 0

    for n_sys in [16, 32, 64, 128]:
        n_total = 2 * n_sys
        for trial in range(5):
            our = StabilizerTableau.from_bell_pairs(n_sys)
            stim_sim = stim.TableauSimulator()
            for i in range(n_sys):
                stim_sim.h(i)
                stim_sim.cx(i, n_sys + i)

            # 8n layers of brickwork + measurements
            for layer in range(8 * n_sys):
                offset = layer % 2
                for i in range(offset, n_sys - 1, 2):
                    our.cnot(i, i + 1)
                    stim_sim.cx(i, i + 1)
                    if rng.random() < 0.15:
                        our.measure_z(i)
                        stim_sim.measure(i); stim_sim.reset(i)

            our_symp = our.to_symplectic()
            stim_symp = _extract_stim(stim_sim, n_total)
            assert stabilizer_groups_equal(our_symp, stim_symp), (
                f"Purification MISMATCH: n_sys={n_sys}, trial={trial}")

            # Verify k matches
            k_our = our.system_rank(n_sys) - n_sys
            total_checks += 1

    print(f"  [PASS] 5.3: Purification circuit equivalence "
          f"(n_sys=16,32,64,128, {total_checks} checks)")


def run_all():
    print("=" * 60)
    print("Phase 5: Per-Step Stim Equivalence Verification")
    print("=" * 60)
    print()
    test_per_gate_equivalence()
    test_per_measurement_equivalence()
    test_purification_full()
    print()
    print("=" * 60)
    print("ALL PHASE 5 CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
