"""
Sanity check: verify that SparseGF2 produces the EXACT SAME stabilizer group as Stim.

This script generates random Clifford circuits and runs them through both Stim's
TableauSimulator and our StabilizerTableau. After each circuit, we extract the
full symplectic tableau from both simulators, reduce both to GF(2) RREF, and
verify they span the identical subspace.

Tested circuit classes
----------------------
1. Random 1-qubit Clifford circuits (H, S only)
2. Random 2-qubit Clifford circuits (CNOT, CZ, H, S)
3. Random circuits with measurements and resets
4. Bell pair initialization + random gates + measurements (purification picture)
5. All 11,520 two-qubit Cliffords individually
6. Deep random circuits at various system sizes
7. apply_clifford_2q consistency with named gate methods
8. iSWAP vs Stim
9. Input validation (error handling, edge cases)
10. Deep random circuits with iSWAP

Subspace equivalence
--------------------
Two stabilizer tableaux generate the same group iff their symplectic matrices
have the same GF(2) RREF (reduced row echelon form). This is the physically
meaningful check: the generators may be written differently, but they stabilize
the same state.

Usage
-----
    python sanity_checks/verify_vs_stim.py
    python -m pytest sanity_checks/verify_vs_stim.py -v
"""

import sys
from pathlib import Path
import numpy as np

try:
    import stim
except ImportError:
    print("ERROR: stim is required for sanity checks. Install with: pip install stim")
    sys.exit(1)

# Allow running as a script without pip install -e .
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from sparsegf2.core.tableau import StabilizerTableau, gf2_rref, stabilizer_groups_equal
from sparsegf2.gates.clifford import symplectic_from_stim_tableau


# ======================================================================
# Helpers
# ======================================================================

def _ensure_stim_qubits(sim: stim.TableauSimulator, n: int):
    """Ensure Stim's TableauSimulator knows about qubits 0..n-1.

    Stim only tracks qubits it has seen. We force it to know about all n
    qubits by resetting any it hasn't encountered yet.
    """
    cur_n = len(sim.current_inverse_tableau())
    for q in range(cur_n, n):
        sim.reset(q)


def extract_stim_symplectic(sim: stim.TableauSimulator, n: int) -> np.ndarray:
    """Extract the full symplectic matrix [X | Z] from a Stim TableauSimulator.

    Returns shape (n, 2*n). Row r = r-th stabilizer generator.
    Columns 0..n-1 are X, columns n..2n-1 are Z.
    """
    _ensure_stim_qubits(sim, n)

    inv_tab = sim.current_inverse_tableau()
    tab = inv_tab.inverse()
    assert len(tab) == n, f"Stim has {len(tab)} qubits, expected {n}"

    symplectic = np.zeros((n, 2 * n), dtype=np.uint8)
    for r in range(n):
        ps = tab.z_output(r)
        for q in range(n):
            pauli = ps[q]  # 0=I, 1=X, 2=Y, 3=Z
            if pauli in (1, 2):  # X or Y
                symplectic[r, q] = 1
            if pauli in (2, 3):  # Y or Z
                symplectic[r, n + q] = 1

    return symplectic


# Dispatch tables: op name -> (our_method, stim_method)
_1Q_DISPATCH = {
    "H": ("h", "h"),
    "S": ("s", "s"),
}
_2Q_DISPATCH = {
    "CNOT": ("cnot", "cx"),
    "CZ":   ("cz", "cz"),
    "SWAP": ("swap", "swap"),
    "ISWAP": ("iswap", "iswap"),
}


def _apply_random_op(op, rng, n, our, stim_sim):
    """Apply a single random operation to both simulators.

    Parameters
    ----------
    op : str
        One of "H", "S", "CNOT", "CZ", "SWAP", "ISWAP", "MR".
    rng : numpy.random.Generator
        Random number generator.
    n : int
        Number of qubits to choose from.
    our : StabilizerTableau
        Our simulator.
    stim_sim : stim.TableauSimulator
        Stim's simulator.
    """
    if op in _1Q_DISPATCH:
        q = int(rng.integers(0, n))
        our_method, stim_method = _1Q_DISPATCH[op]
        getattr(our, our_method)(q)
        getattr(stim_sim, stim_method)(q)
    elif op in _2Q_DISPATCH:
        q0, q1 = [int(x) for x in rng.choice(n, size=2, replace=False)]
        our_method, stim_method = _2Q_DISPATCH[op]
        getattr(our, our_method)(q0, q1)
        getattr(stim_sim, stim_method)(q0, q1)
    elif op == "MR":
        q = int(rng.integers(0, n))
        our.measure_z(q)
        stim_sim.measure(q)
        stim_sim.reset(q)
    else:
        raise ValueError(f"Unknown operation: {op}")


def _run_random_circuit(rng, n, our, stim_sim, n_ops, ops, probs=None):
    """Apply n_ops random operations to both simulators in lockstep.

    Parameters
    ----------
    rng : numpy.random.Generator
    n : int
        Number of qubits to choose from (may be less than our.n for purification).
    our : StabilizerTableau
    stim_sim : stim.TableauSimulator
    n_ops : int
        Number of operations to apply.
    ops : list of str
        Operation names to choose from.
    probs : list of float or None
        Probability weights for each operation.
    """
    for _ in range(n_ops):
        op = rng.choice(ops, p=probs)
        _apply_random_op(op, rng, n, our, stim_sim)


def _assert_groups_equal(our, stim_sim, msg):
    """Extract symplectic matrices from both simulators and assert equality."""
    our_symp = our.to_symplectic()
    stim_symp = extract_stim_symplectic(stim_sim, our.n)
    assert stabilizer_groups_equal(our_symp, stim_symp), f"MISMATCH: {msg}"


# ======================================================================
# Test 1: Single-qubit gates only (H, S)
# ======================================================================

def test_single_qubit_gates():
    """Verify H and S gates produce the same stabilizer group as Stim."""
    rng = np.random.default_rng(42)

    for n in [1, 2, 4, 8]:
        for trial in range(20):
            our = StabilizerTableau.from_zero_state(n)
            stim_sim = stim.TableauSimulator()

            n_gates = rng.integers(1, 4 * n)
            _run_random_circuit(rng, n, our, stim_sim, n_gates, ["H", "S"])
            _assert_groups_equal(our, stim_sim, f"n={n}, trial={trial}, 1Q gates only")

    print("  [PASS] Test 1: Single-qubit gates (H, S)")


# ======================================================================
# Test 2: Two-qubit gates (CNOT, CZ)
# ======================================================================

def test_two_qubit_gates():
    """Verify CNOT and CZ produce the same stabilizer group as Stim."""
    rng = np.random.default_rng(123)

    for n in [2, 4, 8, 16]:
        for trial in range(20):
            our = StabilizerTableau.from_zero_state(n)
            stim_sim = stim.TableauSimulator()

            n_gates = rng.integers(n, 6 * n)
            _run_random_circuit(rng, n, our, stim_sim, n_gates,
                                ["H", "S", "CNOT", "CZ"])
            _assert_groups_equal(our, stim_sim, f"n={n}, trial={trial}, 2Q gates")

    print("  [PASS] Test 2: Two-qubit gates (CNOT, CZ, H, S)")


# ======================================================================
# Test 3: Circuits with measurements
# ======================================================================

def test_with_measurements():
    """Verify circuits with Z measurements produce the same stabilizer group."""
    rng = np.random.default_rng(456)

    for n in [4, 8, 16]:
        for trial in range(20):
            our = StabilizerTableau.from_zero_state(n)
            stim_sim = stim.TableauSimulator()

            n_ops = rng.integers(2 * n, 8 * n)
            _run_random_circuit(rng, n, our, stim_sim, n_ops,
                                ["H", "S", "CNOT", "CZ", "MR"],
                                [0.2, 0.1, 0.25, 0.25, 0.2])
            _assert_groups_equal(our, stim_sim,
                                 f"n={n}, trial={trial}, with measurements")

    print("  [PASS] Test 3: Circuits with measurements")


# ======================================================================
# Test 4: Purification picture (Bell pairs + gates + measurements)
# ======================================================================

def test_purification_picture():
    """Verify the purification picture: Bell pairs, system gates, measurements."""
    rng = np.random.default_rng(789)

    for n_sys in [4, 8, 16]:
        for trial in range(15):
            our = StabilizerTableau.from_bell_pairs(n_sys)

            # Stim: manually create Bell pairs
            stim_sim = stim.TableauSimulator()
            for i in range(n_sys):
                stim_sim.h(i)
                stim_sim.cx(i, n_sys + i)

            # Apply random gates on SYSTEM QUBITS ONLY (0..n_sys-1)
            n_ops = rng.integers(4 * n_sys, 12 * n_sys)
            _run_random_circuit(rng, n_sys, our, stim_sim, n_ops,
                                ["H", "S", "CNOT", "CZ", "MR"],
                                [0.2, 0.1, 0.25, 0.25, 0.2])
            _assert_groups_equal(our, stim_sim,
                                 f"purification n_sys={n_sys}, trial={trial}")

    print("  [PASS] Test 4: Purification picture (Bell pairs + system gates + measurements)")


# ======================================================================
# Test 5: All 11,520 two-qubit Cliffords
# ======================================================================

def test_all_11520_cliffords():
    """Verify every single 2-qubit Clifford gate produces the same stabilizer."""
    rng = np.random.default_rng(999)
    n_checked = 0

    for idx, stim_tab in enumerate(stim.Tableau.iter_all(2)):
        S = symplectic_from_stim_tableau(stim_tab)

        # Start from a random stabilizer state (not just |00>)
        our = StabilizerTableau.from_zero_state(2)
        stim_sim = stim.TableauSimulator()
        _run_random_circuit(rng, 2, our, stim_sim, 3, ["H", "S", "CNOT"])

        # Apply the Clifford under test
        our.apply_clifford_2q(0, 1, S)
        stim_sim.do(stim_tab.to_circuit("elimination"))

        our_symp = our.to_symplectic()
        stim_symp = extract_stim_symplectic(stim_sim, our.n)
        assert stabilizer_groups_equal(our_symp, stim_symp), (
            f"MISMATCH: Clifford index {idx}\n"
            f"Symplectic matrix S:\n{S}\n"
            f"Our RREF:\n{gf2_rref(our_symp)}\n"
            f"Stim RREF:\n{gf2_rref(stim_symp)}"
        )
        n_checked += 1

    print(f"  [PASS] Test 5: All {n_checked} two-qubit Cliffords verified")


# ======================================================================
# Test 6: Deep random circuits at various sizes
# ======================================================================

def test_deep_random_circuits():
    """Verify deep random circuits at n=4,8,16,32."""
    rng = np.random.default_rng(2025)

    for n in [4, 8, 16, 32]:
        for trial in range(5):
            our = StabilizerTableau.from_zero_state(n)
            stim_sim = stim.TableauSimulator()

            _run_random_circuit(rng, n, our, stim_sim, 20 * n,
                                ["H", "S", "CNOT", "CZ", "SWAP", "MR"],
                                [0.15, 0.1, 0.25, 0.2, 0.1, 0.2])
            _assert_groups_equal(our, stim_sim,
                                 f"deep circuit n={n}, trial={trial}")

    print("  [PASS] Test 6: Deep random circuits (n=4,8,16,32)")


# ======================================================================
# Test 7: apply_clifford_2q matches named gates
# ======================================================================

def test_apply_clifford_matches_named():
    """Verify that apply_clifford_2q with known S matrices matches named gate methods."""
    # Map: gate name -> (symplectic matrix, named method name)
    gate_defs = {
        "CNOT": np.array([
            [1, 1, 0, 0],  # X_0 -> X_0 X_1
            [0, 1, 0, 0],  # X_1 -> X_1
            [0, 0, 1, 0],  # Z_0 -> Z_0
            [0, 0, 1, 1],  # Z_1 -> Z_0 Z_1
        ], dtype=np.uint8),
        "CZ": np.array([
            [1, 0, 0, 1],  # X_0 -> X_0 Z_1
            [0, 1, 1, 0],  # X_1 -> Z_0 X_1
            [0, 0, 1, 0],  # Z_0 -> Z_0
            [0, 0, 0, 1],  # Z_1 -> Z_1
        ], dtype=np.uint8),
    }
    named_methods = {"CNOT": "cnot", "CZ": "cz"}

    rng = np.random.default_rng(777)
    n = 8

    for _ in range(50):
        # Create a random starting state
        base = StabilizerTableau.from_zero_state(n)
        for __ in range(3 * n):
            q = int(rng.integers(0, n))
            if rng.random() < 0.5:
                base.h(q)
            else:
                base.s(q)

        q0, q1 = [int(x) for x in rng.choice(n, size=2, replace=False)]

        for gate_name, S_matrix in gate_defs.items():
            tab_named = base.copy()
            getattr(tab_named, named_methods[gate_name])(q0, q1)

            tab_generic = base.copy()
            tab_generic.apply_clifford_2q(q0, q1, S_matrix)

            assert stabilizer_groups_equal(
                tab_named.to_symplectic(), tab_generic.to_symplectic()
            ), f"{gate_name} via apply_clifford_2q doesn't match {named_methods[gate_name]}()"

    print("  [PASS] Test 7: apply_clifford_2q matches named gates (CNOT, CZ)")


# ======================================================================
# Test 8: iSWAP vs Stim
# ======================================================================

def test_iswap_vs_stim():
    """Verify iSWAP gate matches Stim's ISWAP at the GF(2) level."""
    rng = np.random.default_rng(1234)

    for n in [2, 4, 8, 16]:
        for trial in range(20):
            our = StabilizerTableau.from_zero_state(n)
            stim_sim = stim.TableauSimulator()
            _ensure_stim_qubits(stim_sim, n)

            # Randomize initial state with Hadamards, then apply a single iSWAP
            _run_random_circuit(rng, n, our, stim_sim, 3 * n, ["H"])

            q0, q1 = [int(x) for x in rng.choice(n, size=2, replace=False)]
            our.iswap(q0, q1)
            stim_sim.iswap(q0, q1)

            _assert_groups_equal(our, stim_sim,
                                 f"iSWAP n={n}, trial={trial}, qubits=({q0},{q1})")

    print("  [PASS] Test 8: iSWAP vs Stim")


# ======================================================================
# Test 9: Input validation
# ======================================================================

def _assert_raises(exc_type, fn, *args):
    """Assert that fn(*args) raises exc_type. Works both with pytest and standalone."""
    try:
        fn(*args)
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__} from {fn}, args={args}")


def test_input_validation():
    """Verify that invalid inputs raise appropriate errors."""
    tab = StabilizerTableau.from_zero_state(4)

    # Negative qubit count
    _assert_raises(ValueError, StabilizerTableau, -1)

    # Out-of-range qubit
    for method in [tab.h, tab.s, tab.s_dag, tab.sqrt_x, tab.sqrt_x_dag,
                   tab.x_gate, tab.y_gate, tab.z_gate, tab.measure_z]:
        _assert_raises(IndexError, method, 4)
        _assert_raises(IndexError, method, -1)

    # Duplicate qubits in 2-qubit gates
    for method in [tab.cnot, tab.cz, tab.swap, tab.iswap]:
        _assert_raises(ValueError, method, 0, 0)

    # apply_clifford_2q with duplicate qubits
    _assert_raises(ValueError, tab.apply_clifford_2q, 1, 1,
                   np.eye(4, dtype=np.uint8))

    # n=0 and n=1 edge cases should work
    tab0 = StabilizerTableau.from_zero_state(0)
    assert tab0.gf2_rank() == 0

    tab1 = StabilizerTableau.from_zero_state(1)
    tab1.h(0)
    tab1.measure_z(0)
    assert tab1.z.get_bit(0, 0) == 1  # After measure+reset, stabilizer is Z_0

    print("  [PASS] Test 9: Input validation")


# ======================================================================
# Test 10: Deep circuits WITH iSWAP
# ======================================================================

def test_deep_with_iswap():
    """Deep random circuits including iSWAP, verified against Stim."""
    rng = np.random.default_rng(5555)

    for n in [4, 8, 16]:
        for trial in range(10):
            our = StabilizerTableau.from_zero_state(n)
            stim_sim = stim.TableauSimulator()
            _ensure_stim_qubits(stim_sim, n)

            _run_random_circuit(
                rng, n, our, stim_sim, 15 * n,
                ["H", "S", "CNOT", "CZ", "ISWAP", "SWAP", "MR"],
                [0.15, 0.1, 0.2, 0.15, 0.15, 0.05, 0.2])
            _assert_groups_equal(our, stim_sim,
                                 f"deep+iswap n={n}, trial={trial}")

    print("  [PASS] Test 10: Deep circuits with iSWAP")


# ======================================================================
# Main
# ======================================================================

def run_all_checks():
    """Run all sanity checks."""
    print("=" * 60)
    print("SparseGF2 vs Stim: Stabilizer Group Equivalence Checks")
    print("=" * 60)
    print()

    test_single_qubit_gates()
    test_two_qubit_gates()
    test_with_measurements()
    test_purification_picture()
    test_all_11520_cliffords()
    test_deep_random_circuits()
    test_apply_clifford_matches_named()
    test_iswap_vs_stim()
    test_input_validation()
    test_deep_with_iswap()

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_checks()
