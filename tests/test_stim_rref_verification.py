"""
Verify SparseGF2 produces the EXACT same stabilizer group as Stim.

Compares the GF(2) RREF of the symplectic tableaux extracted from both
simulators after identical random Clifford circuits. The stabilizer group
is the same iff the RREF of the two matrices is identical (they span the
same GF(2) row space).

Circuit structure: NN brickwork on n system qubits in the purification
picture (2n total qubits: n system + n reference, initialized as Bell
pairs). Layer t applies random 2-qubit Cliffords on pairs with offset
t%2, then measures each gated qubit with probability p.

Usage:
    py -3.13 -m pytest tests/test_stim_rref_verification.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import stim

# Allow running from repo root without pip install -e .
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from sparsegf2.core.sparse_tableau import SparseGF2, warmup
from sparsegf2.gates.clifford import symplectic_from_stim_tableau


# ======================================================================
# Pre-build Clifford cache (module level, shared by all tests)
# ======================================================================

N_CLIFF = 4096

_CLIFF_TABS = []
_CLIFF_SYMP = np.zeros((N_CLIFF, 4, 4), dtype=np.uint8)

def _build_clifford_cache():
    """Build a deterministic cache of 2-qubit Cliffords (Stim tableaux + symplectic).

    Uses the full enumeration of all 11,520 two-qubit Cliffords via
    stim.Tableau.iter_all(2), then takes a deterministic stride-sampled
    subset for reproducible test runs.
    """
    if len(_CLIFF_TABS) > 0:
        return
    all_tabs = list(stim.Tableau.iter_all(2))  # 11,520 tableaux, deterministic order
    # Stride-sample to get N_CLIFF entries spread across the full set
    stride = max(1, len(all_tabs) // N_CLIFF)
    for i in range(N_CLIFF):
        _CLIFF_TABS.append(all_tabs[(i * stride) % len(all_tabs)])
    for i, tab in enumerate(_CLIFF_TABS):
        _CLIFF_SYMP[i] = symplectic_from_stim_tableau(tab)


# ======================================================================
# Helpers
# ======================================================================

def gf2_rref(mat):
    """Compute Reduced Row Echelon Form over GF(2).

    Parameters
    ----------
    mat : ndarray of uint8, shape (nrows, ncols)
        Binary matrix.

    Returns
    -------
    rref : ndarray of uint8, shape (rank, ncols)
        The RREF rows (zero rows stripped).
    """
    m = mat.copy()
    nrows, ncols = m.shape
    rank = 0
    for col in range(ncols):
        # Find pivot in this column at or below current rank row
        pivot = -1
        for row in range(rank, nrows):
            if m[row, col]:
                pivot = row
                break
        if pivot == -1:
            continue
        # Swap pivot row into position
        if pivot != rank:
            m[[rank, pivot]] = m[[pivot, rank]]
        # Eliminate all other rows with a 1 in this column
        for row in range(nrows):
            if row != rank and m[row, col]:
                m[row] ^= m[rank]
        rank += 1
    return m[:rank]


def extract_stim_sys_matrix(sim, n):
    """Extract the 2n x 2n system sub-matrix from a Stim TableauSimulator.

    The forward tableau's z_output for all 2n qubits gives the stabilizer
    generators. We extract only the system-qubit columns (0..n-1) in the
    [X|Z] format to produce a 2n x 2n matrix.

    Parameters
    ----------
    sim : stim.TableauSimulator
        Stim simulator after circuit execution.
    n : int
        Number of system qubits.

    Returns
    -------
    mat : ndarray of uint8, shape (2n, 2n)
        Symplectic matrix [X|Z] restricted to system qubits.
    """
    N = 2 * n
    inv_tab = sim.current_inverse_tableau()
    fwd = inv_tab.inverse(unsigned=True)

    mat = np.zeros((N, 2 * n), dtype=np.uint8)
    for row in range(N):
        ps = fwd.z_output(row)
        for col in range(n):
            pauli = ps[col]  # 0=I, 1=X, 2=Y, 3=Z
            if pauli in (1, 2):          # X or Y -> x-bit
                mat[row, col] = 1
            if pauli in (2, 3):          # Y or Z -> z-bit
                mat[row, n + col] = 1
    return mat


def generate_schedule(n, p, seed, n_layers):
    """Generate a brickwork circuit schedule.

    Parameters
    ----------
    n : int
        Number of system qubits (must be even).
    p : float
        Measurement probability per gated qubit.
    seed : int
        RNG seed.
    n_layers : int
        Number of circuit layers.

    Returns
    -------
    schedule : list of (pairs, cliff_indices, measurements)
        Each entry is one layer of the circuit.
    """
    rng = np.random.default_rng(seed)
    schedule = []
    for t in range(n_layers):
        offset = t % 2
        n_pairs = n // 2
        pairs = []
        for j in range(n_pairs):
            if offset == 0:
                qi, qj = 2 * j, 2 * j + 1
            else:
                qi = 2 * j + 1
                qj = (2 * j + 2) % n
            pairs.append((qi, qj))
        cliff_indices = rng.integers(0, N_CLIFF, size=n_pairs)
        measurements = []
        for qi, qj in pairs:
            if rng.random() < p:
                measurements.append(qi)
            if rng.random() < p:
                measurements.append(qj)
        measurements = sorted(set(measurements))
        schedule.append((pairs, cliff_indices, measurements))
    return schedule


def run_sparse(n, schedule):
    """Run a circuit schedule through SparseGF2.

    Returns the simulator after execution.
    """
    sim = SparseGF2(n, use_min_weight_pivot=True)
    for pairs, ci, meas in schedule:
        for ip, (qi, qj) in enumerate(pairs):
            sim.apply_gate(qi, qj, _CLIFF_SYMP[ci[ip]])
        for mq in meas:
            sim.apply_measurement_z(mq)
    return sim


def run_stim(n, schedule):
    """Run a circuit schedule through Stim's TableauSimulator.

    Initializes Bell pairs on 2n qubits (H on system, CNOT to reference),
    then applies gates via do_tableau and measurements via measure+reset.

    Returns the simulator after execution.
    """
    sim_st = stim.TableauSimulator()
    # Initialize Bell pairs: |Phi+> between system qubit i and reference qubit n+i
    for i in range(n):
        sim_st.h(i)
        sim_st.cx(i, n + i)
    for pairs, ci, meas in schedule:
        for ip, (qi, qj) in enumerate(pairs):
            sim_st.do_tableau(_CLIFF_TABS[ci[ip]], [qi, qj])
        for mq in meas:
            sim_st.measure(mq)
            sim_st.reset(mq)
    return sim_st


def assert_rref_match(sim_sparse, sim_stim, n, msg=""):
    """Assert that the GF(2) RREF of both simulators' tableaux are identical."""
    mat_sparse = sim_sparse.extract_sys_matrix()
    mat_stim = extract_stim_sys_matrix(sim_stim, n)

    rref_sparse = gf2_rref(mat_sparse)
    rref_stim = gf2_rref(mat_stim)

    assert rref_sparse.shape == rref_stim.shape, (
        f"RREF rank mismatch: sparse rank={rref_sparse.shape[0]}, "
        f"stim rank={rref_stim.shape[0]}. {msg}"
    )
    assert np.array_equal(rref_sparse, rref_stim), (
        f"RREF content mismatch (same rank={rref_sparse.shape[0]}). {msg}"
    )


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_cliffords_and_jit():
    """Build Clifford cache and warm up Numba JIT (once per session)."""
    _build_clifford_cache()
    warmup()


# ======================================================================
# Test 1: Volume-law regime (p=0.05)
# ======================================================================

def test_rref_match_volume_law():
    """RREF match at p=0.05 (volume-law): n=16, 5 seeds, 8n layers."""
    n = 16
    p = 0.05
    n_layers = 8 * n

    for seed in range(5):
        schedule = generate_schedule(n, p, seed, n_layers)
        sim_sparse = run_sparse(n, schedule)
        sim_stim = run_stim(n, schedule)
        assert_rref_match(
            sim_sparse, sim_stim, n,
            msg=f"volume-law p={p}, n={n}, seed={seed}"
        )


# ======================================================================
# Test 2: Area-law regime (p=0.25)
# ======================================================================

def test_rref_match_area_law():
    """RREF match at p=0.25 (area-law): n=16, 5 seeds, 8n layers."""
    n = 16
    p = 0.25
    n_layers = 8 * n

    for seed in range(5):
        schedule = generate_schedule(n, p, seed, n_layers)
        sim_sparse = run_sparse(n, schedule)
        sim_stim = run_stim(n, schedule)
        assert_rref_match(
            sim_sparse, sim_stim, n,
            msg=f"area-law p={p}, n={n}, seed={seed}"
        )


# ======================================================================
# Test 3: Critical regime (p=0.16)
# ======================================================================

def test_rref_match_critical():
    """RREF match at p=0.16 (near critical point): n=16, 5 seeds, 8n layers."""
    n = 16
    p = 0.16
    n_layers = 8 * n

    for seed in range(5):
        schedule = generate_schedule(n, p, seed, n_layers)
        sim_sparse = run_sparse(n, schedule)
        sim_stim = run_stim(n, schedule)
        assert_rref_match(
            sim_sparse, sim_stim, n,
            msg=f"critical p={p}, n={n}, seed={seed}"
        )


# ======================================================================
# Test 4: Step-by-step verification
# ======================================================================

def test_rref_match_step_by_step():
    """RREF match after EVERY gate and measurement, n=8, 1 seed.

    Applies 10 random 2-qubit Cliffords and 3 Z measurements on random
    system qubits, checking RREF equality after each individual operation.
    """
    n = 8
    rng = np.random.default_rng(77)

    sim_sparse = SparseGF2(n, use_min_weight_pivot=True)
    sim_stim = stim.TableauSimulator()
    for i in range(n):
        sim_stim.h(i)
        sim_stim.cx(i, n + i)

    # Check RREF matches after initialization
    assert_rref_match(sim_sparse, sim_stim, n, msg="after Bell pair init")

    # Apply 10 random 2-qubit Cliffords, checking after each
    for step in range(10):
        qi, qj = sorted(rng.choice(n, size=2, replace=False))
        ci = int(rng.integers(0, N_CLIFF))

        sim_sparse.apply_gate(qi, qj, _CLIFF_SYMP[ci])
        sim_stim.do_tableau(_CLIFF_TABS[ci], [int(qi), int(qj)])

        assert_rref_match(
            sim_sparse, sim_stim, n,
            msg=f"after gate step {step}: Clifford[{ci}] on ({qi},{qj})"
        )

    # Apply 3 measurements, checking after each
    for step in range(3):
        q = int(rng.integers(0, n))

        sim_sparse.apply_measurement_z(q)
        sim_stim.measure(q)
        sim_stim.reset(q)

        assert_rref_match(
            sim_sparse, sim_stim, n,
            msg=f"after measurement step {step}: Z measurement on qubit {q}"
        )


# ======================================================================
# Test 5: k-value agreement (n=32, p=0.15, 10 seeds)
# ======================================================================

def test_k_matches_stim():
    """Verify that k = rank - n matches exactly between SparseGF2 and Stim.

    Uses n=32, p=0.15 (near critical), 10 seeds.
    k is computed from the GF(2) rank of the system sub-matrix for both
    simulators and must agree exactly.
    """
    n = 32
    p = 0.15
    n_layers = 8 * n

    for seed in range(10):
        schedule = generate_schedule(n, p, seed, n_layers)
        sim_sparse = run_sparse(n, schedule)
        sim_stim = run_stim(n, schedule)

        # k from SparseGF2's built-in method
        k_sparse = sim_sparse.compute_k()

        # k from Stim: rank of the system sub-matrix minus n
        mat_stim = extract_stim_sys_matrix(sim_stim, n)
        rref_stim = gf2_rref(mat_stim)
        rank_stim = rref_stim.shape[0]
        k_stim = rank_stim - n

        assert k_sparse == k_stim, (
            f"k mismatch: sparse k={k_sparse}, stim k={k_stim} "
            f"(n={n}, p={p}, seed={seed})"
        )

        # Also verify the full RREF matches (stronger than just k)
        mat_sparse = sim_sparse.extract_sys_matrix()
        rref_sparse = gf2_rref(mat_sparse)
        assert rref_sparse.shape[0] == rank_stim, (
            f"rank mismatch: sparse rank={rref_sparse.shape[0]}, "
            f"stim rank={rank_stim} (n={n}, p={p}, seed={seed})"
        )
        assert np.array_equal(rref_sparse, rref_stim), (
            f"RREF mismatch despite k agreement "
            f"(n={n}, p={p}, seed={seed}, k={k_sparse})"
        )


# ======================================================================
# Test 6: Dense mode produces identical RREF to Stim
# ======================================================================

def test_rref_match_dense_mode():
    """Force dense mode and verify RREF matches Stim at n=32, 64 for p=0.05 and p=0.25."""
    for n in [32, 64]:
        for p in [0.05, 0.25]:
            schedule = generate_schedule(n, p, seed=42, n_layers=8 * n)
            # Dense mode
            sim_dense = SparseGF2(n, use_min_weight_pivot=True)
            sim_dense._switch_to_dense()
            for pairs, ci, meas in schedule:
                for ip, (qi, qj) in enumerate(pairs):
                    sim_dense.apply_gate(qi, qj, _CLIFF_SYMP[ci[ip]])
                for mq in meas:
                    sim_dense.apply_measurement_z(mq)
            sim_stim = run_stim(n, schedule)
            assert_rref_match(
                sim_dense, sim_stim, n,
                msg=f"dense mode n={n}, p={p}"
            )


# ======================================================================
# Test 7: Dense-mode compute_k matches Stim
# ======================================================================

def test_compute_k_dense_matches_stim():
    """Verify compute_k in dense mode returns the same k as Stim."""
    n = 32
    for p in [0.05, 0.20]:
        for seed in range(5):
            schedule = generate_schedule(n, p, seed, 8 * n)
            sim_dense = SparseGF2(n, use_min_weight_pivot=True)
            sim_dense._switch_to_dense()
            for pairs, ci, meas in schedule:
                for ip, (qi, qj) in enumerate(pairs):
                    sim_dense.apply_gate(qi, qj, _CLIFF_SYMP[ci[ip]])
                for mq in meas:
                    sim_dense.apply_measurement_z(mq)
            k_dense = sim_dense.compute_k()
            sim_stim = run_stim(n, schedule)
            mat_stim = extract_stim_sys_matrix(sim_stim, n)
            rref_stim = gf2_rref(mat_stim)
            k_stim = rref_stim.shape[0] - n
            assert k_dense == k_stim, (
                f"Dense k={k_dense} != Stim k={k_stim} at n={n}, p={p}, seed={seed}"
            )


# ======================================================================
# Test 8: Sparse and dense modes produce identical RREF
# ======================================================================

def test_sparse_dense_rref_identical():
    """Sparse and dense modes must produce the same RREF for identical circuits."""
    n = 16
    for p in [0.05, 0.16, 0.30]:
        for seed in range(3):
            schedule = generate_schedule(n, p, seed, 8 * n)
            sim_sp = run_sparse(n, schedule)
            sim_dn = SparseGF2(n, use_min_weight_pivot=True)
            sim_dn._switch_to_dense()
            for pairs, ci, meas in schedule:
                for ip, (qi, qj) in enumerate(pairs):
                    sim_dn.apply_gate(qi, qj, _CLIFF_SYMP[ci[ip]])
                for mq in meas:
                    sim_dn.apply_measurement_z(mq)
            mat_sp = sim_sp.extract_sys_matrix()
            mat_dn = sim_dn.extract_sys_matrix()
            rref_sp = gf2_rref(mat_sp)
            rref_dn = gf2_rref(mat_dn)
            assert rref_sp.shape == rref_dn.shape, (
                f"Rank mismatch: sparse={rref_sp.shape[0]}, dense={rref_dn.shape[0]} "
                f"at p={p}, seed={seed}"
            )
            assert np.array_equal(rref_sp, rref_dn), (
                f"RREF mismatch between sparse and dense at p={p}, seed={seed}"
            )


# ======================================================================
# Test 9: Observable extraction (compute_k, entropy, tmi, bandwidth)
# ======================================================================

def test_observables():
    """Test compute_subsystem_entropy, compute_tmi, compute_bandwidth on known circuits."""
    n = 6
    sim = SparseGF2(n)

    # Initial state: N=2n generators (destabs + stabs) for n system qubits.
    # Each qubit has exactly 2 generators (1 destab X_q + 1 stab Z_q).
    assert sim.compute_k() == n
    assert sim.get_active_count() == 2.0  # each qubit: 1 destab + 1 stab
    assert sim.compute_bandwidth() <= 1  # all weight-1 generators

    # Entropy S(A) = rank(restriction to A) - |A|.
    # For initial state, each qubit has 2 independent generators restricted to it:
    # destab X_q -> (1,0) and stab Z_q -> (0,1), giving rank=2 per qubit.
    # S({q0}) = 2 - 1 = 1 (qubit is entangled with its destabilizer partner)
    assert sim.compute_subsystem_entropy([0]) == 1
    assert sim.compute_subsystem_entropy([0, 1]) == 2

    # TMI: for initial independent state, TMI should reflect the destab/stab structure
    tmi = sim.compute_tmi()
    assert isinstance(tmi, (int, float))

    # Apply CNOT(0,1): creates additional correlations between q0 and q1
    S_cnot = np.array([[1, 1, 0, 0], [0, 1, 0, 0],
                       [0, 0, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    sim.apply_gate(0, 1, S_cnot)
    assert sim.compute_k() == n  # no measurements, k unchanged
    # After CNOT: g0=X0X1 and g5=Z0Z1, both span q0 and q1
    assert sim.compute_subsystem_entropy([0]) >= 1

    # Bandwidth is well-defined and returns a non-negative integer
    bw = sim.compute_bandwidth()
    assert bw >= 0

    # Measure q0: k should decrease by 1
    sim.apply_measurement_z(0)
    assert sim.compute_k() == n - 1


# ======================================================================
# Test 10: Single-qubit gate API
# ======================================================================

def test_single_qubit_gates():
    """Test apply_h, apply_s, apply_sqrt_x produce correct stabilizer groups."""
    n = 4

    def compare_1q_gate(gate_fn, stim_gate_fn, gate_name):
        """Apply a 1Q gate to both simulators and compare RREF."""
        sim = SparseGF2(n)
        gate_fn(sim, 0)
        mat = sim.extract_sys_matrix()

        stim_sim = stim.TableauSimulator()
        # Initialize Bell pairs (same as run_stim)
        for i in range(n):
            stim_sim.h(i)
            stim_sim.cx(i, n + i)
        stim_gate_fn(stim_sim, 0)
        mat_stim = extract_stim_sys_matrix(stim_sim, n)

        rref_ours = gf2_rref(mat)
        rref_stim = gf2_rref(mat_stim)
        assert rref_ours.shape == rref_stim.shape, (
            f"{gate_name} RREF rank mismatch: {rref_ours.shape[0]} vs {rref_stim.shape[0]}")
        assert np.array_equal(rref_ours, rref_stim), f"{gate_name} RREF mismatch"

    compare_1q_gate(lambda s, q: s.apply_h(q), lambda s, q: s.h(q), "H")
    compare_1q_gate(lambda s, q: s.apply_s(q), lambda s, q: s.s(q), "S")
    compare_1q_gate(lambda s, q: s.apply_sqrt_x(q), lambda s, q: s.sqrt_x(q), "sqrt_X")


# ======================================================================
# Test 11: Automatic dense/sparse mode switching
# ======================================================================

def test_auto_mode_switching():
    """Run a circuit via run_layer() that triggers automatic mode switching."""
    n = 32
    rng = np.random.default_rng(42)

    sim = SparseGF2(n)
    stim_sim = stim.TableauSimulator()
    # Initialize Bell pairs
    for i in range(n):
        stim_sim.h(i)
        stim_sim.cx(i, n + i)

    assert not sim._dense_mode, "Should start in sparse mode"

    # Phase 1: many gates, no measurements -> should trigger dense mode via run_layer
    for _ in range(8 * n):
        qi = int(rng.integers(0, n - 1))
        qj = qi + 1
        ci = int(rng.integers(0, len(_CLIFF_SYMP)))
        # Use run_layer to trigger _check_mode_switch
        sim.run_layer(
            np.array([qi], dtype=np.int32),
            np.array([qj], dtype=np.int32),
            _CLIFF_SYMP[ci:ci+1],
            np.array([], dtype=np.int32),
        )
        stim_sim.do_tableau(_CLIFF_TABS[ci], [qi, qj])

    # After many gates without measurement, may be in dense mode
    was_dense_after_gates = sim._dense_mode

    # Phase 2: heavy measurements -> should return to sparse mode via run_layer
    for _ in range(4 * n):
        qi = int(rng.integers(0, n - 1))
        qj = qi + 1
        ci = int(rng.integers(0, len(_CLIFF_SYMP)))
        meas = [q for q in range(n) if rng.random() < 0.5]
        sim.run_layer(
            np.array([qi], dtype=np.int32),
            np.array([qj], dtype=np.int32),
            _CLIFF_SYMP[ci:ci+1],
            np.array(meas, dtype=np.int32),
        )
        stim_sim.do_tableau(_CLIFF_TABS[ci], [qi, qj])
        for q in meas:
            stim_sim.measure(q)
            stim_sim.reset(q)

    # Verify RREF match regardless of mode transitions
    assert_rref_match(sim, stim_sim, n, "auto mode switch")


# ======================================================================
# Test 10: Large-scale RREF parity at n=128 (headline claim)
# ======================================================================

@pytest.mark.parametrize("p", [0.10, 0.25])
def test_rref_match_n128(p):
    """RREF-level parity against Stim at n=128.

    Backs the README claim that SparseGF2 is verified identical to Stim at
    system sizes up to n=128. Uses depth = 2n (256 layers) to
    keep the test runtime under a minute while still exercising a full
    MIPT circuit in the relevant phase.
    """
    n = 128
    n_layers = 2 * n  # 256 layers; shorter than the benchmark 8n to cap runtime
    schedule = generate_schedule(n, p, seed=1234, n_layers=n_layers)
    sim_sparse = run_sparse(n, schedule)
    sim_stim = run_stim(n, schedule)
    assert_rref_match(
        sim_sparse, sim_stim, n,
        msg=f"n=128 RREF parity at p={p}"
    )
