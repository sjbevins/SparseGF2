"""phase-free stabilizer observables.

Verifies that :mod:`sparsegf2.core.observables` computes entanglement
entropy, mutual information, tripartite mutual information, code
dimension, and weight diagnostics correctly.

The load-bearing identity to check (Fattal et al. 2004):

    S(A) = rank_{F_2}([X|Z]|_A) - |A|

is established empirically against Stim's separate entropy machinery
(via ``stim.PauliString`` projections + dense GF(2) rank).
"""

from __future__ import annotations

import numpy as np
import pytest

stim = pytest.importorskip("stim")  # skipped when the optional stim extra is absent

from _stim_parity import fresh_stim, stim_symplectic
from sparsegf2 import SparseGF2, from_bell_purification
from sparsegf2.core.observables import (
    average_stabilizer_weight,
    code_dimension,
    entanglement_entropy,
    generator_weights,
    mutual_information,
    stabilizer_weight_spectrum,
    subsystem_rank,
    tripartite_mutual_info,
)
from sparsegf2.core.symplectic import random_symplectic_2q

stim = pytest.importorskip("stim")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _gf2_rank_dense(M: np.ndarray) -> int:
    """Naive GF(2) rank, an independent reference for the tests."""
    A = (np.asarray(M, dtype=np.uint8) & 1).copy()
    n_rows, n_cols = A.shape
    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        r_pivot = -1
        for r in range(pivot_row, n_rows):
            if A[r, col]:
                r_pivot = r
                break
        if r_pivot < 0:
            continue
        if r_pivot != pivot_row:
            A[[pivot_row, r_pivot]] = A[[r_pivot, pivot_row]]
        for r in range(n_rows):
            if r != pivot_row and A[r, col]:
                A[r] ^= A[pivot_row]
        pivot_row += 1
    return pivot_row


def _ref_entropy_from_stim(sim_stim, subsystem, n):
    """Reference entropy via Stim's forward tableau."""
    M = stim_symplectic(sim_stim, n)
    stab = M[n:]
    cols = np.concatenate(
        [np.array(subsystem, dtype=np.int64), np.array(subsystem, dtype=np.int64) + n]
    )
    if cols.size == 0:
        return 0
    sub = stab[:, cols]
    return _gf2_rank_dense(sub) - len(subsystem)


# ----------------------------------------------------------------------
# subsystem_rank + entropy basics
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_zero_state_entropy_zero(n):
    """|0^n⟩ is a product state; S(A) = 0 for every A."""
    sim = SparseGF2(n)
    for size in range(n + 1):
        S = entanglement_entropy(sim, range(size))
        assert S == 0, f"|0^n⟩ should have S = 0 on every subsystem, got S({size})={S}"


@pytest.mark.parametrize("n_system", [1, 2, 3, 5])
def test_bell_purification_system_entropy_equals_n_system(n_system):
    """Bell-purification of n_system qubits has S(system) = n_system."""
    sim = from_bell_purification(n_system)
    S = entanglement_entropy(sim, range(n_system))
    assert n_system == S


@pytest.mark.parametrize("n_system", [1, 2, 3, 5])
def test_bell_purification_code_dimension(n_system):
    """code_dimension(sim, n_system) must equal n_system for a fresh Bell-pair purification."""
    sim = from_bell_purification(n_system)
    k = code_dimension(sim, n_system)
    assert k == n_system


def test_code_dimension_rejects_mismatched_n_system():
    sim = SparseGF2(6)
    with pytest.raises(ValueError, match="2 \\* n_system"):
        code_dimension(sim, 5)


# ----------------------------------------------------------------------
# Stim parity on entropy across random circuits
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n,seed", [(4, 0), (6, 1), (8, 2)])
def test_entropy_stim_parity_random_circuit(n, seed):
    """Apply random gates + measurements; entropy must match Stim's
    forward-tableau computation."""
    rng = np.random.default_rng(seed)
    ours = SparseGF2(n)
    theirs = fresh_stim(n)

    def _symp_to_stim(S):
        rows = [S[i] for i in range(4)]

        def to_ps(row):
            xs = row[:2].astype(bool)
            zs = row[2:].astype(bool)
            return stim.PauliString.from_numpy(xs=xs, zs=zs)

        return stim.Tableau.from_conjugated_generators(
            xs=[to_ps(rows[0]), to_ps(rows[1])],
            zs=[to_ps(rows[2]), to_ps(rows[3])],
        )

    for _ in range(50):
        r = rng.random()
        if r < 0.2:
            q = int(rng.integers(0, n))
            ours.measure_z(q, rng=rng)
            theirs.measure(q)
        elif r < 0.6:
            q = int(rng.integers(0, n))
            ours.apply_h(q)
            theirs.h(q)
        else:
            qi, qj = rng.choice(n, size=2, replace=False).tolist()
            S = random_symplectic_2q(rng)
            ours.apply_gate_2q(int(qi), int(qj), S)
            theirs.do_tableau(_symp_to_stim(S), [int(qi), int(qj)])
    # Sweep all contiguous subsystems
    for size in range(n + 1):
        A = list(range(size))
        S_ours = entanglement_entropy(ours, A)
        S_ref = _ref_entropy_from_stim(theirs, A, n)
        assert S_ours == S_ref, f"n={n} seed={seed} |A|={size}: ours={S_ours} ref={S_ref}"


# ----------------------------------------------------------------------
# Mutual / tripartite mutual information
# ----------------------------------------------------------------------


def test_mutual_info_disjoint_required():
    sim = SparseGF2(4)
    with pytest.raises(ValueError, match="disjoint"):
        mutual_information(sim, [0, 1], [1, 2])


def test_mutual_info_bell_pair():
    """In a Bell pair |Φ⁺⟩ = (|00⟩ + |11⟩)/√2, I(A:B) = 2 between the
    two qubits (S(A) = S(B) = 1, S(A∪B) = 0)."""
    sim = SparseGF2(2)
    sim.apply_h(0)
    sim.apply_cx(0, 1)
    info = mutual_information(sim, [0], [1])
    assert info == 2


def test_tripartite_mutual_info_disjoint_required():
    sim = SparseGF2(6)
    with pytest.raises(ValueError, match="disjoint"):
        tripartite_mutual_info(sim, [0, 1], [1], [2])
    with pytest.raises(ValueError, match="disjoint"):
        tripartite_mutual_info(sim, [0], [1], [0, 2])


def test_tmi_volume_law_strictly_negative_on_average():
    """Volume-law circuits give *strictly* negative TMI on average.

    For a deep random-Clifford circuit (no measurements), the
    stabilizer subspace is uniformly random over the full Sp(2n, F_2),
    and the resulting state is volume-law-entangled. A 4-partition
    (A, B, C, D) with each ≈ n/4 qubits has
        I_3(A : B : C) ≈ -2·|A|  (Page-curve calculation)
    asymptotically. Even at n=12 the mean should be ≤ -1 over a
    handful of seeds, a stronger contract than "≤ 0 on each seed"
    (which a product state would also satisfy).
    """
    n = 12
    block = 3  # so we have 4 disjoint blocks of equal size
    samples = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        sim = SparseGF2(n)
        # Deep enough to saturate volume-law behavior at n=12.
        for _ in range(6 * n):
            qi, qj = rng.choice(n, size=2, replace=False).tolist()
            S = random_symplectic_2q(rng)
            sim.apply_gate_2q(int(qi), int(qj), S)
        A = list(range(0, block))
        B = list(range(block, 2 * block))
        C = list(range(2 * block, 3 * block))
        samples.append(tripartite_mutual_info(sim, A, B, C))
    mean_I3 = float(np.mean(samples))
    # All-product state would give I3 = 0; volume-law at this depth
    # should give I3 ≈ -2*block = -6. Demand mean < -1 over the seeds.
    assert mean_I3 < -1.0, (
        f"Mean TMI over 8 deep random seeds at n={n}, |A|=|B|=|C|={block} "
        f"should be < -1 (volume-law); got {mean_I3:.2f}, samples={samples}"
    )


# ----------------------------------------------------------------------
# Generator-weight diagnostics
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 5, 8])
def test_zero_state_generator_weights(n):
    """|0^n⟩: every generator has weight 1 (single X_i destab or Z_i stab)."""
    sim = SparseGF2(n)
    w = generator_weights(sim)
    assert w.shape == (2 * n,)
    assert np.all(w == 1)


@pytest.mark.parametrize("n", [3, 5, 8])
def test_zero_state_stab_weight_spectrum(n):
    """|0^n⟩: stabilizer spectrum is all weight 1."""
    sim = SparseGF2(n)
    spec = stabilizer_weight_spectrum(sim)
    assert spec.shape == (n + 1,)
    assert spec[1] == n
    assert spec.sum() == n
    # All non-1 buckets are zero
    assert spec[0] == 0
    for w in range(2, n + 1):
        assert spec[w] == 0


def test_ghz_state_weight_spectrum():
    """An n-qubit GHZ stabilizer state has 1 weight-n row + (n-1) weight-2."""
    n = 5
    sim = SparseGF2(n)
    sim.apply_h(0)
    for q in range(1, n):
        sim.apply_cx(0, q)
    spec = stabilizer_weight_spectrum(sim)
    assert spec[2] == n - 1, f"expected {n - 1} weight-2 rows, got {spec[2]}"
    assert spec[n] == 1, f"expected 1 weight-{n} row, got {spec[n]}"


def test_average_stab_weight_zero_state():
    sim = SparseGF2(8)
    assert average_stabilizer_weight(sim) == 1.0


def test_average_stab_weight_empty():
    sim = SparseGF2(0)
    assert average_stabilizer_weight(sim) == 0.0


# ----------------------------------------------------------------------
# Numba-vs-Python parity for the rank kernel
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n,seed", [(4, 100), (6, 101), (8, 102)])
def test_observables_numba_vs_python_parity(n, seed):
    """Observable values match between use_numba=True and use_numba=False."""
    rng = np.random.default_rng(seed)
    sim_jit = SparseGF2(n, use_numba=True)
    sim_py = SparseGF2(n, use_numba=False)
    for _ in range(20):
        qi, qj = rng.choice(n, size=2, replace=False).tolist()
        S = random_symplectic_2q(rng)
        sim_jit.apply_gate_2q(int(qi), int(qj), S)
        sim_py.apply_gate_2q(int(qi), int(qj), S)
    for size in range(n + 1):
        A = list(range(size))
        assert entanglement_entropy(sim_jit, A) == entanglement_entropy(sim_py, A)
        assert subsystem_rank(sim_jit, A) == subsystem_rank(sim_py, A)
