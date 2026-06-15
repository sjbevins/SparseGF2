"""Numba-JIT kernel parity with the pure-Python reference.

For every code path that has a JIT mirror in ``sparsegf2.core.numba_kernels``,
run the same circuit on a pure-Python ``SparseGF2`` *and* a JIT-compiled
``SparseGF2`` from a fresh ``|0^n⟩`` state and confirm the resulting
tableaux are bit-identical at every step.

We also check the data-structure invariants on the JIT path so that the
inverted indices stay consistent after long random sequences.

The pure-Python path is validated against Stim in the construction,
single-qubit, two-qubit, and measurement tests; therefore JIT ≡ pure-Python
implies JIT ≡ Stim.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2 import SparseGF2
from sparsegf2.core.symplectic import random_symplectic_2q

# Skip the whole module if numba isn't importable.
numba = pytest.importorskip("numba")


def _arrays_equal(a: SparseGF2, b: SparseGF2) -> bool:
    if not np.array_equal(a.plt, b.plt):
        return False
    return np.array_equal(a.to_symplectic(), b.to_symplectic())


def _index_invariants_hold(sim: SparseGF2) -> None:
    n, N = sim.n, sim.N
    for q in range(n):
        expected_inv = {int(r) for r in range(N) if int(sim.plt[r, q]) != 0}
        actual_inv = {int(sim.inv[q, k]) for k in range(int(sim.inv_len[q]))}
        assert expected_inv == actual_inv, f"inv[{q}] inconsistent"
        expected_invx = {int(r) for r in range(N) if (int(sim.plt[r, q]) >> 1) & 1}
        actual_invx = {int(sim.inv_x[q, k]) for k in range(int(sim.inv_x_len[q]))}
        assert expected_invx == actual_invx, f"inv_x[{q}] inconsistent"
    for r in range(N):
        expected_supp = {int(q) for q in range(n) if int(sim.plt[r, q]) != 0}
        actual_supp = {int(sim.supp_q[r, k]) for k in range(int(sim.supp_len[r]))}
        assert expected_supp == actual_supp, f"supp_q[{r}] inconsistent"


# ----------------------------------------------------------------------
# Construction parity (init_zero_state)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5, 8, 16])
def test_init_parity_numba_vs_python(n):
    py = SparseGF2(n, use_numba=False)
    nb = SparseGF2(n, use_numba=True)
    assert _arrays_equal(py, nb)
    assert np.array_equal(py.inv_len, nb.inv_len)
    assert np.array_equal(py.inv_x_len, nb.inv_x_len)
    assert np.array_equal(py.supp_len, nb.supp_len)


# ----------------------------------------------------------------------
# 1q gate parity
# ----------------------------------------------------------------------


@pytest.mark.parametrize("gate", ["apply_h", "apply_s", "apply_sqrt_x"])
@pytest.mark.parametrize("n", [3, 8])
def test_1q_gate_parity_numba_vs_python(gate, n):
    rng = np.random.default_rng(7 + hash(gate) % 100)
    py = SparseGF2(n, use_numba=False)
    nb = SparseGF2(n, use_numba=True)
    for _ in range(50):
        q = int(rng.integers(0, n))
        getattr(py, gate)(q)
        getattr(nb, gate)(q)
        assert _arrays_equal(py, nb), f"{gate}({q}) diverged"


# ----------------------------------------------------------------------
# 2q gate parity (named + random Sp(4))
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n, seed", [(4, 0), (8, 1), (16, 2)])
def test_2q_gate_parity_numba_vs_python(n, seed):
    rng_py = np.random.default_rng(seed)
    rng_nb = np.random.default_rng(seed)
    py = SparseGF2(n, use_numba=False)
    nb = SparseGF2(n, use_numba=True)
    for _ in range(80):
        qi, qj = rng_py.choice(n, size=2, replace=False).tolist()
        # Each rng generates the same matrix because they share a seed.
        _ = rng_nb.choice(n, size=2, replace=False)  # keep rngs in sync
        S = random_symplectic_2q(rng_py)
        _ = random_symplectic_2q(rng_nb)
        py.apply_gate_2q(int(qi), int(qj), S)
        nb.apply_gate_2q(int(qi), int(qj), S)
        assert _arrays_equal(py, nb)


# ----------------------------------------------------------------------
# Measurement parity (deterministic + non-deterministic)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n, seed", [(4, 100), (8, 101), (16, 102)])
def test_measurement_parity_numba_vs_python(n, seed):
    rng = np.random.default_rng(seed)
    py = SparseGF2(n, use_numba=False)
    nb = SparseGF2(n, use_numba=True)
    # Build up a generic state.
    for _ in range(3 * n):
        qi, qj = rng.choice(n, size=2, replace=False).tolist()
        S = random_symplectic_2q(rng)
        py.apply_gate_2q(int(qi), int(qj), S)
        nb.apply_gate_2q(int(qi), int(qj), S)
    # Measure each qubit in order. Outcome bits are random, but symplectics
    # must coincide since both paths are phase-free.
    for q in range(n):
        # Determinism flags must agree.
        assert py.is_deterministic_z(q) == nb.is_deterministic_z(q)
        py.measure_z(q, rng=np.random.default_rng(0))
        nb.measure_z(q, rng=np.random.default_rng(0))
        assert _arrays_equal(py, nb)


# ----------------------------------------------------------------------
# Long mixed circuit: gates + measurements
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n, n_steps, seed", [(8, 600, 33), (16, 800, 34), (24, 600, 35)])
def test_mixed_circuit_parity_numba_vs_python(n, n_steps, seed):
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed)
    py = SparseGF2(n, use_numba=False)
    nb = SparseGF2(n, use_numba=True)

    for _ in range(n_steps):
        r_a = rng_a.random()
        r_b = rng_b.random()
        assert r_a == r_b
        if r_a < 0.2:
            q = int(rng_a.integers(0, n))
            _ = int(rng_b.integers(0, n))
            # Outcome bit doesn't matter for symplectic parity; just project.
            py.measure_z(q)
            nb.measure_z(q)
        elif r_a < 0.6:
            g = int(rng_a.integers(0, 3))
            _ = int(rng_b.integers(0, 3))
            q = int(rng_a.integers(0, n))
            _ = int(rng_b.integers(0, n))
            (py.apply_h, py.apply_s, py.apply_sqrt_x)[g](q)
            (nb.apply_h, nb.apply_s, nb.apply_sqrt_x)[g](q)
        else:
            qi_a, qj_a = rng_a.choice(n, size=2, replace=False).tolist()
            _ = rng_b.choice(n, size=2, replace=False)
            S_a = random_symplectic_2q(rng_a)
            _ = random_symplectic_2q(rng_b)
            py.apply_gate_2q(int(qi_a), int(qj_a), S_a)
            nb.apply_gate_2q(int(qi_a), int(qj_a), S_a)
        assert _arrays_equal(py, nb)
    _index_invariants_hold(nb)


# ----------------------------------------------------------------------
# Numba flags: explicit toggles
# ----------------------------------------------------------------------


def test_use_numba_flag_resolves_to_default():
    sim = SparseGF2(2)
    # Default should be True since numba is importable in this venv.
    assert sim.use_numba is True


def test_use_numba_false_skips_jit():
    sim = SparseGF2(2, use_numba=False)
    assert sim.use_numba is False
    sim.apply_h(0)
    sim.measure_z(0)  # should not crash
