"""Two-qubit Clifford gate parity vs Stim.

Verifies that the ``SparseGF2`` two-qubit gate machinery
reproduces ``stim.TableauSimulator`` byte-for-byte for:

1. The named 2q Cliffords ``CX``, ``CZ``, ``SWAP`` at every (qi, qj)
   pair on a range of sizes.
2. *Every* element of the native 720-element ``Sp(4, F_2)`` enumeration
   applied at (qi, qj) = (0, 1) on a 4-qubit register.
3. Random sequences mixing named 1q gates with 2q gates sampled from
   our native sampler, step-by-step Stim parity.
4. Data-structure invariants on ``supp_q / inv / inv_x`` after long
   random mixed circuits.

The 2q gates are applied via :meth:`SparseGF2.apply_gate_2q`, which
takes a ``(4, 4)`` GF(2) symplectic produced by
:mod:`sparsegf2.core.symplectic`. Stim is invoked only as the
ground-truth cross-checker via ``stim.TableauSimulator.do_tableau``.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2 import CX_SYMP, CZ_SYMP, SWAP_SYMP, SparseGF2
from sparsegf2.core.symplectic import (
    enumerate_sp4,
    random_symplectic_2q,
)

stim = pytest.importorskip("stim")

from _stim_parity import fresh_stim, stim_symplectic
from _stim_parity import symp_to_stim_tableau as _symp_to_stim_tableau  # noqa: E402

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _check_indices_consistent(sim: SparseGF2) -> None:
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
# 1. Named 2q gates at every (qi, qj) pair
# ----------------------------------------------------------------------


NAMED_2Q = [
    ("CX", CX_SYMP, lambda s, qi, qj: s.cx(qi, qj)),
    ("CZ", CZ_SYMP, lambda s, qi, qj: s.cz(qi, qj)),
    ("SWAP", SWAP_SYMP, lambda s, qi, qj: s.swap(qi, qj)),
]


@pytest.mark.parametrize("n", [2, 3, 5, 8])
@pytest.mark.parametrize("name,symp,stim_apply", NAMED_2Q, ids=[g[0] for g in NAMED_2Q])
def test_named_2q_gate_at_each_pair(n, name, symp, stim_apply):
    for qi in range(n):
        for qj in range(n):
            if qi == qj:
                continue
            ours = SparseGF2(n)
            ours.apply_gate_2q(qi, qj, symp)

            theirs = fresh_stim(n)
            stim_apply(theirs, qi, qj)

            assert np.array_equal(ours.to_symplectic(), stim_symplectic(theirs, n)), (
                f"{name}({qi},{qj}) diverged on n={n}"
            )


# ----------------------------------------------------------------------
# 2. Every native 720-element Sp(4) symplectic applied to (0, 1)
# ----------------------------------------------------------------------


def test_every_sp4_element_matches_stim_after_application():
    """For each of the 720 native symplectics, apply it to (qi, qj) =
    (0, 1) of a 4-qubit |0^4⟩ and confirm the resulting tableau matches
    what Stim produces from the same symplectic-equivalent tableau."""
    n = 4
    cache = enumerate_sp4()

    for idx, S in enumerate(cache):
        ours = SparseGF2(n)
        ours.apply_gate_2q(0, 1, S)

        t = _symp_to_stim_tableau(S)
        st = fresh_stim(n)
        st.do_tableau(t, [0, 1])

        if not np.array_equal(ours.to_symplectic(), stim_symplectic(st, n)):
            pytest.fail(f"native Sp(4) element #{idx} diverged from Stim when applied to (0, 1)")


# ----------------------------------------------------------------------
# 3. Mixed random 1q + 2q sequences, step-by-step parity
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, n_steps, seed",
    [
        (4, 200, 0),
        (8, 400, 1),
        (16, 600, 2),
        (32, 400, 3),
    ],
)
def test_random_mixed_circuit_step_parity(n, n_steps, seed):
    rng = np.random.default_rng(seed)
    ours = SparseGF2(n)
    theirs = fresh_stim(n)

    gate_1q_methods = [
        ("apply_h", lambda s, q: s.h(q)),
        ("apply_s", lambda s, q: s.s(q)),
        ("apply_sqrt_x", lambda s, q: s.sqrt_x(q)),
    ]

    for step in range(n_steps):
        # 60% 1q gates, 40% 2q gates
        if rng.random() < 0.6:
            gname, stim_fn = gate_1q_methods[int(rng.integers(0, 3))]
            q = int(rng.integers(0, n))
            getattr(ours, gname)(q)
            stim_fn(theirs, q)
        else:
            qi, qj = rng.choice(n, size=2, replace=False).tolist()
            S = random_symplectic_2q(rng)
            ours.apply_gate_2q(int(qi), int(qj), S)
            t = _symp_to_stim_tableau(S)
            theirs.do_tableau(t, [int(qi), int(qj)])

        if not np.array_equal(ours.to_symplectic(), stim_symplectic(theirs, n)):
            pytest.fail(f"step {step} diverged from Stim at n={n}, seed={seed}")


# ----------------------------------------------------------------------
# 4. Data-structure invariants after long mixed sequences
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n, seed", [(6, 11), (12, 22), (24, 33)])
def test_indices_stay_consistent_after_mixed_sequence(n, seed):
    rng = np.random.default_rng(seed)
    sim = SparseGF2(n)

    for _ in range(400):
        if rng.random() < 0.5:
            g = int(rng.integers(0, 3))
            q = int(rng.integers(0, n))
            (sim.apply_h, sim.apply_s, sim.apply_sqrt_x)[g](q)
        else:
            qi, qj = rng.choice(n, size=2, replace=False).tolist()
            S = random_symplectic_2q(rng)
            sim.apply_gate_2q(int(qi), int(qj), S)

    _check_indices_consistent(sim)


# ----------------------------------------------------------------------
# 5. Boundary checks
# ----------------------------------------------------------------------


def test_two_qubit_boundary_errors():
    sim = SparseGF2(3)
    with pytest.raises(ValueError):
        sim.apply_cx(0, 0)
    with pytest.raises(IndexError):
        sim.apply_cx(-1, 1)
    with pytest.raises(IndexError):
        sim.apply_cx(0, 3)


def test_apply_gate_2q_validates_shape():
    sim = SparseGF2(3)
    bad = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        sim.apply_gate_2q(0, 1, bad)


def test_apply_gate_2q_rejects_non_symplectic():
    """Safety net from ROADMAP §5.5a-1. A non-symplectic 4x4 matrix
    must raise instead of silently corrupting the tableau."""
    sim = SparseGF2(4)
    # All-zeros is not symplectic.
    with pytest.raises(ValueError, match="not symplectic"):
        sim.apply_gate_2q(0, 1, np.zeros((4, 4), dtype=np.uint8))
    # All-ones is not symplectic either.
    with pytest.raises(ValueError, match="not symplectic"):
        sim.apply_gate_2q(0, 1, np.ones((4, 4), dtype=np.uint8))
    # The legitimate named gates still work.
    from sparsegf2 import CX_SYMP, CZ_SYMP, SWAP_SYMP

    sim.apply_gate_2q(0, 1, CX_SYMP)
    sim.apply_gate_2q(1, 2, CZ_SYMP)
    sim.apply_gate_2q(2, 3, SWAP_SYMP)
