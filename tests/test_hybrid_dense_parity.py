"""Hybrid (sparse/dense) mode parity.

The hybrid path mirrors the tableau into bit-packed arrays and runs the dense
kernels when the state is volume-law, switching back to sparse when it thins.
This must be a faithful drop-in: identical physical state at every step.

What "identical" means here (and why):

* ``hybrid`` vs pure ``sparse`` -> **same stabilizer group**, asserted
  byte-for-byte via :meth:`SparseGF2.canonical_form` (the GF(2) RREF of the
  stabilizer block, a canonical representative independent of generator choice).
  The dense Z-measurement picks a different (equally valid) pivot than the
  sparse min-weight kernel, so the *generators* differ while the *group*, and
  every group-invariant observable, is identical.
* both vs **Stim** -> :func:`_stim_parity.assert_states_equal` (same RREF).
* every group-invariant observable (entropy, subsystem rank, code dimension)
  agrees across modes.

The stabilizer-group / kernel parity needs no Stim and always runs (both the
numba and the pure-Python dense kernels); only the direct Stim cross-checks are
gated on Stim being installed. The switch metric ``a_bar`` is the total generator
weight summed over all ``2n`` rows, divided by ``n`` (the average column weight;
max ~2n), and the threshold has a floor of 16, so dense mode engages once a
scrambled state pushes ``a_bar`` past 16, reliably for ``n >= 17``.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from sparsegf2 import SparseGF2, random_symplectic
from sparsegf2.core.linalg_gf2 import gf2_rref
from sparsegf2.core.observables import (
    entanglement_entropy,
    subsystem_rank,
)

try:
    import stim  # noqa: F401

    _HAS_STIM = True
except ImportError:  # pragma: no cover - exercised only on the bare floor
    _HAS_STIM = False

needs_stim = pytest.mark.skipif(not _HAS_STIM, reason="stim not installed")

_G1 = ("apply_h", "apply_s", "apply_sqrt_x")
# n values spanning: just above the threshold floor (17), off-boundary (31, 63,
# 65), and the exact 64-bit word boundary (64).
_DENSE_NS = [17, 31, 63, 64, 65]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _scramble_to_dense(sim, rng, n, rounds=10):
    """Gate-only brickwork until the hybrid sim has entered dense mode."""
    for _ in range(rounds):
        for off in (0, 1):
            for q in range(off, n - 1, 2):
                S = random_symplectic(2, np.random.default_rng(int(rng.integers(1 << 30))))
                sim.apply_gate_2q(q, q + 1, S)


def _group_rref(symp):
    """RREF of a (2n,2n) symplectic's stabilizer block: a canonical, mode-
    independent fingerprint of the stabilizer group."""
    symp = np.asarray(symp, dtype=np.uint8)
    return gf2_rref(symp[symp.shape[0] // 2 :].copy())


def _packed_stab_rref(xp, zp, n):
    """Stabilizer-group RREF read straight from the bit-packed arrays."""
    stab = np.zeros((n, 2 * n), np.uint8)
    for i in range(n):
        r = n + i
        for q in range(n):
            w, b = q >> 6, q & 63
            stab[i, q] = (int(xp[r, w]) >> b) & 1
            stab[i, n + q] = (int(zp[r, w]) >> b) & 1
    return gf2_rref(stab)


def _packed_row_weights_oracle(xp, zp, n):
    """Independent bit-by-bit per-row weight (no popcount), the oracle for the
    row_weights kernels, catching a shared tail-mask bug they could both have."""
    out = np.zeros(xp.shape[0], dtype=np.int32)
    for r in range(xp.shape[0]):
        for q in range(n):
            w, b = q >> 6, q & 63
            if ((int(xp[r, w]) >> b) & 1) or ((int(zp[r, w]) >> b) & 1):
                out[r] += 1
    return out


# ----------------------------------------------------------------------
# 1. Kernel parity (no Stim): numba-dense == python-dense == sparse group.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 8, 16, 17, 31, 63, 64, 65])
def test_dense_kernels_numba_vs_python_vs_sparse(n):
    """Drive identical random circuits through the sparse engine and two raw
    dense packed states (one via the @njit kernels, one via the pure-Python
    mirrors). The two dense states must stay byte-identical to each other, and
    both must match the sparse engine's stabilizer group, at every step."""
    import sparsegf2.core.numba_kernels as nk
    from sparsegf2.core.sparse_tableau import (
        H_SYMP,
        S_SYMP,
        SQRT_X_SYMP,
        _apply_1q_packed,
        _apply_2q_packed,
        _build_1q_lut,
        _build_2q_lut,
        _measure_z_packed,
        _plt_to_packed,
        _row_weights_packed,
    )

    lut1 = {0: _build_1q_lut(H_SYMP), 1: _build_1q_lut(S_SYMP), 2: _build_1q_lut(SQRT_X_SYMP)}
    rng = np.random.default_rng(500 + n)
    nw = (n + 63) // 64
    sp = SparseGF2(n, use_numba=True)
    xj = np.zeros((2 * n, nw), np.uint64)
    zj = np.zeros((2 * n, nw), np.uint64)
    nk.plt_to_packed_jit(sp.plt, xj, zj, n)
    xp = np.zeros((2 * n, nw), np.uint64)
    zp = np.zeros((2 * n, nw), np.uint64)
    _plt_to_packed(sp.plt, xp, zp, n)

    for _ in range(250):
        r = rng.random()
        if r < 0.25:
            q = int(rng.integers(n))
            sp.measure_z(q, rng=np.random.default_rng(3))
            nk.measure_z_packed_jit(xj, zj, q)
            _measure_z_packed(xp, zp, q)
        elif r < 0.55:
            g = int(rng.integers(3))
            q = int(rng.integers(n))
            getattr(sp, _G1[g])(q)
            nk.apply_1q_packed_jit(xj, zj, q, lut1[g])
            _apply_1q_packed(xp, zp, q, lut1[g])
        else:
            qi, qj = (int(x) for x in rng.choice(n, 2, replace=False))
            S = random_symplectic(2, np.random.default_rng(int(rng.integers(1 << 30))))
            lut = _build_2q_lut(S)
            sp.apply_gate_2q(qi, qj, S)
            nk.apply_2q_packed_jit(xj, zj, qi, qj, lut)
            _apply_2q_packed(xp, zp, qi, qj, lut)
        # numba-dense and python-dense byte-identical
        assert np.array_equal(xj, xp) and np.array_equal(zj, zp)
        # both describe the sparse engine's stabilizer group
        assert np.array_equal(_packed_stab_rref(xp, zp, n), sp.canonical_form())
    # weights kernels agree with each other AND with an independent bit-by-bit
    # oracle on the same packed state (the dense generators differ from the
    # sparse engine's after measurements, so sp.supp_len is NOT the oracle here).
    jit_w = nk.row_weights_packed_jit(xj, zj, n)
    py_w = _row_weights_packed(xp, zp, n)
    assert np.array_equal(jit_w, py_w)
    assert np.array_equal(jit_w, _packed_row_weights_oracle(xp, zp, n))


# ----------------------------------------------------------------------
# 2. Full hybrid engine, BOTH backends, ALL ops in dense mode (no Stim).
# ----------------------------------------------------------------------


@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("n", _DENSE_NS)
def test_hybrid_all_ops_in_dense_mode(n, use_numba):
    """Scramble a hybrid sim into dense mode, then exercise every op kind
    (1q, 2q, cx/cz/swap, measure_z/x/y, reset_z/x/y, apply_clifford), asserting
    the hybrid engine's stabilizer group matches a pure-sparse twin after each.
    Forces both the numba and pure-Python dense kernels."""
    rng = np.random.default_rng(7000 + n + (1 if use_numba else 0))
    hyb = SparseGF2(n, use_numba=use_numba, hybrid=True)
    spr = SparseGF2(n, use_numba=use_numba, hybrid=False)
    # scramble both identically (gate-only) -> hyb enters dense mode
    for _ in range(10):
        for off in (0, 1):
            for q in range(off, n - 1, 2):
                S = random_symplectic(2, np.random.default_rng(int(rng.integers(1 << 30))))
                hyb.apply_gate_2q(q, q + 1, S)
                spr.apply_gate_2q(q, q + 1, S)
    assert hyb._dense_mode, f"n={n} use_numba={use_numba}: never densified"
    assert np.array_equal(hyb.canonical_form(), spr.canonical_form())

    def both(fn):
        fn(hyb)
        fn(spr)
        assert np.array_equal(hyb.canonical_form(), spr.canonical_form())

    # 1q + 2q + named 2q while dense
    both(lambda s: s.apply_h(0))
    both(lambda s: s.apply_s(1 % n))
    both(lambda s: s.apply_cx(0, 1))
    both(lambda s: s.apply_cz(1 % n, 2 % n))
    both(lambda s: s.apply_swap(0, 3 % n))
    S = random_symplectic(2, np.random.default_rng(99))
    both(lambda s: s.apply_gate_2q(2 % n, 4 % n, S))
    # measurements in all three bases + resets while dense
    both(lambda s: s.measure_z(0, rng=np.random.default_rng(11)))
    both(lambda s: s.measure_x(1 % n, rng=np.random.default_rng(12)))
    both(lambda s: s.measure_y(2 % n, rng=np.random.default_rng(13)))
    both(lambda s: s.reset_z(3 % n, rng=np.random.default_rng(14)))
    both(lambda s: s.reset_x(4 % n, rng=np.random.default_rng(15)))
    both(lambda s: s.reset_y(0, rng=np.random.default_rng(16)))
    # determinism predicate must agree in dense mode
    for q in (0, 1 % n, n - 1):
        assert hyb.is_deterministic_z(q) == spr.is_deterministic_z(q)
    # apply_clifford while dense (densify -> rebuild -> sparse round trip)
    M = random_symplectic(3, np.random.default_rng(21))
    both(lambda s: s.apply_clifford(M, qubits=[0, 1 % n, 2 % n]))

    # observables agree on the final state
    for size in (1, n // 2, n):
        A = list(range(size))
        assert subsystem_rank(hyb, A) == subsystem_rank(spr, A)
        assert entanglement_entropy(hyb, A) == entanglement_entropy(spr, A)


@pytest.mark.parametrize("use_numba", [True, False])
def test_dense_mode_supp_len_value(use_numba):
    """The ``supp_len`` property in dense mode must return the live packed-array
    weights (it backs both a_bar, the switch metric, and the weight-spectrum
    observables), NOT the stale sparse backing array left over from before the
    switch. Pins the dense-branch value directly against the independent oracle."""
    n = 24
    rng = np.random.default_rng(123)
    sim = SparseGF2(n, use_numba=use_numba, hybrid=True)
    _scramble_to_dense(sim, rng, n)
    assert sim._dense_mode
    # A few more dense-mode gates so the stale sparse weights (frozen at the
    # switch point) provably diverge from the current dense weights.
    for _ in range(5):
        qi, qj = (int(x) for x in rng.choice(n, 2, replace=False))
        sim.apply_gate_2q(
            qi, qj, random_symplectic(2, np.random.default_rng(int(rng.integers(1 << 30))))
        )
    oracle = _packed_row_weights_oracle(sim.x_packed, sim.z_packed, n)
    assert np.array_equal(np.asarray(sim.supp_len), oracle)  # property reads dense weights
    assert sim._active_count() == oracle.sum() / n  # a_bar reads dense weights
    # and it is NOT the stale sparse backing array (so the test would catch a
    # regression that returned self._supp_len in dense mode)
    assert not np.array_equal(np.asarray(sim._supp_len), oracle)


# ----------------------------------------------------------------------
# 3. dense -> sparse switch-back rebuild (no Stim).
# ----------------------------------------------------------------------


@pytest.mark.parametrize("use_numba", [True, False])
def test_dense_to_sparse_switchback(use_numba):
    """Force a dense->sparse rebuild and verify the reconstructed sparse engine
    is correct and keeps matching a twin through subsequent sparse-mode ops."""
    n = 24
    rng = np.random.default_rng(31)
    hyb = SparseGF2(n, use_numba=use_numba, hybrid=True)
    spr = SparseGF2(n, use_numba=use_numba, hybrid=False)
    for _ in range(10):
        for off in (0, 1):
            for q in range(off, n - 1, 2):
                S = random_symplectic(2, np.random.default_rng(int(rng.integers(1 << 30))))
                hyb.apply_gate_2q(q, q + 1, S)
                spr.apply_gate_2q(q, q + 1, S)
    assert hyb._dense_mode
    before = hyb.canonical_form()
    hyb._switch_to_sparse()  # force the dense->sparse rebuild
    assert not hyb._dense_mode
    assert np.array_equal(hyb.canonical_form(), before)
    assert np.array_equal(hyb.canonical_form(), spr.canonical_form())
    # the rebuilt sparse indices must support further correct sparse-mode ops
    for _ in range(40):
        if rng.random() < 0.3:
            q = int(rng.integers(n))
            hyb.measure_z(q, rng=np.random.default_rng(5))
            spr.measure_z(q, rng=np.random.default_rng(5))
        else:
            qi, qj = (int(x) for x in rng.choice(n, 2, replace=False))
            S = random_symplectic(2, np.random.default_rng(int(rng.integers(1 << 30))))
            hyb.apply_gate_2q(qi, qj, S)
            spr.apply_gate_2q(qi, qj, S)
        assert np.array_equal(hyb.canonical_form(), spr.canonical_form())


def test_automatic_dense_to_sparse_switch():
    """Exercise the a_bar-driven (automatic, not manually forced) dense->sparse
    transition through _maybe_check_mode_switch, and verify the rebuilt sparse
    engine stays correct. The dense first-pivot keeps a_bar 'sticky', so we raise
    the exit threshold to make the *periodic check* trip the switch, testing the
    automatic code path, not the physics of when it would naturally fire."""
    n = 24
    rng = np.random.default_rng(91)
    hyb = SparseGF2(n, hybrid=True)
    spr = SparseGF2(n, hybrid=False)
    for _ in range(10):
        for off in (0, 1):
            for q in range(off, n - 1, 2):
                S = random_symplectic(2, np.random.default_rng(int(rng.integers(1 << 30))))
                hyb.apply_gate_2q(q, q + 1, S)
                spr.apply_gate_2q(q, q + 1, S)
    assert hyb._dense_mode
    # Make the current a_bar fall below the exit threshold/2, then advance past a
    # check boundary with low-impact ops so the periodic check auto-switches.
    hyb._dense_threshold = 200  # exit threshold/2 = 100 > current a_bar
    hyb._ops_since_check = 0
    for _ in range(hyb._check_interval + 1):
        hyb.measure_z(0, rng=np.random.default_rng(0))
        spr.measure_z(0, rng=np.random.default_rng(0))
    assert not hyb._dense_mode, "automatic dense->sparse switch never fired"
    assert np.array_equal(hyb.canonical_form(), spr.canonical_form())
    # rebuilt sparse indices must support further correct ops
    for _ in range(20):
        qi, qj = (int(x) for x in rng.choice(n, 2, replace=False))
        S = random_symplectic(2, np.random.default_rng(int(rng.integers(1 << 30))))
        hyb.apply_gate_2q(qi, qj, S)
        spr.apply_gate_2q(qi, qj, S)
        assert np.array_equal(hyb.canonical_form(), spr.canonical_form())


def test_row_weights_ignores_tail_garbage():
    """The row_weights tail mask is defensive (the gate/measure kernels never set
    bits beyond column n), so assert it directly: a stray bit past column n in the
    last word must NOT be counted, by both backends, matching the n-bounded oracle."""
    import sparsegf2.core.numba_kernels as nk
    from sparsegf2.core.sparse_tableau import _row_weights_packed

    n = 65  # last word holds qubit 64 (bit 0); bits 1..63 are out-of-range garbage
    nw = (n + 63) // 64
    xp = np.zeros((2 * n, nw), np.uint64)
    zp = np.zeros((2 * n, nw), np.uint64)
    xp[0, 0] |= np.uint64(1)  # real bit: qubit 0 on row 0
    xp[0, nw - 1] |= np.uint64(1) << np.uint64((n & 63) + 7)  # garbage past column n
    zp[1, nw - 1] |= np.uint64(1) << np.uint64(63)  # garbage in the other plane
    oracle = _packed_row_weights_oracle(xp, zp, n)
    assert oracle[0] == 1 and oracle[1] == 0  # only the real bit counts
    assert np.array_equal(nk.row_weights_packed_jit(xp, zp, n), oracle)
    assert np.array_equal(_row_weights_packed(xp, zp, n), oracle)


# ----------------------------------------------------------------------
# 4. copy / pickle of a dense-mode instance (no Stim), both backends.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("use_numba", [True, False])
def test_dense_mode_copy_roundtrip(use_numba):
    n = 24
    rng = np.random.default_rng(3)
    sim = SparseGF2(n, use_numba=use_numba, hybrid=True)
    _scramble_to_dense(sim, rng, n)
    assert sim._dense_mode

    clone = sim.copy()
    assert clone._dense_mode and clone.hybrid
    assert np.array_equal(clone.canonical_form(), sim.canonical_form())
    before = sim.canonical_form()
    clone.apply_gate_2q(0, 1, random_symplectic(2, np.random.default_rng(9)))
    clone.apply_h(2)
    clone.measure_z(3, rng=np.random.default_rng(1))
    assert np.array_equal(sim.canonical_form(), before)  # parent untouched


@pytest.mark.parametrize("use_numba", [True, False])
def test_dense_mode_pickle_roundtrip(use_numba):
    n = 20
    rng = np.random.default_rng(5)
    sim = SparseGF2(n, use_numba=use_numba, hybrid=True)
    _scramble_to_dense(sim, rng, n)
    assert sim._dense_mode

    restored = pickle.loads(pickle.dumps(sim))
    assert restored._dense_mode and restored.hybrid
    assert np.array_equal(restored.canonical_form(), sim.canonical_form())
    assert np.array_equal(restored.to_symplectic(), sim.to_symplectic())


def test_non_hybrid_pickle_handles_none_packed():
    """A pure-sparse instance (x_packed/z_packed = None) pickles cleanly."""
    sim = SparseGF2(12, hybrid=False)
    sim.apply_cx(0, 1)
    restored = pickle.loads(pickle.dumps(sim))
    assert restored.x_packed is None and restored.z_packed is None
    assert not restored.hybrid and not restored._dense_mode
    assert np.array_equal(restored.canonical_form(), sim.canonical_form())


# ----------------------------------------------------------------------
# 5. Direct Stim cross-check (gated on stim).
# ----------------------------------------------------------------------


@needs_stim
@pytest.mark.parametrize("n", [8, 17, 31, 64, 65])
def test_hybrid_vs_sparse_vs_stim(n):
    """hybrid, sparse and Stim driven in lock-step: hybrid==sparse (canonical
    form) AND both==Stim (RREF), at every step, with measurements + gates."""
    from _stim_parity import assert_states_equal, fresh_stim, stim_symplectic
    from _stim_parity import symp_to_stim_tableau as _symp_to_stim_tableau

    def stim_1q(st, g, q):
        st.do(stim.Circuit(f"{('H', 'S', 'SQRT_X')[g]} {q}"))

    rng = np.random.default_rng(2000 + n)
    hyb = SparseGF2(n, hybrid=True)
    spr = SparseGF2(n, hybrid=False)
    ref = fresh_stim(n)
    saw_dense = False
    for _ in range(400):
        r = rng.random()
        if r < 0.08:  # low measurement rate -> the state densifies
            q = int(rng.integers(n))
            det = hyb.is_deterministic_z(q)
            assert det == spr.is_deterministic_z(q)
            assert det == (ref.peek_z(q) != 0)
            hyb.measure_z(q, rng=np.random.default_rng(7))
            spr.measure_z(q, rng=np.random.default_rng(7))
            ref.measure(q)
        elif r < 0.4:
            g = int(rng.integers(3))
            q = int(rng.integers(n))
            getattr(hyb, _G1[g])(q)
            getattr(spr, _G1[g])(q)
            stim_1q(ref, g, q)
        else:
            qi, qj = (int(x) for x in rng.choice(n, 2, replace=False))
            S = random_symplectic(2, np.random.default_rng(int(rng.integers(1 << 30))))
            hyb.apply_gate_2q(qi, qj, S)
            spr.apply_gate_2q(qi, qj, S)
            ref.do_tableau(_symp_to_stim_tableau(S), [qi, qj])
        saw_dense = saw_dense or hyb._dense_mode
        assert np.array_equal(hyb.canonical_form(), spr.canonical_form())
        ref_symp = stim_symplectic(ref, n)
        assert_states_equal(hyb.to_symplectic(), ref_symp, n)
        assert_states_equal(spr.to_symplectic(), ref_symp, n)
    if n >= 17:
        assert saw_dense, "dense mode was never entered: the dense path went untested"


# ----------------------------------------------------------------------
# 6. Config-level parity across architectures (no Stim), with real tableaux.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("gating", ["brickwork", "random_edge", "random_pool"])
@pytest.mark.parametrize("measurement", ["bernoulli", "gated", "uniform_count"])
@pytest.mark.parametrize("picture", ["purification", "single_ref"])
@pytest.mark.parametrize("p", [0.05, 0.3])
def test_config_level_hybrid_parity(gating, measurement, picture, p):
    """A full circuit run with hybrid=True yields identical observables and an
    identical final stabilizer group to hybrid=False, every architecture."""
    from sparsegf2.circuits import CircuitConfig, simulate

    # This checks runner-pipeline parity across the architecture grid. Whether the
    # internal sim densifies is incidental here (and at p=0.3 it stays sparse,
    # area-law); the runner actually entering dense mode is proven separately by
    # test_runner_actually_engages_dense_mode. The load-bearing assertion is the
    # final stabilizer-group RREF equality.
    kw = dict(
        graph_spec="complete",
        n=24,
        picture=picture,
        gating_mode=gating,
        matching_mode="round_robin",
        measurement_mode=measurement,
        p=p,
        depth_mode="O(n)",
        depth_factor=4,
    )
    for seed in (0, 1):
        r_sparse = simulate(CircuitConfig(**kw, hybrid=False), sample_seed=seed, save_tableau=True)
        r_hybrid = simulate(CircuitConfig(**kw, hybrid=True), sample_seed=seed, save_tableau=True)
        # Compare the picture's order parameter (the OTHER is None for this
        # picture, so guard against a vacuous None == None).
        if picture == "purification":
            assert r_sparse.code_dimension is not None
            assert r_sparse.code_dimension == r_hybrid.code_dimension
        else:  # single_ref
            assert r_sparse.ref_entropy is not None
            assert r_sparse.ref_entropy == r_hybrid.ref_entropy
        assert r_sparse.entropy_half_cut == r_hybrid.entropy_half_cut
        # final tableaux describe the SAME stabilizer group (compare via RREF,
        # not raw equality, since the dense pivot choice yields different generators)
        assert r_sparse.final_tableau is not None and r_hybrid.final_tableau is not None
        assert np.array_equal(
            _group_rref(r_sparse.final_tableau), _group_rref(r_hybrid.final_tableau)
        )
        # Non-triviality: brickwork at low p builds a volume-law state, so the
        # compared states carry real entanglement, so the equality above is then a
        # genuine match of non-trivial states, not a 0 == 0 coincidence.
        if p <= 0.05 and gating == "brickwork":
            assert r_sparse.entropy_half_cut > 0


def test_runner_actually_engages_dense_mode(monkeypatch):
    """A low-p volume-law run with hybrid=True must drive the runner's own
    simulator into dense mode (proven by instrumenting _switch_to_dense and
    asserting it fired during the run) and still match the sparse run."""
    from sparsegf2.circuits import CircuitConfig, simulate
    from sparsegf2.core import sparse_tableau as st

    calls = {"n": 0}
    orig = st.SparseGF2._switch_to_dense

    def counting(self):  # count REAL densifications (not mere calls): a no-op'd
        orig(self)  # _switch_to_dense would still be called but leave _dense_mode False
        if self._dense_mode:
            calls["n"] += 1

    monkeypatch.setattr(st.SparseGF2, "_switch_to_dense", counting)

    kw = dict(
        graph_spec="complete",
        n=24,
        picture="purification",
        gating_mode="brickwork",
        matching_mode="round_robin",
        measurement_mode="bernoulli",
        p=0.05,
        depth_mode="O(n)",
        depth_factor=6,
    )
    r_h = simulate(CircuitConfig(**kw, hybrid=True), sample_seed=0, save_tableau=True)
    assert calls["n"] > 0, "runner's hybrid sim never switched to dense"
    r_s = simulate(CircuitConfig(**kw, hybrid=False), sample_seed=0, save_tableau=True)
    assert np.array_equal(_group_rref(r_h.final_tableau), _group_rref(r_s.final_tableau))
