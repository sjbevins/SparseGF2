"""
Gold-standard Stim parity tests for the ``sparsegf2.circuits`` subpackage.

For every matching mode (``round_robin``, ``palette``, ``fresh``) and every
MVP graph (``cycle``, ``complete``), this test builds the circuit schedule
with :class:`CircuitBuilder`, then runs it through:

  1. the new SparseGF2 runner (:class:`SimulationRunner`), and
  2. an independent ``stim.TableauSimulator`` driven layer-by-layer by the
     SAME Clifford indices and SAME measurement positions,

then compares the GF(2) row-reduced echelon form (RREF) of both stabilizer
groups. Identical RREFs prove the two simulators produce the same stabilizer
group — this is the minimum bar for any output to be trusted for code-
discovery analysis.
"""
from __future__ import annotations

import numpy as np
import pytest
import stim

from sparsegf2 import SparseGF2, warmup
from sparsegf2.gates.clifford import symplectic_from_stim_tableau
from sparsegf2.circuits import CircuitConfig, CircuitBuilder
from sparsegf2.circuits.pictures import init_picture


# ══════════════════════════════════════════════════════════════
# Helpers (mirrors test_stim_rref_verification.py conventions)
# ══════════════════════════════════════════════════════════════

_CLIFF_TABS = []        # list[stim.Tableau] — two-qubit tableaus
_CLIFF_SYMP = None      # ndarray[11520, 4, 4] uint8 — symplectic matrices


def _build_clifford_cache():
    global _CLIFF_SYMP, _CLIFF_TABS
    if _CLIFF_SYMP is not None:
        return
    all_tabs = list(stim.Tableau.iter_all(2))
    _CLIFF_TABS = all_tabs
    _CLIFF_SYMP = np.zeros((len(all_tabs), 4, 4), dtype=np.uint8)
    for i, tab in enumerate(all_tabs):
        _CLIFF_SYMP[i] = symplectic_from_stim_tableau(tab)


@pytest.fixture(scope="module", autouse=True)
def _setup_module():
    _build_clifford_cache()
    warmup()


def _gf2_rref(mat: np.ndarray) -> np.ndarray:
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


def _extract_stim_matrix(sim: stim.TableauSimulator, n: int) -> np.ndarray:
    """Symplectic [X | Z] matrix of stim stabilizers restricted to system qubits."""
    N = 2 * n
    inv_tab = sim.current_inverse_tableau()
    fwd = inv_tab.inverse(unsigned=True)
    mat = np.zeros((N, 2 * n), dtype=np.uint8)
    for row in range(N):
        ps = fwd.z_output(row)
        for col in range(n):
            pauli = ps[col]            # 0=I, 1=X, 2=Y, 3=Z
            if pauli in (1, 2):
                mat[row, col] = 1
            if pauli in (2, 3):
                mat[row, n + col] = 1
    return mat


def _extract_sparse_matrix(sim: SparseGF2) -> np.ndarray:
    """Symplectic [X | Z] matrix of SparseGF2 stabilizers restricted to system qubits."""
    return sim.extract_sys_matrix()


# ══════════════════════════════════════════════════════════════
# Runners (shared gate schedule driven by CircuitBuilder)
# ══════════════════════════════════════════════════════════════

def _run_sparse(cfg: CircuitConfig, sample_seed: int) -> SparseGF2:
    builder = CircuitBuilder(cfg, sample_seed=sample_seed)
    sim = init_picture(cfg.picture, cfg.n, hybrid_mode=True)
    for layer in builder.layers():
        for i, (qi, qj) in enumerate(layer.gate_pairs):
            ci = int(layer.cliff_indices[i]) % len(_CLIFF_SYMP)
            sim.apply_gate(qi, qj, _CLIFF_SYMP[ci])
        for q in layer.meas_qubits:
            sim.apply_measurement_z(q)
    return sim


def _run_stim(cfg: CircuitConfig, sample_seed: int) -> stim.TableauSimulator:
    builder = CircuitBuilder(cfg, sample_seed=sample_seed)
    sim = stim.TableauSimulator()
    n = cfg.n
    # Purification picture: Bell pairs between system qubit i and reference n+i.
    for i in range(n):
        sim.h(i)
        sim.cx(i, n + i)
    for layer in builder.layers():
        for i, (qi, qj) in enumerate(layer.gate_pairs):
            ci = int(layer.cliff_indices[i]) % len(_CLIFF_TABS)
            sim.do_tableau(_CLIFF_TABS[ci], [qi, qj])
        for q in layer.meas_qubits:
            sim.measure(q)
            sim.reset(q)
    return sim


def _assert_rref_match(sim_sparse: SparseGF2, sim_stim: stim.TableauSimulator,
                       n: int, msg: str = "") -> None:
    mat_sparse = _extract_sparse_matrix(sim_sparse)
    mat_stim = _extract_stim_matrix(sim_stim, n)
    rref_s = _gf2_rref(mat_sparse)
    rref_t = _gf2_rref(mat_stim)
    assert rref_s.shape == rref_t.shape, (
        f"[{msg}] RREF rank mismatch: sparse={rref_s.shape[0]} stim={rref_t.shape[0]}"
    )
    assert np.array_equal(rref_s, rref_t), (
        f"[{msg}] RREF content mismatch at rank {rref_s.shape[0]}"
    )


# ══════════════════════════════════════════════════════════════
# Parameterized parity tests
# ══════════════════════════════════════════════════════════════

GRAPHS = ("cycle", "complete")
MODES = ("round_robin", "palette", "fresh")
SIZES = (8, 12, 16)
SEEDS = (0, 1, 2)
P_VALUES = (0.0, 0.1, 0.3)


@pytest.mark.parametrize("graph", GRAPHS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("p", P_VALUES)
@pytest.mark.parametrize("seed", SEEDS)
def test_stim_rref_parity(graph: str, mode: str, n: int, p: float, seed: int):
    """SparseGF2 and Stim produce RREF-equivalent stabilizer groups."""
    cfg = CircuitConfig(
        graph_spec=graph,
        n=n,
        picture="purification",
        gating_mode="matching",
        matching_mode=mode,
        measurement_mode="uniform",
        p=p,
        depth_mode="O(n)",
        depth_factor=2,            # keep tests fast; enough depth to mix
        base_seed=42 + seed,
    )
    sim_sparse = _run_sparse(cfg, sample_seed=seed)
    sim_stim = _run_stim(cfg, sample_seed=seed)
    _assert_rref_match(
        sim_sparse, sim_stim, n,
        msg=f"graph={graph} mode={mode} n={n} p={p} seed={seed}",
    )
