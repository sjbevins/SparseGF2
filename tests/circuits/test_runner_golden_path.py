"""The must-pass end-to-end golden path + purification smoke."""

from __future__ import annotations

import sys

import numpy as np

from sparsegf2.circuits import CircuitConfig, Picture, SimulationRunner, simulate


def test_golden_path_brickwork_pure_state_cycle_bernoulli(golden_config):
    rec = simulate(golden_config, sample_seed=0)
    assert rec.picture is Picture.PURE_STATE
    assert rec.total_layers == 32  # depth_factor * n
    assert rec.total_gates == 32 * 4  # brickwork on cycle: n/2 = 4 gates/layer
    assert 0 <= rec.total_measurements <= 32 * 8
    assert rec.gate_to_meas_ratio_actual > 0
    assert rec.entropy_half_cut is not None
    assert 0 <= rec.entropy_half_cut <= 4  # half-cut entropy <= |A| for a pure state
    assert rec.code_dimension is None  # not defined for pure_state


def test_golden_path_is_reproducible(golden_config):
    a = simulate(golden_config, sample_seed=0)
    b = simulate(golden_config, sample_seed=0)
    assert a.total_measurements == b.total_measurements
    assert a.entropy_half_cut == b.entropy_half_cut


def test_different_sample_seed_differs(golden_config):
    a = simulate(golden_config, sample_seed=0)
    b = simulate(golden_config, sample_seed=1)
    # the realizations differ; at least one observable/diagnostic moves
    assert (a.total_measurements, a.entropy_half_cut) != (
        b.total_measurements,
        b.entropy_half_cut,
    ) or a.total_gates != b.total_gates


def test_purification_reports_code_dimension():
    cfg = CircuitConfig(graph_spec="cycle", n=8, picture="purification", p=0.16, depth_factor=4)
    rec = simulate(cfg, sample_seed=3)
    assert rec.code_dimension is not None
    assert 0 <= rec.code_dimension <= 8
    assert rec.entropy_half_cut is not None


def test_save_tableau_returns_symplectic():
    cfg = CircuitConfig(graph_spec="cycle", n=8, p=0.1, depth_factor=2)
    rec = simulate(cfg, sample_seed=0, save_tableau=True)
    assert rec.final_tableau is not None
    assert rec.final_tableau.shape == (16, 16)  # (2N, 2N), N=n=8
    assert rec.final_tableau.dtype == np.uint8


def test_runner_no_runtime_stim():
    """No circuits module has ``stim`` in scope after a full run.

    The ``__dict__`` check catches any module-level Stim import; we avoid
    asserting on ``sys.modules`` since the core parity tests import stim
    into the same process.
    """
    cfg = CircuitConfig(graph_spec="cycle", n=8, p=0.1, depth_factor=2)
    SimulationRunner(cfg).run(0)
    leaked = [
        name
        for name, mod in list(sys.modules.items())
        if name.startswith("sparsegf2.circuits")
        and mod is not None
        and hasattr(mod, "__dict__")
        and "stim" in mod.__dict__
    ]
    assert not leaked, f"circuits modules with stim in scope: {leaked}"


def test_injected_clifford_table_is_used():
    cfg = CircuitConfig(graph_spec="cycle", n=8, p=0.1, depth_factor=2, n_cliffords=720)
    from sparsegf2.circuits._clifford_table import sp4_table

    custom = sp4_table().copy()  # a private copy, still valid
    runner = SimulationRunner(cfg, clifford_table=custom)
    assert runner.clifford_table is custom
    rec = runner.run(0)
    assert rec.total_gates == 16 * 4  # 2*8 layers x n/2 gates
