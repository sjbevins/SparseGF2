"""End-to-end runner coverage across every gating / matching / measurement mode.

The golden-path test only exercises brickwork × round_robin × bernoulli.
These tests drive the *runner* (not just the scheduler) through the other
live modes, so an integration bug (RNG desync, a bad gate pair reaching
``apply_gate_2q``, a wrong observable) surfaces here.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.circuits import CircuitConfig, complete_graph, cycle_graph, simulate
from sparsegf2.circuits.scheduler import CircuitBuilder
from sparsegf2.errors import InvalidArgumentError

# ---- matching modes through the runner ----


@pytest.mark.parametrize("matching_mode", ["round_robin", "palette", "fresh"])
def test_brickwork_matching_modes_run(matching_mode):
    cfg = CircuitConfig(
        graph_spec="complete",
        n=8,
        gating_mode="brickwork",
        matching_mode=matching_mode,
        p=0.1,
        depth_factor=2,
    )
    rec = simulate(cfg, sample_seed=0)
    # complete(8) brickwork: n/2 = 4 gates/layer, 16 layers
    assert rec.total_gates == 16 * 4
    assert rec.entropy_half_cut is not None
    # reproducible
    assert simulate(cfg, sample_seed=0).entropy_half_cut == rec.entropy_half_cut


# ---- measurement modes through the runner ----


def test_gated_measurement_bounded_by_gate_qubits():
    cfg = CircuitConfig(graph_spec="cycle", n=8, measurement_mode="gated", p=1.0, depth_factor=2)
    rec = simulate(cfg, sample_seed=0)
    # p=1.0 gated: every layer measures exactly the 2*n_gates touched qubits,
    # but a brickwork matching covers all n qubits, so n per layer.
    assert rec.total_measurements == rec.total_layers * 8


def test_random_pair_measurement_two_per_layer_at_p1():
    cfg = CircuitConfig(
        graph_spec="cycle", n=8, measurement_mode="random_pair", p=1.0, depth_factor=2
    )
    rec = simulate(cfg, sample_seed=0)
    assert rec.total_measurements == rec.total_layers * 2


# ---- gating modes through the runner ----


def test_random_edge_one_gate_per_layer_runs():
    cfg = CircuitConfig(graph_spec="cycle", n=8, gating_mode="random_edge", p=0.1, depth_factor=3)
    rec = simulate(cfg, sample_seed=0)
    assert rec.total_gates == rec.total_layers  # 1 gate/layer
    assert rec.entropy_half_cut is not None


# ---- purification with a non-default measurement mode ----


def test_purification_with_gated_reports_code_dimension():
    cfg = CircuitConfig(
        graph_spec="cycle",
        n=8,
        picture="purification",
        measurement_mode="gated",
        p=0.2,
        depth_factor=3,
    )
    rec = simulate(cfg, sample_seed=2)
    assert rec.code_dimension is not None
    assert 0 <= rec.code_dimension <= 8


def test_save_tableau_purification_shape():
    cfg = CircuitConfig(graph_spec="cycle", n=8, picture="purification", p=0.1, depth_factor=1)
    rec = simulate(cfg, sample_seed=0, save_tableau=True)
    # purification: 2n = 16 physical qubits -> to_symplectic is (2*16, 2*16)
    assert rec.final_tableau is not None
    assert rec.final_tableau.shape == (32, 32)


# ---- warmup actually changes the state ----


def test_warmup_changes_realized_state():
    # Note: code_dimension is INVARIANT under warmup here. Warmup gates act
    # only on system qubits, and S(system) can't change under system-local
    # unitaries. So we check the realized *tableau* instead: warmup consumes
    # the builder RNG before the measured phase, so the whole realization
    # (and thus to_symplectic()) differs.
    base = dict(graph_spec="cycle", n=8, picture="purification", p=0.1, depth_factor=2)
    no_warm = simulate(CircuitConfig(**base, warmup_layers=0), sample_seed=0, save_tableau=True)
    warm = simulate(CircuitConfig(**base, warmup_layers=6), sample_seed=0, save_tableau=True)
    assert not np.array_equal(no_warm.final_tableau, warm.final_tableau)


# ---- eager validation (the BLOCKER fix): incompatible (graph, mode) ----


@pytest.mark.parametrize("matching_mode", ["round_robin", "palette", "fresh"])
def test_odd_n_brickwork_raises_at_construction(matching_mode):
    # odd-n cycle has no perfect matching -> must fail in __post_init__, not
    # later in the scheduler.
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(graph_spec="cycle", n=7, gating_mode="brickwork", matching_mode=matching_mode)


def test_prebuilt_odd_graph_brickwork_raises():
    g = cycle_graph(9)  # no 1-factorization
    with pytest.raises(InvalidArgumentError):
        CircuitConfig(graph_spec=g, n=9, gating_mode="brickwork", matching_mode="round_robin")


def test_odd_n_random_edge_is_allowed():
    # random_edge needs only an edge, not a perfect matching -> odd n is fine.
    cfg = CircuitConfig(graph_spec="cycle", n=7, gating_mode="random_edge", p=0.1, depth_factor=2)
    rec = simulate(cfg, sample_seed=0)
    assert rec.total_gates == rec.total_layers


# ---- n_cliffords actually restricts the sampled gate indices ----


def test_n_cliffords_restricts_sampled_indices():
    cfg = CircuitConfig(graph_spec="cycle", n=8, n_cliffords=50, p=0.0, depth_factor=4)
    sched = CircuitBuilder(cfg, sample_seed=0).schedule()
    all_idx = np.concatenate([layer.cliff_indices for layer in sched])
    assert all_idx.max() < 50
    # and the restricted run still completes
    assert simulate(cfg, sample_seed=0).total_gates == cfg.total_layers() * 4


# ---- to_dict with a prebuilt graph ----


def test_to_dict_with_prebuilt_graph():
    d = CircuitConfig(graph_spec=complete_graph(8), n=8).to_dict()
    assert d["graph_spec"] == "complete(8)"


# ----------------------------------------------------------------------
# until_purified check-skipping: tau is EXACT, and a_bar is recorded
# ----------------------------------------------------------------------


@pytest.mark.parametrize("picture", ["purification", "single_ref"])
@pytest.mark.parametrize("p", [0.15, 0.3])
def test_until_purified_check_skipping_tau_exact(picture, p):
    """The measurement-count check-skipping path (record_time_series=False) must
    report the identical purified_at_layer and final observables as the
    per-layer-check path (record_time_series=True), trajectory for trajectory.
    The per-layer path evaluates the order parameter after every measured layer,
    so equality here proves the skipping introduces no approximation."""
    kw = dict(
        graph_spec="complete",
        n=16,
        p=p,
        picture=picture,
        gating_mode="brickwork",
        matching_mode="palette",
        measurement_mode="bernoulli",
        depth_mode="until_purified",
        depth_factor=40,
    )
    for seed in range(5):
        r_skip = simulate(CircuitConfig(**kw, record_time_series=False), sample_seed=seed)
        r_full = simulate(CircuitConfig(**kw, record_time_series=True), sample_seed=seed)
        assert r_skip.purified_at_layer == r_full.purified_at_layer
        assert r_skip.code_dimension == r_full.code_dimension
        assert r_skip.ref_entropy == r_full.ref_entropy
        assert r_skip.entropy_half_cut == r_full.entropy_half_cut


def test_mean_active_generators_recorded():
    """mean_active_generators (a_bar averaged over measured layers) is recorded,
    positive for any circuit with layers, and reflects the density physics:
    a low-p (volume-law) circuit is denser than a high-p (area-law) one."""
    kw = dict(
        graph_spec="complete",
        n=24,
        picture="purification",
        gating_mode="brickwork",
        matching_mode="palette",
        measurement_mode="bernoulli",
        depth_mode="O(n)",
        depth_factor=4,
    )
    r_low = simulate(CircuitConfig(**kw, p=0.05), sample_seed=0)
    r_high = simulate(CircuitConfig(**kw, p=0.6), sample_seed=0)
    assert r_low.mean_active_generators is not None
    assert r_high.mean_active_generators is not None
    assert r_low.mean_active_generators > 0
    # volume-law tableau is denser on average than the measurement-dominated one
    assert r_low.mean_active_generators > r_high.mean_active_generators


def test_mean_active_generators_in_sweep_rows():
    """The a_bar diagnostic flows through the sweep into the row dicts."""
    from sparsegf2.analysis import sweep

    base = CircuitConfig(
        graph_spec="complete",
        n=16,
        p=0.2,
        picture="single_ref",
        gating_mode="brickwork",
        matching_mode="palette",
        measurement_mode="bernoulli",
        depth_mode="until_purified",
        depth_factor=20,
    )
    res = sweep(base, {"p": [0.2, 0.4]}, seeds=2, n_jobs=1)
    assert all("mean_active_generators" in row for row in res.rows)
    assert all(
        row["mean_active_generators"] is None or row["mean_active_generators"] > 0
        for row in res.rows
    )
