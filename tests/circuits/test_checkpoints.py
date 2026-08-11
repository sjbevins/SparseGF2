"""Depth-checkpoint tableaux: ``run(..., checkpoint_layers=...)`` snapshots the
full (2N, 2N) tableau at requested measured layers without perturbing the run.

The strong correctness check uses a physical invariant of the purification
picture: ``code_dimension = S(system)`` (system-vs-reference entanglement) is
**monotone non-increasing** over the circuit -- gates act unitarily *within* the
system (leaving S(system) invariant) and single-qubit system measurements can
only reduce it (LOCC monotonicity). So the reconstructed checkpoints must have
non-increasing code dimension with depth.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sparsegf2.analysis.offline import reconstruct
from sparsegf2.circuits import CHECKPOINT_STOP, CircuitConfig, SimulationRunner, simulate
from sparsegf2.core.observables import entanglement_entropy
from sparsegf2.errors import InvalidArgumentError


def _cfg(n=16, beta=0.1, p=0.15, seed=3):
    return CircuitConfig(
        graph_spec=f"watts_strogatz(k=2,beta={beta})",
        n=n,
        picture="purification",
        gating_mode="random_pool",
        measurement_mode="bernoulli",
        p=p,
        depth_mode="O(n)",
        depth_factor=4,
        base_seed=seed,
    )


def test_checkpoint_keys_shapes_and_dtype():
    n = 16
    cfg = _cfg(n)
    tl = cfg.total_layers()
    chk = sorted({1, 3, 5, tl})
    rec = SimulationRunner(cfg).run(sample_seed=7, checkpoint_layers=chk)
    assert set(rec.checkpoint_tableaux) == set(chk)
    for tableau in rec.checkpoint_tableaux.values():
        assert tableau.shape == (4 * n, 4 * n)  # (2N, 2N), N = 2n for purification
        assert tableau.dtype == np.uint8


def test_final_checkpoint_equals_save_tableau():
    cfg = _cfg()
    tl = cfg.total_layers()
    rec = SimulationRunner(cfg).run(sample_seed=1, checkpoint_layers=[1, tl], save_tableau=True)
    assert np.array_equal(rec.checkpoint_tableaux[tl], rec.final_tableau)


def test_checkpoint_snapshots_are_independent_copies():
    cfg = _cfg()
    tl = cfg.total_layers()
    rec = SimulationRunner(cfg).run(sample_seed=1, checkpoint_layers=[1, 2, tl], save_tableau=True)
    first = rec.checkpoint_tableaux[1]
    second = rec.checkpoint_tableaux[2]
    final = rec.final_tableau
    assert not np.shares_memory(first, second)
    assert not np.shares_memory(first, final)
    assert not np.shares_memory(second, final)


def test_checkpoints_do_not_perturb_the_run():
    cfg = _cfg()
    tl = cfg.total_layers()
    with_chk = SimulationRunner(cfg).run(
        sample_seed=9, checkpoint_layers=[1, 4, tl], save_tableau=True
    )
    without = SimulationRunner(cfg).run(sample_seed=9, save_tableau=True)
    assert np.array_equal(with_chk.final_tableau, without.final_tableau)
    assert with_chk.code_dimension == without.code_dimension
    assert with_chk.total_measurements == without.total_measurements


def test_out_of_range_layers_are_ignored():
    cfg = _cfg()
    tl = cfg.total_layers()
    rec = SimulationRunner(cfg).run(
        sample_seed=2, checkpoint_layers=[-10, 0, 1, tl, tl + 100, 10**9]
    )
    assert set(rec.checkpoint_tableaux) == {1, tl}  # layers past the end never fire

    missed = SimulationRunner(cfg).run(checkpoint_layers=[-1, 0, tl + 1])
    assert missed.checkpoint_tableaux is None


def test_no_checkpoints_requested_leaves_field_none():
    rec = simulate(_cfg(), sample_seed=0)
    assert rec.checkpoint_tableaux is None


def test_code_dimension_is_monotone_nonincreasing_across_checkpoints():
    # S(system) can only fall with depth; check it on the reconstructed snapshots.
    n = 24
    cfg = _cfg(n=n, beta=0.05, p=0.2, seed=5)
    tl = cfg.total_layers()
    chk = sorted({1, 2, 4, 8, round(n**0.5), tl})
    rec = SimulationRunner(cfg).run(sample_seed=11, checkpoint_layers=chk)
    codes = []
    for t in sorted(rec.checkpoint_tableaux):
        sim = reconstruct(rec.checkpoint_tableaux[t])
        assert sim.n == 2 * n
        k = int(entanglement_entropy(sim, range(n)))  # code_dimension = S(system)
        assert 0 <= k <= n
        codes.append(k)
    assert codes == sorted(codes, reverse=True), f"code dim not non-increasing: {codes}"


def test_checkpoint_reconstruction_matches_live_final_state():
    # The last checkpoint reconstructs to the same code dimension the runner reports.
    cfg = _cfg(n=20, beta=0.2, p=0.18, seed=8)
    tl = cfg.total_layers()
    rec = SimulationRunner(cfg).run(sample_seed=4, checkpoint_layers=[tl])
    sim = reconstruct(rec.checkpoint_tableaux[tl])
    assert int(entanglement_entropy(sim, range(cfg.n))) == rec.code_dimension


def test_checkpoint_callback_computes_on_live_state_without_tableaux():
    # With a callback, values are computed on the live state and no tableaux saved;
    # the callback result must match what reconstructing the tableau would give.
    n = 16
    cfg = _cfg(n)
    tl = cfg.total_layers()
    chk = [1, 4, tl]

    def cb(sim, spec, layer):
        return int(entanglement_entropy(sim, range(n)))  # code dimension on the live sim

    rec = SimulationRunner(cfg).run(sample_seed=7, checkpoint_layers=chk, checkpoint_callback=cb)
    assert set(rec.checkpoint_values) == set(chk)
    assert rec.checkpoint_tableaux is None  # callback path does not also save tableaux
    assert rec.checkpoint_values[tl] == rec.code_dimension
    # cross-check against the tableau path (same run, same seed)
    rec_t = SimulationRunner(cfg).run(sample_seed=7, checkpoint_layers=chk)
    for t in chk:
        sim = reconstruct(rec_t.checkpoint_tableaux[t])
        assert rec.checkpoint_values[t] == int(entanglement_entropy(sim, range(n)))


def test_separate_run_to_T_equals_checkpoint_at_T():
    # A circuit run to EXACTLY T layers (total_layers_override) is bit-identical to
    # the depth-T checkpoint of a longer circuit with the same seeds: the builder
    # draws layers in order from one stream, so the first T layers coincide. This
    # is the property the factor-4 depth study relies on (checkpoint == separate).
    cfg_long = _cfg(n=20, beta=0.15, p=0.2, seed=6)
    T = 7
    assert cfg_long.total_layers() > T
    rec_long = SimulationRunner(cfg_long).run(sample_seed=5, checkpoint_layers=[T])
    cfg_T = replace(cfg_long, total_layers_override=T)
    rec_T = SimulationRunner(cfg_T).run(sample_seed=5, save_tableau=True)
    assert rec_T.total_layers == T
    assert np.array_equal(rec_long.checkpoint_tableaux[T], rec_T.final_tableau)


def test_checkpoint_callback_does_not_perturb_the_run():
    cfg = _cfg()
    tl = cfg.total_layers()
    ran = SimulationRunner(cfg).run(
        sample_seed=9,
        checkpoint_layers=[1, tl],
        checkpoint_callback=lambda s, sp, t: 0,
        save_tableau=True,
    )
    plain = SimulationRunner(cfg).run(sample_seed=9, save_tableau=True)
    assert np.array_equal(ran.final_tableau, plain.final_tableau)
    assert ran.code_dimension == plain.code_dimension


@pytest.mark.parametrize("depth_mode", ["O(n)", "until_purified"])
@pytest.mark.parametrize("record_time_series", [False, True])
def test_checkpoint_stop_preserves_final_observables(depth_mode, record_time_series):
    cfg = CircuitConfig(
        graph_spec="cycle",
        n=4,
        picture="purification",
        p=1.0,
        depth_mode=depth_mode,
        depth_factor=4,
        record_time_series=record_time_series,
    )
    rec = SimulationRunner(cfg).run(
        sample_seed=0,
        checkpoint_layers=[1],
        checkpoint_callback=lambda sim, spec, layer: CHECKPOINT_STOP,
        save_tableau=True,
    )
    assert rec.total_layers == rec.purified_at_layer == 1
    assert rec.code_dimension == 0
    assert rec.checkpoint_values is None  # the stop sentinel is never stored
    assert rec.final_tableau is not None
    if record_time_series:
        assert rec.time_series == [0]
    else:
        assert rec.time_series is None


def test_checkpoint_stop_before_purification_is_not_labeled_purified():
    cfg = replace(_cfg(n=8, p=0.0), total_layers_override=8, record_time_series=True)
    rec = SimulationRunner(cfg).run(
        sample_seed=3,
        checkpoint_layers=[1],
        checkpoint_callback=lambda sim, spec, layer: CHECKPOINT_STOP,
    )
    assert rec.total_layers == 1
    assert rec.code_dimension == cfg.n
    assert rec.time_series == [cfg.n]
    assert rec.purified_at_layer is None


def test_checkpoint_stop_is_not_stored_and_preserves_earlier_values():
    cfg = replace(_cfg(n=8, p=0.0), total_layers_override=8)

    def callback(sim, spec, layer):
        return "kept" if layer == 1 else CHECKPOINT_STOP

    rec = SimulationRunner(cfg).run(
        sample_seed=3,
        checkpoint_layers=[1, 2, 3],
        checkpoint_callback=callback,
    )
    assert rec.total_layers == 2
    assert rec.checkpoint_values == {1: "kept"}
    assert rec.purified_at_layer is None


def test_checkpoint_stop_reports_first_detected_not_unobserved_exact_layer():
    cfg = CircuitConfig(
        graph_spec="cycle",
        n=4,
        picture="purification",
        p=1.0,
        depth_mode="O(n)",
        total_layers_override=4,
    )
    rec = SimulationRunner(cfg).run(
        sample_seed=0,
        checkpoint_layers=[3],
        checkpoint_callback=lambda sim, spec, layer: CHECKPOINT_STOP,
    )
    assert rec.code_dimension == 0
    assert rec.total_layers == rec.purified_at_layer == 3


def test_checkpoint_stop_in_pure_state_is_not_labeled_purified():
    cfg = CircuitConfig(
        graph_spec="cycle", n=4, picture="pure_state", p=1.0, total_layers_override=4
    )
    rec = SimulationRunner(cfg).run(
        sample_seed=0,
        checkpoint_layers=[1],
        checkpoint_callback=lambda sim, spec, layer: CHECKPOINT_STOP,
    )
    assert rec.total_layers == 1
    assert rec.purified_at_layer is None


@pytest.mark.parametrize("picture", ["pure_state", "purification", "single_ref"])
@pytest.mark.parametrize("gating_mode", ["brickwork", "random_edge", "random_pool", "all_edges"])
def test_checkpoint_stop_matches_independent_exact_depth_across_modes(picture, gating_mode):
    kwargs = {}
    if gating_mode == "random_edge":
        kwargs["gates_per_layer"] = 2
    elif gating_mode == "random_pool":
        kwargs["gates_per_layer"] = 4
    cfg = CircuitConfig(
        graph_spec="cycle",
        n=8,
        picture=picture,
        gating_mode=gating_mode,
        p=0.35,
        total_layers_override=6,
        record_time_series=picture != "pure_state",
        base_seed=17,
        **kwargs,
    )
    stopped = SimulationRunner(cfg).run(
        sample_seed=23,
        checkpoint_layers=[3],
        checkpoint_callback=lambda sim, spec, layer: CHECKPOINT_STOP,
        save_tableau=True,
    )
    exact = SimulationRunner(replace(cfg, total_layers_override=3)).run(
        sample_seed=23, save_tableau=True
    )

    assert stopped.total_layers == exact.total_layers == 3
    assert stopped.total_gates == exact.total_gates
    assert stopped.total_measurements == exact.total_measurements
    assert stopped.code_dimension == exact.code_dimension
    assert stopped.ref_entropy == exact.ref_entropy
    assert stopped.entropy_half_cut == exact.entropy_half_cut
    assert stopped.time_series == exact.time_series
    assert np.array_equal(stopped.final_tableau, exact.final_tableau)


def test_checkpoint_callback_requires_layers():
    with pytest.raises(InvalidArgumentError, match="requires checkpoint_layers"):
        SimulationRunner(_cfg()).run(checkpoint_callback=lambda sim, spec, layer: None)


def test_checkpoint_callback_must_be_callable():
    with pytest.raises(InvalidArgumentError, match="checkpoint_callback must be callable"):
        SimulationRunner(_cfg()).run(checkpoint_layers=[1], checkpoint_callback=object())


def test_checkpoint_layers_rejects_noniterable_container():
    with pytest.raises(InvalidArgumentError, match="iterable of integers"):
        SimulationRunner(_cfg()).run(checkpoint_layers=1)


@pytest.mark.parametrize("bad", [1.5, "2", True, np.float64(3.0)])
def test_checkpoint_layers_reject_lossy_coercion(bad):
    with pytest.raises(InvalidArgumentError, match="must contain integers"):
        SimulationRunner(_cfg()).run(checkpoint_layers=[bad])


def test_checkpoint_layers_accept_numpy_integers_and_deduplicate():
    rec = SimulationRunner(_cfg()).run(checkpoint_layers=[np.int64(1), 1])
    assert set(rec.checkpoint_tableaux) == {1}
