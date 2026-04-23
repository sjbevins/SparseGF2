"""
Tests for the ``warmup_layers`` pre-scrambling feature.

Invariants:
- warmup_layers is validated to be a non-negative integer and is only
  meaningful for picture='single_ref';
- the builder's ``warmup_layers_iter`` emits exactly
  ``cfg.warmup_layers`` layers, each with zero measurements and the
  same gate-placement logic as the main ``layers()``;
- after running the warmup in the simulator, the reference-qubit
  entropy is still 1 (warmup applies no measurements, so the
  Bell-pair correlation is preserved, only spread);
- recorded timeseries[0] is taken AFTER the warmup (i.e. corresponds
  to "t / n = 0" in the user-facing analysis).
"""
from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.circuits import CircuitConfig, SimulationRunner
from sparsegf2.circuits.builder import CircuitBuilder


def test_warmup_layers_must_be_nonneg_integer():
    with pytest.raises(ValueError, match="non-negative"):
        CircuitConfig(graph_spec="complete", n=8, picture="single_ref",
                      p=0.3, warmup_layers=-1)


def test_warmup_layers_allowed_for_single_ref_and_purification():
    # Both pictures with a pre-scrambling phase accept warmup_layers.
    for pic in ("single_ref", "purification"):
        CircuitConfig(graph_spec="complete", n=8, picture=pic,
                      p=0.3, warmup_layers=4)


@pytest.mark.parametrize("warmup", [0, 1, 4, 16])
def test_warmup_iter_yields_expected_count(warmup):
    cfg = CircuitConfig(
        graph_spec="complete", n=8, picture="single_ref",
        gating_mode="random_edge", measurement_mode="gated",
        p=0.3, depth_factor=8, warmup_layers=warmup,
    )
    builder = CircuitBuilder(cfg, sample_seed=0)
    layers = list(builder.warmup_layers_iter())
    assert len(layers) == warmup
    # Every warmup layer has zero measurements and one gate (random_edge).
    for layer in layers:
        assert layer.n_measurements == 0
        assert layer.n_gates == 1


def test_warmup_ref_entropy_stays_one_post_warmup():
    """Warmup is gate-only, so the Bell-pair S(ref) stays 1."""
    cfg = CircuitConfig(
        graph_spec="complete", n=16, picture="single_ref",
        gating_mode="random_edge", measurement_mode="gated",
        p=0.3, depth_factor=4, warmup_layers=32,
        depth_mode="until_purified", record_time_series=True,
    )
    runner = SimulationRunner(cfg, warmup_jit=False)
    for seed in range(5):
        rec = runner.run(sample_seed=seed)
        assert int(rec.ref_entropy_timeseries[0]) == 1, (
            f"seed={seed}: ts[0]={int(rec.ref_entropy_timeseries[0])}, "
            "expected 1 (warmup is unitary)"
        )


def test_p_zero_with_warmup_preserves_bell_pair():
    """At p=0 the circuit is unitary throughout, so S(ref)=1 at end."""
    cfg = CircuitConfig(
        graph_spec="complete", n=16, picture="single_ref",
        gating_mode="random_edge", measurement_mode="gated",
        p=0.0, depth_factor=4, warmup_layers=16,
        depth_mode="until_purified", record_time_series=True,
    )
    rec = SimulationRunner(cfg, warmup_jit=False).run(sample_seed=0)
    assert rec.k == 1
    ts = rec.ref_entropy_timeseries
    assert all(int(v) == 1 for v in ts)


def test_p_one_with_warmup_still_purifies():
    """At p=1 the Bell-pair correlation is destroyed in a handful of
    measurements even starting from the scrambled state after warmup."""
    cfg = CircuitConfig(
        graph_spec="complete", n=16, picture="single_ref",
        gating_mode="random_edge", measurement_mode="gated",
        p=1.0, depth_factor=8, warmup_layers=16,
        depth_mode="until_purified", record_time_series=True,
    )
    rec = SimulationRunner(cfg, warmup_jit=False).run(sample_seed=0)
    assert rec.k == 0
