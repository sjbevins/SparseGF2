"""
Validation of the ``until_purified`` depth mode.

Invariants tested:

- validation: ``depth_mode='until_purified'`` with a non-single_ref
  picture is rejected by :class:`CircuitConfig.__post_init__`;
- early termination: at p well above the all-to-all MIPT critical
  rate (p ~ 0.13), samples purify in far fewer layers than the
  ``depth_factor * n`` cap;
- shape consistency: with ``record_time_series=True`` every per-sample
  trace has length ``total_layers() + 1`` regardless of when
  purification actually happened (early-terminated samples are
  zero-padded, matching the physical invariant that S(qubit n) remains
  0 after purification);
- trace semantics: for samples that purified, the first-zero index of
  the trace equals ``SampleRecord.total_layers``.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.circuits import CircuitConfig, SimulationRunner


def test_until_purified_rejects_non_single_ref():
    with pytest.raises(ValueError, match="until_purified"):
        CircuitConfig(
            graph_spec="complete",
            n=8,
            picture="purification",
            depth_mode="until_purified",
            p=0.3,
        )


@pytest.mark.parametrize("p", [0.3, 0.5, 0.75])
def test_until_purified_terminates_well_below_cap(p):
    """For p >> p_c samples purify in far fewer layers than depth_factor*n."""
    n = 32
    depth_factor = 16
    cc = CircuitConfig(
        graph_spec="complete",
        n=n,
        picture="single_ref",
        matching_mode="round_robin",
        p=p,
        depth_factor=depth_factor,
        depth_mode="until_purified",
    )
    cap = cc.total_layers()
    runner = SimulationRunner(cc, warmup_jit=False)
    layers_used = []
    for seed in range(20):
        rec = runner.run(sample_seed=seed)
        layers_used.append(rec.total_layers)
    # With 20 samples at p >> p_c, the mean termination time should be
    # dramatically less than the cap.
    mean_layers = float(np.mean(layers_used))
    assert mean_layers < 0.5 * cap, (
        f"p={p}, n={n}: expected mean termination < {0.5 * cap} layers, "
        f"got {mean_layers:.2f}"
    )


def test_timeseries_shape_is_total_layers_plus_one():
    """Padded trace length = total_layers() + 1 for every sample, even
    those that terminated long before the cap."""
    n = 16
    cc = CircuitConfig(
        graph_spec="complete",
        n=n,
        picture="single_ref",
        matching_mode="round_robin",
        p=0.4,
        depth_factor=8,
        depth_mode="until_purified",
        record_time_series=True,
    )
    expected = cc.total_layers() + 1
    runner = SimulationRunner(cc, warmup_jit=False)
    for seed in range(10):
        rec = runner.run(sample_seed=seed)
        ts = rec.ref_entropy_timeseries
        assert ts is not None, f"seed={seed}: expected a trace"
        assert ts.shape == (expected,), (
            f"seed={seed}: trace shape {ts.shape} != ({expected},)"
        )
        # Trace starts at 1 (Bell pair), ends at 0 (padded or purified).
        assert int(ts[0]) == 1
        assert int(ts[-1]) == 0


def test_total_layers_equals_first_zero_when_purified():
    """When a sample purified before the cap,
    SampleRecord.total_layers equals argmin(t : S(t) == 0) in its trace."""
    cc = CircuitConfig(
        graph_spec="complete",
        n=16,
        picture="single_ref",
        matching_mode="round_robin",
        p=0.5,
        depth_factor=16,
        depth_mode="until_purified",
        record_time_series=True,
    )
    cap = cc.total_layers()
    runner = SimulationRunner(cc, warmup_jit=False)
    for seed in range(12):
        rec = runner.run(sample_seed=seed)
        ts = rec.ref_entropy_timeseries
        zeros = np.where(ts == 0)[0]
        if rec.total_layers >= cap:
            # Did not purify within the cap; first zero (if any) is padding.
            continue
        # Purified early: first-zero index should be rec.total_layers, and
        # rec.k should be 0.
        assert zeros.size > 0, f"seed={seed}: purified but no zero in trace"
        assert int(zeros[0]) == rec.total_layers, (
            f"seed={seed}: first-zero index {int(zeros[0])} != "
            f"total_layers {rec.total_layers}"
        )
        assert rec.k == 0
