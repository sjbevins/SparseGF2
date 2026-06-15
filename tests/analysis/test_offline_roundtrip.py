"""Offline analysis must equal online analysis bit-for-bit (the round-trip)."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.analysis import analyze, analyze_tableaux, reconstruct
from sparsegf2.circuits import CircuitConfig, simulate
from sparsegf2.circuits.picture import setup_picture
from sparsegf2.errors import InvalidArgumentError

ANALYSES = [
    "half_cut_entropy",
    "code_dimension",
    "ref_entropy",
    "bipartite_mutual_info",
    "tripartite_mutual_info",
    "average_stabilizer_weight",
    "stabilizer_weight_spectrum",
    "entropy_profile",
]


@pytest.mark.parametrize("picture", ["pure_state", "purification", "single_ref"])
def test_online_equals_offline(picture):
    cfg = CircuitConfig(graph_spec="cycle", n=12, picture=picture, p=0.2, depth_factor=4)
    online = simulate(cfg, sample_seed=3, analyses=ANALYSES)
    saved = simulate(cfg, sample_seed=3, save_tableau=True)
    spec = setup_picture(cfg.picture, cfg.n)[1]
    offline = analyze_tableaux([saved.final_tableau], spec, ANALYSES)[0]
    for name in ANALYSES:
        on, off = online.analyses[name], offline[name]
        if on is None:
            assert off is None
        else:
            assert np.array_equal(np.asarray(on), np.asarray(off)), name


def test_reconstruct_and_analyze():
    cfg = CircuitConfig(graph_spec="complete", n=10, picture="purification", p=0.15, depth_factor=3)
    saved = simulate(cfg, sample_seed=1, save_tableau=True)
    spec = setup_picture(cfg.picture, cfg.n)[1]
    sim = reconstruct(saved)
    assert sim.n == saved.final_tableau.shape[0] // 2
    out = analyze(sim, spec, ["code_dimension"])
    assert out["code_dimension"] == saved.code_dimension


def test_reconstruct_without_tableau_raises():
    cfg = CircuitConfig(graph_spec="cycle", n=8, p=0.2, depth_factor=2)
    rec = simulate(cfg, sample_seed=0)  # no save_tableau
    with pytest.raises(InvalidArgumentError):
        reconstruct(rec)


def test_raw_ndarray_source_path():
    from sparsegf2 import SparseGF2

    sim = SparseGF2(4)
    sim.apply_h(0)
    sim.apply_cx(0, 1)
    symp = sim.to_symplectic()
    cfg = CircuitConfig(graph_spec="cycle", n=4, picture="pure_state", p=0.0)
    spec = setup_picture(cfg.picture, cfg.n)[1]
    # reconstruct + analyze accept a bare (2N, 2N) array, matching the record path
    rebuilt = reconstruct(symp)
    assert np.array_equal(rebuilt.to_symplectic(), symp)
    out = analyze(symp, spec, ["half_cut_entropy"])
    assert out["half_cut_entropy"] == analyze(sim, spec, ["half_cut_entropy"])["half_cut_entropy"]


def test_reconstruct_forwards_pivot_and_numba():
    cfg = CircuitConfig(graph_spec="cycle", n=8, picture="purification", p=0.2, depth_factor=2)
    saved = simulate(cfg, sample_seed=0, save_tableau=True)
    sim = reconstruct(saved, pivot_mode="first", use_numba=False)
    assert sim.use_numba is False
    assert sim.use_min_weight_pivot is False  # pivot_mode='first' -> first anticommuter
