"""Tests for the named-analysis registry."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.analysis.registry import (
    ANALYSES,
    register_analysis,
    resolve_analyses,
)
from sparsegf2.circuits import CircuitConfig, simulate
from sparsegf2.errors import InvalidArgumentError

BUILTINS = {
    "half_cut_entropy",
    "code_dimension",
    "ref_entropy",
    "bipartite_mutual_info",
    "tripartite_mutual_info",
    "average_stabilizer_weight",
    "stabilizer_weight_spectrum",
    "entropy_profile",
}


def test_builtins_registered():
    assert set(ANALYSES) >= BUILTINS


def test_resolve_none_empty():
    assert resolve_analyses(None) == {}


def test_resolve_names_and_callables():
    def my_fn(sim, spec):
        return 1

    resolved = resolve_analyses(["half_cut_entropy", my_fn])
    assert list(resolved) == ["half_cut_entropy", "my_fn"]
    assert resolved["half_cut_entropy"] is ANALYSES["half_cut_entropy"]


def test_resolve_mapping_verbatim():
    fn = lambda sim, spec: 7  # noqa: E731
    resolved = resolve_analyses({"x": fn})
    assert resolved == {"x": fn}


def test_resolve_unknown_name_raises():
    with pytest.raises(InvalidArgumentError):
        resolve_analyses(["not_a_real_analysis"])


def test_register_duplicate_raises():
    with pytest.raises(InvalidArgumentError):
        register_analysis("half_cut_entropy", lambda sim, spec: 0)


def test_half_cut_matches_record_field():
    cfg = CircuitConfig(graph_spec="cycle", n=12, picture="purification", p=0.2, depth_factor=4)
    rec = simulate(cfg, sample_seed=0, analyses=["half_cut_entropy", "code_dimension"])
    assert rec.analyses["half_cut_entropy"] == rec.entropy_half_cut
    assert rec.analyses["code_dimension"] == rec.code_dimension


def test_entropy_profile_is_array():
    cfg = CircuitConfig(graph_spec="cycle", n=8, picture="pure_state", p=0.2, depth_factor=4)
    rec = simulate(cfg, sample_seed=0, analyses=["entropy_profile"])
    prof = rec.analyses["entropy_profile"]
    assert isinstance(prof, np.ndarray)
    assert prof.shape == (cfg.n - 1,)


def test_bare_string_is_single_analysis_not_chars():
    resolved = resolve_analyses("half_cut_entropy")
    assert list(resolved) == ["half_cut_entropy"]


def test_duplicate_callable_name_raises():
    def f(sim, spec):
        return 1

    def make_f():
        def f(sim, spec):  # noqa: F811 - intentionally same __name__
            return 2

        return f

    with pytest.raises(InvalidArgumentError, match="duplicate analysis name"):
        resolve_analyses([f, make_f()])


def test_two_lambdas_collide_raises():
    with pytest.raises(InvalidArgumentError, match="duplicate analysis name"):
        resolve_analyses([lambda sim, spec: 1, lambda sim, spec: 2])


def test_mapping_non_callable_value_raises():
    with pytest.raises(InvalidArgumentError, match="must map to a callable"):
        resolve_analyses({"x": 5})


def test_picture_correct_none_gating():
    # pure_state: both picture-specific order parameters are None.
    cfg = CircuitConfig(graph_spec="cycle", n=8, picture="pure_state", p=0.2, depth_factor=2)
    rec = simulate(cfg, sample_seed=0, analyses=["code_dimension", "ref_entropy"])
    assert rec.analyses["code_dimension"] is None
    assert rec.analyses["ref_entropy"] is None


def test_single_ref_binds_ref_entropy_not_code_dimension():
    cfg = CircuitConfig(graph_spec="cycle", n=8, picture="single_ref", p=0.2, depth_factor=2)
    rec = simulate(cfg, sample_seed=0, analyses=["code_dimension", "ref_entropy"])
    assert rec.analyses["code_dimension"] is None
    assert rec.analyses["ref_entropy"] == rec.ref_entropy


def test_tripartite_matches_independent_computation():
    from sparsegf2.analysis import analyze
    from sparsegf2.circuits import simulate as sim_run
    from sparsegf2.circuits.picture import setup_picture
    from sparsegf2.core.observables import tripartite_mutual_info

    cfg = CircuitConfig(graph_spec="complete", n=16, picture="pure_state", p=0.15, depth_factor=3)
    saved = sim_run(cfg, sample_seed=0, save_tableau=True)
    spec = setup_picture(cfg.picture, cfg.n)[1]
    from sparsegf2.analysis import reconstruct

    sim = reconstruct(saved)
    got = analyze(sim, spec, ["tripartite_mutual_info"])["tripartite_mutual_info"]
    q = 16 // 4
    want = tripartite_mutual_info(sim, range(q), range(q, 2 * q), range(2 * q, 3 * q))
    assert got == want
