"""analysis_erasure and analysis_purification against the analysis module called directly."""

import numpy as np
import pytest

from sparsegf2 import analysis, api, mipt

SPEC = {
    "num_system": 12,
    "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": 0, "realizations": 6},
    "circuits_per_graph": 4,
    "layers": 24,
    "measurement_rate": 0.05,
    "schedule": "random_edges",
    "reference": "purification",
    "checkpoints": 12,
    "probes": [
        {"kind": "code_dimension"},
        {"kind": "erasure_first_failure", "samples": 64},
        {"kind": "uncorrectable_region_counts", "max_size": 2},
    ],
}


@pytest.fixture
def erasure_dir(ctx):
    api.run_ensemble(SPEC, seed=1, output_dir=str(ctx.root / "e"))
    return mipt.EnsembleResult.load(ctx.root / "e" / "result.npz")


def test_phi_e_delta_and_mu_match_the_library(run_tool, erasure_dir):
    res = erasure_dir
    out = run_tool("analysis_erasure", source="e", rates=[0.05, 0.1], delta=0.01)
    assert not out.is_error, out.text
    n = 12
    first = res.values["erasure_first_failure"][:, -1]
    counts = res.values["uncorrectable_region_counts[2]"][:, -1]
    k = res.values["code_dimension"][:, -1]
    phi = analysis.erasure_failure_probability(first, n, [0.05, 0.1], exact_counts=counts)
    code = k > 0
    assert f"{phi[:, 1].mean():.4g}" in out.text
    last = out.data["rows"][-1]
    assert last["q_plus"] == pytest.approx(code.mean(), rel=1e-5)
    if code.any():
        e_d = analysis.erasure_tolerance(first[code], n, 0.01, exact_counts=counts[code])
        assert f"{e_d:.4g}" in out.text
        mu = analysis.mean_first_failure(first[code], n, exact_counts=counts[code]).mean()
        assert f"{mu:.4g}" in out.text
        assert last["e_delta"] == pytest.approx(e_d, rel=1e-5)
        assert last["phi_code"][1][0] == pytest.approx(phi[code, 1].mean(), rel=1e-5)


def test_requires_first_failure(run_tool, ctx):
    api.run_ensemble(
        {**SPEC, "probes": [{"kind": "code_dimension"}]}, seed=1, output_dir=str(ctx.root / "k")
    )
    out = run_tool("analysis_erasure", source="k")
    assert out.is_error and "erasure_first_failure" in out.text


def test_no_code_left_is_stated_not_nan(run_tool, ctx):
    spec = {
        **SPEC,
        "num_system": 8,
        "graph": {"family": "cycle"},
        "layers": 60,
        "measurement_rate": 0.5,
        "stop_when_purified": True,
        "checkpoints": [0, 30, 60],
        "probes": [
            {"kind": "code_dimension"},
            {"kind": "erasure_first_failure", "samples": 16},
            {"kind": "uncorrectable_region_counts", "max_size": 2},
        ],
    }
    spec.pop("circuits_per_graph")
    api.run_ensemble(spec, 20, seed=3, output_dir=str(ctx.root / "z"))
    res = mipt.EnsembleResult.load(ctx.root / "z" / "result.npz")
    k = res.values["code_dimension"][:, -1]
    assert not (k > 0).any()  # the configuration purifies everywhere
    out = run_tool("analysis_erasure", source="z", rates=[0.1])
    assert not out.is_error, out.text
    last = out.data["rows"][-1]
    assert last["layer"] == 60 and last["q_plus"] == 0.0
    first = res.values["erasure_first_failure"][:, -1]
    counts = res.values["uncorrectable_region_counts[2]"][:, -1]
    phi = analysis.erasure_failure_probability(first, 8, [0.1], exact_counts=counts)
    assert last["phi_all"][0][0] == pytest.approx(phi[:, 0].mean(), rel=1e-5) == 0.0
    assert "no selected trajectory holds a code at layer 60" in out.text
    assert "nan" not in out.text.lower()
    assert "phi_code" not in last and "e_delta" not in last
    # an empty selection by 'where' is stated too, and does not crash
    out2 = run_tool("analysis_erasure", source="z", rates=[0.1], where="unpurified", depths="[60]")
    assert not out2.is_error, out2.text
    assert out2.data["rows"][0]["count"] == 0
    assert "no selected trajectory holds a code at layer 60" in out2.text


def test_purification(run_tool, ctx):
    spec = {
        "num_system": 8,
        "graph": {"family": "cycle"},
        "layers": 64,
        "measurement_rate": 0.3,
        "reference": "single",
        "checkpoints": "every_layer",
        "stop_when_purified": True,
        "probes": [{"kind": "code_dimension"}],
    }
    api.run_ensemble(spec, 200, seed=2, output_dir=str(ctx.root / "p"))
    res = mipt.EnsembleResult.load(ctx.root / "p" / "result.npz")
    tau = res.purification_layer
    out = run_tool("analysis_purification", source="p", at_layers=[8, 32])
    assert not out.is_error, out.text
    assert f"censored {np.mean(tau < 0):.3g}" in out.text
    assert "survival at layer 8" in out.text
    km = analysis.kaplan_meier(np.where(tau >= 0, tau, res.final_layer), tau >= 0)
    i = np.searchsorted(km["times"], 8, side="right") - 1
    assert out.data["survival_at"]["8"] == pytest.approx(km["survival"][i], rel=1e-5)
    assert out.data["purified"] == int((tau >= 0).sum())
