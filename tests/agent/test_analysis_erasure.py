"""analysis_erasure and analysis_purification against the analysis module called directly."""

import numpy as np
import pytest

from sparsegf2 import analysis, api, mipt
from sparsegf2.agent.tools.analysis import survival_at

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


MIXED = {
    **SPEC,
    "num_system": 10,
    "circuits_per_graph": 5,
    "layers": 40,
    "measurement_rate": 0.2,
    "stop_when_purified": True,
    "checkpoints": [0, 10, 20, 30, 40],
}


def approx(x):
    return pytest.approx(x, rel=1e-5, abs=1e-9)  # data rows are rounded to 6 digits


def test_phi_e_delta_and_mu_match_the_library(run_tool, ctx):
    api.run_ensemble(MIXED, seed=1, output_dir=str(ctx.root / "m"))
    res = mipt.EnsembleResult.load(ctx.root / "m" / "result.npz")
    n, rates = 10, [0.05, 0.1]
    out = run_tool("analysis_erasure", source="m", rates=rates, delta=0.01)
    assert not out.is_error, out.text
    rows = {r["layer"]: r for r in out.data["rows"]}
    differing = 0
    for c, layer in enumerate(res.layers("erasure_first_failure")):
        first = res.values["erasure_first_failure"][:, c]
        counts = res.values["uncorrectable_region_counts[2]"][:, c]
        code = res.values["code_dimension"][:, c] > 0
        r = rows[int(layer)]
        assert r["q_plus"] == approx(code.mean())
        phi = analysis.erasure_failure_probability(first, n, rates, exact_counts=counts)
        for i in range(len(rates)):
            assert r["phi_all"][i][0] == approx(phi[:, i].mean())
        if not code.any():
            continue
        for i in range(len(rates)):
            assert r["phi_code"][i][0] == approx(phi[code, i].mean())
        e_d = analysis.erasure_tolerance(first[code], n, 0.01, exact_counts=counts[code])
        assert r["e_delta"] == approx(e_d)
        mu = analysis.mean_first_failure(first[code], n, exact_counts=counts[code])
        assert r["mu_D"][0] == approx(mu.mean())
        differing += not np.isclose(phi[:, 1].mean(), phi[code, 1].mean(), rtol=1e-3)
    # the fixture really separates "all circuits" from "k>0" at a mixed layer
    c30 = list(res.layers("code_dimension")).index(30)
    q30 = (res.values["code_dimension"][:, c30] > 0).mean()
    assert 0 < q30 < 1
    assert res.values["uncorrectable_region_counts[2]"][:, c30].any()
    assert differing >= 1
    assert "exact d:" in out.text and "next:" in out.text


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
    assert "no trajectory selected at layer 60" in out2.text
    assert "holds a code" not in out2.text


CENSORED = {
    "num_system": 8,
    "graph": {"family": "cycle"},
    "layers": 16,
    "measurement_rate": 0.02,
    "reference": "single",
    "checkpoints": "every_layer",
    "stop_when_purified": True,
    "probes": [{"kind": "code_dimension"}],
}


def test_purification_censored(run_tool, ctx):
    api.run_ensemble(CENSORED, 200, seed=2, output_dir=str(ctx.root / "c"))
    res = mipt.EnsembleResult.load(ctx.root / "c" / "result.npz")
    tau = res.purification_layer
    assert 0.5 < np.mean(tau < 0) < 1  # mostly censored, some purified
    out = run_tool("analysis_purification", source="c", at_layers=[8, 500])
    assert not out.is_error, out.text
    km = analysis.kaplan_meier(np.where(tau >= 0, tau, res.final_layer), tau >= 0)
    assert out.data["tau_median_all"] is None
    assert out.data["censored_fraction"] == approx(np.mean(tau < 0))
    assert np.allclose(out.data["km_times"], km["times"])
    assert np.allclose(out.data["km_survival"], km["survival"], rtol=1e-5)
    summary = api._purification_summary(tau, res.checkpoint_layers)
    assert summary["tau_median_all"] is None
    layers, s = res.survival()
    i = list(layers).index(8)
    assert out.data["survival_at"]["8"] == approx(s[i])
    # beyond the 16-layer horizon the survival is unknown, not S(16)
    assert out.data["survival_at"]["500"] is None
    assert "survival at layer 500: unknown beyond the horizon (last checked layer 16" in out.text
    assert "upper bound" in out.text and "next:" in out.text


def test_purification_median_matches_summary(run_tool, ctx):
    spec = {**CENSORED, "measurement_rate": 0.3, "layers": 64}
    api.run_ensemble(spec, 200, seed=2, output_dir=str(ctx.root / "d"))
    res = mipt.EnsembleResult.load(ctx.root / "d" / "result.npz")
    tau = res.purification_layer
    assert (tau < 0).sum() == 0
    summary = api._purification_summary(tau, res.checkpoint_layers)
    out = run_tool("analysis_purification", source="d", at_layers=[500])
    assert summary["tau_median_all"] is not None
    assert out.data["tau_median_all"] == summary["tau_median_all"]
    # nothing is censored: S past the last event is exactly 0
    assert out.data["survival_at"]["500"] == 0.0


def test_survival_between_checkpoints_is_an_upper_bound(run_tool, ctx):
    spec = {**CENSORED, "measurement_rate": 0.3, "layers": 32, "checkpoints": 8}
    api.run_ensemble(spec, 200, seed=2, output_dir=str(ctx.root / "s"))
    res = mipt.EnsembleResult.load(ctx.root / "s" / "result.npz")
    out = run_tool("analysis_purification", source="s", at_layers=[5, 8])
    layers, s = res.survival()
    assert out.data["survival_at"]["5"] == approx(s[list(layers).index(0)])
    assert out.data["survival_upper_bound_layers"] == [5]
    assert "survival at layer 5" in out.text and "between checkpoints" in out.text
    assert "between checkpoints" not in out.text.split("survival at layer 8")[1]


def test_survival_at_helper():
    km = {"times": np.array([2, 4]), "survival": np.array([0.5, 0.0])}
    assert survival_at(km, 1, horizon=10, censored=3) == 1.0
    assert survival_at(km, 3, horizon=10, censored=3) == 0.5
    assert survival_at(km, 11, horizon=10, censored=3) is None
    assert survival_at(km, 11, horizon=10, censored=0) == 0.0
