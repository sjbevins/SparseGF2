"""analysis_compare over runs and over a sweep."""

import numpy as np

from sparsegf2 import analysis, api, mipt
from sparsegf2.agent.grammar import parse_calls
from sparsegf2.agent.tools.analysis import differing, metric_value

BASE = {
    "num_system": 8,
    "graph": {"family": "cycle"},
    "layers": 8,
    "reference": "purification",
    "checkpoints": 4,
    "probes": [{"kind": "code_dimension"}, {"kind": "erasure_first_failure", "samples": 16}],
}


def test_two_runs(run_tool, ctx):
    for p, name in ((0.05, "a"), (0.3, "b")):
        api.run_ensemble(
            {**BASE, "measurement_rate": p}, 30, seed=1, output_dir=str(ctx.root / name)
        )
    out = run_tool(
        "analysis_compare",
        sources=["a", "b"],
        metrics=["mean(code_dimension)", "phi(e=0.1)", "q_plus", "mu_D"],
    )
    assert not out.is_error, out.text
    assert "measurement_rate" in out.text.splitlines()[1]
    res = mipt.EnsembleResult.load(ctx.root / "a" / "result.npz")
    assert f"{res.values['code_dimension'][:, -1].mean():.4g}" in out.text
    assert "nan" not in out.text.lower()


def test_sweep_source(run_tool):
    sweep = run_tool(
        "ensemble_sweep",
        num_system=8,
        layers=4,
        trajectories=12,
        reference="purification",
        probes=["code_dimension"],
        seed=2,
        grid="measurement_rate=[0.1, 0.3]",
        background="never",
    )
    out = run_tool(
        "analysis_compare",
        source=sweep.data["output_dir"],
        metrics=["mean(code_dimension)", "fraction(code_dimension, above=0)"],
    )
    assert out.text.count("\n") >= 3 and "0.1" in out.text and "0.3" in out.text


def test_unknown_metric_suggests(run_tool, ctx):
    api.run_ensemble({**BASE, "measurement_rate": 0.1}, 4, seed=1, output_dir=str(ctx.root / "c"))
    out = run_tool("analysis_compare", sources=["c"], metrics=["mena(code_dimension)"])
    assert out.is_error and "did you mean mean" in out.text


def test_missing_values_are_dashes_with_a_note(run_tool, ctx):
    # high measurement rate: every trajectory purifies, so no code and no e_delta / mu_D
    spec = {**BASE, "measurement_rate": 0.9, "layers": 24, "checkpoints": [0, 12, 24]}
    api.run_ensemble(spec, 10, seed=1, output_dir=str(ctx.root / "d"))
    out = run_tool(
        "analysis_compare",
        sources=["d"],
        depth=24,
        metrics=["e_delta(0.01)", "mu_D", "phi(e=0.1, where=code)", "q_plus"],
    )
    assert not out.is_error, out.text
    assert "nan" not in out.text.lower().split() and "no value" in out.text
    res = mipt.EnsembleResult.load(ctx.root / "d" / "result.npz")
    assert metric_value(res, parse_calls(["mu_D"])[0], 24) == (None, None)


def test_run_without_the_needed_probe_is_a_note_not_a_crash(run_tool, ctx):
    spec = {**BASE, "measurement_rate": 0.1, "probes": [{"kind": "code_dimension"}]}
    api.run_ensemble(spec, 6, seed=1, output_dir=str(ctx.root / "e"))
    out = run_tool(
        "analysis_compare", sources=["e"], metrics=["phi(e=0.1)", "mean(code_dimension)"]
    )
    assert not out.is_error and "erasure_first_failure" in out.text


def test_survival_beyond_the_horizon_is_unknown(run_tool, ctx):
    spec = {**BASE, "measurement_rate": 0.02, "layers": 16, "checkpoints": "every_layer"}
    api.run_ensemble(spec, 30, seed=3, output_dir=str(ctx.root / "s"))
    res = mipt.EnsembleResult.load(ctx.root / "s" / "result.npz")
    tau = np.asarray(res.purification_layer)
    assert (tau < 0).any()  # heavy censoring
    out = run_tool(
        "analysis_compare", sources=["s"], metrics=["survival(at=100)", "survival(at=8)"]
    )
    assert not out.is_error, out.text
    assert "unknown beyond horizon 16" in out.text
    calls = parse_calls(["survival(at=100)", "survival(at=8)"])
    assert metric_value(res, calls[0], 16) == (None, None)
    km = analysis.kaplan_meier(np.where(tau >= 0, tau, res.final_layer), tau >= 0)
    i = np.searchsorted(km["times"], 8, side="right") - 1
    expect = float(km["survival"][i]) if i >= 0 else 1.0
    assert metric_value(res, calls[1], 16) == (expect, None)


def test_differing_flattens_nested_keys():
    a = {"measurement_rate": 0.1, "graph": {"family": "cycle", "seed": 1}, "probes": []}
    b = {"measurement_rate": 0.2, "graph": {"family": "cycle", "seed": 2}, "probes": []}
    assert differing([a, b]) == ["graph.seed", "measurement_rate"]


def test_bad_metric_arguments_are_tool_errors(run_tool, ctx):
    api.run_ensemble({**BASE, "measurement_rate": 0.1}, 4, seed=1, output_dir=str(ctx.root / "v"))
    for bad, word in (
        ("survival(layer=1)", "survival(at)"),
        ("phi(where=foo)", "phi(e, where)"),
        ("phi()", "needs e"),
        ("phi(e=x)", "number"),
        ("mean()", "needs probe"),
    ):
        out = run_tool("analysis_compare", sources=["v"], metrics=["q_plus", bad])
        assert out.is_error and word in out.text, (bad, out.text)
        assert "metrics[1]" in out.text


def _sweep(run_tool):
    return run_tool(
        "ensemble_sweep",
        num_system=8,
        layers=4,
        trajectories=6,
        reference="purification",
        probes=["code_dimension"],
        seed=2,
        grid="measurement_rate=[0.1, 0.2, 0.3, 0.4]",
        background="never",
    )


def test_missing_sweep_points_are_reported(run_tool, ctx):
    import json
    import shutil

    out_dir = _sweep(run_tool).data["output_dir"]
    root = ctx.root / out_dir
    points = json.loads((root / "sweep.json").read_text(encoding="utf-8"))["points"]
    shutil.rmtree(root / points[1]["dir"])
    (root / points[2]["dir"] / "result.npz").unlink()
    (root / points[2]["dir"] / "chunks").mkdir(exist_ok=True)
    out = run_tool("analysis_compare", source=out_dir, metrics=["mean(code_dimension)"])
    assert not out.is_error, out.text
    assert "2 of 4 sweep points" in out.text
    assert "not run" in out.text and "incomplete (chunks only" in out.text


def test_sweep_with_no_results_is_a_teaching_error(run_tool, ctx):
    import json
    import shutil

    out_dir = _sweep(run_tool).data["output_dir"]
    root = ctx.root / out_dir
    for p in json.loads((root / "sweep.json").read_text(encoding="utf-8"))["points"]:
        shutil.rmtree(root / p["dir"])
    out = run_tool("analysis_compare", source=out_dir, metrics=["q_plus"])
    assert out.is_error and "none of the 4 points" in out.text


def test_malformed_sweep_json(run_tool, ctx):
    (ctx.root / "bad").mkdir()
    (ctx.root / "bad" / "sweep.json").write_text('{"points": [{"params": {}}]}', encoding="utf-8")
    out = run_tool("analysis_compare", source="bad", metrics=["q_plus"])
    assert out.is_error and "bad sweep.json in bad" in out.text


def test_layer_note_when_horizons_differ(run_tool, ctx):
    for name, layers in (("h1", 8), ("h2", 16)):
        spec = {**BASE, "measurement_rate": 0.2, "layers": layers, "checkpoints": [0, layers]}
        api.run_ensemble(spec, 6, seed=1, output_dir=str(ctx.root / name))
    out = run_tool("analysis_compare", sources=["h1", "h2"], metrics=["mean(code_dimension)"])
    assert not out.is_error, out.text
    assert "layers differ (8, 16)" in out.text and "depth=" in out.text
    out = run_tool(
        "analysis_compare", sources=["h1", "h2"], depth=8, metrics=["mean(code_dimension)"]
    )
    assert "layers differ" not in out.text


def test_differing_key_in_one_spec_and_probes():
    a = {"measurement_rate": 0.1, "extra": 1, "probes": [{"kind": "code_dimension"}]}
    b = {"measurement_rate": 0.1, "probes": [{"kind": "code_dimension"}, {"kind": "entropy"}]}
    assert differing([a, b]) == ["extra", "probes"]
    assert differing([a, a]) == []
