"""analysis_summary and analysis_probe against direct numpy on the saved arrays."""

import pytest

from sparsegf2 import api, mipt

SPEC = {
    "num_system": 8,
    "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 1, "realizations": 4},
    "circuits_per_graph": 3,
    "layers": 12,
    "measurement_rate": 0.15,
    "reference": "purification",
    "checkpoints": 4,
    "probes": [
        {"kind": "code_dimension"},
        {"kind": "half_cut_entropy"},
        {"kind": "entropy_profile"},
    ],
}


@pytest.fixture
def run_dir(ctx):
    api.run_ensemble(SPEC, seed=3, output_dir=str(ctx.root / "r"))
    return "r"


def test_summary(run_tool, run_dir):
    out = run_tool("analysis_summary", source=run_dir)
    assert out.text.startswith("analysis_summary: n=8, layers=12") and "code_dimension" in out.text


def test_probe_mean_matches_numpy(run_tool, ctx, run_dir):
    res = mipt.EnsembleResult.load(ctx.root / "r" / "result.npz")
    out = run_tool("analysis_probe", source=run_dir, probe="code_dimension", statistic="mean")
    k = res.values["code_dimension"]
    assert f"{k[:, -1].mean():.4g}" in out.text


def test_where_trajectories_and_graphs(run_tool, ctx, run_dir):
    res = mipt.EnsembleResult.load(ctx.root / "r" / "result.npz")
    k = res.values["code_dimension"][:, -1]
    out = run_tool(
        "analysis_probe", source=run_dir, probe="half_cut_entropy", where="code", depths="[12]"
    )
    s = res.values["half_cut_entropy"][k > 0, -1]
    assert f"{s.mean():.4g}" in out.text
    g = run_tool(
        "analysis_probe", source=run_dir, probe="code_dimension", graph_range="0:2", depths="[12]"
    )
    assert f"{k[res.graph_index < 2].mean():.4g}" in g.text
    bad = run_tool("analysis_probe", source=run_dir, probe="code_dimension", where="k >> 0")
    assert bad.is_error and "where" in bad.text


def test_vector_probe_and_statistics(run_tool, run_dir):
    out = run_tool("analysis_probe", source=run_dir, probe="entropy_profile")
    assert "m=0..8" in out.text
    q = run_tool("analysis_probe", source=run_dir, probe="code_dimension", statistic="quantiles")
    assert "q50" in q.text


def test_traj_run_record_folder_with_global_trajectory_number(run_tool, ctx):
    rec = run_tool(
        "traj_run",
        num_system=8,
        layers=6,
        measurement_rate=0.2,
        reference="purification",
        trajectory=3,
        seed=5,
        checkpoints="[3, 6]",
        probes="code_dimension",
        output_dir="rec",
    )
    assert not rec.is_error, rec.text
    res = mipt.EnsembleResult.load(ctx.root / "rec" / "result.npz")
    assert list(res.trajectories) == [3]
    out = run_tool("analysis_summary", source="rec")
    assert not out.is_error, out.text
    assert out.text.startswith("analysis_summary: n=8, layers=6")
    p = run_tool("analysis_probe", source="rec", probe="code_dimension", depths="[6]")
    assert not p.is_error, p.text
    assert f"{res.values['code_dimension'][0, -1]:.4g}" in p.text
    sel = run_tool(
        "analysis_probe", source="rec", probe="code_dimension", depths="[6]", trajectory_range="[3]"
    )
    assert sel.text == p.text
    none = run_tool(
        "analysis_probe", source="rec", probe="code_dimension", depths="[6]", trajectory_range="[0]"
    )
    assert not none.is_error
