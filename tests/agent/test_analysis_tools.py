"""analysis_summary and analysis_probe against direct numpy on the saved arrays."""

import numpy as np
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


def approx(x):
    return pytest.approx(x, rel=1e-5)  # the rows are rounded to 6 digits


def rows(out):
    return {r["layer"]: r for r in out.data["rows"]}


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
    assert none.is_error
    assert "trajectory_range [0] matches none of the saved trajectories [3]" in none.text


@pytest.fixture
def res(ctx, run_dir):
    return mipt.EnsembleResult.load(ctx.root / "r" / "result.npz")


def test_numbers_come_from_the_structured_rows(run_tool, run_dir, res):
    k = res.values["code_dimension"]
    r = rows(run_tool("analysis_probe", source=run_dir, probe="code_dimension"))
    layers = [int(x) for x in res.layers("code_dimension")]
    assert sorted(r) == layers
    for c, layer in enumerate(layers):
        assert r[layer]["count"] == len(k)
        assert r[layer]["mean"] == approx(k[:, c].mean())
        assert r[layer]["stderr"] == approx(k[:, c].std(ddof=1) / np.sqrt(len(k)))


def test_nan_after_stop_when_purified(run_tool, ctx):
    spec = {
        **SPEC,
        "stop_when_purified": True,
        "layers": 24,
        "checkpoints": 6,
        "measurement_rate": 0.3,
    }
    api.run_ensemble(spec, seed=5, output_dir=str(ctx.root / "s"))
    res = mipt.EnsembleResult.load(ctx.root / "s" / "result.npz")
    h = res.values["half_cut_entropy"]
    assert np.isnan(h).any(), "the fixture must actually stop some trajectories"
    layers = [int(x) for x in res.layers("half_cut_entropy")]
    r = rows(run_tool("analysis_probe", source="s", probe="half_cut_entropy"))
    for c, layer in enumerate(layers):
        v = h[:, c][~np.isnan(h[:, c])]
        assert r[layer]["count"] == v.size
        if v.size > 1:
            assert r[layer]["mean"] == approx(v.mean())
            assert r[layer]["stderr"] == approx(v.std(ddof=1) / np.sqrt(v.size))
    assert r[layers[-1]]["count"] < len(h)


def test_probe_with_its_own_checkpoints_and_where(run_tool, ctx):
    spec = {
        **SPEC,
        "probes": [{"kind": "code_dimension"}, {"kind": "half_cut_entropy", "checkpoints": [4, 8]}],
    }
    api.run_ensemble(spec, seed=2, output_dir=str(ctx.root / "p"))
    res = mipt.EnsembleResult.load(ctx.root / "p" / "result.npz")
    assert [int(x) for x in res.layers("half_cut_entropy")] == [4, 8]
    k_layers = [int(x) for x in res.layers("code_dimension")]
    h = res.values["half_cut_entropy"]
    r = rows(run_tool("analysis_probe", source="p", probe="half_cut_entropy"))
    assert sorted(r) == [4, 8]
    assert r[8]["mean"] == approx(h[:, 1].mean())
    k = res.values["code_dimension"]
    w = rows(run_tool("analysis_probe", source="p", probe="half_cut_entropy", where="code"))
    for c, layer in enumerate([4, 8]):
        sel = k[:, k_layers.index(layer)] > 0
        assert w[layer]["count"] == sel.sum()
        assert w[layer]["mean"] == approx(h[sel, c].mean())
    off = run_tool("analysis_probe", source="p", probe="half_cut_entropy", depths="[5]")
    assert off.is_error and "were not recorded" in off.text


def test_ranges_follow_t_floor_div_circuits_per_graph(run_tool, run_dir, res):
    per_graph = SPEC["circuits_per_graph"]
    k = res.values["code_dimension"][:, -1]
    t = np.arange(len(k))
    g = rows(
        run_tool(
            "analysis_probe",
            source=run_dir,
            probe="code_dimension",
            graph_range="1:3",
            depths="[12]",
        )
    )[12]
    want = (t // per_graph >= 1) & (t // per_graph < 3)
    assert g["count"] == want.sum() == 2 * per_graph
    assert g["mean"] == approx(k[want].mean())
    tr = rows(
        run_tool(
            "analysis_probe",
            source=run_dir,
            probe="code_dimension",
            trajectory_range="[0, 4, 5, 11]",
            depths="[12]",
        )
    )[12]
    assert tr["count"] == 4 and tr["mean"] == approx(k[[0, 4, 5, 11]].mean())
    both = rows(
        run_tool(
            "analysis_probe",
            source=run_dir,
            probe="code_dimension",
            graph_range="[3]",
            trajectory_range="0:11",
            depths="[12]",
        )
    )[12]
    want = (t // per_graph == 3) & (t < 11)
    assert both["count"] == want.sum() and both["mean"] == approx(k[want].mean())


def test_empty_range_is_an_error_naming_what_exists(run_tool, run_dir):
    g = run_tool("analysis_probe", source=run_dir, probe="code_dimension", graph_range="[9]")
    assert g.is_error and "graph_range [9] matches none of the saved graphs 0..3" in g.text
    t = run_tool("analysis_probe", source=run_dir, probe="code_dimension", trajectory_range="50:60")
    assert t.is_error and "matches none of the saved trajectories 0..11" in t.text


def test_open_ended_slices_are_explained(run_tool, run_dir):
    for bad in ("::2", "5:", ":10"):
        out = run_tool(
            "analysis_probe", source=run_dir, probe="code_dimension", trajectory_range=bad
        )
        assert out.is_error and "start and stop must be explicit" in out.text, bad


def test_purified_and_unpurified_with_censoring(run_tool, run_dir, res):
    tau = np.asarray(res.purification_layer)
    assert (tau < 0).any() and (tau >= 0).any(), "the fixture needs both purified and censored"
    k = res.values["code_dimension"]
    layers = [int(x) for x in res.layers("code_dimension")]
    pur = rows(run_tool("analysis_probe", source=run_dir, probe="code_dimension", where="purified"))
    unp = rows(
        run_tool("analysis_probe", source=run_dir, probe="code_dimension", where="unpurified")
    )
    for c, layer in enumerate(layers):
        done = (tau >= 0) & (tau <= layer)
        assert pur[layer]["count"] == done.sum()
        assert unp[layer]["count"] == (~done).sum()
        assert pur[layer]["count"] + unp[layer]["count"] == len(tau)
        if (~done).any():
            assert unp[layer]["mean"] == approx(k[~done, c].mean())


def test_empty_where_is_not_an_error(run_tool, run_dir):
    out = run_tool(
        "analysis_probe",
        source=run_dir,
        probe="code_dimension",
        where="code_dimension > 100",
        depths="[12]",
    )
    assert not out.is_error, out.text
    assert out.data["rows"][0]["count"] == 0
    assert "no trajectories match where=code_dimension > 100" in out.text


def test_other_statistics(run_tool, run_dir, res):
    k = res.values["code_dimension"][:, -1]
    med = rows(
        run_tool(
            "analysis_probe",
            source=run_dir,
            probe="code_dimension",
            statistic="median",
            depths="[12]",
        )
    )[12]
    assert med["median"] == np.median(k)
    fp = rows(
        run_tool(
            "analysis_probe",
            source=run_dir,
            probe="code_dimension",
            statistic="fraction_positive",
            depths="[12]",
        )
    )[12]
    assert fp["fraction_positive"] == approx((k > 0).mean())
    hist = rows(
        run_tool(
            "analysis_probe",
            source=run_dir,
            probe="code_dimension",
            statistic="histogram",
            depths="[12]",
        )
    )[12]["histogram"]
    vals, counts = np.unique(k, return_counts=True)
    assert {float(a): b for a, b in hist.items()} == {
        float(a): int(b) for a, b in zip(vals, counts, strict=True)
    }


def test_scalar_where(run_tool, run_dir, res):
    k = res.values["code_dimension"][:, -1]
    h = res.values["half_cut_entropy"][:, -1]
    r = rows(
        run_tool(
            "analysis_probe",
            source=run_dir,
            probe="half_cut_entropy",
            where="code_dimension > 0",
            depths="[12]",
        )
    )[12]
    assert r["count"] == (k > 0).sum() and r["mean"] == approx(h[k > 0].mean())


def test_a_sweep_folder_is_refused(run_tool, ctx):
    (ctx.root / "sw").mkdir()
    (ctx.root / "sw" / "sweep.json").write_text("{}", encoding="utf-8")
    out = run_tool("analysis_summary", source="sw")
    assert out.is_error and "is a sweep" in out.text and "analysis_compare" in out.text


def test_unreadable_files_are_explained(run_tool, ctx):
    (ctx.root / "bad").mkdir()
    (ctx.root / "bad" / "trajectory.json").write_text("{not json", encoding="utf-8")
    out = run_tool("analysis_summary", source="bad")
    assert out.is_error and "could not read" in out.text and "SparseGF2 result" in out.text
