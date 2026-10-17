"""The diagnostic figures (``sparsegf2.study.plots``): every figure draws headless from a tiny
store, writes SVG, PNG, CSV and JSON, is titled as a diagnostic, and shows exactly the numbers
it writes, which equal independent computations from the stored records (nested means and
cluster errors by loops, a product-limit estimator, per-graph measurements)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from _study_api_support import build, km_reference, nested_reference

pytest.importorskip("matplotlib")

from sparsegf2.study import api as sa  # noqa: E402
from sparsegf2.study import plots  # noqa: E402
from sparsegf2.study.store import TableauStore  # noqa: E402


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build(tmp_path_factory.mktemp("study_plots") / "store")


def table(result: dict) -> list[dict]:
    with open(result["data"]["csv"], encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(text: str) -> float:
    return math.nan if text == "" else float(text)


def check_files(result: dict) -> None:
    svg, png = Path(result["figures"]["svg"]), Path(result["figures"]["png"])
    assert svg.read_text(encoding="utf-8").lstrip().startswith("<?xml")
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    data = json.loads(Path(result["data"]["json"]).read_text(encoding="utf-8"))
    assert data["title"] == result["title"] and data["caption"] == result["caption"]
    assert result["title"].startswith("Diagnostic:")
    assert "does not by itself establish" in result["caption"]
    assert len(data["rows"]) == len(table(result)) == result["points"]


def test_k_trace_shows_the_nested_mean_of_the_stored_traces(store, tmp_path):
    out = plots.k_trace(store, tmp_path, where={"n": 6}, return_figure=True)
    check_files(out)
    rows = table(out)
    with TableauStore.open(store) as s:
        for p in ("0.1", "0.2", "0.3"):
            keys = s.trajectory_keys(n=6, beta="0.2", p=p)
            traces = np.array([s.k_trace(k) for k in keys]) / 6
            mine = [r for r in rows if r["p"] == p]
            assert [int(r["t"]) for r in mine] == list(range(25))
            for r in mine:
                t = int(r["t"])
                mu, se, _ = nested_reference([k.g for k in keys], [k.c for k in keys], traces[:, t])
                assert num(r["mean"]) == pytest.approx(mu) and num(r["se"]) == pytest.approx(se)
    ax = out["figure_object"].axes[0]
    assert len(ax.lines) == 3 and len(ax.collections) == 3  # a line and a band per rate
    first = [num(r["mean"]) for r in rows if r["p"] == "0.1"]
    assert ax.lines[0].get_ydata().tolist() == pytest.approx(first)
    assert ax.get_legend() is not None


def test_survival_is_the_product_limit_estimate_with_censoring_marked(store, tmp_path):
    out = plots.survival(store, tmp_path, where={"n": 8}, ell=0)
    check_files(out)
    rows = table(out)
    data = json.loads(Path(out["data"]["json"]).read_text(encoding="utf-8"))
    for cens in data["censoring"]:
        with TableauStore.open(store) as s:
            tau = s.observations("first_passage", field="tau", n=8, beta="0.2", p=cens["p"])
            seen = s.observations("first_passage", field="observed", n=8, beta="0.2",
                                  p=cens["p"])["observed"][:, 0].astype(bool)  # fmt: skip
        times = np.where(seen, tau["tau"][:, 0], 32.0)
        t_ref, s_ref = km_reference(times, seen)
        mine = [r for r in rows if r["p"] == cens["p"]]
        assert [num(r["t"]) for r in mine] == [0.0, *t_ref]
        assert [num(r["survival"]) for r in mine] == pytest.approx([1.0, *s_ref])
        assert cens["censored"] == int((~seen).sum())
    assert "right-censored" in out["caption"]


def test_d_cyc_states_and_applies_the_zero_extension(store, tmp_path):
    out = plots.d_cyc(store, tmp_path, where={"n": 6})
    check_files(out)
    assert "d_cyc = 0 where k_t = 0" in out["caption"]
    with TableauStore.open(store) as s:
        for r in table(out):
            sel = dict(n=6, beta="0.2", p=r["p"], t=int(r["t"]))
            zero = s.observations("d_cyc", field="zero_extended", **sel)
            code = s.observations("d_cyc", field="has_code", **sel)
            assert num(r["mean"]) == pytest.approx(
                nested_reference(zero["g"], zero["c"], zero["zero_extended"])[0]
            )
            assert num(r["has_code_fraction"]) == pytest.approx(
                nested_reference(code["g"], code["c"], code["has_code"])[0]
            )


def test_peak_distribution_counts_missing_and_horizon_peaks(store, tmp_path):
    out = plots.peak_distribution(store, tmp_path, where={"n": 8})
    check_files(out)
    rows = table(out)
    summary = json.loads(Path(out["data"]["json"]).read_text(encoding="utf-8"))["summary"]
    with TableauStore.open(store) as s:
        for entry in summary:
            got = s.observations("peak_d_cyc", n=8, beta="0.2", p=entry["p"])
            mine = [r for r in rows if r["p"] == entry["p"]]
            defined = np.isfinite(got["value"])
            assert sorted(num(r["t_star"]) for r in mine) == sorted(got["value"][defined])
            last = num(mine[-1]["cumulative"]) if mine else 0.0
            assert last == pytest.approx(1.0 - entry["missing_fraction"])
            weights = [num(r["weight"]) for r in mine]
            assert sum(weights) == pytest.approx(last)
            assert entry["missing_fraction"] == pytest.approx(1.0 - defined.mean())


def test_peak_vs_tau1_pairs_the_same_trajectories(store, tmp_path):
    out = plots.peak_vs_tau1(store, tmp_path, where={"p": "0.3"})
    check_files(out)
    rows = table(out)
    counts = json.loads(Path(out["data"]["json"]).read_text(encoding="utf-8"))["counts"]
    with TableauStore.open(store) as s:
        for n in (6, 8):
            fp = s.observations("first_passage", field="tau", n=n, beta="0.2", p="0.3")
            ok = s.observations("first_passage", field="observed", n=n, beta="0.2",
                                p="0.3")["observed"][:, 1].astype(bool)  # fmt: skip
            pk = s.observations("peak_d_cyc", n=n, beta="0.2", p="0.3")
            expected = {
                (int(g), int(c)): (float(t1), float(ts))
                for g, c, t1, ts, seen in zip(fp["g"], fp["c"], fp["tau"][:, 1], pk["value"], ok,
                                              strict=True)
                if seen and t1 > 0 and np.isfinite(ts)
            }  # fmt: skip
            mine = {(int(r["g"]), int(r["c"])): (num(r["tau_1"]), num(r["t_star"]))
                    for r in rows if int(r["n"]) == n}  # fmt: skip
            assert mine == expected
            entry = next(c for c in counts if c["n"] == n)
            assert entry["pairs"] == len(expected)
            assert entry["pairs"] + entry["tau1_censored"] + entry["tstar_missing"] + entry[
                "tau1_zero"
            ] == len(fp["g"])


def test_versus_p_uses_the_aggregates(store, tmp_path):
    out = plots.versus_p(store, tmp_path, observable="k")
    check_files(out)
    agg = sa.aggregate_arrays(store, "k")
    for cond in agg["conditions"]:
        (row,) = [r for r in table(out) if int(r["n"]) == cond["n"] and r["p"] == cond["p"]]
        assert num(row["value"]) == pytest.approx(cond["mean"][-1])
        assert num(row["se"]) == pytest.approx(cond["se"][-1])
    median = plots.versus_p(store, tmp_path, observable="first_passage", statistic="median",
                            ell=1, stem="median")  # fmt: skip
    assert "no band" in median["caption"] and all(num(r["se"]) != num(r["se"])
                                                  for r in table(median))  # fmt: skip
    peaks = plots.versus_p(store, tmp_path, observable="peak_d_cyc", statistic="missing_fraction",
                           stem="missing")  # fmt: skip
    assert all(0 <= num(r["value"]) <= 1 for r in table(peaks))
    with pytest.raises(ValueError, match="statistic must be one of"):
        plots.versus_p(store, tmp_path, observable="first_passage", statistic="mean")
    with pytest.raises(ValueError, match="depth 3 is not stored"):
        plots.versus_p(store, tmp_path, observable="k", depth=3)


def test_graph_measures_are_equal_weight_means_over_the_stored_graphs(store, tmp_path):
    out = plots.graph_measures(store, tmp_path)
    check_files(out)
    with TableauStore.open(store) as s:
        for r in table(out):
            values = [s.graph_measures(int(r["n"]), r["beta"], g)[r["measure"]] for g in range(3)]
            mean = sum(values) / 3
            se = math.sqrt(sum((v - mean) ** 2 for v in values) / 6)
            assert num(r["mean"]) == pytest.approx(mean) and num(r["se"]) == pytest.approx(se)
            assert int(r["graphs"]) == 3
    with pytest.raises(ValueError, match="measures must be among"):
        plots.graph_measures(store, tmp_path, measures=["diameter"])


def test_collapse_draws_the_reports_scaled_points(store, tmp_path):
    report = sa.fss_report(store, "peak_d_cyc", "0.2", horizon_policy="include",
                           max_horizon_fraction=1.0, starts=1, bootstrap=2,
                           leave_one_out=False)  # fmt: skip
    fit = report["results"][0]["fit"]
    assert fit["status"] in ("ok", "at_bound", "not_converged")  # a fit with parameters
    out = plots.collapse(report, tmp_path, return_figure=True)
    check_files(out)
    for r in table(out):
        n, p = int(r["n"]), num(r["p_float"])
        point = next(pt for pt in report["results"][0]["points"]
                     if pt["n"] == n and pt["p"] == r["p"])  # fmt: skip
        assert num(r["x"]) == pytest.approx((p - fit["p_c"]) * n ** (1 / fit["nu"]))
        assert num(r["Y"]) == pytest.approx(point["y"] * n ** (-fit["z"]))
        assert num(r["dY"]) == pytest.approx(point["se"] * n ** (-fit["z"]))
    assert len(table(out)) == fit["total_points"]
    # a polynomial fit also draws its master curve F(x) = sum_k a_k x^k on its window
    poly = json.loads(json.dumps(sa._exact(sa._report_file(report))))  # as the file holds it
    poly["results"][0]["fit"].update(coefficients=[1.0, -2.0, 0.5], window_range=[-0.2, 0.3])
    drawn = plots.collapse(poly, tmp_path, stem="poly", return_figure=True)
    (curve,) = [ln for ln in drawn["figure_object"].axes[0].lines
                if ln.get_label() == "fitted F (window)"]  # fmt: skip
    x = np.asarray(curve.get_xdata())
    assert x[0] == pytest.approx(-0.2) and x[-1] == pytest.approx(0.3)
    assert np.asarray(curve.get_ydata()) == pytest.approx(1.0 - 2.0 * x + 0.5 * x**2)
    with pytest.raises(ValueError, match="must be an fss report"):
        plots.collapse({"schema": "other"}, tmp_path)


def test_plots_are_reproducible_and_refuse_too_many_series(store, tmp_path):
    outs = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        outs.append(plots.k_trace(store, tmp_path / name, where={"n": 6, "p": "0.1"}))
    a, b = outs
    for kind in ("svg", "png"):
        assert Path(a["figures"][kind]).read_bytes() == Path(b["figures"][kind]).read_bytes()
    assert Path(a["data"]["csv"]).read_bytes() == Path(b["data"]["csv"]).read_bytes()
    with pytest.raises(ValueError, match="at most 8 are distinguishable"):
        plots._colors(9)


def test_the_api_plot_function(store, tmp_path):
    out = sa.plot(store, "d_cyc", str(tmp_path), {"where": {"n": 8, "p": "0.1"}})
    assert out["kind"] == "d_cyc" and Path(out["figures"]["png"]).is_file()
    assert len(json.dumps(out)) < 8000
    with pytest.raises(ValueError, match="kind must be one of"):
        sa.plot(store, "pie", str(tmp_path))
    with pytest.raises(ValueError, match="report: no such file"):
        sa.plot(None, "collapse", str(tmp_path), {"report": str(tmp_path / "none.json")})
