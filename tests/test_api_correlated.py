"""The correlated_erasure probe through the JSON interface (``api``, so the CLI and MCP).

Checked here: the summary (one P_{C,G,J}(m) curve per coupling at the last checkpoint,
averaged over the trajectories that hold a code when a code_dimension probe covers the
layers, else over all with a note; the sizes shown, ``truncated`` when sizes are left out;
a vs_depth grid; the sampler and the graph-side mixing note), ``load_results``,
``capabilities``, the cost estimate, a resume with other couplings refused (and one with
the same probe written differently continued), and the validation messages.
"""

from __future__ import annotations

import json
import math

import numba
import numpy as np
import pytest

from sparsegf2 import api, erasures, mipt

SMALL_JSON = {
    "num_system": 12,
    "graph": {"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 2},
    "layers": 24,
    "measurement_rate": 0.08,
    "schedule": "random_edges",
    "reference": "purification",
    "checkpoints": [8, 16, 24],
}


def _json_probe(**kw) -> dict:
    return {"kind": "correlated_erasure", "couplings": [0, 1.0], "sizes": "2:11:3",
            "samples": 20, "burn_in": 20, **kw}  # fmt: skip


def test_json_summary_conditions_on_a_code(tmp_path):
    spec = dict(SMALL_JSON, probes=[{"kind": "code_dimension"}, _json_probe()])
    out = api.run_ensemble(spec, 8, seed=3, output_dir=str(tmp_path / "run"))
    entry = out["probes"]["correlated_erasure"]
    res = mipt.EnsembleResult.load(tmp_path / "run" / "result.npz")
    couplings, sizes, values = res.correlated()
    assert sizes.tolist() == [2, 5, 8] and couplings.tolist() == [0.0, 1.0]
    assert entry["couplings"] == [0.0, 1.0] and entry["sizes"] == [2, 5, 8]
    assert "P_{C,G,J}(m)" in entry["what"] and "k > 0" in entry["what"]
    assert entry["sampler"].startswith("chain")
    layer = entry["at_layer"]
    c = res.layers("correlated_erasure").tolist().index(layer)
    holds = res.values["code_dimension"][:, c] > 0
    assert entry["count_with_code"] == int(holds.sum()) > 1
    assert entry["count"] == 8
    mean = values[holds, c].mean(axis=0)
    std = values[holds, c].std(axis=0, ddof=1) / math.sqrt(holds.sum())
    np.testing.assert_allclose(np.array(entry["mean"], dtype=float), mean, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(np.array(entry["stderr"], dtype=float), std, rtol=1e-5, atol=1e-9)
    assert 0 < mean.mean() < 1  # not vacuous
    grid = entry["vs_depth"]
    assert grid["layers"] == [8, 16, 24] and grid["sizes"] == [2, 5, 8]
    assert np.array(grid["mean"]).shape == (2, 3, 3)  # couplings x depths x sizes
    for d, e in enumerate([8, 16, 24]):
        col = res.layers("correlated_erasure").tolist().index(e)
        rows = res.values["code_dimension"][:, col] > 0
        assert grid["count_with_code"][d] == int(rows.sum())
        np.testing.assert_allclose(
            np.array(grid["mean"], dtype=float)[:, d], values[rows, col].mean(axis=0), rtol=1e-5
        )
    assert "truncated" not in out
    assert any("sample_erasure_regions" in note for note in out["notes"])
    # load_results gives the same summary
    loaded = api.load_results(str(tmp_path / "run"))
    assert loaded["probes"]["correlated_erasure"] == entry


def test_json_summary_without_code_dimension_counts_k0_as_zero():
    spec = dict(SMALL_JSON, probes=[_json_probe()], measurement_rate=0.3)
    out = api.run_ensemble(spec, 6, seed=3)
    entry = out["probes"]["correlated_erasure"]
    assert "count_with_code" not in entry and "k = 0" in entry["what"]
    assert any("correlated_erasure" in note and "k = 0" in note for note in out["notes"])


def test_json_summary_truncates_many_sizes_and_flags_it():
    spec = dict(
        SMALL_JSON,
        num_system=40,
        graph={"family": "cycle"},
        probes=[{"kind": "code_dimension"}, _json_probe(sizes=None, samples=4)],
        checkpoints="final",
    )
    out = api.run_ensemble(spec, 2, seed=3)
    entry = out["probes"]["correlated_erasure"]
    assert len(entry["sizes"]) == api.MAX_GRID_SIZES and entry["sizes"][0] == 1
    assert entry["sizes"][-1] == 39 and out["truncated"] is True
    assert entry["sampler"] == "exact independent samples (cycle)"
    assert all(len(row) == api.MAX_GRID_SIZES for row in entry["mean"])
    assert any("17 of the 39 sizes are shown" in note for note in out["notes"])
    assert not any("sample_erasure_regions" in note for note in out["notes"])  # exact samples


def test_sampler_lines():
    chain = {"method": "chain", "graphs": 1, "exact_graphs": 0, "rungs": [4, 9],
             "max_step": 0.3, "burn_in": 512, "thin": 2, "tempering": True}  # fmt: skip
    assert api._sampler_text(chain) == (
        "chain with replica exchange over 4-9 rungs, burn_in 512 sweeps, thin 2"
    )
    lone = dict(chain, rungs=[1, 1], tempering=False)
    assert api._sampler_text(lone).startswith("one swap chain per coupling, without exchanges")
    mixed = dict(chain, method="mixed", graphs=4, exact_graphs=2, geometries=["cycle", "path"])
    assert api._sampler_text(mixed) == (
        "exact independent samples (cycle and path) on 2 of 4 graphs, a chain with replica "
        "exchange over 4-9 rungs, burn_in 512 sweeps, thin 2 on the rest"
    )


def test_capabilities_describe_the_probe():
    entry = api.capabilities("probes")["probes"]["correlated_erasure"]
    assert entry["params"]["couplings"] == "required"
    assert entry["params"]["geometry"] == "interaction" and entry["params"]["sizes"] is None
    note = entry["note"]
    for word in ("coupling-major", "J -> inf", "binomial_window", "geometry", "burn_in"):
        assert word in note, word
    # every parameter is explained, and the memory per running thread is stated
    for word in ("method:", "ladder auto", "max_step", "tempering:", "thin sweeps", "Memory:"):
        assert word in note, word
    assert "sample_erasure_regions" in api.capabilities("functions")["functions"]
    # the forms build at n = 32 (the size of the small examples)
    for form in entry["forms"]:
        spec = dict(SMALL_JSON, num_system=32, probes=[{"kind": "correlated_erasure", **form}])
        assert api._build_spec(spec).probes[0].sizes
    assert "correlated_erasure" in api.quickstart()
    assert api.capabilities()  # the overview still fits


def test_summary_shows_many_couplings_within_the_size_limit():
    """Review L2: with 40 couplings beside other vector probes, the couplings and sizes shown
    shrink with the summary's budget (ends kept), nothing is cut by the generic cutter, and
    the indexing of vs_depth.mean is stated."""
    couplings = [round(0.1 * j, 1) for j in range(40)]
    spec = {
        "num_system": 24, "graph": {"family": "cycle"}, "layers": 24, "measurement_rate": 0.05,
        "reference": "purification", "checkpoints": 4,
        "probes": [
            {"kind": "code_dimension"},
            {"kind": "erasure_first_failure", "samples": 20},
            {"kind": "correlated_erasure", "couplings": couplings, "samples": 6},
        ],
    }  # fmt: skip
    out = api.run_ensemble(spec, 3, seed=1)
    assert len(json.dumps(out)) <= api.MAX_OUTPUT_CHARS and out["truncated"] is True
    assert not any("cut to fit" in note for note in out.get("notes", []))
    entry = out["probes"]["correlated_erasure"]
    shown = entry["couplings"]
    assert 2 <= len(shown) < 40 and shown[0] == 0.0 and shown[-1] == 3.9
    assert len(entry["mean"]) == len(entry["stderr"]) == len(shown)
    assert entry["sizes"][0] == 1 and entry["sizes"][-1] == 23
    assert all(len(row) == len(entry["sizes"]) for row in entry["mean"])
    grid = entry["vs_depth"]
    assert np.array(grid["mean"]).shape == (len(shown), len(grid["layers"]), len(grid["sizes"]))
    assert "vs_depth.mean[i][d][j] is couplings[i] at vs_depth.layers[d]" in entry["what"]
    assert any(f"{len(shown)} of the 40 couplings" in note for note in out["notes"])
    # the values shown are the saved ones (the chosen couplings and sizes)
    res = api._build_spec(spec).run(3, seed=1)
    c_all, s_all, values = res.correlated()
    rows = [c_all.tolist().index(j) for j in shown]
    cols = [s_all.tolist().index(m) for m in entry["sizes"]]
    np.testing.assert_allclose(
        np.array(entry["mean"], dtype=float),
        values[:, -1].mean(axis=0)[np.ix_(rows, cols)],
        rtol=1e-5,
        atol=1e-9,
    )


def test_summary_keeps_the_smallest_and_largest_coupling_whatever_the_order():
    # more couplings than the summary shows, given unsorted: evenly spaced positions would
    # have dropped the largest J; the summary shows them in increasing order of J, both ends in
    couplings = [2.5, -1.0, 0.5, 3.0, 0.0, 1.5, 2.0, -0.5, 1.0, 0.25, 0.75, 1.25, 1.75, -2.0]
    spec = dict(SMALL_JSON, probes=[
        {"kind": "code_dimension"}, _json_probe(couplings=couplings, samples=8)
    ])  # fmt: skip
    out = api.run_ensemble(spec, 3, seed=2)
    entry = out["probes"]["correlated_erasure"]
    shown = entry["couplings"]
    assert len(shown) == 12 < len(couplings)
    assert shown == sorted(shown) and shown[0] == -2.0 and shown[-1] == 3.0
    assert any(
        "increasing order of J" in note and "smallest and the largest" in note
        for note in out["notes"]
    )
    # the rows are those couplings' (the saved result has them in the given order)
    result = api._build_spec(spec).run(3, seed=2)
    _, sizes, values = result.correlated()
    rows = [couplings.index(j) for j in shown]
    cols = [sizes.tolist().index(m) for m in entry["sizes"]]
    has_code = result.values["code_dimension"][:, -1] > 0
    expected = values[has_code, -1].mean(axis=0)[np.ix_(rows, cols)]
    np.testing.assert_allclose(np.array(entry["mean"], dtype=float), expected, rtol=1e-5, atol=1e-9)


def test_summary_shows_everything_when_it_fits():
    spec = dict(SMALL_JSON, probes=[{"kind": "code_dimension"}, _json_probe()])
    out = api.run_ensemble(spec, 3, seed=2)
    entry = out["probes"]["correlated_erasure"]
    assert entry["couplings"] == [0.0, 1.0] and entry["sizes"] == [2, 5, 8]
    assert "truncated" not in out
    assert not any("are shown" in note or "vs_depth shows" in note for note in out["notes"])


def test_estimate_states_the_working_memory_of_every_running_thread(monkeypatch):
    """Review L1: the estimate gives the working memory on the threads that will run, per
    thread count too, and says when a run would be refused."""
    spec = dict(SMALL_JSON, probes=[_json_probe(samples=1000)])
    mc = api._build_spec(spec)
    per_thread = mc.correlated_bytes_per_thread()
    assert per_thread == 8 * 2 * 1000 * (8 + 1)
    out = api.estimate_ensemble(spec, 3, threads=[1, 2, 64])
    running = min(numba.get_num_threads(), 3)
    assert out["working_megabytes"] == pytest.approx(running * per_thread / 1e6, rel=1e-5)
    assert [row["working_megabytes"] for row in out["by_threads"]] == pytest.approx(
        [per_thread / 1e6, 2 * per_thread / 1e6, 3 * per_thread / 1e6], rel=1e-5
    )
    (note,) = [n for n in out["notes"] if n.startswith("correlated_erasure holds")]
    assert "is refused" in note and "this one would be" not in note
    monkeypatch.setattr(mipt, "MAX_WORKING_BYTES", per_thread)
    out = api.estimate_ensemble(spec, 3)
    (note,) = [n for n in out["notes"] if n.startswith("correlated_erasure holds")]
    if running > 1:
        assert "this one would be; run on at most 1 threads" in note
    # no such probe, no such field or note
    plain = api.estimate_ensemble(dict(SMALL_JSON, probes=[{"kind": "code_dimension"}]), 3)
    assert "working_megabytes" not in plain
    assert not any("correlated_erasure" in n for n in plain.get("notes", []))


def test_region_failure_pricing_follows_the_picture_and_the_size():
    single = [api._region_failure_us(256, m, True) for m in (16, 32, 64, 128)]
    purification = [api._region_failure_us(256, m, False) for m in (16, 32, 64, 128)]
    assert single == sorted(single) and purification == sorted(purification)
    assert all(a > b for a, b in zip(single, purification, strict=True))  # single costs more
    # about 3 times the fits to the measurements at n = 512, m' = 256
    assert api._region_failure_us(512, 256, True) == pytest.approx(3e-3 * 256**2 * 24.288 + 3)
    assert api._region_failure_us(512, 256, False) == pytest.approx(3e-3 * 256**2 * 12.0088 + 3)


def test_estimate_prices_the_chain_and_grows_with_its_work():
    def seconds(**kw):
        spec = dict(SMALL_JSON, probes=[_json_probe(**kw)])
        return api.estimate_ensemble(spec, 100)["estimated_seconds"]

    base = seconds()
    assert seconds(burn_in=2000) > 2 * base
    assert seconds(samples=200) > base
    assert seconds(couplings=[0, 1, 2, 3, 4]) > base
    mc = api._build_spec(dict(SMALL_JSON, probes=[_json_probe()]))
    assert 0 < api._probe_us(mc.probes[0], mc) < math.inf
    # with the default burn-in, the exact samplers of a ring cost less than the chain
    default = [_json_probe(burn_in=None)]
    ring = api._build_spec(dict(SMALL_JSON, graph={"family": "cycle"}, probes=default))
    chain = api._build_spec(dict(SMALL_JSON, probes=default))
    assert api._probe_us(ring.probes[0], ring) < api._probe_us(chain.probes[0], chain)
    layout = api.estimate_ensemble(dict(SMALL_JSON, probes=[_json_probe()]), 10)["layout"]
    row = next(r for r in layout["arrays"] if r["name"] == "value::correlated_erasure")
    assert row["shape"] == [10, 3, 6]  # trajectories x evaluations x (2 couplings x 3 sizes)


def test_resume_with_other_couplings_is_refused(tmp_path):
    out = tmp_path / "job"
    spec = dict(SMALL_JSON, probes=[{"kind": "code_dimension"}, _json_probe()])
    status = api.start_ensemble_job(spec, 4, output_dir=str(out), seed=5)
    assert api.job_status(status["job_id"], wait_seconds=60)["status"] == "completed"
    for other in (_json_probe(couplings=[0, 1.5]), _json_probe(geometry="cycle")):
        changed = dict(spec, probes=[{"kind": "code_dimension"}, other])
        with pytest.raises(ValueError, match="different spec.*probes"):
            api.start_ensemble_job(changed, 6, output_dir=str(out))
    # the same probe written differently (sizes as a list) continues the run
    same = dict(spec, probes=[{"kind": "code_dimension"}, _json_probe(sizes=[2, 5, 8])])
    more = api.start_ensemble_job(same, 6, output_dir=str(out))
    assert more["resumed_from"] == 4
    assert api.job_status(more["job_id"], wait_seconds=60)["status"] == "completed"
    saved = mipt.EnsembleResult.load(out / "result.npz")
    direct = api._build_spec(spec).run(6, seed=5)
    np.testing.assert_array_equal(
        saved.values["correlated_erasure"], direct.values["correlated_erasure"]
    )


def test_resume_after_a_change_of_the_default_burn_in_is_refused(tmp_path, monkeypatch):
    """Review M1: the saved run pins the burn-in it ran with, so the same spec text, which
    would now resolve "burn_in": null to another default, builds another protocol."""
    out = tmp_path / "job"
    spec = dict(SMALL_JSON, probes=[{"kind": "code_dimension"}, _json_probe(burn_in=None)])
    status = api.start_ensemble_job(spec, 3, output_dir=str(out), seed=5)
    assert api.job_status(status["job_id"], wait_seconds=60)["status"] == "completed"
    assert mipt.EnsembleResult.load(out / "result.npz").spec.probes[1].burn_in == 200
    monkeypatch.setattr(erasures, "MIN_BURN_IN", 400)
    with pytest.raises(ValueError, match="builds another protocol.*probes"):
        api.start_ensemble_job(spec, 5, output_dir=str(out))
    # with the burn-in it ran with written out, the run continues
    pinned = dict(spec, probes=[{"kind": "code_dimension"}, _json_probe(burn_in=200)])
    more = api.start_ensemble_job(pinned, 5, output_dir=str(out))
    assert more["resumed_from"] == 3
    assert api.job_status(more["job_id"], wait_seconds=60)["status"] == "completed"


@pytest.mark.parametrize(
    ("probe", "message"),
    [
        ({"couplings": "x"}, "couplings"),
        ({"couplings": [0, "a"]}, "couplings"),
        ({"couplings": [0], "sizes": "1:99"}, r"probes\[0\].*sizes must lie in \[1, 11\]"),
        ({"couplings": [0], "thin": 0}, "thin must be at least 1"),
        ({"couplings": [0], "geometry": "torus"}, "geometry must be"),
        ({"couplings": [0], "tempering": 1}, "tempering must be true or false"),
        ({"couplings": [0], "method": "exact"}, "method='exact' needs"),
        ({"sizes": [2]}, "missing required parameter"),
        ({"couplings": [0], "temperature": 2}, "unknown key"),
    ],
)
def test_json_validation_messages(probe, message):
    spec = dict(SMALL_JSON, probes=[{"kind": "correlated_erasure", **probe}])
    with pytest.raises(ValueError, match=message):
        api.run_ensemble(spec, 1, seed=1)
