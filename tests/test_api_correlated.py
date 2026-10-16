"""The correlated_erasure probe through the JSON interface (``api``, so the CLI and MCP).

Checked here: the summary (one P_{C,G,J}(m) curve per coupling at the last checkpoint,
averaged over the trajectories that hold a code when a code_dimension probe covers the
layers, else over all with a note; the sizes shown, ``truncated`` when sizes are left out;
a vs_depth grid; the sampler and the graph-side mixing note), ``load_results``,
``capabilities``, the cost estimate, a resume with other couplings refused (and one with
the same probe written differently continued), and the validation messages.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sparsegf2 import api, mipt

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
    assert any("at most 17 sizes are shown" in note for note in out["notes"])
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
    assert "correlated_erasure" in api.quickstart()
    assert api.capabilities()  # the overview still fits


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
