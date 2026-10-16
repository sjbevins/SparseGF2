"""The read-only erasure tools of the JSON service: ``sample_erasure_regions`` and ``erasure_curves``.

``sample_erasure_regions`` is the graph side of geometry-correlated erasures: independent
chains (or exact samples) of regions drawn from ``Q_J(A | m) ~ exp(-J b(A))``, with the
boundary's mean and error, the chain diagnostics, the exact answers where they exist (the
ring and path closed forms, or the enumeration of every region) and notes on mixing.
``erasure_curves`` is the analysis of a saved run: the failure probability ``Phi(e)`` of the
independent channel and of the correlated one, per coupling, and the tolerance ``e_delta``.

The CLI and MCP tests for the same functions are in ``test_cli.py`` and ``test_mcp_server.py``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sparsegf2 import analysis, api, erasures, graphs, mipt


def _nodes(text: str) -> list[int]:
    """The nodes of a region written as ``"3:9, 12, 20:30:2"`` (comma-separated runs)."""
    out: list[int] = []
    for part in text.split(","):
        pieces = [int(x) for x in part.strip().split(":")]
        if len(pieces) == 1:
            out.append(pieces[0])
        else:
            out.extend(range(*pieces))
    return out


def _without_timing(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in ("elapsed_seconds", "estimated_seconds")}


RING = {"family": "cycle"}
WS = {"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 3}


def _regions(graph=RING, size=8, couplings=(0, 1, 2), samples=200, **kw):
    kw.setdefault("num_nodes", 32)
    kw.setdefault("seed", 11)
    return api.sample_erasure_regions(graph, size, list(couplings), samples, **kw)


# ---------------------------------------------------------------------------
# sample_erasure_regions
# ---------------------------------------------------------------------------


def test_ring_means_agree_with_the_closed_form():
    out = _regions()
    assert out["method"] == "exact" and out["graph"]["geometry"] == "cycle"
    assert "ring" in out["exact_source"]
    rows = out["couplings"]
    assert [row["coupling"] for row in rows] == [0, 1, 2]
    for row in rows:
        values, p = erasures.ring_boundary_distribution(32, 8, row["coupling"])
        mean = float((values * p).sum())
        variance = float((p * (values - mean) ** 2).sum())
        assert row["exact"]["mean_boundary"] == pytest.approx(mean, rel=1e-5)
        assert row["exact"]["variance"] == pytest.approx(variance, rel=1e-5)
        assert abs(row["mean_boundary"] - mean) < 4.5 * row["stderr"]  # 4 chains x 200 samples
        assert abs(row["deviation_se"]) < 4.5
        assert row["variance"] > 0 and row["ess"] > 400 and row["r_hat"] < 1.05
        assert row["min_boundary_seen"] >= 2 and row["min_boundary_seen"] % 2 == 0
    assert out["nonincreasing_in_J"] is True
    assert out["mixing"] == []
    assert out["chains"] == 4 and out["samples"] == 200 and out["seed"] == 11
    assert json.dumps(out)  # plain JSON
    assert len(json.dumps(out)) <= api.MAX_OUTPUT_CHARS
    means = [row["mean_boundary"] for row in rows]
    assert means[0] > means[1] > means[2]  # the boundary falls with the coupling


def test_the_chain_on_a_ring_matches_the_closed_form_too():
    out = _regions(samples=2000, method="chain", seed=5)
    assert out["method"] == "chain" and out["graph"]["geometry"] == "cycle"
    assert "ring" in out["exact_source"]  # the closed form holds whatever sampled it
    for row in out["couplings"]:
        assert abs(row["deviation_se"]) < 5, row
        assert row["acceptance"] is not None and 0 < row["acceptance"] <= 1  # 1 at J = 0
    assert out["couplings"][2]["acceptance"] < 1  # the cold chain rejects some swaps
    assert out["ladder"]["rungs"] >= 3 and out["tempering"] is True


def test_relabelled_ring_and_path_use_their_closed_forms():
    n = 12
    labels = np.random.default_rng(4).permutation(n)
    ring = [[int(labels[i]), int(labels[(i + 1) % n])] for i in range(n)]
    out = api.sample_erasure_regions(
        {"edges": ring, "num_nodes": n}, 4, [0, 1.5], 100, seed=2, chains=2
    )
    assert out["graph"]["geometry"] == "cycle" and out["method"] == "exact"
    assert "ring" in out["exact_source"]
    assert all(abs(row["deviation_se"]) < 5 for row in out["couplings"])  # laid along the cycle
    path = [[i, i + 1] for i in range(n - 1)]
    out = api.sample_erasure_regions({"edges": path}, 4, [0, 1.5], 100, num_nodes=n, seed=2)
    assert out["graph"]["geometry"] == "path" and "path" in out["exact_source"]
    for row in out["couplings"]:
        values, p = erasures.path_boundary_distribution(n, 4, row["coupling"])
        assert row["exact"]["mean_boundary"] == pytest.approx(float((values * p).sum()), rel=1e-5)
        assert abs(row["deviation_se"]) < 5


def test_small_graph_is_compared_with_the_enumeration():
    n, m = 14, 5
    edges = graphs.watts_strogatz(n, 4, 0.2, seed=3)
    out = api.sample_erasure_regions(
        WS, m, [0, 0.5, 1.5], 600, num_nodes=n, seed=8, chains=4, burn_in=100
    )
    assert out["method"] == "chain" and out["graph"]["geometry"] == "other"
    assert "enumeration" in out["exact_source"] and "2002" in out["exact_source"]
    moments = erasures.boundary_moments(erasures.boundary_counts(edges, n, m), [0, 0.5, 1.5])
    for row, mean, variance in zip(
        out["couplings"], moments["mean"], moments["variance"], strict=True
    ):
        assert row["exact"]["mean_boundary"] == pytest.approx(mean, rel=1e-5)
        assert row["exact"]["variance"] == pytest.approx(variance, rel=1e-5)
        assert abs(row["deviation_se"]) < 5, row
    assert out["nonincreasing_in_J"] is True
    assert out["graph"]["num_edges"] == len(edges)


def test_a_large_graph_has_only_the_uniform_closed_form():
    n, m = 64, 16
    out = api.sample_erasure_regions(WS, m, [0, 1], 100, num_nodes=n, seed=1, burn_in=50)
    assert "uniform" in out["exact_source"]
    edges = graphs.watts_strogatz(n, 4, 0.2, seed=3)
    mean0, var0 = erasures.uniform_boundary_moments(edges, n, m)
    zero, one = out["couplings"]
    assert zero["exact"] == {"mean_boundary": pytest.approx(mean0), "variance": pytest.approx(var0)}
    assert "exact" not in one or one["exact"] is None


def test_seeds_reproduce_and_differ():
    a = _without_timing(_regions(graph=WS, num_nodes=20, size=6, samples=60, burn_in=40))
    b = _without_timing(_regions(graph=WS, num_nodes=20, size=6, samples=60, burn_in=40))
    c = _without_timing(_regions(graph=WS, num_nodes=20, size=6, samples=60, burn_in=40, seed=12))
    assert a == b and a != c
    fresh = _regions(seed=None)
    assert isinstance(fresh["seed"], int) and 0 <= fresh["seed"] < 2**53
    again = _regions(seed=fresh["seed"])
    assert _without_timing(again) == _without_timing(fresh)


def test_chain_seeds_are_derived_from_the_seed():
    out = _regions(graph=WS, num_nodes=20, size=6, samples=40, burn_in=30, chains=2, seed=9)
    first = erasures.sample_regions(
        graphs.watts_strogatz(20, 4, 0.2, seed=3), 20, 6, [0, 1, 2], 40, burn_in=30, seed=[9, 0]
    )
    # the region examples come from the largest coupling's records of the first chains
    top = max(out["couplings"], key=lambda row: row["coupling"])
    assert top["coupling"] == 2
    example = out["examples"]["regions"][0]
    assert example["boundary"] == erasures.boundary(
        graphs.watts_strogatz(20, 4, 0.2, seed=3), 20, _nodes(example["nodes"])
    )
    assert first.regions.shape == (3, 40, 6)  # [seed, chain] is the documented stream


def test_examples_are_regions_of_the_largest_coupling():
    out = _regions(graph=WS, num_nodes=24, size=6, samples=60, burn_in=60, couplings=(2, 0, 1))
    assert out["examples"]["coupling"] == 2
    edges = graphs.watts_strogatz(24, 4, 0.2, seed=3)
    assert len(out["examples"]["regions"]) == 3
    for region in out["examples"]["regions"]:
        nodes = _nodes(region["nodes"])
        assert len(nodes) == 6 and sorted(set(nodes)) == nodes
        assert region["boundary"] == erasures.boundary(edges, 24, nodes)
    # the rows follow the order the couplings were given in
    assert [row["coupling"] for row in out["couplings"]] == [2, 0, 1]


def test_nonincreasing_check():
    check = api._nonincreasing_in_coupling
    assert check([0.0, 1.0, 2.0], [5.0, 4.0, 3.0], [0.1, 0.1, 0.1]) is True
    assert check([2.0, 0.0, 1.0], [3.0, 5.0, 4.0], [0.1, 0.1, 0.1]) is True  # any order of rows
    assert check([0.0, 1.0], [4.0, 4.2], [0.1, 0.1]) is True  # a rise within 3 standard errors
    assert check([0.0, 1.0], [4.0, 5.0], [0.1, 0.1]) is False
    assert check([1.0], [4.0], [0.1]) is None  # nothing to compare


def test_poor_mixing_is_reported_with_what_to_change():
    out = api.sample_erasure_regions(
        WS, 16, [0, 4], 16, num_nodes=64, seed=1, burn_in=0, tempering=False, chains=4
    )
    text = " ".join(out["mixing"])
    assert "R-hat" in text and "effective sample size" in text
    assert "burn_in" in text and "samples" in text
    rhat = next(note for note in out["mixing"] if note.startswith("R-hat"))
    assert "J = 4 (" in rhat and "4 chains" in rhat  # the coupling, its value and what was followed
    assert len(out["mixing"]) <= 4  # one note per kind of problem, however many couplings


def test_the_mixing_thresholds():
    def notes(method="chain", *, r_hat=1.005, ess=500.0, deviation=0.0, exchange=0.5):
        rows = [{"coupling": 1.0, "ess": ess, "deviation_se": deviation}]
        details = [{"r_hat": r_hat, "r_hat_node": 7, "slowest": None}]
        return api._regions_mixing_notes(
            rows, details, method=method, chains=4, samples=200, burn_in=100,
            pair_acceptance=np.array([exchange]), ladder=np.array([0.0, 1.0]),
        )  # fmt: skip

    assert notes() == []
    (note,) = notes(r_hat=1.02)  # R-hat above 1.01, with the node it was seen on
    assert note.startswith("R-hat over 4 chains above 1.01 at J = 1 (1.020, node 7)")
    assert notes(r_hat=1.0099) == [] and notes(r_hat=1.02, method="exact") == []  # no chain
    (note,) = notes(ess=399.0)
    assert note.startswith("effective sample size below 400 at J = 1 (399, boundary)")
    assert notes(ess=401.0) == [] and len(notes(ess=399.0, method="exact")) == 1  # all samplers
    (note,) = notes(deviation=-5.5)
    assert "more than 5 standard errors" in note and "(-5.5)" in note and "burn_in" in note
    assert notes(deviation=4.9) == []
    assert "report it" in notes(deviation=6.0, method="exact")[0]  # an exact sampler is not biased
    (note,) = notes(exchange=0.049)
    assert note.startswith("replica exchange accepted in under 5%") and "J = 0 and 1 (4.9%)" in note
    assert notes(exchange=0.051) == [] and notes(exchange=0.001, method="exact") == []
    assert notes(exchange=float("nan")) == []  # no exchanges happened (one rung)
    assert len(notes(r_hat=2.0, ess=10.0, deviation=9.0, exchange=0.0)) == 4  # one per kind


def test_chains_stuck_apart_have_an_infinite_r_hat_that_is_reported():
    out = api.sample_erasure_regions(
        WS, 16, [0, 50], 40, num_nodes=64, seed=2, burn_in=200, tempering=False
    )
    cold = out["couplings"][1]
    assert cold["r_hat"] == api.R_HAT_INFINITE and cold["acceptance"] < 0.01
    note = next(n for n in out["mixing"] if n.startswith("R-hat"))
    assert "J = 50 (infinite, boundary)" in note
    assert any("acceptance" in w or "swaps" in w for w in out["warnings"])  # the sampler's own


def test_replica_exchange_that_barely_happens_is_reported():
    out = api.sample_erasure_regions(
        WS, 16, [0, 6], 60, num_nodes=64, seed=2, burn_in=20, ladder="given", chains=2
    )
    texts = [note for note in out["mixing"] if note.startswith("replica exchange")]
    assert texts and "add couplings" in texts[0] and "max_step" in texts[0]
    assert out["couplings"][1]["exchange_acceptance"] < 0.05
    assert out["warnings"]  # the sampler's own warning is passed on


def test_sampler_warnings_are_forwarded_once():
    out = api.sample_erasure_regions(
        WS, 16, [0, 6], 60, num_nodes=64, seed=2, burn_in=20, ladder="given", chains=3
    )
    assert len(out["warnings"]) == len(set(out["warnings"])) > 0
    assert all(w.endswith("chains]") for w in out["warnings"])
    assert any(" of 3 chains]" in w for w in out["warnings"])


def test_a_small_sample_count_is_flagged():
    out = _regions(samples=10, chains=2)
    assert any("effective sample size" in note and "samples" in note for note in out["mixing"])


def test_cost_guard_refuses_what_would_take_minutes():
    big = {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": 1}
    with pytest.raises(ValueError, match=r"estimated at about \d+ s.*max_seconds") as info:
        api.sample_erasure_regions(big, 500, [0, 1, 2], 2000, num_nodes=2000, chains=8, seed=1)
    message = str(info.value)
    for advice in ("burn_in", "chains", "samples"):
        assert advice in message
    with pytest.raises(ValueError, match="max_seconds"):
        api.sample_erasure_regions(WS, 5, [0, 1], 200, num_nodes=20, seed=1, max_seconds=0)
    # a generous limit lets it through
    out = api.sample_erasure_regions(WS, 5, [0, 1], 40, num_nodes=20, seed=1, max_seconds=600)
    assert out["estimated_seconds"] > 0 and out["elapsed_seconds"] >= 0


def test_estimate_grows_with_the_work():
    def estimate(**kw):
        args = dict(n=256, m=64, num_edges=512, exact=False, rungs=20, couplings=4, burn_in=2048,
                    samples=200, thin=1, chains=4, enumerated=0)  # fmt: skip
        return api._regions_seconds(**{**args, **kw})

    base = estimate()
    assert estimate(chains=8) > 1.9 * base and estimate(rungs=40) > 1.9 * base
    assert estimate(burn_in=4096) > 1.7 * base and estimate(thin=10) > base
    assert estimate(num_edges=2048) > base  # denser graphs cost more per proposal
    assert estimate(enumerated=10**8) > base + 1
    # the measured run (n = 256, k = 4, 20 rungs, 4 chains: about 1 s) is priced above itself
    assert 1.5 < base < 60
    # exact samples cost far less than the chain for the same call
    assert estimate(exact=True) < base / 10


def test_the_estimate_prices_the_ladder_the_call_runs():
    out = api.sample_erasure_regions(WS, 16, [0, 0.5, 1], 100, num_nodes=64, seed=1, burn_in=300)
    edges = graphs.watts_strogatz(64, 4, 0.2, seed=3)  # WS: the same family and seed as above
    expected = api._regions_seconds(
        n=64, m=16, num_edges=len(edges), exact=False, rungs=out["ladder"]["rungs"],
        couplings=3, burn_in=300, samples=100, thin=1, chains=4, enumerated=0,
    )  # fmt: skip
    assert out["estimated_seconds"] == pytest.approx(expected, rel=1e-5)
    assert out["elapsed_seconds"] > 0


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"size": 0}, "size must be between 1 and 31"),
        ({"size": 32}, "size must be between 1 and 31"),
        ({"size": -3}, "size must be between 1 and 31"),
        ({"size": True}, "size must be an integer"),
        ({"size": 2.5}, "size must be an integer"),
        ({"couplings": []}, "at least one coupling"),
        ({"couplings": [0, 0]}, "distinct"),
        ({"couplings": [float("inf")]}, "finite"),
        ({"couplings": ["a"]}, "couplings"),
        ({"couplings": [True]}, "couplings"),
        ({"couplings": list(range(40))}, "at most 16 couplings"),
        ({"samples": 3}, "samples must be at least 8"),
        ({"samples": True}, "samples must be an integer"),
        ({"chains": 0}, "chains must be at least 1"),
        ({"chains": 99}, "chains must be at most"),
        ({"thin": 0}, "thin must be at least 1"),
        ({"burn_in": -1}, "burn_in must be at least 0"),
        ({"tempering": 1}, "tempering must be true or false"),
        ({"method": "magic"}, "method must be"),
        ({"method": ["exact"]}, "method must be a string"),
        ({"max_step": 0}, "max_step"),
        ({"ladder": "dense"}, "ladder"),
        ({"seed": -1}, "seed must be at least 0"),
        ({"num_nodes": None}, "give num_nodes"),
        ({"graph": 5}, "graph must be an object"),
        ({"graph": {"k": 4}}, "graph must be an object"),
        ({"graph": {"family": "moebius"}}, "unknown graph family"),
        ({"graph": {"family": ["cycle"]}}, "graph.family must be a string"),
        ({"graph": {**WS, "realizations": 5}}, "realizations"),
        ({"graph": {"edges": [[0, 1]], "weights": 3}, "num_nodes": 4}, "unknown key"),
        ({"graph": {"edges": [[0, 99]]}, "num_nodes": 4}, "out of range"),
        ({"graph": {"edges": [[0, 0]]}, "num_nodes": 4}, "self-loops"),
        ({"graph": {"edges": "01"}, "num_nodes": 4}, "edges must be a list"),
        ({"graph": {"edges": [[0, 1]], "num_nodes": 5}, "num_nodes": 4}, "num_nodes"),
        ({"graph": {"family": "file"}}, "path"),
        ({"graph": {"family": "file", "path": "no/such/file.npz"}}, "no such file"),
        ({"method": "exact", "graph": WS, "num_nodes": 20, "size": 5}, "method='exact' needs"),
    ],
)
def test_validation_messages(args, message):
    base = {"graph": RING, "size": 8, "couplings": [0, 1], "samples": 40, "num_nodes": 32,
            "seed": 1}  # fmt: skip
    arguments = {**base, **args}
    graph, size, couplings = (arguments.pop(k) for k in ("graph", "size", "couplings"))
    with pytest.raises(ValueError, match=message):
        api.sample_erasure_regions(graph, size, couplings, **arguments)


def test_a_graph_file_with_several_graphs_uses_the_first(tmp_path):
    n = 10
    bank = [graphs.watts_strogatz(n, 4, 0.3, seed=s) for s in range(3)]
    graphs.save(tmp_path / "bank.npz", bank, n)
    graph = {"family": "file", "path": str(tmp_path / "bank.npz")}
    out = api.sample_erasure_regions(graph, 3, [0, 1], 60, seed=1, burn_in=40)  # n from the file
    assert out["graph"]["num_nodes"] == n and out["graph"]["num_edges"] == len(bank[0])
    assert any("first of 3 graphs" in note for note in out["notes"])
    with pytest.raises(ValueError, match="num_nodes"):
        api.sample_erasure_regions(graph, 3, [0], 60, num_nodes=n + 1, seed=1)
    graphs.save(tmp_path / "one.npz", bank[:1], n)
    one = api.sample_erasure_regions(
        {"family": "file", "path": str(tmp_path / "one.npz")}, 3, [0, 1], 60, seed=1, burn_in=40
    )
    assert not any("first of" in note for note in one["notes"])


def test_the_result_names_the_settings_it_ran_with():
    out = api.sample_erasure_regions(
        WS, 5, [0, 1], 40, num_nodes=20, seed=4, burn_in=25, thin=2, chains=3
    )
    assert (out["burn_in"], out["thin"], out["chains"], out["samples"]) == (25, 2, 3, 40)
    default = api.sample_erasure_regions(WS, 5, [0, 1], 40, num_nodes=20, seed=4)
    assert default["burn_in"] == erasures.default_burn_in(20)
    assert any("another seed" in note for note in default["notes"])  # no note is no proof
    assert any("seed [4, i]" in note for note in default["notes"])  # chain i's stream


def test_many_couplings_still_fit_the_size_limit():
    out = api.sample_erasure_regions(
        {"family": "cycle"}, 40, [0.1 * i for i in range(16)], 200, num_nodes=128, seed=2
    )
    assert len(json.dumps(out)) <= api.MAX_OUTPUT_CHARS
    assert len(out["couplings"]) == 16


def test_long_example_regions_are_cut_and_flagged():
    out = api.sample_erasure_regions(
        {"family": "cycle"}, 300, [0], 20, num_nodes=600, seed=1, chains=1
    )
    regions = out["examples"]["regions"]
    assert all(len(r["nodes"]) <= 240 for r in regions)
    assert any(r["nodes"].endswith("...") for r in regions)
    assert out["truncated"] is True and any("cut to 240" in note for note in out["notes"])
    short = api.sample_erasure_regions({"family": "cycle"}, 4, [0], 20, num_nodes=60, seed=1)
    assert "truncated" not in short


def test_the_worst_case_still_fits_without_cutting():
    # sixteen couplings, every row with an exact value, every kind of note, the sampler's warnings
    out = api.sample_erasure_regions(
        WS, 6, [0.5 * i - 2 for i in range(16)], 16, num_nodes=20, seed=3, burn_in=0,
        tempering=False, chains=4,
    )  # fmt: skip
    assert "enumeration" in out["exact_source"] and out["mixing"]
    assert len(json.dumps(out)) <= api.MAX_OUTPUT_CHARS and "truncated" not in out


# ---------------------------------------------------------------------------
# erasure_curves
# ---------------------------------------------------------------------------

N = 16
CURVE_SPEC = {
    "num_system": N,
    "graph": {"family": "cycle"},
    "layers": 24,
    "measurement_rate": 0.1,
    "schedule": "matching",
    "reference": "purification",
    "checkpoints": [12, 24],
    "probes": [
        {"kind": "code_dimension"},
        {"kind": "erasure_first_failure", "samples": 300},
        {"kind": "uncorrectable_region_counts", "max_size": 2},
        {"kind": "correlated_erasure", "couplings": [0, 2], "samples": 300},
    ],
}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("curves") / "run"
    api.run_ensemble(CURVE_SPEC, 60, seed=5, output_dir=str(out))
    return out


@pytest.fixture(scope="module")
def result(run_dir):
    return mipt.EnsembleResult.load(run_dir / "result.npz")


def test_curves_uniform_matches_the_analysis_functions(run_dir, result):
    out = api.erasure_curves(str(run_dir), e=[0.05, 0.1, 0.2], delta=0.05)
    assert out["e"] == [0.05, 0.1, 0.2] and out["delta"] == 0.05 and out["num_system"] == N
    entry = out["uniform"]["erasure_first_failure"]
    assert entry["layer"] == 24 and entry["count"] == 60
    first = result.values["erasure_first_failure"][:, 1]
    counts = result.values["uncorrectable_region_counts[2]"][:, 1]
    code = first.min(axis=1) <= N
    assert entry["count_with_code"] == int(code.sum()) > 10
    assert entry["q_plus"] == pytest.approx(code.mean())
    assert entry["exact_counts"] == {"probe": "uncorrectable_region_counts[2]", "max_size": 2}
    phi = analysis.erasure_failure_probability(
        first[code], N, np.array([0.05, 0.1, 0.2]), exact_counts=counts[code]
    )
    mean, error = analysis.mean_and_error(phi, axis=0)
    np.testing.assert_allclose(entry["phi"]["mean"], mean, rtol=1e-5)
    np.testing.assert_allclose(entry["phi"]["stderr"], error, rtol=1e-5)
    assert 0 < mean[0] < mean[1] < mean[2] < 1  # a non-trivial, increasing channel
    tol = [
        analysis.erasure_tolerance(first[i : i + 1], N, 0.05, exact_counts=counts[i : i + 1])
        for i in np.flatnonzero(code)
    ]
    delta = entry["e_delta"]
    assert delta["per_code_mean"] == pytest.approx(np.mean(tol), rel=1e-5)
    assert delta["per_code_median"] == pytest.approx(np.median(tol), rel=1e-5)
    assert delta["per_code_stderr"] == pytest.approx(
        np.std(tol, ddof=1) / math.sqrt(len(tol)), rel=1e-4
    )
    assert delta["of_mean_phi"] == pytest.approx(
        analysis.erasure_tolerance(first[code], N, 0.05, exact_counts=counts[code]), rel=1e-5
    )
    assert "independent erasure" in entry["what"] and "k > 0" in entry["what"]


def test_curves_correlated_matches_the_analysis_functions(run_dir, result):
    es = [0.05, 0.1, 0.2]
    out = api.erasure_curves(str(run_dir), e=es, delta=0.05)
    entry = out["correlated"]["correlated_erasure"]
    couplings, sizes, values = result.correlated()
    assert entry["couplings"] == [0.0, 2.0] and entry["layer"] == 24
    code = result.values["code_dimension"][:, 1] > 0
    assert entry["count_with_code"] == int(code.sum())
    curve, interpolated = analysis.correlated_failure_curve(
        values[code, 1], sizes, N, holds_code=np.ones(int(code.sum()))
    )
    phi = analysis.failure_probability_from_curve(curve, N, es)  # (codes, c, e)
    mean, error = analysis.mean_and_error(phi, axis=0)
    np.testing.assert_allclose(entry["phi"]["mean"], mean, rtol=1e-5)
    np.testing.assert_allclose(entry["phi"]["stderr"], error, rtol=1e-5)
    per_code = analysis.tolerance_from_curve(curve, N, 0.05)  # (codes, c)
    np.testing.assert_allclose(entry["e_delta"]["per_code_mean"], per_code.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(
        entry["e_delta"]["per_code_median"], np.median(per_code, axis=0), rtol=1e-5
    )
    of_mean = analysis.tolerance_from_curve(curve.mean(axis=0), N, 0.05)
    np.testing.assert_allclose(entry["e_delta"]["of_mean_phi"], of_mean, rtol=1e-5)
    assert entry["num_interpolated"] == int(interpolated.sum()) == 0  # every size was sampled
    assert entry["binomial_window"] == list(analysis.binomial_window(N, es, 1e-6))
    assert entry["complete"] is True and entry["interpolated_mass"] == [0.0, 0.0, 0.0]


def test_curves_uniform_and_j0_correlated_agree_within_errors(run_dir):
    out = api.erasure_curves(str(run_dir), e=[0.03, 0.06, 0.1, 0.15, 0.25])
    uniform = out["uniform"]["erasure_first_failure"]["phi"]
    correlated = out["correlated"]["correlated_erasure"]["phi"]
    j0 = 0  # couplings [0, 2]
    for i in range(5):
        gap = abs(uniform["mean"][i] - correlated["mean"][j0][i])
        sigma = math.hypot(uniform["stderr"][i], correlated["stderr"][j0][i])
        assert gap < 4 * sigma + 0.02, (i, gap, sigma)
    # and the correlated channel at J = 2 fails more (compact erasures are worse here)
    assert correlated["mean"][1][2] != correlated["mean"][0][2]


def test_curves_at_an_earlier_layer(run_dir, result):
    out = api.erasure_curves(str(run_dir), e=[0.1], layer=12)
    first = result.values["erasure_first_failure"][:, 0]
    assert out["layer"] == 12
    assert out["uniform"]["erasure_first_failure"]["layer"] == 12
    assert out["correlated"]["correlated_erasure"]["layer"] == 12
    code = first.min(axis=1) <= N
    assert out["uniform"]["erasure_first_failure"]["count_with_code"] == int(code.sum())
    later = api.erasure_curves(str(run_dir), e=[0.1])
    assert later["uniform"]["erasure_first_failure"]["layer"] == 24
    with pytest.raises(ValueError, match=r"layer 7.*evaluated at \[12, 24\]"):
        api.erasure_curves(str(run_dir), layer=7)


def test_curves_read_an_npz_file_and_default_to_a_grid_for_n(run_dir):
    a = api.erasure_curves(str(run_dir))
    b = api.erasure_curves(str(run_dir / "result.npz"))
    assert {k: v for k, v in a.items() if k != "path"} == {
        k: v for k, v in b.items() if k != "path"
    }
    grid = a["e"]
    assert 4 <= len(grid) <= 12 and grid == sorted(grid)
    assert grid[0] == pytest.approx(1 / N, rel=0.1) and grid[-1] == 0.5
    assert all(0 < x <= 0.5 for x in grid)
    assert a["delta"] == 0.01
    assert len(json.dumps(a)) <= api.MAX_OUTPUT_CHARS
    assert api.erasure_curves(str(run_dir), e=0.2)["e"] == [0.2]  # one number is fine


def test_curves_without_erasure_probes_say_so(tmp_path):
    spec = dict(CURVE_SPEC, probes=[{"kind": "code_dimension"}])
    api.run_ensemble(spec, 5, seed=1, output_dir=str(tmp_path / "plain"))
    with pytest.raises(
        ValueError, match=r"no erasure_first_failure or correlated_erasure probe.*code_dimension"
    ):
        api.erasure_curves(str(tmp_path / "plain"))


def test_curves_name_missing_paths(tmp_path):
    with pytest.raises(ValueError, match="no such file or directory"):
        api.erasure_curves(str(tmp_path / "nowhere"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no saved results"):
        api.erasure_curves(str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="path must be a string"):
        api.erasure_curves(5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"e": [0.1, 1.5]}, r"e must be .* in \[0, 1\]"),
        ({"e": []}, "at least one"),
        ({"e": ["a"]}, "e must be"),
        ({"e": [True]}, "e must be"),
        ({"e": [0.01 * i for i in range(1, 40)]}, "at most 16 values"),
        ({"delta": 2}, r"delta must be in \[0, 1\]"),
        ({"delta": "x"}, "delta must be"),
        ({"layer": 2.5}, "layer must be an integer"),
        ({"layer": True}, "layer must be an integer"),
        ({"layer": -1}, "layer must be at least 0"),
    ],
)
def test_curves_validation_messages(run_dir, args, message):
    with pytest.raises(ValueError, match=message):
        api.erasure_curves(str(run_dir), **args)


def test_curves_stay_within_the_size_cap_with_many_couplings(tmp_path):
    couplings = [round(0.25 * i, 2) for i in range(30)]
    spec = dict(CURVE_SPEC, probes=[
        {"kind": "code_dimension"},
        {"kind": "erasure_first_failure", "samples": 50},
        {"kind": "correlated_erasure", "couplings": couplings, "samples": 20},
    ])  # fmt: skip
    api.run_ensemble(spec, 12, seed=2, output_dir=str(tmp_path / "many"))
    out = api.erasure_curves(str(tmp_path / "many"), e=[0.02 * i for i in range(1, 17)])
    assert len(json.dumps(out)) <= api.MAX_OUTPUT_CHARS
    entry = out["correlated"]["correlated_erasure"]
    assert out["truncated"] is True and len(entry["couplings"]) < 30
    assert any("couplings" in note and "result.npz" in note for note in out["notes"])
    # the couplings shown keep both ends
    assert entry["couplings"][0] == 0 and entry["couplings"][-1] == couplings[-1]
    assert len(entry["phi"]["mean"]) == len(entry["couplings"])


def test_curves_report_what_was_interpolated(tmp_path):
    spec = dict(CURVE_SPEC, probes=[
        {"kind": "code_dimension"},
        {"kind": "correlated_erasure", "couplings": [0, 1], "sizes": "2:15:3", "samples": 100},
    ])  # fmt: skip
    api.run_ensemble(spec, 20, seed=2, output_dir=str(tmp_path / "sparse"))
    out = api.erasure_curves(str(tmp_path / "sparse"), e=[0.02, 0.2, 0.5])
    entry = out["correlated"]["correlated_erasure"]
    result = mipt.EnsembleResult.load(tmp_path / "sparse" / "result.npz")
    _, sizes, values = result.correlated()
    assert sizes.tolist() == [2, 5, 8, 11, 14]
    assert _nodes(entry["sampled_sizes"]) == [2, 5, 8, 11, 14]
    _, interpolated = analysis.correlated_failure_curve(values[:, 0, :, :], sizes, N)
    missing = [int(m) for m in np.flatnonzero(interpolated)]
    assert entry["num_interpolated"] == len(missing) > 0
    assert _nodes(entry["interpolated_sizes"]) == missing
    # the binomial mass the interpolated sizes carry at each e
    expected = []
    for x in (0.02, 0.2, 0.5):
        pmf = [math.comb(N, m) * x**m * (1 - x) ** (N - m) for m in range(N + 1)]
        expected.append(sum(pmf[m] for m in missing))
    np.testing.assert_allclose(entry["interpolated_mass"], expected, rtol=1e-4)
    assert entry["complete"] is False
    assert any("interpolat" in note for note in out["notes"])


def test_curves_of_a_swap_chain_point_to_the_mixing_check(tmp_path):
    spec = dict(CURVE_SPEC, graph={"family": "watts_strogatz", "k": 4, "beta": 0.2, "seed": 2},
                schedule="random_edges", probes=[
        {"kind": "code_dimension"},
        {"kind": "correlated_erasure", "couplings": [0, 1], "sizes": "2:11:3", "samples": 20,
         "burn_in": 20},
    ])  # fmt: skip
    api.run_ensemble(spec, 10, seed=2, output_dir=str(tmp_path / "chain"))
    out = api.erasure_curves(str(tmp_path / "chain"), e=[0.1, 0.3])
    entry = out["correlated"]["correlated_erasure"]
    assert entry["sampler"].startswith("chain with replica exchange")
    assert any("sample_erasure_regions" in note for note in out["notes"])
    exact = api.erasure_curves(str(tmp_path / "chain" / "result.npz"), e=[0.1])["correlated"]
    assert exact["correlated_erasure"]["sampler"] == entry["sampler"]


def test_curves_scale_to_many_trajectories(tmp_path):
    # many codes at once: the work is chunked, and the per-code tolerance is capped
    spec = dict(CURVE_SPEC, probes=[
        {"kind": "code_dimension"},
        {"kind": "erasure_first_failure", "samples": 20},
        {"kind": "correlated_erasure", "couplings": [0, 1], "sizes": [3, 8], "samples": 4},
    ], checkpoints="final")  # fmt: skip
    api.run_ensemble(spec, 5000, seed=3, output_dir=str(tmp_path / "many"))
    out = api.erasure_curves(str(tmp_path / "many"), e=[0.1], delta=0.1)
    uniform = out["uniform"]["erasure_first_failure"]
    assert uniform["count"] == 5000 and uniform["e_delta"]["codes"] == api.MAX_TOLERANCE_CODES
    assert any("of the" in note and "evenly spaced" in note for note in out["notes"])
    assert out["correlated"]["correlated_erasure"]["e_delta"]["codes"] == api.MAX_TOLERANCE_CODES
    result = mipt.EnsembleResult.load(tmp_path / "many" / "result.npz")
    first = result.values["erasure_first_failure"][:, -1]
    code = first.min(axis=1) <= N
    phi = analysis.erasure_failure_probability(first[code], N, 0.1)
    assert uniform["phi"]["mean"][0] == pytest.approx(
        phi.mean(), rel=1e-5
    )  # all codes, not the subset


def test_curves_correlated_without_a_code_probe_count_every_trajectory(tmp_path):
    spec = dict(CURVE_SPEC, probes=[
        {"kind": "correlated_erasure", "couplings": [0, 1], "samples": 50}
    ])  # fmt: skip
    api.run_ensemble(spec, 20, seed=2, output_dir=str(tmp_path / "bare"))
    out = api.erasure_curves(str(tmp_path / "bare"), e=[0.1])
    entry = out["correlated"]["correlated_erasure"]
    assert "count_with_code" not in entry and entry["q_plus"] is None
    assert any("k = 0" in note and "code_dimension" in note for note in out["notes"])


def test_curves_when_no_trajectory_holds_a_code(tmp_path):
    spec = dict(CURVE_SPEC, measurement_rate=0.9, layers=40, checkpoints="final", probes=[
        {"kind": "code_dimension"},
        {"kind": "erasure_first_failure", "samples": 20},
        {"kind": "correlated_erasure", "couplings": [0], "samples": 20},
    ])  # fmt: skip
    api.run_ensemble(spec, 6, seed=2, output_dir=str(tmp_path / "pure"))
    out = api.erasure_curves(str(tmp_path / "pure"), e=[0.1])
    for entry in (out["uniform"]["erasure_first_failure"], out["correlated"]["correlated_erasure"]):
        assert entry["count_with_code"] == 0 and entry["q_plus"] == 0
        assert entry["phi"] is None and entry["e_delta"] is None
    assert any("no trajectory holds a code" in note for note in out["notes"])


def test_curves_uniform_without_counts_is_the_plain_estimate(tmp_path):
    spec = dict(CURVE_SPEC, probes=[
        {"kind": "erasure_first_failure", "samples": 100}
    ])  # fmt: skip
    api.run_ensemble(spec, 20, seed=2, output_dir=str(tmp_path / "plain"))
    out = api.erasure_curves(str(tmp_path / "plain"), e=[0.1, 0.2])
    entry = out["uniform"]["erasure_first_failure"]
    assert entry["exact_counts"] is None and "correlated" not in out
    result = mipt.EnsembleResult.load(tmp_path / "plain" / "result.npz")
    first = result.values["erasure_first_failure"][:, -1]
    code = first.min(axis=1) <= N
    phi = analysis.erasure_failure_probability(first[code], N, np.array([0.1, 0.2]))
    np.testing.assert_allclose(entry["phi"]["mean"], phi.mean(axis=0), rtol=1e-5)


def test_curves_counts_probe_alone_is_not_enough(tmp_path):
    spec = dict(CURVE_SPEC, probes=[
        {"kind": "code_dimension"}, {"kind": "uncorrectable_region_counts", "max_size": 2}
    ])  # fmt: skip
    api.run_ensemble(spec, 5, seed=1, output_dir=str(tmp_path / "counts"))
    with pytest.raises(ValueError, match="erasure_first_failure or correlated_erasure"):
        api.erasure_curves(str(tmp_path / "counts"))


def test_curves_write_nothing(run_dir):
    before = sorted(p.name for p in run_dir.rglob("*"))
    api.erasure_curves(str(run_dir))
    assert sorted(p.name for p in run_dir.rglob("*")) == before


def test_capabilities_and_quickstart_list_the_tools():
    caps = api.capabilities()
    for name in ("sample_erasure_regions", "erasure_curves"):
        assert name in caps["functions"]
    text = api.quickstart()
    assert "sample_erasure_regions" in text and "erasure_curves" in text
    assert "correlated_erasure" in text
    assert len(json.dumps(caps)) <= api.MAX_OUTPUT_CHARS
