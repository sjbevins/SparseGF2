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

import inspect
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
        assert row["variance"] > 0 and row["min_boundary_seen"] % 2 == 0 >= 0
        assert row["min_boundary_seen"] >= 2
        # independent draws: tau is 1, the effective sample size is the number of draws, the
        # error is sqrt(variance / draws), and chains, R-hat and the acceptances do not apply
        assert row["tau_int"] == 1 and row["ess"] == row["ess_boundary"] == 800
        assert row["stderr"] == pytest.approx(math.sqrt(row["variance"] / 800), rel=1e-4)
        for field in ("r_hat", "r_hat_limit", "r_hat_at", "slowest", "acceptance"):
            assert row[field] is None, field
        assert row["exchange_acceptance"] is None
    assert out["nonincreasing_in_J"] is True
    assert out["mixing"] == []
    assert out["chains"] == 4 and out["samples"] == 200 and out["seed"] == 11
    assert json.dumps(out)  # plain JSON
    assert len(json.dumps(out)) <= api.MAX_OUTPUT_CHARS
    means = [row["mean_boundary"] for row in rows]
    assert means[0] > means[1] > means[2]  # the boundary falls with the coupling


def test_the_largest_allowed_coupling_still_works_on_the_ring():
    # |J| = 1e6 is the bound: the ring gives its intervals (boundary 2) there, at either sign
    # of the regions' preference, and a coupling above it is refused with the exact routes
    out = _regions(couplings=(0, erasures.MAX_ABS_COUPLING), size=8)
    top = out["couplings"][1]
    assert top["coupling"] == 1e6 and top["mean_boundary"] == 2 and top["min_boundary_seen"] == 2
    assert out["method"] == "exact" and out["nonincreasing_in_J"] is True
    with pytest.raises(ValueError, match=r"\|J\| <= 1,000,000.*interval_failure_by_size"):
        _regions(couplings=(0, 1e6 * 1.0000001))


def test_the_chain_on_a_ring_matches_the_closed_form_too():
    out = _regions(samples=2000, method="chain", seed=5)
    assert out["method"] == "chain" and out["graph"]["geometry"] == "cycle"
    assert "ring" in out["exact_source"]  # the closed form holds whatever sampled it
    for row in out["couplings"]:
        assert abs(row["deviation_se"]) < 5, row
        assert row["acceptance"] is not None and 0 < row["acceptance"] <= 1  # 1 at J = 0
    assert out["couplings"][2]["acceptance"] < 1  # the cold chain rejects some swaps
    for row in out["couplings"]:
        # stderr = sqrt(variance / ess_boundary); ess and tau_int are the slowest of the series
        assert row["stderr"] == pytest.approx(
            math.sqrt(row["variance"] / row["ess_boundary"]), rel=1e-3
        )
        assert row["ess"] <= row["ess_boundary"] * (1 + 1e-9) and row["tau_int"] >= 1
        assert row["r_hat_limit"] >= api.RHAT_WARNING and row["r_hat_at"] in {"boundary"} | {
            f"node {v}" for v in range(32)
        }
        assert row["r_hat"] <= row["r_hat_limit"]  # a correct sampler: nothing flagged here
    assert out["mixing"] == []
    assert out["ladder"]["rungs"] >= 3 and out["tempering"] is True


def test_relabeled_ring_and_path_use_their_closed_forms():
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


def test_effective_sizes_are_capped_at_the_draws_and_the_error_follows():
    # the integrated autocorrelation time of 200 draws can be estimated below 1 from noise
    # (an effective size above the draws); the tool caps ess and ess_boundary at chains x
    # samples, keeps tau_int as estimated, and derives stderr from the capped size
    n = 24
    ws = graphs.watts_strogatz(n, 4, 0.2, seed=3)
    out = api.sample_erasure_regions(
        {"edges": ws.tolist(), "num_nodes": n}, 6, [0, 1], 200, seed=4, thin=5
    )
    draws = out["chains"] * out["samples"]
    assert out["method"] == "chain"
    rows = out["couplings"]
    assert all(r["ess"] <= draws and r["ess_boundary"] <= draws for r in rows)
    assert any(r["ess_boundary"] == draws for r in rows)  # the cap acted
    for r in rows:
        assert r["stderr"] == pytest.approx(math.sqrt(r["variance"] / r["ess_boundary"]), rel=1e-3)
    assert any("capped at chains x samples" in note for note in out["notes"])


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


def test_proposition_two_tests_consecutive_couplings_with_a_correction():
    check = api._proposition_two
    assert check([1.0], [4.0], [0.1]) is None  # nothing to compare
    ok = check([0.0, 1.0, 2.0], [5.0, 4.0, 3.0], [0.1, 0.1, 0.1])
    assert ok["holds"] and ok["failed"] == [] and ok["steps"] == 2
    assert ok["worst"]["J"] == [0.0, 1.0] or ok["worst"]["J"] == [1.0, 2.0]
    assert ok["worst"]["z"] < 0  # every step falls
    # rows in any order are compared along increasing couplings
    assert check([2.0, 0.0, 1.0], [3.0, 5.0, 4.0], [0.1, 0.1, 0.1])["holds"]
    # one step: the normal quantile at 1% (2.33): a rise of 2 se passes, of 3 se fails
    near = check([0.0, 1.0], [4.0, 4.0 + 0.1 * math.sqrt(2) * 2.0], [0.1, 0.1])
    assert near["holds"] and near["z_critical"] == pytest.approx(2.326, abs=1e-3)
    bad = check([0.0, 1.0], [4.0, 4.0 + 0.1 * math.sqrt(2) * 3.0], [0.1, 0.1])
    assert not bad["holds"] and bad["failed"] == [bad["worst"]]
    assert bad["worst"]["J"] == [0.0, 1.0] and bad["worst"]["z"] == pytest.approx(3.0)
    assert bad["worst"]["increase"] == pytest.approx(0.1 * math.sqrt(2) * 3.0)
    # the critical z grows with the number of steps (Bonferroni): 3.1 for 12 couplings
    many = check(list(range(12)), [10.0 - i for i in range(12)], [0.1] * 12)
    assert many["steps"] == 11 and many["z_critical"] == pytest.approx(3.12, abs=0.01)
    # only consecutive couplings are compared: a slow rise spread over small steps is not a failure
    slow = check(list(range(6)), [4.0, 4.1, 4.2, 4.3, 4.4, 4.5], [0.1] * 6)
    assert slow["holds"]
    # no variation at all: a real rise is as significant as can be, equal means are not a rise
    assert not check([0.0, 1.0], [4.0, 5.0], [0.0, 0.0])["holds"]
    assert check([0.0, 1.0], [4.0, 4.0], [0.0, 0.0])["holds"]
    # the failed steps are listed with their z, at most four
    zigzag = check(list(range(12)), [0.0 if i % 2 else 5.0 for i in range(12)][::-1], [0.1] * 12)
    assert len(zigzag["failed"]) <= 4


def test_proposition_two_false_alarms_are_rare():
    # a correct sampler, twelve couplings 0.001 apart (the true means are nearly equal), 400
    # calls: the chance of a false "no" is shared at 1% over the 11 steps
    couplings = [0.001 * i for i in range(12)]
    wrong = sum(
        not api.sample_erasure_regions(RING, 8, couplings, 200, num_nodes=32, seed=seed)[
            "nonincreasing_in_J"
        ]
        for seed in range(400)
    )
    assert wrong <= 12, wrong  # 3%: the expected number is 4


def test_the_result_reports_which_proposition_two_step_fails(monkeypatch):
    out = _regions()
    detail = out["proposition2"]
    assert out["nonincreasing_in_J"] is True and detail["failed"] == []
    assert detail["steps"] == 2 and detail["z_critical"] == pytest.approx(2.576, abs=0.01)
    assert detail["worst"]["J"] in ([0, 1], [1, 2]) and detail["worst"]["z"] < 0
    # force a failure: the estimates at J = 1 and J = 2 are swapped
    real = api._proposition_two

    def rigged(js, means, errors):
        return real(js, [means[0], means[2], means[1]], errors)

    with monkeypatch.context() as patch:
        patch.setattr(api, "_proposition_two", rigged)
        bad = _regions()
    assert bad["nonincreasing_in_J"] is False
    (step,) = bad["proposition2"]["failed"]
    assert step["J"] == [1, 2] and step["z"] > bad["proposition2"]["z_critical"]
    assert any(
        note.startswith("the mean boundary rises with the coupling") for note in bad["mixing"]
    )
    assert "Proposition 2 forbids that" in bad["mixing"][-1] and "J = 1 to 2" in bad["mixing"][-1]
    assert _regions(couplings=[1.0])["proposition2"] is None


def test_poor_mixing_is_reported_with_what_to_change():
    out = api.sample_erasure_regions(
        WS, 16, [0, 4], 16, num_nodes=64, seed=1, burn_in=0, tempering=False, chains=4
    )
    text = " ".join(out["mixing"])
    assert "R-hat" in text and "effective sample size" in text
    assert "burn_in" in text and "samples" in text
    rhat = next(note for note in out["mixing"] if note.startswith("R-hat"))
    assert "J = 4 (" in rhat and "4 chains" in rhat  # the coupling, its value and its limit
    assert " > " in rhat and "autocorrelation" in rhat  # what the R-hat is compared with
    assert len(out["mixing"]) <= 4  # one note per kind of problem, however many couplings


def test_the_mixing_thresholds():
    def notes(method="chain", *, r_hat=1.005, limit=1.03, ess=500.0, deviation=0.0, exchange=0.5,
              proposition=None, chains=4):  # fmt: skip
        rows = [{"coupling": 1.0, "ess": ess, "deviation_se": deviation}]
        details = [{"r_hat": r_hat, "r_hat_limit": limit, "r_hat_node": 7, "slowest": None}]
        return api._regions_mixing_notes(
            rows, details, method=method, chains=chains, samples=200, burn_in=100,
            pair_acceptance=np.array([exchange]), ladder=np.array([0.0, 1.0]),
            proposition=proposition,
        )  # fmt: skip

    assert notes() == []
    (note,) = notes(r_hat=1.04)  # R-hat above its own limit, with the node it was seen on
    assert note.startswith("R-hat over 4 chains above the limit")
    assert "J = 1 (1.040 > 1.030, node 7)" in note
    assert "not in equilibrium yet" in note and "too short for their autocorrelation" in note
    assert "forgotten" not in note  # it does not claim the cause
    assert notes(r_hat=1.029) == [] and notes(r_hat=1.04, method="exact") == []  # no chain
    assert notes(r_hat=1.02, limit=1.01) != []  # the limit is per series, not 1.01 or 1.03
    (note,) = notes(ess=399.0)
    assert note.startswith(
        "effective sample size below 400 (100 per chain) at J = 1 (399, boundary)"
    )
    assert notes(ess=401.0) == [] and len(notes(ess=399.0, method="exact")) == 1  # all samplers
    # the threshold is 100 per chain: 200 for two chains, 800 for eight
    assert notes(ess=250.0, chains=2) == [] and len(notes(ess=150.0, chains=2)) == 1
    (note,) = notes(ess=700.0, chains=8)
    assert note.startswith("effective sample size below 800 (100 per chain) at J = 1 (700,")
    assert notes(ess=801.0, chains=8) == []
    (note,) = notes(deviation=-5.5)
    assert "more than 5 standard errors" in note and "(-5.5)" in note and "burn_in" in note
    assert notes(deviation=4.9) == []
    assert "report it" in notes(deviation=6.0, method="exact")[0]  # an exact sampler is not biased
    (note,) = notes(exchange=0.049)
    assert note.startswith("replica exchange accepted in under 5%") and "J = 0 and 1 (4.9%)" in note
    assert notes(exchange=0.051) == [] and notes(exchange=0.001, method="exact") == []
    assert notes(exchange=float("nan")) == []  # no exchanges happened (one rung)
    failed = {"holds": False, "z_critical": 3.0, "steps": 5, "failed": [
        {"J": [0.5, 1.0], "increase": 0.4, "z": 4.2}]}  # fmt: skip
    (note,) = notes(proposition=failed)
    assert "J = 0.5 to 1 (+0.4, z = 4.2)" in note and "above z = 3.00 (5 steps, 1% " in note
    assert notes(proposition=dict(failed, holds=True)) == []
    everything = notes(r_hat=2.0, ess=10.0, deviation=9.0, exchange=0.0, proposition=failed)
    assert len(everything) == 5  # one per kind


def test_chains_stuck_apart_have_an_infinite_r_hat_that_is_reported():
    for seed in range(1, 6):  # chains frozen apart at J = 50, whatever the seed
        out = api.sample_erasure_regions(
            WS, 16, [0, 50], 40, num_nodes=64, seed=seed, burn_in=200, tempering=False
        )
        cold = out["couplings"][1]
        assert cold["acceptance"] < 0.01
        if cold["r_hat"] == api.R_HAT_INFINITE:
            break
    else:
        pytest.fail("no seed froze the chains apart")
    note = next(n for n in out["mixing"] if n.startswith("R-hat"))
    assert "J = 50 (infinite > " in note
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
        ({"couplings": [1e308]}, r"\|J\| <= 1,000,000.*exact routes"),
        ({"couplings": ["a"]}, "couplings"),
        ({"couplings": [True]}, "couplings"),
        ({"couplings": list(range(40))}, "at most 12 couplings"),
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
        {"family": "cycle"}, 40, [0.1 * i for i in range(12)], 200, num_nodes=128, seed=2
    )
    assert len(json.dumps(out)) <= api.MAX_OUTPUT_CHARS
    assert len(out["couplings"]) == 12


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
    # twelve couplings, every row with an exact value, every kind of note, the sampler's warnings
    out = api.sample_erasure_regions(
        WS, 6, [0.5 * i - 2 for i in range(12)], 16, num_nodes=20, seed=3, burn_in=0,
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
    assert {k: v for k, v in a.items() if k not in ("path", "elapsed_seconds")} == {
        k: v for k, v in b.items() if k not in ("path", "elapsed_seconds")
    }
    assert a["elapsed_seconds"] > 0
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


def test_curves_keep_the_smallest_and_largest_coupling_whatever_the_order(tmp_path):
    couplings = [2.5, -1.0, 0.5, 3.0, 0.0, 1.5, 2.0, -0.5, 1.0, 0.25, 0.75, 1.25, 1.75, -2.0]
    spec = dict(CURVE_SPEC, checkpoints="final", probes=[
        {"kind": "code_dimension"},
        {"kind": "correlated_erasure", "couplings": couplings, "samples": 10},
    ])  # fmt: skip
    api.run_ensemble(spec, 10, seed=2, output_dir=str(tmp_path / "unsorted"))
    out = api.erasure_curves(str(tmp_path / "unsorted"), e=[0.1, 0.3])
    entry = out["correlated"]["correlated_erasure"]
    assert entry["num_couplings"] == 14 and out["truncated"] is True
    shown = entry["couplings"]
    assert shown == sorted(shown) and shown[0] == -2.0 and shown[-1] == 3.0
    assert len(entry["phi"]["mean"]) == len(shown) <= api.MAX_CURVE_COUPLINGS
    assert any("increasing order of J" in note for note in out["notes"])
    # the curve shown for J = 3 is J = 3's: it equals the one-coupling analysis of that column
    result = mipt.EnsembleResult.load(tmp_path / "unsorted" / "result.npz")
    c, sizes, values = result.correlated()
    held = result.values["code_dimension"][:, -1] > 0
    curve, _ = analysis.correlated_failure_curve(
        values[held, -1][:, [c.tolist().index(3.0)]], sizes, N, holds_code=np.ones(int(held.sum()))
    )
    phi = analysis.failure_probability_from_curve(curve, N, [0.1, 0.3]).mean(axis=0)
    np.testing.assert_allclose(entry["phi"]["mean"][-1], phi[0], rtol=1e-5)


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


def _narrow_window_run(tmp_path, sizes):
    spec = dict(CURVE_SPEC, layers=48, checkpoints="final", probes=[
        {"kind": "code_dimension"},
        {"kind": "correlated_erasure", "couplings": [0, 2], "sizes": sizes, "samples": 100},
    ])  # fmt: skip
    out = tmp_path / "narrow"
    api.run_ensemble(spec, 40, seed=5, output_dir=str(out))
    return out


def test_the_tolerance_reports_the_mass_it_leans_on_interpolation_for(tmp_path):
    # sizes 1..3 sampled, 4..15 interpolated: at the small e asked for they carry almost
    # nothing (complete), but e_delta for delta = 0.1 sits where they carry about 3%
    out = api.erasure_curves(str(_narrow_window_run(tmp_path, "1:4")), e=[0.01, 0.02], delta=0.1)
    entry = out["correlated"]["correlated_erasure"]
    assert entry["complete"] is True and max(entry["interpolated_mass"]) < 1e-3
    assert entry["e_delta_complete"] is False
    tolerance = entry["e_delta"]
    mass = tolerance["interpolated_mass_at_e_delta"]  # per coupling: at of_mean_phi, then the range
    assert len(mass) == 2 and all(len(row) == 3 for row in mass)
    for i, row in enumerate(mass):
        at = [tolerance["of_mean_phi"][i], *tolerance["of_mean_phi_range"][i]]
        expected = [
            sum(math.comb(N, m) * x**m * (1 - x) ** (N - m) for m in range(4, N)) for x in at
        ]
        np.testing.assert_allclose(row, expected, rtol=1e-4)
    assert max(max(row) for row in mass) > api.INTERPOLATION_MASS_WARNING
    note = next(x for x in out["notes"] if "e_delta_complete is false" in x)
    assert "interpolated_mass_at_e_delta" in note and "at the e asked for only" in note
    assert tolerance["of_mean_phi_monotone"] == [True, True]  # the flag of the curve behind it


def test_the_tolerance_of_a_fully_sampled_run_is_complete(tmp_path):
    out = api.erasure_curves(str(_narrow_window_run(tmp_path, None)), e=[0.01, 0.02], delta=0.1)
    entry = out["correlated"]["correlated_erasure"]
    assert entry["num_interpolated"] == 0 and entry["complete"] is True
    assert entry["e_delta_complete"] is True
    assert entry["e_delta"]["interpolated_mass_at_e_delta"] == [[0.0] * 3] * 2
    assert not any("e_delta_complete" in x for x in out["notes"])


def test_the_uniform_channel_has_no_interpolation_flags_but_a_monotone_flag(run_dir, result):
    out = api.erasure_curves(str(run_dir), e=[0.05, 0.1], delta=0.05)
    uniform = out["uniform"]["erasure_first_failure"]
    assert "e_delta_complete" not in uniform and "interpolated_mass" not in uniform
    first = result.values["erasure_first_failure"][:, 1]
    counts = result.values["uncorrectable_region_counts[2]"][:, 1]
    code = first.min(axis=1) <= N
    curve = analysis.hybrid_failure_by_size(first[code], N, exact_counts=counts[code])
    _, monotone = analysis.tolerance_from_curve(curve.mean(axis=0), N, 0.05, return_monotone=True)
    assert uniform["e_delta"]["of_mean_phi_monotone"] is bool(monotone)


def test_the_monotone_flag_of_the_mean_curve_is_false_for_a_curve_that_steps_down():
    down = np.array([0.0, 0.5, 0.2, 0.6, 1.0])
    up = np.array([0.0, 0.1, 0.2, 0.6, 1.0])
    for curve, flag in ((down, False), (up, True)):
        chunk = np.tile(curve, (3, 1, 1))  # three codes, one coupling
        stats = api._curve_statistics(iter([chunk]), 4, [0.1], 0.05, 3, 3)
        assert stats["tol_of_mean_monotone"].tolist() == [flag]
        assert api._tolerance_entry(stats, 0)["of_mean_phi_monotone"] is np.bool_(flag)


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


def test_capabilities_give_an_example_of_each_erasure_tool(tmp_path):
    examples = api.capabilities()["examples"]
    regions = examples["sample_erasure_regions"]
    assert set(regions) <= set(inspect.signature(api.sample_erasure_regions).parameters)
    out = api.sample_erasure_regions(**regions)  # the example runs as written
    assert out["size"] == 16 and len(out["couplings"]) == 4
    curves = examples["erasure_curves"]
    assert set(curves) <= set(inspect.signature(api.erasure_curves).parameters)
    # and the example's call works on a saved run once its path names one
    spec = dict(CURVE_SPEC, probes=[{"kind": "code_dimension"},
                                    {"kind": "erasure_first_failure", "samples": 50}])  # fmt: skip
    api.run_ensemble(spec, 8, seed=1, output_dir=str(tmp_path / "erasure"))
    done = api.erasure_curves(**{**curves, "path": str(tmp_path / "erasure")})
    assert done["e"] == curves["e"] and done["delta"] == curves["delta"]
    assert len(json.dumps(examples)) < 1500  # the overview keeps its other sections


# ---------------------------------------------------------------------------
# the corrections of the review (fix round 1)
# ---------------------------------------------------------------------------


def test_the_r_hat_limit_is_the_f_quantile_and_scales_with_autocorrelation():
    # Monte Carlo values for 4 chains of 200 draws: 18 series (one coupling), 144, 288
    for series, expected in ((18, 1.0136), (144, 1.0173), (288, 1.0185)):
        assert api._rhat_limit(4, 200, series) == pytest.approx(expected, abs=3e-4)
    # the figures quoted in the docstring are for the most couplings a call takes (12), not 16
    series = 2 * (1 + 8) * api.MAX_REGION_COUPLINGS
    assert api._rhat_limit(4, 200, series) == pytest.approx(1.0179, abs=5e-5)
    assert "1.0179 for 12" in api._rhat_limit.__doc__ and "for 16" not in api._rhat_limit.__doc__
    f = np.random.default_rng(0).f(7, 792, size=2_000_000)  # M - 1 and M (L - 1) degrees
    for value in (1.0, 2.0, 3.0):
        tail = api._f_upper_tail(value, 7, 792)
        assert tail == pytest.approx(float(np.mean(f > value)), rel=0.03)
    q = api._f_quantile(1e-4, 7, 792)
    assert api._f_upper_tail(q, 7, 792) == pytest.approx(1e-4, rel=1e-6)
    base = api._rhat_limit(4, 200, 18)
    assert api._rhat_limit(4, 200, 18, tau=3.0) > base + 0.02  # autocorrelation raises it
    assert api._rhat_limit(4, 200, 18, tau=0.3) == base  # tau is at least 1
    assert api._rhat_limit(4, 200, 300) > base  # more series, a higher limit
    assert api._rhat_limit(4, 20000, 18) == api.RHAT_WARNING  # long records: the floor
    assert api._rhat_limit(1, 8, 18) > 1.5 and api._rhat_limit(1, 2, 18) == math.inf


def test_the_r_hat_note_is_rare_on_a_correct_chain_and_fires_on_a_slow_one():
    def fires(out):
        return any(note.startswith("R-hat") for note in out["mixing"])

    fired = sum(
        fires(
            api.sample_erasure_regions(
                RING, 8, [0, 1, 2], 200, num_nodes=32, seed=seed, method="chain"
            )
        )
        for seed in range(150)
    )
    assert fired <= 10, fired  # measured 2.7% of 300 calls; the limit for independent draws: 100%
    ws64 = {"family": "watts_strogatz", "k": 4, "beta": 0.1, "seed": 1}
    slow = [
        api.sample_erasure_regions(ws64, 16, [0, 0.5, 1, 2], 200, num_nodes=64, seed=seed)
        for seed in range(5)
    ]  # the regions of this graph move slowly at J >= 1
    assert sum(fires(out) for out in slow) >= 3
    note = next(n for out in slow for n in out["mixing"] if n.startswith("R-hat"))
    assert "equilibrium yet" in note and "thin or samples" in note


def test_num_nodes_may_sit_inside_the_graph_beside_the_family():
    inside = api.sample_erasure_regions(dict(RING, num_nodes=32), 8, [0, 1], 40, seed=3)
    beside = api.sample_erasure_regions(RING, 8, [0, 1], 40, num_nodes=32, seed=3)
    assert _without_timing(inside) == _without_timing(beside)
    both = api.sample_erasure_regions(dict(RING, num_nodes=32), 8, [0, 1], 40, num_nodes=32, seed=3)
    assert _without_timing(both) == _without_timing(beside)
    with pytest.raises(ValueError, match=r"num_nodes is 16 but graph.num_nodes is 32"):
        api.sample_erasure_regions(dict(RING, num_nodes=32), 8, [0, 1], 40, num_nodes=16, seed=3)
    with pytest.raises(ValueError, match="graph.num_nodes must be an integer"):
        api.sample_erasure_regions(dict(RING, num_nodes="many"), 8, [0, 1], 40, seed=3)


def test_small_input_mistakes_get_their_own_messages(tmp_path):
    with pytest.raises(ValueError, match="edge list is empty"):
        api.sample_erasure_regions({"edges": [], "num_nodes": 4}, 1, [0], 20, seed=1)
    with pytest.raises(ValueError, match="couplings must be numbers"):
        api.sample_erasure_regions(RING, 8, ["1"], 20, num_nodes=32, seed=1)
    out = api.sample_erasure_regions(
        {"edges": [[0, 1], [1, 2], [2, 0], [1, 0]], "num_nodes": 3}, 1, [0], 20, seed=1
    )
    assert any("1 repeated edge(s) were merged" in note for note in out["notes"])
    n = 6
    graphs.save(tmp_path / "weighted.npz", [graphs.cycle(n)], n, weights=[np.arange(1.0, n + 1)])
    out = api.sample_erasure_regions(
        {"family": "file", "path": str(tmp_path / "weighted.npz")}, 2, [0], 20, seed=1
    )
    assert any("edge weights are ignored" in note for note in out["notes"])


def test_exact_samples_report_independent_draws_not_estimates():
    out = _regions(couplings=(0, 2), samples=300, chains=2)
    for row in out["couplings"]:
        assert (row["tau_int"], row["ess"], row["ess_boundary"]) == (1, 600, 600)
        assert row["stderr"] == pytest.approx(math.sqrt(row["variance"] / 600), rel=1e-4)
        assert row["r_hat"] is None and row["r_hat_limit"] is None
    assert any("tau_int is 1" in note for note in out["notes"])
    # a chain reports the boundary's own effective sample size beside the slowest series'
    chain = api.sample_erasure_regions(WS, 5, [0, 1.5], 100, num_nodes=14, seed=4, burn_in=60)
    for row in chain["couplings"]:
        expected = math.sqrt(row["variance"] / row["ess_boundary"])
        assert row["stderr"] == pytest.approx(expected, rel=1e-3)
        assert row["ess"] <= row["ess_boundary"] * (1 + 1e-9)
        assert row["r_hat"] > 0 and row["r_hat_limit"] >= api.RHAT_WARNING and row["r_hat_at"]


def test_standard_errors_of_exact_samples_are_calibrated():
    # z = (mean - exact) / stderr over many calls has standard deviation 1 (it was 1.05 when
    # tau was estimated from independent draws)
    z = [
        row["deviation_se"]
        for seed in range(150)
        for row in api.sample_erasure_regions(RING, 8, [0, 1, 2], 200, num_nodes=32, seed=seed)[
            "couplings"
        ]
    ]
    assert abs(float(np.mean(z))) < 0.15 and 0.93 < float(np.std(z)) < 1.07


def test_the_curve_work_is_planned_from_the_size_of_the_problem():
    plan = api._curve_plan
    assert plan(16, 2, 50) == (50, 50)  # few codes: all of them
    used, tolerance = plan(64, 12, 100_000)  # small n: only the cap on codes binds
    assert used == 100_000 and tolerance == api.MAX_TOLERANCE_CODES
    for n, c in ((256, 12), (1024, 12), (4096, 4)):
        used, tolerance = plan(n, c, 10**6)
        assert used * c * (n + 1) <= api.MAX_CURVE_VALUES + c * (n + 1)
        assert tolerance * c * (n + 1) * 2049 <= api.MAX_TOLERANCE_WORK + c * (n + 1) * 2049
        assert api.MIN_TOLERANCE_CODES <= tolerance <= used
    used, tolerance = plan(65_536, 12, 10**6)  # the largest system: the floors keep it working
    assert used == max(api.MIN_CURVE_CODES, int(api.MAX_CURVE_VALUES // (12 * 65_537)))
    assert tolerance == api.MIN_TOLERANCE_CODES
    assert plan(65_536, 12, 5) == (5, 5)  # never more codes than there are


def test_curves_bound_the_codes_they_process_and_say_so(tmp_path, monkeypatch):
    spec = dict(CURVE_SPEC, probes=[
        {"kind": "code_dimension"},
        {"kind": "erasure_first_failure", "samples": 40},
        {"kind": "correlated_erasure", "couplings": [0, 1], "sizes": "1:16", "samples": 10},
    ], checkpoints="final")  # fmt: skip
    api.run_ensemble(spec, 600, seed=3, output_dir=str(tmp_path / "bounded"))
    full = api.erasure_curves(str(tmp_path / "bounded"), e=[0.1, 0.2])
    monkeypatch.setattr(api, "MAX_CURVE_VALUES", 1000.0)  # at the floor: 100 codes
    monkeypatch.setattr(api, "MAX_TOLERANCE_WORK", 1.0)  # at the floor: 10 codes
    out = api.erasure_curves(str(tmp_path / "bounded"), e=[0.1, 0.2])
    uniform, correlated = (
        out["uniform"]["erasure_first_failure"],
        out["correlated"]["correlated_erasure"],
    )
    assert uniform["phi"]["codes"] == correlated["phi"]["codes"] == api.MIN_CURVE_CODES
    assert uniform["e_delta"]["codes"] == api.MIN_TOLERANCE_CODES
    assert any("to bound the work" in note and "100 of the" in note for note in out["notes"])
    assert uniform["count_with_code"] == full["uniform"]["erasure_first_failure"]["count_with_code"]
    # the codes used are evenly spaced, and Phi is theirs
    result = mipt.EnsembleResult.load(tmp_path / "bounded" / "result.npz")
    first = result.values["erasure_first_failure"][:, -1]
    rows = np.flatnonzero(first.min(axis=1) <= N)
    rows = rows[api._subsample(rows.size, api.MIN_CURVE_CODES)]
    phi = analysis.erasure_failure_probability(first[rows], N, np.array([0.1, 0.2]))
    np.testing.assert_allclose(uniform["phi"]["mean"], phi.mean(axis=0), rtol=1e-5)
    # still an unbiased summary of the whole: within a few errors of the full computation
    both = zip(uniform["phi"]["mean"], full["uniform"]["erasure_first_failure"]["phi"]["mean"],
               uniform["phi"]["stderr"], strict=True)  # fmt: skip
    assert all(abs(a - b) < 5 * se for a, b, se in both)
    assert out["elapsed_seconds"] > 0


def test_chunking_does_not_change_the_statistics(run_dir, monkeypatch):
    whole = api.erasure_curves(str(run_dir), e=[0.05, 0.2], delta=0.05)
    monkeypatch.setattr(api, "_ROW_CHUNK", 7)  # a handful of codes at a time
    assert api._chunk_rows(N, 2) == 3
    chunked = api.erasure_curves(str(run_dir), e=[0.05, 0.2], delta=0.05)

    def same(a, b):
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
        if isinstance(a, float):
            return a == pytest.approx(b, rel=1e-5, abs=1e-12)
        return a == b

    strip = ("elapsed_seconds",)
    assert same(
        {k: v for k, v in whole.items() if k not in strip},
        {k: v for k, v in chunked.items() if k not in strip},
    )


def test_the_mean_tolerance_comes_with_a_band(run_dir):
    out = api.erasure_curves(str(run_dir), delta=0.05)
    uniform = out["uniform"]["erasure_first_failure"]["e_delta"]
    low, high = uniform["of_mean_phi_range"]
    assert low < uniform["of_mean_phi"] < high  # the higher curve tolerates less
    correlated = out["correlated"]["correlated_erasure"]["e_delta"]
    for center, (low, high) in zip(
        correlated["of_mean_phi"], correlated["of_mean_phi_range"], strict=True
    ):
        assert low <= center <= high and low < high
    note = " ".join(out["notes"])  # the note on the two tolerances is always there
    assert "of_mean_phi is the tolerance of the ensemble-mean Phi" in note and "headline" in note
    only = api.erasure_curves(str(run_dir.parent / "run" / "result.npz"))
    assert "headline" in " ".join(only["notes"])


def test_curves_uniform_and_j0_correlated_agree_per_trajectory(run_dir, result):
    # the two channels are estimated on the same codes, so compare them code by code
    out = api.erasure_curves(str(run_dir), e=[0.03, 0.06, 0.1, 0.15, 0.25])
    uniform = out["uniform"]["erasure_first_failure"]["phi"]["mean"]
    correlated = out["correlated"]["correlated_erasure"]["phi"]["mean"][0]
    es = np.array([0.03, 0.06, 0.1, 0.15, 0.25])
    first = result.values["erasure_first_failure"][:, 1]
    counts = result.values["uncorrectable_region_counts[2]"][:, 1]
    code = first.min(axis=1) <= N
    couplings, sizes, values = result.correlated()
    p_uniform = analysis.hybrid_failure_by_size(first[code], N, exact_counts=counts[code])
    p_corr = analysis.correlated_failure_curve(values[code, 1, :1, :], sizes, N)[0][:, 0]
    diff = analysis.failure_probability_from_curve(
        p_uniform, N, es
    ) - analysis.failure_probability_from_curve(p_corr, N, es)
    se = diff.std(axis=0, ddof=1) / math.sqrt(diff.shape[0])
    gap = np.array(uniform) - np.array(correlated)
    np.testing.assert_allclose(gap, diff.mean(axis=0), atol=2e-6)  # the tool's means (6 digits)
    assert np.all(np.abs(gap) < 4 * se + 2e-6), (gap, se)  # a paired test: 10% bias would fail


def test_curves_without_a_code_probe_give_p_of_n_zero_to_trajectories_that_lose_nothing(tmp_path):
    spec = dict(CURVE_SPEC, measurement_rate=0.2, checkpoints="final", probes=[
        {"kind": "correlated_erasure", "couplings": [0, 1], "sizes": "2:15:3", "samples": 40}
    ])  # fmt: skip
    api.run_ensemble(spec, 60, seed=4, output_dir=str(tmp_path / "bare"))
    result = mipt.EnsembleResult.load(tmp_path / "bare" / "result.npz")
    _, sizes, values = result.correlated()
    sampled = values[:, -1]
    nothing = sampled.max(axis=(1, 2)) == 0
    assert 0 < nothing.sum() < nothing.size  # some trajectories hold no code, some do
    out = api.erasure_curves(str(tmp_path / "bare"), e=[0.2, 0.5])
    entry = out["correlated"]["correlated_erasure"]
    curve = analysis.correlated_failure_curve(sampled, sizes, N, holds_code=~nothing)[0]
    phi = analysis.failure_probability_from_curve(curve, N, np.array([0.2, 0.5]))
    np.testing.assert_allclose(entry["phi"]["mean"], phi.mean(axis=0), rtol=1e-5)
    # a trajectory that loses nothing does not rise to 1 at the last size: its curve is 0
    assert np.all(curve[nothing] == 0)


def test_curves_of_a_graph_collection_warn_about_correlated_circuits(tmp_path):
    spec = dict(CURVE_SPEC, graph={"family": "watts_strogatz", "k": 4, "beta": 0.4, "seed": 1,
                                   "realizations": 4}, circuits_per_graph=5,
                schedule="random_edges", probes=[
        {"kind": "code_dimension"}, {"kind": "erasure_first_failure", "samples": 30}
    ], checkpoints="final")  # fmt: skip
    api.run_ensemble(spec, 20, seed=2, output_dir=str(tmp_path / "bank"))
    out = api.erasure_curves(str(tmp_path / "bank"), e=[0.1])
    assert any("ran on 4 graphs" in note and "treat every trajectory as independent" in note
               for note in out["notes"])  # fmt: skip


def test_curves_with_stopped_trajectories_and_two_probes_of_a_kind(tmp_path):
    spec = dict(CURVE_SPEC, measurement_rate=0.2, stop_when_purified=True, probes=[
        {"kind": "code_dimension"},
        {"kind": "erasure_first_failure", "samples": 50, "name": "ff50"},
        {"kind": "erasure_first_failure", "samples": 120, "name": "ff120"},
        {"kind": "correlated_erasure", "couplings": [0, 1], "samples": 30, "name": "ce_a"},
        {"kind": "correlated_erasure", "couplings": [0.5], "sizes": "3:14:2", "samples": 30,
         "name": "ce_b"},
    ], checkpoints="final")  # fmt: skip
    api.run_ensemble(spec, 50, seed=5, output_dir=str(tmp_path / "two"))
    out = api.erasure_curves(str(tmp_path / "two"), e=[0.1])
    assert set(out["uniform"]) == {"ff50", "ff120"} and set(out["correlated"]) == {"ce_a", "ce_b"}
    result = mipt.EnsembleResult.load(tmp_path / "two" / "result.npz")
    holders = float(np.mean(result.values["code_dimension"][:, -1] > 0))
    assert 0 < holders < 1  # some trajectories stopped: their reference is pure
    for entry in (*out["uniform"].values(), *out["correlated"].values()):
        assert entry["q_plus"] == pytest.approx(holders)
    assert out["correlated"]["ce_a"]["couplings"] == [0.0, 1.0]
    assert out["correlated"]["ce_b"]["couplings"] == [0.5]
    assert (
        out["correlated"]["ce_b"]["num_interpolated"]
        > 0
        >= out["correlated"]["ce_a"]["num_interpolated"]
    )
