"""The JSON service layer of the study (``sparsegf2.study.api``): results, errors and wiring.

Every number the service reports is checked against an independent computation written in the
tests (``_study_api_support``: nested means and graph-cluster errors by loops, a product-limit
estimator, Pauli strings letter by letter) on a tiny study, or against the store's own records;
every input error is a ``ValueError`` with a message that says what to change; every result is
plain JSON within ``sparsegf2.api.MAX_OUTPUT_CHARS``. No test asserts a wall-clock time.
"""

from __future__ import annotations

import json
import math
import threading

import numpy as np
import pytest

from _study_api_support import (
    MANIFEST,
    build,
    km_median,
    km_reference,
    manifest,
    nested_reference,
    pauli_strings,
)
from sparsegf2 import api, mipt
from sparsegf2.study import api as sa
from sparsegf2.study import observables as ob
from sparsegf2.study import runner
from sparsegf2.study.manifest import ExperimentManifest
from sparsegf2.study.store import TableauStore
from sparsegf2.tableau import PhaseFreeTableau


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build(tmp_path_factory.mktemp("study_api") / "store")


def small(result: dict) -> dict:
    """The result is plain JSON within the size limit (what an agent reads whole)."""
    text = json.dumps(result)
    assert len(text) <= api.MAX_OUTPUT_CHARS, len(text)
    return json.loads(text)


def opened(path):
    return TableauStore.open(path)


# ---------------------------------------------------------------------------------------------
# catalog, init, status
# ---------------------------------------------------------------------------------------------


def test_catalog_lists_every_observable_compactly():
    out = small(sa.catalog())
    assert out["count"] == len(ob.catalog())
    assert [r["id"] for r in out["observables"]] == [r["id"] for r in ob.catalog()]
    assert {"id", "version", "level", "retrospective", "params", "definition"} <= set(
        out["observables"][0]
    )
    one = small(sa.catalog("d_cyc"))
    assert one["observable"] == "d_cyc" and one["versions"][0]["function"]
    with pytest.raises(ValueError, match="no observable 'nope' in the catalog"):
        sa.catalog("nope")


def test_init_reads_a_dict_json_text_or_a_file(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps(MANIFEST), encoding="utf-8")
    digests = []
    for i, given in enumerate((MANIFEST, json.dumps(MANIFEST), str(path))):
        out = small(sa.init(given, str(tmp_path / f"s{i}")))
        digests.append(out["manifest_digest"])
        assert out["revision"] == 1 and out["planned_trajectories"] == 6 * 6
        assert out["planned_units"] == 12  # 3 graphs, 2 per unit, 6 conditions
    assert len(set(digests)) == 1
    assert digests[0] == ExperimentManifest.from_json(MANIFEST).digest()
    with pytest.raises(ValueError, match="exists and is not an empty directory"):
        sa.init(MANIFEST, str(tmp_path / "s0"))
    with pytest.raises(ValueError, match="trials"):
        sa.init(manifest(trials={"graphs": 0, "circuits_per_graph": 2}), str(tmp_path / "bad"))
    with pytest.raises(ValueError, match="no such file"):
        sa.init(str(tmp_path / "missing.json"), str(tmp_path / "bad2"))


def test_status_reports_counts_that_match_the_store(store):
    out = small(sa.status(store))
    assert out["complete"] and out["revision"] == 1
    t = out["trajectories"]
    assert t == {"expected": 36, "stored": 36, "complete": 36, "remaining": 0, "reruns": 0}
    assert out["units"]["done"] == 12 and out["units"]["remaining"] == 0
    with opened(store) as s:
        counts = s.counts()
        assert out["snapshots"]["total"] == counts["snapshots"]
        stored = {(o, v) for o, v, _ in sa._stored_keys(s)}
    assert {(r["id"], r["version"]) for r in out["observables"]} == stored
    assert out["peak_rule"]["plateau_z"] == 1.0  # the plateau width, never called z here
    assert all(c["complete"] == c["expected"] == 6 for c in out["conditions"])
    assert out["storage"]["bytes_per_snapshot"] > 0


def test_status_of_a_fresh_store_lists_the_work(tmp_path):
    path = build(tmp_path / "s", run=False)
    out = small(sa.status(path))
    assert not out["complete"]
    assert out["trajectories"]["remaining"] == 36 and out["units"]["remaining"] == 12
    assert "run(store)" in out["next_step"]
    with pytest.raises(ValueError, match="is not a study store"):
        sa.status(str(tmp_path / "nothing"))


# ---------------------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------------------


def _correlated_manifest() -> dict:
    return manifest(
        observables={
            "tableau_snapshots": False,
            "contiguous_distance": False,
            "uniform_erasure": {"samples": 9, "exact_budget": 8},
            "correlated_erasure": [
                {"couplings": [0, 1.5], "sizes": [1, 3], "samples": 8, "name": "ring"}
            ],
        },
        protocols=["F", "P"],
    )


@pytest.mark.parametrize("make", [lambda: MANIFEST, _correlated_manifest])
def test_the_estimate_spec_has_the_runners_engine_probes(make, tmp_path):
    # plan prices each unit with api.estimate_ensemble on a JSON spec; that spec must be the
    # engine run the runner makes (same keywords, same probes with the same names and settings)
    m = ExperimentManifest.from_json(make())
    with TableauStore.create(tmp_path / "s", m) as s:
        revisions = s.revisions()
        for protocol in m.protocol_names:
            for cond in m.conditions():
                req = runner.requirements(revisions, protocol, cond.n, cond.beta)
                p = cond.rates[0]
                edges = sa._Estimator(s, m).edges(cond.n, cond.beta.token)
                built = api._build_spec(sa._unit_spec(m, req, p.token, edges))
                kwargs = m.monitored_circuit_kwargs(cond.n, cond.beta, p, protocol)
                kwargs.pop("circuits_per_graph")
                expected = mipt.MonitoredCircuit(
                    edges=edges, probes=runner.engine_probes(req), **kwargs
                )
                assert built.probes == expected.probes
                for key in kwargs:
                    assert getattr(built, key) == getattr(expected, key), key


def test_plan_estimates_each_unit_with_estimate_ensemble(tmp_path):
    path = build(tmp_path / "s", run=False)
    out = small(sa.plan(path, threads=2))
    assert out["units"] == 12 and out["trajectories"] == 36 and out["threads"] == 2
    assert out["factor"] is None and out["estimated_seconds"] is None  # nothing measured yet
    assert any("benchmark=true" in note for note in out["notes"])
    with opened(path) as s:
        units = runner.plan(s)
        revisions = s.revisions()
        m = revisions[-1]
        total = 0.0
        for u in units:
            req = runner.requirements(revisions, u.protocol, u.n, u.beta)
            spec = sa._unit_spec(m, req, u.p.token, sa._Estimator(s, m).edges(u.n, u.beta.token))
            est = api.estimate_ensemble(spec, u.trajectories, threads=[2])
            total += est["by_threads"][0]["estimated_seconds"]
    assert out["estimated_engine_seconds"] == pytest.approx(total, rel=1e-5)
    shown = out["units_shown"]
    assert [row["unit"] for row in shown] == [str(u) for u in units[: len(shown)]]
    assert out["storage"]["new_snapshots"] == 36 * 6  # six depths: 0, 1, 2, ..., T
    assert out["peak_memory_megabytes"] > 0


def test_plan_takes_its_factor_from_the_units_run(tmp_path):
    path = build(tmp_path / "s", run=False)
    sa.run(path, max_units=2, num_threads=1)
    out = small(sa.plan(path, threads=1))
    factor = out["factor"]
    assert factor["source"] == "store" and factor["units_measured"] == 2
    with opened(path) as s:
        measured = sum(u["elapsed_seconds"] for u in s.units(status="done"))
        estimator = sa._Estimator(s, s.manifest)
        estimated = sum(
            estimator.estimate(s.requirements(u["requirements"]), u["p"], u["trajectories"],
                               u["threads"])["engine_seconds"]
            for u in s.units(status="done")
        )  # fmt: skip
    assert factor["value"] == pytest.approx(measured / estimated, rel=1e-5)
    assert out["estimated_seconds"] == pytest.approx(
        out["estimated_engine_seconds"] * factor["value"], rel=1e-4
    )
    assert out["storage"]["source"] == "store" and out["storage"]["estimated_megabytes"] > 0


def test_plan_benchmark_measures_a_trial_and_leaves_the_study_untouched(tmp_path):
    path = build(tmp_path / "s", run=False)
    with opened(path) as s:
        before = s.logical_digest()
    out = small(sa.plan(path, threads=1, benchmark=True))
    trial = out["trial"]
    assert out["factor"]["source"] == "trial" and trial["factor"] > 0
    assert trial["trajectories"] >= 1 and trial["bytes_per_snapshot"] > 0
    assert out["storage"]["source"] == "trial"
    with opened(path) as s:
        assert s.logical_digest() == before and s.counts()["trajectories"] == 0


# ---------------------------------------------------------------------------------------------
# run and the background job
# ---------------------------------------------------------------------------------------------


def test_run_stops_between_units_and_resumes_to_the_same_store(tmp_path, store):
    path = build(tmp_path / "s", run=False)
    first = small(sa.run(path, max_units=3, num_threads=1))
    assert first["stopped"] == "max_units" and first["units_run"] == 3
    assert first["units_remaining"] == 9 and not first["complete"]
    nothing = small(sa.run(path, max_seconds=1e-9, num_threads=1))
    assert nothing["stopped"] == "max_seconds" and nothing["units_run"] == 0
    rest = small(sa.run(path, num_threads=1))
    assert rest["stopped"] == "complete" and rest["complete"] and rest["units_run"] == 9
    with opened(path) as a, opened(store) as b:  # interrupted and resumed == run in one go
        assert a.logical_digest(history=False) == b.logical_digest(history=False)


def test_batched_execution_gives_the_same_records(tmp_path, store):
    # a long planning time makes _execute run many units per runner call
    path = build(tmp_path / "s", run=False)
    with TableauStore.open(path, write=True) as s:
        state = sa._Progress()
        sa._execute(s, runner.plan(s), 10.0, num_threads=1, deadline=None, cancel=None,
                    state=state)  # fmt: skip
        assert state.units_done == 12 and state.stopped == "complete"
    with opened(path) as a, opened(store) as b:
        assert a.logical_digest(history=False) == b.logical_digest(history=False)


def test_a_study_job_is_followed_by_the_job_tools(tmp_path):
    path = build(tmp_path / "s", run=False)
    with opened(path) as s:
        first_five = sum(u.trajectories for u in runner.plan(s)[:5])
    started = small(sa.start_run_job(path, max_units=5, num_threads=1))
    assert started["kind"] == sa.JOB_KIND and started["units_total"] == 5
    done = small(api.job_status(started["job_id"], wait_seconds=60))
    assert done["status"] == "stopped" and done["stopped"] == "max_units"
    assert done["units_done"] == 5 and done["units_remaining"] == 7
    assert done["trajectories_done"] == done["trajectories_total"] == first_five
    listed = api.list_jobs()["jobs"]
    assert started["job_id"] in {j["job_id"] for j in listed}
    final = sa.start_run_job(path, num_threads=1)
    finished = api.job_status(final["job_id"], wait_seconds=60)
    assert finished["status"] == "completed" and finished["units_remaining"] == 0
    assert sa.status(path)["complete"]


def test_a_study_job_can_be_cancelled_between_units(tmp_path):
    path = build(tmp_path / "s", run=False)
    started = sa.start_run_job(path, num_threads=1)
    api.cancel_job(started["job_id"])
    out = api.job_status(started["job_id"], wait_seconds=60)
    assert out["status"] in ("cancelled", "completed")
    if out["status"] == "cancelled":
        assert out["units_remaining"] > 0 and out["stopped"] == "cancelled"
    # the lock is released: a later run finishes the plan
    assert sa.run(path, num_threads=1)["complete"]


def test_a_store_with_another_writer_is_refused_by_name(tmp_path):
    path = build(tmp_path / "s", run=False)
    with TableauStore.open(path, write=True):
        with pytest.raises(ValueError, match="is writing to"):
            sa.start_run_job(path)
        with pytest.raises(ValueError, match="is writing to"):
            sa.run(path, num_threads=1)
    assert api.list_jobs()  # the refused job was not left behind as running
    assert all(
        not (j["output_dir"] == str(sa._store_dir(path)) and j["status"] == "running")
        for j in api.list_jobs()["jobs"]
    )


def test_run_arguments_are_checked():
    for kwargs, field in (({"max_units": -1}, "max_units"), ({"max_seconds": 0}, "max_seconds"),
                          ({"num_threads": 0}, "num_threads"), ({"max_units": True}, "max_units")):  # fmt: skip
        with pytest.raises(ValueError, match=field):
            sa.run("anything", **kwargs)


# ---------------------------------------------------------------------------------------------
# extend, tableau, compute, export, verify
# ---------------------------------------------------------------------------------------------


def test_extend_adds_a_revision_or_names_the_forbidden_change(tmp_path):
    path = build(tmp_path / "s")
    same = small(sa.extend(path, MANIFEST))
    assert not same["changed"] and same["revision"] == 1 and same["units_remaining"] == 0
    wider = manifest(conditions=[{"n": [6, 8], "beta": ["0.2"], "p": ["0.1", "0.2", "0.3", "0.4"]}])
    out = small(sa.extend(path, wider))
    assert out["changed"] and out["revision"] == 2
    assert out["trajectories_remaining"] == 12 and out["rerun_units"] == 0
    with pytest.raises(ValueError, match="does not extend revision 2"):
        sa.extend(path, manifest(seeds={"master_graph_seed": 99, "master_circuit_seed": 6}))


def test_tableau_from_the_store(store):
    out = small(sa.tableau(store, n=6, beta="0.2", p=0.1, g=1, c=0, t=5))
    with opened(store) as s:
        tab = s.tableau(n=6, beta="0.2", p="0.1", g=1, c=0, t=5)
        phys = s.snapshot(n=6, beta="0.2", p="0.1", g=1, c=0, t=5).physical_tableau()
    assert out["source"] == "store" and out["joint"]["num_qubits"] == 12
    assert out["joint"] == {"num_qubits": 12, "rank": 12, "pure": True, "digest": tab.digest()}
    assert out["k"] == 6 - phys.rank and out["physical"]["digest"] == phys.digest()
    assert out["generators"] == pauli_strings(tab.canonical())
    assert out["physical_generators"] == pauli_strings(phys.canonical())
    # the strings span the same row space again
    assert PhaseFreeTableau(out["generators"]).digest() == tab.digest()


def test_tableau_replay_reproduces_the_store_and_reaches_any_depth(store, tmp_path):
    stored = sa.tableau(store, n=8, beta="0.2", p="0.3", g=2, c=1, t=6)
    again = small(sa.tableau(store, n=8, beta="0.2", p="0.3", g=2, c=1, t=6, replay=True))
    assert again["joint"] == stored["joint"] and again["source"] == "replay"
    assert again["verified"]["digest_matches_store"] is True
    assert again["verified"]["k_trace_through_layer"] == 6
    with pytest.raises(ValueError, match="pass replay=true"):
        sa.tableau(store, n=8, beta="0.2", p="0.3", g=2, c=1, t=3)
    deeper = small(sa.tableau(store, n=8, beta="0.2", p="0.3", g=2, c=1, t=40, replay=True,
                              out=str(tmp_path / "t40.sgf2rs")))  # fmt: skip
    assert deeper["verified"]["digest_matches_store"] is None  # t = 40 is past the horizon 32
    data = (tmp_path / "t40.sgf2rs").read_bytes()
    assert data.startswith(b"SGF2RS1\x00")
    assert PhaseFreeTableau.from_bytes(data).digest() == deeper["joint"]["digest"]


def test_tableau_out_npz_and_lookup_errors(store, tmp_path):
    out = sa.tableau(store, n=6, beta="0.2", p="0.2", g=0, c=1, t=2, out=str(tmp_path / "a.npz"))
    with np.load(tmp_path / "a.npz") as z:
        tab = PhaseFreeTableau(z["canonical"])
        assert tab.digest() == out["joint"]["digest"] and int(z["num_qubits"]) == 12
        meta = json.loads(str(z["metadata"]))
    assert meta["key"] == out["key"] and meta["written"] == "joint"
    with pytest.raises(ValueError, match="no trajectory"):
        sa.tableau(store, n=6, beta="0.2", p="0.2", g=7, c=0, t=2)
    with pytest.raises(ValueError, match="beta"):
        sa.tableau(store, n=6, beta="2", p="0.2", g=0, c=0, t=2)


def test_compute_is_idempotent(tmp_path):
    path = build(tmp_path / "s")
    first = small(sa.compute(path, "half_cut_entropy", n=6))
    with opened(path) as s:
        snapshots = sum(1 for _ in s.iter_snapshots(n=6, status=None))
    assert first["inserted"] == snapshots and first["existing"] == 0
    again = small(sa.compute(path, "half_cut_entropy", n=6))
    assert again["inserted"] == 0 and again["existing"] == snapshots
    with pytest.raises(ValueError, match="retrospective ones are"):
        sa.compute(path, "first_passage")


def test_export_and_verify(store, tmp_path):
    out = small(sa.export(store, str(tmp_path / "x")))
    assert out["conditions"] == 6 and "manifest.json" in out["top_level"]
    with pytest.raises(ValueError):
        sa.export(store, str(tmp_path / "x"))
    checked = small(sa.verify(store, sample=3, seed=4, num_threads=1))
    assert checked["ok"] and checked["trajectories"] == 3 and not checked["failures"]
    assert (
        small(sa.verify(store, sample=3, seed=4, num_threads=1))["snapshots"]
        == checked["snapshots"]
    )


# ---------------------------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------------------------


def _rows(store, observable, field=None, **cond):
    with opened(store) as s:
        return s.observations(observable, field=field, protocol="F", **cond)


def test_snapshot_means_equal_an_independent_nested_mean(store):
    out = small(sa.aggregate(store, "k", where={"n": 8, "p": ["0.1", "0.3"]}))
    assert [c["p"] for c in out["conditions"]] == ["0.1", "0.3"]
    for cond in out["conditions"]:
        rows = _rows(store, "k", n=8, beta="0.2", p=cond["p"])
        for i, t in enumerate(cond["depths"]):
            sel = rows["t"] == t
            mu, se, graphs = nested_reference(rows["g"][sel], rows["c"][sel], rows["value"][sel])
            assert cond["mean"][i] == pytest.approx(mu, rel=1e-5, abs=1e-9)
            assert cond["se"][i] == pytest.approx(se, rel=1e-5, abs=1e-9)
            assert cond["contributing_graphs"][i] == graphs == 3
        assert cond["trajectories"] == 6 and cond["status_counts"] == {
            "ok": 6 * len(cond["depths"])
        }
    with opened(store) as s:  # the store's own aggregate record agrees
        rec = s.aggregate("k", t=6, n=8, beta="0.2", p="0.3", record=False)
    i = out["conditions"][1]["depths"].index(6)
    assert out["conditions"][1]["mean"][i] == pytest.approx(rec["mean"], rel=1e-5)


def test_d_cyc_without_a_code_is_never_averaged_silently(store):
    where = {"n": 6, "p": "0.3"}
    with pytest.raises(ValueError, match="undefined_no_code.*zero_extended"):
        sa.aggregate(store, "d_cyc", where=where)
    zero = sa.aggregate_arrays(store, "d_cyc", where=where, field="zero_extended")
    omit = sa.aggregate_arrays(store, "d_cyc", where=where, nan_policy="omit")
    rows = _rows(store, "d_cyc", n=6, beta="0.2", p="0.3", field="zero_extended")
    cz, co = zero["conditions"][0], omit["conditions"][0]
    for i, t in enumerate(cz["depths"].tolist()):
        sel = rows["t"] == t
        mu, _, _ = nested_reference(rows["g"][sel], rows["c"][sel], rows["zero_extended"][sel])
        assert cz["mean"][i] == pytest.approx(mu)
        raw = np.where(rows["status"][sel] == "ok", rows["value"][sel], np.nan)
        if np.isfinite(raw).any():
            mu_c, _, graphs = nested_reference(rows["g"][sel], rows["c"][sel], raw)
            assert co["mean"][i] == pytest.approx(mu_c) and co["contributing_graphs"][i] == graphs
        else:
            assert math.isnan(co["mean"][i])
    assert np.any(co["contributing_trajectories"] < co["trajectories"])  # some codes are gone
    assert "defined values only" in omit["estimand"]


def test_paired_rates_are_differences_within_graphs(store):
    out = sa.aggregate_arrays(store, "k", where={"n": 6}, paired_rates=["0.1", "0.3"])
    (pair,) = out["paired"]
    a = _rows(store, "k", n=6, beta="0.2", p="0.1")
    b = _rows(store, "k", n=6, beta="0.2", p="0.3")
    for i, t in enumerate(pair["depths"].tolist()):
        fa = {
            g: nested_reference(
                a["g"][(a["t"] == t) & (a["g"] == g)],
                a["c"][(a["t"] == t) & (a["g"] == g)],
                a["value"][(a["t"] == t) & (a["g"] == g)],
            )[0]
            for g in range(3)
        }
        fb = {g: nested_reference(b["g"][(b["t"] == t) & (b["g"] == g)],
                                  b["c"][(b["t"] == t) & (b["g"] == g)],
                                  b["value"][(b["t"] == t) & (b["g"] == g)])[0] for g in range(3)}  # fmt: skip
        diffs = [fb[g] - fa[g] for g in range(3)]
        mu = sum(diffs) / 3
        se = math.sqrt(sum((d - mu) ** 2 for d in diffs) / 6)
        assert pair["difference"][i] == pytest.approx(mu, abs=1e-12)
        assert pair["se"][i] == pytest.approx(se, abs=1e-12)
    with pytest.raises(ValueError, match="two different rates"):
        sa.aggregate(store, "k", paired_rates=["0.1", "1/10"])


def test_first_passage_summaries_never_average_censored_times(store):
    out = sa.aggregate_arrays(store, "first_passage", where={"n": 6}, ell=[0, 1])
    saw_censored = saw_complete = False
    for cond in out["conditions"]:
        tau = _rows(store, "first_passage", "tau", n=6, beta="0.2", p=cond["p"])
        seen = _rows(store, "first_passage", "observed", n=6, beta="0.2", p=cond["p"])["observed"]
        horizon = 24.0
        for entry in cond["ells"]:
            ell = entry["ell"]
            ok = seen[:, ell].astype(bool)
            times = np.where(ok, tau["tau"][:, ell], horizon)
            assert entry["censored"] == int((~ok).sum())
            assert entry["censored_fraction"] == pytest.approx(nested_reference(
                tau["g"], tau["c"], (~ok).astype(float))[0])  # fmt: skip
            assert entry["median"] == km_median(times, ok)
            t_ref, s_ref = km_reference(times, ok)
            assert entry["_curve"].survival.tolist() == pytest.approx(s_ref)
            assert entry["restricted_mean"] == pytest.approx(
                nested_reference(tau["g"], tau["c"], np.minimum(times, horizon))[0]
            )
            assert entry["horizon"] == horizon
            if ok.all():
                saw_complete = True
                assert entry["mean"] == pytest.approx(
                    nested_reference(tau["g"], tau["c"], times)[0]
                )
            else:
                saw_censored = True
                assert entry["mean"] is None  # never a mean of censored times
    assert saw_censored and saw_complete  # the tiny study exercises both
    with pytest.raises(ValueError, match="ell does not apply"):
        sa.aggregate(store, "k", ell=1)
    with pytest.raises(ValueError, match="no parameters"):
        sa.aggregate(store, "first_passage", params={"x": 1})


def test_peak_summaries_state_their_horizon_policy(store):
    refused = sa.aggregate_arrays(store, "peak_d_cyc", where={"n": 6, "p": "0.1"})
    cond = refused["conditions"][0]
    assert cond["horizon_fraction"] > 0 and cond["arithmetic_mean"] is None
    assert "withheld" in cond["note"]
    kept = sa.aggregate_arrays(store, "peak_d_cyc", where={"n": 6, "p": "0.1"},
                               horizon_policy="include")["conditions"][0]  # fmt: skip
    rows = _rows(store, "peak_d_cyc", n=6, beta="0.2", p="0.1")
    mu, se, _ = nested_reference(rows["g"], rows["c"], rows["value"])
    assert kept["arithmetic_mean"] == pytest.approx(mu) and kept["arithmetic_se"] == pytest.approx(
        se
    )
    logs = nested_reference(rows["g"], rows["c"], np.log(rows["value"]))[0]
    assert kept["geometric_mean"] == pytest.approx(math.exp(logs))
    horizon = _rows(store, "peak_d_cyc", "at_horizon", n=6, beta="0.2", p="0.1")["at_horizon"]
    assert kept["horizon_fraction"] == pytest.approx(
        nested_reference(rows["g"], rows["c"], horizon)[0]
    )
    zero = _rows(store, "d_cyc", "zero_extended", n=6, beta="0.2", p="0.1")
    means = {t: nested_reference(zero["g"][zero["t"] == t], zero["c"][zero["t"] == t],
                                 zero["zero_extended"][zero["t"] == t])[0]
             for t in kept["domain"].tolist() if t > 0}  # fmt: skip
    best = max(means.values())
    assert kept["argmax"] == min(t for t, v in means.items() if v == best)
    excluded = sa.aggregate(store, "peak_d_cyc", where={"n": 6}, horizon_policy="exclude")
    assert excluded["horizon_policy"] == "exclude"


def test_peak_D_and_the_parameter_matching(store):
    out = small(sa.aggregate(store, "peak_D", where={"n": 8}, horizon_policy="include"))
    assert out["params"]["rule"] == "plateau" and len(out["conditions"]) == 3
    same = small(sa.aggregate(store, "peak_D", {"rule": "plateau"}, {"n": 8},
                              horizon_policy="include"))  # fmt: skip
    assert same["conditions"] == out["conditions"]
    with pytest.raises(ValueError, match="no stored values of 'peak_D' match"):
        sa.aggregate(store, "peak_D", {"rule": "other"})
    with pytest.raises(ValueError, match="holds no values of 'peak_D_J'"):
        sa.aggregate(store, "peak_D_J", {"J": 1})


def test_k_trace_is_the_mean_of_the_stored_traces(store):
    out = sa.aggregate_arrays(store, "k_trace", where={"n": 6, "p": "0.2"}, normalized=True)
    (cond,) = out["conditions"]
    with opened(store) as s:
        keys = s.trajectory_keys(n=6, beta="0.2", p="0.2")
        traces = np.array([s.k_trace(k) for k in keys]) / 6
        g = np.array([k.g for k in keys])
        c = np.array([k.c for k in keys])
    for t in (0, 5, 24):
        mu, se, _ = nested_reference(g, c, traces[:, t])
        assert cond["mean"][t] == pytest.approx(mu) and cond["se"][t] == pytest.approx(se)
    assert cond["depths"].tolist() == list(range(25))


def test_aggregate_results_are_capped_and_written_in_full(store, tmp_path):
    out = small(sa.aggregate(store, "k_trace", out=str(tmp_path / "agg")))
    full = json.loads((tmp_path / "agg" / "aggregate_k_trace.json").read_text(encoding="utf-8"))
    assert len(full["conditions"][-1]["depths"]) == 33  # n = 8: layers 0..32, all of them
    assert full["conditions"][-1]["mean"] == pytest.approx(out["conditions"][-1]["mean"], rel=1e-5)
    lines = (tmp_path / "agg" / "aggregate_k_trace.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "protocol,n,beta,p,t,mean,se" and len(lines) == 1 + 3 * 25 + 3 * 33
    # a long trace is shown at evenly spaced depths, both ends kept, and the note says so
    depths = np.arange(301)
    shown = sa._finish(sa._render({"conditions": [{"n": 8, "depths": depths, "mean": depths * 0.5,
                                        "se": depths * 0.0}], "notes": []}))  # fmt: skip
    cond = shown["conditions"][0]
    assert len(cond["depths"]) == api.MAX_CHECKPOINTS_SHOWN and cond["depths_total"] == 301
    assert cond["depths"][0] == 0 and cond["depths"][-1] == 300
    assert cond["mean"] == [0.5 * t for t in cond["depths"]]
    assert any("subsampled" in note for note in shown["notes"])


def test_aggregate_record_stores_one_record_per_depth(tmp_path):
    path = build(tmp_path / "s")
    out = sa.aggregate(path, "k", where={"n": 6, "p": "0.1"}, record=True)
    with opened(path) as s:
        records = s.aggregates("k")
    assert out["recorded"] == len(records) == len(out["conditions"][0]["depths"])
    by_t = {r["t"]: r for r in records}
    for i, t in enumerate(out["conditions"][0]["depths"]):
        assert by_t[t]["mean"] == pytest.approx(out["conditions"][0]["mean"][i], rel=1e-5)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"observable": "nope"}, "no observable 'nope'"),
        ({"observable": "k", "where": {"m": 1}}, "unknown key"),
        ({"observable": "k", "where": {"n": 99}}, "selects no condition"),
        ({"observable": "k", "where": []}, "where must be an object"),
        ({"observable": "uniform_failure"}, "stores a vector per value"),
        ({"observable": "uniform_failure", "field": "curve"}, "give index"),
        ({"observable": "k", "nan_policy": "drop"}, "nan_policy must be one of"),
        ({"observable": "peak_d_cyc", "horizon_policy": "maybe"}, "horizon_policy must be"),
        ({"observable": "first_passage", "where": {"t": 3}}, "where.t selects depths"),
        ({"observable": "k", "normalized": True}, "normalized does not apply"),
    ],
)
def test_aggregate_errors_say_what_to_change(store, kwargs, message):
    observable = kwargs.pop("observable")
    with pytest.raises(ValueError, match=message):
        sa.aggregate(store, observable, **kwargs)


# ---------------------------------------------------------------------------------------------
# fss
# ---------------------------------------------------------------------------------------------

FSS = {"starts": 1, "bootstrap": 3, "seed": 2}


def test_fss_report_lists_points_exclusions_and_settings(store, tmp_path):
    out = small(sa.fss(store, "first_passage", "0.2", ell=0, out=str(tmp_path / "f"), **FSS))
    report = json.loads((tmp_path / "f" / "fss_tau_0_beta0.2.json").read_text(encoding="utf-8"))
    assert report["schema"] == sa.FSS_REPORT_SCHEMA and "unverified" in report["hypothesis"]
    assert report["settings"]["bounds"]["z"] == [0.0, 3.0]
    assert report["sample_hierarchy"]["graphs"] == 3 and set(report["schedules"]) == {"6", "8"}
    (res,) = report["results"]
    tau = {}
    for pt in res["points"]:
        seen = _rows(store, "first_passage", "observed", n=pt["n"], beta="0.2", p=pt["p"])
        ok = seen["observed"][:, 0].astype(bool)
        if ok.all():  # used, and equal to the nested mean of tau_0
            rows = _rows(store, "first_passage", "tau", n=pt["n"], beta="0.2", p=pt["p"])
            mu, se, _ = nested_reference(rows["g"], rows["c"], rows["tau"][:, 0])
            assert pt["used"] and pt["y"] == pytest.approx(mu) and pt["se"] == pytest.approx(se)
        else:  # excluded, with the reason, never fitted on the observed ones
            assert not pt["used"] and pt["reason"].startswith("censored")
        tau[(pt["n"], pt["p"])] = pt["used"]
    assert out["results"][0]["points_excluded"] == sum(not u for u in tau.values()) > 0
    assert out["results"][0]["input_digest"] == res["input_digest"]


def test_fss_rerun_writes_identical_bytes(store, tmp_path):
    for name in ("a", "b"):
        sa.fss(store, "peak_d_cyc", "0.2", horizon_policy="include", max_horizon_fraction=1.0,
               out=str(tmp_path / name), **FSS)  # fmt: skip
    a = (tmp_path / "a" / "fss_Tstar_cyc_beta0.2.json").read_bytes()
    b = (tmp_path / "b" / "fss_Tstar_cyc_beta0.2.json").read_bytes()
    assert a == b
    report = json.loads(a)
    assert report["results"][0]["fit"]["status"] in ("ok", "at_bound", "not_converged")
    assert report["results"][0]["bootstrap"]["replicates"] == 3


def test_fss_censoring_and_peak_policies(store):
    strict = sa.fss_report(store, "peak_d_cyc", "0.2", leave_one_out=False, **FSS)
    reasons = [pt["reason"] for pt in strict["results"][0]["points"] if not pt["used"]]
    assert reasons and all("horizon" in r or "missing" in r for r in reasons)
    strict_tau = sa.fss_report(store, "first_passage", "0.2", ell=0, leave_one_out=False, **FSS)
    lower = sa.fss_report(store, "first_passage", "0.2", ell=0, censoring="restricted_mean",
                          max_censored_fraction=1.0, leave_one_out=False, **FSS)  # fmt: skip
    unused = [pt["reason"] for pt in lower["results"][0]["points"] if not pt["used"]]
    assert all(r.startswith("zero standard error") for r in unused)  # every graph censored
    assert sum(pt["used"] for pt in lower["results"][0]["points"]) > sum(
        pt["used"] for pt in strict_tau["results"][0]["points"]
    )
    assert "lower bound" in lower["results"][0]["estimator"]
    with pytest.raises(ValueError, match="max_censored_fraction applies with censoring"):
        sa.fss(store, "first_passage", "0.2", ell=0, max_censored_fraction=0.1)


def test_fss_shared_test_and_its_wording(store):
    report = sa.fss_report(store, [{"id": "peak_d_cyc"}, {"id": "peak_D"}], "0.2",
                           horizon_policy="include", max_horizon_fraction=1.0,
                           leave_one_out=False, **FSS)  # fmt: skip
    verdict = report["shared"]["verdict"]
    assert verdict.startswith(("consistent with", "shared (p_c, z, nu) rejected", "undetermined"))
    assert "confirmed" not in verdict


@pytest.mark.parametrize(
    "args, kwargs, message",
    [
        (("k", "0.2"), {}, "fss fits first_passage"),
        (("first_passage", "0.2"), {}, "exactly one of ell"),
        (("peak_D_J", "0.2"), {}, "needs J"),
        (("peak_d_cyc", "0.7"), {}, "beta = 0.7 is not a condition"),
        (("peak_d_cyc", "0.2"), {"method": "other"}, "method must be one of"),
        (("peak_d_cyc", "0.2"), {"bounds": {"nu": [0, 1]}}, "bounds.nu must be positive"),
        (("peak_d_cyc", "0.2"), {"sizes": [7]}, "sizes \\[7\\] are not run"),
        (([{"id": "peak_d_cyc"}], "0.2"), {}, "needs at least two"),
    ],
)
def test_fss_errors_say_what_to_change(store, args, kwargs, message):
    with pytest.raises(ValueError, match=message):
        sa.fss(store, *args, **kwargs)


def test_fss_refuses_a_report_beyond_max_seconds(store):
    with pytest.raises(ValueError, match="above max_seconds"):
        sa.fss(store, "peak_d_cyc", "0.2", horizon_policy="include", max_horizon_fraction=1.0,
               starts=1, bootstrap=10_000, max_seconds=1e-6)  # fmt: skip


# ---------------------------------------------------------------------------------------------
# reading while a run writes
# ---------------------------------------------------------------------------------------------


def test_reading_functions_work_while_a_writer_holds_the_store(tmp_path):
    path = build(tmp_path / "s")
    with TableauStore.open(path, write=True):
        assert sa.status(path)["complete"]
        assert sa.aggregate(path, "k", where={"n": 6, "p": "0.1"})["conditions"]
        with pytest.raises(ValueError, match="is writing to"):
            sa.compute(path, "k")
    assert threading.active_count() >= 1
