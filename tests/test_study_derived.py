"""The trajectory-level values of a store: derived from stored data, keyed by their context.

A revision that changes only what is derived (the peak rule, a schedule whose depths are all
stored) runs no circuit: :func:`runner.derive` computes the new values from the stored traces
and values, the same values a store made directly under the new manifest holds (T3-F01). The
current values are those of the latest revision per condition, after any chain of revisions
(T3-F03); aggregates are keyed by their context and their stored parameters (T3-F04, T3-F05);
``peak_D_J`` is keyed by its probe (T3-F06); the current-state digest equals a fresh store's
whatever the earlier schedules (T3-F13, T3-F18); the checkpoint bounds retrieved by default
are over the current domain (T3-F21).
"""

from __future__ import annotations

import numpy as np
import pytest

import _study_store_support as sup
from sparsegf2 import mipt
from sparsegf2.study import observables as ob
from sparsegf2.study import runner
from sparsegf2.study.store import TableauStore

UNIFORM = {"samples": 12, "exact_budget": 6}


def _spy(monkeypatch) -> list[object]:
    """Count the engine calls (``MonitoredCircuit.run``) made after this point."""
    calls: list[object] = []
    original = mipt.MonitoredCircuit.run

    def counted(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(mipt.MonitoredCircuit, "run", counted)
    return calls


def _current_rows(store: TableauStore) -> dict[tuple, tuple]:
    """The current trajectory-level rows of a store, by trajectory and key (no context: the
    contexts of two stores under the same latest revision are equal by construction)."""
    out = {}
    for key in store.trajectory_keys(protocol="F"):
        cur = store._current("F", key.n, key.beta)
        for name, version, params in cur.derived:
            row = store.observation(key, -1, name, version=version, params=params)
            out[(key.p.token, key.g, key.c, name, params)] = (
                row["value"],
                row["se"],
                row["status"],
                row["method"],
                repr(row["data"]),
            )
    return out


def _same_rows(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        va, vb = a[k], b[k]
        if va[2:] != vb[2:] or not np.array_equal(va[:2], vb[:2], equal_nan=True):
            return False
    return True


# ---------------------------------------------------------------------------------------------
# T3-F01: a revision of what is derived runs no circuit
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"analysis": {"peak": {"z": 2.0, "bootstrap": 5}}},  # the plateau width
        {"analysis": {"peak": {"bootstrap": 0}}},  # the bootstrap of T* off
        {"depth": {"layers": {"fixed": 10}, "checkpoints": {"list": [2]}}},  # S_r narrower
    ],
    ids=["z", "bootstrap", "domain"],
)
def test_a_revision_of_what_is_derived_runs_no_circuit(tmp_path, monkeypatch, change):
    first, second = sup.manifest(), sup.manifest(**change)
    with sup.build(tmp_path / "s", first) as store:
        before = sup.records(store)
        calls = _spy(monkeypatch)
        assert runner.extend(store, second) == []  # nothing to simulate
        assert store.status()["derivations_remaining"] == 12
        summary = runner.run(store)
        assert calls == [] and summary.units == 0 and summary.derived == 12
        assert store.status()["derivations_remaining"] == 0
        sup.assert_unchanged(before, sup.records(store))  # nothing stored before changes
        monkeypatch.undo()
        with sup.build(tmp_path / "fresh", second) as fresh:
            # the derived values equal those a run under the new manifest stores
            assert _same_rows(_current_rows(store), _current_rows(fresh))
            assert store.logical_digest(history=False, manifests=False) == fresh.logical_digest(
                history=False, manifests=False
            )
        assert runner.verify(store, sample=4).ok


def test_derive_from_stored_data_gives_the_rows_the_run_stored(tmp_path):
    """The values of a unit and the values derived later from the store are one function of
    the same stored data (_derive_rows): re-deriving every stored trajectory gives equal rows."""
    with sup.build(tmp_path / "s", sup.manifest()) as store:
        level = store.manifest.storage.compression_level
        for key in store.trajectory_keys():
            cur = store._current("F", key.n, key.beta)
            layers, domain = cur.derivation["layers"], cur.derivation["domain"]
            needs = {name for name, _, _ in cur.derived}
            src = runner._stored_sources(
                store, store._require_tid(key), key, cur.requirements, layers, domain, needs
            )
            wanted = [(*k3, p) for k3, p in zip(cur.derived, cur.derived_params, strict=True)]
            for row in runner._derive_rows(key, layers, domain, cur.context, wanted, src, level):
                stored = store.observation(
                    key, -1, row.observable, version=row.version, params=row.params
                )
                assert stored["status"] == row.status
                assert np.array_equal(
                    [stored["value"], stored["se"]],
                    [np.nan if row.value is None else row.value,
                     np.nan if row.se is None else row.se],
                    equal_nan=True,
                )  # fmt: skip
        summary = runner.derive(store)
        assert summary.trajectories == 0  # everything is derived already


# ---------------------------------------------------------------------------------------------
# T3-F03: the current values after any chain of revisions
# ---------------------------------------------------------------------------------------------


def test_a_revision_that_returns_to_an_earlier_rule_reads_the_current_values(tmp_path):
    rule_1 = {"peak": {"z": 1.0, "bootstrap": 5}}
    rule_2 = {"peak": {"z": 2.0, "bootstrap": 5}}
    a = sup.manifest(analysis=rule_1)
    b = sup.manifest(analysis=rule_2)
    back = sup.manifest(analysis=rule_1, storage={"graphs_per_unit": 3})  # returns to rule 1
    params_1 = ob.params_key("peak_D", {**rule_1["peak"]})
    with sup.build(tmp_path / "s", a) as store:
        for m in (b, back):
            runner.extend(store, m)
            runner.run(store)
        key = store.trajectory_keys()[0]
        cur = runner.current(store.revisions(), "F", 6, "0.25")
        assert store.schedule(key)["requirements"] == cur.digest
        assert store.schedule(key)["context"] == cur.context
        rows = store.observations("peak_D", params=params_1, p="0.1")
        assert rows["g"].size == 6 and set(rows["context"]) == {cur.context}
        # the current choice of parameters is found without naming it
        assert np.array_equal(store.observations("peak_D", p="0.1")["value"], rows["value"],
                              equal_nan=True)  # fmt: skip
        store.observation(key, -1, "peak_D", params=params_1)
        store.aggregate("peak_D", t=-1, n=6, beta="0.25", p="0.1", nan_policy="omit")
        with sup.build(tmp_path / "fresh", back) as fresh:
            assert store.logical_digest(history=False, manifests=False) == fresh.logical_digest(
                history=False, manifests=False
            )


def test_an_unfinished_extension_is_not_aggregated_over_two_horizons(tmp_path):
    with sup.build(tmp_path / "s", sup.manifest(storage={"graphs_per_unit": 1})) as store:
        deeper = sup.manifest(
            storage={"graphs_per_unit": 1},
            depth={"layers": {"fixed": 30}, "checkpoints": {"list": [2, 5]}},
        )
        runner.extend(store, deeper)
        runner.run(store, max_units=1)  # one graph of one rate rerun to 30
        rows = store.observations("peak_d_cyc", p="0.1")
        assert rows["g"].size == 2  # the current context holds the rerun trajectories only
        with pytest.raises(ValueError, match="pending"):
            store.aggregate("peak_d_cyc", t=-1, n=6, beta="0.25", p="0.1", nan_policy="omit")
        assert store.aggregates() == []  # nothing was recorded
        partial = store.aggregate("peak_d_cyc", t=-1, n=6, beta="0.25", p="0.1",
                                  nan_policy="omit", partial=True, record=False)  # fmt: skip
        assert partial["trajectories"] == 2
        runner.run(store)
        whole = store.aggregate("peak_d_cyc", t=-1, n=6, beta="0.25", p="0.1", nan_policy="omit")
        assert whole["trajectories"] == 6
        horizons = store.observations("peak_d_cyc", p="0.1", field="horizon")["horizon"]
        assert set(horizons.tolist()) == {30.0}


# ---------------------------------------------------------------------------------------------
# T3-F04, T3-F05: the keys of an aggregate record
# ---------------------------------------------------------------------------------------------


def test_aggregates_of_trajectory_values_are_kept_per_horizon(tmp_path):
    with sup.build(tmp_path / "s", sup.manifest()) as store:
        args = dict(t=-1, n=6, beta="0.25", p="0.1", nan_policy="omit")
        before = store.aggregate("peak_d_cyc", **args)
        tau = store.aggregate("first_passage", field="tau", index=3, **args)
        runner.extend(store, sup.manifest(depth={"layers": {"fixed": 40},
                                                 "checkpoints": {"list": [2, 5]}}))  # fmt: skip
        runner.run(store)
        after = store.aggregate("peak_d_cyc", **args)
        tau_after = store.aggregate("first_passage", field="tau", index=3, **args)
        assert before["sample_set"] == after["sample_set"]  # the same trajectories
        assert before["context"] != after["context"]  # at another horizon: another record
        assert tau["context"] != tau_after["context"]
        recorded = store.aggregates("peak_d_cyc")
        assert {r["context"] for r in recorded} == {before["context"], after["context"]}
        assert store.aggregate("peak_d_cyc", **args) == after  # compared, equal
        snapshot = store.aggregate("D", t=5, n=6, beta="0.25", p="0.1")
        assert snapshot["context"] == ""


def test_an_aggregate_is_recorded_under_the_parameters_of_its_values(tmp_path):
    """BASE stores D with 12 orderings and exact budget 6, not the catalog's defaults."""
    with sup.build(tmp_path / "s", sup.manifest()) as store:
        record = store.aggregate("D", t=5, n=6, beta="0.25", p="0.3")  # params left out
        stored = ob.params_key("D", UNIFORM)
        assert stored != ob.params_key("D", None)
        assert record["params"] == stored and record["version"] == ob.get("D").version
        assert [r["params"] for r in store.aggregates("D")] == [stored]
        again = store.aggregate("D", t=5, n=6, beta="0.25", p="0.3", params=UNIFORM)
        assert again == record and len(store.aggregates("D")) == 1  # one record, one label
        peak = store.aggregate("peak_D", t=-1, n=6, beta="0.25", p="0.3", nan_policy="omit")
        assert peak["params"] == ob.params_key(
            "peak_D", runner._peak_params("peak_D", store.manifest.analysis.peak.to_dict())
        )


# ---------------------------------------------------------------------------------------------
# T3-F06: two correlated probes at one coupling
# ---------------------------------------------------------------------------------------------


def test_two_correlated_probes_sharing_a_coupling_are_two_observables(tmp_path):
    observables = {
        **sup.BASE["observables"],
        "correlated_erasure": [
            {"couplings": [0, 1.5], "sizes": [1, 3], "samples": 8, "name": "ring"},
            {"couplings": [1.5], "sizes": [1, 3], "samples": 8, "geometry": "cycle",
             "name": "cyc"},
        ],
    }  # fmt: skip
    with sup.build(tmp_path / "s", sup.manifest(observables=observables)) as store:
        rule = store.manifest.analysis.peak.to_dict()
        values = {}
        for name in ("ring", "cyc"):
            params = runner._peak_params("peak_D_J", rule, J=1.5, name=name)
            rows = store.observations("peak_D_J", params=params, p="0.3")
            assert rows["g"].size == 6  # one row per trajectory, of this probe only
            key = store.trajectory_keys(p="0.3")[0]
            assert store.observation(key, -1, "peak_D_J", params=params)["data"]["probe"] == name
            store.aggregate("peak_D_J", t=-1, n=6, beta="0.25", p="0.3", params=params,
                            nan_policy="omit")  # fmt: skip
            values[name] = rows["value"]
        assert not np.array_equal(values["ring"], values["cyc"], equal_nan=True)
        with pytest.raises(ValueError, match="name"):
            store.observations("peak_D_J", params={**rule, "J": 1.5}, p="0.3")


# ---------------------------------------------------------------------------------------------
# T3-F13, T3-F18: the current-state digest
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first, second",
    [
        # the geometric rule: raising T moves the depths (the schedules do not nest)
        ({"layers": {"fixed": 10}, "checkpoints": {"count": 4}},
         {"layers": {"fixed": 14}, "checkpoints": {"count": 4}}),
        ({"layers": {"fixed": 10}, "checkpoints": {"list": [2, 5]}},
         {"layers": {"fixed": 14}, "checkpoints": {"list": [2, 5, 12]}}),
    ],
    ids=["geometric", "lists"],
)  # fmt: skip
def test_the_current_state_of_an_extended_store_is_a_fresh_store(tmp_path, first, second):
    with sup.build(tmp_path / "s", sup.manifest(depth=first)) as store:
        runner.extend(store, sup.manifest(depth=second))
        runner.run(store)
        key = store.trajectory_keys()[0]
        with sup.build(tmp_path / "fresh", sup.manifest(depth=second)) as fresh:
            assert set(fresh.depths(key)) < set(store.depths(key))  # E_r holds more depths
            assert store.logical_digest(history=False, manifests=False) == fresh.logical_digest(
                history=False, manifests=False
            )
            assert store.logical_digest(history=False) == fresh.logical_digest(history=False)
            assert store.logical_digest() != fresh.logical_digest()


def test_recorded_replays_are_history_not_the_current_state(tmp_path):
    with sup.build(tmp_path / "a", sup.manifest()) as a, sup.build(tmp_path / "b") as b:
        assert a.logical_digest() == b.logical_digest()
        runner.replay_signed(a, a.trajectory_keys()[0], [5], record=True)
        assert a.logical_digest(history=False) == b.logical_digest(history=False)
        assert a.logical_digest() != b.logical_digest()


# ---------------------------------------------------------------------------------------------
# T3-F21: checkpoint bounds by default over the current domain
# ---------------------------------------------------------------------------------------------


def test_retrieved_checkpoint_bounds_are_those_of_the_stored_record(tmp_path):
    narrower = sup.manifest(depth={"layers": {"fixed": 10}, "checkpoints": {"list": [2]}})
    with sup.build(tmp_path / "s", sup.manifest()) as store:
        runner.extend(store, narrower)
        runner.run(store)
        with sup.build(tmp_path / "fresh", narrower) as fresh:
            differs = 0
            for key in store.trajectory_keys():
                stored = store.observation(key, -1, "first_passage")["data"]["tau_chk"]
                assert store.first_passage(key).to_dict()["tau_chk"] == stored
                assert fresh.first_passage(key).to_dict()["tau_chk"] == stored
                union = store.first_passage(key, checkpoints="stored").to_dict()["tau_chk"]
                differs += union != stored  # depth 5 is stored but not in S_r
            assert differs > 0
