"""The reproducibility invariants of the store and the runner (v1.4 §7.3, §7.4, validation 5-9,
§9.2 (c), (d), (g), (h); design §4, D3).

Stores are compared by their logical digest (``TableauStore.logical_digest``: every record by
its natural key, blobs decoded; row ids, unit numbers, timestamps and compression left out),
never by file bytes, and record by record where a later operation must leave earlier records
unchanged (``_study_store_support.records``).
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

import _study_store_support as sup
from sparsegf2 import mipt
from sparsegf2.study import observables as ob
from sparsegf2.study import runner
from sparsegf2.study.store import TableauStore

PAIRED = {"protocols": {"F": {}, "P": {"checkpoints": {"list": [4]}}}}


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """The uninterrupted store: default units and threads, F and P."""
    store = sup.build(tmp_path_factory.mktemp("reference") / "store", sup.manifest(**PAIRED))
    digest = store.logical_digest()
    yield store, digest
    store.close()


def _sanity(store: TableauStore) -> None:
    assert store.check() == []
    assert store._db().execute("PRAGMA integrity_check").fetchall() == [("ok",)]


# ---------------------------------------------------------------------------------------------
# unit splits, threads, order, resume splits
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "how",
    ["one graph per unit, one thread", "one trajectory per unit, reversed", "resume splits"],
)
def test_how_the_work_is_split_and_run_never_changes_the_content(tmp_path, reference, how):
    ref_store, digest = reference
    m = sup.manifest(**PAIRED)
    store = sup.build(tmp_path / "s", m, run=False)
    try:
        if how == "one graph per unit, one thread":
            runner.run(store, units=runner.plan(store, graphs_per_unit=1), num_threads=1)
        elif how == "one trajectory per unit, reversed":
            units = runner.plan(store, max_unit_trajectories=1)
            assert len(units) == 24
            runner.run(store, units=list(reversed(units)), num_threads=2)
        else:
            for count in (1, 2, 1, 3):  # reopen between parts, as a resumed campaign does
                runner.run(store, max_units=count)
                store.close()
                store = TableauStore.open(tmp_path / "s", write=True)
            runner.run(store)
        assert runner.plan(store) == []
        assert store.logical_digest() == digest
        assert sup.records(store) == sup.records(ref_store)
        _sanity(store)
    finally:
        store.close()


def test_the_logical_digest_sees_every_change_of_content(tmp_path, reference):
    _, digest = reference
    other = sup.manifest(**PAIRED, seeds={"master_graph_seed": 101, "master_circuit_seed": 203})
    with sup.build(tmp_path / "s", other) as store:
        assert store.logical_digest(manifests=False) != digest
    with sup.build(tmp_path / "t", sup.manifest(**PAIRED)) as store:
        assert store.logical_digest() == digest
        seen = {digest}
        for change in (
            "UPDATE observations SET se = se + 1 WHERE rowid = 7",
            "UPDATE observations SET value = value + 1 WHERE rowid = 8 AND value IS NOT NULL",
            "UPDATE snapshots SET k = k + 1 WHERE rowid = 3",
            "UPDATE traces SET final_layer = final_layer - 1 WHERE rowid = 2",
            "UPDATE graphs SET measures = '{}' WHERE g = 1",
        ):
            con = sup.open_for_tampering(store)
            assert con.execute(change).rowcount == 1, change
            con.commit()
            con.close()
            now = store.logical_digest()
            assert now not in seen, change
            seen.add(now)


# ---------------------------------------------------------------------------------------------
# interruption
# ---------------------------------------------------------------------------------------------


def test_an_exception_before_a_commit_leaves_the_store_as_it_was(tmp_path, reference, monkeypatch):
    _, digest = reference
    store = sup.build(tmp_path / "s", sup.manifest(**PAIRED), run=False)
    try:
        calls = []

        def after_the_engine_before_the_commit(unit):
            calls.append(unit)
            if len(calls) == 3:
                raise KeyError("the third unit dies after its inserts")

        monkeypatch.setattr(runner, "_BEFORE_COMMIT", after_the_engine_before_the_commit)
        units = runner.plan(store)
        with pytest.raises(KeyError, match="third unit"):
            runner.run(store)
        done = units[:2]
        assert store.counts()["trajectories"] == sum(u.trajectories for u in done)
        stored = {(k.p.token, k.g, k.c) for k in store.trajectory_keys()}
        assert stored == {(u.p.token, g, c) for u in done for g, c in u.pairs()}
        assert [u["status"] for u in store.units()] == ["done", "done", "failed"]
        _sanity(store)
        monkeypatch.setattr(runner, "_BEFORE_COMMIT", None)
        assert runner.plan(store) == units[2:]  # the unit is redone on resume
        runner.run(store)
        assert store.logical_digest() == digest
    finally:
        store.close()


def test_a_process_killed_inside_a_unit_transaction_leaves_a_consistent_store(tmp_path, reference):
    _, digest = reference
    sup.build(tmp_path / "s", sup.manifest(**PAIRED), run=False).close()
    script = (
        "import os\n"
        "from sparsegf2.study import runner\n"
        "from sparsegf2.study.store import TableauStore\n"
        f"store = TableauStore.open({str(tmp_path / 's')!r}, write=True)\n"
        "seen = []\n"
        "def kill(unit):\n"
        "    seen.append(unit)\n"
        "    if len(seen) == 3:\n"
        "        os._exit(9)  # the process dies inside the transaction: no rollback, no cleanup\n"
        "runner._BEFORE_COMMIT = kill\n"
        "runner.run(store)\n"
    )
    proc = sup.run_python(script)
    assert proc.returncode == 9, proc.stderr
    assert (tmp_path / "s" / ".lock").exists()  # the dead writer's lock was left behind
    with TableauStore.open(tmp_path / "s") as reader:  # readers see the two committed units
        assert reader.counts()["units"] == 2 and reader.counts()["trajectories"] == 6
    store = TableauStore.open(tmp_path / "s", write=True)  # the stale lock is taken over
    try:
        _sanity(store)
        assert [u["status"] for u in store.units()] == ["done", "done"]  # nothing of unit 3
        assert len(runner.plan(store)) == len(runner.plan(store, graphs_per_unit=2)) == 6
        runner.run(store)
        assert store.logical_digest() == digest
    finally:
        store.close()


# ---------------------------------------------------------------------------------------------
# extension (design D3): new conditions, graphs, circuits, rates, sizes, protocols, horizons
# ---------------------------------------------------------------------------------------------

FIRST = dict(
    observables={**sup.BASE["observables"], "uniform_erasure": {"samples": 12, "exact_budget": 8}},
)
SECOND = dict(
    FIRST,
    conditions=[
        {"n": [6], "beta": ["0.25"], "p": ["0.1", "0.2", "0.3"]},  # a rate
        {"n": [8], "beta": ["0.25", "0.5"], "p": ["0.1"]},  # a size, a beta
    ],
    trials={"graphs": 4, "circuits_per_graph": 3, "circuit_index_stride": 4},  # graphs, circuits
    depth={"layers": {"fixed": 14}, "checkpoints": {"list": [2, 5, 10, 12]}},  # deeper, new depth
    protocols=["F", "P"],
)


def test_an_extension_leaves_every_record_unchanged_and_equals_a_fresh_run(tmp_path):
    first, second = sup.manifest(**FIRST), sup.manifest(**SECOND)
    with sup.build(tmp_path / "extended", first) as store:
        before = sup.records(store)
        units = runner.extend(store, second)
        # every old trajectory reruns (deeper horizon, a new depth, protocol P is new)
        old = {(k.p.token, k.g, k.c) for k in store.trajectory_keys()}
        planned = {
            (u.p.token, g, c) for u in units if u.protocol == "F" and u.n == 6 for g, c in u.pairs()
        }
        assert old < planned
        runner.run(store)
        after = sup.records(store)
        sup.assert_unchanged(before, after)  # traces at T = 10, snapshots, values: all kept
        key = store.trajectory_keys(protocol="F", n=6, p="0.1")[0]
        assert [t.layers for t in (store.trace(key, layers=10), store.trace(key))] == [10, 14]
        assert np.array_equal(store.k_trace(key)[:11], store.k_trace(key, layers=10))
        assert store.depths(key) == (0, 1, 2, 5, 10, 12, 14)
        _sanity(store)
        with sup.build(tmp_path / "fresh", second) as fresh:
            assert store.logical_digest(history=False) == fresh.logical_digest(history=False)
            assert store.logical_digest() != fresh.logical_digest()  # the history differs


def test_a_deeper_horizon_fails_loudly_on_a_stored_prefix_that_does_not_reproduce(tmp_path):
    with sup.build(tmp_path / "s", sup.manifest()) as store:
        key = store.trajectory_keys()[2]
        tid = store._tid(key)
        k = store.k_trace(key)
        forged = k.copy()
        forged[1:] = np.minimum(forged[1:], k[0] - 1)  # still a valid trace, another one
        assert not np.array_equal(forged, k)
        con = sup.open_for_tampering(store)
        points = ob.k_change_points(forged).astype("<i4").tobytes()
        tau = ob.first_passage(forged).tau.astype("<i4").tobytes()
        con.execute(
            "UPDATE traces SET k_points = ?, tau = ? WHERE tid = ?",
            (zlib.compress(points), zlib.compress(tau), tid),
        )
        con.commit()
        con.close()
        before = sup.records(store)
        runner.extend(
            store,
            sup.manifest(depth={"layers": {"fixed": 13}, "checkpoints": {"list": [2, 5, 10]}}),
        )
        with pytest.raises(ValueError, match=r"disagree on layers 0\.\.10.*did not reproduce"):
            runner.run(store)
        after = sup.records(store)
        assert {k_: v for k_, v in after.items() if k_[0] != "manifest"} == {
            k_: v for k_, v in before.items() if k_[0] != "manifest"
        }
        assert store.units(status="failed")[-1]["error"].startswith("ValueError: trace of")


def test_a_checkpoint_rule_change_cannot_change_a_circuit_or_a_stored_value(tmp_path):
    with sup.build(tmp_path / "s", sup.manifest()) as store:
        before = sup.records(store)
        circuits = {k: str(runner.circuit(store, k)) for k in store.trajectory_keys()}
        runner.extend(
            store, sup.manifest(depth={"layers": {"fixed": 10}, "checkpoints": {"list": [3, 7]}})
        )
        runner.run(store)  # reruns every trajectory with depths 3 and 7 added, verifying the rest
        sup.assert_unchanged(before, sup.records(store))
        for key, text in circuits.items():
            assert str(runner.circuit(store, key)) == text
            assert store.depths(key) == (0, 1, 2, 3, 5, 7, 10)
            fresh = runner.replay(store, key, [3, 7], verify=True)
            assert fresh.verified_depths == (3, 7)


def test_engine_probes_and_schedules_never_change_the_trajectory(tmp_path):
    """The prefix and checkpoint invariance the store relies on, at the engine (v1.4 §7.3)."""
    with sup.build(tmp_path / "s", sup.manifest()) as store:
        key = store.trajectory_keys()[5]
        a = runner.replay(store, key, [5], verify=True)
        b = runner.replay(store, key, [1, 2, 5, 9, 25], verify=True)
        assert a.tableaux[5] == b.tableaux[5] == store.tableau(key, 5)
        assert np.array_equal(a.k[:6], b.k[:6])


def test_recomputing_an_observable_draws_no_circuit_randomness(tmp_path, monkeypatch):
    with sup.build(tmp_path / "s", sup.manifest()) as store:
        before = sup.records(store)
        params = {"m": 2, "J": 1.0, "samples": 6}
        monkeypatch.setattr(mipt.MonitoredCircuit, "run", _refuse)
        first = runner.compute(store, "mean_leakage", params)
        assert first.inserted == 60
        monkeypatch.undo()
        sup.assert_unchanged(before, sup.records(store))
        values = store.observations("mean_leakage")
        # the seeds come from the trajectory key (region_seed): another store, computed in
        # another order, draws the same regions and gives the same values
        with sup.build(tmp_path / "t", sup.manifest()) as other:
            runner.compute(other, "half_cut_entropy")
            runner.compute(other, "mean_leakage", params, batch_size=5)
            again = other.observations("mean_leakage")
            assert np.array_equal(values["value"], again["value"], equal_nan=True)
            assert np.array_equal(values["se"], again["se"], equal_nan=True)
        for key in store.trajectory_keys()[:3]:  # and the trajectories still replay exactly
            assert runner.replay(
                store, key, list(store.depths(key))
            ).verified_depths == store.depths(key)


def _refuse(*args, **kwargs):
    raise AssertionError("a retrospective observable must not simulate")
