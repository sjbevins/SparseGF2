"""Integrity of a store's records and of what is read from them.

* ``runner.verify`` runs trajectories again and compares every stored record of them: the
  trace, the snapshots, every observation (values, errors, statuses, methods, data), the
  derived values under their own context and the retrospective values; a store whose records
  agree with their own digests but do not reproduce fails (T3-F07, T3-F15).
* A rerun compares both snapshot digests (T3-F14).
* Code observables after a stop get their exact ``k = 0`` values (T3-F08).
* The export says why a value is missing, keeps the current trajectory-level rows apart from
  the history, and reads one committed state (T3-F10, T3-F11); so do the logical digest and
  ``check``.
* The snapshot cursor's plan does not sort and its cost per snapshot does not grow with the
  store (T3-F12).
* A revision repeating an earlier manifest and a creation that never completed are refused
  with messages that say what to do (T3-F19); a unit that fails in the engine is recorded
  (T3-F23).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
import zlib
from pathlib import Path

import numpy as np
import pytest

import _study_store_support as sup
from sparsegf2 import mipt
from sparsegf2.study import observables as ob
from sparsegf2.study import runner
from sparsegf2.study import store as store_module
from sparsegf2.study.store import STORE_FILE, TableauStore
from sparsegf2.tableau import PhaseFreeTableau

UNIFORM = {"samples": 12, "exact_budget": 6}


def _tamper(store: TableauStore, sql: str, args: tuple = ()) -> None:
    con = sup.open_for_tampering(store)
    assert con.execute(sql, args).rowcount >= 1, sql
    con.commit()
    con.close()


def _blob(data: dict) -> bytes:
    return zlib.compress(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())


def _permuted(tab: PhaseFreeTableau, swap: tuple[int, int]) -> bytes:
    """The canonical bytes (written by the test's own elimination) of ``tab`` with two qubits
    exchanged: another state with the same rank."""
    num = tab.num_qubits
    order = list(range(num))
    a, b = swap
    order[a], order[b] = order[b], order[a]
    dense = tab.canonical()
    return sup.canonical_bytes(dense[:, order + [num + q for q in order]], num)


def _forge(store: TableauStore, key, t: int, column: str, swap: tuple[int, int]) -> bool:
    """Replace the ``joint`` or ``phys`` row space of a snapshot by its image under a qubit
    exchange, with a matching digest: a record that agrees with itself. False if unchanged."""
    snap = store.snapshot(key, t)
    tab = snap.tableau() if column == "joint" else snap.physical_tableau()
    raw = _permuted(tab, swap)
    digest = hashlib.sha256(raw).hexdigest()
    if digest == getattr(snap, f"{column}_digest"):
        return False
    _tamper(
        store,
        f"UPDATE snapshots SET {column} = ?, {column}_digest = ? WHERE tid = ? AND t = ?",
        (zlib.compress(raw), digest, store._tid(key), t),
    )
    return True


# ---------------------------------------------------------------------------------------------
# T3-F07, T3-F15: verify reruns and compares every record
# ---------------------------------------------------------------------------------------------


def test_verify_compares_every_stored_record_of_a_clean_store(tmp_path):
    with sup.build(tmp_path / "s") as store:
        runner.compute(store, "half_cut_entropy")
        rows = store.counts()["observations"]
        result = runner.verify(store, sample=None)
        assert result.ok and result.trajectories == 12 and result.snapshots == 60
        assert result.observations == rows  # every stored value was recomputed and compared
    with TableauStore.open(tmp_path / "s") as reader:  # reading only
        assert runner.verify(reader, sample=3).ok


def _value(store, name, t, params=None):
    tid = store._tid(store.trajectory_keys()[1])
    text = ob.params_key(name, params)
    return (
        f"UPDATE observations SET value = value + 1 WHERE tid = {tid} AND t = {t} AND "
        f"observable = '{name}' AND params = '{text}'"
    )


@pytest.mark.parametrize(
    "what, change",
    [
        ("'D'", lambda s: _value(s, "D", 5, UNIFORM)),
        ("'k'", lambda s: _value(s, "k", 2)),
        ("'peak_D'", lambda s: _value(s, "peak_D", -1, runner._peak_params(
            "peak_D", s.manifest.analysis.peak.to_dict()))),
        ("'first_passage'", lambda s: (
            "UPDATE observations SET data = ? WHERE observable = 'first_passage' AND rowid = "
            "(SELECT min(rowid) FROM observations WHERE observable = 'first_passage')",
            (_blob({"tampered": True}),))),
        ("'d_cyc'", lambda s: (
            "UPDATE observations SET data = ? WHERE observable = 'd_cyc' AND rowid = "
            "(SELECT min(rowid) FROM observations WHERE observable = 'd_cyc' AND t = 5)",
            (_blob({"has_code": True, "zero_extended": 99}),))),
        ("'uniform_failure'", lambda s: (
            "UPDATE observations SET method = 'other' WHERE rowid = (SELECT min(rowid) FROM "
            "observations WHERE observable = 'uniform_failure')", ())),
        ("'D_J'", lambda s: (
            "UPDATE observations SET status = 'undefined' WHERE rowid = (SELECT min(rowid) FROM "
            "observations WHERE observable = 'D_J')", ())),
        ("'half_cut_entropy'", lambda s: _value(s, "half_cut_entropy", 5)),
    ],
    ids=["D", "k", "peak_D", "first_passage", "d_cyc", "uniform_failure", "D_J", "retrospective"],
)  # fmt: skip
def test_verify_finds_an_altered_value(tmp_path, what, change):
    with sup.build(tmp_path / "s") as store:
        runner.compute(store, "half_cut_entropy")
        assert runner.verify(store, sample=None).ok
        sql = change(store)
        _tamper(store, *(sql if isinstance(sql, tuple) else (sql,)))
        assert store.check() == []  # the structure and every digest are still sound
        result = runner.verify(store, sample=None)
        assert not result.ok and len(result.failures) == 1
        assert what in result.failures[0], result.failures


def test_verify_finds_records_that_agree_with_their_digests_but_do_not_reproduce(tmp_path):
    with sup.build(tmp_path / "s") as store:
        keys = store.trajectory_keys()
        # (a) a k trace forged with matching tau: the store's own checks pass
        k = store.k_trace(keys[2])
        forged = np.minimum(k, k[0] - 1)
        forged[0] = k[0]
        assert not np.array_equal(forged, k)
        _tamper(store, "UPDATE traces SET k_points = ?, tau = ? WHERE tid = ?", (
            zlib.compress(ob.k_change_points(forged).astype("<i4").tobytes()),
            zlib.compress(ob.first_passage(forged).tau.astype("<i4").tobytes()),
            store._tid(keys[2]),
        ))  # fmt: skip
        # (b) the joint state of t = 10 replaced by that of t = 5, with its digest
        old = store.snapshot(keys[3], 5)
        _tamper(store, "UPDATE snapshots SET joint = ?, joint_digest = ? WHERE tid = ? AND t = 10",
                (old.joint_blob, old.joint_digest, store._tid(keys[3])))  # fmt: skip
        # (c) a physical code replaced by another one of the same rank, with its digest
        assert any(_forge(store, keys[4], t, "phys", (0, 1)) for t in (5, 10, 2))
        assert store.check() == []
        result = runner.verify(store, sample=None)
        text = "\n".join(result.failures)
        assert not result.ok
        assert f"{keys[2]}: the trace at horizon 10 differs" in text
        assert (
            f"snapshot of {keys[3]} at t = 10 is already stored with another joint_digest" in text
        )
        assert f"snapshot of {keys[4]}" in text and "phys_digest" in text


def test_verify_compares_the_native_code_snapshot(tmp_path, monkeypatch):
    """The engine's own code_snapshot route must give the physical code of the rerun's joint
    state (v1.4 §9.2 (g), (h)); a disagreement is reported."""
    with sup.build(tmp_path / "s") as store:
        real = runner._native_codes

        def swapped(store_, key, depths, num_threads):
            out = real(store_, key, depths, num_threads)
            t = max(out)
            moved = PhaseFreeTableau.from_bytes(_permuted(out[t], (0, 1)))
            return {**out, t: moved} if moved.digest() != out[t].digest() else out

        monkeypatch.setattr(runner, "_native_codes", swapped)
        result = runner.verify(store, sample=None)
        assert not result.ok
        assert any("native code_snapshot" in f for f in result.failures)


# ---------------------------------------------------------------------------------------------
# T3-F14: a rerun compares both snapshot digests
# ---------------------------------------------------------------------------------------------

DEEPER = {"depth": {"layers": {"fixed": 13}, "checkpoints": {"list": [2, 5, 10]}}}


@pytest.mark.parametrize(
    "setup, column, swap, protocol",
    [
        ({}, "joint", (6, 7), "F"),  # two references: same k, same physical code
        ({"observables": {**sup.BASE["observables"], "tableau_snapshots": False}}, "phys", (0, 1),
         "F"),
        ({"protocols": {"F": {}, "P": {"checkpoints": {"list": [5]}}}}, "joint", (0, 1), "P"),
    ],
    ids=["joint", "code-only", "protocol-P"],
)  # fmt: skip
def test_a_rerun_compares_both_snapshot_digests(tmp_path, setup, column, swap, protocol):
    with sup.build(tmp_path / "s", sup.manifest(**setup)) as store:
        forged = None
        for key in store.trajectory_keys(protocol=protocol):
            if store.snapshot(key, 5).status != "absent_after_purification" and _forge(
                store, key, 5, column, swap
            ):
                forged = key
                break
        assert forged is not None
        if column == "joint" and protocol == "F":  # the forged state keeps k and the code
            assert (
                store.snapshot(forged, 5)
                .tableau()
                .physical_tableau(6)
                .same_space(store.snapshot(forged, 5).physical_tableau())
            )
        assert store.check() == []
        ident = (forged.protocol, forged.n, forged.beta.token, forged.p.token, forged.g,
                 forged.c, forged.r)  # fmt: skip

        def own(records):  # the forged trajectory's records (other units may commit first)
            return {k: v for k, v in records.items() if tuple(k[1:8]) == ident}

        before = own(sup.records(store))
        runner.extend(store, sup.manifest(**setup, **DEEPER))
        with pytest.raises(ValueError, match=f"another {column}_digest"):
            runner.run(store)
        assert own(sup.records(store)) == before  # its unit wrote nothing
        assert f"another {column}_digest" in store.units(status="failed")[-1]["error"]


# ---------------------------------------------------------------------------------------------
# T3-F08: code observables after a stop
# ---------------------------------------------------------------------------------------------


def test_code_observables_after_a_stop_are_the_exact_values_at_k_zero(tmp_path):
    common = dict(
        conditions=[{"n": [6], "beta": ["0.25"], "p": ["0.3"]}],
        depth={"layers": {"fixed": 24}, "checkpoints": {"list": [2, 8, 16]}},
    )
    stopped = sup.build(tmp_path / "stop", sup.manifest(dynamics={"stop_when_purified": True},
                                                        **common))  # fmt: skip
    running = sup.build(tmp_path / "run", sup.manifest(**common))
    with stopped, running:
        statuses = {s.status for k in stopped.trajectory_keys() for s in
                    (stopped.snapshot(k, t) for t in stopped.depths(k))}  # fmt: skip
        assert "absent_after_purification" in statuses
        cases = [
            ("code_rate", None), ("stabilizer_rank", None), ("d_conn", None),
            ("mean_leakage", {"m": 2, "J": 1.0, "samples": 6}),
        ]  # fmt: skip
        for name, params in cases:
            runner.compute(stopped, name, params)
            runner.compute(running, name, params)
            rows = stopped.observations(name)
            after = rows["method"] == "k=0-after-stop"
            assert after.any() and set(rows["status"]) == {"ok"}
            for t in (8, 16, 24):
                a = stopped.aggregate(name, t=t, n=6, beta="0.25", p="0.3", record=False)
                b = running.aggregate(name, t=t, n=6, beta="0.25", p="0.3", record=False)
                # unconditional: the stop changes no code observable's mean (v1.4 §9)
                assert (a["mean"], a["trajectories"]) == (b["mean"], b["trajectories"]), name
        k = stopped.aggregate("k", t=16, n=6, beta="0.25", p="0.3", record=False)
        rate = stopped.aggregate("code_rate", t=16, n=6, beta="0.25", p="0.3", record=False)
        assert rate["mean"] == pytest.approx(k["mean"] / 6, rel=1e-15)
        runner.compute(stopped, "half_cut_entropy")  # an entanglement stays missing (SI-05)
        half = stopped.observations("half_cut_entropy")
        assert "absent_after_purification" in set(half["status"])
        assert runner.verify(stopped, sample=None).ok


# ---------------------------------------------------------------------------------------------
# T3-F10: the export
# ---------------------------------------------------------------------------------------------


def _csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_the_export_says_why_values_are_missing_and_keeps_the_history_apart(tmp_path):
    m = sup.manifest(conditions=[{"n": [6], "beta": ["0.25"], "p": ["0.6"]}])
    with sup.build(tmp_path / "s", m) as store:
        runner.extend(store, sup.manifest(conditions=m.to_dict()["conditions"],
                                          depth={"layers": {"fixed": 14},
                                                 "checkpoints": {"list": [2, 5]}}))  # fmt: skip
        runner.run(store)
        key = store.trajectory_keys()[0]
        tid = store._tid(key)
        with store.transaction() as tx:  # one infinite value: left out with its own status
            tx.put_observation(tid, key, 2, "d_conn", {"max_n": 20}, value=math.inf, se=0,
                               data=None, method="test")  # fmt: skip
        out = store.export(tmp_path / "export") / "F_n6_beta0.25_p0.6"
        obs = _csv(out / "observations.csv")
        assert "status" in obs[0] and "data" in obs[0]
        assert {"undefined_no_code", "vector", "ok"} <= {r["status"] for r in obs}
        arrays = np.load(out / "observations.npz")
        assert set(arrays["status"]) == {r["status"] for r in obs}
        # one current row per trajectory of every trajectory-level key; the history beside it
        first = [r for r in obs if r["observable"] == "first_passage"]
        assert len(first) == 6 and all(r["t"] == "-1" for r in first)
        history = [r for r in _csv(out / "history.csv") if r["observable"] == "first_passage"]
        assert len(history) == 12 and sorted(r["current"] for r in history) == ["0"] * 6 + ["1"] * 6
        trajectories = _csv(out / "trajectories.csv")
        cur = store._current("F", 6, "0.25")
        assert {r["requirements"] for r in trajectories} == {cur.digest}
        assert {r["context"] for r in trajectories} == {cur.context}
        assert cur.digest in json.loads((out.parent / "requirements.json").read_text())
        peaks = _csv(out / "peaks.csv")
        names = {(r["observable"], r["params"]) for r in peaks}
        assert len(peaks) == 6 * len(names) and {r["horizon"] for r in peaks} == {"14"}
        assert all(r["status"] in ("ok", "missing") for r in peaks)
        summary = json.loads((out / "peaks.json").read_text())
        assert {e["context"] for v in summary.values() for e in v} == {cur.context}
        aggregates = _csv(out / "aggregates.csv")
        conn = [r for r in aggregates if r["observable"] == "d_conn"]
        assert len(conn) == 0  # a single infinite row has no finite value to average
        d_cyc = [r for r in aggregates if r["observable"] == "d_cyc"]
        assert any(json.loads(r["omitted"]).get("undefined_no_code") for r in d_cyc)
        assert all(r["nan_policy"] == ("omit" if json.loads(r["omitted"]) else "raise")
                   for r in aggregates)  # fmt: skip


# ---------------------------------------------------------------------------------------------
# T3-F11: one committed state per export, digest and check
# ---------------------------------------------------------------------------------------------


def test_an_export_and_a_digest_read_one_committed_state(tmp_path, monkeypatch):
    path = tmp_path / "s"
    writer = sup.build(path, sup.manifest(storage={"graphs_per_unit": 1}), run=False)
    try:
        runner.run(writer, max_units=2)
        reader = TableauStore.open(path)
        before_digest = reader.logical_digest()
        before = reader.export(tmp_path / "before")
        committed: list[int] = []
        busy: list[bool] = []

        def commit_once() -> None:  # another process's unit, committed during the read
            if not committed and not busy:
                busy.append(True)
                committed.append(runner.run(writer, max_units=1).units)
                busy.clear()

        real_aggregates = store_module._graph_aggregates

        def between(rows):
            commit_once()
            return real_aggregates(rows)

        monkeypatch.setattr(store_module, "_graph_aggregates", between)
        during = reader.export(tmp_path / "during")
        assert committed == [1]
        for file in sorted(p.relative_to(before) for p in before.rglob("*") if p.is_file()):
            assert (during / file).read_bytes() == (before / file).read_bytes(), file
        assert reader.logical_digest() != before_digest  # the commit is seen afterwards
        # the digest alone: a unit committed while its traces are read changes nothing
        committed.clear()
        middle = reader.logical_digest()
        real_int32 = store_module._int32_array

        def reading(*args, **kwargs):
            commit_once()
            return real_int32(*args, **kwargs)

        monkeypatch.setattr(store_module, "_int32_array", reading)
        assert reader.logical_digest() == middle and committed == [1]
        monkeypatch.setattr(store_module, "_int32_array", real_int32)
        assert reader.logical_digest() != middle
        assert reader.check() == []
        reader.close()
    finally:
        writer.close()


# ---------------------------------------------------------------------------------------------
# T3-F12: the snapshot cursor
# ---------------------------------------------------------------------------------------------


def _synthetic(path: Path, trajectories: int, depths: int) -> TableauStore:
    """A store holding ``trajectories * depths`` snapshot rows written directly (blobs are
    placeholders: the cursor never decodes them)."""
    store = TableauStore.create(path, sup.manifest())
    db = store._db()
    with store.transaction() as tx:
        tx.add_unit(protocol="F", n=6, beta="0.25", p="0.1", graphs="[0,1]", circuits="[0,1]",
                    layers=10, requirements="x", status="done", revision=1, trajectories=1,
                    sparsegf2_version="0", randomness_contract=2, environment="{}",
                    environment_digest="x")  # fmt: skip
        db.execute(
            "INSERT INTO graphs (n, beta, g, seed, generator, edge_order, num_edges, edges, "
            "digest, measures, measures_version) VALUES (6, '0.25', 0, '[]', 'x', 'x', 0, x'', "
            "'x', '{}', 'x')"
        )
        db.executemany(
            "INSERT INTO trajectories (tid, protocol, n, beta, p, g, c, r, u, p_float_hex, "
            "run_seed, graph_digest, dynamics_version, unit_id, revision) VALUES "
            "(?, 'F', 6, '0.25', ?, 0, ?, 0, 0, 'x', '[]', 'x', 'x', 1, 1)",
            [(i + 1, "0.1" if i % 2 else "0.3", i) for i in range(trajectories)],
        )
        db.executemany(
            "INSERT INTO snapshots (tid, t, num_qubits, rank, k, joint, joint_digest, phys, "
            "phys_digest, status, unit_id) VALUES (?, ?, 12, 12, 0, ?, 'x', ?, 'x', 'stored', 1)",
            [(i + 1, t, b"j" * 64, b"p" * 16) for i in range(trajectories) for t in range(depths)],
        )
    return store


def test_the_snapshot_cursor_neither_sorts_nor_revisits_rows(tmp_path):
    costs = []
    for size, name in ((60, "small"), (240, "large")):
        with _synthetic(tmp_path / name, size, 8) as store:
            db = store._db()
            seen: list[str] = []
            db.set_trace_callback(seen.append)
            first = next(store.iter_snapshots(p="0.1", batch_size=8))
            db.set_trace_callback(None)
            sql = next(s for s in seen if "FROM trajectories tr CROSS JOIN snapshots" in s)
            plan = " | ".join(r[3] for r in db.execute("EXPLAIN QUERY PLAN " + sql))
            assert "TEMP B-TREE" not in plan and "SCAN s" not in plan, plan
            assert first.key.p.token == "0.1"
            steps = [0]

            def count(steps: list[int] = steps) -> int:
                steps[0] += 1
                return 0

            db.set_progress_handler(count, 16)
            got = [(s.key.c, s.t) for s in store.iter_snapshots(batch_size=8)]
            db.set_progress_handler(None, 0)
            assert got == sorted(got) and len(got) == size * 8
            filtered = [(s.key.c, s.t) for s in store.iter_snapshots(p="0.1", batch_size=8)]
            assert filtered == [(c, t) for c, t in got if c % 2]
            costs.append(steps[0] / len(got))
    # four times the rows, the same work per row (the old cursor's grew in proportion)
    assert costs[1] <= 1.25 * costs[0], costs


# ---------------------------------------------------------------------------------------------
# T3-F19: refusals that say what to do
# ---------------------------------------------------------------------------------------------


def test_a_revision_may_not_repeat_an_earlier_manifest(tmp_path):
    first = sup.manifest()
    with sup.build(tmp_path / "s", first, run=False) as store:
        runner.extend(store, sup.manifest(storage={"graphs_per_unit": 3}))
        with pytest.raises(ValueError, match="revision 1 again"):
            runner.extend(store, first)
        assert store.revision == 2
        assert runner.extend(store, sup.manifest(storage={"graphs_per_unit": 3})) is not None


def test_a_creation_that_never_completed_is_named(tmp_path, monkeypatch):
    def fail() -> None:
        raise KeyboardInterrupt("stopped inside create")

    monkeypatch.setattr(store_module, "_git_commit", fail)
    with pytest.raises(KeyboardInterrupt):
        TableauStore.create(tmp_path / "a", sup.manifest())
    monkeypatch.undo()
    assert not (tmp_path / "a" / STORE_FILE).exists()  # nothing half made is left
    TableauStore.create(tmp_path / "a", sup.manifest()).close()  # the directory is reusable
    (tmp_path / "b").mkdir()
    sqlite3.connect(tmp_path / "b" / STORE_FILE).close()  # a process killed inside create
    for write in (False, True):
        with pytest.raises(ValueError, match="creation never completed.*remove the directory"):
            TableauStore.open(tmp_path / "b", write=write)


# ---------------------------------------------------------------------------------------------
# T3-F23: a unit that fails before its transaction is recorded
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("where", ["engine", "canonical forms"])
def test_a_unit_that_fails_outside_its_transaction_is_recorded(tmp_path, monkeypatch, where):
    with sup.build(tmp_path / "s", run=False) as store:

        def broken(*args, **kwargs):
            raise MemoryError(f"injected in the {where}")

        if where == "engine":
            monkeypatch.setattr(mipt.MonitoredCircuit, "run", broken)
        else:
            monkeypatch.setattr(PhaseFreeTableau, "from_words_batch", broken)
        with pytest.raises(MemoryError):
            runner.run(store)
        failed = store.units(status="failed")
        assert len(failed) == 1 and failed[0]["error"] == f"MemoryError: injected in the {where}"
        assert store.status()["units_failed"] == 1 and store.counts()["trajectories"] == 0
        monkeypatch.undo()
        assert runner.run(store).remaining_units == 0
