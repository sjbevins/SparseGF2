"""Records of the study store that the requirements review asked for (design gaps G-GR-1,
G-GR-2, G-ENG-1, G-TAB-1, G-TAB-2, G-RUN-1, G-RUN-2, G-RUN-3, G-STO-2, G-STO-3, G-STO-4,
G-OBS-2, G-OBS-4, G-ID-1, G-ID-2; spec issues SI-05, SI-07, SI-08, SI-24).

Graphs made once and checked in every unit; graph measurements on hand cases and by an
independent breadth-first search; the exact engine call of every unit; the recorded versions,
qubit order and contract; the outcome seed of the outcome-aware replay and the Born law of its
outcomes (against Stim, measurement by measurement); observation statuses and the k = 0
conventions; the append-only triggers; reruns before new trajectories and one schedule per
condition; per-unit provenance and the confirmatory environment; peak domains; aggregate
records; sampler diagnostics; the probe streams shared across rates.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import sqlite3
import struct

import numpy as np
import pytest
import stim

import _study_store_support as sup
import _ws_reference as wsref
from _protocol_oracle import probe_orderings
from sparsegf2 import api, graphs, mipt
from sparsegf2.study import observables as ob
from sparsegf2.study import runner
from sparsegf2.study.identity import Token, TrajectoryKey
from sparsegf2.study.manifest import ExperimentManifest
from sparsegf2.study.store import (
    OUTCOME_SEED_TAG,
    QUBIT_ORDER,
    STORE_FILE,
    TableauStore,
    graph_measures,
)


@pytest.fixture(scope="module")
def stopped(tmp_path_factory):
    """A store whose trajectories purify and stop (p = 0.9), with every probe."""
    m = sup.manifest(
        conditions=[{"n": [6], "beta": ["0.25"], "p": ["0.9"]}],
        dynamics={"stop_when_purified": True},
        depth={"layers": {"fixed": 10}, "checkpoints": {"list": [1, 2, 5, 8]}},
    )
    store = sup.build(tmp_path_factory.mktemp("stopped") / "store", m)
    yield store
    store.close()


# ---------------------------------------------------------------------------------------------
# G-GR-1: graphs once; G-GR-2: graph measurements
# ---------------------------------------------------------------------------------------------


def test_graphs_are_generated_once_and_checked_in_every_unit(tmp_path, monkeypatch):
    calls = collections.Counter()
    original = graphs.watts_strogatz

    def counted(n, k, beta, *, seed=None):
        calls[tuple(seed)] += 1
        return original(n, k, beta, seed=seed)

    monkeypatch.setattr(graphs, "watts_strogatz", counted)
    store = sup.build(tmp_path / "s", run=False)  # two rates, three graphs, units of 2 graphs
    try:
        for _ in range(4):  # one unit per run, the store reopened in between
            runner.run(store, max_units=1)
            store.close()
            store = TableauStore.open(tmp_path / "s", write=True)
        assert len(calls) == 3 and set(calls.values()) == {1}  # once per (n, beta, g)
        assert {s[-1] for s in calls} == {0, 1, 2}
        assert store.counts()["graphs"] == 3
    finally:
        store.close()

    # a graph altered in the store makes the next unit that reads it fail, and changes nothing
    store = sup.build(tmp_path / "t", run=False)
    try:
        runner.run(store, max_units=1)  # graphs 0 and 1
        edges = np.array(store.graph(6, "0.25", 0).edges)
        a, b = edges[-1]
        forged = np.array(sorted(map(tuple, [*edges[:-1].tolist(), [a, (b + 1) % 6]])))
        con = sup.open_for_tampering(store)
        con.execute(
            "UPDATE graphs SET edges = ? WHERE g = 0",
            (__import__("zlib").compress(forged.astype("<i4").tobytes()),),
        )
        con.commit()
        con.close()
        before = sup.records(store)
        with pytest.raises(ValueError, match=r"graph \(n = 6, beta = 0.25, g = 0\)"):
            runner.run(store)
        after = sup.records(store)
        assert {k: v for k, v in after.items() if k[0] != "graph"} == {
            k: v for k, v in before.items() if k[0] != "graph"
        }
    finally:
        store.close()


def _bfs_measures(edges, n):
    """Mean shortest-path distance over reachable unordered pairs and the components, by a plain
    breadth-first search written here."""
    nbr = {v: set() for v in range(n)}
    for a, b in np.asarray(edges).tolist():
        nbr[a].add(b)
        nbr[b].add(a)
    total = pairs = 0
    seen_any: set[int] = set()
    components = 0
    for s in range(n):
        dist = {s: 0}
        frontier = [s]
        while frontier:
            nxt = []
            for v in frontier:
                for w in nbr[v]:
                    if w not in dist:
                        dist[w] = dist[v] + 1
                        nxt.append(w)
            frontier = nxt
        if s not in seen_any:
            components += 1
            seen_any |= set(dist)
        for w, d in dist.items():
            if w > s:
                total += d
                pairs += 1
    return (total / pairs if pairs else math.nan), pairs, components


def test_graph_measures_conventions_on_hand_cases_and_by_bfs():
    # two disjoint K_5 (n = 10): two components, lambda_2 = 0, every reachable pair at distance 1
    k5 = [(a, b) for a in range(5) for b in range(a + 1, 5)]
    two = np.array(k5 + [(a + 5, b + 5) for a, b in k5])
    m = graph_measures(two, 10)
    assert (m["components"], m["lambda2"], m["mean_distance"], m["reachable_pairs"]) == (
        2,
        0.0,
        1.0,
        20,
    )
    assert m["degree_variance"] == 0.0
    # the ring C(n, 2) itself: q = 0; an isolated vertex: no pairs from it
    ring = wsref.rewire(12, 0.0, wsref.graph_seed(1, 12, "0", 0))
    assert graph_measures(ring, 12)["q"] == 0.0
    lonely = graph_measures(np.array([[0, 1]]), 3)
    assert (lonely["components"], lonely["reachable_pairs"], lonely["mean_distance"]) == (2, 1, 1.0)
    rng = np.random.default_rng(5)
    for _ in range(12):
        n = int(rng.integers(8, 30))
        beta = float(rng.choice([0.0, 0.2, 0.7, 1.0]))
        edges = graphs.watts_strogatz(n, 4, beta, seed=int(rng.integers(2**31)))
        mean, pairs, components = _bfs_measures(edges, n)
        got = graph_measures(edges, n)
        assert (got["reachable_pairs"], got["components"]) == (pairs, components)
        assert got["mean_distance"] == pytest.approx(mean, rel=1e-15)
        deg = np.bincount(np.asarray(edges).ravel(), minlength=n)
        assert got["degree_variance"] == pytest.approx(float(np.var(deg)), abs=1e-12)


# ---------------------------------------------------------------------------------------------
# G-ENG-1: the engine call of every unit
# ---------------------------------------------------------------------------------------------


def test_every_unit_runs_exactly_the_v14_engine_call(tmp_path, monkeypatch):
    specs = []
    original = mipt.MonitoredCircuit.run

    def spy(self, *args, **kwargs):
        specs.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(mipt.MonitoredCircuit, "run", spy)
    m = sup.manifest(protocols=["F", "P"])
    with sup.build(tmp_path / "s", m):
        pass
    assert len(specs) == 8
    n = 6
    for spec in specs:
        protocol = "F" if spec.reference == "purification" else "P"
        expected = {
            "num_system": n,
            "circuits_per_graph": 4,
            "layers": 10,
            "schedule": "random_edges",
            "gates_per_layer": n // 2,
            "edge_weights": None,
            "reference": "purification" if protocol == "F" else "none",
            "scramble_gates": 0,
            "scramble_matchings": 0,
            "scramble_layers": 0,
            "random_orientation": False,
            "checkpoints": "every_layer",
            "stop_when_purified": False,
            "periodic": True,
        }
        assert {k: getattr(spec, k) for k in expected} == expected
        assert np.array_equal(spec.gate_set, np.arange(11520))
        assert np.array_equal(spec.system_order, np.arange(n))
        assert spec.measurement_rate in (0.1, 0.3)
        probes = {p.name: (p.kind, list(spec.probe_layers(p.name))) for p in spec.probes}
        every = list(range(11))
        cps = [0, 1, 2, 5, 10]
        if protocol == "F":
            assert probes == {
                "code_dimension": ("code_dimension", every),
                "tableau_snapshot": ("tableau_snapshot", cps),
                "contiguous_distance": ("contiguous_distance", cps),
                "erasure_first_failure": ("erasure_first_failure", cps),
                "uncorrectable_region_counts": ("uncorrectable_counts", cps),
                "ring": ("correlated_erasure", cps),
            }
        else:
            assert probes == {"tableau_snapshot": ("tableau_snapshot", cps)}


# ---------------------------------------------------------------------------------------------
# G-TAB-1, G-ID-1: versions, qubit order, contract
# ---------------------------------------------------------------------------------------------


def test_the_versions_the_qubit_order_and_the_bell_pairs_are_recorded(tmp_path):
    with sup.build(tmp_path / "s") as store:
        meta = store.meta()
        v = meta["versions"]
        assert v["randomness_contract"] == mipt.RANDOMNESS_CONTRACT == 2
        assert (v["graph_seed_tag"], v["circuit_seed_tag"]) == (0x57534731, 0x57534331)
        assert (v["region_seed_tag"], v["region_seed_words"]) == (0x5753524F, 19)
        assert v["outcome_seed_tag"] == OUTCOME_SEED_TAG == 0x57534F31
        assert v["token_format"] and v["qubit_order"] == QUBIT_ORDER == meta["qubit_order"]
        assert "reference n+i Bell-paired with system i" in QUBIT_ORDER
        env = json.loads(meta["environment"])
        assert env["numpy"] == np.__version__
        # t = 0 is span{X_i X_{n+i}, Z_i Z_{n+i}} for every trajectory: the pairing, written
        # out and canonicalized by an independent elimination
        n = 6
        rows = np.zeros((2 * n, 4 * n), dtype=np.uint8)
        for i in range(n):
            rows[i, i] = rows[i, n + i] = 1  # X_i X_{n+i}
            rows[n + i, 2 * n + i] = rows[n + i, 2 * n + n + i] = 1  # Z_i Z_{n+i}
        golden = sup.canonical_digest(rows, 2 * n)
        assert {store.snapshot(k, 0).joint_digest for k in store.trajectory_keys()} == {golden}


def test_a_store_of_another_randomness_contract_is_refused(tmp_path, monkeypatch):
    store = sup.build(tmp_path / "s")
    key = store.trajectory_keys()[0]
    store.close()
    monkeypatch.setattr(mipt, "RANDOMNESS_CONTRACT", 3)
    with pytest.raises(ValueError, match="randomness_contract"):
        TableauStore.open(tmp_path / "s", write=True)
    with TableauStore.open(tmp_path / "s") as reader:
        with pytest.raises(ValueError, match="randomness contract 2"):
            runner.replay(reader, key, [2])
        report = runner.verify(reader, None)
        assert not report.ok and "randomness contract" in report.failures[0]
        assert reader.tableau(key, 2).is_pure()  # stored data stay readable


# ---------------------------------------------------------------------------------------------
# G-TAB-2, SI-24: the outcome seed and the Born law
# ---------------------------------------------------------------------------------------------


def _words(text):
    return struct.unpack("<II", hashlib.blake2b(text.encode(), digest_size=8).digest())


def test_the_outcome_seed_is_a_tagged_function_of_the_whole_key():
    key = TrajectoryKey("ws-store-test", 6, "0.25", "0.3", 2, 1)
    seed = runner.outcome_seed_of(key)
    by_hand = (
        0x57534F31,
        *_words("ws-store-test"),
        0,
        6,
        *_words("0.25"),
        *_words("0.3"),
        2,
        1,
        0,
    )
    assert seed == by_hand and len(seed) == 12
    variants = [
        TrajectoryKey("ws-store-test", 6, "0.25", "0.3", 2, 1, 0, "P"),
        TrajectoryKey("ws-store-test", 6, "0.25", "0.1", 2, 1),
        TrajectoryKey("ws-store-test", 6, "0.5", "0.3", 2, 1),
        TrajectoryKey("ws-store-test", 7, "0.25", "0.3", 2, 1),
        TrajectoryKey("ws-store-test", 6, "0.25", "0.3", 3, 1),
        TrajectoryKey("ws-store-test", 6, "0.25", "0.3", 2, 0),
        TrajectoryKey("other", 6, "0.25", "0.3", 2, 1),
    ]
    assert len({seed, *(runner.outcome_seed_of(v) for v in variants)}) == 8
    assert seed[0] not in (0x57534731, 0x57534331, 0x5753524F)  # its own domain


def test_the_signed_replay_is_recorded_and_reproduced_from_the_identity(tmp_path):
    with sup.build(tmp_path / "s") as store:
        key = store.trajectory_keys()[7]
        depths = list(store.depths(key))
        first = runner.replay_signed(store, key, depths, record=True)
        assert first.outcome_seed == runner.outcome_seed_of(key)
        again = runner.replay_signed(store, key, depths, record=True)  # compared, not added
        assert np.array_equal(first.measurements, again.measurements)
        assert first.signed_digests == again.signed_digests
        (stored,) = store.signed_runs(key)
        assert stored["outcome_seed"] == list(first.outcome_seed)
        assert np.array_equal(stored["outcomes"], first.measurements)
        assert np.array_equal(stored["random"], first.random)
        assert stored["signed_digests"] == first.signed_digests
        other = runner.replay_signed(store, key, depths, outcome_seed=12345)
        for t in depths:  # the signs may differ, the phase-free group may not
            assert other.tableaux[t] == first.tableaux[t] == store.tableau(key, t)
            xs, zs, signs = first.signed[t]
            assert runner.signed_digest(xs, zs, signs) == first.signed_digests[t]
        assert store.counts()["observations"] == len(
            store._db().execute("SELECT 1 FROM observations").fetchall()
        )


def _check_born(circuit, measurements, random_flags, num_qubits):
    """Follow the signed replay's outcomes in Stim: a measurement whose outcome the state
    determines must have that outcome and be flagged deterministic; a random one must be
    flagged random. Returns the random outcomes."""
    sim = stim.TableauSimulator()
    sim.set_num_qubits(num_qubits)
    index = 0
    randoms = []
    for ins in circuit.to_stim().flattened():
        if ins.name != "M":
            sim.do(ins)
            continue
        for target in ins.targets_copy():
            q = target.value
            peek = sim.peek_z(q)
            outcome = int(measurements[index])
            if peek == 0:
                assert random_flags[index]
                randoms.append(outcome)
            else:
                assert not random_flags[index]
                assert outcome == (0 if peek == 1 else 1)
            sim.postselect_z(q, desired_value=bool(outcome))
            index += 1
    assert index == len(measurements)
    return randoms


def test_the_signed_replay_follows_the_born_law(tmp_path):
    """Deterministic outcomes are the state's eigenvalues and random ones are fair coins.

    Over 40 outcome seeds of 4 trajectories the random outcomes are tested against
    Binomial(N, 1/2) with |z| < 4.5 (two-sided false alarm 6.8e-6 for a correct sampler).
    """
    with sup.build(tmp_path / "s") as store:
        ones = total = 0
        for key in store.trajectory_keys()[::3]:
            circuit = runner.circuit(store, key)
            for seed in range(40):
                replay = runner.replay_signed(store, key, [10], outcome_seed=seed)
                randoms = _check_born(circuit, replay.measurements, replay.random, 12)
                ones += sum(randoms)
                total += len(randoms)
        assert total > 400
        z = (ones - total / 2) / math.sqrt(total / 4)
        assert abs(z) < 4.5, (ones, total, z)


# ---------------------------------------------------------------------------------------------
# G-STO-2, SI-05, SI-07, SI-08: statuses and the k = 0 conventions
# ---------------------------------------------------------------------------------------------


def test_statuses_and_the_conventions_at_k_zero(stopped):
    n = 6
    seen = collections.Counter()
    for key in stopped.trajectory_keys():
        trace = stopped.trace(key)
        for t in stopped.depths(key):
            k = stopped.snapshot(key, t).k
            d = stopped.observation(key, t, "d_cyc")
            D = stopped.observation(key, t, "D", params={"samples": 12, "exact_budget": 6})
            u = stopped.observation(
                key, t, "uniform_failure", params={"samples": 12, "exact_budget": 6}
            )
            if k == 0:  # SI-08: raw kept undefined, zero extension and has_code beside it
                assert d["status"] == "undefined_no_code" and math.isnan(d["value"])
                assert d["data"] == {"has_code": False, "zero_extended": 0}
                # SI-07: the native n + 1 of every ordering is kept raw; D is the defined 0
                assert u["data"]["first_failure"] == [n + 1] * 12 and u["status"] == "vector"
                assert D["value"] == 0.0 and D["status"] == "ok" and not D["data"]["has_code"]
                seen["k0"] += 1
            else:
                assert d["status"] == "ok" and 1 <= d["value"] <= n
                assert D["data"]["has_code"] and D["value"] >= 1
                seen["code"] += 1
        fp = stopped.observation(key, -1, "first_passage")
        assert fp["status"] == ("ok" if trace.tau0 >= 0 else "censored")
        peak = stopped.observation(key, -1, "peak_d_cyc", params={"rule": "exact"})
        assert peak["status"] == ("missing" if peak["data"]["missing"] else "ok")
        seen[peak["status"]] += 1
    assert seen["k0"] > 0 and seen["code"] > 0
    # SI-05: after a stop, a retrospective entanglement value is missing, never zero
    summary = runner.compute(stopped, "half_cut_entropy")
    values = stopped.observations("half_cut_entropy")
    absent = values["status"] == "absent_after_purification"
    assert absent.any() and np.all(np.isnan(values["value"][absent]))
    assert np.all(values["status"][~absent] == "ok")
    assert summary.snapshots == len(values["value"])


def test_the_store_says_why_a_value_is_missing(tmp_path):
    with sup.build(tmp_path / "s") as store:
        key = store.trajectory_keys()[0]
        tid = store._tid(key)
        with store.transaction() as tx:
            tx.put_observation(tid, key, 2, "d_conn", {"max_n": 20}, value=math.inf, se=0,
                               data=None, method="test")  # fmt: skip
            tx.put_observation(tid, key, 5, "d_conn", {"max_n": 20}, value=math.nan, se=0,
                               data=None, method="test", status="undefined_no_code")  # fmt: skip
            with pytest.raises(ValueError, match="finite value"):
                tx.put_observation(tid, key, 10, "d_conn", {"max_n": 20}, value=math.nan, se=0,
                                   data=None, method="test", status="ok")  # fmt: skip
        got = store.observations("d_conn")
        assert list(got["status"]) == ["infinite", "undefined_no_code"]
        with store.transaction() as tx, pytest.raises(ValueError, match="status"):
            tx.put_observation(tid, key, 5, "d_conn", {"max_n": 20}, value=math.nan, se=0,
                               data=None, method="test", status="undefined")  # fmt: skip


# ---------------------------------------------------------------------------------------------
# G-RUN-1: append-only
# ---------------------------------------------------------------------------------------------


def test_stored_records_cannot_be_updated_or_deleted(tmp_path):
    with sup.build(tmp_path / "s") as store:
        con = sqlite3.connect(store.path / STORE_FILE)  # through SQLite: the triggers hold
        try:
            for sql in (
                "UPDATE trajectories SET u = 99",
                "DELETE FROM trajectories",
                "UPDATE traces SET final_layer = 1",
                "DELETE FROM traces",
                "UPDATE snapshots SET k = 1",
                "UPDATE snapshots SET status = 'absent_after_purification'",
                "DELETE FROM snapshots",
                "UPDATE observations SET value = 1",
                "DELETE FROM observations",
                "UPDATE graphs SET digest = 'x'",
                "UPDATE manifests SET json = '{}'",
                "DELETE FROM completions",
            ):
                with pytest.raises(sqlite3.IntegrityError, match="never"):
                    con.execute(sql)
        finally:
            con.close()
        assert store.check() == []


# ---------------------------------------------------------------------------------------------
# G-RUN-2: one schedule per condition, reruns first
# ---------------------------------------------------------------------------------------------


def test_reruns_come_first_and_every_trajectory_holds_one_schedule(tmp_path):
    first = sup.manifest()
    second = sup.manifest(
        conditions=[{"n": [6], "beta": ["0.25"], "p": ["0.1", "0.2", "0.3"]}],
        depth={"layers": {"fixed": 10}, "checkpoints": {"list": [3, 7]}},
    )
    with sup.build(tmp_path / "s", first) as store:
        units = runner.extend(store, second)
        kinds = [u.rerun for u in units]
        assert kinds == sorted(kinds, reverse=True) and any(kinds) and not all(kinds)
        assert {u.p.token for u in units if not u.rerun} == {"0.2"}
        assert {u.p.token for u in units if u.rerun} == {"0.1", "0.3"}
        runner.run(store)
        depth_sets = {store.depths(k) for k in store.trajectory_keys()}
        assert depth_sets == {(0, 1, 2, 3, 5, 7, 10)}  # old and new rates alike (v1.4 §1)
        schedules = {json.dumps(store.schedule(k)) for k in store.trajectory_keys()}
        assert len(schedules) == 1
        (schedule,) = (store.schedule(k) for k in store.trajectory_keys()[:1])
        assert schedule["checkpoints"] == [0, 1, 2, 3, 5, 7, 10]
        assert schedule["domain"] == [0, 1, 3, 7, 10]  # the last revision's own schedule


# ---------------------------------------------------------------------------------------------
# G-RUN-3: provenance per unit, the confirmatory environment
# ---------------------------------------------------------------------------------------------


def test_units_record_their_provenance(tmp_path):
    with sup.build(tmp_path / "s") as store:
        for unit in store.units():
            row = (
                store._db()
                .execute(
                    "SELECT randomness_contract, environment, environment_digest, git_dirty, "
                    "sparsegf2_version FROM units WHERE unit_id = ?",
                    (unit["unit_id"],),
                )
                .fetchone()
            )
            assert row[0] == mipt.RANDOMNESS_CONTRACT
            env = json.loads(row[1])
            assert env["numpy"] == np.__version__ and set(env) >= {"python", "numba", "llvmlite"}
            assert row[2] == hashlib.sha256(row[1].encode()).hexdigest()
            assert row[3] in (0, 1, None) and row[4]


def test_a_confirmatory_study_keeps_its_environment(tmp_path, monkeypatch):
    real = runner.environment

    def other_numpy():
        env = dict(real())
        env["numpy"] = "0.0.0"
        return env

    for label, refused in (("confirmatory", True), ("exploratory", False)):
        data = copy.deepcopy(sup.BASE)
        data["label"] = label
        store = sup.build(tmp_path / label, ExperimentManifest.from_json(data), run=False)
        try:
            runner.run(store, max_units=1)
            monkeypatch.setattr(runner, "environment", other_numpy)
            if refused:
                with pytest.raises(ValueError, match="numpy"):
                    runner.run(store)
            else:
                runner.run(store)
                digests = {
                    r[0]
                    for r in store._db().execute("SELECT DISTINCT environment_digest FROM units")
                }
                assert len(digests) == 2
            monkeypatch.setattr(runner, "environment", real)
        finally:
            store.close()


# ---------------------------------------------------------------------------------------------
# G-OBS-2: peaks over the last revision's own schedule
# ---------------------------------------------------------------------------------------------


def test_peaks_do_not_depend_on_the_revision_history(tmp_path):
    first = sup.manifest(depth={"layers": {"fixed": 10}, "checkpoints": {"list": [2, 5]}})
    second = sup.manifest(depth={"layers": {"fixed": 10}, "checkpoints": {"list": [3, 7]}})
    with (
        sup.build(tmp_path / "revised", first) as revised,
        sup.build(tmp_path / "fresh", second) as fresh,
    ):
        runner.extend(revised, second)
        runner.run(revised)
        for name, params in (
            ("peak_d_cyc", {"rule": "exact"}),
            ("first_passage", None),
        ):
            for key in fresh.trajectory_keys():
                a = revised.observation(key, -1, name, params=params)
                b = fresh.observation(key, -1, name, params=params)
                assert _same(a["value"], b["value"]) and a["data"] == b["data"], (name, key)
                assert a["data"]["domain"] == [0, 1, 3, 7, 10]
        assert revised.depths(key) != fresh.depths(key)  # the stored depths differ (R-CK)


def _same(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == b


# ---------------------------------------------------------------------------------------------
# G-STO-4: aggregate records
# ---------------------------------------------------------------------------------------------


def test_aggregates_are_recorded_with_their_sample_set(tmp_path, stopped):
    with sup.build(tmp_path / "s") as store:
        params = {"samples": 12, "exact_budget": 6}
        record = store.aggregate("D", t=5, n=6, beta="0.25", p="0.1", params=params)
        rows = store.observations("D", t=5, p="0.1", params=params)
        graph_means = [rows["value"][rows["g"] == g].mean() for g in range(3)]
        assert record["mean"] == pytest.approx(np.mean(graph_means), rel=1e-15)
        assert record["se"] == pytest.approx(np.std(graph_means, ddof=1) / math.sqrt(3), rel=1e-12)
        assert (record["graphs"], record["trajectories"]) == (3, 6)
        again = store.aggregate("D", t=5, n=6, beta="0.25", p="0.1", params=params)  # compared
        assert again == record and len(store.aggregates("D")) == 1
        runner.extend(store, sup.manifest(trials={"graphs": 4, "circuits_per_graph": 2,
                                                  "circuit_index_stride": 4}))  # fmt: skip
        runner.run(store)
        more = store.aggregate("D", t=5, n=6, beta="0.25", p="0.1", params=params)
        assert more["sample_set"] != record["sample_set"] and more["graphs"] == 4
        assert store.aggregates("D")[0]["sample_set"] in {record["sample_set"], more["sample_set"]}
        assert len(store.aggregates("D")) == 2  # a new record; the old one is kept
        robustness = store.aggregate(
            "D_J", t=5, n=6, beta="0.25", p="0.1", field="robustness", index=1,
            params={**sup.BASE["observables"]["correlated_erasure"][0]}, record=False,
        )  # fmt: skip
        assert robustness["graphs"] == 4
    with pytest.raises(ValueError, match="undefined_no_code"):
        stopped.aggregate("d_cyc", t=8, n=6, beta="0.25", p="0.9", record=False)
    omitted = stopped.aggregate(
        "d_cyc", t=8, n=6, beta="0.25", p="0.9", nan_policy="omit", record=False
    )
    assert omitted["contributing_trajectories"] < omitted["trajectories"]


# ---------------------------------------------------------------------------------------------
# G-OBS-4: sampler diagnostics
# ---------------------------------------------------------------------------------------------


def test_the_correlated_samplers_diagnostics_are_archived(tmp_path):
    with sup.build(tmp_path / "s") as store:
        written = runner.diagnose_samplers(store, chains=2, samples=16, graphs=[0, 1], sizes=[1, 3])
        rows = store.sampler_diagnostics()
        assert written == len(rows) == 2 * 2 * 2  # graphs x sizes x couplings
        assert {(r["g"], r["m"], r["coupling"]) for r in rows} == {
            (g, m, j) for g in (0, 1) for m in (1, 3) for j in (0.0, 1.5)
        }
        assert all(
            r["method"] == "chain" and r["chains"] == 2 and r["probe"] == "ring" for r in rows
        )
        assert all(isinstance(r["notes"], list) for r in rows)
        assert (
            runner.diagnose_samplers(store, chains=2, samples=16, graphs=[0, 1], sizes=[1, 3]) == 0
        )
        # the stored numbers are those of the graph-side tool called with the recorded seed
        row = rows[0]
        direct = api.sample_erasure_regions(
            {"edges": store.graph(6, "0.25", row["g"]).edges.tolist(), "num_nodes": 6},
            row["m"], [0.0, 1.5], 16, chains=2, seed=row["seed"],
        )  # fmt: skip
        match = next(c for c in direct["couplings"] if c["coupling"] == row["coupling"])
        assert (match["ess"], match["tau_int"], match["r_hat"]) == (
            row["ess"],
            row["tau_int"],
            row["r_hat"],
        )


# ---------------------------------------------------------------------------------------------
# G-ID-2: the engine's probe streams are shared across rates; retrospective regions are not
# ---------------------------------------------------------------------------------------------


def _first_failure(code, order, n):
    for m in range(1, n + 1):
        if not code.is_correctable(order[:m]):
            return m
    return n + 1


def test_probe_streams_are_shared_across_rates_unlike_the_retrospective_regions(tmp_path):
    """Gate mode shared: trajectory u draws the same first-failure orderings at every rate.

    The orderings of the oracle's transcription of the probe stream (keyed by the run seed, u,
    the probe name and the layer, with neither p nor the protocol), evaluated on each rate's
    stored code, reproduce both rates' stored samples. The retrospective region seeds contain
    p, so their regions differ between the rates.
    """
    with sup.build(tmp_path / "s") as store:
        n = 6
        for g, c in ((0, 1), (2, 0)):
            u = 4 * g + c
            seed = list(wsref.run_seed(202, n, "0.25"))
            for t in (2, 5, 10):
                orders = probe_orderings(seed, u, "erasure_first_failure", t, n, 12)
                for p in ("0.1", "0.3"):
                    key = TrajectoryKey("ws-store-test", n, "0.25", p, g, c)
                    code = store.code(key, t)
                    stored = store.observation(
                        key, t, "uniform_failure", params={"samples": 12, "exact_budget": 6}
                    )["data"]["first_failure"]
                    assert stored == [_first_failure(code, o, n) for o in orders], (p, t)
        a = ob.region_seed(
            TrajectoryKey("ws-store-test", n, "0.25", "0.1", 0, 1), 5, 2, 1.0, "mean_leakage", "1"
        )
        b = ob.region_seed(
            TrajectoryKey("ws-store-test", n, "0.25", "0.3", 0, 1), 5, 2, 1.0, "mean_leakage", "1"
        )
        assert a != b
        assert Token.parse("0.1") != Token.parse("0.3")
