"""The tableau store of the Watts–Strogatz study (v1.4 §5.3, §8; design §4).

Round trips against engine runs built here by hand (graphs from the independent reference
rewiring, seeds from the independent seed reference, the v1.4 §8.1 keywords written out), Stim
on the exported circuits and an independent canonical-bytes writer; retrieval by the parameter
tuple; indexes and bounded memory; never-overwrite and validation; integrity checks; graph
measurements against networkx; the writer lock; export.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sqlite3
import statistics
import tracemalloc
import zlib

import networkx as nx
import numpy as np
import pytest

import _study_store_support as sup
import _ws_reference as wsref
from _rowspace_oracle import dense_rows, int_rows, physical_rows, rank, restrict
from sparsegf2 import graphs, mipt
from sparsegf2.study import observables as ob
from sparsegf2.study import runner, stats
from sparsegf2.study.identity import STORE_SCHEMA_VERSION, Token, TrajectoryKey
from sparsegf2.study.store import (
    SNAPSHOT_ABSENT,
    SNAPSHOT_STORED,
    STORE_FILE,
    TableauStore,
    graph_measures,
)
from sparsegf2.tableau import PhaseFreeTableau


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One small campaign (24 trajectories), run once for the read-only tests."""
    store = sup.build(tmp_path_factory.mktemp("built") / "store")
    yield store
    store.close()


def _fresh_run(store, key, probes, *, layers=None):
    """Trajectory ``key`` run again by hand: reference graphs, reference seeds, the v1.4 §8.1
    keywords written out (not the runner's or the manifest's code)."""
    m = store.manifest
    n, beta, p = key.n, key.beta.token, key.p.token
    collection = [
        wsref.rewire(n, Token.parse(beta).float64, wsref.graph_seed(101, n, beta, g))
        for g in range(key.g + 1)
    ]
    spec = mipt.MonitoredCircuit(
        num_system=n,
        edges=collection,
        circuits_per_graph=4,
        layers=m.depth.layers.t_max(n) if layers is None else layers,
        measurement_rate=float(Token.parse(p).value),
        reference="purification",
        schedule="random_edges",
        gates_per_layer=n // 2,
        random_orientation=False,
        scramble_gates=0,
        scramble_matchings=0,
        scramble_layers=0,
        periodic=True,
        checkpoints="every_layer",
        probes=probes,
    )
    return spec.run(indices=[4 * key.g + key.c], seed=list(wsref.run_seed(202, n, beta)))


# ---------------------------------------------------------------------------------------------
# creation, schema, versions
# ---------------------------------------------------------------------------------------------


def test_create_writes_the_schema_the_versions_and_revision_one(tmp_path):
    m = sup.manifest()
    with sup.build(tmp_path / "s", m, run=False) as store:
        meta = store.meta()
        assert meta["store_schema"] == STORE_SCHEMA_VERSION
        assert meta["experiment_id"] == "ws-store-test"
        assert meta["versions"]["randomness_contract"] == mipt.RANDOMNESS_CONTRACT
        assert store.revision == 1 and store.manifest.digest() == m.digest()
        row = store._db().execute("SELECT digest, json FROM manifests").fetchone()
        assert row[0] == m.digest() and json.loads(row[1]) == m.to_dict()
        assert store.counts()["trajectories"] == 0
    assert (tmp_path / "s" / STORE_FILE).is_file()
    assert not (tmp_path / "s" / ".lock").exists()  # released on close


def test_create_refuses_a_used_directory_and_open_refuses_a_non_store(tmp_path):
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "x.txt").write_text("x")
    with pytest.raises(ValueError, match="not an empty directory"):
        TableauStore.create(tmp_path / "used", sup.manifest())
    with pytest.raises(ValueError, match="not a study store"):
        TableauStore.open(tmp_path / "used")
    with pytest.raises(TypeError, match="ExperimentManifest"):
        TableauStore.create(tmp_path / "new", {"experiment_id": "x"})  # type: ignore[arg-type]


def test_every_table_has_a_unique_natural_key(built):
    sql = {
        name: text
        for name, text in built._db().execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
        )
    }
    natural = {
        "graphs": "PRIMARY KEY (n, beta, g)",
        "trajectories": "UNIQUE (protocol, n, beta, p, g, c, r)",
        "traces": "UNIQUE (tid, layers)",
        "snapshots": "UNIQUE (tid, t)",
        "observations": "UNIQUE (tid, t, observable, version, params, context)",
        "observable_defs": "PRIMARY KEY (observable, version, params)",
        "completions": "PRIMARY KEY (tid, requirements)",
        "manifests": "digest TEXT NOT NULL UNIQUE",
    }
    for table, clause in natural.items():
        assert clause in sql[table], table
    assert built._db().execute("PRAGMA foreign_key_check").fetchall() == []
    assert built.check() == []


def test_a_store_of_another_schema_is_refused(tmp_path):
    with sup.build(tmp_path / "s", run=False) as store:
        db = store.path / STORE_FILE
    con = sqlite3.connect(db)
    con.execute("UPDATE meta SET value = 'sparsegf2.study.store/0' WHERE key = 'store_schema'")
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="schema"):
        TableauStore.open(tmp_path / "s")


def test_writing_needs_the_versions_the_store_was_made_with(tmp_path):
    with sup.build(tmp_path / "s", run=False) as store:
        db = store.path / STORE_FILE
        versions = store.meta()["versions"]
    versions["randomness_contract"] = 1
    con = sqlite3.connect(db)
    con.execute("UPDATE meta SET value = ? WHERE key = 'versions'", (json.dumps(versions),))
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="randomness_contract"):
        TableauStore.open(tmp_path / "s", write=True)
    with TableauStore.open(tmp_path / "s") as reader:  # reading is still possible
        assert reader.revision == 1


# ---------------------------------------------------------------------------------------------
# round trips (v1.4 §9.2 (g), (h); validation 5, 8)
# ---------------------------------------------------------------------------------------------


def test_every_stored_tableau_equals_a_fresh_engine_run(built):
    n = 6
    for key in built.trajectory_keys():
        depths = [t for t in built.depths(key) if built.snapshot(key, t).status == SNAPSHOT_STORED]
        fresh = _fresh_run(built, key, [mipt.tableau_snapshot(checkpoints=depths)])
        words, counts = fresh.snapshots["tableau_snapshot"]
        for j, t in enumerate(depths):
            raw = words[0, j, : counts[0, j]]
            engine = PhaseFreeTableau.from_words(raw, 2 * n)
            stored = built.tableau(key, t)
            assert stored.same_space(engine) and stored.digest() == engine.digest()
            # the canonical bytes written from scratch by an independent elimination
            dense = PhaseFreeTableau.from_words(raw, 2 * n).canonical()  # any basis of the space
            assert stored.to_bytes() == sup.canonical_bytes(dense, 2 * n)
            assert built.snapshot(key, t).joint_digest == sup.canonical_digest(dense, 2 * n)


def test_the_stored_physical_code_equals_the_native_code_snapshot(built):
    n = 6
    for key in built.trajectory_keys()[::3]:
        depths = list(built.depths(key))
        fresh = _fresh_run(built, key, [mipt.code_snapshot(checkpoints=depths)])
        for j, t in enumerate(depths):
            native = fresh.code(0, j)
            stored = built.code(key, t)
            assert stored.num_logical == native.num_logical == built.snapshot(key, t).k
            a = PhaseFreeTableau(native.stabilizer_matrix(), n)
            assert built.snapshot(key, t).physical_tableau().same_space(a)
            # and the physical rows of the stored joint state, by the independent elimination
            joint = built.tableau(key, t).canonical()
            rows = physical_rows(int_rows(joint), n, 2 * n)
            assert sup.canonical_bytes(dense_rows(rows, 2 * n), n) == (
                built.snapshot(key, t).physical_tableau().to_bytes()
            )


def test_stored_tableaux_equal_stim_on_the_exported_circuit(built):
    for key in built.trajectory_keys()[:4]:
        circuit = runner.circuit(built, key)
        states = sup.stim_states(circuit, 2 * key.n, seed=7)
        for t in built.depths(key):
            assert built.snapshot(key, t).joint_digest == sup.canonical_digest(states[t], 2 * key.n)


def test_stored_observations_reproduce_from_the_stored_codes(built):
    n = 6
    for key in built.trajectory_keys():
        trace = built.k_trace(key)
        for t in built.depths(key):
            code = built.code(key, t)
            assert code.num_logical == trace[t] == built.observation(key, t, "k")["value"]
            d = built.observation(key, t, "d_cyc")
            expected = ob.cyclic_distance_from_code(code)
            assert (math.isnan(d["value"]) and math.isnan(expected)) or d["value"] == expected
            uniform = built.observation(
                key, t, "uniform_failure", params={"samples": 12, "exact_budget": 6}
            )["data"]
            w = len(uniform["exact_counts"])
            assert w == 1 and uniform["exact_counts"] == code.uncorrectable_counts(w).tolist()
            assert len(uniform["first_failure"]) == 12
            # every ordering fails within n exactly when the code has a logical qubit
            ff = np.array(uniform["first_failure"])
            assert np.all(ff <= n) if code.num_logical else np.all(ff == n + 1)
            # D(0) and its error from the stored samples and from exact counts made here by
            # brute force on the stored joint state (T3-F17)
            counts = _failing_regions(built.tableau(key, t), n, w)
            assert uniform["exact_counts"] == counts
            D = built.observation(key, t, "D", params={"samples": 12, "exact_budget": 6})
            assert D["method"] == "hybrid" and D["data"]["max_exact"] == w
            value, se = _integrated_robustness(ff, counts, n, code.num_logical)
            assert D["value"] == pytest.approx(value, rel=1e-12, abs=1e-12)
            assert D["se"] == pytest.approx(se, rel=1e-12, abs=1e-12)


def _failing_regions(joint: PhaseFreeTableau, n: int, w: int) -> list[int]:
    """The number of uncorrectable m-qubit erasures for m = 1..w, by enumeration: a region A
    of the system fails when I(A:R) > 0, with S(X) = rank(rows restricted to X) - |X| for a
    pure state (computed with the oracle's own rank, on Python ints)."""
    rows = int_rows(joint.canonical())
    reference = list(range(n, 2 * n))

    def entropy(qubits: list[int]) -> int:
        return rank([restrict(r, qubits, 2 * n) for r in rows]) - len(qubits)

    s_ref = entropy(reference)
    counts = []
    for m in range(1, w + 1):
        fails = 0
        for region in itertools.combinations(range(n), m):
            region = list(region)
            if entropy(region) + s_ref - entropy(region + reference) > 0:
                fails += 1
        counts.append(fails)
    return counts


def _integrated_robustness(
    ff: np.ndarray, counts: list[int], n: int, k: int
) -> tuple[float, float]:
    """D(0) of v1.4 §5.7 from first-failure sizes and exact counts for m = 1..w (doc 04):
    0 when k = 0; else 1 + sum_{m <= w} (1 - c_m / C(n, m)) + mean_j Y_j with
    Y_j = max(D_j, w + 1) - (w + 1), and the error s_Y / sqrt(S)."""
    if k == 0:
        return 0.0, 0.0
    w = len(counts)
    exact = sum(1.0 - c / math.comb(n, m) for m, c in enumerate(counts, start=1))
    y = [max(int(d), w + 1) - (w + 1) for d in ff]
    return 1.0 + exact + statistics.fmean(y), statistics.stdev(y) / math.sqrt(len(y))


def test_k_trace_and_first_passage_by_the_parameter_tuple(built):
    key = built.trajectory_keys()[5]
    fresh = _fresh_run(built, key, [mipt.code_dimension()])
    k = fresh.values["code_dimension"][0].astype(np.int64)
    assert np.array_equal(built.k_trace(key), k)
    same = [
        built.k_trace(n=6, beta=0.25, p=key.p.token, g=key.g, c=key.c),
        built.k_trace(n=6, beta="1/4", p=float(key.p.value), g=key.g, c=key.c),
        built.k_trace(str(key)),
    ]
    assert all(np.array_equal(x, k) for x in same)
    fp = built.first_passage(key)
    for ell in range(7):  # the definition, directly
        hits = [t for t in range(k.size) if k[t] <= ell]
        assert fp.tau[ell] == (hits[0] if hits else -1)
    assert fp.tau[0] == fresh.purification_layer[0]
    trace = built.trace(key)
    assert trace.layers == 10 and trace.tau0 == fp.tau[0] and trace.censored0 == (fp.tau[0] < 0)
    assert list(fp.checkpoints) == [0, 1, 2, 5, 10]


def test_unsaved_depths_and_absent_snapshots_are_explicit(tmp_path, built):
    key = built.trajectory_keys()[0]
    with pytest.raises(KeyError, match="no snapshot at t = 3"):
        built.tableau(key, 3)
    with pytest.raises(KeyError, match="no trajectory"):
        built.tableau(n=6, beta="0.25", p="0.1", g=9, c=0, t=2)
    stop = sup.manifest(
        dynamics={"stop_when_purified": True},
        conditions=[{"n": [6], "beta": ["0.25"], "p": ["0.9"]}],
        depth={"layers": {"fixed": 10}, "checkpoints": {"list": [2, 5, 8]}},
    )
    with sup.build(tmp_path / "stop", stop) as store:
        absent = list(store.iter_snapshots(status=SNAPSHOT_ABSENT))
        assert absent, "at p = 0.9 some trajectory purifies before t = 10"
        for snap in absent:
            trace = store.trace(snap.key)
            assert trace.status == "stopped_at_purification"
            assert trace.final_layer == trace.tau0 < snap.t and snap.k == 0
            assert snap.joint_blob is None and snap.phys_blob is None
            with pytest.raises(ValueError, match="absent_after_purification"):
                store.tableau(snap.key, snap.t)
            with pytest.raises(ValueError, match="absent_after_purification"):
                store.code(snap.key, snap.t)
            assert np.all(store.k_trace(snap.key)[trace.final_layer :] == 0)


# ---------------------------------------------------------------------------------------------
# indexes and memory
# ---------------------------------------------------------------------------------------------


def _plan(db, sql, args):
    return " | ".join(row[-1] for row in db.execute("EXPLAIN QUERY PLAN " + sql, args))


def test_lookups_by_key_use_the_indexes(built):
    db = built._db()
    key = built.trajectory_keys()[3]
    tid = built._tid(key)
    lookups = [
        (
            "SELECT tid FROM trajectories WHERE protocol = ? AND n = ? AND beta = ? AND p = ? "
            "AND g = ? AND c = ? AND r = ?",
            TableauStore._key_params(key),
        ),
        ("SELECT * FROM snapshots s WHERE s.tid = ? AND s.t = ?", (tid, 5)),
        ("SELECT * FROM traces WHERE tid = ? ORDER BY layers", (tid,)),
        (
            "SELECT * FROM observations WHERE tid = ? AND t = ? AND observable = ? AND version = ? "
            "AND params = ? AND context = ?",
            (tid, 5, "D", "1", "{}", ""),
        ),
        (
            "SELECT * FROM observations o WHERE o.observable = ? AND o.version = ? AND o.params = ?",
            ("D", "1", "{}"),
        ),
    ]
    for sql, args in lookups:
        plan = _plan(db, sql, args)
        assert "SEARCH" in plan and "SCAN" not in plan, (sql, plan)


def test_iterating_snapshots_holds_one_batch_at_a_time(tmp_path):
    m = sup.manifest(
        trials={"graphs": 30, "circuits_per_graph": 4, "circuit_index_stride": 4},
        observables={"uniform_erasure": None, "contiguous_distance": False},
        storage={"graphs_per_unit": 30},
    )
    with sup.build(tmp_path / "big", m) as store:
        total = sum(len(s.joint_blob or b"") for s in store.iter_snapshots(batch_size=10_000))
        assert store.counts()["snapshots"] == 30 * 4 * 2 * 5

        def peak(batch, **filters):
            tracemalloc.start()
            count = 0
            for snap in store.iter_snapshots(batch_size=batch, **filters):
                snap.tableau()
                count += 1
            peak_bytes = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            return count, peak_bytes

        small, peak_small = peak(8, p="0.1", g=0)
        everything, peak_all = peak(8)
        _, peak_bulk = peak(10_000)  # every row in one batch: what an unbounded read holds
        assert (small, everything) == (4 * 5, 1200)
        # reading all 1200 snapshots in batches of 8 needs about what reading 20 needs (the
        # peak does not grow with the store), and a fraction of reading them all at once
        assert peak_all < 1.5 * peak_small + 20_000, (peak_small, peak_all, peak_bulk)
        assert peak_all < peak_bulk / 3, (peak_small, peak_all, peak_bulk, total)


# ---------------------------------------------------------------------------------------------
# never overwrite; validation before commit
# ---------------------------------------------------------------------------------------------


def test_meeting_an_identity_again_compares_instead_of_overwriting(tmp_path):
    with sup.build(tmp_path / "s") as store:
        key = store.trajectory_keys()[0]
        tid = store._tid(key)
        snap = store.snapshot(key, 5)
        joint = snap.tableau()
        before = sup.records(store)
        with store.transaction() as tx:  # identical content: nothing changes
            tx.put_snapshot(tid, key, 5, status=SNAPSHOT_STORED, num_qubits=12, k=snap.k,
                            joint=joint, phys=snap.physical_tableau(), unit_id=1)  # fmt: skip
            value = store.observation(key, 5, "D", params={"samples": 12, "exact_budget": 6})
            tx.put_observation(tid, key, 5, "D", {"samples": 12, "exact_budget": 6},
                               value=value["value"], se=value["se"], data=value["data"],
                               method="hybrid", unit_id=1)  # fmt: skip
        sup.assert_unchanged(before, sup.records(store))
        other = store.tableau(store.trajectory_keys()[1], 10)
        cases = [
            lambda tx: tx.put_snapshot(
                tid, key, 5, status=SNAPSHOT_ABSENT, num_qubits=12, k=0, joint=None, phys=None,
                unit_id=1,
            ),  # a stored snapshot never becomes absent
            lambda tx: tx.put_snapshot(tid, key, 5, status=SNAPSHOT_STORED, num_qubits=12,
                                       k=other.physical_code(6).num_logical, joint=other,
                                       phys=other.physical_tableau(6), unit_id=1),
            lambda tx: tx.put_observation(tid, key, 5, "D", {"samples": 12, "exact_budget": 6},
                                          value=99.0, se=0.0, data=None, method="hybrid"),
            lambda tx: tx.put_trajectory(key, u=999, p_float_hex=key.p.hex,
                                         run_seed=[1], graph_digest="x", dynamics_version="d",
                                         unit_id=1, revision=1),
            lambda tx: tx.put_trace(tid, key, layers=10, final_layer=10,
                                    k=np.array([6] * 11), tau=np.array([-1] * 6 + [0]), status="censored",
                                    num_measurements=0, unit_id=1, revision=1),
        ]  # fmt: skip
        for case in cases:
            with (
                pytest.raises(ValueError, match="never overwritten|did not reproduce"),
                store.transaction() as tx,
            ):
                case(tx)
            sup.assert_unchanged(before, sup.records(store))
        assert sup.records(store) == before


def test_invalid_records_are_refused_before_commit(tmp_path):
    with sup.build(tmp_path / "s") as store:
        key = store.trajectory_keys()[0]
        tid = store._tid(key)
        good = store.tableau(key, 10)
        before = sup.records(store)
        rank_deficient = PhaseFreeTableau(good.canonical()[:-1], 12)
        mixed_rows = good.canonical().copy()
        mixed_rows[0, 0] ^= 1  # a row that no longer commutes with the others
        bad_phys = PhaseFreeTableau(good.physical_tableau(6).canonical()[:0], 6)
        cases = [
            (dict(joint=rank_deficient, phys=good.physical_tableau(6), k=6 - good.physical_tableau(6).rank), "pure"),
            (dict(joint=PhaseFreeTableau(mixed_rows, 12), phys=good.physical_tableau(6), k=0), "pure|physical"),
            (dict(joint=good, phys=bad_phys, k=good.physical_code(6).num_logical), "rank n - k"),
            (dict(joint=None, phys=None, k=None), "joint state|physical code"),
        ]  # fmt: skip
        for fields, message in cases:
            with pytest.raises(ValueError, match=message), store.transaction() as tx:
                tx.put_snapshot(tid, key, 3, status=SNAPSHOT_STORED, num_qubits=12, unit_id=1,
                                **fields)  # fmt: skip
        with pytest.raises(ValueError, match="nonincreasing"), store.transaction() as tx:
            tx.put_trace(tid, key, layers=12, final_layer=12, k=np.array([6, 5, 6] + [5] * 10),
                         tau=None, status="censored", num_measurements=0, unit_id=1,
                         revision=1)  # fmt: skip
        with (
            pytest.raises(ValueError, match="absent snapshot stores nothing"),
            store.transaction() as tx,
        ):
            tx.put_snapshot(tid, key, 3, status=SNAPSHOT_ABSENT, num_qubits=12, k=0, joint=good,
                            phys=None, unit_id=1)  # fmt: skip
        sup.assert_unchanged(before, sup.records(store))
        assert sup.records(store) == before


# ---------------------------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------------------------


def _tamper(store, sql, args=()):
    con = sup.open_for_tampering(store)
    con.execute(sql, args)
    con.commit()
    con.close()


def test_altered_blobs_and_digests_are_detected_on_read(tmp_path):
    with sup.build(tmp_path / "s") as store:
        key = store.trajectory_keys()[2]
        tid = store._tid(key)
        assert store.check() == []
        blob = (
            store._db()
            .execute("SELECT joint FROM snapshots WHERE tid = ? AND t = 5", (tid,))
            .fetchone()[0]
        )
        raw = bytearray(zlib.decompress(blob))
        raw[-1] ^= 0x01  # one bit of the last row
        _tamper(
            store,
            "UPDATE snapshots SET joint = ? WHERE tid = ? AND t = 5",
            (zlib.compress(bytes(raw)), tid),
        )
        with pytest.raises(ValueError, match="do not match their digest"):
            store.tableau(key, 5)
        _tamper(store, "UPDATE snapshots SET phys = ? WHERE tid = ? AND t = 2", (b"not zlib", tid))
        with pytest.raises(ValueError, match="corrupt"):
            store.code(key, 2)
        problems = store.check()
        assert any("t = 5" in p for p in problems) and any("t = 2" in p for p in problems)
        edges = store._db().execute("SELECT edges FROM graphs WHERE g = 1").fetchone()[0]
        arr = np.frombuffer(zlib.decompress(edges), dtype="<i4").copy()
        arr[-1] = (arr[-1] + 1) % 6 if arr[-1] + 1 != arr[-2] else arr[-1]
        _tamper(store, "UPDATE graphs SET edges = ? WHERE g = 1", (zlib.compress(arr.tobytes()),))
        with pytest.raises(ValueError, match="graph"):
            store.graph(6, "0.25", 1)


def test_nan_and_infinity_values_round_trip(tmp_path):
    with sup.build(tmp_path / "s") as store:
        key = store.trajectory_keys()[0]
        tid = store._tid(key)
        with store.transaction() as tx:
            tx.put_observation(tid, key, 2, "d_conn", {"max_n": 20}, value=math.inf, se=0.0,
                               data={"note": math.nan}, method="test")  # fmt: skip
            tx.put_observation(tid, key, 5, "d_conn", {"max_n": 20}, value=math.nan, se=0.0,
                               data=None, method="test")  # fmt: skip
        got = store.observations("d_conn")
        assert got["value"][0] == math.inf and math.isnan(got["value"][1])
        assert store.observation(key, 2, "d_conn")["data"] == {"note": None}
        with store.transaction() as tx:  # equal again, NaN included: a no-op
            assert not tx.put_observation(tid, key, 5, "d_conn", {"max_n": 20}, value=math.nan,
                                          se=0.0, data=None, method="test")  # fmt: skip


# ---------------------------------------------------------------------------------------------
# graph records and their measurements (design §12; v0.3 §4)
# ---------------------------------------------------------------------------------------------


def _networkx_measures(edges, n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, np.asarray(edges).tolist()))
    ring = {frozenset((i, (i + d) % n)) for i in range(n) for d in (1, 2)}
    rewired = sum(1 for a, b in G.edges if frozenset((a, b)) not in ring)
    lengths = [
        d for s, dist in nx.all_pairs_shortest_path_length(G) for t, d in dist.items() if s < t
    ]
    components = nx.number_connected_components(G)
    lap = np.diag([d for _, d in sorted(G.degree)]) - nx.to_numpy_array(G, nodelist=range(n))
    lam = np.sort(np.linalg.eigvals(lap).real)
    return {
        "q": rewired / (2 * n),
        "mean_distance": statistics.fmean(lengths) if lengths else math.nan,
        "reachable_pairs": len(lengths),
        "components": components,
        "lambda2": 0.0 if components > 1 else lam[1],
        "degree_variance": statistics.pvariance([d for _, d in G.degree]),
    }


@pytest.mark.parametrize(
    "edges, n",
    [
        (graphs.watts_strogatz(12, 4, 0.0, seed=1), 12),
        (graphs.watts_strogatz(16, 4, 0.3, seed=2), 16),
        (graphs.watts_strogatz(20, 4, 1.0, seed=3), 20),
        (np.array([[0, 1], [1, 2], [3, 4], [4, 5], [3, 5]]), 7),  # two triangles' worth, isolated 6
        (np.array([[0, 1], [1, 2], [2, 3], [3, 4]]), 5),  # a path
    ],
)
def test_graph_measures_against_networkx(edges, n):
    got = graph_measures(edges, n)
    want = _networkx_measures(edges, n)
    assert got["q"] == want["q"] and got["rewired_edges"] == round(want["q"] * 2 * n)
    assert got["components"] == want["components"]
    assert got["reachable_pairs"] == want["reachable_pairs"]
    assert got["mean_distance"] == pytest.approx(want["mean_distance"], rel=1e-15)
    assert got["degree_variance"] == pytest.approx(want["degree_variance"], rel=1e-15, abs=1e-15)
    if want["components"] > 1:
        assert got["lambda2"] == 0.0
    else:
        assert got["lambda2"] == pytest.approx(want["lambda2"], rel=1e-9)


def test_graph_records_are_stored_once_with_their_measurements(built):
    for g in range(3):
        record = built.graph(6, "0.25", g)
        reference = wsref.rewire(6, 0.25, wsref.graph_seed(101, 6, "0.25", g))
        assert np.array_equal(record.edges, reference)
        assert record.regenerate() == record
        measures = built.graph_measures(6, "0.25", g)
        assert measures["version"] == "ws-graph-measures-1"
        for name, value in _networkx_measures(record.edges, 6).items():
            assert measures[name] == pytest.approx(value, rel=1e-9, abs=1e-12)
    rows = built._db().execute("SELECT count(*), count(DISTINCT digest) FROM graphs").fetchone()
    assert rows == (3, 3)
    # every trajectory points at its graph's digest, at both rates
    for key in built.trajectory_keys():
        assert built.trajectory(key).graph_digest == built.graph(6, "0.25", key.g).edge_digest


# ---------------------------------------------------------------------------------------------
# one writer at a time
# ---------------------------------------------------------------------------------------------


def test_a_second_writer_is_refused_with_a_message_naming_the_first(tmp_path):
    import os

    store = sup.build(tmp_path / "s", run=False)
    try:
        with pytest.raises(ValueError, match=rf"this process \(pid {os.getpid()}"):
            TableauStore.open(tmp_path / "s", write=True)
        with TableauStore.open(tmp_path / "s") as reader:  # readers need no lock
            assert reader.revision == 1
            with pytest.raises(ValueError, match="opened for reading"):
                reader.add_revision(sup.manifest(trials={"graphs": 4, "circuits_per_graph": 2,
                                                         "circuit_index_stride": 4}))  # fmt: skip
    finally:
        store.close()
    TableauStore.open(tmp_path / "s", write=True).close()  # released


def test_another_process_holding_the_lock_is_named_and_its_stale_lock_taken_over(tmp_path):
    sup.build(tmp_path / "s", run=False).close()
    code = (
        "from sparsegf2.study.store import TableauStore\n"
        f"s = TableauStore.open({str(tmp_path / 's')!r}, write=True)\n"
        "import os, sys\n"
        "print(os.getpid(), flush=True)\n"
        "sys.stdin.read()\n"
        "os._exit(0)\n"  # dies holding the lock: a stale lock
    )
    import subprocess
    import sys

    proc = subprocess.Popen(
        [sys.executable, "-c", f"import sys; sys.path[:0] = [{str(sup.SRC)!r}]\n" + code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        pid = int(proc.stdout.readline())
        with pytest.raises(ValueError, match=rf"another process \(pid {pid}"):
            TableauStore.open(tmp_path / "s", write=True)
    finally:
        assert proc.stdin is not None
        proc.stdin.close()
        proc.wait(timeout=60)
        assert proc.stdout is not None
        proc.stdout.close()
    assert (tmp_path / "s" / ".lock").exists()  # left behind by the process that ended
    with TableauStore.open(tmp_path / "s", write=True) as store:  # taken over
        assert store.writable


# ---------------------------------------------------------------------------------------------
# revisions
# ---------------------------------------------------------------------------------------------


def test_revisions_must_extend_the_latest(tmp_path):
    with sup.build(tmp_path / "s", run=False) as store:
        assert store.add_revision(sup.manifest()) == 1  # the same manifest: nothing added
        with pytest.raises(ValueError, match="master_circuit_seed"):
            store.add_revision(
                sup.manifest(seeds={"master_graph_seed": 101, "master_circuit_seed": 9})
            )
        more = sup.manifest(
            trials={"graphs": 5, "circuits_per_graph": 2, "circuit_index_stride": 4}
        )
        assert store.add_revision(more) == 2
        assert [m.digest() for m in store.revisions()] == [sup.manifest().digest(), more.digest()]
        with pytest.raises(ValueError, match="graphs"):
            store.add_revision(sup.manifest())  # removing graphs again


# ---------------------------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------------------------


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_export_writes_open_tables_whose_aggregates_reproduce_from_the_rows(tmp_path, built):
    out = built.export(tmp_path / "export")
    assert json.loads((out / "manifest.json").read_text()) == built.manifest.to_dict()
    versions = json.loads((out / "versions.json").read_text())
    assert versions["logical_digest"] == built.logical_digest()
    conditions = _read_csv(out / "conditions.csv")
    assert [(c["protocol"], c["p"], c["trajectories"]) for c in conditions] == [
        ("F", "0.1", "6"),
        ("F", "0.3", "6"),
    ]
    folder = out / conditions[0]["directory"]
    obs = np.load(folder / "observations.npz")
    aggregates = _read_csv(folder / "aggregates.csv")
    for row in aggregates:
        if row["observable"] not in ("D", "k"):
            continue
        sel = (obs["observable"] == row["observable"]) & (obs["t"] == int(row["t"]))
        est = stats.nested_mean(obs["g"][sel], obs["c"][sel], obs["r"][sel], obs["value"][sel])
        assert float(row["mean"]) == float(est.mean)
        assert row["se"] == repr(float(est.se))
    # the aggregate is the equal-graph-weight mean, by hand
    d5 = [r for r in aggregates if r["observable"] == "D" and r["t"] == "5"][0]
    sel = (obs["observable"] == "D") & (obs["t"] == 5)
    graph_means = [obs["value"][sel][obs["g"][sel] == g].mean() for g in range(3)]
    assert float(d5["mean"]) == pytest.approx(statistics.fmean(graph_means), rel=1e-15)
    fp = np.load(folder / "first_passage.npz")
    censoring = _read_csv(folder / "censoring.csv")
    assert len(censoring) == 7 and int(censoring[0]["censored"]) == int((fp["tau"][:, 0] < 0).sum())
    trajectories = _read_csv(folder / "trajectories.csv")
    assert [(int(t["g"]), int(t["c"])) for t in trajectories] == [
        (g, c) for g in range(3) for c in range(2)
    ]
    # two exports of one store are identical, file by file
    again = built.export(tmp_path / "again")
    files = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(again) for p in again.rglob("*") if p.is_file())
    for rel in files:
        if rel.suffix == ".npz":
            a, b = np.load(out / rel), np.load(again / rel)
            assert sorted(a.files) == sorted(b.files)
            assert all(
                np.array_equal(a[f], b[f], equal_nan=a[f].dtype.kind == "f") for f in a.files
            )
        else:
            assert (out / rel).read_bytes() == (again / rel).read_bytes(), rel
    with pytest.raises(ValueError, match="exists"):
        built.export(out)


def test_an_interrupted_export_leaves_nothing(tmp_path, built, monkeypatch):
    from sparsegf2.study import store as store_module

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(store_module, "_export_condition", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        built.export(tmp_path / "export")
    assert list(tmp_path.iterdir()) == []


def test_the_logical_digest_ignores_compression_and_provenance(tmp_path):
    a = sup.build(
        tmp_path / "a", sup.manifest(storage={"graphs_per_unit": 2, "compression_level": 0})
    )
    b = sup.build(
        tmp_path / "b", sup.manifest(storage={"graphs_per_unit": 3, "compression_level": 9})
    )
    try:
        assert a.logical_digest(manifests=False) == b.logical_digest(manifests=False)
        assert a.logical_digest() != b.logical_digest()  # the manifests differ
        size = "SELECT sum(length(joint)) FROM snapshots"
        assert a._db().execute(size).fetchone()[0] > b._db().execute(size).fetchone()[0]
        assert len(a.units()) != len(b.units())
    finally:
        a.close()
        b.close()


def test_a_trajectory_key_of_another_experiment_is_refused(built):
    other = TrajectoryKey("other-experiment", 6, "0.25", "0.1", 0, 0)
    with pytest.raises(ValueError, match="belongs to experiment"):
        built.tableau(other, 2)
