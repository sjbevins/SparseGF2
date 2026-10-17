"""The experiment runner of the Watts–Strogatz study (design §4, §12; v1.4 §7.2–§7.4, §8.1).

The protocol dimension (manifest and key), the requirements and the plan, the engine calls,
failed units, stores that keep codes only, protocol P and its pairing with F (on exported
circuits, gate by gate and measurement by measurement, against the independent draw
transcription of ``tests/_ws_reference.py``), retrospective observables, replay (against Stim on
the exported circuit), the outcome-aware replay, verification and the circuit export.
"""

from __future__ import annotations

import copy
import json
import math
import zlib

import numpy as np
import pytest

import _study_store_support as sup
import _ws_reference as wsref
from _rowspace_oracle import int_rows, rank
from sparsegf2 import mipt
from sparsegf2.study import runner
from sparsegf2.study.identity import Token, TrajectoryKey
from sparsegf2.study.manifest import (
    ExperimentManifest,
    ProtocolSettings,
    check_extension,
    effective_checkpoints,
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    store = sup.build(tmp_path_factory.mktemp("runner") / "store")
    yield store
    store.close()


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    """Protocols F and P, one rate, with stop_when_purified for F (P never stops)."""
    m = sup.manifest(
        protocols={"F": {}, "P": {"checkpoints": {"list": [3, 7]}}},
        conditions=[{"n": [6], "beta": ["0.25"], "p": ["0.2"]}],
        dynamics={"stop_when_purified": True},
    )
    store = sup.build(tmp_path_factory.mktemp("paired") / "store", m)
    yield store
    store.close()


# ---------------------------------------------------------------------------------------------
# the protocol dimension (design §12) in the key and the manifest
# ---------------------------------------------------------------------------------------------


def test_the_trajectory_key_carries_the_protocol():
    f = TrajectoryKey("e", 16, "0.1", "1/3", 2, 1)
    p = TrajectoryKey("e", 16, "0.1", "1/3", 2, 1, 0, "P")
    assert f.protocol == "F" and str(f) == "e:n=16:beta=0.1:p=1/3:g=2:c=1:r=0"
    assert str(p) == "e:protocol=P:n=16:beta=0.1:p=1/3:g=2:c=1:r=0"
    assert TrajectoryKey.parse(str(p)) == p and TrajectoryKey.parse(str(f)) == f
    assert f != p and len({f, p}) == 2 and sorted([p, f]) == [f, p]
    for text in (
        "e:protocol=F:n=16:beta=0.1:p=1/3:g=2:c=1:r=0",  # an F key has no protocol part
        "e:protocol=Q:n=16:beta=0.1:p=1/3:g=2:c=1:r=0",
        "e:protocol=P:n=16:beta=0.1:p=1/3:g=2:c=1",
    ):
        with pytest.raises(ValueError, match="trajectory key"):
            TrajectoryKey.parse(text)
    with pytest.raises(ValueError, match="protocol must be one of"):
        TrajectoryKey("e", 16, "0.1", "1/3", 2, 1, 0, "Q")


def test_the_manifest_protocols_and_their_canonical_form():
    plain = sup.manifest()
    assert plain.protocol_names == ("F",) and "protocols" not in plain.to_dict()
    assert sup.manifest(protocols=["F"]).digest() == plain.digest()  # the default is not written
    both = sup.manifest(protocols=["P", "F"])
    assert both.protocol_names == ("F", "P")
    assert sup.manifest(protocols={"F": {}, "P": {}}).digest() == both.digest()
    assert both.to_dict()["protocols"] == {"F": {}, "P": {"checkpoints": None}}
    again = ExperimentManifest.from_json(both.to_json())
    assert again.digest() == both.digest() and again.protocols == both.protocols
    own = sup.manifest(protocols={"F": {}, "P": {"checkpoints": {"list": [4]}}})
    assert own.protocol_settings("P") == ProtocolSettings(
        "P", own.protocol_settings("P").checkpoints
    )
    assert own.checkpoints(6, "0.25", "P") == (0, 1, 4, 10)
    assert own.checkpoints(6, "0.25") == (0, 1, 2, 5, 10)
    bad = [
        ({"protocols": ["F", "F"]}, "appears twice"),
        ({"protocols": ["Q"]}, "protocol"),
        ({"protocols": {"F": {"checkpoints": {"count": 3}}}}, "F takes no settings"),
        ({"protocols": {"P": {"layers": 3}}}, "unknown key"),
        ({"protocols": []}, "at least one"),
        ({"protocols": {"F": {}, "P": {"checkpoints": {"table": {"6": [11]}}}}}, "exceeds T_max"),
        ({"resources": {"max_unit_bytes": 0}}, "max_unit_bytes"),
    ]
    for change, message in bad:
        data = copy.deepcopy(sup.BASE)
        data.update(change)
        with pytest.raises(ValueError, match=message):
            ExperimentManifest.from_json(data)
    limited = sup.manifest(resources={"max_unit_bytes": 1000})
    assert limited.resources.max_unit_bytes == 1000 and limited.digest() != plain.digest()
    assert limited.to_dict()["resources"]["max_unit_bytes"] == 1000


def test_protocol_p_runs_the_same_circuits_without_the_reference():
    m = sup.manifest(protocols=["F", "P"], dynamics={"stop_when_purified": True})
    f = m.monitored_circuit_kwargs(6, "0.25", "0.1")
    p = m.monitored_circuit_kwargs(6, "0.25", "0.1", "P")
    assert f["reference"] == "purification" and f["stop_when_purified"] is True
    assert p["reference"] == "none" and p["stop_when_purified"] is False
    assert {k: v for k, v in f.items() if k not in ("reference", "stop_when_purified")} == {
        k: v for k, v in p.items() if k not in ("reference", "stop_when_purified")
    }
    assert m.trajectory_key(6, "0.25", "0.1", 1, 1, "P").protocol == "P"
    with pytest.raises(ValueError, match="not run"):
        sup.manifest().trajectory_key(6, "0.25", "0.1", 0, 0, "P")


def test_effective_checkpoints_and_revisions_per_protocol():
    one = sup.manifest()
    two = sup.manifest(protocols={"F": {}, "P": {"checkpoints": {"list": [3]}}})
    three = sup.manifest(
        protocols={"F": {}, "P": {"checkpoints": {"list": [7]}}},
        depth={"layers": {"fixed": 10}, "checkpoints": {"list": [2, 5, 8]}},
    )
    assert check_extension(one, two) is None  # adding a protocol is an extension
    assert "protocol 'P'" in check_extension(two, one)  # removing one is not
    assert effective_checkpoints([one, two, three], 6, "0.25", "P") == (0, 1, 3, 7, 10)
    assert effective_checkpoints([one, two, three], 6, "0.25") == (0, 1, 2, 5, 8, 10)
    with pytest.raises(ValueError, match="not run"):
        effective_checkpoints([one], 6, "0.25", "P")


# ---------------------------------------------------------------------------------------------
# requirements and the plan
# ---------------------------------------------------------------------------------------------


def test_requirements_name_what_a_trajectory_records():
    revisions = [sup.manifest()]
    req = runner.requirements(revisions, "F", 6, "0.25")
    assert req["layers"] == 10 and req["checkpoints"] == [0, 1, 2, 5, 10]
    assert req["snapshot"] == "tableau_snapshot" and req["contiguous_distance"]
    assert req["uniform_erasure"] == {"samples": 12, "exact_budget": 6}
    assert [e["name"] for e in req["correlated_erasure"]] == ["ring"]
    assert "peak" not in req and "domain" not in req  # derived, not simulated (T3-F01)
    digest = runner.requirements_digest(req)
    derived = runner.requirements_digest(runner.derivation(revisions, "F", 6, "0.25"))
    same = [sup.manifest(storage={"graphs_per_unit": 1, "compression_level": 1})]
    assert runner.requirements_digest(runner.requirements(same, "F", 6, "0.25")) == digest
    for other in (
        sup.manifest(depth={"layers": {"fixed": 12}, "checkpoints": {"list": [2, 5]}}),
        sup.manifest(observables={**sup.BASE["observables"], "contiguous_distance": False}),
    ):
        assert runner.requirements_digest(runner.requirements([other], "F", 6, "0.25")) != digest
    # the analysis rule and a schedule with the same depths change what is derived, never what
    # is simulated: no circuit runs again for them
    narrower = sup.manifest(depth={"layers": {"fixed": 10}, "checkpoints": {"list": [2]}})
    for chain in (
        [sup.manifest(analysis={"peak": {"z": 2.0, "bootstrap": 5}})],
        [sup.manifest(analysis={"peak": {"bootstrap": 0}})],
        [sup.manifest(), narrower],
    ):
        assert runner.requirements_digest(runner.requirements(chain, "F", 6, "0.25")) == digest
        assert runner.requirements_digest(runner.derivation(chain, "F", 6, "0.25")) != derived
    narrowed = runner.derivation([sup.manifest(), narrower], "F", 6, "0.25")
    assert narrowed["domain"] == [0, 1, 2, 10] and narrowed["layers"] == 10
    assert runner.derivation(revisions, "P", 6, "0.25") is None
    p = runner.requirements([sup.manifest(protocols=["F", "P"])], "P", 6, "0.25")
    assert p == {
        "protocol": "P",
        "n": 6,
        "beta": "0.25",
        "layers": 10,
        "checkpoints": [0, 1, 2, 5, 10],
        "dynamics": "monitored-random-edges-product-state-phase-free-1",
        "snapshot": "tableau_snapshot",
    }


def _cover(units):
    seen: dict[tuple, int] = {}
    for u in units:
        for g, c in u.pairs():
            key = (u.protocol, u.n, u.beta, u.p, g, c)
            seen[key] = seen.get(key, 0) + 1
    return seen


def test_the_plan_covers_every_trajectory_once_within_the_unit_limits(tmp_path):
    m = sup.manifest(
        protocols=["F", "P"],
        trials={"graphs": 5, "circuits_per_graph": 3, "circuit_index_stride": 4},
    )
    with sup.build(tmp_path / "s", m, run=False) as store:
        everything = {
            (proto, 6, Token.parse("0.25"), Token.parse(p), g, c)
            for proto in ("F", "P")
            for p in ("0.1", "0.3")
            for g in range(5)
            for c in range(3)
        }
        for kwargs, check in (
            ({}, lambda u: u.graphs[1] - u.graphs[0] <= 2),
            ({"graphs_per_unit": 1}, lambda u: u.graphs[1] - u.graphs[0] == 1),
            ({"max_unit_trajectories": 2}, lambda u: u.trajectories <= 2),
            ({"max_unit_bytes": 1}, lambda u: u.trajectories == 1),
        ):
            units = runner.plan(store, **kwargs)
            cover = _cover(units)
            assert set(cover) == everything and set(cover.values()) == {1}, kwargs
            assert all(check(u) for u in units), kwargs
            order = [(u.protocol, u.n, u.beta, u.p, u.graphs[0], u.circuits[0]) for u in units]
            assert order == sorted(order) and units == runner.plan(store, **kwargs)
        assert runner.plan(store)[0].protocol == "F" and runner.plan(store)[-1].protocol == "P"
        runner.run(store, max_units=3)
        rest = runner.plan(store)
        assert len(rest) == len(runner.plan(store)) and set(_cover(rest)) < everything
        runner.run(store)
        assert runner.plan(store) == []


def test_one_engine_call_per_unit_on_its_indices_with_the_v14_protocol(tmp_path, monkeypatch):
    calls = []
    original = mipt.MonitoredCircuit.run

    def spy(self, *args, **kwargs):
        calls.append((self, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(mipt.MonitoredCircuit, "run", spy)
    m = sup.manifest()
    with sup.build(tmp_path / "s", m, run=False) as store:
        units = runner.plan(store)
        runner.run(store)
        assert len(calls) == len(units) == 4
        for (spec, kwargs), unit in zip(calls, units, strict=True):
            assert np.array_equal(kwargs["indices"], unit.indices(4))
            assert np.array_equal(
                kwargs["indices"], [4 * g + c for g, c in unit.pairs()]
            )  # u = g * stride + c
            assert kwargs["seed"] == list(wsref.run_seed(202, 6, "0.25"))
            assert spec.num_system == 6 and spec.layers == 10 and spec.circuits_per_graph == 4
            assert spec.measurement_rate == float(Token.parse(unit.p).value)
            assert spec.reference == "purification" and spec.schedule == "random_edges"
            assert spec.gates_per_layer == 3 and spec.random_orientation is False
            assert (spec.scramble_gates, spec.scramble_matchings, spec.scramble_layers) == (0, 0, 0)
            assert spec.checkpoints == "every_layer" and spec.stop_when_purified is False
            assert len(spec.graphs) == unit.graphs[1]  # the collection 0 .. g1 - 1
            for g in range(unit.graphs[1]):
                assert np.array_equal(
                    spec.graphs[g], wsref.rewire(6, 0.25, wsref.graph_seed(101, 6, "0.25", g))
                )
            names = {p.name: p for p in spec.probes}
            assert names["code_dimension"].checkpoints is None  # every layer, the global schedule
            for name in (
                "tableau_snapshot",
                "contiguous_distance",
                "erasure_first_failure",
                "uncorrectable_region_counts",
                "ring",
            ):
                assert list(spec.probe_layers(name)) == [0, 1, 2, 5, 10], name


def test_a_failed_unit_writes_no_data_and_is_recorded(tmp_path, monkeypatch):
    with sup.build(tmp_path / "s", run=False) as store:
        runner.run(store, max_units=1)
        before = sup.records(store)

        def fail(unit):
            raise RuntimeError("injected after every insert, before the commit")

        monkeypatch.setattr(runner, "_BEFORE_COMMIT", fail)
        with pytest.raises(RuntimeError, match="injected"):
            runner.run(store)
        assert sup.records(store) == before  # nothing of the failed unit
        failed = store.units(status="failed")
        assert len(failed) == 1 and "RuntimeError: injected" in failed[0]["error"]
        assert failed[0]["trajectories"] == 2 and failed[0]["graphs"] == [2, 3]
        monkeypatch.setattr(runner, "_BEFORE_COMMIT", None)
        summary = runner.run(store)
        assert summary.units == 3 and summary.remaining_units == 0
        assert store.counts()["trajectories"] == 12 and store.check() == []


# ---------------------------------------------------------------------------------------------
# stores that keep the physical code only; switching the joint tableaux on
# ---------------------------------------------------------------------------------------------


def test_a_code_only_store_and_a_revision_that_adds_the_joint_tableaux(tmp_path):
    obs = {**sup.BASE["observables"], "tableau_snapshots": False}
    with sup.build(tmp_path / "s", sup.manifest(observables=obs)) as store:
        key = store.trajectory_keys()[0]
        snap = store.snapshot(key, 5)
        assert snap.joint_blob is None and snap.phys_blob is not None
        with pytest.raises(ValueError, match="physical code only"):
            store.tableau(key, 5)
        code = store.code(key, 5)
        before = sup.records(store)
        runner.extend(store, sup.manifest())  # tableau_snapshots on: every trajectory reruns
        runner.run(store)
        after = sup.records(store)
        for k, v in before.items():
            if k[0] == "snapshot":  # the physical code stays; the joint state is filled in
                new = after[k]
                status, qubits, rank_, k_, joint, joint_digest, phys, phys_digest = v
                assert (new[1], new[3], new[6], new[7]) == (qubits, k_, phys, phys_digest)
                assert (status, new[0]) == ("phys_only", "stored")  # G-STO-3, then filled in
                assert (rank_, joint, joint_digest) == (None, None, None)
                assert new[2] == 12 and new[4] is not None and new[5] is not None
            elif k[0] not in ("manifest",):
                assert after[k] == v, k
        joint = store.tableau(key, 5)
        assert joint.physical_tableau(6).same_space(store.snapshot(key, 5).physical_tableau())
        assert store.code(key, 5).num_logical == code.num_logical
        assert store.check() == []


# ---------------------------------------------------------------------------------------------
# protocol P (design §12)
# ---------------------------------------------------------------------------------------------


def _layers_of(circuit):
    return wsref.parse_circuit(str(circuit))


def test_p_and_f_replay_the_same_gate_and_mask_history(paired):
    n, p = 6, Token.parse("0.2")
    for key in paired.trajectory_keys(protocol="F"):
        partner = TrajectoryKey(key.experiment_id, n, key.beta, key.p, key.g, key.c, 0, "P")
        f_rec, p_rec = paired.trajectory(key), paired.trajectory(partner)
        assert (f_rec.u, f_rec.run_seed, f_rec.graph_digest) == (
            p_rec.u,
            p_rec.run_seed,
            p_rec.graph_digest,
        )
        f_circuit = runner.circuit(paired, key, 10)
        p_circuit = runner.circuit(paired, partner, 10)
        assert str(p_circuit).splitlines()[0] == "TICK"  # |0...0>: no setup
        assert str(f_circuit).splitlines()[:2] == [
            "H 0 1 2 3 4 5",
            "CX 0 6 1 7 2 8 3 9 4 10 5 11",
        ]
        f_layers, p_layers = _layers_of(f_circuit), _layers_of(p_circuit)
        assert len(f_layers) == len(p_layers) == 10
        edges = paired.graph(n, key.beta, key.g).edges
        draws = wsref.study_draws(list(wsref.run_seed(202, n, "0.25")), f_rec.u, len(edges), n, 10)
        for t, (lf, lp) in enumerate(zip(f_layers, p_layers, strict=True)):
            assert lf.gates == lp.gates and lf.measured == lp.measured  # gate by gate, mask by mask
            expected = [
                (int(edges[e][0]), int(edges[e][1]), int(j))
                for e, j in zip(draws.edge_index[t], draws.clifford[t], strict=True)
            ]
            assert lf.gates == expected
            assert lf.measured == [i for i in range(n) if draws.uniform[t, i] < p.float64]


def test_p_snapshots_are_the_pure_system_state(paired):
    keys = paired.trajectory_keys(protocol="P")
    assert len(keys) == 6 and all(k.protocol == "P" for k in keys)
    for key in keys[:3]:
        assert paired.depths(key) == (0, 1, 3, 7, 10)
        trace = paired.trace(key)
        assert trace.status == "no_reference" and trace.k is None and trace.final_layer == 10
        states = sup.stim_states(runner.circuit(paired, key), 6, seed=3)
        for t in paired.depths(key):
            tab = paired.tableau(key, t)
            assert tab.num_qubits == 6 and tab.is_pure()
            snap = paired.snapshot(key, t)
            assert snap.k is None and snap.phys_blob is None
            assert snap.joint_digest == sup.canonical_digest(states[t], 6)
            # a pure state: S(A) = S(complement of A)
            assert tab.entropy([0, 1]) == tab.entropy([2, 3, 4, 5])
        with pytest.raises(ValueError, match="no reference"):
            paired.code(key, 3)
        with pytest.raises(ValueError, match="no k trace"):
            paired.k_trace(key)
    same = paired.tableau(n=6, beta="0.25", p="0.2", g=1, c=1, t=7, protocol="P")
    assert same == paired.tableau(keys[3], 7)


# ---------------------------------------------------------------------------------------------
# retrospective observables
# ---------------------------------------------------------------------------------------------


def test_compute_reads_stored_snapshots_only_and_is_idempotent(tmp_path, monkeypatch):
    with sup.build(tmp_path / "s") as store:
        before = sup.records(store)

        def no_simulation(*args, **kwargs):
            raise AssertionError("compute must not simulate")

        monkeypatch.setattr(mipt.MonitoredCircuit, "run", no_simulation)
        first = runner.compute(store, "half_cut_entropy")
        assert (first.inserted, first.existing, first.snapshots) == (60, 0, 60)
        again = runner.compute(store, "half_cut_entropy", batch_size=7)
        assert (again.inserted, again.existing) == (0, 60)
        d = runner.compute(store, "d_cyc", n=6, p="0.3")  # stored by the run: verified
        assert (d.inserted, d.existing, d.snapshots) == (0, 30, 30)
        sup.assert_unchanged(before, sup.records(store))
        for key in store.trajectory_keys()[:4]:
            for t in store.depths(key):
                half = store.observation(key, t, "half_cut_entropy")["value"]
                assert half == store.tableau(key, t).entropy([0, 1, 2])
        with pytest.raises(ValueError, match="not from a stored snapshot"):
            runner.compute(store, "first_passage")


def test_recomputing_the_run_s_values_from_stored_codes_cross_checks_them(tmp_path):
    with sup.build(tmp_path / "s") as store:
        for name in ("d_cyc", "k"):  # the run stored both from the engine and the k trace
            summary = runner.compute(store, name)
            assert (summary.inserted, summary.existing) == (0, 60), name
        key = store.trajectory_keys()[3]
        con = sup.open_for_tampering(store)
        con.execute(
            "UPDATE observations SET value = 99 WHERE observable = 'k' AND t = 5 AND tid = ?",
            (store._tid(key),),
        )
        con.commit()
        con.close()
        with pytest.raises(ValueError, match="'k'.*another value"):
            runner.compute(store, "k")


# ---------------------------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------------------------


def test_replay_reproduces_the_stored_digests_and_gives_unsaved_depths(built):
    for key in built.trajectory_keys()[::5]:
        stored = list(built.depths(key))
        replayed = runner.replay(built, key, [3, 4, 13, *stored])
        assert replayed.verified_depths == tuple(stored)
        assert replayed.verified_layers == 10
        assert np.array_equal(replayed.k[:11], built.k_trace(key))
        states = sup.stim_states(runner.circuit(built, key, 13), 12, seed=11)
        for t in (3, 4, 13, *stored):  # also beyond the horizon T = 10
            assert replayed.tableaux[t].digest() == sup.canonical_digest(states[t], 12)
            assert replayed.tableaux[t].physical_code(6).num_logical == replayed.k[t]


def test_replay_fails_loudly_on_a_store_that_does_not_reproduce(tmp_path):
    with sup.build(tmp_path / "s") as store:
        key = store.trajectory_keys()[4]
        tid = store._tid(key)
        con = sup.open_for_tampering(store)
        con.execute(
            "UPDATE snapshots SET joint_digest = ? WHERE tid = ? AND t = 5", ("0" * 64, tid)
        )
        con.commit()
        con.close()
        with pytest.raises(ValueError, match="t = 5 does not match the stored digest"):
            runner.replay(store, key, [2, 5])
        assert runner.replay(store, key, [5], verify=False).verified_depths == ()


def test_the_outcome_aware_replay_drops_its_signs_to_the_stored_row_space(built):
    key = built.trajectory_keys()[7]
    depths = list(built.depths(key))
    a = runner.replay_signed(built, key, depths, outcome_seed=5)
    b = runner.replay_signed(built, key, depths, outcome_seed=5)
    c = runner.replay_signed(built, key, depths, outcome_seed=6)
    assert a.verified_depths == tuple(depths) == c.verified_depths
    assert np.array_equal(a.measurements, b.measurements)
    for t in depths:
        assert all(np.array_equal(x, y) for x, y in zip(a.signed[t], b.signed[t], strict=True))
        assert a.tableaux[t] == c.tableaux[t] == built.tableau(key, t)
        xs, zs, signs = a.signed[t]
        unsigned = np.concatenate([xs, zs], axis=1).astype(np.uint8)
        assert rank(int_rows(unsigned)) == 12  # n Bell-pair partners: a pure state of 12 qubits
        assert built.tableau(key, t).same_space(type(built.tableau(key, t))(unsigned, 12))
    total = sum(
        len(layer.measured) for layer in wsref.parse_circuit(str(runner.circuit(built, key)))
    )
    assert a.measurements.size == total == built.trace(key).num_measurements


# ---------------------------------------------------------------------------------------------
# verification and circuit export (v1.4 §4 "Tracking", §7.4; R189)
# ---------------------------------------------------------------------------------------------


def test_verify_passes_on_a_clean_store_and_fails_on_a_tampered_one(tmp_path):
    with sup.build(tmp_path / "s") as store:
        clean = runner.verify(store, None)
        assert clean.ok and clean.trajectories == 12 and clean.snapshots == 60
        sampled = runner.verify(store, 3, seed=1)
        assert sampled.ok and sampled.trajectories == 3
        key = store.trajectory_keys()[9]
        tid = store._tid(key)
        blob = (
            store._db()
            .execute("SELECT joint FROM snapshots WHERE tid = ? AND t = 10", (tid,))
            .fetchone()[0]
        )
        raw = bytearray(zlib.decompress(blob))
        raw[20] ^= 0x40
        con = sup.open_for_tampering(store)
        con.execute(
            "UPDATE snapshots SET joint = ? WHERE tid = ? AND t = 10",
            (zlib.compress(bytes(raw)), tid),
        )
        con.commit()
        con.close()
        report = runner.verify(store, None)
        assert not report.ok and len(report.failures) == 1
        assert str(key) in report.failures[0] and "digest" in report.failures[0]


def test_the_exported_circuit_of_a_stored_trajectory(built):
    n = 6
    for key in built.trajectory_keys()[::4]:
        rec = built.trajectory(key)
        circuit = runner.circuit(built, key)
        layers = wsref.parse_circuit(str(circuit))
        assert len(layers) == 10
        edges = built.graph(n, key.beta, key.g).edges
        draws = wsref.study_draws(list(rec.run_seed), rec.u, len(edges), n, 10)
        for t, layer in enumerate(layers):
            assert [(a, b) for a, b, _ in layer.gates] == [
                tuple(edges[e]) for e in draws.edge_index[t]
            ]
            assert [j for _, _, j in layer.gates] == draws.clifford[t].tolist()
            assert layer.measured == [i for i in range(n) if draws.uniform[t, i] < key.p.float64]
            assert all(a < b for a, b, _ in layer.gates)  # the low endpoint first (v1.4 §9)
            assert len(layer.gates) == n // 2
        assert sum(len(x.measured) for x in layers) == built.trace(key).num_measurements


def test_run_summaries_and_progress(tmp_path):
    seen = []
    with sup.build(tmp_path / "s", run=False) as store:
        summary = runner.run(
            store, progress=lambda u, i, total, info: seen.append((i, total, info))
        )
        assert summary.units == 4 and summary.trajectories == 12 and summary.snapshots == 60
        assert summary.remaining_units == 0
        assert [(i, total) for i, total, _ in seen] == [(i, 4) for i in range(4)]
        assert all(info["trajectories"] == 4 or info["trajectories"] == 2 for *_, info in seen)
        units = store.units(status="done")
        assert [u["graphs"] for u in units] == [[0, 2], [2, 3], [0, 2], [2, 3]]
        assert all(u["git_commit"] is None or len(u["git_commit"]) == 40 for u in units)
        status = store.status()
        assert status["units_done"] == 4 and status["units_remaining"] == 0
        assert json.loads(json.dumps(status)) == status
        assert not math.isnan(summary.elapsed_seconds)
