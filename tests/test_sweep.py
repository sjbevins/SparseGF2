"""The sweep layer (``sparsegf2.study.sweep``; ``docs/SWEEPS.md``).

Every test runs at fixed seeds and asserts nothing about time. A sweep's saved trajectories are
compared bitwise with what the engine gives for the same spec, seed and stream indices, across
chunk sizes, interruptions, thread counts and growth of the sweep.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import itertools
import json
import zipfile
from decimal import Decimal, localcontext

import numba
import numpy as np
import pytest
import stim

import _ws_reference as wsref
import sparsegf2
from _rowspace_oracle import bits, canonical, int_rows, stim_tick_states
from _rowspace_oracle import rank as oracle_rank
from sparsegf2 import PhaseFreeTableau, api, cli, graphs, mipt
from sparsegf2.clifford import Clifford
from sparsegf2.study import identity as ident
from sparsegf2.study import sweep
from sparsegf2.study.identity import Token
from sparsegf2.study.sweep import SweepSpec, geometric_checkpoints, round_half_up_power

SPEC = {
    "sizes": [6, 8],
    "betas": ["0", "1/5"],
    "rates": ["0.1", "0.3"],
    "graphs": 2,
    "circuits_per_graph": 2,
    "stride": 4,
    "layers": {"per_n": 2},
    "checkpoints": {"count": 4},
    "master_graph_seed": 17,
    "master_circuit_seed": 29,
}


def spec_with(**changes) -> dict:
    return {**copy.deepcopy(SPEC), **changes}


def assert_same(a: mipt.EnsembleResult, b: mipt.EnsembleResult) -> None:
    """Bitwise equality of two results' trajectories (every array, every snapshot word)."""
    assert np.array_equal(a.trajectories, b.trajectories)
    assert list(a.values) == list(b.values)
    for k in a.values:
        assert np.array_equal(a.values[k], b.values[k], equal_nan=True), k
    for k in a.snapshots:
        assert np.array_equal(a.snapshots[k][0], b.snapshots[k][0]), k
        assert np.array_equal(a.snapshots[k][1], b.snapshots[k][1]), k
    for name in ("purification_layer", "final_layer", "num_measurements"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.seed_entropy == b.seed_entropy


def every_condition(directory) -> dict:
    spec = sweep._spec_of(directory)
    return {(n, str(b), str(p)): sweep.load(directory, n, b, p) for n, b, p in spec.conditions()}


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """The sweep run in one call with the default chunks."""
    directory = tmp_path_factory.mktemp("sweep") / "one"
    sweep.run(SPEC, directory)
    return directory, every_condition(directory)


# ---------------------------------------------------------------------------
# the spec
# ---------------------------------------------------------------------------


def test_the_spec_is_canonical_and_resolves_per_size_values():
    spec = SweepSpec.from_json(SPEC)
    canon = spec.to_json()
    assert canon["rates"] == {"6": ["0.1", "0.3"], "8": ["0.1", "0.3"]}
    assert canon["layers"] == {"6": 12, "8": 16} and canon["betas"] == ["0", "0.2"]
    assert canon["probes"] == [dict(p) for p in sweep.DEFAULT_PROBES]
    assert canon["picture"] == "purification" and canon["stop_when_purified"] is False
    assert SweepSpec.from_json(canon) == spec and SweepSpec.from_json(canon).digest == spec.digest
    assert spec.digest == ident.json_digest(canon)
    # another spelling of the same sweep has the same digest
    same = spec_with(betas=[0.2, "0"], rates=["3/10", ".1"], layers={"6": 12, "8": 16})
    assert SweepSpec.from_json(same).digest == spec.digest
    first = (6, Token.parse("0"), Token.parse("0.1"))
    assert len(spec.conditions()) == 8 and spec.conditions()[0] == first
    assert spec.indices(6).tolist() == [0, 1, 4, 5]  # u = g * stride + c
    assert spec.checkpoints_at(8) == geometric_checkpoints(16, 4) == (0, 1, 3, 6, 16)
    assert SweepSpec.from_json(spec_with(checkpoints={"list": [40, 5, 5]})).checkpoints_at(6) == (
        0,
        1,
        5,
        12,
    )


def test_the_engine_spec_is_the_protocol_of_v1_4():
    spec = SweepSpec.from_json(spec_with(probes=[{"kind": "code_dimension"}, {"kind": "tableau_snapshot", "checkpoints": [2]}]))  # fmt: skip
    engine = spec.engine_spec(8, "0.2", "1/3")
    assert engine.pop("probes") == [
        {"kind": "code_dimension", "checkpoints": [0, 1, 3, 6, 16]},  # the sparse rule filled in
        {"kind": "tableau_snapshot", "checkpoints": [2]},
    ]
    assert engine == {
        "num_system": 8,
        "circuits_per_graph": 4,
        "layers": 16,
        "measurement_rate": 1 / 3,
        "reference": "purification",
        "schedule": "random_edges",
        "gates_per_layer": 4,
        "random_orientation": False,
        "scramble_gates": 0,
        "scramble_matchings": 0,
        "scramble_layers": 0,
        "periodic": True,
        "checkpoints": "every_layer",
        "stop_when_purified": False,
    }
    assert spec.run_seed(8, "0.2") == ident.run_seed(29, 8, "0.2", "shared")


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"sizs": [6]}, "unknown key"),
        ({"sizes": [4]}, "sizes"),
        ({"sizes": [6, 6]}, "appears twice"),
        ({"betas": ["0.1", "1/10"]}, "betas: a value appears twice"),
        ({"rates": ["1.5"]}, "rates"),
        ({"rates": {"6": ["0.1"]}}, "one entry per size"),
        ({"graphs": 0}, "graphs"),
        ({"circuits_per_graph": 5}, "stride"),
        ({"layers": 12}, "layers"),
        ({"checkpoints": {"count": 0}}, "checkpoints.count"),
        ({"master_graph_seed": 2**32}, "master_graph_seed"),
        ({"probes": [{"kind": "nope"}]}, "unknown probe kind"),
        ({"probes": [{"kind": "code_dimension"}] * 2}, "unique"),
        ({"picture": "pure"}, "picture"),
        ({"picture": "none", "stop_when_purified": True}, "needs a reference"),
    ],
)
def test_the_spec_refuses_bad_fields_with_their_names(change, fragment):
    with pytest.raises(ValueError, match=fragment) as info:
        SweepSpec.from_json(spec_with(**change))
    assert str(info.value).startswith("sweep spec:")


# ---------------------------------------------------------------------------
# the checkpoint rule (v1.4 section 9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "t_max, count, expected",
    [
        (128, 8, [0, 1, 2, 4, 8, 16, 32, 64, 128]),
        (64, 4, [0, 1, 4, 16, 64]),
        (1000, 4, [0, 1, 10, 100, 1000]),
        (10, 4, [0, 1, 2, 5, 10]),  # 10^(1/3) = 2.154, 10^(2/3) = 4.642
        (13, 3, [0, 1, 4, 13]),  # sqrt 13 = 3.61
        (128, 12, [0, 1, 2, 4, 6, 9, 14, 22, 34, 53, 82, 128]),
        (50, 1, [0, 1, 50]),
        (1, 7, [0, 1]),
    ],
)
def test_geometric_checkpoints_hand_computed(t_max, count, expected):
    assert list(geometric_checkpoints(t_max, count)) == expected


@pytest.mark.parametrize("t_max", [1, 2, 3, 7, 10, 24, 100, 192, 1000, 4096, 10**6])
@pytest.mark.parametrize("count", [1, 2, 3, 5, 12, 33, 100])
def test_geometric_checkpoints_equal_high_precision_arithmetic(t_max, count):
    points = {0, 1, t_max}
    with localcontext() as ctx:
        ctx.prec = 120
        for j in range(count if count >= 2 else 0):
            a = (Decimal(j) * Decimal(t_max).ln() / Decimal(count - 1)).exp()
            points.add(int((a + Decimal("0.5")).to_integral_value(rounding="ROUND_FLOOR")))
    assert list(geometric_checkpoints(t_max, count)) == sorted(points)


def test_the_checkpoint_rule_rounds_ties_upward_and_checks_its_arguments():
    for s, q, expected in [(9, 2, 2), (25, 2, 3), (1, 2, 1), (27, 3, 2), (8, 3, 1)]:
        assert sweep._half_up_root(s, q) == expected
    for bad in [(0, 3), (10, 0), (10, 1001), (True, 3), (10.0, 3)]:
        with pytest.raises(ValueError):
            geometric_checkpoints(*bad)
    with pytest.raises(ValueError, match="need T"):
        round_half_up_power(5, 4, 3)


# ---------------------------------------------------------------------------
# generation: chunks, resumes, threads
# ---------------------------------------------------------------------------


def test_a_sweep_is_the_engine_run_on_its_graph_files(reference):
    directory, results = reference
    spec = SweepSpec.from_json(SPEC)
    for (n, beta, p), result in results.items():
        records = [ident.build_graph(n, beta, g, 17) for g in range(2)]
        mc = spec.monitored_circuit(n, beta, p, [r.edges for r in records])
        direct = mc.run(indices=spec.indices(n), seed=list(spec.run_seed(n, beta)))
        assert_same(result, direct)
        # the condition's spec.json builds the same protocol through the API's graph files
        folder = directory / sweep._name(n, ident.Token.parse(beta), ident.Token.parse(p))
        stored = json.loads((folder / "spec.json").read_text(encoding="utf-8"))
        assert stored["seed"] == list(spec.run_seed(n, beta))
        assert mipt._protocol_difference(api._build_spec(stored["spec"]), result.spec) is None
    status = sweep.status(directory)
    assert status["complete"] and status["saved"] == status["trajectories"] == 8 * 4


def test_chunked_and_interrupted_runs_equal_one_run(reference, tmp_path):
    _, expected = reference
    chunked = tmp_path / "chunked"
    sweep.run(SPEC, chunked, chunk_size=1)
    for key, result in every_condition(chunked).items():
        assert_same(result, expected[key])
    interrupted = tmp_path / "interrupted"
    calls = []

    def stop_after_three(message):
        calls.append(message)
        if len(calls) == 3:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        sweep.run(SPEC, interrupted, chunk_size=3, log=stop_after_three)
    assert not sweep.status(interrupted)["complete"]
    out = sweep.run(SPEC, interrupted, chunk_size=2)  # resumes with another chunk size
    assert out["complete"] and out["ran"] == 32 - (4 + 3)  # 4 + 3 were saved
    for key, result in every_condition(interrupted).items():
        assert_same(result, expected[key])
    assert sweep.run(SPEC, interrupted)["ran"] == 0  # nothing left to do


def test_results_do_not_depend_on_the_thread_count(reference, tmp_path):
    _, expected = reference
    most = numba.config.NUMBA_NUM_THREADS
    if most < 2:
        pytest.skip("this process may use one thread only: there is no other count to compare")
    counts = [1, min(3, most)]  # what run(num_threads=...) uses: at most NUMBA_NUM_THREADS
    assert len({*counts, numba.get_num_threads()}) >= 2  # the reference ran on the default
    for threads in counts:
        directory = tmp_path / f"t{threads}"
        sweep.run(SPEC, directory, num_threads=threads, chunk_size=3)
        for key, result in every_condition(directory).items():
            assert_same(result, expected[key])


# ---------------------------------------------------------------------------
# growth
# ---------------------------------------------------------------------------


def test_growing_the_sweep_keeps_every_saved_trajectory(reference, tmp_path):
    _, before = reference
    grown_spec = spec_with(
        graphs={"6": 3, "8": 2}, circuits_per_graph=3, rates=["0.1", "0.2", "0.3"], betas=["0", "0.2", "1"]
    )  # fmt: skip
    directory = tmp_path / "grown"
    sweep.run(SPEC, directory)
    out = sweep.run(grown_spec, directory)
    assert out["complete"] and out["ran"] == (3 * 3 + 2 * 3) * 9 - 32
    after = every_condition(directory)
    for key, old in before.items():
        new = after[key]
        keep = np.isin(new.trajectories, old.trajectories)
        assert int(keep.sum()) == old.num_trajectories
        assert_same(sweep._take(new, np.flatnonzero(keep)), old)
    # and the grown directory is exactly a fresh run of the grown spec
    fresh = tmp_path / "fresh"
    sweep.run(grown_spec, fresh)
    for key, result in every_condition(fresh).items():
        assert_same(result, after[key])
    assert json.loads((directory / "sweep.json").read_text())["digest"] == (
        SweepSpec.from_json(grown_spec).digest
    )


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"stride": 8}, "stride"),
        ({"master_graph_seed": 1}, "master_graph_seed"),
        ({"master_circuit_seed": 1}, "master_circuit_seed"),
        ({"picture": "none", "probes": [{"kind": "tableau_snapshot"}]}, "picture"),
        ({"stop_when_purified": True}, "stop_when_purified"),
        ({"probes": [{"kind": "code_dimension"}]}, "probes"),
        ({"checkpoints": {"count": 5}}, "checkpoints"),
        ({"layers": {"per_n": 3}}, "layers at n = 6"),
        ({"sizes": [8]}, "sizes drop"),
        ({"betas": ["0"]}, "betas drop"),
        ({"rates": ["0.1"]}, "rates at n = 6 drop"),
        ({"graphs": 1}, "graphs at n = 6"),
        ({"circuits_per_graph": 1}, "circuits_per_graph at n = 6"),
    ],
)
def test_a_change_that_would_alter_saved_data_is_refused(reference, change, fragment):
    directory, _ = reference
    before = (directory / "sweep.json").read_bytes()
    with pytest.raises(ValueError, match=fragment):
        sweep.run(spec_with(**change), directory)
    assert (directory / "sweep.json").read_bytes() == before


def test_a_directory_of_another_algorithm_or_contract_is_refused(tmp_path):
    directory = tmp_path / "old"
    small = spec_with(sizes=[6], betas=["0"], rates=["0.1"], graphs=1)
    sweep.run(small, directory)
    index = json.loads((directory / "sweep.json").read_text())
    index["versions"]["dynamics"] = "another-dynamics-1"
    (directory / "sweep.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="dynamics"):
        sweep.run(small, directory)
    index["versions"]["randomness_contract"] = 1
    (directory / "sweep.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="randomness contract 1"):
        sweep.load(directory, 6, "0", "0.1")


def made_by_an_older_version(directory, old="3.0.0.dev0") -> None:
    """Rewrite a sweep as an older package version would have left it: every part's version,
    the index's and the provenance's."""
    for path in directory.glob("n*/result-*.npz"):
        part = mipt.EnsembleResult.load(path)
        part.version = old
        part.save(path)
    index_path = directory / "sweep.json"
    index = json.loads(index_path.read_text())
    index["versions"]["sparsegf2"] = old
    index_path.write_text(json.dumps(index))
    for path in directory.glob("n*/provenance.json"):
        record = json.loads(path.read_text())
        for entry in record["parts"].values():
            entry["sparsegf2"] = old
        path.write_text(json.dumps(record))


def test_a_sweep_grows_across_package_versions_after_a_replay_check(reference, tmp_path):
    _, expected = reference
    directory = tmp_path / "older"
    sweep.run(SPEC, directory)
    made_by_an_older_version(directory)
    out = sweep.run(spec_with(graphs=3), directory)  # replays, then grows: 1 graph more each
    assert out["complete"] and out["ran"] == 8 * 2
    folder = directory / "n6_beta0_p0.1"
    versions = {mipt.EnsembleResult.load(p).version for p in folder.glob("result-*.npz")}
    assert versions == {"3.0.0.dev0", sparsegf2.__version__}
    parts = json.loads((folder / "provenance.json").read_text())["parts"]
    assert {entry["sparsegf2"] for entry in parts.values()} == versions
    assert {entry["compression"] for entry in parts.values()} == {"fast"}
    assert len(parts) == len(list(folder.glob("result-*")))
    for key, old in expected.items():  # load reads parts of both versions; old rows unchanged
        new = sweep.load(directory, *key)
        keep = np.isin(new.trajectories, old.trajectories)
        assert_same(sweep._take(new, np.flatnonzero(keep)), old)


def test_a_part_that_does_not_replay_stops_the_growth(tmp_path):
    directory = tmp_path / "altered"
    small = spec_with(sizes=[6], betas=["0"], rates=["0.1"])
    sweep.run(small, directory)
    made_by_an_older_version(directory)
    (path,) = (directory / "n6_beta0_p0.1").glob("result-*.npz")
    part = mipt.EnsembleResult.load(path)
    part.values["code_dimension"][0, 3] += 1  # u = 0, the first saved trajectory
    part.save(path)
    before = (directory / "sweep.json").read_bytes()
    with pytest.raises(ValueError, match="u = 0 does not replay identically.*versions disagree"):
        sweep.run({**small, "graphs": 3}, directory)
    assert (directory / "sweep.json").read_bytes() == before
    assert sweep.status(directory)["saved"] == 4  # nothing was added


# ---------------------------------------------------------------------------
# the coupling across rates, graphs, replay and rows
# ---------------------------------------------------------------------------


def test_gates_are_shared_and_masks_nested_across_rates(reference):
    directory, results = reference
    spec = SweepSpec.from_json(SPEC)
    for n, beta in itertools.product(spec.sizes, spec.betas):
        collection = sweep._collection(directory, spec, n, beta)
        seed = list(spec.run_seed(n, beta))
        for u in spec.indices(n).tolist():
            low, high = (
                wsref.parse_circuit(str(spec.monitored_circuit(n, beta, p, collection).circuit(u, seed=seed)))
                for p in ("0.1", "0.3")
            )  # fmt: skip
            assert [layer.gates for layer in low] == [layer.gates for layer in high]
            assert all(set(a.measured) <= set(b.measured) for a, b in zip(low, high, strict=True))
        # the saved trajectories agree: as many or more measurements at the higher rate
        a, b = results[(n, str(beta), "0.1")], results[(n, str(beta), "0.3")]
        assert np.all(a.num_measurements <= b.num_measurements)
    # a dense grid of 33 rates from 0 to 1 at fixed (n, beta, g, c), 500 layers deep: the same
    # gates at every rate, and the masks nested for every pair of rates, layer by layer
    grid = [f"{j}/32" for j in range(33)]  # increasing
    n, beta = 8, Token.parse("0.2")
    collection = sweep._collection(directory, spec, n, beta)
    seed = list(spec.run_seed(n, beta))
    for u in (1, 4):  # (g, c) = (0, 1) and (1, 0)
        exports = [
            wsref.parse_circuit(
                str(
                    dataclasses.replace(
                        spec.monitored_circuit(n, beta, p, collection), layers=500
                    ).circuit(u, seed=seed)
                )
            )
            for p in grid
        ]
        assert all([x.gates for x in e] == [x.gates for x in exports[0]] for e in exports)
        for low, high in itertools.combinations(exports, 2):
            assert all(set(a.measured) <= set(b.measured) for a, b in zip(low, high, strict=True))
        assert not any(x.measured for x in exports[0])
        assert all(x.measured == list(range(n)) for x in exports[-1])
        counts = [sum(len(x.measured) for x in e) for e in exports]
        assert len(set(counts)) > 25  # the grid is not degenerate


def test_graph_files_reproduce_from_their_seeds(reference):
    directory, _ = reference
    spec = SweepSpec.from_json(SPEC)
    for n, beta in itertools.product(spec.sizes, spec.betas):
        path = sweep._graph_file(directory, n, beta)
        records = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        num_nodes, collection, _ = graphs.load(path)
        assert num_nodes == n and len(collection) == len(records["graphs"]) == 2
        assert records["generator"] == ident.GRAPH_GENERATOR_VERSION
        for g, (edges, record) in enumerate(zip(collection, records["graphs"], strict=True)):
            again = ident.build_graph(n, beta, g, spec.master_graph_seed)
            assert record["seed"] == list(ident.graph_seed(17, n, beta, g))
            assert record["edge_digest"] == again.edge_digest == ident.edge_digest(edges, n)


def test_a_replayed_trajectory_equals_its_saved_row(reference):
    directory, results = reference
    for (n, beta, p), result in results.items():
        u = int(result.trajectories[-1])
        check = sweep.verify(directory, n, beta, p, u)
        assert check["identical"] and (check["g"], check["c"]) == divmod(u, 4)
        replayed = sweep.replay(directory, n, beta, p, u)
        row = int(np.flatnonzero(result.trajectories == u)[0])
        digests = [
            replayed.tableau(0, e).digest() for e in range(len(replayed.layers("tableau_snapshot")))
        ]
        assert digests == check["tableau_digests"]
        assert digests == [
            result.tableau(row, e).digest() for e in range(len(result.layers("tableau_snapshot")))
        ]
    with pytest.raises(ValueError, match="not a trajectory"):
        sweep.replay(directory, 6, "0", "0.1", 2)  # c = 2 >= circuits_per_graph
    with pytest.raises(ValueError, match="no condition"):
        sweep.load(directory, 6, "0", "0.2")


def test_rows_name_every_trajectory_and_its_seeds(reference):
    directory, results = reference
    rows = sweep.rows(directory, 8, "0.2", "0.3")
    result = results[(8, "0.2", "0.3")]
    assert np.array_equal(rows["u"], result.trajectories)
    assert np.array_equal(rows["g"], result.graph_index)
    assert rows["c"].tolist() == [0, 1, 0, 1] and set(rows["beta"]) == {"0.2"}
    assert [tuple(s) for s in rows["graph_seed"]] == [
        ident.graph_seed(17, 8, "0.2", g) for g in rows["g"].tolist()
    ]
    assert all(tuple(s) == tuple(result.seed_entropy) for s in rows["run_seed"])
    assert tuple(rows["outcome_seed"][1]) == ident.outcome_seed(29, 8, "0.2", "0.3", 0, 1)


def test_the_cli_runs_reports_and_replays(tmp_path, capsys):
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec_with(sizes=[6], betas=["0"], rates=["0.2"])))
    directory = str(tmp_path / "cli")
    assert cli.main(["sweep", "run", str(spec_file), directory, "--threads", "1", "--json"]) == 0
    ran = json.loads(capsys.readouterr().out)
    assert ran["complete"] and ran["ran"] == 4
    assert cli.main(["sweep", "status", directory, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["saved"] == 4
    assert cli.main(["sweep", "replay", directory, "6", "0", "0.2", "5", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["identical"] is True
    assert cli.main(["sweep", "replay", directory, "6", "0", "0.2", "x", "--json"]) == 2
    assert "U must be an integer" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_load_results_reads_a_sweep_and_its_conditions(reference, monkeypatch):
    directory, results = reference
    index = api.load_results(str(directory))
    assert index["kind"] == "sweep" and index["complete"] and index["saved"] == 32
    assert len(index["conditions"]) == 8 and index["versions"] == ident.version_record()
    assert index["spec"] == SweepSpec.from_json(SPEC).to_json()
    first = index["conditions"][0]
    summary = api.load_results(str(directory / first["path"]))
    result = results[(first["n"], first["beta"], first["p"])]
    expected = api._bounded_summary(result, {})
    assert summary["num_trajectories"] == expected["num_trajectories"] == 4
    assert summary["probes"] == expected["probes"] and summary["sweep"]["n"] == first["n"]
    assert summary["storage"]["total_megabytes"] > 0
    monkeypatch.setattr(api, "MAX_RESULT_GB", 1e-9)
    with pytest.raises(ValueError, match=r"memory limit.*sweep\.parts"):
        api.load_results(str(directory / first["path"]))


# ---------------------------------------------------------------------------
# Stim, on the trajectories of real sweeps
# ---------------------------------------------------------------------------


def stim_first_passages(k: list[int], n: int) -> list[int]:
    """``tau_ell = min{t : k[t] <= ell}`` for ``ell = 0 .. n`` (-1 if never), by definition."""
    return [next((t for t, v in enumerate(k) if v <= ell), -1) for ell in range(n + 1)]


@pytest.mark.parametrize("picture", ["purification", "single"])
def test_saved_states_and_code_dimensions_equal_stim_on_the_exported_circuits(picture, tmp_path):
    """Every saved trajectory of a sweep (odd and even n, beta 0 and 0.3, two rates), its
    circuit exported and run in Stim: the canonical joint state (signs dropped) and k at every
    saved checkpoint, and the first passages of k = S(R) = rank of the reference columns - |R|
    read at every layer, from Stim's state by the independent elimination of
    tests/_rowspace_oracle.py."""
    spec = spec_with(
        sizes=[7, 8], betas=["0", "0.3"], graphs=2, circuits_per_graph=1, picture=picture
    )
    directory = tmp_path / picture
    sweep.run(spec, directory)
    sweep_spec = SweepSpec.from_json(spec)
    checked = 0
    for n, beta, p in sweep_spec.conditions():
        res = sweep.load(directory, n, beta, p)
        _, collection, _ = graphs.load(sweep._graph_file(directory, n, beta))
        mc = sweep_spec.monitored_circuit(n, beta, p, collection)
        num_qubits = mc.num_qubits  # 2n, or n + 1 with a single reference
        reference = list(range(n, num_qubits))
        columns = reference + [num_qubits + q for q in reference]
        saved_layers = res.layers("tableau_snapshot").tolist()
        assert res.layers("code_dimension").tolist() == saved_layers  # the sparse default
        for row, u in enumerate(res.trajectories.tolist()):
            circuit = mc.circuit(u, seed=list(sweep_spec.run_seed(n, beta))).to_stim()
            states = stim_tick_states(circuit, num_qubits, seed=u)
            assert len(states) == sweep_spec.layers[n] + 1
            k = [oracle_rank(int_rows(state[:, columns])) - len(reference) for state in states]
            for e, layer in enumerate(saved_layers):
                tab = res.tableau(row, e)
                np.testing.assert_array_equal(tab.canonical(), canonical(states[layer]))
                assert tab.digest() == PhaseFreeTableau(states[layer]).digest()
                assert res.values["code_dimension"][row, e] == k[layer]
            assert res.first_passage[row].tolist() == stim_first_passages(k, n)
            checked += 1
    assert checked == 8 * 2


@functools.cache
def stim_clifford(index: int):
    """Two-qubit Clifford ``index`` as a ``stim.Tableau`` (what ``Circuit.to_stim`` expands)."""
    return Clifford.from_index(2, index).to_stim()


def stim_long_run(circuit, num_qubits: int, n: int, keep: set[int], seed: int):
    """Run an exported sweep circuit in Stim's TableauSimulator, gate by gate. Returns the
    canonical unsigned stabilizers after each layer in ``keep`` and the first passages of
    ``k = S(R)`` (rank of the reference columns - |R|, by tests/_rowspace_oracle.py), with ``k``
    read after layer 0 and after every layer that measures until it is 0 (a layer without
    measurements acts unitarily on the system, which leaves S(R) unchanged)."""
    sim = stim.TableauSimulator(seed=seed)
    sim.set_num_qubits(num_qubits)
    reference = range(n, num_qubits)
    columns = [*reference, *(num_qubits + q for q in reference)]
    states: dict[int, np.ndarray] = {}
    first = [-1] * (n + 1)
    layer, measured, k = 0, True, -1
    for ins in circuit:
        if ins.name == "TICK":
            if layer in keep or (measured and k != 0):
                state = np.array([bits(p) for p in sim.canonical_stabilizers()])
                k = oracle_rank(int_rows(state[:, columns])) - len(reference)
                first = [layer if f < 0 and ell >= k else f for ell, f in enumerate(first)]
                if layer in keep:
                    states[layer] = state
            layer, measured = layer + 1, False
        elif ins.name == "CLIFFORD_2Q":
            for j, index in enumerate(ins.args):
                pair = [ins.targets[2 * j].value, ins.targets[2 * j + 1].value]
                sim.do_tableau(stim_clifford(int(index)), pair)
        elif ins.name == "M":
            for t in ins.targets:
                sim.measure(t.value)
            measured = True
        else:  # the Bell pairs of the setup: H, CX
            sim.do(stim.Circuit(str(ins)))
    return states, first


STIM_DEPTH = 10_000


@pytest.mark.parametrize(
    ("picture", "sizes", "betas", "rates"),
    [
        ("purification", [6, 9], ["0", "0.3"], ["0.002", "0.3"]),
        ("single", [7, 8], ["0.3"], ["0.002", "0.02", "0.3"]),
    ],
)
def test_stim_agrees_at_depth_ten_thousand_and_replays_match(
    picture, sizes, betas, rates, tmp_path
):
    """Sweeps 10^4 layers deep (n = 6 to 9, low and high rates): every saved trajectory's
    exported circuit run in Stim gives the canonical joint digest and k at each of the 14
    geometric checkpoints and the exact first passages; replaying each trajectory from its seed
    (sweep.verify) reproduces every stored value, snapshot word and first passage."""
    spec = spec_with(
        sizes=sizes, betas=betas, rates=rates, graphs=1, circuits_per_graph=1, picture=picture,
        layers={str(n): STIM_DEPTH for n in sizes}, checkpoints={"count": 13},
    )  # fmt: skip
    directory = tmp_path / picture
    sweep.run(spec, directory)
    sweep_spec = SweepSpec.from_json(spec)
    deep = 0
    for n, beta, p in sweep_spec.conditions():
        res = sweep.load(directory, n, beta, p)
        _, collection, _ = graphs.load(sweep._graph_file(directory, n, beta))
        mc = sweep_spec.monitored_circuit(n, beta, p, collection)
        layers = res.layers("tableau_snapshot").tolist()
        assert layers == list(geometric_checkpoints(STIM_DEPTH, 13)) and len(layers) == 14
        assert res.layers("code_dimension").tolist() == layers
        for row, u in enumerate(res.trajectories.tolist()):
            circuit = mc.circuit(u, seed=list(sweep_spec.run_seed(n, beta)))
            states, first = stim_long_run(circuit, mc.num_qubits, n, set(layers), seed=u)
            reference = list(range(n, mc.num_qubits))
            columns = reference + [mc.num_qubits + q for q in reference]
            for e, layer in enumerate(layers):
                assert res.tableau(row, e).digest() == PhaseFreeTableau(states[layer]).digest()
                k = oracle_rank(int_rows(states[layer][:, columns])) - len(reference)
                assert res.values["code_dimension"][row, e] == k
            assert res.first_passage[row].tolist() == first
            deep += max(first) > 1000
            assert sweep.verify(directory, n, beta, p, u)["identical"]
    assert deep > 0  # some crossing happened deep in a run


@pytest.mark.parametrize(("compression", "writers"), [("none", 1), ("default", 3)])
def test_compression_and_writers_change_no_result(reference, tmp_path, compression, writers):
    _, expected = reference
    directory = tmp_path / compression
    sweep.run(SPEC, directory, compression=compression, writers=writers, chunk_size=1)
    for key, result in every_condition(directory).items():
        assert_same(result, expected[key])
    folder = directory / "n6_beta0_p0.1"
    parts = json.loads((folder / "provenance.json").read_text())["parts"]
    assert len(parts) == 4 and {e["compression"] for e in parts.values()} == {compression}
    method = zipfile.ZIP_STORED if compression == "none" else zipfile.ZIP_DEFLATED
    for path in folder.glob("result-*.npz"):
        with zipfile.ZipFile(path) as z:
            assert {info.compress_type for info in z.infolist()} == {method}
    with pytest.raises(ValueError, match="compression must be one of"):
        sweep.run(SPEC, directory, compression="best")


def test_a_stride_too_large_for_int64_is_refused_before_anything_is_written(tmp_path):
    for stride, graphs_ in [(2**62, 3), (2**70, 1)]:
        directory = tmp_path / f"s{stride}"
        with pytest.raises(ValueError, match="int64"):
            sweep.run(spec_with(sizes=[6], stride=stride, graphs=graphs_), directory)
        assert not directory.exists()
    with pytest.raises(ValueError, match="graphs"):
        SweepSpec.from_json(spec_with(graphs=2**32 + 1))
    # a 13-digit stream index: part names grow past 12 digits and are still read
    big = spec_with(sizes=[6], betas=["0"], rates=["0.1"], stride=10**12)
    directory = tmp_path / "wide"
    assert sweep.run(big, directory, chunk_size=1)["complete"]
    names = sorted(p.name for p in (directory / "n6_beta0_p0.1").glob("result-*.npz"))
    assert "result-1000000000000.npz" in names and len(names) == 4
    assert sweep.run(big, directory)["ran"] == 0 and sweep.status(directory)["saved"] == 4
    assert sweep.load(directory, 6, "0", "0.1").trajectories.tolist() == [0, 1, 10**12, 10**12 + 1]


def test_a_damaged_part_is_named_with_what_to_do(tmp_path):
    directory = tmp_path / "damaged"
    small = spec_with(sizes=[6], betas=["0"], rates=["0.1"])
    sweep.run(small, directory, chunk_size=2)
    path = sorted((directory / "n6_beta0_p0.1").glob("result-*.npz"))[-1]
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])  # truncated
    for call in (
        lambda: sweep.status(directory),
        lambda: sweep.run(small, directory),
        lambda: sweep.load(directory, 6, "0", "0.1"),
        lambda: api.load_results(str(directory / "n6_beta0_p0.1")),
    ):
        with pytest.raises(ValueError, match=f"{path.name}.*not a readable saved part.*delete it"):
            call()
    path.unlink()  # as the message says: delete it, and the next run recomputes it
    assert sweep.run(small, directory)["ran"] == 2


def test_spec_json_says_which_trajectories_it_counts(reference):
    directory, results = reference
    record = json.loads((directory / "n6_beta0_p0.1" / "spec.json").read_text())
    assert record["trajectories"] == 4 and record["spec"]["circuits_per_graph"] == 4
    sweep_record = record["sweep"]
    assert (sweep_record["graphs"], sweep_record["circuits_per_graph"], sweep_record["stride"]) == (
        2,
        2,
        4,
    )
    g, c = np.meshgrid(range(2), range(2), indexing="ij")
    expected = (g * sweep_record["stride"] + c).ravel().tolist()
    assert expected == results[(6, "0", "0.1")].trajectories.tolist() == [0, 1, 4, 5]


def test_the_records_carry_the_doubles_of_beta_and_p(reference):
    directory, _ = reference
    index = json.loads((directory / "sweep.json").read_text())
    assert {(e["beta"], e["beta_float64"]) for e in index["graphs"]} == {
        ("0", (0.0).hex()),
        ("0.2", (0.2).hex()),
    }
    entry = next(c for c in index["conditions"] if c["path"] == "n8_beta0.2_p0.3")
    assert (entry["beta_float64"], entry["p_float64"]) == ((0.2).hex(), (0.3).hex())
    records = json.loads((directory / "graphs" / "n8_beta0.2.json").read_text())
    assert records["beta_float64"] == (0.2).hex()
    assert all(r["beta_float64"] == (0.2).hex() for r in records["graphs"])


def test_a_stray_temporary_is_deleted_and_changes_nothing(reference, tmp_path):
    _, expected = reference
    directory = tmp_path / "stray"
    sweep.run(SPEC, directory)
    folder = directory / "n6_beta0_p0.1"
    stray = folder / "result-000000000004.npz.4242.deadbeef.tmp.npz"  # a killed writer's
    stray.write_bytes(b"half a part")
    assert sweep.status(directory)["saved"] == 32  # never read as a part
    assert sweep.run(SPEC, directory)["ran"] == 0
    assert not stray.exists()
    for key, result in every_condition(directory).items():
        assert_same(result, expected[key])


def test_growing_by_a_size_keeps_the_saved_one(reference, tmp_path):
    _, expected = reference
    directory = tmp_path / "sizes"
    sweep.run(spec_with(sizes=[6]), directory)
    before = every_condition(directory)
    out = sweep.run(SPEC, directory)  # adds n = 8
    assert out["complete"] and out["ran"] == 4 * 4
    after = every_condition(directory)
    for key, result in before.items():
        assert_same(after[key], result)
    for key, result in expected.items():  # equal to the sweep run in one go
        assert_same(after[key], result)


def test_a_sweep_that_stops_at_purification_grows_and_replays(tmp_path):
    stopping = spec_with(sizes=[6], betas=["0.2"], rates=["0.3", "0.6"], stop_when_purified=True)
    grown = {**stopping, "graphs": 3}
    directory, fresh = tmp_path / "grown", tmp_path / "fresh"
    sweep.run(stopping, directory)
    sweep.run(grown, directory)
    sweep.run(grown, fresh)
    stopped = 0
    for p in ("0.3", "0.6"):
        result = sweep.load(directory, 6, "0.2", p)
        assert_same(result, sweep.load(fresh, 6, "0.2", p))
        layers = result.layers("tableau_snapshot")
        for row, u in enumerate(result.trajectories.tolist()):
            tau = int(result.purification_layer[row])
            digests = sweep.verify(directory, 6, "0.2", p, u)["tableau_digests"]
            assert sweep.verify(directory, 6, "0.2", p, u)["identical"]
            for e, layer in enumerate(layers.tolist()):
                assert (digests[e] is None) == (0 <= tau < layer)  # nothing saved after the stop
            # every layer is a purification test: the stop is at the exact first passage tau_0
            assert tau == result.first_passage[row, 0] and result.final_layer[row] in (tau, 12)
            if 0 <= tau < layers[-1]:
                stopped += 1
                k = result.values["code_dimension"][row]
                assert np.all(k[layers >= tau] == 0)  # exact values after the stop
                assert np.all(k[layers < tau] > 0)
    assert stopped > 0  # the test saw trajectories stop before their last checkpoint
