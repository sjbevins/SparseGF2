"""The sweep layer (``sparsegf2.study.sweep``; ``docs/SWEEPS.md``).

Every test runs at fixed seeds and asserts nothing about time. A sweep's saved trajectories are
compared bitwise with what the engine gives for the same spec, seed and stream indices, across
chunk sizes, interruptions, thread counts and growth of the sweep.
"""

from __future__ import annotations

import copy
import itertools
import json
from decimal import Decimal, localcontext

import numpy as np
import pytest

import _ws_reference as wsref
from sparsegf2 import api, cli, graphs, mipt
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
        ({"picture": "single"}, "picture"),
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
    for threads in (1, 3):
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


def test_a_directory_of_another_version_is_refused(tmp_path):
    directory = tmp_path / "old"
    sweep.run(spec_with(sizes=[6], betas=["0"], rates=["0.1"], graphs=1), directory)
    index = json.loads((directory / "sweep.json").read_text())
    index["versions"]["sparsegf2"] = "0.0.0"
    (directory / "sweep.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="sparsegf2"):
        sweep.run(spec_with(sizes=[6], betas=["0"], rates=["0.1"], graphs=1), directory)
    index["versions"]["randomness_contract"] = 1
    (directory / "sweep.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="randomness contract 1"):
        sweep.load(directory, 6, "0", "0.1")


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
