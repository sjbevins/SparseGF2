"""Correctness and resume tests for generalized raw single-reference data."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from studies.prl_production.single_ref.raw_tau import engine as raw_tau_engine
from studies.prl_production.single_ref.raw_tau.engine import simulate_trajectory
from studies.prl_production.single_ref.raw_tau.io import write_deterministic_npz
from studies.prl_production.single_ref.raw_tau.providers import (
    GridGraphProvider,
    ProviderCell,
    load_edge_bank,
    prepare_edge_bank,
)
from studies.prl_production.single_ref.raw_tau.storage import (
    load_raw_tau_arrays,
    logical_tau_digest,
    raw_tau_path,
    run_work_unit,
)
from studies.prl_production.sweep_spec import (
    GraphCollectionGridSpec,
    GraphParameterGrid,
    ParameterAxis,
    ProbabilityGrid,
    ScientificEnvironmentContract,
    SingleReferenceProtocolSpec,
    SingleReferenceSweepSpec,
)


def _cycle_edges(n: int, _parameters, _seed: int):
    return [(i, (i + 1) % n) for i in range(n)]


def _case(
    *,
    n: int = 5,
    n_graphs: int = 2,
    n_circuits: int = 3,
    p_min: str = "0",
    p_max: str = "1",
    delta_p: str = "1",
    q_scramble: int = 1,
    q_max: int = 2,
    randomness: str = "independent",
):
    graphs = GraphCollectionGridSpec(
        name="raw_tau_test_graphs",
        graph_family="cycle",
        generator_name="tests.cycle_edges",
        generator_version="v1",
        sizes=(n,),
        parameter_grid=GraphParameterGrid(),
        graphs_per_cell=n_graphs,
        master_seed=123,
    )
    provider = GridGraphProvider(graphs, factory=_cycle_edges)
    protocol = SingleReferenceProtocolSpec(
        n_circuits=n_circuits,
        q_scramble=q_scramble,
        q_max=q_max,
        p_grid=ProbabilityGrid(p_min, p_max, delta_p),
        master_seed=456,
        p_randomness_policy=randomness,
    )
    sweep = SingleReferenceSweepSpec(
        name="raw_tau_test",
        graph_collection_sha256=graphs.specification_sha256,
        source_fingerprint_sha256="a" * 64,
        environment_contract=ScientificEnvironmentContract("3.test", "2.test", "0.test"),
        protocol=protocol,
    )
    return graphs, provider, sweep, tuple(sweep.work_units(graphs))


def test_engine_constructs_the_pinned_pcg64_generator() -> None:
    generator = raw_tau_engine._pcg64(123)

    assert type(generator.bit_generator) is np.random.PCG64


def test_exact_protocol_counts_for_odd_n_and_p_zero() -> None:
    _graphs, provider, _sweep, works = _case(n=5, q_scramble=3, q_max=4)
    work = works[0]
    cell = provider.cells()[0]
    edges = np.asarray(tuple(provider.build_edges(cell, provider.graph_seeds(cell)[0])))
    result = simulate_trajectory(work, 0, 0, edges, profile=True)

    assert result.tau_p is None
    assert result.stop_layer == 20
    assert not result.event_observed
    assert result.scramble_gates == 15
    assert result.dynamic_gates == 20 * (5 // 2)
    assert result.measurements == 0
    assert result.reference_system_qubit == 4
    assert result.timings is not None
    assert result.timings.total_s >= result.timings.scramble_s >= 0.0


def test_p_one_purifies_on_first_measured_layer() -> None:
    _graphs, provider, _sweep, works = _case(p_min="1", p_max="1")
    work = works[0]
    cell = provider.cells()[0]
    edges = np.asarray(tuple(provider.build_edges(cell, provider.graph_seeds(cell)[0])))
    result = simulate_trajectory(work, 0, 0, edges)
    assert result.tau_p == result.stop_layer == result.layers_executed == 1
    assert result.event_observed
    assert result.dynamic_gates == 5 // 2
    assert result.measurements == 5


def test_entropy_is_checked_after_every_layer_and_stops_at_first_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _graphs, provider, _sweep, works = _case(
        n=4,
        n_graphs=1,
        n_circuits=1,
        p_min="0",
        p_max="0",
        q_scramble=0,
        q_max=2,
    )
    work = works[0]
    cell = provider.cells()[0]
    edges = np.asarray(tuple(provider.build_edges(cell, provider.graph_seeds(cell)[0])))
    values = iter((1, 1, 1, 1, 0))  # initial, post-scramble, then t=1,2,3
    calls: list[int] = []

    def controlled_entropy(_sim, qubit: int) -> int:
        calls.append(qubit)
        return next(values)

    monkeypatch.setattr(raw_tau_engine, "single_qubit_entropy", controlled_entropy)
    result = simulate_trajectory(work, 0, 0, edges)

    assert result.tau_p == result.stop_layer == result.layers_executed == 3
    assert result.dynamic_gates == 3 * (work.cell.n // 2)
    assert len(calls) == 5


def test_batch_scalar_and_hybrid_sparse_are_bit_identical() -> None:
    _graphs, provider, _sweep, works = _case(n=6, p_min="0.3", p_max="0.3")
    work = works[0]
    cell = provider.cells()[0]
    edges = np.asarray(tuple(provider.build_edges(cell, provider.graph_seeds(cell)[0])))
    results = [
        simulate_trajectory(
            work,
            0,
            1,
            edges,
            execution=execution,
            hybrid=hybrid,
            audit_tableau=True,
        )
        for execution, hybrid in (("batch", True), ("scalar", True), ("batch", False))
    ]
    identities = [
        (
            result.tau_p,
            result.stop_layer,
            result.event_observed,
            result.reference_system_qubit,
            result.final_tableau_sha256,
        )
        for result in results
    ]
    assert identities[0] == identities[1] == identities[2]


def test_common_disorder_is_reserved_not_silently_misimplemented() -> None:
    _graphs, provider, _sweep, works = _case(randomness="common_circuit_disorder")
    work = works[0]
    cell = provider.cells()[0]
    edges = np.asarray(tuple(provider.build_edges(cell, provider.graph_seeds(cell)[0])))
    with pytest.raises(ValueError, match="reserved"):
        simulate_trajectory(work, 0, 0, edges)


def test_engine_rejects_empty_loops_and_duplicate_graph_edges() -> None:
    _graphs, _provider, _sweep, works = _case(n=4, n_graphs=1)
    work = works[0]
    with pytest.raises(ValueError, match="nonempty"):
        simulate_trajectory(work, 0, 0, np.empty((0, 2), dtype=np.int32))
    with pytest.raises(ValueError, match="duplicate"):
        simulate_trajectory(work, 0, 0, np.asarray([[0, 1], [1, 0]], dtype=np.int32))


def test_cartesian_provider_supports_variable_edge_counts(tmp_path: Path) -> None:
    graph_spec = GraphCollectionGridSpec(
        name="two_axis",
        graph_family="fake",
        generator_name="tests.variable_edges",
        generator_version="v1",
        sizes=(4, 6),
        parameter_grid=GraphParameterGrid(
            (
                ParameterAxis("degree", (1, 2)),
                ParameterAxis("variant", ("a", "b")),
            )
        ),
        graphs_per_cell=3,
        master_seed=7,
    )

    def factory(n: int, parameters, seed: int):
        count = int(parameters["degree"]) + seed % 2
        base = [(i, (i + 1) % n) for i in range(n)]
        base.extend((i, i + 2) for i in range(max(0, n - 2)))
        return base[: n - 1 + count]

    class SequentialSeedProvider(GridGraphProvider):
        def graph_seeds(self, cell):
            self._validate_cell(cell)
            return (0, 1, 2)

    provider = SequentialSeedProvider(graph_spec, factory=factory)
    assert len(provider.cells()) == 2 * 2 * 2
    cell = provider.cells()[0]
    with pytest.raises(ValueError, match="path-safe"):
        replace(cell, collection_id="../outside")
    path = prepare_edge_bank(tmp_path, provider, cell)
    assert len(cell.generator_contract_sha256) == 64
    assert f"contract_{cell.generator_contract_sha256[:16]}" in Path(path).parts
    bank = load_edge_bank(path, cell, provider.graph_seeds(cell))
    counts = np.diff(bank.edge_offsets)
    assert bank.n_graphs == 3
    assert len(set(counts.tolist())) > 1

    forged = ProviderCell(
        collection_id=cell.collection_id,
        graph_family=cell.graph_family,
        generator_name=cell.generator_name,
        generator_version=cell.generator_version,
        generator_contract_sha256=cell.generator_contract_sha256,
        spec=type(cell.spec)(
            collection_sha256=cell.spec.collection_sha256,
            cell_index=cell.spec.cell_index,
            n=cell.n,
            parameters=type(cell.spec.parameters)((("degree", 99), ("variant", "a"))),
        ),
        graphs_per_cell=cell.graphs_per_cell,
    )
    with pytest.raises(ValueError, match="canonical member"):
        provider.graph_seeds(forged)


def test_raw_shard_resume_and_clean_run_are_byte_identical(tmp_path: Path) -> None:
    _graphs, provider, sweep, works = _case(
        n=4,
        n_graphs=2,
        n_circuits=3,
        p_min="1",
        p_max="1",
        q_scramble=0,
        q_max=1,
    )
    work = works[0]
    cell = provider.cells()[0]
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    bank_a = prepare_edge_bank(root_a, provider, cell)
    bank_b = prepare_edge_bank(root_b, provider, cell)

    partial = run_work_unit(
        root_a,
        sweep,
        work,
        cell,
        bank_a,
        checkpoint_every=1,
        max_new_trajectories=2,
    )
    assert partial.completed == 2
    assert not partial.is_complete
    resumed = run_work_unit(root_a, sweep, work, cell, bank_a, checkpoint_every=2)
    clean = run_work_unit(root_b, sweep, work, cell, bank_b, checkpoint_every=2)
    assert resumed.is_complete and clean.is_complete
    assert resumed.completed == clean.completed == 6
    assert resumed.logical_result_sha256 == clean.logical_result_sha256

    path_a = raw_tau_path(root_a, work)
    path_b = raw_tau_path(root_b, work)
    assert (
        hashlib.sha256(path_a.read_bytes()).digest() == hashlib.sha256(path_b.read_bytes()).digest()
    )
    before = path_a.read_bytes()
    repeated = run_work_unit(root_a, sweep, work, cell, bank_a, checkpoint_every=1)
    assert repeated.newly_completed == 0
    assert path_a.read_bytes() == before

    bank = load_edge_bank(bank_a, cell)
    arrays = load_raw_tau_arrays(path_a, sweep, work, cell, bank)
    assert np.all(np.asarray(arrays["tau_p"]) == 1)
    assert np.all(np.asarray(arrays["event_observed"]) == 1)
    assert np.all(np.asarray(arrays["complete"]) == 1)
    assert logical_tau_digest(arrays) == resumed.logical_result_sha256

    opposite_endian = dict(arrays)
    for key in ("graph_index", "graph_seed", "circuit_index", "tau_p", "stop_layer"):
        array = np.asarray(arrays[key])
        opposite_endian[key] = array.astype(array.dtype.newbyteorder(">"))
    opposite_endian["reference_system_qubit"] = np.asarray(arrays["reference_system_qubit"]).astype(
        ">i4"
    )
    assert logical_tau_digest(opposite_endian) == resumed.logical_result_sha256


def test_raw_shard_rejects_identity_mismatch(tmp_path: Path) -> None:
    _graphs, provider, sweep, works = _case(n=4, n_graphs=1, n_circuits=1, p_min="1", p_max="1")
    work = works[0]
    cell = provider.cells()[0]
    bank_path = prepare_edge_bank(tmp_path, provider, cell)
    run_work_unit(tmp_path, sweep, work, cell, bank_path, checkpoint_every=1)

    other_protocol = SingleReferenceProtocolSpec(
        n_circuits=1,
        q_scramble=1,
        q_max=3,
        p_grid=ProbabilityGrid("1", "1", "1"),
        master_seed=456,
    )
    other_sweep = SingleReferenceSweepSpec(
        name="raw_tau_test",
        graph_collection_sha256=sweep.graph_collection_sha256,
        source_fingerprint_sha256="a" * 64,
        environment_contract=sweep.environment_contract,
        protocol=other_protocol,
    )
    with pytest.raises(ValueError, match="different experiment"):
        run_work_unit(tmp_path, other_sweep, work, cell, bank_path)


@pytest.mark.parametrize(
    ("key", "value"),
    (("event_observed", 256), ("event_observed", -1), ("tau_p", (1 << 31) + 1)),
)
def test_raw_shard_rejects_values_that_would_narrow_on_load(
    tmp_path: Path,
    key: str,
    value: int,
) -> None:
    _graphs, provider, sweep, works = _case(
        n=4,
        n_graphs=1,
        n_circuits=1,
        p_min="1",
        p_max="1",
        q_scramble=0,
        q_max=1,
    )
    work = works[0]
    cell = provider.cells()[0]
    bank_path = prepare_edge_bank(tmp_path, provider, cell)
    run_work_unit(tmp_path, sweep, work, cell, bank_path, checkpoint_every=1)
    bank = load_edge_bank(bank_path, cell)
    path = raw_tau_path(tmp_path, work)
    arrays = load_raw_tau_arrays(path, sweep, work, cell, bank)
    tampered = dict(arrays)
    replacement = np.asarray(arrays[key], dtype=np.int64).copy()
    replacement[0, 0] = value
    tampered[key] = replacement
    write_deterministic_npz(path, tampered)

    with pytest.raises(ValueError, match=f"{key} must have"):
        load_raw_tau_arrays(path, sweep, work, cell, bank)
    with pytest.raises(ValueError, match=f"{key} must have"):
        logical_tau_digest(tampered)
