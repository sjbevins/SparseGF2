from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest
from studies.prl_production.sweep_spec import (
    TAU_CENSORED,
    TAU_INCOMPLETE,
    GraphCollectionGridSpec,
    GraphParameterGrid,
    ParameterAxis,
    ParameterPoint,
    ProbabilityGrid,
    ScientificEnvironmentContract,
    SingleReferenceProtocolSpec,
    SingleReferenceSweepSpec,
)


def _environment() -> ScientificEnvironmentContract:
    return ScientificEnvironmentContract("3.test", "2.test", "0.test")


def _collection(*, reverse: bool = False) -> GraphCollectionGridSpec:
    axes = (
        ParameterAxis("beta", (0.1, 0.0)),
        ParameterAxis("k", (3, 2)),
    )
    if reverse:
        axes = tuple(reversed(axes))
    return GraphCollectionGridSpec(
        name="generic_graph_smoke_v1",
        graph_family="watts_strogatz",
        generator_name="tests.graph_factory",
        generator_version="1",
        sizes=(12, 8) if reverse else (8, 12),
        parameter_grid=GraphParameterGrid(axes),
        graphs_per_cell=3,
        master_seed=3_700_000_001,
    )


def _protocol() -> SingleReferenceProtocolSpec:
    return SingleReferenceProtocolSpec(
        n_circuits=4,
        q_scramble=8,
        q_max=8,
        p_grid=ProbabilityGrid("0.10", "0.12", "0.01"),
        master_seed=9_100_003,
    )


def test_cartesian_grid_is_complete_and_declaration_order_independent() -> None:
    collection = _collection()
    reordered = _collection(reverse=True)

    assert collection.specification_sha256 == reordered.specification_sha256
    assert collection.collection_id == reordered.collection_id
    assert collection.sizes == (8, 12)
    assert collection.parameter_grid.n_points == 4
    assert collection.n_cells == 8
    assert collection.n_graphs == 24

    cells = tuple(collection.cells())
    assert tuple(cell.cell_index for cell in cells) == tuple(range(8))
    assert {tuple(cell.parameters.items) for cell in cells if cell.n == 8} == {
        (("beta", 0.0), ("k", 2)),
        (("beta", 0.0), ("k", 3)),
        (("beta", 0.1), ("k", 2)),
        (("beta", 0.1), ("k", 3)),
    }


def test_typed_parameter_identity_and_grid_validation() -> None:
    axis = ParameterAxis("value", (1.0, 1, False, None, "1", np.int64(2), np.float64(2.0)))
    assert len(axis.values) == 7

    with pytest.raises(ValueError, match="duplicate"):
        ParameterAxis("beta", (0.1, 0.1))
    with pytest.raises(ValueError, match="NaN"):
        ParameterAxis("beta", (float("nan"),))
    with pytest.raises(ValueError, match="parameter names"):
        ParameterAxis("not-valid", (1,))
    with pytest.raises(ValueError, match="axis names"):
        GraphParameterGrid((ParameterAxis("k", (2,)), ParameterAxis("k", (3,))))


def test_empty_parameter_grid_represents_one_cell_per_size() -> None:
    grid = GraphParameterGrid()
    assert grid.n_points == 1
    assert tuple(point.items for point in grid.points()) == ((),)


def test_collection_resolves_cells_and_derives_stable_graph_seeds() -> None:
    collection = _collection()
    cell = collection.cell(8, {"k": 2, "beta": 0.1})
    same = collection.cell(8, {"beta": 0.1, "k": 2})

    assert cell == same
    seeds = tuple(collection.graph_seed(cell, index) for index in range(3))
    assert seeds == tuple(collection.graph_seed(same, index) for index in range(3))
    assert len(set(seeds)) == 3
    assert all(0 <= seed < 1 << 63 for seed in seeds)
    records = tuple(collection.records())
    assert len(records) == collection.n_graphs
    assert records[cell.cell_index * 3].cell == cell

    with pytest.raises(ValueError, match="exactly the keys"):
        collection.cell(8, {"beta": 0.1})
    with pytest.raises(ValueError, match="not in the grid"):
        collection.cell(8, {"beta": 0.2, "k": 2})
    with pytest.raises(ValueError, match="different graph collection"):
        replace(collection, master_seed=collection.master_seed + 1).graph_seed(cell, 0)
    forged = replace(cell, parameters=ParameterPoint((("beta", 0.2), ("k", 2))))
    with pytest.raises(ValueError, match="canonical cell"):
        collection.graph_seed(forged, 0)


def test_probability_grid_uses_exact_inclusive_decimal_arithmetic() -> None:
    grid = ProbabilityGrid(0.1, Decimal("0.3"), "0.05")

    assert grid.canonical_values == ("0.1", "0.15", "0.2", "0.25", "0.3")
    assert grid.values == pytest.approx((0.1, 0.15, 0.2, 0.25, 0.3))
    assert grid.canonical_payload() == {
        "delta_p": "0.05",
        "p_max": "0.3",
        "p_min": "0.1",
    }

    with pytest.raises(ValueError, match="integer multiple"):
        ProbabilityGrid("0.1", "0.3", "0.07")
    with pytest.raises(ValueError, match="0 <="):
        ProbabilityGrid("-0.1", "0.3", "0.1")
    with pytest.raises(ValueError, match="positive"):
        ProbabilityGrid("0.1", "0.3", "0")
    with pytest.raises(ValueError, match="binary64"):
        ProbabilityGrid(
            "0.5",
            "0.50000000000000000001",
            "0.00000000000000000001",
        )


def test_protocol_keeps_scramble_gate_count_and_depth_cap_explicit() -> None:
    protocol = _protocol()

    assert protocol.scramble_gate_count(64) == 512
    assert protocol.t_max(64) == 512
    assert protocol.canonical_payload()["scramble_gate_count"] == "q_scramble*n"
    assert protocol.canonical_payload()["t_max"] == "q_max*n"
    assert protocol.canonical_payload()["clifford_table_size"] == 720
    assert protocol.t_max_factor == protocol.q_max
    assert protocol.canonical_payload()["dynamic_edge_sampling"].endswith("with_replacement")

    independent_depth = replace(protocol, q_max=12)
    assert independent_depth.scramble_gate_count(64) == 512
    assert independent_depth.t_max(64) == 768
    assert independent_depth.specification_sha256 != protocol.specification_sha256


def test_protocol_rejects_raw_int32_overflow_before_array_casts() -> None:
    protocol = replace(_protocol(), q_max=(1 << 30), q_scramble=(1 << 30))

    with pytest.raises(OverflowError, match=r"q_max\*n"):
        protocol.t_max(3)
    with pytest.raises(OverflowError, match=r"q_scramble\*n"):
        protocol.scramble_gate_count(3)


def test_work_units_link_every_cell_and_p_to_per_graph_circuit_arrays() -> None:
    graphs = _collection()
    protocol = _protocol()
    sweep = SingleReferenceSweepSpec(
        name="tau_smoke_v1",
        graph_collection_sha256=graphs.specification_sha256,
        source_fingerprint_sha256="ab" * 32,
        environment_contract=_environment(),
        protocol=protocol,
    )

    units = tuple(sweep.work_units(graphs))
    assert len(units) == graphs.n_cells * len(protocol.p_grid.values)
    first = units[0]
    assert first.raw_shape == (graphs.graphs_per_cell, protocol.n_circuits)
    assert first.p_decimal == "0.1"
    assert first.artifact_relative_path.as_posix().endswith(".npz")
    assert sweep.specification_sha256 in first.artifact_relative_path.parts

    seeds = {
        first.trajectory_seed(0, 0, "scramble_pairs"),
        first.trajectory_seed(0, 1, "scramble_pairs"),
        first.trajectory_seed(0, 0, "dynamic_edges"),
        units[1].trajectory_seed(0, 0, "scramble_pairs"),
    }
    assert len(seeds) == 4
    assert first.trajectory_seed(0, 0, "scramble_pairs") == first.trajectory_seed(
        0, 0, "scramble_pairs"
    )

    with pytest.raises(ValueError, match="stream must"):
        first.trajectory_seed(0, 0, "unknown")
    with pytest.raises(ValueError, match="circuit_index"):
        first.trajectory_seed(0, protocol.n_circuits, "scramble_pairs")


def test_p_randomness_policy_is_explicit_and_changes_cross_p_seed_linkage() -> None:
    graphs = _collection()
    independent = _protocol()
    independent_sweep = SingleReferenceSweepSpec(
        name="tau_smoke_v1",
        graph_collection_sha256=graphs.specification_sha256,
        source_fingerprint_sha256="ab" * 32,
        environment_contract=_environment(),
        protocol=independent,
    )
    independent_units = tuple(independent_sweep.work_units(graphs))
    assert independent_units[0].trajectory_seed(0, 0, "dynamic_edges") != independent_units[
        1
    ].trajectory_seed(0, 0, "dynamic_edges")

    common = replace(independent, p_randomness_policy="common_circuit_disorder")
    common_sweep = replace(independent_sweep, protocol=common)
    common_units = tuple(common_sweep.work_units(graphs))
    for role in (
        "reference_placement",
        "scramble_pairs",
        "scramble_cliffords",
        "dynamic_edges",
        "dynamic_cliffords",
        "measurement_mask",
        "measurement_outcomes",
    ):
        assert common_units[0].trajectory_seed(0, 0, role) == common_units[1].trajectory_seed(
            0, 0, role
        )

    with pytest.raises(ValueError, match="p_randomness_policy"):
        replace(independent, p_randomness_policy="unspecified")


def test_trajectory_seeds_survive_p_grid_extension_and_campaign_rename() -> None:
    graphs = _collection()
    original_protocol = _protocol()
    original_sweep = SingleReferenceSweepSpec(
        name="tau_original",
        graph_collection_sha256=graphs.specification_sha256,
        source_fingerprint_sha256="ab" * 32,
        environment_contract=_environment(),
        protocol=original_protocol,
    )
    original = next(unit for unit in original_sweep.work_units(graphs) if unit.p_decimal == "0.1")

    extended_protocol = replace(
        original_protocol,
        p_grid=ProbabilityGrid("0.09", "0.12", "0.01"),
    )
    extended_sweep = replace(
        original_sweep,
        name="tau_renamed",
        source_fingerprint_sha256="cd" * 32,
        environment_contract=replace(_environment(), numpy="2.changed"),
        protocol=extended_protocol,
    )
    extended = next(unit for unit in extended_sweep.work_units(graphs) if unit.p_decimal == "0.1")

    assert original.experiment_sha256 != extended.experiment_sha256
    for role in (
        "reference_placement",
        "scramble_pairs",
        "scramble_cliffords",
        "dynamic_edges",
        "dynamic_cliffords",
        "measurement_mask",
        "measurement_outcomes",
    ):
        assert original.trajectory_seed(2, 2, role) == extended.trajectory_seed(2, 2, role)


def test_work_units_refuse_foreign_graph_collections() -> None:
    graphs = _collection()
    sweep = SingleReferenceSweepSpec(
        name="tau_smoke_v1",
        graph_collection_sha256=graphs.specification_sha256,
        source_fingerprint_sha256="ab" * 32,
        environment_contract=_environment(),
        protocol=_protocol(),
    )
    foreign = replace(graphs, master_seed=graphs.master_seed + 1)

    with pytest.raises(ValueError, match="does not match"):
        tuple(sweep.work_units(foreign))


def test_raw_tau_sentinels_distinguish_incomplete_and_censored_rows() -> None:
    assert TAU_INCOMPLETE < TAU_CENSORED < 0
