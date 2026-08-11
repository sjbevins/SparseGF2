from __future__ import annotations

import math

import pytest
from studies.prl_production.graph_registry.spec import (
    BETA_SCALE,
    GRAPH_K,
    GraphCollectionSpec,
    beta_from_key,
    canonical_beta_key,
    production_spec,
    smoke_spec,
)


def test_production_spec_has_the_exact_requested_grid_and_count() -> None:
    spec = production_spec()

    assert spec.sizes == (64, 96, 128, 160, 192, 224, 256)
    assert len(spec.beta_keys) == 50
    assert spec.beta_keys[0] == 0
    assert spec.beta_keys[1] == 5_000_000
    assert spec.beta_keys[-1] == BETA_SCALE
    assert len(set(spec.beta_keys)) == 50
    assert spec.n_cells == 350
    assert spec.graphs_per_cell == 1_000
    assert spec.n_graphs == 350_000
    assert 2 * GRAPH_K == 4

    positive = spec.betas[1:]
    ratios = [right / left for left, right in zip(positive, positive[1:], strict=False)]
    expected = (1.0 / 0.005) ** (1.0 / 48)
    assert max(abs(ratio - expected) for ratio in ratios) < 2e-7


def test_tuple_derived_seeds_are_stable_and_grid_order_independent() -> None:
    spec = production_spec()
    reversed_spec = GraphCollectionSpec(
        name=spec.name,
        sizes=spec.sizes,
        beta_keys=spec.beta_keys,
        graphs_per_cell=spec.graphs_per_cell,
        master_seed=spec.master_seed,
    )

    selected = (128, spec.beta_keys[17], 731)
    seed = spec.graph_seed(*selected)
    assert seed == reversed_spec.graph_seed(*selected)
    assert 0 <= seed < 2**63
    assert seed != spec.graph_seed(selected[0], selected[1], selected[2] + 1)
    assert seed != spec.graph_seed(selected[0] + 32, selected[1], selected[2])


def test_record_iteration_is_canonical_and_content_hash_is_repeatable() -> None:
    spec = smoke_spec()
    rows = list(spec.records())

    assert len(rows) == 24
    assert [(row.n, row.beta_key, row.graph_index) for row in rows[:5]] == [
        (8, 0, 0),
        (8, 0, 1),
        (8, 0, 2),
        (8, 0, 3),
        (8, 5_000_000, 0),
    ]
    assert len(spec.seed_content_sha256()) == 64
    assert spec.seed_content_sha256() == smoke_spec().seed_content_sha256()
    assert spec.collection_id == smoke_spec().collection_id


def test_beta_identity_round_trip_and_strict_integer_validation() -> None:
    assert canonical_beta_key(0.005) == 5_000_000
    assert beta_from_key(5_000_000) == 0.005
    assert math.isclose(beta_from_key(BETA_SCALE), 1.0)

    with pytest.raises(TypeError, match="integer"):
        beta_from_key(True)
    with pytest.raises(TypeError, match="real number"):
        canonical_beta_key(True)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        canonical_beta_key(float("nan"))
    with pytest.raises(ValueError, match="not in this collection"):
        smoke_spec().graph_seed(8, 123, 0)
    with pytest.raises(ValueError, match="graph_index"):
        smoke_spec().graph_seed(8, 0, 4)
