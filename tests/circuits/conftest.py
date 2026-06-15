"""Shared fixtures for the circuits test suite."""

from __future__ import annotations

import pytest

from sparsegf2.circuits import CircuitConfig, Picture


@pytest.fixture
def golden_config() -> CircuitConfig:
    """The canonical golden-path cell: pure_state × cycle × brickwork × bernoulli."""
    return CircuitConfig(
        graph_spec="cycle",
        n=8,
        picture=Picture.PURE_STATE,
        gating_mode="brickwork",
        matching_mode="round_robin",
        measurement_mode="bernoulli",
        p=0.16,
        depth_factor=4,
    )
