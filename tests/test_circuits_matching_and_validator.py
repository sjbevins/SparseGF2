"""Unit tests for matching.py and validator.py."""
from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.circuits.graphs import cycle_graph, complete_graph
from sparsegf2.circuits.matching import select_matching, available_modes
from sparsegf2.circuits.config import CircuitConfig, RunConfig
from sparsegf2.circuits.validator import (
    validate_config, CompatibilityError, ValidationReport,
)


# Matching modes

def test_round_robin_cycles_through_palette():
    g = complete_graph(6)
    rng = np.random.default_rng(0)
    palette = [tuple(sorted(m)) for m in g.one_factorization]
    for t in range(12):
        m = tuple(sorted(select_matching(g, "round_robin", t, rng)))
        expected = palette[t % len(palette)]
        assert m == expected, f"layer {t}: got {m}, expected {expected}"


def test_round_robin_ignores_rng():
    g = complete_graph(6)
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(999)
    for t in range(20):
        ma = tuple(sorted(select_matching(g, "round_robin", t, rng_a)))
        mb = tuple(sorted(select_matching(g, "round_robin", t, rng_b)))
        assert ma == mb


def test_palette_draws_from_palette_only():
    g = complete_graph(8)
    palette = set(tuple(sorted(m)) for m in g.one_factorization)
    rng = np.random.default_rng(42)
    for t in range(200):
        m = tuple(sorted(select_matching(g, "palette", t, rng)))
        assert m in palette


def test_palette_is_seed_reproducible():
    g = complete_graph(8)
    seq_a = [
        tuple(sorted(select_matching(g, "palette", t, np.random.default_rng(17))))
        for t in range(5)
    ]
    seq_b = [
        tuple(sorted(select_matching(g, "palette", t, np.random.default_rng(17))))
        for t in range(5)
    ]
    # Different RNG instances seeded the same produce identical draws
    assert seq_a == seq_b


def test_fresh_complete_all_pms_reachable_k6():
    """Over many draws, fresh sampler on K_6 reaches every perfect matching."""
    g = complete_graph(6)
    rng = np.random.default_rng(0)
    seen = set()
    # K_6 has 5!! = 15 perfect matchings
    for _ in range(20000):
        m = tuple(sorted(select_matching(g, "fresh", 0, rng)))
        seen.add(m)
        if len(seen) == 15:
            break
    assert len(seen) == 15


def test_available_modes_even_complete():
    g = complete_graph(8)
    assert set(available_modes(g)) == {"round_robin", "palette", "fresh"}


def test_available_modes_odd_complete():
    g = complete_graph(5)
    assert available_modes(g) == []


def test_available_modes_odd_cycle():
    g = cycle_graph(5)
    assert available_modes(g) == []


# Validator

def _cfg(**overrides) -> RunConfig:
    defaults = dict(
        graph_spec="cycle", n=8, matching_mode="round_robin",
        measurement_mode="uniform", depth_mode="O(n)", depth_factor=1,
        p=0.15, base_seed=42,
    )
    defaults.update(overrides.pop("circuit_overrides", {}))
    circuit = CircuitConfig(**defaults)
    run_defaults = dict(
        circuit=circuit, sizes=[8], p_min=0.0, p_max=0.3, n_p=2,
        n_samples_per_cell=1,
    )
    run_defaults.update(overrides)
    return RunConfig(**run_defaults)


def test_validator_passes_even_sizes():
    cfg = _cfg(sizes=[8, 16, 32])
    report = validate_config(cfg)
    assert report.passed
    assert report.ok == [8, 16, 32]


def test_validator_rejects_odd_cycle_round_robin():
    cfg = _cfg(sizes=[8, 11, 16])
    with pytest.raises(CompatibilityError) as exc:
        validate_config(cfg)
    report = exc.value.report
    assert 11 in [n for n, _ in report.incompatible]
    assert 8 in report.ok and 16 in report.ok
    # Formatted message must be plain ASCII for Windows cp1252 consoles
    msg = report.format()
    msg.encode("cp1252")      # must not raise


def test_validator_suggests_available_modes():
    cfg = _cfg(sizes=[8, 16], circuit_overrides={"matching_mode": "fresh"})
    # All sizes even -> fresh is available
    assert validate_config(cfg).passed


def test_validator_rejects_odd_complete():
    c = CircuitConfig(
        graph_spec="complete", n=8, matching_mode="round_robin",
        measurement_mode="uniform", depth_mode="O(n)", depth_factor=1,
        p=0.1, base_seed=0,
    )
    cfg = RunConfig(circuit=c, sizes=[4, 5], p_min=0.0, p_max=0.5, n_p=2,
                    n_samples_per_cell=1)
    with pytest.raises(CompatibilityError) as exc:
        validate_config(cfg)
    bad = [n for n, _ in exc.value.report.incompatible]
    assert bad == [5]


def test_validator_report_shape():
    cfg = _cfg(sizes=[8, 11])
    try:
        validate_config(cfg)
    except CompatibilityError as e:
        r: ValidationReport = e.report
        assert r.graph_spec == "cycle"
        assert r.matching_mode == "round_robin"
        assert r.sizes == [8, 11]
        assert 11 in [n for n, _ in r.incompatible]
        assert r.available_modes_by_n[11] == []      # no PM -> empty
        assert set(r.available_modes_by_n[8]) == {"round_robin", "palette", "fresh"}
