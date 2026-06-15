"""Tests for the sweep driver."""

from __future__ import annotations

import pytest

from sparsegf2.analysis import analyze_tableaux, sweep
from sparsegf2.circuits import CircuitConfig
from sparsegf2.circuits.picture import setup_picture
from sparsegf2.errors import InvalidArgumentError

BASE = CircuitConfig(graph_spec="cycle", n=10, picture="purification", p=0.2, depth_factor=3)


def _stable(rows):
    """Rows minus the wall-clock field (non-deterministic by nature)."""
    return [{k: v for k, v in r.items() if k != "runtime_total_s"} for r in rows]


def test_grid_expansion_count():
    res = sweep(BASE, {"p": [0.1, 0.2, 0.3]}, seeds=4, analyses=["code_dimension"])
    assert len(res) == 3 * 4
    assert {r["p"] for r in res.rows} == {0.1, 0.2, 0.3}
    assert {r["sample_seed"] for r in res.rows} == {0, 1, 2, 3}


def test_scalar_and_array_analyses_split():
    res = sweep(BASE, {"p": [0.2]}, seeds=2, analyses=["code_dimension", "entropy_profile"])
    assert all("a.code_dimension" in r for r in res.rows)
    assert "entropy_profile" in res.array_results
    assert len(res.array_results["entropy_profile"]) == len(res)
    assert all(not k.startswith("a.entropy_profile") for r in res.rows for k in r)


def test_determinism_repeated():
    a = sweep(BASE, {"p": [0.15, 0.3]}, seeds=3, analyses=["code_dimension"])
    b = sweep(BASE, {"p": [0.15, 0.3]}, seeds=3, analyses=["code_dimension"])
    assert _stable(a.rows) == _stable(b.rows)


def test_parallel_matches_serial():
    pytest.importorskip("joblib")  # n_jobs>1 needs the 'parallel' extra
    serial = sweep(BASE, {"p": [0.1, 0.25]}, seeds=3, analyses=["code_dimension"], n_jobs=1)
    parallel = sweep(BASE, {"p": [0.1, 0.25]}, seeds=3, analyses=["code_dimension"], n_jobs=2)
    assert _stable(serial.rows) == _stable(parallel.rows)


def test_offline_tableaux_match_online():
    online = sweep(BASE, {"p": [0.2]}, seeds=3, analyses=["code_dimension"])
    offline = sweep(BASE, {"p": [0.2]}, seeds=3, collect_tableaus=True)
    assert len(offline.tableaux) == 3
    spec = setup_picture(BASE.picture, BASE.n)[1]
    offrows = analyze_tableaux(list(offline.tableaux.values()), spec, ["code_dimension"])
    online_k = sorted(r["a.code_dimension"] for r in online.rows)
    offline_k = sorted(int(r["code_dimension"]) for r in offrows)
    assert online_k == offline_k


def test_unknown_grid_key_raises():
    with pytest.raises(InvalidArgumentError):
        sweep(BASE, {"not_a_field": [1, 2]}, seeds=1)


def test_no_grid_just_seeds():
    res = sweep(BASE, seeds=5, analyses=["code_dimension"])
    assert len(res) == 5


def test_provenance_recorded():
    res = sweep(BASE, {"p": [0.1, 0.2]}, seeds=2, analyses=["code_dimension"])
    assert res.grid == {"p": [0.1, 0.2]}
    assert res.seeds == [0, 1]
    assert res.base_config["graph_spec"] == "cycle"
    assert res.analyses == ["code_dimension"]


def test_seeds_nonpositive_raises():
    with pytest.raises(InvalidArgumentError):
        sweep(BASE, {"p": [0.1]}, seeds=0)
    with pytest.raises(InvalidArgumentError):
        sweep(BASE, {"p": [0.1]}, seeds=[])


def test_parallel_preserves_per_row_order():
    pytest.importorskip("joblib")  # n_jobs>1 needs the 'parallel' extra
    serial = sweep(BASE, {"p": [0.1, 0.25]}, seeds=3, analyses=["code_dimension"], n_jobs=1)
    parallel = sweep(BASE, {"p": [0.1, 0.25]}, seeds=3, analyses=["code_dimension"], n_jobs=2)
    # row-by-row (p, seed) alignment, not just set equality
    for rs, rp in zip(serial.rows, parallel.rows, strict=True):
        assert (rs["p"], rs["sample_seed"]) == (rp["p"], rp["sample_seed"])
        assert rs["a.code_dimension"] == rp["a.code_dimension"]


def test_save_load_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    from sparsegf2.analysis import SweepResult

    res = sweep(BASE, {"p": [0.1, 0.2, 0.3]}, seeds=2, analyses=["code_dimension"])
    out = tmp_path / "study.parquet"
    res.save(out)
    loaded = SweepResult.load(out)
    assert len(loaded) == len(res)
    assert loaded.grid == res.grid
    assert loaded.base_config["graph_spec"] == "cycle"
    # offline-only fields are not reloaded by load()
    assert loaded.tableaux == {}


def test_mixed_scalar_and_array_routing_consistent():
    res = sweep(BASE, {"p": [0.2]}, seeds=2, analyses=["code_dimension", "entropy_profile"])
    # scalar -> column, array -> array_results, no leakage either way
    assert all("a.code_dimension" in r for r in res.rows)
    assert all("a.entropy_profile" not in r for r in res.rows)
    assert set(res.array_results) == {"entropy_profile"}
    assert len(res.array_results["entropy_profile"]) == len(res)


# ---- purification time (purified_at_layer) is captured ----

_PURIFY = CircuitConfig(
    graph_spec="complete",
    n=10,
    picture="single_ref",
    gating_mode="random_edge",
    measurement_mode="gated",
    p=0.3,
    depth_mode="until_purified",
    warmup_layers=20,
    depth_factor=40,
)


def test_purified_at_layer_in_rows():
    # tau_p is the order parameter of the single-qubit purification study, so it
    # must reach the sweep rows (every row, int or None).
    res = sweep(_PURIFY, {"p": [0.2, 0.4]}, seeds=4)
    assert all("purified_at_layer" in r for r in res.rows)
    vals = [r["purified_at_layer"] for r in res.rows]
    assert all(v is None or isinstance(v, int) for v in vals)


# ---- optional progress bar ----


def test_progress_bar_runs_and_is_correct():
    pytest.importorskip("tqdm")
    res = sweep(BASE, {"p": [0.1, 0.2]}, seeds=3, analyses=["code_dimension"], progress=True)
    plain = sweep(BASE, {"p": [0.1, 0.2]}, seeds=3, analyses=["code_dimension"], progress=False)
    assert _stable(res.rows) == _stable(plain.rows)


def test_progress_missing_tqdm_degrades_gracefully(monkeypatch):
    # progress=True without tqdm must warn and still run (not crash): the
    # bar is cosmetic, the simulation is not.
    import sys

    monkeypatch.setitem(sys.modules, "tqdm.auto", None)  # force ImportError on `from tqdm.auto`
    monkeypatch.setitem(sys.modules, "tqdm", None)
    with pytest.warns(UserWarning, match="tqdm"):
        res = sweep(BASE, {"p": [0.2]}, seeds=2, analyses=["code_dimension"], progress=True)
    assert len(res) == 2
