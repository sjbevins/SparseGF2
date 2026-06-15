"""On-disk I/O round-trips for sweep results (needs the `data` extra)."""

from __future__ import annotations

import numpy as np
import pytest

from sparsegf2.analysis import SweepResult, sweep
from sparsegf2.circuits import CircuitConfig

pytest.importorskip("pyarrow")

BASE = CircuitConfig(graph_spec="cycle", n=10, picture="purification", p=0.2, depth_factor=3)


def test_parquet_round_trip(tmp_path):
    res = sweep(BASE, {"p": [0.1, 0.2, 0.3]}, seeds=3, analyses=["code_dimension"])
    path = tmp_path / "rows.parquet"
    res.to_parquet(path)
    loaded = SweepResult.load(path)
    assert len(loaded) == len(res)
    assert loaded.grid == res.grid
    assert loaded.base_config["graph_spec"] == "cycle"
    assert sorted(r["a.code_dimension"] for r in loaded.rows) == sorted(
        r["a.code_dimension"] for r in res.rows
    )


def test_save_writes_tableaux_sidecar(tmp_path):
    res = sweep(BASE, {"p": [0.2]}, seeds=2, collect_tableaus=True)
    out = tmp_path / "study.parquet"
    res.save(out)
    assert out.exists()
    side = out.with_suffix(".tableaux.h5")
    assert side.exists()


def test_hdf5_tableaux_round_trip(tmp_path):
    pytest.importorskip("h5py")
    from sparsegf2.analysis import io

    tabs = {"n=4__seed=0": np.eye(8, dtype=np.uint8), "n=4__seed=1": np.ones((8, 8), np.uint8)}
    path = tmp_path / "t.h5"
    io.write_tableaux_hdf5(tabs, path, attrs_by_key={"n=4__seed=0": {"n": "4", "seed": "0"}})
    back = io.read_tableaux_hdf5(path)
    assert set(back) == set(tabs)
    for k in tabs:
        assert np.array_equal(back[k], tabs[k])


def test_npz_tableaux_round_trip(tmp_path):
    from sparsegf2.analysis import io

    tabs = {"a": np.eye(4, dtype=np.uint8), "b": np.zeros((4, 4), np.uint8)}
    path = tmp_path / "t.npz"
    io.write_tableaux_npz(tabs, path)
    back = io.read_tableaux_npz(path)
    assert set(back) == set(tabs)
    for k in tabs:
        assert np.array_equal(back[k], tabs[k])


def test_parquet_manifest_and_empty_rows(tmp_path):
    from sparsegf2.analysis import io

    path = tmp_path / "m.parquet"
    io.write_parquet([], path, manifest={"grid": {"p": [0.1]}, "seeds": [0]})
    rows, manifest = io.read_parquet(path)
    assert rows == []
    assert manifest == {"grid": {"p": [0.1]}, "seeds": [0]}


def test_parquet_null_and_list_columns(tmp_path):
    from sparsegf2.analysis import io

    rows = [{"x": 1, "y": None, "z": [1, 2, 3]}, {"x": 2, "y": 5, "z": [4, 5, 6]}]
    path = tmp_path / "n.parquet"
    io.write_parquet(rows, path)
    back, _ = io.read_parquet(path)
    assert back[0]["y"] is None and back[1]["y"] == 5
    assert list(back[0]["z"]) == [1, 2, 3]


def test_parquet_mixed_type_column_raises_invalid_argument(tmp_path):
    from sparsegf2.analysis import io
    from sparsegf2.errors import InvalidArgumentError

    rows = [{"x": 1}, {"x": "a string"}, {"x": [1, 2]}]  # genuinely unconvertible
    with pytest.raises(InvalidArgumentError):
        io.write_parquet(rows, tmp_path / "bad.parquet")


def test_hdf5_key_with_slash_round_trips(tmp_path):
    pytest.importorskip("h5py")
    from sparsegf2.analysis import io

    tabs = {"p=0.1/seed=0": np.eye(4, dtype=np.uint8), "a/b/c": np.ones((4, 4), np.uint8)}
    path = tmp_path / "slash.h5"
    io.write_tableaux_hdf5(tabs, path)
    back = io.read_tableaux_hdf5(path)
    assert set(back) == set(tabs)
    for k in tabs:
        assert np.array_equal(back[k], tabs[k])


def test_hdf5_missing_group_raises(tmp_path):
    pytest.importorskip("h5py")
    import h5py

    from sparsegf2.analysis import io
    from sparsegf2.errors import InvalidArgumentError

    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("not_tableaux", data=np.zeros(3))
    with pytest.raises(InvalidArgumentError):
        io.read_tableaux_hdf5(path)


def test_npz_key_named_file(tmp_path):
    from sparsegf2.analysis import io

    tabs = {"file": np.eye(4, dtype=np.uint8)}  # 'file' is reserved by np.savez
    path = tmp_path / "f.npz"
    io.write_tableaux_npz(tabs, path)
    back = io.read_tableaux_npz(path)
    assert set(back) == {"file"}
    assert np.array_equal(back["file"], tabs["file"])


def test_parquet_round_trip_all_none_columns(tmp_path):
    # pure_state: code_dimension is always None -> all-None columns.
    res = sweep(
        CircuitConfig(graph_spec="cycle", n=8, picture="pure_state", p=0.2, depth_factor=2),
        {"p": [0.1, 0.2]},
        seeds=2,
        analyses=["code_dimension"],
    )
    path = tmp_path / "none.parquet"
    res.to_parquet(path)
    loaded = SweepResult.load(path)
    assert all(r["code_dimension"] is None for r in loaded.rows)
    assert all(r["a.code_dimension"] is None for r in loaded.rows)
