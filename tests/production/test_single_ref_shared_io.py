from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from studies.prl_production.single_ref.monitor import _read_point
from studies.prl_production.single_ref.shared_io import (
    _open_windows_shared_handle,
    _read_windows_handle,
    load_npz_snapshot,
    read_shared_bytes,
)


def test_read_shared_bytes_returns_exact_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "point.bin"
    expected = bytes(range(256)) * 4
    path.write_bytes(expected)

    assert read_shared_bytes(path) == expected


def test_read_shared_bytes_propagates_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_shared_bytes(tmp_path / "missing.npz")


def test_load_npz_snapshot_reads_arrays_after_source_can_be_replaced(tmp_path: Path) -> None:
    path = tmp_path / "point.npz"
    np.savez(path, values=np.arange(5, dtype=np.int32))

    with load_npz_snapshot(path) as data:
        replacement = tmp_path / "replacement.npz"
        np.savez(replacement, values=np.arange(3, dtype=np.int64))
        os.replace(replacement, path)
        assert np.array_equal(data["values"], np.arange(5, dtype=np.int32))

    with np.load(path, allow_pickle=False) as data:
        assert np.array_equal(data["values"], np.arange(3, dtype=np.int64))


def test_load_npz_snapshot_keeps_pickle_loading_disabled(tmp_path: Path) -> None:
    path = tmp_path / "objects.npz"
    np.savez(path, values=np.array([{"unsafe": True}], dtype=object))

    with (
        load_npz_snapshot(path) as data,
        pytest.raises(ValueError, match="Object arrays cannot be loaded"),
    ):
        data["values"]


def test_monitor_reads_point_from_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "point.npz"
    np.savez(
        path,
        complete=np.array([1, 1, 0], dtype=np.uint8),
        event_observed=np.array([1, 0, 0], dtype=np.uint8),
    )

    assert _read_point(path, {}) == (2, 1, 0)


@pytest.mark.skipif(os.name != "nt", reason="Windows file-sharing semantics")
def test_windows_shared_delete_handle_permits_atomic_replace(tmp_path: Path) -> None:
    path = tmp_path / "point.bin"
    replacement = tmp_path / "replacement.bin"
    path.write_bytes(b"old snapshot")

    with _open_windows_shared_handle(path) as handle:
        os.replace(path, replacement)
        assert _read_windows_handle(handle, path) == b"old snapshot"

    assert not path.exists()
    assert replacement.read_bytes() == b"old snapshot"
