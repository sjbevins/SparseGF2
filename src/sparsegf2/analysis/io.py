"""Lazy on-disk I/O for sweep results.

All third-party imports (pyarrow, h5py) happen *inside* the functions that need
them, so importing :mod:`sparsegf2.analysis` never pulls in the ``data`` extra.
Each writer raises a clear, actionable error naming the missing extra.

Formats:

* **rows → parquet** (pyarrow): one row per swept ``(params, seed)`` cell. A
  JSON manifest (grid, base config, analysis names) rides in the parquet
  file-level key/value metadata, so the file is self-describing.
* **tableaux → HDF5** (h5py): one ``uint8`` dataset per task key under
  ``/tableaux``, chunked + gzip, with ``attrs`` carrying the cell parameters.
  An ``npz`` fallback (numpy only) is used when h5py is unavailable.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import numpy as np

from sparsegf2.errors import InvalidArgumentError

_MANIFEST_KEY = b"sparsegf2_sweep_manifest"


@contextmanager
def _atomic_path(path: str | Path):
    """Write via a same-directory temp file, then ``os.replace`` into place.

    A crash mid-write must never destroy the previous copy of the file: the
    writers here are used by :meth:`Study.augment` to rewrite the study's
    *only* copy of ``rows.parquet`` in place, where a truncate-on-open write
    would leave an unreadable file and the pre-augment rows would be gone.
    ``os.replace`` is atomic on the same filesystem on POSIX and Windows.
    """
    path = Path(path)
    # A fixed ``path.tmp`` name lets overlapping writers operate on the same
    # file. One can then replace or clean up the other's in-progress payload.
    # Reserve a unique file in the target directory so the final replace stays
    # on the same filesystem while concurrent/nested writes remain isolated.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():  # failed mid-write: drop the partial temp file
            with suppress(OSError):  # pragma: no cover (locked temp on Windows)
                tmp.unlink()


def _require(mod: str, extra: str):
    try:
        return __import__(mod)
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise InvalidArgumentError(
            f"this feature needs the optional '{extra}' extra ({mod}); "
            f"install with: pip install sparsegf2[{extra}]"
        ) from e


# ----------------------------------------------------------------------
# Rows <-> parquet
# ----------------------------------------------------------------------


def write_parquet(
    rows: list[dict[str, Any]], path: str | Path, *, manifest: dict | None = None
) -> None:
    """Write ``rows`` to a parquet file, embedding ``manifest`` in metadata."""
    _require("pyarrow", "data")
    import pyarrow as pa
    import pyarrow.parquet as pq

    try:
        table = pa.Table.from_pylist(rows)
    except Exception as e:  # mixed/unconvertible column types, etc.
        raise InvalidArgumentError(
            "could not build a parquet table from the sweep rows; a column likely "
            f"has mixed or unsupported types ({e}). Drop/normalize that analysis, "
            "or save tableaux for offline analysis instead."
        ) from e
    if manifest is not None:
        meta = dict(table.schema.metadata or {})
        meta[_MANIFEST_KEY] = json.dumps(manifest).encode()
        table = table.replace_schema_metadata(meta)
    with _atomic_path(path) as tmp:
        pq.write_table(table, str(tmp))


def read_parquet(path: str | Path) -> tuple[list[dict[str, Any]], dict | None]:
    """Read a parquet sweep file back to ``(rows, manifest)``."""
    _require("pyarrow", "data")
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    manifest = None
    meta = table.schema.metadata or {}
    if _MANIFEST_KEY in meta:
        manifest = json.loads(meta[_MANIFEST_KEY].decode())
    return table.to_pylist(), manifest


# ----------------------------------------------------------------------
# Tableaux <-> HDF5 / npz
# ----------------------------------------------------------------------


def write_tableaux_hdf5(
    tableaux: dict[str, np.ndarray],
    path: str | Path,
    *,
    attrs_by_key: dict[str, dict] | None = None,
    progress: bool = False,
) -> None:
    """Write ``{key: (2N,2N) uint8}`` to an HDF5 file under ``/tableaux``.

    Datasets are named by index (``t0``, ``t1``, …) with the original key stored
    in ``attrs["key"]``, so arbitrary keys (including ones containing ``/``)
    round-trip safely (a raw ``/`` in a dataset name would otherwise create
    nested groups). ``progress=True`` shows a bar over the (potentially slow,
    gzip-compressed) write so the save phase is not a silent wait.
    """
    from sparsegf2.analysis._progress import resolve_bar

    h5py = _require("h5py", "data")
    attrs_by_key = attrs_by_key or {}
    # `progress` is a bool or a shared bar (so Study.run's whole-study meter keeps
    # advancing through the tableau write instead of starting a second bar).
    bar, owned = resolve_bar(progress, total=len(tableaux), desc="writing tableaux", unit="tab")
    with _atomic_path(path) as tmp, h5py.File(str(tmp), "w") as f:
        grp = f.create_group("tableaux")
        for i, (key, arr) in enumerate(tableaux.items()):
            a = np.asarray(arr, dtype=np.uint8)
            ds = grp.create_dataset(f"t{i}", data=a, compression="gzip", chunks=True)
            ds.attrs["key"] = key
            for k, v in attrs_by_key.get(key, {}).items():
                ds.attrs[k] = v
            if bar is not None:
                bar.update(1)
    if owned and bar is not None:
        bar.close()


def read_tableaux_hdf5(path: str | Path) -> dict[str, np.ndarray]:
    """Read all tableaux from ``/tableaux`` into ``{key: array}`` (in memory)."""
    h5py = _require("h5py", "data")
    out: dict[str, np.ndarray] = {}
    with h5py.File(str(path), "r") as f:
        if "tableaux" not in f:
            raise InvalidArgumentError(
                f"{path}: no '/tableaux' group, not a SparseGF2 tableaux file?"
            )
        grp = f["tableaux"]
        for name in grp:
            ds = grp[name]
            key = ds.attrs.get("key", name)
            out[str(key)] = ds[()]
    return out


def write_tableaux_npz(tableaux: dict[str, np.ndarray], path: str | Path) -> None:
    """numpy-only fallback: store tableaux in a compressed ``.npz``.

    Arrays are stored by index (``t0``, ``t1``, …) with the keys in ``__keys__``,
    so arbitrary keys round-trip and none can collide with ``np.savez``'s
    reserved ``file`` argument. Tableaux are binary, so entries are cast to
    ``uint8``; use :func:`write_arrays_npz` for general-dtype analysis arrays.
    """
    keys = list(tableaux)
    arrs = {f"t{i}": np.asarray(v, np.uint8) for i, v in enumerate(tableaux.values())}
    # Close the handle explicitly (numpy does not) before _atomic_path's replace:
    # a leaked open handle would block the temp cleanup / replace on Windows.
    with _atomic_path(path) as tmp, tmp.open("wb") as fh:
        np.savez_compressed(fh, __keys__=np.array(keys, dtype=str), **arrs)


def read_tableaux_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(str(path)) as z:
        keys = [str(k) for k in z["__keys__"]]
        return {keys[i]: z[f"t{i}"] for i in range(len(keys))}


def write_arrays_npz(arrays: dict[str, np.ndarray], path: str | Path) -> None:
    """Store ``{key: ndarray}`` of **arbitrary dtype** in a compressed ``.npz``.

    The dtype-preserving sibling of :func:`write_tableaux_npz` (which casts to
    ``uint8`` — right for binary tableaux, silently corrupting for analysis
    arrays such as entropy profiles at ``n > 255`` or float-valued results).
    Same indexed layout, so arbitrary keys round-trip.
    """
    keys = list(arrays)
    arrs = {f"t{i}": np.asarray(v) for i, v in enumerate(arrays.values())}
    # Close the handle explicitly (numpy does not) before _atomic_path's replace:
    # a leaked open handle would block the temp cleanup / replace on Windows.
    with _atomic_path(path) as tmp, tmp.open("wb") as fh:
        np.savez_compressed(fh, __keys__=np.array(keys, dtype=str), **arrs)


def read_arrays_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Read a :func:`write_arrays_npz` file back to ``{key: ndarray}``."""
    with np.load(str(path)) as z:
        keys = [str(k) for k in z["__keys__"]]
        return {keys[i]: z[f"t{i}"] for i in range(len(keys))}


__all__ = [
    "read_arrays_npz",
    "read_parquet",
    "read_tableaux_hdf5",
    "read_tableaux_npz",
    "write_arrays_npz",
    "write_parquet",
    "write_tableaux_hdf5",
    "write_tableaux_npz",
]
