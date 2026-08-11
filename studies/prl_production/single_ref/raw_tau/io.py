"""Deterministic, atomic storage helpers for raw-tau production artifacts."""

from __future__ import annotations

import hashlib
import io
import os
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np

_REPLACE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)


def array_sha256(array: object) -> str:
    """Return a content digest of one contiguous NumPy representation."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of a file without loading it twice."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_deterministic_npz(path: str | os.PathLike[str], arrays: dict[str, object]) -> None:
    """Atomically publish a byte-reproducible NPZ with bounded Windows retries.

    Array members are sorted, timestamps are fixed, pickles are forbidden, and
    the temporary file is flushed before replacement.  ``PermissionError`` is
    retried for at most 3.15 seconds because antivirus/indexing readers can
    briefly retain a Windows destination handle.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as raw:
            with zipfile.ZipFile(
                raw,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                for key in sorted(arrays):
                    buffer = io.BytesIO()
                    np.lib.format.write_array(
                        buffer,
                        np.asanyarray(arrays[key]),
                        allow_pickle=False,
                    )
                    info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o600 << 16
                    archive.writestr(info, buffer.getvalue(), compresslevel=6)
            raw.flush()
            os.fsync(raw.fileno())
        for delay in (*_REPLACE_RETRY_DELAYS, None):
            try:
                os.replace(temporary, destination)
                break
            except PermissionError:
                if delay is None:
                    raise
                time.sleep(delay)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["array_sha256", "file_sha256", "write_deterministic_npz"]
