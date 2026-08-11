"""Read immutable snapshots with the shortest practical Windows lock window."""

from __future__ import annotations

import contextlib
import io
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np


@contextlib.contextmanager
def _open_windows_shared_handle(path: str | os.PathLike[str]) -> Iterator[int]:
    """Open a Windows read handle that shares read, write, and delete access."""
    if os.name != "nt":
        raise RuntimeError("Windows shared handles are available only on Windows")

    resolved = Path(path)
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    generic_read = 0x80000000
    share_read_write_delete = 0x00000001 | 0x00000002 | 0x00000004
    open_existing = 3
    file_attribute_normal = 0x00000080
    raw_handle = create_file(
        str(resolved),
        generic_read,
        share_read_write_delete,
        None,
        open_existing,
        file_attribute_normal,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if raw_handle == invalid_handle:
        error = ctypes.WinError(ctypes.get_last_error())
        error.filename = str(resolved)
        raise error
    try:
        yield int(raw_handle)
    finally:
        if not close_handle(raw_handle):
            raise ctypes.WinError(ctypes.get_last_error())


def _read_windows_handle(handle: int, path: Path) -> bytes:
    """Read an entire regular file through an existing Windows handle."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_file_size = kernel32.GetFileSizeEx
    get_file_size.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong))
    get_file_size.restype = wintypes.BOOL
    read_file = kernel32.ReadFile
    read_file.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    )
    read_file.restype = wintypes.BOOL

    size = ctypes.c_longlong()
    if not get_file_size(handle, ctypes.byref(size)):
        error = ctypes.WinError(ctypes.get_last_error())
        error.filename = str(path)
        raise error
    if size.value < 0:
        raise OSError(f"negative file size reported for {path}")

    chunks: list[bytes] = []
    remaining = size.value
    while remaining:
        requested = min(remaining, 1024 * 1024)
        buffer = ctypes.create_string_buffer(requested)
        received = wintypes.DWORD()
        if not read_file(handle, buffer, requested, ctypes.byref(received), None):
            error = ctypes.WinError(ctypes.get_last_error())
            error.filename = str(path)
            raise error
        if received.value == 0:
            break
        chunks.append(buffer.raw[: received.value])
        remaining -= received.value
    if remaining:
        raise OSError(f"short read from {path}: missing {remaining} of {size.value} bytes")
    return b"".join(chunks)


def read_shared_bytes(path: str | os.PathLike[str]) -> bytes:
    """Return a byte snapshot through a maximally shared Windows read handle."""
    resolved = Path(path)
    if os.name != "nt":
        return resolved.read_bytes()
    with _open_windows_shared_handle(resolved) as handle:
        return _read_windows_handle(handle, resolved)


@contextlib.contextmanager
def load_npz_snapshot(path: str | os.PathLike[str]) -> Iterator[np.lib.npyio.NpzFile]:
    """Load an NPZ from a closed-handle byte snapshot with pickles disabled."""
    payload = read_shared_bytes(path)
    buffer = io.BytesIO(payload)
    try:
        with np.load(buffer, allow_pickle=False) as data:
            yield data
    finally:
        buffer.close()
