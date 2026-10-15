"""Keep Numba's on-disk kernel caches in step with the kernel sources (private).

Numba's ``cache=True`` keys each function's cache entry on its own source file
only, while a compiled caller embeds the callees it was compiled with. After an
edit to ``rng.py`` alone, ``program.run_many`` would load from a warm cache and
keep running the old ``stream_state``.

:func:`clear_stale_caches` runs when :mod:`sparsegf2._kernel` is imported,
before any kernel module. It hashes the Numba version, the Python version and
every kernel source: all of ``_kernel/*.py``, since no other module of the
package defines compiled code (a test checks this). When the hash differs from
the marker file next to the cache files, it deletes the ``.nbi`` and ``.nbc``
files of these modules there, so every kernel recompiles on first use, and then
writes the new hash. It compiles nothing, reads about ten files, and never
raises: a missing or read-only cache directory, files another process removed
first, or a Numba without the expected API leave the import working, with one
``logging`` debug line.

The cache directory is the one Numba itself picks for a function defined in
this directory: a dispatcher created with ``njit(cache=True)`` and never called
(so never compiled) reports it as ``stats.cache_path``. Numba picks it from the
source file's directory alone (``__pycache__`` next to the file, its
subdirectory of ``NUMBA_CACHE_DIR`` when that is set, or the user-wide cache
when ``__pycache__`` is not writable), so it is the directory of every kernel.

The marker is per Python version (``sparsegf2-kernels.cpython-314.sha256``), so
environments with different Pythons that share a checkout do not clear each
other's caches on every import. Cache files of every Python version are deleted
on a mismatch: a changed source makes all of them stale.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numba

__all__ = [
    "MARKER_PREFIX",
    "clear_stale_caches",
    "fingerprint",
    "kernel_sources",
    "marker_name",
    "refresh",
]

_log = logging.getLogger(__name__)

MARKER_PREFIX = "sparsegf2-kernels"
_KERNEL_DIR = Path(__file__).parent


def _python_tag() -> str:
    """``cpython-314``, plus ABI flags such as ``t``: the Python that compiled the kernels."""
    major, minor = sys.version_info[:2]
    return f"{sys.implementation.name}-{major}{minor}{getattr(sys, 'abiflags', '')}"


def marker_name() -> str:
    """The marker's file name in the cache directory, one per Python version."""
    return f"{MARKER_PREFIX}.{_python_tag()}.sha256"


def kernel_sources() -> list[Path]:
    """Every module whose compiled code can end up in a cached kernel: ``_kernel/*.py``."""
    return sorted(_KERNEL_DIR.glob("*.py"))


def fingerprint(sources: Sequence[Path], *, numba_version: str | None = None) -> str:
    """SHA-256 over the Numba and Python versions and the names and bytes of ``sources``."""
    h = hashlib.sha256()
    version = numba.__version__ if numba_version is None else numba_version
    h.update(f"numba {version}\npython {_python_tag()}\n".encode())
    for path in sorted(sources, key=lambda p: p.name):
        data = path.read_bytes()
        h.update(f"{path.name} {len(data)}\n".encode())
        h.update(data)
    return h.hexdigest()


def refresh(cache_dir: Path, sources: Sequence[Path]) -> int | None:
    """Delete the cached kernels of ``sources`` in ``cache_dir`` unless its marker matches.

    Returns the number of files deleted, or ``None`` when the marker matched or
    ``cache_dir`` does not exist. The new marker is written (atomically) only
    once every file is gone, so a file that could not be deleted is tried again
    on the next import.
    """
    if not cache_dir.is_dir():
        return None
    current = fingerprint(sources)
    marker = cache_dir / marker_name()
    try:
        if marker.read_text(encoding="utf-8").strip() == current:
            return None
    except (OSError, UnicodeDecodeError):
        pass  # no (readable) marker yet: anything cached may be stale
    modules = {path.stem for path in sources}
    removed = failed = 0
    with os.scandir(cache_dir) as entries:
        names = [e.name for e in entries]
    for name in names:
        # Numba names them <module>.<qualname>-<line>.py<ver>.nbi and ...<n>.nbc.
        if not name.endswith((".nbi", ".nbc")) or name.split(".", 1)[0] not in modules:
            continue
        try:
            os.unlink(cache_dir / name)
            removed += 1
        except FileNotFoundError:
            pass  # another process deleted it first
        except OSError:
            failed += 1
    if failed:
        note = f"; {failed} could not be deleted, so the next import tries again"
    elif not _write_marker(marker, current):
        note = "; the marker could not be written, so the next import checks again"
    else:
        note = ""
    _log.debug(
        "kernel sources changed: deleted %d cached kernel files in %s%s", removed, cache_dir, note
    )
    return removed


def _write_marker(marker: Path, value: str) -> bool:
    tmp = marker.with_name(f"{marker.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(value + "\n", encoding="utf-8")
        os.replace(tmp, marker)
        return True
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        return False


def _numba_cache_dir() -> Path | None:
    """Where Numba caches the kernels (module docstring); ``None`` with the JIT disabled."""
    if numba.config.DISABLE_JIT:
        return None

    def probe():  # never called, so never compiled
        pass

    path = numba.njit(cache=True)(probe).stats.cache_path
    return Path(path) if path else None


def clear_stale_caches() -> None:
    """The import hook: :func:`refresh` Numba's kernel cache directory. Never raises."""
    try:
        cache_dir = _numba_cache_dir()
        if cache_dir is not None:
            refresh(cache_dir, kernel_sources())
    except Exception as exc:  # a broken guard must never break the import
        _log.debug("kernel cache check skipped: %r", exc)
