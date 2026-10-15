"""Keep Numba's on-disk kernel caches in step with the kernel sources (private).

Numba's ``cache=True`` keys each function's cache entry on its own source file
only, while a compiled caller embeds the callees it was compiled with. After an
edit to ``rng.py`` alone, ``program.run_many`` would load from a warm cache and
keep running the old ``stream_state``.

:func:`clear_stale_caches` runs when :mod:`sparsegf2._kernel` is imported,
before any kernel module. It computes the *kernel fingerprint*: a hash of the
Numba version, the Python version and every kernel source, that is all of
``_kernel/*.py`` (no other module of the package defines compiled code, which a
test checks). Then it does two things:

1. **Stamps every cache entry with the fingerprint.** It puts a cache locator
   of its own first in Numba's list of locators. For a file in ``_kernel/`` it
   asks the standard locators, exactly as Numba would (``__pycache__`` next to
   the file, its subdirectory of ``NUMBA_CACHE_DIR`` when that is set, or the
   user-wide cache when ``__pycache__`` is not writable), and adds the
   fingerprint this process imported to their source stamp. Numba ignores and
   overwrites an index whose stamp differs from its own, so an entry compiled
   from other kernel sources is never loaded. That includes an entry saved after
   an update by a process that had imported the old sources. Files elsewhere
   are left to the standard locators.
2. **Deletes stale files.** When the fingerprint differs from the marker file
   next to the cache files, it deletes the ``.nbi`` and ``.nbc`` files of the
   kernel modules there and writes the new marker. Every kernel then
   recompiles on first use.

The locator relies on Numba's private ``CacheImpl._locator_classes``. If that is
missing, or ``NUMBA_CACHE_LOCATOR_CLASSES`` replaces the list, only the deletion
runs. A process that imported the old sources can then still save a stale
entry after an update, and nothing removes it until the kernels change again.
Deleting the ``sparsegf2-kernels.*.sha256`` marker clears them on the next
import.

It compiles nothing, reads about ten files, and never raises: a missing or
read-only cache directory, files another process removed first, or a Numba
without the expected API leave the import working, with a ``logging`` debug
line.

The cache directory is the one Numba picks for a function defined in this
directory: a dispatcher created with ``njit(cache=True)`` and never called (so
never compiled) reports it as ``stats.cache_path``. Numba picks it from the
source file's directory alone, so it is the directory of every kernel.

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


def refresh(cache_dir: Path, sources: Sequence[Path], current: str | None = None) -> int | None:
    """Delete the cached kernels of ``sources`` in ``cache_dir`` unless its marker matches.

    ``current`` is the fingerprint of ``sources`` (computed when not given).
    Returns the number of files deleted, or ``None`` when the marker matched or
    ``cache_dir`` does not exist. The new marker is written (atomically) only
    once every file is gone, so a file that could not be deleted is tried again
    on the next import.
    """
    if not cache_dir.is_dir():
        return None
    if current is None:
        current = fingerprint(sources)
    marker = cache_dir / marker_name()
    try:
        previous: str | None = marker.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        previous = None  # no (readable) marker yet: anything cached may be stale
    if previous == current:
        return None
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
    reason = "no marker" if previous is None else "kernel fingerprint changed"
    _log.debug("%s: deleted %d cached kernel files in %s%s", reason, removed, cache_dir, note)
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


def _in_kernel_dir(py_file: str) -> bool:
    here = os.path.normcase(os.path.abspath(_KERNEL_DIR))
    return os.path.normcase(os.path.dirname(os.path.abspath(py_file))) == here


class _KernelLocator:
    """Numba's own locator for a kernel file, with the kernel fingerprint in its source stamp.

    Numba calls ``from_function`` for every cached function; this one answers
    only for files in ``_kernel/`` and delegates everything but the stamp.
    """

    _sparsegf2_kernels = True  # recognises a copy left by an earlier import
    imported_fingerprint = ""  # of the kernel sources this process imported

    def __init__(self, inner):
        self._inner = inner

    @classmethod
    def from_function(cls, py_func, py_file):
        if not cls.imported_fingerprint or not _in_kernel_dir(py_file):
            return None
        try:
            for other in _locator_list():
                if getattr(other, "_sparsegf2_kernels", False):
                    continue
                inner = other.from_function(py_func, py_file)
                if inner is not None:
                    return cls(inner)
        except Exception as exc:  # Numba's own locators then take over: never fail an import
            _log.debug("kernel cache locator skipped for %s: %r", py_file, exc)
        return None

    def get_source_stamp(self):
        return (self._inner.get_source_stamp(), self.imported_fingerprint)

    def get_cache_path(self):
        return self._inner.get_cache_path()

    def get_disambiguator(self):
        return self._inner.get_disambiguator()

    def ensure_cache_path(self):
        return self._inner.ensure_cache_path()

    def __getattr__(self, name):  # anything else Numba reads, such as _py_file
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


def _locator_list() -> list:
    from numba.core import caching

    impl = getattr(caching, "CacheImpl", None) or caching._CacheImpl
    classes = impl._locator_classes
    if not isinstance(classes, list):
        raise TypeError(f"numba's CacheImpl._locator_classes is a {type(classes).__name__}")
    return classes


def _install_locator(current: str) -> None:
    """Put :class:`_KernelLocator` first, stamping with ``current`` (set once per process)."""
    if not _KernelLocator.imported_fingerprint:
        _KernelLocator.imported_fingerprint = current
    classes = _locator_list()
    if not (classes and classes[0] is _KernelLocator):
        classes[:] = [
            _KernelLocator,
            *(c for c in classes if not getattr(c, "_sparsegf2_kernels", False)),
        ]
    if getattr(numba.config, "CACHE_LOCATOR_CLASSES", ""):
        _log.debug("NUMBA_CACHE_LOCATOR_CLASSES is set: stale kernels are only deleted")


def _numba_cache_dir() -> Path | None:
    """Where Numba caches the kernels (module docstring); ``None`` with the JIT disabled."""
    if numba.config.DISABLE_JIT:
        return None

    def probe():  # never called, so never compiled
        pass

    path = numba.njit(cache=True)(probe).stats.cache_path
    return Path(path) if path else None


def clear_stale_caches() -> None:
    """The import hook (module docstring). Never raises."""
    try:
        sources = kernel_sources()
        current = fingerprint(sources)
    except Exception as exc:  # a broken guard must never break the import
        _log.debug("kernel cache check skipped: %r", exc)
        return
    try:
        _install_locator(current)
    except Exception as exc:
        _log.debug("kernel cache locator not installed (%r): stale kernels are only deleted", exc)
    try:
        cache_dir = _numba_cache_dir()
        if cache_dir is not None:
            refresh(cache_dir, sources, current)
    except Exception as exc:
        _log.debug("kernel cache check skipped: %r", exc)
