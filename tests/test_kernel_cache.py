"""Compiled kernels are never stale (:mod:`sparsegf2._kernel._cache`).

Numba keys each cached function on its own source file only, and a cached caller
embeds the callees it was compiled with: editing ``rng.py`` alone used to leave
``program.run_many`` running the old ``stream_state`` from a warm cache. The guard
clears this package's kernel caches whenever any kernel source changes. The
end-to-end test edits that callee in a copy of the package and checks that its
caller picks the change up, in fresh processes with their own Numba cache.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import sparsegf2
from sparsegf2._kernel import _cache

PACKAGE = Path(sparsegf2.__file__).resolve().parent

# 64 shots of four random measurements: every bit comes from the shot's stream_state.
_SAMPLE = (
    "import sparsegf2\n"
    "c = sparsegf2.Circuit('H 0 1 2 3\\nM 0 1 2 3')\n"
    "r = sparsegf2.sample(c, 64, seed=20261014)\n"
    "print(sparsegf2.__file__)\n"
    "print(''.join(map(str, r.measurements.ravel())))\n"
)

# A constant inside rng.stream_state (the key word that seeds the increment). The edit
# replaces one line in place, so every other line of rng.py keeps its content and number.
_ORIGINAL_LINE = b"    gamma = _sm(key[1])\n"
_EDITED_LINE = b"    gamma = _sm(key[0])\n"


def _edit_stream_state(source: bytes) -> bytes:
    body = re.search(rb"\ndef stream_state\(.*?(?=\ndef |\Z)", source, re.S)
    assert body is not None, "rng.py no longer defines stream_state; update this test"
    assert body.group().count(_ORIGINAL_LINE) == 1, "stream_state changed; update _ORIGINAL_LINE"
    edited = body.group().replace(_ORIGINAL_LINE, _EDITED_LINE)
    return source[: body.start()] + edited + source[body.end() :]


def _sample_bits(package_root: Path, numba_cache: Path) -> str:
    """Run ``sample()`` in a new process that imports ``sparsegf2`` from ``package_root``."""
    env = dict(os.environ)
    # The test is about compiled caches, so it compiles even in the JIT-free weekly job.
    env.pop("NUMBA_DISABLE_JIT", None)
    env["NUMBA_CACHE_DIR"] = str(numba_cache)
    env["PYTHONPATH"] = str(package_root)
    proc = subprocess.run(
        [sys.executable, "-c", _SAMPLE],
        cwd=package_root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    where, bits = proc.stdout.split()
    assert Path(where).resolve().is_relative_to(package_root.resolve()), where
    assert len(bits) == 256
    return bits


def test_editing_a_kernel_callee_recompiles_its_callers(tmp_path):
    root = tmp_path / "src"
    shutil.copytree(PACKAGE, root / "sparsegf2", ignore=shutil.ignore_patterns("__pycache__"))
    numba_cache = tmp_path / "numba-cache"
    rng = root / "sparsegf2" / "_kernel" / "rng.py"
    # Bytes, not text, so that restoring gives back the file exactly.
    original = rng.read_bytes()
    edited = _edit_stream_state(original)

    before = _sample_bits(root, numba_cache)  # compiles and fills the cache

    rng.write_bytes(edited)
    after_edit = _sample_bits(root, numba_cache)
    assert after_edit != before, (
        "sample() ran the stream_state compiled before rng.py changed (stale Numba cache)"
    )

    rng.write_bytes(original)
    assert _sample_bits(root, numba_cache) == before

    # The marker sits where Numba really wrote the kernels (NUMBA_CACHE_DIR layout).
    markers = list(numba_cache.rglob(f"{_cache.MARKER_PREFIX}*"))
    assert len(markers) == 1, markers
    assert list(markers[0].parent.glob("program.run_many-*.nbi")), markers[0].parent


def test_only_stale_kernel_cache_files_are_removed(tmp_path):
    sources = tmp_path / "kernel"
    sources.mkdir()
    (sources / "rng.py").write_text("A = 1\n", encoding="utf-8")
    (sources / "program.py").write_text("B = 2\n", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    kernel_files = [
        "rng.stream_state-139.py314.nbi",
        "rng.stream_state-139.py314.1.nbc",
        "program.run_many-188.py311.nbi",
        "program.run_many-188.py311.2.nbc",
    ]
    other_files = [
        "programs.f-1.py314.nbi",  # another module whose name starts like a kernel's
        "graphs.g-7.py314.nbi",
        "rng.cpython-314.pyc",
        "notes.txt",
    ]
    for name in kernel_files + other_files:
        (cache / name).write_bytes(b"x")

    assert _cache.refresh(cache, [sources / "rng.py", sources / "program.py"]) == 4
    assert {p.name for p in cache.iterdir()} == {*other_files, _cache.marker_name()}

    # The marker matches now: new cache files survive the next import.
    (cache / kernel_files[0]).write_bytes(b"x")
    assert _cache.refresh(cache, [sources / "rng.py", sources / "program.py"]) is None
    assert (cache / kernel_files[0]).exists()

    # Any change to any kernel source clears them again.
    (sources / "program.py").write_text("B = 3\n", encoding="utf-8")
    assert _cache.refresh(cache, [sources / "rng.py", sources / "program.py"]) == 1
    assert not (cache / kernel_files[0]).exists()


def test_the_fingerprint_covers_names_contents_and_versions(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_bytes(b"x = 1\n")
    b.write_bytes(b"y = 2\n")
    base = _cache.fingerprint([a, b])
    assert base == _cache.fingerprint([b, a])  # order of discovery does not matter
    a.write_bytes(b"x = 2\n")
    assert _cache.fingerprint([a, b]) != base
    a.write_bytes(b"x = 1\n")
    c = tmp_path / "c.py"
    a.rename(c)
    assert _cache.fingerprint([c, b]) != base
    c.rename(a)
    assert _cache.fingerprint([a, b]) == base
    assert _cache.fingerprint([a, b], numba_version="0.0.0") != base


def test_the_guard_never_breaks_the_import(tmp_path, monkeypatch, caplog):
    sources = tmp_path / "kernel"
    sources.mkdir()
    (sources / "rng.py").write_text("A = 1\n", encoding="utf-8")
    # A cache directory that does not exist, or whose marker cannot be written.
    assert _cache.refresh(tmp_path / "missing", [sources / "rng.py"]) is None
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / _cache.marker_name()).mkdir()
    assert _cache.refresh(cache, [sources / "rng.py"]) == 0

    # A cache file that cannot be deleted (in use, or a read-only directory): the
    # marker is not written, so the next import tries again.
    (cache / _cache.marker_name()).rmdir()
    (cache / "rng.f-1.py314.nbi").write_bytes(b"x")

    def refuse(path, *args, **kwargs):
        raise PermissionError(path)

    with monkeypatch.context() as m:
        m.setattr(os, "unlink", refuse)
        assert _cache.refresh(cache, [sources / "rng.py"]) == 0
    assert (cache / "rng.f-1.py314.nbi").exists()
    assert not (cache / _cache.marker_name()).exists()
    assert _cache.refresh(cache, [sources / "rng.py"]) == 1
    assert (cache / _cache.marker_name()).is_file()
    # A corrupt marker counts as a mismatch and is replaced.
    (cache / _cache.marker_name()).write_bytes(b"\xff\xfe")
    assert _cache.refresh(cache, [sources / "rng.py"]) == 0
    assert _cache.refresh(cache, [sources / "rng.py"]) is None

    # The import hook itself swallows everything and logs at debug level only.
    def broken():
        raise RuntimeError("numba moved its cache API")

    monkeypatch.setattr(_cache, "_numba_cache_dir", broken)
    with caplog.at_level("DEBUG", logger=_cache.__name__):
        _cache.clear_stale_caches()
    assert [r.levelname for r in caplog.records] == ["DEBUG"]
    assert "numba moved its cache API" in caplog.text


def test_the_marker_lives_in_numbas_cache_directory_for_the_kernels():
    import numba

    if numba.config.DISABLE_JIT:
        assert _cache._numba_cache_dir() is None
        return
    from sparsegf2._kernel.program import run_many

    assert _cache._numba_cache_dir() == Path(run_many.stats.cache_path)


_JIT = re.compile(
    r"@\s*(?:numba\.)?(?:njit|jit|vectorize|guvectorize|cfunc|stencil|jitclass)\b"
    r"|\b(?:njit|jit|vectorize|guvectorize|cfunc|stencil|jitclass)\s*\("
    r"|from numba[\w.]* import [^\n]*\b(?:njit|jit|vectorize|guvectorize|cfunc|stencil)\b"
)


def test_only_kernel_modules_define_compiled_code():
    """The fingerprint covers ``_kernel/*.py``, so compiled code must live there."""
    outside = [
        path.relative_to(PACKAGE).as_posix()
        for path in sorted(PACKAGE.rglob("*.py"))
        if "_kernel" not in path.relative_to(PACKAGE).parts
        and _JIT.search(path.read_text(encoding="utf-8"))
    ]
    assert outside == []
    assert _JIT.search((PACKAGE / "_kernel" / "program.py").read_text(encoding="utf-8"))
    kernel_dir = PACKAGE / "_kernel"
    assert {p.name for p in _cache.kernel_sources()} == {p.name for p in kernel_dir.glob("*.py")}
