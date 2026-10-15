"""Cached kernels follow the kernel sources (:mod:`sparsegf2._kernel._cache`).

Numba keys each cached function on its own source file only, and a cached caller
embeds the callees it was compiled with: editing ``rng.py`` alone used to leave
``program.run_many`` running the old ``stream_state`` from a warm cache. The guard
stamps every kernel cache entry with a fingerprint of all kernel sources and
deletes this package's kernel caches whenever that fingerprint changes. The
end-to-end test edits that callee in a copy of the package, with a process that
imported the old sources saving to the cache after the update, and checks that
later processes run the new code.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numba

import sparsegf2
from sparsegf2._kernel import _cache, program

PACKAGE = Path(sparsegf2.__file__).resolve().parent

# 64 shots of four random measurements: every bit comes from the shot's stream_state.
_SAMPLE = (
    "import sparsegf2\n"
    "c = sparsegf2.Circuit('H 0 1 2 3\\nM 0 1 2 3')\n"
    "r = sparsegf2.sample(c, 64, seed=20261014)\n"
    "print(sparsegf2.__file__)\n"
    "print(''.join(map(str, r.measurements.ravel())))\n"
)
# A process that imports the package, waits (a running MCP server or study), then samples.
_OLD_PROCESS = "import sparsegf2\nprint('ready', flush=True)\ninput()\n" + _SAMPLE

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


def _env(package_root: Path, numba_cache: Path) -> dict[str, str]:
    env = dict(os.environ)
    # The test is about compiled caches, so it compiles even in the JIT-free weekly job.
    env.pop("NUMBA_DISABLE_JIT", None)
    env["NUMBA_CACHE_DIR"] = str(numba_cache)
    env["PYTHONPATH"] = str(package_root)
    return env


def _python(package_root: Path, numba_cache: Path, code: str) -> str:
    """Run ``code`` in a new process that imports ``sparsegf2`` from ``package_root``."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_root,
        env=_env(package_root, numba_cache),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _bits(stdout: str, package_root: Path) -> str:
    where, bits = stdout.split()[-2:]
    assert Path(where).resolve().is_relative_to(package_root.resolve()), where
    assert len(bits) == 256
    return bits


def _cache_files(numba_cache: Path) -> dict[str, int]:
    return {
        p.relative_to(numba_cache).as_posix(): p.stat().st_mtime_ns
        for p in numba_cache.rglob("*.nb[ic]")
    }


def test_callers_follow_an_edited_callee_even_after_a_late_save_by_an_old_process(tmp_path):
    root = tmp_path / "src"
    shutil.copytree(PACKAGE, root / "sparsegf2", ignore=shutil.ignore_patterns("__pycache__"))
    numba_cache = tmp_path / "numba-cache"
    rng = root / "sparsegf2" / "_kernel" / "rng.py"
    # Bytes, not text, so that restoring gives back the file exactly.
    original = rng.read_bytes()
    edited = _edit_stream_state(original)

    # A process imports the original sources and waits (a running MCP server or study).
    old = subprocess.Popen(
        [sys.executable, "-c", _OLD_PROCESS],
        cwd=root,
        env=_env(root, numba_cache),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        assert old.stdout is not None
        assert old.stdout.readline().strip() == "ready"
        rng.write_bytes(edited)  # the update
        _python(root, numba_cache, "import sparsegf2._kernel")  # a new process: the guard runs
        # Now the old process compiles sample()'s kernels from the sources it imported
        # and saves them to the cache. run_many's own file, program.py, is unchanged.
        out, err = old.communicate("go\n", timeout=300)
    finally:
        if old.poll() is None:
            old.kill()
            old.communicate()
    assert old.returncode == 0, err
    before = _bits(out, root)  # the original sources' result

    after_edit = _bits(_python(root, numba_cache, _SAMPLE), root)
    assert after_edit != before, (
        "sample() ran a stream_state compiled from the sources before rng.py changed "
        "(stale Numba cache)"
    )

    # Restoring the file brings the original result back.
    rng.write_bytes(original)
    assert _bits(_python(root, numba_cache, _SAMPLE), root) == before

    # The fingerprint is the same in every process: an import with nothing to compile
    # leaves every cache file as it was.
    files = _cache_files(numba_cache)
    assert any(name.endswith(".nbi") for name in files)
    _python(root, numba_cache, "import sparsegf2._kernel")
    assert _cache_files(numba_cache) == files

    # The marker sits where Numba really wrote the kernels (NUMBA_CACHE_DIR layout).
    markers = list(numba_cache.rglob(f"{_cache.MARKER_PREFIX}*"))
    assert len(markers) == 1, markers
    assert list(markers[0].parent.glob("program.run_many-*.nbi")), markers[0].parent


def test_kernel_cache_entries_carry_the_kernel_fingerprint():
    assert _cache._locator_list()[0] is _cache._KernelLocator
    func = getattr(program.run_many, "py_func", program.run_many)
    locator = _cache._KernelLocator.from_function(func, inspect.getfile(func))
    assert locator is not None
    standard, stamp = locator.get_source_stamp()
    assert stamp == _cache.fingerprint(_cache.kernel_sources())
    assert standard == locator._inner.get_source_stamp()
    assert locator.get_cache_path() == locator._inner.get_cache_path()
    assert locator._py_file == inspect.getfile(func)  # read by Numba's warnings
    # Functions elsewhere keep Numba's own locators.
    assert _cache._KernelLocator.from_function(_edit_stream_state, __file__) is None
    if not numba.config.DISABLE_JIT:
        assert isinstance(program.run_many._cache._impl.locator, _cache._KernelLocator)
        assert Path(locator.get_cache_path()) == Path(program.run_many.stats.cache_path)


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
    with caplog.at_level("DEBUG", logger=_cache.__name__):
        assert _cache.refresh(cache, [sources / "rng.py"]) == 1
    assert "no marker: deleted 1 cached kernel files" in caplog.text
    assert (cache / _cache.marker_name()).is_file()
    # A corrupt marker counts as missing and is replaced.
    (cache / _cache.marker_name()).write_bytes(b"\xff\xfe")
    assert _cache.refresh(cache, [sources / "rng.py"]) == 0
    assert _cache.refresh(cache, [sources / "rng.py"]) is None
    caplog.clear()

    # Without Numba's locator list, stale files are still deleted.
    def broken_list():
        raise AttributeError("numba moved its locator list")

    kernel_cache = tmp_path / "kernel-cache"
    kernel_cache.mkdir()
    (kernel_cache / "program.run_many-1.py314.nbi").write_bytes(b"x")
    with monkeypatch.context() as m, caplog.at_level("DEBUG", logger=_cache.__name__):
        m.setattr(_cache, "_locator_list", broken_list)
        m.setattr(_cache, "_numba_cache_dir", lambda: kernel_cache)
        _cache.clear_stale_caches()
        # A kernel module compiled now gets Numba's own locator instead of an error.
        func = getattr(program.run_many, "py_func", program.run_many)
        assert _cache._KernelLocator.from_function(func, inspect.getfile(func)) is None
    assert "numba moved its locator list" in caplog.text
    assert not (kernel_cache / "program.run_many-1.py314.nbi").exists()
    assert (kernel_cache / _cache.marker_name()).is_file()
    caplog.clear()

    # The import hook itself swallows everything and logs at debug level only.
    def broken():
        raise RuntimeError("numba moved its cache API")

    monkeypatch.setattr(_cache, "_numba_cache_dir", broken)
    with caplog.at_level("DEBUG", logger=_cache.__name__):
        _cache.clear_stale_caches()
    assert [r.levelname for r in caplog.records] == ["DEBUG"]
    assert "numba moved its cache API" in caplog.text


def test_the_marker_lives_in_numbas_cache_directory_for_the_kernels():
    if numba.config.DISABLE_JIT:
        assert _cache._numba_cache_dir() is None
        return
    assert _cache._numba_cache_dir() == Path(program.run_many.stats.cache_path)


# numba's compilation API: anything that turns Python into cached machine code.
_JIT_NAMES = {
    "cfunc",
    "generated_jit",
    "guvectorize",
    "intrinsic",
    "jit",
    "jitclass",
    "njit",
    "overload",
    "overload_attribute",
    "overload_method",
    "register_jitable",
    "stencil",
    "vectorize",
}


def _compiles(tree: ast.Module) -> bool:
    """Whether a module uses numba's compilation API (``njit``, ``@intrinsic``, ...)."""
    numba_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "numba":
                    numba_names.add(alias.asname or "numba")
        elif (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "numba"
            and any(alias.name in _JIT_NAMES for alias in node.names)
        ):
            return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _JIT_NAMES:
            base = node.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in numba_names:
                return True
    return False


def _imported_modules(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "kernel modules use absolute imports"
            names.add(node.module or "")
    return names


def test_only_kernel_modules_define_compiled_code():
    """The fingerprint covers ``_kernel/*.py``, so compiled code must live there and
    depend only on modules whose changes the fingerprint (or a version in it) sees."""
    trees = {
        path.relative_to(PACKAGE).as_posix(): ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(PACKAGE.rglob("*.py"))
    }
    compiling = {name for name, tree in trees.items() if _compiles(tree)}
    assert {name for name in compiling if not name.startswith("_kernel/")} == set()
    assert {"_kernel/bits.py", "_kernel/program.py", "_kernel/rng.py"} <= compiling
    assert "_parallel.py" not in compiling  # numba's thread controls only

    for name in sorted(compiling):
        for module in _imported_modules(trees[name]):
            top = module.split(".")[0]
            allowed = (
                module == "sparsegf2._kernel"
                or module.startswith("sparsegf2._kernel.")
                or top in {"numpy", "numba", "llvmlite"}
                or top in sys.stdlib_module_names  # versioned by the Python in the fingerprint
            )
            assert allowed, f"{name} imports {module}, which the kernel fingerprint does not cover"

    kernel_dir = PACKAGE / "_kernel"
    assert {p.name for p in _cache.kernel_sources()} == {p.name for p in kernel_dir.glob("*.py")}
