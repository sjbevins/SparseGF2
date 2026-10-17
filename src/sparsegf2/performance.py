"""Opt-in thread placement: one thread per physical core.

The default runs the engine on every logical processor, without pinning, and is the fastest
setting measured on an AMD Ryzen 9 9950X3D up to n = 1536. The only placement measured to help
is one thread per physical core, from about n = 1800 (9% more trajectories per second at
n = 1792, 19 to 31% at n = 2048; docs/SWEEPS.md section 12). Placements on one CCD, the V-Cache
CCD included, gave no measured gain and are not offered.

:func:`cpu_cores` reads each physical core's logical processors from the operating system
(Windows ``GetLogicalProcessorInformationEx``, Linux ``/sys``; ``()`` elsewhere or when the
query fails). :func:`physical_cores` confines the process's threads to one logical processor
per core for one block and restores the affinity and Numba's thread count afterwards, also on
an exception. It changes no result: trajectory ``u`` draws from streams keyed by ``u`` alone and
writes its own output row (:mod:`sparsegf2._kernel.ensemble`), whichever thread or core runs it.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import struct
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from sparsegf2._parallel import threads as _threads

__all__ = ["cpu_cores", "physical_cores"]


@contextlib.contextmanager
def physical_cores(threads: int | None = None) -> Iterator[tuple[int, ...]]:
    """Run a block with the process's threads on one logical processor per physical core and
    Numba on ``threads`` threads (default: one per core); yields those processors.

    Process-wide (one such block at a time); the affinity and Numba's thread count are
    restored when the block ends, also on an exception. Raises ``ValueError`` when the cores
    cannot be read (neither Windows nor Linux).
    """
    cores = cpu_cores()
    if not cores:
        raise ValueError("one thread per physical core needs the CPU's cores (Windows or Linux)")
    chosen = tuple(sorted(core[0] for core in cores))
    previous = _get_affinity()
    _set_affinity(chosen)
    try:
        with _threads(len(chosen) if threads is None else threads):
            yield chosen
    finally:
        _set_affinity(previous)


def cpu_cores() -> tuple[tuple[int, ...], ...]:
    """Each physical core's logical processors, sorted; ``()`` when the operating system cannot
    be asked (never raises). Windows: processor group 0."""
    try:
        if sys.platform == "win32":
            return _windows_cores()
        if Path("/sys/devices/system/cpu/online").exists():
            return _linux_cores()
    except (OSError, ValueError, struct.error):
        pass
    return ()


# ---------------------------------------------------------------------------
# affinity
# ---------------------------------------------------------------------------


def _get_affinity() -> tuple[int, ...]:
    if sys.platform == "win32":
        kernel = _kernel32()
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(
            kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)
        ):
            raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask failed")
        return tuple(i for i in range(64) if process.value >> i & 1)
    return tuple(sorted(os.sched_getaffinity(0)))


def _set_affinity(processors: Sequence[int]) -> None:
    """Confine every thread of this process (Numba's workers included) to ``processors``."""
    if sys.platform == "win32":
        kernel = _kernel32()
        mask = sum(1 << p for p in processors if p < 64)
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), mask):
            raise OSError(ctypes.get_last_error(), f"SetProcessAffinityMask({mask:#x}) failed")
        return
    for task in Path("/proc/self/task").iterdir():  # every thread, not only this one
        with contextlib.suppress(ProcessLookupError):
            os.sched_setaffinity(int(task.name), processors)


def _kernel32():
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p]
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel.GetLogicalProcessorInformationEx.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)
    ]  # fmt: skip
    return kernel


# ---------------------------------------------------------------------------
# cores
# ---------------------------------------------------------------------------


def _windows_cores() -> tuple[tuple[int, ...], ...]:
    """``GetLogicalProcessorInformationEx(RelationProcessorCore)``, processor group 0."""
    if sys.platform != "win32":
        raise OSError("not Windows")
    from ctypes import wintypes

    kernel = _kernel32()
    size = wintypes.DWORD(0)
    kernel.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    buffer = ctypes.create_string_buffer(size.value)
    if not kernel.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(size)):
        raise OSError(ctypes.get_last_error(), "GetLogicalProcessorInformationEx failed")
    raw, offset, cores = buffer.raw[: size.value], 0, []
    while offset < len(raw):  # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX records
        relation, length = struct.unpack_from("<II", raw, offset)
        if relation == 0:  # PROCESSOR_RELATIONSHIP: its first GROUP_AFFINITY
            mask, group = struct.unpack_from("<QH", raw, offset + 32)
            if group == 0:
                cores.append(tuple(i for i in range(64) if mask >> i & 1))
        offset += length
    return tuple(sorted(cores))


def _cpu_list(text: str) -> tuple[int, ...]:
    """A Linux CPU list such as ``0-3,8,10-11``."""
    out: set[int] = set()
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
    return tuple(sorted(out))


def _linux_cores() -> tuple[tuple[int, ...], ...]:
    root = Path("/sys/devices/system/cpu")
    cores = set()
    for cpu in _cpu_list((root / "online").read_text()):
        topology = root / f"cpu{cpu}" / "topology"
        siblings = topology / "core_cpus_list"
        if not siblings.exists():
            siblings = topology / "thread_siblings_list"
        cores.add(_cpu_list(siblings.read_text()))
    return tuple(sorted(cores))
