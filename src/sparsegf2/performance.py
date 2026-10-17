"""The CPU's topology, and run settings for it: threads, their placement, sweep writers.

:func:`cpu_topology` reads the logical processors, physical cores and caches from the
operating system (Windows ``GetLogicalProcessorInformationEx``, Linux ``/sys``); elsewhere, or
when the query fails, it returns a topology whose ``source`` is ``"unknown"``.
:class:`PerformanceSettings` names the engine's thread count, the logical processors its
threads may run on and the sweep's writer threads; :func:`applied` applies them for one block
and restores the process's affinity and Numba's thread count afterwards, also on an exception.
:func:`preset` gives the measured-best settings for a tableau of ``num_qubits`` qubits:
``"9950x3d"`` (the AMD Ryzen 9 9950X3D layout, checked) or ``"auto"`` (the same rule on any
detected topology).

No setting changes a result: trajectory ``u`` draws its circuit and its probes' samples from
streams keyed by ``u`` alone and writes row ``u`` of the output only
(:mod:`sparsegf2._kernel.ensemble`), so which thread or core runs it, and how many run beside
it, cannot matter. ``docs/SWEEPS.md`` section 12 has the measurements behind :func:`preset`.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import struct
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numba

from sparsegf2._parallel import threads as _threads

__all__ = [
    "PLACEMENTS",
    "PRESETS",
    "CacheDomain",
    "CpuTopology",
    "PerformanceSettings",
    "applied",
    "cpu_topology",
    "preset",
    "resolve",
    "tableau_bytes",
]

PLACEMENTS = ("all", "physical", "vcache", "other_ccd")
"""Named placements: every logical processor; one per physical core; the logical processors
sharing the largest L3 (the V-Cache CCD of an X3D part); those of the other L3 domains."""

PRESETS = ("9950x3d", "auto")


@dataclass(frozen=True)
class CacheDomain:
    """One cache and the logical processors that share it."""

    size: int
    processors: tuple[int, ...]


@dataclass(frozen=True)
class CpuTopology:
    """Logical processors, physical cores and caches, as the operating system reports them."""

    source: str  # "windows", "linux" or "unknown"
    processors: tuple[int, ...] = ()
    cores: tuple[tuple[int, ...], ...] = ()  # each physical core's logical processors
    l3: tuple[CacheDomain, ...] = ()  # one per L3 (per CCD on a Ryzen)
    l2_bytes: int | None = None  # per core
    l1d_bytes: int | None = None  # per core

    @property
    def known(self) -> bool:
        return self.source != "unknown" and bool(self.cores)

    @property
    def x3d(self) -> bool:
        """Two or more L3 domains of different sizes (a 3D V-Cache part)."""
        return len({d.size for d in self.l3}) > 1

    @property
    def vcache(self) -> CacheDomain | None:
        """The L3 domain with the largest cache, on an :attr:`x3d` part."""
        return max(self.l3, key=lambda d: d.size) if self.x3d else None

    def placement(self, placement: str | Sequence[int]) -> tuple[int, ...]:
        """The logical processors of a placement (a name of :data:`PLACEMENTS` or a list)."""
        if not isinstance(placement, str):
            chosen = tuple(sorted({int(p) for p in placement}))
            unknown = sorted(set(chosen) - set(self.processors)) if self.known else []
            if not chosen or unknown:
                raise ValueError(
                    f"placement {list(placement)}: not logical processors of this machine "
                    f"{unknown or ''} (it has {list(self.processors)})"
                )
            return chosen
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {list(PLACEMENTS)} or a list of logical "
                             f"processors, got {placement!r}")  # fmt: skip
        if placement != "all" and not self.known:
            raise ValueError(f"placement {placement!r} needs the CPU topology, which is unknown")
        if placement == "all":
            return self.processors
        if placement == "physical":
            return tuple(sorted(core[0] for core in self.cores))
        big = self.vcache
        if big is None:
            raise ValueError(f"placement {placement!r} needs L3 domains of different sizes "
                             "(an X3D part); this CPU has none")  # fmt: skip
        if placement == "vcache":
            return big.processors
        return tuple(sorted(p for d in self.l3 if d is not big for p in d.processors))


@dataclass(frozen=True)
class PerformanceSettings:
    """Engine threads, where they run, and the sweep's writer threads.

    ``threads=None`` runs one thread per logical processor of the placement (Numba's current
    count for ``"all"``); ``writers=None`` keeps the sweep's default.
    """

    threads: int | None = None
    placement: str | tuple[int, ...] = "all"
    writers: int | None = None

    def __post_init__(self) -> None:
        for name in ("threads", "writers"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                      or value < 1):  # fmt: skip
                raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
        if not isinstance(self.placement, str):
            object.__setattr__(self, "placement", tuple(int(p) for p in self.placement))


def tableau_bytes(num_qubits: int) -> int:
    """The bytes of one trajectory's tableau in the ensemble engine: ``2 N + 1`` columns of
    ``ceil(N / 64)`` words plus the padding of ``mipt._row_padding``, 8 bytes each."""
    from sparsegf2.mipt import _row_padding

    words = (num_qubits + 63) // 64
    return 8 * (2 * num_qubits + 1) * (words + _row_padding(words))


def preset(name: str, num_qubits: int, topology: CpuTopology | None = None) -> PerformanceSettings:
    """The measured-best settings for trajectories of ``num_qubits`` qubits (``N = 2 n`` in the
    purification picture).

    The rule: one thread per physical core when the tableaux of all logical processors would
    not fit in the largest L3 (``processors x tableau_bytes(N) > largest L3``), else every
    logical processor (the default). It is fitted to the measurements, not derived: on the
    9950X3D (32 logical processors, L3 of 96 and 32 MiB) it switches at ``N >= 3521``
    (``n >= 1761`` with a reference per system qubit), between n = 1536, where every logical
    processor ran 3% faster, and n = 1792, where one thread per core ran 9% faster.
    ``"9950x3d"`` refuses a machine with another layout; ``"auto"`` applies the rule to the
    detected topology and returns the default settings when the topology is unknown. Sweep
    writers stay at their default: no other count, and no placement of them, measured faster
    (docs/SWEEPS.md section 12).
    """
    if name not in PRESETS:
        raise ValueError(f"performance preset must be one of {list(PRESETS)}, got {name!r}")
    topo = cpu_topology() if topology is None else topology
    if name == "9950x3d" and not _is_9950x3d(topo):
        raise ValueError(
            "performance preset '9950x3d' was measured on an AMD Ryzen 9 9950X3D (16 cores, 32 "
            "logical processors, L3 domains of 96 and 32 MiB); this machine reports "
            f"{_describe(topo)}: use 'auto'"
        )
    if not topo.known or not topo.l3:
        return PerformanceSettings()
    if len(topo.processors) * tableau_bytes(num_qubits) > max(d.size for d in topo.l3):
        return PerformanceSettings(placement="physical")
    return PerformanceSettings()


def _is_9950x3d(topo: CpuTopology) -> bool:
    sizes = sorted(d.size for d in topo.l3)
    return (
        len(topo.cores) == 16
        and len(topo.processors) == 32
        and sizes == [32 * 2**20, 96 * 2**20]
        and all(len(d.processors) == 16 for d in topo.l3)
    )


def _describe(topo: CpuTopology) -> str:
    if not topo.known:
        return "an unknown topology"
    l3 = ", ".join(f"{d.size // 2**20} MiB" for d in topo.l3) or "none"
    return f"{len(topo.cores)} cores, {len(topo.processors)} logical processors, L3 {l3}"


def resolve(
    performance: str | PerformanceSettings | None, num_qubits: int
) -> PerformanceSettings | None:
    """A preset name or settings as settings for ``num_qubits`` qubits (``None`` stays)."""
    if performance is None or isinstance(performance, PerformanceSettings):
        return performance
    if isinstance(performance, str):
        return preset(performance, num_qubits)
    raise ValueError(
        f"performance must be a preset name {list(PRESETS)} or PerformanceSettings, "
        f"got {performance!r}"
    )


@contextlib.contextmanager
def applied(
    settings: PerformanceSettings | None, topology: CpuTopology | None = None
) -> Iterator[PerformanceSettings | None]:
    """Run a block under ``settings``: the process's threads confined to the placement's
    logical processors, the engine on ``settings.threads`` threads. The affinity and Numba's
    thread count are restored when the block ends, also on an exception."""
    if settings is None:
        yield None
        return
    count = settings.threads
    if settings.placement == "all":
        with _threads(count):
            yield settings
        return
    chosen = (topology or cpu_topology()).placement(settings.placement)
    previous = _get_affinity()
    _set_affinity(chosen)
    try:
        with _threads(len(chosen) if count is None else count):
            yield settings
    finally:
        _set_affinity(previous)


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
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    raise ValueError("thread placement needs Windows or Linux")


def _set_affinity(processors: Sequence[int]) -> None:
    """Confine every thread of this process (Numba's workers included) to ``processors``."""
    if sys.platform == "win32":
        if max(processors) >= 64:
            raise ValueError("thread placement covers the first 64 logical processors only")
        kernel = _kernel32()
        mask = sum(1 << p for p in processors)
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), mask):
            raise OSError(ctypes.get_last_error(), f"SetProcessAffinityMask({mask:#x}) failed")
        return
    if hasattr(os, "sched_setaffinity"):
        for task in Path("/proc/self/task").iterdir():  # every thread, not only this one
            with contextlib.suppress(ProcessLookupError):
                os.sched_setaffinity(int(task.name), processors)
        return
    raise ValueError("thread placement needs Windows or Linux")


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
# topology
# ---------------------------------------------------------------------------


def cpu_topology() -> CpuTopology:
    """The CPU's logical processors, physical cores and caches; ``source="unknown"`` when the
    operating system cannot be asked (never raises)."""
    try:
        if sys.platform == "win32":
            return _windows_topology()
        if Path("/sys/devices/system/cpu").is_dir():
            return _linux_topology()
    except (OSError, ValueError, struct.error, AttributeError):
        pass
    return CpuTopology("unknown", processors=tuple(range(numba.config.NUMBA_NUM_THREADS)))


def _mask_bits(mask: int) -> tuple[int, ...]:
    return tuple(i for i in range(64) if mask >> i & 1)


def _windows_topology() -> CpuTopology:
    """``GetLogicalProcessorInformationEx(RelationAll)``, processor group 0."""
    if sys.platform != "win32":
        raise OSError("not Windows")
    from ctypes import wintypes

    kernel = _kernel32()
    size = wintypes.DWORD(0)
    kernel.GetLogicalProcessorInformationEx(0xFFFF, None, ctypes.byref(size))
    buffer = ctypes.create_string_buffer(size.value)
    if not kernel.GetLogicalProcessorInformationEx(0xFFFF, buffer, ctypes.byref(size)):
        raise OSError(ctypes.get_last_error(), "GetLogicalProcessorInformationEx failed")
    raw, offset = buffer.raw[: size.value], 0
    cores: list[tuple[int, ...]] = []
    l3: list[CacheDomain] = []
    l2 = l1d = None
    while offset < len(raw):
        relation, length = struct.unpack_from("<II", raw, offset)
        body = offset + 8
        if relation == 0:  # RelationProcessorCore: PROCESSOR_RELATIONSHIP
            (groups,) = struct.unpack_from("<H", raw, body + 22)
            mask, group = struct.unpack_from("<QH", raw, body + 24)
            if groups >= 1 and group == 0:
                cores.append(_mask_bits(mask))
        elif relation == 2:  # RelationCache: CACHE_RELATIONSHIP
            level, _, _, cache_size, kind = struct.unpack_from("<BBHII", raw, body)
            mask, group = struct.unpack_from("<QH", raw, body + 32)
            if group == 0:
                if level == 3:
                    l3.append(CacheDomain(cache_size, _mask_bits(mask)))
                elif level == 2:
                    l2 = cache_size
                elif level == 1 and kind == 2:  # CacheData
                    l1d = cache_size
        offset += length
    processors = tuple(sorted(p for core in cores for p in core))
    return CpuTopology("windows", processors, tuple(sorted(cores)), tuple(l3), l2, l1d)


def _cpu_list(text: str) -> tuple[int, ...]:
    """A Linux CPU list such as ``0-3,8,10-11``."""
    out: set[int] = set()
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
    return tuple(sorted(out))


def _size(text: str) -> int:
    text = text.strip()
    scale = {"K": 2**10, "M": 2**20, "G": 2**30}.get(text[-1:].upper(), 1)
    return int(text[:-1] if scale > 1 else text) * scale


def _linux_topology() -> CpuTopology:
    root = Path("/sys/devices/system/cpu")
    online = _cpu_list((root / "online").read_text())
    cores: set[tuple[int, ...]] = set()
    l3: dict[tuple[int, ...], int] = {}
    l2 = l1d = None
    for cpu in online:
        base = root / f"cpu{cpu}"
        topology = base / "topology"
        siblings = topology / "core_cpus_list"
        if not siblings.exists():
            siblings = topology / "thread_siblings_list"
        cores.add(_cpu_list(siblings.read_text()))
        for index in sorted((base / "cache").glob("index*")):
            level = int((index / "level").read_text())
            kind = (index / "type").read_text().strip()
            size = _size((index / "size").read_text())
            if level == 3:
                l3[_cpu_list((index / "shared_cpu_list").read_text())] = size
            elif level == 2:
                l2 = size
            elif level == 1 and kind == "Data":
                l1d = size
    domains = tuple(CacheDomain(size, cpus) for cpus, size in sorted(l3.items()))
    return CpuTopology("linux", online, tuple(sorted(cores)), domains, l2, l1d)
