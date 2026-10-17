"""Where the time of phase-free sweeps goes, and what thread placement and the kernels' memory
layout do to it, on the machine that runs this (written for an AMD Ryzen 9 9950X3D).

The workload is the sweep's (``sparsegf2.study.sweep``): Watts–Strogatz graphs (k = 4,
beta = 0.1), the purification picture, ``n // 2`` random-edge gates per layer, T = 4n layers,
``code_dimension`` at every layer and a ``tableau_snapshot`` on the geometric checkpoints
(count 12), at n = 64, 256, 1024 and p = 0.05, 0.2. Sections:

* ``breakdown``: one thread, milliseconds per trajectory of the full workload, without the
  snapshots, without measurements (p = 0: the gates alone) and with ``code_dimension`` at the
  last layer only, and the fixed cost of one ``MonitoredCircuit.run`` call;
* ``placements``: trajectories per second at 1, 8, 16 and 32 threads, on every logical
  processor, one per physical core, the V-Cache CCD only and the other CCD only
  (``sparsegf2.performance.applied``), and at n = 1280 to 2048 (T = 256) for the
  preset's rule; every configuration's common trajectories must be identical;
* ``padding``: the engine with and without the tableau padding of ``mipt._row_padding``;
* ``gates``: a compiled micro-benchmark of ``apply_2q`` and of its linear part with the masks
  read before the word loop (as now), inside it (as before) and from a compact 16-bit table;
* ``scheduling``: Numba's parallel chunk size with ``stop_when_purified``;
* ``small_batches``: runs of 1 to 32 trajectories on 1 thread and on all of them;
* ``end_to_end``: ``sweep.run`` (``sweep_throughput.measure``) with 2, 4 and 8 writer threads,
  with the engine pinned to the V-Cache CCD and the writers to the other one (Windows), and at
  n = 1024 with parts of 18 trajectories (the cap before chunks became whole thread rounds).

Before each measurement the script records the CPU load of other processes (percent of all
logical processors, over 0.5 s after a 0.2 s settle). Usage, from the repository root, on an
idle machine::

    python benchmarks/performance_x3d.py              # about 7 minutes on a 9950X3D
    python benchmarks/performance_x3d.py --quick --out <dir>   # smoke test
    python benchmarks/performance_x3d.py --from-json benchmarks/results/performance_x3d.json
    python benchmarks/performance_x3d.py --sections gates   # measure one section again

Writes ``performance_x3d.json`` and ``performance_x3d.md`` into ``--out`` (default
``benchmarks/results/``).
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numba
import numpy as np
from numba import njit, prange

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sweep_throughput import measure as sweep_measure  # noqa: E402
from thread_scaling import cpu_info  # noqa: E402

import sparsegf2 as sg  # noqa: E402
from sparsegf2 import mipt, performance  # noqa: E402
from sparsegf2._kernel.tableau import apply_2q  # noqa: E402
from sparsegf2._kernel.tables import ANF2, IDENTITY_SYM2, LIN2  # noqa: E402
from sparsegf2.performance import PerformanceSettings  # noqa: E402
from sparsegf2.study import identity, sweep  # noqa: E402
from sparsegf2.study.sweep import SweepSpec  # noqa: E402

RATES = ("0.05", "0.2")
SECTIONS = (
    "breakdown",
    "placements",
    "padding",
    "gates",
    "scheduling",
    "small_batches",
    "end_to_end",
)
FULL = {
    "sizes": (64, 256, 1024),
    "breakdown": {64: 512, 256: 16, 1024: 2},  # trajectories on one thread
    "per_thread": {64: 1024, 256: 32, 1024: 2},  # trajectories per running thread
    "threads": (1, 8, 16, 32),
    "large": (1280, 1536, 1792, 2048),  # T = 256
    "padding": ((512, None), (1024, None), (2048, 256)),
    "chunks": (256, 2048),  # n, trajectories with stop_when_purified
    "end_to_end": {64: 16384, 256: 6144, 1024: 64},
}
QUICK = {
    "sizes": (16, 32),
    "breakdown": {16: 8, 32: 4},
    "per_thread": {16: 4, 32: 2},
    "threads": (1, 2),
    "large": (40,),
    "padding": ((512, 4),),
    "chunks": (16, 32),
    "end_to_end": {16: 64, 32: 32},
}


# ---------------------------------------------------------------------------
# load and workload
# ---------------------------------------------------------------------------


class _Load:
    """CPU load of other processes: all processors' busy time minus this process's."""

    settle, window = 0.2, 0.5

    def __init__(self) -> None:
        self.ok = sys.platform == "win32"
        if self.ok:
            from ctypes import wintypes

            self.k = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
            self.k.GetCurrentProcess.restype = wintypes.HANDLE
            self.k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.c_void_p] * 4
            self.ft = wintypes.FILETIME

    def _ticks(self) -> tuple[int, int, int]:
        def v(f) -> int:
            return (f.dwHighDateTime << 32) | f.dwLowDateTime

        i, k, u = self.ft(), self.ft(), self.ft()
        self.k.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
        c, e, pk, pu = self.ft(), self.ft(), self.ft(), self.ft()
        self.k.GetProcessTimes(
            self.k.GetCurrentProcess(), ctypes.byref(c), ctypes.byref(e), ctypes.byref(pk),
            ctypes.byref(pu),
        )  # fmt: skip
        return v(i), v(k) + v(u), v(pk) + v(pu)  # kernel time includes idle time

    def __call__(self) -> float | None:
        time.sleep(self.settle)
        if not self.ok:
            if hasattr(os, "getloadavg"):
                return round(100 * os.getloadavg()[0] / (os.cpu_count() or 1), 1)
            return None
        i0, t0, o0 = self._ticks()
        time.sleep(self.window)
        i1, t1, o1 = self._ticks()
        total = max(1, t1 - t0)
        return round(100 * max(0, total - (i1 - i0) - (o1 - o0)) / total, 1)


LOAD = _Load()
GRAPHS: dict = {}


def circuit(n, p, *, snapshots=True, layers=None, probes=None, stop=False, trajectories=4096):
    """The sweep's engine for one condition; graph g carries 1024 circuits."""
    per = 1024
    spec = SweepSpec.from_json(
        {
            "sizes": [n],
            "betas": ["0.1"],
            "rates": [p],
            "graphs": max(2, -(-trajectories // per)),
            "circuits_per_graph": per,
            "stride": per,
            "layers": {"per_n": 4} if layers is None else {str(n): layers},
            "checkpoints": {"count": 12},
            "probes": probes
            or [{"kind": "code_dimension", "checkpoints": "every_layer"}]
            + ([{"kind": "tableau_snapshot"}] if snapshots else []),
            "master_graph_seed": 1,
            "master_circuit_seed": 2,
            "stop_when_purified": stop,
        }
    )
    key = (n, spec.graphs[n])
    if key not in GRAPHS:
        GRAPHS[key] = [identity.build_graph(n, "0.1", g, 1).edges for g in range(spec.graphs[n])]
    return spec.monitored_circuit(n, "0.1", p, GRAPHS[key]), list(spec.run_seed(n, "0.1"))


def timed(mc, seed, count, **kw):
    load = LOAD()
    start = time.perf_counter()
    result = mc.run(count, seed=seed, **kw)
    return time.perf_counter() - start, result, load


def fingerprint(result, rows: int) -> list:
    """Every value and snapshot word of the first ``rows`` rows, as a hashable summary."""
    out = [result.values[k][:rows].tobytes() for k in sorted(result.values)]
    for snap, counts in result.snapshots.values():
        out += [snap[:rows].tobytes(), counts[:rows].tobytes()]
    out += [result.purification_layer[:rows].tobytes(), result.final_layer[:rows].tobytes()]
    return out


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def breakdown(plan) -> tuple[list, float]:
    rows = []
    for n, count in plan["breakdown"].items():
        for p in RATES:
            row = {"n": n, "p": p, "trajectories": count}
            variants = {
                "full": circuit(n, p, trajectories=count),
                "no_snapshots": circuit(n, p, snapshots=False, trajectories=count),
                "gates_only": circuit(n, "0", snapshots=False, trajectories=count),
                "k_at_end": circuit(
                    n,
                    p,
                    probes=[{"kind": "code_dimension", "checkpoints": "final"}],
                    trajectories=count,
                ),  # fmt: skip
            }
            for name, (mc, seed) in variants.items():
                seconds, _, load = timed(mc, seed, count, num_threads=1)
                row[name] = round(seconds / count * 1e3, 4)
                row[f"load_{name}"] = load
            rows.append(row)
            print("breakdown", row, flush=True)
    mc, seed = circuit(64 if 64 in plan["breakdown"] else 16, "0.2", layers=1)
    calls = [timed(mc, seed, 1, num_threads=1)[0] for _ in range(50)]
    return rows, round(float(np.median(calls)) * 1e3, 3)


def placement_configs(topo, counts):
    configs = [("all", t) for t in counts]
    if topo.known:
        configs += [("physical", t) for t in counts if 1 < t <= len(topo.cores)]
    if topo.x3d:
        half = len(topo.vcache.processors)
        configs += [(name, t) for name in ("vcache", "other_ccd") for t in counts if 1 < t <= half]
    return configs


def run_config(mc, seed, count, placement, threads):
    settings = PerformanceSettings(threads=threads, placement=placement)
    load = LOAD()
    start = time.perf_counter()
    result = mc.run(count, seed=seed, performance=settings)
    return time.perf_counter() - start, result, load


def placements(plan, topo) -> tuple[list, bool]:
    rows, identical = [], True
    full = placement_configs(topo, plan["threads"])
    top = max(plan["threads"])
    short = [("all", top)] + ([("physical", len(topo.cores))] if topo.known else [])
    sizes = [(n, p, None, full if p == "0.2" else short) for n in plan["sizes"] for p in RATES]
    ccds = [c for c in full if c[0] in ("vcache", "other_ccd") and c[1] == len(topo.cores)]
    sizes += [(n, "0.2", 256, short + ccds) for n in plan["large"]]
    for n, p, layers, configs in sizes:
        mc, seed = circuit(n, p, layers=layers, trajectories=65536)
        per = plan["per_thread"].get(n, 2)
        reference = None
        for placement, threads in configs:
            count = per * max(threads, 2)
            seconds, result, load = run_config(mc, seed, count, placement, threads)
            common = per * 2
            print_ = fingerprint(result, common)
            if reference is None:
                reference = print_
            identical &= print_ == reference
            row = {
                "n": n, "p": p, "layers": mc.layers, "placement": placement, "threads": threads,
                "trajectories": count, "load": load, "seconds": round(seconds, 3),
                "trajectories_per_second": round(count / seconds, 3),
            }  # fmt: skip
            rows.append(row)
            print("placement", row, flush=True)
    return rows, identical


def padding(plan) -> tuple[list, bool]:
    rows, identical = [], True
    original = mipt._row_padding
    top = max(plan["threads"])
    for n, layers in plan["padding"]:
        mc, seed = circuit(n, "0.2", layers=layers, trajectories=65536)
        count = 2 * top
        prints = []
        for pad in (False, True):
            mipt._row_padding = original if pad else (lambda words: 0)
            try:
                seconds, result, load = timed(mc, seed, count)
            finally:
                mipt._row_padding = original
            prints.append(fingerprint(result, count))
            words = (2 * n + 63) // 64
            row = {
                "n": n, "layers": mc.layers, "words": words,
                "pad": original(words) if pad else 0, "trajectories": count, "load": load,
                "seconds": round(seconds, 3), "trajectories_per_second": round(count / seconds, 3),
            }  # fmt: skip
            rows.append(row)
            print("padding", row, flush=True)
        identical &= prints[0] == prints[1]
    return rows, identical


# The tables are arguments, as in the engine: a global array would be compiled in as a constant,
# which the compiler knows no store can change.
@njit(cache=False)
def _gates_current(tab, ga, gb, gg, ws, reps, lin2, anf2):
    for _ in range(reps):
        for k in range(ga.shape[0]):
            apply_2q(tab, ga[k], gb[k], gg[k], lin2, anf2, IDENTITY_SYM2, ws, False)


@njit(cache=False)
def _gates_hoisted(tab, ga, gb, gg, ws, reps, lin2):
    """apply_2q's linear part as it is: the masks read before the word loop."""
    for _ in range(reps):
        for k in range(ga.shape[0]):
            m = lin2[gg[k] >> 4]
            m00, m01, m02, m03 = m[0, 0], m[0, 1], m[0, 2], m[0, 3]
            m10, m11, m12, m13 = m[1, 0], m[1, 1], m[1, 2], m[1, 3]
            m20, m21, m22, m23 = m[2, 0], m[2, 1], m[2, 2], m[2, 3]
            m30, m31, m32, m33 = m[3, 0], m[3, 1], m[3, 2], m[3, 3]
            ra = 2 * ga[k]
            rb = 2 * gb[k]
            for w in range(ws):
                v0 = tab[ra, w]
                v1 = tab[rb, w]
                v2 = tab[ra + 1, w]
                v3 = tab[rb + 1, w]
                tab[ra, w] = (v0 & m00) ^ (v1 & m10) ^ (v2 & m20) ^ (v3 & m30)
                tab[rb, w] = (v0 & m01) ^ (v1 & m11) ^ (v2 & m21) ^ (v3 & m31)
                tab[ra + 1, w] = (v0 & m02) ^ (v1 & m12) ^ (v2 & m22) ^ (v3 & m32)
                tab[rb + 1, w] = (v0 & m03) ^ (v1 & m13) ^ (v2 & m23) ^ (v3 & m33)


@njit(cache=False)
def _gates_in_loop(tab, ga, gb, gg, ws, reps, lin2):
    """apply_2q's linear part as it was: the masks read inside the word loop."""
    for _ in range(reps):
        for k in range(ga.shape[0]):
            m = lin2[gg[k] >> 4]
            ra = 2 * ga[k]
            rb = 2 * gb[k]
            for w in range(ws):
                v0 = tab[ra, w]
                v1 = tab[rb, w]
                v2 = tab[ra + 1, w]
                v3 = tab[rb + 1, w]
                tab[ra, w] = (v0 & m[0, 0]) ^ (v1 & m[1, 0]) ^ (v2 & m[2, 0]) ^ (v3 & m[3, 0])
                tab[rb, w] = (v0 & m[0, 1]) ^ (v1 & m[1, 1]) ^ (v2 & m[2, 1]) ^ (v3 & m[3, 1])
                tab[ra + 1, w] = (v0 & m[0, 2]) ^ (v1 & m[1, 2]) ^ (v2 & m[2, 2]) ^ (v3 & m[3, 2])
                tab[rb + 1, w] = (v0 & m[0, 3]) ^ (v1 & m[1, 3]) ^ (v2 & m[2, 3]) ^ (v3 & m[3, 3])


@njit(cache=False, inline="always")
def _bit(bits, k):
    return np.uint64(0) - ((np.uint64(bits) >> np.uint64(k)) & np.uint64(1))


@njit(cache=False)
def _gates_compact(tab, ga, gb, gg, lin16, ws, reps):
    """The same with a 16-bit table (1.4 KB): bit 4 i + j of lin16[s] is M_s[i, j]."""
    for _ in range(reps):
        for k in range(ga.shape[0]):
            b = lin16[gg[k] >> 4]
            m00, m01, m02, m03 = _bit(b, 0), _bit(b, 1), _bit(b, 2), _bit(b, 3)
            m10, m11, m12, m13 = _bit(b, 4), _bit(b, 5), _bit(b, 6), _bit(b, 7)
            m20, m21, m22, m23 = _bit(b, 8), _bit(b, 9), _bit(b, 10), _bit(b, 11)
            m30, m31, m32, m33 = _bit(b, 12), _bit(b, 13), _bit(b, 14), _bit(b, 15)
            ra = 2 * ga[k]
            rb = 2 * gb[k]
            for w in range(ws):
                v0 = tab[ra, w]
                v1 = tab[rb, w]
                v2 = tab[ra + 1, w]
                v3 = tab[rb + 1, w]
                tab[ra, w] = (v0 & m00) ^ (v1 & m10) ^ (v2 & m20) ^ (v3 & m30)
                tab[rb, w] = (v0 & m01) ^ (v1 & m11) ^ (v2 & m21) ^ (v3 & m31)
                tab[ra + 1, w] = (v0 & m02) ^ (v1 & m12) ^ (v2 & m22) ^ (v3 & m32)
                tab[rb + 1, w] = (v0 & m03) ^ (v1 & m13) ^ (v2 & m23) ^ (v3 & m33)


def gates(plan) -> tuple[list, bool]:
    lin16 = np.zeros(720, dtype=np.uint16)
    for s in range(720):
        lin16[s] = sum(1 << (4 * i + j) for i in range(4) for j in range(4) if LIN2[s, i, j])
    rng = np.random.default_rng(1)
    rows, identical = [], True
    for n in plan["sizes"]:
        nq = 2 * n
        ws = (nq + 63) // 64
        count = 200_000 if n >= 64 else 2_000
        ga = rng.integers(0, n, count)
        gb = (ga + rng.integers(1, n, count)) % n
        gg = rng.integers(0, 11520, count)
        base = rng.integers(0, 2**63, size=(2 * nq + 1, ws), dtype=np.int64).astype(np.uint64)
        reps = max(1, 2_000_000 // (count * ws)) if n >= 64 else 1
        args = (ga, gb, gg, ws)
        kernels = {
            "current": lambda t, r, a=args: _gates_current(t, *a, r, LIN2, ANF2),
            "hoisted": lambda t, r, a=args: _gates_hoisted(t, *a, r, LIN2),
            "in_loop": lambda t, r, a=args: _gates_in_loop(t, *a, r, LIN2),
            "compact": lambda t, r, a=args: _gates_compact(t, *a[:3], lin16, a[3], r),
        }
        out = {"n": n, "words": ws}
        finals = []
        for name, kernel in kernels.items():
            tab = base.copy()
            kernel(tab, 1)
            finals.append(tab)
            best = np.inf
            for _ in range(5):
                tab = base.copy()
                start = time.perf_counter()
                kernel(tab, reps)
                best = min(best, time.perf_counter() - start)
            out[f"{name}_ns_per_gate"] = round(best / (count * reps) * 1e9, 2)
        identical &= all(np.array_equal(finals[0], f) for f in finals)
        rows.append(out)
        print("gates", out, flush=True)
    return rows, identical


def scheduling(plan) -> list:
    n, count = plan["chunks"]
    mc, seed = circuit(n, "0.2", snapshots=False, stop=True, trajectories=65536)
    rows = []
    for chunk in (0, 1, 16):
        numba.set_parallel_chunksize(chunk)
        try:
            seconds, result, load = timed(mc, seed, count)
        finally:
            numba.set_parallel_chunksize(0)
        final = result.final_layer
        row = {
            "n": n, "p": "0.2", "chunksize": chunk, "trajectories": count, "load": load,
            "seconds": round(seconds, 3), "trajectories_per_second": round(count / seconds, 1),
            "final_layer_min": int(final.min()), "final_layer_mean": round(float(final.mean()), 1),
            "final_layer_max": int(final.max()),
        }  # fmt: skip
        rows.append(row)
        print("scheduling", row, flush=True)
    return rows


def small_batches(plan) -> list:
    n = 64 if 64 in plan["sizes"] else plan["sizes"][0]
    mc, seed = circuit(n, "0.2", trajectories=64)
    top = max(plan["threads"])
    rows = []
    for count in (1, 2, 4, 8, 16, 32):
        row = {"n": n, "trajectories": count}
        for threads in (1, top):
            times = []
            for _ in range(10):
                start = time.perf_counter()
                mc.run(count, seed=seed, num_threads=threads)
                times.append(time.perf_counter() - start)
            row[f"ms_{threads}_threads"] = round(float(np.median(times)) * 1e3, 3)
        rows.append(row)
        print("small batch", row, flush=True)
    return rows


# Windows only: pin Numba's worker threads to one set of logical processors and the sweep's
# writers to another (process affinity cannot: it binds every thread of the process alike).
if sys.platform == "win32":
    _K32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    _thread_handle = _K32.GetCurrentThread
    _thread_handle.restype = ctypes.c_void_p
    _thread_handle.argtypes = []
    _set_thread_mask = _K32.SetThreadAffinityMask
    _set_thread_mask.restype = ctypes.c_size_t
    _set_thread_mask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

    @njit(parallel=True, cache=False)
    def _pin_workers(masks, out):
        for i in prange(masks.shape[0]):  # one iteration per thread: Numba's static schedule
            out[i] = _set_thread_mask(_thread_handle(), masks[i])


def _pinned_sweep(n, p, count, scratch, engine, writers, writer_count):
    """sweep.run with the engine's threads on ``engine`` and its writers on ``writers``."""
    engine_mask = sum(1 << q for q in engine)
    writer_mask = sum(1 << q for q in writers)
    every = sum(1 << q for q in range(64) if q < (os.cpu_count() or 1))
    threads = len(engine)
    out = np.zeros(threads, dtype=np.uint64)
    original = sweep._save_part

    def save(*args):
        _set_thread_mask(_thread_handle(), writer_mask)
        return original(*args)

    numba.set_num_threads(threads)
    _pin_workers(np.full(threads, engine_mask, dtype=np.uint64), out)
    sweep._save_part = save
    try:
        row = sweep_measure(n, p, True, count, scratch, num_threads=threads, writers=writer_count)
    finally:
        sweep._save_part = original
        _pin_workers(np.full(threads, every, dtype=np.uint64), out)
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    return row


def end_to_end(plan, topo, scratch) -> list:
    rows = []
    sizes = list(plan["end_to_end"].items())
    for n, count in sizes[:2]:
        p = "0.05"
        for writers in (2, 4, 8):
            load = LOAD()
            row = sweep_measure(n, p, True, count, scratch, writers=writers)
            row.update(setting=f"all processors, {writers} writers", load=load)
            rows.append(row)
            print("end to end", row, flush=True)
        if sys.platform == "win32" and topo.x3d:
            big = topo.vcache.processors
            other = topo.placement("other_ccd")
            for writer_count in (4, 8):
                load = LOAD()
                row = _pinned_sweep(n, p, count, scratch, big, other, writer_count)
                row.update(
                    setting=f"engine on the V-Cache CCD ({len(big)} threads), "
                    f"{writer_count} writers on the other CCD",
                    load=load,
                )
                rows.append(row)
                print("end to end", row, flush=True)
    n, count = sizes[-1]
    probe = SweepSpec.from_json(
        {"sizes": [n], "betas": ["0.1"], "rates": ["0.2"], "graphs": 1, "circuits_per_graph": 1,
         "stride": 1, "layers": {"per_n": 4}, "probes": [
             {"kind": "code_dimension", "checkpoints": "every_layer"},
             {"kind": "tableau_snapshot"}],
         "master_graph_seed": 1, "master_circuit_seed": 2}
    )  # fmt: skip
    mc = probe.monitored_circuit(n, "0.1", "0.2", [identity.build_graph(n, "0.1", 0, 1).edges])
    old_cap = max(1, sweep.PART_BYTES // sg.api._result_bytes(mc, 1))
    for label, options in (
        (f"parts of {old_cap} trajectories (the PART_BYTES cap alone)", {"chunk_size": old_cap}),
        ("default chunks (whole rounds of the threads)", {}),
    ):
        load = LOAD()
        row = sweep_measure(n, "0.2", True, count, scratch, **options)
        row.update(setting=label, load=load)
        rows.append(row)
        print("end to end", row, flush=True)
    return rows


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _topology_record(topo) -> dict:
    record = dataclasses.asdict(topo)
    record["l3"] = [{"bytes": d.size, "processors": list(d.processors)} for d in topo.l3]
    record["cores"] = [list(c) for c in topo.cores]
    record["processors"] = list(topo.processors)
    record["x3d"] = topo.x3d
    return record


def render_markdown(results: dict) -> str:
    env, topo = results["environment"], results["topology"]
    l3 = ", ".join(
        f"{d['bytes'] // 2**20} MiB shared by logical processors "
        f"{d['processors'][0]}-{d['processors'][-1]}"
        for d in topo["l3"]
    )
    lines = [
        "# Performance on this CPU",
        "",
        f"CPU: {env['cpu_model']} ({env['physical_cores']} physical cores, "
        f"{env['logical_processors']} logical processors), {env['platform']}. Python "
        f"{env['python']}, NumPy {env['numpy']}, Numba {env['numba']} (`NUMBA_NUM_THREADS` = "
        f"{env['numba_num_threads']}, threading layer `{env['threading_layer']}`), sparsegf2 "
        f"{env['sparsegf2']}.",
        "",
        f"Topology (`sparsegf2.performance.cpu_topology`, source `{topo['source']}`): "
        f"{len(topo['cores'])} cores of {len(topo['cores'][0]) if topo['cores'] else '?'} "
        f"logical processors; L1d {topo['l1d_bytes']} bytes and L2 {topo['l2_bytes']} bytes "
        f"per core; L3: {l3 or 'unknown'}.",
        "",
        "Workload: the sweep's (Watts–Strogatz k = 4, beta = 0.1, purification picture, "
        "n // 2 random-edge gates per layer, T = 4n unless stated, `code_dimension` every "
        "layer, `tableau_snapshot` on 12 geometric checkpoints). `load` is the CPU load of "
        "other processes, in percent of all logical processors, just before the measurement.",
        "",
        "## Where the time goes (one thread, ms per trajectory)",
        "",
        "| n | p | full | no snapshots | gates only (p = 0) | k at the end only | load % |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results["breakdown"]:
        loads = max(r[k] or 0 for k in r if k.startswith("load_"))
        lines.append(
            f"| {r['n']} | {r['p']} | {r['full']:.3f} | {r['no_snapshots']:.3f} | "
            f"{r['gates_only']:.3f} | {r['k_at_end']:.3f} | {loads} |"
        )
    lines += [
        "",
        f"One `MonitoredCircuit.run` call of one trajectory and one layer: "
        f"{results['call_ms']} ms (the fixed cost per call).",
        "",
        "## Threads and placement (trajectories per second)",
        "",
        "| n | p | T | placement | threads | trajectories | trajectories/s | load % |",
        "|---:|---:|---:|:---|---:|---:|---:|---:|",
    ]
    for r in results["placements"]:
        lines.append(
            f"| {r['n']} | {r['p']} | {r['layers']} | {r['placement']} | {r['threads']} | "
            f"{r['trajectories']} | {r['trajectories_per_second']:.4g} | {r['load']} |"
        )
    lines += [
        "",
        f"Common trajectories identical in every configuration: {results['placements_identical']}.",
        "",
        "## Tableau padding (all threads, p = 0.2, snapshots included)",
        "",
        "| n | T | words per column | padding words | trajectories/s | load % |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results["padding"]:
        lines.append(
            f"| {r['n']} | {r['layers']} | {r['words']} | {r['pad']} | "
            f"{r['trajectories_per_second']:.4g} | {r['load']} |"
        )
    lines += [
        "",
        f"Identical with and without padding: {results['padding_identical']}.",
        "",
        "## Gate kernel (ns per gate, random gates on a random tableau of 2n qubits)",
        "",
        "`apply_2q` is the package's kernel (phase-free); the other columns are its linear "
        "part alone, with the 16 masks read before the word loop (as `apply_2q` now does), "
        "inside it (as it did before), and from a 16-bit table expanded in registers.",
        "",
        "| n | words | `apply_2q` | masks before the loop | masks in the loop | 16-bit table |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results["gates"]:
        lines.append(
            f"| {r['n']} | {r['words']} | {r['current_ns_per_gate']} | "
            f"{r['hoisted_ns_per_gate']} | {r['in_loop_ns_per_gate']} | "
            f"{r['compact_ns_per_gate']} |"
        )
    lines += [
        "",
        f"Identical tableaux from the three kernels: {results['gates_identical']}.",
        "",
        "## Scheduling with `stop_when_purified` (all threads)",
        "",
        "| n | Numba chunk size | trajectories/s | final layer min / mean / max | load % |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in results["scheduling"]:
        lines.append(
            f"| {r['n']} | {r['chunksize']} | {r['trajectories_per_second']:.4g} | "
            f"{r['final_layer_min']} / {r['final_layer_mean']} / {r['final_layer_max']} | "
            f"{r['load']} |"
        )
    top = max(results["plan_threads"])
    lines += [
        "",
        "## Small batches (ms per `run` call, median of 10)",
        "",
        f"| n | trajectories | 1 thread | {top} threads |",
        "|---:|---:|---:|---:|",
    ]
    for r in results["small_batches"]:
        lines.append(
            f"| {r['n']} | {r['trajectories']} | {r['ms_1_threads']} | {r[f'ms_{top}_threads']} |"
        )
    lines += [
        "",
        "## End to end (`sweep.run` of one condition with snapshots, `sweep_throughput.measure`)",
        "",
        "| n | p | setting | trajectories | parts | seconds | trajectories/s | engine s | load % |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results["end_to_end"]:
        lines.append(
            f"| {r['n']} | {r['p']} | {r['setting']} | {r['trajectories']} | {r['parts']} | "
            f"{r['seconds']:.2f} | {r['trajectories_per_second']:.4g} | "
            f"{r['engine_seconds']:.2f} | {r['load']} |"
        )
    again = results.get("remeasured")
    note = f" Sections measured again in a later call: {', '.join(again)}." if again else ""
    lines += ["", f"Total wall time {results['total_seconds']:.0f} s.{note}", ""]
    return "\n".join(lines)


def write(results: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("performance_x3d.json", json.dumps(results, indent=1) + "\n"),
        ("performance_x3d.md", render_markdown(results)),
    ):
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(HERE / "results"), help="output directory")
    ap.add_argument("--quick", action="store_true", help="tiny sizes, for a smoke test")
    ap.add_argument("--from-json", default=None, help="render a saved results file again")
    ap.add_argument(
        "--sections",
        nargs="+",
        choices=SECTIONS,
        default=None,
        help="measure these sections only and replace them in --out's performance_x3d.json",
    )
    args = ap.parse_args()
    out = Path(args.out)
    if args.from_json:
        write(json.loads(Path(args.from_json).read_text("utf-8")), out)
        return
    plan = dict(QUICK if args.quick else FULL)
    if args.quick:
        LOAD.settle, LOAD.window = 0.0, 0.05
    top = min(numba.config.NUMBA_NUM_THREADS, max(plan["threads"]))
    plan["threads"] = tuple(sorted({t for t in plan["threads"] if t <= top} | {top}))
    topo = performance.cpu_topology()
    start = time.perf_counter()
    mc, seed = circuit(8, "0.1", trajectories=4)
    mc.run(4, seed=seed)  # compiles
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **cpu_info(),
        "logical_processors": os.cpu_count(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "numba_num_threads": numba.config.NUMBA_NUM_THREADS,
        "threading_layer": numba.threading_layer(),
        "sparsegf2": sg.__version__,
    }
    print(env, flush=True)
    results: dict = {"environment": env, "topology": _topology_record(topo)}
    results["quick"] = args.quick
    results["plan_threads"] = list(plan["threads"])
    chosen = args.sections or SECTIONS
    if args.sections:  # the other sections from the saved results
        saved = json.loads((out / "performance_x3d.json").read_text("utf-8"))
        results = {**saved, "remeasured": sorted({*saved.get("remeasured", []), *chosen})}
    if "breakdown" in chosen:
        results["breakdown"], results["call_ms"] = breakdown(plan)
    if "placements" in chosen:
        results["placements"], results["placements_identical"] = placements(plan, topo)
    if "padding" in chosen:
        results["padding"], results["padding_identical"] = padding(plan)
    if "gates" in chosen:
        results["gates"], results["gates_identical"] = gates(plan)
    if "scheduling" in chosen:
        results["scheduling"] = scheduling(plan)
    if "small_batches" in chosen:
        results["small_batches"] = small_batches(plan)
    if "end_to_end" in chosen:
        with tempfile.TemporaryDirectory(prefix="performance_x3d_") as tmp:
            results["end_to_end"] = end_to_end(plan, topo, Path(tmp))
    seconds = round(time.perf_counter() - start, 1)
    results["total_seconds"] = seconds + (results.get("total_seconds", 0) if args.sections else 0)
    if not (
        results["placements_identical"]
        and results["padding_identical"]
        and results["gates_identical"]
    ):
        write(results, out)
        raise SystemExit("results differ between settings: see the report")
    write(results, out)


if __name__ == "__main__":
    main()
