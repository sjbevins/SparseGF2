"""
``distances`` analysis — d_cont (contiguous) and d_min (true code distance).

Ports algorithms from ``studies/code_quality_sweep._compute_d_cont`` and
``benchmarks/benchmark_rate_distance_dmin._compute_d_min``.

d_cont : smallest contiguous block A_ell on the ring such that I(A_ell; R) > 0.
         Binary-searched from four canonical starting positions (0, n/4, n/2, 3n/4).
d_min  : smallest |A| (any subset) such that I(A; R) > 0. Exhaustive search up
         to |A| = ``d_min_max_exhaustive`` (default 3), then greedy shrink from
         the arc that realizes d_cont. d_min <= d_cont always.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sparsegf2 import __version__ as _SGF2_VERSION
from sparsegf2.analysis_pipeline.analyses._common import CellContext, CellRunResult
from sparsegf2.analysis_pipeline.rehydrate import rehydrate_sim
from sparsegf2.analysis_pipeline.registry import (
    make_entry, read_cell_registry, write_cell_registry,
)


NAME = "distances"
OUTPUT_FILENAME = "distances.parquet"
OUTPUT_KIND = "parquet"
CELL_SCOPE = True
EXPENSIVE = False
DEFAULT_PARAMS = {
    "d_cont_starts": [0.0, 0.25, 0.5, 0.75],   # fractional starting positions on the ring
    "d_min_max_exhaustive": 3,                 # exhaustive enumeration up to |A| = this
}


# ══════════════════════════════════════════════════════════════
# Mutual-information primitive
# ══════════════════════════════════════════════════════════════

def _has_info(sim, n: int, k: int, A) -> bool:
    """Return True iff I(A; R) = S(A) + k - S(sys \\ A) > 0."""
    if not A:
        return False
    A_set = set(int(q) for q in A)
    B = [q for q in range(n) if q not in A_set]
    S_A = int(sim.compute_subsystem_entropy(list(A)))
    S_B = int(sim.compute_subsystem_entropy(B)) if B else 0
    return (S_A + int(k) - S_B) > 0


# ══════════════════════════════════════════════════════════════
# d_cont (contiguous distance) + the arc that realizes it
# ══════════════════════════════════════════════════════════════

def _compute_d_cont_with_arc(sim, n: int, k: int, starts_frac: List[float]):
    if k == 0:
        return 0, []
    best_d = n
    best_arc: List[int] = list(range(n))
    starts = sorted(set(int(round(f * n)) % n for f in starts_frac))
    for start in starts:
        lo, hi = 1, n
        while lo < hi:
            mid = (lo + hi) // 2
            qubits = [(start + j) % n for j in range(mid)]
            if _has_info(sim, n, k, qubits):
                hi = mid
            else:
                lo = mid + 1
        if lo < best_d:
            best_d = lo
            best_arc = [(start + j) % n for j in range(lo)]
    return int(best_d), best_arc


# ══════════════════════════════════════════════════════════════
# d_min (true code distance) — exhaustive low-weight + greedy shrink
# ══════════════════════════════════════════════════════════════

def _compute_d_min(sim, n: int, k: int, d_cont: int, arc_cont, max_exhaustive: int):
    """Upper bound on the minimum code distance of the emergent code.

    Exhaustive enumeration up to ``|A| = max_exhaustive`` (default 3, matching
    the benchmark), then a one-pass greedy shrink starting from the arc that
    realizes d_cont.
    """
    if k == 0:
        return 0
    max_exhaustive = max(1, int(max_exhaustive))

    # |A| = 1
    for q in range(n):
        if _has_info(sim, n, k, [q]):
            return 1
    if max_exhaustive < 2 or d_cont <= 1:
        return int(d_cont)

    # |A| = 2
    for i in range(n):
        for j in range(i + 1, n):
            if _has_info(sim, n, k, [i, j]):
                return 2
    if max_exhaustive < 3 or d_cont <= 2:
        return int(d_cont)

    # |A| = 3
    for i in range(n):
        for j in range(i + 1, n):
            for m in range(j + 1, n):
                if _has_info(sim, n, k, [i, j, m]):
                    return 3
    if d_cont <= 3:
        return int(d_cont)

    # Greedy shrink from arc_cont
    A = list(arc_cont)
    changed = True
    while changed and len(A) > 3:
        changed = False
        for q in list(A):
            trial = [x for x in A if x != q]
            if _has_info(sim, n, k, trial):
                A = trial
                changed = True
                if len(A) <= 3:
                    break
    return int(len(A))


# ══════════════════════════════════════════════════════════════
# Cell entry point
# ══════════════════════════════════════════════════════════════

_SCHEMA = pa.schema([
    ("sample_seed", pa.int64()),
    ("d_cont", pa.int32()),
    ("d_min", pa.int32()),
    ("d_cont_method", pa.string()),
    ("d_min_method", pa.string()),
    ("runtime_s", pa.float64()),
])


def run_cell(ctx: CellContext, params: dict, force: bool = False) -> CellRunResult:
    output = ctx.analysis_dir / OUTPUT_FILENAME
    if output.exists() and not force:
        return CellRunResult(
            name=NAME, status="existing", output_path=output,
            n_samples=ctx.n_samples, params=dict(params),
            message=f"{output.name} already exists (use --force {NAME} to recompute)",
        )

    merged = {**DEFAULT_PARAMS, **params}
    starts = list(merged["d_cont_starts"])
    max_exh = int(merged["d_min_max_exhaustive"])

    ctx.analysis_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    data = {
        "sample_seed": [], "d_cont": [], "d_min": [],
        "d_cont_method": [], "d_min_method": [], "runtime_s": [],
    }
    d_cont_method = f"binary_search_starts={starts}"
    d_min_method = f"exhaustive_leq_{max_exh}+greedy_shrink"

    n = ctx.n
    for i in range(ctx.n_samples):
        t_s = time.perf_counter()
        sim = rehydrate_sim(n, ctx.x_stack[i], ctx.z_stack[i])
        k = int(sim.compute_k())
        d_cont, arc = _compute_d_cont_with_arc(sim, n, k, starts)
        d_min = _compute_d_min(sim, n, k, d_cont, arc, max_exh)
        rt = time.perf_counter() - t_s

        data["sample_seed"].append(int(ctx.seeds[i]))
        data["d_cont"].append(int(d_cont))
        data["d_min"].append(int(d_min))
        data["d_cont_method"].append(d_cont_method)
        data["d_min_method"].append(d_min_method)
        data["runtime_s"].append(float(rt))

    table = pa.Table.from_pydict(data, schema=_SCHEMA)
    pq.write_table(table, output)
    runtime = time.perf_counter() - t0

    entries = read_cell_registry(ctx.analysis_dir)
    entries[NAME] = make_entry(
        package_version=_SGF2_VERSION, params=merged,
        runtime_s=runtime, n_samples=ctx.n_samples,
    ).__dict__
    write_cell_registry(ctx.analysis_dir, entries)

    return CellRunResult(
        name=NAME, status="computed", output_path=output,
        runtime_s=runtime, n_samples=ctx.n_samples, params=merged,
    )
